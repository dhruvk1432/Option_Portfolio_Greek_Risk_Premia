"""Broker-neutral forward shadow-trading ledgers.

The shadow layer is deliberately not a production executor. It converts a
target option book plus market-hours NBBO/CBBO snapshots into auditable paper
orders, displayed-size fill assumptions, margin/rejection ledgers, and a
summary. The output is useful for forward validation, but it must not be
treated as broker-executed live evidence.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .execution import FeeSchedule, OrderPolicy, build_execution_ledger, build_fill_ledger, estimate_nbbo_fill, target_weights_to_orders
from .margin import conservative_order_margin
from .market_data import build_market_data_ledger, validate_timestamp_monotonicity
from .risk import RiskGateConfig, evaluate_pre_trade_gate
from .schemas import AccountState, Fill, MarginEstimate, OptionContract, OptionOrder, Position, QuoteSnapshot, normalize_right

SHADOW_FILL_MODEL = "shadow_nbbo_displayed_size_cross"
SHADOW_TARGET_COLUMNS = [
    "decision_time",
    "symbol",
    "underlying",
    "expiry",
    "right",
    "strike",
    "target_weight",
]
SHADOW_QUOTE_COLUMNS = [
    "symbol",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "ts_event",
    "ts_recv",
    "local_receive_ts",
]


@dataclass(frozen=True)
class ShadowRunConfig:
    nav: float
    decision_time: pd.Timestamp
    out_dir: Path
    max_quote_age: timedelta = timedelta(minutes=15)
    max_spread_bps: float = 500.0
    max_participation_of_displayed_size: float = 0.25
    min_contracts: int = 1
    symbol_map_version: str = "shadow-csv-v1"
    allow_non_market_hours: bool = False


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _market_hours_ok(ts: Any) -> bool:
    try:
        local = pd.Timestamp(ts)
        if local.tzinfo is None:
            local = local.tz_localize("UTC")
        local = local.tz_convert(ZoneInfo("America/New_York"))
    except Exception:
        return False
    if local.weekday() >= 5:
        return False
    minute = local.hour * 60 + local.minute
    return 9 * 60 + 30 <= minute <= 16 * 60


def load_shadow_targets(path: str | Path, decision_time: Any | None = None) -> pd.DataFrame:
    targets = _read_csv(path).copy()
    _require_columns(targets, ["symbol", "underlying", "expiry", "right", "strike", "target_weight"], "targets")
    if "decision_time" not in targets:
        if decision_time is None:
            raise ValueError("targets must include decision_time or --decision-time must be supplied")
        targets["decision_time"] = pd.Timestamp(decision_time)
    targets["decision_time"] = pd.to_datetime(targets["decision_time"], errors="coerce", utc=True)
    targets["expiry"] = pd.to_datetime(targets["expiry"], errors="coerce")
    targets["right"] = targets["right"].map(normalize_right)
    targets["strike"] = pd.to_numeric(targets["strike"], errors="coerce")
    targets["target_weight"] = pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0)
    targets["multiplier"] = pd.to_numeric(targets.get("multiplier", 100), errors="coerce").fillna(100).astype(int)
    targets["asset_class"] = targets.get("asset_class", "equity_option")
    return targets.dropna(subset=["symbol", "underlying", "expiry", "strike", "decision_time"])


def load_shadow_quotes(path: str | Path) -> dict[str, QuoteSnapshot]:
    quotes_df = _read_csv(path).copy()
    _require_columns(quotes_df, SHADOW_QUOTE_COLUMNS, "quotes")
    quotes: dict[str, QuoteSnapshot] = {}
    for _, row in quotes_df.iterrows():
        q = QuoteSnapshot(
            symbol=str(row["symbol"]),
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            bid_size=float(row["bid_size"]),
            ask_size=float(row["ask_size"]),
            ts_event=row["ts_event"],
            ts_recv=row["ts_recv"],
            local_receive_ts=row["local_receive_ts"],
            vendor=str(row.get("vendor", "shadow_csv")),
            schema=str(row.get("schema", "nbbo_csv")),
            exchange_ts=row.get("exchange_ts") if pd.notna(row.get("exchange_ts", pd.NA)) else None,
            sequence=int(row["sequence"]) if "sequence" in row and pd.notna(row["sequence"]) else None,
        )
        quotes[q.symbol] = q
    return quotes


def load_positions(path: str | Path | None) -> dict[str, Position]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    _require_columns(frame, ["symbol", "quantity"], "positions")
    out: dict[str, Position] = {}
    for _, row in frame.iterrows():
        out[str(row["symbol"])] = Position(
            symbol=str(row["symbol"]),
            quantity=int(row["quantity"]),
            avg_price=float(row.get("avg_price", 0.0) or 0.0),
            asset_class=str(row.get("asset_class", "option")),
        )
    return out


def _contracts_from_targets(targets: pd.DataFrame) -> dict[str, OptionContract]:
    contracts: dict[str, OptionContract] = {}
    for _, row in targets.drop_duplicates("symbol").iterrows():
        contracts[str(row["symbol"])] = OptionContract(
            symbol=str(row["symbol"]),
            underlying=str(row["underlying"]),
            expiry=row["expiry"],
            right=row["right"],
            strike=float(row["strike"]),
            multiplier=int(row.get("multiplier", 100)),
            asset_class=str(row.get("asset_class", "equity_option")),
            broker_contract_id=str(row.get("broker_contract_id", "")),
            adjusted_deliverable=str(row.get("adjusted_deliverable", "standard")),
        )
    return contracts


def _margin_preview_map(path: str | Path | None) -> dict[str, tuple[float | None, str]]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    _require_columns(frame, ["symbol"], "margin previews")
    out: dict[str, tuple[float | None, str]] = {}
    for _, row in frame.iterrows():
        value = pd.to_numeric(row.get("broker_margin_preview", np.nan), errors="coerce")
        preview = float(value) if np.isfinite(value) else None
        status = str(row.get("margin_preview_status", "pass" if preview is not None else "missing"))
        out[str(row["symbol"])] = (preview, status)
    return out


def _manual_rejection_map(path: str | Path | None) -> dict[str, str]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    _require_columns(frame, ["symbol", "reason"], "manual rejections")
    return {str(row["symbol"]): str(row["reason"]) for _, row in frame.iterrows()}


def _target_weight_series(targets: pd.DataFrame) -> pd.Series:
    weights = targets.groupby("symbol", observed=True)["target_weight"].sum()
    weights.name = "target_weight"
    return weights


def _target_lookup(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.drop_duplicates("symbol").set_index("symbol")


def _shadow_fill(fill: Fill) -> Fill:
    return Fill(
        order_id=fill.order_id,
        symbol=fill.symbol,
        side=fill.side,
        contracts=fill.contracts,
        price=fill.price,
        timestamp=fill.timestamp,
        fees=fill.fees,
        fill_model=SHADOW_FILL_MODEL,
        displayed_size_used=fill.displayed_size_used,
    )


def _base_rejection_row(order: OptionOrder | None, symbol: str, stage: str, reasons: list[str], targets: pd.DataFrame) -> dict[str, Any]:
    lookup = _target_lookup(targets)
    row = lookup.loc[symbol] if symbol in lookup.index else pd.Series(dtype=object)
    return {
        "order_id": order.order_id if order is not None else "",
        "decision_time": order.decision_time if order is not None else row.get("decision_time", ""),
        "symbol": symbol,
        "stage": stage,
        "reasons": ";".join(str(x) for x in reasons),
        "contracts": order.contracts if order is not None else 0,
        "target_weight": row.get("target_weight", np.nan),
    }


def _build_margin_estimate(
    order: OptionOrder,
    contract: OptionContract,
    quote: QuoteSnapshot,
    target_row: pd.Series,
    preview: tuple[float | None, str] | None,
) -> MarginEstimate:
    underlying_price = pd.to_numeric(target_row.get("underlying_price", target_row.get("spot", np.nan)), errors="coerce")
    broker_preview, preview_status = preview if preview is not None else (None, "missing")
    if not np.isfinite(underlying_price):
        return MarginEstimate(
            symbol=order.symbol,
            margin_requirement=float(broker_preview) if broker_preview is not None else np.nan,
            stress_loss=np.nan,
            assignment_notional=np.nan,
            margin_source="broker_preview_only" if broker_preview is not None else "missing_underlying_price",
            preview_status="pass" if broker_preview is not None and str(preview_status).lower() == "pass" else "fail",
        )
    estimate = conservative_order_margin(
        order,
        contract,
        quote,
        float(underlying_price),
        broker_margin_preview=broker_preview,
    )
    if str(preview_status).lower() not in {"pass", "ok", "true", "1"} and broker_preview is not None:
        return MarginEstimate(
            symbol=estimate.symbol,
            margin_requirement=estimate.margin_requirement,
            stress_loss=estimate.stress_loss,
            assignment_notional=estimate.assignment_notional,
            margin_source=estimate.margin_source,
            preview_status=str(preview_status),
        )
    return estimate


def run_shadow_rebalance(
    targets: pd.DataFrame,
    quotes: Mapping[str, QuoteSnapshot],
    *,
    config: ShadowRunConfig,
    positions: Mapping[str, Position] | None = None,
    margin_previews: Mapping[str, tuple[float | None, str]] | None = None,
    manual_rejections: Mapping[str, str] | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    if config.nav <= 0:
        raise ValueError("nav must be positive")
    positions = positions or {}
    margin_previews = margin_previews or {}
    manual_rejections = manual_rejections or {}
    contracts = _contracts_from_targets(targets)
    weights = _target_weight_series(targets)
    policy = OrderPolicy(
        max_spread_bps=float(config.max_spread_bps),
        max_quote_age=config.max_quote_age,
        max_participation_of_displayed_size=float(config.max_participation_of_displayed_size),
    )
    executable_quotes: dict[str, QuoteSnapshot] = {}
    rejections: list[dict[str, Any]] = []
    for symbol, target_weight in weights.items():
        if abs(float(target_weight)) <= 1e-14:
            continue
        quote = quotes.get(symbol)
        if quote is None:
            rejections.append(_base_rejection_row(None, symbol, "quote", ["missing_quote"], targets))
            continue
        market_ok = _market_hours_ok(quote.ts_event)
        ok, reasons = quote.executable_at(config.decision_time, max_age=config.max_quote_age, max_spread_bps=config.max_spread_bps)
        if not market_ok and not config.allow_non_market_hours:
            reasons = [*reasons, "outside_regular_market_hours"]
            ok = False
        if ok:
            executable_quotes[symbol] = quote
        else:
            rejections.append(_base_rejection_row(None, symbol, "quote", reasons, targets))

    orders = target_weights_to_orders(
        weights,
        positions,
        contracts,
        executable_quotes,
        nav=float(config.nav),
        decision_time=config.decision_time,
        policy=policy,
        min_contracts=int(config.min_contracts),
        reason_code="shadow_rebalance",
    )
    order_symbols = {order.symbol for order in orders}
    for symbol, target_weight in weights.items():
        if abs(float(target_weight)) <= 1e-14 or symbol in order_symbols:
            continue
        if symbol in executable_quotes and symbol not in {row["symbol"] for row in rejections}:
            rejections.append(_base_rejection_row(None, symbol, "sizing", ["below_min_contract_or_no_delta"], targets))

    lookup = _target_lookup(targets)
    account = AccountState(net_liquidation=float(config.nav), cash=float(config.nav), timestamp=config.decision_time)
    fills: list[Fill] = []
    margin_rows: list[MarginEstimate] = []
    gate_rows: list[dict[str, Any]] = []
    for order in orders:
        if order.symbol in manual_rejections:
            rejections.append(_base_rejection_row(order, order.symbol, "manual", [manual_rejections[order.symbol]], targets))
            continue
        quote = executable_quotes[order.symbol]
        contract = contracts[order.symbol]
        target_row = lookup.loc[order.symbol]
        margin = _build_margin_estimate(order, contract, quote, target_row, margin_previews.get(order.symbol))
        margin_rows.append(margin)
        displayed = quote.ask_size if order.side == "buy" else quote.bid_size
        fill_probability = min(1.0, max(0.0, float(displayed) * config.max_participation_of_displayed_size / max(order.contracts, 1)))
        gate = evaluate_pre_trade_gate(
            order,
            contract,
            quote,
            account,
            margin,
            open_interest=float(pd.to_numeric(target_row.get("open_interest", np.inf), errors="coerce")),
            volume=float(pd.to_numeric(target_row.get("volume", np.inf), errors="coerce")),
            fill_probability=fill_probability,
            option_mark=float(pd.to_numeric(target_row.get("mark", quote.mid), errors="coerce")),
            underlying_price=float(pd.to_numeric(target_row.get("underlying_price", target_row.get("spot", np.nan)), errors="coerce"))
            if np.isfinite(pd.to_numeric(target_row.get("underlying_price", target_row.get("spot", np.nan)), errors="coerce"))
            else None,
            config=RiskGateConfig(max_spread_bps=config.max_spread_bps, max_quote_age=config.max_quote_age),
        )
        gate_rows.append(gate.ledger_row(order_id=order.order_id, symbol=order.symbol))
        if not gate.passed:
            rejections.append(_base_rejection_row(order, order.symbol, "risk_gate", list(gate.reasons), targets))
            continue
        fill, unfilled, reasons = estimate_nbbo_fill(order, quote, policy=policy)
        if fill is None:
            rejections.append(_base_rejection_row(order, order.symbol, "fill", reasons, targets))
            continue
        fills.append(_shadow_fill(fill))
        if unfilled > 0:
            row = _base_rejection_row(order, order.symbol, "partial_fill", ["unfilled_displayed_size_remainder"], targets)
            row["contracts"] = int(unfilled)
            rejections.append(row)

    target_ledger = targets.loc[:, [col for col in SHADOW_TARGET_COLUMNS if col in targets.columns]].copy()
    quote_ledger = build_market_data_ledger(list(quotes.values()), symbol_map_version=config.symbol_map_version)
    if not quote_ledger.empty:
        quote_ledger["market_hours_ok"] = quote_ledger["ts_event"].map(_market_hours_ok)
    execution_ledger = build_execution_ledger(orders)
    fill_ledger = build_fill_ledger(fills)
    margin_ledger = pd.DataFrame([m.ledger_row() for m in margin_rows])
    gate_ledger = pd.DataFrame(gate_rows)
    rejected_ledger = pd.DataFrame(rejections)
    recon_ledger = _shadow_reconciliation_ledger(target_ledger, quote_ledger, execution_ledger, fill_ledger, margin_ledger, rejected_ledger)
    summary = _shadow_summary(target_ledger, quote_ledger, execution_ledger, fill_ledger, margin_ledger, rejected_ledger, recon_ledger, config)
    return {
        "shadow_target_ledger": target_ledger,
        "shadow_quote_ledger": quote_ledger,
        "shadow_execution_ledger": execution_ledger,
        "shadow_fill_ledger": fill_ledger,
        "shadow_margin_ledger": margin_ledger,
        "shadow_gate_ledger": gate_ledger,
        "shadow_rejected_order_ledger": rejected_ledger,
        "shadow_reconciliation_ledger": recon_ledger,
        "shadow_summary": summary,
    }


def _shadow_reconciliation_ledger(
    targets: pd.DataFrame,
    quotes: pd.DataFrame,
    execution: pd.DataFrame,
    fills: pd.DataFrame,
    margin: pd.DataFrame,
    rejected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rows.append({"check": "targets_exist", "passed": not targets.empty, "observed": len(targets), "expected": ">0"})
    rows.append({"check": "quotes_exist", "passed": not quotes.empty, "observed": len(quotes), "expected": ">0"})
    ok_ts, reasons = validate_timestamp_monotonicity(quotes) if not quotes.empty else (False, ["empty_quote_ledger"])
    rows.append({"check": "quote_timestamp_monotonicity", "passed": ok_ts, "observed": ";".join(reasons), "expected": "pass"})
    if "market_hours_ok" in quotes:
        rows.append({"check": "quotes_market_hours", "passed": bool(quotes["market_hours_ok"].all()), "observed": quotes["market_hours_ok"].value_counts().to_dict(), "expected": "all true"})
    if fills.empty:
        rows.append({"check": "fills_shadow_labeled", "passed": True, "observed": "no fills", "expected": SHADOW_FILL_MODEL})
    else:
        rows.append({"check": "fills_shadow_labeled", "passed": fills["fill_model"].astype(str).eq(SHADOW_FILL_MODEL).all(), "observed": fills["fill_model"].value_counts().to_dict(), "expected": SHADOW_FILL_MODEL})
    rows.append({"check": "production_fill_labels_absent", "passed": "fill_model" not in fills or not fills.get("fill_model", pd.Series(dtype=object)).astype(str).str.startswith("nbbo_displayed_size").any(), "observed": fills.get("fill_model", pd.Series(dtype=object)).astype(str).unique().tolist() if not fills.empty else [], "expected": "no production fill labels"})
    rows.append({"check": "margin_logged_for_evaluated_orders", "passed": execution.empty or len(margin) <= len(execution), "observed": {"orders": len(execution), "margin_rows": len(margin)}, "expected": "margin rows <= orders"})
    rows.append({"check": "rejections_are_audit_only", "passed": True, "observed": len(rejected), "expected": "informational"})
    return pd.DataFrame(rows)


def _shadow_summary(
    targets: pd.DataFrame,
    quotes: pd.DataFrame,
    execution: pd.DataFrame,
    fills: pd.DataFrame,
    margin: pd.DataFrame,
    rejected: pd.DataFrame,
    recon: pd.DataFrame,
    config: ShadowRunConfig,
) -> dict[str, Any]:
    return {
        "status": "pass" if not recon.empty and recon["passed"].astype(bool).all() else "mixed",
        "claim_status": "shadow_only_not_live_trading_evidence",
        "nav": float(config.nav),
        "decision_time": pd.Timestamp(config.decision_time).isoformat(),
        "target_rows": int(len(targets)),
        "quote_rows": int(len(quotes)),
        "orders": int(len(execution)),
        "fills": int(len(fills)),
        "rejections": int(len(rejected)),
        "filled_contracts": int(pd.to_numeric(fills.get("contracts", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not fills.empty else 0,
        "not_production_certification": True,
        "shadow_fill_model": SHADOW_FILL_MODEL,
    }


def write_shadow_outputs(outputs: Mapping[str, pd.DataFrame | dict[str, Any]], out_dir: str | Path) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(path / f"{name}.csv", index=False)
        elif isinstance(value, dict):
            (path / f"{name}.json").write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
            _write_summary_md(value, path / f"{name}.md")


def _write_summary_md(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Shadow Trading Summary",
        "",
        f"Status: `{summary.get('status')}`",
        f"Claim status: `{summary.get('claim_status')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ["target_rows", "quote_rows", "orders", "fills", "rejections", "filled_contracts"]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.append("")
    lines.append("Shadow ledgers are forward-validation evidence only. They are not broker-executed live fills.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ShadowVerifier:
    """Verifier for shadow ledgers that keeps them separate from production proof."""

    def __init__(self, shadow_dir: str | Path) -> None:
        self.shadow_dir = Path(shadow_dir)
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, observed: Any = "", expected: Any = "") -> None:
        self.checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})

    def _read(self, name: str) -> pd.DataFrame:
        path = self.shadow_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def run(self) -> dict[str, Any]:
        self.checks = []
        targets = self._read("shadow_target_ledger.csv")
        quotes = self._read("shadow_quote_ledger.csv")
        execution = self._read("shadow_execution_ledger.csv")
        fills = self._read("shadow_fill_ledger.csv")
        summary_path = self.shadow_dir / "shadow_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        self.check("target ledger exists", not targets.empty, len(targets), ">0 rows")
        self.check("quote ledger exists", not quotes.empty, len(quotes), ">0 rows")
        self.check("execution ledger exists or explicit no orders", not execution.empty or bool(summary), len(execution), "orders or summary")
        if not fills.empty and "fill_model" in fills:
            self.check("fills use shadow label", fills["fill_model"].astype(str).eq(SHADOW_FILL_MODEL).all(), fills["fill_model"].value_counts().to_dict(), SHADOW_FILL_MODEL)
            self.check("production fill labels absent", not fills["fill_model"].astype(str).str.startswith("nbbo_displayed_size").any(), fills["fill_model"].unique().tolist(), "shadow labels only")
        else:
            self.check("fills use shadow label", True, "no fills", SHADOW_FILL_MODEL)
        self.check("summary blocks production claim", bool(summary.get("not_production_certification", False)), summary, "not_production_certification=true")
        checks = pd.DataFrame(self.checks)
        checks.to_csv(self.shadow_dir / "shadow_verification_checks.csv", index=False)
        failed = checks[~checks["passed"]] if not checks.empty else pd.DataFrame()
        failed.to_csv(self.shadow_dir / "shadow_failed_checks.csv", index=False)
        result = {
            "status": "pass" if failed.empty else "fail",
            "failures": int(len(failed)),
            "total_checks": int(len(checks)),
            "shadow_dir": str(self.shadow_dir),
        }
        (self.shadow_dir / "shadow_verification_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run broker-neutral option shadow-trading ledgers.")
    parser.add_argument("--targets", required=True, help="CSV with target contracts and target_weight.")
    parser.add_argument("--quotes", required=True, help="CSV with market-hours NBBO/CBBO snapshots and size.")
    parser.add_argument("--out-dir", required=True, help="Output directory for shadow ledgers.")
    parser.add_argument("--nav", type=float, required=True)
    parser.add_argument("--decision-time", required=True)
    parser.add_argument("--positions", default=None)
    parser.add_argument("--margin-previews", default=None)
    parser.add_argument("--rejections", default=None)
    parser.add_argument("--max-quote-age-seconds", type=float, default=900.0)
    parser.add_argument("--max-spread-bps", type=float, default=500.0)
    parser.add_argument("--max-participation", type=float, default=0.25)
    parser.add_argument("--allow-non-market-hours", action="store_true")
    args = parser.parse_args(argv)

    decision_time = pd.Timestamp(args.decision_time)
    if decision_time.tzinfo is None:
        decision_time = decision_time.tz_localize("UTC")
    else:
        decision_time = decision_time.tz_convert("UTC")
    targets = load_shadow_targets(args.targets, decision_time=decision_time)
    quotes = load_shadow_quotes(args.quotes)
    config = ShadowRunConfig(
        nav=float(args.nav),
        decision_time=decision_time,
        out_dir=Path(args.out_dir),
        max_quote_age=timedelta(seconds=float(args.max_quote_age_seconds)),
        max_spread_bps=float(args.max_spread_bps),
        max_participation_of_displayed_size=float(args.max_participation),
        allow_non_market_hours=bool(args.allow_non_market_hours),
    )
    outputs = run_shadow_rebalance(
        targets,
        quotes,
        config=config,
        positions=load_positions(args.positions),
        margin_previews=_margin_preview_map(args.margin_previews),
        manual_rejections=_manual_rejection_map(args.rejections),
    )
    write_shadow_outputs(outputs, args.out_dir)
    verify = ShadowVerifier(args.out_dir).run()
    print(json.dumps(verify, indent=2))
    return 0 if verify["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
