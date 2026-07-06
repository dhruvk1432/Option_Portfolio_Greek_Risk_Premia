"""Breadth/capacity experiment for the option-only Markowitz paper.

This standalone script widens the equity-option universe when additional
underlyings are present in the local feature store, then reruns the paper's
strategy and cost-ledger path across an AUM sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.run_empirics import (
    BUCKETS,
    MIN_OPTION_MARK,
    PRIMARY_UNDERLYINGS,
    ROOT,
    TRAIN_END,
    build_expiry_proxy_return_panel,
    load_raw_close_panel,
    make_model,
    representative_specs,
    split_adjusted_spot_panel,
    strategy_weights,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    load_cbbo_spread_surface,
)


BREADTH_48 = [
    "AAL",
    "ADBE",
    "AMAT",
    "AMD",
    "AVGO",
    "BA",
    "BAC",
    "BKNG",
    "C",
    "CCL",
    "CHTR",
    "CMCSA",
    "COST",
    "CRM",
    "CSCO",
    "CVX",
    "DAL",
    "DIS",
    "GE",
    "GILD",
    "GOOG",
    "GS",
    "HD",
    "INTC",
    "JNJ",
    "KO",
    "LLY",
    "LRCX",
    "MA",
    "MRK",
    "MU",
    "NFLX",
    "NKE",
    "ORCL",
    "PFE",
    "PG",
    "PYPL",
    "QCOM",
    "SBUX",
    "T",
    "TXN",
    "UAL",
    "UNH",
    "V",
    "VZ",
    "WFC",
    "WMT",
    "XOM",
]

AUMS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000]
OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "breadth_experiment"
STRATEGY_NAMES = ["Greek Markowitz", "Delta neutral", "Equal premium", "Equal risk"]


def load_bucket_panel_poc(
    underlyings: Sequence[str] | None = None,
    poc_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw filtered rows, representative bucket rows and bucket returns."""

    universe = list(underlyings or PRIMARY_UNDERLYINGS)
    columns = [
        "symbol",
        "underlying",
        "snap_date",
        "expiry",
        "strike",
        "kind",
        "spot",
        "close",
        "volume",
        "tenor_days",
        "moneyness_bucket",
        "iv_proxy",
        "cbbo_median_relative_spread",
        "delta",
        "gamma",
        "vega",
        "theta",
    ]
    panel = pd.read_parquet(ROOT / "data/feature_store/option_greek_proxy_panel.parquet", columns=columns)
    raw_spread = pd.to_numeric(panel["cbbo_median_relative_spread"], errors="coerce")
    panel["breadth_spread_source"] = np.where(raw_spread.gt(0), "panel_cbbo", "missing")
    poc_missing = panel["underlying"].isin(set(poc_names or [])) & ~raw_spread.gt(0)
    panel.loc[poc_missing, "breadth_spread_source"] = "poc_missing_cbbo"
    panel["snap_date"] = pd.to_datetime(panel["snap_date"])
    panel["expiry"] = pd.to_datetime(panel["expiry"])
    spread_ok = raw_spread.le(0.20) | poc_missing
    panel = panel[
        panel["underlying"].isin(universe)
        & panel["moneyness_bucket"].isin(BUCKETS)
        & panel["close"].ge(MIN_OPTION_MARK)
        & panel["volume"].ge(10)
        & spread_ok
    ].copy()
    for col in ["delta", "gamma", "vega", "theta", "iv_proxy", "spot", "strike"]:
        panel = panel[np.isfinite(pd.to_numeric(panel[col], errors="coerce"))]
    panel["mark"] = panel["close"].astype(float)
    panel["asset_id"] = (
        panel["underlying"].astype(str)
        + "_"
        + panel["kind"].astype(str)
        + "_"
        + panel["moneyness_bucket"].astype(str)
    )
    panel = panel.sort_values(
        ["snap_date", "asset_id", "volume", "cbbo_median_relative_spread"],
        ascending=[True, True, False, True],
    )
    reps = panel.groupby(["snap_date", "asset_id"], as_index=False).head(1).copy()
    reps = reps.sort_values(["asset_id", "snap_date"])
    raw_spot = (
        panel.groupby(["snap_date", "underlying"])["spot"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=universe)
    )
    raw_close = load_raw_close_panel(universe)
    _, daily_split_factors, _ = split_adjusted_spot_panel(raw_close)
    returns, _ = build_expiry_proxy_return_panel(reps, raw_close, daily_split_factors)
    returns = returns.dropna(how="all")
    enough = returns.loc[:TRAIN_END].count() >= 36
    returns = returns.loc[:, enough]
    reps = reps[reps["asset_id"].isin(returns.columns)].copy()
    return panel, reps, returns


def _sharpe(series: pd.Series) -> float:
    x = series.dropna()
    std = x.std(ddof=1)
    return float(np.sqrt(12) * x.mean() / std) if std > 0 else float("nan")


def _strategy_subset(strat: dict[str, pd.Series]) -> dict[str, pd.Series]:
    strategies = {name: strat.get(name) for name in STRATEGY_NAMES if strat.get(name) is not None}
    missing = [name for name in STRATEGY_NAMES if name not in strategies]
    if missing:
        print(f"skipping missing strategy key(s): {', '.join(missing)}")
    return strategies


def run_universe(
    label: str,
    underlyings: Sequence[str],
    poc_names: Sequence[str],
    aums: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    panel, reps, returns = load_bucket_panel_poc(underlyings, poc_names)
    raw_close = load_raw_close_panel(underlyings)
    _, split_factors, _ = split_adjusted_spot_panel(raw_close)
    _, detail = build_expiry_proxy_return_panel(reps, raw_close, split_factors)
    detail = detail[detail["asset_id"].isin(returns.columns)].copy()
    detail["asset_class"] = "equity_option"
    reps = reps.copy()
    reps["asset_class"] = "equity_option"

    spec = representative_specs(reps, returns)
    returns = returns.reindex(columns=spec.index).dropna(how="all")
    model, _ = make_model(spec, returns, reps, list(underlyings))
    strat = strategy_weights(model, list(underlyings))
    strategies = _strategy_subset(strat)
    test_returns = returns.loc[returns.index > TRAIN_END, model.contracts].fillna(0.0)

    ret_frame = pd.DataFrame(index=test_returns.index)
    for name, weights in strategies.items():
        ret_frame[name] = model.portfolio_return_series(test_returns, weights)

    model_underlyings = spec.reindex(model.contracts)["underlying"].astype(str)
    n_underlyings_held = int(model_underlyings.nunique())
    n_contracts = int(len(model.contracts))

    for aum in aums:
        cfg = ResearchCostConfig(nav_for_capacity=float(aum))
        surface = load_cbbo_spread_surface(ROOT, cfg.cbbo_spread_surface_path) if cfg.use_cbbo_spread_surface else None
        cost_inputs = build_cost_input_ledger(reps, detail, ROOT, cfg, spread_surface=surface)
        gross_frame = ret_frame[list(strategies)].copy()
        net_frame, cost_ledger, capacity_ledger, margin_ledger, assignment_ledger = compute_strategy_cost_ledgers(
            gross_frame, strategies, cost_inputs, cfg
        )
        _ = margin_ledger, assignment_ledger
        for name in strategies:
            g = ret_frame[name].dropna()
            nser = net_frame[name].dropna()
            cl = cost_ledger[cost_ledger["strategy"] == name] if len(cost_ledger) else cost_ledger
            cap = capacity_ledger[capacity_ledger["strategy"] == name] if len(capacity_ledger) else capacity_ledger
            rows.append(
                {
                    "universe": label,
                    "strategy": name,
                    "aum": float(aum),
                    "n_underlyings": n_underlyings_held,
                    "n_contracts": n_contracts,
                    "gross_sharpe": _sharpe(g),
                    "net_sharpe": _sharpe(nser),
                    "gross_ann_ret": float(g.mean() * 12),
                    "net_ann_ret": float(nser.mean() * 12),
                    "mean_monthly_capacity_cost": (
                        float(cl.groupby("return_date")["capacity_cost_nav"].sum().mean()) if len(cl) else 0.0
                    ),
                    "max_capacity_ratio": (
                        float(cap["capacity_ratio"].replace([np.inf, -np.inf], np.nan).max())
                        if len(cap)
                        else float("nan")
                    ),
                    "capacity_penalized_share": (
                        float((cap["capacity_status"] == "penalized").mean()) if len(cap) else 0.0
                    ),
                }
            )
    return rows


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def _fmt_aum(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.0f}"


def _fmt_num(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_summary(results: pd.DataFrame, present_new: Sequence[str], aums: Sequence[float]) -> str:
    gm = results[results["strategy"] == "Greek Markowitz"].copy()
    lines = [
        "# Breadth Capacity Experiment",
        "",
        f"New names present in panel: {len(present_new)}/48",
        "",
        "| Universe | Underlyings | Contracts | $1M net Sharpe | $1M mean capacity cost | $1M max capacity ratio | Smallest AUM with net Sharpe > 0 | Highest AUM with net Sharpe > 0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    one_m = gm[np.isclose(gm["aum"].astype(float), 1_000_000.0)]
    survival: dict[str, dict[str, float | None]] = {}
    for universe in ["n8", "nAll"]:
        sub = gm[gm["universe"] == universe].sort_values("aum")
        positives = sub[sub["net_sharpe"] > 0]
        survival[universe] = {
            "min_positive": float(positives["aum"].min()) if len(positives) else None,
            "max_positive": float(positives["aum"].max()) if len(positives) else None,
        }
        row_1m = one_m[one_m["universe"] == universe]
        if len(row_1m):
            row = row_1m.iloc[0]
            lines.append(
                "| "
                + " | ".join(
                    [
                        universe,
                        str(int(row["n_underlyings"])),
                        str(int(row["n_contracts"])),
                        _fmt_num(row["net_sharpe"]),
                        _fmt_num(row["mean_monthly_capacity_cost"], 6),
                        _fmt_num(row["max_capacity_ratio"]),
                        _fmt_aum(survival[universe]["min_positive"]),
                        _fmt_aum(survival[universe]["max_positive"]),
                    ]
                )
                + " |"
            )
        else:
            lines.append(f"| {universe} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    n8_1m = one_m[one_m["universe"] == "n8"]
    all_1m = one_m[one_m["universe"] == "nAll"]
    lines.extend(["", "## Interpretation", ""])
    if len(n8_1m) and len(all_1m):
        n8_cost = float(n8_1m.iloc[0]["mean_monthly_capacity_cost"])
        all_cost = float(all_1m.iloc[0]["mean_monthly_capacity_cost"])
        n8_survival = survival["n8"]["max_positive"]
        all_survival = survival["nAll"]["max_positive"]
        cost_text = (
            "nAll reduces the $1M mean monthly capacity cost versus n8."
            if all_cost < n8_cost
            else "nAll does not reduce the $1M mean monthly capacity cost versus n8."
        )
        if n8_survival is None and all_survival is None:
            survival_text = "Neither universe has positive net Sharpe at any tested AUM."
        elif n8_survival is None:
            survival_text = "nAll raises survival versus n8 because only nAll has positive net Sharpe in the sweep."
        elif all_survival is None:
            survival_text = "nAll lowers survival versus n8 because nAll has no positive net Sharpe in the sweep."
        elif all_survival > n8_survival:
            survival_text = "nAll raises the survival AUM versus n8."
        elif all_survival < n8_survival:
            survival_text = "nAll lowers the survival AUM versus n8."
        else:
            survival_text = "nAll does not raise the survival AUM versus n8."
        lines.append(f"{cost_text} {survival_text}")
    else:
        lines.append("Greek Markowitz rows were not available for both universes, so the capacity comparison is incomplete.")
    lines.append("")
    lines.append(f"AUM sweep: {', '.join(_fmt_aum(aum) for aum in aums)}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    avail = set(
        pd.read_parquet(
            ROOT / "data/feature_store/option_greek_proxy_panel.parquet",
            columns=["underlying"],
        )["underlying"].unique()
    )
    present_new = [name for name in BREADTH_48 if name in avail]
    universes = [
        ("n8", list(PRIMARY_UNDERLYINGS), []),
        ("nAll", list(PRIMARY_UNDERLYINGS) + present_new, present_new),
    ]
    print(f"new names present in panel: {len(present_new)}/48")

    rows: list[dict[str, object]] = []
    for label, underlyings, poc_names in universes:
        print(f"running {label}: {len(underlyings)} requested underlyings")
        rows.extend(run_universe(label, underlyings, poc_names, AUMS))

    results = pd.DataFrame(rows)
    csv_path = OUT_DIR / "breadth_capacity_results.csv"
    json_path = OUT_DIR / "breadth_capacity_results.json"
    summary_path = OUT_DIR / "breadth_capacity_summary.md"

    results.to_csv(csv_path, index=False)
    payload = {
        "records": results.to_dict(orient="records"),
        "provenance": {
            "universes": [
                {"label": label, "underlyings": list(underlyings), "poc_names": list(poc_names)}
                for label, underlyings, poc_names in universes
            ],
            "AUMS": AUMS,
            "present_new": present_new,
            "timestamp": None,
        },
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    summary = _build_summary(results, present_new, AUMS)
    summary_path.write_text(summary, encoding="utf-8")

    gm = results[results["strategy"] == "Greek Markowitz"].sort_values(["universe", "aum"])
    print("")
    print("Greek Markowitz AUM sweep:")
    if gm.empty:
        print("  no Greek Markowitz rows produced")
    else:
        for row in gm.itertuples(index=False):
            print(
                "  "
                f"{row.universe} aum={_fmt_aum(row.aum)} "
                f"gross_sharpe={_fmt_num(row.gross_sharpe)} "
                f"net_sharpe={_fmt_num(row.net_sharpe)} "
                f"mean_capacity_cost={_fmt_num(row.mean_monthly_capacity_cost, 6)} "
                f"max_capacity_ratio={_fmt_num(row.max_capacity_ratio)}"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
