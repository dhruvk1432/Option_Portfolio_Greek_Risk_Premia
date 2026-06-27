
"""Conservative post-cost research accounting for the option-only Markowitz paper.

This module is intentionally not a live execution simulator. It turns point-in-time
option holding ledgers into transparent research cost, capacity, borrow, margin, and
assignment diagnostics so the paper can separate gross theory evidence from net economic
robustness.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ResearchCostConfig:
    nav_for_capacity: float = 1_000_000.0
    option_multiplier: float = 100.0
    fee_per_contract_per_side: float = 0.75
    spread_cross_fraction: float = 1.0
    default_equity_option_rel_spread: float = 0.10
    default_vix_option_rel_spread: float = 0.15
    slippage_bps_per_side: float = 5.0
    max_volume_participation: float = 0.10
    max_oi_participation: float = 0.02
    impact_cost_rate: float = 0.0025
    margin_funding_rate: float = 0.02
    short_option_margin_floor: float = 0.15
    stress_margin_rate: float = 0.25
    assignment_penalty_bps: float = 10.0


def load_borrow_proxy(root: Path = ROOT) -> pd.DataFrame:
    path = root / "data/feature_store/option_borrow_proxy_layer.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = ["symbol", "underlying", "snap_date", "expiry", "strike", "kind", "borrow_rate_proxy", "borrow_source"]
    df = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path, columns=[]).columns] if False else None)
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    if "snap_date" in out:
        out["snap_date"] = pd.to_datetime(out["snap_date"], errors="coerce").dt.normalize()
    if "expiry" in out:
        out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()
    if "borrow_rate_proxy" in out:
        out["borrow_rate_proxy"] = pd.to_numeric(out["borrow_rate_proxy"], errors="coerce")
    return out


def build_cost_input_ledger(
    reps: pd.DataFrame,
    return_detail: pd.DataFrame,
    root: Path = ROOT,
    config: ResearchCostConfig = ResearchCostConfig(),
) -> pd.DataFrame:
    if return_detail.empty:
        return pd.DataFrame()
    detail = return_detail.copy()
    for col in ["return_date", "decision_date", "expiry"]:
        if col in detail:
            detail[col] = pd.to_datetime(detail[col], errors="coerce").dt.normalize()
    rep_cols = [
        c
        for c in [
            "snap_date",
            "asset_id",
            "symbol",
            "volume",
            "open_interest",
            "cbbo_median_relative_spread",
            "moneyness_bucket",
        ]
        if c in reps.columns
    ]
    reps_slim = reps[rep_cols].copy() if rep_cols else pd.DataFrame(columns=["snap_date", "asset_id"])
    if "snap_date" in reps_slim:
        reps_slim["snap_date"] = pd.to_datetime(reps_slim["snap_date"], errors="coerce").dt.normalize()
    cost = detail.merge(
        reps_slim.drop_duplicates(["snap_date", "asset_id"]),
        left_on=["decision_date", "asset_id"],
        right_on=["snap_date", "asset_id"],
        how="left",
        suffixes=("", "_rep"),
    )
    if "symbol_rep" in cost:
        cost["symbol"] = cost["symbol"].where(cost["symbol"].notna(), cost["symbol_rep"])
    borrow = load_borrow_proxy(root)
    if not borrow.empty and {"symbol", "snap_date", "borrow_rate_proxy"}.issubset(borrow.columns):
        cost = cost.merge(
            borrow[["symbol", "snap_date", "borrow_rate_proxy", "borrow_source"]].drop_duplicates(["symbol", "snap_date"]),
            left_on=["symbol", "decision_date"],
            right_on=["symbol", "snap_date"],
            how="left",
            suffixes=("", "_borrow"),
        )
    else:
        cost["borrow_rate_proxy"] = np.nan
        cost["borrow_source"] = "missing"
    cost["mark"] = pd.to_numeric(cost.get("mark", np.nan), errors="coerce")
    cost["volume"] = pd.to_numeric(cost.get("volume", np.nan), errors="coerce")
    if "open_interest" not in cost:
        cost["open_interest"] = np.nan
    cost["open_interest"] = pd.to_numeric(cost["open_interest"], errors="coerce")
    if "cbbo_median_relative_spread" in cost:
        rel = pd.to_numeric(cost["cbbo_median_relative_spread"], errors="coerce")
    else:
        rel = pd.Series(np.nan, index=cost.index, dtype=float)
    is_vix = cost.get("asset_class", pd.Series("", index=cost.index)).astype(str).eq("vix_option")
    defaults = np.where(is_vix, config.default_vix_option_rel_spread, config.default_equity_option_rel_spread)
    cost["relative_spread"] = rel.where(rel.gt(0), defaults).clip(lower=0.0, upper=1.5)
    cost["borrow_rate_proxy"] = pd.to_numeric(cost["borrow_rate_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    cost["holding_years"] = pd.to_numeric(cost.get("expiry_days", 21), errors="coerce").fillna(21.0).clip(lower=0.0) / 365.0
    cost["available_volume_contracts"] = cost["volume"].fillna(0.0).clip(lower=0.0)
    cost["available_oi_contracts"] = cost["open_interest"].fillna(np.inf)
    return cost.replace([np.inf, -np.inf], np.nan)


def _assignment_risk(row: pd.Series, short_position: bool) -> bool:
    if not short_position:
        return False
    if str(row.get("asset_class", "equity_option")) != "equity_option":
        return False
    spot = float(row.get("start_spot", np.nan))
    strike = float(row.get("strike", np.nan))
    mark = float(row.get("mark", np.nan))
    if not np.isfinite([spot, strike, mark]).all() or mark <= 0:
        return False
    kind = str(row.get("kind", row.get("right", ""))).lower()
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    extrinsic = mark - intrinsic
    deep_itm = (kind == "call" and spot > 1.03 * strike) or (kind == "put" and strike > 1.03 * spot)
    return bool(deep_itm and extrinsic <= max(0.10 * mark, 0.05))


def compute_strategy_cost_ledgers(
    gross_returns: pd.DataFrame,
    strategies: dict[str, pd.Series],
    cost_inputs: pd.DataFrame,
    config: ResearchCostConfig = ResearchCostConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cost_inputs.empty:
        empty = pd.DataFrame(index=gross_returns.index)
        return gross_returns.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    capacity_rows = []
    margin_rows = []
    assignment_rows = []
    cost_by_asset = cost_inputs.set_index(["return_date", "asset_id"], drop=False)
    net = pd.DataFrame(index=gross_returns.index)
    for strategy, weights in strategies.items():
        net[strategy] = gross_returns[strategy] if strategy in gross_returns else np.nan
        for return_date in gross_returns.index:
            gross_cost = 0.0
            margin_req = 0.0
            stress_margin = 0.0
            assignment_notional = 0.0
            for asset_id, weight in weights.items():
                w = float(weight)
                if abs(w) <= 1e-14:
                    continue
                key = (pd.Timestamp(return_date).normalize(), asset_id)
                if key not in cost_by_asset.index:
                    continue
                row = cost_by_asset.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                mark = float(row.get("mark", np.nan))
                if not np.isfinite(mark) or mark <= 0:
                    continue
                rel_spread = float(row.get("relative_spread", 0.10))
                holding_years = float(row.get("holding_years", 21 / 365))
                fee_return = 2.0 * config.fee_per_contract_per_side / (mark * config.option_multiplier)
                spread_return = config.spread_cross_fraction * rel_spread
                slippage_return = 2.0 * config.slippage_bps_per_side / 10_000.0
                is_short = w < 0
                kind = str(row.get("kind", row.get("right", ""))).lower()
                borrow_return = abs(w) * float(row.get("borrow_rate_proxy", 0.0)) * holding_years if is_short and kind == "call" else 0.0
                est_contracts = abs(w) * config.nav_for_capacity / (mark * config.option_multiplier)
                vol_cap = float(row.get("available_volume_contracts", 0.0)) * config.max_volume_participation
                oi_val = row.get("available_oi_contracts", np.nan)
                oi_cap = float(oi_val) * config.max_oi_participation if np.isfinite(float(oi_val)) else np.inf
                cap = max(1.0, min(vol_cap if vol_cap > 0 else np.inf, oi_cap))
                capacity_ratio = est_contracts / cap if cap > 0 and np.isfinite(cap) else np.inf
                capacity_penalty_return = max(capacity_ratio - 1.0, 0.0) ** 2 * config.impact_cost_rate
                start_spot = float(row.get("start_spot", np.nan))
                short_margin = abs(w) * max(config.short_option_margin_floor, 0.20 * start_spot / mark) if is_short and np.isfinite(start_spot) else abs(w)
                margin_component = short_margin if is_short else abs(w)
                margin_drag_return = margin_component * config.margin_funding_rate * holding_years
                assignment_flag = _assignment_risk(row, is_short)
                assignment_penalty = abs(w) * config.assignment_penalty_bps / 10_000.0 if assignment_flag else 0.0
                per_weight_cost = spread_return + fee_return + slippage_return + capacity_penalty_return
                cost_nav = abs(w) * per_weight_cost + borrow_return + margin_drag_return + assignment_penalty
                gross_cost += cost_nav
                margin_req += margin_component
                stress_margin += abs(w) * config.stress_margin_rate
                if assignment_flag:
                    assignment_notional += abs(w) * config.nav_for_capacity
                rows.append(
                    {
                        "return_date": return_date,
                        "strategy": strategy,
                        "asset_id": asset_id,
                        "weight": w,
                        "mark": mark,
                        "relative_spread": rel_spread,
                        "spread_cost_nav": abs(w) * spread_return,
                        "fee_cost_nav": abs(w) * fee_return,
                        "slippage_cost_nav": abs(w) * slippage_return,
                        "borrow_cost_nav": borrow_return,
                        "margin_drag_nav": margin_drag_return,
                        "capacity_cost_nav": abs(w) * capacity_penalty_return,
                        "assignment_penalty_nav": assignment_penalty,
                        "total_cost_nav": cost_nav,
                    }
                )
                capacity_rows.append(
                    {
                        "return_date": return_date,
                        "strategy": strategy,
                        "asset_id": asset_id,
                        "estimated_contracts": est_contracts,
                        "capacity_contracts": cap,
                        "capacity_ratio": capacity_ratio,
                        "capacity_status": "pass" if capacity_ratio <= 1.0 else "penalized",
                    }
                )
                if assignment_flag:
                    assignment_rows.append(
                        {
                            "return_date": return_date,
                            "strategy": strategy,
                            "asset_id": asset_id,
                            "assignment_risk_flag": True,
                            "assignment_notional_nav": abs(w),
                            "reason": "deep_itm_low_extrinsic_short_option",
                        }
                    )
            if strategy in net:
                net.loc[return_date, strategy] = gross_returns.loc[return_date, strategy] - gross_cost
            margin_rows.append(
                {
                    "return_date": return_date,
                    "strategy": strategy,
                    "margin_requirement_nav": margin_req,
                    "stress_margin_nav": stress_margin,
                    "assignment_notional_nav": assignment_notional / config.nav_for_capacity if config.nav_for_capacity else np.nan,
                    "margin_model": "conservative_research_simulation",
                }
            )
    assignment = pd.DataFrame(assignment_rows)
    if assignment.empty:
        assignment = pd.DataFrame([{"return_date": pd.NaT, "strategy": "ALL", "asset_id": "NONE", "assignment_risk_flag": False, "assignment_notional_nav": 0.0, "reason": "no_assignment_risk_flags"}])
    return net, pd.DataFrame(rows), pd.DataFrame(capacity_rows), pd.DataFrame(margin_rows), assignment


def cost_diagnostics_table(cost_ledger: pd.DataFrame, capacity_ledger: pd.DataFrame, margin_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not cost_ledger.empty:
        grouped = cost_ledger.groupby("strategy", observed=True)
        for strategy, grp in grouped:
            rows.append(
                {
                    "Strategy": strategy,
                    "Mean monthly cost": float(grp.groupby("return_date")["total_cost_nav"].sum().mean()),
                    "Ann. cost drag": float(grp.groupby("return_date")["total_cost_nav"].sum().mean() * 12.0),
                    "Spread share": float(grp["spread_cost_nav"].sum() / grp["total_cost_nav"].sum()) if grp["total_cost_nav"].sum() > 0 else np.nan,
                    "Fee share": float(grp["fee_cost_nav"].sum() / grp["total_cost_nav"].sum()) if grp["total_cost_nav"].sum() > 0 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and not margin_ledger.empty:
        margin = margin_ledger.groupby("strategy", observed=True)[["margin_requirement_nav", "stress_margin_nav", "assignment_notional_nav"]].mean().reset_index()
        out = out.merge(margin, left_on="Strategy", right_on="strategy", how="left").drop(columns=["strategy"])
    if not out.empty and not capacity_ledger.empty:
        cap = capacity_ledger.groupby("strategy", observed=True)["capacity_status"].apply(lambda s: float((s.astype(str) == "penalized").mean())).rename("Capacity penalized share").reset_index()
        out = out.merge(cap, left_on="Strategy", right_on="strategy", how="left").drop(columns=["strategy"])
    return out


def artifact_hash_manifest(paths: list[Path], base: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(set(p for p in paths if p.exists() and p.is_file())):
        rows.append({"path": str(path.relative_to(base)), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.DataFrame(rows)


def write_environment_lock(path: Path) -> dict[str, object]:
    lock = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in ["numpy", "pandas", "scipy", "matplotlib", "cvxpy"]:
        try:
            mod = __import__(name)
            lock["packages"][name] = getattr(mod, "__version__", "unknown")
        except Exception:
            lock["packages"][name] = "not installed"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


__all__ = [
    "ResearchCostConfig",
    "artifact_hash_manifest",
    "build_cost_input_ledger",
    "compute_strategy_cost_ledgers",
    "cost_diagnostics_table",
    "load_borrow_proxy",
    "write_environment_lock",
]
