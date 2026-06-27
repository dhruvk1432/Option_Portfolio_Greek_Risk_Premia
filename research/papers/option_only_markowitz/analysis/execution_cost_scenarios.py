"""Executable-cost scenarios for the option-only Markowitz paper.

The functions in this module are research accounting, not broker execution.
They convert the existing point-in-time option holding ledgers into three
explicit fill assumptions: midpoint-plus-fees, half-spread, and full-spread.
Every row records quote-quality and capacity gates so headline net results can
fail closed instead of silently assuming fills in untradeable contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.portfolio.option_only_markowitz_model import performance_stats


SCENARIOS = ("mid", "half_spread", "full_spread")


@dataclass(frozen=True)
class ExecutionCostScenarioConfig:
    nav_for_capacity: float = 1_000_000.0
    option_multiplier: float = 100.0
    broker_fee_per_contract: float = 0.65
    occ_fee_per_contract: float = 0.02
    exchange_fee_per_contract: float = 0.05
    regulatory_fee_per_contract: float = 0.002
    min_tick_under_3: float = 0.01
    min_tick_over_3: float = 0.05
    max_relative_spread: float = 0.50
    max_volume_participation: float = 0.10
    max_oi_participation: float = 0.02
    missing_liquidity_is_reject: bool = False
    stress_margin_rate: float = 0.25
    short_option_margin_floor: float = 0.15

    @property
    def fee_per_contract_per_side(self) -> float:
        return (
            self.broker_fee_per_contract
            + self.occ_fee_per_contract
            + self.exchange_fee_per_contract
            + self.regulatory_fee_per_contract
        )


def _scenario_spread_fraction(scenario: str) -> float:
    if scenario == "mid":
        return 0.0
    if scenario == "half_spread":
        return 0.5
    if scenario == "full_spread":
        return 1.0
    raise ValueError(f"unknown execution-cost scenario {scenario!r}")


def _tick_size(mark: float, config: ExecutionCostScenarioConfig) -> float:
    return config.min_tick_under_3 if float(mark) < 3.0 else config.min_tick_over_3


def _assignment_or_dividend_risk(row: pd.Series, is_short: bool) -> tuple[bool, str]:
    if not is_short:
        return False, ""
    if str(row.get("asset_class", "equity_option")) != "equity_option":
        return False, ""
    spot = float(row.get("start_spot", np.nan))
    strike = float(row.get("strike", np.nan))
    mark = float(row.get("mark", np.nan))
    if not np.isfinite([spot, strike, mark]).all() or mark <= 0:
        return False, "missing_assignment_inputs"
    kind = str(row.get("kind", row.get("right", ""))).lower()
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    extrinsic = mark - intrinsic
    deep_itm = (kind == "call" and spot > 1.03 * strike) or (kind == "put" and strike > 1.03 * spot)
    low_extrinsic = extrinsic <= max(0.10 * mark, 0.05)
    if kind == "call":
        dividend = float(row.get("dividend_amount", 0.0) or 0.0)
        days_to_ex = row.get("days_to_ex_dividend", np.nan)
        if np.isfinite(float(days_to_ex)) and 0 <= float(days_to_ex) <= 3 and dividend > max(extrinsic, 0.0):
            return True, "short_call_dividend_exercise_risk"
    if deep_itm and low_extrinsic:
        return True, "deep_itm_low_extrinsic_short_option"
    if kind == "call" and bool(row.get("hard_to_borrow", False)) and intrinsic > 0 and low_extrinsic:
        return True, "hard_to_borrow_short_call"
    return False, ""


def _capital_terms(weight: float, row: pd.Series, config: ExecutionCostScenarioConfig) -> tuple[float, float, float]:
    mark = max(float(row.get("mark", np.nan)), 1e-12)
    spot = float(row.get("start_spot", np.nan))
    premium = abs(float(weight))
    is_short = weight < 0
    if is_short and np.isfinite(spot) and spot > 0:
        margin = abs(weight) * max(config.short_option_margin_floor, 0.20 * spot / mark)
    else:
        margin = premium
    stress = abs(weight) * config.stress_margin_rate
    required = max(premium, margin, stress, 1e-12)
    return premium, margin, stress if np.isfinite(stress) else premium


def build_execution_cost_scenarios(
    gross_returns: pd.DataFrame,
    strategies: Mapping[str, pd.Series],
    cost_inputs: pd.DataFrame,
    *,
    config: ExecutionCostScenarioConfig = ExecutionCostScenarioConfig(),
    scenarios: tuple[str, ...] = SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return scenario net returns and execution/accounting ledgers."""

    if cost_inputs.empty:
        empty_long = pd.DataFrame()
        return (
            pd.DataFrame(index=gross_returns.index),
            empty_long,
            empty_long,
            empty_long,
            empty_long,
            empty_long,
        )
    indexed = cost_inputs.copy()
    indexed["return_date"] = pd.to_datetime(indexed["return_date"], errors="coerce").dt.normalize()
    indexed = indexed.set_index(["return_date", "asset_id"], drop=False)

    net = pd.DataFrame(index=gross_returns.index)
    cost_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    capital_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    hurdle_rows: list[dict[str, object]] = []

    for scenario in scenarios:
        spread_fraction = _scenario_spread_fraction(scenario)
        for strategy, weights in strategies.items():
            if strategy not in gross_returns:
                continue
            out_col = f"{strategy}::{scenario}"
            net[out_col] = gross_returns[strategy]
            for return_date in gross_returns.index:
                total_cost = 0.0
                premium_capital = 0.0
                margin_capital = 0.0
                stress_capital = 0.0
                rejected_for_date = 0
                for asset_id, weight in weights.items():
                    weight = float(weight)
                    if abs(weight) <= 1e-14:
                        continue
                    key = (pd.Timestamp(return_date).normalize(), asset_id)
                    if key not in indexed.index:
                        rejected_rows.append(
                            {
                                "return_date": return_date,
                                "strategy": strategy,
                                "scenario": scenario,
                                "asset_id": asset_id,
                                "reject_reason": "missing_cost_input",
                            }
                        )
                        rejected_for_date += 1
                        continue
                    row = indexed.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    mark = float(row.get("mark", np.nan))
                    rel_spread = float(row.get("relative_spread", np.nan))
                    volume = float(row.get("available_volume_contracts", np.nan))
                    oi = float(row.get("available_oi_contracts", np.nan))
                    reasons = []
                    if not np.isfinite(mark) or mark <= 0:
                        reasons.append("non_positive_mark")
                    if not np.isfinite(rel_spread) or rel_spread < 0:
                        reasons.append("missing_spread")
                    elif rel_spread > config.max_relative_spread:
                        reasons.append("spread_too_wide")
                    if config.missing_liquidity_is_reject and (not np.isfinite(volume) or volume <= 0):
                        reasons.append("missing_volume")
                    if config.missing_liquidity_is_reject and (not np.isfinite(oi) or oi <= 0):
                        reasons.append("missing_open_interest")
                    if reasons:
                        for reason in reasons:
                            rejected_rows.append(
                                {
                                    "return_date": return_date,
                                    "strategy": strategy,
                                    "scenario": scenario,
                                    "asset_id": asset_id,
                                    "reject_reason": reason,
                                }
                            )
                        rejected_for_date += 1
                        continue

                    tick_cost = min(_tick_size(mark, config) / max(mark, 1e-12), 1.0)
                    fee_cost = 2.0 * config.fee_per_contract_per_side / (mark * config.option_multiplier)
                    spread_cost = spread_fraction * rel_spread
                    estimated_contracts = abs(weight) * config.nav_for_capacity / (mark * config.option_multiplier)
                    volume_cap = volume * config.max_volume_participation if np.isfinite(volume) and volume > 0 else np.inf
                    oi_cap = oi * config.max_oi_participation if np.isfinite(oi) and oi > 0 else np.inf
                    capacity_contracts = min(volume_cap, oi_cap)
                    capacity_ratio = estimated_contracts / capacity_contracts if np.isfinite(capacity_contracts) and capacity_contracts > 0 else 0.0
                    if np.isfinite(capacity_ratio) and capacity_ratio > 1.0:
                        rejected_rows.append(
                            {
                                "return_date": return_date,
                                "strategy": strategy,
                                "scenario": scenario,
                                "asset_id": asset_id,
                                "reject_reason": "capacity_exceeded_no_fill",
                            }
                        )
                        rejected_for_date += 1
                        continue
                    prem, margin, stress = _capital_terms(weight, row, config)
                    premium_capital += prem
                    margin_capital += margin
                    stress_capital += stress
                    assignment_flag, assignment_reason = _assignment_or_dividend_risk(row, weight < 0)
                    if assignment_flag:
                        rejected_rows.append(
                            {
                                "return_date": return_date,
                                "strategy": strategy,
                                "scenario": scenario,
                                "asset_id": asset_id,
                                "reject_reason": assignment_reason,
                            }
                        )
                        assignment_rows.append(
                            {
                                "return_date": return_date,
                                "strategy": strategy,
                                "scenario": scenario,
                                "asset_id": asset_id,
                                "assignment_risk_flag": True,
                                "reason": assignment_reason,
                                "blocked": True,
                            }
                        )
                        rejected_for_date += 1
                        continue
                    unit_cost = fee_cost + spread_cost + tick_cost
                    cost_nav = abs(weight) * unit_cost
                    total_cost += cost_nav
                    cost_rows.append(
                        {
                            "return_date": return_date,
                            "strategy": strategy,
                            "scenario": scenario,
                            "asset_id": asset_id,
                            "weight": weight,
                            "mark": mark,
                            "relative_spread": rel_spread,
                            "fee_cost_nav": abs(weight) * fee_cost,
                            "spread_cost_nav": abs(weight) * spread_cost,
                            "tick_rounding_cost_nav": abs(weight) * tick_cost,
                            "total_cost_nav": cost_nav,
                            "estimated_contracts": estimated_contracts,
                            "capacity_contracts": capacity_contracts,
                            "capacity_ratio": capacity_ratio,
                            "fill_status": "filled_research_proxy",
                        }
                    )
                net.loc[return_date, out_col] = gross_returns.loc[return_date, strategy] - total_cost
                required_capital = max(premium_capital, margin_capital, stress_capital, 1e-12)
                capital_rows.append(
                    {
                        "return_date": return_date,
                        "strategy": strategy,
                        "scenario": scenario,
                        "premium_paid_nav": premium_capital,
                        "simulated_margin_nav": margin_capital,
                        "stress_capital_nav": stress_capital,
                        "required_capital_nav": required_capital,
                        "nav_return": net.loc[return_date, out_col],
                        "required_capital_return": net.loc[return_date, out_col] / required_capital,
                        "rejected_trades": rejected_for_date,
                    }
                )

    cost_ledger = pd.DataFrame(cost_rows)
    rejected = pd.DataFrame(rejected_rows)
    capital = pd.DataFrame(capital_rows)
    assignment = pd.DataFrame(assignment_rows)
    if assignment.empty:
        assignment = pd.DataFrame(
            [
                {
                    "return_date": pd.NaT,
                    "strategy": "ALL",
                    "scenario": "ALL",
                    "asset_id": "NONE",
                    "assignment_risk_flag": False,
                    "reason": "no_assignment_or_dividend_blocks",
                    "blocked": False,
                }
            ]
        )
    required_capital_returns = (
        capital.pivot_table(index="return_date", columns=["strategy", "scenario"], values="required_capital_return", aggfunc="first")
        if not capital.empty
        else pd.DataFrame()
    )
    if not required_capital_returns.empty:
        required_capital_returns.columns = [f"{a}::{b}" for a, b in required_capital_returns.columns]
        required_capital_returns = required_capital_returns.sort_index()
    return net, cost_ledger, rejected, capital, assignment, required_capital_returns


def apply_trade_hurdles(
    expected_returns: pd.Series,
    risk_estimates: pd.Series,
    expected_costs: pd.Series,
    *,
    hurdle_levels: tuple[float, ...] = (0.0, 0.10, 0.25),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    no_trade = []
    idx = expected_returns.index
    risk = risk_estimates.reindex(idx).fillna(0.0)
    costs = expected_costs.reindex(idx).fillna(0.0)
    for h in hurdle_levels:
        required = costs + h * risk
        passed = expected_returns.reindex(idx).fillna(0.0) > required
        for asset_id in idx:
            rows.append(
                {
                    "hurdle": h,
                    "asset_id": asset_id,
                    "expected_return": float(expected_returns.get(asset_id, 0.0)),
                    "expected_cost": float(costs.get(asset_id, 0.0)),
                    "risk_estimate": float(risk.get(asset_id, 0.0)),
                    "passed": bool(passed.get(asset_id, False)),
                }
            )
        if not bool(passed.any()):
            no_trade.append({"hurdle": h, "reason": "no_contract_passed_expected_return_hurdle"})
    return pd.DataFrame(rows), pd.DataFrame(no_trade)


def liquidity_tier_labels(cost_inputs: pd.DataFrame) -> pd.DataFrame:
    if cost_inputs.empty:
        return pd.DataFrame(columns=["asset_id", "liquidity_tier"])
    last = cost_inputs.sort_values("return_date").groupby("asset_id", as_index=False).tail(1).copy()
    volume = pd.to_numeric(last.get("available_volume_contracts", np.nan), errors="coerce")
    oi = pd.to_numeric(last.get("available_oi_contracts", np.nan), errors="coerce")
    spread = pd.to_numeric(last.get("relative_spread", np.nan), errors="coerce")
    rows = [{"asset_id": aid, "liquidity_tier": "all_eligible"} for aid in last["asset_id"]]
    vol_cut = volume.quantile(0.75) if volume.notna().any() else np.nan
    oi_cut = oi.quantile(0.75) if oi.notna().any() else np.nan
    spread_cut = spread.quantile(0.25) if spread.notna().any() else np.nan
    for aid, vol, oiv, spr in zip(last["asset_id"], volume, oi, spread):
        if np.isfinite(vol_cut) and vol >= vol_cut:
            rows.append({"asset_id": aid, "liquidity_tier": "top_volume_quartile"})
        if np.isfinite(spread_cut) and spr <= spread_cut:
            rows.append({"asset_id": aid, "liquidity_tier": "tight_spread_quartile"})
        if np.isfinite(oi_cut) and oiv >= oi_cut:
            rows.append({"asset_id": aid, "liquidity_tier": "high_open_interest_quartile"})
        if (
            np.isfinite(vol_cut)
            and np.isfinite(spread_cut)
            and np.isfinite(oi_cut)
            and vol >= vol_cut
            and spr <= spread_cut
            and oiv >= oi_cut
        ):
            rows.append({"asset_id": aid, "liquidity_tier": "combined_liquid"})
    return pd.DataFrame(rows).drop_duplicates()


def liquidity_tier_performance(
    gross_returns: pd.DataFrame,
    strategies: Mapping[str, pd.Series],
    tier_map: pd.DataFrame,
    *,
    periods_per_year: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diag_rows = []
    if tier_map.empty:
        return pd.DataFrame(), pd.DataFrame()
    for tier, assets in tier_map.groupby("liquidity_tier")["asset_id"]:
        asset_set = set(assets)
        for strategy, weights in strategies.items():
            cols = [c for c in gross_returns.columns if c in asset_set]
            w = weights.reindex(cols).fillna(0.0)
            if len(cols) == 0 or w.abs().sum() <= 1e-14:
                pr = pd.Series(0.0, index=gross_returns.index)
                active_assets = 0
            else:
                w = w / w.abs().sum() * min(float(weights.abs().sum()), 1.0)
                pr = gross_returns.reindex(columns=cols).fillna(0.0).to_numpy(float) @ w.to_numpy(float)
                pr = pd.Series(pr, index=gross_returns.index)
                active_assets = int((w.abs() > 0).sum())
            st = performance_stats(pr, periods_per_year)
            rows.append(
                {
                    "Liquidity tier": tier,
                    "Strategy": strategy,
                    "Ann. return": st["ann_return"],
                    "Ann. vol": st["ann_vol"],
                    "Sharpe": st["sharpe"],
                    "Calmar": st["calmar"],
                    "Omega": st["omega"],
                }
            )
            diag_rows.append(
                {
                    "Liquidity tier": tier,
                    "Strategy": strategy,
                    "Eligible assets": len(asset_set),
                    "Active assets": active_assets,
                    "Gross NAV used": float(w.abs().sum()) if len(cols) else 0.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def forecast_ablation_tables(
    premia_components: pd.DataFrame,
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    periods_per_year: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if premia_components is None or premia_components.empty:
        return pd.DataFrame(), pd.DataFrame()
    component_map = {
        "carry_only": ["theta_carry"],
        "variance_risk_premium_only": ["variance_risk_premium"],
        "skew_tail_only": ["skew_tail_premium"],
        "vix_regime_only": ["vol_premium"],
        "relative_value_only": ["relative_value"],
        "full_conditional_model": ["shrunk_mu"],
    }
    rows = []
    comp_rows = []
    base_abs = weights.abs().sum()
    for label, cols in component_map.items():
        signal = premia_components.reindex(weights.index)[cols].sum(axis=1).fillna(0.0)
        if signal.abs().sum() <= 1e-14:
            w = pd.Series(0.0, index=weights.index)
        else:
            w = signal / signal.abs().sum() * min(max(base_abs, 1e-12), 1.0)
        pr = pd.Series(returns.reindex(columns=w.index).fillna(0.0).to_numpy(float) @ w.to_numpy(float), index=returns.index)
        st = performance_stats(pr, periods_per_year)
        rows.append(
            {
                "Ablation": label,
                "Ann. return": st["ann_return"],
                "Ann. vol": st["ann_vol"],
                "Sharpe": st["sharpe"],
                "Calmar": st["calmar"],
                "Omega": st["omega"],
                "Active assets": int((w.abs() > 1e-14).sum()),
            }
        )
        for c in cols:
            comp_rows.append({"Ablation": label, "Component": c, "Mean signal": float(premia_components[c].mean())})
    return pd.DataFrame(rows), pd.DataFrame(comp_rows)


def capacity_market_impact_diagnostics(
    cost_ledger: pd.DataFrame,
    rejected_ledger: pd.DataFrame,
    capital_ledger: pd.DataFrame,
) -> pd.DataFrame:
    if cost_ledger.empty:
        return pd.DataFrame()
    rows = []
    grouped = cost_ledger.groupby(["strategy", "scenario"], observed=True)
    reject_group = rejected_ledger.groupby(["strategy", "scenario"], observed=True).size() if not rejected_ledger.empty else pd.Series(dtype=float)
    capital_group = capital_ledger.groupby(["strategy", "scenario"], observed=True) if not capital_ledger.empty else None
    for (strategy, scenario), grp in grouped:
        monthly_cost = grp.groupby("return_date")["total_cost_nav"].sum()
        avg_contracts = float(grp["estimated_contracts"].mean())
        capacity_used = float(grp["capacity_ratio"].replace([np.inf, -np.inf], np.nan).max())
        cap = capital_group.get_group((strategy, scenario)) if capital_group is not None and (strategy, scenario) in capital_group.groups else pd.DataFrame()
        rows.append(
            {
                "Strategy": strategy,
                "Scenario": scenario,
                "Avg contracts traded": avg_contracts,
                "Avg quoted spread": float(grp["relative_spread"].mean()),
                "Avg monthly cost": float(monthly_cost.mean()),
                "Max capacity used": capacity_used,
                "Rejected trades": int(reject_group.get((strategy, scenario), 0)),
                "Mean margin/NAV": float(cap["simulated_margin_nav"].mean()) if not cap.empty else np.nan,
                "Mean stress capital/NAV": float(cap["stress_capital_nav"].mean()) if not cap.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def post_cost_survival_table(
    gross_performance: pd.DataFrame,
    net_scenario_returns: pd.DataFrame,
    scenario_cost_ledger: pd.DataFrame,
    capacity_diag: pd.DataFrame,
    *,
    scenario: str = "full_spread",
    periods_per_year: float = 12.0,
) -> pd.DataFrame:
    rows = []
    gross_map = gross_performance.set_index("Strategy") if not gross_performance.empty else pd.DataFrame()
    for col in net_scenario_returns.columns:
        if not col.endswith(f"::{scenario}"):
            continue
        strategy = col.rsplit("::", 1)[0]
        r = net_scenario_returns[col].dropna()
        st = performance_stats(r, periods_per_year)
        costs = scenario_cost_ledger[
            scenario_cost_ledger["strategy"].eq(strategy) & scenario_cost_ledger["scenario"].eq(scenario)
        ]
        diag = capacity_diag[
            capacity_diag["Strategy"].eq(strategy) & capacity_diag["Scenario"].eq(scenario)
        ]
        avg_spread = float(costs["spread_cost_nav"].sum() / max(len(r), 1)) if not costs.empty else np.nan
        capacity_used = float(diag["Max capacity used"].iloc[0]) if not diag.empty else np.nan
        gross_sr = float(gross_map.loc[strategy, "Sharpe"]) if strategy in gross_map.index else np.nan
        survives = (
            np.isfinite(st["sharpe"])
            and st["sharpe"] > 0
            and np.isfinite(st["calmar"])
            and st["calmar"] > 0
            and (not np.isfinite(capacity_used) or capacity_used <= 1.0)
        )
        rows.append(
            {
                "Strategy": strategy,
                "Gross Sharpe": gross_sr,
                "Net Sharpe": st["sharpe"],
                "Net Calmar": st["calmar"],
                "Avg spread cost": avg_spread,
                "Capacity used": capacity_used,
                "Survives?": "yes" if survives else "no",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ExecutionCostScenarioConfig",
    "SCENARIOS",
    "apply_trade_hurdles",
    "build_execution_cost_scenarios",
    "capacity_market_impact_diagnostics",
    "forecast_ablation_tables",
    "liquidity_tier_labels",
    "liquidity_tier_performance",
    "post_cost_survival_table",
]
