"""R1.1 higher-risk development pipeline with an executable VIX overlay.

R1.1 is versioned separately from R1 so the existing R1 source hashes and
freeze manifest are not rewritten.  All output remains retrospective
development evidence.  The VIX arm is deliberately left unscored when the
required licensed event-date CBBO inputs are absent.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import build_configs
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import compute_liquidity_caps
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import _build_config_panel
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import (
    PAPER,
    R1_NAME,
    build_optimization_cost_spec,
    build_r1_model,
    paired_block_bootstrap_comparison,
    survival_diagnostics,
)
from research.papers.option_only_markowitz.analysis.r11_integer_repair import integerize_r11_weights
from research.papers.option_only_markowitz.analysis.run_empirics import ROOT, factor_panels, representative_specs
from research.papers.option_only_markowitz.analysis.vix_option_panel import (
    front_vx_price_series,
    vix_close_series,
)
from src.portfolio.option_only_markowitz_model import (
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionMarkowitzResult,
    OptionOnlyMarkowitzModel,
)
from src.portfolio.r11_risk_controls import (
    EgarchOverlayConfig,
    RiskOffConfig,
    apply_egarch_joint_overlay,
    build_vix_risk_off_events,
    egarch_variance_forecast,
    evaluate_egarch_gate,
    execute_cbbo_orders,
    qlike_loss,
    risk_off_exposure_calendar,
)


DEFAULT_OUT = PAPER / "analysis" / "artifacts" / "r11_higher_risk"
EVENT_QUOTE_DIR = ROOT / "data" / "databento_cache" / "r11_event_cbbo"
R11_NAME = "R1.1 25pct positive-edge deployment"
R11_RISK_OFF_NAME = "R1.1 25pct VIX40 risk-off"
R11_EGARCH_NAME = "R1.1 25pct EGARCH diagnostic"


@dataclass(frozen=True)
class R11NetUtilityConfig(NetUtilityConfig):
    """R1.1 risk policy; R1's 15% default remains untouched."""

    annual_vol_target: float = 0.25
    deployment_target: float = 0.50
    deployment_net_edge_floor: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not 0 <= self.deployment_target <= 1:
            raise ValueError("deployment_target must lie in [0, 1]")
        if self.deployment_net_edge_floor < 0:
            raise ValueError("deployment_net_edge_floor must be nonnegative")


def _r11_result_stats(
    model: OptionOnlyMarkowitzModel,
    weights: np.ndarray,
    scenarios: np.ndarray,
    costs: OptimizationCostSpec,
    config: R11NetUtilityConfig,
    selected_lambda: float,
    stage1_gross: float,
    deployment_feasible: bool,
    deployment_applied: bool,
    eligible_count: int,
    solver: str,
) -> OptionMarkowitzResult:
    base = model._make_result(weights, "optimal", solver)
    long_cost, short_cost, short_margin, _ = costs.aligned(model.contracts)
    long, short = np.maximum(weights, 0.0), np.maximum(-weights, 0.0)
    cost = float(long_cost @ long + short_cost @ short)
    net = scenarios @ weights - cost
    losses = -net
    threshold = float(np.quantile(losses, config.cvar_alpha, method="higher"))
    tail = losses[losses >= threshold - 1e-12]
    stress = model._stress_matrix()
    stats = {
        "objective": "r11_net_mean_variance_with_positive_edge_deployment",
        "risk_aversion": float(selected_lambda),
        "gross_mean": float(model.expected_returns.to_numpy(float) @ weights),
        "predictable_cost": cost,
        "net_mean": float(model.expected_returns.to_numpy(float) @ weights) - cost,
        "variance_penalty": 0.5 * selected_lambda * float(weights @ model.option_cov @ weights),
        "predicted_annual_vol": float(base.volatility * np.sqrt(config.periods_per_year)),
        "scenario_cvar_loss": float(tail.mean()) if len(tail) else threshold,
        "worst_stress_return": float(np.min(stress @ weights)) if stress is not None else np.nan,
        "short_margin_used": float(short_margin @ short),
        "collateral_used": float(long.sum() + short_margin @ short),
        "cash_weight": max(0.0, 1.0 - float(long.sum() + short_margin @ short)),
        "n_scenarios": int(len(scenarios)),
        "stage1_gross_nav": float(stage1_gross),
        "deployment_target": float(config.deployment_target),
        "deployment_target_feasible": bool(deployment_feasible),
        "deployment_target_applied": bool(deployment_applied),
        "deployment_target_met": bool(np.abs(weights).sum() >= config.deployment_target - 1e-6),
        "deployment_shortfall": float(max(config.deployment_target - np.abs(weights).sum(), 0.0)),
        "positive_edge_contracts": int(eligible_count),
    }
    return replace(base, objective_stats=stats)


def solve_r11_net_utility(
    model: OptionOnlyMarkowitzModel,
    scenario_returns: pd.DataFrame,
    costs: OptimizationCostSpec,
    config: R11NetUtilityConfig = R11NetUtilityConfig(),
    *,
    per_contract_caps: pd.Series | None = None,
) -> OptionMarkowitzResult:
    """Solve R1.1 and conditionally apply its sign-restricted 50% target."""

    config.validate()
    stage1 = model.solve_net_utility(
        scenario_returns,
        costs,
        config,
        per_contract_caps=per_contract_caps,
    )
    selected_lambda = float(stage1.objective_stats["risk_aversion"])
    stage1_weights = stage1.weights.reindex(model.contracts).to_numpy(float)
    stage1_gross = float(np.abs(stage1_weights).sum())
    scenarios = scenario_returns.reindex(columns=model.contracts).dropna(how="all").fillna(0.0).to_numpy(float)
    if stage1_gross >= config.deployment_target - 1e-7 or config.deployment_target <= 0:
        return _r11_result_stats(
            model,
            stage1_weights,
            scenarios,
            costs,
            config,
            selected_lambda,
            stage1_gross,
            True,
            False,
            int(np.count_nonzero(stage1_weights)),
            f"{stage1.solver}_r11_stage1",
        )

    long_cost, short_cost, short_margin, short_allowed = costs.aligned(model.contracts)
    mu = model.expected_returns.to_numpy(float)
    signs = np.sign(stage1_weights)
    directional_cost = np.where(signs > 0, long_cost, np.where(signs < 0, short_cost, 0.0))
    directional_mean = signs * mu
    directional_edge = directional_mean - directional_cost
    eligible = (signs != 0) & (directional_edge > config.deployment_net_edge_floor + 1e-12)
    if not eligible.any():
        return _r11_result_stats(
            model,
            stage1_weights,
            scenarios,
            costs,
            config,
            selected_lambda,
            stage1_gross,
            False,
            False,
            0,
            f"{stage1.solver}_r11_no_positive_edge",
        )

    n = len(model.contracts)
    x = cp.Variable(n, nonneg=True)
    w = cp.multiply(signs, x)
    long = cp.multiply((signs > 0).astype(float), x)
    short = cp.multiply((signs < 0).astype(float), x)
    predictable_cost = directional_cost @ x
    eta = cp.Variable()
    constraints = model._cvxpy_net_utility_constraints(
        w,
        long,
        short,
        scenarios,
        predictable_cost,
        eta,
        config,
        short_margin,
        short_allowed,
        per_contract_caps,
    )
    constraints.append(x[~eligible] == 0)
    constraints.append(directional_edge @ x >= 0)
    monthly_variance_cap = (config.annual_vol_target / np.sqrt(config.periods_per_year)) ** 2
    constraints.append(cp.quad_form(w, cp.psd_wrap(model.option_cov)) <= monthly_variance_cap)
    constraints.append(cp.sum(x) <= config.deployment_target)
    feasibility_problem = cp.Problem(cp.Maximize(cp.sum(x)), constraints)
    feasibility_solver, _ = model._cvxpy_run_solvers(feasibility_problem, x)
    feasible_gross = float(np.sum(x.value)) if feasibility_solver is not None and x.value is not None else 0.0
    target_feasible = bool(feasible_gross >= config.deployment_target - 1e-6)
    if not target_feasible:
        return _r11_result_stats(
            model,
            stage1_weights,
            scenarios,
            costs,
            config,
            selected_lambda,
            stage1_gross,
            False,
            False,
            int(eligible.sum()),
            f"{stage1.solver}_r11_target_infeasible",
        )

    target_constraints = list(constraints)
    target_constraints.append(cp.sum(x) == config.deployment_target)
    objective = cp.Maximize(
        directional_mean @ x
        - directional_cost @ x
        - 0.5 * selected_lambda * cp.quad_form(w, cp.psd_wrap(model.option_cov))
    )
    target_problem = cp.Problem(objective, target_constraints)
    target_solver, _ = model._cvxpy_run_solvers(target_problem, x)
    if target_solver is None or x.value is None:
        return _r11_result_stats(
            model,
            stage1_weights,
            scenarios,
            costs,
            config,
            selected_lambda,
            stage1_gross,
            True,
            False,
            int(eligible.sum()),
            f"{stage1.solver}_r11_target_solve_failed",
        )
    target_weights = signs * np.asarray(x.value, dtype=float).ravel()
    target_weights[np.abs(target_weights) < 1e-9] = 0.0
    return _r11_result_stats(
        model,
        target_weights,
        scenarios,
        costs,
        config,
        selected_lambda,
        stage1_gross,
        True,
        True,
        int(eligible.sum()),
        f"cvxpy_{target_solver.lower()}_r11_positive_edge_target",
    )


def load_daily_return_panel(root: Path = ROOT) -> pd.DataFrame:
    """Daily equity and VX-forward returns used only with prior-date cutoffs."""

    raw = pd.read_csv(root / "data/universe/multi_raw_close.csv", index_col=0, parse_dates=True)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    prices = raw.apply(pd.to_numeric, errors="coerce")
    vx = front_vx_price_series(root)
    if len(vx):
        vx.index = pd.DatetimeIndex(vx.index).tz_localize(None).normalize()
        prices["VX_FRONT"] = vx.reindex(prices.index).ffill()
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def _egarch_inputs_for_decision(
    daily_returns: pd.DataFrame,
    underlyings: Sequence[str],
    decision_date: pd.Timestamp,
    config: EgarchOverlayConfig,
    cache: dict[tuple[str, pd.Timestamp], dict[str, object]],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    ratios: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for underlying in underlyings:
        key = (str(underlying), pd.Timestamp(decision_date).normalize())
        if key not in cache:
            series = daily_returns.get(str(underlying), pd.Series(dtype=float))
            cache[key] = egarch_variance_forecast(series, decision_date, config)
        forecast = dict(cache[key])
        ratios[str(underlying)] = float(forecast["variance_ratio"])
        future = daily_returns.get(str(underlying), pd.Series(dtype=float))
        future = future[pd.DatetimeIndex(future.index) > pd.Timestamp(decision_date)].dropna().head(config.horizon_days)
        forecast["realized_variance"] = float(np.square(future).sum()) if len(future) == config.horizon_days else np.nan
        forecast.update({"underlying": str(underlying), "decision_date": pd.Timestamp(decision_date)})
        rows.append(forecast)
    return ratios, rows


def _overlay_model(model: OptionOnlyMarkowitzModel, ratios: dict[str, float]) -> OptionOnlyMarkowitzModel:
    moments = apply_egarch_joint_overlay(model.joint_moments, ratios)
    overlaid = OptionOnlyMarkowitzModel(
        model.options,
        model.shocks,
        expected_returns=model.expected_returns,
        constraints=model.constraints,
        covariance_shrinkage=0.0,
        joint_moments=moments,
    )
    if hasattr(model, "conditional_premia_components"):
        overlaid.conditional_premia_components = model.conditional_premia_components
    return overlaid


def run_r11_config(
    label: str,
    equity_underlyings: Sequence[str],
    poc_names: Sequence[str],
    with_vix: bool,
    daily_returns: pd.DataFrame,
    egarch_cache: dict[tuple[str, pd.Timestamp], dict[str, object]],
    *,
    nav: float = 1_000_000.0,
    evaluation_start: str = "2018-01-01",
    min_train_months: int = 36,
    max_periods: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run base and EGARCH R1.1 arms on one chronological development panel."""

    reps, returns, detail, universe, _ = _build_config_panel(equity_underlyings, poc_names, with_vix)
    underlying_returns, vol_shocks = factor_panels(reps, universe)
    cost_config = ResearchCostConfig(
        nav_for_capacity=nav,
        impact_cost_rate=0.0,
        use_current_spread_assumptions=False,
        use_inferred_spread_proxy=True,
    )
    surface = load_cbbo_spread_surface(ROOT, cost_config.cbbo_spread_surface_path)
    cost_inputs = build_cost_input_ledger(reps, detail, ROOT, cost_config, spread_surface=surface)
    risk_config = R11NetUtilityConfig()
    egarch_config = EgarchOverlayConfig()
    dates = [date for date in pd.DatetimeIndex(returns.index).sort_values() if date >= pd.Timestamp(evaluation_start)]
    if max_periods is not None:
        dates = dates[:max_periods]
    return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    repair_rows: list[pd.DataFrame] = []
    for period_number, return_date in enumerate(dates, start=1):
        if period_number == 1 or period_number % 10 == 0 or period_number == len(dates):
            print(f"[R1.1] {label}: decision {period_number}/{len(dates)} for {return_date.date()}", flush=True)
        prior_dates = pd.DatetimeIndex(returns.index[returns.index < return_date]).sort_values()
        if len(prior_dates) < min_train_months:
            continue
        train_dates = prior_dates[-min_train_months:]
        train_start, train_end = train_dates[0], train_dates[-1]
        period_detail = detail[pd.to_datetime(detail["return_date"]).eq(return_date)]
        decision_date = pd.Timestamp(period_detail["decision_date"].max()) if not period_detail.empty else pd.Timestamp(train_end)
        period_meta = (
            period_detail.sort_values("decision_date").groupby("asset_id", observed=True).tail(1).set_index("asset_id")
            if not period_detail.empty and "asset_id" in period_detail
            else pd.DataFrame()
        )
        spec = representative_specs(reps, returns, train_start=train_start, train_end=decision_date)
        train_returns = returns.reindex(index=train_dates, columns=spec.index).fillna(0.0)
        keep = list(train_returns.count()[lambda value: value >= min(24, min_train_months)].index)
        spec = spec.reindex(keep).dropna(subset=["underlying", "mark"])
        if spec.empty:
            continue
        train_returns = train_returns.reindex(columns=spec.index).fillna(0.0)
        train_under = underlying_returns.reindex(index=train_dates, columns=universe).fillna(0.0)
        train_vol = vol_shocks.reindex(index=train_dates, columns=universe).fillna(0.0)
        base_model = build_r1_model(spec, train_returns, train_under, train_vol, universe)
        caps_frame = compute_liquidity_caps(
            reps,
            spec["mark"],
            nav,
            participation=0.05,
            per_contract_abs=0.18,
            train_end=decision_date,
        )
        caps = caps_frame["bound"].reindex(base_model.contracts).fillna(0.0)
        optimization_costs, _ = build_optimization_cost_spec(
            cost_inputs,
            base_model.contracts,
            decision_date,
            cost_config,
        )
        ratios, forecast = _egarch_inputs_for_decision(
            daily_returns,
            universe,
            decision_date,
            egarch_config,
            egarch_cache,
        )
        for row in forecast:
            row["config"] = label
        forecast_rows.extend(forecast)
        models = [(R11_NAME, base_model), (R11_EGARCH_NAME, _overlay_model(base_model, ratios))]
        for strategy, model in models:
            result = solve_r11_net_utility(
                model,
                train_returns,
                optimization_costs,
                risk_config,
                per_contract_caps=caps,
            )
            repair = integerize_r11_weights(
                model,
                result.weights,
                spec["mark"],
                nav,
                caps,
                train_returns,
                optimization_costs,
                risk_config,
                risk_aversion=float(result.objective_stats["risk_aversion"]),
            )
            realized_weights, execution = repair.weights, repair.diagnostics
            repair_rows.append(
                repair.candidates.assign(
                    config=label,
                    strategy=strategy,
                    return_date=return_date,
                    decision_date=decision_date,
                )
            )
            realized_return = returns.reindex(index=[return_date], columns=model.contracts).fillna(0.0).iloc[0]
            gross_return = float(realized_return @ realized_weights)
            long_cost, short_cost, _, _ = optimization_costs.aligned(model.contracts)
            w = realized_weights.to_numpy(float)
            predicted_cost = float(long_cost @ np.maximum(w, 0.0) + short_cost @ np.maximum(-w, 0.0))
            stats = result.objective_stats
            return_rows.append(
                {
                    "config": label,
                    "strategy": strategy,
                    "evidence_status": "retrospective_development_sample",
                    "return_date": return_date,
                    "decision_date": decision_date,
                    "train_start": train_start,
                    "train_end": train_end,
                    "march_2020_in_training_window": bool(train_start <= pd.Timestamp("2020-03-31") <= train_end),
                    "march_2020_observation_policy": "retained_if_in_window",
                    "gross_return": gross_return,
                    "predicted_cost": predicted_cost,
                    "net_return": gross_return - predicted_cost,
                    "gross_nav": float(realized_weights.abs().sum()),
                    "predicted_annual_vol": stats["predicted_annual_vol"],
                    "risk_aversion": stats["risk_aversion"],
                    "stage1_gross_nav": stats["stage1_gross_nav"],
                    "deployment_target": stats["deployment_target"],
                    "deployment_target_feasible": stats["deployment_target_feasible"],
                    "deployment_target_applied": stats["deployment_target_applied"],
                    "deployment_target_met_after_integer_repair": bool(realized_weights.abs().sum() >= risk_config.deployment_target - 1e-6),
                    "deployment_shortfall_after_integer_repair": max(risk_config.deployment_target - float(realized_weights.abs().sum()), 0.0),
                    "positive_edge_contracts": stats["positive_edge_contracts"],
                    "scenario_cvar_loss": execution["scenario_cvar_loss"],
                    "short_margin_used": execution["short_margin_used"],
                    "collateral_used": execution["collateral_used"],
                    "integer_repair_failed": bool(execution["integer_repair_failed_to_cash"]),
                    "integer_execution_abstained": bool(execution["integer_execution_abstained"]),
                    "integer_conversion_feasible": bool(execution["integer_conversion_feasible"]),
                    "integer_abstention_reason": execution["integer_abstention_reason"],
                    "integer_repair_method": execution["selected_integer_method"],
                    "pre_repair_feasible": execution["pre_repair_feasible"],
                    "pre_repair_max_breach": execution["pre_repair_max_breach"],
                    "best_failed_method": execution.get("best_failed_method", ""),
                    "failed_max_breach": execution.get("failed_max_breach", 0.0),
                    "failed_breach_base": execution.get("failed_breach_base", 0.0),
                    "failed_breach_cap": execution.get("failed_breach_cap", 0.0),
                    "failed_breach_cvar": execution.get("failed_breach_cvar", 0.0),
                    "failed_breach_stress": execution.get("failed_breach_stress", 0.0),
                    "failed_breach_margin": execution.get("failed_breach_margin", 0.0),
                    "failed_breach_collateral": execution.get("failed_breach_collateral", 0.0),
                    "failed_breach_assignment": execution.get("failed_breach_assignment", 0.0),
                    "failed_breach_volatility": execution.get("failed_breach_volatility", 0.0),
                    "failed_scenario_cvar_loss": execution.get("failed_scenario_cvar_loss", np.nan),
                    "failed_worst_stress_return": execution.get("failed_worst_stress_return", np.nan),
                    "failed_short_margin_used": execution.get("failed_short_margin_used", np.nan),
                    "failed_collateral_used": execution.get("failed_collateral_used", np.nan),
                    "failed_predicted_annual_vol": execution.get("failed_predicted_annual_vol", np.nan),
                    "information_set_valid": bool(train_end < return_date and decision_date < return_date),
                }
            )
            for contract, weight in realized_weights.items():
                mark = float(spec.loc[contract, "mark"])
                contracts = float(np.rint(weight * nav / (100.0 * mark))) if mark > 0 else 0.0
                meta = period_meta.loc[contract] if contract in period_meta.index else spec.loc[contract]
                if isinstance(meta, pd.DataFrame):
                    meta = meta.iloc[-1]
                weight_rows.append(
                    {
                        "config": label,
                        "strategy": strategy,
                        "return_date": return_date,
                        "decision_date": decision_date,
                        "asset_id": contract,
                        "symbol": str(meta.get("symbol", contract)),
                        "underlying": str(meta.get("underlying", spec.loc[contract].get("underlying", ""))),
                        "expiry": meta.get("expiry", pd.NaT),
                        "mark": mark,
                        "weight": float(weight),
                        "integer_contracts": contracts,
                    }
                )
    repairs = pd.concat(repair_rows, ignore_index=True) if repair_rows else pd.DataFrame()
    return pd.DataFrame(return_rows), pd.DataFrame(weight_rows), pd.DataFrame(forecast_rows), repairs


def summarize_strategy_returns(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if returns.empty or not {"config", "strategy"}.issubset(returns.columns):
        return pd.DataFrame()
    for (config, strategy), group in returns.groupby(["config", "strategy"], observed=True):
        risk = R11NetUtilityConfig()
        rows.append(
            {
                "config": config,
                "strategy": strategy,
                "evidence_status": "retrospective_development_sample",
                **survival_diagnostics(
                    group.set_index("return_date")["net_return"],
                    margin_breaches=int((group["short_margin_used"] > risk.short_margin_nav + 1e-6).sum()),
                    collateral_breaches=int((group["collateral_used"] > risk.collateral_nav + 1e-6).sum()),
                    integer_failures=int(group["integer_repair_failed"].sum()),
                ),
                "observations": int(len(group)),
                "integer_abstentions": int(group["integer_execution_abstained"].sum()),
                "mean_gross_nav": float(group["gross_nav"].mean()),
                "deployment_target_hit_rate": float(group["deployment_target_met_after_integer_repair"].mean()),
                "all_information_sets_valid": bool(group["information_set_valid"].all()),
            }
        )
    return pd.DataFrame(rows)


def summarize_integer_repairs(repairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the registered integer candidates without hiding failures."""

    if repairs.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (config, strategy, method), group in repairs.groupby(
        ["config", "strategy", "method"], observed=True
    ):
        rows.append(
            {
                "config": config,
                "strategy": strategy,
                "method": method,
                "candidate_periods": int(len(group)),
                "available_periods": int(group["available"].astype(bool).sum()),
                "feasible_periods": int(
                    (group["available"].astype(bool) & group["feasible"].astype(bool)).sum()
                ),
                "selected_periods": int(group["selected"].astype(bool).sum()),
                "mean_net_utility": float(group["net_utility"].mean()),
                "mean_continuous_l1_distance": float(group["continuous_l1_distance"].mean()),
                "maximum_recorded_breach": float(group["max_breach"].max()),
            }
        )
    return pd.DataFrame(rows)


def reconcile_selected_integer_diagnostics(
    returns: pd.DataFrame, repairs: pd.DataFrame
) -> pd.DataFrame:
    """Make headline risk columns describe the selected integer book, not its precursor."""

    if returns.empty or repairs.empty:
        return returns
    keys = ["config", "strategy", "return_date", "decision_date"]
    selected = repairs[repairs["selected"].astype(bool)].copy()
    selected = selected[
        keys
        + [
            "predicted_annual_vol",
            "scenario_cvar_loss",
            "short_margin_used",
            "collateral_used",
            "gross_nav",
        ]
    ].rename(
        columns={
            column: f"selected_integer_{column}"
            for column in [
                "predicted_annual_vol",
                "scenario_cvar_loss",
                "short_margin_used",
                "collateral_used",
                "gross_nav",
            ]
        }
    )
    out = returns.copy()
    for key in ["return_date", "decision_date"]:
        out[key] = pd.to_datetime(out[key], errors="coerce")
        selected[key] = pd.to_datetime(selected[key], errors="coerce")
    out = out.merge(selected, on=keys, how="left", validate="many_to_one")
    direct_columns = [
        "max_breach",
        "scenario_cvar_loss",
        "worst_stress_return",
        "short_margin_used",
        "collateral_used",
        "predicted_annual_vol",
        "breach_base",
        "breach_cap",
        "breach_cvar",
        "breach_stress",
        "breach_margin",
        "breach_collateral",
        "breach_assignment",
        "breach_volatility",
        "breach_positive_edge",
    ]
    if "method" in repairs and set(direct_columns).issubset(repairs.columns):
        out = out.drop(
            columns=[column for column in out if column.startswith("direct_truncation_")],
            errors="ignore",
        )
        direct = repairs[repairs["method"].eq("truncate_toward_cash")][
            keys + direct_columns
        ].rename(columns={column: f"direct_truncation_{column}" for column in direct_columns})
        for key in ["return_date", "decision_date"]:
            direct[key] = pd.to_datetime(direct[key], errors="coerce")
        out = out.merge(direct, on=keys, how="left", validate="many_to_one")
    if "continuous_predicted_annual_vol" not in out:
        out["continuous_predicted_annual_vol"] = out["predicted_annual_vol"]
    for column in [
        "predicted_annual_vol",
        "scenario_cvar_loss",
        "short_margin_used",
        "collateral_used",
        "gross_nav",
    ]:
        integer_column = f"selected_integer_{column}"
        out[column] = out[integer_column].combine_first(out[column])
        out = out.drop(columns=[integer_column])
    return out


def build_event_quote_request(weights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Identify exact held OSI symbols required to score each state change."""

    if weights.empty or "strategy" not in weights:
        return pd.DataFrame()
    base = weights[weights["strategy"].eq(R11_NAME)].copy()
    if base.empty or events.empty:
        return pd.DataFrame()
    for column in ["decision_date", "return_date", "expiry"]:
        base[column] = pd.to_datetime(base[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        execution_date = pd.Timestamp(event["execution_date"])
        active = base[
            base["decision_date"].le(execution_date)
            & base["return_date"].ge(execution_date)
            & base["expiry"].ge(execution_date)
            & base["integer_contracts"].ne(0.0)
        ]
        if active.empty:
            continue
        latest = active.groupby(["config", "asset_id"], observed=True)["decision_date"].transform("max")
        active = active[active["decision_date"].eq(latest)]
        for _, position in active.iterrows():
            contracts = float(position["integer_contracts"])
            order = -contracts if event["action"] == "exit" else contracts
            rows.append(
                {
                    "config": position["config"],
                    "signal_date": event["signal_date"],
                    "execution_date": execution_date,
                    "action": event["action"],
                    "signal_source": event["source"],
                    "asset_id": position["asset_id"],
                    "symbol": position["symbol"],
                    "underlying": position["underlying"],
                    "expiry": position["expiry"],
                    "position_contracts": contracts,
                    "order_contracts": order,
                }
            )
    return pd.DataFrame(rows)


def score_event_executions(
    requests: pd.DataFrame,
    quote_dir: Path = EVENT_QUOTE_DIR,
    config: RiskOffConfig = RiskOffConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score available event quotes and fail closed when any are absent."""

    ledgers: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    if requests.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (execution_date, config_name, action), group in requests.groupby(
        ["execution_date", "config", "action"], observed=True
    ):
        date = pd.Timestamp(execution_date)
        path = quote_dir / f"opra_cbbo1m_{date.date().isoformat()}.parquet"
        if not path.exists():
            summary = {
                "execution_date": date,
                "config": config_name,
                "action": action,
                "execution_feasible": False,
                "missing_executable_quotes": True,
                "quote_path": str(path.relative_to(ROOT)),
                "status": "unscored_missing_licensed_cbbo",
            }
            summaries.append(summary)
            continue
        quotes = pd.read_parquet(path)
        fills, summary = execute_cbbo_orders(group[["symbol", "order_contracts"]], quotes, date, config)
        fills["config"] = config_name
        fills["action"] = action
        ledgers.append(fills)
        summaries.append({**summary, "config": config_name, "action": action, "quote_path": str(path.relative_to(ROOT)), "status": "scored" if summary["execution_feasible"] else "unscored_incomplete_fill"})
    return (pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(), pd.DataFrame(summaries))


def build_risk_off_status_rows(base_returns: pd.DataFrame, execution_summary: pd.DataFrame) -> pd.DataFrame:
    """Create an explicitly unscored comparison arm until executions are complete."""

    if base_returns.empty or "strategy" not in base_returns:
        return pd.DataFrame()
    out = base_returns[base_returns["strategy"].eq(R11_NAME)].copy()
    out["strategy"] = R11_RISK_OFF_NAME
    if execution_summary.empty:
        out["evidence_status"] = "unscored_no_historical_trigger_positions"
        return out
    feasible = bool(execution_summary["execution_feasible"].all())
    if not feasible:
        out["evidence_status"] = "unscored_missing_or_incomplete_executable_quotes"
        missing_dates = pd.to_datetime(
            execution_summary.loc[~execution_summary["execution_feasible"], "execution_date"],
            errors="coerce",
        ).dropna()
        missing_periods = set(missing_dates.dt.to_period("M"))
        affected = pd.to_datetime(out["return_date"]).dt.to_period("M").isin(missing_periods)
        out.loc[affected, ["gross_return", "predicted_cost", "net_return"]] = np.nan
    else:
        # Exact path recomposition requires both exit and re-entry fills plus
        # current re-entry Greeks.  Until that complete ledger is present the
        # arm is not allowed to inherit the no-rule return.
        out["evidence_status"] = "unscored_pending_reentry_constraint_recheck"
        out[["gross_return", "predicted_cost", "net_return"]] = np.nan
    return out


def build_r11_comparisons(returns: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    r1_path = PAPER / "analysis/artifacts/r1_repaired/r1_monthly_development_returns.csv"
    if r1_path.exists() and not returns.empty and "strategy" in returns:
        r1 = pd.read_csv(r1_path, parse_dates=["return_date"])
        for config, group in returns[returns["strategy"].eq(R11_NAME)].groupby("config", observed=True):
            strategy = group.set_index(pd.to_datetime(group["return_date"]))["net_return"]
            benchmark = r1[r1["config"].eq(config)].set_index("return_date")["net_return"]
            rows.append({"config": config, "comparison": "R1.1_25_vs_R1_15", **paired_block_bootstrap_comparison(strategy, benchmark)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "r11_paired_comparisons.csv", index=False)
    return comparison


def build_r11_trial_registry(out_dir: Path) -> pd.DataFrame:
    """Append the identifiable R1.1 arms to the known lower-bound registry."""

    r1_path = PAPER / "analysis/artifacts/r1_repaired/research_trial_registry.csv"
    base = pd.read_csv(r1_path) if r1_path.exists() else pd.DataFrame(columns=["source", "trial_key", "count_status"])
    additions = pd.DataFrame(
        [
            {"source": "analysis/r11_higher_risk_pipeline.py", "trial_key": "R1.1_25pct_no_risk_off", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_higher_risk_pipeline.py", "trial_key": "R1.1_25pct_VIX40_close_next_open", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_higher_risk_pipeline.py", "trial_key": "R1.1_25pct_EGARCH11_student_t", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_integer_repair.py", "trial_key": "R1.1_integer_truncate_toward_cash", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_integer_repair.py", "trial_key": "R1.1_integer_remove_risk_leg", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_integer_repair.py", "trial_key": "R1.1_integer_retain_protective_VIX", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_integer_repair.py", "trial_key": "R1.1_integer_iterative_reduce", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_integer_repair.py", "trial_key": "R1.1_integer_mixed_integer_conic", "count_status": "known_lower_bound"},
            {"source": "analysis/r11_integer_repair.py", "trial_key": "R1.1_integer_direct_or_cash_abstention", "count_status": "known_lower_bound"},
        ]
    )
    registry = pd.concat([base, additions], ignore_index=True).drop_duplicates(["source", "trial_key"])
    registry.to_csv(out_dir / "r11_research_trial_registry.csv", index=False)
    (out_dir / "r11_research_trial_registry.json").write_text(
        json.dumps(
            {
                "known_trial_count_lower_bound": int(len(registry)),
                "is_complete": False,
                "reason": "R1.1 arms are development trials and earlier undocumented iterations remain unreconstructable.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return registry


def build_specification_status(
    summary: pd.DataFrame,
    execution_summary: pd.DataFrame,
    egarch_gate: dict[str, object],
    out_dir: Path,
) -> pd.DataFrame:
    """Create the prominent R1/E1/R1.1 evidence-status comparison."""

    rows: list[dict[str, object]] = []
    r1_path = PAPER / "analysis/artifacts/r1_repaired/r1_survival_summary.csv"
    if r1_path.exists():
        r1 = pd.read_csv(r1_path)
        for _, row in r1.iterrows():
            rows.append(
                {
                    "specification": "R1 15pct",
                    "config": row.get("config"),
                    "status": row.get("verdict"),
                    "evidence": "retrospective_development_sample",
                    "terminal_wealth": row.get("terminal_wealth"),
                    "mean_gross_nav": row.get("mean_gross_nav"),
                }
            )
    for _, row in summary.iterrows():
        rows.append(
            {
                "specification": row.get("strategy"),
                "config": row.get("config"),
                "status": row.get("verdict"),
                "evidence": row.get("evidence_status"),
                "terminal_wealth": row.get("terminal_wealth"),
                "mean_gross_nav": row.get("mean_gross_nav"),
            }
        )
    rows.append(
        {
            "specification": "Legacy E1 VIX CPCV",
            "config": "VIX-enabled books",
            "status": "fail_survival_gate_absorbed_zero",
            "evidence": "legacy_development_CPCV_unchanged",
            "terminal_wealth": 0.0,
            "mean_gross_nav": np.nan,
        }
    )
    frame = pd.DataFrame(rows)
    frame["risk_off_execution_inputs_complete"] = bool(
        len(execution_summary) and execution_summary["execution_feasible"].all()
    )
    frame["egarch_promotion_status"] = str(egarch_gate.get("promotion_status", "diagnostic_only"))
    frame.to_csv(out_dir / "r11_specification_status.csv", index=False)
    return frame


def write_r11_freeze_manifest(out_dir: Path, data_cutoff: pd.Timestamp) -> dict[str, object]:
    tracked = [
        Path(__file__),
        Path(__file__).with_name("r11_integer_repair.py"),
        ROOT / "src/portfolio/r11_risk_controls.py",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
        if path.exists()
    }
    freeze_time = pd.Timestamp.now(tz="UTC")
    first_eligible = (freeze_time.tz_localize(None) + pd.offsets.MonthEnd(1)).normalize()
    manifest = {
        "specification": "R1.1",
        "evidence_before_freeze": "retrospective_development_sample",
        "freeze_timestamp_utc": freeze_time.isoformat(),
        "data_cutoff": pd.Timestamp(data_cutoff).date().isoformat(),
        "first_eligible_decision_date": first_eligible.date().isoformat(),
        "required_untouched_monthly_observations": 36,
        "risk_policy": dataclasses.asdict(R11NetUtilityConfig()),
        "risk_off_policy": dataclasses.asdict(RiskOffConfig()),
        "egarch_policy": dataclasses.asdict(EgarchOverlayConfig()),
        "integer_execution_policy": {
            "target": "continuous_R1.1_solution",
            "conversion": "truncate_each_contract_count_toward_zero",
            "infeasible_action": "cash_abstention_for_the_period",
            "substitute_portfolios_allowed": False,
            "abstention_is_survival_failure": False,
            "failed_conversion_diagnostics_preserved": True,
        },
        "manual_intervention_status": "user_attested_retrospective_development_rule",
        "march_2020_market_data_deleted": False,
        "source_sha256": hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cvxpy": importlib.metadata.version("cvxpy"),
            "arch": importlib.metadata.version("arch"),
        },
        "confirmatory_claim_allowed": False,
    }
    (out_dir / "r11_prospective_freeze_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrl}",
        r"\toprule",
        r"Universe & Arm & Obs. & Mean gross & Target hit & Terminal & ES$_{95}$ & Verdict \\",
        r"\midrule",
    ]
    scored = summary[summary["evidence_status"].astype(str).eq("retrospective_development_sample")]
    for _, row in scored.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row.get("config", "")).replace("_", r"\_"),
                    ("EGARCH" if "EGARCH" in str(row.get("strategy", "")) else r"High Ceiling 25\%"),
                    str(int(row.get("observations", 0))),
                    f"{float(row.get('mean_gross_nav', np.nan)):.3f}",
                    f"{float(row.get('deployment_target_hit_rate', np.nan)):.2f}",
                    f"{float(row.get('terminal_wealth', np.nan)):.3f}",
                    f"{float(row.get('expected_shortfall_95', np.nan)):.3f}",
                    str(row.get("verdict", "")).replace("_", r"\_"),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_integer_summary(repairs: pd.DataFrame, path: Path) -> None:
    labels = {
        "truncate_toward_cash": "Direct truncation",
        "cash_abstention": "Cash abstention",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Candidates & Feasible & Selected & Mean utility \\",
        r"\midrule",
    ]
    for method, group in repairs.groupby("method", observed=True):
        lines.append(
            " & ".join(
                [
                    labels.get(str(method), str(method).replace("_", " ")),
                    str(len(group)),
                    str(
                        int(
                            (
                                group["available"].astype(bool)
                                & group["feasible"].astype(bool)
                            ).sum()
                        )
                    ),
                    str(int(group["selected"].astype(bool).sum())),
                    f"{float(group['net_utility'].mean()):.5f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evaluation-start", default="2018-01-01")
    parser.add_argument("--max-periods", type=int, default=None)
    parser.add_argument("--reuse-core-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs, _ = build_configs()
    selected = list(configs) if args.configs == "all" else [value.strip() for value in args.configs.split(",")]
    daily_returns = load_daily_return_panel(ROOT)
    if args.reuse_core_artifacts:
        return_path = args.out_dir / "r11_monthly_development_returns.csv"
        weight_path = args.out_dir / "r11_monthly_weights.csv"
        forecast_path = args.out_dir / "r11_egarch_forecasts.csv"
        repair_path = args.out_dir / "r11_integer_repair_candidates.csv"
        if not return_path.exists() or not weight_path.exists() or not forecast_path.exists() or not repair_path.exists():
            raise SystemExit("--reuse-core-artifacts requires existing return, weight, forecast, and repair artifacts")
        returns = pd.read_csv(return_path)
        returns = returns[~returns["strategy"].eq(R11_RISK_OFF_NAME)].copy()
        weights = pd.read_csv(weight_path)
        forecasts = pd.read_csv(forecast_path)
        repairs = pd.read_csv(repair_path)
    else:
        egarch_cache: dict[tuple[str, pd.Timestamp], dict[str, object]] = {}
        all_returns, all_weights, all_forecasts, all_repairs = [], [], [], []
        for label in selected:
            equities, poc_names, with_vix = configs[label]
            returns_frame, weights_frame, forecast_frame, repair_frame = run_r11_config(
                label,
                equities,
                poc_names,
                with_vix,
                daily_returns,
                egarch_cache,
                evaluation_start=args.evaluation_start,
                max_periods=args.max_periods,
            )
            all_returns.append(returns_frame)
            all_weights.append(weights_frame)
            all_forecasts.append(forecast_frame)
            all_repairs.append(repair_frame)
        returns = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
        weights = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
        forecasts = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
        repairs = pd.concat(all_repairs, ignore_index=True) if all_repairs else pd.DataFrame()

    returns = reconcile_selected_integer_diagnostics(returns, repairs)
    vix = vix_close_series(ROOT)
    sessions = pd.DatetimeIndex(daily_returns.index)
    events = build_vix_risk_off_events(vix, sessions)
    exposure_calendar = risk_off_exposure_calendar(events, sessions)
    requests = build_event_quote_request(weights, events)
    fills, execution_summary = score_event_executions(requests)
    risk_off = build_risk_off_status_rows(returns, execution_summary)
    combined_returns = pd.concat([returns, risk_off], ignore_index=True, sort=False)

    summary = summarize_strategy_returns(returns)
    if not risk_off.empty:
        for config, group in risk_off.groupby("config", observed=True):
            status = str(group["evidence_status"].iloc[0])
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "config": config,
                                "strategy": R11_RISK_OFF_NAME,
                                "evidence_status": status,
                                "verdict": status,
                                "observations": int(group["net_return"].notna().sum()),
                                "mean_gross_nav": np.nan,
                                "deployment_target_hit_rate": np.nan,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    if summary.empty:
        base_summary = pd.DataFrame()
        egarch_summary = pd.DataFrame()
        added_failures = 0
        worst_es_deterioration = np.inf
    else:
        base_summary = summary[summary["strategy"].eq(R11_NAME)].set_index("config")
        egarch_summary = summary[summary["strategy"].eq(R11_EGARCH_NAME)].set_index("config")
        added_failures = int((egarch_summary.get("verdict") == "fail_survival_gate").sum() - (base_summary.get("verdict") == "fail_survival_gate").sum())
        es_delta = egarch_summary.get("expected_shortfall_95", pd.Series(dtype=float)) - base_summary.get("expected_shortfall_95", pd.Series(dtype=float))
        worst_es_deterioration = float(max(0.0, -es_delta.min())) if len(es_delta.dropna()) else np.inf
    gate = evaluate_egarch_gate(
        forecasts,
        added_survival_failures=max(added_failures, 0),
        worst_es_deterioration=worst_es_deterioration,
    )

    combined_returns.to_csv(args.out_dir / "r11_monthly_development_returns.csv", index=False)
    weights.to_csv(args.out_dir / "r11_monthly_weights.csv", index=False)
    forecasts.to_csv(args.out_dir / "r11_egarch_forecasts.csv", index=False)
    repairs.to_csv(args.out_dir / "r11_integer_repair_candidates.csv", index=False)
    summarize_integer_repairs(repairs).to_csv(
        args.out_dir / "r11_integer_repair_method_summary.csv", index=False
    )
    summary.to_csv(args.out_dir / "r11_survival_summary.csv", index=False)
    events.to_csv(args.out_dir / "r11_vix_risk_off_events.csv", index=False)
    exposure_calendar.to_csv(args.out_dir / "r11_vix_exposure_calendar.csv", index=False)
    requests.to_csv(args.out_dir / "r11_event_quote_request.csv", index=False)
    fills.to_csv(args.out_dir / "r11_intervention_fill_ledger.csv", index=False)
    execution_summary.to_csv(args.out_dir / "r11_intervention_execution_summary.csv", index=False)
    (args.out_dir / "r11_egarch_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
    build_r11_comparisons(returns, args.out_dir)
    build_r11_trial_registry(args.out_dir)
    build_specification_status(summary, execution_summary, gate, args.out_dir)
    write_latex_summary(summary, PAPER / "tables" / "short_r11_development_summary.tex")
    write_latex_integer_summary(
        repairs, PAPER / "tables" / "short_r11_integer_repair_summary.tex"
    )
    cutoff = pd.to_datetime(returns.get("return_date"), errors="coerce").max()
    write_r11_freeze_manifest(args.out_dir, cutoff if pd.notna(cutoff) else pd.Timestamp("2026-04-30"))


if __name__ == "__main__":
    main()
