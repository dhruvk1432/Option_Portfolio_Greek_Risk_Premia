"""Repaired R1 monthly walk-forward pipeline for the option-only paper.

R1 is deliberately separate from the legacy E1 artifact path.  Every R1
decision is fit only on information available by its decision date, uses the
complete Greek factor/residual covariance, prices predictable costs before
allocation, and may hold cash.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import (
    build_configs,
)
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    compute_liquidity_caps,
    integerize_book_weights,
)
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import (
    _build_config_panel,
)
from research.papers.option_only_markowitz.analysis.conditional_premia import (
    ConditionalPremiaConfig,
    conditional_expected_returns,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    ROOT,
    TRAIN_END,
    _augment_spec_with_beta_and_stress,
    factor_panels,
    representative_specs,
)
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    estimate_greek_joint_moments,
    greek_exposure_frame,
    nearest_psd,
)


PAPER = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PAPER / "analysis" / "artifacts" / "r1_repaired"
R1_NAME = "R1 repaired net utility"


def build_factor_frame(
    underlying_returns: pd.DataFrame,
    vol_shocks: pd.DataFrame,
    underlyings: Sequence[str],
    index: pd.Index,
) -> pd.DataFrame:
    """Build the canonical delta/gamma/vega factor panel in model order."""

    aligned_under = underlying_returns.reindex(index=index, columns=underlyings).fillna(0.0)
    aligned_vol = vol_shocks.reindex(index=index, columns=underlyings).fillna(0.0)
    factors = pd.DataFrame(index=index)
    for underlying in underlyings:
        factors[f"r_{underlying}"] = aligned_under[underlying]
    for underlying in underlyings:
        values = aligned_under[underlying]
        factors[f"r2_{underlying}"] = values * values - float((values * values).mean())
    for underlying in underlyings:
        factors[f"dv_{underlying}"] = aligned_vol[underlying]
    return factors


def _assignment_short_allowed(row: pd.Series) -> bool:
    if str(row.get("asset_class", "equity_option")) != "equity_option":
        return True
    spot = float(row.get("start_spot", row.get("spot", np.nan)))
    strike = float(row.get("strike", np.nan))
    mark = float(row.get("mark", np.nan))
    if not np.isfinite([spot, strike, mark]).all() or mark <= 0:
        return False
    kind = str(row.get("kind", row.get("right", ""))).lower()
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    extrinsic = mark - intrinsic
    deep_itm = (kind == "call" and spot > 1.03 * strike) or (kind == "put" and strike > 1.03 * spot)
    return not bool(deep_itm and extrinsic <= max(0.10 * mark, 0.05))


def build_optimization_cost_spec(
    cost_inputs: pd.DataFrame,
    contracts: Sequence[str],
    decision_date: pd.Timestamp,
    config: ResearchCostConfig,
) -> tuple[OptimizationCostSpec, pd.DataFrame]:
    """Build costs and operational coefficients from decision-known rows."""

    observed = cost_inputs.copy()
    observed["decision_date"] = pd.to_datetime(observed.get("decision_date"), errors="coerce")
    observed = observed[observed["decision_date"].le(pd.Timestamp(decision_date))]
    observed = observed.sort_values("decision_date").groupby("asset_id", observed=True).tail(1).set_index("asset_id")

    long_cost = pd.Series(0.0, index=list(contracts), dtype=float)
    short_cost = pd.Series(0.0, index=list(contracts), dtype=float)
    short_margin = pd.Series(1.0, index=list(contracts), dtype=float)
    short_allowed = pd.Series(False, index=list(contracts), dtype=bool)
    diagnostic_rows: list[dict[str, object]] = []
    for contract in contracts:
        if contract not in observed.index:
            default_spread = (
                config.default_vix_option_rel_spread
                if "VIX" in str(contract).upper() or "VX_FRONT" in str(contract).upper()
                else config.default_equity_option_rel_spread
            )
            fallback = config.spread_cross_fraction * default_spread + 2.0 * config.slippage_bps_per_side / 10_000.0
            long_cost.loc[contract] = fallback
            short_cost.loc[contract] = fallback
            diagnostic_rows.append({"asset_id": contract, "source": "default_imputed", "full_cost": fallback})
            continue
        row = observed.loc[contract]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        mark = float(row.get("mark", np.nan))
        relative_spread = float(row.get("relative_spread", np.nan))
        if not np.isfinite(relative_spread) or relative_spread < 0:
            relative_spread = (
                config.default_vix_option_rel_spread
                if str(row.get("asset_class", "")) == "vix_option"
                else config.default_equity_option_rel_spread
            )
        fee_return = (
            2.0 * config.fee_per_contract_per_side / (mark * config.option_multiplier)
            if np.isfinite(mark) and mark > 0
            else 0.0
        )
        entry = (
            config.spread_cross_fraction * relative_spread
            + fee_return
            + 2.0 * config.slippage_bps_per_side / 10_000.0
        )
        long_cost.loc[contract] = entry
        short_cost.loc[contract] = entry
        spot = float(row.get("start_spot", row.get("spot", np.nan)))
        holding_years = float(row.get("holding_years", 21.0 / 365.0))
        margin = (
            max(config.short_option_margin_floor, 0.20 * spot / mark)
            if np.isfinite([spot, mark]).all() and mark > 0
            else 1.0
        )
        short_margin.loc[contract] = margin
        long_cost.loc[contract] += config.margin_funding_rate * holding_years
        borrow = float(row.get("borrow_rate_proxy", 0.0))
        kind = str(row.get("kind", row.get("right", ""))).lower()
        short_cost.loc[contract] += margin * config.margin_funding_rate * holding_years
        if kind == "call" and np.isfinite(borrow):
            short_cost.loc[contract] += max(borrow, 0.0) * holding_years
        short_allowed.loc[contract] = _assignment_short_allowed(row)
        diagnostic_rows.append(
            {
                "asset_id": contract,
                "source": str(row.get("relative_spread_source", "observed")),
                "full_cost": entry,
            }
        )
    return (
        OptimizationCostSpec(long_cost, short_cost, short_margin, short_allowed),
        pd.DataFrame(diagnostic_rows),
    )


def build_r1_model(
    spec: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_under: pd.DataFrame,
    train_vol: pd.DataFrame,
    underlyings: Sequence[str],
) -> OptionOnlyMarkowitzModel:
    """Fit the R1 mean and complete covariance on one training window."""

    spec = _augment_spec_with_beta_and_stress(spec, train_under, train_returns.index)
    premia_config = ConditionalPremiaConfig(
        horizon_years=21.0 / 252.0,
        shrinkage_to_zero=0.75,
        historical_weight=0.0,
        structural_weight=0.75,
    )
    mu, components = conditional_expected_returns(spec, train_returns, train_under, train_vol, premia_config)
    options = OptionOnlySpec(spec)
    B = greek_exposure_frame(options)
    factors = build_factor_frame(train_under, train_vol, underlyings, train_returns.index)
    moments = estimate_greek_joint_moments(
        train_returns.reindex(columns=B.index).fillna(0.0),
        factors,
        B,
        regularize=True,
    )
    under_cov = train_under.reindex(columns=underlyings).fillna(0.0).cov().fillna(0.0)
    vol_cov = train_vol.reindex(columns=underlyings).fillna(0.0).cov().fillna(0.0)
    for frame in (under_cov, vol_cov):
        values = nearest_psd(frame.to_numpy(float))
        frame.iloc[:, :] = values
    constraints = OptionMarkowitzConstraints(
        gross_nav=1.0,
        net_nav_abs=1.0,
        short_nav_abs=0.25,
        per_contract_abs=0.18,
        underlying_gross={u: (0.20 if u == "VX_FRONT" else 0.35) for u in underlyings},
        beta_spy_abs=3.0,
        vix_vega_abs=8.0,
        stress_loss_abs=0.20,
    )
    model = OptionOnlyMarkowitzModel(
        options,
        FactorShockSpec(under_cov, vol_cov),
        expected_returns=mu.reindex(B.index).fillna(0.0),
        constraints=constraints,
        covariance_shrinkage=0.0,
        joint_moments=moments,
    )
    model.conditional_premia_components = components
    return model


def r1_constraint_diagnostics(
    model: OptionOnlyMarkowitzModel,
    weights: pd.Series,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    caps: pd.Series,
    config: NetUtilityConfig,
) -> dict[str, float | bool]:
    """Independently check continuous or integer R1 positions."""

    w = weights.reindex(model.contracts).fillna(0.0).to_numpy(float)
    long_cost, short_cost, short_margin, short_allowed = costs.aligned(model.contracts)
    long = np.maximum(w, 0.0)
    short = np.maximum(-w, 0.0)
    cost = float(long_cost @ long + short_cost @ short)
    net = scenarios.reindex(columns=model.contracts).fillna(0.0).to_numpy(float) @ w - cost
    losses = -net
    threshold = float(np.quantile(losses, config.cvar_alpha, method="higher"))
    tail = losses[losses >= threshold - 1e-12]
    cvar = float(tail.mean()) if len(tail) else threshold
    stress = model._stress_matrix()
    worst_stress = float(np.min(stress @ w)) if stress is not None else 0.0
    margin = float(short_margin @ short)
    collateral = float(long.sum() + margin)
    caps_aligned = caps.reindex(model.contracts).fillna(0.0).to_numpy(float)
    breaches = {
        "base": model._max_constraint_violation(w),
        "cap": float(np.maximum(np.abs(w) - caps_aligned, 0.0).max(initial=0.0)),
        "cvar": max(cvar - config.cvar_loss_nav, 0.0),
        "stress": max(-config.stress_loss_nav - worst_stress, 0.0),
        "margin": max(margin - config.short_margin_nav, 0.0),
        "collateral": max(collateral - config.collateral_nav, 0.0),
        "assignment": float(np.maximum(-w[~short_allowed], 0.0).max(initial=0.0)),
    }
    max_breach = max(breaches.values())
    return {
        "feasible": bool(max_breach <= 1e-6),
        "max_breach": float(max_breach),
        "scenario_cvar_loss": cvar,
        "worst_stress_return": worst_stress,
        "short_margin_used": margin,
        "collateral_used": collateral,
        "gross_nav": float(np.abs(w).sum()),
        **{f"breach_{name}": float(value) for name, value in breaches.items()},
    }


def integerize_r1_weights(
    model: OptionOnlyMarkowitzModel,
    continuous: pd.Series,
    marks: pd.Series,
    nav: float,
    caps: pd.Series,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    config: NetUtilityConfig,
) -> tuple[pd.Series, dict[str, float | bool]]:
    """Round toward cash; fail closed to all-cash if any constraint breaks."""

    sized = integerize_book_weights(continuous, marks, nav=nav, caps=caps)
    # Nearest rounding can increase risk.  Truncate every count toward zero.
    counts = np.sign(sized["continuous_contracts"]) * np.floor(np.abs(sized["continuous_contracts"]))
    realized = counts * 100.0 * marks.reindex(continuous.index).to_numpy(float) / float(nav)
    weights = pd.Series(realized, index=continuous.index, name="weight")
    diagnostics = r1_constraint_diagnostics(model, weights, scenarios, costs, caps, config)
    if not diagnostics["feasible"]:
        weights[:] = 0.0
        diagnostics = r1_constraint_diagnostics(model, weights, scenarios, costs, caps, config)
        diagnostics["integer_repair_failed_to_cash"] = True
    else:
        diagnostics["integer_repair_failed_to_cash"] = False
    diagnostics["integer_contracts"] = float(np.abs(counts).sum())
    return weights, diagnostics


def survival_diagnostics(
    returns: pd.Series,
    *,
    margin_breaches: int = 0,
    collateral_breaches: int = 0,
    integer_failures: int = 0,
    absorbed_validation_paths: int = 0,
) -> dict[str, float | int | str]:
    """Apply the paper's hard survival gate; Sharpe cannot override failure."""

    values = pd.Series(returns, dtype=float).dropna()
    wealth = 1.0
    path = []
    ruined = False
    for value in values:
        if ruined or value <= -1.0:
            wealth = 0.0
            ruined = True
        else:
            wealth *= 1.0 + float(value)
        path.append(wealth)
    wealth_series = pd.Series(path, index=values.index, dtype=float)
    running_peak = pd.Series(
        np.maximum.accumulate(np.r_[1.0, wealth_series.to_numpy(float)])[1:],
        index=wealth_series.index,
    )
    drawdown = wealth_series / running_peak.replace(0.0, np.nan) - 1.0
    q05 = float(values.quantile(0.05)) if len(values) else np.nan
    tail = values[values <= q05] if len(values) else values
    expected_shortfall = float(tail.mean()) if len(tail) else np.nan
    hard_fail = bool(
        ruined
        or margin_breaches
        or collateral_breaches
        or integer_failures
        or absorbed_validation_paths
    )
    geometric = (
        float(wealth ** (12.0 / len(values)) - 1.0)
        if len(values) and wealth > 0
        else -1.0 if len(values) else np.nan
    )
    return {
        "verdict": "fail_survival_gate" if hard_fail else "development_survived",
        "terminal_wealth": float(wealth),
        "annualized_geometric_return": geometric,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "worst_month": float(values.min()) if len(values) else np.nan,
        "expected_shortfall_95": expected_shortfall,
        "ruin_count": int(ruined),
        "margin_breaches": int(margin_breaches),
        "collateral_breaches": int(collateral_breaches),
        "integer_failures": int(integer_failures),
        "absorbed_validation_paths": int(absorbed_validation_paths),
    }


def paired_block_bootstrap_comparison(
    strategy: pd.Series,
    benchmark: pd.Series,
    *,
    block_length: int = 6,
    n_boot: int = 1000,
    seed: int = 20260711,
) -> dict[str, float | int]:
    """Paired circular-block inference for log growth and expected shortfall."""

    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 3:
        return {"observations": int(len(aligned))}
    values = aligned.to_numpy(float)
    n = len(values)
    block = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)

    def metrics(sample: np.ndarray) -> tuple[float, float]:
        clipped = np.maximum(sample, -0.999999)
        log_diff = float(np.mean(np.log1p(clipped[:, 0]) - np.log1p(clipped[:, 1])))
        q_strategy = float(np.quantile(sample[:, 0], 0.05))
        q_benchmark = float(np.quantile(sample[:, 1], 0.05))
        es_strategy = float(sample[sample[:, 0] <= q_strategy, 0].mean())
        es_benchmark = float(sample[sample[:, 1] <= q_benchmark, 1].mean())
        return log_diff, es_strategy - es_benchmark

    draws = np.zeros((n_boot, 2), dtype=float)
    blocks_needed = int(np.ceil(n / block))
    for draw in range(n_boot):
        starts = rng.integers(0, n, size=blocks_needed)
        idx = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
        draws[draw] = metrics(values[idx])
    observed = metrics(values)
    return {
        "observations": n,
        "monthly_log_growth_difference": observed[0],
        "log_growth_ci_lo": float(np.quantile(draws[:, 0], 0.05)),
        "log_growth_ci_hi": float(np.quantile(draws[:, 0], 0.95)),
        "expected_shortfall_difference": observed[1],
        "expected_shortfall_ci_lo": float(np.quantile(draws[:, 1], 0.05)),
        "expected_shortfall_ci_hi": float(np.quantile(draws[:, 1], 0.95)),
        "bootstrap_probability_log_growth_le_zero": float(np.mean(draws[:, 0] <= 0.0)),
        "bootstrap_probability_es_difference_le_zero": float(np.mean(draws[:, 1] <= 0.0)),
    }


def build_r1_baseline_comparisons(returns_frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Compare R1 to the existing matched-naive and stock walk-forward paths."""

    path = PAPER / "analysis/artifacts/breadth_solutions/robustness/final_walk_forward_return_paths.csv"
    if not path.exists() or returns_frame.empty:
        comparison = pd.DataFrame()
        comparison.to_csv(out_dir / "r1_paired_bootstrap_comparisons.csv", index=False)
        return comparison
    baseline = pd.read_csv(path, parse_dates=["return_date"])
    rows: list[dict[str, object]] = []
    for config, group in returns_frame.groupby("config", observed=True):
        strategy = group.set_index(pd.to_datetime(group["return_date"]))["net_return"]
        candidates = [
            ("matched_capped_naive", baseline[(baseline["config"] == config) & (baseline["family"] == "Matched capped naive")]),
            (
                "stock_markowitz",
                baseline[
                    (baseline["family"] == "Stock baseline")
                    & (baseline["config"] == ("stock_56" if str(config).startswith("larger") else "stock"))
                ],
            ),
        ]
        for benchmark_name, frame in candidates:
            if frame.empty:
                continue
            benchmark = frame.set_index("return_date")["return"]
            rows.append(
                {
                    "config": config,
                    "benchmark": benchmark_name,
                    **paired_block_bootstrap_comparison(strategy, benchmark),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "r1_paired_bootstrap_comparisons.csv", index=False)
    return comparison


def run_r1_config(
    label: str,
    equity_underlyings: Sequence[str],
    poc_names: Sequence[str],
    with_vix: bool,
    *,
    nav: float = 1_000_000.0,
    min_train_months: int = 36,
    max_periods: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run monthly R1 decisions for one universe on the development sample."""

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
    risk_config = NetUtilityConfig()
    dates = [date for date in pd.DatetimeIndex(returns.index).sort_values() if date > TRAIN_END]
    if max_periods is not None:
        dates = dates[:max_periods]
    return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for period_number, return_date in enumerate(dates, start=1):
        if period_number == 1 or period_number % 10 == 0 or period_number == len(dates):
            print(f"[R1] {label}: decision {period_number}/{len(dates)} for {return_date.date()}", flush=True)
        prior_dates = pd.DatetimeIndex(returns.index[returns.index < return_date]).sort_values()
        if len(prior_dates) < min_train_months:
            continue
        train_dates = prior_dates[-min_train_months:]
        train_start, train_end = train_dates[0], train_dates[-1]
        period_detail = detail[pd.to_datetime(detail["return_date"]).eq(return_date)]
        decision_date = (
            pd.Timestamp(period_detail["decision_date"].max())
            if not period_detail.empty
            else pd.Timestamp(train_end)
        )
        spec = representative_specs(reps, returns, train_start=train_start, train_end=decision_date)
        train_returns = returns.reindex(index=train_dates, columns=spec.index).fillna(0.0)
        counts = train_returns.count()
        keep = list(counts[counts >= min(24, min_train_months)].index)
        spec = spec.reindex(keep).dropna(subset=["underlying", "mark"])
        if spec.empty:
            continue
        train_returns = train_returns.reindex(columns=spec.index).fillna(0.0)
        train_under = underlying_returns.reindex(index=train_dates, columns=universe).fillna(0.0)
        train_vol = vol_shocks.reindex(index=train_dates, columns=universe).fillna(0.0)
        model = build_r1_model(spec, train_returns, train_under, train_vol, universe)
        caps_frame = compute_liquidity_caps(
            reps,
            spec["mark"],
            nav,
            participation=0.05,
            per_contract_abs=0.18,
            train_end=decision_date,
        )
        caps = caps_frame["bound"].reindex(model.contracts).fillna(0.0)
        optimization_costs, _ = build_optimization_cost_spec(
            cost_inputs,
            model.contracts,
            decision_date,
            cost_config,
        )
        result = model.solve_net_utility(
            train_returns,
            optimization_costs,
            risk_config,
            per_contract_caps=caps,
        )
        realized_weights, execution = integerize_r1_weights(
            model,
            result.weights,
            spec["mark"],
            nav,
            caps,
            train_returns,
            optimization_costs,
            risk_config,
        )
        gross_return = float(returns.reindex(index=[return_date], columns=model.contracts).fillna(0.0).iloc[0] @ realized_weights)
        long_cost, short_cost, _, _ = optimization_costs.aligned(model.contracts)
        w = realized_weights.to_numpy(float)
        predicted_cost = float(long_cost @ np.maximum(w, 0.0) + short_cost @ np.maximum(-w, 0.0))
        net_return = gross_return - predicted_cost
        return_rows.append(
            {
                "config": label,
                "strategy": R1_NAME,
                "return_date": return_date,
                "decision_date": decision_date,
                "train_start": train_start,
                "train_end": train_end,
                "gross_return": gross_return,
                "predicted_cost": predicted_cost,
                "net_return": net_return,
                "gross_nav": float(realized_weights.abs().sum()),
                "predicted_annual_vol": result.objective_stats["predicted_annual_vol"],
                "scenario_cvar_loss": execution["scenario_cvar_loss"],
                "short_margin_used": execution["short_margin_used"],
                "collateral_used": execution["collateral_used"],
                "integer_repair_failed": bool(execution["integer_repair_failed_to_cash"]),
                "information_set_valid": bool(train_end < return_date and decision_date < return_date),
            }
        )
        for contract, weight in realized_weights.items():
            weight_rows.append(
                {
                    "config": label,
                    "return_date": return_date,
                    "decision_date": decision_date,
                    "asset_id": contract,
                    "weight": float(weight),
                }
            )
    returns_frame = pd.DataFrame(return_rows)
    weights_frame = pd.DataFrame(weight_rows)
    if returns_frame.empty:
        summary: dict[str, object] = {"config": label, "verdict": "no_development_observations"}
    else:
        summary = {
            "config": label,
            "strategy": R1_NAME,
            "evidence_status": "retrospective_development_sample",
            **survival_diagnostics(
                returns_frame.set_index("return_date")["net_return"],
                margin_breaches=int((returns_frame["short_margin_used"] > risk_config.short_margin_nav + 1e-6).sum()),
                collateral_breaches=int((returns_frame["collateral_used"] > risk_config.collateral_nav + 1e-6).sum()),
                integer_failures=int(returns_frame["integer_repair_failed"].sum()),
            ),
            "observations": int(len(returns_frame)),
            "mean_gross_nav": float(returns_frame["gross_nav"].mean()),
            "all_information_sets_valid": bool(returns_frame["information_set_valid"].all()),
        }
    return returns_frame, weights_frame, summary


def build_trial_registry(out_dir: Path) -> pd.DataFrame:
    """Reconstruct the known research trials; the count remains a lower bound."""

    sources = [
        PAPER / "analysis/artifacts/breadth_solutions/p1_regularization_results.csv",
        PAPER / "analysis/artifacts/breadth_solutions/p2_liquidity_results.csv",
        PAPER / "analysis/artifacts/breadth_solutions/p3_combined_results.csv",
        PAPER / "analysis/artifacts/breadth_solutions/robustness/breadth_realized_candidate_summary.csv",
        PAPER / "analysis/artifacts/breadth_solutions/robustness/breadth_e1_channel_ablation.csv",
        PAPER / "analysis/artifacts/breadth_solutions/robustness/breadth_simulation_summary.csv",
    ]
    rows: list[dict[str, object]] = []
    for source in sources:
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        identity = [
            column
            for column in [
                "config", "strategy", "arm", "point_id", "participation", "aum", "cap_mode", "mode",
                "Return basis", "Strategy", "Requested method", "Simulation",
            ]
            if column in frame
        ]
        unique = frame[identity].drop_duplicates() if identity else frame.iloc[:, :0].drop_duplicates()
        for _, row in unique.iterrows():
            rows.append(
                {
                    "source": str(source.relative_to(PAPER)),
                    "trial_key": "|".join(f"{name}={row[name]}" for name in identity),
                    "count_status": "known_lower_bound",
                }
            )
    rows.append({"source": "analysis/r1_repaired_pipeline.py", "trial_key": "R1_fixed_policy", "count_status": "known_lower_bound"})
    registry = pd.DataFrame(rows).drop_duplicates(["source", "trial_key"])
    out_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out_dir / "research_trial_registry.csv", index=False)
    (out_dir / "research_trial_registry.json").write_text(
        json.dumps(
            {
                "known_trial_count_lower_bound": int(len(registry)),
                "is_complete": False,
                "reason": "Earlier undocumented researcher iterations cannot be reconstructed from artifacts.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return registry


def write_prospective_freeze_manifest(out_dir: Path, data_cutoff: pd.Timestamp) -> dict[str, object]:
    """Freeze the R1 policy and require 36 future monthly observations."""

    tracked = [
        ROOT / "src/portfolio/option_only_markowitz_model.py",
        Path(__file__),
        PAPER / "sections/short_paper.tex",
        PAPER / "sections/short_appendix.tex",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
        if path.exists()
    }
    freeze_time = pd.Timestamp.now(tz="UTC")
    first_eligible = (freeze_time.tz_localize(None) + pd.offsets.MonthEnd(1)).normalize()
    manifest = {
        "specification": "R1",
        "evidence_before_freeze": "retrospective_development_sample",
        "freeze_timestamp_utc": freeze_time.isoformat(),
        "data_cutoff": pd.Timestamp(data_cutoff).date().isoformat(),
        "first_eligible_decision_date": first_eligible.date().isoformat(),
        "required_untouched_monthly_observations": 36,
        "training_window_months": 36,
        "nav": 1_000_000.0,
        "volume_participation": 0.05,
        "covariance_estimator": "Greek B with Ledoit-Wolf joint factor/residual correlation covariance",
        "optimizer": "net mean-variance utility with cash",
        "risk_policy": dataclasses.asdict(NetUtilityConfig()),
        "primary_endpoints": [
            "terminal_wealth",
            "annualized_geometric_return",
            "max_drawdown",
            "worst_month",
            "expected_shortfall_95",
            "ruin_count",
            "margin_breaches",
            "collateral_breaches",
            "integer_failures",
        ],
        "secondary_endpoints": ["sharpe", "sortino"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cvxpy": importlib.metadata.version("cvxpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "source_sha256": hashes,
        "confirmatory_claim_allowed": False,
    }
    (out_dir / "prospective_freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_r1_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    """Write the compact R1 survival-first table consumed by the paper."""

    lines = [
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Universe & Obs. & Mean gross & Terminal wealth & Ann. geom. & Worst month & ES$_{95}$ & Verdict \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        config = str(row.get("config", "")).replace("_", r"\_")
        verdict = str(row.get("verdict", "")).replace("_", r"\_")
        values = [
            config,
            str(int(row.get("observations", 0))),
            f"{float(row.get('mean_gross_nav', np.nan)):.3f}",
            f"{float(row.get('terminal_wealth', np.nan)):.3f}",
            f"{float(row.get('annualized_geometric_return', np.nan)):.3f}",
            f"{float(row.get('worst_month', np.nan)):.3f}",
            f"{float(row.get('expected_shortfall_95', np.nan)):.3f}",
            verdict,
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-periods", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs, _ = build_configs()
    selected = list(configs) if args.configs == "all" else [value.strip() for value in args.configs.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_returns = []
    all_weights = []
    summaries = []
    for label in selected:
        equities, poc_names, with_vix = configs[label]
        returns, weights, summary = run_r1_config(
            label,
            equities,
            poc_names,
            with_vix,
            max_periods=args.max_periods,
        )
        all_returns.append(returns)
        all_weights.append(weights)
        summaries.append(summary)
    returns_frame = pd.concat(all_returns, ignore_index=True) if all_returns else pd.DataFrame()
    weights_frame = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    returns_frame.to_csv(args.out_dir / "r1_monthly_development_returns.csv", index=False)
    weights_frame.to_csv(args.out_dir / "r1_monthly_weights.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.out_dir / "r1_survival_summary.csv", index=False)
    (args.out_dir / "r1_survival_summary.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    write_r1_latex_summary(summary_frame, PAPER / "tables" / "short_r1_survival_summary.tex")
    build_r1_baseline_comparisons(returns_frame, args.out_dir)
    build_trial_registry(args.out_dir)
    cutoff = pd.to_datetime(returns_frame.get("return_date"), errors="coerce").max()
    if pd.isna(cutoff):
        cutoff = TRAIN_END
    write_prospective_freeze_manifest(args.out_dir, pd.Timestamp(cutoff))


if __name__ == "__main__":
    main()
