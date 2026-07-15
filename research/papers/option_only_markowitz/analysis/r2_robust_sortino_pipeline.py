"""Chronological R2 robust-Sortino development replay and evidence artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import build_configs
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import compute_liquidity_caps
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import _build_config_panel
from research.papers.option_only_markowitz.analysis.conditional_premia import (
    ConditionalPremiaConfig,
    conditional_expected_returns,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import (
    PAPER,
    build_factor_frame,
    build_optimization_cost_spec,
    survival_diagnostics,
)
from research.papers.option_only_markowitz.analysis.r11_higher_risk_pipeline import (
    R11_NAME,
    load_daily_return_panel,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    ROOT,
    _augment_spec_with_beta_and_stress,
    factor_panels,
    representative_specs,
)
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    GreekJointMomentSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    greek_exposure_frame,
    nearest_psd,
)
from src.portfolio.r2_robust_sortino import (
    R2MomentSpec,
    RobustSortinoConfig,
    apply_joint_volatility_scaling,
    circular_block_scenarios,
    estimate_r2_moments,
    exponentially_weighted_mean,
    integerize_r2_direct_or_abstain,
    option_covariance,
    select_daily_volatility_overlay,
    select_recent_covariance_weight,
    solve_r2_robust_sortino,
)


DEFAULT_OUT = PAPER / "analysis" / "artifacts" / "r2_robust_sortino"
R2_NAME = "R2 robust net Sortino"
EVIDENCE_STATUS = "retrospective_development_sample"
_DAILY_FORECAST_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _ewm_transform(frame: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """Transform a panel so the existing structural model sees EWM moments."""

    values = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    mean = exponentially_weighted_mean(values, half_life)
    age = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * age / half_life)
    centered = values - mean
    variance = centered.pow(2).mul(weights, axis=0).sum().div(
        values.notna().mul(weights, axis=0).sum().replace(0.0, np.nan)
    )
    target_std = np.sqrt(variance.clip(lower=0.0))
    ordinary_mean = values.mean()
    ordinary_std = values.std(ddof=1).replace(0.0, np.nan)
    transformed = (values - ordinary_mean).div(ordinary_std).mul(target_std).add(mean)
    return transformed.fillna(mean).fillna(0.0)


def r2_expected_returns(
    spec: pd.DataFrame,
    option_returns: pd.DataFrame,
    underlying_returns: pd.DataFrame,
    vol_shocks: pd.DataFrame,
    config: RobustSortinoConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    """Preserve R1's structural channels with expanding EWM premia and no option mean."""

    under = _ewm_transform(underlying_returns, config.premia_half_life_months)
    vol = _ewm_transform(vol_shocks, config.premia_half_life_months)
    premia_config = ConditionalPremiaConfig(
        horizon_years=21.0 / 252.0,
        shrinkage_to_zero=0.75,
        historical_weight=0.0,
        structural_weight=0.75,
    )
    mu, components = conditional_expected_returns(spec, option_returns, under, vol, premia_config)
    components["r2_premia_half_life_months"] = config.premia_half_life_months
    components["r2_direct_option_mean_weight"] = 0.0
    return mu, components


def _inner_qlike_selection(
    option_returns: pd.DataFrame,
    factors: pd.DataFrame,
    loadings: pd.DataFrame,
    config: RobustSortinoConfig,
) -> tuple[float, pd.DataFrame]:
    """Build up to twelve one-step covariance forecasts using prior rows only."""

    if len(option_returns) < config.recent_months + config.min_inner_forecasts:
        return select_recent_covariance_weight([], [], [], config)
    recent_forecasts: list[np.ndarray] = []
    expanding_forecasts: list[np.ndarray] = []
    realized: list[np.ndarray] = []
    start = max(config.recent_months, len(option_returns) - config.min_inner_forecasts)
    systematic = pd.DataFrame(
        factors.to_numpy(float) @ loadings.to_numpy(float).T,
        index=factors.index,
        columns=loadings.index,
    )
    for position in range(start, len(option_returns)):
        past_returns = option_returns.iloc[:position]
        past_factors = factors.iloc[:position]
        try:
            moments = estimate_r2_moments(
                past_returns,
                past_factors,
                loadings,
                recent_weight=config.default_recent_weight,
                config=config,
            )
        except ValueError:
            continue
        observed = option_returns.iloc[position].reindex(loadings.index)
        fallback = systematic.iloc[position].reindex(loadings.index)
        recent_forecasts.append(option_covariance(moments.recent, loadings).to_numpy(float))
        expanding_forecasts.append(option_covariance(moments.expanding, loadings).to_numpy(float))
        realized.append(observed.fillna(fallback).to_numpy(float))
    return select_recent_covariance_weight(recent_forecasts, expanding_forecasts, realized, config)


def _daily_overlay(
    moment: GreekJointMomentSpec,
    loadings: pd.DataFrame,
    daily_returns: pd.DataFrame,
    underlyings: Sequence[str],
    decision_date: pd.Timestamp,
    config: RobustSortinoConfig,
) -> tuple[GreekJointMomentSpec, pd.DataFrame, list[dict[str, Any]]]:
    ratios: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for underlying in underlyings:
        cache_key = (
            str(underlying),
            pd.Timestamp(decision_date).normalize(),
            config.daily_window,
            config.min_daily_observations,
            config.volatility_horizon_days,
            config.volatility_blend_weights,
            config.variance_ratio_floor,
            config.variance_ratio_ceiling,
        )
        if cache_key not in _DAILY_FORECAST_CACHE:
            series = daily_returns.get(str(underlying), pd.Series(dtype=float))
            history = series.loc[pd.to_datetime(series.index) <= decision_date]
            _DAILY_FORECAST_CACHE[cache_key] = select_daily_volatility_overlay(history, config)
        cached = _DAILY_FORECAST_CACHE[cache_key]
        forecast = {
            key: (value.copy() if isinstance(value, pd.DataFrame) else value)
            for key, value in cached.items()
        }
        ratios[str(underlying)] = float(forecast["variance_ratio"])
        ledger = forecast.pop("qlike_ledger", pd.DataFrame())
        row = {**forecast, "underlying": str(underlying), "decision_date": decision_date}
        rows.append(row)
        if isinstance(ledger, pd.DataFrame) and not ledger.empty:
            for record in ledger.to_dict("records"):
                rows.append({**row, **record, "row_type": "inner_qlike"})
    scaled, covariance = apply_joint_volatility_scaling(moment, loadings, ratios)
    return scaled, covariance, rows


def build_r2_model(
    spec: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_under: pd.DataFrame,
    train_vol: pd.DataFrame,
    underlyings: Sequence[str],
    daily_returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    config: RobustSortinoConfig,
) -> tuple[OptionOnlyMarkowitzModel, R2MomentSpec, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Fit the complete R2 information set for one decision."""

    spec = _augment_spec_with_beta_and_stress(spec, train_under, train_returns.index)
    options = OptionOnlySpec(spec)
    loadings = greek_exposure_frame(options)
    # The 24-of-36 availability filter can remove every representative
    # contract for an underlying.  Covariance factors must follow the surviving
    # Greek matrix exactly, not the wider pre-filter universe.
    active_underlyings = sorted(spec["underlying"].astype(str).unique())
    active_under = train_under.reindex(columns=active_underlyings)
    active_vol = train_vol.reindex(columns=active_underlyings)
    factors = build_factor_frame(active_under, active_vol, active_underlyings, train_returns.index)
    recent_weight, qlike = _inner_qlike_selection(train_returns, factors, loadings, config)
    moments = estimate_r2_moments(
        train_returns,
        factors,
        loadings,
        recent_weight=recent_weight,
        train_end=decision_date,
        config=config,
        qlike_ledger=qlike,
    )
    scaled_joint, scaled_covariance, volatility_rows = _daily_overlay(
        moments.blended, loadings, daily_returns, active_underlyings, decision_date, config
    )
    moments = dataclasses.replace(
        moments,
        blended=scaled_joint,
        blended_option_cov=scaled_covariance,
        volatility_ledger={"decision_date": str(decision_date)},
    )
    mu, components = r2_expected_returns(spec, train_returns, active_under, active_vol, config)
    under_cov = active_under.cov().fillna(0.0)
    vol_cov = active_vol.cov().fillna(0.0)
    for frame in (under_cov, vol_cov):
        frame.iloc[:, :] = nearest_psd(frame.to_numpy(float))
    constraints = OptionMarkowitzConstraints(
        gross_nav=1.0,
        net_nav_abs=1.0,
        short_nav_abs=0.25,
        per_contract_abs=0.18,
        underlying_gross={u: (0.20 if u == "VX_FRONT" else 0.35) for u in active_underlyings},
        beta_spy_abs=3.0,
        vix_vega_abs=8.0,
        stress_loss_abs=0.20,
    )
    model = OptionOnlyMarkowitzModel(
        options,
        FactorShockSpec(under_cov, vol_cov),
        expected_returns=mu.reindex(loadings.index).fillna(0.0),
        constraints=constraints,
        covariance_shrinkage=0.0,
        joint_moments=scaled_joint,
    )
    model.conditional_premia_components = components
    base = moments.option_returns_imputed.reindex(columns=model.contracts)
    recent = base.iloc[-config.recent_months :]
    bootstrap = circular_block_scenarios(
        base,
        paths=config.bootstrap_scenarios,
        block_length=config.bootstrap_block_months,
        seed=config.random_seed + int(pd.Timestamp(decision_date).strftime("%Y%m%d")),
    )
    families: dict[str, pd.DataFrame] = {"recent": recent, "expanding_bootstrap": bootstrap}
    for scenario_id, frame in enumerate(moments.imputation_scenarios, start=1):
        families[f"imputation_{scenario_id}"] = frame.reindex(columns=model.contracts)
    return model, moments, families, components, pd.DataFrame(volatility_rows)


def run_r2_config(
    label: str,
    equity_underlyings: Sequence[str],
    poc_names: Sequence[str],
    with_vix: bool,
    daily_returns: pd.DataFrame,
    *,
    nav: float = 1_000_000.0,
    evaluation_start: str = "2018-02-01",
    max_periods: int | None = None,
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay one universe from February 2018 with expanding training history."""

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
    dates = [date for date in pd.DatetimeIndex(returns.index).sort_values() if date >= pd.Timestamp(evaluation_start)]
    if max_periods is not None:
        dates = dates[:max_periods]
    return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    abstention_rows: list[dict[str, Any]] = []
    premia_rows: list[dict[str, Any]] = []
    for period_number, return_date in enumerate(dates, start=1):
        if period_number == 1 or period_number % 10 == 0 or period_number == len(dates):
            print(f"[R2] {label}: decision {period_number}/{len(dates)} for {return_date.date()}", flush=True)
        prior_dates = pd.DatetimeIndex(returns.index[returns.index < return_date]).sort_values()
        if len(prior_dates) < config.recent_months:
            continue
        train_dates = prior_dates
        train_start, train_end = train_dates[0], train_dates[-1]
        period_detail = detail[pd.to_datetime(detail["return_date"]).eq(return_date)]
        decision_date = pd.Timestamp(period_detail["decision_date"].max()) if not period_detail.empty else pd.Timestamp(train_end)
        spec = representative_specs(reps, returns, train_start=train_start, train_end=decision_date)
        recent_raw = returns.reindex(index=train_dates[-config.recent_months :], columns=spec.index)
        keep = recent_raw.notna().sum()[lambda value: value >= config.min_recent_observations].index
        spec = spec.reindex(keep).dropna(subset=["underlying", "mark"])
        if spec.empty:
            continue
        train_returns = returns.reindex(index=train_dates, columns=spec.index)
        train_under = underlying_returns.reindex(index=train_dates, columns=universe)
        train_vol = vol_shocks.reindex(index=train_dates, columns=universe)
        model, moments, families, components, vol_ledger = build_r2_model(
            spec, train_returns, train_under, train_vol, universe, daily_returns, decision_date, config
        )
        caps_frame = compute_liquidity_caps(
            reps,
            spec["mark"],
            nav,
            participation=0.05,
            per_contract_abs=0.18,
            train_end=decision_date,
        )
        caps = caps_frame["bound"].reindex(model.contracts).fillna(0.0)
        costs, _ = build_optimization_cost_spec(cost_inputs, model.contracts, decision_date, cost_config)
        result = solve_r2_robust_sortino(
            model,
            moments.option_returns_imputed,
            families,
            costs,
            caps,
            config,
        )
        integer = integerize_r2_direct_or_abstain(
            model,
            result.weights,
            spec["mark"],
            nav,
            caps,
            moments.option_returns_imputed,
            families,
            costs,
            config,
        )
        selected = integer.weights
        realized = returns.reindex(index=[return_date], columns=model.contracts).iloc[0].fillna(0.0)
        long_cost, short_cost, _, _ = costs.aligned(model.contracts)
        w = selected.to_numpy(float)
        predicted_cost = float(long_cost @ np.maximum(w, 0.0) + short_cost @ np.maximum(-w, 0.0))
        gross_return = float(realized @ selected)
        diag = integer.diagnostics
        stats = result.objective_stats
        return_rows.append(
            {
                "config": label,
                "strategy": R2_NAME,
                "evidence_status": EVIDENCE_STATUS,
                "return_date": return_date,
                "decision_date": decision_date,
                "train_start": train_start,
                "train_end": train_end,
                "train_observations": len(train_dates),
                "recent_observations": min(len(train_dates), config.recent_months),
                "gross_return": gross_return,
                "predicted_cost": predicted_cost,
                "net_return": gross_return - predicted_cost,
                "gross_nav": float(selected.abs().sum()),
                "selected_scale": stats.get("selected_scale", 0.0),
                "expected_net_log_growth": stats.get("expected_net_log_growth", 0.0),
                "robust_sortino": stats.get("robust_sortino", np.nan),
                "recent_covariance_weight": moments.recent_weight,
                "predicted_annual_vol": diag["predicted_annual_vol"],
                "worst_annual_downside": diag["worst_annual_downside"],
                "scenario_cvar_loss": diag["scenario_cvar_loss"],
                "worst_three_month_loss": diag["worst_three_month_loss"],
                "worst_six_month_loss": diag["worst_six_month_loss"],
                "short_margin_used": diag["short_margin_used"],
                "collateral_used": diag["collateral_used"],
                "integer_execution_abstained": diag["integer_execution_abstained"],
                "integer_conversion_feasible": diag["integer_conversion_feasible"],
                "integer_contracts": diag["integer_contracts"],
                "selected_feasible": diag["feasible"],
                "selected_max_breach": diag["max_breach"],
                "rejected_max_breach": diag["rejected_max_breach"],
                "rejected_breach_volatility": diag.get("rejected_breach_volatility", 0.0),
                "rejected_breach_downside": diag.get("rejected_breach_downside", 0.0),
                "rejected_breach_three_month_loss": diag.get("rejected_breach_three_month_loss", 0.0),
                "rejected_breach_six_month_loss": diag.get("rejected_breach_six_month_loss", 0.0),
                "rejected_breach_assignment": diag.get("rejected_breach_assignment", 0.0),
                "information_set_valid": bool(train_end < return_date and decision_date < return_date),
            }
        )
        for contract in model.contracts:
            mark = float(spec.loc[contract, "mark"])
            weight_rows.append(
                {
                    "config": label,
                    "strategy": R2_NAME,
                    "return_date": return_date,
                    "decision_date": decision_date,
                    "asset_id": contract,
                    "underlying": str(spec.loc[contract, "underlying"]),
                    "mark": mark,
                    "continuous_weight": float(result.weights.loc[contract]),
                    "weight": float(selected.loc[contract]),
                    "integer_contracts": int(np.rint(selected.loc[contract] * nav / (100.0 * mark))),
                }
            )
        qlike = moments.qlike_ledger.copy()
        if not qlike.empty:
            qlike["config"] = label
            qlike["return_date"] = return_date
            qlike["decision_date"] = decision_date
            moment_rows.extend(qlike.to_dict("records"))
        if not vol_ledger.empty:
            vol_ledger["config"] = label
            vol_ledger["return_date"] = return_date
            moment_rows.extend(vol_ledger.to_dict("records"))
        for family, frame in families.items():
            scenario_rows.append(
                {
                    "config": label,
                    "return_date": return_date,
                    "family": family,
                    "observations": len(frame),
                    "annual_downside": stats.get(f"downside_{family}", np.nan) * np.sqrt(12.0),
                }
            )
        if diag["integer_execution_abstained"]:
            abstention_rows.append(
                {
                    "config": label,
                    "return_date": return_date,
                    "decision_date": decision_date,
                    "reason": diag["abstention_reason"],
                    **{key: value for key, value in diag.items() if key.startswith("rejected_")},
                }
            )
        component_frame = components.reset_index(names="asset_id")
        component_frame["config"] = label
        component_frame["return_date"] = return_date
        component_frame["decision_date"] = decision_date
        premia_rows.extend(component_frame.to_dict("records"))
    return (
        pd.DataFrame(return_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(moment_rows),
        pd.DataFrame(scenario_rows),
        pd.DataFrame(abstention_rows),
        pd.DataFrame(premia_rows),
    )


def summarize_r2(returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_name, group in returns.groupby("config", observed=True):
        values = group.set_index(pd.to_datetime(group["return_date"]))["net_return"]
        performance = performance_metrics(values)
        survival = survival_diagnostics(
            values,
            margin_breaches=int((group["short_margin_used"] > 0.75 + 1e-8).sum()),
            collateral_breaches=int((group["collateral_used"] > 1.0 + 1e-8).sum()),
            integer_failures=0,
        )
        rows.append(
            {
                "config": config_name,
                "strategy": R2_NAME,
                "evidence_status": EVIDENCE_STATUS,
                "observations": len(group),
                "integer_abstentions": int(group["integer_execution_abstained"].sum()),
                "mean_gross_nav": float(group["gross_nav"].mean()),
                "all_information_sets_valid": bool(group["information_set_valid"].all()),
                "geometric_annual_return": performance.get("annualized_return", np.nan),
                "sortino": performance.get("sortino", np.nan),
                "sharpe": performance.get("sharpe", np.nan),
                "max_drawdown": performance.get("max_drawdown", np.nan),
                "expected_shortfall_95": performance.get("cvar_95", np.nan),
                **survival,
            }
        )
    return pd.DataFrame(rows)


def aligned_r11_comparison(r2_returns: pd.DataFrame, r11_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare R2 with the current direct-or-abstain R1.1 on identical dates."""

    if not r11_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    r11 = pd.read_csv(r11_path, parse_dates=["return_date"])
    r11 = r11[r11["strategy"].eq(R11_NAME)]
    aligned_rows, summary_rows = [], []
    for config_name, r2_group in r2_returns.groupby("config", observed=True):
        left = r2_group[["return_date", "net_return"]].rename(columns={"net_return": "r2_net_return"})
        right = r11[r11["config"].eq(config_name)][["return_date", "net_return"]].rename(columns={"net_return": "r11_net_return"})
        left["return_date"] = pd.to_datetime(left["return_date"])
        pair = left.merge(right, on="return_date", how="inner")
        pair["config"] = config_name
        aligned_rows.append(pair)
        for strategy, column in [(R2_NAME, "r2_net_return"), (R11_NAME, "r11_net_return")]:
            metrics = performance_metrics(pair[column])
            summary_rows.append({"config": config_name, "strategy": strategy, **metrics})
    return (
        pd.concat(aligned_rows, ignore_index=True) if aligned_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


def build_trial_registry(out_dir: Path) -> pd.DataFrame:
    legacy = PAPER / "analysis" / "artifacts" / "r11_higher_risk" / "r11_research_trial_registry.csv"
    registry = pd.read_csv(legacy) if legacy.exists() else pd.DataFrame()
    added = pd.DataFrame(
        [
            {"source": "analysis/r2_robust_sortino_pipeline.py", "trial_key": "R2_recent_weight_025", "count_status": "known_lower_bound"},
            {"source": "analysis/r2_robust_sortino_pipeline.py", "trial_key": "R2_recent_weight_050", "count_status": "known_lower_bound"},
            {"source": "analysis/r2_robust_sortino_pipeline.py", "trial_key": "R2_recent_weight_075", "count_status": "known_lower_bound"},
            *[
                {"source": "src/portfolio/r2_robust_sortino.py", "trial_key": f"R2_HAR_weight_{int(weight * 100):03d}", "count_status": "known_lower_bound"}
                for weight in [0.0, 0.25, 0.50, 0.75, 1.0]
            ],
            {"source": "src/portfolio/r2_robust_sortino.py", "trial_key": "R2_five_residual_imputation_scenarios", "count_status": "known_lower_bound"},
            {"source": "src/portfolio/r2_robust_sortino.py", "trial_key": "R2_robust_sortino_log_growth", "count_status": "known_lower_bound"},
            {"source": "src/portfolio/r2_robust_sortino.py", "trial_key": "R2_direct_integer_or_cash", "count_status": "known_lower_bound"},
            {"source": "analysis/r2_stability.py", "trial_key": "R2_joint_GARCH_repricing", "count_status": "known_lower_bound"},
            {"source": "analysis/r2_stability.py", "trial_key": "R2_Gaussian_copula_sensitivity", "count_status": "known_lower_bound"},
        ]
    )
    registry = pd.concat([registry, added], ignore_index=True)
    registry.to_csv(out_dir / "r2_research_trial_registry.csv", index=False)
    return registry


def write_freeze_manifest(out_dir: Path, data_cutoff: pd.Timestamp) -> dict[str, Any]:
    sources = [
        Path(__file__),
        Path(__file__).with_name("r2_stability.py"),
        ROOT / "src/portfolio/r2_robust_sortino.py",
        Path(__file__).with_name("simulation.py"),
    ]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = {
        "specification": "R2 robust Sortino diagnostic",
        "evidence_status": EVIDENCE_STATUS,
        "data_cutoff": str(pd.Timestamp(data_cutoff).date()),
        "first_eligible_decision_date": str((pd.Timestamp(data_cutoff) + pd.offsets.MonthEnd(1)).date()),
        "confirmatory_observations_required": 36,
        "confirmatory_claim_allowed": False,
        "source_sha256": hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "cvxpy"]
        },
        "parameters": dataclasses.asdict(RobustSortinoConfig()),
        "endpoints": {
            "primary": ["net_log_growth", "zero_target_sortino", "maximum_drawdown", "survival"],
            "simulation": ["p05_terminal_wealth", "p05_sortino", "severe_drawdown_quantile", "refit_coverage"],
            "promotion": "all gates in r2_promotion_gate.json",
        },
        "promotion_policy": "diagnostic unless every historical, bootstrap, repricing, and refit gate passes",
        "vix40_overlay": "unscored_until_complete_executable_OPRA_quotes; excluded_from_R2_returns",
    }
    (out_dir / "r2_prospective_freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Universe & $N$ & Geom. return & Sortino & Max DD & Abstain & Verdict \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row["config"]).replace("_", r"\_"),
                    str(int(row["observations"])),
                    f"{float(row['geometric_annual_return']):.3f}",
                    f"{float(row['sortino']):.3f}",
                    f"{float(row['max_drawdown']):.3f}",
                    str(int(row["integer_abstentions"])),
                    str(row["verdict"]).replace("_", r"\_"),
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
    parser.add_argument("--evaluation-start", default="2018-02-01")
    parser.add_argument("--max-periods", type=int, default=None)
    parser.add_argument("--reuse-core-artifacts", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_map, _ = build_configs()
    selected = list(config_map) if args.configs == "all" else [item.strip() for item in args.configs.split(",")]
    paths = {
        "returns": args.out_dir / "r2_monthly_development_returns.csv",
        "weights": args.out_dir / "r2_monthly_weights.csv",
        "moments": args.out_dir / "r2_moment_ledger.csv",
        "scenarios": args.out_dir / "r2_scenario_diagnostics.csv",
        "abstentions": args.out_dir / "r2_abstention_ledger.csv",
        "premia": args.out_dir / "r2_premia_ledger.csv",
    }
    if args.reuse_core_artifacts:
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise SystemExit(f"missing R2 core artifacts: {missing}")
        returns = pd.read_csv(paths["returns"], parse_dates=["return_date", "decision_date"])
        weights = pd.read_csv(paths["weights"])
        moments = pd.read_csv(paths["moments"])
        scenarios = pd.read_csv(paths["scenarios"])
        abstentions = pd.read_csv(paths["abstentions"])
        premia = pd.read_csv(paths["premia"])
    else:
        daily = load_daily_return_panel(ROOT)
        collections: dict[str, list[pd.DataFrame]] = {key: [] for key in paths}
        for label in selected:
            equities, poc_names, with_vix = config_map[label]
            result = run_r2_config(
                label,
                equities,
                poc_names,
                with_vix,
                daily,
                evaluation_start=args.evaluation_start,
                max_periods=args.max_periods,
            )
            for key, frame in zip(collections, result):
                collections[key].append(frame)
        frames = {
            key: pd.concat(items, ignore_index=True, sort=False) if items else pd.DataFrame()
            for key, items in collections.items()
        }
        returns, weights, moments, scenarios, abstentions, premia = [frames[key] for key in paths]
        for key, path in paths.items():
            frames[key].to_csv(path, index=False)
    summary = summarize_r2(returns)
    summary.to_csv(args.out_dir / "r2_survival_summary.csv", index=False)
    aligned, comparison = aligned_r11_comparison(
        returns, PAPER / "analysis" / "artifacts" / "r11_higher_risk" / "r11_monthly_development_returns.csv"
    )
    aligned.to_csv(args.out_dir / "r2_r11_aligned_returns.csv", index=False)
    comparison.to_csv(args.out_dir / "r2_r11_comparison_summary.csv", index=False)
    build_trial_registry(args.out_dir)
    write_latex_summary(summary, PAPER / "tables" / "short_r2_development_summary.tex")
    cutoff = pd.to_datetime(returns["return_date"]).max() if len(returns) else pd.Timestamp("2026-04-30")
    write_freeze_manifest(args.out_dir, cutoff)
    if not args.skip_stability:
        from research.papers.option_only_markowitz.analysis.r2_stability import run_stability_suites

        run_stability_suites(returns, weights, aligned, args.out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
