"""Locked R2 stability suites and prespecified promotion gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import build_configs
from research.papers.option_only_markowitz.analysis.breadth_robustness_experiment import _iv_level_panel, _vix_level
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import compute_liquidity_caps
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import _build_config_panel
from research.papers.option_only_markowitz.analysis.monte_carlo_repricing import (
    RepriceConfig,
    contract_static_params,
    fit_joint_state_model,
    reprice_contract_returns,
    simulate_state_paths,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import build_optimization_cost_spec
from research.papers.option_only_markowitz.analysis.r11_higher_risk_pipeline import R11_NAME, load_daily_return_panel
from research.papers.option_only_markowitz.analysis.r2_robust_sortino_pipeline import (
    EVIDENCE_STATUS,
    R2_NAME,
    build_r2_model,
)
from research.papers.option_only_markowitz.analysis.run_empirics import ROOT, factor_panels, representative_specs
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics
from research.papers.option_only_markowitz.analysis.vix_option_panel import vix_state_panel
from src.portfolio.r2_robust_sortino import (
    RobustSortinoConfig,
    integerize_r2_direct_or_abstain,
    solve_r2_robust_sortino,
)


def _circular_indices(n: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(length / block)))
    return np.concatenate([(start + np.arange(block)) % n for start in starts])[:length]


def _stationary_indices(n: int, length: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.empty(length, dtype=int)
    indices[0] = rng.integers(0, n)
    restart = 1.0 / float(mean_block)
    for i in range(1, length):
        indices[i] = rng.integers(0, n) if rng.random() < restart else (indices[i - 1] + 1) % n
    return indices


def circular_block_path_suite(
    aligned: pd.DataFrame,
    *,
    paths: int = 5_000,
    block_length: int = 6,
    seed: int = 20260713,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_name, group in aligned.groupby("config", observed=True):
        values = group[["r2_net_return", "r11_net_return"]].dropna().to_numpy(float)
        rng = np.random.default_rng(seed + sum(map(ord, str(config_name))))
        for path_id in range(paths):
            index = _circular_indices(len(values), len(values), block_length, rng)
            for strategy, column in [(R2_NAME, 0), (R11_NAME, 1)]:
                metrics = performance_metrics(values[index, column])
                rows.append(
                    {
                        "config": config_name,
                        "strategy": strategy,
                        "method": "circular_block_6",
                        "path_id": path_id,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def paired_stationary_bootstrap(
    aligned: pd.DataFrame,
    *,
    paths: int = 5_000,
    mean_block: int = 6,
    seed: int = 20260714,
) -> tuple[pd.DataFrame, dict[str, float]]:
    groups = [group[["r2_net_return", "r11_net_return"]].dropna().to_numpy(float) for _, group in aligned.groupby("config", observed=True)]
    rows = []
    rng = np.random.default_rng(seed)
    for path_id in range(paths):
        r2_values, r11_values = [], []
        for values in groups:
            index = _stationary_indices(len(values), len(values), mean_block, rng)
            r2_values.extend(values[index, 0])
            r11_values.extend(values[index, 1])
        r2_metrics = performance_metrics(r2_values)
        r11_metrics = performance_metrics(r11_values)
        r2_log = float(np.mean(np.log1p(np.clip(r2_values, -0.999999, None))))
        r11_log = float(np.mean(np.log1p(np.clip(r11_values, -0.999999, None))))
        rows.append(
            {
                "path_id": path_id,
                "sortino_improvement": float(r2_metrics.get("sortino", np.nan) - r11_metrics.get("sortino", np.nan)),
                "net_log_growth_improvement": r2_log - r11_log,
            }
        )
    frame = pd.DataFrame(rows)
    bounds = {
        "sortino_improvement_90pct_lower": float(frame["sortino_improvement"].quantile(0.05)),
        "net_log_growth_improvement_90pct_lower": float(frame["net_log_growth_improvement"].quantile(0.05)),
    }
    return frame, bounds


def _average_weight_book(weights: pd.DataFrame, config_name: str, strategy: str) -> pd.Series:
    subset = weights[(weights["config"] == config_name) & (weights["strategy"] == strategy)]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.groupby("asset_id", observed=True)["weight"].mean()


def repriced_state_suite(
    r2_weights: pd.DataFrame,
    r2_returns: pd.DataFrame | None = None,
    *,
    paths: int = 2_000,
    horizon: int = 60,
    seed: int = 20260715,
) -> pd.DataFrame:
    """Reprice average implemented R2/R1.1 books under two state generators."""

    r11_path = ROOT / "research/papers/option_only_markowitz/analysis/artifacts/r11_higher_risk/r11_monthly_weights.csv"
    r11_weights = pd.read_csv(r11_path) if r11_path.exists() else pd.DataFrame()
    r11_return_path = ROOT / "research/papers/option_only_markowitz/analysis/artifacts/r11_higher_risk/r11_monthly_development_returns.csv"
    r11_returns = pd.read_csv(r11_return_path) if r11_return_path.exists() else pd.DataFrame()
    r2_return_frame = pd.DataFrame() if r2_returns is None else r2_returns.copy()
    config_map, _ = build_configs()
    rows: list[pd.DataFrame] = []
    for config_number, (config_name, (equities, poc_names, with_vix)) in enumerate(config_map.items()):
        reps, panel_returns, _, universe, _ = _build_config_panel(equities, poc_names, with_vix)
        under, _ = factor_panels(reps, universe)
        state_underlyings = list(universe)
        iv = _iv_level_panel(reps, state_underlyings)
        vix_state = vix_state_panel(panel_returns.index, ROOT)
        vix = _vix_level(vix_state, panel_returns.index).ffill().bfill()
        config = RepriceConfig(
            n_paths=paths,
            n_sensitivity_paths=paths,
            horizon_months=horizon,
            block_length=6,
            seed=seed + config_number,
        )
        state_model = fit_joint_state_model(
            under.reindex(columns=state_underlyings).fillna(0.0),
            iv.reindex(columns=state_underlyings).ffill().bfill(),
            vix,
            config,
        )
        params = contract_static_params(reps, pd.Timestamp(panel_returns.index.max()))
        books = {
            R2_NAME: _average_weight_book(r2_weights, config_name, R2_NAME),
            R11_NAME: _average_weight_book(r11_weights, config_name, R11_NAME),
        }
        costs = {
            R2_NAME: float(r2_return_frame.loc[r2_return_frame["config"].eq(config_name), "predicted_cost"].mean()),
            R11_NAME: float(r11_returns.loc[r11_returns["config"].eq(config_name) & r11_returns["strategy"].eq(R11_NAME), "predicted_cost"].mean()),
        }
        params = params.loc[params.index.intersection(pd.Index(set().union(*(book.index for book in books.values()))))]
        if params.empty:
            continue
        for method in ("joint_garch_block", "gaussian_copula"):
            states = simulate_state_paths(state_model, config, method=method, n_paths=paths)
            contract_returns = reprice_contract_returns(states, params, config)
            for strategy, raw_weights in books.items():
                aligned_weights = raw_weights.reindex(params.index).fillna(0.0)
                net_paths = np.nan_to_num(contract_returns, nan=0.0) @ aligned_weights.to_numpy(float) - costs[strategy]
                records = []
                for path_id in range(paths):
                    records.append(
                        {
                            "config": config_name,
                            "strategy": strategy,
                            "method": method,
                            "path_id": path_id,
                            "predictable_monthly_cost": costs[strategy],
                            **performance_metrics(net_paths[path_id]),
                        }
                    )
                rows.append(pd.DataFrame(records))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def refit_r2_suite(
    *,
    paths: int = 200,
    horizon: int = 60,
    seed: int = 20260716,
    nav: float = 1_000_000.0,
) -> pd.DataFrame:
    """Refit R2 to each joint resampled history rather than applying fixed weights."""

    config_map, _ = build_configs()
    daily = load_daily_return_panel(ROOT)
    risk = RobustSortinoConfig(scalar_grid_points=101)
    rows: list[dict[str, Any]] = []
    for config_number, (config_name, (equities, poc_names, with_vix)) in enumerate(config_map.items()):
        reps, returns, detail, universe, _ = _build_config_panel(equities, poc_names, with_vix)
        under, vol = factor_panels(reps, universe)
        dates = pd.DatetimeIndex(returns.index).sort_values()
        return_date = dates[-1]
        train_dates = dates[:-1]
        period_detail = detail[pd.to_datetime(detail["return_date"]).eq(return_date)]
        decision_date = pd.Timestamp(period_detail["decision_date"].max()) if not period_detail.empty else train_dates[-1]
        spec = representative_specs(reps, returns, train_start=train_dates[0], train_end=decision_date)
        recent = returns.reindex(index=train_dates[-risk.recent_months :], columns=spec.index)
        keep = recent.notna().sum()[lambda value: value >= risk.min_recent_observations].index
        spec = spec.reindex(keep).dropna(subset=["underlying", "mark"])
        original_returns = returns.reindex(index=train_dates, columns=spec.index)
        original_under = under.reindex(index=train_dates, columns=universe)
        original_vol = vol.reindex(index=train_dates, columns=universe)
        cost_config = ResearchCostConfig(nav_for_capacity=nav, impact_cost_rate=0.0, use_current_spread_assumptions=False, use_inferred_spread_proxy=True)
        surface = load_cbbo_spread_surface(ROOT, cost_config.cbbo_spread_surface_path)
        cost_inputs = build_cost_input_ledger(reps, detail, ROOT, cost_config, spread_surface=surface)
        rng = np.random.default_rng(seed + config_number)
        for path_id in range(paths):
            try:
                sample_index = _circular_indices(len(train_dates), len(train_dates), 6, rng)
                pseudo_returns = original_returns.iloc[sample_index].copy()
                pseudo_under = original_under.iloc[sample_index].copy()
                pseudo_vol = original_vol.iloc[sample_index].copy()
                for frame in (pseudo_returns, pseudo_under, pseudo_vol):
                    frame.index = train_dates
                # A refit must rerun contract eligibility on its own resampled
                # information set.  Reusing the original eligible set makes a
                # bootstrap-induced missingness pattern look like a model error.
                path_keep = pseudo_returns.iloc[-risk.recent_months :].notna().sum()
                path_keep = path_keep[path_keep >= risk.min_recent_observations].index
                path_spec = spec.reindex(path_keep).dropna(subset=["underlying", "mark"])
                if path_spec.empty:
                    raise ValueError("no contracts pass the refit 24-of-36 eligibility rule")
                pseudo_returns = pseudo_returns.reindex(columns=path_spec.index)
                model, moments, families, _, _ = build_r2_model(
                    path_spec, pseudo_returns, pseudo_under, pseudo_vol, universe, daily, decision_date, risk
                )
                caps = compute_liquidity_caps(
                    reps, path_spec["mark"], nav, participation=0.05, per_contract_abs=0.18, train_end=decision_date
                )["bound"].reindex(model.contracts).fillna(0.0)
                costs, _ = build_optimization_cost_spec(cost_inputs, model.contracts, decision_date, cost_config)
                solved = solve_r2_robust_sortino(
                    model, moments.option_returns_imputed, families, costs, caps, risk
                )
                integer = integerize_r2_direct_or_abstain(
                    model,
                    solved.weights,
                    path_spec["mark"],
                    nav,
                    caps,
                    moments.option_returns_imputed,
                    families,
                    costs,
                    risk,
                )
                evaluation_index = _circular_indices(len(original_returns), horizon, 6, rng)
                gross = original_returns.iloc[evaluation_index].fillna(0.0).to_numpy(float) @ integer.weights.to_numpy(float)
                long_cost, short_cost, _, _ = costs.aligned(model.contracts)
                w = integer.weights.to_numpy(float)
                cost = float(long_cost @ np.maximum(w, 0.0) + short_cost @ np.maximum(-w, 0.0))
                metrics = performance_metrics(gross - cost)
                rows.append(
                    {
                        "config": config_name,
                        "path_id": path_id,
                        "status": "ok",
                        "integer_abstained": integer.diagnostics["integer_execution_abstained"],
                        **metrics,
                    }
                )
            except Exception as exc:  # full-run audit: preserve every failed refit
                rows.append(
                    {
                        "config": config_name,
                        "path_id": path_id,
                        "status": f"error:{type(exc).__name__}:{exc}",
                        "defaulted": True,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_promotion_gate(
    comparison: pd.DataFrame,
    bootstrap_bounds: dict[str, float],
    repriced: pd.DataFrame,
    refit: pd.DataFrame,
    r2_returns: pd.DataFrame,
) -> dict[str, Any]:
    pivot = comparison.pivot(index="config", columns="strategy")
    historical_wins = 0
    no_material_historical_harm = True
    for config_name in pivot.index:
        r2 = comparison[(comparison["config"] == config_name) & (comparison["strategy"] == R2_NAME)].iloc[0]
        r11 = comparison[(comparison["config"] == config_name) & (comparison["strategy"] == R11_NAME)].iloc[0]
        wins = (
            r2["sortino"] > r11["sortino"]
            and r2["annualized_return"] > r11["annualized_return"]
            and r2["max_drawdown"] > r11["max_drawdown"]
        )
        historical_wins += int(wins)
        no_material_historical_harm &= bool(
            r2["annualized_return"] >= r11["annualized_return"] - 0.01
            and r2["max_drawdown"] >= r11["max_drawdown"] - 0.02
            and r2["cvar_95"] <= r11["cvar_95"] + 0.01
        )
    hard_failures = int(
        (r2_returns["short_margin_used"] > 0.75 + 1e-8).sum()
        + (r2_returns["collateral_used"] > 1.0 + 1e-8).sum()
        + (~r2_returns["selected_feasible"].astype(bool)).sum()
        + (r2_returns["net_return"] <= -1.0).sum()
    )
    repriced_gate = True
    repriced_better = 0
    severe_gate = True
    if repriced.empty:
        repriced_gate = False
        severe_gate = False
    else:
        for config_name, config_group in repriced.groupby("config", observed=True):
            config_no_worse = True
            config_better = True
            for _, group in config_group.groupby("method", observed=True):
                r2 = group[group["strategy"] == R2_NAME]
                r11 = group[group["strategy"] == R11_NAME]
                if r2.empty or r11.empty:
                    config_no_worse = False
                    config_better = False
                    severe_gate = False
                    continue
                r2_terminal, r11_terminal = r2["terminal_wealth"].quantile(0.05), r11["terminal_wealth"].quantile(0.05)
                r2_sortino, r11_sortino = r2["sortino"].quantile(0.05), r11["sortino"].quantile(0.05)
                config_no_worse &= bool(r2_terminal >= r11_terminal and r2_sortino >= r11_sortino)
                config_better &= bool(r2_terminal > r11_terminal and r2_sortino > r11_sortino)
                severe_gate &= bool(r2["max_drawdown"].quantile(0.05) >= r11["max_drawdown"].quantile(0.05) - 0.02)
            repriced_gate &= config_no_worse
            repriced_better += int(config_better)
    valid_refit = float(refit["status"].eq("ok").mean()) if len(refit) else 0.0
    refit_survival = bool(
        len(refit)
        and refit["status"].eq("ok").all()
        and not refit["defaulted"].fillna(False).astype(bool).any()
    )
    gates = {
        "zero_hard_failures": hard_failures == 0,
        "historical_three_of_four": historical_wins >= 3,
        "no_material_historical_harm": no_material_historical_harm,
        "stationary_sortino_lower_positive": bootstrap_bounds["sortino_improvement_90pct_lower"] > 0.0,
        "stationary_log_growth_lower_nonnegative": bootstrap_bounds["net_log_growth_improvement_90pct_lower"] >= 0.0,
        "repriced_p05_no_worse_everywhere": repriced_gate,
        "repriced_p05_better_three_of_four": repriced_better >= 3,
        "severe_drawdown_within_two_points": severe_gate,
        "refit_coverage_at_least_95pct": valid_refit >= 0.95,
        "refit_no_defaults": refit_survival,
    }
    promoted = all(gates.values())
    return {
        "specification": "R2 robust Sortino",
        "evidence_status": EVIDENCE_STATUS,
        "promoted": promoted,
        "active_development_extension": "R2" if promoted else "R1.1",
        "historical_universe_wins": historical_wins,
        "repriced_universe_wins": repriced_better,
        "valid_refit_coverage": valid_refit,
        "hard_failures": hard_failures,
        "gates": gates,
        "bootstrap_bounds": bootstrap_bounds,
    }


def run_stability_suites(
    r2_returns: pd.DataFrame,
    r2_weights: pd.DataFrame,
    aligned: pd.DataFrame,
    out_dir: Path,
    *,
    block_paths: int = 5_000,
    repriced_paths: int = 2_000,
    refit_paths: int = 200,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    block = circular_block_path_suite(aligned, paths=block_paths)
    block.to_csv(out_dir / "r2_block_bootstrap_paths.csv", index=False)
    paired, bounds = paired_stationary_bootstrap(aligned, paths=block_paths)
    paired.to_csv(out_dir / "r2_paired_stationary_bootstrap.csv", index=False)
    repriced = repriced_state_suite(r2_weights, r2_returns, paths=repriced_paths)
    repriced.to_csv(out_dir / "r2_repriced_state_paths.csv", index=False)
    refit = refit_r2_suite(paths=refit_paths)
    refit.to_csv(out_dir / "r2_refit_monte_carlo.csv", index=False)
    comparison_path = out_dir / "r2_r11_comparison_summary.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    gate = evaluate_promotion_gate(comparison, bounds, repriced, refit, r2_returns)
    (out_dir / "r2_promotion_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


__all__ = [
    "circular_block_path_suite",
    "paired_stationary_bootstrap",
    "repriced_state_suite",
    "refit_r2_suite",
    "evaluate_promotion_gate",
    "run_stability_suites",
]
