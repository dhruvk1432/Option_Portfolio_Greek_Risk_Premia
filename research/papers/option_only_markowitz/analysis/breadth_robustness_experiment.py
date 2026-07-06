"""Robustness validation for the breadth/capacity production candidates.

The older ``run_empirics --stage robustness`` path validates the paper's
original strategy registry.  This module validates the four breadth/capacity
universes introduced later, using the corrected historical/inferred CBBO cost
stack and the locked E1 estimator/cap rule.  It intentionally writes to a
separate artifact directory so legacy paper artifacts remain reproducible.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import (
    OUT_DIR,
    build_configs,
)
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    EstimatorKnobs,
    TrainingContext,
    cap_feasibility,
    capped_naive_weights,
    compute_liquidity_caps,
    naive_weights,
    rebuild_model,
    solve_gm,
    spread_source_coverage,
)
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import (
    _build_config_panel,
)
from research.papers.option_only_markowitz.analysis.cross_validation import (
    CVConfig,
    FoldSpec,
    assemble_cpcv_paths,
    build_folds,
    probability_of_backtest_overfitting,
)
from research.papers.option_only_markowitz.analysis.inference import (
    BootstrapConfig,
    circular_block_sample,
    sharpe_reality_check,
)
from research.papers.option_only_markowitz.analysis.monte_carlo_repricing import (
    RepriceConfig,
    contract_static_params,
    fit_joint_state_model,
    reprice_assumptions,
    reprice_contract_returns,
    simulate_state_paths,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.resampled_universes import (
    ResampleConfig,
    fixed_weight_universe_distribution,
    month_index_paths,
    resampled_summary,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    MIN_OPTION_MARK,
    PRIMARY_UNDERLYINGS,
    ROOT,
    TABLE_DIR,
    TRAIN_END,
    VIX_FACTOR,
    factor_panels,
    make_model,
    representative_specs,
    vix_state_panel,
)
from research.papers.option_only_markowitz.analysis.simulation import (
    SimulationConfig,
    performance_metrics,
    run_tail_path_simulations,
)


ROBUSTNESS_DIR = OUT_DIR / "robustness"
CONFIG_ORDER = ["orig", "orig+VIX", "larger", "larger+VIX"]
PRIMARY_STRATEGY = "E1 capped"
BENCHMARK_STRATEGIES = ("GM paper", "Equal premium capped", "Equal risk capped")
STRATEGY_ORDER = (PRIMARY_STRATEGY, *BENCHMARK_STRATEGIES)
E1_KNOBS = EstimatorKnobs(
    residual_estimator="diag",
    cov_shrinkage="n_scaled",
    historical_weight=0.0,
    shrinkage_to_zero=0.75,
)
DEFAULT_NAV = 1_000_000.0
DEFAULT_PARTICIPATION = 0.05
DEFAULT_CV_CONFIG = CVConfig(n_groups=12, n_test_groups=2, purge_months=1, embargo_months=1)


@dataclass
class BreadthPanel:
    label: str
    universe: list[str]
    reps: pd.DataFrame
    returns: pd.DataFrame
    detail: pd.DataFrame
    has_vix: bool


@dataclass
class FittedBook:
    config: str
    strategy: str
    display_strategy: str
    weights: pd.Series
    model_contracts: pd.Index
    solver_status: str
    mode: str
    capacity_infeasible: bool
    sum_of_caps: float
    deployed_gross: float

    @property
    def full_name(self) -> str:
        return f"{self.config} {self.strategy}"


@dataclass
class FullContext:
    panel: BreadthPanel
    training: TrainingContext
    cost_inputs: pd.DataFrame
    spread_coverage: pd.DataFrame
    books: dict[str, FittedBook]
    gross_returns: pd.DataFrame
    net_returns: pd.DataFrame
    underlying_returns: pd.DataFrame
    vol_shocks: pd.DataFrame


def breadth_cost_config(nav: float = DEFAULT_NAV) -> ResearchCostConfig:
    """Cost policy used by every breadth robustness stage."""

    return ResearchCostConfig(
        nav_for_capacity=float(nav),
        use_current_spread_assumptions=False,
        use_inferred_spread_proxy=True,
    )


def expected_cv_split_count(config: CVConfig = DEFAULT_CV_CONFIG) -> int:
    return int(config.n_groups + math.comb(config.n_groups, config.n_test_groups))


def spread_policy_status(spread_coverage: pd.DataFrame) -> dict[str, object]:
    """Return a fail-closed spread-source policy audit."""

    if spread_coverage.empty:
        return {
            "status": "fail",
            "current_cboe_rows": 0,
            "default_rows": 0,
            "message": "empty spread coverage",
        }
    src = spread_coverage.get("relative_spread_source", pd.Series(dtype=object)).astype(str)
    rows = pd.to_numeric(spread_coverage.get("rows", 0), errors="coerce").fillna(0).astype(int)
    current_rows = int(rows[src.eq("current_cboe_liquid_quote")].sum())
    default_rows = int(rows[src.eq("default")].sum())
    status = "pass" if current_rows == 0 and default_rows == 0 else "fail"
    return {
        "status": status,
        "current_cboe_rows": current_rows,
        "default_rows": default_rows,
        "message": "ok" if status == "pass" else "stale/current/default spread source present",
    }


def build_panels(selected_configs: Sequence[str]) -> dict[str, BreadthPanel]:
    configs, _present_new = build_configs()
    out: dict[str, BreadthPanel] = {}
    for label in selected_configs:
        underlyings, poc_names, with_vix = configs[label]
        print(f"[breadth-robustness] building panel {label}", flush=True)
        reps, returns, detail, universe, has_vix = _build_config_panel(underlyings, poc_names, with_vix)
        out[label] = BreadthPanel(
            label=label,
            universe=list(universe),
            reps=reps.copy(),
            returns=returns.copy(),
            detail=detail.copy(),
            has_vix=bool(has_vix),
        )
    return out


def build_training_from_panel(
    panel: BreadthPanel,
    *,
    train_dates: Sequence[pd.Timestamp] | None = None,
    min_contracts: int = 8,
    min_obs_per_contract: int = 18,
) -> tuple[TrainingContext | None, str]:
    returns = panel.returns.copy()
    reps = panel.reps.copy()
    if train_dates is None:
        train_index = pd.DatetimeIndex(returns.loc[returns.index <= TRAIN_END].index)
        if len(train_index) == 0:
            return None, "skipped_no_train_dates"
        spec = representative_specs(reps, returns)
        if len(spec) < int(min_contracts):
            return None, "skipped_too_few_specs"
        returns_model = returns.reindex(columns=spec.index).dropna(how="all")
        model, residuals = make_model(spec, returns_model, reps, panel.universe)
        aligned_train = returns_model.loc[returns_model.index <= TRAIN_END, spec.index].dropna(how="all")
        under_ret, vol_shocks = factor_panels(reps, panel.universe)
        train_under = under_ret.reindex(aligned_train.index).fillna(0.0)
        train_vol = vol_shocks.reindex(aligned_train.index).fillna(0.0)
        return (
            TrainingContext(
                label=panel.label,
                universe=list(panel.universe),
                reps=reps.copy(),
                returns=returns_model,
                detail=panel.detail[panel.detail["asset_id"].isin(spec.index)].copy(),
                spec=spec,
                base_model=model,
                residuals=residuals,
                train_returns=aligned_train,
                train_under=train_under,
                train_vol=train_vol,
            ),
            "ok",
        )
    else:
        train_index = pd.DatetimeIndex(pd.to_datetime(pd.Index(train_dates))).sort_values()
    if len(train_index) == 0:
        return None, "skipped_no_train_dates"

    train_returns = returns.loc[train_index].dropna(how="all")
    columns = train_returns.columns[train_returns.count() >= int(min_obs_per_contract)]
    if len(columns) < int(min_contracts):
        return None, "skipped_too_few_contracts"
    train_returns = train_returns.reindex(columns=columns)
    sub_reps = reps[
        reps["snap_date"].isin(train_index)
        & reps["asset_id"].isin(columns)
    ].copy()
    if sub_reps.empty:
        return None, "skipped_no_reps"
    train_start = pd.Timestamp(train_index.min())
    train_end = pd.Timestamp(train_index.max())
    spec = representative_specs(sub_reps, train_returns, train_start=train_start, train_end=train_end)
    if len(spec) < int(min_contracts):
        return None, "skipped_too_few_specs"
    train_returns = train_returns.reindex(columns=spec.index).dropna(how="all")
    model, residuals = make_model(
        spec,
        train_returns,
        sub_reps,
        panel.universe,
        train_start=train_start,
        train_end=train_end,
    )
    under_ret, vol_shocks = factor_panels(sub_reps, panel.universe)
    aligned_train = train_returns.loc[:, spec.index].dropna(how="all")
    train_under = under_ret.reindex(aligned_train.index).fillna(0.0)
    train_vol = vol_shocks.reindex(aligned_train.index).fillna(0.0)
    return (
        TrainingContext(
            label=panel.label,
            universe=list(panel.universe),
            reps=sub_reps,
            returns=returns.reindex(columns=spec.index).dropna(how="all"),
            detail=panel.detail[panel.detail["asset_id"].isin(spec.index)].copy(),
            spec=spec,
            base_model=model,
            residuals=residuals,
            train_returns=aligned_train,
            train_under=train_under,
            train_vol=train_vol,
        ),
        "ok",
    )


def build_cost_inputs(panel: BreadthPanel, nav: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = breadth_cost_config(nav)
    surface = load_cbbo_spread_surface(ROOT, cfg.cbbo_spread_surface_path) if cfg.use_cbbo_spread_surface else None
    cost_inputs = build_cost_input_ledger(panel.reps, panel.detail, ROOT, cfg, spread_surface=surface)
    return cost_inputs, spread_source_coverage(panel.label, cost_inputs)


def fit_books(
    ctx: TrainingContext,
    *,
    nav: float,
    participation: float,
    strategy_prefix: str | None = None,
) -> dict[str, FittedBook]:
    prefix = strategy_prefix or ctx.label
    books: dict[str, FittedBook] = {}

    paper_weights, paper_status = solve_gm(ctx.base_model, "cvxpy")
    books["GM paper"] = FittedBook(
        config=prefix,
        strategy="GM paper",
        display_strategy="GM paper",
        weights=paper_weights,
        model_contracts=pd.Index(ctx.base_model.contracts),
        solver_status=paper_status,
        mode="uncapped",
        capacity_infeasible=False,
        sum_of_caps=np.nan,
        deployed_gross=float(paper_weights.abs().sum()),
    )

    caps_df = compute_liquidity_caps(
        ctx.reps,
        ctx.spec["mark"],
        nav=float(nav),
        participation=float(participation),
        train_end=pd.Timestamp(ctx.train_returns.index.max()),
    )
    feasibility = cap_feasibility(caps_df, ctx.base_model.constraints)
    caps = caps_df["bound"]
    e1_model = rebuild_model(ctx, E1_KNOBS, per_contract_caps=caps)
    e1_weights, e1_status = solve_gm(e1_model, "cvxpy")
    books[PRIMARY_STRATEGY] = FittedBook(
        config=prefix,
        strategy=PRIMARY_STRATEGY,
        display_strategy=PRIMARY_STRATEGY,
        weights=e1_weights,
        model_contracts=pd.Index(e1_model.contracts),
        solver_status=e1_status,
        mode="hard",
        capacity_infeasible=not bool(feasibility["gross_feasible"]),
        sum_of_caps=float(feasibility["sum_of_caps"]),
        deployed_gross=float(e1_weights.abs().sum()),
    )

    gross_feasible = bool(feasibility["gross_feasible"])
    naive_mode = "hard" if gross_feasible else "relaxed"
    target_gross = (
        float(ctx.base_model.constraints.gross_nav)
        if gross_feasible
        else float(feasibility["suggested_gross"])
    )
    for naive_name, naive_weights_raw in naive_weights(ctx.base_model).items():
        strategy = f"{naive_name} capped"
        weights = capped_naive_weights(naive_weights_raw, caps, target_gross)
        books[strategy] = FittedBook(
            config=prefix,
            strategy=strategy,
            display_strategy=strategy,
            weights=weights,
            model_contracts=pd.Index(ctx.base_model.contracts),
            solver_status="reference",
            mode=naive_mode,
            capacity_infeasible=not gross_feasible,
            sum_of_caps=float(feasibility["sum_of_caps"]),
            deployed_gross=float(weights.abs().sum()),
        )

    return {name: books[name] for name in STRATEGY_ORDER if name in books}


def score_books(
    panel: BreadthPanel,
    books: dict[str, FittedBook],
    cost_inputs: pd.DataFrame,
    *,
    nav: float,
    dates: Sequence[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if dates is None:
        eval_dates = pd.DatetimeIndex(panel.returns.index)
    else:
        eval_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).sort_values()
    eval_dates = eval_dates.intersection(pd.DatetimeIndex(panel.returns.index))
    gross = pd.DataFrame(index=eval_dates)
    raw_strategies: dict[str, pd.Series] = {}
    for key, book in books.items():
        cols = pd.Index(book.model_contracts).intersection(panel.returns.columns)
        frame = panel.returns.loc[eval_dates, cols].fillna(0.0)
        weights = book.weights.reindex(cols).fillna(0.0)
        gross[key] = frame.to_numpy(float) @ weights.to_numpy(float)
        raw_strategies[key] = book.weights
    if gross.empty:
        return gross, gross.copy(), pd.DataFrame(), pd.DataFrame()
    cfg = breadth_cost_config(nav)
    net, cost_ledger, capacity_ledger, *_ = compute_strategy_cost_ledgers(
        gross,
        raw_strategies,
        cost_inputs,
        cfg,
    )
    return gross, net, cost_ledger, capacity_ledger


def build_full_context(panel: BreadthPanel, nav: float, participation: float) -> FullContext:
    ctx, status = build_training_from_panel(panel)
    if ctx is None:
        raise RuntimeError(f"{panel.label} context failed: {status}")
    cost_inputs, spread_cov = build_cost_inputs(panel, nav)
    books = fit_books(ctx, nav=nav, participation=participation)
    test_dates = pd.DatetimeIndex(panel.returns.index[panel.returns.index > TRAIN_END])
    gross, net, _cost_ledger, _capacity_ledger = score_books(
        panel,
        books,
        cost_inputs,
        nav=nav,
        dates=test_dates,
    )
    gross.columns = [f"{panel.label} {c}" for c in gross.columns]
    net.columns = [f"{panel.label} {c}" for c in net.columns]
    under_ret, vol_shocks = factor_panels(panel.reps, panel.universe)
    return FullContext(
        panel=panel,
        training=ctx,
        cost_inputs=cost_inputs,
        spread_coverage=spread_cov,
        books=books,
        gross_returns=gross,
        net_returns=net,
        underlying_returns=under_ret,
        vol_shocks=vol_shocks,
    )


def _series_stats(series: pd.Series) -> dict[str, float | bool]:
    stats = performance_metrics(pd.Series(series).dropna(), 12)
    return {
        "sharpe": float(stats.get("sharpe", np.nan)),
        "sortino": float(stats.get("sortino", np.nan)),
        "ann_return": float(stats.get("annualized_return", np.nan)),
        "ann_vol": float(stats.get("annualized_volatility", np.nan)),
        "max_drawdown": float(stats.get("max_drawdown", np.nan)),
        "terminal_wealth": float(stats.get("terminal_wealth", np.nan)),
        "defaulted": bool(stats.get("defaulted", False)),
    }


def _append_fold_metric(
    rows: list[dict[str, object]],
    *,
    config: str,
    fold: FoldSpec,
    strategy: str,
    basis: str,
    series: pd.Series,
    status: str,
    book: FittedBook,
) -> None:
    stats = _series_stats(series)
    rows.append(
        {
            "config": config,
            "fold_id": fold.fold_id,
            "scheme": fold.scheme,
            "strategy": f"{config} {strategy}",
            "display_strategy": strategy,
            "basis": basis,
            "status": status,
            "solver_status": book.solver_status,
            "mode": book.mode,
            "capacity_infeasible": bool(book.capacity_infeasible),
            "sum_of_caps": book.sum_of_caps,
            "deployed_gross": book.deployed_gross,
            **stats,
        }
    )


def _append_month_rows(
    rows: list[dict[str, object]],
    *,
    config: str,
    fold: FoldSpec,
    strategy: str,
    basis: str,
    series: pd.Series,
) -> None:
    for dt, value in pd.Series(series).dropna().items():
        rows.append(
            {
                "config": config,
                "fold_id": fold.fold_id,
                "scheme": fold.scheme,
                "return_date": pd.Timestamp(dt),
                "strategy": f"{config} {strategy}",
                "display_strategy": strategy,
                "basis": basis,
                "ret": float(value),
            }
        )


def run_cv_stage(
    panels: dict[str, BreadthPanel],
    full_contexts: dict[str, FullContext],
    *,
    cv_config: CVConfig,
    nav: float,
    participation: float,
) -> dict[str, pd.DataFrame]:
    schedule_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    expected = expected_cv_split_count(cv_config)

    for config, panel in panels.items():
        folds = build_folds(panel.returns.index, cv_config, "kfold") + build_folds(panel.returns.index, cv_config, "cpcv")
        print(f"[breadth-robustness] CV {config}: {len(folds)} folds", flush=True)
        for fold in folds:
            started = time.monotonic()
            status = "ok"
            books: dict[str, FittedBook] = {}
            try:
                ctx, fit_status = build_training_from_panel(
                    panel,
                    train_dates=fold.train_dates,
                    min_contracts=cv_config.min_contracts,
                    min_obs_per_contract=cv_config.min_obs_per_contract,
                )
                status = fit_status
                if ctx is not None and fit_status == "ok":
                    books = fit_books(ctx, nav=nav, participation=participation, strategy_prefix=config)
                    train_gross, train_net, _tcost, _tcap = score_books(
                        panel,
                        books,
                        full_contexts[config].cost_inputs,
                        nav=nav,
                        dates=fold.train_dates,
                    )
                    test_gross, test_net, _cost, _cap = score_books(
                        panel,
                        books,
                        full_contexts[config].cost_inputs,
                        nav=nav,
                        dates=fold.test_dates,
                    )
                    for strategy, book in books.items():
                        for basis, train_frame, test_frame in [
                            ("gross", train_gross, test_gross),
                            ("full_cost_net", train_net, test_net),
                        ]:
                            if strategy not in test_frame:
                                continue
                            _append_fold_metric(
                                ledger_rows,
                                config=config,
                                fold=fold,
                                strategy=strategy,
                                basis=basis,
                                series=test_frame[strategy],
                                status=status,
                                book=book,
                            )
                            _append_month_rows(
                                month_rows,
                                config=config,
                                fold=fold,
                                strategy=strategy,
                                basis=basis,
                                series=test_frame[strategy],
                            )
                            is_stats = _series_stats(train_frame[strategy]) if strategy in train_frame else {}
                            oos_stats = _series_stats(test_frame[strategy])
                            split_rows.append(
                                {
                                    "config": config,
                                    "fold_id": fold.fold_id,
                                    "scheme": fold.scheme,
                                    "strategy": f"{config} {strategy}",
                                    "display_strategy": strategy,
                                    "basis": basis,
                                    "is_sharpe": is_stats.get("sharpe", np.nan),
                                    "oos_sharpe": oos_stats.get("sharpe", np.nan),
                                    "status": status,
                                }
                            )
            except Exception as exc:  # pragma: no cover - exercised in full runs on solver/data surprises
                status = f"error:{type(exc).__name__}"
            runtime_rows.append(
                {
                    "config": config,
                    "fold_id": fold.fold_id,
                    "seconds": time.monotonic() - started,
                    "status": status,
                }
            )
            schedule_rows.append(
                {
                    "config": config,
                    "fold_id": fold.fold_id,
                    "scheme": fold.scheme,
                    "test_groups": "_".join(str(g) for g in fold.test_groups),
                    "test_start": min(fold.test_dates) if fold.test_dates else pd.NaT,
                    "test_end": max(fold.test_dates) if fold.test_dates else pd.NaT,
                    "n_train": int(len(fold.train_dates)),
                    "n_test": int(len(fold.test_dates)),
                    "n_purged": int(len(fold.purged_dates)),
                    "n_embargoed": int(len(fold.embargoed_dates)),
                    "status": status,
                    "expected_splits_per_config": expected,
                }
            )
            done = sum(1 for row in schedule_rows if row["config"] == config)
            if done == 1 or done == len(folds) or done % 10 == 0:
                print(
                    f"[breadth-robustness] CV {config}: completed {done}/{len(folds)} folds "
                    f"(last={fold.fold_id}, status={status})",
                    flush=True,
                )

    fold_schedule = pd.DataFrame(schedule_rows)
    fold_ledger = pd.DataFrame(ledger_rows)
    split_is_oos = pd.DataFrame(split_rows)
    test_month_returns = pd.DataFrame(month_rows)
    cpcv_returns, cpcv_metrics = assemble_cpcv_paths(test_month_returns, cv_config)
    pbo = _pbo_with_scopes(split_is_oos)
    return {
        "fold_schedule": fold_schedule,
        "fold_ledger": fold_ledger,
        "split_is_oos": split_is_oos,
        "test_month_returns": test_month_returns,
        "cpcv_path_month_returns": cpcv_returns,
        "cpcv_path_metrics": cpcv_metrics,
        "pbo_summary": pbo,
        "runtime_log": pd.DataFrame(runtime_rows),
    }


def _pbo_with_scopes(split_is_oos: pd.DataFrame) -> pd.DataFrame:
    if split_is_oos.empty:
        return pd.DataFrame()
    frames = []
    for basis in sorted(split_is_oos["basis"].dropna().astype(str).unique()):
        overall = probability_of_backtest_overfitting(split_is_oos, basis=basis)
        if not overall.empty:
            overall.insert(0, "scope", "all_configs")
            overall.insert(1, "config", "all")
            frames.append(overall)
        for config, group in split_is_oos.groupby("config", dropna=False):
            one = probability_of_backtest_overfitting(group, basis=basis)
            if not one.empty:
                one.insert(0, "scope", "within_config")
                one.insert(1, "config", config)
                frames.append(one)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def realized_summary(full_contexts: dict[str, FullContext]) -> pd.DataFrame:
    rows = []
    for config, ctx in full_contexts.items():
        for strategy in STRATEGY_ORDER:
            col = f"{config} {strategy}"
            if col not in ctx.gross_returns or col not in ctx.net_returns:
                continue
            book = ctx.books[strategy]
            g = _series_stats(ctx.gross_returns[col])
            n = _series_stats(ctx.net_returns[col])
            rows.append(
                {
                    "config": config,
                    "strategy": strategy,
                    "deployable": bool(not book.capacity_infeasible and book.solver_status != "infeasible"),
                    "mode": book.mode,
                    "solver_status": book.solver_status,
                    "capacity_infeasible": bool(book.capacity_infeasible),
                    "sum_of_caps": book.sum_of_caps,
                    "deployed_gross": book.deployed_gross,
                    "gross_sharpe": g["sharpe"],
                    "net_sharpe": n["sharpe"],
                    "gross_sortino": g["sortino"],
                    "net_sortino": n["sortino"],
                    "gross_max_drawdown": g["max_drawdown"],
                    "net_max_drawdown": n["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def run_mc_resampled(
    gross_returns: pd.DataFrame,
    net_returns: pd.DataFrame,
    config: ResampleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    paths = month_index_paths(len(gross_returns), config.n_paths, config.block_length, rng)
    frames = [
        fixed_weight_universe_distribution(
            gross_returns,
            paths,
            basis="gross",
            universe_family="resampled",
            periods_per_year=config.periods_per_year,
        ),
        fixed_weight_universe_distribution(
            net_returns,
            paths,
            basis="full_cost_net",
            universe_family="resampled",
            periods_per_year=config.periods_per_year,
        ),
    ]
    path_frame = pd.concat(frames, ignore_index=True, sort=False)
    realized = pd.concat(
        [
            _realized_sharpe_rows(gross_returns, basis="gross", universe_family="resampled"),
            _realized_sharpe_rows(net_returns, basis="full_cost_net", universe_family="resampled"),
        ],
        ignore_index=True,
        sort=False,
    )
    return path_frame, resampled_summary(path_frame, realized)


def run_mc_refit(
    panels: dict[str, BreadthPanel],
    full_contexts: dict[str, FullContext],
    *,
    config: ResampleConfig,
    nav: float,
    participation: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for label, panel in panels.items():
        train_dates = pd.DatetimeIndex(panel.returns.loc[panel.returns.index <= TRAIN_END].index)
        if len(train_dates) == 0:
            continue
        rng = np.random.default_rng(config.refit_seed)
        paths = month_index_paths(len(train_dates), config.n_refit_paths, config.block_length, rng)
        train_returns = panel.returns.loc[train_dates]
        under_ret, vol_shocks = factor_panels(panel.reps, panel.universe)
        train_under = under_ret.reindex(train_dates).fillna(0.0)
        train_vol = vol_shocks.reindex(train_dates).fillna(0.0)
        test_dates = pd.DatetimeIndex(panel.returns.index[panel.returns.index > TRAIN_END])
        for path_id, path in enumerate(paths):
            if path_id == 0 or (path_id + 1) % 25 == 0 or path_id + 1 == len(paths):
                print(
                    f"[breadth-robustness] refit MC {label}: path {path_id + 1}/{len(paths)}",
                    flush=True,
                )
            try:
                pseudo_returns = _slot_relabel(train_returns, path, train_dates)
                pseudo_under = _slot_relabel(train_under, path, train_dates)
                pseudo_vol = _slot_relabel(train_vol, path, train_dates)
                ctx, status = _training_from_pseudo(
                    panel,
                    pseudo_returns,
                    pseudo_under,
                    pseudo_vol,
                    train_dates,
                )
                if ctx is None:
                    raise RuntimeError(status)
                books = fit_books(ctx, nav=nav, participation=participation, strategy_prefix=label)
                gross, net, _cost, _cap = score_books(
                    panel,
                    books,
                    full_contexts[label].cost_inputs,
                    nav=nav,
                    dates=test_dates,
                )
                for basis, frame in [("gross", gross), ("full_cost_net", net)]:
                    for strategy in STRATEGY_ORDER:
                        if strategy not in frame:
                            continue
                        stats = _series_stats(frame[strategy])
                        book = books[strategy]
                        rows.append(
                            {
                                "config": label,
                                "path_id": int(path_id),
                                "strategy": f"{label} {strategy}",
                                "display_strategy": strategy,
                                "basis": basis,
                                "status": "ok",
                                "capacity_infeasible": bool(book.capacity_infeasible),
                                "deployed_gross": book.deployed_gross,
                                **stats,
                            }
                        )
            except Exception as exc:  # pragma: no cover - full-run guardrail
                rows.append(
                    {
                        "config": label,
                        "path_id": int(path_id),
                        "strategy": f"{label} {PRIMARY_STRATEGY}",
                        "display_strategy": PRIMARY_STRATEGY,
                        "basis": "gross",
                        "status": f"error:{type(exc).__name__}",
                    }
                )
    paths_frame = pd.DataFrame(rows)
    return paths_frame, _refit_summary(paths_frame)


def _training_from_pseudo(
    panel: BreadthPanel,
    pseudo_returns: pd.DataFrame,
    pseudo_under: pd.DataFrame,
    pseudo_vol: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    min_contracts: int = 8,
) -> tuple[TrainingContext | None, str]:
    train_start = pd.Timestamp(train_dates.min())
    train_end = pd.Timestamp(train_dates.max())
    reps_train = panel.reps[
        panel.reps["snap_date"].isin(train_dates)
        & panel.reps["asset_id"].isin(pseudo_returns.columns)
    ].copy()
    spec = representative_specs(reps_train, pseudo_returns, train_start=train_start, train_end=train_end)
    if len(spec) < min_contracts:
        return None, "skipped_too_few_specs"
    pseudo_returns = pseudo_returns.reindex(columns=spec.index).dropna(how="all")
    model, residuals = make_model(
        spec,
        pseudo_returns,
        reps_train,
        panel.universe,
        train_start=train_start,
        train_end=train_end,
        under_ret=pseudo_under,
        vol_shocks=pseudo_vol,
    )
    return (
        TrainingContext(
            label=panel.label,
            universe=list(panel.universe),
            reps=reps_train,
            returns=panel.returns.reindex(columns=spec.index).dropna(how="all"),
            detail=panel.detail[panel.detail["asset_id"].isin(spec.index)].copy(),
            spec=spec,
            base_model=model,
            residuals=residuals,
            train_returns=pseudo_returns,
            train_under=pseudo_under.reindex(pseudo_returns.index).fillna(0.0),
            train_vol=pseudo_vol.reindex(pseudo_returns.index).fillna(0.0),
        ),
        "ok",
    )


def run_mc_repriced(
    full_contexts: dict[str, FullContext],
    *,
    config: RepriceConfig,
    resample_config: ResampleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_paths: list[pd.DataFrame] = []
    all_assumptions: list[pd.DataFrame] = []
    realized = pd.concat(
        [
            _realized_sharpe_rows(
                pd.concat([ctx.gross_returns for ctx in full_contexts.values()], axis=1),
                basis="gross",
                universe_family="repriced",
            ),
            _realized_sharpe_rows(
                pd.concat([ctx.net_returns for ctx in full_contexts.values()], axis=1),
                basis="full_cost_net",
                universe_family="repriced",
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    for label, ctx in full_contexts.items():
        state_underlyings = list(dict.fromkeys(list(ctx.panel.universe) + ([VIX_FACTOR] if ctx.panel.has_vix else [])))
        state_under_ret = ctx.underlying_returns.loc[:TRAIN_END].reindex(columns=state_underlyings).fillna(0.0)
        vix_state = vix_state_panel(ctx.panel.returns.index, ROOT)
        vix_level = _vix_level(vix_state, ctx.panel.returns.index).loc[:TRAIN_END].ffill().bfill()
        iv_levels = _iv_level_panel(ctx.panel.reps, state_underlyings)
        state_model = fit_joint_state_model(
            state_under_ret,
            iv_levels.loc[:TRAIN_END].reindex(columns=state_underlyings).ffill().bfill(),
            vix_level,
            config,
        )
        params = contract_static_params(ctx.panel.reps, TRAIN_END)
        contracts = pd.Index(ctx.training.base_model.contracts)
        params = params.loc[params.index.intersection(contracts)] if not params.empty else params
        if params.empty:
            continue
        states = simulate_state_paths(state_model, config, method="joint_garch_block")
        contract_returns = reprice_contract_returns(states, params, config)
        strategy_weights = {f"{label} {name}": book.weights for name, book in ctx.books.items()}
        gross_paths = _repriced_paths_with_basis(
            contract_returns,
            strategy_weights,
            params.index,
            basis="gross",
            cost_overlay=None,
            block_length=resample_config.block_length,
            seed=config.seed,
        )
        cost_overlay = {
            f"{label} {name}": (
                ctx.gross_returns[f"{label} {name}"] - ctx.net_returns[f"{label} {name}"]
            ).dropna()
            for name in ctx.books
            if f"{label} {name}" in ctx.gross_returns and f"{label} {name}" in ctx.net_returns
        }
        net_paths = _repriced_paths_with_basis(
            contract_returns,
            strategy_weights,
            params.index,
            basis="full_cost_net_overlay",
            cost_overlay=cost_overlay,
            block_length=resample_config.block_length,
            seed=config.seed + 17,
        )
        all_paths.extend([gross_paths, net_paths])
        assumptions = reprice_assumptions(state_model, params, config, "joint_garch_block")
        assumptions.insert(0, "config", label)
        assumptions["Cost overlay"] = "gross structural repricing; net rows subtract resampled realized full-cost drag, not synthetic NBBO"
        all_assumptions.append(assumptions)
    paths = pd.concat(all_paths, ignore_index=True, sort=False) if all_paths else pd.DataFrame()
    summary = _repriced_summary_with_basis(paths, realized)
    return paths, summary, pd.concat(all_assumptions, ignore_index=True, sort=False) if all_assumptions else pd.DataFrame()


def run_rolling_oos(
    panels: dict[str, BreadthPanel],
    full_contexts: dict[str, FullContext],
    *,
    nav: float,
    participation: float,
    window: int = 36,
    step: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for label, panel in panels.items():
        dates = pd.DatetimeIndex(panel.returns.index).sort_values()
        total = sum(1 for pos, dt in enumerate(dates) if dt > TRAIN_END and pos >= window and ((pos - window) % step == 0))
        completed = 0
        for pos, dt in enumerate(dates):
            if dt <= TRAIN_END or pos < window or ((pos - window) % step != 0):
                continue
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(
                    f"[breadth-robustness] rolling OOS {label}: month {completed}/{total}",
                    flush=True,
                )
            train_dates = dates[max(0, pos - window):pos]
            try:
                ctx, status = build_training_from_panel(panel, train_dates=train_dates)
                if ctx is None:
                    continue
                books = fit_books(ctx, nav=nav, participation=participation, strategy_prefix=label)
                gross, net, _cost, _cap = score_books(
                    panel,
                    books,
                    full_contexts[label].cost_inputs,
                    nav=nav,
                    dates=[dt],
                )
                for strategy, book in books.items():
                    rows.append(
                        {
                            "config": label,
                            "return_date": pd.Timestamp(dt),
                            "strategy": f"{label} {strategy}",
                            "display_strategy": strategy,
                            "train_start": pd.Timestamp(train_dates.min()),
                            "train_end": pd.Timestamp(train_dates.max()),
                            "status": status,
                            "solver_status": book.solver_status,
                            "capacity_infeasible": bool(book.capacity_infeasible),
                            "deployed_gross": book.deployed_gross,
                            "gross_ret": float(gross[strategy].iloc[0]) if strategy in gross else np.nan,
                            "net_ret": float(net[strategy].iloc[0]) if strategy in net else np.nan,
                        }
                    )
            except Exception:
                continue
    detail = pd.DataFrame(rows)
    summary_rows = []
    if not detail.empty:
        for (config, strategy, display), grp in detail.groupby(["config", "strategy", "display_strategy"], dropna=False):
            for basis, col in [("gross", "gross_ret"), ("full_cost_net", "net_ret")]:
                stats = _series_stats(grp[col])
                summary_rows.append(
                    {
                        "config": config,
                        "strategy": strategy,
                        "display_strategy": display,
                        "basis": basis,
                        "n_months": int(pd.to_numeric(grp[col], errors="coerce").notna().sum()),
                        **stats,
                    }
                )
    return detail, pd.DataFrame(summary_rows)


def run_validation(
    *,
    selected_configs: Sequence[str] = CONFIG_ORDER,
    nav: float = DEFAULT_NAV,
    participation: float = DEFAULT_PARTICIPATION,
    cv_config: CVConfig = DEFAULT_CV_CONFIG,
    resample_config: ResampleConfig = ResampleConfig(),
    reprice_config: RepriceConfig = RepriceConfig(),
    simulation_config: SimulationConfig = SimulationConfig(),
    out_dir: Path = ROBUSTNESS_DIR,
    rolling_step: int = 1,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [c for c in CONFIG_ORDER if c in set(selected_configs)]
    started = time.monotonic()
    panels = build_panels(selected)
    full_contexts = {label: build_full_context(panel, nav, participation) for label, panel in panels.items()}
    gross_returns = pd.concat([ctx.gross_returns for ctx in full_contexts.values()], axis=1)
    net_returns = pd.concat([ctx.net_returns for ctx in full_contexts.values()], axis=1)
    spread_coverage = pd.concat([ctx.spread_coverage for ctx in full_contexts.values()], ignore_index=True, sort=False)
    spread_audit = spread_policy_status(spread_coverage)

    realized = realized_summary(full_contexts)
    print("[breadth-robustness] running CV/CPCV/PBO", flush=True)
    cv = run_cv_stage(panels, full_contexts, cv_config=cv_config, nav=nav, participation=participation)
    print("[breadth-robustness] running resampled MC", flush=True)
    mc_paths, mc_summary = run_mc_resampled(gross_returns, net_returns, resample_config)
    print("[breadth-robustness] running refit MC", flush=True)
    refit_paths, refit_summary = run_mc_refit(
        panels,
        full_contexts,
        config=resample_config,
        nav=nav,
        participation=participation,
    )
    print("[breadth-robustness] running repriced MC", flush=True)
    repriced_paths, repriced_summary_table, repriced_assumptions = run_mc_repriced(
        full_contexts,
        config=reprice_config,
        resample_config=resample_config,
    )
    print("[breadth-robustness] running path simulations", flush=True)
    sim_summary, sim_assumptions, drawdown, sim_paths = run_tail_path_simulations(
        {
            "Gross before costs": gross_returns,
            "Full-cost net": net_returns,
        },
        strategies=tuple(gross_returns.columns),
        config=simulation_config,
    )
    print("[breadth-robustness] running reality check", flush=True)
    scenario_variants = pd.concat(
        [gross_returns.add_suffix("::gross"), net_returns.add_suffix("::full_cost_net")],
        axis=1,
    )
    reality = sharpe_reality_check(scenario_variants, config=BootstrapConfig(n_boot=simulation_config.block_paths, seed=simulation_config.seed))
    print("[breadth-robustness] running rolling OOS", flush=True)
    rolling_detail, rolling_summary = run_rolling_oos(
        panels,
        full_contexts,
        nav=nav,
        participation=participation,
        step=rolling_step,
    )

    summary_csv, summary_json, summary_md = _build_validation_summary(
        realized,
        cv["cpcv_path_metrics"],
        cv["pbo_summary"],
        mc_summary,
        refit_summary,
        repriced_summary_table,
        sim_summary,
        drawdown,
        reality,
        rolling_summary,
        spread_audit,
        cv_config,
        nav,
        participation,
        elapsed=time.monotonic() - started,
    )

    _write_outputs(
        out_dir,
        spread_coverage=spread_coverage,
        realized=realized,
        gross_returns=gross_returns,
        net_returns=net_returns,
        cv=cv,
        mc_paths=mc_paths,
        mc_summary=mc_summary,
        refit_paths=refit_paths,
        refit_summary=refit_summary,
        repriced_paths=repriced_paths,
        repriced_summary=repriced_summary_table,
        repriced_assumptions=repriced_assumptions,
        sim_summary=sim_summary,
        sim_assumptions=sim_assumptions,
        drawdown=drawdown,
        sim_paths=sim_paths,
        reality=reality,
        rolling_detail=rolling_detail,
        rolling_summary=rolling_summary,
        validation_summary=summary_csv,
        validation_json=summary_json,
        validation_markdown=summary_md,
    )
    _write_latex_outputs(summary_csv, cv["cpcv_path_metrics"], cv["pbo_summary"], mc_summary, sim_summary, rolling_summary)
    return {
        "realized": realized,
        "spread_audit": spread_audit,
        "cv": cv,
        "summary": summary_json,
        "out_dir": str(out_dir),
    }


def _write_outputs(
    out_dir: Path,
    *,
    spread_coverage: pd.DataFrame,
    realized: pd.DataFrame,
    gross_returns: pd.DataFrame,
    net_returns: pd.DataFrame,
    cv: dict[str, pd.DataFrame],
    mc_paths: pd.DataFrame,
    mc_summary: pd.DataFrame,
    refit_paths: pd.DataFrame,
    refit_summary: pd.DataFrame,
    repriced_paths: pd.DataFrame,
    repriced_summary: pd.DataFrame,
    repriced_assumptions: pd.DataFrame,
    sim_summary: pd.DataFrame,
    sim_assumptions: pd.DataFrame,
    drawdown: pd.DataFrame,
    sim_paths: dict[str, pd.DataFrame],
    reality: pd.DataFrame,
    rolling_detail: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    validation_json: dict[str, object],
    validation_markdown: str,
) -> None:
    outputs = {
        "breadth_spread_source_coverage.csv": spread_coverage,
        "breadth_realized_candidate_summary.csv": realized,
        "breadth_strategy_returns_gross.csv": gross_returns,
        "breadth_strategy_returns_net.csv": net_returns,
        "breadth_cv_fold_schedule.csv": cv["fold_schedule"],
        "breadth_cv_fold_ledger.csv": cv["fold_ledger"],
        "breadth_cv_split_is_oos.csv": cv["split_is_oos"],
        "breadth_cv_test_month_returns.csv": cv["test_month_returns"],
        "breadth_cv_cpcv_path_month_returns.csv": cv["cpcv_path_month_returns"],
        "breadth_cv_cpcv_path_metrics.csv": cv["cpcv_path_metrics"],
        "breadth_cv_pbo_summary.csv": cv["pbo_summary"],
        "breadth_cv_runtime_log.csv": cv["runtime_log"],
        "breadth_mc_resampled_paths.csv": mc_paths,
        "breadth_mc_resampled_summary.csv": mc_summary,
        "breadth_mc_refit_paths.csv": refit_paths,
        "breadth_mc_refit_summary.csv": refit_summary,
        "breadth_mc_repriced_paths.csv": repriced_paths,
        "breadth_mc_repriced_summary.csv": repriced_summary,
        "breadth_mc_repriced_assumptions.csv": repriced_assumptions,
        "breadth_simulation_summary.csv": sim_summary,
        "breadth_simulation_assumptions.csv": sim_assumptions,
        "breadth_drawdown_breach_rates.csv": drawdown,
        "breadth_reality_check_inference.csv": reality,
        "breadth_rolling_oos.csv": rolling_detail,
        "breadth_rolling_oos_summary.csv": rolling_summary,
        "breadth_validation_summary.csv": validation_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False)
    (out_dir / "breadth_validation_summary.json").write_text(
        json.dumps(_json_safe(validation_json), indent=2),
        encoding="utf-8",
    )
    (out_dir / "breadth_validation_summary.md").write_text(validation_markdown, encoding="utf-8")
    for key, frame in sim_paths.items():
        frame.to_csv(out_dir / f"breadth_simulation_paths_{_slug(key)}.csv", index=False)


def _build_validation_summary(
    realized: pd.DataFrame,
    cpcv_metrics: pd.DataFrame,
    pbo: pd.DataFrame,
    mc_summary: pd.DataFrame,
    refit_summary: pd.DataFrame,
    repriced_summary_table: pd.DataFrame,
    sim_summary: pd.DataFrame,
    drawdown: pd.DataFrame,
    reality: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    spread_audit: dict[str, object],
    cv_config: CVConfig,
    nav: float,
    participation: float,
    *,
    elapsed: float,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    rows = []
    for _, row in realized.iterrows():
        if row["strategy"] != PRIMARY_STRATEGY:
            continue
        strategy_name = f"{row['config']} {row['strategy']}"
        cpcv_net = _metric_quantiles(cpcv_metrics, strategy_name, "full_cost_net", "sharpe")
        cpcv_gross = _metric_quantiles(cpcv_metrics, strategy_name, "gross", "sharpe")
        mc_net = _summary_lookup(mc_summary, strategy_name, "full_cost_net")
        refit_net = _refit_lookup(refit_summary, strategy_name, "full_cost_net")
        repriced_net = _repriced_lookup(repriced_summary_table, strategy_name, "full_cost_net_overlay")
        rolling_net = _rolling_lookup(rolling_summary, strategy_name, "full_cost_net")
        reality_row = reality[reality.get("Variant", pd.Series(dtype=object)).astype(str).eq(f"{strategy_name}::full_cost_net")]
        reality_p = float(pd.to_numeric(reality_row.get("Reality check p", pd.Series(dtype=float)), errors="coerce").iloc[0]) if not reality_row.empty else np.nan
        verdict = _verdict(
            deployable=bool(row["deployable"]),
            net_sharpe=float(row["net_sharpe"]),
            cpcv_p05=cpcv_net.get("p05", np.nan),
            mc_p05=mc_net.get("P05 Sharpe", np.nan),
            refit_p05=refit_net.get("p05", np.nan),
            rolling_sharpe=rolling_net.get("sharpe", np.nan),
        )
        rows.append(
            {
                "config": row["config"],
                "strategy": row["strategy"],
                "deployable": bool(row["deployable"]),
                "verdict": verdict,
                "net_sharpe": row["net_sharpe"],
                "net_sortino": row["net_sortino"],
                "deployed_gross": row["deployed_gross"],
                "sum_of_caps": row["sum_of_caps"],
                "cpcv_net_p05": cpcv_net.get("p05", np.nan),
                "cpcv_net_p50": cpcv_net.get("p50", np.nan),
                "cpcv_net_p95": cpcv_net.get("p95", np.nan),
                "cpcv_gross_p50": cpcv_gross.get("p50", np.nan),
                "mc_resampled_net_p05": mc_net.get("P05 Sharpe", np.nan),
                "mc_resampled_net_p50": mc_net.get("P50 Sharpe", np.nan),
                "mc_refit_net_p05": refit_net.get("p05", np.nan),
                "mc_refit_net_p50": refit_net.get("p50", np.nan),
                "repriced_net_overlay_p05": repriced_net.get("P05 Sharpe", np.nan),
                "repriced_net_overlay_p50": repriced_net.get("P50 Sharpe", np.nan),
                "rolling_net_sharpe": rolling_net.get("sharpe", np.nan),
                "reality_check_p": reality_p,
            }
        )
    summary = pd.DataFrame(rows)
    payload = {
        "provenance": {
            "git_rev": _git_rev(),
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "elapsed_seconds": float(elapsed),
            "nav": float(nav),
            "participation": float(participation),
            "cv_config": dataclasses.asdict(cv_config),
            "expected_cv_splits_per_config": expected_cv_split_count(cv_config),
            "spread_audit": spread_audit,
            "primary_knobs": dataclasses.asdict(E1_KNOBS),
        },
        "rows": summary.to_dict(orient="records"),
        "pbo": pbo.to_dict(orient="records") if not pbo.empty else [],
        "cost_overlay_note": "Repriced net rows subtract a circular-block sample of realized full-cost drag; they are not synthetic NBBO/CBBO quotes.",
    }
    markdown = _summary_markdown(summary, payload)
    return summary, payload, markdown


def _write_latex_outputs(
    summary: pd.DataFrame,
    cpcv_metrics: pd.DataFrame,
    pbo_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    sim_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    headline_cols = [
        "config",
        "verdict",
        "deployable",
        "net_sharpe",
        "net_sortino",
        "cpcv_net_p05",
        "cpcv_net_p50",
        "mc_resampled_net_p05",
        "mc_refit_net_p05",
        "rolling_net_sharpe",
    ]
    _write_latex_table(summary.reindex(columns=headline_cols), TABLE_DIR / "breadth_robustness_summary.tex")
    cpcv_table = _compact_cpcv_table(cpcv_metrics)
    _write_latex_table(cpcv_table, TABLE_DIR / "breadth_robustness_cpcv.tex")
    _write_latex_table(_compact_pbo_table(pbo_summary), TABLE_DIR / "breadth_robustness_pbo.tex")
    mc_table = mc_summary[mc_summary.get("Basis", pd.Series(dtype=object)).astype(str).eq("full_cost_net")].copy()
    _write_latex_table(mc_table.head(40), TABLE_DIR / "breadth_robustness_mc_resampled.tex")
    _write_latex_table(_compact_simulation_table(sim_summary), TABLE_DIR / "breadth_robustness_simulation.tex")
    _write_latex_table(rolling_summary.head(40), TABLE_DIR / "breadth_robustness_rolling_oos.tex")


def _compact_cpcv_table(cpcv_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cpcv_metrics.empty:
        return pd.DataFrame()
    complete = cpcv_metrics[cpcv_metrics.get("status", pd.Series("complete", index=cpcv_metrics.index)).astype(str).eq("complete")]
    for (strategy, basis), group in complete.groupby(["strategy", "basis"], dropna=False):
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce")
        rows.append(
            {
                "Strategy": strategy,
                "Basis": basis,
                "Paths": int(sharpe.notna().sum()),
                "P05 Sharpe": float(sharpe.quantile(0.05)) if sharpe.notna().any() else np.nan,
                "P50 Sharpe": float(sharpe.quantile(0.50)) if sharpe.notna().any() else np.nan,
                "P95 Sharpe": float(sharpe.quantile(0.95)) if sharpe.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["Strategy", "Basis"]).reset_index(drop=True)


def _compact_pbo_table(pbo_summary: pd.DataFrame) -> pd.DataFrame:
    if pbo_summary.empty:
        return pd.DataFrame()
    cols = [
        "scope",
        "config",
        "Basis",
        "N splits",
        "N strategies",
        "PBO",
        "Median lambda",
        "Rank correlation IS OOS",
    ]
    return pbo_summary.reindex(columns=cols)


def _compact_simulation_table(sim_summary: pd.DataFrame) -> pd.DataFrame:
    if sim_summary.empty:
        return pd.DataFrame()
    table = sim_summary[
        sim_summary.get("Strategy", pd.Series(dtype=object)).astype(str).str.contains("E1 capped", regex=False)
    ].copy()
    cols = [
        "Return basis",
        "Strategy",
        "Requested method",
        "Simulation",
        "N paths",
        "Defaulted path share",
        "Ann. return p05",
        "Ann. return p50",
        "Sortino p50",
        "Max DD p50",
        "Terminal wealth p50",
    ]
    return table.reindex(columns=cols)


def _summary_markdown(summary: pd.DataFrame, payload: dict[str, object]) -> str:
    lines = ["# Breadth Robustness Validation", ""]
    prov = payload["provenance"]
    lines.append(
        f"Cost policy: full cost stack, NAV ${prov['nav']:,.0f}, X={prov['participation']:.2f}, "
        "current Cboe fills disabled, inferred CBBO proxy enabled."
    )
    lines.append(
        f"CV policy: {prov['cv_config']['n_groups']} chronological groups, "
        f"{prov['cv_config']['n_test_groups']} test groups, purge/embargo="
        f"{prov['cv_config']['purge_months']}/{prov['cv_config']['embargo_months']} month(s), "
        f"{prov['expected_cv_splits_per_config']} splits per config."
    )
    audit = prov["spread_audit"]
    lines.append(
        f"Spread-source audit: {audit['status']} "
        f"(current Cboe rows={audit['current_cboe_rows']}, default rows={audit['default_rows']})."
    )
    lines.append("")
    lines.extend(_markdown_table(summary))
    lines.append("")
    lines.append(
        "Repriced synthetic net paths use a circular-block sample of realized full-cost drag; "
        "they are not synthetic NBBO/CBBO quotes."
    )
    return "\n".join(lines) + "\n"


def _verdict(
    *,
    deployable: bool,
    net_sharpe: float,
    cpcv_p05: float,
    mc_p05: float,
    refit_p05: float,
    rolling_sharpe: float,
) -> str:
    if not deployable:
        return "diagnostic_capacity_infeasible"
    checks = [
        np.isfinite(net_sharpe) and net_sharpe > 0.0,
        np.isfinite(cpcv_p05) and cpcv_p05 > 0.0,
        np.isfinite(mc_p05) and mc_p05 > 0.0,
        np.isfinite(refit_p05) and refit_p05 > 0.0,
        np.isfinite(rolling_sharpe) and rolling_sharpe > 0.0,
    ]
    passed = sum(bool(x) for x in checks)
    if passed >= 4:
        return "pass"
    if passed >= 2:
        return "mixed"
    return "fail"


def _metric_quantiles(frame: pd.DataFrame, strategy: str, basis: str, metric: str) -> dict[str, float]:
    if frame.empty:
        return {}
    sub = frame[
        frame.get("strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
        & frame.get("basis", pd.Series(dtype=object)).astype(str).eq(basis)
        & frame.get("status", pd.Series("complete", index=frame.index)).astype(str).eq("complete")
    ]
    values = pd.to_numeric(sub.get(metric, pd.Series(dtype=float)), errors="coerce")
    if not values.notna().any():
        return {}
    return {
        "p05": float(values.quantile(0.05)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
    }


def _summary_lookup(summary: pd.DataFrame, strategy: str, basis: str) -> dict[str, float]:
    if summary.empty:
        return {}
    sub = summary[
        summary.get("Strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
        & summary.get("Basis", pd.Series(dtype=object)).astype(str).eq(basis)
    ]
    if sub.empty:
        return {}
    row = sub.iloc[0].to_dict()
    for target, source in [
        ("P05 Sharpe", "Path P05 Sharpe"),
        ("P50 Sharpe", "Path P50 Sharpe"),
        ("P95 Sharpe", "Path P95 Sharpe"),
    ]:
        if target not in row and source in row:
            row[target] = row[source]
    return row


def _refit_lookup(summary: pd.DataFrame, strategy: str, basis: str) -> dict[str, float]:
    if summary.empty:
        return {}
    sub = summary[
        summary.get("strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
        & summary.get("basis", pd.Series(dtype=object)).astype(str).eq(basis)
    ]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {"p05": row.get("p05_sharpe", np.nan), "p50": row.get("p50_sharpe", np.nan), "p95": row.get("p95_sharpe", np.nan)}


def _repriced_lookup(summary: pd.DataFrame, strategy: str, basis: str) -> dict[str, float]:
    if summary.empty:
        return {}
    sub = summary[
        summary.get("Strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
        & summary.get("basis", pd.Series(dtype=object)).astype(str).eq(basis)
    ]
    if sub.empty:
        sub = summary[
            summary.get("Strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
            & summary.get("Basis", pd.Series(dtype=object)).astype(str).eq(basis)
        ]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _rolling_lookup(summary: pd.DataFrame, strategy: str, basis: str) -> dict[str, float]:
    if summary.empty:
        return {}
    sub = summary[
        summary.get("strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
        & summary.get("basis", pd.Series(dtype=object)).astype(str).eq(basis)
    ]
    return sub.iloc[0].to_dict() if not sub.empty else {}


def _refit_summary(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if paths.empty:
        return pd.DataFrame()
    for (strategy, basis), group in paths.groupby(["strategy", "basis"], dropna=False):
        ok = group[group.get("status", pd.Series("ok", index=group.index)).astype(str).eq("ok")]
        sharpe = pd.to_numeric(ok.get("sharpe", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "strategy": strategy,
                "basis": basis,
                "paths": int(len(group)),
                "ok_paths": int(len(ok)),
                "p05_sharpe": float(sharpe.quantile(0.05)) if sharpe.notna().any() else np.nan,
                "p50_sharpe": float(sharpe.quantile(0.50)) if sharpe.notna().any() else np.nan,
                "p95_sharpe": float(sharpe.quantile(0.95)) if sharpe.notna().any() else np.nan,
                "error_paths": int(len(group) - len(ok)),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "basis"]).reset_index(drop=True)


def _realized_sharpe_rows(frame: pd.DataFrame, *, basis: str, universe_family: str) -> pd.DataFrame:
    rows = []
    for strategy in frame.columns:
        stats = _series_stats(frame[strategy])
        rows.append(
            {
                "strategy": strategy,
                "basis": basis,
                "universe_family": universe_family,
                "sharpe": stats["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def _repriced_paths_with_basis(
    contract_returns: np.ndarray,
    weights_by_strategy: dict[str, pd.Series],
    contract_index: pd.Index,
    *,
    basis: str,
    cost_overlay: dict[str, pd.Series] | None,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    returns = np.asarray(contract_returns, dtype=float)
    contracts = pd.Index(contract_index)
    rng = np.random.default_rng(seed)
    rows = []
    for strategy, weights_raw in weights_by_strategy.items():
        weights = pd.Series(weights_raw, dtype=float)
        total_abs = float(weights.abs().sum())
        aligned = weights.reindex(contracts).fillna(0.0)
        covered_abs = float(aligned.abs().sum())
        coverage = 1.0 if total_abs <= 1e-12 else covered_abs / total_abs
        weighted_returns = np.nan_to_num(returns, nan=0.0) @ aligned.to_numpy(float)
        costs = None
        if cost_overlay is not None and strategy in cost_overlay:
            costs = pd.Series(cost_overlay[strategy], dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        for path_id in range(returns.shape[0]):
            series = weighted_returns[path_id].copy()
            if costs is not None and len(costs):
                sampled = circular_block_sample(costs, rng, block_length).astype(float)
                if len(sampled) != len(series):
                    sampled = np.resize(sampled, len(series))
                series = series - sampled
            stats = performance_metrics(pd.Series(series), 12)
            rows.append(
                {
                    "method": "joint_garch_block",
                    "basis": basis,
                    "path_id": int(path_id),
                    "strategy": strategy,
                    "sharpe": float(stats.get("sharpe", np.nan)),
                    "sortino": float(stats.get("sortino", np.nan)),
                    "max_drawdown": float(stats.get("max_drawdown", np.nan)),
                    "ann_return": float(stats.get("annualized_return", np.nan)),
                    "terminal_wealth": float(stats.get("terminal_wealth", np.nan)),
                    "defaulted": bool(stats.get("defaulted", False)),
                    "weight_coverage": float(coverage),
                }
            )
    return pd.DataFrame(rows)


def _repriced_summary_with_basis(paths_frame: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Strategy",
        "basis",
        "Method",
        "Realized Sharpe",
        "P05 Sharpe",
        "P50 Sharpe",
        "P95 Sharpe",
        "P50 Max Drawdown",
        "P Sharpe Less Than 0",
        "P Default",
    ]
    paths = pd.DataFrame(paths_frame).copy()
    if paths.empty:
        return pd.DataFrame(columns=columns)
    realized_lookup = (
        {
            (str(row.strategy), str(row.basis)): float(row.sharpe)
            for row in realized.itertuples(index=False)
            if np.isfinite(float(row.sharpe))
        }
        if not realized.empty and {"strategy", "sharpe"}.issubset(realized.columns)
        else {}
    )
    rows = []
    for (strategy, basis, method), group in paths.groupby(["strategy", "basis", "method"], dropna=False):
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce")
        max_dd = pd.to_numeric(group["max_drawdown"], errors="coerce")
        defaulted = group.get("defaulted", pd.Series(False, index=group.index)).astype(bool)
        rows.append(
            {
                "Strategy": strategy,
                "basis": basis,
                "Method": method,
                "Realized Sharpe": realized_lookup.get(
                    (str(strategy), "full_cost_net" if str(basis) == "full_cost_net_overlay" else str(basis)),
                    np.nan,
                ),
                "P05 Sharpe": float(sharpe.quantile(0.05)) if sharpe.notna().any() else np.nan,
                "P50 Sharpe": float(sharpe.quantile(0.50)) if sharpe.notna().any() else np.nan,
                "P95 Sharpe": float(sharpe.quantile(0.95)) if sharpe.notna().any() else np.nan,
                "P50 Max Drawdown": float(max_dd.quantile(0.50)) if max_dd.notna().any() else np.nan,
                "P Sharpe Less Than 0": float((sharpe < 0.0).mean()) if sharpe.notna().any() else np.nan,
                "P Default": float(defaulted.mean()) if len(defaulted) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["Strategy", "basis", "Method"]).reset_index(drop=True)


def _slot_relabel(frame: pd.DataFrame, path: np.ndarray, train_dates: pd.DatetimeIndex) -> pd.DataFrame:
    sampled = pd.DataFrame(frame).iloc[np.asarray(path, dtype=int)].copy()
    sampled.index = train_dates
    return sampled


def _iv_level_panel(reps: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    frame = reps.copy()
    if frame.empty or "iv_proxy" not in frame:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=list(universe))
    frame["snap_date"] = pd.to_datetime(frame["snap_date"])
    bucket = frame.get("moneyness_bucket", pd.Series("", index=frame.index)).astype(str)
    atm = frame[bucket.isin(["atm", "vix_atm"])].copy()
    if atm.empty:
        atm = frame.copy()
    return (
        atm.groupby(["snap_date", "underlying"])["iv_proxy"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=list(universe))
        .replace([np.inf, -np.inf], np.nan)
    )


def _vix_level(vix_state: pd.DataFrame, dates: pd.Index) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if "VIX" in vix_state:
        return pd.to_numeric(vix_state["VIX"], errors="coerce").reindex(idx)
    if "VX_FRONT" in vix_state:
        return pd.to_numeric(vix_state["VX_FRONT"], errors="coerce").reindex(idx)
    return pd.Series(np.nan, index=idx, name="VIX")


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if not (pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col])):
            out[col] = out[col].map(_latex_escape)
    out.columns = [_latex_escape(str(col)) for col in out.columns]
    path.write_text(out.to_latex(index=False, escape=False, float_format="%.3f", na_rep=""), encoding="utf-8")


def _latex_escape(value: object) -> object:
    if not isinstance(value, str):
        return value
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["(empty)"]
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in cols) + " |")
    return lines


def _format_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _slug(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "blank"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _parse_list(raw: str, valid: Sequence[str]) -> list[str]:
    if raw.strip().lower() in {"all", "*"}:
        return list(valid)
    requested = {x.strip() for x in raw.split(",") if x.strip()}
    bad = requested.difference(valid)
    if bad:
        raise SystemExit(f"unknown config(s): {', '.join(sorted(bad))}")
    return [x for x in valid if x in requested]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all")
    parser.add_argument("--nav", type=float, default=DEFAULT_NAV)
    parser.add_argument("--participation", type=float, default=DEFAULT_PARTICIPATION)
    parser.add_argument("--cv-groups", type=int, default=12)
    parser.add_argument("--cv-test-groups", type=int, default=2)
    parser.add_argument("--mc-paths", type=int, default=1000)
    parser.add_argument("--mc-refit-paths", type=int, default=200)
    parser.add_argument("--mc-reprice-paths", type=int, default=1000)
    parser.add_argument("--simulation-paths", type=int, default=1000)
    parser.add_argument("--rolling-step", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=ROBUSTNESS_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configs = _parse_list(args.configs, CONFIG_ORDER)
    if args.smoke:
        args.mc_paths = min(args.mc_paths, 20)
        args.mc_refit_paths = min(args.mc_refit_paths, 5)
        args.mc_reprice_paths = min(args.mc_reprice_paths, 20)
        args.simulation_paths = min(args.simulation_paths, 20)
        args.rolling_step = max(args.rolling_step, 6)
    cv_config = CVConfig(n_groups=args.cv_groups, n_test_groups=args.cv_test_groups, purge_months=1, embargo_months=1)
    resample_config = ResampleConfig(n_paths=args.mc_paths, n_refit_paths=args.mc_refit_paths)
    reprice_config = RepriceConfig(n_paths=args.mc_reprice_paths)
    simulation_config = SimulationConfig(block_paths=args.simulation_paths, vol_paths=args.simulation_paths)
    run_validation(
        selected_configs=configs,
        nav=args.nav,
        participation=args.participation,
        cv_config=cv_config,
        resample_config=resample_config,
        reprice_config=reprice_config,
        simulation_config=simulation_config,
        out_dir=args.out_dir,
        rolling_step=args.rolling_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
