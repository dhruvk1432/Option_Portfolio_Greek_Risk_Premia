"""Cross-validation diagnostics for the option-only Markowitz paper.

This module is intentionally pure: it does not import ``run_empirics`` and all
pipeline-specific builders are injected by the caller.  CPCV trains on data
after test folds by construction; this is a distributional-robustness /
overfitting diagnostic (backtest-path distribution, PBO), NOT a tradable
point-in-time backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.simulation import performance_metrics


@dataclass(frozen=True)
class CVConfig:
    n_groups: int = 12
    n_test_groups: int = 2
    purge_months: int = 1
    embargo_months: int = 1
    min_train_months: int = 36
    min_contracts: int = 8
    min_obs_per_contract: int = 18
    seed: int = 20260625
    periods_per_year: float = 12.0


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    scheme: str
    test_groups: tuple[int, ...]
    test_dates: tuple[pd.Timestamp, ...]
    train_dates: tuple[pd.Timestamp, ...]
    purged_dates: tuple[pd.Timestamp, ...]
    embargoed_dates: tuple[pd.Timestamp, ...]


@dataclass
class CVResults:
    fold_schedule: pd.DataFrame
    fold_ledger: pd.DataFrame
    split_is_oos: pd.DataFrame
    test_month_returns: pd.DataFrame
    runtime_log: pd.DataFrame


EVENT_WINDOWS = {
    "Volmageddon Feb 2018": ("2018-02-01", "2018-03-31"),
    "COVID crash": ("2020-02-01", "2020-04-30"),
    "2022 tightening bear": ("2022-01-01", "2022-10-31"),
}

# Mirrors run_empirics.run_all headline strategy names, omitting
# "Cost-aware Sortino + VIX" because per-fold solve_max_sortino is expensive
# and requires fold-specific execution-cost inputs.
CV_STRATEGIES: tuple[str, ...] = (
    "Equity-option Greek Markowitz",
    "Greek Markowitz + VIX",
    "Beta/delta-neutral + VIX",
    "Equal premium",
    "Equal risk",
    "VIX hedge sleeve",
)


def _as_sorted_unique_dates(dates: pd.DatetimeIndex | Sequence[Any]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    return pd.DatetimeIndex(idx.unique()).sort_values()


def _date_tuple(dates: Sequence[Any]) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(d) for d in dates)


def _iso_join(dates: Sequence[pd.Timestamp]) -> str:
    return ";".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates)


def _fold_width(n_groups: int) -> int:
    return max(2, len(str(max(int(n_groups) - 1, 0))))


def _fold_id(scheme: str, groups: Sequence[int], n_groups: int) -> str:
    width = _fold_width(n_groups)
    suffix = "_".join(f"{int(g):0{width}d}" for g in groups)
    return f"{scheme}_{suffix}"


def _call_weights_builder(
    weights_builder: Callable[..., Mapping[str, pd.Series]],
    model: Any,
    universe: Sequence[str],
) -> Mapping[str, pd.Series]:
    try:
        return weights_builder(model, universe)
    except TypeError:
        return weights_builder(model)


def _contracts(model: Any) -> list[Any]:
    return list(getattr(model, "contracts", []))


def _is_zero_weights(weights: pd.Series) -> bool:
    if weights is None:
        return True
    values = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).to_numpy(float)
    return bool(len(values) == 0 or np.all(np.abs(values) <= 1e-14))


def _series_metrics(series: pd.Series, periods_per_year: float) -> dict[str, float]:
    r = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "terminal_wealth": np.nan,
            "defaulted": False,
        }
    # Keep CPCV/fold metrics aligned with simulation.performance_metrics:
    # returns <= -100% absorb wealth at zero instead of sign-flipping wealth.
    stats = performance_metrics(r, periods_per_year)
    return {
        "sharpe": float(stats.get("sharpe", np.nan)),
        "sortino": float(stats.get("sortino", np.nan)),
        "max_drawdown": float(stats.get("max_drawdown", np.nan)),
        "ann_return": float(stats.get("annualized_return", np.nan)),
        "ann_vol": float(stats.get("annualized_volatility", np.nan)),
        "terminal_wealth": float(stats.get("terminal_wealth", np.nan)),
        "defaulted": bool(stats.get("defaulted", False)),
    }


def build_group_schedule(dates: pd.DatetimeIndex, n_groups: int) -> pd.Series:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    unique_dates = _as_sorted_unique_dates(dates)
    values = np.empty(len(unique_dates), dtype=int)
    for group_id, positions in enumerate(np.array_split(np.arange(len(unique_dates)), n_groups)):
        values[positions] = group_id
    return pd.Series(values, index=unique_dates, name="group")


def _contiguous_blocks(positions: Sequence[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    ordered = sorted(int(p) for p in positions)
    blocks: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        blocks.append((start, prev))
        start = prev = pos
    blocks.append((start, prev))
    return blocks


def build_folds(dates: pd.DatetimeIndex, config: CVConfig, scheme: str) -> list[FoldSpec]:
    scheme = str(scheme).lower()
    if scheme not in {"kfold", "cpcv"}:
        raise ValueError("scheme must be 'kfold' or 'cpcv'")
    if config.n_groups <= 0:
        raise ValueError("config.n_groups must be positive")
    if config.n_test_groups <= 0 or config.n_test_groups > config.n_groups:
        raise ValueError("config.n_test_groups must be in [1, n_groups]")

    unique_dates = _as_sorted_unique_dates(dates)
    schedule = build_group_schedule(unique_dates, config.n_groups)
    positions_by_date = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}
    all_positions = set(range(len(unique_dates)))
    group_sets: list[tuple[int, ...]]
    if scheme == "kfold":
        group_sets = [(g,) for g in range(config.n_groups)]
    else:
        group_sets = [tuple(c) for c in combinations(range(config.n_groups), config.n_test_groups)]

    folds: list[FoldSpec] = []
    for groups in group_sets:
        test_dates = schedule.index[schedule.isin(groups)]
        test_pos = {positions_by_date[pd.Timestamp(d)] for d in test_dates}
        purged_pos: set[int] = set()
        embargoed_pos: set[int] = set()
        for start, end in _contiguous_blocks(sorted(test_pos)):
            purge_start = max(0, start - config.purge_months)
            purge_end = min(len(unique_dates) - 1, end + config.purge_months)
            if purge_start <= purge_end:
                purged_pos.update(range(purge_start, purge_end + 1))
            embargo_start = end + config.purge_months + 1
            embargo_end = min(len(unique_dates) - 1, end + config.purge_months + config.embargo_months)
            if embargo_start <= embargo_end:
                embargoed_pos.update(range(embargo_start, embargo_end + 1))
        purged_pos.difference_update(test_pos)
        embargoed_pos.difference_update(test_pos)
        embargoed_pos.difference_update(purged_pos)
        train_pos = sorted(all_positions.difference(test_pos).difference(purged_pos).difference(embargoed_pos))
        if len(train_pos) < config.min_train_months:
            train_pos = []
        folds.append(
            FoldSpec(
                fold_id=_fold_id(scheme, groups, config.n_groups),
                scheme=scheme,
                test_groups=tuple(int(g) for g in groups),
                test_dates=_date_tuple(test_dates),
                train_dates=_date_tuple(unique_dates[train_pos]),
                purged_dates=_date_tuple(unique_dates[sorted(purged_pos)]),
                embargoed_dates=_date_tuple(unique_dates[sorted(embargoed_pos)]),
            )
        )
    return folds


def _restricted_reps(reps: pd.DataFrame, train_dates: Sequence[pd.Timestamp], columns: Sequence[Any]) -> pd.DataFrame:
    sub_reps = reps[reps["snap_date"].isin(train_dates)].copy()
    if "asset_id" in sub_reps.columns:
        sub_reps = sub_reps[sub_reps["asset_id"].isin(columns)].copy()
    return sub_reps


def _fit_one_model(
    returns: pd.DataFrame,
    reps: pd.DataFrame,
    universe: Sequence[str],
    fold: FoldSpec,
    *,
    spec_builder: Callable[..., pd.DataFrame],
    model_factory: Callable[..., tuple[Any, Any]],
    config: CVConfig,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_dates = list(fold.train_dates)
    if len(train_dates) < config.min_train_months:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "skipped_min_train_months"
    train_returns = returns.loc[train_dates]
    columns = list(train_returns.columns[train_returns.count() >= config.min_obs_per_contract])
    if len(columns) < config.min_contracts:
        return None, pd.DataFrame(), pd.DataFrame(), train_returns, "skipped_too_few_contracts"
    train_returns = train_returns.loc[:, columns]
    sub_reps = _restricted_reps(reps, train_dates, columns)
    train_start = min(train_dates)
    train_end = max(train_dates)
    spec = spec_builder(sub_reps, train_returns, train_start=train_start, train_end=train_end)
    if len(spec) < config.min_contracts:
        return None, spec, sub_reps, train_returns, "skipped_too_few_specs"
    model_result = model_factory(
        spec,
        train_returns,
        sub_reps,
        universe,
        train_start=train_start,
        train_end=train_end,
    )
    model = model_result[0] if isinstance(model_result, tuple) else model_result
    return model, spec, sub_reps, train_returns, "ok"


def refit_fold(
    fold,
    returns,
    reps,
    equity_returns,
    equity_reps,
    universe,
    equity_universe,
    *,
    spec_builder,
    model_factory,
    weights_builder,
    vix_sleeve_builder=None,
    equity_tangency_builder=None,
    delta_map_builder=None,
    config: CVConfig = CVConfig(),
) -> dict:
    model, spec, sub_reps, train_returns, status = _fit_one_model(
        returns,
        reps,
        universe,
        fold,
        spec_builder=spec_builder,
        model_factory=model_factory,
        config=config,
    )
    if status != "ok":
        return {
            "status": status,
            "strategies": {},
            "strategy_status": {},
            "equity_benchmarks": {},
            "model": None,
            "equity_model": None,
            "n_train": int(len(fold.train_dates)),
            "n_contracts": int(len(spec) if len(spec) else 0),
        }

    combined = dict(_call_weights_builder(weights_builder, model, universe))
    strategies: dict[str, pd.Series] = {}
    equity_model = None
    equity_status = "ok"
    try:
        equity_model, _equity_spec, _equity_sub_reps, _equity_train_returns, equity_status = _fit_one_model(
            equity_returns,
            equity_reps,
            equity_universe,
            fold,
            spec_builder=spec_builder,
            model_factory=model_factory,
            config=config,
        )
        if equity_status == "ok":
            equity_combined = dict(_call_weights_builder(weights_builder, equity_model, equity_universe))
            if "Greek Markowitz" in equity_combined:
                strategies["Equity-option Greek Markowitz"] = equity_combined["Greek Markowitz"]
    except Exception:
        equity_status = "skipped_equity_model_error"

    if "Greek Markowitz" in combined:
        strategies["Greek Markowitz + VIX"] = combined["Greek Markowitz"]
    if "Delta neutral" in combined:
        strategies["Beta/delta-neutral + VIX"] = combined["Delta neutral"]
    if "Equal premium" in combined:
        strategies["Equal premium"] = combined["Equal premium"]
    if "Equal risk" in combined:
        strategies["Equal risk"] = combined["Equal risk"]
    if vix_sleeve_builder is not None:
        strategies["VIX hedge sleeve"] = vix_sleeve_builder(model)

    strategy_status = {
        name: ("infeasible_zero_weights" if _is_zero_weights(weights) else "ok")
        for name, weights in strategies.items()
    }

    equity_benchmarks: dict[str, pd.Series] = {}
    if delta_map_builder is not None and "Greek Markowitz + VIX" in strategies:
        equity_benchmarks["Delta-matched equities"] = delta_map_builder(
            model,
            strategies["Greek Markowitz + VIX"],
        )
    if equity_tangency_builder is not None:
        train_under = (
            equity_returns.loc[list(fold.train_dates)]
            .reindex(columns=list(equity_universe))
            .fillna(0.0)
        )
        equity_benchmarks["Underlying Markowitz"] = equity_tangency_builder(train_under)

    return {
        "status": "ok",
        "strategies": strategies,
        "strategy_status": strategy_status,
        "equity_benchmarks": equity_benchmarks,
        "model": model,
        "equity_model": equity_model,
        "n_train": int(len(fold.train_dates)),
        "n_contracts": int(len(_contracts(model)) or len(spec)),
        "equity_status": equity_status,
    }


def _net_frame_from_cost_result(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if hasattr(result, "net"):
        return result.net
    if isinstance(result, tuple) and len(result):
        first = result[0]
        if isinstance(first, pd.DataFrame):
            return first
    return pd.DataFrame()


def _append_return_rows(
    rows: list[dict[str, Any]],
    fold: FoldSpec,
    name: str,
    basis: str,
    series: pd.Series,
) -> None:
    for dt, value in pd.Series(series).items():
        rows.append(
            {
                "fold_id": fold.fold_id,
                "scheme": fold.scheme,
                "return_date": pd.Timestamp(dt),
                "strategy": name,
                "basis": basis,
                "ret": float(value) if np.isfinite(value) else np.nan,
            }
        )


def _append_ledger_row(
    rows: list[dict[str, Any]],
    fold: FoldSpec,
    name: str,
    basis: str,
    series: pd.Series,
    status: str,
    periods_per_year: float,
) -> None:
    metrics = _series_metrics(series, periods_per_year)
    rows.append(
        {
            "fold_id": fold.fold_id,
            "scheme": fold.scheme,
            "strategy": name,
            "basis": basis,
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "ann_return": metrics["ann_return"],
            "ann_vol": metrics["ann_vol"],
            "defaulted": metrics["defaulted"],
            "n_test_months": int(pd.Series(series).dropna().shape[0]),
            "status": status,
        }
    )


def _equity_portfolio_returns(
    returns: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    columns: Sequence[str],
    weights: pd.Series,
) -> pd.Series:
    frame = returns.loc[list(dates)].reindex(columns=list(columns)).fillna(0.0)
    w = weights.reindex(list(columns)).fillna(0.0).to_numpy(float)
    return pd.Series(frame.to_numpy(float) @ w, index=frame.index)


def evaluate_folds(
    folds,
    returns,
    reps,
    equity_returns,
    equity_reps,
    universe,
    equity_universe,
    cost_inputs,
    *,
    spec_builder,
    model_factory,
    weights_builder,
    cost_scenario_builder=None,
    scenario_config=None,
    vix_sleeve_builder=None,
    equity_tangency_builder=None,
    delta_map_builder=None,
    config: CVConfig = CVConfig(),
) -> CVResults:
    schedule_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for fold in folds:
        started = time.monotonic()
        status = "ok"
        refit: dict[str, Any] | None = None
        try:
            refit = refit_fold(
                fold,
                returns,
                reps,
                equity_returns,
                equity_reps,
                universe,
                equity_universe,
                spec_builder=spec_builder,
                model_factory=model_factory,
                weights_builder=weights_builder,
                vix_sleeve_builder=vix_sleeve_builder,
                equity_tangency_builder=equity_tangency_builder,
                delta_map_builder=delta_map_builder,
                config=config,
            )
            status = str(refit.get("status", "ok"))
            if status == "ok":
                model = refit["model"]
                strategies: dict[str, pd.Series] = dict(refit.get("strategies", {}))
                strategy_status: dict[str, str] = dict(refit.get("strategy_status", {}))
                test_dates = list(fold.test_dates)
                train_dates = list(fold.train_dates)
                test = returns.loc[test_dates, _contracts(model)].fillna(0.0)
                train = returns.loc[train_dates, _contracts(model)].fillna(0.0)
                gross_frame = pd.DataFrame(index=test.index)
                for name, weights in strategies.items():
                    pr = model.portfolio_return_series(test, weights)
                    gross_frame[name] = pr
                    s_status = strategy_status.get(name, "ok")
                    _append_return_rows(return_rows, fold, name, "gross", pr)
                    _append_ledger_row(
                        ledger_rows,
                        fold,
                        name,
                        "gross",
                        pr,
                        s_status,
                        config.periods_per_year,
                    )
                    is_pr = model.portfolio_return_series(train, weights)
                    is_metrics = _series_metrics(is_pr, config.periods_per_year)
                    oos_metrics = _series_metrics(pr, config.periods_per_year)
                    split_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "scheme": fold.scheme,
                            "strategy": name,
                            "basis": "gross",
                            "is_sharpe": is_metrics["sharpe"],
                            "oos_sharpe": oos_metrics["sharpe"],
                            "status": s_status,
                        }
                    )

                for name, weights in dict(refit.get("equity_benchmarks", {})).items():
                    pr = _equity_portfolio_returns(equity_returns, test_dates, equity_universe, weights)
                    _append_return_rows(return_rows, fold, name, "gross", pr)
                    _append_ledger_row(
                        ledger_rows,
                        fold,
                        name,
                        "gross",
                        pr,
                        "ok",
                        config.periods_per_year,
                    )
                    is_pr = _equity_portfolio_returns(equity_returns, train_dates, equity_universe, weights)
                    split_rows.append(
                        {
                            "fold_id": fold.fold_id,
                            "scheme": fold.scheme,
                            "strategy": name,
                            "basis": "gross",
                            "is_sharpe": _series_metrics(is_pr, config.periods_per_year)["sharpe"],
                            "oos_sharpe": _series_metrics(pr, config.periods_per_year)["sharpe"],
                            "status": "ok",
                        }
                    )

                if cost_scenario_builder is not None and not gross_frame.empty and strategies:
                    cost_result = cost_scenario_builder(
                        gross_frame,
                        strategies,
                        cost_inputs,
                        config=scenario_config,
                        scenarios=("full_spread",),
                    )
                    net_frame = _net_frame_from_cost_result(cost_result)
                    for column in net_frame.columns:
                        text = str(column)
                        if not text.endswith("::full_spread"):
                            continue
                        name = text.rsplit("::", 1)[0]
                        series = net_frame[column]
                        s_status = strategy_status.get(name, "ok")
                        _append_return_rows(return_rows, fold, name, "full_spread_post_cost", series)
                        _append_ledger_row(
                            ledger_rows,
                            fold,
                            name,
                            "full_spread_post_cost",
                            series,
                            s_status,
                            config.periods_per_year,
                        )
        except Exception as exc:
            status = f"error_{type(exc).__name__}"
        finally:
            seconds = time.monotonic() - started
            runtime_rows.append({"fold_id": fold.fold_id, "seconds": seconds, "status": status})
            schedule_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "scheme": fold.scheme,
                    "test_groups": "_".join(str(g) for g in fold.test_groups),
                    "test_start": min(fold.test_dates) if fold.test_dates else pd.NaT,
                    "test_end": max(fold.test_dates) if fold.test_dates else pd.NaT,
                    "n_train": int(refit.get("n_train", len(fold.train_dates))) if refit else int(len(fold.train_dates)),
                    "n_test": int(len(fold.test_dates)),
                    "n_purged": int(len(fold.purged_dates)),
                    "n_embargoed": int(len(fold.embargoed_dates)),
                    "purged_dates": _iso_join(fold.purged_dates),
                    "embargoed_dates": _iso_join(fold.embargoed_dates),
                    "status": status,
                }
            )

    return CVResults(
        fold_schedule=pd.DataFrame(schedule_rows),
        fold_ledger=pd.DataFrame(ledger_rows),
        split_is_oos=pd.DataFrame(split_rows),
        test_month_returns=pd.DataFrame(return_rows),
        runtime_log=pd.DataFrame(runtime_rows),
    )


def assemble_cpcv_paths(test_month_returns: pd.DataFrame, config: CVConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_month_returns.empty:
        return (
            pd.DataFrame(columns=["path_id", "return_date", "strategy", "basis", "ret"]),
            pd.DataFrame(
                columns=[
                    "path_id",
                    "strategy",
                    "basis",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "ann_return",
                    "terminal_wealth",
                    "defaulted",
                    "n_months",
                    "status",
                ]
            ),
        )

    df = test_month_returns.copy()
    df["return_date"] = pd.to_datetime(df["return_date"])
    if "scheme" in df.columns:
        df = df[df["scheme"].astype(str).eq("cpcv")].copy()
    if df.empty:
        return (
            pd.DataFrame(columns=["path_id", "return_date", "strategy", "basis", "ret"]),
            pd.DataFrame(
                columns=[
                    "path_id",
                    "strategy",
                    "basis",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "ann_return",
                    "terminal_wealth",
                    "defaulted",
                    "n_months",
                    "status",
                ]
            ),
        )

    all_dates = _as_sorted_unique_dates(df["return_date"])
    schedule = build_group_schedule(all_dates, config.n_groups)
    expected_folds = build_folds(all_dates, config, "cpcv")
    expected_by_group: dict[int, list[str]] = {g: [] for g in range(config.n_groups)}
    for fold in expected_folds:
        for group in fold.test_groups:
            expected_by_group[group].append(fold.fold_id)
    for group in expected_by_group:
        expected_by_group[group] = sorted(expected_by_group[group])

    path_count = math.comb(config.n_groups - 1, config.n_test_groups - 1)
    path_width = max(2, len(str(max(path_count - 1, 0))))
    fold_data = {str(fid): grp.copy() for fid, grp in df.groupby("fold_id", sort=False)}
    combos = (
        df[["strategy", "basis"]]
        .drop_duplicates()
        .sort_values(["strategy", "basis"])
        .itertuples(index=False, name=None)
    )
    strategy_basis = [(str(strategy), str(basis)) for strategy, basis in combos]
    path_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for path_num in range(path_count):
        path_id = f"path_{path_num:0{path_width}d}"
        chunks: list[pd.DataFrame] = []
        incomplete = False
        for group in range(config.n_groups):
            appearances = expected_by_group.get(group, [])
            if path_num >= len(appearances):
                incomplete = True
                continue
            fold_id = appearances[path_num]
            fold_df = fold_data.get(fold_id)
            if fold_df is None or fold_df.empty:
                incomplete = True
                continue
            group_dates = set(schedule.index[schedule.eq(group)])
            group_df = fold_df[fold_df["return_date"].isin(group_dates)].copy()
            if group_df.empty:
                incomplete = True
                continue
            chunks.append(group_df)
        if incomplete:
            for strategy, basis in strategy_basis:
                metric_rows.append(
                    {
                        "path_id": path_id,
                        "strategy": strategy,
                        "basis": basis,
                        "sharpe": np.nan,
                        "sortino": np.nan,
                        "max_drawdown": np.nan,
                        "ann_return": np.nan,
                        "terminal_wealth": np.nan,
                        "defaulted": False,
                        "n_months": 0,
                        "status": "incomplete",
                    }
                )
            continue

        path_df = pd.concat(chunks, ignore_index=True, sort=False)
        path_df = path_df.sort_values(["return_date", "strategy", "basis"])
        for _, row in path_df.iterrows():
            path_rows.append(
                {
                    "path_id": path_id,
                    "return_date": pd.Timestamp(row["return_date"]),
                    "strategy": row["strategy"],
                    "basis": row["basis"],
                    "ret": row["ret"],
                }
            )
        for strategy, basis in strategy_basis:
            series_df = path_df[
                path_df["strategy"].astype(str).eq(strategy)
                & path_df["basis"].astype(str).eq(basis)
            ].sort_values("return_date")
            series = pd.Series(series_df["ret"].to_numpy(float), index=series_df["return_date"])
            metrics = _series_metrics(series, config.periods_per_year)
            metric_rows.append(
                {
                    "path_id": path_id,
                    "strategy": strategy,
                    "basis": basis,
                    "sharpe": metrics["sharpe"],
                    "sortino": metrics["sortino"],
                    "max_drawdown": metrics["max_drawdown"],
                    "ann_return": metrics["ann_return"],
                    "terminal_wealth": metrics["terminal_wealth"],
                    "defaulted": metrics["defaulted"],
                    "n_months": int(series.dropna().shape[0]),
                    "status": "complete",
                }
            )

    return pd.DataFrame(path_rows), pd.DataFrame(metric_rows)


def probability_of_backtest_overfitting(split_is_oos: pd.DataFrame, basis: str = "gross") -> pd.DataFrame:
    out_columns = ["Basis", "N splits", "N strategies", "PBO", "Median lambda", "Rank correlation IS OOS"]
    if split_is_oos.empty:
        return pd.DataFrame([[basis, 0, 0, np.nan, np.nan, np.nan]], columns=out_columns)

    df = split_is_oos.copy()
    if "basis" in df.columns:
        df = df[df["basis"].astype(str).eq(str(basis))].copy()
    split_col = "fold_id" if "fold_id" in df.columns else "split"
    if split_col not in df.columns:
        df["split"] = "split_00"
        split_col = "split"

    lambdas: list[float] = []
    correlations: list[float] = []
    n_strategy_values: list[int] = []
    for _, grp in df.groupby(split_col, sort=True):
        sub = grp[["strategy", "is_sharpe", "oos_sharpe"]].copy()
        sub["is_sharpe"] = pd.to_numeric(sub["is_sharpe"], errors="coerce")
        sub["oos_sharpe"] = pd.to_numeric(sub["oos_sharpe"], errors="coerce")
        sub = sub.dropna(subset=["is_sharpe", "oos_sharpe"])
        n_strategies = int(sub["strategy"].nunique())
        if n_strategies < 2:
            continue
        n_strategy_values.append(n_strategies)
        best_idx = sub["is_sharpe"].idxmax()
        ranks = sub["oos_sharpe"].rank(method="average", ascending=True)
        omega = float(ranks.loc[best_idx]) / float(n_strategies + 1)
        if omega <= 0.0 or omega >= 1.0:
            continue
        lambdas.append(float(math.log(omega / (1.0 - omega))))
        is_rank = sub["is_sharpe"].rank(method="average", ascending=True)
        oos_rank = sub["oos_sharpe"].rank(method="average", ascending=True)
        corr = is_rank.corr(oos_rank, method="spearman")
        if np.isfinite(corr):
            correlations.append(float(corr))

    pbo = float(np.mean(np.array(lambdas) < 0.0)) if lambdas else np.nan
    if np.isfinite(pbo):
        pbo = min(1.0, max(0.0, pbo))
    return pd.DataFrame(
        [
            {
                "Basis": basis,
                "N splits": int(len(lambdas)),
                "N strategies": int(max(n_strategy_values) if n_strategy_values else 0),
                "PBO": pbo,
                "Median lambda": float(np.median(lambdas)) if lambdas else np.nan,
                "Rank correlation IS OOS": float(np.mean(correlations)) if correlations else np.nan,
            }
        ],
        columns=out_columns,
    )


def tag_regimes(dates: pd.DatetimeIndex, vix_level: pd.Series, event_windows=None) -> pd.DataFrame:
    idx = _as_sorted_unique_dates(dates)
    vix = pd.to_numeric(pd.Series(vix_level), errors="coerce").reindex(idx)
    q1, q2 = vix.quantile([1.0 / 3.0, 2.0 / 3.0])
    tercile = pd.Series("Mid VIX", index=idx)
    tercile[vix <= q1] = "Low VIX"
    tercile[vix >= q2] = "High VIX"

    windows = EVENT_WINDOWS if event_windows is None else event_windows
    events = pd.Series("", index=idx, dtype=object)
    for name, (start, end) in windows.items():
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        events.loc[mask & events.eq("")] = name
    return pd.DataFrame(
        {
            "return_date": idx,
            "vix_tercile": tercile.to_numpy(object),
            "event": events.to_numpy(object),
        }
    )


def cpcv_regime_table(path_month_returns, regime_tags, grouped_inference_fn, metric: str = "sharpe") -> pd.DataFrame:
    if path_month_returns.empty:
        return pd.DataFrame()
    df = path_month_returns.copy()
    df["return_date"] = pd.to_datetime(df["return_date"])
    gross = df[df["basis"].astype(str).eq("gross")].copy()
    if gross.empty:
        return pd.DataFrame()
    averaged = (
        gross.groupby(["return_date", "strategy"], as_index=False)["ret"]
        .mean()
        .pivot(index="return_date", columns="strategy", values="ret")
        .sort_index()
    )
    tags = regime_tags.copy()
    tags["return_date"] = pd.to_datetime(tags["return_date"])
    tags = tags.set_index("return_date").reindex(averaged.index)

    frames: list[pd.DataFrame] = []
    vix_out = grouped_inference_fn(averaged, tags["vix_tercile"], metric=metric)
    if not vix_out.empty:
        vix_out = vix_out.copy()
        vix_out.insert(0, "Regime family", "VIX tercile")
        frames.append(vix_out)

    event_groups = tags["event"].replace("", np.nan)
    event_mask = event_groups.notna()
    if bool(event_mask.any()):
        event_out = grouped_inference_fn(averaged.loc[event_mask], event_groups.loc[event_mask], metric=metric)
        if not event_out.empty:
            event_out = event_out.copy()
            event_out.insert(0, "Regime family", "Event window")
            frames.append(event_out)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


__all__ = [
    "CVConfig",
    "CVResults",
    "CV_STRATEGIES",
    "EVENT_WINDOWS",
    "FoldSpec",
    "assemble_cpcv_paths",
    "build_folds",
    "build_group_schedule",
    "cpcv_regime_table",
    "evaluate_folds",
    "probability_of_backtest_overfitting",
    "refit_fold",
    "tag_regimes",
]
