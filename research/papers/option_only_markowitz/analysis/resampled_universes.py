"""Joint month-index resampling for option-only Markowitz universes.

The helpers in this module resample month positions, not individual return
values.  A single sampled index path is applied to every strategy or factor
frame involved, preserving cross-sectional dependence by construction while
circular blocks retain short-run serial dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.inference import circular_block_sample
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics


DEFAULT_REFIT_STRATEGY = "Greek Markowitz + VIX"


@dataclass(frozen=True)
class ResampleConfig:
    n_paths: int = 1000
    n_refit_paths: int = 200
    block_length: int = 6
    seed: int = 20260625
    refit_seed: int = 20260626
    periods_per_year: int = 12


def month_index_paths(
    n_months: int,
    n_paths: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return circular block-bootstrap paths over month positions."""

    n = int(n_months)
    p = int(n_paths)
    if n < 0:
        raise ValueError("n_months must be non-negative")
    if p < 0:
        raise ValueError("n_paths must be non-negative")
    if p == 0:
        return np.empty((0, n), dtype=int)
    values = np.arange(n, dtype=int)
    out = np.empty((p, n), dtype=int)
    for path_id in range(p):
        out[path_id] = circular_block_sample(values, rng, block_length).astype(int, copy=False)
    return out


def stratified_month_index_paths(
    regime_labels: pd.Series,
    n_paths: int,
    block_length: int,
    rng,
) -> np.ndarray:
    """Resample month positions within regime strata and fill original slots."""

    labels = pd.Series(regime_labels).reset_index(drop=True)
    n = len(labels)
    p = int(n_paths)
    if p < 0:
        raise ValueError("n_paths must be non-negative")
    if p == 0:
        return np.empty((0, n), dtype=int)

    out = np.empty((p, n), dtype=int)
    regimes = pd.unique(labels)
    for path_id in range(p):
        path = np.empty(n, dtype=int)
        for regime in regimes:
            if pd.isna(regime):
                slots = np.flatnonzero(labels.isna().to_numpy())
            else:
                slots = np.flatnonzero((labels == regime).to_numpy())
            if len(slots) == 0:
                continue
            sample = circular_block_sample(slots.astype(int), rng, block_length).astype(int, copy=False)
            path[slots] = sample
        out[path_id] = path
    return out


def fixed_weight_universe_distribution(
    strategy_returns: pd.DataFrame,
    index_paths: np.ndarray,
    *,
    basis: str,
    universe_family: str,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Evaluate fixed-weight strategies over jointly resampled month paths."""

    returns = pd.DataFrame(strategy_returns)
    paths = _validate_index_paths(index_paths, len(returns))
    rows: list[dict[str, Any]] = []
    for path_id, path in enumerate(paths):
        sample = returns.iloc[path]
        for strategy in returns.columns:
            stats = performance_metrics(sample[strategy], periods_per_year)
            rows.append(
                {
                    "universe_family": universe_family,
                    "basis": basis,
                    "path_id": int(path_id),
                    "strategy": strategy,
                    "sharpe": float(stats.get("sharpe", np.nan)),
                    "sortino": float(stats.get("sortino", np.nan)),
                    "max_drawdown": float(stats.get("max_drawdown", np.nan)),
                    "ann_return": float(stats.get("annualized_return", np.nan)),
                    "terminal_wealth": float(stats.get("terminal_wealth", np.nan)),
                    "defaulted": bool(stats.get("defaulted", False)),
                }
            )
    return pd.DataFrame(rows)


def refit_universe_distribution(
    returns,
    reps,
    universe,
    train_dates: pd.DatetimeIndex,
    test_returns: pd.DataFrame,
    *,
    spec_builder,
    model_factory,
    weights_builder_single,
    under_ret: pd.DataFrame,
    vol_shocks: pd.DataFrame,
    config: ResampleConfig,
) -> pd.DataFrame:
    """Estimate one strategy on pseudo-train samples and score fixed OOS data."""

    train_index = pd.DatetimeIndex(train_dates)
    n_paths = int(config.n_refit_paths)
    if n_paths < 0:
        raise ValueError("config.n_refit_paths must be non-negative")
    if len(train_index) == 0:
        return _refit_status_rows(n_paths, "skipped_no_train_months")

    train_returns = pd.DataFrame(returns).loc[train_index]
    train_under = pd.DataFrame(under_ret).loc[train_index]
    train_vol = pd.DataFrame(vol_shocks).loc[train_index]
    reps_train = _restrict_reps_to_train_dates(pd.DataFrame(reps), train_index, train_returns.columns)
    train_start = train_index.min()
    train_end = train_index.max()
    rng = np.random.default_rng(config.refit_seed)
    paths = month_index_paths(len(train_index), n_paths, config.block_length, rng)

    rows: list[dict[str, Any]] = []
    for path_id, path in enumerate(paths):
        try:
            pseudo_returns = _slot_relabel(train_returns, path, train_index)
            pseudo_under = _slot_relabel(train_under, path, train_index)
            pseudo_vol = _slot_relabel(train_vol, path, train_index)
            spec = spec_builder(reps_train, train_returns, train_start=train_start, train_end=train_end)
            model_result = model_factory(
                spec,
                pseudo_returns,
                reps_train,
                universe,
                train_start=train_start,
                train_end=train_end,
                under_ret=pseudo_under,
                vol_shocks=pseudo_vol,
            )
            model = model_result[0] if isinstance(model_result, tuple) else model_result
            weights = pd.Series(weights_builder_single(model), dtype=float)
            strategy = str(weights.name) if weights.name is not None else DEFAULT_REFIT_STRATEGY
            series = _portfolio_return_series(model, test_returns, weights)
            stats = performance_metrics(series, config.periods_per_year)
            rows.append(
                {
                    "path_id": int(path_id),
                    "strategy": strategy,
                    "sharpe": float(stats.get("sharpe", np.nan)),
                    "sortino": float(stats.get("sortino", np.nan)),
                    "max_drawdown": float(stats.get("max_drawdown", np.nan)),
                    "ann_return": float(stats.get("annualized_return", np.nan)),
                    "gross_nav": float(weights.abs().sum()),
                    "status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by callers on bad inputs
            rows.append(
                {
                    "path_id": int(path_id),
                    "strategy": DEFAULT_REFIT_STRATEGY,
                    "sharpe": np.nan,
                    "sortino": np.nan,
                    "max_drawdown": np.nan,
                    "ann_return": np.nan,
                    "gross_nav": np.nan,
                    "status": f"error:{type(exc).__name__}",
                }
            )
    return pd.DataFrame(rows)


def resampled_summary(paths: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Summarize path distributions against realized Sharpe values."""

    if paths.empty:
        return pd.DataFrame(
            columns=[
                "Universe Family",
                "Basis",
                "Strategy",
                "Realized Value",
                "Path P05 Sharpe",
                "Path P25 Sharpe",
                "Path P50 Sharpe",
                "Path P75 Sharpe",
                "Path P95 Sharpe",
                "Path P50 Max Drawdown",
                "P Sharpe Less Than 0",
                "P Max Drawdown Less Than -0.5",
                "P Default",
            ]
        )
    group_cols = ["strategy", "basis", "universe_family"]
    realized_lookup = _realized_lookup(realized)
    rows: list[dict[str, Any]] = []
    for (strategy, basis, universe_family), group in paths.groupby(group_cols, dropna=False):
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce").to_numpy(float)
        max_dd = pd.to_numeric(group["max_drawdown"], errors="coerce").to_numpy(float)
        defaulted = group["defaulted"].astype(bool).to_numpy() if "defaulted" in group.columns else None
        key = (strategy, basis, universe_family)
        rows.append(
            {
                "Universe Family": universe_family,
                "Basis": basis,
                "Strategy": strategy,
                "Realized Value": realized_lookup.get(key, np.nan),
                "Path P05 Sharpe": _nanquantile(sharpe, 0.05),
                "Path P25 Sharpe": _nanquantile(sharpe, 0.25),
                "Path P50 Sharpe": _nanquantile(sharpe, 0.50),
                "Path P75 Sharpe": _nanquantile(sharpe, 0.75),
                "Path P95 Sharpe": _nanquantile(sharpe, 0.95),
                "Path P50 Max Drawdown": _nanquantile(max_dd, 0.50),
                "P Sharpe Less Than 0": _nanmean(sharpe < 0.0, np.isfinite(sharpe)),
                "P Max Drawdown Less Than -0.5": _nanmean(max_dd < -0.5, np.isfinite(max_dd)),
                "P Default": float(np.mean(defaulted)) if defaulted is not None and len(defaulted) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["Universe Family", "Basis", "Strategy"]).reset_index(drop=True)


def resample_assumptions(config: ResampleConfig, regime_counts: dict, notes: dict) -> pd.DataFrame:
    """Return a small table documenting the resampling design."""

    counts = "; ".join(f"{key}={value}" for key, value in sorted(regime_counts.items())) if regime_counts else "not used"
    rows: list[dict[str, Any]] = [
        {
            "Assumption": "Month-Index Block Bootstrap",
            "Value": "same sampled month path applied to every frame",
            "Notes": "Resamples positions, not values, so cross-sectional rows move intact.",
        },
        {"Assumption": "Block Length", "Value": int(config.block_length), "Notes": "Circular blocks wrap within the source month sequence."},
        {"Assumption": "Fixed Weight Path Count", "Value": int(config.n_paths), "Notes": ""},
        {"Assumption": "Refit Path Count", "Value": int(config.n_refit_paths), "Notes": ""},
        {"Assumption": "Fixed Weight Seed", "Value": int(config.seed), "Notes": ""},
        {"Assumption": "Refit Seed", "Value": int(config.refit_seed), "Notes": ""},
        {"Assumption": "Periods Per Year", "Value": int(config.periods_per_year), "Notes": ""},
        {"Assumption": "Stratification Counts", "Value": counts, "Notes": "Each path preserves historical regime counts."},
        {
            "Assumption": "Slot Relabeling Approximation",
            "Value": "pseudo rows relabeled to original train dates",
            "Notes": "SPY beta/stress augmentation uses slot-calendar dates; constraints only.",
        },
    ]
    for key, value in sorted((notes or {}).items()):
        rows.append({"Assumption": str(key), "Value": value, "Notes": ""})
    return pd.DataFrame(rows)


def _validate_index_paths(index_paths: np.ndarray, n_months: int) -> np.ndarray:
    paths = np.asarray(index_paths, dtype=int)
    if paths.ndim != 2:
        raise ValueError("index_paths must be a 2D array")
    if paths.shape[1] != int(n_months):
        raise ValueError(f"index_paths second dimension must equal number of months ({n_months})")
    if n_months and ((paths < 0).any() or (paths >= n_months).any()):
        raise ValueError("index_paths contains out-of-range month positions")
    return paths


def _restrict_reps_to_train_dates(reps: pd.DataFrame, train_dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    out = reps.copy()
    if "snap_date" in out.columns:
        out = out[pd.to_datetime(out["snap_date"]).isin(train_dates)]
    if "asset_id" in out.columns:
        out = out[out["asset_id"].isin(columns)]
    return out.copy()


def _slot_relabel(frame: pd.DataFrame, path: np.ndarray, train_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = frame.iloc[np.asarray(path, dtype=int)].copy()
    out.index = train_dates
    return out


def _portfolio_return_series(model, test_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if hasattr(model, "portfolio_return_series"):
        return pd.Series(model.portfolio_return_series(test_returns, weights), dtype=float)
    contracts = pd.Index(getattr(model, "contracts", weights.index))
    aligned = pd.DataFrame(test_returns).reindex(columns=contracts).fillna(0.0)
    w = weights.reindex(contracts).fillna(0.0).to_numpy(float)
    return pd.Series(aligned.to_numpy(float) @ w, index=aligned.index)


def _refit_status_rows(n_paths: int, status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path_id": int(path_id),
                "strategy": DEFAULT_REFIT_STRATEGY,
                "sharpe": np.nan,
                "sortino": np.nan,
                "max_drawdown": np.nan,
                "ann_return": np.nan,
                "gross_nav": np.nan,
                "status": status,
            }
            for path_id in range(int(n_paths))
        ]
    )


def _realized_lookup(realized: pd.DataFrame) -> dict[tuple[Any, Any, Any], float]:
    if realized.empty:
        return {}
    metric = _first_existing_column(realized, ["value", "sharpe", "Realized Value", "Sharpe"])
    keys = ["strategy", "basis", "universe_family"]
    title_keys = ["Strategy", "Basis", "Universe Family"]
    if all(key in realized.columns for key in keys):
        use_keys = keys
    elif all(key in realized.columns for key in title_keys):
        use_keys = title_keys
    else:
        return {}
    out: dict[tuple[Any, Any, Any], float] = {}
    if metric is None:
        return out
    for _, row in realized.iterrows():
        key = (row[use_keys[0]], row[use_keys[1]], row[use_keys[2]])
        out[key] = float(row[metric])
    return out


def _first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _nanquantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanquantile(arr, q))


def _nanmean(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(values)[mask]))


__all__ = [
    "ResampleConfig",
    "fixed_weight_universe_distribution",
    "month_index_paths",
    "refit_universe_distribution",
    "resample_assumptions",
    "resampled_summary",
    "stratified_month_index_paths",
]
