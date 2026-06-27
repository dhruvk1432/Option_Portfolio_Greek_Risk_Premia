
"""Publication inference utilities for the option-only Markowitz paper.

The empirical runner uses these functions only on out-of-sample return artifacts.
They deliberately use fixed seeds and block resampling so publication tables are
reproducible and do not pretend that monthly option returns are iid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.portfolio.option_only_markowitz_model import performance_stats
from src.portfolio.multi_asset_derivative_portfolio_model import (
    deflated_sharpe_ratio as _dsr,
    probabilistic_sharpe_ratio as _psr,
)


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int = 1000
    seed: int = 20260625
    alpha: float = 0.10
    block_size: int | None = None
    periods_per_year: float = 12.0


def _clean_returns(returns: pd.Series | np.ndarray) -> np.ndarray:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    return r[np.isfinite(r)]


def circular_block_sample(values: np.ndarray, rng: np.random.Generator, block_size: int) -> np.ndarray:
    """Sample a circular block bootstrap path with the same length as ``values``."""

    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return x.copy()
    b = max(1, min(int(block_size), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / b)))
    pieces = []
    for s in starts:
        idx = (np.arange(s, s + b) % n).astype(int)
        pieces.append(x[idx])
    return np.concatenate(pieces)[:n]


def metric_value(returns: np.ndarray, metric: str, periods_per_year: float = 12.0, benchmark: np.ndarray | None = None) -> float:
    s = pd.Series(returns, dtype=float)
    b = pd.Series(benchmark, dtype=float) if benchmark is not None else None
    stats = performance_stats(s, periods_per_year, benchmark_returns=b)
    aliases = {
        "sharpe": "sharpe",
        "sortino": "sortino",
        "calmar": "calmar",
        "omega": "omega",
        "information_ratio": "information_ratio",
        "ann_return": "ann_return",
        "ann_vol": "ann_vol",
    }
    key = aliases.get(metric.lower(), metric)
    return float(stats.get(key, np.nan))


def block_bootstrap_metric_ci(
    returns: pd.Series | np.ndarray,
    metric: str,
    config: BootstrapConfig = BootstrapConfig(),
    benchmark_returns: pd.Series | np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return point, lower, upper CI for a performance metric."""

    r = _clean_returns(returns)
    if len(r) < 4:
        return (np.nan, np.nan, np.nan)
    block_size = config.block_size or max(2, int(round(np.sqrt(len(r)))))
    point = metric_value(r, metric, config.periods_per_year, _clean_returns(benchmark_returns) if benchmark_returns is not None else None)
    rng = np.random.default_rng(config.seed)
    vals = []
    b = _clean_returns(benchmark_returns) if benchmark_returns is not None else None
    for _ in range(config.n_boot):
        idx_path = circular_block_sample(np.arange(len(r)), rng, block_size).astype(int)
        boot_r = r[idx_path]
        boot_b = b[idx_path] if b is not None and len(b) == len(r) else None
        vals.append(metric_value(boot_r, metric, config.periods_per_year, boot_b))
    arr = np.asarray(vals, dtype=float)
    if not np.isfinite(arr).any():
        return float(point), np.nan, np.nan
    lo, hi = np.nanquantile(arr, [config.alpha / 2.0, 1.0 - config.alpha / 2.0])
    return float(point), float(lo), float(hi)


def strategy_metric_inference(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    metrics: tuple[str, ...] = ("sharpe", "sortino", "calmar", "omega", "information_ratio"),
    config: BootstrapConfig = BootstrapConfig(),
) -> pd.DataFrame:
    rows = []
    for strategy in returns.columns:
        bench = benchmark_returns.reindex(returns.index) if benchmark_returns is not None else None
        for metric in metrics:
            point, lo, hi = block_bootstrap_metric_ci(returns[strategy], metric, config, bench)
            rows.append(
                {
                    "Strategy": strategy,
                    "Metric": metric.replace("_", " ").title(),
                    "Estimate": point,
                    "CI lo": lo,
                    "CI hi": hi,
                    "N": int(pd.Series(returns[strategy]).dropna().shape[0]),
                    "Method": "circular block bootstrap",
                    "Block size": config.block_size or max(2, int(round(np.sqrt(max(len(returns), 1))))),
                    "Seed": config.seed,
                }
            )
    return pd.DataFrame(rows)


def hac_ols(y: pd.Series, x: pd.DataFrame, lags: int | None = None) -> dict[str, object]:
    """OLS with Newey--West/HAC standard errors for monthly returns."""

    aligned = pd.concat([pd.Series(y).rename("portfolio"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    xcols = list(x.columns)
    if len(aligned) <= len(xcols) + 2:
        return {
            "alpha": np.nan,
            "alpha_t": np.nan,
            "alpha_se": np.nan,
            "betas": pd.Series(np.nan, index=xcols),
            "r2": np.nan,
            "residual_vol": np.nan,
            "nobs": int(len(aligned)),
            "hac_lags": 0,
        }
    yv = aligned["portfolio"].to_numpy(float)
    xm = np.column_stack([np.ones(len(aligned)), aligned[xcols].to_numpy(float)])
    coef = np.linalg.pinv(xm) @ yv
    resid = yv - xm @ coef
    n, k = xm.shape
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(int(lags), n - 1))
    s = np.zeros((k, k), dtype=float)
    for t in range(n):
        xt = xm[t : t + 1].T
        s += resid[t] * resid[t] * (xt @ xt.T)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = np.zeros((k, k), dtype=float)
        for t in range(lag, n):
            gamma += resid[t] * resid[t - lag] * np.outer(xm[t], xm[t - lag])
        s += weight * (gamma + gamma.T)
    xtx_inv = np.linalg.pinv(xm.T @ xm)
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float((resid @ resid) / denom) if denom > 0 else np.nan
    return {
        "alpha": float(coef[0]),
        "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else np.nan,
        "alpha_se": float(se[0]),
        "betas": pd.Series(coef[1:], index=xcols),
        "r2": float(r2),
        "residual_vol": float(pd.Series(resid).std(ddof=1)),
        "nobs": int(n),
        "hac_lags": int(lags),
    }


def grouped_metric_inference(
    returns: pd.DataFrame,
    groups: pd.Series,
    metric: str = "sharpe",
    config: BootstrapConfig = BootstrapConfig(n_boot=500),
) -> pd.DataFrame:
    rows = []
    groups = groups.reindex(returns.index)
    for strategy in returns.columns:
        for group in [g for g in pd.Series(groups.dropna().unique()).sort_values()]:
            r = returns.loc[groups.eq(group), strategy].dropna()
            point, lo, hi = block_bootstrap_metric_ci(r, metric, config)
            rows.append({"Strategy": strategy, "Group": group, "Metric": metric.title(), "Estimate": point, "CI lo": lo, "CI hi": hi, "N": len(r), "Seed": config.seed})
    return pd.DataFrame(rows)


def sharpe_reality_check(
    returns: pd.DataFrame,
    *,
    config: BootstrapConfig = BootstrapConfig(n_boot=1000),
    sr_star: float = 0.0,
) -> pd.DataFrame:
    """PSR/DSR and block-bootstrap max-Sharpe screen for tested variants."""

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if clean.empty:
        return pd.DataFrame()
    sharpes = []
    moments: dict[str, tuple[float, float, int]] = {}
    for col in clean.columns:
        r = pd.Series(clean[col], dtype=float).dropna()
        if len(r) < 4:
            sharpes.append(np.nan)
            moments[col] = (np.nan, np.nan, len(r))
            continue
        st = performance_stats(r, config.periods_per_year)
        sharpes.append(st["sharpe"])
        moments[col] = (float(r.skew()), float(r.kurtosis() + 3.0), len(r))
    sharpe_ser = pd.Series(sharpes, index=clean.columns, dtype=float)
    block_size = config.block_size or max(2, int(round(np.sqrt(len(clean)))))
    rng = np.random.default_rng(config.seed)
    max_boot = []
    arr = clean.fillna(0.0).to_numpy(float)
    for _ in range(config.n_boot):
        idx = circular_block_sample(np.arange(len(clean)), rng, block_size).astype(int)
        vals = []
        for j in range(arr.shape[1]):
            vals.append(metric_value(arr[idx, j], "sharpe", config.periods_per_year))
        max_boot.append(np.nanmax(vals))
    max_boot_arr = np.asarray(max_boot, dtype=float)
    rows = []
    for col in clean.columns:
        sr = float(sharpe_ser.get(col, np.nan))
        skew, kurt, nobs = moments[col]
        psr = _psr(sr, sr_star, nobs, skew=skew if np.isfinite(skew) else 0.0, kurt=kurt if np.isfinite(kurt) else 3.0)
        dsr = _dsr(sr, sharpe_ser.dropna().to_numpy(float), nobs, skew=skew if np.isfinite(skew) else 0.0, kurt=kurt if np.isfinite(kurt) else 3.0)
        rows.append(
            {
                "Variant": col,
                "Sharpe": sr,
                "Probabilistic Sharpe": psr,
                "Deflated Sharpe": dsr,
                "Bootstrap max Sharpe p95": float(np.nanquantile(max_boot_arr, 0.95)) if len(max_boot_arr) else np.nan,
                "N": nobs,
                "Block size": block_size,
                "Seed": config.seed,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BootstrapConfig",
    "block_bootstrap_metric_ci",
    "circular_block_sample",
    "grouped_metric_inference",
    "hac_ols",
    "sharpe_reality_check",
    "strategy_metric_inference",
]
