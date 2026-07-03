
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
from scipy import stats as _scipy_stats

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


def _aligned_strategy_benchmark(
    returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair strategy and benchmark on a common complete-case sample.

    Both series are aligned first (on index when the indexes match, otherwise
    positionally) and rows with a missing value in either series are dropped
    jointly, so bootstrap resampling keeps strategy/benchmark months paired.
    """

    rs = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan)
    bs = pd.Series(benchmark_returns).astype(float).replace([np.inf, -np.inf], np.nan)
    if len(bs) == len(rs) and not rs.index.equals(bs.index):
        # Index-free inputs (arrays) of equal length are paired positionally.
        bs.index = rs.index
    aligned = pd.concat([rs.rename("r"), bs.rename("b")], axis=1).dropna()
    return aligned["r"].to_numpy(float), aligned["b"].to_numpy(float)


def block_bootstrap_metric_ci(
    returns: pd.Series | np.ndarray,
    metric: str,
    config: BootstrapConfig = BootstrapConfig(),
    benchmark_returns: pd.Series | np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return point, lower, upper CI for a performance metric.

    Benchmark handling: joint strategy/benchmark alignment (concat + dropna,
    resampled row-by-row so months stay paired) is applied ONLY to
    benchmark-dependent metrics such as the information ratio.  Standalone
    metrics (Sharpe, Sortino, Calmar, Omega) always use the strategy's own
    complete series; intersecting them with the benchmark would silently
    change the point estimate whenever the benchmark has missing months.
    """

    needs_benchmark = metric in ("information_ratio",)
    if benchmark_returns is not None and needs_benchmark:
        r, b = _aligned_strategy_benchmark(returns, benchmark_returns)
    else:
        r = _clean_returns(returns)
        b = None
    if len(r) < 4:
        return (np.nan, np.nan, np.nan)
    block_size = config.block_size or max(2, int(round(np.sqrt(len(r)))))
    point = metric_value(r, metric, config.periods_per_year, b)
    rng = np.random.default_rng(config.seed)
    vals = []
    for _ in range(config.n_boot):
        idx_path = circular_block_sample(np.arange(len(r)), rng, block_size).astype(int)
        boot_r = r[idx_path]
        boot_b = b[idx_path] if b is not None else None
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
        # Report the block size actually used by the bootstrap for standalone
        # metrics, i.e. computed from the cleaned series length rather than
        # the raw frame length.
        n_used = len(_clean_returns(returns[strategy]))
        block_size = config.block_size or max(2, int(round(np.sqrt(max(n_used, 1)))))
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
                    "Block size": block_size,
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
            "alpha_p": np.nan,
            "alpha_se": np.nan,
            "betas": pd.Series(np.nan, index=xcols),
            "r2": np.nan,
            "residual_vol": np.nan,
            "nobs": int(len(aligned)),
            "hac_df": 0,
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
    # Small-sample degrees-of-freedom correction for the HAC covariance, and
    # p-values from t(n-k) instead of the asymptotic normal.
    dof = max(n - k, 1)
    cov = cov * (n / dof)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float((resid @ resid) / denom) if denom > 0 else np.nan
    alpha_t = float(coef[0] / se[0]) if se[0] > 0 else np.nan
    alpha_p = float(2.0 * _scipy_stats.t.sf(abs(alpha_t), dof)) if np.isfinite(alpha_t) else np.nan
    return {
        "alpha": float(coef[0]),
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "alpha_se": float(se[0]),
        "betas": pd.Series(coef[1:], index=xcols),
        "r2": float(r2),
        "residual_vol": float(pd.Series(resid).std(ddof=1)),
        "nobs": int(n),
        "hac_df": int(dof),
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


def _monthly_sharpe(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd) if sd > 0 else np.nan


def _trial_basis(variant: str, delimiter: str = "::") -> str:
    """Cost-basis group of a variant, e.g. 'X::full_spread' -> 'full_spread'."""

    name = str(variant)
    return name.rsplit(delimiter, 1)[1] if delimiter in name else "unsuffixed"


def sharpe_reality_check(
    returns: pd.DataFrame,
    *,
    config: BootstrapConfig = BootstrapConfig(n_boot=1000),
    sr_star: float = 0.0,
    trial_basis_delimiter: str = "::",
) -> pd.DataFrame:
    """PSR/DSR (per-period units) and a centered max-Sharpe reality check.

    Statistical conventions (fixes to a previously flawed implementation):

    * PSR/DSR follow Bailey & Lopez de Prado and are evaluated in *per-period
      (monthly) Sharpe units*: ``sr_m = mean/std`` with ``T`` monthly
      observations and monthly skew/kurtosis.  ``sr_star`` is interpreted as an
      annualized hurdle and converted to monthly units.
    * The reality check is a *centered* max-statistic test in the spirit of
      White (2000) / Hansen (2005): with ``V = max_k sqrt(T) * SR_hat_k`` and
      bootstrap draws ``V*_b = max_k sqrt(T) * (SR*_{k,b} - SR_hat_k)``, the
      p-value is ``mean(V*_b >= V)``.  Only complete-case rows are used; no
      missing month is imputed with a fake zero return.
    * The DSR trial set is restricted to variants sharing the same cost basis
      (suffix after ``trial_basis_delimiter``) so near-perfectly correlated
      cost replicas of the same strategy do not inflate the trial variance.
      The effective number of trials used is reported per row.
    """

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if clean.empty:
        return pd.DataFrame()
    sqrt_ppy = float(np.sqrt(config.periods_per_year))
    sr_star_m = float(sr_star) / sqrt_ppy

    monthly_sr: dict[str, float] = {}
    moments: dict[str, tuple[float, float, int]] = {}
    for col in clean.columns:
        r = pd.Series(clean[col], dtype=float).dropna()
        if len(r) < 4:
            monthly_sr[col] = np.nan
            moments[col] = (np.nan, np.nan, len(r))
            continue
        monthly_sr[col] = _monthly_sharpe(r.to_numpy(float))
        moments[col] = (float(r.skew()), float(r.kurtosis() + 3.0), int(len(r)))
    sharpe_m = pd.Series(monthly_sr, dtype=float)

    # Centered reality check on the complete-case sample (joint rows).
    complete = clean.dropna(how="any")
    t_complete = len(complete)
    reality_p = np.nan
    centered_p95 = np.nan
    observed_max_stat = np.nan
    block_size = config.block_size or max(2, int(round(np.sqrt(max(t_complete, 1)))))
    if t_complete >= 8:
        arr = complete.to_numpy(float)
        sr_hat = np.array([_monthly_sharpe(arr[:, j]) for j in range(arr.shape[1])], dtype=float)
        observed_max_stat = float(np.sqrt(t_complete) * np.nanmax(sr_hat))
        rng = np.random.default_rng(config.seed)
        centered_max = np.empty(config.n_boot, dtype=float)
        for i in range(config.n_boot):
            idx = circular_block_sample(np.arange(t_complete), rng, block_size).astype(int)
            boot = arr[idx, :]
            sr_boot = np.array([_monthly_sharpe(boot[:, j]) for j in range(boot.shape[1])], dtype=float)
            centered_max[i] = float(np.sqrt(t_complete) * np.nanmax(sr_boot - sr_hat))
        reality_p = float(np.mean(centered_max >= observed_max_stat))
        centered_p95 = float(np.nanquantile(centered_max, 0.95))

    basis_groups = {col: _trial_basis(col, trial_basis_delimiter) for col in clean.columns}
    rows = []
    for col in clean.columns:
        sr_m = float(sharpe_m.get(col, np.nan))
        skew, kurt, nobs = moments[col]
        skew_use = skew if np.isfinite(skew) else 0.0
        kurt_use = kurt if np.isfinite(kurt) else 3.0
        psr = _psr(sr_m, sr_star_m, nobs, skew=skew_use, kurt=kurt_use)
        # DSR trials: same-cost-basis strategy variants only (monthly units).
        trial_cols = [c for c, g in basis_groups.items() if g == basis_groups[col]]
        trials = sharpe_m.reindex(trial_cols).dropna().to_numpy(float)
        dsr = _dsr(sr_m, trials, nobs, skew=skew_use, kurt=kurt_use)
        rows.append(
            {
                "Variant": col,
                "Sharpe": sr_m * sqrt_ppy,
                "Monthly Sharpe": sr_m,
                "Probabilistic Sharpe": psr,
                "Deflated Sharpe": dsr,
                "DSR trials": int(len(trials)),
                "Reality check p": reality_p,
                "Centered max stat": observed_max_stat,
                "Centered max p95": centered_p95,
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
