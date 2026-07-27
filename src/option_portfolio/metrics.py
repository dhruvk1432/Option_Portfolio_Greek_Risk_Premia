"""Portfolio path metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def validation_status(
    *,
    sharpe_p05: float,
    sharpe_p50: float,
    default_share: float = 0.0,
    minimum_terminal_wealth: float = 1.0,
    feasible: bool = True,
) -> str:
    """Classify a validation distribution from its lower and median Sharpe."""

    if not feasible:
        return "fail_infeasible"
    if (
        not np.isfinite(default_share)
        or default_share > 0.0
        or not np.isfinite(minimum_terminal_wealth)
        or minimum_terminal_wealth <= 0.0
    ):
        return "fail_default"
    if np.isfinite(sharpe_p05) and sharpe_p05 > 0.0:
        return "pass"
    if np.isfinite(sharpe_p50) and sharpe_p50 > 0.0:
        return "mixed"
    return "fail_sharpe"


def performance_metrics(
    returns: pd.Series | np.ndarray,
    periods_per_year: float = 12.0,
    target_return: float = 0.0,
    benchmark_returns: pd.Series | np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Return funded path metrics with absorbing limited liability."""

    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be finite and positive")
    if not np.isfinite(target_return):
        raise ValueError("target_return must be finite")
    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "n_obs": 0,
            "annualized_mean_return": np.nan,
            "cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "downside_annualized_deviation": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "omega": np.nan,
            "information_ratio": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
            "terminal_wealth": np.nan,
            "defaulted": False,
        }

    growth = 1.0 + r.to_numpy(float)
    wealth = np.empty(len(growth), dtype=float)
    current = 1.0
    defaulted = False
    for index, gross_return in enumerate(growth):
        if not defaulted:
            current *= gross_return
            if current <= 0.0:
                current = 0.0
                defaulted = True
        wealth[index] = current

    terminal_wealth = float(wealth[-1])
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    max_drawdown = -1.0 if defaulted else float(np.min(wealth / peaks - 1.0))
    cagr = -1.0 if defaulted else float(
        terminal_wealth ** (periods_per_year / len(r)) - 1.0
    )
    excess = r - target_return
    annualized_mean_return = float(r.mean() * periods_per_year)
    annualized_excess_return = float(excess.mean() * periods_per_year)
    annualized_volatility = (
        float(r.std(ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else np.nan
    )

    downside = np.minimum(excess.to_numpy(float), 0.0)
    downside_deviation = float(
        np.sqrt(np.mean(downside * downside)) * math.sqrt(periods_per_year)
    )
    if defaulted:
        sharpe = np.nan
        sortino = np.nan
    else:
        sharpe = (
            annualized_excess_return / annualized_volatility
            if annualized_volatility > 0.0 and np.isfinite(annualized_volatility)
            else np.nan
        )
        sortino = (
            annualized_excess_return / downside_deviation
            if downside_deviation > 0.0 and np.isfinite(downside_deviation)
            else np.nan
        )

    gains = np.maximum(excess.to_numpy(float), 0.0).sum()
    losses = -np.minimum(excess.to_numpy(float), 0.0).sum()
    omega = float(gains / losses) if losses > 0.0 else np.nan
    information_ratio = np.nan
    if benchmark_returns is not None and not defaulted:
        aligned = pd.concat(
            [r.rename("portfolio"), pd.Series(benchmark_returns).rename("benchmark")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) > 1:
            active = aligned["portfolio"] - aligned["benchmark"]
            active_volatility = float(
                active.std(ddof=1) * math.sqrt(periods_per_year)
            )
            if active_volatility > 0.0:
                information_ratio = float(
                    active.mean() * periods_per_year / active_volatility
                )

    q05 = float(r.quantile(0.05))
    tail = r[r <= q05]
    return {
        "n_obs": int(len(r)),
        "annualized_mean_return": annualized_mean_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "downside_annualized_deviation": downside_deviation,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown < 0.0 else np.nan,
        "omega": omega,
        "information_ratio": information_ratio,
        "var_95": -q05,
        "cvar_95": float(-tail.mean()) if not tail.empty else np.nan,
        "terminal_wealth": terminal_wealth,
        "defaulted": defaulted,
    }
