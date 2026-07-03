"""Tail-path simulation diagnostics for the option-only Markowitz paper.

These routines are evaluation-only diagnostics. They consume already-realized
out-of-sample strategy return paths and never feed simulated paths back into
portfolio construction.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    block_paths: int = 1000
    vol_paths: int = 1000
    block_length: int = 6
    seed: int = 20260625
    periods_per_year: int = 12
    min_egarch_obs: int = 120
    breach_limits: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


def clean_returns(returns: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()


def performance_metrics(returns: pd.Series | np.ndarray, periods: int = 12) -> dict[str, float | int]:
    """Path metrics with limited-liability (absorbing default) wealth.

    Wealth compounds multiplicatively; once path wealth reaches zero or below
    (a period return <= -100%), the path is absorbed at zero: terminal wealth
    is 0, the maximum drawdown is -100%, and the annualized return is -100%.
    This prevents sign-flipping cumulative products from returns below -1.
    """

    r = clean_returns(returns)
    if r.empty:
        return {"n_obs": 0}
    growth = (1.0 + r).to_numpy(float)
    wealth = np.empty(len(growth), dtype=float)
    current = 1.0
    defaulted = False
    for i, g in enumerate(growth):
        if not defaulted:
            current *= g
            if current <= 0.0:
                current = 0.0
                defaulted = True
        wealth[i] = current
    terminal = float(wealth[-1])
    if defaulted:
        ann_return = -1.0
        max_dd = -1.0
    else:
        total = terminal - 1.0
        ann_return = float((1.0 + total) ** (periods / len(r)) - 1.0)
        wealth_series = pd.Series(wealth, index=r.index)
        max_dd = float((wealth_series / wealth_series.cummax() - 1.0).min())
    ann_vol = float(r.std(ddof=1) * math.sqrt(periods)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() * periods / max(ann_vol, 1e-12))
    downside = np.minimum(r.to_numpy(float), 0.0)
    down_dev = float(np.sqrt(np.mean(downside * downside)) * math.sqrt(periods))
    sortino = float("nan") if down_dev <= 1e-12 else float(r.mean() * periods / down_dev)
    q05 = float(r.quantile(0.05))
    cvar = float(-r[r <= q05].mean()) if (r <= q05).any() else float("nan")
    return {
        "n_obs": int(len(r)),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "var_95": -q05,
        "cvar_95": cvar,
        "terminal_wealth": terminal,
        "defaulted": bool(defaulted),
    }


def simulation_assumptions(
    returns: pd.Series,
    *,
    strategy: str,
    basis: str,
    method: str,
    config: SimulationConfig = SimulationConfig(),
) -> dict[str, float | str | int]:
    r = clean_returns(returns)
    if r.empty:
        return {"Strategy": strategy, "Return basis": basis, "Method": method, "Status": "not_applicable", "Reason": "no_returns"}
    lag1 = float(r.autocorr(lag=1)) if len(r) > 2 else float("nan")
    return {
        "Strategy": strategy,
        "Return basis": basis,
        "Method": method,
        "Status": "ok",
        "N obs": int(len(r)),
        "Source start": _date_string(r.index.min()) if isinstance(r.index, pd.DatetimeIndex) else "",
        "Source end": _date_string(r.index.max()) if isinstance(r.index, pd.DatetimeIndex) else "",
        "Periods/year": int(config.periods_per_year),
        "Period mean": float(r.mean()),
        "Period volatility": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "Skewness": float(r.skew()) if len(r) > 2 else float("nan"),
        "Excess kurtosis": float(r.kurtosis()) if len(r) > 3 else float("nan"),
        "Lag1 autocorr": lag1,
        "Block length": int(config.block_length) if method == "circular_block_bootstrap" else np.nan,
        "Path count": int(config.block_paths if method == "circular_block_bootstrap" else config.vol_paths),
        "Interpretation": "tail-path diagnostic; not a trading signal",
    }


def circular_block_path_distribution(
    returns: pd.Series | np.ndarray,
    *,
    n_paths: int = 1000,
    block_length: int = 6,
    seed: int = 20260625,
    periods: int = 12,
) -> pd.DataFrame:
    r = clean_returns(returns).to_numpy(float)
    if len(r) < max(5, min(block_length, 5)):
        return pd.DataFrame([{"status": "not_applicable", "method": "circular_block_bootstrap", "reason": "insufficient_observations"}])
    block = max(1, min(int(block_length), len(r)))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for path_id in range(int(n_paths)):
        idx: list[int] = []
        while len(idx) < len(r):
            start = int(rng.integers(0, len(r)))
            idx.extend(((start + np.arange(block)) % len(r)).tolist())
        sample = pd.Series(r[np.asarray(idx[: len(r)], dtype=int)])
        rows.append(_path_metrics(sample, path_id, "circular_block_bootstrap", len(r), periods) | {"block_length": block})
    return pd.DataFrame(rows)


def volatility_clustered_path_distribution(
    returns: pd.Series | np.ndarray,
    *,
    n_paths: int = 1000,
    seed: int = 20260625,
    periods: int = 12,
    min_egarch_obs: int = 120,
) -> pd.DataFrame:
    r = clean_returns(returns)
    if len(r) < 12:
        return pd.DataFrame([{"status": "not_applicable", "method": "egarch_or_ewma", "reason": "insufficient_observations"}])
    if len(r) < int(min_egarch_obs):
        return _garch_residual_paths(r, n_paths, seed, "insufficient_egarch_obs", periods=periods)
    try:
        from arch import arch_model
    except Exception as exc:  # pragma: no cover - depends on optional package availability
        return _garch_residual_paths(r, n_paths, seed, f"arch_import_failed:{exc}", periods=periods)

    y = (r * 100.0).astype(float)
    try:  # pragma: no cover - exercised when arch is available and enough data exist
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = arch_model(y, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False).fit(
                disp="off", update_freq=0, show_warning=False
            )
    except Exception as exc:  # pragma: no cover
        return _garch_residual_paths(r, n_paths, seed, f"egarch_fit_failed:{exc}", periods=periods)

    params = fit.params
    mu = float(params.get("mu", params.get("Const", y.mean())))
    omega = float(params.get("omega", np.log(max(float(y.var()), 1e-8))))
    alpha = float(params.get("alpha[1]", 0.08))
    gamma = float(params.get("gamma[1]", 0.0))
    beta = float(params.get("beta[1]", 0.90))
    nu = float(max(params.get("nu", 8.0), 2.1))
    if not all(np.isfinite([mu, omega, alpha, gamma, beta, nu])):
        return _garch_residual_paths(r, n_paths, seed, "egarch_fit_nonfinite_parameters", periods=periods)

    std_resid = pd.Series(fit.std_resid).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    std_resid = std_resid[np.abs(std_resid) < 25.0]
    expected_abs_z = float(std_resid.abs().mean()) if len(std_resid) >= 20 else math.sqrt(2.0 / math.pi)
    innovation_source = "fitted_standardized_residual_bootstrap" if len(std_resid) >= 20 else "standardized_student_t"
    rng = np.random.default_rng(seed)
    initial_log_var = float(np.log(max(float(y.var()), 1e-8)))
    rows: list[dict[str, object]] = []
    for path_id in range(int(n_paths)):
        if innovation_source == "fitted_standardized_residual_bootstrap":
            eps = rng.choice(std_resid.to_numpy(float), size=len(r), replace=True)
        else:
            eps = rng.standard_t(nu, size=len(r)) * math.sqrt((nu - 2.0) / nu)
        log_var = np.empty(len(r))
        sim_pct = np.empty(len(r))
        log_var[0] = initial_log_var
        sim_pct[0] = mu + math.sqrt(max(math.exp(log_var[0]), 1e-12)) * eps[0]
        for t in range(1, len(r)):
            log_var[t] = omega + beta * log_var[t - 1] + alpha * (abs(eps[t - 1]) - expected_abs_z) + gamma * eps[t - 1]
            variance_t = max(math.exp(float(np.clip(log_var[t], -30.0, 30.0))), 1e-12)
            sim_pct[t] = mu + math.sqrt(variance_t) * eps[t]
        row = _path_metrics(pd.Series(sim_pct / 100.0), path_id, "egarch_1_1_t", len(r), periods)
        row.update({"innovation_source": innovation_source, "egarch_fit_obs": int(len(y))})
        rows.append(row)
    return pd.DataFrame(rows)


def drawdown_breach_rates(paths: pd.DataFrame, limits: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)) -> dict[str, float]:
    if paths.empty or "max_drawdown" not in paths.columns:
        return {f"Breach {int(limit * 100)}%": float("nan") for limit in limits}
    ok = paths[paths.get("status", "ok").eq("ok")] if "status" in paths.columns else paths
    if ok.empty:
        return {f"Breach {int(limit * 100)}%": float("nan") for limit in limits}
    dd = ok["max_drawdown"].astype(float)
    return {f"Breach {int(limit * 100)}%": float((dd <= -abs(limit)).mean()) for limit in limits}


def summarize_paths(paths: pd.DataFrame, simulation: str) -> dict[str, object]:
    if paths.empty or ("status" in paths.columns and not paths["status"].eq("ok").any()):
        row = paths.iloc[0].to_dict() if not paths.empty else {}
        return {"Status": row.get("status", "not_applicable"), "Reason": row.get("reason", "no_paths")}
    ok = paths[paths.get("status", pd.Series("ok", index=paths.index)).fillna("ok").eq("ok")]

    def q(col: str, p: float) -> float:
        return float(np.nanquantile(ok[col].astype(float), p)) if col in ok.columns else float("nan")

    method = str(ok["method"].iloc[0]) if "method" in ok.columns else simulation
    reason = str(ok["reason"].iloc[0]) if "reason" in ok.columns and ok["reason"].notna().any() else ""
    defaulted_share = float(ok["defaulted"].astype(bool).mean()) if "defaulted" in ok.columns else 0.0
    return {
        "Status": "ok",
        "Simulation": method,
        "Reason": reason,
        "N paths": int(len(ok)),
        "Defaulted path share": defaulted_share,
        "Ann. return p05": q("annualized_return", 0.05),
        "Ann. return p50": q("annualized_return", 0.50),
        "Ann. return p95": q("annualized_return", 0.95),
        "Sortino p05": q("sortino", 0.05),
        "Sortino p50": q("sortino", 0.50),
        "Sortino p95": q("sortino", 0.95),
        "Max DD p05": q("max_drawdown", 0.05),
        "Max DD p50": q("max_drawdown", 0.50),
        "Max DD p95": q("max_drawdown", 0.95),
        "Terminal wealth p05": q("terminal_wealth", 0.05),
        "Terminal wealth p50": q("terminal_wealth", 0.50),
        "Terminal wealth p95": q("terminal_wealth", 0.95),
    }


def run_tail_path_simulations(
    returns_by_basis: dict[str, pd.DataFrame],
    *,
    strategies: tuple[str, ...],
    config: SimulationConfig = SimulationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows: list[dict[str, object]] = []
    assumption_rows: list[dict[str, object]] = []
    breach_rows: list[dict[str, object]] = []
    path_outputs: dict[str, pd.DataFrame] = {}
    for basis, frame in returns_by_basis.items():
        for strategy in strategies:
            if strategy not in frame.columns:
                continue
            returns = pd.Series(frame[strategy], index=frame.index, dtype=float).dropna()
            methods = {
                "circular_block_bootstrap": circular_block_path_distribution(
                    returns,
                    n_paths=config.block_paths,
                    block_length=config.block_length,
                    seed=config.seed,
                    periods=config.periods_per_year,
                ),
                "egarch_or_ewma": volatility_clustered_path_distribution(
                    returns,
                    n_paths=config.vol_paths,
                    seed=config.seed,
                    periods=config.periods_per_year,
                    min_egarch_obs=config.min_egarch_obs,
                ),
            }
            for requested_method, paths in methods.items():
                actual_method = _actual_method(paths, requested_method)
                assumptions = simulation_assumptions(returns, strategy=strategy, basis=basis, method=actual_method, config=config)
                if requested_method == "egarch_or_ewma" and actual_method.startswith(("ewma_residual_fallback", "garch11_residual_fallback")):
                    assumptions["Status"] = "fallback"
                    assumptions["Reason"] = actual_method.replace("garch11_residual_fallback_", "").replace("ewma_residual_fallback_", "")
                assumption_rows.append(assumptions)
                summary_rows.append(
                    {
                        "Return basis": basis,
                        "Strategy": strategy,
                        "Requested method": requested_method,
                        **summarize_paths(paths, actual_method),
                    }
                )
                breach_rows.append(
                    {
                        "Return basis": basis,
                        "Strategy": strategy,
                        "Requested method": requested_method,
                        "Simulation": actual_method,
                        **drawdown_breach_rates(paths, config.breach_limits),
                    }
                )
                out = paths.copy()
                out.insert(0, "Strategy", strategy)
                out.insert(1, "Return basis", basis)
                path_outputs[_path_key(strategy, basis, actual_method)] = out
    return pd.DataFrame(summary_rows), pd.DataFrame(assumption_rows), pd.DataFrame(breach_rows), path_outputs


def compact_simulation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["Return basis", "Strategy", "Simulation", "N paths", "Ann. return p50", "Sortino p50", "Max DD p05", "Max DD p50", "Terminal wealth p05", "Defaulted path share"]
    return summary.reindex(columns=cols)


def compact_assumptions(assumptions: pd.DataFrame) -> pd.DataFrame:
    cols = ["Return basis", "Strategy", "Method", "Status", "Reason", "N obs", "Source start", "Source end", "Block length", "Path count", "Lag1 autocorr"]
    return assumptions.reindex(columns=cols)


def _path_metrics(sample: pd.Series, path_id: int, method: str, source_obs: int, periods: int) -> dict[str, object]:
    return {"path_id": int(path_id), "method": method, "n_source_obs": int(source_obs), "status": "ok", **performance_metrics(sample, periods)}


def _garch_residual_paths(returns: pd.Series, n_paths: int, seed: int, reason: str, periods: int = 12) -> pd.DataFrame:
    """Fixed-parameter GARCH(1,1) residual-bootstrap fallback paths.

    This was previously mislabeled "EWMA": the volatility recursion is a
    fixed-parameter GARCH(1,1) (alpha=0.08, beta=0.90) driven by resampled
    standardized residuals; EWMA volatility is used only to standardize the
    historical residuals.  Simulated returns include the historical mean
    return (``mu``), which was previously dropped; the reconstruction is
    ``r*_t = mu + sigma_t * eps*_t`` (documented behavior).
    """

    r = clean_returns(returns)
    mu = float(r.mean())
    centered = r - mu
    base_vol = max(float(centered.std(ddof=1)), 1e-4)
    vol = centered.ewm(span=6, min_periods=6).std().bfill().fillna(base_vol).clip(lower=0.25 * base_vol)
    resid = (centered / vol).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=-10.0, upper=10.0)
    if len(resid) < 12:
        return pd.DataFrame([{"status": "not_applicable", "method": "garch11_residual_fallback", "reason": "insufficient_residuals"}])
    rng = np.random.default_rng(seed)
    alpha, beta = 0.08, 0.90
    omega = max(float(centered.var()) * (1.0 - alpha - beta), 1e-10)
    rows: list[dict[str, object]] = []
    for path_id in range(int(n_paths)):
        eps = rng.choice(resid.to_numpy(float), size=len(r), replace=True)
        sigma2 = np.empty(len(r))
        sim = np.empty(len(r))
        sigma2[0] = max(float(centered.var()), 1e-10)
        sim[0] = mu + math.sqrt(sigma2[0]) * eps[0]
        for t in range(1, len(r)):
            shock = sim[t - 1] - mu
            sigma2[t] = max(omega + alpha * shock * shock + beta * sigma2[t - 1], 1e-10)
            sim[t] = mu + math.sqrt(sigma2[t]) * eps[t]
        row = _path_metrics(pd.Series(sim), path_id, f"garch11_residual_fallback_{reason}", len(r), periods)
        row["reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _actual_method(paths: pd.DataFrame, requested_method: str) -> str:
    if paths.empty or "method" not in paths.columns:
        return requested_method
    valid = paths["method"].dropna()
    return str(valid.iloc[0]) if not valid.empty else requested_method


def _path_key(strategy: str, basis: str, method: str) -> str:
    return f"{_slug(strategy)}_{_slug(basis)}_{_slug(method)}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _date_string(value: object) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return ""


__all__ = [
    "SimulationConfig",
    "clean_returns",
    "compact_assumptions",
    "compact_simulation_summary",
    "circular_block_path_distribution",
    "drawdown_breach_rates",
    "performance_metrics",
    "run_tail_path_simulations",
    "simulation_assumptions",
    "summarize_paths",
    "volatility_clustered_path_distribution",
]
