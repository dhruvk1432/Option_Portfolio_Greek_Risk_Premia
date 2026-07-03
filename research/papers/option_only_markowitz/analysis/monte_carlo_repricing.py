"""Parametric repriced synthetic-universe Monte Carlo for option-only portfolios.

The simulation is deliberately separated from ``run_empirics``.  Callers pass
train-window state panels and representative contract rows; this module fits a
joint state model, simulates spot/volatility states, and re-prices each option
contract along the synthetic paths using the same payoff-over-entry-mark return
kernel used by the realized expiry proxy.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.papers.option_only_markowitz.analysis.inference import circular_block_sample
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics
from research.papers.option_only_markowitz.analysis.vix_option_panel import black76_price

try:  # pragma: no cover - fallback path depends on optional runtime packaging
    from arch import arch_model
except Exception:  # pragma: no cover
    arch_model = None


ONE_STEP_TENOR_YEARS = 1.0 / 12.0


@dataclass(frozen=True)
class RepriceConfig:
    n_paths: int = 1000
    n_sensitivity_paths: int = 250
    horizon_months: int = 60
    block_length: int = 6
    seed: int = 20260625
    rate: float = 0.02
    iv_floor: float = 0.05
    iv_cap: float = 2.0
    min_mark: float = 0.25
    min_garch_obs: int = 60


@dataclass(frozen=True)
class StateDimension:
    name: str
    kind: str
    underlying: str
    mu: float
    omega: float
    alpha: float
    beta: float
    last_epsilon: float
    last_sigma2: float
    unconditional_mean: float
    unconditional_var: float
    n_obs: int
    fallback: bool
    fallback_reason: str


@dataclass(frozen=True)
class StateModel:
    underlyings: tuple[str, ...]
    dimensions: tuple[StateDimension, ...]
    Z: np.ndarray
    z_index: pd.Index
    initial_iv: pd.Series
    initial_spot: pd.Series
    initial_vix: float
    vix_forward_spot_ratio: float
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None


def fit_joint_state_model(
    under_ret: pd.DataFrame,
    iv_levels: pd.DataFrame,
    vix_level: pd.Series,
    config: RepriceConfig,
) -> StateModel:
    """Fit per-dimension GARCH state dynamics and empirical joint innovations.

    ``under_ret`` is interpreted as monthly simple spot returns and converted to
    log returns.  ``iv_levels`` and ``vix_level`` are levels, so their state
    variables are first log differences.  Each dimension is fitted with a
    GARCH(1,1) model on returns scaled by 100 for numerical stability; the
    stored mean, residuals, conditional variances, and omega are unscaled back
    to decimal monthly units.  If the fit fails or has fewer than
    ``config.min_garch_obs`` observations, a constant-volatility recursion is
    used and the fallback flag records why.
    """

    under = pd.DataFrame(under_ret).copy()
    iv = pd.DataFrame(iv_levels).copy()
    vix = pd.Series(vix_level, dtype=float).copy()

    under.index = pd.to_datetime(under.index)
    iv.index = pd.to_datetime(iv.index)
    vix.index = pd.to_datetime(vix.index)
    underlyings = tuple(str(col) for col in under.columns)

    series_by_dim: dict[str, pd.Series] = {}
    for underlying in underlyings:
        spot = pd.to_numeric(under[underlying], errors="coerce")
        series_by_dim[f"spot:{underlying}"] = _safe_log1p(spot)

    log_iv = _safe_log(iv.reindex(columns=underlyings))
    for underlying in underlyings:
        series_by_dim[f"iv:{underlying}"] = log_iv[underlying].diff()

    series_by_dim["vix:VIX"] = _safe_log(vix).diff()
    state_frame = pd.DataFrame(series_by_dim).replace([np.inf, -np.inf], np.nan)

    dims: list[StateDimension] = []
    z_parts: list[pd.Series] = []
    for name, values in state_frame.items():
        kind, underlying = name.split(":", 1)
        fitted, z = _fit_garch_dimension(
            values,
            name=name,
            kind=kind,
            underlying=underlying,
            config=config,
        )
        dims.append(fitted)
        z_parts.append(z.rename(name))

    z_frame = pd.concat(z_parts, axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if z_frame.empty:
        z_frame = pd.DataFrame(np.zeros((1, len(dims)), dtype=float), columns=[d.name for d in dims])

    initial_iv = _last_finite_by_column(iv.reindex(columns=underlyings), default=0.20).clip(config.iv_floor, config.iv_cap)
    initial_iv.index = pd.Index(underlyings)
    initial_spot = pd.Series(1.0, index=pd.Index(underlyings), dtype=float)
    initial_vix = _last_finite(vix, default=20.0)
    fwd_ratio = _vix_forward_spot_ratio(vix)
    train_index = state_frame.dropna(how="all").index

    return StateModel(
        underlyings=underlyings,
        dimensions=tuple(dims),
        Z=z_frame.to_numpy(float),
        z_index=z_frame.index,
        initial_iv=initial_iv.astype(float),
        initial_spot=initial_spot,
        initial_vix=float(np.clip(initial_vix, 5.0, 150.0)),
        vix_forward_spot_ratio=float(fwd_ratio),
        train_start=pd.Timestamp(train_index.min()) if len(train_index) else None,
        train_end=pd.Timestamp(train_index.max()) if len(train_index) else None,
    )


def simulate_state_paths(
    model: StateModel,
    config: RepriceConfig,
    method: str = "joint_garch_block",
    n_paths: int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate spot, ATM-IV, and VIX-level state paths.

    The default method samples rows of the fitted standardized innovation
    matrix ``Z`` by circular block bootstrap, so each simulated month reuses one
    historical joint innovation row.  ``gaussian_copula`` draws a multivariate
    normal with the training correlation of ``Z`` and maps each dimension back
    through that dimension's empirical inverse CDF.
    """

    if method not in {"joint_garch_block", "gaussian_copula"}:
        raise ValueError("method must be 'joint_garch_block' or 'gaussian_copula'")
    paths = int(n_paths if n_paths is not None else (config.n_sensitivity_paths if method == "gaussian_copula" else config.n_paths))
    horizon = int(config.horizon_months)
    if paths < 0:
        raise ValueError("n_paths must be non-negative")
    if horizon < 0:
        raise ValueError("horizon_months must be non-negative")

    rng = np.random.default_rng(config.seed)
    z = np.asarray(model.Z, dtype=float)
    if z.ndim != 2 or z.shape[1] != len(model.dimensions):
        raise ValueError("model.Z must have shape (n_train, n_dimensions)")
    if z.shape[0] == 0:
        z = np.zeros((1, len(model.dimensions)), dtype=float)

    n_under = len(model.underlyings)
    spot = np.empty((paths, horizon + 1, n_under), dtype=float)
    iv = np.empty((paths, horizon + 1, n_under), dtype=float)
    vix = np.empty((paths, horizon + 1), dtype=float)
    innovations = np.empty((paths, horizon, len(model.dimensions)), dtype=float)
    innovation_rows = np.empty((paths, horizon), dtype=int)

    spot[:, 0, :] = model.initial_spot.reindex(model.underlyings).fillna(1.0).to_numpy(float)
    iv[:, 0, :] = model.initial_iv.reindex(model.underlyings).fillna(0.20).clip(config.iv_floor, config.iv_cap).to_numpy(float)
    vix[:, 0] = float(np.clip(model.initial_vix, 5.0, 150.0))

    if method == "gaussian_copula":
        corr = _nearest_correlation(np.corrcoef(z, rowvar=False))
        chol = np.linalg.cholesky(corr)
        sorted_z = np.sort(z, axis=0)

    for path_id in range(paths):
        if horizon == 0:
            continue
        if method == "joint_garch_block":
            idx = _sample_innovation_rows(z.shape[0], horizon, config.block_length, rng)
            z_path = z[idx]
            innovation_rows[path_id] = idx
        else:
            normals = rng.standard_normal(size=(horizon, len(model.dimensions))) @ chol.T
            uniforms = np.clip(norm.cdf(normals), 1e-12, 1.0 - 1e-12)
            z_path = np.column_stack(
                [
                    np.quantile(sorted_z[:, dim_id], uniforms[:, dim_id], method="linear")
                    for dim_id in range(len(model.dimensions))
                ]
            )
            innovation_rows[path_id] = -1
        innovations[path_id] = z_path
        _reconstruct_path(model, config, z_path, spot[path_id], iv[path_id], vix[path_id])

    return {
        "spot": spot,
        "iv": iv,
        "vix": vix,
        "innovations": innovations,
        "innovation_row_index": innovation_rows,
        "underlyings": np.asarray(model.underlyings, dtype=object),
        "fwd_ratio": np.asarray(float(model.vix_forward_spot_ratio)),
        "method": np.asarray(method, dtype=object),
    }


def contract_static_params(reps: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Freeze representative contract geometry from rows at or before train end."""

    frame = pd.DataFrame(reps).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "underlying",
                "kind",
                "asset_class",
                "log_moneyness",
                "tenor_years",
                "pricing_tenor_years",
                "skew_ratio",
                "anchor_spot",
                "contract_iv",
                "underlying_atm_iv",
            ]
        )
    frame["snap_date"] = pd.to_datetime(frame["snap_date"])
    train = frame[frame["snap_date"].le(pd.Timestamp(train_end))].copy()
    if train.empty:
        return pd.DataFrame()
    train["asset_class"] = train.get("asset_class", "equity_option")

    numeric_cols = ["strike", "spot", "tenor_days", "iv_proxy"]
    for col in numeric_cols:
        if col in train:
            train[col] = pd.to_numeric(train[col], errors="coerce")
    if "tenor_days" not in train and {"expiry", "snap_date"}.issubset(train.columns):
        train["expiry"] = pd.to_datetime(train["expiry"])
        train["tenor_days"] = (train["expiry"] - train["snap_date"]).dt.days

    atm = train[train.get("moneyness_bucket", pd.Series("", index=train.index)).astype(str).eq("atm")]
    atm_iv = atm.groupby("underlying")["iv_proxy"].median() if not atm.empty else pd.Series(dtype=float)
    under_iv = train.groupby("underlying")["iv_proxy"].median()

    rows: list[dict[str, Any]] = []
    for asset_id, group in train.groupby("asset_id", sort=True):
        ordered = group.sort_values("snap_date")
        last = ordered.iloc[-1]
        underlying = str(last.get("underlying", ""))
        contract_iv = _finite_median(ordered.get("iv_proxy", pd.Series(dtype=float)), default=np.nan)
        base_iv = float(atm_iv.get(underlying, np.nan))
        if not np.isfinite(base_iv) or base_iv <= 0.0:
            base_iv = float(under_iv.get(underlying, np.nan))
        if not np.isfinite(base_iv) or base_iv <= 0.0:
            base_iv = contract_iv
        skew = contract_iv / base_iv if np.isfinite(contract_iv) and np.isfinite(base_iv) and base_iv > 0.0 else 1.0

        log_moneyness = _finite_median(np.log(ordered["strike"] / ordered["spot"]), default=0.0)
        tenor_years = _finite_median(ordered["tenor_days"] / 365.0, default=30.0 / 365.0)
        anchor_spot = _last_finite(ordered["spot"], default=100.0)
        rows.append(
            {
                "asset_id": asset_id,
                "underlying": underlying,
                "kind": _normalize_kind_scalar(last.get("kind", "call")),
                "asset_class": str(last.get("asset_class", "equity_option")),
                "log_moneyness": float(log_moneyness),
                "tenor_years": float(max(tenor_years, 1e-8)),
                "pricing_tenor_years": ONE_STEP_TENOR_YEARS,
                "skew_ratio": float(skew if np.isfinite(skew) and skew > 0.0 else 1.0),
                "anchor_spot": float(max(anchor_spot, 1e-12)),
                "contract_iv": float(contract_iv if np.isfinite(contract_iv) and contract_iv > 0.0 else base_iv if np.isfinite(base_iv) else 0.20),
                "underlying_atm_iv": float(base_iv if np.isfinite(base_iv) and base_iv > 0.0 else np.nan),
            }
        )
    return pd.DataFrame(rows).set_index("asset_id").sort_index()


def reprice_contract_returns(states: dict, params: pd.DataFrame, config: RepriceConfig) -> np.ndarray:
    """Re-price static option contracts on simulated state paths.

    Spot paths from ``simulate_state_paths`` are relative paths initialized at
    1.0.  ``contract_static_params`` stores each contract's train-end
    ``anchor_spot``; equity contracts are priced with
    ``S_t = relative_spot_t * anchor_spot``.  This anchored convention keeps the
    realized-kernel entry-mark floor literal: the premium denominator is
    ``max(model_price, config.min_mark)`` in dollars, not a normalized cents
    value.
    """

    params = pd.DataFrame(params).copy()
    if params.empty:
        spot = np.asarray(states["spot"], dtype=float)
        return np.empty((spot.shape[0], max(spot.shape[1] - 1, 0), 0), dtype=float)

    spot_state = np.asarray(states["spot"], dtype=float)
    iv_state = np.asarray(states["iv"], dtype=float)
    vix_state = np.asarray(states["vix"], dtype=float)
    if spot_state.ndim != 3 or iv_state.ndim != 3 or vix_state.ndim != 2:
        raise ValueError("states must contain spot (P,T+1,U), iv (P,T+1,U), and vix (P,T+1)")
    if spot_state.shape[:2] != iv_state.shape[:2] or spot_state.shape[0] != vix_state.shape[0] or spot_state.shape[1] != vix_state.shape[1]:
        raise ValueError("state arrays have incompatible path/time dimensions")

    paths, steps_plus_one, n_under = spot_state.shape
    horizon = max(steps_plus_one - 1, 0)
    n_contracts = len(params)
    out = np.full((paths, horizon, n_contracts), np.nan, dtype=float)
    if horizon == 0:
        return out

    underlyings = [str(x) for x in np.asarray(states.get("underlyings", np.arange(n_under)), dtype=object).tolist()]
    under_pos = {underlying: idx for idx, underlying in enumerate(underlyings)}

    kinds = _normalize_option_type(params["kind"].to_numpy(object))
    log_m = pd.to_numeric(params["log_moneyness"], errors="coerce").fillna(0.0).to_numpy(float)
    if "pricing_tenor_years" in params:
        tenor_source = params["pricing_tenor_years"]
    else:
        tenor_source = pd.Series(ONE_STEP_TENOR_YEARS, index=params.index)
    tenor = pd.to_numeric(tenor_source, errors="coerce").fillna(ONE_STEP_TENOR_YEARS).clip(lower=1e-8).to_numpy(float)
    skew = pd.to_numeric(params.get("skew_ratio", 1.0), errors="coerce").fillna(1.0).clip(lower=1e-8).to_numpy(float)
    anchor = pd.to_numeric(params.get("anchor_spot", 100.0), errors="coerce").fillna(100.0).clip(lower=1e-12).to_numpy(float)
    contract_iv = pd.to_numeric(params.get("contract_iv", 0.20), errors="coerce").fillna(0.20).clip(lower=config.iv_floor, upper=config.iv_cap).to_numpy(float)
    asset_class = params.get("asset_class", pd.Series("equity_option", index=params.index)).astype(str).to_numpy(object)
    underlying_col = params["underlying"].astype(str).to_numpy(object)

    for contract_id, underlying in enumerate(underlying_col):
        is_vix = _is_vix_contract(asset_class[contract_id], underlying)
        kind = kinds[contract_id]
        if is_vix:
            vix_next = vix_state[:, 1:]
            pos = under_pos.get(str(underlying))
            if pos is None:
                continue
            forward = spot_state[:, :-1, pos] * anchor[contract_id]
            strike = forward * math.exp(log_m[contract_id])
            vol = _contract_vol_surface(
                iv_state,
                under_pos,
                str(underlying),
                skew[contract_id],
                contract_iv[contract_id],
                config,
            )
            premium = _black76_price_vec(forward, strike, tenor[contract_id], config.rate, vol, kind)
            payoff = np.maximum(vix_next - strike, 0.0) if kind == "call" else np.maximum(strike - vix_next, 0.0)
        else:
            pos = under_pos.get(str(underlying))
            if pos is None:
                continue
            spot_t = spot_state[:, :-1, pos] * anchor[contract_id]
            spot_next = spot_state[:, 1:, pos] * anchor[contract_id]
            strike = spot_t * math.exp(log_m[contract_id])
            vol = np.clip(iv_state[:, :-1, pos] * skew[contract_id], config.iv_floor, config.iv_cap)
            premium = bs_price_vec(
                spot_t,
                strike,
                tenor[contract_id],
                config.rate,
                0.0,
                vol,
                kind,
            )
            payoff = np.maximum(spot_next - strike, 0.0) if kind == "call" else np.maximum(strike - spot_next, 0.0)
        denom = np.maximum(premium, config.min_mark)
        out[:, :, contract_id] = payoff / denom - 1.0
    return out


def repriced_strategy_paths(
    contract_returns: np.ndarray,
    weights_by_strategy: dict[str, pd.Series],
    contract_index: pd.Index,
    config: RepriceConfig,
) -> pd.DataFrame:
    """Score fixed contract weights on repriced return paths."""

    returns = np.asarray(contract_returns, dtype=float)
    if returns.ndim != 3:
        raise ValueError("contract_returns must have shape (paths, months, contracts)")
    if returns.shape[2] != len(contract_index):
        raise ValueError("contract_index length must match contract_returns third dimension")

    rows: list[dict[str, Any]] = []
    contracts = pd.Index(contract_index)
    for strategy, weights_raw in weights_by_strategy.items():
        weights = pd.Series(weights_raw, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        total_abs = float(weights.abs().sum())
        aligned = weights.reindex(contracts).fillna(0.0)
        covered_abs = float(aligned.abs().sum())
        coverage = 1.0 if total_abs <= 1e-12 else covered_abs / total_abs
        weighted_returns = np.nan_to_num(returns, nan=0.0) @ aligned.to_numpy(float)
        for path_id in range(returns.shape[0]):
            stats = performance_metrics(pd.Series(weighted_returns[path_id]), 12)
            rows.append(
                {
                    "method": "joint_garch_block",
                    "path_id": int(path_id),
                    "strategy": str(strategy),
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


def repriced_summary(paths_frame: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Summarize repriced path Sharpe distributions against realized Sharpe."""

    columns = [
        "Strategy",
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
    if "method" not in paths:
        paths["method"] = "joint_garch_block"

    realized_lookup = _realized_sharpe_lookup(realized)
    rows: list[dict[str, Any]] = []
    for (strategy, method), group in paths.groupby(["strategy", "method"], dropna=False):
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce").to_numpy(float)
        max_dd = pd.to_numeric(group["max_drawdown"], errors="coerce").to_numpy(float)
        defaulted = group.get("defaulted", pd.Series(False, index=group.index)).astype(bool).to_numpy()
        rows.append(
            {
                "Strategy": strategy,
                "Method": method,
                "Realized Sharpe": realized_lookup.get((strategy, method), realized_lookup.get((strategy, None), np.nan)),
                "P05 Sharpe": _nanquantile(sharpe, 0.05),
                "P50 Sharpe": _nanquantile(sharpe, 0.50),
                "P95 Sharpe": _nanquantile(sharpe, 0.95),
                "P50 Max Drawdown": _nanquantile(max_dd, 0.50),
                "P Sharpe Less Than 0": _nanmean(sharpe < 0.0, np.isfinite(sharpe)),
                "P Default": float(np.mean(defaulted)) if len(defaulted) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["Strategy", "Method"]).reset_index(drop=True)


def reprice_assumptions(
    model: StateModel,
    params: pd.DataFrame,
    config: RepriceConfig,
    method: str,
) -> pd.DataFrame:
    """Return an audit table for state dynamics and repricing assumptions."""

    rows: list[dict[str, Any]] = []
    for dim in model.dimensions:
        rows.append(
            {
                "Section": "State Dimension",
                "Item": dim.name,
                "Value": f"mu={dim.mu:.6g}; omega={dim.omega:.6g}; alpha={dim.alpha:.6g}; beta={dim.beta:.6g}",
                "Notes": f"n_obs={dim.n_obs}; fallback={dim.fallback}; reason={dim.fallback_reason}",
            }
        )
    kernel_rows = [
        ("Simulation Method", method, "joint block rows preserve empirical cross-dependence; gaussian_copula is rank-mapped sensitivity"),
        ("Strike Rule", "K_t = entry state_t * exp(train median log moneyness)", "Static moneyness, repriced each month"),
        ("Original Tenor Reference", "train median tenor_days / 365", "Stored as tenor_years for audit only"),
        (
            "Pricing Tenor Rule",
            ONE_STEP_TENOR_YEARS,
            "Synthetic contracts are one-step (1-month) options so premium and payoff horizons match; realized-kernel tenors average ~0.6 months",
        ),
        ("Skew Freeze", "contract median IV / underlying ATM median IV", f"{len(params)} contracts"),
        ("Rate", config.rate, "Continuously-compounded annual rate used in BS and Black-76"),
        ("IV Floors Caps", f"{config.iv_floor} to {config.iv_cap}", "Applied to ATM and skew-adjusted contract IV"),
        ("VIX Level Caps", "5.0 to 150.0", "Applied during state reconstruction"),
        ("Min Mark Rule", config.min_mark, "Entry denominator uses max(model premium, min_mark) in dollars"),
        ("Spot Anchor", "contract train-end spot", "State spot paths are relative to 1.0 and scaled by anchor_spot at repricing"),
        ("VIX Forward Convention", "simulated spot:VX_FRONT entry forward; VIX level settlement proxy at t+1", "Premium uses the VX-front state, payoff uses simulated VIX level"),
        ("Seed", config.seed, ""),
        ("Path Count", config.n_paths if method == "joint_garch_block" else config.n_sensitivity_paths, ""),
        ("Horizon Months", config.horizon_months, ""),
        ("Block Length", config.block_length, ""),
        ("Min GARCH Obs", config.min_garch_obs, ""),
    ]
    rows.extend({"Section": "Kernel", "Item": item, "Value": value, "Notes": notes} for item, value, notes in kernel_rows)
    return pd.DataFrame(rows)


def universe_comparison_table(
    realized: pd.DataFrame,
    resampled_summary: pd.DataFrame | None,
    repriced_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare realized, resampled, and repriced Sharpe distributions."""

    realized_lookup = _realized_sharpe_lookup(realized)
    resampled_lookup = _summary_quantile_lookup(resampled_summary, prefix="Resampled") if resampled_summary is not None else {}
    repriced_lookup = _summary_quantile_lookup(repriced_summary, prefix="Repriced")
    strategies = sorted({key[0] for key in realized_lookup} | set(resampled_lookup) | set(repriced_lookup))
    rows = []
    for strategy in strategies:
        res = resampled_lookup.get(strategy, {})
        rep = repriced_lookup.get(strategy, {})
        rows.append(
            {
                "Strategy": strategy,
                "Sharpe Realized": realized_lookup.get((strategy, None), np.nan),
                "Resampled P05": res.get("P05", np.nan),
                "Resampled P50": res.get("P50", np.nan),
                "Resampled P95": res.get("P95", np.nan),
                "Repriced P05": rep.get("P05", np.nan),
                "Repriced P50": rep.get("P50", np.nan),
                "Repriced P95": rep.get("P95", np.nan),
            }
        )
    return pd.DataFrame(rows)


# Vendored from:
# /Users/dhruvkohli/Desktop/Github Repos/Options_Portfolio_Model/
# DATA_ANALYSIS/data_analysis/lib/pricing_utils.py::bs_price_vec
# Copied to keep this publication module pure and avoid cross-repo imports.
def is_call_option(option_type: Any) -> np.ndarray:
    """Boolean mask accepting common call labels: ``call``/``C``/``c``."""

    raw = np.asarray(option_type).astype(str)
    cleaned = np.char.lower(np.char.strip(raw))
    return (cleaned == "call") | (cleaned == "c")


def normalize_option_type(option_type: Any) -> np.ndarray:
    """Normalize option-right labels to ``call`` or ``put``."""

    return np.where(is_call_option(option_type), "call", "put")


def bs_price_vec(
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    vol: np.ndarray,
    option_type: np.ndarray,
) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    vol = np.asarray(vol, dtype=float)
    sqrt_t = np.sqrt(np.maximum(t, 1e-12))
    sig_sqrt = np.maximum(vol, 1e-8) * sqrt_t
    d1 = (np.log(np.maximum(s, 1e-12) / np.maximum(k, 1e-12)) + (r - q + 0.5 * vol**2) * t) / sig_sqrt
    d2 = d1 - sig_sqrt
    df_r = np.exp(-r * t)
    df_q = np.exp(-q * t)
    call = s * df_q * norm.cdf(d1) - k * df_r * norm.cdf(d2)
    put = k * df_r * norm.cdf(-d2) - s * df_q * norm.cdf(-d1)
    return np.where(is_call_option(option_type), call, put)


def _fit_garch_dimension(
    values: pd.Series,
    *,
    name: str,
    kind: str,
    underlying: str,
    config: RepriceConfig,
) -> tuple[StateDimension, pd.Series]:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < int(config.min_garch_obs):
        return _constant_vol_dimension(clean, name, kind, underlying, f"n_obs<{config.min_garch_obs}")
    if arch_model is None:
        return _constant_vol_dimension(clean, name, kind, underlying, "arch_unavailable")

    try:
        scaled = clean * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off", show_warning=False)
        params = result.params
        mu = float(params.get("mu", params.get("Const", 0.0))) / 100.0
        omega = float(params.get("omega", 0.0)) / 10000.0
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        cond_vol = pd.Series(np.asarray(result.conditional_volatility, dtype=float) / 100.0, index=clean.index)
        resid = pd.Series(np.asarray(result.resid, dtype=float) / 100.0, index=clean.index)
        std_resid = pd.Series(np.asarray(result.std_resid, dtype=float), index=clean.index).replace([np.inf, -np.inf], np.nan).dropna()
        if std_resid.empty or not np.isfinite([mu, omega, alpha, beta]).all():
            raise ValueError("nonfinite_garch_fit")
        last_sigma2 = float(max(cond_vol.iloc[-1] ** 2, 1e-12))
        last_epsilon = float(resid.iloc[-1])
        sample_var = float(clean.var(ddof=1)) if len(clean) > 1 else 1e-8
        denom = 1.0 - alpha - beta
        uncond_var = float(omega / denom) if denom > 1e-8 and omega > 0.0 else sample_var
        dim = StateDimension(
            name=name,
            kind=kind,
            underlying=underlying,
            mu=mu,
            omega=max(omega, 1e-12),
            alpha=max(alpha, 0.0),
            beta=max(beta, 0.0),
            last_epsilon=last_epsilon,
            last_sigma2=last_sigma2,
            unconditional_mean=float(clean.mean()),
            unconditional_var=float(max(uncond_var, 1e-12)),
            n_obs=int(len(clean)),
            fallback=False,
            fallback_reason="",
        )
        return dim, std_resid.reindex(clean.index).dropna()
    except Exception as exc:
        return _constant_vol_dimension(clean, name, kind, underlying, f"garch_failed:{type(exc).__name__}")


def _constant_vol_dimension(clean: pd.Series, name: str, kind: str, underlying: str, reason: str) -> tuple[StateDimension, pd.Series]:
    clean = pd.Series(clean, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    mu = float(clean.mean()) if len(clean) else 0.0
    var = float(clean.var(ddof=1)) if len(clean) > 1 else 1e-8
    var = max(var, 1e-12)
    residual = clean - mu if len(clean) else pd.Series([0.0])
    z = residual / math.sqrt(var)
    if not len(clean):
        z.index = pd.RangeIndex(1)
    dim = StateDimension(
        name=name,
        kind=kind,
        underlying=underlying,
        mu=mu,
        omega=var,
        alpha=0.0,
        beta=0.0,
        last_epsilon=float(residual.iloc[-1]) if len(residual) else 0.0,
        last_sigma2=var,
        unconditional_mean=mu,
        unconditional_var=var,
        n_obs=int(len(clean)),
        fallback=True,
        fallback_reason=reason,
    )
    return dim, z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _reconstruct_path(
    model: StateModel,
    config: RepriceConfig,
    z_path: np.ndarray,
    spot_path: np.ndarray,
    iv_path: np.ndarray,
    vix_path: np.ndarray,
) -> None:
    eps_prev = np.asarray([dim.last_epsilon for dim in model.dimensions], dtype=float)
    sigma2_prev = np.asarray([dim.last_sigma2 for dim in model.dimensions], dtype=float)
    mu = np.asarray([dim.mu for dim in model.dimensions], dtype=float)
    omega = np.asarray([dim.omega for dim in model.dimensions], dtype=float)
    alpha = np.asarray([dim.alpha for dim in model.dimensions], dtype=float)
    beta = np.asarray([dim.beta for dim in model.dimensions], dtype=float)
    spot_dims = {dim.underlying: idx for idx, dim in enumerate(model.dimensions) if dim.kind == "spot"}
    iv_dims = {dim.underlying: idx for idx, dim in enumerate(model.dimensions) if dim.kind == "iv"}
    vix_dim = next((idx for idx, dim in enumerate(model.dimensions) if dim.kind == "vix"), None)

    for step in range(z_path.shape[0]):
        sigma2 = np.maximum(omega + alpha * eps_prev * eps_prev + beta * sigma2_prev, 1e-12)
        eps = np.sqrt(sigma2) * z_path[step]
        shock = mu + eps
        for under_id, underlying in enumerate(model.underlyings):
            spot_path[step + 1, under_id] = spot_path[step, under_id] * math.exp(float(shock[spot_dims[underlying]]))
            iv_path[step + 1, under_id] = float(
                np.clip(iv_path[step, under_id] * math.exp(float(shock[iv_dims[underlying]])), config.iv_floor, config.iv_cap)
            )
        if vix_dim is not None:
            vix_path[step + 1] = float(np.clip(vix_path[step] * math.exp(float(shock[vix_dim])), 5.0, 150.0))
        else:
            vix_path[step + 1] = vix_path[step]
        eps_prev = eps
        sigma2_prev = sigma2


def _sample_innovation_rows(n_rows: int, horizon: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    values = np.arange(max(int(n_rows), 1), dtype=int)
    pieces = []
    while sum(len(piece) for piece in pieces) < horizon:
        pieces.append(circular_block_sample(values, rng, block_length).astype(int, copy=False))
    return np.concatenate(pieces)[:horizon]


def _contract_vol_surface(
    iv_state: np.ndarray,
    under_pos: dict[str, int],
    underlying: str,
    skew_ratio: float,
    contract_iv: float,
    config: RepriceConfig,
) -> np.ndarray:
    pos = under_pos.get(underlying)
    if pos is None:
        base = np.full(iv_state.shape[:2], contract_iv, dtype=float)[:, :-1]
    else:
        base = iv_state[:, :-1, pos] * skew_ratio
    return np.clip(base, config.iv_floor, config.iv_cap)


def _black76_price_vec(forward: np.ndarray, strike: np.ndarray, tenor: float, rate: float, vol: np.ndarray, kind: str) -> np.ndarray:
    f = np.maximum(np.asarray(forward, dtype=float), 1e-12)
    k = np.maximum(np.asarray(strike, dtype=float), 1e-12)
    t = np.maximum(np.asarray(tenor, dtype=float), 1e-12)
    v = np.maximum(np.asarray(vol, dtype=float), 1e-8)
    sigt = v * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * v * v * t) / sigt
    d2 = d1 - sigt
    df = np.exp(-float(rate) * t)
    if kind == "call":
        return df * (f * norm.cdf(d1) - k * norm.cdf(d2))
    return df * (k * norm.cdf(-d2) - f * norm.cdf(-d1))


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    arr = np.asarray(corr, dtype=float)
    if arr.ndim == 0:
        arr = np.array([[1.0]])
    if arr.ndim == 1:
        arr = np.diag(np.ones_like(arr))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 1.0)
    vals, vecs = np.linalg.eigh(arr)
    vals = np.maximum(vals, 1e-8)
    psd = (vecs * vals) @ vecs.T
    diag = np.sqrt(np.maximum(np.diag(psd), 1e-12))
    psd = psd / np.outer(diag, diag)
    np.fill_diagonal(psd, 1.0)
    return psd


def _safe_log1p(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(vals > -1.0, np.log1p(vals), np.nan), index=vals.index)


def _safe_log(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    vals = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")
    return np.log(vals.where(vals > 0.0))


def _last_finite(values: pd.Series, default: float) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.iloc[-1]) if len(clean) else float(default)


def _last_finite_by_column(frame: pd.DataFrame, default: float) -> pd.Series:
    out = {}
    for col in frame.columns:
        out[col] = _last_finite(frame[col], default)
    return pd.Series(out, dtype=float)


def _finite_median(values: pd.Series, default: float) -> float:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.median()) if len(clean) else float(default)


def _vix_forward_spot_ratio(vix_level: pd.Series) -> float:
    for key in ("forward_spot_ratio", "fwd_ratio", "vix_forward_spot_ratio"):
        value = getattr(vix_level, "attrs", {}).get(key)
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    return float("nan")


def _normalize_option_type(option_type: Any) -> np.ndarray:
    return normalize_option_type(option_type)


def _normalize_kind_scalar(value: Any) -> str:
    return str(normalize_option_type([value])[0])


def _is_vix_contract(asset_class: Any, underlying: Any) -> bool:
    text = str(asset_class).lower()
    under = str(underlying).upper()
    return text == "vix_option" or under in {"VIX", "VX_FRONT"} or "VIX" in under


def _nanquantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanquantile(arr, q))


def _nanmean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.asarray(mask, dtype=bool).any():
        return float("nan")
    return float(np.mean(np.asarray(values)[mask]))


def _realized_sharpe_lookup(realized: pd.DataFrame) -> dict[tuple[Any, Any | None], float]:
    frame = pd.DataFrame(realized)
    if frame.empty:
        return {}
    strategy_col = _first_existing_column(frame, ["strategy", "Strategy"])
    method_col = _first_existing_column(frame, ["method", "Method"])
    value_col = _first_existing_column(frame, ["sharpe", "Sharpe", "Realized Sharpe", "Realized Value", "value"])
    if strategy_col is None or value_col is None:
        return {}
    out: dict[tuple[Any, Any | None], float] = {}
    for _, row in frame.iterrows():
        strategy = row[strategy_col]
        method = row[method_col] if method_col is not None else None
        value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]
        if np.isfinite(value):
            out[(strategy, method)] = float(value)
            out.setdefault((strategy, None), float(value))
    return out


def _summary_quantile_lookup(summary: pd.DataFrame | None, prefix: str) -> dict[Any, dict[str, float]]:
    frame = pd.DataFrame(summary)
    if frame.empty:
        return {}
    strategy_col = _first_existing_column(frame, ["Strategy", "strategy"])
    if strategy_col is None:
        return {}
    col_candidates = {
        "P05": [f"{prefix} P05", "P05 Sharpe", "Path P05 Sharpe"],
        "P50": [f"{prefix} P50", "P50 Sharpe", "Path P50 Sharpe"],
        "P95": [f"{prefix} P95", "P95 Sharpe", "Path P95 Sharpe"],
    }
    out: dict[Any, dict[str, float]] = {}
    for _, row in frame.iterrows():
        values = {}
        for key, names in col_candidates.items():
            col = _first_existing_column(frame, names)
            values[key] = float(row[col]) if col is not None and pd.notna(row[col]) else np.nan
        out[row[strategy_col]] = values
    return out


def _first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


__all__ = [
    "RepriceConfig",
    "StateDimension",
    "StateModel",
    "black76_price",
    "bs_price_vec",
    "contract_static_params",
    "fit_joint_state_model",
    "normalize_option_type",
    "reprice_assumptions",
    "reprice_contract_returns",
    "repriced_strategy_paths",
    "repriced_summary",
    "simulate_state_paths",
    "universe_comparison_table",
]
