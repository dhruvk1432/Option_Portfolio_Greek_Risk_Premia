"""Point-in-time conditional option premia for the option-only paper.

The forecasts here are deliberately simple and auditable.  They decompose a long
option's expected return into carry, delta risk premium, volatility/variance risk
premium, skew/tail insurance premium, and a relative-value term, then shrink the
sum strongly toward zero.  The goal is not to fit a black-box alpha model; it is
to make explicit where the optimizer's expected returns come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConditionalPremiaConfig:
    horizon_years: float = 21.0 / 252.0
    historical_weight: float = 0.25
    structural_weight: float = 0.75
    shrinkage_to_zero: float = 0.60
    max_abs_monthly_mu: float = 0.35
    carry_scale: float = 1.0
    equity_scale: float = 0.35
    vrp_scale: float = 0.10
    skew_scale: float = 0.02
    relative_value_scale: float = 0.025
    tail_hedge_credit: float = 0.015


def _safe_zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    sd = float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=x.index)
    return ((x - float(x.mean())) / sd).clip(-3.0, 3.0).fillna(0.0)


def estimate_underlying_premia(
    train_underlying_returns: pd.DataFrame,
    train_vol_shocks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate slow-moving PIT premia from the training window only."""

    under = train_underlying_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    cols = list(under.columns)
    mu_s = under.mean().reindex(cols).fillna(0.0)
    rv = under.std(ddof=1).reindex(cols).fillna(0.0) * np.sqrt(12.0)
    if train_vol_shocks is None or train_vol_shocks.empty:
        vol_mu = pd.Series(0.0, index=cols)
    else:
        vol_mu = train_vol_shocks.reindex(columns=cols).mean().fillna(0.0)
    return pd.DataFrame({"lambda_s": mu_s, "realized_vol": rv, "lambda_vol": vol_mu}, index=cols)


def conditional_expected_returns(
    spec: pd.DataFrame,
    train_option_returns: pd.DataFrame,
    train_underlying_returns: pd.DataFrame,
    train_vol_shocks: pd.DataFrame | None = None,
    config: ConditionalPremiaConfig = ConditionalPremiaConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    """Return shrunk monthly expected option returns and component ledger.

    All inputs must already be restricted to the training window.  The function
    does no forward filling from the future and writes a component table so the
    paper can audit why each option has its expected return.
    """

    frame = spec.copy()
    idx = frame.index
    hist_mu = train_option_returns.reindex(columns=idx).mean().fillna(0.0)
    premia = estimate_underlying_premia(train_underlying_returns, train_vol_shocks)

    mark = pd.to_numeric(frame["mark"], errors="coerce").replace(0.0, np.nan)
    spot = pd.to_numeric(frame.get("spot", 1.0), errors="coerce").fillna(1.0)
    delta_nav = pd.to_numeric(frame["delta"], errors="coerce").fillna(0.0) * spot / mark
    gamma_nav = pd.to_numeric(frame["gamma"], errors="coerce").fillna(0.0) * spot * spot / mark
    vega_nav = pd.to_numeric(frame["vega"], errors="coerce").fillna(0.0) / mark
    theta = pd.to_numeric(frame["theta"], errors="coerce").fillna(0.0) * config.horizon_years / mark
    underlying = frame["underlying"].astype(str)

    lambda_s = underlying.map(premia["lambda_s"]).fillna(0.0)
    realized_vol = underlying.map(premia["realized_vol"]).fillna(float(premia["realized_vol"].median()) if not premia.empty else 0.0)
    lambda_vol = underlying.map(premia["lambda_vol"]).fillna(0.0)
    iv = pd.to_numeric(frame.get("iv_proxy", np.nan), errors="coerce")
    iv_gap = (iv - realized_vol).fillna(0.0)

    kind = frame.get("kind", pd.Series("", index=idx)).astype(str)
    bucket = frame.get("moneyness_bucket", pd.Series("", index=idx)).astype(str)
    asset_class = frame.get("asset_class", pd.Series("equity_option", index=idx)).astype(str)
    is_vix = asset_class.eq("vix_option") | underlying.str.upper().isin(["VX_FRONT", "VIX"])

    equity_premium = config.equity_scale * delta_nav * lambda_s
    # Long high-IV options have negative variance/vol premia; short positions receive it through negative q.
    vrp = -config.vrp_scale * vega_nav * iv_gap
    vol_premium = config.vrp_scale * vega_nav * lambda_vol
    skew = pd.Series(0.0, index=idx)
    skew -= np.where(kind.eq("put") & bucket.str.contains("wing|near", regex=True), config.skew_scale * vega_nav.abs(), 0.0)
    skew -= np.where(is_vix & kind.eq("call") & bucket.str.contains("wing|near", regex=True), config.tail_hedge_credit * vega_nav.abs(), 0.0)
    rv_score = _safe_zscore(iv.groupby(underlying).transform(lambda s: s - s.median())) if len(frame) else pd.Series(dtype=float)
    relative_value = -config.relative_value_scale * rv_score.reindex(idx).fillna(0.0)

    structural = (
        config.carry_scale * theta.fillna(0.0)
        + equity_premium.fillna(0.0)
        + vrp.fillna(0.0)
        + vol_premium.fillna(0.0)
        + skew.fillna(0.0)
        + relative_value.fillna(0.0)
    )
    combined = config.historical_weight * hist_mu.reindex(idx).fillna(0.0) + config.structural_weight * structural
    mu = (1.0 - config.shrinkage_to_zero) * combined
    mu = mu.clip(-config.max_abs_monthly_mu, config.max_abs_monthly_mu).fillna(0.0)

    components = pd.DataFrame(
        {
            "historical_mean": hist_mu.reindex(idx).fillna(0.0),
            "theta_carry": theta.fillna(0.0),
            "equity_premium": equity_premium.fillna(0.0),
            "variance_risk_premium": vrp.fillna(0.0),
            "vol_premium": vol_premium.fillna(0.0),
            "skew_tail_premium": skew.fillna(0.0),
            "relative_value": relative_value.fillna(0.0),
            "shrunk_mu": mu,
        },
        index=idx,
    )
    return mu.rename("conditional_expected_return"), components


def rolling_walk_forward_weights(
    dates: Sequence[pd.Timestamp],
    fit_fn,
    min_train_obs: int = 36,
) -> dict[pd.Timestamp, pd.Series]:
    """Generic rolling driver used by the empirical script.

    ``fit_fn(date, train_start, train_end)`` must return the portfolio weights
    used for the following date.  This helper enforces that each fit receives a
    train end strictly before the test return date.
    """

    out: dict[pd.Timestamp, pd.Series] = {}
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    for pos, dt in enumerate(idx):
        if pos < min_train_obs:
            continue
        train_end = idx[pos - 1]
        train_start = idx[max(0, pos - min_train_obs)]
        weights = fit_fn(pd.Timestamp(dt), pd.Timestamp(train_start), pd.Timestamp(train_end))
        out[pd.Timestamp(dt)] = weights
    return out


__all__ = [
    "ConditionalPremiaConfig",
    "conditional_expected_returns",
    "estimate_underlying_premia",
    "rolling_walk_forward_weights",
]
