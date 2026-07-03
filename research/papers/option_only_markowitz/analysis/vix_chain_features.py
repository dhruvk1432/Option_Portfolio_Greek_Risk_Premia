"""Diagnostic VIX option-chain state features.

These features describe the VIX option surface state for paper diagnostics only.
They are deliberately not expected-return inputs for the option-only Markowitz
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .vix_option_panel import align_vx_forward, implied_vol_black76, stack_vix_option_shards


FEATURE_COLUMNS = [
    "atm_iv_proxy",
    "skew_proxy",
    "call_wing_premium_share",
    "term_slope",
    "n_contracts",
]


@dataclass(frozen=True)
class VixChainFeatureConfig:
    min_mark: float = 0.05
    atm_band: float = 0.05
    call_wing_moneyness: float = 1.30
    tenor_days: tuple[int, int] = (20, 45)


def build_vix_chain_state_features(
    root: Path,
    snap_dates: Sequence[pd.Timestamp],
    config: VixChainFeatureConfig = VixChainFeatureConfig(),
) -> pd.DataFrame:
    """Build point-in-time VIX chain diagnostics for each decision date.

    The raw chain is loaded through ``stack_vix_option_shards`` from
    ``vix_option_panel.py``, so monthly files are discovered as
    ``root / "data" / "databento_cache" / "opra_vix_chain_*.parquet"`` and
    normalized by parsing OPRA symbols into ``trade_date``, ``expiry``,
    ``kind``, ``strike``, and ``close``.  For a decision date ``d`` this module
    mirrors the VIX representative-panel convention: use the latest observable
    chain snapshot with ``trade_date <= d`` and require it to be no more than
    five calendar days stale.  It does not forward-fill output features; if the
    selected snapshot has no usable contracts in the requested tenor window,
    the returned row is all NaN.

    Formulas, all computed after ``min_mark`` and remaining-tenor filters:

    * ``atm_iv_proxy`` is the median expiry-level ATM volatility proxy.  ATM
      contracts are those nearest ``strike / forward`` of 1.0 within
      ``config.atm_band`` for each expiry.  Native IV columns are used when
      present; otherwise Black-76 implied volatility is inverted from ``close``
      and the matched VX forward.  If neither IV nor VX forward is available,
      ``mark / atm_reference`` is used as a price-based vol-of-vol proxy.
    * ``skew_proxy`` is median OTM call-wing volatility proxy for calls with
      ``strike / forward >= config.call_wing_moneyness`` minus
      ``atm_iv_proxy``.
    * ``call_wing_premium_share`` is call-wing premium divided by total chain
      premium, using ``mark * open_interest`` when an OI column is available
      and populated, otherwise ``mark``.
    * ``term_slope`` is near-expiry ATM proxy minus far-expiry ATM proxy,
      where near and far are the minimum and maximum remaining tenors in the
      configured window.  It is NaN unless at least two expiries have ATM
      proxies.
    * ``n_contracts`` is the number of rows used after the date, tenor, mark,
      and moneyness filters.
    """

    snap_idx = _snap_index(snap_dates)
    if len(snap_idx) == 0:
        return _empty_feature_frame(snap_idx)

    panel = _prepare_chain_panel(root)
    if panel.empty:
        return _empty_feature_frame(snap_idx)

    rows = []
    available_dates = np.array(sorted(panel["trade_date"].dropna().unique()), dtype="datetime64[ns]")
    for snap_date in snap_idx:
        row = _nan_feature_row()
        if not pd.isna(snap_date) and len(available_dates):
            pos = np.searchsorted(available_dates, np.datetime64(snap_date), side="right") - 1
            if pos >= 0:
                source_date = pd.Timestamp(available_dates[pos]).normalize()
                if (pd.Timestamp(snap_date).normalize() - source_date).days <= 5:
                    day = panel[panel["trade_date"].eq(source_date)].copy()
                    row = _features_for_snapshot(day, pd.Timestamp(snap_date).normalize(), config)
        rows.append(row)

    out = pd.DataFrame(rows, index=snap_idx, columns=FEATURE_COLUMNS)
    out.index.name = "snap_date"
    return out.replace([np.inf, -np.inf], np.nan)


def vol_of_vol_regime_table(
    strategy_returns: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature: str = "atm_iv_proxy",
    n_buckets: int = 3,
) -> pd.DataFrame:
    """Condition strategy returns on prior-date VIX chain feature buckets.

    The regime signal is ``features[feature].shift(1)`` after exact date
    alignment to ``strategy_returns.index``.  Thus a return at month ``t`` is
    labelled only with information from the prior decision date.  Rows with a
    missing shifted feature or missing strategy return are dropped from the
    bucket summaries.
    """

    if n_buckets < 1:
        raise ValueError("n_buckets must be at least 1")
    if feature not in features.columns:
        raise KeyError(f"features does not contain {feature!r}")

    returns = pd.DataFrame(strategy_returns).copy()
    returns.index = _normalize_index(returns.index)
    returns = returns[~returns.index.isna()]
    if returns.empty:
        return _empty_regime_table()

    feature_series = pd.to_numeric(features[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    feature_series.index = _normalize_index(feature_series.index)
    feature_series = feature_series[~feature_series.index.isna()]
    feature_series = feature_series.groupby(level=0).last().sort_index()
    prior_feature = feature_series.reindex(returns.index).shift(1)
    buckets = _qcut_buckets(prior_feature, n_buckets)

    rows: list[dict[str, object]] = []
    for strategy in returns.columns:
        r = pd.to_numeric(returns[strategy], errors="coerce").replace([np.inf, -np.inf], np.nan)
        labelled = pd.DataFrame({"return": r, "bucket": buckets}).dropna(subset=["return", "bucket"])
        for bucket, grp in labelled.groupby("bucket", sort=True, observed=True):
            monthly = grp["return"].astype(float)
            std = float(monthly.std(ddof=1)) if len(monthly) > 1 else np.nan
            sharpe = float(monthly.mean() / std * np.sqrt(12.0)) if std and np.isfinite(std) and std > 0 else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "bucket": int(bucket),
                    "n_months": int(len(monthly)),
                    "mean_monthly_return": float(monthly.mean()),
                    "annualized_sharpe": sharpe,
                }
            )

    if not rows:
        return _empty_regime_table()
    out = pd.DataFrame(rows).set_index(["strategy", "bucket"]).sort_index()
    out.attrs["n_months_total"] = int(len(returns))
    out.attrs["n_months_with_nan_prior_feature"] = int(prior_feature.isna().sum())
    return out


def _prepare_chain_panel(root: Path) -> pd.DataFrame:
    panel = stack_vix_option_shards(root)
    if panel.empty:
        return pd.DataFrame()

    out = panel.copy()
    out["trade_date"] = _normalize_index(out["trade_date"])
    out["expiry"] = _normalize_index(out["expiry"])
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    out["kind"] = _normalize_kind(out.get("kind", pd.Series(index=out.index, dtype=object)))

    mark_col = _first_existing(out, ("mark", "close", "premium", "mid", "midpoint", "settle"))
    if mark_col is None:
        return pd.DataFrame()
    out["mark"] = pd.to_numeric(out[mark_col], errors="coerce")
    out["tenor_days_at_trade"] = (out["expiry"] - out["trade_date"]).dt.days
    out["vix_forward"] = _forward_series(out, root)
    out["iv_measure"] = _native_iv_series(out)
    out = _fill_black76_iv(out)

    out = out[
        out["trade_date"].notna()
        & out["expiry"].notna()
        & out["kind"].isin(["call", "put"])
        & out["strike"].gt(0)
        & out["mark"].gt(0)
    ].copy()
    return out.sort_values(["trade_date", "expiry", "kind", "strike"]).reset_index(drop=True)


def _features_for_snapshot(day: pd.DataFrame, snap_date: pd.Timestamp, config: VixChainFeatureConfig) -> dict[str, float]:
    if day.empty:
        return _nan_feature_row()

    low, high = sorted(config.tenor_days)
    usable = day.copy()
    usable["remaining_tenor_days"] = (usable["expiry"] - snap_date).dt.days
    usable = usable[
        usable["remaining_tenor_days"].between(low, high)
        & usable["mark"].ge(config.min_mark)
        & usable["strike"].gt(0)
    ].copy()
    if usable.empty:
        return _nan_feature_row()

    usable = _with_moneyness_and_proxy(usable)
    usable = usable[usable["moneyness"].gt(0)].copy()
    if usable.empty:
        return _nan_feature_row()

    usable["abs_atm_distance"] = (usable["moneyness"] - 1.0).abs()
    atm_candidates = usable[
        usable["abs_atm_distance"].le(config.atm_band)
        & pd.to_numeric(usable["iv_measure"], errors="coerce").gt(0)
    ].copy()
    expiry_atm = _expiry_atm_proxies(atm_candidates)

    atm_iv = float(expiry_atm["atm_iv_proxy"].median()) if not expiry_atm.empty else np.nan
    wing_mask = usable["kind"].eq("call") & usable["moneyness"].ge(config.call_wing_moneyness)
    wing_iv_values = pd.to_numeric(usable.loc[wing_mask, "iv_measure"], errors="coerce")
    wing_iv_values = wing_iv_values[wing_iv_values.gt(0)]
    wing_iv = float(wing_iv_values.median()) if not wing_iv_values.empty else np.nan
    skew = wing_iv - atm_iv if np.isfinite(wing_iv) and np.isfinite(atm_iv) else np.nan

    premium = _premium_weights(usable)
    total_premium = float(premium.sum())
    wing_share = float(premium.loc[wing_mask].sum() / total_premium) if total_premium > 0 else np.nan

    term_slope = np.nan
    if len(expiry_atm) >= 2:
        ordered = expiry_atm.sort_values(["remaining_tenor_days", "expiry"])
        term_slope = float(ordered["atm_iv_proxy"].iloc[0] - ordered["atm_iv_proxy"].iloc[-1])

    return {
        "atm_iv_proxy": atm_iv,
        "skew_proxy": skew,
        "call_wing_premium_share": wing_share,
        "term_slope": term_slope,
        "n_contracts": float(len(usable)),
    }


def _with_moneyness_and_proxy(day: pd.DataFrame) -> pd.DataFrame:
    out = day.copy()
    out["atm_reference"] = pd.to_numeric(out.get("vix_forward", np.nan), errors="coerce")
    refs = _expiry_atm_reference(out)
    missing_ref = ~out["atm_reference"].gt(0)
    if missing_ref.any():
        out.loc[missing_ref, "atm_reference"] = out.loc[missing_ref, "expiry"].map(refs)
    out["moneyness"] = out["strike"] / out["atm_reference"]

    iv = pd.to_numeric(out["iv_measure"], errors="coerce")
    premium_proxy = out["mark"] / out["atm_reference"]
    out["iv_measure"] = iv.where(iv.gt(0), premium_proxy.where(premium_proxy.gt(0)))
    return out


def _expiry_atm_reference(day: pd.DataFrame) -> dict[pd.Timestamp, float]:
    refs: dict[pd.Timestamp, float] = {}
    for expiry, grp in day.groupby("expiry", observed=True):
        pivot = grp.pivot_table(index="strike", columns="kind", values="mark", aggfunc="median")
        if {"call", "put"}.issubset(set(pivot.columns)):
            diff = (pivot["call"] - pivot["put"]).abs().dropna()
            if not diff.empty:
                refs[pd.Timestamp(expiry)] = float(diff.idxmin())
                continue
        strikes = pd.to_numeric(grp["strike"], errors="coerce").dropna()
        if not strikes.empty:
            refs[pd.Timestamp(expiry)] = float(strikes.median())
    return refs


def _expiry_atm_proxies(atm_candidates: pd.DataFrame) -> pd.DataFrame:
    if atm_candidates.empty:
        return pd.DataFrame(columns=["expiry", "remaining_tenor_days", "atm_iv_proxy"])
    nearest = atm_candidates[
        atm_candidates["abs_atm_distance"].eq(atm_candidates.groupby("expiry")["abs_atm_distance"].transform("min"))
    ].copy()
    out = (
        nearest.groupby("expiry", as_index=False, observed=True)
        .agg(remaining_tenor_days=("remaining_tenor_days", "median"), atm_iv_proxy=("iv_measure", "median"))
        .dropna(subset=["atm_iv_proxy"])
    )
    return out[out["atm_iv_proxy"].gt(0)]


def _premium_weights(day: pd.DataFrame) -> pd.Series:
    mark = pd.to_numeric(day["mark"], errors="coerce").clip(lower=0).fillna(0.0)
    oi_col = _first_existing(day, ("open_interest", "open_int", "oi"))
    if oi_col is None:
        return mark
    oi = pd.to_numeric(day[oi_col], errors="coerce")
    if not oi.gt(0).any():
        return mark
    return mark * oi.clip(lower=0).fillna(0.0)


def _forward_series(panel: pd.DataFrame, root: Path) -> pd.Series:
    fwd_col = _first_existing(panel, ("vix_forward", "forward", "forward_price", "fwd", "underlying_forward"))
    if fwd_col is None:
        base = pd.Series(np.nan, index=panel.index, dtype=float)
    else:
        base = pd.to_numeric(panel[fwd_col], errors="coerce")
    if base.gt(0).all():
        return base.astype(float)
    try:
        aligned = align_vx_forward(panel["trade_date"], panel["expiry"], root=root)
        aligned_fwd = pd.to_numeric(aligned["vix_forward"], errors="coerce")
        return base.where(base.gt(0), aligned_fwd).astype(float)
    except Exception:
        return base.astype(float)


def _native_iv_series(panel: pd.DataFrame) -> pd.Series:
    iv_col = _first_existing(panel, ("iv", "implied_vol", "implied_volatility", "iv_proxy", "sigma", "volatility"))
    if iv_col is None:
        return pd.Series(np.nan, index=panel.index, dtype=float)
    iv = pd.to_numeric(panel[iv_col], errors="coerce")
    iv = iv.where(iv.le(5.0), iv / 100.0)
    return iv.astype(float)


def _fill_black76_iv(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    need = (
        out["iv_measure"].isna()
        & out["mark"].gt(0)
        & out["vix_forward"].gt(0)
        & out["strike"].gt(0)
        & out["tenor_days_at_trade"].gt(0)
        & out["kind"].isin(["call", "put"])
    )
    if not need.any():
        return out
    vols = []
    for row in out.loc[need, ["mark", "vix_forward", "strike", "tenor_days_at_trade", "kind"]].itertuples(index=False):
        vols.append(
            implied_vol_black76(
                float(row.mark),
                float(row.vix_forward),
                float(row.strike),
                float(row.tenor_days_at_trade) / 365.0,
                0.02,
                str(row.kind),
            )
        )
    out.loc[need, "iv_measure"] = vols
    return out


def _qcut_buckets(feature_values: pd.Series, n_buckets: int) -> pd.Series:
    buckets = pd.Series(pd.NA, index=feature_values.index, dtype="object")
    valid = pd.to_numeric(feature_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return buckets
    try:
        codes = pd.qcut(valid, q=n_buckets, labels=False, duplicates="drop")
        codes = pd.Series(codes, index=valid.index).dropna()
        if codes.empty:
            codes = pd.Series(0, index=valid.index)
    except ValueError:
        codes = pd.Series(0, index=valid.index)
    buckets.loc[codes.index] = codes.astype(int) + 1
    return buckets


def _snap_index(snap_dates: Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
    idx = _normalize_index(pd.Index(list(snap_dates)))
    idx.name = "snap_date"
    return idx


def _normalize_index(values: object) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce")
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    return pd.DatetimeIndex(idx).normalize()


def _normalize_kind(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().replace({"c": "call", "p": "put"})


def _first_existing(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {str(c).lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _nan_feature_row() -> dict[str, float]:
    return {col: np.nan for col in FEATURE_COLUMNS}


def _empty_feature_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=index, columns=FEATURE_COLUMNS)
    out.index.name = "snap_date"
    return out


def _empty_regime_table() -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays([[], []], names=["strategy", "bucket"])
    return pd.DataFrame(columns=["n_months", "mean_monthly_return", "annualized_sharpe"], index=idx)


__all__ = [
    "VixChainFeatureConfig",
    "build_vix_chain_state_features",
    "vol_of_vol_regime_table",
]
