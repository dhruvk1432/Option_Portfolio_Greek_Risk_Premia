"""Risk controls and conditional covariance tools for the R1.1 development arm.

This module is intentionally separate from the frozen R1 implementation.  It
contains no retrospective deletion rule: VIX signals alter strategy exposure,
while the underlying market observations remain available to later estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .option_only_markowitz_model import GreekJointMomentSpec


@dataclass(frozen=True)
class RiskOffConfig:
    """Close-to-next-open VIX risk-off policy."""

    vix_threshold: float = 40.0
    manual_exit_date: str = "2020-03-01"
    market_timezone: str = "America/New_York"
    regular_open: time = time(9, 30)
    regular_close: time = time(16, 0)
    fee_per_contract: float = 0.75
    slippage_bps: float = 5.0

    def validate(self) -> None:
        if not np.isfinite(self.vix_threshold) or self.vix_threshold <= 0:
            raise ValueError("vix_threshold must be finite and positive")
        if self.fee_per_contract < 0 or self.slippage_bps < 0:
            raise ValueError("execution costs must be nonnegative")


@dataclass(frozen=True)
class EgarchOverlayConfig:
    """Training-only EGARCH(1,1)-Student-t covariance overlay."""

    lookback_days: int = 756
    min_observations: int = 500
    horizon_days: int = 21
    variance_ratio_floor: float = 0.50
    variance_ratio_ceiling: float = 2.00
    qlike_improvement_required: float = 0.02
    required_coverage: float = 0.95
    bootstrap_draws: int = 1000
    bootstrap_block_length: int = 21
    bootstrap_seed: int = 20260712

    def validate(self) -> None:
        if self.lookback_days < self.min_observations or self.min_observations < 50:
            raise ValueError("EGARCH lookback must cover the minimum observation count")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be positive")
        if not 0 < self.variance_ratio_floor <= self.variance_ratio_ceiling:
            raise ValueError("variance-ratio bounds must be positive and ordered")
        if not 0 < self.required_coverage <= 1:
            raise ValueError("required_coverage must lie in (0, 1]")


def _next_session(after: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp | pd.NaT:
    normalized = pd.DatetimeIndex(pd.to_datetime(sessions)).tz_localize(None).normalize().sort_values().unique()
    later = normalized[normalized > pd.Timestamp(after).tz_localize(None).normalize()]
    return pd.Timestamp(later[0]) if len(later) else pd.NaT


def build_vix_risk_off_events(
    vix_close: pd.Series,
    sessions: Sequence[pd.Timestamp],
    config: RiskOffConfig = RiskOffConfig(),
) -> pd.DataFrame:
    """Translate completed official closes into next-session executions.

    A close at or above the threshold changes the state to risk-off; the first
    later close below it changes the state back to risk-on.  The user-attested
    2020-03-01 instruction is merged with an identical threshold execution so
    it cannot create a duplicate transaction.
    """

    config.validate()
    series = pd.Series(vix_close, dtype=float).dropna().copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    calendar = pd.DatetimeIndex(pd.to_datetime(sessions)).tz_localize(None).normalize().sort_values().unique()
    if len(calendar):
        series = series[(series.index >= calendar.min()) & (series.index <= calendar.max())]
    rows: list[dict[str, object]] = []
    risk_on = True
    for signal_date, close in series.items():
        action: str | None = None
        state_after = "risk_on"
        if risk_on and float(close) >= config.vix_threshold:
            action, risk_on, state_after = "exit", False, "risk_off"
        elif not risk_on and float(close) < config.vix_threshold:
            action, risk_on, state_after = "reenter", True, "risk_on"
        if action is None:
            continue
        execution_date = _next_session(pd.Timestamp(signal_date), calendar)
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "execution_date": execution_date,
                "action": action,
                "vix_close": float(close),
                "threshold": float(config.vix_threshold),
                "state_after": state_after,
                "source": "official_vix_close",
            }
        )

    manual_date = pd.Timestamp(config.manual_exit_date).normalize()
    manual_execution = _next_session(manual_date, calendar)
    rows.append(
        {
            "signal_date": manual_date,
            "execution_date": manual_execution,
            "action": "exit",
            "vix_close": np.nan,
            "threshold": float(config.vix_threshold),
            "state_after": "risk_off",
            "source": "user_attested_manual_2020",
        }
    )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.dropna(subset=["execution_date"]).sort_values(["execution_date", "action", "signal_date"])
    merged: list[dict[str, object]] = []
    for (execution_date, action), group in events.groupby(["execution_date", "action"], sort=True):
        observed = group["vix_close"].dropna()
        merged.append(
            {
                "signal_date": pd.Timestamp(group["signal_date"].min()),
                "execution_date": pd.Timestamp(execution_date),
                "action": action,
                "vix_close": float(observed.iloc[0]) if len(observed) else np.nan,
                "threshold": float(config.vix_threshold),
                "state_after": str(group["state_after"].iloc[-1]),
                "source": "|".join(sorted(set(group["source"].astype(str)))),
                "deduplicated_signal_count": int(len(group)),
            }
        )
    return pd.DataFrame(merged).sort_values(["execution_date", "action"]).reset_index(drop=True)


def risk_off_exposure_calendar(
    events: pd.DataFrame,
    sessions: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Return the executable start-of-session exposure state."""

    calendar = pd.DatetimeIndex(pd.to_datetime(sessions)).tz_localize(None).normalize().sort_values().unique()
    event_frame = events.copy()
    if event_frame.empty:
        return pd.DataFrame({"session": calendar, "risk_state": "risk_on", "exposure_multiplier": 1.0})
    event_frame["execution_date"] = pd.to_datetime(event_frame["execution_date"], errors="coerce").dt.normalize()
    event_frame = event_frame.sort_values(["execution_date", "action"])
    risk_on = True
    rows: list[dict[str, object]] = []
    for session in calendar:
        todays_events = event_frame[event_frame["execution_date"].eq(session)]
        for _, event in todays_events.iterrows():
            risk_on = str(event["action"]) == "reenter"
        rows.append(
            {
                "session": pd.Timestamp(session),
                "risk_state": "risk_on" if risk_on else "risk_off",
                "exposure_multiplier": 1.0 if risk_on else 0.0,
            }
        )
    return pd.DataFrame(rows)


def execute_cbbo_orders(
    orders: pd.DataFrame,
    quotes: pd.DataFrame,
    execution_date: pd.Timestamp,
    config: RiskOffConfig = RiskOffConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fill signed option orders against sequential displayed CBBO size.

    ``order_contracts`` is positive for a buy and negative for a sell.  Buys
    pay the ask; sells receive the bid.  Every quote row is consumed at most
    once, and an incomplete order makes the event execution infeasible.
    """

    config.validate()
    required = {"symbol", "order_contracts"}
    if not required.issubset(orders.columns):
        raise ValueError(f"orders must contain {sorted(required)}")
    column_aliases = {
        "bid": "bid_px_00",
        "ask": "ask_px_00",
        "bid_size": "bid_sz_00",
        "ask_size": "ask_sz_00",
    }
    normalized = quotes.rename(columns={alias: canonical for alias, canonical in column_aliases.items()}).copy()
    quote_required = {"ts_event", "symbol", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
    if not quote_required.issubset(normalized.columns):
        missing = sorted(quote_required - set(normalized.columns))
        normalized = pd.DataFrame(columns=sorted(quote_required)) if normalized.empty else normalized
        if len(normalized):
            raise ValueError(f"quotes missing columns: {missing}")
    if len(normalized):
        ts = pd.to_datetime(normalized["ts_event"], utc=True, errors="coerce")
        local = ts.dt.tz_convert(config.market_timezone)
        target = pd.Timestamp(execution_date).normalize().date()
        in_session = (
            local.dt.date.eq(target)
            & (local.dt.time >= config.regular_open)
            & (local.dt.time <= config.regular_close)
        )
        normalized = normalized.loc[in_session].copy()
        normalized["ts_event"] = ts.loc[in_session]
        normalized = normalized.sort_values(["symbol", "ts_event"])

    fills: list[dict[str, object]] = []
    incomplete_symbols: list[str] = []
    for _, order in orders.iterrows():
        symbol = str(order["symbol"])
        requested = float(order["order_contracts"])
        side = "buy" if requested > 0 else "sell"
        remaining = abs(requested)
        symbol_quotes = normalized[normalized["symbol"].astype(str).eq(symbol)]
        for quote_index, quote in symbol_quotes.iterrows():
            if remaining <= 1e-12:
                break
            price_column = "ask_px_00" if side == "buy" else "bid_px_00"
            size_column = "ask_sz_00" if side == "buy" else "bid_sz_00"
            price = float(quote[price_column])
            displayed = float(quote[size_column])
            opposite = float(quote["bid_px_00"] if side == "buy" else quote["ask_px_00"])
            if not np.isfinite([price, displayed, opposite]).all() or price <= 0 or displayed <= 0:
                continue
            if float(quote["ask_px_00"]) < float(quote["bid_px_00"]):
                continue
            quantity = min(remaining, displayed)
            slip = config.slippage_bps / 10_000.0
            executable_price = price * (1.0 + slip if side == "buy" else 1.0 - slip)
            fee = quantity * config.fee_per_contract
            fills.append(
                {
                    "execution_date": pd.Timestamp(execution_date).normalize(),
                    "ts_event": quote["ts_event"],
                    "quote_row": int(quote_index) if isinstance(quote_index, (int, np.integer)) else str(quote_index),
                    "symbol": symbol,
                    "side": side,
                    "filled_contracts": float(quantity),
                    "displayed_contracts": displayed,
                    "quote_price": price,
                    "execution_price": float(executable_price),
                    "fee": float(fee),
                }
            )
            remaining -= quantity
        if remaining > 1e-9:
            incomplete_symbols.append(symbol)
        fills.append(
            {
                "execution_date": pd.Timestamp(execution_date).normalize(),
                "ts_event": pd.NaT,
                "quote_row": "order_summary",
                "symbol": symbol,
                "side": side,
                "requested_contracts": abs(requested),
                "unfilled_contracts": float(max(remaining, 0.0)),
            }
        )
    fill_frame = pd.DataFrame(fills)
    actual = fill_frame[fill_frame.get("quote_row", pd.Series(dtype=object)).ne("order_summary")].copy()
    summary = {
        "execution_date": pd.Timestamp(execution_date).normalize(),
        "execution_feasible": not incomplete_symbols,
        "incomplete_symbols": sorted(set(incomplete_symbols)),
        "requested_contracts": float(orders["order_contracts"].abs().sum()),
        "filled_contracts": float(actual.get("filled_contracts", pd.Series(dtype=float)).sum()),
        "fees": float(actual.get("fee", pd.Series(dtype=float)).sum()),
        "missing_executable_quotes": bool(incomplete_symbols),
    }
    return fill_frame, summary


def egarch_variance_forecast(
    daily_returns: pd.Series,
    train_end: pd.Timestamp,
    config: EgarchOverlayConfig = EgarchOverlayConfig(),
) -> dict[str, object]:
    """Fit a cutoff-safe EGARCH model and return a monthly variance ratio."""

    config.validate()
    values = pd.Series(daily_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    values.index = pd.DatetimeIndex(pd.to_datetime(values.index)).tz_localize(None)
    values = values.loc[: pd.Timestamp(train_end).tz_localize(None)].tail(config.lookback_days)
    baseline_daily = float(values.var(ddof=1)) if len(values) > 1 else np.nan
    result: dict[str, object] = {
        "train_end": pd.Timestamp(train_end),
        "n_obs": int(len(values)),
        "valid": False,
        "fallback": True,
        "variance_ratio": 1.0,
        "forecast_variance": np.nan,
        "baseline_variance": baseline_daily * config.horizon_days if np.isfinite(baseline_daily) else np.nan,
        "failure_reason": "insufficient_observations",
    }
    if len(values) < config.min_observations or not np.isfinite(baseline_daily) or baseline_daily <= 0:
        return result
    try:
        from arch import arch_model

        model = arch_model(
            100.0 * values,
            mean="Zero",
            vol="EGARCH",
            p=1,
            o=1,
            q=1,
            dist="StudentsT",
            rescale=False,
        )
        fitted = model.fit(disp="off", show_warning=False, update_freq=0)
        one_day_percent = float(fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        forecast = one_day_percent / 10_000.0 * config.horizon_days
        baseline = baseline_daily * config.horizon_days
        raw_ratio = forecast / baseline
        ratio = float(np.clip(raw_ratio, config.variance_ratio_floor, config.variance_ratio_ceiling))
        if not np.isfinite([forecast, ratio]).all() or forecast <= 0:
            raise ValueError("nonfinite EGARCH forecast")
        result.update(
            {
                "valid": True,
                "fallback": False,
                "variance_ratio": ratio,
                "raw_variance_ratio": float(raw_ratio),
                "forecast_variance": float(forecast),
                "baseline_variance": float(baseline),
                "failure_reason": "",
            }
        )
    except Exception as exc:  # diagnostic must fail closed to the base covariance
        result["failure_reason"] = type(exc).__name__
    return result


def apply_egarch_joint_overlay(
    moments: GreekJointMomentSpec,
    variance_ratios: Mapping[str, float],
) -> GreekJointMomentSpec:
    """Scale joint factor/residual covariance without losing cross terms."""

    factor_scales: list[float] = []
    for name in moments.factor_names:
        if name.startswith("r2_"):
            underlying = name[3:]
            factor_scales.append(float(variance_ratios.get(underlying, 1.0)))
        elif name.startswith("r_"):
            underlying = name[2:]
            factor_scales.append(float(np.sqrt(variance_ratios.get(underlying, 1.0))))
        else:  # implied-volatility changes are not daily-return EGARCH targets
            factor_scales.append(1.0)
    scales = np.asarray(factor_scales + [1.0] * len(moments.contract_names), dtype=float)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("EGARCH covariance scales must be finite and positive")
    joint = moments.joint_covariance().to_numpy(float)
    overlaid = scales[:, None] * joint * scales[None, :]
    k = len(moments.factor_names)
    factors, contracts = moments.factor_names, moments.contract_names
    result = GreekJointMomentSpec(
        factor_cov=pd.DataFrame(overlaid[:k, :k], index=factors, columns=factors),
        factor_residual_cov=pd.DataFrame(overlaid[:k, k:], index=factors, columns=contracts),
        residual_cov=pd.DataFrame(overlaid[k:, k:], index=contracts, columns=contracts),
        n_obs=moments.n_obs,
        estimator=f"{moments.estimator}+egarch11_student_t",
    )
    result.validate(factors, contracts)
    return result


def qlike_loss(realized_variance: np.ndarray, forecast_variance: np.ndarray) -> np.ndarray:
    realized = np.maximum(np.asarray(realized_variance, dtype=float), 1e-12)
    forecast = np.maximum(np.asarray(forecast_variance, dtype=float), 1e-12)
    return np.log(forecast) + realized / forecast


def evaluate_egarch_gate(
    forecast_rows: pd.DataFrame,
    *,
    added_survival_failures: int,
    worst_es_deterioration: float,
    config: EgarchOverlayConfig = EgarchOverlayConfig(),
) -> dict[str, object]:
    """Apply the prespecified forecast and survival promotion gate."""

    config.validate()
    rows = forecast_rows.copy()
    expected = int(len(rows))
    valid = rows.get("valid", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
    scored = rows.loc[valid].dropna(subset=["realized_variance", "forecast_variance", "baseline_variance"])
    coverage = float(len(scored) / expected) if expected else 0.0
    if len(scored):
        egarch_loss = qlike_loss(scored["realized_variance"], scored["forecast_variance"])
        base_loss = qlike_loss(scored["realized_variance"], scored["baseline_variance"])
        difference = egarch_loss - base_loss
        denominator = max(abs(float(np.mean(base_loss))), 1e-12)
        relative_improvement = float(-np.mean(difference) / denominator)
        rng = np.random.default_rng(config.bootstrap_seed)
        n = len(difference)
        block = min(max(config.bootstrap_block_length, 1), n)
        draws = np.zeros(config.bootstrap_draws)
        for i in range(config.bootstrap_draws):
            starts = rng.integers(0, n, size=int(np.ceil(n / block)))
            indices = np.concatenate([(start + np.arange(block)) % n for start in starts])[:n]
            draws[i] = float(np.mean(difference[indices]))
        ci_hi = float(np.quantile(draws, 0.95))
        mean_difference = float(np.mean(difference))
    else:
        relative_improvement, ci_hi, mean_difference = np.nan, np.nan, np.nan
    passed = bool(
        coverage >= config.required_coverage
        and np.isfinite(relative_improvement)
        and relative_improvement >= config.qlike_improvement_required
        and ci_hi < 0.0
        and added_survival_failures == 0
        and worst_es_deterioration <= 0.01 + 1e-12
    )
    return {
        "promotion_status": "promote_development_candidate" if passed else "diagnostic_only",
        "passed": passed,
        "forecast_coverage": coverage,
        "valid_forecasts": int(len(scored)),
        "expected_forecasts": expected,
        "mean_qlike_difference": mean_difference,
        "relative_qlike_improvement": relative_improvement,
        "qlike_difference_bootstrap_90_ci_hi": ci_hi,
        "added_survival_failures": int(added_survival_failures),
        "worst_es_deterioration": float(worst_es_deterioration),
    }


__all__ = [
    "EgarchOverlayConfig",
    "RiskOffConfig",
    "apply_egarch_joint_overlay",
    "build_vix_risk_off_events",
    "egarch_variance_forecast",
    "evaluate_egarch_gate",
    "execute_cbbo_orders",
    "qlike_loss",
    "risk_off_exposure_calendar",
]
