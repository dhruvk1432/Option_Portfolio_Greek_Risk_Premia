"""Black-Scholes-Merton pricing primitives used by the paper."""

from __future__ import annotations

import math

from scipy import stats


def _validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    values = (spot, strike, maturity, rate, volatility, dividend_yield)
    try:
        finite = all(math.isfinite(value) for value in values)
    except TypeError as exc:
        raise ValueError("pricing inputs must be finite numbers") from exc
    if not finite:
        raise ValueError("pricing inputs must be finite numbers")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be nonnegative")
    if volatility < 0.0 or (maturity > 0.0 and volatility == 0.0):
        raise ValueError("volatility must be positive before expiry")


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    root_maturity = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_maturity)
    return d1, d1 - volatility * root_maturity


def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    kind: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Return a European option price per share."""

    if kind not in {"call", "put"}:
        raise ValueError(f"unknown option kind {kind!r}")
    _validate_inputs(spot, strike, maturity, rate, volatility, dividend_yield)
    if maturity == 0.0:
        return max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    d1, d2 = _d1_d2(
        spot, strike, maturity, rate, volatility, dividend_yield
    )
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    if kind == "call":
        return float(
            discounted_spot * stats.norm.cdf(d1)
            - discounted_strike * stats.norm.cdf(d2)
        )
    return float(
        discounted_strike * stats.norm.cdf(-d2)
        - discounted_spot * stats.norm.cdf(-d1)
    )


def bs_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    kind: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Return delta, gamma, vega, and annual theta."""

    if kind not in {"call", "put"}:
        raise ValueError(f"unknown option kind {kind!r}")
    _validate_inputs(spot, strike, maturity, rate, volatility, dividend_yield)
    if maturity == 0.0:
        in_the_money = spot > strike if kind == "call" else spot < strike
        delta = (1.0 if kind == "call" else -1.0) * float(in_the_money)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1, d2 = _d1_d2(
        spot, strike, maturity, rate, volatility, dividend_yield
    )
    density = stats.norm.pdf(d1)
    discounted_dividend = math.exp(-dividend_yield * maturity)
    discounted_rate = math.exp(-rate * maturity)
    root_maturity = math.sqrt(maturity)
    delta = (
        discounted_dividend * stats.norm.cdf(d1)
        if kind == "call"
        else -discounted_dividend * stats.norm.cdf(-d1)
    )
    gamma = discounted_dividend * density / (
        spot * volatility * root_maturity
    )
    vega = spot * discounted_dividend * density * root_maturity
    common_theta = (
        -spot
        * discounted_dividend
        * density
        * volatility
        / (2.0 * root_maturity)
    )
    if kind == "call":
        theta = (
            common_theta
            - rate * strike * discounted_rate * stats.norm.cdf(d2)
            + dividend_yield
            * spot
            * discounted_dividend
            * stats.norm.cdf(d1)
        )
    else:
        theta = (
            common_theta
            + rate * strike * discounted_rate * stats.norm.cdf(-d2)
            - dividend_yield
            * spot
            * discounted_dividend
            * stats.norm.cdf(-d1)
        )
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }
