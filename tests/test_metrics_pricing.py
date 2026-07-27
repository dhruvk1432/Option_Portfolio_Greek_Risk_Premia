from __future__ import annotations

import math

import numpy as np
import pytest

from option_portfolio.metrics import performance_metrics, validation_status
from option_portfolio.pricing import bs_greeks, bs_price


def test_drawdown_starts_from_initial_nav() -> None:
    metrics = performance_metrics([-0.10, 0.0])

    assert metrics["max_drawdown"] == pytest.approx(-0.10)


def test_annualized_mean_and_cagr_are_distinct() -> None:
    metrics = performance_metrics([0.10, -0.10])

    assert metrics["annualized_mean_return"] == pytest.approx(0.0)
    assert metrics["cagr"] < 0.0


def test_sortino_uses_excess_return_in_the_numerator() -> None:
    metrics = performance_metrics([0.02, 0.0], target_return=0.01)

    assert metrics["sortino"] == pytest.approx(0.0)
    assert metrics["sharpe"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("periods_per_year", "target_return"),
    [(0.0, 0.0), (-12.0, 0.0), (np.inf, 0.0), (12.0, np.nan)],
)
def test_metrics_reject_invalid_annualization_inputs(
    periods_per_year: float,
    target_return: float,
) -> None:
    with pytest.raises(ValueError):
        performance_metrics(
            [0.01, 0.02],
            periods_per_year=periods_per_year,
            target_return=target_return,
        )


def test_ruin_makes_ratio_metrics_unavailable() -> None:
    metrics = performance_metrics([0.20, -1.0, 0.50])

    assert metrics["defaulted"] is True
    assert metrics["terminal_wealth"] == 0.0
    assert math.isnan(metrics["sharpe"])
    assert math.isnan(metrics["sortino"])


def test_default_and_infeasibility_override_sharpe() -> None:
    assert validation_status(sharpe_p05=1.0, sharpe_p50=2.0, default_share=0.1) == "fail_default"
    assert validation_status(
        sharpe_p05=1.0,
        sharpe_p50=2.0,
        minimum_terminal_wealth=0.0,
    ) == "fail_default"
    assert validation_status(sharpe_p05=1.0, sharpe_p50=2.0, feasible=False) == "fail_infeasible"


def test_black_scholes_value_and_greeks_are_finite() -> None:
    value = bs_price(100.0, 100.0, 1.0, 0.05, 0.20, "call")
    greeks = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.20, "call")

    assert value == pytest.approx(10.4506, abs=1e-4)
    assert set(greeks) == {"delta", "gamma", "vega", "theta"}
    assert np.isfinite(list(greeks.values())).all()


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (np.nan, 100.0, 1.0, 0.05, 0.20),
        (-1.0, 100.0, 0.0, 0.05, 0.20),
        (100.0, 100.0, -1.0, 0.05, 0.20),
        (100.0, 100.0, 1.0, np.inf, 0.20),
        (100.0, 100.0, 1.0, 0.05, 0.0),
        (100.0, 100.0, 0.0, 0.05, np.nan),
    ],
)
def test_black_scholes_rejects_invalid_domain_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    with pytest.raises(ValueError):
        bs_price(spot, strike, maturity, rate, volatility)
    with pytest.raises(ValueError):
        bs_greeks(spot, strike, maturity, rate, volatility)
