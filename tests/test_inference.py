from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kurtosis, norm, skew

from analysis.inference import (
    circular_block_sample,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def test_probabilistic_and_deflated_sharpe_are_bounded() -> None:
    returns = np.array([0.01, -0.005, 0.012, 0.006, -0.003] * 12)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    dsr = deflated_sharpe_ratio(returns, trials=20)

    assert 0.0 <= psr <= 1.0
    assert 0.0 <= dsr <= 1.0
    assert dsr <= psr


def test_inference_uses_per_period_sharpe_in_finite_sample_variance() -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.012, -0.008])
    benchmark_annualized = 0.25
    period_sharpe = returns.mean() / returns.std(ddof=1)
    variance = (
        1.0
        - skew(returns, bias=False) * period_sharpe
        + 0.25 * (kurtosis(returns, fisher=False, bias=False) - 1.0) * period_sharpe**2
    ) / (len(returns) - 1)
    expected = norm.cdf(
        (period_sharpe - benchmark_annualized / np.sqrt(12.0)) / np.sqrt(variance)
    )

    observed = probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=benchmark_annualized,
        periods_per_year=12,
    )

    assert observed == pytest.approx(expected)


@pytest.mark.parametrize("feasible", [True, False])
def test_sharpe_inference_is_unavailable_after_ruin_or_infeasibility(
    feasible: bool,
) -> None:
    returns = [0.20, -1.0, 0.50, 0.10] if feasible else [0.01, 0.02, -0.01]

    assert np.isnan(probabilistic_sharpe_ratio(returns, feasible=feasible))
    assert np.isnan(deflated_sharpe_ratio(returns, trials=20, feasible=feasible))


def test_block_sample_is_seeded_and_length_preserving() -> None:
    values = np.arange(20)
    first = circular_block_sample(values, block_size=4, seed=7)
    second = circular_block_sample(values, block_size=4, seed=7)

    np.testing.assert_array_equal(first, second)
    assert len(first) == len(values)


def test_inference_validates_inputs() -> None:
    with pytest.raises(ValueError, match="observations"):
        probabilistic_sharpe_ratio([0.1], benchmark_sharpe=0.0)
    with pytest.raises(ValueError, match="trials"):
        deflated_sharpe_ratio([0.1, -0.1, 0.2], trials=0)
