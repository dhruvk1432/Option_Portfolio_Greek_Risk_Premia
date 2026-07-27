"""Small, deterministic inference helpers used by the paper."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew


def _clean_returns(returns: Sequence[float]) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("at least two finite observations are required")
    return values


def _sharpe_is_available(values: np.ndarray, feasible: bool) -> bool:
    if not feasible:
        return False
    wealth = 1.0
    for gross_return in 1.0 + values:
        wealth *= gross_return
        if wealth <= 0.0:
            return False
    return True


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 12.0,
    feasible: bool = True,
) -> float:
    """Probability that the annualized Sharpe exceeds ``benchmark_sharpe``."""

    values = _clean_returns(returns)
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    if not _sharpe_is_available(values, feasible):
        return math.nan
    volatility = float(values.std(ddof=1))
    if volatility == 0.0:
        return float(values.mean() > 0.0)
    sharpe = float(values.mean() / volatility)
    benchmark = float(benchmark_sharpe / math.sqrt(periods_per_year))
    return _sharpe_probability(values, sharpe, benchmark)


def _sharpe_probability(
    values: np.ndarray,
    sharpe: float,
    benchmark: float,
) -> float:
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    variance = (
        1.0
        - sample_skew * sharpe
        + 0.25 * (sample_kurtosis - 1.0) * sharpe**2
    ) / (len(values) - 1)
    if not np.isfinite(variance) or variance <= 0.0:
        return float(sharpe > benchmark)
    return float(norm.cdf((sharpe - benchmark) / math.sqrt(variance)))


def deflated_sharpe_ratio(
    returns: Sequence[float],
    trials: int,
    periods_per_year: float = 12.0,
    feasible: bool = True,
) -> float:
    """Probabilistic Sharpe ratio after a multiple-testing benchmark."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    values = _clean_returns(returns)
    if not _sharpe_is_available(values, feasible):
        return math.nan
    volatility = float(values.std(ddof=1))
    if volatility == 0.0:
        return float(values.mean() > 0.0)
    observed = float(values.mean() / volatility)
    if trials == 1:
        benchmark = 0.0
    else:
        standard_error = math.sqrt(max((1.0 + 0.5 * observed**2) / (len(values) - 1), 0.0))
        euler_gamma = 0.5772156649015329
        expected_max = (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * math.e))
        )
        benchmark = max(0.0, float(standard_error * expected_max))
    return _sharpe_probability(values, observed, benchmark)


def circular_block_sample(
    values: Sequence[float],
    block_size: int,
    seed: int,
) -> np.ndarray:
    """Return one circular block-bootstrap sample with the original length."""

    array = np.asarray(values)
    if array.ndim == 0 or len(array) == 0:
        raise ValueError("values must be nonempty")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(array), size=math.ceil(len(array) / block_size))
    indices = np.concatenate(
        [(start + np.arange(block_size)) % len(array) for start in starts]
    )[: len(array)]
    return array[indices]
