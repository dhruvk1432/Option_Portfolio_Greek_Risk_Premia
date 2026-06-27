"""Production wrappers around research weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .execution import FeeSchedule
from .schemas import OptionContract, QuoteSnapshot


@dataclass(frozen=True)
class ProductionOptimizerConfig:
    financing_rate_annual: float = 0.05
    periods_per_year: float = 12.0
    expected_slippage_bps: float = 0.0
    shrink_post_cost_alpha: float = 0.0


def estimate_round_trip_cost_return(
    contract: OptionContract,
    quote: QuoteSnapshot,
    *,
    fee_schedule: FeeSchedule = FeeSchedule(),
    expected_slippage_bps: float = 0.0,
) -> float:
    mid = max(quote.mid, 1e-12)
    half_spread = 0.5 * max(quote.ask - quote.bid, 0.0)
    fees_per_contract = fee_schedule.option_fees(1) / contract.multiplier
    slippage = mid * expected_slippage_bps / 10_000.0
    return float((2.0 * (half_spread + slippage) + 2.0 * fees_per_contract) / mid)


def post_cost_expected_returns(
    expected_returns: pd.Series,
    contracts: Mapping[str, OptionContract],
    quotes: Mapping[str, QuoteSnapshot],
    *,
    config: ProductionOptimizerConfig = ProductionOptimizerConfig(),
    fee_schedule: FeeSchedule = FeeSchedule(),
) -> pd.Series:
    out = expected_returns.astype(float).copy()
    financing = config.financing_rate_annual / config.periods_per_year
    for symbol in out.index:
        if symbol in contracts and symbol in quotes:
            cost = estimate_round_trip_cost_return(
                contracts[symbol], quotes[symbol], fee_schedule=fee_schedule, expected_slippage_bps=config.expected_slippage_bps
            )
            out.loc[symbol] = out.loc[symbol] - cost - financing
        else:
            out.loc[symbol] = np.nan
    out = out.fillna(0.0)
    if config.shrink_post_cost_alpha > 0:
        out = (1.0 - config.shrink_post_cost_alpha) * out
    return out


def mask_inexecutable_weights(weights: pd.Series, executable_symbols: set[str]) -> pd.Series:
    out = weights.copy().astype(float)
    out.loc[~out.index.isin(executable_symbols)] = 0.0
    gross = out.abs().sum()
    if gross > 0:
        out = out / gross * weights.abs().sum()
    return out
