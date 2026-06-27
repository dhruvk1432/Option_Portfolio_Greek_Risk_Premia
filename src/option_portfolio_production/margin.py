"""Conservative internal option margin and stress estimates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .lifecycle import intrinsic_value
from .schemas import BUY, SELL, AccountState, MarginEstimate, OptionContract, OptionOrder, Position, QuoteSnapshot, normalize_side


@dataclass(frozen=True)
class MarginConfig:
    short_option_underlying_pct: float = 0.20
    short_option_min_underlying_pct: float = 0.10
    stress_move_pct: float = 0.30
    vix_stress_move_abs: float = 15.0
    min_margin_per_short_contract: float = 100.0


def _option_price_for_order(order: OptionOrder, quote: QuoteSnapshot) -> float:
    return quote.ask if normalize_side(order.side) == BUY else quote.bid


def conservative_order_margin(
    order: OptionOrder,
    contract: OptionContract,
    quote: QuoteSnapshot,
    underlying_price: float,
    *,
    config: MarginConfig = MarginConfig(),
    broker_margin_preview: float | None = None,
) -> MarginEstimate:
    price = _option_price_for_order(order, quote)
    qty = order.contracts
    premium = price * contract.multiplier * qty
    if normalize_side(order.side) == BUY:
        requirement = premium
        assignment_notional = 0.0
    else:
        otm = max(contract.strike - underlying_price, 0.0) if contract.right == "call" else max(underlying_price - contract.strike, 0.0)
        base = max(
            config.short_option_underlying_pct * underlying_price - otm + price,
            config.short_option_min_underlying_pct * underlying_price + price,
            config.min_margin_per_short_contract / contract.multiplier,
        )
        requirement = base * contract.multiplier * qty
        assignment_notional = contract.strike * contract.multiplier * qty
    if broker_margin_preview is not None and np.isfinite(broker_margin_preview):
        requirement = max(requirement, float(broker_margin_preview))
        source = "max_internal_broker_preview"
    else:
        source = "internal_stress"
    stress_loss = stress_loss_for_order(order, contract, quote, underlying_price, config=config)
    return MarginEstimate(
        symbol=order.symbol,
        margin_requirement=float(requirement),
        stress_loss=float(stress_loss),
        assignment_notional=float(assignment_notional),
        margin_source=source,
        preview_status="pass" if np.isfinite(requirement) and requirement >= 0 else "fail",
    )


def stress_loss_for_order(
    order: OptionOrder,
    contract: OptionContract,
    quote: QuoteSnapshot,
    underlying_price: float,
    *,
    config: MarginConfig = MarginConfig(),
) -> float:
    price = _option_price_for_order(order, quote)
    if contract.is_vix:
        shocked = underlying_price + config.vix_stress_move_abs
    else:
        shocked = underlying_price * (1.0 + config.stress_move_pct)
    stressed_intrinsic = intrinsic_value(contract, shocked)
    pnl_per_contract = (stressed_intrinsic - price) * contract.multiplier
    signed = 1.0 if normalize_side(order.side) == BUY else -1.0
    pnl = signed * pnl_per_contract * order.contracts
    return max(-pnl, 0.0)


def build_margin_ledger(estimates: list[MarginEstimate]) -> pd.DataFrame:
    return pd.DataFrame([e.ledger_row() for e in estimates])
