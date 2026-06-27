"""Contract-order generation, NBBO fill simulation, and explicit costs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping, Optional
import math

import numpy as np
import pandas as pd

from .schemas import BUY, SELL, Fill, OptionContract, OptionOrder, Position, QuoteSnapshot, normalize_side


@dataclass(frozen=True)
class FeeSchedule:
    broker_per_contract: float = 0.65
    occ_per_contract: float = 0.02
    exchange_per_contract: float = 0.05
    regulatory_per_contract: float = 0.002

    def option_fees(self, contracts: int) -> float:
        return float(abs(int(contracts)) * (self.broker_per_contract + self.occ_per_contract + self.exchange_per_contract + self.regulatory_per_contract))


@dataclass(frozen=True)
class OrderPolicy:
    max_spread_bps: float = 500.0
    max_quote_age: timedelta = timedelta(seconds=5)
    max_participation_of_displayed_size: float = 0.25
    max_adverse_price_drift_bps: float = 100.0
    time_in_force: str = "DAY"
    routing_policy: str = "SMART_LIMIT"
    min_tick_under_3: float = 0.01
    min_tick_over_3: float = 0.05
    allow_market_orders: bool = False


def option_tick(price: float, policy: OrderPolicy = OrderPolicy()) -> float:
    return policy.min_tick_under_3 if price < 3.0 else policy.min_tick_over_3


def round_limit_price(price: float, side: str, policy: OrderPolicy = OrderPolicy()) -> float:
    tick = option_tick(price, policy)
    scaled = price / tick
    rounded = math.ceil(scaled) * tick if normalize_side(side) == BUY else math.floor(scaled) * tick
    return float(max(tick, round(rounded, 4)))


def default_limit_price(quote: QuoteSnapshot, side: str, policy: OrderPolicy = OrderPolicy()) -> float:
    side = normalize_side(side)
    raw = quote.ask if side == BUY else quote.bid
    return round_limit_price(raw, side, policy)


def target_weights_to_orders(
    target_weights: pd.Series,
    current_positions: Mapping[str, Position],
    contracts: Mapping[str, OptionContract],
    quotes: Mapping[str, QuoteSnapshot],
    *,
    nav: float,
    decision_time,
    policy: OrderPolicy = OrderPolicy(),
    min_contracts: int = 1,
    reason_code: str = "rebalance",
) -> list[OptionOrder]:
    """Convert NAV weights into executable contract orders.

    The conversion uses executable bid/ask prices, never midpoint prices.  If the quote is
    not executable at the decision time, no order is emitted for that symbol.
    """

    orders: list[OptionOrder] = []
    for symbol, target_weight in target_weights.items():
        if symbol not in contracts or symbol not in quotes:
            continue
        contract = contracts[symbol]
        quote = quotes[symbol]
        ok, _ = quote.executable_at(decision_time, max_age=policy.max_quote_age, max_spread_bps=policy.max_spread_bps)
        if not ok:
            continue
        current_qty = int(current_positions.get(symbol, Position(symbol, 0)).quantity)
        px = quote.ask if float(target_weight) >= 0 else quote.bid
        target_qty = int(round(float(target_weight) * nav / (px * contract.multiplier)))
        delta_qty = target_qty - current_qty
        if abs(delta_qty) < min_contracts:
            continue
        side = BUY if delta_qty > 0 else SELL
        orders.append(
            OptionOrder(
                order_id=f"{pd.Timestamp(decision_time).strftime('%Y%m%d%H%M%S')}_{symbol}_{side}",
                decision_time=decision_time,
                symbol=symbol,
                side=side,
                contracts=abs(delta_qty),
                limit_price=default_limit_price(quote, side, policy),
                time_in_force=policy.time_in_force,
                routing_policy=policy.routing_policy,
                reason_code=reason_code,
                allow_market=False,
            )
        )
    return orders


def estimate_nbbo_fill(
    order: OptionOrder,
    quote: QuoteSnapshot,
    *,
    fee_schedule: FeeSchedule = FeeSchedule(),
    policy: OrderPolicy = OrderPolicy(),
) -> tuple[Optional[Fill], int, list[str]]:
    """Deterministic executable NBBO fill estimate.

    Crossing limit orders fill against displayed size up to the participation cap.  Passive
    midpoint assumptions intentionally produce no fill.
    """

    ok, reasons = quote.executable_at(order.decision_time, max_age=policy.max_quote_age, max_spread_bps=policy.max_spread_bps)
    if not ok:
        return None, order.contracts, reasons
    side = normalize_side(order.side)
    displayed = quote.ask_size if side == BUY else quote.bid_size
    executable_price = quote.ask if side == BUY else quote.bid
    crosses = order.limit_price >= quote.ask if side == BUY else order.limit_price <= quote.bid
    if not crosses:
        return None, order.contracts, ["passive_limit_not_filled"]
    max_fill = int(math.floor(displayed * policy.max_participation_of_displayed_size))
    filled = min(order.contracts, max(max_fill, 0))
    if filled <= 0:
        return None, order.contracts, ["insufficient_displayed_size"]
    fill = Fill(
        order_id=order.order_id,
        symbol=order.symbol,
        side=side,
        contracts=filled,
        price=float(executable_price),
        timestamp=quote.ts_recv,
        fees=fee_schedule.option_fees(filled),
        fill_model="nbbo_displayed_size_cross",
        displayed_size_used=float(filled),
    )
    return fill, order.contracts - filled, []


def build_execution_ledger(orders: list[OptionOrder]) -> pd.DataFrame:
    return pd.DataFrame([o.ledger_row() for o in orders])


def build_fill_ledger(fills: list[Fill]) -> pd.DataFrame:
    return pd.DataFrame([f.ledger_row() for f in fills])
