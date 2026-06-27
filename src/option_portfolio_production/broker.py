"""Broker-neutral execution interface and deterministic paper adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from .execution import estimate_nbbo_fill, FeeSchedule, OrderPolicy
from .schemas import AccountState, Fill, GateResult, OptionOrder, Position, QuoteSnapshot


class BrokerAdapter(Protocol):
    def get_account_state(self) -> AccountState: ...
    def get_positions(self) -> dict[str, Position]: ...
    def preview_order(self, order: OptionOrder) -> GateResult: ...
    def submit_order(self, order: OptionOrder) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def replace_order(self, order_id: str, replacement: OptionOrder) -> str: ...
    def get_fills(self) -> list[Fill]: ...
    def get_margin_preview(self, order: OptionOrder) -> float: ...
    def reconcile_positions(self, expected: dict[str, Position]) -> GateResult: ...


@dataclass
class PaperBrokerAdapter:
    """In-memory paper broker for certification tests.

    This adapter is intentionally simple and safe: live trading is not implemented, market
    orders are refused by default, and a kill switch blocks submissions immediately.
    """

    account: AccountState
    quotes: dict[str, QuoteSnapshot] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    policy: OrderPolicy = field(default_factory=OrderPolicy)
    fee_schedule: FeeSchedule = field(default_factory=FeeSchedule)
    kill_switch: bool = False
    submitted_orders: dict[str, OptionOrder] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def get_account_state(self) -> AccountState:
        return self.account

    def get_positions(self) -> dict[str, Position]:
        return dict(self.positions)

    def preview_order(self, order: OptionOrder) -> GateResult:
        if self.kill_switch:
            return GateResult.fail(["kill_switch_enabled"])
        if order.allow_market and not self.policy.allow_market_orders:
            return GateResult.fail(["market_orders_disabled"])
        if order.symbol not in self.quotes:
            return GateResult.fail(["missing_broker_quote"])
        ok, reasons = self.quotes[order.symbol].executable_at(order.decision_time, max_age=self.policy.max_quote_age, max_spread_bps=self.policy.max_spread_bps)
        return GateResult.pass_() if ok else GateResult.fail(reasons)

    def submit_order(self, order: OptionOrder) -> str:
        preview = self.preview_order(order)
        if not preview.passed:
            raise RuntimeError("order rejected: " + ";".join(preview.reasons))
        self.submitted_orders[order.order_id] = order
        fill, _, _ = estimate_nbbo_fill(order, self.quotes[order.symbol], fee_schedule=self.fee_schedule, policy=self.policy)
        if fill is not None:
            self.fills.append(fill)
            old = self.positions.get(fill.symbol, Position(fill.symbol, 0))
            self.positions[fill.symbol] = Position(fill.symbol, old.quantity + fill.signed_quantity, fill.price)
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        return self.submitted_orders.pop(order_id, None) is not None

    def replace_order(self, order_id: str, replacement: OptionOrder) -> str:
        self.cancel_order(order_id)
        return self.submit_order(replacement)

    def get_fills(self) -> list[Fill]:
        return list(self.fills)

    def get_margin_preview(self, order: OptionOrder) -> float:
        # Paper placeholder: real IBKR adapter must replace this with broker preview.
        return 0.0

    def reconcile_positions(self, expected: dict[str, Position]) -> GateResult:
        reasons = []
        keys = set(expected) | set(self.positions)
        for key in sorted(keys):
            if expected.get(key, Position(key, 0)).quantity != self.positions.get(key, Position(key, 0)).quantity:
                reasons.append(f"position_mismatch:{key}")
        return GateResult.pass_() if not reasons else GateResult.fail(reasons)
