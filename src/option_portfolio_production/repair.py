"""Controlled order replacement for paper-broker production certification.

This module is certification machinery for the deterministic paper broker.  It is
not a live-trading order manager and deliberately reuses broker preview plus the
hard pre-trade risk gate before any replacement order can be submitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .broker import BrokerAdapter
from .execution import FeeSchedule, OrderPolicy, estimate_nbbo_fill, round_limit_price
from .risk import RiskGateConfig, evaluate_pre_trade_gate
from .schemas import BUY, AccountState, Fill, MarginEstimate, OptionContract, OptionOrder, QuoteSnapshot, normalize_side, utc_ts


REPAIR_LEDGER_COLUMNS = (
    "original_order_id",
    "replacement_order_id",
    "symbol",
    "side",
    "repair_reason",
    "decision_mark",
    "original_limit_price",
    "replacement_limit_price",
    "effective_fill_price",
    "adverse_drift_bps",
    "fill_fraction",
    "filled_contracts",
    "unfilled_contracts",
    "action",
    "timestamp",
)

REPAIR_ACTIONS = frozenset(
    {
        "abandoned_not_repairable",
        "abandoned_adverse_drift",
        "abandoned_spread_too_wide",
        "abandoned_broker_preview",
        "abandoned_risk_gate",
        "abandoned_broker_reject",
        "abandoned_sliver_fill",
        "abandoned_no_fill",
        "replaced_filled",
        "replaced_partial",
    }
)

NON_REPAIRABLE_REASONS = frozenset(
    {
        "kill_switch_enabled",
        "market_orders_disabled",
        "missing_broker_quote",
        "non_positive_nbbo",
        "locked_or_crossed_quote",
        "no_displayed_size",
    }
)


@dataclass(frozen=True)
class RepairPolicy:
    max_adverse_drift_bps: float = 100.0
    max_spread_bps: float = 500.0
    min_fill_fraction: float = 0.10
    allow_partial: bool = True


@dataclass(frozen=True)
class RepairEvent:
    original_order_id: str
    replacement_order_id: str
    symbol: str
    side: str
    repair_reason: str
    decision_mark: float
    original_limit_price: float
    replacement_limit_price: Optional[float]
    effective_fill_price: Optional[float]
    adverse_drift_bps: float
    fill_fraction: float
    filled_contracts: int
    unfilled_contracts: int
    action: str
    timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", normalize_side(self.side))
        object.__setattr__(self, "timestamp", utc_ts(self.timestamp))

    def ledger_row(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in REPAIR_LEDGER_COLUMNS}


def adverse_drift_bps(side: str, decision_mark: float, touch_price: float) -> float:
    """Return adverse drift in bps, clipped at zero for favorable touch moves."""

    mark = float(decision_mark)
    touch = float(touch_price)
    if not np.isfinite(mark) or not np.isfinite(touch) or mark <= 0:
        return float("inf")
    normalized_side = normalize_side(side)
    raw_drift = (touch - mark) / mark * 10_000.0 if normalized_side == BUY else (mark - touch) / mark * 10_000.0
    return float(max(raw_drift, 0.0))


def propose_repair_order(
    order: OptionOrder,
    quote: QuoteSnapshot,
    *,
    order_policy: OrderPolicy = OrderPolicy(),
    contracts: int | None = None,
) -> OptionOrder:
    """Create a touch-crossing limit replacement for paper-broker certification."""

    side = normalize_side(order.side)
    touch_price = quote.ask if side == BUY else quote.bid
    repair_contracts = order.contracts if contracts is None else int(contracts)
    return OptionOrder(
        order_id=f"{order.order_id}__repair",
        decision_time=order.decision_time,
        symbol=order.symbol,
        side=side,
        contracts=repair_contracts,
        limit_price=round_limit_price(touch_price, side, order_policy),
        time_in_force=order.time_in_force,
        routing_policy=order.routing_policy,
        reason_code="repair_touch_replace",
        allow_market=False,
    )


def attempt_order_repair(
    order: OptionOrder,
    quote: QuoteSnapshot,
    decision_mark: float,
    broker: BrokerAdapter,
    contract: OptionContract,
    account: AccountState,
    margin: MarginEstimate,
    *,
    unfilled_contracts: int,
    rejection_reasons: list[str] | tuple[str, ...],
    risk_config: RiskGateConfig = RiskGateConfig(),
    order_policy: OrderPolicy = OrderPolicy(),
    repair_policy: RepairPolicy = RepairPolicy(),
    fee_schedule: FeeSchedule = FeeSchedule(),
    gate_inputs: Mapping | None = None,
) -> tuple[Optional[Fill], RepairEvent]:
    """Attempt one controlled repair through the paper broker and risk gate.

    The function fails closed on every rejection and does not chase residual
    unfilled contracts after a successful partial replacement fill.  The
    expected replacement fill is sized against the quote BEFORE submission so
    a sliver abandonment never leaves contracts sitting in broker state; with
    ``allow_partial=False`` the replacement must be expected to fill in full.
    """

    side = normalize_side(order.side)
    reasons = tuple(str(reason) for reason in rejection_reasons)
    not_repairable = sorted(set(reasons) & NON_REPAIRABLE_REASONS)
    if not_repairable:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            drift_bps=0.0,
            repair_reason=_join_reasons(not_repairable),
            action="abandoned_not_repairable",
            unfilled_contracts=unfilled_contracts,
        )

    touch_price = quote.ask if side == BUY else quote.bid
    drift_bps = adverse_drift_bps(side, decision_mark, touch_price)
    if drift_bps > repair_policy.max_adverse_drift_bps:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            drift_bps=drift_bps,
            repair_reason="adverse_drift_exceeded",
            action="abandoned_adverse_drift",
            unfilled_contracts=unfilled_contracts,
        )

    if quote.spread_bps > repair_policy.max_spread_bps:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            drift_bps=drift_bps,
            repair_reason="spread_too_wide",
            action="abandoned_spread_too_wide",
            unfilled_contracts=unfilled_contracts,
        )

    repair_contracts = _repair_contracts(order, int(unfilled_contracts))
    replacement = propose_repair_order(order, quote, order_policy=order_policy, contracts=repair_contracts)

    preview = broker.preview_order(replacement)
    if not preview.passed:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            replacement=replacement,
            drift_bps=drift_bps,
            repair_reason=_join_reasons(preview.reasons),
            action="abandoned_broker_preview",
            unfilled_contracts=repair_contracts,
        )

    gate = evaluate_pre_trade_gate(
        replacement,
        contract,
        quote,
        account,
        margin,
        config=risk_config,
        **dict(gate_inputs or {}),
    )
    if not gate.passed:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            replacement=replacement,
            drift_bps=drift_bps,
            repair_reason=_join_reasons(gate.reasons),
            action="abandoned_risk_gate",
            unfilled_contracts=repair_contracts,
        )

    # Size the expected replacement fill against the quote BEFORE touching
    # broker state: an abandoned repair must never leave contracts behind.
    expected_fill, _, _ = estimate_nbbo_fill(replacement, quote, fee_schedule=fee_schedule, policy=order_policy)
    expected_contracts = int(expected_fill.contracts) if expected_fill is not None else 0
    expected_fraction = expected_contracts / replacement.contracts if replacement.contracts else 0.0
    min_required_fraction = repair_policy.min_fill_fraction if repair_policy.allow_partial else 1.0
    if expected_fraction < min_required_fraction:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            replacement=replacement,
            drift_bps=drift_bps,
            repair_reason="expected_fill_below_min_fraction",
            action="abandoned_sliver_fill",
            unfilled_contracts=repair_contracts,
        )

    try:
        broker.replace_order(order.order_id, replacement)
    except Exception as exc:
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            replacement=replacement,
            drift_bps=drift_bps,
            repair_reason=str(exc),
            action="abandoned_broker_reject",
            unfilled_contracts=repair_contracts,
        )

    matching_fills = [fill for fill in broker.get_fills() if fill.order_id == replacement.order_id]
    filled_contracts = int(sum(fill.contracts for fill in matching_fills))
    fill_fraction = float(filled_contracts / replacement.contracts) if replacement.contracts else 0.0
    effective_price = _effective_fill_price(matching_fills)
    remaining = max(int(replacement.contracts - filled_contracts), 0)

    if filled_contracts <= 0:
        # Nothing executed at the broker; cancel the resting replacement and
        # report honestly instead of pretending a sliver fill happened.
        cancel_order = getattr(broker, "cancel_order", None)
        if callable(cancel_order):
            cancel_order(replacement.order_id)
        return None, _repair_event(
            order,
            quote,
            decision_mark,
            replacement=replacement,
            drift_bps=drift_bps,
            repair_reason="no_fill_after_replace",
            action="abandoned_no_fill",
            unfilled_contracts=replacement.contracts,
        )

    # Contracts are actually held from here on, so the event must report a
    # replacement even if the realized fraction drifted below the pre-check
    # threshold (possible when the broker's book differs from the quote).
    repair_reason = _join_reasons(reasons) or "repair_touch_replace"
    if fill_fraction < min_required_fraction:
        repair_reason = "below_min_fill_fraction_post_submit"
    action = "replaced_filled" if remaining == 0 else "replaced_partial"
    return (_aggregate_fill(matching_fills) if matching_fills else None), _repair_event(
        order,
        quote,
        decision_mark,
        replacement=replacement,
        drift_bps=drift_bps,
        repair_reason=repair_reason,
        action=action,
        effective_fill_price=effective_price,
        fill_fraction=fill_fraction,
        filled_contracts=filled_contracts,
        unfilled_contracts=remaining,
    )


def build_repair_ledger(events: list[RepairEvent]) -> pd.DataFrame:
    return pd.DataFrame([event.ledger_row() for event in events], columns=list(REPAIR_LEDGER_COLUMNS))


def _repair_contracts(order: OptionOrder, unfilled_contracts: int) -> int:
    """Never re-order contracts the original order already filled."""

    if 0 < unfilled_contracts <= order.contracts:
        return int(unfilled_contracts)
    return int(order.contracts)


def _aggregate_fill(fills: list[Fill]) -> Fill:
    """Collapse multi-part replacement fills into one contract-weighted Fill."""

    if len(fills) == 1:
        return fills[0]
    first = fills[0]
    total_contracts = int(sum(fill.contracts for fill in fills))
    price = _effective_fill_price(fills)
    return Fill(
        order_id=first.order_id,
        symbol=first.symbol,
        side=first.side,
        contracts=total_contracts,
        price=float(price if price is not None else first.price),
        timestamp=max(fill.timestamp for fill in fills),
        fees=float(sum(fill.fees for fill in fills)),
        fill_model=first.fill_model,
        displayed_size_used=float(sum(fill.displayed_size_used for fill in fills)),
    )


def _join_reasons(reasons: tuple[str, ...] | list[str]) -> str:
    return ";".join(str(reason) for reason in reasons if str(reason))


def _effective_fill_price(fills: list[Fill]) -> Optional[float]:
    total_contracts = sum(fill.contracts for fill in fills)
    if total_contracts <= 0:
        return None
    notional = sum(fill.price * fill.contracts for fill in fills)
    return float(notional / total_contracts)


def _repair_event(
    order: OptionOrder,
    quote: QuoteSnapshot,
    decision_mark: float,
    *,
    drift_bps: float,
    repair_reason: str,
    action: str,
    replacement: OptionOrder | None = None,
    effective_fill_price: Optional[float] = None,
    fill_fraction: float = 0.0,
    filled_contracts: int = 0,
    unfilled_contracts: int | None = None,
) -> RepairEvent:
    remaining = order.contracts if unfilled_contracts is None else int(unfilled_contracts)
    return RepairEvent(
        original_order_id=order.order_id,
        replacement_order_id=replacement.order_id if replacement is not None else "",
        symbol=order.symbol,
        side=normalize_side(order.side),
        repair_reason=repair_reason,
        decision_mark=float(decision_mark),
        original_limit_price=float(order.limit_price),
        replacement_limit_price=float(replacement.limit_price) if replacement is not None else None,
        effective_fill_price=effective_fill_price,
        adverse_drift_bps=float(drift_bps),
        fill_fraction=float(fill_fraction),
        filled_contracts=int(filled_contracts),
        unfilled_contracts=remaining,
        action=action,
        timestamp=quote.ts_recv,
    )
