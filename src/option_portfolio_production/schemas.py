"""Shared production schemas for the option-only portfolio stack.

The research paper works with NAV-normalized option weights.  A production
system must eventually convert those weights into timestamped quotes, orders,
fills, margin previews, assignment events, and reconciliation ledgers.  These
schemas are intentionally broker-neutral so the first concrete adapter can be
IBKR paper without leaking IBKR-specific fields into the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

BUY = "buy"
SELL = "sell"
CALL = "call"
PUT = "put"


def normalize_side(side: Any) -> str:
    key = str(side).strip().lower()
    if key in {"buy", "b", "bid", "+1", "1"}:
        return BUY
    if key in {"sell", "s", "ask", "offer", "-1"}:
        return SELL
    raise ValueError(f"unknown side {side!r}")


def normalize_right(right: Any) -> str:
    key = str(right).strip().lower()
    if key in {"c", "call"}:
        return CALL
    if key in {"p", "put"}:
        return PUT
    raise ValueError(f"unknown option right {right!r}")


def utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    expiry: pd.Timestamp
    right: str
    strike: float
    multiplier: int = 100
    asset_class: str = "equity_option"
    broker_contract_id: str = ""
    adjusted_deliverable: str = "standard"

    def __post_init__(self) -> None:
        object.__setattr__(self, "right", normalize_right(self.right))
        object.__setattr__(self, "expiry", pd.Timestamp(self.expiry).normalize())
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")

    @property
    def is_vix(self) -> bool:
        return self.underlying.upper() in {"VIX", "VX", "VX_FRONT"} or self.asset_class.lower() == "vix_option"


@dataclass(frozen=True)
class QuoteSnapshot:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    ts_event: pd.Timestamp
    ts_recv: pd.Timestamp
    local_receive_ts: pd.Timestamp
    vendor: str = "unknown"
    schema: str = "unknown"
    exchange_ts: Optional[pd.Timestamp] = None
    sequence: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts_event", utc_ts(self.ts_event))
        object.__setattr__(self, "ts_recv", utc_ts(self.ts_recv))
        object.__setattr__(self, "local_receive_ts", utc_ts(self.local_receive_ts))
        if self.exchange_ts is not None:
            object.__setattr__(self, "exchange_ts", utc_ts(self.exchange_ts))
        for name in ["bid", "ask", "bid_size", "ask_size"]:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return np.inf if mid <= 0 else self.spread / mid * 10_000.0

    @property
    def is_crossed_or_locked(self) -> bool:
        return self.bid >= self.ask

    def executable_at(
        self,
        decision_time: Any,
        *,
        max_age: timedelta = timedelta(seconds=5),
        max_spread_bps: float = 500.0,
    ) -> tuple[bool, list[str]]:
        dt = utc_ts(decision_time)
        reasons: list[str] = []
        if self.ts_event > dt or self.ts_recv > dt or self.local_receive_ts > dt:
            reasons.append("quote_after_decision_time")
        if dt - self.ts_event > max_age:
            reasons.append("stale_quote")
        if self.bid <= 0 or self.ask <= 0:
            reasons.append("non_positive_nbbo")
        if self.bid_size <= 0 or self.ask_size <= 0:
            reasons.append("no_displayed_size")
        if self.is_crossed_or_locked:
            reasons.append("locked_or_crossed_quote")
        if self.spread_bps > max_spread_bps:
            reasons.append("spread_too_wide")
        return not reasons, reasons

    def ledger_row(self, *, symbol_map_version: str = "unknown") -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "schema": self.schema,
            "symbol": self.symbol,
            "ts_event": self.ts_event,
            "ts_recv": self.ts_recv,
            "exchange_timestamp": self.exchange_ts,
            "local_receive_timestamp": self.local_receive_ts,
            "symbol_mapping_version": symbol_map_version,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    avg_price: float = 0.0
    asset_class: str = "option"


@dataclass(frozen=True)
class AccountState:
    net_liquidation: float
    cash: float
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_ts(self.timestamp))
        if self.net_liquidation <= 0:
            raise ValueError("net_liquidation must be positive")


@dataclass(frozen=True)
class OptionOrder:
    order_id: str
    decision_time: pd.Timestamp
    symbol: str
    side: str
    contracts: int
    limit_price: float
    time_in_force: str = "DAY"
    routing_policy: str = "SMART_LIMIT"
    reason_code: str = "rebalance"
    allow_market: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", utc_ts(self.decision_time))
        object.__setattr__(self, "side", normalize_side(self.side))
        if self.contracts <= 0:
            raise ValueError("contracts must be positive")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be positive")

    def ledger_row(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "decision_time": self.decision_time,
            "symbol": self.symbol,
            "side": self.side,
            "contracts": self.contracts,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force,
            "routing_policy": self.routing_policy,
            "reason_code": self.reason_code,
            "allow_market": self.allow_market,
        }


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: str
    contracts: int
    price: float
    timestamp: pd.Timestamp
    fees: float = 0.0
    fill_model: str = "nbbo_cross"
    displayed_size_used: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", utc_ts(self.timestamp))
        object.__setattr__(self, "side", normalize_side(self.side))
        if self.contracts <= 0:
            raise ValueError("contracts must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")

    @property
    def signed_quantity(self) -> int:
        return self.contracts if self.side == BUY else -self.contracts

    def ledger_row(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "contracts": self.contracts,
            "price": self.price,
            "timestamp": self.timestamp,
            "fees": self.fees,
            "fill_model": self.fill_model,
            "displayed_size_used": self.displayed_size_used,
        }


@dataclass(frozen=True)
class MarginEstimate:
    symbol: str
    margin_requirement: float
    stress_loss: float
    assignment_notional: float
    margin_source: str = "internal_stress"
    preview_status: str = "pass"

    def ledger_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "margin_requirement": self.margin_requirement,
            "stress_loss": self.stress_loss,
            "assignment_notional": self.assignment_notional,
            "margin_source": self.margin_source,
            "margin_preview_status": self.preview_status,
        }


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    severity: str = "pass"

    @classmethod
    def pass_(cls) -> "GateResult":
        return cls(True, (), "pass")

    @classmethod
    def fail(cls, reasons: Iterable[str], *, severity: str = "critical") -> "GateResult":
        return cls(False, tuple(str(r) for r in reasons), severity)

    def ledger_row(self, *, order_id: str = "", symbol: str = "") -> dict[str, Any]:
        return {
            "order_id": order_id,
            "symbol": symbol,
            "passed": self.passed,
            "severity": self.severity,
            "reasons": ";".join(self.reasons),
        }
