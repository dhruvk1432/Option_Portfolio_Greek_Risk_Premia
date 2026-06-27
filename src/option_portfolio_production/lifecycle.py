"""American option lifecycle, assignment, and expiry utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .schemas import CALL, PUT, OptionContract, Position, normalize_right


@dataclass(frozen=True)
class AssignmentEvent:
    symbol: str
    event_time: pd.Timestamp
    stock_symbol: str
    stock_quantity: int
    cash_flow: float
    reason: str

    def ledger_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_time": self.event_time,
            "stock_symbol": self.stock_symbol,
            "stock_quantity": self.stock_quantity,
            "cash_flow": self.cash_flow,
            "reason": self.reason,
        }


def intrinsic_value(contract: OptionContract, underlying_price: float) -> float:
    if contract.right == CALL:
        return max(float(underlying_price) - contract.strike, 0.0)
    return max(contract.strike - float(underlying_price), 0.0)


def expiry_payoff(contract: OptionContract, settlement_price: float, quantity: int) -> float:
    return float(quantity) * contract.multiplier * intrinsic_value(contract, settlement_price)


def extrinsic_value(contract: OptionContract, option_mark: float, underlying_price: float) -> float:
    return max(float(option_mark) - intrinsic_value(contract, underlying_price), 0.0)


def early_exercise_risk(
    contract: OptionContract,
    *,
    option_mark: float,
    underlying_price: float,
    dividend_amount: float = 0.0,
    days_to_ex_dividend: Optional[int] = None,
    hard_to_borrow: bool = False,
    extrinsic_threshold: float = 0.05,
) -> tuple[bool, list[str]]:
    """Conservative American exercise/assignment trigger.

    For calls, low extrinsic value before an ex-dividend date is a standard early exercise
    risk.  For puts, deep ITM/low extrinsic options are flagged because short puts can be
    assigned into stock, particularly around funding or borrow stress.
    """

    reasons: list[str] = []
    ext = extrinsic_value(contract, option_mark, underlying_price)
    moneyness = intrinsic_value(contract, underlying_price)
    if contract.right == CALL and dividend_amount > ext and days_to_ex_dividend is not None and 0 <= days_to_ex_dividend <= 3:
        reasons.append("call_dividend_exercise_risk")
    if contract.right == PUT and moneyness > 0 and ext <= extrinsic_threshold:
        reasons.append("deep_itm_put_assignment_risk")
    if hard_to_borrow and contract.right == CALL and moneyness > 0 and ext <= extrinsic_threshold:
        reasons.append("hard_to_borrow_call_assignment_risk")
    return bool(reasons), reasons


def assign_short_option(contract: OptionContract, short_contracts: int, event_time, *, reason: str = "assignment") -> AssignmentEvent:
    if short_contracts >= 0:
        raise ValueError("short_contracts must be negative for assignment")
    qty = abs(int(short_contracts)) * contract.multiplier
    if contract.right == CALL:
        stock_quantity = -qty
        cash_flow = qty * contract.strike
    else:
        stock_quantity = qty
        cash_flow = -qty * contract.strike
    return AssignmentEvent(
        symbol=contract.symbol,
        event_time=pd.Timestamp(event_time),
        stock_symbol=contract.underlying,
        stock_quantity=stock_quantity,
        cash_flow=float(cash_flow),
        reason=reason,
    )
