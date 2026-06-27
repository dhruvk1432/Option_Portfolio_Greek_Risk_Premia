"""Hard production pre-trade gates for option orders."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np

from .lifecycle import early_exercise_risk
from .schemas import BUY, SELL, AccountState, GateResult, MarginEstimate, OptionContract, OptionOrder, QuoteSnapshot, normalize_side


@dataclass(frozen=True)
class RiskGateConfig:
    max_margin_to_nav: float = 0.50
    max_assignment_notional_to_nav: float = 0.50
    max_single_name_stress_loss_to_nav: float = 0.10
    max_spread_bps: float = 500.0
    max_quote_age: timedelta = timedelta(seconds=5)
    min_open_interest: float = 50.0
    min_volume: float = 10.0
    min_fill_probability: float = 0.50
    block_hard_to_borrow_short_calls: bool = True
    block_unsupported_corporate_action_window: bool = True


def evaluate_pre_trade_gate(
    order: OptionOrder,
    contract: OptionContract,
    quote: QuoteSnapshot,
    account: AccountState,
    margin: MarginEstimate,
    *,
    open_interest: float = np.inf,
    volume: float = np.inf,
    fill_probability: float = 1.0,
    hard_to_borrow: bool = False,
    unsupported_corporate_action_window: bool = False,
    option_mark: float | None = None,
    underlying_price: float | None = None,
    dividend_amount: float = 0.0,
    days_to_ex_dividend: int | None = None,
    config: RiskGateConfig = RiskGateConfig(),
) -> GateResult:
    reasons: list[str] = []
    ok, quote_reasons = quote.executable_at(order.decision_time, max_age=config.max_quote_age, max_spread_bps=config.max_spread_bps)
    if not ok:
        reasons.extend(quote_reasons)
    if margin.preview_status != "pass" or not np.isfinite(margin.margin_requirement):
        reasons.append("missing_or_failed_margin_preview")
    if margin.margin_requirement / account.net_liquidation > config.max_margin_to_nav:
        reasons.append("margin_to_nav_exceeded")
    if margin.assignment_notional / account.net_liquidation > config.max_assignment_notional_to_nav:
        reasons.append("assignment_notional_exceeded")
    if margin.stress_loss / account.net_liquidation > config.max_single_name_stress_loss_to_nav:
        reasons.append("single_name_stress_loss_exceeded")
    if open_interest < config.min_open_interest:
        reasons.append("open_interest_too_low")
    if volume < config.min_volume:
        reasons.append("volume_too_low")
    if fill_probability < config.min_fill_probability:
        reasons.append("fill_probability_too_low")
    is_short_open = normalize_side(order.side) == SELL
    if is_short_open and hard_to_borrow and contract.right == "call" and config.block_hard_to_borrow_short_calls:
        reasons.append("hard_to_borrow_short_call_blocked")
    if is_short_open and unsupported_corporate_action_window and config.block_unsupported_corporate_action_window:
        reasons.append("unsupported_corporate_action_window")
    if is_short_open and option_mark is not None and underlying_price is not None:
        risk, er_reasons = early_exercise_risk(
            contract,
            option_mark=option_mark,
            underlying_price=underlying_price,
            dividend_amount=dividend_amount,
            days_to_ex_dividend=days_to_ex_dividend,
            hard_to_borrow=hard_to_borrow,
        )
        if risk:
            reasons.extend(er_reasons)
    if reasons:
        return GateResult.fail(reasons)
    return GateResult.pass_()
