from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.option_portfolio_production import (
    AccountState,
    MarginEstimate,
    OptionContract,
    OptionOrder,
    OrderPolicy,
    PaperBrokerAdapter,
    Position,
    QuoteSnapshot,
    REPAIR_LEDGER_COLUMNS,
    RepairEvent,
    RepairPolicy,
    RiskGateConfig,
    assign_short_option,
    attach_exact_vix_settlement,
    build_execution_ledger,
    build_fill_ledger,
    build_margin_ledger,
    build_market_data_ledger,
    build_repair_ledger,
    conservative_order_margin,
    early_exercise_risk,
    estimate_nbbo_fill,
    evaluate_pre_trade_gate,
    normalize_vro_soq_frame,
    post_cost_expected_returns,
    reconcile_quote_pair,
    require_exact_vix_settlement,
    attempt_order_repair,
    target_weights_to_orders,
    validate_timestamp_monotonicity,
)
from src.option_portfolio_production.verification import ProductionVerifier


def quote(symbol="AAPL  260116C00100000", bid=1.04, ask=1.06, bid_size=100, ask_size=100):
    return QuoteSnapshot(
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        ts_event="2026-01-02T14:30:00Z",
        ts_recv="2026-01-02T14:30:00.050Z",
        local_receive_ts="2026-01-02T14:30:00.100Z",
        vendor="databento",
        schema="cbbo-1m",
    )


def contract(symbol="AAPL  260116C00100000", right="call", strike=100.0):
    return OptionContract(symbol=symbol, underlying="AAPL", expiry="2026-01-16", right=right, strike=strike)


def test_vro_soq_normalization_and_exact_join():
    raw = pd.DataFrame({"date": ["2026-01-14"], "VRO": [23.45]})
    settlements = normalize_vro_soq_frame(raw, source="cboe_vro.csv", source_url="https://www.cboe.com/")
    assert list(settlements.columns)[0] == "settlement_date"
    assert settlements.loc[0, "settlement_value"] == pytest.approx(23.45)

    detail = pd.DataFrame({
        "symbol": ["VIX   260114C00020000"],
        "asset_class": ["vix_option"],
        "expiry": ["2026-01-14"],
    })
    ledger = attach_exact_vix_settlement(detail, settlements)
    ok, reasons = require_exact_vix_settlement(ledger)
    assert ok, reasons
    assert ledger.loc[0, "production_settlement_source"] == "vro_soq_exact"


def test_missing_vro_settlement_fails_headline_requirement():
    detail = pd.DataFrame({
        "symbol": ["VIX   260114C00020000"],
        "asset_class": ["vix_option"],
        "expiry": ["2026-01-14"],
    })
    ledger = attach_exact_vix_settlement(detail, pd.DataFrame())
    ok, reasons = require_exact_vix_settlement(ledger)
    assert not ok
    assert "missing_exact_vro_soq" in reasons[0]


def test_market_data_ledger_and_quote_reconciliation():
    q1 = quote()
    q2 = quote(bid=1.01, ask=1.11)
    ledger = build_market_data_ledger([q1, q2], symbol_map_version="osi-v1")
    ok, reasons = validate_timestamp_monotonicity(ledger)
    assert ok, reasons
    rec = reconcile_quote_pair(q1, q2, max_mid_diff_bps=200.0)
    assert rec.passed
    assert rec.mid_diff_bps > 0


def test_order_generation_uses_nbbo_not_midpoint_and_partial_fill():
    q = quote(ask_size=8)
    c = contract()
    orders = target_weights_to_orders(
        pd.Series({c.symbol: 0.10}),
        {},
        {c.symbol: c},
        {c.symbol: q},
        nav=100_000,
        decision_time="2026-01-02T14:30:01Z",
    )
    assert len(orders) == 1
    assert orders[0].limit_price >= q.ask
    fill, unfilled, reasons = estimate_nbbo_fill(orders[0], q, policy=OrderPolicy(max_participation_of_displayed_size=0.25))
    assert fill is not None
    assert fill.fill_model == "nbbo_displayed_size_cross"
    assert fill.price == pytest.approx(q.ask)
    assert fill.contracts == 2
    assert unfilled == orders[0].contracts - 2


def test_short_option_hard_gate_blocks_margin_and_assignment_risk():
    c = contract(right="call", strike=100)
    q = quote(bid=0.05, ask=0.10)
    order = OptionOrder("o1", "2026-01-02T14:30:01Z", c.symbol, "sell", 10, q.bid)
    account = AccountState(net_liquidation=10_000, cash=10_000)
    margin = conservative_order_margin(order, c, q, underlying_price=120.0)
    result = evaluate_pre_trade_gate(
        order,
        c,
        q,
        account,
        margin,
        open_interest=1_000,
        volume=100,
        hard_to_borrow=True,
        option_mark=0.05,
        underlying_price=120.0,
        config=RiskGateConfig(max_margin_to_nav=0.05, max_assignment_notional_to_nav=0.05),
    )
    assert not result.passed
    assert "hard_to_borrow_short_call_blocked" in result.reasons
    assert "assignment_notional_exceeded" in result.reasons


def test_early_exercise_and_assignment_ledger():
    c = contract(right="call", strike=100)
    risk, reasons = early_exercise_risk(c, option_mark=5.05, underlying_price=105, dividend_amount=0.25, days_to_ex_dividend=1)
    assert risk
    assert "call_dividend_exercise_risk" in reasons
    event = assign_short_option(c, -2, "2026-01-15T21:00:00Z")
    row = event.ledger_row()
    assert row["stock_quantity"] == -200
    assert row["cash_flow"] == pytest.approx(20_000)


def test_post_cost_expected_returns_reduces_alpha():
    c = contract()
    q = quote(bid=1.0, ask=1.2)
    mu = post_cost_expected_returns(pd.Series({c.symbol: 0.10}), {c.symbol: c}, {c.symbol: q})
    assert mu.loc[c.symbol] < 0.10


def test_paper_broker_preview_submit_and_reconcile():
    q = quote(ask_size=100)
    order = OptionOrder("paper1", "2026-01-02T14:30:01Z", q.symbol, "buy", 5, q.ask)
    broker = PaperBrokerAdapter(AccountState(net_liquidation=100_000, cash=100_000), quotes={q.symbol: q})
    assert broker.preview_order(order).passed
    broker.submit_order(order)
    assert broker.get_fills()[0].contracts == 5
    expected = {q.symbol: Position(q.symbol, 5, q.ask)}
    assert broker.reconcile_positions(expected).passed


def test_repair_replaces_at_touch_within_drift_bound():
    q = quote(ask_size=100)
    c = contract()
    order = OptionOrder("repair1", "2026-01-02T14:30:01Z", q.symbol, "buy", 5, q.bid)
    original_fill, unfilled, reasons = estimate_nbbo_fill(order, q)
    assert original_fill is None
    assert reasons == ["passive_limit_not_filled"]

    account = AccountState(net_liquidation=100_000, cash=100_000)
    margin = conservative_order_margin(order, c, q, underlying_price=100.0)
    broker = PaperBrokerAdapter(account, quotes={q.symbol: q})
    fill, event = attempt_order_repair(
        order,
        q,
        q.ask,
        broker,
        c,
        account,
        margin,
        unfilled_contracts=unfilled,
        rejection_reasons=reasons,
    )

    assert fill is not None
    assert fill.order_id == "repair1__repair"
    assert fill.price == pytest.approx(q.ask)
    assert event.action == "replaced_filled"
    assert event.replacement_limit_price == pytest.approx(q.ask)
    assert event.effective_fill_price == pytest.approx(q.ask)
    assert event.fill_fraction == pytest.approx(1.0)
    assert event.filled_contracts == 5
    assert event.unfilled_contracts == 0

    ledger = build_repair_ledger([event])
    assert list(ledger.columns) == list(REPAIR_LEDGER_COLUMNS)
    row = ledger.iloc[0].to_dict()
    assert row["original_order_id"] == order.order_id
    assert row["replacement_order_id"] == "repair1__repair"
    assert row["repair_reason"] == "passive_limit_not_filled"
    assert row["adverse_drift_bps"] == pytest.approx(0.0)


def test_repair_abandons_on_adverse_drift():
    q = quote(bid=1.18, ask=1.20, ask_size=100)
    c = contract()
    order = OptionOrder("repair_drift", "2026-01-02T14:30:01Z", q.symbol, "buy", 5, 1.00)
    account = AccountState(net_liquidation=100_000, cash=100_000)
    margin = conservative_order_margin(order, c, q, underlying_price=100.0)
    broker = PaperBrokerAdapter(account, quotes={q.symbol: q})

    fill, event = attempt_order_repair(
        order,
        q,
        1.00,
        broker,
        c,
        account,
        margin,
        unfilled_contracts=5,
        rejection_reasons=["passive_limit_not_filled"],
        repair_policy=RepairPolicy(max_adverse_drift_bps=100.0),
    )

    assert fill is None
    assert event.action == "abandoned_adverse_drift"
    assert event.replacement_order_id == ""
    assert broker.submitted_orders == {}


def test_repair_never_bypasses_kill_switch_or_risk_gate():
    q = quote(ask_size=100)
    c = contract()
    order = OptionOrder("repair_gate", "2026-01-02T14:30:01Z", q.symbol, "buy", 5, q.bid)
    account = AccountState(net_liquidation=100_000, cash=100_000)
    margin = conservative_order_margin(order, c, q, underlying_price=100.0)

    kill_broker = PaperBrokerAdapter(account, quotes={q.symbol: q}, kill_switch=True)
    fill, event = attempt_order_repair(
        order,
        q,
        q.ask,
        kill_broker,
        c,
        account,
        margin,
        unfilled_contracts=5,
        rejection_reasons=["passive_limit_not_filled"],
    )
    assert fill is None
    assert event.action == "abandoned_broker_preview"
    assert event.repair_reason == "kill_switch_enabled"
    assert kill_broker.submitted_orders == {}

    risk_broker = PaperBrokerAdapter(account, quotes={q.symbol: q})
    bad_margin = MarginEstimate(
        symbol=q.symbol,
        margin_requirement=75_000.0,
        stress_loss=0.0,
        assignment_notional=0.0,
        preview_status="pass",
    )
    fill, event = attempt_order_repair(
        order,
        q,
        q.ask,
        risk_broker,
        c,
        account,
        bad_margin,
        unfilled_contracts=5,
        rejection_reasons=["passive_limit_not_filled"],
    )
    assert fill is None
    assert event.action == "abandoned_risk_gate"
    assert "margin_to_nav_exceeded" in event.repair_reason
    assert risk_broker.submitted_orders == {}


def test_repair_partial_fill_and_sliver_rejection():
    c = contract()
    account = AccountState(net_liquidation=100_000, cash=100_000)
    partial_quote = quote(ask_size=20)
    partial_order = OptionOrder("repair_partial", "2026-01-02T14:30:01Z", partial_quote.symbol, "buy", 10, partial_quote.bid)
    margin = conservative_order_margin(partial_order, c, partial_quote, underlying_price=100.0)
    broker = PaperBrokerAdapter(account, quotes={partial_quote.symbol: partial_quote})

    fill, event = attempt_order_repair(
        partial_order,
        partial_quote,
        partial_quote.ask,
        broker,
        c,
        account,
        margin,
        unfilled_contracts=10,
        rejection_reasons=["passive_limit_not_filled"],
    )
    assert fill is not None
    assert event.action == "replaced_partial"
    assert event.filled_contracts == 5
    assert event.unfilled_contracts == 5
    assert event.fill_fraction == pytest.approx(0.5)

    sliver_quote = quote(ask_size=3)
    sliver_order = OptionOrder("repair_sliver", "2026-01-02T14:30:01Z", sliver_quote.symbol, "buy", 10, sliver_quote.bid)
    sliver_margin = conservative_order_margin(sliver_order, c, sliver_quote, underlying_price=100.0)
    sliver_broker = PaperBrokerAdapter(account, quotes={sliver_quote.symbol: sliver_quote})
    fill, event = attempt_order_repair(
        sliver_order,
        sliver_quote,
        sliver_quote.ask,
        sliver_broker,
        c,
        account,
        sliver_margin,
        unfilled_contracts=10,
        rejection_reasons=["passive_limit_not_filled"],
    )
    assert fill is None
    assert event.action == "abandoned_sliver_fill"
    assert event.filled_contracts == 0
    assert event.fill_fraction == pytest.approx(0.0)
    assert "repair_sliver__repair" not in sliver_broker.submitted_orders


def test_repair_sliver_abandonment_never_touches_broker_state():
    # A repair whose expected fill sits below min_fill_fraction must abandon
    # BEFORE submission: no fills, no positions, no resting replacement.
    c = contract()
    account = AccountState(net_liquidation=100_000, cash=100_000)
    q = quote(ask_size=4)  # participation cap 0.25 -> expected fill 1 of 20 = 0.05 < 0.10
    order = OptionOrder("repair_tiny", "2026-01-02T14:30:01Z", q.symbol, "buy", 20, q.bid)
    margin = conservative_order_margin(order, c, q, underlying_price=100.0)
    broker = PaperBrokerAdapter(account, quotes={q.symbol: q})

    fill, event = attempt_order_repair(
        order,
        q,
        q.ask,
        broker,
        c,
        account,
        margin,
        unfilled_contracts=20,
        rejection_reasons=["passive_limit_not_filled"],
    )
    assert fill is None
    assert event.action == "abandoned_sliver_fill"
    assert event.repair_reason == "expected_fill_below_min_fraction"
    assert broker.get_fills() == []
    assert broker.get_positions() == {}
    assert "repair_tiny__repair" not in broker.submitted_orders


def test_repair_never_reorders_contracts_the_original_already_filled():
    # Original 10-lot already filled 6; the replacement must be sized to the
    # 4-contract remainder even when allow_partial=False (no over-ordering).
    c = contract()
    account = AccountState(net_liquidation=100_000, cash=100_000)
    q = quote(ask_size=100)
    order = OptionOrder("repair_remainder", "2026-01-02T14:30:01Z", q.symbol, "buy", 10, q.bid)
    margin = conservative_order_margin(order, c, q, underlying_price=100.0)
    broker = PaperBrokerAdapter(account, quotes={q.symbol: q})

    fill, event = attempt_order_repair(
        order,
        q,
        q.ask,
        broker,
        c,
        account,
        margin,
        unfilled_contracts=4,
        rejection_reasons=["passive_limit_not_filled"],
        repair_policy=RepairPolicy(allow_partial=False),
    )
    assert fill is not None
    assert event.action == "replaced_filled"
    assert event.filled_contracts == 4
    assert event.unfilled_contracts == 0
    assert broker.submitted_orders["repair_remainder__repair"].contracts == 4
    position = broker.get_positions()[q.symbol]
    assert position.quantity == 4


def test_repair_partial_disallowed_requires_expected_full_fill():
    # allow_partial=False turns the pre-submission check into a full-fill
    # requirement: an expected 5-of-10 fill abandons without touching broker state.
    c = contract()
    account = AccountState(net_liquidation=100_000, cash=100_000)
    q = quote(ask_size=20)  # expected fill 5 of 10
    order = OptionOrder("repair_strict", "2026-01-02T14:30:01Z", q.symbol, "buy", 10, q.bid)
    margin = conservative_order_margin(order, c, q, underlying_price=100.0)
    broker = PaperBrokerAdapter(account, quotes={q.symbol: q})

    fill, event = attempt_order_repair(
        order,
        q,
        q.ask,
        broker,
        c,
        account,
        margin,
        unfilled_contracts=10,
        rejection_reasons=["passive_limit_not_filled"],
        repair_policy=RepairPolicy(allow_partial=False),
    )
    assert fill is None
    assert event.action == "abandoned_sliver_fill"
    assert broker.get_fills() == []
    assert "repair_strict__repair" not in broker.submitted_orders


def _write_synthetic_production_ledgers(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "symbol": ["VIX   260114C00020000"],
        "asset_class": ["vix_option"],
        "expiry": ["2026-01-14"],
        "production_settlement_source": ["vro_soq_exact"],
    }).to_csv(out / "settlement_ledger.csv", index=False)
    build_execution_ledger([
        OptionOrder("o1", "2026-01-02T14:30:01Z", "AAPL  260116C00100000", "buy", 1, 1.10)
    ]).to_csv(out / "execution_ledger.csv", index=False)
    build_fill_ledger([
        estimate_nbbo_fill(OptionOrder("o1", "2026-01-02T14:30:01Z", quote().symbol, "buy", 1, 1.10), quote())[0]
    ]).to_csv(out / "fill_ledger.csv", index=False)
    build_margin_ledger([
        conservative_order_margin(OptionOrder("o1", "2026-01-02T14:30:01Z", contract().symbol, "buy", 1, 1.10), contract(), quote(), 100.0)
    ]).to_csv(out / "margin_ledger.csv", index=False)
    pd.DataFrame({"symbol": ["NONE"], "event_time": ["2026-01-02T21:00:00Z"], "stock_symbol": ["NONE"], "stock_quantity": [0], "cash_flow": [0.0], "reason": ["no_assignment"]}).to_csv(out / "assignment_ledger.csv", index=False)
    pd.DataFrame({"symbol": [quote().symbol], "passed": [True], "reasons": [""]}).to_csv(out / "data_reconciliation_ledger.csv", index=False)
    pd.DataFrame({"passed": [True], "reasons": [""]}).to_csv(out / "broker_position_reconciliation.csv", index=False)
    build_repair_ledger([
        RepairEvent(
            original_order_id="o1",
            replacement_order_id="o1__repair",
            symbol="AAPL  260116C00100000",
            side="buy",
            repair_reason="passive_limit_not_filled",
            decision_mark=1.06,
            original_limit_price=1.00,
            replacement_limit_price=1.06,
            effective_fill_price=1.06,
            adverse_drift_bps=0.0,
            fill_fraction=1.0,
            filled_contracts=1,
            unfilled_contracts=0,
            action="replaced_filled",
            timestamp=pd.Timestamp("2026-01-02T14:30:00.050Z"),
        )
    ]).to_csv(out / "repair_ledger.csv", index=False)


def test_production_verifier_passes_on_complete_synthetic_ledgers(tmp_path):
    out = tmp_path / "prod"
    _write_synthetic_production_ledgers(out)
    summary = ProductionVerifier(ROOT / "research/papers/option_only_markowitz", out).run()
    assert summary["status"] == "pass"
    assert summary["critical_failures"] == 0

    (out / "repair_ledger.csv").unlink()
    summary = ProductionVerifier(ROOT / "research/papers/option_only_markowitz", out).run()
    assert summary["status"] == "pass"
    assert summary["critical_failures"] == 0


def test_production_verifier_fails_without_ledgers(tmp_path):
    out = tmp_path / "empty"
    summary = ProductionVerifier(ROOT / "research/papers/option_only_markowitz", out).run()
    assert summary["status"] == "fail"
    assert summary["critical_failures"] > 0
