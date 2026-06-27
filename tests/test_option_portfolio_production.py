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
    OptionContract,
    OptionOrder,
    OrderPolicy,
    PaperBrokerAdapter,
    Position,
    QuoteSnapshot,
    RiskGateConfig,
    assign_short_option,
    attach_exact_vix_settlement,
    build_execution_ledger,
    build_fill_ledger,
    build_margin_ledger,
    build_market_data_ledger,
    conservative_order_margin,
    early_exercise_risk,
    estimate_nbbo_fill,
    evaluate_pre_trade_gate,
    normalize_vro_soq_frame,
    post_cost_expected_returns,
    reconcile_quote_pair,
    require_exact_vix_settlement,
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


def test_production_verifier_passes_on_complete_synthetic_ledgers(tmp_path):
    out = tmp_path / "prod"
    _write_synthetic_production_ledgers(out)
    summary = ProductionVerifier(ROOT / "research/papers/option_only_markowitz", out).run()
    assert summary["status"] == "pass"
    assert summary["critical_failures"] == 0


def test_production_verifier_fails_without_ledgers(tmp_path):
    out = tmp_path / "empty"
    summary = ProductionVerifier(ROOT / "research/papers/option_only_markowitz", out).run()
    assert summary["status"] == "fail"
    assert summary["critical_failures"] > 0
