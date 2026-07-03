from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_ingestion.build_cbbo_cost_surface import (
    assign_moneyness_bucket,
    assign_tenor_bucket,
    build_daily_spread_surface,
    parse_osi_symbol,
)


def _ny_quote_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")


def _row(
    *,
    symbol: str = "AAPL  240216C00100000",
    snap_date: str = "2024-01-02",
    ts_recv: str = "2024-01-02 15:35:00",
    bid: float = 99.0,
    ask: float = 101.0,
    bid_size: int = 10,
    ask_size: int = 100,
    spot: float = 100.0,
) -> dict[str, object]:
    return {
        "ts_recv": _ny_quote_ts(ts_recv),
        "bid_px_00": bid,
        "ask_px_00": ask,
        "bid_sz_00": bid_size,
        "ask_sz_00": ask_size,
        "symbol": symbol,
        "snap_date": pd.Timestamp(snap_date),
        "spot": spot,
        "underlying": "AAPL",
    }


def test_parse_osi_symbol_goldens() -> None:
    assert parse_osi_symbol("AAPL  260116C00100000") == (
        "AAPL",
        pd.Timestamp("2026-01-16"),
        "C",
        100.0,
    )
    assert parse_osi_symbol("SPY   200221P00328000") == (
        "SPY",
        pd.Timestamp("2020-02-21"),
        "P",
        328.0,
    )
    assert parse_osi_symbol("FB    200221C00200000") == (
        "FB",
        pd.Timestamp("2020-02-21"),
        "C",
        200.0,
    )


def test_moneyness_bucket_edges_match_equity_panel_thresholds() -> None:
    spot = 100.0
    assert assign_moneyness_bucket(spot, spot * 1.0, "C") == "atm"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**0.03, "C") == "atm"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**0.0301, "P") == "call_near"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**0.10, "C") == "call_near"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**0.1001, "C") == "call_wing"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**0.2001, "C") == "other"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**-0.0301, "C") == "put_near"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**-0.10, "P") == "put_near"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**-0.1001, "P") == "put_wing"
    assert assign_moneyness_bucket(spot, spot * 2.718281828459045**-0.2001, "P") == "other"


def test_tenor_bucket_edges() -> None:
    assert assign_tenor_bucket(45) == "le_45d"
    assert assign_tenor_bucket(46) == "46_120d"
    assert assign_tenor_bucket(120) == "46_120d"
    assert assign_tenor_bucket(121) == "gt_120d"


def test_window_filtering_and_invalid_quotes_are_excluded() -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(ts_recv="2024-01-02 15:29:00"),
            _row(symbol="AAPL  240216P00100000", bid=100.0, ask=100.0),
            _row(symbol="AAPL  240216C00105000", bid=0.0, ask=101.0),
            _row(symbol="AAPL  240216P00095000", bid=99.0, ask=101.0, ask_size=0),
        ]
    )

    surface = build_daily_spread_surface(frame)

    assert len(surface) == 1
    assert surface.loc[0, "n_quotes"] == 1
    assert surface.loc[0, "n_contracts"] == 1
    assert surface.loc[0, "median_relative_spread"] == pytest.approx(0.02)


def test_aggregation_math_on_hand_computed_fixture() -> None:
    frame = pd.DataFrame(
        [
            _row(symbol="AAPL  240216C00100000", ts_recv="2024-01-02 15:35:00", bid=99, ask=101, bid_size=10),
            _row(symbol="AAPL  240216P00100000", ts_recv="2024-01-02 15:40:00", bid=98, ask=102, bid_size=20),
            _row(symbol="AAPL  240216C00100000", ts_recv="2024-01-02 15:45:00", bid=97, ask=103, bid_size=30),
            _row(symbol="AAPL  240216P00100000", ts_recv="2024-01-02 15:50:00", bid=96, ask=104, bid_size=40),
        ]
    )

    surface = build_daily_spread_surface(frame)

    assert len(surface) == 1
    row = surface.iloc[0]
    assert row["underlying"] == "AAPL"
    assert row["snap_date"] == pd.Timestamp("2024-01-02")
    assert row["moneyness_bucket"] == "atm"
    assert row["tenor_bucket"] == "le_45d"
    assert row["n_quotes"] == 4
    assert row["n_contracts"] == 2
    assert row["median_relative_spread"] == pytest.approx(0.05)
    assert row["p25_relative_spread"] == pytest.approx(0.035)
    assert row["p75_relative_spread"] == pytest.approx(0.065)
    assert row["median_mid"] == pytest.approx(100.0)
    assert row["median_displayed_size"] == pytest.approx(25.0)


def test_snap_dates_are_independent_no_point_in_time_bleed() -> None:
    date_a = _row(snap_date="2024-01-02", ts_recv="2024-01-02 15:35:00", bid=99, ask=101)
    date_b = _row(snap_date="2024-01-03", ts_recv="2024-01-03 15:35:00", bid=90, ask=110)
    base = build_daily_spread_surface(pd.DataFrame([date_a, date_b]))

    perturbed_b = _row(snap_date="2024-01-03", ts_recv="2024-01-03 15:35:00", bid=50, ask=150)
    changed = build_daily_spread_surface(pd.DataFrame([date_a, perturbed_b]))

    base_a = base.loc[base["snap_date"].eq(pd.Timestamp("2024-01-02"))].reset_index(drop=True)
    changed_a = changed.loc[changed["snap_date"].eq(pd.Timestamp("2024-01-02"))].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_a, changed_a)
