"""Tests for diagnostic VIX option-chain state features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.vix_chain_features import (  # noqa: E402
    build_vix_chain_state_features,
    vol_of_vol_regime_table,
)


def _symbol(expiry: str, kind: str, strike: float) -> str:
    cp = "C" if kind == "call" else "P"
    ymd = pd.Timestamp(expiry).strftime("%y%m%d")
    strike_int = int(round(strike * 1000))
    return f"VIX   {ymd}{cp}{strike_int:08d}"


def _row(
    trade_date: str,
    expiry: str,
    kind: str,
    strike: float,
    close: float,
    iv: float,
    *,
    open_interest: float = 10.0,
    forward: float = 20.0,
) -> dict[str, object]:
    return {
        "ts_event": pd.Timestamp(trade_date, tz="UTC"),
        "rtype": 35,
        "publisher_id": 22,
        "instrument_id": abs(hash((trade_date, expiry, kind, strike))) % 1_000_000,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1,
        "symbol": _symbol(expiry, kind, strike),
        "iv": iv,
        "open_interest": open_interest,
        "vix_forward": forward,
    }


def _write_chain(root: Path, rows: list[dict[str, object]]) -> None:
    cache = root / "data" / "databento_cache"
    cache.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(cache / "opra_vix_chain_2024-01.parquet", index=False)


def _math_chain_rows(trade_date: str = "2024-01-31") -> list[dict[str, object]]:
    return [
        _row(trade_date, "2024-02-28", "call", 20.0, 2.0, 0.80, open_interest=10),
        _row(trade_date, "2024-02-28", "put", 20.0, 2.1, 0.82, open_interest=10),
        _row(trade_date, "2024-02-28", "call", 26.0, 1.0, 1.10, open_interest=30),
        _row(trade_date, "2024-03-13", "call", 20.0, 2.5, 0.90, open_interest=10),
        _row(trade_date, "2024-03-13", "put", 20.0, 2.6, 0.92, open_interest=10),
        _row(trade_date, "2024-03-13", "call", 26.0, 1.5, 1.20, open_interest=20),
    ]


def test_feature_math_on_hand_built_chain(tmp_path: Path) -> None:
    _write_chain(tmp_path, _math_chain_rows())

    out = build_vix_chain_state_features(tmp_path, [pd.Timestamp("2024-01-31")])
    row = out.loc[pd.Timestamp("2024-01-31")]

    assert row["n_contracts"] == 6
    assert row["atm_iv_proxy"] == pytest.approx(0.86)
    assert row["skew_proxy"] == pytest.approx(1.15 - 0.86)
    assert row["term_slope"] == pytest.approx(0.81 - 0.91)
    assert row["call_wing_premium_share"] == pytest.approx(60.0 / 152.0)


def test_point_in_time_chain_features_do_not_read_future_rows(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    poison_root = tmp_path / "poison"
    base_rows = _math_chain_rows()
    poison_rows = _math_chain_rows("2024-02-01")
    for row in poison_rows:
        row["iv"] = 9.99
        row["close"] = 99.0
        row["open"] = 99.0
        row["high"] = 99.0
        row["low"] = 99.0

    _write_chain(clean_root, base_rows)
    _write_chain(poison_root, base_rows + poison_rows)

    clean = build_vix_chain_state_features(clean_root, [pd.Timestamp("2024-01-31")])
    poisoned = build_vix_chain_state_features(poison_root, [pd.Timestamp("2024-01-31")])
    pd.testing.assert_series_equal(clean.loc[pd.Timestamp("2024-01-31")], poisoned.loc[pd.Timestamp("2024-01-31")])


def test_regime_table_uses_prior_date_feature_not_same_date_poison() -> None:
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({"strategy": [0.01, 0.02, -0.01, 0.03, 0.00, 0.04]}, index=idx)
    features = pd.DataFrame({"atm_iv_proxy": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    poisoned = features.copy()
    poisoned.iloc[-1, 0] = 9999.0

    base_table = vol_of_vol_regime_table(returns, features, n_buckets=3)
    poisoned_table = vol_of_vol_regime_table(returns, poisoned, n_buckets=3)
    base_table.attrs.clear()
    poisoned_table.attrs.clear()
    pd.testing.assert_frame_equal(base_table, poisoned_table)


def test_empty_tenor_window_and_single_expiry_handling(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty_window"
    _write_chain(empty_root, [_row("2024-01-31", "2024-02-07", "call", 20.0, 1.0, 0.8)])
    empty = build_vix_chain_state_features(empty_root, [pd.Timestamp("2024-01-31")])
    assert empty.loc[pd.Timestamp("2024-01-31")].isna().all()

    single_root = tmp_path / "single_expiry"
    _write_chain(
        single_root,
        [
            _row("2024-01-31", "2024-02-28", "call", 20.0, 2.0, 0.80),
            _row("2024-01-31", "2024-02-28", "put", 20.0, 2.1, 0.82),
            _row("2024-01-31", "2024-02-28", "call", 26.0, 1.0, 1.10),
        ],
    )
    single = build_vix_chain_state_features(single_root, [pd.Timestamp("2024-01-31")])
    row = single.loc[pd.Timestamp("2024-01-31")]
    assert row["atm_iv_proxy"] == pytest.approx(0.81)
    assert np.isnan(row["term_slope"])
    assert row["n_contracts"] == 3


def test_regime_table_bucket_invariants_and_nan_feature_drops() -> None:
    idx = pd.date_range("2024-01-31", periods=7, freq="ME")
    returns = pd.DataFrame(
        {
            "a": [0.01, 0.02, 0.03, np.nan, -0.01, 0.00, 0.04],
            "b": [0.02, 0.01, -0.02, 0.03, 0.01, 0.02, 0.05],
        },
        index=idx,
    )
    features = pd.DataFrame({"atm_iv_proxy": [np.nan, 1.0, 2.0, 2.0, 3.0, np.nan, 4.0]}, index=idx)

    table = vol_of_vol_regime_table(returns, features, n_buckets=3)
    assert table.index.get_level_values("bucket").nunique() <= 3
    assert table["n_months"].gt(0).all()

    prior = features["atm_iv_proxy"].reindex(idx).shift(1)
    for strategy in returns.columns:
        expected = pd.DataFrame({"return": returns[strategy], "feature": prior}).dropna().shape[0]
        observed = int(table.xs(strategy, level="strategy")["n_months"].sum())
        assert observed == expected
