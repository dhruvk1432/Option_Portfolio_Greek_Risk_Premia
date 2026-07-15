from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import research.papers.option_only_markowitz.analysis.r1_r11_execution_audit as audit
from data_ingestion.market_data.fetch_r1_r11_databento_audit import close_window, open_window


def _request(
    cache: Path,
    ledger: dict[str, dict[str, object]],
    request_id: str,
    purpose: str,
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    schema: str = "cbbo-1m",
) -> None:
    path = cache / "phase2" / purpose / f"{request_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    ledger[request_id] = {
        "status": "complete",
        "columns": frame.columns.tolist(),
        "request": {
            "phase": 2,
            "purpose": purpose,
            "schema": schema,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": sorted(frame["symbol"].astype(str).unique()),
        },
    }


def _quote_frame(
    symbol: str,
    timestamps: list[pd.Timestamp],
    bids: list[float],
    asks: list[float],
    bid_sizes: list[float] | None = None,
    ask_sizes: list[float] | None = None,
) -> pd.DataFrame:
    size = len(timestamps)
    return pd.DataFrame(
        {
            "ts_recv": pd.to_datetime(timestamps, utc=True),
            "ts_event": pd.to_datetime(timestamps, utc=True),
            "symbol": symbol,
            "bid_px_00": bids,
            "ask_px_00": asks,
            "bid_sz_00": bid_sizes or [10.0] * size,
            "ask_sz_00": ask_sizes or [12.0] * size,
        }
    )


def _trades(symbol: str = "TEST  180720C00100000") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["R1.1"],
            "config": ["orig+VIX"],
            "decision_date": [pd.Timestamp("2018-07-03")],
            "return_date": [pd.Timestamp("2018-07-31")],
            "expiry": [pd.Timestamp("2018-07-20")],
            "asset_id": ["TEST_call_atm"],
            "symbol": [symbol],
            "underlying": ["TEST"],
            "mark": [1.05],
            "weight": [0.10],
            "integer_contracts": [5.0],
            "contracts_source": ["frozen_integerized"],
            "modeled_relative_spread": [0.10],
            "modeled_fee_rate": [0.005],
            "modeled_slippage_rate": [0.001],
            "modeled_long_funding_rate": [0.019],
            "modeled_short_funding_rate": [0.029],
            "modeled_short_borrow_rate": [0.0],
            "modeled_long_rate": [0.125],
            "modeled_short_rate": [0.135],
        }
    )


def _quotes_for_trade(symbol: str = "TEST  180720C00100000") -> pd.DataFrame:
    close_start, close_end = close_window(pd.Timestamp("2018-07-03"))
    open_start, open_end = open_window(pd.Timestamp("2018-07-03"))
    exit_start, exit_end = close_window(pd.Timestamp("2018-07-20"))
    pieces = []
    for purpose, start, end, bid, ask in [
        ("candidate_close_quotes", close_start, close_end, 1.0, 1.1),
        ("held_next_open", open_start, open_end, 0.95, 1.15),
        ("held_exit_close", exit_start, exit_end, 0.45, 0.55),
    ]:
        frame = _quote_frame(symbol, [end - pd.Timedelta(seconds=1)], [bid], [ask])
        frame["purpose"] = purpose
        frame["request_start"] = start
        frame["request_end"] = end
        frame["schema"] = "cbbo-1m"
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def test_loader_complete_purpose_and_missing_count(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    ledger: dict[str, dict[str, object]] = {}
    start = pd.Timestamp("2023-03-27 19:50", tz="UTC")
    end = pd.Timestamp("2023-03-27 20:00", tz="UTC")
    frame = _quote_frame("ABC   230421C00100000", [end], [1.0], [1.2])
    _request(cache, ledger, "good", "candidate_close_quotes", frame, start, end, "cbbo-1m")
    ledger["missing"] = {
        "status": "complete",
        "columns": frame.columns.tolist(),
        "request": {
            "phase": 2,
            "purpose": "candidate_close_quotes",
            "schema": "cmbp-1",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": ["ABC   230421C00100000"],
        },
    }
    ledger["incomplete"] = {
        **ledger["missing"],
        "status": "pending",
    }
    (cache / "request_ledger.json").parent.mkdir(parents=True, exist_ok=True)
    (cache / "request_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    loaded = audit.load_audit_frames(cache, {"candidate_close_quotes"})

    assert len(loaded) == 1
    assert loaded.iloc[0]["purpose"] == "candidate_close_quotes"
    assert loaded.iloc[0]["schema"] == "cbbo-1m"
    assert loaded.attrs["loaded_files"] == 1
    assert loaded.attrs["skipped_files"] == 1


def test_match_quotes_last_valid_and_early_close_boundary() -> None:
    trades = _trades()
    quotes = _quotes_for_trade()
    start, end = close_window(pd.Timestamp("2018-07-03"))
    assert end == pd.Timestamp("2018-07-03 17:00", tz="UTC")
    extra = _quote_frame(
        trades.iloc[0]["symbol"],
        [end - pd.Timedelta(seconds=2), end + pd.Timedelta(seconds=1)],
        [0.90, 2.0],
        [1.10, 2.1],
    )
    extra["purpose"] = "candidate_close_quotes"
    extra["request_start"] = start
    extra["request_end"] = end + pd.Timedelta(minutes=5)
    extra["schema"] = "cbbo-1m"
    quotes = pd.concat([quotes, extra], ignore_index=True)

    matched = audit.match_quotes(trades, quotes)

    assert matched.iloc[0]["obs_bid"] == pytest.approx(1.0)
    assert matched.iloc[0]["coverage"] == "covered"
    assert matched.iloc[0]["entry_window_max_rel_spread"] == pytest.approx(0.20)


def test_invalid_and_one_sided_quotes_are_not_covered() -> None:
    trades = _trades()
    quotes = _quotes_for_trade()
    entry = quotes["purpose"].eq("candidate_close_quotes")
    quotes.loc[entry, ["bid_px_00", "ask_px_00"]] = [0.0, 1.0]

    matched = audit.match_quotes(trades, quotes)

    assert matched.iloc[0]["coverage"] == "one_sided"
    assert np.isnan(matched.iloc[0]["obs_mid"])


def test_schema_cutover_is_preserved_by_loader(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    ledger: dict[str, dict[str, object]] = {}
    for request_id, day, schema in [
        ("pre", "2023-03-27", "cbbo-1m"),
        ("post", "2023-03-28", "cmbp-1"),
    ]:
        start = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=19, minutes=50)
        end = start + pd.Timedelta(minutes=10)
        frame = _quote_frame(f"{request_id.upper():<6}230421C00100000", [end], [1.0], [1.1])
        _request(cache, ledger, request_id, "candidate_close_quotes", frame, start, end, schema)
    (cache / "request_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    loaded = audit.load_audit_frames(cache, {"candidate_close_quotes"})

    assert set(loaded["schema"]) == {"cbbo-1m", "cmbp-1"}


def test_modeled_cost_uses_latest_prior_input_and_exact_terms() -> None:
    trades = pd.DataFrame(
        {
            "config": ["orig"],
            "asset_id": ["TEST_call_atm"],
            "decision_date": [pd.Timestamp("2020-02-01")],
            "expiry": [pd.Timestamp("2020-02-21")],
            "underlying": ["TEST"],
        }
    )
    modeled = pd.DataFrame(
        {
            "config": ["orig", "orig"],
            "asset_id": ["TEST_call_atm", "TEST_call_atm"],
            "modeled_cost_input_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")],
            "modeled_cost_input_mark": [2.0, 99.0],
            "relative_spread": [0.02, 0.90],
            "relative_spread_source": ["historical", "future"],
            "holding_years": [20 / 365, 20 / 365],
            "start_spot": [100.0, 100.0],
            "kind": ["call", "call"],
            "asset_class": ["equity_option", "equity_option"],
            "borrow_rate_proxy": [0.01, 0.01],
        }
    )

    result = audit._attach_modeled_costs(trades, modeled)

    expected_fee = 2 * 0.75 / (2.0 * 100)
    expected_long = 0.02 + expected_fee + 0.001 + 0.02 * 20 / 365
    assert result.iloc[0]["modeled_cost_input_date"] == pd.Timestamp("2020-01-01")
    assert result.iloc[0]["modeled_cost_input_lag_days"] == 31
    assert result.iloc[0]["modeled_long_rate"] == pytest.approx(expected_long, abs=1e-15)


def test_vix_exit_uses_cached_last_tradable_session_path() -> None:
    symbol = "VIX   210217C00035000"
    trades = _trades(symbol)
    trades["underlying"] = "VX_FRONT"
    trades["decision_date"] = pd.Timestamp("2021-01-29")
    trades["return_date"] = pd.Timestamp("2021-02-26")
    trades["expiry"] = pd.Timestamp("2021-02-17")
    entry_start, entry_end = close_window(pd.Timestamp("2021-01-29"))
    open_start, open_end = open_window(pd.Timestamp("2021-01-29"))
    pieces = []
    for purpose, start, end in [
        ("candidate_close_quotes", entry_start, entry_end),
        ("held_next_open", open_start, open_end),
    ]:
        frame = _quote_frame(symbol, [end - pd.Timedelta(seconds=1)], [1.0], [1.2])
        frame["purpose"] = purpose
        frame["request_start"] = start
        frame["request_end"] = end
        frame["schema"] = "cbbo-1m"
        pieces.append(frame)
    path_end = pd.Timestamp("2021-02-16 21:15", tz="UTC")
    path = _quote_frame(
        symbol,
        [path_end - pd.Timedelta(minutes=9), path_end],
        [0.30, 0.35],
        [0.40, 0.45],
    )
    path["purpose"] = audit.PATH_QUOTE_PURPOSE
    path["request_start"] = pd.Timestamp("2021-02-01 14:30", tz="UTC")
    path["request_end"] = pd.Timestamp("2021-02-17 21:01", tz="UTC")
    path["schema"] = "cbbo-1m"
    pieces.append(path)

    matched = audit.match_quotes(trades, pd.concat(pieces, ignore_index=True))

    assert matched.iloc[0]["exit_obs_coverage"] == "covered"
    assert matched.iloc[0]["exit_obs_source"] == "held_cbbo_path_last_tradable_session"
    assert matched.iloc[0]["exit_obs_bid"] == pytest.approx(0.35)


def test_recompute_touch_cost_and_missing_quote_fallback() -> None:
    matched = audit.match_quotes(_trades(), _quotes_for_trade())
    monthly = pd.DataFrame(
        {
            "arm": ["R1.1"],
            "config": ["orig+VIX"],
            "return_date": [pd.Timestamp("2018-07-31")],
            "decision_date": [pd.Timestamp("2018-07-03")],
            "gross_return": [0.20],
            "gross_nav": [0.10],
            "predicted_cost": [0.0125],
            "net_return": [0.1875],
        }
    )

    result = audit.recompute_costs(matched, monthly)

    entry_spread = (1.1 - 1.0) / 1.05
    exit_spread = (0.55 - 0.45) / 0.50
    expected = 0.10 * (0.025 + (entry_spread + exit_spread) / 2.0)
    assert result.iloc[0]["observed_cost_touch"] == pytest.approx(expected, abs=1e-12)
    assert result.iloc[0]["net_return_touch"] <= result.iloc[0]["net_return_mid"]

    missing = matched.copy()
    missing["exit_obs_coverage"] = "missing"
    fallback = audit.recompute_costs(missing, monthly)
    assert fallback.iloc[0]["observed_cost_touch"] == pytest.approx(0.0125)
    assert fallback.iloc[0]["gross_return"] == monthly.iloc[0]["gross_return"]
    assert fallback.iloc[0]["coverage_weight_fraction"] == 0.0


def test_zero_position_month_has_nan_coverage_and_modeled_cost_passthrough() -> None:
    matched = audit.match_quotes(_trades(), _quotes_for_trade())
    monthly = pd.DataFrame(
        {
            "arm": ["R1.1"],
            "config": ["orig+VIX"],
            "return_date": [pd.Timestamp("2018-08-31")],
            "decision_date": [pd.Timestamp("2018-08-01")],
            "gross_return": [0.02],
            "gross_nav": [0.0],
            "predicted_cost": [0.001],
            "net_return": [0.019],
        }
    )

    result, positions = audit._recompute_cost_details(matched, monthly)

    assert np.isnan(result.iloc[0]["coverage_weight_fraction"])
    assert result.iloc[0]["observed_cost_mid"] == pytest.approx(0.001)
    assert result.iloc[0]["observed_cost_touch"] == pytest.approx(0.001)
    assert result.iloc[0]["observed_cost_worst"] == pytest.approx(0.001)
    assert not positions.empty
    assert not hasattr(audit, "_LAST_SCENARIO_POSITIONS")


def test_build_trade_table_labels_contract_count_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r11_path = tmp_path / "r11.csv"
    r1_path = tmp_path / "r1.csv"
    pd.DataFrame(
        {
            "strategy": [audit.R11_NAME],
            "config": ["orig"],
            "decision_date": ["2018-07-03"],
            "return_date": ["2018-07-31"],
            "asset_id": ["TEST_call_atm"],
            "symbol": ["TEST  180720C00100000"],
            "underlying": ["TEST"],
            "expiry": ["2018-07-20"],
            "mark": [1.0],
            "weight": [0.1],
            "integer_contracts": [1000.0],
        }
    ).to_csv(r11_path, index=False)
    pd.DataFrame(
        {
            "config": ["orig"],
            "decision_date": ["2018-07-03"],
            "return_date": ["2018-07-31"],
            "asset_id": ["TEST_call_atm"],
            "weight": [0.1],
        }
    ).to_csv(r1_path, index=False)
    monkeypatch.setattr(audit, "R11_WEIGHTS", r11_path)
    monkeypatch.setattr(audit, "R1_WEIGHTS", r1_path)
    monkeypatch.setattr(audit, "_apply_resolved_symbols", lambda trades: trades)
    monkeypatch.setattr(audit, "_modeled_cost_inputs", pd.DataFrame)
    monkeypatch.setattr(audit, "_attach_modeled_costs", lambda trades, _modeled: trades)

    result = audit.build_trade_table().set_index("arm")

    assert result.loc["R1.1", "contracts_source"] == "frozen_integerized"
    assert result.loc["R1", "contracts_source"] == "implied_from_weight_at_1mm_nav"


def test_volume_publisher_aggregation_and_oi_filter() -> None:
    trades = _trades()
    symbol = trades.iloc[0]["symbol"]
    decision = pd.Timestamp("2018-07-03", tz="UTC")
    data = pd.DataFrame(
        {
            "purpose": ["candidate_daily_volume"] * 2 + ["candidate_open_interest"] * 2,
            "request_start": [decision] * 4,
            "symbol": [symbol] * 4,
            "volume": [60.0, 40.0, np.nan, np.nan],
            "stat_type": [np.nan, np.nan, 9.0, 8.0],
            "quantity": [np.nan, np.nan, 200.0, 9999.0],
        }
    )

    result = audit._liquidity_validation(trades, data)

    assert result.iloc[0]["volume_sum"] == 100.0
    assert result.iloc[0]["volume_max"] == 60.0
    assert result.iloc[0]["participation_volume_sum"] == pytest.approx(0.05)
    assert result.iloc[0]["participation_volume_max"] == pytest.approx(5 / 60)
    assert result.iloc[0]["open_interest"] == 200.0
    assert result.iloc[0]["participation_open_interest"] == pytest.approx(0.025)
    assert bool(result.iloc[0]["breach_capacity_oi_0_02"])


def test_intervention_presence_and_vwap_mid() -> None:
    symbol = "VIX   200320C00030000"
    orders = pd.DataFrame(
        {"execution_date": ["2020-03-02"], "symbol": [symbol], "order_contracts": [2.0]}
    )
    start = pd.Timestamp("2020-03-02 14:30", tz="UTC")
    end = pd.Timestamp("2020-03-02 14:34", tz="UTC")
    quotes = _quote_frame(
        symbol,
        [start + pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=3)],
        [1.0, 2.0],
        [1.2, 2.4],
        [1.0, 3.0],
        [1.0, 3.0],
    )
    quotes["purpose"] = "vix_intervention_cbbo"
    quotes["request_start"] = start
    quotes["request_end"] = end

    result = audit._intervention_evidence(orders, quotes)

    assert result.iloc[0]["quote_presence_fraction"] == pytest.approx(0.5)
    assert result.iloc[0]["quote_vwap_mid"] == pytest.approx((1.1 * 2 + 2.2 * 6) / 8)
    assert result.iloc[0]["first_valid_touch"] == pytest.approx(1.2)
    assert bool(result.iloc[0]["evidence_only"])


def test_mark_accuracy_regimes() -> None:
    matched = pd.DataFrame(
        {
            "arm": ["R1", "R1.1"],
            "config": ["orig", "orig+VIX"],
            "decision_date": ["2023-03-27", "2023-03-28"],
            "symbol": ["A", "B"],
            "mark": [1.0, 2.0],
            "obs_mid": [1.1, 1.5],
            "coverage": ["covered", "covered"],
            "quote_schema": ["cbbo-1m", "cmbp-1"],
        }
    )
    result = audit.mark_accuracy(matched)
    assert result["regime"].tolist() == ["pre_2023_03_28", "post_2023_03_28"]
    assert result.iloc[0]["absolute_error"] == pytest.approx(0.1)


def test_main_writes_only_custom_out_and_preserves_frozen_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    frozen = audit.R11_MONTHLY
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    trades = _trades()
    quotes = _quotes_for_trade()
    symbol = trades.iloc[0]["symbol"]
    liquidity = pd.DataFrame(
        {
            "purpose": ["candidate_daily_volume", "candidate_open_interest"],
            "request_start": [pd.Timestamp("2018-07-03", tz="UTC")] * 2,
            "symbol": [symbol] * 2,
            "volume": [100.0, np.nan],
            "stat_type": [np.nan, 9.0],
            "quantity": [np.nan, 500.0],
        }
    )
    monthly = pd.DataFrame(
        {
            "arm": ["R1.1"],
            "config": ["orig+VIX"],
            "return_date": [pd.Timestamp("2018-07-31")],
            "decision_date": [pd.Timestamp("2018-07-03")],
            "gross_return": [0.20],
            "gross_nav": [0.10],
            "predicted_cost": [0.0125],
            "net_return": [0.1875],
            "_gross_return_frozen_text": ["0.200000000000000000"],
        }
    )

    def fake_loader(_cache: Path, purposes: set[str], _symbols: set[str]) -> pd.DataFrame:
        if purposes == audit.BASE_QUOTE_PURPOSES:
            frame = quotes
        elif purposes == {audit.PATH_QUOTE_PURPOSE}:
            frame = quotes.iloc[0:0]
        else:
            frame = liquidity
        frame = frame.copy()
        frame.attrs.update({"loaded_files": 1, "skipped_files": 0, "selected_files": 1})
        return frame

    monkeypatch.setattr(audit, "build_trade_table", lambda: trades)
    monkeypatch.setattr(audit, "_load_monthly", lambda: monthly)
    monkeypatch.setattr(audit, "_load_audit_frames", fake_loader)
    monkeypatch.setattr(sys, "argv", ["execution-audit", "--cache-root", str(tmp_path), "--out", str(out), "--skip-intervention"])

    audit.main()

    expected = {
        "execution_fill_ledger.csv",
        "execution_audit_monthly_returns.csv",
        "execution_audit_summary.json",
        "liquidity_gate_validation.csv",
        "intervention_day_fill_evidence.csv",
        "mark_accuracy.csv",
        "README.md",
        "short_execution_audit_summary.tex",
        "short_execution_spread_comparison.tex",
        "short_liquidity_validation.tex",
    }
    assert expected.issubset({path.name for path in out.iterdir()})
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before
    summary = json.loads((out / "execution_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["raw_licensed_data_committed"] is False
    assert summary["coverage"][0]["entry_coverage"] == pytest.approx(1.0)
    assert summary["coverage"][0]["roundtrip_coverage"] == pytest.approx(1.0)
    assert "Entry cov. \\%" in (out / "short_execution_audit_summary.tex").read_text(
        encoding="utf-8"
    )
    assert "Round-trip cov. \\%" in (out / "short_execution_audit_summary.tex").read_text(
        encoding="utf-8"
    )
    fills = pd.read_csv(out / "execution_fill_ledger.csv")
    liquidity_output = pd.read_csv(out / "liquidity_gate_validation.csv")
    assert fills["contracts_source"].eq("frozen_integerized").all()
    assert liquidity_output["contracts_source"].eq("frozen_integerized").all()
    with (out / "execution_audit_monthly_returns.csv").open(newline="") as handle:
        written = next(csv.DictReader(handle))
    assert written["gross_return"] == "0.200000000000000000"


def test_analysis_module_has_no_banned_rowwise_pandas_patterns() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    banned = ["iterrows", "itertuples", "applymap", ".apply(", "transform(lambda"]
    assert not any(pattern in source for pattern in banned)
