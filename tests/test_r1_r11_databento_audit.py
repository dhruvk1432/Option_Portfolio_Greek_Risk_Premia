from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import data_ingestion.market_data.fetch_r1_r11_databento_audit as audit


class FakeStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def to_df(self, **_kwargs):
        return self.frame.copy()

    def to_file(self, path, **_kwargs):
        Path(path).write_bytes(b"fake-dbn")


class FakeMetadata:
    def __init__(self, cost: float):
        self.cost = cost
        self.calls = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        return self.cost


class FakeTimeseries:
    def __init__(self, frame: pd.DataFrame | None = None):
        self.frame = frame if frame is not None else pd.DataFrame({"symbol": ["AAPL  200117C00300000"]})
        self.calls = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        Path(kwargs["path"]).write_bytes(b"fake-dbn")
        return FakeStore(self.frame)


class FailingTimeseries:
    def get_range(self, **_kwargs):
        raise RuntimeError("402 account_insufficient_funds")


class FakeClient:
    def __init__(self, cost: float = 1.0, frame: pd.DataFrame | None = None):
        self.metadata = FakeMetadata(cost)
        self.timeseries = FakeTimeseries(frame)


def request(phase: int = 1, purpose: str = "definition") -> audit.DataRequest:
    return audit.DataRequest(
        phase=phase,
        purpose=purpose,
        dataset="OPRA.PILLAR",
        schema="definition",
        start="2020-01-02T00:00:00",
        end="2020-01-03T00:00:00",
        symbols=("AAPL  200117C00300000",),
    )


def test_primary_project_key_is_the_only_accepted_key(tmp_path):
    env = tmp_path / ".env"
    primary = "a" * 32
    env.write_text(f"DATABENTO_API_KEY={primary}\nDATABENTO_API_KEY2={'b' * 32}\n")
    assert audit.load_primary_key(env) == primary
    env.write_text(f"DATABENTO_API_KEY2={'b' * 32}\n")
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        audit.load_primary_key(env)


def test_request_hash_is_order_independent_and_contains_no_credentials():
    left = request()
    right = audit.DataRequest(**{**left.__dict__, "symbols": tuple(reversed(left.symbols))})
    assert left.request_id == right.request_id
    assert "KEY" not in json.dumps(left.normalized())


def test_close_window_uses_the_early_close_and_open_uses_next_session():
    start, end = audit.close_window(pd.Timestamp("2020-11-27"))
    assert end == pd.Timestamp("2020-11-27 18:00:00+00:00")
    assert end - start == pd.Timedelta(minutes=10)
    opened, finish = audit.open_window(pd.Timestamp("2020-02-28"))
    assert opened == pd.Timestamp("2020-03-02 14:30:00+00:00")
    assert finish - opened == pd.Timedelta(minutes=10)


def test_option_schema_cutover_and_vx_symbol_conversion():
    assert audit._schema_for_option_events(pd.Timestamp("2023-03-27")) == "cbbo-1m"
    assert audit._schema_for_option_events(pd.Timestamp("2023-03-28", tz="UTC")) == "cmbp-1"
    assert audit._vx_raw_symbol("VXZ18") == "VX/Z8"
    assert audit._vx_raw_symbol("VXG5") == "VX/G5"
    assert audit._vix_moneyness_bucket(20.0, 20.0, "call") == "vix_atm"
    assert audit._vix_moneyness_bucket(20.0, 25.0, "call") == "vix_call_near"


def test_gap_resolution_applies_mark_volume_spread_and_stable_tie_break():
    candidates = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2020-01-31"] * 3),
            "asset_id": ["AAPL_call_atm"] * 3,
            "underlying": ["AAPL"] * 3,
            "symbol": ["B", "A", "C"],
            "expiry": pd.to_datetime(["2020-02-21"] * 3),
        }
    )
    quotes = [
        pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "bid_px_00": [1.0, 1.0, 0.10],
                "ask_px_00": [1.1, 1.1, 0.50],
            }
        )
    ]
    volumes = [pd.DataFrame({"symbol": ["A", "B", "C"], "volume": [20, 20, 100]})]
    selected = audit.resolve_gap_contracts(candidates, quotes, volumes)
    assert selected["symbol"].tolist() == ["A"]
    assert selected["resolution_status"].tolist() == ["resolved_exact_osi"]


def test_no_eligible_gap_contract_is_explicit():
    candidates = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2020-01-31"]),
            "asset_id": ["AAPL_call_atm"],
            "underlying": ["AAPL"],
            "symbol": ["A"],
            "expiry": pd.to_datetime(["2020-02-21"]),
        }
    )
    quotes = [pd.DataFrame({"symbol": ["A"], "bid_px_00": [1.0], "ask_px_00": [1.5]})]
    volumes = [pd.DataFrame({"symbol": ["A"], "volume": [5]})]
    assert audit.resolve_gap_contracts(candidates, quotes, volumes).empty


def test_phase_two_is_blocked_before_download_when_cumulative_cap_is_exceeded(tmp_path):
    client = FakeClient(cost=6.0)
    runner = audit.AcquisitionRunner(client, tmp_path / "cache", tmp_path / "summary", max_cost=10.0, sleep=lambda _: None)
    runner.execute([request(1, "phase1")])
    assert len(client.timeseries.calls) == 1
    with pytest.raises(RuntimeError, match="exceeds.*cap"):
        runner.execute([request(2, "phase2")])
    assert len(client.timeseries.calls) == 1


def test_resume_is_idempotent_and_corrupt_cache_is_replaced(tmp_path):
    client = FakeClient(cost=0.25)
    runner = audit.AcquisitionRunner(client, tmp_path / "cache", tmp_path / "summary", max_cost=10.0, sleep=lambda _: None)
    req = request()
    runner.execute([req])
    runner.execute([req])
    assert len(client.timeseries.calls) == 1
    _, parquet = runner._paths(req)
    parquet.write_bytes(b"corrupt")
    runner.execute([req])
    assert len(client.timeseries.calls) == 2
    assert runner.verify()["verification_passed"] is True


def test_parent_definitions_use_supported_instrument_id_output(tmp_path):
    client = FakeClient(cost=0.1)
    runner = audit.AcquisitionRunner(client, tmp_path / "cache", tmp_path / "summary", max_cost=1.0, sleep=lambda _: None)
    parent = audit.DataRequest(
        phase=1,
        purpose="discover_gap_definition",
        dataset="OPRA.PILLAR",
        schema="definition",
        start="2020-01-02T00:00:00",
        end="2020-01-03T00:00:00",
        symbols=("AAPL.OPT",),
        stype_in="parent",
    )
    runner.execute([parent])
    assert client.timeseries.calls[0]["stype_out"] == "instrument_id"


def test_repository_manifest_counts_and_excludes_corporate_actions():
    inputs = audit.load_inputs()
    assert len(inputs.candidate_rows) == 24_831
    assert len(inputs.stale_rows) == 2_412
    definitions = audit.build_definition_requests(inputs)
    known = audit.build_known_selection_requests(inputs)
    execution = audit.build_execution_requests(inputs, pd.DataFrame())
    phase3, gaps = audit.build_phase3_requests(inputs, pd.DataFrame())
    all_requests = definitions + known + execution + phase3
    assert len([r for r in phase3 if r.purpose == "vx_decision_close"]) == 84
    assert len([g for g in gaps if g["kind"] == "vx_front"]) == 9
    assert not any("corporate" in r.dataset.lower() or "assignment" in r.purpose.lower() for r in all_requests)


def test_ledger_never_contains_the_api_key(tmp_path):
    client = FakeClient(cost=0.1)
    runner = audit.AcquisitionRunner(client, tmp_path / "cache", tmp_path / "summary", max_cost=1.0, sleep=lambda _: None)
    runner.execute([request()])
    assert "a" * 32 not in runner.ledger_path.read_text()


def test_failed_download_records_sanitized_account_error(tmp_path):
    client = FakeClient(cost=0.1)
    client.timeseries = FailingTimeseries()
    runner = audit.AcquisitionRunner(client, tmp_path / "cache", tmp_path / "summary", max_cost=1.0, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="account_insufficient_funds"):
        runner.execute([request()])
    entry = runner.ledger[request().request_id]
    assert entry["status"] == "failed"
    assert entry["error_code"] == "account_insufficient_funds"
