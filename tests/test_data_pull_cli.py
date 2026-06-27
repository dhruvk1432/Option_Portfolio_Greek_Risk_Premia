from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_pull.config import validate_expected_inputs
from data_pull.cboe_vro_soq import (
    normalize_cboe_settlements,
    parse_cboe_index_settlement_payload,
    parse_cboe_scalar_settlement,
    required_vix_expiries,
    soq_archive_url,
    soq_component_url,
)
from data_pull.pull import JOBS, build_plan, selected_jobs


def test_option_paper_plan_contains_paid_databento_jobs() -> None:
    jobs = selected_jobs("option-paper", "")
    names = [job.name for job in jobs]
    assert "public-vro-soq" in names
    assert "databento-opra-equity" in names
    assert "databento-vix-options" in names
    assert any(job.paid for job in jobs)


def test_public_vro_soq_job_is_free_and_in_public_preset() -> None:
    jobs = selected_jobs("public", "")
    names = [job.name for job in jobs]
    assert "public-vro-soq" in names
    assert JOBS["public-vro-soq"].paid is False


def test_plan_reports_missing_paid_credentials_without_values() -> None:
    jobs = [JOBS["databento-vix-options"]]
    with patch.dict(os.environ, {"DATABENTO_API_KEY": "", "DATABENTO_API_KEY2": ""}, clear=False):
        plan = build_plan(jobs, python="python")
    assert plan[0]["env_ready"] is False
    assert "DATABENTO_API_KEY" in plan[0]["required_any_env"]
    assert "secret" not in str(plan).lower()


def test_validate_expected_inputs_is_schema_stable(tmp_path: Path) -> None:
    rows = validate_expected_inputs(tmp_path)
    assert rows
    required = {"name", "pattern", "required_for", "licensed", "exists", "n_files", "total_bytes", "example"}
    assert required <= set(rows[0])
    assert all(row["exists"] is False for row in rows)


def test_cboe_scalar_settlement_parser_selects_expiring_vx_row() -> None:
    payload = b"Product,Symbol,Expiration Date,Price\nVX,VX24/F4,2024-01-17,15.25\nVX,VX24/G4,2024-02-14,16.0\n"
    row = parse_cboe_scalar_settlement(payload, pd.Timestamp("2024-01-17"))
    assert row is not None
    assert row["settlement_value"] == 15.25
    assert row["settlement_symbol"] == "VX24/F4"


def test_soq_component_url_uses_cboe_settlement_pattern() -> None:
    url = soq_component_url(pd.Timestamp("2026-06-24"))
    assert url.endswith("/2026/06/soq_vxs_20260624.csv-dl")
    assert soq_archive_url(pd.Timestamp("2015-02-18")).endswith("/Vix_Series_02182015.xls")


def test_cboe_index_settlement_parser_extracts_historical_vro_html() -> None:
    payload = (
        b'{"data":"<h4>February 2015 Settlement Values</h4><table><tbody>'
        b'<tr><td> S&amp;P 500 (SET)</td><td>2100.00</td></tr>'
        b'<tr><td> VIX Options (VRO)</td><td>15.29</td></tr>'
        b'</tbody></table>"}'
    )
    values = parse_cboe_index_settlement_payload(payload)
    assert values[pd.Period("2015-02", freq="M")] == 15.29


def test_cboe_index_settlement_parser_preserves_exact_vro_json_date() -> None:
    payload = (
        b'{"data":[{"description":"VIX Option","trading_symbol":"VIX",'
        b'"expiration_date":"2026-06-17","settlement_symbol":"VRO",'
        b'"settlement_value":16.2}]}'
    )
    values = parse_cboe_index_settlement_payload(payload)
    assert values[pd.Timestamp("2026-06-17")] == 16.2


def test_normalized_cboe_settlements_are_vro_loader_compatible() -> None:
    rows = [
        {
            "settlement_date": pd.Timestamp("2024-01-17"),
            "expiration": pd.Timestamp("2024-01-17"),
            "product": "VIX option",
            "root": "VIX",
            "settlement_symbol": "VX24/F4",
            "settlement_value": 15.25,
            "source": "cboe_vx_final_settlement",
            "source_url": "https://example.test",
            "source_file_hash": "abc",
            "component_source_url": "https://example.test/soq",
            "component_file_hash": "def",
            "component_status": "downloaded",
        }
    ]
    frame = normalize_cboe_settlements(rows, ingested_timestamp=pd.Timestamp("2026-01-01T00:00:00Z"))
    assert {"settlement_date", "settlement_value", "component_status"} <= set(frame.columns)
    assert frame.loc[0, "root"] == "VIX"
    assert frame.loc[0, "settlement_value"] == 15.25


def test_required_vix_expiries_prefers_paper_holding_ledger(tmp_path: Path) -> None:
    art = tmp_path / "research/papers/option_only_markowitz/artifacts"
    art.mkdir(parents=True)
    pd.DataFrame({"expiry": ["2024-01-17", "2024-01-17", "2024-02-14"]}).to_csv(art / "vix_holding_return_detail.csv", index=False)
    assert required_vix_expiries(tmp_path) == [pd.Timestamp("2024-01-17"), pd.Timestamp("2024-02-14")]
