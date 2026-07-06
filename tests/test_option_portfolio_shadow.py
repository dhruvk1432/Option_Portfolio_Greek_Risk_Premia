from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.option_portfolio_production import (  # noqa: E402
    SHADOW_FILL_MODEL,
    ShadowRunConfig,
    ShadowVerifier,
    load_shadow_quotes,
    load_shadow_targets,
    run_shadow_rebalance,
    write_shadow_outputs,
)
from src.option_portfolio_production.verification import ProductionVerifier  # noqa: E402


def _target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": ["2026-01-02T20:45:01Z"],
            "symbol": ["AAPL  260116C00100000"],
            "underlying": ["AAPL"],
            "expiry": ["2026-01-16"],
            "right": ["call"],
            "strike": [100.0],
            "target_weight": [0.01],
            "spot": [100.0],
            "underlying_price": [100.0],
            "volume": [10_000],
            "open_interest": [50_000],
            "asset_class": ["equity_option"],
            "multiplier": [100],
        }
    )


def _quote_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL  260116C00100000"],
            "bid": [1.04],
            "ask": [1.06],
            "bid_size": [100],
            "ask_size": [100],
            "ts_event": ["2026-01-02T20:45:00Z"],
            "ts_recv": ["2026-01-02T20:45:00.050Z"],
            "local_receive_ts": ["2026-01-02T20:45:00.100Z"],
            "vendor": ["shadow_csv"],
            "schema": ["nbbo_csv"],
        }
    )


def test_shadow_rebalance_writes_shadow_labeled_ledgers(tmp_path):
    target_path = tmp_path / "targets.csv"
    quote_path = tmp_path / "quotes.csv"
    _target_frame().to_csv(target_path, index=False)
    _quote_frame().to_csv(quote_path, index=False)

    targets = load_shadow_targets(target_path)
    quotes = load_shadow_quotes(quote_path)
    outputs = run_shadow_rebalance(
        targets,
        quotes,
        config=ShadowRunConfig(
            nav=100_000.0,
            decision_time=pd.Timestamp("2026-01-02T20:45:01Z"),
            out_dir=tmp_path,
        ),
    )
    fills = outputs["shadow_fill_ledger"]
    assert not fills.empty
    assert fills["fill_model"].eq(SHADOW_FILL_MODEL).all()
    assert not fills["fill_model"].str.startswith("nbbo_displayed_size").any()
    assert outputs["shadow_summary"]["not_production_certification"] is True

    write_shadow_outputs(outputs, tmp_path)
    summary = ShadowVerifier(tmp_path).run()
    assert summary["status"] == "pass"


def test_shadow_outputs_do_not_satisfy_production_verifier(tmp_path):
    target_path = tmp_path / "targets.csv"
    quote_path = tmp_path / "quotes.csv"
    _target_frame().to_csv(target_path, index=False)
    _quote_frame().to_csv(quote_path, index=False)
    outputs = run_shadow_rebalance(
        load_shadow_targets(target_path),
        load_shadow_quotes(quote_path),
        config=ShadowRunConfig(
            nav=100_000.0,
            decision_time=pd.Timestamp("2026-01-02T20:45:01Z"),
            out_dir=tmp_path,
        ),
    )
    write_shadow_outputs(outputs, tmp_path)
    prod = ProductionVerifier(ROOT / "research/papers/option_only_markowitz", tmp_path).run()
    assert prod["status"] == "fail"
    assert prod["critical_failures"] > 0


def test_breadth_e1_spread_source_policy_has_no_current_or_default_rows():
    path = (
        ROOT
        / "research/papers/option_only_markowitz/analysis/artifacts/breadth_solutions/robustness/breadth_spread_source_coverage.csv"
    )
    assert path.exists()
    frame = pd.read_csv(path)
    sources = frame["relative_spread_source"].astype(str)
    assert "current_cboe_liquid_quote" not in set(sources)
    assert "default" not in set(sources)
