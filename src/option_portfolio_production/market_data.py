"""Executable market-data ledger and vendor reconciliation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Mapping
import hashlib

import numpy as np
import pandas as pd

from .schemas import QuoteSnapshot, utc_ts


@dataclass(frozen=True)
class QuoteReconciliationResult:
    symbol: str
    passed: bool
    reasons: tuple[str, ...]
    databento_mid: float
    broker_mid: float
    mid_diff_bps: float
    timestamp_diff_ms: float

    def ledger_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "passed": self.passed,
            "reasons": ";".join(self.reasons),
            "databento_mid": self.databento_mid,
            "broker_mid": self.broker_mid,
            "mid_diff_bps": self.mid_diff_bps,
            "timestamp_diff_ms": self.timestamp_diff_ms,
        }


def ingestion_hash(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame.sort_index(axis=1), index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_market_data_ledger(quotes: Iterable[QuoteSnapshot], *, symbol_map_version: str = "unknown") -> pd.DataFrame:
    rows = [q.ledger_row(symbol_map_version=symbol_map_version) for q in quotes]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ingestion_hash"] = ingestion_hash(out[["symbol", "ts_event", "ts_recv", "bid", "ask", "bid_size", "ask_size"]])
    return out


def validate_timestamp_monotonicity(ledger: pd.DataFrame) -> tuple[bool, list[str]]:
    if ledger.empty:
        return False, ["empty_market_data_ledger"]
    reasons: list[str] = []
    required = {"symbol", "ts_event", "ts_recv", "local_receive_timestamp"}
    missing = required - set(ledger.columns)
    if missing:
        return False, ["missing_columns:" + ",".join(sorted(missing))]
    work = ledger.copy()
    for col in ["ts_event", "ts_recv", "local_receive_timestamp"]:
        work[col] = pd.to_datetime(work[col], errors="coerce", utc=True)
        if work[col].isna().any():
            reasons.append(f"invalid_{col}")
    if ((work["ts_recv"] < work["ts_event"]) | (work["local_receive_timestamp"] < work["ts_recv"])).any():
        reasons.append("timestamp_causality_violation")
    for symbol, grp in work.sort_values(["symbol", "ts_event"]).groupby("symbol", observed=True):
        if grp["ts_event"].diff().dropna().lt(pd.Timedelta(0)).any():
            reasons.append(f"non_monotone_ts_event:{symbol}")
    return not reasons, reasons


def reconcile_quote_pair(
    databento: QuoteSnapshot,
    broker: QuoteSnapshot,
    *,
    max_mid_diff_bps: float = 50.0,
    max_timestamp_diff_ms: float = 1000.0,
) -> QuoteReconciliationResult:
    reasons: list[str] = []
    if databento.symbol != broker.symbol:
        reasons.append("symbol_mismatch")
    ref_mid = max(databento.mid, 1e-12)
    mid_diff_bps = abs(broker.mid - databento.mid) / ref_mid * 10_000.0
    timestamp_diff_ms = abs((broker.ts_event - databento.ts_event).total_seconds()) * 1000.0
    if mid_diff_bps > max_mid_diff_bps:
        reasons.append("mid_diff_too_large")
    if timestamp_diff_ms > max_timestamp_diff_ms:
        reasons.append("timestamp_diff_too_large")
    return QuoteReconciliationResult(
        symbol=databento.symbol,
        passed=not reasons,
        reasons=tuple(reasons),
        databento_mid=databento.mid,
        broker_mid=broker.mid,
        mid_diff_bps=float(mid_diff_bps),
        timestamp_diff_ms=float(timestamp_diff_ms),
    )


def executable_quote_check(
    quote: QuoteSnapshot,
    decision_time,
    *,
    max_age: timedelta = timedelta(seconds=5),
    max_spread_bps: float = 500.0,
) -> tuple[bool, list[str]]:
    return quote.executable_at(decision_time, max_age=max_age, max_spread_bps=max_spread_bps)
