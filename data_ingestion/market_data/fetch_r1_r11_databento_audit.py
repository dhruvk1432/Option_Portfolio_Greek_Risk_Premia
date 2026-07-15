"""Cost-capped Databento acquisition for the R1/R1.1 execution audit.

The CLI is deliberately separate from the repository's older Databento jobs:
it reads only ``DATABENTO_API_KEY2`` from the project ``.env`` and passes that
value explicitly to the client. Licensed symbol manifests and market records
stay under the gitignored cache. Only aggregate, nonlicensed audit summaries
are written under the paper artifacts directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from dotenv import dotenv_values
import exchange_calendars as xcals

from data_ingestion.build_cbbo_cost_surface import assign_moneyness_bucket, parse_osi_symbol


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "research/papers/option_only_markowitz/analysis/artifacts"
R11_WEIGHTS = ARTIFACT_ROOT / "r11_higher_risk/r11_monthly_weights.csv"
R1_WEIGHTS = ARTIFACT_ROOT / "r1_repaired/r1_monthly_weights.csv"
EVENT_REQUEST = ARTIFACT_ROOT / "r11_higher_risk/r11_event_quote_request.csv"
FEATURE_PANEL = ROOT / "data/feature_store/option_greek_proxy_panel.parquet"
RAW_CLOSE = ROOT / "data/universe/multi_raw_close.csv"
VX_DAILY = ROOT / "data/universe/vx_futures_daily.parquet"
VIX_HOLDING_DETAIL = ROOT / "research/papers/option_only_markowitz/artifacts/vix_holding_return_detail.csv"
DEFAULT_CACHE = ROOT / "data/databento_cache/r1_r11_audit"
DEFAULT_SUMMARY = ARTIFACT_ROOT / "databento_audit"

R11_BASE = "R1.1 25pct positive-edge deployment"
CMBP_START = pd.Timestamp("2023-03-28")
XCBF_START = pd.Timestamp("2018-11-04")
EQUITY_FEED_START = pd.Timestamp("2018-05-01")
DEFAULT_MAX_COST = 40.0
CREDENTIAL_VARIABLE = "DATABENTO_API_KEY2"
AUDIT_SCOPE = "entry_exit_intervention_quotes_only"

NASDAQ = {
    "AAL", "AAPL", "ADBE", "AMAT", "AMD", "AMZN", "AVGO", "CHTR", "CMCSA",
    "COST", "CSCO", "GILD", "GOOG", "GOOGL", "INTC", "LRCX", "META", "MSFT",
    "MU", "NFLX", "NVDA", "PYPL", "QCOM", "SBUX", "TSLA", "TXN",
}
NYSE = {
    "BA", "BAC", "C", "CCL", "CRM", "CVX", "DAL", "DIS", "GE", "GS", "HD",
    "JNJ", "JPM", "KO", "LLY", "MA", "MRK", "NKE", "ORCL", "PFE", "PG", "T",
    "UAL", "UNH", "V", "VZ", "WFC", "WMT", "XOM",
}
VX_MONTH_CODES = dict(zip("FGHJKMNQUVXZ", range(1, 13)))


@dataclass(frozen=True)
class DataRequest:
    phase: int
    purpose: str
    dataset: str
    schema: str
    start: str
    end: str
    symbols: tuple[str, ...]
    stype_in: str = "raw_symbol"

    def normalized(self) -> dict[str, object]:
        row = asdict(self)
        row["symbols"] = sorted(set(map(str, self.symbols)))
        return row

    @property
    def request_id(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def api_args(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "schema": self.schema,
            "start": self.start,
            "end": self.end,
            "symbols": list(sorted(set(self.symbols))),
            "stype_in": self.stype_in,
        }


@dataclass(frozen=True)
class AcquisitionInputs:
    candidate_rows: pd.DataFrame
    stale_rows: pd.DataFrame
    r11_active: pd.DataFrame
    r1_active: pd.DataFrame
    event_rows: pd.DataFrame


def load_primary_key(env_file: Path = ROOT / ".env") -> str:
    """Return only the explicitly selected project key without fallback."""

    value = dotenv_values(env_file).get(CREDENTIAL_VARIABLE)
    key = str(value).strip() if value is not None else ""
    if len(key) != 32 or not key.isascii() or any(ch.isspace() for ch in key):
        raise RuntimeError(f"project .env {CREDENTIAL_VARIABLE} is missing or malformed")
    return key


def make_client(env_file: Path = ROOT / ".env"):
    import databento as db

    return db.Historical(load_primary_key(env_file))


def credential_fingerprint(env_file: Path = ROOT / ".env") -> str:
    return hashlib.sha256(load_primary_key(env_file).encode()).hexdigest()[:12]


def _is_raw_osi(symbol: object) -> bool:
    try:
        parse_osi_symbol(str(symbol))
        return True
    except ValueError:
        return False


def load_inputs() -> AcquisitionInputs:
    r11 = pd.read_csv(R11_WEIGHTS)
    base = r11[r11["strategy"].eq(R11_BASE)].copy()
    base["decision_date"] = pd.to_datetime(base["decision_date"])
    base["expiry"] = pd.to_datetime(base["expiry"], errors="coerce")
    base["known_osi"] = base["symbol"].map(_is_raw_osi) & base["expiry"].notna()
    candidates = base.drop_duplicates(["decision_date", "asset_id"]).copy()
    stale = candidates[~candidates["known_osi"]].copy()
    r11_active = base[base["integer_contracts"].abs().gt(1e-12)].copy()

    r1 = pd.read_csv(R1_WEIGHTS)
    r1["decision_date"] = pd.to_datetime(r1["decision_date"])
    r1 = r1[r1["weight"].abs().gt(1e-12)].copy()
    lookup = base[
        ["config", "decision_date", "asset_id", "symbol", "underlying", "expiry", "known_osi"]
    ].drop_duplicates(["config", "decision_date", "asset_id"])
    r1_active = r1.merge(lookup, on=["config", "decision_date", "asset_id"], how="left")

    events = pd.read_csv(EVENT_REQUEST)
    events["execution_date"] = pd.to_datetime(events["execution_date"])
    return AcquisitionInputs(candidates, stale, r11_active, r1_active, events)


def _calendar():
    return xcals.get_calendar("XNYS")


def _session_for_day(value: pd.Timestamp, direction: str = "previous") -> pd.Timestamp:
    return _calendar().date_to_session(pd.Timestamp(value), direction=direction)


def _next_session(value: pd.Timestamp) -> pd.Timestamp:
    cal = _calendar()
    session = cal.date_to_session(pd.Timestamp(value), direction="next")
    if session.date() == pd.Timestamp(value).date():
        session = cal.next_session(session)
    return session


def close_window(value: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    session = _session_for_day(value, "previous")
    close = _calendar().session_close(session)
    return close - pd.Timedelta(minutes=10), close


def open_window(value: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    session = _next_session(value)
    opened = _calendar().session_open(session)
    return opened, opened + pd.Timedelta(minutes=10)


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).isoformat()


def _request(
    phase: int,
    purpose: str,
    dataset: str,
    schema: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    symbols: Iterable[str],
    stype_in: str = "raw_symbol",
) -> DataRequest:
    clean = tuple(sorted(set(str(s) for s in symbols if str(s))))
    if not clean:
        raise ValueError(f"{purpose} request has no symbols")
    return DataRequest(phase, purpose, dataset, schema, _iso(pd.Timestamp(start)), _iso(pd.Timestamp(end)), clean, stype_in)


def _option_root(underlying: str, decision: pd.Timestamp) -> str:
    if underlying in {"VX_FRONT", "VIX"}:
        return "VIX"
    if underlying == "META" and pd.Timestamp(decision) < pd.Timestamp("2022-06-09"):
        return "FB"
    return underlying


def _option_parent(underlying: str, decision: pd.Timestamp) -> str:
    return f"{_option_root(underlying, decision)}.OPT"


def build_definition_requests(inputs: AcquisitionInputs) -> list[DataRequest]:
    requests: list[DataRequest] = []
    known = inputs.candidate_rows[inputs.candidate_rows["known_osi"]]
    for decision, group in known.groupby("decision_date", observed=True):
        start, _ = close_window(decision)
        requests.append(_request(1, "verify_known_definition", "OPRA.PILLAR", "definition", start.normalize(), start.normalize() + pd.Timedelta(days=1), group["symbol"]))
    for decision, group in inputs.stale_rows.groupby("decision_date", observed=True):
        start, _ = close_window(decision)
        parents = [_option_parent(x, decision) for x in group["underlying"].dropna().astype(str).unique()]
        requests.append(_request(1, "discover_gap_definition", "OPRA.PILLAR", "definition", start.normalize(), start.normalize() + pd.Timedelta(days=1), parents, "parent"))
    return requests


def _schema_for_option_events(decision: pd.Timestamp) -> str:
    stamp = pd.Timestamp(decision)
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return "cmbp-1" if stamp.normalize() >= CMBP_START else "cbbo-1m"


def build_known_selection_requests(inputs: AcquisitionInputs) -> list[DataRequest]:
    requests: list[DataRequest] = []
    known = inputs.candidate_rows[inputs.candidate_rows["known_osi"]]
    for decision, group in known.groupby("decision_date", observed=True):
        start, end = close_window(decision)
        symbols = group["symbol"]
        requests.append(_request(2, "candidate_close_quotes", "OPRA.PILLAR", _schema_for_option_events(decision), start, end, symbols))
        requests.append(_request(3, "candidate_daily_volume", "OPRA.PILLAR", "ohlcv-1d", start.normalize(), start.normalize() + pd.Timedelta(days=1), symbols))
        requests.append(_request(3, "candidate_open_interest", "OPRA.PILLAR", "statistics", start.normalize(), start.normalize() + pd.Timedelta(days=1), symbols))
    return requests


def _spot_lookup() -> dict[tuple[pd.Timestamp, str], float]:
    out: dict[tuple[pd.Timestamp, str], float] = {}
    panel = pd.read_parquet(FEATURE_PANEL, columns=["snap_date", "underlying", "spot"])
    panel["snap_date"] = pd.to_datetime(panel["snap_date"]).dt.normalize()
    spots = panel.groupby(["snap_date", "underlying"], observed=True)["spot"].median()
    for (dt, symbol), value in spots.items():
        if np.isfinite(value) and value > 0:
            out[(pd.Timestamp(dt), str(symbol))] = float(value)
    if RAW_CLOSE.exists():
        raw = pd.read_csv(RAW_CLOSE, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index).normalize()
        for dt in raw.index:
            for symbol, value in raw.loc[dt].dropna().items():
                if np.isfinite(value) and value > 0:
                    out.setdefault((pd.Timestamp(dt), str(symbol)), float(value))
    if VIX_HOLDING_DETAIL.exists():
        vix = pd.read_csv(VIX_HOLDING_DETAIL, usecols=["decision_date", "start_spot"])
        vix["decision_date"] = pd.to_datetime(vix["decision_date"]).dt.normalize()
        forward = vix.groupby("decision_date", observed=True)["start_spot"].median()
        for dt, value in forward.items():
            if np.isfinite(value) and value > 0:
                out[(pd.Timestamp(dt), "VX_FRONT")] = float(value)
    return out


def _vix_moneyness_bucket(spot: float, strike: float, kind: str) -> str:
    log_moneyness = float(np.log(float(strike) / float(spot)))
    if abs(log_moneyness) <= 0.05:
        return "vix_atm"
    if kind == "call" and 0.05 < log_moneyness <= 0.25:
        return "vix_call_near"
    if kind == "call" and 0.25 < log_moneyness <= 0.75:
        return "vix_call_wing"
    if kind == "put" and -0.35 <= log_moneyness < -0.05:
        return "vix_put_near"
    return "other"


def _symbol_column(frame: pd.DataFrame) -> str | None:
    for name in ("symbol", "raw_symbol"):
        if name in frame.columns:
            return name
    return None


def candidate_symbols_from_definitions(
    inputs: AcquisitionInputs,
    definition_frames: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter gap definitions to contracts in each requested bucket.

    This only creates the Phase 2 selection universe. It never labels a
    contract resolved until quote, volume, and spread eligibility are known.
    """

    symbols: set[str] = set()
    for frame in definition_frames:
        col = _symbol_column(frame)
        if col:
            symbols.update(str(x) for x in frame[col].dropna().unique() if _is_raw_osi(x))
    parsed: list[dict[str, object]] = []
    for symbol in sorted(symbols):
        root, expiry, kind, strike = parse_osi_symbol(symbol)
        parsed.append({"symbol": symbol, "root": root, "expiry": expiry, "kind": "call" if kind == "C" else "put", "strike": strike})
    definitions = pd.DataFrame(parsed, columns=["symbol", "root", "expiry", "kind", "strike"])
    spots = _spot_lookup()
    rows: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    for stale in inputs.stale_rows.itertuples(index=False):
        decision = pd.Timestamp(stale.decision_date).normalize()
        underlying = str(stale.underlying)
        spot = spots.get((decision, underlying))
        if not spot:
            status.append({"decision_date": decision, "asset_id": stale.asset_id, "resolution_status": "unresolved_missing_spot"})
            continue
        try:
            _, requested_kind, bucket = str(stale.asset_id).split("_", 2)
        except ValueError:
            status.append({"decision_date": decision, "asset_id": stale.asset_id, "resolution_status": "unresolved_bad_asset_id"})
            continue
        root = _option_root(underlying, decision)
        subset = definitions[(definitions["root"].eq(root)) & (definitions["kind"].eq(requested_kind))].copy()
        if underlying == "VX_FRONT":
            dte_low, dte_high = 7, 45
        else:
            dte_low, dte_high = 15, 21
        subset["dte"] = (subset["expiry"] - decision).dt.days
        subset = subset[subset["dte"].between(dte_low, dte_high)]
        if underlying == "VX_FRONT":
            subset["bucket"] = [_vix_moneyness_bucket(spot, strike, kind) for strike, kind in zip(subset["strike"], subset["kind"])]
        else:
            subset["bucket"] = [assign_moneyness_bucket(spot, strike, kind) for strike, kind in zip(subset["strike"], subset["kind"])]
        subset = subset[subset["bucket"].eq(bucket)]
        if subset.empty:
            status.append({"decision_date": decision, "asset_id": stale.asset_id, "resolution_status": "unresolved_no_definition_candidate"})
            continue
        for item in subset.itertuples(index=False):
            rows.append({
                "decision_date": decision,
                "asset_id": stale.asset_id,
                "underlying": underlying,
                "symbol": item.symbol,
                "expiry": item.expiry,
                "kind": item.kind,
                "bucket": bucket,
                "strike": item.strike,
                "spot": spot,
            })
        status.append({"decision_date": decision, "asset_id": stale.asset_id, "resolution_status": "awaiting_phase2_quote_and_volume"})
    return pd.DataFrame(rows), pd.DataFrame(status)


def build_gap_selection_requests(candidates: pd.DataFrame) -> list[DataRequest]:
    requests: list[DataRequest] = []
    if candidates.empty:
        return requests
    for decision, group in candidates.groupby("decision_date", observed=True):
        start, end = close_window(pd.Timestamp(decision))
        symbols = group["symbol"]
        requests.append(_request(2, "gap_candidate_close_quotes", "OPRA.PILLAR", _schema_for_option_events(decision), start, end, symbols))
        requests.append(_request(3, "gap_candidate_daily_volume", "OPRA.PILLAR", "ohlcv-1d", start.normalize(), start.normalize() + pd.Timedelta(days=1), symbols))
        requests.append(_request(3, "gap_candidate_open_interest", "OPRA.PILLAR", "statistics", start.normalize(), start.normalize() + pd.Timedelta(days=1), symbols))
    return requests


def _normalize_quote_metrics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in frames:
        col = _symbol_column(frame)
        if col is None or not {"bid_px_00", "ask_px_00"}.issubset(frame.columns):
            continue
        q = frame.copy()
        q["symbol"] = q[col].astype(str)
        bid = pd.to_numeric(q["bid_px_00"], errors="coerce")
        ask = pd.to_numeric(q["ask_px_00"], errors="coerce")
        mid = (bid + ask) / 2.0
        q["mid"] = mid
        q["relative_spread"] = (ask - bid) / mid.replace(0.0, np.nan)
        q = q[(bid.gt(0)) & (ask.ge(bid)) & np.isfinite(q["relative_spread"])]
        if len(q):
            rows.append(q[["symbol", "mid", "relative_spread"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "mark", "median_relative_spread"])
    all_quotes = pd.concat(rows, ignore_index=True)
    return all_quotes.groupby("symbol", observed=True).agg(mark=("mid", "last"), median_relative_spread=("relative_spread", "median")).reset_index()


def _normalize_volume(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in frames:
        col = _symbol_column(frame)
        if col is None or "volume" not in frame:
            continue
        part = frame[[col, "volume"]].copy().rename(columns={col: "symbol"})
        part["symbol"] = part["symbol"].astype(str)
        part["volume"] = pd.to_numeric(part["volume"], errors="coerce")
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["symbol", "volume"])
    return pd.concat(rows, ignore_index=True).groupby("symbol", observed=True)["volume"].max().reset_index()


def resolve_gap_contracts(
    candidates: pd.DataFrame,
    quote_frames: Sequence[pd.DataFrame],
    volume_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=list(candidates.columns) + ["mark", "median_relative_spread", "volume", "resolution_status"])
    quotes = _normalize_quote_metrics(quote_frames)
    volumes = _normalize_volume(volume_frames)
    merged = candidates.merge(quotes, on="symbol", how="left").merge(volumes, on="symbol", how="left")
    eligible = merged[
        merged["mark"].ge(0.25)
        & merged["volume"].ge(10)
        & merged["median_relative_spread"].le(0.20)
    ].copy()
    eligible = eligible.sort_values(
        ["decision_date", "asset_id", "volume", "median_relative_spread", "symbol"],
        ascending=[True, True, False, True, True],
    )
    selected = eligible.groupby(["decision_date", "asset_id"], observed=True, as_index=False).head(1).copy()
    selected["resolution_status"] = "resolved_exact_osi"
    return selected


def _resolved_active(inputs: AcquisitionInputs, resolved: pd.DataFrame) -> pd.DataFrame:
    key = ["decision_date", "asset_id"]
    if resolved.empty:
        selected = pd.DataFrame(columns=key + ["symbol", "underlying", "expiry"])
    else:
        selected = resolved[key + ["symbol", "underlying", "expiry"]].drop_duplicates(key)
    out: list[pd.DataFrame] = []
    for frame, source in ((inputs.r11_active, "r11"), (inputs.r1_active, "r1")):
        work = frame.copy()
        work["decision_date"] = pd.to_datetime(work["decision_date"])
        work = work.merge(selected, on=key, how="left", suffixes=("", "_resolved"))
        for col in ("symbol", "underlying", "expiry"):
            replacement = f"{col}_resolved"
            if replacement in work:
                work[col] = work[replacement].where(work[replacement].notna(), work[col])
        work = work[work["symbol"].map(_is_raw_osi) & pd.to_datetime(work["expiry"], errors="coerce").notna()].copy()
        work["source_strategy"] = source
        out.append(work)
    combined = pd.concat(out, ignore_index=True)
    combined["expiry"] = pd.to_datetime(combined["expiry"])
    return combined.drop_duplicates(["decision_date", "symbol"])


def build_execution_requests(inputs: AcquisitionInputs, resolved: pd.DataFrame) -> list[DataRequest]:
    requests: list[DataRequest] = []
    held = _resolved_active(inputs, resolved)
    for decision, group in held.groupby("decision_date", observed=True):
        start, end = open_window(pd.Timestamp(decision))
        requests.append(_request(2, "held_next_open", "OPRA.PILLAR", _schema_for_option_events(start), start, end, group["symbol"]))
        for expiry, cohort in group.groupby("expiry", observed=True):
            expiry_session = _session_for_day(pd.Timestamp(expiry), "previous")
            exit_start = _calendar().session_close(expiry_session) - pd.Timedelta(minutes=10)
            exit_end = _calendar().session_close(expiry_session)
            requests.append(_request(2, "held_exit_close", "OPRA.PILLAR", _schema_for_option_events(exit_start), exit_start, exit_end, cohort["symbol"]))
    for execution, group in inputs.event_rows.groupby("execution_date", observed=True):
        session = _session_for_day(pd.Timestamp(execution), "next")
        start = _calendar().session_open(session)
        end = _calendar().session_close(session)
        requests.append(_request(2, "vix_intervention_cbbo", "OPRA.PILLAR", "cbbo-1m", start, end, group["symbol"]))
    return requests


def _primary_dataset(symbol: str) -> str:
    if symbol in NASDAQ:
        return "XNAS.ITCH"
    if symbol in NYSE:
        return "XNYS.PILLAR"
    raise ValueError(f"no single primary-venue mapping for {symbol}")


def _vx_raw_symbol(value: str) -> str:
    text = str(value).strip().upper()
    match = re.fullmatch(r"VX([FGHJKMNQUVXZ])(\d{1,2})", text)
    if not match:
        raise ValueError(f"unsupported VX artifact symbol {value!r}")
    code, year = match.groups()
    return f"VX/{code}{int(year) % 10}"


def build_phase3_requests(inputs: AcquisitionInputs, resolved: pd.DataFrame) -> tuple[list[DataRequest], list[dict[str, object]]]:
    return [], []


def _dedupe_requests(requests: Iterable[DataRequest]) -> list[DataRequest]:
    by_id = {request.request_id: request for request in requests}
    return sorted(by_id.values(), key=lambda x: (x.phase, x.purpose, x.start, x.request_id))


def _read_dbn_frame(path: Path) -> tuple[pd.DataFrame, str | None]:
    """Read every complete record from a possibly truncated DBN stream."""

    import databento as db

    raw = db.DBNStore.from_file(path).to_df(map_symbols=True)
    time_column = str(raw.index.name) if raw.index.name is not None else None
    return raw.reset_index(), time_column


class AcquisitionRunner:
    def __init__(
        self,
        client,
        cache_root: Path = DEFAULT_CACHE,
        summary_root: Path = DEFAULT_SUMMARY,
        max_cost: float = DEFAULT_MAX_COST,
        sleep: Callable[[float], None] = time.sleep,
        credential_id: str = "test",
    ) -> None:
        if not np.isfinite(max_cost) or max_cost <= 0:
            raise ValueError("max_cost must be positive")
        self.client = client
        self.cache_root = Path(cache_root)
        self.summary_root = Path(summary_root)
        self.max_cost = float(max_cost)
        self.sleep = sleep
        self.credential_id = str(credential_id)
        self.ledger_path = self.cache_root / "request_ledger.json"
        self.ledger = self._load_ledger()

    def _load_ledger(self) -> dict[str, dict[str, object]]:
        if not self.ledger_path.exists():
            return {}
        try:
            raw = json.loads(self.ledger_path.read_text())
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_ledger(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(self.ledger, indent=2, sort_keys=True) + "\n")

    def _paths(self, request: DataRequest) -> tuple[Path, Path]:
        directory = self.cache_root / f"phase{request.phase}" / request.purpose
        return directory / f"{request.request_id}.dbn.zst", directory / f"{request.request_id}.parquet"

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _valid_cached(self, request: DataRequest) -> bool:
        entry = self.ledger.get(request.request_id, {})
        _, parquet = self._paths(request)
        if entry.get("status") != "complete" or not parquet.exists():
            return False
        return entry.get("parquet_sha256") == self._sha256(parquet)

    def _retry(self, func: Callable[[], object], attempts: int = 6):
        error: Exception | None = None
        for attempt in range(attempts):
            try:
                return func(), attempt
            except Exception as exc:  # Databento exposes several transport exception types.
                error = exc
                if isinstance(exc, FileExistsError):
                    raise
                if "422" in str(exc) or "symbology_invalid_request" in str(exc):
                    raise
                if attempt + 1 < attempts:
                    self.sleep(min(2 ** attempt, 15))
        assert error is not None
        raise error

    def estimate(self, requests: Sequence[DataRequest]) -> float:
        pending = _dedupe_requests(requests)
        incomplete = [request for request in pending if not self._valid_cached(request)]

        def has_current_estimate(request: DataRequest) -> bool:
            entry = self.ledger.get(request.request_id, {})
            value = entry.get("estimated_cost")
            return (
                entry.get("request") == request.normalized()
                and entry.get("credential_id") == self.credential_id
                and isinstance(value, (int, float))
                and np.isfinite(value)
            )

        total = float(sum(float(self.ledger[r.request_id]["estimated_cost"]) for r in incomplete if has_current_estimate(r)))
        uncached = [request for request in incomplete if not has_current_estimate(request)]

        def estimate_one(request: DataRequest) -> tuple[DataRequest, float, int]:
            value, retries = self._retry(lambda: self.client.metadata.get_cost(**request.api_args()))
            return request, float(value), retries

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(estimate_one, request): request for request in uncached}
            for number, future in enumerate(as_completed(futures), start=1):
                request, estimate, retries = future.result()
                entry = self.ledger.setdefault(request.request_id, {})
                entry.update({
                    "request": request.normalized(),
                    "estimated_cost": estimate,
                    "metadata_retries": retries,
                    "credential_id": self.credential_id,
                    "status": entry.get("status", "estimated"),
                })
                total += estimate
                if number == 1 or number % 25 == 0 or number == len(uncached):
                    self._write_ledger()
                    print(f"cost preflight {number}/{len(uncached)}: ${total:.6f} uncompleted", flush=True)
        self._write_ledger()
        return total

    def completed_estimated_cost(self) -> float:
        return float(sum(float(x.get("estimated_cost", 0.0)) for x in self.ledger.values() if x.get("status") == "complete"))

    def enforce_cap(self, requests: Sequence[DataRequest]) -> float:
        remaining = self.estimate(requests)
        projected = self.completed_estimated_cost() + remaining
        if projected > self.max_cost + 1e-12:
            raise RuntimeError(f"projected Databento cost ${projected:.6f} exceeds ${self.max_cost:.2f} cap")
        return projected

    def _prepare_download_resume(
        self,
        request: DataRequest,
        dbn: Path,
        tmp: Path,
    ) -> tuple[dict[str, object], list[pd.DataFrame], list[Path], pd.Timestamp | None]:
        partial_frames: list[pd.DataFrame] = []
        partial_paths = sorted(dbn.parent.glob(f"{request.request_id}.part-*.dbn.zst"))
        if tmp.exists():
            try:
                _read_dbn_frame(tmp)
            except Exception:
                tmp.unlink()
            else:
                part = dbn.parent / f"{request.request_id}.part-{len(partial_paths) + 1:03d}.dbn.zst"
                tmp.replace(part)
                partial_paths.append(part)

        resume_start: pd.Timestamp | None = None
        for part in partial_paths:
            recovered, time_column = _read_dbn_frame(part)
            if recovered.empty or time_column is None or time_column not in recovered:
                continue
            timestamps = pd.to_datetime(recovered[time_column], utc=True, errors="coerce")
            if timestamps.notna().any():
                last = timestamps.max()
                resume_start = last if resume_start is None else max(resume_start, last)
                partial_frames.append(recovered)

        api_args = request.api_args()
        if resume_start is not None:
            request_start = pd.Timestamp(request.start)
            request_end = pd.Timestamp(request.end)
            if request_start.tzinfo is None:
                request_start = request_start.tz_localize("UTC")
            if request_end.tzinfo is None:
                request_end = request_end.tz_localize("UTC")
            if request_start <= resume_start < request_end:
                # Databento start timestamps are inclusive. Re-requesting the
                # final complete timestamp prevents a boundary-record gap;
                # exact duplicate records are removed after concatenation.
                api_args["start"] = resume_start.isoformat()
            else:
                resume_start = None
                partial_frames = []
        return api_args, partial_frames, partial_paths, resume_start

    def execute(self, requests: Sequence[DataRequest]) -> None:
        self.enforce_cap(requests)
        pending = _dedupe_requests(requests)
        for number, request in enumerate(pending, start=1):
            if self._valid_cached(request):
                continue
            dbn, parquet = self._paths(request)
            parquet.parent.mkdir(parents=True, exist_ok=True)
            tmp = dbn.with_suffix(dbn.suffix + ".tmp")
            stype_out = "instrument_id"
            store = None
            retries = 0
            partial_frames: list[pd.DataFrame] = []
            partial_paths: list[Path] = []
            resume_start: pd.Timestamp | None = None
            error: Exception | None = None
            for attempt in range(6):
                api_args, partial_frames, partial_paths, resume_start = self._prepare_download_resume(request, dbn, tmp)
                try:
                    store = self.client.timeseries.get_range(**api_args, stype_out=stype_out, path=tmp)
                    retries = attempt
                    break
                except Exception as exc:  # Databento exposes several transport exception types.
                    error = exc
                    message = str(exc)
                    if "422" in message or "symbology_invalid_request" in message:
                        break
                    if attempt + 1 < 6:
                        self.sleep(min(2 ** attempt, 15))
            if store is None:
                entry = self.ledger.setdefault(request.request_id, {})
                assert error is not None
                message = str(error)
                error_code = "account_insufficient_funds" if "account_insufficient_funds" in message else type(error).__name__
                entry.update({
                    "request": request.normalized(),
                    "status": "failed",
                    "error_code": error_code,
                })
                self._write_ledger()
                raise error
            frame = store.to_df(map_symbols=True).reset_index()
            partial_rows = int(sum(len(part) for part in partial_frames))
            if partial_frames:
                frame = pd.concat([*partial_frames, frame], ignore_index=True).drop_duplicates(ignore_index=True)
            if tmp.exists():
                tmp.replace(dbn)
            elif not dbn.exists():
                store.to_file(dbn, compression="zstd")
            frame.to_parquet(parquet, index=False)
            entry = self.ledger.setdefault(request.request_id, {})
            entry.update({
                "request": request.normalized(),
                "status": "complete",
                "download_retries": retries,
                "rows": int(len(frame)),
                "dbn_sha256": self._sha256(dbn),
                "dbn_parts": [
                    {
                        "path": str(path.relative_to(self.cache_root)),
                        "sha256": self._sha256(path),
                    }
                    for path in [*partial_paths, dbn]
                ],
                "parquet_sha256": self._sha256(parquet),
                "columns": list(map(str, frame.columns)),
                "resumed_partial_rows": partial_rows,
                "resumed_from_timestamp": resume_start.isoformat() if resume_start is not None else None,
            })
            self._write_ledger()
            if number == 1 or number % 10 == 0 or number == len(pending):
                print(f"download {number}/{len(pending)}: {request.purpose}, {len(frame):,} rows", flush=True)

    def frames(self, purposes: set[str]) -> list[pd.DataFrame]:
        out: list[pd.DataFrame] = []
        for request_id, entry in self.ledger.items():
            request = entry.get("request", {})
            if entry.get("status") != "complete" or request.get("purpose") not in purposes:
                continue
            phase = request.get("phase")
            purpose = request.get("purpose")
            path = self.cache_root / f"phase{phase}" / str(purpose) / f"{request_id}.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                if "request_start" not in frame:
                    frame["request_start"] = request.get("start")
                out.append(frame)
        return out

    def verify(self) -> dict[str, object]:
        invalid: list[str] = []
        rows = 0
        for request_id, entry in self.ledger.items():
            if entry.get("status") != "complete":
                continue
            request = entry["request"]
            path = self.cache_root / f"phase{request['phase']}" / str(request["purpose"]) / f"{request_id}.parquet"
            if not path.exists() or self._sha256(path) != entry.get("parquet_sha256"):
                invalid.append(request_id)
            rows += int(entry.get("rows", 0))
        return {
            "complete_requests": int(sum(x.get("status") == "complete" for x in self.ledger.values())),
            "verified_rows": rows,
            "invalid_cache_files": invalid,
            "verification_passed": not invalid,
        }


def _write_private_frame(cache_root: Path, name: str, frame: pd.DataFrame) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_root / f"{name}.parquet", index=False)


def _write_public_summary(
    runner: AcquisitionRunner,
    inputs: AcquisitionInputs,
    resolved: pd.DataFrame,
    resolution_status: pd.DataFrame,
    coverage_gaps: Sequence[dict[str, object]],
    pipeline_status: str,
) -> dict[str, object]:
    verification = runner.verify()
    resolved_keys = len(resolved.drop_duplicates(["decision_date", "asset_id"])) if len(resolved) else 0
    completed = [entry for entry in runner.ledger.values() if entry.get("status") == "complete"]
    failed = [entry for entry in runner.ledger.values() if entry.get("status") == "failed"]
    phase_costs: dict[str, float] = {}
    phase_requests: dict[str, int] = {}
    for entry in completed:
        phase = str(entry.get("request", {}).get("phase", "unknown"))
        phase_costs[phase] = phase_costs.get(phase, 0.0) + float(entry.get("estimated_cost", 0.0))
        phase_requests[phase] = phase_requests.get(phase, 0) + 1
    ledger_sha256 = runner._sha256(runner.ledger_path) if runner.ledger_path.exists() else ""
    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "pipeline_status": pipeline_status,
        "audit_scope": AUDIT_SCOPE,
        "max_authorized_cost_usd": runner.max_cost,
        "credential_variable": CREDENTIAL_VARIABLE,
        "credential_fingerprint_prefix": runner.credential_id,
        "completed_estimated_cost_usd": runner.completed_estimated_cost(),
        "completed_estimated_cost_by_phase_usd": phase_costs,
        "completed_requests_by_phase": phase_requests,
        "failed_requests": int(len(failed)),
        "failure_codes": sorted(set(str(entry.get("error_code", "unknown")) for entry in failed)),
        "candidate_date_contract_slots": int(len(inputs.candidate_rows)),
        "initial_stale_slots": int(len(inputs.stale_rows)),
        "resolved_stale_slots": int(resolved_keys),
        "unresolved_stale_slots": int(len(inputs.stale_rows) - resolved_keys),
        "coverage_gap_count": int(len(coverage_gaps)),
        "corporate_actions_acquired": False,
        "assignment_validation": "not_attempted_user_excluded",
        "pre_2023_option_execution_precision": "minute_observable_not_tick_exact",
        "raw_licensed_data_committed": False,
        "private_request_ledger_sha256": ledger_sha256,
        **verification,
    }
    runner.summary_root.mkdir(parents=True, exist_ok=True)
    (runner.summary_root / "databento_acquisition_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(coverage_gaps).to_csv(runner.summary_root / "databento_coverage_gaps.csv", index=False)
    status_counts = (
        resolution_status.groupby("resolution_status", dropna=False).size().rename("slots").reset_index()
        if len(resolution_status)
        else pd.DataFrame(columns=["resolution_status", "slots"])
    )
    status_counts.to_csv(runner.summary_root / "databento_resolution_summary.csv", index=False)
    return summary


def run_pipeline(command: str, max_cost: float, cache_root: Path, summary_root: Path) -> dict[str, object]:
    client = make_client()
    runner = AcquisitionRunner(client, cache_root, summary_root, max_cost, credential_id=credential_fingerprint())
    inputs = load_inputs()
    definitions = build_definition_requests(inputs)
    known_selection = build_known_selection_requests(inputs)
    known_execution = build_execution_requests(inputs, pd.DataFrame())
    base_phase3, base_coverage_gaps = build_phase3_requests(inputs, pd.DataFrame())

    if command == "plan":
        estimated = runner.estimate(definitions + known_selection + known_execution + base_phase3)
        summary = _write_public_summary(runner, inputs, pd.DataFrame(), pd.DataFrame(), base_coverage_gaps, "planned_known_requests_only")
        summary["currently_estimated_uncompleted_cost_usd"] = estimated
        (summary_root / "databento_acquisition_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    if command == "verify":
        resolved_path = cache_root / "resolved_gap_contracts.parquet"
        status_path = cache_root / "resolution_status.parquet"
        resolved = pd.read_parquet(resolved_path) if resolved_path.exists() else pd.DataFrame()
        status = pd.read_parquet(status_path) if status_path.exists() else pd.DataFrame()
        existing_gaps = summary_root / "databento_coverage_gaps.csv"
        gaps = pd.read_csv(existing_gaps).to_dict("records") if existing_gaps.exists() and existing_gaps.stat().st_size else []
        existing_summary = summary_root / "databento_acquisition_summary.json"
        prior_status = ""
        if existing_summary.exists():
            try:
                prior_status = str(json.loads(existing_summary.read_text()).get("pipeline_status", ""))
            except json.JSONDecodeError:
                prior_status = ""
        pipeline_status = prior_status if prior_status.startswith("blocked_") else "verified_existing_cache"
        return _write_public_summary(runner, inputs, resolved, status, gaps, pipeline_status)

    runner.execute(definitions)
    definition_frames = runner.frames({"verify_known_definition", "discover_gap_definition"})
    gap_candidates, resolution_status = candidate_symbols_from_definitions(inputs, definition_frames)
    _write_private_frame(cache_root, "gap_candidate_manifest", gap_candidates)
    gap_selection = build_gap_selection_requests(gap_candidates)
    runner.enforce_cap(known_selection + gap_selection + known_execution + base_phase3)
    runner.execute(known_selection + gap_selection)

    quote_frames = runner.frames({"gap_candidate_close_quotes"})
    volume_frames = runner.frames({"gap_candidate_daily_volume"})
    resolved = resolve_gap_contracts(gap_candidates, quote_frames, volume_frames)
    if len(resolved):
        _write_private_frame(cache_root, "resolved_gap_contracts", resolved)
    resolved_status = resolution_status.copy()
    if len(resolved):
        keys = set(zip(resolved["decision_date"], resolved["asset_id"]))
        mask = [
            (pd.Timestamp(dt), asset) in keys
            for dt, asset in zip(resolved_status["decision_date"], resolved_status["asset_id"])
        ]
        resolved_status.loc[mask, "resolution_status"] = "resolved_exact_osi"
    resolved_status.loc[
        resolved_status["resolution_status"].eq("awaiting_phase2_quote_and_volume"),
        "resolution_status",
    ] = "unresolved_no_eligible_phase2_contract"
    _write_private_frame(cache_root, "resolution_status", resolved_status)

    execution = build_execution_requests(inputs, resolved)
    phase3, coverage_gaps = build_phase3_requests(inputs, resolved)
    runner.execute(execution + phase3)
    summary = _write_public_summary(runner, inputs, resolved, resolved_status, coverage_gaps, "complete")
    if not summary["verification_passed"]:
        raise RuntimeError("cache verification failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "execute", "resume", "verify"])
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    try:
        summary = run_pipeline(args.command, args.max_cost, args.cache_root, args.summary_root)
    except Exception as exc:
        if "account_insufficient_funds" not in str(exc):
            raise
        runner = AcquisitionRunner(
            make_client(),
            args.cache_root,
            args.summary_root,
            args.max_cost,
            credential_id=credential_fingerprint(),
        )
        summary = _write_public_summary(
            runner,
            load_inputs(),
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            "blocked_primary_account_insufficient_funds",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
