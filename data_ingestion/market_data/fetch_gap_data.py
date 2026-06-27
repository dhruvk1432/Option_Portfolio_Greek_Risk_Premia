"""Gap-fill data acquisition outside WRDS.

This module adds non-WRDS sources for the blockers identified by the missing
analysis completion pass. It is deliberately conservative:

- Free/current jobs can download directly.
- Vendor-metered Databento jobs estimate cost first and only download when
  `--paid` is supplied with a positive `--max-paid-dollars` budget.
- Paid jobs can be run in manifest-only mode first. The default OPRA CBBO
  request is only a close-window per existing monthly slice, not a full day.
- The full-day OPRA CBBO request is a separate cache scope for option
  contracts already present in `data_analysis/opra_surface_panel.parquet`.
- Credential values are loaded from `.env` but never printed.

Examples
--------
python3 -m data_ingestion.market_data.fetch_gap_data --jobs free_sectors,yfinance_options
python3 -m data_ingestion.market_data.fetch_gap_data --jobs databento_opra_cbbo --underlyings AAPL,MSFT --plan-only
python3 -m data_ingestion.market_data.fetch_gap_data --jobs databento_opra_cbbo --underlyings AAPL,MSFT --paid --max-paid-dollars 20
python3 -m data_ingestion.market_data.fetch_gap_data --jobs databento_opra_surface_full_day_cbbo --max-paid-dollars 25
python3 -m data_ingestion.market_data.fetch_gap_data --jobs databento_futures_contracts --plan-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path
from typing import Any
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FREE = DATA / "free_cache"
DBC = DATA / "databento_cache"
REPORTS = ROOT / "research" / "reports" / "pipeline_reports"
UNI = DATA / "universe"
OPRA_FULL_DAY_CBBO_DIR = DBC / "opra_surface_full_day_cbbo"

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
POLYGON_BASE = "https://api.polygon.io"

DEFAULT_OPTION_UNDERLYINGS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]
DEFAULT_FUTURE_ROOTS = ["ES", "NQ", "ZN", "CL", "GC", "SI", "HG", "NG", "ZC", "6E"]
FUTURES_MONTH_CODES = list("FGHJKMNQUVXZ")
FUTURES_QUARTERLY_CODES = list("HMUZ")


class DatabentoCallTimeout(TimeoutError):
    """Raised when a vendor call exceeds the local safety timeout."""


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip().upper() for x in value.split(",") if x.strip()]


def parse_schema_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def load_environment() -> None:
    load_dotenv(ROOT / ".env")


def write_source_notes() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "free_gap_data_sources.md").write_text(
        """# Free And Non-WRDS Gap Data Sources

Evidence label: external data acquisition plan.

- Wikipedia current S&P 500 constituents table: current GICS sector and sub-industry only; not point-in-time.
- yfinance option chains: current listed-chain snapshot with bid, ask, last, volume, open interest, and implied volatility; not historical and no raw Greeks.
- Polygon/Massive option snapshots: current option chain snapshots can include quote, IV, and Greeks if the configured plan is entitled.
- Databento OPRA CBBO: historical quote-mid/spread backfill for existing monthly slice contracts, metered and cost-guarded.
- Databento OPRA full-day surface-panel CBBO: historical full-session quote-mid/spread backfill for the exact option/date keys already present in `data_analysis/opra_surface_panel.parquet`, metered and cost-guarded.
- Databento GLBX parent futures: contract-specific futures bars/definitions, metered and cost-guarded.

Historical raw vendor Greeks and non-SPY historical OPRA bid/ask are not available from truly free public sources at the required depth. They require an entitled API or paid data product.

Paid acquisition guardrails:

- The OPRA CBBO job only requests contracts already present in `data/databento_cache/opra_*_slices_*.parquet`.
- By default, OPRA CBBO requests use a narrow market-close UTC window sufficient for end-of-day surface mids/spreads; use `--opra-cbbo-full-day` only if intraday quote dynamics are explicitly needed.
- The `databento_opra_surface_full_day_cbbo` job writes to a separate full-day cache directory and is intended as the cheaper intermediate between close-window snapshots and complete parent-chain OPRA history.
- All Databento jobs write request manifests and cost logs under
  `research/reports/pipeline_reports/` before writing paid cache files.
- If Databento cost estimation fails or times out, the job records the failure and does not download the request.
""",
        encoding="utf-8",
    )


def write_paid_data_plan() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "paid_data_gap_fill_plan.md").write_text(
        """# Paid Gap-Fill Acquisition Plan

Evidence label: generated acquisition-control artifact.

The remaining historical data holes are not free-data problems:

- Non-SPY OPRA bid/ask requires historical OPRA quote data. The fetcher requests Databento `OPRA.PILLAR` `cbbo-1m` only for contracts already present in the monthly OPRA slice files and only around the market close by default.
- The cheaper intermediate for surface research is `databento_opra_surface_full_day_cbbo`: full-session Databento `OPRA.PILLAR` `cbbo-1m` for the exact option/date keys already in `data_analysis/opra_surface_panel.parquet`, written under `data/databento_cache/opra_surface_full_day_cbbo/`.
- Futures roll/carry outside VX requires contract-specific futures bars and definitions. Parent futures streams can be too broad, so the fetcher supports exact liquid-contract symbol batches for Databento `GLBX.MDP3` `ohlcv-1d`; definition files remain useful when they return, but non-VX maturities are otherwise labeled as symbol-inferred approximations.
- Historical vendor Greeks are still not downloaded as a separate product because the existing analysis computes transparent model-implied Greeks from spot, strike, tenor, call/put, and IV. If a paid vendor Greek product is later added, it should be joined through the option identifier and date keys without replacing the proxy lineage.

Recommended first paid probes:

```bash
.venv/bin/python -m data_ingestion.market_data.fetch_gap_data --jobs databento_opra_cbbo --underlyings AAPL --max-snapshots-total 1 --paid --max-paid-dollars 1 --databento-timeout-seconds 20
.venv/bin/python -m data_ingestion.market_data.fetch_gap_data --jobs databento_opra_cbbo --underlyings AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,JPM,QQQ,IWM --opra-start-date 2024-01-01 --opra-end-date 2024-01-31 --max-snapshots-total 10 --opra-instrument-batch-size 5 --paid --max-paid-dollars 1 --databento-timeout-seconds 120
.venv/bin/python -m data_ingestion.market_data.fetch_gap_data --jobs databento_opra_surface_full_day_cbbo --opra-instrument-batch-size 50 --max-paid-dollars 25 --databento-timeout-seconds 120
.venv/bin/python -m data_ingestion.market_data.fetch_gap_data --jobs databento_futures_contracts --futures-roots ES,NQ,ZN,CL,GC,SI,HG,NG,ZC,6E --futures-schemas ohlcv-1d --futures-symbol-mode liquid --futures-symbol-batch-size 3 --start-year 2024 --end-year 2024 --paid --max-paid-dollars 1 --databento-timeout-seconds 120
```

Scale only after the manifest and estimate logs show acceptable costs.
""",
        encoding="utf-8",
    )


def call_with_timeout(seconds: int, label: str, func: Any, **kwargs: Any) -> Any:
    """Run a blocking vendor call with a local alarm on platforms that support it."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return func(**kwargs)

    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise DatabentoCallTimeout(f"{label} exceeded {seconds}s")

    try:
        previous = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(seconds))
    except Exception:  # noqa: BLE001
        return func(**kwargs)
    try:
        return func(**kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def databento_child_call(kind: str, kwargs: dict[str, Any], queue: Any) -> None:
    """Run one Databento call in a child process so socket hangs are killable."""
    try:
        client = databento_client()
        if kind == "cost":
            value = float(client.metadata.get_cost(**kwargs))
            queue.put({"ok": True, "value": value})
        elif kind == "download":
            data = client.timeseries.get_range(**kwargs)
            df = data.to_df().reset_index()
            queue.put({"ok": True, "value": df})
        else:
            queue.put({"ok": False, "error": f"unknown Databento child call kind {kind!r}"})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})


def run_databento_call(kind: str, timeout_seconds: int, label: str, client: Any, **kwargs: Any) -> Any:
    if timeout_seconds <= 0:
        if kind == "cost":
            return float(client.metadata.get_cost(**kwargs))
        if kind == "download":
            return client.timeseries.get_range(**kwargs).to_df().reset_index()
        raise RuntimeError(f"unknown Databento call kind {kind!r}")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=databento_child_call, args=(kind, kwargs, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        raise DatabentoCallTimeout(f"{label} exceeded {timeout_seconds}s")
    if queue.empty():
        raise RuntimeError(f"{label} finished without returning a result")
    result = queue.get()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", f"{label} failed"))
    return result["value"]


def second_sunday(year: int, month: int = 3) -> pd.Timestamp:
    days = pd.date_range(f"{year}-{month:02d}-01", f"{year}-{month:02d}-31", freq="D")
    sundays = [d for d in days if d.weekday() == 6]
    return sundays[1]


def first_sunday(year: int, month: int = 11) -> pd.Timestamp:
    days = pd.date_range(f"{year}-{month:02d}-01", f"{year}-{month:02d}-30", freq="D")
    return next(d for d in days if d.weekday() == 6)


def is_us_equity_early_close(day: pd.Timestamp) -> bool:
    """Recognize common 1pm ET US equity/OPRA half-days.

    This intentionally covers deterministic calendar cases relevant to the
    planned month-end OPRA close-window pulls. It is not a full exchange
    holiday calendar and is recorded as a data-acquisition assumption.
    """
    d = pd.Timestamp(day).normalize()
    if d.weekday() >= 5:
        return False
    thanksgiving = [x for x in pd.date_range(f"{d.year}-11-01", f"{d.year}-11-30", freq="D") if x.weekday() == 3][3]
    if d == thanksgiving + pd.Timedelta(days=1):
        return True
    if d.month == 12 and d.day == 24:
        return True
    if d.month == 7 and d.day == 3:
        return True
    return False


def us_market_close_utc(day: pd.Timestamp, early_close: bool = False) -> pd.Timestamp:
    """Approximate US equity/OPRA close in UTC for date-only requests."""
    d = pd.Timestamp(day).normalize()
    dst = second_sunday(d.year) <= d < first_sunday(d.year)
    if early_close:
        close_hour = 17 if dst else 18
    else:
        close_hour = 20 if dst else 21
    return pd.Timestamp(f"{d.date()} {close_hour:02d}:00:00", tz="UTC")


def opra_cbbo_window(day: pd.Timestamp, args: argparse.Namespace) -> tuple[str, str, str]:
    d = pd.Timestamp(day).normalize()
    if args.opra_cbbo_full_day:
        return d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "full_day"
    close_calendar = getattr(args, "opra_close_calendar", "auto")
    early_close = close_calendar == "early" or (close_calendar == "auto" and is_us_equity_early_close(d))
    close_utc = us_market_close_utc(d, early_close=early_close)
    start = close_utc - pd.Timedelta(minutes=args.opra_cbbo_window_minutes)
    end = close_utc + pd.Timedelta(minutes=args.opra_cbbo_after_close_minutes)
    window_type = "early_close" if early_close else "regular_close"
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ"), window_type


def resolve_opra_instrument_ids(client: Any, symbols: list[str], day_ts: pd.Timestamp, args: argparse.Namespace) -> tuple[list[str], dict[str, str], list[str], str]:
    """Resolve OPRA raw symbols to Databento instrument IDs for timeseries pulls."""
    day_key = day_ts.strftime("%Y-%m-%d")
    end_key = (day_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        result = call_with_timeout(
            args.databento_timeout_seconds,
            f"symbology OPRA {day_key}",
            client.symbology.resolve,
            dataset="OPRA.PILLAR",
            symbols=symbols,
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date=day_key,
            end_date=end_key,
        )
    except Exception as exc:  # noqa: BLE001
        return [], {}, symbols, f"symbology_failed: {type(exc).__name__}: {str(exc)[:180]}"

    raw_by_instrument: dict[str, str] = {}
    for raw_symbol, intervals in (result.get("result") or {}).items():
        for interval in intervals or []:
            instrument_id = interval.get("s")
            if instrument_id is not None:
                raw_by_instrument[str(instrument_id)] = str(raw_symbol)
                break
    not_found = [str(s) for s in (result.get("not_found") or [])]
    instrument_ids = sorted(raw_by_instrument)
    status = "resolved_all" if len(instrument_ids) == len(symbols) else "resolved_partial"
    return instrument_ids, raw_by_instrument, not_found, status


def restore_raw_opra_symbol(df: pd.DataFrame, raw_by_instrument: dict[str, str]) -> pd.DataFrame:
    if df.empty or not raw_by_instrument:
        return df
    out = df.copy()
    key = None
    if "symbol" in out:
        key = out["symbol"].astype(str)
        out["databento_symbol"] = out["symbol"].astype(str)
    elif "instrument_id" in out:
        key = out["instrument_id"].astype(str)
        out["databento_symbol"] = out["instrument_id"].astype(str)
    if key is not None:
        out["symbol"] = key.map(raw_by_instrument).fillna(key)
    return out


def chunks(values: list[str], size: int) -> list[list[str]]:
    step = max(int(size), 1)
    return [values[i : i + step] for i in range(0, len(values), step)]


def liquid_futures_symbols(root: str, year: int) -> list[str]:
    """Small liquid-contract set for term structure/carry research."""
    root = root.upper()
    digit = str(year % 10)
    next_digit = str((year + 1) % 10)
    if root in {"ES", "NQ", "ZN", "6E"}:
        codes = FUTURES_QUARTERLY_CODES
        return [f"{root}{code}{digit}" for code in codes] + [f"{root}H{next_digit}"]
    if root == "ZC":
        codes = list("HKNUZ")
        return [f"{root}{code}{digit}" for code in codes] + [f"{root}H{next_digit}"]
    if root in {"CL", "GC", "SI", "HG", "NG"}:
        return [f"{root}{code}{digit}" for code in FUTURES_MONTH_CODES] + [f"{root}{code}{next_digit}" for code in "FGH"]
    return [f"{root}{code}{digit}" for code in FUTURES_QUARTERLY_CODES]


def append_rows_to_parquet(path: Path, df: pd.DataFrame, dedupe_candidates: list[str]) -> pd.DataFrame:
    """Persist newly downloaded rows immediately, preserving an idempotent cache."""
    if df.empty:
        return df
    out = df.copy()
    if path.exists():
        try:
            old = pd.read_parquet(path)
            out = pd.concat([old, out], ignore_index=True)
            dedupe_cols = [c for c in dedupe_candidates if c in old.columns and c in out.columns]
            if dedupe_cols:
                out = out.drop_duplicates(subset=dedupe_cols, keep="last")
        except Exception:  # noqa: BLE001
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def cached_opra_symbols_for_day(path: Path, day_key: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        existing = pd.read_parquet(path, columns=["snap_date", "symbol"])
        existing["snap_date"] = pd.to_datetime(existing["snap_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        existing["symbol"] = existing["symbol"].astype(str)
        return set(existing.loc[existing["snap_date"] == day_key, "symbol"].dropna().astype(str))
    except Exception:  # noqa: BLE001
        return set()


def opra_surface_contracts_from_panel(args: argparse.Namespace) -> pd.DataFrame:
    """Return unique option/date keys already present in the local surface panel."""
    panel_path = ROOT / "data_analysis" / "opra_surface_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{rel(panel_path)} is required for databento_opra_surface_full_day_cbbo. "
            "Run `python3 data_analysis/complete_missing_analysis.py --section opra` first."
        )
    cols = ["underlying", "symbol", "snap_date", "expiry", "spot"]
    panel = pd.read_parquet(panel_path, columns=cols)
    if panel.empty:
        return panel
    panel["underlying"] = panel["underlying"].astype(str).str.upper()
    underlyings = parse_list(args.underlyings, [])
    if underlyings:
        panel = panel[panel["underlying"].isin(underlyings)].copy()
    panel["snap_date"] = pd.to_datetime(panel["snap_date"], errors="coerce")
    panel["expiry"] = pd.to_datetime(panel["expiry"], errors="coerce")
    if args.opra_start_date:
        panel = panel[panel["snap_date"] >= pd.Timestamp(args.opra_start_date)]
    if args.opra_end_date:
        panel = panel[panel["snap_date"] <= pd.Timestamp(args.opra_end_date)]
    panel = panel.dropna(subset=["underlying", "symbol", "snap_date"])
    panel["symbol"] = panel["symbol"].astype(str)
    return panel.drop_duplicates(["underlying", "snap_date", "symbol"]).sort_values(["underlying", "snap_date", "symbol"])


_GLBX_END_CACHE: dict[str, pd.Timestamp] = {}


def glbx_available_end() -> pd.Timestamp:
    """Last date GLBX.MDP3 actually has data (queried once, memoized).

    Capping the request window at *today* is not enough: the dataset's available
    end lags real time (e.g. ~T-1), and Databento returns 422 dataset_unavailable_range
    if the exclusive end exceeds it. We normalize to midnight so the exclusive end
    yields whole days through the last fully-available session.
    """
    if "end" not in _GLBX_END_CACHE:
        try:
            rng = databento_client().metadata.get_dataset_range(dataset="GLBX.MDP3")
            _GLBX_END_CACHE["end"] = pd.Timestamp(rng["end"]).tz_localize(None).normalize()
        except Exception:  # noqa: BLE001 — fall back to a conservative lag
            _GLBX_END_CACHE["end"] = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    return _GLBX_END_CACHE["end"]


def annual_window_end(year: int) -> str:
    """End of the annual request window, capped at the dataset's available end.

    Databento's get_range/get_cost error on windows that extend past available data,
    so the current calendar year must stop at the live dataset end, not {year+1}-01-01.
    """
    full = pd.Timestamp(f"{year + 1}-01-01")
    return min(full, glbx_available_end()).strftime("%Y-%m-%d")


def futures_definition_request_groups(
    out_dir: Path,
    root: str,
    year: int,
    request_symbols: list[str],
    args: argparse.Namespace,
) -> list[tuple[list[str], str, str, str]]:
    """Build narrow definition windows from already cached futures bars.

    Databento GLBX definition streams can be large over full annual windows.
    For strategy carry/roll research we only need definitions for exact
    contracts already in the OHLCV cache, so anchor each request to the
    contract's first observed bar date when available.
    """
    annual = [(request_symbols, f"{year}-01-01", annual_window_end(year), "annual")]
    if not getattr(args, "futures_definition_from_bars", True) or not request_symbols:
        return annual
    if args.futures_symbol_mode != "liquid":
        return annual
    window_days = int(getattr(args, "futures_definition_window_days", 2) or 0)
    if window_days <= 0:
        return annual
    bar_path = out_dir / f"glbx_{root}_ohlcv-1d_{year}_liquid.parquet"
    if not bar_path.exists():
        return annual
    try:
        bars = pd.read_parquet(bar_path)
    except Exception:  # noqa: BLE001
        return annual
    if bars.empty or "symbol" not in bars:
        return annual
    date_col = "ts_event" if "ts_event" in bars else ("trade_date" if "trade_date" in bars else None)
    if date_col is None:
        return annual
    bars = bars[["symbol", date_col]].copy()
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars["first_date"] = pd.to_datetime(bars[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    first_dates = bars.dropna(subset=["first_date"]).groupby("symbol", observed=True)["first_date"].min()
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for symbol in request_symbols:
        first = first_dates.get(symbol.upper())
        if pd.isna(first):
            start_ts = pd.Timestamp(f"{year}-01-01")
            label = "fallback_start_of_year"
        else:
            start_ts = pd.Timestamp(first).normalize()
            label = "first_observed_bar_date"
        end_ts = start_ts + pd.Timedelta(days=window_days)
        key = (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), label)
        grouped.setdefault(key, []).append(symbol)
    return [(symbols, start, end, label) for (start, end, label), symbols in sorted(grouped.items())]


def job_free_sectors(_: argparse.Namespace) -> dict[str, Any]:
    FREE.mkdir(parents=True, exist_ok=True)
    response = requests.get(WIKI_SP500_URL, headers={"User-Agent": "OptionsPortfolioModel research bot (contact: local)"}, timeout=45)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("no tables returned from Wikipedia S&P 500 page")
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "Symbol": "ticker",
        "Security": "security",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "industry",
        "Date added": "date_added",
        "CIK": "cik",
        "Founded": "founded",
    }
    df = df.rename(columns=rename)
    if "ticker" not in df or "sector" not in df:
        raise RuntimeError(f"unexpected Wikipedia sector schema: {df.columns.tolist()}")
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.upper()
    df["source"] = WIKI_SP500_URL
    df["metadata_scope"] = "current_sp500_constituents_not_point_in_time"
    path = FREE / "sp500_current_sectors_wikipedia.csv"
    df.to_csv(path, index=False)
    coverage_path = ROOT / "data_analysis" / "sector_metadata_gap_fill.csv"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(coverage_path, index=False)
    return {"rows": len(df), "path": rel(path), "analysis_copy": rel(coverage_path)}


def option_underlyings_from_local_slices() -> list[str]:
    found = set(DEFAULT_OPTION_UNDERLYINGS)
    for path in DBC.glob("opra_*_slices_*.parquet"):
        parts = path.stem.split("_")
        if len(parts) >= 3 and parts[1].lower() != "vix":
            found.add(parts[1].upper())
    return sorted(found)


def job_yfinance_options(args: argparse.Namespace) -> dict[str, Any]:
    import yfinance as yf

    FREE.mkdir(parents=True, exist_ok=True)
    symbols = parse_list(args.underlyings, option_underlyings_from_local_slices())
    rows = []
    snap_date = pd.Timestamp.utcnow().date().isoformat()
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            expiries = list(ticker.options or [])
            spot = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("last_price")
        except Exception as exc:  # noqa: BLE001
            rows.append({"underlying": symbol, "fetch_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        for expiry in expiries[: args.max_expiries]:
            try:
                chain = ticker.option_chain(expiry)
            except Exception as exc:  # noqa: BLE001
                rows.append({"underlying": symbol, "expiry": expiry, "fetch_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
                continue
            for kind, df in [("call", chain.calls), ("put", chain.puts)]:
                if df is None or df.empty:
                    continue
                keep = [c for c in ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "inTheMoney", "lastTradeDate"] if c in df]
                part = df[keep].copy()
                part["underlying"] = symbol
                part["expiry"] = expiry
                part["kind"] = kind
                part["spot"] = spot
                part["snap_date"] = snap_date
                part["source"] = "yfinance_current_option_chain"
                part["metadata_scope"] = "current_snapshot_not_historical"
                rows.extend(part.to_dict("records"))
        time.sleep(args.sleep_seconds)
    out = pd.DataFrame(rows)
    path = FREE / f"yfinance_option_snapshot_{snap_date}.parquet"
    out.to_parquet(path, index=False)
    return {"rows": len(out), "underlyings": len(symbols), "path": rel(path)}


def flatten_polygon_result(result: dict[str, Any], underlying: str, snap_date: str) -> dict[str, Any]:
    details = result.get("details") or {}
    greeks = result.get("greeks") or {}
    quote = result.get("last_quote") or {}
    trade = result.get("last_trade") or {}
    day = result.get("day") or {}
    underlying_asset = result.get("underlying_asset") or {}
    row = {
        "underlying": underlying,
        "snap_date": snap_date,
        "ticker": details.get("ticker"),
        "contract_type": details.get("contract_type"),
        "expiry": details.get("expiration_date"),
        "strike": details.get("strike_price"),
        "open_interest": result.get("open_interest"),
        "implied_volatility": result.get("implied_volatility"),
        "break_even_price": result.get("break_even_price"),
        "underlying_price": underlying_asset.get("price"),
        "day_open": day.get("open"),
        "day_high": day.get("high"),
        "day_low": day.get("low"),
        "day_close": day.get("close"),
        "day_volume": day.get("volume"),
        "last_quote_bid": quote.get("bid"),
        "last_quote_ask": quote.get("ask"),
        "last_quote_bid_size": quote.get("bid_size"),
        "last_quote_ask_size": quote.get("ask_size"),
        "last_trade_price": trade.get("price"),
        "source": "polygon_option_chain_snapshot",
        "metadata_scope": "current_snapshot_if_plan_entitled",
    }
    for name in ["delta", "gamma", "theta", "vega"]:
        row[name] = greeks.get(name)
    return row


def polygon_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=45)
    if response.status_code >= 400:
        return {"status": "error", "status_code": response.status_code, "text": response.text[:500]}
    return response.json()


def job_polygon_options(args: argparse.Namespace) -> dict[str, Any]:
    load_environment()
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        return {"rows": 0, "status": "POLYGON_API_KEY_not_configured"}
    FREE.mkdir(parents=True, exist_ok=True)
    symbols = parse_list(args.underlyings, DEFAULT_OPTION_UNDERLYINGS)
    snap_date = pd.Timestamp.utcnow().date().isoformat()
    rows = []
    errors = []
    for symbol in symbols:
        url = f"{POLYGON_BASE}/v3/snapshot/options/{symbol}"
        params = {"apiKey": key, "limit": 250}
        pages = 0
        while url and pages < args.max_pages:
            data = polygon_get(url, params)
            params = {"apiKey": key}
            if data.get("status") == "error":
                errors.append({"underlying": symbol, **data})
                break
            for result in data.get("results") or []:
                rows.append(flatten_polygon_result(result, symbol, snap_date))
            next_url = data.get("next_url")
            url = next_url if next_url else ""
            pages += 1
            time.sleep(args.sleep_seconds)
    out = pd.DataFrame(rows)
    path = FREE / f"polygon_option_snapshot_{snap_date}.parquet"
    out.to_parquet(path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(FREE / f"polygon_option_snapshot_errors_{snap_date}.csv", index=False)
    return {"rows": len(out), "underlyings": len(symbols), "path": rel(path), "errors": len(errors)}


def job_lseg_workspace_probe(_: argparse.Namespace) -> dict[str, Any]:
    """Probe local LSEG Workspace/Eikon desktop entitlements without secrets."""
    load_environment()
    app_key = (
        os.environ.get("LSEG_APP_KEY_SIDE_BY_SIDE")
        or os.environ.get("LSEG_APP_KEY")
        or os.environ.get("LSEG_APP_KEY_EIKON")
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    FREE.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if not app_key:
        rows.append({"area": "session", "probe": "app_key", "result": "missing", "detail": "No LSEG app key environment variable found."})
        pd.DataFrame(rows).to_csv(REPORTS / "lseg_workspace_probe_results.csv", index=False)
        return {"status": "missing_app_key", "rows": len(rows)}
    try:
        import lseg.data as ld
    except Exception as exc:  # noqa: BLE001
        rows.append({"area": "session", "probe": "import lseg.data", "result": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:240]}"})
        pd.DataFrame(rows).to_csv(REPORTS / "lseg_workspace_probe_results.csv", index=False)
        return {"status": "missing_lseg_data", "rows": len(rows)}

    session_open = False
    today = pd.Timestamp.utcnow().date().isoformat()
    try:
        session = ld.open_session("desktop.workspace", app_key=app_key)
        session_open = True
        rows.append({"area": "session", "probe": "lseg.data desktop.workspace", "result": "success", "detail": f"Opened {type(session).__name__}."})

        sector_fields = [
            "TR.CommonName",
            "TR.GICSSector",
            "TR.GICSSectorCode",
            "TR.GICSIndustryGroup",
            "TR.GICSIndustry",
            "TR.GICSSubIndustry",
            "TR.TRBCIndustryGroup",
            "TR.ICBIndustry",
        ]
        sector = ld.get_data(["AAPL.O", "MSFT.O"], sector_fields)
        sector_path = FREE / f"lseg_workspace_sector_probe_{today}.csv"
        sector.to_csv(sector_path, index=False)
        rows.append({"area": "sector_metadata", "probe": "current GICS/TRBC/ICB fields", "result": "success_current_only", "detail": f"Returned {sector.shape[0]} rows and {sector.shape[1]} columns.", "path": rel(sector_path)})

        try:
            sector_hist = ld.get_data(
                ["AAPL.O", "MSFT.O"],
                sector_fields,
                parameters={"SDate": "2015-01-01", "EDate": today, "Frq": "FY"},
            )
            hist_path = FREE / f"lseg_workspace_sector_probe_with_dates_{today}.csv"
            sector_hist.to_csv(hist_path, index=False)
            result = "current_values_only" if len(sector_hist) <= 2 else "possible_history"
            rows.append({"area": "sector_metadata", "probe": "GICS/TRBC/ICB with SDate/EDate", "result": result, "detail": f"Returned {sector_hist.shape[0]} rows; not PIT history if one row per RIC.", "path": rel(hist_path)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"area": "sector_metadata", "probe": "GICS/TRBC/ICB with SDate/EDate", "result": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:300]}"})

        try:
            discovery = ld.discovery.search(query="SPY call option", top=10)
            discovery_path = FREE / f"lseg_workspace_option_discovery_probe_{today}.csv"
            discovery.to_csv(discovery_path, index=False)
            sample_ric = ""
            if "RIC" in discovery and discovery["RIC"].notna().any():
                sample_ric = str(discovery["RIC"].dropna().iloc[0])
            rows.append({"area": "options", "probe": "active OPRA option discovery", "result": "success" if sample_ric else "no_ric_found", "detail": f"Sample RIC: {sample_ric}", "path": rel(discovery_path)})
        except Exception as exc:  # noqa: BLE001
            sample_ric = "SPYF182675000.U"
            rows.append({"area": "options", "probe": "active OPRA option discovery", "result": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:300]}"})

        option_ric = sample_ric or "SPYF182675000.U"
        option_fields = ["DSPLY_NAME", "TRDPRC_1", "BID", "ASK", "IMP_VOLT", "DELTA", "GAMMA", "VEGA", "THETA", "RHO"]
        try:
            option = ld.get_data([option_ric], option_fields)
            option_path = FREE / f"lseg_workspace_active_option_greek_probe_{today}.csv"
            option.to_csv(option_path, index=False)
            finite_cols = [c for c in ["IMP_VOLT", "DELTA", "GAMMA", "VEGA", "THETA", "RHO"] if c in option and option[c].notna().any()]
            rows.append({"area": "options", "probe": "active option quote/Greek fields", "result": "success" if finite_cols else "no_greeks_returned", "detail": f"{option_ric}; populated fields: {','.join(finite_cols)}", "path": rel(option_path)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"area": "options", "probe": "active option quote/Greek fields", "result": "failed", "detail": f"{option_ric}; {type(exc).__name__}: {str(exc)[:300]}"})

        try:
            hist = ld.get_history(
                [option_ric],
                fields=["BID", "ASK", "TRDPRC_1", "IMP_VOLT", "DELTA", "GAMMA", "VEGA", "THETA", "RHO"],
                start="2026-06-01",
                end=today,
                interval="1D",
            )
            hist_path = FREE / f"lseg_workspace_active_option_history_probe_{today}.csv"
            hist.to_csv(hist_path)
            rows.append({"area": "options", "probe": "active option daily history", "result": "success" if len(hist) else "empty", "detail": f"{option_ric}; returned {len(hist)} rows.", "path": rel(hist_path)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"area": "options", "probe": "active option daily history", "result": "failed", "detail": f"{option_ric}; {type(exc).__name__}: {str(exc)[:300]}"})

        expired_rics = ["SPYB162447800.U", "SPYN162447800.U"]
        for ric in expired_rics:
            try:
                hist = ld.get_history(
                    [ric],
                    fields=["BID", "ASK", "TRDPRC_1", "IMP_VOLT", "DELTA", "GAMMA", "VEGA", "THETA", "RHO"],
                    start="2024-01-29",
                    end="2024-02-02",
                    interval="1D",
                )
                result = "success" if len(hist) else "empty"
                rows.append({"area": "options", "probe": "expired 2024 constructed OPRA RIC", "result": result, "detail": f"{ric}; returned {len(hist)} rows."})
            except Exception as exc:  # noqa: BLE001
                rows.append({"area": "options", "probe": "expired 2024 constructed OPRA RIC", "result": "not_found", "detail": f"{ric}; {type(exc).__name__}: {str(exc)[:240]}"})
    except Exception as exc:  # noqa: BLE001
        rows.append({"area": "session", "probe": "lseg workspace probe", "result": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:500]}"})
    finally:
        if session_open:
            try:
                ld.close_session()
            except Exception:  # noqa: BLE001
                pass

    result_path = REPORTS / "lseg_workspace_probe_results.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    summary = """# LSEG Workspace Data Availability Probe

Evidence label: local LSEG Workspace/Eikon desktop-session probe. Credential values were not printed.

## Findings

- `lseg.data` can open a `desktop.workspace` session when Workspace/Eikon is logged in and a local app key is configured.
- Current equity classification fields are available through `TR.GICSSector`, `TR.GICSSubIndustry`, `TR.TRBCIndustryGroup`, and `TR.ICBIndustry`.
- The tested `SDate`/`EDate` sector request returned current rows, not point-in-time historical GICS.
- Active OPRA option RICs are discoverable and can return quote, implied volatility, and raw Greek fields such as `IMP_VOLT`, `DELTA`, `GAMMA`, `VEGA`, `THETA`, and `RHO`.
- Constructed expired 2024 OPRA RICs from local slice contracts were not found in this desktop-session probe.

## Implication

LSEG Workspace is useful now for current classification metadata and active-option Greek validation. It does not, through the tested desktop path, replace historical raw Greeks for expired OPRA monthly slices or point-in-time sector metadata.
"""
    (REPORTS / "lseg_workspace_probe_summary.md").write_text(summary, encoding="utf-8")
    return {"status": "completed", "rows": len(rows), "results": rel(result_path)}


def estimate_or_fetch_databento(
    client: Any,
    *,
    tag: str,
    paid: bool,
    remaining_budget: float,
    output_path: Path,
    cost_rows: list[dict[str, Any]],
    timeout_seconds: int,
    plan_only: bool,
    n_symbols: int | None = None,
    **kwargs: Any,
) -> tuple[float, pd.DataFrame | None, str]:
    base = {
        "tag": tag,
        "output_path": rel(output_path),
        "n_symbols": n_symbols if n_symbols is not None else len(kwargs.get("symbols") or []),
        **{k: kwargs.get(k) for k in ["dataset", "schema", "start", "end", "stype_in"]},
    }
    if plan_only:
        cost_rows.append({**base, "estimate_dollars": None, "download": False, "status": "planned_no_api_call"})
        return 0.0, None, "planned_no_api_call"

    try:
        est = float(run_databento_call("cost", timeout_seconds, f"cost {tag}", client, **kwargs))
    except Exception as exc:  # noqa: BLE001
        cost_rows.append({**base, "estimate_dollars": None, "download": False, "status": "cost_failed_no_download", "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
        return 0.0, None, "cost_failed_no_download"

    should_download = bool(paid and est <= remaining_budget and remaining_budget > 0)
    over_budget = bool(paid and est > remaining_budget)
    cost_rows.append({**base, "estimate_dollars": est, "download": should_download, "status": "download_requested" if should_download else "estimate_only", "over_remaining_budget": over_budget})
    if not should_download:
        return est, None, "estimate_over_budget" if over_budget else "estimate_only"

    try:
        df = run_databento_call("download", timeout_seconds, f"download {tag}", client, **kwargs)
        if cost_rows:
            cost_rows[-1]["status"] = "downloaded_empty" if df.empty else "downloaded"
            cost_rows[-1]["returned_rows"] = len(df)
        return est, df, "downloaded_empty" if df.empty else "downloaded"
    except Exception as exc:  # noqa: BLE001
        if cost_rows and cost_rows[-1].get("tag") == tag:
            cost_rows[-1]["download"] = False
            cost_rows[-1]["status"] = "download_failed"
            cost_rows[-1]["error"] = f"{type(exc).__name__}: {str(exc)[:220]}"
        else:
            cost_rows.append({**base, "estimate_dollars": est, "download": False, "status": "download_failed", "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
        return est, None, "download_failed"


def databento_client() -> Any:
    from data_ingestion.market_data.fetch_databento import _client

    return _client()


def job_databento_opra_cbbo(args: argparse.Namespace) -> dict[str, Any]:
    client = None if args.plan_only else databento_client()
    underlyings = parse_list(args.underlyings, [])
    slice_files = sorted(DBC.glob("opra_*_slices_*.parquet"))
    if underlyings:
        slice_files = [p for p in slice_files if p.stem.split("_")[1].upper() in underlyings]
    cost_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    spent = 0.0
    downloaded_rows = 0
    planned_requests = 0
    total_attempted_snapshots = 0
    for slice_path in slice_files:
        parts = slice_path.stem.split("_")
        if len(parts) < 4 or parts[1].lower() == "vix":
            continue
        underlying = parts[1].upper()
        year = parts[-1]
        out_path = DBC / f"opra_{underlying}_cbbo1m_{year}.parquet"
        slices = pd.read_parquet(slice_path)
        if slices.empty or "symbol" not in slices or "snap_date" not in slices:
            continue
        estimated_snapshots = 0
        for day, g in slices.groupby("snap_date", observed=True):
            if args.max_snapshots and estimated_snapshots >= args.max_snapshots:
                break
            if args.max_snapshots_total and total_attempted_snapshots >= args.max_snapshots_total:
                break
            day_ts = pd.Timestamp(day)
            if args.opra_start_date and day_ts < pd.Timestamp(args.opra_start_date):
                continue
            if args.opra_end_date and day_ts > pd.Timestamp(args.opra_end_date):
                continue
            day_key = day_ts.strftime("%Y-%m-%d")
            start, end, close_window_type = opra_cbbo_window(day_ts, args)
            symbols = sorted(g["symbol"].dropna().astype(str).unique())
            if not symbols:
                continue
            cached_symbols = set() if args.force else cached_opra_symbols_for_day(out_path, day_key)
            missing_symbols = [s for s in symbols if s not in cached_symbols]
            planned_requests += 1
            if not missing_symbols:
                manifest_rows.append({
                    "job": "databento_opra_cbbo",
                    "underlying": underlying,
                    "year": year,
                    "snap_date": day_key,
                    "n_symbols": len(symbols),
                    "cached_symbols": len(cached_symbols),
                    "missing_symbols": 0,
                    "start": start,
                    "end": end,
                    "close_calendar": args.opra_close_calendar,
                    "close_window_type": close_window_type,
                    "output_path": rel(out_path),
                    "symbol_route": "cached",
                    "status": "cached_snapshot_skipped",
                })
                continue
            manifest_rows.append({
                "job": "databento_opra_cbbo",
                "underlying": underlying,
                "year": year,
                "snap_date": day_key,
                "n_symbols": len(symbols),
                "cached_symbols": len(cached_symbols),
                "missing_symbols": len(missing_symbols),
                "start": start,
                "end": end,
                "close_calendar": args.opra_close_calendar,
                "close_window_type": close_window_type,
                "output_path": rel(out_path),
                "symbol_route": "raw_symbol_planned" if args.plan_only else ("raw_symbol" if args.opra_raw_symbol_timeseries else "instrument_id"),
                "status": "planned",
            })
            request_symbols = missing_symbols
            raw_by_instrument: dict[str, str] = {}
            stype_kwargs: dict[str, str] = {}
            if not args.plan_only and not args.opra_raw_symbol_timeseries:
                request_symbols, raw_by_instrument, not_found, resolve_status = resolve_opra_instrument_ids(client, request_symbols, day_ts, args)
                manifest_rows[-1]["resolve_status"] = resolve_status
                manifest_rows[-1]["resolved_symbols"] = len(request_symbols)
                manifest_rows[-1]["not_found_symbols"] = len(not_found)
                if not request_symbols:
                    manifest_rows[-1]["status"] = "no_resolved_symbols_no_download"
                    estimated_snapshots += 1
                    total_attempted_snapshots += 1
                    continue
                stype_kwargs["stype_in"] = "instrument_id"
            batch_statuses = []
            batch_estimate = 0.0
            batch_returned_rows = 0
            print(
                f"OPRA {underlying} {day_key}: requesting {len(request_symbols)}/{len(symbols)} missing symbols "
                f"(cached={len(cached_symbols)}) from {start} to {end}",
                flush=True,
            )
            for batch_index, batch_symbols in enumerate(chunks(request_symbols, args.opra_instrument_batch_size), start=1):
                est, df, status = estimate_or_fetch_databento(
                    client,
                    tag=f"opra-cbbo-{underlying}-{day_ts.date()}-b{batch_index:03d}",
                    paid=args.paid,
                    remaining_budget=max(args.max_paid_dollars - spent, 0.0),
                    output_path=out_path,
                    cost_rows=cost_rows,
                    timeout_seconds=args.databento_timeout_seconds,
                    plan_only=args.plan_only,
                    n_symbols=len(batch_symbols),
                    dataset="OPRA.PILLAR",
                    schema="cbbo-1m",
                    symbols=batch_symbols,
                    start=start,
                    end=end,
                    **stype_kwargs,
                )
                batch_statuses.append(status)
                batch_estimate += est
                spent += est if args.paid and df is not None else 0.0
                if df is not None and not df.empty:
                    df = restore_raw_opra_symbol(df, raw_by_instrument)
                    meta = g[["symbol", "snap_date", "spot", "expiry"]].drop_duplicates("symbol")
                    df = df.merge(meta, on="symbol", how="left")
                    batch_returned_rows += len(df)
                    append_rows_to_parquet(
                        out_path,
                        df,
                        ["symbol", "snap_date", "ts_recv", "ts_event", "bid_px_00", "ask_px_00"],
                    )
                    downloaded_rows += len(df)
                print(
                    f"  batch {batch_index}/{len(chunks(request_symbols, args.opra_instrument_batch_size))}: "
                    f"{status}, returned_rows={0 if df is None else len(df)}",
                    flush=True,
                )
            estimated_snapshots += 1
            total_attempted_snapshots += 1
            if batch_returned_rows and any(s != "downloaded" for s in batch_statuses):
                snapshot_status = "downloaded_partial"
            elif batch_returned_rows:
                snapshot_status = "downloaded"
            else:
                snapshot_status = ";".join(sorted(set(batch_statuses))) if batch_statuses else "not_attempted"
            manifest_rows[-1]["status"] = snapshot_status
            manifest_rows[-1]["estimate_dollars"] = batch_estimate if not args.plan_only else None
            manifest_rows[-1]["batches"] = len(batch_statuses)
            manifest_rows[-1]["returned_rows"] = batch_returned_rows
        if args.max_snapshots_total and total_attempted_snapshots >= args.max_snapshots_total:
            break
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS / ("data_gap_fill_opra_cbbo_plan_manifest.csv" if args.plan_only else "data_gap_fill_opra_cbbo_manifest.csv")
    cost_path = REPORTS / ("data_gap_fill_costs_opra_cbbo_plan.csv" if args.plan_only else "data_gap_fill_costs_opra_cbbo.csv")
    pd.DataFrame(cost_rows).to_csv(cost_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    estimated_requests = len([r for r in cost_rows if r.get("estimate_dollars") is not None])
    return {
        "planned_requests": planned_requests,
        "estimated_requests": estimated_requests,
        "api_attempted_snapshots": total_attempted_snapshots,
        "downloaded_rows": downloaded_rows,
        "spent_estimate_dollars": spent,
        "manifest": rel(manifest_path),
    }


def job_databento_opra_surface_full_day_cbbo(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch full-session CBBO for option/date keys already in opra_surface_panel."""
    client = None if args.plan_only else databento_client()
    contracts = opra_surface_contracts_from_panel(args)
    OPRA_FULL_DAY_CBBO_DIR.mkdir(parents=True, exist_ok=True)
    cost_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    spent = 0.0
    downloaded_rows = 0
    planned_requests = 0
    total_attempted_snapshots = 0
    if contracts.empty:
        manifest_path = REPORTS / ("data_gap_fill_opra_surface_full_day_cbbo_plan_manifest.csv" if args.plan_only else "data_gap_fill_opra_surface_full_day_cbbo_manifest.csv")
        cost_path = REPORTS / ("data_gap_fill_costs_opra_surface_full_day_cbbo_plan.csv" if args.plan_only else "data_gap_fill_costs_opra_surface_full_day_cbbo.csv")
        pd.DataFrame().to_csv(manifest_path, index=False)
        pd.DataFrame().to_csv(cost_path, index=False)
        return {"planned_requests": 0, "estimated_requests": 0, "api_attempted_snapshots": 0, "downloaded_rows": 0, "spent_estimate_dollars": 0.0, "manifest": rel(manifest_path)}

    for day, g in contracts.groupby("snap_date", observed=True):
        if args.max_snapshots_total and total_attempted_snapshots >= args.max_snapshots_total:
            break
        day_ts = pd.Timestamp(day)
        year = str(day_ts.year)
        day_key = day_ts.strftime("%Y-%m-%d")
        start = day_ts.strftime("%Y-%m-%d")
        end = (day_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        cached_symbols: set[str] = set()
        if not args.force:
            for underlying, ug in g.groupby("underlying", observed=True):
                out_path = OPRA_FULL_DAY_CBBO_DIR / f"opra_{underlying}_cbbo1m_fullday_{year}.parquet"
                under_cached = cached_opra_symbols_for_day(out_path, day_key)
                under_symbols = set(ug["symbol"].dropna().astype(str))
                cached_symbols.update(under_symbols & under_cached)
        symbols = sorted(g["symbol"].dropna().astype(str).unique())
        if not symbols:
            continue
        missing_symbols = [s for s in symbols if s not in cached_symbols]
        planned_requests += 1
        if not missing_symbols:
            manifest_rows.append({
                "job": "databento_opra_surface_full_day_cbbo",
                "cache_scope": "surface_panel_full_day",
                "underlying": ",".join(sorted(g["underlying"].dropna().astype(str).unique())),
                "year": year,
                "snap_date": day_key,
                "n_underlyings": int(g["underlying"].nunique()),
                "n_symbols": len(symbols),
                "cached_symbols": len(cached_symbols),
                "missing_symbols": 0,
                "start": start,
                "end": end,
                "output_path": rel(OPRA_FULL_DAY_CBBO_DIR),
                "symbol_route": "cached",
                "status": "cached_snapshot_skipped",
            })
            continue
        manifest_rows.append({
            "job": "databento_opra_surface_full_day_cbbo",
            "cache_scope": "surface_panel_full_day",
            "underlying": ",".join(sorted(g["underlying"].dropna().astype(str).unique())),
            "year": year,
            "snap_date": day_key,
            "n_underlyings": int(g["underlying"].nunique()),
            "n_symbols": len(symbols),
            "cached_symbols": len(cached_symbols),
            "missing_symbols": len(missing_symbols),
            "start": start,
            "end": end,
            "output_path": rel(OPRA_FULL_DAY_CBBO_DIR),
            "symbol_route": "raw_symbol_planned" if args.plan_only else ("raw_symbol" if args.opra_raw_symbol_timeseries else "instrument_id"),
            "status": "planned",
        })
        request_symbols = missing_symbols
        raw_by_instrument: dict[str, str] = {}
        stype_kwargs: dict[str, str] = {}
        if not args.plan_only and not args.opra_raw_symbol_timeseries:
            request_symbols, raw_by_instrument, not_found, resolve_status = resolve_opra_instrument_ids(client, request_symbols, day_ts, args)
            manifest_rows[-1]["resolve_status"] = resolve_status
            manifest_rows[-1]["resolved_symbols"] = len(request_symbols)
            manifest_rows[-1]["not_found_symbols"] = len(not_found)
            if not request_symbols:
                manifest_rows[-1]["status"] = "no_resolved_symbols_no_download"
                total_attempted_snapshots += 1
                continue
            stype_kwargs["stype_in"] = "instrument_id"
        batch_statuses = []
        batch_estimate = 0.0
        batch_returned_rows = 0
        symbol_batches = chunks(request_symbols, args.opra_instrument_batch_size)
        print(
            f"OPRA full-day {day_key}: requesting {len(request_symbols)}/{len(symbols)} missing symbols "
            f"across {g['underlying'].nunique()} underlyings (cached={len(cached_symbols)}) from {start} to {end}",
            flush=True,
        )
        for batch_index, batch_symbols in enumerate(symbol_batches, start=1):
            est, df, status = estimate_or_fetch_databento(
                client,
                tag=f"opra-surface-fullday-cbbo-{day_ts.date()}-b{batch_index:03d}",
                paid=args.paid,
                remaining_budget=max(args.max_paid_dollars - spent, 0.0),
                output_path=OPRA_FULL_DAY_CBBO_DIR,
                cost_rows=cost_rows,
                timeout_seconds=args.databento_timeout_seconds,
                plan_only=args.plan_only,
                n_symbols=len(batch_symbols),
                dataset="OPRA.PILLAR",
                schema="cbbo-1m",
                symbols=batch_symbols,
                start=start,
                end=end,
                **stype_kwargs,
            )
            batch_statuses.append(status)
            batch_estimate += est
            spent += est if args.paid and df is not None else 0.0
            if df is not None and not df.empty:
                df = restore_raw_opra_symbol(df, raw_by_instrument)
                meta = g[["underlying", "symbol", "snap_date", "spot", "expiry"]].drop_duplicates("symbol")
                df = df.merge(meta, on="symbol", how="left")
                df["cbbo_cache_scope"] = "surface_panel_full_day"
                batch_returned_rows += len(df)
                for underlying, part in df.dropna(subset=["underlying"]).groupby("underlying", observed=True):
                    out_path = OPRA_FULL_DAY_CBBO_DIR / f"opra_{str(underlying).upper()}_cbbo1m_fullday_{year}.parquet"
                    append_rows_to_parquet(
                        out_path,
                        part.drop(columns=["underlying"]),
                        ["symbol", "snap_date", "ts_recv", "ts_event", "bid_px_00", "ask_px_00"],
                    )
                downloaded_rows += len(df)
            print(
                f"  full-day batch {batch_index}/{len(symbol_batches)}: "
                f"{status}, returned_rows={0 if df is None else len(df)}",
                flush=True,
            )
        total_attempted_snapshots += 1
        if batch_returned_rows and any(s != "downloaded" for s in batch_statuses):
            snapshot_status = "downloaded_partial"
        elif batch_returned_rows:
            snapshot_status = "downloaded"
        else:
            snapshot_status = ";".join(sorted(set(batch_statuses))) if batch_statuses else "not_attempted"
        manifest_rows[-1]["status"] = snapshot_status
        manifest_rows[-1]["estimate_dollars"] = batch_estimate if not args.plan_only else None
        manifest_rows[-1]["batches"] = len(batch_statuses)
        manifest_rows[-1]["returned_rows"] = batch_returned_rows

    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS / ("data_gap_fill_opra_surface_full_day_cbbo_plan_manifest.csv" if args.plan_only else "data_gap_fill_opra_surface_full_day_cbbo_manifest.csv")
    cost_path = REPORTS / ("data_gap_fill_costs_opra_surface_full_day_cbbo_plan.csv" if args.plan_only else "data_gap_fill_costs_opra_surface_full_day_cbbo.csv")
    pd.DataFrame(cost_rows).to_csv(cost_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    estimated_requests = len([r for r in cost_rows if r.get("estimate_dollars") is not None])
    return {
        "planned_requests": planned_requests,
        "estimated_requests": estimated_requests,
        "api_attempted_snapshots": total_attempted_snapshots,
        "downloaded_rows": downloaded_rows,
        "spent_estimate_dollars": spent,
        "manifest": rel(manifest_path),
        "cache_dir": rel(OPRA_FULL_DAY_CBBO_DIR),
    }


def job_databento_futures_contracts(args: argparse.Namespace) -> dict[str, Any]:
    client = None if args.plan_only else databento_client()
    roots = parse_list(args.futures_roots, DEFAULT_FUTURE_ROOTS)
    schemas = parse_schema_list(args.futures_schemas, ["definition", "ohlcv-1d"])
    invalid_schemas = sorted(set(schemas) - {"definition", "ohlcv-1d"})
    if invalid_schemas:
        raise SystemExit(f"unsupported futures schema(s): {invalid_schemas}; use definition and/or ohlcv-1d")
    out_dir = DBC / "futures_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    spent = 0.0
    downloaded_rows = 0
    planned_requests = 0
    attempted_requests = 0
    for root in roots:
        parent = f"{root}.FUT"
        for schema in schemas:
            for year in range(args.start_year, args.end_year + 1):
                if args.futures_max_requests_total and attempted_requests >= args.futures_max_requests_total:
                    break
                symbol_mode = args.futures_symbol_mode
                request_symbols = [parent]
                stype_in = "parent"
                suffix = ""
                if symbol_mode == "liquid":
                    request_symbols = liquid_futures_symbols(root, year)
                    stype_in = "raw_symbol"
                    suffix = "_liquid"
                elif symbol_mode != "parent":
                    raise SystemExit("--futures-symbol-mode must be parent or liquid")
                out_path = out_dir / f"glbx_{root}_{schema}_{year}{suffix}.parquet"
                cached_contracts: set[str] = set()
                if out_path.exists() and not args.force and symbol_mode == "liquid":
                    try:
                        cached = pd.read_parquet(out_path, columns=["symbol"])
                        cached_contracts = set(cached["symbol"].dropna().astype(str))
                    except Exception:  # noqa: BLE001
                        cached_contracts = set()
                    request_symbols = [s for s in request_symbols if s not in cached_contracts]
                if out_path.exists() and not args.force and (symbol_mode != "liquid" or not request_symbols):
                    manifest_rows.append({"job": "databento_futures_contracts", "root": root, "schema": schema, "year": year, "symbol": ",".join(request_symbols), "symbol_mode": symbol_mode, "output_path": rel(out_path), "status": "cached_file_skipped"})
                    continue
                request_groups = [(request_symbols, f"{year}-01-01", annual_window_end(year), "annual")]
                if schema == "definition":
                    request_groups = futures_definition_request_groups(out_dir, root, year, request_symbols, args)
                for group_index, (group_symbols, start, end, window_source) in enumerate(request_groups, start=1):
                    if args.futures_max_requests_total and attempted_requests >= args.futures_max_requests_total:
                        break
                    planned_requests += 1
                    manifest_rows.append({
                        "job": "databento_futures_contracts",
                        "root": root,
                        "schema": schema,
                        "year": year,
                        "symbol": ",".join(group_symbols),
                        "symbol_mode": symbol_mode,
                        "cached_symbols": len(cached_contracts),
                        "start": start,
                        "end": end,
                        "definition_window_source": window_source if schema == "definition" else "",
                        "output_path": rel(out_path),
                        "status": "planned",
                    })
                    symbol_batches = chunks(group_symbols, args.futures_symbol_batch_size) if args.futures_symbol_batch_size > 0 else [group_symbols]
                    print(
                        f"FUT {root} {schema} {year} group {group_index}/{len(request_groups)}: "
                        f"requesting {len(group_symbols)} {symbol_mode} symbol(s) in {len(symbol_batches)} "
                        f"batch(es) from {start} to {end}",
                        flush=True,
                    )
                    batch_statuses = []
                    batch_estimate = 0.0
                    batch_returned_rows = 0
                    for batch_index, batch_symbols in enumerate(symbol_batches, start=1):
                        est, df, status = estimate_or_fetch_databento(
                            client,
                            tag=f"glbx-{root}-{schema}-{year}-g{group_index:03d}-b{batch_index:03d}",
                            paid=args.paid,
                            remaining_budget=max(args.max_paid_dollars - spent, 0.0),
                            output_path=out_path,
                            cost_rows=cost_rows,
                            timeout_seconds=args.databento_timeout_seconds,
                            plan_only=args.plan_only,
                            n_symbols=len(batch_symbols),
                            dataset="GLBX.MDP3",
                            schema=schema,
                            symbols=batch_symbols,
                            stype_in=stype_in,
                            start=start,
                            end=end,
                        )
                        attempted_requests += 1
                        batch_statuses.append(status)
                        batch_estimate += est
                        spent += est if args.paid and df is not None else 0.0
                        if df is not None and not df.empty:
                            df["future_root"] = root
                            df["definition_window_source"] = window_source if schema == "definition" else ""
                            append_rows_to_parquet(out_path, df, ["symbol", "ts_event", "instrument_id"])
                            downloaded_rows += len(df)
                            batch_returned_rows += len(df)
                        print(f"  FUT {root} {schema} {year} batch {batch_index}/{len(symbol_batches)}: {status}, returned_rows={0 if df is None else len(df)}", flush=True)
                        if args.futures_max_requests_total and attempted_requests >= args.futures_max_requests_total:
                            break
                    manifest_rows[-1]["status"] = "downloaded" if batch_returned_rows and all(s == "downloaded" for s in batch_statuses) else ("downloaded_partial" if batch_returned_rows else ";".join(sorted(set(batch_statuses))))
                    manifest_rows[-1]["estimate_dollars"] = batch_estimate if not args.plan_only else None
                    manifest_rows[-1]["batches"] = len(batch_statuses)
                    manifest_rows[-1]["returned_rows"] = batch_returned_rows
            if args.futures_max_requests_total and attempted_requests >= args.futures_max_requests_total:
                break
        if args.futures_max_requests_total and attempted_requests >= args.futures_max_requests_total:
            break
    REPORTS.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS / ("data_gap_fill_futures_contracts_plan_manifest.csv" if args.plan_only else "data_gap_fill_futures_contracts_manifest.csv")
    cost_path = REPORTS / ("data_gap_fill_costs_futures_contracts_plan.csv" if args.plan_only else "data_gap_fill_costs_futures_contracts.csv")
    pd.DataFrame(cost_rows).to_csv(cost_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    estimated_requests = len([r for r in cost_rows if r.get("estimate_dollars") is not None])
    return {
        "planned_requests": planned_requests,
        "estimated_requests": estimated_requests,
        "api_attempted_requests": attempted_requests,
        "downloaded_rows": downloaded_rows,
        "spent_estimate_dollars": spent,
        "manifest": rel(manifest_path),
    }


JOBS = {
    "free_sectors": job_free_sectors,
    "yfinance_options": job_yfinance_options,
    "polygon_options": job_polygon_options,
    "lseg_workspace_probe": job_lseg_workspace_probe,
    "databento_opra_cbbo": job_databento_opra_cbbo,
    "databento_opra_surface_full_day_cbbo": job_databento_opra_surface_full_day_cbbo,
    "databento_futures_contracts": job_databento_futures_contracts,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="free_sectors,yfinance_options", help=f"Comma-separated jobs from: {','.join(JOBS)}")
    parser.add_argument("--underlyings", default="", help="Comma-separated option underlyings")
    parser.add_argument("--futures-roots", default="", help="Comma-separated futures roots")
    parser.add_argument("--max-expiries", type=int, default=6)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-snapshots", type=int, default=0, help="Cap Databento OPRA snapshot cost probes/downloads per slice file; 0 means no cap")
    parser.add_argument("--max-snapshots-total", type=int, default=0, help="Cap total Databento OPRA snapshot cost probes/downloads across the whole run; 0 means no cap")
    parser.add_argument("--futures-max-requests-total", type=int, default=0, help="Cap total Databento futures request probes/downloads; 0 means no cap")
    parser.add_argument("--futures-schemas", default="", help="Comma-separated Databento futures schemas to fetch: definition,ohlcv-1d")
    parser.add_argument("--futures-symbol-mode", choices=["parent", "liquid"], default="parent", help="Use parent roots or generated liquid contract symbols for futures requests")
    parser.add_argument("--futures-symbol-batch-size", type=int, default=0, help="Batch size for liquid futures symbols; 0 requests all symbols together")
    parser.add_argument("--futures-definition-window-days", type=int, default=2, help="Definition-schema window length when anchoring exact futures definitions to cached bar dates")
    parser.add_argument("--no-futures-definition-from-bars", dest="futures_definition_from_bars", action="store_false", help="Use full annual definition windows instead of cached-bar anchored windows")
    parser.set_defaults(futures_definition_from_bars=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--paid", action="store_true", help="Allow metered vendor downloads after cost checks")
    parser.add_argument("--max-paid-dollars", type=float, default=0.0)
    parser.add_argument("--plan-only", action="store_true", help="Write exact paid-data manifests without creating a Databento client or making API calls")
    parser.add_argument("--databento-timeout-seconds", type=int, default=30, help="Local timeout for each Databento cost/download call; 0 disables")
    parser.add_argument("--opra-cbbo-window-minutes", type=int, default=75, help="Minutes before OPRA market close to request for CBBO EOD surfaces")
    parser.add_argument("--opra-cbbo-after-close-minutes", type=int, default=15, help="Minutes after OPRA market close to include for late quotes")
    parser.add_argument("--opra-cbbo-full-day", action="store_true", help="Request full OPRA days instead of the default close window")
    parser.add_argument("--opra-close-calendar", choices=["auto", "regular", "early"], default="auto", help="Close time calendar for OPRA close-window pulls; auto handles common 1pm ET half-days")
    parser.add_argument("--opra-raw-symbol-timeseries", action="store_true", help="Do not resolve OPRA raw symbols to instrument IDs before timeseries pulls")
    parser.add_argument("--opra-instrument-batch-size", type=int, default=2, help="Instrument IDs per OPRA CBBO request after symbology resolution")
    parser.add_argument("--opra-start-date", default="", help="Optional first OPRA snap_date to cost/download")
    parser.add_argument("--opra-end-date", default="", help="Optional last OPRA snap_date to cost/download")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    write_source_notes()
    write_paid_data_plan()
    summaries = {}
    for job_name in parse_list(args.jobs, []):
        key = job_name.lower()
        if key not in JOBS:
            raise SystemExit(f"unknown job {job_name!r}; valid jobs: {sorted(JOBS)}")
        print(f"[{key}]")
        summaries[key] = JOBS[key](args)
        print(json.dumps(summaries[key], default=str, indent=2))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_gap_fill_run_summary.json").write_text(json.dumps(summaries, default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
