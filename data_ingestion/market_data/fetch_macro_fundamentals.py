"""Vector 4 macro + commodity fundamentals fetcher (free, credentialed-but-no-marginal-cost).

Pulls four free official sources into ``data/free_cache/`` for cross-asset
macro-regime overlays and commodity positioning/fundamental signals:

  * fred  — FRED market series (rates curve, real/breakeven inflation, credit OAS,
            financial-conditions, commodity spot, FX/dollar). FRED_API_KEY.
  * eia   — EIA weekly energy fundamentals (crude/Cushing/product stocks, crude
            production, working gas in storage) via the EIA v2 seriesid bridge. EIA_API_KEY.
  * cftc  — CFTC Commitments of Traders (Legacy Futures-Only, Socrata dataset
            6dca-aqww) for the futures roots we trade. No key required.
  * bea   — BEA NIPA macro (real GDP growth, PCE price index). BEA_API_KEY.

Run:
  .venv/bin/python -m data_ingestion.market_data.fetch_macro_fundamentals --jobs fred,eia,cftc,bea
  .venv/bin/python -m data_ingestion.market_data.fetch_macro_fundamentals --jobs cftc --start 2010-01-01

POINT-IN-TIME (critical for backtesting):
  * CFTC CoT positions are *as of* the Tuesday ``report_date`` but are PUBLISHED the
    following Friday (~3 calendar days later). We add a ``release_date`` column
    (report_date + 3d) — align any trading signal to ``release_date``, never ``report_date``.
  * EIA weekly stocks are *for* the Friday week-ending ``period`` and are released the
    next week (crude Wed, gas Thu). We add ``release_date_est`` (period + 5d) — a
    conservative lag. Treat as approximate; confirm against the EIA release calendar
    for high-frequency use.
  * FRED ``series/observations`` returns the LATEST-REVISED value, not the vintage that
    was public on a given day. Market series here (rates, OAS, VIX, FX, spot) are
    effectively revision-free, so this is safe. Revision-prone macro (GDP, PCE) is left
    to the BEA job and flagged below — for strict PIT macro backtests use FRED ALFRED
    vintages (realtime_start/realtime_end) instead.

SECURITY (repo contract): API keys are loaded from .env via python-dotenv and used
only inside request parameters. They are NEVER printed, logged, or written to output;
all error text is passed through ``_san()`` so a key cannot leak in a traceback.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
FREE = ROOT / "data" / "free_cache"
REPORTS = ROOT / "research" / "reports" / "pipeline_reports"

FRED_KEY = os.getenv("FRED_API_KEY")
EIA_KEY = os.getenv("EIA_API_KEY")
BEA_KEY = os.getenv("BEA_API_KEY")

DEFAULT_START = "2010-01-01"


def _san(msg: str) -> str:
    """Scrub any known API key from a message before it is printed or raised."""
    out = str(msg)
    for k in (FRED_KEY, EIA_KEY, BEA_KEY):
        if k:
            out = out.replace(k, "***")
    return out


def _get(url: str, params: dict[str, Any], *, tries: int = 4, timeout: int = 40) -> requests.Response:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r
        except Exception as exc:  # noqa: BLE001 — sanitize before retry/raise
            last = RuntimeError(f"{type(exc).__name__}: {_san(str(exc))}")
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


# --------------------------------------------------------------------------- FRED

# (series_id, human label, group). Resilient: any series that 404s is skipped.
FRED_SERIES: list[tuple[str, str, str]] = [
    # Treasury constant-maturity curve
    ("DGS1MO", "UST 1M", "rates"), ("DGS3MO", "UST 3M", "rates"),
    ("DGS6MO", "UST 6M", "rates"), ("DGS1", "UST 1Y", "rates"),
    ("DGS2", "UST 2Y", "rates"), ("DGS3", "UST 3Y", "rates"),
    ("DGS5", "UST 5Y", "rates"), ("DGS7", "UST 7Y", "rates"),
    ("DGS10", "UST 10Y", "rates"), ("DGS20", "UST 20Y", "rates"),
    ("DGS30", "UST 30Y", "rates"),
    # Curve slopes, real yields, inflation breakevens
    ("T10Y2Y", "10Y-2Y slope", "curve"), ("T10Y3M", "10Y-3M slope", "curve"),
    ("DFII5", "5Y real (TIPS)", "real"), ("DFII10", "10Y real (TIPS)", "real"),
    ("T5YIE", "5Y breakeven inflation", "inflation"),
    ("T10YIE", "10Y breakeven inflation", "inflation"),
    ("T5YIFR", "5y5y fwd inflation", "inflation"),
    # Credit / risk / financial conditions
    # NB: ICE BofA OAS series (BAMLH0A0HYM2/BAMLC0A0CM) are licensing-limited on FRED to a
    # rolling ~3yr window, so we use Moody's seasoned-corporate spreads which have full history.
    ("BAA10Y", "Moody's Baa - 10Y Treasury spread", "credit"),
    ("AAA10Y", "Moody's Aaa - 10Y Treasury spread", "credit"),
    ("BAMLH0A0HYM2", "US HY OAS (recent only)", "credit"),
    ("VIXCLS", "VIX", "vol"),
    ("NFCI", "Chicago Fed NFCI", "fin_conditions"),
    ("ANFCI", "Adjusted NFCI", "fin_conditions"),
    ("STLFSI4", "St Louis Fin Stress", "fin_conditions"),
    # Commodity spot / reference (daily where available)
    ("DCOILWTICO", "WTI spot", "commodity"),
    ("DCOILBRENTEU", "Brent spot", "commodity"),
    ("DHHNGSP", "Henry Hub nat gas spot", "commodity"),
    ("GOLDAMGBD228NLBM", "Gold LBMA AM fix", "commodity"),
    ("PCOPPUSDM", "Global copper price (monthly)", "commodity"),
    ("PPIACO", "PPI all commodities (monthly)", "commodity"),
    # FX / dollar
    ("DTWEXBGS", "Broad USD index", "fx"),
    ("DTWEXAFEGS", "Advanced-economies USD index", "fx"),
    ("DEXUSEU", "USD/EUR", "fx"),
    ("DEXJPUS", "JPY/USD", "fx"),
    ("DEXCHUS", "CNY/USD", "fx"),
]


def fetch_fred_series(series_id: str, start: str) -> pd.DataFrame:
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY not found in .env")
    r = _get(
        "https://api.stlouisfed.org/fred/series/observations",
        {"series_id": series_id, "api_key": FRED_KEY, "file_type": "json",
         "observation_start": start},
    )
    if r.status_code == 400:  # series not found / discontinued id — skip gracefully
        return pd.DataFrame(columns=["date", "series_id", "value"])
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = [
        {"date": pd.Timestamp(o["date"]), "series_id": series_id,
         "value": float(o["value"]) if o["value"] not in (".", "") else float("nan")}
        for o in obs
    ]
    return pd.DataFrame(rows)


def job_fred(args: argparse.Namespace) -> dict[str, Any]:
    FREE.mkdir(parents=True, exist_ok=True)
    frames, meta, skipped = [], [], []
    for sid, label, group in FRED_SERIES:
        try:
            df = fetch_fred_series(sid, args.start)
        except Exception as exc:  # noqa: BLE001
            skipped.append((sid, _san(str(exc))[:80]))
            continue
        if df.empty:
            skipped.append((sid, "no observations / discontinued"))
            continue
        df["label"] = label
        df["group"] = group
        frames.append(df)
        d = df.dropna(subset=["value"])
        lo = d["date"].min().date() if len(d) else "?"
        hi = d["date"].max().date() if len(d) else "?"
        meta.append((sid, group, len(d), lo, hi))
        print(f"  FRED {sid:18s} {group:14s} n={len(d):6d} {lo}..{hi}")
        time.sleep(0.1)
    if not frames:
        raise RuntimeError("FRED returned no series — check FRED_API_KEY / connectivity")
    long = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"])
    long_path = FREE / "fred_macro.parquet"
    long.to_parquet(long_path, index=False)
    wide = long.pivot_table(index="date", columns="series_id", values="value").sort_index()
    wide_path = FREE / "fred_macro_wide.csv"
    wide.to_csv(wide_path)
    for sid, why in skipped:
        print(f"  FRED skipped {sid}: {why}")
    return {"job": "fred", "series_ok": len(frames), "series_skipped": len(skipped),
            "rows": len(long), "long": rel(long_path), "wide": rel(wide_path)}


# --------------------------------------------------------------------------- EIA

# (eia v2 series id, human label). Skipped individually on failure.
EIA_SERIES: list[tuple[str, str]] = [
    ("PET.WCESTUS1.W", "US crude stocks excl SPR (kbbl, weekly)"),
    ("PET.W_EPC0_SAX_YCUOK_MBBL.W", "Cushing OK crude stocks (kbbl, weekly)"),
    ("PET.WCRFPUS2.W", "US crude field production (kbbl/d, weekly)"),
    ("PET.WGTSTUS1.W", "US total gasoline stocks (kbbl, weekly)"),
    ("PET.WDISTUS1.W", "US distillate stocks (kbbl, weekly)"),
    ("NG.NW2_EPG0_SWO_R48_BCF.W", "Working gas in storage, Lower 48 (Bcf, weekly)"),
]


def fetch_eia_series(series_id: str) -> pd.DataFrame:
    if not EIA_KEY:
        raise RuntimeError("EIA_API_KEY not found in .env")
    r = _get(f"https://api.eia.gov/v2/seriesid/{series_id}", {"api_key": EIA_KEY})
    r.raise_for_status()
    data = r.json().get("response", {}).get("data", [])
    rows = []
    for d in data:
        v = d.get("value")
        rows.append({
            "period": pd.Timestamp(d["period"]),
            "series_id": series_id,
            "value": float(v) if v not in (None, "", ".") else float("nan"),
            "units": d.get("units"),
            "description": d.get("series-description"),
        })
    return pd.DataFrame(rows)


def job_eia(args: argparse.Namespace) -> dict[str, Any]:
    FREE.mkdir(parents=True, exist_ok=True)
    frames, skipped = [], []
    for sid, label in EIA_SERIES:
        try:
            df = fetch_eia_series(sid)
        except Exception as exc:  # noqa: BLE001
            skipped.append((sid, _san(str(exc))[:80]))
            continue
        if df.empty:
            skipped.append((sid, "no data"))
            continue
        df["label"] = label
        frames.append(df)
        d = df.dropna(subset=["value"])
        print(f"  EIA {sid:32s} n={len(d):5d} {d['period'].min().date()}..{d['period'].max().date()}")
        time.sleep(0.1)
    if not frames:
        raise RuntimeError("EIA returned no series — check EIA_API_KEY / connectivity")
    long = pd.concat(frames, ignore_index=True).sort_values(["series_id", "period"])
    # Point-in-time: weekly stocks/production are released ~5 days after the week-ending period.
    long["release_date_est"] = long["period"] + pd.Timedelta(days=5)
    # Keep full EIA history (back to 1982) — useful for robust seasonal inventory norms.
    path = FREE / "eia_energy.parquet"
    long.to_parquet(path, index=False)
    for sid, why in skipped:
        print(f"  EIA skipped {sid}: {why}")
    return {"job": "eia", "series_ok": len(frames), "series_skipped": len(skipped),
            "rows": len(long), "path": rel(path)}


# --------------------------------------------------------------------------- CFTC

CFTC_DATASET = "6dca-aqww"  # Legacy Commitments of Traders — Futures Only
# root -> (display, [exact market_and_exchange_names...]). CFTC renamed a batch of major
# contracts around 2022-02-01, so several roots stitch a legacy name (2010->2022) with the
# current name (2022->2026). Where names overlap in time, dedup keeps the higher-OI row.
# Names verified against the live dataset's distinct market_and_exchange_names.
CFTC_MARKETS: dict[str, tuple[str, list[str]]] = {
    "GC": ("Gold (COMEX)", ["GOLD - COMMODITY EXCHANGE INC."]),
    "SI": ("Silver (COMEX)", ["SILVER - COMMODITY EXCHANGE INC."]),
    "HG": ("Copper (COMEX)", ["COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
                              "COPPER- #1 - COMMODITY EXCHANGE INC."]),
    "CL": ("WTI Crude (NYMEX)", ["CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
                                 "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE"]),
    "NG": ("Henry Hub Nat Gas (NYMEX)", ["NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
                                         "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE"]),
    "ZC": ("Corn (CBOT)", ["CORN - CHICAGO BOARD OF TRADE"]),
    "ES": ("S&P 500 (CME, consolidated)", ["S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE"]),
    "NQ": ("Nasdaq-100 (CME, consolidated)", ["NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE"]),
    "ZN": ("10Y T-Note (CBOT)", ["10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
                                 "UST 10Y NOTE - CHICAGO BOARD OF TRADE"]),
    "6E": ("Euro FX (CME)", ["EURO FX - CHICAGO MERCANTILE EXCHANGE"]),
}

CFTC_FIELDS = [
    "market_and_exchange_names", "report_date_as_yyyy_mm_dd", "contract_market_name",
    "cftc_contract_market_code", "commodity_name", "commodity_subgroup_name",
    "open_interest_all",
    "noncomm_positions_long_all", "noncomm_positions_short_all", "noncomm_postions_spread_all",
    "comm_positions_long_all", "comm_positions_short_all",
    "nonrept_positions_long_all", "nonrept_positions_short_all",
    "traders_tot_all",
]


def fetch_cftc_name(name: str, start: str) -> pd.DataFrame:
    url = f"https://publicreporting.cftc.gov/resource/{CFTC_DATASET}.json"
    safe = name.replace("'", "''")  # SoQL string escaping
    where = f"market_and_exchange_names = '{safe}'"
    if start:
        where += f" and report_date_as_yyyy_mm_dd >= '{start}T00:00:00.000'"
    out, offset, page = [], 0, 50000
    while True:
        r = _get(url, {"$select": ",".join(CFTC_FIELDS), "$where": where,
                       "$order": "report_date_as_yyyy_mm_dd", "$limit": page, "$offset": offset})
        r.raise_for_status()
        batch = r.json()
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return pd.DataFrame(out)


def job_cftc(args: argparse.Namespace) -> dict[str, Any]:
    FREE.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in CFTC_FIELDS if c not in (
        "market_and_exchange_names", "report_date_as_yyyy_mm_dd", "contract_market_name",
        "cftc_contract_market_code", "commodity_name", "commodity_subgroup_name")]
    frames, skipped = [], []
    for root, (label, names) in CFTC_MARKETS.items():
        parts = []
        for name in names:
            try:
                d = fetch_cftc_name(name, args.start)
            except Exception as exc:  # noqa: BLE001
                skipped.append((f"{root}:{name}", _san(str(exc))[:60]))
                continue
            if not d.empty:
                parts.append(d)
            time.sleep(0.15)
        if not parts:
            skipped.append((root, "no rows for any configured name"))
            continue
        df = pd.concat(parts, ignore_index=True)
        df["root"] = root
        df["root_label"] = label
        df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Stitch legacy+current names: where they overlap on a report_date, keep the higher-OI row.
        df = (df.sort_values(["report_date", "open_interest_all"])
                .drop_duplicates("report_date", keep="last")
                .sort_values("report_date").reset_index(drop=True))
        # Derived positioning signals
        df["noncomm_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["comm_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
        oi = df["open_interest_all"].replace(0, pd.NA)
        df["noncomm_net_pct_oi"] = df["noncomm_net"] / oi
        # Point-in-time: Tuesday report -> released the following Friday (~3 days).
        df["release_date"] = df["report_date"] + pd.Timedelta(days=3)
        frames.append(df)
        print(f"  CFTC {root:3s} {label:30s} n={len(df):5d} names={len(names)} "
              f"{df['report_date'].min().date()}..{df['report_date'].max().date()}")
    if not frames:
        raise RuntimeError("CFTC returned no rows — check connectivity / patterns")
    out = pd.concat(frames, ignore_index=True).sort_values(["root", "report_date"])
    path = FREE / "cftc_cot_futures_only.parquet"
    out.to_parquet(path, index=False)
    for root, why in skipped:
        print(f"  CFTC skipped {root}: {why}")
    return {"job": "cftc", "roots_ok": len(frames), "roots_skipped": len(skipped),
            "rows": len(out), "path": rel(path)}


# --------------------------------------------------------------------------- BEA

# (NIPA table, frequency, label). Quarterly real activity + price indices.
BEA_TABLES: list[tuple[str, str, str]] = [
    ("T10101", "Q", "Real GDP % change (SAAR)"),
    ("T10109", "Q", "PCE price index % change (SAAR)"),
]


def fetch_bea_table(table: str, freq: str, start_year: int) -> pd.DataFrame:
    if not BEA_KEY:
        raise RuntimeError("BEA_API_KEY not found in .env")
    years = ",".join(str(y) for y in range(start_year, pd.Timestamp.today().year + 1))
    r = _get("https://apps.bea.gov/api/data", {
        "UserID": BEA_KEY, "method": "GetData", "datasetname": "NIPA",
        "TableName": table, "Frequency": freq, "Year": years, "ResultFormat": "JSON"})
    r.raise_for_status()
    res = r.json().get("BEAAPI", {}).get("Results", {})
    data = res.get("Data") if isinstance(res, dict) else None
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    keep = [c for c in ["TableName", "SeriesCode", "LineNumber", "LineDescription",
                        "TimePeriod", "DataValue", "METRIC_NAME"] if c in df.columns]
    df = df[keep].copy()
    df["DataValue"] = pd.to_numeric(df["DataValue"].astype(str).str.replace(",", ""), errors="coerce")
    return df


def job_bea(args: argparse.Namespace) -> dict[str, Any]:
    FREE.mkdir(parents=True, exist_ok=True)
    start_year = int(args.start[:4]) if args.start else 2010
    frames, skipped = [], []
    for table, freq, label in BEA_TABLES:
        try:
            df = fetch_bea_table(table, freq, start_year)
        except Exception as exc:  # noqa: BLE001
            skipped.append((table, _san(str(exc))[:80]))
            continue
        if df.empty:
            skipped.append((table, "no data"))
            continue
        df["table_label"] = label
        frames.append(df)
        print(f"  BEA {table} {label:30s} n={len(df):5d} lines={df['LineNumber'].nunique()}")
        time.sleep(0.2)
    if not frames:
        raise RuntimeError("BEA returned no tables — check BEA_API_KEY / connectivity")
    out = pd.concat(frames, ignore_index=True)
    path = FREE / "bea_nipa.parquet"
    out.to_parquet(path, index=False)
    for table, why in skipped:
        print(f"  BEA skipped {table}: {why}")
    return {"job": "bea", "tables_ok": len(frames), "tables_skipped": len(skipped),
            "rows": len(out), "path": rel(path),
            "caveat": "NIPA values are latest-vintage (revision-prone); for strict PIT use ALFRED."}


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


JOBS = {"fred": job_fred, "eia": job_eia, "cftc": job_cftc, "bea": job_bea}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch free macro + commodity fundamentals (Vector 4)")
    ap.add_argument("--jobs", default="fred,eia,cftc,bea",
                    help=f"Comma-separated jobs from: {','.join(JOBS)}")
    ap.add_argument("--start", default=DEFAULT_START, help="Earliest observation/report date (YYYY-MM-DD)")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    selected = [j.strip() for j in args.jobs.split(",") if j.strip()]
    bad = [j for j in selected if j not in JOBS]
    if bad:
        raise SystemExit(f"unknown job(s): {bad}; choose from {list(JOBS)}")

    summary: list[dict[str, Any]] = []
    for job in selected:
        print(f"\n=== {job.upper()} ===", flush=True)
        try:
            summary.append(JOBS[job](args))
        except Exception as exc:  # noqa: BLE001
            print(f"  {job} FAILED: {_san(str(exc))[:160]}")
            summary.append({"job": job, "error": _san(str(exc))[:160]})

    pd.DataFrame(summary).to_csv(REPORTS / "macro_fundamentals_manifest.csv", index=False)
    print("\n=== summary ===")
    for s in summary:
        print(" ", s)
    print(f"\nManifest: {rel(REPORTS / 'macro_fundamentals_manifest.csv')}")


if __name__ == "__main__":
    main()
