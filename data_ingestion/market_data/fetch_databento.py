"""Cost-guarded Databento pulls for the expanded model.

Run:  .venv/bin/python -m data_ingestion.market_data.fetch_databento --job [costs|opra|es_minute|bbo|equity_minute]

SECURITY: the API key is loaded into the environment via python-dotenv and
consumed implicitly by databento.Historical(); it is never read, printed,
or logged by this module.

Every download is preceded by metadata.get_cost; a job aborts if its
estimate exceeds its budget.  All estimates and actuals are appended to
research/reports/pipeline_reports/databento_costs.log (amounts only — no credentials).

Jobs
----
opra           Month-end SPY option chains (definition + ohlcv-1d),
               2013-04 .. 2026-05, one batch per year.
es_minute      ES continuous front-month (volume-rolled) ohlcv-1m,
               2010-06 .. 2026-06, one batch per year.
bbo            ES bbo-1s top-of-book for selected calm/stress days
               (effective-spread calibration).
equity_minute  ohlcv-1m for a few large Nasdaq names post-2018
               (XNAS.ITCH), realized-vol cross-check.
"""

from __future__ import annotations

import argparse
import calendar
import os

import pandas as pd
from dotenv import load_dotenv

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE = os.path.join(_ROOT, "data", "databento_cache")
COSTLOG = os.path.join(_ROOT, "research", "reports", "pipeline_reports", "databento_costs.log")

BUDGETS = {"opra": 10.0, "es_minute": 8.0, "bbo": 6.0, "equity_minute": 4.0}

# calm + stress days for spread calibration (ES, RTH-containing sessions)
BBO_DAYS = ["2012-05-15", "2015-08-24", "2017-07-19", "2018-02-06",
            "2020-03-16", "2020-09-15", "2022-06-13", "2024-08-05",
            "2025-04-07", "2026-02-17"]

EQUITY_MINUTE_SYMBOLS = ["AAPL", "NVDA", "AMZN"]


def _client():
    """Implicit credentials; prefers the funded secondary key.

    The primary account hit account_insufficient_funds (2026-06-11);
    DATABENTO_API_KEY2 is the funded fallback.  The value is copied
    env-to-env inside this process only — never read into code,
    printed, or logged.
    """
    load_dotenv(os.path.join(_ROOT, ".env"))
    if os.environ.get("DATABENTO_API_KEY2"):
        os.environ["DATABENTO_API_KEY"] = os.environ["DATABENTO_API_KEY2"]
    if not os.environ.get("DATABENTO_API_KEY"):
        raise SystemExit("credentials unavailable; aborting without bypass")
    import databento as db
    return db.Historical()


def _log_cost(tag: str, est: float, note: str = "") -> None:
    os.makedirs(os.path.dirname(COSTLOG), exist_ok=True)
    with open(COSTLOG, "a") as fh:
        fh.write(f"{tag}\testimate=${est:.4f}\t{note}\n")


def _guarded_get(client, tag: str, budget: float, **kw) -> pd.DataFrame | None:
    est = client.metadata.get_cost(**kw)
    _log_cost(tag, est, str({k: kw[k] for k in ("dataset", "schema", "start", "end")}))
    print(f"  {tag}: estimated ${est:.4f}")
    if est > budget:
        print(f"  SKIP {tag}: estimate ${est:.2f} > budget ${budget:.2f}")
        return None
    data = client.timeseries.get_range(**kw)
    return data.to_df()


def month_ends(start: str, end: str) -> list[str]:
    out = []
    for ts in pd.date_range(start, end, freq="ME"):
        d = ts
        while d.weekday() >= 5:  # back up to a weekday
            d -= pd.Timedelta(days=1)
        out.append(d.strftime("%Y-%m-%d"))
    return out


def _spy_raw_close() -> pd.Series:
    """Unadjusted SPY close (strikes live in raw-price space)."""
    path = os.path.join(_ROOT, "data", "universe", "spy_raw_close.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    import yfinance as yf
    df = yf.download("SPY", start="2012-01-01", end="2026-06-11",
                     auto_adjust=False, progress=False)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.to_csv(path)
    return s


def third_friday(year: int, month: int) -> pd.Timestamp:
    cal = calendar.monthcalendar(year, month)
    fridays = [w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY]]
    return pd.Timestamp(year, month, fridays[2])


def osi(root: str, expiry: pd.Timestamp, kind: str, strike: float) -> str:
    return (f"{root:<6}{expiry.strftime('%y%m%d')}{kind}"
            f"{int(round(strike * 1000)):08d}")


def opra_symbols_for(day: str, spot: float) -> tuple[list[str], pd.Timestamp]:
    """~1M-maturity SPY contracts on a moneyness grid around spot."""
    d = pd.Timestamp(day)
    nm = d + pd.offsets.MonthBegin(1)
    exp = third_friday(nm.year, nm.month)
    if (exp - d).days < 14:  # too close: use following month
        nm = d + pd.offsets.MonthBegin(2)
        exp = third_friday(nm.year, nm.month)
    strikes = sorted({float(round(spot * m)) for m in
                      [0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.965, 0.98,
                       0.99, 1.00, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10]})
    syms = [osi("SPY", exp, k, s) for s in strikes for k in ("C", "P")]
    return syms, exp


def job_opra() -> None:
    """Month-end ~1M SPY smile slices from real OPRA data, 2013-2026."""
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    spy = _spy_raw_close()
    days = month_ends("2013-04-01", "2026-05-31")
    by_year: dict[int, list[str]] = {}
    for d in days:
        by_year.setdefault(int(d[:4]), []).append(d)
    for year, ds in sorted(by_year.items()):
        opath = os.path.join(CACHE, f"opra_spy_slices_{year}.parquet")
        if os.path.exists(opath):
            print(f"{year}: cached")
            continue
        rows = []
        for d in ds:
            ts = pd.Timestamp(d)
            try:
                spot = float(spy.loc[:ts].iloc[-1])
            except (IndexError, KeyError):
                continue
            syms, exp = opra_symbols_for(d, spot)
            nxt = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                b = _guarded_get(client, f"opra-slice-{d}", BUDGETS["opra"],
                                 dataset="OPRA.PILLAR", schema="ohlcv-1d",
                                 symbols=syms, start=d, end=nxt)
            except Exception as e:  # noqa: BLE001 - early years unresolvable
                print(f"  {d}: skipped ({type(e).__name__})")
                continue
            if b is None or not len(b):
                continue
            b = b.reset_index()
            b["snap_date"] = d
            b["spot"] = spot
            b["expiry"] = exp
            rows.append(b)
        if rows:
            pd.concat(rows, ignore_index=True).to_parquet(opath)
            print(f"{year}: {sum(len(r) for r in rows)} contract-days "
                  f"across {len(rows)} snapshots")


def job_es_minute() -> None:
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    for year in range(2010, 2027):
        path = os.path.join(CACHE, f"es_ohlcv1m_{year}.parquet")
        if os.path.exists(path):
            print(f"{year}: cached")
            continue
        start = f"{year}-01-01" if year > 2010 else "2010-06-07"
        end = f"{year}-12-31" if year < 2026 else "2026-06-10"
        df = _guarded_get(client, f"es-1m-{year}", BUDGETS["es_minute"],
                          dataset="GLBX.MDP3", schema="ohlcv-1m",
                          symbols=["ES.v.0"], stype_in="continuous",
                          start=start, end=end)
        if df is not None and len(df):
            df.to_parquet(path)
            print(f"{year}: {len(df)} minute bars")


def job_bbo() -> None:
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    for d in BBO_DAYS:
        path = os.path.join(CACHE, f"es_bbo1s_{d}.parquet")
        if os.path.exists(path):
            continue
        nxt = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = _guarded_get(client, f"es-bbo1s-{d}", BUDGETS["bbo"],
                          dataset="GLBX.MDP3", schema="bbo-1s",
                          symbols=["ES.v.0"], stype_in="continuous",
                          start=d, end=nxt)
        if df is not None and len(df):
            df.to_parquet(path)
            print(f"{d}: {len(df)} bbo rows")


def job_equity_minute() -> None:
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    for sym in EQUITY_MINUTE_SYMBOLS:
        path = os.path.join(CACHE, f"xnas_{sym}_ohlcv1m.parquet")
        if os.path.exists(path):
            continue
        df = _guarded_get(client, f"xnas-1m-{sym}", BUDGETS["equity_minute"],
                          dataset="XNAS.ITCH", schema="ohlcv-1m",
                          symbols=[sym], start="2018-05-01", end="2026-06-10")
        if df is not None and len(df):
            df.to_parquet(path)
            print(f"{sym}: {len(df)} minute bars")


OPRA_MULTI_UNDERLYINGS = ["QQQ", "IWM", "AAPL", "MSFT", "NVDA",
                          "AMZN", "GOOGL", "META", "TSLA", "JPM"]


def _raw_closes(symbols: list[str]) -> pd.DataFrame:
    """Contemporaneous (true, unadjusted) daily closes.

    Yahoo's Close column is retroactively SPLIT-adjusted even with
    auto_adjust=False; option strikes live in the contemporaneous price
    space, so we multiply back the cumulative product of all splits
    occurring AFTER each date (verified: TSLA 2018 ~ $320, AAPL 2015
    ~ $116, AMZN 2015 ~ $430)."""
    path = os.path.join(_ROOT, "data", "universe", "multi_raw_close.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(symbols) <= set(df.columns):
            return df
    import yfinance as yf
    df = yf.download(symbols, start="2014-12-01", end="2026-06-11",
                     auto_adjust=False, progress=False)["Close"]
    for u in df.columns:
        try:
            splits = yf.Ticker(u).splits
        except Exception:  # noqa: BLE001
            continue
        if splits is None or len(splits) == 0:
            continue
        splits.index = pd.to_datetime(splits.index).tz_localize(None)
        for dt, ratio in splits.items():
            if ratio > 0:
                df.loc[df.index < dt, u] *= float(ratio)
    df.to_csv(path)
    return df


def _root_for(underlying: str, day: pd.Timestamp) -> str:
    """OSI root aliases for renames."""
    if underlying == "META" and day < pd.Timestamp("2022-06-09"):
        return "FB"
    return underlying


def _strike_step(price: float) -> float:
    if price < 100:
        return 1.0
    if price < 250:
        return 2.5
    if price < 500:
        return 5.0
    if price < 1000:
        return 10.0
    if price < 2500:
        return 25.0
    return 50.0


def job_opra_multi() -> None:
    """Month-end ~1M smile slices for ten more underlyings, 2015-2026.

    Same construction as the SPY slices: ~16 moneyness points x C/P at
    listed strike increments, ohlcv-1d composites, get_cost-guarded."""
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    raw = _raw_closes(OPRA_MULTI_UNDERLYINGS)
    days = month_ends("2015-01-01", "2026-05-31")
    ms = [0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98, 0.99,
          1.00, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10]
    for u in OPRA_MULTI_UNDERLYINGS:
        by_year: dict[int, list[str]] = {}
        for d in days:
            by_year.setdefault(int(d[:4]), []).append(d)
        for year, ds in sorted(by_year.items()):
            opath = os.path.join(CACHE, f"opra_{u}_slices_{year}.parquet")
            if os.path.exists(opath):
                continue
            rows = []
            for d in ds:
                ts = pd.Timestamp(d)
                try:
                    spot = float(raw[u].loc[:ts].dropna().iloc[-1])
                except (IndexError, KeyError):
                    continue
                nm = ts + pd.offsets.MonthBegin(1)
                exp = third_friday(nm.year, nm.month)
                if (exp - ts).days < 14:
                    nm = ts + pd.offsets.MonthBegin(2)
                    exp = third_friday(nm.year, nm.month)
                step = _strike_step(spot)
                strikes = sorted({round(spot * m / step) * step for m in ms})
                root = _root_for(u, ts)
                syms = [osi(root, exp, k, s) for s in strikes
                        for k in ("C", "P")]
                nxt = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    b = _guarded_get(client, f"opra-{u}-{d}", 5.0,
                                     dataset="OPRA.PILLAR",
                                     schema="ohlcv-1d", symbols=syms,
                                     start=d, end=nxt)
                except Exception as e:  # noqa: BLE001
                    print(f"  {u} {d}: skipped ({type(e).__name__})")
                    continue
                if b is None or not len(b):
                    continue
                b = b.reset_index()
                b["snap_date"] = d
                b["spot"] = spot
                b["expiry"] = exp
                b["underlying"] = u
                rows.append(b)
            if rows:
                pd.concat(rows, ignore_index=True).to_parquet(opath)
                print(f"{u} {year}: {sum(len(r) for r in rows):,} rows, "
                      f"{len(rows)} snapshots")


def job_vix_options() -> None:
    """FULL daily VIX option chains, 2015-2026 (~$0.0125/day, parent
    symbology -- no strike construction needed).  Batched by month."""
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    months = pd.date_range("2015-01-01", "2026-06-01", freq="MS")
    for m0 in months:
        m1 = min(m0 + pd.offsets.MonthBegin(1), pd.Timestamp("2026-06-10"))
        tag = m0.strftime("%Y-%m")
        path = os.path.join(CACHE, f"opra_vix_chain_{tag}.parquet")
        if os.path.exists(path):
            continue
        try:
            b = _guarded_get(client, f"vix-opt-{tag}", 2.0,
                             dataset="OPRA.PILLAR", schema="ohlcv-1d",
                             symbols=["VIX.OPT"], stype_in="parent",
                             start=m0.strftime("%Y-%m-%d"),
                             end=m1.strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            print(f"  {tag}: skipped ({type(e).__name__})")
            continue
        if b is None or not len(b):
            continue
        b = b.reset_index()
        b.to_parquet(path)
        print(f"{tag}: {len(b):,} contract-days")


def job_opra_minute() -> None:
    """Minute trades + minute NBBO for the 32-contract slices, each
    month-end snapshot day, 2015-2026 (~$7.5 total, get_cost-guarded)."""
    client = _client()
    os.makedirs(CACHE, exist_ok=True)
    spy = _spy_raw_close()
    days = month_ends("2015-01-01", "2026-05-31")
    for schema, tag in (("ohlcv-1m", "1m"), ("cbbo-1m", "cbbo1m")):
        by_year: dict[int, list[str]] = {}
        for d in days:
            by_year.setdefault(int(d[:4]), []).append(d)
        for year, ds in sorted(by_year.items()):
            opath = os.path.join(CACHE, f"opra_spy_{tag}_{year}.parquet")
            if os.path.exists(opath):
                print(f"{tag} {year}: cached")
                continue
            rows = []
            for d in ds:
                ts = pd.Timestamp(d)
                try:
                    spot = float(spy.loc[:ts].iloc[-1])
                except (IndexError, KeyError):
                    continue
                syms, exp = opra_symbols_for(d, spot)
                nxt = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    b = _guarded_get(client, f"opra-{tag}-{d}", 5.0,
                                     dataset="OPRA.PILLAR", schema=schema,
                                     symbols=syms, start=d, end=nxt)
                except Exception as e:  # noqa: BLE001
                    print(f"  {d}: skipped ({type(e).__name__})")
                    continue
                if b is None or not len(b):
                    continue
                b = b.reset_index()
                b["snap_date"] = d
                b["spot"] = spot
                b["expiry"] = exp
                rows.append(b)
            if rows:
                pd.concat(rows, ignore_index=True).to_parquet(opath)
                print(f"{tag} {year}: {sum(len(r) for r in rows):,} rows, "
                      f"{len(rows)} snapshots")


def job_universe_minute() -> None:
    """Minute bars for the ~150 most liquid S&P members, 2018-2026.

    Liquidity-ranked so the budget binds on the least useful tail;
    cumulative job cap UNIVERSE_MINUTE_CAP on get_cost estimates."""
    import pandas as pd_  # local alias to appease linters

    from data_ingestion.market_data.constituents import load_snapshots

    cap = 115.0
    client = _client()
    outdir = os.path.join(CACHE, "universe_minute")
    os.makedirs(outdir, exist_ok=True)

    snaps = load_snapshots("2018-01-01")
    members = sorted({t for row in snaps["tickers"] for t in row})
    px = pd_.read_parquet(os.path.join(_ROOT, "data", "universe",
                                       "universe_prices.parquet"))
    vol = pd_.read_parquet(os.path.join(_ROOT, "data", "universe",
                                        "universe_volume.parquet"))
    dollar = (px * vol).loc["2018-01-01":].mean()
    have = [t for t in members if t in dollar.index
            and pd_.notna(dollar[t])]
    ranked = sorted(have, key=lambda t: -float(dollar[t]))[:150]
    print(f"{len(ranked)} symbols, liquidity-ranked")

    spent, manifest = 0.0, []
    stop = False
    # symbol-major (liquidity order), year-minor: each request is
    # ~10 symbols x 1 year (~2M rows max) -- large streams were flaky
    for ci in range(0, len(ranked), 10):
        if stop:
            break
        batch = ranked[ci:ci + 10]
        for year in range(2018, 2027):
            start = "2018-05-01" if year == 2018 else f"{year}-01-01"
            end = f"{year}-12-31" if year < 2026 else "2026-06-10"
            tag = f"b{ci // 10:02d}_{year}"
            path = os.path.join(outdir, f"xnas1m_{tag}.parquet")
            if os.path.exists(path):
                continue
            est = client.metadata.get_cost(
                dataset="XNAS.ITCH", schema="ohlcv-1m", symbols=batch,
                start=start, end=end)
            _log_cost(f"universe-minute-{tag}", est,
                      f"{len(batch)} syms {year}")
            if spent + est > cap:
                print(f"STOP: cumulative ${spent + est:.0f} would exceed "
                      f"${cap:.0f} cap at batch {ci // 10} year {year}")
                stop = True
                break
            df = None
            for attempt in range(3):
                try:
                    df = client.timeseries.get_range(
                        dataset="XNAS.ITCH", schema="ohlcv-1m",
                        symbols=batch, start=start, end=end).to_df()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"{tag}: attempt {attempt + 1} failed "
                          f"{type(e).__name__}: {str(e)[:90]}")
            if df is None:
                continue
            spent += est
            df.to_parquet(path)
            got = sorted(df["symbol"].unique()) if len(df) else []
            manifest += [dict(symbol=s, year=year) for s in got]
            print(f"{tag}: {len(df):,} bars, {len(got)}/{len(batch)} syms, "
                  f"est ${est:.2f} (cum ${spent:.2f})")
    pd_.DataFrame(manifest).to_csv(
        os.path.join(outdir, "manifest.csv"), index=False)
    print(f"done: estimated job spend ${spent:.2f}")


def job_costs() -> None:
    """Estimate everything, download nothing."""
    client = _client()
    probes = [
        dict(dataset="OPRA.PILLAR", schema="ohlcv-1d", symbols=["SPY.OPT"],
             stype_in="parent", start="2024-06-28", end="2024-06-29"),
        dict(dataset="OPRA.PILLAR", schema="definition", symbols=["SPY.OPT"],
             stype_in="parent", start="2024-06-28", end="2024-06-29"),
        dict(dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=["ES.v.0"],
             stype_in="continuous", start="2024-01-01", end="2024-12-31"),
        dict(dataset="GLBX.MDP3", schema="bbo-1s", symbols=["ES.v.0"],
             stype_in="continuous", start="2020-03-16", end="2020-03-17"),
        dict(dataset="XNAS.ITCH", schema="ohlcv-1m", symbols=["AAPL"],
             start="2018-05-01", end="2026-06-10"),
    ]
    for p in probes:
        est = client.metadata.get_cost(**p)
        print(f"{p['dataset']}/{p['schema']} {p['start']}..{p['end']}: ${est:.4f}")
        _log_cost("probe", est, f"{p['dataset']}/{p['schema']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True,
                    choices=["costs", "opra", "es_minute", "bbo",
                             "equity_minute", "universe_minute", "opra_minute",
                             "opra_multi", "vix_options"])
    job = ap.parse_args().job
    {"costs": job_costs, "opra": job_opra, "es_minute": job_es_minute,
     "bbo": job_bbo, "equity_minute": job_equity_minute,
     "universe_minute": job_universe_minute,
     "opra_minute": job_opra_minute,
     "opra_multi": job_opra_multi,
     "vix_options": job_vix_options}[job]()
