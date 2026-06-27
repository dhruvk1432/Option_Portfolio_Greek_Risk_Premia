"""The VIX complex: free daily OHLC for every Cboe vol index + VX futures.

Run:  .venv/bin/python -m data_ingestion.market_data.fetch_vix_complex

(1) yfinance (free): daily OHLC, maximum history, for the volatility
    index family --- VIX (spot, 1990-), VIX9D/VIX3M/VIX6M (term
    structure points), VVIX (vol of vol), SKEW, VXN (Nasdaq), RVX
    (Russell), OVX (oil), GVZ (gold).
    -> data/universe/vix_complex.parquet  (long: date x [index, OHLC])

(2) Cboe CDN (free): historical VX futures daily settlement files per
    contract (CFE archive), giving the actual tradable term structure.
    -> data/universe/vx_futures_daily.parquet

VIX *options* are pulled separately via Databento OPRA
(fetch_databento --job vix_options) since no free historical source
exists.
"""

from __future__ import annotations

import io
import os
import time
import warnings

import pandas as pd
import requests
import yfinance as yf

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "universe")

INDICES = ["^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW",
           "^VXN", "^RVX", "^OVX", "^GVZ"]

CBOE_URL = ("https://cdn.cboe.com/data/us/futures/market_statistics/"
            "historical_data/VX/VX_{date}.csv")
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def fetch_indices() -> pd.DataFrame:
    frames = []
    for sym in INDICES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(sym, period="max", interval="1d",
                             auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"{sym}: EMPTY")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df["index_name"] = sym.lstrip("^")
        df = df.reset_index().rename(columns=str.lower)
        frames.append(df)
        print(f"{sym}: {len(df)} days ({df['date'].min().date()} -> "
              f"{df['date'].max().date()})")
        time.sleep(0.4)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(os.path.join(_DIR, "vix_complex.parquet"))
    return out


def fetch_vx_futures(start_year: int = 2013) -> pd.DataFrame | None:
    """Per-expiry VX daily files from the Cboe CDN (free).

    The CDN serves one CSV per contract keyed by final settlement date;
    we walk monthly expiries (Wednesday ~30d before next month's SPX
    opex; in practice Cboe keys by the actual settlement date, so we
    try the Wednesdays of each month).
    """
    frames = []
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (research; data archive)"
    for year in range(start_year, 2027):
        for month in range(1, 13):
            # VX settles Wednesday mornings; try all Wednesdays
            days = pd.date_range(f"{year}-{month:02d}-01",
                                 periods=31, freq="D")
            wednesdays = [d for d in days
                          if d.weekday() == 2 and d.month == month]
            got = None
            for w in wednesdays:
                url = CBOE_URL.format(date=w.strftime("%Y-%m-%d"))
                try:
                    r = sess.get(url, timeout=20)
                except requests.RequestException:
                    continue
                if r.status_code == 200 and len(r.content) > 200:
                    got = (w, r.content)
                    break
            if got is None:
                continue
            w, content = got
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:  # noqa: BLE001
                continue
            df["contract"] = f"VX{MONTH_CODES[month]}{str(year)[-2:]}"
            df["settlement_date"] = w
            frames.append(df)
        print(f"{year}: {sum(1 for f in frames if f['settlement_date'].iloc[0].year == year)} contracts")
    if not frames:
        print("Cboe CDN yielded nothing")
        return None
    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    out.to_parquet(os.path.join(_DIR, "vx_futures_daily.parquet"))
    print(f"VX futures: {len(out):,} rows, "
          f"{out['contract'].nunique()} contracts")
    return out


if __name__ == "__main__":
    fetch_indices()
    fetch_vx_futures()
