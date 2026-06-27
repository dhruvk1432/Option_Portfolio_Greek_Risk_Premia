"""Bulk-download daily OHLCV for every S&P 500 member since 2000.

Run:  .venv/bin/python -m data_ingestion.market_data.fetch_universe_prices

Downloads in chunks via yfinance (credential-free), caches each chunk to
data/universe/chunks/, then assembles:

  data/universe/universe_prices.parquet   adjusted close  [date x ticker]
  data/universe/universe_volume.parquet   share volume    [date x ticker]
  data/universe/universe_coverage.csv     per-ticker first/last date, n_obs

Delisted names that Yahoo no longer serves simply come back empty; the
coverage file is the honest record of survivorship coverage and is
reported in the paper's data section.
"""

from __future__ import annotations

import os
import time
import warnings

import pandas as pd
import yfinance as yf

from data_ingestion.market_data.constituents import all_tickers, to_yahoo

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "universe")
CHUNK_DIR = os.path.join(_DIR, "chunks")
START, END = "1999-12-01", "2026-06-11"
CHUNK = 50


def fetch_chunk(symbols: list[str], retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(symbols, start=START, end=END, interval="1d",
                                 auto_adjust=True, progress=False,
                                 group_by="column", threads=True)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:  # noqa: BLE001 - network layer, retry
            print(f"  retry {attempt + 1}: {type(e).__name__}")
            time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    os.makedirs(CHUNK_DIR, exist_ok=True)
    tickers = all_tickers()
    ymap = {t: to_yahoo(t) for t in tickers}
    print(f"{len(tickers)} tickers in historical membership")

    closes, vols = [], []
    for c0 in range(0, len(tickers), CHUNK):
        batch = tickers[c0:c0 + CHUNK]
        tag = f"chunk_{c0:04d}"
        cpath = os.path.join(CHUNK_DIR, f"{tag}_close.parquet")
        vpath = os.path.join(CHUNK_DIR, f"{tag}_vol.parquet")
        if os.path.exists(cpath):
            closes.append(pd.read_parquet(cpath))
            vols.append(pd.read_parquet(vpath))
            print(f"{tag}: cached")
            continue
        df = fetch_chunk([ymap[t] for t in batch])
        if df is None:
            print(f"{tag}: FAILED")
            continue
        inv = {v: k for k, v in ymap.items()}
        close = df["Close"].rename(columns=inv) if "Close" in df else pd.DataFrame()
        vol = df["Volume"].rename(columns=inv) if "Volume" in df else pd.DataFrame()
        close = close.dropna(axis=1, how="all")
        vol = vol.reindex(columns=close.columns)
        close.to_parquet(cpath)
        vol.to_parquet(vpath)
        closes.append(close)
        vols.append(vol)
        print(f"{tag}: {close.shape[1]}/{len(batch)} tickers, {len(close)} rows")
        time.sleep(1.0)

    px = pd.concat(closes, axis=1).sort_index()
    vv = pd.concat(vols, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    vv = vv.reindex(columns=px.columns)
    px.to_parquet(os.path.join(_DIR, "universe_prices.parquet"))
    vv.to_parquet(os.path.join(_DIR, "universe_volume.parquet"))

    cov = pd.DataFrame({
        "first": px.apply(lambda s: s.first_valid_index()),
        "last": px.apply(lambda s: s.last_valid_index()),
        "n_obs": px.notna().sum(),
    })
    cov.to_csv(os.path.join(_DIR, "universe_coverage.csv"))
    print(f"assembled: {px.shape[0]} days x {px.shape[1]} tickers "
          f"({px.shape[1]}/{len(tickers)} = {px.shape[1] / len(tickers):.0%} coverage)")


if __name__ == "__main__":
    main()
