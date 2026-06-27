"""Expanded option-chain snapshots for the liquid S&P subset.

Run:  .venv/bin/python -m data_ingestion.market_data.fetch_option_chains

Pulls current listed chains (yfinance, credential-free) for the most
liquid current S&P 500 names plus index ETFs; keeps ~1M/2M/3M/6M
expiries.  Output: data/universe/chains_expanded.parquet — used for
per-name SVI surface calibration (the cross-section of smiles), not for
historical backtesting (that role belongs to the OPRA slices).
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "universe")
ETFS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
N_NAMES = 60
TARGET_DAYS = [30, 60, 91, 182]


def liquid_names() -> list[str]:
    px = pd.read_parquet(os.path.join(_DIR, "universe_prices.parquet"))
    vol = pd.read_parquet(os.path.join(_DIR, "universe_volume.parquet"))
    dollar = (px * vol).tail(63).mean()
    live = px.tail(5).notna().any()
    dollar = dollar[live[live].index].dropna()
    return dollar.sort_values(ascending=False).head(N_NAMES).index.tolist()


def fetch_chain(symbol: str) -> pd.DataFrame | None:
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options
        if not expiries:
            return None
        today = pd.Timestamp.now().normalize()
        exp_ts = pd.to_datetime(list(expiries))
        chosen = []
        for tgt in TARGET_DAYS:
            i = int(np.argmin(np.abs((exp_ts - today).days - tgt)))
            if expiries[i] not in chosen:
                chosen.append(expiries[i])
        spot = tk.fast_info.get("lastPrice") or tk.fast_info.get("last_price")
        frames = []
        for e in chosen:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oc = tk.option_chain(e)
            for kind, df in (("C", oc.calls), ("P", oc.puts)):
                df = df[["strike", "bid", "ask", "lastPrice", "volume",
                         "openInterest", "impliedVolatility"]].copy()
                df["kind"], df["expiry"], df["underlying"] = kind, e, symbol
                df["spot"] = spot
                frames.append(df)
        return pd.concat(frames, ignore_index=True)
    except Exception as e:  # noqa: BLE001 - skip names with broken chains
        print(f"  {symbol}: {type(e).__name__}")
        return None


def main() -> None:
    names = ETFS + [n for n in liquid_names() if n not in ETFS]
    print(f"fetching chains for {len(names)} underlyings")
    out = []
    for i, sym in enumerate(names):
        df = fetch_chain(sym)
        if df is not None:
            out.append(df)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(names)} done ({sum(len(d) for d in out)} rows)")
        time.sleep(0.5)
    allc = pd.concat(out, ignore_index=True)
    allc["snap_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    allc.to_parquet(os.path.join(_DIR, "chains_expanded.parquet"))
    print(f"saved {len(allc)} rows, {allc.underlying.nunique()} underlyings")


if __name__ == "__main__":
    main()
