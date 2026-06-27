"""Repair universe_prices.parquet by masking out-of-membership-window cells to NaN.

WHY: the parquet was fetched (yfinance) over the full 1999-2026 calendar for the whole
historical S&P 500 ticker union, without delisting handling. For tickers that had left the
index, the feed returned recycled/placeholder garbage for the post-delisting dead zone --
e.g. CBE (Cooper Industries, delisted 2012) flip-flops $0.005<->$170 throughout 2015-2017,
fabricating +3,399,900% one-day "returns". 85% of all >1000% single-day returns, and 100%
of those inside the 2016+ momentum window, fall OUTSIDE the ticker's valid membership window.

THE FIX (authoritative, deterministic, offline): a name that is not in the index has no
price. Using the membership windows in ``sp500_ticker_start_end.csv``, set every cell outside
a ticker's [start_date, end_date] window(s) to NaN. ``end_date`` is NaN for still-active
members (open-ended). Tickers absent from the window table are left untouched (fail-safe) and
warned about, so the tool never silently destroys data it cannot classify.

Residual in-window data errors (~15% of spikes, e.g. MEE) are out of scope here -- they are a
separate, smaller issue best caught by the engine's value-level ingestion guard.

    python -m data_ingestion.repair_universe_prices   # universe_prices_raw.parquet -> universe_prices.parquet

``universe_prices_raw.parquet`` is the untouched yfinance fetch (the archive); the cleaned,
window-masked result is written to ``universe_prices.parquet`` (the default every backtest loads),
so the build is fully reproducible from the raw archive + the membership table.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_DATA_UNIVERSE = Path(__file__).resolve().parents[1] / "data" / "universe"


def load_windows(csv_path) -> pd.DataFrame:
    """Read the membership table into ``ticker / start_date / end_date`` (dates parsed)."""
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df[["ticker", "start_date", "end_date"]]


def window_mask(prices: pd.DataFrame, windows: pd.DataFrame, *, open_end=None) -> pd.DataFrame:
    """Return ``prices`` with every cell outside a ticker's membership window(s) set to NaN.

    - A ticker may have several windows (multiple index stints); a cell is kept if it lies in
      ANY of them, inclusive of both bounds.
    - ``end_date`` NaT means still active -> the window runs to ``open_end`` (default: the last
      date in ``prices``).
    - A ticker present in ``prices`` but absent from ``windows`` is left untouched and warned
      about (a repair must not destroy data it cannot classify).
    Shape, index and columns are preserved exactly.
    """
    idx = prices.index if isinstance(prices.index, pd.DatetimeIndex) else pd.to_datetime(prices.index)
    open_end = pd.Timestamp(open_end) if open_end is not None else idx.max()

    w = windows.copy()
    w["start_date"] = pd.to_datetime(w["start_date"], errors="coerce")
    w["end_date"] = pd.to_datetime(w["end_date"], errors="coerce").fillna(open_end)

    in_window = {}
    for tk, grp in w.groupby("ticker"):
        col = np.zeros(len(idx), dtype=bool)
        for s, e in zip(grp["start_date"], grp["end_date"]):
            if pd.isna(s):
                continue
            col |= np.asarray(idx >= s) & np.asarray(idx <= e)
        in_window[tk] = col

    keep = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    unmatched = []
    for c in prices.columns:
        if c in in_window:
            keep[c] = in_window[c]
        else:
            keep[c] = True  # fail-safe: leave unclassifiable tickers as-is
            unmatched.append(c)
    if unmatched:
        warnings.warn(
            f"window_mask: {len(unmatched)} ticker(s) had no membership window and were left "
            f"untouched: {unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")

    return prices.where(keep)


def repair(prices_path, windows_csv, out_path) -> dict:
    """Mask ``prices_path`` to its membership windows and write a clean parquet. Non-destructive."""
    prices = pd.read_parquet(prices_path).select_dtypes("number")
    clean = window_mask(prices, load_windows(windows_csv))

    live_before = int(prices.notna().to_numpy().sum())
    masked = int((prices.notna() & clean.isna()).to_numpy().sum())
    rr_before = int((prices.pct_change(fill_method=None) > 10.0).to_numpy().sum())
    rr_after = int((clean.pct_change(fill_method=None) > 10.0).to_numpy().sum())
    clean.to_parquet(out_path)
    return {
        "out": str(out_path), "tickers": clean.shape[1], "dates": clean.shape[0],
        "live_cells_before": live_before, "cells_masked": masked,
        "pct_cells_masked": round(masked / live_before * 100, 2) if live_before else 0.0,
        "extreme_returns_before": rr_before, "extreme_returns_after": rr_after,
    }


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Mask out-of-membership-window prices to NaN.")
    p.add_argument("--prices", default=str(_DATA_UNIVERSE / "universe_prices_raw.parquet"))
    p.add_argument("--windows", default=str(_DATA_UNIVERSE / "sp500_ticker_start_end.csv"))
    p.add_argument("--out", default=str(_DATA_UNIVERSE / "universe_prices.parquet"))
    a = p.parse_args(argv)
    rep = repair(a.prices, a.windows, a.out)
    print(f"wrote {rep['out']}  ({rep['dates']} dates x {rep['tickers']} tickers)")
    print(f"  masked {rep['cells_masked']:,} of {rep['live_cells_before']:,} live cells "
          f"({rep['pct_cells_masked']}%) as out-of-window")
    print(f"  single-day returns >1000x: {rep['extreme_returns_before']:,} -> {rep['extreme_returns_after']:,}")
    return rep


if __name__ == "__main__":
    main()
