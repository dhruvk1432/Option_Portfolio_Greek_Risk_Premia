"""Point-in-time S&P 500 membership since 2000.

Source: fja05680/sp500 (GitHub) — dated constituent snapshots reconstructed
from S&P press releases / Wikipedia change logs.  Two artifacts:

  data/universe/sp500_components_history.csv   date -> comma-joined tickers
  data/universe/sp500_ticker_start_end.csv     ticker -> [start, end) spells

Symbology: the source uses '.' for share classes (BF.B); Yahoo uses '-'
(BF-B).  `to_yahoo` maps between them.  Membership masks are *point in
time*: a name is investable at month t iff it was in the index on the
last snapshot date <= t (no peeking at future adds/drops).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "universe")


def to_yahoo(ticker: str) -> str:
    return ticker.replace(".", "-")


def load_membership_spells() -> pd.DataFrame:
    """Per-ticker membership spells (a ticker can have several)."""
    se = pd.read_csv(os.path.join(_DIR, "sp500_ticker_start_end.csv"))
    se["start_date"] = pd.to_datetime(se["start_date"])
    se["end_date"] = pd.to_datetime(se["end_date"])  # NaT = still a member
    return se


def load_snapshots(start: str = "2000-01-01") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_DIR, "sp500_components_history.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    df["tickers"] = df["tickers"].str.split(",")
    return df


def all_tickers(start: str = "2000-01-01") -> list[str]:
    """Every ticker that was a member at any snapshot since `start`."""
    snaps = load_snapshots(start)
    out: set[str] = set()
    for row in snaps["tickers"]:
        out.update(row)
    return sorted(out)


def membership_mask(dates: pd.DatetimeIndex,
                    tickers: list[str] | None = None,
                    start: str = "2000-01-01") -> pd.DataFrame:
    """Boolean DataFrame [dates x tickers]: in-index at each date,
    using the last snapshot at or before each date (point in time)."""
    snaps = load_snapshots(start)
    if tickers is None:
        tickers = all_tickers(start)
    tick_ix = {t: j for j, t in enumerate(tickers)}
    mask = np.zeros((len(dates), len(tickers)), dtype=bool)
    snap_dates = snaps["date"].to_numpy()
    # last snapshot <= each requested date
    pos = np.searchsorted(snap_dates, np.asarray(dates, dtype="datetime64[ns]"),
                          side="right") - 1
    for i, p in enumerate(pos):
        if p < 0:
            continue
        for t in snaps["tickers"].iloc[p]:
            j = tick_ix.get(t)
            if j is not None:
                mask[i, j] = True
    return pd.DataFrame(mask, index=dates, columns=tickers)
