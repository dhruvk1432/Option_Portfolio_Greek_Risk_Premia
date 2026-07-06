"""Exchange-hours checks for public delayed option-chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
RTH_START = time(9, 30)
RTH_END = time(16, 15)
EARLY_CLOSE_END = time(13, 0)


@dataclass(frozen=True)
class MarketHoursCheck:
    valid: bool
    reason: str
    timestamp_eastern: str
    session_start_eastern: str
    session_end_eastern: str
    timestamp_tz_assumed: str


def classify_cboe_option_rth_timestamp(
    value: object,
    *,
    timestamp_tz: str = "UTC",
) -> MarketHoursCheck:
    """Classify a Cboe delayed-chain timestamp against regular option hours.

    Cboe's public delayed-chain JSON currently emits a chain-level timestamp
    without an explicit offset.  We default to UTC because the CDN timestamp is
    an internet feed timestamp, but keep the assumption explicit in artifacts.
    """

    ts = _to_eastern(value, timestamp_tz=timestamp_tz)
    if ts is pd.NaT:
        return MarketHoursCheck(False, "missing_or_unparseable_timestamp", "", "", "", timestamp_tz)

    session = cboe_option_rth_session(ts.date())
    timestamp_eastern = ts.isoformat()
    if session is None:
        return MarketHoursCheck(False, "no_regular_trading_session", timestamp_eastern, "", "", timestamp_tz)

    start, end = session
    start_ts = pd.Timestamp.combine(ts.date(), start).tz_localize(EASTERN)
    end_ts = pd.Timestamp.combine(ts.date(), end).tz_localize(EASTERN)
    valid = start_ts <= ts <= end_ts
    reason = "regular_trading_hours" if valid else "outside_regular_trading_hours"
    return MarketHoursCheck(valid, reason, timestamp_eastern, start_ts.isoformat(), end_ts.isoformat(), timestamp_tz)


def is_cboe_option_rth_timestamp(value: object, *, timestamp_tz: str = "UTC") -> bool:
    return classify_cboe_option_rth_timestamp(value, timestamp_tz=timestamp_tz).valid


def cboe_option_rth_session(session_date: date) -> tuple[time, time] | None:
    """Return regular-hours bounds for a U.S. options trading date.

    The rule intentionally models the regular session only.  It excludes Cboe
    global trading hours and the 4:15-5:00 p.m. curb session because those are
    not representative liquidity snapshots for single-name option execution.
    """

    if session_date.weekday() >= 5:
        return None
    if session_date in _market_holidays(session_date.year):
        return None
    end = EARLY_CLOSE_END if session_date in _early_closes(session_date.year) else RTH_END
    return RTH_START, end


def _to_eastern(value: object, *, timestamp_tz: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(ZoneInfo(timestamp_tz))
    return out.tz_convert(EASTERN)


@lru_cache(maxsize=None)
def _market_holidays(year: int) -> frozenset[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _good_friday(year),
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    return frozenset(d for d in holidays if d.year == year)


@lru_cache(maxsize=None)
def _early_closes(year: int) -> frozenset[date]:
    candidates = {
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),  # day after Thanksgiving
        date(year, 12, 24),
    }
    july_3 = date(year, 7, 3)
    if july_3.weekday() < 5 and july_3 not in _market_holidays(year):
        candidates.add(july_3)
    return frozenset(d for d in candidates if d.weekday() < 5 and d not in _market_holidays(year))


def _observed(actual: date) -> date:
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year, 12, 31)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _good_friday(year: int) -> date:
    return _easter_sunday(year) - timedelta(days=2)


def _easter_sunday(year: int) -> date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
