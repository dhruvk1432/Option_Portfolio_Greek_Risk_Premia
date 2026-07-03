"""Build a compact CBBO spread cost surface from full-day OPRA NBBO cache."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_CBBO_DIR = Path("data/databento_cache/opra_surface_full_day_cbbo")
DEFAULT_OUT = Path("data/feature_store/cbbo_spread_surface.parquet")
SURFACE_COLUMNS = [
    "underlying",
    "snap_date",
    "moneyness_bucket",
    "tenor_bucket",
    "n_quotes",
    "n_contracts",
    "median_relative_spread",
    "p25_relative_spread",
    "p75_relative_spread",
    "median_mid",
    "median_displayed_size",
]


def parse_osi_symbol(symbol: str) -> tuple[str, pd.Timestamp, str, float]:
    """Parse 21-character OSI symbology into root, expiry, kind, strike."""

    text = str(symbol)
    if len(text) != 21:
        match = re.fullmatch(r"([A-Za-z0-9.\s]{1,6})(\d{6})([CP])(\d{8})", text)
        if not match:
            raise ValueError(f"not a 21-character OSI symbol: {symbol!r}")
        root, expiry_text, kind, strike_text = match.groups()
    else:
        root = text[:6]
        expiry_text = text[6:12]
        kind = text[12]
        strike_text = text[13:21]
    if kind not in {"C", "P"}:
        raise ValueError(f"invalid OSI option kind in {symbol!r}")
    if not expiry_text.isdigit() or not strike_text.isdigit():
        raise ValueError(f"invalid OSI date/strike fields in {symbol!r}")
    expiry = pd.Timestamp(
        year=2000 + int(expiry_text[:2]),
        month=int(expiry_text[2:4]),
        day=int(expiry_text[4:6]),
    )
    return root.strip().upper(), expiry, kind, int(strike_text) / 1000.0


def assign_moneyness_bucket(spot: float, strike: float, kind: str) -> str:
    """Assign the equity option Greek-panel moneyness bucket."""

    _ = kind
    try:
        spot_f = float(spot)
        strike_f = float(strike)
    except (TypeError, ValueError):
        return "other"
    if not np.isfinite([spot_f, strike_f]).all() or spot_f <= 0 or strike_f <= 0:
        return "other"
    log_moneyness = float(np.log(strike_f / spot_f))
    atm = 0.03
    near = 0.10
    wing = 0.20
    eps = 1e-12
    if abs(log_moneyness) <= atm + eps:
        return "atm"
    if -near - eps <= log_moneyness < -atm - eps:
        return "put_near"
    if atm + eps < log_moneyness <= near + eps:
        return "call_near"
    if -wing - eps <= log_moneyness < -near - eps:
        return "put_wing"
    if near + eps < log_moneyness <= wing + eps:
        return "call_wing"
    return "other"


def assign_tenor_bucket(days_to_expiry: float) -> str:
    if days_to_expiry <= 45:
        return "le_45d"
    if days_to_expiry <= 120:
        return "46_120d"
    return "gt_120d"


def _empty_surface() -> pd.DataFrame:
    return pd.DataFrame(columns=SURFACE_COLUMNS)


def _first_present(columns: pd.Index, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _parse_clock_seconds(value: str) -> float:
    ts = pd.Timestamp(f"2000-01-01 {value}")
    return float(ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1_000_000)


def _series_to_snap_date(values: pd.Series, tz: str) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert(tz).dt.tz_localize(None)
    return out.dt.normalize()


def _quote_times_in_window(
    values: pd.Series,
    snap_date: pd.Series,
    *,
    window: tuple[str, str],
    tz: str,
) -> pd.Series:
    quote_time = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(tz)
    quote_date = quote_time.dt.normalize().dt.tz_localize(None)
    seconds = (
        quote_time.dt.hour * 3600
        + quote_time.dt.minute * 60
        + quote_time.dt.second
        + quote_time.dt.microsecond / 1_000_000
    )
    start = _parse_clock_seconds(window[0])
    end = _parse_clock_seconds(window[1])
    if start <= end:
        in_clock_window = seconds.between(start, end, inclusive="both")
    else:
        in_clock_window = seconds.ge(start) | seconds.le(end)
    return quote_date.eq(snap_date) & in_clock_window & quote_time.notna()


def _safe_parse_symbols(symbols: pd.Series) -> pd.DataFrame:
    records = []
    unique_symbols = pd.Series(symbols.dropna().astype(str).unique(), name="symbol")
    for symbol in unique_symbols:
        try:
            underlying, expiry, kind, strike = parse_osi_symbol(symbol)
        except ValueError:
            underlying, expiry, kind, strike = pd.NA, pd.NaT, pd.NA, np.nan
        records.append(
            {
                "symbol": symbol,
                "_osi_underlying": underlying,
                "_osi_expiry": expiry,
                "_osi_kind": kind,
                "_osi_strike": strike,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["symbol", "_osi_underlying", "_osi_expiry", "_osi_kind", "_osi_strike"],
    )


def _prepare_contract_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str)
    parsed = _safe_parse_symbols(out["symbol"])
    out = out.merge(parsed, on="symbol", how="left")
    if "underlying" not in out.columns:
        out["underlying"] = out["_osi_underlying"]
    else:
        out["underlying"] = out["underlying"].fillna(out["_osi_underlying"])
    if "expiry" not in out.columns:
        out["expiry"] = out["_osi_expiry"]
    else:
        out["expiry"] = out["expiry"].fillna(out["_osi_expiry"])
    if "kind" not in out.columns:
        out["kind"] = out["_osi_kind"]
    else:
        out["kind"] = out["kind"].fillna(out["_osi_kind"])
    if "strike" not in out.columns:
        out["strike"] = out["_osi_strike"]
    else:
        out["strike"] = out["strike"].fillna(out["_osi_strike"])
    return out.drop(columns=["_osi_underlying", "_osi_expiry", "_osi_kind", "_osi_strike"])


def build_daily_spread_surface(
    df: pd.DataFrame,
    *,
    window: tuple[str, str] = ("15:30", "16:00"),
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Aggregate valid end-of-day CBBO quotes into daily bucket spread rows."""

    if df.empty:
        return _empty_surface()
    required = ["symbol", "snap_date", "spot"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CBBO frame missing required column(s): {missing}")
    bid_col = _first_present(df.columns, ["bid_px_00", "bid_px", "bid", "bid_price"])
    ask_col = _first_present(df.columns, ["ask_px_00", "ask_px", "ask", "ask_price"])
    bid_size_col = _first_present(df.columns, ["bid_sz_00", "bid_sz", "bid_size"])
    ask_size_col = _first_present(df.columns, ["ask_sz_00", "ask_sz", "ask_size"])
    time_col = _first_present(df.columns, ["ts_recv", "ts_event", "timestamp"])
    missing_quote = [
        name
        for name, col in [
            ("bid", bid_col),
            ("ask", ask_col),
            ("bid size", bid_size_col),
            ("ask size", ask_size_col),
            ("quote time", time_col),
        ]
        if col is None
    ]
    if missing_quote:
        raise ValueError(f"CBBO frame missing required quote field(s): {missing_quote}")

    out = _prepare_contract_columns(df)
    out["snap_date"] = _series_to_snap_date(out["snap_date"], tz)
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    numeric_cols = {
        "spot": "spot",
        "strike": "strike",
        "_bid": bid_col,
        "_ask": ask_col,
        "_bid_size": bid_size_col,
        "_ask_size": ask_size_col,
    }
    for dest, src in numeric_cols.items():
        out[dest] = pd.to_numeric(out[src], errors="coerce")

    in_window = _quote_times_in_window(out[time_col], out["snap_date"], window=window, tz=tz)
    valid = (
        in_window
        & out["_bid"].gt(0)
        & out["_ask"].gt(out["_bid"])
        & out["_bid_size"].gt(0)
        & out["_ask_size"].gt(0)
        & out["spot"].gt(0)
        & out["strike"].gt(0)
        & out["snap_date"].notna()
        & out["expiry"].notna()
        & out["symbol"].notna()
        & out["underlying"].notna()
    )
    out = out.loc[valid].copy()
    if out.empty:
        return _empty_surface()

    out["mid"] = (out["_bid"] + out["_ask"]) / 2.0
    out = out[out["mid"].gt(0)].copy()
    out["relative_spread"] = ((out["_ask"] - out["_bid"]) / out["mid"]).clip(upper=1.5)
    out["displayed_size"] = out[["_bid_size", "_ask_size"]].min(axis=1)
    out["moneyness_bucket"] = [
        assign_moneyness_bucket(spot, strike, kind)
        for spot, strike, kind in zip(out["spot"], out["strike"], out["kind"], strict=False)
    ]
    out["days_to_expiry"] = (out["expiry"].dt.normalize() - out["snap_date"]).dt.days.astype(float)
    out["tenor_bucket"] = out["days_to_expiry"].map(assign_tenor_bucket)

    grouped = out.groupby(["underlying", "snap_date", "moneyness_bucket", "tenor_bucket"], observed=True)
    surface = grouped.agg(
        n_quotes=("relative_spread", "size"),
        n_contracts=("symbol", "nunique"),
        median_relative_spread=("relative_spread", "median"),
        p25_relative_spread=("relative_spread", lambda s: s.quantile(0.25)),
        p75_relative_spread=("relative_spread", lambda s: s.quantile(0.75)),
        median_mid=("mid", "median"),
        median_displayed_size=("displayed_size", "median"),
    )
    surface = surface.reset_index().sort_values(
        ["underlying", "snap_date", "moneyness_bucket", "tenor_bucket"],
        kind="mergesort",
    )
    return surface.loc[:, SURFACE_COLUMNS].reset_index(drop=True)


def _underlying_from_path(path: Path) -> str:
    match = re.fullmatch(r"opra_(.+)_cbbo1m_fullday_(\d{4})\.parquet", path.name)
    if not match:
        raise ValueError(f"unexpected CBBO file name: {path.name}")
    return match.group(1).upper()


def _matching_cbbo_files(cbbo_dir: Path, underlyings: Sequence[str] | None) -> list[Path]:
    wanted = {u.upper() for u in underlyings} if underlyings else None
    files = []
    for path in sorted(cbbo_dir.glob("opra_*_cbbo1m_fullday_*.parquet")):
        underlying = _underlying_from_path(path)
        if wanted is None or underlying in wanted:
            files.append(path)
    return files


def _write_surface(surface: pd.DataFrame, out_path: Path) -> None:
    if out_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite symlinked output path: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_parquet(out_path, index=False)


def _build_cbbo_cost_surface(
    cbbo_dir: Path,
    out_path: Path,
    *,
    underlyings: Sequence[str] | None = None,
    window: tuple[str, str] = ("15:30", "16:00"),
) -> pd.DataFrame:
    files = _matching_cbbo_files(cbbo_dir, underlyings)
    if not files:
        raise FileNotFoundError(f"no full-day CBBO parquet files matched in {cbbo_dir}")

    parts = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["underlying"] = _underlying_from_path(path)
        daily = build_daily_spread_surface(frame, window=window)
        if not daily.empty:
            parts.append(daily)
    surface = (
        pd.concat(parts, ignore_index=True, sort=False)
        if parts
        else _empty_surface()
    )
    if not surface.empty:
        surface = surface.sort_values(
            ["underlying", "snap_date", "moneyness_bucket", "tenor_bucket"],
            kind="mergesort",
        ).reset_index(drop=True)
    _write_surface(surface, out_path)
    return surface


def build_cbbo_cost_surface(
    cbbo_dir: Path,
    out_path: Path,
    underlyings: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build and persist the default close-window CBBO spread cost surface."""

    return _build_cbbo_cost_surface(cbbo_dir, out_path, underlyings=underlyings)


def _parse_underlyings(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = [part.strip().upper() for part in value.split(",") if part.strip()]
    return parsed or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbbo-dir", type=Path, default=DEFAULT_CBBO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--underlyings", help="Comma-separated underlying filter, e.g. SPY,AAPL")
    parser.add_argument("--window-start", default="15:30")
    parser.add_argument("--window-end", default="16:00")
    args = parser.parse_args(argv)

    surface = _build_cbbo_cost_surface(
        args.cbbo_dir,
        args.out,
        underlyings=_parse_underlyings(args.underlyings),
        window=(args.window_start, args.window_end),
    )
    print(f"wrote {len(surface):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
