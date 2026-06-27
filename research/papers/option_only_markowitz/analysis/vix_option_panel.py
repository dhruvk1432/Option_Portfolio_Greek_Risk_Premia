"""Contract-level VIX option panel for the option-only Markowitz paper.

The functions in this module intentionally do not use the daily-median VIX option
feature-store panel.  A trading backtest needs contract identity, listed expiry,
right, strike, entry mark, and a settlement source flag.  VIX options are priced
with Black-76 Greeks against a matched VX futures forward; VIX spot and VVIX are
state variables, not tradable underlyings.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[4]
VIX_FACTOR = "VX_FRONT"
_OSI_RE = re.compile(r"^(?P<root>.{1,6})\s*(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_osi_symbol(symbol: object) -> tuple[str | None, pd.Timestamp | None, str | None, float | None]:
    """Parse an OPRA OSI-like symbol such as ``VIX   260617C00010000``."""

    if not isinstance(symbol, str):
        return None, None, None, None
    s = symbol.strip()
    m = _OSI_RE.match(s)
    if not m and len(s) >= 21 and s[6:12].isdigit() and s[12:13] in {"C", "P"} and s[13:21].isdigit():
        root, ymd, cp, strike = s[:6].strip(), s[6:12], s[12], s[13:21]
    elif m:
        root, ymd, cp, strike = m.group("root").strip(), m.group("ymd"), m.group("cp"), m.group("strike")
    else:
        return None, None, None, None
    try:
        expiry = pd.Timestamp(f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}")
    except Exception:
        expiry = None
    return root or None, expiry, "call" if cp == "C" else "put", int(strike) / 1000.0


def stack_vix_option_shards(root: Path = ROOT) -> pd.DataFrame:
    """Load and normalize raw Databento OPRA VIX option shards."""

    files = sorted((root / "data" / "databento_cache").glob("opra_vix_chain_*.parquet"))
    frames: list[pd.DataFrame] = []
    for source_rank, path in enumerate(files):
        df = pd.read_parquet(path)
        if df.empty or "symbol" not in df or "ts_event" not in df:
            continue
        parsed = [parse_osi_symbol(x) for x in df["symbol"]]
        out = df.copy()
        out[["root", "expiry", "kind", "strike"]] = pd.DataFrame(parsed, index=out.index)
        out["trade_date"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out["volume"] = pd.to_numeric(out.get("volume", 0.0), errors="coerce").fillna(0.0)
        out["source_file"] = path.name
        out["source_rank"] = source_rank
        out["row_rank"] = np.arange(len(out), dtype=int)
        out = out[
            out["root"].astype(str).str.upper().eq("VIX")
            & out["trade_date"].notna()
            & out["expiry"].notna()
            & out["strike"].gt(0)
            & out["close"].gt(0)
        ].copy()
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = panel.sort_values(["trade_date", "symbol", "source_rank", "row_rank"])
    panel = panel.drop_duplicates(["trade_date", "symbol"], keep="last")
    panel["tenor_days"] = (panel["expiry"] - panel["trade_date"]).dt.days
    return panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def load_vix_complex(root: Path = ROOT) -> pd.DataFrame:
    path = root / "data" / "universe" / "vix_complex.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date")


def vix_close_series(root: Path = ROOT, name: str = "VIX") -> pd.Series:
    df = load_vix_complex(root)
    if df.empty:
        return pd.Series(dtype=float, name=name)
    sub = df[df["index_name"].astype(str).str.upper().eq(name.upper())]
    if sub.empty:
        return pd.Series(dtype=float, name=name)
    out = pd.Series(sub["close"].to_numpy(float), index=sub["date"], name=name).sort_index()
    return out[~out.index.duplicated(keep="last")]


def vx_curve(root: Path = ROOT) -> pd.DataFrame:
    path = root / "data" / "universe" / "vx_futures_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    vx = pd.read_parquet(path)
    if vx.empty:
        return pd.DataFrame()
    vx["trade_date"] = pd.to_datetime(vx["trade_date"], errors="coerce").dt.normalize()
    vx["settlement_date"] = pd.to_datetime(vx["settlement_date"], errors="coerce").dt.normalize()
    settle = pd.to_numeric(vx.get("settle", np.nan), errors="coerce")
    close = pd.to_numeric(vx.get("close", np.nan), errors="coerce")
    vx["forward_price"] = settle.where(settle.gt(0), close)
    vx = vx.dropna(subset=["trade_date", "settlement_date", "forward_price"])
    vx = vx[vx["forward_price"].gt(0)].copy()
    return vx.sort_values(["trade_date", "settlement_date", "total_volume"])


def vx_forward_lookup(root: Path = ROOT) -> dict[pd.Timestamp, pd.DataFrame]:
    vx = vx_curve(root)
    if vx.empty:
        return {}
    return {pd.Timestamp(d).normalize(): g.copy() for d, g in vx.groupby("trade_date", observed=True)}


def align_vx_forward(
    dates: pd.Series,
    expiries: pd.Series,
    root: Path = ROOT,
    lookup: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return matched VX forward price and contract for each date/expiry pair."""

    lookup = vx_forward_lookup(root) if lookup is None else lookup
    if not lookup:
        return pd.DataFrame({"vix_forward": np.nan, "vx_contract": ""}, index=dates.index)
    available = np.array(sorted(lookup.keys()), dtype="datetime64[ns]")
    pairs = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce").dt.normalize(),
            "expiry": pd.to_datetime(expiries, errors="coerce").dt.normalize(),
        },
        index=dates.index,
    )
    cache: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[float, str]] = {}
    for _, pair in pairs.dropna().drop_duplicates().iterrows():
        d = pd.Timestamp(pair["date"]).normalize()
        e = pd.Timestamp(pair["expiry"]).normalize()
        chosen = d
        if chosen not in lookup:
            pos = np.searchsorted(available, np.datetime64(d), side="right") - 1
            if pos < 0:
                cache[(d, e)] = (np.nan, "")
                continue
            chosen = pd.Timestamp(available[pos]).normalize()
        curve = lookup.get(chosen, pd.DataFrame())
        eligible = curve[curve["settlement_date"].ge(d + pd.Timedelta(days=1))].copy()
        if eligible.empty:
            cache[(d, e)] = (np.nan, "")
            continue
        distances = (eligible["settlement_date"] - e).abs().dt.days.to_numpy()
        row = eligible.iloc[int(np.argmin(distances))]
        cache[(d, e)] = (float(row["forward_price"]), str(row.get("contract", row.get("futures", ""))))
    rows = []
    for _, pair in pairs.iterrows():
        if pd.isna(pair["date"]) or pd.isna(pair["expiry"]):
            rows.append((np.nan, ""))
        else:
            rows.append(cache.get((pd.Timestamp(pair["date"]).normalize(), pd.Timestamp(pair["expiry"]).normalize()), (np.nan, "")))
    return pd.DataFrame(rows, index=dates.index, columns=["vix_forward", "vx_contract"])


def _black76_d1d2(forward: np.ndarray, strike: np.ndarray, tenor: np.ndarray, vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.maximum(np.asarray(forward, dtype=float), 1e-12)
    k = np.maximum(np.asarray(strike, dtype=float), 1e-12)
    t = np.maximum(np.asarray(tenor, dtype=float), 1e-12)
    v = np.maximum(np.asarray(vol, dtype=float), 1e-8)
    sigt = v * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * v * v * t) / sigt
    return d1, d1 - sigt


def black76_price(forward: float, strike: float, tenor: float, rate: float, vol: float, kind: str) -> float:
    d1, d2 = _black76_d1d2(np.array([forward]), np.array([strike]), np.array([tenor]), np.array([vol]))
    df = math.exp(-rate * max(tenor, 0.0))
    if kind == "call":
        return float(df * (forward * stats.norm.cdf(d1[0]) - strike * stats.norm.cdf(d2[0])))
    if kind == "put":
        return float(df * (strike * stats.norm.cdf(-d2[0]) - forward * stats.norm.cdf(-d1[0])))
    raise ValueError("kind must be 'call' or 'put'")


def black76_greeks(forward: float, strike: float, tenor: float, rate: float, vol: float, kind: str) -> dict[str, float]:
    d1, _ = _black76_d1d2(np.array([forward]), np.array([strike]), np.array([tenor]), np.array([vol]))
    d1 = float(d1[0])
    t = max(float(tenor), 1e-12)
    f = max(float(forward), 1e-12)
    v = max(float(vol), 1e-8)
    df = math.exp(-float(rate) * t)
    pdf = stats.norm.pdf(d1)
    delta = df * stats.norm.cdf(d1) if kind == "call" else -df * stats.norm.cdf(-d1)
    gamma = df * pdf / (f * v * math.sqrt(t))
    vega = df * f * pdf * math.sqrt(t)
    dt = min(1 / 365.0, 0.5 * t)
    p0 = black76_price(forward, strike, t, rate, v, kind)
    p1 = black76_price(forward, strike, max(t - dt, 1e-8), rate, v, kind)
    theta = (p1 - p0) / dt
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


def implied_vol_black76(price: float, forward: float, strike: float, tenor: float, rate: float, kind: str) -> float:
    if not all(np.isfinite([price, forward, strike, tenor, rate])) or price <= 0 or forward <= 0 or strike <= 0 or tenor <= 0:
        return float("nan")
    try:
        lo, hi = 1e-4, 5.0
        f = lambda sig: black76_price(forward, strike, tenor, rate, sig, kind) - price
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(optimize.brentq(f, lo, hi, xtol=1e-8, maxiter=100))
    except Exception:
        return float("nan")


def add_black76_columns(panel: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    out = panel.copy()
    aligned = align_vx_forward(out["trade_date"], out["expiry"], root=root)
    for col in ["vix_forward", "vx_contract"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = pd.concat([out, aligned], axis=1)
    out["tenor_years"] = out["tenor_days"] / 365.0
    out["rate"] = 0.02
    vols = []
    greeks = []
    for row in out.itertuples(index=False):
        iv = implied_vol_black76(float(row.close), float(row.vix_forward), float(row.strike), float(row.tenor_years), float(row.rate), str(row.kind))
        vols.append(iv)
        if np.isfinite(iv):
            greeks.append(black76_greeks(float(row.vix_forward), float(row.strike), float(row.tenor_years), float(row.rate), iv, str(row.kind)))
        else:
            greeks.append({"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan})
    out["iv_proxy"] = vols
    gdf = pd.DataFrame(greeks, index=out.index)
    for col in ["delta", "gamma", "vega", "theta"]:
        out[col] = gdf[col]
    out["greek_model"] = np.where(out["iv_proxy"].notna(), "black76_vx_forward", "missing_black76")
    return out


def _bucketize(log_moneyness: pd.Series, kind: pd.Series) -> pd.Series:
    lm = pd.to_numeric(log_moneyness, errors="coerce")
    k = kind.astype(str)
    bucket = pd.Series(pd.NA, index=lm.index, dtype="object")
    bucket[lm.abs().le(0.05)] = "vix_atm"
    bucket[k.eq("call") & lm.gt(0.05) & lm.le(0.25)] = "vix_call_near"
    bucket[k.eq("call") & lm.gt(0.25) & lm.le(0.75)] = "vix_call_wing"
    bucket[k.eq("put") & lm.lt(-0.05) & lm.ge(-0.35)] = "vix_put_near"
    return bucket


def _vro_candidate_paths(root: Path = ROOT) -> list[Path]:
    """Return configured and default exact VRO/SOQ settlement files."""

    candidates: list[Path] = []
    env_file = os.environ.get("OPTION_MARKOWITZ_VRO_FILE")
    env_dir = os.environ.get("OPTION_MARKOWITZ_VRO_DIR")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    if env_dir:
        d = Path(env_dir).expanduser()
        candidates.extend(list(d.glob("**/*vro*.*")) + list(d.glob("**/*VRO*.*")) + list(d.glob("**/*soq*.*")) + list(d.glob("**/*SOQ*.*")))
    candidates.extend(list((root / "data").glob("**/*vro*.*")))
    candidates.extend(list((root / "data").glob("**/*VRO*.*")))
    candidates.extend(list((root / "data").glob("**/*soq*.*")))
    candidates.extend(list((root / "data").glob("**/*SOQ*.*")))
    seen = set()
    out = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved not in seen and path.exists() and path.is_file():
            seen.add(resolved)
            out.append(path)
    return out


def load_vro_series(root: Path = ROOT) -> pd.Series:
    """Load exact VIX VRO/SOQ settlement values from configured local files.

    The series is indexed by settlement/expiration date. It is deliberately not
    forward-filled: if an expiry date is missing, the VIX option holding ledger
    must remain a settlement proxy rather than silently using an adjacent value.
    """

    frames = []
    for path in _vro_candidate_paths(root):
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                continue
        except Exception:
            continue
        lower = {c.lower(): c for c in df.columns}
        date_col = next((lower[c] for c in ["settlement_date", "expiration", "date", "trade_date"] if c in lower), None)
        val_col = next((lower[c] for c in ["settlement_value", "vro", "soq", "settle", "settlement", "value"] if c in lower), None)
        if date_col and val_col:
            dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
            vals = pd.to_numeric(df[val_col], errors="coerce")
            one = pd.DataFrame({"date": dates, "value": vals, "source_file": path.name}).dropna(subset=["date", "value"])
            one = one[one["value"].gt(0)]
            frames.append(one)
    if not frames:
        return pd.Series(dtype=float, name="VRO")
    table = pd.concat(frames, ignore_index=True, sort=False).sort_values(["date", "source_file"])
    table = table.drop_duplicates("date", keep="last")
    out = pd.Series(table["value"].to_numpy(float), index=table["date"], name="VRO").sort_index()
    return out[~out.index.duplicated(keep="last")]


def _last_value_on_or_before(series: pd.Series, target: pd.Timestamp, lower: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    if series.empty:
        return None, None
    candidates = series.loc[(series.index <= target) & (series.index >= lower)].dropna()
    if candidates.empty:
        return None, None
    dt = pd.Timestamp(candidates.index[-1]).normalize()
    return dt, float(candidates.iloc[-1])


def build_vix_option_bucket_panel(
    rebalance_dates: Sequence[pd.Timestamp],
    root: Path = ROOT,
    min_mark: float = 0.25,
    min_volume: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw panel, representatives, returns, holding ledger, and audit rows.

    The function aligns all raw rows to VX forwards, but it computes expensive
    Black-76 implied vols/Greeks only after monthly representative selection.
    """

    raw = stack_vix_option_shards(root)
    if raw.empty:
        return raw, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), vix_data_audit(raw, pd.DataFrame(), pd.DataFrame())
    panel = raw.copy()
    aligned = align_vx_forward(panel["trade_date"], panel["expiry"], root=root)
    panel = pd.concat([panel, aligned], axis=1)
    panel = panel[
        panel["close"].ge(min_mark)
        & panel["volume"].ge(min_volume)
        & panel["tenor_days"].between(7, 45)
        & panel["vix_forward"].gt(0)
    ].copy()
    panel["log_moneyness"] = np.log(panel["strike"] / panel["vix_forward"]).replace([np.inf, -np.inf], np.nan)
    panel["moneyness_bucket"] = _bucketize(panel["log_moneyness"], panel["kind"])
    panel = panel[panel["moneyness_bucket"].notna()].copy()
    panel["asset_id"] = "VIX_" + panel["kind"].astype(str) + "_" + panel["moneyness_bucket"].astype(str)
    panel["snap_date_source"] = panel["trade_date"]
    panel["underlying"] = VIX_FACTOR
    panel["underlying_or_forward"] = VIX_FACTOR
    panel["spot"] = panel["vix_forward"]
    panel["mark"] = panel["close"].astype(float)
    panel["asset_class"] = "vix_option"

    rebal = sorted(pd.to_datetime(pd.Index(rebalance_dates)).dropna().unique())
    rows = []
    for decision_date in rebal:
        d = pd.Timestamp(decision_date).normalize()
        hist = panel[panel["trade_date"].le(d)].copy()
        if hist.empty:
            continue
        source_date = hist["trade_date"].max()
        if (d - source_date).days > 5:
            continue
        day = hist[hist["trade_date"].eq(source_date)].copy()
        day["tenor_distance"] = (day["tenor_days"] - 30).abs()
        day["abs_moneyness"] = day["log_moneyness"].abs()
        day = day.sort_values(
            ["asset_id", "volume", "tenor_distance", "abs_moneyness"],
            ascending=[True, False, True, True],
        )
        chosen = day.groupby("asset_id", as_index=False).head(1).copy()
        chosen["snap_date"] = d
        rows.append(chosen)
    reps = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if reps.empty:
        return panel, reps, pd.DataFrame(), pd.DataFrame(), vix_data_audit(panel, reps, pd.DataFrame())

    reps = add_black76_columns(reps, root=root)
    reps = reps[reps["iv_proxy"].notna()].copy()
    returns, detail = build_vix_expiry_proxy_returns(reps, rebal, root=root)
    return panel, reps, returns, detail, vix_data_audit(panel, reps, detail)


def build_vix_expiry_proxy_returns(
    reps: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    root: Path = ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(pd.Index(rebalance_dates)).dropna().unique())
    next_date = dict(zip(dates[:-1], dates[1:]))
    vix = vix_close_series(root, "VIX")
    vro = load_vro_series(root)
    exact_mode = not vro.empty
    vx_lookup = vx_forward_lookup(root)
    rows = []
    for _, row in reps.iterrows():
        decision_date = pd.Timestamp(row["snap_date"]).normalize()
        realization_date = next_date.get(decision_date)
        if realization_date is None:
            continue
        expiry = pd.Timestamp(row["expiry"]).normalize()
        proxy_settle_date, proxy_value = _last_value_on_or_before(vix, expiry, decision_date)
        if not vro.empty and expiry in vro.index:
            settle_date = expiry
            settlement_value = float(vro.loc[expiry])
            settlement_source = "vro_soq_exact"
        elif exact_mode:
            continue
        else:
            settle_date, settlement_value = proxy_settle_date, proxy_value
            settlement_source = "vix_close_settlement_proxy"
        if settle_date is None or settlement_value is None:
            continue
        fwd = align_vx_forward(pd.Series([settle_date]), pd.Series([expiry]), root=root, lookup=vx_lookup)
        terminal_forward = float(fwd["vix_forward"].iloc[0]) if fwd["vix_forward"].notna().iloc[0] else float(settlement_value)
        strike = float(row["strike"])
        mark = float(row["mark"])
        if row["kind"] == "call":
            payoff = max(float(settlement_value) - strike, 0.0)
        else:
            payoff = max(strike - float(settlement_value), 0.0)
        rows.append(
            {
                "return_date": realization_date,
                "decision_date": decision_date,
                "state_snapshot_date": pd.Timestamp(row.get("snap_date_source", decision_date)).normalize(),
                "expiry": expiry,
                "payoff_date": settle_date,
                "asset_id": row["asset_id"],
                "symbol": row["symbol"],
                "underlying": VIX_FACTOR,
                "underlying_or_forward": VIX_FACTOR,
                "kind": row["kind"],
                "right": row["kind"],
                "moneyness_bucket": row["moneyness_bucket"],
                "mark": mark,
                "entry_price": mark,
                "strike": strike,
                "start_spot": float(row["vix_forward"]),
                "payoff_raw_close": settlement_value,
                "terminal_spot_proxy": settlement_value,
                "expiry_spot_proxy": settlement_value,
                "vix_close_proxy_settlement": proxy_value,
                "exact_minus_vix_close_proxy": float(settlement_value) - float(proxy_value) if proxy_value is not None else np.nan,
                "terminal_forward_proxy": terminal_forward,
                "expiry_weight": 1.0,
                "payoff_proxy": payoff,
                "exit_price": payoff,
                "option_return": payoff / mark - 1.0,
                "delta": float(row["delta"]),
                "gamma": float(row["gamma"]),
                "vega": float(row["vega"]),
                "theta": float(row["theta"]),
                "iv_proxy": float(row["iv_proxy"]),
                "expiry_days": int((expiry - decision_date).days),
                "settlement_source": settlement_source,
                "greek_model": row.get("greek_model", "black76_vx_forward"),
                "vx_contract": row.get("vx_contract", ""),
                "train_end_date": pd.Timestamp("2020-12-31"),
                "asset_class": "vix_option",
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail
    returns = detail.pivot(index="return_date", columns="asset_id", values="option_return").sort_index()
    returns.index.name = "snap_date"
    return returns.replace([np.inf, -np.inf], np.nan), detail


def front_vx_price_series(root: Path = ROOT) -> pd.Series:
    vx = vx_curve(root)
    if vx.empty:
        return pd.Series(dtype=float, name="VX_FRONT")
    rows = []
    for d, grp in vx.groupby("trade_date", observed=True):
        g = grp[grp["settlement_date"].gt(pd.Timestamp(d) + pd.Timedelta(days=1))].sort_values(
            ["settlement_date", "total_volume"], ascending=[True, False]
        )
        if g.empty:
            continue
        row = g.iloc[0]
        rows.append({"date": pd.Timestamp(d), "price": float(row["forward_price"]), "contract": str(row.get("contract", ""))})
    if not rows:
        return pd.Series(dtype=float, name="VX_FRONT")
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return pd.Series(df["price"].to_numpy(float), index=df.index, name="VX_FRONT")


def vix_state_panel(dates: Iterable[pd.Timestamp], root: Path = ROOT) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    complex_df = load_vix_complex(root)
    if complex_df.empty:
        return pd.DataFrame(index=idx)
    pivot = complex_df.pivot_table(index="date", columns="index_name", values="close", aggfunc="last").sort_index()
    out = pivot.reindex(idx, method="ffill")
    if "VIX" in out:
        out["dVIX"] = out["VIX"].diff()
    if "VVIX" in out:
        out["dVVIX"] = out["VVIX"].diff()
    vx_front = front_vx_price_series(root).reindex(idx, method="ffill")
    out["VX_FRONT"] = vx_front
    out["VX_FRONT_return"] = vx_front.pct_change(fill_method=None)
    if "VIX" in out:
        out["vx_basis"] = out["VX_FRONT"] - out["VIX"]
    if "VIX3M" in out and "VIX" in out:
        out["vix_term_slope"] = out["VIX3M"] - out["VIX"]
    return out.replace([np.inf, -np.inf], np.nan)


def vix_data_audit(panel: pd.DataFrame, reps: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"Check": "Raw VIX option rows after contract parsing", "Value": f"{len(panel):,}", "Pass": "yes" if len(panel) > 1000 else "no"})
    rows.append({"Check": "Representative VIX option choices", "Value": f"{len(reps):,}", "Pass": "yes" if len(reps) > 50 else "no"})
    rows.append({"Check": "VIX option holding ledger rows", "Value": f"{len(detail):,}", "Pass": "yes" if len(detail) > 50 else "no"})
    if not panel.empty:
        dupes = int(panel.duplicated(["trade_date", "symbol"]).sum()) if {"trade_date", "symbol"}.issubset(panel.columns) else -1
        rows.append({"Check": "Duplicate VIX date-symbol rows after dedupe", "Value": str(dupes), "Pass": "yes" if dupes == 0 else "no"})
    if not reps.empty:
        share = float(reps.get("greek_model", pd.Series(dtype=object)).astype(str).eq("black76_vx_forward").mean()) if "greek_model" in reps else 0.0
        rows.append({"Check": "Black-76 VX-forward Greek coverage", "Value": f"{share:.3f}", "Pass": "yes" if share > 0.50 else "no"})
    if not detail.empty:
        source_counts = detail["settlement_source"].value_counts().to_dict()
        rows.append({"Check": "VIX settlement source", "Value": ", ".join(f"{k}:{v}" for k, v in source_counts.items()), "Pass": "yes" if all(str(k) == "vro_soq_exact" for k in source_counts) else "proxy"})
        rows.append({"Check": "All VIX decisions before payoff dates", "Value": str(bool((detail["decision_date"] < detail["payoff_date"]).all())), "Pass": "yes" if bool((detail["decision_date"] < detail["payoff_date"]).all()) else "no"})
        rows.append({"Check": "Minimum VIX long-option return", "Value": f"{float(detail['option_return'].min()):.3f}", "Pass": "yes" if float(detail["option_return"].min()) >= -1.0000001 else "no"})
    return pd.DataFrame(rows)


__all__ = [
    "VIX_FACTOR",
    "add_black76_columns",
    "align_vx_forward",
    "black76_greeks",
    "black76_price",
    "build_vix_option_bucket_panel",
    "front_vx_price_series",
    "implied_vol_black76",
    "load_vro_series",
    "parse_osi_symbol",
    "stack_vix_option_shards",
    "vix_close_series",
    "vix_data_audit",
    "vix_state_panel",
    "vx_curve",
    "vx_forward_lookup",
]
