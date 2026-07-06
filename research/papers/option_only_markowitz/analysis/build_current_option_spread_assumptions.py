"""Build market-hours option spread assumptions from public Cboe chains.

The breadth proof-of-concept names do not all have historical CBBO in the local
feature store.  This script replaces the old blanket 10% relative-spread fill
with a symbol/bucket/tenor table derived from public Cboe delayed option quotes
captured during regular option-market hours.  The cost ledger uses the resulting
relative spreads only as a fallback after historical panel CBBO and the
historical CBBO spread surface fail.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_ingestion.build_cbbo_cost_surface import assign_moneyness_bucket, assign_tenor_bucket
from research.papers.option_only_markowitz.analysis.breadth_capacity_experiment import BREADTH_48
from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import OUT_DIR
from research.papers.option_only_markowitz.analysis.option_market_hours import (
    classify_cboe_option_rth_timestamp,
)
from research.papers.option_only_markowitz.analysis.run_empirics import PRIMARY_UNDERLYINGS
from research.papers.option_only_markowitz.analysis.vix_option_panel import _bucketize


CBOE_DELAYED_OPTION_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
DEFAULT_OUT = OUT_DIR / "current_option_spread_assumptions.csv"
DEFAULT_AUDIT_OUT = OUT_DIR / "current_option_spread_fetch_audit.csv"
OPTION_RE = re.compile(r"^(?P<root>.*?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
EQUITY_BUCKETS = {"atm", "put_near", "call_near", "put_wing"}
VIX_BUCKETS = {"vix_atm", "vix_call_near", "vix_call_wing", "vix_put_near"}
QUOTE_SYMBOL_OVERRIDES = {"VX_FRONT": "_VIX", "VIX": "_VIX"}
MARKET_HOURS_RULE = "Cboe options regular session, 09:30-16:15 America/New_York, holidays and early closes excluded"


def default_underlyings() -> list[str]:
    names = list(PRIMARY_UNDERLYINGS) + list(BREADTH_48) + ["VX_FRONT"]
    return list(dict.fromkeys(str(x).upper() for x in names))


def quote_symbol_for_underlying(underlying: str) -> str:
    return QUOTE_SYMBOL_OVERRIDES.get(str(underlying).upper(), str(underlying).upper())


def fetch_cboe_chain(quote_symbol: str, timeout: float = 20.0) -> dict[str, object]:
    url = CBOE_DELAYED_OPTION_URL.format(symbol=quote_symbol)
    req = urllib.request.Request(url, headers={"User-Agent": "option-markowitz-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parse_option_symbol(option_symbol: object) -> tuple[pd.Timestamp | None, str | None, float | None]:
    match = OPTION_RE.match(str(option_symbol or "").strip())
    if not match:
        return None, None, None
    ymd = match.group("ymd")
    try:
        expiry = pd.Timestamp(year=2000 + int(ymd[:2]), month=int(ymd[2:4]), day=int(ymd[4:6]))
    except ValueError:
        return None, None, None
    kind = "call" if match.group("cp") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0
    return expiry, kind, strike


def chain_rows(
    underlying: str,
    payload: dict[str, object],
    *,
    timestamp_tz: str,
    market_hours_check: object | None = None,
) -> list[dict[str, object]]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return []
    request_symbol = quote_symbol_for_underlying(underlying)
    quote_symbol = str(data.get("symbol") or request_symbol).upper()
    asset_class = "vix_option" if str(underlying).upper() in {"VIX", "VX_FRONT"} else "equity_option"
    normalized_underlying = "VX_FRONT" if asset_class == "vix_option" else str(underlying).upper()
    timestamp = pd.to_datetime(payload.get("timestamp"), errors="coerce")
    market_check = market_hours_check or classify_cboe_option_rth_timestamp(payload.get("timestamp"), timestamp_tz=timestamp_tz)
    trade_date = timestamp.normalize() if not pd.isna(timestamp) else pd.NaT
    spot = pd.to_numeric(
        pd.Series([data.get("current_price", data.get("close", data.get("prev_day_close", np.nan)))]),
        errors="coerce",
    ).iloc[0]
    options = data.get("options") or []
    if pd.isna(spot) or float(spot) <= 0 or not isinstance(options, list):
        return []

    rows: list[dict[str, object]] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        expiry, kind, strike = parse_option_symbol(option.get("option"))
        if expiry is None or kind is None or strike is None or pd.isna(trade_date):
            continue
        dte = int((expiry - trade_date).days)
        if dte < 7 or dte > 120:
            continue
        bid = _to_float(option.get("bid"))
        ask = _to_float(option.get("ask"))
        if not np.isfinite([bid, ask]).all() or bid <= 0 or ask <= bid:
            continue
        mid = 0.5 * (bid + ask)
        if mid <= 0.05:
            continue
        volume = max(_to_float(option.get("volume")), 0.0)
        open_interest = max(_to_float(option.get("open_interest")), 0.0)
        if volume <= 0 and open_interest <= 0:
            continue

        if asset_class == "vix_option":
            log_moneyness = math.log(float(strike) / float(spot))
            bucket = _bucketize(pd.Series([log_moneyness]), pd.Series([kind])).iloc[0]
            if pd.isna(bucket) or str(bucket) not in VIX_BUCKETS:
                continue
        else:
            bucket = assign_moneyness_bucket(float(spot), float(strike), kind)
            if bucket not in EQUITY_BUCKETS:
                continue

        rows.append(
            {
                "underlying": normalized_underlying,
                "quote_symbol": quote_symbol,
                "asset_class": asset_class,
                "moneyness_bucket": str(bucket),
                "tenor_bucket": assign_tenor_bucket(float(dte)),
                "chain_timestamp": str(payload.get("timestamp", "")),
                "snapshot_eastern": getattr(market_check, "timestamp_eastern", ""),
                "timestamp_tz_assumed": timestamp_tz,
                "market_hours_snapshot": bool(getattr(market_check, "valid", False)),
                "market_hours_reason": getattr(market_check, "reason", ""),
                "market_hours_rule": MARKET_HOURS_RULE,
                "source_url": CBOE_DELAYED_OPTION_URL.format(symbol=request_symbol),
                "option_symbol": option.get("option"),
                "expiry": expiry.date().isoformat(),
                "days_to_expiry": dte,
                "kind": kind,
                "strike": float(strike),
                "spot": float(spot),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "abs_spread": ask - bid,
                "relative_spread": (ask - bid) / mid,
                "volume": volume,
                "open_interest": open_interest,
            }
        )
    return rows


def _to_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def liquid_slice(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return group
    n = int(max(5, math.ceil(0.25 * len(group))))
    ranked = group.assign(
        _liquidity_score=pd.to_numeric(group["volume"], errors="coerce").fillna(0.0)
        + 0.01 * pd.to_numeric(group["open_interest"], errors="coerce").fillna(0.0)
    ).sort_values(
        ["_liquidity_score", "volume", "open_interest", "relative_spread"],
        ascending=[False, False, False, True],
    )
    return ranked.head(min(n, len(ranked))).drop(columns=["_liquidity_score"])


def summarize_group(keys: dict[str, object], group: pd.DataFrame) -> dict[str, object]:
    liquid = liquid_slice(group)
    rel = pd.to_numeric(liquid["relative_spread"], errors="coerce").dropna()
    abs_spread = pd.to_numeric(liquid["abs_spread"], errors="coerce").dropna()
    mid = pd.to_numeric(liquid["mid"], errors="coerce").dropna()
    return {
        **keys,
        "chain_timestamp": str(group["chain_timestamp"].dropna().iloc[0]) if group["chain_timestamp"].notna().any() else "",
        "snapshot_eastern": str(group["snapshot_eastern"].dropna().iloc[0]) if "snapshot_eastern" in group and group["snapshot_eastern"].notna().any() else "",
        "timestamp_tz_assumed": str(group["timestamp_tz_assumed"].dropna().iloc[0]) if "timestamp_tz_assumed" in group and group["timestamp_tz_assumed"].notna().any() else "",
        "market_hours_snapshot": bool(group["market_hours_snapshot"].fillna(False).all()) if "market_hours_snapshot" in group else False,
        "market_hours_reason": str(group["market_hours_reason"].dropna().iloc[0]) if "market_hours_reason" in group and group["market_hours_reason"].notna().any() else "",
        "market_hours_rule": MARKET_HOURS_RULE,
        "source_url": str(group["source_url"].dropna().iloc[0]) if group["source_url"].notna().any() else "",
        "n_quotes": int(len(group)),
        "n_liquid_quotes": int(len(liquid)),
        "total_volume": float(pd.to_numeric(group["volume"], errors="coerce").fillna(0.0).sum()),
        "total_open_interest": float(pd.to_numeric(group["open_interest"], errors="coerce").fillna(0.0).sum()),
        "fill_relative_spread": float(rel.quantile(0.25)) if len(rel) else np.nan,
        "fill_abs_spread": float(abs_spread.quantile(0.25)) if len(abs_spread) else np.nan,
        "median_relative_spread": float(rel.median()) if len(rel) else np.nan,
        "p25_relative_spread": float(rel.quantile(0.25)) if len(rel) else np.nan,
        "p75_relative_spread": float(rel.quantile(0.75)) if len(rel) else np.nan,
        "median_abs_spread": float(abs_spread.median()) if len(abs_spread) else np.nan,
        "p25_abs_spread": float(abs_spread.quantile(0.25)) if len(abs_spread) else np.nan,
        "p75_abs_spread": float(abs_spread.quantile(0.75)) if len(abs_spread) else np.nan,
        "median_mid": float(mid.median()) if len(mid) else np.nan,
        "fill_method": "cboe_current_liquid_quartile_p25_relative_spread",
    }


def build_assumptions(chains: pd.DataFrame) -> pd.DataFrame:
    if chains.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["underlying", "quote_symbol", "asset_class", "moneyness_bucket", "tenor_bucket"]
    for keys, group in chains.groupby(group_cols, dropna=False, observed=True):
        rows.append(summarize_group(dict(zip(group_cols, keys, strict=True)), group))

    bucket_cols = ["underlying", "quote_symbol", "asset_class", "moneyness_bucket"]
    for keys, group in chains.groupby(bucket_cols, dropna=False, observed=True):
        key_dict = dict(zip(bucket_cols, keys, strict=True))
        key_dict["tenor_bucket"] = "all"
        rows.append(summarize_group(key_dict, group))

    underlying_cols = ["underlying", "quote_symbol", "asset_class"]
    for keys, group in chains.groupby(underlying_cols, dropna=False, observed=True):
        key_dict = dict(zip(underlying_cols, keys, strict=True))
        key_dict["moneyness_bucket"] = "all"
        key_dict["tenor_bucket"] = "all"
        rows.append(summarize_group(key_dict, group))

    out = pd.DataFrame(rows)
    return out.sort_values(["asset_class", "underlying", "moneyness_bucket", "tenor_bucket"]).reset_index(drop=True)


def fetch_all(
    underlyings: Iterable[str],
    sleep_seconds: float,
    *,
    timestamp_tz: str,
    require_market_hours: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    fetched_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    for underlying in underlyings:
        normalized = str(underlying).upper()
        quote_symbol = quote_symbol_for_underlying(normalized)
        try:
            payload = fetch_cboe_chain(quote_symbol)
            check = classify_cboe_option_rth_timestamp(payload.get("timestamp"), timestamp_tz=timestamp_tz)
            if require_market_hours and not check.valid:
                chain = []
                status = "off_hours_snapshot"
            else:
                chain = chain_rows(
                    normalized,
                    payload,
                    timestamp_tz=timestamp_tz,
                    market_hours_check=check,
                )
                status = "ok"
            rows.extend(chain)
            audit.append(
                {
                    "underlying": normalized,
                    "quote_symbol": quote_symbol,
                    "source_url": CBOE_DELAYED_OPTION_URL.format(symbol=quote_symbol),
                    "status": status,
                    "usable_quotes": int(len(chain)),
                    "chain_timestamp": str(payload.get("timestamp", "")),
                    "snapshot_eastern": check.timestamp_eastern,
                    "timestamp_tz_assumed": timestamp_tz,
                    "market_hours_snapshot": bool(check.valid),
                    "market_hours_reason": check.reason,
                    "market_hours_rule": MARKET_HOURS_RULE,
                    "fetched_at_utc": fetched_at_utc,
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - depends on external service.
            audit.append(
                {
                    "underlying": normalized,
                    "quote_symbol": quote_symbol,
                    "source_url": CBOE_DELAYED_OPTION_URL.format(symbol=quote_symbol),
                    "status": "error",
                    "usable_quotes": 0,
                    "chain_timestamp": "",
                    "snapshot_eastern": "",
                    "timestamp_tz_assumed": timestamp_tz,
                    "market_hours_snapshot": False,
                    "market_hours_reason": "fetch_error",
                    "market_hours_rule": MARKET_HOURS_RULE,
                    "fetched_at_utc": fetched_at_utc,
                    "error": repr(exc),
                }
            )
        time.sleep(max(float(sleep_seconds), 0.0))
    return pd.DataFrame(rows), pd.DataFrame(audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--symbols", nargs="*", default=None, help="Override underlyings to fetch.")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument(
        "--timestamp-tz",
        default="UTC",
        help="Timezone to assume for Cboe chain timestamps, which do not include offsets.",
    )
    parser.add_argument(
        "--allow-off-hours",
        action="store_true",
        help="Permit off-hours delayed-chain snapshots. Intended only for stress diagnostics.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write assumptions even if some requested symbols fail or have no usable regular-hours rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    underlyings = [s.upper() for s in args.symbols] if args.symbols else default_underlyings()
    run_check = classify_cboe_option_rth_timestamp(pd.Timestamp.now(tz="UTC"), timestamp_tz="UTC")
    if not args.allow_off_hours and not run_check.valid:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        audit = pd.DataFrame(
            [
                {
                    "underlying": str(symbol).upper(),
                    "quote_symbol": quote_symbol_for_underlying(str(symbol).upper()),
                    "source_url": CBOE_DELAYED_OPTION_URL.format(symbol=quote_symbol_for_underlying(str(symbol).upper())),
                    "status": "run_clock_off_hours",
                    "usable_quotes": 0,
                    "chain_timestamp": "",
                    "snapshot_eastern": run_check.timestamp_eastern,
                    "timestamp_tz_assumed": args.timestamp_tz,
                    "market_hours_snapshot": False,
                    "market_hours_reason": run_check.reason,
                    "market_hours_rule": MARKET_HOURS_RULE,
                    "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "error": "Refused to build spread assumptions outside regular option-market hours.",
                }
                for symbol in underlyings
            ]
        )
        audit.to_csv(args.audit_out, index=False)
        print(f"refused off-hours build; wrote audit rows to {args.audit_out}")
        print("existing assumptions file left unchanged")
        return 2

    chains, audit = fetch_all(
        underlyings,
        args.sleep_seconds,
        timestamp_tz=args.timestamp_tz,
        require_market_hours=not args.allow_off_hours,
    )
    assumptions = build_assumptions(chains)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_out, index=False)
    print(f"wrote {len(audit)} fetch audit rows to {args.audit_out}")
    missing = audit.loc[~audit["status"].eq("ok") | audit["usable_quotes"].eq(0), "underlying"].astype(str).tolist()
    if missing:
        print("missing_or_empty=" + ",".join(missing))
    if assumptions.empty:
        print("no valid market-hours assumption rows; existing assumptions file left unchanged")
        return 1
    if missing and not args.allow_partial:
        print("partial market-hours coverage refused; existing assumptions file left unchanged")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assumptions.to_csv(args.out, index=False)
    print(f"wrote {len(assumptions)} assumption rows to {args.out}")
    return 0 if assumptions["underlying"].nunique() else 1


if __name__ == "__main__":
    raise SystemExit(main())
