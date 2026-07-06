"""Export locked E1 breadth targets for forward shadow trading.

This utility bridges the research candidate to the broker-neutral shadow layer.
It exports target weights and the most recent representative listed contracts
available on or before a decision date. It does not fetch quotes, submit orders,
or certify live tradability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_robustness_experiment import (
    DEFAULT_NAV,
    DEFAULT_PARTICIPATION,
    PRIMARY_STRATEGY,
    build_full_context,
    build_panels,
)


def _latest_representatives(reps: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    work = reps.copy()
    work["snap_date"] = pd.to_datetime(work["snap_date"], errors="coerce").dt.normalize()
    work = work[work["snap_date"].le(pd.Timestamp(decision_date).normalize())]
    if work.empty:
        return work
    return work.sort_values(["asset_id", "snap_date"]).groupby("asset_id", as_index=False).tail(1)


def export_locked_e1_shadow_targets(
    *,
    config: str,
    decision_date: str | pd.Timestamp | None,
    out_path: str | Path,
    nav: float = DEFAULT_NAV,
    participation: float = DEFAULT_PARTICIPATION,
    min_abs_weight: float = 1e-8,
) -> pd.DataFrame:
    panels = build_panels([config])
    ctx = build_full_context(panels[config], nav=float(nav), participation=float(participation))
    book = ctx.books[PRIMARY_STRATEGY]
    decision = (
        pd.Timestamp(decision_date).normalize()
        if decision_date is not None
        else pd.to_datetime(ctx.panel.reps["snap_date"], errors="coerce").max().normalize()
    )
    latest = _latest_representatives(ctx.panel.reps, decision)
    weights = pd.Series(book.weights, dtype=float, name="target_weight")
    latest = latest.merge(weights.rename("target_weight"), left_on="asset_id", right_index=True, how="inner")
    latest = latest[pd.to_numeric(latest["target_weight"], errors="coerce").abs().gt(float(min_abs_weight))].copy()
    if latest.empty:
        raise RuntimeError(f"no nonzero locked E1 targets for {config} at {decision.date()}")

    right = latest["kind"] if "kind" in latest else latest.get("right", pd.Series("", index=latest.index))
    asset_class = latest["asset_class"] if "asset_class" in latest else pd.Series("equity_option", index=latest.index)
    moneyness = latest["moneyness_bucket"] if "moneyness_bucket" in latest else pd.Series("", index=latest.index)
    symbol = latest["symbol"] if "symbol" in latest else latest["asset_id"]
    out = pd.DataFrame(
        {
            "decision_time": pd.Timestamp(decision).tz_localize("America/New_York").replace(hour=15, minute=45).tz_convert("UTC"),
            "config": config,
            "strategy": PRIMARY_STRATEGY,
            "asset_id": latest["asset_id"].astype(str),
            "symbol": symbol.astype(str),
            "underlying": latest["underlying"].astype(str),
            "expiry": pd.to_datetime(latest.get("expiry", pd.NaT), errors="coerce").dt.strftime("%Y-%m-%d"),
            "right": right.astype(str),
            "strike": pd.to_numeric(latest.get("strike", np.nan), errors="coerce"),
            "target_weight": pd.to_numeric(latest["target_weight"], errors="coerce"),
            "mark": pd.to_numeric(latest.get("mark", np.nan), errors="coerce"),
            "spot": pd.to_numeric(latest.get("spot", np.nan), errors="coerce"),
            "underlying_price": pd.to_numeric(latest.get("spot", np.nan), errors="coerce"),
            "volume": pd.to_numeric(latest.get("volume", np.nan), errors="coerce"),
            "open_interest": pd.to_numeric(latest.get("open_interest", np.nan), errors="coerce"),
            "moneyness_bucket": moneyness.astype(str),
            "asset_class": asset_class.astype(str),
            "multiplier": 100,
            "nav": float(nav),
            "participation": float(participation),
            "claim_status": "shadow_target_only_not_live_trading_evidence",
        }
    )
    out = out.sort_values(["underlying", "expiry", "right", "strike", "symbol"]).reset_index(drop=True)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    summary = {
        "config": config,
        "strategy": PRIMARY_STRATEGY,
        "decision_date": str(decision.date()),
        "rows": int(len(out)),
        "gross_target_weight": float(out["target_weight"].abs().sum()),
        "nav": float(nav),
        "participation": float(participation),
        "output": str(path),
        "claim_status": "shadow_target_only_not_live_trading_evidence",
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export locked E1 target contracts for shadow trading.")
    parser.add_argument("--config", default="larger+VIX", choices=["orig", "orig+VIX", "larger", "larger+VIX"])
    parser.add_argument("--decision-date", default=None, help="Decision date. Defaults to latest available representative snapshot.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--nav", type=float, default=DEFAULT_NAV)
    parser.add_argument("--participation", type=float, default=DEFAULT_PARTICIPATION)
    parser.add_argument("--min-abs-weight", type=float, default=1e-8)
    args = parser.parse_args(argv)
    out = export_locked_e1_shadow_targets(
        config=args.config,
        decision_date=args.decision_date,
        out_path=args.out,
        nav=args.nav,
        participation=args.participation,
        min_abs_weight=args.min_abs_weight,
    )
    print(json.dumps({"rows": int(len(out)), "output": args.out}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
