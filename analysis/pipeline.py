"""Functional coordinator for the committed derived evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from option_portfolio.metrics import performance_metrics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper" / "evidence"
DISPLAY_NAMES = {
    "R1 repaired net utility": "15% volatility-ceiling specification (R1)",
    "R1.1 25pct positive-edge deployment": "25% volatility-ceiling specification (R1.1)",
    "R1.1 25pct EGARCH diagnostic": "25% volatility-ceiling EGARCH diagnostic (R1.1)",
    "R1.1 25pct VIX40 risk-off": "25% volatility-ceiling VIX-40 diagnostic (R1.1)",
    "larger E1 capped": "larger legacy E1 specification",
    "larger+VIX E1 capped": "larger+VIX legacy E1 specification",
    "orig E1 capped": "orig legacy E1 specification",
    "orig+VIX E1 capped": "orig+VIX legacy E1 specification",
}
FROZEN_RETURN_FILES = {"r1_monthly_returns.csv", "r11_monthly_returns.csv"}


def summarize_returns(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Recompute corrected portfolio metrics from aggregate monthly returns."""

    rows: list[dict[str, object]] = []
    for labels, group in frame.groupby(keys, sort=True, observed=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        stats = performance_metrics(returns, periods_per_year=12.0)
        row: dict[str, object] = dict(zip(keys, labels, strict=True))
        row.update(
            observations=int(stats["n_obs"]),
            annualized_mean_return=stats["annualized_mean_return"],
            cagr=stats["cagr"],
            annualized_volatility=stats["annualized_volatility"],
            sharpe=stats["sharpe"],
            sortino=stats["sortino"],
            max_drawdown=stats["max_drawdown"],
            terminal_wealth=stats["terminal_wealth"],
            defaulted=bool(stats["defaulted"]),
            worst_month=float(returns.min()) if len(returns) else np.nan,
        )
        for source, target in (
            ("integer_repair_failed", "integer_failures"),
            ("integer_execution_abstained", "integer_abstentions"),
        ):
            if source in group:
                row[target] = int(group[source].fillna(False).astype(bool).sum())
        if "gross_nav" in group:
            row["mean_gross_nav"] = float(
                pd.to_numeric(group["gross_nav"], errors="coerce").mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_derived_evidence(destination: Path, source: Path = EVIDENCE) -> None:
    """Build deterministic public evidence without reading licensed inputs."""

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copyfile(path, destination / path.name)

    for path in sorted(destination.glob("*.csv")):
        if path.name in FROZEN_RETURN_FILES:
            continue
        frame = pd.read_csv(path, float_precision="round_trip")
        frame = frame.replace(DISPLAY_NAMES)
        if path.name == "r1_r11_aligned_summary.csv":
            required = {"annualized_mean_return", "cagr"}
            if not required.issubset(frame) or "annualized_return" in frame:
                raise ValueError("aligned summary must use the final return-metric schema")
        frame.to_csv(
            path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )

    quote_summary = destination / "quote_sensitivity_summary.json"
    payload = json.loads(quote_summary.read_text(encoding="utf-8"))
    if any(
        "cagr" not in row or "annualized_return" in row
        for row in payload.get("headline_r11", [])
    ):
        raise ValueError("quote summary must use the final CAGR schema")
    quote_summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    r1 = pd.read_csv(source / "r1_monthly_returns.csv", float_precision="round_trip")
    r11 = pd.read_csv(source / "r11_monthly_returns.csv", float_precision="round_trip")
    summarize_returns(r1, ["config", "strategy"]).replace(DISPLAY_NAMES).to_csv(
        destination / "r1_performance_summary.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    summarize_returns(
        r11,
        ["config", "strategy", "evidence_status"],
    ).replace(DISPLAY_NAMES).to_csv(
        destination / "r11_performance_summary.csv",
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
