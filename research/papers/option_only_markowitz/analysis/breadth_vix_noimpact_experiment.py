"""Breadth/VIX experiment with the capacity-impact penalty removed.

This standalone script reruns the paper's option-only sleeves across four
universe configurations while setting only the quadratic market-impact penalty
to zero. Spread, fee, slippage, borrow, margin, and assignment accounting remain
enabled through the publication cost ledger.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_capacity_experiment import (
    BREADTH_48,
    load_bucket_panel_poc,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    MIN_OPTION_MARK,
    PRIMARY_UNDERLYINGS,
    ROOT,
    TRAIN_END,
    VIX_FACTOR,
    build_expiry_proxy_return_panel,
    build_vix_option_bucket_panel,
    load_raw_close_panel,
    make_model,
    representative_specs,
    split_adjusted_spot_panel,
    strategy_weights,
)
from src.portfolio.option_only_markowitz_model import OptionOnlyMarkowitzModel


OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "breadth_experiment"
RESULTS_PATH = OUT_DIR / "breadth_vix_noimpact_results.csv"
SUMMARY_PATH = OUT_DIR / "breadth_vix_noimpact_summary.md"
STRATEGY_NAMES = ["Greek Markowitz", "Delta neutral", "Equal premium", "Equal risk"]
ANCHORS = {"Greek Markowitz": 1.374, "Delta neutral": 1.414}
ANCHOR_TOL = 0.05
_VIX_CACHE: dict[tuple[int, ...], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def _sharpe(series: pd.Series) -> float:
    x = series.dropna()
    std = x.std(ddof=1)
    if len(x) < 2 or not np.isfinite(std) or std <= 0:
        return float("nan")
    return float(np.sqrt(12.0) * x.mean() / std)


def _fmt_num(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _available_new_names() -> list[str]:
    avail = set(
        pd.read_parquet(
            ROOT / "data/feature_store/option_greek_proxy_panel.parquet",
            columns=["underlying"],
        )["underlying"].unique()
    )
    return [name for name in BREADTH_48 if name in avail]


def _strategy_subset(strat: dict[str, pd.Series], label: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name in STRATEGY_NAMES:
        weights = strat.get(name)
        if weights is None:
            print(f"{label}: skipping missing strategy key {name!r}", flush=True)
            continue
        out[name] = weights
    return out


@contextmanager
def _cvxpy_strategy_solver():
    """Use the model's SOCP max-Sharpe solver inside the imported strategy helper.

    ``strategy_weights`` intentionally calls ``solve_max_sharpe`` without a
    method argument. The SLSQP default reproduces historical behavior but can be
    prohibitively slow for the 56-name universe; the built-in CVXPY SOCP path
    solves the same constrained max-Sharpe problem and matches the original+VIX
    validation anchor.
    """

    original = OptionOnlyMarkowitzModel.solve_max_sharpe

    def solve_max_sharpe_socp(
        self: OptionOnlyMarkowitzModel,
        method: str = "slsqp",
        raise_on_infeasible: bool = False,
    ):
        _ = method
        return original(self, method="cvxpy", raise_on_infeasible=raise_on_infeasible)

    OptionOnlyMarkowitzModel.solve_max_sharpe = solve_max_sharpe_socp
    try:
        yield
    finally:
        OptionOnlyMarkowitzModel.solve_max_sharpe = original


def _cached_vix_option_bucket_panel(
    rebalance_dates: Sequence[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = tuple(pd.Timestamp(d).normalize().value for d in pd.to_datetime(pd.Index(rebalance_dates)).dropna())
    if normalized not in _VIX_CACHE:
        _VIX_CACHE[normalized] = build_vix_option_bucket_panel(
            sorted(pd.Timestamp(v) for v in pd.to_datetime(list(normalized))),
            ROOT,
            MIN_OPTION_MARK,
        )
    return _VIX_CACHE[normalized]


def _mean_capacity_cost(cost_ledger: pd.DataFrame, strategy: str) -> float:
    if cost_ledger.empty or "capacity_cost_nav" not in cost_ledger:
        return 0.0
    sub = cost_ledger[cost_ledger["strategy"].eq(strategy)]
    if sub.empty:
        return 0.0
    monthly = pd.to_numeric(sub["capacity_cost_nav"], errors="coerce").fillna(0.0)
    return float(monthly.groupby(sub["return_date"]).sum().mean())


def _build_config_panel(
    equity_underlyings: Sequence[str],
    poc_names: Sequence[str],
    with_vix: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], bool]:
    _, reps, returns = load_bucket_panel_poc(equity_underlyings, poc_names)

    raw_close = load_raw_close_panel(equity_underlyings)
    _, split_factors, _ = split_adjusted_spot_panel(raw_close)
    _, detail = build_expiry_proxy_return_panel(reps, raw_close, split_factors)
    detail = detail[detail["asset_id"].isin(returns.columns)].copy()
    detail["asset_class"] = "equity_option"
    reps = reps.copy()
    reps["asset_class"] = "equity_option"

    universe = list(equity_underlyings)
    has_vix = False
    if with_vix:
        _, vix_reps, vix_returns, vix_detail, _ = _cached_vix_option_bucket_panel(
            sorted(pd.to_datetime(reps["snap_date"].dropna().unique())),
        )
        if not vix_returns.empty and not vix_reps.empty:
            reps = pd.concat([reps, vix_reps], ignore_index=True, sort=False)
            returns = returns.join(vix_returns, how="outer").sort_index()
            detail = pd.concat([detail, vix_detail], ignore_index=True, sort=False)
            universe = list(equity_underlyings) + [VIX_FACTOR]
            has_vix = True

    return reps, returns, detail, universe, has_vix


def run_config(
    label: str,
    equity_underlyings: Sequence[str],
    *,
    with_vix: bool,
    poc_names: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, pd.Series], bool]:
    reps, returns, detail, universe, has_vix = _build_config_panel(equity_underlyings, poc_names, with_vix)

    spec = representative_specs(reps, returns)
    returns = returns.reindex(columns=spec.index).dropna(how="all")
    model, _ = make_model(spec, returns, reps, universe)
    with _cvxpy_strategy_solver():
        strat = strategy_weights(model, universe)
    strategies = _strategy_subset(strat, label)
    test = returns.loc[returns.index > TRAIN_END, model.contracts].fillna(0.0)

    gross_frame = pd.DataFrame(index=test.index)
    for name, weights in strategies.items():
        gross_frame[name] = model.portfolio_return_series(test, weights)

    cfg = ResearchCostConfig(impact_cost_rate=0.0)
    surface = load_cbbo_spread_surface(ROOT, cfg.cbbo_spread_surface_path) if cfg.use_cbbo_spread_surface else None
    cost_inputs = build_cost_input_ledger(reps, detail, ROOT, cfg, spread_surface=surface)
    net_frame, cost_ledger, cap_ledger, *_ = compute_strategy_cost_ledgers(
        gross_frame,
        strategies,
        cost_inputs,
        cfg,
    )

    rows: list[dict[str, object]] = []
    for name in STRATEGY_NAMES:
        if name not in strategies or name not in gross_frame or name not in net_frame:
            continue
        rows.append(
            {
                "config": label,
                "with_vix": bool(with_vix),
                "n_underlyings": int(len(set(universe))),
                "n_contracts": int(len(model.contracts)),
                "strategy": name,
                "gross_sharpe": _sharpe(gross_frame[name]),
                "net_sharpe_noimpact": _sharpe(net_frame[name]),
                "mean_capacity_cost": _mean_capacity_cost(cost_ledger, name),
            }
        )

    max_cap_cost = 0.0
    if not cost_ledger.empty and "capacity_cost_nav" in cost_ledger:
        max_cap_cost = float(pd.to_numeric(cost_ledger["capacity_cost_nav"], errors="coerce").abs().max())
    penalized_share = float((cap_ledger["capacity_status"] == "penalized").mean()) if not cap_ledger.empty else 0.0
    print(
        f"{label}: contracts={len(model.contracts)} vix_included={has_vix} "
        f"max_capacity_cost={max_cap_cost:.3e} penalized_rows={penalized_share:.1%}",
        flush=True,
    )
    return rows, {name: gross_frame[name] for name in gross_frame.columns}, has_vix


def _read_interpretation(gm: pd.DataFrame) -> str:
    lookup = gm.set_index("config")

    def val(config: str, col: str) -> float:
        if config not in lookup.index:
            return float("nan")
        return float(lookup.loc[config, col])

    gross_breadth_vix = val("larger+VIX", "gross_sharpe") - val("orig+VIX", "gross_sharpe")
    gross_breadth_no_vix = val("larger", "gross_sharpe") - val("orig", "gross_sharpe")
    net_breadth_vix = val("larger+VIX", "net_sharpe_noimpact") - val("orig+VIX", "net_sharpe_noimpact")
    net_breadth_no_vix = val("larger", "net_sharpe_noimpact") - val("orig", "net_sharpe_noimpact")
    gross_vix_orig = val("orig+VIX", "gross_sharpe") - val("orig", "gross_sharpe")
    gross_vix_larger = val("larger+VIX", "gross_sharpe") - val("larger", "gross_sharpe")
    net_vix_orig = val("orig+VIX", "net_sharpe_noimpact") - val("orig", "net_sharpe_noimpact")
    net_vix_larger = val("larger+VIX", "net_sharpe_noimpact") - val("larger", "net_sharpe_noimpact")

    breadth_gross_help = gross_breadth_vix > 0 and gross_breadth_no_vix > 0
    breadth_net_help = net_breadth_vix > 0 and net_breadth_no_vix > 0
    vix_gross_help = gross_vix_orig > 0 and gross_vix_larger > 0
    vix_net_help = net_vix_orig > 0 and net_vix_larger > 0

    breadth_text = (
        "Breadth improves both gross and impact-free net Sharpe in both VIX settings"
        if breadth_gross_help and breadth_net_help
        else "Breadth does not consistently improve both gross and impact-free net Sharpe across the VIX and no-VIX settings"
    )
    vix_text = (
        "VIX improves both gross and impact-free net Sharpe in both universe sizes"
        if vix_gross_help and vix_net_help
        else "VIX does not consistently improve both gross and impact-free net Sharpe across the original and larger universes"
    )
    return (
        f"{breadth_text}: gross breadth deltas are {_fmt_num(gross_breadth_vix)} with VIX "
        f"and {_fmt_num(gross_breadth_no_vix)} without VIX, while impact-free net deltas are "
        f"{_fmt_num(net_breadth_vix)} and {_fmt_num(net_breadth_no_vix)}. {vix_text}: gross VIX "
        f"deltas are {_fmt_num(gross_vix_orig)} in the original universe and {_fmt_num(gross_vix_larger)} "
        f"in the larger universe, while impact-free net deltas are {_fmt_num(net_vix_orig)} "
        f"and {_fmt_num(net_vix_larger)}."
    )


def build_summary(results: pd.DataFrame, present_new: Sequence[str]) -> str:
    gm = results[results["strategy"].eq("Greek Markowitz")].copy()
    gm["config_order"] = gm["config"].map({"orig+VIX": 0, "larger+VIX": 1, "orig": 2, "larger": 3})
    gm = gm.sort_values("config_order")

    lines = [
        "# Breadth/VIX No-Impact Experiment",
        "",
        f"New breadth names present in panel: {len(present_new)}/48",
        "",
        "| Config | With VIX | Underlyings | Contracts | Gross Sharpe | Net Sharpe (impact removed) | Mean capacity cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gm.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.config),
                    str(bool(row.with_vix)),
                    str(int(row.n_underlyings)),
                    str(int(row.n_contracts)),
                    _fmt_num(row.gross_sharpe),
                    _fmt_num(row.net_sharpe_noimpact),
                    _fmt_num(row.mean_capacity_cost, 8),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Read", "", _read_interpretation(gm) if not gm.empty else "Greek Markowitz rows were not produced.", ""])
    return "\n".join(lines)


def print_validation(results: pd.DataFrame) -> None:
    print("")
    print("Validation anchor for orig+VIX:")
    sub = results[results["config"].eq("orig+VIX")]
    for strategy, expected in ANCHORS.items():
        row = sub[sub["strategy"].eq(strategy)]
        if row.empty:
            print(f"  {strategy}: missing, expected ~{expected:.3f}", flush=True)
            continue
        actual = float(row.iloc[0]["gross_sharpe"])
        matched = bool(np.isfinite(actual) and abs(actual - expected) <= ANCHOR_TOL)
        status = "MATCH" if matched else "MISS"
        print(
            f"  {strategy}: gross Sharpe {actual:.3f}, expected ~{expected:.3f}, tol={ANCHOR_TOL:.2f} -> {status}",
            flush=True,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    present_new = _available_new_names()
    configs = [
        ("orig+VIX", list(PRIMARY_UNDERLYINGS), True, []),
        ("larger+VIX", list(PRIMARY_UNDERLYINGS) + present_new, True, present_new),
        ("orig", list(PRIMARY_UNDERLYINGS), False, []),
        ("larger", list(PRIMARY_UNDERLYINGS) + present_new, False, present_new),
    ]

    print(f"new names present in panel: {len(present_new)}/48", flush=True)
    rows: list[dict[str, object]] = []
    for label, underlyings, with_vix, poc_names in configs:
        print(f"running {label}: requested_underlyings={len(underlyings)} with_vix={with_vix}", flush=True)
        config_rows, _, _ = run_config(label, underlyings, with_vix=with_vix, poc_names=poc_names)
        rows.extend(config_rows)

    results = pd.DataFrame(rows)
    results = results[
        [
            "config",
            "with_vix",
            "n_underlyings",
            "n_contracts",
            "strategy",
            "gross_sharpe",
            "net_sharpe_noimpact",
            "mean_capacity_cost",
        ]
    ].copy()
    results.to_csv(RESULTS_PATH, index=False)

    summary = build_summary(results, present_new)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print_validation(results)
    print("")
    print("Greek Markowitz summary:")
    print(summary)
    print(f"wrote {RESULTS_PATH}", flush=True)
    print(f"wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
