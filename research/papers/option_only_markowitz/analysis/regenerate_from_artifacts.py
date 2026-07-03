"""Regenerate published tables from cached artifacts using the FIXED code paths.

This CLI rebuilds the paper tables affected by the econometric audit without
raw vendor data: it reads the cached per-period artifacts under
``research/papers/option_only_markowitz/artifacts`` and re-runs the corrected
inference/simulation code on them.

Tables whose statistics were fixed (reality check, PSR/DSR, simulation,
diagnostics CIs) get NEW numbers; formatting-only tables are rewritten with
identical numbers from the artifact CSVs (3-decimal rounding, ``--`` for
missing values, canonical display names).

Run with:
    python -m research.papers.option_only_markowitz.analysis.regenerate_from_artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis import run_empirics as emp
from research.papers.option_only_markowitz.analysis.inference import (
    BootstrapConfig,
    sharpe_reality_check,
    strategy_metric_inference,
)
from research.papers.option_only_markowitz.analysis.publication_costs import artifact_hash_manifest
from research.papers.option_only_markowitz.analysis.simulation import (
    SimulationConfig,
    compact_assumptions,
    compact_simulation_summary,
    run_tail_path_simulations,
)

PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
ART_DIR = PAPER / "artifacts"
FIG_DIR = PAPER / "figures"
SEED = 20260625

# One canonical mapping from raw artifact keys to publication display names.
CANONICAL_LABELS: dict[str, str] = {
    # simulation methods
    "circular_block_bootstrap": "Circular block bootstrap",
    "egarch_or_ewma": "EGARCH / GARCH fallback",
    "egarch_1_1_t": "EGARCH(1,1)-t",
    "ewma_residual_fallback_insufficient_egarch_obs": "GARCH residual fallback",
    "garch11_residual_fallback_insufficient_egarch_obs": "GARCH residual fallback",
    # cost scenarios
    "mid": "Mid",
    "half_spread": "Half spread",
    "full_spread": "Full spread",
    # liquidity tiers
    "all_eligible": "All eligible",
    "top_volume_quartile": "Top volume quartile",
    "tight_spread_quartile": "Tight spread quartile",
    "high_open_interest_quartile": "High open interest quartile",
    "combined_liquid": "Combined liquid",
    # forecast ablations
    "carry_only": "Carry only",
    "variance_risk_premium_only": "Variance risk premium only",
    "skew_tail_only": "Skew/tail only",
    "vix_regime_only": "VIX regime only",
    "relative_value_only": "Relative value only",
    "full_conditional_model": "Full conditional model",
}
SUFFIX_LABELS = {"gross": "gross", "mid": "mid", "half_spread": "half spread", "full_spread": "full spread"}


def canonical_name(value: object) -> object:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s in CANONICAL_LABELS:
        return CANONICAL_LABELS[s]
    if "::" in s:
        base, suffix = s.rsplit("::", 1)
        return f"{base} ({SUFFIX_LABELS.get(suffix, suffix.replace('_', ' '))})"
    if s.startswith(("garch11_residual_fallback", "ewma_residual_fallback")):
        return "GARCH residual fallback"
    return s


def write_table(df: pd.DataFrame, name: str, canonical_cols: tuple[str, ...] = ()) -> Path:
    out = df.copy()
    for col in canonical_cols:
        if col in out.columns:
            out[col] = out[col].map(canonical_name)
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            # Escape LaTeX specials but leave missing cells to na_rep ('--').
            out[col] = out[col].map(lambda v: emp._latex_escape(v) if pd.notna(v) else np.nan)
    path = TABLE_DIR / name
    emp._write_latex_table(out, path, na_rep="--")
    _assert_tex_sane(path, expected_cols=len(out.columns))
    return path


def _assert_tex_sane(path: Path, expected_cols: int | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count("{") != text.count("}"):
        raise AssertionError(f"unbalanced braces in {path}")
    if "\\begin{tabular}" not in text or "\\end{tabular}" not in text:
        raise AssertionError(f"missing tabular environment in {path}")
    body_rows = [ln for ln in text.splitlines() if ln.strip().endswith("\\\\")]
    if expected_cols is not None:
        for ln in body_rows:
            if ln.count("&") != expected_cols - 1:
                raise AssertionError(f"row/column mismatch in {path}: {ln[:80]}")
    for token in ("e+1", "e+2"):  # no scientific blow-ups / 200-digit numbers
        if token in text:
            raise AssertionError(f"suspicious huge number in {path}")


def _read_returns(name: str) -> pd.DataFrame:
    return pd.read_csv(ART_DIR / name, parse_dates=["snap_date"]).set_index("snap_date")


def _load_summary() -> dict:
    return json.loads((TABLE_DIR / "empirical_summary.json").read_text(encoding="utf-8"))


PERF_DIAG_COLS = [
    "Strategy",
    "Downside ann. dev",
    "Max drawdown",
    "Worst month",
    "SR 90\\% CI lo",
    "SR 90\\% CI hi",
    "Pred./realized vol",
    "Gross NAV",
    "Net NAV",
]


def rebuild_performance_diagnostics(gross: pd.DataFrame, net: pd.DataFrame, summary: dict) -> None:
    """Fixed CI columns for the gross and post-cost diagnostics tables (F3)."""

    for frame, record_key, table_name in (
        (gross, "performance_gross_only", "portfolio_performance_diagnostics.tex"),
        (net, "performance_post_cost", "portfolio_performance_net_diagnostics.tex"),
    ):
        records = {r["Strategy"]: r for r in summary.get(record_key, [])}
        rows = []
        for name in frame.columns:
            rec = records.get(name, {})
            row = emp.performance_summary_row(
                name,
                frame[name].dropna(),
                None,
                pred_realized_vol=_as_float(rec.get("Pred./realized vol")),
                gross_nav=_as_float(rec.get("Gross NAV")),
                net_nav=_as_float(rec.get("Net NAV")),
            )
            # Cross-check the recomputation against the cached summary values.
            for key in ("Sharpe", "Sortino", "Max drawdown"):
                cached = _as_float(rec.get(key))
                if np.isfinite(cached) and np.isfinite(row[key]) and abs(cached - row[key]) > 1e-6:
                    raise AssertionError(f"{table_name}: {name}/{key} mismatch {row[key]} vs cached {cached}")
            rows.append(row)
            # Patch the summary records with the corrected/added CI fields.
            if rec:
                for key in ("SR 90\\% CI lo", "SR 90\\% CI hi", "Sortino 90\\% CI lo", "Sortino 90\\% CI hi"):
                    rec[key] = _json_float(row[key])
        diag = pd.DataFrame(rows)[PERF_DIAG_COLS]
        write_table(diag, table_name)
        print(f"regenerated {table_name}")
    summary["performance"] = list(summary.get("performance_gross_only", [])) + list(
        summary.get("performance_post_cost", [])
    )


def rebuild_reality_check(gross: pd.DataFrame, scenario: pd.DataFrame, summary: dict) -> None:
    variants = pd.concat([gross.add_suffix("::gross"), scenario], axis=1)
    reality = sharpe_reality_check(variants, config=BootstrapConfig(n_boot=1000, seed=SEED))
    reality.to_csv(ART_DIR / "reality_check_inference.csv", index=False)
    summary["reality_check_inference"] = json.loads(reality.to_json(orient="records"))
    write_table(reality, "reality_check_inference.tex", canonical_cols=("Variant",))
    print("regenerated reality_check_inference (p = %.3f, max PSR = %.3f)" % (
        float(reality["Reality check p"].iloc[0]),
        float(reality["Probabilistic Sharpe"].max()),
    ))


def rebuild_inference_summary(gross: pd.DataFrame, net: pd.DataFrame, summary: dict) -> None:
    """Fixed-code inference table; Sharpe/CI values reproduce the cached ones.

    Information-ratio rows require the SPY benchmark series, which needs raw
    vendor data; those rows were computed correctly by the original pipeline
    and are carried through from the cached artifact unchanged.
    """

    artifact = pd.read_csv(ART_DIR / "inference_summary.csv")
    metric_order = ["Sharpe", "Sortino", "Calmar", "Omega", "Information Ratio"]
    out_rows = []
    for frame, basis in ((gross, "Gross before costs"), (net, "Post-cost research")):
        fresh = strategy_metric_inference(
            frame, None, metrics=("sharpe", "sortino", "calmar", "omega"), config=BootstrapConfig(n_boot=1000)
        )
        fresh["Return basis"] = basis
        cached = artifact[artifact["Return basis"].eq(basis)]
        for strategy in frame.columns:
            for metric in metric_order:
                if metric == "Information Ratio":
                    row = cached[cached["Strategy"].eq(strategy) & cached["Metric"].eq(metric)]
                else:
                    row = fresh[fresh["Strategy"].eq(strategy) & fresh["Metric"].eq(metric)]
                    check = cached[cached["Strategy"].eq(strategy) & cached["Metric"].eq(metric)]
                    if not row.empty and not check.empty:
                        a, b = float(row["Estimate"].iloc[0]), float(check["Estimate"].iloc[0])
                        if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 1e-8:
                            raise AssertionError(f"inference mismatch {strategy}/{metric}/{basis}: {a} vs {b}")
                if not row.empty:
                    out_rows.append(row.iloc[0])
    inference = pd.DataFrame(out_rows).reset_index(drop=True)
    inference = inference[list(artifact.columns)]
    inference.to_csv(ART_DIR / "inference_summary.csv", index=False)
    summary["inference"] = json.loads(inference.to_json(orient="records"))
    write_table(inference, "inference_summary.tex")
    print("regenerated inference_summary")


def rebuild_simulations(gross: pd.DataFrame, net: pd.DataFrame, scenario: pd.DataFrame, summary: dict) -> None:
    strategies = emp.SIMULATION_STRATEGIES
    gross_sim = gross[[c for c in strategies if c in gross.columns]]
    post = pd.DataFrame(index=gross.index)
    for strategy in strategies:
        scenario_col = f"{strategy}::full_spread"
        if scenario_col in scenario.columns:
            post[strategy] = scenario[scenario_col]
        elif strategy in net.columns:
            post[strategy] = net[strategy]
    sim_summary, sim_assumptions, breaches, paths = run_tail_path_simulations(
        {"Gross before costs": gross_sim, "Full-spread post-cost": post},
        strategies=strategies,
        config=SimulationConfig(seed=SEED),
    )
    sim_summary.to_csv(ART_DIR / "simulation_summary.csv", index=False)
    sim_assumptions.to_csv(ART_DIR / "simulation_assumptions.csv", index=False)
    breaches.to_csv(ART_DIR / "drawdown_breach_rates.csv", index=False)
    for stale in ART_DIR.glob("simulation_paths_*ewma_residual_fallback*.csv"):
        stale.unlink()
    for key, frame in paths.items():
        frame.to_csv(ART_DIR / f"simulation_paths_{key}.csv", index=False)
    summary["simulation_summary"] = json.loads(sim_summary.to_json(orient="records"))
    summary["simulation_assumptions"] = json.loads(sim_assumptions.to_json(orient="records"))
    summary["drawdown_breach_rates"] = json.loads(breaches.to_json(orient="records"))

    write_table(compact_simulation_summary(sim_summary), "simulation_summary.tex", canonical_cols=("Simulation",))
    breach_table = breaches.rename(columns={c: str(c).replace("%", "\\%") for c in breaches.columns})
    write_table(breach_table, "drawdown_breach_rates.tex", canonical_cols=("Requested method", "Simulation"))
    write_table(compact_assumptions(sim_assumptions), "simulation_assumptions.tex", canonical_cols=("Method",))
    worst_terminal = float(pd.to_numeric(sim_summary.get("Terminal wealth p95"), errors="coerce").abs().max())
    if not worst_terminal < 1e12:
        raise AssertionError(f"terminal wealth still diverges: {worst_terminal}")
    print(
        "regenerated simulation tables (max |terminal wealth p95| = %.3f, defaulted-share max = %.3f)"
        % (worst_terminal, float(pd.to_numeric(sim_summary.get("Defaulted path share"), errors="coerce").max()))
    )


def rebuild_formatting_only_tables() -> None:
    """Same numbers as the artifact CSVs; publication formatting only."""

    jobs: list[tuple[str, str, tuple[str, ...]]] = [
        ("post_cost_survival.csv", "post_cost_survival.tex", ()),
        ("capacity_market_impact_diagnostics.csv", "capacity_market_impact_diagnostics.tex", ("Scenario",)),
        ("cost_capacity_margin_diagnostics.csv", "cost_capacity_margin_diagnostics.tex", ()),
        ("liquidity_tier_performance.csv", "liquidity_tier_performance.tex", ("Liquidity tier",)),
        ("regime_performance.csv", "regime_performance.tex", ()),
        ("vix_regime_performance.csv", "vix_regime_performance.tex", ()),
        ("leave_one_out.csv", "leave_one_out.tex", ()),
        ("rolling_oos.csv", "rolling_oos.tex", ()),
    ]
    for csv_name, tex_name, canon in jobs:
        frame = pd.read_csv(ART_DIR / csv_name)
        write_table(frame, tex_name, canonical_cols=canon)
        print(f"rewrote (formatting only) {tex_name}")

    # Forecast ablation: label the sample basis. The artifact was produced by
    # forecast_ablation_tables(...) on the gross out-of-sample test returns
    # (see run_empirics.run_all), so the basis column is derived from the
    # pipeline call site rather than the CSV itself.
    ablation = pd.read_csv(ART_DIR / "forecast_ablation_performance.csv")
    ablation.insert(1, "Return basis", "Gross OOS test window")
    write_table(ablation, "forecast_ablation_performance.tex", canonical_cols=("Ablation",))
    print("rewrote (formatting only) forecast_ablation_performance.tex")


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_float(value: object) -> float | None:
    v = _as_float(value)
    return float(v) if np.isfinite(v) else None


def refresh_hash_manifest() -> None:
    hash_paths = []
    for directory in (TABLE_DIR, FIG_DIR, ART_DIR):
        hash_paths.extend([p for p in directory.rglob("*") if p.is_file()])
    hash_paths.extend(
        [
            PAPER / "option_only_portfolio_optimization_dhruv_kohli.tex",
            PAPER / "REPRODUCIBILITY.md",
            PAPER / "environment_lock.json",
        ]
    )
    artifact_hash_manifest(hash_paths, PAPER).to_csv(PAPER / "artifact_hash_manifest.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-simulation", action="store_true", help="Skip the (slower) tail-path simulation rebuild.")
    args = parser.parse_args(argv)

    gross = _read_returns("strategy_returns.csv")
    net = _read_returns("strategy_returns_post_cost.csv")
    scenario = _read_returns("net_strategy_returns_by_cost_scenario.csv")
    summary = _load_summary()

    rebuild_performance_diagnostics(gross, net, summary)
    rebuild_reality_check(gross, scenario, summary)
    rebuild_inference_summary(gross, net, summary)
    if not args.skip_simulation:
        rebuild_simulations(gross, net, scenario, summary)
    rebuild_formatting_only_tables()

    (TABLE_DIR / "empirical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    refresh_hash_manifest()
    print("empirical_summary.json updated; hash manifest refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
