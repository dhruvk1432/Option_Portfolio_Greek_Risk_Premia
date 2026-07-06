"""P3 combined regularization and liquidity-cap decision experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import (
    BASELINE_CBBO_NOTE,
    OUT_DIR,
    POC_NOTE,
    build_configs,
)
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    EstimatorKnobs,
    TrainingContext,
    build_training_context,
    cap_feasibility,
    capped_naive_weights,
    compute_liquidity_caps,
    delta_neutral_weights,
    evaluate,
    gross_sharpe_for_weights,
    naive_weights,
    rebuild_model,
    solve_gm,
    spread_source_coverage,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.run_empirics import ROOT


RESULTS_CSV = "p3_combined_results.csv"
RESULTS_JSON = "p3_combined_results.json"
DECISION_MD = "p3_decision_table.md"
SPREAD_COVERAGE_CSV = "p3_spread_source_coverage.csv"
P1_RESULTS = OUT_DIR / "p1_regularization_results.csv"
CONFIG_ORDER = ["orig+VIX", "larger+VIX", "orig", "larger"]
DEFAULT_AUMS = [1_000_000.0, 5_000_000.0, 10_000_000.0, 25_000_000.0]
PRIMARY_KNOBS = EstimatorKnobs(shrinkage_to_zero=0.75, historical_weight=0.0)
ALT_KNOBS = EstimatorKnobs(
    residual_estimator="diag",
    cov_shrinkage="n_scaled",
    historical_weight=0.0,
    shrinkage_to_zero=0.75,
)
ORIG_VIX_GROSS_ANCHOR = 1.3743892124363595


def run_experiment(
    selected_configs: Sequence[str],
    aums: Sequence[float],
    participation: float,
    primary_knobs: EstimatorKnobs,
    alt_knobs: EstimatorKnobs | None,
    out_dir: Path,
) -> int:
    configs, present_new = build_configs()
    p1_results = _load_p1_results()
    messages: list[str] = []
    contexts: dict[str, TrainingContext] = {}
    rows: list[dict[str, object]] = []
    spread_coverage_rows: list[pd.DataFrame] = []
    spread_proxy_fill_by_config: dict[str, bool] = {}
    max_cap_violation = 0.0

    def get_context(label: str) -> TrainingContext:
        if label not in contexts:
            underlyings, poc_names, with_vix = configs[label]
            print(
                f"building {label}: requested_underlyings={len(underlyings)} with_vix={with_vix}",
                flush=True,
            )
            contexts[label] = build_training_context(label, underlyings, poc_names, with_vix)
        return contexts[label]

    for label in selected_configs:
        ctx = get_context(label)
        cost_inputs = _build_cost_inputs(ctx)
        spread_cov = spread_source_coverage(label, cost_inputs)
        spread_coverage_rows.append(spread_cov)
        spread_proxy_fill_by_config[label] = bool(
            (
                spread_cov["relative_spread_source"].astype(str).isin(
                    ["inferred_cbbo_proxy", "current_cboe_liquid_quote"]
                )
                & spread_cov["asset_class"].astype(str).isin(["equity_option", "vix_option"])
            ).any()
            if not spread_cov.empty
            else False
        )
        paper_weights, paper_status = solve_gm(ctx.base_model, "cvxpy")
        delta_weights = delta_neutral_weights(ctx, ctx.base_model, caps=None, method="cvxpy")
        naive = naive_weights(ctx.base_model)
        print(f"{label} GM paper: status={paper_status}", flush=True)

        capped_by_aum: dict[float, list[dict[str, object]]] = {}
        for aum in aums:
            aum_value = float(aum)
            capped_by_aum[aum_value] = []
            primary_specs, primary_violation = _solve_combined(
                ctx,
                "GM combined",
                "primary",
                primary_knobs,
                participation,
                aum_value,
                p1_results,
            )
            capped_by_aum[aum_value].extend(primary_specs)
            max_cap_violation = max(max_cap_violation, primary_violation)
            if alt_knobs is not None:
                alt_specs, alt_violation = _solve_combined(
                    ctx,
                    "GM combined alt",
                    "alt",
                    alt_knobs,
                    participation,
                    aum_value,
                    p1_results,
                )
                capped_by_aum[aum_value].extend(alt_specs)
                max_cap_violation = max(max_cap_violation, alt_violation)
            capped_by_aum[aum_value].extend(
                _solve_capped_naive_specs(ctx, participation, aum_value)
            )

        for aum in aums:
            aum_value = float(aum)
            strategies: dict[str, pd.Series] = {
                "GM paper": paper_weights,
                "Delta neutral": delta_weights,
                "Equal premium": naive["Equal premium"],
                "Equal risk": naive["Equal risk"],
            }
            row_specs = [
                _reference_spec("GM paper", "default", "uncapped", paper_status, paper_weights),
                _reference_spec("Delta neutral", "delta", "uncapped", "reference", delta_weights),
                _reference_spec("Equal premium", "naive", "naive", "reference", naive["Equal premium"]),
                _reference_spec("Equal risk", "naive", "naive", "reference", naive["Equal risk"]),
            ]
            for spec in capped_by_aum[aum_value]:
                strategies[spec["internal_strategy"]] = spec["weights"]
                row_specs.append(spec)

            eval_frame = evaluate(
                ctx,
                strategies,
                aums=[aum_value],
                cost_inputs=cost_inputs,
            ).set_index("strategy")

            for spec in row_specs:
                eval_row = eval_frame.loc[spec["internal_strategy"]]
                rows.append(
                    {
                        "config": label,
                        "strategy": spec["strategy"],
                        "knobs_label": spec["knobs_label"],
                        "participation": spec["participation"],
                        "aum": aum_value,
                        "mode": spec["mode"],
                        "solver_status": spec["solver_status"],
                        "capacity_infeasible": bool(spec["capacity_infeasible"]),
                        "sum_of_caps": spec["sum_of_caps"],
                        "deployed_gross": spec["deployed_gross"],
                        "gross_sharpe": float(eval_row["gross_sharpe"]),
                        "net_sharpe": float(eval_row["net_sharpe"]),
                        "gross_ann_ret": float(eval_row["gross_ann_ret"]),
                        "net_ann_ret": float(eval_row["net_ann_ret"]),
                        "mean_monthly_capacity_cost": float(eval_row["mean_monthly_capacity_cost"]),
                        "max_capacity_ratio": float(eval_row["max_capacity_ratio"]),
                        "capacity_penalized_share": float(eval_row["capacity_penalized_share"]),
                        "spread_proxy_fill": bool(spread_proxy_fill_by_config.get(label, False)),
                    }
                )

        _run_paper_gross_anchor(label, rows, p1_results, messages)
        print(f"{label}: completed {len(aums)} AUM bucket(s)", flush=True)

    _print_match(
        messages,
        "capped optimal weights respect caps",
        max_cap_violation <= 1e-8,
        f"max_violation={max_cap_violation:.3e}",
    )

    results = pd.DataFrame(rows)
    spread_coverage = (
        pd.concat(spread_coverage_rows, ignore_index=True, sort=False)
        if spread_coverage_rows
        else pd.DataFrame()
    )
    _write_outputs(
        results,
        spread_coverage,
        out_dir,
        selected_configs,
        aums,
        participation,
        primary_knobs,
        alt_knobs,
        present_new,
    )
    return 1 if any(line.startswith("MISS ") for line in messages) else 0


def _solve_combined(
    ctx: TrainingContext,
    strategy: str,
    knobs_label: str,
    knobs: EstimatorKnobs,
    participation: float,
    aum: float,
    p1_results: pd.DataFrame,
) -> tuple[list[dict[str, object]], float]:
    caps_df = compute_liquidity_caps(ctx.reps, ctx.spec["mark"], nav=aum, participation=participation)
    feasibility = cap_feasibility(caps_df, ctx.base_model.constraints)
    cap_bound = caps_df["bound"]
    participation_label = _participation_label(participation)

    hard_model = rebuild_model(ctx, knobs, per_contract_caps=cap_bound)
    hard_weights, hard_status = solve_gm(hard_model, "cvxpy")
    hard_violation = _cap_violation(hard_weights, cap_bound)
    specs = [
        _capped_spec(
            internal_strategy=f"{strategy} {knobs_label} hard",
            strategy=strategy,
            knobs_label=knobs_label,
            participation=participation_label,
            mode="hard",
            status=hard_status,
            capacity_infeasible=not bool(feasibility["gross_feasible"]),
            feasibility=feasibility,
            weights=hard_weights,
        )
    ]
    max_violation = hard_violation if hard_status == "optimal" else 0.0
    _print_p1_comparison(ctx, strategy, "hard", aum, knobs, hard_weights, cap_bound, p1_results)

    if not bool(feasibility["gross_feasible"]):
        relaxed_constraints = replace(ctx.base_model.constraints, gross_nav=float(feasibility["suggested_gross"]))
        relaxed_model = rebuild_model(
            ctx,
            knobs,
            per_contract_caps=cap_bound,
            constraints=relaxed_constraints,
        )
        relaxed_weights, relaxed_status = solve_gm(relaxed_model, "cvxpy")
        relaxed_violation = _cap_violation(relaxed_weights, cap_bound)
        specs.append(
            _capped_spec(
                internal_strategy=f"{strategy} {knobs_label} relaxed",
                strategy=strategy,
                knobs_label=knobs_label,
                participation=participation_label,
                mode="relaxed",
                status=relaxed_status,
                capacity_infeasible=True,
                feasibility=feasibility,
                weights=relaxed_weights,
            )
        )
        if relaxed_status == "optimal":
            max_violation = max(max_violation, relaxed_violation)
        _print_p1_comparison(ctx, strategy, "relaxed", aum, knobs, relaxed_weights, cap_bound, p1_results)

    print(
        f"{ctx.label} {strategy} aum={aum:.0f} X={participation_label}: "
        f"hard_status={hard_status} gross_feasible={feasibility['gross_feasible']}",
        flush=True,
    )
    return specs, max_violation


def _solve_capped_naive_specs(
    ctx: TrainingContext,
    participation: float,
    aum: float,
) -> list[dict[str, object]]:
    caps_df = compute_liquidity_caps(ctx.reps, ctx.spec["mark"], nav=aum, participation=participation)
    feasibility = cap_feasibility(caps_df, ctx.base_model.constraints)
    cap_bound = caps_df["bound"]
    participation_label = _participation_label(participation)
    gross_feasible = bool(feasibility["gross_feasible"])
    target_gross = (
        float(ctx.base_model.constraints.gross_nav)
        if gross_feasible
        else float(feasibility["suggested_gross"])
    )
    mode = "hard" if gross_feasible else "relaxed"
    status = "reference" if target_gross > 1e-14 else "infeasible"

    specs: list[dict[str, object]] = []
    for name, weights in naive_weights(ctx.base_model).items():
        capped_weights = capped_naive_weights(weights, cap_bound, target_gross)
        specs.append(
            _capped_spec(
                internal_strategy=f"{name} capped {participation_label}",
                strategy=f"{name} capped",
                knobs_label="naive_capped",
                participation=participation_label,
                mode=mode,
                status=status,
                capacity_infeasible=not gross_feasible,
                feasibility=feasibility,
                weights=capped_weights,
            )
        )
    return specs


def _reference_spec(
    strategy: str,
    knobs_label: str,
    mode: str,
    status: str,
    weights: pd.Series,
) -> dict[str, object]:
    return {
        "internal_strategy": strategy,
        "strategy": strategy,
        "knobs_label": knobs_label,
        "participation": "",
        "mode": mode,
        "solver_status": status,
        "capacity_infeasible": False,
        "sum_of_caps": np.nan,
        "deployed_gross": float(weights.abs().sum()),
        "weights": weights,
    }


def _capped_spec(
    internal_strategy: str,
    strategy: str,
    knobs_label: str,
    participation: str,
    mode: str,
    status: str,
    capacity_infeasible: bool,
    feasibility: dict[str, object],
    weights: pd.Series,
) -> dict[str, object]:
    return {
        "internal_strategy": internal_strategy,
        "strategy": strategy,
        "knobs_label": knobs_label,
        "participation": participation,
        "mode": mode,
        "solver_status": status,
        "capacity_infeasible": capacity_infeasible,
        "sum_of_caps": float(feasibility["sum_of_caps"]),
        "deployed_gross": float(weights.abs().sum()),
        "weights": weights,
    }


def _run_paper_gross_anchor(
    label: str,
    rows: list[dict[str, object]],
    p1_results: pd.DataFrame,
    messages: list[str],
) -> None:
    sub = [
        row
        for row in rows
        if row["config"] == label
        and row["strategy"] == "GM paper"
        and row["mode"] == "uncapped"
    ]
    if not sub:
        _print_match(messages, f"{label} GM paper gross P1 anchor", False, "missing GM paper row")
        return
    actual = float(sub[0]["gross_sharpe"])
    expected = _p1_default_gross(p1_results, label)
    _print_match(
        messages,
        f"{label} GM paper gross P1 anchor",
        np.isfinite(expected) and abs(actual - expected) <= 1e-6,
        f"value={actual:.15f} expected={expected:.15f}",
    )
    if label == "orig+VIX":
        _print_match(
            messages,
            "orig+VIX GM paper fixed gross anchor",
            abs(actual - ORIG_VIX_GROSS_ANCHOR) <= 1e-6,
            f"value={actual:.15f} expected={ORIG_VIX_GROSS_ANCHOR:.15f}",
        )


def _build_cost_inputs(ctx: TrainingContext) -> pd.DataFrame:
    cfg = ResearchCostConfig(
        use_current_spread_assumptions=False,
        use_inferred_spread_proxy=True,
    )
    surface = load_cbbo_spread_surface(ROOT, cfg.cbbo_spread_surface_path) if cfg.use_cbbo_spread_surface else None
    return build_cost_input_ledger(ctx.reps, ctx.detail, ROOT, cfg, spread_surface=surface)


def _write_outputs(
    results: pd.DataFrame,
    spread_coverage: pd.DataFrame,
    out_dir: Path,
    selected_configs: Sequence[str],
    aums: Sequence[float],
    participation: float,
    primary_knobs: EstimatorKnobs,
    alt_knobs: EstimatorKnobs | None,
    present_new: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = results[_result_columns()].copy() if not results.empty else pd.DataFrame(columns=_result_columns())
    csv_path = out_dir / RESULTS_CSV
    json_path = out_dir / RESULTS_JSON
    md_path = out_dir / DECISION_MD
    spread_path = out_dir / SPREAD_COVERAGE_CSV
    results.to_csv(csv_path, index=False)
    spread_cols = [
        "config",
        "relative_spread_source",
        "asset_class",
        "rows",
        "asset_ids",
        "underlyings",
        "mean_relative_spread",
        "median_relative_spread",
    ]
    spread_coverage = (
        spread_coverage[spread_cols].copy()
        if not spread_coverage.empty
        else pd.DataFrame(columns=spread_cols)
    )
    spread_coverage.to_csv(spread_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "rows": _json_records(results),
                "spread_coverage": _json_records(spread_coverage),
                "provenance": {
                    "git_rev": _git_rev(),
                    "configs": list(selected_configs),
                    "aums": [float(x) for x in aums],
                    "participation": float(participation),
                    "primary_knobs": asdict(primary_knobs),
                    "alt_knobs": asdict(alt_knobs) if alt_knobs is not None else None,
                    "impact_cost_rate": ResearchCostConfig().impact_cost_rate,
                    "present_new_count": int(len(present_new)),
                    "present_new": list(present_new),
                    "poc_note": POC_NOTE,
                    "anchors": {"orig+VIX GM paper gross": ORIG_VIX_GROSS_ANCHOR},
                },
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    md_path.write_text(build_decision_table(results, spread_coverage), encoding="utf-8")
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {spread_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


def build_decision_table(results: pd.DataFrame, spread_coverage: pd.DataFrame | None = None) -> str:
    lines = [
        "# P3 Combined Decision Table",
        "",
        f"POC note: {POC_NOTE}",
        "",
        f"Baseline note: {BASELINE_CBBO_NOTE}",
        "",
        "Net cells are marked with `*` when that config uses `inferred_cbbo_proxy` rows for VIX options or added-name equity options without historical panel CBBO. The regenerated checked-in run does not consume current Cboe spread fills; exact panel rows remain unmarked in the spread-source coverage table.",
        "",
        "## Decision Table",
        "",
    ]
    if results.empty:
        lines.append("No rows produced.")
        return "\n".join(lines) + "\n"

    lines.extend(_decision_markdown(results))
    if spread_coverage is not None and not spread_coverage.empty:
        lines.extend(["", "## Spread Source Coverage", ""])
        lines.extend(_markdown_table(spread_coverage))
    lines.extend(["", "## Verdict", ""])
    lines.extend(_verdict_lines(results))
    return "\n".join(lines) + "\n"


def _decision_markdown(results: pd.DataFrame) -> list[str]:
    aums = sorted(float(x) for x in results["aum"].dropna().unique())
    headers = [
        "Config",
        "Strategy",
        "Mode",
        "Gross Sharpe",
        *[f"Net@{_aum_header(aum)}" for aum in aums],
        "Breakeven AUM",
        "Capacity Infeasible AUMs",
        "Deployed Gross@25M",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    grouped = results.copy()
    grouped["decision_strategy"] = grouped.apply(_decision_strategy_name, axis=1)
    for (config, strategy, mode), grp in grouped.groupby(["config", "decision_strategy", "mode"], sort=False):
        gross = float(grp["gross_sharpe"].dropna().iloc[0]) if grp["gross_sharpe"].notna().any() else np.nan
        net_cells = []
        for aum in aums:
            sub = grp[grp["aum"].eq(aum)]
            value = float(sub.iloc[0]["net_sharpe"]) if not sub.empty else np.nan
            has_poc_fill = (
                bool(sub["spread_proxy_fill"].astype(bool).any())
                if not sub.empty and "spread_proxy_fill" in sub
                else False
            )
            suffix = "*" if has_poc_fill and np.isfinite(value) else ""
            net_cells.append(_fmt(value) + suffix)
        infeasible = grp[grp["capacity_infeasible"].astype(bool)]["aum"].tolist()
        dg25 = grp[grp["aum"].eq(25_000_000.0)]["deployed_gross"]
        deployed_25 = float(dg25.iloc[0]) if not dg25.empty else np.nan
        row = [
            str(config),
            str(strategy),
            str(mode),
            _fmt(gross),
            *net_cells,
            _breakeven_text(grp),
            ", ".join(_fmt_aum(float(x)) for x in infeasible) if infeasible else "",
            _fmt(deployed_25),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _verdict_lines(results: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for large, small in [("larger", "orig"), ("larger+VIX", "orig+VIX")]:
        large_gross = _strategy_gross(results, large, "GM combined", "hard")
        small_gross = _strategy_gross(results, small, "GM combined", "hard")
        if np.isfinite(large_gross) and np.isfinite(small_gross):
            if large == "larger+VIX":
                alt_large = _strategy_gross(results, large, "GM combined alt", "hard")
                alt_small = _strategy_gross(results, small, "GM combined alt", "hard")
                if np.isfinite(alt_large) and np.isfinite(alt_small):
                    status = "PASS" if alt_large >= alt_small else "FAIL"
                    primary_status = "passes" if large_gross >= small_gross else "fails"
                    lines.append(
                        f"- Breadth pays gross? {large} vs {small}: {status} on the selected E1 "
                        f"regularized/capped row ({alt_large:.3f} vs {alt_small:.3f}); the "
                        f"primary combined row also {primary_status} ({large_gross:.3f} vs "
                        f"{small_gross:.3f})."
                    )
                    continue
            status = "PASS" if large_gross >= small_gross else "FAIL"
            lines.append(
                f"- Breadth pays gross? {large} vs {small}: {status} on the primary "
                f"regularized row ({large_gross:.3f} vs {small_gross:.3f})."
            )
    for config in ["larger", "larger+VIX"]:
        gm = _best_family_by_aum(results, config, ["GM combined", "GM combined alt"], "hard")
        naive = _best_family_by_aum(results, config, ["Equal premium capped", "Equal risk capped"], "hard")
        if gm.empty or naive.empty:
            continue
        merged = gm[["aum", "strategy", "net_sharpe"]].merge(
            naive[["aum", "strategy", "net_sharpe"]],
            on="aum",
            suffixes=("_gm", "_capped_naive"),
        )
        diffs = [
            f"{_fmt_aum(float(r.aum))}: {float(r.net_sharpe_gm - r.net_sharpe_capped_naive):.3f}"
            for r in merged.itertuples()
        ]
        cross = _crossover_text(merged["aum"], merged["net_sharpe_gm"] - merged["net_sharpe_capped_naive"])
        lines.append(
            f"- Optimizer vs capped naive at breadth (net), {config}: "
            f"{'; '.join(diffs)}. Crossover: {cross}."
        )
    for large, small in [("larger", "orig"), ("larger+VIX", "orig+VIX")]:
        large_be = _breakeven_value(_best_family_by_aum(results, large, ["GM combined", "GM combined alt"], "hard"))
        small_be = _breakeven_value(_best_family_by_aum(results, small, ["GM combined", "GM combined alt"], "hard"))
        paper_be = _breakeven_value(_strategy_mode(results, large, "GM paper", "uncapped"))
        if any(np.isfinite(x) for x in [large_be, small_be, paper_be]):
            lines.append(
                f"- Capacity, {large}: best regularized GM breakeven {_fmt_aum(large_be)} vs "
                f"{small} {_fmt_aum(small_be)} and GM-paper {_fmt_aum(paper_be)}."
            )
    return lines or ["- Verdict unavailable for the selected smoke subset."]


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["(empty)"]
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in cols) + " |")
    return lines


def _format_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_p1_results() -> pd.DataFrame:
    if not P1_RESULTS.exists():
        raise FileNotFoundError(f"P1 results not found: {P1_RESULTS}")
    return pd.read_csv(P1_RESULTS)


def _best_family_by_aum(
    results: pd.DataFrame,
    config: str,
    strategies: Sequence[str],
    mode: str,
) -> pd.DataFrame:
    sub = results[
        results["config"].eq(config)
        & results["strategy"].isin(strategies)
        & results["mode"].eq(mode)
    ].copy()
    if sub.empty:
        return sub
    sub["net_sharpe"] = pd.to_numeric(sub["net_sharpe"], errors="coerce")
    sub = sub.dropna(subset=["net_sharpe"])
    if sub.empty:
        return sub
    idx = sub.groupby("aum")["net_sharpe"].idxmax()
    return sub.loc[idx].sort_values("aum").copy()


def _p1_default_gross(p1_results: pd.DataFrame, config: str) -> float:
    sub = p1_results[
        p1_results["config"].eq(config)
        & p1_results["strategy"].eq("Greek Markowitz")
        & p1_results["arm"].eq("default")
        & p1_results["point_id"].eq("default")
    ]
    return float(sub.iloc[0]["gross_sharpe"]) if not sub.empty else float("nan")


def _p1_gross_for_knobs(p1_results: pd.DataFrame, config: str, knobs: EstimatorKnobs) -> float:
    sub = p1_results[p1_results["config"].eq(config) & p1_results["strategy"].eq("Greek Markowitz")].copy()
    if sub.empty:
        return float("nan")
    fields = asdict(knobs)
    mask = pd.Series(True, index=sub.index)
    for col, value in fields.items():
        if isinstance(value, str):
            mask &= sub[col].astype(str).eq(value)
        else:
            mask &= np.isclose(pd.to_numeric(sub[col], errors="coerce"), float(value), atol=1e-12)
    matched = sub[mask]
    return float(matched.iloc[0]["gross_sharpe"]) if not matched.empty else float("nan")


def _print_p1_comparison(
    ctx: TrainingContext,
    strategy: str,
    mode: str,
    aum: float,
    knobs: EstimatorKnobs,
    weights: pd.Series,
    caps: pd.Series,
    p1_results: pd.DataFrame,
) -> None:
    p1_gross = _p1_gross_for_knobs(p1_results, ctx.label, knobs)
    actual_gross = gross_sharpe_for_weights(ctx, ctx.base_model, weights)
    diff = actual_gross - p1_gross if np.isfinite(p1_gross) and np.isfinite(actual_gross) else np.nan
    deployed = float(weights.abs().sum())
    active_caps = int((weights.reindex(caps.index).abs().fillna(0.0) >= caps - 1e-8).sum())
    print(
        f"INFO {ctx.label} {strategy} {mode} aum={aum:.0f}: "
        f"gross={_fmt(actual_gross)} P1_gross={_fmt(p1_gross)} diff={_fmt(diff)} "
        f"deployed_gross={deployed:.3f} active_caps={active_caps}",
        flush=True,
    )


def _parse_knobs(raw: str | None, default: EstimatorKnobs) -> EstimatorKnobs:
    if not raw:
        return default
    path = Path(raw)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("knobs JSON must decode to an object")
    merged = asdict(default)
    merged.update(data)
    return EstimatorKnobs(**merged)


def _parse_configs(raw: str) -> list[str]:
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = sorted(set(requested) - set(CONFIG_ORDER))
    if invalid:
        raise SystemExit(f"unknown configs: {', '.join(invalid)}")
    return [label for label in CONFIG_ORDER if label in set(requested)]


def _parse_aums(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("at least one AUM is required")
    return values


def _result_columns() -> list[str]:
    return [
        "config",
        "strategy",
        "knobs_label",
        "participation",
        "aum",
        "mode",
        "solver_status",
        "capacity_infeasible",
        "sum_of_caps",
        "deployed_gross",
        "gross_sharpe",
        "net_sharpe",
        "gross_ann_ret",
        "net_ann_ret",
        "mean_monthly_capacity_cost",
        "max_capacity_ratio",
        "capacity_penalized_share",
        "spread_proxy_fill",
    ]


def _cap_violation(weights: pd.Series, caps: pd.Series) -> float:
    aligned_w = weights.reindex(caps.index).abs().fillna(0.0)
    return float(np.max(np.maximum(aligned_w.to_numpy(dtype=float) - caps.to_numpy(dtype=float), 0.0)))


def _strategy_mode(results: pd.DataFrame, config: str, strategy: str, mode: str) -> pd.DataFrame:
    return results[results["config"].eq(config) & results["strategy"].eq(strategy) & results["mode"].eq(mode)].copy()


def _strategy_gross(results: pd.DataFrame, config: str, strategy: str, mode: str) -> float:
    sub = _strategy_mode(results, config, strategy, mode)
    return float(sub.iloc[0]["gross_sharpe"]) if not sub.empty else float("nan")


def _decision_strategy_name(row: pd.Series) -> str:
    strategy = str(row["strategy"])
    if strategy in {"GM combined", "GM combined alt"} and str(row["mode"]) == "relaxed":
        return f"{strategy} relaxed"
    return strategy


def _breakeven_text(grp: pd.DataFrame) -> str:
    grid = _breakeven_value(grp)
    crossing = _zero_crossing(grp["aum"], grp["net_sharpe"])
    if np.isfinite(crossing):
        return f"{_fmt_aum(grid)}; zero {_fmt_aum(crossing)}"
    return _fmt_aum(grid)


def _breakeven_value(grp: pd.DataFrame) -> float:
    positive = grp[pd.to_numeric(grp["net_sharpe"], errors="coerce").gt(0)]
    return float(positive["aum"].max()) if not positive.empty else float("nan")


def _zero_crossing(aums: Sequence[float], values: Sequence[float]) -> float:
    frame = pd.DataFrame({"aum": aums, "value": values}).dropna().sort_values("aum")
    if len(frame) < 2:
        return float("nan")
    for left, right in zip(frame.iloc[:-1].itertuples(index=False), frame.iloc[1:].itertuples(index=False)):
        y0, y1 = float(left.value), float(right.value)
        if y0 == 0:
            return float(left.aum)
        if y0 * y1 < 0:
            log0, log1 = np.log(float(left.aum)), np.log(float(right.aum))
            return float(np.exp(log0 + (0.0 - y0) * (log1 - log0) / (y1 - y0)))
    return float("nan")


def _crossover_text(aums: Sequence[float], diffs: Sequence[float]) -> str:
    crossing = _zero_crossing(aums, diffs)
    return _fmt_aum(crossing) if np.isfinite(crossing) else "none on grid"


def _participation_label(value: float) -> str:
    return f"{float(value):.2f}"


def _print_match(messages: list[str], name: str, ok: bool, details: str = "") -> None:
    line = f"{'MATCH' if ok else 'MISS'} {name}{(': ' + details) if details else ''}"
    messages.append(line)
    print(line, flush=True)


def _aum_header(aum: float) -> str:
    if aum >= 1_000_000:
        return f"{aum / 1_000_000:g}M"
    return f"{aum:g}"


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_aum(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


def _build_cost_inputs(ctx: TrainingContext) -> pd.DataFrame:
    cfg = ResearchCostConfig(
        use_current_spread_assumptions=False,
        use_inferred_spread_proxy=True,
    )
    surface = load_cbbo_spread_surface(ROOT, cfg.cbbo_spread_surface_path) if cfg.use_cbbo_spread_surface else None
    return build_cost_input_ledger(ctx.reps, ctx.detail, ROOT, cfg, spread_surface=surface)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [{key: _json_default(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(CONFIG_ORDER))
    parser.add_argument("--aums", default=",".join(str(int(x)) for x in DEFAULT_AUMS))
    parser.add_argument("--participation", type=float, default=0.05)
    parser.add_argument("--knobs-json", default=None)
    parser.add_argument("--alt-knobs-json", default=None)
    parser.add_argument("--no-alt", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    configs = _parse_configs(args.configs)
    aums = _parse_aums(args.aums)
    primary_knobs = _parse_knobs(args.knobs_json, PRIMARY_KNOBS)
    alt_knobs = None if args.no_alt else _parse_knobs(args.alt_knobs_json, ALT_KNOBS)
    print(f"selected configs: {', '.join(configs)}", flush=True)
    print(f"aums: {', '.join(f'{x:.0f}' for x in aums)}", flush=True)
    print(f"participation: {_participation_label(args.participation)}", flush=True)
    print(f"primary knobs: {asdict(primary_knobs)}", flush=True)
    print(f"alt knobs: {asdict(alt_knobs) if alt_knobs is not None else 'skipped'}", flush=True)
    return run_experiment(configs, aums, args.participation, primary_knobs, alt_knobs, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
