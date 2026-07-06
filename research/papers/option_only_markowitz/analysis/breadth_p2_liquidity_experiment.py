"""P2 pre-trade liquidity-cap sweep for option-only breadth experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
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
    CapConstrainedMarkowitzModel,
    EstimatorKnobs,
    TrainingContext,
    build_training_context,
    cap_feasibility,
    capped_naive_weights,
    compute_liquidity_caps,
    evaluate,
    naive_weights,
    rebuild_model,
    solve_gm,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    PRIMARY_UNDERLYINGS,
    ROOT,
)


RESULTS_CSV = "p2_liquidity_results.csv"
CAPS_CSV = "p2_caps_detail.csv"
RESULTS_JSON = "p2_liquidity_results.json"
SUMMARY_MD = "p2_summary.md"
PRIMARY_CONFIG_ORDER = ["orig+VIX", "larger+VIX"]
NO_VIX_CONFIG_ORDER = ["orig", "larger"]
DEFAULT_PARTICIPATIONS = [0.02, 0.05, 0.10]
DEFAULT_AUMS = [1_000_000.0, 5_000_000.0, 10_000_000.0, 25_000_000.0]
ANCHORS = {
    "orig_slsqp_gross": 0.8421194565895301,
    "orig_slsqp_net": -1.4425788318790798,
}


def run_experiment(
    selected_configs: Sequence[str],
    participations: Sequence[float],
    aums: Sequence[float],
    out_dir: Path,
) -> int:
    configs, present_new = build_configs()
    contexts: dict[str, TrainingContext] = {}
    messages: list[str] = []

    def get_context(label: str) -> TrainingContext:
        if label not in contexts:
            underlyings, poc_names, with_vix = configs[label]
            print(
                f"building {label}: requested_underlyings={len(underlyings)} with_vix={with_vix}",
                flush=True,
            )
            contexts[label] = build_training_context(label, underlyings, poc_names, with_vix)
        return contexts[label]

    _run_anchors(get_context, messages)

    result_rows: list[dict[str, object]] = []
    cap_detail_rows: list[dict[str, object]] = []
    max_cap_violation = 0.0

    for label in selected_configs:
        ctx = get_context(label)
        cost_inputs = _build_cost_inputs(ctx)
        spread_proxy_fill = bool(
            (
                cost_inputs.get("relative_spread_source", pd.Series("", index=cost_inputs.index))
                .fillna("")
                .astype(str)
                .isin(["inferred_cbbo_proxy", "current_cboe_liquid_quote"])
                & cost_inputs.get("asset_class", pd.Series("", index=cost_inputs.index))
                .fillna("")
                .astype(str)
                .isin(["equity_option", "vix_option"])
            ).any()
        )
        baseline_weights, baseline_status = solve_gm(ctx.base_model, "cvxpy")
        naive = naive_weights(ctx.base_model)
        print(f"{label} baseline X=inf: status={baseline_status}", flush=True)

        capped_by_aum: dict[float, list[dict[str, object]]] = {}
        for aum in aums:
            capped_by_aum[float(aum)] = []
            for participation in participations:
                capped_specs, details, violation = _solve_capped_pair(ctx, participation, float(aum))
                capped_by_aum[float(aum)].extend(capped_specs)
                capped_by_aum[float(aum)].extend(_solve_capped_naive_specs(ctx, participation, float(aum)))
                cap_detail_rows.extend(details)
                max_cap_violation = max(max_cap_violation, violation)

        for aum in aums:
            aum_value = float(aum)
            strategies: dict[str, pd.Series] = {"GM X=inf": baseline_weights}
            row_specs: list[dict[str, object]] = [
                {
                    "internal_strategy": "GM X=inf",
                    "strategy": "GM X=inf",
                    "participation": "inf",
                    "mode": "uncapped",
                    "solver_status": baseline_status,
                    "capacity_infeasible": False,
                    "sum_of_caps": np.nan,
                    "n_binding": np.nan,
                    "deployed_gross": float(baseline_weights.abs().sum()),
                }
            ]

            for spec in capped_by_aum[aum_value]:
                strategies[str(spec["internal_strategy"])] = spec["weights"]
                row_specs.append(spec)

            for name, weights in naive.items():
                strategies[name] = weights
                row_specs.append(
                    {
                        "internal_strategy": name,
                        "strategy": name,
                        "participation": "",
                        "mode": "naive",
                        "solver_status": "reference",
                        "capacity_infeasible": False,
                        "sum_of_caps": np.nan,
                        "n_binding": np.nan,
                        "deployed_gross": float(weights.abs().sum()),
                    }
                )

            eval_frame = evaluate(
                ctx,
                strategies,
                aums=[aum_value],
                cost_inputs=cost_inputs,
            ).set_index("strategy")

            for spec in row_specs:
                eval_row = eval_frame.loc[str(spec["internal_strategy"])]
                result_rows.append(
                    {
                        "config": label,
                        "strategy": spec["strategy"],
                        "participation": spec["participation"],
                        "aum": aum_value,
                        "mode": spec["mode"],
                        "solver_status": spec["solver_status"],
                        "capacity_infeasible": bool(spec["capacity_infeasible"]),
                        "sum_of_caps": spec["sum_of_caps"],
                        "n_binding": spec["n_binding"],
                        "deployed_gross": spec["deployed_gross"],
                        "gross_sharpe": float(eval_row["gross_sharpe"]),
                        "net_sharpe": float(eval_row["net_sharpe"]),
                        "gross_ann_ret": float(eval_row["gross_ann_ret"]),
                        "net_ann_ret": float(eval_row["net_ann_ret"]),
                        "mean_monthly_capacity_cost": float(eval_row["mean_monthly_capacity_cost"]),
                        "max_capacity_ratio": float(eval_row["max_capacity_ratio"]),
                        "mean_capacity_ratio": float(eval_row["mean_capacity_ratio"]),
                        "capacity_penalized_share": float(eval_row["capacity_penalized_share"]),
                        "spread_proxy_fill": spread_proxy_fill,
                    }
                )

        print(f"{label}: completed {len(aums)} AUM bucket(s)", flush=True)

    _print_match(
        messages,
        "capped optimal weights respect caps",
        max_cap_violation <= 1e-8,
        f"max_violation={max_cap_violation:.3e}",
    )

    results = pd.DataFrame(result_rows)
    caps_detail = pd.DataFrame(cap_detail_rows)
    _write_outputs(results, caps_detail, out_dir, selected_configs, participations, aums, present_new)
    return 1 if any(line.startswith("MISS ") for line in messages) else 0


def _solve_capped_pair(
    ctx: TrainingContext,
    participation: float,
    aum: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    caps_df = compute_liquidity_caps(ctx.reps, ctx.spec["mark"], nav=aum, participation=participation)
    feasibility = cap_feasibility(caps_df, ctx.base_model.constraints)
    cap_bound = caps_df["bound"]
    participation_label = _participation_label(participation)

    hard_model = rebuild_model(ctx, EstimatorKnobs(), per_contract_caps=cap_bound)
    hard_weights, hard_status = solve_gm(hard_model, "cvxpy")
    hard_violation = _cap_violation(hard_weights, cap_bound)
    specs = [
        _row_spec(
            internal_strategy=f"GM X={participation_label}",
            strategy=f"GM X={participation_label}",
            participation=participation_label,
            mode="hard",
            status=hard_status,
            capacity_infeasible=not bool(feasibility["gross_feasible"]),
            feasibility=feasibility,
            weights=hard_weights,
        )
    ]
    detail_weights = hard_weights
    max_violation = hard_violation if hard_status == "optimal" else 0.0

    if not bool(feasibility["gross_feasible"]):
        relaxed_constraints = replace(ctx.base_model.constraints, gross_nav=float(feasibility["suggested_gross"]))
        relaxed_model = rebuild_model(
            ctx,
            EstimatorKnobs(),
            per_contract_caps=cap_bound,
            constraints=relaxed_constraints,
        )
        relaxed_weights, relaxed_status = solve_gm(relaxed_model, "cvxpy")
        relaxed_violation = _cap_violation(relaxed_weights, cap_bound)
        specs.append(
            _row_spec(
                internal_strategy=f"GM X={participation_label} relaxed",
                strategy=f"GM X={participation_label}",
                participation=participation_label,
                mode="relaxed",
                status=relaxed_status,
                capacity_infeasible=True,
                feasibility=feasibility,
                weights=relaxed_weights,
            )
        )
        detail_weights = relaxed_weights if relaxed_status == "optimal" else hard_weights
        if relaxed_status == "optimal":
            max_violation = max(max_violation, relaxed_violation)

    print(
        f"{ctx.label} X={participation_label} aum={aum:.0f}: "
        f"hard_status={hard_status} gross_feasible={feasibility['gross_feasible']}",
        flush=True,
    )
    return specs, _cap_detail_rows(ctx, participation_label, aum, caps_df, detail_weights), max_violation


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
        specs.append(
            _row_spec(
                internal_strategy=f"{name} capped X={participation_label}",
                strategy=f"{name} capped X={participation_label}",
                participation=participation_label,
                mode=mode,
                status=status,
                capacity_infeasible=not gross_feasible,
                feasibility=feasibility,
                weights=capped_naive_weights(weights, cap_bound, target_gross),
            )
        )
    return specs


def _row_spec(
    internal_strategy: str,
    strategy: str,
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
        "participation": participation,
        "mode": mode,
        "solver_status": status,
        "capacity_infeasible": capacity_infeasible,
        "sum_of_caps": float(feasibility["sum_of_caps"]),
        "n_binding": int(feasibility["n_binding"]),
        "deployed_gross": float(weights.abs().sum()),
        "weights": weights,
    }


def _cap_detail_rows(
    ctx: TrainingContext,
    participation: str,
    aum: float,
    caps_df: pd.DataFrame,
    weights: pd.Series,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    aligned_weights = weights.reindex(caps_df.index).fillna(0.0)
    for asset_id, rec in caps_df.iterrows():
        bound = float(rec["bound"])
        weight = float(aligned_weights.loc[asset_id])
        rows.append(
            {
                "config": ctx.label,
                "participation": participation,
                "aum": float(aum),
                "asset_id": asset_id,
                "train_volume": float(rec["train_volume"]) if pd.notna(rec["train_volume"]) else np.nan,
                "cap_contracts": float(rec["cap_contracts"]) if pd.notna(rec["cap_contracts"]) else np.nan,
                "w_cap": float(rec["w_cap"]) if pd.notna(rec["w_cap"]) else np.nan,
                "bound": bound,
                "has_volume": bool(rec["has_volume"]),
                "weight": weight,
                "utilization": abs(weight) / bound if bound > 0 else np.inf,
            }
        )
    return rows


def _run_anchors(get_context, messages: list[str]) -> None:
    ctx_vix = get_context("orig+VIX")
    cap_none = CapConstrainedMarkowitzModel(
        ctx_vix.base_model.options,
        ctx_vix.base_model.shocks,
        ctx_vix.base_model.expected_returns,
        residual_cov=ctx_vix.residuals.cov().fillna(0.0),
        constraints=ctx_vix.base_model.constraints,
        covariance_shrinkage=0.20,
        per_contract_caps=None,
    )
    base_weights, base_status = solve_gm(ctx_vix.base_model, "cvxpy")
    cap_weights, cap_status = solve_gm(cap_none, "cvxpy")
    _print_match(
        messages,
        "orig+VIX cap-none CVXPY weights byte-identical",
        base_status != "infeasible"
        and cap_status != "infeasible"
        and np.array_equal(base_weights.to_numpy(dtype=float), cap_weights.to_numpy(dtype=float)),
        f"base_status={base_status} cap_status={cap_status}",
    )

    ctx_orig = get_context("orig")
    slsqp_weights, slsqp_status = solve_gm(ctx_orig.base_model, "slsqp")
    eval_row = evaluate(ctx_orig, {"Greek Markowitz": slsqp_weights}, aums=[1_000_000]).iloc[0]
    gross = float(eval_row["gross_sharpe"])
    net = float(eval_row["net_sharpe"])
    _print_match(
        messages,
        "orig SLSQP gross cost anchor",
        slsqp_status != "infeasible" and abs(gross - ANCHORS["orig_slsqp_gross"]) <= 0.02,
        f"value={gross:.15f} expected={ANCHORS['orig_slsqp_gross']:.15f} status={slsqp_status}",
    )
    _print_match(
        messages,
        "orig SLSQP net cost anchor",
        slsqp_status != "infeasible" and abs(net - ANCHORS["orig_slsqp_net"]) <= 0.02,
        f"value={net:.15f} expected={ANCHORS['orig_slsqp_net']:.15f} status={slsqp_status}",
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
    caps_detail: pd.DataFrame,
    out_dir: Path,
    selected_configs: Sequence[str],
    participations: Sequence[float],
    aums: Sequence[float],
    present_new: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = results[_result_columns()].copy() if not results.empty else pd.DataFrame(columns=_result_columns())
    caps_cols = [
        "config",
        "participation",
        "aum",
        "asset_id",
        "train_volume",
        "cap_contracts",
        "w_cap",
        "bound",
        "has_volume",
        "weight",
        "utilization",
    ]
    caps_detail = caps_detail[caps_cols].copy() if not caps_detail.empty else pd.DataFrame(columns=caps_cols)

    results_path = out_dir / RESULTS_CSV
    caps_path = out_dir / CAPS_CSV
    json_path = out_dir / RESULTS_JSON
    summary_path = out_dir / SUMMARY_MD
    results.to_csv(results_path, index=False)
    caps_detail.to_csv(caps_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "rows": _json_records(results),
                "caps_detail_rows": int(len(caps_detail)),
                "provenance": {
                    "git_rev": _git_rev(),
                    "sweep_grid": {
                        "configs": list(selected_configs),
                        "participations": [float(x) for x in participations],
                        "aums": [float(x) for x in aums],
                        "impact_cost_rate": ResearchCostConfig().impact_cost_rate,
                    },
                    "present_new_count": int(len(present_new)),
                    "present_new": list(present_new),
                    "poc_note": POC_NOTE,
                    "anchors": ANCHORS,
                },
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(build_summary(results), encoding="utf-8")
    print(f"wrote {results_path}", flush=True)
    print(f"wrote {caps_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)


def build_summary(results: pd.DataFrame) -> str:
    lines = [
        "# P2 Liquidity-Cap Sweep",
        "",
        f"POC note: {POC_NOTE}",
        "",
        f"Baseline note: {BASELINE_CBBO_NOTE}",
        "",
    ]
    if results.empty:
        lines.append("No rows produced.")
        return "\n".join(lines) + "\n"

    for config in results["config"].drop_duplicates():
        sub = results[results["config"].eq(config)].copy()
        lines.extend([f"## {config}", ""])
        table = sub[
            [
                "strategy",
                "participation",
                "aum",
                "mode",
                "net_sharpe",
                "max_capacity_ratio",
                "capacity_infeasible",
            ]
        ].sort_values(["aum", "strategy", "mode"])
        lines.extend(_markdown_table(table))
        lines.extend(["", _verdict(config, sub), ""])
    return "\n".join(lines) + "\n"


def _verdict(config: str, sub: pd.DataFrame) -> str:
    capped = sub[
        sub["strategy"].astype(str).str.startswith("GM X=")
        & ~sub["strategy"].eq("GM X=inf")
        & sub["net_sharpe"].gt(0)
    ]
    baseline = sub[sub["strategy"].eq("GM X=inf") & sub["net_sharpe"].gt(0)]
    capped_aum = float(capped["aum"].max()) if not capped.empty else float("nan")
    baseline_aum = float(baseline["aum"].max()) if not baseline.empty else float("nan")
    uncapped_ratio = sub[sub["strategy"].eq("GM X=inf")]["max_capacity_ratio"].replace([np.inf, -np.inf], np.nan)
    capped_ratio = sub[
        sub["strategy"].astype(str).str.startswith("GM X=") & ~sub["strategy"].eq("GM X=inf")
    ]["max_capacity_ratio"].replace([np.inf, -np.inf], np.nan)
    uncapped_max = float(uncapped_ratio.max()) if uncapped_ratio.notna().any() else float("nan")
    capped_min = float(capped_ratio.min()) if capped_ratio.notna().any() else float("nan")
    collapsed = np.isfinite(uncapped_max) and np.isfinite(capped_min) and capped_min < uncapped_max
    return (
        f"Verdict: {config} largest positive-net capped AUM={_fmt_aum(capped_aum)} vs "
        f"uncapped positive-net AUM={_fmt_aum(baseline_aum)}. "
        f"Caps {'do' if collapsed else 'do not'} collapse max capacity ratio vs X=inf "
        f"(best capped {_fmt(capped_min)}, uncapped max {_fmt(uncapped_max)})."
    )


def _result_columns() -> list[str]:
    return [
        "config",
        "strategy",
        "participation",
        "aum",
        "mode",
        "solver_status",
        "capacity_infeasible",
        "sum_of_caps",
        "n_binding",
        "deployed_gross",
        "gross_sharpe",
        "net_sharpe",
        "gross_ann_ret",
        "net_ann_ret",
        "mean_monthly_capacity_cost",
        "max_capacity_ratio",
        "mean_capacity_ratio",
        "capacity_penalized_share",
        "spread_proxy_fill",
    ]


def _parse_csv_floats(raw: str, label: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid {label}: {raw!r}") from exc
    if not values:
        raise SystemExit(f"{label} must contain at least one value")
    return values


def _parse_configs(raw: str, include_no_vix: bool) -> list[str]:
    valid_order = PRIMARY_CONFIG_ORDER + (NO_VIX_CONFIG_ORDER if include_no_vix else [])
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = sorted(set(requested) - set(valid_order))
    if invalid:
        raise SystemExit(f"unknown configs: {', '.join(invalid)}")
    return [label for label in valid_order if label in set(requested)]


def _participation_label(value: float) -> str:
    return f"{float(value):.2f}"


def _cap_violation(weights: pd.Series, caps: pd.Series) -> float:
    aligned_w = weights.reindex(caps.index).abs().fillna(0.0)
    aligned_caps = caps.reindex(caps.index).fillna(np.inf)
    return float(np.max(np.maximum(aligned_w.to_numpy(dtype=float) - aligned_caps.to_numpy(dtype=float), 0.0)))


def _print_match(messages: list[str], name: str, ok: bool, details: str = "") -> None:
    line = f"{'MATCH' if ok else 'MISS'} {name}{(': ' + details) if details else ''}"
    messages.append(line)
    print(line, flush=True)


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Strategy | X | AUM | Mode | Net Sharpe | Max Capacity Ratio | Capacity Infeasible |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.participation} | {float(row.aum):.0f} | {row.mode} | "
            f"{_fmt(row.net_sharpe)} | {_fmt(row.max_capacity_ratio)} | {bool(row.capacity_infeasible)} |"
        )
    return lines


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_aum(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.0f}"


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
    parser.add_argument("--configs", default=None)
    parser.add_argument("--participations", default=",".join(str(x) for x in DEFAULT_PARTICIPATIONS))
    parser.add_argument("--aums", default=",".join(str(int(x)) for x in DEFAULT_AUMS))
    parser.add_argument("--include-no-vix", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    default_configs = PRIMARY_CONFIG_ORDER + (NO_VIX_CONFIG_ORDER if args.include_no_vix else [])
    configs = _parse_configs(args.configs or ",".join(default_configs), args.include_no_vix)
    participations = _parse_csv_floats(args.participations, "participations")
    aums = _parse_csv_floats(args.aums, "aums")
    print(f"selected configs: {', '.join(configs)}", flush=True)
    print(f"participations: {', '.join(_participation_label(x) for x in participations)}", flush=True)
    print(f"aums: {', '.join(f'{x:.0f}' for x in aums)}", flush=True)
    return run_experiment(configs, participations, aums, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
