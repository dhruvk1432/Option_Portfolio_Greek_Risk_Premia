"""P1 regularization sweep for option-only breadth degradation."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    EstimatorKnobs,
    build_training_context,
    evaluate,
    gross_sharpe_for_weights,
    naive_weights,
    rebuild_model,
    resolve_cov_shrinkage,
    solve_gm,
)
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import (
    _available_new_names,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    PRIMARY_UNDERLYINGS,
    ROOT,
)


OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "breadth_solutions"
RESULTS_CSV = "p1_regularization_results.csv"
RESULTS_JSON = "p1_regularization_results.json"
SUMMARY_MD = "p1_summary.md"
ANCHORS = {
    "orig+VIX": 1.3743892124363595,
    "larger+VIX": 0.7646532722533432,
    "orig": 0.8421199757145471,
    "larger": 0.4562219234084396,
}
ANCHOR_TOL = 0.05
DEFAULT_CONFIG_ORDER = ["orig+VIX", "larger+VIX", "orig", "larger"]
DEFAULT_ARM_ORDER = ["default", "A", "B", "C", "D", "E"]
POC_NOTE = (
    "spread inputs are source-audited; added-name and VIX rows use measured panel CBBO when "
    "present and otherwise use a point-in-time inferred CBBO proxy calibrated from the "
    "historical liquid equity/ETF CBBO surface; off-hours Cboe snapshots are rejected and "
    "the old blanket 10%/15% class defaults are not used in the breadth-solution reruns"
)
BASELINE_CBBO_NOTE = (
    "`orig` uses measured historical panel CBBO for all eight equity underlyings. "
    "`orig+VIX` uses the same exact equity-option CBBO rows, while VIX option spreads "
    "use the inferred liquid-option CBBO proxy."
)
N_SCALED_RULE = "min(0.90, 0.20 + 0.80*max(0, 1 - t_train/n_contracts))"


def build_configs() -> tuple[dict[str, tuple[list[str], tuple[str, ...], bool]], list[str]]:
    present_new = _available_new_names()
    configs = {
        "orig+VIX": (list(PRIMARY_UNDERLYINGS), (), True),
        "larger+VIX": (list(PRIMARY_UNDERLYINGS) + present_new, tuple(present_new), True),
        "orig": (list(PRIMARY_UNDERLYINGS), (), False),
        "larger": (list(PRIMARY_UNDERLYINGS) + present_new, tuple(present_new), False),
    }
    return configs, present_new


def build_knob_grid(selected_arms: Sequence[str]) -> list[dict[str, object]]:
    selected = set(selected_arms)
    candidates: list[tuple[str, str, EstimatorKnobs]] = []

    candidates.append(("default", "default", EstimatorKnobs()))
    for value in [0.20, 0.35, 0.50, 0.65, 0.80, 0.90, "n_scaled"]:
        candidates.append(("A", f"A_cov_{_slug(value)}", EstimatorKnobs(cov_shrinkage=value)))
    for residual in ["diag", "lw"]:
        for cov in [0.20, 0.50]:
            candidates.append(
                (
                    "B",
                    f"B_residual_{residual}_cov_{_slug(cov)}",
                    EstimatorKnobs(residual_estimator=residual, cov_shrinkage=cov),
                )
            )
    for under in ["lw", "single_factor"]:
        candidates.append(("C", f"C_under_{under}", EstimatorKnobs(under_cov_estimator=under)))
    for stz, hw in [(0.75, 0.25), (0.90, 0.25), (0.60, 0.0), (0.75, 0.0), (0.90, 0.0)]:
        candidates.append(
            (
                "D",
                f"D_stz_{_slug(stz)}_hw_{_slug(hw)}",
                EstimatorKnobs(shrinkage_to_zero=stz, historical_weight=hw, structural_weight=0.75),
            )
        )
    candidates.extend(
        [
            (
                "E",
                "E1_residual_diag_n_scaled_hw0_stz075",
                EstimatorKnobs(
                    residual_estimator="diag",
                    cov_shrinkage="n_scaled",
                    historical_weight=0.0,
                    shrinkage_to_zero=0.75,
                ),
            ),
            (
                "E",
                "E2_residual_lw_cov050_hw0_stz060",
                EstimatorKnobs(
                    residual_estimator="lw",
                    cov_shrinkage=0.50,
                    historical_weight=0.0,
                    shrinkage_to_zero=0.60,
                ),
            ),
            (
                "E",
                "E3_under_lw_residual_diag_n_scaled",
                EstimatorKnobs(
                    under_cov_estimator="lw",
                    residual_estimator="diag",
                    cov_shrinkage="n_scaled",
                    shrinkage_to_zero=0.60,
                    historical_weight=0.25,
                ),
            ),
            (
                "E",
                "E4_residual_diag_cov065_stz075",
                EstimatorKnobs(
                    residual_estimator="diag",
                    cov_shrinkage=0.65,
                    shrinkage_to_zero=0.75,
                    historical_weight=0.25,
                ),
            ),
        ]
    )

    points: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for arm, point_id, knobs in candidates:
        key = _knob_tuple(knobs)
        if key in seen:
            continue
        seen.add(key)
        if arm not in selected:
            continue
        points.append({"arm": arm, "point_id": point_id, "knobs": knobs})
    return points


def run_config(
    label: str,
    underlyings: Sequence[str],
    poc_names: Sequence[str],
    with_vix: bool,
    grid: Sequence[dict[str, object]],
    aum: float,
) -> tuple[list[dict[str, object]], list[str]]:
    messages: list[str] = []
    ctx = build_training_context(label, underlyings, poc_names, with_vix)
    strategies: dict[str, pd.Series] = {}
    metadata_by_strategy: dict[str, dict[str, object]] = {}

    default_strategy_key = ""
    default_weights: pd.Series | None = None
    for point in grid:
        arm = str(point["arm"])
        point_id = str(point["point_id"])
        knobs = point["knobs"]
        assert isinstance(knobs, EstimatorKnobs)
        model = rebuild_model(ctx, knobs)
        weights, status = solve_gm(model, "cvxpy")
        progress_gross = gross_sharpe_for_weights(ctx, model, weights)
        strategy_key = f"GM|{arm}|{point_id}"
        strategies[strategy_key] = weights
        resolved = resolve_cov_shrinkage(knobs, len(ctx.spec.index), len(ctx.train_returns))
        metadata_by_strategy[strategy_key] = {
            "config": label,
            "strategy": "Greek Markowitz",
            "arm": arm,
            "point_id": point_id,
            **_knob_fields(knobs),
            "resolved_lambda": resolved,
            "t_train": int(len(ctx.train_returns)),
            "n_underlyings": int(len(set(ctx.universe))),
            "n_contracts": int(len(ctx.base_model.contracts)),
            "solver_status": status,
        }
        if arm == "default" and point_id == "default":
            default_strategy_key = strategy_key
            default_weights = weights
            base_weights, base_status = solve_gm(ctx.base_model, "cvxpy")
            match = bool(
                status != "infeasible"
                and base_status != "infeasible"
                and np.allclose(
                    weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                    base_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                    atol=1e-10,
                )
            )
            _print_match(
                messages,
                f"{label} default rebuilt weights",
                match,
                f"rebuilt_status={status} base_status={base_status}",
            )
        print(f"{label} {arm} {point_id}: status={status} gross={progress_gross:.6f}", flush=True)

    for name, weights in naive_weights(ctx.base_model).items():
        strategies[name] = weights
        metadata_by_strategy[name] = {
            "config": label,
            "strategy": name,
            "arm": "naive",
            "point_id": name,
            **_empty_knob_fields(),
            "resolved_lambda": np.nan,
            "t_train": int(len(ctx.train_returns)),
            "n_underlyings": int(len(set(ctx.universe))),
            "n_contracts": int(len(ctx.base_model.contracts)),
            "solver_status": "reference",
        }

    eval_frame = evaluate(
        ctx,
        strategies,
        aums=[float(aum)],
        cost_kwargs={
            "impact_cost_rate": 0.0,
            "use_current_spread_assumptions": False,
            "use_inferred_spread_proxy": True,
        },
    )
    eval_by_strategy = eval_frame.set_index("strategy")

    rows: list[dict[str, object]] = []
    for strategy_key in strategies:
        row = dict(metadata_by_strategy[strategy_key])
        eval_row = eval_by_strategy.loc[strategy_key]
        row["aum"] = float(aum)
        row["gross_sharpe"] = float(eval_row["gross_sharpe"])
        row["net_sharpe_noimpact"] = float(eval_row["net_sharpe"])
        rows.append(row)

    if default_strategy_key and default_weights is not None:
        direct_gross = gross_sharpe_for_weights(ctx, ctx.base_model, default_weights)
        eval_gross = float(eval_by_strategy.loc[default_strategy_key, "gross_sharpe"])
        _print_match(
            messages,
            f"{label} default evaluate gross cross-check",
            abs(direct_gross - eval_gross) <= 1e-12,
            f"direct={direct_gross:.15f} eval={eval_gross:.15f}",
        )
        expected = ANCHORS[label]
        _print_match(
            messages,
            f"{label} default gross anchor",
            abs(eval_gross - expected) <= ANCHOR_TOL,
            f"value={eval_gross:.15f} expected={expected:.15f} tol={ANCHOR_TOL:.2f}",
        )

    return rows, messages


def write_outputs(
    rows: list[dict[str, object]],
    out_dir: Path,
    knob_grid: Sequence[dict[str, object]],
    selected_configs: Sequence[str],
    selected_arms: Sequence[str],
    present_new: Sequence[str],
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results[
            [
                "config",
                "strategy",
                "arm",
                "point_id",
                "cov_shrinkage",
                "under_cov_estimator",
                "vol_cov_estimator",
                "residual_estimator",
                "shrinkage_to_zero",
                "historical_weight",
                "structural_weight",
                "resolved_lambda",
                "t_train",
                "n_underlyings",
                "n_contracts",
                "solver_status",
                "aum",
                "gross_sharpe",
                "net_sharpe_noimpact",
            ]
        ].copy()
    csv_path = out_dir / RESULTS_CSV
    json_path = out_dir / RESULTS_JSON
    summary_path = out_dir / SUMMARY_MD
    results.to_csv(csv_path, index=False)

    payload = {
        "rows": _json_records(results),
        "provenance": {
            "git_rev": _git_rev(),
            "knob_grid": _grid_for_json(knob_grid),
            "selected_configs": list(selected_configs),
            "selected_arms": list(selected_arms),
            "present_new_count": int(len(present_new)),
            "present_new": list(present_new),
            "n_scaled_rule": N_SCALED_RULE,
            "anchors": ANCHORS,
            "poc_note": POC_NOTE,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    summary_path.write_text(build_summary(results), encoding="utf-8")
    return csv_path, json_path, summary_path


def build_summary(results: pd.DataFrame) -> str:
    lines = [
        "# P1 Regularization Sweep",
        "",
        f"POC note: {POC_NOTE}",
        "",
        f"Baseline note: {BASELINE_CBBO_NOTE}",
        "",
        "## Results",
        "",
    ]
    if results.empty:
        lines.append("No rows produced.")
        return "\n".join(lines) + "\n"

    table = results.sort_values(["config", "gross_sharpe"], ascending=[True, False]).copy()
    lines.extend(_markdown_table(table))
    lines.extend(["", "## Verdict", ""])
    for config, bar_config in [("larger", "orig"), ("larger+VIX", "orig+VIX")]:
        sub = results[
            results["config"].eq(config)
            & results["strategy"].eq("Greek Markowitz")
            & results["arm"].ne("default")
        ]
        if sub.empty:
            lines.append(f"- {config}: not run; verdict unavailable.")
            continue
        best = sub.sort_values("gross_sharpe", ascending=False).iloc[0]
        equal = results[results["config"].eq(config) & results["strategy"].eq("Equal premium")]
        equal_gross = float(equal.iloc[0]["gross_sharpe"]) if not equal.empty else float("nan")
        bar = ANCHORS[bar_config]
        passed = bool(float(best["gross_sharpe"]) >= bar)
        lines.append(
            "- "
            f"{config}: best GM {float(best['gross_sharpe']):.3f} "
            f"({best['arm']}/{best['point_id']}) vs 8-name default gross bar {bar:.3f} "
            f"and Equal-premium gross {_fmt(equal_gross)}. "
            f"{'PASS' if passed else 'FAIL'} regularized 56-name gross >= 8-name gross."
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIG_ORDER), help="comma-separated config subset")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARM_ORDER), help="comma-separated arm subset")
    parser.add_argument("--aum", type=float, default=1_000_000.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    selected_configs = _parse_subset(args.configs, DEFAULT_CONFIG_ORDER, "configs")
    selected_arms = _parse_subset(args.arms, DEFAULT_ARM_ORDER, "arms")
    configs, present_new = build_configs()
    grid = build_knob_grid(selected_arms)
    if not grid:
        raise SystemExit("no knob points selected")

    print(f"new names present in panel: {len(present_new)}/48", flush=True)
    print(f"selected configs: {', '.join(selected_configs)}", flush=True)
    print(f"selected arms: {', '.join(selected_arms)}", flush=True)
    print(f"knob points: {len(grid)}", flush=True)

    all_rows: list[dict[str, object]] = []
    messages: list[str] = []
    for label in selected_configs:
        underlyings, poc_names, with_vix = configs[label]
        print(
            f"running {label}: requested_underlyings={len(underlyings)} with_vix={with_vix}",
            flush=True,
        )
        rows, config_messages = run_config(label, underlyings, poc_names, with_vix, grid, args.aum)
        all_rows.extend(rows)
        messages.extend(config_messages)

    csv_path, json_path, summary_path = write_outputs(
        all_rows,
        args.out_dir,
        grid,
        selected_configs,
        selected_arms,
        present_new,
    )

    misses = [m for m in messages if m.startswith("MISS ")]
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    if misses:
        print(f"anchor failures: {len(misses)}", flush=True)
        return 1
    return 0


def _parse_subset(raw: str, valid_order: Sequence[str], label: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = sorted(set(values) - set(valid_order))
    if invalid:
        raise SystemExit(f"unknown {label}: {', '.join(invalid)}")
    return [value for value in valid_order if value in set(values)]


def _knob_tuple(knobs: EstimatorKnobs) -> tuple[object, ...]:
    return (
        knobs.cov_shrinkage,
        knobs.under_cov_estimator,
        knobs.vol_cov_estimator,
        knobs.residual_estimator,
        knobs.shrinkage_to_zero,
        knobs.historical_weight,
        knobs.structural_weight,
    )


def _knob_fields(knobs: EstimatorKnobs) -> dict[str, object]:
    return asdict(knobs)


def _empty_knob_fields() -> dict[str, object]:
    return {
        "cov_shrinkage": np.nan,
        "under_cov_estimator": "",
        "vol_cov_estimator": "",
        "residual_estimator": "",
        "shrinkage_to_zero": np.nan,
        "historical_weight": np.nan,
        "structural_weight": np.nan,
    }


def _grid_for_json(grid: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for point in grid:
        knobs = point["knobs"]
        assert isinstance(knobs, EstimatorKnobs)
        out.append({"arm": point["arm"], "point_id": point["point_id"], "knobs": _knob_fields(knobs)})
    return out


def _print_match(messages: list[str], name: str, ok: bool, details: str = "") -> None:
    line = f"{'MATCH' if ok else 'MISS'} {name}{(': ' + details) if details else ''}"
    messages.append(line)
    print(line, flush=True)


def _slug(value: object) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}".replace(".", "p")


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
    return [_clean_json_record(record) for record in frame.to_dict(orient="records")]


def _clean_json_record(record: dict[str, object]) -> dict[str, object]:
    return {key: _json_default(value) for key, value in record.items()}


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


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    cols = [
        "config",
        "strategy",
        "arm",
        "point_id",
        "solver_status",
        "n_underlyings",
        "n_contracts",
        "gross_sharpe",
        "net_sharpe_noimpact",
    ]
    lines = [
        "| Config | Strategy | Arm | Point | Status | Underlyings | Contracts | Gross Sharpe | Net Sharpe No-Impact |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in frame[cols].itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.config),
                    str(row.strategy),
                    str(row.arm),
                    str(row.point_id),
                    str(row.solver_status),
                    str(int(row.n_underlyings)),
                    str(int(row.n_contracts)),
                    _fmt(row.gross_sharpe),
                    _fmt(row.net_sharpe_noimpact),
                ]
            )
            + " |"
        )
    return lines


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
