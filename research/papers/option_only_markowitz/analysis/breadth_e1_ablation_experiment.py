"""Run the locked E1 structural-premia channel ablation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_robustness_experiment import (
    CONFIG_ORDER,
    DEFAULT_NAV,
    DEFAULT_PARTICIPATION,
    E1_KNOBS,
    FittedBook,
    ROBUSTNESS_DIR,
    TABLE_DIR,
    build_cost_inputs,
    build_panels,
    build_training_from_panel,
    score_books,
    _series_stats,
)
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    cap_feasibility,
    compute_liquidity_caps,
    integerize_book_weights,
    rebuild_model,
    solve_gm,
)
from research.papers.option_only_markowitz.analysis.conditional_premia import (
    ConditionalPremiaConfig,
)
from research.papers.option_only_markowitz.analysis.run_empirics import TRAIN_END


ARM_ORDER = [
    "Full E1",
    "No carry",
    "No delta",
    "No vol/VRP",
    "No skew/tail",
    "No relative value",
]
WEIGHTS_CSV = "breadth_e1_book_weights.csv"


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, escape=False, float_format="%.3f"), encoding="utf-8")


def _base_premia_config() -> ConditionalPremiaConfig:
    return ConditionalPremiaConfig(
        horizon_years=21.0 / 252.0,
        historical_weight=E1_KNOBS.historical_weight,
        structural_weight=E1_KNOBS.structural_weight,
        shrinkage_to_zero=E1_KNOBS.shrinkage_to_zero,
    )


def _premia_config_for_arm(arm: str) -> ConditionalPremiaConfig | None:
    base = _base_premia_config()
    if arm == "Full E1":
        return None
    if arm == "No carry":
        return replace(base, carry_scale=0.0)
    if arm == "No delta":
        return replace(base, equity_scale=0.0)
    if arm == "No vol/VRP":
        return replace(base, vrp_scale=0.0)
    if arm == "No skew/tail":
        return replace(base, skew_scale=0.0, tail_hedge_credit=0.0)
    if arm == "No relative value":
        return replace(base, relative_value_scale=0.0)
    raise ValueError(f"unknown ablation arm: {arm!r}")


def _fit_e1_book(
    ctx,
    *,
    config: str,
    arm: str,
    nav: float,
    participation: float,
) -> tuple[FittedBook, pd.DataFrame]:
    caps_df = compute_liquidity_caps(
        ctx.reps,
        ctx.spec["mark"],
        nav=float(nav),
        participation=float(participation),
        train_end=pd.Timestamp(ctx.train_returns.index.max()),
    )
    feasibility = cap_feasibility(caps_df, ctx.base_model.constraints)
    model = rebuild_model(
        ctx,
        E1_KNOBS,
        per_contract_caps=caps_df["bound"],
        premia_config=_premia_config_for_arm(arm),
    )
    weights_cont, status = solve_gm(model, "cvxpy")
    weights = integerize_book_weights(
        weights_cont, ctx.spec["mark"], nav=float(nav), caps=caps_df["bound"]
    )["realized_weight"]
    return (
        FittedBook(
            config=config,
            strategy=arm,
            display_strategy=arm,
            weights=weights,
            model_contracts=pd.Index(model.contracts),
            solver_status=status,
            mode="hard",
            capacity_infeasible=not bool(feasibility["gross_feasible"]),
            sum_of_caps=float(feasibility["sum_of_caps"]),
            deployed_gross=float(weights.abs().sum()),
        ),
        caps_df,
    )


def _book_weight_ledger(
    ctx,
    book: FittedBook,
    caps_df: pd.DataFrame,
) -> pd.DataFrame:
    idx = pd.Index(book.model_contracts)
    weight = pd.Series(book.weights, dtype=float).reindex(idx).fillna(0.0)
    cap_bound = pd.to_numeric(caps_df["bound"], errors="coerce").reindex(idx)
    utilization = weight.abs() / cap_bound.where(cap_bound.gt(0.0))
    out = pd.DataFrame(
        {
            "config": book.config,
            "asset_id": idx.astype(str),
            "weight": weight.to_numpy(float),
            "cap_bound": cap_bound.to_numpy(float),
            "utilization": utilization.to_numpy(float),
        },
        index=idx,
    )
    spec = ctx.spec.reindex(idx)
    for col in ["underlying", "mark"]:
        if col in spec.columns:
            out[col] = spec[col].to_numpy()
    return out.reset_index(drop=True)


def _greek_exposures(spec: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    aligned = weights.reindex(spec.index).fillna(0.0)
    mark = pd.to_numeric(spec["mark"], errors="coerce").replace(0.0, np.nan)
    spot = pd.to_numeric(spec.get("spot", pd.Series(1.0, index=spec.index)), errors="coerce").fillna(1.0)
    delta_nav = pd.to_numeric(spec["delta"], errors="coerce").fillna(0.0) * spot / mark
    gamma_nav = pd.to_numeric(spec["gamma"], errors="coerce").fillna(0.0) * spot * spot / mark
    vega_nav = pd.to_numeric(spec["vega"], errors="coerce").fillna(0.0) / mark
    theta_nav = pd.to_numeric(spec["theta"], errors="coerce").fillna(0.0) * _base_premia_config().horizon_years / mark
    return {
        "net_delta_nav": float(delta_nav.fillna(0.0).dot(aligned)),
        "gamma_nav": float(gamma_nav.fillna(0.0).dot(aligned)),
        "vega_nav": float(vega_nav.fillna(0.0).dot(aligned)),
        "theta_nav": float(theta_nav.fillna(0.0).dot(aligned)),
    }


def _load_scoreboard() -> pd.DataFrame:
    path = ROBUSTNESS_DIR / "final_result_scoreboard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    scoreboard = pd.read_csv(path)
    required = {"config", "e1_net_sharpe"}
    missing = sorted(required.difference(scoreboard.columns))
    if missing:
        raise ValueError(f"final_result_scoreboard.csv missing columns: {missing}")
    return scoreboard


def _assert_full_e1_matches_scoreboard(results: pd.DataFrame) -> None:
    scoreboard = _load_scoreboard().set_index("config")
    full = results.loc[results["arm"].eq("Full E1")].set_index("config")
    for config in CONFIG_ORDER:
        actual = float(full.loc[config, "net_sharpe"])
        expected = float(scoreboard.loc[config, "e1_net_sharpe"])
        if not np.isfinite(actual) or abs(actual - expected) > 1e-6:
            raise RuntimeError(
                f"Full E1 net Sharpe mismatch for {config}: "
                f"recomputed {actual:.15f}, scoreboard {expected:.15f}"
            )


def run_ablation(
    selected_configs: Sequence[str] = CONFIG_ORDER,
    *,
    nav: float = DEFAULT_NAV,
    participation: float = DEFAULT_PARTICIPATION,
) -> pd.DataFrame:
    selected = [config for config in CONFIG_ORDER if config in set(selected_configs)]
    panels = build_panels(selected)
    rows: list[dict[str, object]] = []
    weight_ledgers: list[pd.DataFrame] = []
    for config in selected:
        print(f"[e1-ablation] fitting {config}", flush=True)
        panel = panels[config]
        ctx, status = build_training_from_panel(panel)
        if ctx is None:
            raise RuntimeError(f"{config} context failed: {status}")
        cost_inputs, _spread_coverage = build_cost_inputs(panel, nav)
        test_dates = pd.DatetimeIndex(panel.returns.index[panel.returns.index > TRAIN_END])
        for arm in ARM_ORDER:
            book, caps_df = _fit_e1_book(
                ctx,
                config=config,
                arm=arm,
                nav=nav,
                participation=participation,
            )
            gross, net, _cost_ledger, _capacity_ledger = score_books(
                panel,
                {arm: book},
                cost_inputs,
                nav=nav,
                dates=test_dates,
            )
            gross_stats = _series_stats(gross[arm]) if arm in gross else {}
            net_stats = _series_stats(net[arm]) if arm in net else {}
            if arm == "Full E1":
                weight_ledgers.append(_book_weight_ledger(ctx, book, caps_df))
            exposures = (
                _greek_exposures(ctx.spec, book.weights)
                if arm == "Full E1"
                else {"net_delta_nav": np.nan, "gamma_nav": np.nan, "vega_nav": np.nan, "theta_nav": np.nan}
            )
            rows.append(
                {
                    "config": config,
                    "arm": arm,
                    "gross_sharpe": float(gross_stats.get("sharpe", np.nan)),
                    "net_sharpe": float(net_stats.get("sharpe", np.nan)),
                    "net_sortino": float(net_stats.get("sortino", np.nan)),
                    "deployed_gross": book.deployed_gross,
                    "n_active_contracts": int(book.weights.abs().gt(1e-6).sum()),
                    **exposures,
                    "solver_status": book.solver_status,
                }
            )
    results = pd.DataFrame(rows)
    _assert_full_e1_matches_scoreboard(results)
    results.attrs["full_e1_book_weights"] = (
        pd.concat(weight_ledgers, ignore_index=True, sort=False) if weight_ledgers else pd.DataFrame()
    )
    return results


def build_short_net_sharpe_table(results: pd.DataFrame) -> pd.DataFrame:
    required = {"config", "arm", "net_sharpe"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"ablation results missing columns: {missing}")
    pivot = (
        results.pivot_table(index="arm", columns="config", values="net_sharpe", aggfunc="first")
        .reindex(index=ARM_ORDER, columns=CONFIG_ORDER)
        .reset_index()
        .rename(columns={"arm": "Arm"})
    )
    pivot["Arm"] = pivot["Arm"].replace({"Full E1": "Full model"})
    pivot.columns.name = None
    return pivot


def write_outputs(results: pd.DataFrame, weights: pd.DataFrame | None = None) -> None:
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ROBUSTNESS_DIR / "breadth_e1_channel_ablation.csv", index=False)
    if weights is None:
        weights = results.attrs.get("full_e1_book_weights")
    if weights is None:
        raise ValueError("missing full E1 book weights ledger")
    weights.to_csv(ROBUSTNESS_DIR / WEIGHTS_CSV, index=False)
    _write_latex_table(build_short_net_sharpe_table(results), TABLE_DIR / "short_e1_channel_ablation.tex")


def main() -> int:
    results = run_ablation()
    write_outputs(results)
    print(build_short_net_sharpe_table(results).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
