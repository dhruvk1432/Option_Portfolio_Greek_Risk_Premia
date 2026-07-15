from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.breadth_robustness_experiment import (
    DEFAULT_CV_CONFIG,
    E1_KNOBS,
    _compact_cpcv_table,
    _compact_pbo_table,
    _compact_simulation_table,
    _write_short_cpcv_windows_table,
    build_relative_cpcv_paths,
    breadth_cost_config,
    expected_cv_split_count,
    spread_policy_status,
    _summary_lookup,
)
from research.papers.option_only_markowitz.analysis.cross_validation import (
    CVConfig,
    build_folds,
)
from research.papers.option_only_markowitz.analysis.breadth_e1_ablation_experiment import (
    ARM_ORDER,
    _write_latex_table as _write_ablation_latex_table,
    build_short_net_sharpe_table,
)
from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    EstimatorKnobs,
    TrainingContext,
    rebuild_model,
)
from research.papers.option_only_markowitz.analysis.conditional_premia import (
    ConditionalPremiaConfig,
)
from research.papers.option_only_markowitz.analysis import breadth_robustness_experiment
from research.papers.option_only_markowitz.analysis import build_final_results_summary
from research.papers.option_only_markowitz.analysis import build_e1_concentration
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
)


def test_default_cv_split_count_matches_plan():
    assert DEFAULT_CV_CONFIG.n_groups == 12
    assert DEFAULT_CV_CONFIG.n_test_groups == 2
    assert DEFAULT_CV_CONFIG.purge_months == 1
    assert DEFAULT_CV_CONFIG.embargo_months == 1
    assert expected_cv_split_count(DEFAULT_CV_CONFIG) == 12 + math.comb(12, 2)


def test_breadth_cost_config_uses_corrected_spread_policy():
    cfg = breadth_cost_config(nav=1_000_000)
    assert cfg.use_current_spread_assumptions is False
    assert cfg.use_inferred_spread_proxy is True
    assert cfg.nav_for_capacity == 1_000_000


def test_primary_knobs_are_locked_e1_specification():
    assert E1_KNOBS.residual_estimator == "diag"
    assert E1_KNOBS.cov_shrinkage == "n_scaled"
    assert E1_KNOBS.historical_weight == 0.0
    assert E1_KNOBS.shrinkage_to_zero == 0.75


def _synthetic_training_context() -> TrainingContext:
    contracts = pd.Index(["ABC_call_atm", "ABC_put_atm"], name="asset_id")
    spec = pd.DataFrame(
        {
            "underlying": ["ABC", "ABC"],
            "mark": [10.0, 12.0],
            "spot": [100.0, 100.0],
            "delta": [0.55, -0.45],
            "gamma": [0.02, 0.018],
            "vega": [0.30, 0.28],
            "theta": [-0.80, -0.65],
            "kind": ["call", "put"],
            "moneyness_bucket": ["atm", "atm"],
            "iv_proxy": [0.24, 0.28],
            "asset_class": ["equity_option", "equity_option"],
        },
        index=contracts,
    )
    spec.attrs["_breadth_solutions_augmented_spec"] = spec.copy()
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    train_returns = pd.DataFrame(
        {
            "ABC_call_atm": [0.02, -0.01, 0.03, 0.01],
            "ABC_put_atm": [-0.01, 0.02, -0.02, 0.00],
        },
        index=dates,
    )
    train_under = pd.DataFrame({"ABC": [0.01, -0.02, 0.03, 0.01]}, index=dates)
    train_vol = pd.DataFrame({"ABC": [0.01, 0.02, -0.01, 0.00]}, index=dates)
    residuals = train_returns - train_returns.mean()
    cov = pd.DataFrame([[0.0025]], index=["ABC"], columns=["ABC"])
    vol_cov = pd.DataFrame([[0.0004]], index=["ABC"], columns=["ABC"])
    residual_cov = pd.DataFrame(np.diag([0.0008, 0.0010]), index=contracts, columns=contracts)
    base_model = OptionOnlyMarkowitzModel(
        OptionOnlySpec(spec),
        FactorShockSpec(underlying_cov=cov, vol_cov=vol_cov),
        expected_returns=pd.Series([0.03, -0.01], index=contracts),
        residual_cov=residual_cov,
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.20),
        covariance_shrinkage=0.20,
    )
    return TrainingContext(
        label="synthetic",
        universe=["ABC"],
        reps=pd.DataFrame(),
        returns=train_returns,
        detail=pd.DataFrame(),
        spec=spec,
        base_model=base_model,
        residuals=residuals,
        train_returns=train_returns,
        train_under=train_under,
        train_vol=train_vol,
    )


def test_rebuild_model_premia_config_is_backward_compatible():
    ctx = _synthetic_training_context()
    rebuilt = rebuild_model(ctx, EstimatorKnobs())
    assert np.array_equal(
        rebuilt.expected_returns.to_numpy(dtype=float),
        ctx.base_model.expected_returns.to_numpy(dtype=float),
    )

    e1_like = EstimatorKnobs(historical_weight=0.0, shrinkage_to_zero=0.75, structural_weight=0.75)
    full = rebuild_model(
        ctx,
        e1_like,
        premia_config=ConditionalPremiaConfig(
            historical_weight=0.0,
            structural_weight=0.75,
            shrinkage_to_zero=0.75,
        ),
    )
    no_carry = rebuild_model(
        ctx,
        e1_like,
        premia_config=ConditionalPremiaConfig(
            historical_weight=0.0,
            structural_weight=0.75,
            shrinkage_to_zero=0.75,
            carry_scale=0.0,
        ),
    )
    assert not np.allclose(full.expected_returns.to_numpy(float), no_carry.expected_returns.to_numpy(float))


def test_e1_ablation_short_table_formats_net_sharpe(tmp_path):
    rows = []
    for arm_idx, arm in enumerate(ARM_ORDER):
        for config_idx, config in enumerate(["orig", "orig+VIX", "larger", "larger+VIX"]):
            rows.append(
                {
                    "config": config,
                    "arm": arm,
                    "net_sharpe": arm_idx + config_idx / 10.0,
                    "gross_sharpe": -99.0,
                }
            )
    table = build_short_net_sharpe_table(pd.DataFrame(rows))
    assert table.columns.tolist() == ["Arm", "orig", "orig+VIX", "larger", "larger+VIX"]
    assert table.columns.name is None
    assert ARM_ORDER[0] == "Full E1"
    assert table["Arm"].tolist() == ["Full model", *ARM_ORDER[1:]]
    assert float(table.loc[table["Arm"].eq("No vol/VRP"), "larger+VIX"].iloc[0]) == 3.3
    path = tmp_path / "short_e1_channel_ablation.tex"
    _write_ablation_latex_table(table, path)
    text = path.read_text(encoding="utf-8")
    assert "gross_sharpe" not in text
    assert "orig+VIX" in text
    assert "No vol/VRP" in text


def test_e1_concentration_reads_locked_weight_ledger(tmp_path, monkeypatch):
    robust_dir = tmp_path / "robustness"
    table_dir = tmp_path / "tables"
    robust_dir.mkdir()
    table_dir.mkdir()

    weight_rows = []
    realized_rows = []
    for idx, config in enumerate(["orig", "orig+VIX", "larger", "larger+VIX"]):
        scale = 1.0 + idx
        rows = [
            {
                "config": config,
                "asset_id": f"{config}_call",
                "weight": 0.20 * scale,
                "cap_bound": 0.20 * scale,
                "utilization": 1.0,
                "underlying": "ABC",
                "mark": 10.0,
            },
            {
                "config": config,
                "asset_id": f"{config}_put",
                "weight": -0.10 * scale,
                "cap_bound": 0.20 * scale,
                "utilization": 0.5,
                "underlying": "ABC",
                "mark": 8.0,
            },
            {
                "config": config,
                "asset_id": f"{config}_wing",
                "weight": 0.0,
                "cap_bound": 0.30 * scale,
                "utilization": 0.0,
                "underlying": "ABC",
                "mark": 2.0,
            },
        ]
        weight_rows.extend(rows)
        realized_rows.append(
            {
                "config": config,
                "strategy": "E1 capped",
                "deployed_gross": sum(abs(row["weight"]) for row in rows),
            }
        )
    pd.DataFrame(weight_rows).to_csv(robust_dir / "breadth_e1_book_weights.csv", index=False)
    pd.DataFrame(realized_rows).to_csv(robust_dir / "breadth_realized_candidate_summary.csv", index=False)

    monkeypatch.setattr(build_e1_concentration, "ROBUSTNESS_DIR", robust_dir)
    monkeypatch.setattr(build_e1_concentration, "TABLE_DIR", table_dir)

    panel = build_e1_concentration.build_concentration_panel()
    build_e1_concentration.write_outputs(panel)
    first = panel.set_index("config").loc["orig"]
    assert first["n_candidate_contracts"] == 3
    assert first["n_active_contracts"] == 2
    assert math.isclose(first["deployed_gross"], 0.30)
    assert math.isclose(first["top5_share"], 1.0)
    assert math.isclose(first["at_cap_share"], 0.5)
    assert math.isclose(first["cap_budget"], 0.70)
    assert (robust_dir / "final_e1_concentration.csv").exists()
    assert "Top 5 share" in (table_dir / "short_e1_concentration.tex").read_text(encoding="utf-8")


def test_spread_policy_status_is_fail_closed():
    ok = pd.DataFrame(
        {
            "relative_spread_source": ["panel_cbbo", "inferred_cbbo_proxy"],
            "rows": [10, 5],
        }
    )
    assert spread_policy_status(ok)["status"] == "pass"

    bad = pd.DataFrame(
        {
            "relative_spread_source": ["panel_cbbo", "default", "current_cboe_liquid_quote"],
            "rows": [10, 2, 3],
        }
    )
    status = spread_policy_status(bad)
    assert status["status"] == "fail"
    assert status["default_rows"] == 2
    assert status["current_cboe_rows"] == 3


def test_summary_lookup_accepts_resampled_summary_column_names():
    frame = pd.DataFrame(
        {
            "Strategy": ["larger+VIX E1 capped"],
            "Basis": ["full_cost_net"],
            "Path P05 Sharpe": [0.1],
            "Path P50 Sharpe": [1.2],
            "Path P95 Sharpe": [2.3],
        }
    )
    row = _summary_lookup(frame, "larger+VIX E1 capped", "full_cost_net")
    assert row["P05 Sharpe"] == 0.1
    assert row["P50 Sharpe"] == 1.2
    assert row["P95 Sharpe"] == 2.3


def test_build_relative_cpcv_paths_diffs_e1_against_capped_naive():
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    cfg = CVConfig(n_groups=2, n_test_groups=1, purge_months=0, embargo_months=0, min_train_months=0)
    folds = build_folds(dates, cfg, "cpcv")
    returns = {
        "E1 capped": [0.03, 0.04, 0.01, 0.02],
        "Equal premium capped": [0.01, 0.02, -0.01, 0.00],
        "Equal risk capped": [0.02, 0.01, 0.00, 0.01],
    }
    by_date = {
        strategy: dict(zip(dates, values))
        for strategy, values in returns.items()
    }
    rows = []
    for fold in folds:
        for date in fold.test_dates:
            for strategy in returns:
                rows.append(
                    {
                        "config": "orig",
                        "fold_id": fold.fold_id,
                        "scheme": fold.scheme,
                        "return_date": date,
                        "strategy": f"orig {strategy}",
                        "display_strategy": strategy,
                        "basis": "full_cost_net",
                        "ret": by_date[strategy][date],
                    }
                )

    metrics = build_relative_cpcv_paths(pd.DataFrame(rows), cfg)
    by_strategy = metrics.set_index("strategy")

    premium = by_strategy.loc["orig E1 minus Equal premium capped"]
    risk = by_strategy.loc["orig E1 minus Equal risk capped"]
    assert premium["status"] == "complete"
    assert risk["status"] == "complete"
    assert premium["basis"] == "full_cost_net"
    assert premium["n_months"] == 4
    assert math.isclose(float(premium["terminal_wealth"]), 1.02**4)
    assert math.isclose(float(risk["terminal_wealth"]), 1.01 * 1.03 * 1.01 * 1.01)


def test_compact_cpcv_table_reports_default_share():
    frame = pd.DataFrame(
        {
            "strategy": ["orig E1 capped", "orig E1 capped", "orig E1 capped"],
            "basis": ["full_cost_net", "full_cost_net", "full_cost_net"],
            "status": ["complete", "complete", "incomplete"],
            "sharpe": [0.2, -0.1, np.nan],
            "defaulted": [True, False, False],
        }
    )
    table = _compact_cpcv_table(frame)
    row = table.iloc[0]
    assert row["Paths"] == 2
    assert math.isclose(float(row["Default share"]), 0.5)


def test_short_cpcv_windows_table_reports_full_and_claim_relative_p05(tmp_path, monkeypatch):
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    monkeypatch.setattr(breadth_robustness_experiment, "TABLE_DIR", table_dir)

    configs = ["orig", "orig+VIX", "larger", "larger+VIX"]
    matched = {
        "orig": "Equal premium capped",
        "orig+VIX": "Equal premium capped",
        "larger": "Equal risk capped",
        "larger+VIX": "Equal risk capped",
    }
    summary = pd.DataFrame(
        [
            {
                "config": config,
                "strategy": "E1 capped",
                "cpcv_net_p05": -0.1,
                "cpcv_net_p50": 0.2,
                "cpcv_claim_net_p05": 0.3,
                "cpcv_claim_net_p50": 0.4,
                "rel_cpcv_net_p05": -99.0,
            }
            for config in configs
        ]
    )
    cpcv_metrics = pd.DataFrame(
        [
            {
                "strategy": f"{config} E1 capped",
                "basis": "full_cost_net",
                "status": "complete",
                "sharpe": 0.0,
                "defaulted": config in {"orig", "larger"},
            }
            for config in configs
        ]
    )
    relative_rows = []
    claim_relative_rows = []
    for config in configs:
        for naive in ["Equal premium capped", "Equal risk capped"]:
            base = 0.0 if naive == matched[config] else 9.0
            claim_base = 0.2 if naive == matched[config] else 8.0
            for value in [base, base + 1.0]:
                relative_rows.append(
                    {
                        "strategy": f"{config} E1 minus {naive}",
                        "basis": "full_cost_net",
                        "status": "complete",
                        "sharpe": value,
                    }
                )
            for value in [claim_base, claim_base + 1.0]:
                claim_relative_rows.append(
                    {
                        "strategy": f"{config} E1 minus {naive}",
                        "basis": "full_cost_net",
                        "status": "complete",
                        "sharpe": value,
                    }
                )
    final_scoreboard = pd.DataFrame(
        [{"config": config, "best_naive_strategy": matched[config]} for config in configs]
    )
    realized = pd.DataFrame(
        [
            {"config": "orig", "strategy": "Equal risk capped", "net_sharpe": 9.0},
            {"config": "orig", "strategy": "Equal premium capped", "net_sharpe": 0.0},
        ]
    )

    _write_short_cpcv_windows_table(
        summary,
        cpcv_metrics,
        relative_metrics=pd.DataFrame(relative_rows),
        claim_relative_metrics=pd.DataFrame(claim_relative_rows),
        final_scoreboard=final_scoreboard,
        realized=realized,
        claim_metrics_available=True,
    )

    text = (table_dir / "short_cpcv_windows.tex").read_text(encoding="utf-8")
    assert "Rel full p05" in text
    assert "Rel claim p05" in text
    assert "Rel naive p05" not in text
    assert "0.050" in text
    assert "0.250" in text
    assert "-99.000" not in text


def test_build_validation_path_values_omits_missing_or_partial_claim(tmp_path, monkeypatch):
    robust_dir = tmp_path / "robustness"
    robust_dir.mkdir()
    configs = build_final_results_summary.CONFIG_ORDER

    cpcv_rows = []
    resampled_rows = []
    refit_rows = []
    for config_idx, config in enumerate(configs):
        full_strategy = f"{config} E1 capped"
        for path in range(11):
            cpcv_rows.append(
                {
                    "strategy": full_strategy,
                    "basis": "full_cost_net",
                    "status": "complete",
                    "sharpe": config_idx + path / 100.0,
                }
            )
        for path in range(3):
            resampled_rows.append(
                {
                    "strategy": full_strategy,
                    "basis": "full_cost_net",
                    "universe_family": "resampled",
                    "sharpe": 0.2 + config_idx + path / 10.0,
                }
            )
        for path in range(2):
            refit_rows.append(
                {
                    "config": config,
                    "display_strategy": "E1 capped",
                    "basis": "full_cost_net",
                    "status": "ok",
                    "sharpe": 0.3 + config_idx + path / 10.0,
                }
            )
    pd.DataFrame(cpcv_rows).to_csv(robust_dir / "breadth_cv_cpcv_path_metrics.csv", index=False)
    pd.DataFrame(resampled_rows).to_csv(robust_dir / "breadth_mc_resampled_paths.csv", index=False)
    pd.DataFrame(refit_rows).to_csv(robust_dir / "breadth_mc_refit_paths.csv", index=False)

    monkeypatch.setattr(build_final_results_summary, "ROBUSTNESS_DIR", robust_dir)
    values = build_final_results_summary.build_validation_path_values()
    assert values.columns.tolist() == ["config", "validation", "sharpe"]
    assert "CPCV claim window" not in set(values["validation"])
    assert len(values.loc[values["validation"].eq("CPCV complete paths")]) == 44
    assert len(values.loc[values["validation"].eq("MC resampled histories")]) == 12
    assert len(values.loc[values["validation"].eq("MC refit stability")]) == 8

    partial_claim = cpcv_rows[:-1]
    pd.DataFrame(partial_claim).to_csv(robust_dir / "breadth_cv_claim_cpcv_path_metrics.csv", index=False)
    partial_values = build_final_results_summary.build_validation_path_values()
    assert "CPCV claim window" not in set(partial_values["validation"])

    pd.DataFrame(cpcv_rows).to_csv(robust_dir / "breadth_cv_claim_cpcv_path_metrics.csv", index=False)
    full_values = build_final_results_summary.build_validation_path_values()
    assert len(full_values.loc[full_values["validation"].eq("CPCV claim window")]) == 44


def test_compact_pbo_table_schema_matches_artifact_contract():
    frame = pd.DataFrame(
        {
            "scope": ["all_configs"],
            "config": ["all"],
            "Basis": ["full_cost_net"],
            "N splits": [78],
            "N strategies": [16],
            "PBO": [0.15],
            "Median lambda": [0.6],
            "Rank correlation IS OOS": [0.7],
            "extra": ["ignored"],
        }
    )
    table = _compact_pbo_table(frame)
    assert table.columns.tolist() == [
        "scope",
        "config",
        "Basis",
        "N splits",
        "N strategies",
        "PBO",
        "Median lambda",
        "Rank correlation IS OOS",
    ]
    assert table["N splits"].iloc[0] == 78


def test_compact_simulation_table_filters_to_e1_rows():
    frame = pd.DataFrame(
        {
            "Return basis": ["Full-cost net", "Full-cost net"],
            "Strategy": ["larger+VIX E1 capped", "larger+VIX GM paper"],
            "Requested method": ["circular_block_bootstrap", "circular_block_bootstrap"],
            "Simulation": ["circular_block_bootstrap", "circular_block_bootstrap"],
            "N paths": [1000, 1000],
            "Defaulted path share": [0.0, 1.0],
            "Ann. return p05": [0.1, -1.0],
            "Ann. return p50": [1.0, -1.0],
            "Sortino p50": [3.0, -1.0],
            "Max DD p50": [-0.4, -1.0],
            "Terminal wealth p50": [10.0, 0.0],
        }
    )
    table = _compact_simulation_table(frame)
    assert table["Strategy"].tolist() == ["larger+VIX E1 capped"]
    assert table["N paths"].iloc[0] == 1000


def test_final_scoreboard_compares_stock_and_naive_baselines(tmp_path, monkeypatch):
    table_dir = tmp_path / "tables"
    robust_dir = tmp_path / "robustness"
    figure_dir = tmp_path / "figures"
    table_dir.mkdir()
    robust_dir.mkdir()
    figure_dir.mkdir()

    (table_dir / "empirical_summary.json").write_text(
        '{"performance": [{"Strategy": "Underlying Markowitz", "Sharpe": 0.9}]}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "config": ["orig", "orig+VIX", "larger", "larger+VIX"],
            "strategy": ["E1 capped"] * 4,
            "deployable": [False, True, True, True],
            "verdict": ["diagnostic_capacity_infeasible", "pass", "mixed", "pass"],
            "net_sharpe": [0.72, 1.28, 0.55, 1.50],
            "net_sortino": [1.6, 3.2, 1.2, 4.0],
            "sum_of_caps": [0.7, 1.1, 1.2, 1.5],
            "cpcv_net_p05": [-0.2, -0.1, -0.4, -0.5],
            "cpcv_net_p50": [-0.1, -0.05, -0.3, -0.4],
            "cpcv_claim_net_p05": [0.1, 0.2, -0.1, 0.3],
            "cpcv_claim_net_p50": [0.4, 0.5, 0.1, 0.7],
            "cpcv_claim_net_p95": [0.8, 0.9, 0.4, 1.0],
            "rel_cpcv_net_p05": [-0.05, 0.1, -0.2, 0.2],
            "rel_cpcv_net_p50": [0.1, 0.3, 0.0, 0.4],
            "mc_resampled_net_p50": [0.7, 1.2, 0.5, 1.4],
            "rolling_net_sharpe": [0.58, 1.0, 0.46, 1.2],
            "mc_resampled_net_p05": [0.1, 0.7, -0.03, 0.98],
            "mc_refit_net_p05": [0.5, 1.1, 0.49, 1.26],
            "mc_refit_net_p50": [0.8, 1.3, 0.7, 1.5],
            "repriced_net_overlay_p05": [-3.0, -2.0, -4.0, -5.0],
            "repriced_net_overlay_p50": [0.3, 0.2, -1.0, -1.5],
        }
    ).to_csv(robust_dir / "breadth_validation_summary.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["within_config"] * 4,
            "config": ["orig", "orig+VIX", "larger", "larger+VIX"],
            "Basis": ["full_cost_net"] * 4,
            "PBO": [0.18, 0.20, 0.10, 0.27],
        }
    ).to_csv(robust_dir / "breadth_cv_pbo_summary.csv", index=False)
    pd.DataFrame(
        {
            "config": ["orig", "orig+VIX", "orig+VIX", "larger", "larger", "larger+VIX", "larger+VIX"],
            "relative_spread_source": [
                "panel_cbbo",
                "panel_cbbo",
                "inferred_cbbo_proxy",
                "panel_cbbo",
                "inferred_cbbo_proxy",
                "panel_cbbo",
                "inferred_cbbo_proxy",
            ],
            "asset_class": ["equity_option", "equity_option", "vix_option", "equity_option", "equity_option", "equity_option", "vix_option"],
            "rows": [10, 10, 2, 10, 30, 10, 32],
            "median_relative_spread": [0.02, 0.02, 0.03, 0.02, 0.025, 0.02, 0.03],
        }
    ).to_csv(robust_dir / "breadth_spread_source_coverage.csv", index=False)

    realized_rows = []
    for config, e1_gross, e1_gross_sortino, equal_premium, equal_risk in [
        ("orig", 0.92, 2.1, 0.10, 0.35),
        ("orig+VIX", 1.44, 3.7, -0.85, -0.97),
        ("larger", 0.80, 1.8, 0.47, 0.55),
        ("larger+VIX", 1.70, 4.5, 0.07, 0.26),
    ]:
        realized_rows.extend(
            [
                {
                    "config": config,
                    "strategy": "E1 capped",
                    "net_sharpe": np.nan,
                    "gross_sharpe": e1_gross,
                    "gross_sortino": e1_gross_sortino,
                },
                {"config": config, "strategy": "Equal premium capped", "net_sharpe": equal_premium},
                {"config": config, "strategy": "Equal risk capped", "net_sharpe": equal_risk},
            ]
        )
    pd.DataFrame(realized_rows).to_csv(robust_dir / "breadth_realized_candidate_summary.csv", index=False)
    dates = pd.to_datetime(["2021-01-29", "2021-02-26", "2021-03-31"])
    rolling_rows = []
    naive_by_config = {
        "orig": "Equal risk capped",
        "orig+VIX": "Equal premium capped",
        "larger": "Equal risk capped",
        "larger+VIX": "Equal risk capped",
    }
    for config in ["orig", "orig+VIX", "larger", "larger+VIX"]:
        for date, ret in zip(dates, [0.03, -0.01, 0.02]):
            rolling_rows.append(
                {
                    "config": config,
                    "return_date": date,
                    "strategy": f"{config} E1 capped",
                    "display_strategy": "E1 capped",
                    "status": "ok",
                    "solver_status": "optimal",
                    "capacity_infeasible": False,
                    "deployed_gross": 1.0,
                    "gross_ret": ret + 0.01,
                    "net_ret": ret,
                }
            )
        for date, ret in zip(dates, [0.01, -0.02, 0.01]):
            display = naive_by_config[config]
            rolling_rows.append(
                {
                    "config": config,
                    "return_date": date,
                    "strategy": f"{config} {display}",
                    "display_strategy": display,
                    "status": "ok",
                    "solver_status": "reference",
                    "capacity_infeasible": False,
                    "deployed_gross": 1.0,
                    "gross_ret": ret + 0.01,
                    "net_ret": ret,
                }
            )
    pd.DataFrame(rolling_rows).to_csv(robust_dir / "breadth_rolling_oos.csv", index=False)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    pd.DataFrame(
        {
            "snap_date": dates,
            "Underlying Markowitz": [0.02, 0.01, -0.005],
        }
    ).to_csv(artifacts_dir / "strategy_returns_post_cost.csv", index=False)

    monkeypatch.setattr(build_final_results_summary, "TABLE_DIR", table_dir)
    monkeypatch.setattr(build_final_results_summary, "ROBUSTNESS_DIR", robust_dir)
    monkeypatch.setattr(build_final_results_summary, "SUMMARY_DIR", robust_dir)
    monkeypatch.setattr(build_final_results_summary, "FIG_DIR", figure_dir)
    monkeypatch.setattr(build_final_results_summary, "PAPER", tmp_path)

    scoreboard = build_final_results_summary.build_baseline_scoreboard()
    by_config = scoreboard.set_index("config")
    assert math.isclose(float(by_config.loc["orig+VIX", "e1_gross_sharpe"]), 1.44)
    assert math.isclose(float(by_config.loc["larger+VIX", "e1_gross_sortino"]), 4.5)
    assert bool(by_config.loc["larger+VIX", "beats_best_naive"])
    assert bool(by_config.loc["larger+VIX", "beats_stock_markowitz"])
    assert bool(by_config.loc["orig+VIX", "beats_stock_markowitz"])
    assert not bool(by_config.loc["larger", "beats_stock_markowitz"])
    assert by_config.loc["larger", "edge_vs_best_naive"] == 0.0

    robustness = build_final_results_summary.build_short_robustness_matrix(scoreboard)
    checks = set(robustness["check"])
    assert "CPCV net full" in checks
    assert "CPCV net claim" in checks
    spread = build_final_results_summary.build_short_spread_summary()
    build_final_results_summary.write_short_tables(scoreboard, robustness, spread)
    build_final_results_summary.plot_short_robustness_heatmap(robustness)
    short_table = (table_dir / "short_robustness_summary.tex").read_text(encoding="utf-8")
    short_scoreboard = (table_dir / "short_final_scoreboard.tex").read_text(encoding="utf-8")
    assert "diagnostic_capacity_infeasible" not in short_table
    assert "CPCV net full" in short_table
    assert "CPCV net claim" in short_table
    assert "Gross Sharpe" in short_scoreboard
    assert "Net Sharpe" in short_scoreboard
    assert "E1 Sharpe" not in short_scoreboard
    assert (table_dir / "short_final_scoreboard.tex").exists()
    assert (figure_dir / "short_robustness_heatmap.pdf").exists()

    validation_without_claim = pd.read_csv(robust_dir / "breadth_validation_summary.csv").drop(
        columns=["cpcv_claim_net_p05", "cpcv_claim_net_p50", "cpcv_claim_net_p95", "rel_cpcv_net_p05", "rel_cpcv_net_p50"]
    )
    validation_without_claim.to_csv(robust_dir / "breadth_validation_summary.csv", index=False)
    robustness_without_claim = build_final_results_summary.build_short_robustness_matrix(scoreboard)
    checks_without_claim = set(robustness_without_claim["check"])
    assert "CPCV net full" in checks_without_claim
    assert "CPCV net claim" not in checks_without_claim
    build_final_results_summary.write_short_tables(scoreboard, robustness_without_claim, spread)
    short_table_without_claim = (table_dir / "short_robustness_summary.tex").read_text(encoding="utf-8")
    assert "CPCV net claim" not in short_table_without_claim

    path_rows = []
    for config_idx, config in enumerate(build_final_results_summary.CONFIG_ORDER):
        for path in range(11):
            path_rows.append(
                {
                    "config": config,
                    "validation": "CPCV complete paths",
                    "sharpe": -0.35 + config_idx * 0.15 + path / 40.0,
                }
            )
        for path in range(200):
            path_rows.append(
                {
                    "config": config,
                    "validation": "MC resampled histories",
                    "sharpe": -0.20 + config_idx * 0.25 + (path - 100) / 180.0,
                }
            )
    build_final_results_summary.plot_validation_distributions(pd.DataFrame(path_rows), scoreboard)
    assert (figure_dir / "short_validation_distributions.pdf").exists()

    paths = build_final_results_summary.build_walk_forward_return_paths(scoreboard)
    build_final_results_summary.plot_walk_forward_return_paths(paths)
    assert (robust_dir / "final_walk_forward_return_paths.csv").exists()
    assert (figure_dir / "short_walk_forward_return_paths.pdf").exists()
    assert {"Locked E1", "Matched capped naive", "Stock baseline"}.issubset(set(paths["family"]))
