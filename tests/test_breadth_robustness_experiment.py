from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.breadth_robustness_experiment import (
    DEFAULT_CV_CONFIG,
    E1_KNOBS,
    _compact_pbo_table,
    _compact_simulation_table,
    breadth_cost_config,
    expected_cv_split_count,
    spread_policy_status,
    _summary_lookup,
)
from research.papers.option_only_markowitz.analysis import build_final_results_summary


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
            "rolling_net_sharpe": [0.58, 1.0, 0.46, 1.2],
            "mc_resampled_net_p05": [0.1, 0.7, -0.03, 0.98],
            "mc_refit_net_p05": [0.5, 1.1, 0.49, 1.26],
        }
    ).to_csv(robust_dir / "breadth_validation_summary.csv", index=False)

    realized_rows = []
    for config, equal_premium, equal_risk in [
        ("orig", 0.10, 0.35),
        ("orig+VIX", -0.85, -0.97),
        ("larger", 0.47, 0.55),
        ("larger+VIX", 0.07, 0.26),
    ]:
        realized_rows.extend(
            [
                {"config": config, "strategy": "Equal premium capped", "net_sharpe": equal_premium},
                {"config": config, "strategy": "Equal risk capped", "net_sharpe": equal_risk},
            ]
        )
    pd.DataFrame(realized_rows).to_csv(robust_dir / "breadth_realized_candidate_summary.csv", index=False)

    monkeypatch.setattr(build_final_results_summary, "TABLE_DIR", table_dir)
    monkeypatch.setattr(build_final_results_summary, "ROBUSTNESS_DIR", robust_dir)
    monkeypatch.setattr(build_final_results_summary, "SUMMARY_DIR", robust_dir)
    monkeypatch.setattr(build_final_results_summary, "FIG_DIR", figure_dir)

    scoreboard = build_final_results_summary.build_baseline_scoreboard()
    by_config = scoreboard.set_index("config")
    assert bool(by_config.loc["larger+VIX", "beats_best_naive"])
    assert bool(by_config.loc["larger+VIX", "beats_stock_markowitz"])
    assert bool(by_config.loc["orig+VIX", "beats_stock_markowitz"])
    assert not bool(by_config.loc["larger", "beats_stock_markowitz"])
    assert by_config.loc["larger", "edge_vs_best_naive"] == 0.0
