"""Hermetic wiring tests for the option-only robustness stage."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis import run_empirics  # noqa: E402
from research.papers.option_only_markowitz.analysis.cross_validation import CVResults  # noqa: E402


REQUIRED_SUMMARY_KEYS = {
    "cv_config",
    "cv_fold_schedule",
    "cv_fold_ledger",
    "cv_cpcv_path_metrics",
    "cv_pbo",
    "cv_regime_performance",
    "cv_context_consistency",
    "mc_resampled_summary",
    "mc_refit_summary",
    "mc_repriced_summary",
    "mc_repriced_assumptions",
    "mc_universe_comparison",
    "runtime_seconds",
    "seeds",
}


def test_argparse_accepts_robustness_stages_and_knobs():
    parser = run_empirics._build_arg_parser()
    for stage in ["all", "robustness", "cv", "mc", "robustness-figures"]:
        args = parser.parse_args(
            [
                "--stage",
                stage,
                "--cv-groups",
                "4",
                "--cv-test-groups",
                "1",
                "--mc-paths",
                "7",
                "--mc-refit-paths",
                "3",
                "--mc-reprice-paths",
                "5",
            ]
        )
        assert args.stage == stage
        assert args.cv_groups == 4
        assert args.cv_test_groups == 1
        assert args.mc_paths == 7
        assert args.mc_refit_paths == 3
        assert args.mc_reprice_paths == 5


def test_run_all_calls_hash_manifest_helper():
    source = inspect.getsource(run_empirics.run_all)
    assert "_write_hash_manifest()" in source


def test_write_cv_outputs_uses_artifact_contract_and_tex_headers(tmp_path, monkeypatch):
    art_dir = tmp_path / "artifacts"
    table_dir = tmp_path / "tables"
    art_dir.mkdir()
    table_dir.mkdir()
    monkeypatch.setattr(run_empirics, "ART_DIR", art_dir)
    monkeypatch.setattr(run_empirics, "TABLE_DIR", table_dir)

    dates = pd.date_range("2021-01-31", periods=24, freq="ME")
    results = CVResults(
        fold_schedule=pd.DataFrame(
            [
                {
                    "fold_id": "kfold_00",
                    "scheme": "kfold",
                    "test_groups": "0",
                    "test_start": dates[0],
                    "test_end": dates[5],
                    "n_train": 18,
                    "n_test": 6,
                    "n_purged": 0,
                    "n_embargoed": 0,
                    "purged_dates": "",
                    "embargoed_dates": "",
                    "status": "ok",
                }
            ]
        ),
        fold_ledger=pd.DataFrame(
            [
                {
                    "fold_id": "kfold_00",
                    "scheme": "kfold",
                    "strategy": "Greek Markowitz + VIX",
                    "basis": "gross",
                    "sharpe": 1.2,
                    "status": "ok",
                },
                {
                    "fold_id": "kfold_00",
                    "scheme": "kfold",
                    "strategy": "Greek Markowitz + VIX",
                    "basis": "full_spread_post_cost",
                    "sharpe": 0.8,
                    "status": "ok",
                },
            ]
        ),
        split_is_oos=pd.DataFrame(
            [
                {
                    "fold_id": "kfold_00",
                    "scheme": "kfold",
                    "strategy": "Greek Markowitz + VIX",
                    "basis": "gross",
                    "is_sharpe": 1.0,
                    "oos_sharpe": 1.2,
                    "status": "ok",
                }
            ]
        ),
        test_month_returns=pd.DataFrame(),
        runtime_log=pd.DataFrame([{"fold_id": "kfold_00", "seconds": 0.01, "status": "ok"}]),
    )
    path_month_returns = pd.DataFrame(
        [
            {
                "path_id": "path_00",
                "return_date": dt,
                "strategy": "Greek Markowitz + VIX",
                "basis": "gross",
                "ret": 0.01,
            }
            for dt in dates
        ]
    )
    path_metrics = pd.DataFrame(
        [
            {
                "path_id": "path_00",
                "strategy": "Greek Markowitz + VIX",
                "basis": "gross",
                "sharpe": 1.1,
                "status": "complete",
            },
            {
                "path_id": "path_00",
                "strategy": "Greek Markowitz + VIX",
                "basis": "full_spread_post_cost",
                "sharpe": 0.7,
                "status": "complete",
            },
        ]
    )
    pbo_summary = pd.DataFrame(
        [
            {"Basis": "gross", "N splits": 1, "N strategies": 2, "PBO": 0.0},
            {"Basis": "full_spread_post_cost", "N splits": 1, "N strategies": 2, "PBO": 0.5},
        ]
    )
    regime_performance = pd.DataFrame(
        [
            {
                "Regime family": "VIX tercile",
                "Strategy": "Greek Markowitz + VIX",
                "Group": "Low VIX",
                "Metric": "Sharpe",
                "Estimate": 1.0,
                "CI lo": 0.1,
                "CI hi": 1.9,
                "N": 12,
                "Seed": 1,
            }
        ]
    )
    ret_frame = pd.DataFrame({"Greek Markowitz + VIX": [0.01] * 24}, index=dates)
    net_frame = pd.DataFrame({"Greek Markowitz + VIX": [0.005] * 24}, index=dates)

    run_empirics._write_cv_outputs(
        results,
        path_month_returns,
        path_metrics,
        pbo_summary,
        regime_performance,
        ret_frame,
        net_frame,
    )

    for name in [
        "cv_fold_schedule.csv",
        "cv_fold_ledger.csv",
        "cv_split_is_oos.csv",
        "cv_cpcv_path_metrics.csv",
        "cv_cpcv_path_month_returns.csv",
        "cv_pbo_summary.csv",
        "cv_regime_performance.csv",
        "cv_runtime_log.csv",
    ]:
        assert (art_dir / name).exists()

    for name in ["cv_fold_performance.tex", "cv_cpcv_distribution.tex", "cv_regime_performance.tex"]:
        path = table_dir / name
        assert path.exists()
        header = _latex_header(path.read_text(encoding="utf-8"))
        assert "_" not in header


def test_distributional_robustness_summary_writer_valid_json(tmp_path, monkeypatch):
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    monkeypatch.setattr(run_empirics, "TABLE_DIR", table_dir)
    summary = {key: [] for key in REQUIRED_SUMMARY_KEYS}
    summary["runtime_seconds"] = {"context": 0.0, "cv": 0.0, "mc": 0.0}
    summary["seeds"] = {"cv": 1, "resample": 2, "refit": 3, "reprice": 4}

    run_empirics._write_distributional_robustness_summary(summary)

    loaded = json.loads((table_dir / "distributional_robustness_summary.json").read_text(encoding="utf-8"))
    assert REQUIRED_SUMMARY_KEYS.issubset(loaded)


def test_plot_cpcv_sharpe_distribution_smoke(tmp_path):
    strategies = list(run_empirics.CV_STRATEGIES[:2])
    rows = []
    for path_id in ["path_00", "path_01", "path_02"]:
        for i, strategy in enumerate(strategies):
            rows.append(
                {
                    "path_id": path_id,
                    "strategy": strategy,
                    "basis": "gross",
                    "sharpe": 0.6 + 0.1 * i + 0.05 * len(path_id),
                    "status": "complete",
                }
            )
            rows.append(
                {
                    "path_id": path_id,
                    "strategy": strategy,
                    "basis": "full_spread_post_cost",
                    "sharpe": 0.2 + 0.1 * i,
                    "status": "complete",
                }
            )
    realized = pd.DataFrame(
        [
            {"strategy": strategies[0], "basis": "gross", "sharpe": 0.72},
            {"strategy": strategies[1], "basis": "gross", "sharpe": 0.91},
        ]
    )
    out = tmp_path / "cv_cpcv_sharpe_distribution.pdf"

    run_empirics.plot_cpcv_sharpe_distribution(pd.DataFrame(rows), realized, out)

    _assert_pdf_nonempty(out)


def test_plot_fold_sharpe_heatmap_smoke(tmp_path):
    strategies = list(run_empirics.CV_STRATEGIES[:2])
    fold_schedule = pd.DataFrame(
        [
            {
                "fold_id": "kfold_00",
                "scheme": "kfold",
                "test_start": "2015-02-27",
                "test_end": "2015-12-31",
                "status": "ok",
            },
            {
                "fold_id": "kfold_01",
                "scheme": "kfold",
                "test_start": "2016-01-29",
                "test_end": "2016-11-30",
                "status": "ok",
            },
        ]
    )
    fold_ledger = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "scheme": "kfold",
                "strategy": strategy,
                "basis": "gross",
                "sharpe": 0.4 + 0.2 * i - 0.1 * j,
                "status": "ok",
            }
            for j, fold_id in enumerate(["kfold_00", "kfold_01"])
            for i, strategy in enumerate(strategies)
        ]
    )
    out = tmp_path / "cv_fold_sharpe_heatmap.pdf"

    run_empirics.plot_fold_sharpe_heatmap(fold_ledger, fold_schedule, out)

    _assert_pdf_nonempty(out)


def test_plot_mc_universe_distributions_smoke(tmp_path):
    strategies = list(run_empirics.MC_ROBUSTNESS_STRATEGIES)
    realized = {
        "Greek Markowitz + VIX": 1.1,
        "Equity-option Greek Markowitz": 0.8,
        "Beta/delta-neutral + VIX": 1.0,
        "Cost-aware Sortino + VIX": 0.7,
    }
    fixed_rows = []
    for family in ["resampled", "resampled_stratified"]:
        for path_id in range(3):
            for i, strategy in enumerate(strategies):
                fixed_rows.append(
                    {
                        "universe_family": family,
                        "basis": "gross",
                        "path_id": path_id,
                        "strategy": strategy,
                        "sharpe": realized[strategy] + 0.05 * path_id - 0.03 * i,
                        "realized_sharpe": realized[strategy],
                    }
                )
    repriced_rows = [
        {
            "method": "joint_garch_block",
            "path_id": path_id,
            "strategy": strategy,
            "sharpe": realized[strategy] - 0.8 + 0.1 * path_id,
            "realized_sharpe": realized[strategy],
        }
        for path_id in range(3)
        for strategy in strategies
    ]
    refit_paths = pd.DataFrame(
        [
            {
                "path_id": path_id,
                "strategy": "Greek Markowitz + VIX",
                "sharpe": 0.9 + 0.05 * path_id,
                "status": "ok",
                "realized_sharpe": realized["Greek Markowitz + VIX"],
            }
            for path_id in range(5)
        ]
    )
    out = tmp_path / "mc_universe_sharpe_distributions.pdf"

    run_empirics.plot_mc_universe_distributions(
        pd.DataFrame(fixed_rows),
        pd.DataFrame(repriced_rows),
        refit_paths,
        out,
    )

    _assert_pdf_nonempty(out)


def _assert_pdf_nonempty(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def _latex_header(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == r"\toprule":
            for candidate in lines[i + 1 :]:
                if " & " in candidate:
                    return candidate
    raise AssertionError("no LaTeX header row found")
