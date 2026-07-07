"""Hermetic wiring tests for the option-only robustness stage."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

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


def test_sharpe_difference_test_symmetry_identical_and_known_case():
    from research.papers.option_only_markowitz.analysis.inference import (
        BootstrapConfig,
        sharpe_difference_test,
    )

    rng = np.random.default_rng(20260706)
    dates = pd.date_range("2020-01-31", periods=120, freq="ME")
    common = rng.normal(0.0, 0.015, len(dates))
    a = pd.Series(0.020 + common + rng.normal(0.0, 0.006, len(dates)), index=dates)
    b = pd.Series(-0.004 + common + rng.normal(0.0, 0.006, len(dates)), index=dates)
    cfg = BootstrapConfig(n_boot=40, seed=7, block_size=6)

    ab = sharpe_difference_test(a, b, cfg)
    ba = sharpe_difference_test(b, a, cfg)
    assert ab["delta_sharpe"] == pytest.approx(-ba["delta_sharpe"])
    assert ab["jk_z"] == pytest.approx(-ba["jk_z"])

    same = sharpe_difference_test(a, a, cfg)
    assert same["delta_sharpe"] == pytest.approx(0.0)
    assert same["jk_z"] == pytest.approx(0.0)
    assert same["jk_p"] == pytest.approx(1.0)

    assert ab["jk_p"] < 0.05


def test_build_inference_panel_artifacts_and_guard(tmp_path, monkeypatch):
    from research.papers.option_only_markowitz.analysis import build_inference_panel
    from research.papers.option_only_markowitz.analysis.inference import BootstrapConfig

    scoreboard_path = _write_synthetic_inference_artifacts(tmp_path, monkeypatch, build_inference_panel)
    cfg = BootstrapConfig(n_boot=40, seed=11, block_size=5)

    panel = build_inference_panel.main(config=cfg)

    csv_path = build_inference_panel.ROBUSTNESS_DIR / "final_inference_panel.csv"
    tex_path = build_inference_panel.TABLE_DIR / "short_inference_panel.tex"
    assert csv_path.exists()
    assert tex_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 8
    static = loaded.loc[loaded["basis"].eq("static")]
    assert len(static) == 4
    assert (static["net_sharpe_ci_lo"] <= static["net_sharpe"]).all()
    assert (static["net_sharpe"] <= static["net_sharpe_ci_hi"]).all()
    assert set(static["dsr_trials"].astype(int)) == {22}
    assert set(static["dsr_trials_sensitivity"].astype(int)) == {25}
    assert panel.shape == loaded.shape
    assert "_" not in _latex_header(tex_path.read_text(encoding="utf-8"))

    scoreboard = pd.read_csv(scoreboard_path)
    scoreboard.loc[scoreboard["config"].eq("orig"), "e1_net_sharpe"] += 0.01
    scoreboard.to_csv(scoreboard_path, index=False)
    with pytest.raises(RuntimeError, match="Static return order validation failed"):
        build_inference_panel.main(config=cfg)


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


def _write_synthetic_inference_artifacts(tmp_path: Path, monkeypatch, module) -> Path:
    breadth_dir = tmp_path / "breadth_solutions"
    robustness_dir = breadth_dir / "robustness"
    table_dir = tmp_path / "tables"
    artifact_dir = tmp_path / "artifacts"
    for directory in (breadth_dir, robustness_dir, table_dir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "BREADTH_DIR", breadth_dir)
    monkeypatch.setattr(module, "ROBUSTNESS_DIR", robustness_dir)
    monkeypatch.setattr(module, "TABLE_DIR", table_dir)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifact_dir)

    configs = ["orig", "orig+VIX", "larger", "larger+VIX"]
    best_naive = {
        "orig": "Equal premium capped",
        "orig+VIX": "Equal premium capped",
        "larger": "Equal risk capped",
        "larger+VIX": "Equal risk capped",
    }
    dates = pd.date_range("2021-01-31", periods=60, freq="ME")
    t = np.arange(len(dates), dtype=float)
    base = 0.012 + 0.018 * np.sin(t / 3.0) + 0.006 * np.cos(t / 5.0)
    offsets = {"orig": 0.000, "orig+VIX": 0.006, "larger": -0.002, "larger+VIX": 0.004}

    static_cols: dict[str, np.ndarray] = {}
    scoreboard_rows = []
    rolling_rows = []
    stock = pd.Series(0.004 + 0.40 * base, index=dates, name="Underlying Markowitz")
    _append_path_rows(
        rolling_rows,
        dates,
        config="stock",
        config_label="Stock Markowitz",
        family="Stock baseline",
        strategy="Underlying Markowitz",
        values=stock.to_numpy(float),
    )
    for config in configs:
        e1 = base + offsets[config]
        static_cols[f"{config} E1 capped"] = e1
        static_cols[f"{config} GM paper"] = 0.75 * e1 - 0.004
        static_cols[f"{config} Equal premium capped"] = 0.45 * base - 0.003
        static_cols[f"{config} Equal risk capped"] = 0.35 * base - 0.002
        naive_strategy = best_naive[config]
        scoreboard_rows.append(
            {
                "config": config,
                "config_label": config,
                "e1_net_sharpe": _ann_sharpe(e1),
                "best_naive_strategy": naive_strategy,
            }
        )
        _append_path_rows(
            rolling_rows,
            dates,
            config=config,
            config_label=f"{config} E1",
            family="Locked E1",
            strategy=f"{config} E1 capped",
            values=0.80 * e1 + 0.001,
        )
        _append_path_rows(
            rolling_rows,
            dates,
            config=config,
            config_label=f"{config} naive",
            family="Matched capped naive",
            strategy=f"{config} {naive_strategy}",
            values=0.90 * static_cols[f"{config} {naive_strategy}"],
        )

    pd.DataFrame(static_cols).to_csv(robustness_dir / "breadth_strategy_returns_net.csv", index=False)
    scoreboard = pd.DataFrame(scoreboard_rows)
    scoreboard_path = robustness_dir / "final_result_scoreboard.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    pd.DataFrame(rolling_rows).to_csv(robustness_dir / "final_walk_forward_return_paths.csv", index=False)
    pd.DataFrame({"snap_date": dates, "Underlying Markowitz": stock.to_numpy(float)}).to_csv(
        artifact_dir / "strategy_returns_post_cost.csv",
        index=False,
    )

    p1_rows = []
    p3_rows = []
    for cfg_i, config in enumerate(configs):
        for i in range(22):
            p1_rows.append(
                {
                    "config": config,
                    "point_id": f"candidate_{i:02d}",
                    "strategy": "Greek Markowitz",
                    "arm": "E" if i >= 18 else "A",
                    "net_sharpe_noimpact": 0.25 + 0.01 * i + 0.03 * cfg_i,
                }
            )
        p1_rows.extend(
            [
                {
                    "config": config,
                    "point_id": "Equal premium",
                    "strategy": "Equal premium",
                    "arm": "naive",
                    "net_sharpe_noimpact": -0.20,
                },
                {
                    "config": config,
                    "point_id": "Equal risk",
                    "strategy": "Equal risk",
                    "arm": "naive",
                    "net_sharpe_noimpact": -0.10,
                },
            ]
        )
        for j in range(3):
            p3_rows.append(
                {
                    "config": config,
                    "strategy": f"GM combined {j}",
                    "knobs_label": "primary",
                    "mode": "hard",
                    "net_sharpe": 0.50 + 0.05 * j,
                }
            )
        p3_rows.extend(
            [
                {
                    "config": config,
                    "strategy": "Equal premium capped",
                    "knobs_label": "naive_capped",
                    "mode": "hard",
                    "net_sharpe": 0.1,
                },
                {
                    "config": config,
                    "strategy": "Equal risk capped",
                    "knobs_label": "naive_capped",
                    "mode": "hard",
                    "net_sharpe": 0.2,
                },
            ]
        )
    pd.DataFrame(p1_rows).to_csv(breadth_dir / "p1_regularization_results.csv", index=False)
    pd.DataFrame(p3_rows).to_csv(breadth_dir / "p3_combined_results.csv", index=False)
    return scoreboard_path


def _append_path_rows(
    rows: list[dict[str, object]],
    dates: pd.DatetimeIndex,
    *,
    config: str,
    config_label: str,
    family: str,
    strategy: str,
    values: np.ndarray,
) -> None:
    wealth = np.cumprod(1.0 + values)
    for dt, ret, path_wealth in zip(dates, values, wealth):
        rows.append(
            {
                "return_date": dt,
                "config": config,
                "config_label": config_label,
                "family": family,
                "strategy": strategy,
                "return": ret,
                "gross_growth": 1.0 + ret,
                "wealth": path_wealth,
            }
        )


def _ann_sharpe(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    return float(np.sqrt(12.0) * x.mean() / x.std(ddof=1))
