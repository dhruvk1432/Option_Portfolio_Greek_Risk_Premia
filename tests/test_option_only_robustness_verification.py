"""Synthetic artifact tests for distributional-robustness verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.cross_validation import CVConfig, build_folds  # noqa: E402
from research.papers.option_only_markowitz.verification import verify  # noqa: E402


def test_distributional_robustness_synthetic_good_set_passes(tmp_path):
    paper = _write_synthetic_package(tmp_path)
    v = verify.Verifier()

    verify.check_distributional_robustness(v, paper)

    assert v.fail_count(critical_only=True) == 0, [r for r in v.results if r.status != "pass"]
    assert v.fail_count(critical_only=False) == 0, [r for r in v.results if r.status != "pass"]


def test_distributional_robustness_synthetic_violations_fail(tmp_path):
    paper = _write_synthetic_package(tmp_path)
    art = paper / "artifacts"
    summary_path = paper / "tables/distributional_robustness_summary.json"

    schedule = pd.read_csv(art / "cv_fold_schedule.csv")
    schedule.loc[0, "purged_dates"] = ""
    schedule.to_csv(art / "cv_fold_schedule.csv", index=False)

    pbo = pd.read_csv(art / "cv_pbo_summary.csv")
    pbo.loc[0, "PBO"] = 1.2
    pbo.to_csv(art / "cv_pbo_summary.csv", index=False)

    path_returns = pd.read_csv(art / "cv_cpcv_path_month_returns.csv")
    path_returns = pd.concat([path_returns, path_returns.iloc[[0]]], ignore_index=True)
    path_returns.to_csv(art / "cv_cpcv_path_month_returns.csv", index=False)

    context = pd.read_csv(art / "cv_context_consistency.csv")
    context.loc[0, "max_abs_diff"] = 1e-3
    context.to_csv(art / "cv_context_consistency.csv", index=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cv_pbo"][0]["PBO"] = 1.2
    summary["cv_context_consistency"][0]["max_abs_diff"] = 1e-3
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    v = verify.Verifier()
    verify.check_distributional_robustness(v, paper)

    failed = {r.name: r for r in v.results if r.status != "pass"}
    assert "CV purge/embargo invariant recomputed from schedule" in failed
    assert "PBO values bounded" in failed
    assert "complete CPCV paths cover every month exactly once" in failed
    assert "CV context consistency diffs are negligible" in failed
    assert v.fail_count(critical_only=True) >= 4


def _write_synthetic_package(tmp_path: Path) -> Path:
    paper = tmp_path / "paper"
    art = paper / "artifacts"
    tables = paper / "tables"
    art.mkdir(parents=True)
    tables.mkdir(parents=True)

    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    cfg = CVConfig(n_groups=4, n_test_groups=2, purge_months=1, embargo_months=1, min_train_months=0)
    folds = build_folds(dates, cfg, "kfold") + build_folds(dates, cfg, "cpcv")
    schedule_rows = []
    ledger_rows = []
    for fold in folds:
        schedule_rows.append(
            {
                "fold_id": fold.fold_id,
                "scheme": fold.scheme,
                "test_groups": "_".join(str(g) for g in fold.test_groups),
                "test_start": min(fold.test_dates).strftime("%Y-%m-%d"),
                "test_end": max(fold.test_dates).strftime("%Y-%m-%d"),
                "n_train": len(fold.train_dates),
                "n_test": len(fold.test_dates),
                "n_purged": len(fold.purged_dates),
                "n_embargoed": len(fold.embargoed_dates),
                "purged_dates": _date_join(fold.purged_dates),
                "embargoed_dates": _date_join(fold.embargoed_dates),
                "status": "ok",
            }
        )
        ledger_rows.append(
            {
                "fold_id": fold.fold_id,
                "scheme": fold.scheme,
                "strategy": "S",
                "basis": "gross",
                "sharpe": 0.1,
                "status": "ok",
                "n_test_months": len(fold.test_dates),
            }
        )
    pd.DataFrame(schedule_rows).to_csv(art / "cv_fold_schedule.csv", index=False)
    pd.DataFrame(ledger_rows).to_csv(art / "cv_fold_ledger.csv", index=False)
    pd.DataFrame(
        [
            {"fold_id": row["fold_id"], "scheme": row["scheme"], "strategy": "S", "basis": "gross", "is_sharpe": 0.2, "oos_sharpe": 0.1}
            for row in ledger_rows
        ]
    ).to_csv(art / "cv_split_is_oos.csv", index=False)

    path_ids = [f"path_{i:02d}" for i in range(3)]
    path_metrics = pd.DataFrame(
        [
            {
                "path_id": path_id,
                "strategy": "S",
                "basis": "gross",
                "sharpe": 0.1,
                "sortino": 0.1,
                "max_drawdown": -0.1,
                "ann_return": 0.1,
                "terminal_wealth": 1.1,
                "n_months": len(dates),
                "status": "complete",
            }
            for path_id in path_ids
        ]
    )
    path_metrics.to_csv(art / "cv_cpcv_path_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"path_id": path_id, "return_date": dt.strftime("%Y-%m-%d"), "strategy": "S", "basis": "gross", "ret": 0.01}
            for path_id in path_ids
            for dt in dates
        ]
    ).to_csv(art / "cv_cpcv_path_month_returns.csv", index=False)
    pbo = pd.DataFrame([{"Basis": "gross", "N splits": 1, "N strategies": 1, "PBO": 0.5, "Median lambda": 0.0, "Rank correlation IS OOS": 1.0}])
    pbo.to_csv(art / "cv_pbo_summary.csv", index=False)
    pd.DataFrame([{"Regime family": "VIX tercile", "Strategy": "S", "Group": "Low VIX", "Metric": "Sharpe", "Estimate": 0.1}]).to_csv(
        art / "cv_regime_performance.csv", index=False
    )
    pd.DataFrame([{"fold_id": folds[0].fold_id, "seconds": 0.0, "status": "ok"}]).to_csv(art / "cv_runtime_log.csv", index=False)
    context = pd.DataFrame([{"strategy": "S", "status": "ok", "max_abs_diff": 0.0, "n_context_weights": 1, "n_reference_weights": 1}])
    context.to_csv(art / "cv_context_consistency.csv", index=False)

    pd.DataFrame(
        [
            {"universe_family": "resampled", "basis": "gross", "path_id": i, "strategy": "S", "sharpe": 0.1, "defaulted": False}
            for i in range(2)
        ]
    ).to_csv(art / "mc_resampled_fixed_paths.csv", index=False)
    pd.DataFrame([{"Universe Family": "resampled", "Basis": "gross", "Strategy": "S", "Path P50 Sharpe": 0.1}]).to_csv(
        art / "mc_resampled_summary.csv", index=False
    )
    pd.DataFrame([{"path_id": i, "strategy": "Greek Markowitz + VIX", "sharpe": 0.1, "status": "ok"} for i in range(2)]).to_csv(
        art / "mc_refit_paths.csv", index=False
    )
    pd.DataFrame([{"Strategy": "Greek Markowitz + VIX", "Paths": 2, "OK Paths": 2, "P50 Sharpe": 0.1}]).to_csv(
        art / "mc_refit_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {"Assumption": "Fixed Weight Path Count", "Value": 2, "Notes": ""},
            {"Assumption": "Refit Path Count", "Value": 2, "Notes": ""},
        ]
    ).to_csv(art / "mc_resampled_assumptions.csv", index=False)
    pd.DataFrame(
        [
            {"method": "joint_garch_block", "path_id": i, "strategy": "S", "sharpe": 0.1, "defaulted": False, "weight_coverage": 1.0}
            for i in range(2)
        ]
    ).to_csv(art / "mc_repriced_paths.csv", index=False)
    pd.DataFrame([{"Strategy": "S", "Method": "joint_garch_block", "P50 Sharpe": 0.1}]).to_csv(art / "mc_repriced_summary.csv", index=False)
    pd.DataFrame([{"method": "gaussian_copula", "path_id": 0, "strategy": "S", "sharpe": 0.1, "defaulted": False, "weight_coverage": 1.0}]).to_csv(
        art / "mc_repriced_paths_gauss_copula.csv", index=False
    )
    pd.DataFrame([{"Strategy": "S", "Method": "gaussian_copula", "P50 Sharpe": 0.1}]).to_csv(
        art / "mc_repriced_summary_gauss_copula.csv", index=False
    )
    assumptions = pd.DataFrame(
        [
            {
                "Section": "Kernel",
                "Item": "Pricing Tenor Rule",
                "Value": "0.08333333333333333",
                "Notes": "Synthetic contracts are one-step (1-month) options so premium and payoff horizons match.",
            },
            {
                "Section": "Kernel",
                "Item": "VIX Forward Convention",
                "Value": "simulated spot:VX_FRONT entry forward; VIX level settlement proxy at t+1",
                "Notes": "Premium uses the VX-front state, payoff uses simulated VIX level.",
            },
            {"Section": "Kernel", "Item": "Path Count", "Value": 2, "Notes": ""},
        ]
    )
    assumptions.to_csv(art / "mc_repriced_assumptions.csv", index=False)
    pd.DataFrame([{"Strategy": "S", "Sharpe Realized": 0.1, "Resampled P50": 0.1, "Repriced P50": 0.1}]).to_csv(
        art / "mc_universe_comparison.csv", index=False
    )

    for rel in verify.ROBUSTNESS_TABLES:
        (paper / rel).write_text(
            "\\begin{tabular}{ll}\n"
            "\\toprule\n"
            "Strategy & Value \\\\\n"
            "\\midrule\n"
            "S & 0.1 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n",
            encoding="utf-8",
        )

    summary = {
        "cv_config": {
            "n_groups": cfg.n_groups,
            "n_test_groups": cfg.n_test_groups,
            "purge_months": cfg.purge_months,
            "embargo_months": cfg.embargo_months,
        },
        "resample_config": {"n_paths": 2, "n_refit_paths": 2},
        "reprice_config": {"n_paths": 2, "n_sensitivity_paths": 1},
        "cv_fold_schedule": schedule_rows,
        "cv_fold_ledger": ledger_rows,
        "cv_cpcv_path_metrics": path_metrics.to_dict(orient="records"),
        "cv_pbo": pbo.to_dict(orient="records"),
        "cv_regime_performance": [{"Regime family": "VIX tercile", "Strategy": "S"}],
        "cv_context_consistency": context.to_dict(orient="records"),
        "mc_resampled_summary": [{"Universe Family": "resampled", "Strategy": "S"}],
        "mc_refit_summary": [{"Strategy": "Greek Markowitz + VIX", "Paths": 2}],
        "mc_repriced_summary": [{"Strategy": "S", "Method": "joint_garch_block"}],
        "mc_repriced_assumptions": assumptions.to_dict(orient="records"),
        "mc_universe_comparison": [{"Strategy": "S"}],
        "runtime_seconds": {"cv": 0.0, "mc": 0.0},
        "seeds": {"cv": 1, "resample": 2, "refit": 3, "reprice": 4},
    }
    (tables / "distributional_robustness_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return paper


def _date_join(dates) -> str:
    return ";".join(pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates)
