"""Tests for option-only Markowitz joint universe resampling."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.resampled_universes import (  # noqa: E402
    ResampleConfig,
    fixed_weight_universe_distribution,
    month_index_paths,
    refit_universe_distribution,
    resampled_summary,
    stratified_month_index_paths,
)
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics  # noqa: E402


def _month_dates(n: int, start: str = "2020-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


class FakeModel:
    def __init__(self, contracts):
        self.contracts = list(contracts)

    def portfolio_return_series(self, option_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        frame = option_returns.reindex(columns=self.contracts).fillna(0.0)
        w = weights.reindex(self.contracts).fillna(0.0).to_numpy(float)
        return pd.Series(frame.to_numpy(float) @ w, index=frame.index)


class TestOptionOnlyResampledUniverses(unittest.TestCase):
    def test_module_ast_parses(self):
        module = ROOT / "research/papers/option_only_markowitz/analysis/resampled_universes.py"
        ast.parse(module.read_text(encoding="utf-8"))

    def test_row_integrity_same_path_moves_whole_month_vectors(self):
        dates = _month_dates(8)
        panel = pd.DataFrame({"A": np.arange(8, dtype=float)}, index=dates)
        panel["B"] = 2.0 * panel["A"]
        strategies = pd.DataFrame({"S1": np.linspace(-0.03, 0.04, 8)}, index=dates)
        strategies["S2"] = -strategies["S1"]

        paths = month_index_paths(8, 5, 3, np.random.default_rng(42))
        for path in paths:
            panel_sample = panel.iloc[path]
            strategy_sample = strategies.iloc[path]
            self.assertTrue(np.allclose(panel_sample["B"], 2.0 * panel_sample["A"]))
            self.assertTrue(np.allclose(strategy_sample["S2"], -strategy_sample["S1"]))

    def test_month_index_paths_range_determinism_and_block_runs(self):
        n_months = 10
        block = 3
        p1 = month_index_paths(n_months, 4, block, np.random.default_rng(7))
        p2 = month_index_paths(n_months, 4, block, np.random.default_rng(7))

        self.assertTrue(np.array_equal(p1, p2))
        self.assertEqual(p1.shape, (4, n_months))
        self.assertTrue(((0 <= p1) & (p1 < n_months)).all())
        for path in p1:
            for start in range(0, n_months, block):
                chunk = path[start : start + block]
                if len(chunk) > 1:
                    self.assertTrue(np.array_equal((np.diff(chunk) % n_months), np.ones(len(chunk) - 1, dtype=int)))

    def test_stratified_month_index_paths_preserve_counts_and_slot_strata(self):
        labels = pd.Series(["Low VIX", "High VIX", "Low VIX", "Mid VIX", "High VIX", "Mid VIX", "Low VIX"])
        paths = stratified_month_index_paths(labels, 10, 2, np.random.default_rng(11))
        historical = labels.value_counts().to_dict()

        self.assertEqual(paths.shape, (10, len(labels)))
        for path in paths:
            source_labels = labels.iloc[path].reset_index(drop=True)
            self.assertEqual(source_labels.value_counts().to_dict(), historical)
            self.assertTrue(source_labels.eq(labels.reset_index(drop=True)).all())

    def test_identity_path_reproduces_realized_performance_metrics(self):
        returns = pd.DataFrame(
            {
                "S1": [0.01, -0.02, 0.03, 0.00, 0.02],
                "S2": [-0.01, 0.01, 0.02, -0.03, 0.04],
            },
            index=_month_dates(5),
        )
        dist = fixed_weight_universe_distribution(
            returns,
            np.arange(len(returns), dtype=int)[None, :],
            basis="gross",
            universe_family="synthetic",
            periods_per_year=12,
        )

        for strategy in returns.columns:
            row = dist[dist["strategy"].eq(strategy)].iloc[0]
            stats = performance_metrics(returns[strategy], 12)
            self.assertAlmostEqual(row["sharpe"], stats["sharpe"])
            self.assertAlmostEqual(row["sortino"], stats["sortino"])
            self.assertAlmostEqual(row["max_drawdown"], stats["max_drawdown"])
            self.assertAlmostEqual(row["ann_return"], stats["annualized_return"])
            self.assertAlmostEqual(row["terminal_wealth"], stats["terminal_wealth"])
            self.assertEqual(row["defaulted"], stats["defaulted"])

    def test_refit_variant_slot_relabels_joint_pseudo_frames(self):
        train_dates = _month_dates(6)
        test_dates = _month_dates(3, "2020-07-31")
        returns = pd.DataFrame(
            {
                "c0": np.arange(6, dtype=float) / 100.0,
                "c1": np.arange(10, 16, dtype=float) / 100.0,
                "marker": np.arange(6, dtype=float),
            },
            index=train_dates,
        )
        under_ret = pd.DataFrame({"u0": np.arange(20, 26, dtype=float) / 100.0, "marker": np.arange(6, dtype=float)}, index=train_dates)
        vol_shocks = pd.DataFrame({"u0": np.arange(30, 36, dtype=float) / 100.0, "marker": np.arange(6, dtype=float)}, index=train_dates)
        reps = pd.DataFrame(
            [{"snap_date": dt, "asset_id": col, "underlying": "u0"} for dt in train_dates for col in returns.columns]
        )
        test_returns = pd.DataFrame({"c0": [0.01, 0.02, -0.01], "c1": [0.00, 0.03, 0.01]}, index=test_dates)
        calls = []

        def spec_builder(reps_train, original_returns, train_start=None, train_end=None):
            self.assertTrue(pd.DatetimeIndex(original_returns.index).equals(train_dates))
            self.assertTrue(np.array_equal(original_returns["marker"].to_numpy(float), np.arange(6, dtype=float)))
            self.assertEqual(set(pd.to_datetime(reps_train["snap_date"])), set(train_dates))
            return pd.DataFrame(index=["c0", "c1"])

        def model_factory(spec, pseudo_returns, reps_train, universe, train_start=None, train_end=None, under_ret=None, vol_shocks=None):
            self.assertTrue(pd.DatetimeIndex(pseudo_returns.index).equals(train_dates))
            self.assertTrue(pd.DatetimeIndex(under_ret.index).equals(train_dates))
            self.assertTrue(pd.DatetimeIndex(vol_shocks.index).equals(train_dates))
            self.assertTrue(np.array_equal(pseudo_returns["marker"].to_numpy(float), under_ret["marker"].to_numpy(float)))
            self.assertTrue(np.array_equal(pseudo_returns["marker"].to_numpy(float), vol_shocks["marker"].to_numpy(float)))
            calls.append(pseudo_returns["marker"].to_numpy(float))
            return FakeModel(spec.index), pd.DataFrame()

        def weights_builder_single(model):
            return pd.Series({"c0": 0.6, "c1": 0.4}, name="Greek Markowitz + VIX")

        out = refit_universe_distribution(
            returns,
            reps,
            ["u0"],
            train_dates,
            test_returns,
            spec_builder=spec_builder,
            model_factory=model_factory,
            weights_builder_single=weights_builder_single,
            under_ret=under_ret,
            vol_shocks=vol_shocks,
            config=ResampleConfig(n_refit_paths=4, block_length=2, refit_seed=123),
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(len(out), 4)
        self.assertTrue(out["status"].eq("ok").all())
        self.assertTrue(np.allclose(out["gross_nav"], 1.0))

    def test_resampled_summary_pinned_quantiles(self):
        paths = pd.DataFrame(
            {
                "strategy": ["S"] * 4,
                "basis": ["gross"] * 4,
                "universe_family": ["all"] * 4,
                "path_id": [0, 1, 2, 3],
                "sharpe": [-1.0, 0.0, 1.0, 2.0],
                "max_drawdown": [-0.6, -0.4, -0.2, -0.8],
                "defaulted": [True, False, False, True],
            }
        )
        realized = pd.DataFrame(
            [{"strategy": "S", "basis": "gross", "universe_family": "all", "sharpe": 0.75}]
        )
        summary = resampled_summary(paths, realized).iloc[0]

        self.assertEqual(summary["Universe Family"], "all")
        self.assertEqual(summary["Basis"], "gross")
        self.assertEqual(summary["Strategy"], "S")
        self.assertAlmostEqual(summary["Realized Value"], 0.75)
        self.assertAlmostEqual(summary["Path P05 Sharpe"], -0.85)
        self.assertAlmostEqual(summary["Path P25 Sharpe"], -0.25)
        self.assertAlmostEqual(summary["Path P50 Sharpe"], 0.5)
        self.assertAlmostEqual(summary["Path P75 Sharpe"], 1.25)
        self.assertAlmostEqual(summary["Path P95 Sharpe"], 1.85)
        self.assertAlmostEqual(summary["Path P50 Max Drawdown"], -0.5)
        self.assertAlmostEqual(summary["P Sharpe Less Than 0"], 0.25)
        self.assertAlmostEqual(summary["P Max Drawdown Less Than -0.5"], 0.5)
        self.assertAlmostEqual(summary["P Default"], 0.5)


if __name__ == "__main__":
    unittest.main()
