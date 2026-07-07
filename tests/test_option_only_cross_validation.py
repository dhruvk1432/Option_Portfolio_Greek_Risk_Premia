"""Tests for option-only Markowitz cross-validation diagnostics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.cross_validation import (
    CVConfig,
    FoldSpec,
    assemble_cpcv_paths,
    build_folds,
    build_group_schedule,
    probability_of_backtest_overfitting,
    refit_fold,
    tag_regimes,
)


def _month_dates(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="MS")


def _fold_rows(folds: list[FoldSpec], strategy: str = "S") -> pd.DataFrame:
    rows = []
    for fold in folds:
        for dt in fold.test_dates:
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "scheme": fold.scheme,
                    "return_date": dt,
                    "strategy": strategy,
                    "basis": "gross",
                    "ret": 0.01,
                }
            )
    return pd.DataFrame(rows)


def _blocks(positions: list[int]) -> list[tuple[int, int]]:
    positions = sorted(positions)
    out = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            out.append((start, prev))
            start = prev = pos
    out.append((start, prev))
    return out


class FakeModel:
    def __init__(self, contracts):
        self.contracts = list(contracts)

    def portfolio_return_series(self, option_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        frame = option_returns.reindex(columns=self.contracts).fillna(0.0)
        w = weights.reindex(self.contracts).fillna(0.0).to_numpy(float)
        return pd.Series(frame.to_numpy(float) @ w, index=frame.index)


def fake_spec_builder(reps, returns, train_start=None, train_end=None):
    return pd.DataFrame({"asset_id": returns.columns}, index=returns.columns)


def fake_reps(dates: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"snap_date": dt, "asset_id": col}
            for dt in dates
            for col in columns
        ]
    )


class TestOptionOnlyCrossValidation(unittest.TestCase):
    def test_build_group_schedule_contiguous_balanced_coverage(self):
        dates = _month_dates(25)
        schedule = build_group_schedule(dates[::-1], 6)
        self.assertEqual(set(schedule.index), set(dates))
        sizes = schedule.value_counts().sort_index()
        self.assertLessEqual(int(sizes.max() - sizes.min()), 1)
        for group in range(6):
            pos = np.flatnonzero(schedule.to_numpy() == group)
            self.assertTrue(np.array_equal(pos, np.arange(pos.min(), pos.max() + 1)))

    def test_fold_counts(self):
        dates = _month_dates(72)
        cfg6 = CVConfig(n_groups=6, n_test_groups=2, min_train_months=0)
        cfg12 = CVConfig(n_groups=12, n_test_groups=2, min_train_months=0)
        self.assertEqual(len(build_folds(dates, cfg6, "kfold")), 6)
        self.assertEqual(len(build_folds(dates, cfg6, "cpcv")), math.comb(6, 2))
        self.assertEqual(len(build_folds(dates, cfg12, "cpcv")), math.comb(12, 2))

    def test_build_folds_claim_window_keeps_full_grid_train_candidates(self):
        dates = _month_dates(120)
        test_window = dates[-58:]
        cfg = CVConfig(
            n_groups=12,
            n_test_groups=2,
            purge_months=1,
            embargo_months=1,
            min_train_months=0,
        )

        kfold = build_folds(dates, cfg, "kfold", test_window=test_window)
        cpcv = build_folds(dates, cfg, "cpcv", test_window=test_window)
        self.assertEqual(len(kfold), 12)
        self.assertEqual(len(cpcv), math.comb(12, 2))
        folds = kfold + cpcv
        window_set = set(test_window)
        full_pos = {pd.Timestamp(dt): i for i, dt in enumerate(dates)}

        self.assertTrue(all(set(fold.test_dates).issubset(window_set) for fold in folds))
        self.assertTrue(
            any(any(pd.Timestamp(dt) < test_window[0] for dt in fold.train_dates) for fold in folds)
        )
        for fold in folds:
            test_pos = [full_pos[dt] for dt in fold.test_dates]
            train_set = {full_pos[dt] for dt in fold.train_dates}
            for start, end in _blocks(test_pos):
                purged = range(
                    max(0, start - cfg.purge_months),
                    min(len(dates) - 1, end + cfg.purge_months) + 1,
                )
                embargoed = range(
                    end + cfg.purge_months + 1,
                    min(len(dates) - 1, end + cfg.purge_months + cfg.embargo_months) + 1,
                )
                excluded = set(purged).union(embargoed).difference(test_pos)
                self.assertTrue(train_set.isdisjoint(excluded), fold.fold_id)

        self.assertEqual(
            build_folds(dates, cfg, "cpcv"),
            build_folds(dates, cfg, "cpcv", test_window=None),
        )
        self.assertEqual(
            build_folds(dates, cfg, "kfold"),
            build_folds(dates, cfg, "kfold", test_window=None),
        )

    def test_purge_embargo_invariant(self):
        dates = _month_dates(48)
        cfg = CVConfig(
            n_groups=6,
            n_test_groups=2,
            purge_months=1,
            embargo_months=2,
            min_train_months=0,
        )
        folds = build_folds(dates, cfg, "cpcv")
        pos = {pd.Timestamp(dt): i for i, dt in enumerate(dates)}
        for fold in folds:
            test_pos = [pos[dt] for dt in fold.test_dates]
            train_pos = [pos[dt] for dt in fold.train_dates]
            for tr in train_pos:
                for te in test_pos:
                    self.assertGreater(abs(tr - te), cfg.purge_months)
            train_set = set(train_pos)
            for start, end in _blocks(test_pos):
                embargo = range(
                    end + cfg.purge_months + 1,
                    min(len(dates) - 1, end + cfg.purge_months + cfg.embargo_months) + 1,
                )
                self.assertTrue(train_set.isdisjoint(embargo))

    def test_cpcv_combinatorics_and_complete_paths(self):
        dates = _month_dates(36)
        cfg = CVConfig(n_groups=6, n_test_groups=2, purge_months=0, embargo_months=0, min_train_months=0)
        folds = build_folds(dates, cfg, "cpcv")
        expected = math.comb(cfg.n_groups - 1, cfg.n_test_groups - 1)
        counts = {group: 0 for group in range(cfg.n_groups)}
        for fold in folds:
            for group in fold.test_groups:
                counts[group] += 1
        self.assertTrue(all(count == expected for count in counts.values()))

        path_returns, path_metrics = assemble_cpcv_paths(_fold_rows(folds), cfg)
        complete = path_metrics[path_metrics["status"].eq("complete")]
        self.assertEqual(complete["path_id"].nunique(), expected)
        self.assertEqual(path_returns["path_id"].nunique(), expected)
        for path_id, grp in path_returns.groupby("path_id"):
            one = grp[grp["strategy"].eq("S") & grp["basis"].eq("gross")]
            self.assertEqual(len(one), len(dates), path_id)
            self.assertEqual(set(one["return_date"]), set(dates))

    def test_probability_of_backtest_overfitting_pins(self):
        rows_bad = []
        rows_good = []
        for split in range(5):
            for strategy, is_sharpe, bad_oos, good_oos in [
                ("A", 3.0, 1.0, 3.0),
                ("B", 2.0, 2.0, 2.0),
                ("C", 1.0, 3.0, 1.0),
            ]:
                rows_bad.append(
                    {
                        "fold_id": f"s{split}",
                        "strategy": strategy,
                        "basis": "gross",
                        "is_sharpe": is_sharpe,
                        "oos_sharpe": bad_oos,
                    }
                )
                rows_good.append(
                    {
                        "fold_id": f"s{split}",
                        "strategy": strategy,
                        "basis": "gross",
                        "is_sharpe": is_sharpe,
                        "oos_sharpe": good_oos,
                    }
                )
        pbo_bad = probability_of_backtest_overfitting(pd.DataFrame(rows_bad))["PBO"].iloc[0]
        pbo_good = probability_of_backtest_overfitting(pd.DataFrame(rows_good))["PBO"].iloc[0]
        self.assertEqual(pbo_bad, 1.0)
        self.assertEqual(pbo_good, 0.0)
        self.assertGreaterEqual(pbo_bad, 0.0)
        self.assertLessEqual(pbo_bad, 1.0)
        self.assertGreaterEqual(pbo_good, 0.0)
        self.assertLessEqual(pbo_good, 1.0)

    def test_refit_fold_guards_zero_weights_and_noncontiguous_train_leak(self):
        dates = _month_dates(8)
        few_cols = ["c0", "c1"]
        few_returns = pd.DataFrame(0.01, index=dates, columns=few_cols)
        fold_few = FoldSpec(
            fold_id="kfold_00",
            scheme="kfold",
            test_groups=(0,),
            test_dates=(dates[-1],),
            train_dates=tuple(dates[:4]),
            purged_dates=(),
            embargoed_dates=(),
        )
        cfg = CVConfig(min_train_months=4, min_contracts=3, min_obs_per_contract=1)
        skipped = refit_fold(
            fold_few,
            few_returns,
            fake_reps(dates, few_cols),
            few_returns,
            fake_reps(dates, few_cols),
            ["U"],
            few_cols,
            spec_builder=fake_spec_builder,
            model_factory=lambda spec, returns, reps, universe, train_start=None, train_end=None: (FakeModel(spec.index), None),
            weights_builder=lambda model, universe=None: {"Greek Markowitz": pd.Series(0.0, index=model.contracts)},
            config=cfg,
        )
        self.assertEqual(skipped["status"], "skipped_too_few_contracts")

        cols = ["c0", "c1", "c2", "c3"]
        returns = pd.DataFrame(
            np.arange(len(dates) * len(cols), dtype=float).reshape(len(dates), len(cols)) / 1000.0,
            index=dates,
            columns=cols,
        )
        train_dates = (dates[0], dates[2], dates[5], dates[7])
        fold = FoldSpec(
            fold_id="cpcv_00_02",
            scheme="cpcv",
            test_groups=(0, 2),
            test_dates=(dates[1], dates[3]),
            train_dates=train_dates,
            purged_dates=(),
            embargoed_dates=(),
        )
        received_indexes = []

        def model_factory(spec, returns, reps, universe, train_start=None, train_end=None):
            received_indexes.append(tuple(returns.index))
            return FakeModel(spec.index), None

        def zero_weights(model, universe=None):
            zero = pd.Series(0.0, index=model.contracts)
            return {
                "Greek Markowitz": zero,
                "Delta neutral": zero,
                "Equal premium": zero,
                "Equal risk": zero,
            }

        fitted = refit_fold(
            fold,
            returns,
            fake_reps(dates, cols),
            returns,
            fake_reps(dates, cols),
            ["U"],
            cols,
            spec_builder=fake_spec_builder,
            model_factory=model_factory,
            weights_builder=zero_weights,
            config=cfg,
        )
        self.assertEqual(fitted["status"], "ok")
        self.assertTrue(received_indexes)
        self.assertTrue(all(idx == train_dates for idx in received_indexes))
        self.assertTrue(fitted["strategy_status"])
        self.assertTrue(all(status == "infeasible_zero_weights" for status in fitted["strategy_status"].values()))

    def test_tag_regimes_events_and_terciles(self):
        dates = _month_dates(24, "2019-01-01")
        vix = pd.Series(np.arange(1, len(dates) + 1, dtype=float), index=dates)
        tags = tag_regimes(dates, vix)
        covid = tags[tags["return_date"].isin(pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"]))]
        self.assertTrue(covid["event"].eq("COVID crash").all())
        counts = tags["vix_tercile"].value_counts()
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_assemble_cpcv_paths_skipped_fold_flags_incomplete_and_excludes_returns(self):
        dates = _month_dates(24)
        cfg = CVConfig(n_groups=4, n_test_groups=2, purge_months=0, embargo_months=0, min_train_months=0)
        folds = build_folds(dates, cfg, "cpcv")
        rows = _fold_rows(folds[1:])
        path_returns, path_metrics = assemble_cpcv_paths(rows, cfg)
        incomplete = path_metrics[path_metrics["status"].eq("incomplete")]
        self.assertFalse(incomplete.empty)
        self.assertTrue(incomplete["sharpe"].isna().all())
        incomplete_paths = set(incomplete["path_id"])
        self.assertTrue(incomplete_paths.isdisjoint(set(path_returns["path_id"])))
        complete = path_metrics[path_metrics["status"].eq("complete")]
        self.assertFalse(complete.empty)
        for path_id in complete["path_id"].unique():
            one = path_returns[path_returns["path_id"].eq(path_id)]
            self.assertEqual(set(one["return_date"]), set(dates))

    def test_assemble_cpcv_paths_absorbs_defaulted_wealth(self):
        dates = _month_dates(4)
        cfg = CVConfig(n_groups=2, n_test_groups=1, purge_months=0, embargo_months=0, min_train_months=0)
        folds = build_folds(dates, cfg, "cpcv")
        rows = []
        returns_by_date = {dates[0]: -1.50, dates[1]: 0.10, dates[2]: 0.10, dates[3]: 0.10}
        for fold in folds:
            for dt in fold.test_dates:
                rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "scheme": fold.scheme,
                        "return_date": dt,
                        "strategy": "S",
                        "basis": "gross",
                        "ret": returns_by_date[dt],
                    }
                )

        _path_returns, path_metrics = assemble_cpcv_paths(pd.DataFrame(rows), cfg)
        metric = path_metrics.iloc[0]

        self.assertEqual(metric["status"], "complete")
        self.assertTrue(bool(metric["defaulted"]))
        self.assertEqual(metric["terminal_wealth"], 0.0)
        self.assertEqual(metric["max_drawdown"], -1.0)
        self.assertEqual(metric["n_months_le_neg100"], 1)


if __name__ == "__main__":
    unittest.main()
