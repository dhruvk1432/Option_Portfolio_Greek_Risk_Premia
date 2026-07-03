
"""Tests for the publication-grade option-only Markowitz upgrades."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.inference import (  # noqa: E402
    BootstrapConfig,
    block_bootstrap_metric_ci,
    hac_ols,
    sharpe_reality_check,
)
from research.papers.option_only_markowitz.analysis.execution_cost_scenarios import (  # noqa: E402
    ExecutionCostScenarioConfig,
    RepairConfig,
    apply_trade_hurdles,
    build_execution_cost_scenarios,
    capacity_market_impact_diagnostics,
    execution_repair_comparison_table,
    forecast_ablation_tables,
    liquidity_tier_labels,
    liquidity_tier_performance,
    post_cost_survival_table,
    repair_diagnostics_table,
)
from src.portfolio.option_only_markowitz_model import performance_stats  # noqa: E402
from src.portfolio.option_only_markowitz_model import (  # noqa: E402
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (  # noqa: E402
    ResearchCostConfig,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    cost_diagnostics_table,
    derive_entry_cost_series,
)
from research.papers.option_only_markowitz.analysis.run_empirics import _sortino_weights_with_guard  # noqa: E402
from research.papers.option_only_markowitz.analysis.simulation import (  # noqa: E402
    SimulationConfig,
    circular_block_path_distribution,
    clean_returns,
    drawdown_breach_rates,
    run_tail_path_simulations,
    volatility_clustered_path_distribution,
)
from research.papers.option_only_markowitz.analysis.vix_option_panel import (  # noqa: E402
    build_vix_expiry_proxy_returns,
    load_vro_series,
)




class TestTailPathSimulationDiagnostics(unittest.TestCase):
    def test_clean_returns_removes_nonfinite_values(self):
        cleaned = clean_returns(pd.Series([0.01, np.nan, np.inf, -np.inf, -0.02]))
        self.assertEqual(cleaned.tolist(), [0.01, -0.02])

    def test_circular_block_bootstrap_preserves_path_length(self):
        r = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02, -0.01], index=pd.date_range("2021-01-31", periods=6, freq="ME"))
        paths = circular_block_path_distribution(r, n_paths=25, block_length=3, seed=7)
        self.assertEqual(len(paths), 25)
        self.assertTrue(paths["status"].eq("ok").all())
        self.assertTrue(paths["n_source_obs"].eq(len(r)).all())

    def test_volatility_clustered_fallback_is_explicit_and_finite(self):
        r = pd.Series(np.sin(np.arange(24)) * 0.02, index=pd.date_range("2021-01-31", periods=24, freq="ME"))
        paths = volatility_clustered_path_distribution(r, n_paths=20, seed=11, min_egarch_obs=120)
        self.assertFalse(paths.empty)
        self.assertTrue(str(paths["method"].iloc[0]).startswith("garch11_residual_fallback"))
        self.assertTrue(np.isfinite(paths["max_drawdown"].astype(float)).all())

    def test_drawdown_breach_rates_match_toy_paths(self):
        paths = pd.DataFrame({"status": ["ok", "ok", "ok"], "max_drawdown": [-0.05, -0.30, -0.80]})
        rates = drawdown_breach_rates(paths, limits=(0.10, 0.50, 0.90))
        self.assertAlmostEqual(rates["Breach 10%"], 2 / 3)
        self.assertAlmostEqual(rates["Breach 50%"], 1 / 3)
        self.assertAlmostEqual(rates["Breach 90%"], 0.0)

    def test_tail_path_simulations_are_seed_reproducible(self):
        idx = pd.date_range("2021-01-31", periods=24, freq="ME")
        frame = pd.DataFrame({"Strategy": np.cos(np.arange(24)) * 0.015}, index=idx)
        cfg = SimulationConfig(block_paths=30, vol_paths=30, block_length=4, seed=123)
        out1 = run_tail_path_simulations({"Gross before costs": frame}, strategies=("Strategy",), config=cfg)
        out2 = run_tail_path_simulations({"Gross before costs": frame}, strategies=("Strategy",), config=cfg)
        pd.testing.assert_frame_equal(out1[0], out2[0])
        pd.testing.assert_frame_equal(out1[1], out2[1])
        pd.testing.assert_frame_equal(out1[2], out2[2])
        self.assertIn("Strategy", out1[0]["Strategy"].unique())

class TestExactVroSettlement(unittest.TestCase):
    def test_vro_loader_uses_configured_file_without_forward_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vro_exact.csv"
            pd.DataFrame(
                {
                    "settlement_date": ["2024-01-17", "2024-02-14"],
                    "settlement_value": [15.25, 18.75],
                }
            ).to_csv(path, index=False)
            with patch.dict(os.environ, {"OPTION_MARKOWITZ_VRO_FILE": str(path)}, clear=False):
                series = load_vro_series(Path(tmp))
            self.assertAlmostEqual(float(series.loc[pd.Timestamp("2024-01-17")]), 15.25)
            self.assertNotIn(pd.Timestamp("2024-01-18"), series.index)

    def test_vix_expiry_rows_are_exact_only_when_expiry_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir(parents=True)
            path = root / "data" / "vro_exact.csv"
            pd.DataFrame({"settlement_date": ["2024-01-17"], "settlement_value": [15.0]}).to_csv(path, index=False)
            reps = pd.DataFrame(
                {
                    "snap_date": [pd.Timestamp("2023-12-29")],
                    "snap_date_source": [pd.Timestamp("2023-12-29")],
                    "expiry": [pd.Timestamp("2024-01-17")],
                    "asset_id": ["VIX_call_vix_atm"],
                    "symbol": ["VIX   240117C00015000"],
                    "kind": ["call"],
                    "moneyness_bucket": ["vix_atm"],
                    "mark": [2.0],
                    "strike": [15.0],
                    "vix_forward": [16.0],
                    "delta": [0.5],
                    "gamma": [0.1],
                    "vega": [4.0],
                    "theta": [-1.0],
                    "iv_proxy": [0.8],
                    "greek_model": ["black76_vx_forward"],
                    "vx_contract": ["VXF4"],
                }
            )
            with patch.dict(os.environ, {"OPTION_MARKOWITZ_VRO_FILE": str(path)}, clear=False):
                returns, detail = build_vix_expiry_proxy_returns(
                    reps, [pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-31")], root=root
                )
            self.assertFalse(returns.empty)
            self.assertEqual(str(detail["settlement_source"].iloc[0]), "vro_soq_exact")
            self.assertAlmostEqual(float(detail["exit_price"].iloc[0]), 0.0)

    def test_incomplete_exact_vro_file_skips_missing_vix_expiries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir(parents=True)
            path = root / "data" / "vro_exact.csv"
            pd.DataFrame({"settlement_date": ["2024-01-17"], "settlement_value": [15.0]}).to_csv(path, index=False)
            reps = pd.DataFrame(
                {
                    "snap_date": [pd.Timestamp("2023-12-29"), pd.Timestamp("2023-12-29")],
                    "snap_date_source": [pd.Timestamp("2023-12-29"), pd.Timestamp("2023-12-29")],
                    "expiry": [pd.Timestamp("2024-01-17"), pd.Timestamp("2024-02-14")],
                    "asset_id": ["VIX_call_jan", "VIX_call_feb"],
                    "symbol": ["VIX   240117C00015000", "VIX   240214C00015000"],
                    "kind": ["call", "call"],
                    "moneyness_bucket": ["vix_atm", "vix_atm"],
                    "mark": [2.0, 2.0],
                    "strike": [15.0, 15.0],
                    "vix_forward": [16.0, 16.0],
                    "delta": [0.5, 0.5],
                    "gamma": [0.1, 0.1],
                    "vega": [4.0, 4.0],
                    "theta": [-1.0, -1.0],
                    "iv_proxy": [0.8, 0.8],
                    "greek_model": ["black76_vx_forward", "black76_vx_forward"],
                    "vx_contract": ["VXF4", "VXG4"],
                }
            )
            with patch.dict(os.environ, {"OPTION_MARKOWITZ_VRO_FILE": str(path)}, clear=False):
                _, detail = build_vix_expiry_proxy_returns(
                    reps, [pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-31")], root=root
                )
            self.assertEqual(set(detail["asset_id"]), {"VIX_call_jan"})
            self.assertTrue(detail["settlement_source"].eq("vro_soq_exact").all())


class TestPublicationCostsAndInference(unittest.TestCase):
    def test_post_cost_returns_are_lower_and_ledgers_reconcile(self):
        dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
        gross = pd.DataFrame({"Strategy": [0.10, -0.02]}, index=dates)
        weights = {"Strategy": pd.Series({"AAA_call": 0.5, "AAA_put": -0.5})}
        cost_inputs = pd.DataFrame(
            {
                "return_date": [dates[0], dates[0], dates[1], dates[1]],
                "asset_id": ["AAA_call", "AAA_put", "AAA_call", "AAA_put"],
                "mark": [5.0, 4.0, 5.0, 4.0],
                "relative_spread": [0.10, 0.12, 0.10, 0.12],
                "holding_years": [21 / 365] * 4,
                "available_volume_contracts": [1000, 1000, 1000, 1000],
                "available_oi_contracts": [5000, 5000, 5000, 5000],
                "borrow_rate_proxy": [0.02, 0.02, 0.02, 0.02],
                "kind": ["call", "put", "call", "put"],
                "asset_class": ["equity_option"] * 4,
                "start_spot": [100.0, 100.0, 100.0, 100.0],
                "strike": [105.0, 95.0, 105.0, 95.0],
            }
        )
        net, costs, capacity, margin, assignment = compute_strategy_cost_ledgers(
            gross, weights, cost_inputs, ResearchCostConfig(nav_for_capacity=100_000.0)
        )
        self.assertTrue((net["Strategy"] < gross["Strategy"]).all())
        self.assertGreater(float(costs["total_cost_nav"].sum()), 0.0)
        self.assertFalse(capacity.empty)
        self.assertFalse(margin.empty)
        self.assertFalse(assignment.empty)
        diag = cost_diagnostics_table(costs, capacity, margin)
        self.assertIn("Ann. cost drag", diag.columns)

    def test_build_cost_input_uses_borrow_and_spread_defaults(self):
        reps = pd.DataFrame(
            {
                "snap_date": [pd.Timestamp("2024-01-31")],
                "asset_id": ["VIX_call_vix_atm"],
                "symbol": ["VIX   240221C00015000"],
                "volume": [100.0],
            }
        )
        detail = pd.DataFrame(
            {
                "return_date": [pd.Timestamp("2024-02-29")],
                "decision_date": [pd.Timestamp("2024-01-31")],
                "expiry": [pd.Timestamp("2024-02-21")],
                "asset_id": ["VIX_call_vix_atm"],
                "symbol": ["VIX   240221C00015000"],
                "asset_class": ["vix_option"],
                "mark": [2.0],
                "expiry_days": [21],
            }
        )
        ledger = build_cost_input_ledger(reps, detail, Path(tempfile.gettempdir()))
        self.assertAlmostEqual(float(ledger["relative_spread"].iloc[0]), ResearchCostConfig().default_vix_option_rel_spread)
        self.assertEqual(float(ledger["borrow_rate_proxy"].iloc[0]), 0.0)

    def test_block_bootstrap_and_hac_are_reproducible(self):
        r = pd.Series([0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.04, 0.01])
        cfg = BootstrapConfig(n_boot=200, seed=123, block_size=3)
        ci1 = block_bootstrap_metric_ci(r, "sharpe", cfg)
        ci2 = block_bootstrap_metric_ci(r, "sharpe", cfg)
        self.assertEqual(ci1, ci2)
        x = pd.DataFrame({"SPY": [0.01, -0.02, 0.03, 0.00, 0.01, -0.01, 0.02, 0.01]})
        fit = hac_ols(r, x, lags=1)
        self.assertIn("alpha_t", fit)
        self.assertEqual(fit["hac_lags"], 1)
        self.assertTrue(np.isfinite(fit["betas"].iloc[0]))

    def test_execution_cost_scenarios_and_required_capital(self):
        dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
        gross = pd.DataFrame({"Strategy": [0.08, 0.03]}, index=dates)
        weights = {"Strategy": pd.Series({"AAA_call": 0.4, "AAA_put": -0.2})}
        cost_inputs = pd.DataFrame(
            {
                "return_date": [dates[0], dates[0], dates[1], dates[1]],
                "asset_id": ["AAA_call", "AAA_put", "AAA_call", "AAA_put"],
                "mark": [5.0, 4.0, 5.0, 4.0],
                "relative_spread": [0.08, 0.10, 0.08, 0.10],
                "available_volume_contracts": [10_000, 10_000, 10_000, 10_000],
                "available_oi_contracts": [50_000, 50_000, 50_000, 50_000],
                "kind": ["call", "put", "call", "put"],
                "asset_class": ["equity_option"] * 4,
                "start_spot": [100.0] * 4,
                "strike": [105.0, 95.0, 105.0, 95.0],
            }
        )
        net, costs, rejected, capital, assignment, required_returns = build_execution_cost_scenarios(
            gross,
            weights,
            cost_inputs,
            config=ExecutionCostScenarioConfig(nav_for_capacity=100_000.0),
        )
        self.assertTrue({"Strategy::mid", "Strategy::half_spread", "Strategy::full_spread"}.issubset(net.columns))
        self.assertLess(float(net["Strategy::full_spread"].mean()), float(net["Strategy::mid"].mean()))
        self.assertFalse(costs.empty)
        self.assertFalse(capital.empty)
        self.assertIn("Strategy::full_spread", required_returns.columns)
        diag = capacity_market_impact_diagnostics(costs, rejected, capital)
        self.assertIn("Max capacity used", diag.columns)
        perf = pd.DataFrame({"Strategy": ["Strategy"], "Sharpe": [1.0]})
        survival = post_cost_survival_table(perf, net, costs, diag)
        self.assertIn("Survives?", survival.columns)
        self.assertFalse(assignment.empty)

    def test_hurdle_liquidity_ablation_and_reality_check(self):
        assets = ["AAA_call", "AAA_put", "BBB_call"]
        mu = pd.Series([0.04, 0.01, -0.01], index=assets)
        risk = pd.Series([0.10, 0.10, 0.10], index=assets)
        cost = pd.Series([0.01, 0.02, 0.01], index=assets)
        hurdle, no_trade = apply_trade_hurdles(mu, risk, cost, hurdle_levels=(0.0, 0.25, 1.0))
        self.assertTrue(hurdle.loc[hurdle["hurdle"].eq(0.0), "passed"].any())
        self.assertFalse(hurdle.loc[hurdle["hurdle"].eq(1.0), "passed"].any())
        self.assertFalse(no_trade.empty)
        cost_inputs = pd.DataFrame(
            {
                "asset_id": assets,
                "available_volume_contracts": [100, 1000, 500],
                "available_oi_contracts": [1000, 2000, 3000],
                "relative_spread": [0.20, 0.05, 0.10],
                "return_date": [pd.Timestamp("2024-01-31")] * 3,
            }
        )
        tiers = liquidity_tier_labels(cost_inputs)
        self.assertIn("all_eligible", set(tiers["liquidity_tier"]))
        returns = pd.DataFrame(np.eye(3) * 0.02, columns=assets)
        strategies = {"Strategy": pd.Series([0.4, -0.2, 0.4], index=assets)}
        perf, diag = liquidity_tier_performance(returns, strategies, tiers)
        self.assertFalse(perf.empty)
        self.assertFalse(diag.empty)
        components = pd.DataFrame(
            {
                "theta_carry": [0.01, 0.02, 0.01],
                "variance_risk_premium": [0.02, -0.01, 0.0],
                "skew_tail_premium": [0.0, -0.01, 0.02],
                "vol_premium": [0.01, 0.0, 0.01],
                "relative_value": [0.0, 0.01, -0.01],
                "shrunk_mu": [0.02, 0.01, 0.03],
            },
            index=assets,
        )
        ablation, ab_components = forecast_ablation_tables(components, returns, strategies["Strategy"])
        self.assertIn("full_conditional_model", set(ablation["Ablation"]))
        self.assertFalse(ab_components.empty)
        rc = sharpe_reality_check(pd.DataFrame({"a": [0.01, -0.01, 0.02, 0.0], "b": [0.0, 0.01, 0.01, -0.01]}), config=BootstrapConfig(n_boot=50, seed=7, block_size=2))
        self.assertTrue({"Probabilistic Sharpe", "Deflated Sharpe"}.issubset(rc.columns))


class TestCostAwareSortinoWiring(unittest.TestCase):
    def test_entry_costs_use_train_window_only(self):
        train_end = pd.Timestamp("2020-12-31")
        cost_inputs = pd.DataFrame(
            {
                "return_date": [pd.Timestamp("2020-06-30"), pd.Timestamp("2021-01-31")],
                "asset_id": ["AAA_call", "AAA_call"],
                "mark": [5.0, 5.0],
                "relative_spread": [0.10, 99.0],
                "asset_class": ["equity_option", "equity_option"],
            }
        )
        filtered = cost_inputs.loc[cost_inputs["return_date"].le(train_end)].copy()

        full_costs, full_diag = derive_entry_cost_series(cost_inputs, ["AAA_call"], train_end=train_end)
        filt_costs, filt_diag = derive_entry_cost_series(filtered, ["AAA_call"], train_end=train_end)

        pd.testing.assert_series_equal(full_costs, filt_costs)
        pd.testing.assert_frame_equal(full_diag, filt_diag)

    def test_entry_cost_formula_half_spread_plus_fees(self):
        train_end = pd.Timestamp("2020-12-31")
        cfg = ResearchCostConfig()
        cost_inputs = pd.DataFrame(
            {
                "return_date": [pd.Timestamp("2020-06-30")],
                "asset_id": ["AAA_call"],
                "mark": [5.0],
                "relative_spread": [0.10],
                "asset_class": ["equity_option"],
            }
        )
        entry_costs, diag = derive_entry_cost_series(cost_inputs, ["AAA_call"], train_end=train_end, config=cfg)
        expected = 0.5 * 0.10 + cfg.fee_per_contract_per_side / (5.0 * cfg.option_multiplier)
        self.assertAlmostEqual(float(entry_costs.loc["AAA_call"]), expected, places=12)
        self.assertAlmostEqual(float(diag.loc[0, "entry_cost"]), expected, places=12)

    def test_entry_cost_default_imputation_and_nonnegativity(self):
        train_end = pd.Timestamp("2020-12-31")
        contracts = ["AAA_call", "VIX_call_vix_atm"]
        cost_inputs = pd.DataFrame(
            {
                "return_date": [pd.Timestamp("2020-06-30")],
                "asset_id": ["AAA_call"],
                "mark": [5.0],
                "relative_spread": [0.10],
                "asset_class": ["equity_option"],
            }
        )
        entry_costs, diag = derive_entry_cost_series(cost_inputs, contracts, train_end=train_end)

        self.assertEqual(entry_costs.index.tolist(), contracts)
        self.assertTrue(np.isfinite(entry_costs.to_numpy(float)).all())
        self.assertTrue((entry_costs >= 0).all())
        self.assertTrue(np.isfinite(pd.to_numeric(diag["entry_cost"], errors="coerce")).all())
        source = dict(zip(diag["asset_id"], diag["source"]))
        self.assertEqual(source["VIX_call_vix_atm"], "default_imputed")

    def test_sortino_guard_returns_cash_on_infeasible(self):
        underlyings = ["AAA", "BBB", "CCC"]
        frame = pd.DataFrame(
            {
                "underlying": underlyings,
                "mark": [5.0, 3.0, 6.5],
                "spot": [100.0, 50.0, 80.0],
                "delta": [0.55, 0.50, 0.60],
                "gamma": [0.020, 0.035, 0.018],
                "vega": [20.0, 12.0, 17.0],
                "theta": [-4.0, -2.0, -3.0],
            },
            index=["AAA_call", "BBB_call", "CCC_call"],
        )
        model = OptionOnlyMarkowitzModel(
            OptionOnlySpec(frame),
            FactorShockSpec(
                pd.DataFrame(
                    [[0.040, 0.010, 0.006], [0.010, 0.050, 0.012], [0.006, 0.012, 0.060]],
                    index=underlyings,
                    columns=underlyings,
                )
            ),
            expected_returns=pd.Series([0.04, 0.03, 0.02], index=frame.index),
            constraints=OptionMarkowitzConstraints(gross_nav=1.0, delta_abs=0.5, long_only=True),
        )
        scenarios = pd.DataFrame(
            np.random.default_rng(7).normal(0.0, 0.08, size=(24, 3)),
            index=pd.date_range("2019-01-31", periods=24, freq="ME"),
            columns=model.contracts,
        )
        weights, diag = _sortino_weights_with_guard(
            model, scenarios, pd.Series(0.0, index=model.contracts), "unit_test"
        )

        self.assertTrue((weights == 0.0).all())
        self.assertEqual(diag["status"], "infeasible")


class TestCbboSurfaceCostInputs(unittest.TestCase):
    DATE = pd.Timestamp("2024-01-31")
    RETURN_DATE = pd.Timestamp("2024-02-29")

    def _fixture(self):
        assets = ["AAA_call_atm", "BBB_call_atm", "CCC_call_atm"]
        reps = pd.DataFrame(
            {
                "snap_date": [self.DATE] * 3,
                "asset_id": assets,
                "symbol": ["AAA   240221C00100000", "BBB   240221C00100000", "CCC   240221C00100000"],
                "underlying": ["AAA", "BBB", "CCC"],
                "volume": [100.0, 100.0, 100.0],
                "open_interest": [1000.0, 1000.0, 1000.0],
                "cbbo_median_relative_spread": [0.04, np.nan, np.nan],
                "moneyness_bucket": ["atm", "atm", "atm"],
                "expiry": [pd.Timestamp("2024-02-21")] * 3,
                "tenor_days": [21] * 3,
            }
        )
        detail = pd.DataFrame(
            {
                "return_date": [self.RETURN_DATE] * 3,
                "decision_date": [self.DATE] * 3,
                "expiry": [pd.Timestamp("2024-02-21")] * 3,
                "asset_id": assets,
                "symbol": ["AAA   240221C00100000", "BBB   240221C00100000", "CCC   240221C00100000"],
                "underlying": ["AAA", "BBB", "CCC"],
                "kind": ["call", "call", "call"],
                "moneyness_bucket": ["atm", "atm", "atm"],
                "asset_class": ["equity_option"] * 3,
                "mark": [5.0, 5.0, 5.0],
                "strike": [100.0, 100.0, 100.0],
                "start_spot": [100.0, 100.0, 100.0],
                "expiry_days": [21] * 3,
            }
        )
        return reps, detail

    def _surface(self, snap_date=None, spread=0.07):
        return pd.DataFrame(
            {
                "underlying": ["BBB"],
                "snap_date": [pd.Timestamp(snap_date or self.DATE)],
                "moneyness_bucket": ["atm"],
                "tenor_bucket": ["le_45d"],
                "median_relative_spread": [spread],
            }
        )

    def test_cost_input_spread_precedence_panel_surface_default(self):
        reps, detail = self._fixture()

        ledger = build_cost_input_ledger(
            reps,
            detail,
            Path(tempfile.gettempdir()),
            spread_surface=self._surface(spread=0.07),
        ).set_index("asset_id")

        self.assertEqual(ledger.loc["AAA_call_atm", "relative_spread_source"], "panel_cbbo")
        self.assertEqual(ledger.loc["BBB_call_atm", "relative_spread_source"], "surface_cbbo")
        self.assertEqual(ledger.loc["CCC_call_atm", "relative_spread_source"], "default")
        self.assertAlmostEqual(float(ledger.loc["AAA_call_atm", "relative_spread"]), 0.04)
        self.assertAlmostEqual(float(ledger.loc["BBB_call_atm", "relative_spread"]), 0.07)
        self.assertAlmostEqual(
            float(ledger.loc["CCC_call_atm", "relative_spread"]),
            ResearchCostConfig().default_equity_option_rel_spread,
        )

    def test_surface_join_is_exact_decision_date_only(self):
        reps, detail = self._fixture()

        ledger = build_cost_input_ledger(
            reps,
            detail,
            Path(tempfile.gettempdir()),
            spread_surface=self._surface(snap_date="2024-02-29", spread=0.03),
        ).set_index("asset_id")

        self.assertEqual(ledger.loc["BBB_call_atm", "relative_spread_source"], "default")
        self.assertAlmostEqual(
            float(ledger.loc["BBB_call_atm", "relative_spread"]),
            ResearchCostConfig().default_equity_option_rel_spread,
        )

    def test_cost_input_ledger_without_surface_is_unchanged(self):
        reps, detail = self._fixture()

        implicit = build_cost_input_ledger(reps, detail, Path(tempfile.gettempdir()))
        explicit_none = build_cost_input_ledger(
            reps,
            detail,
            Path(tempfile.gettempdir()),
            spread_surface=None,
        )

        self.assertIn("relative_spread_source", implicit.columns)
        self.assertEqual(implicit["relative_spread_source"].tolist(), ["panel_cbbo", "default", "default"])
        pd.testing.assert_frame_equal(
            implicit.drop(columns=["relative_spread_source"]),
            explicit_none.drop(columns=["relative_spread_source"]),
            check_exact=True,
        )


def _repair_cost_row(
    date,
    asset_id,
    mark=5.0,
    rel_spread=0.04,
    volume=1_000_000.0,
    oi=1_000_000.0,
    option_return=0.20,
    kind="call",
    start_spot=100.0,
    strike=105.0,
):
    return {
        "return_date": date,
        "asset_id": asset_id,
        "mark": mark,
        "relative_spread": rel_spread,
        "available_volume_contracts": volume,
        "available_oi_contracts": oi,
        "option_return": option_return,
        "kind": kind,
        "asset_class": "equity_option",
        "start_spot": start_spot,
        "strike": strike,
    }


class TestOrderRepairExecution(unittest.TestCase):
    """Order-repair semantics of build_execution_cost_scenarios."""

    DATE = pd.Timestamp("2024-01-31")
    CONFIG = ExecutionCostScenarioConfig(nav_for_capacity=100_000.0, max_relative_spread=0.05)

    def _unit_costs(self, mark):
        fee = 2.0 * self.CONFIG.fee_per_contract_per_side / (mark * self.CONFIG.option_multiplier)
        tick = (0.01 if mark < 3.0 else 0.05) / mark
        return fee, tick

    def _run(self, weights, cost_rows, scenarios=("mid",), repair=None, gross=0.05):
        gross_frame = pd.DataFrame({"S": [gross]}, index=[self.DATE])
        strategies = {"S": pd.Series(weights)}
        cost_inputs = pd.DataFrame(cost_rows)
        return build_execution_cost_scenarios(
            gross_frame,
            strategies,
            cost_inputs,
            config=self.CONFIG,
            scenarios=scenarios,
            repair=repair,
        )

    def test_repair_none_reproduces_legacy_fail_closed_nets(self):
        weights = {"GOOD": 0.5, "WIDE": 0.3, "CAP": 0.4, "MISS": 0.2}
        cost_rows = [
            _repair_cost_row(self.DATE, "GOOD"),
            _repair_cost_row(self.DATE, "WIDE", rel_spread=0.12),
            # 0.4 * 100k / (5.0 * 100) = 80 contracts vs 400 * 10% = 40 => ratio 2.
            _repair_cost_row(self.DATE, "CAP", volume=400.0),
        ]
        scenarios = ("mid", "full_spread")
        legacy = self._run(weights, cost_rows, scenarios=scenarios)
        explicit = self._run(weights, cost_rows, scenarios=scenarios, repair=None)
        self.assertIsInstance(explicit, tuple)
        for a, b in zip(legacy, explicit):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        # Hand-computed legacy nets: GOOD fills, WIDE/CAP/MISS forfeit gross.
        fee, tick = self._unit_costs(5.0)
        foregone = 0.3 * 0.20 + 0.4 * 0.20 + 0.0  # WIDE + CAP + MISS (no ledger row => zero)
        self.assertAlmostEqual(
            float(legacy[0].loc[self.DATE, "S::mid"]), 0.05 - 0.5 * (fee + tick) - foregone, places=12
        )
        self.assertAlmostEqual(
            float(legacy[0].loc[self.DATE, "S::full_spread"]),
            0.05 - 0.5 * (fee + 0.04 + tick) - foregone,
            places=12,
        )
        self.assertEqual(
            set(legacy[2]["reject_reason"]),
            {"spread_too_wide", "capacity_exceeded_no_fill", "missing_cost_input"},
        )
        self.assertTrue(explicit.repaired_rows.empty)
        self.assertNotIn("fill_fraction", legacy[1].columns)

    def test_wide_spread_inside_band_is_repaired_at_touch(self):
        weights = {"WIDE": 0.4}
        # rel_spread 0.12 > config cap 0.05 => rejected fail-closed; repaired
        # because half-spread 0.06 <= price_band_frac 0.10 and 0.12 <= 0.50.
        cost_rows = [_repair_cost_row(self.DATE, "WIDE", mark=5.0, rel_spread=0.12)]
        result = self._run(weights, cost_rows, repair=RepairConfig())
        net = result[0]
        self.assertIn("S::mid_repaired", net.columns)
        fee, tick = self._unit_costs(5.0)
        actual_cost = 0.4 * (fee + 0.5 * 0.12 + tick)
        # Gross contribution kept; only the actual crossing cost is charged.
        self.assertAlmostEqual(float(net.loc[self.DATE, "S::mid_repaired"]), 0.05 - actual_cost, places=12)
        self.assertTrue(result[2].empty)
        ledger = result.repaired_rows
        self.assertEqual(len(ledger), 1)
        row = ledger.iloc[0]
        self.assertEqual(row["repair_reason"], "spread_too_wide")
        self.assertAlmostEqual(float(row["decision_mark"]), 5.0)
        self.assertAlmostEqual(float(row["effective_fill_price"]), 5.0 * (1.0 + 0.06))
        # mid scenario intends zero spread cost => extra cost is the crossing.
        self.assertAlmostEqual(float(row["extra_cost_nav"]), 0.4 * 0.5 * 0.12, places=12)
        self.assertAlmostEqual(float(row["fill_fraction"]), 1.0)
        self.assertEqual(result[1]["fill_status"].iloc[0], "repaired_fill")
        # Repaired and unrepaired scenarios coexist in one table.
        base = self._run(weights, cost_rows, repair=None)
        combined = pd.concat([base[0], net], axis=1)
        self.assertIn("S::mid", combined.columns)
        self.assertIn("S::mid_repaired", combined.columns)
        self.assertAlmostEqual(float(combined.loc[self.DATE, "S::mid"]), 0.05 - 0.4 * 0.20, places=12)

    def test_wide_spread_outside_band_or_above_cap_stays_rejected(self):
        # Half-spread 0.15 > price_band_frac 0.10 => outside band => reject.
        result = self._run({"WIDE": 0.4}, [_repair_cost_row(self.DATE, "WIDE", rel_spread=0.30)], repair=RepairConfig())
        self.assertAlmostEqual(float(result[0].loc[self.DATE, "S::mid_repaired"]), 0.05 - 0.4 * 0.20, places=12)
        self.assertEqual(result[2]["reject_reason"].tolist(), ["spread_too_wide"])
        self.assertTrue(result.repaired_rows.empty)
        # Inside a wide band but above max_rel_spread => junk quote => reject.
        junk = self._run(
            {"WIDE": 0.4},
            [_repair_cost_row(self.DATE, "WIDE", rel_spread=0.55)],
            repair=RepairConfig(price_band_frac=0.40, max_rel_spread=0.50),
        )
        self.assertEqual(junk[2]["reject_reason"].tolist(), ["spread_too_wide"])
        self.assertTrue(junk.repaired_rows.empty)

    def test_capacity_breach_becomes_pro_rata_half_fill(self):
        # 80 intended contracts vs capacity 40 => ratio 2 => fill 0.5.
        cost_rows = [_repair_cost_row(self.DATE, "CAP", volume=400.0)]
        result = self._run({"CAP": 0.4}, cost_rows, repair=RepairConfig())
        fee, tick = self._unit_costs(5.0)
        half_cost = 0.4 * 0.5 * (fee + tick)  # costs on the filled half only
        foregone_half = 0.5 * 0.4 * 0.20  # unfilled half forfeits pro-rata gross
        self.assertAlmostEqual(
            float(result[0].loc[self.DATE, "S::mid_repaired"]), 0.05 - half_cost - foregone_half, places=12
        )
        self.assertTrue(result[2].empty)
        row = result.repaired_rows.iloc[0]
        self.assertEqual(row["repair_reason"], "capacity_partial_fill")
        self.assertAlmostEqual(float(row["fill_fraction"]), 0.5)
        self.assertAlmostEqual(float(row["extra_cost_nav"]), 0.0)
        self.assertAlmostEqual(float(row["foregone_gross_return_nav"]), foregone_half, places=12)
        self.assertAlmostEqual(float(result[1]["total_cost_nav"].iloc[0]), half_cost, places=12)
        diag = repair_diagnostics_table(result.repaired_rows)
        self.assertEqual(int(diag["Repaired orders"].iloc[0]), 1)
        self.assertEqual(int(diag["Partial fills"].iloc[0]), 1)
        self.assertAlmostEqual(float(diag["Avg extra cost per repaired order"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(diag["Foregone gross from unfilled remainders"].iloc[0]), foregone_half, places=12)

    def test_capacity_fill_below_min_fill_frac_is_full_rejection(self):
        # 80 intended contracts vs capacity 4 => ratio 20 => fraction 0.05 < 0.10.
        cost_rows = [_repair_cost_row(self.DATE, "CAP", volume=40.0)]
        result = self._run({"CAP": 0.4}, cost_rows, repair=RepairConfig())
        self.assertAlmostEqual(float(result[0].loc[self.DATE, "S::mid_repaired"]), 0.05 - 0.4 * 0.20, places=12)
        self.assertEqual(result[2]["reject_reason"].tolist(), ["capacity_exceeded_no_fill"])
        self.assertTrue(result.repaired_rows.empty)

    def test_missing_input_and_assignment_risk_are_not_repairable(self):
        weights = {"MISS": 0.2, "SHORTC": -0.1}
        # Deep-ITM low-extrinsic short call: spot 100 vs strike 50, mark 50.5.
        cost_rows = [
            _repair_cost_row(
                self.DATE,
                "SHORTC",
                mark=50.5,
                rel_spread=0.02,
                option_return=-0.30,
                start_spot=100.0,
                strike=50.0,
            )
        ]
        result = self._run(weights, cost_rows, repair=RepairConfig())
        # MISS forfeits zero (no ledger row); SHORTC forfeits -0.1 * -0.30.
        self.assertAlmostEqual(float(result[0].loc[self.DATE, "S::mid_repaired"]), 0.05 - 0.03, places=12)
        self.assertEqual(
            set(result[2]["reject_reason"]),
            {"missing_cost_input", "deep_itm_low_extrinsic_short_option"},
        )
        self.assertTrue(result.repaired_rows.empty)
        self.assertTrue(result[4]["blocked"].any())


class TestRepairWiringTables(unittest.TestCase):
    """Publication-pipeline tables and labels for repaired execution scenarios."""

    DATE = pd.Timestamp("2024-01-31")
    CONFIG = ExecutionCostScenarioConfig(nav_for_capacity=100_000.0, max_relative_spread=0.05)

    def _run(self, weights, cost_rows, scenarios=("mid", "full_spread"), repair=None, gross=0.05):
        gross_frame = pd.DataFrame({"S": [gross]}, index=[self.DATE])
        strategies = {"S": pd.Series(weights)}
        cost_inputs = pd.DataFrame(cost_rows)
        return build_execution_cost_scenarios(
            gross_frame,
            strategies,
            cost_inputs,
            config=self.CONFIG,
            scenarios=scenarios,
            repair=repair,
        )

    def test_repair_comparison_table_matches_performance_stats(self):
        dates = pd.date_range("2024-01-31", periods=4, freq="ME")
        legacy = pd.Series([0.01, -0.02, 0.03, 0.00], index=dates)
        repaired = pd.Series([0.02, -0.01, 0.04, 0.01], index=dates)
        net_legacy = pd.DataFrame(
            {
                "S::mid": legacy,
                "S::full_spread": [0.00, -0.03, 0.02, -0.01],
            },
            index=dates,
        )
        net_repaired = pd.DataFrame({"S::mid_repaired": repaired}, index=dates)
        table = execution_repair_comparison_table(net_legacy, net_repaired, periods_per_year=12.0)

        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        legacy_stats = performance_stats(legacy, 12.0)
        repaired_stats = performance_stats(repaired, 12.0)
        self.assertEqual(row["Strategy"], "S")
        self.assertEqual(row["Scenario"], "mid")
        self.assertAlmostEqual(float(row["Fail-closed Sharpe"]), legacy_stats["sharpe"], places=12)
        self.assertAlmostEqual(float(row["Repaired Sharpe"]), repaired_stats["sharpe"], places=12)
        self.assertAlmostEqual(float(row["Fail-closed ann. return"]), legacy_stats["ann_return"], places=12)
        self.assertAlmostEqual(float(row["Repaired ann. return"]), repaired_stats["ann_return"], places=12)
        self.assertAlmostEqual(
            float(row["Repair uplift (ann.)"]),
            repaired_stats["ann_return"] - legacy_stats["ann_return"],
            places=12,
        )

    def test_repaired_and_legacy_scenario_labels_disjoint(self):
        weights = {"GOOD": 0.2, "WIDE": 0.3, "CAP": 0.4, "MISS": 0.2, "SHORTC": -0.1}
        cost_rows = [
            _repair_cost_row(self.DATE, "GOOD"),
            _repair_cost_row(self.DATE, "WIDE", rel_spread=0.12),
            _repair_cost_row(self.DATE, "CAP", volume=400.0),
            _repair_cost_row(
                self.DATE,
                "SHORTC",
                mark=50.5,
                rel_spread=0.02,
                option_return=-0.30,
                start_spot=100.0,
                strike=50.0,
            ),
        ]
        legacy_default = self._run(weights, cost_rows)
        legacy_explicit = self._run(weights, cost_rows, repair=None)
        repaired = self._run(weights, cost_rows, repair=RepairConfig())

        self.assertEqual(list(legacy_default.net.columns), list(legacy_explicit.net.columns))
        self.assertTrue(set(legacy_default.net.columns).isdisjoint(set(repaired.net.columns)))
        self.assertTrue(all(str(col).rsplit("::", 1)[-1].endswith("_repaired") for col in repaired.net.columns))
        self.assertFalse(repaired.repaired_rows.empty)
        self.assertTrue(repaired.repaired_rows["scenario"].astype(str).str.endswith("_repaired").all())
        repair_reasons = repaired.repaired_rows["repair_reason"].astype(str)
        forbidden_reasons = (
            "missing_cost_input",
            "deep_itm_low_extrinsic_short_option",
            "short_call_dividend_exercise_risk",
            "hard_to_borrow_short_call",
        )
        for reason in forbidden_reasons:
            self.assertFalse(repair_reasons.str.contains(reason, regex=False).any())
        self.assertNotIn("MISS", set(repaired.repaired_rows["asset_id"]))
        self.assertNotIn("SHORTC", set(repaired.repaired_rows["asset_id"]))

    def test_repair_diagnostics_empty_ledger_returns_schema_only(self):
        diag = repair_diagnostics_table(pd.DataFrame())
        self.assertEqual(
            list(diag.columns),
            [
                "Scenario",
                "Repaired orders",
                "Partial fills",
                "Avg extra cost per repaired order",
                "Foregone gross from unfilled remainders",
            ],
        )
        self.assertEqual(len(diag), 0)


if __name__ == "__main__":
    unittest.main()
