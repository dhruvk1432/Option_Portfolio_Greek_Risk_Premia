
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
    apply_trade_hurdles,
    build_execution_cost_scenarios,
    capacity_market_impact_diagnostics,
    forecast_ablation_tables,
    liquidity_tier_labels,
    liquidity_tier_performance,
    post_cost_survival_table,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (  # noqa: E402
    ResearchCostConfig,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    cost_diagnostics_table,
)
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
        self.assertTrue(str(paths["method"].iloc[0]).startswith("ewma_residual_fallback"))
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


if __name__ == "__main__":
    unittest.main()
