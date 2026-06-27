"""Tests for the option-only Markowitz model and paper data contract."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis import run_empirics as empirics
from research.papers.option_only_markowitz.analysis.conditional_premia import (
    ConditionalPremiaConfig,
    conditional_expected_returns,
)
from research.papers.option_only_markowitz.analysis.vix_option_panel import (
    align_vx_forward,
    black76_greeks,
    black76_price,
    parse_osi_symbol,
)
from src.portfolio import (
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    bs_greeks,
    bs_price,
    nearest_psd,
    performance_stats,
    shrink_covariance,
    taylor_option_pnl,
)


def make_option_only_model(
    constraints: OptionMarkowitzConstraints | None = None,
) -> OptionOnlyMarkowitzModel:
    underlyings = ["AAA", "BBB", "CCC"]
    options = pd.DataFrame(
        [
            ("AAA_call", "AAA", 5.00, 100.0, 0.55, 0.020, 20.0, -4.0, 0.040),
            ("AAA_put", "AAA", 4.50, 100.0, -0.45, 0.022, 19.0, -3.5, 0.015),
            ("BBB_call", "BBB", 3.00, 50.0, 0.50, 0.035, 12.0, -2.0, 0.035),
            ("BBB_put", "BBB", 2.80, 50.0, -0.42, 0.032, 11.0, -1.8, 0.010),
            ("CCC_call", "CCC", 6.50, 80.0, 0.60, 0.018, 17.0, -3.0, 0.025),
            ("CCC_put", "CCC", 4.00, 80.0, -0.35, 0.020, 14.0, -2.5, 0.005),
        ],
        columns=[
            "contract",
            "underlying",
            "mark",
            "spot",
            "delta",
            "gamma",
            "vega",
            "theta",
            "expected_return",
        ],
    ).set_index("contract")
    underlying_cov = pd.DataFrame(
        [[0.040, 0.010, 0.006], [0.010, 0.050, 0.012], [0.006, 0.012, 0.060]],
        index=underlyings,
        columns=underlyings,
    )
    vol_cov = pd.DataFrame(
        [[0.0040, 0.0010, 0.0008], [0.0010, 0.0030, 0.0009], [0.0008, 0.0009, 0.0035]],
        index=underlyings,
        columns=underlyings,
    )
    return OptionOnlyMarkowitzModel(
        OptionOnlySpec(options.drop(columns=["expected_return"])),
        FactorShockSpec(underlying_cov=underlying_cov, vol_cov=vol_cov, horizon_years=21 / 252),
        expected_returns=options["expected_return"],
        constraints=constraints or OptionMarkowitzConstraints(gross_nav=1.0),
        covariance_shrinkage=0.05,
    )


class TestBlackScholesAndTaylor(unittest.TestCase):
    def test_bsm_greeks_match_finite_differences(self):
        S, K, T, r, sigma = 101.0, 99.0, 0.4, 0.03, 0.24
        h_s = 1e-2
        h_v = 1e-4
        price = bs_price(S, K, T, r, sigma, "call")
        price_up = bs_price(S + h_s, K, T, r, sigma, "call")
        price_dn = bs_price(S - h_s, K, T, r, sigma, "call")
        vol_up = bs_price(S, K, T, r, sigma + h_v, "call")
        vol_dn = bs_price(S, K, T, r, sigma - h_v, "call")
        greeks = bs_greeks(S, K, T, r, sigma, "call")

        self.assertAlmostEqual(greeks["delta"], (price_up - price_dn) / (2 * h_s), places=5)
        self.assertAlmostEqual(
            greeks["gamma"], (price_up - 2 * price + price_dn) / (h_s * h_s), places=5
        )
        self.assertAlmostEqual(greeks["vega"], (vol_up - vol_dn) / (2 * h_v), places=5)

    def test_taylor_option_pnl_formula(self):
        pnl = taylor_option_pnl(
            delta=np.array([0.5, -0.4]),
            gamma=np.array([0.1, 0.2]),
            vega=np.array([2.0, 1.5]),
            theta=np.array([-1.0, -0.5]),
            dS=np.array([1.0, -2.0]),
            dvol=np.array([0.02, -0.01]),
            dt=1 / 252,
        )
        expected = np.array(
            [
                0.5 * 1.0 + 0.5 * 0.1 * 1.0**2 + 2.0 * 0.02 - 1.0 / 252,
                -0.4 * -2.0 + 0.5 * 0.2 * (-2.0) ** 2 + 1.5 * -0.01 - 0.5 / 252,
            ]
        )
        np.testing.assert_allclose(pnl, expected)

    def test_taylor_approximates_small_bsm_move(self):
        S, K, T, r, sigma = 100.0, 101.0, 0.25, 0.03, 0.22
        dS, dvol, dt = 0.20, 0.002, 1 / 252
        price0 = bs_price(S, K, T, r, sigma, "put")
        price1 = bs_price(S + dS, K, T - dt, r, sigma + dvol, "put")
        g = bs_greeks(S, K, T, r, sigma, "put")
        approx = taylor_option_pnl(
            np.array([g["delta"]]),
            np.array([g["gamma"]]),
            np.array([g["vega"]]),
            np.array([g["theta"]]),
            np.array([dS]),
            np.array([dvol]),
            dt,
        )[0]
        self.assertLess(abs((price1 - price0) - approx), 2e-3)


class TestVixAndConditionalPremia(unittest.TestCase):
    def test_vix_osi_parser_preserves_contract_terms(self):
        root, expiry, kind, strike = parse_osi_symbol("VIX   260617C00030000")
        self.assertEqual(root, "VIX")
        self.assertEqual(expiry, pd.Timestamp("2026-06-17"))
        self.assertEqual(kind, "call")
        self.assertEqual(strike, 30.0)

        root, expiry, kind, strike = parse_osi_symbol("VIX260715P00012500")
        self.assertEqual(root, "VIX")
        self.assertEqual(expiry, pd.Timestamp("2026-07-15"))
        self.assertEqual(kind, "put")
        self.assertEqual(strike, 12.5)

    def test_black76_greeks_match_finite_differences(self):
        F, K, T, r, sigma = 22.0, 20.0, 35 / 365, 0.03, 0.75
        h_f = 1e-3
        h_v = 1e-4
        price = black76_price(F, K, T, r, sigma, "call")
        price_up = black76_price(F + h_f, K, T, r, sigma, "call")
        price_dn = black76_price(F - h_f, K, T, r, sigma, "call")
        vol_up = black76_price(F, K, T, r, sigma + h_v, "call")
        vol_dn = black76_price(F, K, T, r, sigma - h_v, "call")
        greeks = black76_greeks(F, K, T, r, sigma, "call")

        self.assertAlmostEqual(greeks["delta"], (price_up - price_dn) / (2 * h_f), places=5)
        self.assertAlmostEqual(greeks["gamma"], (price_up - 2 * price + price_dn) / (h_f * h_f), places=5)
        self.assertAlmostEqual(greeks["vega"], (vol_up - vol_dn) / (2 * h_v), places=5)

    def test_vx_forward_alignment_uses_latest_prior_curve_and_nearest_expiry(self):
        lookup = {
            pd.Timestamp("2024-01-02"): pd.DataFrame(
                {
                    "settlement_date": [pd.Timestamp("2024-01-17"), pd.Timestamp("2024-02-14")],
                    "forward_price": [14.5, 16.25],
                    "contract": ["VXF4", "VXG4"],
                }
            )
        }
        out = align_vx_forward(
            pd.Series([pd.Timestamp("2024-01-03")]),
            pd.Series([pd.Timestamp("2024-02-16")]),
            lookup=lookup,
        )
        self.assertAlmostEqual(float(out["vix_forward"].iloc[0]), 16.25)
        self.assertEqual(out["vx_contract"].iloc[0], "VXG4")

    def test_conditional_expected_returns_are_pit_shrunk_and_bounded(self):
        spec = pd.DataFrame(
            {
                "underlying": ["AAA", "AAA", "VX_FRONT"],
                "mark": [5.0, 4.0, 2.5],
                "spot": [100.0, 100.0, 18.0],
                "delta": [0.5, -0.4, 0.3],
                "gamma": [0.02, 0.03, 0.08],
                "vega": [20.0, 19.0, 4.0],
                "theta": [-4.0, -3.0, -1.0],
                "kind": ["call", "put", "call"],
                "moneyness_bucket": ["atm", "put_near", "vix_call_wing"],
                "asset_class": ["equity_option", "equity_option", "vix_option"],
                "iv_proxy": [0.35, 0.40, 0.95],
            },
            index=["AAA_call", "AAA_put", "VIX_call"],
        )
        dates = pd.date_range("2020-01-31", periods=4, freq="ME")
        option_returns = pd.DataFrame(
            [[0.1, -0.2, -0.5], [0.0, 0.1, 0.2], [0.3, -0.1, -1.0], [-0.1, 0.0, 0.1]],
            index=dates,
            columns=spec.index,
        )
        under_returns = pd.DataFrame(
            {"AAA": [0.01, 0.02, -0.01, 0.00], "VX_FRONT": [0.05, -0.02, 0.04, -0.01]},
            index=dates,
        )
        vol_shocks = pd.DataFrame({"AAA": [0.01, 0.00, -0.02, 0.01], "VX_FRONT": [0.10, -0.05, 0.03, 0.00]}, index=dates)
        config = ConditionalPremiaConfig(shrinkage_to_zero=0.50, max_abs_monthly_mu=0.05)
        mu, components = conditional_expected_returns(spec, option_returns, under_returns, vol_shocks, config)

        self.assertEqual(list(mu.index), list(spec.index))
        self.assertTrue(np.isfinite(mu.to_numpy(float)).all())
        self.assertLessEqual(float(mu.abs().max()), 0.05 + 1e-12)
        self.assertIn("skew_tail_premium", components.columns)
        self.assertLess(float(components.loc["VIX_call", "skew_tail_premium"]), 0.0)

        zero_mu, _ = conditional_expected_returns(
            spec,
            option_returns,
            under_returns,
            vol_shocks,
            ConditionalPremiaConfig(shrinkage_to_zero=1.0),
        )
        np.testing.assert_allclose(zero_mu.to_numpy(float), 0.0)


class TestOptionOnlyModel(unittest.TestCase):
    def test_validation_rejects_bad_option_inputs(self):
        model = make_option_only_model()
        bad = model.frame.drop(columns=["vega"])
        with self.assertRaises(ValueError):
            OptionOnlySpec(bad).validate()
        bad = model.frame.copy()
        bad.iloc[0, bad.columns.get_loc("mark")] = 0.0
        with self.assertRaises(ValueError):
            OptionOnlySpec(bad).validate()

    def test_validation_rejects_missing_underlying_covariance(self):
        model = make_option_only_model()
        cov = pd.DataFrame([[0.04]], index=["AAA"], columns=["AAA"])
        with self.assertRaises(ValueError):
            FactorShockSpec(cov).validate(model.underlyings)

    def test_psd_covariance_construction(self):
        model = make_option_only_model()
        cov = model.covariance_frame()
        self.assertEqual(list(cov.index), model.contracts)
        self.assertEqual(list(cov.columns), model.contracts)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov.to_numpy()).min(), -1e-9)
        self.assertGreaterEqual(np.linalg.eigvalsh(model.factor_cov).min(), -1e-9)

    def test_covariance_helpers_repair_indefinite_matrix(self):
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
        repaired = nearest_psd(indefinite)
        self.assertGreaterEqual(np.linalg.eigvalsh(repaired).min(), -1e-10)
        shrunk = shrink_covariance(indefinite, shrinkage=0.5)
        self.assertGreaterEqual(np.linalg.eigvalsh(shrunk).min(), -1e-10)

    def test_closed_form_tangency_matches_formula(self):
        model = make_option_only_model()
        weights = model.tangency_weights()
        raw = np.linalg.pinv(nearest_psd(model.option_cov)) @ model.expected_returns.to_numpy(float)
        expected = raw / np.abs(raw).sum()
        np.testing.assert_allclose(weights.to_numpy(), expected, atol=1e-10)
        self.assertAlmostEqual(weights.abs().sum(), 1.0, places=10)
        self.assertTrue(math.isfinite(float(weights.sum())))

    def test_constrained_optimizer_respects_nav_and_concentration(self):
        constraints = OptionMarkowitzConstraints(
            gross_nav=1.0,
            net_nav_abs=1.0,
            short_nav_abs=0.20,
            per_contract_abs=0.45,
            underlying_gross={"AAA": 0.60, "BBB": 0.60, "CCC": 0.60},
            long_only=True,
        )
        model = make_option_only_model(constraints)
        result = model.solve_max_sharpe()
        self.assertIn(result.status, ("optimal", "feasible_suboptimal"))
        self.assertAlmostEqual(result.gross_nav, 1.0, places=6)
        self.assertGreater(result.volatility, 0.0)
        self.assertTrue(math.isfinite(result.sharpe))
        self.assertGreaterEqual(result.weights.min(), -1e-7)
        self.assertLessEqual(float(np.maximum(-result.weights, 0.0).sum()), 0.20 + 1e-6)
        self.assertLessEqual(result.weights.abs().max(), 0.45 + 1e-6)
        for under, limit in constraints.underlying_gross.items():
            idx = model.frame["underlying"].eq(under)
            self.assertLessEqual(result.weights.loc[idx].abs().sum(), limit + 1e-6)

    def test_short_nav_constraint_limits_short_option_premium(self):
        constraints = OptionMarkowitzConstraints(
            gross_nav=1.0,
            net_nav_abs=1.0,
            short_nav_abs=0.15,
            per_contract_abs=0.50,
        )
        model = make_option_only_model(constraints)
        result = model.solve_max_sharpe()
        self.assertLessEqual(float(np.maximum(-result.weights, 0.0).sum()), 0.15 + 1e-6)
        self.assertAlmostEqual(result.weights.abs().sum(), 1.0, places=6)

    def test_vix_beta_named_factor_and_stress_constraints_bind(self):
        base = make_option_only_model()
        frame = base.frame.copy()
        frame["asset_class"] = "equity_option"
        frame.loc[["CCC_call", "CCC_put"], "asset_class"] = "vix_option"
        frame["beta_spy_nav"] = [0.40, -0.40, 0.25, -0.25, 0.00, 0.00]
        frame["exposure_vx_front"] = [0.00, 0.00, 0.00, 0.00, 0.30, -0.30]
        frame["stress_scenario_crash"] = [-0.10, 0.08, -0.06, 0.05, -0.02, 0.10]
        constraints = OptionMarkowitzConstraints(
            gross_nav=1.0,
            net_nav_abs=1.0,
            short_nav_abs=0.35,
            per_contract_abs=0.40,
            beta_spy_abs=0.12,
            vix_vega_abs=1.25,
            stress_loss_abs=0.05,
            factor_exposure_abs={"vx_front": 0.08},
        )
        model = OptionOnlyMarkowitzModel(
            OptionOnlySpec(frame),
            base.shocks,
            base.expected_returns,
            constraints=constraints,
            covariance_shrinkage=0.05,
        )
        result = model.solve_max_sharpe()
        w = result.weights.to_numpy(float)
        self.assertIn(result.status, ("optimal", "feasible_suboptimal"))
        self.assertLessEqual(abs(float(model.greeks["beta_spy_nav"].to_numpy(float) @ w)), 0.12 + 2e-5)
        self.assertLessEqual(abs(float(model.greeks["vix_vega_nav"].to_numpy(float) @ w)), 1.25 + 2e-5)
        self.assertLessEqual(abs(float(model.greeks["exposure_vx_front"].to_numpy(float) @ w)), 0.08 + 2e-5)
        self.assertGreaterEqual(float(model.greeks["stress_scenario_crash"].to_numpy(float) @ w), -0.05 - 2e-5)
        self.assertLessEqual(model._max_constraint_violation(w), 2e-5)

    def test_portfolio_return_and_risk_calibration_schema(self):
        model = make_option_only_model()
        dates = pd.date_range("2024-01-31", periods=5, freq="ME")
        returns = pd.DataFrame(
            np.linspace(-0.02, 0.03, num=len(dates) * len(model.contracts)).reshape(
                len(dates), len(model.contracts)
            ),
            index=dates,
            columns=model.contracts,
        )
        weights = model.equal_premium_weights()
        series = model.portfolio_return_series(returns, weights)
        calibration = model.risk_calibration(returns, weights)
        self.assertEqual(len(series), len(dates))
        self.assertSetEqual(
            set(calibration),
            {"predicted_vol", "realized_vol", "realized_to_predicted"},
        )
        self.assertGreaterEqual(calibration["predicted_vol"], 0.0)

    def test_portfolio_return_uses_paid_option_premium_as_denominator(self):
        model = make_option_only_model()
        returns = pd.DataFrame(
            [[0.20, -1.00, 0.00, 0.00, 0.00, 0.00]],
            index=[pd.Timestamp("2024-01-31")],
            columns=model.contracts,
        )
        weights = pd.Series(
            [0.10, 0.05, 0.0, 0.0, 0.0, 0.0],
            index=model.contracts,
        )

        series = model.portfolio_return_series(returns, weights)
        self.assertAlmostEqual(float(series.iloc[0]), 0.10 * 0.20 + 0.05 * -1.00)

    def test_performance_stats_include_downside_and_active_ratios(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02])
        benchmark = pd.Series([0.01, 0.00, 0.02, -0.01])
        stats = performance_stats(returns, periods_per_year=12.0, benchmark_returns=benchmark)
        downside = math.sqrt((0.0**2 + (-0.01) ** 2 + 0.0**2 + (-0.02) ** 2) / 4.0) * math.sqrt(12.0)

        self.assertSetEqual(
            set(stats),
            {
                "ann_return",
                "ann_vol",
                "sharpe",
                "downside_ann_dev",
                "sortino",
                "max_drawdown",
                "calmar",
                "omega",
                "information_ratio",
            },
        )
        self.assertAlmostEqual(stats["downside_ann_dev"], downside)
        self.assertAlmostEqual(stats["sortino"], stats["ann_return"] / downside)
        self.assertGreater(stats["omega"], 0.0)
        self.assertTrue(math.isfinite(stats["information_ratio"]))


class TestEmpiricalPaperContract(unittest.TestCase):
    DATA = ROOT / "data/feature_store/option_greek_proxy_panel.parquet"
    QUALITY = ROOT / "data/feature_store/option_greek_quality.csv"
    PAPER = ROOT / "research/papers/option_only_markowitz"

    @unittest.skipUnless(DATA.exists(), "local OPRA-derived feature store not present")
    def test_local_panel_schema_split_and_returns(self):
        panel, reps, returns = empirics.load_bucket_panel()
        train = returns.loc[: empirics.TRAIN_END]
        test = returns.loc[returns.index > empirics.TRAIN_END]
        spec = empirics.representative_specs(reps, returns)

        self.assertGreater(len(panel), 100_000)
        self.assertGreater(returns.index.nunique(), 100)
        self.assertGreaterEqual(len(spec), 30)
        self.assertLessEqual(train.index.max(), empirics.TRAIN_END)
        self.assertGreater(test.index.min(), empirics.TRAIN_END)
        values = returns.to_numpy(float)
        self.assertTrue(np.isfinite(values[~np.isnan(values)]).all())
        self.assertTrue((spec["mark"] > 0).all())
        self.assertTrue(np.isfinite(spec[["delta", "gamma", "vega", "theta"]].to_numpy(float)).all())

    @unittest.skipUnless(QUALITY.exists(), "Greek quality summary not present")
    def test_local_greek_coverage_for_primary_underlyings(self):
        quality = pd.read_csv(self.QUALITY)
        primary = quality[quality["underlying"].isin(empirics.PRIMARY_UNDERLYINGS)]
        self.assertEqual(set(primary["underlying"]), set(empirics.PRIMARY_UNDERLYINGS))
        for col in ["valid_delta_share", "valid_gamma_share", "valid_vega_share"]:
            self.assertGreaterEqual(float(primary[col].min()), 0.95)
        self.assertEqual(int(primary["impossible_delta_count"].sum()), 0)
        self.assertEqual(int(primary["impossible_gamma_count"].sum()), 0)

    def test_generated_paper_artifact_contract(self):
        required = [
            "option_only_markowitz_cashflow_engineering_dhruv_kohli.pdf",
            "option_only_markowitz_cashflow_engineering_dhruv_kohli.tex",
            "REPRODUCIBILITY.md",
            "docs/source_ledger.md",
            "tables/data_summary.tex",
            "tables/portfolio_performance.tex",
            "tables/portfolio_performance_diagnostics.tex",
            "tables/portfolio_performance_net_diagnostics.tex",
            "tables/inference_summary.tex",
            "tables/cost_capacity_margin_diagnostics.tex",
            "tables/vix_settlement_coverage.tex",
            "tables/risk_calibration.tex",
            "tables/approximation_diagnostics.tex",
            "tables/timing_diagnostics.tex",
            "tables/trading_data_audit.tex",
            "tables/exposure_summary.tex",
            "tables/greek_exposure_summary.tex",
            "tables/factor_regression.tex",
            "tables/pnl_attribution.tex",
            "tables/regime_performance.tex",
            "tables/vix_regime_performance.tex",
            "tables/leave_one_out.tex",
            "tables/rolling_oos.tex",
            "tables/claim_strength_summary.tex",
            "tables/claim_audit.tex",
            "tables/empirical_summary.json",
            "figures/portfolio_growth.pdf",
            "figures/portfolio_growth_all_strategies.pdf",
            "figures/random_sharpe_histogram.pdf",
            "figures/risk_calibration.pdf",
            "figures/regime_sharpes.pdf",
            "figures/vix_regime_sharpes.pdf",
            "figures/leave_one_out_sharpe.pdf",
            "artifacts/strategy_weights.csv",
            "artifacts/strategy_returns.csv",
            "artifacts/strategy_returns_post_cost.csv",
            "artifacts/cost_ledger.csv",
            "artifacts/capacity_ledger.csv",
            "artifacts/research_margin_ledger.csv",
            "artifacts/assignment_risk_ledger.csv",
            "artifacts/inference_summary.csv",
            "artifacts/vix_settlement_coverage.csv",
            "artifacts/equity_benchmark_weights.csv",
            "artifacts/factor_regression.csv",
            "artifacts/pnl_attribution.csv",
            "artifacts/regime_performance.csv",
            "artifacts/vix_regime_performance.csv",
            "artifacts/leave_one_out.csv",
            "artifacts/rolling_oos.csv",
            "artifacts/claim_strength_summary.csv",
            "artifacts/claim_audit.csv",
            "artifacts/holding_return_detail.csv",
            "artifacts/vix_holding_return_detail.csv",
            "artifacts/vix_data_audit.csv",
            "artifacts/timing_diagnostics.csv",
            "artifacts/trading_data_audit.csv",
            "artifacts/split_adjustments.csv",
            "artifacts/random_feasible_sharpes.csv",
            "artifacts/conditional_premia_components.csv",
            "artifacts/figure_visibility_audit.csv",
        ]
        missing = [rel for rel in required if not (self.PAPER / rel).exists()]
        self.assertEqual(missing, [])

        summary = json.loads((self.PAPER / "tables/empirical_summary.json").read_text())
        perf = {row["Strategy"]: row for row in summary.get("performance_gross_only", summary["performance"])}
        for strategy in [
            "Equity-option Greek Markowitz",
            "Greek Markowitz + VIX",
            "Beta/delta-neutral + VIX",
            "VIX hedge sleeve",
            "Delta-matched equities",
            "Underlying Markowitz",
        ]:
            self.assertIn(strategy, perf)
        self.assertAlmostEqual(perf["Equity-option Greek Markowitz"]["Sharpe"], 0.8421194565895301, places=9)
        self.assertAlmostEqual(perf["Greek Markowitz + VIX"]["Sharpe"], 1.3743885779509968, places=9)
        self.assertGreater(perf["Greek Markowitz + VIX"]["Sharpe"], perf["Equity-option Greek Markowitz"]["Sharpe"])
        self.assertGreater(perf["Greek Markowitz + VIX"]["Sharpe"], perf["Delta-matched equities"]["Sharpe"])
        self.assertLess(perf["VIX hedge sleeve"]["Sharpe"], 0.0)
        for metric in ["Sortino", "Calmar", "Omega", "Info. ratio"]:
            self.assertIn(metric, perf["Greek Markowitz + VIX"])
        self.assertAlmostEqual(summary["random_feasible"]["p95_sharpe"], 1.1449656633511185, places=9)
        self.assertGreater(perf["Greek Markowitz + VIX"]["Sharpe"], summary["random_feasible"]["p95_sharpe"])
        self.assertGreaterEqual(summary["data"]["bucket_assets"], 50)
        self.assertGreaterEqual(summary["data"]["raw_vix_rows_after_filters"], 100_000)
        self.assertEqual(summary["data"]["vix_settlement_source"], "vro_soq_exact:536")
        self.assertTrue(summary["data"]["vix_headline_eligible"])

        timing = {row["Diagnostic"]: row["Value"] for row in summary["timing_diagnostics"]}
        self.assertEqual(timing["Max train realization date"], "2020-12-31")
        self.assertEqual(timing["First test decision date"], "2020-12-31")
        self.assertIn("Raw daily close", timing["Expiry spot source"])

        self.assertTrue(all(row["Pass"] == "yes" for row in summary["trading_data_audit"]))
        trading_audit = {row["Check"]: row for row in summary["trading_data_audit"]}
        self.assertEqual(trading_audit["Exact listed-expiry close share"]["Value"], "1.000")
        self.assertEqual(trading_audit["Max expiry-to-payoff-date lag in days"]["Value"], "0")
        self.assertEqual(trading_audit["VIX settlement source"]["Pass"], "yes")
        self.assertEqual(trading_audit["VIX settlement source"]["Value"], "vro_soq_exact:536")
        self.assertEqual(trading_audit["Duplicate VIX date-symbol rows after dedupe"]["Value"], "0")
        self.assertEqual(trading_audit["Black-76 VX-forward Greek coverage"]["Value"], "1.000")
        self.assertGreaterEqual(len(summary["split_adjustments"]), 5)

        exposure = {row["Strategy"]: row for row in summary["exposure"]}
        self.assertGreater(exposure["Greek Markowitz + VIX"]["VIX option gross"], 0.10)
        self.assertLessEqual(exposure["Greek Markowitz + VIX"]["Short gross"], 0.251)
        self.assertLessEqual(abs(exposure["Beta/delta-neutral + VIX"]["Net delta"]), 0.15)
        self.assertLessEqual(abs(exposure["Beta/delta-neutral + VIX"]["Beta SPY proxy"]), 0.251)
        self.assertLessEqual(exposure["Beta/delta-neutral + VIX"]["Worst stress return"], 0.0)
        self.assertGreater(exposure["Greek Markowitz + VIX"]["Long premium paid"], 0.0)
        self.assertAlmostEqual(
            exposure["Greek Markowitz + VIX"]["Long premium paid"]
            - exposure["Greek Markowitz + VIX"]["Short premium sold"],
            exposure["Greek Markowitz + VIX"]["Net option premium"],
        )

        regression = {row["Strategy"]: row for row in summary["factor_regression"]}
        for col in ["Beta SPY", "Beta AAPL", "Beta NVDA", "Beta VX front", "Beta dVIX", "Beta dVVIX"]:
            self.assertIn(col, regression["Greek Markowitz + VIX"])
        self.assertGreater(regression["Greek Markowitz + VIX"]["Ann. alpha"], 0.0)
        self.assertGreaterEqual(regression["Greek Markowitz + VIX"]["N"], 55)

        attribution = {row["Strategy"]: row for row in summary["pnl_attribution"]}
        for col in ["Equity delta", "Equity gamma", "Equity vega", "VIX-forward delta", "VIX-forward gamma", "VIX-option vega", "Theta/carry", "VX roll", "Skew/tail", "Residual"]:
            self.assertIn(col, attribution["Greek Markowitz + VIX"])
        self.assertNotEqual(attribution["Greek Markowitz + VIX"]["VIX-option vega"], 0.0)

        self.assertEqual(len(summary["vix_regime_performance"]), 24)
        rolling = {row["Diagnostic"]: row["Value"] for row in summary["rolling_oos"]}
        self.assertEqual(rolling["Rolling 36M OOS months"], 20.0)
        self.assertGreater(rolling["Rolling 36M OOS Sharpe"], 0.0)

        leave_one = {row["Exclusion"]: row for row in summary["leave_one_out"]}
        for exclusion in ["No META", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]:
            self.assertIn(exclusion, leave_one)
        self.assertGreater(leave_one["No META/NVDA/TSLA"]["Sharpe"], 0.0)
        self.assertGreaterEqual(leave_one["No META/NVDA/TSLA"]["Option assets"], 30)

        self.assertEqual(len(summary["claim_strength"]), 4)
        claim_strength = {row["Strength"]: row for row in summary["claim_strength"]}
        self.assertIn("No claim", claim_strength)
        self.assertIn("production tradability", claim_strength["No claim"]["Claim"].lower())
        self.assertGreaterEqual(len(summary["claim_audit"]), 10)
        claim_audit = {row["Claim"]: row for row in summary["claim_audit"]}
        vix_claim = claim_audit["VIX option expiry P\\&L is exact listed settlement P\\&L"]
        self.assertEqual(vix_claim["Type"], "Generated empirical")
        self.assertEqual(vix_claim["Status"], "Supported")
        self.assertIn("exact", vix_claim["Evidence"].lower())
        self.assertTrue(all(row["Pass"] == "yes" for row in summary["figure_visibility"]))
        self.assertEqual(len(summary["figure_visibility"]), len(perf))


if __name__ == "__main__":
    unittest.main()
