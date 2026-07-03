"""Tests for the option-only Markowitz model and paper data contract."""

from __future__ import annotations

import json
import math
import sys
import unittest
import warnings
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

    def test_bsm_theta_matches_finite_differences_call_and_put(self):
        S, K, T, r, sigma = 101.0, 99.0, 0.4, 0.03, 0.24
        h_t = 1e-5
        for kind in ("call", "put"):
            greeks = bs_greeks(S, K, T, r, sigma, kind)
            # theta = dV/dt = -dV/dT, estimated by central difference in T.
            fd_theta = (
                bs_price(S, K, T - h_t, r, sigma, kind)
                - bs_price(S, K, T + h_t, r, sigma, kind)
            ) / (2 * h_t)
            self.assertAlmostEqual(greeks["theta"], fd_theta, places=5)

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


def make_single_option_model(
    frame: pd.DataFrame,
    spot_vol_cov: pd.DataFrame | None = None,
    covariance_shrinkage: float = 0.10,
) -> OptionOnlyMarkowitzModel:
    underlyings = ["AAA"]
    underlying_cov = pd.DataFrame([[0.04]], index=underlyings, columns=underlyings)
    vol_cov = pd.DataFrame([[0.003]], index=underlyings, columns=underlyings)
    return OptionOnlyMarkowitzModel(
        OptionOnlySpec(frame),
        FactorShockSpec(underlying_cov, vol_cov=vol_cov, spot_vol_cov=spot_vol_cov),
        expected_returns=pd.Series(0.02, index=frame.index),
        covariance_shrinkage=covariance_shrinkage,
    )


def single_call_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "underlying": ["AAA"],
            "mark": [5.0],
            "spot": [100.0],
            "delta": [0.5],
            "gamma": [0.02],
            "vega": [20.0],
            "theta": [-4.0],
        },
        index=["AAA_call"],
    )


def single_put_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "underlying": ["AAA"],
            "mark": [4.5],
            "spot": [100.0],
            "delta": [-0.45],
            "gamma": [0.022],
            "vega": [19.0],
            "theta": [-3.5],
        },
        index=["AAA_put"],
    )


def make_infeasible_long_only_model() -> OptionOnlyMarkowitzModel:
    """Audit P7 example: all-positive deltas, long-only, gross 1, delta cap 0.5."""

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
    underlying_cov = pd.DataFrame(
        [[0.040, 0.010, 0.006], [0.010, 0.050, 0.012], [0.006, 0.012, 0.060]],
        index=underlyings,
        columns=underlyings,
    )
    constraints = OptionMarkowitzConstraints(gross_nav=1.0, delta_abs=0.5, long_only=True)
    return OptionOnlyMarkowitzModel(
        OptionOnlySpec(frame),
        FactorShockSpec(underlying_cov),
        expected_returns=pd.Series([0.04, 0.03, 0.02], index=frame.index),
        constraints=constraints,
    )


class TestAuditFixes(unittest.TestCase):
    # ------------------------------------------------------------------ P1
    def test_missing_spot_column_raises(self):
        frame = single_call_frame().drop(columns=["spot"])
        with self.assertRaises(ValueError):
            OptionOnlySpec(frame).validate()
        with self.assertRaises(ValueError):
            make_single_option_model(frame)

    def test_nonfinite_or_nonpositive_spot_raises(self):
        for bad_spot in [np.nan, 0.0, -100.0, np.inf]:
            frame = single_call_frame()
            frame.loc["AAA_call", "spot"] = bad_spot
            with self.assertRaises(ValueError):
                OptionOnlySpec(frame).validate()

    # ------------------------------------------------------------------ P2
    def test_nan_mark_raises(self):
        frame = single_call_frame()
        frame.loc["AAA_call", "mark"] = np.nan
        with self.assertRaises(ValueError):
            OptionOnlySpec(frame).validate()

    # ------------------------------------------------------------------ P3
    def test_spot_vol_cross_covariance_direction_by_book_sign(self):
        # Cov(R, dsigma) < 0 (correlation -0.5 between spot and vol shocks).
        scov = pd.DataFrame(
            [[-0.5 * math.sqrt(0.04 * 0.003)]], index=["AAA"], columns=["AAA"]
        )
        # Long put: negative delta, positive vega -> negative spot-vol
        # covariance ADDS variance versus the block-diagonal default.
        put_block = make_single_option_model(single_put_frame()).option_cov[0, 0]
        put_cross = make_single_option_model(single_put_frame(), spot_vol_cov=scov).option_cov[0, 0]
        self.assertGreater(put_cross, put_block)
        # Long call: positive delta, positive vega -> the same cross block
        # REMOVES variance.
        call_block = make_single_option_model(single_call_frame()).option_cov[0, 0]
        call_cross = make_single_option_model(single_call_frame(), spot_vol_cov=scov).option_cov[0, 0]
        self.assertLess(call_cross, call_block)

    def test_spot_vol_cov_default_none_preserves_block_diagonal(self):
        model = make_single_option_model(single_call_frame())
        k = 1
        np.testing.assert_allclose(model.factor_cov[:k, 2 * k :], 0.0, atol=1e-12)
        np.testing.assert_allclose(model.factor_cov[2 * k :, :k], 0.0, atol=1e-12)

    def test_spot_vol_cov_missing_underlying_rejected(self):
        scov = pd.DataFrame([[0.001]], index=["ZZZ"], columns=["ZZZ"])
        with self.assertRaises(ValueError):
            make_single_option_model(single_call_frame(), spot_vol_cov=scov)

    # ------------------------------------------------------ covariance pin
    def test_hand_computed_single_option_covariance_pin(self):
        # S=100, C=5, delta=0.5, gamma=0.02, vega=20; var_R=0.04, var_sig=0.003.
        # B row = [delta*S/C, 0.5*gamma*S^2/C, vega/C] = [10, 20, 4].
        # Omega diag = [var_R, 2*var_R^2, var_sig] = [0.04, 0.0032, 0.003].
        # factor variance = 10^2*0.04 + 20^2*0.0032 + 4^2*0.003
        #                 = 4.0 + 1.28 + 0.048 = 5.328.
        # residual default adds 5% of factor variance; diagonal shrinkage is a
        # no-op on a 1x1 matrix, so option_cov[0,0] = 1.05 * 5.328 = 5.5944.
        model = make_single_option_model(single_call_frame(), covariance_shrinkage=0.10)
        self.assertAlmostEqual(float(model.option_cov[0, 0]), 5.5944, places=10)

    # -------------------------------------------------------- tangency pin
    def test_hand_computed_two_contract_tangency(self):
        underlyings = ["AAA"]
        frame = pd.concat([single_call_frame(), single_put_frame()])
        model = OptionOnlyMarkowitzModel(
            OptionOnlySpec(frame),
            FactorShockSpec(
                pd.DataFrame([[0.04]], index=underlyings, columns=underlyings),
                vol_cov=pd.DataFrame([[0.003]], index=underlyings, columns=underlyings),
            ),
            expected_returns=pd.Series([0.02, 0.03], index=frame.index),
        )
        # Override the model covariance with a hand-checkable matrix.
        model.option_cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        model.expected_returns = pd.Series([0.02, 0.03], index=model.contracts)
        # By hand: det = 0.04*0.09 - 0.01^2 = 0.0035,
        # Sigma^{-1} mu = (1/0.0035) * [0.09*0.02 - 0.01*0.03,
        #                               -0.01*0.02 + 0.04*0.03]
        #               = (1/0.0035) * [0.0015, 0.0010] = [3/7, 2/7],
        # L1-normalized -> [0.6, 0.4].
        weights = model.tangency_weights()
        np.testing.assert_allclose(weights.to_numpy(float), [0.6, 0.4], atol=1e-9)

    # ------------------------------------------------------------ optimality
    def test_solver_matches_closed_form_tangency_when_constraints_slack(self):
        model = make_option_only_model()  # only the gross-NAV budget binds
        result = model.solve_max_sharpe()
        w = model.tangency_weights().to_numpy(float)
        mu = model.expected_returns.to_numpy(float)
        tangency_sharpe = float(mu @ w) / math.sqrt(float(w @ model.option_cov @ w))
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.sharpe, tangency_sharpe, delta=1e-6)

    # ------------------------------------------------------------------ P7
    def test_infeasible_solve_is_flagged(self):
        model = make_infeasible_long_only_model()
        result = model.solve_max_sharpe()
        self.assertEqual(result.status, "infeasible")
        self.assertGreater(result.max_violation, 1e-5)

    def test_raise_on_infeasible_opt_in(self):
        model = make_infeasible_long_only_model()
        with self.assertRaises(ValueError):
            model.solve_max_sharpe(raise_on_infeasible=True)

    def test_feasible_solve_reports_max_violation_within_tolerance(self):
        model = make_option_only_model()
        result = model.solve_max_sharpe()
        self.assertIn(result.status, ("optimal", "feasible_suboptimal"))
        self.assertLessEqual(result.max_violation, 1e-5)

    # ------------------------------------------------------------------ P8
    def test_socp_matches_or_beats_slsqp_on_feasible_problem(self):
        constraints = OptionMarkowitzConstraints(
            gross_nav=1.0,
            net_nav_abs=1.0,
            short_nav_abs=0.20,
            per_contract_abs=0.45,
            underlying_gross={"AAA": 0.60, "BBB": 0.60, "CCC": 0.60},
            long_only=True,
        )
        model = make_option_only_model(constraints)
        slsqp = model.solve_max_sharpe()
        socp = model.solve_max_sharpe(method="cvxpy")
        self.assertTrue(socp.solver.startswith("cvxpy_socp"))
        self.assertIn(socp.status, ("optimal", "feasible_suboptimal"))
        self.assertGreaterEqual(socp.sharpe, slsqp.sharpe - 1e-6)
        self.assertLessEqual(socp.max_violation, 1e-5)
        self.assertLessEqual(model._max_constraint_violation(socp.weights.to_numpy(float)), 1e-5)

    def test_socp_reports_infeasibility_cleanly(self):
        model = make_infeasible_long_only_model()
        result = model.solve_max_sharpe(method="cvxpy")
        self.assertEqual(result.status, "infeasible")
        with self.assertRaises(ValueError):
            model.solve_max_sharpe(method="cvxpy", raise_on_infeasible=True)

    def test_socp_alias_and_unknown_method_rejected(self):
        model = make_option_only_model()
        result = model.solve_max_sharpe_socp()
        self.assertTrue(result.solver.startswith("cvxpy_socp"))
        with self.assertRaises(ValueError):
            model.solve_max_sharpe(method="bogus")

    # ----------------------------------------------------------------- P10
    def test_per_contract_abs_zero_never_fills_a_contract(self):
        model = make_option_only_model(
            OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.0)
        )
        result = model.solve_max_sharpe()
        if result.status != "infeasible":
            self.assertLessEqual(float(result.weights.abs().max()), 1e-12)

    # ----------------------------------------------------------------- P12
    def test_under_gross_deployment_is_not_a_violation(self):
        model = make_option_only_model()
        half = 0.5 * model.tangency_weights().to_numpy(float)  # gross 0.5 < 1.0
        self.assertEqual(model._max_constraint_violation(half), 0.0)
        over = 2.0 * model.tangency_weights().to_numpy(float)  # gross 2.0 > 1.0
        self.assertAlmostEqual(model._max_constraint_violation(over), 1.0, places=9)

    # ----------------------------------------------------------------- P13
    def test_return_series_fill_policy(self):
        model = make_option_only_model()
        dates = pd.date_range("2024-01-31", periods=3, freq="ME")
        full = pd.DataFrame(0.01, index=dates, columns=model.contracts)
        weights = model.equal_premium_weights()
        # Missing contract column: 'zero' keeps historical behavior, 'raise' errors.
        partial = full.drop(columns=[model.contracts[0]])
        series = model.portfolio_return_series(partial, weights)
        expected = model.portfolio_return_series(
            partial.reindex(columns=model.contracts).fillna(0.0), weights
        )
        pd.testing.assert_series_equal(series, expected)
        with self.assertRaises(ValueError):
            model.portfolio_return_series(partial, weights, fill_policy="raise")
        # NaN observation under 'raise' also errors; complete panel passes.
        holed = full.copy()
        holed.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.portfolio_return_series(holed, weights, fill_policy="raise")
        clean = model.portfolio_return_series(full, weights, fill_policy="raise")
        self.assertEqual(len(clean), len(dates))
        with self.assertRaises(ValueError):
            model.portfolio_return_series(full, weights, fill_policy="bogus")

    # -------------------------------------------------------------- P4/P5
    def test_residual_cov_missing_contracts_warns_without_changing_values(self):
        underlyings = ["AAA"]
        frame = pd.concat([single_call_frame(), single_put_frame()])
        ucov = pd.DataFrame([[0.04]], index=underlyings, columns=underlyings)
        vcov = pd.DataFrame([[0.003]], index=underlyings, columns=underlyings)
        resid = pd.DataFrame([[0.01]], index=["AAA_call"], columns=["AAA_call"])
        with self.assertWarns(UserWarning):
            warned = OptionOnlyMarkowitzModel(
                OptionOnlySpec(frame),
                FactorShockSpec(ucov, vol_cov=vcov),
                expected_returns=pd.Series(0.02, index=frame.index),
                residual_cov=resid,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            silent = OptionOnlyMarkowitzModel(
                OptionOnlySpec(frame),
                FactorShockSpec(ucov, vol_cov=vcov),
                expected_returns=pd.Series(0.02, index=frame.index),
                residual_cov=resid.reindex(index=frame.index, columns=frame.index).fillna(0.0),
            )
        np.testing.assert_allclose(warned.option_cov, silent.option_cov, atol=1e-14)


def make_sortino_two_contract_model() -> OptionOnlyMarkowitzModel:
    """Long-only 2-contract model with hand-set mu for the Sortino tests."""

    frame = pd.concat([single_call_frame(), single_put_frame()])
    underlyings = ["AAA"]
    return OptionOnlyMarkowitzModel(
        OptionOnlySpec(frame),
        FactorShockSpec(
            pd.DataFrame([[0.04]], index=underlyings, columns=underlyings),
            vol_cov=pd.DataFrame([[0.003]], index=underlyings, columns=underlyings),
        ),
        expected_returns=pd.Series([0.04, 0.03], index=frame.index),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, long_only=True),
    )


def sortino_training_scenarios() -> pd.DataFrame:
    """T=8 training scenarios with a joint-loss row so DD > 0 on the simplex."""

    return pd.DataFrame(
        {
            "AAA_call": [0.15, -0.10, 0.05, -0.06, 0.12, -0.04, -0.18, 0.08],
            "AAA_put": [-0.08, 0.12, -0.03, 0.10, -0.06, 0.05, -0.11, 0.02],
        },
        index=pd.date_range("2020-01-31", periods=8, freq="ME"),
    )


def make_sortino_four_contract_model(
    delta_abs: float | None = None,
) -> tuple[OptionOnlyMarkowitzModel, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA", "AAA", "BBB", "BBB"],
            "mark": [5.0, 4.5, 3.0, 2.8],
            "spot": [100.0, 100.0, 50.0, 50.0],
            "delta": [0.55, -0.45, 0.50, -0.42],
            "gamma": [0.020, 0.022, 0.035, 0.032],
            "vega": [20.0, 19.0, 12.0, 11.0],
            "theta": [-4.0, -3.5, -2.0, -1.8],
        },
        index=["AAA_call", "AAA_put", "BBB_call", "BBB_put"],
    )
    ucov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.05]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    model = OptionOnlyMarkowitzModel(
        OptionOnlySpec(frame),
        FactorShockSpec(ucov),
        expected_returns=pd.Series([0.03, 0.02, 0.025, 0.015], index=frame.index),
        constraints=OptionMarkowitzConstraints(
            gross_nav=1.0, short_nav_abs=0.30, per_contract_abs=0.50, delta_abs=delta_abs
        ),
    )
    rng = np.random.default_rng(42)
    common = rng.normal(0.0, 0.06, size=(60, 1))
    scenarios = pd.DataFrame(
        common @ np.array([[1.0, -0.8, 0.9, -0.7]]) + rng.normal(0.01, 0.05, size=(60, 4)),
        index=pd.date_range("2016-01-31", periods=60, freq="ME"),
        columns=frame.index,
    )
    return model, scenarios


def net_sortino_ratio(
    weights: np.ndarray,
    scenarios: np.ndarray,
    mu: np.ndarray,
    costs: np.ndarray,
    target: float,
) -> float:
    """Reference implementation of the paper objective m(q) / DD(q)."""

    net_mean = float(mu @ weights) - float(costs @ np.abs(weights))
    shortfall = np.maximum(target - scenarios @ weights, 0.0)
    downside = float(np.sqrt(np.mean(shortfall * shortfall)))
    if downside > 0:
        return net_mean / downside
    return math.inf if net_mean > 0 else math.nan


class TestMaxSortino(unittest.TestCase):
    def test_two_contract_solution_matches_dense_grid(self):
        model = make_sortino_two_contract_model()
        scenarios = sortino_training_scenarios()
        mu = model.expected_returns.to_numpy(float)
        R = scenarios[model.contracts].to_numpy(float)
        costs = np.zeros(2)
        # The objective is scale-invariant for tau=0, so the dense sweep of the
        # long-only gross=1 face {(w, 1-w) : w in [0, 1]} is an exhaustive grid.
        grid = np.linspace(0.0, 1.0, 4001)
        ratios = [net_sortino_ratio(np.array([w, 1.0 - w]), R, mu, costs, 0.0) for w in grid]
        best_idx = int(np.argmax(ratios))
        grid_best_ratio = ratios[best_idx]
        grid_best_w = np.array([grid[best_idx], 1.0 - grid[best_idx]])

        for method in ("cvxpy", "slsqp"):
            result = model.solve_max_sortino(scenarios, method=method)
            self.assertIn(result.status, ("optimal", "feasible_suboptimal"))
            self.assertGreaterEqual(result.sharpe, grid_best_ratio - 1e-4)
            np.testing.assert_allclose(result.weights.to_numpy(float), grid_best_w, atol=5e-3)
            stats = result.objective_stats
            self.assertAlmostEqual(
                stats["sortino_net"],
                net_sortino_ratio(result.weights.to_numpy(float), R, mu, costs, 0.0),
                places=10,
            )

    def test_entry_costs_shift_weights_and_lower_objective(self):
        model = make_sortino_two_contract_model()
        scenarios = sortino_training_scenarios()
        free = model.solve_max_sortino(scenarios)
        costed = model.solve_max_sortino(
            scenarios, entry_costs=pd.Series({"AAA_call": 0.02})
        )
        self.assertIn(free.status, ("optimal", "feasible_suboptimal"))
        self.assertIn(costed.status, ("optimal", "feasible_suboptimal"))
        # The costed contract's optimal weight strictly decreases ...
        self.assertLess(
            float(costed.weights["AAA_call"]), float(free.weights["AAA_call"]) - 1e-3
        )
        # ... and the attainable net objective strictly decreases.
        self.assertLess(
            costed.objective_stats["sortino_net"],
            free.objective_stats["sortino_net"] - 1e-3,
        )
        self.assertGreater(costed.objective_stats["entry_cost"], 0.0)
        self.assertAlmostEqual(
            costed.objective_stats["net_mean"],
            costed.objective_stats["gross_mean"] - costed.objective_stats["entry_cost"],
            places=12,
        )
        self.assertAlmostEqual(free.objective_stats["entry_cost"], 0.0, places=12)

    def test_cvxpy_and_slsqp_agree_on_four_contract_problem(self):
        model, scenarios = make_sortino_four_contract_model()
        socp = model.solve_max_sortino(scenarios, method="cvxpy")
        slsqp = model.solve_max_sortino(scenarios, method="slsqp")
        self.assertTrue(socp.solver.startswith("cvxpy_sortino_socp"))
        self.assertEqual(slsqp.solver, "scipy_slsqp_sortino_split")
        self.assertIn(socp.status, ("optimal", "feasible_suboptimal"))
        self.assertIn(slsqp.status, ("optimal", "feasible_suboptimal"))
        self.assertLessEqual(abs(socp.sharpe - slsqp.sharpe), 1e-4)
        self.assertLessEqual(socp.max_violation, 1e-5)
        self.assertLessEqual(slsqp.max_violation, 1e-5)

    def test_downside_free_degenerate_falls_back_to_net_mean(self):
        model = make_sortino_two_contract_model()
        all_positive = pd.DataFrame(
            {
                "AAA_call": [0.05, 0.02, 0.08, 0.01],
                "AAA_put": [0.03, 0.06, 0.01, 0.04],
            },
            index=pd.date_range("2020-01-31", periods=4, freq="ME"),
        )
        for method in ("cvxpy", "slsqp"):
            result = model.solve_max_sortino(all_positive, method=method)
            self.assertEqual(result.status, "optimal")
            self.assertEqual(result.solver, "linprog_net_mean_downside_free")
            stats = result.objective_stats
            self.assertTrue(stats["degenerate_downside_free"])
            self.assertEqual(stats["downside_deviation"], 0.0)
            self.assertTrue(math.isinf(stats["sortino_net"]))
            self.assertTrue(math.isinf(result.sharpe))
            # The documented fallback maximizes the net mean over the budgets:
            # with mu = (0.04, 0.03), long-only, gross = 1 that is all-in mu_1.
            self.assertAlmostEqual(stats["net_mean"], 0.04, places=8)
            np.testing.assert_allclose(result.weights.to_numpy(float), [1.0, 0.0], atol=1e-8)

    def test_delta_budget_compliance_and_infeasible_status(self):
        unconstrained, scenarios = make_sortino_four_contract_model()
        capped, _ = make_sortino_four_contract_model(delta_abs=1.0)
        free = unconstrained.solve_max_sortino(scenarios)
        result = capped.solve_max_sortino(scenarios)
        delta_free = float(
            unconstrained.greeks["delta_nav"].to_numpy(float) @ free.weights.to_numpy(float)
        )
        delta_capped = float(
            capped.greeks["delta_nav"].to_numpy(float) @ result.weights.to_numpy(float)
        )
        self.assertGreater(abs(delta_free), 1.0)  # the budget genuinely binds
        self.assertIn(result.status, ("optimal", "feasible_suboptimal"))
        self.assertLessEqual(abs(delta_capped), 1.0 + 2e-5)
        self.assertLessEqual(result.max_violation, 1e-5)

        infeasible_model = make_infeasible_long_only_model()
        rng = np.random.default_rng(7)
        infeasible_scen = pd.DataFrame(
            rng.normal(0.0, 0.08, size=(24, 3)),
            index=pd.date_range("2019-01-31", periods=24, freq="ME"),
            columns=infeasible_model.contracts,
        )
        for method in ("cvxpy", "slsqp"):
            bad = infeasible_model.solve_max_sortino(infeasible_scen, method=method)
            self.assertEqual(bad.status, "infeasible")
            with self.assertRaises(ValueError):
                infeasible_model.solve_max_sortino(
                    infeasible_scen, method=method, raise_on_infeasible=True
                )

    def test_nonzero_target_lowers_ratio_for_same_weights(self):
        model = make_sortino_two_contract_model()
        scenarios = sortino_training_scenarios()
        mu = model.expected_returns.to_numpy(float)
        R = scenarios[model.contracts].to_numpy(float)
        costs = np.zeros(2)
        result = model.solve_max_sortino(scenarios, target=0.01)
        self.assertIn(result.status, ("optimal", "feasible_suboptimal"))
        q = result.weights.to_numpy(float)
        stats = result.objective_stats
        self.assertAlmostEqual(stats["target"], 0.01, places=12)
        self.assertAlmostEqual(
            stats["sortino_net"], net_sortino_ratio(q, R, mu, costs, 0.01), places=10
        )
        self.assertEqual(result.sharpe, stats["sortino_net"])
        # tau = 0.01 raises the shortfall of every scenario, so for the SAME q
        # the net Sortino at tau = 0.01 is strictly below the tau = 0 ratio.
        self.assertLess(stats["sortino_net"], net_sortino_ratio(q, R, mu, costs, 0.0) - 1e-6)

    def test_input_validation_and_missing_data_policy(self):
        model = make_sortino_two_contract_model()
        scenarios = sortino_training_scenarios()
        # Entirely missing contract (absent column or all-NaN) raises.
        with self.assertRaises(ValueError):
            model.solve_max_sortino(scenarios.drop(columns=["AAA_put"]))
        all_nan = scenarios.copy()
        all_nan["AAA_put"] = np.nan
        with self.assertRaises(ValueError):
            model.solve_max_sortino(all_nan)
        # Negative or non-finite entry costs raise.
        with self.assertRaises(ValueError):
            model.solve_max_sortino(scenarios, entry_costs=pd.Series({"AAA_call": -0.01}))
        with self.assertRaises(ValueError):
            model.solve_max_sortino(scenarios, entry_costs=pd.Series({"AAA_call": np.inf}))
        with self.assertRaises(ValueError):
            model.solve_max_sortino(scenarios, method="bogus")
        # All-NaN rows are dropped; sporadic NaN cells are treated as zero.
        holed = scenarios.copy()
        holed.iloc[3, :] = np.nan
        holed.iloc[0, 1] = np.nan
        filled = scenarios.drop(index=scenarios.index[3]).copy()
        filled.iloc[0, 1] = 0.0
        res_holed = model.solve_max_sortino(holed)
        res_filled = model.solve_max_sortino(filled)
        self.assertEqual(res_holed.objective_stats["n_scenarios"], len(scenarios) - 1)
        self.assertAlmostEqual(
            res_holed.objective_stats["sortino_net"],
            res_filled.objective_stats["sortino_net"],
            places=8,
        )
        # The appended diagnostics field defaults to None on legacy solvers.
        self.assertIsNone(model.solve_max_sharpe().objective_stats)


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
            "option_only_portfolio_optimization_dhruv_kohli.pdf",
            "option_only_portfolio_optimization_dhruv_kohli.tex",
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
        # places=6 (not 9): SLSQP produces ~1e-8 drift across numpy/pandas
        # versions; the pin still detects any economically meaningful change.
        self.assertAlmostEqual(perf["Equity-option Greek Markowitz"]["Sharpe"], 0.8421194565895301, places=6)
        self.assertAlmostEqual(perf["Greek Markowitz + VIX"]["Sharpe"], 1.3743885779509968, places=6)
        self.assertGreater(perf["Greek Markowitz + VIX"]["Sharpe"], perf["Equity-option Greek Markowitz"]["Sharpe"])
        self.assertGreater(perf["Greek Markowitz + VIX"]["Sharpe"], perf["Delta-matched equities"]["Sharpe"])
        self.assertLess(perf["VIX hedge sleeve"]["Sharpe"], 0.0)
        for metric in ["Sortino", "Calmar", "Omega", "Info. ratio"]:
            self.assertIn(metric, perf["Greek Markowitz + VIX"])
        self.assertAlmostEqual(summary["random_feasible"]["p95_sharpe"], 1.1449656633511185, places=6)
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

        # 9 strategies (incl. the cost-aware Sortino variant) x 3 VIX regimes.
        self.assertEqual(len(summary["vix_regime_performance"]), 27)
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
