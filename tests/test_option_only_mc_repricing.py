"""Tests for option-only Markowitz Monte Carlo repricing."""

from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.monte_carlo_repricing import (  # noqa: E402
    ONE_STEP_TENOR_YEARS,
    RepriceConfig,
    StateDimension,
    StateModel,
    black76_price,
    bs_price_vec,
    contract_static_params,
    reprice_contract_returns,
    repriced_strategy_paths,
    simulate_state_paths,
)
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics  # noqa: E402


def _dim(name: str, kind: str, underlying: str) -> StateDimension:
    return StateDimension(
        name=name,
        kind=kind,
        underlying=underlying,
        mu=0.0,
        omega=1e-6,
        alpha=0.0,
        beta=0.0,
        last_epsilon=0.0,
        last_sigma2=1e-6,
        unconditional_mean=0.0,
        unconditional_var=1e-6,
        n_obs=4,
        fallback=True,
        fallback_reason="test",
    )


class TestOptionOnlyMonteCarloRepricing(unittest.TestCase):
    def test_module_ast_parses(self):
        module = ROOT / "research/papers/option_only_markowitz/analysis/monte_carlo_repricing.py"
        ast.parse(module.read_text(encoding="utf-8"))

    def test_vendored_bs_price_pin_and_parity(self):
        s = np.array([100.0])
        k = np.array([100.0])
        t = np.array([0.25])
        r = np.array([0.02])
        q = np.array([0.0])
        sigma = np.array([0.20])

        call = float(bs_price_vec(s, k, t, r, q, sigma, np.array(["call"]))[0])
        put = float(bs_price_vec(s, k, t, r, q, sigma, np.array(["put"]))[0])

        self.assertAlmostEqual(call, 4.2322, places=4)
        self.assertAlmostEqual(call - put, 100.0 - 100.0 * math.exp(-0.02 * 0.25), places=10)

    def test_black76_forward_equals_bs_call(self):
        s = 100.0
        k = 100.0
        t = 0.25
        r = 0.02
        sigma = 0.20
        bs_call = float(bs_price_vec(np.array([s]), np.array([k]), np.array([t]), np.array([r]), 0.0, sigma, "C")[0])
        b76_call = black76_price(s * math.exp(r * t), k, t, r, sigma, "call")

        self.assertAlmostEqual(b76_call, bs_call, places=8)

    def test_reprice_kernel_uses_one_step_pricing_tenor(self):
        config = RepriceConfig(n_paths=1, horizon_months=1, rate=0.02, iv_floor=0.05, iv_cap=2.0, min_mark=0.25)
        states = {
            "spot": np.array([[[1.0], [1.1]]]),
            "iv": np.array([[[0.20], [0.20]]]),
            "vix": np.array([[20.0, 20.0]]),
            "underlyings": np.array(["U"], dtype=object),
        }
        params = pd.DataFrame(
            {
                "underlying": ["U"],
                "kind": ["call"],
                "asset_class": ["equity_option"],
                "log_moneyness": [0.0],
                "tenor_years": [1.0 / 365.0],
                "skew_ratio": [1.0],
                "anchor_spot": [100.0],
                "contract_iv": [0.20],
            },
            index=pd.Index(["U_call_atm"], name="asset_id"),
        )

        returns = reprice_contract_returns(states, params, config)
        premium = float(bs_price_vec(np.array([100.0]), np.array([100.0]), np.array([ONE_STEP_TENOR_YEARS]), 0.02, 0.0, 0.20, "call")[0])
        payoff = max(110.0 - 100.0, 0.0)
        expected = payoff / max(premium, 0.25) - 1.0

        self.assertAlmostEqual(premium, 2.3853, places=4)
        self.assertGreater(premium, 0.25)
        self.assertAlmostEqual(float(returns[0, 0, 0]), expected, places=12)

    def test_one_step_premium_payoff_consistency_driftless_world(self):
        rng = np.random.default_rng(20260625)
        n_paths = 8000
        sigma = 0.20
        growth = np.exp(
            -0.5 * sigma * sigma * ONE_STEP_TENOR_YEARS
            + sigma * math.sqrt(ONE_STEP_TENOR_YEARS) * rng.standard_normal(n_paths)
        )
        spot = np.empty((n_paths, 2, 1), dtype=float)
        spot[:, 0, 0] = 1.0
        spot[:, 1, 0] = growth
        states = {
            "spot": spot,
            "iv": np.full((n_paths, 2, 1), sigma, dtype=float),
            "vix": np.full((n_paths, 2), 20.0, dtype=float),
            "underlyings": np.array(["U"], dtype=object),
        }
        params = pd.DataFrame(
            {
                "underlying": ["U"],
                "kind": ["call"],
                "asset_class": ["equity_option"],
                "log_moneyness": [0.0],
                "tenor_years": [18.0 / 365.0],
                "pricing_tenor_years": [ONE_STEP_TENOR_YEARS],
                "skew_ratio": [1.0],
                "anchor_spot": [100.0],
                "contract_iv": [sigma],
            },
            index=pd.Index(["U_call_atm"], name="asset_id"),
        )

        returns = reprice_contract_returns(states, params, RepriceConfig(rate=0.0, min_mark=0.25))

        self.assertGreaterEqual(returns.size, 2000)
        self.assertLess(abs(float(np.nanmean(returns[:, :, 0]))), 0.15)

    def test_vix_contract_uses_vx_front_state_for_entry_forward(self):
        config = RepriceConfig(n_paths=1, horizon_months=1, rate=0.0, iv_floor=0.05, iv_cap=2.0, min_mark=0.01)

        def states(vx_entry: float) -> dict[str, np.ndarray]:
            return {
                "spot": np.array([[[1.0, vx_entry], [1.1, vx_entry]]], dtype=float),
                "iv": np.array([[[0.20, 0.20], [0.20, 0.20]]], dtype=float),
                "vix": np.array([[20.0, 30.0]], dtype=float),
                "underlyings": np.array(["U", "VX_FRONT"], dtype=object),
            }

        params = pd.DataFrame(
            {
                "underlying": ["U", "VX_FRONT"],
                "kind": ["call", "call"],
                "asset_class": ["equity_option", "vix_option"],
                "log_moneyness": [0.0, 0.0],
                "tenor_years": [18.0 / 365.0, 18.0 / 365.0],
                "pricing_tenor_years": [ONE_STEP_TENOR_YEARS, ONE_STEP_TENOR_YEARS],
                "skew_ratio": [1.0, 1.0],
                "anchor_spot": [100.0, 20.0],
                "contract_iv": [0.20, 0.20],
            },
            index=pd.Index(["U_call_atm", "VIX_call_atm"], name="asset_id"),
        )

        base = reprice_contract_returns(states(1.0), params, config)
        perturbed = reprice_contract_returns(states(2.0), params, config)

        self.assertAlmostEqual(float(base[0, 0, 0]), float(perturbed[0, 0, 0]), places=12)
        self.assertNotAlmostEqual(float(base[0, 0, 1]), float(perturbed[0, 0, 1]), places=6)
        self.assertNotAlmostEqual(
            black76_price(20.0, 20.0, ONE_STEP_TENOR_YEARS, 0.0, 0.20, "call"),
            black76_price(40.0, 40.0, ONE_STEP_TENOR_YEARS, 0.0, 0.20, "call"),
            places=6,
        )

    def test_contract_static_params_ignores_post_train_rows(self):
        reps = pd.DataFrame(
            [
                {
                    "snap_date": "2020-01-31",
                    "asset_id": "U_call_otm",
                    "underlying": "U",
                    "kind": "call",
                    "asset_class": "equity_option",
                    "strike": 110.0,
                    "spot": 100.0,
                    "tenor_days": 30,
                    "moneyness_bucket": "otm",
                    "iv_proxy": 0.30,
                },
                {
                    "snap_date": "2020-01-31",
                    "asset_id": "U_call_atm",
                    "underlying": "U",
                    "kind": "call",
                    "asset_class": "equity_option",
                    "strike": 100.0,
                    "spot": 100.0,
                    "tenor_days": 30,
                    "moneyness_bucket": "atm",
                    "iv_proxy": 0.20,
                },
                {
                    "snap_date": "2020-03-31",
                    "asset_id": "U_call_otm",
                    "underlying": "U",
                    "kind": "put",
                    "asset_class": "equity_option",
                    "strike": 999.0,
                    "spot": 10.0,
                    "tenor_days": 999,
                    "moneyness_bucket": "poison",
                    "iv_proxy": 9.0,
                },
            ]
        )

        params = contract_static_params(reps, pd.Timestamp("2020-02-29"))
        row = params.loc["U_call_otm"]

        self.assertEqual(row["kind"], "call")
        self.assertAlmostEqual(row["log_moneyness"], math.log(1.1))
        self.assertAlmostEqual(row["tenor_years"], 30.0 / 365.0)
        self.assertAlmostEqual(row["pricing_tenor_years"], ONE_STEP_TENOR_YEARS)
        self.assertAlmostEqual(row["skew_ratio"], 1.5)
        self.assertAlmostEqual(row["anchor_spot"], 100.0)

    def test_simulate_state_paths_determinism_bounds_and_joint_blocks(self):
        model = StateModel(
            underlyings=("U",),
            dimensions=(
                _dim("spot:U", "spot", "U"),
                _dim("iv:U", "iv", "U"),
                _dim("vix:VIX", "vix", "VIX"),
            ),
            Z=np.array(
                [
                    [0.1, 1.0, -0.2],
                    [-0.3, -3.0, 0.4],
                    [0.5, 5.0, -0.6],
                    [-0.7, -7.0, 0.8],
                ]
            ),
            z_index=pd.RangeIndex(4),
            initial_iv=pd.Series({"U": 0.20}),
            initial_spot=pd.Series({"U": 1.0}),
            initial_vix=20.0,
            vix_forward_spot_ratio=1.0,
            train_start=None,
            train_end=None,
        )
        config = RepriceConfig(n_paths=3, horizon_months=9, block_length=3, seed=123, iv_floor=0.05, iv_cap=0.50)

        first = simulate_state_paths(model, config)
        second = simulate_state_paths(model, config)

        for key in ["spot", "iv", "vix", "innovations", "innovation_row_index"]:
            self.assertTrue(np.array_equal(first[key], second[key]))
        self.assertTrue(np.isfinite(first["spot"]).all())
        self.assertTrue((first["spot"] > 0.0).all())
        self.assertTrue(((first["iv"] >= config.iv_floor) & (first["iv"] <= config.iv_cap)).all())
        self.assertTrue(np.allclose(first["innovations"][:, :, 1], 10.0 * first["innovations"][:, :, 0]))

    def test_repriced_strategy_paths_weighted_sum_and_coverage(self):
        contract_returns = np.array([[[0.10, -0.20], [0.05, 0.30]]])
        weights = {"S": pd.Series({"c0": 0.25, "c1": 0.75, "missing": 1.0})}
        out = repriced_strategy_paths(contract_returns, weights, pd.Index(["c0", "c1"]), RepriceConfig()).iloc[0]
        hand_returns = pd.Series([0.25 * 0.10 + 0.75 * -0.20, 0.25 * 0.05 + 0.75 * 0.30])
        stats = performance_metrics(hand_returns, 12)

        self.assertAlmostEqual(out["sharpe"], stats["sharpe"])
        self.assertAlmostEqual(out["sortino"], stats["sortino"])
        self.assertAlmostEqual(out["max_drawdown"], stats["max_drawdown"])
        self.assertAlmostEqual(out["ann_return"], stats["annualized_return"])
        self.assertAlmostEqual(out["terminal_wealth"], stats["terminal_wealth"])
        self.assertEqual(out["defaulted"], stats["defaulted"])
        self.assertAlmostEqual(out["weight_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
