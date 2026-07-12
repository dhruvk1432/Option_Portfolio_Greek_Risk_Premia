from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    estimate_greek_joint_moments,
    greek_exposure_frame,
)
from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import (
    build_optimization_cost_spec,
    integerize_r1_weights,
    paired_block_bootstrap_comparison,
    survival_diagnostics,
)
from research.papers.option_only_markowitz.analysis.publication_costs import ResearchCostConfig


def _inputs() -> tuple[OptionOnlySpec, pd.DataFrame, pd.DataFrame]:
    options = OptionOnlySpec(
        pd.DataFrame(
            {
                "underlying": ["AAA", "AAA"],
                "mark": [5.0, 4.0],
                "spot": [100.0, 100.0],
                "delta": [0.55, -0.40],
                "gamma": [0.020, 0.025],
                "vega": [18.0, 16.0],
                "theta": [-0.03, -0.02],
                "asset_class": ["equity_option", "equity_option"],
                "stress_scenario_crash": [-0.45, 0.25],
            },
            index=["call", "put"],
        )
    )
    rng = np.random.default_rng(20260711)
    dates = pd.date_range("2010-01-31", periods=96, freq="ME")
    spot = rng.normal(0.005, 0.04, len(dates))
    factors = pd.DataFrame(
        {
            "r_AAA": spot,
            "r2_AAA": spot**2 - float(np.mean(spot**2)),
            "dv_AAA": -0.35 * spot + rng.normal(0.0, 0.012, len(dates)),
        },
        index=dates,
    )
    B = greek_exposure_frame(options)
    # Residuals deliberately share factor risk, so Gamma is materially nonzero.
    residual = np.column_stack(
        [0.30 * spot + rng.normal(0.0, 0.03, len(dates)), -0.20 * spot + rng.normal(0.0, 0.025, len(dates))]
    )
    returns = pd.DataFrame(factors.to_numpy() @ B.T.to_numpy() + residual, index=dates, columns=B.index)
    return options, factors, returns


def _model() -> tuple[OptionOnlyMarkowitzModel, pd.DataFrame]:
    options, factors, returns = _inputs()
    B = greek_exposure_frame(options)
    moments = estimate_greek_joint_moments(returns, factors, B, regularize=True)
    shocks = FactorShockSpec(
        underlying_cov=pd.DataFrame([[0.02]], index=["AAA"], columns=["AAA"]),
        vol_cov=pd.DataFrame([[0.01]], index=["AAA"], columns=["AAA"]),
    )
    model = OptionOnlyMarkowitzModel(
        options,
        shocks,
        expected_returns=pd.Series([0.035, 0.015], index=options.frame.index),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.8),
        covariance_shrinkage=0.0,
        joint_moments=moments,
    )
    return model, returns


def _costs(model: OptionOnlyMarkowitzModel, call_cost: float = 0.002) -> OptimizationCostSpec:
    return OptimizationCostSpec(
        long_cost=pd.Series([call_cost, 0.002], index=model.contracts),
        short_cost=pd.Series([0.012, 0.012], index=model.contracts),
        short_margin=pd.Series([1.5, 1.5], index=model.contracts),
        assignment_short_allowed=pd.Series([False, True], index=model.contracts),
    )


def test_cross_terms_are_required_and_sample_identity_is_exact():
    options, factors, returns = _inputs()
    B_frame = greek_exposure_frame(options)
    moments = estimate_greek_joint_moments(returns, factors, B_frame, regularize=False)
    B = B_frame.to_numpy(float)
    omega = moments.factor_cov.to_numpy(float)
    gamma = moments.factor_residual_cov.to_numpy(float)
    residual = moments.residual_cov.to_numpy(float)
    old = B @ omega @ B.T + residual
    complete = old + B @ gamma + gamma.T @ B.T
    sample = returns.cov().to_numpy(float)

    assert np.max(np.abs(gamma)) > 1e-5
    assert not np.allclose(old, sample, atol=1e-6)
    np.testing.assert_allclose(complete, sample, atol=1e-10, rtol=1e-10)


def test_regularized_joint_covariance_is_labeled_finite_and_psd():
    options, factors, returns = _inputs()
    B = greek_exposure_frame(options)
    moments = estimate_greek_joint_moments(returns, factors, B, regularize=True)
    joint = moments.joint_covariance()

    assert moments.factor_names == list(B.columns)
    assert moments.contract_names == list(B.index)
    assert np.isfinite(joint.to_numpy()).all()
    assert np.allclose(joint, joint.T)
    assert np.linalg.eigvalsh(joint.to_numpy()).min() >= -1e-10


def test_joint_estimator_honors_training_cutoff():
    options, factors, returns = _inputs()
    B = greek_exposure_frame(options)
    cutoff = returns.index[60]
    baseline = estimate_greek_joint_moments(returns, factors, B, regularize=False, train_end=cutoff)
    changed = returns.copy()
    changed.loc[changed.index > cutoff] = 1e6
    after = estimate_greek_joint_moments(changed, factors, B, regularize=False, train_end=cutoff)
    np.testing.assert_allclose(baseline.joint_covariance(), after.joint_covariance())


@pytest.mark.skipif(pytest.importorskip("cvxpy") is None, reason="cvxpy unavailable")
def test_net_utility_costs_choose_scale_and_cash():
    model, scenarios = _model()
    config = NetUtilityConfig(cvar_loss_nav=0.50, stress_loss_nav=0.50)
    low_cost = model.solve_net_utility(scenarios, _costs(model), config, risk_aversion=1.0)
    high_cost = model.solve_net_utility(scenarios, _costs(model, call_cost=0.10), config, risk_aversion=1.0)
    high_risk_aversion = model.solve_net_utility(scenarios, _costs(model), config, risk_aversion=100.0)

    assert low_cost.status == "optimal"
    assert low_cost.gross_nav <= model.constraints.gross_nav + 1e-7
    assert high_cost.weights["call"] < low_cost.weights["call"]
    assert high_risk_aversion.volatility < low_cost.volatility
    assert high_risk_aversion.gross_nav < low_cost.gross_nav
    assert low_cost.weights["call"] >= -1e-8  # assignment-risk short is prohibited

    negative = OptionOnlyMarkowitzModel(
        model.options,
        model.shocks,
        expected_returns=pd.Series(-0.05, index=model.contracts),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.8, long_only=True),
        covariance_shrinkage=0.0,
        joint_moments=model.joint_moments,
    )
    cash = negative.solve_net_utility(scenarios, _costs(negative), config, risk_aversion=1.0)
    assert cash.gross_nav == pytest.approx(0.0, abs=1e-7)
    assert cash.objective_stats["cash_weight"] == pytest.approx(1.0, abs=1e-7)


def test_model_uses_complete_joint_covariance():
    model, _ = _model()
    B = model.B
    expected = (
        B @ model.factor_cov @ B.T
        + B @ model.factor_residual_cov
        + model.factor_residual_cov.T @ B.T
        + model.residual_cov
    )
    np.testing.assert_allclose(model.option_cov, expected, atol=1e-8)


def test_survival_gate_overrides_positive_arithmetic_performance():
    # Repeated gains and one option-style crash can have an attractive
    # arithmetic mean, but the absorbing wealth path must fail.
    returns = pd.Series([0.20] * 12 + [-1.20] + [0.20] * 12)
    result = survival_diagnostics(returns)
    assert returns.mean() > 0
    assert result["terminal_wealth"] == 0.0
    assert result["ruin_count"] == 1
    assert result["verdict"] == "fail_survival_gate"


def test_survival_gate_treats_validation_absorption_as_failure():
    result = survival_diagnostics(pd.Series([0.01, 0.02]), absorbed_validation_paths=1)
    assert result["terminal_wealth"] > 1.0
    assert result["verdict"] == "fail_survival_gate"


def test_net_utility_hard_tail_margin_and_liquidity_limits():
    model, scenarios = _model()
    costs = OptimizationCostSpec(
        long_cost=pd.Series(0.001, index=model.contracts),
        short_cost=pd.Series(0.001, index=model.contracts),
        short_margin=pd.Series(10.0, index=model.contracts),
        assignment_short_allowed=pd.Series(True, index=model.contracts),
    )
    caps = pd.Series(0.03, index=model.contracts)
    config = NetUtilityConfig(
        cvar_loss_nav=0.02,
        stress_loss_nav=0.02,
        short_margin_nav=0.05,
        collateral_nav=0.10,
    )
    result = model.solve_net_utility(
        scenarios,
        costs,
        config,
        per_contract_caps=caps,
        risk_aversion=1e-4,
    )
    stats = result.objective_stats
    assert (result.weights.abs() <= caps + 1e-7).all()
    assert stats["scenario_cvar_loss"] <= config.cvar_loss_nav + 1e-6
    assert stats["worst_stress_return"] >= -config.stress_loss_nav - 1e-6
    assert stats["short_margin_used"] <= config.short_margin_nav + 1e-6
    assert stats["collateral_used"] <= config.collateral_nav + 1e-6


def test_r1_cost_spec_matches_full_cost_convention():
    date = pd.Timestamp("2020-12-31")
    ledger = pd.DataFrame(
        {
            "asset_id": ["call"],
            "decision_date": [date],
            "return_date": [pd.Timestamp("2021-01-31")],
            "mark": [5.0],
            "relative_spread": [0.04],
            "holding_years": [21.0 / 365.0],
            "start_spot": [100.0],
            "strike": [100.0],
            "kind": ["call"],
            "asset_class": ["equity_option"],
            "borrow_rate_proxy": [0.03],
        }
    )
    config = ResearchCostConfig(fee_per_contract_per_side=0.75, slippage_bps_per_side=5.0)
    costs, _ = build_optimization_cost_spec(ledger, ["call"], date, config)
    expected_entry = 0.04 + 2.0 * 0.75 / (5.0 * 100.0) + 2.0 * 5.0 / 10_000.0
    expected_long = expected_entry + config.margin_funding_rate * 21.0 / 365.0
    assert costs.long_cost.loc["call"] == pytest.approx(expected_long)
    assert costs.short_cost.loc["call"] > costs.long_cost.loc["call"]


def test_integer_execution_is_feasible_or_fails_closed_to_cash():
    model, scenarios = _model()
    costs = _costs(model)
    caps = pd.Series(0.08, index=model.contracts)
    config = NetUtilityConfig(cvar_loss_nav=0.50, stress_loss_nav=0.50)
    result = model.solve_net_utility(scenarios, costs, config, per_contract_caps=caps, risk_aversion=5.0)
    integer, diagnostics = integerize_r1_weights(
        model,
        result.weights,
        model.frame["mark"],
        100_000.0,
        caps,
        scenarios,
        costs,
        config,
    )
    assert diagnostics["feasible"] is True
    assert integer.abs().sum() <= model.constraints.gross_nav + 1e-8
    if diagnostics["integer_repair_failed_to_cash"]:
        assert integer.abs().sum() == 0.0


def test_paired_block_bootstrap_reports_growth_and_tail_differences():
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    strategy = pd.Series([0.02, 0.01, -0.01] * 8, index=index)
    benchmark = pd.Series([0.005, 0.0, -0.02] * 8, index=index)
    result = paired_block_bootstrap_comparison(strategy, benchmark, n_boot=100, seed=7)
    assert result["observations"] == 24
    assert result["monthly_log_growth_difference"] > 0
    assert result["expected_shortfall_difference"] > 0
    assert result["log_growth_ci_lo"] <= result["log_growth_ci_hi"]
