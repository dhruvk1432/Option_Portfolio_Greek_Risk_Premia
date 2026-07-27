from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from option_portfolio.model import (
    FactorShockSpec,
    GreekJointMomentSpec,
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionConstraints,
    OptionMarkowitzModel,
    OptionSpec,
    empirical_cvar_loss,
    estimate_greek_joint_moments,
    greek_exposure_frame,
    nearest_psd,
    risk_exposure_frame,
)


@pytest.fixture
def inputs() -> tuple[OptionSpec, FactorShockSpec, pd.Series]:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA", "AAA", "BBB"],
            "mark": [5.0, 4.0, 3.0],
            "spot": [100.0, 100.0, 50.0],
            "delta": [0.55, -0.40, 0.30],
            "gamma": [0.020, 0.025, 0.015],
            "vega": [18.0, 16.0, 10.0],
            "theta": [-0.03, -0.02, -0.01],
        },
        index=["call", "put", "wing"],
    )
    covariance = pd.DataFrame(
        [[0.02, 0.004], [0.004, 0.015]],
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )
    shocks = FactorShockSpec(
        underlying_cov=covariance,
        vol_cov=pd.DataFrame(np.eye(2) * 0.01, index=covariance.index, columns=covariance.columns),
    )
    return OptionSpec(frame), shocks, pd.Series([0.04, 0.02, 0.01], index=frame.index)


def test_public_state_dataclasses_are_frozen(inputs) -> None:
    options, _, _ = inputs
    constraints = OptionConstraints()

    with pytest.raises(FrozenInstanceError):
        constraints.gross_nav = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        options.frame = options.frame.copy()  # type: ignore[misc]


def test_complete_covariance_includes_both_cross_terms(inputs) -> None:
    options, shocks, means = inputs
    exposure = greek_exposure_frame(options)
    factors = list(exposure.columns)
    contracts = list(exposure.index)
    omega = pd.DataFrame(np.eye(len(factors)) * 0.01, index=factors, columns=factors)
    gamma = pd.DataFrame(0.001, index=factors, columns=contracts)
    residual = pd.DataFrame(np.eye(len(contracts)) * 0.02, index=contracts, columns=contracts)
    moments = GreekJointMomentSpec(omega, gamma, residual, n_obs=100)
    model = OptionMarkowitzModel(options, shocks, means, joint_moments=moments)
    b = exposure.to_numpy(float)
    expected = b @ omega.to_numpy() @ b.T + b @ gamma.to_numpy()
    expected += gamma.to_numpy().T @ b.T + residual.to_numpy()

    np.testing.assert_allclose(model.option_covariance, expected, atol=1e-8)


def test_unregularized_joint_estimator_reconstructs_sample_covariance() -> None:
    factors = pd.DataFrame(
        {
            "f1": [-0.04, -0.01, 0.02, 0.05, 0.01, -0.03],
            "f2": [0.03, -0.02, 0.01, -0.01, 0.04, -0.05],
        }
    )
    loadings = pd.DataFrame(
        [[1.2, -0.4], [0.3, 0.8]],
        index=["a", "b"],
        columns=factors.columns,
    )
    residuals = pd.DataFrame(
        {
            "a": 0.25 * factors["f1"] + np.array([0.01, -0.01, 0.00, 0.02, -0.02, 0.00]),
            "b": -0.20 * factors["f2"] + np.array([0.00, 0.01, -0.01, 0.00, 0.02, -0.02]),
        }
    )
    option_returns = factors.to_numpy() @ loadings.T.to_numpy() + residuals

    moments = estimate_greek_joint_moments(
        option_returns,
        factors,
        loadings,
        regularize=False,
    )
    b = loadings.to_numpy()
    reconstructed = (
        b @ moments.factor_cov.to_numpy() @ b.T
        + b @ moments.factor_residual_cov.to_numpy()
        + moments.factor_residual_cov.to_numpy().T @ b.T
        + moments.residual_cov.to_numpy()
    )

    np.testing.assert_allclose(reconstructed, option_returns.cov().to_numpy(), atol=1e-12)
    assert not np.allclose(moments.factor_residual_cov.to_numpy(), 0.0)


def test_joint_model_preserves_singular_residual_until_final_floor(inputs) -> None:
    options, shocks, means = inputs
    exposure = greek_exposure_frame(options)
    factors = list(exposure.columns)
    contracts = list(exposure.index)
    factor_covariance = pd.DataFrame(
        np.eye(len(factors)) * 0.01,
        index=factors,
        columns=factors,
    )
    residual_covariance = pd.DataFrame(
        np.zeros((len(contracts), len(contracts))),
        index=contracts,
        columns=contracts,
    )
    moments = GreekJointMomentSpec(
        factor_cov=factor_covariance,
        factor_residual_cov=pd.DataFrame(
            0.0,
            index=factors,
            columns=contracts,
        ),
        residual_cov=residual_covariance,
        n_obs=100,
    )

    model = OptionMarkowitzModel(options, shocks, means, joint_moments=moments)
    expected = exposure.to_numpy() @ factor_covariance.to_numpy() @ exposure.to_numpy().T

    np.testing.assert_array_equal(model.residual_covariance, residual_covariance)
    np.testing.assert_allclose(
        model.option_covariance,
        nearest_psd(expected),
        rtol=0.0,
        atol=1e-14,
    )


def test_joint_moment_validation_rejects_indefinite_cross_block() -> None:
    labels = ["factor"]
    contracts = ["contract"]
    moments = GreekJointMomentSpec(
        factor_cov=pd.DataFrame([[1.0]], index=labels, columns=labels),
        factor_residual_cov=pd.DataFrame([[2.0]], index=labels, columns=contracts),
        residual_cov=pd.DataFrame([[1.0]], index=contracts, columns=contracts),
        n_obs=10,
    )

    with pytest.raises(ValueError, match="joint"):
        moments.validate(labels, contracts)


def test_empirical_cvar_matches_fractional_tail_epigraph() -> None:
    losses = np.array([1.0, 2.0, 3.0])

    assert empirical_cvar_loss(losses, alpha=0.5) == pytest.approx(8.0 / 3.0)


def test_per_contract_caps_live_in_constraints(inputs) -> None:
    options, shocks, means = inputs
    caps = pd.Series([0.05, 0.50, 0.50], index=options.frame.index)
    model = OptionMarkowitzModel(
        options,
        shocks,
        means,
        constraints=OptionConstraints(
            gross_nav=0.60,
            per_contract_abs=0.50,
            per_contract_caps=caps,
        ),
    )
    result = model.solve_max_sharpe()

    assert result.status == "optimal"
    assert result.weights.abs().le(caps + 1e-6).all()
    assert abs(result.weights["call"]) <= 0.05 + 1e-6


def test_caps_must_cover_every_contract(inputs) -> None:
    options, shocks, means = inputs

    with pytest.raises(ValueError, match="cover every model contract"):
        OptionMarkowitzModel(
            options,
            shocks,
            means,
            constraints=OptionConstraints(
                per_contract_caps=pd.Series(0.1, index=options.frame.index[:-1])
            ),
        )


def test_option_input_validation_rejects_bad_marks(inputs) -> None:
    options, shocks, means = inputs
    bad = options.frame.copy()
    bad.loc["call", "mark"] = 0.0

    with pytest.raises(ValueError, match="mark"):
        OptionMarkowitzModel(OptionSpec(bad), shocks, means)


def test_model_rejects_nonfinite_expected_returns(inputs) -> None:
    options, shocks, means = inputs
    means.iloc[0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        OptionMarkowitzModel(options, shocks, means)


def _single_contract_model(
    *,
    expected_return: float,
    constraints: OptionConstraints | None = None,
    underlying: str = "AAA",
    extra_columns: dict[str, list[float]] | None = None,
) -> OptionMarkowitzModel:
    frame = pd.DataFrame(
        {
            "underlying": [underlying],
            "mark": [10.0],
            "spot": [100.0],
            "delta": [0.5],
            "gamma": [0.01],
            "vega": [5.0],
            "theta": [-0.01],
            "stress_scenario_reference": [0.0],
            **(extra_columns or {}),
        },
        index=["contract"],
    )
    covariance = pd.DataFrame([[1e-8]], index=[underlying], columns=[underlying])
    return OptionMarkowitzModel(
        OptionSpec(frame),
        FactorShockSpec(underlying_cov=covariance),
        pd.Series([expected_return], index=frame.index),
        constraints=constraints,
    )


def _zero_costs(*, short_allowed: bool = True) -> OptimizationCostSpec:
    index = ["contract"]
    return OptimizationCostSpec(
        long_cost=pd.Series(0.0, index=index),
        short_cost=pd.Series(0.0, index=index),
        short_margin=pd.Series(0.0, index=index),
        short_allowed=pd.Series(short_allowed, index=index),
    )


def test_infeasible_optimizer_result_reports_sharpe_as_unavailable() -> None:
    model = _single_contract_model(
        expected_return=0.10,
        constraints=OptionConstraints(gross_nav=1.0, per_contract_abs=0.10),
    )

    result = model.solve_max_sharpe()

    assert result.status == "infeasible"
    assert np.isnan(result.sharpe)


def test_net_utility_enforces_assignment_without_breaking_dcp() -> None:
    model = _single_contract_model(expected_return=-0.10)
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})

    result = model.solve_net_utility(scenarios, _zero_costs(short_allowed=False))

    assert result.status == "optimal"
    assert result.weights["contract"] >= -1e-8


def test_short_allowed_rejects_string_booleans() -> None:
    costs = OptimizationCostSpec(
        long_cost=pd.Series([0.0], index=["contract"]),
        short_cost=pd.Series([0.0], index=["contract"]),
        short_margin=pd.Series([0.0], index=["contract"]),
        short_allowed=pd.Series(["False"], index=["contract"]),
    )

    with pytest.raises(ValueError, match="boolean"):
        costs.aligned(["contract"])


def test_cost_inputs_must_be_finite() -> None:
    model = _single_contract_model(expected_return=0.10)
    costs = _zero_costs()
    invalid = OptimizationCostSpec(
        long_cost=pd.Series(np.inf, index=["contract"]),
        short_cost=costs.short_cost,
        short_margin=costs.short_margin,
        short_allowed=costs.short_allowed,
    )

    with pytest.raises(ValueError, match="finite"):
        model.solve_net_utility(
            pd.DataFrame({"contract": [0.0, 0.0, 0.0]}),
            invalid,
        )


def test_active_net_utility_stress_policy_requires_scenario_inputs(inputs) -> None:
    options, shocks, means = inputs
    model = OptionMarkowitzModel(options, shocks, means)
    labels = options.frame.index
    costs = OptimizationCostSpec(
        long_cost=pd.Series(0.0, index=labels),
        short_cost=pd.Series(0.0, index=labels),
        short_margin=pd.Series(0.0, index=labels),
        short_allowed=pd.Series(True, index=labels),
    )
    scenarios = pd.DataFrame(0.0, index=range(3), columns=labels)

    with pytest.raises(ValueError, match="active stress-loss limit"):
        model.solve_net_utility(scenarios, costs)


def test_net_utility_enforces_configured_stress_limit() -> None:
    model = _single_contract_model(
        expected_return=0.10,
        extra_columns={"stress_scenario_crash": [-1.0]},
    )
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})

    result = model.solve_net_utility(
        scenarios,
        _zero_costs(),
        NetUtilityConfig(stress_loss_nav=1e-6),
    )

    assert result.status == "optimal"
    assert result.weights["contract"] <= 1e-5


def test_beta_and_vix_vega_limits_are_enforced() -> None:
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})
    beta_model = _single_contract_model(
        expected_return=0.10,
        constraints=OptionConstraints(beta_spy_abs=0.0),
        extra_columns={"underlying_beta_spy": [1.0]},
    )
    vix_model = _single_contract_model(
        expected_return=0.10,
        constraints=OptionConstraints(vix_vega_abs=0.0),
        underlying="VIX",
    )

    beta_result = beta_model.solve_net_utility(scenarios, _zero_costs())
    vix_result = vix_model.solve_net_utility(scenarios, _zero_costs())

    assert beta_result.weights["contract"] == pytest.approx(0.0, abs=1e-7)
    assert vix_result.weights["contract"] == pytest.approx(0.0, abs=1e-7)


def test_net_utility_selects_risk_aversion_for_volatility_ceiling() -> None:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA"],
            "mark": [10.0],
            "spot": [100.0],
            "delta": [0.5],
            "gamma": [0.01],
            "vega": [5.0],
            "theta": [-0.01],
            "stress_scenario_reference": [0.0],
        },
        index=["contract"],
    )
    covariance = pd.DataFrame([[0.01]], index=["AAA"], columns=["AAA"])
    model = OptionMarkowitzModel(
        OptionSpec(frame),
        FactorShockSpec(underlying_cov=covariance),
        pd.Series([0.10], index=frame.index),
    )
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})
    config = NetUtilityConfig(annual_volatility_ceiling=0.15)

    result = model.solve_net_utility(scenarios, _zero_costs(), config)

    assert result.status == "optimal"
    assert result.risk_aversion is not None
    assert result.risk_aversion > config.lambda_floor
    assert result.volatility * np.sqrt(config.periods_per_year) <= 0.15 + 1e-5


def test_log_bisection_selects_smallest_passing_upper_bracket() -> None:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA"],
            "mark": [10.0],
            "spot": [100.0],
            "delta": [0.0],
            "gamma": [0.0],
            "vega": [0.0],
            "theta": [0.0],
            "stress_scenario_reference": [0.0],
        },
        index=["contract"],
    )
    factor_covariance = pd.DataFrame([[0.01]], index=["AAA"], columns=["AAA"])
    residual_covariance = pd.DataFrame(
        [[0.04]],
        index=frame.index,
        columns=frame.index,
    )
    model = OptionMarkowitzModel(
        OptionSpec(frame),
        FactorShockSpec(underlying_cov=factor_covariance),
        pd.Series([0.02], index=frame.index),
        residual_cov=residual_covariance,
        covariance_shrinkage=0.0,
    )
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})
    config = NetUtilityConfig(
        annual_volatility_ceiling=0.12,
        lambda_floor=1.0,
        lambda_ceiling=16.0,
        bisection_steps=4,
    )

    result = model.solve_net_utility(scenarios, _zero_costs(), config)

    assert result.risk_aversion == pytest.approx(2**1.75)
    assert result.volatility * np.sqrt(12.0) <= 0.12 + 1e-5
    lower_lambda = 2**1.5
    lower_weight = 0.02 / (lower_lambda * 0.04)
    assert 0.2 * lower_weight * np.sqrt(12.0) > 0.12


def test_missing_scenario_contract_is_rejected() -> None:
    model = _single_contract_model(expected_return=0.10)
    scenarios = pd.DataFrame({"other": [0.0, 0.0, 0.0]})

    with pytest.raises(ValueError, match="no observations"):
        model.solve_net_utility(scenarios, _zero_costs())


def test_unattainable_lambda_ceiling_fails_explicitly() -> None:
    model = _single_contract_model(expected_return=0.10)
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})
    config = NetUtilityConfig(
        annual_volatility_ceiling=1e-12,
        lambda_floor=1.0,
        lambda_ceiling=1.0,
        bisection_steps=1,
    )

    with pytest.raises(RuntimeError, match="lambda ceiling"):
        model.solve_net_utility(scenarios, _zero_costs(), config)


@pytest.mark.parametrize(
    ("model_limit", "policy_limit"),
    [(0.05, 0.20), (0.20, 0.05)],
)
def test_model_and_policy_use_the_tighter_stress_limit(
    model_limit: float,
    policy_limit: float,
) -> None:
    model = _single_contract_model(
        expected_return=0.10,
        constraints=OptionConstraints(stress_loss_abs=model_limit),
        extra_columns={"stress_scenario_crash": [-1.0]},
    )
    scenarios = pd.DataFrame({"contract": [0.0, 0.0, 0.0]})

    result = model.solve_net_utility(
        scenarios,
        _zero_costs(),
        NetUtilityConfig(stress_loss_nav=policy_limit),
    )

    assert result.weights["contract"] <= min(model_limit, policy_limit) + 1e-5


def test_risk_gamma_is_twice_centered_gamma_factor_loading() -> None:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA"],
            "mark": [10.0],
            "spot": [100.0],
            "delta": [0.0],
            "gamma": [0.01],
            "vega": [0.0],
            "theta": [0.0],
        },
        index=["contract"],
    )
    options = OptionSpec(frame)

    factor_gamma = greek_exposure_frame(options).loc["contract", "r2_AAA"]
    risk_gamma = risk_exposure_frame(options).loc["contract", "gamma_nav"]

    assert factor_gamma == pytest.approx(5.0)
    assert risk_gamma == pytest.approx(10.0)
