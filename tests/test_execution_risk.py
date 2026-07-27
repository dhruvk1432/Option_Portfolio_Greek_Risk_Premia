from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import option_portfolio.risk_controls as risk_controls
from option_portfolio.execution import ExecutionScenarioResult, execution_scenarios
from option_portfolio.model import (
    FactorShockSpec,
    OptimizationCostSpec,
    OptionConstraints,
    OptionMarkowitzModel,
    OptionSpec,
)
from option_portfolio.risk_controls import (
    R1_POLICY,
    R11_POLICY,
    R11Policy,
    integerize_or_cash,
    r1_constraints,
    solve_r11_net_utility,
)


def test_touch_prices_are_labeled_as_sensitivity_not_fills() -> None:
    result = execution_scenarios(
        gross_return=0.10,
        modeled_cost=0.01,
        midpoint_cost=0.02,
        touch_cost=0.03,
        worst_cost=0.04,
        entry_coverage=0.941,
        roundtrip_coverage=0.255,
    )

    assert result.touch_price_return == pytest.approx(0.07)
    assert result.entry_coverage == pytest.approx(0.941)
    assert result.roundtrip_coverage == pytest.approx(0.255)
    assert "fill" not in result.evidence_label.lower()
    with pytest.raises(FrozenInstanceError):
        result.evidence_label = "realized fill"  # type: ignore[misc]


def test_execution_scenario_requires_both_coverage_rates() -> None:
    with pytest.raises(ValueError, match="coverage"):
        ExecutionScenarioResult(
            modeled_return=0.1,
            midpoint_return=0.1,
            touch_price_return=0.1,
            worst_case_return=0.1,
            entry_coverage=0.9,
            roundtrip_coverage=None,
        )


def test_execution_scenario_rejects_nonfinite_returns() -> None:
    with pytest.raises(ValueError, match="finite"):
        execution_scenarios(
            gross_return=np.nan,
            modeled_cost=0.01,
            midpoint_cost=0.02,
            touch_cost=0.03,
            worst_cost=0.04,
            entry_coverage=0.941,
            roundtrip_coverage=0.255,
        )


def test_r1_and_r11_volatility_ceilings_are_frozen() -> None:
    assert R1_POLICY.annual_volatility_ceiling == 0.15
    assert R11_POLICY.annual_volatility_ceiling == 0.25


def test_r1_constraint_factory_freezes_every_policy_limit() -> None:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA", "VX_FRONT"],
            "mark": [10.0, 20.0],
            "spot": [100.0, 20.0],
            "delta": [0.1, 0.1],
            "gamma": [0.0, 0.0],
            "vega": [0.0, 0.0],
            "theta": [0.0, 0.0],
            "underlying_beta_spy": [1.0, 0.0],
            "stress_scenario_crash": [-2.0, 0.0],
        },
        index=["equity_call", "vix_call"],
    )
    options = OptionSpec(frame)
    constraints = r1_constraints(
        options,
        pd.Series({"equity_call": 0.30, "vix_call": 0.04}),
    )

    assert constraints.gross_nav == 1.0
    assert constraints.net_nav_abs == 1.0
    assert constraints.short_nav_abs == 0.25
    assert constraints.per_contract_abs == 0.18
    assert constraints.underlying_gross == {"AAA": 0.35, "VX_FRONT": 0.20}
    assert constraints.beta_spy_abs == 3.0
    assert constraints.vix_vega_abs == 8.0
    assert constraints.stress_loss_abs == 0.20
    covariance = pd.DataFrame(
        np.eye(2) * 1e-8,
        index=["AAA", "VX_FRONT"],
        columns=["AAA", "VX_FRONT"],
    )
    model = OptionMarkowitzModel(
        options,
        FactorShockSpec(underlying_cov=covariance),
        pd.Series(0.01, index=frame.index),
        constraints=constraints,
    )
    np.testing.assert_allclose(model._caps, [0.18, 0.04])
    assert model._max_constraint_violation(np.array([0.18, 0.0])) > 0.0


def test_r1_constraint_factory_requires_exact_liquidity_coverage() -> None:
    model = _model()

    with pytest.raises(ValueError, match="exactly"):
        r1_constraints(
            model.options,
            pd.Series({"call": 0.10}),
        )


def test_frozen_stress_limit_requires_scenario_inputs() -> None:
    model = _model()
    frame = model.options.frame.drop(
        columns=["stress_scenario_reference"]
    ).copy()
    frame["underlying_beta_spy"] = 1.0
    options = OptionSpec(frame)
    constraints = r1_constraints(
        options,
        pd.Series(0.10, index=model.contracts),
    )

    with pytest.raises(ValueError, match="stress_scenario"):
        OptionMarkowitzModel(
            options,
            model.shocks,
            model.expected_returns,
            constraints=constraints,
        )


def test_r11_rejects_a_negative_edge_floor() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        R11Policy(deployment_edge_floor=-1e-6).validate()


def _model(
    expected_returns: list[float] | None = None,
    constraints: OptionConstraints | None = None,
) -> OptionMarkowitzModel:
    frame = pd.DataFrame(
        {
            "underlying": ["AAA", "BBB"],
            "mark": [10.0, 20.0],
            "spot": [100.0, 100.0],
            "delta": [0.1, 0.1],
            "gamma": [0.0, 0.0],
            "vega": [0.0, 0.0],
            "theta": [0.0, 0.0],
            "stress_scenario_reference": [0.0, 0.0],
        },
        index=["call", "put"],
    )
    covariance = pd.DataFrame(
        np.eye(2) * 1e-8,
        index=["AAA", "BBB"],
        columns=["AAA", "BBB"],
    )
    return OptionMarkowitzModel(
        OptionSpec(frame),
        FactorShockSpec(underlying_cov=covariance),
        pd.Series(expected_returns or [0.02, 0.01], index=frame.index),
        constraints=constraints,
    )


def _costs(short_allowed: bool = True) -> OptimizationCostSpec:
    index = ["call", "put"]
    return OptimizationCostSpec(
        long_cost=pd.Series(0.0, index=index),
        short_cost=pd.Series(0.0, index=index),
        short_margin=pd.Series(0.0, index=index),
        short_allowed=pd.Series(short_allowed, index=index),
    )


def _scenarios() -> pd.DataFrame:
    return pd.DataFrame({"call": [0.0, 0.0, 0.0], "put": [0.0, 0.0, 0.0]})


def test_integer_conversion_is_exact_signed_truncation() -> None:
    model = _model()
    result = integerize_or_cash(
        model,
        pd.Series({"call": 0.15, "put": -0.25}),
        pd.Series({"call": 10.0, "put": 20.0}),
        nav=10_000.0,
        scenarios=_scenarios(),
        costs=_costs(),
        policy=R1_POLICY,
    )

    pd.testing.assert_series_equal(
        result.contracts,
        pd.Series({"call": 1.0, "put": -1.0}, name="contracts"),
    )
    pd.testing.assert_series_equal(
        result.weights,
        pd.Series({"call": 0.10, "put": -0.20}, name="weight"),
    )
    assert result.abstained is False


def test_all_zero_truncation_is_feasible_direct_cash() -> None:
    model = _model()
    result = integerize_or_cash(
        model,
        pd.Series({"call": 0.003, "put": -0.002}),
        pd.Series({"call": 10.0, "put": 20.0}),
        nav=1_000.0,
        scenarios=_scenarios(),
        costs=_costs(),
        policy=R1_POLICY,
    )

    assert result.weights.eq(0.0).all()
    assert result.abstained is False
    assert result.reason == "direct_truncation"


def test_infeasible_direct_book_selects_cash_and_preserves_breach() -> None:
    model = _model(constraints=OptionConstraints(per_contract_abs=0.05))
    result = integerize_or_cash(
        model,
        pd.Series({"call": 0.15, "put": 0.0}),
        pd.Series({"call": 10.0, "put": 20.0}),
        nav=10_000.0,
        scenarios=_scenarios(),
        costs=_costs(),
        policy=R1_POLICY,
    )

    assert result.abstained is True
    assert result.weights.eq(0.0).all()
    assert result.rejected_diagnostics is not None
    assert result.rejected_diagnostics["caps"] > 0.0


def test_negative_standalone_edge_hedge_does_not_trigger_abstention() -> None:
    model = _model(expected_returns=[0.02, -0.01])
    result = integerize_or_cash(
        model,
        pd.Series({"call": 0.10, "put": -0.20}),
        pd.Series({"call": 10.0, "put": 20.0}),
        nav=10_000.0,
        scenarios=_scenarios(),
        costs=_costs(),
        policy=R1_POLICY,
    )

    assert result.abstained is False
    assert result.contracts["put"] == -1.0


def test_r11_stage1_at_target_is_unchanged(monkeypatch) -> None:
    model = _model()
    stage1 = model._result(
        np.array([0.5, 0.0]),
        "optimal",
        0.0,
        "stub",
        risk_aversion=1.0,
    )
    monkeypatch.setattr(model, "solve_net_utility", lambda *args, **kwargs: stage1)

    result = solve_r11_net_utility(model, _scenarios(), _costs(), R11_POLICY)

    pd.testing.assert_series_equal(result.weights, stage1.weights)
    assert result.diagnostics["deployment_target_applied"] is False


def test_r11_no_positive_edge_retains_stage1(monkeypatch) -> None:
    model = _model(expected_returns=[-0.02, -0.01])
    stage1 = model._result(
        np.array([0.1, 0.0]),
        "optimal",
        0.0,
        "stub",
        risk_aversion=1.0,
    )
    monkeypatch.setattr(model, "solve_net_utility", lambda *args, **kwargs: stage1)

    result = solve_r11_net_utility(model, _scenarios(), _costs(), R11_POLICY)

    pd.testing.assert_series_equal(result.weights, stage1.weights)
    assert result.diagnostics["positive_edge_contracts"] == 0


def test_r11_infeasible_target_retains_stage1(monkeypatch) -> None:
    model = _model(constraints=OptionConstraints(per_contract_abs=0.2))
    stage1 = model._result(
        np.array([0.1, 0.0]),
        "optimal",
        0.0,
        "stub",
        risk_aversion=1.0,
    )
    monkeypatch.setattr(model, "solve_net_utility", lambda *args, **kwargs: stage1)

    result = solve_r11_net_utility(model, _scenarios(), _costs(), R11_POLICY)

    pd.testing.assert_series_equal(result.weights, stage1.weights)
    assert result.diagnostics["deployment_target_feasible"] is False


def test_r11_feasible_target_hits_exactly_point_five(monkeypatch) -> None:
    model = _model()
    stage1 = model._result(
        np.array([0.1, 0.0]),
        "optimal",
        0.0,
        "stub",
        risk_aversion=1.0,
    )
    monkeypatch.setattr(model, "solve_net_utility", lambda *args, **kwargs: stage1)

    result = solve_r11_net_utility(model, _scenarios(), _costs(), R11_POLICY)

    assert result.weights.abs().sum() == pytest.approx(0.5, abs=1e-6)
    assert result.weights["call"] > 0.0
    assert result.weights["put"] == pytest.approx(0.0, abs=1e-8)
    assert result.diagnostics["deployment_target_applied"] is True


def test_r11_rejects_a_target_that_fails_full_validation(monkeypatch) -> None:
    model = _model()
    stage1 = model._result(
        np.array([0.1, 0.0]),
        "optimal",
        0.0,
        "stub",
        risk_aversion=1.0,
    )
    monkeypatch.setattr(model, "solve_net_utility", lambda *args, **kwargs: stage1)
    monkeypatch.setattr(
        risk_controls,
        "validate_portfolio",
        lambda *args, **kwargs: {"feasible": False, "max_violation": 0.1},
    )

    result = solve_r11_net_utility(model, _scenarios(), _costs(), R11_POLICY)

    pd.testing.assert_series_equal(result.weights, stage1.weights)
    assert result.diagnostics["deployment_target_applied"] is False
