from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis.breadth_solutions_lib import (
    CapConstrainedMarkowitzModel,
    cap_feasibility,
    capped_naive_weights,
    compute_liquidity_caps,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
)
from research.papers.option_only_markowitz.analysis.option_market_hours import (
    classify_cboe_option_rth_timestamp,
)
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
)


def _synthetic_inputs():
    contracts = [f"c{i}" for i in range(7)]
    frame = pd.DataFrame(
        {
            "underlying": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "mark": [10.0, 12.0, 8.0, 9.0, 11.0, 7.0, 13.0],
            "spot": [100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0],
            "delta": [0.45, -0.35, 0.25, -0.20, 0.40, -0.30, 0.15],
            "gamma": [0.010, 0.008, 0.009, 0.007, 0.012, 0.006, 0.005],
            "vega": [0.30, 0.25, 0.20, 0.22, 0.24, 0.18, 0.16],
            "theta": [-0.02, -0.015, -0.018, -0.014, -0.016, -0.012, -0.010],
            "kind": ["call", "put", "call", "put", "call", "put", "call"],
            "moneyness_bucket": ["atm", "atm", "near", "near", "atm", "near", "wing"],
        },
        index=contracts,
    )
    under = ["AAA", "BBB"]
    underlying_cov = pd.DataFrame([[0.030, 0.006], [0.006, 0.025]], index=under, columns=under)
    vol_cov = pd.DataFrame([[0.010, 0.002], [0.002, 0.012]], index=under, columns=under)
    expected_returns = pd.Series(
        [0.12, 0.06, 0.05, 0.04, 0.035, 0.025, 0.015],
        index=contracts,
        name="conditional_expected_return",
    )
    residual_cov = pd.DataFrame(np.eye(len(contracts)) * 0.015, index=contracts, columns=contracts)
    constraints = OptionMarkowitzConstraints(
        gross_nav=1.0,
        net_nav_abs=1.0,
        short_nav_abs=0.25,
        per_contract_abs=0.18,
        underlying_gross={"AAA": 0.72, "BBB": 0.72},
    )
    return OptionOnlySpec(frame), FactorShockSpec(underlying_cov, vol_cov), expected_returns, residual_cov, constraints


def _model(cls=OptionOnlyMarkowitzModel, caps=None):
    options, shocks, expected_returns, residual_cov, constraints = _synthetic_inputs()
    kwargs = {"per_contract_caps": caps} if cls is CapConstrainedMarkowitzModel else {}
    return cls(
        options,
        shocks,
        expected_returns,
        residual_cov=residual_cov,
        constraints=constraints,
        covariance_shrinkage=0.20,
        **kwargs,
    )


@pytest.mark.parametrize("caps", [None, "full"])
def test_noncap_equals_base_cvxpy(caps):
    base = _model()
    cap_series = None if caps is None else pd.Series(0.18, index=base.contracts)
    capped = _model(CapConstrainedMarkowitzModel, cap_series)

    base_res = base.solve_max_sharpe(method="cvxpy")
    capped_res = capped.solve_max_sharpe(method="cvxpy")

    assert base_res.status != "infeasible"
    assert capped_res.status != "infeasible"
    assert np.array_equal(base_res.weights.to_numpy(), capped_res.weights.to_numpy())


@pytest.mark.parametrize("caps", [None, "full"])
def test_noncap_equals_base_slsqp(caps):
    base = _model()
    cap_series = None if caps is None else pd.Series(0.18, index=base.contracts)
    capped = _model(CapConstrainedMarkowitzModel, cap_series)

    base_res = base.solve_max_sharpe(method="slsqp")
    capped_res = capped.solve_max_sharpe(method="slsqp")

    assert base_res.status != "infeasible"
    assert capped_res.status != "infeasible"
    assert np.array_equal(base_res.weights.to_numpy(), capped_res.weights.to_numpy())


def test_binding_caps_respected():
    base = _model()
    caps = pd.Series(0.18, index=base.contracts)
    caps.loc["c0"] = 0.05
    capped = _model(CapConstrainedMarkowitzModel, caps)

    res = capped.solve_max_sharpe(method="cvxpy")

    assert res.status != "infeasible"
    assert (res.weights.abs().reindex(base.contracts) <= caps + 1e-8).all()
    assert abs(res.weights.loc["c0"]) <= 0.05 + 1e-8


def test_infeasible_when_caps_sum_below_gross():
    base = _model()
    caps = pd.Series(0.3 / len(base.contracts), index=base.contracts)
    capped = _model(CapConstrainedMarkowitzModel, caps)

    cvxpy_res = capped.solve_max_sharpe(method="cvxpy")
    slsqp_res = capped.solve_max_sharpe(method="slsqp")

    assert cvxpy_res.status == "infeasible"
    assert slsqp_res.status != "optimal" or slsqp_res.weights.abs().sum() < base.constraints.gross_nav - 1e-5
    feasibility = cap_feasibility(pd.DataFrame({"bound": caps}), base.constraints)
    assert feasibility["gross_feasible"] is False


def test_burn_does_not_consume_caps():
    options, shocks, expected_returns, residual_cov, constraints = _synthetic_inputs()
    expected_returns = pd.Series(0.01, index=options.frame.index, name="conditional_expected_return")
    expected_returns.loc["c0"] = 10.0
    caps = pd.Series(0.18, index=options.frame.index)
    caps.loc["c0"] = 0.05
    capped = CapConstrainedMarkowitzModel(
        options,
        shocks,
        expected_returns,
        residual_cov=residual_cov,
        constraints=constraints,
        covariance_shrinkage=0.20,
        per_contract_caps=caps,
    )

    res = capped.solve_max_sharpe(method="cvxpy")

    assert res.status != "infeasible"
    assert abs(res.weights.loc["c0"]) == pytest.approx(0.05, abs=1e-6)
    assert (res.weights.abs().reindex(caps.index) <= caps + 1e-8).all()


def test_violation_detects_cap_breach():
    base = _model()
    caps = pd.Series(0.18, index=base.contracts)
    caps.loc["c0"] = 0.05
    capped = _model(CapConstrainedMarkowitzModel, caps)
    weights = pd.Series(0.0, index=base.contracts)
    weights.loc["c0"] = 0.06

    assert capped._max_constraint_violation(weights.to_numpy()) > 0.0


def test_compute_liquidity_caps_formula():
    reps = pd.DataFrame(
        {
            "asset_id": ["a", "a", "a", "b", "c"],
            "snap_date": [
                "2020-01-31",
                "2020-02-29",
                "2021-01-31",
                pd.NaT,
                "2020-04-30",
            ],
            "trade_date": [pd.NaT, pd.NaT, pd.NaT, "2020-03-31", pd.NaT],
            "volume": [4.0, 20.0, 1000.0, 1.5, 100.0],
        }
    )
    spec_mark = pd.Series({"a": 10.0, "b": 5.0, "c": 100.0, "missing": 1.0})

    caps = compute_liquidity_caps(
        reps,
        spec_mark,
        nav=10_000.0,
        participation=0.10,
        per_contract_abs=0.18,
        option_multiplier=100.0,
        train_end=pd.Timestamp("2020-12-31"),
    )

    assert caps.loc["a", "train_volume"] == 12.0
    assert caps.loc["a", "cap_contracts"] == pytest.approx(1.2)
    assert caps.loc["a", "w_cap"] == pytest.approx(0.12)
    assert caps.loc["a", "bound"] == pytest.approx(0.12)
    assert caps.loc["b", "cap_contracts"] == 1.0
    assert caps.loc["b", "bound"] == pytest.approx(0.05)
    assert caps.loc["c", "bound"] == pytest.approx(0.18)
    assert caps.loc["missing", "bound"] == pytest.approx(0.18)
    assert bool(caps.loc["missing", "has_volume"]) is False


def test_capped_naive_weights_redistribute_without_optimizing():
    weights = pd.Series({"a": 0.50, "b": 0.30, "c": 0.20})
    caps = pd.Series({"a": 0.40, "b": 0.40, "c": 0.40})

    capped = capped_naive_weights(weights, caps, target_gross=1.0)

    assert capped.abs().sum() == pytest.approx(1.0)
    assert capped.loc["a"] == pytest.approx(0.40)
    assert capped.loc["b"] == pytest.approx(0.36)
    assert capped.loc["c"] == pytest.approx(0.24)
    assert (capped.abs() <= caps + 1e-12).all()


def test_capped_naive_weights_deploys_cap_budget_when_target_infeasible():
    weights = pd.Series({"a": 0.70, "b": 0.30})
    caps = pd.Series({"a": 0.10, "b": 0.20})

    capped = capped_naive_weights(weights, caps, target_gross=1.0)

    assert capped.abs().sum() == pytest.approx(0.30)
    assert (capped.abs() <= caps + 1e-12).all()


def test_poc_missing_cbbo_uses_current_assumption_ledger(tmp_path):
    assumptions = pd.DataFrame(
        {
            "underlying": ["AAA"],
            "quote_symbol": ["AAA"],
            "asset_class": ["equity_option"],
            "moneyness_bucket": ["atm"],
            "tenor_bucket": ["le_45d"],
            "chain_timestamp": ["2026-07-06 14:00:00"],
            "timestamp_tz_assumed": ["UTC"],
            "market_hours_snapshot": [True],
            "fill_relative_spread": [0.04],
            "fill_abs_spread": [0.20],
            "source_url": ["https://example.invalid/AAA"],
            "fill_method": ["test"],
        }
    )
    assumption_path = tmp_path / "current_option_spread_assumptions.csv"
    assumptions.to_csv(assumption_path, index=False)
    reps = pd.DataFrame(
        {
            "snap_date": [pd.Timestamp("2020-01-31")],
            "asset_id": ["AAA_call_atm"],
            "symbol": ["AAA 20200221 C100"],
            "volume": [100.0],
            "cbbo_median_relative_spread": [0.10],
            "breadth_spread_source": ["poc_missing_cbbo"],
            "moneyness_bucket": ["atm"],
        }
    )
    detail = pd.DataFrame(
        {
            "return_date": [pd.Timestamp("2020-02-21")],
            "decision_date": [pd.Timestamp("2020-01-31")],
            "expiry": [pd.Timestamp("2020-02-21")],
            "asset_id": ["AAA_call_atm"],
            "symbol": ["AAA 20200221 C100"],
            "underlying": ["AAA"],
            "kind": ["call"],
            "moneyness_bucket": ["atm"],
            "mark": [5.0],
            "option_return": [0.10],
            "expiry_days": [21],
            "asset_class": ["equity_option"],
        }
    )

    cost = build_cost_input_ledger(
        reps,
        detail,
        tmp_path,
        ResearchCostConfig(
            use_cbbo_spread_surface=False,
            current_spread_assumptions_path=str(assumption_path),
        ),
    )

    assert cost.loc[0, "relative_spread"] == pytest.approx(0.04)
    assert cost.loc[0, "relative_spread_source"] == "current_cboe_liquid_quote"


def test_off_hours_current_assumption_is_ignored(tmp_path):
    assumptions = pd.DataFrame(
        {
            "underlying": ["AAA"],
            "quote_symbol": ["AAA"],
            "asset_class": ["equity_option"],
            "moneyness_bucket": ["atm"],
            "tenor_bucket": ["le_45d"],
            "chain_timestamp": ["2026-07-04 14:00:00"],
            "timestamp_tz_assumed": ["UTC"],
            "market_hours_snapshot": [False],
            "fill_relative_spread": [0.04],
            "fill_abs_spread": [0.20],
            "source_url": ["https://example.invalid/AAA"],
            "fill_method": ["test"],
        }
    )
    assumption_path = tmp_path / "current_option_spread_assumptions.csv"
    assumptions.to_csv(assumption_path, index=False)
    reps = pd.DataFrame(
        {
            "snap_date": [pd.Timestamp("2020-01-31")],
            "asset_id": ["AAA_call_atm"],
            "symbol": ["AAA 20200221 C100"],
            "volume": [100.0],
            "cbbo_median_relative_spread": [0.10],
            "breadth_spread_source": ["poc_missing_cbbo"],
            "moneyness_bucket": ["atm"],
        }
    )
    detail = pd.DataFrame(
        {
            "return_date": [pd.Timestamp("2020-02-21")],
            "decision_date": [pd.Timestamp("2020-01-31")],
            "expiry": [pd.Timestamp("2020-02-21")],
            "asset_id": ["AAA_call_atm"],
            "symbol": ["AAA 20200221 C100"],
            "underlying": ["AAA"],
            "kind": ["call"],
            "moneyness_bucket": ["atm"],
            "mark": [5.0],
            "option_return": [0.10],
            "expiry_days": [21],
            "asset_class": ["equity_option"],
        }
    )

    cost = build_cost_input_ledger(
        reps,
        detail,
        tmp_path,
        ResearchCostConfig(
            use_cbbo_spread_surface=False,
            current_spread_assumptions_path=str(assumption_path),
        ),
    )

    assert cost.loc[0, "relative_spread"] == pytest.approx(0.10)
    assert cost.loc[0, "relative_spread_source"] == "default"


def test_cboe_option_market_hours_classification():
    valid = classify_cboe_option_rth_timestamp("2026-07-06 14:00:00", timestamp_tz="UTC")
    observed_independence_day = classify_cboe_option_rth_timestamp("2026-07-03 19:00:00", timestamp_tz="UTC")
    saturday = classify_cboe_option_rth_timestamp("2026-07-04 14:00:00", timestamp_tz="UTC")

    assert valid.valid is True
    assert valid.reason == "regular_trading_hours"
    assert observed_independence_day.valid is False
    assert observed_independence_day.reason == "no_regular_trading_session"
    assert saturday.valid is False
    assert saturday.reason == "no_regular_trading_session"
