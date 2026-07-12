from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_ingestion.market_data.fetch_r11_event_cbbo import build_requests
from research.papers.option_only_markowitz.analysis.r11_higher_risk_pipeline import (
    R11NetUtilityConfig,
    R11_NAME,
    build_risk_off_status_rows,
    reconcile_selected_integer_diagnostics,
    solve_r11_net_utility,
)
from research.papers.option_only_markowitz.analysis.r11_integer_repair import (
    integerize_r11_weights,
)
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    OptimizationCostSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    estimate_greek_joint_moments,
    greek_exposure_frame,
)
from src.portfolio.r11_risk_controls import (
    EgarchOverlayConfig,
    RiskOffConfig,
    apply_egarch_joint_overlay,
    build_vix_risk_off_events,
    egarch_variance_forecast,
    evaluate_egarch_gate,
    execute_cbbo_orders,
    risk_off_exposure_calendar,
)


def _model() -> tuple[OptionOnlyMarkowitzModel, pd.DataFrame, OptimizationCostSpec]:
    options = OptionOnlySpec(
        pd.DataFrame(
            {
                "underlying": ["AAA", "AAA", "AAA"],
                "mark": [5.0, 4.0, 3.0],
                "spot": [100.0, 100.0, 100.0],
                "delta": [0.55, -0.40, 0.25],
                "gamma": [0.020, 0.025, 0.012],
                "vega": [18.0, 16.0, 10.0],
                "theta": [-0.03, -0.02, -0.01],
                "asset_class": ["equity_option"] * 3,
                "stress_scenario_crash": [-0.45, 0.25, -0.10],
            },
            index=["call", "put", "wing"],
        )
    )
    rng = np.random.default_rng(20260712)
    dates = pd.date_range("2010-01-31", periods=96, freq="ME")
    spot = rng.normal(0.004, 0.035, len(dates))
    factors = pd.DataFrame(
        {
            "r_AAA": spot,
            "r2_AAA": spot**2 - float(np.mean(spot**2)),
            "dv_AAA": -0.30 * spot + rng.normal(0.0, 0.012, len(dates)),
        },
        index=dates,
    )
    B = greek_exposure_frame(options)
    residual = rng.normal(0.0, 0.025, (len(dates), 3))
    residual[:, 0] += 0.30 * spot
    residual[:, 1] -= 0.20 * spot
    returns = pd.DataFrame(
        factors.to_numpy() @ B.T.to_numpy() + residual,
        index=dates,
        columns=B.index,
    )
    moments = estimate_greek_joint_moments(returns, factors, B, regularize=True)
    shocks = FactorShockSpec(
        underlying_cov=pd.DataFrame([[0.02]], index=["AAA"], columns=["AAA"]),
        vol_cov=pd.DataFrame([[0.01]], index=["AAA"], columns=["AAA"]),
    )
    model = OptionOnlyMarkowitzModel(
        options,
        shocks,
        expected_returns=pd.Series([0.035, 0.015, -0.02], index=options.frame.index),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.8),
        covariance_shrinkage=0.0,
        joint_moments=moments,
    )
    costs = OptimizationCostSpec(
        long_cost=pd.Series(0.002, index=model.contracts),
        short_cost=pd.Series(0.012, index=model.contracts),
        short_margin=pd.Series(1.5, index=model.contracts),
        assignment_short_allowed=pd.Series([False, True, True], index=model.contracts),
    )
    return model, returns, costs


def test_vix_close_rule_deduplicates_manual_march_exit_and_reenters_next_session():
    close = pd.Series(
        [39.16, 40.11, 33.42],
        index=pd.to_datetime(["2020-02-27", "2020-02-28", "2020-03-02"]),
    )
    sessions = pd.bdate_range("2020-02-27", "2020-03-05")
    events = build_vix_risk_off_events(close, sessions)
    assert list(events["action"]) == ["exit", "reenter"]
    assert list(events["execution_date"].dt.strftime("%Y-%m-%d")) == ["2020-03-02", "2020-03-03"]
    assert events.iloc[0]["deduplicated_signal_count"] == 2
    assert "user_attested_manual_2020" in events.iloc[0]["source"]
    calendar = risk_off_exposure_calendar(events, sessions).set_index("session")
    assert calendar.loc[pd.Timestamp("2020-03-02"), "exposure_multiplier"] == 0.0
    assert calendar.loc[pd.Timestamp("2020-03-03"), "exposure_multiplier"] == 1.0


def test_cbbo_execution_uses_bid_for_sells_ask_for_buys_and_displayed_size_once():
    quotes = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                ["2020-03-02 14:30:01Z", "2020-03-02 14:31:01Z", "2020-03-02 14:30:02Z"]
            ),
            "symbol": ["SELL", "SELL", "BUY"],
            "bid_px_00": [2.00, 1.90, 3.00],
            "ask_px_00": [2.10, 2.00, 3.20],
            "bid_sz_00": [2, 3, 5],
            "ask_sz_00": [2, 3, 5],
        }
    )
    orders = pd.DataFrame({"symbol": ["SELL", "BUY"], "order_contracts": [-4, 3]})
    fills, summary = execute_cbbo_orders(orders, quotes, pd.Timestamp("2020-03-02"), RiskOffConfig(slippage_bps=0))
    actual = fills[fills["quote_row"].ne("order_summary")]
    assert summary["execution_feasible"] is True
    assert actual.loc[actual["symbol"].eq("SELL"), "filled_contracts"].tolist() == [2.0, 2.0]
    assert actual.loc[actual["symbol"].eq("SELL"), "execution_price"].tolist() == [2.0, 1.9]
    assert actual.loc[actual["symbol"].eq("BUY"), "execution_price"].tolist() == [3.2]


def test_missing_or_insufficient_cbbo_fails_closed():
    orders = pd.DataFrame({"symbol": ["X"], "order_contracts": [-4]})
    quotes = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(["2020-03-02 14:30:01Z"]),
            "symbol": ["X"],
            "bid_px_00": [2.0],
            "ask_px_00": [2.1],
            "bid_sz_00": [1],
            "ask_sz_00": [1],
        }
    )
    _, summary = execute_cbbo_orders(orders, quotes, pd.Timestamp("2020-03-02"))
    assert summary["execution_feasible"] is False
    assert summary["filled_contracts"] == 1.0
    assert summary["incomplete_symbols"] == ["X"]


@pytest.mark.skipif(pytest.importorskip("cvxpy") is None, reason="cvxpy unavailable")
def test_r11_raises_risk_cap_without_forcing_negative_edge_or_split_burn():
    model, scenarios, costs = _model()
    config = R11NetUtilityConfig(
        annual_vol_target=0.25,
        deployment_target=0.50,
        cvar_loss_nav=0.50,
        stress_loss_nav=0.50,
    )
    result = solve_r11_net_utility(model, scenarios, costs, config)
    stats = result.objective_stats
    assert stats["predicted_annual_vol"] <= 0.25 + 1e-5
    assert stats["risk_aversion"] > 0
    assert result.gross_nav <= 1.0 + 1e-7
    assert result.weights["call"] >= -1e-8
    assert result.weights["wing"] <= 1e-8  # negative long edge is never added for deployment
    assert np.abs(result.weights.to_numpy()).sum() == pytest.approx(result.gross_nav)


def test_negative_net_opportunities_still_hold_cash_under_r11():
    model, scenarios, costs = _model()
    negative = OptionOnlyMarkowitzModel(
        model.options,
        model.shocks,
        expected_returns=pd.Series(-0.10, index=model.contracts),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.8, long_only=True),
        covariance_shrinkage=0.0,
        joint_moments=model.joint_moments,
    )
    result = solve_r11_net_utility(
        negative,
        scenarios,
        costs,
        R11NetUtilityConfig(cvar_loss_nav=0.50, stress_loss_nav=0.50),
    )
    assert result.gross_nav == pytest.approx(0.0, abs=1e-7)
    assert result.objective_stats["deployment_target_applied"] is False


def test_egarch_joint_scaling_is_psd_and_preserves_cross_term_transformation():
    model, _, _ = _model()
    original = model.joint_moments.joint_covariance().to_numpy(float)
    overlaid = apply_egarch_joint_overlay(model.joint_moments, {"AAA": 1.44})
    joint = overlaid.joint_covariance().to_numpy(float)
    scales = np.array([1.2, 1.44, 1.0] + [1.0] * len(model.contracts))
    np.testing.assert_allclose(joint, scales[:, None] * original * scales[None, :])
    assert np.linalg.eigvalsh(joint).min() >= -1e-9
    assert np.max(np.abs(overlaid.factor_residual_cov.to_numpy(float))) > 0


def test_egarch_forecast_is_cutoff_safe_and_retains_march_observation():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2017-01-02", periods=900)
    returns = pd.Series(rng.normal(0, 0.012, len(dates)), index=dates)
    march_date = pd.Timestamp("2020-03-16")
    returns.loc[march_date] = -0.12
    cutoff = pd.Timestamp("2020-04-30")
    config = EgarchOverlayConfig(min_observations=500, lookback_days=900)
    baseline = egarch_variance_forecast(returns, cutoff, config)
    changed_future = returns.copy()
    changed_future.loc[changed_future.index > cutoff] = 10.0
    after = egarch_variance_forecast(changed_future, cutoff, config)
    assert baseline["valid"] is True
    assert baseline["n_obs"] == after["n_obs"]
    assert baseline["forecast_variance"] == pytest.approx(after["forecast_variance"])
    without_march = egarch_variance_forecast(returns.drop(march_date), cutoff, config)
    assert without_march["n_obs"] == baseline["n_obs"] - 1


def test_egarch_gate_stays_diagnostic_when_coverage_or_survival_fails():
    rows = pd.DataFrame(
        {
            "valid": [True, False],
            "realized_variance": [0.02, np.nan],
            "forecast_variance": [0.019, np.nan],
            "baseline_variance": [0.03, np.nan],
        }
    )
    gate = evaluate_egarch_gate(rows, added_survival_failures=1, worst_es_deterioration=0.0)
    assert gate["passed"] is False
    assert gate["promotion_status"] == "diagnostic_only"


def test_risk_off_arm_is_unscored_instead_of_inheriting_no_rule_return():
    base = pd.DataFrame(
        {
            "config": ["orig"],
            "strategy": [R11_NAME],
            "return_date": [pd.Timestamp("2020-03-31")],
            "gross_return": [-0.8],
            "predicted_cost": [0.01],
            "net_return": [-0.81],
        }
    )
    execution = pd.DataFrame(
        {"execution_feasible": [False], "execution_date": [pd.Timestamp("2020-03-02")]}
    )
    risk_off = build_risk_off_status_rows(base, execution)
    assert pd.isna(risk_off.loc[0, "net_return"])
    assert risk_off.loc[0, "evidence_status"] == "unscored_missing_or_incomplete_executable_quotes"


def test_targeted_cbbo_pull_deduplicates_symbols_by_execution_date():
    manifest = pd.DataFrame(
        {
            "execution_date": ["2020-03-02", "2020-03-02", "2020-03-03"],
            "symbol": ["AAPL  200320C00270000", "AAPL  200320C00270000", "TSLA  200320C00650000"],
        }
    )
    requests = build_requests(manifest)
    assert len(requests) == 2
    assert requests[0]["schema"] == "cbbo-1m"
    assert requests[0]["symbols"] == ["AAPL  200320C00270000"]
    assert requests[0]["end"] == "2020-03-03"


def test_integer_execution_uses_direct_conversion_and_returns_whole_contracts():
    model, scenarios, costs = _model()
    config = R11NetUtilityConfig(cvar_loss_nav=0.50, stress_loss_nav=0.50)
    continuous = solve_r11_net_utility(model, scenarios, costs, config)
    marks = model.frame["mark"]
    caps = pd.Series(0.8, index=model.contracts)
    nav = 100_000.0
    repaired = integerize_r11_weights(
        model,
        continuous.weights,
        marks,
        nav,
        caps,
        scenarios,
        costs,
        config,
        risk_aversion=float(continuous.objective_stats["risk_aversion"]),
    )
    assert set(repaired.candidates["method"]) == {
        "truncate_toward_cash",
        "cash_abstention",
    }
    implied_counts = repaired.weights * nav / (100.0 * marks)
    np.testing.assert_allclose(implied_counts, np.rint(implied_counts), atol=1e-8)
    assert repaired.diagnostics["feasible"] is True
    assert "pre_repair_max_breach" in repaired.diagnostics
    assert repaired.diagnostics["integer_execution_abstained"] is False
    assert repaired.diagnostics["selected_integer_method"] == "truncate_toward_cash"


def test_failed_rounding_abstains_without_hiding_original_diagnostics():
    model, scenarios, costs = _model()
    config = R11NetUtilityConfig(
        annual_vol_target=0.25,
        cvar_loss_nav=0.0001,
        stress_loss_nav=0.50,
    )
    continuous = pd.Series([0.20, 0.0, 0.0], index=model.contracts)
    repaired = integerize_r11_weights(
        model,
        continuous,
        model.frame["mark"],
        100_000.0,
        pd.Series(0.8, index=model.contracts),
        scenarios,
        costs,
        config,
        risk_aversion=1.0,
    )
    truncate = repaired.candidates.set_index("method").loc["truncate_toward_cash"]
    assert bool(truncate["feasible"]) is False
    assert truncate["breach_cvar"] > 0
    assert repaired.diagnostics["pre_repair_feasible"] is False
    assert repaired.diagnostics["pre_repair_max_breach"] > 0
    assert repaired.diagnostics["failed_breach_cvar"] > 0
    assert repaired.diagnostics["feasible"] is True
    assert repaired.diagnostics["integer_execution_abstained"] is True
    assert repaired.diagnostics["integer_repair_failed_to_cash"] is False
    assert repaired.diagnostics["selected_integer_method"] == "cash_abstention"
    assert repaired.weights.abs().sum() == pytest.approx(0.0)
    selected = repaired.candidates[repaired.candidates["selected"]].iloc[0]
    assert selected["method"] == "cash_abstention"


def test_negative_standalone_edge_hedge_does_not_force_abstention():
    model, scenarios, costs = _model()
    continuous = pd.Series([0.0, 0.0, 0.03], index=model.contracts)
    repaired = integerize_r11_weights(
        model,
        continuous,
        model.frame["mark"],
        100_000.0,
        pd.Series(0.8, index=model.contracts),
        scenarios,
        costs,
        R11NetUtilityConfig(
            annual_vol_target=1.0,
            cvar_loss_nav=0.50,
            stress_loss_nav=0.50,
        ),
        risk_aversion=1.0,
    )
    direct = repaired.candidates.set_index("method").loc["truncate_toward_cash"]
    assert direct["breach_positive_edge"] > 0
    assert bool(direct["feasible"]) is True
    assert repaired.diagnostics["integer_execution_abstained"] is False
    assert repaired.diagnostics["selected_integer_method"] == "truncate_toward_cash"


def test_selected_integer_risk_diagnostics_replace_continuous_headline_values():
    returns = pd.DataFrame(
        {
            "config": ["orig+VIX"],
            "strategy": [R11_NAME],
            "return_date": ["2020-04-30"],
            "decision_date": ["2020-02-28"],
            "predicted_annual_vol": [0.2499],
            "scenario_cvar_loss": [0.0],
            "short_margin_used": [0.0],
            "collateral_used": [0.0],
            "gross_nav": [0.0],
        }
    )
    repairs = pd.DataFrame(
        {
            "config": ["orig+VIX"],
            "strategy": [R11_NAME],
            "return_date": ["2020-04-30"],
            "decision_date": ["2020-02-28"],
            "selected": [True],
            "predicted_annual_vol": [0.2390],
            "scenario_cvar_loss": [0.0997],
            "short_margin_used": [0.1688],
            "collateral_used": [0.1761],
            "gross_nav": [0.0566],
        }
    )
    reconciled = reconcile_selected_integer_diagnostics(returns, repairs).iloc[0]
    assert reconciled["continuous_predicted_annual_vol"] == pytest.approx(0.2499)
    assert reconciled["predicted_annual_vol"] == pytest.approx(0.2390)
    assert reconciled["scenario_cvar_loss"] == pytest.approx(0.0997)
    assert reconciled["gross_nav"] == pytest.approx(0.0566)
