from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.papers.option_only_markowitz.analysis.simulation import performance_metrics
from research.papers.option_only_markowitz.analysis.r2_stability import (
    circular_block_path_suite,
    evaluate_promotion_gate,
    paired_stationary_bootstrap,
)
from research.papers.option_only_markowitz.analysis.r2_robust_sortino_pipeline import build_r2_model
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    OptimizationCostSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    estimate_greek_joint_moments,
    greek_exposure_frame,
)
from src.portfolio.r2_robust_sortino import (
    RobustSortinoConfig,
    apply_joint_volatility_scaling,
    estimate_r2_moments,
    integerize_r2_direct_or_abstain,
    option_covariance,
    select_daily_volatility_overlay,
    select_log_growth_scale,
    select_recent_covariance_weight,
    solve_robust_sortino_direction,
)


def _data() -> tuple[OptionOnlySpec, pd.DataFrame, pd.DataFrame]:
    options = OptionOnlySpec(
        pd.DataFrame(
            {
                "underlying": ["AAA", "AAA"],
                "mark": [5.0, 4.0],
                "spot": [100.0, 100.0],
                "delta": [0.50, -0.35],
                "gamma": [0.015, 0.020],
                "vega": [15.0, 13.0],
                "theta": [-0.02, -0.015],
                "asset_class": ["equity_option", "equity_option"],
                "stress_scenario_crash": [-0.30, 0.20],
            },
            index=["call", "put"],
        )
    )
    rng = np.random.default_rng(41)
    dates = pd.date_range("2015-02-28", periods=90, freq="ME")
    spot = rng.normal(0.005, 0.025, len(dates))
    factors = pd.DataFrame(
        {
            "r_AAA": spot,
            "r2_AAA": spot**2 - np.mean(spot**2),
            "dv_AAA": -0.25 * spot + rng.normal(0, 0.01, len(dates)),
        },
        index=dates,
    )
    b = greek_exposure_frame(options)
    residual = rng.normal(0, 0.015, (len(dates), 2))
    returns = pd.DataFrame(factors.to_numpy() @ b.T.to_numpy() + residual, index=dates, columns=b.index)
    return options, factors, returns


def _model(mu: tuple[float, float] = (0.025, 0.015)) -> tuple[OptionOnlyMarkowitzModel, pd.DataFrame]:
    options, factors, returns = _data()
    b = greek_exposure_frame(options)
    moments = estimate_greek_joint_moments(returns.iloc[-36:], factors.iloc[-36:], b, regularize=True)
    model = OptionOnlyMarkowitzModel(
        options,
        FactorShockSpec(
            pd.DataFrame([[0.001]], index=["AAA"], columns=["AAA"]),
            pd.DataFrame([[0.001]], index=["AAA"], columns=["AAA"]),
        ),
        expected_returns=pd.Series(mu, index=options.frame.index),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.9),
        covariance_shrinkage=0.0,
        joint_moments=moments,
    )
    return model, returns


def _costs(model: OptionOnlyMarkowitzModel, call: float = 0.001) -> OptimizationCostSpec:
    return OptimizationCostSpec(
        pd.Series([call, 0.001], index=model.contracts),
        pd.Series([0.005, 0.005], index=model.contracts),
        pd.Series([1.0, 1.0], index=model.contracts),
        pd.Series([False, True], index=model.contracts),
    )


def _config(**kwargs) -> RobustSortinoConfig:
    defaults = dict(
        cvar_loss_nav=0.95,
        stress_loss_nav=0.95,
        short_margin_nav=2.0,
        collateral_nav=2.0,
        annual_downside_target=0.99,
        annual_vol_target=2.0,
        max_three_month_loss=0.95,
        max_six_month_loss=0.95,
        scalar_grid_points=101,
    )
    defaults.update(kwargs)
    return RobustSortinoConfig(**defaults)


def test_r2_moments_are_cutoff_safe_psd_and_use_residual_imputation():
    options, factors, returns = _data()
    b = greek_exposure_frame(options)
    returns.iloc[-30::4, 0] = np.nan
    cutoff = returns.index[-5]
    baseline = estimate_r2_moments(returns, factors, b, train_end=cutoff)
    changed = returns.copy()
    changed.loc[changed.index > cutoff] = 1e8
    after = estimate_r2_moments(changed, factors, b, train_end=cutoff)
    np.testing.assert_allclose(baseline.blended_option_cov, after.blended_option_cov)
    assert len(baseline.imputation_scenarios) == 5
    assert all(not frame.isna().any().any() for frame in baseline.imputation_scenarios)
    assert np.linalg.eigvalsh(baseline.blended_option_cov).min() >= -1e-9
    # Missing observations are factor predictions, not literal zeroes.
    missing = returns.loc[:cutoff, "call"].isna()
    assert (baseline.option_returns_imputed.loc[missing, "call"].abs() > 0).any()


def test_r2_requires_24_recent_contract_observations():
    options, factors, returns = _data()
    returns.loc[returns.index[-36:-20], "call"] = np.nan
    with pytest.raises(ValueError, match="24 recent"):
        estimate_r2_moments(returns, factors, greek_exposure_frame(options))


def test_r2_factor_panel_tracks_only_surviving_contract_underlyings():
    options, _, returns = _data()
    options.frame["iv_proxy"] = [0.22, 0.24]
    options.frame["kind"] = ["call", "put"]
    options.frame["moneyness_bucket"] = ["atm", "atm"]
    dates = returns.index
    rng = np.random.default_rng(19)
    underlying = pd.DataFrame(
        {"AAA": rng.normal(0, 0.02, len(dates)), "BBB": rng.normal(0, 0.03, len(dates))},
        index=dates,
    )
    vol = pd.DataFrame(
        {"AAA": rng.normal(0, 0.01, len(dates)), "BBB": rng.normal(0, 0.01, len(dates))},
        index=dates,
    )
    daily_dates = pd.bdate_range("2014-01-02", periods=1_000)
    daily = pd.DataFrame(
        {"AAA": rng.normal(0, 0.01, len(daily_dates)), "BBB": rng.normal(0, 0.01, len(daily_dates))},
        index=daily_dates,
    )
    model, moments, _, _, _ = build_r2_model(
        options.frame,
        returns,
        underlying,
        vol,
        ["AAA", "BBB"],
        daily,
        dates[-1],
        RobustSortinoConfig(),
    )
    assert set(model.frame["underlying"]) == {"AAA"}
    assert moments.recent.factor_names == ["r_AAA", "r2_AAA", "dv_AAA"]


def test_covariance_weight_default_then_one_se_prefers_stability():
    config = RobustSortinoConfig()
    recent = [np.eye(2)] * 11
    expanding = [2 * np.eye(2)] * 11
    realized = [np.ones(2)] * 11
    weight, ledger = select_recent_covariance_weight(recent, expanding, realized, config)
    assert weight == 0.50
    assert ledger.iloc[0]["reason"] == "default_before_12"

    recent = [np.eye(2)] * 12
    expanding = [np.eye(2)] * 12
    weight, ledger = select_recent_covariance_weight(recent, expanding, realized + [np.ones(2)], config)
    assert weight == 0.25
    assert ledger.loc[ledger["selected"], "recent_weight"].item() == 0.25


def test_daily_volatility_overlay_is_deterministic_cutoff_safe_and_bounded():
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2018-01-02", periods=900)
    values = pd.Series(rng.normal(0, 0.012, len(dates)), index=dates)
    first = select_daily_volatility_overlay(values)
    second = select_daily_volatility_overlay(values.copy())
    assert first["available"] is True
    assert first["variance_ratio"] == pytest.approx(second["variance_ratio"])
    assert 0.67 <= first["variance_ratio"] <= 1.50
    assert first["har_weight"] in {0.0, 0.25, 0.50, 0.75, 1.0}
    assert first["horizon_days"] == 21


def test_joint_daily_scaling_preserves_cross_terms_and_psd():
    options, factors, returns = _data()
    b = greek_exposure_frame(options)
    moment = estimate_greek_joint_moments(returns, factors, b, regularize=True)
    scaled, covariance = apply_joint_volatility_scaling(moment, b, {"AAA": 1.44})
    original = moment.joint_covariance().to_numpy(float)
    scales = np.array([1.2, 1.44, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(scaled.joint_covariance(), scales[:, None] * original * scales[None, :])
    assert np.linalg.eigvalsh(covariance).min() >= -1e-9
    assert np.max(np.abs(scaled.factor_residual_cov.to_numpy(float))) > 0


@pytest.mark.skipif(pytest.importorskip("cvxpy") is None, reason="cvxpy unavailable")
def test_robust_sortino_direction_matches_small_bruteforce_and_costs_change_it():
    model, returns = _model(mu=(0.03, 0.02))
    scenarios = {"recent": returns.iloc[-36:]}
    direction, stats = solve_robust_sortino_direction(model, scenarios, _costs(model), _config())
    assert stats["status"] == "optimal"
    # Coarse brute-force on the L1 unit diamond must not beat the SOCP direction materially.
    grid = np.linspace(-1, 1, 401)
    candidates = []
    r = returns.iloc[-36:].to_numpy(float)
    for first in grid:
        second = 1.0 - abs(first)
        for sign in (-1.0, 1.0):
            w = np.array([first, sign * second])
            if w[0] < 0:  # assignment restricted
                continue
            cost = 0.001 * np.maximum(w, 0).sum() + 0.005 * np.maximum(-w, 0).sum()
            net = r @ w - cost
            downside = np.sqrt(np.mean(np.minimum(net, 0) ** 2))
            numerator = np.array([0.03, 0.02]) @ w - cost
            candidates.append(numerator / max(downside, 1e-12))
    assert stats["robust_sortino"] >= max(candidates) - 0.03
    expensive, _ = solve_robust_sortino_direction(model, scenarios, _costs(model, call=0.04), _config())
    assert expensive["call"] < direction["call"]
    assert direction["call"] >= -1e-8


def test_log_growth_scale_is_unique_and_cash_is_allowed():
    model, returns = _model()
    caps = pd.Series(1.0, index=model.contracts)
    scenarios = {"recent": returns.iloc[-36:]}
    direction = pd.Series([1.0, 0.0], index=model.contracts)
    weights, stats = select_log_growth_scale(
        model, direction, returns.iloc[-36:], scenarios, _costs(model), caps, _config()
    )
    assert 0 < stats["selected_scale"] <= 1
    assert stats["expected_net_log_growth"] > 0
    negative, negative_returns = _model(mu=(-0.10, -0.10))
    cash, cash_stats = select_log_growth_scale(
        negative, direction, negative_returns.iloc[-36:], {"recent": negative_returns.iloc[-36:]}, _costs(negative), caps, _config()
    )
    # Realized scenario log growth, not a positive arithmetic forecast, controls scale.
    assert cash_stats["selected_scale"] >= 0
    punitive = _costs(negative, call=1.0)
    cash, cash_stats = select_log_growth_scale(
        negative, direction, negative_returns.iloc[-36:], {"recent": negative_returns.iloc[-36:]}, punitive, caps, _config()
    )
    assert cash.abs().sum() == 0
    assert cash_stats["status"] == "cash_nonpositive_log_growth"


def test_direct_integer_conversion_abstains_and_preserves_rejected_diagnostics():
    model, returns = _model()
    caps = pd.Series(1.0, index=model.contracts)
    config = _config(annual_vol_target=0.0001)
    target = pd.Series([0.20, 0.0], index=model.contracts)
    result = integerize_r2_direct_or_abstain(
        model,
        target,
        pd.Series([5.0, 4.0], index=model.contracts),
        100_000.0,
        caps,
        returns.iloc[-36:],
        {"recent": returns.iloc[-36:]},
        _costs(model),
        config,
    )
    assert result.weights.abs().sum() == 0
    assert result.diagnostics["integer_execution_abstained"] is True
    assert result.diagnostics["rejected_breach_volatility"] > 0
    assert result.diagnostics["rejected_predicted_annual_vol"] > 0


def test_simulation_max_drawdown_includes_initial_nav_and_absorbs_default():
    metrics = performance_metrics(pd.Series([-0.20, 0.10]))
    assert metrics["max_drawdown"] == pytest.approx(-0.20)
    ruined = performance_metrics(pd.Series([0.10, -1.20, 1.0]))
    assert ruined["terminal_wealth"] == 0.0
    assert ruined["max_drawdown"] == -1.0


def test_locked_bootstrap_uses_independent_paths_and_preserves_pairing():
    aligned = pd.DataFrame(
        {
            "config": ["orig"] * 24,
            "r2_net_return": np.tile([0.02, -0.01, 0.01], 8),
            "r11_net_return": np.tile([0.01, -0.02, 0.00], 8),
        }
    )
    paths = circular_block_path_suite(aligned, paths=20, seed=5)
    r2 = paths[paths["strategy"] == "R2 robust net Sortino"]
    assert r2["terminal_wealth"].nunique() > 1
    paired, bounds = paired_stationary_bootstrap(aligned, paths=50, seed=6)
    assert len(paired) == 50
    assert bounds["sortino_improvement_90pct_lower"] > 0
    assert bounds["net_log_growth_improvement_90pct_lower"] > 0


def test_any_promotion_gate_failure_keeps_r11_active():
    comparison = pd.DataFrame(
        [
            {"config": name, "strategy": strategy, "sortino": value, "annualized_return": value / 10, "max_drawdown": -0.1, "cvar_95": 0.05}
            for name in ["a", "b", "c", "d"]
            for strategy, value in [("R2 robust net Sortino", 2.0), ("R1.1 25pct positive-edge deployment", 1.0)]
        ]
    )
    repriced = pd.DataFrame(
        [
            {"config": name, "strategy": strategy, "method": method, "terminal_wealth": wealth, "sortino": wealth, "max_drawdown": -0.1}
            for name in ["a", "b", "c", "d"]
            for strategy, wealth in [("R2 robust net Sortino", 2.0), ("R1.1 25pct positive-edge deployment", 1.0)]
            for method in ["joint_garch_block", "gaussian_copula"]
            for _ in range(10)
        ]
    )
    refit = pd.DataFrame({"status": ["ok"] * 19 + ["error"], "defaulted": [False] * 20})
    returns = pd.DataFrame(
        {"short_margin_used": [0.0], "collateral_used": [0.0], "selected_feasible": [True], "net_return": [0.01]}
    )
    gate = evaluate_promotion_gate(
        comparison,
        {"sortino_improvement_90pct_lower": 0.1, "net_log_growth_improvement_90pct_lower": 0.1},
        repriced,
        refit,
        returns,
    )
    assert gate["gates"]["refit_coverage_at_least_95pct"] is True
    # One refit error is exactly 95%; force a distinct gate failure.
    refit.loc[18, "status"] = "error"
    gate = evaluate_promotion_gate(
        comparison,
        {"sortino_improvement_90pct_lower": 0.1, "net_log_growth_improvement_90pct_lower": 0.1},
        repriced,
        refit,
        returns,
    )
    assert gate["promoted"] is False
    assert gate["active_development_extension"] == "R1.1"
