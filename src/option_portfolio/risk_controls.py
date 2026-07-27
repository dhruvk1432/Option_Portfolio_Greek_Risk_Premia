"""Frozen R1/R1.1 policies and fail-closed whole-contract acceptance."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
import pandas as pd

from option_portfolio.model import (
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionConstraints,
    OptionMarkowitzModel,
    OptionResult,
    OptionSpec,
    _solve_cvxpy,
    empirical_cvar_loss,
)


@dataclass(frozen=True)
class R11Policy(NetUtilityConfig):
    deployment_target: float = 0.50
    deployment_edge_floor: float = 0.0

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.deployment_target <= 1.0:
            raise ValueError("deployment_target must lie between zero and one")
        if (
            not np.isfinite(self.deployment_edge_floor)
            or self.deployment_edge_floor < 0.0
        ):
            raise ValueError("deployment_edge_floor must be finite and nonnegative")


@dataclass(frozen=True)
class IntegerExecutionResult:
    weights: pd.Series
    contracts: pd.Series
    abstained: bool
    reason: str
    rejected_diagnostics: dict[str, object] | None = None


R1_POLICY = NetUtilityConfig()
R11_POLICY = R11Policy(annual_volatility_ceiling=0.25)


def r1_constraints(
    options: OptionSpec,
    liquidity_caps: pd.Series,
) -> OptionConstraints:
    """Construct the frozen R1/R1.1 feasible-set limits."""

    options.validate()
    caps = pd.Series(liquidity_caps, dtype=float)
    labels = options.frame.index
    if caps.index.has_duplicates or set(caps.index) != set(labels):
        raise ValueError("liquidity_caps must cover exactly the option contracts")
    caps = caps.reindex(labels)
    if not np.isfinite(caps.to_numpy()).all() or (caps < 0.0).any():
        raise ValueError("liquidity_caps must be finite and nonnegative")
    underlyings = options.frame["underlying"].astype(str).unique()
    return OptionConstraints(
        gross_nav=1.0,
        net_nav_abs=1.0,
        short_nav_abs=0.25,
        per_contract_abs=0.18,
        per_contract_caps=caps,
        underlying_gross={
            name: 0.20 if name == "VX_FRONT" else 0.35
            for name in underlyings
        },
        beta_spy_abs=3.0,
        vix_vega_abs=8.0,
        stress_loss_abs=0.20,
    )


def solve_r11_net_utility(
    model: OptionMarkowitzModel,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    policy: R11Policy = R11_POLICY,
) -> OptionResult:
    """Solve R1.1 and apply its sign-restricted target only when feasible."""

    policy.validate()
    stage1 = model.solve_net_utility(scenarios, costs, policy)
    stage1_weights = stage1.weights.reindex(model.contracts).to_numpy(float)
    stage1_gross = float(np.abs(stage1_weights).sum())
    if stage1.status != "optimal":
        return _with_r11_diagnostics(
            stage1,
            target=policy.deployment_target,
            stage1_gross=stage1_gross,
            feasible=False,
            applied=False,
            eligible=0,
        )
    if stage1_gross >= policy.deployment_target - 1e-7:
        return _with_r11_diagnostics(
            stage1,
            target=policy.deployment_target,
            stage1_gross=stage1_gross,
            feasible=True,
            applied=False,
            eligible=int(np.count_nonzero(stage1_weights)),
        )
    if stage1.risk_aversion is None:
        raise RuntimeError("R1.1 requires the selected R1 risk aversion")

    returns = model._aligned_scenarios(scenarios).to_numpy(float)
    long_cost, short_cost, short_margin, short_allowed = costs.aligned(model.contracts)
    means = model.expected_returns.to_numpy(float)
    signs = np.sign(stage1_weights)
    directional_cost = np.where(
        signs > 0.0,
        long_cost,
        np.where(signs < 0.0, short_cost, 0.0),
    )
    directional_mean = signs * means
    edge = directional_mean - directional_cost
    eligible = (signs != 0.0) & (edge > policy.deployment_edge_floor + 1e-12)
    eligible_count = int(eligible.sum())
    if not eligible_count:
        return _with_r11_diagnostics(
            stage1,
            target=policy.deployment_target,
            stage1_gross=stage1_gross,
            feasible=False,
            applied=False,
            eligible=0,
        )

    size = cp.Variable(len(model.contracts), nonneg=True)
    weights = cp.multiply(signs, size)
    long = cp.multiply((signs > 0.0).astype(float), size)
    short = cp.multiply((signs < 0.0).astype(float), size)
    predictable_cost = directional_cost @ size
    constraints = model._net_utility_constraints(
        weights,
        long,
        short,
        returns,
        predictable_cost,
        policy,
        short_margin,
        short_allowed,
    )
    if (~eligible).any():
        constraints.append(size[np.flatnonzero(~eligible)] == 0.0)
    constraints.extend(
        [
            edge @ size >= 0.0,
            cp.quad_form(weights, cp.psd_wrap(model.option_covariance))
            <= (
                policy.annual_volatility_ceiling
                / np.sqrt(policy.periods_per_year)
            )
            ** 2,
            cp.sum(size) <= policy.deployment_target,
        ]
    )
    feasibility_problem = cp.Problem(cp.Maximize(cp.sum(size)), constraints)
    feasibility_solver = _solve_cvxpy(feasibility_problem)
    feasible_gross = (
        float(np.sum(size.value))
        if feasibility_solver is not None and size.value is not None
        else 0.0
    )
    if feasible_gross < policy.deployment_target - 1e-6:
        return _with_r11_diagnostics(
            stage1,
            target=policy.deployment_target,
            stage1_gross=stage1_gross,
            feasible=False,
            applied=False,
            eligible=eligible_count,
        )

    constraints.append(cp.sum(size) == policy.deployment_target)
    objective = cp.Maximize(
        directional_mean @ size
        - directional_cost @ size
        - 0.5
        * stage1.risk_aversion
        * cp.quad_form(weights, cp.psd_wrap(model.option_covariance))
    )
    target_problem = cp.Problem(objective, constraints)
    target_solver = _solve_cvxpy(target_problem)
    if target_solver is None or size.value is None:
        return _with_r11_diagnostics(
            stage1,
            target=policy.deployment_target,
            stage1_gross=stage1_gross,
            feasible=True,
            applied=False,
            eligible=eligible_count,
        )

    target = signs * np.asarray(size.value, dtype=float).ravel()
    target[np.abs(target) < 1e-9] = 0.0
    diagnostics = validate_portfolio(model, target, scenarios, costs, policy)
    diagnostics.update(
        {
            "stage1_gross_nav": stage1_gross,
            "deployment_target": policy.deployment_target,
            "deployment_target_feasible": True,
            "deployment_target_applied": True,
            "deployment_target_met": bool(
                np.abs(target).sum() >= policy.deployment_target - 1e-6
            ),
            "positive_edge_contracts": eligible_count,
        }
    )
    if not diagnostics["feasible"]:
        retained = _with_r11_diagnostics(
            stage1,
            target=policy.deployment_target,
            stage1_gross=stage1_gross,
            feasible=True,
            applied=False,
            eligible=eligible_count,
        )
        retained_diagnostics = dict(retained.diagnostics)
        retained_diagnostics["deployment_target_rejected_diagnostics"] = diagnostics
        return replace(retained, diagnostics=retained_diagnostics)
    return model._result(
        target,
        "optimal",
        float(diagnostics["max_violation"]),
        f"cvxpy_{target_solver.lower()}_r11_target",
        risk_aversion=stage1.risk_aversion,
        diagnostics=diagnostics,
    )


def validate_portfolio(
    model: OptionMarkowitzModel,
    weights: pd.Series | np.ndarray,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    policy: NetUtilityConfig,
) -> dict[str, object]:
    """Recompute every funded policy limit on one exact portfolio."""

    policy.validate()
    if isinstance(weights, pd.Series):
        aligned = weights.reindex(model.contracts).astype(float)
        if aligned.isna().any():
            raise ValueError("weights must cover every model contract")
        values = aligned.to_numpy(float)
    else:
        values = np.asarray(weights, dtype=float).ravel()
        if len(values) != len(model.contracts):
            raise ValueError("weights must cover every model contract")
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")

    returns = model._aligned_scenarios(scenarios).to_numpy(float)
    long_cost, short_cost, short_margin, short_allowed = costs.aligned(model.contracts)
    long = np.maximum(values, 0.0)
    short = np.maximum(-values, 0.0)
    constraints = model.constraints
    diagnostics: dict[str, object] = {
        "gross": max(float(np.abs(values).sum()) - constraints.gross_nav, 0.0),
        "caps": float(
            np.maximum(np.abs(values) - model._caps, 0.0).max(initial=0.0)
        ),
        "long_only": (
            max(float(-values.min(initial=0.0)), 0.0)
            if constraints.long_only
            else 0.0
        ),
        "net": (
            max(abs(float(values.sum())) - constraints.net_nav_abs, 0.0)
            if constraints.net_nav_abs is not None
            else 0.0
        ),
        "short": (
            max(float(short.sum()) - constraints.short_nav_abs, 0.0)
            if constraints.short_nav_abs is not None
            else 0.0
        ),
    }

    underlyings = model.frame["underlying"].astype(str).to_numpy()
    diagnostics["underlying"] = max(
        (
            max(
                float(np.abs(values[underlyings == name]).sum()) - limit,
                0.0,
            )
            for name, limit in constraints.underlying_gross.items()
        ),
        default=0.0,
    )
    exposure_breaches = []
    for name, vector, limit in model._exposure_limits():
        breach = max(abs(float(vector @ values)) - limit, 0.0)
        diagnostics[f"exposure_{name}"] = breach
        exposure_breaches.append(breach)
    diagnostics["exposure"] = max(exposure_breaches, default=0.0)

    stress_limit = model._effective_stress_limit(policy)
    stress = model._stress_matrix_for_limit(stress_limit)
    diagnostics["stress"] = (
        max(-stress_limit - float(np.min(stress @ values)), 0.0)
        if stress is not None
        else 0.0
    )
    margin = float(short_margin @ short)
    collateral = float(long.sum() + margin)
    diagnostics["margin"] = max(margin - policy.short_margin_nav, 0.0)
    diagnostics["collateral"] = max(
        collateral - policy.collateral_nav,
        0.0,
    )
    diagnostics["assignment"] = (
        float(np.maximum(-values[~short_allowed], 0.0).max(initial=0.0))
        if (~short_allowed).any()
        else 0.0
    )
    predictable_cost = float(long_cost @ long + short_cost @ short)
    losses = -(returns @ values - predictable_cost)
    cvar = empirical_cvar_loss(losses, policy.cvar_alpha)
    diagnostics["cvar"] = max(cvar - policy.cvar_loss_nav, 0.0)
    annual_volatility = model._volatility(values) * np.sqrt(policy.periods_per_year)
    diagnostics["annual_volatility"] = max(
        annual_volatility - policy.annual_volatility_ceiling,
        0.0,
    )
    breaches = [
        float(value)
        for value in diagnostics.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    diagnostics["max_violation"] = max(breaches, default=0.0)
    diagnostics["feasible"] = bool(diagnostics["max_violation"] <= 1e-5)
    return diagnostics


def integerize_or_cash(
    model: OptionMarkowitzModel,
    weights: pd.Series,
    marks: pd.Series,
    *,
    nav: float,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    policy: NetUtilityConfig = R1_POLICY,
    multiplier: float = 100.0,
) -> IntegerExecutionResult:
    """Use the exact signed truncation when feasible; otherwise select cash."""

    labels = model.contracts
    target = weights.reindex(labels).astype(float)
    aligned_marks = marks.reindex(labels).astype(float)
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("weights must be finite and cover every model contract")
    if nav <= 0.0 or multiplier <= 0.0:
        raise ValueError("nav and multiplier must be positive")
    if (
        aligned_marks.isna().any()
        or not np.isfinite(aligned_marks.to_numpy()).all()
        or (aligned_marks <= 0.0).any()
    ):
        raise ValueError("marks must be positive and cover every model contract")

    continuous = target * nav / (aligned_marks * multiplier)
    contracts = np.sign(continuous) * np.floor(continuous.abs())
    contracts = contracts.rename("contracts")
    realized = (contracts * aligned_marks * multiplier / nav).rename("weight")
    diagnostics = validate_portfolio(model, realized, scenarios, costs, policy)
    if diagnostics["feasible"]:
        return IntegerExecutionResult(
            weights=realized,
            contracts=contracts,
            abstained=False,
            reason="direct_truncation",
        )
    return IntegerExecutionResult(
        weights=pd.Series(0.0, index=labels, name="weight"),
        contracts=pd.Series(0.0, index=labels, name="contracts"),
        abstained=True,
        reason="infeasible_direct_book",
        rejected_diagnostics=diagnostics,
    )


def _with_r11_diagnostics(
    result: OptionResult,
    *,
    target: float,
    stage1_gross: float,
    feasible: bool,
    applied: bool,
    eligible: int,
) -> OptionResult:
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "stage1_gross_nav": stage1_gross,
            "deployment_target": target,
            "deployment_target_feasible": feasible,
            "deployment_target_applied": applied,
            "deployment_target_met": bool(
                stage1_gross >= target - 1e-6
            ),
            "positive_edge_contracts": eligible,
        }
    )
    return replace(result, diagnostics=diagnostics)
