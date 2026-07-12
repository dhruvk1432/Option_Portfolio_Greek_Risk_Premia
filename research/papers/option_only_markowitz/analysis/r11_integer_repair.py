"""Whole-contract execution policy for the R1.1 development specification.

The continuous target is truncated toward cash.  If that exact conversion is
not feasible, the strategy abstains for the period instead of substituting a
different portfolio.  The rejected conversion's diagnostics remain available
alongside the selected all-cash book.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import (
    r1_constraint_diagnostics,
)
from src.portfolio.option_only_markowitz_model import (
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionOnlyMarkowitzModel,
)

@dataclass
class IntegerRepairResult:
    """Selected whole-contract book plus an auditable conversion record."""

    weights: pd.Series
    diagnostics: dict[str, Any]
    candidates: pd.DataFrame


def _integer_context(
    model: OptionOnlyMarkowitzModel,
    continuous: pd.Series,
    marks: pd.Series,
    nav: float,
    caps: pd.Series,
    costs: OptimizationCostSpec,
    config: NetUtilityConfig,
) -> dict[str, np.ndarray]:
    labels = model.contracts
    continuous_values = continuous.reindex(labels).fillna(0.0).to_numpy(float)
    mark_values = marks.reindex(labels).to_numpy(float)
    if not np.isfinite(mark_values).all() or (mark_values <= 0).any():
        raise ValueError("R1.1 integer repair requires finite positive marks")
    unit = 100.0 * mark_values / float(nav)
    signs = np.sign(continuous_values)
    long_cost, short_cost, _, _ = costs.aligned(labels)
    directional_cost = np.where(signs > 0, long_cost, np.where(signs < 0, short_cost, 0.0))
    directional_edge = signs * model.expected_returns.reindex(labels).to_numpy(float) - directional_cost
    edge_floor = float(getattr(config, "deployment_net_edge_floor", 0.0))
    eligible = (signs != 0.0) & (directional_edge > edge_floor + 1e-12)
    cap_values = caps.reindex(labels).fillna(0.0).to_numpy(float)
    scalar_cap = model.constraints.gross_nav
    if model.constraints.per_contract_abs is not None:
        scalar_cap = min(scalar_cap, model.constraints.per_contract_abs)
    cap_values = np.minimum(cap_values, float(scalar_cap))
    upper_counts = np.floor(np.maximum(cap_values, 0.0) / unit + 1e-10).astype(int)
    upper_counts[~eligible] = 0
    continuous_counts = np.abs(continuous_values) / unit
    raw_upper_counts = np.floor(np.maximum(cap_values, 0.0) / unit + 1e-10).astype(int)
    raw_floor_counts = np.minimum(
        np.floor(continuous_counts + 1e-10).astype(int), raw_upper_counts
    )
    floor_counts = np.minimum(raw_floor_counts, upper_counts)
    floor_counts[~eligible] = 0
    return {
        "continuous_values": continuous_values,
        "continuous_counts": continuous_counts,
        "unit": unit,
        "signs": signs,
        "eligible": eligible,
        "directional_edge": directional_edge,
        "upper_counts": upper_counts,
        "raw_floor_counts": raw_floor_counts,
        "floor_counts": floor_counts,
    }


def _evaluate_candidate(
    method: str,
    status: str,
    counts: np.ndarray,
    context: dict[str, np.ndarray],
    model: OptionOnlyMarkowitzModel,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    caps: pd.Series,
    config: NetUtilityConfig,
    *,
    solver: str = "deterministic",
) -> tuple[pd.Series, dict[str, Any]]:
    counts = np.maximum(np.rint(counts), 0.0).astype(int)
    signed_counts = context["signs"] * counts
    weights = pd.Series(
        signed_counts * context["unit"],
        index=model.contracts,
        name="weight",
    )
    diagnostics = dict(r1_constraint_diagnostics(model, weights, scenarios, costs, caps, config))
    w = weights.to_numpy(float)
    monthly_variance = max(float(w @ model.option_cov @ w), 0.0)
    annual_vol = float(np.sqrt(monthly_variance * config.periods_per_year))
    vol_breach = max(annual_vol - config.annual_vol_target, 0.0)
    sign_breach = float(
        np.maximum(-signed_counts * context["signs"], 0.0).max(initial=0.0)
    )
    edge_breach = float(np.abs(w[~context["eligible"]]).max(initial=0.0))
    # A negative standalone edge is not a portfolio infeasibility: the optimizer may
    # rationally select such a leg as a hedge.  Keep it as an audit diagnostic, but
    # abstain only for an actual portfolio-constraint or volatility breach.
    max_breach = max(float(diagnostics["max_breach"]), vol_breach, sign_breach)
    long_cost, short_cost, _, _ = costs.aligned(model.contracts)
    predictable_cost = float(long_cost @ np.maximum(w, 0.0) + short_cost @ np.maximum(-w, 0.0))
    expected_net_return = float(model.expected_returns.to_numpy(float) @ w - predictable_cost)
    risk_aversion = float(context.get("risk_aversion", np.asarray(config.lambda_floor)))
    utility = expected_net_return - 0.5 * risk_aversion * monthly_variance
    distance = float(np.abs(w - context["continuous_values"]).sum())
    diagnostics.update(
        {
            "method": method,
            "method_status": status,
            "solver": solver,
            "feasible": bool(max_breach <= 1e-6),
            "max_breach": float(max_breach),
            "predicted_annual_vol": annual_vol,
            "breach_volatility": float(vol_breach),
            "breach_sign": sign_breach,
            "breach_positive_edge": edge_breach,
            "expected_net_return": expected_net_return,
            "predictable_cost": predictable_cost,
            "net_utility": float(utility),
            "continuous_l1_distance": distance,
            "integer_contracts": int(counts.sum()),
            "gross_nav": float(np.abs(w).sum()),
        }
    )
    return weights, diagnostics


def integerize_r11_weights(
    model: OptionOnlyMarkowitzModel,
    continuous: pd.Series,
    marks: pd.Series,
    nav: float,
    caps: pd.Series,
    scenarios: pd.DataFrame,
    costs: OptimizationCostSpec,
    config: NetUtilityConfig,
    *,
    risk_aversion: float,
) -> IntegerRepairResult:
    """Use the direct integer conversion when feasible and otherwise abstain."""

    context = _integer_context(model, continuous, marks, nav, caps, costs, config)
    context["risk_aversion"] = np.asarray(float(risk_aversion))
    direct_weights, direct = _evaluate_candidate(
        "truncate_toward_cash",
        "evaluated",
        context["raw_floor_counts"],
        context,
        model,
        scenarios,
        costs,
        caps,
        config,
    )
    direct["available"] = True

    cash_weights, cash = _evaluate_candidate(
        "cash_abstention",
        "available_if_direct_conversion_infeasible",
        np.zeros_like(context["raw_floor_counts"]),
        context,
        model,
        scenarios,
        costs,
        caps,
        config,
    )
    cash["available"] = True

    abstained = not bool(direct["feasible"])
    selected_weights = cash_weights if abstained else direct_weights
    selected = dict(cash if abstained else direct)
    selected = dict(selected)
    selected["integer_repair_failed_to_cash"] = False
    selected["integer_execution_abstained"] = abstained
    selected["integer_conversion_feasible"] = bool(direct["feasible"])
    selected["integer_abstention_reason"] = (
        "direct_integer_conversion_infeasible" if abstained else ""
    )
    selected["selected_integer_method"] = selected["method"]
    selected["pre_repair_feasible"] = bool(direct["feasible"])
    selected["pre_repair_max_breach"] = float(direct["max_breach"])
    if abstained:
        selected["best_failed_method"] = direct["method"]
        for key, value in direct.items():
            if key.startswith("breach_") or key in {
                "max_breach",
                "scenario_cvar_loss",
                "worst_stress_return",
                "short_margin_used",
                "collateral_used",
                "predicted_annual_vol",
            }:
                selected[f"failed_{key}"] = value
    else:
        selected["best_failed_method"] = ""

    candidate_frame = pd.DataFrame([direct, cash])
    candidate_frame["selected"] = candidate_frame["method"].eq(selected["selected_integer_method"])
    return IntegerRepairResult(selected_weights, selected, candidate_frame)
