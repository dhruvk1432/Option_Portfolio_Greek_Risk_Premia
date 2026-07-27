"""Public option-portfolio research API."""

from .execution import ExecutionScenarioResult, execution_scenarios
from .metrics import performance_metrics, validation_status
from .model import (
    FactorShockSpec,
    GreekJointMomentSpec,
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionConstraints,
    OptionMarkowitzModel,
    OptionResult,
    OptionSpec,
    estimate_greek_joint_moments,
    greek_exposure_frame,
    greek_factor_names,
    nearest_psd,
    risk_exposure_frame,
    shrink_covariance,
)
from .pricing import bs_greeks, bs_price
from .risk_controls import (
    R1_POLICY,
    R11_POLICY,
    IntegerExecutionResult,
    R11Policy,
    integerize_or_cash,
    r1_constraints,
    solve_r11_net_utility,
    validate_portfolio,
)

__all__ = [
    "ExecutionScenarioResult",
    "FactorShockSpec",
    "GreekJointMomentSpec",
    "IntegerExecutionResult",
    "NetUtilityConfig",
    "OptimizationCostSpec",
    "OptionConstraints",
    "OptionMarkowitzModel",
    "OptionResult",
    "OptionSpec",
    "R1_POLICY",
    "R11_POLICY",
    "R11Policy",
    "bs_greeks",
    "bs_price",
    "estimate_greek_joint_moments",
    "execution_scenarios",
    "greek_exposure_frame",
    "greek_factor_names",
    "integerize_or_cash",
    "nearest_psd",
    "performance_metrics",
    "r1_constraints",
    "risk_exposure_frame",
    "shrink_covariance",
    "solve_r11_net_utility",
    "validate_portfolio",
    "validation_status",
]
