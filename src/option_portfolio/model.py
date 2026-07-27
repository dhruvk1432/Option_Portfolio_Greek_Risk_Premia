"""Funded option-only Markowitz model with complete covariance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import optimize

REQUIRED_OPTION_COLUMNS = (
    "underlying",
    "mark",
    "spot",
    "delta",
    "gamma",
    "vega",
    "theta",
)


@dataclass(frozen=True)
class OptionSpec:
    frame: pd.DataFrame

    def validate(self) -> None:
        missing = [name for name in REQUIRED_OPTION_COLUMNS if name not in self.frame]
        if missing:
            raise ValueError(f"OptionSpec.frame missing columns: {missing}")
        if self.frame.empty or self.frame.index.has_duplicates:
            raise ValueError("OptionSpec.frame must have unique contract identifiers")
        for name in ("mark", "spot"):
            values = pd.to_numeric(self.frame[name], errors="coerce").to_numpy(float)
            if not (np.isfinite(values) & (values > 0.0)).all():
                raise ValueError(f"{name} must be finite and positive")
        for name in ("delta", "gamma", "vega", "theta"):
            values = pd.to_numeric(self.frame[name], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class FactorShockSpec:
    underlying_cov: pd.DataFrame
    vol_cov: pd.DataFrame | None = None
    gamma_var_cov: pd.DataFrame | None = None
    spot_vol_cov: pd.DataFrame | None = None

    def validate(self, underlyings: Sequence[str]) -> None:
        _validate_covariance(self.underlying_cov, "underlying_cov")
        required = set(underlyings)
        for name, frame in (
            ("underlying_cov", self.underlying_cov),
            ("vol_cov", self.vol_cov),
            ("gamma_var_cov", self.gamma_var_cov),
        ):
            if frame is not None:
                _validate_covariance(frame, name)
                if not required.issubset(frame.index):
                    raise ValueError(f"{name} missing underlyings")
        if self.spot_vol_cov is not None:
            frame = self.spot_vol_cov
            if not required.issubset(frame.index) or not required.issubset(frame.columns):
                raise ValueError("spot_vol_cov missing underlyings")
            if not np.isfinite(frame.to_numpy(float)).all():
                raise ValueError("spot_vol_cov must be finite")


@dataclass(frozen=True)
class GreekJointMomentSpec:
    factor_cov: pd.DataFrame
    factor_residual_cov: pd.DataFrame
    residual_cov: pd.DataFrame
    n_obs: int
    estimator: str = "sample"

    def validate(self, factors: Sequence[str], contracts: Sequence[str]) -> None:
        _validate_covariance(self.factor_cov, "factor_cov")
        _validate_covariance(self.residual_cov, "residual_cov")
        if list(self.factor_cov.index) != list(factors):
            raise ValueError("factor_cov labels must match the exposure matrix")
        if list(self.residual_cov.index) != list(contracts):
            raise ValueError("residual_cov labels must match contracts")
        if list(self.factor_residual_cov.index) != list(factors):
            raise ValueError("factor_residual_cov rows must match factors")
        if list(self.factor_residual_cov.columns) != list(contracts):
            raise ValueError("factor_residual_cov columns must match contracts")
        cross = self.factor_residual_cov.to_numpy(float)
        if not np.isfinite(cross).all():
            raise ValueError("factor_residual_cov must be finite")
        joint = np.block(
            [
                [self.factor_cov.to_numpy(float), cross],
                [cross.T, self.residual_cov.to_numpy(float)],
            ]
        )
        if np.linalg.eigvalsh(joint).min() < -1e-8:
            raise ValueError("joint factor-residual covariance must be positive semidefinite")
        if self.n_obs < 2:
            raise ValueError("joint moments require at least two observations")


@dataclass(frozen=True)
class OptionConstraints:
    gross_nav: float = 1.0
    net_nav_abs: float | None = None
    short_nav_abs: float | None = None
    per_contract_abs: float | None = None
    per_contract_caps: pd.Series | None = None
    underlying_gross: dict[str, float] = field(default_factory=dict)
    delta_abs: float | None = None
    gamma_abs: float | None = None
    vega_abs: float | None = None
    vix_vega_abs: float | None = None
    beta_spy_abs: float | None = None
    stress_loss_abs: float | None = None
    long_only: bool = False

    def validate(self) -> None:
        if not np.isfinite(self.gross_nav) or self.gross_nav <= 0.0:
            raise ValueError("gross_nav must be finite and positive")
        values = (
            self.net_nav_abs,
            self.short_nav_abs,
            self.per_contract_abs,
            self.delta_abs,
            self.gamma_abs,
            self.vega_abs,
            self.vix_vega_abs,
            self.beta_spy_abs,
            self.stress_loss_abs,
            *self.underlying_gross.values(),
        )
        if any(value is not None and (not np.isfinite(value) or value < 0.0) for value in values):
            raise ValueError("constraint limits must be finite and nonnegative")
        if self.per_contract_caps is not None:
            caps = pd.Series(self.per_contract_caps, dtype=float)
            if caps.index.has_duplicates:
                raise ValueError("per_contract_caps index must be unique")
            if not np.isfinite(caps.to_numpy()).all() or (caps < 0.0).any():
                raise ValueError("per_contract_caps must be finite and nonnegative")


@dataclass(frozen=True)
class OptimizationCostSpec:
    long_cost: pd.Series
    short_cost: pd.Series
    short_margin: pd.Series
    short_allowed: pd.Series

    def aligned(self, contracts: Sequence[str]) -> tuple[np.ndarray, ...]:
        labels = list(contracts)
        frames = [
            self.long_cost.reindex(labels).astype(float),
            self.short_cost.reindex(labels).astype(float),
            self.short_margin.reindex(labels).astype(float),
        ]
        allowed = self.short_allowed.reindex(labels)
        if any(frame.isna().any() for frame in frames) or allowed.isna().any():
            raise ValueError("cost inputs must cover every model contract")
        if any(not np.isfinite(frame.to_numpy(float)).all() for frame in frames):
            raise ValueError("cost and margin inputs must be finite")
        if any((frame < 0.0).any() for frame in frames):
            raise ValueError("cost and margin inputs must be nonnegative")
        if not pd.api.types.is_bool_dtype(allowed.dtype):
            raise ValueError("short_allowed must contain boolean values")
        return *(frame.to_numpy(float) for frame in frames), allowed.to_numpy(dtype=bool)


@dataclass(frozen=True)
class NetUtilityConfig:
    annual_volatility_ceiling: float = 0.15
    periods_per_year: float = 12.0
    cvar_alpha: float = 0.95
    cvar_loss_nav: float = 0.10
    stress_loss_nav: float = 0.20
    short_margin_nav: float = 0.75
    collateral_nav: float = 1.0
    lambda_floor: float = 1e-6
    lambda_ceiling: float = 1e6
    bisection_steps: int = 18

    def validate(self) -> None:
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie between zero and one")
        positive = (
            self.annual_volatility_ceiling,
            self.periods_per_year,
            self.lambda_floor,
            self.lambda_ceiling,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("volatility, annualization, and lambda limits must be positive")
        nonnegative = (
            self.cvar_loss_nav,
            self.stress_loss_nav,
            self.short_margin_nav,
            self.collateral_nav,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("funding and loss limits must be finite and nonnegative")
        if self.lambda_floor > self.lambda_ceiling:
            raise ValueError("lambda bounds must be ordered")
        if self.bisection_steps < 1:
            raise ValueError("bisection_steps must be positive")


@dataclass(frozen=True)
class OptionResult:
    status: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    gross_nav: float
    net_nav: float
    max_violation: float
    solver: str
    risk_aversion: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def shrink_covariance(matrix: np.ndarray, shrinkage: float) -> np.ndarray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie between zero and one")
    values = nearest_psd(matrix)
    target = np.diag(np.diag(values))
    return nearest_psd((1.0 - shrinkage) * values + shrinkage * target)


def _ledoit_wolf_covariance(centered: np.ndarray) -> np.ndarray:
    """Return Ledoit-Wolf covariance for an already centered observation matrix."""

    values = np.asarray(centered, dtype=float)
    n_samples, n_features = values.shape
    empirical = values.T @ values / n_samples
    mu = float(np.trace(empirical) / n_features)
    squared = values**2
    beta_sum = float(np.sum(squared.T @ squared))
    delta_sum = float(np.sum(empirical**2))
    beta = (beta_sum / n_samples - delta_sum) / (n_features * n_samples)
    delta = (delta_sum - 2.0 * mu * np.trace(empirical) + n_features * mu**2)
    delta /= n_features
    shrinkage = 0.0 if delta <= 0.0 else min(max(beta, 0.0), delta) / delta
    covariance = (1.0 - shrinkage) * empirical
    covariance.flat[:: n_features + 1] += shrinkage * mu
    return 0.5 * (covariance + covariance.T)


def empirical_cvar_loss(losses: np.ndarray, alpha: float) -> float:
    """Return the finite-sample CVaR represented by the optimization epigraph."""

    values = np.asarray(losses, dtype=float).ravel()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("losses must be a nonempty finite vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie between zero and one")
    tail_count = (1.0 - alpha) * len(values)
    nearest_integer = round(tail_count)
    if math.isclose(tail_count, nearest_integer, rel_tol=0.0, abs_tol=1e-12):
        tail_count = float(nearest_integer)
    whole = int(math.floor(tail_count))
    fraction = tail_count - whole
    ordered = np.sort(values)[::-1]
    total = float(ordered[:whole].sum())
    if fraction > 0.0:
        total += fraction * float(ordered[whole])
    return total / tail_count


def greek_factor_names(underlyings: Sequence[str]) -> list[str]:
    labels = [str(name) for name in underlyings]
    return [
        *(f"r_{name}" for name in labels),
        *(f"r2_{name}" for name in labels),
        *(f"dv_{name}" for name in labels),
    ]


def greek_exposure_frame(options: OptionSpec) -> pd.DataFrame:
    options.validate()
    frame = options.frame
    underlyings = sorted(frame["underlying"].astype(str).unique())
    factors = greek_factor_names(underlyings)
    exposure = pd.DataFrame(0.0, index=frame.index, columns=factors)
    for underlying in underlyings:
        mask = frame["underlying"].astype(str).eq(underlying)
        mark = frame.loc[mask, "mark"].astype(float)
        spot = frame.loc[mask, "spot"].astype(float)
        exposure.loc[mask, f"r_{underlying}"] = frame.loc[mask, "delta"] * spot / mark
        exposure.loc[mask, f"r2_{underlying}"] = 0.5 * frame.loc[mask, "gamma"] * spot**2 / mark
        exposure.loc[mask, f"dv_{underlying}"] = frame.loc[mask, "vega"] / mark
    return exposure


def risk_exposure_frame(options: OptionSpec) -> pd.DataFrame:
    """Return NAV-normalized Greeks used by portfolio risk limits."""

    options.validate()
    frame = options.frame
    mark = frame["mark"].to_numpy(float)
    spot = frame["spot"].to_numpy(float)
    underlyings = frame["underlying"].astype(str).str.upper()
    asset_class = frame.get(
        "asset_class",
        pd.Series("equity_option", index=frame.index),
    ).astype(str).str.lower()
    is_vix = underlyings.isin({"VX_FRONT", "VIX", "VIX_OPTION"}) | asset_class.eq(
        "vix_option"
    )
    exposure = pd.DataFrame(
        {
            "delta_nav": frame["delta"].to_numpy(float) * spot / mark,
            "gamma_nav": frame["gamma"].to_numpy(float) * spot**2 / mark,
            "vega_nav": frame["vega"].to_numpy(float) / mark,
        },
        index=frame.index,
    )
    exposure["vix_vega_nav"] = exposure["vega_nav"].where(is_vix.to_numpy(), 0.0)
    if "beta_spy_nav" in frame:
        exposure["beta_spy_nav"] = pd.to_numeric(
            frame["beta_spy_nav"], errors="coerce"
        ).to_numpy(float)
    elif "underlying_beta_spy" in frame:
        beta = pd.to_numeric(
            frame["underlying_beta_spy"], errors="coerce"
        ).to_numpy(float)
        exposure["beta_spy_nav"] = exposure["delta_nav"].to_numpy(float) * beta
    if not np.isfinite(exposure.to_numpy(float)).all():
        raise ValueError("risk exposures must be finite")
    return exposure


def estimate_greek_joint_moments(
    option_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    regularize: bool = True,
) -> GreekJointMomentSpec:
    """Estimate the complete covariance of Greek factors and residual returns."""

    common = option_returns.index.intersection(factor_returns.index)
    contracts = list(exposure.index)
    factors = list(exposure.columns)
    states = factor_returns.reindex(index=common, columns=factors).to_numpy(float)
    options = option_returns.reindex(index=common, columns=contracts).to_numpy(float)
    complete = np.isfinite(states).all(axis=1) & np.isfinite(options).all(axis=1)
    states = states[complete]
    options = options[complete]
    if len(states) < 2:
        raise ValueError("joint moments require at least two complete observations")
    states -= states.mean(axis=0, keepdims=True)
    options -= options.mean(axis=0, keepdims=True)
    loadings = exposure.reindex(index=contracts, columns=factors).to_numpy(float)
    residuals = options - states @ loadings.T
    joint_observations = np.column_stack([states, residuals])
    if regularize:
        scales = joint_observations.std(axis=0, ddof=1)
        safe_scales = np.where(scales > 1e-14, scales, 1.0)
        standardized = joint_observations / safe_scales
        joint = _ledoit_wolf_covariance(standardized)
        joint *= scales[:, None] * scales[None, :]
        estimator = "ledoit_wolf_correlation"
    else:
        joint = joint_observations.T @ joint_observations / (len(states) - 1)
        estimator = "sample"
    joint = 0.5 * (joint + joint.T)
    split = len(factors)
    moments = GreekJointMomentSpec(
        factor_cov=pd.DataFrame(
            joint[:split, :split],
            index=factors,
            columns=factors,
        ),
        factor_residual_cov=pd.DataFrame(
            joint[:split, split:],
            index=factors,
            columns=contracts,
        ),
        residual_cov=pd.DataFrame(
            joint[split:, split:],
            index=contracts,
            columns=contracts,
        ),
        n_obs=len(states),
        estimator=estimator,
    )
    moments.validate(factors, contracts)
    return moments


class OptionMarkowitzModel:
    """Stateful optimizer over a validated option universe."""

    def __init__(
        self,
        options: OptionSpec,
        shocks: FactorShockSpec,
        expected_returns: pd.Series,
        *,
        residual_cov: pd.DataFrame | None = None,
        constraints: OptionConstraints | None = None,
        covariance_shrinkage: float = 0.10,
        joint_moments: GreekJointMomentSpec | None = None,
    ) -> None:
        options.validate()
        self.options = options
        self.frame = options.frame.copy()
        self.contracts = list(self.frame.index)
        self.underlyings = sorted(self.frame["underlying"].astype(str).unique())
        shocks.validate(self.underlyings)
        self.shocks = shocks
        self.constraints = constraints or OptionConstraints()
        self.constraints.validate()
        self.expected_returns = expected_returns.reindex(self.contracts).astype(float)
        if self.expected_returns.isna().any():
            raise ValueError("expected_returns must cover every model contract")
        if not np.isfinite(self.expected_returns.to_numpy(float)).all():
            raise ValueError("expected_returns must be finite")
        self.exposure = greek_exposure_frame(options)
        self.risk_exposure = risk_exposure_frame(options)
        self.factor_names = list(self.exposure.columns)
        self.factor_covariance = self._factor_covariance()
        self.factor_residual_covariance = np.zeros(
            (len(self.factor_names), len(self.contracts))
        )

        if joint_moments is not None:
            if residual_cov is not None:
                raise ValueError("provide joint_moments or residual_cov, not both")
            joint_moments.validate(self.factor_names, self.contracts)
            self.factor_covariance = joint_moments.factor_cov.to_numpy(float)
            self.factor_residual_covariance = joint_moments.factor_residual_cov.to_numpy(float)
            residual = joint_moments.residual_cov.to_numpy(float)
        elif residual_cov is None:
            systematic = (
                self.exposure.to_numpy(float)
                @ self.factor_covariance
                @ self.exposure.to_numpy(float).T
            )
            residual = np.diag(np.maximum(np.diag(systematic) * 0.05, 1e-8))
        else:
            aligned = residual_cov.reindex(index=self.contracts, columns=self.contracts)
            if aligned.isna().any().any():
                raise ValueError("residual_cov must cover every model contract")
            residual = aligned.to_numpy(float)
        self.residual_covariance = (
            np.array(residual, dtype=float, copy=True)
            if joint_moments is not None
            else nearest_psd(residual)
        )

        b = self.exposure.to_numpy(float)
        covariance = (
            b @ self.factor_covariance @ b.T
            + b @ self.factor_residual_covariance
            + self.factor_residual_covariance.T @ b.T
            + self.residual_covariance
        )
        self.option_covariance = (
            nearest_psd(covariance)
            if joint_moments is not None
            else shrink_covariance(covariance, covariance_shrinkage)
        )
        self._caps = self._aligned_caps()
        self._exposure_limits()
        if (
            self.constraints.stress_loss_abs is not None
            and self._stress_matrix() is None
        ):
            raise ValueError(
                "stress_loss_abs requires at least one stress_scenario_ input"
            )

    def _factor_covariance(self) -> np.ndarray:
        k = len(self.underlyings)
        underlying = self.shocks.underlying_cov.reindex(
            index=self.underlyings, columns=self.underlyings
        ).to_numpy(float)
        gamma = (
            2.0 * underlying**2
            if self.shocks.gamma_var_cov is None
            else self.shocks.gamma_var_cov.reindex(
                index=self.underlyings, columns=self.underlyings
            ).to_numpy(float)
        )
        volatility = (
            np.diag(np.maximum(np.diag(underlying) * 0.25, 1e-8))
            if self.shocks.vol_cov is None
            else self.shocks.vol_cov.reindex(
                index=self.underlyings, columns=self.underlyings
            ).to_numpy(float)
        )
        covariance = np.zeros((3 * k, 3 * k))
        covariance[:k, :k] = underlying
        covariance[k : 2 * k, k : 2 * k] = gamma
        covariance[2 * k :, 2 * k :] = volatility
        if self.shocks.spot_vol_cov is not None:
            cross = self.shocks.spot_vol_cov.reindex(
                index=self.underlyings, columns=self.underlyings
            ).to_numpy(float)
            covariance[:k, 2 * k :] = cross
            covariance[2 * k :, :k] = cross.T
        return nearest_psd(covariance)

    def _aligned_caps(self) -> np.ndarray:
        scalar = (
            self.constraints.gross_nav
            if self.constraints.per_contract_abs is None
            else self.constraints.per_contract_abs
        )
        caps = pd.Series(float(scalar), index=self.contracts)
        if self.constraints.per_contract_caps is not None:
            specified = pd.Series(self.constraints.per_contract_caps, dtype=float).reindex(
                self.contracts
            )
            if specified.isna().any():
                raise ValueError("per_contract_caps must cover every model contract")
            caps = pd.concat([caps, specified], axis=1).min(axis=1)
        return caps.to_numpy(float)

    def _volatility(self, weights: np.ndarray) -> float:
        return math.sqrt(
            max(float(weights @ self.option_covariance @ weights), 0.0)
        )

    def tangency_weights(self) -> pd.Series:
        raw = np.linalg.pinv(self.option_covariance) @ self.expected_returns.to_numpy(float)
        gross = float(np.abs(raw).sum())
        weights = (
            np.zeros_like(raw)
            if gross <= 1e-14
            else raw * self.constraints.gross_nav / gross
        )
        return pd.Series(weights, index=self.contracts, name="weight")

    def solve_max_sharpe(self) -> OptionResult:
        """Solve the legacy E1 constrained maximum-Sharpe problem."""

        covariance = self.option_covariance
        mean = self.expected_returns.to_numpy(float)

        def objective(weights: np.ndarray) -> float:
            volatility = math.sqrt(max(float(weights @ covariance @ weights), 1e-16))
            return -float(mean @ weights) / volatility

        constraints = [
            {
                "type": "eq",
                "fun": lambda weights: np.abs(weights).sum()
                - self.constraints.gross_nav,
            },
            *self._scipy_constraints(),
        ]
        bounds = [
            (0.0 if self.constraints.long_only else -cap, cap) for cap in self._caps
        ]
        start = np.clip(self.tangency_weights().to_numpy(float), -self._caps, self._caps)
        if np.abs(start).sum() <= 1e-12:
            start = np.minimum(
                self._caps,
                self.constraints.gross_nav / len(self.contracts),
            )
        gross = float(np.abs(start).sum())
        if gross > 0.0:
            start *= min(1.0, self.constraints.gross_nav / gross)
        solved = optimize.minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        weights = np.asarray(solved.x, dtype=float)
        violation = self._max_constraint_violation(weights, require_full_gross=True)
        status = "optimal" if solved.success and violation <= 1e-5 else "infeasible"
        return self._result(weights, status, violation, "scipy_slsqp")

    def solve_net_utility(
        self,
        scenarios: pd.DataFrame,
        costs: OptimizationCostSpec,
        config: NetUtilityConfig | None = None,
    ) -> OptionResult:
        """Solve R1 and select the smallest risk aversion below its volatility ceiling."""

        config = config or NetUtilityConfig()
        config.validate()
        returns = self._aligned_scenarios(scenarios)
        return_matrix = returns.to_numpy(float)
        long_cost, short_cost, short_margin, short_allowed = costs.aligned(self.contracts)
        weights = cp.Variable(len(self.contracts))
        long = cp.pos(weights)
        short = cp.pos(-weights)
        risk_aversion = cp.Parameter(nonneg=True)
        predictable_cost = long_cost @ long + short_cost @ short
        objective = cp.Maximize(
            self.expected_returns.to_numpy(float) @ weights
            - predictable_cost
            - 0.5
            * risk_aversion
            * cp.quad_form(weights, cp.psd_wrap(self.option_covariance))
        )
        stress_limit = self._effective_stress_limit(config)
        constraints = self._net_utility_constraints(
            weights,
            long,
            short,
            return_matrix,
            predictable_cost,
            config,
            short_margin,
            short_allowed,
        )
        problem = cp.Problem(objective, constraints)

        def solve_at(value: float) -> tuple[np.ndarray, str] | None:
            risk_aversion.value = value
            solver = _solve_cvxpy(problem)
            if solver is None or weights.value is None:
                return None
            solved = np.asarray(weights.value, dtype=float).ravel()
            solved[np.abs(solved) < 1e-9] = 0.0
            return solved, solver

        low = config.lambda_floor
        high = config.lambda_ceiling
        selected = solve_at(low)
        if selected is None:
            raise RuntimeError("net-utility solve failed at the lambda floor")
        selected_lambda = low
        target = config.annual_volatility_ceiling / math.sqrt(config.periods_per_year)
        if self._volatility(selected[0]) > target + 1e-8:
            upper = solve_at(high)
            if upper is None:
                raise RuntimeError("net-utility solve failed at the lambda ceiling")
            if self._volatility(upper[0]) > target + 1e-7:
                raise RuntimeError(
                    "lambda ceiling does not achieve the volatility ceiling"
                )
            selected = upper
            selected_lambda = high
            for _ in range(config.bisection_steps):
                midpoint = math.sqrt(low * high)
                candidate = solve_at(midpoint)
                if candidate is None:
                    raise RuntimeError("net-utility solve failed during lambda search")
                if self._volatility(candidate[0]) > target:
                    low = midpoint
                else:
                    high = midpoint
                    selected = candidate
                    selected_lambda = high

        value, solver = selected
        long_value = np.maximum(value, 0.0)
        short_value = np.maximum(-value, 0.0)
        violation = self._max_constraint_violation(
            value,
            stress_limit=stress_limit,
        )
        violation = max(
            violation,
            float(short_margin @ short_value - config.short_margin_nav),
            float(long_value.sum() + short_margin @ short_value - config.collateral_nav),
        )
        if (~short_allowed).any():
            violation = max(
                violation,
                float(np.maximum(-value[~short_allowed], 0.0).max(initial=0.0)),
            )
        realized_cost = float(long_cost @ long_value + short_cost @ short_value)
        realized_losses = -(return_matrix @ value - realized_cost)
        realized_cvar = empirical_cvar_loss(realized_losses, config.cvar_alpha)
        violation = max(violation, realized_cvar - config.cvar_loss_nav)
        violation = max(violation, self._volatility(value) - target, 0.0)
        status = "optimal" if violation <= 1e-5 else "infeasible"
        return self._result(
            value,
            status,
            violation,
            f"cvxpy_{solver.lower()}",
            risk_aversion=selected_lambda,
        )

    def _aligned_scenarios(self, scenarios: pd.DataFrame) -> pd.DataFrame:
        aligned = scenarios.reindex(columns=self.contracts)
        missing = [name for name in self.contracts if aligned[name].isna().all()]
        if missing:
            raise ValueError(f"scenarios has no observations for contracts: {missing}")
        returns = aligned.dropna(how="all").fillna(0.0)
        if len(returns) < 2:
            raise ValueError("scenarios require at least two observations")
        if not np.isfinite(returns.to_numpy(float)).all():
            raise ValueError("scenarios must be finite after alignment")
        return returns

    def _effective_stress_limit(self, config: NetUtilityConfig) -> float:
        model_limit = self.constraints.stress_loss_abs
        return (
            config.stress_loss_nav
            if model_limit is None
            else min(config.stress_loss_nav, model_limit)
        )

    def _scipy_constraints(self) -> list[dict[str, object]]:
        constraints: list[dict[str, object]] = []
        c = self.constraints
        if c.net_nav_abs is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w: c.net_nav_abs - abs(w.sum())}
            )
        if c.short_nav_abs is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: c.short_nav_abs - np.maximum(-w, 0.0).sum(),
                }
            )
        for underlying, limit in c.underlying_gross.items():
            mask = self.frame["underlying"].astype(str).eq(underlying).to_numpy()
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, mask=mask, limit=limit: limit
                    - np.abs(w[mask]).sum(),
                }
            )
        for _, vector, limit in self._exposure_limits():
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, vector=vector, limit=limit: limit
                    - abs(vector @ w),
                }
            )
        stress = self._stress_matrix()
        if stress is not None and c.stress_loss_abs is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: float(np.min(stress @ w)) + c.stress_loss_abs,
                }
            )
        return constraints

    def _cvxpy_constraints(
        self,
        weights,
        long,
        short,
        *,
        stress_limit: float | None = None,
    ) -> list:
        c = self.constraints
        constraints = [cp.norm1(weights) <= c.gross_nav, cp.abs(weights) <= self._caps]
        if c.long_only:
            constraints.append(weights >= 0.0)
        if c.net_nav_abs is not None:
            constraints.append(cp.abs(cp.sum(weights)) <= c.net_nav_abs)
        if c.short_nav_abs is not None:
            constraints.append(cp.sum(short) <= c.short_nav_abs)
        underlyings = self.frame["underlying"].astype(str).to_numpy()
        for underlying, limit in c.underlying_gross.items():
            indices = np.flatnonzero(underlyings == underlying)
            if len(indices):
                constraints.append(cp.norm1(weights[indices]) <= limit)
        for _, vector, limit in self._exposure_limits():
            constraints.append(cp.abs(vector @ weights) <= limit)
        limit = c.stress_loss_abs if stress_limit is None else stress_limit
        stress = self._stress_matrix_for_limit(limit)
        if stress is not None and limit is not None:
            constraints.append(stress @ weights >= -limit)
        return constraints

    def _net_utility_constraints(
        self,
        weights,
        long,
        short,
        scenario_returns: np.ndarray,
        predictable_cost,
        config: NetUtilityConfig,
        short_margin: np.ndarray,
        short_allowed: np.ndarray,
    ) -> list:
        constraints = self._cvxpy_constraints(
            weights,
            long,
            short,
            stress_limit=self._effective_stress_limit(config),
        )
        constraints.extend(
            [
                short_margin @ short <= config.short_margin_nav,
                cp.sum(long) + short_margin @ short <= config.collateral_nav,
            ]
        )
        if (~short_allowed).any():
            constraints.append(weights[np.flatnonzero(~short_allowed)] >= 0.0)
        losses = -(scenario_returns @ weights - predictable_cost)
        threshold = cp.Variable()
        cvar = threshold + cp.sum(cp.pos(losses - threshold)) / (
            (1.0 - config.cvar_alpha) * len(scenario_returns)
        )
        constraints.append(cvar <= config.cvar_loss_nav)
        return constraints

    def _exposure_limits(self) -> list[tuple[str, np.ndarray, float]]:
        c = self.constraints
        vectors = self.risk_exposure
        limits = [
            ("delta", vectors["delta_nav"].to_numpy(float), c.delta_abs),
            ("gamma", vectors["gamma_nav"].to_numpy(float), c.gamma_abs),
            ("vega", vectors["vega_nav"].to_numpy(float), c.vega_abs),
            ("vix_vega", vectors["vix_vega_nav"].to_numpy(float), c.vix_vega_abs),
        ]
        if c.beta_spy_abs is not None:
            if "beta_spy_nav" not in vectors:
                raise ValueError(
                    "beta_spy_abs requires beta_spy_nav or underlying_beta_spy inputs"
                )
            limits.append(
                (
                    "beta_spy",
                    vectors["beta_spy_nav"].to_numpy(float),
                    c.beta_spy_abs,
                )
            )
        return [
            (name, vector, float(limit))
            for name, vector, limit in limits
            if limit is not None
        ]

    def _stress_matrix(self) -> np.ndarray | None:
        columns = [name for name in self.frame if name.startswith("stress_scenario_")]
        if not columns:
            return None
        matrix = self.frame[columns].to_numpy(float).T
        if not np.isfinite(matrix).all():
            raise ValueError("stress scenario inputs must be finite")
        return matrix

    def _stress_matrix_for_limit(self, limit: float | None) -> np.ndarray | None:
        stress = self._stress_matrix()
        if limit is not None and stress is None:
            raise ValueError(
                "active stress-loss limit requires at least one stress_scenario_ input"
            )
        return stress

    def _max_constraint_violation(
        self,
        weights: np.ndarray,
        *,
        require_full_gross: bool = False,
        stress_limit: float | None = None,
    ) -> float:
        w = np.asarray(weights, dtype=float)
        c = self.constraints
        gross = float(np.abs(w).sum())
        violations = [max(0.0, gross - c.gross_nav)]
        if require_full_gross:
            violations.append(abs(gross - c.gross_nav))
        violations.append(float(np.maximum(np.abs(w) - self._caps, 0.0).max(initial=0.0)))
        if c.long_only:
            violations.append(max(0.0, float(-w.min())))
        if c.net_nav_abs is not None:
            violations.append(max(0.0, abs(float(w.sum())) - c.net_nav_abs))
        if c.short_nav_abs is not None:
            violations.append(max(0.0, float(np.maximum(-w, 0.0).sum()) - c.short_nav_abs))
        underlyings = self.frame["underlying"].astype(str).to_numpy()
        for underlying, limit in c.underlying_gross.items():
            violations.append(max(0.0, float(np.abs(w[underlyings == underlying]).sum()) - limit))
        for _, vector, limit in self._exposure_limits():
            violations.append(max(0.0, abs(float(vector @ w)) - limit))
        limit = c.stress_loss_abs if stress_limit is None else stress_limit
        stress = self._stress_matrix_for_limit(limit)
        if stress is not None and limit is not None:
            violations.append(max(0.0, -limit - float(np.min(stress @ w))))
        return float(max(violations))

    def _result(
        self,
        weights: np.ndarray,
        status: str,
        violation: float,
        solver: str,
        *,
        risk_aversion: float | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> OptionResult:
        mean = float(self.expected_returns.to_numpy(float) @ weights)
        volatility = math.sqrt(max(float(weights @ self.option_covariance @ weights), 0.0))
        sharpe = (
            mean / volatility
            if status == "optimal" and volatility > 0.0
            else math.nan
        )
        return OptionResult(
            status=status,
            weights=pd.Series(weights, index=self.contracts, name="weight"),
            expected_return=mean,
            volatility=volatility,
            sharpe=sharpe,
            gross_nav=float(np.abs(weights).sum()),
            net_nav=float(weights.sum()),
            max_violation=float(violation),
            solver=solver,
            risk_aversion=risk_aversion,
            diagnostics={} if diagnostics is None else diagnostics,
        )


def _validate_covariance(frame: pd.DataFrame, name: str) -> None:
    if list(frame.index) != list(frame.columns):
        raise ValueError(f"{name} must have identical index and columns")
    values = frame.to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values, values.T, atol=1e-10):
        raise ValueError(f"{name} must be finite and symmetric")
    if np.linalg.eigvalsh(values).min(initial=0.0) < -1e-8:
        raise ValueError(f"{name} must be positive semidefinite")


def _solve_cvxpy(problem: cp.Problem) -> str | None:
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return solver
    return None
