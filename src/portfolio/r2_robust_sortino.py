"""Core estimators and optimizer for the R2 robust-Sortino development arm.

R2 is intentionally separate from the frozen R1 and R1.1 implementations.
The direction problem is homogeneous; portfolio size is selected afterwards by
net log growth subject to all operational and survival constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import (
    r1_constraint_diagnostics,
)
from src.portfolio.option_only_markowitz_model import (
    GreekJointMomentSpec,
    NetUtilityConfig,
    OptimizationCostSpec,
    OptionOnlyMarkowitzModel,
    estimate_greek_joint_moments,
    nearest_psd,
)


@dataclass(frozen=True)
class RobustSortinoConfig(NetUtilityConfig):
    """Prespecified R2 risk and estimation policy."""

    annual_vol_target: float = 0.25
    annual_downside_target: float = 0.10
    recent_months: int = 36
    min_recent_observations: int = 24
    premia_half_life_months: float = 36.0
    recent_weights: tuple[float, ...] = (0.25, 0.50, 0.75)
    default_recent_weight: float = 0.50
    min_inner_forecasts: int = 12
    imputation_sets: int = 5
    bootstrap_scenarios: int = 500
    bootstrap_block_months: int = 6
    daily_window: int = 756
    min_daily_observations: int = 500
    volatility_horizon_days: int = 21
    volatility_blend_weights: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    variance_ratio_floor: float = 0.67
    variance_ratio_ceiling: float = 1.50
    max_three_month_loss: float = 0.15
    max_six_month_loss: float = 0.20
    scalar_grid_points: int = 401
    solver_tolerance: float = 1e-6
    random_seed: int = 20260713

    def validate(self) -> None:
        super().validate()
        if not 0 < self.annual_downside_target < 1:
            raise ValueError("annual_downside_target must lie between zero and one")
        if self.recent_months < self.min_recent_observations:
            raise ValueError("recent_months must cover min_recent_observations")
        if self.imputation_sets != 5:
            raise ValueError("R2 requires exactly five residual-imputation sets")
        if self.volatility_horizon_days != 21:
            raise ValueError("R2 daily volatility horizon is fixed at 21 trading days")
        if self.scalar_grid_points < 51:
            raise ValueError("scalar_grid_points must be at least 51")


@dataclass(frozen=True)
class R2MomentSpec:
    """Auditable recent/expanding complete joint covariance estimate."""

    recent: GreekJointMomentSpec
    expanding: GreekJointMomentSpec
    blended: GreekJointMomentSpec
    blended_option_cov: pd.DataFrame
    recent_weight: float
    option_returns_imputed: pd.DataFrame
    imputation_scenarios: tuple[pd.DataFrame, ...]
    qlike_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    volatility_ledger: Mapping[str, Any] = field(default_factory=dict)

    @property
    def contract_names(self) -> list[str]:
        return list(self.blended_option_cov.index)

    def validate(self) -> None:
        labels = self.contract_names
        if list(self.blended_option_cov.columns) != labels:
            raise ValueError("blended covariance labels are not square-aligned")
        values = self.blended_option_cov.to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(values, values.T, atol=1e-10):
            raise ValueError("blended covariance must be finite and symmetric")
        if np.linalg.eigvalsh(values).min() < -1e-8:
            raise ValueError("blended covariance must be positive semidefinite")
        if not 0 <= self.recent_weight <= 1:
            raise ValueError("recent_weight must lie in [0, 1]")


@dataclass
class R2OptimizationResult:
    """Continuous R2 decision and its preserved audit diagnostics."""

    direction: pd.Series
    weights: pd.Series
    status: str
    solver: str
    objective_stats: dict[str, Any]
    constraint_diagnostics: dict[str, Any]


@dataclass
class R2IntegerResult:
    weights: pd.Series
    diagnostics: dict[str, Any]
    rejected_weights: pd.Series


def option_covariance(moment: GreekJointMomentSpec, loadings: pd.DataFrame) -> pd.DataFrame:
    """Transform a complete factor/residual covariance into option space."""

    factors = list(loadings.columns)
    contracts = list(loadings.index)
    moment.validate(factors, contracts)
    b = loadings.to_numpy(float)
    transform = np.column_stack([b, np.eye(len(contracts))])
    values = transform @ moment.joint_covariance().to_numpy(float) @ transform.T
    values = nearest_psd(0.5 * (values + values.T), floor=1e-10)
    return pd.DataFrame(values, index=contracts, columns=contracts)


def qlike_loss(covariance: np.ndarray, realized: np.ndarray) -> float:
    """Multivariate Gaussian covariance QLIKE (constant omitted)."""

    cov = nearest_psd(np.asarray(covariance, dtype=float), floor=1e-9)
    value = np.asarray(realized, dtype=float).ravel()
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("inf")
    return float(logdet + value @ np.linalg.solve(cov, value))


def select_recent_covariance_weight(
    recent_forecasts: Sequence[np.ndarray],
    expanding_forecasts: Sequence[np.ndarray],
    realized_returns: Sequence[np.ndarray],
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> tuple[float, pd.DataFrame]:
    """Chronological QLIKE selection with a lower-recent-weight one-SE rule."""

    n = min(len(recent_forecasts), len(expanding_forecasts), len(realized_returns))
    if n < config.min_inner_forecasts:
        return config.default_recent_weight, pd.DataFrame(
            [{"recent_weight": config.default_recent_weight, "n_forecasts": n, "selected": True, "reason": "default_before_12"}]
        )
    rows: list[dict[str, Any]] = []
    loss_by_weight: dict[float, np.ndarray] = {}
    for weight in config.recent_weights:
        losses = np.asarray(
            [
                qlike_loss(weight * recent_forecasts[i] + (1.0 - weight) * expanding_forecasts[i], realized_returns[i])
                for i in range(n)
            ],
            dtype=float,
        )
        loss_by_weight[weight] = losses
        rows.append(
            {
                "recent_weight": weight,
                "n_forecasts": n,
                "mean_qlike": float(losses.mean()),
                "se_qlike": float(losses.std(ddof=1) / np.sqrt(n)),
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["mean_qlike"].idxmin()]
    threshold = float(best["mean_qlike"] + best["se_qlike"])
    eligible = frame[frame["mean_qlike"] <= threshold + 1e-12]
    selected = float(eligible["recent_weight"].min())
    frame["selected"] = frame["recent_weight"].eq(selected)
    frame["one_se_threshold"] = threshold
    frame["reason"] = "one_se_lower_recent_weight"
    return selected, frame


def _deterministic_residual_imputations(
    returns: pd.DataFrame,
    systematic: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, ...]]:
    observed = returns.notna()
    residuals = returns - systematic
    base = returns.where(observed, systematic)
    outputs: list[pd.DataFrame] = []
    for scenario_id in range(count):
        rng = np.random.default_rng(seed + 7919 * scenario_id)
        imputed = base.copy()
        for column in returns.columns:
            missing = ~observed[column]
            pool = residuals.loc[observed[column], column].dropna().to_numpy(float)
            if missing.any() and len(pool):
                draws = pool[rng.integers(0, len(pool), size=int(missing.sum()))]
                imputed.loc[missing, column] = systematic.loc[missing, column].to_numpy(float) + draws
        outputs.append(imputed)
    return base, tuple(outputs)


def estimate_r2_moments(
    option_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    greek_loadings: pd.DataFrame,
    *,
    recent_weight: float = 0.50,
    train_end: pd.Timestamp | None = None,
    config: RobustSortinoConfig = RobustSortinoConfig(),
    qlike_ledger: pd.DataFrame | None = None,
) -> R2MomentSpec:
    """Estimate R2 moments without treating absent option returns as zero."""

    config.validate()
    cutoff = pd.Timestamp(train_end) if train_end is not None else None
    returns = option_returns.copy()
    factors = factor_returns.copy()
    if cutoff is not None:
        returns = returns.loc[pd.to_datetime(returns.index) <= cutoff]
        factors = factors.loc[pd.to_datetime(factors.index) <= cutoff]
    common = returns.index.intersection(factors.index).sort_values()
    returns = returns.reindex(index=common, columns=greek_loadings.index).replace([np.inf, -np.inf], np.nan)
    factors = factors.reindex(index=common, columns=greek_loadings.columns)
    if len(common) < config.recent_months:
        raise ValueError("R2 requires at least 36 observed monthly factor dates")
    recent_index = common[-config.recent_months :]
    counts = returns.loc[recent_index].notna().sum()
    deficient = counts[counts < config.min_recent_observations]
    if len(deficient):
        raise ValueError(f"contracts lack 24 recent observations: {list(deficient.index)}")
    systematic = pd.DataFrame(
        factors.to_numpy(float) @ greek_loadings.to_numpy(float).T,
        index=common,
        columns=greek_loadings.index,
    )
    base, scenarios = _deterministic_residual_imputations(
        returns, systematic, count=config.imputation_sets, seed=config.random_seed
    )
    recent = estimate_greek_joint_moments(
        base.loc[recent_index], factors.loc[recent_index], greek_loadings, regularize=True
    )
    expanding = estimate_greek_joint_moments(base, factors, greek_loadings, regularize=True)
    recent_joint = recent.joint_covariance().to_numpy(float)
    expanding_joint = expanding.joint_covariance().to_numpy(float)
    joint = recent_weight * recent_joint + (1.0 - recent_weight) * expanding_joint
    joint = nearest_psd(0.5 * (joint + joint.T), floor=1e-10)
    k = len(greek_loadings.columns)
    contracts = list(greek_loadings.index)
    factors_list = list(greek_loadings.columns)
    blended_moment = GreekJointMomentSpec(
        pd.DataFrame(joint[:k, :k], index=factors_list, columns=factors_list),
        pd.DataFrame(joint[:k, k:], index=factors_list, columns=contracts),
        pd.DataFrame(joint[k:, k:], index=contracts, columns=contracts),
        min(recent.n_obs, expanding.n_obs),
        estimator=f"r2_blend_recent_{recent_weight:.2f}",
    )
    blended_moment.validate(factors_list, contracts)
    blend_frame = option_covariance(blended_moment, greek_loadings)
    spec = R2MomentSpec(
        recent=recent,
        expanding=expanding,
        blended=blended_moment,
        blended_option_cov=blend_frame,
        recent_weight=float(recent_weight),
        option_returns_imputed=base,
        imputation_scenarios=scenarios,
        qlike_ledger=pd.DataFrame() if qlike_ledger is None else qlike_ledger.copy(),
    )
    spec.validate()
    return spec


def exponentially_weighted_mean(frame: pd.DataFrame, half_life: float = 36.0) -> pd.Series:
    """Expanding-history mean with a fixed observation half-life."""

    clean = frame.astype(float)
    age = np.arange(len(clean) - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * age / float(half_life))
    weighted = clean.mul(weights, axis=0)
    denominator = clean.notna().mul(weights, axis=0).sum(axis=0)
    return weighted.sum(axis=0).div(denominator.replace(0.0, np.nan))


def _ewma_variance(values: np.ndarray, decay: float = 0.94) -> float:
    variance = float(np.var(values[: min(20, len(values))], ddof=1)) if len(values) > 1 else 0.0
    for value in values:
        variance = decay * variance + (1.0 - decay) * float(value * value)
    return max(variance, 1e-12)


def _har_variance(values: np.ndarray, horizon: int = 1) -> float:
    """Recursive HAR-RV average daily variance over the requested horizon."""

    squared = np.asarray(values, dtype=float) ** 2
    rows, target = [], []
    for i in range(22, len(squared)):
        rows.append([1.0, squared[i - 1], squared[i - 5 : i].mean(), squared[i - 21 : i].mean()])
        target.append(squared[i])
    if len(rows) < 30:
        return _ewma_variance(values)
    x = np.asarray(rows, dtype=float)
    y = np.asarray(target, dtype=float)
    ridge = 1e-8 * np.eye(x.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    working = list(squared)
    forecasts = []
    for _ in range(horizon):
        array = np.asarray(working, dtype=float)
        forecast = np.asarray([1.0, array[-1], array[-5:].mean(), array[-21:].mean()]) @ beta
        forecast = max(float(forecast), 1e-12)
        forecasts.append(forecast)
        working.append(forecast)
    return float(np.mean(forecasts))


def select_daily_volatility_overlay(
    daily_returns: pd.Series,
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> dict[str, Any]:
    """Training-only HAR/EWMA blend selected by rolling one-step QLIKE."""

    values = pd.Series(daily_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().iloc[-config.daily_window :]
    if len(values) < config.min_daily_observations:
        return {
            "available": False,
            "fallback": True,
            "n_obs": int(len(values)),
            "horizon_days": config.volatility_horizon_days,
            "variance_ratio": 1.0,
        }
    array = values.to_numpy(float)
    horizon = config.volatility_horizon_days
    validation_start = max(252, len(array) - 126 - horizon + 1)
    har_losses: list[float] = []
    ewma_losses: list[float] = []
    har_forecasts: list[float] = []
    ewma_forecasts: list[float] = []
    realized: list[float] = []
    for i in range(validation_start, len(array) - horizon + 1):
        history = array[:i]
        har = _har_variance(history, horizon=horizon)
        ewma = _ewma_variance(history)
        actual = max(float(np.mean(array[i : i + horizon] ** 2)), 1e-12)
        har_forecasts.append(har)
        ewma_forecasts.append(ewma)
        realized.append(actual)
        har_losses.append(np.log(har) + actual / har)
        ewma_losses.append(np.log(ewma) + actual / ewma)
    rows = []
    for har_weight in config.volatility_blend_weights:
        forecast = har_weight * np.asarray(har_forecasts) + (1.0 - har_weight) * np.asarray(ewma_forecasts)
        losses = np.log(forecast) + np.asarray(realized) / forecast
        rows.append((har_weight, float(losses.mean()), float(losses.std(ddof=1) / np.sqrt(len(losses)))))
    ledger = pd.DataFrame(rows, columns=["har_weight", "mean_qlike", "se_qlike"])
    best = ledger.loc[ledger["mean_qlike"].idxmin()]
    eligible = ledger[ledger["mean_qlike"] <= float(best["mean_qlike"] + best["se_qlike"]) + 1e-12]
    selected = float(eligible["har_weight"].min())
    ledger["selected"] = ledger["har_weight"].eq(selected)
    forecast = selected * _har_variance(array, horizon=horizon) + (1.0 - selected) * _ewma_variance(array)
    baseline = max(float(np.mean(array[-horizon:] ** 2)), 1e-12)
    ratio = float(np.clip(forecast / baseline, config.variance_ratio_floor, config.variance_ratio_ceiling))
    return {
        "available": True,
        "fallback": False,
        "n_obs": int(len(values)),
        "horizon_days": horizon,
        "har_weight": selected,
        "forecast_variance": float(forecast),
        "baseline_variance": baseline,
        "variance_ratio": ratio,
        "qlike_ledger": ledger,
    }


def apply_joint_volatility_scaling(
    moment: GreekJointMomentSpec,
    loadings: pd.DataFrame,
    variance_ratios: Mapping[str, float],
) -> tuple[GreekJointMomentSpec, pd.DataFrame]:
    """Apply one joint diagonal transform, preserving PSD and cross blocks."""

    factors = list(loadings.columns)
    contracts = list(loadings.index)
    scales = []
    for name in factors:
        underlying = name.split("_", 1)[1]
        ratio = float(variance_ratios.get(underlying, 1.0))
        if name.startswith("r2_"):
            scales.append(ratio)
        elif name.startswith("r_"):
            scales.append(np.sqrt(ratio))
        else:
            scales.append(1.0)
    diagonal = np.asarray(scales + [1.0] * len(contracts), dtype=float)
    joint = moment.joint_covariance().to_numpy(float)
    transformed = diagonal[:, None] * joint * diagonal[None, :]
    k = len(factors)
    updated = GreekJointMomentSpec(
        pd.DataFrame(transformed[:k, :k], index=factors, columns=factors),
        pd.DataFrame(transformed[:k, k:], index=factors, columns=contracts),
        pd.DataFrame(transformed[k:, k:], index=contracts, columns=contracts),
        moment.n_obs,
        estimator=f"{moment.estimator}_r2_har_ewma_scaled",
    )
    updated.validate(factors, contracts)
    return updated, option_covariance(updated, loadings)


def circular_block_scenarios(
    returns: pd.DataFrame,
    *,
    paths: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministic expanding-history block resample used in optimization."""

    values = returns.to_numpy(float)
    if len(values) == 0:
        return returns.copy()
    rng = np.random.default_rng(seed)
    needed_blocks = int(np.ceil(paths / block_length))
    starts = rng.integers(0, len(values), size=needed_blocks)
    indices = np.concatenate([(start + np.arange(block_length)) % len(values) for start in starts])[:paths]
    return pd.DataFrame(values[indices], columns=returns.columns)


def _scenario_arrays(
    scenario_families: Mapping[str, pd.DataFrame], contracts: Sequence[str]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, frame in scenario_families.items():
        values = frame.reindex(columns=contracts).to_numpy(float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"scenario family {name!r} must contain finite aligned rows")
        arrays[name] = values
    if not arrays:
        raise ValueError("at least one scenario family is required")
    return arrays


def solve_robust_sortino_direction(
    model: OptionOnlyMarkowitzModel,
    scenario_families: Mapping[str, pd.DataFrame],
    costs: OptimizationCostSpec,
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> tuple[pd.Series, dict[str, Any]]:
    """Maximize net mean divided by the worst zero-target downside deviation."""

    config.validate()
    arrays = _scenario_arrays(scenario_families, model.contracts)
    n = len(model.contracts)
    long_cost, short_cost, _, short_allowed = costs.aligned(model.contracts)
    y = cp.Variable(n)
    long = cp.pos(y)
    short = cp.pos(-y)
    predictable_cost = long_cost @ long + short_cost @ short
    constraints: list[Any] = [cp.norm(y, 2) <= 1e3]
    for values in arrays.values():
        downside = cp.pos(-(values @ y - predictable_cost))
        constraints.append(cp.norm(downside, 2) <= np.sqrt(len(values)))
    if (~short_allowed).any():
        constraints.append(y[~short_allowed] >= 0.0)
    objective = cp.Maximize(model.expected_returns.to_numpy(float) @ y - predictable_cost)
    problem = cp.Problem(objective, constraints)
    used_solver = ""
    for solver in ("CLARABEL", "SCS"):
        try:
            kwargs = {"solver": solver, "verbose": False}
            if solver == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 50_000})
            problem.solve(**kwargs)
        except cp.error.SolverError:
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and y.value is not None:
            used_solver = solver
            break
    if not used_solver:
        return pd.Series(0.0, index=model.contracts), {"status": str(problem.status), "solver": "none"}
    raw = np.asarray(y.value, dtype=float).ravel()
    gross = float(np.abs(raw).sum())
    if gross <= 1e-10 or float(problem.value) <= 1e-10:
        return pd.Series(0.0, index=model.contracts), {
            "status": "cash_nonpositive_net_direction",
            "solver": used_solver,
            "robust_sortino_numerator": float(problem.value or 0.0),
        }
    direction = raw / gross
    family_downside = {}
    cost = float(long_cost @ np.maximum(direction, 0.0) + short_cost @ np.maximum(-direction, 0.0))
    for name, values in arrays.items():
        net = values @ direction - cost
        family_downside[name] = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))
    numerator = float(model.expected_returns.to_numpy(float) @ direction - cost)
    worst = max(family_downside.values())
    return pd.Series(direction, index=model.contracts, name="direction"), {
        "status": "optimal",
        "solver": used_solver,
        "robust_sortino_numerator": numerator,
        "worst_monthly_downside": worst,
        "robust_sortino": numerator / max(worst, 1e-12),
        **{f"downside_{name}": value for name, value in family_downside.items()},
    }


def _path_loss(net_returns: np.ndarray, horizon: int) -> float:
    if len(net_returns) < horizon:
        return 0.0
    losses = []
    for start in range(len(net_returns) - horizon + 1):
        compounded = float(np.prod(1.0 + net_returns[start : start + horizon]) - 1.0)
        losses.append(-compounded)
    return max(max(losses), 0.0)


def r2_constraint_diagnostics(
    model: OptionOnlyMarkowitzModel,
    weights: pd.Series,
    chronological_scenarios: pd.DataFrame,
    scenario_families: Mapping[str, pd.DataFrame],
    costs: OptimizationCostSpec,
    caps: pd.Series,
    config: RobustSortinoConfig,
) -> dict[str, Any]:
    """Evaluate all R2 constraints without replacing a rejected portfolio."""

    base = dict(r1_constraint_diagnostics(model, weights, chronological_scenarios, costs, caps, config))
    w = weights.reindex(model.contracts).fillna(0.0).to_numpy(float)
    long_cost, short_cost, _, _ = costs.aligned(model.contracts)
    predictable_cost = float(long_cost @ np.maximum(w, 0.0) + short_cost @ np.maximum(-w, 0.0))
    annual_vol = float(np.sqrt(max(w @ model.option_cov @ w, 0.0) * config.periods_per_year))
    family_downside: dict[str, float] = {}
    for name, values in _scenario_arrays(scenario_families, model.contracts).items():
        net = values @ w - predictable_cost
        family_downside[name] = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)) * np.sqrt(config.periods_per_year))
    worst_downside = max(family_downside.values())
    chronological = chronological_scenarios.reindex(columns=model.contracts).to_numpy(float) @ w - predictable_cost
    loss3 = _path_loss(chronological, 3)
    loss6 = _path_loss(chronological, 6)
    extra = {
        "volatility": max(annual_vol - config.annual_vol_target, 0.0),
        "downside": max(worst_downside - config.annual_downside_target, 0.0),
        "three_month_loss": max(loss3 - config.max_three_month_loss, 0.0),
        "six_month_loss": max(loss6 - config.max_six_month_loss, 0.0),
    }
    max_breach = max(float(base["max_breach"]), *extra.values())
    base.update(
        {
            "feasible": bool(max_breach <= config.solver_tolerance),
            "max_breach": float(max_breach),
            "predicted_annual_vol": annual_vol,
            "worst_annual_downside": worst_downside,
            "worst_three_month_loss": loss3,
            "worst_six_month_loss": loss6,
            "predictable_cost": predictable_cost,
            **{f"breach_{name}": float(value) for name, value in extra.items()},
            **{f"annual_downside_{name}": value for name, value in family_downside.items()},
        }
    )
    return base


def select_log_growth_scale(
    model: OptionOnlyMarkowitzModel,
    direction: pd.Series,
    chronological_scenarios: pd.DataFrame,
    scenario_families: Mapping[str, pd.DataFrame],
    costs: OptimizationCostSpec,
    caps: pd.Series,
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> tuple[pd.Series, dict[str, Any]]:
    """Choose the unique feasible scale that maximizes expected net log growth."""

    d = direction.reindex(model.contracts).fillna(0.0)
    if float(d.abs().sum()) <= 1e-12:
        cash = pd.Series(0.0, index=model.contracts, name="weight")
        return cash, {"selected_scale": 0.0, "expected_net_log_growth": 0.0, "status": "cash_no_direction"}
    d = d / float(d.abs().sum())
    long_cost, short_cost, _, _ = costs.aligned(model.contracts)
    vector = d.to_numpy(float)
    directional_cost = float(long_cost @ np.maximum(vector, 0.0) + short_cost @ np.maximum(-vector, 0.0))
    scenario = chronological_scenarios.reindex(columns=model.contracts).to_numpy(float) @ vector - directional_cost
    cap_values = caps.reindex(model.contracts).fillna(0.0).to_numpy(float)
    nonzero = np.abs(vector) > 1e-12
    cap_scale = float(np.min(cap_values[nonzero] / np.abs(vector[nonzero]))) if nonzero.any() else 0.0
    upper = max(min(float(model.constraints.gross_nav), cap_scale), 0.0)

    def evaluate(scale: float) -> tuple[float, dict[str, Any]]:
        weights = pd.Series(scale * vector, index=model.contracts)
        diagnostics = r2_constraint_diagnostics(
            model, weights, chronological_scenarios, scenario_families, costs, caps, config
        )
        scaled = scale * scenario
        if (scaled <= -1.0).any():
            return -np.inf, diagnostics
        growth = float(np.mean(np.log1p(scaled)))
        return growth, diagnostics

    grid = np.linspace(0.0, upper, config.scalar_grid_points)
    feasible: list[tuple[float, float, dict[str, Any]]] = []
    for scale in grid:
        growth, diagnostics = evaluate(float(scale))
        if diagnostics["feasible"]:
            feasible.append((float(scale), growth, diagnostics))
    if not feasible:
        cash = pd.Series(0.0, index=model.contracts, name="weight")
        return cash, {"selected_scale": 0.0, "expected_net_log_growth": 0.0, "status": "cash_no_feasible_scale"}
    best_scale, best_growth, best_diagnostics = max(feasible, key=lambda item: item[1])
    step = upper / max(config.scalar_grid_points - 1, 1)
    lo, hi = max(0.0, best_scale - step), min(upper, best_scale + step)
    if hi > lo and best_scale > 0:
        optimized = minimize_scalar(lambda value: -evaluate(float(value))[0], bounds=(lo, hi), method="bounded")
        candidate_growth, candidate_diagnostics = evaluate(float(optimized.x))
        if candidate_diagnostics["feasible"] and candidate_growth > best_growth:
            best_scale, best_growth, best_diagnostics = float(optimized.x), candidate_growth, candidate_diagnostics
    if not np.isfinite(best_growth) or best_growth <= 0.0:
        cash = pd.Series(0.0, index=model.contracts, name="weight")
        return cash, {"selected_scale": 0.0, "expected_net_log_growth": 0.0, "status": "cash_nonpositive_log_growth"}
    weights = pd.Series(best_scale * vector, index=model.contracts, name="weight")
    return weights, {
        **best_diagnostics,
        "selected_scale": best_scale,
        "expected_net_log_growth": best_growth,
        "status": "optimal",
    }


def solve_r2_robust_sortino(
    model: OptionOnlyMarkowitzModel,
    chronological_scenarios: pd.DataFrame,
    scenario_families: Mapping[str, pd.DataFrame],
    costs: OptimizationCostSpec,
    caps: pd.Series,
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> R2OptimizationResult:
    direction, direction_stats = solve_robust_sortino_direction(model, scenario_families, costs, config)
    weights, scale_stats = select_log_growth_scale(
        model, direction, chronological_scenarios, scenario_families, costs, caps, config
    )
    diagnostics = r2_constraint_diagnostics(
        model, weights, chronological_scenarios, scenario_families, costs, caps, config
    )
    return R2OptimizationResult(
        direction=direction,
        weights=weights,
        status=str(scale_stats.get("status", direction_stats.get("status", "unknown"))),
        solver=str(direction_stats.get("solver", "none")),
        objective_stats={**direction_stats, **scale_stats},
        constraint_diagnostics=diagnostics,
    )


def integerize_r2_direct_or_abstain(
    model: OptionOnlyMarkowitzModel,
    continuous: pd.Series,
    marks: pd.Series,
    nav: float,
    caps: pd.Series,
    chronological_scenarios: pd.DataFrame,
    scenario_families: Mapping[str, pd.DataFrame],
    costs: OptimizationCostSpec,
    config: RobustSortinoConfig = RobustSortinoConfig(),
) -> R2IntegerResult:
    """Truncate the exact target to contracts; select cash on any breach."""

    labels = model.contracts
    target = continuous.reindex(labels).fillna(0.0)
    mark_values = marks.reindex(labels).to_numpy(float)
    if not np.isfinite(mark_values).all() or (mark_values <= 0).any():
        raise ValueError("R2 integer execution requires finite positive marks")
    unit = 100.0 * mark_values / float(nav)
    counts = np.sign(target.to_numpy(float)) * np.floor(np.abs(target.to_numpy(float)) / unit + 1e-12)
    rejected = pd.Series(counts * unit, index=labels, name="rejected_weight")
    rejected_diagnostics = r2_constraint_diagnostics(
        model, rejected, chronological_scenarios, scenario_families, costs, caps, config
    )
    abstained = not bool(rejected_diagnostics["feasible"])
    selected = pd.Series(0.0, index=labels, name="weight") if abstained else rejected.rename("weight")
    selected_diagnostics = r2_constraint_diagnostics(
        model, selected, chronological_scenarios, scenario_families, costs, caps, config
    )
    selected_diagnostics.update(
        {
            "integer_execution_abstained": abstained,
            "integer_conversion_feasible": not abstained,
            "integer_contracts": int(np.abs(counts).sum()) if not abstained else 0,
            "rejected_integer_contracts": int(np.abs(counts).sum()),
            "rejected_feasible": bool(rejected_diagnostics["feasible"]),
            "rejected_max_breach": float(rejected_diagnostics["max_breach"]),
            "abstention_reason": "direct_integer_conversion_infeasible" if abstained else "",
        }
    )
    for key, value in rejected_diagnostics.items():
        selected_diagnostics[f"rejected_{key}"] = value
    return R2IntegerResult(selected, selected_diagnostics, rejected)
