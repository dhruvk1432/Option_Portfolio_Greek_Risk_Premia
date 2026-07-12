"""Option-only Markowitz portfolio model.

This module implements the optimizer used by
``research/papers/option_only_markowitz``.  It deliberately treats listed
options as the only risky investable instruments.  Cash is a numeraire for
NAV and collateral accounting, not an optimized asset.

The risk model is the option analogue of Markowitz: contract P&L is mapped
to delta, centered squared-return and implied-volatility shocks through
Greeks.  The repaired R1 covariance keeps the generally nonzero covariance
between those factors and the Greek residual,

    Sigma_O = B Omega B' + B Gamma + Gamma' B' + Sigma_epsilon.

Positions are dollar/NAV weights in option contracts.  A value of ``0.10`` in
one contract means the portfolio has a ten percent NAV exposure to that
option's mark.  The legacy maximum-Sharpe machinery remains available for
diagnostics.  R1 uses ``solve_net_utility``, which prices long/short costs
before allocation and permits cash instead of forcing full gross deployment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

try:  # keep the module usable without cvxpy
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVXPY = False


OPTION_REQUIRED_COLUMNS = [
    "underlying",
    "mark",
    "spot",
    "delta",
    "gamma",
    "vega",
    "theta",
]


@dataclass(frozen=True)
class OptionOnlySpec:
    """Option universe inputs.

    ``frame`` is indexed by contract identifier.  The required columns are:
    ``underlying``, ``mark``, ``spot``, ``delta``, ``gamma``, ``vega`` and
    ``theta``.  Greeks are per share or per contract in the same convention as
    ``mark``; the model rescales them into dollar-return exposures.  ``spot``
    is the underlying price used to convert per-share delta/gamma into
    NAV-return loadings (delta scales by ``spot`` and gamma by ``spot**2``),
    so it must be finite and strictly positive for every contract.

    ``multiplier`` is retained for API compatibility only.  Positions are
    premium (NAV) weights, so the contract multiplier cancels out of every
    return computation and the field is unused by this model.
    """

    frame: pd.DataFrame
    multiplier: float = 100.0

    def validate(self) -> None:
        missing = [c for c in OPTION_REQUIRED_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"OptionOnlySpec.frame missing columns: {missing}")
        if self.frame.index.has_duplicates:
            raise ValueError("OptionOnlySpec.frame index must be unique contract ids")
        marks = pd.to_numeric(self.frame["mark"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(marks) & (marks > 0)).all():
            raise ValueError("Option marks must be finite and strictly positive")
        spots = pd.to_numeric(self.frame["spot"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(spots) & (spots > 0)).all():
            raise ValueError("Option spots must be finite and strictly positive")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be strictly positive")
        for col in ["delta", "gamma", "vega", "theta"]:
            vals = pd.to_numeric(self.frame[col], errors="coerce")
            if not np.isfinite(vals).all():
                raise ValueError(f"Option column {col!r} must be finite")


@dataclass(frozen=True)
class FactorShockSpec:
    """Systematic shocks that drive option returns.

    ``underlying_cov`` is the covariance of underlying *simple returns* over
    the allocation horizon.  ``vol_cov`` is the covariance of implied-vol
    shocks, indexed by underlying.  If omitted, a diagonal volatility-shock
    covariance is inferred from ``underlying_cov``.

    ``spot_vol_cov`` is an optional ``k x k`` cross-covariance matrix
    ``Cov(R_u, Delta sigma_v)`` between underlying returns (rows) and
    implied-vol shocks (columns).  When provided it populates the
    spot-vol off-diagonal blocks of the factor covariance (followed by the
    usual nearest-PSD repair).  When omitted (the default) the factor
    covariance is block-diagonal, i.e. zero spot-vol correlation is assumed;
    this understates risk for books whose delta and vega exposures have
    aligned signs (and overstates it for opposed signs), since the usually
    negative spot-vol correlation is dropped.
    """

    underlying_cov: pd.DataFrame
    vol_cov: Optional[pd.DataFrame] = None
    gamma_var_cov: Optional[pd.DataFrame] = None
    horizon_years: float = 21.0 / 252.0
    spot_vol_cov: Optional[pd.DataFrame] = None

    def validate(self, underlyings: Sequence[str]) -> None:
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be strictly positive")
        _validate_square_psd(self.underlying_cov, "underlying_cov")
        missing = set(underlyings) - set(self.underlying_cov.index)
        if missing:
            raise ValueError(f"underlying_cov missing underlyings: {sorted(missing)}")
        if self.vol_cov is not None:
            _validate_square_psd(self.vol_cov, "vol_cov")
            missing = set(underlyings) - set(self.vol_cov.index)
            if missing:
                raise ValueError(f"vol_cov missing underlyings: {sorted(missing)}")
        if self.gamma_var_cov is not None:
            _validate_square_psd(self.gamma_var_cov, "gamma_var_cov")
            missing = set(underlyings) - set(self.gamma_var_cov.index)
            if missing:
                raise ValueError(f"gamma_var_cov missing underlyings: {sorted(missing)}")
        if self.spot_vol_cov is not None:
            missing = set(underlyings) - set(self.spot_vol_cov.index)
            if missing:
                raise ValueError(f"spot_vol_cov missing underlyings in index: {sorted(missing)}")
            missing = set(underlyings) - set(self.spot_vol_cov.columns)
            if missing:
                raise ValueError(f"spot_vol_cov missing underlyings in columns: {sorted(missing)}")
            vals = self.spot_vol_cov.to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError("spot_vol_cov must be finite")


@dataclass(frozen=True)
class GreekJointMomentSpec:
    """Joint covariance blocks of Greek factors and Greek residual returns.

    ``factor_residual_cov`` is ``Cov(f, epsilon)`` with factors on rows and
    option contracts on columns.  This block is generally nonzero when the
    exposure matrix is fixed from Greeks rather than estimated by OLS.
    """

    factor_cov: pd.DataFrame
    factor_residual_cov: pd.DataFrame
    residual_cov: pd.DataFrame
    n_obs: int
    estimator: str = "sample"

    @property
    def factor_names(self) -> list[str]:
        return list(self.factor_cov.index)

    @property
    def contract_names(self) -> list[str]:
        return list(self.residual_cov.index)

    def joint_covariance(self) -> pd.DataFrame:
        names = [f"factor::{name}" for name in self.factor_names] + [
            f"residual::{name}" for name in self.contract_names
        ]
        values = np.block(
            [
                [self.factor_cov.to_numpy(float), self.factor_residual_cov.to_numpy(float)],
                [self.factor_residual_cov.to_numpy(float).T, self.residual_cov.to_numpy(float)],
            ]
        )
        return pd.DataFrame(values, index=names, columns=names)

    def validate(self, factor_names: Sequence[str], contract_names: Sequence[str]) -> None:
        factors = list(factor_names)
        contracts = list(contract_names)
        if self.n_obs < 2:
            raise ValueError("GreekJointMomentSpec requires at least two observations")
        if list(self.factor_cov.index) != factors or list(self.factor_cov.columns) != factors:
            raise ValueError("factor_cov labels must match the model factor order")
        if list(self.factor_residual_cov.index) != factors or list(self.factor_residual_cov.columns) != contracts:
            raise ValueError("factor_residual_cov labels must be factor x contract")
        if list(self.residual_cov.index) != contracts or list(self.residual_cov.columns) != contracts:
            raise ValueError("residual_cov labels must match the model contract order")
        joint = self.joint_covariance().to_numpy(float)
        if not np.isfinite(joint).all():
            raise ValueError("joint Greek/residual covariance must be finite")
        if not np.allclose(joint, joint.T, atol=1e-10):
            raise ValueError("joint Greek/residual covariance must be symmetric")
        if np.linalg.eigvalsh(joint).min() < -1e-8:
            raise ValueError("joint Greek/residual covariance must be positive semidefinite")


@dataclass(frozen=True)
class NetUtilityConfig:
    """Fixed R1 economic-risk policy, expressed per monthly holding period."""

    annual_vol_target: float = 0.15
    periods_per_year: float = 12.0
    cvar_alpha: float = 0.95
    cvar_loss_nav: float = 0.10
    stress_loss_nav: float = 0.20
    short_margin_nav: float = 0.75
    collateral_nav: float = 1.00
    lambda_floor: float = 1e-6
    lambda_ceiling: float = 1e6
    bisection_steps: int = 18

    def validate(self) -> None:
        if self.annual_vol_target <= 0 or self.periods_per_year <= 0:
            raise ValueError("volatility target and periods_per_year must be positive")
        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must lie strictly between zero and one")
        for name in ["cvar_loss_nav", "stress_loss_nav", "short_margin_nav", "collateral_nav"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 < self.lambda_floor <= self.lambda_ceiling:
            raise ValueError("lambda bounds must be positive and ordered")
        if self.bisection_steps < 1:
            raise ValueError("bisection_steps must be positive")


@dataclass(frozen=True)
class OptimizationCostSpec:
    """Decision-time R1 costs and operational coefficients by contract."""

    long_cost: pd.Series
    short_cost: pd.Series
    short_margin: pd.Series
    assignment_short_allowed: pd.Series

    def aligned(self, contracts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        labels = list(contracts)
        long_cost = self.long_cost.reindex(labels).astype(float)
        short_cost = self.short_cost.reindex(labels).astype(float)
        short_margin = self.short_margin.reindex(labels).astype(float)
        allowed = self.assignment_short_allowed.reindex(labels)
        if long_cost.isna().any() or short_cost.isna().any() or short_margin.isna().any() or allowed.isna().any():
            raise ValueError("OptimizationCostSpec must cover every model contract")
        if (long_cost < 0).any() or (short_cost < 0).any() or (short_margin < 0).any():
            raise ValueError("cost and margin coefficients must be nonnegative")
        return (
            long_cost.to_numpy(float),
            short_cost.to_numpy(float),
            short_margin.to_numpy(float),
            allowed.astype(bool).to_numpy(),
        )


@dataclass(frozen=True)
class OptionMarkowitzConstraints:
    """Convex constraints for the option-only portfolio."""

    gross_nav: float = 1.0
    net_nav_abs: Optional[float] = None
    short_nav_abs: Optional[float] = None
    per_contract_abs: Optional[float] = None
    underlying_gross: Dict[str, float] = field(default_factory=dict)
    delta_abs: Optional[float] = None
    gamma_abs: Optional[float] = None
    vega_abs: Optional[float] = None
    vix_vega_abs: Optional[float] = None
    beta_spy_abs: Optional[float] = None
    stress_loss_abs: Optional[float] = None
    factor_exposure_abs: Dict[str, float] = field(default_factory=dict)
    long_only: bool = False

    def validate(self) -> None:
        if self.gross_nav <= 0:
            raise ValueError("gross_nav must be positive")
        for name, value in [
            ("net_nav_abs", self.net_nav_abs),
            ("short_nav_abs", self.short_nav_abs),
            ("per_contract_abs", self.per_contract_abs),
            ("delta_abs", self.delta_abs),
            ("gamma_abs", self.gamma_abs),
            ("vega_abs", self.vega_abs),
            ("vix_vega_abs", self.vix_vega_abs),
            ("beta_spy_abs", self.beta_spy_abs),
            ("stress_loss_abs", self.stress_loss_abs),
        ]:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for under, value in self.underlying_gross.items():
            if value < 0:
                raise ValueError(f"underlying_gross[{under!r}] must be nonnegative")
        for exposure, value in self.factor_exposure_abs.items():
            if value < 0:
                raise ValueError(f"factor_exposure_abs[{exposure!r}] must be nonnegative")


@dataclass(frozen=True)
class OptionMarkowitzResult:
    """Solver output.

    ``status`` is one of ``optimal`` (solver converged, feasible within
    ``1e-5``), ``feasible_suboptimal`` (feasible within ``1e-5`` but the
    solver did not report convergence, so the point may be suboptimal) or
    ``infeasible`` (maximum constraint violation exceeds ``1e-5``; the
    returned weights do NOT satisfy the requested constraints).
    ``max_violation`` reports the maximum constraint violation of the
    returned weights.

    ``sharpe`` is the objective ratio at the optimum: ``mu'q / vol`` for the
    max-Sharpe solvers and the net Sortino ratio for ``solve_max_sortino``.
    ``objective_stats`` is populated only by ``solve_max_sortino`` and holds
    per-call diagnostics (net mean, entry cost, downside deviation, net
    Sortino, target and the degenerate downside-free flag); it defaults to
    ``None`` so the dataclass remains positionally compatible with older
    callers.
    """

    status: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    gross_nav: float
    net_nav: float
    delta: float
    gamma: float
    vega: float
    solver: str
    max_violation: float = float("nan")
    objective_stats: Optional[Dict[str, object]] = None


def _validate_square_psd(frame: pd.DataFrame, name: str) -> None:
    if list(frame.index) != list(frame.columns):
        raise ValueError(f"{name} must have identical index and columns")
    values = frame.to_numpy(dtype=float)
    # Relative symmetry tolerance scaled by matrix magnitude, with the old
    # absolute floor retained so previously accepted inputs still pass.
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if not np.allclose(values, values.T, atol=max(1e-10, 1e-8 * scale)):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(values).min() < -1e-8:
        raise ValueError(f"{name} must be positive semidefinite")


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Return a symmetric PSD matrix by clipping eigenvalues."""

    values = np.asarray(matrix, dtype=float)
    sym = 0.5 * (values + values.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, floor)
    repaired = (eigvecs * clipped) @ eigvecs.T
    return 0.5 * (repaired + repaired.T)


def shrink_covariance(sample_cov: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    """Shrink a covariance matrix toward its diagonal and repair PSD drift."""

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    cov = np.asarray(sample_cov, dtype=float)
    target = np.diag(np.diag(cov))
    return nearest_psd((1.0 - shrinkage) * cov + shrinkage * target)


def greek_factor_names(underlyings: Sequence[str]) -> list[str]:
    """Canonical R1 factor order: spot, centered squared spot, then IV."""

    names = list(underlyings)
    return [f"r_{u}" for u in names] + [f"r2_{u}" for u in names] + [f"dv_{u}" for u in names]


def greek_exposure_frame(options: OptionOnlySpec) -> pd.DataFrame:
    """Return the premium-normalized delta/gamma/vega loading matrix."""

    options.validate()
    frame = options.frame
    contracts = list(frame.index)
    underlyings = sorted(frame["underlying"].astype(str).unique())
    columns = greek_factor_names(underlyings)
    values = np.zeros((len(contracts), len(columns)), dtype=float)
    k = len(underlyings)
    under_to_col = {underlying: j for j, underlying in enumerate(underlyings)}
    for row, (_, rec) in enumerate(frame.iterrows()):
        j = under_to_col[str(rec["underlying"])]
        mark = float(rec["mark"])
        spot = float(rec["spot"])
        values[row, j] = float(rec["delta"]) * spot / mark
        values[row, k + j] = 0.5 * float(rec["gamma"]) * spot * spot / mark
        values[row, 2 * k + j] = float(rec["vega"]) / mark
    return pd.DataFrame(values, index=contracts, columns=columns)


def estimate_greek_joint_moments(
    option_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    greek_loadings: pd.DataFrame,
    *,
    regularize: bool = True,
    train_end: Optional[pd.Timestamp] = None,
) -> GreekJointMomentSpec:
    """Estimate the complete joint covariance of factors and Greek residuals.

    Regularization is Ledoit-Wolf shrinkage on standardized joint observations,
    followed by rescaling.  With ``regularize=False`` the returned blocks are
    ordinary sample covariances and reconstruct the aligned sample option-return
    covariance exactly, including nonzero factor/residual cross terms.
    """

    contracts = list(greek_loadings.index)
    factors = list(greek_loadings.columns)
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        option_returns = option_returns.loc[pd.to_datetime(option_returns.index) <= cutoff]
        factor_returns = factor_returns.loc[pd.to_datetime(factor_returns.index) <= cutoff]
    common = option_returns.index.intersection(factor_returns.index)
    aligned = pd.concat(
        [
            factor_returns.reindex(index=common, columns=factors).add_prefix("factor::"),
            option_returns.reindex(index=common, columns=contracts).add_prefix("option::"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 2:
        raise ValueError("at least two complete aligned observations are required")
    F = aligned[[f"factor::{name}" for name in factors]].to_numpy(float)
    R = aligned[[f"option::{name}" for name in contracts]].to_numpy(float)
    F = F - F.mean(axis=0, keepdims=True)
    R = R - R.mean(axis=0, keepdims=True)
    B = greek_loadings.reindex(index=contracts, columns=factors).to_numpy(float)
    E = R - F @ B.T
    Z = np.column_stack([F, E])
    if regularize:
        from sklearn.covariance import ledoit_wolf

        scales = Z.std(axis=0, ddof=1)
        safe = np.where(scales > 1e-14, scales, 1.0)
        standardized = Z / safe
        corr_cov, _ = ledoit_wolf(standardized, assume_centered=True)
        joint = corr_cov * scales[:, None] * scales[None, :]
        estimator = "ledoit_wolf_correlation"
    else:
        joint = Z.T @ Z / float(len(Z) - 1)
        estimator = "sample"
    joint = 0.5 * (joint + joint.T)
    k = len(factors)
    factor_cov = pd.DataFrame(joint[:k, :k], index=factors, columns=factors)
    cross_cov = pd.DataFrame(joint[:k, k:], index=factors, columns=contracts)
    residual_cov = pd.DataFrame(joint[k:, k:], index=contracts, columns=contracts)
    spec = GreekJointMomentSpec(
        factor_cov=factor_cov,
        factor_residual_cov=cross_cov,
        residual_cov=residual_cov,
        n_obs=len(Z),
        estimator=estimator,
    )
    spec.validate(factors, contracts)
    return spec


def taylor_option_pnl(
    delta: np.ndarray,
    gamma: np.ndarray,
    vega: np.ndarray,
    theta: np.ndarray,
    dS: np.ndarray,
    dvol: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Second-order option P&L approximation used in the paper."""

    return delta * dS + 0.5 * gamma * dS * dS + vega * dvol + theta * dt


def performance_stats(
    returns: pd.Series,
    periods_per_year: float = 12.0,
    target_return: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict[str, float]:
    """Annualized return, volatility and standard performance ratios.

    ``target_return`` is expressed per observation.  The paper uses a zero
    monthly target so downside deviation is the annualized root mean squared
    shortfall below zero.  ``benchmark_returns`` is used only for the
    information ratio; when omitted the information ratio is not reported.
    """

    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "downside_ann_dev": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "omega": np.nan,
            "information_ratio": np.nan,
        }
    excess = r - target_return
    mu = float(excess.mean() * periods_per_year)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan
    sr = mu / vol if vol and np.isfinite(vol) else np.nan
    downside = np.minimum(excess.to_numpy(float), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside * downside)) * np.sqrt(periods_per_year))
    sortino = mu / downside_dev if downside_dev > 0 and np.isfinite(downside_dev) else np.nan
    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    calmar = mu / abs(max_drawdown) if max_drawdown < 0 and np.isfinite(max_drawdown) else np.nan
    gains = np.maximum(excess.to_numpy(float), 0.0).sum()
    losses = -np.minimum(excess.to_numpy(float), 0.0).sum()
    omega = float(gains / losses) if losses > 0 else np.nan
    information_ratio = np.nan
    if benchmark_returns is not None:
        aligned = pd.concat(
            [r.rename("portfolio"), pd.Series(benchmark_returns).rename("benchmark")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) > 1:
            active = aligned["portfolio"] - aligned["benchmark"]
            active_vol = float(active.std(ddof=1) * np.sqrt(periods_per_year))
            active_mean = float(active.mean() * periods_per_year)
            information_ratio = active_mean / active_vol if active_vol > 0 else np.nan
    return {
        "ann_return": mu,
        "ann_vol": vol,
        "sharpe": sr,
        "downside_ann_dev": downside_dev,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "omega": omega,
        "information_ratio": information_ratio,
    }


def bootstrap_sharpe_ci(
    returns: Sequence[float],
    periods_per_year: float = 12.0,
    n_boot: int = 500,
    seed: int = 7,
    alpha: float = 0.10,
) -> tuple[float, float]:
    """Simple iid bootstrap confidence interval for the annualized Sharpe."""

    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(r, size=len(r), replace=True)
        vol = sample.std(ddof=1) * np.sqrt(periods_per_year)
        vals.append((sample.mean() * periods_per_year) / vol if vol > 0 else np.nan)
    lo, hi = np.nanquantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


class OptionOnlyMarkowitzModel:
    """Greek-induced Markowitz optimizer over listed options only."""

    def __init__(
        self,
        options: OptionOnlySpec,
        shocks: FactorShockSpec,
        expected_returns: pd.Series,
        residual_cov: Optional[pd.DataFrame] = None,
        constraints: Optional[OptionMarkowitzConstraints] = None,
        covariance_shrinkage: float = 0.10,
        joint_moments: Optional[GreekJointMomentSpec] = None,
    ) -> None:
        """Assemble the Greek-induced option covariance and store inputs.

        ``residual_cov`` hazard: when ``residual_cov`` is ``None`` the
        idiosyncratic variance defaults to 5% of each contract's systematic
        (factor-induced) variance.  A contract whose Greeks are all (near)
        zero then receives (near) zero residual variance and can look
        artificially risk-free to the optimizer; supply an explicit
        ``residual_cov`` for such contracts.  When ``residual_cov`` is
        provided but misses some contracts, the missing entries are filled
        with zeros and a ``UserWarning`` is emitted.
        """

        options.validate()
        self.options = options
        self.frame = options.frame.copy()
        self.contracts = list(self.frame.index)
        self.underlyings = sorted(self.frame["underlying"].astype(str).unique())
        shocks.validate(self.underlyings)
        self.shocks = shocks
        self.constraints = constraints or OptionMarkowitzConstraints()
        self.constraints.validate()
        self.expected_returns = expected_returns.reindex(self.contracts).astype(float).fillna(0.0)
        self.covariance_shrinkage = covariance_shrinkage
        self.B = self._build_exposure_matrix()
        self.factor_names = greek_factor_names(self.underlyings)
        self.joint_moments = joint_moments
        if joint_moments is not None:
            if residual_cov is not None:
                raise ValueError("provide joint_moments or residual_cov, not both")
            joint_moments.validate(self.factor_names, self.contracts)
            self.factor_cov = joint_moments.factor_cov.to_numpy(float)
            self.factor_residual_cov = joint_moments.factor_residual_cov.to_numpy(float)
            self.residual_cov = joint_moments.residual_cov.to_numpy(float)
            option_cov = (
                self.B @ self.factor_cov @ self.B.T
                + self.B @ self.factor_residual_cov
                + self.factor_residual_cov.T @ self.B.T
                + self.residual_cov
            )
            # The joint estimator is already regularized.  Only a final
            # numerical eigenvalue floor is permitted on the transformed
            # option covariance.
            self.option_cov = nearest_psd(option_cov)
        else:
            self.factor_cov = self._build_factor_covariance()
            self.factor_residual_cov = np.zeros((len(self.factor_names), len(self.contracts)))
            systematic_cov = self.B @ self.factor_cov @ self.B.T
            if residual_cov is None:
                resid = np.diag(np.maximum(np.diag(systematic_cov), 1e-8)) * 0.05
            else:
                missing_contracts = [
                    c
                    for c in self.contracts
                    if c not in residual_cov.index or c not in residual_cov.columns
                ]
                if missing_contracts:
                    warnings.warn(
                        "residual_cov is missing contracts; their residual variance is "
                        f"filled with zeros: {missing_contracts}",
                        UserWarning,
                        stacklevel=2,
                    )
                resid_frame = residual_cov.reindex(index=self.contracts, columns=self.contracts).fillna(0.0)
                resid = resid_frame.to_numpy(dtype=float)
            self.residual_cov = resid
            self.option_cov = shrink_covariance(systematic_cov + nearest_psd(resid), covariance_shrinkage)
        self.greeks = self._portfolio_greek_loadings()

    def _build_exposure_matrix(self) -> np.ndarray:
        return greek_exposure_frame(self.options).to_numpy(float)

    def _build_factor_covariance(self) -> np.ndarray:
        k = len(self.underlyings)
        ucov = self.shocks.underlying_cov.reindex(index=self.underlyings, columns=self.underlyings).to_numpy(float)
        if self.shocks.gamma_var_cov is None:
            # Approximate covariance of centered squared-return shocks under
            # joint normality: Cov(r_i^2, r_j^2) = 2 Cov(r_i, r_j)^2.
            gcov = 2.0 * ucov * ucov
        else:
            gcov = self.shocks.gamma_var_cov.reindex(index=self.underlyings, columns=self.underlyings).to_numpy(float)
        if self.shocks.vol_cov is None:
            diag = np.maximum(np.diag(ucov), 1e-8)
            vcov = np.diag(0.25 * diag)
        else:
            vcov = self.shocks.vol_cov.reindex(index=self.underlyings, columns=self.underlyings).to_numpy(float)
        out = np.zeros((3 * k, 3 * k), dtype=float)
        out[:k, :k] = ucov
        out[k : 2 * k, k : 2 * k] = gcov
        out[2 * k :, 2 * k :] = vcov
        if self.shocks.spot_vol_cov is not None:
            scov = self.shocks.spot_vol_cov.reindex(
                index=self.underlyings, columns=self.underlyings
            ).to_numpy(dtype=float)
            out[:k, 2 * k :] = scov
            out[2 * k :, :k] = scov.T
        return nearest_psd(out)

    def _portfolio_greek_loadings(self) -> pd.DataFrame:
        mark = self.frame["mark"].to_numpy(dtype=float)
        spot = self.frame["spot"].to_numpy(dtype=float)
        under = self.frame["underlying"].astype(str).str.upper()
        asset_class = self.frame.get("asset_class", pd.Series("equity_option", index=self.frame.index)).astype(str).str.lower()
        is_vix = under.isin({"VX_FRONT", "VIX", "VIX_OPTION"}) | asset_class.eq("vix_option")
        out = pd.DataFrame(
            {
                "delta_nav": self.frame["delta"].to_numpy(float) * spot / mark,
                "gamma_nav": self.frame["gamma"].to_numpy(float) * spot * spot / mark,
                "vega_nav": self.frame["vega"].to_numpy(float) / mark,
            },
            index=self.contracts,
        )
        out["vix_vega_nav"] = out["vega_nav"].where(is_vix.to_numpy(), 0.0)
        out["equity_vega_nav"] = out["vega_nav"].where(~is_vix.to_numpy(), 0.0)
        for col in self.frame.columns:
            if col.startswith("exposure_") or col.startswith("beta_") or col.startswith("stress_"):
                out[col] = pd.to_numeric(self.frame[col], errors="coerce").fillna(0.0).to_numpy(float)
        if "beta_spy_nav" not in out:
            out["beta_spy_nav"] = out["delta_nav"] * pd.to_numeric(
                self.frame.get("underlying_beta_spy", pd.Series(0.0, index=self.frame.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(float)
        return out

    def _stress_matrix(self) -> Optional[np.ndarray]:
        cols = [c for c in self.greeks.columns if c.startswith("stress_scenario_")]
        if not cols:
            return None
        return self.greeks[cols].to_numpy(dtype=float).T

    def _named_exposure_vector(self, name: str) -> Optional[np.ndarray]:
        candidates = [name, f"{name}_nav", f"exposure_{name}", f"exposure_{name}_nav"]
        for col in candidates:
            if col in self.greeks.columns:
                return self.greeks[col].to_numpy(dtype=float)
        return None

    def covariance_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.option_cov, index=self.contracts, columns=self.contracts)

    def tangency_weights(self, gross_nav: Optional[float] = None) -> pd.Series:
        """Closed-form maximum-Sharpe weights scaled to gross NAV."""

        cov = nearest_psd(self.option_cov)
        mu = self.expected_returns.to_numpy(dtype=float)
        raw = np.linalg.pinv(cov) @ mu
        gross = np.abs(raw).sum()
        if gross <= 1e-14:
            weights = np.zeros_like(raw)
        else:
            target = self.constraints.gross_nav if gross_nav is None else gross_nav
            weights = raw * (target / gross)
        return pd.Series(weights, index=self.contracts, name="weight")

    def solve_max_sharpe(
        self,
        method: str = "slsqp",
        raise_on_infeasible: bool = False,
    ) -> OptionMarkowitzResult:
        """Maximize the Sharpe ratio under NAV-normalized constraints.

        The unconstrained tangency portfolio has a closed form, but gross-NAV,
        sign and exposure budgets make the investable problem a fractional
        constrained program.  We therefore solve the ratio directly and enforce
        ``sum(abs(q)) == gross_nav`` so reported positions are comparable across
        candidate portfolios.

        ``method='slsqp'`` (default) uses the SLSQP split-variable solver and
        reproduces the historical numerical behavior exactly.
        ``method='cvxpy'`` uses a homogenized (Charnes-Cooper) max-Sharpe SOCP
        with a global optimality guarantee; it requires cvxpy.

        The result ``status`` is one of ``optimal``, ``feasible_suboptimal``
        or ``infeasible`` (max constraint violation above ``1e-5``).  By
        default an infeasible result is still returned so callers can inspect
        it; pass ``raise_on_infeasible=True`` to raise instead.
        """

        if method == "slsqp":
            result = self._solve_scipy()
        elif method == "cvxpy":
            result = self._solve_cvxpy()
        else:
            raise ValueError(f"Unknown method {method!r}; expected 'slsqp' or 'cvxpy'")
        if raise_on_infeasible and result.status == "infeasible":
            raise ValueError(
                "solve_max_sharpe: constraints are infeasible "
                f"(max violation {result.max_violation:.6g} > 1e-5)"
            )
        return result

    def solve_max_sharpe_socp(self, raise_on_infeasible: bool = False) -> OptionMarkowitzResult:
        """Convenience alias for ``solve_max_sharpe(method='cvxpy')``."""

        return self.solve_max_sharpe(method="cvxpy", raise_on_infeasible=raise_on_infeasible)

    def _solve_cvxpy(self) -> OptionMarkowitzResult:
        """Homogenized (Charnes-Cooper) max-Sharpe SOCP.

        Variables are the split legs ``y+ >= 0``, ``y- >= 0`` (``y = y+ - y-``)
        and the scale ``t >= 0``.  We maximize ``mu' y`` subject to
        ``y' Sigma y <= 1``, the gross normalization ``sum(y+ + y-) ==
        gross_nav * t`` and every budget homogenized by ``t`` (per-contract
        caps, net/short NAV, underlying gross, linear Greek/beta budgets and
        stress rows).  The portfolio is recovered as ``q = y / t``; ``t ~ 0``
        means no nonzero portfolio satisfies the budgets at the requested
        gross (or no feasible direction has positive expected return) and is
        reported as ``infeasible``.
        """

        if not _HAS_CVXPY:
            raise ImportError(
                "cvxpy is required for solve_max_sharpe(method='cvxpy'); "
                "install cvxpy or use the default method='slsqp'"
            )
        n = len(self.contracts)
        mu = self.expected_returns.to_numpy(dtype=float)
        Sigma = nearest_psd(self.option_cov)
        y_pos = cp.Variable(n, nonneg=True)
        y_neg = cp.Variable(n, nonneg=True)
        t = cp.Variable(nonneg=True)
        y = y_pos - y_neg
        cons = [cp.quad_form(y, cp.psd_wrap(Sigma)) <= 1.0]
        cons.extend(self._cvxpy_homogenized_budget_constraints(y_pos, y_neg, t))
        problem = cp.Problem(cp.Maximize(mu @ y), cons)
        used_solver, _ = self._cvxpy_run_solvers(problem, y_pos)
        if used_solver is None:
            raise RuntimeError(f"cvxpy SOCP solve failed (last status: {problem.status!r})")
        solver_tag = f"cvxpy_socp_{used_solver.lower()}"
        t_val = float(t.value) if t.value is not None else 0.0
        if not np.isfinite(t_val) or t_val <= 1e-9:
            return self._make_result(np.zeros(n), "infeasible", solver_tag)
        q = (np.asarray(y_pos.value, dtype=float) - np.asarray(y_neg.value, dtype=float)).ravel() / t_val
        violation = self._max_constraint_violation(q)
        status = "optimal" if violation <= 1e-5 else "infeasible"
        return self._make_result(q, status, solver_tag)

    def _cvxpy_homogenized_budget_constraints(self, y_pos, y_neg, t) -> list:
        """Charnes-Cooper homogenized budget constraints in ``(y+, y-, t)``.

        Shared by the max-Sharpe and max-Sortino SOCPs.  Every constraint of
        the original set ``K`` (gross-NAV normalization, long-only flag,
        per-contract caps, net/short NAV, per-underlying gross, linear
        Greek/beta/factor budgets and stress floors) is scaled by ``t`` so
        that ``q = y / t`` satisfies ``K`` whenever ``t > 0``.
        """

        c = self.constraints
        y = y_pos - y_neg
        cons = [cp.sum(y_pos + y_neg) == c.gross_nav * t]
        if c.long_only:
            cons.append(y_neg == 0)
        cap = c.gross_nav if c.per_contract_abs is None else c.per_contract_abs
        cons.append(y_pos + y_neg <= cap * t)
        if c.net_nav_abs is not None:
            cons.append(cp.abs(cp.sum(y)) <= c.net_nav_abs * t)
        if c.short_nav_abs is not None:
            cons.append(cp.sum(y_neg) <= c.short_nav_abs * t)
        for under, limit in c.underlying_gross.items():
            idx = np.flatnonzero(self.frame["underlying"].astype(str).to_numpy() == under)
            if len(idx):
                cons.append(cp.sum(y_pos[idx] + y_neg[idx]) <= limit * t)
        for name, limit in [
            ("delta_nav", c.delta_abs),
            ("gamma_nav", c.gamma_abs),
            ("vega_nav", c.vega_abs),
            ("vix_vega_nav", c.vix_vega_abs),
            ("beta_spy_nav", c.beta_spy_abs),
        ]:
            if limit is not None:
                vec = self.greeks[name].to_numpy(dtype=float)
                cons.append(cp.abs(vec @ y) <= limit * t)
        for exposure, limit in c.factor_exposure_abs.items():
            vec = self._named_exposure_vector(exposure)
            if vec is not None:
                cons.append(cp.abs(vec @ y) <= limit * t)
        R = self._stress_matrix()
        if c.stress_loss_abs is not None and R is not None:
            cons.append(R @ y >= -c.stress_loss_abs * t)
        return cons

    # ------------------------------------------------------------------
    # R1 cost-aware mean-variance utility solver
    # ------------------------------------------------------------------

    def solve_net_utility(
        self,
        scenario_returns: pd.DataFrame,
        costs: OptimizationCostSpec,
        config: NetUtilityConfig = NetUtilityConfig(),
        *,
        per_contract_caps: Optional[pd.Series] = None,
        risk_aversion: Optional[float] = None,
    ) -> OptionMarkowitzResult:
        """Solve the R1 net-utility program with cash and hard tail limits.

        Gross exposure is an inequality.  The optimizer therefore chooses
        scale rather than manufacturing split-leg gross, and zero risky
        exposure is always feasible.  When ``risk_aversion`` is omitted a
        deterministic log-bisection uses training scenarios only to find the
        smallest positive value whose predicted annualized volatility does
        not exceed ``config.annual_vol_target``.
        """

        if not _HAS_CVXPY:
            raise ImportError("cvxpy is required for solve_net_utility")
        config.validate()
        long_cost, short_cost, short_margin, short_allowed = costs.aligned(self.contracts)
        scenarios = scenario_returns.reindex(columns=self.contracts)
        entirely_missing = [name for name in self.contracts if scenarios[name].isna().all()]
        if entirely_missing:
            raise ValueError(f"scenario_returns has no observations for contracts: {entirely_missing}")
        scenarios = scenarios.dropna(how="all").fillna(0.0)
        if len(scenarios) < 2:
            raise ValueError("solve_net_utility requires at least two training scenarios")
        R_scenario = scenarios.to_numpy(float)
        if not np.isfinite(R_scenario).all():
            raise ValueError("scenario_returns must be finite after alignment")

        n = len(self.contracts)
        w = cp.Variable(n)
        eta = cp.Variable()
        lam = cp.Parameter(nonneg=True)
        mu = self.expected_returns.to_numpy(float)
        Sigma = nearest_psd(self.option_cov)
        long = cp.pos(w)
        short = cp.pos(-w)
        predictable_cost = long_cost @ long + short_cost @ short
        objective = cp.Maximize(
            mu @ w - predictable_cost - 0.5 * lam * cp.quad_form(w, cp.psd_wrap(Sigma))
        )
        cons = self._cvxpy_net_utility_constraints(
            w,
            long,
            short,
            R_scenario,
            predictable_cost,
            eta,
            config,
            short_margin,
            short_allowed,
            per_contract_caps,
        )
        problem = cp.Problem(objective, cons)

        def solve_at(value: float) -> tuple[np.ndarray, str]:
            lam.value = float(value)
            used_solver, _ = self._cvxpy_run_solvers(problem, w)
            if used_solver is None or w.value is None:
                raise RuntimeError(f"R1 net-utility solve failed (status={problem.status!r})")
            weights = np.asarray(w.value, dtype=float).ravel()
            weights[np.abs(weights) < 1e-9] = 0.0
            return weights, used_solver

        if risk_aversion is not None:
            if not np.isfinite(risk_aversion) or risk_aversion <= 0:
                raise ValueError("risk_aversion must be finite and positive")
            weights, used_solver = solve_at(float(risk_aversion))
            selected_lambda = float(risk_aversion)
        else:
            target_monthly = config.annual_vol_target / np.sqrt(config.periods_per_year)
            low = float(config.lambda_floor)
            high = float(config.lambda_ceiling)
            low_weights, low_solver = solve_at(low)
            low_vol = float(np.sqrt(max(low_weights @ Sigma @ low_weights, 0.0)))
            if low_vol <= target_monthly + 1e-8:
                weights, used_solver, selected_lambda = low_weights, low_solver, low
            else:
                high_weights, high_solver = solve_at(high)
                high_vol = float(np.sqrt(max(high_weights @ Sigma @ high_weights, 0.0)))
                if high_vol > target_monthly + 1e-7:
                    raise RuntimeError("risk-aversion ceiling does not achieve the R1 volatility target")
                weights, used_solver = high_weights, high_solver
                for _ in range(config.bisection_steps):
                    mid = float(np.sqrt(low * high))
                    mid_weights, mid_solver = solve_at(mid)
                    mid_vol = float(np.sqrt(max(mid_weights @ Sigma @ mid_weights, 0.0)))
                    if mid_vol > target_monthly:
                        low = mid
                    else:
                        high = mid
                        weights, used_solver = mid_weights, mid_solver
                selected_lambda = high

        base = self._make_result(weights, "optimal", f"cvxpy_net_utility_{used_solver.lower()}")
        w_value = base.weights.to_numpy(float)
        long_value = np.maximum(w_value, 0.0)
        short_value = np.maximum(-w_value, 0.0)
        cost_value = float(long_cost @ long_value + short_cost @ short_value)
        net_scenarios = R_scenario @ w_value - cost_value
        losses = -net_scenarios
        quantile = float(np.quantile(losses, config.cvar_alpha, method="higher"))
        tail = losses[losses >= quantile - 1e-12]
        cvar = float(tail.mean()) if len(tail) else quantile
        stress = self._stress_matrix()
        worst_stress = float(np.min(stress @ w_value)) if stress is not None else np.nan
        margin_used = float(short_margin @ short_value)
        collateral_used = float(long_value.sum() + margin_used)
        annual_vol = float(base.volatility * np.sqrt(config.periods_per_year))
        stats: Dict[str, object] = {
            "objective": "r1_net_mean_variance_utility",
            "risk_aversion": float(selected_lambda),
            "gross_mean": float(mu @ w_value),
            "predictable_cost": cost_value,
            "net_mean": float(mu @ w_value) - cost_value,
            "variance_penalty": 0.5 * float(selected_lambda) * float(w_value @ Sigma @ w_value),
            "predicted_annual_vol": annual_vol,
            "cvar_alpha": config.cvar_alpha,
            "scenario_cvar_loss": cvar,
            "worst_stress_return": worst_stress,
            "short_margin_used": margin_used,
            "collateral_used": collateral_used,
            "cash_weight": max(0.0, 1.0 - collateral_used),
            "n_scenarios": int(len(R_scenario)),
        }
        return replace(base, objective_stats=stats)

    def _cvxpy_net_utility_constraints(
        self,
        w,
        long,
        short,
        scenario_returns: np.ndarray,
        predictable_cost,
        eta,
        config: NetUtilityConfig,
        short_margin: np.ndarray,
        short_allowed: np.ndarray,
        per_contract_caps: Optional[pd.Series],
    ) -> list:
        c = self.constraints
        cons = [cp.norm1(w) <= c.gross_nav]
        if c.long_only:
            cons.append(w >= 0)
        scalar_cap = c.gross_nav if c.per_contract_abs is None else c.per_contract_abs
        cap = np.repeat(float(scalar_cap), len(self.contracts))
        if per_contract_caps is not None:
            aligned_caps = per_contract_caps.reindex(self.contracts).astype(float)
            if aligned_caps.isna().any() or (aligned_caps < 0).any():
                raise ValueError("per_contract_caps must be finite, nonnegative, and complete")
            cap = np.minimum(cap, aligned_caps.to_numpy(float))
        cons.append(cp.abs(w) <= cap)
        if c.net_nav_abs is not None:
            cons.append(cp.abs(cp.sum(w)) <= c.net_nav_abs)
        if c.short_nav_abs is not None:
            cons.append(cp.sum(short) <= c.short_nav_abs)
        under_arr = self.frame["underlying"].astype(str).to_numpy()
        for underlying, limit in c.underlying_gross.items():
            idx = np.flatnonzero(under_arr == underlying)
            if len(idx):
                cons.append(cp.norm1(w[idx]) <= limit)
        for name, limit in [
            ("delta_nav", c.delta_abs),
            ("gamma_nav", c.gamma_abs),
            ("vega_nav", c.vega_abs),
            ("vix_vega_nav", c.vix_vega_abs),
            ("beta_spy_nav", c.beta_spy_abs),
        ]:
            if limit is not None:
                cons.append(cp.abs(self.greeks[name].to_numpy(float) @ w) <= limit)
        for exposure, limit in c.factor_exposure_abs.items():
            vector = self._named_exposure_vector(exposure)
            if vector is not None:
                cons.append(cp.abs(vector @ w) <= limit)
        stress = self._stress_matrix()
        if stress is not None:
            limit = min(
                config.stress_loss_nav,
                c.stress_loss_abs if c.stress_loss_abs is not None else config.stress_loss_nav,
            )
            cons.append(stress @ w >= -limit)
        cons.append(short_margin @ short <= config.short_margin_nav)
        cons.append(cp.sum(long) + short_margin @ short <= config.collateral_nav)
        if (~short_allowed).any():
            cons.append(w[np.flatnonzero(~short_allowed)] >= 0)
        losses = -(scenario_returns @ w - predictable_cost)
        tail_scale = (1.0 - config.cvar_alpha) * float(len(scenario_returns))
        cons.append(eta + cp.sum(cp.pos(losses - eta)) / tail_scale <= config.cvar_loss_nav)
        return cons

    def _cvxpy_run_solvers(self, problem, check_variable) -> tuple:
        """Try installed conic solvers in preference order.

        Returns ``(used_solver, saw_unbounded)``.  ``used_solver`` is the
        first solver that reported ``optimal``/``optimal_inaccurate`` with
        populated variable values, or ``None`` if every attempt failed.
        ``saw_unbounded`` flags an ``unbounded`` report, which for the
        homogenized Sortino program certifies a feasible downside-free
        direction with positive net mean (an unbounded ratio).
        """

        saw_unbounded = False
        for solver in ("CLARABEL", "ECOS", "SCS"):
            if solver not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=solver, verbose=False, warm_start=True)
            except Exception:
                continue
            if problem.status in ("optimal", "optimal_inaccurate") and check_variable.value is not None:
                return solver, saw_unbounded
            if problem.status in ("unbounded", "unbounded_inaccurate"):
                saw_unbounded = True
        return None, saw_unbounded

    # ------------------------------------------------------------------
    # Cost-aware maximum-Sortino solver
    # ------------------------------------------------------------------

    def solve_max_sortino(
        self,
        scenario_returns: pd.DataFrame,
        entry_costs: Optional[pd.Series] = None,
        target: float = 0.0,
        method: str = "cvxpy",
        raise_on_infeasible: bool = False,
    ) -> OptionMarkowitzResult:
        """Maximize the Sortino ratio net of expected entry costs.

        With ``mu`` the model expected returns (per holding period), ``c >= 0``
        the expected entry frictions per unit of ``|q|`` (half-spread plus
        fees as return-on-premium) and ``R`` the ``T x n`` matrix of training
        scenario returns, the solver maximizes over the existing constraint
        set ``K``::

            maximize  m(q) / DD(q),
            m(q)  = mu'q - c'|q|                       (net mean),
            DD(q) = sqrt((1/T) sum_t max(tau - R_t'q, 0)^2)
                                                       (downside deviation),

        where ``tau`` is ``target`` (per holding period, default 0).  ``m``
        is concave, ``DD`` is convex, and both are positively homogeneous for
        ``tau = 0``, so the Charnes-Cooper reduction (split legs ``y+``/``y-``
        with ``y = t q`` and scale ``t >= 0``) turns the fractional program
        into the SOCP ``max mu'y - c'(y+ + y-)`` subject to ``DD``
        homogenized to at most one -- ``u_t >= tau*t - R_t'y``, ``u >= 0``,
        ``||u||_2 <= sqrt(T)`` -- and every budget of ``K`` homogenized by
        ``t``.  The ``tau*t`` term keeps the reduction exact for ``tau != 0``.
        The portfolio is recovered as ``q = y / t``.

        Data handling: ``scenario_returns`` columns are reindexed to
        ``self.contracts``.  Contracts that are entirely missing (absent
        column or all-NaN) raise ``ValueError``; rows that are all-NaN are
        dropped; sporadic missing cells are filled with ``0.0`` (a missing
        mark contributes zero scenario P&L, which mirrors
        ``portfolio_return_series(fill_policy='zero')`` and mildly understates
        downside for gappy contracts).  ``entry_costs`` is reindexed to the
        contracts with missing entries set to ``0`` and must be finite and
        nonnegative.

        Degenerate downside-free case: if some feasible ``q`` has
        ``DD(q) = 0`` with ``m(q) > 0`` (no training scenario produces a
        shortfall below ``tau``), the ratio is unbounded.  This is detected
        up front (the net-mean LP maximizer has zero downside) or via an
        ``unbounded`` SOCP status, and the solver falls back to maximizing
        the net mean ``m(q)`` over the budgets alone.  The fallback result is
        tagged with solver ``'linprog_net_mean_downside_free'`` and
        ``objective_stats['degenerate_downside_free'] = True``; its net
        Sortino (and the ``sharpe`` field) is ``inf`` when the returned
        portfolio itself has zero downside.

        ``method='cvxpy'`` (default) solves the homogenized SOCP and falls
        back to SLSQP automatically if cvxpy is unavailable or the conic
        solve fails.  ``method='slsqp'`` maximizes the ratio directly with
        the same split-variable multi-start machinery as ``solve_max_sharpe``.

        Status semantics match ``solve_max_sharpe``: ``optimal`` /
        ``feasible_suboptimal`` / ``infeasible`` (max violation above
        ``1e-5``), with ``max_violation`` populated; pass
        ``raise_on_infeasible=True`` to raise instead of returning an
        infeasible result.  The returned ``sharpe`` field holds the net
        Sortino ratio at the optimum and ``objective_stats`` reports the net
        mean, entry cost, downside deviation, net Sortino, target and the
        degenerate flag.
        """

        if method not in ("cvxpy", "slsqp"):
            raise ValueError(f"Unknown method {method!r}; expected 'cvxpy' or 'slsqp'")
        R, costs, tau = self._prepare_sortino_inputs(scenario_returns, entry_costs, target)
        result = self._sortino_degenerate_precheck(R, costs, tau)
        if result is None:
            if method == "cvxpy" and _HAS_CVXPY:
                try:
                    result = self._solve_sortino_cvxpy(R, costs, tau)
                except (RuntimeError, ImportError):
                    result = self._solve_sortino_scipy(R, costs, tau)
            else:
                result = self._solve_sortino_scipy(R, costs, tau)
        if raise_on_infeasible and result.status == "infeasible":
            raise ValueError(
                "solve_max_sortino: constraints are infeasible "
                f"(max violation {result.max_violation:.6g} > 1e-5)"
            )
        return result

    def _prepare_sortino_inputs(
        self,
        scenario_returns: pd.DataFrame,
        entry_costs: Optional[pd.Series],
        target: float,
    ) -> tuple:
        """Align, validate and convert the Sortino solver inputs."""

        n = len(self.contracts)
        missing = [c for c in self.contracts if c not in scenario_returns.columns]
        if missing:
            raise ValueError(f"scenario_returns is missing contracts: {missing}")
        aligned = scenario_returns.reindex(columns=self.contracts)
        all_nan = [c for c in self.contracts if aligned[c].isna().all()]
        if all_nan:
            raise ValueError(f"scenario_returns has all-NaN contracts: {all_nan}")
        aligned = aligned.dropna(how="all")
        if aligned.empty:
            raise ValueError("scenario_returns has no usable scenario rows")
        R = aligned.fillna(0.0).to_numpy(dtype=float)
        if not np.isfinite(R).all():
            raise ValueError("scenario_returns must be finite")
        if entry_costs is None:
            costs = np.zeros(n, dtype=float)
        else:
            costs = (
                pd.Series(entry_costs).reindex(self.contracts).astype(float).fillna(0.0).to_numpy(dtype=float)
            )
            if not np.isfinite(costs).all() or (costs < 0).any():
                raise ValueError("entry_costs must be finite and nonnegative")
        tau = float(target)
        if not np.isfinite(tau):
            raise ValueError("target must be finite")
        return R, costs, tau

    @staticmethod
    def _downside_deviation(R: np.ndarray, weights: np.ndarray, target: float) -> float:
        """Empirical downside deviation sqrt(mean(max(target - R q, 0)^2))."""

        shortfall = np.maximum(target - R @ np.asarray(weights, dtype=float), 0.0)
        return float(np.sqrt(np.mean(shortfall * shortfall)))

    def _sortino_net_mean(self, weights: np.ndarray, costs: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        return float(self.expected_returns.to_numpy(dtype=float) @ w) - float(costs @ np.abs(w))

    def _sortino_degenerate_precheck(
        self, R: np.ndarray, costs: np.ndarray, target: float
    ) -> Optional[OptionMarkowitzResult]:
        """Detect the unbounded downside-free case before solving the ratio.

        The net-mean LP maximizer ``q* = argmax_K m(q)`` certifies the
        degenerate case whenever ``m(q*) > 0`` and ``DD(q*) = 0``; the
        documented fallback then returns ``q*`` itself.
        """

        n = len(self.contracts)
        lp = self._linear_feasible_split_start(costs=costs)
        if lp is None:
            return None
        q = lp[:n] - lp[n:]
        if self._sortino_net_mean(q, costs) > 1e-12 and self._downside_deviation(R, q, target) <= 1e-12:
            return self._make_sortino_result(
                q, "optimal", "linprog_net_mean_downside_free", R, costs, target, degenerate=True
            )
        return None

    def _solve_sortino_net_mean_fallback(
        self, R: np.ndarray, costs: np.ndarray, target: float
    ) -> OptionMarkowitzResult:
        """Documented fallback: maximize the net mean over the budgets alone."""

        lp = self._linear_feasible_split_start(costs=costs)
        if lp is None:
            raise RuntimeError(
                "Sortino ratio is unbounded (downside-free direction) and the "
                "net-mean LP fallback failed"
            )
        n = len(self.contracts)
        q = lp[:n] - lp[n:]
        return self._make_sortino_result(
            q, "optimal", "linprog_net_mean_downside_free", R, costs, target, degenerate=True
        )

    def _solve_sortino_cvxpy(
        self, R: np.ndarray, costs: np.ndarray, target: float
    ) -> OptionMarkowitzResult:
        """Homogenized (Charnes-Cooper) cost-aware max-Sortino SOCP.

        Variables are the split legs ``y+ >= 0``, ``y- >= 0``
        (``y = y+ - y- = t q``), the scale ``t >= 0`` and the shortfall
        auxiliaries ``u`` with ``u_t >= tau*t - R_t'y``, ``u >= 0`` and
        ``||u||_2 <= sqrt(T)`` (the SOC representation of the homogenized
        ``DD <= 1``).  The objective ``mu'y - c'(y+ + y-)`` equals
        ``t (mu'q - c'|q|)`` at any non-overlapping split, so the optimum is
        the maximum net Sortino ratio and ``q = y / t``.
        """

        if not _HAS_CVXPY:
            raise ImportError(
                "cvxpy is required for solve_max_sortino(method='cvxpy'); "
                "install cvxpy or use method='slsqp'"
            )
        n = len(self.contracts)
        T = R.shape[0]
        mu = self.expected_returns.to_numpy(dtype=float)
        y_pos = cp.Variable(n, nonneg=True)
        y_neg = cp.Variable(n, nonneg=True)
        t = cp.Variable(nonneg=True)
        u = cp.Variable(T, nonneg=True)
        y = y_pos - y_neg
        cons = [
            u >= target * t - R @ y,
            cp.norm(u, 2) <= float(np.sqrt(T)),
        ]
        cons.extend(self._cvxpy_homogenized_budget_constraints(y_pos, y_neg, t))
        problem = cp.Problem(cp.Maximize(mu @ y - costs @ (y_pos + y_neg)), cons)
        used_solver, saw_unbounded = self._cvxpy_run_solvers(problem, y_pos)
        if used_solver is None:
            if saw_unbounded:
                return self._solve_sortino_net_mean_fallback(R, costs, target)
            raise RuntimeError(
                f"cvxpy Sortino SOCP solve failed (last status: {problem.status!r})"
            )
        solver_tag = f"cvxpy_sortino_socp_{used_solver.lower()}"
        t_val = float(t.value) if t.value is not None else 0.0
        if not np.isfinite(t_val) or t_val <= 1e-9:
            return self._make_sortino_result(np.zeros(n), "infeasible", solver_tag, R, costs, target)
        q = (np.asarray(y_pos.value, dtype=float) - np.asarray(y_neg.value, dtype=float)).ravel() / t_val
        violation = self._max_constraint_violation(q)
        status = "optimal" if violation <= 1e-5 else "infeasible"
        return self._make_sortino_result(q, status, solver_tag, R, costs, target)

    def _solve_sortino_scipy(
        self, R: np.ndarray, costs: np.ndarray, target: float
    ) -> OptionMarkowitzResult:
        """Direct ratio maximization with the split-variable SLSQP machinery."""

        mu = self.expected_returns.to_numpy(dtype=float)
        n = len(mu)
        starts = self._split_variable_starts()
        lp_cost_aware = self._linear_feasible_split_start(costs=costs)
        if lp_cost_aware is not None:
            starts.insert(0, lp_cost_aware)

        def neg_sortino(x: np.ndarray) -> float:
            q = x[:n] - x[n:]
            net_mean = float(mu @ q) - float(costs @ (x[:n] + x[n:]))
            shortfall = np.maximum(target - R @ q, 0.0)
            dd = float(np.sqrt(max(float(np.mean(shortfall * shortfall)), 1e-18)))
            wasted_gross = float((x[:n] + x[n:]).sum() - np.abs(q).sum())
            return -net_mean / dd + 1e-7 * max(wasted_gross, 0.0)

        constraints = self._split_constraints()
        c = self.constraints
        hi = c.gross_nav if c.per_contract_abs is None else c.per_contract_abs
        short_hi = 0.0 if c.long_only else hi
        bounds = [(0.0, hi)] * n + [(0.0, short_hi)] * n

        best = None
        for x0 in starts:
            res = optimize.minimize(
                neg_sortino,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            q = np.asarray(res.x[:n] - res.x[n:], dtype=float)
            violation = self._max_constraint_violation(q)
            net_mean = self._sortino_net_mean(q, costs)
            dd = self._downside_deviation(R, q, target)
            if dd > 1e-12:
                ratio = net_mean / dd
            else:
                ratio = np.inf if net_mean > 0 else -np.inf
            candidate = (violation, -ratio, res.success, q)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        assert best is not None
        violation, _, success, weights = best
        if violation > 1e-5:
            status = "infeasible"
        elif success:
            status = "optimal"
        else:
            status = "feasible_suboptimal"
        return self._make_sortino_result(weights, status, "scipy_slsqp_sortino_split", R, costs, target)

    def _make_sortino_result(
        self,
        weights: np.ndarray,
        status: str,
        solver: str,
        R: np.ndarray,
        costs: np.ndarray,
        target: float,
        degenerate: bool = False,
    ) -> OptionMarkowitzResult:
        """Attach net-Sortino diagnostics to the standard result payload."""

        base = self._make_result(weights, status, solver)
        w = base.weights.to_numpy(dtype=float)
        gross_mean = float(self.expected_returns.to_numpy(dtype=float) @ w)
        entry_cost = float(costs @ np.abs(w))
        net_mean = gross_mean - entry_cost
        dd = self._downside_deviation(R, w, target)
        if dd > 1e-12:
            ratio = net_mean / dd
        elif net_mean > 0:
            ratio = float("inf")
            degenerate = True
        else:
            ratio = float("nan")
        stats: Dict[str, object] = {
            "objective": "max_sortino_net_of_costs",
            "net_mean": net_mean,
            "gross_mean": gross_mean,
            "entry_cost": entry_cost,
            "downside_deviation": dd,
            "sortino_net": ratio,
            "target": float(target),
            "n_scenarios": int(R.shape[0]),
            "degenerate_downside_free": bool(degenerate),
        }
        return replace(base, sharpe=ratio, objective_stats=stats)

    def _solve_scipy(self) -> OptionMarkowitzResult:
        mu = self.expected_returns.to_numpy(dtype=float)
        Sigma = nearest_psd(self.option_cov)
        n = len(mu)
        starts = self._split_variable_starts()

        def neg_sharpe(x: np.ndarray) -> float:
            q = x[:n] - x[n:]
            vol = float(np.sqrt(max(q @ Sigma @ q, 1e-18)))
            wasted_gross = float((x[:n] + x[n:]).sum() - np.abs(q).sum())
            return -float(mu @ q) / vol + 1e-7 * max(wasted_gross, 0.0)

        constraints = self._split_constraints()
        c = self.constraints
        hi = c.gross_nav if c.per_contract_abs is None else c.per_contract_abs
        long_bounds = [(0.0, hi)] * n
        short_hi = 0.0 if self.constraints.long_only else hi
        bounds = long_bounds + [(0.0, short_hi)] * n

        best = None
        for x0 in starts:
            res = optimize.minimize(
                neg_sharpe,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            q = np.asarray(res.x[:n] - res.x[n:], dtype=float)
            violation = self._max_constraint_violation(q)
            vol = float(np.sqrt(max(q @ Sigma @ q, 1e-18)))
            sharpe = float(mu @ q) / vol if vol > 0 else -np.inf
            candidate = (violation, -sharpe, res.success, q)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        assert best is not None
        violation, _, success, weights = best
        if violation > 1e-5:
            status = "infeasible"
        elif success:
            status = "optimal"
        else:
            status = "feasible_suboptimal"
        return self._make_result(weights, status, "scipy_slsqp_split")

    def _split_constraints(self):
        n = len(self.contracts)
        c = self.constraints
        constraints = [{"type": "eq", "fun": lambda x: (x[:n] + x[n:]).sum() - c.gross_nav}]
        if c.net_nav_abs is not None:
            constraints.extend(
                [
                    {"type": "ineq", "fun": lambda x: c.net_nav_abs - (x[:n] - x[n:]).sum()},
                    {"type": "ineq", "fun": lambda x: c.net_nav_abs + (x[:n] - x[n:]).sum()},
                ]
            )
        if c.short_nav_abs is not None:
            constraints.append({"type": "ineq", "fun": lambda x: c.short_nav_abs - x[n:].sum()})
        for under, limit in self.constraints.underlying_gross.items():
            idx = np.flatnonzero(self.frame["underlying"].astype(str).to_numpy() == under)
            if len(idx):
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, idx=idx, limit=limit: limit - (x[idx] + x[n + idx]).sum(),
                    }
                )
        for name, limit in [
            ("delta_nav", c.delta_abs),
            ("gamma_nav", c.gamma_abs),
            ("vega_nav", c.vega_abs),
            ("vix_vega_nav", c.vix_vega_abs),
            ("beta_spy_nav", c.beta_spy_abs),
        ]:
            if limit is not None:
                v = self.greeks[name].to_numpy(dtype=float)
                constraints.extend(
                    [
                        {"type": "ineq", "fun": lambda x, v=v, limit=limit: limit - v @ (x[:n] - x[n:])},
                        {"type": "ineq", "fun": lambda x, v=v, limit=limit: limit + v @ (x[:n] - x[n:])},
                    ]
                )
        for exposure, limit in c.factor_exposure_abs.items():
            v = self._named_exposure_vector(exposure)
            if v is not None:
                constraints.extend(
                    [
                        {"type": "ineq", "fun": lambda x, v=v, limit=limit: limit - v @ (x[:n] - x[n:])},
                        {"type": "ineq", "fun": lambda x, v=v, limit=limit: limit + v @ (x[:n] - x[n:])},
                    ]
                )
        R = self._stress_matrix()
        if c.stress_loss_abs is not None and R is not None:
            constraints.append({"type": "ineq", "fun": lambda x, R=R, limit=c.stress_loss_abs: float(np.min(R @ (x[:n] - x[n:]))) + limit})
        return constraints

    def _split_variable_starts(self) -> list[np.ndarray]:
        n = len(self.contracts)
        starts: list[np.ndarray] = []
        for q in [self.tangency_weights().to_numpy(dtype=float), np.ones(n, dtype=float)]:
            q = self._make_feasible_start(q)
            starts.append(np.r_[np.maximum(q, 0.0), np.maximum(-q, 0.0)])
        lp = self._linear_feasible_split_start()
        if lp is not None:
            starts.insert(0, lp)
        return starts

    def _linear_feasible_split_start(self, costs: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Maximize ``mu'q`` (or ``mu'q - c'|q|`` when ``costs`` is given) over
        the budget set via a split-leg LP; used as a warm start and as the
        downside-free net-mean fallback of ``solve_max_sortino``."""

        n = len(self.contracts)
        c = self.constraints
        hi = c.gross_nav if c.per_contract_abs is None else c.per_contract_abs
        bounds = [(0.0, hi)] * n
        short_hi = 0.0 if c.long_only else hi
        bounds.extend([(0.0, short_hi)] * n)
        a_eq = [np.r_[np.ones(n), np.ones(n)]]
        b_eq = [c.gross_nav]
        a_ub = []
        b_ub = []
        if c.net_nav_abs is not None:
            net = np.r_[np.ones(n), -np.ones(n)]
            a_ub.extend([net, -net])
            b_ub.extend([c.net_nav_abs, c.net_nav_abs])
        if c.short_nav_abs is not None:
            a_ub.append(np.r_[np.zeros(n), np.ones(n)])
            b_ub.append(c.short_nav_abs)
        under_arr = self.frame["underlying"].astype(str).to_numpy()
        for under, limit in c.underlying_gross.items():
            idx = np.flatnonzero(under_arr == under)
            if len(idx):
                row = np.zeros(2 * n)
                row[idx] = 1.0
                row[n + idx] = 1.0
                a_ub.append(row)
                b_ub.append(limit)
        for name, limit in [
            ("delta_nav", c.delta_abs),
            ("gamma_nav", c.gamma_abs),
            ("vega_nav", c.vega_abs),
            ("vix_vega_nav", c.vix_vega_abs),
            ("beta_spy_nav", c.beta_spy_abs),
        ]:
            if limit is not None:
                vec = self.greeks[name].to_numpy(dtype=float)
                row = np.r_[vec, -vec]
                a_ub.extend([row, -row])
                b_ub.extend([limit, limit])
        for exposure, limit in c.factor_exposure_abs.items():
            vec = self._named_exposure_vector(exposure)
            if vec is not None:
                row = np.r_[vec, -vec]
                a_ub.extend([row, -row])
                b_ub.extend([limit, limit])
        R = self._stress_matrix()
        if c.stress_loss_abs is not None and R is not None:
            for row in R:
                a_ub.append(-np.r_[row, -row])
                b_ub.append(c.stress_loss_abs)
        if costs is None:
            objective = -np.r_[self.expected_returns.to_numpy(dtype=float), -self.expected_returns.to_numpy(dtype=float)]
        else:
            mu = self.expected_returns.to_numpy(dtype=float)
            objective = -np.r_[mu - costs, -mu - costs]
        try:
            res = optimize.linprog(
                objective,
                A_ub=np.vstack(a_ub) if a_ub else None,
                b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
                A_eq=np.vstack(a_eq),
                b_eq=np.asarray(b_eq, dtype=float),
                bounds=bounds,
                method="highs",
            )
        except Exception:
            return None
        if res.success and res.x is not None:
            return np.asarray(res.x, dtype=float)
        return None

    def _max_constraint_violation(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        c = self.constraints
        # Project onto non-overlapping legs before checking, and treat the
        # gross budget as one-sided: deploying less than the full gross NAV
        # is feasible, only exceeding it is a violation.
        pos = np.maximum(w, 0.0)
        neg = np.maximum(-w, 0.0)
        gross = float(pos.sum() + neg.sum())
        violations = [float(max(0.0, gross - c.gross_nav))]
        if c.long_only:
            violations.append(float(max(0.0, -w.min())))
        if c.net_nav_abs is not None:
            violations.append(float(max(0.0, abs(w.sum()) - c.net_nav_abs)))
        if c.short_nav_abs is not None:
            violations.append(float(max(0.0, np.maximum(-w, 0.0).sum() - c.short_nav_abs)))
        if c.per_contract_abs is not None:
            violations.append(float(max(0.0, np.abs(w).max() - c.per_contract_abs)))
        for under, limit in c.underlying_gross.items():
            idx = self.frame["underlying"].astype(str).to_numpy() == under
            if idx.any():
                violations.append(float(max(0.0, np.abs(w[idx]).sum() - limit)))
        for name, limit in [
            ("delta_nav", c.delta_abs),
            ("gamma_nav", c.gamma_abs),
            ("vega_nav", c.vega_abs),
            ("vix_vega_nav", c.vix_vega_abs),
            ("beta_spy_nav", c.beta_spy_abs),
        ]:
            if limit is not None:
                val = float(self.greeks[name].to_numpy(dtype=float) @ w)
                violations.append(float(max(0.0, abs(val) - limit)))
        for exposure, limit in c.factor_exposure_abs.items():
            vec = self._named_exposure_vector(exposure)
            if vec is not None:
                val = float(vec @ w)
                violations.append(float(max(0.0, abs(val) - limit)))
        R = self._stress_matrix()
        if c.stress_loss_abs is not None and R is not None:
            worst = float(np.min(R @ w))
            violations.append(float(max(0.0, -c.stress_loss_abs - worst)))
        return max(violations)

    def _make_feasible_start(self, candidate: np.ndarray) -> np.ndarray:
        """Build a stable SLSQP starting point with the requested gross NAV."""

        x = np.asarray(candidate, dtype=float).copy()
        if not np.isfinite(x).all() or np.abs(x).sum() <= 1e-12:
            x = np.ones(len(self.contracts), dtype=float)
        if self.constraints.long_only:
            x = np.abs(x)
        if self.constraints.per_contract_abs is not None:
            x = np.clip(x, -self.constraints.per_contract_abs, self.constraints.per_contract_abs)
            if self.constraints.long_only:
                x = np.clip(x, 0.0, self.constraints.per_contract_abs)
        gross = np.abs(x).sum()
        if gross <= 1e-12:
            x = np.ones(len(self.contracts), dtype=float)
            gross = np.abs(x).sum()
        x *= self.constraints.gross_nav / gross
        # If per-contract clipping made gross infeasible, use equal weights.
        if self.constraints.per_contract_abs is not None and np.abs(x).max() > self.constraints.per_contract_abs + 1e-10:
            sign = np.sign(x)
            sign[sign == 0] = 1.0
            x = sign * min(self.constraints.per_contract_abs, self.constraints.gross_nav / len(x))
            total = float(np.abs(x).sum())
            if total > 1e-12:
                x *= self.constraints.gross_nav / total
            else:
                # per_contract_abs == 0: no gross can be deployed at all.
                x = np.zeros(len(self.contracts), dtype=float)
                return x
        if self.constraints.short_nav_abs is not None and not self.constraints.long_only:
            short = float(np.maximum(-x, 0.0).sum())
            limit = min(float(self.constraints.short_nav_abs), self.constraints.gross_nav)
            if short > limit + 1e-12:
                neg = x < 0
                pos = x > 0
                x[neg] *= limit / short if short > 0 else 0.0
                pos_gross = float(np.maximum(x, 0.0).sum())
                target_pos = self.constraints.gross_nav - float(np.maximum(-x, 0.0).sum())
                if pos_gross > 1e-12:
                    x[pos] *= target_pos / pos_gross
                else:
                    x[:] = self.constraints.gross_nav / len(x)
        return x

    def _make_result(self, weights: np.ndarray, status: str, solver: str) -> OptionMarkowitzResult:
        weights = np.asarray(weights, dtype=float)
        # Rescale only if a numerical solver drifted above the gross budget.
        gross = float(np.abs(weights).sum())
        if gross > self.constraints.gross_nav * (1.0 + 1e-7):
            weights = weights * (self.constraints.gross_nav / gross)
            gross = float(np.abs(weights).sum())
        max_violation = self._max_constraint_violation(weights)
        if max_violation > 1e-5:
            status = "infeasible"
        exp_ret = float(self.expected_returns.to_numpy(float) @ weights)
        variance = float(weights @ self.option_cov @ weights)
        vol = float(np.sqrt(max(variance, 0.0)))
        sharpe = exp_ret / vol if vol > 0 else np.nan
        series = pd.Series(weights, index=self.contracts, name="weight")
        return OptionMarkowitzResult(
            status=status,
            weights=series,
            expected_return=exp_ret,
            volatility=vol,
            sharpe=sharpe,
            gross_nav=gross,
            net_nav=float(weights.sum()),
            delta=float(self.greeks["delta_nav"].to_numpy(float) @ weights),
            gamma=float(self.greeks["gamma_nav"].to_numpy(float) @ weights),
            vega=float(self.greeks["vega_nav"].to_numpy(float) @ weights),
            solver=solver,
            max_violation=max_violation,
        )

    def equal_premium_weights(self) -> pd.Series:
        n = len(self.contracts)
        return pd.Series(np.repeat(self.constraints.gross_nav / n, n), index=self.contracts, name="weight")

    def equal_risk_weights(self) -> pd.Series:
        diag = np.sqrt(np.maximum(np.diag(self.option_cov), 1e-12))
        inv = 1.0 / diag
        weights = inv / inv.sum() * self.constraints.gross_nav
        return pd.Series(weights, index=self.contracts, name="weight")

    def portfolio_return_series(
        self,
        option_returns: pd.DataFrame,
        weights: pd.Series,
        fill_policy: str = "zero",
    ) -> pd.Series:
        """Return NAV growth from option-premium weights and option mark returns.

        If beginning NAV is ``W`` and ``q_i`` is the signed premium exposure,
        the contract count is ``q_i W / C_i``.  Mark-to-market P&L divided by
        NAV is therefore ``q_i * Delta C_i / C_i``.  The option premium paid or
        received is embedded in ``q_i`` and in the mark-return denominator.

        ``fill_policy`` controls missing-data handling.  The default
        ``'zero'`` (historical behavior) silently treats missing contracts or
        missing observations as a zero return, which understates both risk and
        P&L when the return panel has gaps — a held position with no mark
        contributes nothing that period.  Use ``'raise'`` to raise a
        ``ValueError`` when any portfolio contract is entirely missing from
        the panel or any aligned return is NaN.
        """

        if fill_policy not in ("zero", "raise"):
            raise ValueError(f"fill_policy must be 'zero' or 'raise', got {fill_policy!r}")
        aligned = option_returns.reindex(columns=self.contracts)
        if fill_policy == "raise":
            missing = [c for c in self.contracts if c not in option_returns.columns]
            if missing:
                raise ValueError(f"option_returns is missing contracts: {missing}")
            if aligned.isna().to_numpy().any():
                raise ValueError("option_returns contains NaN values under fill_policy='raise'")
        aligned = aligned.fillna(0.0)
        w = weights.reindex(self.contracts).fillna(0.0).to_numpy(float)
        return pd.Series(aligned.to_numpy(float) @ w, index=aligned.index)

    def risk_calibration(self, option_returns: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
        w = weights.reindex(self.contracts).fillna(0.0).to_numpy(float)
        pred = float(np.sqrt(max(w @ self.option_cov @ w, 0.0)))
        realized = float(self.portfolio_return_series(option_returns, weights).std(ddof=1))
        return {
            "predicted_vol": pred,
            "realized_vol": realized,
            "realized_to_predicted": realized / pred if pred > 0 else np.nan,
        }


__all__ = [
    "FactorShockSpec",
    "GreekJointMomentSpec",
    "NetUtilityConfig",
    "OptionMarkowitzConstraints",
    "OptionMarkowitzResult",
    "OptionOnlyMarkowitzModel",
    "OptionOnlySpec",
    "OptimizationCostSpec",
    "bootstrap_sharpe_ci",
    "estimate_greek_joint_moments",
    "greek_exposure_frame",
    "greek_factor_names",
    "nearest_psd",
    "performance_stats",
    "shrink_covariance",
    "taylor_option_pnl",
]
