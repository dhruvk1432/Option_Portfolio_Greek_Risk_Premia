"""Option-only Markowitz portfolio model.

This module implements the optimizer used by
``research/papers/option_only_markowitz``.  It deliberately treats listed
options as the only risky investable instruments.  Cash is a numeraire for
NAV and collateral accounting, not an optimized asset.

The risk model is the option analogue of Markowitz: contract P&L is mapped
to a small set of systematic shocks through Greeks, and the option covariance
matrix is assembled as

    Sigma_O = B Omega B' + Sigma_epsilon.

Positions are dollar/NAV weights in option contracts.  A value of ``0.10`` in
one contract means the portfolio has a ten percent NAV exposure to that
option's mark.  No transaction costs or slippage are modeled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:  # keep the module usable without cvxpy
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVXPY = False


OPTION_REQUIRED_COLUMNS = [
    "underlying",
    "mark",
    "delta",
    "gamma",
    "vega",
    "theta",
]


@dataclass(frozen=True)
class OptionOnlySpec:
    """Option universe inputs.

    ``frame`` is indexed by contract identifier.  The required columns are:
    ``underlying``, ``mark``, ``delta``, ``gamma``, ``vega`` and ``theta``.
    Greeks are per share or per contract in the same convention as ``mark``;
    the model rescales them into dollar-return exposures.
    """

    frame: pd.DataFrame
    multiplier: float = 100.0

    def validate(self) -> None:
        missing = [c for c in OPTION_REQUIRED_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"OptionOnlySpec.frame missing columns: {missing}")
        if self.frame.index.has_duplicates:
            raise ValueError("OptionOnlySpec.frame index must be unique contract ids")
        if (pd.to_numeric(self.frame["mark"], errors="coerce") <= 0).any():
            raise ValueError("Option marks must be strictly positive")
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
    """

    underlying_cov: pd.DataFrame
    vol_cov: Optional[pd.DataFrame] = None
    gamma_var_cov: Optional[pd.DataFrame] = None
    horizon_years: float = 21.0 / 252.0

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


def _validate_square_psd(frame: pd.DataFrame, name: str) -> None:
    if list(frame.index) != list(frame.columns):
        raise ValueError(f"{name} must have identical index and columns")
    values = frame.to_numpy(dtype=float)
    if not np.allclose(values, values.T, atol=1e-10):
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
    ) -> None:
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
        self.factor_cov = self._build_factor_covariance()
        factor_cov = self.B @ self.factor_cov @ self.B.T
        if residual_cov is None:
            resid = np.diag(np.maximum(np.diag(factor_cov), 1e-8)) * 0.05
        else:
            resid_frame = residual_cov.reindex(index=self.contracts, columns=self.contracts).fillna(0.0)
            resid = resid_frame.to_numpy(dtype=float)
        self.option_cov = shrink_covariance(factor_cov + nearest_psd(resid), covariance_shrinkage)
        self.greeks = self._portfolio_greek_loadings()

    def _build_exposure_matrix(self) -> np.ndarray:
        n, k = len(self.contracts), len(self.underlyings)
        col_count = 3 * k
        B = np.zeros((n, col_count), dtype=float)
        mark = self.frame["mark"].to_numpy(dtype=float)
        spot = self.frame.get("spot", pd.Series(1.0, index=self.frame.index)).to_numpy(dtype=float)
        dt = self.shocks.horizon_years
        under_to_col = {u: i for i, u in enumerate(self.underlyings)}
        for row, (_, rec) in enumerate(self.frame.iterrows()):
            u = str(rec["underlying"])
            j = under_to_col[u]
            S = float(rec.get("spot", 1.0))
            B[row, j] = float(rec["delta"]) * S / mark[row]
            B[row, k + j] = 0.5 * float(rec["gamma"]) * S * S / mark[row]
            B[row, 2 * k + j] = float(rec["vega"]) / mark[row]
        theta_return = self.frame["theta"].to_numpy(dtype=float) * dt / mark
        self.theta_return = pd.Series(theta_return, index=self.contracts, name="theta_return")
        return B

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
        return nearest_psd(out)

    def _portfolio_greek_loadings(self) -> pd.DataFrame:
        mark = self.frame["mark"].to_numpy(dtype=float)
        spot = self.frame.get("spot", pd.Series(1.0, index=self.frame.index)).to_numpy(dtype=float)
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

    def solve_max_sharpe(self) -> OptionMarkowitzResult:
        """Maximize the Sharpe ratio under NAV-normalized constraints.

        The unconstrained tangency portfolio has a closed form, but gross-NAV,
        sign and exposure budgets make the investable problem a fractional
        constrained program.  We therefore solve the ratio directly and enforce
        ``sum(abs(q)) == gross_nav`` so reported positions are comparable across
        candidate portfolios.
        """

        return self._solve_scipy()

    def _solve_cvxpy(self) -> OptionMarkowitzResult:
        n = len(self.contracts)
        q = cp.Variable(n)
        mu = self.expected_returns.to_numpy(dtype=float)
        Sigma = nearest_psd(self.option_cov)
        constraints = [cp.quad_form(q, Sigma) <= 1.0]
        constraints.extend(self._cvx_constraints(q))
        problem = cp.Problem(cp.Maximize(mu @ q), constraints)
        for solver in ("CLARABEL", "ECOS", "SCS"):
            try:
                if solver in cp.installed_solvers():
                    problem.solve(solver=solver, verbose=False)
                    if q.value is not None:
                        return self._make_result(np.asarray(q.value).ravel(), str(problem.status), solver)
            except Exception:
                continue
        return self._solve_scipy()

    def _cvx_constraints(self, q):
        c = self.constraints
        out = [cp.norm1(q) <= c.gross_nav]
        if c.long_only:
            out.append(q >= 0)
        if c.net_nav_abs is not None:
            out.append(cp.abs(cp.sum(q)) <= c.net_nav_abs)
        if c.short_nav_abs is not None:
            out.append(cp.sum(cp.pos(-q)) <= c.short_nav_abs)
        if c.per_contract_abs is not None:
            out.extend([q <= c.per_contract_abs, q >= -c.per_contract_abs])
        for under, limit in c.underlying_gross.items():
            idx = np.flatnonzero(self.frame["underlying"].astype(str).to_numpy() == under)
            if len(idx):
                out.append(cp.norm1(q[idx]) <= limit)
        for name, limit in [
            ("delta_nav", c.delta_abs),
            ("gamma_nav", c.gamma_abs),
            ("vega_nav", c.vega_abs),
            ("vix_vega_nav", c.vix_vega_abs),
            ("beta_spy_nav", c.beta_spy_abs),
        ]:
            if limit is not None:
                vec = self.greeks[name].to_numpy(dtype=float)
                out.append(cp.abs(vec @ q) <= limit)
        for exposure, limit in c.factor_exposure_abs.items():
            vec = self._named_exposure_vector(exposure)
            if vec is not None:
                out.append(cp.abs(vec @ q) <= limit)
        R = self._stress_matrix()
        if c.stress_loss_abs is not None and R is not None:
            out.append(cp.min(R @ q) >= -c.stress_loss_abs)
        return out

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
        hi = self.constraints.per_contract_abs or self.constraints.gross_nav
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
        status = "optimal" if success and violation <= 1e-5 else "feasible_suboptimal"
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

    def _linear_feasible_split_start(self) -> Optional[np.ndarray]:
        n = len(self.contracts)
        c = self.constraints
        hi = c.per_contract_abs or c.gross_nav
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
        objective = -np.r_[self.expected_returns.to_numpy(dtype=float), -self.expected_returns.to_numpy(dtype=float)]
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
        violations = [abs(float(np.abs(w).sum() - c.gross_nav))]
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
            x *= self.constraints.gross_nav / np.abs(x).sum()
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
        )

    def equal_premium_weights(self) -> pd.Series:
        n = len(self.contracts)
        return pd.Series(np.repeat(self.constraints.gross_nav / n, n), index=self.contracts, name="weight")

    def equal_risk_weights(self) -> pd.Series:
        diag = np.sqrt(np.maximum(np.diag(self.option_cov), 1e-12))
        inv = 1.0 / diag
        weights = inv / inv.sum() * self.constraints.gross_nav
        return pd.Series(weights, index=self.contracts, name="weight")

    def portfolio_return_series(self, option_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """Return NAV growth from option-premium weights and option mark returns.

        If beginning NAV is ``W`` and ``q_i`` is the signed premium exposure,
        the contract count is ``q_i W / C_i``.  Mark-to-market P&L divided by
        NAV is therefore ``q_i * Delta C_i / C_i``.  The option premium paid or
        received is embedded in ``q_i`` and in the mark-return denominator.
        """

        aligned = option_returns.reindex(columns=self.contracts).fillna(0.0)
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
    "OptionMarkowitzConstraints",
    "OptionMarkowitzResult",
    "OptionOnlyMarkowitzModel",
    "OptionOnlySpec",
    "bootstrap_sharpe_ci",
    "nearest_psd",
    "performance_stats",
    "shrink_covariance",
    "taylor_option_pnl",
]
