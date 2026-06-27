"""Multi-asset derivative-aware portfolio optimization with endogenous
universe selection, hedging, and roll costs.

This module implements the general model of the companion paper

    "The Friction-Deflated Span: Robust Multi-Asset Portfolio Choice
     with Options, Futures, and Endogenous Implementation Costs"
    (research/papers/multi_asset_derivatives/main.tex in this repository).

The central object is :class:`MultiAssetDerivativePortfolioModel`, a
strategy- and asset-class-agnostic optimizer over a joint universe of
cash, equities, ETFs, listed options, financial futures, commodity
futures, and options on futures.  Three primitives distinguish it from
the repository's earlier stock+option model
(``mixed_asset_options_portfolio_model.py``):

1.  *Friction-deflated span*.  Every instrument's contribution to the
    objective is deflated by its full implementation cost path: entry
    spread/impact, endogenous gamma-hedging cost (degree-4/3
    homogeneous), futures/commodity roll cost (degree-1), funding and
    margin drag, and a data-quality penalty.  The attainable payoff set
    is convex but *not* a linear span, which is what makes universe
    selection a non-trivial optimization in its own right.

2.  *Admission gauge / no-admission band*.  Exclusion of instrument i
    is characterized by a subdifferential condition

        |mu_i - kappa * (U x*)_i / ||U^{1/2} x*|| - lam * (Sigma x*)_i|
            <= t_i,

    where t_i collects all degree-1 cost rates (spread, roll, quality
    penalty).  Because the hedging cost is homogeneous of degree 4/3 it
    has *zero* marginal cost at x_i = 0: gamma-hedging frictions never
    veto admission, they only cap size.  :meth:`admission_gauge` and
    :meth:`select_universe` implement the resulting active-set
    algorithm.

3.  *Roll operator and delivery avoidance*.  Futures and commodity
    contracts carry a roll intensity and a first-notice calendar.
    Physically settled contracts must be rolled or closed before
    first notice minus a liquidity buffer; contracts that cannot
    satisfy this are inadmissible (hard filter), and admissible ones
    are charged an explicit annualized roll cost in the objective.

The robust layer uses a *regime-coupled ambiguity field*
kappa(p) = kappa_0 + kappa_1 * p, where p in [0,1] is the filtered
stress-regime posterior, so ambiguity aversion endogenously tightens in
stress.

Solvers: CVXPY (Clarabel/ECOS/SCS) when available, then SciPy SLSQP on
a smoothed objective, then a projected random search.  All paths return
the same :class:`PortfolioResult` schema.

No credentials are read anywhere in this module; data access is by
plain numpy/pandas inputs or whitelisted local CSV caches.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:  # pragma: no cover - environment dependent
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVXPY = False

__all__ = [
    "AssetClass", "Settlement", "InstrumentSpec", "CostModel",
    "RegimeAmbiguity", "ConstraintSet", "DeliveryPolicy", "Strategy",
    "StrategyRestriction", "HedgePolicy", "PortfolioResult",
    "MultiAssetDerivativePortfolioModel",
    "bs_price", "bs_greeks", "implied_vol",
    "perf_stats", "block_bootstrap_ci", "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]


# --------------------------------------------------------------------------
# Black-Scholes utilities (self-contained so the module has no intra-repo
# import dependency; numerically identical to the earlier module's versions)
# --------------------------------------------------------------------------

def _d1d2(S: float, K: float, T: float, r: float, sigma: float,
          q_div: float = 0.0) -> Tuple[float, float]:
    if min(S, K, T, sigma) <= 0:
        raise ValueError("S, K, T, sigma must be positive")
    d1 = (math.log(S / K) + (r - q_div + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", q_div: float = 0.0) -> float:
    """Black-Scholes-Merton European option price per share."""
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma, q_div)
    if kind == "call":
        return S * math.exp(-q_div * T) * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
    if kind == "put":
        return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * math.exp(-q_div * T) * stats.norm.cdf(-d1)
    raise ValueError(f"unknown option kind {kind!r}")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              kind: str = "call", q_div: float = 0.0) -> Dict[str, float]:
    """Delta, gamma (per share), vega (per unit vol), theta (per year)."""
    if T <= 0:
        itm = (S > K) if kind == "call" else (S < K)
        return {"delta": (1.0 if kind == "call" else -1.0) * float(itm),
                "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1, d2 = _d1d2(S, K, T, r, sigma, q_div)
    pdf = stats.norm.pdf(d1)
    disc_q, disc_r = math.exp(-q_div * T), math.exp(-r * T)
    delta = disc_q * stats.norm.cdf(d1) if kind == "call" else -disc_q * stats.norm.cdf(-d1)
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)
    if kind == "call":
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T))
                 - r * K * disc_r * stats.norm.cdf(d2) + q_div * S * disc_q * stats.norm.cdf(d1))
    else:
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T))
                 + r * K * disc_r * stats.norm.cdf(-d2) - q_div * S * disc_q * stats.norm.cdf(-d1))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                kind: str = "call", q_div: float = 0.0) -> float:
    """Invert BSM via Brent bracketing; returns nan if outside bounds."""
    if price <= 0 or T <= 0:
        return float("nan")
    intrinsic = max(S * math.exp(-q_div * T) - K * math.exp(-r * T), 0.0) if kind == "call" \
        else max(K * math.exp(-r * T) - S * math.exp(-q_div * T), 0.0)
    if price < intrinsic - 1e-12:
        return float("nan")
    f = lambda sig: bs_price(S, K, T, r, sig, kind, q_div) - price
    lo, hi = 1e-4, 5.0
    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return float(optimize.brentq(f, lo, hi, xtol=1e-10))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# Specification dataclasses
# --------------------------------------------------------------------------

class AssetClass(str, Enum):
    CASH = "cash"
    EQUITY = "equity"
    ETF = "etf"
    OPTION = "option"                      # listed option on equity/ETF/index
    FUTURE = "future"                      # financial future (index, rates, FX)
    COMMODITY_FUTURE = "commodity_future"
    COMMODITY_OPTION = "commodity_option"  # option on a commodity future


LINEAR_CLASSES = {AssetClass.EQUITY, AssetClass.ETF, AssetClass.FUTURE,
                  AssetClass.COMMODITY_FUTURE}
OPTION_CLASSES = {AssetClass.OPTION, AssetClass.COMMODITY_OPTION}
FUTURE_CLASSES = {AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE,
                  AssetClass.COMMODITY_OPTION}


class Settlement(str, Enum):
    CASH = "cash"
    PHYSICAL = "physical"


@dataclass
class InstrumentSpec:
    """One tradable instrument.  All exposures are *notional weights*:
    x_i = dollar notional exposure / wealth.  For equities/ETFs this is the
    ordinary portfolio weight; for futures it is contract notional over
    wealth; for options it is *underlying* notional over wealth (premium
    outlay is x_i * premium/spot).

    ``mu`` and ``sigma`` are annualized and per unit of notional weight.
    ``half_spread`` is relative to the instrument's own price (option
    spreads are quoted on premium and converted internally).
    """
    instrument_id: str
    asset_class: AssetClass
    price: float                      # instrument price (option premium per share)
    mu: float = 0.0                   # annualized expected excess return per notional
    sigma: float = 0.0                # annualized vol per notional (linear instruments)
    half_spread: float = 0.0005       # relative half-spread on own price
    margin_rate: float = 0.0          # initial margin per unit |notional|
    capacity: float = 1.0             # max |x_i| from liquidity/participation
    data_quality: float = 1.0         # in (0,1]; <1 adds admission penalty
    underlying: Optional[str] = None  # driver id for derivatives
    # option fields
    strike: Optional[float] = None
    maturity_years: Optional[float] = None
    kind: Optional[str] = None        # 'call' | 'put'
    iv: Optional[float] = None
    dividend_yield: float = 0.0
    open_interest: Optional[float] = None
    # futures fields
    settlement: Settlement = Settlement.CASH
    rolls_per_year: float = 0.0
    roll_half_spread: float = 0.0     # relative cost per side at each roll
    first_notice_days: Optional[float] = None  # days until first notice/delivery window
    carry: float = 0.0                # annualized carry estimate (roll yield), optional

    def validate(self) -> None:
        if self.price <= 0:
            raise ValueError(f"{self.instrument_id}: price must be > 0")
        if not (0 < self.data_quality <= 1):
            raise ValueError(f"{self.instrument_id}: data_quality must be in (0,1]")
        if self.half_spread < 0 or self.margin_rate < 0 or self.capacity <= 0:
            raise ValueError(f"{self.instrument_id}: negative friction/capacity")
        if self.asset_class in OPTION_CLASSES:
            for f in ("strike", "maturity_years", "kind", "iv", "underlying"):
                if getattr(self, f) is None:
                    raise ValueError(f"{self.instrument_id}: option missing field {f}")
            if self.kind not in ("call", "put"):
                raise ValueError(f"{self.instrument_id}: kind must be call/put")
            if self.maturity_years <= 0 or self.iv <= 0:
                raise ValueError(f"{self.instrument_id}: bad maturity/iv")
        if self.asset_class in (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE):
            if self.rolls_per_year < 0:
                raise ValueError(f"{self.instrument_id}: rolls_per_year < 0")
        if self.asset_class in LINEAR_CLASSES and self.sigma < 0:
            raise ValueError(f"{self.instrument_id}: sigma < 0")


@dataclass
class CostModel:
    """Friction parameters shared across the book."""
    commission_bps: float = 1.0       # per trade, bps of notional traded
    impact_coeff: float = 0.0         # sqrt-impact coefficient (bps at 100% adv) - linearized
    funding_spread: float = 0.005     # annualized borrow/financing spread
    margin_funding_spread: float = 0.002  # drag on posted margin
    hedge_eps: float = 0.0005         # proportional cost of trading hedge instrument
    hedge_rho: float = 1.0            # quadratic running risk penalty rate (hedge layer)
    kappa_H: float = (3.0 / 4.0) ** (2.0 / 3.0)  # band-law constant
    quality_penalty: float = 0.02     # annualized admission penalty per unit (1-quality)
    admission_cost: float = 0.0       # fixed annualized cost per active instrument

    def validate(self) -> None:
        if min(self.commission_bps, self.funding_spread, self.hedge_eps,
               self.hedge_rho, self.quality_penalty) < 0:
            raise ValueError("cost parameters must be nonnegative")


@dataclass
class RegimeAmbiguity:
    """Regime-coupled ambiguity field  kappa(p) = kappa0 + kappa1 * p.

    ``p`` is the filtered stress posterior in [0,1].  ``iota0/iota1``
    inflate volatility the same way:  iota(p) = iota0 + iota1 * p.
    ``n_eff`` is the effective sample size used for the default
    estimation-error metric U = Sigma / n_eff.
    """
    kappa0: float = 0.5
    kappa1: float = 0.5
    p_stress: float = 0.0
    iota0: float = 1.0
    iota1: float = 0.0
    n_eff: float = 120.0

    def validate(self) -> None:
        if self.kappa0 < 0 or self.kappa1 < 0:
            raise ValueError("kappa0, kappa1 must be >= 0")
        if not (0.0 <= self.p_stress <= 1.0):
            raise ValueError("p_stress must be in [0,1]")
        if self.iota0 < 1.0 or self.iota1 < 0 or self.n_eff <= 0:
            raise ValueError("iota0 >= 1, iota1 >= 0, n_eff > 0 required")

    @property
    def kappa(self) -> float:
        return self.kappa0 + self.kappa1 * self.p_stress

    @property
    def iota(self) -> float:
        return self.iota0 + self.iota1 * self.p_stress


@dataclass
class ConstraintSet:
    """Convex feasible set."""
    gross_leverage: float = 2.0           # sum |x_i| over all non-cash instruments
    class_gross: Dict[str, float] = field(default_factory=dict)  # per-asset-class gross caps
    max_single: float = 0.5               # per-instrument |x_i| cap (also capped by capacity)
    margin_cap: float = 0.5               # sum margin_rate_i |x_i| <= margin_cap
    net_delta: Tuple[float, float] = (-2.0, 2.0)
    gamma_band: Optional[Tuple[float, float]] = None   # dollar gamma per wealth
    vega_band: Optional[Tuple[float, float]] = None    # per unit vol per wealth
    turnover_cap: Optional[float] = None  # sum |x - x_prev|
    long_only_linear: bool = False
    cvar_cap: Optional[float] = None      # cap on CVaR_alpha of horizon loss
    cvar_alpha: float = 0.95
    stress_floor: Optional[float] = None  # worst scenario return floor (negative number)

    def validate(self) -> None:
        if self.gross_leverage <= 0 or self.max_single <= 0 or self.margin_cap <= 0:
            raise ValueError("leverage/box/margin caps must be positive")
        if self.net_delta[0] > self.net_delta[1]:
            raise ValueError("net_delta band inverted")


@dataclass
class DeliveryPolicy:
    """No-physical-delivery policy: every physically settled contract must
    be rolled or closed at least ``buffer_days`` before first notice."""
    buffer_days: float = 5.0
    horizon_days: float = 21.0

    def admissible(self, spec: InstrumentSpec) -> Tuple[bool, str]:
        if spec.asset_class not in (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE):
            return True, ""
        if spec.settlement == Settlement.CASH:
            return True, ""
        if spec.first_notice_days is None:
            return False, "physical settlement with unknown first-notice date"
        if spec.first_notice_days <= self.buffer_days:
            return False, (f"first notice in {spec.first_notice_days:.0f}d "
                           f"<= buffer {self.buffer_days:.0f}d")
        if spec.first_notice_days <= self.horizon_days + self.buffer_days \
                and spec.rolls_per_year <= 0:
            return False, "delivery window inside horizon and no roll schedule"
        return True, ""


class Strategy(str, Enum):
    GENERAL = "general"
    STOCK_ONLY = "stock_only"
    NO_OPTIONS = "no_options"
    FUTURES_ONLY = "futures_only"
    FUTURES_TREND = "futures_trend"
    COMMODITY_SLEEVE = "commodity_sleeve"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"
    SHORT_VOL = "short_vol"
    LONG_VOL = "long_vol"
    STRADDLE = "straddle"
    TAIL_HEDGE = "tail_hedge"
    DISPERSION = "dispersion"
    CROSS_ASSET_HEDGE = "cross_asset_hedge"
    DELTA_NEUTRAL = "delta_neutral"


@dataclass
class StrategyRestriction:
    """Linear restriction of the general feasible set: sign constraints,
    class exclusions, coverage inequalities, and Greek bands."""
    name: str = "general"
    excluded_classes: Tuple[AssetClass, ...] = ()
    sign: Dict[str, int] = field(default_factory=dict)        # id -> +1 (long only) / -1
    coverage: List[Tuple[str, str]] = field(default_factory=list)  # (option_id, underlying_equity_id): |x_opt| <= x_eq
    budget_cap: Optional[float] = None                        # cap on total option premium outlay
    extra_gamma_band: Optional[Tuple[float, float]] = None
    extra_vega_band: Optional[Tuple[float, float]] = None
    extra_net_delta: Optional[Tuple[float, float]] = None


@dataclass
class HedgePolicy:
    """No-trade band policy per underlying (cube-root band law)."""
    bands: Dict[str, float]
    cost_rates: Dict[str, float]
    dollar_gamma: Dict[str, float]

    def hedge_trade(self, underlying: str, net_dollar_delta: float) -> float:
        b = self.bands.get(underlying, 0.0)
        if abs(net_dollar_delta) <= b:
            return 0.0
        return -(net_dollar_delta - math.copysign(b, net_dollar_delta))


@dataclass
class PortfolioResult:
    """Solve output: positions, costs, exposures, diagnostics."""
    x: pd.Series
    solver: str
    status: str
    expected_return_gross: float
    haircut: float
    entry_cost: float
    hedging_cost: float
    roll_cost: float
    funding_cost: float
    quality_cost: float
    risk: float
    risk_measure: str
    objective: float
    greeks: Dict[str, float]
    margin_used: float
    gross: float
    net: float
    cash: float
    kappa_used: float
    warnings_list: List[str] = field(default_factory=list)
    gauge: Optional[pd.DataFrame] = None

    @property
    def expected_return_net(self) -> float:
        return (self.expected_return_gross - self.entry_cost - self.hedging_cost
                - self.roll_cost - self.funding_cost - self.quality_cost)

    def summary(self) -> str:
        act = self.x[self.x.abs() > 1e-6]
        lines = [
            f"solver={self.solver} status={self.status} kappa={self.kappa_used:.3f}",
            f"E[R] gross={self.expected_return_gross:+.4f} net={self.expected_return_net:+.4f} "
            f"haircut={self.haircut:.4f} risk[{self.risk_measure}]={self.risk:.4f}",
            f"costs: entry={self.entry_cost:.4f} hedge={self.hedging_cost:.4f} "
            f"roll={self.roll_cost:.4f} funding={self.funding_cost:.4f} quality={self.quality_cost:.4f}",
            f"gross={self.gross:.3f} net={self.net:+.3f} cash={self.cash:+.3f} "
            f"margin={self.margin_used:.3f}",
            "greeks: " + " ".join(f"{k}={v:+.4f}" for k, v in self.greeks.items()),
            f"active positions ({len(act)}):",
        ]
        lines += [f"  {k:24s} {v:+.4f}" for k, v in act.sort_values(key=abs, ascending=False).items()]
        lines += [f"  WARN: {w}" for w in self.warnings_list]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Performance / statistical-validation utilities (module level)
# --------------------------------------------------------------------------

def perf_stats(returns: pd.Series, benchmark: Optional[pd.Series] = None,
               rf: Union[float, pd.Series] = 0.0, periods: int = 12) -> Dict[str, float]:
    """Annualized performance statistics for a periodic return series."""
    r = returns.dropna().astype(float)
    if len(r) < 3:
        return {}
    rf_ser = (pd.Series(rf, index=r.index) if np.isscalar(rf) else rf.reindex(r.index).fillna(0.0))
    ex = r - rf_ser
    mu, sd = ex.mean() * periods, r.std(ddof=1) * math.sqrt(periods)
    downside = r[r < rf_ser.reindex(r.index)] - rf_ser[r < rf_ser.reindex(r.index)]
    dvol = math.sqrt((downside ** 2).sum() / len(r)) * math.sqrt(periods) if len(downside) else np.nan
    curve = (1 + r).cumprod()
    dd = (curve / curve.cummax() - 1.0)
    out = {
        "ann_return": float((1 + r.mean()) ** periods - 1),
        "ann_vol": float(sd),
        "sharpe": float(mu / sd) if sd > 0 else np.nan,
        "sortino": float(mu / dvol) if dvol and dvol > 0 else np.nan,
        "max_drawdown": float(dd.min()),
        "calmar": float(((1 + r.mean()) ** periods - 1) / abs(dd.min())) if dd.min() < 0 else np.nan,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "cvar95": float(-r[r <= r.quantile(0.05)].mean()) if (r <= r.quantile(0.05)).any() else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n_periods": int(len(r)),
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        rr = r.reindex(b.index)
        a = rr - b
        te = a.std(ddof=1) * math.sqrt(periods)
        cov = np.cov(rr - rf_ser.reindex(b.index), b - rf_ser.reindex(b.index))
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        out.update({
            "information_ratio": float(a.mean() * periods / te) if te > 0 else np.nan,
            "tracking_error": float(te),
            "beta": float(beta),
            "alpha_ann": float((rr - rf_ser.reindex(b.index)).mean() * periods
                               - beta * (b - rf_ser.reindex(b.index)).mean() * periods),
        })
    return out


def block_bootstrap_ci(returns: pd.Series, stat_fn: Callable[[pd.Series], float],
                       n_boot: int = 2000, block: int = 6, alpha: float = 0.10,
                       seed: int = 7) -> Tuple[float, float, float]:
    """Stationary (circular block) bootstrap CI for a return statistic."""
    r = returns.dropna().to_numpy()
    T = len(r)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(((s + np.arange(block)) % T).tolist())
        sample = pd.Series(r[np.array(idx[:T])])
        vals[b] = stat_fn(sample)
    vals = vals[np.isfinite(vals)]
    point = stat_fn(pd.Series(r))
    return (float(point), float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def probabilistic_sharpe_ratio(sr_hat: float, sr_star: float, T: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR of Bailey & Lopez de Prado (2012): P(true SR > sr_star)."""
    if T <= 1:
        return float("nan")
    denom = math.sqrt(max(1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2, 1e-12) / (T - 1))
    return float(stats.norm.cdf((sr_hat - sr_star) / denom))


def deflated_sharpe_ratio(sr_hat: float, sr_trials: Sequence[float], T: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR: PSR against the expected max SR over the trials actually run."""
    trials = np.asarray([s for s in sr_trials if np.isfinite(s)])
    N = max(len(trials), 2)
    var_tr = float(np.var(trials)) if len(trials) > 1 else 0.0
    emc = 0.5772156649015329
    z1, z2 = stats.norm.ppf(1 - 1.0 / N), stats.norm.ppf(1 - 1.0 / (N * math.e))
    sr0 = math.sqrt(max(var_tr, 1e-12)) * ((1 - emc) * z1 + emc * z2)
    return probabilistic_sharpe_ratio(sr_hat, sr0, T, skew, kurt)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

class MultiAssetDerivativePortfolioModel:
    """General multi-asset derivative-aware optimizer.

    Parameters
    ----------
    instruments : list of InstrumentSpec
        Candidate universe (cash is implicit; the residual budget earns rf).
    cov : pd.DataFrame
        Annualized covariance of *driver* returns.  Drivers are the union of
        linear instrument ids and option underlyings.
    cost_model, ambiguity, constraints, delivery : parameter blocks.
    risk_measure : 'variance' | 'cvar' | 'semivariance'
    risk_aversion : lambda multiplying the risk term.
    horizon_days : allocation review horizon (annualization of entry costs).
    x_prev : previous positions for turnover/entry-cost accounting.
    iv_vol : annualized vol of implied vol (vega residual risk), default 1.0
        in vol points terms scaled by 0.15 internally if not given per spec.
    """

    TRADING_DAYS = 252.0

    def __init__(self,
                 instruments: Sequence[InstrumentSpec],
                 cov: pd.DataFrame,
                 cost_model: Optional[CostModel] = None,
                 ambiguity: Optional[RegimeAmbiguity] = None,
                 constraints: Optional[ConstraintSet] = None,
                 delivery: Optional[DeliveryPolicy] = None,
                 restriction: Optional[StrategyRestriction] = None,
                 risk_measure: str = "variance",
                 risk_aversion: float = 4.0,
                 horizon_days: float = 21.0,
                 x_prev: Optional[pd.Series] = None,
                 rf: float = 0.04,
                 iv_vol: float = 0.15,
                 n_scenarios: int = 2000,
                 t_dof: float = 5.0,
                 seed: int = 0) -> None:
        self.instruments = list(instruments)
        self.cov = cov.copy()
        self.cost_model = cost_model or CostModel()
        self.ambiguity = ambiguity or RegimeAmbiguity()
        self.constraints = constraints or ConstraintSet()
        self.delivery = delivery or DeliveryPolicy(horizon_days=horizon_days)
        self.restriction = restriction or StrategyRestriction()
        self.risk_measure = risk_measure
        self.risk_aversion = float(risk_aversion)
        self.horizon_days = float(horizon_days)
        self.rf = float(rf)
        self.iv_vol = float(iv_vol)
        self.n_scenarios = int(n_scenarios)
        self.t_dof = float(t_dof)
        self.seed = int(seed)
        self._warnings: List[str] = []
        self._validate()
        self._build_arrays()
        self.x_prev = (x_prev.reindex(self.ids).fillna(0.0).to_numpy()
                       if x_prev is not None else np.zeros(self.n))

    # ------------------------------------------------------------- setup

    def _validate(self) -> None:
        if not self.instruments:
            raise ValueError("instrument list is empty")
        ids = [s.instrument_id for s in self.instruments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate instrument ids")
        self.cost_model.validate()
        self.ambiguity.validate()
        self.constraints.validate()
        for s in self.instruments:
            s.validate()
        if self.risk_measure not in ("variance", "cvar", "semivariance"):
            raise ValueError(f"unknown risk measure {self.risk_measure!r}")
        # covariance must be square, symmetric PSD over drivers
        c = self.cov.to_numpy()
        if c.shape[0] != c.shape[1] or not np.allclose(c, c.T, atol=1e-10):
            raise ValueError("cov must be square symmetric")
        ev = np.linalg.eigvalsh(c)
        if ev.min() < -1e-8:
            raise ValueError("cov is not PSD")
        needed = {s.underlying or s.instrument_id for s in self.instruments
                  if s.asset_class != AssetClass.CASH}
        missing = needed - set(self.cov.index)
        if missing:
            raise ValueError(f"cov missing drivers: {sorted(missing)}")

    def _build_arrays(self) -> None:
        cm, amb = self.cost_model, self.ambiguity
        # delivery / admissibility hard filter
        keep, dropped = [], []
        for s in self.instruments:
            ok, why = self.delivery.admissible(s)
            if ok:
                keep.append(s)
            else:
                dropped.append((s.instrument_id, why))
        for iid, why in dropped:
            self._warnings.append(f"excluded {iid}: {why}")
        if not keep:
            raise ValueError("no instruments survive the delivery/admissibility filter")
        self.active_specs = keep
        self.ids = [s.instrument_id for s in keep]
        self.n = len(keep)
        self.drivers = list(self.cov.index)
        ndrv = len(self.drivers)
        didx = {d: k for k, d in enumerate(self.drivers)}

        h = self.horizon_days / self.TRADING_DAYS
        self.h_years = h
        A = np.zeros((self.n, ndrv))      # delta map instrument -> driver
        self.basis = np.ones(self.n)      # cash outlay per unit notional weight
        self.delta_n = np.zeros(self.n)
        self.gamma_n = np.zeros(self.n)   # dollar gamma per wealth per unit x
        self.vega_n = np.zeros(self.n)
        self.theta_n = np.zeros(self.n)
        self.lin_cost = np.zeros(self.n)  # annualized degree-1 cost rate t_i (ex haircut)
        self.roll_rate = np.zeros(self.n)
        self.qual_rate = np.zeros(self.n)
        self.margin = np.zeros(self.n)
        self.cap = np.zeros(self.n)
        self.mu = np.zeros(self.n)
        self.is_option = np.zeros(self.n, dtype=bool)
        self.is_future = np.zeros(self.n, dtype=bool)
        self.is_linear = np.zeros(self.n, dtype=bool)
        self.underlier_idx = np.full(self.n, -1, dtype=int)
        spread_rate = np.zeros(self.n)

        for i, s in enumerate(keep):
            drv = s.underlying or s.instrument_id
            self.margin[i] = s.margin_rate
            self.cap[i] = min(s.capacity, self.constraints.max_single)
            self.mu[i] = s.mu
            self.qual_rate[i] = cm.quality_penalty * (1.0 - s.data_quality)
            if s.asset_class == AssetClass.CASH:
                continue
            if s.asset_class in OPTION_CLASSES:
                self.is_option[i] = True
                S0 = self._spot_of(drv)
                g = bs_greeks(S0, s.strike, s.maturity_years, self.rf, s.iv,
                              s.kind, s.dividend_yield)
                self.delta_n[i] = g["delta"]
                self.gamma_n[i] = g["gamma"] * S0
                self.vega_n[i] = g["vega"] / S0
                self.theta_n[i] = g["theta"] / S0
                self.basis[i] = s.price / S0
                A[i, didx[drv]] = g["delta"]
                self.underlier_idx[i] = didx[drv]
                spread_rate[i] = s.half_spread * self.basis[i]
            else:
                self.is_linear[i] = True
                self.delta_n[i] = 1.0
                A[i, didx[drv]] = 1.0
                self.underlier_idx[i] = didx[drv]
                spread_rate[i] = s.half_spread
                if s.asset_class in (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE):
                    self.is_future[i] = True
                    self.basis[i] = 0.0      # futures consume no cash, only margin
                    self.roll_rate[i] = s.rolls_per_year * 2.0 * s.roll_half_spread
            # annualized proportional trading cost (charged per review cycle)
            self.lin_cost[i] = (spread_rate[i] + cm.commission_bps * 1e-4) / h \
                + self.roll_rate[i] + self.qual_rate[i]

        self.A = A
        sig = self.cov.to_numpy() * (amb.iota ** 2)
        self.Sigma_drv = sig
        vega_resid = (self.vega_n * self.iv_vol) ** 2
        self.Sigma_inst = A @ sig @ A.T + np.diag(vega_resid)
        # default estimation-error metric U = Sigma_inst / n_eff
        self.U = self.Sigma_inst / amb.n_eff
        # per-underlying hedge-cost parameters
        self.sigma_drv = np.sqrt(np.diag(sig))

    def _spot_of(self, driver: str) -> float:
        for s in self.instruments:
            if s.instrument_id == driver and s.asset_class in LINEAR_CLASSES | {AssetClass.CASH}:
                return s.price
        return 100.0  # driver not traded directly; only relative greeks matter

    # ----------------------------------------------------- cost functionals

    def entry_cost(self, x: np.ndarray) -> float:
        cm = self.cost_model
        dx = np.abs(x - self.x_prev)
        spread = float(np.sum((self.lin_cost - self.roll_rate - self.qual_rate)
                              * self.h_years * dx)) / self.h_years
        impact = cm.impact_coeff * float(np.sum(dx ** 2))
        return spread + impact

    def hedging_cost(self, x: np.ndarray) -> float:
        """Endogenous gamma-hedging cost, kappa_H eps^{2/3} rho^{1/3}
        (sigma_u |Gamma_u|)^{4/3}, summed per underlying."""
        cm = self.cost_model
        g = self._gamma_by_driver(x)
        rate = cm.kappa_H * cm.hedge_eps ** (2.0 / 3.0) * cm.hedge_rho ** (1.0 / 3.0)
        return float(rate * np.sum((self.sigma_drv * np.abs(g)) ** (4.0 / 3.0)))

    def roll_cost(self, x: np.ndarray) -> float:
        return float(np.sum(self.roll_rate * np.abs(x)))

    def quality_cost(self, x: np.ndarray) -> float:
        return float(np.sum(self.qual_rate * np.abs(x)))

    def funding_cost(self, x: np.ndarray) -> float:
        cm = self.cost_model
        cash_used = float(np.sum(self.basis * np.where(self.is_option, np.abs(x), x)
                                 * (~self.is_future)))
        borrow = max(0.0, cash_used - 1.0)
        margin_drag = cm.margin_funding_spread * float(np.sum(self.margin * np.abs(x)))
        return cm.funding_spread * borrow + margin_drag

    def _gamma_by_driver(self, x: np.ndarray) -> np.ndarray:
        g = np.zeros(len(self.drivers))
        for i in np.where(self.is_option)[0]:
            g[self.underlier_idx[i]] += x[i] * self.gamma_n[i]
        return g

    def haircut(self, x: np.ndarray) -> float:
        v = x @ self.U @ x
        return self.ambiguity.kappa * math.sqrt(max(v, 0.0))

    def portfolio_greeks(self, x: np.ndarray) -> Dict[str, float]:
        return {
            "delta": float(np.sum(self.delta_n * x)),
            "gamma$": float(np.sum(self.gamma_n * x)),
            "vega": float(np.sum(self.vega_n * x)),
            "theta": float(np.sum(self.theta_n * x)),
        }

    # ----------------------------------------------------------- scenarios

    def generate_scenarios(self, n: Optional[int] = None, seed: Optional[int] = None
                           ) -> np.ndarray:
        """Scenario matrix of horizon instrument returns per unit notional
        weight (n_scen x n).  Drivers: multivariate Student-t; options:
        full BSM repricing with correlated IV shocks (leverage -0.7)."""
        n = n or self.n_scenarios
        rng = np.random.default_rng(self.seed if seed is None else seed)
        h = self.h_years
        ndrv = len(self.drivers)
        L = np.linalg.cholesky(self.Sigma_drv * h + 1e-12 * np.eye(ndrv))
        z = rng.standard_normal((n, ndrv))
        chi = rng.chisquare(self.t_dof, n) / self.t_dof
        t_scale = math.sqrt((self.t_dof - 2) / self.t_dof)
        r_drv = (z @ L.T) / np.sqrt(chi)[:, None] * t_scale
        iv_z = -0.7 * z[:, :] + math.sqrt(1 - 0.49) * rng.standard_normal((n, ndrv))
        iv_shock = iv_z * self.iv_vol * math.sqrt(h)

        R = np.zeros((n, self.n))
        for i, s in enumerate(self.active_specs):
            if s.asset_class == AssetClass.CASH:
                continue
            k = self.underlier_idx[i]
            if self.is_linear[i]:
                R[:, i] = r_drv[:, k] + (s.carry - 0.0) * h * self.is_future[i]
                continue
            S0 = self._spot_of(s.underlying)
            T_left = max(s.maturity_years - h, 1e-4)
            pi0 = s.price
            for j in range(n):
                S1 = S0 * math.exp(r_drv[j, k] - 0.5 * (self.sigma_drv[k] ** 2) * h)
                iv1 = max(s.iv + iv_shock[j, k], 0.02)
                pi1 = bs_price(S1, s.strike, T_left, self.rf, iv1, s.kind, s.dividend_yield)
                R[j, i] = (pi1 - pi0) / S0
        return R

    # ----------------------------------------- admission gauge / selection

    def admission_gauge(self, x: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Universe Selection Theorem in code.

        For each instrument compute the frictionless marginal value g_i at
        the current point x (default x = x_prev mapped to active ids) and
        the no-admission band half-width t_i.  Instrument is *admitted*
        (i.e. a nonzero position is optimal at the margin) iff |g_i| > t_i.
        Gamma-hedging cost contributes 0 to t_i at x_i = 0 (degree-4/3
        homogeneity) -- it caps size but never vetoes admission.
        """
        if x is None:
            x = np.zeros(self.n)
        amb = self.ambiguity
        Ux = self.U @ x
        nUx = math.sqrt(max(x @ Ux, 0.0))
        Sx = self.Sigma_inst @ x
        rows = []
        for i, s in enumerate(self.active_specs):
            if s.asset_class == AssetClass.CASH:
                continue
            grad_risk = self.risk_aversion * Sx[i] if self.risk_measure == "variance" else \
                self.risk_aversion * Sx[i]  # variance gradient as smooth proxy for gauge
            if nUx > 1e-14:
                amb_term = amb.kappa * Ux[i] / nUx
                band_amb = 0.0
            else:
                amb_term = 0.0
                band_amb = amb.kappa * math.sqrt(max(self.U[i, i], 0.0))
            g = self.mu[i] - amb_term - grad_risk
            t = self.lin_cost[i] + band_amb + self.cost_model.admission_cost
            rows.append({
                "instrument": s.instrument_id,
                "asset_class": s.asset_class.value,
                "marginal_value": g,
                "band": t,
                "spread_component": self.lin_cost[i] - self.roll_rate[i] - self.qual_rate[i],
                "roll_component": self.roll_rate[i],
                "quality_component": self.qual_rate[i],
                "ambiguity_component": band_amb if nUx <= 1e-14 else abs(amb_term),
                "hedge_component": 0.0,   # theorem: zero marginal hedge cost at x_i = 0
                "admitted": bool(abs(g) > t),
                "gauge_ratio": abs(g) / t if t > 0 else np.inf,
            })
        return pd.DataFrame(rows).set_index("instrument")

    def select_universe(self, max_instruments: Optional[int] = None,
                        max_rounds: int = 25) -> Tuple["MultiAssetDerivativePortfolioModel", pd.DataFrame]:
        """Active-set universe selection driven by the admission gauge.

        Start from the empty book, repeatedly solve on the admitted set and
        admit the strongest gauge violator among excluded instruments until
        no violation remains (or caps bind).  Returns (restricted model,
        final gauge report)."""
        admitted: List[str] = []
        gauge = self.admission_gauge(np.zeros(self.n))
        order = gauge[gauge["admitted"]].sort_values("gauge_ratio", ascending=False)
        if order.empty:
            return self._submodel([]), gauge
        admitted.append(order.index[0])
        for _ in range(max_rounds):
            sub = self._submodel(admitted)
            res = sub.solve()
            x_full = np.zeros(self.n)
            for iid, v in res.x.items():
                if iid in self.ids:
                    x_full[self.ids.index(iid)] = v
            gauge = self.admission_gauge(x_full)
            cand = gauge[(gauge["admitted"]) & (~gauge.index.isin(admitted))]
            if cand.empty:
                break
            if max_instruments is not None and len(admitted) >= max_instruments:
                break
            admitted.append(cand.sort_values("gauge_ratio", ascending=False).index[0])
        return self._submodel(admitted), gauge

    def _submodel(self, ids: Sequence[str]) -> "MultiAssetDerivativePortfolioModel":
        keep = [s for s in self.active_specs
                if s.instrument_id in ids or s.asset_class == AssetClass.CASH]
        if not keep:
            keep = [InstrumentSpec("CASH", AssetClass.CASH, 1.0)]
        return MultiAssetDerivativePortfolioModel(
            keep, self.cov, self.cost_model, self.ambiguity, self.constraints,
            self.delivery, self.restriction, self.risk_measure, self.risk_aversion,
            self.horizon_days, None, self.rf, self.iv_vol, self.n_scenarios,
            self.t_dof, self.seed)

    # ------------------------------------------------------------ restrict

    def restrict(self, strategy: Union[Strategy, str], **kw) -> "MultiAssetDerivativePortfolioModel":
        """Return a copy of the model restricted to a named strategy
        subspace.  Restrictions only ever *add* convex constraints."""
        strategy = Strategy(strategy)
        r = StrategyRestriction(name=strategy.value)
        ids_by_class = {}
        for s in self.active_specs:
            ids_by_class.setdefault(s.asset_class, []).append(s.instrument_id)
        eq_ids = ids_by_class.get(AssetClass.EQUITY, []) + ids_by_class.get(AssetClass.ETF, [])

        if strategy == Strategy.STOCK_ONLY:
            r.excluded_classes = tuple(OPTION_CLASSES | {AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE})
        elif strategy == Strategy.NO_OPTIONS:
            r.excluded_classes = tuple(OPTION_CLASSES)
        elif strategy in (Strategy.FUTURES_ONLY, Strategy.FUTURES_TREND):
            r.excluded_classes = (AssetClass.EQUITY, AssetClass.ETF,
                                  AssetClass.OPTION, AssetClass.COMMODITY_OPTION)
        elif strategy == Strategy.COMMODITY_SLEEVE:
            r.excluded_classes = (AssetClass.EQUITY, AssetClass.ETF,
                                  AssetClass.OPTION, AssetClass.FUTURE)
        elif strategy == Strategy.COVERED_CALL:
            r.excluded_classes = (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE,
                                  AssetClass.COMMODITY_OPTION)
            for s in self.active_specs:
                if s.asset_class == AssetClass.OPTION:
                    if s.kind == "call":
                        r.sign[s.instrument_id] = -1
                        if s.underlying in eq_ids:
                            r.coverage.append((s.instrument_id, s.underlying))
                    else:
                        r.sign[s.instrument_id] = 0
        elif strategy == Strategy.PROTECTIVE_PUT:
            r.excluded_classes = (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE,
                                  AssetClass.COMMODITY_OPTION)
            for s in self.active_specs:
                if s.asset_class == AssetClass.OPTION:
                    if s.kind == "put":
                        r.sign[s.instrument_id] = +1
                        if s.underlying in eq_ids:
                            r.coverage.append((s.instrument_id, s.underlying))
                    else:
                        r.sign[s.instrument_id] = 0
        elif strategy == Strategy.COLLAR:
            r.excluded_classes = (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE,
                                  AssetClass.COMMODITY_OPTION)
            for s in self.active_specs:
                if s.asset_class == AssetClass.OPTION:
                    r.sign[s.instrument_id] = -1 if s.kind == "call" else +1
                    if s.underlying in eq_ids:
                        r.coverage.append((s.instrument_id, s.underlying))
        elif strategy == Strategy.SHORT_VOL:
            r.extra_gamma_band = (-np.inf, 0.0)
            r.extra_vega_band = (-np.inf, 0.0)
        elif strategy in (Strategy.LONG_VOL, Strategy.STRADDLE):
            r.extra_gamma_band = (0.0, np.inf)
            r.extra_vega_band = (0.0, np.inf)
            if strategy == Strategy.STRADDLE:
                for s in self.active_specs:
                    if s.asset_class in OPTION_CLASSES:
                        r.sign[s.instrument_id] = +1
        elif strategy == Strategy.TAIL_HEDGE:
            mny = kw.get("max_moneyness", 0.97)
            budget = kw.get("budget", 0.02)
            for s in self.active_specs:
                if s.asset_class in OPTION_CLASSES:
                    S0 = self._spot_of(s.underlying)
                    if s.kind == "put" and s.strike <= mny * S0:
                        r.sign[s.instrument_id] = +1
                    else:
                        r.sign[s.instrument_id] = 0
            r.budget_cap = budget
        elif strategy == Strategy.DISPERSION:
            index_ids = set(kw.get("index_underlyings", ["SPY"]))
            for s in self.active_specs:
                if s.asset_class in OPTION_CLASSES:
                    r.sign[s.instrument_id] = -1 if s.underlying in index_ids else +1
            r.extra_vega_band = kw.get("vega_band", (-0.02, 0.02))
        elif strategy == Strategy.CROSS_ASSET_HEDGE:
            r.excluded_classes = tuple(OPTION_CLASSES)
            r.extra_net_delta = kw.get("net_delta", (-0.25, 0.25))
        elif strategy == Strategy.DELTA_NEUTRAL:
            tol = kw.get("tolerance", 0.05)
            r.extra_net_delta = (-tol, tol)
        elif strategy == Strategy.GENERAL:
            pass
        out = self.copy()
        out.restriction = r
        return out

    def copy(self) -> "MultiAssetDerivativePortfolioModel":
        m = MultiAssetDerivativePortfolioModel(
            self.instruments, self.cov, self.cost_model, self.ambiguity,
            self.constraints, self.delivery, self.restriction, self.risk_measure,
            self.risk_aversion, self.horizon_days,
            pd.Series(self.x_prev, index=self.ids), self.rf, self.iv_vol,
            self.n_scenarios, self.t_dof, self.seed)
        return m

    # --------------------------------------------------------------- solve

    def solve(self, verbose: bool = False) -> PortfolioResult:
        R = self.generate_scenarios() if self.risk_measure in ("cvar", "semivariance") \
            or self.constraints.cvar_cap is not None \
            or self.constraints.stress_floor is not None else None
        if _HAS_CVXPY:
            try:
                return self._solve_cvxpy(R, verbose)
            except Exception as e:  # pragma: no cover
                self._warnings.append(f"cvxpy failed ({type(e).__name__}: {e}); falling back")
        try:
            return self._solve_slsqp(R)
        except Exception as e:  # pragma: no cover
            self._warnings.append(f"slsqp failed ({type(e).__name__}: {e}); falling back")
        return self._solve_search(R)

    # constraint assembly shared by solvers ------------------------------

    def _excluded_mask(self) -> np.ndarray:
        r = self.restriction
        m = np.zeros(self.n, dtype=bool)
        for i, s in enumerate(self.active_specs):
            if s.asset_class in r.excluded_classes:
                m[i] = True
            if r.sign.get(s.instrument_id, None) == 0:
                m[i] = True
        return m

    def _solve_cvxpy(self, R: Optional[np.ndarray], verbose: bool) -> PortfolioResult:
        con_set, cm, r = self.constraints, self.cost_model, self.restriction
        x = cp.Variable(self.n)
        excl = self._excluded_mask()
        cons = [x[excl] == 0] if excl.any() else []
        ax = cp.abs(x)
        cons += [ax <= self.cap,
                 cp.sum(ax) <= con_set.gross_leverage,
                 self.margin @ ax <= con_set.margin_cap]
        for cls, capv in con_set.class_gross.items():
            idx = [i for i, s in enumerate(self.active_specs) if s.asset_class.value == cls]
            if idx:
                cons.append(cp.sum(ax[idx]) <= capv)
        nd = self.delta_n @ x
        lo, hi = con_set.net_delta
        if r.extra_net_delta is not None:
            lo, hi = max(lo, r.extra_net_delta[0]), min(hi, r.extra_net_delta[1])
        cons += [nd >= lo, nd <= hi]
        gam = self.gamma_n @ x
        for band, expr in ((con_set.gamma_band, gam), (con_set.vega_band, self.vega_n @ x)):
            if band is not None:
                if np.isfinite(band[0]):
                    cons.append(expr >= band[0])
                if np.isfinite(band[1]):
                    cons.append(expr <= band[1])
        for band, expr in ((r.extra_gamma_band, gam), (r.extra_vega_band, self.vega_n @ x)):
            if band is not None:
                if np.isfinite(band[0]):
                    cons.append(expr >= band[0])
                if np.isfinite(band[1]):
                    cons.append(expr <= band[1])
        for iid, sgn in r.sign.items():
            if iid in self.ids and sgn != 0:
                i = self.ids.index(iid)
                cons.append(x[i] * sgn >= 0)
        for opt_id, eq_id in r.coverage:
            if opt_id in self.ids and eq_id in self.ids:
                io, ie = self.ids.index(opt_id), self.ids.index(eq_id)
                cons.append(cp.abs(x[io]) <= x[ie])
        if r.budget_cap is not None:
            opt_idx = np.where(self.is_option)[0]
            if len(opt_idx):
                cons.append(self.basis[opt_idx] @ cp.abs(x[opt_idx]) <= r.budget_cap)
        if con_set.long_only_linear:
            cons += [x[i] >= 0 for i in np.where(self.is_linear)[0]]
        if con_set.turnover_cap is not None:
            cons.append(cp.sum(cp.abs(x - self.x_prev)) <= con_set.turnover_cap)

        # objective
        ret = self.mu @ x
        haircut = self.ambiguity.kappa * cp.norm(
            np.linalg.cholesky(self.U + 1e-12 * np.eye(self.n)).T @ x, 2)
        spread_vec = self.lin_cost - self.roll_rate - self.qual_rate
        entry = spread_vec @ cp.abs(x - self.x_prev) + cm.impact_coeff * cp.sum_squares(x - self.x_prev)
        rollc = self.roll_rate @ ax
        qualc = self.qual_rate @ ax
        # hedging cost: per-driver power 4/3 on |dollar gamma|
        hedge_terms = []
        rate = cm.kappa_H * cm.hedge_eps ** (2 / 3) * cm.hedge_rho ** (1 / 3)
        for k in range(len(self.drivers)):
            idx = [i for i in np.where(self.is_option)[0] if self.underlier_idx[i] == k]
            if idx:
                g = cp.sum(cp.multiply(self.gamma_n[idx], x[idx]))
                hedge_terms.append(rate * self.sigma_drv[k] ** (4 / 3) * cp.power(cp.abs(g), 4 / 3))
        hedge = cp.sum(hedge_terms) if hedge_terms else cp.Constant(0.0)
        cash_used = self.basis @ cp.abs(x)
        fund = cm.funding_spread * cp.pos(cash_used - 1.0) \
            + cm.margin_funding_spread * (self.margin @ ax)

        if self.risk_measure == "variance":
            Lr = np.linalg.cholesky(self.Sigma_inst + 1e-12 * np.eye(self.n))
            risk = cp.sum_squares(Lr.T @ x)
        else:
            losses = -(R @ x) / self.h_years   # annualized scenario returns
            if self.risk_measure == "cvar":
                a = self.constraints.cvar_alpha
                zvar = cp.Variable()
                risk = zvar + cp.sum(cp.pos(losses - zvar)) / ((1 - a) * R.shape[0])
            else:  # semivariance
                risk = cp.sum_squares(cp.pos(losses)) / R.shape[0]
        if self.constraints.cvar_cap is not None and self.risk_measure != "cvar":
            a = self.constraints.cvar_alpha
            z2 = cp.Variable()
            losses2 = -(R @ x) / self.h_years
            cons.append(z2 + cp.sum(cp.pos(losses2 - z2)) / ((1 - a) * R.shape[0])
                        <= self.constraints.cvar_cap)
        if self.constraints.stress_floor is not None:
            cons.append(cp.min(R @ x) >= self.constraints.stress_floor)

        obj = cp.Maximize(ret - haircut - entry - rollc - qualc - hedge - fund
                          - self.risk_aversion * risk)
        prob = cp.Problem(obj, cons)
        for solver in ("CLARABEL", "ECOS", "SCS"):
            try:
                prob.solve(solver=solver, verbose=verbose)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue
        if x.value is None:
            raise RuntimeError(f"cvxpy returned no solution (status={prob.status})")
        xv = np.asarray(x.value).ravel()
        xv[np.abs(xv) < 1e-9] = 0.0
        return self._package(xv, "cvxpy", prob.status, R)

    def _smooth_objective(self, x: np.ndarray, R: Optional[np.ndarray],
                          delta: float = 1e-6) -> float:
        sm_abs = lambda v: np.sqrt(v * v + delta)
        cm = self.cost_model
        ret = float(self.mu @ x)
        hair = self.ambiguity.kappa * math.sqrt(float(x @ self.U @ x) + delta)
        spread_vec = self.lin_cost - self.roll_rate - self.qual_rate
        entry = float(spread_vec @ sm_abs(x - self.x_prev)) \
            + cm.impact_coeff * float(np.sum((x - self.x_prev) ** 2))
        rollc = float(self.roll_rate @ sm_abs(x))
        qualc = float(self.qual_rate @ sm_abs(x))
        hedge = self.hedging_cost(x)
        cash_used = float(self.basis @ sm_abs(x))
        fund = cm.funding_spread * max(0.0, cash_used - 1.0) \
            + cm.margin_funding_spread * float(self.margin @ sm_abs(x))
        if self.risk_measure == "variance":
            risk = float(x @ self.Sigma_inst @ x)
        else:
            losses = -(R @ x) / self.h_years
            if self.risk_measure == "cvar":
                a = self.constraints.cvar_alpha
                q = np.quantile(losses, a)
                risk = float(q + np.mean(np.maximum(losses - q, 0)) / (1 - a))
            else:
                risk = float(np.mean(np.maximum(losses, 0.0) ** 2))
        return -(ret - hair - entry - rollc - qualc - hedge - fund
                 - self.risk_aversion * risk)

    def _constraint_list(self, R: Optional[np.ndarray]):
        con_set, r = self.constraints, self.restriction
        cons = []
        cons.append({"type": "ineq",
                     "fun": lambda x: con_set.gross_leverage - np.sum(np.abs(x))})
        cons.append({"type": "ineq",
                     "fun": lambda x: con_set.margin_cap - float(self.margin @ np.abs(x))})
        lo, hi = con_set.net_delta
        if r.extra_net_delta is not None:
            lo, hi = max(lo, r.extra_net_delta[0]), min(hi, r.extra_net_delta[1])
        cons.append({"type": "ineq", "fun": lambda x, lo=lo: float(self.delta_n @ x) - lo})
        cons.append({"type": "ineq", "fun": lambda x, hi=hi: hi - float(self.delta_n @ x)})
        for band, vec in ((con_set.gamma_band, self.gamma_n), (con_set.vega_band, self.vega_n),
                          (r.extra_gamma_band, self.gamma_n), (r.extra_vega_band, self.vega_n)):
            if band is not None:
                if np.isfinite(band[0]):
                    cons.append({"type": "ineq",
                                 "fun": lambda x, v=vec, b=band[0]: float(v @ x) - b})
                if np.isfinite(band[1]):
                    cons.append({"type": "ineq",
                                 "fun": lambda x, v=vec, b=band[1]: b - float(v @ x)})
        for opt_id, eq_id in r.coverage:
            if opt_id in self.ids and eq_id in self.ids:
                io, ie = self.ids.index(opt_id), self.ids.index(eq_id)
                cons.append({"type": "ineq",
                             "fun": lambda x, io=io, ie=ie: x[ie] - abs(x[io])})
        if r.budget_cap is not None:
            oi = np.where(self.is_option)[0]
            cons.append({"type": "ineq",
                         "fun": lambda x, oi=oi: r.budget_cap - float(self.basis[oi] @ np.abs(x[oi]))})
        if con_set.turnover_cap is not None:
            cons.append({"type": "ineq",
                         "fun": lambda x: con_set.turnover_cap - float(np.sum(np.abs(x - self.x_prev)))})
        if con_set.cvar_cap is not None and R is not None:
            a = con_set.cvar_alpha

            def cvar_con(x, R=R, a=a):
                losses = -(R @ x) / self.h_years
                q = np.quantile(losses, a)
                return con_set.cvar_cap - float(q + np.mean(np.maximum(losses - q, 0)) / (1 - a))
            cons.append({"type": "ineq", "fun": cvar_con})
        if con_set.stress_floor is not None and R is not None:
            cons.append({"type": "ineq",
                         "fun": lambda x, R=R: float(np.min(R @ x)) - con_set.stress_floor})
        return cons

    def _bounds(self) -> List[Tuple[float, float]]:
        excl = self._excluded_mask()
        bounds = []
        for i, s in enumerate(self.active_specs):
            if excl[i] or s.asset_class == AssetClass.CASH:
                bounds.append((0.0, 0.0))
                continue
            lo, hi = -self.cap[i], self.cap[i]
            sgn = self.restriction.sign.get(s.instrument_id, None)
            if sgn == 1:
                lo = 0.0
            elif sgn == -1:
                hi = 0.0
            if self.constraints.long_only_linear and self.is_linear[i]:
                lo = max(lo, 0.0)
            bounds.append((lo, hi))
        return bounds

    def _solve_slsqp(self, R: Optional[np.ndarray]) -> PortfolioResult:
        bounds = self._bounds()
        x0 = np.clip(self.x_prev, [b[0] for b in bounds], [b[1] for b in bounds])
        res = optimize.minimize(self._smooth_objective, x0, args=(R,), method="SLSQP",
                                bounds=bounds, constraints=self._constraint_list(R),
                                options={"maxiter": 400, "ftol": 1e-10})
        xv = res.x.copy()
        xv[np.abs(xv) < 1e-7] = 0.0
        status = "optimal" if res.success else "feasible_suboptimal"
        return self._package(xv, "slsqp", status, R)

    def _solve_search(self, R: Optional[np.ndarray], n_iter: int = 4000) -> PortfolioResult:
        rng = np.random.default_rng(self.seed + 1)
        bounds = self._bounds()
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        cons = self._constraint_list(R)

        def feasible(x):
            return all(c["fun"](x) >= -1e-9 for c in cons)

        def project(x):
            x = np.clip(x, lo, hi)
            g = np.sum(np.abs(x))
            if g > self.constraints.gross_leverage:
                x *= self.constraints.gross_leverage / g
            return x
        best, best_val = np.zeros(self.n), self._smooth_objective(np.zeros(self.n), R)
        scale = 0.25
        for it in range(n_iter):
            cand = project(best + rng.standard_normal(self.n) * scale)
            if not feasible(cand):
                continue
            v = self._smooth_objective(cand, R)
            if v < best_val:
                best, best_val = cand, v
            if it % 500 == 499:
                scale *= 0.7
        return self._package(best, "search", "heuristic", R)

    def _package(self, x: np.ndarray, solver: str, status: str,
                 R: Optional[np.ndarray]) -> PortfolioResult:
        if self.risk_measure == "variance":
            risk = float(x @ self.Sigma_inst @ x)
        else:
            if R is None:
                R = self.generate_scenarios()
            losses = -(R @ x) / self.h_years
            if self.risk_measure == "cvar":
                a = self.constraints.cvar_alpha
                q = np.quantile(losses, a)
                risk = float(q + np.mean(np.maximum(losses - q, 0)) / (1 - a))
            else:
                risk = float(np.mean(np.maximum(losses, 0.0) ** 2))
        entry, hedge = self.entry_cost(x), self.hedging_cost(x)
        roll, fund, qual = self.roll_cost(x), self.funding_cost(x), self.quality_cost(x)
        hair = self.haircut(x)
        gross_ret = float(self.mu @ x)
        obj = gross_ret - hair - entry - roll - qual - hedge - fund - self.risk_aversion * risk
        cash_used = float(self.basis @ np.abs(x))
        return PortfolioResult(
            x=pd.Series(x, index=self.ids), solver=solver, status=status,
            expected_return_gross=gross_ret, haircut=hair, entry_cost=entry,
            hedging_cost=hedge, roll_cost=roll, funding_cost=fund, quality_cost=qual,
            risk=risk, risk_measure=self.risk_measure, objective=obj,
            greeks=self.portfolio_greeks(x),
            margin_used=float(self.margin @ np.abs(x)),
            gross=float(np.sum(np.abs(x))), net=float(np.sum(x)),
            cash=1.0 - cash_used, kappa_used=self.ambiguity.kappa,
            warnings_list=list(self._warnings))

    # ----------------------------------------------------- dynamic layers

    def hedging_policy(self, x: np.ndarray) -> HedgePolicy:
        """Cube-root no-trade band per underlying for the option book."""
        cm = self.cost_model
        g = self._gamma_by_driver(np.asarray(x))
        bands, rates, gam = {}, {}, {}
        for k, d in enumerate(self.drivers):
            nu = self.sigma_drv[k] * abs(g[k])
            if nu <= 0:
                continue
            bands[d] = (3.0 * cm.hedge_eps * nu ** 2 / (4.0 * cm.hedge_rho)) ** (1.0 / 3.0)
            rates[d] = cm.kappa_H * cm.hedge_eps ** (2 / 3) * cm.hedge_rho ** (1 / 3) * nu ** (4 / 3)
            gam[d] = g[k]
        return HedgePolicy(bands=bands, cost_rates=rates, dollar_gamma=gam)

    def roll_schedule(self, as_of_day: float = 0.0) -> pd.DataFrame:
        """Roll calendar for futures/commodity positions: roll date =
        first notice - buffer.  Guarantees no physical delivery."""
        rows = []
        for s in self.active_specs:
            if s.asset_class not in (AssetClass.FUTURE, AssetClass.COMMODITY_FUTURE):
                continue
            fn = s.first_notice_days
            roll_day = (fn - self.delivery.buffer_days) if fn is not None else \
                (self.TRADING_DAYS / max(s.rolls_per_year, 1e-9))
            rows.append({
                "instrument": s.instrument_id,
                "settlement": s.settlement.value,
                "first_notice_day": fn,
                "roll_day": max(roll_day - as_of_day, 0.0),
                "rolls_per_year": s.rolls_per_year,
                "roll_cost_per_cycle": 2.0 * s.roll_half_spread,
                "delivery_risk": s.settlement == Settlement.PHYSICAL
                                 and (fn is None or roll_day <= 0),
            })
        df = pd.DataFrame(rows)
        if not df.empty and df["delivery_risk"].any():
            bad = df[df["delivery_risk"]]["instrument"].tolist()
            raise RuntimeError(f"delivery-avoidance violated for {bad}")
        return df

    # ------------------------------------------------- evaluation hooks

    def stress_test(self, x: np.ndarray,
                    spot_shocks: Sequence[float] = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1),
                    vol_shocks: Sequence[float] = (-0.05, 0.0, 0.10, 0.25)) -> pd.DataFrame:
        """Joint parallel spot/vol shock grid; full option repricing."""
        rows = []
        for ds in spot_shocks:
            for dv in vol_shocks:
                pnl = 0.0
                for i, s in enumerate(self.active_specs):
                    if x[i] == 0 or s.asset_class == AssetClass.CASH:
                        continue
                    if self.is_linear[i]:
                        pnl += x[i] * ds
                    else:
                        S0 = self._spot_of(s.underlying)
                        iv1 = max(s.iv + dv, 0.02)
                        p1 = bs_price(S0 * (1 + ds), s.strike,
                                      max(s.maturity_years - self.h_years, 1e-4),
                                      self.rf, iv1, s.kind, s.dividend_yield)
                        pnl += x[i] * (p1 - s.price) / S0
                rows.append({"spot_shock": ds, "vol_shock": dv, "return": pnl})
        return pd.DataFrame(rows)

    def frontier(self, risk_aversions: Sequence[float]) -> pd.DataFrame:
        rows = []
        for lam in risk_aversions:
            m = self.copy()
            m.risk_aversion = lam
            res = m.solve()
            rows.append({"risk_aversion": lam, "expected_return_net": res.expected_return_net,
                         "risk": res.risk, "gross": res.gross,
                         "objective": res.objective, "status": res.status})
        return pd.DataFrame(rows)

    @staticmethod
    def walk_forward(returns: pd.DataFrame,
                     build_model: Callable[[pd.DataFrame], Tuple["MultiAssetDerivativePortfolioModel", pd.Index]],
                     min_train: int = 120, step: int = 1
                     ) -> Tuple[pd.Series, pd.DataFrame]:
        """Generic expanding-window walk-forward driver.

        ``build_model(history) -> (model, column_index)`` is refit at each
        step using ONLY past data; the realized next-period return is
        computed from the model's solved weights and the next row of
        ``returns``.  Returns (oos_return_series, weights_history)."""
        dates = returns.index
        oos, wrows = {}, {}
        for t in range(min_train, len(dates) - 1, step):
            hist = returns.iloc[:t]
            model, cols = build_model(hist)
            res = model.solve()
            w = res.x.reindex(cols).fillna(0.0)
            nxt = returns[cols].iloc[t + 1].fillna(0.0)
            oos[dates[t + 1]] = float(w @ nxt)
            wrows[dates[t + 1]] = w
        return pd.Series(oos).sort_index(), pd.DataFrame(wrows).T

    # ----------------------------------------------------------- reporting

    @staticmethod
    def capability_map(data_dir: str) -> pd.DataFrame:
        """Inventory of local cached datasets (no credentials touched)."""
        import os
        rows = []
        if os.path.isdir(data_dir):
            for f in sorted(os.listdir(data_dir)):
                p = os.path.join(data_dir, f)
                if not f.endswith(".csv"):
                    continue
                try:
                    head = pd.read_csv(p, nrows=3)
                    n = sum(1 for _ in open(p)) - 1
                    rows.append({"file": f, "rows": n, "columns": len(head.columns),
                                 "fields": ", ".join(map(str, head.columns[:12]))})
                except Exception as e:
                    rows.append({"file": f, "rows": -1, "columns": -1,
                                 "fields": f"unreadable: {type(e).__name__}"})
        return pd.DataFrame(rows)

    @staticmethod
    def pm_report(name: str, oos_stats: Dict[str, float],
                  bench_stats: Dict[str, float], dsr: float,
                  costs_share: float, capacity_note: str = "") -> str:
        """PM-facing summary with a mechanical recommendation rule."""
        sharpe_edge = oos_stats.get("sharpe", np.nan) - bench_stats.get("sharpe", np.nan)
        sortino_edge = oos_stats.get("sortino", np.nan) - bench_stats.get("sortino", np.nan)
        if dsr >= 0.95 and sharpe_edge > 0 and sortino_edge > 0:
            rec = "PAPER TRADE with small allocation; promote on live confirmation"
        elif dsr >= 0.80 and (sharpe_edge > 0 or sortino_edge > 0):
            rec = "CONTINUE RESEARCH; edge plausible but not statistically decisive"
        elif sharpe_edge <= 0 and sortino_edge <= 0:
            rec = "REJECT in current form; revisit after model revision"
        else:
            rec = "CONTINUE RESEARCH"
        lines = [
            f"=== Investment Committee Summary: {name} ===",
            f"OOS Sharpe {oos_stats.get('sharpe', float('nan')):.2f} vs benchmark "
            f"{bench_stats.get('sharpe', float('nan')):.2f} (edge {sharpe_edge:+.2f})",
            f"OOS Sortino {oos_stats.get('sortino', float('nan')):.2f} vs benchmark "
            f"{bench_stats.get('sortino', float('nan')):.2f} (edge {sortino_edge:+.2f})",
            f"Max drawdown {oos_stats.get('max_drawdown', float('nan')):.1%}; "
            f"CVaR95 {oos_stats.get('cvar95', float('nan')):.1%}/period",
            f"Deflated Sharpe ratio {dsr:.2f}; cost share of gross return {costs_share:.0%}",
            f"Capacity/liquidity: {capacity_note or 'see capability map'}",
            f"RECOMMENDATION: {rec}",
        ]
        return "\n".join(lines)
