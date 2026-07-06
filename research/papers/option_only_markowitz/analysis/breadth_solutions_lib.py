"""Standalone regularized-estimation helpers for the breadth experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import (
    _build_config_panel,
)
from research.papers.option_only_markowitz.analysis.conditional_premia import (
    ConditionalPremiaConfig,
    conditional_expected_returns,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.run_empirics import (
    PRIMARY_UNDERLYINGS,
    ROOT,
    TRAIN_END,
    VIX_FACTOR,
    _augment_spec_with_beta_and_stress,
    factor_panels,
    make_model,
    representative_specs,
)
from src.portfolio.option_only_markowitz_model import (
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
)


_AUGMENTED_SPEC_ATTR = "_breadth_solutions_augmented_spec"
_DEFAULT_PREMIA_KNOBS = (0.60, 0.25, 0.75)


class CapConstrainedMarkowitzModel(OptionOnlyMarkowitzModel):
    """Option Markowitz model with pre-trade liquidity caps on net weights.

    The base optimizer enforces split-leg gross equality.  With tight net caps,
    it can satisfy undeployable gross by holding overlapping long/short legs
    ("burn-as-cash").  Burn consumes the short budget, so hard-mode deployment
    can drop as low as ``gross_nav - 2 * short_nav_abs`` when that budget binds;
    ``deployed_gross = sum(abs(weights))`` is the real book size.  Liquidity
    caps bind net traded exposure ``abs(q_i)``, not the split legs, so burn does
    not consume scarce contract-level capacity.
    """

    def __init__(self, *args, per_contract_caps: pd.Series | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scalar_hi = (
            self.constraints.gross_nav
            if self.constraints.per_contract_abs is None
            else self.constraints.per_contract_abs
        )
        self._scalar_hi = float(scalar_hi)
        self._caps_provided = per_contract_caps is not None
        if per_contract_caps is None:
            caps = pd.Series(self._scalar_hi, index=self.contracts, dtype=float)
        else:
            caps = (
                pd.Series(per_contract_caps, dtype=float)
                .reindex(self.contracts)
                .fillna(self._scalar_hi)
                .clip(upper=self._scalar_hi)
            )
        self._cap_vec = caps.to_numpy(dtype=float)

    def _caps_binding(self) -> bool:
        return bool(self._caps_provided and (self._cap_vec < self._scalar_hi - 1e-12).any())

    def _cvxpy_homogenized_budget_constraints(self, y_pos, y_neg, t) -> list:
        cons = super()._cvxpy_homogenized_budget_constraints(y_pos, y_neg, t)
        if not self._caps_binding():
            return cons
        net = y_pos - y_neg
        cons.append(net <= self._cap_vec * t)
        cons.append(-net <= self._cap_vec * t)
        return cons

    def _split_constraints(self):
        cons = super()._split_constraints()
        if not self._caps_binding():
            return cons
        n = len(self.contracts)
        cons.append({"type": "ineq", "fun": lambda x, n=n: self._cap_vec - (x[:n] - x[n:])})
        cons.append({"type": "ineq", "fun": lambda x, n=n: self._cap_vec + (x[:n] - x[n:])})
        return cons

    def _max_constraint_violation(self, weights: np.ndarray) -> float:
        base = super()._max_constraint_violation(weights)
        if not self._caps_binding():
            return base
        w = np.asarray(weights, dtype=float)
        cap_breach = float(np.max(np.maximum(np.abs(w) - self._cap_vec, 0.0)))
        # With net caps, sum(caps) < gross_nav is not automatically infeasible:
        # overlapping split legs can burn undeployed gross as cash.  Hard-mode
        # CVXPY becomes genuinely infeasible only when net caps plus the burn
        # allowed by the short budget cannot reach gross_nav.
        return max(base, cap_breach)

    def _make_feasible_start(self, candidate: np.ndarray) -> np.ndarray:
        x = super()._make_feasible_start(candidate)
        if not self._caps_binding():
            return x
        return np.clip(x, -self._cap_vec, self._cap_vec)

    def _linear_feasible_split_start(self, costs: np.ndarray | None = None) -> np.ndarray | None:
        lp = super()._linear_feasible_split_start(costs=costs)
        if not self._caps_binding():
            return lp
        if lp is not None:
            lp = np.minimum(lp, np.r_[self._cap_vec, self._cap_vec])
        return lp


@dataclass
class TrainingContext:
    label: str
    universe: list[str]
    reps: pd.DataFrame
    returns: pd.DataFrame
    detail: pd.DataFrame
    spec: pd.DataFrame
    base_model: OptionOnlyMarkowitzModel
    residuals: pd.DataFrame
    train_returns: pd.DataFrame
    train_under: pd.DataFrame
    train_vol: pd.DataFrame


def build_training_context(
    label: str,
    equity_underlyings: Sequence[str],
    poc_names: Sequence[str],
    with_vix: bool,
) -> TrainingContext:
    reps, returns, detail, universe, _has_vix = _build_config_panel(equity_underlyings, poc_names, with_vix)

    spec = representative_specs(reps, returns)
    returns = returns.reindex(columns=spec.index).dropna(how="all")
    base_model, residuals = make_model(spec, returns, reps, universe)

    train_returns = returns.loc[:TRAIN_END, spec.index].dropna(how="all")
    under_ret, vol_shocks = factor_panels(reps, universe)
    train_under = under_ret.loc[train_returns.index].dropna(how="all").fillna(0.0)
    train_vol = vol_shocks.loc[train_returns.index].dropna(how="all").fillna(0.0)

    augmented_spec = _validated_augmented_spec(base_model, spec, train_returns, train_under, train_vol)
    spec.attrs[_AUGMENTED_SPEC_ATTR] = augmented_spec

    ctx = TrainingContext(
        label=label,
        universe=list(universe),
        reps=reps,
        returns=returns,
        detail=detail,
        spec=spec,
        base_model=base_model,
        residuals=residuals,
        train_returns=train_returns,
        train_under=train_under,
        train_vol=train_vol,
    )

    rebuilt = rebuild_model(ctx, EstimatorKnobs())
    if not np.allclose(rebuilt.option_cov, base_model.option_cov, atol=1e-12):
        diff = float(np.max(np.abs(rebuilt.option_cov - base_model.option_cov)))
        raise AssertionError(f"default rebuild option covariance mismatch: max_abs_diff={diff:.6g}")

    return ctx


@dataclass(frozen=True)
class EstimatorKnobs:
    cov_shrinkage: float | str = 0.20
    under_cov_estimator: str = "sample"
    vol_cov_estimator: str = "sample"
    residual_estimator: str = "sample"
    shrinkage_to_zero: float = 0.60
    historical_weight: float = 0.25
    structural_weight: float = 0.75


def rebuild_model(
    ctx: TrainingContext,
    knobs: EstimatorKnobs,
    per_contract_caps: pd.Series | None = None,
    constraints: OptionMarkowitzConstraints | None = None,
) -> OptionOnlyMarkowitzModel:
    if knobs.under_cov_estimator == "sample":
        under_cov = ctx.base_model.shocks.underlying_cov
    elif knobs.under_cov_estimator == "lw":
        under_cov = _align_cov_to_universe(lw_cov(ctx.train_under), ctx.universe)
    elif knobs.under_cov_estimator == "single_factor":
        under_cov = _align_cov_to_universe(single_factor_cov(ctx.train_under), ctx.universe)
    else:
        raise ValueError(
            "under_cov_estimator must be one of {'sample', 'lw', 'single_factor'}, "
            f"got {knobs.under_cov_estimator!r}"
        )

    if knobs.vol_cov_estimator == "sample":
        vol_cov = ctx.base_model.shocks.vol_cov
    elif knobs.vol_cov_estimator == "lw":
        vol_cov = _align_cov_to_universe(lw_cov(ctx.train_vol), ctx.universe)
    else:
        raise ValueError("vol_cov_estimator must be one of {'sample', 'lw'}, got " f"{knobs.vol_cov_estimator!r}")

    premia_knobs = (knobs.shrinkage_to_zero, knobs.historical_weight, knobs.structural_weight)
    if premia_knobs == _DEFAULT_PREMIA_KNOBS:
        mu = ctx.base_model.expected_returns
    else:
        augmented_spec = _context_augmented_spec(ctx)
        mu, _components = conditional_expected_returns(
            augmented_spec,
            ctx.train_returns,
            ctx.train_under.reindex(ctx.train_returns.index).fillna(0.0),
            ctx.train_vol.reindex(ctx.train_returns.index).fillna(0.0),
            ConditionalPremiaConfig(
                horizon_years=21.0 / 252.0,
                shrinkage_to_zero=knobs.shrinkage_to_zero,
                historical_weight=knobs.historical_weight,
                structural_weight=knobs.structural_weight,
            ),
        )
        mu = mu.reindex(ctx.spec.index).fillna(0.0)

    sample_residual_cov = ctx.residuals.cov().fillna(0.0)
    if knobs.residual_estimator == "sample":
        residual_cov = sample_residual_cov
    elif knobs.residual_estimator == "diag":
        residual_cov = pd.DataFrame(
            np.diag(np.diag(sample_residual_cov.to_numpy(dtype=float))),
            index=sample_residual_cov.index,
            columns=sample_residual_cov.columns,
        )
    elif knobs.residual_estimator == "lw":
        residual_cov = lw_cov(ctx.residuals).reindex(
            index=sample_residual_cov.index,
            columns=sample_residual_cov.columns,
        ).fillna(0.0)
    else:
        raise ValueError("residual_estimator must be one of {'sample', 'diag', 'lw'}, got " f"{knobs.residual_estimator!r}")

    covariance_shrinkage = resolve_cov_shrinkage(
        knobs,
        n_contracts=len(ctx.spec.index),
        t_train=len(ctx.train_returns),
    )
    options = getattr(ctx.base_model, "options", None) or OptionOnlySpec(_context_augmented_spec(ctx))
    model_cls = CapConstrainedMarkowitzModel if per_contract_caps is not None else OptionOnlyMarkowitzModel

    return model_cls(
        options,
        FactorShockSpec(underlying_cov=under_cov, vol_cov=vol_cov),
        expected_returns=mu,
        residual_cov=residual_cov,
        constraints=constraints or ctx.base_model.constraints,
        covariance_shrinkage=covariance_shrinkage,
        **({"per_contract_caps": per_contract_caps} if per_contract_caps is not None else {}),
    )


def lw_cov(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    cols = list(frame.columns)
    if clean.empty:
        cov = np.zeros((len(cols), len(cols)), dtype=float)
    else:
        from sklearn.covariance import ledoit_wolf

        cov, _shrinkage = ledoit_wolf(clean.reindex(columns=cols).to_numpy(dtype=float))
    return pd.DataFrame(cov, index=cols, columns=cols)


def single_factor_cov(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    cols = list(frame.columns)
    if clean.empty:
        return pd.DataFrame(np.zeros((len(cols), len(cols)), dtype=float), index=cols, columns=cols)

    clean = clean.reindex(columns=cols).fillna(0.0)
    factor = clean.mean(axis=1)
    var_f = float(factor.var(ddof=1)) if len(factor) > 1 else 0.0
    var_x = clean.var(ddof=1).reindex(cols).fillna(0.0)
    if not np.isfinite(var_f) or var_f <= 0.0:
        cov = np.diag(var_x.to_numpy(dtype=float))
        return pd.DataFrame(cov, index=cols, columns=cols)

    beta = clean.apply(lambda col: col.cov(factor) / var_f).reindex(cols).fillna(0.0)
    beta_values = beta.to_numpy(dtype=float)
    systematic = np.outer(beta_values, beta_values) * var_f
    residual_var = np.maximum(var_x.to_numpy(dtype=float) - beta_values * beta_values * var_f, 0.0)
    cov = systematic + np.diag(residual_var)
    return pd.DataFrame(cov, index=cols, columns=cols)


def apply_diag_floor(cov: pd.DataFrame, floor: float = 1e-6, trigger: float = 1e-10) -> pd.DataFrame:
    out = cov.copy()
    for col in out.columns:
        if col in out.index and out.loc[col, col] <= trigger:
            out.loc[col, col] = floor
    return out


def resolve_cov_shrinkage(knobs: EstimatorKnobs, n_contracts: int, t_train: int) -> float:
    value = knobs.cov_shrinkage
    if isinstance(value, str):
        if value != "n_scaled":
            raise ValueError(f"unknown covariance shrinkage rule {value!r}")
        if n_contracts <= 0:
            raise ValueError("n_contracts must be positive")
        return float(min(0.90, 0.20 + 0.80 * max(0.0, 1.0 - float(t_train) / float(n_contracts))))
    return float(value)


def naive_weights(base_model: OptionOnlyMarkowitzModel) -> dict[str, pd.Series]:
    return {
        "Equal premium": base_model.equal_premium_weights(),
        "Equal risk": base_model.equal_risk_weights(),
    }


def capped_naive_weights(
    weights: pd.Series,
    caps: pd.Series,
    target_gross: float,
) -> pd.Series:
    """Clip a naive book to contract caps and redistribute without optimizing.

    The routine preserves the input signs and relative naive magnitudes as far as
    the cap vector allows.  If caps cannot support ``target_gross``, it deploys
    the full cap budget instead of fabricating cash-like split legs.
    """

    cap = pd.Series(caps, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    raw = pd.Series(weights, dtype=float).reindex(cap.index).fillna(0.0)
    base_abs = raw.abs()
    target = float(min(max(float(target_gross), 0.0), cap.sum()))
    if target <= 1e-14 or cap.empty:
        return pd.Series(0.0, index=cap.index, name=getattr(weights, "name", "weight"))

    eligible = cap.gt(0.0) & base_abs.gt(0.0)
    if not eligible.any():
        eligible = cap.gt(0.0)
        base_abs = pd.Series(1.0, index=cap.index).where(eligible, 0.0)

    alloc = pd.Series(0.0, index=cap.index, dtype=float)
    remaining = eligible.copy()
    remaining_target = target
    for _ in range(len(cap) + 1):
        if remaining_target <= 1e-14 or not remaining.any():
            break
        scores = base_abs.where(remaining, 0.0)
        score_sum = float(scores.sum())
        if score_sum <= 1e-14:
            scores = pd.Series(1.0, index=cap.index).where(remaining, 0.0)
            score_sum = float(scores.sum())
        proposed = scores / score_sum * remaining_target
        cap_room = (cap - alloc).clip(lower=0.0)
        binding = remaining & proposed.ge(cap_room - 1e-14)
        if binding.any():
            add = cap_room.where(binding, 0.0)
            alloc = alloc + add
            remaining_target -= float(add.sum())
            remaining = remaining & ~binding
            continue
        alloc = alloc + proposed.where(remaining, 0.0)
        remaining_target = 0.0
        break

    signs = np.sign(raw).replace(0.0, 1.0)
    out = alloc * signs
    out.name = getattr(weights, "name", "weight")
    return out.astype(float)


def solve_gm(model: OptionOnlyMarkowitzModel, method: str = "cvxpy") -> tuple[pd.Series, str]:
    res = model.solve_max_sharpe(method=method)
    status = str(getattr(res, "status", "optimal"))
    if status == "infeasible":
        return pd.Series(0.0, index=model.contracts, name="weight"), status
    return res.weights, status


def sharpe(series: pd.Series) -> float:
    x = series.dropna()
    std = x.std(ddof=1)
    if len(x) < 2 or not np.isfinite(std) or std <= 0:
        return float("nan")
    return float(np.sqrt(12.0) * x.mean() / std)


def gross_sharpe_for_weights(
    ctx: TrainingContext,
    model: OptionOnlyMarkowitzModel,
    weights: pd.Series,
) -> float:
    test = ctx.returns.loc[ctx.returns.index > TRAIN_END, model.contracts].fillna(0.0)
    return sharpe(model.portfolio_return_series(test, weights))


def compute_liquidity_caps(
    reps: pd.DataFrame,
    spec_mark: pd.Series,
    nav: float,
    participation: float,
    per_contract_abs: float = 0.18,
    option_multiplier: float = 100.0,
    train_end: pd.Timestamp = TRAIN_END,
    volume_stat: str = "median",
) -> pd.DataFrame:
    if volume_stat not in {"median", "mean"}:
        raise ValueError(f"volume_stat must be 'median' or 'mean', got {volume_stat!r}")
    marks = pd.Series(spec_mark, dtype=float)
    frame = reps.copy()
    if "snap_date" in frame:
        snap_date = pd.to_datetime(frame["snap_date"], errors="coerce")
        if "trade_date" in frame:
            snap_date = snap_date.fillna(pd.to_datetime(frame["trade_date"], errors="coerce"))
    elif "trade_date" in frame:
        snap_date = pd.to_datetime(frame["trade_date"], errors="coerce")
    else:
        snap_date = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    frame["_cap_snap_date"] = snap_date

    if "asset_id" not in frame or "volume" not in frame:
        train_volume = pd.Series(dtype=float)
    else:
        train = frame[frame["_cap_snap_date"].le(pd.Timestamp(train_end))].copy()
        volumes = pd.to_numeric(train["volume"], errors="coerce")
        train_volume = getattr(volumes.groupby(train["asset_id"]), volume_stat)()

    train_volume = train_volume.reindex(marks.index).astype(float)
    has_volume = train_volume.notna()
    cap_contracts = pd.Series(
        np.maximum(1.0, participation * train_volume.to_numpy(dtype=float)),
        index=marks.index,
        dtype=float,
    )
    w_cap = cap_contracts * marks * float(option_multiplier) / float(nav)
    bound = w_cap.clip(upper=per_contract_abs).where(has_volume, per_contract_abs)
    return pd.DataFrame(
        {
            "train_volume": train_volume,
            "cap_contracts": cap_contracts,
            "w_cap": w_cap,
            "bound": bound.astype(float),
            "has_volume": has_volume.astype(bool),
        },
        index=marks.index,
    )


def cap_feasibility(caps: pd.DataFrame, constraints: OptionMarkowitzConstraints) -> dict[str, object]:
    bound = pd.to_numeric(caps["bound"], errors="coerce").fillna(0.0)
    sum_of_caps = float(bound.sum())
    per_contract_abs = constraints.gross_nav if constraints.per_contract_abs is None else constraints.per_contract_abs
    return {
        "sum_of_caps": sum_of_caps,
        "gross_feasible": bool(sum_of_caps >= constraints.gross_nav),
        "n_binding": int((bound < float(per_contract_abs) - 1e-12).sum()),
        "min_bound": float(bound.min()) if len(bound) else float("nan"),
        "median_bound": float(bound.median()) if len(bound) else float("nan"),
        "suggested_gross": float(min(constraints.gross_nav, 0.95 * sum_of_caps)),
    }


def spread_source_coverage(
    config: str,
    cost_inputs: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "config",
        "relative_spread_source",
        "asset_class",
        "rows",
        "asset_ids",
        "underlyings",
        "mean_relative_spread",
        "median_relative_spread",
    ]
    if cost_inputs.empty:
        return pd.DataFrame(columns=columns)
    frame = cost_inputs.copy()
    frame["relative_spread_source"] = (
        frame.get("relative_spread_source", pd.Series("default", index=frame.index))
        .fillna("default")
        .astype(str)
    )
    frame["asset_class"] = (
        frame.get("asset_class", pd.Series("unknown", index=frame.index))
        .fillna("unknown")
        .astype(str)
    )
    frame["relative_spread"] = pd.to_numeric(frame.get("relative_spread", np.nan), errors="coerce")
    grouped = (
        frame.groupby(["relative_spread_source", "asset_class"], dropna=False, observed=True)
        .agg(
            rows=("relative_spread", "size"),
            asset_ids=("asset_id", "nunique"),
            underlyings=("underlying", "nunique"),
            mean_relative_spread=("relative_spread", "mean"),
            median_relative_spread=("relative_spread", "median"),
        )
        .reset_index()
    )
    grouped.insert(0, "config", config)
    return grouped[columns]


def delta_neutral_weights(
    ctx: TrainingContext,
    model: OptionOnlyMarkowitzModel,
    caps: pd.Series | None = None,
    method: str = "cvxpy",
) -> pd.Series:
    model_cls = CapConstrainedMarkowitzModel if caps is not None else OptionOnlyMarkowitzModel
    kwargs = {"per_contract_caps": caps} if caps is not None else {}
    delta_model = model_cls(
        model.options,
        model.shocks,
        model.expected_returns,
        residual_cov=model.covariance_frame() * 0.0,
        constraints=OptionMarkowitzConstraints(
            gross_nav=1.0,
            net_nav_abs=1.0,
            short_nav_abs=0.25,
            per_contract_abs=0.18,
            underlying_gross={u: (0.35 if u != VIX_FACTOR else 0.20) for u in ctx.universe},
            delta_abs=0.05,
            beta_spy_abs=0.25,
            vix_vega_abs=5.00,
            stress_loss_abs=0.30,
        ),
        covariance_shrinkage=0.20,
        **kwargs,
    )
    delta_model.option_cov = model.option_cov
    weights, _status = solve_gm(delta_model, method)
    return weights


def evaluate(
    ctx: TrainingContext,
    strategies: dict[str, pd.Series],
    aums: Sequence[float],
    cost_kwargs: dict | None = None,
    cost_inputs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cost_kwargs = cost_kwargs or {}
    test = ctx.returns.loc[ctx.returns.index > TRAIN_END, ctx.base_model.contracts].fillna(0.0)
    gross_frame = pd.DataFrame(index=test.index)
    for name, weights in strategies.items():
        gross_frame[name] = ctx.base_model.portfolio_return_series(test, weights)

    if cost_inputs is None:
        cfg0 = ResearchCostConfig(**cost_kwargs)
        surface = load_cbbo_spread_surface(ROOT, cfg0.cbbo_spread_surface_path) if cfg0.use_cbbo_spread_surface else None
        cost_inputs = build_cost_input_ledger(ctx.reps, ctx.detail, ROOT, cfg0, spread_surface=surface)

    rows: list[dict[str, object]] = []
    for aum in aums:
        cfg = ResearchCostConfig(nav_for_capacity=float(aum), **cost_kwargs)
        net_frame, cost_ledger, cap_ledger, *_ = compute_strategy_cost_ledgers(
            gross_frame,
            strategies,
            cost_inputs,
            cfg,
        )
        for name, weights in strategies.items():
            _ = weights
            gross_series = gross_frame[name].dropna() if name in gross_frame else pd.Series(dtype=float)
            net_series = net_frame[name].dropna() if name in net_frame else pd.Series(dtype=float)
            cl = cost_ledger[cost_ledger["strategy"] == name] if len(cost_ledger) else cost_ledger
            cap = cap_ledger[cap_ledger["strategy"] == name] if len(cap_ledger) else cap_ledger
            cap_ratio = (
                pd.to_numeric(cap["capacity_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                if len(cap) and "capacity_ratio" in cap
                else pd.Series(dtype=float)
            )
            rows.append(
                {
                    "strategy": name,
                    "aum": float(aum),
                    "gross_sharpe": sharpe(gross_series),
                    "net_sharpe": sharpe(net_series),
                    "gross_ann_ret": float(gross_series.mean() * 12.0) if len(gross_series) else float("nan"),
                    "net_ann_ret": float(net_series.mean() * 12.0) if len(net_series) else float("nan"),
                    "mean_monthly_capacity_cost": (
                        float(cl.groupby("return_date")["capacity_cost_nav"].sum().mean())
                        if len(cl) and "capacity_cost_nav" in cl
                        else 0.0
                    ),
                    "max_capacity_ratio": float(cap_ratio.max()) if len(cap_ratio) else float("nan"),
                    "mean_capacity_ratio": float(cap_ratio.mean()) if len(cap_ratio) else float("nan"),
                    "capacity_penalized_share": (
                        float((cap["capacity_status"] == "penalized").mean())
                        if len(cap) and "capacity_status" in cap
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def selftest() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, details: str = "") -> None:
        if ok:
            print(f"MATCH {name}{(': ' + details) if details else ''}", flush=True)
        else:
            failures.append(name)
            print(f"MISS {name}{(': ' + details) if details else ''}", flush=True)

    print("building orig+VIX context", flush=True)
    try:
        ctx = build_training_context("orig+VIX", PRIMARY_UNDERLYINGS, poc_names=(), with_vix=True)
        check("orig+VIX construction guards", True)
    except Exception as exc:
        check("orig+VIX construction guards", False, repr(exc))
        return 1

    try:
        m = rebuild_model(ctx, EstimatorKnobs())
        check(
            "default expected returns",
            np.array_equal(m.expected_returns.to_numpy(), ctx.base_model.expected_returns.to_numpy()),
        )
        check(
            "default option covariance",
            np.allclose(m.option_cov, ctx.base_model.option_cov, atol=1e-12),
        )
        base_weights, base_status = solve_gm(ctx.base_model, method="cvxpy")
        rebuild_weights, rebuild_status = solve_gm(m, method="cvxpy")
        check("orig+VIX base cvxpy status", base_status != "infeasible", base_status)
        check("orig+VIX rebuilt cvxpy status", rebuild_status != "infeasible", rebuild_status)
        check(
            "default cvxpy weights",
            np.allclose(
                base_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                rebuild_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                atol=1e-10,
            ),
        )
        sr_vix = gross_sharpe_for_weights(ctx, ctx.base_model, base_weights)
        check(
            "orig+VIX gross Sharpe anchor",
            abs(sr_vix - 1.3743892124363595) <= 1e-6,
            f"value={sr_vix:.15f}",
        )
        changed = rebuild_model(ctx, EstimatorKnobs(cov_shrinkage=0.50))
        changed_weights, changed_status = solve_gm(changed, method="cvxpy")
        check("non-default cvxpy status", changed_status != "infeasible", changed_status)
        check(
            "non-default knobs change weights",
            not np.allclose(
                base_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                changed_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                atol=1e-10,
            ),
        )
    except Exception as exc:
        check("orig+VIX solve/rebuild checks", False, repr(exc))

    print("building orig context", flush=True)
    try:
        ctx2 = build_training_context("orig", PRIMARY_UNDERLYINGS, poc_names=(), with_vix=False)
        check("orig construction guards", True)
        base_weights2, base_status2 = solve_gm(ctx2.base_model, method="cvxpy")
        check("orig base cvxpy status", base_status2 != "infeasible", base_status2)
        sr_no_vix = gross_sharpe_for_weights(ctx2, ctx2.base_model, base_weights2)
        check(
            "orig gross Sharpe anchor",
            abs(sr_no_vix - 0.8421199757145471) <= 1e-6,
            f"value={sr_no_vix:.15f}",
        )
    except Exception as exc:
        check("orig construction/anchor", False, repr(exc))

    return 1 if failures else 0


def selftest_caps() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, details: str = "") -> None:
        if ok:
            print(f"MATCH {name}{(': ' + details) if details else ''}", flush=True)
        else:
            failures.append(name)
            print(f"MISS {name}{(': ' + details) if details else ''}", flush=True)

    print("building orig context", flush=True)
    try:
        ctx = build_training_context("orig", PRIMARY_UNDERLYINGS, poc_names=(), with_vix=False)
        check("orig construction guards", True)
    except Exception as exc:
        check("orig construction guards", False, repr(exc))
        return 1

    try:
        base_slsqp_weights, base_slsqp_status = solve_gm(ctx.base_model, method="slsqp")
        cap_none_model = CapConstrainedMarkowitzModel(
            ctx.base_model.options,
            ctx.base_model.shocks,
            ctx.base_model.expected_returns,
            residual_cov=ctx.residuals.cov().fillna(0.0),
            constraints=ctx.base_model.constraints,
            covariance_shrinkage=0.20,
            per_contract_caps=None,
        )
        cap_none_weights, cap_none_status = solve_gm(cap_none_model, method="slsqp")
        check("base SLSQP status", base_slsqp_status != "infeasible", base_slsqp_status)
        check("cap-none SLSQP status", cap_none_status != "infeasible", cap_none_status)
        check(
            "cap-none SLSQP weights byte-identical",
            np.array_equal(
                base_slsqp_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
                cap_none_weights.reindex(ctx.base_model.contracts).to_numpy(dtype=float),
            ),
        )
        sr_slsqp = gross_sharpe_for_weights(ctx, ctx.base_model, base_slsqp_weights)
        check(
            "orig SLSQP gross Sharpe anchor",
            abs(sr_slsqp - 0.8421194565895301) <= 1e-6,
            f"value={sr_slsqp:.15f}",
        )
    except Exception as exc:
        check("cap-none SLSQP checks", False, repr(exc))
        base_slsqp_weights = pd.Series(0.0, index=ctx.base_model.contracts, name="weight")

    try:
        caps_frame = compute_liquidity_caps(
            ctx.reps,
            ctx.spec["mark"],
            nav=1_000_000,
            participation=0.10,
        )
        caps = caps_frame["bound"]
        cap_model = CapConstrainedMarkowitzModel(
            ctx.base_model.options,
            ctx.base_model.shocks,
            ctx.base_model.expected_returns,
            residual_cov=ctx.residuals.cov().fillna(0.0),
            constraints=ctx.base_model.constraints,
            covariance_shrinkage=0.20,
            per_contract_caps=caps,
        )
        cap_weights, cap_status = solve_gm(cap_model, method="cvxpy")
        binding = int((caps.reindex(ctx.base_model.contracts) < 0.18 - 1e-12).sum())
        cap_abs = cap_weights.reindex(ctx.base_model.contracts).abs()
        cap_bound = caps.reindex(ctx.base_model.contracts).fillna(0.18)
        check("liquidity caps binding count", binding >= 0, f"n_binding={binding}")
        check("liquidity-capped CVXPY status", cap_status != "infeasible", cap_status)
        check("liquidity-capped weights respect caps", bool((cap_abs <= cap_bound + 1e-8).all()))
    except Exception as exc:
        check("liquidity-capped CVXPY checks", False, repr(exc))

    try:
        eval_frame = evaluate(ctx, {"Greek Markowitz": base_slsqp_weights}, aums=[1_000_000])
        row = eval_frame[eval_frame["strategy"].eq("Greek Markowitz")].iloc[0]
        net_sharpe = float(row["net_sharpe"])
        gross_sharpe = float(row["gross_sharpe"])
        check(
            "evaluate net Sharpe anchor",
            abs(net_sharpe - (-1.4425788318790798)) <= 0.02,
            f"value={net_sharpe:.15f}",
        )
        check(
            "evaluate gross Sharpe anchor",
            abs(gross_sharpe - 0.8421194565895301) <= 0.02,
            f"value={gross_sharpe:.15f}",
        )
    except Exception as exc:
        check("evaluate cost checks", False, repr(exc))

    return 1 if failures else 0


def _align_cov_to_universe(cov: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    aligned = cov.reindex(index=list(universe), columns=list(universe)).fillna(0.0)
    return apply_diag_floor(aligned)


def _context_augmented_spec(ctx: TrainingContext) -> pd.DataFrame:
    augmented = ctx.spec.attrs.get(_AUGMENTED_SPEC_ATTR)
    if augmented is not None:
        return augmented
    return _augment_spec_with_beta_and_stress(ctx.spec, ctx.train_under, ctx.train_returns.index)


def _validated_augmented_spec(
    base_model: OptionOnlyMarkowitzModel,
    spec: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_under: pd.DataFrame,
    train_vol: pd.DataFrame,
) -> pd.DataFrame:
    candidates = [base_model.frame]
    recomputed = _augment_spec_with_beta_and_stress(spec, train_under, train_returns.index)
    if not candidates[0].equals(recomputed):
        candidates.append(recomputed)
    else:
        candidates.append(recomputed)

    last_mu: pd.Series | None = None
    expected = base_model.expected_returns
    for augmented_spec in candidates:
        mu = _conditional_mu(
            augmented_spec,
            train_returns,
            train_under,
            train_vol,
            ConditionalPremiaConfig(horizon_years=21.0 / 252.0),
        )
        last_mu = mu
        if np.array_equal(mu.to_numpy(), expected.to_numpy()):
            return augmented_spec

    assert last_mu is not None
    expected_values = expected.to_numpy(dtype=float)
    actual_values = last_mu.to_numpy(dtype=float)
    diff = np.abs(actual_values - expected_values)
    bad_pos = np.flatnonzero(actual_values != expected_values)
    bad_cols = [str(last_mu.index[i]) for i in bad_pos[:10]]
    raise AssertionError(
        "conditional expected return equality guard failed: "
        f"max_abs_diff={float(np.nanmax(diff)):.12g}; differing_columns={bad_cols}"
    )


def _conditional_mu(
    augmented_spec: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_under: pd.DataFrame,
    train_vol: pd.DataFrame,
    config: ConditionalPremiaConfig,
) -> pd.Series:
    mu, _components = conditional_expected_returns(
        augmented_spec,
        train_returns,
        train_under.reindex(train_returns.index).fillna(0.0),
        train_vol.reindex(train_returns.index).fillna(0.0),
        config,
    )
    return mu.reindex(augmented_spec.index).fillna(0.0)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run anchor checks")
    parser.add_argument("--self-test-caps", action="store_true", help="run liquidity cap anchor checks")
    args = parser.parse_args()
    if args.self_test:
        return selftest()
    if args.self_test_caps:
        return selftest_caps()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CapConstrainedMarkowitzModel",
    "EstimatorKnobs",
    "TrainingContext",
    "apply_diag_floor",
    "build_training_context",
    "cap_feasibility",
    "compute_liquidity_caps",
    "delta_neutral_weights",
    "evaluate",
    "gross_sharpe_for_weights",
    "lw_cov",
    "naive_weights",
    "rebuild_model",
    "resolve_cov_shrinkage",
    "selftest",
    "selftest_caps",
    "sharpe",
    "single_factor_cov",
    "solve_gm",
]
