"""Empirical pipeline for the option-only Markowitz paper.

The pipeline is intentionally offline-first. It consumes the local
OPRA/Databento-derived feature store and writes every table, figure and
machine-readable number used by the paper's LaTeX root.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/option_only_markowitz_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"
ART_DIR = PAPER / "artifacts"

sys.path.insert(0, str(ROOT))

from src.portfolio.option_only_markowitz_model import (  # noqa: E402
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    nearest_psd,
    performance_stats,
)

from research.papers.option_only_markowitz.analysis.conditional_premia import (  # noqa: E402
    ConditionalPremiaConfig,
    conditional_expected_returns,
)
from research.papers.option_only_markowitz.analysis.execution_cost_scenarios import (  # noqa: E402
    ExecutionCostScenarioConfig,
    RepairConfig,
    apply_trade_hurdles,
    build_execution_cost_scenarios,
    capacity_market_impact_diagnostics,
    execution_repair_comparison_table,
    forecast_ablation_tables,
    liquidity_tier_labels,
    post_cost_survival_table,
    repair_diagnostics_table,
)
from research.papers.option_only_markowitz.analysis.cross_validation import (  # noqa: E402
    CVConfig,
    CVResults,
    CV_STRATEGIES,
    EVENT_WINDOWS,
    assemble_cpcv_paths,
    build_folds,
    cpcv_regime_table,
    evaluate_folds,
    probability_of_backtest_overfitting,
    tag_regimes,
)
from research.papers.option_only_markowitz.analysis.inference import (  # noqa: E402
    BootstrapConfig,
    block_bootstrap_metric_ci,
    grouped_metric_inference,
    hac_ols,
    sharpe_reality_check,
    strategy_metric_inference,
)
from research.papers.option_only_markowitz.analysis.monte_carlo_repricing import (  # noqa: E402
    RepriceConfig,
    contract_static_params,
    fit_joint_state_model,
    reprice_assumptions,
    reprice_contract_returns,
    repriced_strategy_paths,
    repriced_summary,
    simulate_state_paths,
    universe_comparison_table,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (  # noqa: E402
    ResearchCostConfig,
    artifact_hash_manifest,
    build_cost_input_ledger,
    compute_strategy_cost_ledgers,
    cost_diagnostics_table,
    derive_entry_cost_series,
    load_cbbo_spread_surface,
    write_environment_lock,
)
from research.papers.option_only_markowitz.analysis.resampled_universes import (  # noqa: E402
    ResampleConfig,
    fixed_weight_universe_distribution,
    refit_universe_distribution,
    resample_assumptions,
    resampled_summary,
    stratified_month_index_paths,
    month_index_paths,
)
from research.papers.option_only_markowitz.analysis.simulation import (  # noqa: E402
    SimulationConfig,
    compact_assumptions,
    compact_simulation_summary,
    run_tail_path_simulations,
)
from research.papers.option_only_markowitz.analysis.vix_option_panel import (  # noqa: E402
    VIX_FACTOR,
    build_vix_option_bucket_panel,
    vix_state_panel,
)
from research.papers.option_only_markowitz.analysis.vix_chain_features import (  # noqa: E402
    build_vix_chain_state_features,
    vol_of_vol_regime_table,
)


PRIMARY_UNDERLYINGS = ["AAPL", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA"]
SPY_UNDERLYING = "SPY"
BUCKETS = ["atm", "put_near", "call_near", "put_wing"]
TRAIN_END = pd.Timestamp("2020-12-31")
VIX_OPTION_UNDERLYINGS = [VIX_FACTOR]
PERIODS_PER_YEAR = 12.0
JOURNAL_COLORS = ["#00552B", "#2F6F9F", "#8B1E3F", "#7A6A2B", "#4C566A", "#A65E2E", "#4F7C45", "#6B4E71"]
MC_ROBUSTNESS_STRATEGIES = (
    "Greek Markowitz + VIX",
    "Equity-option Greek Markowitz",
    "Beta/delta-neutral + VIX",
    "Cost-aware Sortino + VIX",
)
HEADLINE_GROWTH_STRATEGIES = [
    "Equity-option Greek Markowitz",
    "Greek Markowitz + VIX",
    "Beta/delta-neutral + VIX",
    "Delta-matched equities",
    "Underlying Markowitz",
]
SIMULATION_STRATEGIES = (
    "Equity-option Greek Markowitz",
    "Greek Markowitz + VIX",
    "Beta/delta-neutral + VIX",
    "Delta-matched equities",
    "Underlying Markowitz",
    "Equal premium",
    "Equal risk",
    "VIX hedge sleeve",
)
MIN_OPTION_MARK = 0.25
COMMON_SPLIT_RATIOS = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0])


def _ensure_dirs() -> None:
    for path in (TABLE_DIR, FIG_DIR, ART_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_hash_manifest() -> None:
    write_environment_lock(PAPER / "environment_lock.json")
    hash_paths = []
    for directory in (TABLE_DIR, FIG_DIR, ART_DIR):
        hash_paths.extend([p for p in directory.rglob("*") if p.is_file()])
    hash_paths.extend([
        PAPER / "option_only_portfolio_optimization_dhruv_kohli.tex",
        PAPER / "REPRODUCIBILITY.md",
        PAPER / "environment_lock.json",
    ])
    artifact_hash_manifest(hash_paths, PAPER).to_csv(PAPER / "artifact_hash_manifest.csv", index=False)


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.3f", na_rep: str = "") -> None:
    path.write_text(
        df.to_latex(index=False, escape=False, float_format=float_format, na_rep=na_rep),
        encoding="utf-8",
    )


def _escape_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(_latex_escape)
    return out


def _json_safe_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_safe_dict(values: dict) -> dict:
    return {str(k): _json_safe_value(v) for k, v in values.items()}


def _nearest_split_ratio(value: float) -> float | None:
    if not np.isfinite(value) or value <= 1.5:
        return None
    ratio = float(COMMON_SPLIT_RATIOS[np.argmin(np.abs(COMMON_SPLIT_RATIOS - value))])
    if abs(value / ratio - 1.0) <= 0.25:
        return ratio
    return None


def split_adjusted_spot_panel(
    raw_spot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return split-adjusted spots, raw-to-adjusted factors, and detected events."""

    raw = raw_spot.sort_index().astype(float)
    factors = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    events = []
    for col in raw.columns:
        current_factor = 1.0
        previous_raw = np.nan
        for dt, value in raw[col].items():
            if np.isfinite(previous_raw) and previous_raw > 0 and np.isfinite(value) and value > 0:
                down_ratio = previous_raw / value
                up_ratio = value / previous_raw
                split_ratio = _nearest_split_ratio(down_ratio)
                if split_ratio is not None:
                    current_factor *= split_ratio
                    events.append(
                        {
                            "Underlying": col,
                            "Date": pd.Timestamp(dt).date().isoformat(),
                            "Type": "split",
                            "Raw spot ratio": down_ratio,
                            "Applied ratio": split_ratio,
                        }
                    )
                else:
                    reverse_ratio = _nearest_split_ratio(up_ratio)
                    if reverse_ratio is not None:
                        current_factor /= reverse_ratio
                        events.append(
                            {
                                "Underlying": col,
                                "Date": pd.Timestamp(dt).date().isoformat(),
                                "Type": "reverse split or unit repair",
                                "Raw spot ratio": up_ratio,
                                "Applied ratio": reverse_ratio,
                            }
                        )
            factors.loc[dt, col] = current_factor
            if np.isfinite(value) and value > 0:
                previous_raw = value
        factors[col] = factors[col].ffill().fillna(1.0)
    adjusted = raw * factors
    return adjusted, factors, pd.DataFrame(events)


def load_raw_close_panel(underlyings: Sequence[str]) -> pd.DataFrame:
    """Load raw daily closes in the same units as the OPRA strikes."""

    raw = pd.read_csv(ROOT / "data/universe/multi_raw_close.csv", parse_dates=["Date"]).set_index("Date")
    missing = [u for u in underlyings if u not in raw.columns]
    if missing:
        raise ValueError(f"Raw daily close panel missing underlyings: {missing}")
    return raw.reindex(columns=list(underlyings)).sort_index()


def _last_available_close(
    raw_close: pd.DataFrame,
    underlying: str,
    target_date: pd.Timestamp,
    lower_bound: pd.Timestamp,
) -> tuple[pd.Timestamp | None, float | None]:
    series = raw_close[underlying].dropna()
    candidates = series.loc[(series.index <= target_date) & (series.index >= lower_bound)]
    if candidates.empty:
        return None, None
    dt = pd.Timestamp(candidates.index[-1])
    return dt, float(candidates.iloc[-1])


def build_expiry_proxy_return_panel(
    reps: pd.DataFrame,
    raw_close: pd.DataFrame,
    daily_split_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-period option returns from prior-date selections.

    The local OPRA feature store contains month-end snapshots of 15-21 DTE
    options, so the same option usually expires before the next snapshot.  A
    bucket return therefore selects the option at the decision date, pays or
    receives the option mark, holds to the listed expiry, and computes payoff
    from raw daily underlying closes.  If a stock split occurs between the
    decision date and expiry, the terminal spot is converted back into the
    decision-date contract units before payoff is evaluated against the
    decision-date strike.
    """

    dates = sorted(pd.to_datetime(reps["snap_date"].dropna().unique()))
    next_date = dict(zip(dates[:-1], dates[1:]))
    rows = []
    for _, row in reps.iterrows():
        decision_date = pd.Timestamp(row["snap_date"])
        realization_date = next_date.get(decision_date)
        if realization_date is None:
            continue
        underlying = str(row["underlying"])
        expiry = pd.Timestamp(row["expiry"])
        if underlying not in raw_close.columns or underlying not in daily_split_factors.columns:
            continue
        payoff_date, payoff_raw_close = _last_available_close(raw_close, underlying, expiry, decision_date)
        if payoff_date is None or payoff_raw_close is None:
            continue
        if decision_date not in daily_split_factors.index or payoff_date not in daily_split_factors.index:
            continue
        decision_factor = float(daily_split_factors.loc[decision_date, underlying])
        payoff_factor = float(daily_split_factors.loc[payoff_date, underlying])
        if not np.isfinite(decision_factor) or decision_factor <= 0:
            continue
        terminal_spot = float(payoff_raw_close * payoff_factor / decision_factor)
        start_spot = float(row["spot"])
        expiry_days = (expiry - decision_date).days
        strike = float(row["strike"])
        mark = float(row["mark"])
        if str(row["kind"]) == "call":
            payoff = max(terminal_spot - strike, 0.0)
        else:
            payoff = max(strike - terminal_spot, 0.0)
        option_return = payoff / mark - 1.0
        rows.append(
            {
                "return_date": realization_date,
                "decision_date": decision_date,
                "expiry": expiry,
                "payoff_date": payoff_date,
                "asset_id": row["asset_id"],
                "symbol": row["symbol"],
                "underlying": underlying,
                "kind": row["kind"],
                "moneyness_bucket": row["moneyness_bucket"],
                "mark": mark,
                "strike": strike,
                "start_spot": start_spot,
                "payoff_raw_close": payoff_raw_close,
                "decision_split_factor": decision_factor,
                "payoff_split_factor": payoff_factor,
                "split_factor_ratio": payoff_factor / decision_factor,
                "terminal_spot_proxy": terminal_spot,
                "expiry_spot_proxy": terminal_spot,
                "expiry_weight": 1.0,
                "payoff_proxy": payoff,
                "option_return": option_return,
                "delta": float(row["delta"]),
                "gamma": float(row["gamma"]),
                "vega": float(row["vega"]),
                "theta": float(row["theta"]),
                "iv_proxy": float(row["iv_proxy"]),
                "expiry_days": expiry_days,
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(), detail
    returns = (
        detail.pivot(index="return_date", columns="asset_id", values="option_return")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    returns.index.name = "snap_date"
    return returns, detail


def load_bucket_panel(
    underlyings: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw filtered rows, representative bucket rows and bucket returns."""

    universe = list(underlyings or PRIMARY_UNDERLYINGS)
    columns = [
        "symbol",
        "underlying",
        "snap_date",
        "expiry",
        "strike",
        "kind",
        "spot",
        "close",
        "volume",
        "tenor_days",
        "moneyness_bucket",
        "iv_proxy",
        "cbbo_median_relative_spread",
        "delta",
        "gamma",
        "vega",
        "theta",
    ]
    panel = pd.read_parquet(ROOT / "data/feature_store/option_greek_proxy_panel.parquet", columns=columns)
    panel["snap_date"] = pd.to_datetime(panel["snap_date"])
    panel["expiry"] = pd.to_datetime(panel["expiry"])
    panel = panel[
        panel["underlying"].isin(universe)
        & panel["moneyness_bucket"].isin(BUCKETS)
        & panel["close"].ge(MIN_OPTION_MARK)
        & panel["volume"].ge(10)
        & panel["cbbo_median_relative_spread"].le(0.20)
    ].copy()
    for col in ["delta", "gamma", "vega", "theta", "iv_proxy", "spot", "strike"]:
        panel = panel[np.isfinite(pd.to_numeric(panel[col], errors="coerce"))]
    panel["mark"] = panel["close"].astype(float)
    panel["asset_id"] = (
        panel["underlying"].astype(str)
        + "_"
        + panel["kind"].astype(str)
        + "_"
        + panel["moneyness_bucket"].astype(str)
    )
    panel = panel.sort_values(
        ["snap_date", "asset_id", "volume", "cbbo_median_relative_spread"],
        ascending=[True, True, False, True],
    )
    reps = panel.groupby(["snap_date", "asset_id"], as_index=False).head(1).copy()
    reps = reps.sort_values(["asset_id", "snap_date"])
    raw_spot = (
        panel.groupby(["snap_date", "underlying"])["spot"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=universe)
    )
    raw_close = load_raw_close_panel(universe)
    _, daily_split_factors, _ = split_adjusted_spot_panel(raw_close)
    returns, _ = build_expiry_proxy_return_panel(reps, raw_close, daily_split_factors)
    returns = returns.dropna(how="all")
    enough = returns.loc[:TRAIN_END].count() >= 36
    returns = returns.loc[:, enough]
    reps = reps[reps["asset_id"].isin(returns.columns)].copy()
    return panel, reps, returns


def load_spot_returns(underlyings: Sequence[str], dates: pd.Index | None = None) -> pd.DataFrame:
    """Return local spot-return panel for equities or ETFs in the OPRA files."""

    columns = ["underlying", "snap_date", "spot"]
    panel = pd.read_parquet(ROOT / "data/feature_store/option_greek_proxy_panel.parquet", columns=columns)
    panel["snap_date"] = pd.to_datetime(panel["snap_date"])
    spot = (
        panel[panel["underlying"].isin(list(underlyings))]
        .groupby(["snap_date", "underlying"])["spot"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=list(underlyings))
    )
    spot, _, _ = split_adjusted_spot_panel(spot)
    returns = spot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    if dates is not None:
        returns = returns.reindex(dates)
    return returns


def load_extended_factor_returns(dates: pd.Index | None = None) -> pd.DataFrame:
    """Regression/state factor panel: equities plus VX/VIX/VVIX controls."""

    eq = load_spot_returns([SPY_UNDERLYING] + PRIMARY_UNDERLYINGS, dates)
    idx = eq.index if dates is None else pd.DatetimeIndex(pd.to_datetime(dates))
    state = vix_state_panel(idx, ROOT)
    out = eq.copy()
    if not state.empty:
        for col in ["VX_FRONT_return", "dVIX", "dVVIX"]:
            if col in state:
                name = "VX_FRONT" if col == "VX_FRONT_return" else col
                out[name] = state[col].reindex(idx)
    for col in ["VX_FRONT", "dVIX", "dVVIX"]:
        if col not in out:
            out[col] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _split_adjust_selected_spots(spot: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    """Apply stock split repair to equities only; VX forwards are not split-adjusted."""

    out = spot.copy()
    equity_cols = [u for u in universe if u in PRIMARY_UNDERLYINGS and u in out.columns]
    if equity_cols:
        adjusted, _, _ = split_adjusted_spot_panel(out[equity_cols])
        out.loc[:, equity_cols] = adjusted
    return out


def representative_specs(
    reps: pd.DataFrame,
    returns: pd.DataFrame,
    train_start: pd.Timestamp | None = None,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Latest in-training-window representative contract per asset.

    ``train_end`` defaults to the global TRAIN_END for backward compatibility;
    rolling estimations must pass their own window bounds so the specs roll.
    """

    end = TRAIN_END if train_end is None else pd.Timestamp(train_end)
    mask = reps["snap_date"].le(end)
    if train_start is not None:
        mask &= reps["snap_date"].ge(pd.Timestamp(train_start))
    train_reps = reps[mask]
    rows = []
    for asset_id, grp in train_reps.groupby("asset_id"):
        last = grp.sort_values("snap_date").iloc[-1]
        rows.append(
            {
                "asset_id": asset_id,
                "underlying": str(last["underlying"]),
                "mark": float(max(last["mark"], MIN_OPTION_MARK)),
                "spot": float(max(last["spot"], 1.0)),
                "delta": float(last["delta"]),
                "gamma": float(max(last["gamma"], 0.0)),
                "vega": float(max(last["vega"], 1e-8)),
                "theta": float(last["theta"]),
                "kind": str(last["kind"]),
                "moneyness_bucket": str(last["moneyness_bucket"]),
                "asset_class": str(last.get("asset_class", "equity_option")),
                "iv_proxy": float(last.get("iv_proxy", np.nan)),
            }
        )
    spec = pd.DataFrame(rows).set_index("asset_id").reindex(returns.columns)
    return spec.dropna(subset=["underlying", "mark"])


def factor_panels(
    reps: pd.DataFrame,
    underlyings: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = list(underlyings or PRIMARY_UNDERLYINGS)
    spot = (
        reps.groupby(["snap_date", "underlying"])["spot"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=universe)
    )
    spot = _split_adjust_selected_spots(spot, universe)
    underlying_returns = spot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    atm = reps[reps["moneyness_bucket"].isin(["atm", "vix_atm"])]
    iv = (
        atm.groupby(["snap_date", "underlying"])["iv_proxy"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=universe)
    )
    vol_shocks = iv.diff().replace([np.inf, -np.inf], np.nan)
    return underlying_returns, vol_shocks


def _augment_spec_with_beta_and_stress(
    spec: pd.DataFrame,
    under_ret: pd.DataFrame,
    train_index: pd.Index,
) -> pd.DataFrame:
    out = spec.copy()
    factors = load_extended_factor_returns(train_index)
    spy = factors.get(SPY_UNDERLYING, pd.Series(index=train_index, dtype=float)).reindex(train_index)
    betas: dict[str, float] = {}
    for u in out["underlying"].astype(str).unique():
        if u in under_ret.columns and spy.notna().sum() > 3:
            aligned = pd.concat([under_ret[u].rename("u"), spy.rename("spy")], axis=1).dropna()
            var = float(aligned["spy"].var(ddof=1)) if len(aligned) > 3 else np.nan
            betas[u] = float(aligned["u"].cov(aligned["spy"]) / var) if np.isfinite(var) and var > 0 else 0.0
        else:
            betas[u] = 0.0
    mark = out["mark"].astype(float).replace(0.0, np.nan)
    spot = out.get("spot", pd.Series(1.0, index=out.index)).astype(float)
    delta_nav = out["delta"].astype(float) * spot / mark
    gamma_nav = out["gamma"].astype(float) * spot * spot / mark
    vega_nav = out["vega"].astype(float) / mark
    out["underlying_beta_spy"] = out["underlying"].astype(str).map(betas).fillna(0.0)
    out["beta_spy_nav"] = delta_nav * out["underlying_beta_spy"]
    # Coarse but explicit portfolio stress scenarios used as constraints/audit hooks.
    out["stress_scenario_spy_down_10"] = -0.10 * delta_nav + 0.5 * gamma_nav * (0.10**2) + 0.03 * vega_nav
    out["stress_scenario_vol_spike"] = -0.04 * delta_nav + 0.5 * gamma_nav * (0.04**2) + 0.10 * vega_nav
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_model(
    spec: pd.DataFrame,
    returns: pd.DataFrame,
    reps: pd.DataFrame,
    underlyings: Sequence[str] | None = None,
    train_start: pd.Timestamp | None = None,
    train_end: pd.Timestamp | None = None,
    under_ret: pd.DataFrame | None = None,
    vol_shocks: pd.DataFrame | None = None,
) -> tuple[OptionOnlyMarkowitzModel, pd.DataFrame]:
    universe = list(underlyings or PRIMARY_UNDERLYINGS)
    # Explicit training bounds; TRAIN_END remains the default so the headline
    # pipeline is unchanged while rolling estimations get a real roll.
    end = TRAIN_END if train_end is None else pd.Timestamp(train_end)
    start = pd.Timestamp(train_start) if train_start is not None else None
    train_returns = returns.loc[start:end, spec.index].dropna(how="all")
    if len(train_returns):
        assert pd.Timestamp(train_returns.index.max()) <= end, (
            f"training returns leak past train_end={end.date()}: "
            f"max index {pd.Timestamp(train_returns.index.max()).date()}"
        )
    if under_ret is None or vol_shocks is None:
        under_ret, vol_shocks = factor_panels(reps, universe)
    train_under = under_ret.loc[train_returns.index].dropna(how="all").fillna(0.0)
    train_vol = vol_shocks.loc[train_returns.index].dropna(how="all").fillna(0.0)
    under_cov = train_under.cov().reindex(index=universe, columns=universe).fillna(0.0)
    vol_cov = train_vol.cov().reindex(index=universe, columns=universe).fillna(0.0)
    for frame in (under_cov, vol_cov):
        for col in frame.columns:
            if frame.loc[col, col] <= 1e-10:
                frame.loc[col, col] = 1e-6
    spec = _augment_spec_with_beta_and_stress(spec, train_under, train_returns.index)
    cond_mu, premia_components = conditional_expected_returns(
        spec,
        train_returns,
        train_under.reindex(train_returns.index).fillna(0.0),
        train_vol.reindex(train_returns.index).fillna(0.0),
        ConditionalPremiaConfig(horizon_years=21.0 / 252.0),
    )
    spec.attrs["conditional_premia_components"] = premia_components

    tmp_model = OptionOnlyMarkowitzModel(
        OptionOnlySpec(spec),
        FactorShockSpec(underlying_cov=under_cov, vol_cov=vol_cov),
        expected_returns=cond_mu.reindex(spec.index).fillna(0.0),
        constraints=OptionMarkowitzConstraints(gross_nav=1.0, per_contract_abs=0.20),
    )
    factors = pd.DataFrame(index=train_returns.index)
    aligned_under = train_under.reindex(train_returns.index).fillna(0.0)
    aligned_vol = train_vol.reindex(train_returns.index).fillna(0.0)
    for u in tmp_model.underlyings:
        factors[f"r_{u}"] = aligned_under[u]
    for u in tmp_model.underlyings:
        ru = aligned_under[u]
        factors[f"r2_{u}"] = ru * ru - float((ru * ru).mean())
    for u in tmp_model.underlyings:
        factors[f"dv_{u}"] = aligned_vol[u]
    fitted = pd.DataFrame(factors.to_numpy(float) @ tmp_model.B.T, index=factors.index, columns=tmp_model.contracts)
    residuals = train_returns.reindex(index=factors.index, columns=tmp_model.contracts).fillna(0.0) - fitted
    residual_cov = residuals.cov().fillna(0.0)
    constraints = OptionMarkowitzConstraints(
        gross_nav=1.0,
        net_nav_abs=1.0,
        short_nav_abs=0.25,
        per_contract_abs=0.18,
        underlying_gross={u: (0.35 if u != VIX_FACTOR else 0.20) for u in universe},
        beta_spy_abs=3.00,
        vix_vega_abs=8.00,
        stress_loss_abs=0.35,
    )
    model = OptionOnlyMarkowitzModel(
        OptionOnlySpec(spec),
        FactorShockSpec(underlying_cov=under_cov, vol_cov=vol_cov),
        expected_returns=cond_mu.reindex(spec.index).fillna(0.0),
        residual_cov=residual_cov,
        constraints=constraints,
        covariance_shrinkage=0.20,
    )
    model.conditional_premia_components = premia_components
    return model, residuals


def _solve_weights_with_guard(model: OptionOnlyMarkowitzModel, label: str = "") -> pd.Series:
    """Defensive wrapper around ``solve_max_sharpe`` that respects ``status``.

    Contract: status in {'optimal', 'feasible_suboptimal', 'infeasible'};
    'infeasible' means the max constraint violation exceeds 1e-5.  On an
    infeasible solve the strategy holds zero weights (cash) and the event is
    logged.  Works whether or not the solver reports the new status field.
    """

    res = model.solve_max_sharpe()
    status = getattr(res, "status", "optimal")
    if str(status) == "infeasible":
        print(f"[run_empirics] infeasible max-Sharpe solve ({label or 'unnamed'}); using zero weights")
        return pd.Series(0.0, index=model.contracts, name="weight")
    return res.weights


def _sortino_weights_with_guard(
    model: OptionOnlyMarkowitzModel,
    train_scenarios: pd.DataFrame,
    entry_costs: pd.Series,
    label: str = "",
) -> tuple[pd.Series, dict]:
    res = model.solve_max_sortino(train_scenarios, entry_costs=entry_costs, target=0.0)
    status = getattr(res, "status", "optimal")
    if str(status) == "infeasible":
        print(f"[run_empirics] infeasible max-Sortino solve ({label or 'unnamed'}); using zero weights")
        return pd.Series(0.0, index=model.contracts, name="weight"), {"status": "infeasible"}
    return (
        res.weights,
        {
            "status": str(status),
            "method_used": getattr(res, "solver", "") or "",
            **(res.objective_stats or {}),
        },
    )


def strategy_weights(
    model: OptionOnlyMarkowitzModel,
    underlyings: Sequence[str] | None = None,
) -> dict[str, pd.Series]:
    universe = list(underlyings or PRIMARY_UNDERLYINGS)
    opt = _solve_weights_with_guard(model, "greek_markowitz")
    equal = model.equal_premium_weights()
    erisk = model.equal_risk_weights()

    delta_model = OptionOnlyMarkowitzModel(
        model.options,
        model.shocks,
        model.expected_returns,
        residual_cov=model.covariance_frame() * 0.0,
        constraints=OptionMarkowitzConstraints(
            gross_nav=1.0,
            net_nav_abs=1.0,
            short_nav_abs=0.25,
            per_contract_abs=0.18,
            underlying_gross={u: (0.35 if u != VIX_FACTOR else 0.20) for u in universe},
            delta_abs=0.05,
            beta_spy_abs=0.25,
            vix_vega_abs=5.00,
            stress_loss_abs=0.30,
        ),
        covariance_shrinkage=0.20,
    )
    delta_model.option_cov = model.option_cov
    dneutral = _solve_weights_with_guard(delta_model, "delta_neutral")

    return {
        "Greek Markowitz": opt,
        "Equal premium": equal,
        "Equal risk": erisk,
        "Delta neutral": dneutral,
    }


def vix_hedge_sleeve_weights(model: OptionOnlyMarkowitzModel) -> pd.Series:
    w = pd.Series(0.0, index=model.contracts, name="weight")
    if "asset_class" not in model.frame:
        return w
    eligible = model.frame[
        model.frame["asset_class"].astype(str).eq("vix_option")
        & model.frame["kind"].astype(str).eq("call")
    ].index
    if len(eligible) == 0:
        eligible = model.frame[model.frame["asset_class"].astype(str).eq("vix_option")].index
    if len(eligible):
        w.loc[eligible] = 1.0 / len(eligible)
    return w


def random_feasible(model: OptionOnlyMarkowitzModel, returns: pd.DataFrame, n: int = 250) -> pd.Series:
    rng = np.random.default_rng(11)
    sharpes = []
    cols = model.contracts
    test = returns.loc[returns.index > TRAIN_END, cols].fillna(0.0)
    for _ in range(n):
        raw = rng.normal(size=len(cols))
        raw /= np.abs(raw).sum()
        w = pd.Series(raw, index=cols)
        r = model.portfolio_return_series(test, w)
        sharpes.append(performance_stats(r, PERIODS_PER_YEAR)["sharpe"])
    return pd.Series(sharpes, name="random_sharpe")


def option_delta_by_underlying(model: OptionOnlyMarkowitzModel, weights: pd.Series) -> pd.Series:
    """Return first-order equity delta exposure by underlying."""

    weighted = model.greeks["delta_nav"].mul(weights.reindex(model.contracts).fillna(0.0))
    out = weighted.groupby(model.frame["underlying"].astype(str)).sum()
    return out.reindex(PRIMARY_UNDERLYINGS).fillna(0.0)


def equity_tangency_weights(train_underlying_returns: pd.DataFrame) -> pd.Series:
    """Closed-form gross-NAV tangency portfolio over the eight equities."""

    train = train_underlying_returns.reindex(columns=PRIMARY_UNDERLYINGS).dropna(how="all").fillna(0.0)
    cov = nearest_psd(train.cov().to_numpy(float))
    mu = train.mean().to_numpy(float)
    raw = np.linalg.pinv(cov) @ mu
    if not np.isfinite(raw).all() or np.abs(raw).sum() <= 1e-12:
        raw = np.repeat(1.0 / len(PRIMARY_UNDERLYINGS), len(PRIMARY_UNDERLYINGS))
    weights = raw / np.abs(raw).sum()
    return pd.Series(weights, index=PRIMARY_UNDERLYINGS, name="weight")


def _ols(y: pd.Series, x: pd.DataFrame) -> dict[str, object]:
    """OLS fit with HAC/Newey--West alpha standard errors."""

    return hac_ols(y, x)


def factor_regression_table(ret_frame: pd.DataFrame, equity_returns: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [SPY_UNDERLYING] + PRIMARY_UNDERLYINGS + ["VX_FRONT", "dVIX", "dVVIX"]
    factors = equity_returns.reindex(ret_frame.index).reindex(columns=factor_cols)
    rows = []
    for name in ret_frame.columns:
        fit = _ols(ret_frame[name], factors)
        betas = fit["betas"]
        row = {
            "Strategy": name,
            "Ann. alpha": fit["alpha"] * PERIODS_PER_YEAR if np.isfinite(fit["alpha"]) else np.nan,
            "Alpha HAC t": fit["alpha_t"],
            "Alpha HAC se": fit.get("alpha_se", np.nan) * PERIODS_PER_YEAR if np.isfinite(fit.get("alpha_se", np.nan)) else np.nan,
            "HAC lags": fit.get("hac_lags", 0),
            "$R^2$": fit["r2"],
            "Residual ann. vol": fit["residual_vol"] * np.sqrt(PERIODS_PER_YEAR)
            if np.isfinite(fit["residual_vol"])
            else np.nan,
            "N": fit["nobs"],
        }
        row["Beta SPY"] = betas.get(SPY_UNDERLYING, np.nan)
        for under in PRIMARY_UNDERLYINGS:
            row[f"Beta {under}"] = betas.get(under, np.nan)
        row["Beta VX front"] = betas.get("VX_FRONT", np.nan)
        row["Beta dVIX"] = betas.get("dVIX", np.nan)
        row["Beta dVVIX"] = betas.get("dVVIX", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def exposure_summary_table(
    model: OptionOnlyMarkowitzModel,
    option_strategies: dict[str, pd.Series],
    equity_benchmarks: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    kind = model.frame["kind"].astype(str)
    for name, weights in option_strategies.items():
        w = weights.reindex(model.contracts).fillna(0.0)
        delta_by_under = option_delta_by_underlying(model, w)
        rows.append(
            {
                "Strategy": name,
                "Book": "Options",
                "Gross NAV": float(w.abs().sum()),
                "Net NAV": float(w.sum()),
                "Long premium paid": float(w[w > 0].sum()),
                "Short premium sold": float(w[w < 0].abs().sum()),
                "Net option premium": float(w.sum()),
                "Net delta": float(delta_by_under.sum()),
                "Gross delta": float(delta_by_under.abs().sum()),
                "Net gamma": float(model.greeks["gamma_nav"].dot(w)),
                "Net vega": float(model.greeks["vega_nav"].dot(w)),
                "Equity vega": float(model.greeks.get("equity_vega_nav", pd.Series(0.0, index=model.contracts)).dot(w)),
                "VIX vega": float(model.greeks.get("vix_vega_nav", pd.Series(0.0, index=model.contracts)).dot(w)),
                "Beta SPY proxy": float(model.greeks.get("beta_spy_nav", pd.Series(0.0, index=model.contracts)).dot(w)),
                "Worst stress return": float(np.min(model._stress_matrix() @ w.to_numpy(float))) if model._stress_matrix() is not None else np.nan,
                "VIX option gross": float(w[model.frame.get("asset_class", pd.Series("", index=model.contracts)).astype(str).eq("vix_option")].abs().sum()),
                "Call gross": float(w[kind.eq("call")].abs().sum()),
                "Put gross": float(w[kind.eq("put")].abs().sum()),
                "Short gross": float(w[w < 0].abs().sum()),
            }
        )
    for name, weights in equity_benchmarks.items():
        w = weights.reindex(PRIMARY_UNDERLYINGS).fillna(0.0)
        rows.append(
            {
                "Strategy": name,
                "Book": "Equities",
                "Gross NAV": float(w.abs().sum()),
                "Net NAV": float(w.sum()),
                "Long premium paid": np.nan,
                "Short premium sold": np.nan,
                "Net option premium": np.nan,
                "Net delta": float(w.sum()),
                "Gross delta": float(w.abs().sum()),
                "Net gamma": 0.0,
                "Net vega": 0.0,
                "Equity vega": 0.0,
                "VIX vega": 0.0,
                "Beta SPY proxy": np.nan,
                "Worst stress return": np.nan,
                "VIX option gross": 0.0,
                "Call gross": np.nan,
                "Put gross": np.nan,
                "Short gross": float(w[w < 0].abs().sum()),
            }
        )
    return pd.DataFrame(rows)


def pnl_attribution_table(
    model: OptionOnlyMarkowitzModel,
    option_strategies: dict[str, pd.Series],
    test_returns: pd.DataFrame,
    return_detail: pd.DataFrame,
    reps: pd.DataFrame,
) -> pd.DataFrame:
    idx = test_returns.index
    hist = return_detail[return_detail["asset_id"].isin(model.contracts)].copy()
    hist = hist[hist["return_date"].isin(idx)].copy()
    next_iv = (
        reps.groupby(["snap_date", "asset_id"])["iv_proxy"]
        .median()
        .unstack("asset_id")
        .sort_index()
        .reindex(index=idx, columns=model.contracts)
    )
    current_iv = (
        hist.pivot(index="return_date", columns="asset_id", values="iv_proxy")
        .reindex(index=idx, columns=model.contracts)
    )
    dvol = (next_iv - current_iv).stack(future_stack=True).rename("dvol").reset_index()
    dvol = dvol.rename(columns={"snap_date": "return_date"})
    hist = hist.merge(dvol, on=["return_date", "asset_id"], how="left")
    hist["dvol"] = hist["dvol"].fillna(0.0) * hist["expiry_weight"].fillna(1.0)
    if "terminal_forward_proxy" not in hist:
        hist["terminal_forward_proxy"] = np.nan
    is_vix = hist.get("asset_class", pd.Series("", index=hist.index)).astype(str).eq("vix_option") | hist["underlying"].astype(str).eq(VIX_FACTOR)
    hist["dS"] = np.where(
        is_vix & hist["terminal_forward_proxy"].notna(),
        hist["terminal_forward_proxy"] - hist["start_spot"],
        hist["expiry_spot_proxy"] - hist["start_spot"],
    )
    denom = hist["mark"].replace(0.0, np.nan)
    dt = hist["expiry_days"].clip(lower=0).fillna(0.0) / 365.0
    delta_component = hist["delta"] * hist["dS"] / denom
    gamma_component = 0.5 * hist["gamma"] * hist["dS"] * hist["dS"] / denom
    vega_component = hist["vega"] * hist["dvol"] / denom
    hist["Equity delta"] = np.where(is_vix, 0.0, delta_component)
    hist["Equity gamma"] = np.where(is_vix, 0.0, gamma_component)
    hist["Equity vega"] = np.where(is_vix, 0.0, vega_component)
    hist["VIX-forward delta"] = np.where(is_vix, delta_component, 0.0)
    hist["VIX-forward gamma"] = np.where(is_vix, gamma_component, 0.0)
    hist["VIX-option vega"] = np.where(is_vix, vega_component, 0.0)
    hist["Theta/carry"] = hist["theta"] * dt / denom
    hist["VX roll"] = 0.0
    hist["Skew/tail"] = 0.0
    component_names = [
        "Equity delta",
        "Equity gamma",
        "Equity vega",
        "VIX-forward delta",
        "VIX-forward gamma",
        "VIX-option vega",
        "Theta/carry",
        "VX roll",
        "Skew/tail",
    ]
    components = {
        name: hist.pivot(index="return_date", columns="asset_id", values=name)
        .reindex(index=idx, columns=model.contracts)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        for name in component_names
    }

    rows = []
    actual = test_returns.reindex(columns=model.contracts).fillna(0.0)
    for name, weights in option_strategies.items():
        w = weights.reindex(model.contracts).fillna(0.0).to_numpy(float)
        series = {part: pd.Series(frame.to_numpy(float) @ w, index=idx) for part, frame in components.items()}
        fitted = sum(series.values())
        realized = pd.Series(actual.to_numpy(float) @ w, index=idx)
        residual = realized - fitted
        row = {"Strategy": name}
        for part in component_names:
            row[part] = float(series[part].mean() * PERIODS_PER_YEAR)
        row["Residual"] = float(residual.mean() * PERIODS_PER_YEAR)
        row["Realized ann. mean"] = float(realized.mean() * PERIODS_PER_YEAR)
        row["Residual vol share"] = float(residual.var(ddof=1) / realized.var(ddof=1)) if realized.var(ddof=1) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def regime_performance_table(ret_frame: pd.DataFrame, spy_returns: pd.Series) -> pd.DataFrame:
    spy = spy_returns.reindex(ret_frame.index).replace([np.inf, -np.inf], np.nan)
    q1, q2 = spy.quantile([1.0 / 3.0, 2.0 / 3.0])
    regimes = pd.Series("Flat", index=ret_frame.index)
    regimes[spy <= q1] = "Down"
    regimes[spy >= q2] = "Up"
    rows = []
    for name in ret_frame.columns:
        for regime in ["Down", "Flat", "Up"]:
            r = ret_frame.loc[regimes.eq(regime), name].dropna()
            st = performance_stats(r, PERIODS_PER_YEAR, benchmark_returns=spy_returns.reindex(r.index))
            sharpe_ci = block_bootstrap_metric_ci(r, "sharpe", BootstrapConfig(n_boot=500))
            rows.append(
                {
                    "Strategy": name,
                    "Regime": regime,
                    "Months": int(len(r)),
                    "Ann. return": st["ann_return"],
                    "Ann. vol": st["ann_vol"],
                    "Sharpe": st["sharpe"],
                    "Sharpe CI lo": sharpe_ci[1],
                    "Sharpe CI hi": sharpe_ci[2],
                    "Sortino": st["sortino"],
                    "Calmar": st["calmar"],
                    "Omega": st["omega"],
                    "Info. ratio": st["information_ratio"],
                }
            )
    return pd.DataFrame(rows)


def leave_one_out_table(reps: pd.DataFrame, returns: pd.DataFrame, spy_returns: pd.Series) -> pd.DataFrame:
    exclusions: list[tuple[str, list[str]]] = [("All underlyings", [])]
    exclusions.extend((f"No {u}", [u]) for u in PRIMARY_UNDERLYINGS)
    exclusions.append(("No META/NVDA/TSLA", ["META", "NVDA", "TSLA"]))
    rows = []
    for label, excluded in exclusions:
        universe = [u for u in PRIMARY_UNDERLYINGS if u not in excluded]
        model_universe = universe + ([VIX_FACTOR] if VIX_FACTOR in set(reps["underlying"].astype(str)) else [])
        asset_ids = reps.loc[reps["underlying"].isin(model_universe), "asset_id"].unique()
        sub_returns = returns.reindex(columns=[c for c in returns.columns if c in asset_ids]).dropna(how="all")
        sub_reps = reps[reps["asset_id"].isin(sub_returns.columns)].copy()
        sub_spec = representative_specs(sub_reps, sub_returns)
        sub_returns = sub_returns.reindex(columns=sub_spec.index).dropna(how="all")
        if len(sub_spec) < 8:
            continue
        sub_model, _ = make_model(sub_spec, sub_returns, sub_reps, model_universe)
        weights = strategy_weights(sub_model, model_universe)["Greek Markowitz"]
        test = sub_returns.loc[sub_returns.index > TRAIN_END, sub_model.contracts].fillna(0.0)
        pr = sub_model.portfolio_return_series(test, weights)
        st = performance_stats(pr, PERIODS_PER_YEAR, benchmark_returns=spy_returns.reindex(pr.index))
        beta_fit = _ols(pr, spy_returns.rename(SPY_UNDERLYING).to_frame())
        sharpe_ci = block_bootstrap_metric_ci(pr, "sharpe", BootstrapConfig(n_boot=500))
        rows.append(
            {
                "Exclusion": label,
                "Remaining underlyings": len(universe),
                "Option assets": len(sub_model.contracts),
                "Ann. return": st["ann_return"],
                "Ann. vol": st["ann_vol"],
                "Sharpe": st["sharpe"],
                "Sharpe CI lo": sharpe_ci[1],
                "Sharpe CI hi": sharpe_ci[2],
                "Sortino": st["sortino"],
                "Calmar": st["calmar"],
                "Omega": st["omega"],
                "Info. ratio": st["information_ratio"],
                "Net delta": float(option_delta_by_underlying(sub_model, weights).sum()),
                "Beta SPY": beta_fit["betas"].get(SPY_UNDERLYING, np.nan),
            }
        )
    return pd.DataFrame(rows)


def timing_diagnostics_table(
    returns: pd.DataFrame,
    return_detail: pd.DataFrame,
    split_events: pd.DataFrame,
) -> pd.DataFrame:
    train_detail = return_detail[return_detail["return_date"].le(TRAIN_END)]
    test_detail = return_detail[return_detail["return_date"].gt(TRAIN_END)]
    rows = [
        {
            "Diagnostic": "Return construction",
            "Value": "Prior-date option selection, split-adjusted listed-expiry payoff",
        },
        {
            "Diagnostic": "Minimum option mark filter",
            "Value": f"{MIN_OPTION_MARK:.2f}",
        },
        {
            "Diagnostic": "Max train decision date",
            "Value": pd.Timestamp(train_detail["decision_date"].max()).date().isoformat()
            if not train_detail.empty
            else "",
        },
        {
            "Diagnostic": "Max train realization date",
            "Value": pd.Timestamp(train_detail["return_date"].max()).date().isoformat()
            if not train_detail.empty
            else "",
        },
        {
            "Diagnostic": "First test decision date",
            "Value": pd.Timestamp(test_detail["decision_date"].min()).date().isoformat()
            if not test_detail.empty
            else "",
        },
        {
            "Diagnostic": "First test realization date",
            "Value": pd.Timestamp(test_detail["return_date"].min()).date().isoformat()
            if not test_detail.empty
            else "",
        },
        {
            "Diagnostic": "Detected split/unit adjustments",
            "Value": f"{len(split_events):,}",
        },
        {
            "Diagnostic": "Max single option return after repair",
            "Value": f"{np.nanmax(returns.to_numpy(float)):.3f}",
        },
        {
            "Diagnostic": "Min single option return after repair",
            "Value": f"{np.nanmin(returns.to_numpy(float)):.3f}",
        },
        {
            "Diagnostic": "Expiry spot source",
            "Value": "Raw daily close on listed expiry, or prior trading day if missing",
        },
    ]
    return pd.DataFrame(rows)


def trading_data_audit_table(
    full_panel: pd.DataFrame,
    reps: pd.DataFrame,
    returns: pd.DataFrame,
    return_detail: pd.DataFrame,
    raw_close: pd.DataFrame,
    split_events: pd.DataFrame,
) -> pd.DataFrame:
    """Generate mechanical checks for the PIT option holding model."""

    detail = return_detail.copy()
    opra_spot = (
        reps.groupby(["snap_date", "underlying"])["spot"]
        .median()
        .unstack("underlying")
        .sort_index()
    )
    raw_on_snap = raw_close.reindex(opra_spot.index).reindex(columns=opra_spot.columns)
    spot_mismatch = ((opra_spot - raw_on_snap).abs() / opra_spot.abs().replace(0.0, np.nan)).stack()
    payoff_lag = (
        pd.to_datetime(detail["expiry"]) - pd.to_datetime(detail["payoff_date"])
        if not detail.empty
        else pd.Series(dtype="timedelta64[ns]")
    )
    test_detail = detail[detail["return_date"].gt(TRAIN_END)]
    train_detail = detail[detail["return_date"].le(TRAIN_END)]
    rows = [
        {
            "Check": "OPRA-derived rows after liquidity filters",
            "Value": f"{len(full_panel):,}",
            "Pass": "yes" if len(full_panel) > 100_000 else "no",
        },
        {
            "Check": "Representative option choices",
            "Value": f"{len(reps):,}",
            "Pass": "yes" if len(reps) > 1_000 else "no",
        },
        {
            "Check": "Nonmissing option return cells",
            "Value": f"{int(returns.count().sum()):,}",
            "Pass": "yes" if int(returns.count().sum()) > 1_000 else "no",
        },
        {
            "Check": "Max OPRA spot vs raw close mismatch",
            "Value": f"{float(spot_mismatch.max()):.3e}" if len(spot_mismatch) else "",
            "Pass": "yes" if len(spot_mismatch) and float(spot_mismatch.max()) < 1e-8 else "no",
        },
        {
            "Check": "All option choices made before payoff date",
            "Value": f"{bool((detail['decision_date'] < detail['payoff_date']).all())}",
            "Pass": "yes" if not detail.empty and bool((detail["decision_date"] < detail["payoff_date"]).all()) else "no",
        },
        {
            "Check": "All payoff dates no later than next rebalance",
            "Value": f"{bool((detail['payoff_date'] <= detail['return_date']).all())}",
            "Pass": "yes" if not detail.empty and bool((detail["payoff_date"] <= detail["return_date"]).all()) else "no",
        },
        {
            "Check": "Exact listed-expiry close share",
            "Value": f"{float((detail['payoff_date'] == detail['expiry']).mean()):.3f}" if not detail.empty else "",
            "Pass": "yes" if not detail.empty and float((detail["payoff_date"] == detail["expiry"]).mean()) > 0.95 else "no",
        },
        {
            "Check": "Max expiry-to-payoff-date lag in days",
            "Value": f"{int(payoff_lag.dt.days.max())}" if len(payoff_lag) else "",
            "Pass": "yes" if len(payoff_lag) and int(payoff_lag.dt.days.max()) <= 3 else "no",
        },
        {
            "Check": "Holding period days min/median/max",
            "Value": (
                f"{int(detail['expiry_days'].min())}/"
                f"{float(detail['expiry_days'].median()):.1f}/"
                f"{int(detail['expiry_days'].max())}"
            )
            if not detail.empty
            else "",
            "Pass": "yes" if not detail.empty and detail["expiry_days"].between(7, 31).all() else "no",
        },
        {
            "Check": "Rows requiring split conversion during holding",
            "Value": f"{int((detail['split_factor_ratio'].round(8) != 1.0).sum())}" if not detail.empty else "",
            "Pass": "yes" if not detail.empty else "no",
        },
        {
            "Check": "Detected raw-close split/unit events",
            "Value": f"{len(split_events):,}",
            "Pass": "yes" if len(split_events) >= 5 else "no",
        },
        {
            "Check": "Minimum single-option return",
            "Value": f"{float(np.nanmin(returns.to_numpy(float))):.3f}",
            "Pass": "yes" if float(np.nanmin(returns.to_numpy(float))) >= -1.0000001 else "no",
        },
        {
            "Check": "Maximum single-option return",
            "Value": f"{float(np.nanmax(returns.to_numpy(float))):.3f}",
            "Pass": "yes" if np.isfinite(np.nanmax(returns.to_numpy(float))) else "no",
        },
        {
            "Check": "Max training decision before first test decision",
            "Value": (
                f"{pd.Timestamp(train_detail['decision_date'].max()).date().isoformat()} before "
                f"{pd.Timestamp(test_detail['decision_date'].min()).date().isoformat()}"
            )
            if not train_detail.empty and not test_detail.empty
            else "",
            "Pass": "yes"
            if not train_detail.empty
            and not test_detail.empty
            and pd.Timestamp(train_detail["decision_date"].max()) < pd.Timestamp(test_detail["decision_date"].min())
            else "no",
        },
    ]
    return pd.DataFrame(rows)


def plot_growth(
    ret_frame: pd.DataFrame,
    path: Path,
    columns: list[str] | None = None,
    title: str = "Portfolio growth, test window",
) -> None:
    plot_frame = ret_frame.copy()
    if columns is not None:
        plot_frame = plot_frame[[c for c in columns if c in plot_frame.columns]]
    cumulative = (1.0 + plot_frame.fillna(0.0)).cumprod()
    plot_values = cumulative.clip(lower=1e-4)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    linestyles = ["-", "--", "-.", ":", "-"]
    marker_step = max(len(plot_values) // 6, 1)
    for i, col in enumerate(plot_values.columns):
        ax.plot(
            plot_values.index,
            plot_values[col],
            label=col,
            color=JOURNAL_COLORS[i % len(JOURNAL_COLORS)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2.15,
            marker="o",
            markevery=(i % marker_step, marker_step),
            markersize=3.0,
            alpha=0.96,
        )
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale; floor at 1e-4)$")
    ax.set_xlabel("Out-of-sample month")
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    ax.grid(True, axis="x", alpha=0.08, linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2 if len(plot_values.columns) <= 5 else 3, fontsize=7.8, frameon=False)
    ax.margins(x=0.01)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(path)
    plt.close(fig)

def plot_regime_sharpes(
    regime: pd.DataFrame,
    path: Path,
    title: str = "Performance by SPY market regime",
    clip_floor: float = -8.0,
) -> None:
    # Preserve the regime order in which the table was built; hardcoding
    # Down/Flat/Up silently blanked the Low/Mid/High VIX variant of this plot.
    regime_order = list(pd.unique(regime["Regime"]))
    pivot = regime.pivot(index="Strategy", columns="Regime", values="Sharpe").reindex(
        columns=regime_order
    )
    # Clip extreme negative outliers so one failed sleeve does not compress
    # every other strategy onto a single pixel row; annotate the true value.
    clipped = pivot.where(pivot >= clip_floor)
    clip_notes = pivot[pivot < clip_floor]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(pivot.columns))
    colors = plt.get_cmap("tab10").colors
    n = len(pivot)
    for i, (name, row) in enumerate(clipped.iterrows()):
        x_jitter = x + (i - (n - 1) / 2.0) * 0.012
        vals = row.to_numpy(float)
        ax.plot(
            x_jitter,
            vals,
            label=name,
            color=colors[i % len(colors)],
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
        if name in clip_notes.index:
            for j, col in enumerate(clipped.columns):
                true_val = clip_notes.loc[name, col] if col in clip_notes.columns else np.nan
                if np.isfinite(true_val):
                    ax.plot(x_jitter[j], clip_floor, marker="v", color=colors[i % len(colors)], markersize=6)
                    ax.annotate(
                        f"{true_val:.1f}",
                        (x_jitter[j], clip_floor),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                        color=colors[i % len(colors)],
                    )
    if not clip_notes.dropna(how="all").empty:
        ax.set_ylim(bottom=clip_floor - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.columns)
    ax.set_ylabel("Sharpe")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(path)
    plt.close(fig)


def plot_leave_one_out(leave_one: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    data = leave_one.set_index("Exclusion")["Sharpe"]
    colors = ["#1f77b4" if "META/NVDA/TSLA" not in idx else "#8b1e3f" for idx in data.index]
    ax.bar(np.arange(len(data)), data.to_numpy(float), color=colors, edgecolor="#334", linewidth=0.5)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.index, rotation=35, ha="right")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.set_title("Greek Markowitz leave-one-underlying-out tests")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _strategy_order_from_frame(frame: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    strategy_col = _first_existing_column(frame, ["strategy", "Strategy"])
    if strategy_col is None:
        return []
    present = set(frame[strategy_col].dropna().astype(str))
    ordered = [strategy for strategy in preferred if strategy in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _realized_sharpe_lookup(realized: pd.DataFrame, basis: str = "gross") -> dict[str, float]:
    if realized.empty:
        return {}
    strategy_col = _first_existing_column(realized, ["strategy", "Strategy"])
    sharpe_col = _first_existing_column(realized, ["sharpe", "Sharpe", "Realized Gross Sharpe", "Sharpe Realized"])
    if strategy_col is None or sharpe_col is None:
        return {}
    frame = realized.copy()
    if "basis" in frame.columns:
        frame = frame[frame["basis"].astype(str).eq(basis)]
    elif "Basis" in frame.columns:
        frame = frame[frame["Basis"].astype(str).eq(basis)]
    frame[sharpe_col] = pd.to_numeric(frame[sharpe_col], errors="coerce")
    frame = frame.dropna(subset=[strategy_col, sharpe_col])
    return frame.groupby(frame[strategy_col].astype(str))[sharpe_col].first().to_dict()


def _realized_gross_sharpes_from_returns(ret_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in ret_frame.columns:
        stats = performance_stats(pd.Series(ret_frame[strategy]).dropna(), PERIODS_PER_YEAR)
        rows.append({"strategy": strategy, "basis": "gross", "sharpe": float(stats.get("sharpe", np.nan))})
    return pd.DataFrame(rows)


def _realized_gross_sharpes_from_empirical_summary(summary_path: Path) -> pd.DataFrame:
    """Load realized static-split gross Sharpes from empirical_summary.json."""

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing realized-Sharpe source: {summary_path.relative_to(PAPER)}. "
            "Run the empirical stage before regenerating robustness figures."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("performance", []):
        if str(row.get("Return basis", "")) != "Gross before costs":
            continue
        rows.append(
            {
                "strategy": row.get("Strategy"),
                "basis": "gross",
                "sharpe": row.get("Sharpe"),
            }
        )
    out = pd.DataFrame(rows, columns=["strategy", "basis", "sharpe"])
    if not out.empty:
        out["sharpe"] = pd.to_numeric(out["sharpe"], errors="coerce")
    return out


def _attach_realized_sharpes(paths: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    out = paths.copy()
    lookup = _realized_sharpe_lookup(realized, "gross")
    if "strategy" in out.columns and lookup:
        out["realized_sharpe"] = out["strategy"].astype(str).map(lookup)
    return out


def _realized_lookup_from_paths(*frames: pd.DataFrame) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for frame in frames:
        if frame.empty or not {"strategy", "realized_sharpe"}.issubset(frame.columns):
            continue
        sub = frame[["strategy", "realized_sharpe"]].copy()
        sub["realized_sharpe"] = pd.to_numeric(sub["realized_sharpe"], errors="coerce")
        sub = sub.dropna(subset=["strategy", "realized_sharpe"])
        lookup.update(sub.groupby(sub["strategy"].astype(str))["realized_sharpe"].first().to_dict())
    return lookup


def _finite_sharpes(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "sharpe" not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame["sharpe"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _set_common_sharpe_xlim(axes: Sequence[plt.Axes], values: Sequence[float]) -> None:
    finite = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo = min(float(finite.min()), 0.0)
        hi = max(float(finite.max()), 0.0)
        pad = max((hi - lo) * 0.08, 0.25)
        lo -= pad
        hi += pad
    for ax in axes:
        ax.set_xlim(lo, hi)


def plot_cpcv_sharpe_distribution(
    path_metrics: pd.DataFrame,
    realized: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot complete CPCV path Sharpes against realized static-split gross Sharpe."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    strategy_order = _strategy_order_from_frame(path_metrics, CV_STRATEGIES)
    strategy_order = [strategy for strategy in CV_STRATEGIES if strategy in strategy_order]
    if not strategy_order:
        strategy_order = list(CV_STRATEGIES)
    # The standalone hedge sleeve's realized Sharpe (~-6.4) would compress the
    # x-axis and hide the interesting dispersion; it stays in the fold table.
    strategy_order = [s for s in strategy_order if s != "VIX hedge sleeve"] or strategy_order

    frame = path_metrics.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).eq("complete")]
    frame = frame[frame.get("strategy", pd.Series(dtype=object)).astype(str).isin(strategy_order)]
    realized_lookup = _realized_sharpe_lookup(realized, "gross")

    basis_specs = [
        ("gross", "Gross CPCV", JOURNAL_COLORS[0], -0.10),
        ("full_spread_post_cost", "Full-spread CPCV", JOURNAL_COLORS[2], 0.10),
    ]
    y_positions = np.arange(len(strategy_order), dtype=float)
    for basis, _label, color, offset in basis_specs:
        basis_frame = frame[frame.get("basis", pd.Series(dtype=object)).astype(str).eq(basis)]
        for i, strategy in enumerate(strategy_order):
            sub = basis_frame[basis_frame["strategy"].astype(str).eq(strategy)].sort_values("path_id")
            sharpe = _finite_sharpes(sub)
            if sharpe.empty:
                continue
            jitter = np.linspace(-0.035, 0.035, len(sharpe)) if len(sharpe) > 1 else np.array([0.0])
            ax.scatter(
                sharpe.to_numpy(float),
                np.full(len(sharpe), y_positions[i] + offset) + jitter,
                s=28,
                color=color,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
    for i, strategy in enumerate(strategy_order):
        marker = realized_lookup.get(strategy, np.nan)
        if np.isfinite(marker):
            ax.scatter(
                marker,
                y_positions[i],
                marker="D",
                s=48,
                color=JOURNAL_COLORS[1],
                edgecolor="#24313A",
                linewidth=0.55,
                zorder=4,
            )

    all_values = _finite_sharpes(frame).to_list() + [
        v for s, v in realized_lookup.items() if s in strategy_order and np.isfinite(v)
    ]
    _set_common_sharpe_xlim([ax], all_values)
    ax.axvline(0.0, color="#555", linestyle="--", linewidth=1.0, alpha=0.70, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(strategy_order)
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe")
    ax.set_ylabel("Strategy")
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.7)
    ax.grid(True, axis="y", alpha=0.07, linewidth=0.5)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=JOURNAL_COLORS[0], markeredgecolor="white", markersize=6, label="Gross CPCV"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=JOURNAL_COLORS[2], markeredgecolor="white", markersize=6, label="Full-spread CPCV"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=JOURNAL_COLORS[1], markeredgecolor="#24313A", markersize=6, label="Realized gross"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(out_path)
    plt.close(fig)


def _fold_window_label(row: pd.Series) -> str:
    start = pd.to_datetime(row.get("test_start"), errors="coerce")
    end = pd.to_datetime(row.get("test_end"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return str(row.get("fold_id", ""))
    return f"{start:%Y-%m}..{end:%Y-%m}"


def plot_fold_sharpe_heatmap(
    fold_ledger: pd.DataFrame,
    fold_schedule: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot blocked k-fold gross Sharpe values as a strategy-by-window heatmap."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schedule = fold_schedule.copy()
    if not schedule.empty and "scheme" in schedule.columns:
        schedule = schedule[schedule["scheme"].astype(str).eq("kfold")]
    if not schedule.empty and "test_start" in schedule.columns:
        schedule = schedule.assign(_test_start=pd.to_datetime(schedule["test_start"], errors="coerce"))
        schedule = schedule.sort_values(["_test_start", "fold_id"])
    fold_ids = schedule.get("fold_id", pd.Series(dtype=object)).astype(str).tolist()
    labels = [_fold_window_label(row) for _, row in schedule.iterrows()]

    ledger = fold_ledger.copy()
    if not ledger.empty:
        if "scheme" in ledger.columns:
            ledger = ledger[ledger["scheme"].astype(str).eq("kfold")]
        if "basis" in ledger.columns:
            ledger = ledger[ledger["basis"].astype(str).eq("gross")]
        if "status" in ledger.columns:
            ledger = ledger[ledger["status"].astype(str).eq("ok")]
    strategies = [strategy for strategy in CV_STRATEGIES if strategy in set(ledger.get("strategy", pd.Series(dtype=object)).astype(str))]
    if not strategies:
        strategies = list(CV_STRATEGIES)
    if not fold_ids:
        fold_ids = sorted(ledger.get("fold_id", pd.Series(dtype=object)).astype(str).unique())
        labels = fold_ids

    pivot = (
        ledger.assign(sharpe=pd.to_numeric(ledger.get("sharpe", pd.Series(dtype=float)), errors="coerce"))
        .pivot_table(index="strategy", columns="fold_id", values="sharpe", aggfunc="first")
        .reindex(index=strategies, columns=fold_ids)
    )
    values = pivot.to_numpy(float) if not pivot.empty else np.empty((len(strategies), len(fold_ids)))
    finite = values[np.isfinite(values)]
    # Clip the color scale at +/-3: the hedge sleeve's extreme fold Sharpes
    # (~-20) would otherwise wash out every other cell.  Annotations keep the
    # true values, and the caption discloses the clipping.
    vmax = min(max(float(np.nanmax(np.abs(finite))) if finite.size else 1.0, 1.0), 3.0)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#F1F1F1")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(fold_ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Test window")
    ax.set_ylabel("Strategy")
    ax.set_xticks(np.arange(-0.5, len(fold_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(strategies), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > 0.58 * vmax else "#222"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Gross Sharpe")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _mc_distribution_values(
    paths: pd.DataFrame,
    strategy: str,
    *,
    universe_family: str | None = None,
    basis: str | None = None,
    method: str | None = None,
) -> pd.Series:
    frame = paths.copy()
    if universe_family is not None and "universe_family" in frame.columns:
        frame = frame[frame["universe_family"].astype(str).eq(universe_family)]
    if basis is not None and "basis" in frame.columns:
        frame = frame[frame["basis"].astype(str).eq(basis)]
    if method is not None and "method" in frame.columns:
        frame = frame[frame["method"].astype(str).eq(method)]
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).eq("ok")]
    if "strategy" not in frame.columns:
        return pd.Series(dtype=float)
    frame = frame[frame["strategy"].astype(str).eq(strategy)]
    return _finite_sharpes(frame)


def _plot_strategy_violins(
    ax: plt.Axes,
    paths: pd.DataFrame,
    strategies: Sequence[str],
    realized_lookup: dict[str, float],
    *,
    universe_family: str | None = None,
    basis: str | None = None,
    method: str | None = None,
) -> list[float]:
    positions: list[int] = []
    data: list[np.ndarray] = []
    colors: list[str] = []
    all_values: list[float] = []
    for i, strategy in enumerate(strategies):
        vals = _mc_distribution_values(paths, strategy, universe_family=universe_family, basis=basis, method=method)
        if vals.empty:
            continue
        arr = vals.to_numpy(float)
        positions.append(i)
        data.append(arr)
        colors.append(JOURNAL_COLORS[i % len(JOURNAL_COLORS)])
        all_values.extend(arr.tolist())
    if data:
        parts = ax.violinplot(data, positions=positions, orientation="horizontal", widths=0.72, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#263238")
            body.set_alpha(0.45)
            body.set_linewidth(0.55)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#202020")
            parts["cmedians"].set_linewidth(1.0)
    for i, strategy in enumerate(strategies):
        marker = realized_lookup.get(strategy, np.nan)
        if np.isfinite(marker):
            ax.scatter(
                marker,
                i,
                marker="D",
                s=30,
                color=JOURNAL_COLORS[2],
                edgecolor="#202020",
                linewidth=0.45,
                zorder=4,
            )
            all_values.append(float(marker))
    ax.axvline(0.0, color="#555", linestyle="--", linewidth=0.9, alpha=0.65, zorder=1)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Sharpe")
    ax.grid(True, axis="x", alpha=0.22, linewidth=0.7)
    ax.grid(True, axis="y", alpha=0.06, linewidth=0.5)
    return all_values


def plot_mc_universe_distributions(
    fixed_paths: pd.DataFrame,
    repriced_paths: pd.DataFrame,
    refit_paths: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot resampled, repriced, and refit Sharpe distributions."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    strategies = _strategy_order_from_frame(fixed_paths, MC_ROBUSTNESS_STRATEGIES)
    strategies = [strategy for strategy in MC_ROBUSTNESS_STRATEGIES if strategy in strategies]
    if not strategies:
        strategies = list(MC_ROBUSTNESS_STRATEGIES)
    realized_lookup = _realized_lookup_from_paths(fixed_paths, repriced_paths, refit_paths)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    top_values = []
    top_values.extend(
        _plot_strategy_violins(
            axes[0, 0],
            fixed_paths,
            strategies,
            realized_lookup,
            universe_family="resampled",
            basis="gross",
        )
    )
    top_values.extend(
        _plot_strategy_violins(
            axes[0, 1],
            fixed_paths,
            strategies,
            realized_lookup,
            universe_family="resampled_stratified",
            basis="gross",
        )
    )
    repriced_values = _plot_strategy_violins(
        axes[1, 0],
        repriced_paths,
        strategies,
        realized_lookup,
        method="joint_garch_block",
    )

    refit_strategy = "Greek Markowitz + VIX"
    refit_vals = _mc_distribution_values(refit_paths, refit_strategy)
    axes[1, 1].hist(refit_vals, bins=min(28, max(8, int(np.sqrt(max(len(refit_vals), 1))))), color="#D6E0DF", edgecolor="#40534C", linewidth=0.6, alpha=0.95)
    refit_realized = realized_lookup.get(refit_strategy, np.nan)
    refit_values = refit_vals.to_list()
    if np.isfinite(refit_realized):
        axes[1, 1].axvline(refit_realized, color=JOURNAL_COLORS[2], linewidth=2.4, label="Realized gross", zorder=4)
        refit_values.append(float(refit_realized))
    axes[1, 1].axvline(0.0, color="#555", linestyle="--", linewidth=0.9, alpha=0.65)
    axes[1, 1].set_xlabel("Sharpe")
    axes[1, 1].set_ylabel("Refit paths")
    axes[1, 1].grid(True, axis="y", alpha=0.22, linewidth=0.7)

    axes[0, 0].set_title("A. Resampled universe", fontsize=10, pad=7)
    axes[0, 1].set_title("B. Regime-stratified resampling", fontsize=10, pad=7)
    axes[1, 0].set_title("C. Repriced universe", fontsize=10, pad=7)
    axes[1, 1].set_title("D. Refit stability", fontsize=10, pad=7)
    axes[0, 1].set_ylabel("")
    axes[1, 0].set_ylabel("")
    axes[1, 1].legend(fontsize=8, frameon=False, loc="upper right")
    marker_handle = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=JOURNAL_COLORS[2], markeredgecolor="#202020", markersize=5.5, label="Realized gross")
    violin_handle = plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=JOURNAL_COLORS[0], markeredgecolor="#263238", alpha=0.50, markersize=7, label="Path distribution")
    axes[0, 0].legend(handles=[violin_handle, marker_handle], fontsize=8, frameon=False, loc="upper right")

    _set_common_sharpe_xlim([axes[0, 0], axes[0, 1]], top_values)
    _set_common_sharpe_xlim([axes[1, 0]], repriced_values)
    _set_common_sharpe_xlim([axes[1, 1]], refit_values)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def claim_audit_table(vix_headline_eligible: bool = False) -> pd.DataFrame:
    """Return the paper's claim-strength audit as a generated artifact."""

    exact_status = "Supported" if vix_headline_eligible else "Downgraded to diagnostic"
    exact_evidence = "All VIX headline rows use exact VRO/SOQ source flags" if vix_headline_eligible else "Exact VRO/SOQ coverage incomplete; proxy rows excluded from headline claims"
    return pd.DataFrame(
        [
            {
                "Claim": "Option covariance equals $B\\Omega B^\\top+\\Sigma_\\varepsilon$",
                "Type": "Theorem",
                "Evidence": "Displayed proof and standard covariance algebra",
                "Status": "Proved",
            },
            {
                "Claim": "Unconstrained option tangency is $\\Sigma_O^{-1}\\mu_O$",
                "Type": "Theorem",
                "Evidence": "Displayed Lagrange multiplier proof",
                "Status": "Proved",
            },
            {
                "Claim": "Greek Markowitz Sharpe exceeds random feasible p95",
                "Type": "Generated empirical",
                "Evidence": "performance table and random feasible Sharpe artifact",
                "Status": "Not supported after exact-expiry PIT audit",
            },
            {
                "Claim": "Result is not only long-call equity drift",
                "Type": "Generated diagnostic",
                "Evidence": "exposure, regression, regimes, equity benchmarks, leave-one-out",
                "Status": "Not supported as an alpha-independence claim",
            },
            {
                "Claim": "Option premium cost is included in P\\&L and Sharpe",
                "Type": "Accounting identity",
                "Evidence": "NAV weights, mark-return construction, exposure premium columns",
                "Status": "Implemented by construction",
            },
            {
                "Claim": "Empirical returns use point-in-time option choices",
                "Type": "Generated diagnostic",
                "Evidence": "timing diagnostics, trading-data audit, and holding-return detail artifacts",
                "Status": "Checked against train/test dates",
            },
            {
                "Claim": "Stock splits do not create option windfalls",
                "Type": "Generated diagnostic",
                "Evidence": "raw-close split factors and exact-expiry payoff construction",
                "Status": "Checked for detected split/unit jumps",
            },
            {
                "Claim": "Greek risk model fully explains option P\\&L",
                "Type": "Rejected overclaim",
                "Evidence": "approximation diagnostics and P\\&L attribution artifacts",
                "Status": "Not claimed",
            },
            {
                "Claim": "Post-cost research returns include implementation frictions",
                "Type": "Generated empirical",
                "Evidence": "Generated mid, half-spread, full-spread, fee, capacity, required-capital, and assignment-risk ledgers",
                "Status": "Implemented as conservative research simulation",
            },
            {
                "Claim": "Pre-production results are broker-executed live evidence",
                "Type": "Rejected overclaim",
                "Evidence": "No live fills, order routing, broker margin preview, or broker reconciliation",
                "Status": "Not claimed",
            },
            {
                "Claim": "VIX option results are headline-grade listed-settlement evidence",
                "Type": "Conditional empirical",
                "Evidence": exact_evidence,
                "Status": exact_status,
            },
            {
                "Claim": "Strategy is production tradable after costs",
                "Type": "Production claim",
                "Evidence": "No live fills, live broker margin preview, order routing, or broker reconciliation",
                "Status": "Not claimed",
            },
        ]
    )


def volatility_regime_performance_table(ret_frame: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    state = vix_state_panel(dates, ROOT)
    if state.empty or "VIX" not in state:
        return pd.DataFrame(columns=["Strategy", "Regime", "Months", "Ann. return", "Ann. vol", "Sharpe", "Sortino", "Calmar", "Omega", "Info. ratio"])
    vix = state["VIX"].reindex(ret_frame.index).replace([np.inf, -np.inf], np.nan)
    q1, q2 = vix.quantile([1.0 / 3.0, 2.0 / 3.0])
    regimes = pd.Series("Mid VIX", index=ret_frame.index)
    regimes[vix <= q1] = "Low VIX"
    regimes[vix >= q2] = "High VIX"
    vx_bench = state.get("VX_FRONT_return", pd.Series(index=ret_frame.index, dtype=float)).reindex(ret_frame.index)
    rows = []
    for name in ret_frame.columns:
        for regime in ["Low VIX", "Mid VIX", "High VIX"]:
            r = ret_frame.loc[regimes.eq(regime), name].dropna()
            st = performance_stats(r, PERIODS_PER_YEAR, benchmark_returns=vx_bench.reindex(r.index))
            sharpe_ci = block_bootstrap_metric_ci(r, "sharpe", BootstrapConfig(n_boot=500))
            rows.append(
                {
                    "Strategy": name,
                    "Regime": regime,
                    "Months": int(len(r)),
                    "Ann. return": st["ann_return"],
                    "Ann. vol": st["ann_vol"],
                    "Sharpe": st["sharpe"],
                    "Sharpe CI lo": sharpe_ci[1],
                    "Sharpe CI hi": sharpe_ci[2],
                    "Sortino": st["sortino"],
                    "Calmar": st["calmar"],
                    "Omega": st["omega"],
                    "Info. ratio": st["information_ratio"],
                }
            )
    return pd.DataFrame(rows)


def cost_input_spread_source_coverage_table(cost_inputs: pd.DataFrame) -> pd.DataFrame:
    columns = ["relative_spread_source", "asset_class", "rows", "mean_relative_spread"]
    if cost_inputs.empty:
        return pd.DataFrame(columns=columns)
    frame = cost_inputs.copy()
    source = frame.get("relative_spread_source", pd.Series("default", index=frame.index)).fillna("default").astype(str)
    asset_class_raw = frame.get("asset_class", pd.Series("", index=frame.index)).astype(str)
    underlying = frame.get("underlying", pd.Series("", index=frame.index)).astype(str).str.upper()
    asset_id = frame.get("asset_id", pd.Series("", index=frame.index)).astype(str).str.upper()
    frame["relative_spread_source"] = source
    frame["asset_class"] = np.where(
        asset_class_raw.eq("vix_option") | underlying.isin([VIX_FACTOR, "VX_FRONT"]) | asset_id.str.contains("VIX", regex=False),
        "VIX",
        "equity",
    )
    frame["relative_spread"] = pd.to_numeric(frame.get("relative_spread", np.nan), errors="coerce")
    return (
        frame.groupby(["relative_spread_source", "asset_class"], dropna=False, observed=True)
        .agg(rows=("relative_spread", "size"), mean_relative_spread=("relative_spread", "mean"))
        .reset_index()
        .sort_values(["relative_spread_source", "asset_class"])
    )


def data_extension_manifest(root: Path, cost_config: ResearchCostConfig, spread_surface: pd.DataFrame | None) -> pd.DataFrame:
    vix_paths = sorted((root / "data/databento_cache").glob("opra_vix_chain_*.parquet"))
    vix_size = sum(path.stat().st_size for path in vix_paths if path.exists())
    return pd.DataFrame(
        [
            {
                "dataset": "opra_surface_full_day_cbbo",
                "location": cost_config.cbbo_spread_surface_path,
                "size_approx": f"{len(spread_surface) if spread_surface is not None else 0:,} rows",
                "status": "integrated",
                "reason": "CBBO spread cost surface refines default spreads",
                "artifacts_produced": "cost_input_ledger.csv; cost_input_spread_source_coverage.csv",
            },
            {
                "dataset": "opra_vix_chain_*",
                "location": "data/databento_cache/opra_vix_chain_*.parquet",
                "size_approx": f"{len(vix_paths):,} files; {vix_size:,} bytes",
                "status": "extended",
                "reason": "state features + regime diagnostics; expected-return feed-in deferred",
                "artifacts_produced": "vix_chain_state_features.csv; vol_of_vol_regime_performance.csv",
            },
            {
                "dataset": "opra_{UND}_slices_*",
                "location": "data/databento_cache/opra_{UND}_slices_*.parquet",
                "size_approx": "not regenerated in this pipeline",
                "status": "deferred",
                "reason": "IWM/QQQ universe expansion requires regenerating option_greek_proxy_panel in the sibling-repo pipeline; META/TSLA already in universe",
                "artifacts_produced": "none",
            },
            {
                "dataset": "sibling DATA_ANALYSIS loaders",
                "location": "DATA_ANALYSIS loaders; data_ingestion",
                "size_approx": "referenced source code",
                "status": "referenced",
                "reason": "OSI parser vendored into data_ingestion",
                "artifacts_produced": "data_ingestion/build_cbbo_cost_surface.py",
            },
        ]
    )


def rolling_oos_table(
    returns: pd.DataFrame,
    reps: pd.DataFrame,
    universe: Sequence[str],
    *,
    spec_builder=representative_specs,
    model_factory=make_model,
) -> pd.DataFrame:
    """Rolling 36M out-of-sample diagnostic with a *real* rolling window.

    Each evaluation date refits specs and the model on the trailing 36-month
    window by passing explicit ``train_start``/``train_end`` bounds (previous
    behavior silently truncated every window at the global TRAIN_END, so the
    "roll" never rolled).  Portfolio weights use the same constrained
    ``solve_max_sharpe`` solver as the headline strategies rather than the
    unconstrained closed-form tangency.  ``spec_builder``/``model_factory``
    are injectable for testing.
    """

    rows = []
    dates = pd.DatetimeIndex(returns.index).sort_values()
    realized = []
    for pos, dt in enumerate(dates):
        if dt <= TRAIN_END or pos < 36 or (pos % 3 != 0):
            continue
        train_start = dates[max(0, pos - 36)]
        train_end = dates[pos - 1]
        if train_end >= dt:
            continue
        sub_returns = returns.loc[train_start:train_end].dropna(how="all")
        cols = sub_returns.columns[sub_returns.count() >= 18]
        if len(cols) < 8:
            continue
        sub_returns = returns.reindex(columns=cols).dropna(how="all")
        sub_reps = reps[reps["asset_id"].isin(cols)].copy()
        try:
            sub_spec = spec_builder(sub_reps, sub_returns, train_start=train_start, train_end=train_end)
            if len(sub_spec) < 8:
                continue
            model, _ = model_factory(
                sub_spec,
                sub_returns,
                sub_reps,
                universe,
                train_start=train_start,
                train_end=train_end,
            )
            weights = _solve_weights_with_guard(model, f"rolling_oos_{pd.Timestamp(dt).date()}")
            one = returns.loc[[dt], model.contracts].fillna(0.0)
            realized.append(float(model.portfolio_return_series(one, weights).iloc[0]))
            rows.append({"return_date": dt, "train_start": train_start, "train_end": train_end, "gross_nav": float(weights.abs().sum())})
        except Exception:
            continue
    series = pd.Series(realized, index=[r["return_date"] for r in rows], name="rolling_oos")
    st = performance_stats(series, PERIODS_PER_YEAR) if len(series) else {k: np.nan for k in ["ann_return", "ann_vol", "sharpe", "sortino", "calmar", "omega", "information_ratio"]}
    return pd.DataFrame(
        [
            {"Diagnostic": "Rolling 36M OOS months", "Value": len(series)},
            {"Diagnostic": "Rolling 36M OOS ann. return", "Value": st.get("ann_return", np.nan)},
            {"Diagnostic": "Rolling 36M OOS ann. vol", "Value": st.get("ann_vol", np.nan)},
            {"Diagnostic": "Rolling 36M OOS Sharpe", "Value": st.get("sharpe", np.nan)},
            {"Diagnostic": "Rolling 36M OOS Sortino", "Value": st.get("sortino", np.nan)},
            {"Diagnostic": "Rolling 36M OOS Calmar", "Value": st.get("calmar", np.nan)},
            {"Diagnostic": "Rolling 36M OOS Omega", "Value": st.get("omega", np.nan)},
        ]
    )


def liquidity_tier_rerun_tables(
    returns: pd.DataFrame,
    reps: pd.DataFrame,
    tier_map: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerun the option optimizer inside each liquidity-tier universe."""

    if tier_map.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    diag = []
    for tier, tier_rows in tier_map.groupby("liquidity_tier"):
        asset_ids = [c for c in returns.columns if c in set(tier_rows["asset_id"])]
        sub_returns = returns.reindex(columns=asset_ids).dropna(how="all")
        sub_reps = reps[reps["asset_id"].isin(asset_ids)].copy()
        if sub_returns.shape[1] < 6 or sub_reps.empty:
            diag.append({"Liquidity tier": tier, "Method": "rerun_optimizer_on_tier", "Eligible assets": len(asset_ids), "Status": "skipped_insufficient_assets"})
            continue
        sub_universe = sorted(sub_reps["underlying"].astype(str).unique())
        try:
            sub_spec = representative_specs(sub_reps, sub_returns)
            sub_returns = sub_returns.reindex(columns=sub_spec.index).dropna(how="all")
            if len(sub_spec) < 6:
                diag.append({"Liquidity tier": tier, "Method": "rerun_optimizer_on_tier", "Eligible assets": len(asset_ids), "Status": "skipped_insufficient_specs"})
                continue
            sub_model, _ = make_model(sub_spec, sub_returns, sub_reps, sub_universe)
            sub_strategies = strategy_weights(sub_model, sub_universe)
            if any(sub_model.frame.get("asset_class", pd.Series("", index=sub_model.contracts)).astype(str).eq("vix_option")):
                sub_strategies["VIX hedge sleeve"] = vix_hedge_sleeve_weights(sub_model)
            test = sub_returns.loc[sub_returns.index > TRAIN_END, sub_model.contracts].fillna(0.0)
            spy = factor_returns[SPY_UNDERLYING].reindex(test.index) if SPY_UNDERLYING in factor_returns else None
            for strategy, weights in sub_strategies.items():
                pr = sub_model.portfolio_return_series(test, weights)
                st = performance_stats(pr, PERIODS_PER_YEAR, benchmark_returns=spy)
                rows.append(
                    {
                        "Liquidity tier": tier,
                        "Strategy": strategy,
                        "Ann. return": st["ann_return"],
                        "Ann. vol": st["ann_vol"],
                        "Sharpe": st["sharpe"],
                        "Calmar": st["calmar"],
                        "Omega": st["omega"],
                    }
                )
                diag.append(
                    {
                        "Liquidity tier": tier,
                        "Strategy": strategy,
                        "Method": "rerun_optimizer_on_tier",
                        "Eligible assets": len(asset_ids),
                        "Active assets": int((weights.abs() > 1e-14).sum()),
                        "Gross NAV used": float(weights.abs().sum()),
                        "Status": "ok",
                    }
                )
        except Exception as exc:
            diag.append({"Liquidity tier": tier, "Method": "rerun_optimizer_on_tier", "Eligible assets": len(asset_ids), "Status": f"failed:{type(exc).__name__}"})
    return pd.DataFrame(rows), pd.DataFrame(diag)


def _write_visibility_audit(ret_frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    cumulative = (1.0 + ret_frame.fillna(0.0)).cumprod().clip(lower=1e-4)
    for col in cumulative.columns:
        series = cumulative[col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "Figure": path.name,
                "Series": col,
                "Visible points": int(series.count()),
                "Min": float(series.min()) if not series.empty else np.nan,
                "Max": float(series.max()) if not series.empty else np.nan,
                "Pass": "yes" if series.count() > 0 and float(series.max() - series.min()) >= 0 else "no",
            }
        )
    return pd.DataFrame(rows)


def claim_strength_table(vix_headline_eligible: bool) -> pd.DataFrame:
    vix_condition = "exact VRO/SOQ rows present" if vix_headline_eligible else "proxy rows force diagnostic status"
    return pd.DataFrame(
        [
            {
                "Strength": "Strong",
                "Claim": "Option cashflow accounting, Greek-induced covariance, premium-weighted Markowitz formulation, and validation discipline are coherent.",
                "Boundary": "Mathematical construction and reproducibility checks.",
            },
            {
                "Strength": "Moderate",
                "Claim": "The point-in-time research implementation produces useful gross and post-cost diagnostics in this sample.",
                "Boundary": "Monthly OPRA/Databento-derived panels, exact-settlement rows, and research cost assumptions.",
            },
            {
                "Strength": "Weak/conditional",
                "Claim": "Convexity, VIX exposure, and short-premium sleeves help only in selected states after premium, skew, hedge cost, settlement, and risk-budget checks.",
                "Boundary": f"State- and settlement-dependent; {vix_condition}.",
            },
            {
                "Strength": "No claim",
                "Claim": "Live alpha, production tradability, executable capacity, broker-realistic fills, or live margin parity.",
                "Boundary": "No broker-routed orders, live fills, live margin preview, or broker reconciliation.",
            },
        ]
    )


def performance_summary_row(
    name: str,
    pr: pd.Series,
    benchmark: pd.Series | None = None,
    *,
    pred_realized_vol: float = np.nan,
    gross_nav: float = np.nan,
    net_nav: float = np.nan,
    n_boot: int = 1000,
) -> dict[str, object]:
    """One performance-table row with correctly ordered bootstrap CIs.

    ``block_bootstrap_metric_ci`` returns (point, lo, hi); the CI columns are
    populated from indices 1 and 2 (a previous version wrote (point, lo) into
    the lo/hi columns for the gross and equity-benchmark rows).  The Sortino
    CI, previously computed and discarded, is emitted as columns.
    """

    pr = pd.Series(pr).dropna()
    st = performance_stats(pr, PERIODS_PER_YEAR, benchmark_returns=benchmark)
    ci = block_bootstrap_metric_ci(pr, "sharpe", BootstrapConfig(n_boot=n_boot))
    sortino_ci = block_bootstrap_metric_ci(pr, "sortino", BootstrapConfig(n_boot=n_boot))
    return {
        "Strategy": name,
        "Ann. return": st["ann_return"],
        "Ann. vol": st["ann_vol"],
        "Sharpe": st["sharpe"],
        "Downside ann. dev": st["downside_ann_dev"],
        "Sortino": st["sortino"],
        "Max drawdown": st["max_drawdown"],
        "Worst month": float(pr.min()) if len(pr) else np.nan,
        "Calmar": st["calmar"],
        "Omega": st["omega"],
        "Info. ratio": st["information_ratio"],
        "SR 90\\% CI lo": ci[1],
        "SR 90\\% CI hi": ci[2],
        "Sortino 90\\% CI lo": sortino_ci[1],
        "Sortino 90\\% CI hi": sortino_ci[2],
        "Pred./realized vol": pred_realized_vol,
        "Gross NAV": gross_nav,
        "Net NAV": net_nav,
    }


def zero_imputation_diagnostics(
    raw_returns: pd.DataFrame,
    strategies: dict[str, pd.Series],
    train_end: pd.Timestamp = TRAIN_END,
) -> pd.DataFrame:
    """Per-strategy share of zero-imputed (missing) bucket-month cells.

    The pipeline zero-fills missing option bucket months before computing
    strategy returns; this diagnostic measures and exposes the imputed share
    without changing the imputation itself (cached tables depend on it).
    """

    test = raw_returns.loc[raw_returns.index > pd.Timestamp(train_end)]
    rows = []
    for name, weights in strategies.items():
        active = [c for c in weights[weights.abs() > 1e-14].index if c in test.columns]
        cells = test[active]
        total = int(cells.shape[0] * cells.shape[1])
        missing = int(cells.isna().sum().sum())
        rows.append(
            {
                "Strategy": name,
                "Test months": int(len(test)),
                "Active assets": int(len(active)),
                "Imputed cells": missing,
                "Total cells": total,
                "Imputed cell share": (missing / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


@dataclasses.dataclass
class RobustnessContext:
    full_panel: pd.DataFrame
    equity_reps: pd.DataFrame
    equity_returns: pd.DataFrame
    equity_detail: pd.DataFrame
    reps: pd.DataFrame
    returns: pd.DataFrame
    return_detail: pd.DataFrame
    has_vix: bool
    universe: list[str]
    spec: pd.DataFrame
    model: OptionOnlyMarkowitzModel
    residuals: pd.DataFrame
    cost_config: ResearchCostConfig
    cost_inputs: pd.DataFrame
    equity_spec: pd.DataFrame
    equity_model: OptionOnlyMarkowitzModel
    strategies: dict[str, pd.Series]
    equity_benchmarks: dict[str, pd.Series]
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    underlying_returns: pd.DataFrame
    vol_shocks: pd.DataFrame
    train_under: pd.DataFrame
    test_under: pd.DataFrame
    ret_frame: pd.DataFrame
    full_spread_net_frame: pd.DataFrame
    net_scenario_returns: pd.DataFrame
    vix_state: pd.DataFrame
    vix_level: pd.Series
    cv_context_consistency: pd.DataFrame


def _vix_level_from_state(vix_state: pd.DataFrame, dates: pd.Index) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if "VIX" in vix_state:
        return pd.to_numeric(vix_state["VIX"], errors="coerce").reindex(idx)
    if "VX_FRONT" in vix_state:
        return pd.to_numeric(vix_state["VX_FRONT"], errors="coerce").reindex(idx)
    return pd.Series(np.nan, index=idx, name="VIX")


def _iv_level_panel(reps: pd.DataFrame, universe: Sequence[str]) -> pd.DataFrame:
    frame = reps.copy()
    if frame.empty or "iv_proxy" not in frame:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=list(universe))
    frame["snap_date"] = pd.to_datetime(frame["snap_date"])
    bucket = frame.get("moneyness_bucket", pd.Series("", index=frame.index)).astype(str)
    atm = frame[bucket.isin(["atm", "vix_atm"])].copy()
    if atm.empty:
        atm = frame.copy()
    iv = (
        atm.groupby(["snap_date", "underlying"])["iv_proxy"]
        .median()
        .unstack("underlying")
        .sort_index()
        .reindex(columns=list(universe))
    )
    return iv.replace([np.inf, -np.inf], np.nan)


def _strategy_sharpe_rows(
    returns: pd.DataFrame,
    *,
    basis: str = "gross",
    universe_family: str = "realized",
) -> pd.DataFrame:
    rows = []
    for strategy in returns.columns:
        stats = performance_stats(pd.Series(returns[strategy]).dropna(), PERIODS_PER_YEAR)
        rows.append(
            {
                "strategy": strategy,
                "basis": basis,
                "universe_family": universe_family,
                "sharpe": float(stats.get("sharpe", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _context_weight_consistency(strategies: dict[str, pd.Series], contracts: Sequence[str]) -> pd.DataFrame:
    path = ART_DIR / "strategy_weights.csv"
    rows: list[dict[str, object]] = []
    contract_index = pd.Index(contracts)
    if not path.exists():
        for strategy, weights in strategies.items():
            rows.append(
                {
                    "strategy": strategy,
                    "status": "reference_file_missing",
                    "max_abs_diff": np.nan,
                    "n_context_weights": int(pd.Series(weights).abs().gt(1e-14).sum()),
                    "n_reference_weights": 0,
                }
            )
        out = pd.DataFrame(rows)
        out.to_csv(ART_DIR / "cv_context_consistency.csv", index=False)
        return out

    reference = pd.read_csv(path, index_col=0)
    for strategy, weights in strategies.items():
        context_weights = pd.to_numeric(pd.Series(weights), errors="coerce").reindex(contract_index).fillna(0.0)
        if strategy not in reference.columns:
            rows.append(
                {
                    "strategy": strategy,
                    "status": "strategy_missing_in_reference",
                    "max_abs_diff": np.nan,
                    "n_context_weights": int(context_weights.abs().gt(1e-14).sum()),
                    "n_reference_weights": 0,
                }
            )
            continue
        ref_weights = pd.to_numeric(reference[strategy], errors="coerce").reindex(contract_index).fillna(0.0)
        rows.append(
            {
                "strategy": strategy,
                "status": "ok",
                "max_abs_diff": float((context_weights - ref_weights).abs().max()) if len(contract_index) else 0.0,
                "n_context_weights": int(context_weights.abs().gt(1e-14).sum()),
                "n_reference_weights": int(ref_weights.abs().gt(1e-14).sum()),
            }
        )
    for strategy in reference.columns:
        if strategy not in strategies:
            rows.append(
                {
                    "strategy": strategy,
                    "status": "strategy_missing_in_context",
                    "max_abs_diff": np.nan,
                    "n_context_weights": 0,
                    "n_reference_weights": int(pd.to_numeric(reference[strategy], errors="coerce").fillna(0.0).abs().gt(1e-14).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ART_DIR / "cv_context_consistency.csv", index=False)
    return out


def _robustness_context() -> RobustnessContext:
    _ensure_dirs()
    full_panel, equity_reps, equity_returns = load_bucket_panel()
    raw_close = load_raw_close_panel(PRIMARY_UNDERLYINGS)
    _, daily_split_factors, _split_events = split_adjusted_spot_panel(raw_close)
    _, equity_detail = build_expiry_proxy_return_panel(equity_reps, raw_close, daily_split_factors)
    equity_detail = equity_detail[equity_detail["asset_id"].isin(equity_returns.columns)].copy()
    equity_detail["asset_class"] = "equity_option"
    equity_reps["asset_class"] = equity_reps.get("asset_class", "equity_option")

    vix_full, vix_reps, vix_returns, vix_detail, _vix_audit = build_vix_option_bucket_panel(
        sorted(pd.to_datetime(equity_reps["snap_date"].dropna().unique())), ROOT, MIN_OPTION_MARK
    )
    has_vix = not vix_returns.empty and not vix_reps.empty
    if has_vix:
        reps = pd.concat([equity_reps, vix_reps], ignore_index=True, sort=False)
        returns = equity_returns.join(vix_returns, how="outer").sort_index()
        return_detail = pd.concat([equity_detail, vix_detail], ignore_index=True, sort=False)
    else:
        reps = equity_reps.copy()
        returns = equity_returns.copy()
        return_detail = equity_detail.copy()
        _ = vix_full

    spec = representative_specs(reps, returns)
    returns = returns.reindex(columns=spec.index).dropna(how="all")
    universe = PRIMARY_UNDERLYINGS + ([VIX_FACTOR] if VIX_FACTOR in set(spec["underlying"].astype(str)) else [])
    model, residuals = make_model(spec, returns, reps, universe)

    cost_config = ResearchCostConfig()
    spread_surface = (
        load_cbbo_spread_surface(ROOT, cost_config.cbbo_spread_surface_path)
        if cost_config.use_cbbo_spread_surface
        else None
    )
    cost_inputs = build_cost_input_ledger(reps, return_detail, ROOT, cost_config, spread_surface=spread_surface)

    equity_spec = representative_specs(equity_reps, equity_returns)
    equity_returns = equity_returns.reindex(columns=equity_spec.index).dropna(how="all")
    equity_model, _ = make_model(equity_spec, equity_returns, equity_reps, PRIMARY_UNDERLYINGS)
    equity_strategy = strategy_weights(equity_model, PRIMARY_UNDERLYINGS)["Greek Markowitz"]
    combined_strategies = strategy_weights(model, universe)
    train_scenarios = returns.loc[:TRAIN_END, model.contracts].fillna(0.0)
    entry_costs, _entry_cost_diag = derive_entry_cost_series(
        cost_inputs, list(model.contracts), train_end=TRAIN_END, config=cost_config
    )
    sortino_weights, _sortino_diag = _sortino_weights_with_guard(
        model, train_scenarios, entry_costs, "cost_aware_sortino"
    )
    strategies = {
        "Equity-option Greek Markowitz": equity_strategy,
        "Greek Markowitz + VIX": combined_strategies["Greek Markowitz"],
        "Beta/delta-neutral + VIX": combined_strategies["Delta neutral"],
        "Cost-aware Sortino + VIX": sortino_weights,
        "Equal premium": combined_strategies["Equal premium"],
        "Equal risk": combined_strategies["Equal risk"],
    }
    if has_vix:
        strategies["VIX hedge sleeve"] = vix_hedge_sleeve_weights(model)

    test_returns = returns.loc[returns.index > TRAIN_END, model.contracts].fillna(0.0)
    train_returns = train_scenarios.copy()
    underlying_returns, vol_shocks = factor_panels(reps, universe)
    train_under = underlying_returns.loc[:TRAIN_END].reindex(columns=PRIMARY_UNDERLYINGS).fillna(0.0)
    test_under = underlying_returns.loc[test_returns.index].reindex(columns=PRIMARY_UNDERLYINGS).fillna(0.0)
    equity_benchmarks = {
        "Delta-matched equities": option_delta_by_underlying(model, strategies["Greek Markowitz + VIX"]),
        "Underlying Markowitz": equity_tangency_weights(train_under),
    }

    ret_frame = pd.DataFrame(index=test_returns.index)
    for name, weights in strategies.items():
        ret_frame[name] = model.portfolio_return_series(test_returns, weights)
    for name, weights in equity_benchmarks.items():
        ret_frame[name] = pd.Series(
            test_under.to_numpy(float) @ weights.reindex(PRIMARY_UNDERLYINGS).fillna(0.0).to_numpy(float),
            index=test_under.index,
        )

    option_gross_frame = ret_frame[[name for name in strategies if name in ret_frame.columns]].copy()
    cost_result = build_execution_cost_scenarios(
        option_gross_frame,
        strategies,
        cost_inputs,
        config=ExecutionCostScenarioConfig(nav_for_capacity=cost_config.nav_for_capacity),
        scenarios=("full_spread",),
    )
    net_scenario_returns = cost_result.net
    full_spread_net_frame = ret_frame.copy()
    for strategy in strategies:
        scenario_col = f"{strategy}::full_spread"
        if scenario_col in net_scenario_returns:
            full_spread_net_frame[strategy] = net_scenario_returns[scenario_col]

    vix_state = vix_state_panel(returns.index, ROOT)
    vix_level = _vix_level_from_state(vix_state, returns.index)
    consistency = _context_weight_consistency(strategies, model.contracts)

    return RobustnessContext(
        full_panel=full_panel,
        equity_reps=equity_reps,
        equity_returns=equity_returns,
        equity_detail=equity_detail,
        reps=reps,
        returns=returns,
        return_detail=return_detail,
        has_vix=has_vix,
        universe=universe,
        spec=spec,
        model=model,
        residuals=residuals,
        cost_config=cost_config,
        cost_inputs=cost_inputs,
        equity_spec=equity_spec,
        equity_model=equity_model,
        strategies=strategies,
        equity_benchmarks=equity_benchmarks,
        train_returns=train_returns,
        test_returns=test_returns,
        underlying_returns=underlying_returns,
        vol_shocks=vol_shocks,
        train_under=train_under,
        test_under=test_under,
        ret_frame=ret_frame,
        full_spread_net_frame=full_spread_net_frame,
        net_scenario_returns=net_scenario_returns,
        vix_state=vix_state,
        vix_level=vix_level,
        cv_context_consistency=consistency,
    )


def _basis_label(value: object) -> str:
    text = str(value)
    if text == "full_spread_post_cost":
        return "Full Spread"
    if text == "gross":
        return "Gross"
    return text.replace("_", " ").title()


def _cv_fold_performance_table(fold_ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per blocked k-fold with a compact ``gross / net`` Sharpe cell
    per strategy — a long fold x strategy layout is too tall for one page."""

    if fold_ledger.empty:
        return pd.DataFrame(columns=["Fold"])
    ledger = fold_ledger.copy()
    ledger = ledger[ledger.get("scheme", pd.Series(index=ledger.index, dtype=object)).astype(str).eq("kfold")]
    if ledger.empty:
        return pd.DataFrame(columns=["Fold"])
    ledger = ledger[ledger["strategy"].isin(CV_STRATEGIES)].copy()
    gross = ledger[ledger["basis"].eq("gross")].set_index(["fold_id", "strategy"])["sharpe"]
    net = ledger[ledger["basis"].eq("full_spread_post_cost")].set_index(["fold_id", "strategy"])["sharpe"]

    def _cell(fold_id: str, strategy: str) -> str:
        g = gross.get((fold_id, strategy), np.nan)
        n = net.get((fold_id, strategy), np.nan)
        g_txt = f"{g:.2f}" if np.isfinite(g) else "--"
        n_txt = f"{n:.2f}" if np.isfinite(n) else "--"
        return f"{g_txt} / {n_txt}"

    fold_ids = sorted(ledger["fold_id"].astype(str).unique())
    strategies = [s for s in CV_STRATEGIES if s in set(ledger["strategy"])]
    rows = [{"Fold": fold_id, **{s: _cell(fold_id, s) for s in strategies}} for fold_id in fold_ids]
    return pd.DataFrame(rows, columns=["Fold", *strategies])


def _cv_cpcv_distribution_table(
    ret_frame: pd.DataFrame,
    full_spread_net_frame: pd.DataFrame,
    path_metrics: pd.DataFrame,
    pbo_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Strategy",
        "Realized Gross Sharpe",
        "CPCV Gross P05",
        "CPCV Gross P50",
        "CPCV Gross P95",
        "Realized Full Spread Sharpe",
        "CPCV Full Spread P05",
        "CPCV Full Spread P50",
        "CPCV Full Spread P95",
        "PBO Gross",
        "PBO Full Spread",
    ]
    strategies = sorted(set(ret_frame.columns) | set(path_metrics.get("strategy", pd.Series(dtype=object)).astype(str)))
    pbo_lookup = {}
    if not pbo_summary.empty and {"Basis", "PBO"}.issubset(pbo_summary.columns):
        pbo_lookup = dict(zip(pbo_summary["Basis"].astype(str), pd.to_numeric(pbo_summary["PBO"], errors="coerce")))
    rows = []
    for strategy in strategies:
        row: dict[str, object] = {"Strategy": strategy}
        if strategy in ret_frame:
            row["Realized Gross Sharpe"] = performance_stats(ret_frame[strategy].dropna(), PERIODS_PER_YEAR)["sharpe"]
        else:
            row["Realized Gross Sharpe"] = np.nan
        if strategy in full_spread_net_frame:
            row["Realized Full Spread Sharpe"] = performance_stats(full_spread_net_frame[strategy].dropna(), PERIODS_PER_YEAR)["sharpe"]
        else:
            row["Realized Full Spread Sharpe"] = np.nan
        for basis, prefix in [("gross", "CPCV Gross"), ("full_spread_post_cost", "CPCV Full Spread")]:
            sub = path_metrics[
                path_metrics.get("strategy", pd.Series(dtype=object)).astype(str).eq(strategy)
                & path_metrics.get("basis", pd.Series(dtype=object)).astype(str).eq(basis)
                & path_metrics.get("status", pd.Series("complete", index=path_metrics.index)).astype(str).eq("complete")
            ]
            sharpe = pd.to_numeric(sub.get("sharpe", pd.Series(dtype=float)), errors="coerce")
            row[f"{prefix} P05"] = float(sharpe.quantile(0.05)) if sharpe.notna().any() else np.nan
            row[f"{prefix} P50"] = float(sharpe.quantile(0.50)) if sharpe.notna().any() else np.nan
            row[f"{prefix} P95"] = float(sharpe.quantile(0.95)) if sharpe.notna().any() else np.nan
        row["PBO Gross"] = pbo_lookup.get("gross", np.nan)
        row["PBO Full Spread"] = pbo_lookup.get("full_spread_post_cost", np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _write_cv_outputs(
    results: CVResults,
    path_month_returns: pd.DataFrame,
    path_metrics: pd.DataFrame,
    pbo_summary: pd.DataFrame,
    regime_performance: pd.DataFrame,
    ret_frame: pd.DataFrame,
    full_spread_net_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    results.fold_schedule.to_csv(ART_DIR / "cv_fold_schedule.csv", index=False)
    results.fold_ledger.to_csv(ART_DIR / "cv_fold_ledger.csv", index=False)
    results.split_is_oos.to_csv(ART_DIR / "cv_split_is_oos.csv", index=False)
    path_metrics.to_csv(ART_DIR / "cv_cpcv_path_metrics.csv", index=False)
    path_month_returns.to_csv(ART_DIR / "cv_cpcv_path_month_returns.csv", index=False)
    pbo_summary.to_csv(ART_DIR / "cv_pbo_summary.csv", index=False)
    regime_performance.to_csv(ART_DIR / "cv_regime_performance.csv", index=False)
    results.runtime_log.to_csv(ART_DIR / "cv_runtime_log.csv", index=False)

    fold_table = _cv_fold_performance_table(results.fold_ledger)
    distribution_table = _cv_cpcv_distribution_table(ret_frame, full_spread_net_frame, path_metrics, pbo_summary)
    regime_table = regime_performance.copy()

    _write_latex_table(_escape_object_columns(fold_table), TABLE_DIR / "cv_fold_performance.tex")
    _write_latex_table(_escape_object_columns(distribution_table), TABLE_DIR / "cv_cpcv_distribution.tex")
    _write_latex_table(_escape_object_columns(regime_table), TABLE_DIR / "cv_regime_performance.tex")
    return {
        "fold_performance": fold_table,
        "cpcv_distribution": distribution_table,
        "regime_performance": regime_table,
    }


def _with_full_spread_split_rows(split_is_oos: pd.DataFrame, fold_ledger: pd.DataFrame) -> pd.DataFrame:
    if split_is_oos.empty or fold_ledger.empty:
        return split_is_oos
    split = split_is_oos.copy()
    if split.get("basis", pd.Series(dtype=object)).astype(str).eq("full_spread_post_cost").any():
        return split
    net = fold_ledger[fold_ledger.get("basis", pd.Series(index=fold_ledger.index, dtype=object)).astype(str).eq("full_spread_post_cost")].copy()
    if net.empty or "sharpe" not in net:
        return split
    key_cols = [col for col in ["fold_id", "scheme", "strategy"] if col in split.columns and col in net.columns]
    if "strategy" not in key_cols:
        return split
    net_lookup = net.set_index(key_cols)["sharpe"].to_dict()
    rows = []
    gross = split[split.get("basis", pd.Series(index=split.index, dtype=object)).astype(str).eq("gross")]
    for _, row in gross.iterrows():
        key = tuple(row[col] for col in key_cols)
        if key not in net_lookup:
            continue
        new_row = row.copy()
        new_row["basis"] = "full_spread_post_cost"
        new_row["oos_sharpe"] = net_lookup[key]
        rows.append(new_row)
    if not rows:
        return split
    return pd.concat([split, pd.DataFrame(rows)], ignore_index=True, sort=False)


def run_cv_stage(ctx: RobustnessContext, cv_config: CVConfig) -> dict[str, object]:
    folds = build_folds(ctx.returns.index, cv_config, "kfold") + build_folds(ctx.returns.index, cv_config, "cpcv")
    results = evaluate_folds(
        folds,
        ctx.returns,
        ctx.reps,
        ctx.equity_returns,
        ctx.equity_reps,
        ctx.universe,
        PRIMARY_UNDERLYINGS,
        ctx.cost_inputs,
        spec_builder=representative_specs,
        model_factory=make_model,
        weights_builder=strategy_weights,
        cost_scenario_builder=build_execution_cost_scenarios,
        scenario_config=ExecutionCostScenarioConfig(nav_for_capacity=ctx.cost_config.nav_for_capacity),
        vix_sleeve_builder=vix_hedge_sleeve_weights,
        # Equity benchmarks (Delta-matched equities, Underlying Markowitz) hold
        # UNDERLYING weights; the fold evaluator scores option-contract weights,
        # so per-fold benchmark rows would be all-NaN.  They stay headline-only.
        config=cv_config,
    )
    results.split_is_oos = _with_full_spread_split_rows(results.split_is_oos, results.fold_ledger)
    path_month_returns, path_metrics = assemble_cpcv_paths(results.test_month_returns, cv_config)
    pbo_summary = pd.concat(
        [
            probability_of_backtest_overfitting(results.split_is_oos, "gross"),
            probability_of_backtest_overfitting(results.split_is_oos, "full_spread_post_cost"),
        ],
        ignore_index=True,
        sort=False,
    )
    regime_dates = (
        pd.DatetimeIndex(path_month_returns["return_date"].dropna().unique()).sort_values()
        if "return_date" in path_month_returns and not path_month_returns.empty
        else pd.DatetimeIndex(ctx.ret_frame.index)
    )
    regime_tags = tag_regimes(regime_dates, ctx.vix_level.reindex(regime_dates), EVENT_WINDOWS)
    regime_performance = cpcv_regime_table(path_month_returns, regime_tags, grouped_metric_inference)
    tables = _write_cv_outputs(
        results,
        path_month_returns,
        path_metrics,
        pbo_summary,
        regime_performance,
        ctx.ret_frame,
        ctx.full_spread_net_frame,
    )
    realized = _realized_gross_sharpes_from_returns(ctx.ret_frame)
    plot_cpcv_sharpe_distribution(path_metrics, realized, FIG_DIR / "cv_cpcv_sharpe_distribution.pdf")
    plot_fold_sharpe_heatmap(results.fold_ledger, results.fold_schedule, FIG_DIR / "cv_fold_sharpe_heatmap.pdf")
    return {
        "results": results,
        "path_month_returns": path_month_returns,
        "path_metrics": path_metrics,
        "pbo_summary": pbo_summary,
        "regime_performance": regime_performance,
        "tables": tables,
    }


def _refit_summary(refit_paths: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Strategy",
        "Paths",
        "OK Paths",
        "Realized Sharpe",
        "P05 Sharpe",
        "P50 Sharpe",
        "P95 Sharpe",
        "P50 Max Drawdown",
        "Median Gross NAV",
        "Error Paths",
    ]
    if refit_paths.empty:
        return pd.DataFrame(columns=columns)
    realized_lookup = dict(zip(realized["strategy"], pd.to_numeric(realized["sharpe"], errors="coerce"))) if not realized.empty else {}
    rows = []
    for strategy, group in refit_paths.groupby("strategy", dropna=False):
        ok = group[group.get("status", pd.Series("ok", index=group.index)).astype(str).eq("ok")]
        sharpe = pd.to_numeric(ok.get("sharpe", pd.Series(dtype=float)), errors="coerce")
        max_dd = pd.to_numeric(ok.get("max_drawdown", pd.Series(dtype=float)), errors="coerce")
        gross_nav = pd.to_numeric(ok.get("gross_nav", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "Strategy": strategy,
                "Paths": int(len(group)),
                "OK Paths": int(len(ok)),
                "Realized Sharpe": realized_lookup.get(strategy, np.nan),
                "P05 Sharpe": float(sharpe.quantile(0.05)) if sharpe.notna().any() else np.nan,
                "P50 Sharpe": float(sharpe.quantile(0.50)) if sharpe.notna().any() else np.nan,
                "P95 Sharpe": float(sharpe.quantile(0.95)) if sharpe.notna().any() else np.nan,
                "P50 Max Drawdown": float(max_dd.quantile(0.50)) if max_dd.notna().any() else np.nan,
                "Median Gross NAV": float(gross_nav.median()) if gross_nav.notna().any() else np.nan,
                "Error Paths": int(len(group) - len(ok)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("Strategy").reset_index(drop=True)


def _mc_refit_weights(model: OptionOnlyMarkowitzModel) -> pd.Series:
    weights = _solve_weights_with_guard(model, "mc_refit")
    weights.name = "Greek Markowitz + VIX"
    return weights


def run_mc_stage(
    ctx: RobustnessContext,
    resample_config: ResampleConfig,
    reprice_config: RepriceConfig,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(resample_config.seed)
    paths = month_index_paths(len(ctx.ret_frame), resample_config.n_paths, resample_config.block_length, rng)
    regime_tags = tag_regimes(ctx.ret_frame.index, ctx.vix_level.reindex(ctx.ret_frame.index), EVENT_WINDOWS)
    regime_labels = regime_tags.set_index("return_date")["vix_tercile"].reindex(ctx.ret_frame.index)
    stratified_paths = stratified_month_index_paths(regime_labels, resample_config.n_paths, resample_config.block_length, rng)

    fixed_frames = [
        fixed_weight_universe_distribution(
            ctx.ret_frame,
            paths,
            basis="gross",
            universe_family="resampled",
            periods_per_year=resample_config.periods_per_year,
        ),
        fixed_weight_universe_distribution(
            ctx.full_spread_net_frame,
            paths,
            basis="full_spread_post_cost",
            universe_family="resampled",
            periods_per_year=resample_config.periods_per_year,
        ),
        fixed_weight_universe_distribution(
            ctx.ret_frame,
            stratified_paths,
            basis="gross",
            universe_family="resampled_stratified",
            periods_per_year=resample_config.periods_per_year,
        ),
        fixed_weight_universe_distribution(
            ctx.full_spread_net_frame,
            stratified_paths,
            basis="full_spread_post_cost",
            universe_family="resampled_stratified",
            periods_per_year=resample_config.periods_per_year,
        ),
    ]
    fixed_paths = pd.concat(fixed_frames, ignore_index=True, sort=False)
    realized = pd.concat(
        [
            _strategy_sharpe_rows(ctx.ret_frame, basis="gross", universe_family="resampled"),
            _strategy_sharpe_rows(ctx.full_spread_net_frame, basis="full_spread_post_cost", universe_family="resampled"),
            _strategy_sharpe_rows(ctx.ret_frame, basis="gross", universe_family="resampled_stratified"),
            _strategy_sharpe_rows(ctx.full_spread_net_frame, basis="full_spread_post_cost", universe_family="resampled_stratified"),
        ],
        ignore_index=True,
        sort=False,
    )
    fixed_summary = resampled_summary(fixed_paths, realized)
    resample_assumption_table = resample_assumptions(
        resample_config,
        regime_labels.value_counts(dropna=False).to_dict(),
        {"stratification": "VIX terciles from the realized OOS return months"},
    )

    train_dates = pd.DatetimeIndex(ctx.returns.loc[ctx.returns.index <= TRAIN_END].index)
    refit_paths = refit_universe_distribution(
        ctx.returns,
        ctx.reps,
        ctx.universe,
        train_dates,
        ctx.test_returns,
        spec_builder=representative_specs,
        model_factory=make_model,
        weights_builder_single=_mc_refit_weights,
        under_ret=ctx.underlying_returns,
        vol_shocks=ctx.vol_shocks,
        config=resample_config,
    )
    realized_gross = _strategy_sharpe_rows(ctx.ret_frame, basis="gross", universe_family="realized")
    refit_summary = _refit_summary(refit_paths, realized_gross)

    state_underlyings = list(dict.fromkeys(list(ctx.universe) + ([VIX_FACTOR] if ctx.has_vix else [])))
    state_under_ret = ctx.underlying_returns.loc[:TRAIN_END].copy()
    if VIX_FACTOR in state_underlyings and VIX_FACTOR not in state_under_ret.columns and "VX_FRONT_return" in ctx.vix_state:
        state_under_ret[VIX_FACTOR] = pd.to_numeric(ctx.vix_state["VX_FRONT_return"], errors="coerce").reindex(state_under_ret.index)
    iv_levels = _iv_level_panel(ctx.reps, state_underlyings)
    state_model = fit_joint_state_model(
        state_under_ret.reindex(columns=state_underlyings).fillna(0.0),
        iv_levels.loc[:TRAIN_END].reindex(columns=state_underlyings).ffill().bfill(),
        ctx.vix_level.loc[:TRAIN_END].ffill().bfill(),
        reprice_config,
    )
    params = contract_static_params(ctx.reps, TRAIN_END)
    if not params.empty:
        params = params.loc[params.index.intersection(pd.Index(ctx.model.contracts))]

    states = simulate_state_paths(state_model, reprice_config, method="joint_garch_block")
    contract_returns = reprice_contract_returns(states, params, reprice_config)
    repriced_paths = repriced_strategy_paths(contract_returns, ctx.strategies, params.index, reprice_config)
    repriced_summary_table = repriced_summary(repriced_paths, realized_gross)

    gauss_states = simulate_state_paths(state_model, reprice_config, method="gaussian_copula")
    gauss_returns = reprice_contract_returns(gauss_states, params, reprice_config)
    repriced_paths_gauss = repriced_strategy_paths(gauss_returns, ctx.strategies, params.index, reprice_config)
    if not repriced_paths_gauss.empty:
        repriced_paths_gauss["method"] = "gaussian_copula"
    repriced_summary_gauss = repriced_summary(repriced_paths_gauss, realized_gross)

    reprice_assumption_table = reprice_assumptions(state_model, params, reprice_config, "joint_garch_block")
    comparison = universe_comparison_table(
        realized_gross,
        fixed_summary[
            fixed_summary["Basis"].astype(str).eq("gross")
            & fixed_summary["Universe Family"].astype(str).eq("resampled")
        ] if not fixed_summary.empty else fixed_summary,
        repriced_summary_table,
    )

    fixed_paths.to_csv(ART_DIR / "mc_resampled_fixed_paths.csv", index=False)
    fixed_summary.to_csv(ART_DIR / "mc_resampled_summary.csv", index=False)
    refit_paths.to_csv(ART_DIR / "mc_refit_paths.csv", index=False)
    refit_summary.to_csv(ART_DIR / "mc_refit_summary.csv", index=False)
    resample_assumption_table.to_csv(ART_DIR / "mc_resampled_assumptions.csv", index=False)
    repriced_paths.to_csv(ART_DIR / "mc_repriced_paths.csv", index=False)
    repriced_summary_table.to_csv(ART_DIR / "mc_repriced_summary.csv", index=False)
    repriced_paths_gauss.to_csv(ART_DIR / "mc_repriced_paths_gauss_copula.csv", index=False)
    repriced_summary_gauss.to_csv(ART_DIR / "mc_repriced_summary_gauss_copula.csv", index=False)
    reprice_assumption_table.to_csv(ART_DIR / "mc_repriced_assumptions.csv", index=False)
    comparison.to_csv(ART_DIR / "mc_universe_comparison.csv", index=False)

    _write_latex_table(_escape_object_columns(fixed_summary), TABLE_DIR / "mc_resampled_universes.tex")
    _write_latex_table(_escape_object_columns(refit_summary), TABLE_DIR / "mc_refit_stability.tex")
    _write_latex_table(_escape_object_columns(repriced_summary_table), TABLE_DIR / "mc_repriced_universes.tex")
    _write_latex_table(_escape_object_columns(comparison), TABLE_DIR / "mc_universe_comparison.tex")
    _write_latex_table(_escape_object_columns(reprice_assumption_table), TABLE_DIR / "mc_repriced_assumptions.tex")

    plot_mc_universe_distributions(
        _attach_realized_sharpes(fixed_paths, realized_gross),
        _attach_realized_sharpes(repriced_paths, realized_gross),
        _attach_realized_sharpes(refit_paths, realized_gross),
        FIG_DIR / "mc_universe_sharpe_distributions.pdf",
    )

    return {
        "resampled_paths": fixed_paths,
        "resampled_summary": fixed_summary,
        "refit_paths": refit_paths,
        "refit_summary": refit_summary,
        "resampled_assumptions": resample_assumption_table,
        "repriced_paths": repriced_paths,
        "repriced_summary": repriced_summary_table,
        "repriced_paths_gauss": repriced_paths_gauss,
        "repriced_summary_gauss": repriced_summary_gauss,
        "repriced_assumptions": reprice_assumption_table,
        "universe_comparison": comparison,
    }


def _json_safe_object(value: object) -> object:
    if isinstance(value, pd.DataFrame):
        return _json_safe_object(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe_object(value.to_dict())
    if isinstance(value, np.ndarray):
        return _json_safe_object(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe_object(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_object(v) for v in value]
    return _json_safe_value(value)


def _write_distributional_robustness_summary(summary: dict[str, object]) -> None:
    (TABLE_DIR / "distributional_robustness_summary.json").write_text(
        json.dumps(_json_safe_object(summary), indent=2),
        encoding="utf-8",
    )


def _read_required_figure_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing robustness figure input: {path.relative_to(PAPER)}")
    return pd.read_csv(path)


def run_robustness_figures() -> dict[str, Path]:
    """Regenerate robustness figures from persisted CV/MC artifacts only."""

    _ensure_dirs()
    required_inputs = [
        ART_DIR / "cv_cpcv_path_metrics.csv",
        ART_DIR / "cv_fold_ledger.csv",
        ART_DIR / "cv_fold_schedule.csv",
        ART_DIR / "mc_resampled_fixed_paths.csv",
        ART_DIR / "mc_repriced_paths.csv",
        ART_DIR / "mc_refit_paths.csv",
        TABLE_DIR / "empirical_summary.json",
    ]
    missing = [path.relative_to(PAPER).as_posix() for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing robustness figure inputs: "
            + ", ".join(missing)
            + ". Run --stage robustness, --stage cv/--stage mc, or the full empirical stage first."
        )

    realized = _realized_gross_sharpes_from_empirical_summary(TABLE_DIR / "empirical_summary.json")
    path_metrics = _read_required_figure_csv(ART_DIR / "cv_cpcv_path_metrics.csv")
    fold_ledger = _read_required_figure_csv(ART_DIR / "cv_fold_ledger.csv")
    fold_schedule = _read_required_figure_csv(ART_DIR / "cv_fold_schedule.csv")
    fixed_paths = _read_required_figure_csv(ART_DIR / "mc_resampled_fixed_paths.csv")
    repriced_paths = _read_required_figure_csv(ART_DIR / "mc_repriced_paths.csv")
    refit_paths = _read_required_figure_csv(ART_DIR / "mc_refit_paths.csv")

    outputs = {
        "cpcv_distribution": FIG_DIR / "cv_cpcv_sharpe_distribution.pdf",
        "fold_heatmap": FIG_DIR / "cv_fold_sharpe_heatmap.pdf",
        "mc_universes": FIG_DIR / "mc_universe_sharpe_distributions.pdf",
    }
    plot_cpcv_sharpe_distribution(path_metrics, realized, outputs["cpcv_distribution"])
    plot_fold_sharpe_heatmap(fold_ledger, fold_schedule, outputs["fold_heatmap"])
    plot_mc_universe_distributions(
        _attach_realized_sharpes(fixed_paths, realized),
        _attach_realized_sharpes(repriced_paths, realized),
        _attach_realized_sharpes(refit_paths, realized),
        outputs["mc_universes"],
    )
    return outputs


def run_robustness(
    cv_config: CVConfig | None = None,
    resample_config: ResampleConfig | None = None,
    reprice_config: RepriceConfig | None = None,
) -> dict[str, object]:
    cv_config = CVConfig() if cv_config is None else cv_config
    resample_config = ResampleConfig() if resample_config is None else resample_config
    reprice_config = RepriceConfig() if reprice_config is None else reprice_config

    runtime_seconds: dict[str, float] = {}
    started = time.monotonic()
    ctx = _robustness_context()
    runtime_seconds["context"] = time.monotonic() - started

    started = time.monotonic()
    cv = run_cv_stage(ctx, cv_config)
    runtime_seconds["cv"] = time.monotonic() - started

    started = time.monotonic()
    mc = run_mc_stage(ctx, resample_config, reprice_config)
    runtime_seconds["mc"] = time.monotonic() - started

    summary = {
        "cv_config": dataclasses.asdict(cv_config),
        "cv_fold_schedule": cv["results"].fold_schedule.to_dict(orient="records"),
        "cv_fold_ledger": cv["results"].fold_ledger.to_dict(orient="records"),
        "cv_cpcv_path_metrics": cv["path_metrics"].to_dict(orient="records"),
        "cv_pbo": cv["pbo_summary"].to_dict(orient="records"),
        "cv_regime_performance": cv["regime_performance"].to_dict(orient="records"),
        "cv_context_consistency": ctx.cv_context_consistency.to_dict(orient="records"),
        "mc_resampled_summary": mc["resampled_summary"].to_dict(orient="records"),
        "mc_refit_summary": mc["refit_summary"].to_dict(orient="records"),
        "mc_repriced_summary": mc["repriced_summary"].to_dict(orient="records"),
        "mc_repriced_assumptions": mc["repriced_assumptions"].to_dict(orient="records"),
        "mc_universe_comparison": mc["universe_comparison"].to_dict(orient="records"),
        "runtime_seconds": runtime_seconds,
        "seeds": {
            "cv": cv_config.seed,
            "resample": resample_config.seed,
            "refit": resample_config.refit_seed,
            "reprice": reprice_config.seed,
        },
    }
    _write_distributional_robustness_summary(summary)
    _write_hash_manifest()
    return summary


def run_all() -> dict[str, object]:
    _ensure_dirs()
    full_panel, equity_reps, equity_returns = load_bucket_panel()
    raw_close = load_raw_close_panel(PRIMARY_UNDERLYINGS)
    _, daily_split_factors, split_events = split_adjusted_spot_panel(raw_close)
    _, equity_detail = build_expiry_proxy_return_panel(equity_reps, raw_close, daily_split_factors)
    equity_detail = equity_detail[equity_detail["asset_id"].isin(equity_returns.columns)].copy()
    equity_detail["asset_class"] = "equity_option"
    equity_reps["asset_class"] = equity_reps.get("asset_class", "equity_option")

    vix_full, vix_reps, vix_returns, vix_detail, vix_audit = build_vix_option_bucket_panel(
        sorted(pd.to_datetime(equity_reps["snap_date"].dropna().unique())), ROOT, MIN_OPTION_MARK
    )
    has_vix = not vix_returns.empty and not vix_reps.empty
    if has_vix:
        reps = pd.concat([equity_reps, vix_reps], ignore_index=True, sort=False)
        returns = equity_returns.join(vix_returns, how="outer").sort_index()
        return_detail = pd.concat([equity_detail, vix_detail], ignore_index=True, sort=False)
    else:
        reps = equity_reps.copy()
        returns = equity_returns.copy()
        return_detail = equity_detail.copy()

    spec = representative_specs(reps, returns)
    returns = returns.reindex(columns=spec.index).dropna(how="all")
    universe = PRIMARY_UNDERLYINGS + ([VIX_FACTOR] if VIX_FACTOR in set(spec["underlying"].astype(str)) else [])
    model, residuals = make_model(spec, returns, reps, universe)
    cost_config = ResearchCostConfig()
    spread_surface = (
        load_cbbo_spread_surface(ROOT, cost_config.cbbo_spread_surface_path)
        if cost_config.use_cbbo_spread_surface
        else None
    )
    cost_inputs = build_cost_input_ledger(reps, return_detail, ROOT, cost_config, spread_surface=spread_surface)
    cost_input_spread_source_coverage = cost_input_spread_source_coverage_table(cost_inputs)

    equity_spec = representative_specs(equity_reps, equity_returns)
    equity_returns = equity_returns.reindex(columns=equity_spec.index).dropna(how="all")
    equity_model, _ = make_model(equity_spec, equity_returns, equity_reps, PRIMARY_UNDERLYINGS)
    equity_strategy = strategy_weights(equity_model, PRIMARY_UNDERLYINGS)["Greek Markowitz"]
    combined_strategies = strategy_weights(model, universe)
    train_scenarios = returns.loc[:TRAIN_END, model.contracts].fillna(0.0)
    entry_costs, entry_cost_diag = derive_entry_cost_series(
        cost_inputs, list(model.contracts), train_end=TRAIN_END, config=cost_config
    )
    sortino_weights, sortino_diag = _sortino_weights_with_guard(
        model, train_scenarios, entry_costs, "cost_aware_sortino"
    )
    sortino_objective_diag = pd.DataFrame([{**sortino_diag, "n_train_scenarios": int(len(train_scenarios))}])
    mean_sortino_entry_cost = float(pd.to_numeric(entry_cost_diag["entry_cost"], errors="coerce").mean()) if not entry_cost_diag.empty else np.nan
    sortino_entry_cost_summary = {
        "n_assets": int(len(entry_cost_diag)),
        "n_train_observed": int(entry_cost_diag["source"].eq("train_observed").sum()) if not entry_cost_diag.empty else 0,
        "n_default_imputed": int(entry_cost_diag["source"].eq("default_imputed").sum()) if not entry_cost_diag.empty else 0,
        "mean_entry_cost": _json_safe_value(mean_sortino_entry_cost),
    }
    strategies = {
        "Equity-option Greek Markowitz": equity_strategy,
        "Greek Markowitz + VIX": combined_strategies["Greek Markowitz"],
        "Beta/delta-neutral + VIX": combined_strategies["Delta neutral"],
        "Cost-aware Sortino + VIX": sortino_weights,
        "Equal premium": combined_strategies["Equal premium"],
        "Equal risk": combined_strategies["Equal risk"],
    }
    if has_vix:
        strategies["VIX hedge sleeve"] = vix_hedge_sleeve_weights(model)

    test_returns = returns.loc[returns.index > TRAIN_END, model.contracts].fillna(0.0)
    train_returns = train_scenarios.copy()
    underlying_returns, vol_shocks = factor_panels(reps, universe)
    factor_returns = load_extended_factor_returns(returns.index)
    spy_benchmark = factor_returns[SPY_UNDERLYING].reindex(test_returns.index)
    train_under = underlying_returns.loc[:TRAIN_END].reindex(columns=PRIMARY_UNDERLYINGS).fillna(0.0)
    test_under = underlying_returns.loc[test_returns.index].reindex(columns=PRIMARY_UNDERLYINGS).fillna(0.0)

    greek_delta_weights = option_delta_by_underlying(model, strategies["Greek Markowitz + VIX"])
    underlying_weights = equity_tangency_weights(train_under)
    equity_benchmarks = {
        "Delta-matched equities": greek_delta_weights,
        "Underlying Markowitz": underlying_weights,
    }

    perf_rows = []
    ret_frame = pd.DataFrame(index=test_returns.index)
    for name, weights in strategies.items():
        pr = model.portfolio_return_series(test_returns, weights)
        ret_frame[name] = pr
        calib = model.risk_calibration(test_returns, weights)
        perf_rows.append(
            performance_summary_row(
                name,
                pr,
                spy_benchmark,
                pred_realized_vol=calib["predicted_vol"] / calib["realized_vol"] if calib["realized_vol"] > 0 else np.nan,
                gross_nav=float(weights.abs().sum()),
                net_nav=float(weights.sum()),
            )
        )
    for name, weights in equity_benchmarks.items():
        pr = pd.Series(test_under.to_numpy(float) @ weights.reindex(PRIMARY_UNDERLYINGS).fillna(0.0).to_numpy(float), index=test_under.index)
        ret_frame[name] = pr
        perf_rows.append(
            performance_summary_row(
                name,
                pr,
                spy_benchmark,
                gross_nav=float(weights.abs().sum()),
                net_nav=float(weights.sum()),
            )
        )
    perf = pd.DataFrame(perf_rows)
    perf["Return basis"] = "Gross before costs"

    option_gross_frame = ret_frame[[name for name in strategies if name in ret_frame.columns]].copy()
    net_option_frame, cost_ledger, capacity_ledger, margin_ledger, assignment_ledger = compute_strategy_cost_ledgers(
        option_gross_frame, strategies, cost_inputs, cost_config
    )
    net_ret_frame = ret_frame.copy()
    for col in net_option_frame.columns:
        net_ret_frame[col] = net_option_frame[col]

    net_rows = []
    for name in net_ret_frame.columns:
        pr = net_ret_frame[name].dropna()
        bench = spy_benchmark.reindex(pr.index)
        row = performance_summary_row(
            name,
            pr,
            bench,
            gross_nav=float(strategies[name].abs().sum()) if name in strategies else np.nan,
            net_nav=float(strategies[name].sum()) if name in strategies else np.nan,
        )
        row["Return basis"] = "Post-cost research"
        net_rows.append(row)
    perf_net = pd.DataFrame(net_rows)

    vix_source_counts = vix_detail["settlement_source"].value_counts().to_dict() if has_vix and not vix_detail.empty else {}
    vix_headline_eligible = bool(vix_source_counts) and all(str(k) == "vro_soq_exact" for k in vix_source_counts)
    def _headline_eligible(strategy: str) -> str:
        if "VIX" in strategy and not vix_headline_eligible:
            return "No: VIX settlement proxy"
        return "Yes"
    perf["Headline eligible"] = perf["Strategy"].map(_headline_eligible)
    perf_net["Headline eligible"] = perf_net["Strategy"].map(_headline_eligible)
    headline_performance = pd.concat([perf, perf_net], ignore_index=True, sort=False)

    gross_inference = strategy_metric_inference(ret_frame, spy_benchmark, config=BootstrapConfig(n_boot=1000))
    gross_inference["Return basis"] = "Gross before costs"
    net_inference = strategy_metric_inference(net_ret_frame, spy_benchmark, config=BootstrapConfig(n_boot=1000))
    net_inference["Return basis"] = "Post-cost research"
    inference = pd.concat([gross_inference, net_inference], ignore_index=True, sort=False)
    cost_diagnostics = cost_diagnostics_table(cost_ledger, capacity_ledger, margin_ledger).rename(
        columns={
            "margin_requirement_nav": "Mean margin/NAV",
            "stress_margin_nav": "Mean stress margin/NAV",
            "assignment_notional_nav": "Mean assignment notional/NAV",
        }
    )
    (
        net_scenario_returns,
        cost_scenario_ledger,
        rejected_trade_ledger,
        required_capital_ledger,
        scenario_assignment_ledger,
        required_capital_returns,
    ) = build_execution_cost_scenarios(
        option_gross_frame,
        strategies,
        cost_inputs,
        config=ExecutionCostScenarioConfig(nav_for_capacity=cost_config.nav_for_capacity),
    )
    repair_config = RepairConfig()
    repaired_result = build_execution_cost_scenarios(
        option_gross_frame,
        strategies,
        cost_inputs,
        config=ExecutionCostScenarioConfig(nav_for_capacity=cost_config.nav_for_capacity),
        repair=repair_config,
    )
    repair_diag = repair_diagnostics_table(repaired_result.repaired_rows)
    repair_comparison = execution_repair_comparison_table(
        net_scenario_returns,
        repaired_result.net,
        periods_per_year=PERIODS_PER_YEAR,
    )
    diag_cov = pd.Series(np.sqrt(np.maximum(np.diag(model.option_cov), 0.0)), index=model.contracts)
    avg_expected_cost = (
        cost_inputs.groupby("asset_id")["relative_spread"].mean().reindex(model.contracts).fillna(cost_config.default_equity_option_rel_spread)
        if not cost_inputs.empty
        else pd.Series(cost_config.default_equity_option_rel_spread, index=model.contracts)
    )
    hurdle_selection, no_trade_flags = apply_trade_hurdles(
        pd.Series(model.expected_returns, index=model.contracts),
        diag_cov,
        avg_expected_cost,
    )
    no_trade_rows = []
    no_trade_returns = pd.DataFrame(index=ret_frame.index)
    hurdle_values = sorted(hurdle_selection["hurdle"].dropna().unique()) if not hurdle_selection.empty else [0.0, 0.10, 0.25]
    for h in hurdle_values:
        passed_any = bool(hurdle_selection.loc[hurdle_selection["hurdle"].eq(h), "passed"].any()) if not hurdle_selection.empty else False
        for col in option_gross_frame.columns:
            out_col = f"{col}::h{h:.2f}"
            no_trade_returns[out_col] = option_gross_frame[col] if passed_any else 0.0
        if not passed_any:
            for dt in ret_frame.index:
                no_trade_rows.append({"return_date": dt, "hurdle": h, "reason": "cash_collateral_no_contract_passed_hurdle"})
    no_trade_periods = pd.DataFrame(no_trade_rows)
    if no_trade_periods.empty:
        no_trade_periods = pd.DataFrame(columns=["return_date", "hurdle", "reason"])
    # PIT liquidity tiers: classify only on rows observable by TRAIN_END (M8).
    tier_map = liquidity_tier_labels(cost_inputs, train_end=TRAIN_END)
    liquidity_perf, liquidity_diag = liquidity_tier_rerun_tables(returns, reps, tier_map, factor_returns)
    premia_components = getattr(model, "conditional_premia_components", None)
    if premia_components is None:
        premia_components = model.options.frame.attrs.get("conditional_premia_components")
    ablation_perf, ablation_components = forecast_ablation_tables(
        premia_components,
        test_returns,
        strategies["Greek Markowitz + VIX"],
        periods_per_year=PERIODS_PER_YEAR,
    )
    capacity_market_diag = capacity_market_impact_diagnostics(cost_scenario_ledger, rejected_trade_ledger, required_capital_ledger)
    survival = post_cost_survival_table(perf, net_scenario_returns, cost_scenario_ledger, capacity_market_diag, scenario="full_spread", periods_per_year=PERIODS_PER_YEAR)
    scenario_variants = pd.concat([ret_frame.add_suffix("::gross"), net_scenario_returns], axis=1)
    reality_check = sharpe_reality_check(scenario_variants, config=BootstrapConfig(n_boot=1000, seed=20260625))
    simulation_post_cost = pd.DataFrame(index=ret_frame.index)
    for strategy in SIMULATION_STRATEGIES:
        scenario_col = f"{strategy}::full_spread"
        if scenario_col in net_scenario_returns.columns:
            simulation_post_cost[strategy] = net_scenario_returns[scenario_col]
        elif strategy in net_ret_frame.columns:
            simulation_post_cost[strategy] = net_ret_frame[strategy]
    simulation_summary, simulation_assumptions, drawdown_breaches, simulation_paths = run_tail_path_simulations(
        {
            "Gross before costs": ret_frame[[c for c in SIMULATION_STRATEGIES if c in ret_frame.columns]],
            "Full-spread post-cost": simulation_post_cost,
        },
        strategies=SIMULATION_STRATEGIES,
        config=SimulationConfig(),
    )
    simulation_summary_table = compact_simulation_summary(simulation_summary)
    simulation_assumption_table = compact_assumptions(simulation_assumptions)
    model_variant_registry = pd.DataFrame(
        {
            "Variant": list(scenario_variants.columns),
            "Basis": ["gross" if str(c).endswith("::gross") else str(c).rsplit("::", 1)[-1] for c in scenario_variants.columns],
            "OOS start": [str(pd.Timestamp(scenario_variants.index.min()).date()) if len(scenario_variants.index) else ""] * len(scenario_variants.columns),
            "OOS end": [str(pd.Timestamp(scenario_variants.index.max()).date()) if len(scenario_variants.index) else ""] * len(scenario_variants.columns),
            "Seed": [20260625] * len(scenario_variants.columns),
        }
    )
    vix_settlement_coverage = pd.DataFrame(
        [
            {
                "Settlement source": str(k),
                "Rows": int(v),
                "Headline eligible": "yes" if str(k) == "vro_soq_exact" and vix_headline_eligible else "no",
            }
            for k, v in (vix_source_counts or {"none": 0}).items()
        ]
    )
    if has_vix and not vix_detail.empty:
        audit_cols = ["expiry", "settlement_source", "payoff_date", "payoff_raw_close", "vix_close_proxy_settlement", "exact_minus_vix_close_proxy"]
        available_cols = [c for c in audit_cols if c in vix_detail.columns]
        vix_settlement_audit = (
            vix_detail[available_cols]
            .assign(expiry=lambda x: pd.to_datetime(x["expiry"], errors="coerce").dt.strftime("%Y-%m-%d"))
            .groupby(["expiry", "settlement_source"], dropna=False)
            .agg(
                Rows=("settlement_source", "size"),
                PayoffDate=("payoff_date", "first") if "payoff_date" in available_cols else ("settlement_source", "first"),
                SettlementValue=("payoff_raw_close", "mean") if "payoff_raw_close" in available_cols else ("settlement_source", "size"),
                VixCloseProxy=("vix_close_proxy_settlement", "mean") if "vix_close_proxy_settlement" in available_cols else ("settlement_source", "size"),
                ExactMinusProxy=("exact_minus_vix_close_proxy", "mean") if "exact_minus_vix_close_proxy" in available_cols else ("settlement_source", "size"),
            )
            .reset_index()
            .rename(columns={"expiry": "Expiration", "settlement_source": "Settlement source"})
            .sort_values(["Expiration", "Settlement source"])
        )
        vix_settlement_audit["Headline eligible"] = np.where(vix_settlement_audit["Settlement source"].astype(str).eq("vro_soq_exact"), "yes", "no")
        if "PayoffDate" in vix_settlement_audit:
            vix_settlement_audit["PayoffDate"] = pd.to_datetime(vix_settlement_audit["PayoffDate"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    else:
        vix_settlement_audit = pd.DataFrame(columns=["Expiration", "Settlement source", "Rows", "Headline eligible"])
    download_audit_path = ROOT / "data/public/cboe/vro_soq/vro_soq_download_audit.csv"
    if download_audit_path.exists():
        raw_download_audit = pd.read_csv(download_audit_path)
        raw_download_audit["Year"] = pd.to_datetime(raw_download_audit["expiration"], errors="coerce").dt.year
        vix_required_settlement_download_audit = (
            raw_download_audit.groupby(["Year", "parsed_settlement", "component_status"], dropna=False)
            .agg(
                RequiredExpiries=("expiration", "size"),
                FirstExpiration=("expiration", "min"),
                LastExpiration=("expiration", "max"),
            )
            .reset_index()
            .rename(
                columns={
                    "parsed_settlement": "Exact scalar parsed",
                    "component_status": "SOQ component status",
                }
            )
            .sort_values(["Year", "Exact scalar parsed", "SOQ component status"])
        )
    else:
        raw_download_audit = pd.DataFrame()
        vix_required_settlement_download_audit = pd.DataFrame(columns=["Year", "Exact scalar parsed", "SOQ component status", "RequiredExpiries"])

    # M9: measure (do not change) the zero-imputation of missing bucket months.
    imputation_diag = zero_imputation_diagnostics(
        returns.reindex(columns=model.contracts), strategies, TRAIN_END
    )

    random_sharpes = random_feasible(model, returns)

    risk_rows = []
    for name, weights in strategies.items():
        calib_train = model.risk_calibration(train_returns, weights)
        calib_test = model.risk_calibration(test_returns, weights)
        risk_rows.append(
            {
                "Strategy": name,
                "Predicted monthly vol": calib_test["predicted_vol"],
                "Train realized vol": calib_train["realized_vol"],
                "Test realized vol": calib_test["realized_vol"],
                "Test / predicted": calib_test["realized_to_predicted"],
            }
        )
    risk = pd.DataFrame(risk_rows)

    vix_settlement = "none"
    if has_vix and not vix_detail.empty:
        vix_settlement = ", ".join(f"{k}:{v}" for k, v in vix_detail["settlement_source"].value_counts().items())
    data_summary = pd.DataFrame(
        [
            {"Item": "Raw OPRA-derived equity option rows", "Value": f"{len(full_panel):,}"},
            {"Item": "Raw VIX option rows after filters", "Value": f"{len(vix_full):,}"},
            {"Item": "Monthly snapshot dates", "Value": f"{returns.index.nunique():,}"},
            {"Item": "Training dates", "Value": f"{train_returns.index.nunique():,}"},
            {"Item": "Test dates", "Value": f"{test_returns.index.nunique():,}"},
            {"Item": "Primary equity underlyings", "Value": ", ".join(PRIMARY_UNDERLYINGS)},
            {"Item": "VIX option treatment", "Value": "VX-forward Black-76 Greeks; VIX/VVIX state only"},
            {"Item": "VIX settlement source", "Value": vix_settlement.replace("vro_soq_exact:", "exact VRO/SOQ (") + (" rows)" if vix_settlement.startswith("vro_soq_exact:") else "")},
            {"Item": "VIX headline status", "Value": "exact VRO/SOQ settlement (excluded expiries disclosed in Section 2)" if vix_headline_eligible else "diagnostic only: exact VRO/SOQ incomplete"},
            {"Item": "Post-cost layers", "Value": "executable-cost scenarios (mid, half, full spread) and conservative stress-cost layer"},
            {"Item": "Broker execution status", "Value": "not broker-executed live evidence"},
            {"Item": "Rolling option bucket assets", "Value": f"{len(model.contracts):,}"},
            {"Item": "Train/test split", "Value": f"through {TRAIN_END.date()} / after {TRAIN_END.date()}"},
        ]
    )
    approximation = pd.DataFrame(
        [
            {"Moment": "Median residual mean", "Value": float(residuals.mean().median())},
            {"Moment": "Median residual vol", "Value": float(residuals.std(ddof=1).median())},
            {"Moment": "Median absolute residual autocorr(1)", "Value": float(residuals.apply(lambda s: s.autocorr(1)).abs().median())},
            {"Moment": "Systematic covariance rank", "Value": int(np.linalg.matrix_rank(model.B @ model.factor_cov @ model.B.T))},
        ]
    )
    exposure = exposure_summary_table(model, strategies, equity_benchmarks)
    factor_regression = factor_regression_table(ret_frame, factor_returns)
    attribution = pnl_attribution_table(model, strategies, test_returns, return_detail, reps)
    regime = regime_performance_table(ret_frame, factor_returns[SPY_UNDERLYING])
    vix_regime = volatility_regime_performance_table(ret_frame, ret_frame.index)
    vix_feature_dates = pd.DatetimeIndex(sorted(pd.to_datetime(reps["snap_date"].dropna().unique()))).normalize()
    vix_feature_columns = ["atm_iv_proxy", "skew_proxy", "call_wing_premium_share", "term_slope", "n_contracts"]
    vol_regime_columns = ["strategy", "bucket", "n_months", "mean_monthly_return", "annualized_sharpe"]
    try:
        vix_features = build_vix_chain_state_features(ROOT, vix_feature_dates)
        if vix_features.empty:
            vix_features = pd.DataFrame(columns=vix_feature_columns)
        vol_regimes = vol_of_vol_regime_table(ret_frame, vix_features).reset_index()
    except Exception:
        vix_features = pd.DataFrame(columns=vix_feature_columns)
        vix_features.index.name = "snap_date"
        vol_regimes = pd.DataFrame(columns=vol_regime_columns)
    if vol_regimes.empty:
        vol_regimes = pd.DataFrame(columns=vol_regime_columns)
    extension_manifest = data_extension_manifest(ROOT, cost_config, spread_surface)
    leave_one = leave_one_out_table(reps, returns, factor_returns[SPY_UNDERLYING])
    timing_diagnostics = timing_diagnostics_table(returns, equity_detail, split_events)
    if has_vix and not vix_detail.empty:
        extra_timing = pd.DataFrame(
            [
                {"Diagnostic": "VIX option return construction", "Value": "Prior-date VIX option selection, VX-forward Greeks, expiry settlement proxy flagged"},
                {"Diagnostic": "VIX settlement source", "Value": vix_settlement},
                {"Diagnostic": "VIX first test decision date", "Value": pd.Timestamp(vix_detail[vix_detail["return_date"].gt(TRAIN_END)]["decision_date"].min()).date().isoformat() if not vix_detail[vix_detail["return_date"].gt(TRAIN_END)].empty else ""},
            ]
        )
        timing_diagnostics = pd.concat([timing_diagnostics, extra_timing], ignore_index=True)
    trading_audit = trading_data_audit_table(full_panel, equity_reps, equity_returns, equity_detail, raw_close, split_events)
    if not vix_audit.empty:
        trading_audit = pd.concat([trading_audit, vix_audit], ignore_index=True)
    rolling_oos = rolling_oos_table(returns, reps, universe)
    claim_audit = claim_audit_table(vix_headline_eligible)
    if has_vix:
        exact_pnl_status = "Supported" if vix_headline_eligible else "Not claimed"
        exact_pnl_type = "Generated empirical" if vix_headline_eligible else "Rejected overclaim"
        exact_pnl_evidence = "All VIX expiry rows use exact Cboe VRO/SOQ settlement" if vix_headline_eligible else "VIX settlement source flags include proxy or missing rows"
        claim_audit = pd.concat(
            [
                claim_audit,
                pd.DataFrame(
                    [
                        {
                            "Claim": "VIX options are modeled as volatility derivatives, not equity underlyings",
                            "Type": "Modeling convention",
                            "Evidence": "VIX option panel uses VX-forward Black-76 Greeks and VIX/VVIX state variables",
                            "Status": "Implemented",
                        },
                        {
                            "Claim": "VIX option expiry P\\&L is exact listed settlement P\\&L",
                            "Type": exact_pnl_type,
                            "Evidence": exact_pnl_evidence,
                            "Status": exact_pnl_status,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

    claim_strength = claim_strength_table(vix_headline_eligible)

    perf_main_cols = ["Return basis", "Strategy", "Ann. return", "Ann. vol", "Sharpe", "Sortino", "Calmar", "Omega", "Info. ratio"]
    factor_headline_cols = ["Strategy", "Ann. alpha", "Alpha HAC t", "$R^2$", "Residual ann. vol", "Beta SPY", "Beta VX front", "Beta dVIX", "Beta dVVIX"]
    perf_diag_cols = ["Strategy", "Downside ann. dev", "Max drawdown", "Worst month", "SR 90\\% CI lo", "SR 90\\% CI hi", "Pred./realized vol", "Gross NAV", "Net NAV"]
    exposure_premium_cols = ["Strategy", "Book", "Gross NAV", "Net NAV", "Long premium paid", "Short premium sold", "Net option premium", "Short gross"]
    exposure_greek_cols = ["Strategy", "Book", "Net delta", "Gross delta", "Net gamma", "Net vega", "Equity vega", "VIX vega", "Beta SPY proxy", "Worst stress return", "VIX option gross", "Call gross", "Put gross"]

    _write_latex_table(data_summary.map(_latex_escape), TABLE_DIR / "data_summary.tex")
    _write_latex_table(headline_performance[perf_main_cols], TABLE_DIR / "portfolio_performance.tex")
    _write_latex_table(perf[perf_diag_cols], TABLE_DIR / "portfolio_performance_diagnostics.tex")
    _write_latex_table(perf_net[perf_diag_cols], TABLE_DIR / "portfolio_performance_net_diagnostics.tex")
    _write_latex_table(survival.map(_latex_escape), TABLE_DIR / "post_cost_survival.tex")
    _write_latex_table(liquidity_perf.map(_latex_escape), TABLE_DIR / "liquidity_tier_performance.tex")
    _write_latex_table(ablation_perf.map(_latex_escape), TABLE_DIR / "forecast_ablation_performance.tex")
    _write_latex_table(reality_check.map(_latex_escape), TABLE_DIR / "reality_check_inference.tex")
    _write_latex_table(inference, TABLE_DIR / "inference_summary.tex")
    _write_latex_table(_escape_object_columns(simulation_summary_table), TABLE_DIR / "simulation_summary.tex")
    drawdown_breach_table = drawdown_breaches.rename(columns={c: str(c).replace("%", "\\%") for c in drawdown_breaches.columns})
    _write_latex_table(_escape_object_columns(drawdown_breach_table), TABLE_DIR / "drawdown_breach_rates.tex")
    _write_latex_table(_escape_object_columns(simulation_assumption_table), TABLE_DIR / "simulation_assumptions.tex")
    _write_latex_table(cost_diagnostics, TABLE_DIR / "cost_capacity_margin_diagnostics.tex")
    # CSV artifacts keep raw snake_case schemas for the verifier; the LaTeX
    # variants need underscore-free headers or tabular emits "Missing $".
    spread_source_tex = cost_input_spread_source_coverage.rename(
        columns={
            "relative_spread_source": "Spread source",
            "asset_class": "Asset class",
            "rows": "Rows",
            "mean_relative_spread": "Mean relative spread",
        }
    )
    _write_latex_table(_escape_object_columns(spread_source_tex), TABLE_DIR / "cost_input_spread_source_coverage.tex")
    sortino_diag_tex = sortino_objective_diag.rename(
        columns={
            "status": "Status",
            "method_used": "Method",
            "objective": "Objective",
            "net_mean": "Net mean",
            "gross_mean": "Gross mean",
            "entry_cost": "Entry cost",
            "downside_deviation": "Downside deviation",
            "sortino_net": "Net Sortino",
            "target": "Target",
            "n_scenarios": "Scenarios",
            "degenerate_downside_free": "Downside-free",
            "n_train_scenarios": "Train scenarios",
        }
    )
    _write_latex_table(_escape_object_columns(sortino_diag_tex), TABLE_DIR / "sortino_objective_diagnostics.tex")
    _write_latex_table(capacity_market_diag.map(_latex_escape), TABLE_DIR / "capacity_market_impact_diagnostics.tex")
    _write_latex_table(_escape_object_columns(repair_diag), TABLE_DIR / "execution_repair_diagnostics.tex")
    _write_latex_table(_escape_object_columns(repair_comparison), TABLE_DIR / "execution_repair_comparison.tex")
    _write_latex_table(vix_settlement_coverage.map(_latex_escape), TABLE_DIR / "vix_settlement_coverage.tex")
    _write_latex_table(vix_settlement_audit.map(_latex_escape), TABLE_DIR / "vix_settlement_audit.tex")
    _write_latex_table(vix_required_settlement_download_audit.map(_latex_escape), TABLE_DIR / "vix_required_settlement_download_audit.tex")
    _write_latex_table(risk, TABLE_DIR / "risk_calibration.tex")
    _write_latex_table(approximation, TABLE_DIR / "approximation_diagnostics.tex")
    _write_latex_table(timing_diagnostics.map(_latex_escape), TABLE_DIR / "timing_diagnostics.tex")
    _write_latex_table(trading_audit.map(_latex_escape), TABLE_DIR / "trading_data_audit.tex")
    _write_latex_table(exposure[exposure_premium_cols], TABLE_DIR / "exposure_summary.tex")
    _write_latex_table(exposure[exposure_greek_cols], TABLE_DIR / "greek_exposure_summary.tex")
    _write_latex_table(factor_regression[factor_headline_cols], TABLE_DIR / "factor_regression_headline.tex")
    _write_latex_table(factor_regression, TABLE_DIR / "factor_regression.tex")
    _write_latex_table(attribution, TABLE_DIR / "pnl_attribution.tex")
    _write_latex_table(regime, TABLE_DIR / "regime_performance.tex")
    _write_latex_table(vix_regime, TABLE_DIR / "vix_regime_performance.tex")
    vol_regimes_tex = vol_regimes.rename(
        columns={
            "strategy": "Strategy",
            "bucket": "Bucket",
            "n_months": "Months",
            "mean_monthly_return": "Mean monthly return",
            "annualized_sharpe": "Annualized Sharpe",
        }
    )
    _write_latex_table(_escape_object_columns(vol_regimes_tex), TABLE_DIR / "vol_of_vol_regime_performance.tex")
    _write_latex_table(leave_one, TABLE_DIR / "leave_one_out.tex")
    _write_latex_table(rolling_oos, TABLE_DIR / "rolling_oos.tex")
    _write_latex_table(claim_strength.map(_latex_escape), TABLE_DIR / "claim_strength_summary.tex")
    _write_latex_table(claim_audit, TABLE_DIR / "claim_audit.tex")

    weights = pd.DataFrame(strategies)
    weights.to_csv(ART_DIR / "strategy_weights.csv")
    pd.DataFrame(equity_benchmarks).to_csv(ART_DIR / "equity_benchmark_weights.csv")
    ret_frame.to_csv(ART_DIR / "strategy_returns.csv")
    net_ret_frame.to_csv(ART_DIR / "strategy_returns_post_cost.csv")
    net_scenario_returns.to_csv(ART_DIR / "net_strategy_returns_by_cost_scenario.csv")
    repaired_result.net.to_csv(ART_DIR / "net_strategy_returns_by_cost_scenario_repaired.csv")
    required_capital_returns.to_csv(ART_DIR / "required_capital_returns.csv")
    cost_inputs.to_csv(ART_DIR / "cost_input_ledger.csv", index=False)
    cost_input_spread_source_coverage.to_csv(ART_DIR / "cost_input_spread_source_coverage.csv", index=False)
    cost_ledger.to_csv(ART_DIR / "cost_ledger.csv", index=False)
    cost_scenario_ledger.to_csv(ART_DIR / "cost_scenario_ledger.csv", index=False)
    repaired_result.cost_ledger.to_csv(ART_DIR / "cost_scenario_ledger_repaired.csv", index=False)
    entry_cost_diag.to_csv(ART_DIR / "sortino_entry_costs.csv", index=False)
    sortino_objective_diag.to_csv(ART_DIR / "sortino_objective_diagnostics.csv", index=False)
    rejected_trade_ledger.to_csv(ART_DIR / "rejected_trade_ledger.csv", index=False)
    repaired_result.rejected.to_csv(ART_DIR / "rejected_trade_ledger_repaired.csv", index=False)
    required_capital_ledger.to_csv(ART_DIR / "required_capital_ledger.csv", index=False)
    repaired_result.capital.to_csv(ART_DIR / "required_capital_ledger_repaired.csv", index=False)
    repaired_result.repaired_rows.to_csv(ART_DIR / "repaired_trade_ledger.csv", index=False)
    repair_diag.to_csv(ART_DIR / "execution_repair_diagnostics.csv", index=False)
    repair_comparison.to_csv(ART_DIR / "execution_repair_comparison.csv", index=False)
    capacity_ledger.to_csv(ART_DIR / "capacity_ledger.csv", index=False)
    margin_ledger.to_csv(ART_DIR / "research_margin_ledger.csv", index=False)
    assignment_ledger.to_csv(ART_DIR / "assignment_risk_ledger.csv", index=False)
    scenario_assignment_ledger.to_csv(ART_DIR / "dividend_risk_filter_ledger.csv", index=False)
    capacity_market_diag.to_csv(ART_DIR / "capacity_market_impact_diagnostics.csv", index=False)
    hurdle_selection.to_csv(ART_DIR / "hurdle_selection_ledger.csv", index=False)
    no_trade_periods.to_csv(ART_DIR / "no_trade_periods.csv", index=False)
    no_trade_returns.to_csv(ART_DIR / "strategy_returns_with_no_trade_state.csv")
    imputation_diag.to_csv(ART_DIR / "zero_imputation_diagnostics.csv", index=False)
    liquidity_perf.to_csv(ART_DIR / "liquidity_tier_performance.csv", index=False)
    liquidity_diag.to_csv(ART_DIR / "liquidity_tier_diagnostics.csv", index=False)
    ablation_perf.to_csv(ART_DIR / "forecast_ablation_performance.csv", index=False)
    ablation_components.to_csv(ART_DIR / "forecast_ablation_components.csv", index=False)
    survival.to_csv(ART_DIR / "post_cost_survival.csv", index=False)
    reality_check.to_csv(ART_DIR / "reality_check_inference.csv", index=False)
    simulation_summary.to_csv(ART_DIR / "simulation_summary.csv", index=False)
    simulation_assumptions.to_csv(ART_DIR / "simulation_assumptions.csv", index=False)
    drawdown_breaches.to_csv(ART_DIR / "drawdown_breach_rates.csv", index=False)
    for key, frame in simulation_paths.items():
        frame.to_csv(ART_DIR / f"simulation_paths_{key}.csv", index=False)
    model_variant_registry.to_json(ART_DIR / "model_variant_registry.json", orient="records", indent=2)
    cost_diagnostics.to_csv(ART_DIR / "cost_capacity_margin_diagnostics.csv", index=False)
    inference.to_csv(ART_DIR / "inference_summary.csv", index=False)
    vix_settlement_coverage.to_csv(ART_DIR / "vix_settlement_coverage.csv", index=False)
    vix_settlement_audit.to_csv(ART_DIR / "vix_settlement_audit.csv", index=False)
    raw_download_audit.to_csv(ART_DIR / "vro_soq_download_audit.csv", index=False)
    vix_required_settlement_download_audit.to_csv(ART_DIR / "vix_required_settlement_download_audit.csv", index=False)
    random_sharpes.to_csv(ART_DIR / "random_feasible_sharpes.csv", index=False)
    attribution.to_csv(ART_DIR / "pnl_attribution.csv", index=False)
    factor_regression.to_csv(ART_DIR / "factor_regression.csv", index=False)
    regime.to_csv(ART_DIR / "regime_performance.csv", index=False)
    vix_regime.to_csv(ART_DIR / "vix_regime_performance.csv", index=False)
    vix_features.to_csv(ART_DIR / "vix_chain_state_features.csv")
    vol_regimes.to_csv(ART_DIR / "vol_of_vol_regime_performance.csv", index=False)
    extension_manifest.to_csv(ART_DIR / "data_extension_manifest.csv", index=False)
    leave_one.to_csv(ART_DIR / "leave_one_out.csv", index=False)
    rolling_oos.to_csv(ART_DIR / "rolling_oos.csv", index=False)
    claim_audit.to_csv(ART_DIR / "claim_audit.csv", index=False)
    return_detail.to_csv(ART_DIR / "holding_return_detail.csv", index=False)
    if has_vix:
        vix_detail.to_csv(ART_DIR / "vix_holding_return_detail.csv", index=False)
        vix_audit.to_csv(ART_DIR / "vix_data_audit.csv", index=False)
    timing_diagnostics.to_csv(ART_DIR / "timing_diagnostics.csv", index=False)
    trading_audit.to_csv(ART_DIR / "trading_data_audit.csv", index=False)
    split_events.to_csv(ART_DIR / "split_adjustments.csv", index=False)
    claim_strength.to_csv(ART_DIR / "claim_strength_summary.csv", index=False)
    premia_components = getattr(model, "conditional_premia_components", None)
    if premia_components is None:
        premia_components = model.options.frame.attrs.get("conditional_premia_components")
    if premia_components is not None:
        premia_components.to_csv(ART_DIR / "conditional_premia_components.csv")

    plot_growth(ret_frame, FIG_DIR / "portfolio_growth.pdf", columns=HEADLINE_GROWTH_STRATEGIES, title="Headline strategy growth, test window")
    plot_growth(ret_frame, FIG_DIR / "portfolio_growth_all_strategies.pdf", title="All strategy growth, test window")
    figure_visibility = _write_visibility_audit(ret_frame, FIG_DIR / "portfolio_growth_all_strategies.pdf")
    figure_visibility.to_csv(ART_DIR / "figure_visibility_audit.csv", index=False)

    rand = random_sharpes.dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(rand, bins=30, color="#D6E0DF", edgecolor="#40534C", linewidth=0.6, alpha=0.95)
    opt_label = "Greek Markowitz + VIX" if "Greek Markowitz + VIX" in set(perf["Strategy"]) else perf["Strategy"].iloc[0]
    opt_sr = float(perf.loc[perf["Strategy"].eq(opt_label), "Sharpe"].iloc[0])
    ax.axvline(opt_sr, color="#8B1E3F", linewidth=3.2, label=opt_label, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.annotate(
        "optimized",
        xy=(opt_sr, ymax * 0.82),
        xytext=(8, 0),
        textcoords="offset points",
        color="#8B1E3F",
        fontsize=8.5,
        fontweight="bold",
        va="center",
    )
    ax.set_title("Random feasible option portfolios", fontsize=11, pad=8)
    ax.set_xlabel("Out-of-sample Sharpe")
    ax.set_ylabel("Number of random feasible books")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "random_sharpe_histogram.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5.2))
    x = risk["Predicted monthly vol"]
    y = risk["Test realized vol"]
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "P", "X"]
    lim_base = max(float(x.max()), float(y.max()), 1e-6)
    for i, row in risk.iterrows():
        jitter = (i - (len(risk) - 1.0) / 2.0) * lim_base * 0.006
        ax.scatter(row["Predicted monthly vol"] + jitter, row["Test realized vol"] + jitter, s=64, color=colors[i % len(colors)], marker=markers[i % len(markers)], label=row["Strategy"], zorder=3)
    lim = lim_base * 1.15
    ax.plot([0, lim], [0, lim], color="#555", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted monthly volatility")
    ax.set_ylabel("Realized monthly volatility")
    ax.set_title("Greek covariance calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(FIG_DIR / "risk_calibration.pdf")
    plt.close()

    plot_regime_sharpes(regime, FIG_DIR / "regime_sharpes.pdf")
    plot_regime_sharpes(vix_regime, FIG_DIR / "vix_regime_sharpes.pdf", title="Performance by VIX regime") if not vix_regime.empty else None
    plot_leave_one_out(leave_one, FIG_DIR / "leave_one_out_sharpe.pdf")

    summary = {
        "data": {
            "raw_equity_rows_after_filters": int(len(full_panel)),
            "raw_vix_rows_after_filters": int(len(vix_full)),
            "snapshot_dates": int(returns.index.nunique()),
            "train_dates": int(train_returns.index.nunique()),
            "test_dates": int(test_returns.index.nunique()),
            "bucket_assets": int(len(model.contracts)),
            "primary_underlyings": PRIMARY_UNDERLYINGS,
            "vix_settlement_source": vix_settlement,
            "vix_headline_eligible": bool(vix_headline_eligible),
            "post_cost_model": "conservative research simulation",
        },
        "performance": headline_performance.to_dict(orient="records"),
        "performance_gross_only": perf.to_dict(orient="records"),
        "performance_post_cost": perf_net.to_dict(orient="records"),
        "post_cost_survival": survival.to_dict(orient="records"),
        "cost_scenario_diagnostics": capacity_market_diag.to_dict(orient="records"),
        "execution_repair_diagnostics": repair_diag.to_dict(orient="records"),
        "execution_repair_comparison": repair_comparison.to_dict(orient="records"),
        "repair_config": dataclasses.asdict(repair_config),
        "sortino_diagnostics": _json_safe_dict(sortino_diag),
        "sortino_entry_cost_summary": sortino_entry_cost_summary,
        "cost_input_spread_sources": cost_input_spread_source_coverage.to_dict(orient="records"),
        "liquidity_tier_performance": liquidity_perf.to_dict(orient="records"),
        "liquidity_tier_diagnostics": liquidity_diag.to_dict(orient="records"),
        "forecast_ablation_performance": ablation_perf.to_dict(orient="records"),
        "forecast_ablation_components": ablation_components.to_dict(orient="records"),
        "reality_check_inference": reality_check.to_dict(orient="records"),
        "simulation_summary": simulation_summary.to_dict(orient="records"),
        "simulation_assumptions": simulation_assumptions.to_dict(orient="records"),
        "drawdown_breach_rates": drawdown_breaches.to_dict(orient="records"),
        "zero_imputation_diagnostics": imputation_diag.to_dict(orient="records"),
        "hurdle_summary": {
            "hurdles": sorted([float(x) for x in hurdle_selection["hurdle"].dropna().unique()]) if not hurdle_selection.empty else [],
            "no_trade_rows": int(len(no_trade_periods)),
        },
        "inference": inference.to_dict(orient="records"),
        "cost_diagnostics": cost_diagnostics.to_dict(orient="records"),
        "vix_settlement_coverage": vix_settlement_coverage.to_dict(orient="records"),
        "vix_settlement_audit": vix_settlement_audit.to_dict(orient="records"),
        "vix_required_settlement_download_audit": vix_required_settlement_download_audit.to_dict(orient="records"),
        "risk_calibration": risk.to_dict(orient="records"),
        "approximation": approximation.to_dict(orient="records"),
        "timing_diagnostics": timing_diagnostics.to_dict(orient="records"),
        "trading_data_audit": trading_audit.to_dict(orient="records"),
        "split_adjustments": split_events.to_dict(orient="records"),
        "exposure": exposure.to_dict(orient="records"),
        "factor_regression": factor_regression.to_dict(orient="records"),
        "pnl_attribution": attribution.to_dict(orient="records"),
        "regime_performance": regime.to_dict(orient="records"),
        "vix_regime_performance": vix_regime.to_dict(orient="records"),
        "vix_chain_feature_summary": {
            "n_dates": int(len(vix_features)),
            "n_nonnull_atm_iv": int(pd.to_numeric(vix_features.get("atm_iv_proxy", pd.Series(dtype=float)), errors="coerce").notna().sum()),
        },
        "vol_of_vol_regime_performance": vol_regimes.to_dict(orient="records"),
        "data_extension_manifest": extension_manifest.to_dict(orient="records"),
        "leave_one_out": leave_one.to_dict(orient="records"),
        "rolling_oos": rolling_oos.to_dict(orient="records"),
        "claim_strength": claim_strength.to_dict(orient="records"),
        "claim_audit": claim_audit.to_dict(orient="records"),
        "figure_visibility": figure_visibility.to_dict(orient="records"),
        "random_feasible": {
            "median_sharpe": float(rand.median()) if len(rand) else np.nan,
            "p95_sharpe": float(rand.quantile(0.95)) if len(rand) else np.nan,
            "max_sharpe": float(rand.max()) if len(rand) else np.nan,
        },
    }
    (TABLE_DIR / "empirical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_hash_manifest()
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "robustness", "cv", "mc", "robustness-figures"], default="all")
    parser.add_argument("--cv-groups", type=int, default=12)
    parser.add_argument("--cv-test-groups", type=int, default=2)
    parser.add_argument("--mc-paths", type=int, default=1000)
    parser.add_argument("--mc-refit-paths", type=int, default=200)
    parser.add_argument("--mc-reprice-paths", type=int, default=1000)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.stage == "all":
        run_all()
        return
    if args.stage == "robustness-figures":
        try:
            run_robustness_figures()
        except FileNotFoundError as exc:
            parser.exit(2, f"{exc}\n")
        return

    cv_config = CVConfig(n_groups=args.cv_groups, n_test_groups=args.cv_test_groups)
    resample_config = ResampleConfig(n_paths=args.mc_paths, n_refit_paths=args.mc_refit_paths)
    reprice_config = RepriceConfig(n_paths=args.mc_reprice_paths)
    if args.stage == "robustness":
        run_robustness(cv_config, resample_config, reprice_config)
        return

    ctx = _robustness_context()
    if args.stage == "cv":
        run_cv_stage(ctx, cv_config)
    elif args.stage == "mc":
        run_mc_stage(ctx, resample_config, reprice_config)
    _write_hash_manifest()


if __name__ == "__main__":
    main()
