
"""Conservative post-cost research accounting for the option-only Markowitz paper.

This module is intentionally not a live execution simulator. It turns point-in-time
option holding ledgers into transparent research cost, capacity, borrow, margin, and
assignment diagnostics so the paper can separate gross theory evidence from net economic
robustness.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.option_market_hours import (
    classify_cboe_option_rth_timestamp,
)

ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ResearchCostConfig:
    nav_for_capacity: float = 1_000_000.0
    option_multiplier: float = 100.0
    fee_per_contract_per_side: float = 0.75
    spread_cross_fraction: float = 1.0
    default_equity_option_rel_spread: float = 0.10
    default_vix_option_rel_spread: float = 0.15
    slippage_bps_per_side: float = 5.0
    max_volume_participation: float = 0.10
    max_oi_participation: float = 0.02
    impact_cost_rate: float = 0.0025
    margin_funding_rate: float = 0.02
    short_option_margin_floor: float = 0.15
    stress_margin_rate: float = 0.25
    assignment_penalty_bps: float = 10.0
    use_cbbo_spread_surface: bool = True
    cbbo_spread_surface_path: str = "data/feature_store/cbbo_spread_surface.parquet"
    use_current_spread_assumptions: bool = True
    current_spread_assumptions_path: str = (
        "research/papers/option_only_markowitz/analysis/artifacts/"
        "breadth_solutions/current_option_spread_assumptions.csv"
    )
    use_inferred_spread_proxy: bool = False
    inferred_spread_proxy_quantile: float = 0.50
    inferred_spread_proxy_min_observations: int = 20


CBBO_SPREAD_SURFACE_COLUMNS = [
    "underlying",
    "snap_date",
    "moneyness_bucket",
    "tenor_bucket",
    "n_quotes",
    "n_contracts",
    "median_relative_spread",
    "p25_relative_spread",
    "p75_relative_spread",
    "median_mid",
    "median_displayed_size",
]


def _empty_cbbo_spread_surface() -> pd.DataFrame:
    return pd.DataFrame(columns=CBBO_SPREAD_SURFACE_COLUMNS)


def load_cbbo_spread_surface(root: Path, path: str | None = None) -> pd.DataFrame:
    rel_path = path or ResearchCostConfig().cbbo_spread_surface_path
    surface_path = Path(rel_path)
    if not surface_path.is_absolute():
        surface_path = root / surface_path
    if not surface_path.exists():
        return _empty_cbbo_spread_surface()
    df = pd.read_parquet(surface_path)
    out = df[[c for c in CBBO_SPREAD_SURFACE_COLUMNS if c in df.columns]].copy()
    for col in CBBO_SPREAD_SURFACE_COLUMNS:
        if col not in out:
            out[col] = np.nan
    out = out[CBBO_SPREAD_SURFACE_COLUMNS].copy()
    out["snap_date"] = pd.to_datetime(out["snap_date"], errors="coerce").dt.normalize()
    out["underlying"] = out["underlying"].astype(str).str.upper()
    out["moneyness_bucket"] = out["moneyness_bucket"].astype(str)
    out["tenor_bucket"] = out["tenor_bucket"].astype(str)
    out["median_relative_spread"] = pd.to_numeric(out["median_relative_spread"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


CURRENT_SPREAD_ASSUMPTION_COLUMNS = [
    "underlying",
    "quote_symbol",
    "asset_class",
    "moneyness_bucket",
    "tenor_bucket",
    "chain_timestamp",
    "snapshot_eastern",
    "timestamp_tz_assumed",
    "market_hours_snapshot",
    "market_hours_reason",
    "market_hours_rule",
    "source_url",
    "n_quotes",
    "n_liquid_quotes",
    "total_volume",
    "total_open_interest",
    "fill_relative_spread",
    "fill_abs_spread",
    "median_relative_spread",
    "p25_relative_spread",
    "p75_relative_spread",
    "median_abs_spread",
    "p25_abs_spread",
    "p75_abs_spread",
    "median_mid",
    "fill_method",
]


def _empty_current_spread_assumptions() -> pd.DataFrame:
    return pd.DataFrame(columns=CURRENT_SPREAD_ASSUMPTION_COLUMNS)


def load_current_spread_assumptions(root: Path, path: str | None = None) -> pd.DataFrame:
    rel_path = path or ResearchCostConfig().current_spread_assumptions_path
    assumptions_path = Path(rel_path)
    if not assumptions_path.is_absolute():
        assumptions_path = root / assumptions_path
    if not assumptions_path.exists():
        return _empty_current_spread_assumptions()
    df = pd.read_csv(assumptions_path)
    out = df[[c for c in CURRENT_SPREAD_ASSUMPTION_COLUMNS if c in df.columns]].copy()
    for col in CURRENT_SPREAD_ASSUMPTION_COLUMNS:
        if col not in out:
            out[col] = np.nan
    out = out[CURRENT_SPREAD_ASSUMPTION_COLUMNS].copy()
    for col in [
        "underlying",
        "quote_symbol",
        "asset_class",
        "moneyness_bucket",
        "tenor_bucket",
        "chain_timestamp",
        "snapshot_eastern",
        "timestamp_tz_assumed",
        "market_hours_reason",
        "market_hours_rule",
        "source_url",
        "fill_method",
    ]:
        out[col] = out[col].fillna("").astype(str)
    out["market_hours_snapshot"] = out["market_hours_snapshot"].map(_coerce_nullable_bool)
    for col in [
        "n_quotes",
        "n_liquid_quotes",
        "total_volume",
        "total_open_interest",
        "fill_relative_spread",
        "fill_abs_spread",
        "median_relative_spread",
        "p25_relative_spread",
        "p75_relative_spread",
        "median_abs_spread",
        "p25_abs_spread",
        "p75_abs_spread",
        "median_mid",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["underlying"] = out["underlying"].str.upper()
    out["moneyness_bucket"] = out["moneyness_bucket"].str.strip()
    out["tenor_bucket"] = out["tenor_bucket"].str.strip()
    out = _filter_market_hours_assumptions(out)
    return out.replace([np.inf, -np.inf], np.nan)


def _coerce_nullable_bool(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return pd.NA


def _filter_market_hours_assumptions(assumptions: pd.DataFrame) -> pd.DataFrame:
    if assumptions.empty:
        return assumptions

    computed_valid = []
    for _, row in assumptions.iterrows():
        timestamp_tz = str(row.get("timestamp_tz_assumed") or "UTC").strip() or "UTC"
        check = classify_cboe_option_rth_timestamp(row.get("chain_timestamp"), timestamp_tz=timestamp_tz)
        computed_valid.append(bool(check.valid))
    computed = pd.Series(computed_valid, index=assumptions.index, dtype=bool)

    declared = assumptions["market_hours_snapshot"] if "market_hours_snapshot" in assumptions else pd.Series(pd.NA, index=assumptions.index)
    declared_valid = declared.where(declared.notna(), computed).astype(bool)
    return assumptions.loc[computed & declared_valid].copy()


def load_borrow_proxy(root: Path = ROOT) -> pd.DataFrame:
    path = root / "data/feature_store/option_borrow_proxy_layer.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = ["symbol", "underlying", "snap_date", "expiry", "strike", "kind", "borrow_rate_proxy", "borrow_source"]
    df = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path, columns=[]).columns] if False else None)
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    if "snap_date" in out:
        out["snap_date"] = pd.to_datetime(out["snap_date"], errors="coerce").dt.normalize()
    if "expiry" in out:
        out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()
    if "borrow_rate_proxy" in out:
        out["borrow_rate_proxy"] = pd.to_numeric(out["borrow_rate_proxy"], errors="coerce")
    return out


def build_cost_input_ledger(
    reps: pd.DataFrame,
    return_detail: pd.DataFrame,
    root: Path = ROOT,
    config: ResearchCostConfig = ResearchCostConfig(),
    *,
    spread_surface: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if return_detail.empty:
        return pd.DataFrame()
    detail = return_detail.copy()
    for col in ["return_date", "decision_date", "expiry"]:
        if col in detail:
            detail[col] = pd.to_datetime(detail[col], errors="coerce").dt.normalize()
    rep_col_candidates = [
        "snap_date",
        "asset_id",
        "symbol",
        "volume",
        "open_interest",
        "cbbo_median_relative_spread",
        "breadth_spread_source",
        "moneyness_bucket",
    ]
    if spread_surface is not None and not spread_surface.empty:
        rep_col_candidates.extend(["underlying", "expiry", "tenor_days", "expiry_days"])
    rep_cols = [
        c
        for c in rep_col_candidates
        if c in reps.columns
    ]
    reps_slim = reps[rep_cols].copy() if rep_cols else pd.DataFrame(columns=["snap_date", "asset_id"])
    if "snap_date" in reps_slim:
        reps_slim["snap_date"] = pd.to_datetime(reps_slim["snap_date"], errors="coerce").dt.normalize()
    cost = detail.merge(
        reps_slim.drop_duplicates(["snap_date", "asset_id"]),
        left_on=["decision_date", "asset_id"],
        right_on=["snap_date", "asset_id"],
        how="left",
        suffixes=("", "_rep"),
    )
    if "symbol_rep" in cost:
        cost["symbol"] = cost["symbol"].where(cost["symbol"].notna(), cost["symbol_rep"])
    if spread_surface is not None and not spread_surface.empty:
        for col in ["underlying", "moneyness_bucket"]:
            rep_col = f"{col}_rep"
            if col in cost and rep_col in cost:
                cost[col] = cost[col].where(cost[col].notna(), cost[rep_col])
    borrow = load_borrow_proxy(root)
    if not borrow.empty and {"symbol", "snap_date", "borrow_rate_proxy"}.issubset(borrow.columns):
        cost = cost.merge(
            borrow[["symbol", "snap_date", "borrow_rate_proxy", "borrow_source"]].drop_duplicates(["symbol", "snap_date"]),
            left_on=["symbol", "decision_date"],
            right_on=["symbol", "snap_date"],
            how="left",
            suffixes=("", "_borrow"),
        )
    else:
        cost["borrow_rate_proxy"] = np.nan
        cost["borrow_source"] = "missing"
    cost["mark"] = pd.to_numeric(cost.get("mark", np.nan), errors="coerce")
    cost["volume"] = pd.to_numeric(cost.get("volume", np.nan), errors="coerce")
    if "open_interest" not in cost:
        cost["open_interest"] = np.nan
    cost["open_interest"] = pd.to_numeric(cost["open_interest"], errors="coerce")
    if "cbbo_median_relative_spread" in cost:
        rel = pd.to_numeric(cost["cbbo_median_relative_spread"], errors="coerce")
    else:
        rel = pd.Series(np.nan, index=cost.index, dtype=float)
    is_vix = cost.get("asset_class", pd.Series("", index=cost.index)).astype(str).eq("vix_option")
    defaults = np.where(is_vix, config.default_vix_option_rel_spread, config.default_equity_option_rel_spread)
    breadth_source = cost.get("breadth_spread_source", pd.Series("", index=cost.index)).fillna("").astype(str)
    poc_missing_mask = breadth_source.isin(["poc_missing_cbbo", "poc_default_fill"])
    panel_mask = rel.gt(0) & ~poc_missing_mask
    surface_rel = _surface_relative_spread(cost, spread_surface) if spread_surface is not None else pd.Series(np.nan, index=cost.index, dtype=float)
    surface_mask = (~panel_mask) & surface_rel.gt(0)
    inferred_rel = (
        _inferred_cbbo_proxy_relative_spread(cost, spread_surface, config)
        if config.use_inferred_spread_proxy and spread_surface is not None
        else pd.Series(np.nan, index=cost.index, dtype=float)
    )
    inferred_mask = (~panel_mask) & (~surface_mask) & inferred_rel.gt(0)
    current_assumptions = (
        load_current_spread_assumptions(root, config.current_spread_assumptions_path)
        if config.use_current_spread_assumptions
        else _empty_current_spread_assumptions()
    )
    current_rel = _current_assumption_relative_spread(cost, current_assumptions)
    current_mask = (~panel_mask) & (~surface_mask) & (~inferred_mask) & current_rel.gt(0)
    cost["relative_spread"] = rel.where(
        panel_mask,
        surface_rel.where(
            surface_mask,
            inferred_rel.where(inferred_mask, current_rel.where(current_mask, defaults)),
        ),
    ).clip(lower=0.0, upper=1.5)
    cost["relative_spread_source"] = np.select(
        [panel_mask, surface_mask, inferred_mask, current_mask],
        ["panel_cbbo", "surface_cbbo", "inferred_cbbo_proxy", "current_cboe_liquid_quote"],
        default="default",
    )
    cost["borrow_rate_proxy"] = pd.to_numeric(cost["borrow_rate_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    cost["holding_years"] = pd.to_numeric(cost.get("expiry_days", 21), errors="coerce").fillna(21.0).clip(lower=0.0) / 365.0
    cost["available_volume_contracts"] = cost["volume"].fillna(0.0).clip(lower=0.0)
    cost["available_oi_contracts"] = cost["open_interest"].fillna(np.inf)
    return cost.replace([np.inf, -np.inf], np.nan)


def _surface_relative_spread(cost: pd.DataFrame, spread_surface: pd.DataFrame | None) -> pd.Series:
    if spread_surface is None or spread_surface.empty or cost.empty:
        return pd.Series(np.nan, index=cost.index, dtype=float)
    required = {"underlying", "snap_date", "moneyness_bucket", "tenor_bucket", "median_relative_spread"}
    if not required.issubset(spread_surface.columns):
        return pd.Series(np.nan, index=cost.index, dtype=float)

    surface = spread_surface[list(required)].copy()
    surface["underlying"] = surface["underlying"].astype(str).str.upper()
    surface["snap_date"] = pd.to_datetime(surface["snap_date"], errors="coerce").dt.normalize()
    surface["moneyness_bucket"] = surface["moneyness_bucket"].astype(str)
    surface["tenor_bucket"] = surface["tenor_bucket"].astype(str)
    surface["median_relative_spread"] = pd.to_numeric(surface["median_relative_spread"], errors="coerce")
    surface = (
        surface.dropna(subset=["underlying", "snap_date", "moneyness_bucket", "tenor_bucket", "median_relative_spread"])
        .drop_duplicates(["underlying", "snap_date", "moneyness_bucket", "tenor_bucket"])
    )
    if surface.empty:
        return pd.Series(np.nan, index=cost.index, dtype=float)

    lookup = pd.DataFrame(index=cost.index)
    lookup["underlying"] = cost.get("underlying", pd.Series(np.nan, index=cost.index)).astype(str).str.upper()
    lookup["snap_date"] = pd.to_datetime(cost.get("decision_date", pd.Series(pd.NaT, index=cost.index)), errors="coerce").dt.normalize()
    lookup["moneyness_bucket"] = cost.get("moneyness_bucket", pd.Series(np.nan, index=cost.index)).astype(str)
    lookup["tenor_bucket"] = _tenor_bucket_for_cost_rows(cost)
    lookup["_row"] = cost.index
    joined = lookup.merge(
        surface[["underlying", "snap_date", "moneyness_bucket", "tenor_bucket", "median_relative_spread"]],
        on=["underlying", "snap_date", "moneyness_bucket", "tenor_bucket"],
        how="left",
    )
    return joined.set_index("_row")["median_relative_spread"].reindex(cost.index).astype(float)


def _inferred_cbbo_proxy_relative_spread(
    cost: pd.DataFrame,
    spread_surface: pd.DataFrame | None,
    config: ResearchCostConfig,
) -> pd.Series:
    """Infer missing liquid-option spreads from historical CBBO buckets.

    The proxy is point-in-time: each decision row sees only CBBO surface observations
    whose snapshot date is no later than that decision date.  It deliberately does not
    condition on underlying, because it is used only where an exact underlying/date
    panel row is absent.
    """

    if spread_surface is None or spread_surface.empty or cost.empty:
        return pd.Series(np.nan, index=cost.index, dtype=float)
    required = {"snap_date", "moneyness_bucket", "tenor_bucket", "median_relative_spread"}
    if not required.issubset(spread_surface.columns):
        return pd.Series(np.nan, index=cost.index, dtype=float)

    source = spread_surface[list(required)].copy()
    source["snap_date"] = pd.to_datetime(source["snap_date"], errors="coerce").dt.normalize()
    source["moneyness_bucket"] = _normalize_proxy_moneyness(source["moneyness_bucket"])
    source["tenor_bucket"] = source["tenor_bucket"].astype(str)
    source["median_relative_spread"] = pd.to_numeric(source["median_relative_spread"], errors="coerce")
    source = source.dropna(subset=["snap_date", "moneyness_bucket", "tenor_bucket", "median_relative_spread"])
    source = source[source["median_relative_spread"].gt(0)].copy()
    if source.empty:
        return pd.Series(np.nan, index=cost.index, dtype=float)

    q = float(config.inferred_spread_proxy_quantile)
    if not np.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError("inferred_spread_proxy_quantile must be between 0 and 1")
    min_obs = max(1, int(config.inferred_spread_proxy_min_observations))

    lookup = pd.DataFrame(index=cost.index)
    lookup["decision_date"] = pd.to_datetime(
        cost.get("decision_date", pd.Series(pd.NaT, index=cost.index)),
        errors="coerce",
    ).dt.normalize()
    lookup["moneyness_bucket"] = _normalize_proxy_moneyness(
        cost.get("moneyness_bucket", pd.Series(pd.NA, index=cost.index))
    )
    lookup["tenor_bucket"] = _tenor_bucket_for_cost_rows(cost)

    values: dict[tuple[pd.Timestamp, str, str], float] = {}
    unique_keys = (
        lookup.dropna(subset=["decision_date", "moneyness_bucket", "tenor_bucket"])
        [["decision_date", "moneyness_bucket", "tenor_bucket"]]
        .drop_duplicates()
    )
    for key in unique_keys.itertuples(index=False):
        decision_date = pd.Timestamp(key.decision_date)
        moneyness = str(key.moneyness_bucket)
        tenor = str(key.tenor_bucket)
        history = source[source["snap_date"].le(decision_date)]
        values[(decision_date, moneyness, tenor)] = _proxy_quantile_from_history(
            history,
            moneyness,
            tenor,
            q,
            min_obs,
        )

    out = pd.Series(np.nan, index=cost.index, dtype=float)
    for idx, row in lookup.iterrows():
        decision_date = row["decision_date"]
        moneyness = row["moneyness_bucket"]
        tenor = row["tenor_bucket"]
        if pd.isna(decision_date) or pd.isna(moneyness) or pd.isna(tenor):
            continue
        out.loc[idx] = values.get((pd.Timestamp(decision_date), str(moneyness), str(tenor)), np.nan)
    return out.astype(float)


def _proxy_quantile_from_history(
    history: pd.DataFrame,
    moneyness: str,
    tenor: str,
    quantile: float,
    min_obs: int,
) -> float:
    if history.empty:
        return float("nan")
    candidates = [
        history[
            history["moneyness_bucket"].eq(moneyness)
            & history["tenor_bucket"].eq(tenor)
        ]["median_relative_spread"],
        history[history["moneyness_bucket"].eq(moneyness)]["median_relative_spread"],
        history[history["tenor_bucket"].eq(tenor)]["median_relative_spread"],
        history["median_relative_spread"],
    ]
    for series in candidates:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) >= min_obs:
            return float(clean.quantile(quantile))
    clean = pd.to_numeric(history["median_relative_spread"], errors="coerce").dropna()
    return float(clean.quantile(quantile)) if len(clean) else float("nan")


def _normalize_proxy_moneyness(values: object) -> pd.Series:
    series = pd.Series(values).astype("string").str.strip().str.lower()
    series = series.str.replace(r"^vix_", "", regex=True)
    invalid = series.isna() | series.isin(["", "nan", "nat", "none", "<na>"])
    return series.mask(invalid, pd.NA)


def _current_assumption_relative_spread(cost: pd.DataFrame, assumptions: pd.DataFrame | None) -> pd.Series:
    if assumptions is None or assumptions.empty or cost.empty:
        return pd.Series(np.nan, index=cost.index, dtype=float)
    required = {"underlying", "moneyness_bucket", "tenor_bucket", "fill_relative_spread"}
    if not required.issubset(assumptions.columns):
        return pd.Series(np.nan, index=cost.index, dtype=float)

    source = assumptions[list(required)].copy()
    source["underlying"] = source["underlying"].astype(str).str.upper()
    source["moneyness_bucket"] = source["moneyness_bucket"].astype(str)
    source["tenor_bucket"] = source["tenor_bucket"].astype(str)
    source["fill_relative_spread"] = pd.to_numeric(source["fill_relative_spread"], errors="coerce")
    source = source.dropna(subset=["underlying", "moneyness_bucket", "tenor_bucket", "fill_relative_spread"])
    source = source[source["fill_relative_spread"].gt(0)].copy()
    if source.empty:
        return pd.Series(np.nan, index=cost.index, dtype=float)

    lookup = pd.DataFrame(index=cost.index)
    lookup["underlying"] = cost.get("underlying", pd.Series(np.nan, index=cost.index)).astype(str).str.upper()
    lookup["moneyness_bucket"] = cost.get("moneyness_bucket", pd.Series(np.nan, index=cost.index)).astype(str)
    lookup["tenor_bucket"] = _tenor_bucket_for_cost_rows(cost)
    lookup["_row"] = cost.index

    exact = _lookup_current_spread(lookup, source, ["underlying", "moneyness_bucket", "tenor_bucket"])
    by_bucket = _lookup_current_spread(
        lookup,
        source[source["tenor_bucket"].eq("all")],
        ["underlying", "moneyness_bucket"],
    )
    by_underlying = _lookup_current_spread(
        lookup,
        source[source["moneyness_bucket"].eq("all") & source["tenor_bucket"].eq("all")],
        ["underlying"],
    )
    return exact.combine_first(by_bucket).combine_first(by_underlying).reindex(cost.index).astype(float)


def _lookup_current_spread(lookup: pd.DataFrame, source: pd.DataFrame, keys: list[str]) -> pd.Series:
    if source.empty:
        return pd.Series(np.nan, index=lookup.index, dtype=float)
    source_slim = source[keys + ["fill_relative_spread"]].drop_duplicates(keys)
    joined = lookup[keys + ["_row"]].merge(source_slim, on=keys, how="left")
    return joined.set_index("_row")["fill_relative_spread"].reindex(lookup.index).astype(float)


def _tenor_bucket_for_cost_rows(cost: pd.DataFrame) -> pd.Series:
    if "tenor_bucket" in cost:
        existing = cost["tenor_bucket"].astype(str)
        valid = existing.notna() & ~existing.isin(["", "nan", "NaT", "None", "<NA>"])
    else:
        existing = pd.Series(pd.NA, index=cost.index, dtype="object")
        valid = pd.Series(False, index=cost.index)
    days = _tenor_days_for_cost_rows(cost)
    derived = days.map(_assign_cbbo_tenor_bucket)
    return existing.where(valid, derived).astype(str)


def _tenor_days_for_cost_rows(cost: pd.DataFrame) -> pd.Series:
    for col in ("expiry_days", "tenor_days", "dte", "days_to_expiry"):
        if col in cost:
            days = pd.to_numeric(cost[col], errors="coerce")
            if days.notna().any():
                return days
    if {"expiry", "decision_date"}.issubset(cost.columns):
        expiry = pd.to_datetime(cost["expiry"], errors="coerce")
        decision = pd.to_datetime(cost["decision_date"], errors="coerce")
        return (expiry - decision).dt.days.astype(float)
    return pd.Series(np.nan, index=cost.index, dtype=float)


def _assign_cbbo_tenor_bucket(days_to_expiry: object) -> object:
    try:
        days = float(days_to_expiry)
    except (TypeError, ValueError):
        return pd.NA
    if not np.isfinite(days):
        return pd.NA
    if days <= 45:
        return "le_45d"
    if days <= 120:
        return "46_120d"
    return "gt_120d"


def _is_vix_option_asset(asset_id: object, rows: pd.DataFrame | None = None) -> bool:
    if rows is not None and not rows.empty:
        if "asset_class" in rows and rows["asset_class"].astype(str).eq("vix_option").any():
            return True
        if "underlying" in rows and rows["underlying"].astype(str).str.upper().isin(["VIX", "VX_FRONT"]).any():
            return True
    aid = str(asset_id).upper()
    return "VIX" in aid or "VX_FRONT" in aid


def _option_fee_return(mark: float, config: ResearchCostConfig) -> float:
    denom = float(mark) * float(config.option_multiplier)
    fee = float(config.fee_per_contract_per_side)
    if np.isfinite(denom) and denom > 0 and np.isfinite(fee) and fee >= 0:
        return fee / denom
    return 0.0


def derive_entry_cost_series(
    cost_inputs: pd.DataFrame,
    contracts: Sequence[str],
    *,
    train_end: pd.Timestamp,
    config: ResearchCostConfig = ResearchCostConfig(),
) -> tuple[pd.Series, pd.DataFrame]:
    """Derive point-in-time one-way entry costs for Sortino optimization.

    Costs are expressed as return-on-premium: half the quoted relative spread
    plus one per-side contract fee scaled by ``mark * option_multiplier``.
    Only rows with ``return_date <= train_end`` are used. Rows with non-finite
    or non-positive marks, or non-finite/negative relative spreads, are dropped
    before averaging because they cannot define a finite quoted entry cost.
    Contracts without surviving training rows use the class default spread
    (equity vs. VIX option) plus a fee term based on the median training mark
    for that class; if the class has no finite positive training marks, the fee
    term is skipped.
    """

    contract_list = list(contracts)
    out_index = pd.Index(contract_list)
    empty_diag_cols = [
        "asset_id",
        "n_train_rows",
        "mean_relative_spread",
        "mean_mark",
        "entry_cost",
        "source",
    ]
    if len(contract_list) == 0:
        return pd.Series(dtype=float, index=out_index), pd.DataFrame(columns=empty_diag_cols)

    if cost_inputs.empty:
        pool = pd.DataFrame(columns=["asset_id", "return_date", "mark", "relative_spread"])
    else:
        pool = cost_inputs.copy()
        pool["return_date"] = pd.to_datetime(pool.get("return_date"), errors="coerce")
        pool = pool.loc[pool["return_date"].le(pd.Timestamp(train_end))]

    if pool.empty:
        valid = pd.DataFrame(columns=list(pool.columns) + ["entry_cost", "asset_class_derived"])
    else:
        pool["mark"] = pd.to_numeric(pool.get("mark", np.nan), errors="coerce")
        pool["relative_spread"] = pd.to_numeric(pool.get("relative_spread", np.nan), errors="coerce")
        finite_mark = np.isfinite(pool["mark"]) & pool["mark"].gt(0)
        finite_spread = np.isfinite(pool["relative_spread"]) & pool["relative_spread"].ge(0)
        valid = pool.loc[finite_mark & finite_spread].copy()
        if valid.empty:
            valid["entry_cost"] = pd.Series(dtype=float)
            valid["asset_class_derived"] = pd.Series(dtype=object)
        else:
            valid["entry_cost"] = 0.5 * valid["relative_spread"] + valid["mark"].map(
                lambda mark: _option_fee_return(float(mark), config)
            )
            valid["asset_class_derived"] = [
                "vix_option" if _is_vix_option_asset(aid, valid.loc[[idx]]) else "equity_option"
                for idx, aid in valid["asset_id"].items()
            ]

    if valid.empty:
        observed = pd.DataFrame(columns=empty_diag_cols).set_index("asset_id")
        class_median_mark = pd.Series(dtype=float)
    else:
        observed = (
            valid.groupby("asset_id")
            .agg(
                n_train_rows=("entry_cost", "size"),
                mean_relative_spread=("relative_spread", "mean"),
                mean_mark=("mark", "mean"),
                entry_cost=("entry_cost", "mean"),
            )
            .sort_index()
        )
        class_median_mark = valid.groupby("asset_class_derived")["mark"].median()

    diagnostics = []
    entry_costs = []
    for asset_id in contract_list:
        if asset_id in observed.index:
            row = observed.loc[asset_id]
            cost = max(float(row["entry_cost"]), 0.0)
            diagnostics.append(
                {
                    "asset_id": asset_id,
                    "n_train_rows": int(row["n_train_rows"]),
                    "mean_relative_spread": float(row["mean_relative_spread"]),
                    "mean_mark": float(row["mean_mark"]),
                    "entry_cost": cost,
                    "source": "train_observed",
                }
            )
            entry_costs.append(cost)
            continue

        asset_rows = pool.loc[pool.get("asset_id", pd.Series(dtype=object)).astype(str).eq(str(asset_id))] if not pool.empty else pd.DataFrame()
        asset_class = "vix_option" if _is_vix_option_asset(asset_id, asset_rows) else "equity_option"
        default_spread = (
            config.default_vix_option_rel_spread
            if asset_class == "vix_option"
            else config.default_equity_option_rel_spread
        )
        median_mark = float(class_median_mark.get(asset_class, np.nan))
        cost = max(0.5 * float(default_spread) + _option_fee_return(median_mark, config), 0.0)
        diagnostics.append(
            {
                "asset_id": asset_id,
                "n_train_rows": 0,
                "mean_relative_spread": np.nan,
                "mean_mark": median_mark if np.isfinite(median_mark) else np.nan,
                "entry_cost": cost,
                "source": "default_imputed",
            }
        )
        entry_costs.append(cost)

    entry_series = pd.Series(entry_costs, index=out_index, dtype=float)
    entry_series = entry_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    diag = pd.DataFrame(diagnostics, columns=empty_diag_cols)
    return entry_series, diag


def _assignment_risk(row: pd.Series, short_position: bool) -> bool:
    if not short_position:
        return False
    if str(row.get("asset_class", "equity_option")) != "equity_option":
        return False
    spot = float(row.get("start_spot", np.nan))
    strike = float(row.get("strike", np.nan))
    mark = float(row.get("mark", np.nan))
    if not np.isfinite([spot, strike, mark]).all() or mark <= 0:
        return False
    kind = str(row.get("kind", row.get("right", ""))).lower()
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    extrinsic = mark - intrinsic
    deep_itm = (kind == "call" and spot > 1.03 * strike) or (kind == "put" and strike > 1.03 * spot)
    return bool(deep_itm and extrinsic <= max(0.10 * mark, 0.05))


def compute_strategy_cost_ledgers(
    gross_returns: pd.DataFrame,
    strategies: dict[str, pd.Series],
    cost_inputs: pd.DataFrame,
    config: ResearchCostConfig = ResearchCostConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cost_inputs.empty:
        empty = pd.DataFrame(index=gross_returns.index)
        return gross_returns.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    capacity_rows = []
    margin_rows = []
    assignment_rows = []
    cost_by_asset = cost_inputs.set_index(["return_date", "asset_id"], drop=False)
    net = pd.DataFrame(index=gross_returns.index)
    for strategy, weights in strategies.items():
        net[strategy] = gross_returns[strategy] if strategy in gross_returns else np.nan
        for return_date in gross_returns.index:
            gross_cost = 0.0
            # Fail-closed accounting mirror of execution_cost_scenarios: when a
            # position cannot be costed (unusable mark), it cannot keep its
            # gross P&L, so its w_i * r_it contribution is excluded from the
            # month's net return.  Positions with no ledger row at all
            # correspond to zero-filled test returns (contribution already 0).
            excluded_gross = 0.0
            margin_req = 0.0
            stress_margin = 0.0
            assignment_notional = 0.0
            for asset_id, weight in weights.items():
                w = float(weight)
                if abs(w) <= 1e-14:
                    continue
                key = (pd.Timestamp(return_date).normalize(), asset_id)
                if key not in cost_by_asset.index:
                    continue
                row = cost_by_asset.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                mark = float(row.get("mark", np.nan))
                if not np.isfinite(mark) or mark <= 0:
                    try:
                        asset_return = float(row.get("option_return", np.nan))
                    except (TypeError, ValueError):
                        asset_return = np.nan
                    excluded_gross += w * asset_return if np.isfinite(asset_return) else 0.0
                    continue
                rel_spread = float(row.get("relative_spread", 0.10))
                holding_years = float(row.get("holding_years", 21 / 365))
                fee_return = 2.0 * config.fee_per_contract_per_side / (mark * config.option_multiplier)
                spread_return = config.spread_cross_fraction * rel_spread
                slippage_return = 2.0 * config.slippage_bps_per_side / 10_000.0
                is_short = w < 0
                kind = str(row.get("kind", row.get("right", ""))).lower()
                borrow_return = abs(w) * float(row.get("borrow_rate_proxy", 0.0)) * holding_years if is_short and kind == "call" else 0.0
                est_contracts = abs(w) * config.nav_for_capacity / (mark * config.option_multiplier)
                vol_cap = float(row.get("available_volume_contracts", 0.0)) * config.max_volume_participation
                oi_val = row.get("available_oi_contracts", np.nan)
                oi_cap = float(oi_val) * config.max_oi_participation if np.isfinite(float(oi_val)) else np.inf
                cap = max(1.0, min(vol_cap if vol_cap > 0 else np.inf, oi_cap))
                capacity_ratio = est_contracts / cap if cap > 0 and np.isfinite(cap) else np.inf
                capacity_penalty_return = max(capacity_ratio - 1.0, 0.0) ** 2 * config.impact_cost_rate
                start_spot = float(row.get("start_spot", np.nan))
                short_margin = abs(w) * max(config.short_option_margin_floor, 0.20 * start_spot / mark) if is_short and np.isfinite(start_spot) else abs(w)
                margin_component = short_margin if is_short else abs(w)
                margin_drag_return = margin_component * config.margin_funding_rate * holding_years
                assignment_flag = _assignment_risk(row, is_short)
                assignment_penalty = abs(w) * config.assignment_penalty_bps / 10_000.0 if assignment_flag else 0.0
                per_weight_cost = spread_return + fee_return + slippage_return + capacity_penalty_return
                cost_nav = abs(w) * per_weight_cost + borrow_return + margin_drag_return + assignment_penalty
                gross_cost += cost_nav
                margin_req += margin_component
                stress_margin += abs(w) * config.stress_margin_rate
                if assignment_flag:
                    assignment_notional += abs(w) * config.nav_for_capacity
                rows.append(
                    {
                        "return_date": return_date,
                        "strategy": strategy,
                        "asset_id": asset_id,
                        "weight": w,
                        "mark": mark,
                        "relative_spread": rel_spread,
                        "spread_cost_nav": abs(w) * spread_return,
                        "fee_cost_nav": abs(w) * fee_return,
                        "slippage_cost_nav": abs(w) * slippage_return,
                        "borrow_cost_nav": borrow_return,
                        "margin_drag_nav": margin_drag_return,
                        "capacity_cost_nav": abs(w) * capacity_penalty_return,
                        "assignment_penalty_nav": assignment_penalty,
                        "total_cost_nav": cost_nav,
                    }
                )
                capacity_rows.append(
                    {
                        "return_date": return_date,
                        "strategy": strategy,
                        "asset_id": asset_id,
                        "estimated_contracts": est_contracts,
                        "capacity_contracts": cap,
                        "capacity_ratio": capacity_ratio,
                        "capacity_status": "pass" if capacity_ratio <= 1.0 else "penalized",
                    }
                )
                if assignment_flag:
                    assignment_rows.append(
                        {
                            "return_date": return_date,
                            "strategy": strategy,
                            "asset_id": asset_id,
                            "assignment_risk_flag": True,
                            "assignment_notional_nav": abs(w),
                            "reason": "deep_itm_low_extrinsic_short_option",
                        }
                    )
            if strategy in net:
                net.loc[return_date, strategy] = (
                    gross_returns.loc[return_date, strategy] - gross_cost - excluded_gross
                )
            margin_rows.append(
                {
                    "return_date": return_date,
                    "strategy": strategy,
                    "margin_requirement_nav": margin_req,
                    "stress_margin_nav": stress_margin,
                    "assignment_notional_nav": assignment_notional / config.nav_for_capacity if config.nav_for_capacity else np.nan,
                    "excluded_gross_return_nav": excluded_gross,
                    "margin_model": "conservative_research_simulation",
                }
            )
    assignment = pd.DataFrame(assignment_rows)
    if assignment.empty:
        assignment = pd.DataFrame([{"return_date": pd.NaT, "strategy": "ALL", "asset_id": "NONE", "assignment_risk_flag": False, "assignment_notional_nav": 0.0, "reason": "no_assignment_risk_flags"}])
    return net, pd.DataFrame(rows), pd.DataFrame(capacity_rows), pd.DataFrame(margin_rows), assignment


def cost_diagnostics_table(cost_ledger: pd.DataFrame, capacity_ledger: pd.DataFrame, margin_ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not cost_ledger.empty:
        grouped = cost_ledger.groupby("strategy", observed=True)
        for strategy, grp in grouped:
            rows.append(
                {
                    "Strategy": strategy,
                    "Mean monthly cost": float(grp.groupby("return_date")["total_cost_nav"].sum().mean()),
                    "Ann. cost drag": float(grp.groupby("return_date")["total_cost_nav"].sum().mean() * 12.0),
                    "Spread share": float(grp["spread_cost_nav"].sum() / grp["total_cost_nav"].sum()) if grp["total_cost_nav"].sum() > 0 else np.nan,
                    "Fee share": float(grp["fee_cost_nav"].sum() / grp["total_cost_nav"].sum()) if grp["total_cost_nav"].sum() > 0 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and not margin_ledger.empty:
        margin = margin_ledger.groupby("strategy", observed=True)[["margin_requirement_nav", "stress_margin_nav", "assignment_notional_nav"]].mean().reset_index()
        out = out.merge(margin, left_on="Strategy", right_on="strategy", how="left").drop(columns=["strategy"])
    if not out.empty and not capacity_ledger.empty:
        cap = capacity_ledger.groupby("strategy", observed=True)["capacity_status"].apply(lambda s: float((s.astype(str) == "penalized").mean())).rename("Capacity penalized share").reset_index()
        out = out.merge(cap, left_on="Strategy", right_on="strategy", how="left").drop(columns=["strategy"])
    return out


def artifact_hash_manifest(paths: list[Path], base: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(set(p for p in paths if p.exists() and p.is_file())):
        rows.append({"path": str(path.relative_to(base)), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.DataFrame(rows)


def write_environment_lock(path: Path) -> dict[str, object]:
    lock = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
    }
    for name in ["numpy", "pandas", "scipy", "matplotlib", "cvxpy"]:
        try:
            mod = __import__(name)
            lock["packages"][name] = getattr(mod, "__version__", "unknown")
        except Exception:
            lock["packages"][name] = "not installed"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


__all__ = [
    "ResearchCostConfig",
    "artifact_hash_manifest",
    "build_cost_input_ledger",
    "compute_strategy_cost_ledgers",
    "cost_diagnostics_table",
    "load_borrow_proxy",
    "load_cbbo_spread_surface",
    "write_environment_lock",
]
