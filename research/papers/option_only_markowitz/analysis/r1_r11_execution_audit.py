"""Quote-grounded execution audit for the frozen R1 and R1.1 replays.

Licensed Databento rows remain in ``data/databento_cache``.  This module writes
only position-, month-, and session-level aggregates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data_ingestion.market_data.fetch_r1_r11_databento_audit import (
    _schema_for_option_events,
    close_window,
    open_window,
)
from research.papers.option_only_markowitz.analysis.publication_costs import (
    ResearchCostConfig,
    build_cost_input_ledger,
    load_cbbo_spread_surface,
)
from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import (
    PAPER,
    R1_NAME,
)
from research.papers.option_only_markowitz.analysis.r11_higher_risk_pipeline import (
    R11_NAME,
)
from research.papers.option_only_markowitz.analysis.r1_r11_aligned_comparison import (
    summarize_aligned,
)
from research.papers.option_only_markowitz.analysis.run_empirics import ROOT


DEFAULT_CACHE = Path("data/databento_cache/r1_r11_audit")
DEFAULT_OUT = PAPER / "analysis" / "artifacts" / "execution_audit"
TABLES = PAPER / "tables"
R11_WEIGHTS = PAPER / "analysis" / "artifacts" / "r11_higher_risk" / "r11_monthly_weights.csv"
R1_WEIGHTS = PAPER / "analysis" / "artifacts" / "r1_repaired" / "r1_monthly_weights.csv"
R11_MONTHLY = PAPER / "analysis" / "artifacts" / "r11_higher_risk" / "r11_monthly_development_returns.csv"
R1_MONTHLY = PAPER / "analysis" / "artifacts" / "r1_repaired" / "r1_monthly_development_returns.csv"
EVENT_ORDERS = PAPER / "analysis" / "artifacts" / "r11_higher_risk" / "r11_event_quote_request.csv"

BASE_QUOTE_PURPOSES = {
    "candidate_close_quotes",
    "gap_candidate_close_quotes",
    "held_next_open",
    "held_exit_close",
}
PATH_QUOTE_PURPOSE = "held_cbbo_path"
QUOTE_PURPOSES = BASE_QUOTE_PURPOSES | {PATH_QUOTE_PURPOSE}
VOLUME_PURPOSES = {"candidate_daily_volume", "gap_candidate_daily_volume"}
OI_PURPOSES = {"candidate_open_interest", "gap_candidate_open_interest"}
QUOTE_COLUMNS = [
    "ts_recv",
    "ts_event",
    "symbol",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
]
EXIT_OBS_COLUMNS = [
    "exit_obs_bid",
    "exit_obs_ask",
    "exit_obs_mid",
    "exit_obs_half_spread",
    "exit_obs_rel_spread",
    "exit_obs_bid_size",
    "exit_obs_ask_size",
    "exit_obs_schema",
    "exit_obs_ts_recv",
    "exit_obs_window_max_rel_spread",
    "exit_obs_coverage",
    "exit_obs_source",
]
def _read_columns_for_purpose(purpose: str, available: Iterable[str]) -> list[str]:
    if purpose in QUOTE_PURPOSES or purpose == "vix_intervention_cbbo":
        wanted = QUOTE_COLUMNS
    elif purpose in VOLUME_PURPOSES:
        wanted = ["ts_event", "publisher_id", "symbol", "volume"]
    elif purpose in OI_PURPOSES:
        wanted = ["ts_recv", "ts_event", "symbol", "stat_type", "quantity"]
    else:
        wanted = list(available)
    present = set(available)
    return [column for column in wanted if column in present]


def _load_audit_frames(
    cache_root: Path,
    purposes: set[str],
    symbols: set[str] | None = None,
) -> pd.DataFrame:
    ledger_path = Path(cache_root) / "request_ledger.json"
    with ledger_path.open(encoding="utf-8") as handle:
        ledger = json.load(handle)
    entries = ledger.items() if isinstance(ledger, dict) else ((entry.get("request_id", ""), entry) for entry in ledger)
    selected: list[tuple[str, dict[str, Any]]] = []
    for request_id, entry in entries:
        request = entry.get("request", {})
        if entry.get("status") != "complete" or request.get("purpose") not in purposes:
            continue
        requested_symbols = set(map(str, request.get("symbols", [])))
        if symbols is not None and requested_symbols and requested_symbols.isdisjoint(symbols):
            continue
        selected.append((str(request_id), entry))

    frames: list[pd.DataFrame] = []
    skipped = 0
    loaded_files = 0
    for request_id, entry in selected:
        request = entry["request"]
        purpose = str(request["purpose"])
        path = (
            Path(cache_root)
            / f"phase{request['phase']}"
            / purpose
            / f"{request_id}.parquet"
        )
        if not path.exists():
            skipped += 1
            continue
        columns = _read_columns_for_purpose(purpose, entry.get("columns", []))
        filters = None
        if symbols is not None and "symbol" in columns:
            relevant = sorted(symbols.intersection(set(map(str, request.get("symbols", [])))))
            if not relevant:
                continue
            filters = [("symbol", "in", relevant)]
        try:
            frame = pd.read_parquet(path, columns=columns or None, filters=filters)
        except (OSError, ValueError, TypeError):
            if filters is None:
                skipped += 1
                continue
            try:
                frame = pd.read_parquet(path, columns=columns or None)
                frame = frame[frame["symbol"].astype(str).isin(relevant)]
            except (OSError, ValueError, TypeError, KeyError):
                skipped += 1
                continue
        frame["purpose"] = purpose
        frame["request_start"] = pd.Timestamp(request["start"])
        frame["request_end"] = pd.Timestamp(request["end"])
        frame["schema"] = request.get("schema", "")
        frame["request_id"] = request_id
        frames.append(frame)
        loaded_files += 1
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    out.attrs["skipped_files"] = skipped
    out.attrs["loaded_files"] = loaded_files
    out.attrs["selected_files"] = len(selected)
    return out


def load_audit_frames(cache_root: Path, purposes: set[str]) -> pd.DataFrame:
    """Load completed ledger requests for the requested purposes.

    Missing and unreadable parquet files are skipped and counted in
    ``result.attrs['skipped_files']``.
    """

    return _load_audit_frames(Path(cache_root), set(purposes))


def _modeled_cost_inputs() -> pd.DataFrame:
    from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import (
        build_configs,
    )
    from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import (
        _build_config_panel,
    )

    configs, _ = build_configs()
    config = ResearchCostConfig(
        nav_for_capacity=1_000_000.0,
        impact_cost_rate=0.0,
        use_current_spread_assumptions=False,
        use_inferred_spread_proxy=True,
    )
    surface = load_cbbo_spread_surface(ROOT, config.cbbo_spread_surface_path)
    keep = [
        "decision_date",
        "asset_id",
        "mark",
        "relative_spread",
        "relative_spread_source",
        "holding_years",
        "start_spot",
        "kind",
        "asset_class",
        "borrow_rate_proxy",
    ]
    def build_for_config(label: str) -> pd.DataFrame:
        panel_args = configs[label]
        reps, _, detail, _, _ = _build_config_panel(*panel_args)
        costs = build_cost_input_ledger(
            reps,
            detail,
            ROOT,
            config,
            spread_surface=surface,
        )
        costs["config"] = label
        return costs[["config", *keep]]

    frames = [
        build_for_config("orig+VIX"),
        build_for_config("larger+VIX"),
        build_for_config("orig"),
        build_for_config("larger"),
    ]
    modeled = pd.concat(frames, ignore_index=True)
    modeled["decision_date"] = pd.to_datetime(modeled["decision_date"]).dt.normalize()
    modeled = modeled.sort_values("decision_date").drop_duplicates(
        ["config", "decision_date", "asset_id"], keep="last"
    )
    return modeled.rename(
        columns={"decision_date": "modeled_cost_input_date", "mark": "modeled_cost_input_mark"}
    )


def _apply_resolved_symbols(trades: pd.DataFrame) -> pd.DataFrame:
    path = DEFAULT_CACHE / "resolved_gap_contracts.parquet"
    if not path.exists():
        return trades
    resolved = pd.read_parquet(
        path,
        columns=["decision_date", "asset_id", "symbol", "underlying", "expiry"],
    )
    resolved["decision_date"] = pd.to_datetime(resolved["decision_date"]).dt.normalize()
    resolved = resolved.drop_duplicates(["decision_date", "asset_id"])
    out = trades.merge(
        resolved,
        on=["decision_date", "asset_id"],
        how="left",
        suffixes=("", "_resolved"),
    )
    out["symbol"] = out["symbol_resolved"].where(out["symbol_resolved"].notna(), out["symbol"])
    out["underlying"] = out["underlying_resolved"].where(
        out["underlying_resolved"].notna(), out["underlying"]
    )
    out["expiry"] = out["expiry_resolved"].where(out["expiry_resolved"].notna(), out["expiry"])
    return out.drop(columns=["symbol_resolved", "underlying_resolved", "expiry_resolved"])


def _attach_modeled_costs(trades: pd.DataFrame, modeled: pd.DataFrame) -> pd.DataFrame:
    """Attach the last cost input known on each decision date and rebuild its rates."""

    left = trades.copy()
    left["_trade_order"] = np.arange(len(left))
    left["config"] = left["config"].astype(str)
    left["asset_id"] = left["asset_id"].astype(str)
    left["decision_date"] = pd.to_datetime(left["decision_date"]).astype("datetime64[ns]")
    right = modeled.copy()
    right["config"] = right["config"].astype(str)
    right["asset_id"] = right["asset_id"].astype(str)
    right["modeled_cost_input_date"] = pd.to_datetime(
        right["modeled_cost_input_date"]
    ).astype("datetime64[ns]")
    left = left.sort_values(["decision_date", "config", "asset_id"])
    right = right.sort_values(["modeled_cost_input_date", "config", "asset_id"])
    out = pd.merge_asof(
        left,
        right,
        left_on="decision_date",
        right_on="modeled_cost_input_date",
        by=["config", "asset_id"],
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_trade_order")
    out = out.drop(columns="_trade_order").reset_index(drop=True)

    config = ResearchCostConfig()
    available = out["modeled_cost_input_date"].notna()
    model_mark = pd.to_numeric(out["modeled_cost_input_mark"], errors="coerce")
    spread = pd.to_numeric(out["relative_spread"], errors="coerce")
    vix_name = out["asset_id"].str.upper().str.contains("VIX|VX_FRONT", regex=True)
    is_vix = out["asset_class"].astype(str).eq("vix_option") | vix_name
    default_spread = pd.Series(np.where(is_vix, 0.15, 0.10), index=out.index)
    spread = spread.where(spread.notna() & spread.ge(0.0), default_spread)
    holding = pd.to_numeric(out["holding_years"], errors="coerce").fillna(21.0 / 365.0)
    fee_rate = (
        2.0 * config.fee_per_contract_per_side / (model_mark * config.option_multiplier)
    ).where(available & model_mark.gt(0.0), 0.0)
    slippage_rate = pd.Series(
        2.0 * config.slippage_bps_per_side / 10_000.0,
        index=out.index,
    )
    spot = pd.to_numeric(out["start_spot"], errors="coerce")
    margin = pd.Series(
        np.maximum(config.short_option_margin_floor, 0.20 * spot / model_mark),
        index=out.index,
    ).where(spot.notna() & model_mark.gt(0.0), 1.0)
    borrow = pd.to_numeric(out["borrow_rate_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    call = out["kind"].astype(str).str.lower().eq("call")
    long_funding = config.margin_funding_rate * holding
    short_funding = margin * config.margin_funding_rate * holding
    short_borrow = borrow.where(call, 0.0) * holding
    observed_base = spread + fee_rate + slippage_rate
    fallback_base = default_spread + slippage_rate

    out["modeled_input_available"] = available
    out["modeled_cost_input_lag_days"] = (
        out["decision_date"] - out["modeled_cost_input_date"]
    ).dt.days
    out["modeled_relative_spread"] = spread.where(available, default_spread)
    out["modeled_fee_rate"] = fee_rate
    out["modeled_slippage_rate"] = slippage_rate
    out["modeled_long_funding_rate"] = long_funding.where(available, 0.0)
    out["modeled_short_funding_rate"] = short_funding.where(available, 0.0)
    out["modeled_short_borrow_rate"] = short_borrow.where(available, 0.0)
    out["modeled_long_rate"] = (
        observed_base + long_funding
    ).where(available, fallback_base)
    out["modeled_short_rate"] = (
        observed_base + short_funding + short_borrow
    ).where(available, fallback_base)
    return out


def build_trade_table() -> pd.DataFrame:
    """Return one active-position row per arm, config, date, and exact OSI."""

    r11_all = pd.read_csv(R11_WEIGHTS)
    r11_all["decision_date"] = pd.to_datetime(r11_all["decision_date"]).dt.normalize()
    r11_all["return_date"] = pd.to_datetime(r11_all["return_date"]).dt.normalize()
    base = r11_all[r11_all["strategy"].eq(R11_NAME)].copy()
    r11 = base[pd.to_numeric(base["integer_contracts"], errors="coerce").ne(0)].copy()
    r11["arm"] = "R1.1"
    r11["contracts_source"] = "frozen_integerized"

    r1 = pd.read_csv(R1_WEIGHTS)
    r1["decision_date"] = pd.to_datetime(r1["decision_date"]).dt.normalize()
    r1["return_date"] = pd.to_datetime(r1["return_date"]).dt.normalize()
    r1 = r1[pd.to_numeric(r1["weight"], errors="coerce").ne(0)].copy()
    lookup = base[
        ["config", "decision_date", "asset_id", "symbol", "underlying", "expiry", "mark"]
    ].drop_duplicates(["config", "decision_date", "asset_id"])
    r1 = r1.merge(lookup, on=["config", "decision_date", "asset_id"], how="left")
    r1["integer_contracts"] = np.rint(
        pd.to_numeric(r1["weight"], errors="coerce")
        * 1_000_000.0
        / (pd.to_numeric(r1["mark"], errors="coerce") * 100.0)
    )
    r1["arm"] = "R1"
    r1["contracts_source"] = "implied_from_weight_at_1mm_nav"

    columns = [
        "arm",
        "config",
        "decision_date",
        "return_date",
        "asset_id",
        "symbol",
        "underlying",
        "expiry",
        "mark",
        "weight",
        "integer_contracts",
        "contracts_source",
    ]
    trades = pd.concat([r1[columns], r11[columns]], ignore_index=True)
    trades["expiry"] = pd.to_datetime(trades["expiry"], errors="coerce").dt.normalize()
    trades = _apply_resolved_symbols(trades)
    trades = _attach_modeled_costs(trades, _modeled_cost_inputs())
    group = ["arm", "config", "decision_date", "symbol"]
    trades = trades.sort_values(group).drop_duplicates(group, keep="last")
    return trades.reset_index(drop=True)


def _request_dates(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["request_start"], utc=True).dt.tz_convert(None).dt.normalize()


def _coerce_quote_columns(q: pd.DataFrame) -> None:
    for column in ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"):
        q[column] = pd.to_numeric(q.get(column), errors="coerce")


def _last_valid_quote_metrics(q: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Per-group last valid quote, worst relative spread, and coverage flag."""

    q["valid"] = q["bid_px_00"].gt(0) & q["ask_px_00"].ge(q["bid_px_00"])
    q["one_sided"] = q["bid_px_00"].le(0) & q["ask_px_00"].gt(0)
    flags = q.groupby(group, observed=True)[["valid", "one_sided"]].max()
    valid = q[q["valid"]].copy()
    valid["mid"] = (valid["bid_px_00"] + valid["ask_px_00"]) / 2.0
    valid["half_spread"] = (valid["ask_px_00"] - valid["bid_px_00"]) / 2.0
    valid["rel_spread"] = 2.0 * valid["half_spread"] / valid["mid"]
    maximum = valid.groupby(group, observed=True)["rel_spread"].max()
    last = (
        valid.sort_values([*group, "ts_recv"])
        .groupby(group, observed=True)
        .tail(1)
        .set_index(group)
    )
    selected = (
        last[
            [
                "bid_px_00",
                "ask_px_00",
                "mid",
                "half_spread",
                "rel_spread",
                "bid_sz_00",
                "ask_sz_00",
                "schema",
                "ts_recv",
            ]
        ]
        .join(maximum.rename("window_max_rel_spread"), how="outer")
        .join(flags, how="outer")
    )
    selected["coverage"] = np.select(
        [selected["valid"].fillna(False), selected["one_sided"].fillna(False)],
        ["covered", "one_sided"],
        default="missing",
    )
    return selected.drop(columns=["valid", "one_sided"])


def _quote_snapshot(
    quotes: pd.DataFrame,
    purposes: set[str],
    key_dates: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    q = quotes[quotes["purpose"].isin(purposes)].copy()
    wanted = key_dates[["symbol", "key_date", "window_start", "window_end"]].drop_duplicates()
    if q.empty:
        empty = wanted[["symbol", "key_date"]].copy()
        empty[f"{prefix}_bid"] = np.nan
        empty[f"{prefix}_ask"] = np.nan
        empty[f"{prefix}_mid"] = np.nan
        empty[f"{prefix}_half_spread"] = np.nan
        empty[f"{prefix}_rel_spread"] = np.nan
        empty[f"{prefix}_bid_size"] = np.nan
        empty[f"{prefix}_ask_size"] = np.nan
        empty[f"{prefix}_schema"] = pd.NA
        empty[f"{prefix}_ts_recv"] = pd.NaT
        empty[f"{prefix}_window_max_rel_spread"] = np.nan
        empty[f"{prefix}_coverage"] = "missing"
        return empty
    q["key_date"] = _request_dates(q)
    q = q.merge(wanted, on=["symbol", "key_date"], how="inner")
    _coerce_quote_columns(q)
    q["ts_recv"] = pd.to_datetime(q["ts_recv"], utc=True)
    q["request_start"] = pd.to_datetime(q["request_start"], utc=True)
    q["request_end"] = pd.to_datetime(q["request_end"], utc=True)
    q["window_start"] = pd.to_datetime(q["window_start"], utc=True)
    q["window_end"] = pd.to_datetime(q["window_end"], utc=True)
    in_window = (
        q["ts_recv"].gt(q["request_start"])
        & q["ts_recv"].le(q["request_end"])
        & q["ts_recv"].gt(q["window_start"])
        & q["ts_recv"].le(q["window_end"])
    )
    q = q[in_window].copy()
    selected = _last_valid_quote_metrics(q, ["symbol", "key_date"]).reset_index()
    selected = selected.rename(
        columns={
            "bid_px_00": f"{prefix}_bid",
            "ask_px_00": f"{prefix}_ask",
            "mid": f"{prefix}_mid",
            "half_spread": f"{prefix}_half_spread",
            "rel_spread": f"{prefix}_rel_spread",
            "bid_sz_00": f"{prefix}_bid_size",
            "ask_sz_00": f"{prefix}_ask_size",
            "schema": f"{prefix}_schema",
            "ts_recv": f"{prefix}_ts_recv",
            "window_max_rel_spread": f"{prefix}_window_max_rel_spread",
            "coverage": f"{prefix}_coverage",
        }
    )
    return wanted[["symbol", "key_date"]].merge(
        selected, on=["symbol", "key_date"], how="left"
    )


def _window_keys(symbols: pd.Series, values: pd.Series, window: str) -> pd.DataFrame:
    source = pd.to_datetime(values).dt.normalize()
    unique = pd.DatetimeIndex(source.dropna().unique())
    boundary_function = open_window if window == "open" else close_window
    boundaries = pd.Series(unique).map(boundary_function)
    boundary_values = pd.DataFrame(boundaries.tolist(), columns=["window_start", "window_end"])
    lookup = pd.DataFrame(
        {
            "source_date": unique,
            "window_start": boundary_values["window_start"],
            "window_end": boundary_values["window_end"],
        }
    )
    lookup["key_date"] = (
        pd.to_datetime(lookup["window_start"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    keys = pd.DataFrame({"symbol": symbols.to_numpy(), "source_date": source.to_numpy()})
    return keys.merge(lookup, on="source_date", how="left").drop(columns="source_date")


def _last_path_exit_snapshot(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    wanted = trades[["symbol", "expiry"]].drop_duplicates()
    q = quotes[quotes["purpose"].eq(PATH_QUOTE_PURPOSE)].copy()
    if q.empty:
        empty = wanted[["symbol"]].drop_duplicates().copy()
        return empty.reindex(columns=["symbol", *EXIT_OBS_COLUMNS])
    q = q.merge(wanted, on="symbol", how="inner")
    _coerce_quote_columns(q)
    q["ts_recv"] = pd.to_datetime(q["ts_recv"], utc=True)
    q["expiry_utc"] = pd.to_datetime(q["expiry"], utc=True)
    q = q[q["ts_recv"].dt.normalize().lt(q["expiry_utc"])].copy()
    q = q.drop_duplicates(
        ["symbol", "ts_recv", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    )
    q["quote_day"] = q["ts_recv"].dt.normalize()
    q["last_quote_day"] = q.groupby("symbol", observed=True)["quote_day"].transform("max")
    q = q[q["quote_day"].eq(q["last_quote_day"])].copy()
    q["window_end"] = q.groupby("symbol", observed=True)["ts_recv"].transform("max")
    q = q[q["ts_recv"].gt(q["window_end"] - pd.Timedelta(minutes=10))].copy()
    selected = _last_valid_quote_metrics(q, ["symbol"])
    selected["source"] = "held_cbbo_path_last_tradable_session"
    selected = selected.reset_index().rename(
        columns={
            "bid_px_00": "exit_obs_bid",
            "ask_px_00": "exit_obs_ask",
            "mid": "exit_obs_mid",
            "half_spread": "exit_obs_half_spread",
            "rel_spread": "exit_obs_rel_spread",
            "bid_sz_00": "exit_obs_bid_size",
            "ask_sz_00": "exit_obs_ask_size",
            "schema": "exit_obs_schema",
            "ts_recv": "exit_obs_ts_recv",
            "window_max_rel_spread": "exit_obs_window_max_rel_spread",
            "coverage": "exit_obs_coverage",
            "source": "exit_obs_source",
        }
    )
    return wanted[["symbol"]].drop_duplicates().merge(selected, on="symbol", how="left")


def match_quotes(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Vector-match decision-close, next-open, and expiry-close quotes."""

    out = trades.copy()
    out["decision_date"] = pd.to_datetime(out["decision_date"]).dt.normalize()
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.normalize()
    entry_keys = _window_keys(out["symbol"], out["decision_date"], "close")
    open_keys = _window_keys(out["symbol"], out["decision_date"], "open")
    exit_keys = _window_keys(out["symbol"], out["expiry"], "close")
    entry = _quote_snapshot(
        quotes,
        {"candidate_close_quotes", "gap_candidate_close_quotes"},
        entry_keys,
        "obs",
    )
    opened = _quote_snapshot(quotes, {"held_next_open"}, open_keys, "open_obs")
    exited = _quote_snapshot(quotes, {"held_exit_close"}, exit_keys, "exit_obs")
    path_exited = _last_path_exit_snapshot(quotes, out)
    out["_entry_key"] = entry_keys["key_date"].to_numpy()
    out["_open_key"] = open_keys["key_date"].to_numpy()
    out["_exit_key"] = exit_keys["key_date"].to_numpy()
    out = out.merge(entry, left_on=["symbol", "_entry_key"], right_on=["symbol", "key_date"], how="left")
    out = out.drop(columns="key_date")
    out = out.merge(opened, left_on=["symbol", "_open_key"], right_on=["symbol", "key_date"], how="left")
    out = out.drop(columns="key_date")
    out = out.merge(exited, left_on=["symbol", "_exit_key"], right_on=["symbol", "key_date"], how="left")
    out = out.drop(columns=["key_date", "_entry_key", "_open_key", "_exit_key"])
    out["exit_obs_source"] = np.where(
        out["exit_obs_coverage"].isin(["covered", "one_sided"]),
        "held_exit_close",
        "missing",
    )
    path_exited = path_exited.rename(
        columns={column: f"{column}_path" for column in EXIT_OBS_COLUMNS}
    )
    out = out.merge(path_exited, on="symbol", how="left")
    is_vix = out["underlying"].astype(str).str.upper().isin(["VIX", "VX_FRONT"])
    use_path = (
        is_vix
        & ~out["exit_obs_coverage"].eq("covered")
        & out["exit_obs_coverage_path"].eq("covered")
    )
    exit_columns = EXIT_OBS_COLUMNS
    path_columns = [f"{column}_path" for column in exit_columns]
    if use_path.any():
        out["exit_obs_ts_recv"] = pd.to_datetime(out["exit_obs_ts_recv"], utc=True)
        out["exit_obs_ts_recv_path"] = pd.to_datetime(out["exit_obs_ts_recv_path"], utc=True)
        path_values = out[path_columns].copy()
        path_values.columns = exit_columns
        out[exit_columns] = out[exit_columns].mask(use_path, path_values, axis=0)
    out = out.drop(columns=path_columns)
    out["obs_coverage"] = out["obs_coverage"].fillna("missing")
    out["open_obs_coverage"] = out["open_obs_coverage"].fillna("missing")
    out["exit_obs_coverage"] = out["exit_obs_coverage"].fillna("missing")
    out["exit_obs_source"] = out["exit_obs_source"].fillna("missing")
    out["coverage"] = out["obs_coverage"]
    out["quote_schema"] = out["obs_schema"].fillna(
        out["decision_date"].map(_schema_for_option_events)
    )
    out["entry_window_max_rel_spread"] = out["obs_window_max_rel_spread"]
    return out


def _load_monthly() -> pd.DataFrame:
    r11 = pd.read_csv(R11_MONTHLY, dtype={"gross_return": "string"})
    r11 = r11[r11["strategy"].eq(R11_NAME)].copy()
    r11["arm"] = "R1.1"
    r1 = pd.read_csv(R1_MONTHLY, dtype={"gross_return": "string"})
    r1 = r1[r1["strategy"].eq(R1_NAME)].copy()
    r1["arm"] = "R1"
    monthly = pd.concat([r1, r11], ignore_index=True, sort=False)
    monthly["_gross_return_frozen_text"] = monthly["gross_return"].astype(str)
    monthly["gross_return"] = pd.to_numeric(monthly["gross_return"], errors="raise")
    monthly["return_date"] = pd.to_datetime(monthly["return_date"]).dt.normalize()
    monthly["decision_date"] = pd.to_datetime(monthly["decision_date"]).dt.normalize()
    return monthly


def _recompute_cost_details(
    matched: pd.DataFrame, monthly: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = matched.copy()
    absolute_weight = pd.to_numeric(positions["weight"], errors="coerce").abs()
    is_long = pd.to_numeric(positions["weight"], errors="coerce").ge(0)
    modeled_rate = pd.Series(
        np.where(is_long, positions["modeled_long_rate"], positions["modeled_short_rate"]),
        index=positions.index,
        dtype=float,
    )
    modeled_spread = pd.to_numeric(positions["modeled_relative_spread"], errors="coerce")
    nonspread_rate = (modeled_rate - modeled_spread).clip(lower=0.0)
    covered = positions["obs_coverage"].eq("covered") & positions["exit_obs_coverage"].eq("covered")
    touch_spread = (
        pd.to_numeric(positions["obs_rel_spread"], errors="coerce")
        + pd.to_numeric(positions["exit_obs_rel_spread"], errors="coerce")
    ) / 2.0
    worst_spread = (
        pd.to_numeric(positions["entry_window_max_rel_spread"], errors="coerce")
        + pd.to_numeric(positions["exit_obs_rel_spread"], errors="coerce")
    ) / 2.0
    positions["modeled_position_cost"] = absolute_weight * modeled_rate
    positions["position_cost_mid"] = np.where(
        covered,
        absolute_weight * nonspread_rate,
        positions["modeled_position_cost"],
    )
    positions["position_cost_touch"] = np.where(
        covered,
        absolute_weight * (nonspread_rate + touch_spread),
        positions["modeled_position_cost"],
    )
    positions["position_cost_worst"] = np.where(
        covered,
        absolute_weight * (nonspread_rate + worst_spread),
        positions["modeled_position_cost"],
    )
    displayed = positions[["obs_bid_size", "obs_ask_size", "exit_obs_bid_size", "exit_obs_ask_size"]].min(axis=1)
    positions["size_exceeds_displayed"] = pd.to_numeric(
        positions["integer_contracts"], errors="coerce"
    ).abs().gt(displayed)
    positions["covered_abs_weight"] = absolute_weight.where(covered, 0.0)
    positions["gross_abs_weight"] = absolute_weight
    keys = ["arm", "config", "return_date"]
    aggregate = positions.groupby(keys, observed=True).agg(
        observed_cost_mid=("position_cost_mid", "sum"),
        observed_cost_touch=("position_cost_touch", "sum"),
        observed_cost_worst=("position_cost_worst", "sum"),
        reconstructed_modeled_cost=("modeled_position_cost", "sum"),
        covered_abs_weight=("covered_abs_weight", "sum"),
        gross_abs_weight=("gross_abs_weight", "sum"),
        size_exceeds_displayed=("size_exceeds_displayed", "max"),
    ).reset_index()
    base = monthly.copy()
    base["return_date"] = pd.to_datetime(base["return_date"]).dt.normalize()
    result = base.merge(aggregate, on=keys, how="left")
    empty_month = result["gross_abs_weight"].isna()
    result["observed_cost_mid"] = result["observed_cost_mid"].where(
        ~empty_month, result["predicted_cost"]
    )
    result["observed_cost_touch"] = result["observed_cost_touch"].where(
        ~empty_month, result["predicted_cost"]
    )
    result["observed_cost_worst"] = result["observed_cost_worst"].where(
        ~empty_month, result["predicted_cost"]
    )
    result["reconstructed_modeled_cost"] = result["reconstructed_modeled_cost"].where(
        ~empty_month, result["predicted_cost"]
    )
    result["coverage_weight_fraction"] = (
        result["covered_abs_weight"] / result["gross_abs_weight"].replace(0.0, np.nan)
    ).where(~empty_month)
    result["uncovered_cost_source"] = "modeled"
    result["net_return_mid"] = result["gross_return"] - result["observed_cost_mid"]
    result["net_return_touch"] = result["gross_return"] - result["observed_cost_touch"]
    result["net_return_worst"] = result["gross_return"] - result["observed_cost_worst"]
    result["cost_reconstruction_gap"] = result["reconstructed_modeled_cost"] - result["predicted_cost"]
    result["cost_reconstruction"] = np.where(
        result["cost_reconstruction_gap"].abs().le(1e-6), "exact", "approximate"
    )
    if "_gross_return_frozen_text" in result:
        result["gross_return"] = result["_gross_return_frozen_text"]
    keep = [
        "arm",
        "config",
        "return_date",
        "decision_date",
        "gross_return",
        "gross_nav",
        "predicted_cost",
        "net_return",
        "observed_cost_mid",
        "observed_cost_touch",
        "observed_cost_worst",
        "net_return_mid",
        "net_return_touch",
        "net_return_worst",
        "coverage_weight_fraction",
        "uncovered_cost_source",
        "cost_reconstruction",
        "cost_reconstruction_gap",
        "size_exceeds_displayed",
    ]
    return result[[column for column in keep if column in result.columns]], positions


def recompute_costs(matched: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Recompute monthly costs under mid, touch, and worst quote scenarios."""

    return _recompute_cost_details(matched, monthly)[0]


def mark_accuracy(matched: pd.DataFrame) -> pd.DataFrame:
    """Return per-position artifact-mark versus observed-mid errors."""

    out = matched[
        ["arm", "config", "decision_date", "symbol", "mark", "obs_mid", "coverage", "quote_schema"]
    ].copy()
    out = out.rename(columns={"mark": "artifact_mark", "obs_mid": "observed_mid"})
    out["absolute_error"] = (out["artifact_mark"] - out["observed_mid"]).abs()
    out["relative_error"] = out["absolute_error"] / out["observed_mid"].abs().replace(0.0, np.nan)
    out["regime"] = np.where(
        pd.to_datetime(out["decision_date"]).lt(pd.Timestamp("2023-03-28")),
        "pre_2023_03_28",
        "post_2023_03_28",
    )
    return out


def _liquidity_validation(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    volume = data[data["purpose"].isin(VOLUME_PURPOSES)].copy()
    volume["decision_date"] = _request_dates(volume)
    volume["volume"] = pd.to_numeric(volume["volume"], errors="coerce")
    volume = volume.groupby(["symbol", "decision_date"], observed=True)["volume"].agg(
        volume_sum="sum", volume_max="max"
    ).reset_index()
    oi = data[data["purpose"].isin(OI_PURPOSES)].copy()
    oi["decision_date"] = _request_dates(oi)
    oi["stat_type"] = pd.to_numeric(oi["stat_type"], errors="coerce")
    oi["quantity"] = pd.to_numeric(oi["quantity"], errors="coerce")
    oi = oi[oi["stat_type"].eq(9)]
    oi = oi.groupby(["symbol", "decision_date"], observed=True)["quantity"].max().rename("open_interest").reset_index()
    out = trades.merge(volume, on=["symbol", "decision_date"], how="left").merge(
        oi, on=["symbol", "decision_date"], how="left"
    )
    contracts = pd.to_numeric(out["integer_contracts"], errors="coerce").abs()
    out["participation_volume_sum"] = contracts / out["volume_sum"].replace(0.0, np.nan)
    out["participation_volume_max"] = contracts / out["volume_max"].replace(0.0, np.nan)
    out["participation_open_interest"] = contracts / out["open_interest"].replace(0.0, np.nan)
    out["breach_optimizer_volume_0_05"] = out["participation_volume_sum"].gt(0.05)
    out["breach_capacity_volume_0_10"] = out["participation_volume_sum"].gt(0.10)
    out["breach_capacity_oi_0_02"] = out["participation_open_interest"].gt(0.02)
    out["validation_scope"] = "entry-day only; not the holding path"
    return out


def run_liquidity_validation() -> pd.DataFrame:
    """Load and validate entry-day volume and open-interest participation."""

    trades = build_trade_table()
    data = _load_audit_frames(DEFAULT_CACHE, VOLUME_PURPOSES | OI_PURPOSES, set(trades["symbol"]))
    return _liquidity_validation(trades, data)


def _intervention_evidence(orders: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["execution_date"] = pd.to_datetime(orders["execution_date"]).dt.normalize()
    q = quotes[quotes["purpose"].eq("vix_intervention_cbbo")].copy()
    if q.empty:
        out = orders.copy()
        out["quote_presence_fraction"] = 0.0
        return out
    q["execution_date"] = _request_dates(q)
    q = q.merge(orders[["execution_date", "symbol"]].drop_duplicates(), on=["execution_date", "symbol"], how="inner")
    _coerce_quote_columns(q)
    q["ts_recv"] = pd.to_datetime(q["ts_recv"], utc=True)
    q["request_start"] = pd.to_datetime(q["request_start"], utc=True)
    q["request_end"] = pd.to_datetime(q["request_end"], utc=True)
    q = q[q["ts_recv"].ge(q["request_start"]) & q["ts_recv"].lt(q["request_end"])].copy()
    q = q[q["bid_px_00"].gt(0) & q["ask_px_00"].ge(q["bid_px_00"])].copy()
    q["mid"] = (q["bid_px_00"] + q["ask_px_00"]) / 2.0
    q["relative_spread"] = (q["ask_px_00"] - q["bid_px_00"]) / q["mid"]
    q["minute"] = q["ts_recv"].dt.floor("min")
    q["session_minutes"] = (q["request_end"] - q["request_start"]).dt.total_seconds() / 60.0
    q["size_weight"] = q["bid_sz_00"].fillna(0.0) + q["ask_sz_00"].fillna(0.0)
    q["weighted_mid"] = q["mid"] * q["size_weight"]
    q["weighted_bid"] = q["bid_px_00"] * q["size_weight"]
    q["weighted_ask"] = q["ask_px_00"] * q["size_weight"]
    keys = ["execution_date", "symbol"]
    stats = q.groupby(keys, observed=True).agg(
        valid_quote_minutes=("minute", "nunique"),
        session_minutes=("session_minutes", "max"),
        median_relative_spread=("relative_spread", "median"),
        weighted_mid_sum=("weighted_mid", "sum"),
        weighted_bid_sum=("weighted_bid", "sum"),
        weighted_ask_sum=("weighted_ask", "sum"),
        size_weight_sum=("size_weight", "sum"),
        quote_mean_mid=("mid", "mean"),
        quote_mean_bid=("bid_px_00", "mean"),
        quote_mean_ask=("ask_px_00", "mean"),
    ).reset_index()
    p90 = q.groupby(keys, observed=True)["relative_spread"].quantile(0.90).rename("p90_relative_spread").reset_index()
    stats = stats.merge(p90, on=keys, how="left")
    first = q.sort_values([*keys, "ts_recv"]).groupby(keys, observed=True).head(1)
    first = first[keys + ["ts_recv", "bid_px_00", "ask_px_00", "mid"]].rename(
        columns={
            "ts_recv": "first_valid_ts",
            "bid_px_00": "first_valid_bid",
            "ask_px_00": "first_valid_ask",
            "mid": "first_valid_mid",
        }
    )
    result = orders.merge(stats, on=keys, how="left").merge(first, on=keys, how="left")
    result["quote_presence_fraction"] = (
        result["valid_quote_minutes"] / result["session_minutes"]
    ).fillna(0.0)
    result["quote_vwap_mid"] = (result["weighted_mid_sum"] / result["size_weight_sum"].replace(0.0, np.nan)).fillna(
        result["quote_mean_mid"]
    )
    result["quote_vwap_bid"] = (result["weighted_bid_sum"] / result["size_weight_sum"].replace(0.0, np.nan)).fillna(
        result["quote_mean_bid"]
    )
    result["quote_vwap_ask"] = (result["weighted_ask_sum"] / result["size_weight_sum"].replace(0.0, np.nan)).fillna(
        result["quote_mean_ask"]
    )
    buy = pd.to_numeric(result["order_contracts"], errors="coerce").gt(0)
    result["first_valid_touch"] = np.where(buy, result["first_valid_ask"], result["first_valid_bid"])
    result["quote_vwap_touch"] = np.where(buy, result["quote_vwap_ask"], result["quote_vwap_bid"])
    result["evidence_only"] = True
    return result.drop(
        columns=[
            "weighted_mid_sum",
            "weighted_bid_sum",
            "weighted_ask_sum",
            "size_weight_sum",
            "quote_mean_mid",
            "quote_mean_bid",
            "quote_mean_ask",
        ]
    )


def run_intervention_evidence() -> pd.DataFrame:
    """Return aggregate executable-quote evidence for intervention orders."""

    orders = pd.read_csv(EVENT_ORDERS)
    quotes = _load_audit_frames(DEFAULT_CACHE, {"vix_intervention_cbbo"}, set(orders["symbol"]))
    return _intervention_evidence(orders, quotes)


def _quantile_records(frame: pd.DataFrame, value: str, groups: list[str]) -> list[dict[str, Any]]:
    clean = frame.dropna(subset=[value])
    if clean.empty:
        return []
    quantiles = clean.groupby(groups, observed=True)[value].quantile([0.25, 0.50, 0.75, 0.90]).unstack()
    quantiles.columns = ["p25", "p50", "p75", "p90"]
    return quantiles.reset_index().to_dict(orient="records")


def _headline_stats(monthly: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    r11 = monthly[monthly["arm"].eq("R1.1")].copy()
    panel = r11.melt(
        id_vars=["config", "return_date", "gross_nav"],
        value_vars=["net_return", "net_return_mid", "net_return_touch", "net_return_worst"],
        var_name="strategy",
        value_name="_net_return",
    )
    panel = panel.rename(columns={"_net_return": "net_return"})
    panel["strategy"] = panel["strategy"].replace(
        {
            "net_return": "modeled",
            "net_return_mid": "mid",
            "net_return_touch": "touch",
            "net_return_worst": "worst",
        }
    )
    panel["window"] = "aligned_2018_2026"
    summary = summarize_aligned(panel)
    records = summary.replace({np.nan: None}).to_dict(orient="records")
    return summary, records


def _spread_comparison(matched: pd.DataFrame) -> pd.DataFrame:
    frame = matched[["arm", "decision_date", "modeled_relative_spread", "obs_rel_spread"]].copy()
    frame["regime"] = np.where(
        pd.to_datetime(frame["decision_date"]).lt(pd.Timestamp("2023-03-28")),
        "pre_2023_03_28",
        "post_2023_03_28",
    )
    long = frame.melt(
        id_vars=["arm", "regime"],
        value_vars=["modeled_relative_spread", "obs_rel_spread"],
        var_name="source",
        value_name="relative_spread",
    )
    long["source"] = long["source"].replace(
        {"modeled_relative_spread": "modeled", "obs_rel_spread": "observed"}
    )
    return long.groupby(["arm", "regime", "source"], observed=True)["relative_spread"].quantile(
        [0.25, 0.50, 0.75, 0.90]
    ).unstack().rename(columns={0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90"}).reset_index()


def _write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frame.to_latex(
            index=False,
            escape=True,
            float_format="%.3f",
            na_rep="--",
            column_format="l" + "r" * max(len(frame.columns) - 1, 0),
        ),
        encoding="utf-8",
    )


def _write_tables(
    table_dir: Path,
    headline: pd.DataFrame,
    spreads: pd.DataFrame,
    coverage: pd.DataFrame,
    liquidity: pd.DataFrame,
) -> None:
    coverage_r11 = coverage[coverage["arm"].eq("R1.1")].set_index("config")
    scenarios = ["modeled", "mid", "touch", "worst"]
    metrics = ["annualized_return", "sortino", "max_drawdown"]
    desired = pd.MultiIndex.from_product([scenarios, metrics], names=["strategy", "metric"])
    wide = headline.pivot(index="config", columns="strategy", values=metrics).swaplevel(0, 1, axis=1)
    wide = wide.reindex(columns=desired)
    wide.columns = [
        "modeled ann.", "modeled Sortino", "modeled maxDD",
        "mid ann.", "mid Sortino", "mid maxDD",
        "touch ann.", "touch Sortino", "touch maxDD",
        "worst ann.", "worst Sortino", "worst maxDD",
    ]
    audit_table = wide.join(
        (100.0 * coverage_r11[["entry_coverage", "roundtrip_coverage"]]).rename(
            columns={
                "entry_coverage": "Entry cov. %",
                "roundtrip_coverage": "Round-trip cov. %",
            }
        )
    ).sort_index().reset_index()
    audit_table = audit_table.rename(columns={"config": "Config"})
    _write_latex_table(audit_table, table_dir / "short_execution_audit_summary.tex")
    _write_spread_and_liquidity_tables(table_dir, spreads, liquidity)


def _write_spread_and_liquidity_tables(
    table_dir: Path,
    spreads: pd.DataFrame,
    liquidity: pd.DataFrame,
) -> None:
    display_arm = {"R1": "Survival", "R1.1": "High Ceiling"}
    spread_table = spreads.copy()
    spread_table["arm"] = spread_table["arm"].replace(display_arm)
    spread_table = spread_table.rename(
        columns={"arm": "Arm", "regime": "Regime", "source": "Source", "p25": "p25", "p50": "p50", "p75": "p75", "p90": "p90"}
    )
    _write_latex_table(spread_table, table_dir / "short_execution_spread_comparison.tex")
    participation = [
        "participation_volume_sum",
        "participation_volume_max",
        "participation_open_interest",
    ]
    liquidity_quantiles = liquidity.groupby("arm", observed=True)[participation].quantile(
        [0.25, 0.50, 0.75, 0.90]
    ).unstack()
    liquidity_quantiles.columns = [
        f"{metric.removeprefix('participation_')}_{label}"
        for metric, label in liquidity_quantiles.columns.set_levels(
            [liquidity_quantiles.columns.levels[0], ["p25", "p50", "p75", "p90"]]
        )
    ]
    breaches = liquidity.groupby("arm", observed=True).agg(
        optimizer_005_breaches=("breach_optimizer_volume_0_05", "sum"),
        volume_010_breaches=("breach_capacity_volume_0_10", "sum"),
        oi_002_breaches=("breach_capacity_oi_0_02", "sum"),
    )
    liquidity_summary = liquidity_quantiles.join(breaches).reset_index()
    liquidity_summary["arm"] = liquidity_summary["arm"].replace(display_arm)
    liquidity_summary = liquidity_summary.rename(columns={"arm": "Arm"})
    _write_latex_table(liquidity_summary, table_dir / "short_liquidity_validation.tex")


def rebuild_tables_from_artifacts(
    artifact_dir: Path = DEFAULT_OUT,
    table_dir: Path = TABLES,
) -> None:
    summary = json.loads((artifact_dir / "execution_audit_summary.json").read_text(encoding="utf-8"))
    spreads = pd.DataFrame(summary["spread_distribution"])
    liquidity = pd.read_csv(artifact_dir / "liquidity_gate_validation.csv")
    _write_spread_and_liquidity_tables(table_dir, spreads, liquidity)


def _summary_payload(
    monthly: pd.DataFrame,
    matched: pd.DataFrame,
    marks: pd.DataFrame,
    liquidity: pd.DataFrame,
    loader_reports: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    headline, headline_records = _headline_stats(monthly)
    spreads = _spread_comparison(matched)
    mark_quantiles = _quantile_records(marks, "relative_error", ["arm", "regime"])
    breach_columns = [
        "breach_optimizer_volume_0_05",
        "breach_capacity_volume_0_10",
        "breach_capacity_oi_0_02",
    ]
    breach = liquidity.groupby("arm", observed=True)[breach_columns].sum().reset_index()
    gap = pd.to_numeric(monthly["cost_reconstruction_gap"], errors="coerce").abs()
    status = "exact" if gap.fillna(0.0).le(1e-6).all() else "approximate"
    diagnostic = matched.copy()
    diagnostic["asset_class_group"] = np.where(
        diagnostic["underlying"].astype(str).str.upper().isin(["VIX", "VX_FRONT"]),
        "vix_option",
        "equity_option",
    )
    diagnostic["absolute_weight"] = pd.to_numeric(diagnostic["weight"], errors="coerce").abs()
    diagnostic["entry_covered_weight"] = diagnostic["absolute_weight"].where(
        diagnostic["obs_coverage"].eq("covered"), 0.0
    )
    diagnostic["open_covered_weight"] = diagnostic["absolute_weight"].where(
        diagnostic["open_obs_coverage"].eq("covered"), 0.0
    )
    diagnostic["exit_covered_weight"] = diagnostic["absolute_weight"].where(
        diagnostic["exit_obs_coverage"].eq("covered"), 0.0
    )
    diagnostic["scenario_covered_weight"] = diagnostic["absolute_weight"].where(
        diagnostic["obs_coverage"].eq("covered")
        & diagnostic["exit_obs_coverage"].eq("covered"),
        0.0,
    )
    coverage = diagnostic.groupby(["arm", "config"], observed=True).agg(
        absolute_weight=("absolute_weight", "sum"),
        entry_covered_weight=("entry_covered_weight", "sum"),
        roundtrip_covered_weight=("scenario_covered_weight", "sum"),
    ).reset_index()
    coverage_denominator = coverage["absolute_weight"].replace(0.0, np.nan)
    coverage["entry_coverage"] = coverage["entry_covered_weight"] / coverage_denominator
    coverage["roundtrip_coverage"] = (
        coverage["roundtrip_covered_weight"] / coverage_denominator
    )
    diagnostic["path_exit_weight"] = diagnostic["absolute_weight"].where(
        diagnostic["exit_obs_source"].eq("held_cbbo_path_last_tradable_session"), 0.0
    )
    diagnostic["unresolved_placeholder"] = ~diagnostic["symbol"].astype(str).str.match(
        r"^[A-Z0-9 ]{6}\d{6}[CP]\d{8}$"
    )
    coverage_detail = diagnostic.groupby(
        ["arm", "config", "asset_class_group"], observed=True
    ).agg(
        positions=("symbol", "size"),
        absolute_weight=("absolute_weight", "sum"),
        entry_covered_weight=("entry_covered_weight", "sum"),
        open_covered_weight=("open_covered_weight", "sum"),
        exit_covered_weight=("exit_covered_weight", "sum"),
        scenario_covered_weight=("scenario_covered_weight", "sum"),
        path_exit_weight=("path_exit_weight", "sum"),
        unresolved_placeholders=("unresolved_placeholder", "sum"),
    ).reset_index()
    denominator = coverage_detail["absolute_weight"].replace(0.0, np.nan)
    coverage_detail["entry_coverage_fraction"] = coverage_detail["entry_covered_weight"] / denominator
    coverage_detail["open_coverage_fraction"] = coverage_detail["open_covered_weight"] / denominator
    coverage_detail["exit_coverage_fraction"] = coverage_detail["exit_covered_weight"] / denominator
    coverage_detail["scenario_coverage_fraction"] = coverage_detail["scenario_covered_weight"] / denominator

    vix = diagnostic[diagnostic["asset_class_group"].eq("vix_option")].copy()
    vix_exact = ~vix["unresolved_placeholder"]
    vix_symbol_alignment = {
        "active_position_rows": int(len(vix)),
        "exact_osi_position_rows": int(vix_exact.sum()),
        "unresolved_placeholder_position_rows": int(vix["unresolved_placeholder"].sum()),
        "exact_osi_unique_symbols": int(vix.loc[vix_exact, "symbol"].nunique()),
        "exact_osi_entry_covered_rows": int(
            (vix_exact & vix["obs_coverage"].eq("covered")).sum()
        ),
        "exact_osi_entry_missing_rows": int(
            (vix_exact & ~vix["obs_coverage"].eq("covered")).sum()
        ),
        "path_exit_covered_rows": int(
            vix["exit_obs_source"].eq("held_cbbo_path_last_tradable_session").sum()
        ),
        "exit_missing_rows": int(vix["exit_obs_coverage"].eq("missing").sum()),
    }

    weights = pd.to_numeric(matched["weight"], errors="coerce")
    absolute_weight = weights.abs()
    is_long = weights.ge(0.0)
    cost_terms = {
        "spread": float((absolute_weight * matched["modeled_relative_spread"]).sum()),
        "fees": float((absolute_weight * matched["modeled_fee_rate"]).sum()),
        "slippage": float((absolute_weight * matched["modeled_slippage_rate"]).sum()),
        "funding": float(
            (
                absolute_weight
                * np.where(
                    is_long,
                    matched["modeled_long_funding_rate"],
                    matched["modeled_short_funding_rate"],
                )
            ).sum()
        ),
        "short_call_borrow": float(
            (absolute_weight * np.where(is_long, 0.0, matched["modeled_short_borrow_rate"])).sum()
        ),
    }
    payload = {
        "headline_r11": headline_records,
        "spread_distribution": spreads.replace({np.nan: None}).to_dict(orient="records"),
        "mark_accuracy": mark_quantiles,
        "coverage": coverage.replace({np.nan: None}).to_dict(orient="records"),
        "coverage_by_asset_class_and_leg": coverage_detail.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "vix_symbol_alignment": vix_symbol_alignment,
        "liquidity_breach_counts": breach.to_dict(orient="records"),
        "cost_reconstruction": {
            "status": status,
            "mean_absolute_gap": float(gap.mean()),
            "max_absolute_gap": float(gap.max()),
            "reconstructed_term_totals": cost_terms,
        },
        "loader_reports": loader_reports,
        "raw_licensed_data_committed": False,
    }
    return payload, headline, spreads, coverage


def _write_readme(out: Path) -> None:
    text = (
        "# Execution audit\n\n"
        "This directory contains aggregate execution evidence only: `execution_fill_ledger.csv` records per-position quote and scenario-cost diagnostics; `execution_audit_monthly_returns.csv` records frozen gross-return text with modeled and observed cost scenarios; `execution_audit_summary.json` records headline, spread, mark, coverage, liquidity, and reconciliation statistics; `liquidity_gate_validation.csv` validates entry-day volume and open-interest participation; `intervention_day_fill_evidence.csv` contains evidence-only intervention-session quote aggregates; and `mark_accuracy.csv` compares modeled marks with observed mids. R1 contract counts are implied at $1M NAV via the R1.1 symbol lookup. For VIX options, zero-row settlement-Wednesday exit requests are supplemented only when the cached holding path supplies a final-tradable-session quote, and the fill ledger labels that source explicitly. Liquidity data exist for decision dates only, so the validation covers entry-day participation, not the holding path. Raw licensed quote rows remain under `data/databento_cache/`; every file here is an aggregate.\n"
    )
    (out / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-intervention", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trades = build_trade_table()
    symbols = set(trades["symbol"].dropna().astype(str))
    base_quotes = _load_audit_frames(args.cache_root, BASE_QUOTE_PURPOSES, symbols)
    vix_symbols = set(
        trades.loc[
            trades["underlying"].astype(str).str.upper().isin(["VIX", "VX_FRONT"]),
            "symbol",
        ].astype(str)
    )
    path_quotes = _load_audit_frames(args.cache_root, {PATH_QUOTE_PURPOSE}, vix_symbols)
    quotes = pd.concat([base_quotes, path_quotes], ignore_index=True, sort=False)
    quotes.attrs = {
        key: int(base_quotes.attrs.get(key, 0)) + int(path_quotes.attrs.get(key, 0))
        for key in ["loaded_files", "skipped_files", "selected_files"]
    }
    matched = match_quotes(trades, quotes)
    monthly, scenario_positions = _recompute_cost_details(matched, _load_monthly())
    marks = mark_accuracy(matched)
    liquidity_data = _load_audit_frames(args.cache_root, VOLUME_PURPOSES | OI_PURPOSES, symbols)
    liquidity = _liquidity_validation(trades, liquidity_data)
    if args.skip_intervention:
        intervention = pd.DataFrame()
        intervention_report = {"loaded_files": 0, "skipped_files": 0, "selected_files": 0}
    else:
        orders = pd.read_csv(EVENT_ORDERS)
        intervention_quotes = _load_audit_frames(
            args.cache_root,
            {"vix_intervention_cbbo"},
            set(orders["symbol"].astype(str)),
        )
        intervention = _intervention_evidence(orders, intervention_quotes)
        intervention_report = {
            key: int(intervention_quotes.attrs.get(key, 0))
            for key in ["loaded_files", "skipped_files", "selected_files"]
        }
    reports = {
        "quotes": {key: int(quotes.attrs.get(key, 0)) for key in ["loaded_files", "skipped_files", "selected_files"]},
        "liquidity": {key: int(liquidity_data.attrs.get(key, 0)) for key in ["loaded_files", "skipped_files", "selected_files"]},
        "intervention": intervention_report,
    }
    summary, headline, spreads, coverage = _summary_payload(
        monthly, matched, marks, liquidity, reports
    )
    scenario_positions.to_csv(args.out / "execution_fill_ledger.csv", index=False)
    monthly.to_csv(args.out / "execution_audit_monthly_returns.csv", index=False)
    marks.to_csv(args.out / "mark_accuracy.csv", index=False)
    liquidity.to_csv(args.out / "liquidity_gate_validation.csv", index=False)
    intervention.to_csv(args.out / "intervention_day_fill_evidence.csv", index=False)
    (args.out / "execution_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_readme(args.out)
    table_dir = TABLES if args.out.resolve() == DEFAULT_OUT.resolve() else args.out
    _write_tables(table_dir, headline, spreads, coverage, liquidity)
    print(json.dumps(summary["cost_reconstruction"], indent=2))


if __name__ == "__main__":
    main()
