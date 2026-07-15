"""Build final visual result summaries for the option-only Markowitz paper.

The figures are deliberately artifact-backed. They read the locked breadth
robustness outputs and the existing baseline performance summary, then write
compact visuals for the conclusion so the paper ends with the actual decision
boundary rather than another dense table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/option_only_markowitz_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"
ROBUSTNESS_DIR = PAPER / "analysis/artifacts/breadth_solutions/robustness"
SUMMARY_DIR = ROBUSTNESS_DIR

CONFIG_ORDER = ["orig", "orig+VIX", "larger", "larger+VIX"]
BROAD_CONFIGS = ["larger", "larger+VIX"]
DISPLAY_MODEL = {
    "E1": "Sharpe Prototype",
    "R1": "Survival Allocator",
    "R1.1": "High Ceiling Allocator",
}
DISPLAY_CONFIG = {
    "orig": "orig",
    "orig+VIX": "orig+VIX",
    "larger": "larger",
    "larger+VIX": "larger+VIX",
}
JOURNAL_COLORS = ["#00552B", "#2F6F9F", "#8B1E3F", "#7A6A2B", "#4C566A", "#A65E2E", "#4F7C45", "#6B4E71"]
PRIMARY_STRATEGY = "E1 capped"
STOCK_MARKOWITZ_STRATEGY = "Stock Markowitz"
BROAD_STOCK_LABEL = "Stock Markowitz (56)"
NAIVE_STRATEGIES = ["Equal premium capped", "Equal risk capped"]
COLORS = {
    "pass": "#00552B",
    "mixed": "#A65E2E",
    "fail": "#9A3412",
    "diagnostic_capacity_infeasible": "#6E7781",
    "naive": "#B9C0C6",
    "stock": "#2F6F9F",
    "realized": "#8B1E3F",
    "interval": "#40534C",
    "exact": "#00552B",
    "proxy": "#A65E2E",
    "orig_line": "#6E7781",
    "orig_vix_line": "#00552B",
    "larger_line": "#A65E2E",
    "larger_vix_line": "#5B4B8A",
}

LINE_COLORS = {
    "orig": COLORS["orig_line"],
    "orig+VIX": COLORS["orig_vix_line"],
    "larger": COLORS["larger_line"],
    "larger+VIX": COLORS["larger_vix_line"],
}

STATUS_CODE = {
    "fail": 0,
    "diagnostic_capacity_infeasible": 0,
    "mixed": 1,
    "pass": 2,
}
ROBUSTNESS_CHECK_ORDER = [
    "Baselines",
    "Capacity",
    "CPCV net (2018+)",
    "CPCV net claim",
    "MC resampled",
    "MC refit",
    "Rolling OOS",
    "PBO rank",
    "Repriced overlay",
]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def _q(series: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return {"p05": np.nan, "p50": np.nan, "p95": np.nan, "n": 0}
    return {
        "p05": float(vals.quantile(0.05)),
        "p50": float(vals.quantile(0.50)),
        "p95": float(vals.quantile(0.95)),
        "n": int(vals.shape[0]),
    }


def _load_underlying_markowitz_sharpe() -> float:
    summary = json.loads(_require(TABLE_DIR / "empirical_summary.json").read_text(encoding="utf-8"))
    for row in summary.get("performance", []):
        if row.get("Strategy") == "Underlying Markowitz":
            return float(row["Sharpe"])
    raise ValueError("Underlying Markowitz Sharpe not found in empirical_summary.json")


def _load_broad_stock_markowitz_sharpes() -> dict[str, float]:
    realized = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_realized_candidate_summary.csv"))
    rows = realized.loc[
        realized["config"].astype(str).isin(BROAD_CONFIGS)
        & realized["strategy"].astype(str).eq(STOCK_MARKOWITZ_STRATEGY)
    ].copy()
    if rows.empty:
        return {}
    rows["net_sharpe"] = pd.to_numeric(rows["net_sharpe"], errors="coerce")
    return {
        str(row["config"]): float(row["net_sharpe"])
        for _, row in rows.dropna(subset=["net_sharpe"]).iterrows()
    }


def build_baseline_scoreboard() -> pd.DataFrame:
    validation = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_validation_summary.csv"))
    realized = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_realized_candidate_summary.csv"))
    stock_sharpe = _load_underlying_markowitz_sharpe()

    rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        primary = validation.loc[
            validation["config"].astype(str).eq(config)
            & validation["strategy"].astype(str).eq(PRIMARY_STRATEGY)
        ]
        if primary.empty:
            raise ValueError(f"Missing primary validation row for {config}")
        primary_row = primary.iloc[0]

        e1_realized = realized.loc[
            realized["config"].astype(str).eq(config)
            & realized["strategy"].astype(str).eq(PRIMARY_STRATEGY)
        ]
        if e1_realized.empty:
            raise ValueError(f"Missing E1 realized row for {config}")
        e1_realized_row = e1_realized.iloc[0]

        naive = realized.loc[
            realized["config"].astype(str).eq(config)
            & realized["strategy"].astype(str).isin(NAIVE_STRATEGIES)
        ].copy()
        if naive.empty:
            raise ValueError(f"Missing capped-naive rows for {config}")
        naive["net_sharpe"] = pd.to_numeric(naive["net_sharpe"], errors="coerce")
        best_naive = naive.sort_values("net_sharpe", ascending=False).iloc[0]

        e1_net = float(primary_row["net_sharpe"])
        naive_net = float(best_naive["net_sharpe"])
        # A book that survives the robustness screens but does not beat its capped-naive
        # baseline has not met the paper's stated bar (beat both baselines), so it is not
        # promoted to a headline "pass" on robustness alone; it is reported as "mixed".
        verdict = str(primary_row["verdict"])
        if verdict == "pass" and not bool(e1_net > naive_net):
            verdict = "mixed"
        rows.append(
            {
                "config": config,
                "config_label": DISPLAY_CONFIG[config],
                "verdict": verdict,
                "deployable": bool(primary_row["deployable"]),
                "e1_net_sharpe": e1_net,
                "e1_net_sortino": float(primary_row["net_sortino"]),
                "e1_gross_sharpe": float(e1_realized_row.get("gross_sharpe", np.nan)),
                "e1_gross_sortino": float(e1_realized_row.get("gross_sortino", np.nan)),
                "rolling_net_sharpe": float(primary_row["rolling_net_sharpe"]),
                "mc_resampled_net_p05": float(primary_row["mc_resampled_net_p05"]),
                "mc_refit_net_p05": float(primary_row["mc_refit_net_p05"]),
                "best_naive_strategy": str(best_naive["strategy"]),
                "best_naive_net_sharpe": naive_net,
                "stock_markowitz_sharpe": stock_sharpe,
                "edge_vs_best_naive": e1_net - naive_net,
                "edge_vs_stock_markowitz": e1_net - stock_sharpe,
                "beats_best_naive": bool(e1_net > naive_net),
                "beats_stock_markowitz": bool(e1_net > stock_sharpe),
            }
        )
    return pd.DataFrame(rows)


def build_validation_distribution_summary() -> pd.DataFrame:
    cpcv = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_cv_cpcv_path_metrics.csv"))
    resampled = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_mc_resampled_paths.csv"))
    refit = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_mc_refit_paths.csv"))

    rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        full_strategy = f"{config} {PRIMARY_STRATEGY}"

        cpcv_slice = cpcv.loc[
            cpcv["strategy"].astype(str).eq(full_strategy)
            & cpcv["basis"].astype(str).eq("full_cost_net")
            & cpcv["status"].astype(str).eq("complete")
        ]
        rows.append({"config": config, "validation": "CPCV complete paths", **_q(cpcv_slice["sharpe"])})

        resampled_slice = resampled.loc[
            resampled["strategy"].astype(str).eq(full_strategy)
            & resampled["basis"].astype(str).eq("full_cost_net")
            & resampled["universe_family"].astype(str).eq("resampled")
        ]
        rows.append({"config": config, "validation": "MC resampled histories", **_q(resampled_slice["sharpe"])})

        refit_slice = refit.loc[
            refit["config"].astype(str).eq(config)
            & refit["display_strategy"].astype(str).eq(PRIMARY_STRATEGY)
            & refit["basis"].astype(str).eq("full_cost_net")
            & refit["status"].astype(str).eq("ok")
        ]
        rows.append({"config": config, "validation": "MC refit stability", **_q(refit_slice["sharpe"])})
    return pd.DataFrame(rows)


def _append_path_values(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    config: str,
    validation: str,
    series: str = "E1",
) -> None:
    values = pd.to_numeric(frame.get("sharpe", pd.Series(dtype=float)), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    for value in values:
        rows.append({"config": config, "validation": validation, "series": series, "sharpe": float(value)})


def _read_optional_claim_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def build_validation_path_values() -> pd.DataFrame:
    cpcv = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_cv_cpcv_path_metrics.csv"))
    claim = _read_optional_claim_metrics(ROBUSTNESS_DIR / "breadth_cv_claim_cpcv_path_metrics.csv")
    resampled = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_mc_resampled_paths.csv"))
    refit = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_mc_refit_paths.csv"))
    claim_slices: dict[str, pd.DataFrame] = {}
    claim_complete = not claim.empty and {"strategy", "basis", "status", "sharpe"}.issubset(claim.columns)
    if claim_complete:
        for config in CONFIG_ORDER:
            full_strategy = f"{config} {PRIMARY_STRATEGY}"
            claim_slice = claim.loc[
                claim["strategy"].astype(str).eq(full_strategy)
                & claim["basis"].astype(str).eq("full_cost_net")
                & claim["status"].astype(str).eq("complete")
            ]
            claim_values = (
                pd.to_numeric(claim_slice["sharpe"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(claim_values) != 11:
                claim_complete = False
                claim_slices = {}
                break
            claim_slices[config] = claim_slice

    rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        full_strategy = f"{config} {PRIMARY_STRATEGY}"

        cpcv_slice = cpcv.loc[
            cpcv["strategy"].astype(str).eq(full_strategy)
            & cpcv["basis"].astype(str).eq("full_cost_net")
            & cpcv["status"].astype(str).eq("complete")
        ]
        _append_path_values(rows, cpcv_slice, config=config, validation="CPCV complete paths")

        if claim_complete:
            _append_path_values(rows, claim_slices[config], config=config, validation="CPCV claim window")

        resampled_slice = resampled.loc[
            resampled["strategy"].astype(str).eq(full_strategy)
            & resampled["basis"].astype(str).eq("full_cost_net")
            & resampled["universe_family"].astype(str).eq("resampled")
        ]
        _append_path_values(rows, resampled_slice, config=config, validation="MC resampled histories")

        refit_slice = refit.loc[
            refit["config"].astype(str).eq(config)
            & refit["display_strategy"].astype(str).eq(PRIMARY_STRATEGY)
            & refit["basis"].astype(str).eq("full_cost_net")
            & refit["status"].astype(str).eq("ok")
        ]
        _append_path_values(rows, refit_slice, config=config, validation="MC refit stability")

        if config in BROAD_CONFIGS:
            stock_strategy = f"{config} {STOCK_MARKOWITZ_STRATEGY}"
            stock_cpcv_slice = cpcv.loc[
                cpcv["strategy"].astype(str).eq(stock_strategy)
                & cpcv["basis"].astype(str).eq("full_cost_net")
                & cpcv["status"].astype(str).eq("complete")
            ]
            _append_path_values(
                rows,
                stock_cpcv_slice,
                config=config,
                validation="CPCV complete paths",
                series=STOCK_MARKOWITZ_STRATEGY,
            )

            if claim_complete:
                stock_claim_slice = claim.loc[
                    claim["strategy"].astype(str).eq(stock_strategy)
                    & claim["basis"].astype(str).eq("full_cost_net")
                    & claim["status"].astype(str).eq("complete")
                ]
                _append_path_values(
                    rows,
                    stock_claim_slice,
                    config=config,
                    validation="CPCV claim window",
                    series=STOCK_MARKOWITZ_STRATEGY,
                )

            stock_resampled_slice = resampled.loc[
                resampled["strategy"].astype(str).eq(stock_strategy)
                & resampled["basis"].astype(str).eq("full_cost_net")
                & resampled["universe_family"].astype(str).eq("resampled")
            ]
            _append_path_values(
                rows,
                stock_resampled_slice,
                config=config,
                validation="MC resampled histories",
                series=STOCK_MARKOWITZ_STRATEGY,
            )

            stock_refit_slice = refit.loc[
                refit["config"].astype(str).eq(config)
                & refit["display_strategy"].astype(str).eq(STOCK_MARKOWITZ_STRATEGY)
                & refit["basis"].astype(str).eq("full_cost_net")
                & refit["status"].astype(str).eq("ok")
            ]
            _append_path_values(
                rows,
                stock_refit_slice,
                config=config,
                validation="MC refit stability",
                series=STOCK_MARKOWITZ_STRATEGY,
            )
    return pd.DataFrame(rows, columns=["config", "validation", "series", "sharpe"])


def build_walk_forward_return_paths(scoreboard: pd.DataFrame) -> pd.DataFrame:
    rolling = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_rolling_oos.csv"), parse_dates=["return_date"])
    stock = pd.read_csv(_require(PAPER / "artifacts/strategy_returns_post_cost.csv"), parse_dates=["snap_date"])
    stock = stock[["snap_date", "Underlying Markowitz"]].rename(
        columns={"snap_date": "return_date", "Underlying Markowitz": "return"}
    )
    rolling_return_dates = pd.Series(
        pd.to_datetime(rolling["return_date"]).dt.normalize().dropna().unique()
    ).sort_values(ignore_index=True)
    rolling_dates = set(rolling_return_dates)
    stock = stock.loc[pd.to_datetime(stock["return_date"]).dt.normalize().isin(rolling_dates)]

    rows: list[pd.DataFrame] = []
    stock_path = stock.sort_values("return_date").copy()
    stock_path["config"] = "stock"
    stock_path["config_label"] = "Stock Markowitz"
    stock_path["family"] = "Stock baseline"
    stock_path["strategy"] = "Underlying Markowitz"
    rows.append(stock_path[["return_date", "config", "config_label", "family", "strategy", "return"]])

    breadth_returns_path = ROBUSTNESS_DIR / "breadth_strategy_returns_net.csv"
    broad_stock_col = "larger+VIX Stock Markowitz"
    if breadth_returns_path.exists():
        breadth_returns = pd.read_csv(breadth_returns_path)
        if broad_stock_col not in breadth_returns.columns:
            print(
                f"WARNING: skipping {BROAD_STOCK_LABEL} walk-forward line; "
                f"missing column {broad_stock_col!r} in {breadth_returns_path.name}"
            )
        elif len(rolling_return_dates) != 60 or len(breadth_returns) != len(rolling_return_dates):
            print(
                f"WARNING: skipping {BROAD_STOCK_LABEL} walk-forward line; "
                f"cannot positionally align {len(breadth_returns)} return rows "
                f"to {len(rolling_return_dates)} rolling dates"
            )
        else:
            broad_returns = pd.to_numeric(breadth_returns[broad_stock_col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if broad_returns.isna().any():
                print(
                    f"WARNING: skipping {BROAD_STOCK_LABEL} walk-forward line; "
                    f"{broad_stock_col!r} contains non-numeric or missing returns"
                )
            else:
                broad_stock_path = pd.DataFrame({"return_date": rolling_return_dates, "return": broad_returns})
                broad_stock_path = broad_stock_path.loc[
                    pd.to_datetime(broad_stock_path["return_date"]).dt.normalize().isin(rolling_dates)
                ].copy()
                broad_stock_path["config"] = "stock_56"
                broad_stock_path["config_label"] = BROAD_STOCK_LABEL
                broad_stock_path["family"] = "Stock baseline"
                broad_stock_path["strategy"] = broad_stock_col
                rows.append(
                    broad_stock_path[["return_date", "config", "config_label", "family", "strategy", "return"]]
                )
    else:
        print(
            f"WARNING: skipping {BROAD_STOCK_LABEL} walk-forward line; "
            f"missing {breadth_returns_path}"
        )

    best_naive_by_config = dict(zip(scoreboard["config"], scoreboard["best_naive_strategy"]))
    for config in CONFIG_ORDER:
        config_roll = rolling.loc[rolling["config"].astype(str).eq(config)].copy()
        for family, display_strategy, label_suffix in [
            ("Locked E1", PRIMARY_STRATEGY, "E1"),
            ("Matched capped naive", best_naive_by_config[config], "naive"),
        ]:
            sub = config_roll.loc[config_roll["display_strategy"].astype(str).eq(display_strategy)].copy()
            if sub.empty:
                raise ValueError(f"Missing rolling OOS path for {config} {display_strategy}")
            sub = sub.sort_values("return_date")
            sub["return"] = pd.to_numeric(sub["net_ret"], errors="coerce")
            sub["config_label"] = f"{DISPLAY_CONFIG[config]} {label_suffix}"
            sub["family"] = family
            rows.append(sub[["return_date", "config", "config_label", "family", "strategy", "return"]])

    paths = pd.concat(rows, ignore_index=True)
    paths = paths.sort_values(["family", "config", "return_date"])
    paths["return"] = pd.to_numeric(paths["return"], errors="coerce")
    paths["gross_growth"] = 1.0 + paths["return"]
    if (paths["gross_growth"] <= 0).any():
        bad = paths.loc[paths["gross_growth"] <= 0, ["config_label", "return_date", "return"]]
        raise ValueError(f"Walk-forward path has non-positive gross growth rows: {bad.head().to_dict(orient='records')}")
    paths["wealth"] = paths.groupby(["family", "config_label"], sort=False)["gross_growth"].cumprod()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    paths.to_csv(SUMMARY_DIR / "final_walk_forward_return_paths.csv", index=False)
    return paths


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda x: "yes" if bool(x) else "no")
    path.write_text(out.to_latex(index=False, escape=False, float_format="%.3f"), encoding="utf-8")


def _status_from_tail(p05: float, p50: float) -> str:
    if np.isfinite(p05) and p05 > 0.0:
        return "pass"
    if np.isfinite(p50) and p50 > 0.0:
        return "mixed"
    return "fail"


def build_short_robustness_matrix(scoreboard: pd.DataFrame) -> pd.DataFrame:
    validation = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_validation_summary.csv"))
    pbo = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_cv_pbo_summary.csv"))
    score_configs = [str(config) for config in scoreboard["config"]]
    claim_cols = {"cpcv_claim_net_p05", "cpcv_claim_net_p50"}
    claim_available = claim_cols.issubset(validation.columns)
    if claim_available:
        for config in score_configs:
            claim_row = validation.loc[
                validation["config"].astype(str).eq(config)
                & validation["strategy"].astype(str).eq(PRIMARY_STRATEGY)
            ]
            if claim_row.empty:
                claim_available = False
                break
            p05 = pd.to_numeric(claim_row["cpcv_claim_net_p05"], errors="coerce").iloc[0]
            p50 = pd.to_numeric(claim_row["cpcv_claim_net_p50"], errors="coerce").iloc[0]
            if not (np.isfinite(float(p05)) and np.isfinite(float(p50))):
                claim_available = False
                break

    rows: list[dict[str, object]] = []
    for _, score in scoreboard.iterrows():
        config = str(score["config"])
        val = validation.loc[
            validation["config"].astype(str).eq(config)
            & validation["strategy"].astype(str).eq(PRIMARY_STRATEGY)
        ].iloc[0]
        pbo_row = pbo.loc[
            pbo["scope"].astype(str).eq("within_config")
            & pbo["config"].astype(str).eq(config)
            & pbo["Basis"].astype(str).eq("full_cost_net")
        ]
        pbo_value = float(pbo_row["PBO"].iloc[0]) if not pbo_row.empty else np.nan

        if not bool(score["deployable"]):
            baseline_status = "diagnostic_capacity_infeasible"
        elif bool(score["beats_best_naive"]) and bool(score["beats_stock_markowitz"]):
            baseline_status = "pass"
        elif float(score["e1_net_sharpe"]) > 0.0:
            baseline_status = "mixed"
        else:
            baseline_status = "fail"

        checks = [
            ("Baselines", baseline_status, float(score["e1_net_sharpe"])),
            ("Capacity", "pass" if bool(score["deployable"]) and float(val["sum_of_caps"]) >= 1.0 else "fail", float(val["sum_of_caps"])),
            ("CPCV net (2018+)", _status_from_tail(float(val["cpcv_net_p05"]), float(val["cpcv_net_p50"])), float(val["cpcv_net_p50"])),
            ("MC resampled", _status_from_tail(float(val["mc_resampled_net_p05"]), float(val["mc_resampled_net_p50"])), float(val["mc_resampled_net_p05"])),
            ("MC refit", _status_from_tail(float(val["mc_refit_net_p05"]), float(val["mc_refit_net_p50"])), float(val["mc_refit_net_p05"])),
            ("Rolling OOS", "pass" if float(val["rolling_net_sharpe"]) > 0.0 else "fail", float(val["rolling_net_sharpe"])),
            ("PBO rank", "pass" if pbo_value <= 0.30 else ("mixed" if pbo_value <= 0.50 else "fail"), pbo_value),
            ("Repriced overlay", _status_from_tail(float(val["repriced_net_overlay_p05"]), float(val["repriced_net_overlay_p50"])), float(val["repriced_net_overlay_p50"])),
        ]
        if claim_available:
            checks.insert(
                3,
                (
                    "CPCV net claim",
                    _status_from_tail(float(val["cpcv_claim_net_p05"]), float(val["cpcv_claim_net_p50"])),
                    float(val["cpcv_claim_net_p50"]),
                ),
            )
        for check, status, value in checks:
            rows.append(
                {
                    "config": config,
                    "config_label": DISPLAY_CONFIG[config],
                    "check": check,
                    "status": status,
                    "score": STATUS_CODE.get(status, 0),
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def build_short_spread_summary() -> pd.DataFrame:
    spread = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_spread_source_coverage.csv"))
    rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        subset = spread.loc[spread["config"].astype(str).eq(config)].copy()
        exact_rows = int(subset.loc[subset["relative_spread_source"].eq("panel_cbbo"), "rows"].sum())
        proxy_rows = int(subset.loc[subset["relative_spread_source"].eq("inferred_cbbo_proxy"), "rows"].sum())
        total_rows = exact_rows + proxy_rows
        proxy_share = proxy_rows / total_rows if total_rows else 0.0
        medians = pd.to_numeric(subset["median_relative_spread"], errors="coerce").dropna()
        rows.append(
            {
                "config": config,
                "config_label": DISPLAY_CONFIG[config],
                "exact_cbbo_rows": exact_rows,
                "proxy_cbbo_rows": proxy_rows,
                "proxy_row_share": proxy_share,
                "median_relative_spread": float(medians.median()) if not medians.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_short_tables(scoreboard: pd.DataFrame, robustness: pd.DataFrame, spread: pd.DataFrame) -> None:
    scenario = scoreboard.merge(spread, on=["config", "config_label"], how="left")
    scenario["Universe"] = scenario["config"].map(
        {
            "orig": "8 equities",
            "orig+VIX": "8 equities + VIX",
            "larger": "56 equities",
            "larger+VIX": "56 equities + VIX",
        }
    )
    scenario["Spread input"] = scenario.apply(
        lambda r: "exact panel CBBO" if int(r["proxy_cbbo_rows"]) == 0 else "exact panel CBBO + inferred CBBO proxy",
        axis=1,
    )
    scenario["Status"] = scenario["verdict"].map(
        {
            "pass": "research candidate",
            "mixed": "mixed",
            "diagnostic_capacity_infeasible": "capacity diagnostic",
        }
    ).fillna(scenario["verdict"])
    scenario_table = scenario[
        ["config_label", "Universe", "Spread input", "Status", "e1_net_sharpe", "best_naive_net_sharpe", "stock_markowitz_sharpe"]
    ].rename(
        columns={
            "config_label": "Config",
            "e1_net_sharpe": "Prototype net",
            "best_naive_net_sharpe": "Naive net",
            "stock_markowitz_sharpe": "Stock Markowitz",
        }
    )
    _write_latex_table(scenario_table, TABLE_DIR / "short_four_scenario_assumptions.tex")

    spread_table = spread.copy()
    spread_table["Proxy share"] = spread_table["proxy_row_share"]
    spread_table = spread_table[
        ["config_label", "exact_cbbo_rows", "proxy_cbbo_rows", "Proxy share", "median_relative_spread"]
    ].rename(
        columns={
            "config_label": "Config",
            "exact_cbbo_rows": "Exact CBBO rows",
            "proxy_cbbo_rows": "Proxy rows",
            "median_relative_spread": "Median spread",
        }
    )
    _write_latex_table(spread_table, TABLE_DIR / "short_spread_source_ladder.tex")

    robust_wide = robustness.pivot(index="config_label", columns="check", values="status").reset_index()
    robust_wide["config_order"] = robust_wide["config_label"].map(
        {DISPLAY_CONFIG[config]: i for i, config in enumerate(CONFIG_ORDER)}
    )
    robust_wide = robust_wide.sort_values("config_order").drop(columns=["config_order"])
    robust_wide = robust_wide.rename(columns={"config_label": "Config"})
    for col in robust_wide.columns:
        if col != "Config":
            robust_wide[col] = robust_wide[col].replace(
                {"diagnostic_capacity_infeasible": "diagnostic"}
            )
    ordered_cols = ["Config"] + [check for check in ROBUSTNESS_CHECK_ORDER if check in robust_wide.columns]
    _write_latex_table(robust_wide[ordered_cols], TABLE_DIR / "short_robustness_summary.tex")

    compact = scoreboard[
        [
            "config_label",
            "verdict",
            "e1_gross_sharpe",
            "e1_net_sharpe",
            "e1_net_sortino",
            "best_naive_net_sharpe",
            "stock_markowitz_sharpe",
            "rolling_net_sharpe",
            "mc_resampled_net_p05",
            "mc_refit_net_p05",
        ]
    ].rename(
        columns={
            "config_label": "Config",
            "verdict": "Verdict",
            "e1_gross_sharpe": "Gross Sharpe",
            "e1_net_sharpe": "Net Sharpe",
            "e1_net_sortino": "Net Sortino",
            "best_naive_net_sharpe": "Naive Sharpe",
            "stock_markowitz_sharpe": "Stock Sharpe",
            "rolling_net_sharpe": "Rolling Sharpe",
            "mc_resampled_net_p05": "MC resample p05",
            "mc_refit_net_p05": "MC refit p05",
        }
    )
    compact["Verdict"] = compact["Verdict"].map(
        {
            "pass": "pass",
            "mixed": "mixed",
            "diagnostic_capacity_infeasible": "capacity diagnostic",
        }
    ).fillna(compact["Verdict"])
    _write_latex_table(compact, TABLE_DIR / "short_final_scoreboard.tex")


def write_scoreboard_tables(scoreboard: pd.DataFrame, distributions: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(SUMMARY_DIR / "final_result_scoreboard.csv", index=False)
    distributions.to_csv(SUMMARY_DIR / "final_validation_distribution_summary.csv", index=False)

    compact = scoreboard[
        [
            "config_label",
            "verdict",
            "e1_net_sharpe",
            "best_naive_net_sharpe",
            "stock_markowitz_sharpe",
            "rolling_net_sharpe",
            "edge_vs_best_naive",
            "edge_vs_stock_markowitz",
        ]
    ].rename(
        columns={
            "config_label": "Config",
            "verdict": "Verdict",
            "e1_net_sharpe": "E1 net Sharpe",
            "best_naive_net_sharpe": "Best naive net",
            "stock_markowitz_sharpe": "Stock Markowitz",
            "rolling_net_sharpe": "Rolling net",
            "edge_vs_best_naive": "Edge vs naive",
            "edge_vs_stock_markowitz": "Edge vs stock",
        }
    )
    _write_latex_table(compact, TABLE_DIR / "final_result_scoreboard.tex")


def plot_validation_distributions(path_values: pd.DataFrame, scoreboard: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = path_values.copy()
    if "series" not in frame.columns:
        frame["series"] = "E1"
    frame["series"] = frame["series"].fillna("E1").astype(str)
    frame["sharpe"] = pd.to_numeric(frame.get("sharpe", pd.Series(dtype=float)), errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["sharpe"])
    panel_order = [
        "CPCV complete paths",
        "CPCV claim window",
        "MC resampled histories",
        "MC refit stability",
    ]
    validations = [name for name in panel_order if name in set(frame["validation"].astype(str))]
    if not validations:
        validations = ["CPCV complete paths"]
    y_labels = [DISPLAY_CONFIG[c] for c in CONFIG_ORDER]
    y_positions = np.arange(len(CONFIG_ORDER))
    realized_by_config = dict(zip(scoreboard["config"], scoreboard["e1_net_sharpe"]))
    has_stock_series = frame["series"].eq(STOCK_MARKOWITZ_STRATEGY).any()

    def _series_slot(subset: pd.DataFrame, config: str, series: str, y: float) -> tuple[float, float]:
        stock_values = subset.loc[
            subset["config"].astype(str).eq(config)
            & subset["series"].eq(STOCK_MARKOWITZ_STRATEGY),
            "sharpe",
        ].dropna()
        if config in BROAD_CONFIGS and not stock_values.empty:
            return (float(y) + 0.18, 0.34) if series == STOCK_MARKOWITZ_STRATEGY else (float(y) - 0.18, 0.34)
        return float(y), 0.72

    fig, axes_raw = plt.subplots(1, len(validations), figsize=(11.2, 4.25), sharex=True, sharey=True, squeeze=False)
    axes = axes_raw[0]
    all_vals: list[float] = [0.0]
    all_vals.extend(frame["sharpe"].to_numpy(float).tolist())
    all_vals.extend(float(v) for v in realized_by_config.values() if np.isfinite(float(v)))
    for ax, validation in zip(axes, validations):
        subset = frame.loc[frame["validation"].astype(str).eq(validation)]
        positions: list[float] = []
        widths: list[float] = []
        data: list[np.ndarray] = []
        colors: list[str] = []
        alphas: list[float] = []
        for y, config in zip(y_positions, CONFIG_ORDER):
            for series in ["E1", STOCK_MARKOWITZ_STRATEGY]:
                if series == STOCK_MARKOWITZ_STRATEGY and config not in BROAD_CONFIGS:
                    continue
                values = subset.loc[
                    subset["config"].astype(str).eq(config)
                    & subset["series"].eq(series),
                    "sharpe",
                ].dropna()
                if values.empty:
                    continue
                arr = values.to_numpy(float)
                y_slot, width = _series_slot(subset, config, series, float(y))
                positions.append(y_slot)
                widths.append(width)
                data.append(arr)
                colors.append(COLORS["stock"] if series == STOCK_MARKOWITZ_STRATEGY else LINE_COLORS.get(config, COLORS["interval"]))
                alphas.append(0.36 if series == STOCK_MARKOWITZ_STRATEGY else 0.45)
        if data:
            parts = ax.violinplot(
                data,
                positions=positions,
                orientation="horizontal",
                widths=widths,
                showmedians=True,
                showextrema=False,
            )
            for body, color, alpha in zip(parts["bodies"], colors, alphas):
                body.set_facecolor(color)
                body.set_edgecolor("#263238")
                body.set_alpha(alpha)
                body.set_linewidth(0.8)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#202020")
                parts["cmedians"].set_linewidth(1.1)
        if validation.startswith("CPCV"):
            for y, config in zip(y_positions, CONFIG_ORDER):
                for series in ["E1", STOCK_MARKOWITZ_STRATEGY]:
                    if series == STOCK_MARKOWITZ_STRATEGY and config not in BROAD_CONFIGS:
                        continue
                    values = subset.loc[
                        subset["config"].astype(str).eq(config)
                        & subset["series"].eq(series),
                        "sharpe",
                    ].dropna()
                    if values.empty:
                        continue
                    arr = values.to_numpy(float)
                    y_slot, width = _series_slot(subset, config, series, float(y))
                    span = 0.22 if width > 0.5 else 0.10
                    offsets = np.linspace(-span, span, len(arr)) if len(arr) > 1 else np.array([0.0])
                    ax.scatter(
                        arr,
                        np.full(len(arr), y_slot, dtype=float) + offsets,
                        s=14,
                        color="#202020",
                        alpha=0.68,
                        linewidths=0,
                        zorder=4,
                    )
        for y, config in zip(y_positions, CONFIG_ORDER):
            realized = float(realized_by_config.get(config, np.nan))
            if np.isfinite(realized):
                y_slot, _ = _series_slot(subset, config, "E1", float(y))
                ax.scatter(
                    realized,
                    y_slot,
                    s=50,
                    marker="D",
                    color=COLORS["realized"],
                    edgecolor="#202020",
                    linewidth=0.5,
                    zorder=5,
                )
        ax.axvline(0.0, color="#555", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_title(validation, fontsize=10, pad=8)
        ax.set_xlabel("Full-cost net Sharpe")
        ax.grid(True, axis="x", alpha=0.20, linewidth=0.7)
        ax.grid(True, axis="y", alpha=0.06, linewidth=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(len(CONFIG_ORDER) - 0.5, -0.5)

    finite = np.array([x for x in all_vals if np.isfinite(x)], dtype=float)
    if finite.size >= 3:
        xmin, xmax = np.quantile(finite, [0.01, 0.99])
    elif finite.size:
        xmin, xmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        xmin, xmax = -1.0, 1.0
    xmin = min(float(xmin), 0.0)
    xmax = max(float(xmax), 0.0)
    if xmax <= xmin:
        xmax = xmin + 1.0
    pad = max(0.12, 0.06 * (xmax - xmin))
    for ax in axes:
        ax.set_xlim(xmin - pad, xmax + pad)

    median_handle = plt.Line2D([0], [0], color="#202020", linewidth=1.1, label="median")
    realized_handle = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor=COLORS["realized"],
        markeredgecolor="#202020",
        markersize=6,
        label="realized static book",
    )
    path_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#202020",
        markeredgewidth=0,
        markersize=4,
        alpha=0.68,
        label="CPCV path Sharpe",
    )
    handles = [median_handle, realized_handle, path_handle]
    ncol = 3
    if has_stock_series:
        e1_handle = plt.Line2D(
            [0],
            [0],
            color=COLORS["interval"],
            linewidth=6,
            alpha=0.45,
            label=DISPLAY_MODEL["E1"],
        )
        stock_handle = plt.Line2D(
            [0],
            [0],
            color=COLORS["stock"],
            linewidth=6,
            alpha=0.36,
            label=BROAD_STOCK_LABEL,
        )
        handles = [e1_handle, stock_handle, *handles]
        ncol = 5
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        fontsize=8,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.savefig(FIG_DIR / "final_breadth_validation_distributions.pdf")
    plt.savefig(FIG_DIR / "short_validation_distributions.pdf")
    plt.close(fig)


def plot_baseline_comparison(scoreboard: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(scoreboard))
    width = 0.34
    e1_colors = [COLORS.get(v, COLORS["interval"]) for v in scoreboard["verdict"]]
    stock = float(scoreboard["stock_markowitz_sharpe"].iloc[0])
    broad_stock_sharpes = _load_broad_stock_markowitz_sharpes()

    fig, ax = plt.subplots(figsize=(8.8, 4.45))
    ax.bar(
        x - width / 2,
        scoreboard["e1_net_sharpe"].to_numpy(float),
        width,
        label=f"{DISPLAY_MODEL['E1']} development book",
        color=e1_colors,
        edgecolor="#263238",
        linewidth=0.55,
    )
    ax.bar(
        x + width / 2,
        scoreboard["best_naive_net_sharpe"].to_numpy(float),
        width,
        label="Best capped naive option book",
        color=COLORS["naive"],
        edgecolor="#263238",
        linewidth=0.55,
    )
    broad_positions: list[float] = []
    broad_values: list[float] = []
    for i, row in scoreboard.iterrows():
        config = str(row["config"])
        if config not in broad_stock_sharpes:
            continue
        broad_positions.append(float(i) + 1.45 * width)
        broad_values.append(float(broad_stock_sharpes[config]))
    if broad_values:
        ax.bar(
            broad_positions,
            broad_values,
            width * 0.78,
            label=BROAD_STOCK_LABEL,
            color=COLORS["stock"],
            edgecolor="#263238",
            linewidth=0.55,
            hatch="//",
            alpha=0.78,
        )
    ax.axhline(stock, color=COLORS["stock"], linewidth=2.0, linestyle="-.", label=f"Underlying Markowitz ({stock:.3f})")
    ax.axhline(0.0, color="#555", linewidth=0.9, linestyle="--", alpha=0.65)
    for i, row in scoreboard.iterrows():
        val = float(row["e1_net_sharpe"])
        if bool(row["beats_best_naive"]) and bool(row["beats_stock_markowitz"]):
            label = "beats both"
            y_text = val + 0.06
            va = "bottom"
        elif str(row["verdict"]) == "diagnostic_capacity_infeasible":
            label = "cap infeasible"
            y_text = val - 0.08
            va = "top"
        else:
            label = str(row["verdict"]).replace("_", " ")
            y_text = val + 0.06
            va = "bottom"
        ax.text(i - width / 2, y_text, label, ha="center", va=va, fontsize=7.2, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(scoreboard["config_label"].tolist())
    ax.set_ylabel("Full-cost net Sharpe")
    ax.set_title("Final baseline comparison at $1M NAV")
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4 if broad_values else 3, fontsize=8, frameon=False)
    plotted_values = scoreboard[["e1_net_sharpe", "best_naive_net_sharpe"]].to_numpy(float).ravel().tolist()
    plotted_values.extend(broad_values)
    finite_values = [float(v) for v in plotted_values if np.isfinite(float(v))]
    ymin = min(-0.25, min(finite_values) - 0.15)
    ymax = max(1.75, max(finite_values) + 0.28, stock + 0.25)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(FIG_DIR / "final_baseline_comparison.pdf")
    plt.savefig(FIG_DIR / "short_four_variant_scoreboard.pdf")
    plt.close(fig)


def plot_short_theory_flow() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 3.6))
    ax.axis("off")

    boxes = [
        ("1. Listed option", "premium, payoff,\nsettlement, Greeks"),
        ("2. Cashflow map", "NAV return,\nconditional premium"),
        ("3. Joint risk model", "$\\Sigma_O=[B\\ I]\\Sigma_{f,\\varepsilon}[B\\ I]^\\top$"),
        ("4. Survival Allocator", "net mean - variance\nCVaR, stress, margin, cash"),
        ("5. Survival first", "integer feasibility,\nwalk-forward, ruin gate"),
    ]
    xs = np.linspace(0.015, 0.795, len(boxes))
    w, h = 0.19, 0.56
    for i, ((title, body), x) in enumerate(zip(boxes, xs)):
        rect = FancyBboxPatch(
            (x, 0.24),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#263238",
            facecolor="#E8EEE8" if i in {0, 3} else "#F6F8F7",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.66, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#00552B")
        ax.text(x + w / 2, 0.42, body, ha="center", va="center", fontsize=9, color="#1F2933")
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x + w + 0.013, 0.52),
                xytext=(xs[i + 1] - 0.013, 0.52),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#40534C"},
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "short_theory_flow.pdf")
    plt.close(fig)


def plot_short_robustness_heatmap(robustness: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    available_checks = set(robustness["check"].astype(str))
    checks = [check for check in ROBUSTNESS_CHECK_ORDER if check in available_checks]
    pivot = robustness.pivot(index="config_label", columns="check", values="score").loc[
        [DISPLAY_CONFIG[c] for c in CONFIG_ORDER], checks
    ]
    labels = robustness.pivot(index="config_label", columns="check", values="status").loc[
        [DISPLAY_CONFIG[c] for c in CONFIG_ORDER], checks
    ]

    cmap = matplotlib.colors.ListedColormap(["#9A3412", "#A65E2E", "#00552B"])
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    ax.imshow(pivot.to_numpy(float), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(checks)))
    ax.set_xticklabels(checks, rotation=24, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Robustness summary by test")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(labels.iloc[i, j]).replace("diagnostic_capacity_infeasible", "diagnostic"), ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xticks(np.arange(-0.5, len(checks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "short_robustness_heatmap.pdf")
    plt.close(fig)


def plot_short_capacity_spread_panel(scoreboard: pd.DataFrame, spread: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_validation_summary.csv"))
    validation = validation.loc[validation["strategy"].astype(str).eq(PRIMARY_STRATEGY)]
    merged = scoreboard.merge(validation[["config", "sum_of_caps", "deployed_gross"]], on="config", how="left").merge(
        spread, on=["config", "config_label"], how="left"
    )
    x = np.arange(len(merged))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    ax = axes[0]
    ax.bar(x - 0.18, merged["sum_of_caps"].to_numpy(float), 0.36, color="#00552B", label="sum of liquidity caps")
    ax.bar(x + 0.18, merged["deployed_gross"].to_numpy(float), 0.36, color="#8B1E3F", label="deployed gross")
    ax.axhline(1.0, color="#263238", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["config_label"].tolist())
    ax.set_ylabel("NAV units")
    ax.set_title("Capacity at $1M NAV")
    ax.grid(True, axis="y", alpha=0.20)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    totals = (merged["exact_cbbo_rows"] + merged["proxy_cbbo_rows"]).replace(0, np.nan)
    exact_share = merged["exact_cbbo_rows"] / totals
    proxy_share = merged["proxy_cbbo_rows"] / totals
    ax.barh(x, exact_share, color=COLORS["exact"], label="historical panel CBBO")
    ax.barh(x, proxy_share, left=exact_share, color=COLORS["proxy"], label="inferred CBBO proxy")
    for i, row in merged.iterrows():
        med = float(row["median_relative_spread"]) * 100.0
        ax.text(1.02, i, f"median {med:.2f}%", va="center", fontsize=7.6, clip_on=False)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks(x)
    ax.set_yticklabels(merged["config_label"].tolist())
    ax.set_xlabel("Cost rows by spread source")
    ax.set_title("Spread-source coverage")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(True, axis="x", alpha=0.16)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "short_capacity_spread_panel.pdf")
    plt.close(fig)


def plot_walk_forward_return_paths(paths: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = paths.copy()
    paths["return_date"] = pd.to_datetime(paths["return_date"])
    stock = paths.loc[paths["family"].eq("Stock baseline")].sort_values("return_date")

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.9), sharex=True)
    panel_specs = [
        (axes[0], "Sharpe Prototype variants", "Locked E1"),
        (axes[1], "Matched capped-naive option baselines", "Matched capped naive"),
    ]
    label_offsets = {
        ("Locked E1", "orig"): 1.26,
        ("Locked E1", "orig+VIX"): 1.00,
        ("Locked E1", "larger"): 0.82,
        ("Locked E1", "larger+VIX"): 1.00,
        ("Matched capped naive", "orig"): 1.32,
        ("Matched capped naive", "orig+VIX"): 1.65,
        ("Matched capped naive", "larger"): 0.82,
        ("Matched capped naive", "larger+VIX"): 0.86,
    }
    stock_offsets = {
        ("Locked E1", "Stock Markowitz"): 0.82,
        ("Locked E1", BROAD_STOCK_LABEL): 1.04,
        ("Matched capped naive", "Stock Markowitz"): 1.22,
        ("Matched capped naive", BROAD_STOCK_LABEL): 0.92,
    }
    stock_styles = {
        "Stock Markowitz": (COLORS["stock"], "-.", "Underlying Markowitz", 2.1),
        BROAD_STOCK_LABEL: ("#174A7A", ":", BROAD_STOCK_LABEL, 2.0),
    }
    for ax, title, family in panel_specs:
        for stock_label, stock_line in stock.groupby("config_label", sort=False):
            stock_line = stock_line.sort_values("return_date")
            color, linestyle, legend_label, linewidth = stock_styles.get(
                str(stock_label),
                (COLORS["stock"], "-.", str(stock_label), 2.0),
            )
            ax.plot(
                stock_line["return_date"],
                stock_line["wealth"],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=legend_label,
                zorder=4,
            )
        subset = paths.loc[paths["family"].eq(family)].copy()
        for config in CONFIG_ORDER:
            line = subset.loc[subset["config"].eq(config)].sort_values("return_date")
            if line.empty:
                continue
            linestyle = "-" if family == "Locked E1" else "--"
            linewidth = 2.0 if family == "Locked E1" else 1.8
            ax.plot(
                line["return_date"],
                line["wealth"],
                color=LINE_COLORS[config],
                linewidth=linewidth,
                linestyle=linestyle,
                label=(
                    f"{DISPLAY_CONFIG[config]} {DISPLAY_MODEL['E1']}"
                    if family == "Locked E1"
                    else f"{DISPLAY_CONFIG[config]} capped naive"
                ),
            )
            last = line.iloc[-1]
            ax.text(
                last["return_date"],
                float(last["wealth"]) * label_offsets.get((family, config), 1.0),
                f" {float(last['wealth']):.2g}x",
                fontsize=7.2,
                color=LINE_COLORS[config],
                va="center",
            )
        for stock_label, stock_line in stock.groupby("config_label", sort=False):
            stock_line = stock_line.sort_values("return_date")
            if stock_line.empty:
                continue
            color, _, _, _ = stock_styles.get(
                str(stock_label),
                (COLORS["stock"], "-.", str(stock_label), 2.0),
            )
            stock_last = stock_line.iloc[-1]
            ax.text(
                stock_last["return_date"],
                float(stock_last["wealth"]) * stock_offsets.get((family, str(stock_label)), 1.0),
                f" {float(stock_last['wealth']):.2g}x",
                fontsize=7.2,
                color=color,
                va="center",
            )
        ax.axhline(1.0, color="#555", linewidth=0.85, linestyle=":", alpha=0.75)
        ax.set_yscale("log")
        ax.set_ylabel("Cumulative wealth\n(log scale)")
        ax.set_title(title, fontsize=10, pad=7)
        ax.grid(True, which="major", axis="y", alpha=0.22, linewidth=0.7)
        ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.4)
        ax.set_xlim(right=paths["return_date"].max() + pd.Timedelta(days=170))

    handles: list = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.2, frameon=False)

    axes[1].set_xlabel("Rolling OOS return date")
    fig.suptitle("Rolling OOS full-cost return paths: baselines and four variants", fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.88, bottom=0.17, left=0.10, right=0.98, hspace=0.24)
    plt.savefig(FIG_DIR / "final_walk_forward_return_paths.pdf")
    plt.savefig(FIG_DIR / "short_walk_forward_return_paths.pdf")
    plt.close(fig)


def plot_short_headline_wealth() -> None:
    source = PAPER / "analysis/artifacts/r11_higher_risk/r11_monthly_development_returns.csv"
    monthly = pd.read_csv(_require(source), parse_dates=["return_date", "decision_date"])
    path = monthly.loc[
        monthly["config"].astype(str).eq("orig+VIX")
        & monthly["strategy"].astype(str).eq("R1.1 25pct positive-edge deployment")
    ].sort_values("return_date")
    if len(path) != 93:
        raise ValueError(f"Expected 93 headline months, found {len(path)}")
    if path["return_date"].duplicated().any():
        raise ValueError("Headline return dates are not unique")

    net_returns = pd.to_numeric(path["net_return"], errors="raise")
    net_returns.index = pd.DatetimeIndex(path["return_date"])
    if net_returns.isna().any():
        raise ValueError("Headline net-return path is incomplete")
    wealth = (1.0 + net_returns).cumprod()
    initial_date = pd.Timestamp(path["decision_date"].iloc[0])
    wealth = pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([initial_date])), wealth])
    drawdown = wealth / wealth.cummax() - 1.0
    abstained = path["integer_execution_abstained"].astype(str).str.lower().eq("true")
    abstention_periods = path.loc[abstained, ["decision_date", "return_date"]]
    band_starts = mdates.date2num(abstention_periods["decision_date"])
    band_ends = mdates.date2num(abstention_periods["return_date"])
    lower = np.zeros(len(abstention_periods))
    upper = np.ones(len(abstention_periods))
    band_vertices = np.stack(
        [
            np.column_stack([band_starts, lower]),
            np.column_stack([band_starts, upper]),
            np.column_stack([band_ends, upper]),
            np.column_stack([band_ends, lower]),
        ],
        axis=1,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True)
    axes[0].plot(
        wealth.index,
        wealth.to_numpy(float),
        color=JOURNAL_COLORS[0],
        linewidth=2.2,
        label=DISPLAY_MODEL["R1.1"],
    )
    axes[0].add_collection(
        PolyCollection(
            band_vertices,
            facecolor=JOURNAL_COLORS[4],
            edgecolor="none",
            alpha=0.12,
            label="Integer abstention month",
            transform=axes[0].get_xaxis_transform(),
        ),
    )
    axes[0].axhline(1.0, color="#6E7781", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Net wealth")
    axes[0].set_title("(a) Headline cumulative net wealth", fontsize=11)
    axes[0].grid(True, axis="y", alpha=0.20, linewidth=0.7)

    axes[1].plot(drawdown.index, drawdown.to_numpy(float), color=JOURNAL_COLORS[2], linewidth=2.0)
    minimum_date = pd.Timestamp(drawdown.idxmin())
    minimum_drawdown = float(drawdown.min())
    axes[1].scatter([minimum_date], [minimum_drawdown], color=JOURNAL_COLORS[2], s=22, zorder=3)
    axes[1].annotate(
        f"{minimum_drawdown:.1%}",
        xy=(minimum_date, minimum_drawdown),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color=JOURNAL_COLORS[2],
    )
    axes[1].add_collection(
        PolyCollection(
            band_vertices,
            facecolor=JOURNAL_COLORS[4],
            edgecolor="none",
            alpha=0.12,
            transform=axes[1].get_xaxis_transform(),
        ),
    )
    axes[1].axhline(0.0, color="#6E7781", linewidth=0.8, linestyle=":")
    axes[1].set_ylim(minimum_drawdown - 0.025, 0.01)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Return date")
    axes[1].set_title("(b) Drawdown from prior peak", fontsize=11)
    axes[1].grid(True, axis="y", alpha=0.20, linewidth=0.7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_DIR / "short_headline_wealth.pdf")
    plt.close(fig)


def plot_short_model_progression() -> None:
    aligned_path = PAPER / "analysis/artifacts/r1_r11_aligned/r1_r11_aligned_return_panel.csv"
    aligned = pd.read_csv(_require(aligned_path), parse_dates=["return_date"])
    aligned = aligned.loc[
        aligned["config"].astype(str).eq("orig+VIX")
        & aligned["window"].astype(str).eq("aligned_2018_2026")
        & aligned["strategy"].astype(str).isin(
            ["R1 repaired net utility", "R1.1 25pct positive-edge deployment"]
        )
    ]
    aligned_returns = aligned.pivot(index="return_date", columns="strategy", values="net_return").sort_index()

    rolling = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_rolling_oos.csv"), parse_dates=["return_date"])
    common_dates = pd.DatetimeIndex(pd.to_datetime(rolling["return_date"]).dropna().unique()).sort_values()
    prototype_frame = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_strategy_returns_net.csv"))
    prototype_column = "orig+VIX E1 capped"
    if prototype_column not in prototype_frame.columns:
        raise ValueError(f"Missing prototype return column {prototype_column!r}")
    if len(prototype_frame) != len(common_dates):
        raise ValueError(
            "Cannot align prototype returns to the common date ledger: "
            f"{len(prototype_frame)} rows vs {len(common_dates)} dates"
        )
    if not common_dates.isin(aligned_returns.index).all():
        raise ValueError("Aligned R1/R1.1 panel does not cover every prototype return date")

    common_returns = pd.DataFrame(
        {
            "E1": pd.to_numeric(prototype_frame[prototype_column], errors="raise").to_numpy(float),
            "R1": pd.to_numeric(
                aligned_returns.loc[common_dates, "R1 repaired net utility"], errors="raise"
            ).to_numpy(float),
            "R1.1": pd.to_numeric(
                aligned_returns.loc[common_dates, "R1.1 25pct positive-edge deployment"], errors="raise"
            ).to_numpy(float),
        },
        index=common_dates,
    )
    if common_returns.isna().any().any():
        raise ValueError("Common model-progression return panel is incomplete")
    initial_date = common_dates.min() - pd.offsets.MonthEnd(1)
    initial = pd.DataFrame(0.0, index=pd.DatetimeIndex([initial_date]), columns=common_returns.columns)
    wealth = (1.0 + pd.concat([initial, common_returns])).cumprod()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(wealth.index, wealth["E1"], color=JOURNAL_COLORS[2], linewidth=1.8, label=DISPLAY_MODEL["E1"])
    ax.plot(wealth.index, wealth["R1"], color=JOURNAL_COLORS[1], linewidth=2.0, label=DISPLAY_MODEL["R1"])
    ax.plot(wealth.index, wealth["R1.1"], color=JOURNAL_COLORS[0], linewidth=2.3, label=DISPLAY_MODEL["R1.1"])
    ax.axhline(1.0, color="#6E7781", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative net wealth (log scale)")
    ax.set_xlabel("Common return date")
    ax.set_title("Model progression on the common 2021-2026 return window", fontsize=11)
    ax.grid(True, which="major", axis="y", alpha=0.20, linewidth=0.7)
    ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG_DIR / "short_model_progression.pdf")
    plt.close(fig)


def plot_short_prototype_failure() -> None:
    source = ROBUSTNESS_DIR / "breadth_cv_cpcv_path_month_returns.csv"
    paths = pd.read_csv(_require(source), parse_dates=["return_date"])
    paths = paths.loc[
        paths["strategy"].astype(str).eq("orig+VIX E1 capped")
        & paths["basis"].astype(str).eq("full_cost_net")
    ]
    returns = paths.pivot(index="return_date", columns="path_id", values="ret").sort_index()
    if returns.shape[1] != 11:
        raise ValueError(f"Expected 11 CPCV paths, found {returns.shape[1]}")
    returns = returns.astype(float)
    if returns.isna().any().any():
        raise ValueError("Prototype CPCV path pivot is incomplete")
    gross_growth = 1.0 + returns
    raw_wealth = gross_growth.cumprod()
    absorbed = gross_growth.le(0.0).cummax()
    wealth = raw_wealth.mask(absorbed, 0.0)
    initial_date = returns.index.min() - pd.offsets.MonthEnd(1)
    wealth = pd.concat([pd.DataFrame(1.0, index=pd.DatetimeIndex([initial_date]), columns=wealth.columns), wealth])
    absorbed = pd.concat([pd.DataFrame(False, index=pd.DatetimeIndex([initial_date]), columns=absorbed.columns), absorbed])
    absorbed_columns = absorbed.any(axis=0)
    if not absorbed_columns.any():
        raise ValueError("No prototype CPCV path reaches the zero-wealth absorption boundary")
    first_absorption = gross_growth.loc[:, absorbed_columns].le(0.0).idxmax()
    if not first_absorption.eq(pd.Timestamp("2020-03-31")).all():
        raise ValueError(f"Prototype absorption dates changed: {first_absorption.to_dict()}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(wealth.index, wealth.to_numpy(float), color="#B9C0C6", linewidth=1.0, alpha=0.72)
    ax.plot(
        wealth.index,
        wealth.loc[:, absorbed_columns].where(absorbed.loc[:, absorbed_columns]).to_numpy(float),
        color=JOURNAL_COLORS[2],
        linewidth=1.4,
        alpha=0.75,
    )
    march_2020 = pd.Timestamp("2020-03-31")
    ax.axvline(march_2020, color=JOURNAL_COLORS[2], linewidth=1.2, linestyle="--")
    ax.annotate(
        f"{int(absorbed_columns.sum())} paths absorb at zero",
        xy=(pd.Timestamp("2021-01-31"), 2.2),
        ha="left",
        va="center",
        fontsize=8,
        color=JOURNAL_COLORS[2],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )
    handles = [
        plt.Line2D([0], [0], color="#B9C0C6", linewidth=1.5, label="CPCV net wealth path"),
        plt.Line2D([0], [0], color=JOURNAL_COLORS[2], linewidth=1.8, label="Absorbed at zero"),
        plt.Line2D([0], [0], color=JOURNAL_COLORS[2], linewidth=1.2, linestyle="--", label="March 2020"),
    ]
    ax.axhline(1.0, color="#6E7781", linewidth=0.8, linestyle=":")
    ax.set_ylim(bottom=-0.03)
    ax.set_ylabel("Full-cost net wealth")
    ax.set_xlabel("CPCV return date")
    ax.set_title(f"{DISPLAY_MODEL['E1']} fails the CPCV survival standard", fontsize=11)
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG_DIR / "short_prototype_failure.pdf")
    plt.close(fig)


def plot_short_deployment_constraints() -> None:
    source = PAPER / "analysis/artifacts/r11_higher_risk/r11_monthly_development_returns.csv"
    monthly = pd.read_csv(_require(source), parse_dates=["return_date"])
    base = monthly.loc[
        monthly["strategy"].astype(str).eq("R1.1 25pct positive-edge deployment")
    ].copy()
    gross_nav = base.pivot(index="return_date", columns="config", values="gross_nav").reindex(columns=CONFIG_ORDER)
    gross_nav = gross_nav.astype(float)
    if gross_nav.shape != (93, len(CONFIG_ORDER)) or gross_nav.isna().any().any():
        raise ValueError(f"Deployment gross/NAV panel is incomplete: shape={gross_nav.shape}")
    target_values = pd.to_numeric(base["deployment_target"], errors="raise").drop_duplicates()
    if len(target_values) != 1:
        raise ValueError(f"Expected one deployment target, found {target_values.tolist()}")
    deployment_target = float(target_values.iloc[0])
    abstained = base["integer_execution_abstained"].astype(str).str.lower().eq("true")
    abstention_counts = abstained.groupby(base["config"]).sum().reindex(CONFIG_ORDER).astype(int)
    expected_counts = pd.Series([0, 32, 0, 34], index=CONFIG_ORDER)
    if not abstention_counts.equals(expected_counts):
        raise ValueError(f"Unexpected abstention counts: {abstention_counts.to_dict()}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].boxplot(
        gross_nav.to_numpy(float),
        tick_labels=[DISPLAY_CONFIG[config] for config in CONFIG_ORDER],
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": JOURNAL_COLORS[1], "alpha": 0.45, "edgecolor": "#40534C"},
        medianprops={"color": JOURNAL_COLORS[0], "linewidth": 1.5},
        whiskerprops={"color": "#6E7781"},
        capprops={"color": "#6E7781"},
    )
    axes[0].axhline(deployment_target, color=JOURNAL_COLORS[2], linewidth=1.4, linestyle="--")
    axes[0].set_ylim(bottom=0.0, top=max(0.55, float(gross_nav.max().max()) * 1.08))
    axes[0].set_ylabel("Deployed premium gross / NAV")
    axes[0].set_title("(a) Deployment is pursued, not forced", fontsize=11)
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(True, axis="y", alpha=0.20, linewidth=0.7)
    axes[0].legend(
        handles=[
            Patch(facecolor=JOURNAL_COLORS[1], alpha=0.45, edgecolor="#40534C", label="Monthly distribution"),
            plt.Line2D(
                [0],
                [0],
                color=JOURNAL_COLORS[2],
                linestyle="--",
                label=f"{deployment_target:.0%} target",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    bars = axes[1].bar(
        np.arange(len(CONFIG_ORDER)),
        abstention_counts.to_numpy(int),
        color=JOURNAL_COLORS[0],
        alpha=0.82,
        edgecolor="#263238",
        linewidth=0.5,
    )
    axes[1].bar_label(bars, padding=2, fontsize=8)
    axes[1].set_xticks(np.arange(len(CONFIG_ORDER)), [DISPLAY_CONFIG[config] for config in CONFIG_ORDER], rotation=20)
    axes[1].set_ylim(0.0, max(36.0, float(abstention_counts.max()) * 1.12))
    axes[1].set_ylabel("Abstention months")
    axes[1].set_title("(b) Direct-or-abstain integer execution", fontsize=11)
    axes[1].grid(True, axis="y", alpha=0.20, linewidth=0.7)
    fig.suptitle(f"{DISPLAY_MODEL['R1.1']} deployment constraints", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    fig.savefig(FIG_DIR / "short_deployment_constraints.pdf")
    plt.close(fig)


def plot_short_audit_scenario_ladder() -> None:
    source = PAPER / "analysis/artifacts/execution_audit/execution_audit_summary.json"
    payload = json.loads(_require(source).read_text(encoding="utf-8"))
    headline = pd.DataFrame(payload["headline_r11"])
    headline = headline.loc[headline["window"].astype(str).eq("aligned_2018_2026")]
    scenarios = ["modeled", "mid", "touch", "worst"]
    sortino = headline.pivot(index="strategy", columns="config", values="sortino").reindex(
        index=scenarios, columns=CONFIG_ORDER
    )
    annual_return = headline.pivot(index="strategy", columns="config", values="annualized_return").reindex(
        index=scenarios, columns=CONFIG_ORDER
    )
    if sortino.isna().any().any() or annual_return.isna().any().any():
        raise ValueError("Execution-audit scenario ladder is incomplete")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    x = np.arange(len(scenarios))
    muted_styles = {"orig": "-", "larger": "--", "larger+VIX": ":"}
    for config in CONFIG_ORDER:
        emphasized = config == "orig+VIX"
        color = JOURNAL_COLORS[0] if emphasized else "#9AA1A6"
        linewidth = 2.5 if emphasized else 1.1
        linestyle = "-" if emphasized else muted_styles[config]
        alpha = 1.0 if emphasized else 0.72
        axes[0].plot(
            x,
            sortino[config].to_numpy(float),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker="o",
            markersize=4.5 if emphasized else 3.2,
            alpha=alpha,
            label=DISPLAY_CONFIG[config],
        )
        axes[1].plot(
            x,
            annual_return[config].to_numpy(float),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker="o",
            markersize=4.5 if emphasized else 3.2,
            alpha=alpha,
        )

    for x_value, y_value in zip(x, sortino["orig+VIX"].to_numpy(float)):
        axes[0].annotate(
            f"{y_value:.3f}",
            xy=(x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=JOURNAL_COLORS[0],
        )
    for ax in axes:
        ax.set_xticks(x, [scenario.title() for scenario in scenarios])
        ax.grid(True, axis="y", alpha=0.20, linewidth=0.7)
    axes[0].set_ylabel("Sortino ratio")
    axes[0].set_title("(a) Sortino under observed-fill scenarios", fontsize=11)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylabel("Annualized return")
    axes[1].set_title("(b) Annualized return under the same ladder", fontsize=11)
    fig.suptitle(f"{DISPLAY_MODEL['R1.1']} execution audit", fontsize=11, y=0.995)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(FIG_DIR / "short_audit_scenario_ladder.pdf")
    plt.close(fig)


def plot_short_proxy_coverage_evidence() -> None:
    spread_source = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_spread_source_coverage.csv"))
    spread_source["rows"] = pd.to_numeric(spread_source["rows"], errors="raise")
    spread_source["proxy_rows"] = spread_source["rows"].where(
        spread_source["relative_spread_source"].astype(str).eq("inferred_cbbo_proxy"), 0.0
    )
    proxy_totals = spread_source.groupby("config", observed=True)[["proxy_rows", "rows"]].sum()
    proxy_share = (proxy_totals["proxy_rows"] / proxy_totals["rows"]).reindex(CONFIG_ORDER)

    source = PAPER / "analysis/artifacts/execution_audit/execution_audit_summary.json"
    payload = json.loads(_require(source).read_text(encoding="utf-8"))
    coverage = pd.DataFrame(payload["coverage"])
    coverage = coverage.loc[coverage["arm"].astype(str).eq("R1.1")].set_index("config").reindex(CONFIG_ORDER)
    coverage["entry_coverage"] = pd.to_numeric(coverage["entry_coverage"], errors="raise")
    coverage["roundtrip_coverage"] = pd.to_numeric(coverage["roundtrip_coverage"], errors="raise")
    spreads = pd.DataFrame(payload["spread_distribution"])
    spreads = spreads.loc[spreads["arm"].astype(str).eq("R1.1")].set_index(["regime", "source"])
    if (
        proxy_share.isna().any()
        or proxy_totals["rows"].le(0.0).any()
        or coverage[["entry_coverage", "roundtrip_coverage"]].isna().any().any()
    ):
        raise ValueError("Proxy or quote-coverage evidence is incomplete")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    x = np.arange(len(CONFIG_ORDER))
    axes[0].bar(
        x,
        proxy_share.to_numpy(float),
        color=JOURNAL_COLORS[5],
        alpha=0.66,
        edgecolor="#263238",
        linewidth=0.5,
        label="Inferred-proxy row share",
    )
    axes[0].scatter(
        x,
        coverage["entry_coverage"],
        marker="o",
        s=36,
        color=JOURNAL_COLORS[0],
        label="Entry quote coverage",
        zorder=3,
    )
    axes[0].scatter(
        x,
        coverage["roundtrip_coverage"],
        marker="D",
        s=30,
        color=JOURNAL_COLORS[1],
        label="Round-trip quote coverage",
        zorder=3,
    )
    axes[0].set_xticks(x, [DISPLAY_CONFIG[config] for config in CONFIG_ORDER], rotation=20)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Share / quote-coverage weight")
    axes[0].set_title("(a) Proxy reliance and observed-quote coverage", fontsize=11)
    axes[0].grid(True, axis="y", alpha=0.20, linewidth=0.7)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=7.5, frameon=False)

    regimes = ["pre_2023_03_28", "post_2023_03_28"]
    regime_labels = ["Before 2023-03-28", "From 2023-03-28"]
    centers = np.arange(len(regimes))
    width = 0.18
    modeled_p50 = 100.0 * pd.to_numeric(spreads.loc[(regimes, "modeled"), "p50"], errors="raise").to_numpy(float)
    observed_p50 = 100.0 * pd.to_numeric(spreads.loc[(regimes, "observed"), "p50"], errors="raise").to_numpy(float)
    modeled_p90 = 100.0 * pd.to_numeric(spreads.loc[(regimes, "modeled"), "p90"], errors="raise").to_numpy(float)
    observed_p90 = 100.0 * pd.to_numeric(spreads.loc[(regimes, "observed"), "p90"], errors="raise").to_numpy(float)
    axes[1].bar(centers - 1.5 * width, modeled_p50, width, color=JOURNAL_COLORS[1], alpha=0.55, label="Modeled p50")
    axes[1].bar(centers - 0.5 * width, observed_p50, width, color=JOURNAL_COLORS[0], alpha=0.55, label="Observed p50")
    axes[1].bar(centers + 0.5 * width, modeled_p90, width, color=JOURNAL_COLORS[1], alpha=0.90, label="Modeled p90")
    axes[1].bar(centers + 1.5 * width, observed_p90, width, color=JOURNAL_COLORS[0], alpha=0.90, label="Observed p90")
    axes[1].set_xticks(centers, regime_labels, rotation=12)
    axes[1].set_ylabel("Relative spread (% of premium)")
    axes[1].set_title(f"(b) {DISPLAY_MODEL['R1.1']} spread quantiles", fontsize=11)
    axes[1].grid(True, axis="y", alpha=0.20, linewidth=0.7)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(FIG_DIR / "short_proxy_coverage_evidence.pdf")
    plt.close(fig)


def main() -> None:
    scoreboard = build_baseline_scoreboard()
    distributions = build_validation_distribution_summary()
    path_values = build_validation_path_values()
    walk_paths = build_walk_forward_return_paths(scoreboard)
    robustness = build_short_robustness_matrix(scoreboard)
    spread = build_short_spread_summary()
    write_scoreboard_tables(scoreboard, distributions)
    write_short_tables(scoreboard, robustness, spread)
    plot_validation_distributions(path_values, scoreboard)
    plot_baseline_comparison(scoreboard)
    plot_short_theory_flow()
    plot_short_robustness_heatmap(robustness)
    plot_short_capacity_spread_panel(scoreboard, spread)
    plot_walk_forward_return_paths(walk_paths)
    plot_short_headline_wealth()
    plot_short_model_progression()
    plot_short_prototype_failure()
    plot_short_deployment_constraints()
    plot_short_audit_scenario_ladder()
    plot_short_proxy_coverage_evidence()


if __name__ == "__main__":
    main()
