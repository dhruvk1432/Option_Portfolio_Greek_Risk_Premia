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
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"
ROBUSTNESS_DIR = PAPER / "analysis/artifacts/breadth_solutions/robustness"
SUMMARY_DIR = ROBUSTNESS_DIR

CONFIG_ORDER = ["orig", "orig+VIX", "larger", "larger+VIX"]
DISPLAY_CONFIG = {
    "orig": "orig",
    "orig+VIX": "orig+VIX",
    "larger": "larger",
    "larger+VIX": "larger+VIX",
}
PRIMARY_STRATEGY = "E1 capped"
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
    "CPCV net full",
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
        rows.append(
            {
                "config": config,
                "config_label": DISPLAY_CONFIG[config],
                "verdict": str(primary_row["verdict"]),
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


def _append_path_values(rows: list[dict[str, object]], frame: pd.DataFrame, *, config: str, validation: str) -> None:
    values = pd.to_numeric(frame.get("sharpe", pd.Series(dtype=float)), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    for value in values:
        rows.append({"config": config, "validation": validation, "sharpe": float(value)})


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
    return pd.DataFrame(rows, columns=["config", "validation", "sharpe"])


def build_walk_forward_return_paths(scoreboard: pd.DataFrame) -> pd.DataFrame:
    rolling = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_rolling_oos.csv"), parse_dates=["return_date"])
    stock = pd.read_csv(_require(PAPER / "artifacts/strategy_returns_post_cost.csv"), parse_dates=["snap_date"])
    stock = stock[["snap_date", "Underlying Markowitz"]].rename(
        columns={"snap_date": "return_date", "Underlying Markowitz": "return"}
    )
    rolling_dates = set(pd.to_datetime(rolling["return_date"]).dt.normalize())
    stock = stock.loc[pd.to_datetime(stock["return_date"]).dt.normalize().isin(rolling_dates)]

    rows: list[pd.DataFrame] = []
    stock_path = stock.sort_values("return_date").copy()
    stock_path["config"] = "stock"
    stock_path["config_label"] = "Stock Markowitz"
    stock_path["family"] = "Stock baseline"
    stock_path["strategy"] = "Underlying Markowitz"
    rows.append(stock_path[["return_date", "config", "config_label", "family", "strategy", "return"]])

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
            ("CPCV net full", _status_from_tail(float(val["cpcv_net_p05"]), float(val["cpcv_net_p50"])), float(val["cpcv_net_p50"])),
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
            "e1_net_sharpe": "E1 net",
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

    fig, axes_raw = plt.subplots(1, len(validations), figsize=(11.2, 4.25), sharex=True, sharey=True, squeeze=False)
    axes = axes_raw[0]
    all_vals: list[float] = [0.0]
    all_vals.extend(frame["sharpe"].to_numpy(float).tolist())
    all_vals.extend(float(v) for v in realized_by_config.values() if np.isfinite(float(v)))
    for ax, validation in zip(axes, validations):
        subset = frame.loc[frame["validation"].astype(str).eq(validation)]
        positions: list[int] = []
        data: list[np.ndarray] = []
        colors: list[str] = []
        for y, config in zip(y_positions, CONFIG_ORDER):
            values = subset.loc[subset["config"].astype(str).eq(config), "sharpe"].dropna()
            if values.empty:
                continue
            arr = values.to_numpy(float)
            positions.append(int(y))
            data.append(arr)
            colors.append(LINE_COLORS.get(config, COLORS["interval"]))
        if data:
            parts = ax.violinplot(
                data,
                positions=positions,
                orientation="horizontal",
                widths=0.72,
                showmedians=True,
                showextrema=False,
            )
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor("#263238")
                body.set_alpha(0.45)
                body.set_linewidth(0.8)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#202020")
                parts["cmedians"].set_linewidth(1.1)
        if validation.startswith("CPCV"):
            for y, config in zip(y_positions, CONFIG_ORDER):
                values = subset.loc[subset["config"].astype(str).eq(config), "sharpe"].dropna()
                if values.empty:
                    continue
                arr = values.to_numpy(float)
                offsets = np.linspace(-0.22, 0.22, len(arr)) if len(arr) > 1 else np.array([0.0])
                ax.scatter(
                    arr,
                    np.full(len(arr), y, dtype=float) + offsets,
                    s=14,
                    color="#202020",
                    alpha=0.68,
                    linewidths=0,
                    zorder=4,
                )
        for y, config in zip(y_positions, CONFIG_ORDER):
            realized = float(realized_by_config.get(config, np.nan))
            if np.isfinite(realized):
                ax.scatter(
                    realized,
                    y,
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
    fig.legend(
        handles=[median_handle, realized_handle, path_handle],
        loc="lower center",
        ncol=3,
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

    fig, ax = plt.subplots(figsize=(8.8, 4.45))
    ax.bar(
        x - width / 2,
        scoreboard["e1_net_sharpe"].to_numpy(float),
        width,
        label="Locked E1 option book",
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ymin = min(-0.25, float(scoreboard[["e1_net_sharpe", "best_naive_net_sharpe"]].min().min()) - 0.15)
    ymax = max(1.75, float(scoreboard[["e1_net_sharpe", "best_naive_net_sharpe"]].max().max()) + 0.28, stock + 0.25)
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
        ("3. Risk model", "$\\Sigma_O=B\\Omega B^\\top+\\Sigma_\\varepsilon$"),
        ("4. Robust allocation", "$\\max_w\\ \\hat\\mu^\\top w-\\frac{\\gamma}{2}w^\\top\\Sigma w$\n$-\\ c^\\top|w|$"),
        ("5. Validation", "net caps, costs,\nCPCV, MC, rolling OOS"),
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
        (axes[0], "Locked E1 variants", "Locked E1"),
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
    stock_offsets = {"Locked E1": 0.82, "Matched capped naive": 1.22}
    for ax, title, family in panel_specs:
        ax.plot(
            stock["return_date"],
            stock["wealth"],
            color=COLORS["stock"],
            linewidth=2.1,
            linestyle="-.",
            label="Underlying Markowitz",
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
                label=line["config_label"].iloc[0],
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
        stock_last = stock.iloc[-1]
        ax.text(
            stock_last["return_date"],
            float(stock_last["wealth"]) * stock_offsets.get(family, 1.0),
            f" {float(stock_last['wealth']):.2g}x",
            fontsize=7.2,
            color=COLORS["stock"],
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


if __name__ == "__main__":
    main()
