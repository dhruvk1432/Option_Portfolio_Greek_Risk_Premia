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
    "diagnostic_capacity_infeasible": "#6E7781",
    "naive": "#B9C0C6",
    "stock": "#2F6F9F",
    "realized": "#8B1E3F",
    "interval": "#40534C",
}


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


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda x: "yes" if bool(x) else "no")
    path.write_text(out.to_latex(index=False, escape=False, float_format="%.3f"), encoding="utf-8")


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


def plot_validation_distributions(distributions: pd.DataFrame, scoreboard: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    validations = ["CPCV complete paths", "MC resampled histories", "MC refit stability"]
    y_labels = [DISPLAY_CONFIG[c] for c in CONFIG_ORDER]
    y_positions = np.arange(len(CONFIG_ORDER))
    verdict_by_config = dict(zip(scoreboard["config"], scoreboard["verdict"]))
    realized_by_config = dict(zip(scoreboard["config"], scoreboard["e1_net_sharpe"]))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.25), sharey=True)
    all_vals: list[float] = [0.0]
    for ax, validation in zip(axes, validations):
        subset = distributions.loc[distributions["validation"].eq(validation)]
        for y, config in zip(y_positions, CONFIG_ORDER):
            row = subset.loc[subset["config"].eq(config)]
            if row.empty:
                continue
            r = row.iloc[0]
            color = COLORS.get(verdict_by_config.get(config, ""), COLORS["interval"])
            p05, p50, p95 = float(r["p05"]), float(r["p50"]), float(r["p95"])
            realized = float(realized_by_config.get(config, np.nan))
            ax.hlines(y, p05, p95, color=color, linewidth=3.0, alpha=0.82)
            ax.scatter(p50, y, s=42, marker="o", color=color, edgecolor="white", linewidth=0.7, zorder=3)
            ax.scatter(realized, y, s=50, marker="D", color=COLORS["realized"], edgecolor="#202020", linewidth=0.5, zorder=4)
            all_vals.extend([p05, p50, p95, realized])
        ax.axvline(0.0, color="#555", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_title(validation, fontsize=10, pad=8)
        ax.set_xlabel("Full-cost net Sharpe")
        ax.grid(True, axis="x", alpha=0.20, linewidth=0.7)
        ax.grid(True, axis="y", alpha=0.06, linewidth=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()

    finite = np.array([x for x in all_vals if np.isfinite(x)], dtype=float)
    xmin = min(-0.75, float(np.nanmin(finite)) - 0.12)
    xmax = max(1.75, float(np.nanmax(finite)) + 0.12)
    for ax in axes:
        ax.set_xlim(xmin, xmax)

    interval_handle = plt.Line2D([0], [0], color=COLORS["pass"], linewidth=3, label="p05-p95 interval")
    median_handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["pass"], markeredgecolor="white", markersize=6, label="median")
    realized_handle = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["realized"], markeredgecolor="#202020", markersize=6, label="realized static book")
    axes[0].legend(handles=[interval_handle, median_handle, realized_handle], fontsize=8, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_breadth_validation_distributions.pdf")
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
    plt.close(fig)


def main() -> None:
    scoreboard = build_baseline_scoreboard()
    distributions = build_validation_distribution_summary()
    write_scoreboard_tables(scoreboard, distributions)
    plot_validation_distributions(distributions, scoreboard)
    plot_baseline_comparison(scoreboard)


if __name__ == "__main__":
    main()
