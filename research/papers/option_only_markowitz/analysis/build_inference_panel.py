"""Build the final inference panel for the option-only Markowitz paper.

The panel is deliberately artifact-backed.  It reads locked return and
scoreboard artifacts, validates the static return row order against the final
scoreboard, then writes a detailed CSV plus the compact paper table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.papers.option_only_markowitz.analysis.inference import (
    BootstrapConfig,
    _monthly_sharpe,
    block_bootstrap_metric_ci,
    sharpe_difference_test,
)
from src.portfolio.multi_asset_derivative_portfolio_model import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
ARTIFACT_DIR = PAPER / "artifacts"
BREADTH_DIR = PAPER / "analysis/artifacts/breadth_solutions"
ROBUSTNESS_DIR = BREADTH_DIR / "robustness"

CONFIG_ORDER = ["orig", "orig+VIX", "larger", "larger+VIX"]
PRIMARY_STRATEGY = "E1 capped"
STOCK_STRATEGY = "Underlying Markowitz"
NAIVE_POINT_IDS = {"Equal premium", "Equal risk"}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def _annualized_sharpe(returns: pd.Series, periods_per_year: float = 12.0) -> float:
    sr_m = _monthly_sharpe(pd.Series(returns, dtype=float).to_numpy(float))
    return float(sr_m * np.sqrt(periods_per_year)) if np.isfinite(sr_m) else np.nan


def _block_size(n_obs: int, config: BootstrapConfig) -> int:
    return int(config.block_size or max(2, int(round(np.sqrt(max(n_obs, 1))))))


def _date_grid(paths: pd.DataFrame) -> pd.DatetimeIndex:
    if "return_date" not in paths.columns:
        raise ValueError("final_walk_forward_return_paths.csv missing return_date")
    dates = pd.DatetimeIndex(pd.to_datetime(paths["return_date"]).dropna().unique()).sort_values()
    if len(dates) != 60:
        raise ValueError(f"Expected exactly 60 walk-forward dates, found {len(dates)}")
    return dates


def _load_static_net_returns(date_grid: pd.DatetimeIndex) -> pd.DataFrame:
    static = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_strategy_returns_net.csv"))
    if len(static) != len(date_grid):
        raise ValueError(
            "breadth_strategy_returns_net.csv row count does not match "
            f"walk-forward date grid: {len(static)} rows vs {len(date_grid)} dates"
        )
    static = static.apply(pd.to_numeric, errors="coerce")
    static.index = date_grid
    return static


def _load_scoreboard() -> pd.DataFrame:
    scoreboard = pd.read_csv(_require(ROBUSTNESS_DIR / "final_result_scoreboard.csv"))
    required = {"config", "config_label", "e1_net_sharpe", "best_naive_strategy"}
    missing = sorted(required.difference(scoreboard.columns))
    if missing:
        raise ValueError(f"final_result_scoreboard.csv missing columns: {missing}")
    return scoreboard


def _validate_static_scoreboard(static: pd.DataFrame, scoreboard: pd.DataFrame) -> None:
    indexed = scoreboard.set_index("config")
    for config in CONFIG_ORDER:
        col = f"{config} {PRIMARY_STRATEGY}"
        if col not in static.columns:
            raise ValueError(f"Missing static return column: {col}")
        if config not in indexed.index:
            raise ValueError(f"Missing scoreboard row for config: {config}")
        actual = _annualized_sharpe(static[col])
        expected = float(indexed.loc[config, "e1_net_sharpe"])
        if not np.isfinite(actual) or abs(actual - expected) >= 1e-9:
            raise RuntimeError(
                f"Static return order validation failed for {config}: "
                f"recomputed {actual:.15f}, scoreboard {expected:.15f}"
            )


def _load_stock_baseline(date_grid: pd.DatetimeIndex) -> pd.Series:
    stock = pd.read_csv(_require(ARTIFACT_DIR / "strategy_returns_post_cost.csv"))
    if STOCK_STRATEGY not in stock.columns:
        raise ValueError(f"strategy_returns_post_cost.csv missing {STOCK_STRATEGY!r}")
    if "snap_date" in stock.columns:
        date_col = "snap_date"
    elif "return_date" in stock.columns:
        date_col = "return_date"
    else:
        date_col = str(stock.columns[0])
    stock = stock[[date_col, STOCK_STRATEGY]].copy()
    stock[date_col] = pd.to_datetime(stock[date_col])
    if stock[date_col].duplicated().any():
        raise ValueError("strategy_returns_post_cost.csv has duplicate date rows")
    aligned = (
        stock.set_index(date_col)
        .sort_index()[STOCK_STRATEGY]
        .astype(float)
        .reindex(date_grid)
    )
    if int(aligned.dropna().shape[0]) != len(date_grid):
        raise ValueError(
            f"Underlying Markowitz baseline has {aligned.dropna().shape[0]} "
            f"aligned observations; expected {len(date_grid)}"
        )
    aligned.name = STOCK_STRATEGY
    return aligned


def _optimizer_trials_p1(p1: pd.DataFrame, config: str) -> np.ndarray:
    required = {"config", "point_id", "net_sharpe_noimpact"}
    missing = sorted(required.difference(p1.columns))
    if missing:
        raise ValueError(f"p1_regularization_results.csv missing columns: {missing}")
    sub = p1.loc[p1["config"].astype(str).eq(config)].copy()
    if sub.empty:
        raise ValueError(f"Missing P1 rows for config: {config}")
    sub["point_id"] = sub["point_id"].astype(str)
    trials = sub.loc[~sub["point_id"].isin(NAIVE_POINT_IDS)].drop_duplicates("point_id")
    values = pd.to_numeric(trials["net_sharpe_noimpact"], errors="coerce").dropna()
    return values.to_numpy(float) / np.sqrt(12.0)


def _candidate_trials_p3(p3: pd.DataFrame, config: str) -> np.ndarray:
    required = {"config", "strategy", "knobs_label", "mode", "net_sharpe"}
    missing = sorted(required.difference(p3.columns))
    if missing:
        raise ValueError(f"p3_combined_results.csv missing columns: {missing}")
    sub = p3.loc[p3["config"].astype(str).eq(config)].copy()
    if sub.empty:
        return np.array([], dtype=float)
    strategy = sub["strategy"].astype(str)
    knobs = sub["knobs_label"].astype(str)
    mode = sub["mode"].astype(str)
    non_naive = ~(
        strategy.str.contains("Equal premium|Equal risk", case=False, regex=True, na=False)
        | knobs.str.contains("naive", case=False, regex=False, na=False)
        | mode.str.contains("naive", case=False, regex=False, na=False)
    )
    values = pd.to_numeric(sub.loc[non_naive, "net_sharpe"], errors="coerce").dropna()
    return values.to_numpy(float) / np.sqrt(12.0)


def _psr_dsr(
    returns: pd.Series,
    p1_trials_m: np.ndarray,
    sensitivity_trials_m: np.ndarray,
) -> dict[str, float]:
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    sr_hat = _monthly_sharpe(clean.to_numpy(float))
    skew = float(clean.skew()) if len(clean) >= 3 else 0.0
    kurt = float(clean.kurtosis() + 3.0) if len(clean) >= 4 else 3.0
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 3.0
    n_obs = int(len(clean))
    return {
        "psr": probabilistic_sharpe_ratio(sr_hat, 0.0, n_obs, skew=skew, kurt=kurt),
        "dsr": deflated_sharpe_ratio(sr_hat, p1_trials_m, n_obs, skew=skew, kurt=kurt),
        "dsr_sensitivity": deflated_sharpe_ratio(sr_hat, sensitivity_trials_m, n_obs, skew=skew, kurt=kurt),
    }


def _series_from_paths(paths: pd.DataFrame, *, family: str, strategy: str) -> pd.Series:
    sub = paths.loc[
        paths["family"].astype(str).eq(family)
        & paths["strategy"].astype(str).eq(strategy)
    ].copy()
    if sub.empty:
        raise ValueError(f"Missing rolling path for {family} / {strategy}")
    sub["return_date"] = pd.to_datetime(sub["return_date"])
    sub = sub.sort_values("return_date")
    if sub["return_date"].duplicated().any():
        raise ValueError(f"Duplicate rolling dates for {family} / {strategy}")
    out = pd.Series(pd.to_numeric(sub["return"], errors="coerce").to_numpy(float), index=sub["return_date"], name=strategy)
    if int(out.dropna().shape[0]) != 60:
        raise ValueError(f"Rolling path for {strategy} has {out.dropna().shape[0]} observations; expected 60")
    return out


def _inference_row(
    *,
    config: str,
    config_label: str,
    basis: str,
    e1: pd.Series,
    stock: pd.Series,
    naive: pd.Series,
    naive_strategy: str,
    p1_trials_m: np.ndarray,
    sensitivity_trials_m: np.ndarray,
    bootstrap: BootstrapConfig,
    include_dsr: bool,
) -> dict[str, object]:
    sharpe, sharpe_lo, sharpe_hi = block_bootstrap_metric_ci(e1, "sharpe", bootstrap)
    sortino, sortino_lo, sortino_hi = block_bootstrap_metric_ci(e1, "sortino", bootstrap)
    stock_test = sharpe_difference_test(e1, stock, bootstrap)
    naive_test = sharpe_difference_test(e1, naive, bootstrap)
    n_obs = int(pd.Series(e1, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().shape[0])
    ratios = _psr_dsr(e1, p1_trials_m, sensitivity_trials_m) if include_dsr else {
        "psr": np.nan,
        "dsr": np.nan,
        "dsr_sensitivity": np.nan,
    }
    return {
        "config": config,
        "config_label": config_label,
        "basis": basis,
        "e1_strategy": f"{config} {PRIMARY_STRATEGY}",
        "naive_strategy": naive_strategy,
        "net_sharpe": sharpe,
        "net_sharpe_ci_lo": sharpe_lo,
        "net_sharpe_ci_hi": sharpe_hi,
        "net_sortino": sortino,
        "net_sortino_ci_lo": sortino_lo,
        "net_sortino_ci_hi": sortino_hi,
        "stock_sharpe": stock_test["sharpe_b"],
        "delta_sharpe_stock": stock_test["delta_sharpe"],
        "jk_z_stock": stock_test["jk_z"],
        "jk_p_stock": stock_test["jk_p"],
        "boot_p_stock": stock_test["boot_p"],
        "n_obs_stock": stock_test["n_obs"],
        "naive_sharpe": naive_test["sharpe_b"],
        "delta_sharpe_naive": naive_test["delta_sharpe"],
        "jk_z_naive": naive_test["jk_z"],
        "jk_p_naive": naive_test["jk_p"],
        "boot_p_naive": naive_test["boot_p"],
        "n_obs_naive": naive_test["n_obs"],
        "psr": ratios["psr"],
        "dsr": ratios["dsr"],
        "dsr_trials": int(len(p1_trials_m)),
        "dsr_trials_sensitivity": int(len(sensitivity_trials_m)),
        "dsr_sensitivity": ratios["dsr_sensitivity"],
        "n_obs": n_obs,
        "block_size": _block_size(n_obs, bootstrap),
        "n_boot": int(bootstrap.n_boot),
        "seed": int(bootstrap.seed),
        "alpha": float(bootstrap.alpha),
    }


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, escape=False, float_format="%.3f"), encoding="utf-8")


def _write_short_table(panel: pd.DataFrame) -> None:
    static = panel.loc[panel["basis"].eq("static")].set_index("config").loc[CONFIG_ORDER]
    out = pd.DataFrame(
        {
            "Config": static["config_label"],
            "Net Sharpe": static["net_sharpe"],
            "CI lo": static["net_sharpe_ci_lo"],
            "CI hi": static["net_sharpe_ci_hi"],
            "PSR": static["psr"],
            "DSR": static["dsr"],
            "dSR stock": static["delta_sharpe_stock"],
            "p stock": static["jk_p_stock"],
            "dSR naive": static["delta_sharpe_naive"],
            "p naive": static["jk_p_naive"],
        }
    )
    _write_latex_table(out, TABLE_DIR / "short_inference_panel.tex")


def build_inference_panel(bootstrap: BootstrapConfig | None = None) -> pd.DataFrame:
    cfg = bootstrap or BootstrapConfig()
    paths = pd.read_csv(_require(ROBUSTNESS_DIR / "final_walk_forward_return_paths.csv"))
    date_grid = _date_grid(paths)
    static = _load_static_net_returns(date_grid)
    scoreboard = _load_scoreboard()
    _validate_static_scoreboard(static, scoreboard)
    stock = _load_stock_baseline(date_grid)
    p1 = pd.read_csv(_require(BREADTH_DIR / "p1_regularization_results.csv"))
    p3 = pd.read_csv(_require(BREADTH_DIR / "p3_combined_results.csv"))

    score_by_config = scoreboard.set_index("config")
    stock_rolling = _series_from_paths(paths, family="Stock baseline", strategy=STOCK_STRATEGY)
    rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        score = score_by_config.loc[config]
        config_label = str(score["config_label"])
        naive_strategy = str(score["best_naive_strategy"])
        e1_col = f"{config} {PRIMARY_STRATEGY}"
        naive_col = f"{config} {naive_strategy}"
        if naive_col not in static.columns:
            raise ValueError(f"Missing matched naive static return column: {naive_col}")
        p1_trials = _optimizer_trials_p1(p1, config)
        sensitivity_trials = np.concatenate([p1_trials, _candidate_trials_p3(p3, config)])

        rows.append(
            _inference_row(
                config=config,
                config_label=config_label,
                basis="static",
                e1=static[e1_col],
                stock=stock,
                naive=static[naive_col],
                naive_strategy=naive_strategy,
                p1_trials_m=p1_trials,
                sensitivity_trials_m=sensitivity_trials,
                bootstrap=cfg,
                include_dsr=True,
            )
        )

        rolling_e1 = _series_from_paths(paths, family="Locked E1", strategy=e1_col)
        rolling_naive = _series_from_paths(paths, family="Matched capped naive", strategy=naive_col)
        rows.append(
            _inference_row(
                config=config,
                config_label=config_label,
                basis="rolling",
                e1=rolling_e1,
                stock=stock_rolling,
                naive=rolling_naive,
                naive_strategy=naive_strategy,
                p1_trials_m=p1_trials,
                sensitivity_trials_m=sensitivity_trials,
                bootstrap=cfg,
                include_dsr=False,
            )
        )

    panel = pd.DataFrame(rows)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(ROBUSTNESS_DIR / "final_inference_panel.csv", index=False)
    _write_short_table(panel)
    return panel


def main(config: BootstrapConfig | None = None) -> pd.DataFrame:
    return build_inference_panel(config)


if __name__ == "__main__":
    main()
