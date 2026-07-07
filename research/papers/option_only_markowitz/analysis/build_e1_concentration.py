"""Build the locked E1 concentration table from E1 book-weight artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
BREADTH_DIR = PAPER / "analysis/artifacts/breadth_solutions"
ROBUSTNESS_DIR = BREADTH_DIR / "robustness"
WEIGHTS_CSV = "breadth_e1_book_weights.csv"

CONFIG_ORDER = ["orig", "orig+VIX", "larger", "larger+VIX"]
LOCKED_PARTICIPATION = 0.05
LOCKED_AUM = 1_000_000.0
ACTIVE_TOL = 1e-6
MATCH_TOL = 1e-6


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def _require_weights(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required locked E1 weights artifact missing: {path}. Run `make e1-ablation` first.")
    return path


def _write_latex_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, escape=False, float_format="%.3f"), encoding="utf-8")


def _load_locked_weights() -> pd.DataFrame:
    detail = pd.read_csv(_require_weights(ROBUSTNESS_DIR / WEIGHTS_CSV))
    required = {
        "config",
        "asset_id",
        "weight",
        "cap_bound",
        "utilization",
    }
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError(f"{WEIGHTS_CSV} missing columns: {missing}")

    locked = detail.loc[detail["config"].astype(str).isin(CONFIG_ORDER)].copy()
    if locked.empty:
        raise ValueError(f"No locked E1 rows found in {WEIGHTS_CSV}")
    observed = sorted(locked["config"].astype(str).unique())
    missing_configs = [config for config in CONFIG_ORDER if config not in observed]
    if missing_configs:
        raise ValueError(f"Locked E1 weights missing configs: {missing_configs}")
    locked["config"] = pd.Categorical(locked["config"].astype(str), categories=CONFIG_ORDER, ordered=True)
    return locked.sort_values(["config", "asset_id"]).reset_index(drop=True)


def _validate_utilization(detail: pd.DataFrame) -> None:
    weight = pd.to_numeric(detail["weight"], errors="coerce").fillna(0.0)
    bound = pd.to_numeric(detail["cap_bound"], errors="coerce")
    utilization = pd.to_numeric(detail["utilization"], errors="coerce")
    calc = weight.abs() / bound.where(bound.gt(0.0))
    finite = calc.replace([np.inf, -np.inf], np.nan).notna() & utilization.notna()
    if finite.any():
        max_diff = float((calc.loc[finite] - utilization.loc[finite]).abs().max())
        if max_diff > 1e-9:
            raise RuntimeError(f"utilization is not abs(weight)/bound: max_abs_diff={max_diff:.12g}")
    if utilization.notna().any():
        min_util = float(utilization.min())
        max_util = float(utilization.max())
        if min_util < -1e-12 or max_util > 1.0 + 1e-8:
            raise RuntimeError(f"utilization outside [0, 1]: min={min_util:.12g}, max={max_util:.12g}")


def build_concentration_panel() -> pd.DataFrame:
    detail = _load_locked_weights()
    _validate_utilization(detail)

    rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        group = detail.loc[detail["config"].astype(str).eq(config)].copy()
        weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
        abs_weight = weights.abs()
        active = abs_weight > ACTIVE_TOL
        deployed_gross = float(abs_weight.sum())
        top5_share = float(abs_weight.nlargest(5).sum() / deployed_gross) if deployed_gross > 0 else np.nan
        utilization = pd.to_numeric(group["utilization"], errors="coerce")
        at_cap_share = float(utilization.loc[active].ge(0.99).mean()) if active.any() else np.nan
        rows.append(
            {
                "config": config,
                "participation": LOCKED_PARTICIPATION,
                "aum": LOCKED_AUM,
                "n_candidate_contracts": int(len(group)),
                "n_active_contracts": int(active.sum()),
                "deployed_gross": deployed_gross,
                "top5_share": top5_share,
                "n_active_at_cap": int(utilization.loc[active].ge(0.99).sum()) if active.any() else 0,
                "at_cap_share": at_cap_share,
                "cap_budget": float(pd.to_numeric(group["cap_bound"], errors="coerce").sum()),
            }
        )
    panel = pd.DataFrame(rows)
    _validate_against_realized(panel)
    return panel


def _validate_against_realized(panel: pd.DataFrame) -> None:
    realized = pd.read_csv(_require(ROBUSTNESS_DIR / "breadth_realized_candidate_summary.csv"))
    required = {"config", "strategy", "deployed_gross"}
    missing = sorted(required.difference(realized.columns))
    if missing:
        raise ValueError(f"breadth_realized_candidate_summary.csv missing columns: {missing}")
    e1 = realized.loc[realized["strategy"].astype(str).eq("E1 capped"), ["config", "deployed_gross"]].copy()
    e1["deployed_gross"] = pd.to_numeric(e1["deployed_gross"], errors="coerce")
    expected = e1.drop_duplicates("config").set_index("config")["deployed_gross"]
    actual = panel.set_index("config")["deployed_gross"]
    missing_configs = [config for config in CONFIG_ORDER if config not in expected.index]
    if missing_configs:
        raise ValueError(f"Realized summary missing E1 capped configs: {missing_configs}")
    diff = (actual.reindex(CONFIG_ORDER) - expected.reindex(CONFIG_ORDER)).abs()
    bad = diff[diff > MATCH_TOL]
    if not bad.empty:
        detail = ", ".join(
            f"{config}: book_weights={actual.loc[config]:.12f}, realized={expected.loc[config]:.12f}, "
            f"diff={diff.loc[config]:.12g}"
            for config in bad.index
        )
        raise RuntimeError(f"E1 concentration deployed_gross guard failed: {detail}")


def _short_table(panel: pd.DataFrame) -> pd.DataFrame:
    indexed = panel.set_index("config").loc[CONFIG_ORDER]
    return pd.DataFrame(
        {
            "Config": indexed.index,
            "Candidates": indexed["n_candidate_contracts"].astype(int).to_numpy(),
            "Active": indexed["n_active_contracts"].astype(int).to_numpy(),
            "Top 5 share": indexed["top5_share"].to_numpy(float),
            "Deployed gross": indexed["deployed_gross"].to_numpy(float),
            "At cap share": indexed["at_cap_share"].to_numpy(float),
            "Cap budget": indexed["cap_budget"].to_numpy(float),
        }
    )


def write_outputs(panel: pd.DataFrame) -> None:
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(ROBUSTNESS_DIR / "final_e1_concentration.csv", index=False)
    _write_latex_table(_short_table(panel), TABLE_DIR / "short_e1_concentration.tex")


def main() -> int:
    panel = build_concentration_panel()
    write_outputs(panel)
    print(panel.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
