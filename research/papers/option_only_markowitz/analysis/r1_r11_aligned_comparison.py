"""Date-aligned R1 versus R1.1 retrospective development comparison.

The frozen R1 source and post-2020 artifacts are not changed.  This driver runs
only R1's missing pre-2021 segment, appends the existing frozen R1 returns, and
slices the existing R1.1 direct-or-abstain path to the same dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from research.papers.option_only_markowitz.analysis.breadth_p1_regularization_experiment import build_configs
from research.papers.option_only_markowitz.analysis.breadth_vix_noimpact_experiment import _build_config_panel
from research.papers.option_only_markowitz.analysis.r1_repaired_pipeline import PAPER, R1_NAME
from research.papers.option_only_markowitz.analysis.r11_higher_risk_pipeline import R11_NAME
from research.papers.option_only_markowitz.analysis.simulation import performance_metrics
import research.papers.option_only_markowitz.analysis.r1_repaired_pipeline as r1_pipeline


DEFAULT_OUT = PAPER / "analysis" / "artifacts" / "r1_r11_aligned"
R1_FROZEN = PAPER / "analysis" / "artifacts" / "r1_repaired" / "r1_monthly_development_returns.csv"
R11_FROZEN = PAPER / "analysis" / "artifacts" / "r11_higher_risk" / "r11_monthly_development_returns.csv"


def summarize_aligned(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (config, strategy, window), group in frame.groupby(["config", "strategy", "window"], observed=True):
        values = group.sort_values("return_date")["net_return"]
        rows.append(
            {
                "config": config,
                "strategy": strategy,
                "window": window,
                "start": pd.to_datetime(group["return_date"]).min(),
                "end": pd.to_datetime(group["return_date"]).max(),
                **performance_metrics(values),
                "mean_gross_nav": float(group["gross_nav"].mean()),
                "integer_abstentions": int(group.get("integer_execution_abstained", False).fillna(False).astype(bool).sum())
                if "integer_execution_abstained" in group
                else 0,
            }
        )
    return pd.DataFrame(rows)


def run_comparison(out_dir: Path = DEFAULT_OUT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_r1 = pd.read_csv(R1_FROZEN, parse_dates=["return_date", "decision_date", "train_start", "train_end"])
    frozen_r11 = pd.read_csv(R11_FROZEN, parse_dates=["return_date", "decision_date", "train_start", "train_end"])
    frozen_r11 = frozen_r11[frozen_r11["strategy"].eq(R11_NAME)].copy()
    post_start = pd.Timestamp(frozen_r1["return_date"].min())
    early_cutoff = post_start - pd.Timedelta(days=1)

    configs, _ = build_configs()
    early_returns: list[pd.DataFrame] = []
    early_weights: list[pd.DataFrame] = []
    original_train_end = r1_pipeline.TRAIN_END
    try:
        r1_pipeline.TRAIN_END = pd.Timestamp("2018-01-31")
        for label, (equities, poc_names, with_vix) in configs.items():
            _, panel_returns, _, _, _ = _build_config_panel(equities, poc_names, with_vix)
            eligible = pd.DatetimeIndex(panel_returns.index)
            periods = int(((eligible > r1_pipeline.TRAIN_END) & (eligible <= early_cutoff)).sum())
            returns, weights, _ = r1_pipeline.run_r1_config(
                label,
                equities,
                poc_names,
                with_vix,
                max_periods=periods,
            )
            returns["evidence_status"] = "retrospective_development_extension"
            returns["segment"] = "new_pre_2021_R1_run"
            weights["segment"] = "new_pre_2021_R1_run"
            early_returns.append(returns)
            early_weights.append(weights)
    finally:
        r1_pipeline.TRAIN_END = original_train_end

    early = pd.concat(early_returns, ignore_index=True, sort=False)
    weights = pd.concat(early_weights, ignore_index=True, sort=False)
    frozen_r1 = frozen_r1.copy()
    frozen_r1["evidence_status"] = "retrospective_development_sample"
    frozen_r1["segment"] = "frozen_post_2020_R1_artifact"
    extended_r1 = pd.concat([early, frozen_r1], ignore_index=True, sort=False)
    extended_r1 = extended_r1.sort_values(["config", "return_date"]).reset_index(drop=True)

    common_start = max(pd.Timestamp(extended_r1["return_date"].min()), pd.Timestamp(frozen_r11["return_date"].min()))
    common_end = min(pd.Timestamp(extended_r1["return_date"].max()), pd.Timestamp(frozen_r11["return_date"].max()))
    full_r1 = extended_r1[extended_r1["return_date"].between(common_start, common_end)].copy()
    full_r11 = frozen_r11[frozen_r11["return_date"].between(common_start, common_end)].copy()
    full_r1["window"] = "aligned_2018_2026"
    full_r11["window"] = "aligned_2018_2026"

    post_r1 = frozen_r1[frozen_r1["return_date"].between(post_start, common_end)].copy()
    post_r11 = frozen_r11[frozen_r11["return_date"].between(post_start, common_end)].copy()
    post_r1["window"] = "aligned_post_2020"
    post_r11["window"] = "aligned_post_2020"
    aligned = pd.concat([full_r1, full_r11, post_r1, post_r11], ignore_index=True, sort=False)

    counts = aligned.groupby(["window", "config", "strategy"], observed=True)["return_date"].nunique()
    if counts.groupby(["window", "config"], observed=True).nunique().ne(1).any():
        raise RuntimeError("R1 and R1.1 date counts are not aligned")
    summary = summarize_aligned(aligned)
    early.to_csv(out_dir / "r1_pre_2021_extension_returns.csv", index=False)
    weights.to_csv(out_dir / "r1_pre_2021_extension_weights.csv", index=False)
    extended_r1.to_csv(out_dir / "r1_extended_2018_2026_returns.csv", index=False)
    aligned.to_csv(out_dir / "r1_r11_aligned_return_panel.csv", index=False)
    summary.to_csv(out_dir / "r1_r11_aligned_summary.csv", index=False)
    return extended_r1, aligned, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, summary = run_comparison(args.out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
