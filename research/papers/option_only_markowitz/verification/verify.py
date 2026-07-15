"""Independent verification harness for the option-only Markowitz paper.

The empirical runner is the producer.  This module is the auditor: it rebuilds or
reads generated artifacts, recomputes the most important quantities from those
artifacts, checks point-in-time ledgers and claim boundaries, compiles the paper,
and writes a machine-readable and human-readable verification record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PAPER = Path(__file__).resolve().parents[1]
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"
ART_DIR = PAPER / "artifacts"
VERIFY_DIR = PAPER / "verification"
BREADTH_DIR = PAPER / "analysis/artifacts/breadth_solutions"
BREADTH_ROBUSTNESS_DIR = BREADTH_DIR / "robustness"
R1_DIR = PAPER / "analysis/artifacts/r1_repaired"
R11_DIR = PAPER / "analysis/artifacts/r11_higher_risk"
R2_DIR = PAPER / "analysis/artifacts/r2_robust_sortino"
EXECUTION_AUDIT_DIR = PAPER / "analysis/artifacts/execution_audit"
PUBLISHED_STEM = "option_only_portfolio_optimization_dhruv_kohli"
PUBLISHED_TEX_NAME = f"{PUBLISHED_STEM}.tex"
PUBLISHED_PDF_NAME = f"{PUBLISHED_STEM}.pdf"
PUBLISHED_PDF = PAPER / PUBLISHED_PDF_NAME
PUBLISHED_TEX = PAPER / PUBLISHED_TEX_NAME
BREADTH_CONFIG_ORDER = ["orig", "orig+VIX", "larger", "larger+VIX"]
BREADTH_PRIMARY_STRATEGY = "E1 capped"
BREADTH_NAIVE_POINT_IDS = {"Equal premium", "Equal risk"}

sys.path.insert(0, str(ROOT))

from research.papers.option_only_markowitz.analysis import run_empirics as emp  # noqa: E402
from research.papers.option_only_markowitz.analysis.vix_option_panel import (  # noqa: E402
    VIX_FACTOR,
    black76_greeks,
    black76_price,
    parse_osi_symbol,
    stack_vix_option_shards,
)
from src.portfolio import (  # noqa: E402
    FactorShockSpec,
    OptionMarkowitzConstraints,
    OptionOnlyMarkowitzModel,
    OptionOnlySpec,
    bs_greeks,
    bs_price,
    nearest_psd,
    performance_stats,
)

PERIODS_PER_YEAR = emp.PERIODS_PER_YEAR
TRAIN_END = emp.TRAIN_END
PRIMARY = emp.PRIMARY_UNDERLYINGS
CRITICAL = "critical"
WARNING = "warning"
TOL = 5e-7


@dataclass
class CheckResult:
    name: str
    category: str
    status: str
    severity: str = CRITICAL
    observed: Any = ""
    expected: Any = ""
    details: str = ""


class Verifier:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []
        self.outputs: dict[str, Path] = {}
        VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    def check(
        self,
        name: str,
        category: str,
        passed: bool,
        observed: Any = "",
        expected: Any = "",
        details: str = "",
        severity: str = CRITICAL,
    ) -> None:
        self.results.append(
            CheckResult(
                name=name,
                category=category,
                status="pass" if bool(passed) else "fail",
                severity=severity,
                observed=_json_scalar(observed),
                expected=_json_scalar(expected),
                details=str(details),
            )
        )

    def fail_count(self, critical_only: bool = False) -> int:
        return sum(
            r.status != "pass" and (not critical_only or r.severity == CRITICAL)
            for r in self.results
        )

    def write_outputs(self, command_line: str, hash_manifest: pd.DataFrame) -> None:
        failed = pd.DataFrame([asdict(r) for r in self.results if r.status != "pass"])
        if failed.empty:
            failed = pd.DataFrame(columns=list(CheckResult.__dataclass_fields__.keys()))
        all_checks = pd.DataFrame([asdict(r) for r in self.results])
        failed_path = VERIFY_DIR / "failed_checks.csv"
        summary_path = VERIFY_DIR / "verification_summary.json"
        report_path = VERIFY_DIR / "verification_report.md"
        manifest_path = VERIFY_DIR / "hash_manifest.csv"
        all_path = VERIFY_DIR / "verification_checks.csv"

        failed.to_csv(failed_path, index=False)
        all_checks.to_csv(all_path, index=False)
        hash_manifest.to_csv(manifest_path, index=False)

        summary = {
            "status": "pass" if self.fail_count(critical_only=True) == 0 else "fail",
            "critical_failures": self.fail_count(critical_only=True),
            "total_failures": self.fail_count(critical_only=False),
            "total_checks": len(self.results),
            "command": command_line,
            "paper": str(PAPER),
            "outputs": {
                "verification_report": str(report_path),
                "failed_checks": str(failed_path),
                "all_checks": str(all_path),
                "hash_manifest": str(manifest_path),
            },
            "checks_by_category": _counts_by_category(self.results),
        }
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        report_path.write_text(_render_report(summary, self.results, hash_manifest), encoding="utf-8")
        self.outputs = {
            "summary": summary_path,
            "report": report_path,
            "failed": failed_path,
            "checks": all_path,
            "manifest": manifest_path,
        }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if not math.isfinite(val) else val
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _counts_by_category(results: list[CheckResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        out.setdefault(r.category, {"pass": 0, "fail": 0})[r.status] += 1
    return out


def _render_report(summary: dict[str, Any], results: list[CheckResult], manifest: pd.DataFrame) -> str:
    lines = [
        "# Option-Only Markowitz Verification Report",
        "",
        f"Status: **{summary['status'].upper()}**",
        f"Critical failures: `{summary['critical_failures']}`",
        f"Total checks: `{summary['total_checks']}`",
        f"Hash manifest rows: `{len(manifest)}`",
        "",
        "## Category Summary",
        "",
        "| Category | Passed | Failed |",
        "|---|---:|---:|",
    ]
    for category, counts in sorted(summary["checks_by_category"].items()):
        lines.append(f"| {category} | {counts.get('pass', 0)} | {counts.get('fail', 0)} |")
    failed = [r for r in results if r.status != "pass"]
    lines += ["", "## Failed Checks", ""]
    if not failed:
        lines.append("No failed checks.")
    else:
        lines += ["| Severity | Category | Check | Observed | Expected | Details |", "|---|---|---|---|---|---|"]
        for r in failed:
            lines.append(
                f"| {r.severity} | {r.category} | {r.name} | {_md(r.observed)} | {_md(r.expected)} | {_md(r.details)} |"
            )
    lines += ["", "## Passed Critical Evidence", ""]
    for r in results:
        if r.status == "pass" and r.severity == CRITICAL:
            lines.append(f"- `{r.category}` / `{r.name}`: {r.details or r.observed}")
    return "\n".join(lines) + "\n"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:500]


def _run(
    cmd: list[str],
    cwd: Path,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )


def regenerate_artifacts(v: Verifier, skip: bool) -> None:
    if skip:
        v.check("empirical regeneration skipped", "producer", True, details="Fast audit mode explicitly skipped regeneration.", severity=WARNING)
        return
    cmd = [sys.executable, "-m", "research.papers.option_only_markowitz.analysis.run_empirics"]
    res = _run(cmd, ROOT, timeout=1800)
    v.check(
        "empirical runner exits cleanly",
        "producer",
        res.returncode == 0,
        observed=f"exit={res.returncode}",
        expected="exit=0",
        details=res.stdout[-1200:],
    )


def regenerate_robustness_artifacts(v: Verifier, skip: bool) -> None:
    if skip:
        v.check(
            "robustness regeneration skipped",
            "producer",
            True,
            details="Robustness regeneration is opt-in; pass --with-robustness to rerun the long stage.",
            severity=WARNING,
        )
        return
    cmd = [
        sys.executable,
        "-m",
        "research.papers.option_only_markowitz.analysis.run_empirics",
        "--stage",
        "robustness",
    ]
    res = _run(cmd, ROOT, timeout=3600)
    v.check(
        "robustness runner exits cleanly",
        "producer",
        res.returncode == 0,
        observed=f"exit={res.returncode}",
        expected="exit=0",
        details=res.stdout[-1200:],
    )


def compile_paper(v: Verifier, skip: bool) -> None:
    if skip:
        v.check("paper compile skipped", "paper", True, details="Fast audit mode explicitly skipped LaTeX compilation.", severity=WARNING)
        return
    latex = "/Library/TeX/texbin/lualatex" if Path("/Library/TeX/texbin/lualatex").exists() else "lualatex"
    commands = [
        [latex, "-interaction=nonstopmode", PUBLISHED_TEX_NAME],
        ["bibtex", PUBLISHED_STEM],
        [latex, "-interaction=nonstopmode", PUBLISHED_TEX_NAME],
        [latex, "-interaction=nonstopmode", PUBLISHED_TEX_NAME],
        [latex, "-interaction=nonstopmode", PUBLISHED_TEX_NAME],
    ]
    outputs = []
    ok = True
    tex_cache = Path("/tmp/option_only_markowitz_texmfvar")
    tex_cache.mkdir(parents=True, exist_ok=True)
    tex_env = os.environ.copy()
    tex_env["TEXMFVAR"] = str(tex_cache)
    tex_env["TEXMFCACHE"] = str(tex_cache)
    for cmd in commands:
        res = _run(cmd, PAPER, timeout=240, env=tex_env)
        outputs.append(f"$ {' '.join(cmd)}\nexit={res.returncode}\n{res.stdout[-1200:]}")
        ok = ok and res.returncode == 0
    ok = ok and PUBLISHED_PDF.exists()
    v.check("latex bibtex compile pipeline", "paper", ok, observed="; ".join(x.split("\n")[1] for x in outputs), expected="all exit=0", details="\n\n".join(outputs)[-3000:])


def load_summary(v: Verifier) -> dict[str, Any]:
    p = TABLE_DIR / "empirical_summary.json"
    v.check("empirical summary exists", "artifacts", p.exists(), observed=p)
    if not p.exists():
        return {}
    try:
        summary = json.loads(p.read_text())
        required = {
            "data",
            "performance",
            "risk_calibration",
            "timing_diagnostics",
            "trading_data_audit",
            "exposure",
            "factor_regression",
            "pnl_attribution",
            "regime_performance",
            "vix_regime_performance",
            "leave_one_out",
            "rolling_oos",
            "claim_strength",
            "claim_audit",
            "figure_visibility",
            "random_feasible",
            "performance_post_cost",
            "post_cost_survival",
            "cost_scenario_diagnostics",
            "execution_repair_diagnostics",
            "execution_repair_comparison",
            "repair_config",
            "sortino_diagnostics",
            "sortino_entry_cost_summary",
            "cost_input_spread_sources",
            "liquidity_tier_performance",
            "liquidity_tier_diagnostics",
            "forecast_ablation_performance",
            "forecast_ablation_components",
            "reality_check_inference",
            "simulation_summary",
            "simulation_assumptions",
            "drawdown_breach_rates",
            "hurdle_summary",
            "inference",
            "cost_diagnostics",
            "vix_settlement_coverage",
            "vix_settlement_audit",
            "vix_required_settlement_download_audit",
            "vix_chain_feature_summary",
            "vol_of_vol_regime_performance",
            "data_extension_manifest",
        }
        v.check("empirical summary schema", "artifacts", required.issubset(summary.keys()), observed=sorted(summary.keys()), expected=sorted(required))
        return summary
    except Exception as exc:
        v.check("empirical summary parses", "artifacts", False, observed=type(exc).__name__, details=str(exc))
        return {}


def check_required_outputs(v: Verifier) -> None:
    required = [
        PUBLISHED_PDF,
        PUBLISHED_TEX,
        PAPER / "REPRODUCIBILITY.md",
        PAPER / "docs/source_ledger.md",
        TABLE_DIR / "empirical_summary.json",
        TABLE_DIR / "portfolio_performance.tex",
        TABLE_DIR / "inference_summary.tex",
        TABLE_DIR / "cost_capacity_margin_diagnostics.tex",
        TABLE_DIR / "cost_input_spread_source_coverage.tex",
        TABLE_DIR / "sortino_objective_diagnostics.tex",
        TABLE_DIR / "execution_repair_diagnostics.tex",
        TABLE_DIR / "execution_repair_comparison.tex",
        TABLE_DIR / "vix_settlement_coverage.tex",
        TABLE_DIR / "vix_settlement_audit.tex",
        TABLE_DIR / "vix_required_settlement_download_audit.tex",
        TABLE_DIR / "short_inference_panel.tex",
        TABLE_DIR / "short_e1_channel_ablation.tex",
        TABLE_DIR / "short_e1_concentration.tex",
        TABLE_DIR / "short_cpcv_windows.tex",
        TABLE_DIR / "factor_regression.tex",
        TABLE_DIR / "pnl_attribution.tex",
        TABLE_DIR / "claim_strength_summary.tex",
        TABLE_DIR / "claim_audit.tex",
        FIG_DIR / "portfolio_growth.pdf",
        FIG_DIR / "portfolio_growth_all_strategies.pdf",
        FIG_DIR / "random_sharpe_histogram.pdf",
        FIG_DIR / "risk_calibration.pdf",
        FIG_DIR / "regime_sharpes.pdf",
        FIG_DIR / "leave_one_out_sharpe.pdf",
        ART_DIR / "strategy_returns.csv",
        ART_DIR / "strategy_returns_post_cost.csv",
        ART_DIR / "net_strategy_returns_by_cost_scenario.csv",
        ART_DIR / "net_strategy_returns_by_cost_scenario_repaired.csv",
        ART_DIR / "required_capital_returns.csv",
        ART_DIR / "cost_ledger.csv",
        ART_DIR / "cost_scenario_ledger.csv",
        ART_DIR / "cost_scenario_ledger_repaired.csv",
        ART_DIR / "rejected_trade_ledger.csv",
        ART_DIR / "rejected_trade_ledger_repaired.csv",
        ART_DIR / "required_capital_ledger.csv",
        ART_DIR / "required_capital_ledger_repaired.csv",
        ART_DIR / "repaired_trade_ledger.csv",
        ART_DIR / "execution_repair_diagnostics.csv",
        ART_DIR / "execution_repair_comparison.csv",
        ART_DIR / "sortino_entry_costs.csv",
        ART_DIR / "sortino_objective_diagnostics.csv",
        ART_DIR / "cost_input_spread_source_coverage.csv",
        ART_DIR / "capacity_ledger.csv",
        ART_DIR / "research_margin_ledger.csv",
        ART_DIR / "assignment_risk_ledger.csv",
        ART_DIR / "dividend_risk_filter_ledger.csv",
        ART_DIR / "capacity_market_impact_diagnostics.csv",
        ART_DIR / "hurdle_selection_ledger.csv",
        ART_DIR / "no_trade_periods.csv",
        ART_DIR / "strategy_returns_with_no_trade_state.csv",
        ART_DIR / "liquidity_tier_performance.csv",
        ART_DIR / "liquidity_tier_diagnostics.csv",
        ART_DIR / "forecast_ablation_performance.csv",
        ART_DIR / "forecast_ablation_components.csv",
        ART_DIR / "post_cost_survival.csv",
        ART_DIR / "reality_check_inference.csv",
        ART_DIR / "model_variant_registry.json",
        ART_DIR / "inference_summary.csv",
        ART_DIR / "vix_settlement_coverage.csv",
        ART_DIR / "vix_settlement_audit.csv",
        ART_DIR / "vro_soq_download_audit.csv",
        ART_DIR / "vix_required_settlement_download_audit.csv",
        ART_DIR / "vix_chain_state_features.csv",
        ART_DIR / "vol_of_vol_regime_performance.csv",
        ART_DIR / "data_extension_manifest.csv",
        ART_DIR / "strategy_weights.csv",
        ART_DIR / "holding_return_detail.csv",
        ART_DIR / "vix_holding_return_detail.csv",
        ART_DIR / "trading_data_audit.csv",
        ART_DIR / "figure_visibility_audit.csv",
        ART_DIR / "claim_strength_summary.csv",
        ART_DIR / "claim_audit.csv",
        ART_DIR / "conditional_premia_components.csv",
        TABLE_DIR / "post_cost_survival.tex",
        TABLE_DIR / "liquidity_tier_performance.tex",
        TABLE_DIR / "forecast_ablation_performance.tex",
        TABLE_DIR / "reality_check_inference.tex",
        TABLE_DIR / "simulation_summary.tex",
        TABLE_DIR / "drawdown_breach_rates.tex",
        TABLE_DIR / "simulation_assumptions.tex",
        TABLE_DIR / "capacity_market_impact_diagnostics.tex",
        TABLE_DIR / "vol_of_vol_regime_performance.tex",
        ART_DIR / "simulation_summary.csv",
        ART_DIR / "simulation_assumptions.csv",
        ART_DIR / "drawdown_breach_rates.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_e1_channel_ablation.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_e1_book_weights.csv",
        BREADTH_ROBUSTNESS_DIR / "final_e1_concentration.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_claim_cpcv_path_metrics.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_claim_fold_schedule.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_relative_paths.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_claim_relative_paths.csv",
    ]
    missing = [str(p.relative_to(PAPER)) for p in required if not p.exists()]
    v.check("required generated outputs exist", "artifacts", not missing, observed=missing, expected="no missing outputs")


def check_inputs_and_data(v: Verifier, summary: dict[str, Any]) -> None:
    input_specs = {
        "equity option feature store": ROOT / "data/feature_store/option_greek_proxy_panel.parquet",
        "Greek quality summary": ROOT / "data/feature_store/option_greek_quality.csv",
        "raw close panel": ROOT / "data/universe/multi_raw_close.csv",
        "VIX complex": ROOT / "data/universe/vix_complex.parquet",
        "VX futures curve": ROOT / "data/universe/vx_futures_daily.parquet",
    }
    shards = sorted((ROOT / "data/databento_cache").glob("opra_vix_chain_*.parquet"))
    raw_inputs_available = all(path.exists() for path in input_specs.values()) and len(shards) >= 100

    for name, path in input_specs.items():
        v.check(
            f"input exists or is externally licensed: {name}",
            "data",
            path.exists() or not raw_inputs_available,
            observed=path,
            expected="present for full rebuild; may be omitted from public standalone artifact package",
        )
    v.check(
        "VIX raw monthly shards present or externally licensed",
        "data",
        len(shards) >= 100 or not raw_inputs_available,
        observed=len(shards),
        expected=">=100 for full rebuild; may be omitted from public standalone artifact package",
    )

    if not raw_inputs_available:
        v.check(
            "standalone package uses generated data artifacts when licensed raw inputs are absent",
            "data",
            True,
            observed="raw OPRA/Databento inputs omitted",
            expected="generated artifacts and data/README.md document rebuild inputs",
        )
        parsed = parse_osi_symbol("VIX   260617C00030000")
        v.check("VIX OSI parser preserves terms", "data", parsed == ("VIX", pd.Timestamp("2026-06-17"), "call", 30.0), observed=parsed)
    else:
        try:
            panel, reps, returns = emp.load_bucket_panel()
            v.check("filtered equity panel row count", "data", len(panel) > 100_000, observed=len(panel), expected=">100000")
            v.check("equity representative choices", "data", len(reps) > 1_000, observed=len(reps), expected=">1000")
            # Missing bucket-months are legitimate panel sparsity; the check guards
            # against infinities among observed cells (stack() keeps NaN under pandas>=3).
            v.check("equity return cells finite", "data", np.isfinite(returns.stack().dropna().to_numpy(float)).all(), observed=int(returns.count().sum()))
            v.check("equity return lower bound", "data", float(np.nanmin(returns.to_numpy(float))) >= -1.0000001, observed=float(np.nanmin(returns.to_numpy(float))), expected=">=-1")
            v.check("primary underlyings present", "data", set(PRIMARY).issubset(set(panel["underlying"].unique())), observed=sorted(panel["underlying"].unique()))
            finite_cols = ["close", "spot", "strike", "delta", "gamma", "vega", "theta", "iv_proxy"]
            finite = np.isfinite(panel[finite_cols].to_numpy(float)).all()
            v.check("equity marks and Greeks finite", "data", finite, observed=finite_cols)
            expiry_ok = (pd.to_datetime(panel["expiry"]) > pd.to_datetime(panel["snap_date"])).all()
            v.check("equity option expiry after snapshot", "data", bool(expiry_ok), observed=bool(expiry_ok))
            if summary:
                v.check("summary equity row count matches recompute", "data", summary["data"].get("raw_equity_rows_after_filters") == len(panel), observed=len(panel), expected=summary["data"].get("raw_equity_rows_after_filters"))
        except Exception as exc:
            v.check("equity data audit executes", "data", False, observed=type(exc).__name__, details=str(exc))

        quality_path = ROOT / "data/feature_store/option_greek_quality.csv"
        if quality_path.exists():
            quality = pd.read_csv(quality_path)
            primary = quality[quality["underlying"].isin(PRIMARY)]
            for col in ["valid_delta_share", "valid_gamma_share", "valid_vega_share"]:
                v.check(f"primary Greek coverage {col}", "data", float(primary[col].min()) >= 0.95, observed=float(primary[col].min()), expected=">=0.95")

        try:
            raw_vix = stack_vix_option_shards(ROOT)
            dupes = int(raw_vix.duplicated(["trade_date", "symbol"]).sum()) if not raw_vix.empty else -1
            v.check("VIX stack nonempty", "data", len(raw_vix) > 100_000, observed=len(raw_vix), expected=">100000")
            v.check("VIX dedupe key is date-symbol", "data", dupes == 0, observed=dupes, expected=0)
            parsed = parse_osi_symbol("VIX   260617C00030000")
            v.check("VIX OSI parser preserves terms", "data", parsed == ("VIX", pd.Timestamp("2026-06-17"), "call", 30.0), observed=parsed)
            if summary:
                v.check("summary VIX filtered rows plausible", "data", summary["data"].get("raw_vix_rows_after_filters", 0) > 50_000, observed=summary["data"].get("raw_vix_rows_after_filters"), expected=">50000")
        except Exception as exc:
            v.check("VIX raw shard audit executes", "data", False, observed=type(exc).__name__, details=str(exc))

    vix_detail_path = ART_DIR / "vix_holding_return_detail.csv"
    if vix_detail_path.exists():
        vix_detail = pd.read_csv(vix_detail_path, parse_dates=["return_date", "decision_date", "state_snapshot_date", "expiry", "payoff_date", "train_end_date"])
        v.check("VIX detail settlement source complete", "data", vix_detail["settlement_source"].notna().all(), observed=vix_detail["settlement_source"].value_counts().to_dict())
        source_counts = vix_detail["settlement_source"].value_counts().to_dict()
        has_exact_vro = any(str(k).lower() == "vro_soq_exact" for k in source_counts)
        if has_exact_vro:
            v.check("VIX headline rows use exact VRO/SOQ", "data", all(str(k).lower() == "vro_soq_exact" for k in source_counts), observed=source_counts, expected="all vro_soq_exact")
        else:
            v.check("VIX proxy caveat preserved when VRO absent", "data", all("proxy" in str(x) for x in source_counts), observed=source_counts, expected="all settlement sources proxy")
        settlement_audit = ART_DIR / "vix_settlement_audit.csv"
        v.check("VIX settlement audit artifact exists", "data", settlement_audit.exists(), observed=settlement_audit)
        v.check("VIX Greek model is Black-76 VX-forward", "data", vix_detail["greek_model"].astype(str).eq("black76_vx_forward").all(), observed=vix_detail["greek_model"].value_counts().to_dict())
        v.check("VIX underlying is VX forward", "data", vix_detail["underlying_or_forward"].astype(str).eq(VIX_FACTOR).all(), observed=vix_detail["underlying_or_forward"].value_counts().to_dict())
        v.check("VIX long-option return lower bound", "data", float(vix_detail["option_return"].min()) >= -1.0000001, observed=float(vix_detail["option_return"].min()), expected=">=-1")


def check_pit_ledgers(v: Verifier) -> None:
    detail_path = ART_DIR / "holding_return_detail.csv"
    if not detail_path.exists():
        v.check("combined holding ledger exists", "pit", False, observed=detail_path)
        return
    detail = pd.read_csv(detail_path)
    for col in ["return_date", "decision_date", "expiry", "payoff_date", "state_snapshot_date", "train_end_date"]:
        if col in detail:
            detail[col] = pd.to_datetime(detail[col], errors="coerce")
    state = detail["state_snapshot_date"].fillna(detail["decision_date"])
    train_end = detail["train_end_date"].fillna(TRAIN_END)
    oos = detail["return_date"] > TRAIN_END
    v.check("ledger has rows", "pit", len(detail) > 1_000, observed=len(detail))
    v.check("decisions precede payoff", "pit", bool((detail["decision_date"] < detail["payoff_date"]).all()), observed="decision_date < payoff_date")
    is_vix = detail.get("asset_class", pd.Series("", index=detail.index)).astype(str).eq("vix_option") | detail.get("underlying", pd.Series("", index=detail.index)).astype(str).eq(VIX_FACTOR)
    equity_detail = detail.loc[~is_vix]
    vix_detail = detail.loc[is_vix]
    equity_payoff_ok = equity_detail.empty or bool((equity_detail["payoff_date"] <= equity_detail["return_date"]).all())
    vix_payoff_ok = vix_detail.empty or bool(
        ((vix_detail["decision_date"] < vix_detail["return_date"]) & (vix_detail["decision_date"] < vix_detail["payoff_date"])).all()
    )
    v.check("equity payoff no later than return date", "pit", equity_payoff_ok, observed=f"equity rows={len(equity_detail)}")
    v.check("VIX proxy payoff timing is after decision", "pit", vix_payoff_ok, observed=f"VIX rows={len(vix_detail)}")
    v.check("state snapshot observable by decision", "pit", bool((state <= detail["decision_date"]).all()), observed="state_snapshot_date <= decision_date")
    v.check("OOS forecast train end before return", "pit", bool((train_end[oos] < detail.loc[oos, "return_date"]).all()), observed=f"OOS rows={int(oos.sum())}")
    v.check("OOS decision dates after frozen train split", "pit", bool((detail.loc[oos, "decision_date"] >= TRAIN_END).all()), observed=str(detail.loc[oos, "decision_date"].min()))
    v.check("all option returns finite", "pit", np.isfinite(detail["option_return"].to_numpy(float)).all(), observed=len(detail))
    v.check("no long premium return below -100 percent", "pit", float(detail["option_return"].min()) >= -1.0000001, observed=float(detail["option_return"].min()))

    timing_path = ART_DIR / "timing_diagnostics.csv"
    audit_path = ART_DIR / "trading_data_audit.csv"
    if timing_path.exists() and audit_path.exists():
        timing = pd.read_csv(timing_path)
        audit = pd.read_csv(audit_path)
        v.check("timing diagnostic train/test split recorded", "pit", "2020-12-31" in " ".join(timing.astype(str).stack().tolist()), observed=timing.to_dict(orient="records")[:3])
        passes = set(audit["Pass"].astype(str).str.lower())
        v.check("trading audit pass/proxy only", "pit", passes.issubset({"yes", "proxy"}), observed=sorted(passes), expected="yes/proxy")


def check_math_and_optimizer(v: Verifier) -> None:
    try:
        S, K, T, r, sigma = 101.0, 99.0, 0.4, 0.03, 0.24
        h_s, h_v = 1e-2, 1e-4
        price = bs_price(S, K, T, r, sigma, "call")
        g = bs_greeks(S, K, T, r, sigma, "call")
        delta_fd = (bs_price(S + h_s, K, T, r, sigma, "call") - bs_price(S - h_s, K, T, r, sigma, "call")) / (2 * h_s)
        gamma_fd = (bs_price(S + h_s, K, T, r, sigma, "call") - 2 * price + bs_price(S - h_s, K, T, r, sigma, "call")) / (h_s * h_s)
        vega_fd = (bs_price(S, K, T, r, sigma + h_v, "call") - bs_price(S, K, T, r, sigma - h_v, "call")) / (2 * h_v)
        v.check("BSM finite-difference delta", "math", abs(g["delta"] - delta_fd) < 1e-5, observed=g["delta"], expected=delta_fd)
        v.check("BSM finite-difference gamma", "math", abs(g["gamma"] - gamma_fd) < 1e-5, observed=g["gamma"], expected=gamma_fd)
        v.check("BSM finite-difference vega", "math", abs(g["vega"] - vega_fd) < 1e-5, observed=g["vega"], expected=vega_fd)

        F, K2, T2, r2, sig2 = 22.0, 20.0, 35 / 365, 0.03, 0.75
        p = black76_price(F, K2, T2, r2, sig2, "call")
        bg = black76_greeks(F, K2, T2, r2, sig2, "call")
        b_delta_fd = (black76_price(F + 1e-3, K2, T2, r2, sig2, "call") - black76_price(F - 1e-3, K2, T2, r2, sig2, "call")) / 2e-3
        b_gamma_fd = (black76_price(F + 1e-3, K2, T2, r2, sig2, "call") - 2 * p + black76_price(F - 1e-3, K2, T2, r2, sig2, "call")) / (1e-3**2)
        b_vega_fd = (black76_price(F, K2, T2, r2, sig2 + 1e-4, "call") - black76_price(F, K2, T2, r2, sig2 - 1e-4, "call")) / 2e-4
        v.check("Black-76 finite-difference delta", "math", abs(bg["delta"] - b_delta_fd) < 1e-5, observed=bg["delta"], expected=b_delta_fd)
        v.check("Black-76 finite-difference gamma", "math", abs(bg["gamma"] - b_gamma_fd) < 1e-5, observed=bg["gamma"], expected=b_gamma_fd)
        v.check("Black-76 finite-difference vega", "math", abs(bg["vega"] - b_vega_fd) < 1e-5, observed=bg["vega"], expected=b_vega_fd)
    except Exception as exc:
        v.check("finite-difference Greek checks execute", "math", False, observed=type(exc).__name__, details=str(exc))

    try:
        under = ["AAA", "BBB"]
        frame = pd.DataFrame(
            {
                "underlying": ["AAA", "BBB"],
                "mark": [5.0, 4.0],
                "spot": [100.0, 80.0],
                "delta": [0.5, -0.35],
                "gamma": [0.02, 0.01],
                "vega": [20.0, 10.0],
                "theta": [-2.0, -1.0],
            },
            index=["a", "b"],
        )
        ucov = pd.DataFrame([[0.04, 0.01], [0.01, 0.05]], index=under, columns=under)
        vcov = pd.DataFrame([[0.003, 0.001], [0.001, 0.002]], index=under, columns=under)
        residual = pd.DataFrame(np.diag([0.01, 0.02]), index=frame.index, columns=frame.index)
        model = OptionOnlyMarkowitzModel(
            OptionOnlySpec(frame),
            FactorShockSpec(ucov, vol_cov=vcov),
            expected_returns=pd.Series([0.04, 0.02], index=frame.index),
            residual_cov=residual,
            constraints=OptionMarkowitzConstraints(gross_nav=1.0),
            covariance_shrinkage=0.0,
        )
        reconstructed = model.B @ model.factor_cov @ model.B.T + nearest_psd(residual.to_numpy(float))
        v.check("Greek covariance construction identity", "math", np.allclose(model.option_cov, reconstructed, atol=1e-8), observed=float(np.max(np.abs(model.option_cov - reconstructed))))
        eig_min = float(np.linalg.eigvalsh(model.option_cov).min())
        v.check("Greek covariance PSD", "math", eig_min >= -1e-9, observed=eig_min, expected=">=-1e-9")
        w = model.tangency_weights()
        raw = np.linalg.pinv(nearest_psd(model.option_cov)) @ model.expected_returns.to_numpy(float)
        expected = raw / np.abs(raw).sum()
        v.check("closed-form tangency formula", "math", np.allclose(w.to_numpy(float), expected, atol=1e-10), observed=w.to_dict(), expected=dict(zip(model.contracts, expected)))
    except Exception as exc:
        v.check("covariance and tangency checks execute", "math", False, observed=type(exc).__name__, details=str(exc))

    constraint_audit = verify_constraints_from_artifacts(v)
    if not constraint_audit.empty:
        constraint_audit.to_csv(VERIFY_DIR / "constraint_slack.csv", index=False)


def verify_constraints_from_artifacts(v: Verifier) -> pd.DataFrame:
    weights_path = ART_DIR / "strategy_weights.csv"
    exposure_path = ART_DIR / "greek_exposure_summary.csv"
    # greek_exposure_summary is only a LaTeX table; use summary JSON exposure instead when available.
    if not weights_path.exists() or not (TABLE_DIR / "empirical_summary.json").exists():
        v.check("constraint artifacts available", "optimizer", False, observed="missing weights or summary")
        return pd.DataFrame()
    weights = pd.read_csv(weights_path).rename(columns={"Unnamed: 0": "asset_id"}).set_index("asset_id")
    summary = json.loads((TABLE_DIR / "empirical_summary.json").read_text())
    exposure = pd.DataFrame(summary.get("exposure", []))
    rows = []
    optimized = ["Equity-option Greek Markowitz", "Greek Markowitz + VIX", "Beta/delta-neutral + VIX"]
    for strategy in [c for c in weights.columns if c in set(exposure["Strategy"])]:
        w = pd.to_numeric(weights[strategy], errors="coerce").fillna(0.0)
        gross = float(w.abs().sum())
        short = float(w[w < 0].abs().sum())
        max_abs = float(w.abs().max())
        row_exp = exposure[exposure["Strategy"].eq(strategy)].iloc[0]
        underlying = pd.Series(weights.index, index=weights.index).map(_asset_underlying)
        max_under_gross = float(w.abs().groupby(underlying).sum().max())
        budgeted = strategy in optimized
        beta_limit = 0.25 if strategy == "Beta/delta-neutral + VIX" else 3.0
        stress_limit = 0.35
        rows.append(
            {
                "Strategy": strategy,
                "Gross NAV slack": 1.000001 - gross,
                "Short gross slack": 0.250001 - short if budgeted else np.nan,
                "Per-contract slack": (0.180001 - max_abs) if budgeted else np.nan,
                "Max underlying gross slack": (0.350001 - max_under_gross) if budgeted else np.nan,
                "Beta SPY slack": beta_limit - abs(float(row_exp.get("Beta SPY proxy", 0.0))) if budgeted else np.nan,
                "Stress slack": float(row_exp.get("Worst stress return", 0.0)) + stress_limit if budgeted else np.nan,
                "VIX vega finite": np.isfinite(float(row_exp.get("VIX vega", 0.0))),
            }
        )
    audit = pd.DataFrame(rows)
    numeric_cols = [c for c in audit.columns if c.endswith("slack")]
    for _, row in audit.iterrows():
        passed = all(pd.isna(row[c]) or float(row[c]) >= -5e-5 for c in numeric_cols)
        v.check(f"constraint slack nonnegative: {row['Strategy']}", "optimizer", passed, observed=row.to_dict(), expected="all slack >= 0")
    v.check("constraint audit rows written", "optimizer", len(audit) >= 3, observed=len(audit), expected=">=3")
    return audit


def _asset_underlying(asset_id: str) -> str:
    if str(asset_id).startswith("VIX_"):
        return VIX_FACTOR
    return str(asset_id).split("_")[0]


def check_empirical_reproduction(v: Verifier, summary: dict[str, Any]) -> None:
    if not summary:
        return
    ret_path = ART_DIR / "strategy_returns.csv"
    if not ret_path.exists():
        v.check("strategy returns artifact exists", "empirical", False, observed=ret_path)
        return
    returns = pd.read_csv(ret_path, parse_dates=["snap_date"]).set_index("snap_date")
    net_path = ART_DIR / "strategy_returns_post_cost.csv"
    if net_path.exists():
        net_returns = pd.read_csv(net_path, parse_dates=["snap_date"]).set_index("snap_date")
        common = [c for c in returns.columns if c in net_returns.columns]
        option_cols = [c for c in common if c not in {"Delta-matched equities", "Underlying Markowitz"}]
        if option_cols:
            v.check("post-cost returns no greater than gross for option strategies on average", "empirical", bool((net_returns[option_cols].mean() <= returns[option_cols].mean() + 1e-12).all()), observed=(net_returns[option_cols].mean() - returns[option_cols].mean()).to_dict())
        v.check("post-cost return schema matches gross", "empirical", set(net_returns.columns) == set(returns.columns), observed=sorted(net_returns.columns), expected=sorted(returns.columns))
    else:
        v.check("post-cost strategy returns artifact exists", "empirical", False, observed=net_path)
    scenario_path = ART_DIR / "net_strategy_returns_by_cost_scenario.csv"
    if scenario_path.exists():
        scenario_returns = pd.read_csv(scenario_path, parse_dates=["snap_date"]).set_index("snap_date")
        scenarios = {str(c).rsplit("::", 1)[-1] for c in scenario_returns.columns if "::" in str(c)}
        v.check("all executable cost scenarios present", "empirical", {"mid", "half_spread", "full_spread"}.issubset(scenarios), observed=sorted(scenarios), expected="mid/half_spread/full_spread")
        for base in [c for c in returns.columns if c in {"Equity-option Greek Markowitz", "Greek Markowitz + VIX", "Beta/delta-neutral + VIX", "Equal premium", "Equal risk", "VIX hedge sleeve"}]:
            required = {f"{base}::mid", f"{base}::half_spread", f"{base}::full_spread"}
            v.check(f"strategy has all cost scenarios: {base}", "empirical", required.issubset(set(scenario_returns.columns)), observed=sorted(required & set(scenario_returns.columns)), expected=sorted(required))
    else:
        v.check("scenario cost returns artifact exists", "empirical", False, observed=scenario_path)
    required_cap = ART_DIR / "required_capital_returns.csv"
    v.check("required-capital returns artifact exists", "empirical", required_cap.exists(), observed=required_cap)
    rejected = ART_DIR / "rejected_trade_ledger.csv"
    if rejected.exists():
        rej = pd.read_csv(rejected)
        v.check("rejected/no-fill ledger is auditable", "empirical", {"strategy", "scenario", "asset_id", "reject_reason"}.issubset(rej.columns), observed=list(rej.columns))
    else:
        v.check("rejected/no-fill ledger exists", "empirical", False, observed=rejected)
    for path_name, required_cols in {
        "hurdle_selection_ledger.csv": {"hurdle", "asset_id", "expected_return", "expected_cost", "risk_estimate", "passed"},
        "liquidity_tier_performance.csv": {"Liquidity tier", "Strategy", "Sharpe"},
        "forecast_ablation_performance.csv": {"Ablation", "Sharpe"},
        "reality_check_inference.csv": {"Variant", "Probabilistic Sharpe", "Deflated Sharpe"},
        "capacity_market_impact_diagnostics.csv": {"Strategy", "Scenario", "Avg contracts traded", "Max capacity used"},
        "simulation_summary.csv": {"Return basis", "Strategy", "Requested method", "Simulation", "N paths", "Max DD p50"},
        "simulation_assumptions.csv": {"Return basis", "Strategy", "Method", "Status", "N obs", "Source start", "Source end"},
        "drawdown_breach_rates.csv": {"Return basis", "Strategy", "Requested method", "Simulation", "Breach 10%", "Breach 25%", "Breach 50%", "Breach 75%", "Breach 90%"},
    }.items():
        p = ART_DIR / path_name
        if p.exists():
            frame = pd.read_csv(p)
            v.check(f"{path_name} schema", "empirical", required_cols.issubset(frame.columns), observed=list(frame.columns), expected=sorted(required_cols))
        else:
            v.check(f"{path_name} exists", "empirical", False, observed=p)
    full_return_index = returns.index
    detail_path = ART_DIR / "holding_return_detail.csv"
    if detail_path.exists():
        detail_dates = pd.read_csv(detail_path, usecols=["return_date"], parse_dates=["return_date"])["return_date"].dropna()
        if not detail_dates.empty:
            full_return_index = pd.DatetimeIndex(pd.to_datetime(detail_dates.unique())).sort_values()
    factor_input = ROOT / "data/feature_store/option_greek_proxy_panel.parquet"
    factors_available = factor_input.exists()
    if factors_available:
        factors = emp.load_extended_factor_returns(full_return_index).reindex(returns.index)
        spy = factors[emp.SPY_UNDERLYING].reindex(returns.index)
    else:
        factors = pd.DataFrame(index=returns.index)
        spy = None
        v.check(
            "factor-control raw inputs absent; generated regression artifacts used",
            "empirical",
            True,
            observed=factor_input,
            expected="licensed input required for independent factor recomputation",
        )
    perf_source = summary.get("performance_gross_only", summary.get("performance", []))
    perf_summary = {row["Strategy"]: row for row in perf_source}
    for strategy in returns.columns:
        st = performance_stats(returns[strategy], PERIODS_PER_YEAR, benchmark_returns=spy)
        expected = perf_summary.get(strategy, {})
        checks = {
            "Ann. return": st["ann_return"],
            "Ann. vol": st["ann_vol"],
            "Sharpe": st["sharpe"],
            "Sortino": st["sortino"],
            "Calmar": st["calmar"],
            "Omega": st["omega"],
        }
        if spy is not None:
            checks["Info. ratio"] = st["information_ratio"]
        else:
            v.check(f"information ratio recorded from generated summary: {strategy}", "empirical", "Info. ratio" in expected, observed=expected.get("Info. ratio"))
        for key, observed in checks.items():
            target = expected.get(key, np.nan)
            ok = _close_or_both_nan(observed, target, tol=1e-6)
            v.check(f"performance metric matches summary: {strategy} / {key}", "empirical", ok, observed=observed, expected=target)
        worst = float(returns[strategy].min())
        v.check(f"worst month matches summary: {strategy}", "empirical", _close_or_both_nan(worst, expected.get("Worst month", np.nan), tol=1e-9), observed=worst, expected=expected.get("Worst month"))

    rand_path = ART_DIR / "random_feasible_sharpes.csv"
    if rand_path.exists():
        rand = pd.read_csv(rand_path).iloc[:, 0].dropna()
        v.check("random feasible p95 matches summary", "empirical", _close_or_both_nan(float(rand.quantile(0.95)), summary["random_feasible"].get("p95_sharpe"), tol=1e-9), observed=float(rand.quantile(0.95)), expected=summary["random_feasible"].get("p95_sharpe"))
        v.check("random feasible seed output count", "empirical", len(rand) == 250, observed=len(rand), expected=250)

    regression_expected = pd.DataFrame(summary.get("factor_regression", []))
    if not regression_expected.empty and factors_available:
        regression_actual = _independent_factor_regression(returns, factors)
        for strategy in regression_expected["Strategy"]:
            a = regression_actual[regression_actual["Strategy"].eq(strategy)].iloc[0]
            e = regression_expected[regression_expected["Strategy"].eq(strategy)].iloc[0]
            for col in ["Ann. alpha", "$R^2$", "Residual ann. vol", "Beta SPY", "Beta VX front", "Beta dVIX", "Beta dVVIX"]:
                v.check(f"factor regression matches summary: {strategy} / {col}", "empirical", _close_or_both_nan(a[col], e[col], tol=1e-6), observed=a[col], expected=e[col])
    elif not regression_expected.empty:
        regression_artifact = ART_DIR / "factor_regression.csv"
        if regression_artifact.exists():
            artifact = pd.read_csv(regression_artifact)
            v.check("factor regression artifact exists for standalone verification", "empirical", not artifact.empty, observed=len(artifact))
            v.check("factor regression artifact strategies match summary", "empirical", set(artifact["Strategy"]) == set(regression_expected["Strategy"]), observed=sorted(artifact["Strategy"]), expected=sorted(regression_expected["Strategy"]))
        else:
            v.check("factor regression artifact exists for standalone verification", "empirical", False, observed=regression_artifact)

    attribution = pd.DataFrame(summary.get("pnl_attribution", []))
    comp_cols = ["Equity delta", "Equity gamma", "Equity vega", "VIX-forward delta", "VIX-forward gamma", "VIX-option vega", "Theta/carry", "VX roll", "Skew/tail", "Residual"]
    for _, row in attribution.iterrows():
        total = float(pd.to_numeric(row[comp_cols], errors="coerce").fillna(0.0).sum())
        ok = abs(total - float(row["Realized ann. mean"])) <= 5e-6
        v.check(f"P&L attribution reconciles: {row['Strategy']}", "empirical", ok, observed=total, expected=row["Realized ann. mean"])

    central_simulation_strategies = {
        "Equity-option Greek Markowitz",
        "Greek Markowitz + VIX",
        "Beta/delta-neutral + VIX",
        "Delta-matched equities",
        "Underlying Markowitz",
        "Equal premium",
        "Equal risk",
        "VIX hedge sleeve",
    }
    sim_summary = pd.DataFrame(summary.get("simulation_summary", []))
    sim_assumptions = pd.DataFrame(summary.get("simulation_assumptions", []))
    sim_breaches = pd.DataFrame(summary.get("drawdown_breach_rates", []))
    if not sim_summary.empty:
        expected_pairs = {
            (basis, strategy)
            for basis in {"Gross before costs", "Full-spread post-cost"}
            for strategy in central_simulation_strategies
            if strategy in returns.columns
        }
        observed_pairs = set(zip(sim_summary["Return basis"], sim_summary["Strategy"]))
        v.check("simulation covers central strategy/basis pairs", "empirical", expected_pairs.issubset(observed_pairs), observed=sorted(observed_pairs), expected=sorted(expected_pairs))
        methods = set(sim_summary["Requested method"].astype(str))
        v.check("simulation methods include block and volatility clustered", "empirical", {"circular_block_bootstrap", "egarch_or_ewma"}.issubset(methods), observed=sorted(methods))
    if not sim_assumptions.empty:
        starts = pd.to_datetime(sim_assumptions["Source start"], errors="coerce").dropna()
        v.check("simulation inputs are OOS only", "empirical", bool((starts > TRAIN_END).all()) if len(starts) else False, observed=starts.min() if len(starts) else "missing", expected=f">{TRAIN_END.date()}")
        v.check("simulation source length matches OOS returns", "empirical", int(pd.to_numeric(sim_assumptions["N obs"], errors="coerce").min()) == len(returns), observed=sim_assumptions[["Strategy", "Return basis", "N obs"]].to_dict(orient="records")[:8], expected=len(returns))
    if not sim_breaches.empty:
        breach_cols = [c for c in sim_breaches.columns if str(c).startswith("Breach ")]
        vals = sim_breaches[breach_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = vals[np.isfinite(vals)]
        v.check("simulation breach probabilities bounded", "empirical", bool(len(finite) and ((finite >= 0.0) & (finite <= 1.0)).all()), observed={"min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))}, expected="[0,1]")
    v.check("VIX regime table complete", "empirical", len(summary.get("vix_regime_performance", [])) == len(returns.columns) * 3, observed=len(summary.get("vix_regime_performance", [])), expected=len(returns.columns) * 3)
    leave = {row["Exclusion"]: row for row in summary.get("leave_one_out", [])}
    for key in ["No META", "No NVDA", "No TSLA", "No META/NVDA/TSLA"]:
        v.check(f"leave-one-out row present: {key}", "empirical", key in leave, observed=list(leave))
    rolling = {row["Diagnostic"]: row["Value"] for row in summary.get("rolling_oos", [])}
    v.check("rolling 36M OOS recorded", "empirical", float(rolling.get("Rolling 36M OOS months", 0)) > 0, observed=rolling.get("Rolling 36M OOS months"))


def _read_artifact_csv(v: Verifier, path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        v.check(f"{label} parses", "empirical", False, observed=path, details=f"{type(exc).__name__}: {exc}")
        return None


ROBUSTNESS_ARTIFACTS = [
    "artifacts/cv_fold_schedule.csv",
    "artifacts/cv_fold_ledger.csv",
    "artifacts/cv_split_is_oos.csv",
    "artifacts/cv_cpcv_path_metrics.csv",
    "artifacts/cv_cpcv_path_month_returns.csv",
    "artifacts/cv_pbo_summary.csv",
    "artifacts/cv_regime_performance.csv",
    "artifacts/cv_runtime_log.csv",
    "artifacts/cv_context_consistency.csv",
    "artifacts/mc_resampled_fixed_paths.csv",
    "artifacts/mc_resampled_summary.csv",
    "artifacts/mc_refit_paths.csv",
    "artifacts/mc_refit_summary.csv",
    "artifacts/mc_resampled_assumptions.csv",
    "artifacts/mc_repriced_paths.csv",
    "artifacts/mc_repriced_summary.csv",
    "artifacts/mc_repriced_paths_gauss_copula.csv",
    "artifacts/mc_repriced_summary_gauss_copula.csv",
    "artifacts/mc_repriced_assumptions.csv",
    "artifacts/mc_universe_comparison.csv",
]

ROBUSTNESS_TABLES = [
    "tables/cv_fold_performance.tex",
    "tables/cv_cpcv_distribution.tex",
    "tables/cv_regime_performance.tex",
    "tables/mc_resampled_universes.tex",
    "tables/mc_refit_stability.tex",
    "tables/mc_repriced_universes.tex",
    "tables/mc_universe_comparison.tex",
    "tables/mc_repriced_assumptions.tex",
]

ROBUSTNESS_FIGURES = [
    "figures/cv_cpcv_sharpe_distribution.pdf",
    "figures/cv_fold_sharpe_heatmap.pdf",
    "figures/mc_universe_sharpe_distributions.pdf",
]

ROBUSTNESS_SUMMARY = "tables/distributional_robustness_summary.json"

ROBUSTNESS_SUMMARY_KEYS = {
    "cv_config",
    "cv_fold_schedule",
    "cv_fold_ledger",
    "cv_cpcv_path_metrics",
    "cv_pbo",
    "cv_regime_performance",
    "cv_context_consistency",
    "mc_resampled_summary",
    "mc_refit_summary",
    "mc_repriced_summary",
    "mc_repriced_assumptions",
    "mc_universe_comparison",
    "runtime_seconds",
    "seeds",
}


def check_distributional_robustness(v: Verifier, paper_dir: Path = PAPER) -> None:
    """Audit robustness-stage artifacts when they are present.

    Fresh clones may not ship the 35-40 minute robustness outputs.  If every
    robustness artifact is absent, this records a warning-level pass and leaves
    the headline verifier usable.  Once any robustness output exists, the
    complete artifact contract is enforced.
    """

    base = Path(paper_dir)
    required_rel = ROBUSTNESS_ARTIFACTS + ROBUSTNESS_TABLES + ROBUSTNESS_FIGURES + [ROBUSTNESS_SUMMARY]
    existing = [rel for rel in required_rel if (base / rel).exists()]
    if not existing:
        v.check(
            "distributional robustness artifacts absent",
            "robustness",
            True,
            observed="absent",
            expected="optional long-running stage",
            details="Run make robustness or pass --with-robustness to the verifier to generate these diagnostics.",
            severity=WARNING,
        )
        return

    missing = [rel for rel in required_rel if not (base / rel).exists()]
    v.check(
        "distributional robustness outputs exist",
        "robustness",
        not missing,
        observed=missing,
        expected="all cv_*/mc_* artifacts, tables, figures, and summary JSON",
    )

    summary_path = base / ROBUSTNESS_SUMMARY
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            v.check(
                "distributional robustness summary schema",
                "robustness",
                ROBUSTNESS_SUMMARY_KEYS.issubset(summary.keys()),
                observed=sorted(summary.keys()),
                expected=sorted(ROBUSTNESS_SUMMARY_KEYS),
            )
        except Exception as exc:
            v.check("distributional robustness summary parses", "robustness", False, observed=type(exc).__name__, details=str(exc))
            return
    else:
        v.check("distributional robustness summary exists", "robustness", False, observed=summary_path)
        return

    art = base / "artifacts"
    table = base / "tables"
    schedule = _read_csv_for_check(v, art / "cv_fold_schedule.csv", "CV fold schedule", "robustness")
    fold_ledger = _read_csv_for_check(v, art / "cv_fold_ledger.csv", "CV fold ledger", "robustness")
    path_metrics = _read_csv_for_check(v, art / "cv_cpcv_path_metrics.csv", "CPCV path metrics", "robustness")
    path_returns = _read_csv_for_check(v, art / "cv_cpcv_path_month_returns.csv", "CPCV path month returns", "robustness")
    pbo = _read_csv_for_check(v, art / "cv_pbo_summary.csv", "CV PBO summary", "robustness")
    context = _read_csv_for_check(v, art / "cv_context_consistency.csv", "CV context consistency", "robustness")

    _check_cv_combinatorics(v, summary, schedule, path_metrics)
    _check_cv_purge_embargo(v, summary, schedule, fold_ledger, path_returns)
    _check_cpcv_path_coverage(v, path_metrics, path_returns)
    _check_pbo_bounds(v, summary, pbo)
    _check_cv_context_consistency(v, context)
    _check_mc_path_counts(v, summary, art)
    _check_repriced_assumptions(v, summary, art / "mc_repriced_assumptions.csv")
    _check_robustness_tex_headers(v, [table / Path(rel).name for rel in ROBUSTNESS_TABLES])


def _read_csv_for_check(v: Verifier, path: Path, label: str, category: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        v.check(f"{label} is nonempty", category, False, observed=path)
        return pd.DataFrame()
    except Exception as exc:
        v.check(f"{label} parses", category, False, observed=path, details=f"{type(exc).__name__}: {exc}")
        return None


def _check_cv_combinatorics(
    v: Verifier,
    summary: dict[str, Any],
    schedule: pd.DataFrame | None,
    path_metrics: pd.DataFrame | None,
) -> None:
    config = summary.get("cv_config", {}) if isinstance(summary.get("cv_config"), dict) else {}
    try:
        n_groups = int(config.get("n_groups"))
        n_test_groups = int(config.get("n_test_groups"))
    except Exception:
        v.check("CV config has combinatoric fields", "robustness", False, observed=config)
        return

    if schedule is not None and not schedule.empty and {"scheme", "fold_id"}.issubset(schedule.columns):
        kfold_count = int(schedule.loc[schedule["scheme"].astype(str).eq("kfold"), "fold_id"].nunique())
        cpcv_count = int(schedule.loc[schedule["scheme"].astype(str).eq("cpcv"), "fold_id"].nunique())
        v.check("blocked k-fold count matches config", "robustness", kfold_count == n_groups, observed=kfold_count, expected=n_groups)
        v.check(
            "CPCV split count matches config",
            "robustness",
            cpcv_count == math.comb(n_groups, n_test_groups),
            observed=cpcv_count,
            expected=math.comb(n_groups, n_test_groups),
        )
    else:
        v.check("CV schedule has scheme/fold_id columns", "robustness", False, observed=list(schedule.columns) if schedule is not None else "missing")

    if path_metrics is not None and not path_metrics.empty and {"path_id", "status"}.issubset(path_metrics.columns):
        status = path_metrics["status"].astype(str)
        complete_count = int(path_metrics.loc[status.eq("complete"), "path_id"].nunique())
        expected_paths = math.comb(n_groups - 1, n_test_groups - 1)
        v.check(
            "complete CPCV path count matches config",
            "robustness",
            complete_count == expected_paths,
            observed={"complete": complete_count, "status_counts": status.value_counts().to_dict()},
            expected=expected_paths,
        )
    else:
        v.check("CPCV path metrics have path_id/status columns", "robustness", False, observed=list(path_metrics.columns) if path_metrics is not None else "missing")


def _check_cv_purge_embargo(
    v: Verifier,
    summary: dict[str, Any],
    schedule: pd.DataFrame | None,
    fold_ledger: pd.DataFrame | None,
    path_returns: pd.DataFrame | None,
) -> None:
    config = summary.get("cv_config", {}) if isinstance(summary.get("cv_config"), dict) else {}
    try:
        n_groups = int(config.get("n_groups"))
        purge = int(config.get("purge_months", 0))
        embargo = int(config.get("embargo_months", 0))
    except Exception:
        v.check("CV purge config available", "robustness", False, observed=config)
        return
    if schedule is None or schedule.empty:
        v.check("CV purge schedule available", "robustness", False, observed="missing")
        return
    if path_returns is None or path_returns.empty or "return_date" not in path_returns.columns:
        v.check("CPCV path returns provide month grid", "robustness", False, observed=list(path_returns.columns) if path_returns is not None else "missing")
        return

    month_grid = pd.DatetimeIndex(pd.to_datetime(path_returns["return_date"], errors="coerce").dropna().unique()).sort_values()
    if len(month_grid) == 0:
        v.check("CPCV path returns month grid nonempty", "robustness", False, observed=0)
        return
    group_ids = np.empty(len(month_grid), dtype=int)
    for group_id, positions in enumerate(np.array_split(np.arange(len(month_grid)), n_groups)):
        group_ids[positions] = group_id
    pos_by_date = {pd.Timestamp(dt): pos for pos, dt in enumerate(month_grid)}

    violations: list[dict[str, Any]] = []
    checked = 0
    for _, row in schedule.iterrows():
        fold_id = str(row.get("fold_id", ""))
        groups = _parse_group_ids(row.get("test_groups", ""))
        if not groups:
            violations.append({"fold_id": fold_id, "issue": "missing test_groups"})
            continue
        test_pos = sorted(int(pos) for pos, gid in enumerate(group_ids) if int(gid) in groups)
        if not test_pos:
            violations.append({"fold_id": fold_id, "issue": "empty reconstructed test positions", "groups": groups})
            continue
        expected_purge_pos: set[int] = set()
        expected_embargo_pos: set[int] = set()
        for start, end in _contiguous_position_blocks(test_pos):
            expected_purge_pos.update(range(max(0, start - purge), min(len(month_grid) - 1, end + purge) + 1))
            expected_embargo_pos.update(range(end + purge + 1, min(len(month_grid) - 1, end + purge + embargo) + 1))
        test_set = set(test_pos)
        expected_purge_pos.difference_update(test_set)
        expected_embargo_pos.difference_update(test_set)
        expected_embargo_pos.difference_update(expected_purge_pos)

        parsed_purge = _parse_semicolon_dates(row.get("purged_dates", ""))
        parsed_embargo = _parse_semicolon_dates(row.get("embargoed_dates", ""))
        parsed_purge_pos = {pos_by_date[dt] for dt in parsed_purge if dt in pos_by_date}
        parsed_embargo_pos = {pos_by_date[dt] for dt in parsed_embargo if dt in pos_by_date}
        unknown_dates = sorted((parsed_purge | parsed_embargo).difference(pos_by_date))
        train_pos = set(range(len(month_grid))).difference(test_set).difference(parsed_purge_pos).difference(parsed_embargo_pos)
        n_train_observed = _maybe_int(row.get("n_train"))

        if expected_purge_pos != parsed_purge_pos:
            violations.append(
                {
                    "fold_id": fold_id,
                    "issue": "purged_dates mismatch",
                    "missing": _dates_from_positions(month_grid, expected_purge_pos - parsed_purge_pos),
                    "extra": _dates_from_positions(month_grid, parsed_purge_pos - expected_purge_pos),
                }
            )
        if expected_embargo_pos != parsed_embargo_pos:
            violations.append(
                {
                    "fold_id": fold_id,
                    "issue": "embargoed_dates mismatch",
                    "missing": _dates_from_positions(month_grid, expected_embargo_pos - parsed_embargo_pos),
                    "extra": _dates_from_positions(month_grid, parsed_embargo_pos - expected_embargo_pos),
                }
            )
        leaked = sorted(train_pos.intersection(expected_purge_pos | expected_embargo_pos))
        if leaked:
            violations.append({"fold_id": fold_id, "issue": "train month inside purge/embargo window", "dates": _dates_from_positions(month_grid, leaked)})
        if unknown_dates:
            violations.append({"fold_id": fold_id, "issue": "scheduled date outside month grid", "dates": [dt.strftime("%Y-%m-%d") for dt in unknown_dates]})
        if n_train_observed is not None and int(n_train_observed) != len(train_pos):
            violations.append({"fold_id": fold_id, "issue": "n_train mismatch", "observed": int(n_train_observed), "recomputed": len(train_pos)})
        checked += 1

    ledger_folds = int(fold_ledger["fold_id"].nunique()) if fold_ledger is not None and "fold_id" in fold_ledger else 0
    v.check(
        "CV purge/embargo invariant recomputed from schedule",
        "robustness",
        checked > 0 and not violations,
        observed=violations[:10],
        expected=f"no train month within purge={purge}/embargo={embargo} windows; ledger_folds={ledger_folds}",
    )


def _check_cpcv_path_coverage(
    v: Verifier,
    path_metrics: pd.DataFrame | None,
    path_returns: pd.DataFrame | None,
) -> None:
    needed = {"path_id", "strategy", "basis", "status"}
    if path_metrics is None or path_metrics.empty or not needed.issubset(path_metrics.columns):
        v.check("CPCV path metrics schema for coverage", "robustness", False, observed=list(path_metrics.columns) if path_metrics is not None else "missing", expected=sorted(needed))
        return
    needed_returns = {"path_id", "return_date", "strategy", "basis"}
    if path_returns is None or path_returns.empty or not needed_returns.issubset(path_returns.columns):
        v.check("CPCV path returns schema for coverage", "robustness", False, observed=list(path_returns.columns) if path_returns is not None else "missing", expected=sorted(needed_returns))
        return

    ret = path_returns.copy()
    ret["return_date"] = pd.to_datetime(ret["return_date"], errors="coerce")
    month_grid = set(pd.DatetimeIndex(ret["return_date"].dropna().unique()))
    complete = path_metrics[path_metrics["status"].astype(str).eq("complete")]
    violations: list[dict[str, Any]] = []
    checked = 0
    for _, row in complete[["path_id", "strategy", "basis"]].drop_duplicates().iterrows():
        mask = (
            ret["path_id"].astype(str).eq(str(row["path_id"]))
            & ret["strategy"].astype(str).eq(str(row["strategy"]))
            & ret["basis"].astype(str).eq(str(row["basis"]))
        )
        sub = ret.loc[mask, "return_date"].dropna()
        counts = sub.value_counts()
        missing = sorted(month_grid.difference(set(sub)))
        duplicated = sorted(counts[counts > 1].index)
        if len(sub) != len(month_grid) or missing or duplicated:
            violations.append(
                {
                    "path_id": row["path_id"],
                    "strategy": row["strategy"],
                    "basis": row["basis"],
                    "rows": int(len(sub)),
                    "months": int(len(month_grid)),
                    "missing": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in missing[:5]],
                    "duplicated": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in duplicated[:5]],
                }
            )
        checked += 1
    v.check(
        "complete CPCV paths cover every month exactly once",
        "robustness",
        checked > 0 and not violations,
        observed=violations[:10],
        expected="one row per return month for each complete path/strategy/basis",
    )


def _check_pbo_bounds(v: Verifier, summary: dict[str, Any], pbo_csv: pd.DataFrame | None) -> None:
    pbo = pd.DataFrame(summary.get("cv_pbo", []))
    if pbo.empty and pbo_csv is not None:
        pbo = pbo_csv
    if pbo.empty or "PBO" not in pbo.columns:
        v.check("PBO summary schema", "robustness", False, observed=list(pbo.columns), expected="PBO")
        return
    values = pd.to_numeric(pbo["PBO"], errors="coerce")
    ok = bool(values.notna().all() and ((values >= 0.0) & (values <= 1.0)).all())
    v.check("PBO values bounded", "robustness", ok, observed=values.to_list(), expected="[0, 1]")


def _check_cv_context_consistency(v: Verifier, context: pd.DataFrame | None) -> None:
    if context is None or context.empty or not {"status", "max_abs_diff"}.issubset(context.columns):
        v.check("CV context consistency schema", "robustness", False, observed=list(context.columns) if context is not None else "missing", expected="status/max_abs_diff")
        return
    ok_rows = context[context["status"].astype(str).eq("ok")]
    values = pd.to_numeric(ok_rows["max_abs_diff"], errors="coerce")
    passed = bool(values.notna().all() and (values < 1e-8).all())
    v.check(
        "CV context consistency diffs are negligible",
        "robustness",
        passed,
        observed={"max": float(values.max()) if len(values) else np.nan, "rows": int(len(ok_rows))},
        expected="<1e-8 for status ok",
    )


def _check_mc_path_counts(v: Verifier, summary: dict[str, Any], art: Path) -> None:
    expected = _expected_mc_counts(summary, art)
    fixed = _read_csv_for_check(v, art / "mc_resampled_fixed_paths.csv", "MC fixed paths", "robustness")
    refit = _read_csv_for_check(v, art / "mc_refit_paths.csv", "MC refit paths", "robustness")
    repriced = _read_csv_for_check(v, art / "mc_repriced_paths.csv", "MC repriced paths", "robustness")
    gauss = _read_csv_for_check(v, art / "mc_repriced_paths_gauss_copula.csv", "MC gaussian-copula repriced paths", "robustness")

    if fixed is not None and not fixed.empty and {"universe_family", "basis", "path_id"}.issubset(fixed.columns):
        counts = fixed.groupby(["universe_family", "basis"], dropna=False)["path_id"].nunique()
        target = expected.get("fixed")
        v.check("MC fixed-weight path counts match config", "robustness", target is not None and bool((counts == target).all()), observed=counts.to_dict(), expected=target)
    else:
        v.check("MC fixed-weight path schema", "robustness", False, observed=list(fixed.columns) if fixed is not None else "missing")

    if refit is not None and not refit.empty and "path_id" in refit.columns:
        count = int(refit["path_id"].nunique())
        target = expected.get("refit")
        v.check("MC refit path count matches config", "robustness", target is not None and count == target, observed=count, expected=target)
    else:
        v.check("MC refit path schema", "robustness", False, observed=list(refit.columns) if refit is not None else "missing")

    if repriced is not None and not repriced.empty and {"method", "path_id"}.issubset(repriced.columns):
        counts = repriced.groupby("method", dropna=False)["path_id"].nunique()
        target = expected.get("reprice")
        joint_ok = target is not None and bool((counts == target).all())
        v.check("MC repriced path counts match config", "robustness", joint_ok, observed=counts.to_dict(), expected=target)
    else:
        v.check("MC repriced path schema", "robustness", False, observed=list(repriced.columns) if repriced is not None else "missing")

    if gauss is not None and not gauss.empty and {"method", "path_id"}.issubset(gauss.columns):
        counts = gauss.groupby("method", dropna=False)["path_id"].nunique()
        target = expected.get("reprice_sensitivity")
        v.check("MC gaussian-copula path counts match config", "robustness", target is not None and bool((counts == target).all()), observed=counts.to_dict(), expected=target)
    else:
        v.check("MC gaussian-copula path schema", "robustness", False, observed=list(gauss.columns) if gauss is not None else "missing")


def _check_repriced_assumptions(v: Verifier, summary: dict[str, Any], csv_path: Path) -> None:
    assumptions = pd.DataFrame(summary.get("mc_repriced_assumptions", []))
    if assumptions.empty and csv_path.exists():
        assumptions = pd.read_csv(csv_path)
    if assumptions.empty:
        v.check("repriced assumptions available", "robustness", False, observed=csv_path)
        return
    text = " ".join(assumptions.fillna("").astype(str).stack().tolist()).lower()
    has_one_step = "pricing tenor rule" in text and "one-step" in text and "1-month" in text
    has_vx_front = "vix forward convention" in text and ("vx-front" in text or "vx_front" in text) and "vix level" in text
    v.check("repriced assumptions include one-step tenor row", "robustness", has_one_step, observed="Pricing Tenor Rule", expected="one-step 1-month")
    v.check("repriced assumptions include VX-front convention row", "robustness", has_vx_front, observed="VIX Forward Convention", expected="VX-front premium / VIX-level settlement")


def _check_robustness_tex_headers(v: Verifier, paths: list[Path]) -> None:
    violations = []
    parsed = 0
    for path in paths:
        if not path.exists():
            continue
        header, _rows = _parse_tex_table(path)
        if not header:
            violations.append({"table": path.name, "issue": "no parsed header"})
            continue
        parsed += 1
        raw = [cell for cell in header if re.search(r"(?<!\\)_", cell)]
        if raw:
            violations.append({"table": path.name, "raw_underscore_headers": raw})
    v.check(
        "robustness table headers escape underscores",
        "robustness",
        parsed > 0 and not violations,
        observed=violations,
        expected="no raw underscore in parsed header rows",
    )


def _expected_mc_counts(summary: dict[str, Any], art: Path) -> dict[str, int | None]:
    resample_config = _first_mapping(summary, ["resample_config", "mc_resample_config", "mc_config"])
    reprice_config = _first_mapping(summary, ["reprice_config", "mc_reprice_config", "mc_config"])
    out: dict[str, int | None] = {
        "fixed": _first_int(resample_config, ["n_paths", "fixed_weight_path_count", "fixed_paths", "mc_paths", "resample_paths"]),
        "refit": _first_int(resample_config, ["n_refit_paths", "refit_path_count", "mc_refit_paths", "refit_paths"]),
        "reprice": _first_int(reprice_config, ["n_paths", "reprice_path_count", "mc_reprice_paths", "reprice_paths"]),
        "reprice_sensitivity": _first_int(reprice_config, ["n_sensitivity_paths", "sensitivity_path_count", "gaussian_copula_paths"]),
    }
    resampled_assumptions = art / "mc_resampled_assumptions.csv"
    if resampled_assumptions.exists():
        frame = pd.read_csv(resampled_assumptions)
        out["fixed"] = out["fixed"] or _assumption_int(frame, "Fixed Weight Path Count")
        out["refit"] = out["refit"] or _assumption_int(frame, "Refit Path Count")
    reprice_assumptions = pd.DataFrame(summary.get("mc_repriced_assumptions", []))
    if reprice_assumptions.empty and (art / "mc_repriced_assumptions.csv").exists():
        reprice_assumptions = pd.read_csv(art / "mc_repriced_assumptions.csv")
    if not reprice_assumptions.empty:
        out["reprice"] = out["reprice"] or _assumption_int(reprice_assumptions, "Path Count")
    # The current summary schema records the default repriced count but not the
    # gaussian-copula sensitivity count.  Keep the stable stage default as the
    # fallback so the shipped artifact inventory remains auditable.
    out["reprice_sensitivity"] = out["reprice_sensitivity"] or 250
    return out


def _first_mapping(summary: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _first_int(mapping: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        if key in mapping:
            val = _maybe_int(mapping[key])
            if val is not None:
                return val
    return None


def _assumption_int(frame: pd.DataFrame, item: str) -> int | None:
    if frame.empty:
        return None
    key_col = "Assumption" if "Assumption" in frame.columns else "Item" if "Item" in frame.columns else None
    if key_col is None or "Value" not in frame.columns:
        return None
    match = frame[frame[key_col].astype(str).str.lower().eq(item.lower())]
    if match.empty:
        return None
    return _maybe_int(match["Value"].iloc[0])


def _maybe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _parse_group_ids(value: Any) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", str(value))}


def _parse_semicolon_dates(value: Any) -> set[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    out: set[pd.Timestamp] = set()
    for part in str(value).split(";"):
        text = part.strip()
        if not text:
            continue
        ts = pd.to_datetime(text, errors="coerce")
        if not pd.isna(ts):
            out.add(pd.Timestamp(ts))
    return out


def _contiguous_position_blocks(positions: Sequence[int]) -> list[tuple[int, int]]:
    ordered = sorted(int(p) for p in positions)
    if not ordered:
        return []
    blocks: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        blocks.append((start, prev))
        start = prev = pos
    blocks.append((start, prev))
    return blocks


def _dates_from_positions(month_grid: pd.DatetimeIndex, positions: Iterable[int]) -> list[str]:
    return [pd.Timestamp(month_grid[int(pos)]).strftime("%Y-%m-%d") for pos in sorted(positions)]


def _ledger_key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    if frame is None or frame.empty or not set(columns).issubset(frame.columns):
        return set()
    out = frame[columns].copy()
    if "return_date" in out:
        out["return_date"] = pd.to_datetime(out["return_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return set(map(tuple, out.fillna("").astype(str).to_numpy()))


def check_pipeline_extension_artifacts(v: Verifier) -> None:
    repaired_path = ART_DIR / "repaired_trade_ledger.csv"
    repaired = _read_artifact_csv(v, repaired_path, "repaired trade ledger")
    if repaired is not None:
        required_cols = {
            "return_date",
            "strategy",
            "scenario",
            "asset_id",
            "repair_reason",
            "decision_mark",
            "effective_fill_price",
            "fill_fraction",
        }
        has_cols = required_cols.issubset(repaired.columns)
        v.check("repaired trade ledger schema", "empirical", has_cols, observed=list(repaired.columns), expected=sorted(required_cols))
        if "scenario" in repaired:
            labels = sorted(set(repaired["scenario"].dropna().astype(str)))
            bad = [label for label in labels if not label.endswith("_repaired")]
            v.check("repaired trade ledger scenarios are suffixed", "empirical", not bad, observed=bad, expected="all scenario labels end with _repaired")
        if "repair_reason" in repaired:
            hard_pattern = r"missing_cost_input|assignment|dividend|exercise|deep_itm|hard_to_borrow"
            reasons = repaired["repair_reason"].fillna("").astype(str).str.lower()
            hard_rows = repaired.loc[reasons.str.contains(hard_pattern, regex=True, na=False)]
            v.check(
                "repaired trade ledger excludes hard-gate reasons",
                "empirical",
                hard_rows.empty,
                observed=hard_rows[["return_date", "strategy", "scenario", "asset_id", "repair_reason"]].head(10).to_dict(orient="records") if has_cols else int(len(hard_rows)),
                expected="no missing_cost_input, assignment, dividend, exercise, deep-ITM, or hard-to-borrow repair reasons",
            )
        if {"repair_reason", "decision_mark", "effective_fill_price", "fill_fraction"}.issubset(repaired.columns):
            reasons = repaired["repair_reason"].fillna("").astype(str)
            quote_reasons = (
                reasons.str.replace("capacity_partial_fill", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.strip()
            )
            quote_mask = quote_reasons.ne("")
            marks = pd.to_numeric(repaired["decision_mark"], errors="coerce")
            fills = pd.to_numeric(repaired["effective_fill_price"], errors="coerce")
            deviation = (fills / marks - 1.0).abs()
            quote_dev = deviation.loc[quote_mask]
            quote_ok = bool(quote_dev.notna().all() and (quote_dev <= 0.10 + 1e-9).all())
            v.check(
                "quote repairs respect 10 percent fill band",
                "empirical",
                quote_ok,
                observed=float(quote_dev.max()) if len(quote_dev) else 0.0,
                expected="<=0.10 + 1e-9",
            )
            capacity_mask = reasons.str.contains("capacity_partial_fill", regex=False, na=False)
            fractions = pd.to_numeric(repaired.loc[capacity_mask, "fill_fraction"], errors="coerce")
            frac_ok = bool(fractions.notna().all() and ((fractions >= 0.10 - 1e-9) & (fractions <= 1.0 + 1e-9)).all())
            v.check(
                "capacity partial repairs respect fill fraction bounds",
                "empirical",
                frac_ok,
                observed={"min": float(fractions.min()) if len(fractions) else 1.0, "max": float(fractions.max()) if len(fractions) else 1.0},
                expected="[0.10 - 1e-9, 1.0 + 1e-9]",
            )

    net_repaired_path = ART_DIR / "net_strategy_returns_by_cost_scenario_repaired.csv"
    net_repaired = _read_artifact_csv(v, net_repaired_path, "repaired scenario returns")
    if net_repaired is not None:
        scenario_labels = sorted({str(c).rsplit("::", 1)[-1] for c in net_repaired.columns if "::" in str(c)})
        bad = [label for label in scenario_labels if not label.endswith("_repaired")]
        v.check(
            "repaired scenario return columns are suffixed",
            "empirical",
            bool(scenario_labels) and not bad,
            observed=bad if bad else scenario_labels,
            expected="all scenario labels end with _repaired",
        )

    rejected_repaired = _read_artifact_csv(v, ART_DIR / "rejected_trade_ledger_repaired.csv", "repaired rejected trade ledger")
    if repaired is not None and rejected_repaired is not None:
        hard_pattern = r"missing_cost_input|assignment|dividend|exercise|deep_itm|hard_to_borrow"
        if {"return_date", "strategy", "scenario", "asset_id"}.issubset(repaired.columns) and {"return_date", "strategy", "scenario", "asset_id", "reject_reason"}.issubset(rejected_repaired.columns):
            hard_rejected = rejected_repaired.loc[
                rejected_repaired["reject_reason"].fillna("").astype(str).str.lower().str.contains(hard_pattern, regex=True, na=False)
            ]
            key_cols = ["return_date", "strategy", "scenario", "asset_id"]
            overlap = _ledger_key_set(repaired, key_cols) & _ledger_key_set(hard_rejected, key_cols)
            v.check(
                "repaired trades are not hard-gate rejections",
                "empirical",
                not overlap,
                observed=sorted(overlap)[:10],
                expected="no repaired row also rejected for missing_cost_input or assignment/dividend hard gates",
            )

    sortino_costs = _read_artifact_csv(v, ART_DIR / "sortino_entry_costs.csv", "Sortino entry costs")
    if sortino_costs is not None:
        has_entry_cost = "entry_cost" in sortino_costs.columns
        v.check("Sortino entry cost schema", "empirical", has_entry_cost, observed=list(sortino_costs.columns), expected="entry_cost")
        if has_entry_cost:
            entry_cost = pd.to_numeric(sortino_costs["entry_cost"], errors="coerce")
            vals = entry_cost.to_numpy(float)
            ok = bool(np.isfinite(vals).all() and (vals >= 0.0).all())
            v.check(
                "Sortino entry costs finite nonnegative",
                "empirical",
                ok,
                observed={"min": float(entry_cost.min()) if len(entry_cost) else 0.0, "max": float(entry_cost.max()) if len(entry_cost) else 0.0},
                expected="finite and >= 0",
            )

    spread_sources = _read_artifact_csv(v, ART_DIR / "cost_input_spread_source_coverage.csv", "cost input spread source coverage")
    if spread_sources is not None:
        has_source = "relative_spread_source" in spread_sources.columns
        v.check("cost input spread source coverage schema", "empirical", has_source, observed=list(spread_sources.columns), expected="relative_spread_source")
        if has_source:
            sources = set(spread_sources["relative_spread_source"].dropna().astype(str))
            allowed = {"panel_cbbo", "surface_cbbo", "default"}
            v.check("cost input spread sources are recognized", "empirical", sources.issubset(allowed), observed=sorted(sources), expected=sorted(allowed))

    manifest = _read_artifact_csv(v, ART_DIR / "data_extension_manifest.csv", "data extension manifest")
    if manifest is not None:
        has_dataset = "dataset" in manifest.columns
        v.check("data extension manifest schema", "empirical", has_dataset, observed=list(manifest.columns), expected="dataset")
        if has_dataset:
            datasets = set(manifest["dataset"].dropna().astype(str))
            required = {"opra_surface_full_day_cbbo", "opra_vix_chain_*", "opra_{UND}_slices_*"}
            v.check(
                "data extension manifest records expected dataset families",
                "empirical",
                required.issubset(datasets),
                observed=sorted(datasets),
                expected=sorted(required),
            )


def _parse_tex_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Parse a booktabs tabular written by ``_write_latex_table``."""

    if not path.exists():
        return [], []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    header: list[str] = []
    rows: list[list[str]] = []
    seen_toprule = False
    seen_midrule = False
    for ln in lines:
        if ln.startswith("\\toprule"):
            seen_toprule = True
            continue
        if ln.startswith("\\midrule"):
            seen_midrule = True
            continue
        if ln.startswith("\\bottomrule"):
            break
        if not ln.endswith("\\\\"):
            continue
        cells = [c.strip() for c in ln[:-2].split("&")]
        if seen_toprule and not seen_midrule and not header:
            header = cells
        elif seen_midrule:
            rows.append(cells)
    return header, rows


def _tex_cell_float(cell: str) -> float:
    text = str(cell).strip().replace("\\%", "%")
    if text in {"", "--", "nan", "NaN"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _tex_cell_text(cell: str) -> str:
    return str(cell).strip().replace("\\_", "_").replace("\\&", "&").replace("\\%", "%")


def check_final_inference_panel(v: Verifier) -> None:
    """Audit the compact final inference panel against locked breadth artifacts."""

    table_path = TABLE_DIR / "short_inference_panel.tex"
    returns_path = BREADTH_ROBUSTNESS_DIR / "breadth_strategy_returns_net.csv"
    scoreboard_path = BREADTH_ROBUSTNESS_DIR / "final_result_scoreboard.csv"
    panel_path = BREADTH_ROBUSTNESS_DIR / "final_inference_panel.csv"
    p1_path = BREADTH_DIR / "p1_regularization_results.csv"

    inputs = [table_path, returns_path, scoreboard_path, panel_path, p1_path]
    missing = [str(path.relative_to(PAPER)) for path in inputs if not path.exists()]
    v.check(
        "final inference panel inputs exist",
        "inference",
        not missing,
        observed=missing,
        expected="short table plus locked breadth returns/scoreboard/inference/P1 artifacts",
    )
    if missing:
        return

    header, rows = _parse_tex_table(table_path)
    idx = {_tex_cell_text(name): i for i, name in enumerate(header)}
    needed_table_cols = {"Config", "Net Sharpe", "CI lo", "CI hi", "PSR", "DSR", "p stock", "p naive"}
    missing_cols = sorted(needed_table_cols.difference(idx))
    v.check("final inference panel table schema", "inference", bool(header) and not missing_cols, observed=header, expected=sorted(needed_table_cols))
    if missing_cols:
        return

    try:
        static_returns = pd.read_csv(returns_path).apply(pd.to_numeric, errors="coerce")
        scoreboard = pd.read_csv(scoreboard_path)
        panel = pd.read_csv(panel_path)
        p1 = pd.read_csv(p1_path)
    except Exception as exc:
        v.check("final inference panel artifacts parse", "inference", False, observed=type(exc).__name__, details=str(exc))
        return

    score_required = {"config", "config_label", "e1_net_sharpe"}
    panel_required = {"config", "basis", "dsr_trials"}
    p1_required = {"config", "point_id"}
    schema_missing = {
        "scoreboard": sorted(score_required.difference(scoreboard.columns)),
        "final_inference_panel": sorted(panel_required.difference(panel.columns)),
        "p1_regularization_results": sorted(p1_required.difference(p1.columns)),
    }
    schema_ok = not any(schema_missing.values())
    v.check("final inference panel artifact schemas", "inference", schema_ok, observed=schema_missing, expected="required columns present")
    if not schema_ok:
        return

    row_by_label = {
        _tex_cell_text(cells[idx["Config"]]): cells
        for cells in rows
        if idx["Config"] < len(cells)
    }
    score_by_config = scoreboard.set_index("config")

    range_violations: list[dict[str, Any]] = []
    for cells in rows:
        if len(cells) <= max(idx.values()):
            range_violations.append({"row": cells, "issue": "short row"})
            continue
        label = _tex_cell_text(cells[idx["Config"]])
        net = _tex_cell_float(cells[idx["Net Sharpe"]])
        lo = _tex_cell_float(cells[idx["CI lo"]])
        hi = _tex_cell_float(cells[idx["CI hi"]])
        if not (np.isfinite(lo) and np.isfinite(net) and np.isfinite(hi) and lo <= net <= hi):
            range_violations.append({"config": label, "issue": "Net Sharpe outside CI", "lo": lo, "net": net, "hi": hi})
        for col in ["PSR", "DSR", "p stock", "p naive"]:
            value = _tex_cell_float(cells[idx[col]])
            if not (np.isfinite(value) and 0.0 <= value <= 1.0):
                range_violations.append({"config": label, "column": col, "value": value})
    v.check(
        "final inference panel intervals and probabilities valid",
        "inference",
        not range_violations,
        observed=range_violations[:10],
        expected="CI lo <= Net Sharpe <= CI hi and probabilities in [0,1]",
    )

    for config in BREADTH_CONFIG_ORDER:
        label = str(score_by_config.loc[config, "config_label"]) if config in score_by_config.index else config
        cells = row_by_label.get(label)
        col = f"{config} {BREADTH_PRIMARY_STRATEGY}"
        if cells is None:
            v.check(f"final inference panel row present: {config}", "inference", False, observed=label, expected="row in short_inference_panel.tex")
            continue
        if col not in static_returns.columns:
            v.check(f"final inference return column present: {config}", "inference", False, observed=col, expected="breadth_strategy_returns_net.csv column")
            continue
        x = pd.to_numeric(static_returns[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        ann_vol = float(np.std(x, ddof=1) * np.sqrt(12.0)) if len(x) >= 2 else float("nan")
        recomputed = float(np.mean(x) * 12.0 / ann_vol) if np.isfinite(ann_vol) and ann_vol > 0.0 else float("nan")
        table_value = _tex_cell_float(cells[idx["Net Sharpe"]])
        scoreboard_value = float(score_by_config.loc[config, "e1_net_sharpe"]) if config in score_by_config.index else float("nan")
        v.check(
            f"final inference table Net Sharpe matches returns: {config}",
            "inference",
            np.isfinite(recomputed) and abs(table_value - recomputed) < 2e-3,
            observed={"table": table_value, "recomputed": recomputed},
            expected="<2e-3 absolute difference",
        )
        v.check(
            f"final inference scoreboard Net Sharpe matches returns: {config}",
            "inference",
            np.isfinite(recomputed) and abs(recomputed - scoreboard_value) < 1e-9,
            observed={"recomputed": recomputed, "scoreboard": scoreboard_value},
            expected="<1e-9 absolute difference",
        )

    static_panel = panel.loc[panel["basis"].astype(str).eq("static")].copy()
    trial_violations: list[dict[str, Any]] = []
    for config in BREADTH_CONFIG_ORDER:
        sub = p1.loc[p1["config"].astype(str).eq(config)].copy()
        trials = int(sub.loc[~sub["point_id"].astype(str).isin(BREADTH_NAIVE_POINT_IDS), "point_id"].astype(str).nunique())
        panel_sub = static_panel.loc[static_panel["config"].astype(str).eq(config)]
        observed = _maybe_int(panel_sub["dsr_trials"].iloc[0]) if not panel_sub.empty else None
        if trials != 22 or observed != trials:
            trial_violations.append({"config": config, "recomputed": trials, "final_inference_panel": observed, "expected": 22})
    v.check(
        "final inference DSR trial counts match P1 grid",
        "inference",
        not trial_violations,
        observed=trial_violations,
        expected="22 non-naive unique point_id values per static config",
    )


def check_e1_ablation_and_concentration(v: Verifier) -> None:
    """Audit the locked E1 ablation and concentration paper tables."""

    scoreboard_path = BREADTH_ROBUSTNESS_DIR / "final_result_scoreboard.csv"
    realized_path = BREADTH_ROBUSTNESS_DIR / "breadth_realized_candidate_summary.csv"
    ablation_path = TABLE_DIR / "short_e1_channel_ablation.tex"
    concentration_path = TABLE_DIR / "short_e1_concentration.tex"

    scoreboard = pd.DataFrame()
    if scoreboard_path.exists():
        try:
            scoreboard = pd.read_csv(scoreboard_path)
        except Exception as exc:
            v.check("E1 ablation scoreboard parses", "inference", False, observed=type(exc).__name__, details=str(exc))
    else:
        v.check("E1 ablation scoreboard exists", "inference", False, observed=scoreboard_path)

    if ablation_path.exists():
        header, rows = _parse_tex_table(ablation_path)
        idx = {_tex_cell_text(name): i for i, name in enumerate(header)}
        missing_cols = [col for col in ["Arm", *BREADTH_CONFIG_ORDER] if col not in idx]
        v.check("E1 channel ablation table schema", "inference", bool(header) and not missing_cols, observed=header, expected=["Arm", *BREADTH_CONFIG_ORDER])
        if not missing_cols:
            finite_violations: list[dict[str, Any]] = []
            full_row: list[str] | None = None
            for cells in rows:
                if len(cells) <= max(idx.values()):
                    finite_violations.append({"row": cells, "issue": "short row"})
                    continue
                arm = _tex_cell_text(cells[idx["Arm"]])
                if arm == "Full E1":
                    full_row = cells
                for config in BREADTH_CONFIG_ORDER:
                    value = _tex_cell_float(cells[idx[config]])
                    if not np.isfinite(value):
                        finite_violations.append({"arm": arm, "config": config, "value": value})
            v.check("E1 channel ablation table cells finite", "inference", not finite_violations, observed=finite_violations[:10], expected="all numeric cells finite")
            if full_row is None:
                v.check("E1 channel ablation Full E1 row present", "inference", False, observed=[_tex_cell_text(r[idx["Arm"]]) for r in rows if idx["Arm"] < len(r)])
            elif {"config", "e1_net_sharpe"}.issubset(scoreboard.columns):
                score_by_config = scoreboard.set_index("config")
                violations = []
                for config in BREADTH_CONFIG_ORDER:
                    table_value = _tex_cell_float(full_row[idx[config]])
                    expected = float(score_by_config.loc[config, "e1_net_sharpe"]) if config in score_by_config.index else float("nan")
                    if not (np.isfinite(table_value) and np.isfinite(expected) and abs(table_value - expected) < 2e-3):
                        violations.append({"config": config, "table": table_value, "scoreboard": expected})
                v.check(
                    "E1 channel ablation Full E1 row matches scoreboard",
                    "inference",
                    not violations,
                    observed=violations,
                    expected="<2e-3 vs final_result_scoreboard.csv e1_net_sharpe",
                )
            else:
                v.check("E1 channel ablation scoreboard schema", "inference", False, observed=list(scoreboard.columns), expected="config/e1_net_sharpe")
    else:
        v.check(
            "E1 channel ablation table exists",
            "inference",
            False,
            observed=ablation_path,
            expected="required output of final paper build",
        )

    if concentration_path.exists():
        header, rows = _parse_tex_table(concentration_path)
        idx = {_tex_cell_text(name): i for i, name in enumerate(header)}
        needed_cols = {"Config", "Candidates", "Active", "Top 5 share", "Deployed gross", "At cap share"}
        missing_cols = sorted(needed_cols.difference(idx))
        v.check("E1 concentration table schema", "inference", bool(header) and not missing_cols, observed=header, expected=sorted(needed_cols))
        if not missing_cols:
            realized = pd.DataFrame()
            if realized_path.exists():
                try:
                    realized = pd.read_csv(realized_path)
                except Exception as exc:
                    v.check("E1 realized candidate summary parses", "inference", False, observed=type(exc).__name__, details=str(exc))
            else:
                v.check("E1 realized candidate summary exists", "inference", False, observed=realized_path)
            realized_ok = {"config", "strategy", "deployed_gross"}.issubset(realized.columns)
            v.check("E1 realized candidate summary schema", "inference", realized_ok, observed=list(realized.columns), expected="config/strategy/deployed_gross")
            expected_gross = pd.Series(dtype=float)
            if realized_ok:
                e1 = realized.loc[realized["strategy"].astype(str).eq(BREADTH_PRIMARY_STRATEGY), ["config", "deployed_gross"]].copy()
                e1["deployed_gross"] = pd.to_numeric(e1["deployed_gross"], errors="coerce")
                expected_gross = e1.drop_duplicates("config").set_index("config")["deployed_gross"]

            violations: list[dict[str, Any]] = []
            for cells in rows:
                if len(cells) <= max(idx.values()):
                    violations.append({"row": cells, "issue": "short row"})
                    continue
                config = _tex_cell_text(cells[idx["Config"]])
                candidates = _tex_cell_float(cells[idx["Candidates"]])
                active = _tex_cell_float(cells[idx["Active"]])
                top5 = _tex_cell_float(cells[idx["Top 5 share"]])
                deployed = _tex_cell_float(cells[idx["Deployed gross"]])
                at_cap = _tex_cell_float(cells[idx["At cap share"]])
                if not (np.isfinite(top5) and 0.0 <= top5 <= 1.0):
                    violations.append({"config": config, "column": "Top 5 share", "value": top5})
                if not (np.isfinite(at_cap) and 0.0 <= at_cap <= 1.0):
                    violations.append({"config": config, "column": "At cap share", "value": at_cap})
                if not (np.isfinite(active) and np.isfinite(candidates) and active <= candidates + 1e-9):
                    violations.append({"config": config, "issue": "Active > Candidates", "active": active, "candidates": candidates})
                expected = float(expected_gross.loc[config]) if config in expected_gross.index else float("nan")
                if not (np.isfinite(deployed) and np.isfinite(expected) and abs(deployed - expected) < 2e-3):
                    violations.append({"config": config, "column": "Deployed gross", "table": deployed, "realized": expected})
            v.check(
                "E1 concentration table values match artifacts",
                "inference",
                not violations,
                observed=violations[:10],
                expected="deployed gross within 2e-3, shares in [0,1], Active <= Candidates",
            )
    else:
        v.check(
            "E1 concentration table exists",
            "inference",
            False,
            observed=concentration_path,
            expected="required output of final paper build",
        )


def check_cpcv_windows_table(v: Verifier) -> None:
    """Audit the compact full-vs-claim CPCV windows table."""

    table_path = TABLE_DIR / "short_cpcv_windows.tex"
    if not table_path.exists():
        v.check(
            "CPCV windows table exists",
            "robustness",
            False,
            observed=table_path.relative_to(PAPER),
            expected="tables/short_cpcv_windows.tex",
        )
        return

    inputs = [
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_cpcv_path_metrics.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_claim_cpcv_path_metrics.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_relative_paths.csv",
        BREADTH_ROBUSTNESS_DIR / "breadth_cv_claim_relative_paths.csv",
        BREADTH_ROBUSTNESS_DIR / "final_result_scoreboard.csv",
    ]
    missing = [str(path.relative_to(PAPER)) for path in inputs if not path.exists()]
    v.check(
        "CPCV windows table inputs exist",
        "robustness",
        not missing,
        observed=missing,
        expected="full CPCV metrics, claim CPCV metrics, claim relative paths, and final scoreboard",
    )
    if missing:
        return

    try:
        header, rows = _parse_tex_table(table_path)
        full_metrics = pd.read_csv(inputs[0])
        claim_metrics = pd.read_csv(inputs[1])
        full_relative = pd.read_csv(inputs[2])
        claim_relative = pd.read_csv(inputs[3])
        scoreboard = pd.read_csv(inputs[4])
    except Exception as exc:
        v.check("CPCV windows table artifacts parse", "robustness", False, observed=type(exc).__name__, details=str(exc))
        return

    idx = {_tex_cell_text(name): i for i, name in enumerate(header)}
    expected_cols = {
        "Config",
        "Liquid net p05",
        "Liquid net p50",
        "Default share",
        "Claim net p05",
        "Claim net p50",
        "Rel liquid p05",
        "Rel claim p05",
    }
    missing_cols = sorted(expected_cols.difference(idx))
    v.check(
        "CPCV windows table schema",
        "robustness",
        bool(header) and len(rows) == 4 and not missing_cols,
        observed={"header": header, "rows": len(rows), "missing_cols": missing_cols},
        expected="4 config rows with full/claim/relative CPCV columns",
    )
    if missing_cols or len(rows) != 4:
        return

    scoreboard_required = {"config", "best_naive_strategy"}
    artifact_missing = {
        "final_result_scoreboard": sorted(scoreboard_required.difference(scoreboard.columns)),
        "breadth_cv_cpcv_path_metrics": sorted({"strategy", "basis", "status", "sharpe", "defaulted"}.difference(full_metrics.columns)),
        "breadth_cv_claim_cpcv_path_metrics": sorted({"strategy", "basis", "status", "sharpe"}.difference(claim_metrics.columns)),
        "breadth_cv_relative_paths": sorted({"strategy", "basis", "status", "sharpe"}.difference(full_relative.columns)),
        "breadth_cv_claim_relative_paths": sorted({"strategy", "basis", "status", "sharpe"}.difference(claim_relative.columns)),
    }
    schema_ok = not any(artifact_missing.values())
    v.check("CPCV windows artifact schemas", "robustness", schema_ok, observed=artifact_missing, expected="required columns present")
    if not schema_ok:
        return

    score_by_config = scoreboard.drop_duplicates("config").set_index("config")
    violations: list[dict[str, Any]] = []
    default_share_violations: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    for cells in rows:
        if len(cells) <= max(idx.values()):
            violations.append({"row": cells, "issue": "short row"})
            continue
        config = _tex_cell_text(cells[idx["Config"]])
        seen_configs.add(config)
        full_strategy = f"{config} {BREADTH_PRIMARY_STRATEGY}"
        full_sub = _cpcv_metric_slice(full_metrics, full_strategy)
        claim_sub = _cpcv_metric_slice(claim_metrics, full_strategy)
        if full_sub.empty or claim_sub.empty:
            violations.append({"config": config, "issue": "missing full or claim CPCV path metrics"})
            continue
        if config not in score_by_config.index:
            violations.append({"config": config, "issue": "missing final_result_scoreboard row"})
            continue
        best_naive = str(score_by_config.loc[config, "best_naive_strategy"])
        rel_strategy = f"{config} E1 minus {best_naive}"
        rel_full_sub = _cpcv_metric_slice(full_relative, rel_strategy)
        rel_claim_sub = _cpcv_metric_slice(claim_relative, rel_strategy)
        if rel_full_sub.empty or rel_claim_sub.empty:
            violations.append({"config": config, "issue": "missing relative CPCV path metrics", "strategy": rel_strategy})
            continue

        expected = {
            "Liquid net p05": _quantile(full_sub["sharpe"], 0.05),
            "Liquid net p50": _quantile(full_sub["sharpe"], 0.50),
            "Default share": _default_share(full_sub["defaulted"]),
            "Claim net p05": _quantile(claim_sub["sharpe"], 0.05),
            "Claim net p50": _quantile(claim_sub["sharpe"], 0.50),
            "Rel liquid p05": _quantile(rel_full_sub["sharpe"], 0.05),
            "Rel claim p05": _quantile(rel_claim_sub["sharpe"], 0.05),
        }
        default_share = expected["Default share"]
        if not (np.isfinite(default_share) and 0.0 <= default_share <= 1.0):
            default_share_violations.append({"config": config, "default_share": default_share})
        for col, expected_value in expected.items():
            table_value = _tex_cell_float(cells[idx[col]])
            if not (np.isfinite(table_value) and np.isfinite(expected_value) and abs(table_value - expected_value) <= 2e-3):
                violations.append({"config": config, "column": col, "table": table_value, "recomputed": expected_value})

    missing_configs = sorted(set(BREADTH_CONFIG_ORDER).difference(seen_configs))
    if missing_configs:
        violations.append({"issue": "missing config rows", "configs": missing_configs})
    v.check(
        "CPCV windows table values match artifacts",
        "robustness",
        not violations,
        observed=violations[:12],
        expected="<2e-3 absolute difference vs recomputed artifact values",
    )
    v.check(
        "CPCV windows default shares bounded",
        "robustness",
        not default_share_violations,
        observed=default_share_violations,
        expected="Default share in [0,1]",
    )


def _cpcv_metric_slice(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return frame.loc[
        frame["strategy"].astype(str).eq(strategy)
        & frame["basis"].astype(str).eq("full_cost_net")
        & frame["status"].astype(str).eq("complete")
    ].copy()


def _quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.quantile(q)) if len(numeric) else float("nan")


def _default_share(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    if pd.api.types.is_bool_dtype(values):
        return float(values.fillna(False).astype(bool).mean())
    text = values.astype(str).str.strip().str.lower()
    return float(text.isin({"true", "1", "yes"}).mean())


def check_cv_purge_gap(v: Verifier) -> None:
    """Measure the realized calendar gap between pre-test training and test months."""

    _check_breadth_cv_purge_gap(
        v,
        artifact_prefix="breadth_cv",
        label="breadth CV",
        grid_source_path=BREADTH_ROBUSTNESS_DIR / "breadth_cv_full_month_grid.csv",
        result_check_name="breadth CV purge gap exceeds monthly tenor span",
    )


def check_claim_cv_purge_gap(v: Verifier) -> None:
    """Measure the claim-window CV train/test calendar gap."""

    _check_breadth_cv_purge_gap(
        v,
        artifact_prefix="breadth_cv_claim",
        label="breadth claim CV",
        grid_source_path=BREADTH_ROBUSTNESS_DIR / "breadth_cv_full_month_grid.csv",
        result_check_name="breadth claim CV purge gap exceeds monthly tenor span",
    )


def _check_breadth_cv_purge_gap(
    v: Verifier,
    *,
    artifact_prefix: str,
    label: str,
    grid_source_path: Path | None,
    result_check_name: str,
) -> None:
    schedule_path = BREADTH_ROBUSTNESS_DIR / f"{artifact_prefix}_fold_schedule.csv"
    ledger_path = BREADTH_ROBUSTNESS_DIR / f"{artifact_prefix}_fold_ledger.csv"
    returns_path = BREADTH_ROBUSTNESS_DIR / f"{artifact_prefix}_test_month_returns.csv"
    inputs = [schedule_path, ledger_path, returns_path]
    if grid_source_path is not None:
        inputs.append(grid_source_path)
    missing = [str(path.relative_to(PAPER)) for path in inputs if not path.exists()]
    v.check(
        f"{label} purge-gap inputs exist",
        "robustness",
        not missing,
        observed=missing,
        expected=f"{label} schedule, ledger, month-return grid, and full-history month grid when needed",
    )
    if missing:
        return

    try:
        schedule = pd.read_csv(schedule_path)
        ledger = pd.read_csv(ledger_path)
        month_returns = pd.read_csv(returns_path, usecols=["config", "return_date"])
        grid_returns = month_returns if grid_source_path is None else pd.read_csv(grid_source_path, usecols=["config", "return_date"])
    except Exception as exc:
        v.check(f"{label} purge-gap artifacts parse", "robustness", False, observed=type(exc).__name__, details=str(exc))
        return

    schedule_required = {"config", "fold_id", "scheme", "test_groups", "n_train", "n_purged", "n_embargoed"}
    ledger_required = {"config", "fold_id", "status"}
    month_required = {"config", "return_date"}
    schema_missing = {
        f"{artifact_prefix}_fold_schedule": sorted(schedule_required.difference(schedule.columns)),
        f"{artifact_prefix}_fold_ledger": sorted(ledger_required.difference(ledger.columns)),
        f"{artifact_prefix}_test_month_returns": sorted(month_required.difference(month_returns.columns)),
        "full_history_month_grid": sorted(month_required.difference(grid_returns.columns)),
    }
    schema_ok = not any(schema_missing.values())
    v.check(f"{label} purge-gap artifact schemas", "robustness", schema_ok, observed=schema_missing, expected="required columns present")
    if not schema_ok:
        return

    ledger_ok = ledger.loc[ledger["status"].astype(str).eq("ok"), ["config", "fold_id"]].drop_duplicates()
    realized_folds = set(map(tuple, ledger_ok.astype(str).to_numpy()))
    if not realized_folds:
        v.check(f"{label} ledger has realized folds", "robustness", False, observed=0)
        return

    month_returns = month_returns.copy()
    month_returns["return_date"] = pd.to_datetime(month_returns["return_date"], errors="coerce")
    grid_returns = grid_returns.copy()
    grid_returns["return_date"] = pd.to_datetime(grid_returns["return_date"], errors="coerce")
    min_gap_days: int | None = None
    min_case: dict[str, Any] = {}
    reconstruction_violations: list[dict[str, Any]] = []
    checked_blocks = 0

    for config in sorted(schedule["config"].dropna().astype(str).unique()):
        cfg_schedule = schedule.loc[schedule["config"].astype(str).eq(config)].copy()
        cfg_test_months = pd.DatetimeIndex(month_returns.loc[month_returns["config"].astype(str).eq(config), "return_date"].dropna().unique()).sort_values()
        cfg_months = pd.DatetimeIndex(grid_returns.loc[grid_returns["config"].astype(str).eq(config), "return_date"].dropna().unique()).sort_values()
        if len(cfg_months) == 0:
            reconstruction_violations.append({"config": config, "issue": "empty month grid"})
            continue
        if len(cfg_test_months) == 0:
            reconstruction_violations.append({"config": config, "issue": "empty test month grid"})
            continue
        group_ids_observed = set()
        for value in cfg_schedule["test_groups"]:
            group_ids_observed.update(_parse_group_ids(value))
        if not group_ids_observed:
            reconstruction_violations.append({"config": config, "issue": "no parsed test_groups"})
            continue
        n_groups = max(group_ids_observed) + 1
        group_ids = np.full(len(cfg_months), -1, dtype=int)
        full_pos_by_date = {pd.Timestamp(date): pos for pos, date in enumerate(cfg_months)}
        missing_test_dates = sorted(pd.Timestamp(date).strftime("%Y-%m-%d") for date in cfg_test_months if pd.Timestamp(date) not in full_pos_by_date)
        if missing_test_dates:
            reconstruction_violations.append({"config": config, "issue": "test month not in full grid", "dates": missing_test_dates[:5]})
            continue
        for group_id, positions in enumerate(np.array_split(np.arange(len(cfg_test_months)), n_groups)):
            for test_window_pos in positions:
                full_pos = full_pos_by_date[pd.Timestamp(cfg_test_months[int(test_window_pos)])]
                group_ids[full_pos] = group_id

        purge_months, embargo_months, infer_violations = _infer_cv_purge_embargo_months(cfg_schedule, cfg_months, group_ids)
        if infer_violations:
            reconstruction_violations.extend({"config": config, **item} for item in infer_violations[:5])
            continue

        for _, row in cfg_schedule.iterrows():
            fold_key = (str(row["config"]), str(row["fold_id"]))
            if fold_key not in realized_folds:
                continue
            groups = _parse_group_ids(row.get("test_groups", ""))
            train_pos, test_pos, _purged_pos, _embargoed_pos = _breadth_cv_positions(
                cfg_months,
                group_ids,
                groups,
                purge_months=purge_months,
                embargo_months=embargo_months,
            )
            for start, end in _contiguous_position_blocks(sorted(test_pos)):
                # Two-sided gap: combinatorial folds train on months both before
                # and after each test block, so both boundaries must clear the
                # tenor span (Lemma lem:purge_gap in the paper appendix).
                block_gaps: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
                prior_train = [pos for pos in train_pos if pos < start]
                if prior_train:
                    block_gaps.append(
                        ("pre_test", pd.Timestamp(cfg_months[max(prior_train)]), pd.Timestamp(cfg_months[start]))
                    )
                later_train = [pos for pos in train_pos if pos > end]
                if later_train:
                    block_gaps.append(
                        ("post_test", pd.Timestamp(cfg_months[end]), pd.Timestamp(cfg_months[min(later_train)]))
                    )
                if not block_gaps:
                    continue
                checked_blocks += 1
                for side, earlier_month, later_month in block_gaps:
                    gap_days = int((later_month - earlier_month).days)
                    if min_gap_days is None or gap_days < min_gap_days:
                        min_gap_days = gap_days
                        min_case = {
                            "config": config,
                            "fold_id": str(row["fold_id"]),
                            "scheme": str(row["scheme"]),
                            "side": side,
                            "train_month": (earlier_month if side == "pre_test" else later_month).strftime("%Y-%m-%d"),
                            "test_month": (later_month if side == "pre_test" else earlier_month).strftime("%Y-%m-%d"),
                            "gap_days": gap_days,
                            "purge_months": purge_months,
                            "embargo_months": embargo_months,
                        }

    passed = min_gap_days is not None and min_gap_days > 44 and not reconstruction_violations
    v.check(
        result_check_name,
        "robustness",
        passed,
        observed=min_case if min_case else {"min_gap_days": min_gap_days, "checked_blocks": checked_blocks},
        expected="minimum realized train/test calendar gap on both sides of every test block >44 days",
        details=(
            f"Minimum realized train/test calendar gap (both sides) after purge/embargo is "
            f"{min_gap_days} days across {checked_blocks} test blocks. "
            f"Reconstruction issues: {reconstruction_violations[:5]}"
        ),
    )


def _breadth_cv_positions(
    month_grid: pd.DatetimeIndex,
    group_ids: np.ndarray,
    test_groups: set[int],
    *,
    purge_months: int,
    embargo_months: int,
) -> tuple[set[int], set[int], set[int], set[int]]:
    all_pos = set(range(len(month_grid)))
    test_pos = {int(pos) for pos, group_id in enumerate(group_ids) if int(group_id) in test_groups}
    purged_pos: set[int] = set()
    embargoed_pos: set[int] = set()
    for start, end in _contiguous_position_blocks(sorted(test_pos)):
        purged_pos.update(range(max(0, start - purge_months), min(len(month_grid) - 1, end + purge_months) + 1))
        embargoed_pos.update(range(end + purge_months + 1, min(len(month_grid) - 1, end + purge_months + embargo_months) + 1))
    purged_pos.difference_update(test_pos)
    embargoed_pos.difference_update(test_pos)
    embargoed_pos.difference_update(purged_pos)
    train_pos = all_pos.difference(test_pos).difference(purged_pos).difference(embargoed_pos)
    return train_pos, test_pos, purged_pos, embargoed_pos


def _infer_cv_purge_embargo_months(
    schedule: pd.DataFrame,
    month_grid: pd.DatetimeIndex,
    group_ids: np.ndarray,
) -> tuple[int, int, list[dict[str, Any]]]:
    best: tuple[int, int, list[dict[str, Any]]] | None = None
    for purge_months in range(0, 7):
        for embargo_months in range(0, 7):
            violations: list[dict[str, Any]] = []
            for _, row in schedule.iterrows():
                groups = _parse_group_ids(row.get("test_groups", ""))
                train_pos, _test_pos, purged_pos, embargoed_pos = _breadth_cv_positions(
                    month_grid,
                    group_ids,
                    groups,
                    purge_months=purge_months,
                    embargo_months=embargo_months,
                )
                observed = {
                    "n_train": _maybe_int(row.get("n_train")),
                    "n_purged": _maybe_int(row.get("n_purged")),
                    "n_embargoed": _maybe_int(row.get("n_embargoed")),
                }
                expected = {
                    "n_train": len(train_pos),
                    "n_purged": len(purged_pos),
                    "n_embargoed": len(embargoed_pos),
                }
                bad = {key: {"observed": observed[key], "recomputed": expected[key]} for key in expected if observed[key] is not None and observed[key] != expected[key]}
                if bad:
                    violations.append({"fold_id": str(row.get("fold_id", "")), "purge": purge_months, "embargo": embargo_months, "mismatch": bad})
                    if len(violations) > 10:
                        break
            if not violations:
                return purge_months, embargo_months, []
            if best is None or len(violations) < len(best[2]):
                best = (purge_months, embargo_months, violations)
    return best if best is not None else (0, 0, [{"issue": "no candidate purge/embargo windows evaluated"}])


def check_independent_stat_recomputation(v: Verifier) -> None:
    """Raw-numpy re-implementation of Sharpe/ann. return/vol vs published table.

    Deliberately avoids the producer modules' performance functions: the
    statistics are recomputed from artifacts/strategy_returns*.csv with plain
    numpy (mean*12, std(ddof=1)*sqrt(12), ratio) and compared against
    tables/portfolio_performance.tex within rounding tolerance.
    """

    perf_path = TABLE_DIR / "portfolio_performance.tex"
    gross_path = ART_DIR / "strategy_returns.csv"
    net_path = ART_DIR / "strategy_returns_post_cost.csv"
    if not (perf_path.exists() and gross_path.exists() and net_path.exists()):
        v.check("independent stat recompute inputs available", "inference", False, observed=str(perf_path))
        return
    header, rows = _parse_tex_table(perf_path)
    if not header or "Strategy" not in header:
        v.check("portfolio performance table parses", "inference", False, observed=header)
        return
    idx = {name: i for i, name in enumerate(header)}
    frames = {
        "Gross before costs": pd.read_csv(gross_path, parse_dates=["snap_date"]).set_index("snap_date"),
        "Post-cost research": pd.read_csv(net_path, parse_dates=["snap_date"]).set_index("snap_date"),
    }
    tol = 2e-3  # published table is rounded to 3 decimals
    checked = 0
    for cells in rows:
        basis = _tex_cell_text(cells[idx["Return basis"]]) if "Return basis" in idx else "Gross before costs"
        strategy = _tex_cell_text(cells[idx["Strategy"]])
        frame = frames.get(basis)
        if frame is None or strategy not in frame.columns:
            v.check(f"independent recompute source column exists: {basis}/{strategy}", "inference", False, observed=strategy)
            continue
        x = frame[strategy].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        ann_ret = float(np.mean(x) * 12.0)
        ann_vol = float(np.std(x, ddof=1) * np.sqrt(12.0))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        for label, observed in [("Ann. return", ann_ret), ("Ann. vol", ann_vol), ("Sharpe", sharpe)]:
            published = _tex_cell_float(cells[idx[label]]) if label in idx else float("nan")
            ok = _close_or_both_nan(observed, published, tol=tol)
            v.check(
                f"independent numpy stat matches table: {basis} / {strategy} / {label}",
                "inference",
                ok,
                observed=observed,
                expected=published,
            )
            checked += 1
    v.check("independent stat recompute coverage", "inference", checked >= 24, observed=checked, expected=">=24")


def check_ci_pairs_ordered(v: Verifier) -> None:
    """Every emitted 'CI lo'/'CI hi' pair in published tables must satisfy lo <= hi."""

    tables_with_ci = 0
    for tex in sorted(TABLE_DIR.glob("*.tex")):
        header, rows = _parse_tex_table(tex)
        if not header:
            continue
        pairs = []
        for i, name in enumerate(header):
            clean = _tex_cell_text(name)
            if "CI lo" in clean:
                target = clean.replace("CI lo", "CI hi")
                for j, other in enumerate(header):
                    if _tex_cell_text(other) == target:
                        pairs.append((clean, i, j))
                        break
        if not pairs:
            continue
        tables_with_ci += 1
        violations = []
        for cells in rows:
            for label, i, j in pairs:
                if i >= len(cells) or j >= len(cells):
                    continue
                lo = _tex_cell_float(cells[i])
                hi = _tex_cell_float(cells[j])
                if np.isfinite(lo) and np.isfinite(hi) and lo > hi + 1e-9:
                    violations.append({"table": tex.name, "column": label, "lo": lo, "hi": hi})
        v.check(
            f"CI lo <= CI hi in {tex.name}",
            "inference",
            not violations,
            observed=violations[:5],
            expected="all CI pairs ordered",
        )
    v.check("CI ordering audit covers tables", "inference", tables_with_ci >= 4, observed=tables_with_ci, expected=">=4 tables with CI pairs")


def _independent_factor_regression(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [emp.SPY_UNDERLYING] + PRIMARY + ["VX_FRONT", "dVIX", "dVVIX"]
    x = factors.reindex(returns.index).reindex(columns=factor_cols)
    rows = []
    for strategy in returns.columns:
        aligned = pd.concat([returns[strategy].rename("portfolio"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        xcols = factor_cols
        if len(aligned) <= len(xcols) + 2:
            continue
        yv = aligned["portfolio"].to_numpy(float)
        xm = np.column_stack([np.ones(len(aligned)), aligned[xcols].to_numpy(float)])
        coef = np.linalg.pinv(xm) @ yv
        fitted = xm @ coef
        resid = yv - fitted
        denom = float(((yv - yv.mean()) ** 2).sum())
        row = {
            "Strategy": strategy,
            "Ann. alpha": float(coef[0] * PERIODS_PER_YEAR),
            "$R^2$": 1.0 - float((resid @ resid) / denom) if denom > 0 else np.nan,
            "Residual ann. vol": float(pd.Series(resid).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)),
        }
        for col, beta in zip(xcols, coef[1:]):
            label = "Beta SPY" if col == emp.SPY_UNDERLYING else f"Beta {col}"
            if col == "VX_FRONT":
                label = "Beta VX front"
            row[label] = float(beta)
        rows.append(row)
    return pd.DataFrame(rows)


def _close_or_both_nan(a: Any, b: Any, tol: float = TOL) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False
    if math.isnan(af) and math.isnan(bf):
        return True
    return abs(af - bf) <= tol


def check_claims_and_bibliography(v: Verifier) -> None:
    claim_path = ART_DIR / "claim_audit.csv"
    if claim_path.exists():
        claims = pd.read_csv(claim_path)
        claim_map = {row["Claim"]: row for _, row in claims.iterrows()}
        exact_key = "VIX option expiry P\\&L is exact listed settlement P\\&L"
        vix_claim = claim_map.get(exact_key)
        vix_ok = vix_claim is not None and (
            (vix_claim["Type"] == "Rejected overclaim" and vix_claim["Status"] == "Not claimed")
            or (vix_claim["Type"] == "Generated empirical" and vix_claim["Status"] == "Supported")
        )
        v.check("VIX exact settlement claim gated by source coverage", "claims", vix_ok, observed=vix_claim.to_dict() if vix_claim is not None else None)
        post_cost = claim_map.get("Post-cost research returns include implementation frictions")
        v.check("post-cost research claim is generated", "claims", post_cost is not None and "Implemented" in str(post_cost["Status"]), observed=post_cost.to_dict() if post_cost is not None else None)
        preprod = claim_map.get("Pre-production results are broker-executed live evidence")
        v.check("broker-executed evidence overclaim rejected", "claims", preprod is not None and preprod["Status"] == "Not claimed", observed=preprod.to_dict() if preprod is not None else None)
        prod = claim_map.get("Strategy is production tradable after costs")
        v.check("production tradability not claimed", "claims", prod is not None and prod["Status"] == "Not claimed", observed=prod.to_dict() if prod is not None else None)
        drift = claim_map.get("Result is not only long-call equity drift")
        v.check("alpha-independence overclaim downgraded", "claims", drift is not None and "Not supported" in str(drift["Status"]), observed=drift.to_dict() if drift is not None else None)
        v.check("claim audit has theorem and empirical claim types", "claims", {"Theorem", "Generated empirical", "Rejected overclaim"}.issubset(set(claims["Type"])), observed=claims["Type"].value_counts().to_dict())
    else:
        v.check("claim audit artifact exists", "claims", False, observed=claim_path)

    bib = PAPER / "references.bib"
    if bib.exists():
        text = bib.read_text(encoding="utf-8")
        entry_types = re.findall(r"@(\w+)\s*\{", text)
        disallowed = [t for t in entry_types if t.lower() not in {"article", "book"}]
        ops_terms = [term for term in ["cboe", "databento", "fred", "notebooklm", "class notes"] if term in text.lower()]
        v.check("bibliography entry types papers/books only", "claims", not disallowed, observed=disallowed, expected="article/book")
        v.check("operational sources absent from bibliography", "claims", not ops_terms, observed=ops_terms, expected="no Cboe/Databento/FRED/NotebookLM/class notes")
    else:
        v.check("bibliography exists", "claims", False, observed=bib)

    source = PAPER / "docs/source_ledger.md"
    if source.exists():
        ledger = source.read_text(encoding="utf-8").lower()
        for term in ["cboe", "databento", "vix option", "artifact"]:
            v.check(f"source ledger records {term}", "claims", term in ledger, observed=term)
        v.check(
            "source ledger keeps NotebookLM/class notes out of paper references",
            "claims",
            "notebooklm" not in ledger or "not cited as scholarly" in ledger or "not in bibliography" in ledger,
            observed="notebooklm" in ledger,
            severity=WARNING,
        )
    else:
        v.check("source ledger exists", "claims", False, observed=source)


def check_r1_repaired_artifacts(v: Verifier) -> None:
    required = {
        "returns": R1_DIR / "r1_monthly_development_returns.csv",
        "weights": R1_DIR / "r1_monthly_weights.csv",
        "summary": R1_DIR / "r1_survival_summary.csv",
        "registry": R1_DIR / "research_trial_registry.csv",
        "registry_meta": R1_DIR / "research_trial_registry.json",
        "freeze": R1_DIR / "prospective_freeze_manifest.json",
        "comparisons": R1_DIR / "r1_paired_bootstrap_comparisons.csv",
        "table": TABLE_DIR / "short_r1_survival_summary.tex",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    v.check("R1 repaired artifacts exist", "r1", not missing, observed=missing)
    if missing:
        return
    returns = pd.read_csv(required["returns"], parse_dates=["return_date", "decision_date", "train_end"])
    required_return_cols = {
        "config", "strategy", "return_date", "decision_date", "train_end", "gross_return",
        "predicted_cost", "net_return", "gross_nav", "scenario_cvar_loss",
        "short_margin_used", "collateral_used", "integer_repair_failed", "information_set_valid",
    }
    v.check("R1 return schema", "r1", required_return_cols.issubset(returns.columns), observed=sorted(set(required_return_cols) - set(returns.columns)))
    timing_ok = bool(
        len(returns)
        and (returns["train_end"] < returns["return_date"]).all()
        and (returns["decision_date"] < returns["return_date"]).all()
        and returns["information_set_valid"].astype(bool).all()
    )
    v.check("R1 information sets are point-in-time", "r1", timing_ok, observed=len(returns))
    v.check("R1 gross is an upper bound", "r1", bool((returns["gross_nav"] <= 1.0 + 1e-7).all()), observed=float(returns["gross_nav"].max()))
    v.check("R1 hard operational limits hold", "r1", bool(
        (returns["scenario_cvar_loss"] <= 0.10 + 1e-6).all()
        and (returns["short_margin_used"] <= 0.75 + 1e-6).all()
        and (returns["collateral_used"] <= 1.0 + 1e-6).all()
    ), observed={
        "max_cvar": float(returns["scenario_cvar_loss"].max()),
        "max_margin": float(returns["short_margin_used"].max()),
        "max_collateral": float(returns["collateral_used"].max()),
    })
    summary = pd.read_csv(required["summary"])
    v.check("R1 labels all existing evidence development", "r1", bool(summary["evidence_status"].astype(str).eq("retrospective_development_sample").all()), observed=summary.get("evidence_status", pd.Series(dtype=str)).tolist())
    hard_failure = (
        summary.get("ruin_count", 0).fillna(0).gt(0)
        | summary.get("margin_breaches", 0).fillna(0).gt(0)
        | summary.get("collateral_breaches", 0).fillna(0).gt(0)
        | summary.get("integer_failures", 0).fillna(0).gt(0)
        | summary.get("absorbed_validation_paths", 0).fillna(0).gt(0)
    )
    verdict_ok = summary.loc[hard_failure, "verdict"].astype(str).eq("fail_survival_gate").all()
    v.check("R1 hard survival gate controls verdict", "r1", bool(verdict_ok), observed=summary[["config", "verdict"]].to_dict(orient="records"))
    registry_meta = json.loads(required["registry_meta"].read_text(encoding="utf-8"))
    registry = pd.read_csv(required["registry"])
    v.check("R1 trial registry reports a lower bound", "r1", bool(
        not registry_meta.get("is_complete", True)
        and int(registry_meta.get("known_trial_count_lower_bound", -1)) == len(registry)
    ), observed=registry_meta)
    freeze = json.loads(required["freeze"].read_text(encoding="utf-8"))
    v.check("R1 prospective protocol requires 36 untouched months", "r1", bool(
        freeze.get("required_untouched_monthly_observations") == 36
        and freeze.get("confirmatory_claim_allowed") is False
        and freeze.get("evidence_before_freeze") == "retrospective_development_sample"
    ), observed=freeze)
    source_hashes = freeze.get("source_sha256", {})
    hash_matches = {
        rel: Path(ROOT / rel).exists() and hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
        for rel, expected in source_hashes.items()
    }
    v.check("R1 frozen source hashes remain unchanged", "r1", bool(hash_matches) and all(hash_matches.values()), observed=hash_matches)
    comparisons = pd.read_csv(required["comparisons"])
    expected_pairs = {(config, benchmark) for config in BREADTH_CONFIG_ORDER for benchmark in ["matched_capped_naive", "stock_markowitz"]}
    observed_pairs = set(zip(comparisons.get("config", []), comparisons.get("benchmark", [])))
    v.check("R1 paired growth/tail comparisons cover both baselines", "r1", expected_pairs.issubset(observed_pairs), observed=sorted(observed_pairs), expected=sorted(expected_pairs))


def check_r11_higher_risk_artifacts(v: Verifier) -> None:
    required = {
        "returns": R11_DIR / "r11_monthly_development_returns.csv",
        "weights": R11_DIR / "r11_monthly_weights.csv",
        "repairs": R11_DIR / "r11_integer_repair_candidates.csv",
        "repair_summary": R11_DIR / "r11_integer_repair_method_summary.csv",
        "summary": R11_DIR / "r11_survival_summary.csv",
        "events": R11_DIR / "r11_vix_risk_off_events.csv",
        "calendar": R11_DIR / "r11_vix_exposure_calendar.csv",
        "requests": R11_DIR / "r11_event_quote_request.csv",
        "execution": R11_DIR / "r11_intervention_execution_summary.csv",
        "gate": R11_DIR / "r11_egarch_gate.json",
        "registry": R11_DIR / "r11_research_trial_registry.csv",
        "registry_meta": R11_DIR / "r11_research_trial_registry.json",
        "freeze": R11_DIR / "r11_prospective_freeze_manifest.json",
        "status": R11_DIR / "r11_specification_status.csv",
        "table": TABLE_DIR / "short_r11_development_summary.tex",
        "repair_table": TABLE_DIR / "short_r11_integer_repair_summary.tex",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    v.check("R1.1 artifacts exist", "r11", not missing, observed=missing)
    if missing:
        return
    returns = pd.read_csv(required["returns"], parse_dates=["return_date", "decision_date", "train_end"])
    core = returns[~returns["strategy"].astype(str).str.contains("risk-off", regex=False)].copy()
    required_cols = {
        "config", "strategy", "return_date", "decision_date", "train_end", "gross_nav",
        "predicted_annual_vol", "continuous_predicted_annual_vol", "deployment_target", "deployment_target_feasible",
        "deployment_target_met_after_integer_repair", "march_2020_observation_policy",
        "scenario_cvar_loss", "short_margin_used", "collateral_used", "integer_repair_failed",
        "integer_execution_abstained", "integer_conversion_feasible", "integer_abstention_reason",
        "integer_repair_method", "pre_repair_feasible", "pre_repair_max_breach",
        "best_failed_method", "failed_max_breach", "failed_breach_cvar",
        "direct_truncation_max_breach", "direct_truncation_breach_cvar",
        "information_set_valid",
    }
    v.check("R1.1 return schema", "r11", required_cols.issubset(returns.columns), observed=sorted(required_cols - set(returns.columns)))
    v.check("R1.1 information sets are point-in-time", "r11", bool(
        len(core)
        and (core["train_end"] < core["return_date"]).all()
        and (core["decision_date"] < core["return_date"]).all()
        and core["information_set_valid"].astype(bool).all()
    ), observed=len(core))
    v.check("R1.1 uses the 25 percent risk cap", "r11", bool(
        core["deployment_target"].eq(0.50).all()
        and (core["predicted_annual_vol"] <= 0.25 + 1e-6).all()
        and (core["gross_nav"] <= 1.0 + 1e-7).all()
    ), observed={
        "max_vol": float(core["predicted_annual_vol"].max()),
        "max_gross": float(core["gross_nav"].max()),
        "target_hits": int(core["deployment_target_met_after_integer_repair"].sum()),
    })
    v.check("R1.1 keeps March 2020 observations", "r11", bool(
        core["march_2020_observation_policy"].astype(str).eq("retained_if_in_window").all()
        and core.loc[core["train_end"].ge(pd.Timestamp("2020-03-31")), "march_2020_in_training_window"].astype(bool).any()
    ), observed=core["march_2020_observation_policy"].astype(str).unique().tolist())
    v.check("R1.1 hard limits hold for returned books", "r11", bool(
        (core["scenario_cvar_loss"] <= 0.10 + 1e-6).all()
        and (core["short_margin_used"] <= 0.75 + 1e-6).all()
        and (core["collateral_used"] <= 1.0 + 1e-6).all()
    ), observed={
        "max_cvar": float(core["scenario_cvar_loss"].max()),
        "max_margin": float(core["short_margin_used"].max()),
        "max_collateral": float(core["collateral_used"].max()),
    })
    repairs = pd.read_csv(required["repairs"])
    expected_methods = {
        "truncate_toward_cash",
        "cash_abstention",
    }
    repair_groups = repairs.groupby(
        ["config", "strategy", "return_date", "decision_date"], observed=True
    )
    method_sets = repair_groups["method"].agg(lambda values: set(values.astype(str)))
    selected_counts = repair_groups["selected"].sum()
    selected_rows = repairs[repairs["selected"].astype(bool)]
    v.check("R1.1 records direct conversion and cash abstention", "r11", bool(
        len(method_sets) == len(core)
        and method_sets.map(lambda methods: methods == expected_methods).all()
        and selected_counts.eq(1).all()
        and selected_rows["feasible"].astype(bool).all()
        and (~core["integer_repair_failed"].astype(bool)).all()
        and core["integer_execution_abstained"].astype(bool).eq(
            ~core["integer_conversion_feasible"].astype(bool)
        ).all()
    ), observed={
        "repair_groups": len(method_sets),
        "core_rows": len(core),
        "methods": sorted(set(repairs["method"].astype(str))),
        "bad_selected_counts": int(selected_counts.ne(1).sum()),
    })
    repair_summary = pd.read_csv(required["repair_summary"])
    v.check("R1.1 integer method summary reconciles", "r11", bool(
        set(repair_summary["method"].astype(str)) == expected_methods
        and int(repair_summary["selected_periods"].sum()) == len(core)
        and int(repair_summary["candidate_periods"].sum()) == len(repairs)
    ), observed={
        "selected_periods": int(repair_summary["selected_periods"].sum()),
        "core_rows": len(core),
        "candidate_periods": int(repair_summary["candidate_periods"].sum()),
        "candidate_rows": len(repairs),
    })
    direct = repairs[repairs["method"].astype(str).eq("truncate_toward_cash")]
    cash = repairs[repairs["method"].astype(str).eq("cash_abstention")]
    v.check("R1.1 abstains exactly when direct conversion is infeasible", "r11", bool(
        len(direct) == len(core)
        and len(cash) == len(core)
        and int(cash["selected"].astype(bool).sum()) == int(core["integer_execution_abstained"].sum())
        and int(direct["selected"].astype(bool).sum()) == int(core["integer_conversion_feasible"].sum())
        and cash["feasible"].astype(bool).all()
    ), observed={
        "direct_feasible": int(direct["feasible"].astype(bool).sum()),
        "cash_abstentions": int(cash["selected"].astype(bool).sum()),
    })
    failed_truncation = core[~core["pre_repair_feasible"].astype(bool)]
    v.check("R1.1 preserves pre-repair failure diagnostics", "r11", bool(
        len(failed_truncation)
        and failed_truncation["pre_repair_max_breach"].gt(0).all()
        and failed_truncation["direct_truncation_max_breach"].gt(0).all()
        and failed_truncation["best_failed_method"].fillna("").astype(str).ne("").all()
        and failed_truncation["failed_max_breach"].gt(0).all()
    ), observed={
        "failed_direct_truncations": len(failed_truncation),
        "recorded_cvar_breaches": int(failed_truncation["failed_breach_cvar"].gt(0).sum()),
    })

    events = pd.read_csv(required["events"], parse_dates=["signal_date", "execution_date"])
    march_exit = events[(events["execution_date"].eq(pd.Timestamp("2020-03-02"))) & events["action"].eq("exit")]
    march_reentry = events[(events["execution_date"].eq(pd.Timestamp("2020-03-03"))) & events["action"].eq("reenter")]
    v.check("R1.1 March instruction is deduplicated and next-session", "r11", bool(
        len(march_exit) == 1
        and int(march_exit.iloc[0]["deduplicated_signal_count"]) == 2
        and "user_attested_manual_2020" in str(march_exit.iloc[0]["source"])
        and len(march_reentry) == 1
    ), observed=events[events["execution_date"].between("2020-03-02", "2020-03-03")].to_dict(orient="records"))
    calendar = pd.read_csv(required["calendar"], parse_dates=["session"]).set_index("session")
    v.check("R1.1 risk-off exposure is cash then re-enters", "r11", bool(
        calendar.loc[pd.Timestamp("2020-03-02"), "exposure_multiplier"] == 0.0
        and calendar.loc[pd.Timestamp("2020-03-03"), "exposure_multiplier"] == 1.0
    ), observed=calendar.loc["2020-03-02":"2020-03-03"].to_dict(orient="index"))
    requests = pd.read_csv(required["requests"])
    execution = pd.read_csv(required["execution"])
    risk_off = returns[returns["strategy"].astype(str).str.contains("risk-off", regex=False)]
    v.check("R1.1 missing licensed event quotes remain unscored", "r11", bool(
        len(requests)
        and len(execution)
        and (~execution["execution_feasible"].astype(bool)).all()
        and risk_off["evidence_status"].astype(str).eq("unscored_missing_or_incomplete_executable_quotes").all()
        and risk_off["net_return"].isna().any()
    ), observed={"requests": len(requests), "execution_rows": len(execution), "unscored_returns": int(risk_off["net_return"].isna().sum())})

    summary = pd.read_csv(required["summary"])
    scored_summary = summary[summary["evidence_status"].astype(str).eq("retrospective_development_sample")]
    hard_failure = (
        scored_summary.get("ruin_count", 0).fillna(0).gt(0)
        | scored_summary.get("margin_breaches", 0).fillna(0).gt(0)
        | scored_summary.get("collateral_breaches", 0).fillna(0).gt(0)
        | scored_summary.get("integer_failures", 0).fillna(0).gt(0)
    )
    v.check("R1.1 cash abstentions are not integer failures", "r11", bool(
        scored_summary["integer_failures"].fillna(0).eq(0).all()
        and scored_summary.loc[hard_failure, "verdict"].astype(str).eq("fail_survival_gate").all()
    ), observed=scored_summary[["config", "strategy", "integer_failures", "integer_abstentions", "verdict"]].to_dict(orient="records"))
    gate = json.loads(required["gate"].read_text(encoding="utf-8"))
    v.check("R1.1 EGARCH obeys its promotion gate", "r11", bool(
        gate.get("passed") is False and gate.get("promotion_status") == "diagnostic_only"
    ), observed=gate)
    registry = pd.read_csv(required["registry"])
    registry_meta = json.loads(required["registry_meta"].read_text(encoding="utf-8"))
    v.check("R1.1 trial registry includes all new arms", "r11", bool(
        int(registry_meta.get("known_trial_count_lower_bound", -1)) == len(registry)
        and registry["trial_key"].astype(str).str.startswith("R1.1_").sum() == 9
    ), observed=registry_meta)
    freeze = json.loads(required["freeze"].read_text(encoding="utf-8"))
    v.check("R1.1 has a separate 36-month prospective freeze", "r11", bool(
        freeze.get("specification") == "R1.1"
        and freeze.get("required_untouched_monthly_observations") == 36
        and freeze.get("confirmatory_claim_allowed") is False
        and freeze.get("march_2020_market_data_deleted") is False
        and freeze.get("integer_execution_policy", {}).get("infeasible_action") == "cash_abstention_for_the_period"
        and freeze.get("integer_execution_policy", {}).get("substitute_portfolios_allowed") is False
    ), observed=freeze)
    status = pd.read_csv(required["status"])
    v.check("R1.1 status preserves legacy E1 absorbed-zero failure", "r11", bool(
        status["status"].astype(str).eq("fail_survival_gate_absorbed_zero").any()
    ), observed=status[status["specification"].astype(str).str.contains("E1", regex=False)].to_dict(orient="records"))


def check_execution_audit_artifacts(v: Verifier) -> None:
    required = {
        "fill_ledger": EXECUTION_AUDIT_DIR / "execution_fill_ledger.csv",
        "monthly_returns": EXECUTION_AUDIT_DIR / "execution_audit_monthly_returns.csv",
        "summary": EXECUTION_AUDIT_DIR / "execution_audit_summary.json",
        "liquidity": EXECUTION_AUDIT_DIR / "liquidity_gate_validation.csv",
        "intervention": EXECUTION_AUDIT_DIR / "intervention_day_fill_evidence.csv",
        "mark_accuracy": EXECUTION_AUDIT_DIR / "mark_accuracy.csv",
        "readme": EXECUTION_AUDIT_DIR / "README.md",
        "summary_table": TABLE_DIR / "short_execution_audit_summary.tex",
        "spread_table": TABLE_DIR / "short_execution_spread_comparison.tex",
        "liquidity_table": TABLE_DIR / "short_liquidity_validation.tex",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    v.check("execution audit artifacts exist", "execution_audit", not missing, observed=missing)
    if missing:
        return

    monthly = pd.read_csv(required["monthly_returns"], dtype={"gross_return": "string"})
    fills = pd.read_csv(required["fill_ledger"])
    liquidity = pd.read_csv(required["liquidity"])
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    intervention = pd.read_csv(required["intervention"])
    monthly_columns = {
        "arm", "config", "return_date", "gross_return", "predicted_cost",
        "observed_cost_mid", "observed_cost_touch", "observed_cost_worst",
        "net_return_mid", "net_return_touch", "net_return_worst",
        "coverage_weight_fraction", "uncovered_cost_source",
    }
    source_schema_ok = bool(
        monthly_columns.issubset(monthly.columns)
        and "contracts_source" in fills.columns
        and "contracts_source" in liquidity.columns
    )
    v.check(
        "execution audit monthly and source schema",
        "execution_audit",
        source_schema_ok,
        observed={
            "missing_monthly": sorted(monthly_columns - set(monthly.columns)),
            "fill_contracts_source": "contracts_source" in fills.columns,
            "liquidity_contracts_source": "contracts_source" in liquidity.columns,
        },
    )

    ordered = (
        pd.to_numeric(monthly["net_return_worst"], errors="coerce")
        <= pd.to_numeric(monthly["net_return_touch"], errors="coerce") + 1e-12
    ) & (
        pd.to_numeric(monthly["net_return_touch"], errors="coerce")
        <= pd.to_numeric(monthly["net_return_mid"], errors="coerce") + 1e-12
    )
    v.check(
        "execution audit scenario returns are ordered",
        "execution_audit",
        bool(len(monthly) and ordered.all()),
        observed={"ordered_rows": int(ordered.sum()), "rows": len(monthly)},
        expected="net_return_worst <= net_return_touch <= net_return_mid",
    )

    from research.papers.option_only_markowitz.analysis import (  # noqa: E402
        r1_r11_execution_audit as execution_audit,
    )

    identity: dict[str, dict[str, Any]] = {}
    identity_ok = True
    for arm, source_path, strategy, expected_rows in [
        ("R1.1", R11_DIR / "r11_monthly_development_returns.csv", execution_audit.R11_NAME, 372),
        ("R1", R1_DIR / "r1_monthly_development_returns.csv", execution_audit.R1_NAME, 240),
    ]:
        source = pd.read_csv(source_path, dtype={"gross_return": "string"})
        source = source[source["strategy"].eq(strategy)][["config", "return_date", "gross_return"]]
        observed = monthly[monthly["arm"].eq(arm)][["config", "return_date", "gross_return"]]
        joined = observed.merge(
            source,
            on=["config", "return_date"],
            how="outer",
            suffixes=("_audit", "_frozen"),
            indicator=True,
            validate="one_to_one",
        )
        exact = joined["gross_return_audit"].astype("string").eq(
            joined["gross_return_frozen"].astype("string")
        )
        arm_ok = bool(
            len(observed) == expected_rows
            and len(source) == expected_rows
            and joined["_merge"].eq("both").all()
            and exact.all()
        )
        identity_ok = identity_ok and arm_ok
        identity[arm] = {
            "audit_rows": len(observed),
            "frozen_rows": len(source),
            "exact_rows": int(exact.sum()),
        }
    v.check(
        "execution audit gross-return text matches frozen arms",
        "execution_audit",
        identity_ok,
        observed=identity,
        expected={"R1.1": 372, "R1": 240},
    )

    reconstruction = summary.get("cost_reconstruction", {})
    reconstruction_ok = bool(
        reconstruction.get("status") == "exact"
        and float(reconstruction.get("max_absolute_gap", np.inf)) < 1e-6
    )
    v.check(
        "execution audit cost reconstruction is exact",
        "execution_audit",
        reconstruction_ok,
        observed=reconstruction,
        expected="status exact and max_absolute_gap < 1e-6",
    )

    coverage = pd.DataFrame(summary.get("coverage", []))
    r11_coverage = coverage[coverage.get("arm", pd.Series(dtype=str)).eq("R1.1")]
    entry = pd.to_numeric(r11_coverage.get("entry_coverage"), errors="coerce")
    roundtrip = pd.to_numeric(r11_coverage.get("roundtrip_coverage"), errors="coerce")
    coverage_ok = bool(
        len(r11_coverage) == 4
        and entry.ge(0.90).all()
        and roundtrip.gt(0.0).all()
        and roundtrip.le(1.0).all()
        and summary.get("raw_licensed_data_committed") is False
    )
    v.check(
        "execution audit coverage and licensing flags are sane",
        "execution_audit",
        coverage_ok,
        observed=r11_coverage.to_dict(orient="records"),
        expected="R1.1 entry >= 0.90, round-trip in (0,1], raw licensed data absent",
    )

    header, rows = _parse_tex_table(required["summary_table"])
    header_index = {_tex_cell_text(value): index for index, value in enumerate(header)}
    table_records = {
        _tex_cell_text(row[header_index["Config"]]): row
        for row in rows
        if "Config" in header_index and len(row) > header_index["Config"]
    }
    headline = {
        (str(record.get("config")), str(record.get("strategy"))): record
        for record in summary.get("headline_r11", [])
    }
    comparisons = [
        ("modeled", "modeled ann.", "annualized_return"),
        ("modeled", "modeled Sortino", "sortino"),
        ("modeled", "modeled maxDD", "max_drawdown"),
        ("mid", "mid ann.", "annualized_return"),
        ("mid", "mid Sortino", "sortino"),
        ("mid", "mid maxDD", "max_drawdown"),
        ("touch", "touch ann.", "annualized_return"),
        ("touch", "touch Sortino", "sortino"),
        ("touch", "touch maxDD", "max_drawdown"),
        ("worst", "worst ann.", "annualized_return"),
        ("worst", "worst Sortino", "sortino"),
        ("worst", "worst maxDD", "max_drawdown"),
    ]
    table_violations = []
    expected_configs = {"orig", "orig+VIX", "larger", "larger+VIX"}
    if set(table_records) != expected_configs or len(rows) != len(expected_configs):
        table_violations.append(
            {"table_configs": sorted(table_records), "expected_configs": sorted(expected_configs), "rows": len(rows)}
        )
    coverage_by_config = {
        str(record.get("config")): record
        for record in summary.get("coverage", [])
        if record.get("arm") == "R1.1"
    }
    for config in sorted(expected_configs):
        row = table_records.get(config)
        for strategy, column, field in comparisons:
            published = _tex_cell_float(row[header_index[column]]) if row is not None and column in header_index else np.nan
            record = headline.get((config, strategy), {})
            source_value = float(record.get(field, np.nan))
            if not np.isfinite(published) or not np.isfinite(source_value) or abs(published - source_value) > 5e-4:
                table_violations.append(
                    {"config": config, "column": column, "table": published, "summary": source_value}
                )
        for column, field in [("Entry cov. %", "entry_coverage"), ("Round-trip cov. %", "roundtrip_coverage")]:
            published = _tex_cell_float(row[header_index[column]]) if row is not None and column in header_index else np.nan
            source_value = 100.0 * float(coverage_by_config.get(config, {}).get(field, np.nan))
            if not np.isfinite(published) or not np.isfinite(source_value) or abs(published - source_value) > 5e-4:
                table_violations.append(
                    {"config": config, "column": column, "table": published, "summary": source_value}
                )
    v.check(
        "execution audit summary table matches JSON",
        "execution_audit",
        not table_violations,
        observed=table_violations,
        expected="all headline and coverage table values match JSON within display precision",
    )

    cache_root = ROOT / "data/databento_cache/r1_r11_audit"
    ledger_path = cache_root / "request_ledger.json"
    if ledger_path.exists():
        sample = (
            fills[fills["obs_coverage"].eq("covered")]
            .drop_duplicates(["symbol", "decision_date"])
            .head(3)
            .copy()
        )
        trade_columns = fills.columns[: fills.columns.get_loc("obs_bid")].tolist()
        trade_sample = sample[trade_columns].copy()
        trade_sample["_verification_row"] = np.arange(len(trade_sample))
        quotes = execution_audit._load_audit_frames(
            cache_root,
            {"candidate_close_quotes", "gap_candidate_close_quotes"},
            set(trade_sample["symbol"].astype(str)),
        )
        recomputed = execution_audit.match_quotes(trade_sample, quotes).sort_values(
            "_verification_row"
        )
        sample = sample.reset_index(drop=True)
        bid_match = np.allclose(
            pd.to_numeric(sample["obs_bid"], errors="coerce"),
            pd.to_numeric(recomputed["obs_bid"], errors="coerce"),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        ask_match = np.allclose(
            pd.to_numeric(sample["obs_ask"], errors="coerce"),
            pd.to_numeric(recomputed["obs_ask"], errors="coerce"),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        v.check(
            "execution audit sampled quotes recompute from licensed cache",
            "execution_audit",
            bool(len(sample) >= 3 and len(recomputed) == len(sample) and bid_match and ask_match),
            observed={"sample_rows": len(sample), "bid_match": bid_match, "ask_match": ask_match},
            expected=">=3 covered rows match the last valid in-window quote",
        )
    else:
        v.check(
            "execution audit sampled quotes recompute from licensed cache",
            "execution_audit",
            True,
            observed="licensed cache absent",
            details="Committed aggregate execution artifacts remain the source of record.",
            severity=WARNING,
        )

    frozen_intervention = R11_DIR / "r11_intervention_fill_ledger.csv"
    frozen_text = frozen_intervention.read_text(encoding="utf-8") if frozen_intervention.exists() else "missing"
    frozen_header_only = frozen_intervention.exists() and not frozen_text.strip()
    if frozen_intervention.exists() and frozen_text.strip():
        try:
            frozen_header_only = pd.read_csv(frozen_intervention).empty
        except pd.errors.EmptyDataError:
            frozen_header_only = True
    evidence_only = intervention.get("evidence_only", pd.Series(dtype=bool)).astype(str).str.lower().eq("true")
    v.check(
        "execution audit preserves unscored intervention arm",
        "execution_audit",
        bool(frozen_header_only and len(intervention) and evidence_only.all()),
        observed={
            "frozen_intervention_header_only": frozen_header_only,
            "evidence_rows": len(intervention),
            "evidence_only_rows": int(evidence_only.sum()),
        },
        expected="frozen fill ledger header-only and all new intervention rows evidence-only",
    )


def check_r2_robust_sortino_artifacts(v: Verifier) -> None:
    required = {
        "returns": R2_DIR / "r2_monthly_development_returns.csv",
        "weights": R2_DIR / "r2_monthly_weights.csv",
        "moments": R2_DIR / "r2_moment_ledger.csv",
        "scenarios": R2_DIR / "r2_scenario_diagnostics.csv",
        "abstentions": R2_DIR / "r2_abstention_ledger.csv",
        "summary": R2_DIR / "r2_survival_summary.csv",
        "aligned": R2_DIR / "r2_r11_aligned_returns.csv",
        "comparison": R2_DIR / "r2_r11_comparison_summary.csv",
        "block": R2_DIR / "r2_block_bootstrap_paths.csv",
        "paired": R2_DIR / "r2_paired_stationary_bootstrap.csv",
        "repriced": R2_DIR / "r2_repriced_state_paths.csv",
        "refit": R2_DIR / "r2_refit_monte_carlo.csv",
        "gate": R2_DIR / "r2_promotion_gate.json",
        "registry": R2_DIR / "r2_research_trial_registry.csv",
        "freeze": R2_DIR / "r2_prospective_freeze_manifest.json",
        "table": TABLE_DIR / "short_r2_development_summary.tex",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    v.check("R2 artifacts exist", "r2", not missing, observed=missing)
    if missing:
        return
    returns = pd.read_csv(required["returns"], parse_dates=["return_date", "decision_date", "train_end"])
    required_columns = {
        "config", "strategy", "evidence_status", "return_date", "decision_date", "train_end",
        "train_observations", "recent_observations", "recent_covariance_weight", "net_return",
        "predicted_cost", "gross_nav", "predicted_annual_vol", "worst_annual_downside",
        "scenario_cvar_loss", "worst_three_month_loss", "worst_six_month_loss",
        "short_margin_used", "collateral_used", "integer_execution_abstained",
        "integer_conversion_feasible", "selected_feasible", "selected_max_breach",
        "rejected_max_breach", "information_set_valid",
    }
    v.check("R2 return schema", "r2", required_columns.issubset(returns.columns), observed=sorted(required_columns - set(returns.columns)))
    v.check("R2 replay is cutoff-safe and spans four universes", "r2", bool(
        len(returns)
        and returns["config"].nunique() == 4
        and returns["evidence_status"].eq("retrospective_development_sample").all()
        and (returns["train_end"] < returns["return_date"]).all()
        and (returns["decision_date"] < returns["return_date"]).all()
        and returns["information_set_valid"].astype(bool).all()
        and pd.Timestamp(returns["return_date"].min()) >= pd.Timestamp("2018-02-01")
    ), observed={"rows": len(returns), "universes": sorted(returns["config"].unique()), "start": str(returns["return_date"].min()), "end": str(returns["return_date"].max())})
    v.check("R2 selected books satisfy every scalar-stage hard limit", "r2", bool(
        (returns["predicted_annual_vol"] <= 0.25 + 1e-6).all()
        and (returns["worst_annual_downside"] <= 0.10 + 1e-6).all()
        and (returns["scenario_cvar_loss"] <= 0.10 + 1e-6).all()
        and (returns["worst_three_month_loss"] <= 0.15 + 1e-6).all()
        and (returns["worst_six_month_loss"] <= 0.20 + 1e-6).all()
        and (returns["short_margin_used"] <= 0.75 + 1e-6).all()
        and (returns["collateral_used"] <= 1.0 + 1e-6).all()
        and (returns["gross_nav"] <= 1.0 + 1e-7).all()
    ), observed={column: float(returns[column].max()) for column in ["predicted_annual_vol", "worst_annual_downside", "scenario_cvar_loss", "worst_three_month_loss", "worst_six_month_loss", "short_margin_used", "collateral_used", "gross_nav"]})
    allowed_weights = {0.25, 0.50, 0.75}
    v.check("R2 covariance weights obey the registered QLIKE grid", "r2", bool(
        set(np.round(returns["recent_covariance_weight"].astype(float), 2)).issubset(allowed_weights)
        and returns.loc[returns["train_observations"] < 48, "recent_covariance_weight"].eq(0.50).all()
    ), observed=returns["recent_covariance_weight"].value_counts().to_dict())
    scenarios = pd.read_csv(required["scenarios"])
    expected_families = {"recent", "expanding_bootstrap", "imputation_1", "imputation_2", "imputation_3", "imputation_4", "imputation_5"}
    family_sets = scenarios.groupby(["config", "return_date"], observed=True)["family"].agg(set)
    v.check("R2 evaluates all seven robust downside families", "r2", bool(
        len(family_sets) == len(returns) and family_sets.map(lambda value: value == expected_families).all()
    ), observed={"decision_families": len(family_sets), "return_rows": len(returns)})
    abstentions = pd.read_csv(required["abstentions"])
    abstained = returns[returns["integer_execution_abstained"].astype(bool)]
    v.check("R2 direct-or-abstain preserves rejected diagnostics", "r2", bool(
        len(abstentions) == len(abstained)
        and returns["integer_execution_abstained"].astype(bool).eq(~returns["integer_conversion_feasible"].astype(bool)).all()
        and (abstained["gross_nav"] <= 1e-12).all()
        and (abstained["rejected_max_breach"] > 0).all()
    ), observed={"abstentions": len(abstentions), "return_abstentions": len(abstained)})
    aligned = pd.read_csv(required["aligned"])
    v.check("R2/R1.1 comparison is date aligned", "r2", bool(
        len(aligned) and aligned[["r2_net_return", "r11_net_return"]].notna().all().all()
        and aligned.groupby("config", observed=True)["return_date"].nunique().min() > 0
    ), observed=aligned.groupby("config", observed=True).size().to_dict())
    block = pd.read_csv(required["block"])
    repriced = pd.read_csv(required["repriced"])
    refit = pd.read_csv(required["refit"])
    v.check("R2 locked Monte Carlo path counts are complete", "r2", bool(
        block.groupby(["config", "strategy"], observed=True)["path_id"].nunique().eq(5000).all()
        and repriced.groupby(["config", "strategy", "method"], observed=True)["path_id"].nunique().eq(2000).all()
        and refit.groupby("config", observed=True)["path_id"].nunique().eq(200).all()
    ), observed={"block_rows": len(block), "repriced_rows": len(repriced), "refit_rows": len(refit)})
    gate = json.loads(required["gate"].read_text(encoding="utf-8"))
    v.check("R2 promotion status follows every registered gate", "r2", bool(
        gate.get("promoted") == all(gate.get("gates", {}).values())
        and gate.get("active_development_extension") == ("R2" if gate.get("promoted") else "R1.1")
    ), observed=gate)
    freeze = json.loads(required["freeze"].read_text(encoding="utf-8"))
    hash_matches = {
        rel: (ROOT / rel).exists() and hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected
        for rel, expected in freeze.get("source_sha256", {}).items()
    }
    v.check("R2 has a separate 36-month freeze with matching source hashes", "r2", bool(
        freeze.get("confirmatory_observations_required") == 36
        and freeze.get("evidence_status") == "retrospective_development_sample"
        and bool(hash_matches) and all(hash_matches.values())
    ), observed={"freeze": freeze, "hash_matches": hash_matches})


def check_paper_quality(v: Verifier, skip_render: bool, skip_compile: bool = False) -> None:
    log = PAPER / f"{PUBLISHED_STEM}.log"
    if log.exists():
        text = log.read_text(errors="ignore")
        pattern = re.compile(r"Undefined|undefined|Warning|Error|Overfull|Underfull|Extra alignment|Missing \$|Citation")
        matches = pattern.findall(text)
        v.check("LaTeX log clean", "paper", not matches, observed=matches[:10], expected="no warnings/errors/undefined refs")
    else:
        v.check(
            "LaTeX log exists",
            "paper",
            bool(skip_compile),
            observed=log,
            details="LaTeX compilation was skipped; existing PDF checks remain active." if skip_compile else "",
            severity=WARNING if skip_compile else CRITICAL,
        )

    pdf = PUBLISHED_PDF
    v.check("compiled PDF exists", "paper", pdf.exists() and pdf.stat().st_size > 100_000, observed=pdf.stat().st_size if pdf.exists() else 0, expected=">100KB")
    if pdf.exists():
        info = _run(["pdfinfo", PUBLISHED_PDF_NAME], PAPER, timeout=60) if shutil.which("pdfinfo") else None
        if info is not None:
            page_match = re.search(r"Pages:\s+(\d+)", info.stdout)
            pages = int(page_match.group(1)) if page_match else 0
            v.check("PDF page count plausible", "paper", 25 <= pages <= 45, observed=pages, expected="25-45 research-paper pages")

    pdf_text = extract_pdf_text(pdf)
    if pdf_text:
        lower = pdf_text.lower()
        try:
            summary = json.loads((TABLE_DIR / "empirical_summary.json").read_text())
            vix_headline = bool(summary.get("data", {}).get("vix_headline_eligible"))
        except Exception:
            vix_headline = False
        for phrase in [
            "not claimed",
            "premium weights",
        ]:
            v.check(f"PDF caveat text includes {phrase}", "paper", phrase in lower, observed=phrase)
        if vix_headline:
            v.check("PDF text includes exact VRO/SOQ", "paper", "vro/soq" in lower or "vro\\_soq" in lower, observed="VRO/SOQ")
        else:
            for phrase in ["settlement proxy", "vix-close settlement proxy"]:
                v.check(f"PDF caveat text includes {phrase}", "paper", phrase in lower, observed=phrase)
        v.check("PDF caveat text includes transaction costs", "paper", "transaction costs" in lower, observed="transaction costs")
        v.check("PDF caveat text includes slippage", "paper", "slippage" in lower, observed="slippage")
        v.check("PDF text includes tail-path simulation caveat", "paper", "path-risk diagnostics only" in lower or "tail-path simulation diagnostics" in lower, observed="tail-path simulation diagnostics")
        v.check("references include option-risk-premium papers", "paper", all(x in lower for x in ["expected option returns", "variance risk premia", "commodity contracts"]), observed="reference text")
    else:
        v.check("PDF text extract available", "paper", False, observed="pdftotext/pypdf unavailable or empty")

    if skip_render:
        v.check("PDF render skipped", "paper", True, details="Fast audit mode explicitly skipped rendering.", severity=WARNING)
        return
    if not shutil.which("pdftoppm"):
        v.check("PDF renderer available", "paper", False, observed="pdftoppm missing")
        return
    with tempfile.TemporaryDirectory(prefix="oom_verify_pdf_") as tmp:
        prefix = Path(tmp) / "page"
        res = _run(["pdftoppm", "-png", "-f", "1", "-l", "36", "-r", "90", PUBLISHED_PDF_NAME, str(prefix)], PAPER, timeout=120)
        rendered = list(Path(tmp).glob("page-*.png"))
        v.check("PDF pages render to PNG", "paper", res.returncode == 0 and len(rendered) >= 25, observed=f"exit={res.returncode}, pages={len(rendered)}", expected=">=25 rendered pages")
        nonempty = all(p.stat().st_size > 10_000 for p in rendered[:5] + rendered[-3:]) if rendered else False
        v.check("rendered PDF sample pages nonempty", "paper", nonempty, observed=[p.stat().st_size for p in rendered[:2]])

    fig_vis = ART_DIR / "figure_visibility_audit.csv"
    returns = ART_DIR / "strategy_returns.csv"
    if fig_vis.exists() and returns.exists():
        vis = pd.read_csv(fig_vis)
        ret_cols = pd.read_csv(returns, nrows=1).columns.tolist()[1:]
        v.check("Appendix all-strategy growth figure has all strategy series", "paper", set(vis["Series"]) == set(ret_cols), observed=sorted(vis["Series"]), expected=sorted(ret_cols))
        v.check("Appendix all-strategy growth figure series visible", "paper", vis["Pass"].astype(str).str.lower().eq("yes").all(), observed=vis.to_dict(orient="records"))


def extract_pdf_text(pdf: Path) -> str:
    if not pdf.exists():
        return ""
    if shutil.which("pdftotext"):
        res = _run(["pdftotext", str(pdf), "-"], PAPER, timeout=60)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def build_hash_manifest(v: Verifier) -> pd.DataFrame:
    rows = []
    roots = [TABLE_DIR, FIG_DIR, ART_DIR, R1_DIR, R11_DIR]
    files = [PUBLISHED_PDF]
    for root in roots:
        if root.exists():
            files.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    for path in sorted(set(files)):
        try:
            rel = path.relative_to(PAPER)
            rows.append(
                {
                    "path": str(rel),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        except Exception as exc:
            v.check(f"hash manifest reads {path.name}", "artifacts", False, observed=path, details=str(exc))
    manifest = pd.DataFrame(rows)
    v.check("hash manifest covers outputs", "artifacts", len(manifest) >= 30, observed=len(manifest), expected=">=30")
    return manifest


def run_verification(
    skip_regenerate: bool = False,
    skip_compile: bool = False,
    skip_render: bool = False,
    with_robustness: bool = False,
    skip_robustness: bool = False,
) -> Verifier:
    v = Verifier()
    regenerate_artifacts(v, skip_regenerate)
    regenerate_robustness_artifacts(v, skip=(skip_robustness or not with_robustness))
    compile_paper(v, skip_compile)
    check_required_outputs(v)
    summary = load_summary(v)
    check_inputs_and_data(v, summary)
    check_pit_ledgers(v)
    check_math_and_optimizer(v)
    check_empirical_reproduction(v, summary)
    check_pipeline_extension_artifacts(v)
    check_distributional_robustness(v)
    check_cv_purge_gap(v)
    check_cpcv_windows_table(v)
    check_claim_cv_purge_gap(v)
    check_independent_stat_recomputation(v)
    check_final_inference_panel(v)
    check_e1_ablation_and_concentration(v)
    check_ci_pairs_ordered(v)
    check_r1_repaired_artifacts(v)
    check_r11_higher_risk_artifacts(v)
    check_execution_audit_artifacts(v)
    check_r2_robust_sortino_artifacts(v)
    check_claims_and_bibliography(v)
    check_paper_quality(v, skip_render, skip_compile=skip_compile)
    manifest = build_hash_manifest(v)
    return v, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the option-only Markowitz paper end to end.")
    parser.add_argument("--skip-regenerate", action="store_true", help="Do not rerun the empirical artifact producer.")
    parser.add_argument("--skip-compile", action="store_true", help="Do not rerun pdflatex/bibtex compilation.")
    parser.add_argument("--skip-render", action="store_true", help="Do not render PDF pages to PNG.")
    parser.add_argument("--with-robustness", action="store_true", help="Regenerate the long distributional-robustness stage before checking artifacts.")
    parser.add_argument("--skip-robustness", action="store_true", help="Explicitly skip robustness regeneration (the default).")
    args = parser.parse_args(argv)
    if args.with_robustness and args.skip_robustness:
        parser.error("--with-robustness and --skip-robustness are mutually exclusive")
    v, manifest = run_verification(
        skip_regenerate=args.skip_regenerate,
        skip_compile=args.skip_compile,
        skip_render=args.skip_render,
        with_robustness=args.with_robustness,
        skip_robustness=args.skip_robustness,
    )
    v.write_outputs(" ".join([Path(sys.executable).name, "-m", "research.papers.option_only_markowitz.verification.verify"] + (argv or sys.argv[1:])), manifest)
    return 1 if v.fail_count(critical_only=True) else 0


if __name__ == "__main__":
    raise SystemExit(main())
