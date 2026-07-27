"""Build and verify the small, public working-paper release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import DISPLAY_NAMES, build_derived_evidence, summarize_returns

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.csv"
BUILD_ROOT = ROOT / "build"
BASELINE_RETURN_HASHES = {
    "paper/evidence/r1_monthly_returns.csv": (
        "61b29ffc5faf9ca2e33f32671a8c928547b17c71dd1d8d3c1839180ba04edec9"
    ),
    "paper/evidence/r11_monthly_returns.csv": (
        "e0a292a187c848cb01ecda637d68fdb1af3ef071dce56326f9f5f345b6d6a662"
    ),
}
BASELINE_WEIGHT_HASHES = {
    "r1_weights_sha256": (
        "ce8be5caa03ca3d6b105c829693886ecfac21ef58083069a525498847558590d"
    ),
    "r11_weights_sha256": (
        "bf59c5766b947b16dffb066d4f29f96e524da0dc41342b006b53f745f21233b5"
    ),
}
PRIVATE_INPUTS = (
    Path("data/feature_store/option_greek_proxy_panel.parquet"),
    Path("data/feature_store/opra_surface_panel.parquet"),
    Path("data/feature_store/option_greek_quality.csv"),
    Path("data/universe/multi_raw_close.csv"),
    Path("data/universe/vx_futures_daily.parquet"),
    Path("data/universe/vix_complex.parquet"),
)
PRIVATE_REBUILD_HOOK = Path("data/private_rebuild.py")
PROSPECTIVE_DECISION_DATE = "2026-07-31"
TOP_LEVEL_FILES = (
    Path(".github/workflows/ci.yml"),
    Path(".gitignore"),
    Path("AGENTS.md"),
    Path("Makefile"),
    Path("README.md"),
    Path("data/README.md"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)
ANALYSIS_FILES = tuple(
    Path("analysis") / name
    for name in ("__init__.py", "inference.py", "pipeline.py", "release.py")
)
PACKAGE_FILES = tuple(
    Path("src/option_portfolio") / name
    for name in (
        "__init__.py",
        "execution.py",
        "metrics.py",
        "model.py",
        "pricing.py",
        "risk_controls.py",
    )
)
TEST_FILES = tuple(
    Path("tests") / name
    for name in (
        "test_execution_risk.py",
        "test_inference.py",
        "test_metrics_pricing.py",
        "test_model.py",
        "test_release.py",
    )
)
PAPER_ROOT_FILES = tuple(
    Path("paper") / name
    for name in ("README.md", "paper.pdf", "paper.tex", "references.bib")
)
PAPER_SECTION_FILES = tuple(
    Path("paper/sections") / name
    for name in ("short_appendix.tex", "short_execution_audit.tex", "short_paper.tex")
)
EVIDENCE_FILES = tuple(
    Path("paper/evidence") / name
    for name in (
        "claims.csv",
        "cpcv_summary.csv",
        "historical_artifact_provenance.json",
        "inference_summary.csv",
        "legacy_e1_ablation.csv",
        "legacy_e1_cpcv_path_returns.csv",
        "legacy_e1_concentration.csv",
        "legacy_e1_scoreboard.csv",
        "model_progression_returns.csv",
        "pbo_claim_window.csv",
        "pbo_liquid_era.csv",
        "quote_sensitivity_monthly_returns.csv",
        "quote_sensitivity_summary.json",
        "r11_monthly_returns.csv",
        "r11_performance_summary.csv",
        "r11_weight_summary.csv",
        "r1_monthly_returns.csv",
        "r1_performance_summary.csv",
        "r1_r11_aligned_summary.csv",
        "r1_weight_summary.csv",
        "robustness_status.csv",
        "spread_source_summary.csv",
        "trial_counts.csv",
        "walk_forward_returns.csv",
    )
)
FIGURE_FILES = tuple(
    Path("paper/figures") / name
    for name in (
        "short_audit_scenario_ladder.pdf",
        "short_capacity_spread_panel.pdf",
        "short_deployment_constraints.pdf",
        "short_four_variant_scoreboard.pdf",
        "short_headline_wealth.pdf",
        "short_model_progression.pdf",
        "short_prototype_failure.pdf",
        "short_proxy_coverage_evidence.pdf",
        "short_robustness_heatmap.pdf",
        "short_theory_flow.pdf",
        "short_validation_distributions.pdf",
        "short_walk_forward_return_paths.pdf",
    )
)
TABLE_FILES = tuple(
    Path("paper/tables") / name
    for name in (
        "short_cpcv_windows.tex",
        "short_e1_channel_ablation.tex",
        "short_e1_concentration.tex",
        "short_execution_audit_summary.tex",
        "short_execution_spread_comparison.tex",
        "short_four_scenario_assumptions.tex",
        "short_inference_panel.tex",
        "short_liquidity_validation.tex",
        "short_r11_development_summary.tex",
        "short_r11_integer_repair_summary.tex",
        "short_r1_survival_summary.tex",
        "short_spread_source_ladder.tex",
    )
)
RELEASE_PATHS = tuple(
    sorted(
        {
            *TOP_LEVEL_FILES,
            *ANALYSIS_FILES,
            *PACKAGE_FILES,
            *TEST_FILES,
            *PAPER_ROOT_FILES,
            *PAPER_SECTION_FILES,
            *EVIDENCE_FILES,
            *FIGURE_FILES,
            *TABLE_FILES,
        }
    )
)
BANNED_EVIDENCE_COLUMNS = {
    "ask",
    "asset_id",
    "bid",
    "contract_id",
    "expiration",
    "expiry",
    "option_id",
    "option_symbol",
    "order_id",
    "quote_time",
    "strike",
    "symbol",
    "timestamp",
    "trade_id",
    "underlying_symbol",
}
LATEX_INTERMEDIATE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".toc",
}


@dataclass(frozen=True)
class VerificationResult:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _role(path: Path) -> str:
    if path == Path("paper/paper.pdf"):
        return "output"
    if path.parts[:2] in {
        ("paper", "figures"),
        ("paper", "tables"),
    }:
        return "frozen-asset"
    if path.parts[:2] == ("paper", "evidence"):
        return "evidence"
    if path.parts and path.parts[0] == "tests":
        return "test"
    if path.suffix in {".py", ".tex", ".bib"}:
        return "source"
    return "documentation"


def _producer(path: Path) -> str:
    if path.parts[:2] == ("paper", "evidence"):
        return "analysis.pipeline"
    if path == Path("paper/paper.pdf"):
        return "lualatex+bibtex"
    if path.parts[:2] in {("paper", "figures"), ("paper", "tables")}:
        return "historical-derived-asset"
    return "maintained-source"


def release_files(root: Path = ROOT) -> list[Path]:
    """Return the explicit public-release inventory."""

    del root
    return list(RELEASE_PATHS)


def _observed_release_files(root: Path) -> set[Path]:
    root = Path(root)
    observed: set[Path]
    if (root / ".git").exists() and shutil.which("git"):
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError("git could not enumerate the public repository")
        observed = {
            Path(item.decode("utf-8", errors="surrogateescape"))
            for item in result.stdout.split(b"\0")
            if item and (root / item.decode("utf-8", errors="surrogateescape")).is_file()
        }
    else:
        excluded = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".uv-cache",
            ".venv",
            "__pycache__",
            "build",
            "scratch",
            "tmp",
        }
        observed = {
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file()
            and not excluded.intersection(path.relative_to(root).parts)
            and not path.name.endswith(tuple(LATEX_INTERMEDIATE_SUFFIXES))
            and path.suffix not in {".pyc", ".tmp"}
        }
    observed.discard(Path(MANIFEST.name))
    return observed


def _inventory_errors(root: Path) -> list[str]:
    expected = set(RELEASE_PATHS)
    observed = _observed_release_files(root)
    errors = []
    if missing := sorted(expected - observed):
        errors.append(f"release files are missing: {missing}")
    if unexpected := sorted(observed - expected):
        errors.append(f"unexpected release files: {unexpected}")
    return errors


def write_manifest(root: Path = ROOT) -> None:
    rows = []
    for relative in release_files(root):
        path = root / relative
        rows.append(
            {
                "path": relative.as_posix(),
                "role": _role(relative),
                "producer": _producer(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with (root / MANIFEST.name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "role", "producer", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _prepare_empty_destination(destination: Path, clean: bool) -> None:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        if not clean:
            raise ValueError(f"destination must be empty: {destination}")
        if BUILD_ROOT.resolve() not in destination.parents:
            raise ValueError("clean builds are restricted to the repository build directory")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def build_candidate(destination: Path, root: Path = ROOT, clean: bool = False) -> None:
    """Copy the exact release into an empty candidate directory."""

    destination = Path(destination)
    if errors := _inventory_errors(root):
        raise ValueError("; ".join(errors))
    _prepare_empty_destination(destination, clean)
    paths = release_files(root)
    for relative in paths:
        if relative.parts[:2] == ("paper", "evidence"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    build_derived_evidence(
        destination / "paper/evidence",
        source=root / "paper/evidence",
    )
    if (root / MANIFEST.name).is_file():
        shutil.copyfile(root / MANIFEST.name, destination / MANIFEST.name)


def missing_private_inputs(root: Path = ROOT) -> list[Path]:
    return [root / path for path in PRIVATE_INPUTS if not (root / path).is_file()]


def run_private_rebuild(
    root: Path = ROOT,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VerificationResult:
    """Run the ignored maintainer rebuild hook and verify its complete candidate."""

    root = Path(root).resolve()
    missing = missing_private_inputs(root)
    if missing:
        return VerificationResult(
            tuple(
                f"private input is missing: {path.relative_to(root)}"
                for path in missing
            )
        )
    hook = root / PRIVATE_REBUILD_HOOK
    if not hook.is_file():
        return VerificationResult(
            (
                f"private rebuild hook is missing: {PRIVATE_REBUILD_HOOK}",
            )
        )
    destination = (root / "build/private-release").resolve()
    if destination.parent != (root / "build").resolve():
        raise ValueError("private rebuild destination escaped the build directory")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    completed = runner(
        [
            sys.executable,
            str(hook),
            "--destination",
            str(destination),
        ],
        cwd=root,
        check=False,
    )
    if completed.returncode:
        return VerificationResult(
            (f"private rebuild hook exited with status {completed.returncode}",)
        )
    return verify_candidate(destination)


def _check_manifest(root: Path, errors: list[str]) -> None:
    path = root / MANIFEST.name
    if not path.is_file():
        errors.append("release_manifest.csv is missing")
        return
    rows = pd.read_csv(path)
    required = ["path", "role", "producer", "bytes", "sha256"]
    if list(rows.columns) != required:
        errors.append("release_manifest.csv has the wrong schema")
        return
    expected = {item.as_posix() for item in release_files(root)}
    manifest_paths = rows["path"].astype(str)
    if manifest_paths.duplicated().any():
        errors.append("release_manifest.csv contains duplicate paths")
    observed = set(manifest_paths)
    if expected != observed:
        errors.append(
            f"manifest closure mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    for row in rows.to_dict(orient="records"):
        raw_path = str(row["path"])
        relative = Path(raw_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != raw_path
        ):
            errors.append(f"manifest path is not a safe relative path: {raw_path}")
            continue
        if str(row["role"]) != _role(relative):
            errors.append(f"manifest role differs: {raw_path}")
        if str(row["producer"]) != _producer(relative):
            errors.append(f"manifest producer differs: {raw_path}")
        target = root / relative
        if not target.is_file():
            errors.append(f"manifest path is missing: {raw_path}")
            continue
        if target.stat().st_size != int(row["bytes"]):
            errors.append(f"manifest byte size differs: {raw_path}")
        if sha256(target) != str(row["sha256"]):
            errors.append(f"manifest hash differs: {raw_path}")


def _check_candidate_closure(root: Path, errors: list[str]) -> None:
    expected = set(release_files(root)) | {Path(MANIFEST.name)}
    observed = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not path.name.endswith(tuple(LATEX_INTERMEDIATE_SUFFIXES))
    }
    if observed != expected:
        errors.append(
            f"candidate closure mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _check_evidence(root: Path, errors: list[str]) -> None:
    evidence = root / "paper/evidence"
    for path in evidence.glob("*.csv"):
        columns = {
            str(column).strip().lower()
            for column in pd.read_csv(path, nrows=0).columns
        }
        overlap = columns & BANNED_EVIDENCE_COLUMNS
        if overlap:
            errors.append(f"row-level columns in {path.relative_to(root)}: {sorted(overlap)}")
    for relative, expected in BASELINE_RETURN_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen return evidence differs: {relative}")

    summary_specs = (
        (
            "r1_monthly_returns.csv",
            "r1_performance_summary.csv",
            ["config", "strategy"],
        ),
        (
            "r11_monthly_returns.csv",
            "r11_performance_summary.csv",
            ["config", "strategy", "evidence_status"],
        ),
    )
    for returns_name, summary_name, keys in summary_specs:
        returns_path = evidence / returns_name
        summary_path = evidence / summary_name
        if not returns_path.is_file() or not summary_path.is_file():
            continue
        expected = summarize_returns(
            pd.read_csv(returns_path, float_precision="round_trip"),
            keys,
        ).replace(DISPLAY_NAMES)
        observed = pd.read_csv(summary_path, float_precision="round_trip")
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError:
            errors.append(f"corrected performance summary differs: {summary_name}")

    cpcv = pd.read_csv(evidence / "cpcv_summary.csv")
    ruined = cpcv[cpcv["default_share"] > 0]
    if len(ruined) != 2:
        errors.append("expected two defaulted full-window E1 CPCV rows")
    elif not (
        ruined["status"].eq("fail_default").all()
        and ruined[["sharpe_p05", "sharpe_p50", "sharpe_p95"]].isna().all().all()
    ):
        errors.append("ruined CPCV rows must fail and report Sharpe as NA")
    if (
        "relative_sharpe_p05" not in cpcv
        or cpcv["relative_sharpe_p05"].isna().any()
    ):
        errors.append("CPCV evidence omits relative-path p05 values")
    cpcv_table = (
        root / "paper/tables/short_cpcv_windows.tex"
    ).read_text(encoding="utf-8")
    for config in ("orig", "orig+VIX", "larger", "larger+VIX"):
        liquid = cpcv[
            cpcv["config"].eq(config)
            & cpcv["scope"].eq("liquid_era_2018_plus")
        ].iloc[0]
        claim = cpcv[
            cpcv["config"].eq(config)
            & cpcv["scope"].eq("claim_window_2020_plus")
        ].iloc[0]
        relative_values = (
            f"{liquid['relative_sharpe_p05']:.3f}",
            f"{claim['relative_sharpe_p05']:.3f}",
        )
        if not all(value in cpcv_table for value in relative_values):
            errors.append("CPCV table omits retained relative-path evidence")
            break

    cpcv_paths = pd.read_csv(evidence / "legacy_e1_cpcv_path_returns.csv")
    path_sizes = cpcv_paths.groupby("path_id", observed=True).size()
    absorbed = cpcv_paths[cpcv_paths["net_return"] <= -1.0].groupby(
        "path_id", observed=True
    )["return_date"].first()
    if (
        len(cpcv_paths) != 1034
        or len(path_sizes) != 11
        or not path_sizes.eq(94).all()
        or len(absorbed) != 11
        or not absorbed.eq("2020-03-31").all()
    ):
        errors.append("legacy E1 CPCV path evidence differs from the plotted failure")

    claims = (evidence / "claims.csv").read_text(encoding="utf-8")
    if "B Gamma + Gamma^T B^T" not in claims:
        errors.append("claim evidence omits factor-residual covariance cross terms")
    if "not realized fills" not in claims:
        errors.append("touch-price claim is not qualified")

    aligned = pd.read_csv(evidence / "r1_r11_aligned_summary.csv")
    required_metrics = {"annualized_mean_return", "cagr"}
    if not required_metrics.issubset(aligned):
        errors.append("aligned summary must report annualized mean return and CAGR")
    elif not np.allclose(
        aligned["annualized_mean_return"],
        aligned["sharpe"] * aligned["annualized_volatility"],
    ):
        errors.append("aligned summary annualized mean return is mislabeled")

    quote_summary = json.loads(
        (evidence / "quote_sensitivity_summary.json").read_text(encoding="utf-8")
    )
    ambiguous_keys: list[str] = []

    def find_ambiguous(value: object, path: str = "") -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                find_ambiguous(item, f"{path}/{index}")
        elif isinstance(value, dict):
            for key, item in value.items():
                if key == "annualized_return":
                    ambiguous_keys.append(f"{path}/{key}")
                find_ambiguous(item, f"{path}/{key}")

    find_ambiguous(quote_summary)
    if ambiguous_keys:
        errors.append(f"ambiguous annualized_return keys remain: {ambiguous_keys}")
    for row in quote_summary.get("headline_r11", []):
        try:
            expected_cagr = float(row["terminal_wealth"]) ** (
                12.0 / float(row["n_obs"])
            ) - 1.0
            if not np.isclose(float(row["cagr"]), expected_cagr, atol=1e-14):
                errors.append("quote-sensitivity CAGR differs from terminal wealth")
                break
        except (KeyError, TypeError, ValueError):
            errors.append("quote-sensitivity headline has an incomplete CAGR record")
            break
    medians = pd.DataFrame(
        quote_summary.get("liquidity_participation_medians", [])
    )
    median_columns = {
        "arm",
        "participation_volume_sum",
        "participation_volume_max",
        "participation_open_interest",
    }
    if len(medians) != 2 or set(medians.columns) != median_columns:
        errors.append("liquidity participation medians are missing")

    spread = pd.read_csv(evidence / "spread_source_summary.csv")
    if (
        len(spread) != 4
        or set(spread["config"]) != {"orig", "orig+VIX", "larger", "larger+VIX"}
        or not np.allclose(
            spread["proxy_row_share"],
            spread["proxy_cbbo_rows"]
            / (spread["exact_cbbo_rows"] + spread["proxy_cbbo_rows"]),
        )
        or set(spread["equity_universe_size"]) != {8, 56}
    ):
        errors.append("spread-source summary is incomplete or inconsistent")
    spread_table = (
        root / "paper/tables/short_spread_source_ladder.tex"
    ).read_text(encoding="utf-8")
    scenario_table = (
        root / "paper/tables/short_four_scenario_assumptions.tex"
    ).read_text(encoding="utf-8")
    for row in spread.to_dict(orient="records"):
        expected_spread_row = (
            f"{row['config']} & {int(row['exact_cbbo_rows'])} & "
            f"{int(row['proxy_cbbo_rows'])} & {row['proxy_row_share']:.3f} & "
            f"{row['median_relative_spread']:.3f}"
        )
        universe = (
            f"{int(row['equity_universe_size'])} equities"
            + (" + VIX" if bool(row["includes_vix"]) else "")
        )
        if expected_spread_row not in spread_table or (
            f"{row['config']} & {universe}" not in scenario_table
        ):
            errors.append("spread or universe table differs from aggregate evidence")
            break

    robustness = pd.read_csv(evidence / "robustness_status.csv")
    expected_checks = {
        "Claim CPCV",
        "Liquid CPCV",
        "MC refit",
        "MC resample",
        "PBO rank",
        "Repriced",
        "Rolling OOS",
    }
    if (
        len(robustness) != 28
        or set(robustness["check"]) != expected_checks
        or set(robustness["config"])
        != {"orig", "orig+VIX", "larger", "larger+VIX"}
    ):
        errors.append("robustness-status evidence has the wrong closure")
    else:
        for row in robustness.to_dict(orient="records"):
            primary = float(row["primary_value"])
            secondary = float(row["secondary_value"])
            check = str(row["check"])
            status = str(row["status"])
            if check in {"MC refit", "MC resample", "Repriced"}:
                expected_status = (
                    "pass"
                    if primary > 0.0
                    else "mixed"
                    if secondary > 0.0
                    else "fail"
                )
            elif check == "Rolling OOS":
                expected_status = "pass" if primary > 0.0 else "fail"
            elif check == "PBO rank":
                expected_status = (
                    "pass"
                    if primary <= 0.30
                    else "mixed"
                    if primary <= 0.50
                    else "fail"
                )
            else:
                match = cpcv[
                    cpcv["config"].eq(row["config"])
                    & cpcv["scope"].eq(
                        "liquid_era_2018_plus"
                        if check == "Liquid CPCV"
                        else "claim_window_2020_plus"
                    )
                ].iloc[0]
                expected_status = (
                    "pass"
                    if match["status"] == "survived"
                    else str(match["status"])
                )
            if status != expected_status:
                errors.append("robustness-status decision differs from its inputs")
                break

    r11_returns = pd.read_csv(evidence / "r11_monthly_returns.csv")
    scored = r11_returns[
        r11_returns["evidence_status"].eq("retrospective_development_sample")
    ]
    candidates = len(scored)
    feasible = int(scored["integer_conversion_feasible"].fillna(False).sum())
    abstentions = int(scored["integer_execution_abstained"].fillna(False).sum())
    integer_table = (
        root / "paper/tables/short_r11_integer_repair_summary.tex"
    ).read_text(encoding="utf-8")
    expected_integer_rows = {
        f"Cash abstention & {candidates} & {candidates} & {abstentions}",
        f"Direct truncation & {candidates} & {feasible} & {feasible}",
    }
    if not all(row in integer_table for row in expected_integer_rows):
        errors.append("integer outcome table differs from aggregate evidence")

    liquidity_table = (
        root / "paper/tables/short_liquidity_validation.tex"
    ).read_text(encoding="utf-8")
    breaches = pd.DataFrame(quote_summary.get("liquidity_breach_counts", []))
    for row in medians.merge(breaches, on="arm").to_dict(orient="records"):
        expected_row = (
            f"{row['arm']} & {row['participation_volume_sum']:.3f} & "
            f"{row['participation_volume_max']:.3f} & "
            f"{row['participation_open_interest']:.3f} & "
            f"{int(row['breach_optimizer_volume_0_05'])} & "
            f"{int(row['breach_capacity_volume_0_10'])} & "
            f"{int(row['breach_capacity_oi_0_02'])}"
        )
        if expected_row not in liquidity_table:
            errors.append("liquidity table differs from aggregate evidence")
            break

    provenance = json.loads(
        (evidence / "historical_artifact_provenance.json").read_text(encoding="utf-8")
    )
    snapshot = provenance.get("pre_cleanup_snapshot", {})
    artifact_hashes = {
        str(item.get("sha256")) for item in provenance.get("artifacts", [])
    }
    for field, expected in BASELINE_WEIGHT_HASHES.items():
        if snapshot.get(field) != expected or expected not in artifact_hashes:
            errors.append(f"historical weight provenance differs: {field}")


def _check_paper_assets(root: Path, errors: list[str]) -> None:
    paper = root / "paper"
    tex_files = [paper / "paper.tex", *sorted((paper / "sections").glob("*.tex"))]
    source = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    figures = set(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", source)
    )
    tables = set(re.findall(r"\\input\{tables/([^}]+)\}", source))
    actual_figures = {path.name for path in (paper / "figures").glob("*.pdf")}
    actual_tables = {path.stem for path in (paper / "tables").glob("*.tex")}
    if figures != actual_figures:
        errors.append("figure directory does not exactly match manuscript references")
    if tables != actual_tables:
        errors.append("table directory does not exactly match manuscript references")
    if len(figures) != 12 or len(tables) != 12:
        errors.append(
            "expected 12 referenced figures and 12 referenced tables, "
            f"got {len(figures)}/{len(tables)}"
        )

    for path in [paper / "README.md", *tex_files]:
        text = path.read_text(encoding="utf-8")
        if "\u2013" in text or "\u2014" in text:
            errors.append(f"Unicode dash in {path.relative_to(root)}")
        if str(root) in text:
            errors.append(f"absolute checkout path in {path.relative_to(root)}")
        if "settlement files whose historical source hashes" in text:
            errors.append(
                f"unsupported settlement-file hash claim in {path.relative_to(root)}"
            )
    if shutil.which("pdftotext"):
        figure = paper / "figures/short_audit_scenario_ladder.pdf"
        result = subprocess.run(
            ["pdftotext", str(figure), "-"],
            text=True,
            capture_output=True,
        )
        old_labels = ("observed-fill", "Annualized return")
        if result.returncode or any(label in result.stdout for label in old_labels):
            errors.append("quote-cost sensitivity figure retains an unsupported label")
        if "CAGR" not in result.stdout or "quote-cost sensitivity" not in result.stdout:
            errors.append("quote-cost sensitivity figure omits its corrected labels")


def _check_pdf(root: Path, errors: list[str]) -> None:
    pdf = root / "paper/paper.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        errors.append("paper/paper.pdf is missing or empty")
        return
    if shutil.which("pdfinfo"):
        result = subprocess.run(["pdfinfo", str(pdf)], text=True, capture_output=True)
        if result.returncode:
            errors.append("pdfinfo could not read paper/paper.pdf")
    if shutil.which("pdffonts"):
        result = subprocess.run(["pdffonts", str(pdf)], text=True, capture_output=True)
        if result.returncode or re.search(r"\bType\s+3\b", result.stdout):
            errors.append("paper/paper.pdf contains Type 3 fonts")


def verify_artifacts(root: Path = ROOT) -> VerificationResult:
    """Run read-only checks against committed derived evidence."""

    root = Path(root)
    errors = _inventory_errors(root)
    _check_manifest(root, errors)
    _check_evidence(root, errors)
    _check_paper_assets(root, errors)
    _check_pdf(root, errors)
    if (root / "paper/artifacts").exists() or (root / "analysis/artifacts").exists():
        errors.append("unfiltered legacy artifact directories remain")
    provenance = json.loads(
        (root / "paper/evidence/historical_artifact_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    decision = provenance.get("prospective_decision_date")
    if decision != PROSPECTIVE_DECISION_DATE:
        errors.append("prospective decision date differs from the registered release")
    return VerificationResult(tuple(errors))


def verify_candidate(root: Path) -> VerificationResult:
    """Verify a clean candidate, including exact file closure."""

    result = verify_artifacts(root)
    errors = list(result.errors)
    _check_candidate_closure(Path(root), errors)
    return VerificationResult(tuple(errors))


def promote_candidate(candidate: Path, root: Path = ROOT) -> VerificationResult:
    """Replace tracked outputs only after the candidate passes every public check."""

    candidate = Path(candidate)
    log = candidate / "paper/paper.log"
    if not log.is_file():
        return VerificationResult(("paper compilation log is missing",))
    log_text = log.read_text(encoding="utf-8", errors="replace")
    warnings = (
        "There were undefined references",
        "There were undefined citations",
        "multiply defined",
        "Overfull",
    )
    if found := [warning for warning in warnings if warning in log_text]:
        return VerificationResult(
            (f"paper compilation log contains release warnings: {found}",)
        )
    write_manifest(candidate)
    result = verify_candidate(candidate)
    if not result.ok:
        return result

    generated = candidate / "paper/evidence"
    tracked = root / "paper/evidence"
    generated_names = {path.name for path in generated.iterdir() if path.is_file()}
    tracked_names = {path.name for path in tracked.iterdir() if path.is_file()}
    if generated_names != tracked_names:
        return VerificationResult(
            (
                "evidence output closure differs: "
                f"missing={sorted(tracked_names - generated_names)}, "
                f"unexpected={sorted(generated_names - tracked_names)}",
            )
        )

    for path in generated.iterdir():
        if path.is_file():
            shutil.copyfile(path, tracked / path.name)
    shutil.copyfile(candidate / "paper/paper.pdf", root / "paper/paper.pdf")
    shutil.copyfile(candidate / MANIFEST.name, root / MANIFEST.name)
    return verify_artifacts(root)


def _print_result(result: VerificationResult) -> int:
    if result.ok:
        print("Artifact verification passed.")
        return 0
    for error in result.errors:
        print(f"FAIL: {error}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "build",
            "release",
            "verify-artifacts",
            "verify-candidate",
            "verify-full",
        ),
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--inputs-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inputs_only and args.command != "verify-full":
        parser.error("--inputs-only is only valid with verify-full")

    if args.command == "build":
        build_candidate(args.destination or BUILD_ROOT / "release", clean=True)
        return 0
    if args.command == "release":
        candidate = BUILD_ROOT / "release"
        if not candidate.is_dir():
            print("FAIL: build/release is missing; run `make paper` first")
            return 1
        return _print_result(promote_candidate(candidate))
    if args.command == "verify-artifacts":
        return _print_result(verify_artifacts())
    if args.command == "verify-candidate":
        return _print_result(
            verify_candidate(args.destination or BUILD_ROOT / "release")
        )

    missing = missing_private_inputs()
    if missing:
        print("Private rebuild inputs are missing:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        return 2
    if args.inputs_only:
        return 0
    return _print_result(run_private_rebuild())


if __name__ == "__main__":
    raise SystemExit(main())
