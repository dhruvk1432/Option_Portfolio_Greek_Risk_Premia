from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from data_pull.config import REPO_ROOT, credential_status, load_repo_env, validate_expected_inputs


@dataclass(frozen=True)
class Job:
    name: str
    description: str
    command: tuple[str, ...] = ()
    paid: bool = False
    required_any_env: tuple[str, ...] = ()
    internal: bool = False


JOBS: dict[str, Job] = {
    "validate-inputs": Job(
        name="validate-inputs",
        description="Check local expected paper-input files and credential presence without network calls.",
        internal=True,
    ),
    "public-universe-prices": Job(
        name="public-universe-prices",
        description="Fetch credential-free daily equity prices and volumes through yfinance.",
        command=("-m", "data_ingestion.market_data.fetch_universe_prices"),
    ),
    "public-vix-complex": Job(
        name="public-vix-complex",
        description="Fetch VIX family indices and VX futures files from public/yfinance/Cboe sources.",
        command=("-m", "data_ingestion.market_data.fetch_vix_complex"),
    ),
    "public-vro-soq": Job(
        name="public-vro-soq",
        description="Fetch free/public Cboe VIX VRO/SOQ final settlement files for option-paper expiry P&L.",
        command=("-m", "data_pull.cboe_vro_soq"),
    ),
    "public-cftc": Job(
        name="public-cftc",
        description="Fetch public CFTC CoT futures positioning used by broader regime research.",
        command=("-m", "data_ingestion.market_data.fetch_macro_fundamentals", "--jobs", "cftc"),
    ),
    "fred-macro": Job(
        name="fred-macro",
        description="Fetch FRED macro/rates series. Requires a free FRED_API_KEY.",
        command=("-m", "data_ingestion.market_data.fetch_macro_fundamentals", "--jobs", "fred"),
        required_any_env=("FRED_API_KEY",),
    ),
    "databento-probe": Job(
        name="databento-probe",
        description="Probe licensed Databento dataset coverage without printing credentials.",
        command=("-m", "data_ingestion.market_data.databento_probe"),
        paid=True,
        required_any_env=("DATABENTO_API_KEY", "DATABENTO_API_KEY2"),
    ),
    "databento-opra-equity": Job(
        name="databento-opra-equity",
        description="Fetch cost-guarded equity/ETF OPRA option slices for the option-only paper.",
        command=("-m", "data_ingestion.market_data.fetch_databento", "--job", "opra_multi"),
        paid=True,
        required_any_env=("DATABENTO_API_KEY", "DATABENTO_API_KEY2"),
    ),
    "databento-vix-options": Job(
        name="databento-vix-options",
        description="Fetch cost-guarded VIX option chain shards from OPRA via Databento.",
        command=("-m", "data_ingestion.market_data.fetch_databento", "--job", "vix_options"),
        paid=True,
        required_any_env=("DATABENTO_API_KEY", "DATABENTO_API_KEY2"),
    ),
    "paper-empirics": Job(
        name="paper-empirics",
        description="Regenerate option-only Markowitz empirical tables, figures, ledgers, and hashes.",
        command=("-m", "research.papers.option_only_markowitz.analysis.run_empirics", "--stage", "all"),
    ),
    "paper-verifier": Job(
        name="paper-verifier",
        description="Run the independent option-only Markowitz verification harness.",
        command=("-m", "research.papers.option_only_markowitz.verification.verify"),
    ),
}

PRESETS: dict[str, tuple[str, ...]] = {
    "validate": ("validate-inputs",),
    "public": ("validate-inputs", "public-universe-prices", "public-vix-complex", "public-vro-soq", "public-cftc"),
    "option-paper": (
        "validate-inputs",
        "public-universe-prices",
        "public-vix-complex",
        "public-vro-soq",
        "databento-probe",
        "databento-opra-equity",
        "databento-vix-options",
        "paper-empirics",
        "paper-verifier",
    ),
}


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def selected_jobs(preset: str, explicit_jobs: str) -> list[Job]:
    names: list[str] = []
    if preset:
        if preset not in PRESETS:
            raise SystemExit(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        names.extend(PRESETS[preset])
    if explicit_jobs:
        names.extend(_split_csv(explicit_jobs))
    if not names:
        names = ["validate-inputs"]
    bad = [name for name in names if name not in JOBS]
    if bad:
        raise SystemExit(f"unknown job(s) {bad}; choose from {sorted(JOBS)}")
    deduped = list(dict.fromkeys(names))
    return [JOBS[name] for name in deduped]


def build_plan(jobs: Sequence[Job], python: str) -> list[dict[str, object]]:
    env = os.environ
    plan = []
    for idx, job in enumerate(jobs, start=1):
        env_ready = True
        if job.required_any_env:
            env_ready = any(env.get(key) for key in job.required_any_env)
        plan.append(
            {
                "order": idx,
                "job": job.name,
                "description": job.description,
                "paid": job.paid,
                "required_any_env": list(job.required_any_env),
                "env_ready": env_ready,
                "internal": job.internal,
                "command": " ".join([python, *job.command]) if job.command else "internal",
            }
        )
    return plan


def run_internal(job: Job, repo_root: Path) -> dict[str, object]:
    if job.name != "validate-inputs":
        raise ValueError(f"unknown internal job {job.name}")
    return {
        "job": job.name,
        "credentials": credential_status(),
        "expected_inputs": validate_expected_inputs(repo_root),
    }


def run_job(job: Job, python: str, repo_root: Path, allow_paid: bool) -> dict[str, object]:
    if job.paid and not allow_paid:
        return {"job": job.name, "status": "skipped_paid_requires_allow_paid"}
    if job.required_any_env and not any(os.environ.get(key) for key in job.required_any_env):
        return {"job": job.name, "status": "skipped_missing_env", "required_any_env": list(job.required_any_env)}
    if job.internal:
        payload = run_internal(job, repo_root)
        payload["status"] = "ok"
        return payload
    cmd = [python, *job.command]
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return {"job": job.name, "status": "ok" if completed.returncode == 0 else "failed", "returncode": completed.returncode}


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute publication data pulls for Option_Only_Markowitz_Cashflow_Engineering.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root. Defaults to this checkout.")
    parser.add_argument("--env-file", default=".env", help="Environment file relative to repo root.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="validate")
    parser.add_argument("--jobs", default="", help=f"Comma-separated extra jobs from: {','.join(sorted(JOBS))}")
    parser.add_argument("--execute", action="store_true", help="Execute selected jobs. Default is dry-run planning only.")
    parser.add_argument("--allow-paid", action="store_true", help="Permit paid/licensed Databento jobs to run.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocess jobs.")
    parser.add_argument("--manifest", default="research/reports/pipeline_reports/data_pull_manifest.json")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if validation inputs are missing.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    env_path = load_repo_env(repo_root, args.env_file)

    jobs = selected_jobs(args.preset, args.jobs)
    plan = build_plan(jobs, args.python)
    payload: dict[str, object] = {
        "repo_root": str(repo_root),
        "env_file": str(env_path),
        "execute": bool(args.execute),
        "allow_paid": bool(args.allow_paid),
        "plan": plan,
    }

    if args.execute:
        results = [run_job(job, args.python, repo_root, args.allow_paid) for job in jobs]
        payload["results"] = results
    else:
        payload["note"] = "Dry run only. Re-run with --execute to perform non-paid jobs; add --allow-paid for Databento jobs."

    # Always include current local input status so users can see what remains missing.
    payload["expected_inputs"] = validate_expected_inputs(repo_root)
    manifest_path = repo_root / args.manifest
    write_manifest(manifest_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote manifest: {manifest_path}")

    if args.strict:
        missing = [row for row in validate_expected_inputs(repo_root) if not row["exists"]]
        if missing:
            raise SystemExit(2)
    if args.execute and any(result.get("status") == "failed" for result in payload.get("results", [])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
