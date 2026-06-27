"""Integration guard for the option-only Markowitz verification harness."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "research/papers/option_only_markowitz"
VERIFY_DIR = PAPER / "verification"


class TestOptionOnlyVerificationHarness(unittest.TestCase):
    def test_fast_verification_harness_passes_and_writes_contract_outputs(self):
        required_artifact = PAPER / "tables/empirical_summary.json"
        if not required_artifact.exists():
            self.skipTest("option-only Markowitz artifacts have not been generated")

        cmd = [
            sys.executable,
            "-m",
            "research.papers.option_only_markowitz.verification.verify",
            "--skip-regenerate",
            "--skip-compile",
            "--skip-render",
        ]
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout[-2000:])

        summary_path = VERIFY_DIR / "verification_summary.json"
        failed_path = VERIFY_DIR / "failed_checks.csv"
        checks_path = VERIFY_DIR / "verification_checks.csv"
        manifest_path = VERIFY_DIR / "hash_manifest.csv"
        report_path = VERIFY_DIR / "verification_report.md"
        for path in [summary_path, failed_path, checks_path, manifest_path, report_path]:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 0, path)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["critical_failures"], 0)
        self.assertGreaterEqual(summary["total_checks"], 170)

        failed = pd.read_csv(failed_path)
        self.assertTrue(failed.empty, failed.to_dict(orient="records"))
        manifest = pd.read_csv(manifest_path)
        self.assertGreaterEqual(len(manifest), 30)
        self.assertIn(
            "option_only_portfolio_optimization_dhruv_kohli.pdf",
            set(manifest["path"]),
        )


if __name__ == "__main__":
    unittest.main()
