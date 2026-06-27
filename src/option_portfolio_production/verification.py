"""Independent production verification harness.

This verifier is stricter than the paper verifier.  It fails when the available artifacts
are still research-grade proxies: VIX proxy settlement, missing fills, missing margin,
missing assignment ledger, or unreconciled broker/data state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import json

import numpy as np
import pandas as pd


@dataclass
class ProductionCheck:
    name: str
    category: str
    passed: bool
    critical: bool = True
    observed: Any = None
    expected: Any = None

    def row(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "passed": bool(self.passed),
            "critical": bool(self.critical),
            "observed": json.dumps(self.observed, default=str) if self.observed is not None else "",
            "expected": json.dumps(self.expected, default=str) if self.expected is not None else "",
        }


class ProductionVerifier:
    def __init__(self, paper_root: Path, output_dir: Path | None = None) -> None:
        self.paper_root = Path(paper_root)
        self.output_dir = Path(output_dir) if output_dir else self.paper_root / "production_verification"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checks: list[ProductionCheck] = []

    def check(self, name: str, category: str, passed: bool, *, critical: bool = True, observed: Any = None, expected: Any = None) -> None:
        self.checks.append(ProductionCheck(name, category, bool(passed), critical, observed, expected))

    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.output_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def run(self) -> dict[str, Any]:
        self.checks = []
        settlement = self._read_csv("settlement_ledger.csv")
        execution = self._read_csv("execution_ledger.csv")
        fills = self._read_csv("fill_ledger.csv")
        margin = self._read_csv("margin_ledger.csv")
        assignment = self._read_csv("assignment_ledger.csv")
        recon = self._read_csv("data_reconciliation_ledger.csv")
        broker_recon = self._read_csv("broker_position_reconciliation.csv")

        self._check_settlement(settlement)
        self._check_execution_and_fills(execution, fills)
        self._check_margin(execution, margin)
        self._check_assignment(execution, assignment)
        self._check_reconciliation(recon, broker_recon)
        return self.write_outputs()

    def _check_settlement(self, settlement: pd.DataFrame) -> None:
        self.check("settlement ledger exists", "settlement", not settlement.empty, observed=len(settlement), expected=">0 rows")
        if settlement.empty:
            return
        required = {"symbol", "asset_class", "expiry", "production_settlement_source"}
        self.check("settlement ledger schema", "settlement", required.issubset(settlement.columns), observed=sorted(settlement.columns), expected=sorted(required))
        if required.issubset(settlement.columns):
            vix = settlement[settlement["asset_class"].astype(str).str.lower().eq("vix_option")]
            exact = vix["production_settlement_source"].astype(str).eq("vro_soq_exact") if not vix.empty else pd.Series(dtype=bool)
            self.check("headline VIX rows use exact VRO/SOQ", "settlement", not vix.empty and bool(exact.all()), observed=vix["production_settlement_source"].value_counts().to_dict() if not vix.empty else {}, expected="all vro_soq_exact")

    def _check_execution_and_fills(self, execution: pd.DataFrame, fills: pd.DataFrame) -> None:
        self.check("execution ledger exists", "execution", not execution.empty, observed=len(execution), expected=">0 rows")
        self.check("fill ledger exists", "execution", not fills.empty, observed=len(fills), expected=">0 rows")
        if not execution.empty:
            cols = {"decision_time", "symbol", "side", "contracts", "limit_price", "routing_policy"}
            self.check("execution ledger schema", "execution", cols.issubset(execution.columns), observed=sorted(execution.columns), expected=sorted(cols))
            if "allow_market" in execution:
                self.check("no non-emergency market orders", "execution", ~execution["allow_market"].astype(str).str.lower().isin({"true", "1"}).any(), observed=execution.get("allow_market", pd.Series(dtype=object)).value_counts().to_dict())
        if not fills.empty:
            cols = {"order_id", "symbol", "contracts", "price", "fees", "fill_model"}
            self.check("fill ledger schema", "execution", cols.issubset(fills.columns), observed=sorted(fills.columns), expected=sorted(cols))
            if cols.issubset(fills.columns):
                optimistic = fills["fill_model"].astype(str).str.contains("midpoint", case=False, na=False).any()
                self.check("no optimistic midpoint fill model", "execution", not bool(optimistic), observed=fills["fill_model"].value_counts().to_dict())
                self.check("positive fill prices", "execution", pd.to_numeric(fills["price"], errors="coerce").gt(0).all(), observed=fills["price"].describe().to_dict())
                self.check("nonnegative explicit fees", "execution", pd.to_numeric(fills["fees"], errors="coerce").ge(0).all(), observed=fills["fees"].describe().to_dict())

    def _check_margin(self, execution: pd.DataFrame, margin: pd.DataFrame) -> None:
        self.check("margin ledger exists", "margin", not margin.empty, observed=len(margin), expected=">0 rows")
        if margin.empty:
            return
        cols = {"symbol", "margin_requirement", "stress_loss", "assignment_notional", "margin_preview_status"}
        self.check("margin ledger schema", "margin", cols.issubset(margin.columns), observed=sorted(margin.columns), expected=sorted(cols))
        if cols.issubset(margin.columns):
            self.check("margin requirements finite", "margin", np.isfinite(pd.to_numeric(margin["margin_requirement"], errors="coerce")).all(), observed=margin["margin_requirement"].head().tolist())
            self.check("margin previews pass", "margin", margin["margin_preview_status"].astype(str).str.lower().eq("pass").all(), observed=margin["margin_preview_status"].value_counts().to_dict())
            if not execution.empty and "side" in execution:
                short_symbols = set(execution.loc[execution["side"].astype(str).str.lower().eq("sell"), "symbol"].astype(str))
                margin_symbols = set(margin["symbol"].astype(str))
                self.check("short orders have margin rows", "margin", short_symbols.issubset(margin_symbols), observed=sorted(short_symbols - margin_symbols))

    def _check_assignment(self, execution: pd.DataFrame, assignment: pd.DataFrame) -> None:
        self.check("assignment ledger exists", "assignment", not assignment.empty, observed=len(assignment), expected=">0 rows or explicit no-assignment row")
        if assignment.empty:
            return
        cols = {"symbol", "event_time", "stock_symbol", "stock_quantity", "cash_flow", "reason"}
        self.check("assignment ledger schema", "assignment", cols.issubset(assignment.columns), observed=sorted(assignment.columns), expected=sorted(cols))

    def _check_reconciliation(self, recon: pd.DataFrame, broker_recon: pd.DataFrame) -> None:
        self.check("data reconciliation ledger exists", "reconciliation", not recon.empty, observed=len(recon), expected=">0 rows")
        if not recon.empty and "passed" in recon:
            self.check("vendor quote reconciliation passes", "reconciliation", recon["passed"].astype(str).str.lower().isin({"true", "1", "yes"}).all(), observed=recon.get("reasons", pd.Series(dtype=object)).value_counts().to_dict())
        self.check("broker position reconciliation exists", "reconciliation", not broker_recon.empty, observed=len(broker_recon), expected=">0 rows")
        if not broker_recon.empty and "passed" in broker_recon:
            self.check("broker positions reconcile", "reconciliation", broker_recon["passed"].astype(str).str.lower().isin({"true", "1", "yes"}).all(), observed=broker_recon.get("reasons", pd.Series(dtype=object)).value_counts().to_dict())

    def write_outputs(self) -> dict[str, Any]:
        rows = pd.DataFrame([c.row() for c in self.checks])
        failed = rows[~rows["passed"]] if not rows.empty else pd.DataFrame()
        critical_failed = failed[failed["critical"]] if not failed.empty else pd.DataFrame()
        rows.to_csv(self.output_dir / "production_verification_checks.csv", index=False)
        failed.to_csv(self.output_dir / "production_failed_checks.csv", index=False)
        summary = {
            "status": "pass" if critical_failed.empty else "fail",
            "critical_failures": int(len(critical_failed)),
            "total_failures": int(len(failed)),
            "total_checks": int(len(rows)),
            "output_dir": str(self.output_dir),
        }
        (self.output_dir / "production_verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report = ["# Production Verification Report", "", f"Status: **{summary['status'].upper()}**", f"Critical failures: `{summary['critical_failures']}`", ""]
        if failed.empty:
            report.append("No failed checks.")
        else:
            report.extend(["## Failed Checks", ""])
            for _, row in failed.iterrows():
                report.append(f"- `{row['category']}` / `{row['name']}`: observed={row['observed']} expected={row['expected']}")
        (self.output_dir / "production_verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production-grade verification for option-only Markowitz.")
    parser.add_argument("--paper-root", default="research/papers/option_only_markowitz")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    verifier = ProductionVerifier(Path(args.paper_root), Path(args.output_dir) if args.output_dir else None)
    summary = verifier.run()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
