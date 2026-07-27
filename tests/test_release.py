from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import analysis.release as release
from analysis.pipeline import build_derived_evidence
from analysis.release import (
    BANNED_EVIDENCE_COLUMNS,
    PRIVATE_INPUTS,
    PRIVATE_REBUILD_HOOK,
    PROSPECTIVE_DECISION_DATE,
    build_candidate,
    missing_private_inputs,
    run_private_rebuild,
    verify_artifacts,
    verify_candidate,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_derived_evidence_build_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_derived_evidence(first)
    build_derived_evidence(second)

    assert _tree_hash(first) == _tree_hash(second)


def test_derived_evidence_build_is_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_derived_evidence(first)
    build_derived_evidence(second, source=first)

    assert _tree_hash(first) == _tree_hash(second)


def test_regenerated_summaries_use_neutral_reader_labels(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"

    build_derived_evidence(evidence)

    r1 = pd.read_csv(evidence / "r1_performance_summary.csv")
    r11 = pd.read_csv(evidence / "r11_performance_summary.csv")
    labels = " ".join([*r1["strategy"].astype(str), *r11["strategy"].astype(str)])
    assert "repaired" not in labels.lower()
    assert "positive-edge deployment" not in labels.lower()
    assert "15% volatility-ceiling specification (R1)" in set(r1["strategy"])
    assert "25% volatility-ceiling specification (R1.1)" in set(r11["strategy"])


def test_candidate_build_is_deterministic_and_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_candidate(first)
    build_candidate(second)

    assert _tree_hash(first) == _tree_hash(second)
    assert (first / "paper/paper.pdf").is_file()
    assert not (first / "paper/option_only_portfolio_optimization_dhruv_kohli.pdf").exists()


def test_public_artifact_verification_is_read_only() -> None:
    before = {
        path: path.stat().st_mtime_ns
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "build" not in path.parts
    }

    result = verify_artifacts(ROOT)

    after = {path: path.stat().st_mtime_ns for path in before}
    assert result.ok, result.errors
    assert after == before


def test_missing_private_inputs_are_reported_per_path(tmp_path: Path) -> None:
    missing = missing_private_inputs(tmp_path)

    assert missing == [tmp_path / path for path in PRIVATE_INPUTS]


def test_verify_full_preflight_stops_before_data_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = ROOT / PRIVATE_INPUTS[0]
    monkeypatch.setattr(release, "missing_private_inputs", lambda: [missing])
    monkeypatch.setattr(
        release,
        "run_private_rebuild",
        lambda: pytest.fail("preflight must not start the private rebuild"),
    )

    status = release.main(["verify-full", "--inputs-only"])

    assert status == 2
    assert str(PRIVATE_INPUTS[0]) in capsys.readouterr().out


def test_prospective_date_matches_the_registered_release() -> None:
    provenance = json.loads(
        (ROOT / "paper/evidence/historical_artifact_provenance.json").read_text(
            encoding="utf-8"
        )
    )

    assert provenance["prospective_decision_date"] == PROSPECTIVE_DECISION_DATE
    snapshot = provenance["pre_cleanup_snapshot"]
    assert snapshot["dirty_worktree_diff_sha256"] == (
        "9b12c7943804e02fdc300d3600ba95a6c65a8e99923ad13c0c7fd23152c5043b"
    )
    assert snapshot["r1_returns_sha256"] == (
        "61b29ffc5faf9ca2e33f32671a8c928547b17c71dd1d8d3c1839180ba04edec9"
    )
    assert snapshot["r11_returns_sha256"] == (
        "e0a292a187c848cb01ecda637d68fdb1af3ef071dce56326f9f5f345b6d6a662"
    )
    assert snapshot["r1_weights_sha256"] == (
        "ce8be5caa03ca3d6b105c829693886ecfac21ef58083069a525498847558590d"
    )
    assert snapshot["r11_weights_sha256"] == (
        "bf59c5766b947b16dffb066d4f29f96e524da0dc41342b006b53f745f21233b5"
    )


def test_no_contract_or_quote_level_columns_in_public_evidence() -> None:
    for path in (ROOT / "paper/evidence").glob("*.csv"):
        columns = {
            str(column).strip().lower()
            for column in pd.read_csv(path, nrows=0).columns
        }
        assert BANNED_EVIDENCE_COLUMNS.isdisjoint(columns), path


def test_candidate_rejects_unexpected_row_level_evidence(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    build_candidate(candidate)
    (candidate / "paper/evidence/raw_ledger.csv").write_text(
        "contract_id,quote_time,return\nexample,2026-01-01,0.1\n",
        encoding="utf-8",
    )

    errors = verify_candidate(candidate).errors

    assert any("unexpected release files" in error for error in errors)
    assert any("row-level columns" in error for error in errors)


def test_candidate_build_rejects_an_unexpected_public_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build_candidate(source)
    unexpected = source / "scripts/unexpected.py"
    unexpected.parent.mkdir()
    unexpected.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected release files"):
        build_candidate(tmp_path / "candidate", root=source)


def test_public_verifier_recomputes_corrected_performance_summaries(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    build_candidate(candidate)
    summary_path = candidate / "paper/evidence/r1_performance_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[0, "max_drawdown"] = 0.0
    summary.to_csv(summary_path, index=False)
    write_manifest(candidate)

    errors = verify_artifacts(candidate).errors

    assert any("corrected performance summary differs" in error for error in errors)


def test_quote_summary_uses_cagr_without_ambiguous_return_names(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    build_derived_evidence(evidence)
    payload = json.loads(
        (evidence / "quote_sensitivity_summary.json").read_text(encoding="utf-8")
    )

    def keys(value: object):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert "annualized_return" not in set(keys(payload))
    for row in payload["headline_r11"]:
        expected = row["terminal_wealth"] ** (12.0 / row["n_obs"]) - 1.0
        assert row["cagr"] == pytest.approx(expected, abs=1e-14)


def _create_private_inputs(root: Path) -> None:
    for relative in PRIVATE_INPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_private_rebuild_requires_the_ignored_maintainer_hook(
    tmp_path: Path,
) -> None:
    _create_private_inputs(tmp_path)

    result = run_private_rebuild(tmp_path)

    assert result.errors == (
        f"private rebuild hook is missing: {PRIVATE_REBUILD_HOOK}",
    )


def test_private_rebuild_has_a_verified_success_path(tmp_path: Path) -> None:
    _create_private_inputs(tmp_path)
    hook = tmp_path / PRIVATE_REBUILD_HOOK
    hook.write_text("# private maintainer implementation\n", encoding="utf-8")

    def fake_runner(command, **kwargs):
        assert command[1] == str(hook)
        assert command[2] == "--destination"
        assert kwargs["cwd"] == tmp_path.resolve()
        build_candidate(Path(command[3]))
        return subprocess.CompletedProcess(command, 0)

    result = run_private_rebuild(tmp_path, runner=fake_runner)

    assert result.ok, result.errors


def test_cpcv_default_precedence_is_published() -> None:
    summary = pd.read_csv(ROOT / "paper/evidence/cpcv_summary.csv")
    ruined = summary[summary["default_share"] > 0]

    assert len(ruined) == 2
    assert ruined["status"].eq("fail_default").all()
    assert ruined[["sharpe_p05", "sharpe_p50", "sharpe_p95"]].isna().all().all()


def test_model_progression_figure_has_compact_machine_evidence() -> None:
    returns = pd.read_csv(ROOT / "paper/evidence/model_progression_returns.csv")

    assert len(returns) == 60
    assert list(returns.columns) == [
        "return_date",
        "legacy_e1_net_return",
        "r1_net_return",
        "r11_net_return",
    ]
    wealth = (1.0 + returns.iloc[:, 1:]).prod()
    assert wealth["legacy_e1_net_return"] == pytest.approx(83.91046218418842)
    assert wealth["r1_net_return"] == pytest.approx(2.814024614739748)
    assert wealth["r11_net_return"] == pytest.approx(3.201297786396169)


def test_aligned_summary_distinguishes_mean_return_from_cagr(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    build_derived_evidence(evidence)
    summary = pd.read_csv(evidence / "r1_r11_aligned_summary.csv")

    row = summary[
        summary["config"].eq("orig+VIX")
        & summary["strategy"].eq("25% volatility-ceiling specification (R1.1)")
        & summary["window"].eq("aligned_2018_2026")
    ].iloc[0]
    assert row["annualized_mean_return"] == pytest.approx(0.15814938309994767)
    assert row["cagr"] == pytest.approx(0.1600499557839527)
    assert row["annualized_mean_return"] != row["cagr"]


def test_paper_does_not_claim_unpublished_settlement_hashes() -> None:
    paper = (ROOT / "paper/paper.tex").read_text(encoding="utf-8")

    assert "settlement files whose historical source hashes" not in paper
