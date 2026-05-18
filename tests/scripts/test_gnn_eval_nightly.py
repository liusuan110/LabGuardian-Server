"""``scripts/gnn_eval_nightly.sh`` — exit-code contract tests.

The script is the CI entry point. Its exit codes are part of the
public surface (see ``docs/CI_NIGHTLY.md``):

| code | meaning                                              |
|------|------------------------------------------------------|
| 0    | both splits passed gate                              |
| 2    | hard failure (crash / bad args)                      |
| 3    | one split exceeded ``false_pass_gate``               |
| 4    | skip — dataset / checkpoint missing on this machine  |

These tests pin the **soft skip vs hard fail** behaviour because that's
what makes CI on a fresh checkout green instead of red. The happy-path
exit 0 is verified opportunistically — only when the real dataset
artifacts happen to exist locally (we never generate them inside the
test).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NIGHTLY_SCRIPT = REPO_ROOT / "scripts" / "gnn_eval_nightly.sh"


def _run(
    *,
    ckpt: str = "checkpoints/p3_followup_v2/best_f1.pt",
    label_dir: str | None = None,
    skip_if_missing_data: str = "1",
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SKIP_IF_MISSING_DATA"] = skip_if_missing_data
    if label_dir is not None:
        env["LABEL_DIR"] = label_dir
    return subprocess.run(
        ["bash", str(NIGHTLY_SCRIPT), ckpt],
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Skip behaviour (exit 4) — the critical "CI doesn't fail-loud" contract
# ---------------------------------------------------------------------------


def test_exit_4_when_checkpoint_missing_and_skip_flag_on(tmp_path: Path) -> None:
    """A non-existent ckpt + ``SKIP_IF_MISSING_DATA=1`` (default) →
    exit 4 + friendly skip message. This is what CI sees on a fresh
    checkout before secrets are wired up."""

    proc = _run(ckpt=str(tmp_path / "nonexistent.pt"), skip_if_missing_data="1")
    assert proc.returncode == 4, (
        f"expected exit 4, got {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    # The skip message names what's missing and how to regenerate it
    assert "skipping" in proc.stdout
    assert "nonexistent.pt" in proc.stdout
    assert "gnn_generate_dataset" in proc.stdout
    assert "gnn_train_full" in proc.stdout


def test_exit_4_when_label_dir_missing(tmp_path: Path) -> None:
    """Missing dataset (LABEL_DIR points to nowhere) is also a soft
    skip, not a crash."""

    proc = _run(label_dir=str(tmp_path / "missing"), skip_if_missing_data="1")
    assert proc.returncode == 4
    assert "skipping" in proc.stdout


def test_exit_2_when_missing_artifacts_and_skip_flag_off(tmp_path: Path) -> None:
    """``SKIP_IF_MISSING_DATA=0`` flips the contract — same missing
    artifact yields exit 2 (hard failure). Production setups that have
    seeded artifacts should use this strictness."""

    proc = _run(
        ckpt=str(tmp_path / "nonexistent.pt"),
        skip_if_missing_data="0",
    )
    assert proc.returncode == 2
    assert "hard failure" in proc.stdout


# ---------------------------------------------------------------------------
# Happy path (exit 0) — only when real dataset/ckpt happen to exist
# ---------------------------------------------------------------------------


def _real_artifacts_present() -> bool:
    return (
        (REPO_ROOT / "checkpoints" / "p3_followup_v2" / "best_f1.pt").is_file()
        and (REPO_ROOT / "datasets" / "circuit_compare" / "labels").is_dir()
        and (REPO_ROOT / "datasets" / "circuit_compare" / "splits" / "test.json").is_file()
    )


@pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="real GNN dataset + checkpoint not present on this box",
)
def test_exit_0_on_full_run_with_clean_gate() -> None:
    """End-to-end smoke: when the real artifacts are seeded, the
    nightly should exit 0 (current state per RULE_SEMANTICS §6
    post-R6 wrap-up: rule_false_pass=0.0000 on both splits)."""

    proc = _run()
    assert proc.returncode == 0, (
        f"expected exit 0, got {proc.returncode}\n"
        f"stdout (tail): {proc.stdout[-2000:]}\n"
        f"stderr (tail): {proc.stderr[-500:]}"
    )
    # The summary section pins the headline number
    assert "rule_false_pass=0.0000" in proc.stdout, (
        "nightly script must print rule_false_pass=0.0000 on test split"
    )
    # Both eval dirs should now have metrics.json
    for sub in ("p5_eval", "p5_eval_val", "p5_eval_rule_only"):
        assert (REPO_ROOT / "checkpoints" / sub / "metrics.json").is_file(), (
            f"expected metrics.json under checkpoints/{sub}/"
        )


# ---------------------------------------------------------------------------
# Workflow file shape — does the file exist + look sane?
# ---------------------------------------------------------------------------


def test_github_actions_workflow_file_exists() -> None:
    """The CI surface is the workflow file; pin its existence so we
    notice if it gets accidentally deleted in a refactor."""

    wf = REPO_ROOT / ".github" / "workflows" / "gnn-eval-nightly.yml"
    assert wf.is_file(), f"missing CI workflow at {wf}"
    text = wf.read_text()
    # Schedule + manual dispatch + push-on-relevant-paths all present
    assert "cron:" in text
    assert "workflow_dispatch" in text
    assert "gnn_eval_nightly.sh" in text


def test_ci_docs_pointer_present() -> None:
    """docs/CI_NIGHTLY.md is referenced from RISK_REGISTER + README;
    make sure it still exists."""

    doc = REPO_ROOT / "docs" / "CI_NIGHTLY.md"
    assert doc.is_file()
    text = doc.read_text()
    assert "Exit codes" in text or "exit codes" in text.lower()
    assert "SKIP_IF_MISSING_DATA" in text
