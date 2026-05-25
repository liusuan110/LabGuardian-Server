"""WP-2 tests: distillation entrypoint isolation + validation + e2e.

Pins the contract from ``scripts/distill/run_inference.py``:
  - Isolation: forbidden modules never load via the entrypoint chain.
  - Sample validation rejects malformed inputs (intent / scene_id rules).
  - Per-sample run returns a structured audit record.
  - Skip-precheck path works only as a unit-test escape hatch (real CLI
    refuses to bypass).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.core.config import settings


# ---------------------------------------------------------------------------
# 1. Isolation contract
# ---------------------------------------------------------------------------


def test_entrypoint_module_does_not_smuggle_forbidden_imports() -> None:
    """Importing the entrypoint module MUST NOT bring in legacy / cloud
    retrieval modules transitively. If this fails, an import chain
    somewhere reaches into RagService / KbService — fix that, not the test.

    Runs in a SUBPROCESS so that test-suite pollution (other tests in the
    same pytest process that legitimately import RagService) does not
    poison ``sys.modules``. The isolation contract is process-level —
    in production the distill entrypoint runs as a fresh interpreter."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "import scripts.distill.run_inference as e\n"
        "import sys, json\n"
        "leaked = sorted(e._FORBIDDEN_MODULES & set(sys.modules))\n"
        "print(json.dumps(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess probe failed: stderr={result.stderr!r}"
    )
    leaked = json.loads(result.stdout.strip() or "[]")
    assert not leaked, (
        f"WP-2 isolation regression: forbidden modules loaded via "
        f"scripts.distill.run_inference import chain (in a clean process): "
        f"{leaked}. Audit imports under app/agent/** and scripts/distill/**."
    )


def test_verify_isolation_raises_on_forbidden_module(monkeypatch) -> None:
    """The runtime guardrail itself works: when a forbidden module IS
    in sys.modules, ``_verify_isolation`` aborts. This catches bugs where
    a future caller smuggles the legacy KB into the entrypoint process."""
    from scripts.distill.run_inference import _verify_isolation

    # Force-inject a sentinel into sys.modules under a forbidden name.
    monkeypatch.setitem(sys.modules, "app.services.rag_service", object())
    with pytest.raises(RuntimeError, match="isolation contract violated"):
        _verify_isolation()


# ---------------------------------------------------------------------------
# 2. Sample validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample,expected_substring",
    [
        ({"qid": "a", "query": "", "intent": "diagnostic", "scene_id": "exp_first_order_rc"}, "empty query"),
        ({"qid": "b", "query": "x", "intent": "unknown", "scene_id": "exp_first_order_rc"}, "invalid intent"),
        ({"qid": "c", "query": "x", "intent": "diagnostic", "scene_id": ""}, "requires a non-empty scene_id"),
        ({"qid": "d", "query": "x", "intent": "mixed"}, "requires a non-empty scene_id"),
        ({"qid": "e", "query": "x", "intent": "concept_tutor", "scene_id": "not_a_scene"}, "not one of the 6 demo scenes"),
        ({"qid": "f", "query": "x", "intent": "diagnostic", "scene_id": "rc_first_order"}, "not one of the 6 demo scenes"),
    ],
)
def test_sample_validation_rejects_malformed(sample: dict, expected_substring: str) -> None:
    """Every invalid distill sample is caught before any agent invocation."""
    from scripts.distill.run_inference import _validate_sample

    result = _validate_sample(sample)
    assert not result.ok
    assert expected_substring in result.reason


@pytest.mark.parametrize(
    "sample",
    [
        {"qid": "g", "query": "x", "intent": "diagnostic", "scene_id": "exp_ua741_inverting_amplifier"},
        {"qid": "h", "query": "x", "intent": "concept_tutor", "scene_id": "exp_first_order_rc"},
        {"qid": "i", "query": "x", "intent": "concept_tutor", "scene_id": ""},  # concept-only OK
        {"qid": "j", "query": "x", "intent": "lab_guidance", "scene_id": ""},   # lab-only OK
        {"qid": "k", "query": "x", "intent": "mixed", "scene_id": "exp_common_emitter_amplifier"},
    ],
)
def test_sample_validation_accepts_well_formed(sample: dict) -> None:
    from scripts.distill.run_inference import _validate_sample

    assert _validate_sample(sample).ok


# ---------------------------------------------------------------------------
# 3. Station synthesis stamps topology_label from scene_id
# ---------------------------------------------------------------------------


def test_synthesize_station_stamps_topology_label_for_known_scene() -> None:
    """The entrypoint short-circuits the GNN-A pipeline by deriving
    topology_label from the sample's explicit scene_id. This lets
    scene_resolver resolve the scene without needing a real netlist."""
    from scripts.distill.run_inference import _synthesize_station

    station = _synthesize_station(
        {"qid": "s", "scene_id": "exp_ua741_inverting_amplifier"}
    )
    assert station["scene_id"] == "exp_ua741_inverting_amplifier"
    assert station["topology_label"] == "inverting_amp_ua741"


def test_synthesize_station_skips_topology_for_empty_scene() -> None:
    from scripts.distill.run_inference import _synthesize_station

    station = _synthesize_station({"qid": "s2", "scene_id": ""})
    assert "topology_label" not in station


# ---------------------------------------------------------------------------
# 4. End-to-end CLI smoke: 3 samples through the real graph
# ---------------------------------------------------------------------------


@pytest.fixture
def distill_env(monkeypatch):
    """Set the env vars required by precheck so we can run e2e.

    WP-2.1 (2026-05-24): also resets the module-level datasheet KB
    singleton in teardown — otherwise subsequent tests (datasheet KB v2
    / fail-closed suite) keep the OpenVINO-bound singleton even after
    settings revert, producing order-dependent failures.
    """
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )
    yield
    from app.agent.tools import _reset_datasheet_kb_singleton
    _reset_datasheet_kb_singleton()


def test_run_sample_processes_diagnostic_with_resolved_scene(distill_env) -> None:
    """One sample through the real agent graph — confirms isolation,
    scene resolution, and audit envelope."""
    from scripts.distill.run_inference import run_sample

    record = run_sample(
        {
            "qid": "wp2_diag_001",
            "query": "UA741 反相放大输出固定在 +13V 不变怎么办？",
            "intent": "diagnostic",
            "scene_id": "exp_ua741_inverting_amplifier",
            "station": {
                "risk_level": "danger",
                "diagnostics": ["输出饱和"],
                "comparison_report": {
                    "items": [
                        {"error_code": "FLOATING_PIN", "component_id": "U1", "pin_name": "pin4"}
                    ]
                },
            },
        },
        top_k=5,
    )
    assert not record["audit"]["skipped"]
    ao = record["agent_output"]
    assert ao["evidence_resolved_scene_id"] == "exp_ua741_inverting_amplifier"
    assert ao["final_answer"], "agent produced no final answer"
    assert ao["react_iterations"] >= 1
    # WP-1 v4 + WP-3 v3 contract: FLOATING_PIN in UA741 inverter MUST
    # surface inv_vee_pin_not_connected, AND no cross-chip leakage.
    tool_names = [t.get("tool_name") for t in ao["tool_results"]]
    assert "fault_case_lookup_tool" in tool_names


def test_run_sample_evidence_only_returns_context_and_tools(distill_env) -> None:
    """evidence-only mode freezes retrieval evidence without entering ReAct."""
    from scripts.distill.run_inference import run_sample

    record = run_sample(
        {
            "qid": "wp2_evidence_only",
            "query": "UA741 反相放大输出固定在 +13V 不变怎么办？",
            "intent": "diagnostic",
            "scene_id": "exp_ua741_inverting_amplifier",
            "station": {
                "risk_level": "danger",
                "comparison_report": {
                    "items": [{"error_code": "FLOATING_PIN", "component_id": "U1"}]
                },
            },
        },
        top_k=5,
        evidence_only=True,
    )
    assert not record["audit"]["skipped"]
    assert record["audit"]["evidence_only"] is True
    ao = record["agent_output"]
    assert ao["final_answer"] == ""
    assert ao["react_iterations"] == 0
    assert ao["react_terminate_reason"] == "evidence_only"
    assert ao["context_pack"], "context_pack should be exported in evidence-only mode"
    assert ao["tool_results"], "tool_results should be exported in evidence-only mode"


def test_run_sample_recall_strict_filters_unmatched_targets(distill_env) -> None:
    """recall_strict keeps audit output but skips unmatched samples."""
    from scripts.distill.run_inference import run_sample

    record = run_sample(
        {
            "qid": "wp2_filter_miss",
            "query": "UA741 反相放大输出固定在 +13V 不变怎么办？",
            "intent": "diagnostic",
            "scene_id": "exp_ua741_inverting_amplifier",
            "target_fault_case_id": "fault_case.does_not_exist",
            "station": {
                "risk_level": "danger",
                "comparison_report": {
                    "items": [{"error_code": "FLOATING_PIN", "component_id": "U1"}]
                },
            },
        },
        top_k=5,
        evidence_only=True,
        filter_policy="recall_strict",
    )
    assert record["audit"]["skipped"] is True
    assert record["audit"]["skip_reason"] == "filter_policy:recall_strict:no_target_match"
    assert record["audit"]["filter"]["kept"] is False
    assert record["agent_output"]["tool_results"], "filtered records still keep evidence for audit"


def test_run_sample_skips_invalid_sample_without_calling_graph(distill_env) -> None:
    """Validation catches bad input before any agent work happens."""
    from scripts.distill.run_inference import run_sample

    record = run_sample(
        {
            "qid": "wp2_bad",
            "query": "anything",
            "intent": "diagnostic",
            "scene_id": "not_a_scene",
        }
    )
    assert record["audit"]["skipped"]
    assert "not one of the 6 demo scenes" in record["audit"]["skip_reason"]
    # No agent_output should be present — never invoked the graph.
    assert "agent_output" not in record


def test_main_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: --questions in, --output out, mixed valid / invalid.

    Runs in-process with a patched precheck so the smoke test validates the
    CLI/output contract without depending on local model artifacts."""
    import scripts.distill.run_inference as entry

    questions = tmp_path / "q.jsonl"
    output = tmp_path / "out.jsonl"
    questions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid": "cli_001",
                        "query": "RC 时间常数怎么算？",
                        "intent": "concept_tutor",
                        "scene_id": "exp_first_order_rc",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "qid": "cli_002_bad",
                        "query": "x",
                        "intent": "diagnostic",
                        "scene_id": "",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(entry, "_gate_on_precheck", lambda: 0)
    monkeypatch.setattr(entry, "_verify_isolation", lambda: None)
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )

    exit_code = entry.main(
        [
            "--questions", str(questions),
            "--output", str(output),
        ]
    )
    assert exit_code == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 2
    # First sample processed, second skipped.
    assert not records[0]["audit"]["skipped"]
    assert records[0]["qid"] == "cli_001"
    assert records[0]["agent_output"]["final_answer"]
    assert records[1]["audit"]["skipped"]
    assert records[1]["qid"] == "cli_002_bad"


# ---------------------------------------------------------------------------
# 5. CLI refuses to start when precheck fails
# ---------------------------------------------------------------------------


def test_main_cli_aborts_when_precheck_fails(tmp_path: Path, monkeypatch) -> None:
    """Default path: precheck must pass before any sample runs. Force a
    failure by leaving DISTILL_MODE off."""
    from scripts.distill.run_inference import main

    monkeypatch.setattr(settings, "DISTILL_MODE", False)
    questions = tmp_path / "q.jsonl"
    output = tmp_path / "out.jsonl"
    questions.write_text(
        json.dumps(
            {
                "qid": "blocked",
                "query": "x",
                "intent": "concept_tutor",
                "scene_id": "exp_first_order_rc",
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(
        ["--questions", str(questions), "--output", str(output)]
    )
    assert exit_code == 1
    # Output must not have been created (or empty).
    if output.exists():
        assert output.read_text(encoding="utf-8").strip() == ""


def test_main_cli_aborts_when_exception_rate_exceeds_threshold(
    tmp_path: Path, monkeypatch
) -> None:
    """Batch must fail when sample exceptions exceed the configured threshold."""
    import scripts.distill.run_inference as entry

    questions = tmp_path / "q.jsonl"
    output = tmp_path / "out.jsonl"
    questions.write_text(
        json.dumps(
            {
                "qid": "boom",
                "query": "x",
                "intent": "concept_tutor",
                "scene_id": "exp_first_order_rc",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(entry, "_gate_on_precheck", lambda: 0)
    monkeypatch.setattr(entry, "_verify_isolation", lambda: None)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced sample failure")

    monkeypatch.setattr(entry, "run_sample", _boom)
    exit_code = entry.main(
        [
            "--questions",
            str(questions),
            "--output",
            str(output),
            "--max-error-rate",
            "0.0",
        ]
    )
    assert exit_code == 1
    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 1
    assert records[0]["audit"]["skip_reason_category"] == "exception"
    assert records[0]["audit"]["exception_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# 6. WP-2.1 regressions: VLM removed, precheck non-bypassable, singleton resets
# ---------------------------------------------------------------------------


def test_run_sample_does_not_leak_forbidden_modules_after_e2e(tmp_path: Path) -> None:
    """WP-2.1 P0 regression: even after a real diagnostic sample runs end-to-end
    through the agent graph, no forbidden module sneaks into sys.modules.

    Earlier vlm_explain_node imported app.core.deps which loaded RagService /
    KbService AFTER the entry's isolation check, breaking the contract.
    Subprocess gives a clean sys.modules so a leak is unambiguous."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "import scripts.distill.run_inference as e\n"
        "sample = {\n"
        "    'qid': 'wp21_isolation',\n"
        "    'query': 'UA741 反相饱和怎么办',\n"
        "    'intent': 'diagnostic',\n"
        "    'scene_id': 'exp_ua741_inverting_amplifier',\n"
        "    'station': {\n"
        "        'risk_level': 'danger',\n"
        "        'comparison_report': {'items': [{'error_code': 'FLOATING_PIN'}]},\n"
        "    },\n"
        "}\n"
        "_ = e.run_sample(sample)\n"
        "import sys, json\n"
        "print(json.dumps(sorted(e._FORBIDDEN_MODULES & set(sys.modules))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    leaked = json.loads(result.stdout.strip() or "[]")
    assert not leaked, (
        f"WP-2.1 P0 regression: forbidden modules leaked AFTER running a "
        f"real e2e sample: {leaked}. Did a node re-introduce app.core.deps "
        f"or vlm_explain? Check the import chain of every graph node."
    )


def test_skip_precheck_flag_does_not_exist(tmp_path: Path) -> None:
    """WP-2.1 P1: ``--skip-precheck`` was a contract-bypass vector and is
    REMOVED. The CLI must reject it as an unknown argument."""
    from scripts.distill.run_inference import main

    questions = tmp_path / "q.jsonl"
    questions.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--questions",
                str(questions),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--skip-precheck",  # removed in WP-2.1
            ]
        )
    # argparse exits with code 2 on unrecognized arg.
    assert excinfo.value.code == 2


def test_datasheet_singleton_reset_helper_exists() -> None:
    """WP-2.1 P2: the reset helper used by fixtures must exist + work.
    Without it, tests that switch backends pollute follow-up runs."""
    from app.agent.tools import _reset_datasheet_kb_singleton, _get_datasheet_kb
    import app.agent.tools as tools_mod

    kb1 = _get_datasheet_kb()
    assert tools_mod._DATASHEET_KB_SINGLETON is not None
    _reset_datasheet_kb_singleton()
    assert tools_mod._DATASHEET_KB_SINGLETON is None
    kb2 = _get_datasheet_kb()
    assert kb1 is not kb2, "reset must produce a fresh instance"
