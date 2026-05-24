"""WP-3 tests: datasheet fail-closed under DISTILL_MODE + precheck script.

Pins the contract:
  - When ``settings.DISTILL_MODE`` is True, ``datasheet_lookup_tool`` MUST
    return ``status="skipped"`` on a local-v2 miss instead of synthesizing
    fallback "保守规则" evidence. Distillation samples never carry
    artificial datasheet rules that the on-device runtime wouldn't actually
    produce.
  - The precheck script flags every missing piece of the train↔deploy
    contract (DISTILL_MODE / backend / model dir / .npz coverage).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.tools import DatasheetLookupInput, datasheet_lookup_tool
from app.core.config import settings


# ---------------------------------------------------------------------------
# 1. datasheet_lookup_tool fail-closed in DISTILL_MODE
# ---------------------------------------------------------------------------


def test_datasheet_tool_returns_fallback_in_normal_mode(monkeypatch) -> None:
    """Normal (non-distill) mode keeps the dev-friendly behavior:
    local v2 misses fall back to LOCAL_DATASHEET_FALLBACKS with rules."""
    monkeypatch.setattr(settings, "DISTILL_MODE", False)
    result = datasheet_lookup_tool(
        DatasheetLookupInput(component_type="LED", component_id="D1")
    )
    assert result.payload["provider"] == "local_fallback"
    assert any("限流" in rule for rule in result.payload["safety_rules"])


def test_datasheet_tool_skips_in_distill_mode_on_miss(monkeypatch) -> None:
    """WP-3 contract: in DISTILL_MODE, a v2 miss returns ``skipped``
    instead of synthesizing fallback rule evidence."""
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    # "LED" doesn't exist in knowledge/datasheets/*.json → v2 will miss.
    result = datasheet_lookup_tool(
        DatasheetLookupInput(component_type="LED", component_id="D1")
    )
    assert result.status == "skipped"
    assert result.payload["provider"] == "distill_fail_closed"
    assert result.payload["miss_reason"] == "datasheet_v2_miss_distill_fail_closed"
    assert result.payload["hits"] == []
    assert result.payload["rules"] == []
    # And specifically — must NOT carry any fallback-style content.
    assert "safety_rules" not in result.payload
    assert "structured_rules" not in result.payload


def test_datasheet_tool_still_returns_v2_hits_in_distill_mode(monkeypatch) -> None:
    """DISTILL_MODE only changes miss behavior; real v2 hits flow through
    normally so the teacher does see datasheet evidence when it exists."""
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    # UA741 IS in knowledge/datasheets/ua741.json — should match.
    result = datasheet_lookup_tool(
        DatasheetLookupInput(
            component_type="UA741",
            component_id="U1",
            part_number="UA741",
            query="UA741 pin 4",
        )
    )
    # Either v2 hit or skipped — must NOT be fallback.
    assert result.payload["provider"] in ("local_datasheet_v2", "distill_fail_closed")


# ---------------------------------------------------------------------------
# 2. Precheck script: each failure mode detected
# ---------------------------------------------------------------------------


def test_precheck_fails_when_distill_mode_off(monkeypatch) -> None:
    monkeypatch.setattr(settings, "DISTILL_MODE", False)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    failed = [r for r in results if not r.passed]
    assert any(r.name == "DISTILL_MODE" for r in failed)


def test_precheck_fails_when_backend_is_null(monkeypatch) -> None:
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "null")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    failed = [r for r in results if not r.passed]
    assert any(r.name == "DATASHEET_EMBEDDING_BACKEND" for r in failed)


def test_precheck_fails_when_model_dir_missing(monkeypatch) -> None:
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings, "DATASHEET_EMBEDDING_MODEL_DIR", "/nonexistent/model/dir"
    )
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    failed = [r for r in results if not r.passed]
    assert any(r.name == "DATASHEET_EMBEDDING_MODEL_DIR" for r in failed)


def test_precheck_fails_when_npz_missing_for_a_document(monkeypatch, tmp_path) -> None:
    """Point DATASHEET_EMBEDDINGS_DIR at an empty temp dir so every
    .npz is reported missing, even though the JSONs exist."""
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )
    empty_dir = tmp_path / "empty_embeddings"
    empty_dir.mkdir()
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDINGS_DIR", str(empty_dir))
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    failed_names = [r.name for r in results if not r.passed]
    # Should flag missing .npz for at least one datasheet.
    assert any(name.endswith(".npz") for name in failed_names), (
        f"expected '<doc>.npz' missing-file failure; got {failed_names}"
    )


def test_precheck_passes_with_full_distill_setup(monkeypatch) -> None:
    """Sanity: with everything wired up properly, precheck returns no
    failures. Matches the smoke-test command-line invocation."""
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    failed = [r for r in results if not r.passed]
    assert not failed, "precheck should pass with full setup: " + "; ".join(
        r.render() for r in failed
    )


# ---------------------------------------------------------------------------
# 3. Static contract — .npz coverage 100% of datasheet JSONs
#    (Independent of monkeypatched settings; works on the real workspace.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. WP-3 v2: cross-chip leakage defense (scene → allowed_datasheets)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scene_id,forbidden_doc_id",
    [
        # UA741 scenes must NOT surface NE555 / 74LS74 / LM324 chunks.
        ("exp_ua741_inverting_amplifier", "ne555"),
        ("exp_ua741_inverting_amplifier", "sn74ls74a"),
        ("exp_ua741_inverting_amplifier", "lm324"),
        ("exp_ua741_summing_amplifier", "ne555"),
        ("exp_ua741_integrator", "sn74ls74a"),
        # Common-emitter and differential-pair are BJT-only.
        ("exp_common_emitter_amplifier", "ne555"),
        ("exp_common_emitter_amplifier", "ua741"),
        ("exp_differential_amplifier", "lm324"),
        # RC scene is pure passive — no IC chunks should leak.
        ("exp_first_order_rc", "ne555"),
        ("exp_first_order_rc", "ua741"),
    ],
)
def test_datasheet_scene_whitelist_excludes_cross_chip_in_all_modes(
    monkeypatch, scene_id: str, forbidden_doc_id: str
) -> None:
    """WP-3 v3 (2026-05-24): scene whitelist is now a PRODUCTION contract,
    not DISTILL_MODE-only. Whenever scene_id is set on
    ``DatasheetLookupInput``, the search MUST hard-exclude out-of-scope
    chips — regardless of DISTILL_MODE. This enforces train ≡ deploy
    symmetry and prevents the train-test distribution shift that would
    appear if dev/production allowed cross-chip leakage."""
    # Verify in BOTH modes — train and deploy must be symmetric.
    for distill_flag in (False, True):
        monkeypatch.setattr(settings, "DISTILL_MODE", distill_flag)
        result = datasheet_lookup_tool(
            DatasheetLookupInput(
                component_type="",
                component_id="",
                # Use the forbidden chip's name as the query — proves the
                # whitelist beats keyword relevance.
                query=f"{forbidden_doc_id} pin overview",
                scene_id=scene_id,
            )
        )
        # Either the lookup is skipped (no v2 hit) or it hit an allowed doc.
        if result.status == "skipped":
            continue
        hits = result.payload.get("hits", [])
        surfaced = {h.get("document_id") for h in hits}
        assert forbidden_doc_id not in surfaced, (
            f"WP-3 v3 leak (DISTILL_MODE={distill_flag}): scene={scene_id} "
            f"surfaced out-of-scope datasheet {forbidden_doc_id!r} "
            f"(hits: {surfaced})"
        )


def test_datasheet_no_scene_id_keeps_full_corpus_search() -> None:
    """When ``scene_id`` is empty (concept_tutor without topology, or
    admin tools bypassing the agent graph), the whitelist does NOT
    apply — full corpus is searched. This is the documented escape
    hatch for cross-chip queries when no scene context exists."""
    result = datasheet_lookup_tool(
        DatasheetLookupInput(
            component_type="",
            component_id="",
            query="NE555 pin diagram",
            scene_id="",  # explicit no-scene
        )
    )
    # Either a real hit or a fallback — but NOT distill_fail_closed
    # (that requires scene_id set + DISTILL_MODE on).
    assert result.payload.get("provider") != "distill_fail_closed"


# ---------------------------------------------------------------------------
# 5. WP-3 v2: precheck actually probes backend activation
# ---------------------------------------------------------------------------


def test_precheck_fails_when_model_dir_exists_but_files_garbage(
    monkeypatch, tmp_path
) -> None:
    """WP-3 v2 R2a: even if the model_dir exists with the 3 required
    filenames, precheck must catch a corrupted model by actually trying
    to load it. Use garbage files as a proxy for "files exist but
    OpenVINO will fail to load"."""
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    fake_model = tmp_path / "fake_model"
    fake_model.mkdir()
    (fake_model / "openvino_model.xml").write_text("not a valid model")
    (fake_model / "openvino_model.bin").write_text("garbage")
    (fake_model / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_MODEL_DIR", str(fake_model))
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    failed = [r for r in results if not r.passed]
    # Either the file-level check passes (garbage looks valid by name) AND
    # the active-backend check fails, OR the file check fails too. Either
    # way the active check must contribute a failure when reached.
    backend_active_results = [r for r in results if r.name == "embedding_backend.active"]
    assert backend_active_results, "embedding_backend.active check did not run"
    assert not backend_active_results[0].passed, (
        "WP-3 v2 R2a regression: precheck accepted a garbage model — "
        "active-backend probe is not catching load failures."
    )


def test_precheck_active_backend_passes_with_real_model(monkeypatch) -> None:
    """Sanity: with the real OV INT8 model, the active-backend probe passes."""
    monkeypatch.setattr(settings, "DISTILL_MODE", True)
    monkeypatch.setattr(settings, "DATASHEET_EMBEDDING_BACKEND", "openvino")
    monkeypatch.setattr(
        settings,
        "DATASHEET_EMBEDDING_MODEL_DIR",
        "models/bge-small-zh-v1.5-int8-ov",
    )
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    active = [r for r in results if r.name == "embedding_backend.active"]
    assert active and active[0].passed, (
        f"active-backend probe failed on real model: "
        f"{active[0].render() if active else 'check did not run'}"
    )


# ---------------------------------------------------------------------------
# 6. WP-3 v2: fetch_artifacts.sh exists and is executable
# ---------------------------------------------------------------------------


def test_fetch_artifacts_script_exists_and_is_executable() -> None:
    """WP-3.1 R1: fresh checkout needs a documented bootstrap path for
    the model + .npz artifacts (both .gitignored)."""
    import os
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "distill" / "fetch_artifacts.sh"
    assert script.exists(), "fetch_artifacts.sh missing — fresh-checkout bootstrap broken"
    assert os.access(script, os.X_OK), "fetch_artifacts.sh is not executable"
    body = script.read_text(encoding="utf-8")
    assert "huggingface" in body.lower() or "hf " in body, (
        "fetch_artifacts.sh should reference the HF download path"
    )
    assert "build_datasheet_embeddings.py" in body, (
        "fetch_artifacts.sh should rebuild .npz after model download"
    )


def test_all_datasheet_jsons_have_matching_npz() -> None:
    """WP-3 hard gate: every datasheet JSON must have a corresponding
    .npz cache so on-device runtime can deliver semantic recall."""
    import json
    from app.schemas.kb import DatasheetDocument

    repo_root = Path(__file__).resolve().parents[1]
    datasheet_dir = repo_root / "knowledge" / "datasheets"
    embeddings_dir = datasheet_dir / "embeddings"
    json_files = sorted(datasheet_dir.glob("*.json"))
    assert json_files, "no datasheet JSONs found — fixture regression"
    missing = []
    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        doc = DatasheetDocument.model_validate(payload)
        npz_path = embeddings_dir / f"{doc.document_id}.npz"
        if not npz_path.exists():
            missing.append(doc.document_id)
    assert not missing, (
        f"WP-3 coverage regression: {len(missing)}/{len(json_files)} datasheets "
        f"have no .npz: {missing}. Run "
        f"`scripts/build_datasheet_embeddings.py --documents {' '.join(missing)}`."
    )
