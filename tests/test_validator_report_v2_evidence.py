from __future__ import annotations

import json
from pathlib import Path

from app.agent.answering import build_diagnostic_evidence
from app.agent.context_pack import build_context_pack
from app.agent.evidence import build_runtime_evidence_from_station
from app.domain.validator import CircuitValidator
from app.pipeline.topology_input import build_analyzer_from_components

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "validator_error_codes"


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _reference_resistor_components() -> list[dict]:
    return [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "package_type": "axial_2pin",
            "symmetry_group": [["pin1", "pin2"]],
            "pins": [
                {
                    "pin_id": 1,
                    "pin_name": "pin1",
                    "hole_id": "B12",
                    "electrical_node_id": "ROW_12_L",
                    "confidence": 1.0,
                },
                {
                    "pin_id": 2,
                    "pin_name": "pin2",
                    "hole_id": "F12",
                    "electrical_node_id": "ROW_12_R",
                    "confidence": 1.0,
                },
            ],
        }
    ]


def _validator_with_reference() -> CircuitValidator:
    reference_analyzer, _normalized = build_analyzer_from_components(
        _reference_resistor_components()
    )
    validator = CircuitValidator()
    validator.set_reference(reference_analyzer)
    return validator


def test_validator_report_v2_exposes_locator_fields_and_evidence_objects() -> None:
    mapped_components = _load_json(FIXTURE_DIR / "mapped_node_mismatch.json")
    mapped_components[0]["bbox"] = [100, 100, 180, 140]
    analyzer, _normalized = build_analyzer_from_components(
        mapped_components
    )
    validator = _validator_with_reference()

    result = validator.compare(analyzer)
    item = next(
        item
        for item in result["report"]["items"]
        if item["error_code"] == "NODE_MISMATCH"
    )

    assert item["current_component_id"] == "R1"
    assert item["current_hole_id"] == ["B13", "F12"]
    assert item["current_node_id"] == ["ROW_12_R", "ROW_13_L"]
    assert item["target_node_id"] == ["ROW_12_L", "ROW_12_R"]
    assert item["current_observation_refs"]

    evidence_kinds = {ref["kind"] for ref in item["evidence_refs"]}
    assert "component_bbox_ref" in evidence_kinds
    assert "pin_keypoint_ref" in evidence_kinds
    assert "hole_candidate_ref" in evidence_kinds
    assert "node_trace_ref" in evidence_kinds
    assert "validator_rule_ref" in evidence_kinds

    target_kinds = {target["kind"] for target in item["highlight_targets"]}
    assert "component_bbox_ref" in target_kinds
    assert "pin_keypoint_ref" in target_kinds
    assert "hole_candidate_ref" in target_kinds
    assert result["report"]["highlight_protocol"]["version"] == "labguardian_highlight_v1"
    assert result["report"]["summary"]["highlight_target_count"] >= 3


def test_agent_evidence_exposes_frontend_highlight_protocol() -> None:
    mapped_components = _load_json(FIXTURE_DIR / "mapped_node_mismatch.json")
    mapped_components[0]["bbox"] = [100, 100, 180, 140]
    analyzer, _normalized = build_analyzer_from_components(mapped_components)
    validator = _validator_with_reference()
    comparison_report = validator.compare(analyzer)["report"]
    evidence = build_runtime_evidence_from_station(
        station_id="S01",
        station={
            "risk_level": "warning",
            "comparison_report": comparison_report,
        },
    )
    diagnostic_evidence = build_diagnostic_evidence(
        evidence=evidence,
        context_pack=build_context_pack(evidence),
        tool_results=[],
        verification_passed=True,
        verification_issues=[],
    )

    highlight = next(
        item for item in diagnostic_evidence if item.evidence_type == "highlight_protocol"
    )
    highlight_kinds = {target["kind"] for target in highlight.payload["targets"]}
    assert {"component_bbox_ref", "pin_keypoint_ref", "hole_candidate_ref"} <= highlight_kinds
