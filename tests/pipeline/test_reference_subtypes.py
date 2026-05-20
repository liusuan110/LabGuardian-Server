from __future__ import annotations

from app.pipeline.reference_subtypes import apply_reference_ic_subtypes
from app.schemas.pipeline import CorrectedRecomputeRequest, ManualCorrectionPatch
from app.services.pipeline_service import PipelineService


def _ic_component(*, subtype: str = "") -> dict:
    return {
        "component_id": "IC1",
        "component_type": "IC",
        "package_type": "dip8",
        "part_subtype": subtype,
        "pins": [
            {"pin_id": 1, "pin_name": "pin1", "hole_id": "F19"},
            {"pin_id": 2, "pin_name": "pin2", "hole_id": "F20"},
        ],
    }


def test_apply_reference_ic_subtypes_fills_single_reference_ic() -> None:
    components = [_ic_component()]
    reference = {
        "components": [
            {"ref_id": "U1", "type": "IC", "subtype": "UA741", "pins": []},
        ],
    }

    applied = apply_reference_ic_subtypes(components, reference)

    assert components[0]["part_subtype"] == "UA741"
    assert applied == [{
        "component_id": "IC1",
        "part_subtype": "UA741",
        "source": "reference_circuit",
        "matched_by": "single_reference_ic",
    }]


def test_apply_reference_ic_subtypes_does_not_override_user_subtype() -> None:
    components = [_ic_component(subtype="LM358")]
    reference = {
        "components": [
            {"ref_id": "U1", "type": "IC", "subtype": "UA741", "pins": []},
        ],
    }

    applied = apply_reference_ic_subtypes(components, reference)

    assert components[0]["part_subtype"] == "LM358"
    assert applied == []


def test_recompute_corrected_backfills_ua741_subtype_into_netlist_v2() -> None:
    service = PipelineService()
    request = CorrectedRecomputeRequest(
        station_id="S01",
        components=[_ic_component()],
        corrections=[
            ManualCorrectionPatch(
                component_id="IC1",
                pin_name="pin1",
                from_hole_id="F19",
                to_hole_id="F19",
                source="manual_drag",
            ),
        ],
        reference_id="ua741_inverting_amp_gain10_v1",
    )

    result = service.recompute_corrected(request)
    topology = next(s for s in result.stages if s.stage.value == "topology").data
    ic = next(
        c for c in topology["netlist_v2"]["components"]
        if c["component_id"] == "IC1"
    )

    assert ic["part_subtype"] == "UA741"
    assert result.runtime_metadata["reference_ic_subtypes_applied"] == [{
        "component_id": "IC1",
        "part_subtype": "UA741",
        "source": "reference_circuit",
        "matched_by": "single_reference_ic",
    }]
