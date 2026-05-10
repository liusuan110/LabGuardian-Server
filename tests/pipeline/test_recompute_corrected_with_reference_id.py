from __future__ import annotations

import pytest

from app.schemas.pipeline import CorrectedRecomputeRequest, ManualCorrectionPatch
from app.services.pipeline_service import PipelineService


def _build_components_wrong() -> list[dict]:
    """R1 两脚都在同一行（ROW_12_L），形成自短路，不匹配参考"""
    return [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "package_type": "axial_2pin",
            "symmetry_group": [["pin1", "pin2"]],
            "confidence": 1.0,
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "B12", "electrical_node_id": "ROW_12_L", "confidence": 1.0},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "C12", "electrical_node_id": "ROW_12_L", "confidence": 1.0},
            ],
        },
    ]


def _build_corrections_fix() -> list[ManualCorrectionPatch]:
    """把 pin2 从 C12 移到 F12，使 R1 跨到 ROW_12_R"""
    return [
        ManualCorrectionPatch(
            component_id="R1",
            pin_name="pin2",
            from_hole_id="C12",
            to_hole_id="F12",
            source="manual_drag",
        )
    ]


def _build_dummy_correction() -> list[ManualCorrectionPatch]:
    """一个不做任何实际改变的 dummy correction（用于绕过非空校验）"""
    return [
        ManualCorrectionPatch(
            component_id="R1",
            pin_name="pin1",
            from_hole_id="B12",
            to_hole_id="B12",
            source="manual_drag",
        )
    ]


def _build_components_correct() -> list[dict]:
    """R1 已经跨行，不需要修正"""
    return [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "package_type": "axial_2pin",
            "symmetry_group": [["pin1", "pin2"]],
            "confidence": 1.0,
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "B12", "electrical_node_id": "ROW_12_L", "confidence": 1.0},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "F12", "electrical_node_id": "ROW_12_R", "confidence": 1.0},
            ],
        },
    ]


class TestRecomputeCorrectedWithReferenceId:
    def test_recompute_corrected_loads_reference_by_id(self) -> None:
        service = PipelineService()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=_build_components_correct(),
            corrections=_build_dummy_correction(),
            reference_id="test_rc_v1",
        )
        result = service.recompute_corrected(request)

        s4 = result.stages[2]  # validate stage
        assert s4.data["comparison_report"]["summary"]["comparison_mode"] == "logical_graph"
        assert result.runtime_metadata["reference"]["source"] == "reference_id"
        assert result.runtime_metadata["reference"]["reference_id"] == "test_rc_v1"
        assert result.runtime_metadata["manual_corrections_applied"] is True

    def test_recompute_corrected_wrong_to_correct(self) -> None:
        """手动修正前电路自短路，修正后跨行正确"""
        service = PipelineService()

        # 修正前：自短路（直接用 run_validate 验证，因为 recompute_corrected 要求至少一条 correction）
        from app.pipeline.stages.s4_validate import run_validate
        topology_graph = {"nodes": [], "links": []}
        s4_wrong = run_validate(
            topology_graph=topology_graph,
            reference_circuit={
                "format": "logical_reference_v1",
                "reference_id": "test",
                "components": [
                    {"ref_id": "R1", "type": "Resistor", "pins": [{"pin": "pin1", "net": "N1"}, {"pin": "pin2", "net": "N2"}]}
                ],
                "nets": [{"net": "N1"}, {"net": "N2"}],
            },
            components=_build_components_wrong(),
        )
        assert s4_wrong["is_correct"] is False

        # 修正后：跨行正确
        request_fixed = CorrectedRecomputeRequest(
            station_id="S01",
            components=_build_components_wrong(),
            corrections=_build_corrections_fix(),
            reference_id="test_all_signal_v1",
        )
        result_fixed = service.recompute_corrected(request_fixed)
        s4_fixed = result_fixed.stages[2]
        assert s4_fixed.data["is_correct"] is True
        assert s4_fixed.data["similarity"] == 1.0

    def test_recompute_corrected_runtime_metadata(self) -> None:
        service = PipelineService()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=_build_components_correct(),
            corrections=_build_dummy_correction(),
            reference_id="test_all_signal_v1",
        )
        result = service.recompute_corrected(request)

        meta = result.runtime_metadata
        assert meta["manual_corrections_applied"] is True
        assert meta["reference"]["source"] == "reference_id"
        assert meta["reference"]["reference_id"] == "test_all_signal_v1"
        assert meta["reference"]["format"] == "logical_reference_v1"
        assert "name" in meta["reference"]

    def test_recompute_corrected_without_reference_id_preserves_old_behavior(self) -> None:
        service = PipelineService()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=_build_components_correct(),
            corrections=_build_dummy_correction(),
        )
        result = service.recompute_corrected(request)

        meta = result.runtime_metadata
        assert meta["reference"]["source"] == "none"
        s4 = result.stages[2]
        # 没有参考电路时，只做独立诊断
        assert s4.data["diagnosis"] != ""

    def test_recompute_corrected_invalid_reference_id_rejected(self) -> None:
        service = PipelineService()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=_build_components_correct(),
            corrections=_build_dummy_correction(),
            reference_id="../secret",
        )
        with pytest.raises(ValueError, match="非法 reference_id"):
            service.recompute_corrected(request)
