from __future__ import annotations

import pytest

from app.domain.logical_reference import current_netlist_v2_to_graph, normalize_net_role
from app.domain.graph_compare import compare_logical_graphs
from app.pipeline.net_roles import apply_net_role_assignments
from app.schemas.pipeline import (
    CorrectedRecomputeRequest,
    ManualCorrectionPatch,
    ManualNetRoleAssignment,
    PinSelector,
    PortAnnotation,
)
from app.services.pipeline_service import PipelineService


def _build_components_for_role_test() -> list[dict]:
    """构建一个简单 RC 电路的元件，用于角色测试。

    R1 跨接在 ROW_1_L 和 ROW_3_L
    C1 一端接 ROW_3_L，另一端接 PWR_MINUS(GND 轨)
    """
    return [
        {
            "component_id": "R1",
            "component_type": "Resistor",
            "package_type": "axial_2pin",
            "symmetry_group": [["pin1", "pin2"]],
            "confidence": 1.0,
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1", "electrical_node_id": "ROW_1_L", "confidence": 1.0},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3", "electrical_node_id": "ROW_3_L", "confidence": 1.0},
            ],
        },
        {
            "component_id": "C1",
            "component_type": "CapacitorCeramic",
            "package_type": "axial_2pin",
            "symmetry_group": [["pin1", "pin2"]],
            "confidence": 1.0,
            "pins": [
                {"pin_id": 1, "pin_name": "pin1", "hole_id": "B3", "electrical_node_id": "ROW_3_L", "confidence": 1.0},
                {"pin_id": 2, "pin_name": "pin2", "hole_id": "PWR_MINUS", "electrical_node_id": "PWR_MINUS", "confidence": 1.0},
            ],
        },
    ]


def _build_dummy_correction() -> list[ManualCorrectionPatch]:
    return [
        ManualCorrectionPatch(
            component_id="R1",
            pin_name="pin1",
            from_hole_id="A1",
            to_hole_id="A1",
            source="manual_drag",
        )
    ]


def _build_reference_logical_v1() -> dict:
    return {
        "format": "logical_reference_v1",
        "reference_id": "rc_role_test",
        "name": "RC 角色测试电路",
        "components": [
            {
                "ref_id": "R1",
                "type": "Resistor",
                "pins": [{"pin": "pin1", "net": "VIN"}, {"pin": "pin2", "net": "VC"}],
            },
            {
                "ref_id": "C1",
                "type": "CapacitorCeramic",
                "pins": [{"pin": "pin1", "net": "VC"}, {"pin": "pin2", "net": "GND"}],
            },
        ],
        "nets": [
            {"net": "VIN", "role": "input"},
            {"net": "VC", "role": "signal"},
            {"net": "GND", "role": "ground"},
        ],
    }


class TestManualNetRoleAssignmentSchema:
    def test_manual_net_role_assignment_creation(self) -> None:
        assignment = ManualNetRoleAssignment(role="VIN", component_id="R1", pin_name="pin1")
        assert assignment.role == "VIN"
        assert assignment.component_id == "R1"
        assert assignment.pin_name == "pin1"
        assert assignment.source == "manual_netlist_select"

    def test_corrected_recompute_request_accepts_net_role_assignments(self) -> None:
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=[{"component_id": "R1"}],
            corrections=[ManualCorrectionPatch(component_id="R1", pin_name="pin1", from_hole_id="A1", to_hole_id="A2")],
            net_role_assignments=[
                ManualNetRoleAssignment(role="VIN", component_id="R1", pin_name="pin1"),
                ManualNetRoleAssignment(role="GND", hole_id="PWR_MINUS"),
            ],
        )
        assert len(request.net_role_assignments) == 2
        assert request.net_role_assignments[0].role == "VIN"

    def test_port_annotation_schema_uses_pin_selector(self) -> None:
        annotation = PortAnnotation(
            role="input",
            target=PinSelector(component_id="R1", pin_name="pin1"),
        )
        assert annotation.role == "input"
        assert annotation.target.component_id == "R1"
        assert annotation.label is None
        assert annotation.source == "port_annotation"

    def test_corrected_recompute_request_accepts_port_annotations(self) -> None:
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=[{"component_id": "R1"}],
            port_annotations=[
                PortAnnotation(
                    role="input",
                    target=PinSelector(component_id="R1", pin_name="pin1"),
                    label="UI1",
                ),
            ],
        )
        assert len(request.port_annotations) == 1
        assert request.port_annotations[0].label == "UI1"


class TestCurrentNetlistV2ToGraphRoles:
    def test_netlist_v2_graph_reads_input_output_power_ground(self) -> None:
        netlist = {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_001"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_002"},
                    ],
                },
                {
                    "component_id": "C1",
                    "component_type": "CapacitorCeramic",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_002"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_003"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "NET_001", "role": "input"},
                {"electrical_net_id": "NET_002", "manual_role": "output"},
                {"electrical_net_id": "NET_003", "power_role": "GND"},
                {"electrical_net_id": "NET_004", "power_role": "VCC"},
            ],
        }
        graph = current_netlist_v2_to_graph(netlist)
        assert graph.nodes["cur_net:NET_001"]["role"] == "input"
        assert graph.nodes["cur_net:NET_002"]["role"] == "output"
        assert graph.nodes["cur_net:NET_003"]["role"] == "ground"
        assert graph.nodes["cur_net:NET_004"]["role"] == "power"

    def test_netlist_v2_graph_reads_role_label(self) -> None:
        """current_netlist_v2_to_graph 应通过 role_label 识别 VIN/VOUT/VCC/GND。"""
        netlist = {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_001"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_002"},
                        {"pin_name": "pin3", "electrical_net_id": "NET_003"},
                        {"pin_name": "pin4", "electrical_net_id": "NET_004"},
                    ],
                },
            ],
            "nets": [
                {"electrical_net_id": "NET_001", "role_label": "VIN"},
                {"electrical_net_id": "NET_002", "role_label": "VOUT"},
                {"electrical_net_id": "NET_003", "role_label": "VCC"},
                {"electrical_net_id": "NET_004", "role_label": "GND"},
            ],
        }
        graph = current_netlist_v2_to_graph(netlist)
        assert graph.nodes["cur_net:NET_001"]["role"] == "input"
        assert graph.nodes["cur_net:NET_002"]["role"] == "output"
        assert graph.nodes["cur_net:NET_003"]["role"] == "power"
        assert graph.nodes["cur_net:NET_004"]["role"] == "ground"
        assert graph.nodes["cur_net:NET_001"]["role_label"] == "VIN"


class TestApplyNetRoleAssignments:
    def test_apply_net_role_assignments_by_electrical_net_id(self) -> None:
        netlist = {
            "components": [
                {
                    "component_id": "R1",
                    "pins": [{"pin_name": "pin1", "electrical_net_id": "NET_001"}],
                }
            ],
            "nets": [{"electrical_net_id": "NET_001", "member_hole_ids": ["A1"]}],
        }

        warnings, applied = apply_net_role_assignments(
            netlist,
            [
                {
                    "role": "VIN",
                    "role_label": "VIN",
                    "electrical_net_id": "NET_001",
                    "source": "manual_netlist_select",
                }
            ],
        )

        assert warnings == []
        assert applied[0]["electrical_net_id"] == "NET_001"
        net = netlist["nets"][0]
        assert net["role"] == "input"
        assert net["manual_role"] == "input"
        assert net["role_label"] == "VIN"

    def test_apply_port_annotation_by_component_pin(self) -> None:
        netlist = {
            "components": [
                {
                    "component_id": "R1",
                    "pins": [{"pin_name": "pin1", "electrical_net_id": "NET_001"}],
                }
            ],
            "nets": [{"electrical_net_id": "NET_001", "member_hole_ids": ["A1"]}],
        }

        warnings, applied = apply_net_role_assignments(
            netlist,
            [],
            port_annotations=[
                {
                    "role": "input",
                    "target": {"component_id": "R1", "pin_name": "pin1"},
                    "label": "UI1",
                }
            ],
        )

        assert warnings == []
        assert applied == [
            {
                "role": "input",
                "role_label": "UI1",
                "electrical_net_id": "NET_001",
                "source": "port_annotation",
                "resolved_by": "component_pin",
                "component_id": "R1",
                "pin_name": "pin1",
            }
        ]
        net = netlist["nets"][0]
        assert net["role"] == "input"
        assert net["manual_role"] == "input"
        assert net["role_label"] == "UI1"
        assert net["role_source"] == "port_annotation"

    def test_apply_port_annotation_without_label_keeps_label_blank(self) -> None:
        netlist = {
            "components": [
                {
                    "component_id": "J1",
                    "pins": [{"pin_name": "pin1", "electrical_net_id": "NET_001"}],
                }
            ],
            "nets": [{"electrical_net_id": "NET_001", "member_hole_ids": ["A1"]}],
        }

        warnings, applied = apply_net_role_assignments(
            netlist,
            [],
            port_annotations=[
                {
                    "role": "output",
                    "target": {"component_id": "J1", "pin_name": "pin1"},
                }
            ],
        )

        assert warnings == []
        assert applied[0]["role"] == "output"
        assert applied[0]["role_label"] == ""
        assert netlist["nets"][0]["role"] == "output"
        assert netlist["nets"][0]["role_label"] == ""


class TestGraphCompareRoleMismatch:
    def test_input_node_mismatch_detected(self) -> None:
        """参考 VIN=input，当前把 input 错标到别的 net，应检测到 INPUT_NODE_MISMATCH。"""
        import networkx as nx

        ref_graph = nx.Graph()
        ref_graph.add_node("ref_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
        ref_graph.add_node("ref_net:VIN", kind="net", role="input", source_id="VIN")
        ref_graph.add_node("ref_net:VC", kind="net", role="signal", source_id="VC")
        ref_graph.add_edge("ref_comp:R1", "ref_net:VIN", pin="pin1")
        ref_graph.add_edge("ref_comp:R1", "ref_net:VC", pin="pin2")

        # 当前电路：元件连接结构相同，但 NET_000 被错误标记为 signal 而非 input
        cur_graph = nx.Graph()
        cur_graph.add_node("cur_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
        cur_graph.add_node("cur_net:NET_000", kind="net", role="signal", source_id="NET_000")
        cur_graph.add_node("cur_net:NET_001", kind="net", role="signal", source_id="NET_001")
        cur_graph.add_edge("cur_comp:R1", "cur_net:NET_000", pin="pin1")
        cur_graph.add_edge("cur_comp:R1", "cur_net:NET_001", pin="pin2")

        ref_payload = {
            "format": "logical_reference_v1",
            "reference_id": "test",
            "components": [
                {"ref_id": "R1", "type": "Resistor", "pins": [{"pin": "pin1", "net": "VIN"}, {"pin": "pin2", "net": "VC"}]}
            ],
            "nets": [{"net": "VIN", "role": "input"}, {"net": "VC", "role": "signal"}],
        }
        cur_netlist_v2 = {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_000"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_001"},
                    ],
                }
            ],
            "nets": [
                {"electrical_net_id": "NET_000", "role": "signal"},
                {"electrical_net_id": "NET_001", "role": "signal"},
            ],
        }

        result = compare_logical_graphs(
            ref_graph, cur_graph, ref_payload=ref_payload, cur_netlist_v2=cur_netlist_v2
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        codes = {i["error_code"] for i in items}
        assert "INPUT_NODE_MISMATCH" in codes or "WRONG_CONNECTION" in codes

    def test_ground_node_mismatch_detected(self) -> None:
        """参考 GND=ground，当前 net 标记为 power，应检测到 GROUND_NODE_MISMATCH 或 ROLE_MISMATCH。"""
        import networkx as nx

        ref_graph = nx.Graph()
        ref_graph.add_node("ref_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
        ref_graph.add_node("ref_net:GND", kind="net", role="ground", source_id="GND")
        ref_graph.add_node("ref_net:VC", kind="net", role="signal", source_id="VC")
        ref_graph.add_edge("ref_comp:R1", "ref_net:GND", pin="pin1")
        ref_graph.add_edge("ref_comp:R1", "ref_net:VC", pin="pin2")

        cur_graph = nx.Graph()
        cur_graph.add_node("cur_comp:R1", kind="comp", ctype="Resistor", source_id="R1")
        cur_graph.add_node("cur_net:NET_000", kind="net", role="power", source_id="NET_000")
        cur_graph.add_node("cur_net:NET_001", kind="net", role="signal", source_id="NET_001")
        cur_graph.add_edge("cur_comp:R1", "cur_net:NET_000", pin="pin1")
        cur_graph.add_edge("cur_comp:R1", "cur_net:NET_001", pin="pin2")

        ref_payload = {
            "format": "logical_reference_v1",
            "reference_id": "test",
            "components": [
                {"ref_id": "R1", "type": "Resistor", "pins": [{"pin": "pin1", "net": "GND"}, {"pin": "pin2", "net": "VC"}]}
            ],
            "nets": [{"net": "GND", "role": "ground"}, {"net": "VC", "role": "signal"}],
        }
        cur_netlist_v2 = {
            "components": [
                {
                    "component_id": "R1",
                    "component_type": "Resistor",
                    "pins": [
                        {"pin_name": "pin1", "electrical_net_id": "NET_000"},
                        {"pin_name": "pin2", "electrical_net_id": "NET_001"},
                    ],
                }
            ],
            "nets": [
                {"electrical_net_id": "NET_000", "role": "power"},
                {"electrical_net_id": "NET_001", "role": "signal"},
            ],
        }

        result = compare_logical_graphs(
            ref_graph, cur_graph, ref_payload=ref_payload, cur_netlist_v2=cur_netlist_v2
        )
        assert result["logic_correct"] is False
        items = result["report"]["items"]
        codes = {i["error_code"] for i in items}
        assert "GROUND_NODE_MISMATCH" in codes or "ROLE_MISMATCH" in codes or "WRONG_CONNECTION" in codes


class TestRecomputeCorrectedWithNetRoles:
    def test_recompute_corrected_applies_net_role_by_component_pin(self) -> None:
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=_build_dummy_correction(),
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="GND", component_id="C1", pin_name="pin2"),
            ],
        )
        result = service.recompute_corrected(request)
        meta = result.runtime_metadata
        assert "manual_roles_applied" in meta
        applied = meta["manual_roles_applied"]
        assert any(a["role"] == "ground" for a in applied)

    def test_recompute_corrected_applies_net_role_by_hole_id(self) -> None:
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=_build_dummy_correction(),
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="VIN", hole_id="A1"),
            ],
        )
        result = service.recompute_corrected(request)
        meta = result.runtime_metadata
        applied = meta["manual_roles_applied"]
        assert any(a["role"] == "input" for a in applied)

    def test_recompute_corrected_applies_net_role_by_electrical_net_id(self) -> None:
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=_build_dummy_correction(),
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="VCC", electrical_net_id="NET_001"),
            ],
        )
        result = service.recompute_corrected(request)
        meta = result.runtime_metadata
        applied = meta["manual_roles_applied"]
        assert any(a["role"] == "power" for a in applied)

    def test_recompute_corrected_role_mismatch_reported(self) -> None:
        """指定 VIN 到 GND 所在的 net，应触发 input/ground 角色不匹配。"""
        service = PipelineService()
        components = _build_components_for_role_test()
        # 把 GND 轨（PWR_MINUS/C1.pin2）错误指定为 VIN(input)
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=_build_dummy_correction(),
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="VIN", component_id="C1", pin_name="pin2"),
            ],
        )
        result = service.recompute_corrected(request)
        s4 = result.stages[2]
        items = s4.data.get("comparison_report", {}).get("items", [])
        codes = {i["error_code"] for i in items}
        # 参考 VIN=input，当前 C1.pin2 实际接到 GND 轨，若被强制标为 input，
        # 则与参考的 ground 角色冲突，应报告角色不匹配
        assert "INPUT_NODE_MISMATCH" in codes or "GROUND_NODE_MISMATCH" in codes or "WRONG_CONNECTION" in codes

    def test_recompute_corrected_correct_roles_full_match(self) -> None:
        """正确指定 VIN 和 GND 角色后，logical graph 应完全匹配。"""
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=_build_dummy_correction(),
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="VIN", component_id="R1", pin_name="pin1"),
                ManualNetRoleAssignment(role="GND", component_id="C1", pin_name="pin2"),
            ],
        )
        result = service.recompute_corrected(request)
        s4 = result.stages[2]
        assert s4.data["is_correct"] is True
        assert s4.data["similarity"] == 1.0

    def test_recompute_corrected_only_net_roles_no_corrections(self) -> None:
        """只提交 net_role_assignments、不提交 corrections 时也能正常重算。"""
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=[],
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="VIN", component_id="R1", pin_name="pin1"),
                ManualNetRoleAssignment(role="GND", component_id="C1", pin_name="pin2"),
            ],
        )
        result = service.recompute_corrected(request)
        s4 = result.stages[2]
        assert s4.data["is_correct"] is True
        assert s4.data["similarity"] == 1.0
        meta = result.runtime_metadata
        assert "manual_roles_applied" in meta
        assert len(meta["manual_roles_applied"]) == 2

    def test_recompute_corrected_only_port_annotations_no_corrections(self) -> None:
        """只提交 port_annotations、不提交 corrections 时也能正常重算。"""
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=[],
            reference_circuit=_build_reference_logical_v1(),
            port_annotations=[
                PortAnnotation(
                    role="input",
                    target=PinSelector(component_id="R1", pin_name="pin1"),
                ),
            ],
        )
        result = service.recompute_corrected(request)
        meta = result.runtime_metadata
        assert meta["port_annotations_applied"][0]["role"] == "input"
        assert meta["manual_roles_applied"][0]["source"] == "port_annotation"

    def test_recompute_corrected_empty_hole_warning(self) -> None:
        """点击未连接的空孔时，应返回 ROLE_TARGET_NOT_CONNECTED warning，不崩溃。"""
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=[],
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                # A99 是一个空孔，不在任何 net 中
                ManualNetRoleAssignment(role="VIN", hole_id="A99", x_image=100.0, y_image=200.0),
            ],
        )
        result = service.recompute_corrected(request)
        meta = result.runtime_metadata
        assert "manual_role_warnings" in meta
        warnings = meta["manual_role_warnings"]
        assert any(w["warning_code"] == "ROLE_TARGET_NOT_CONNECTED" for w in warnings)
        # 没有成功应用的角色
        assert len(meta.get("manual_roles_applied", [])) == 0

    def test_recompute_corrected_role_label_written_to_netlist(self) -> None:
        """手动角色应写入 netlist_v2 的 role、manual_role、role_label、role_source。"""
        service = PipelineService()
        components = _build_components_for_role_test()
        request = CorrectedRecomputeRequest(
            station_id="S01",
            components=components,
            corrections=[],
            reference_circuit=_build_reference_logical_v1(),
            net_role_assignments=[
                ManualNetRoleAssignment(role="VIN", hole_id="A1"),
            ],
        )
        result = service.recompute_corrected(request)
        # 从 topology stage 获取 netlist_v2
        topology_stage = next(s for s in result.stages if s.stage.value == "topology")
        netlist_v2 = topology_stage.data.get("netlist_v2", {})
        vin_net = next(
            (n for n in netlist_v2.get("nets", []) if n.get("role_label") == "VIN"),
            None,
        )
        assert vin_net is not None
        assert vin_net["role"] == "input"
        assert vin_net["manual_role"] == "input"
        assert vin_net["role_source"] == "manual_netlist_select"


class TestNormalizeNetRole:
    def test_vin_maps_to_input(self) -> None:
        assert normalize_net_role("VIN") == "input"

    def test_vout_maps_to_output(self) -> None:
        assert normalize_net_role("VOUT") == "output"

    def test_vcc_maps_to_power(self) -> None:
        assert normalize_net_role("VCC") == "power"

    def test_gnd_maps_to_ground(self) -> None:
        assert normalize_net_role("GND") == "ground"
