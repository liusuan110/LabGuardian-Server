from __future__ import annotations

import json
from pathlib import Path

from app.domain.logical_reference import (
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
)
from app.domain.graph_compare import compare_logical_graphs
from app.pipeline.topology_input import build_analyzer_from_components
from app.pipeline.stages.s4_validate import run_validate
from app.services.reference_service import ReferenceService

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "references"


def _build_correct_led_components() -> list[dict]:
    """R1 at B12-F12, LED1 at F12-B14 — correct topology for basic_series_resistor_v1"""
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
        {
            "component_id": "LED1",
            "component_type": "LED",
            "package_type": "5mm",
            "polarity": "forward",
            "symmetry_group": [],
            "confidence": 1.0,
            "pins": [
                {"pin_id": 1, "pin_name": "anode", "hole_id": "F12", "electrical_node_id": "ROW_12_R", "confidence": 1.0},
                {"pin_id": 2, "pin_name": "cathode", "hole_id": "B14", "electrical_node_id": "ROW_14_L", "confidence": 1.0},
            ],
        },
    ]


def _build_missing_led_components() -> list[dict]:
    """Only R1 — missing LED"""
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


def _build_wrong_connection_components() -> list[dict]:
    """R1 and LED1 in parallel instead of series"""
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
        {
            "component_id": "LED1",
            "component_type": "LED",
            "package_type": "5mm",
            "polarity": "forward",
            "symmetry_group": [],
            "confidence": 1.0,
            "pins": [
                {"pin_id": 1, "pin_name": "anode", "hole_id": "B12", "electrical_node_id": "ROW_12_L", "confidence": 1.0},
                {"pin_id": 2, "pin_name": "cathode", "hole_id": "F12", "electrical_node_id": "ROW_12_R", "confidence": 1.0},
            ],
        },
    ]


class TestReferenceServiceLoad:
    def test_load_basic_series_resistor(self) -> None:
        svc = ReferenceService()
        payload = svc.load_reference("basic_series_resistor_v1")
        assert payload["reference_id"] == "basic_series_resistor_v1"
        assert payload["format"] == "logical_reference_v1"
        assert len(payload["components"]) == 2
        assert len(payload["nets"]) == 3


class TestLogicalReferenceToGraph:
    def test_basic_series_resistor_graph(self) -> None:
        svc = ReferenceService()
        payload = svc.load_reference("basic_series_resistor_v1")
        graph = logical_reference_to_graph(payload)
        assert graph.number_of_nodes() == 5  # 2 comp + 3 net
        assert graph.number_of_edges() == 4
        comp_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "comp"]
        assert len(comp_nodes) == 2


class TestEndToEndS4Validate:
    def test_s4_validate_with_logical_reference_correct(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        ref_payload = svc.load_reference("test_all_signal_v1")

        components = _build_correct_led_components()
        analyzer, _ = build_analyzer_from_components(components)
        topology_graph = analyzer.to_node_link_data()
        netlist_v2 = analyzer.export_netlist_v2()

        result = run_validate(
            topology_graph=topology_graph,
            reference_circuit=ref_payload,
            components=components,
            current_netlist_v2=netlist_v2,
        )

        assert result["is_correct"] is True
        assert result["similarity"] == 1.0
        report = result.get("comparison_report", {})
        assert report.get("summary", {}).get("comparison_mode") == "logical_graph"

    def test_s4_validate_infers_missing_reference_roles_when_topology_matches(self) -> None:
        svc = ReferenceService()
        ref_payload = svc.load_reference("basic_series_resistor_v1")

        components = _build_correct_led_components()
        analyzer, _ = build_analyzer_from_components(components)
        topology_graph = analyzer.to_node_link_data()
        netlist_v2 = analyzer.export_netlist_v2()

        result = run_validate(
            topology_graph=topology_graph,
            reference_circuit=ref_payload,
            components=components,
            current_netlist_v2=netlist_v2,
        )

        assert result["is_correct"] is True
        report = result.get("comparison_report", {})
        summary = report.get("summary", {})
        assert summary.get("role_inference_applied") is True
        inferred = summary.get("inferred_net_roles", [])
        assert {item["role_label"] for item in inferred} >= {"VCC", "GND"}

    def test_s4_validate_does_not_override_explicit_wrong_roles(self) -> None:
        svc = ReferenceService()
        ref_payload = svc.load_reference("basic_series_resistor_v1")

        components = _build_correct_led_components()
        analyzer, _ = build_analyzer_from_components(components)
        topology_graph = analyzer.to_node_link_data()
        netlist_v2 = analyzer.export_netlist_v2()

        for net in netlist_v2.get("nets", []):
            member_holes = set(net.get("member_hole_ids", []))
            if "B12" in member_holes:
                net["role"] = "ground"
                net["role_label"] = "GND"
            elif "B14" in member_holes:
                net["role"] = "power"
                net["role_label"] = "VCC"

        result = run_validate(
            topology_graph=topology_graph,
            reference_circuit=ref_payload,
            components=components,
            current_netlist_v2=netlist_v2,
        )

        assert result["is_correct"] is False
        summary = result.get("comparison_report", {}).get("summary", {})
        assert summary.get("role_inference_applied") is not True
        codes = {item.get("error_code") for item in result.get("comparison_report", {}).get("items", [])}
        assert codes & {
            "POWER_NODE_MISMATCH",
            "GROUND_NODE_MISMATCH",
            "ROLE_LABEL_MISMATCH",
            "WRONG_CONNECTION",
        }

    def test_s4_validate_with_logical_reference_missing_component(self) -> None:
        svc = ReferenceService()
        ref_payload = svc.load_reference("basic_series_resistor_v1")

        components = _build_missing_led_components()
        analyzer, _ = build_analyzer_from_components(components)
        topology_graph = analyzer.to_node_link_data()
        netlist_v2 = analyzer.export_netlist_v2()

        result = run_validate(
            topology_graph=topology_graph,
            reference_circuit=ref_payload,
            components=components,
            current_netlist_v2=netlist_v2,
        )

        assert result["is_correct"] is False
        report = result.get("comparison_report", {})
        items = report.get("items", [])
        assert any(i["error_code"] == "COMPONENT_MISSING" for i in items)

    def test_s4_validate_with_logical_reference_wrong_connection(self) -> None:
        svc = ReferenceService()
        ref_payload = svc.load_reference("basic_series_resistor_v1")

        components = _build_wrong_connection_components()
        analyzer, _ = build_analyzer_from_components(components)
        topology_graph = analyzer.to_node_link_data()
        netlist_v2 = analyzer.export_netlist_v2()

        result = run_validate(
            topology_graph=topology_graph,
            reference_circuit=ref_payload,
            components=components,
            current_netlist_v2=netlist_v2,
        )

        assert result["is_correct"] is False
        report = result.get("comparison_report", {})
        items = report.get("items", [])
        assert any(i["error_code"] == "WRONG_CONNECTION" for i in items)

    def test_no_hole_mismatch_in_logical_comparison(self) -> None:
        svc = ReferenceService(reference_dir=FIXTURE_DIR)
        ref_payload = svc.load_reference("test_all_signal_v1")

        components = _build_correct_led_components()
        analyzer, _ = build_analyzer_from_components(components)
        topology_graph = analyzer.to_node_link_data()
        netlist_v2 = analyzer.export_netlist_v2()

        result = run_validate(
            topology_graph=topology_graph,
            reference_circuit=ref_payload,
            components=components,
            current_netlist_v2=netlist_v2,
        )

        report = result.get("comparison_report", {})
        assert report.get("hole_errors", []) == []
        items = report.get("items", [])
        assert not any(i.get("error_code") == "HOLE_MISMATCH" for i in items)


class TestPipelineServiceResolveReference:
    def test_resolve_reference_by_id(self) -> None:
        from app.schemas.pipeline import PipelineRequest
        from app.services.pipeline_service import PipelineService

        request = PipelineRequest(
            station_id="S01",
            images_b64=["dummy"],
            reference_id="basic_series_resistor_v1",
        )
        service = PipelineService()
        ref_circuit, ref_meta = service._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )

        assert ref_circuit is not None
        assert ref_circuit.get("reference_id") == "basic_series_resistor_v1"
        assert ref_meta["source"] == "reference_id"
        assert ref_meta["reference_id"] == "basic_series_resistor_v1"

    def test_resolve_reference_inline_payload(self) -> None:
        from app.schemas.pipeline import PipelineRequest
        from app.services.pipeline_service import PipelineService

        inline = {"format": "logical_reference_v1", "reference_id": "inline"}
        request = PipelineRequest(
            station_id="S01",
            images_b64=["dummy"],
            reference_circuit=inline,
        )
        service = PipelineService()
        ref_circuit, ref_meta = service._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )

        assert ref_circuit == inline
        assert ref_meta["source"] == "inline_payload"

    def test_resolve_reference_none(self) -> None:
        from app.schemas.pipeline import PipelineRequest
        from app.services.pipeline_service import PipelineService

        request = PipelineRequest(
            station_id="S01",
            images_b64=["dummy"],
        )
        service = PipelineService()
        ref_circuit, ref_meta = service._resolve_reference(
            reference_id=request.reference_id,
            reference_circuit=request.reference_circuit,
        )

        assert ref_circuit is None
        assert ref_meta["source"] == "none"
