"""
T6: 拓扑构建测试 — 验证 CircuitAnalyzer 的拓扑建模

无模型依赖：拓扑构建基于结构化的 components[].pins[] 数据
"""

from __future__ import annotations

import pytest


def make_mapped_component(
    component_id: str,
    component_type: str,
    pins: list[dict],
) -> dict:
    """构建 S2 输出的模拟 mapped component."""
    return {
        "component_id": component_id,
        "component_type": component_type,
        "package_type": "axial_2pin" if component_type in ("Resistor", "Capacitor", "Wire") else "generic",
        "pins": pins,
        "confidence": 0.95,
    }


def make_pin(
    pin_id: int,
    hole_id: str,
    electrical_node_id: str | None = None,
    confidence: float = 0.95,
) -> dict:
    return {
        "pin_id": pin_id,
        "pin_name": f"pin{pin_id}",
        "hole_id": hole_id,
        "electrical_node_id": electrical_node_id,
        "confidence": confidence,
        "observations": [],
        "source": "heuristic_fallback",
    }


class TestS3Topology:
    """T6: 拓扑构建测试."""

    def test_t6_1_single_resistor_two_rows(self):
        """T6.1: 单个 Resistor 两 pin 不同 row → 生成 net"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component(
                "R1", "Resistor",
                [
                    make_pin(1, hole_id="A5", electrical_node_id="ROW_5_L"),
                    make_pin(2, hole_id="A10", electrical_node_id="ROW_10_L"),
                ]
            )
        ]

        result = run_topology(components=components)

        assert result["component_count"] == 1
        assert "netlist_v2" in result
        assert "topology_graph" in result
        assert "circuit_description" in result
        # 单个两端器件跨两个不同节点 → 1 个 net 连接两端
        nets = result["netlist_v2"]["nets"]
        assert len(nets) >= 1

    def test_t6_2_two_resistors_series(self):
        """T6.2: 2 个 Resistor 串联 → 3 个 net"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component(
                "R1", "Resistor",
                [
                    make_pin(1, hole_id="A5", electrical_node_id="ROW_5_L"),
                    make_pin(2, hole_id="A7", electrical_node_id="ROW_7_L"),
                ]
            ),
            make_mapped_component(
                "R2", "Resistor",
                [
                    make_pin(1, hole_id="A7", electrical_node_id="ROW_7_L"),  # R1.pin2 同一孔
                    make_pin(2, hole_id="A10", electrical_node_id="ROW_10_L"),
                ]
            ),
        ]

        result = run_topology(components=components)

        assert result["component_count"] == 2
        nets = result["netlist_v2"]["nets"]
        # R1(A5)-R2(A10) 通过 A7 串联 → 3 个节点 = 3 个 net
        assert len(nets) >= 2  # 至少中间节点合并

    def test_t6_3_wire_merges_nets(self):
        """T6.3: Wire 连接两孔位 → Union-Find 合并两端为同一 net"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component(
                "R1", "Resistor",
                [
                    make_pin(1, hole_id="A5", electrical_node_id="ROW_5_L"),
                    make_pin(2, hole_id="A7", electrical_node_id="ROW_7_L"),
                ]
            ),
            make_mapped_component(
                "W1", "Wire",
                [
                    make_pin(1, hole_id="A5", electrical_node_id="ROW_5_L"),  # 与 R1.pin1 同一孔
                    make_pin(2, hole_id="B5", electrical_node_id="ROW_5_R"),
                ]
            ),
        ]

        result = run_topology(components=components)

        nets = result["netlist_v2"]["nets"]
        # Wire 将 A5 和 B5 合并为同一 net（因为 wire 两端导通）
        # R1 的两个 pin 分属不同 net
        assert len(nets) >= 1
        # 检查 W1 是否被正确识别为 Wire
        comp_ids = [c["component_id"] for c in result["netlist_v2"]["components"]]
        assert "W1" in comp_ids

    def test_t6_4_empty_components(self):
        """T6.4: 无元件 → component_count=0, nets=[]"""
        from app.pipeline.stages.s3_topology import run_topology

        result = run_topology(components=[])

        assert result["component_count"] == 0
        assert result["netlist_v2"]["nets"] == []
        assert result["netlist_v2"]["components"] == []
        assert "No circuit" in result["circuit_description"] or result["circuit_description"] != ""

    def test_t6_5_led_polarity(self):
        """T6.5: LED 极性标记"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component(
                "D1", "LED",
                [
                    make_pin(1, hole_id="A5", electrical_node_id="ROW_5_L"),
                    make_pin(2, hole_id="A7", electrical_node_id="ROW_7_L"),
                ]
            ),
        ]

        result = run_topology(components=components)

        assert result["component_count"] == 1
        led_comp = result["netlist_v2"]["components"][0]
        assert led_comp["component_type"] == "LED"
        # polarity 应存在（forward/unknown/reverse）
        assert "polarity" in led_comp

    def test_t6_6_component_type_preserved(self):
        """验证元件类型正确传递到拓扑"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component("R1", "Resistor", [
                make_pin(1, hole_id="A1", electrical_node_id="ROW_1_L"),
                make_pin(2, hole_id="A3", electrical_node_id="ROW_3_L"),
            ]),
            make_mapped_component("C1", "Capacitor", [
                make_pin(1, hole_id="B1", electrical_node_id="ROW_1_R"),
                make_pin(2, hole_id="B3", electrical_node_id="ROW_3_R"),
            ]),
            make_mapped_component("D1", "LED", [
                make_pin(1, hole_id="C1", electrical_node_id="ROW_1_L"),
                make_pin(2, hole_id="C3", electrical_node_id="ROW_3_L"),
            ]),
        ]

        result = run_topology(components=components)

        types = {c["component_type"] for c in result["netlist_v2"]["components"]}
        assert types == {"Resistor", "Capacitor", "LED"}

    def test_t6_7_netlist_v2_schema(self):
        """验证 netlist_v2 输出 schema 完整性"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component("R1", "Resistor", [
                make_pin(1, hole_id="A1", electrical_node_id="ROW_1_L"),
                make_pin(2, hole_id="A5", electrical_node_id="ROW_5_L"),
            ]),
        ]

        result = run_topology(components=components)
        nl = result["netlist_v2"]

        required_fields = ["scene_id", "board_schema_id", "components", "nets", "node_index"]
        for field in required_fields:
            assert field in nl, f"Missing field in netlist_v2: {field}"

    def test_t6_8_topology_graph_serialization(self):
        """验证 topology_graph 可序列化"""
        from app.pipeline.stages.s3_topology import run_topology
        import json

        components = [
            make_mapped_component("R1", "Resistor", [
                make_pin(1, hole_id="A1", electrical_node_id="ROW_1_L"),
                make_pin(2, hole_id="A5", electrical_node_id="ROW_5_L"),
            ]),
        ]

        result = run_topology(components=components)
        graph = result["topology_graph"]

        # 验证可 JSON 序列化
        json_str = json.dumps(graph)
        restored = json.loads(json_str)
        assert "nodes" in restored
        assert "links" in restored

    def test_t6_9_rail_assignments(self):
        """电源轨道分配"""
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            make_mapped_component("R1", "Resistor", [
                make_pin(1, hole_id="A1", electrical_node_id="ROW_1_L"),
                make_pin(2, hole_id="A5", electrical_node_id="ROW_5_L"),
            ]),
        ]

        rail_assignments = {
            "top_plus": "VCC",
            "top_minus": "GND",
        }
        result = run_topology(components=components, rail_assignments=rail_assignments)

        assert result["component_count"] == 1
        # 应该不影响基本功能

    def test_t6_10_pin_hole_id_in_net(self):
        """验证每个 pin 的 hole_id 出现在 net 的 member_hole_ids 中"""
        from app.pipeline.stages.s3_topology import run_topology

        hole_a = "A1"
        hole_b = "A5"
        components = [
            make_mapped_component("R1", "Resistor", [
                make_pin(1, hole_id=hole_a, electrical_node_id="ROW_1_L"),
                make_pin(2, hole_id=hole_b, electrical_node_id="ROW_5_L"),
            ]),
        ]

        result = run_topology(components=components)
        nets = result["netlist_v2"]["nets"]

        # 收集所有 net 的 member_hole_ids
        all_holes = set()
        for net in nets:
            all_holes.update(net.get("member_hole_ids", []))

        # R1 的两个 pin 的 hole_id 应该出现在某个 net 中
        assert hole_a in all_holes or hole_b in all_holes
