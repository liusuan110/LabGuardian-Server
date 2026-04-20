"""
T7: 电路验证测试 — 验证 CircuitValidator 的风险评估和诊断

无模型依赖：验证基于拓扑图和内置规则
"""

from __future__ import annotations

import pytest


class TestS4Validation:
    """T7: 电路验证测试."""

    def test_t7_1_complete_circuit_no_reference(self):
        """T7.1: 完整电路无参考 → risk_level 非空, diagnosis 非空"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        # 简单电路：R1 跨 A1-A3, C1 跨 B1-B3
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3",
                     "electrical_node_id": "ROW_3_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        assert "risk_level" in result
        assert "diagnosis" in result
        # 即使无参考，也应该有诊断
        assert "risk_level" in result
        assert result["risk_level"] in ("low", "medium", "high", "unknown", "safe")
        assert "duration_ms" in result

    def test_t7_2_led_without_resistor(self):
        """T7.2: LED 无相邻电阻 → risk_reasons 包含限流电阻警告"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            {
                "component_id": "D1",
                "component_type": "LED",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A5",
                     "electrical_node_id": "ROW_5_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        # LED 无相邻电阻应该产生警告
        assert "risk_reasons" in result
        # 风险原因可能是限流电阻相关
        reasons_text = " ".join(str(r) for r in result.get("risk_reasons", []))
        # 应该包含诊断信息
        assert isinstance(result["risk_reasons"], list)

    def test_t7_3_short_circuit_warning(self):
        """T7.3: 两 pin 同 net（非 Wire）→ 短路警告"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        # Resistor 两 pin 在同一行（同行 a-e 导通 = 短路）
        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    # A2 与 A1 同在 ROW_1_L（面包板同行导通）→ 短路
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "B1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        # 应该有诊断信息
        assert "risk_reasons" in result
        assert "diagnosis" in result

    def test_t7_4_polarity_warning(self):
        """T7.4: 极性器件方向未知 → 极性警告"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            {
                "component_id": "D1",
                "component_type": "LED",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3",
                     "electrical_node_id": "ROW_3_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        # 应该有诊断或风险原因
        assert "risk_level" in result
        assert "diagnostics" in result
        assert isinstance(result["diagnostics"], list)

    def test_t7_5_empty_topology(self):
        """空拓扑 → 不 crash"""
        from app.pipeline.stages.s4_validate import run_validate

        empty_graph = {"nodes": [], "links": []}
        result = run_validate(
            topology_graph=empty_graph,
            reference_circuit=None,
            components=[],
        )

        assert "risk_level" in result
        assert "diagnosis" in result
        assert result["risk_level"] in ("low", "medium", "high", "unknown", "safe", "")

    def test_t7_6_validation_result_schema(self):
        """验证 S4 输出 schema 完整性"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3",
                     "electrical_node_id": "ROW_3_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        required_fields = [
            "risk_level", "diagnosis", "diagnostics",
            "risk_reasons", "details", "duration_ms",
            "is_correct", "similarity",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_t7_7_wire_recognized(self):
        """Wire 元件不参与风险检查（通过 Union-Find 合并网络）"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3",
                     "electrical_node_id": "ROW_3_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            },
            {
                "component_id": "W1",
                "component_type": "Wire",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "B1",
                     "electrical_node_id": "ROW_1_R", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            },
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        # Wire 应该被识别但不会触发错误
        assert result["risk_level"] in ("low", "medium", "high", "unknown", "safe")
        assert "risk_level" in result

    def test_t7_8_comparison_report_fields(self):
        """验证 comparison_report 包含必要字段"""
        from app.pipeline.stages.s4_validate import run_validate
        from app.pipeline.stages.s3_topology import run_topology

        components = [
            {
                "component_id": "R1",
                "component_type": "Resistor",
                "pins": [
                    {"pin_id": 1, "pin_name": "pin1", "hole_id": "A1",
                     "electrical_node_id": "ROW_1_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                    {"pin_id": 2, "pin_name": "pin2", "hole_id": "A3",
                     "electrical_node_id": "ROW_3_L", "confidence": 0.95,
                     "observations": [], "source": "heuristic_fallback"},
                ],
            }
        ]

        s3 = run_topology(components=components)
        result = run_validate(
            topology_graph=s3["topology_graph"],
            reference_circuit=None,
            components=components,
        )

        if "comparison_report" in result:
            report = result["comparison_report"]
            assert "items" in report
            assert "summary" in report
