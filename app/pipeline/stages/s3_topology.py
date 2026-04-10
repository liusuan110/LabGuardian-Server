"""
Stage 3: 拓扑构建

读取 S2 的映射结果，利用 CircuitAnalyzer 构建面包板电路图（NetworkX 拓扑）。
输出电路拓扑的 netlist 描述 + 图对象序列化。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.domain.board_schema import BoardSchema
from app.domain.circuit import CircuitAnalyzer
from app.domain.polarity import PolarityResolver
from app.pipeline.topology_input import build_analyzer_from_components

logger = logging.getLogger(__name__)


def run_topology(
    components: List[dict],
    polarity_resolver: PolarityResolver | None = None,
    rail_assignments: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """从映射好的元件列表构建电路拓扑

    Args:
        components: S2 映射后的元件列表
        polarity_resolver: 极性解析器
        rail_assignments: 电源轨道指定, 如
            {"top_plus": "VCC", "top_minus": "GND",
             "bot_plus": "VCC", "bot_minus": "GND"}

    Returns:
        {
            "circuit_description": str,
            "netlist": dict,
            "topology_graph": dict,
            "component_count": int,
            "duration_ms": float,
        }
    """
    t0 = time.time()

    board_schema = BoardSchema.default_breadboard()

    analyzer, normalized_components = build_analyzer_from_components(
        components,
        board_schema=board_schema,
        polarity_resolver=polarity_resolver,
    )
    if rail_assignments:
        for track_id, label in rail_assignments.items():
            analyzer.set_rail_assignment(track_id, label)

    circuit_description = analyzer.describe()
    netlist = analyzer.export_netlist()
    netlist_v2 = analyzer.export_netlist_v2()
    topology_graph = analyzer.to_node_link_data()
    component_count = analyzer.component_count()

    duration_ms = (time.time() - t0) * 1000

    return {
        "circuit_description": circuit_description,
        "netlist": netlist,
        "netlist_v2": netlist_v2,
        "normalized_components": [
            {
                "component_id": comp.component_id,
                "component_type": comp.component_type,
                "package_type": comp.package_type,
                "polarity": comp.polarity,
                "pins": [
                    {
                        "pin_id": pin.pin_id,
                        "pin_name": pin.pin_name,
                        "hole_id": pin.hole_id,
                        "electrical_node_id": pin.electrical_node_id,
                    }
                    for pin in comp.pins
                ],
            }
            for comp in normalized_components
        ],
        "topology_graph": topology_graph,
        "component_count": component_count,
        "duration_ms": duration_ms,
    }
