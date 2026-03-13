"""
Stage 3: 拓扑构建

读取 S2 的映射结果，利用 CircuitAnalyzer 构建面包板电路图（NetworkX 拓扑）。
输出电路拓扑的 netlist 描述 + 图对象序列化。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.domain.circuit import CircuitAnalyzer, CircuitComponent, Polarity
from app.domain.ic_models import build_dip8_component
from app.domain.polarity import PolarityResolver

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

    analyzer = CircuitAnalyzer()

    if rail_assignments:
        for track_id, label in rail_assignments.items():
            analyzer.set_rail_assignment(track_id, label)

    for comp in components:
        class_name = comp["class_name"]
        pin1 = tuple(comp["pin1_logic"]) if comp.get("pin1_logic") else None
        pin2 = tuple(comp["pin2_logic"]) if comp.get("pin2_logic") else None

        if pin1 is None or pin2 is None:
            logger.debug("跳过缺失引脚的元件: %s", class_name)
            continue

        polarity = Polarity.NONE
        if class_name.lower() in ("led", "diode"):
            if polarity_resolver:
                bbox = tuple(comp["bbox"])
                positive_first = polarity_resolver.infer(bbox)
                if positive_first is True:
                    polarity = Polarity.FORWARD
                elif positive_first is False:
                    polarity = Polarity.REVERSE
                else:
                    polarity = Polarity.UNKNOWN
            else:
                polarity = Polarity.UNKNOWN

        if class_name == "IC":
            circuit_comp = build_dip8_component(
                class_name=class_name,
                pin1=pin1,
                pin2=pin2,
                confidence=comp.get("confidence", 1.0),
            )
        else:
            circuit_comp = CircuitComponent(
                name="",
                type=class_name,
                pin1_loc=pin1,
                pin2_loc=pin2,
                polarity=polarity,
                confidence=comp.get("confidence", 1.0),
            )
        analyzer.add_component(circuit_comp)

    circuit_description = analyzer.describe()
    netlist = analyzer.export_netlist()
    topology_graph = analyzer.to_node_link_data()
    component_count = analyzer.component_count()

    duration_ms = (time.time() - t0) * 1000

    return {
        "circuit_description": circuit_description,
        "netlist": netlist,
        "topology_graph": topology_graph,
        "component_count": component_count,
        "duration_ms": duration_ms,
    }
