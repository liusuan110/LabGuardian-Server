"""
Stage 3: 拓扑构建

读取 S2 的映射结果，利用 CircuitAnalyzer 构建面包板电路图（NetworkX 拓扑）。
输出电路拓扑的 netlist 描述 + 图对象序列化。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.domain.circuit import CircuitAnalyzer
from app.domain.polarity import PolarityResolver

logger = logging.getLogger(__name__)


def run_topology(
    components: List[dict],
    polarity_resolver: PolarityResolver | None = None,
) -> Dict[str, Any]:
    """从映射好的元件列表构建电路拓扑

    Returns:
        {
            "circuit_description": str,
            "netlist": dict,
            "topology_graph": dict,    # node_link_data
            "component_count": int,
            "duration_ms": float,
        }
    """
    t0 = time.time()

    analyzer = CircuitAnalyzer()

    for comp in components:
        class_name = comp["class_name"]
        pin1 = tuple(comp["pin1_logic"]) if comp.get("pin1_logic") else None
        pin2 = tuple(comp["pin2_logic"]) if comp.get("pin2_logic") else None

        if pin1 is None or pin2 is None:
            logger.debug("跳过缺失引脚的元件: %s", class_name)
            continue

        # LED 极性推理
        positive_first: bool | None = None
        if polarity_resolver and class_name.lower() == "led":
            bbox = tuple(comp["bbox"])
            positive_first = polarity_resolver.infer(bbox)

        analyzer.add_component(
            class_name,
            pin1,
            pin2,
            positive_first=positive_first,
        )

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
