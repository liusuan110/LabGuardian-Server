"""
Stage 4: 电路检错

对比 S3 构建的拓扑与参考电路，执行 L0-L3 多级诊断，输出风险等级与反馈文本。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.domain.circuit import CircuitAnalyzer, CircuitComponent, Polarity
from app.domain.ic_models import build_dip8_component
from app.domain.risk import RiskLevel, classify_risk
from app.domain.validator import CircuitValidator

logger = logging.getLogger(__name__)


def run_validate(
    topology_graph: dict,
    reference_path: str | None = None,
    components: List[dict] | None = None,
) -> Dict[str, Any]:
    """执行电路验证

    Args:
        topology_graph: S3 输出的 node_link_data
        reference_path: 参考电路 JSON 路径（可选）
        components: S2 输出的映射元件列表 (用于重建 CircuitAnalyzer 进行比较)

    Returns:
        {
            "is_correct": bool,
            "diagnosis": str,
            "risk_level": str,
            "similarity": float,
            "details": dict,
            "diagnostics": list,
            "duration_ms": float,
        }
    """
    t0 = time.time()

    validator = CircuitValidator()
    topology_meta = {
        "topology_node_count": len(topology_graph.get("nodes", [])) if isinstance(topology_graph, dict) else 0,
        "topology_edge_count": len(topology_graph.get("links", [])) if isinstance(topology_graph, dict) else 0,
    }

    if reference_path:
        try:
            validator.load_reference(reference_path)
        except Exception as e:
            logger.warning("加载参考电路失败: %s", e)

    if validator.has_reference and components:
        curr_analyzer = _rebuild_analyzer(components)
        result = validator.compare(curr_analyzer)
        errors = result.get("errors", [])
        polarity_errors = result.get("polarity_errors", [])
        all_diagnostics = errors + polarity_errors
        diagnosis_text = "\n".join(all_diagnostics) if all_diagnostics else ""

        independent_diags = CircuitValidator.diagnose(curr_analyzer)
        all_diagnostics.extend(independent_diags)
    elif validator.has_reference:
        result = {
            "is_correct": False,
            "diagnosis": "无法从检测结果重建电路进行比较",
            "similarity": 0.0,
            "details": {},
        }
        diagnosis_text = result["diagnosis"]
        all_diagnostics = [diagnosis_text]
    else:
        if components:
            curr_analyzer = _rebuild_analyzer(components)
            independent_diags = CircuitValidator.diagnose(curr_analyzer)
            diagnosis_text = "未设置参考电路，无法验证" + (
                "\n" + "\n".join(independent_diags) if independent_diags else ""
            )
            all_diagnostics = independent_diags
        else:
            diagnosis_text = "未设置参考电路，无法验证"
            all_diagnostics = []
        result = {
            "is_correct": False,
            "diagnosis": diagnosis_text,
            "similarity": 0.0,
            "details": {},
        }

    diag_lines = [l for l in diagnosis_text.splitlines() if l.strip()] if diagnosis_text else []
    diag_lines.extend(all_diagnostics)
    diag_lines = list(dict.fromkeys(diag_lines))
    risk_level, risk_reasons = classify_risk(diag_lines)

    duration_ms = (time.time() - t0) * 1000

    return {
        "is_correct": result.get("is_correct", result.get("is_match", False)),
        "diagnosis": diagnosis_text,
        "risk_level": risk_level.value,
        "similarity": result.get("similarity", 0.0),
        "progress": result.get("progress", 0.0),
        "diagnostics": diag_lines,
        "risk_reasons": risk_reasons,
        "details": {
            **result.get("details", {}),
            **topology_meta,
        },
        "duration_ms": duration_ms,
    }


def _rebuild_analyzer(components: List[dict]) -> CircuitAnalyzer:
    """从 S2 输出的映射元件列表重建 CircuitAnalyzer"""
    analyzer = CircuitAnalyzer()
    for comp in components:
        pin1 = comp.get("pin1_logic")
        pin2 = comp.get("pin2_logic")
        if not pin1 or not pin2:
            continue
        pin1 = (str(pin1[0]), str(pin1[1]))
        pin2 = (str(pin2[0]), str(pin2[1]))
        class_name = comp["class_name"]
        polarity = Polarity.UNKNOWN if class_name.lower() == "led" else Polarity.NONE
        if class_name == "IC":
            cc = build_dip8_component(
                class_name=class_name,
                pin1=pin1,
                pin2=pin2,
                confidence=comp.get("confidence", 1.0),
            )
        else:
            cc = CircuitComponent(
                name="",
                type=class_name,
                pin1_loc=pin1,
                pin2_loc=pin2,
                polarity=polarity,
                confidence=comp.get("confidence", 1.0),
            )
        analyzer.add_component(cc)
    return analyzer
