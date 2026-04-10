"""
Stage 4: 电路检错

对比 S3 构建的拓扑与参考电路，执行 L0-L3 多级诊断，输出风险等级与反馈文本。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.domain.board_schema import BoardSchema
from app.domain.circuit import CircuitAnalyzer
from app.domain.risk import RiskLevel, classify_risk
from app.domain.validator import CircuitValidator
from app.pipeline.topology_input import build_analyzer_from_components

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

        independent_diag_items = CircuitValidator.diagnose_items(curr_analyzer)
        independent_diags = [item["message"] for item in independent_diag_items]
        all_diagnostics.extend(independent_diags)
    elif validator.has_reference:
        result = {
            "is_correct": False,
            "diagnosis": "无法从检测结果重建电路进行比较",
            "similarity": 0.0,
            "details": {},
            "report": {
                "version": "validator_report_v2",
                "items": [],
                "summary": {},
                "topology_errors": [],
                "node_errors": [],
                "hole_errors": [],
                "polarity_errors": [],
                "component_errors": [],
            },
        }
        diagnosis_text = result["diagnosis"]
        all_diagnostics = [diagnosis_text]
        independent_diag_items = []
    else:
        if components:
            curr_analyzer = _rebuild_analyzer(components)
            independent_diag_items = CircuitValidator.diagnose_items(curr_analyzer)
            independent_diags = [item["message"] for item in independent_diag_items]
            diagnosis_text = "未设置参考电路，无法验证" + (
                "\n" + "\n".join(independent_diags) if independent_diags else ""
            )
            all_diagnostics = independent_diags
        else:
            diagnosis_text = "未设置参考电路，无法验证"
            all_diagnostics = []
            independent_diag_items = []
        result = {
            "is_correct": False,
            "diagnosis": diagnosis_text,
            "similarity": 0.0,
            "details": {},
            "report": {
                "version": "validator_report_v2",
                "items": [],
                "summary": {},
                "topology_errors": [],
                "node_errors": [],
                "hole_errors": [],
                "polarity_errors": [],
                "component_errors": [],
            },
        }

    comparison_report = dict(result.get("report", {}))
    report_items = list(comparison_report.get("items", []))
    for item in independent_diag_items:
        if item not in report_items:
            report_items.append(item)
    comparison_report["items"] = report_items
    comparison_report.setdefault("version", "validator_report_v2")
    summary = dict(comparison_report.get("summary", {}))
    summary["independent_diagnostic_count"] = len(independent_diag_items)
    summary["total_item_count"] = len(report_items)
    comparison_report["summary"] = summary

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
        "comparison_report": comparison_report,
        "risk_reasons": risk_reasons,
        "details": {
            **result.get("details", {}),
            **topology_meta,
            "topology_errors": comparison_report.get("topology_errors", []),
            "node_errors": comparison_report.get("node_errors", []),
            "hole_errors": comparison_report.get("hole_errors", []),
            "polarity_errors": comparison_report.get("polarity_errors", []),
            "component_errors": comparison_report.get("component_errors", []),
        },
        "duration_ms": duration_ms,
    }


def _rebuild_analyzer(components: List[dict]) -> CircuitAnalyzer:
    """从 S2 输出的映射元件列表重建 CircuitAnalyzer"""
    board_schema = BoardSchema.default_breadboard()
    analyzer, _normalized_components = build_analyzer_from_components(
        components,
        board_schema=board_schema,
        polarity_resolver=None,
    )
    return analyzer
