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
from app.domain.graph_compare import compare_logical_graphs
from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph
from app.domain.risk import RiskLevel, classify_risk
from app.domain.validator import CircuitValidator
from app.pipeline.topology_input import build_analyzer_from_components

logger = logging.getLogger(__name__)


def run_validate(
    topology_graph: dict,
    reference_circuit: Dict[str, Any] | str | None = None,
    components: List[dict] | None = None,
    current_netlist_v2: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """执行电路验证

    Args:
        topology_graph: S3 输出的 node_link_data
        reference_circuit: 参考电路 JSON 路径或内联 reference payload（可选）
        components: S2 输出的映射元件列表 (用于重建 CircuitAnalyzer 进行比较)
        current_netlist_v2: 已应用手动角色等后处理的 netlist_v2（可选，
            logical_reference_v1 分支会优先使用它而不是从 components 重建）

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

    if isinstance(reference_circuit, dict) and reference_circuit.get("format") == "logical_reference_v1":
        return _run_logical_reference_validate(
            reference_circuit=reference_circuit,
            components=components,
            topology_meta=topology_meta,
            started_at=t0,
            current_netlist_v2=current_netlist_v2,
        )

    if reference_circuit:
        try:
            if isinstance(reference_circuit, dict):
                validator.load_reference_payload(reference_circuit)
            else:
                validator.load_reference(reference_circuit)
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
    )
    return analyzer


def _run_logical_reference_validate(
    *,
    reference_circuit: Dict[str, Any],
    components: List[dict] | None,
    topology_meta: Dict[str, Any],
    started_at: float,
    current_netlist_v2: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not components:
        item = {
            "error_code": "INCOMPLETE_CIRCUIT",
            "error_family": "incomplete_circuit",
            "severity": "error",
            "message": "无法从检测结果重建当前电路，无法进行逻辑参考比较。",
            "expected": {"components": "non_empty"},
            "actual": {"components": 0},
            "suggested_action": "请先确认图像识别结果中包含元件与引脚。",
        }
        comparison_report = _logical_error_report(item, similarity=0.0)
        return _logical_s4_response(
            is_correct=False,
            diagnosis=item["message"],
            diagnostics=[item["message"]],
            comparison_report=comparison_report,
            similarity=0.0,
            progress=0.0,
            details={"comparison_mode": "logical_graph", **topology_meta},
            started_at=started_at,
        )

    try:
        if current_netlist_v2 is None:
            curr_analyzer = _rebuild_analyzer(components)
            current_netlist_v2 = curr_analyzer.export_netlist_v2()
        reference_graph = logical_reference_to_graph(reference_circuit)
        current_graph = current_netlist_v2_to_graph(current_netlist_v2)
        result = compare_logical_graphs(
            reference_graph,
            current_graph,
            ref_payload=reference_circuit,
            cur_netlist_v2=current_netlist_v2,
        )
    except Exception as exc:
        logger.warning("logical reference validation failed: %s", exc)
        item = {
            "error_code": "REFERENCE_INVALID",
            "error_family": "reference_format",
            "severity": "error",
            "message": f"logical_reference_v1 参考电路格式无效: {exc}",
            "expected": {"format": "logical_reference_v1"},
            "actual": {},
            "suggested_action": "请检查上传的参考电路 JSON 格式。",
        }
        comparison_report = _logical_error_report(item, similarity=0.0)
        return _logical_s4_response(
            is_correct=False,
            diagnosis=item["message"],
            diagnostics=[item["message"]],
            comparison_report=comparison_report,
            similarity=0.0,
            progress=0.0,
            details={"comparison_mode": "logical_graph", **topology_meta},
            started_at=started_at,
        )

    comparison_report = dict(result.get("report", {}))
    report_items = list(comparison_report.get("items", []))
    diagnostics = [str(item.get("message")) for item in report_items if item.get("message")]
    if not diagnostics:
        diagnostics = [str(result.get("message") or "电路逻辑连接与参考电路一致")]

    return _logical_s4_response(
        is_correct=bool(result.get("logic_correct", False)),
        diagnosis="\n".join(diagnostics),
        diagnostics=diagnostics,
        comparison_report=comparison_report,
        similarity=float(result.get("similarity", 0.0)),
        progress=float(result.get("progress", 0.0)),
        details={**result.get("details", {}), "comparison_mode": "logical_graph", **topology_meta},
        started_at=started_at,
    )


def _logical_error_report(item: Dict[str, Any], *, similarity: float) -> Dict[str, Any]:
    return {
        "version": "validator_report_v2",
        "summary": {
            "total_item_count": 1,
            "logic_correct": False,
            "similarity": similarity,
            "comparison_mode": "logical_graph",
            "ignore_component_id": True,
            "ignore_hole_id": True,
            "ignore_passive_pin_order": True,
            "ignore_polarity": True,
        },
        "items": [item],
        "topology_errors": [],
        "node_errors": [],
        "hole_errors": [],
        "component_errors": [],
        "polarity_errors": [],
    }


def _logical_s4_response(
    *,
    is_correct: bool,
    diagnosis: str,
    diagnostics: List[str],
    comparison_report: Dict[str, Any],
    similarity: float,
    progress: float,
    details: Dict[str, Any],
    started_at: float,
) -> Dict[str, Any]:
    risk_level, risk_reasons = classify_risk(diagnostics)
    return {
        "is_correct": is_correct,
        "diagnosis": diagnosis,
        "risk_level": risk_level.value,
        "similarity": similarity,
        "progress": progress,
        "diagnostics": diagnostics,
        "comparison_report": comparison_report,
        "risk_reasons": risk_reasons,
        "details": {
            **details,
            "topology_errors": comparison_report.get("topology_errors", []),
            "node_errors": comparison_report.get("node_errors", []),
            "hole_errors": comparison_report.get("hole_errors", []),
            "polarity_errors": comparison_report.get("polarity_errors", []),
            "component_errors": comparison_report.get("component_errors", []),
        },
        "duration_ms": (time.time() - started_at) * 1000,
    }
