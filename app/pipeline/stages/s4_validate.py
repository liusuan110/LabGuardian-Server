"""
Stage 4: 电路检错

对比 S3 构建的拓扑与参考电路 logical_reference_v1，输出风险等级与反馈文本。

只支持 logical_reference_v1 格式；不再支持孔位级/物理点位级 reference 对比。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.domain.graph_compare import compare_logical_graphs
from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph
from app.domain.reference_formats import (
    SUPPORTED_REFERENCE_FORMAT,
    get_reference_format,
    unsupported_reference_format_message,
)
from app.domain.risk import classify_risk, RiskLevel

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
        reference_circuit: 参考电路内联 payload（可选）
        components: S2 输出的映射元件列表（用于 current_netlist_v2 缺失时的兜底重建）
        current_netlist_v2: 已应用手动角色等后处理的 netlist_v2（优先使用）

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

    topology_meta = {
        "topology_node_count": len(topology_graph.get("nodes", [])) if isinstance(topology_graph, dict) else 0,
        "topology_edge_count": len(topology_graph.get("links", [])) if isinstance(topology_graph, dict) else 0,
    }

    # 1. 未设置参考电路
    if not reference_circuit:
        item = {
            "error_code": "REFERENCE_NOT_SET",
            "error_family": "reference_format",
            "severity": "info",
            "message": "未设置参考电路，无法验证。",
            "expected": {"reference_circuit": "provided"},
            "actual": {"reference_circuit": None},
            "suggested_action": "请在请求中提供 reference_id 或 reference_circuit。",
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
            started_at=t0,
        )

    reference_format = get_reference_format(reference_circuit)

    # 2. 支持 logical_reference_v1
    if isinstance(reference_circuit, dict) and reference_format == SUPPORTED_REFERENCE_FORMAT:
        return _run_logical_reference_validate(
            reference_circuit=reference_circuit,
            components=components,
            topology_meta=topology_meta,
            started_at=t0,
            current_netlist_v2=current_netlist_v2,
        )

    # 3. 其他格式不再支持
    item = {
        "error_code": "UNSUPPORTED_REFERENCE_FORMAT",
        "error_family": "reference_format",
        "severity": "error",
        "message": unsupported_reference_format_message(reference_format),
        "expected": {"format": SUPPORTED_REFERENCE_FORMAT},
        "actual": {"format": reference_format or str(type(reference_circuit).__name__)},
        "suggested_action": f"请将参考电路转换为 {SUPPORTED_REFERENCE_FORMAT} 格式后重试。",
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
        started_at=t0,
    )


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

    if current_netlist_v2 is None:
        item = {
            "error_code": "INCOMPLETE_CIRCUIT",
            "error_family": "incomplete_circuit",
            "severity": "error",
            "message": "缺少当前电路的 netlist_v2，无法进行逻辑参考比较。",
            "expected": {"current_netlist_v2": "provided"},
            "actual": {"current_netlist_v2": None},
            "suggested_action": "请确保 pipeline 的 topology 阶段成功输出了 netlist_v2。",
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
            "expected": {"format": SUPPORTED_REFERENCE_FORMAT},
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
    item = {
        "title": item.get("title") or item.get("error_code") or "logical validation error",
        "component_ref": item.get("component_ref"),
        "component_actual": item.get("component_actual"),
        "evidence_refs": item.get("evidence_refs", []),
        **item,
    }
    return {
        "version": "validator_report_v2",
        "summary": {
            "total_item_count": 1,
            "logic_correct": False,
            "similarity": similarity,
            "comparison_mode": "logical_graph",
            "match_type": "reference_error",
            "ignore_component_id": True,
            "ignore_hole_id": True,
            "ignore_passive_pin_order": True,
            "allow_extra_wires": True,
            "strict_functional_pin_roles": True,
            "equivalence_rule": "logical_topology_with_port_semantics",
            "report_layers": {
                "erc": {"source": "semantic_analysis", "included": False},
                "reference_compare": {"source": "s4_validate", "included": True},
            },
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
    # 确定性比对已判 is_correct=False(电路与参考不一致)时,即便诊断文案未命中
    # risk.py 的危险/警告关键词,也不应判 safe —— 至少 WARNING,避免连接/角色不
    # 匹配这类错误被纯关键词匹配机制掩盖(此前 risk_level=safe 掩盖真实故障的根因)。
    if not is_correct and risk_level == RiskLevel.SAFE:
        risk_level = RiskLevel.WARNING
        if not risk_reasons:
            risk_reasons = [d for d in diagnostics if d]
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
