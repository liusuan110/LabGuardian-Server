"""
Stage 4: 电路检错

对比 S3 构建的拓扑与参考电路，执行 L0-L3 多级诊断，输出风险等级与反馈文本。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.domain.risk import RiskLevel, classify_risk
from app.domain.validator import CircuitValidator

logger = logging.getLogger(__name__)


def run_validate(
    topology_graph: dict,
    reference_path: str | None = None,
) -> Dict[str, Any]:
    """执行电路验证

    Args:
        topology_graph: S3 输出的 node_link_data
        reference_path: 参考电路 JSON 路径（可选）

    Returns:
        {
            "is_correct": bool,
            "diagnosis": str,
            "risk_level": str,
            "similarity": float,
            "details": dict,
            "duration_ms": float,
        }
    """
    t0 = time.time()

    validator = CircuitValidator()

    # 加载参考电路
    if reference_path:
        try:
            validator.load_reference(reference_path)
        except Exception as e:
            logger.warning("加载参考电路失败: %s", e)

    # 执行比较
    if validator.has_reference:
        result = validator.compare(topology_graph)
    else:
        result = {
            "is_correct": False,
            "diagnosis": "未设置参考电路，无法验证",
            "similarity": 0.0,
            "details": {},
        }

    # 风险分类
    diagnosis_text = result.get("diagnosis", "")
    risk = classify_risk(diagnosis_text)

    duration_ms = (time.time() - t0) * 1000

    return {
        "is_correct": result.get("is_correct", False),
        "diagnosis": diagnosis_text,
        "risk_level": risk.value,
        "similarity": result.get("similarity", 0.0),
        "details": result.get("details", {}),
        "duration_ms": duration_ms,
    }
