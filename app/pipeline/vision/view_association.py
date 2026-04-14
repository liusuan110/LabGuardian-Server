"""
Side-view ROI association helpers.

这一层的目标是把侧视图 ROI 来源从单纯 `shared_bbox_fallback`
逐步升级成“有明确来源的关联结果”。

当前第一版策略:
- top 视图仍然直接使用主检测 bbox
- side 视图优先匹配 S1 输出的 supplemental_detections
- 若无可靠候选, 再显式回退到 shared bbox

后续可以在不改 S1.5 协议的前提下替换成:
- 固定机位几何投影
- 多视图 instance association
- side-view detector + re-id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class SideViewAssociation:
    view_id: str
    matched: bool
    source: str
    bbox: tuple[int, int, int, int]
    confidence: float
    candidate_id: str = ""
    metadata: dict[str, Any] | None = None


class SideViewRoiResolver:
    """侧视图 ROI 关联器.

    当前只实现一个轻量骨架:
    - 根据 supplemental detections 做类内候选匹配
    - 后续可在内部替换成更强的几何/外观关联
    """

    interface_version = "side_view_roi_assoc_v1"

    def resolve(
        self,
        *,
        component_detection: dict[str, Any],
        view_id: str,
        supplemental_detections: Iterable[dict[str, Any]] | None,
    ) -> SideViewAssociation | None:
        if view_id == "top":
            return None

        component_type = str(component_detection.get("component_type") or component_detection.get("class_name") or "")
        package_type = str(component_detection.get("package_type") or "")
        top_bbox = tuple(component_detection.get("bbox") or (0, 0, 0, 0))

        candidates = [
            item for item in (supplemental_detections or [])
            if str(item.get("view_id") or "") == view_id
        ]
        if not candidates:
            return SideViewAssociation(
                view_id=view_id,
                matched=False,
                source="shared_bbox_fallback",
                bbox=top_bbox,
                confidence=0.0,
                metadata={"reason": "no_side_candidates"},
            )

        matched = _select_best_candidate(
            component_type=component_type,
            package_type=package_type,
            top_bbox=top_bbox,
            candidates=candidates,
        )
        if matched is None:
            return SideViewAssociation(
                view_id=view_id,
                matched=False,
                source="shared_bbox_fallback",
                bbox=top_bbox,
                confidence=0.0,
                metadata={"reason": "no_compatible_candidate"},
            )

        return SideViewAssociation(
            view_id=view_id,
            matched=True,
            source="associated_bbox_candidate",
            bbox=tuple(matched.get("bbox") or top_bbox),
            confidence=float(matched.get("confidence", 0.0)),
            candidate_id=str(matched.get("candidate_id") or ""),
            metadata={
                "matched_component_type": matched.get("component_type"),
                "matched_package_type": matched.get("package_type"),
                "instance_status": matched.get("instance_status", "candidate"),
            },
        )


def _select_best_candidate(
    *,
    component_type: str,
    package_type: str,
    top_bbox: tuple[int, int, int, int],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    best_score = -1.0
    for item in candidates:
        score = 0.0
        if str(item.get("component_type") or item.get("class_name") or "") == component_type:
            score += 2.0
        if str(item.get("package_type") or "") == package_type:
            score += 1.0
        score += float(item.get("confidence", 0.0))
        score -= _bbox_distance_penalty(top_bbox, tuple(item.get("bbox") or top_bbox))
        if score > best_score:
            best = item
            best_score = score
    return best


def _bbox_distance_penalty(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax = (bbox_a[0] + bbox_a[2]) / 2.0
    ay = (bbox_a[1] + bbox_a[3]) / 2.0
    bx = (bbox_b[0] + bbox_b[2]) / 2.0
    by = (bbox_b[1] + bbox_b[3]) / 2.0
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) / 1000.0
