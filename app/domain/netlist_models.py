"""
新一代网表领域模型。

目标是显式保留:
`component_id + pin_name + hole_id -> electrical_node_id -> electrical_net_id`
这条链路，供后续 validator / RAG / agent 直接复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PinObservation:
    view_id: str
    keypoint: Optional[tuple[float, float]] = None
    visibility: int = 0
    confidence: float = 0.0


@dataclass
class PinAssignment:
    pin_id: int
    pin_name: str
    hole_id: str
    electrical_node_id: Optional[str] = None
    electrical_net_id: Optional[str] = None
    observations: List[PinObservation] = field(default_factory=list)
    confidence: float = 0.0
    is_ambiguous: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentInstance:
    component_id: str
    component_type: str
    package_type: str
    part_subtype: str = ""
    polarity: str = "none"
    orientation: float = 0.0
    symmetry_group: List[List[str]] = field(default_factory=list)
    pins: List[PinAssignment] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElectricalNet:
    electrical_net_id: str
    member_node_ids: List[str]
    member_hole_ids: List[str]
    power_role: str = ""
    labels: List[str] = field(default_factory=list)


@dataclass
class NetlistV2:
    scene_id: str
    board_schema_id: str
    components: List[ComponentInstance]
    nets: List[ElectricalNet]
    node_index: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
