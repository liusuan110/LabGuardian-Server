"""
S2 -> S3 标准化输入适配层。

当前仅接受结构化 `components[].pins[]` 作为正式输入。
旧的 `pin1_logic/pin2_logic` 兼容入口已经移除，避免新链路继续被旧语义拖回去。
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from app.domain.board_schema import BoardSchema
from app.domain.circuit import CircuitAnalyzer
from app.domain.ic_models import UA741_PIN_ROLES, build_dip8_pin_locs
from app.domain.netlist_models import ComponentInstance, PinAssignment, PinObservation


_TYPE_PREFIX = {
    "resistor": "R",
    "capacitor": "C",
    "wire": "W",
    "led": "LED",
    "diode": "D",
    "ic": "IC",
    "potentiometer": "POT",
}


def normalize_components_for_topology(
    components: List[dict],
    board_schema: BoardSchema,
) -> List[ComponentInstance]:
    counters: defaultdict[str, int] = defaultdict(int)
    normalized: List[ComponentInstance] = []
    for comp in components:
        instance = _normalize_component(comp, counters, board_schema)
        if instance is not None:
            normalized.append(instance)
    return normalized


def build_analyzer_from_components(
    components: List[dict],
    board_schema: BoardSchema | None = None,
) -> tuple[CircuitAnalyzer, List[ComponentInstance]]:
    """统一的结构化组件 -> CircuitAnalyzer 构建入口。

    注意:
    - 这里只接受已经完成 `components[].pins[]` 组装的输入
    - 如果上游仍然只提供 `pin1_logic/pin2_logic`，应先在 S2 完成结构化转换
    """
    schema = board_schema or BoardSchema.default_breadboard()
    normalized_components = normalize_components_for_topology(
        components,
        board_schema=schema,
    )
    analyzer = CircuitAnalyzer(board_schema=schema)
    for comp in normalized_components:
        analyzer.add_component_instance(comp)
    return analyzer, normalized_components


def _normalize_component(
    comp: dict,
    counters: defaultdict[str, int],
    board_schema: BoardSchema,
) -> ComponentInstance | None:
    if not comp.get("pins"):
        component_id = comp.get("component_id") or comp.get("class_name") or comp.get("component_type") or "UNKNOWN"
        raise ValueError(
            f"Component {component_id} missing structured pins[]. "
            "S2 output must provide component-centered pin assignments before topology build."
        )
    return _from_structured_component(comp, counters, board_schema)


def _from_structured_component(
    comp: dict,
    counters: defaultdict[str, int],
    board_schema: BoardSchema,
) -> ComponentInstance | None:
    # 新链路优先: 只要 S2 已经给出了 `pins[]`, 后端后续都以结构化 pin 为准。
    component_type = str(comp.get("component_type") or comp.get("class_name") or "UNKNOWN")
    component_id = comp.get("component_id") or _next_component_id(component_type, counters)
    package_type = str(comp.get("package_type") or _default_package_type(component_type))
    pin_schema_id = str(comp.get("pin_schema_id") or "")
    pins_payload = comp.get("pins") or []

    if component_type == "IC" and pin_schema_id == "dip8_anchor_pair":
        return _from_structured_ic_anchor_pair(
            comp,
            component_id=str(component_id),
            package_type=package_type,
            board_schema=board_schema,
        )

    pins: List[PinAssignment] = []
    for idx, pin in enumerate(pins_payload, start=1):
        hole_id = pin.get("hole_id")
        if not hole_id and pin.get("logic_loc"):
            hole_id = board_schema.logic_loc_to_hole_id(tuple(pin["logic_loc"]))
        if not hole_id:
            continue
        pin_name = str(pin.get("pin_name") or f"pin{idx}")
        pins.append(
            PinAssignment(
                pin_id=int(pin.get("pin_id") or idx),
                pin_name=pin_name,
                hole_id=board_schema.normalize_hole_id(str(hole_id)),
                electrical_node_id=pin.get("electrical_node_id"),
                observations=_pin_observations_from_payload(pin.get("observations", [])),
                confidence=float(pin.get("confidence", comp.get("confidence", 1.0))),
                is_ambiguous=bool(pin.get("is_ambiguous", False)),
                metadata=_pin_metadata_from_payload(pin),
            )
        )

    if not pins:
        return None

    return ComponentInstance(
        component_id=str(component_id),
        component_type=component_type,
        package_type=package_type,
        part_subtype=str(comp.get("part_subtype") or ""),
        polarity=str(comp.get("polarity") or "none"),
        orientation=float(comp.get("orientation", 0.0)),
        symmetry_group=[list(group) for group in comp.get("symmetry_group", [])],
        pins=pins,
        confidence=float(comp.get("confidence", 1.0)),
        metadata={
            "source": "structured",
            "bbox": comp.get("bbox"),
        },
    )


def _from_structured_ic_anchor_pair(
    comp: dict,
    component_id: str,
    package_type: str,
    board_schema: BoardSchema,
) -> ComponentInstance | None:
    pins_payload = comp.get("pins") or []
    if len(pins_payload) < 2:
        return None

    logic_locs = []
    for pin in pins_payload[:2]:
        if pin.get("logic_loc"):
            logic_locs.append(tuple(pin["logic_loc"]))
            continue
        if pin.get("hole_id"):
            logic_loc = board_schema.hole_id_to_logic_loc(str(pin["hole_id"]))
            if logic_loc:
                logic_locs.append(logic_loc)
    if len(logic_locs) < 2:
        return None

    dip8_pin_locs = build_dip8_pin_locs(
        pin1=(str(logic_locs[0][0]), str(logic_locs[0][1])),
        pin2=(str(logic_locs[1][0]), str(logic_locs[1][1])),
    )

    pins: List[PinAssignment] = []
    for idx, loc in enumerate(dip8_pin_locs, start=1):
        hole_id = board_schema.logic_loc_to_hole_id(loc)
        pins.append(
            PinAssignment(
                pin_id=idx,
                pin_name=UA741_PIN_ROLES[idx - 1] if idx - 1 < len(UA741_PIN_ROLES) else f"pin{idx}",
                hole_id=hole_id,
                confidence=float(comp.get("confidence", 1.0)),
            )
        )

    return ComponentInstance(
        component_id=component_id,
        component_type="IC",
        package_type=package_type or "dip8",
        part_subtype=str(comp.get("part_subtype") or ""),
        polarity=str(comp.get("polarity") or "none"),
        orientation=float(comp.get("orientation", 0.0)),
        symmetry_group=[list(group) for group in comp.get("symmetry_group", [])],
        pins=pins,
        confidence=float(comp.get("confidence", 1.0)),
        metadata={
            "source": "structured_anchor_pair",
            "bbox": comp.get("bbox"),
            "pin_schema_id": comp.get("pin_schema_id"),
        },
    )

def _default_package_type(component_type: str) -> str:
    c = component_type.lower()
    if c == "resistor":
        return "axial_2pin"
    if c == "wire":
        return "jumper_wire_2pin"
    if c == "led":
        return "led_2pin"
    if c == "diode":
        return "diode_2pin"
    if c == "capacitor":
        return "capacitor_2pin"
    if c == "potentiometer":
        return "potentiometer_3pin"
    if c == "ic":
        return "dip8"
    return "generic"


def _default_symmetry_group(component_type: str) -> List[List[str]]:
    c = component_type.lower()
    if c in ("resistor", "wire", "capacitor"):
        return [["pin1", "pin2"]]
    return []


def _next_component_id(component_type: str, counters: defaultdict[str, int]) -> str:
    c = component_type.lower()
    prefix = _TYPE_PREFIX.get(c, c[:3].upper() or "CMP")
    counters[c] += 1
    return f"{prefix}{counters[c]}"


def _pin_observations_from_payload(payload: list[dict]) -> List[PinObservation]:
    observations: List[PinObservation] = []
    for item in payload:
        keypoint = item.get("keypoint")
        observations.append(
            PinObservation(
                view_id=str(item.get("view_id") or "top"),
                keypoint=tuple(keypoint) if keypoint else None,
                visibility=int(item.get("visibility", 0)),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return observations


def _pin_metadata_from_payload(pin: dict) -> dict:
    """保留会被后续 guidance / agent 用到, 但当前 analyzer 不直接消费的 pin 辅助信息。"""
    metadata = dict(pin.get("metadata") or {})
    for key in (
        "candidate_hole_ids",
        "candidate_node_ids",
        "candidate_count",
        "primary_visibility",
        "visible_view_ids",
        "observation_count",
        "ambiguity_reasons",
        "is_anchor_pin",
    ):
        if key in pin:
            metadata[key] = pin.get(key)
    return metadata
