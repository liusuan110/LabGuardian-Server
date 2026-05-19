from __future__ import annotations

import re
from typing import Any

from app.domain.board_schema import BoardSchema
from app.domain.logical_reference import normalize_net_role, normalize_role_label

_ROW_NODE_RE = re.compile(r"^ROW_(\d+)(?:_([LR]))?$", re.IGNORECASE)


def apply_net_role_assignments(
    netlist_v2: dict[str, Any],
    assignments: list[Any] | None,
    *,
    port_annotations: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply port and manual net role assignments to a netlist_v2 in place.

    Returns (warnings, applied_records).
    """
    warnings: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    normalized_assignments = [
        *_port_annotations_to_assignments(port_annotations),
        *[_assignment_dict(item) for item in assignments or []],
    ]
    if not normalized_assignments:
        return warnings, applied

    indexes = _build_net_indexes(netlist_v2)

    for raw in normalized_assignments:
        warning, record = _apply_single_role_assignment(netlist_v2, indexes, raw)
        if warning:
            warnings.append(warning)
        if record:
            applied.append(record)

    return warnings, applied


def _build_net_indexes(netlist_v2: dict[str, Any]) -> dict[str, Any]:
    netlist_nets: dict[str, dict[str, Any]] = {}
    netlist_nets_by_hole: dict[str, dict[str, Any]] = {}
    netlist_nets_by_node: dict[str, dict[str, Any]] = {}
    netlist_nets_by_comp_pin: dict[tuple[str, str], dict[str, Any]] = {}
    board_schema = BoardSchema.default_breadboard()

    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if not net_id:
            continue
        netlist_nets[net_id] = net
        for hole_id in net.get("member_hole_ids", []) or []:
            netlist_nets_by_hole[str(hole_id)] = net
        for node_id in net.get("member_node_ids", []) or []:
            netlist_nets_by_node[str(node_id)] = net
            for hole_id in _holes_for_node_id(str(node_id), board_schema):
                netlist_nets_by_hole.setdefault(hole_id, net)

    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        comp_id = str(comp.get("component_id") or "")
        for pin in comp.get("pins", []) or []:
            if not isinstance(pin, dict):
                continue
            pin_name = str(pin.get("pin_name") or "")
            pin_net_id = str(pin.get("electrical_net_id") or "")
            if comp_id and pin_name and pin_net_id:
                netlist_nets_by_comp_pin[(comp_id, pin_name)] = netlist_nets.get(
                    pin_net_id,
                    {"electrical_net_id": pin_net_id},
                )
    return {
        "by_id": netlist_nets,
        "by_hole": netlist_nets_by_hole,
        "by_node": netlist_nets_by_node,
        "by_comp_pin": netlist_nets_by_comp_pin,
    }


def _apply_single_role_assignment(
    netlist_v2: dict[str, Any],
    indexes: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    role = str(raw.get("role") or "")
    source = raw.get("source") or "manual_netlist_select"
    explicit_label = raw.get("role_label") or raw.get("label")
    if explicit_label is None and source != "port_annotation":
        explicit_label = role
    role_label = normalize_role_label(explicit_label or "")
    normalized_role = normalize_net_role(role or role_label)
    if normalized_role == "signal":
        return (
            {
                "warning_code": "ROLE_INVALID",
                "message": f"非法或未知的网络角色: {role}",
                "assignment": raw,
            },
            None,
        )

    target_net_id, resolved_by, warning = _resolve_target_net_id(indexes, raw)
    if warning:
        return warning, None
    if not target_net_id:
        return (
            {
                "warning_code": "ROLE_TARGET_NOT_CONNECTED",
                "message": (
                    f"无法为角色指定 {role} 找到对应电气网络，"
                    "该孔位/节点可能未连接到任何元件或导线（空孔）"
                ),
                "assignment": raw,
            },
            None,
        )

    net_obj = indexes["by_id"].get(target_net_id)
    if net_obj is None:
        return (
            {
                "warning_code": "ROLE_TARGET_NOT_FOUND",
                "message": f"角色指定目标网络 {target_net_id} 在 netlist_v2 中不存在",
                "assignment": raw,
            },
            None,
        )

    net_obj["role"] = normalized_role
    net_obj["manual_role"] = normalized_role
    net_obj["role_label"] = role_label
    net_obj["role_source"] = source
    if normalized_role == "power":
        net_obj["power_role"] = role_label if role_label in {"VCC", "VEE", "VDD", "VSS"} else "VCC"
    elif normalized_role == "ground":
        net_obj["power_role"] = "GND"

    record: dict[str, Any] = {
        "role": normalized_role,
        "role_label": role_label,
        "electrical_net_id": target_net_id,
        "source": source,
        "resolved_by": resolved_by,
    }
    for key in (
        "hole_id",
        "component_id",
        "pin_name",
        "electrical_node_id",
        "x_image",
        "y_image",
    ):
        if raw.get(key) is not None:
            record[key] = raw[key]
    return None, record


def _resolve_target_net_id(
    indexes: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[str | None, str, dict[str, Any] | None]:
    attempted_electrical_net_id: str | None = None
    if raw.get("electrical_net_id"):
        target_net_id = str(raw["electrical_net_id"])
        if target_net_id in indexes["by_id"]:
            return target_net_id, "electrical_net_id", None
        attempted_electrical_net_id = target_net_id
        net_obj = _resolve_locator_as_board_node_or_hole(indexes, target_net_id)
        if net_obj:
            return str(net_obj.get("electrical_net_id") or ""), "electrical_net_id_locator", None

    if raw.get("component_id") and raw.get("pin_name"):
        net_obj = indexes["by_comp_pin"].get((str(raw["component_id"]), str(raw["pin_name"])))
        if net_obj:
            return str(net_obj.get("electrical_net_id") or ""), "component_pin", None

    if raw.get("hole_id"):
        hole_id = str(raw["hole_id"])
        net_obj = indexes["by_hole"].get(hole_id)
        if not net_obj:
            normalized_hole = BoardSchema.default_breadboard().normalize_hole_id(hole_id)
            net_obj = indexes["by_hole"].get(normalized_hole)
        if net_obj:
            return str(net_obj.get("electrical_net_id") or ""), "hole_id", None

    if raw.get("electrical_node_id"):
        node_id = str(raw["electrical_node_id"])
        net_obj = indexes["by_node"].get(node_id)
        if not net_obj:
            net_obj = _resolve_row_node_prefix(indexes["by_node"], node_id)
        if net_obj:
            return str(net_obj.get("electrical_net_id") or ""), "electrical_node_id", None

    if attempted_electrical_net_id:
        return (
            attempted_electrical_net_id,
            "electrical_net_id",
            {
                "warning_code": "ROLE_TARGET_NOT_FOUND",
                "message": f"指定的电气网络 {attempted_electrical_net_id} 在当前 netlist 中不存在",
                "assignment": raw,
            },
        )

    return None, "", None


def _resolve_locator_as_board_node_or_hole(
    indexes: dict[str, Any],
    value: str,
) -> dict[str, Any] | None:
    """Treat a non-NET target value as a breadboard locator.

    Some port-annotation clicks send ROW_16_R / ROW_21_L / H24-like board
    primitives in ``target.electrical_net_id``. They are not final NET_xxx ids,
    so resolve them through the same node/hole indexes before declaring a miss.
    """
    locator = str(value or "")
    if not locator:
        return None

    net_obj = indexes["by_node"].get(locator)
    if not net_obj:
        net_obj = _resolve_row_node_prefix(indexes["by_node"], locator)
    if net_obj:
        return net_obj

    net_obj = indexes["by_hole"].get(locator)
    if not net_obj:
        normalized_hole = BoardSchema.default_breadboard().normalize_hole_id(locator)
        net_obj = indexes["by_hole"].get(normalized_hole)
    return net_obj


def _holes_for_node_id(node_id: str, board_schema: BoardSchema) -> list[str]:
    """Expand a board electrical node to every hole in that node.

    S3 stores occupied holes only, but manual role selection often clicks an
    empty hole in the same breadboard row. Indexing all holes for each occupied
    ROW/TRACK node lets ROW_x / hole_id resolve back to the existing NET_x.
    """
    normalized = str(node_id or "")
    if not normalized:
        return []
    return [
        hole_id
        for hole_id, spec in board_schema.holes.items()
        if spec.electrical_node_id == normalized
    ]


def _resolve_row_node_prefix(
    nodes: dict[str, dict[str, Any]],
    raw_node_id: str,
) -> dict[str, Any] | None:
    normalized = str(raw_node_id or "").strip().upper()
    match = _ROW_NODE_RE.match(normalized)
    if not match or match.group(2):
        return None
    prefix = f"ROW_{match.group(1)}_"
    matches = [
        net
        for node_id, net in nodes.items()
        if str(node_id).upper().startswith(prefix)
    ]
    unique_by_id = {
        str(net.get("electrical_net_id") or net.get("net_id") or ""): net
        for net in matches
    }
    unique_by_id.pop("", None)
    if len(unique_by_id) == 1:
        return next(iter(unique_by_id.values()))
    return None


def _port_annotations_to_assignments(annotations: list[Any] | None) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for annotation in annotations or []:
        raw = _assignment_dict(annotation)
        target = raw.get("target") or {}
        if hasattr(target, "model_dump"):
            target = target.model_dump()
        target = dict(target) if isinstance(target, dict) else {}
        assignments.append(
            {
                **target,
                "role": raw.get("role"),
                "role_label": raw.get("label") or raw.get("role_label") or "",
                "source": raw.get("source") or "port_annotation",
            }
        )
    return assignments


def _assignment_dict(assignment: Any) -> dict[str, Any]:
    if isinstance(assignment, dict):
        return dict(assignment)
    if hasattr(assignment, "model_dump"):
        return assignment.model_dump()
    return {
        key: getattr(assignment, key)
        for key in (
            "role",
            "role_label",
            "source",
            "hole_id",
            "component_id",
            "pin_name",
            "electrical_net_id",
            "electrical_node_id",
            "x_image",
            "y_image",
        )
        if hasattr(assignment, key)
    }
