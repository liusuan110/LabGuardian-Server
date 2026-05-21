from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from app.domain.logical_reference import (
    current_netlist_v2_to_graph,
    logical_reference_to_graph,
    normalize_net_role,
    normalize_pin_role,
    normalize_role_label,
)


@dataclass(frozen=True)
class LogicalNet:
    source_id: str
    canonical_name: str
    role: str
    role_label: str
    aliases: tuple[str, ...]
    pins: tuple[dict[str, Any], ...]
    member_node_ids: tuple[str, ...]
    member_hole_ids: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        payload["pins"] = [dict(pin) for pin in self.pins]
        payload["member_node_ids"] = list(self.member_node_ids)
        payload["member_hole_ids"] = list(self.member_hole_ids)
        return payload


def normalize_current_netlist(
    netlist_v2: dict[str, Any],
    *,
    reference_circuit: dict[str, Any] | None = None,
    net_alias_assignments: list[Any] | None = None,
    net_merge_assignments: list[Any] | None = None,
) -> dict[str, Any]:
    """Normalize current netlist nets in-place.

    The physical electrical_net_id remains the source id. Stable logical names are
    stored as canonical_name / aliases and propagated to graph/report consumers.
    """
    warnings: list[dict[str, Any]] = []
    applied_aliases: list[dict[str, Any]] = []
    applied_merges: list[dict[str, Any]] = []
    inferred_aliases: list[dict[str, Any]] = []

    if not isinstance(netlist_v2, dict):
        return {
            "logical_nets": [],
            "warnings": [{"warning_code": "NETLIST_INVALID", "message": "netlist_v2 must be a dict"}],
            "applied_aliases": [],
            "applied_merges": [],
            "inferred_aliases": [],
        }

    _ensure_net_defaults(netlist_v2)
    merge_warnings, applied_merges = apply_net_merge_assignments(netlist_v2, net_merge_assignments)
    warnings.extend(merge_warnings)

    alias_warnings, applied_aliases = apply_net_alias_assignments(netlist_v2, net_alias_assignments)
    warnings.extend(alias_warnings)

    if isinstance(reference_circuit, dict) and reference_circuit.get("format") == "logical_reference_v1":
        inferred_aliases = infer_net_aliases_from_reference(netlist_v2, reference_circuit)

    logical_nets = build_logical_nets(netlist_v2)
    netlist_v2["logical_nets"] = [item.to_dict() for item in logical_nets]
    netlist_v2["net_normalization"] = {
        "warnings": warnings,
        "applied_aliases": applied_aliases,
        "applied_merges": applied_merges,
        "inferred_aliases": inferred_aliases,
    }
    return {
        "logical_nets": netlist_v2["logical_nets"],
        "warnings": warnings,
        "applied_aliases": applied_aliases,
        "applied_merges": applied_merges,
        "inferred_aliases": inferred_aliases,
    }


def apply_net_alias_assignments(
    netlist_v2: dict[str, Any],
    assignments: list[Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    if not assignments:
        return warnings, applied

    indexes = _build_net_indexes(netlist_v2)
    for assignment in assignments:
        raw = _assignment_dict(assignment)
        canonical_name = normalize_role_label(raw.get("canonical_name") or raw.get("role_label"))
        aliases = _normalize_aliases(raw.get("aliases"))
        if not canonical_name and aliases:
            canonical_name = aliases[0]
        if not canonical_name:
            warnings.append(
                {
                    "warning_code": "ALIAS_INVALID",
                    "message": "net alias assignment requires canonical_name or aliases",
                    "assignment": raw,
                }
            )
            continue

        net_obj, resolved_by = _resolve_net(raw, indexes)
        if net_obj is None:
            warnings.append(
                {
                    "warning_code": "ALIAS_TARGET_NOT_FOUND",
                    "message": f"无法为网络别名 {canonical_name} 找到对应电气网络",
                    "assignment": raw,
                }
            )
            continue

        _set_net_canonical(
            net_obj,
            canonical_name,
            aliases=aliases,
            source=str(raw.get("source") or "manual_net_alias"),
            role=raw.get("role"),
        )
        applied.append(
            {
                "electrical_net_id": str(net_obj.get("electrical_net_id") or net_obj.get("net_id") or ""),
                "canonical_name": canonical_name,
                "aliases": list(dict.fromkeys([canonical_name, *aliases])),
                "source": raw.get("source") or "manual_net_alias",
                "resolved_by": resolved_by,
            }
        )
    return warnings, applied


def apply_net_merge_assignments(
    netlist_v2: dict[str, Any],
    assignments: list[Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    if not assignments:
        return warnings, applied

    for assignment in assignments:
        raw = _assignment_dict(assignment)
        source_net_ids = [str(item) for item in raw.get("source_net_ids", []) or [] if str(item)]
        if len(source_net_ids) < 2:
            warnings.append(
                {
                    "warning_code": "MERGE_INVALID",
                    "message": "net merge assignment requires at least two source_net_ids",
                    "assignment": raw,
                }
            )
            continue

        nets_by_id = _nets_by_id(netlist_v2)
        missing = [net_id for net_id in source_net_ids if net_id not in nets_by_id]
        if missing:
            warnings.append(
                {
                    "warning_code": "MERGE_TARGET_NOT_FOUND",
                    "message": f"待合并网络不存在: {', '.join(missing)}",
                    "assignment": raw,
                }
            )
            continue

        keep_id = source_net_ids[0]
        keep = nets_by_id[keep_id]
        merged_ids = [net_id for net_id in source_net_ids[1:]]
        for merge_id in merged_ids:
            other = nets_by_id[merge_id]
            _merge_net_fields(keep, other)
            _rewrite_component_pin_nets(netlist_v2, from_net=merge_id, to_net=keep_id)

        netlist_v2["nets"] = [
            net
            for net in netlist_v2.get("nets", []) or []
            if str(net.get("electrical_net_id") or net.get("net_id") or "") not in set(merged_ids)
        ]
        merged_source_ids = list(dict.fromkeys([
            *keep.get("merged_source_ids", []),
            *source_net_ids,
        ]))
        keep["merged_source_ids"] = merged_source_ids
        if raw.get("target_canonical_name"):
            _set_net_canonical(
                keep,
                normalize_role_label(raw.get("target_canonical_name")),
                aliases=[],
                source=str(raw.get("source") or "manual_net_merge"),
            )
        keep["canonical_source"] = raw.get("source") or "manual_net_merge"

        applied.append(
            {
                "kept_net_id": keep_id,
                "merged_source_ids": source_net_ids,
                "target_canonical_name": keep.get("canonical_name"),
                "source": raw.get("source") or "manual_net_merge",
            }
        )
    return warnings, applied


def infer_net_aliases_from_reference(
    netlist_v2: dict[str, Any],
    reference_circuit: dict[str, Any],
) -> list[dict[str, Any]]:
    """**[DEPRECATED · superseded by Phase E `propagate_canonical_via_alignment`]**

    Original isomorphism-based alias inference. Same failure mode as
    ``role_inference._infer_current_net_roles_from_reference`` —
    ``GraphMatcher.is_isomorphic()`` returns ``False`` whenever cur has
    extra jumper wires (which is always on real student boards).
    Kept for now because ``normalize_current_netlist`` still calls it as a
    second pass after Phase E propagation; can be removed in the same
    cleanup pass as the role_inference twin.
    """
    current_for_match = copy.deepcopy(netlist_v2)
    _strip_auto_aliases(current_for_match)
    try:
        reference_graph = logical_reference_to_graph(reference_circuit)
        current_graph = current_netlist_v2_to_graph(current_for_match)
    except Exception:
        return []

    matcher = GraphMatcher(
        reference_graph,
        current_graph,
        node_match=_node_match_for_alias_inference,
        edge_match=_edge_match_for_alias_inference,
    )
    if not matcher.is_isomorphic():
        return []

    selected: list[dict[str, Any]] | None = None
    checked = 0
    for mapping in matcher.isomorphisms_iter():
        checked += 1
        inferences = _alias_inferences_for_mapping(mapping, reference_graph, current_graph)
        if not inferences:
            return []
        if selected is None:
            selected = inferences
        elif _alias_signature(selected) != _alias_signature(inferences):
            return []
        if checked >= 50:
            break

    if not selected:
        return []

    nets_by_id = _nets_by_id(netlist_v2)
    applied: list[dict[str, Any]] = []
    for item in selected:
        net = nets_by_id.get(str(item.get("source_id") or ""))
        if not net or _has_manual_alias(net):
            continue
        _set_net_canonical(
            net,
            str(item.get("canonical_name") or ""),
            aliases=[],
            source="inferred_from_reference",
            role=item.get("role"),
        )
        net["inferred_reference_net"] = item.get("reference_net")
        applied.append(item)
    return applied


def build_logical_nets(netlist_v2: dict[str, Any]) -> list[LogicalNet]:
    pin_index = _pins_by_net(netlist_v2)
    out: list[LogicalNet] = []
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        source_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if not source_id:
            continue
        role_label = normalize_role_label(net.get("role_label") or net.get("power_role"))
        canonical_name = str(net.get("canonical_name") or role_label or source_id)
        role = normalize_net_role(net.get("role") or net.get("manual_role") or role_label)
        aliases = tuple(_normalize_aliases([canonical_name, role_label, source_id, *list(net.get("aliases") or [])]))
        out.append(
            LogicalNet(
                source_id=source_id,
                canonical_name=canonical_name,
                role=role,
                role_label=role_label,
                aliases=aliases,
                pins=tuple(pin_index.get(source_id, [])),
                member_node_ids=tuple(str(item) for item in net.get("member_node_ids", []) or []),
                member_hole_ids=tuple(str(item) for item in net.get("member_hole_ids", []) or []),
                source=str(net.get("canonical_source") or net.get("role_source") or "netlist_v2"),
            )
        )
    return out


def net_display_name(net: dict[str, Any] | None, fallback: Any = "") -> str:
    if not isinstance(net, dict):
        return str(fallback or "")
    return str(
        net.get("canonical_name")
        or net.get("role_label")
        or net.get("power_role")
        or net.get("electrical_net_id")
        or net.get("net_id")
        or fallback
        or ""
    )


def _ensure_net_defaults(netlist_v2: dict[str, Any]) -> None:
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        source_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if not source_id:
            continue
        role_label = normalize_role_label(net.get("role_label") or net.get("power_role"))
        if role_label and not net.get("role_label"):
            net["role_label"] = role_label
        if not net.get("canonical_name"):
            net["canonical_name"] = role_label or str(net.get("name") or net.get("net_name") or source_id)
        aliases = _normalize_aliases([net.get("canonical_name"), role_label, source_id, *list(net.get("aliases") or [])])
        net["aliases"] = aliases


def _set_net_canonical(
    net: dict[str, Any],
    canonical_name: str,
    *,
    aliases: list[str],
    source: str,
    role: Any = None,
) -> None:
    canonical = normalize_role_label(canonical_name) or str(canonical_name or "")
    if not canonical:
        return
    net["canonical_name"] = canonical
    net["role_label"] = canonical
    net["aliases"] = _normalize_aliases([canonical, *aliases, *list(net.get("aliases") or [])])
    net["canonical_source"] = source
    if source.startswith("manual"):
        net["alias_source"] = source
    if role:
        net["role"] = normalize_net_role(role)
    elif not net.get("manual_role"):
        inferred_role = normalize_net_role(canonical)
        if inferred_role != "signal":
            net["role"] = inferred_role
            net["role_source"] = source
    if normalize_net_role(net.get("role") or canonical) == "power":
        net["power_role"] = canonical if canonical in {"VCC", "VDD", "VEE", "VSS"} else "VCC"
    elif normalize_net_role(net.get("role") or canonical) == "ground":
        net["power_role"] = "GND"


def _merge_net_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("member_node_ids", "member_hole_ids", "labels", "aliases"):
        target[key] = list(dict.fromkeys([*list(target.get(key) or []), *list(source.get(key) or [])]))
    target["merged_role_labels"] = _normalize_aliases(
        [
            *list(target.get("merged_role_labels") or []),
            target.get("role_label"),
            target.get("power_role"),
            source.get("role_label"),
            source.get("power_role"),
        ]
    )
    if not target.get("role") and source.get("role"):
        target["role"] = source["role"]
    if not target.get("role_label") and source.get("role_label"):
        target["role_label"] = source["role_label"]
    if not target.get("canonical_name") and source.get("canonical_name"):
        target["canonical_name"] = source["canonical_name"]
    if not target.get("power_role") and source.get("power_role"):
        target["power_role"] = source["power_role"]


def _rewrite_component_pin_nets(netlist_v2: dict[str, Any], *, from_net: str, to_net: str) -> None:
    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        for pin in comp.get("pins", []) or []:
            if isinstance(pin, dict) and str(pin.get("electrical_net_id") or "") == from_net:
                pin["electrical_net_id"] = to_net
                metadata = dict(pin.get("metadata") or {})
                metadata["merged_from_net_id"] = from_net
                pin["metadata"] = metadata


def _build_net_indexes(netlist_v2: dict[str, Any]) -> dict[str, dict[Any, dict[str, Any]]]:
    nets_by_id = _nets_by_id(netlist_v2)
    by_hole: dict[Any, dict[str, Any]] = {}
    by_node: dict[Any, dict[str, Any]] = {}
    by_comp_pin: dict[Any, dict[str, Any]] = {}
    for net in nets_by_id.values():
        for hole_id in net.get("member_hole_ids", []) or []:
            by_hole[str(hole_id)] = net
        for node_id in net.get("member_node_ids", []) or []:
            by_node[str(node_id)] = net
    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        comp_id = str(comp.get("component_id") or "")
        for pin in comp.get("pins", []) or []:
            if not isinstance(pin, dict):
                continue
            net_id = str(pin.get("electrical_net_id") or "")
            pin_name = str(pin.get("pin_name") or "")
            if comp_id and pin_name and net_id in nets_by_id:
                by_comp_pin[(comp_id, pin_name)] = nets_by_id[net_id]
    return {"id": nets_by_id, "hole": by_hole, "node": by_node, "comp_pin": by_comp_pin}


def _resolve_net(
    raw: dict[str, Any],
    indexes: dict[str, dict[Any, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    """**R11 follow-up fix (2026-05-19)** — try every present locator and
    return the first that successfully maps to a real net.

    Previously this function early-returned on the **first non-empty
    field** (electrical_net_id checked first), so a frontend that sent
    a stale or synthetic ``electrical_net_id`` (e.g. ``LOCAL_NET_0``
    introduced by the R11 frontend recompute) would short-circuit
    here and the whole annotation would be silently dropped — even
    though the same payload also carried a perfectly valid
    ``hole_id`` / ``component_id`` + ``pin_name`` / ``electrical_node_id``.

    Order of preference is unchanged (electrical_net_id is still tried
    first when present); the only change is the fallback path.
    """

    candidates: list[tuple[dict[str, Any] | None, str]] = []
    if raw.get("electrical_net_id"):
        candidates.append((
            indexes["id"].get(str(raw["electrical_net_id"])),
            "electrical_net_id",
        ))
    if raw.get("component_id") and raw.get("pin_name"):
        candidates.append((
            indexes["comp_pin"].get(
                (str(raw["component_id"]), str(raw["pin_name"]))
            ),
            "component_pin",
        ))
    if raw.get("hole_id"):
        candidates.append((
            indexes["hole"].get(str(raw["hole_id"])),
            "hole_id",
        ))
    if raw.get("electrical_node_id"):
        candidates.append((
            indexes["node"].get(str(raw["electrical_node_id"])),
            "electrical_node_id",
        ))
    for net, source in candidates:
        if net is not None:
            return net, source
    # Nothing resolved — return the first attempt's miss tag for
    # downstream warning consistency (callers log the resolver they
    # tried first; preserves old log shape on outright misses).
    if candidates:
        return None, candidates[0][1]
    return None, ""


def _nets_by_id(netlist_v2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if net_id:
            out[net_id] = net
    return out


def _pins_by_net(netlist_v2: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        comp_id = str(comp.get("component_id") or "")
        comp_type = str(comp.get("component_type") or comp.get("type") or "")
        for pin in comp.get("pins", []) or []:
            if not isinstance(pin, dict):
                continue
            net_id = str(pin.get("electrical_net_id") or "")
            if not net_id:
                continue
            out.setdefault(net_id, []).append(
                {
                    "component_id": comp_id,
                    "component_type": comp_type,
                    "pin_name": pin.get("pin_name") or pin.get("pin"),
                    "hole_id": pin.get("hole_id"),
                    "electrical_node_id": pin.get("electrical_node_id"),
                }
            )
    return out


def _strip_auto_aliases(netlist_v2: dict[str, Any]) -> None:
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict) or _has_manual_alias(net):
            continue
        source_id = str(net.get("electrical_net_id") or net.get("net_id") or "")
        if net.get("canonical_source") == "inferred_from_reference":
            net.pop("canonical_name", None)
            net.pop("role_label", None)
            net.pop("role", None)
            net.pop("power_role", None)
            net["aliases"] = [source_id] if source_id else []


def _has_manual_alias(net: dict[str, Any]) -> bool:
    source = str(net.get("alias_source") or net.get("canonical_source") or "")
    return source.startswith("manual")


def _normalize_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    aliases: list[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if not text:
            continue
        aliases.append(normalize_role_label(text) or text)
    return list(dict.fromkeys(aliases))


def _assignment_dict(assignment: Any) -> dict[str, Any]:
    if isinstance(assignment, dict):
        return dict(assignment)
    if hasattr(assignment, "model_dump"):
        return assignment.model_dump()
    if hasattr(assignment, "__dict__"):
        return dict(assignment.__dict__)
    return {}


def _node_match_for_alias_inference(ref_data: dict[str, Any], cur_data: dict[str, Any]) -> bool:
    if ref_data.get("kind") != cur_data.get("kind"):
        return False
    if ref_data.get("kind") == "comp":
        left = str(ref_data.get("ctype"))
        right = str(cur_data.get("ctype"))
        return left == right or {left, right} <= {"Capacitor", "CapacitorCeramic"}
    canonical_source = str(cur_data.get("canonical_source") or "")
    if cur_data.get("role_source") == "manual_role" or canonical_source.startswith("manual"):
        ref_label = normalize_role_label(ref_data.get("role_label"))
        cur_label = normalize_role_label(cur_data.get("role_label") or cur_data.get("canonical_name"))
        return not ref_label or not cur_label or ref_label == cur_label
    return True


def _edge_match_for_alias_inference(left: dict[str, Any], right: dict[str, Any]) -> bool:
    comp_type = left.get("comp_type") or right.get("comp_type")
    pin_left = normalize_pin_role(comp_type, left.get("pin_role") or left.get("pin"))
    pin_right = normalize_pin_role(comp_type, right.get("pin_role") or right.get("pin"))
    if comp_type in {"Resistor", "Capacitor", "CapacitorCeramic", "Wire"}:
        return True
    if comp_type == "Potentiometer" and {pin_left, pin_right} == {"terminal_a", "terminal_b"}:
        return True
    return pin_left == pin_right


def _alias_inferences_for_mapping(
    mapping: dict[Any, Any],
    reference_graph: nx.Graph,
    current_graph: nx.Graph,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ref_node, cur_node in mapping.items():
        ref_data = reference_graph.nodes.get(ref_node, {})
        cur_data = current_graph.nodes.get(cur_node, {})
        if ref_data.get("kind") != "net" or cur_data.get("kind") != "net":
            continue
        current_source = str(cur_data.get("source_id") or "")
        ref_source = str(ref_data.get("source_id") or "")
        ref_label = normalize_role_label(ref_data.get("role_label") or ref_source)
        if not current_source or not ref_label:
            continue
        out.append(
            {
                "source_id": current_source,
                "canonical_name": ref_label,
                "role": normalize_net_role(ref_data.get("role") or ref_label),
                "role_label": ref_label,
                "reference_net": ref_source,
                "source": "inferred_from_reference",
            }
        )
    return sorted(out, key=lambda item: str(item.get("source_id") or ""))


def _alias_signature(inferences: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            item.get("source_id"),
            item.get("canonical_name"),
            item.get("role"),
            item.get("reference_net"),
        )
        for item in inferences
    )
