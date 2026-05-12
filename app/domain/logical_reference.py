from __future__ import annotations

from typing import Any

import networkx as nx

from app.domain.circuit import norm_component_type


VALID_NET_ROLES = {"signal", "ground", "power", "input", "output"}
# These labels are only strict comparison hints for input/output ports. Internal
# signal labels remain topology-only in the logical graph matcher.
CRITICAL_ROLE_LABELS = {"UI1", "UI2", "UO1", "UO2", "VCC", "VEE", "GND"}


def normalize_component_type(value: Any) -> str:
    return norm_component_type(str(value or "UNKNOWN"))


def normalize_net_role(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"gnd", "ground", "0v", "earth"}:
        return "ground"
    if raw in {
        "vcc",
        "vdd",
        "vee",
        "vss",
        "power",
        "pwr",
        "vin_power",
        "supply",
        "negative_supply",
        "+",
        "+5v",
        "+3v3",
        "-5v",
        "-12v",
    }:
        return "power"
    if raw in {"input", "in", "vin", "ui1", "ui2"}:
        return "input"
    if raw in {"output", "out", "vout", "uo1", "uo2"}:
        return "output"
    if raw in VALID_NET_ROLES:
        return raw
    return "signal"


def normalize_role_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.upper()


def normalize_pin_role(component_type: Any, pin: Any) -> str:
    ctype = normalize_component_type(component_type)
    if isinstance(pin, dict):
        raw = (
            pin.get("polarity_role")
            or pin.get("pin_display_name")
            or pin.get("pin_name")
            or pin.get("pin")
            or ""
        )
    else:
        raw = pin
    value = str(raw or "").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    if ctype == "Transistor":
        if value in {"e", "emitter"}:
            return "emitter"
        if value in {"b", "base"}:
            return "base"
        if value in {"c", "collector"}:
            return "collector"
    if ctype == "Potentiometer":
        if value in {"w", "wiper", "center", "middle"}:
            return "wiper"
        if value in {"terminal_a", "a", "pin1", "1"}:
            return "terminal_a"
        if value in {"terminal_b", "b", "pin2", "2"}:
            return "terminal_b"
    if ctype in {"LED", "Diode"}:
        if value in {"a", "anode", "positive", "+"}:
            return "anode"
        if value in {"k", "cathode", "negative", "-"}:
            return "cathode"
    if ctype == "CapacitorElectrolytic":
        if value in {"positive", "pos", "+", "anode"}:
            return "positive"
        if value in {"negative", "neg", "-", "cathode"}:
            return "negative"
    if value in {"p1", "1"}:
        return "pin1"
    if value in {"p2", "2"}:
        return "pin2"
    return value


def validate_logical_reference(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("logical reference payload must be a dict")
    if payload.get("format") != "logical_reference_v1":
        raise ValueError("logical reference format must be 'logical_reference_v1'")

    components = payload.get("components")
    nets = payload.get("nets", [])
    if not isinstance(components, list) or not components:
        raise ValueError("logical reference must contain a non-empty components list")
    if not isinstance(nets, list):
        raise ValueError("logical reference nets must be a list when present")

    net_names: set[str] = set()
    for idx, net in enumerate(nets):
        if not isinstance(net, dict):
            raise ValueError(f"nets[{idx}] must be an object")
        net_name = str(net.get("net") or "").strip()
        if not net_name:
            raise ValueError(f"nets[{idx}] must contain net")
        if net_name in net_names:
            raise ValueError(f"duplicate logical net '{net_name}'")
        net_names.add(net_name)

    comp_ids: set[str] = set()
    for idx, comp in enumerate(components):
        if not isinstance(comp, dict):
            raise ValueError(f"components[{idx}] must be an object")
        ref_id = str(comp.get("ref_id") or "").strip()
        if not ref_id:
            raise ValueError(f"components[{idx}] must contain ref_id")
        if ref_id in comp_ids:
            raise ValueError(f"duplicate component ref_id '{ref_id}'")
        comp_ids.add(ref_id)
        if not str(comp.get("type") or "").strip():
            raise ValueError(f"components[{idx}] must contain type")
        pins = comp.get("pins")
        if not isinstance(pins, list) or not pins:
            raise ValueError(f"components[{idx}] must contain a non-empty pins list")
        for pin_idx, pin in enumerate(pins):
            if not isinstance(pin, dict):
                raise ValueError(f"components[{idx}].pins[{pin_idx}] must be an object")
            if not str(pin.get("pin") or "").strip():
                raise ValueError(f"components[{idx}].pins[{pin_idx}] must contain pin")
            if pin.get("nc") is True:
                continue
            pin_net = str(pin.get("net") or "").strip()
            if not pin_net:
                raise ValueError(f"components[{idx}].pins[{pin_idx}] must contain net")
            if net_names and pin_net not in net_names:
                raise ValueError(f"component {ref_id} pin references unknown net '{pin_net}'")


def logical_reference_to_graph(payload: dict[str, Any]) -> nx.Graph:
    validate_logical_reference(payload)

    graph = nx.Graph()
    net_roles: dict[str, str] = {}
    net_role_labels: dict[str, str] = {}
    for net in payload.get("nets", []):
        net_name = str(net["net"])
        label = normalize_role_label(net.get("role_label") or net.get("label") or net.get("net"))
        net_roles[net_name] = normalize_net_role(net.get("role") or label)
        net_role_labels[net_name] = label

    allowed_role_label_swaps = _allowed_role_label_swaps(payload.get("symmetry_groups", []))

    referenced_nets = {
        str(pin["net"])
        for comp in payload.get("components", [])
        for pin in comp.get("pins", [])
        if pin.get("nc") is not True
    }
    for net_name in sorted(referenced_nets | set(net_roles)):
        graph.add_node(
            _ref_net_node_id(net_name),
            kind="net",
            role=net_roles.get(net_name, "signal"),
            role_label=net_role_labels.get(net_name, normalize_role_label(net_name)),
            allowed_role_labels=sorted(allowed_role_label_swaps.get(normalize_role_label(net_name), set())),
            source_id=net_name,
        )

    for comp in payload.get("components", []):
        ref_id = str(comp["ref_id"])
        comp_node = _comp_node_id(ref_id)
        ctype = normalize_component_type(comp.get("type"))
        graph.add_node(
            comp_node,
            kind="comp",
            ctype=ctype,
            source_id=ref_id,
        )
        for pin in comp.get("pins", []):
            if pin.get("nc") is True:
                continue
            net_name = str(pin["net"])
            graph.add_edge(
                comp_node,
                _ref_net_node_id(net_name),
                pin=str(pin.get("pin") or ""),
                pin_role=normalize_pin_role(ctype, pin),
                comp_type=ctype,
            )

    graph.graph["format"] = "logical_reference_v1"
    graph.graph["reference_id"] = payload.get("reference_id")
    graph.graph["name"] = payload.get("name")
    graph.graph["symmetry_groups"] = payload.get("symmetry_groups", [])
    return graph


def current_netlist_v2_to_graph(netlist_v2: dict[str, Any]) -> nx.Graph:
    if not isinstance(netlist_v2, dict):
        raise ValueError("netlist_v2 must be a dict")

    graph = nx.Graph()
    net_roles: dict[str, str] = {}
    net_role_labels: dict[str, str] = {}
    net_role_sources: dict[str, str] = {}
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "").strip()
        if not net_id:
            continue
        canonical_name = normalize_role_label(net.get("canonical_name"))
        role_label = normalize_role_label(net.get("role_label"))
        power_role = normalize_role_label(net.get("power_role"))
        if not role_label and power_role in {"VCC", "VDD", "VEE", "VSS", "GND"}:
            role_label = power_role
        if not role_label and canonical_name and not canonical_name.startswith("NET_"):
            role_label = canonical_name

        manual_role = net.get("manual_role")
        explicit_role = net.get("role")
        if manual_role:
            role = manual_role
            role_source = "manual_role"
        elif role_label:
            role = role_label
            role_source = "role_label"
        elif power_role:
            role = power_role
            role_source = "power_role"
        elif explicit_role:
            role = explicit_role
            role_source = "explicit_role"
        else:
            role = "signal"
            role_source = "default_signal"
        net_roles[net_id] = normalize_net_role(role)
        net_role_labels[net_id] = role_label
        net_role_sources[net_id] = role_source
        graph.add_node(
            _cur_net_node_id(net_id),
            kind="net",
            role=net_roles[net_id],
            source_id=net_id,
            role_label=role_label,
            role_source=role_source,
            canonical_name=canonical_name or role_label or net_id,
            aliases=[normalize_role_label(value) for value in net.get("aliases", []) or []],
            canonical_source=net.get("canonical_source"),
        )

    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        ctype = normalize_component_type(comp.get("component_type") or comp.get("type"))
        if ctype == "Wire":
            continue
        component_id = str(comp.get("component_id") or comp.get("ref_id") or "").strip()
        if not component_id:
            component_id = f"{ctype}_{graph.number_of_nodes()}"
        comp_node = _cur_comp_node_id(component_id)
        graph.add_node(
            comp_node,
            kind="comp",
            ctype=ctype,
            source_id=component_id,
        )

        for pin in comp.get("pins", []) or []:
            if not isinstance(pin, dict):
                continue
            net_id = str(pin.get("electrical_net_id") or pin.get("net_id") or "").strip()
            if not net_id:
                continue
            net_node = _cur_net_node_id(net_id)
            if not graph.has_node(net_node):
                graph.add_node(
                    net_node,
                    kind="net",
                    role=net_roles.get(net_id, "signal"),
                    source_id=net_id,
                    role_label=net_role_labels.get(net_id, ""),
                    role_source=net_role_sources.get(net_id, "default_signal"),
                    canonical_name=net_role_labels.get(net_id, "") or net_id,
                    aliases=[],
                )
            graph.add_edge(
                comp_node,
                net_node,
                pin=str(pin.get("pin_name") or pin.get("pin") or ""),
                pin_role=normalize_pin_role(ctype, pin),
                comp_type=ctype,
            )

    graph.graph["format"] = "netlist_v2_logical_graph"
    return graph


def _allowed_role_label_swaps(symmetry_groups: Any) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    if not isinstance(symmetry_groups, list):
        return allowed
    for group in symmetry_groups:
        if not isinstance(group, dict) or group.get("mode") != "swap_allowed":
            continue
        for pair in group.get("nets", []) or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            labels = [normalize_role_label(value) for value in pair if normalize_role_label(value)]
            for label in labels:
                allowed.setdefault(label, set()).update(other for other in labels if other != label)
    return allowed


def _comp_node_id(component_id: str) -> str:
    return _ref_comp_node_id(component_id)


def _net_node_id(net_id: str) -> str:
    return _ref_net_node_id(net_id)


def _ref_comp_node_id(component_id: str) -> str:
    return f"ref_comp:{component_id}"


def _ref_net_node_id(net_id: str) -> str:
    return f"ref_net:{net_id}"


def _cur_comp_node_id(component_id: str) -> str:
    return f"cur_comp:{component_id}"


def _cur_net_node_id(net_id: str) -> str:
    return f"cur_net:{net_id}"
