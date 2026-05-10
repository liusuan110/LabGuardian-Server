from __future__ import annotations

from typing import Any

import networkx as nx

from app.domain.circuit import norm_component_type


VALID_NET_ROLES = {"signal", "ground", "power", "input", "output"}


def normalize_component_type(value: Any) -> str:
    return norm_component_type(str(value or "UNKNOWN"))


def normalize_net_role(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"gnd", "ground", "vss", "0v", "earth"}:
        return "ground"
    if raw in {"vcc", "vdd", "power", "pwr", "vin_power", "supply", "+", "+5v", "+3v3"}:
        return "power"
    if raw in {"input", "in", "vin"}:
        return "input"
    if raw in {"output", "out", "vout"}:
        return "output"
    if raw in VALID_NET_ROLES:
        return raw
    return "signal"


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
            pin_net = str(pin.get("net") or "").strip()
            if not pin_net:
                raise ValueError(f"components[{idx}].pins[{pin_idx}] must contain net")
            if net_names and pin_net not in net_names:
                raise ValueError(f"component {ref_id} pin references unknown net '{pin_net}'")


def logical_reference_to_graph(payload: dict[str, Any]) -> nx.Graph:
    validate_logical_reference(payload)

    graph = nx.Graph()
    net_roles: dict[str, str] = {
        str(net["net"]): normalize_net_role(net.get("role"))
        for net in payload.get("nets", [])
    }

    referenced_nets = {
        str(pin["net"])
        for comp in payload.get("components", [])
        for pin in comp.get("pins", [])
    }
    for net_name in sorted(referenced_nets | set(net_roles)):
        graph.add_node(
            _ref_net_node_id(net_name),
            kind="net",
            role=net_roles.get(net_name, "signal"),
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
            net_name = str(pin["net"])
            graph.add_edge(
                comp_node,
                _ref_net_node_id(net_name),
                pin=str(pin.get("pin") or ""),
            )

    graph.graph["format"] = "logical_reference_v1"
    graph.graph["reference_id"] = payload.get("reference_id")
    graph.graph["name"] = payload.get("name")
    return graph


def current_netlist_v2_to_graph(netlist_v2: dict[str, Any]) -> nx.Graph:
    if not isinstance(netlist_v2, dict):
        raise ValueError("netlist_v2 must be a dict")

    graph = nx.Graph()
    net_roles: dict[str, str] = {}
    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net_id = str(net.get("electrical_net_id") or net.get("net_id") or "").strip()
        if not net_id:
            continue
        role = net.get("role") or net.get("manual_role")
        if not role:
            role_label = str(net.get("role_label") or "").strip()
            if role_label in {"VIN", "input", "in"}:
                role = "input"
            elif role_label in {"VOUT", "output", "out"}:
                role = "output"
            elif role_label in {"VCC", "power", "pwr"}:
                role = "power"
            elif role_label in {"GND", "ground", "gnd"}:
                role = "ground"
        if not role:
            power_role = str(net.get("power_role") or "").strip()
            if power_role == "GND":
                role = "ground"
            elif power_role == "VCC":
                role = "power"
        net_roles[net_id] = normalize_net_role(role)
        graph.add_node(
            _cur_net_node_id(net_id),
            kind="net",
            role=net_roles[net_id],
            source_id=net_id,
            role_label=net.get("role_label"),
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
                )
            graph.add_edge(
                comp_node,
                net_node,
                pin=str(pin.get("pin_name") or pin.get("pin") or ""),
            )

    graph.graph["format"] = "netlist_v2_logical_graph"
    return graph


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
