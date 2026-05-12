from __future__ import annotations

from typing import Any

from app.domain.circuit import norm_component_type
from app.domain.dsl.core import Circuit, Component, Net, Pin
from app.domain.logical_reference import normalize_net_role

DSL_SOURCE_TYPE = "dsl_python_v1"


def circuit_to_logical_reference(circuit: Circuit) -> dict[str, Any]:
    """Compile a DSL Circuit into the existing logical_reference_v1 payload."""
    payload: dict[str, Any] = {
        "format": "logical_reference_v1",
        "reference_id": circuit.reference_id,
    }
    if circuit.name:
        payload["name"] = circuit.name
    if circuit.description:
        payload["description"] = circuit.description
    if circuit.created_at:
        payload["created_at"] = circuit.created_at

    source = dict(circuit.source)
    original_type = source.get("type")
    source["type"] = DSL_SOURCE_TYPE
    if original_type and original_type != DSL_SOURCE_TYPE:
        source.setdefault("original_type", original_type)
    payload["source"] = source

    payload.update(circuit.metadata)
    payload["nets"] = [_compile_net(net) for net in circuit.nets]
    payload["components"] = [_compile_component(component) for component in circuit.components]
    if circuit.compare_options:
        payload["compare_options"] = dict(circuit.compare_options)
    if circuit.symmetry_groups:
        payload["symmetry_groups"] = circuit.symmetry_groups
    return payload


def _compile_net(net: Net) -> dict[str, Any]:
    item: dict[str, Any] = {"net": net.name}
    role = normalize_net_role(net.role)
    if role != "signal":
        item["role"] = role
    elif net.role:
        item["role"] = "signal"
    if net.label:
        item["label"] = net.label
    item.update(net.metadata)
    return item


def _compile_component(component: Component) -> dict[str, Any]:
    item: dict[str, Any] = {
        "ref_id": component.ref_id,
        "type": norm_component_type(component.component_type),
    }
    if component.value is not None:
        item["value"] = component.value
    if component.description:
        item["description"] = component.description
    if component.subtype:
        item["subtype"] = component.subtype
    item.update(component.metadata)
    item["pins"] = [_compile_pin(pin) for pin in component.pins if pin.net is not None or pin.nc]
    if not item["pins"]:
        raise ValueError(f"component {component.ref_id} has no connected or NC pins")
    return item


def _compile_pin(pin: Pin) -> dict[str, Any]:
    item: dict[str, Any] = {"pin": pin.name}
    if pin.nc:
        item["nc"] = True
        return item
    if pin.net is None:
        raise ValueError(f"{pin.component.ref_id}.{pin.name} is not connected")
    item["net"] = pin.net.name
    return item
