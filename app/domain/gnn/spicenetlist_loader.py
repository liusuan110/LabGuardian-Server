"""GNN 模块 · SpiceNetlist loader（P2.5 预训练数据源）

把 GNN-ACLP (Dong et al. 2024) 公开发布的 SpiceNetlist 数据集
（``GNN_ACLP-main/SpiceNetlist/JSON/<id>.json``）转成
:class:`HeteroCircuitGraph`，供 :mod:`pretrain_dataset` 做 masked-edge
自监督预训练。

JSON 格式（每个文件一个电路）::

    [
      {
        "component_type": "NMOS",
        "port_connection": {"Drain": "1", "Gate": "2", "Source": "0"}
      },
      {
        "component_type": "Voltage",
        "port_connection": {"Pos": "2", "Neg": "0"}
      }
    ]

**Schema 兼容**（plan 协商决定）：
- MOSFET (NMOS/PMOS) → :class:`ComponentType.TRANSISTOR` —— Drain→COLLECTOR,
  Gate→BASE, Source→EMITTER（loose 但 graph topology 模型不依赖 pin 名）
- Inductor (Ind) → :class:`ComponentType.UNKNOWN` —— Pos→PIN1, Neg→PIN2
- 其余按字面映射

不引入 torch（loader 纯 Python）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.domain.gnn.graph_schema import (
    ComponentType,
    PortType,
)
from app.domain.gnn.hetero_circuit import (
    ComponentNode,
    HeteroCircuitGraph,
    NetNode,
    PortConnectsNetEdge,
    PortNode,
)

# ---------------------------------------------------------------------------
# Component / pin mapping tables
# ---------------------------------------------------------------------------


# SpiceNetlist component_type → our ComponentType
COMPONENT_TYPE_MAP: dict[str, ComponentType] = {
    "Res": ComponentType.RESISTOR,
    "Resistor": ComponentType.RESISTOR,
    "Cap": ComponentType.CAPACITOR_CERAMIC,
    "Capacitor": ComponentType.CAPACITOR_CERAMIC,
    "Ind": ComponentType.UNKNOWN,        # we have no INDUCTOR in our schema
    "Inductor": ComponentType.UNKNOWN,
    "NMOS": ComponentType.TRANSISTOR,
    "PMOS": ComponentType.TRANSISTOR,
    "MOSFET": ComponentType.TRANSISTOR,
    "NPN": ComponentType.TRANSISTOR,
    "PNP": ComponentType.TRANSISTOR,
    "Diode": ComponentType.DIODE,
    "diode": ComponentType.DIODE,
    "Voltage": ComponentType.VOLTAGE_SOURCE,
    "Current": ComponentType.CURRENT_SOURCE,
    "IC": ComponentType.IC,
    "Op_amp": ComponentType.OPAMP,
}

# Per-component pin name → PortType. Falls back to PIN_N_GENERIC for unknown
# pin names. The mapping is keyed by the SpiceNetlist component_type so we
# can disambiguate (e.g. "Pos" means POSITIVE on a Voltage source but ANODE
# on a Diode).
_PIN_MAPS: dict[str, dict[str, PortType]] = {
    "Res": {"Pos": PortType.PIN1, "Neg": PortType.PIN2},
    "Resistor": {"Pos": PortType.PIN1, "Neg": PortType.PIN2},
    "Cap": {"Pos": PortType.PIN1, "Neg": PortType.PIN2},
    "Capacitor": {"Pos": PortType.PIN1, "Neg": PortType.PIN2},
    "Ind": {"Pos": PortType.PIN1, "Neg": PortType.PIN2},
    "Inductor": {"Pos": PortType.PIN1, "Neg": PortType.PIN2},
    "NMOS": {
        "Drain": PortType.COLLECTOR,
        "Gate": PortType.BASE,
        "Source": PortType.EMITTER,
    },
    "PMOS": {
        "Drain": PortType.COLLECTOR,
        "Gate": PortType.BASE,
        "Source": PortType.EMITTER,
    },
    "MOSFET": {
        "Drain": PortType.COLLECTOR,
        "Gate": PortType.BASE,
        "Source": PortType.EMITTER,
    },
    "NPN": {
        "Collector": PortType.COLLECTOR,
        "Base": PortType.BASE,
        "Emitter": PortType.EMITTER,
    },
    "PNP": {
        "Collector": PortType.COLLECTOR,
        "Base": PortType.BASE,
        "Emitter": PortType.EMITTER,
    },
    "Diode": {"Pos": PortType.ANODE, "Neg": PortType.CATHODE},
    "diode": {"Pos": PortType.ANODE, "Neg": PortType.CATHODE},
    "Voltage": {"Pos": PortType.POSITIVE, "Neg": PortType.NEGATIVE},
    "Current": {"Pos": PortType.POSITIVE, "Neg": PortType.NEGATIVE},
}

# Net "0" is ground convention in SPICE. Map to NetRole.GND so the
# downstream encoder can lift the is_power_rail flag.
_GROUND_NET_NAMES: frozenset[str] = frozenset({"0"})


# ---------------------------------------------------------------------------
# Loader API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpiceNetlistCircuit:
    """One parsed circuit; carries the original id for traceability."""

    circuit_id: str
    hcg: HeteroCircuitGraph
    raw_components: list[dict]  # untouched JSON payload (for debugging)


def _map_component_type(raw: str) -> str:
    """Returns our ComponentType.value, falling back to UNKNOWN."""

    mapped = COMPONENT_TYPE_MAP.get(raw)
    if mapped is None:
        return ComponentType.UNKNOWN.value
    return mapped.value


def _map_port_type(component_type_raw: str, pin_name: str) -> str:
    table = _PIN_MAPS.get(component_type_raw, {})
    pt = table.get(pin_name)
    if pt is None:
        return PortType.PIN_N_GENERIC.value
    return pt.value


def load_circuit_json(payload: list[dict], *, circuit_id: str) -> SpiceNetlistCircuit:
    """Build a single :class:`SpiceNetlistCircuit` from one parsed JSON.

    side is ``"ref"`` since the SpiceNetlist circuits act as the
    ground-truth pretraining graphs (we'll mask edges to derive positives
    and negatives, but the circuit itself is canonical)."""

    hcg = HeteroCircuitGraph(side="ref")
    nets_seen: set[str] = set()

    for comp_idx, comp_raw in enumerate(payload):
        c_type_raw = str(comp_raw.get("component_type", "")).strip()
        port_conn = comp_raw.get("port_connection") or {}
        if not isinstance(port_conn, dict) or not c_type_raw:
            continue

        ctype_value = _map_component_type(c_type_raw)
        comp_source_id = f"C{comp_idx}_{c_type_raw}"
        comp_node_id = f"ref_comp:{comp_source_id}"
        # Track polarity_class via the lookup table (none if not present)
        from app.domain.gnn.graph_schema import POLARITY_CLASS_OF

        polarity_class = POLARITY_CLASS_OF.get(
            ComponentType(ctype_value),
            POLARITY_CLASS_OF[ComponentType.UNKNOWN],
        ).value
        comp_node = ComponentNode(
            node_id=comp_node_id,
            side="ref",
            source_id=comp_source_id,
            ctype=ctype_value,
            package=None,
            polarity_class=polarity_class,
            pin_count=len(port_conn),
            value=None,
            confidence=1.0,
        )
        hcg.components[comp_node_id] = comp_node
        hcg.port_of_component[comp_node_id] = []

        for pin_name, net_raw in port_conn.items():
            net_source_id = str(net_raw)
            net_node_id = f"ref_net:{net_source_id}"
            # Register the net if first sighting
            if net_source_id not in nets_seen:
                nets_seen.add(net_source_id)
                role = "gnd" if net_source_id in _GROUND_NET_NAMES else "signal"
                hcg.nets[net_node_id] = NetNode(
                    node_id=net_node_id,
                    side="ref",
                    source_id=net_source_id,
                    role=role,
                    role_label=None,
                    is_power_rail=(role == "gnd"),
                    voltage_hint=None,
                    aliases=(),
                )

            port_type = _map_port_type(c_type_raw, str(pin_name))
            port_key = str(pin_name).lower().replace(" ", "_")
            port_node_id = f"ref_port:{comp_source_id}.{port_key}"
            # If duplicate within same component, add suffix
            if port_node_id in hcg.ports:
                i = 2
                while f"{port_node_id}_{i}" in hcg.ports:
                    i += 1
                port_node_id = f"{port_node_id}_{i}"
                port_key = f"{port_key}_{i}"

            # Symmetry class for two-pin passives: PIN1+PIN2 are siblings.
            # We compute it later in a second pass when all ports of the
            # component are known.
            port_node = PortNode(
                node_id=port_node_id,
                side="ref",
                parent_component_id=comp_node_id,
                port_key=port_key,
                port_type=port_type,
                parent_ctype=ctype_value,
                polarity_sensitive=False,  # set below
                is_power_port=False,
                is_ground_port=(net_source_id in _GROUND_NET_NAMES),
                is_floating=False,
                pin_number=None,
                connection_policy="required",
                symmetry_class_id=0,  # set in second pass
            )
            hcg.ports[port_node_id] = port_node
            hcg.port_of_component[comp_node_id].append(port_node_id)

            edge = PortConnectsNetEdge(
                src_port_id=port_node_id,
                dst_net_id=net_node_id,
                connection_confidence=1.0,
                source_type="dsl",  # treat SpiceNetlist as canonical
                is_observed_in_cur=False,
            )
            hcg.edges.append(edge)

    _assign_symmetry_classes(hcg)
    hcg.metadata["source"] = "spicenetlist"
    hcg.metadata["circuit_id"] = circuit_id
    return SpiceNetlistCircuit(circuit_id=circuit_id, hcg=hcg, raw_components=list(payload))


def _assign_symmetry_classes(hcg: HeteroCircuitGraph) -> None:
    """Two-pin passives (Resistor / Capacitor / Inductor / Wire) get the
    same symmetry_class_id on both pins (since swap is electrically
    equivalent). Polar / multi-pin components get distinct classes per
    port (no swap allowed)."""

    # Use replace because PortNode is frozen
    from dataclasses import replace

    for comp_node_id, port_ids in hcg.port_of_component.items():
        comp = hcg.components[comp_node_id]
        if comp.ctype in {
            ComponentType.RESISTOR.value,
            ComponentType.CAPACITOR_CERAMIC.value,
            ComponentType.CAPACITOR.value,
            ComponentType.WIRE.value,
            ComponentType.UNKNOWN.value,  # Inductor falls here
        } and len(port_ids) == 2:
            # Both pins share symmetry_class_id=0
            for pid in port_ids:
                hcg.ports[pid] = replace(hcg.ports[pid], symmetry_class_id=0)
        else:
            # Distinct class per pin (polar / multi-pin components)
            for idx, pid in enumerate(port_ids):
                hcg.ports[pid] = replace(hcg.ports[pid], symmetry_class_id=idx)


def load_spicenetlist_dir(json_dir: Path) -> list[SpiceNetlistCircuit]:
    """Walk every ``<id>.json`` in ``json_dir`` and parse it. Returns
    circuits sorted by integer id (stable across runs)."""

    if not json_dir.is_dir():
        raise FileNotFoundError(f"json_dir not found: {json_dir}")
    # Sort by integer id first (digit stems), then string stems alphabetically
    digit_files = sorted(
        (p for p in json_dir.glob("*.json") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    string_files = sorted(
        p for p in json_dir.glob("*.json") if not p.stem.isdigit()
    )
    files = list(digit_files) + list(string_files)
    out: list[SpiceNetlistCircuit] = []
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            out.append(load_circuit_json(payload, circuit_id=fp.stem))
        except Exception as e:  # noqa: BLE001 — skip bad files but log
            import logging

            logging.getLogger(__name__).warning(
                "skipped malformed SpiceNetlist file %s: %r", fp, e
            )
    return out


__all__ = [
    "COMPONENT_TYPE_MAP",
    "SpiceNetlistCircuit",
    "load_circuit_json",
    "load_spicenetlist_dir",
]
