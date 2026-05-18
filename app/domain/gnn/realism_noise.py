"""Sim → real distribution shift — netlist_v2 noise operators (plan §十 R6).

The P5 evaluator currently runs on netlist_v2 dicts **synthesised from
the perturbation pipeline's cur HCGs** — every field is pristine,
every component_id matches ref, every IC has its part_subtype set,
every net has a clean role_label.

Production netlist_v2 (S3 output) looks nothing like that:
- vision pipeline assigns ad-hoc component_ids (``"Resistor_001"``)
- ``part_subtype`` may be empty when the OCR pass missed it
- ``role_label`` / ``canonical_name`` may be ``"NET_037"`` (auto-generated)
- ``pins[*].confidence`` lives in [0.4, 0.95] not [1.0, 1.0]
- visible wires explode into separate ``Wire`` ComponentInstance entries
- ``electrical_net_id`` is sometimes None for a barely-detected pin

This module ships **deterministic, composable noise operators** so we
can take a synthetic netlist_v2 and **age** it into something
production-shaped, then re-run the evaluator to measure how much the
rule comparator + GNN advisor degrade on sim-but-noisier data. Once
real student exports arrive (Phase 3) the same operators can be
re-tuned against the observed distribution.

Three profiles ship out of the box:

| profile    | use                          | shape |
|------------|------------------------------|---|
| ``clean``  | sanity check                 | pristine (identity op only) |
| ``low``    | "well-labelled production"   | pin confidence drift only |
| ``high``   | "OCR had a bad day"          | the real risk surface |

Per plan §一 / RISK_REGISTER §1: this is an offline measurement tool.
None of these operators mutate live data.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Operator framework
# ---------------------------------------------------------------------------


NoiseOpFn = Callable[[dict[str, Any], random.Random], dict[str, Any]]


@dataclass(frozen=True)
class NoiseOperator:
    """A single named, deterministic netlist_v2 mutation.

    Operators take a netlist_v2 dict (shallow-copied by the caller),
    mutate it **in place**, and return it. They get a seeded
    ``random.Random`` so the same (netlist, seed) pair always produces
    the same noisy output — important for reproducible drift studies.
    """

    name: str
    op: NoiseOpFn
    description: str = ""

    def apply(
        self, netlist_v2: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        return self.op(netlist_v2, rng)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def _strip_role_labels(netlist_v2: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Drop ``role_label`` / ``manual_role`` / ``canonical_name`` so the
    cur graph falls back to role inference. Production S3 emits these
    only when vision is confident; OCR fail / missing label → empty."""

    for net in netlist_v2.get("nets", []) or []:
        if not isinstance(net, dict):
            continue
        net.pop("role_label", None)
        net.pop("manual_role", None)
        # canonical_name often degenerates to "NET_037" style autogenered tag
        cur_canonical = net.get("canonical_name")
        if cur_canonical and not str(cur_canonical).startswith("NET_"):
            net["canonical_name"] = f"NET_{rng.randint(0, 999):03d}"
        net.pop("power_role", None)
    return netlist_v2


def _drop_role_keep_canonical(
    netlist_v2: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Drop the explicit ``role`` field but keep ``canonical_name``.
    Models the production pattern where the topology stage left a name
    behind but the role classifier didn't fire."""

    for net in netlist_v2.get("nets", []) or []:
        if isinstance(net, dict):
            net.pop("role", None)
            net.pop("manual_role", None)
    return netlist_v2


def _lower_pin_confidence(
    netlist_v2: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Pin confidences drop from 1.0 into the [0.55, 0.95] band, with a
    small chance (5%) of < 0.5 (production "borderline detection")."""

    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        for pin in comp.get("pins", []) or []:
            if not isinstance(pin, dict):
                continue
            if rng.random() < 0.05:
                pin["confidence"] = round(rng.uniform(0.30, 0.50), 3)
            else:
                pin["confidence"] = round(rng.uniform(0.55, 0.95), 3)
    return netlist_v2


def _rename_components_to_production_pattern(
    netlist_v2: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Re-id components from ``R1`` → ``Resistor_001``. The new IDs
    don't align with ref payload IDs anymore, so the rule path has to
    fall back on type-based component matching."""

    counters: dict[str, int] = {}
    rename: dict[str, str] = {}
    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        ctype = str(comp.get("component_type") or "Unknown")
        counters[ctype] = counters.get(ctype, 0) + 1
        old_id = str(comp.get("component_id") or "")
        new_id = f"{ctype}_{counters[ctype]:03d}"
        rename[old_id] = new_id
        comp["component_id"] = new_id
    # No pin-id rewrite needed — netlist_v2 doesn't cross-reference
    # component_id from pins. The mapping is retained in metadata so
    # downstream tooling can do its own alignment if needed.
    netlist_v2.setdefault("metadata", {})["realism_renames"] = rename
    return netlist_v2


def _drop_ic_subtype(
    netlist_v2: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Clear ``part_subtype`` on IC components. Production: OCR didn't
    catch the chip's printed model number, so PortType normalisation
    falls back to generic pin labels. Closes the loop on the bug we
    found in §6 — that path is now exercised here on purpose."""

    for comp in netlist_v2.get("components", []) or []:
        if not isinstance(comp, dict):
            continue
        if str(comp.get("component_type") or "") == "IC":
            comp["part_subtype"] = ""
    return netlist_v2


def _add_wire_clutter(
    netlist_v2: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Inject 1-3 disconnected Wire components mirroring what vision
    does when it sees a stray wire crossing the board. Wires here are
    "no-op" (they connect to no net), so they should be ignored by the
    rule comparator — this is a regression guard."""

    n_clutter = rng.randint(1, 3)
    existing_ids = {
        c.get("component_id")
        for c in netlist_v2.get("components", []) or []
    }
    for _ in range(n_clutter):
        # generate a unique id
        for _ in range(10):
            tag = f"Wire_clutter_{rng.randint(0, 999):03d}"
            if tag not in existing_ids:
                break
        netlist_v2.setdefault("components", []).append({
            "component_id": tag,
            "component_type": "Wire",
            "package_type": "",
            "part_subtype": "",
            "polarity": "none",
            "pins": [],
            "confidence": round(rng.uniform(0.40, 0.70), 3),
            "metadata": {"realism": "wire_clutter"},
        })
        existing_ids.add(tag)
    return netlist_v2


def _identity(netlist_v2: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return netlist_v2


STRIP_ROLE_LABELS = NoiseOperator(
    "strip_role_labels", _strip_role_labels,
    "Drop role_label / manual_role / power_role; canonical_name → NET_037 style.",
)
DROP_ROLE_KEEP_CANONICAL = NoiseOperator(
    "drop_role_keep_canonical", _drop_role_keep_canonical,
    "Drop explicit role field but keep canonical_name (production fall-through path).",
)
LOWER_PIN_CONFIDENCE = NoiseOperator(
    "lower_pin_confidence", _lower_pin_confidence,
    "Pin confidences drop from 1.0 → [0.55, 0.95] (5% chance < 0.5).",
)
RENAME_COMPONENTS = NoiseOperator(
    "rename_components", _rename_components_to_production_pattern,
    "R1 → Resistor_001 style — breaks naive id-based matching.",
)
DROP_IC_SUBTYPE = NoiseOperator(
    "drop_ic_subtype", _drop_ic_subtype,
    "Clear IC part_subtype → generic pin role normalisation.",
)
ADD_WIRE_CLUTTER = NoiseOperator(
    "add_wire_clutter", _add_wire_clutter,
    "Inject 1-3 disconnected Wire components (vision stray-wire artefacts).",
)
IDENTITY = NoiseOperator(
    "identity", _identity,
    "No-op — useful as a baseline so the same export pipeline serves all profiles.",
)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealismProfile:
    """A named bundle of NoiseOperators applied in order.

    Profiles ship pre-baked but the dataclass is open: callers can
    compose new ones for ad-hoc experiments without touching this
    module. ``seed`` is folded into the per-sample RNG so the whole
    pipeline is reproducible.
    """

    name: str
    operators: tuple[NoiseOperator, ...] = ()
    description: str = ""

    def apply(
        self,
        netlist_v2: dict[str, Any],
        *,
        seed: int = 0,
    ) -> dict[str, Any]:
        out = copy.deepcopy(netlist_v2)
        rng = random.Random(seed)
        for op in self.operators:
            out = op.apply(out, rng)
        out.setdefault("metadata", {})["realism_profile"] = self.name
        return out


CLEAN_PROFILE = RealismProfile(
    name="clean",
    operators=(IDENTITY,),
    description="Pristine synthetic — no noise (control baseline).",
)
LOW_NOISE_PROFILE = RealismProfile(
    name="low",
    operators=(LOWER_PIN_CONFIDENCE,),
    description="Pin confidences degrade only — labels, subtypes, IDs intact.",
)
HIGH_NOISE_PROFILE = RealismProfile(
    name="high",
    operators=(
        LOWER_PIN_CONFIDENCE,
        STRIP_ROLE_LABELS,
        RENAME_COMPONENTS,
        DROP_IC_SUBTYPE,
        ADD_WIRE_CLUTTER,
    ),
    description=(
        "Worst-plausible production: low confidence + no labels + "
        "renamed comps + no subtypes + stray wires. This is the actual "
        "sim→real risk surface."
    ),
)


PROFILES: dict[str, RealismProfile] = {
    "clean": CLEAN_PROFILE,
    "low": LOW_NOISE_PROFILE,
    "high": HIGH_NOISE_PROFILE,
}


def get_profile(name: str) -> RealismProfile:
    """Look up a built-in profile by name. Raises KeyError on miss."""

    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(
            f"unknown realism profile {name!r}. Available: {available}"
        )
    return PROFILES[name]


__all__ = [
    "NoiseOperator",
    "RealismProfile",
    "STRIP_ROLE_LABELS",
    "DROP_ROLE_KEEP_CANONICAL",
    "LOWER_PIN_CONFIDENCE",
    "RENAME_COMPONENTS",
    "DROP_IC_SUBTYPE",
    "ADD_WIRE_CLUTTER",
    "IDENTITY",
    "CLEAN_PROFILE",
    "LOW_NOISE_PROFILE",
    "HIGH_NOISE_PROFILE",
    "PROFILES",
    "get_profile",
]
