"""GNN 模块 · Perturbation Pipeline（P1 Phase A）

把一个 ref ``HeteroCircuitGraph`` 通过受控扰动变成 cur HCG +
``ComponentAlignment``，供 :func:`build_seal_samples_with_coverage_check`
打标签生成 P1 训练样本。

**P1 Phase A 范围**（4 个 perturbation operator，覆盖三类 LabelSource 路径）：

- :class:`IdentityPerturbation` —— 完美 cur copy；produce REF_PRESENT positives
- :class:`PinSwapSymmetricPerturbation` —— 同 symmetry_class pin swap（如
  R.pin1↔pin2）；produce REF_SYMMETRIC_SWAP（仍是正样本，验证 label_builder
  的 sibling 展开逻辑）
- :class:`WrongConnectionPerturbation` —— 把一条 edge 重新指到错误的 net；
  produce WRONG_OBSERVED 强负 + MISSING_EDGE wrong_redirect group
- :class:`PinReversedPerturbation` —— 极性器件（LED/Diode/Cap 电解）的
  anode↔cathode swap；produce WRONG_OBSERVED + MISSING_EDGE

**P1 Phase B 已落地**（8 个新 operator，registry 现 12 个）：
:class:`MissingComponentPerturbation` / :class:`ExtraComponentPerturbation` /
:class:`FloatingNetPerturbation` / :class:`ShortCircuitPerturbation` /
:class:`PowerSwappedPerturbation` / :class:`InputOutputSwappedPerturbation` /
:class:`ExtraWireBridgePerturbation` / :class:`ChainedPerturbation`。
Phase B 复用 raw_pin_edges + identity_alignment 流水线；新增的 add/drop
组件 / net 由 :func:`_rebuild_cur_from_raw` 的 ``extra_components`` /
``extra_nets`` / ``dropped_components`` / ``dropped_nets`` kwargs 支持。

**架构原则**：
- 每个 Perturbation 不直接改 HeteroCircuitGraph（dataclasses frozen）；
  它在 ``nx.Graph`` 中间表示上做修改，再走 ``build_hetero_circuit_graph``
  重建 cur HCG。这样所有 P0.6 materialize / connection_policy / symmetry
  自动重算，避免手动维护一致性。
- 每个 op 是确定性的：给同样的 ref + seed 必出同样的 cur。
- 每个 op 返回完整的 ``ComponentAlignment``（即便没有 rename，也填好）。

**禁止** import torch / torch_geometric。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple  # noqa: UP035 — runtime value on py<3.10

import networkx as nx  # type: ignore[import-untyped]

from app.domain.gnn.alignment import (
    ComponentAlignment,
    identity_alignment,
)
from app.domain.gnn.graph_schema import ConnectionPolicy
from app.domain.gnn.port_graph import build_hetero_circuit_graph

if TYPE_CHECKING:  # pragma: no cover
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph

# (comp_source_id, ctype, port_key, pin_role, net_source_id, confidence, source_type)
# Identical shape to ``port_graph.RawPinEdge``.
RawPinEdge = Tuple[str, str, str, str, str, Optional[float], Optional[str]]  # noqa: UP006, UP045 — runtime value on py<3.10


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbedCur:
    """单个扰动后的 cur，配套 alignment + 元数据。"""

    cur_hcg: HeteroCircuitGraph
    alignment: ComponentAlignment
    perturbation_chain: tuple[str, ...]  # e.g. ("pin_reversed:LED1",)
    expected_outcome: str  # "positive" | "wrong_observed" | "missing_required"
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HCG → nx 转换（让 perturbation 在 nx 层做修改，再 rebuild HCG）
# ---------------------------------------------------------------------------


def hcg_to_nx(
    ref_hcg: HeteroCircuitGraph,
    *,
    target_side: str = "cur",
) -> nx.Graph:
    """把 ref HCG 转换成 ``target_side`` (= "cur" 默认) 前缀的 nx.Graph，
    保留 component/net 节点属性与 (comp, net) 边的 pin / pin_role / comp_type
    属性。

    ⚠️ **平行 pin 折叠风险**：``nx.Graph`` 在同一对节点间最多保留一条边。
    当 component 的多个 pin 都连同一个 net（如 UA741 单位增益缓冲器中
    pin 2 与 pin 6 同接 VOUT），后写入的会覆盖前者，cur 会**丢一条边**。
    **Perturbation 实现不要走这条路径**，改用
    :func:`hcg_to_raw_pin_edges` + :func:`_rebuild_cur_from_raw` 的 raw-edge
    pipeline。本函数保留只是为了向后兼容旧测试。
    """

    if target_side not in ("ref", "cur"):
        raise ValueError(f"target_side must be 'ref' or 'cur', got {target_side!r}")
    g = nx.Graph()
    for _, comp in ref_hcg.components.items():
        g.add_node(
            f"{target_side}_comp:{comp.source_id}",
            kind="comp",
            ctype=comp.ctype,
            source_id=comp.source_id,
        )
    for _, net in ref_hcg.nets.items():
        # R6 (RULE_SEMANTICS §6.4) — propagate role_label so the rule
        # comparator's ``_node_match`` + ``_node_match_for_role_inference``
        # can distinguish e.g. ref:NET_B(signal/INTERNAL) from
        # cur:NET_A(input/UI1). Without this, role-inference happily
        # remaps cur's labelled signal nets onto wrong ref nets and
        # ``wrong_connection`` perturbations slip through as
        # ``full_isomorphism_with_inferred_roles``.
        node_attrs: dict[str, Any] = {
            "kind": "net",
            "role": net.role,
            "source_id": net.source_id,
        }
        if net.role_label:
            node_attrs["role_label"] = net.role_label
            # When the cur side originated from a netlist_v2 with an
            # explicit role_label, the production graph builder tags
            # role_source="role_label"; mirror that here so
            # role-inference treats the label as authoritative.
            node_attrs["role_source"] = "role_label"
        if net.aliases:
            node_attrs["aliases"] = list(net.aliases)
        g.add_node(
            f"{target_side}_net:{net.source_id}",
            **node_attrs,
        )
    # Late import to avoid pulling logical_reference into the
    # perturbation module's import-time graph (it pulls big deps).
    from app.domain.logical_reference import normalize_pin_role
    for edge in ref_hcg.edges:
        rp = ref_hcg.ports[edge.src_port_id]
        comp = ref_hcg.components[rp.parent_component_id]
        net = ref_hcg.nets[edge.dst_net_id]
        g.add_edge(
            f"{target_side}_comp:{comp.source_id}",
            f"{target_side}_net:{net.source_id}",
            pin=rp.port_key,
            # R6 fix: normalize pin_role so it agrees with the ref-side
            # produced by ``logical_reference_to_graph``. Without this
            # IC pin "2" / "1" stays as "2"/"1" but ref-side has
            # "pin2"/"pin1" → ``_edge_match`` fails → identity on
            # LM358 / UA741 was returning logic_correct=False.
            pin_role=normalize_pin_role(comp.ctype, rp.port_key),
            comp_type=comp.ctype,
        )
    return g


# ---------------------------------------------------------------------------
# Raw-edge pipeline (P1 Phase A audit fix — preserves parallel pins)
# ---------------------------------------------------------------------------


def hcg_to_raw_pin_edges(ref_hcg: HeteroCircuitGraph) -> list[RawPinEdge]:
    """Lossless extraction of per-pin edges from a ref HCG.

    Returns one tuple per ``PortConnectsNetEdge`` —— **preserves parallel
    pins on the same net** (which ``hcg_to_nx`` cannot). Output shape
    matches ``port_graph.RawPinEdge`` so it can be fed directly to
    ``build_hetero_circuit_graph(..., raw_pin_edges=...)``.
    """

    out: list[RawPinEdge] = []
    for edge in ref_hcg.edges:
        rp = ref_hcg.ports[edge.src_port_id]
        comp = ref_hcg.components[rp.parent_component_id]
        net = ref_hcg.nets[edge.dst_net_id]
        out.append(
            (
                comp.source_id,
                comp.ctype,
                rp.port_key,
                rp.port_key,  # pin_role — using port_key as approximation
                net.source_id,
                None,
                None,
            )
        )
    return out


def _hcg_to_nodes_only_nx(
    ref_hcg: HeteroCircuitGraph,
    target_side: str = "cur",
    *,
    dropped_components: set[str] | None = None,
    dropped_nets: set[str] | None = None,
) -> nx.Graph:
    """Comp + net nodes only; no edges. Edges flow via ``raw_pin_edges``.

    Optional ``dropped_components`` / ``dropped_nets`` skip the named
    source_ids during node seeding —— used by Phase B operators that
    delete a component or net from the cur side.
    """

    drop_c = dropped_components or set()
    drop_n = dropped_nets or set()
    g = nx.Graph()
    for comp in ref_hcg.components.values():
        if comp.source_id in drop_c:
            continue
        g.add_node(
            f"{target_side}_comp:{comp.source_id}",
            kind="comp",
            ctype=comp.ctype,
            source_id=comp.source_id,
        )
    for net in ref_hcg.nets.values():
        if net.source_id in drop_n:
            continue
        # R6 (RULE_SEMANTICS §6.4): propagate role_label / aliases so
        # the rebuilt cur HCG retains the disambiguating metadata
        # that lets _node_match + role-inference treat e.g. UI1 vs
        # internal as different nets. Without this, wrong_connection
        # perturbations on signal-labelled refs slip through
        # `full_isomorphism_with_inferred_roles`.
        node_attrs: dict[str, Any] = {
            "kind": "net",
            "role": net.role,
            "source_id": net.source_id,
        }
        if net.role_label:
            node_attrs["role_label"] = net.role_label
        if net.aliases:
            node_attrs["aliases"] = list(net.aliases)
        g.add_node(
            f"{target_side}_net:{net.source_id}",
            **node_attrs,
        )
    return g


def _rebuild_cur_from_raw(
    ref_hcg: HeteroCircuitGraph,
    raw_edges: list[RawPinEdge],
    subtype_by_source_id: dict[str, str] | None = None,
    *,
    extra_components: list[tuple[str, str]] | None = None,
    extra_nets: list[tuple[str, str]] | None = None,
    dropped_components: set[str] | None = None,
    dropped_nets: set[str] | None = None,
) -> HeteroCircuitGraph:
    """Rebuild cur HCG from a raw_pin_edges list, using ref for node metadata.

    All P0.6 materialize / connection_policy / symmetry recompute happens
    automatically inside ``build_hetero_circuit_graph``.

    Phase B kwargs:
        extra_components: list of ``(source_id, ctype)`` to inject into cur.
            Used by ExtraComponent / ExtraWireBridge / Chained composers.
        extra_nets: list of ``(source_id, role)`` to inject. Roles ∈
            {input, output, vcc, gnd, signal, unknown}.
        dropped_components: source_ids of components to delete from cur.
            Their edges in ``raw_edges`` are also filtered out for safety.
        dropped_nets: source_ids of nets to delete from cur. Their incoming
            edges in ``raw_edges`` are filtered.
    """

    drop_c = dropped_components or set()
    drop_n = dropped_nets or set()
    cur_g = _hcg_to_nodes_only_nx(
        ref_hcg,
        target_side="cur",
        dropped_components=drop_c,
        dropped_nets=drop_n,
    )
    # Inject extras
    for sid, ctype in extra_components or []:
        node_id = f"cur_comp:{sid}"
        if node_id in cur_g.nodes:
            continue
        cur_g.add_node(node_id, kind="comp", ctype=ctype, source_id=sid)
    for sid, role in extra_nets or []:
        node_id = f"cur_net:{sid}"
        if node_id in cur_g.nodes:
            continue
        cur_g.add_node(node_id, kind="net", role=role, source_id=sid)
    # Filter raw_edges against drops (safety against operators that forgot)
    safe_edges = [
        e for e in raw_edges
        if e[0] not in drop_c and e[4] not in drop_n
    ]
    return build_hetero_circuit_graph(
        cur_g,
        side="cur",
        subtype_by_source_id=subtype_by_source_id,
        raw_pin_edges=safe_edges,
    )


# ---------------------------------------------------------------------------
# Raw-edge mutation primitives
# ---------------------------------------------------------------------------


def _remove_pin_edges(
    raw_edges: list[RawPinEdge],
    *,
    comp_source_id: str,
    port_key: str | None = None,
    net_source_id: str | None = None,
) -> list[RawPinEdge]:
    """In-place remove all edges matching the given (comp, port?, net?)
    filter. Returns the removed entries (preserves order)."""

    removed: list[RawPinEdge] = []
    kept: list[RawPinEdge] = []
    for e in raw_edges:
        comp_match = e[0] == comp_source_id
        port_match = port_key is None or e[2] == port_key
        net_match = net_source_id is None or e[4] == net_source_id
        if comp_match and port_match and net_match:
            removed.append(e)
        else:
            kept.append(e)
    raw_edges[:] = kept
    return removed


def _add_pin_edge(
    raw_edges: list[RawPinEdge],
    *,
    comp_source_id: str,
    ctype: str,
    port_key: str,
    pin_role: str,
    net_source_id: str,
) -> None:
    raw_edges.append(
        (comp_source_id, ctype, port_key, pin_role, net_source_id, None, None)
    )


def _find_edges_for_component(
    raw_edges: list[RawPinEdge], comp_source_id: str
) -> list[RawPinEdge]:
    return [e for e in raw_edges if e[0] == comp_source_id]


def _collect_subtypes(ref_hcg: HeteroCircuitGraph) -> dict[str, str]:
    """Best-effort: extract IC subtypes from ref metadata.

    P0.5 stored subtype as build-time kwarg, not on ComponentNode itself.
    For P1, the dataset_builder is expected to pass subtype dict alongside
    the ref payload. Here we provide a fallback that returns empty dict,
    which means IC pins won't get fine-grained PortType on cur —— a known
    limitation. P1 dataset_builder should always pass explicit subtypes.
    """

    # Ref hcg metadata may carry it; P1 dataset_builder is responsible.
    value = ref_hcg.metadata.get("subtype_by_source_id", {})
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


# ---------------------------------------------------------------------------
# Base Perturbation protocol
# ---------------------------------------------------------------------------


class Perturbation:
    """Base class for all perturbation operators.

    Subclasses implement :meth:`apply`. Each operator is **stateless**;
    randomness comes from the caller-provided ``rng``."""

    name: str = ""
    expected_outcome: str = "positive"

    def apply(
        self,
        ref_hcg: HeteroCircuitGraph,
        rng: random.Random,
        *,
        subtype_by_source_id: dict[str, str] | None = None,
    ) -> PerturbedCur:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class IdentityPerturbation(Perturbation):
    """完美 cur copy。Produce only REF_PRESENT positives (+ symmetric
    sibling expansion for components with sym_class)."""

    name = "identity"
    expected_outcome = "positive"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        raw = hcg_to_raw_pin_edges(ref_hcg)  # parallel-pin safe
        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(self.name,),
            expected_outcome=self.expected_outcome,
            notes={"perturbation": self.name},
        )


class PinSwapSymmetricPerturbation(Perturbation):
    """选一个 component 上的某个 symmetry_class（size ≥ 2），把它内部两个
    port 的 net 互换。最常见场景：Resistor.pin1 ↔ pin2。

    本扰动**保持电气等价**：cur 应该被 label_builder 通过 sibling 展开认
    定为正确连接 (REF_SYMMETRIC_SWAP)。所以 ``expected_outcome="positive"``。
    """

    name = "pin_swap_symmetric"
    expected_outcome = "positive"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        # 找出所有有 sym_class size ≥ 2 的 component
        candidates: list[tuple[str, tuple[str, ...]]] = []
        for comp in ref_hcg.components.values():
            for group in comp.pin_symmetry_groups:
                if len(group) >= 2:
                    candidates.append((comp.source_id, group))
        if not candidates:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        comp_source_id, group = rng.choice(candidates)
        pin_a, pin_b = rng.sample(list(group), 2)

        raw = hcg_to_raw_pin_edges(ref_hcg)
        # raw-edge level lookup for the two pins on this component
        edges_a = [
            e for e in raw if e[0] == comp_source_id and e[2] == pin_a
        ]
        edges_b = [
            e for e in raw if e[0] == comp_source_id and e[2] == pin_b
        ]
        if not edges_a or not edges_b:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        # If a pin has multiple edges (parallel to multiple nets), pick first;
        # the swap semantics are well-defined for the typical 1-edge case.
        net_a = edges_a[0][4]
        net_b = edges_b[0][4]
        if net_a == net_b:
            # Swap is a no-op (both pins already on same net); fallback
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        ctype = edges_a[0][1]

        # Remove first edge of each pin, then re-add swapped
        _remove_pin_edges(
            raw, comp_source_id=comp_source_id, port_key=pin_a,
            net_source_id=net_a,
        )
        _remove_pin_edges(
            raw, comp_source_id=comp_source_id, port_key=pin_b,
            net_source_id=net_b,
        )
        _add_pin_edge(
            raw, comp_source_id=comp_source_id, ctype=ctype,
            port_key=pin_a, pin_role=pin_a, net_source_id=net_b,
        )
        _add_pin_edge(
            raw, comp_source_id=comp_source_id, ctype=ctype,
            port_key=pin_b, pin_role=pin_b, net_source_id=net_a,
        )

        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{comp_source_id}:{pin_a}↔{pin_b}",),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "component": comp_source_id,
                "swapped_pins": [pin_a, pin_b],
                "swapped_nets": [net_a, net_b],
            },
        )


class WrongConnectionPerturbation(Perturbation):
    """随机选一条 REQUIRED edge，把它的 net 端点重新指到另一个**不同的**
    net（不是当前 net 也不是该 port 的其它正确目标）。

    Produce: WRONG_OBSERVED (strong negative) + MISSING_EDGE wrong_redirect
    group (suggested_target = original correct net).

    **平行 pin 安全**：使用 raw_pin_edges，即便 wrong_net 已被同 component 的
    其它 pin 占用，新加的 (port, wrong_net) 边仍作为**独立**记录存在；不会
    被 nx.Graph 折叠掉任何 pin。
    """

    name = "wrong_connection"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        # 选一条 ref edge，其 port 是 REQUIRED policy
        eligible_edges: list[tuple[str, str, str, str]] = []
        # (comp_source_id, pin_key, net_source_id, ctype)
        for edge in ref_hcg.edges:
            rp = ref_hcg.ports[edge.src_port_id]
            comp = ref_hcg.components[rp.parent_component_id]
            net = ref_hcg.nets[edge.dst_net_id]
            if rp.connection_policy == ConnectionPolicy.REQUIRED.value:
                eligible_edges.append(
                    (comp.source_id, rp.port_key, net.source_id, comp.ctype)
                )
        if not eligible_edges:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        comp_source_id, pin_key, orig_net_sid, ctype = rng.choice(eligible_edges)

        # 选一个不同的 net 作为错误目标
        all_net_sids = [n.source_id for n in ref_hcg.nets.values()]
        wrong_candidates = [n for n in all_net_sids if n != orig_net_sid]
        if not wrong_candidates:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        wrong_net_sid = rng.choice(wrong_candidates)

        raw = hcg_to_raw_pin_edges(ref_hcg)
        # Remove ONLY the specific (comp, pin_key, orig_net) edge.
        _remove_pin_edges(
            raw, comp_source_id=comp_source_id, port_key=pin_key,
            net_source_id=orig_net_sid,
        )
        _add_pin_edge(
            raw, comp_source_id=comp_source_id, ctype=ctype,
            port_key=pin_key, pin_role=pin_key, net_source_id=wrong_net_sid,
        )

        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(
                f"{self.name}:{comp_source_id}.{pin_key}:"
                f"{orig_net_sid}→{wrong_net_sid}",
            ),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "component": comp_source_id,
                "pin": pin_key,
                "original_net": orig_net_sid,
                "wrong_net": wrong_net_sid,
            },
        )


# 极性器件 ctype 集合（LED / Diode / 电解 Cap），用于 PinReversed
_POLARIZED_CTYPES = frozenset({"LED", "Diode", "CapacitorElectrolytic"})

# 极性 pin pair（互换即极性反转）
_POLARITY_PAIRS: dict[str, tuple[str, str]] = {
    "LED": ("anode", "cathode"),
    "Diode": ("anode", "cathode"),
    "CapacitorElectrolytic": ("positive", "negative"),
}


class PinReversedPerturbation(Perturbation):
    """极性器件 (LED / Diode / 电解 Cap) anode↔cathode 互换。这是 wrong
    connection 的一个特例，但语义独立（"接反了"vs"接到别的 net 了"）。

    Produce: WRONG_OBSERVED for 两条边（两个 pin 都接错了 net）+ MISSING_EDGE
    wrong_redirect group × 2。
    """

    name = "pin_reversed"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        candidates: list[tuple[str, str, tuple[str, str], str, str]] = []
        # (comp_source_id, ctype, (pin_a, pin_b), net_at_a, net_at_b)
        for comp in ref_hcg.components.values():
            if comp.ctype not in _POLARIZED_CTYPES:
                continue
            pair = _POLARITY_PAIRS.get(comp.ctype)
            if pair is None:
                continue
            pin_a, pin_b = pair
            # find the two edges
            net_a, net_b = None, None
            for edge in ref_hcg.edges:
                rp = ref_hcg.ports[edge.src_port_id]
                if rp.parent_component_id != comp.node_id:
                    continue
                net_src = ref_hcg.nets[edge.dst_net_id].source_id
                if rp.port_key == pin_a:
                    net_a = net_src
                elif rp.port_key == pin_b:
                    net_b = net_src
            if net_a is None or net_b is None or net_a == net_b:
                continue
            candidates.append(
                (comp.source_id, comp.ctype, pair, net_a, net_b)
            )
        if not candidates:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        comp_source_id, ctype, (pin_a, pin_b), net_a, net_b = rng.choice(
            candidates
        )

        raw = hcg_to_raw_pin_edges(ref_hcg)
        # Remove ONLY the two specific (comp, pin_a, net_a) and (comp, pin_b, net_b) edges.
        _remove_pin_edges(
            raw, comp_source_id=comp_source_id, port_key=pin_a, net_source_id=net_a,
        )
        _remove_pin_edges(
            raw, comp_source_id=comp_source_id, port_key=pin_b, net_source_id=net_b,
        )
        # Reversed: pin_a now on net_b, pin_b now on net_a
        _add_pin_edge(
            raw, comp_source_id=comp_source_id, ctype=ctype,
            port_key=pin_a, pin_role=pin_a, net_source_id=net_b,
        )
        _add_pin_edge(
            raw, comp_source_id=comp_source_id, ctype=ctype,
            port_key=pin_b, pin_role=pin_b, net_source_id=net_a,
        )

        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(
                f"{self.name}:{comp_source_id}:{pin_a}↔{pin_b}",
            ),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "component": comp_source_id,
                "swapped_pins": [pin_a, pin_b],
                "swapped_nets": [net_a, net_b],
            },
        )


# ---------------------------------------------------------------------------
# Phase B operators (8 new — broaden negative coverage to match error_type head)
# ---------------------------------------------------------------------------
#
# Design notes:
# - All ops build on the raw_pin_edges + _rebuild_cur_from_raw pipeline so
#   parallel pins / IC spec / symmetry / connection_policy stay correct.
# - alignment is ``identity_alignment`` for every op (no component rename);
#   missing/extra components naturally surface in the unmatched_* notes,
#   and label_builder handles n_skipped_missing_component bookkeeping.
# - Each op falls back to IdentityPerturbation when it can't find a valid
#   anchor (e.g. PowerSwapped on a circuit with no VCC net), so callers
#   never hit "no-op crash" mid-pipeline.


def _pick_non_critical_component(
    ref_hcg: HeteroCircuitGraph, rng: random.Random
) -> str | None:
    """Pick a component to drop —— prefer leaf-like (low fanout) ones, never
    leave the circuit empty."""

    comps = list(ref_hcg.components.values())
    if len(comps) < 2:
        return None
    # Score each comp by total edges; pick uniformly from the low-fanout half
    fanout = {
        c.source_id: sum(
            1 for e in ref_hcg.edges
            if ref_hcg.ports[e.src_port_id].parent_component_id == c.node_id
        )
        for c in comps
    }
    sorted_sids = sorted(fanout, key=lambda s: fanout[s])
    # Keep the lower half (or at least 1)
    candidates = sorted_sids[: max(1, len(sorted_sids) // 2)]
    return rng.choice(candidates)


class MissingComponentPerturbation(Perturbation):
    """Drop one component from cur. Produces ``missing_required``: ref edges
    pointing at the dropped component get ``n_skipped_missing_component`` +1
    in label_stats, the SEAL head won't get direct supervision for those
    pins, but the graph-level error_type head learns the pattern."""

    name = "missing_component"
    expected_outcome = "missing_required"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        target = _pick_non_critical_component(ref_hcg, rng)
        if target is None:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        raw = hcg_to_raw_pin_edges(ref_hcg)
        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
            dropped_components={target},
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{target}",),
            expected_outcome=self.expected_outcome,
            notes={"perturbation": self.name, "dropped_component": target},
        )


# Synthetic extra-component templates: (ctype, n_pins, pin_keys).
_EXTRA_COMPONENT_TEMPLATES: list[tuple[str, list[str]]] = [
    ("Resistor", ["pin1", "pin2"]),
    ("CapacitorCeramic", ["pin1", "pin2"]),
    ("Wire", ["pin1", "pin2"]),
]


class ExtraComponentPerturbation(Perturbation):
    """Inject a parasitic extra component into cur. Both pins land on
    randomly chosen existing nets → label_builder Step 2.5 emits
    WRONG_OBSERVED for each because they aren't in the ref-sym-aware
    correct set."""

    name = "extra_component"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        all_nets = [n.source_id for n in ref_hcg.nets.values()]
        if len(all_nets) < 2:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        ctype, pin_keys = rng.choice(_EXTRA_COMPONENT_TEMPLATES)
        # Pick a fresh source_id that doesn't clash with ref components
        existing = {c.source_id for c in ref_hcg.components.values()}
        idx = 0
        while f"X_{ctype}_{idx}" in existing:
            idx += 1
        new_sid = f"X_{ctype}_{idx}"
        nets_used = rng.sample(all_nets, k=min(len(pin_keys), len(all_nets)))
        # If fewer nets than pins, recycle (extra pin parallels first net)
        while len(nets_used) < len(pin_keys):
            nets_used.append(nets_used[0])

        raw = hcg_to_raw_pin_edges(ref_hcg)
        for pin_key, net_sid in zip(pin_keys, nets_used):
            _add_pin_edge(
                raw, comp_source_id=new_sid, ctype=ctype,
                port_key=pin_key, pin_role=pin_key, net_source_id=net_sid,
            )

        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
            extra_components=[(new_sid, ctype)],
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(
                f"{self.name}:{new_sid}@{','.join(nets_used)}",
            ),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "extra_component": new_sid,
                "extra_ctype": ctype,
                "attached_nets": list(nets_used),
            },
        )


class FloatingNetPerturbation(Perturbation):
    """Remove all-but-one edges into a chosen net N (degree ≥ 2 in ref). N
    becomes effectively floating with a single dangling pin. Ref edges
    that pointed at N become REF_ABSENT_REQUIRED + wrong_redirect groups
    on the affected ports."""

    name = "floating_net"
    expected_outcome = "missing_required"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        net_degree: dict[str, int] = {}
        for e in ref_hcg.edges:
            sid = ref_hcg.nets[e.dst_net_id].source_id
            net_degree[sid] = net_degree.get(sid, 0) + 1
        candidates = [sid for sid, d in net_degree.items() if d >= 2]
        if not candidates:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        target_net = rng.choice(candidates)
        raw = hcg_to_raw_pin_edges(ref_hcg)
        target_edges = [e for e in raw if e[4] == target_net]
        keep = rng.choice(target_edges)
        # Remove all edges to target_net, keep `keep`
        raw[:] = [e for e in raw if e[4] != target_net]
        raw.append(keep)
        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{target_net}",),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "floated_net": target_net,
                "kept_edge_pin": (keep[0], keep[2]),
                "removed_edge_count": len(target_edges) - 1,
            },
        )


class ShortCircuitPerturbation(Perturbation):
    """Re-route every pin originally on net N2 onto net N1 (N2 ≠ N1). N2
    stays in cur as an isolated node so alignment maps cleanly. Each
    moved pin yields WRONG_OBSERVED + a wrong_redirect group pointing
    back at N2."""

    name = "short_circuit"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        net_sids = [n.source_id for n in ref_hcg.nets.values()]
        if len(net_sids) < 2:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        # Pick two distinct nets where N2 has at least one ref edge
        per_net = {sid: 0 for sid in net_sids}
        for e in ref_hcg.edges:
            per_net[ref_hcg.nets[e.dst_net_id].source_id] += 1
        eligible_n2 = [sid for sid, d in per_net.items() if d >= 1]
        if not eligible_n2:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        n2 = rng.choice(eligible_n2)
        n1_candidates = [sid for sid in net_sids if sid != n2]
        n1 = rng.choice(n1_candidates)
        raw = hcg_to_raw_pin_edges(ref_hcg)
        new_raw: list[RawPinEdge] = []
        for e in raw:
            if e[4] == n2:
                new_raw.append((e[0], e[1], e[2], e[3], n1, e[5], e[6]))
            else:
                new_raw.append(e)
        cur = _rebuild_cur_from_raw(
            ref_hcg, new_raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{n2}→{n1}",),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "from_net": n2,
                "into_net": n1,
            },
        )


def _swap_nets_in_raw(
    raw: list[RawPinEdge], net_a: str, net_b: str
) -> list[RawPinEdge]:
    out: list[RawPinEdge] = []
    for e in raw:
        if e[4] == net_a:
            out.append((e[0], e[1], e[2], e[3], net_b, e[5], e[6]))
        elif e[4] == net_b:
            out.append((e[0], e[1], e[2], e[3], net_a, e[5], e[6]))
        else:
            out.append(e)
    return out


class PowerSwappedPerturbation(Perturbation):
    """Swap every edge between the VCC net (role=power) and the GND net
    (role=ground). Educationally critical: in real hardware this is a
    safety-relevant mistake. Each rerouted pin yields WRONG_OBSERVED."""

    name = "power_swapped"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        vcc = [n.source_id for n in ref_hcg.nets.values() if n.role == "power"]
        gnd = [n.source_id for n in ref_hcg.nets.values() if n.role == "ground"]
        if not vcc or not gnd:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        vcc_sid = rng.choice(vcc)
        gnd_sid = rng.choice(gnd)
        raw = _swap_nets_in_raw(hcg_to_raw_pin_edges(ref_hcg), vcc_sid, gnd_sid)
        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{vcc_sid}↔{gnd_sid}",),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "vcc_net": vcc_sid,
                "gnd_net": gnd_sid,
            },
        )


class InputOutputSwappedPerturbation(Perturbation):
    """Swap every edge between an input-role net and an output-role net.
    Used to teach the model that signal direction matters."""

    name = "input_output_swapped"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        inputs = [n.source_id for n in ref_hcg.nets.values() if n.role == "input"]
        outputs = [n.source_id for n in ref_hcg.nets.values() if n.role == "output"]
        if not inputs or not outputs:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        in_sid = rng.choice(inputs)
        out_sid = rng.choice(outputs)
        raw = _swap_nets_in_raw(hcg_to_raw_pin_edges(ref_hcg), in_sid, out_sid)
        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{in_sid}↔{out_sid}",),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "input_net": in_sid,
                "output_net": out_sid,
            },
        )


class ExtraWireBridgePerturbation(Perturbation):
    """Inject a 2-pin Wire connecting two nets that weren't directly bridged
    in ref. The wire's pins both yield WRONG_OBSERVED. Stronger than
    ShortCircuit because it preserves *all* original ref edges — just
    adds an unwanted parasitic path."""

    name = "extra_wire_bridge"
    expected_outcome = "wrong_observed"

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        net_sids = [n.source_id for n in ref_hcg.nets.values()]
        if len(net_sids) < 2:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        # Choose two distinct nets that don't already share a component
        shared_pairs: set[frozenset] = set()
        for comp in ref_hcg.components.values():
            comp_nets = {
                ref_hcg.nets[e.dst_net_id].source_id
                for e in ref_hcg.edges
                if ref_hcg.ports[e.src_port_id].parent_component_id == comp.node_id
            }
            for a in comp_nets:
                for b in comp_nets:
                    if a != b:
                        shared_pairs.add(frozenset({a, b}))
        all_pairs = [
            (a, b)
            for i, a in enumerate(net_sids)
            for b in net_sids[i + 1 :]
            if frozenset({a, b}) not in shared_pairs
        ]
        if not all_pairs:
            return IdentityPerturbation().apply(
                ref_hcg, rng, subtype_by_source_id=subtype_by_source_id
            )
        net_a, net_b = rng.choice(all_pairs)
        existing = {c.source_id for c in ref_hcg.components.values()}
        idx = 0
        while f"X_Wire_{idx}" in existing:
            idx += 1
        wire_sid = f"X_Wire_{idx}"
        raw = hcg_to_raw_pin_edges(ref_hcg)
        _add_pin_edge(raw, comp_source_id=wire_sid, ctype="Wire",
                      port_key="pin1", pin_role="pin1", net_source_id=net_a)
        _add_pin_edge(raw, comp_source_id=wire_sid, ctype="Wire",
                      port_key="pin2", pin_role="pin2", net_source_id=net_b)
        cur = _rebuild_cur_from_raw(
            ref_hcg, raw,
            subtype_by_source_id or _collect_subtypes(ref_hcg),
            extra_components=[(wire_sid, "Wire")],
        )
        alignment = identity_alignment(ref_hcg, cur)
        return PerturbedCur(
            cur_hcg=cur,
            alignment=alignment,
            perturbation_chain=(f"{self.name}:{wire_sid}:{net_a}↔{net_b}",),
            expected_outcome=self.expected_outcome,
            notes={
                "perturbation": self.name,
                "wire_id": wire_sid,
                "bridged_nets": [net_a, net_b],
            },
        )


# Hard-sample composer ----------------------------------------------------

# Default chain pool: a mix of negatives that compose well (no double-drop).
_DEFAULT_CHAIN_POOL: tuple[str, ...] = (
    "wrong_connection",
    "pin_reversed",
    "floating_net",
    "extra_component",
    "power_swapped",
)


class ChainedPerturbation(Perturbation):
    """Apply 2–3 perturbations sequentially. Each link receives the prior
    link's cur HCG as its new "ref" — so misalignment-on-misalignment is
    valid. The final expected_outcome is the **most severe** seen along
    the chain (priority: wrong_observed > missing_required > positive)."""

    name = "chained"
    expected_outcome = "wrong_observed"  # default; overwritten per-sample

    def __init__(
        self,
        chain_pool: tuple[str, ...] = _DEFAULT_CHAIN_POOL,
        min_links: int = 2,
        max_links: int = 3,
    ):
        self.chain_pool = chain_pool
        self.min_links = min_links
        self.max_links = max_links

    def apply(self, ref_hcg, rng, *, subtype_by_source_id=None):
        n_links = rng.randint(self.min_links, self.max_links)
        chosen = rng.sample(self.chain_pool, k=min(n_links, len(self.chain_pool)))
        current_hcg = ref_hcg
        chain_entries: list[str] = []
        notes_chain: list[dict] = []
        most_severe = "positive"
        severity_rank = {"positive": 0, "missing_required": 1, "wrong_observed": 2}
        for link_name in chosen:
            op = PERTURBATION_REGISTRY[link_name]
            p = op.apply(
                current_hcg, rng,
                subtype_by_source_id=subtype_by_source_id,
            )
            chain_entries.extend(p.perturbation_chain)
            notes_chain.append(p.notes)
            if severity_rank[p.expected_outcome] > severity_rank[most_severe]:
                most_severe = p.expected_outcome
            current_hcg = p.cur_hcg
        alignment = identity_alignment(ref_hcg, current_hcg)
        return PerturbedCur(
            cur_hcg=current_hcg,
            alignment=alignment,
            perturbation_chain=tuple([self.name + ":"] + chain_entries),
            expected_outcome=most_severe,
            notes={
                "perturbation": self.name,
                "links": list(chosen),
                "link_notes": notes_chain,
            },
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _make_registry() -> dict[str, Perturbation]:
    return {
        op.name: op
        for op in (
            IdentityPerturbation(),
            PinSwapSymmetricPerturbation(),
            WrongConnectionPerturbation(),
            PinReversedPerturbation(),
            # Phase B
            MissingComponentPerturbation(),
            ExtraComponentPerturbation(),
            FloatingNetPerturbation(),
            ShortCircuitPerturbation(),
            PowerSwappedPerturbation(),
            InputOutputSwappedPerturbation(),
            ExtraWireBridgePerturbation(),
            ChainedPerturbation(),
        )
    }


PERTURBATION_REGISTRY: dict[str, Perturbation] = _make_registry()


def get_perturbation(name: str) -> Perturbation:
    """Look up a perturbation by name."""

    if name not in PERTURBATION_REGISTRY:
        raise KeyError(
            f"unknown perturbation {name!r}; available: "
            f"{sorted(PERTURBATION_REGISTRY)}"
        )
    return PERTURBATION_REGISTRY[name]


def apply_perturbation(
    name: str,
    ref_hcg: HeteroCircuitGraph,
    seed: int = 0,
    *,
    subtype_by_source_id: dict[str, str] | None = None,
) -> PerturbedCur:
    """便利：按名字应用单个 perturbation。"""

    op = get_perturbation(name)
    rng = random.Random(seed)
    return op.apply(ref_hcg, rng, subtype_by_source_id=subtype_by_source_id)


__all__ = [
    "PerturbedCur",
    "Perturbation",
    # Phase A
    "IdentityPerturbation",
    "PinSwapSymmetricPerturbation",
    "WrongConnectionPerturbation",
    "PinReversedPerturbation",
    # Phase B
    "MissingComponentPerturbation",
    "ExtraComponentPerturbation",
    "FloatingNetPerturbation",
    "ShortCircuitPerturbation",
    "PowerSwappedPerturbation",
    "InputOutputSwappedPerturbation",
    "ExtraWireBridgePerturbation",
    "ChainedPerturbation",
    # Registry / helpers
    "PERTURBATION_REGISTRY",
    "get_perturbation",
    "apply_perturbation",
    "hcg_to_nx",
    "hcg_to_raw_pin_edges",
    "RawPinEdge",
]
