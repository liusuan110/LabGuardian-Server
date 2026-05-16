"""GNN 模块 · ref ↔ cur 节点对齐（P0.8）

P1 perturbation 流程会从一个参考电路 (ref HeteroCircuitGraph) 衍生出多个
学生电路 (cur HeteroCircuitGraph)，需要一个**显式数据结构**告诉 label
builder："cur 中的 ``U_R_3`` 对应 ref 中的 ``R1``"、"cur 中的 ``n_07``
对应 ref 中的 ``VIN``" 等等。这就是 :class:`ComponentAlignment`。

P0.8 提供两个 constructor：

- :func:`identity_alignment` —— 同 source_id 自动对齐（用于不重命名的简单
  perturbation：pin_swap / wrong_connection / missing_component 等）。
- :func:`alignment_from_dicts` —— 显式 dict 覆盖（用于 component / net 重命
  名的 perturbation；P1 perturbation 自己构造）。

**禁止** 引入 torch / torch_geometric。本模块纯 Python。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — 只为静态类型，运行时不引入循环依赖
    from app.domain.gnn.hetero_circuit import HeteroCircuitGraph


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentAlignment:
    """ref ↔ cur 节点对齐表。

    ``ref_to_cur_*`` 是必填字段（由 constructor 提供）；``cur_to_ref_*``
    是反向缓存，``__post_init__`` 自动从正向 map 派生。

    Attributes:
        ref_to_cur_component: ref ``ComponentNode.source_id`` →
            cur ``ComponentNode.source_id``
        ref_to_cur_net: ref ``NetNode.source_id`` → cur ``NetNode.source_id``
        cur_to_ref_component: 反向缓存
        cur_to_ref_net: 反向缓存
        notes: perturbation log，例如 ``{"perturbation": "pin_swap:C1",
            "seed": 42, "unmatched_ref_components": ["R2"]}``
    """

    ref_to_cur_component: dict[str, str]
    ref_to_cur_net: dict[str, str]
    cur_to_ref_component: dict[str, str] = field(init=False, default_factory=dict)
    cur_to_ref_net: dict[str, str] = field(init=False, default_factory=dict)
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen dataclass 必须通过 object.__setattr__ 写字段。
        object.__setattr__(
            self,
            "cur_to_ref_component",
            {v: k for k, v in self.ref_to_cur_component.items()},
        )
        object.__setattr__(
            self,
            "cur_to_ref_net",
            {v: k for k, v in self.ref_to_cur_net.items()},
        )

    # -- 节点 ID 级别的映射 -----------------------------------------------

    def map_ref_port_to_cur_port_id(
        self,
        ref_port_id: str,
        ref_hcg: HeteroCircuitGraph,
        cur_hcg: HeteroCircuitGraph,
    ) -> str | None:
        """``ref_port:R1.pin1`` → ``cur_port:U_R_3.pin1`` 或 None。

        返回 None 的两种情形：
        1. ``ref_port_id`` 在 ref_hcg 中不存在（防御性）
        2. ref port 所属 component 不在对齐表中（missing_component perturbation
           会触发；label builder 应静默 skip + 计数）
        3. 期望的 cur port 不存在于 cur_hcg（cur 侧没 materialize 该 pin）
        """

        if ref_port_id not in ref_hcg.ports:
            return None
        ref_port = ref_hcg.ports[ref_port_id]
        ref_comp = ref_hcg.components.get(ref_port.parent_component_id)
        if ref_comp is None:
            return None
        cur_comp_source_id = self.ref_to_cur_component.get(ref_comp.source_id)
        if cur_comp_source_id is None:
            return None
        cur_port_id = f"cur_port:{cur_comp_source_id}.{ref_port.port_key}"
        if cur_port_id not in cur_hcg.ports:
            return None
        return cur_port_id

    def map_ref_net_to_cur_net_id(
        self,
        ref_net_id: str,
        cur_hcg: HeteroCircuitGraph,
    ) -> str | None:
        """``ref_net:VIN`` → ``cur_net:n_07`` 或 None。"""

        if not ref_net_id.startswith("ref_net:"):
            return None
        ref_source_id = ref_net_id[len("ref_net:"):]
        cur_source_id = self.ref_to_cur_net.get(ref_source_id)
        if cur_source_id is None:
            return None
        cur_net_id = f"cur_net:{cur_source_id}"
        if cur_net_id not in cur_hcg.nets:
            return None
        return cur_net_id

    # -- 反向（用于 WRONG_OBSERVED 推断 expected_edge）---------------------

    def map_cur_port_to_ref_port_id(
        self,
        cur_port_id: str,
        ref_hcg: HeteroCircuitGraph,
        cur_hcg: HeteroCircuitGraph,
    ) -> str | None:
        """``cur_port:U_R_3.pin1`` → ``ref_port:R1.pin1`` 或 None。"""

        if cur_port_id not in cur_hcg.ports:
            return None
        cur_port = cur_hcg.ports[cur_port_id]
        cur_comp = cur_hcg.components.get(cur_port.parent_component_id)
        if cur_comp is None:
            return None
        ref_comp_source_id = self.cur_to_ref_component.get(cur_comp.source_id)
        if ref_comp_source_id is None:
            return None
        ref_port_id = f"ref_port:{ref_comp_source_id}.{cur_port.port_key}"
        if ref_port_id not in ref_hcg.ports:
            return None
        return ref_port_id

    # -- 序列化（P1 perturbation log + dataset manifest 用） ----------------

    def to_dict(self) -> dict:
        """JSON-friendly dict；与 :func:`alignment_from_dict_payload` 互逆。"""

        return {
            "ref_to_cur_component": dict(self.ref_to_cur_component),
            "ref_to_cur_net": dict(self.ref_to_cur_net),
            "notes": dict(self.notes),
        }


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def identity_alignment(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
) -> ComponentAlignment:
    """按 ``source_id`` 同名自动对齐。

    适用场景：perturbation 没有重命名 component / net（如 pin_swap、
    pin_reversed、forbidden_violation 等），ref / cur 共享 source_id 空间。

    notes 中会记录两侧的"未匹配"集合，便于 label builder 推断 missing_component
    数量与 dataset 健康度检查。
    """

    ref_comp_sids = {c.source_id for c in ref_hcg.components.values()}
    cur_comp_sids = {c.source_id for c in cur_hcg.components.values()}
    ref_to_cur_component = {
        sid: sid for sid in ref_comp_sids if sid in cur_comp_sids
    }

    ref_net_sids = {n.source_id for n in ref_hcg.nets.values()}
    cur_net_sids = {n.source_id for n in cur_hcg.nets.values()}
    ref_to_cur_net = {sid: sid for sid in ref_net_sids if sid in cur_net_sids}

    notes: dict = {
        "constructor": "identity_alignment",
        "unmatched_ref_components": sorted(ref_comp_sids - cur_comp_sids),
        "unmatched_cur_components": sorted(cur_comp_sids - ref_comp_sids),
        "unmatched_ref_nets": sorted(ref_net_sids - cur_net_sids),
        "unmatched_cur_nets": sorted(cur_net_sids - ref_net_sids),
    }
    return ComponentAlignment(
        ref_to_cur_component=ref_to_cur_component,
        ref_to_cur_net=ref_to_cur_net,
        notes=notes,
    )


def alignment_from_dicts(
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
    component_map: dict[str, str],
    net_map: dict[str, str],
    extra_notes: dict | None = None,
) -> ComponentAlignment:
    """显式覆盖。``component_map`` / ``net_map`` 直接作为
    ``ref_to_cur_*`` 字段。

    输入 dict 中**仅**保留两侧 hcg 都实际存在的 source_id；其余在 notes
    中记录为 ``unmatched_*``。
    """

    ref_comp_sids = {c.source_id for c in ref_hcg.components.values()}
    cur_comp_sids = {c.source_id for c in cur_hcg.components.values()}
    ref_net_sids = {n.source_id for n in ref_hcg.nets.values()}
    cur_net_sids = {n.source_id for n in cur_hcg.nets.values()}

    filtered_comp = {
        k: v
        for k, v in component_map.items()
        if k in ref_comp_sids and v in cur_comp_sids
    }
    filtered_net = {
        k: v
        for k, v in net_map.items()
        if k in ref_net_sids and v in cur_net_sids
    }

    notes: dict = {
        "constructor": "alignment_from_dicts",
        "unmatched_ref_components": sorted(
            set(component_map) - filtered_comp.keys()
        ),
        "unmatched_cur_components": sorted(
            cur_comp_sids - set(filtered_comp.values())
        ),
        "unmatched_ref_nets": sorted(set(net_map) - filtered_net.keys()),
        "unmatched_cur_nets": sorted(cur_net_sids - set(filtered_net.values())),
    }
    if extra_notes:
        notes.update(extra_notes)

    return ComponentAlignment(
        ref_to_cur_component=filtered_comp,
        ref_to_cur_net=filtered_net,
        notes=notes,
    )


def alignment_from_dict_payload(
    payload: dict,
    ref_hcg: HeteroCircuitGraph,
    cur_hcg: HeteroCircuitGraph,
) -> ComponentAlignment:
    """``ComponentAlignment.to_dict()`` 的逆：从 JSON-friendly dict 重建。"""

    return alignment_from_dicts(
        ref_hcg,
        cur_hcg,
        component_map=payload.get("ref_to_cur_component", {}),
        net_map=payload.get("ref_to_cur_net", {}),
        extra_notes=payload.get("notes", {}),
    )


__all__ = [
    "ComponentAlignment",
    "identity_alignment",
    "alignment_from_dicts",
    "alignment_from_dict_payload",
]
