"""LabGuardian GNN 比较模块（learning-guided graph comparison）。

当前阶段：**P0 · Schema**。仅暴露图常量、中间数据结构与 NetworkX → 异构图
的构建器。模型 / 训练 / 推理在后续阶段实现，参见
``~/.claude/plans/labguardian-server-glowing-galaxy.md``。

公开 API：

- 图常量：``ComponentType`` / ``PortType`` / ``NetRole`` / ``PolarityClass``
  / ``SourceType`` / ``COMPONENT_FEAT_DIM`` / ``PORT_FEAT_DIM`` /
  ``NET_FEAT_DIM``
- 中间结构：``HeteroCircuitGraph`` 及其节点 / 边 dataclass
- 构造器：``build_hetero_circuit_graph`` / ``build_from_logical_reference``
  / ``build_from_netlist_v2``
- 推理 stub：``GNNAdvisor.get()`` 暂抛 NotImplementedError，``should_use_gnn``
  暂返回 False（P4 实现）

**禁止**在 P0 阶段引入 torch / torch_geometric。
"""

from __future__ import annotations

from typing import Any

from app.domain.gnn.alignment import (
    ComponentAlignment,
    alignment_from_dict_payload,
    alignment_from_dicts,
    identity_alignment,
)
from app.domain.gnn.graph_schema import (
    COMPONENT_FEAT_DIM,
    IC_PIN_MAPS,
    NET_FEAT_DIM,
    PACKAGE_PIN_SPECS,
    PORT_FEAT_DIM,
    ComponentType,
    ConnectionPolicy,
    NetRole,
    PinSpec,
    PolarityClass,
    PortType,
    SourceType,
    get_expected_pin_specs,
)
from app.domain.gnn.hetero_circuit import (
    ComponentNode,
    HeteroCircuitGraph,
    NetNode,
    PortConnectsNetEdge,
    PortNode,
)
from app.domain.gnn.label_builder import (
    SCHEMA_VERSION,
    CoverageError,
    LabelBuildResult,
    LabelSource,
    LabelStats,
    SealSample,
    SealSampleGroup,
    TaskType,
    assert_observed_edges_covered,
    build_seal_samples,
    build_seal_samples_with_coverage_check,
    deserialize_label_build_result,
    serialize_label_build_result,
)
from app.domain.gnn.label_manifest import (
    LabelManifest,
    assert_manifest_healthy,
)
from app.domain.gnn.port_graph import (
    build_from_logical_reference,
    build_from_netlist_v2,
    build_hetero_circuit_graph,
)
from app.domain.gnn.seal_subgraph import (
    SealSubgraph,
    extract_seal_subgraph,
    extract_subgraphs_for_floating_ports,
    extract_subgraphs_for_observed_edges,
)


class GNNAdvisor:
    """GNN 推理入口（P4 实现）。

    P0 阶段保留类骨架以固定外部 import 路径，避免 P4 时改动 callsite。任何
    早期调用都必须显式失败而不是悄悄返回 None。
    """

    @classmethod
    def get(cls) -> GNNAdvisor:
        raise NotImplementedError(
            "GNNAdvisor will be implemented in plan phase P4. "
            "P0 only ships the graph schema."
        )


def should_use_gnn(_ctx: Any) -> bool:
    """P0 stub：永远返回 False，让 orchestrator 走纯规则路径。

    P4 替换为 plan §七 的真实触发逻辑。
    """

    return False


__all__ = [
    # enums
    "ComponentType",
    "PortType",
    "NetRole",
    "PolarityClass",
    "SourceType",
    "ConnectionPolicy",
    # dims
    "COMPONENT_FEAT_DIM",
    "PORT_FEAT_DIM",
    "NET_FEAT_DIM",
    # IC metadata
    "IC_PIN_MAPS",
    # package specs (P0.6)
    "PinSpec",
    "PACKAGE_PIN_SPECS",
    "get_expected_pin_specs",
    # data classes
    "HeteroCircuitGraph",
    "ComponentNode",
    "PortNode",
    "NetNode",
    "PortConnectsNetEdge",
    # builders
    "build_hetero_circuit_graph",
    "build_from_logical_reference",
    "build_from_netlist_v2",
    # SEAL pipeline (P0.7)
    "SealSubgraph",
    "extract_seal_subgraph",
    "extract_subgraphs_for_observed_edges",
    "extract_subgraphs_for_floating_ports",
    # Alignment + Label Builder (P0.8)
    "ComponentAlignment",
    "identity_alignment",
    "alignment_from_dicts",
    "alignment_from_dict_payload",
    "TaskType",
    "LabelSource",
    "SealSample",
    "SealSampleGroup",
    "LabelStats",
    "LabelBuildResult",
    "build_seal_samples",
    "build_seal_samples_with_coverage_check",
    "assert_observed_edges_covered",
    "CoverageError",
    "LabelManifest",
    "assert_manifest_healthy",
    "SCHEMA_VERSION",
    "serialize_label_build_result",
    "deserialize_label_build_result",
    # P4 stubs
    "GNNAdvisor",
    "should_use_gnn",
]
