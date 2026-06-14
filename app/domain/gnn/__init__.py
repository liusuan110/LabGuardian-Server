"""LabGuardian GNN 比较模块（learning-guided graph comparison）。

当前阶段：**P0 · Schema**。仅暴露图常量、中间数据结构与 NetworkX → 异构图
的构建器。模型 / 训练 / 推理在后续阶段实现，参见
``docs/plans/labguardian-server-glowing-galaxy.md``。

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

from dataclasses import dataclass
from typing import Any

from app.domain.gnn.alignment import (
    ComponentAlignment,
    alignment_from_dict_payload,
    alignment_from_dicts,
    identity_alignment,
)
from app.domain.gnn.dataset_builder import (
    DatasetSpec,
    DatasetSpecError,
    PerturbationPlan,
    RefSpec,
    generate_dataset,
    validate_dataset_spec,
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
from app.domain.gnn.perturbation import (
    PERTURBATION_REGISTRY,
    ChainedPerturbation,
    ExtraComponentPerturbation,
    ExtraWireBridgePerturbation,
    FloatingNetPerturbation,
    IdentityPerturbation,
    InputOutputSwappedPerturbation,
    MissingComponentPerturbation,
    Perturbation,
    PerturbedCur,
    PinReversedPerturbation,
    PinSwapSymmetricPerturbation,
    PowerSwappedPerturbation,
    ShortCircuitPerturbation,
    WrongConnectionPerturbation,
    apply_perturbation,
    get_perturbation,
    hcg_to_nx,
    hcg_to_raw_pin_edges,
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

# P2 PyG bridge — guarded behind torch/PyG availability so callers that
# only need the schema / label_builder / dataset_builder layers (which are
# pure Python) can import this package even on a CPU-only / no-extras box.
try:
    from app.domain.gnn.backbone import (
        HeteroNodeEncoder,
        HeteroSAGEBackbone,
        embeddings_for_subgraph,
    )
    from app.domain.gnn.model import CircuitMatchNet
    from app.domain.gnn.prebaked_dataset import (
        PREBAKED_SCHEMA_VERSION,
        PrebakedSealDataset,
        PrebakeStats,
        prebake_to_disk,
    )
    from app.domain.gnn.pretrain_dataset import SpiceNetlistPretrainDataset
    from app.domain.gnn.pyg_converter import (
        encode_component_features,
        encode_net_features,
        encode_port_features,
        encode_port_net_edge_features,
        seal_subgraph_to_pyg_data,
        to_hetero_data,
    )
    from app.domain.gnn.pyg_dataset import (
        FlatSealDataset,
        RefEntry,
        RefRegistry,
        reconstruct_cur_hcg,
    )
    from app.domain.gnn.seal_dgcnn import SealDGCNN, predict_prob
    from app.domain.gnn.spicenetlist_loader import (
        COMPONENT_TYPE_MAP,
        SpiceNetlistCircuit,
        load_circuit_json,
        load_spicenetlist_dir,
    )

    _PYG_AVAILABLE = True
except ImportError:  # torch / torch_geometric not installed
    _PYG_AVAILABLE = False

from app.domain.gnn.splits import (
    DatasetSplits,
    SplitsError,
    SplitSpec,
    build_splits,
    discover_samples,
    load_splits,
    write_splits,
)


# P4 inference layer — guarded behind torch availability (same as P2 / P3).
# The shim below lets non-extra installs still import the package; calling
# GNNAdvisor.get() without torch raises a clear RuntimeError.
try:
    from app.domain.gnn.inference import (
        GNNAdvice,
        GNNAdvisor,
        should_use_gnn,
    )
except ImportError:  # torch missing — fall back to stub raising on use
    @dataclass(frozen=True)  # type: ignore[no-redef]
    class GNNAdvice:  # noqa: D101
        model_version: str = ""
        inference_ms: float = 0.0
        n_edges_scored: int = 0
        enabled: bool = False

    class GNNAdvisor:  # type: ignore[no-redef]
        """Stub raised when torch is missing — match P4 API surface so
        ``from app.domain.gnn import GNNAdvisor`` works even without
        ``[gnn]`` extras installed."""

        @classmethod
        def get(cls) -> "GNNAdvisor":
            raise RuntimeError(
                "GNNAdvisor requires the [gnn] extra (torch + "
                "torch_geometric). Install with: pip install -e '.[gnn]'"
            )

        @classmethod
        def checkpoint_available(cls) -> bool:
            return False

    def should_use_gnn(_ctx: Any) -> bool:  # type: ignore[no-redef]
        """No GNN available → never use it."""
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
    # P1 Phase A: perturbation + dataset_builder
    "PerturbedCur",
    "Perturbation",
    "IdentityPerturbation",
    "PinSwapSymmetricPerturbation",
    "WrongConnectionPerturbation",
    "PinReversedPerturbation",
    # P1 Phase B: additional perturbation operators
    "MissingComponentPerturbation",
    "ExtraComponentPerturbation",
    "FloatingNetPerturbation",
    "ShortCircuitPerturbation",
    "PowerSwappedPerturbation",
    "InputOutputSwappedPerturbation",
    "ExtraWireBridgePerturbation",
    "ChainedPerturbation",
    "PERTURBATION_REGISTRY",
    "get_perturbation",
    "apply_perturbation",
    "hcg_to_nx",
    "hcg_to_raw_pin_edges",
    "RefSpec",
    "PerturbationPlan",
    "DatasetSpec",
    "DatasetSpecError",
    "validate_dataset_spec",
    "generate_dataset",
    # P1 Phase C: splits
    "SplitSpec",
    "DatasetSplits",
    "SplitsError",
    "discover_samples",
    "build_splits",
    "write_splits",
    "load_splits",
    # P2 PyG (guarded — only importable if torch + torch_geometric extras)
    "to_hetero_data",
    "encode_component_features",
    "encode_port_features",
    "encode_net_features",
    "encode_port_net_edge_features",
    "seal_subgraph_to_pyg_data",
    "RefEntry",
    "RefRegistry",
    "FlatSealDataset",
    "reconstruct_cur_hcg",
    # P2.5 SpiceNetlist pretrain
    "SpiceNetlistCircuit",
    "load_circuit_json",
    "load_spicenetlist_dir",
    "COMPONENT_TYPE_MAP",
    "SpiceNetlistPretrainDataset",
    "SealDGCNN",
    "predict_prob",
    # P3 multi-task wrapper
    "CircuitMatchNet",
    # P3.1 L1 HeteroConv backbone (standalone module, integration deferred)
    "HeteroNodeEncoder",
    "HeteroSAGEBackbone",
    "embeddings_for_subgraph",
    # P3.2 prebaked dataset (data pipeline 25x speedup)
    "PrebakedSealDataset",
    "PrebakeStats",
    "prebake_to_disk",
    "PREBAKED_SCHEMA_VERSION",
    # P4 inference integration
    "GNNAdvisor",
    "GNNAdvice",
    "should_use_gnn",
]
