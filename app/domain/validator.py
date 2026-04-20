"""
电路验证器 (← src_v2/logic/validator.py)

多级对比管线: L0 预检 → L1 全图同构 → L2 子图 → L2.5 极性 → L3 GED
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from .board_schema import BoardSchema
from .circuit import (
    CircuitAnalyzer,
    Polarity,
    POLARIZED_TYPES,
    norm_component_type,
)
from .netlist_models import ComponentInstance, PinAssignment

logger = logging.getLogger(__name__)


def _graph_signature(g: nx.Graph) -> Tuple:
    """图结构不变量签名 — O(1) 快速排除不可能的同构"""
    degrees = sorted([d for _, d in g.degree()], reverse=True)
    comp_types = Counter(
        d.get("ctype", "NET")
        for _, d in g.nodes(data=True)
        if d.get("kind") == "comp"
    )
    net_count = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "net")
    return (
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(degrees),
        tuple(sorted(comp_types.items())),
        net_count,
    )


class CircuitValidator:
    """电路比对与诊断系统。

    多级渐进式验证管线 L0 → L1 → L2 → L2.5 → L3:
    - L0: 元件数量层级预检（快速元器件计数对比）
    - L1: 全图同构验证（拓扑图完全一致性检查，包括极性与引脚角色）
    - L2: 子图同构容忍（允许当前电路是参考电路的有效子集）
    - L2.5: 极性细粒度扫描（正反向、极性未知状态诊断）
    - L3: 图编辑距离相似度（启发式或精确GED，用于评分与排序）
    
    业务能力三要素：
    1. compare(): 当前电路 vs 参考电路的分级对比
    2. diagnose(): 独立电路风险诊断（无需参考即可检出空载/短路）
    3. validator_report_v2: 生成结构化诊断报告（供RAG/Agent下游消费）

    Attributes:
        ref_graph: 参考电路的 NetworkX 图表示。
        ref_component_instances: 参考电路被平铺解析的组件实例集合。
        ref_netlist_v2: 参考电路底层承载的无向网表黑盒。
        ref_topology: 解析后的高层拓扑逻辑图（用于同构检测）。
    """

    def __init__(self):
        self.ref_graph: Optional[nx.Graph] = None
        self.ref_component_instances: List[ComponentInstance] = []
        self.ref_netlist_v2: Optional[Dict] = None
        self.ref_topology: Optional[nx.Graph] = None

    @property
    def has_reference(self) -> bool:
        return bool(self.ref_component_instances)

    def set_reference(self, analyzer: CircuitAnalyzer):
        """将当前电路设为 Golden Reference"""
        self.ref_graph = analyzer.graph.copy()
        self.ref_component_instances = list(analyzer.component_instances)
        self.ref_netlist_v2 = analyzer.export_netlist_v2(scene_id="reference_scene")
        try:
            self.ref_topology = analyzer.build_topology_graph()
        except Exception:
            self.ref_topology = None

    def save_reference(self, file_path: str):
        if not self.has_reference:
            raise ValueError("No reference circuit set.")
        topo_payload = None
        if self.ref_topology is not None:
            topo_payload = json_graph.node_link_data(self.ref_topology)
        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "format": "labguardian_ref_v4",
            },
            "components": [],
            "netlist_v2": self.ref_netlist_v2,
            "topology": topo_payload,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_reference(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._load_reference_payload(payload)

    def load_reference_payload(self, payload: Dict):
        """从内联 dict 加载 reference.

        支持两种输入:
        1. 完整 `labguardian_ref_v4` payload, 包含 `netlist_v2`
        2. 直接传 `netlist_v2` 对象
        """
        self._load_reference_payload(payload)

    def _load_reference_payload(self, payload: Dict):
        if not isinstance(payload, dict):
            raise ValueError("Reference payload must be a dict.")

        if "netlist_v2" in payload:
            netlist_v2 = payload.get("netlist_v2")
            topo_data = payload.get("topology")
        elif "components" in payload and "nets" in payload:
            netlist_v2 = payload
            topo_data = None
        else:
            raise ValueError("Reference payload must contain netlist_v2 or be a netlist_v2 object.")

        self.ref_netlist_v2 = netlist_v2
        if not self.ref_netlist_v2:
            raise ValueError("Reference payload must contain netlist_v2 in labguardian_ref_v4 format.")

        self.ref_component_instances = []
        netlist_v2 = self.ref_netlist_v2 or {}
        board_schema_id = netlist_v2.get("board_schema_id", "breadboard_legacy_v1")
        board_schema = BoardSchema.default_breadboard()
        if board_schema.schema_id != board_schema_id:
            logger.info("Reference requested board schema %s, fallback to default schema", board_schema_id)

        tmp = CircuitAnalyzer(board_schema=board_schema)
        for item in netlist_v2.get("components", []):
            instance = _component_instance_from_dict(item)
            tmp.add_component_instance(instance)

        self.ref_graph = tmp.graph.copy()
        self.ref_component_instances = list(tmp.component_instances)
        if topo_data:
            try:
                self.ref_topology = json_graph.node_link_graph(topo_data)
                return
            except Exception:
                self.ref_topology = None
        try:
            self.ref_topology = tmp.build_topology_graph()
        except Exception:
            self.ref_topology = None

    def compare(self, curr_analyzer: CircuitAnalyzer) -> Dict:
        """分级比对管线：L0→L1→L2→L2.5→L3，生成结构化诊断报告。

        管线回退逻辑:
        1. L0 快速排除：元件数量统计，发现明显缺失/多余。
        2. v2组件级细粒度比对：O(N^2)贪心匹配，检查孔位/节点/极性差异。
        3. L1-L3拓扑同构：
           - `v2`组件级检查优先，但拓扑同构负责确保"电路物理意义上连通"。
           - 两者并存（"拓扑对但孔位不同"是比赛中允许存在的重要场景）。
           - 逐级尝试全同构（含极性）、子图同构（仅类型）、独立极性校验。
           - 若全败，最后使用 GED 计算整体完成度得分 (<50 节点精确, 否则启发式)。
        
        Args:
            curr_analyzer: 当前已解析待验的 CircuitAnalyzer 实例。

        Returns:
            Dict: validator_report_v2 结构化汇报，包含进度 progress 与诊断数组项 diagnostic_items。
        """
        result = {
            "errors": [],
            "missing_links": [],
            "extra_links": [],
            "is_match": False,
            "similarity": 0.0,
            "progress": 0.0,
            "matched_components": [],
            "missing_components": [],
            "extra_components": [],
            "polarity_errors": [],
            "pin_mismatches": [],
            "hole_mismatches": [],
            "node_mismatches": [],
            "component_mismatches": [],
            "topology_errors": [],
            "matched_component_pairs": [],
            "diagnostic_items": [],
            "report": {
                "version": "validator_report_v2",
                "items": [],
                "topology_errors": [],
                "node_errors": [],
                "hole_errors": [],
                "polarity_errors": [],
                "component_errors": [],
                "summary": {},
            },
        }

        if not self.has_reference:
            _append_report(
                result,
                "component_errors",
                "No reference circuit set.",
                error_code="REFERENCE_NOT_SET",
                severity="error",
            )
            return result

        # L0: 元件数量统计
        ref_counts = Counter(c.component_type for c in self.ref_component_instances)
        curr_counts = Counter(c.component_type for c in curr_analyzer.component_instances)
        for t in sorted(set(ref_counts.keys()) | set(curr_counts.keys())):
            r_c, c_c = ref_counts[t], curr_counts[t]
            if c_c < r_c:
                _append_report(
                    result,
                    "component_errors",
                    f"Missing {r_c - c_c} x {t}",
                    error_code="COMPONENT_MISSING",
                    severity="error",
                    expected=r_c,
                    actual=c_c,
                    context={"component_type": t},
                )
                result["missing_components"].extend([t] * (r_c - c_c))
            elif c_c > r_c:
                _append_report(
                    result,
                    "component_errors",
                    f"Extra {c_c - r_c} x {t}",
                    error_code="COMPONENT_EXTRA",
                    severity="error",
                    expected=r_c,
                    actual=c_c,
                    context={"component_type": t},
                )
                result["extra_components"].extend([t] * (c_c - r_c))

        v2_exact = None
        v2_hole_only = False
        has_v2_polarity = False
        v2_cmp = self._compare_component_instances(curr_analyzer.component_instances)
        result["matched_components"] = v2_cmp["matched_components"]
        result["matched_component_pairs"] = v2_cmp["matched_component_pairs"]
        result["pin_mismatches"].extend(v2_cmp["pin_mismatches"])
        result["hole_mismatches"].extend(v2_cmp["hole_mismatches"])
        result["node_mismatches"].extend(v2_cmp["node_mismatches"])
        result["component_mismatches"].extend(v2_cmp["component_mismatches"])
        for item in v2_cmp["component_diagnostics"]:
            _append_diagnostic(result, item)
        for item in v2_cmp["node_diagnostics"]:
            _append_diagnostic(result, item)
        for item in v2_cmp["hole_diagnostics"]:
            _append_diagnostic(result, item)
        for item in v2_cmp["polarity_diagnostics"]:
            _append_diagnostic(result, item)
        if result.get("progress", 0.0) < v2_cmp["progress"]:
            result["progress"] = v2_cmp["progress"]
        if result.get("similarity", 0.0) < v2_cmp["similarity"]:
            result["similarity"] = v2_cmp["similarity"]
        v2_exact = v2_cmp["is_exact_match"]
        v2_hole_only = v2_cmp["is_hole_placement_only_mismatch"]
        has_v2_polarity = bool(v2_cmp["polarity_diagnostics"])

        # L1-L3
        polarity_checked = has_v2_polarity
        try:
            if self.ref_topology is not None:
                curr_topo = curr_analyzer.build_topology_graph()
                ref_sig = _graph_signature(self.ref_topology)
                cur_sig = _graph_signature(curr_topo)

                if ref_sig == cur_sig:
                    from networkx.algorithms.isomorphism import GraphMatcher

                    gm = GraphMatcher(
                        self.ref_topology,
                        curr_topo,
                        node_match=self._node_match_full,
                        edge_match=self._edge_match,
                    )
                    if gm.is_isomorphic():
                        if not has_v2_polarity:
                            self._check_polarity_errors(result, curr_topo)
                            polarity_checked = True
                        else:
                            polarity_checked = True
                        if v2_exact is False:
                            result["similarity"] = max(result["similarity"], 0.95)
                            result["progress"] = max(result["progress"], 1.0)
                            if v2_hole_only:
                                _append_report(
                                    result,
                                    "topology_errors",
                                    "Topology matches reference, but pin placement differs.",
                                    error_code="TOPOLOGY_MATCH_PIN_PLACEMENT_DIFFERS",
                                    severity="warning",
                                )
                        elif not result["report"].get("items"):
                            result["errors"] = ["Topology matches reference."]
                            result["is_match"] = True
                            result["similarity"] = 1.0
                            result["progress"] = 1.0
                            if not result["matched_components"]:
                                result["matched_components"] = [c.component_id for c in self.ref_component_instances]
                            _finalize_report(result)
                            return result

                self._check_subgraph_match(result, curr_topo)
                if not polarity_checked:
                    self._check_polarity_errors(result, curr_topo)
                self._compute_ged_similarity(result, curr_topo)
        except Exception as e:
            _append_report(
                result,
                "topology_errors",
                f"Topology check failed: {e}",
                error_code="TOPOLOGY_CHECK_FAILED",
                severity="error",
            )
            logger.exception("Topology check failed")

        if not result["is_match"]:
            self._heuristic_position_match(result, curr_analyzer)

        if not result["errors"]:
            result["errors"].append("Circuit matches Reference!")
            result["is_match"] = True

        _finalize_report(result)
        return result

    def _compare_component_instances(self, curr_instances: List[ComponentInstance]) -> Dict:
        """核心组件级 O(N²) 二部图匹配与细粒度硬件级差异挖掘。

        通过贪心策略将参考部件与距离最近（基于行定位与类型等权重）的
        新部件进行配对。匹配完成后深入进行以下比对：
        1. 对称引脚组对比（允许等价脚互换）
        2. 孔位/节点匹配度差异（判断走线跳线偏差）
        3. 细粒度极性倒置判断

        Args:
            curr_instances: 解析自新电路图像拓扑的组件实例集。

        Returns:
            字典：包含各类匹配列表 (matched_pairs), 每类错误的扁平集合，
            以及相似度进度 (`progress`, `similarity`) 评分子指标。
        """
        ref_instances = list(self.ref_component_instances)
        unmatched_curr = set(range(len(curr_instances)))
        matched_components: List[str] = []
        matched_pairs: List[Dict[str, str]] = []
        pin_mismatches: List[str] = []
        hole_mismatches: List[str] = []
        node_mismatches: List[str] = []
        component_mismatches: List[str] = []
        hole_diagnostics: List[Dict] = []
        node_diagnostics: List[Dict] = []
        component_diagnostics: List[Dict] = []
        polarity_diagnostics: List[Dict] = []
        exact_pins = 0
        total_ref_pins = sum(len(comp.pins) for comp in ref_instances) or 1

        for ref_comp in ref_instances:
            best_idx = None
            best_cost = float("inf")
            for idx in unmatched_curr:
                curr_comp = curr_instances[idx]
                cost = _component_match_cost(ref_comp, curr_comp)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx is None or best_cost >= 1000:
                message = f"{ref_comp.component_id}: 未找到匹配的 {ref_comp.component_type} 组件实例"
                component_mismatches.append(message)
                component_diagnostics.append(
                    _make_diagnostic(
                        category="component_errors",
                        error_code="COMPONENT_INSTANCE_MISSING",
                        message=message,
                        severity="error",
                        component_id=ref_comp.component_id,
                        expected=ref_comp.component_type,
                        actual=None,
                    )
                )
                continue

            curr_comp = curr_instances[best_idx]
            unmatched_curr.remove(best_idx)
            matched_components.append(ref_comp.component_id)
            matched_pairs.append(
                {
                    "reference_component_id": ref_comp.component_id,
                    "current_component_id": curr_comp.component_id,
                }
            )

            pin_result = _compare_component_pins(ref_comp, curr_comp)
            exact_pins += pin_result["exact_pin_matches"]
            pin_mismatches.extend(pin_result["pin_mismatches"])
            hole_mismatches.extend(pin_result["hole_mismatches"])
            node_mismatches.extend(pin_result["node_mismatches"])
            component_mismatches.extend(pin_result["component_mismatches"])
            hole_diagnostics.extend(pin_result["hole_diagnostics"])
            node_diagnostics.extend(pin_result["node_diagnostics"])
            component_diagnostics.extend(pin_result["component_diagnostics"])
            if _needs_polarity_check(ref_comp):
                ref_pol = ref_comp.polarity
                curr_pol = curr_comp.polarity
                if ref_pol in ("forward", "reverse") and curr_pol in ("forward", "reverse") and ref_pol != curr_pol:
                    polarity_diagnostics.append(
                        _make_diagnostic(
                            category="polarity_errors",
                            error_code="POLARITY_REVERSED",
                            message=f"{ref_comp.component_id}: 极性反接，期望 {ref_pol}，当前 {curr_pol}",
                            severity="error",
                            component_id=ref_comp.component_id,
                            current_component_id=curr_comp.component_id,
                            expected=ref_pol,
                            actual=curr_pol,
                        )
                    )
                elif ref_pol in ("forward", "reverse") and curr_pol == "unknown":
                    polarity_diagnostics.append(
                        _make_diagnostic(
                            category="polarity_errors",
                            error_code="POLARITY_UNKNOWN",
                            message=f"{ref_comp.component_id}: 极性无法判断，期望 {ref_pol}",
                            severity="warning",
                            component_id=ref_comp.component_id,
                            current_component_id=curr_comp.component_id,
                            expected=ref_pol,
                            actual=curr_pol,
                        )
                    )

        progress = len(matched_components) / len(ref_instances) if ref_instances else 0.0
        similarity = exact_pins / total_ref_pins if total_ref_pins else 0.0
        return {
            "matched_components": matched_components,
            "matched_component_pairs": matched_pairs,
            "pin_mismatches": list(dict.fromkeys(pin_mismatches)),
            "hole_mismatches": list(dict.fromkeys(hole_mismatches)),
            "node_mismatches": list(dict.fromkeys(node_mismatches)),
            "component_mismatches": list(dict.fromkeys(component_mismatches)),
            "hole_diagnostics": hole_diagnostics,
            "node_diagnostics": node_diagnostics,
            "component_diagnostics": component_diagnostics,
            "polarity_diagnostics": polarity_diagnostics,
            "progress": progress,
            "similarity": similarity,
            "is_exact_match": not (pin_mismatches or hole_mismatches or polarity_diagnostics) and progress >= 1.0 and similarity >= 1.0,
            "is_hole_placement_only_mismatch": bool(hole_diagnostics) and not (node_diagnostics or component_diagnostics or polarity_diagnostics),
        }

    # ---- VF2++ 回调 ----

    @staticmethod
    def _node_match_full(a: dict, b: dict) -> bool:
        """VF2++ 同构匹配钩子：要求类型、极性和电源网络等价。"""
        if a.get("kind") != b.get("kind"):
            return False
        if a.get("kind") == "comp":
            if a.get("ctype") != b.get("ctype"):
                return False
            ref_pol = a.get("polarity", "none")
            cur_pol = b.get("polarity", "none")
            if ref_pol in ("forward", "reverse") and cur_pol in ("forward", "reverse"):
                if ref_pol != cur_pol:
                    return False
            return True
        if a.get("kind") == "net":
            ref_power = a.get("power")
            cur_power = b.get("power")
            if ref_power and cur_power:
                return ref_power == cur_power
        return True

    @staticmethod
    def _node_match_no_polarity(a: dict, b: dict) -> bool:
        if a.get("kind") != b.get("kind"):
            return False
        if a.get("kind") == "comp":
            return a.get("ctype") == b.get("ctype")
        return True

    @staticmethod
    def _node_match_type_only(a: dict, b: dict) -> bool:
        """VF2++ 降级同构钩子：仅要求类型一致，忽略极性。"""
        if a.get("kind") != b.get("kind"):
            return False
        if a.get("kind") == "comp":
            return a.get("ctype") == b.get("ctype")
        return True

    @staticmethod
    def _edge_match(a: dict, b: dict) -> bool:
        """VF2++ 边同构钩子：检查引脚角色标注（如 Base/Collector 等）一致性。"""
        ref_role = a.get("pin_role")
        cur_role = b.get("pin_role")
        if ref_role is None or cur_role is None:
            return True
        return ref_role == cur_role

    # ---- L2: 子图同构 ----

    def _check_subgraph_match(self, result: Dict, curr_topo: nx.Graph):
        """L2 子图同构容错检测。

        当全图同构（组件多一根线或少一根）无法覆盖时，检查"参考要求"是否是"当前成品"
        的一个合法子集。如果匹配，意味着同学多接了组件，或使用了中间中转跳线，
        这是教育与竞技环境里允许的有效错误，不应直接判零。
        """
        if self.ref_topology is None:
            return
        from networkx.algorithms.isomorphism import GraphMatcher

        gm = GraphMatcher(
            self.ref_topology,
            curr_topo,
            node_match=self._node_match_type_only,
        )
        if gm.subgraph_is_isomorphic():
            mapping = gm.mapping
            matched_ref_comps = set()
            for ref_node in mapping:
                data = self.ref_topology.nodes[ref_node]
                if data.get("kind") == "comp":
                    matched_ref_comps.add(ref_node)
            total_ref_comps = sum(
                1 for _, d in self.ref_topology.nodes(data=True) if d.get("kind") == "comp"
            )
            progress = len(matched_ref_comps) / total_ref_comps if total_ref_comps > 0 else 0.0
            result["progress"] = progress

            all_ref_comps = set(
                n for n, d in self.ref_topology.nodes(data=True) if d.get("kind") == "comp"
            )
            for comp_node in all_ref_comps - matched_ref_comps:
                ctype = self.ref_topology.nodes[comp_node].get("ctype", "?")
                result["missing_components"].append(ctype)

            if progress < 1.0:
                _append_report(
                    result,
                    "topology_errors",
                    f"Circuit is a valid subset (progress: {progress:.0%}, "
                    f"{len(matched_ref_comps)}/{total_ref_comps} matched)",
                    error_code="TOPOLOGY_VALID_SUBSET",
                    severity="warning",
                    expected=total_ref_comps,
                    actual=len(matched_ref_comps),
                )
            result["matched_components"] = [
                self.ref_topology.nodes[n].get("ctype", "?") for n in matched_ref_comps
            ]

    # ---- L2.5: 极性诊断 ----

    def _check_polarity_errors(self, result: Dict, curr_topo: nx.Graph):
        """L2.5 极性细粒度扫描。

        如果拓扑结构基本一致，但纯极性接反（如 LED 正负对调）不应作为
        同构失败被忽略，而应该被记录为单独的错误报告。
        本方法用在全同构宽松检测（跳过极性）之后专门锁定此类引脚错误。
        """
        if self.ref_topology is None:
            return
        from networkx.algorithms.isomorphism import GraphMatcher

        gm = GraphMatcher(
            self.ref_topology, curr_topo, node_match=self._node_match_no_polarity
        )
        if gm.is_isomorphic():
            mapping = gm.mapping
            for ref_node, curr_node in mapping.items():
                ref_data = self.ref_topology.nodes[ref_node]
                cur_data = curr_topo.nodes[curr_node]
                if ref_data.get("kind") != "comp":
                    continue
                ref_pol = ref_data.get("polarity", "none")
                cur_pol = cur_data.get("polarity", "none")
                ctype = ref_data.get("ctype", "?")
                if ref_pol in ("forward", "reverse") and cur_pol in ("forward", "reverse"):
                    if ref_pol != cur_pol:
                        _append_report(
                            result,
                            "polarity_errors",
                            f"{ctype} ({curr_node}) 极性反接",
                            error_code="POLARITY_REVERSED",
                            severity="error",
                            expected=ref_pol,
                            actual=cur_pol,
                            context={"component_type": ctype, "graph_node_id": curr_node},
                        )
                elif ref_pol in ("forward", "reverse") and cur_pol == "unknown":
                    _append_report(
                        result,
                        "polarity_errors",
                        f"{ctype} ({curr_node}) 极性无法判断",
                        error_code="POLARITY_UNKNOWN",
                        severity="warning",
                        expected=ref_pol,
                        actual=cur_pol,
                        context={"component_type": ctype, "graph_node_id": curr_node},
                    )

    # ---- L3: GED 相似度 ----

    def _compute_ged_similarity(self, result: Dict, curr_topo: nx.Graph):
        """L3 图编辑距离相似度计算回退节点。

        目的：打分/定级系统最后防线。
        如果当前电路和黄金架构连部分子集也算不上（大量改接、漏连）：
        - 对 <50 个节点的图精确求解 networkx 最优 GED。
        - 对大型系统转为启发式近似计算。
        以此换得最终比赛评分界面的定序。
        """
        if self.ref_topology is None:
            return
        ref_size = self.ref_topology.number_of_nodes() + self.ref_topology.number_of_edges()
        cur_size = curr_topo.number_of_nodes() + curr_topo.number_of_edges()
        max_size = max(ref_size, cur_size, 1)

        if ref_size > 50 or cur_size > 50:
            similarity = self._approximate_ged_similarity(curr_topo)
            result["similarity"] = max(result.get("similarity", 0), similarity)
            return

        try:
            def _node_subst_cost(a, b):
                if a.get("kind") != b.get("kind"):
                    return 2.0
                if a.get("kind") == "comp":
                    if a.get("ctype") != b.get("ctype"):
                        return 1.5
                    if a.get("polarity", "none") != b.get("polarity", "none"):
                        return 0.5
                    return 0.0
                return 0.0

            best_ged = max_size
            for ged in nx.optimize_graph_edit_distance(
                self.ref_topology,
                curr_topo,
                node_subst_cost=_node_subst_cost,
                node_del_cost=lambda a: 1.0,
                node_ins_cost=lambda a: 1.0,
                edge_subst_cost=lambda a, b: 0.5
                if a.get("pin_role") and b.get("pin_role") and a.get("pin_role") != b.get("pin_role")
                else 0.0,
                edge_del_cost=lambda a: 1.0,
                edge_ins_cost=lambda a: 1.0,
            ):
                best_ged = ged
                break

            similarity = max(0.0, 1.0 - best_ged / max_size)
            result["similarity"] = max(result.get("similarity", 0), similarity)
        except Exception:
            similarity = self._approximate_ged_similarity(curr_topo)
            result["similarity"] = max(result.get("similarity", 0), similarity)

    def _approximate_ged_similarity(self, curr_topo: nx.Graph) -> float:
        if self.ref_topology is None:
            return 0.0
        ref_types = Counter(
            d.get("ctype", "NET") for _, d in self.ref_topology.nodes(data=True) if d.get("kind") == "comp"
        )
        cur_types = Counter(
            d.get("ctype", "NET") for _, d in curr_topo.nodes(data=True) if d.get("kind") == "comp"
        )
        all_types = set(ref_types.keys()) | set(cur_types.keys())
        if not all_types:
            return 1.0
        dot = sum(ref_types.get(t, 0) * cur_types.get(t, 0) for t in all_types)
        norm_r = math.sqrt(sum(v**2 for v in ref_types.values())) or 1
        norm_c = math.sqrt(sum(v**2 for v in cur_types.values())) or 1
        type_sim = dot / (norm_r * norm_c)

        ref_deg = sorted([d for _, d in self.ref_topology.degree()], reverse=True)
        cur_deg = sorted([d for _, d in curr_topo.degree()], reverse=True)
        max_len = max(len(ref_deg), len(cur_deg), 1)
        ref_deg.extend([0] * (max_len - len(ref_deg)))
        cur_deg.extend([0] * (max_len - len(cur_deg)))
        deg_diff = sum(abs(a - b) for a, b in zip(ref_deg, cur_deg))
        deg_sum = sum(ref_deg) + sum(cur_deg) or 1
        deg_sim = 1.0 - deg_diff / deg_sum

        ref_e = self.ref_topology.number_of_edges() or 1
        cur_e = curr_topo.number_of_edges() or 1
        edge_sim = min(ref_e, cur_e) / max(ref_e, cur_e)

        return max(0.0, min(1.0, 0.5 * type_sim + 0.3 * deg_sim + 0.2 * edge_sim))

    def _heuristic_position_match(self, result: Dict, curr_analyzer: CircuitAnalyzer):
        matched = set()
        ref_instances = self.ref_component_instances
        curr_instances = curr_analyzer.component_instances
        for ref_c in ref_instances:
            best_idx, min_dist = None, 999
            ref_row = _first_pin_row(ref_c)
            if ref_row is None:
                continue
            for idx, curr_c in enumerate(curr_instances):
                if idx in matched or curr_c.component_type != ref_c.component_type:
                    continue
                curr_row = _first_pin_row(curr_c)
                if curr_row is None:
                    continue
                dist = abs(curr_row - ref_row)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            if best_idx is not None:
                matched.add(best_idx)

    # ---- 独立诊断 (无需参考电路) ----

    @staticmethod
    def diagnose_items(analyzer: CircuitAnalyzer) -> List[Dict]:
        """基于拓扑的独立电路诊断（结构化对象版）"""
        issues: List[Dict] = []
        g = analyzer.graph

        for comp in analyzer.component_instances:
            ctype = norm_component_type(comp.component_type)
            pin_nodes = [
                (
                    pin,
                    pin.electrical_node_id or analyzer.board_schema.resolve_hole_to_node(pin.hole_id),
                )
                for pin in comp.pins
            ]

            if ctype == "LED" and len(pin_nodes) >= 2:
                n1 = pin_nodes[0][1]
                n2 = pin_nodes[1][1]
                has_resistor = False
                # Union-Find 感知: 通过 Wire 连接的 Resistor 也算相邻
                led_nets = {analyzer._uf.find(n1), analyzer._uf.find(n2)}
                for other in analyzer.component_instances:
                    if norm_component_type(other.component_type) == "Resistor":
                        r_nets = set()
                        for other_pin in other.pins:
                            other_node = other_pin.electrical_node_id or analyzer.board_schema.resolve_hole_to_node(other_pin.hole_id)
                            r_nets.add(analyzer._uf.find(other_node))
                        if led_nets & r_nets:
                            has_resistor = True
                            break
                if not has_resistor:
                    issues.append(
                        _make_diagnostic(
                            category="component_errors",
                            error_code="LED_SERIES_RESISTOR_MISSING",
                            message=(
                                f"{comp.component_id}: LED所在网络中未检测到限流电阻, "
                                f"建议在{n1}或{n2}串联220Ω-1kΩ电阻"
                            ),
                            severity="warning",
                            component_id=comp.component_id,
                            expected="series_resistor_present",
                            actual="series_resistor_missing",
                            context={"component_type": ctype, "net_a": n1, "net_b": n2},
                        )
                    )

            if ctype in POLARIZED_TYPES and str(comp.polarity) == "unknown":
                issues.append(
                    _make_diagnostic(
                        category="polarity_errors",
                        error_code="POLARITY_UNKNOWN",
                        message=f"{comp.component_id}: {ctype}极性未确定, 请目视检查安装方向",
                        severity="warning",
                        component_id=comp.component_id,
                        expected="known_polarity",
                        actual="unknown",
                        context={"component_type": ctype},
                    )
                )

            if len(pin_nodes) >= 2:
                n1 = pin_nodes[0][1]
                n2 = pin_nodes[1][1]
                # Union-Find 感知: 检查 Wire 合并后的等电位短路
                same_net = (n1 == n2) or analyzer._uf.connected(n1, n2)
                if same_net and ctype not in ("Wire",):
                    issues.append(
                        _make_diagnostic(
                            category="node_errors",
                            error_code="COMPONENT_SHORTED_SAME_NET",
                            message=(
                                f"{comp.component_id}: {ctype}两引脚在同一导通组({n1}), "
                                f"元件被短路或未正确跨行插入"
                            ),
                            severity="error",
                            component_id=comp.component_id,
                            expected="different_conductive_groups",
                            actual=n1,
                            context={"component_type": ctype, "net_a": n1, "net_b": n2},
                        )
                    )

        for comp in analyzer.component_instances:
            ctype = norm_component_type(comp.component_type)
            if ctype == "Wire":
                continue
            for pin in comp.pins:
                node = pin.electrical_node_id or analyzer.board_schema.resolve_hole_to_node(pin.hole_id)
                if node in g and g.degree(node) == 1:
                    issues.append(
                        _make_diagnostic(
                            category="topology_errors",
                            error_code="FLOATING_PIN",
                            message=f"{comp.component_id}: 引脚{pin.pin_name}({pin.hole_id})仅连接到该元件自身, 可能为悬空引脚",
                            severity="warning",
                            component_id=comp.component_id,
                            actual=node,
                            context={"pin_name": pin.pin_name, "hole_id": pin.hole_id},
                        )
                    )
                    break

        if g.number_of_nodes() > 0:
            n_components = nx.number_connected_components(g)
            if n_components > 1:
                issues.append(
                    _make_diagnostic(
                        category="topology_errors",
                        error_code="MULTIPLE_DISCONNECTED_SUBGRAPHS",
                        message=f"电路图有 {n_components} 个独立子网络, 可能存在断路或缺少连线",
                        severity="warning",
                        expected=1,
                        actual=n_components,
                    )
                )

        return issues

    @staticmethod
    def diagnose(analyzer: CircuitAnalyzer) -> List[str]:
        """兼容层：返回结构化独立诊断的 message 列表。"""
        return [item["message"] for item in CircuitValidator.diagnose_items(analyzer)]


def _component_instance_from_dict(item: Dict) -> ComponentInstance:
    pins = []
    for pin in item.get("pins", []):
        pins.append(
            PinAssignment(
                pin_id=int(pin.get("pin_id", len(pins) + 1)),
                pin_name=str(pin.get("pin_name") or f"pin{len(pins) + 1}"),
                hole_id=str(pin.get("hole_id") or ""),
                electrical_node_id=pin.get("electrical_node_id"),
                electrical_net_id=pin.get("electrical_net_id"),
                confidence=float(pin.get("confidence", 0.0)),
                is_ambiguous=bool(pin.get("is_ambiguous", False)),
                metadata=dict(pin.get("metadata") or {}),
            )
        )
    return ComponentInstance(
        component_id=str(item.get("component_id") or "UNKNOWN"),
        component_type=str(item.get("component_type") or item.get("type") or "UNKNOWN"),
        package_type=str(item.get("package_type") or "legacy"),
        part_subtype=str(item.get("part_subtype") or ""),
        polarity=str(item.get("polarity") or "none"),
        orientation=float(item.get("orientation", 0.0)),
        symmetry_group=[list(group) for group in item.get("symmetry_group", [])],
        pins=pins,
        confidence=float(item.get("confidence", 1.0)),
        metadata=dict(item.get("metadata") or {}),
    )


def _needs_polarity_check(comp: ComponentInstance) -> bool:
    return str(comp.polarity or "none") in ("forward", "reverse", "unknown")


def _first_pin_row(comp: ComponentInstance) -> Optional[int]:
    if not comp.pins:
        return None
    board_schema = BoardSchema.default_breadboard()
    logic_loc = board_schema.hole_id_to_logic_loc(comp.pins[0].hole_id)
    if logic_loc is None:
        return None
    try:
        return int(logic_loc[0])
    except (ValueError, TypeError):
        return None


def _component_match_cost(ref_comp: ComponentInstance, curr_comp: ComponentInstance) -> float:
    if ref_comp.component_type != curr_comp.component_type:
        return 1000.0

    cost = 0.0
    if ref_comp.package_type and curr_comp.package_type and ref_comp.package_type != curr_comp.package_type:
        cost += 20.0
    cost += abs(len(ref_comp.pins) - len(curr_comp.pins)) * 10.0
    if ref_comp.component_id == curr_comp.component_id:
        cost -= 2.0

    ref_row = _first_pin_row(ref_comp)
    curr_row = _first_pin_row(curr_comp)
    if ref_row is not None and curr_row is not None:
        cost += abs(ref_row - curr_row)

    ref_holes = {pin.hole_id for pin in ref_comp.pins}
    curr_holes = {pin.hole_id for pin in curr_comp.pins}
    overlap = len(ref_holes & curr_holes)
    cost += max(0, len(ref_holes) - overlap) * 2.0
    return cost


def _compare_component_pins(ref_comp: ComponentInstance, curr_comp: ComponentInstance) -> Dict:
    pin_mismatches: List[str] = []
    hole_mismatches: List[str] = []
    node_mismatches: List[str] = []
    component_mismatches: List[str] = []
    hole_diagnostics: List[Dict] = []
    node_diagnostics: List[Dict] = []
    component_diagnostics: List[Dict] = []
    exact_pin_matches = 0

    ref_pin_map = {pin.pin_name: pin for pin in ref_comp.pins}
    curr_pin_map = {pin.pin_name: pin for pin in curr_comp.pins}
    handled_pin_names = set()

    for group in ref_comp.symmetry_group:
        if not group:
            continue
        if not all(name in ref_pin_map for name in group):
            continue
        curr_group_pins = [curr_pin_map[name] for name in group if name in curr_pin_map]
        if len(curr_group_pins) != len(group):
            message = f"{ref_comp.component_id}: 对称引脚组 {group} 在当前组件 {curr_comp.component_id} 中不完整"
            component_mismatches.append(message)
            component_diagnostics.append(
                _make_diagnostic(
                    category="component_errors",
                    error_code="COMPONENT_SYMMETRY_GROUP_INCOMPLETE",
                    message=message,
                    severity="error",
                    component_id=ref_comp.component_id,
                    current_component_id=curr_comp.component_id,
                    expected=group,
                    actual=sorted(curr_pin_map.keys()),
                )
            )
            handled_pin_names.update(group)
            continue

        ref_holes = sorted(ref_pin_map[name].hole_id for name in group)
        curr_holes = sorted(pin.hole_id for pin in curr_group_pins)
        ref_nodes = sorted((ref_pin_map[name].electrical_node_id or "") for name in group)
        curr_nodes = sorted((pin.electrical_node_id or "") for pin in curr_group_pins)
        if ref_nodes != curr_nodes:
            message = f"{ref_comp.component_id}: 对称引脚组 {group} 的导通节点不一致，期望 {ref_nodes}，当前 {curr_nodes}"
            node_mismatches.append(message)
            node_diagnostics.append(
                _make_diagnostic(
                    category="node_errors",
                    error_code="NODE_MISMATCH",
                    message=message,
                    severity="error",
                    component_id=ref_comp.component_id,
                    current_component_id=curr_comp.component_id,
                    expected=ref_nodes,
                    actual=curr_nodes,
                    context={"pin_group": list(group)},
                )
            )
        elif ref_holes != curr_holes:
            message = f"{ref_comp.component_id}: 对称引脚组 {group} 的孔位不同，期望 {ref_holes}，当前 {curr_holes}"
            hole_mismatches.append(message)
            hole_diagnostics.append(
                _make_diagnostic(
                    category="hole_errors",
                    error_code="HOLE_MISMATCH",
                    message=message,
                    severity="warning",
                    component_id=ref_comp.component_id,
                    current_component_id=curr_comp.component_id,
                    expected=ref_holes,
                    actual=curr_holes,
                    context={"pin_group": list(group)},
                )
            )
        else:
            exact_pin_matches += len(group)
        handled_pin_names.update(group)

    for pin_name, ref_pin in ref_pin_map.items():
        if pin_name in handled_pin_names:
            continue
        curr_pin = curr_pin_map.get(pin_name)
        if curr_pin is None:
            message = f"{ref_comp.component_id}: 缺少引脚 {pin_name}（当前组件 {curr_comp.component_id}）"
            component_mismatches.append(message)
            component_diagnostics.append(
                _make_diagnostic(
                    category="component_errors",
                    error_code="PIN_MISSING",
                    message=message,
                    severity="error",
                    component_id=ref_comp.component_id,
                    current_component_id=curr_comp.component_id,
                    pin_name=pin_name,
                )
            )
            continue
        ref_node = ref_pin.electrical_node_id or ""
        curr_node = curr_pin.electrical_node_id or ""
        if ref_node != curr_node:
            message = f"{ref_comp.component_id}.{pin_name}: 节点不一致，期望 {ref_node}，当前 {curr_node}"
            node_mismatches.append(message)
            node_diagnostics.append(
                _make_diagnostic(
                    category="node_errors",
                    error_code="NODE_MISMATCH",
                    message=message,
                    severity="error",
                    component_id=ref_comp.component_id,
                    current_component_id=curr_comp.component_id,
                    pin_name=pin_name,
                    expected=ref_node,
                    actual=curr_node,
                )
            )
            continue
        if ref_pin.hole_id != curr_pin.hole_id:
            message = f"{ref_comp.component_id}.{pin_name}: 孔位不同，期望 {ref_pin.hole_id}，当前 {curr_pin.hole_id}"
            hole_mismatches.append(message)
            hole_diagnostics.append(
                _make_diagnostic(
                    category="hole_errors",
                    error_code="HOLE_MISMATCH",
                    message=message,
                    severity="warning",
                    component_id=ref_comp.component_id,
                    current_component_id=curr_comp.component_id,
                    pin_name=pin_name,
                    expected=ref_pin.hole_id,
                    actual=curr_pin.hole_id,
                )
            )
            continue
        exact_pin_matches += 1

    for pin_name in curr_pin_map:
        if pin_name not in ref_pin_map:
            message = f"{curr_comp.component_id}: 存在参考电路未定义的额外引脚 {pin_name}"
            component_mismatches.append(message)
            component_diagnostics.append(
                _make_diagnostic(
                    category="component_errors",
                    error_code="PIN_EXTRA",
                    message=message,
                    severity="error",
                    component_id=curr_comp.component_id,
                    pin_name=pin_name,
                )
            )

    return {
        "pin_mismatches": component_mismatches + node_mismatches,
        "hole_mismatches": hole_mismatches,
        "node_mismatches": node_mismatches,
        "component_mismatches": component_mismatches,
        "hole_diagnostics": hole_diagnostics,
        "node_diagnostics": node_diagnostics,
        "component_diagnostics": component_diagnostics,
        "exact_pin_matches": exact_pin_matches,
    }


def _append_report(
    result: Dict,
    category: str,
    message: str,
    *,
    error_code: str,
    severity: str = "error",
    component_id: str | None = None,
    current_component_id: str | None = None,
    pin_name: str | None = None,
    expected=None,
    actual=None,
    context: Dict | None = None,
):
    report_key = {
        "topology_errors": "topology_errors",
        "node_errors": "node_errors",
        "hole_errors": "hole_errors",
        "polarity_errors": "polarity_errors",
        "component_errors": "component_errors",
    }[category]

    bucket = result.setdefault(report_key, [])
    if message not in bucket:
        bucket.append(message)

    diagnostic = _make_diagnostic(
        category=report_key,
        error_code=error_code,
        message=message,
        severity=severity,
        component_id=component_id,
        current_component_id=current_component_id,
        pin_name=pin_name,
        expected=expected,
        actual=actual,
        context=context,
    )

    report = result.setdefault("report", {})
    report.setdefault("version", "validator_report_v2")
    items = report.setdefault("items", [])
    if diagnostic not in items:
        items.append(diagnostic)
    report_bucket = report.setdefault(report_key, [])
    if diagnostic not in report_bucket:
        report_bucket.append(diagnostic)

    flat_items = result.setdefault("diagnostic_items", [])
    if diagnostic not in flat_items:
        flat_items.append(diagnostic)

    if message not in result.setdefault("errors", []):
        result["errors"].append(message)


def _append_diagnostic(result: Dict, diagnostic: Dict):
    report_key = diagnostic["category"]
    message = diagnostic["message"]

    bucket_key = {
        "topology_errors": "topology_errors",
        "node_errors": "node_mismatches",
        "hole_errors": "hole_mismatches",
        "polarity_errors": "polarity_errors",
        "component_errors": "component_mismatches",
    }[report_key]
    if message not in result.setdefault(bucket_key, []):
        result[bucket_key].append(message)

    report = result.setdefault("report", {})
    report.setdefault("version", "validator_report_v2")
    items = report.setdefault("items", [])
    if diagnostic not in items:
        items.append(diagnostic)
    bucket = report.setdefault(report_key, [])
    if diagnostic not in bucket:
        bucket.append(diagnostic)

    flat_items = result.setdefault("diagnostic_items", [])
    if diagnostic not in flat_items:
        flat_items.append(diagnostic)

    if message not in result.setdefault("errors", []):
        result["errors"].append(message)


def _make_diagnostic(
    *,
    category: str,
    error_code: str,
    message: str,
    severity: str,
    component_id: str | None = None,
    current_component_id: str | None = None,
    pin_name: str | None = None,
    expected=None,
    actual=None,
    context: Dict | None = None,
) -> Dict:
    """统一生成结构化诊断对象。

    这里是整个 validator_report_v2 的唯一出口, 所以建议动作、证据引用、
    以及未来的 agent 友好字段都优先在这里集中维护。
    """
    suggested_action = _suggested_action_for_diagnostic(
        error_code=error_code,
        category=category,
        component_id=component_id,
        pin_name=pin_name,
        expected=expected,
        actual=actual,
        context=context,
    )
    evidence_refs = _build_evidence_refs(
        category=category,
        error_code=error_code,
        component_id=component_id,
        current_component_id=current_component_id,
        pin_name=pin_name,
        expected=expected,
        actual=actual,
        context=context,
    )

    item = {
        "error_code": error_code,
        "category": category,
        "severity": severity,
        "message": message,
        "suggested_action": suggested_action,
        "evidence_refs": evidence_refs,
    }
    if component_id is not None:
        item["component_id"] = component_id
    if current_component_id is not None:
        item["current_component_id"] = current_component_id
    if pin_name is not None:
        item["pin_name"] = pin_name
    if expected is not None:
        item["expected"] = expected
    if actual is not None:
        item["actual"] = actual
    if context:
        item["context"] = context
    return item


def _finalize_report(result: Dict):
    report = result.setdefault("report", {})
    summary = {
        "topology_error_count": len(report.get("topology_errors", [])),
        "node_error_count": len(report.get("node_errors", [])),
        "hole_error_count": len(report.get("hole_errors", [])),
        "polarity_error_count": len(report.get("polarity_errors", [])),
        "component_error_count": len(report.get("component_errors", [])),
        "total_error_count": len(report.get("items", [])),
    }
    report["summary"] = summary


def _suggested_action_for_diagnostic(
    *,
    error_code: str,
    category: str,
    component_id: str | None,
    pin_name: str | None,
    expected,
    actual,
    context: Dict | None,
) -> str:
    """把 error_code 翻译成可直接展示给老师端或 agent 的一句修复建议。"""
    comp = component_id or "该元件"
    pin_label = f"{comp}.{pin_name}" if component_id and pin_name else (pin_name or comp)

    actions = {
        "REFERENCE_NOT_SET": "先设置或加载参考电路，再执行 compare 流程。",
        "COMPONENT_MISSING": "补齐缺失元件后重新运行识别与验证。",
        "COMPONENT_EXTRA": "检查是否误放多余元件，或是否存在检测误检。",
        "COMPONENT_INSTANCE_MISSING": f"检查 {comp} 的检测结果、组件类型和孔位映射是否正确。",
        "COMPONENT_SYMMETRY_GROUP_INCOMPLETE": f"检查 {comp} 的对称引脚是否都被识别并正确建模。",
        "PIN_MISSING": f"补齐 {pin_label} 的引脚识别或人工确认其孔位。",
        "PIN_EXTRA": f"检查 {comp} 是否误生成了额外引脚，必要时回看 pin schema。",
        "NODE_MISMATCH": f"将 {pin_label} 调整到正确导通节点，避免接到错误行或轨道。",
        "HOLE_MISMATCH": f"将 {pin_label} 插回参考孔位，或更新参考电路孔位定义。",
        "POLARITY_REVERSED": f"翻转 {comp} 的安装方向，使其极性与参考一致。",
        "POLARITY_UNKNOWN": f"确认 {comp} 的方向标记，并补充极性识别证据。",
        "TOPOLOGY_VALID_SUBSET": "补齐缺失连接或元件后再次验证整体拓扑。",
        "TOPOLOGY_MATCH_PIN_PLACEMENT_DIFFERS": "保持当前拓扑的同时，微调元件插孔位置到参考布局。",
        "TOPOLOGY_CHECK_FAILED": "检查当前 netlist_v2 和 topology_graph 是否完整，必要时回退到最小样例排查。",
        "FLOATING_PIN": f"检查 {comp} 是否有悬空引脚，并补上连线或调整插孔。",
        "MULTIPLE_DISCONNECTED_SUBGRAPHS": "检查是否存在断路、漏接导线或未接入电源轨的子网络。",
        "COMPONENT_SHORTED_SAME_NET": f"确认 {comp} 是否跨行插入，避免两脚落在同一导通组。",
        "LED_SERIES_RESISTOR_MISSING": f"为 {comp} 所在支路串联限流电阻，再重新验证。",
    }
    return actions.get(error_code, f"检查 {category} 对应证据，并根据参考电路修正 {comp}。")


def _build_evidence_refs(
    *,
    category: str,
    error_code: str,
    component_id: str | None,
    current_component_id: str | None,
    pin_name: str | None,
    expected,
    actual,
    context: Dict | None,
) -> List[Dict]:
    """把 compare / diagnose 的上下文压成轻量证据引用。

    当前先输出可序列化的索引对象, 后续可以继续升级成:
    - netlist pin ref
    - topology edge ref
    - risk rule ref
    - KB citation ref
    """
    refs: List[Dict] = []

    if component_id is not None:
        refs.append({"kind": "reference_component", "component_id": component_id})
    if current_component_id is not None and current_component_id != component_id:
        refs.append({"kind": "current_component", "component_id": current_component_id})
    if pin_name is not None:
        refs.append(
            {
                "kind": "pin",
                "component_id": current_component_id or component_id,
                "pin_name": pin_name,
            }
        )

    if error_code in ("HOLE_MISMATCH", "PIN_MISSING", "PIN_EXTRA"):
        if expected is not None:
            refs.append({"kind": "expected_hole", "value": expected})
        if actual is not None:
            refs.append({"kind": "actual_hole", "value": actual})

    if error_code in ("NODE_MISMATCH", "COMPONENT_SHORTED_SAME_NET"):
        if expected is not None:
            refs.append({"kind": "expected_node", "value": expected})
        if actual is not None:
            refs.append({"kind": "actual_node", "value": actual})

    if error_code in ("POLARITY_REVERSED", "POLARITY_UNKNOWN"):
        if expected is not None:
            refs.append({"kind": "expected_polarity", "value": expected})
        if actual is not None:
            refs.append({"kind": "actual_polarity", "value": actual})

    if error_code in ("COMPONENT_MISSING", "COMPONENT_EXTRA", "COMPONENT_INSTANCE_MISSING"):
        component_type = (context or {}).get("component_type")
        if component_type:
            refs.append({"kind": "component_type", "value": component_type})

    if error_code == "LED_SERIES_RESISTOR_MISSING":
        if context:
            if context.get("net_a") is not None:
                refs.append({"kind": "net", "value": context["net_a"]})
            if context.get("net_b") is not None:
                refs.append({"kind": "net", "value": context["net_b"]})

    if error_code == "TOPOLOGY_VALID_SUBSET":
        if expected is not None:
            refs.append({"kind": "expected_component_count", "value": expected})
        if actual is not None:
            refs.append({"kind": "matched_component_count", "value": actual})

    if context:
        if "pin_group" in context:
            refs.append({"kind": "pin_group", "value": context["pin_group"]})
        if "graph_node_id" in context:
            refs.append({"kind": "graph_node", "value": context["graph_node_id"]})

    refs.append({"kind": "diagnostic_code", "value": error_code})
    refs.append({"kind": "diagnostic_category", "value": category})
    return refs
