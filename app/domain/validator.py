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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from .circuit import (
    CircuitAnalyzer,
    CircuitComponent,
    Polarity,
    POLARIZED_TYPES,
    norm_component_type,
)

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
    """电路验证器 — 支持拓扑同构比较与位置启发式比较"""

    def __init__(self):
        self.ref_graph: Optional[nx.Graph] = None
        self.ref_components: List[CircuitComponent] = []
        self.ref_topology: Optional[nx.Graph] = None

    @property
    def has_reference(self) -> bool:
        return len(self.ref_components) > 0

    def set_reference(self, analyzer: CircuitAnalyzer):
        """将当前电路设为 Golden Reference"""
        self.ref_graph = analyzer.graph.copy()
        self.ref_components = [
            CircuitComponent(
                name=c.name,
                type=c.type,
                pin1_loc=c.pin1_loc,
                pin2_loc=c.pin2_loc,
                polarity=c.polarity,
                confidence=c.confidence,
                orientation_deg=c.orientation_deg,
            )
            for c in analyzer.components
        ]
        try:
            self.ref_topology = analyzer.build_topology_graph()
        except Exception:
            self.ref_topology = None

    def save_reference(self, file_path: str):
        if not self.ref_components:
            raise ValueError("No reference circuit set.")
        topo_payload = None
        if self.ref_topology is not None:
            topo_payload = json_graph.node_link_data(self.ref_topology)
        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "format": "labguardian_ref_v3",
            },
            "components": [
                {
                    "name": c.name,
                    "type": c.type,
                    "pin1_loc": list(c.pin1_loc) if c.pin1_loc else None,
                    "pin2_loc": list(c.pin2_loc) if c.pin2_loc else None,
                    "polarity": c.polarity.value,
                }
                for c in self.ref_components
            ],
            "topology": topo_payload,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_reference(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        comps = []
        for item in payload.get("components", []):
            pin1 = tuple(item["pin1_loc"]) if item.get("pin1_loc") else None
            pin2 = tuple(item["pin2_loc"]) if item.get("pin2_loc") else None
            if pin1 is None:
                continue
            pol_str = item.get("polarity", "none")
            try:
                polarity = Polarity(pol_str)
            except ValueError:
                polarity = Polarity.NONE
            comps.append(
                CircuitComponent(
                    name=item.get("name", "UNKNOWN"),
                    type=item.get("type", "UNKNOWN"),
                    pin1_loc=pin1,
                    pin2_loc=pin2,
                    polarity=polarity,
                )
            )
        self.ref_components = comps
        tmp = CircuitAnalyzer()
        for c in self.ref_components:
            tmp.add_component(c)
        self.ref_graph = tmp.graph.copy()
        topo_data = payload.get("topology")
        if topo_data:
            try:
                self.ref_topology = json_graph.node_link_graph(topo_data)
            except Exception:
                self.ref_topology = None
        else:
            try:
                self.ref_topology = tmp.build_topology_graph()
            except Exception:
                self.ref_topology = None

    def compare(self, curr_analyzer: CircuitAnalyzer) -> Dict:
        """分级诊断管线: L0→L1→L2→L2.5→L3"""
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
        }

        if not self.has_reference:
            result["errors"].append("No reference circuit set.")
            return result

        # L0: 元件数量统计
        ref_counts = Counter(c.type for c in self.ref_components)
        curr_counts = Counter(c.type for c in curr_analyzer.components)
        for t in sorted(set(ref_counts.keys()) | set(curr_counts.keys())):
            r_c, c_c = ref_counts[t], curr_counts[t]
            if c_c < r_c:
                result["errors"].append(f"Missing {r_c - c_c} x {t}")
                result["missing_components"].extend([t] * (r_c - c_c))
            elif c_c > r_c:
                result["errors"].append(f"Extra {c_c - r_c} x {t}")
                result["extra_components"].extend([t] * (c_c - r_c))

        # L1-L3
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
                        result["errors"] = ["Topology matches reference."]
                        result["is_match"] = True
                        result["similarity"] = 1.0
                        result["progress"] = 1.0
                        result["matched_components"] = [c.name for c in self.ref_components]
                        return result

                self._check_subgraph_match(result, curr_topo)
                self._check_polarity_errors(result, curr_topo)
                self._compute_ged_similarity(result, curr_topo)
        except Exception as e:
            result["errors"].append(f"Topology check failed: {e}")
            logger.exception("Topology check failed")

        if not result["is_match"]:
            self._heuristic_position_match(result, curr_analyzer)

        if not result["errors"]:
            result["errors"].append("Circuit matches Reference!")
            result["is_match"] = True

        return result

    # ---- VF2++ 回调 ----

    @staticmethod
    def _node_match_full(a: dict, b: dict) -> bool:
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
        if a.get("kind") != b.get("kind"):
            return False
        if a.get("kind") == "comp":
            return a.get("ctype") == b.get("ctype")
        return True

    @staticmethod
    def _edge_match(a: dict, b: dict) -> bool:
        ref_role = a.get("pin_role")
        cur_role = b.get("pin_role")
        if ref_role is None or cur_role is None:
            return True
        return ref_role == cur_role

    # ---- L2: 子图同构 ----

    def _check_subgraph_match(self, result: Dict, curr_topo: nx.Graph):
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
                result["errors"].append(
                    f"Circuit is a valid subset (progress: {progress:.0%}, "
                    f"{len(matched_ref_comps)}/{total_ref_comps} matched)"
                )
            result["matched_components"] = [
                self.ref_topology.nodes[n].get("ctype", "?") for n in matched_ref_comps
            ]

    # ---- L2.5: 极性诊断 ----

    def _check_polarity_errors(self, result: Dict, curr_topo: nx.Graph):
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
                        result["polarity_errors"].append(f"{ctype} ({curr_node}) 极性反接")
                elif ref_pol in ("forward", "reverse") and cur_pol == "unknown":
                    result["polarity_errors"].append(f"{ctype} ({curr_node}) 极性无法判断")

    # ---- L3: GED 相似度 ----

    def _compute_ged_similarity(self, result: Dict, curr_topo: nx.Graph):
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
        for ref_c in self.ref_components:
            best_idx, min_dist = None, 999
            try:
                ref_row = int(ref_c.pin1_loc[0])
            except (ValueError, TypeError):
                continue
            for idx, curr_c in enumerate(curr_analyzer.components):
                if idx in matched or curr_c.type != ref_c.type:
                    continue
                try:
                    dist = abs(int(curr_c.pin1_loc[0]) - ref_row)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                except (ValueError, TypeError):
                    continue
            if best_idx is not None:
                matched.add(best_idx)

    # ---- 独立诊断 (无需参考电路) ----

    @staticmethod
    def diagnose(analyzer: CircuitAnalyzer) -> List[str]:
        """基于拓扑的独立电路诊断"""
        issues = []
        g = analyzer.graph

        for comp in analyzer.components:
            ctype = norm_component_type(comp.type)

            if ctype == "LED" and comp.pin2_loc:
                n1 = analyzer._get_node_name(comp.pin1_loc)
                n2 = analyzer._get_node_name(comp.pin2_loc)
                has_resistor = False
                for node in (n1, n2):
                    if node not in g:
                        continue
                    for neighbor in g.neighbors(node):
                        edge_data = g.get_edge_data(node, neighbor)
                        if edge_data and norm_component_type(edge_data.get("type", "")) == "Resistor":
                            has_resistor = True
                            break
                    if has_resistor:
                        break
                if not has_resistor:
                    issues.append(
                        f"{comp.name}: LED所在网络中未检测到限流电阻, "
                        f"建议在{n1}或{n2}串联220Ω-1kΩ电阻"
                    )

            if ctype in POLARIZED_TYPES and comp.polarity == Polarity.UNKNOWN:
                issues.append(f"{comp.name}: {ctype}极性未确定, 请目视检查安装方向")

            if comp.pin2_loc:
                n1 = analyzer._get_node_name(comp.pin1_loc)
                n2 = analyzer._get_node_name(comp.pin2_loc)
                if n1 == n2 and ctype not in ("Wire",):
                    issues.append(
                        f"{comp.name}: {ctype}两引脚在同一导通组({n1}), "
                        f"元件被短路或未正确跨行插入"
                    )

        for comp in analyzer.components:
            ctype = norm_component_type(comp.type)
            if ctype == "Wire":
                continue
            nodes_of_comp = set()
            nodes_of_comp.add(analyzer._get_node_name(comp.pin1_loc))
            if comp.pin2_loc:
                nodes_of_comp.add(analyzer._get_node_name(comp.pin2_loc))
            for node in nodes_of_comp:
                if node in g and g.degree(node) == 1:
                    issues.append(
                        f"{comp.name}: 引脚{node}仅连接到该元件自身, 可能为悬空引脚"
                    )
                    break

        if g.number_of_nodes() > 0:
            n_components = nx.number_connected_components(g)
            if n_components > 1:
                issues.append(f"电路图有 {n_components} 个独立子网络, 可能存在断路或缺少连线")

        return issues
