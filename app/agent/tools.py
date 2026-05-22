from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.concepts import CONCEPT_LIBRARY, lookup_concept
from app.agent.contracts import RuntimeEvidence
from app.core.config import settings
from app.domain.board_schema import BoardSchema
from app.services.circuit_kb_service import CircuitKbService, looks_like_circuit_query
from app.services.datasheet_kb_service import DatasheetKbService
from app.services.embedding_backend import create_embedding_backend
from app.services.kb_service import KbService
from app.services.teaching_kb_service import TeachingKbService

_DATASHEET_KB_SINGLETON: DatasheetKbService | None = None


def _get_datasheet_kb() -> DatasheetKbService:
    global _DATASHEET_KB_SINGLETON
    if _DATASHEET_KB_SINGLETON is None:
        backend = create_embedding_backend(
            kind=getattr(settings, "DATASHEET_EMBEDDING_BACKEND", "null"),
            model_dir=getattr(settings, "DATASHEET_EMBEDDING_MODEL_DIR", None),
            device=getattr(settings, "DATASHEET_EMBEDDING_DEVICE", "CPU"),
            max_length=getattr(settings, "DATASHEET_EMBEDDING_MAX_LEN", 256),
        )
        _DATASHEET_KB_SINGLETON = DatasheetKbService(
            embedding=backend,
            embeddings_dir=getattr(settings, "DATASHEET_EMBEDDINGS_DIR", None),
            fusion_weight=float(
                getattr(settings, "DATASHEET_EMBEDDING_FUSION_WEIGHT", 0.55)
            ),
        )
    return _DATASHEET_KB_SINGLETON


class ToolResult(BaseModel):
    tool_name: str
    status: str = "ok"
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class NetlistTraceInput(BaseModel):
    component_id: str = ""
    pin_name: str = ""
    node_id: str = ""


class BoardSchemaLookupInput(BaseModel):
    hole_id: str = ""
    node_id: str = ""


class FaultCaseLookupInput(BaseModel):
    query: str = ""
    error_tags: list[str] = Field(default_factory=list)
    scene_id: str = "exp_first_order_rc"
    top_k: int = Field(default=3, ge=1, le=10)


class CircuitLookupInput(BaseModel):
    query: str = ""
    circuit_id: str = ""
    top_k: int = Field(default=3, ge=1, le=5)


class SafetyRuleLookupInput(BaseModel):
    risk_level: str = "unknown"
    error_family: str = "unknown"


class DatasheetLookupInput(BaseModel):
    component_type: str = ""
    component_id: str = ""
    part_number: str = ""
    package_type: str = ""
    query: str = ""
    error_family: str = "unknown"


LOCAL_DATASHEET_FALLBACKS: dict[str, dict[str, Any]] = {
    "led": {
        "component_type": "LED",
        "package": "two_pin_polarized",
        "pin_rules": ["anode 接高电位侧", "cathode 接低电位侧"],
        "safety_rules": ["必须串联限流电阻", "调试时优先使用低压限流电源"],
        "notes": "LED 为极性器件，反接通常不亮，缺少限流可能导致器件损坏。",
    },
    "diode": {
        "component_type": "Diode",
        "package": "two_pin_polarized",
        "pin_rules": ["anode 到 cathode 为正向导通方向", "cathode 通常由色环或标记识别"],
        "safety_rules": ["确认方向后再通电", "避免直接跨接电源轨"],
        "notes": "二极管方向错误会改变支路导通状态。",
    },
    "capacitorelectrolytic": {
        "component_type": "CapacitorElectrolytic",
        "package": "two_pin_polarized",
        "pin_rules": ["positive 接较高电位", "negative 接较低电位或 GND"],
        "safety_rules": ["通电前确认极性", "反接电解电容存在发热或损坏风险"],
        "notes": "电解电容是极性器件，长脚通常为正极，外壳负极侧常有标记。",
    },
    "resistor": {
        "component_type": "Resistor",
        "package": "two_pin_non_polarized",
        "pin_rules": ["两个引脚无极性", "应跨接到两个不同导通节点"],
        "safety_rules": ["避免两脚落在同一导通组造成元件被短接"],
        "notes": "电阻常用于限流、分压和反馈网络。",
    },
    "transistor": {
        "component_type": "Transistor",
        "package": "three_pin_polarized",
        "pin_rules": ["核对 base / collector / emitter 引脚顺序", "不同封装引脚序可能不同"],
        "safety_rules": ["先查封装方向，再接入电路", "避免把电源直接接到错误引脚"],
        "notes": "三极管引脚顺序强依赖具体型号和封装。",
    },
}


def netlist_trace_tool(
    evidence: RuntimeEvidence,
    args: NetlistTraceInput,
) -> ToolResult:
    """Trace component/pin/node facts inside runtime netlist_v2."""

    netlist = evidence.netlist_v2 or {}
    components = netlist.get("components", [])
    matched_components = []
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            if args.component_id and component.get("component_id") != args.component_id:
                continue
            matched_components.append(component)

    nets = netlist.get("nets", [])
    matched_nets = []
    if isinstance(nets, list):
        for net in nets:
            if not isinstance(net, dict):
                continue
            haystack = str(net)
            if args.node_id and args.node_id in haystack:
                matched_nets.append(net)
            elif args.component_id and args.component_id in haystack:
                matched_nets.append(net)

    summary = "未在 netlist_v2 中找到匹配项。"
    if matched_components or matched_nets:
        summary = (
            f"匹配 components={len(matched_components)}, "
            f"nets={len(matched_nets)}。"
        )
    return ToolResult(
        tool_name="netlist_trace_tool",
        summary=summary,
        payload={
            "components": matched_components[:5],
            "nets": matched_nets[:5],
        },
    )


def board_schema_lookup_tool(
    args: BoardSchemaLookupInput,
    *,
    board_schema: BoardSchema | None = None,
) -> ToolResult:
    schema = board_schema or BoardSchema.default_breadboard()
    payload: dict[str, Any] = {"schema_id": schema.schema_id}
    summaries: list[str] = []

    if args.hole_id:
        spec = schema.hole_to_spec(args.hole_id)
        payload["hole"] = {
            "hole_id": spec.hole_id,
            "electrical_node_id": spec.electrical_node_id,
            "group_type": spec.group_type,
            "row": spec.row,
            "col": spec.col,
        }
        summaries.append(f"{spec.hole_id}->{spec.electrical_node_id}")

    if args.node_id:
        matched = [
            {
                "hole_id": spec.hole_id,
                "row": spec.row,
                "col": spec.col,
                "group_type": spec.group_type,
            }
            for spec in schema.holes.values()
            if spec.electrical_node_id == args.node_id
        ]
        payload["node_holes"] = matched[:20]
        summaries.append(f"{args.node_id} contains {len(matched)} holes")

    return ToolResult(
        tool_name="board_schema_lookup_tool",
        summary="；".join(summaries) or "未提供 hole_id 或 node_id。",
        payload=payload,
    )


def fault_case_lookup_tool(
    args: FaultCaseLookupInput,
    *,
    teaching_kb_service: TeachingKbService | None = None,
) -> ToolResult:
    service = teaching_kb_service or TeachingKbService()
    cases = service.search_fault_cases(
        query=args.query,
        scene_id=args.scene_id,
        error_tags=args.error_tags,
        top_k=args.top_k,
    )
    return ToolResult(
        tool_name="fault_case_lookup_tool",
        summary=f"命中 fault_cases={len(cases)}。",
        payload={
            "fault_cases": [
                {
                    "knowledge_id": case.get("knowledge_id", ""),
                    "title": case.get("title", ""),
                    "error_tags": case.get("error_tags", []),
                    "related_error_codes": case.get("related_error_codes", []),
                    "fix_steps": case.get("fix_steps", [])[:4],
                }
                for case in cases
            ]
        },
    )


def datasheet_lookup_tool(args: DatasheetLookupInput) -> ToolResult:
    key = _datasheet_key(args.component_type or args.component_id or args.query)
    query = " ".join(
        part
        for part in [args.part_number, args.component_id, args.component_type, args.package_type, args.query]
        if part
    ).strip()
    if not query:
        query = args.query or args.component_id or args.component_type
    query_intent = "general"

    # Phase 1: try the offline, structured local datasheet KB first. It returns
    # uniform RetrievedChunk dicts with `chunk_id` + `modality` (the contract the
    # verifier enforces). Falls through to legacy Chroma/PDF retrieval and then
    # to rule-based fallback to preserve existing behavior.
    v2_hits: list[dict[str, Any]] = []
    try:
        retrieved = _get_datasheet_kb().search(
            query=query,
            part_numbers=[args.part_number] if args.part_number else None,
            top_k=4,
        )
        for chunk in retrieved:
            query_intent = chunk.query_intent or query_intent
            v2_hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "modality": chunk.modality,
                    "title": chunk.title,
                    "snippet": chunk.snippet,
                    "score": chunk.score,
                    "confidence": chunk.confidence,
                    "matched_features": chunk.matched_features,
                    "debug": chunk.debug,
                    "document_id": chunk.document_id,
                    "page": chunk.page,
                    "asset_path": chunk.asset_path,
                    "table_html": chunk.table_html,
                    "source_ref": chunk.source_ref,
                    "source_id": chunk.chunk_id,
                    "filename": (chunk.source_ref or {}).get("source_path") or chunk.document_id,
                    "query_intent": chunk.query_intent,
                }
            )
    except Exception:
        v2_hits = []

    if v2_hits:
        rules = [
            f"参考资料：{item.get('title') or item.get('document_id') or 'datasheet'}：{str(item.get('snippet') or '').strip()}"
            for item in v2_hits[:3]
            if item.get("snippet")
        ]
        return ToolResult(
            tool_name="datasheet_lookup_tool",
            summary=(
                f"datasheet 本地检索命中 {len(v2_hits)} 段"
                f"（{v2_hits[0].get('document_id') or 'local'}）。"
            ),
            payload={
                "provider": "local_datasheet_v2",
                "component_id": args.component_id,
                "component_type": args.component_type,
                "part_number": args.part_number,
                "package_type": args.package_type,
                "query": query,
                "query_intent": query_intent,
                "error_family": args.error_family,
                "miss_reason": "",
                "hits": v2_hits,
                "rules": rules[:4],
            },
        )

    hits: list[dict[str, Any]] = []
    kb_query_allowed = True
    if key in LOCAL_DATASHEET_FALLBACKS and not args.part_number:
        kb_query_allowed = bool(KbService()._chip_hints_from_query(query))
    if kb_query_allowed:
        try:
            kb = KbService()
            raw_hits = kb.retrieve(query=query, top_k=min(4, max(1, int(getattr(settings, "KB_DEFAULT_TOP_K", 6)))))
            for hit, _ in raw_hits[:4]:
                meta = hit.get("metadata", {}) or {}
                doc_id = meta.get("doc_id", "")
                chunk_index = meta.get("chunk_index", "")
                chunk_id = f"{doc_id}:{chunk_index}" if doc_id else ""
                hits.append(
                    {
                        "chunk_id": chunk_id,
                        "modality": "text",
                        "document_id": doc_id,
                        "title": hit.get("title") or "",
                        "snippet": hit.get("snippet") or "",
                        "filename": meta.get("filename") or meta.get("source") or "",
                        "page": meta.get("page"),
                        "score": hit.get("score", 0.0),
                        "confidence": 0.0,
                        "matched_features": [],
                        "debug": {},
                        "query_intent": query_intent,
                        # `source_id` retained as alias for downstream citation
                        # builders that haven't migrated to `chunk_id` yet.
                        "source_id": chunk_id,
                    }
                )
        except Exception:
            hits = []

    if hits:
        rules = [
            f"参考资料：{item.get('title') or item.get('filename') or 'datasheet'}：{str(item.get('snippet') or '').strip()}"
            for item in hits[:3]
            if item.get("snippet")
        ]
        return ToolResult(
            tool_name="datasheet_lookup_tool",
            summary=f"datasheet 检索命中 {len(hits)} 段（{hits[0].get('filename') or 'pdf'}）。",
            payload={
                "provider": "kb_retrieval",
                "component_id": args.component_id,
                "component_type": args.component_type,
                "part_number": args.part_number,
                "package_type": args.package_type,
                "query": query,
                "query_intent": query_intent,
                "error_family": args.error_family,
                "miss_reason": "",
                "hits": hits,
                "rules": rules[:4],
            },
        )

    fallback = LOCAL_DATASHEET_FALLBACKS.get(key)
    matched_key = key if key in LOCAL_DATASHEET_FALLBACKS else ""
    if fallback is None:
        fallback = {
            "component_type": args.component_type or "unknown",
            "package": "local_fallback_unknown",
            "pin_rules": ["本地 datasheet 未命中该器件；请以实物丝印和课程参考电路为准。"],
            "safety_rules": ["通电前先确认器件方向、限流条件和电源轨连接。"],
            "notes": "未命中本地 PDF；返回保守规则。",
        }

    rule_namespace = matched_key or "unknown"
    structured_rules: list[dict[str, Any]] = []
    flat_rules: list[str] = []
    for idx, rule in enumerate(fallback.get("pin_rules", [])):
        rid = f"fallback.{rule_namespace}.pin.{idx + 1}"
        structured_rules.append({"rule_id": rid, "category": "pin", "text": rule})
        flat_rules.append(rule)
    for idx, rule in enumerate(fallback.get("safety_rules", [])):
        rid = f"fallback.{rule_namespace}.safety.{idx + 1}"
        structured_rules.append({"rule_id": rid, "category": "safety", "text": rule})
        flat_rules.append(rule)

    return ToolResult(
        tool_name="datasheet_lookup_tool",
        summary=f"datasheet 未命中 PDF，回退到本地规则：{fallback['component_type']} / {fallback['package']}。",
        payload={
            "provider": "local_fallback",
            "component_id": args.component_id,
            "component_type": fallback["component_type"],
            "part_number": args.part_number,
            "package_type": args.package_type,
            "package": fallback["package"],
            "pin_rules": fallback["pin_rules"],
            "safety_rules": fallback["safety_rules"],
            "notes": fallback["notes"],
            "matched_key": matched_key,
            "query": query,
            "query_intent": query_intent,
            "error_family": args.error_family,
            "miss_reason": "no_structured_datasheet_hit",
            "rules": flat_rules[:6],
            "structured_rules": structured_rules[:6],
            "hits": [],
        },
    )


class TeachingConceptLookupInput(BaseModel):
    query: str = ""
    concept_id: str = ""
    error_family: str = "unknown"


def teaching_concept_lookup_tool(args: TeachingConceptLookupInput) -> ToolResult:
    """Deterministic local lookup over CONCEPT_LIBRARY.

    Resolution order: explicit concept_id → keyword scan over query. Misses
    return status="not_found" so callers can route to a generic answer rather
    than fabricating one.
    """

    pack = None
    if args.concept_id:
        pack = CONCEPT_LIBRARY.get(args.concept_id)
    if pack is None and args.query:
        pack = lookup_concept(args.query)

    if pack is None:
        return ToolResult(
            tool_name="teaching_concept_lookup_tool",
            status="not_found",
            summary="未在本地概念库中找到匹配条目",
            payload={
                "provider": "local_concept_library",
                "query": args.query,
                "concept_id": args.concept_id,
                "available_concepts": list(CONCEPT_LIBRARY.keys()),
            },
        )

    return ToolResult(
        tool_name="teaching_concept_lookup_tool",
        status="ok",
        summary=f"{pack.concept_id}: {pack.title}",
        payload={
            "provider": "local_concept_library",
            "concept": pack.model_dump(),
            "query": args.query,
        },
    )


def circuit_lookup_tool(
    args: CircuitLookupInput,
    *,
    circuit_kb_service: CircuitKbService | None = None,
) -> ToolResult:
    """Search the local circuit knowledge base for schematic-level circuit info.

    Gating:
    * If the query doesn't look circuit-related at all, return empty immediately.
    * If a specific circuit_id is requested, return that circuit directly.
    * Otherwise keyword-search and return matches above the score threshold.
    """
    service = circuit_kb_service or CircuitKbService()

    # Gate 1: explicit circuit_id lookup
    if args.circuit_id:
        circuit = service.get_circuit(args.circuit_id)
        if circuit is None:
            return ToolResult(
                tool_name="circuit_lookup_tool",
                status="not_found",
                summary=f"未找到电路 {args.circuit_id}",
                payload={"circuit_id": args.circuit_id, "circuits": [], "hits": 0},
            )
        return ToolResult(
            tool_name="circuit_lookup_tool",
            status="ok",
            summary=f"命中电路：{circuit.get('name', args.circuit_id)}",
            payload={"circuit_id": args.circuit_id, "circuits": [circuit], "hits": 1},
        )

    # Search first.  The service expands common circuit paraphrases before
    # scoring, so we should not block it with a brittle keyword gate.
    query = (args.query or "").strip()
    hits = service.search(query=query, top_k=args.top_k)

    if not hits:
        if not looks_like_circuit_query(query):
            return ToolResult(
                tool_name="circuit_lookup_tool",
                status="not_relevant",
                summary="查询与电路知识库不相关，跳过检索",
                payload={"query": query, "circuits": [], "hits": 0},
            )
        return ToolResult(
            tool_name="circuit_lookup_tool",
            status="not_found",
            summary="电路知识库未命中相关条目",
            payload={"query": query, "circuits": [], "hits": 0},
        )

    circuits_payload = [
        {
            "circuit_id": item["circuit"].get("circuit_id", ""),
            "name": item["circuit"].get("name", ""),
            "category": item["circuit"].get("category", ""),
            "summary": item["circuit"].get("summary", ""),
            "components": item["circuit"].get("components", []),
            "connections": item["circuit"].get("connections", []),
            "analysis": item["circuit"].get("analysis", {}),
            "common_faults": item["circuit"].get("common_faults", []),
            "teaching_points": item["circuit"].get("teaching_points", []),
            "aliases": item["circuit"].get("aliases", []),
            "retrieval_queries": item["circuit"].get("retrieval_queries", []),
            "image": item["circuit"].get("image", ""),
            "image_annotations": item["circuit"].get("image_annotations", {}),
            "score": item["score"],
            "matched_features": item.get("matched_features", []),
        }
        for item in hits
    ]

    top = circuits_payload[0]
    return ToolResult(
        tool_name="circuit_lookup_tool",
        status="ok",
        summary=f"电路知识库命中 {len(hits)} 个电路（top: {top['name']}, score={top['score']}）",
        payload={
            "query": query,
            "circuits": circuits_payload,
            "hits": len(hits),
        },
    )


def safety_rule_lookup_tool(args: SafetyRuleLookupInput) -> ToolResult:
    rules: list[str] = []
    if args.risk_level == "danger" or args.error_family == "short_circuit":
        rules.extend(
            [
                "先断开电源，再移动导线或元件。",
                "检查电源轨 VCC/GND 是否被同一元件或导线直接连通。",
                "复查限流元件，避免 LED 或电源输出直接短路。",
            ]
        )
    else:
        rules.append("保持低压限流条件下逐项复查连接。")

    return ToolResult(
        tool_name="safety_rule_lookup_tool",
        summary="；".join(rules),
        payload={"rules": rules},
    )


def _datasheet_key(value: str) -> str:
    normalized = str(value or "").replace("_", "").replace("-", "").replace(" ", "").lower()
    if "electrolytic" in normalized or normalized.startswith("ce"):
        return "capacitorelectrolytic"
    if "led" in normalized:
        return "led"
    if "diode" in normalized:
        return "diode"
    if "resistor" in normalized or normalized.startswith("r"):
        return "resistor"
    if "transistor" in normalized or normalized.startswith("q"):
        return "transistor"
    return normalized
