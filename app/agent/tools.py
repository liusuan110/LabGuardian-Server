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


def _reset_datasheet_kb_singleton() -> None:
    """WP-2.1 (2026-05-24): test-only helper.

    Tests that mutate ``settings.DATASHEET_EMBEDDING_*`` to load a different
    backend (e.g. ``test_wp2_distill_entrypoint.py`` switches to OpenVINO)
    must reset this singleton in teardown — otherwise subsequent tests
    that revert to the null backend still see the OV-bound singleton
    (test-order-dependent failures).

    Production code MUST NOT call this; the singleton lifetime is process
    lifetime by design.
    """
    global _DATASHEET_KB_SINGLETON
    _DATASHEET_KB_SINGLETON = None


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
    # WP-1 v4 (2026-05-24): ``error_codes`` is the canonical validator↔KB
    # bridge. fault_case JSONs declare ``related_error_codes`` (e.g.
    # FLOATING_PIN, COMPONENT_MISSING). Validator emits the same codes.
    # Tag matching is supplementary because the renamed scene-agnostic
    # tags (``missing_required_component`` etc.) don't appear in the KB
    # vocabulary, while ``related_error_codes`` always does.
    error_codes: list[str] = Field(default_factory=list)
    # WP-1 (2026-05-24): default removed. Empty scene_id means "no
    # topology context known" → the tool skips retrieval rather than
    # silently defaulting to RC. Callers should pass
    # ``state.runtime_evidence.current_scene_id`` (which the scene
    # resolver populates from station / comparison_report).
    scene_id: str = ""
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
    # WP-3 v3 (2026-05-24): when ``scene_id`` is set, the underlying
    # ``DatasheetKbService.search`` ALWAYS hard-filters candidate
    # documents to the scene's whitelist (see
    # ``app.services.scene_resolver.allowed_datasheets_for_scene``).
    # This is the production retrieval contract — train ≡ deploy. The
    # earlier v2 design that gated this on ``DISTILL_MODE`` produced a
    # train-test distribution shift: student trained on UA741+passive
    # only, but at deploy saw BJT/NE555 chunks too. Restoring symmetric
    # behavior makes the shift impossible.
    #
    # Admin/debug paths that genuinely need cross-chip search should
    # call ``DatasheetKbService.search()`` directly without scene_id
    # (e.g. the ``/api/v1/kb/*`` admin endpoints).
    scene_id: str = ""


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
    # WP-1 (2026-05-24): no scene context → skip retrieval. We MUST NOT
    # silently fall back to the RC scene (which is the bug this WP fixes).
    # See ``docs/retrieval-contract.md`` and ``app/services/scene_resolver.py``.
    scene_id = (args.scene_id or "").strip()
    if not scene_id:
        return ToolResult(
            tool_name="fault_case_lookup_tool",
            status="skipped",
            summary="无场景上下文（current_scene_id 为空），跳过 fault_case 检索。",
            payload={
                "fault_cases": [],
                "skip_reason": "no_scene_context",
            },
        )

    service = teaching_kb_service or TeachingKbService()
    # WP-1 v4: pass error_codes through — this is the canonical recall
    # signal (related_error_codes in fault_case JSONs). Without it the
    # agent path could only match on the renamed scene-agnostic tags,
    # which don't appear in the KB → fault_case_pack stays empty even
    # when validator clearly identifies the fault family.
    cases = service.search_fault_cases(
        query=args.query,
        scene_id=scene_id,
        error_tags=args.error_tags,
        error_codes=args.error_codes,
        top_k=args.top_k,
    )
    return ToolResult(
        tool_name="fault_case_lookup_tool",
        summary=f"命中 fault_cases={len(cases)}（scene_id={scene_id}）。",
        payload={
            "scene_id": scene_id,
            "fault_cases": [
                {
                    "knowledge_id": case.get("knowledge_id", ""),
                    "title": case.get("title", ""),
                    "error_tags": case.get("error_tags", []),
                    "related_error_codes": case.get("related_error_codes", []),
                    "fix_steps": case.get("fix_steps", [])[:4],
                }
                for case in cases
            ],
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
    #
    # WP-3 v3 (2026-05-24): scene whitelist is now a PRODUCTION contract.
    # Whenever scene_id is set, restrict the candidate set so train ≡ deploy.
    # Empty scene_id keeps the full-corpus search for concept_tutor questions
    # without topology context and for admin tools that bypass the agent graph.
    #
    # WP-3 v4 (2026-05-24): in DISTILL_MODE, hard-fail on empty / unknown
    # scene_id. The distillation entrypoint MUST validate scene_id upfront;
    # this is defense-in-depth so a malformed distill sample can never
    # silently retrieve cross-chip / out-of-scene evidence.
    distill_on = bool(getattr(settings, "DISTILL_MODE", False))
    scene_id_clean = (args.scene_id or "").strip()
    allowed_docs: frozenset[str] | None = None
    if scene_id_clean:
        from app.services.scene_resolver import allowed_datasheets_for_scene
        allowed_docs = allowed_datasheets_for_scene(scene_id_clean)
        if allowed_docs is None and distill_on:
            # WP-3 v4 (P1): scene_id is non-empty but not one of the 6 demos
            # → distill sample is malformed; refuse to synthesize evidence.
            return ToolResult(
                tool_name="datasheet_lookup_tool",
                status="skipped",
                summary=(
                    f"DISTILL_MODE 下 scene_id={scene_id_clean!r} 不在 6 个 demo "
                    f"场景中；拒绝从全 corpus 取证。"
                ),
                payload={
                    "provider": "distill_fail_closed",
                    "component_id": args.component_id,
                    "component_type": args.component_type,
                    "part_number": args.part_number,
                    "package_type": args.package_type,
                    "query": query,
                    "query_intent": query_intent,
                    "error_family": args.error_family,
                    "miss_reason": "distill_invalid_scene_id",
                    "hits": [],
                    "rules": [],
                },
            )
    elif distill_on:
        # WP-3 v4 (P1): empty scene_id in DISTILL_MODE → refuse. The
        # production agent graph always passes evidence.current_scene_id;
        # the distill entrypoint must guarantee it's non-empty.
        return ToolResult(
            tool_name="datasheet_lookup_tool",
            status="skipped",
            summary=(
                "DISTILL_MODE 下 scene_id 为空；拒绝在无场景上下文中取证。"
            ),
            payload={
                "provider": "distill_fail_closed",
                "component_id": args.component_id,
                "component_type": args.component_type,
                "part_number": args.part_number,
                "package_type": args.package_type,
                "query": query,
                "query_intent": query_intent,
                "error_family": args.error_family,
                "miss_reason": "distill_no_scene_id",
                "hits": [],
                "rules": [],
            },
        )

    v2_hits: list[dict[str, Any]] = []
    try:
        retrieved = _get_datasheet_kb().search(
            query=query,
            part_numbers=[args.part_number] if args.part_number else None,
            allowed_document_ids=allowed_docs,
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

    # WP-3 (2026-05-24): in distillation mode, when local v2 misses we
    # MUST NOT fall back to LOCAL_DATASHEET_FALLBACKS — those hand-coded
    # rules are useful in dev but would inject synthetic "rule_id=fallback.*"
    # evidence into shrinkage data that the on-device runtime (with the
    # same DATASHEET_EMBEDDING_BACKEND=openvino setup) would never actually
    # produce. Return an explicit skipped result so the agent answers
    # without datasheet evidence in this case.
    #
    # WP-3 v4 (2026-05-24): widened to ALSO skip on miss when scene_id is
    # set (production main path), regardless of DISTILL_MODE. Symmetric
    # train ≡ deploy semantics for the miss path: a scene-anchored turn
    # that misses v2 means evidence is absent in BOTH modes, never replaced
    # with synthetic fallback rules. When scene_id is empty (admin / no-topo
    # concept_tutor), the legacy fallback still fires for usability.
    if not v2_hits and (distill_on or scene_id_clean):
        reason = (
            "datasheet_v2_miss_distill_fail_closed"
            if distill_on
            else "datasheet_v2_miss_scene_anchored_no_fallback"
        )
        summary = (
            "datasheet 本地 v2 未命中；DISTILL_MODE 下不回落保守规则集，"
            "返回 skipped 以保持训练↔部署一致。"
            if distill_on
            else (
                f"datasheet 本地 v2 未命中（scene_id={scene_id_clean}）；"
                "场景锚定路径不回落保守规则集，保 train ≡ deploy。"
            )
        )
        return ToolResult(
            tool_name="datasheet_lookup_tool",
            status="skipped",
            summary=summary,
            payload={
                "provider": "distill_fail_closed" if distill_on else "scene_anchored_no_fallback",
                "component_id": args.component_id,
                "component_type": args.component_type,
                "part_number": args.part_number,
                "package_type": args.package_type,
                "query": query,
                "query_intent": query_intent,
                "error_family": args.error_family,
                "miss_reason": reason,
                "scene_id": scene_id_clean,
                "hits": [],
                "rules": [],
            },
        )

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

    # WP-0 (2026-05-24): the legacy KbService / Chroma / OpenAI fallback
    # that lived between ``local_datasheet_v2`` and ``LOCAL_DATASHEET_FALLBACKS``
    # has been removed. The flow is now:
    #   local_datasheet_v2 (DatasheetKbService, on-device OV embeddings)
    #     → LOCAL_DATASHEET_FALLBACKS (structured rules in this file)
    #     → "no_structured_datasheet_hit" miss
    # See ``docs/retrieval-contract.md``.

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
