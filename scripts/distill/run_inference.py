"""WP-2: 蒸馏推理的唯一授权入口（物理隔离）。

本脚本是 ``docs/retrieval-contract.md`` §3 中规定的唯一蒸馏入口。约束：

  1. **物理隔离**——只 import 统一 agent graph + agent tools，**不**
     import ``RagService`` / ``KbService`` / OpenAI embeddings 任何路径。
     启动时 ``_verify_isolation`` 扫描 ``sys.modules``，若 forbidden
     模块被任何 import chain 偷偷加载则立即 abort。
  2. **Precheck 闸门**——任何样本生成前必须 ``scripts/distill/precheck_retrieval.py``
     全 PASS（DISTILL_MODE / backend / 模型 / .npz / 维度）。
  3. **样本强校验**——每条输入样本必须有 (a) ``query`` 非空, (b)
     ``intent`` 在 4 个合法之列, (c) ``scene_id`` 在 6 demo 之内
     （diagnostic / mixed 必填；concept_tutor / lab_guidance 可空）。
     不合格样本被 skip 并记 audit reason，**绝不**用空 scene 调 agent。
  4. **审计输出**——每条样本写完整 evidence + tool_results + final_answer
     + run_metadata（distill_mode、cap、scene_resolved、verify_passed 等）
     便于追溯与论文复现。

输入格式（JSONL，每行一条样本）::

    {
      "qid": "ua741_inv_001",
      "query": "UA741 反相放大输出固定在 +13V 不变怎么办？",
      "intent": "diagnostic",
      "scene_id": "exp_ua741_inverting_amplifier",
      "station": {
        "station_id": "S_distill_001",
        "risk_level": "danger",
        "diagnostics": ["输出饱和"],
        "comparison_report": {
          "items": [{"error_code": "FLOATING_PIN", "component_id": "U1", "pin_name": "pin4"}]
        }
      }
    }

输出格式（JSONL，每行一条结果）::

    {
      "qid": "...",
      "query": "...", "intent": "...", "scene_id": "...",
      "agent_output": {
        "final_answer": "...",
        "tool_results": [...],
        "evidence_resolved_scene_id": "exp_ua741_inverting_amplifier",
        "react_iterations": 4,
        "verification_passed": true,
        ...
      },
      "audit": {
        "distill_mode": true,
        "run_at_iso": "2026-05-24T10:30:00Z",
        "skipped": false
      }
    }

调用方式::

    .venv/bin/python -m scripts.distill.run_inference \\
        --questions data/distill/questions_pilot20.jsonl \\
        --output    data/distill/teacher_traces_pilot20.jsonl

环境变量（precheck 会校验）::

    DISTILL_MODE=true
    DATASHEET_EMBEDDING_BACKEND=openvino
    DATASHEET_EMBEDDING_MODEL_DIR=models/bge-small-zh-v1.5-int8-ov

退出码::

    0 — 全部样本处理完毕（含部分 skip）。报告写完。
    1 — precheck 失败或 isolation 违约（未生成任何样本）。
    2 — IO / 参数错误。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# WP-2 isolation: ONLY import agent graph + agent tools. NEVER import
# ``app.services.rag_service`` or ``app.services.kb_service`` — those are
# blacklisted by ``_FORBIDDEN_MODULES`` and ``_verify_isolation``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent.contracts import AgentIntent  # noqa: E402
from app.agent.context_pack import build_context_pack  # noqa: E402
from app.agent.evidence import build_runtime_evidence_from_station  # noqa: E402
from app.agent.graph import run_diagnostic_graph  # noqa: E402
from app.agent.tool_runner import run_diagnostic_tools  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.error_tag_service import ErrorTagService  # noqa: E402
from app.services.scene_resolver import (  # noqa: E402
    TOPOLOGY_LABEL_TO_SCENE_ID,
    VALID_SCENE_IDS,
)

logger = logging.getLogger("scripts.distill.run_inference")


# Reverse lookup: scene_id → topology_label (so the entrypoint can stamp
# topology_label into the synthesized station, letting scene_resolver
# resolve the scene without requiring an automatic classifier).
_SCENE_ID_TO_TOPOLOGY_LABEL = {
    scene_id: topology_label
    for topology_label, scene_id in TOPOLOGY_LABEL_TO_SCENE_ID.items()
}

# Intents that require a resolved scene_id (no concept-only fallback).
_SCENE_REQUIRED_INTENTS: frozenset[str] = frozenset(["diagnostic", "mixed"])
_VALID_INTENTS: frozenset[str] = frozenset(
    ["diagnostic", "concept_tutor", "lab_guidance", "mixed"]
)
_VALID_FILTER_POLICIES: frozenset[str] = frozenset(["none", "recall_strict"])

# WP-2 isolation contract: the distillation entrypoint MUST NOT load any
# of these modules — they are the legacy / cloud / dev-only paths that
# would inject evidence the on-device runtime never produces.
_FORBIDDEN_MODULES: frozenset[str] = frozenset(
    [
        "app.services.rag_service",
        "app.services.kb_service",
        "langchain_openai",
        "langchain_community.vectorstores",
    ]
)


# ---------------------------------------------------------------------------
# Isolation + precheck guardrails
# ---------------------------------------------------------------------------


def _verify_isolation() -> None:
    """Refuse to run if any forbidden module is in ``sys.modules``.

    The entrypoint's own imports are intentionally narrow (see top of
    file). If a forbidden module shows up here, an indirect import chain
    smuggled it in — e.g. a newly added tool reaches into RagService.
    Caller must fix the import chain before distillation can proceed.
    """
    leaked = sorted(_FORBIDDEN_MODULES & set(sys.modules))
    if leaked:
        raise RuntimeError(
            "WP-2 isolation contract violated — these modules MUST NOT be "
            f"loaded in the distillation entrypoint: {leaked}. "
            "Audit the imports under app/agent/** and scripts/distill/** "
            "for a chain that reaches into the legacy / cloud retrieval "
            "paths. See docs/retrieval-contract.md §2."
        )


def _gate_on_precheck() -> int:
    """Run scripts/distill/precheck_retrieval.py; return its exit code."""
    from scripts.distill.precheck_retrieval import run_all_checks

    results = run_all_checks()
    for r in results:
        print(r.render())
    failed = [r for r in results if not r.passed]
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} precheck items FAILED. "
            "Distillation refuses to start. See docs/retrieval-contract.md §3.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(results)} precheck items passed.\n")
    return 0


# ---------------------------------------------------------------------------
# Sample validation + synthesis
# ---------------------------------------------------------------------------


@dataclass
class SampleValidation:
    ok: bool
    reason: str = ""


def _validate_sample(sample: dict) -> SampleValidation:
    """Per-sample contract enforcement. WP-2 + WP-3 v4 P1 alignment.

    Diagnostic / mixed must have a non-empty scene_id in the 6-demo set.
    concept_tutor / lab_guidance may have empty scene_id (concept-only)
    OR a valid demo scene_id; an invalid non-empty scene_id is rejected
    regardless of intent.
    """
    qid = sample.get("qid") or "<no-qid>"
    if not str(sample.get("query") or "").strip():
        return SampleValidation(False, f"{qid}: empty query")
    intent = sample.get("intent")
    if intent not in _VALID_INTENTS:
        return SampleValidation(
            False, f"{qid}: invalid intent {intent!r}; must be one of {sorted(_VALID_INTENTS)}"
        )
    scene_id = (sample.get("scene_id") or "").strip()
    if intent in _SCENE_REQUIRED_INTENTS and not scene_id:
        return SampleValidation(
            False,
            f"{qid}: intent={intent} requires a non-empty scene_id in "
            f"{sorted(VALID_SCENE_IDS)}",
        )
    if scene_id and scene_id not in VALID_SCENE_IDS:
        return SampleValidation(
            False,
            f"{qid}: scene_id={scene_id!r} is not one of the 6 demo scenes "
            f"({sorted(VALID_SCENE_IDS)})",
        )
    return SampleValidation(True)


def _synthesize_station(sample: dict) -> dict:
    """Build a minimal classroom station dict from a distill sample.

    The distill entrypoint sets ``topology_label`` from the sample's explicit
    ``scene_id``. This lets ``scene_resolver`` resolve the scene without
    needing a real netlist_v2 per sample.
    """
    base = dict(sample.get("station") or {})
    base.setdefault("station_id", sample.get("station_id") or f"S_distill_{sample.get('qid', 'x')}")
    scene_id = (sample.get("scene_id") or "").strip()
    if scene_id and scene_id in VALID_SCENE_IDS:
        base["scene_id"] = scene_id  # explicit override path
        topology = _SCENE_ID_TO_TOPOLOGY_LABEL.get(scene_id)
        if topology:
            base["topology_label"] = topology
    base.setdefault("risk_level", "safe")
    base.setdefault("diagnostics", [])
    base.setdefault("comparison_report", {"items": []})
    return base


def _collect_target_ids(sample: dict, plural_key: str, singular_key: str) -> set[str]:
    values: set[str] = set()
    plural = sample.get(plural_key)
    if isinstance(plural, list):
        values.update(str(item).strip() for item in plural if str(item).strip())
    singular = str(sample.get(singular_key) or "").strip()
    if singular:
        values.add(singular)
    return values


def _collect_matched_fault_case_ids(tool_results: list[dict]) -> set[str]:
    matched: set[str] = set()
    for result in tool_results:
        if result.get("tool_name") != "fault_case_lookup_tool":
            continue
        payload = result.get("payload") or {}
        for case in payload.get("fault_cases", []):
            knowledge_id = str(case.get("knowledge_id") or "").strip()
            if knowledge_id:
                matched.add(knowledge_id)
    return matched


def _collect_matched_datasheet_chunk_ids(tool_results: list[dict]) -> set[str]:
    matched: set[str] = set()
    for result in tool_results:
        if result.get("tool_name") != "datasheet_lookup_tool":
            continue
        payload = result.get("payload") or {}
        for hit in payload.get("hits", []):
            chunk_id = str(hit.get("chunk_id") or "").strip()
            if chunk_id:
                matched.add(chunk_id)
    return matched


def _count_tool_errors(tool_results: list[dict]) -> int:
    return sum(1 for result in tool_results if result.get("status") == "error")


def _evaluate_filter_policy(sample: dict, tool_results: list[dict], filter_policy: str) -> dict:
    if filter_policy == "none":
        return {
            "filter_policy": filter_policy,
            "kept": True,
            "reason": "disabled",
            "matched_fault_case_ids": [],
            "matched_datasheet_chunk_ids": [],
            "target_fault_case_ids": [],
            "target_datasheet_chunk_ids": [],
        }

    target_fault_case_ids = _collect_target_ids(
        sample,
        plural_key="target_fault_case_ids",
        singular_key="target_fault_case_id",
    )
    target_datasheet_chunk_ids = _collect_target_ids(
        sample,
        plural_key="target_datasheet_chunk_ids",
        singular_key="target_datasheet_chunk_id",
    )
    if not target_fault_case_ids and not target_datasheet_chunk_ids:
        return {
            "filter_policy": filter_policy,
            "kept": True,
            "reason": "no_targets_declared",
            "matched_fault_case_ids": [],
            "matched_datasheet_chunk_ids": [],
            "target_fault_case_ids": [],
            "target_datasheet_chunk_ids": [],
        }

    matched_fault_case_ids = sorted(
        target_fault_case_ids & _collect_matched_fault_case_ids(tool_results)
    )
    matched_datasheet_chunk_ids = sorted(
        target_datasheet_chunk_ids & _collect_matched_datasheet_chunk_ids(tool_results)
    )
    kept = bool(matched_fault_case_ids or matched_datasheet_chunk_ids)
    reason = (
        "matched_target_ids"
        if kept
        else "no_target_match"
    )
    return {
        "filter_policy": filter_policy,
        "kept": kept,
        "reason": reason,
        "matched_fault_case_ids": matched_fault_case_ids,
        "matched_datasheet_chunk_ids": matched_datasheet_chunk_ids,
        "target_fault_case_ids": sorted(target_fault_case_ids),
        "target_datasheet_chunk_ids": sorted(target_datasheet_chunk_ids),
    }


def _run_evidence_only(sample: dict, *, evidence, query: str, intent: AgentIntent, top_k: int) -> dict:
    context_pack = build_context_pack(
        evidence,
        query=query,
        user_message=sample.get("user_message", "") or query,
        intent=intent,
    )
    tool_results = [
        result.model_dump()
        for result in run_diagnostic_tools(
            evidence=evidence,
            context_pack=context_pack,
            query=query,
            top_k=top_k,
        )
    ]
    return {
        "final_answer": "",
        "draft_answer": "",
        "context_pack": context_pack.model_dump(),
        "tool_results": tool_results,
        "react_iterations": 0,
        "react_terminate_reason": "evidence_only",
        "evidence_resolved_scene_id": evidence.current_scene_id,
        "evidence_error_codes": list(evidence.error_codes),
        "evidence_error_tags": list(evidence.error_tags),
        "verification_passed": False,
    }


# ---------------------------------------------------------------------------
# Per-sample execution
# ---------------------------------------------------------------------------


def run_sample(
    sample: dict,
    *,
    top_k: int = 5,
    evidence_only: bool = False,
    filter_policy: str = "none",
) -> dict:
    """Process one sample. Always returns a record (skipped or processed).

    .. warning:: **Internal API** — production distillation MUST go through
        :func:`main` so the precheck gate and isolation guardrail fire.
        Calling ``run_sample`` directly skips both contracts and is meant
        for unit tests only (which set ``DISTILL_MODE`` via monkeypatch).
        Any new script that imports this function is a contract violation
        — file a follow-up to extend the entrypoint instead.
    """
    validation = _validate_sample(sample)
    base_record = {
        "qid": sample.get("qid"),
        "query": sample.get("query"),
        "intent": sample.get("intent"),
        "scene_id": sample.get("scene_id"),
    }
    if not validation.ok:
        return {
            **base_record,
            "audit": {
                "distill_mode": bool(getattr(settings, "DISTILL_MODE", False)),
                "run_at_iso": _now_iso(),
                "skipped": True,
                "skip_reason": validation.reason,
                "skip_reason_category": "validation",
            },
        }

    station = _synthesize_station(sample)
    evidence = build_runtime_evidence_from_station(
        station_id=station["station_id"],
        station=station,
        error_tag_service=ErrorTagService(),
    )

    intent: AgentIntent = sample["intent"]
    if evidence_only:
        agent_output = _run_evidence_only(
            sample,
            evidence=evidence,
            query=sample["query"],
            intent=intent,
            top_k=top_k,
        )
    else:
        state = run_diagnostic_graph(
            evidence=evidence,
            query=sample["query"],
            user_message=sample.get("user_message", ""),
            top_k=top_k,
            intent=intent,
        )

        context_pack = (
            state.context_pack.model_dump() if state.context_pack else None
        )
        verification_passed = bool(
            state.verification_report and state.verification_report.passed
        )
        agent_output = {
            "final_answer": state.final_answer,
            "draft_answer": state.draft_answer,
            "context_pack": context_pack,
            "tool_results": list(state.tool_results),
            "react_iterations": state.react_iterations,
            "react_terminate_reason": state.react_terminate_reason,
            "evidence_resolved_scene_id": evidence.current_scene_id,
            "evidence_error_codes": list(evidence.error_codes),
            "evidence_error_tags": list(evidence.error_tags),
            "verification_passed": verification_passed,
        }

    tool_results = list(agent_output.get("tool_results") or [])
    filter_audit = _evaluate_filter_policy(sample, tool_results, filter_policy)
    skipped = not filter_audit["kept"]
    skip_reason = ""
    skip_reason_category = ""
    if skipped:
        skip_reason = f"filter_policy:{filter_policy}:{filter_audit['reason']}"
        skip_reason_category = "filter"

    return {
        **base_record,
        "agent_output": agent_output,
        "audit": {
            "distill_mode": bool(getattr(settings, "DISTILL_MODE", False)),
            "run_at_iso": _now_iso(),
            "evidence_only": evidence_only,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "skip_reason_category": skip_reason_category,
            "filter": filter_audit,
            "tool_error_count": _count_tool_errors(tool_results),
        },
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "skip malformed JSON at %s:%d — %s", path, line_no, exc
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questions", required=True, type=Path,
        help="JSONL input — one sample per line (see module docstring)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="JSONL output — one record per line, includes evidence + audit",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process only the first N samples (smoke testing)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="top_k for the agent graph (default: 5)",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="only build evidence/context_pack and execute tools once; skip ReAct/verify/finalize",
    )
    parser.add_argument(
        "--filter-policy",
        choices=sorted(_VALID_FILTER_POLICIES),
        default="none",
        help="post-run filtering policy for distill samples (default: none)",
    )
    parser.add_argument(
        "--fail-on-exception",
        action="store_true",
        help="abort the batch immediately on the first sample exception",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.0,
        help="abort with exit 1 when sample exception rate exceeds this ratio (0.0-1.0, default: 0.0)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    # WP-2.1 (2026-05-24): ``--skip-precheck`` REMOVED. Earlier audit
    # flagged it as a contract-bypass vector — any operator could ship
    # production 5k data with retrieval mis-configured. Precheck is now
    # unconditionally enforced by the CLI; tests that want to skip the
    # gate must call ``_run_one_sample`` directly (not through main).
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not 0.0 <= args.max_error_rate <= 1.0:
        print("--max-error-rate must be within [0.0, 1.0]", file=sys.stderr)
        return 2

    if not args.questions.is_file():
        print(f"--questions does not exist: {args.questions}", file=sys.stderr)
        return 2

    # WP-3 precheck gate — unconditionally enforced (WP-2.1 hardened).
    code = _gate_on_precheck()
    if code != 0:
        return code

    # WP-2 isolation guardrail — runs AFTER precheck has imported its own
    # bits, so this catches forbidden modules that *precheck itself* would
    # have smuggled in (it doesn't, but the defense is in depth).
    _verify_isolation()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    exception_count = 0
    total_seen = 0
    exit_code = 0
    with args.output.open("w", encoding="utf-8") as out:
        for line_no, sample in iter_jsonl(args.questions):
            if args.limit is not None and processed + skipped >= args.limit:
                break
            try:
                record = run_sample(
                    sample,
                    top_k=args.top_k,
                    evidence_only=args.evidence_only,
                    filter_policy=args.filter_policy,
                )
            except Exception as exc:  # noqa: BLE001
                exception_count += 1
                logger.exception("sample %s (line %d) raised", sample.get("qid"), line_no)
                record = {
                    "qid": sample.get("qid"),
                    "query": sample.get("query"),
                    "intent": sample.get("intent"),
                    "scene_id": sample.get("scene_id"),
                    "audit": {
                        "distill_mode": bool(getattr(settings, "DISTILL_MODE", False)),
                        "run_at_iso": _now_iso(),
                        "evidence_only": args.evidence_only,
                        "skipped": True,
                        "skip_reason": f"exception: {type(exc).__name__}: {exc}",
                        "skip_reason_category": "exception",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                }
            if record.get("audit", {}).get("skipped"):
                skipped += 1
            else:
                processed += 1
            total_seen += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.fail_on_exception and exception_count > 0:
                exit_code = 1
                break

    if total_seen:
        error_rate = exception_count / total_seen
        if error_rate > args.max_error_rate:
            print(
                f"sample exception rate {error_rate:.3f} exceeded --max-error-rate={args.max_error_rate:.3f}",
                file=sys.stderr,
            )
            exit_code = 1

    logger.info(
        "done — processed=%d skipped=%d exceptions=%d output=%s",
        processed,
        skipped,
        exception_count,
        args.output.relative_to(REPO_ROOT)
        if args.output.is_relative_to(REPO_ROOT)
        else args.output,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
