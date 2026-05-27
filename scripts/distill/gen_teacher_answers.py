"""Generate teacher answers from frozen distillation evidence.

This script consumes the JSONL emitted by ``scripts.distill.run_inference``
in ``--evidence-only`` mode and calls one or more teacher LLMs against the
same frozen evidence. Each teacher call is treated as an auditable sample:

* input is the original ``query`` plus frozen ``context_pack`` / ``tool_results``
* output is normalized JSON with ``answer`` / ``citations`` / ``safety_notes``
* transport is currently ``openai_compatible`` only, which fits DeepSeek's
  official API and most hosted Qwen services

Example::

    python -m scripts.distill.gen_teacher_answers ^
      --evidence datasets\\distill\\pilot20_evidence_strict.jsonl ^
      --output datasets\\distill\\pilot20_teacher_answers.jsonl ^
      --teacher "name=qwen,model=Qwen3-32B,base_url=https://example/v1,api_key_env=QWEN_API_KEY" ^
      --teacher "name=deepseek,model=deepseek-chat,base_url=https://api.deepseek.com/v1,api_key_env=DEEPSEEK_API_KEY"

Exit codes::

    0 — finished (may include per-sample error records unless --fail-on-error)
    1 — at least one teacher call failed and --fail-on-error is set
    2 — bad CLI args / malformed teacher specs / unreadable input
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402

logger = logging.getLogger("scripts.distill.gen_teacher_answers")


_SYSTEM_PROMPT = """你是 LabGuardian 蒸馏教师模型。

任务要求：
1. 只能基于提供的 frozen evidence 回答，禁止补充证据中不存在的器件、引脚、节点、拓扑、测量结果。
2. 优先给出可执行、可核查的教学回答，不要写空洞套话。
3. 如果 evidence 显示存在危险或 warning，先给安全提示，再给诊断或原理解释。
4. 如果证据不足，必须明确说“证据不足”，不能假装确定。
5. 输出必须是 JSON 对象，字段如下：
   {
     "answer": "面向学生的最终回答",
     "citations": ["引用1", "引用2"],
     "safety_notes": ["安全提示1"],
     "reasoning_brief": "不超过120字，简述为何这样回答"
   }

约束：
- `answer` 用中文，尽量先结论后步骤。
- `citations` 只引用 evidence 中真实出现的信息，如 error_code、tool 名称、knowledge_id、chunk_id、scene_id、ref_id。
- 只要 `answer` 给出具体原理、诊断、结论或步骤，`citations` 就必须至少包含 1 条可核对引用。
- 如果找不到可引用证据，`answer` 必须明确以“证据不足”开头，`citations` 留空，不得输出教材式泛化讲解。
- 禁止使用 frozen evidence 之外的常识来补全答案。
- 不要输出 markdown 代码块，不要输出 JSON 之外的任何文本。
"""


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str
    api_key: str
    provider: str = "openai_compatible"
    temperature: float = 0.0
    max_tokens: int = 900


def _sanitize_base_url(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.strip("`").strip().strip("'").strip('"').strip()
    return text.rstrip("/")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON at {path}:{line_no}: {exc}") from exc


def _record_resume_key(payload: dict[str, Any]) -> tuple[str, str]:
    return (
        str(payload.get("qid") or "").strip(),
        str(payload.get("teacher_name") or "").strip(),
    )


def _parse_teacher_spec(raw: str) -> TeacherSpec:
    fields: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"invalid --teacher segment {part!r}; expected key=value pairs"
            )
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()

    name = fields.get("name", "")
    model = fields.get("model", "")
    base_url = _sanitize_base_url(fields.get("base_url", ""))
    api_key_env = fields.get("api_key_env", "")
    provider = (fields.get("provider") or "openai_compatible").strip().lower()
    if provider != "openai_compatible":
        raise ValueError(f"unsupported teacher provider: {provider!r}")
    if not name or not model or not base_url or not api_key_env:
        raise ValueError(
            "--teacher requires name=...,model=...,base_url=...,api_key_env=..."
        )
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"teacher {name!r} requires env {api_key_env!r}, but it is empty"
        )
    try:
        temperature = float(fields.get("temperature", "0.0"))
        max_tokens = int(fields.get("max_tokens", "900"))
    except ValueError as exc:
        raise ValueError(f"invalid numeric teacher option in {raw!r}") from exc
    return TeacherSpec(
        name=name,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _teacher_from_settings() -> TeacherSpec:
    model = str(getattr(settings, "LLM_MODEL", "") or "").strip()
    base_url = _sanitize_base_url(getattr(settings, "LLM_BASE_URL", "") or "")
    api_key = str(getattr(settings, "LLM_API_KEY", "") or "").strip()
    if not model or not base_url or not api_key:
        raise ValueError(
            "no --teacher provided and settings LLM_API_KEY / LLM_BASE_URL / LLM_MODEL are incomplete"
        )
    return TeacherSpec(
        name="default",
        model=model,
        base_url=base_url,
        api_key_env="LLM_API_KEY",
        api_key=api_key,
    )


def _build_user_prompt(record: dict[str, Any]) -> str:
    compact = {
        "qid": record.get("qid"),
        "query": record.get("query"),
        "intent": record.get("intent"),
        "scene_id": record.get("scene_id"),
        "agent_output": {
            "context_pack": ((record.get("agent_output") or {}).get("context_pack")),
            "tool_results": ((record.get("agent_output") or {}).get("tool_results")),
            "evidence_resolved_scene_id": (
                (record.get("agent_output") or {}).get("evidence_resolved_scene_id")
            ),
            "evidence_error_codes": (
                (record.get("agent_output") or {}).get("evidence_error_codes")
            ),
            "evidence_error_tags": (
                (record.get("agent_output") or {}).get("evidence_error_tags")
            ),
        },
        "audit": {
            "filter": ((record.get("audit") or {}).get("filter")),
            "tool_error_count": ((record.get("audit") or {}).get("tool_error_count")),
            "evidence_only": ((record.get("audit") or {}).get("evidence_only")),
        },
    }
    frozen_evidence = json.dumps(compact, ensure_ascii=False, indent=2)
    return (
        "下面是同一条蒸馏样本的 frozen evidence。\n"
        "你必须只依据这些证据回答学生原问题。\n\n"
        f"{frozen_evidence}\n\n"
        "请按 system prompt 规定输出 JSON。"
    )


def _extract_first_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty teacher response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("teacher response does not contain a JSON object")


def _normalize_teacher_json(parsed: dict[str, Any]) -> dict[str, Any]:
    answer = str(parsed.get("answer") or "").strip()
    citations_raw = parsed.get("citations")
    safety_raw = parsed.get("safety_notes")
    reasoning_brief = str(parsed.get("reasoning_brief") or "").strip()

    citations = []
    if isinstance(citations_raw, list):
        citations = [str(item).strip() for item in citations_raw if str(item).strip()]
    elif citations_raw:
        citations = [str(citations_raw).strip()]

    safety_notes = []
    if isinstance(safety_raw, list):
        safety_notes = [str(item).strip() for item in safety_raw if str(item).strip()]
    elif safety_raw:
        safety_notes = [str(safety_raw).strip()]

    return {
        "answer": answer,
        "citations": citations,
        "safety_notes": safety_notes,
        "reasoning_brief": reasoning_brief[:120],
    }


def _add_labeled_token(target: set[str], label: str, value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    target.add(text)
    target.add(f"{label}={text}")


def _collect_allowed_citation_tokens(payload: Any, target: set[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {
                "ref_id",
                "knowledge_id",
                "chunk_id",
                "error_code",
                "scene_id",
                "tool_name",
                "component_id",
                "pin_name",
                "electrical_node_id",
                "hole_id",
            }:
                _add_labeled_token(target, key, value)
            _collect_allowed_citation_tokens(value, target)
    elif isinstance(payload, list):
        for item in payload:
            _collect_allowed_citation_tokens(item, target)


def _collect_allowed_citations(record: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    _add_labeled_token(allowed, "scene_id", record.get("scene_id"))
    agent_output = record.get("agent_output") or {}
    _add_labeled_token(
        allowed,
        "scene_id",
        agent_output.get("evidence_resolved_scene_id"),
    )
    for code in agent_output.get("evidence_error_codes") or []:
        _add_labeled_token(allowed, "error_code", code)
    for tag in agent_output.get("evidence_error_tags") or []:
        _add_labeled_token(allowed, "error_tag", tag)
    _collect_allowed_citation_tokens(agent_output.get("context_pack") or {}, allowed)
    _collect_allowed_citation_tokens(agent_output.get("tool_results") or [], allowed)
    return allowed


def _is_supported_citation(citation: str, allowed: set[str]) -> bool:
    text = str(citation or "").strip()
    if not text:
        return False
    if text in allowed:
        return True
    if "=" in text:
        _, rhs = text.split("=", 1)
        rhs = rhs.strip()
        if rhs and rhs in allowed:
            return True
    return False


def _make_evidence_insufficient_output(
    *,
    answer: str,
    safety_notes: list[str],
    reasoning_brief: str,
) -> dict[str, Any]:
    # Keep the downgraded output terse and deterministic so the student model
    # does not learn free-form generic advice when frozen evidence is absent.
    final_answer = "证据不足，无法基于当前 frozen evidence 给出确定回答。"
    brief = "当前回答未提供可核对引用，已降级为证据不足。"
    return {
        "answer": final_answer,
        "citations": [],
        "safety_notes": safety_notes,
        "reasoning_brief": brief[:120],
    }


def _post_chat_completion(
    *,
    client: httpx.Client,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    response = client.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def _enforce_teacher_output_contract(record: dict[str, Any], teacher_output: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed_citations = _collect_allowed_citations(record)
    supported_citations = [
        citation
        for citation in teacher_output.get("citations") or []
        if _is_supported_citation(citation, allowed_citations)
    ]
    answer = str(teacher_output.get("answer") or "").strip()
    safety_notes = [
        str(item).strip()
        for item in teacher_output.get("safety_notes") or []
        if str(item).strip()
    ]
    reasoning_brief = str(teacher_output.get("reasoning_brief") or "").strip()

    downgraded = False
    if supported_citations:
        teacher_output["citations"] = supported_citations
    else:
        teacher_output = _make_evidence_insufficient_output(
            answer=answer,
            safety_notes=safety_notes,
            reasoning_brief=reasoning_brief,
        )
        downgraded = True

    context_pack = ((record.get("agent_output") or {}).get("context_pack") or {})
    forced_poweroff = False
    if str(context_pack.get("risk_level") or "").strip().lower() == "danger":
        poweroff_note = "先断电，再调整接线或元件。"
        combined_safety = " ".join(teacher_output.get("safety_notes") or [])
        answer_text = str(teacher_output.get("answer") or "")
        has_poweroff = any(keyword in answer_text or keyword in combined_safety for keyword in ("断电", "断开电源"))
        if not has_poweroff:
            teacher_output["answer"] = f"先断电，再进行下面检查。\n{answer_text}".strip()
            notes = list(teacher_output.get("safety_notes") or [])
            notes.insert(0, poweroff_note)
            teacher_output["safety_notes"] = notes
            forced_poweroff = True

    return teacher_output, {
        "allowed_citation_count": len(allowed_citations),
        "supported_citation_count": len(supported_citations),
        "downgraded_to_evidence_insufficient": downgraded,
        "forced_poweroff_warning": forced_poweroff,
    }


def _call_openai_compatible(
    *,
    teacher: TeacherSpec,
    messages: list[dict[str, str]],
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    endpoint = f"{teacher.base_url}/chat/completions"
    payload = {
        "model": teacher.model,
        "messages": messages,
        "temperature": teacher.temperature,
        "max_tokens": teacher.max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {teacher.api_key}",
        "Content-Type": "application/json",
    }
    start = time.perf_counter()
    fallback_without_response_format = False
    with httpx.Client(timeout=timeout_s, trust_env=False) as client:
        try:
            body = _post_chat_completion(
                client=client,
                endpoint=endpoint,
                payload=payload,
                headers=headers,
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 400:
                raise
            retry_payload = dict(payload)
            retry_payload.pop("response_format", None)
            body = _post_chat_completion(
                client=client,
                endpoint=endpoint,
                payload=retry_payload,
                headers=headers,
            )
            fallback_without_response_format = True
    latency_ms = (time.perf_counter() - start) * 1000.0
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("teacher API returned no choices")
    choice0 = choices[0] or {}
    message = choice0.get("message") if isinstance(choice0, dict) else None
    content = str((message or {}).get("content") or "").strip()
    parsed = _extract_first_json_object(content)
    normalized = _normalize_teacher_json(parsed)
    meta = {
        "finish_reason": choice0.get("finish_reason"),
        "usage": body.get("usage"),
        "raw_content": content,
        "fallback_without_response_format": fallback_without_response_format,
    }
    return normalized, meta, latency_ms


def _sample_fingerprint(record: dict[str, Any]) -> str:
    payload = {
        "qid": record.get("qid"),
        "query": record.get("query"),
        "scene_id": record.get("scene_id"),
        "context_pack": ((record.get("agent_output") or {}).get("context_pack")),
        "tool_results": ((record.get("agent_output") or {}).get("tool_results")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _validate_evidence_record(record: dict[str, Any]) -> str | None:
    if not str(record.get("qid") or "").strip():
        return "missing qid"
    if not str(record.get("query") or "").strip():
        return "missing query"
    agent_output = record.get("agent_output") or {}
    if not agent_output.get("context_pack"):
        return "missing agent_output.context_pack"
    if not isinstance(agent_output.get("tool_results"), list):
        return "missing agent_output.tool_results"
    audit = record.get("audit") or {}
    if audit.get("skipped"):
        return f"evidence record already skipped: {audit.get('skip_reason') or 'unknown'}"
    return None


def _build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(record)},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help=(
            "teacher spec: "
            "name=<name>,model=<model>,base_url=<url>,api_key_env=<ENV>[,temperature=0,max_tokens=900]"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip teacher records already written there.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.evidence.is_file():
        print(f"--evidence does not exist: {args.evidence}", file=sys.stderr)
        return 2
    if args.timeout_s <= 0:
        print("--timeout-s must be > 0", file=sys.stderr)
        return 2

    try:
        teachers = (
            [_parse_teacher_spec(raw) for raw in args.teacher]
            if args.teacher
            else [_teacher_from_settings()]
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed_keys: set[tuple[str, str]] = set()
    if args.resume and args.output.exists():
        for _, payload in iter_jsonl(args.output):
            if not isinstance(payload, dict):
                continue
            key = _record_resume_key(payload)
            if key[0] and key[1]:
                completed_keys.add(key)
        logger.info(
            "resume mode loaded existing teacher records=%d from %s",
            len(completed_keys),
            args.output.relative_to(REPO_ROOT)
            if args.output.is_relative_to(REPO_ROOT)
            else args.output,
        )

    total = len(completed_keys)
    error_count = 0
    open_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(open_mode, encoding="utf-8", newline="\n") as out:
        for _, record in iter_jsonl(args.evidence):
            if args.limit is not None and total >= args.limit:
                break
            validation_error = _validate_evidence_record(record)
            for teacher in teachers:
                resume_key = (
                    str(record.get("qid") or "").strip(),
                    teacher.name,
                )
                if resume_key in completed_keys:
                    continue
                total += 1
                base = {
                    "qid": record.get("qid"),
                    "teacher_name": teacher.name,
                    "teacher_model": teacher.model,
                    "teacher_provider": teacher.provider,
                    "scene_id": record.get("scene_id"),
                    "source_query": record.get("query"),
                    "source_evidence_fingerprint": _sample_fingerprint(record),
                    "source_evidence_path": str(args.evidence),
                    "generated_at_iso": _now_iso(),
                }
                if validation_error:
                    error_count += 1
                    result = {
                        **base,
                        "teacher_output": {
                            "answer": "",
                            "citations": [],
                            "safety_notes": [],
                            "reasoning_brief": "",
                        },
                        "generation": {
                            "ok": False,
                            "error_type": "ValidationError",
                            "error_message": validation_error,
                        },
                    }
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed_keys.add(resume_key)
                    if args.fail_on_error:
                        return 1
                    continue

                messages = _build_messages(record)
                try:
                    teacher_output, meta, latency_ms = _call_openai_compatible(
                        teacher=teacher,
                        messages=messages,
                        timeout_s=args.timeout_s,
                    )
                    teacher_output, contract_audit = _enforce_teacher_output_contract(
                        record,
                        teacher_output,
                    )
                    result = {
                        **base,
                        "teacher_output": teacher_output,
                        "generation": {
                            "ok": True,
                            "latency_ms": round(latency_ms, 2),
                            "finish_reason": meta.get("finish_reason"),
                            "usage": meta.get("usage"),
                            "api_key_env": teacher.api_key_env,
                            "prompt_version": "teacher_v1_frozen_evidence",
                            "fallback_without_response_format": bool(
                                meta.get("fallback_without_response_format")
                            ),
                            "contract_audit": contract_audit,
                        },
                    }
                except Exception as exc:  # noqa: BLE001
                    error_count += 1
                    logger.exception("teacher=%s qid=%s failed", teacher.name, record.get("qid"))
                    result = {
                        **base,
                        "teacher_output": {
                            "answer": "",
                            "citations": [],
                            "safety_notes": [],
                            "reasoning_brief": "",
                        },
                        "generation": {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "api_key_env": teacher.api_key_env,
                            "prompt_version": "teacher_v1_frozen_evidence",
                        },
                    }
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed_keys.add(resume_key)
                    if args.fail_on_error:
                        return 1
                    continue

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed_keys.add(resume_key)

    logger.info(
        "done — teacher_calls=%d errors=%d output=%s",
        total,
        error_count,
        args.output.relative_to(REPO_ROOT)
        if args.output.is_relative_to(REPO_ROOT)
        else args.output,
    )
    return 0 if error_count == 0 or not args.fail_on_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
