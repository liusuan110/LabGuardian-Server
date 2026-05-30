"""OpenVINO GenAI text-LLM provider (board-side distilled student model).

Board-StageB-2.1 (2026-05-29): real implementation. Runs the distilled
``labguardian-student-1p5-int4-ov`` (Qwen2.5-1.5B INT4) on the DK-2500 iGPU
via ``openvino_genai.LLMPipeline``.

## Division of labour (project philosophy: deterministic truth, LLM = mouth)

The ReAct *orchestration* (which tool to call, when to stop) stays
**deterministic**: ``plan()`` / ``reflect()`` delegate to
:class:`TemplateLLMProvider`, whose ``reflect()`` uses the rule-based
``verify_draft_answer`` as a hard gate. We do NOT let a 1.5B model decide
tool calls — that would reintroduce hallucination into the orchestration
layer that the whole retrieval contract exists to prevent.

The student model is used ONLY for ``generate()`` — turning an
evidence-grounded draft / prompt into fluent natural-language teaching
text. Callers (``AgentService``) anchor the prompt to deterministic facts
(error_codes / component_ids / fault_cases) and re-run the deterministic
verifier on the output, so the "mouth" can never invent structural facts.

## Failure semantics

Any load/generation failure raises (``warmup`` / ``generate``). The factory
(``app/agent/llm/factory.py``) catches it and falls back to the
``template`` provider, keeping the graph alive. ``generate()`` callers also
treat an empty/raised result as "use the deterministic draft".

Mirrors the OpenVINO GenAI calling pattern in
``app/services/vlm_service.py::_explain_with_openvino_genai`` (singleton
pipeline, lazy load, ``GenerationConfig``), but uses ``LLMPipeline``
(text) instead of ``VLMPipeline``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agent.contracts import ReflectionResult, ToolCall
from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest
from app.agent.llm.template_provider import TemplateLLMProvider

logger = logging.getLogger(__name__)


class OpenVinoGenAITextProvider(LLMProvider):
    name = "openvino_genai_text"

    def __init__(
        self,
        *,
        model_dir: str = "",
        device: str = "GPU",
        max_new_tokens: int = 320,
        repetition_penalty: float = 1.15,
    ) -> None:
        self._model_dir = model_dir
        self._device = device or "GPU"
        self._max_new_tokens = max(1, int(max_new_tokens))
        # Small INT4 models occasionally fall into a line-level repetition
        # loop (e.g. repeating a citation bullet N times). A mild penalty
        # discourages it; ``_dedup_repeated_lines`` cleans up any residue.
        self._repetition_penalty = float(repetition_penalty)
        # Lazy singletons — first ``warmup``/``generate`` loads them.
        self._pipeline: Any | None = None
        self._genai: Any | None = None
        # Deterministic delegate: tool orchestration must NOT be model-driven.
        self._orchestrator = TemplateLLMProvider()

    # ------------------------------------------------------------------
    # ReAct orchestration — delegated to the deterministic template planner
    # ------------------------------------------------------------------
    def plan(self, request: PlanRequest) -> ToolCall | None:
        return self._orchestrator.plan(request)

    def reflect(self, request: ReflectRequest) -> ReflectionResult:
        return self._orchestrator.reflect(request)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _load_modules(self) -> Any:
        """Import OpenVINO GenAI. Raises ImportError if runtime missing."""
        import openvino  # noqa: F401 — ensures runtime present
        import openvino_genai

        return openvino_genai

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        genai = self._load_modules()
        model_dir = Path(str(self._model_dir or ""))
        if not model_dir.exists():
            raise FileNotFoundError(
                f"OpenVINO LLM model dir not found: {model_dir!s}"
            )
        logger.info(
            "[openvino_genai_text] loading LLMPipeline dir=%s device=%s",
            model_dir,
            self._device,
        )
        self._pipeline = genai.LLMPipeline(str(model_dir), self._device)
        self._genai = genai
        return self._pipeline

    def warmup(self) -> None:
        """Eagerly load + JIT-compile the pipeline.

        Raises on any failure so the factory falls back to ``template``
        rather than caching a half-broken provider.
        """
        pipe = self._get_pipeline()
        cfg = self._genai.GenerationConfig()
        cfg.max_new_tokens = 1
        # A 1-token generate forces the GPU plugin to compile now (the
        # ~8s one-time cost observed on DK-2500) instead of on the first
        # real user turn.
        pipe.generate("ok", generation_config=cfg)
        logger.info("[openvino_genai_text] warmup complete")

    # ------------------------------------------------------------------
    # The "mouth": grounded natural-language generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> str:
        """Generate text from the distilled student model.

        Raises on pipeline/runtime failure; callers fall back to the
        deterministic draft.
        """
        pipe = self._get_pipeline()
        cfg = self._genai.GenerationConfig()
        cfg.max_new_tokens = int(max_new_tokens or self._max_new_tokens)
        # Low-temperature, near-greedy decoding for stable, grounded answers.
        try:
            if temperature and temperature > 0:
                cfg.do_sample = True
                cfg.temperature = float(temperature)
            else:
                cfg.do_sample = False
            if hasattr(cfg, "repetition_penalty"):
                cfg.repetition_penalty = self._repetition_penalty
        except Exception:  # pragma: no cover - older GenerationConfig schemas
            pass
        full_prompt = prompt if not system else f"{system}\n\n{prompt}"
        result = pipe.generate(full_prompt, generation_config=cfg)
        return self._dedup_repeated_lines(self._extract_text(result).strip())

    @staticmethod
    def _dedup_repeated_lines(text: str) -> str:
        """Collapse the small-model line-repetition artifact (display layer).

        Drops a line if its stripped form already appeared earlier in the
        output. Blank lines are always kept (they preserve paragraph breaks),
        and a duplicate is only removed when its content is non-trivial
        (≥ 2 chars) so we never merge legitimately-repeated short markers
        like numbering. This only fires on the verbatim-duplicate case
        (e.g. ``- COMPONENT_SHORTED_SAME_NET:R2:`` repeated 10×).
        """
        seen: set[str] = set()
        out: list[str] = []
        for raw in text.splitlines():
            key = raw.strip()
            if not key:
                out.append(raw)
                continue
            if len(key) >= 2 and key in seen:
                continue
            seen.add(key)
            out.append(raw)
        return "\n".join(out)

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Normalize openvino_genai generate() output to plain text.

        ``LLMPipeline.generate`` returns either a plain ``str`` or a
        ``DecodedResults`` whose ``.texts[0]`` holds the completion.
        """
        if isinstance(result, str):
            return result
        texts = getattr(result, "texts", None)
        if texts:
            try:
                return str(texts[0])
            except (IndexError, TypeError):
                pass
        return str(result)
