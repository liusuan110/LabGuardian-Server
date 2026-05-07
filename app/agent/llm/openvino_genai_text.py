"""OpenVINO GenAI text-LLM provider stub.

Reserved for Phase 7+ once DK-2500 NPU is validated. Until then, this module
intentionally raises `NotImplementedError` so the factory falls back to the
deterministic `template` provider — keeping CI green and memory budget safe
on 16GB DK-2500.

When implementing, mirror the contract in `template_provider.py`:
- `plan()` must restrict to `context_pack.allowed_tools`
- `reflect()` must call `verify_draft_answer` (or stricter critic) so the
  termination semantics stay aligned with the deterministic baseline
"""

from __future__ import annotations

from app.agent.contracts import ReflectionResult, ToolCall
from app.agent.llm.base import LLMProvider, PlanRequest, ReflectRequest


class OpenVinoGenAITextProvider(LLMProvider):
    name = "openvino_genai_text"

    def __init__(self, *, model_dir: str = "", device: str = "GPU") -> None:
        self._model_dir = model_dir
        self._device = device

    def warmup(self) -> None:  # pragma: no cover - Phase 7+
        raise NotImplementedError(
            "openvino_genai_text provider is reserved for Phase 7+ "
            "(DK-2500 NPU validation). Use VLM_PROVIDER=template until then."
        )

    def plan(self, request: PlanRequest) -> ToolCall | None:  # pragma: no cover
        raise NotImplementedError(
            "openvino_genai_text.plan is not implemented yet"
        )

    def reflect(self, request: ReflectRequest) -> ReflectionResult:  # pragma: no cover
        raise NotImplementedError(
            "openvino_genai_text.reflect is not implemented yet"
        )
