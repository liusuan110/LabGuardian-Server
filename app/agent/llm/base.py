"""LLMProvider abstract interface for the ReAct loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.agent.contracts import (
    ContextPack,
    ReActStep,
    ReflectionResult,
    RuntimeEvidence,
    ToolCall,
    VerificationReport,
)


@dataclass
class PlanRequest:
    """Inputs handed to a planner to decide the next ReAct step."""

    iteration: int
    evidence: RuntimeEvidence
    context_pack: ContextPack
    query: str
    user_message: str
    prior_steps: list[ReActStep]
    tool_results_so_far: list[dict[str, Any]]


@dataclass
class ReflectRequest:
    """Inputs handed to a reflector to critique the draft answer."""

    iteration: int
    evidence: RuntimeEvidence
    context_pack: ContextPack
    draft_answer: str
    verification_report: VerificationReport | None
    prior_steps: list[ReActStep]


class LLMProvider(ABC):
    """Provider-agnostic interface for the diagnostic ReAct loop.

    Concrete providers must be deterministic when seeded the same way; this
    keeps CI green without real model weights.
    """

    name: str = "abstract"

    @abstractmethod
    def plan(self, request: PlanRequest) -> ToolCall | None:
        """Return the next tool to call, or `None` to skip the Act step.

        Implementations MUST restrict the tool name to `request.context_pack
        .allowed_tools` to enforce the white-box action whitelist.
        """

    @abstractmethod
    def reflect(self, request: ReflectRequest) -> ReflectionResult:
        """Critique the draft answer and decide whether to terminate the loop."""

    def warmup(self) -> None:
        """Optional pre-load hook (no-op for deterministic providers)."""
        return None

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        system: str | None = None,
    ) -> str:
        """Free-form text generation (the "mouth").

        Optional capability: the ReAct loop only requires ``plan``/
        ``reflect``. ``AgentService`` uses ``generate`` to turn an
        evidence-grounded draft / prompt into fluent natural language.
        Providers that wrap a real text LLM (``openvino_genai_text``)
        override this; deterministic/template providers do not support it
        and raise ``NotImplementedError`` so callers fall back to the
        rule-based draft.
        """
        raise NotImplementedError(
            f"{self.name} provider does not support generate()"
        )
