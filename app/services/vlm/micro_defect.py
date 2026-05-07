"""Helper that runs a defect-type-specialized VLM inspection.

Reuses the existing `VlmService.explain_rc_pack` so all three providers
(template / openai_compatible / openvino_genai) return the same schema.
"""

from __future__ import annotations

from typing import Any

from app.services.vlm.defect_types import DEFECT_TYPE_PROMPTS, MicroDefectType


def analyze_micro_defect(
    *,
    vlm_service,  # type: ignore[no-untyped-def] — avoid import cycle
    defect_type: MicroDefectType,
    mrag_pack: dict[str, Any],
    user_query: str = "",
    current_image: str | None = None,
    reference_image: str | None = None,
) -> dict[str, Any]:
    """Run a focused VLM inspection for a single micro-defect type.

    The returned dict matches `vlm_explanation_v1` plus an extra
    `defect_type` key so downstream consumers can attribute the conclusion.
    """
    prompt_addendum = DEFECT_TYPE_PROMPTS[defect_type]
    base_query = (user_query or mrag_pack.get("query", "")).strip()
    composed_query = (
        f"[微观缺陷复检 · {defect_type.value}] {prompt_addendum} "
        + (f"原始问题: {base_query}" if base_query else "")
    ).strip()

    result = vlm_service.explain_rc_pack(
        mrag_pack=mrag_pack,
        user_query=composed_query,
        current_image=current_image,
        reference_image=reference_image,
    )
    # Annotate so consumers can route results without re-parsing the prompt.
    if isinstance(result, dict):
        result.setdefault("defect_type", defect_type.value)
    return result
