"""Phase 6 VLM extension package.

Phase 6 keeps the existing `app/services/vlm_service.py` intact (provider
selection, OpenVINO GenAI wiring, OpenAI-compatible fallback) and adds:

- `defect_types.MicroDefectType` enum + per-type prompts for white-box-gated
  micro-defect inspection (BURN_MARK / UNSTRIPPED_WIRE / COLD_SOLDER)
- `analyze_micro_defect()` that wraps `VlmService.explain_rc_pack` with a
  defect-specialized prompt so any provider (template / openai_compatible /
  openvino_genai) returns the same `vlm_explanation_v1` shape

A full provider repackage (Phase 6 stretch goal) is intentionally deferred —
the existing single-file VlmService is already thin and works.
"""

from app.services.vlm.defect_types import (
    DEFECT_TYPE_PROMPTS,
    MicroDefectType,
    SUSPICIOUS_TAGS_BY_TYPE,
    suggest_defect_types,
)
from app.services.vlm.micro_defect import analyze_micro_defect

__all__ = [
    "MicroDefectType",
    "DEFECT_TYPE_PROMPTS",
    "SUSPICIOUS_TAGS_BY_TYPE",
    "suggest_defect_types",
    "analyze_micro_defect",
]
