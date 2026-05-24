"""Per-node modules for the diagnostic LangGraph.

This package contains one module per graph node so each node can be unit-tested
in isolation.

WP-2.1 (2026-05-24): ``vlm_explain_node`` was removed — it transitively
imported ``app.core.deps`` which loads ``RagService`` / ``KbService``,
breaking the WP-2 distillation isolation contract. VLM micro-defect
inspection is out of project scope for the current iteration.

Public surface for `app.agent.graph`:
"""

from app.agent.nodes.classify import classify_error_node
from app.agent.nodes.context import build_context_pack_node
from app.agent.nodes.finalize import finalize_answer_node
from app.agent.nodes.react_observe import react_observe_node
from app.agent.nodes.react_plan import react_plan_node
from app.agent.nodes.react_reflect import react_reflect_node, should_continue_react
from app.agent.nodes.repair import repair_answer_node
from app.agent.nodes.tools_node import run_tools_node
from app.agent.nodes.verify import verify_answer_node

__all__ = [
    "classify_error_node",
    "build_context_pack_node",
    "run_tools_node",
    "react_plan_node",
    "react_observe_node",
    "react_reflect_node",
    "should_continue_react",
    "verify_answer_node",
    "repair_answer_node",
    "finalize_answer_node",
]
