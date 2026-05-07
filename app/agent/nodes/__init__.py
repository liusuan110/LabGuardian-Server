"""Per-node modules for the diagnostic LangGraph.

This package contains one module per graph node so each node can be unit-tested
in isolation and so Phase 6 (`vlm_explain`) can plug in without touching the
other nodes.

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
from app.agent.nodes.vlm_explain import vlm_explain_node

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
    "vlm_explain_node",
]
