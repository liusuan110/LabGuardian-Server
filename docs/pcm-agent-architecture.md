# PCM Diagnostic Agent Architecture

本阶段已经落地不依赖大模型的 PCM 基建。它的目标不是替代视觉、拓扑或
validator，而是把确定性事实转成后续 LangGraph / ReAct 能稳定消费的最小上下文。

## Current Modules

```text
app/agent/contracts.py
  Pydantic 契约：RuntimeEvidence / ContextPack / DiagnosticState /
  VerificationReport 等。

app/agent/evidence.py
  从 ClassroomState station heartbeat 抽取 RuntimeEvidence。

app/agent/context_pack.py
  error_code / error_tag -> error_family，并构建 ContextPack。

app/agent/tools.py
  第一批白盒工具骨架：netlist_trace_tool、board_schema_lookup_tool、
  fault_case_lookup_tool、safety_rule_lookup_tool。

app/agent/tool_runner.py
  根据 ContextPack allowed_tools 顺序执行 deterministic tools。

app/agent/answering.py
  生成 template draft answer，并提供 verifier 失败后的规则化 repair。

app/agent/verification.py
  第一版 Reflection Node 规则校验器，强制检查 error_code、evidence_ref
  和 danger 安全提示。

app/agent/graph.py
  第一版 LangGraph 状态机壳子，不接 LLM，只跑白盒状态流转。
```

## Data Flow

```text
ClassroomState station
-> RuntimeEvidence
-> classify_error_family()
-> ContextPack
-> deterministic tools
-> draft answer
-> verify_draft_answer()
-> repair_answer?
-> final_answer
```

第一版完全规则化，便于测试和论文消融。**Phase 4 已经把单次 generate_draft
升级为 ReAct 循环 (Plan → Act → Observe → Reflect)**：

```text
START
-> classify_error
-> build_context_pack
-> react_plan ──→ react_observe ──→ react_reflect ─┬─(continue)→ react_plan
                                                     └─(finalize)→ verify_answer
-> verify_answer
-> repair_answer?
-> final_answer
END
```

ReAct 节点拆分到 `app/agent/nodes/`，LLM provider 抽到
`app/agent/llm/`：

- `react_plan_node` 用 `LLMProvider.plan()` 决定下一个工具，**强制限制在
  `ContextPack.allowed_tools` 白名单内**（防幻觉）
- `react_observe_node` 调度该工具并把 `ToolResult` 摘要写回 `ReActStep.observation`
- `react_reflect_node` 用 `LLMProvider.reflect()`（默认 = `verify_draft_answer`
  规则评分）形成 `ReflectionResult`，并据此决定循环控制
- 终止条件：planner 返回 `tool_call=None`（已没有更多工具可用）或迭代到
  `max_react_iterations` 上限（默认 4）
- 每个迭代写入 `DiagnosticState.react_trace` 一个 `ReActStep`
  + `graph_metrics` 一条 `react_{plan|observe|reflect}_{i}` 度量
- Phase 4 默认 `AGENT_LLM_PROVIDER=template` 走确定性模板 provider；
  `openvino_genai_text` 已留 stub，等 Phase 7+ DK-2500 NPU 验证后接入

## Error Routing

当前已覆盖的 error family：

```text
COMPONENT_SHORTED_SAME_NET -> short_circuit
NODE_MISMATCH / HOLE_MISMATCH / FLOATING_PIN -> wiring_mismatch
POLARITY_REVERSED / POLARITY_UNKNOWN -> polarity_error
LED_SERIES_RESISTOR_MISSING -> missing_protection
COMPONENT_MISSING / COMPONENT_INSTANCE_MISSING -> missing_component
PIN_MISSING / MULTIPLE_DISCONNECTED_SUBGRAPHS -> incomplete_circuit
```

## Verifier Rule

`verify_draft_answer()` 当前强制检查：

- 若存在 `error_codes`，回答必须包含至少一个当前错误码。
- 若存在 `evidence_refs`，回答必须引用至少一个 ref/component/pin/hole。
- 若 `risk_level=danger`，回答必须包含断电、电源或短路复查提示。

这对应后续论文中的 Reflection / Critic Node，而不是由大模型自由选择调用的 tool。

## Current Integration

当前已接入 `AgentService mode="diagnostic_agent"`：

```text
AgentService.submit()
-> build_runtime_evidence_from_classroom()
-> run_diagnostic_graph()
-> classify_error
-> build_context_pack
-> run_tools
-> generate_draft
-> verify_answer / repair_answer
-> AngntJobResult
```

LangGraph 已经只承担编排职责，不重新判断事实。LLM 适配器和 OpenVINO chat model
后续应继续保持可选，不影响离线 template 诊断闭环。

## Next Step

1. ✅ 增加 graph metrics：节点耗时、context facts count、tool call count。
2. ✅ 为 verifier 失败后进入 `repair_answer` 的分支补 golden tests。
3. ✅ Phase 4 — 把 generate_draft 升级为 ReAct + Self-Reflection 循环（template provider）。
4. ✅ Phase 6 — 在 `verify_answer` 后插入 `vlm_explain_node`，仅在
   `verification_report.needs_micro_inspection=True` 时触发（白盒优先）。
   微观缺陷类型见 `app/services/vlm/defect_types.py`。
   DK-2500 NPU smoke 见 `scripts/manual/tools/vlm/smoke_npu_vlm.py`。
5. Phase 7+ — 接入 `openvino_genai_text` 真实 LLM provider，DK-2500 NPU 验证后启用。
6. 增加 `datasheet_lookup_tool` 本地 fallback，再进入外部文档检索。

## Roadmap Link

完整工程计划和论文实验路线见：

- [development-roadmap.md](development-roadmap.md)
