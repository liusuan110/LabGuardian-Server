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

第一版完全规则化，便于测试和论文消融。当前 LangGraph 壳子已经落地为：

```text
START
-> classify_error
-> build_context_pack
-> run_tools
-> generate_draft
-> verify_answer
-> repair_answer?
-> final_answer
END
```

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

1. 增加 graph metrics：节点耗时、context facts count、tool call count。
2. 为 verifier 失败后进入 `repair_answer` 的分支补 golden tests。
3. 在 feature flag 后接入可选 LLM `generate_draft` node。
4. 增加 `datasheet_lookup_tool` 本地 fallback，再进入外部文档检索。

## Roadmap Link

完整工程计划和论文实验路线见：

- [development-roadmap.md](development-roadmap.md)
