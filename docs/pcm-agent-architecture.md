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
  fault_case_lookup_tool。

app/agent/verification.py
  第一版 Reflection Node 规则校验器，强制检查 error_code、evidence_ref
  和 danger 安全提示。
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
```

第一版完全规则化，便于测试和论文消融。后续 LangGraph 只需要把这些函数包装成
node：

```text
START
-> build_runtime_evidence
-> classify_error
-> build_context_pack
-> tool_loop / answer_generation
-> verify_answer
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

## Next Step

下一阶段再接入 `AgentService`，建议新增 `mode="diagnostic_agent"`：

```text
AgentService.submit()
-> build_runtime_evidence_from_classroom()
-> build_context_pack()
-> deterministic tool calls
-> template draft answer
-> verify_draft_answer()
-> AngntJobResult
```

LangGraph 和 LLM 适配器应在这条白盒链路稳定后再接入。

## Roadmap Link

完整工程计划和论文实验路线见：

- [development-roadmap.md](development-roadmap.md)
