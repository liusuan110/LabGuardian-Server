# Development Roadmap

这份文档是后续工程和论文实验的总计划。主线是：

```text
结构化事实层
-> RuntimeEvidence
-> PCM ContextPack
-> deterministic tools
-> LangGraph state machine
-> optional LLM/VLM
-> edge benchmark / paper ablation
```

## Guiding Principle

LabGuardian 不把事实判断交给大模型。

- 事实层：视觉、几何、board schema、topology、validator。
- 编排层：PCM / LangGraph 按需推送上下文和工具。
- 解释层：RAG / LLM / VLM 只生成可引用、可校验的回答。

## Phase 0: Stability Baseline

状态：已完成第一轮。

交付物：

- 统一模型路径和 `LABGUARDIAN_MODEL_ROOT`
- 统一 `imgsz=960`
- 修正默认 board schema 电源轨分段
- `runtime_metadata`
- 全量测试基线

后续只做小修，不再把部署路径散回 `train_demo`。

## Phase 1: Evidence Contract

状态：已完成第一版。

交付物：

- `app/agent/contracts.py`
- `RuntimeEvidence`
- `EvidenceRef`
- `DiagnosticFinding`
- `ContextPack`
- `DiagnosticState`
- `VerificationReport`

下一步：

- 从 `PipelineResult` 和 `ClassroomState` 同时抽取完整 `netlist_v2`
- 增强 `evidence_refs`，让 ref 能直接定位 component / pin / net / validator item

## Phase 2: PCM Context Routing

状态：已完成第一版规则路由。

当前覆盖：

- short circuit
- wiring mismatch
- polarity error
- missing protection
- missing component
- incomplete circuit
- measurement error

下一步：

- 扩展仪器测量类 error family
- 将 context pack size、pushed facts count、allowed tool count 写入后续指标
- 为每个 error family 补最小 golden case

## Phase 3: Deterministic Tools

状态：已有骨架。

当前工具：

- `netlist_trace_tool`
- `board_schema_lookup_tool`
- `fault_case_lookup_tool`

下一步工具：

- `datasheet_lookup_tool`
- `safety_rule_lookup_tool`
- `heatmap_overlay_tool`
- `answer_template_tool`

约束：

- 工具输入使用 Pydantic args schema。
- 工具返回 `ToolResult`，包含 `summary` 和结构化 `payload`。
- 工具不做自然语言自由发挥，只返回可引用事实。

## Phase 4: AgentService Integration

状态：未接入。

建议新增：

```text
mode="diagnostic_agent"
```

最小流程：

```text
AgentService.submit()
-> build_runtime_evidence_from_classroom()
-> build_context_pack()
-> call deterministic tools
-> build template draft answer
-> verify_draft_answer()
-> AngntJobResult
```

验收标准：

- 不配置 LLM 时也能完成诊断回答。
- `danger` 风险回答必须包含安全提示。
- 回答必须包含 error code 或 evidence ref。

## Phase 5: LangGraph State Machine

状态：计划中。

第一版图：

```text
START
-> build_runtime_evidence
-> classify_error
-> build_context_pack
-> select_tools
-> run_tools
-> generate_draft
-> verify_answer
-> final_answer
END
```

重要约束：

- `verify_answer` 是强制 Reflection Node。
- `classify_error` 第一版保持规则化。
- LLM 只在 `generate_draft` 或后续 `repair_answer` 参与。

## Phase 6: Model Adapter

状态：计划中。

接入顺序：

1. fake/template model
2. OpenAI-compatible local server
3. OpenVINO `BaseChatModel` adapter

OpenVINO adapter 建议放在：

```text
app/llm/openvino_chat_model.py
```

它应复用现有 `VlmService` 的 OpenVINO 加载经验，但暴露为 LangChain chat model。

## Phase 7: Edge Optimization

状态：计划中。

优先任务：

- S1 / S1.5 导出 ONNX
- PT vs ONNX 输出一致性测试
- INT8 视觉量化
- stage-level latency / RSS 采集
- edge-cpu / edge-openvino 镜像拆分

记录指标：

- total latency
- stage latency
- p50 / p90
- peak RSS
- model path / model version
- board schema id
- context pack pushed facts count
- tool call count

## Phase 8: Paper Experiments

目标：证明结构化事实层 + PCM Agent 比普通 RAG 或端到端 VLM 更稳、更省上下文、更可解释。

消融组：

- no-RAG vs RAG
- no-PCM vs PCM
- no-tool vs deterministic tools
- no-verifier vs Reflection Node
- template vs LLM
- no-VLM vs optional VLM
- PT vs ONNX vs INT8

质量指标：

- error_code exact match
- risk_level accuracy
- citation coverage
- answer faithfulness
- answer relevance
- context precision
- prompt/context size
- latency / RSS / index size

后续可引入 RAGAS 或自定义 LLM-as-a-Judge，但必须保留白盒规则指标作为主结果。

## Near-Term Checklist

1. 接入 `AgentService mode="diagnostic_agent"`。
2. 为 short_circuit / wiring_mismatch 各补 2 个 agent golden tests。
3. 增加 `datasheet_lookup_tool` 的本地 fallback。
4. 设计 LangGraph state object 与 node wrappers。
5. 增加 agent metrics 字段。
6. 写 ONNX export 和 parity 脚本。
7. 为论文实验准备 30-50 个固定案例。
