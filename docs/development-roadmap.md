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

### 多视图融合 (Phase 0.5)

状态：已完成第一轮。

交付物：

- `_vote_hole_from_observations` 输出归一化置信度 / margin / cross-view agreement
- 遮挡感知动态重权 `_compute_occlusion_boost`：top 不可见时 side 自动接管
- pin 级 `evidence_source` (`top|left_front|right_front|fused|explicit_or_fallback|none`)
- `metadata.fusion.{per_view_contribution, per_view_top1, occlusion_boost}` 完整溯源
- 回归测试 `test_t5_8a_top_occluded_side_takes_over` /
  `test_t5_8b_fused_when_views_agree`

### Phase 6 — VLM 微观缺陷接入 + 白盒门控

状态：已完成第一轮 (增量包；vlm_service.py 保留向后兼容)。

交付物：

- `app/services/vlm/` 增量包：
  - `defect_types.MicroDefectType` 三类枚举 + 中英别名 + 每类 prompt
  - `suggest_defect_types()` 按 error_tags 反推 likely defect
  - `analyze_micro_defect()` 复用 `VlmService.explain_rc_pack`，
    在 prompt 前缀加缺陷指引，输出 schema 不变 + 加 `defect_type` 字段
- `app/agent/contracts.py` `VerificationReport` 加
  `needs_micro_inspection / suspected_defect_types`；新 `VlmFinding` 模型；
  `DiagnosticState.vlm_findings` 字段
- `app/agent/verification.py` 白盒门控：仅在 missing_component /
  incomplete_circuit / unknown family + 有 finding 或显式微观 tag 时开启
- `app/agent/nodes/vlm_explain.py` 新节点，门关闭时 no-op，门开时
  按建议缺陷类型逐一调用 VLM 并把简短结论 append 到 draft
- `app/agent/graph.py` `verify_answer` 后条件边扩成三向
  (`pass→finalize / fail→repair / passed+needs_vlm→vlm_explain`)
- `scripts/manual/tools/vlm/smoke_npu_vlm.py` DK-2500 NPU smoke
  (NOT in CI)，参数 `--device NPU/GPU/CPU`，输出 `vlm_explanation_v1` JSON
- 回归测试 `tests/test_vlm_provider_contract.py` 10 条覆盖：
  schema 一致性、defect prompt 完备、tag→defect 推断、白盒门控正反例、
  节点 no-op 路径、三向 routing

下一步：

- Phase 7+ 真机 NPU 验证：跑 smoke_npu_vlm.py 比对 template baseline
- 加微观缺陷标注 fixture 数据集，做端到端定量测试
- (stretch) 把 `vlm_service.py` 拆 provider 包 (base/template/openai/openvino)，
  本轮为最小风险保留单文件

### Phase 5 — 硬件遥测后端 + WebSocket

状态：已完成第一轮 (后端 + schema + 集成测试 + 协议文档；前端独立)。

交付物：

- `app/services/telemetry/{schema,service,samplers}.py` — 5Hz 异步采样、
  ring buffer、pub/sub fanout (慢消费者丢老不阻塞采样)
- `samplers/{cpu,igpu,npu}.py` — psutil + sysfs 优先、defensive、缺失时
  返回 `None` 不抛
- `app/api/v1/telemetry_ws.py` — `/ws/telemetry/system` WS 推流
  + `/api/v1/telemetry/latest` REST smoke
- `app/main.py` lifespan 启停；feature flag `TELEMETRY_ENABLED`
- 配置项 `TELEMETRY_ENABLED / TELEMETRY_HZ / TELEMETRY_RING_SECONDS`
- `docs/telemetry-protocol.md` — `telemetry_frame_v1` schema、curl/wscat
  示例、降级行为、性能预算
- `tests/test_telemetry_service.py` 7 条覆盖：
  start/stop、disabled 短路、subscriber 收帧、mark_stage 透传、
  macOS 降级、WS 端到端、REST 拉最新

下一步：

- DK-2500 现场 1h soak (RSS / CPU 自身占用) 数据沉淀
- NPU sysfs 路径在 DK-2500 验证后细化 `samplers/npu.py`
- pipeline orchestrator 加 `mark_stage` 钩子 (一行调用)

### Phase 4 — Diagnostic Agent ReAct + Self-Reflection

状态：已完成第一轮 (template provider, 无真实 LLM)。

交付物：

- `app/agent/contracts.py` 新增 `ToolCall / ReActStep / ReflectionResult`
  以及 `DiagnosticState.{react_trace, react_iterations, max_react_iterations,
  react_terminate_reason}` 字段
- `app/agent/llm/` 抽象层：`base.LLMProvider` ABC、`template_provider`
  (规则 emulator)、`openvino_genai_text` stub (Phase 7+ 接入)、`factory`
  自动 fallback
- `app/agent/nodes/` 节点拆分：classify / context / tools_node /
  react_plan / react_observe / react_reflect / verify / repair / finalize
- `app/agent/graph.py` 重构：3 节点 ReAct 子循环，硬上限 + planner 主动
  声明无更多工具时终止，verifier-pass 仅记录不短路终止 (避免漏掉
  fault_case / safety 等后续工具上下文)
- 配置项 `AGENT_LLM_PROVIDER`, `REACT_MAX_ITERATIONS`,
  `AGENT_LLM_OPENVINO_MODEL_DIR`, `AGENT_LLM_OPENVINO_DEVICE`
- 回归测试 `tests/test_agent_react_loop.py` 10 条覆盖：硬上限触发、
  无工具时早终止、planner 工具白名单、坏 provider 工具被丢弃、
  trace 形态、template provider plan/reflect 契约、sequential fallback

下一步：

- Phase 7+ 在 DK-2500 NPU validated 后接 `openvino_genai_text`
  小 LLM (Qwen2.5-1.5B-Int4 候选)，complex 路径下取代规则 plan/reflect

### 孔洞吸附质量 (Phase 0.6)

状态：已完成第一轮。

交付物：

- `BreadboardCalibrator.{board_point,frame_pixel}_to_logic_candidates_scored`
  返回 `(logic_loc, distance_px)`，路由穿过 unscored API 兼容 monkey-patch
- `representative_pitch_px()` 暴露代表 grid pitch
- `_snap_confidence_from_distance` (二次衰减 `1 - (d/pitch)^2`) 形成 [0,1] 置信度
- observation / pin 上 `snap_distance_px` / `snap_confidence` / `snap_normalized`
  / `pitch_px` / `candidate_distances_px` 全部展开
- `_snap_weight` 把吸附质量乘进多视图投票权重，最低 0.4 防止完全失声
- `low_snap_confidence` 进入 ambiguity reasons
- 回归测试: `test_t5_snap_confidence_high_when_pixel_on_grid` /
  `test_t5_snap_confidence_low_when_pixel_far_from_hole` /
  `test_t5_snap_low_confidence_loses_vote_to_well_snapped_view`

下一步：

- 用真实多视图样本跑 top-only vs multi-view A/B，作为 Phase 8 论文的消融数据
- 加 cross-view 几何一致性约束（同一 hole 的 top 与 side 投影必须落在同一 rail/segment）
- evidence overlay：把 `decisive_view_id` / `per_view_contribution` /
  `snap_distance_px` 渲染到前端
- 标定误差分解：在已检测网格上计算 keypoint→hole 的系统偏移分布
  (mean / p50 / p95)，沉淀为 `tests/manual/smoke/test_snap_quality_distribution.py`

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
- `safety_rule_lookup_tool`

下一步工具：

- `datasheet_lookup_tool`
- `heatmap_overlay_tool`
- `answer_template_tool`

约束：

- 工具输入使用 Pydantic args schema。
- 工具返回 `ToolResult`，包含 `summary` 和结构化 `payload`。
- 工具不做自然语言自由发挥，只返回可引用事实。

## Phase 4: AgentService Integration

状态：已完成第一版。

已新增：

```text
mode="diagnostic_agent"
```

最小流程：

```text
AgentService.submit()
-> build_runtime_evidence_from_classroom()
-> run_diagnostic_graph()
-> classify_error
-> build_context_pack
-> run deterministic tools
-> build template draft answer
-> verify / repair
-> AngntJobResult
```

验收标准：

- 不配置 LLM 时也能完成诊断回答。已完成。
- `danger` 风险回答必须包含安全提示。已完成。
- 回答必须包含 error code 或 evidence ref。已完成。
- 输出 `runtime_evidence / context_pack / tool_results / verification_report` evidence。已完成。
- template answer 生成逻辑已拆到 `app/agent/answering.py`。

下一步：

- 为 short_circuit / wiring_mismatch 各补更多 golden tests。
- 增加 agent metrics 字段。
- 将 graph node 级状态、耗时和工具调用数写入 metrics。

## Phase 5: LangGraph State Machine

状态：已完成第一版白盒壳子。

第一版图：

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

当前实现：

- `app/agent/graph.py`: LangGraph `DiagnosticState` 编排。
- `app/agent/tool_runner.py`: deterministic tools 统一执行入口。
- `AgentService mode="diagnostic_agent"` 已改为调用 `run_diagnostic_graph()`。
- `tests/test_agent_graph.py` 已覆盖 short circuit 和空 station 两条状态流转。

重要约束：

- `verify_answer` 是强制 Reflection Node。
- `classify_error` 第一版保持规则化。
- LLM 只在 `generate_draft` 或后续 `repair_answer` 参与。

下一步：

- 给图节点增加可观测 metrics：node latency、tool count、context facts count。
- 增加 verifier 失败后进入 `repair_answer` 的显式 golden case。
- 在 feature flag 后接入可选 LLM `generate_draft` node，默认仍走 template。

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

1. 为 short_circuit / wiring_mismatch 各补 2 个 agent golden tests。
2. 增加 `datasheet_lookup_tool` 的本地 fallback。
3. 增加 graph / agent metrics 字段。
4. 给 `repair_answer` 分支补失败-修复回归样例。
5. 写 ONNX export 和 parity 脚本。
6. 为论文实验准备 30-50 个固定案例。
