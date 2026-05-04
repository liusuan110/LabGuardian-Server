# Current Status

这份文档回答三个问题：

1. 当前主链路和可信边界是什么？
2. 哪些模块已经完成第一轮工程收口？
3. 下一步新增功能应该落在哪里？

## 当前结论

项目已经从“demo 可跑”推进到“结构化事实层 + PCM Agent 基建”阶段。

当前事实主链固定为：

```text
component detect
-> full-image pose pin detect
-> pin keypoint / observation
-> hole_id
-> electrical_node_id
-> electrical_net_id
-> netlist_v2
-> validator_report_v2
```

RAG、PCM Agent、VLM 和前端指导都应消费这条主链输出，不应绕开 validator
重新猜测元件、孔位、节点或网表。

## 已完成

### 1. 视觉与 pipeline 合同

- S1 默认主路径：`YOLO-Detect`
- S1.5 默认主路径：`YOLO-Pose full-image pin detect`
- S2 原生输出 `components[].pins[]`
- S3 / S4 已消费结构化 pins 输入
- `topology_input.py` 已将正式主输入收口到结构化 `components[].pins[]`
- OBB 仅作为历史权重兼容解析分支，不再是默认检测路线

阶段合同详见：

- [vision-stage-contracts.md](vision-stage-contracts.md)

### 2. 领域模型、网表与 validator

- `board_schema.py`
- `netlist_models.py`
- `export_netlist_v2()`
- `labguardian_ref_v4`
- `validator_report_v2`
- `error_code + suggested_action + evidence_refs`
- `circuit.py / validator.py / ic_models.py / polarity.py` 已围绕
  `ComponentInstance + pins[]` 工作

错误码体系详见：

- [validator-error-codes.md](validator-error-codes.md)

### 3. Edge 部署第一轮收口

- 模型根目录统一为 `LABGUARDIAN_MODEL_ROOT`
- 容器默认模型路径收口到 `/app/models/component/best.pt` 和
  `/app/models/pin/best.pt`
- `YOLO_IMGSZ`、`PIPELINE_HIGH_RES_IMGSZ`、`PipelineRequest.imgsz` 已统一为 `960`
- `run_pipeline()` 顶层输出 `runtime_metadata`
- 默认 63 行比赛板 schema 的电源轨分段已统一为 `1-31 / 32-63`

部署约定详见：

- [edge-deployment.md](edge-deployment.md)
- [board-schema-format.md](board-schema-format.md)

### 4. PCM Agent 白盒基建

`app/agent/` 已新增第一版不依赖大模型的 PCM 基建：

- `contracts.py`: Pydantic 契约
- `evidence.py`: `ClassroomState` station -> `RuntimeEvidence`
- `context_pack.py`: error code / error tag -> error family -> `ContextPack`
- `tools.py`: deterministic tool skeleton
- `tool_runner.py`: deterministic tools 统一执行入口
- `graph.py`: LangGraph `DiagnosticState` 白盒状态机壳子
- `answering.py`: template answer / repair answer 生成
- `verification.py`: Reflection / verifier node

当前已支持的 error family：

```text
COMPONENT_SHORTED_SAME_NET -> short_circuit
NODE_MISMATCH / HOLE_MISMATCH / FLOATING_PIN -> wiring_mismatch
POLARITY_REVERSED / POLARITY_UNKNOWN -> polarity_error
LED_SERIES_RESISTOR_MISSING -> missing_protection
COMPONENT_MISSING / COMPONENT_INSTANCE_MISSING -> missing_component
PIN_MISSING / MULTIPLE_DISCONNECTED_SUBGRAPHS -> incomplete_circuit
```

Agent 设计详见：

- [pcm-agent-architecture.md](pcm-agent-architecture.md)

### 5. 教学知识与 M-RAG / VLM 边界

- 一阶 RC 教学场景已结构化：`knowledge/teaching_scenes/first_order_rc_experiment.json`
- 一阶 RC fault cases 已结构化：`knowledge/fault_cases/rc/*.json`
- `TeachingKbService`、`MragService`、`VlmService` 已有最小边界
- VLM 只做解释，不负责识别、孔位定位或网表恢复

详见：

- [rag-teaching-kb-design.md](rag-teaching-kb-design.md)
- [knowledge/README.md](../knowledge/README.md)

### 6. 回归与冒烟

当前自动测试覆盖：

- `tests/pipeline/` 阶段合同与集成回归
- `tests/test_agent_pcm_contracts.py`
- `tests/test_agent_graph.py`
- `tests/test_agent_service_diagnostic.py`
- `tests/test_board_schema_default.py`
- `tests/test_pipeline_schema_defaults.py`
- `tests/test_mrag_service.py`
- `tests/test_teaching_kb_service.py`
- `tests/test_vlm_service.py`

当前手工 smoke：

- `tests/manual/smoke/test_board_schema_default.py`
- `tests/manual/smoke/test_reference_v4_roundtrip.py`
- `tests/manual/smoke/test_validator_error_codes.py`

## 当前仍需推进

### Pipeline / Vision

- 用真实图片继续 A/B 比较 `pose_roi_context_v12` 和 `train_demo/models`
- 增强 side-view observation，不再只停留在轻量占位
- 将真实多视图 pin 证据更完整地接到 S2 vote metadata
- 继续优化 `calibrator.py` 的实物板网格拟合策略
- 若比赛实物板和默认 schema 有差异，补正式 board schema JSON

### Validator / Evidence

- 将 `evidence_refs` 更直接绑定到 netlist / topology / component pin 对象
- 给前端和 Agent 输出更稳定的 evidence bundle
- 增加更多 validator fixture，覆盖复杂多元件和仪器测量类错误

### PCM Agent / LangGraph

- `RuntimeEvidence -> ContextPack -> deterministic tools -> template answer -> verifier`
  已接入 `AgentService mode="diagnostic_agent"`
- 当前白盒链路已包装为 `app/agent/graph.py` LangGraph state machine
- 将 `answer_verifier` 保持为强制 Reflection Node，而不是可选 tool
- 增加 graph / agent metrics 和 verifier repair 分支 golden tests
- 后续再接 OpenAI-compatible / OpenVINO `BaseChatModel` 适配器

### Edge / Paper

- 导出 S1 / S1.5 ONNX
- 建立 PT vs ONNX vs INT8 精度和速度对齐报告
- 增加 p50 / p90 latency、RSS、模型版本、context pack token size 等指标
- 准备 PCM vs no-PCM、tool vs no-tool、verifier vs no-verifier 消融实验

完整计划见：

- [development-roadmap.md](development-roadmap.md)

## 当前推荐阅读顺序

1. [README.md](../README.md)
2. [current-status.md](current-status.md)
3. [development-roadmap.md](development-roadmap.md)
4. [backend-architecture.md](backend-architecture.md)
5. [vision-stage-contracts.md](vision-stage-contracts.md)
6. [pcm-agent-architecture.md](pcm-agent-architecture.md)
7. [edge-deployment.md](edge-deployment.md)
8. [rag-teaching-kb-design.md](rag-teaching-kb-design.md)

## 协作原则

- 不再为旧 `pin1_logic / pin2_logic` 新增主流程兼容入口。
- 兼容旧字段时优先收口在 `topology_input.py`。
- S1 / S1.5 / S2 的 JSON 合同保持稳定，模型训练完成后优先替换推理内核。
- fallback 必须显式标记来源。
- Pipeline 只输出事实，不生成教学话术。
- Agent / RAG / VLM 只解释事实，不重判事实。
- 文档、fixture、合同测试必须跟代码一起更新。
