# Documentation Index

这里是 LabGuardian-Server 文档的主索引。文档按“先比较主链路、再视觉协议、再专项设计”的顺序阅读。

## Start Here

1. [comparison-architecture.md](comparison-architecture.md)  
   当前 reference DSL、netlist normalization、graph compare、role inference 和 S4/API 接入架构。

2. [development-roadmap.md](development-roadmap.md)  
   后续工程开发、PCM Agent、LangGraph、edge benchmark 和论文实验计划。

## Pipeline / Vision

- [vision-stage-contracts.md](vision-stage-contracts.md)  
  S1 / S1.5 / S2 的阶段协议。

- [vision-model-inventory.md](vision-model-inventory.md)  
  当前视觉模型资产、候选权重和推荐默认模型。

## Domain / Compare

- [board-schema-format.md](board-schema-format.md)  
  `hole_id -> electrical_node_id` 的 board schema 格式和默认比赛板分段。

- [validator-error-codes.md](validator-error-codes.md)  
  `validator_report_v2` 报告协议、错误码、风险语义和 fixture 维护规则。

## Agent / RAG / VLM

- [pcm-agent-architecture.md](pcm-agent-architecture.md)  
  `RuntimeEvidence -> ContextPack -> deterministic tools -> LangGraph verifier` 的 PCM Agent 设计。

- [rag-teaching-kb-design.md](rag-teaching-kb-design.md)  
  一阶 RC 教学知识库、M-RAG pack、VLM 解释边界和后续检索计划。

## Edge / Paper

- [edge-deployment.md](edge-deployment.md)  
  板端模型路径、默认推理尺寸、runtime metadata 和 edge 后续优化方向。

- [telemetry-protocol.md](telemetry-protocol.md)  
  DK-2500 硬件遥测 WebSocket / REST 协议、`telemetry_frame_v1` schema、配置与降级行为。

## Maintenance Rule

- 改 pipeline 协议时，同步更新 `vision-stage-contracts.md` 和 `tests/pipeline/`。
- 改比较架构、reference DSL 或 graph matching 行为时，同步更新
  `comparison-architecture.md` 和 `tests/domain/test_graph_compare*.py`。
- 改 report error code 时，同步更新 `validator-error-codes.md` 和
  `tests/fixtures/validator_error_codes/`。
- 改 Agent / RAG 行为时，同步更新 `pcm-agent-architecture.md`、
  `rag-teaching-kb-design.md` 和 `tests/test_agent_pcm_contracts.py`。
- 改部署路径、模型路径或 board schema 时，同步更新 `edge-deployment.md` 和
  `board-schema-format.md`。
