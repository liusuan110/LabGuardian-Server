# 文档索引

这里是 LabGuardian-Server 的文档主索引。文档按"先看契约、再看主链路架构、再看视觉协议、最后专项设计"的顺序阅读。

> **写作规范**：所有新建文档用**中文**，技术名词/API/路径/代码标识符保留英文。代码内 docstring 仍用英文以与上游生态一致。详见本页 [§写作规范](#写作规范)。

## 必读契约

- [retrieval-contract.md](retrieval-contract.md) **🔒 权威**  
  检索契约（训练 ≡ 部署）。任何在 `app/agent/**` / `app/services/rag_service.py` / `app/services/agent_service.py` / `scripts/distill/**` 下新增检索调用点的人**必读**。包含合法/禁用源清单、训练-部署不变量、Intent 路由子路由、蒸馏 gold set 锚点、硬门槛阈值、完整变更日志（WP-0..WP-1 v6）。

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

- [rag-teaching-kb-design.md](rag-teaching-kb-design.md) ⚠️ **部分超越**  
  Phase 1–3 RAG/PCM 演进史。头部 banner 标注了被 WP-0/WP-1 取代的具体章节（如旧 Chroma 三段回退、RC 隐含默认）。仍然有效的部分：4 通道架构、`mrag_pack` 版本契约、`OpenVINOEmbeddingBackend` 设计、ingest 流水线。最新检索契约以 [retrieval-contract.md](retrieval-contract.md) 为准。

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
- **改检索行为（agent 主链路 / 蒸馏入口 / KB schema）时，必须同步更新
  `retrieval-contract.md` 的变更日志**，并在 PR 中由 reviewer 检查这一条。
- 改部署路径、模型路径或 board schema 时，同步更新 `edge-deployment.md` 和
  `board-schema-format.md`。

---

## 写作规范

### 一、语言

- **设计稿 / 契约 / 决策记录 / 变更日志**：中文。
- **代码内 docstring / 单元测试断言文案 / 日志输出**：英文（与上游 LangChain / OpenVINO / Pydantic 生态一致）。
- **混排**：技术名词、API 名、文件路径、代码标识符、第三方产品名保持英文不翻译（如 `RagService`、`fault_case_lookup_tool`、`exp_ua741_inverting_amplifier`、`OpenVINO`、`bge-m3`）。

### 二、文件命名

- 用短横线连接英文小写词：`retrieval-contract.md`、`rag-teaching-kb-design.md`。
- 不用拼音、不用中文文件名（兼容跨平台 git 工具链）。

### 三、文档骨架

每篇新文档需有：

```markdown
# <一级中文标题>

> 可选 banner —— 标注废弃 / 部分超越 / 高优先级警告

**状态**：<草稿 | 现行 | 部分超越 | 已废弃>，自 <WP / 日期> 起。
**读者**：<目标读者 — 例如"在 X 下新增 Y 的开发者">

## 1. 背景 / 目标
## 2. 主体内容（按场景分节）
## N. 变更日志（表格形式）
```

变更日志表格列：`日期 | WP | 改动`。日期格式 ISO `YYYY-MM-DD`。

### 四、外部引用

- 跨文档引用用相对路径：`[retrieval-contract.md](./retrieval-contract.md)`。
- 跨文档引用具体章节用 anchor：`[§3 训练 ≡ 部署不变量](./retrieval-contract.md#3-训练--部署不变量)`。
- 引用代码文件用相对仓库根：`app/services/scene_resolver.py:67`。

### 五、术语对齐

| 中文 | 英文 / 代码标识 |
|---|---|
| 检索契约 | retrieval contract |
| 蒸馏 | distillation |
| 教学场景 | teaching scene / `teaching_scene` |
| 故障案例 | fault case / `fault_case` |
| 拓扑标签 | topology label |
| 主链路 | main path / production agent path |
| Fail-closed | fail-closed（不译） |
| Fail-open | fail-open（不译） |
| 软地板 / 硬上限 | soft floor / hard ceiling |
| 上下文包 | context pack / `ContextPack` |
| 工具 | tool |
| 工位 | station |

新术语首次出现时附英文括注，例如"**检索契约（retrieval contract）**"。

### 六、变更日志写法

每条变更条目至少给出：

1. **发生了什么**（"删除了 X" / "新增 Y 字段"）
2. **为什么**（"否则 Z 漏洞会导致 W"）
3. **修后契约**（"现在 A 必须 B"）

反例（禁止）：
> "修了一些 bug。"

正例：
> "把 `KbService` 从 agent 主链路移除，否则蒸馏数据会混入 RC 内容污染非 RC 拓扑。现在 agent 主链路只读 `teaching_scene` / `fault_case` / `datasheet_v2` / `circuit_kb` 4 通道。"

### 七、Banner 模板

文档头部 banner 用 blockquote + emoji：

```markdown
> **⚠️ 历史设计稿 — 部分章节已超越 (since WP-0, 2026-05-24)**
>
> 最新契约见 [retrieval-contract.md](./retrieval-contract.md)。两者冲突时
> 以新文档为准。已超越的具体章节：
> - <章节 1>：<被什么取代>
> - <章节 2>：<被什么取代>
```

可用 emoji：

| Emoji | 语义 |
|---|---|
| `⚠️` | 部分超越 / 警告 |
| `🚫` | 已废弃 / 禁止 |
| `🔒` | 权威契约 / 必读 |
| `✅` | 当前最佳实践 / 现行规范 |
| `📋` | 操作 checklist |
| `🔬` | 实验性 / 待验证 |

### 八、禁止 vapor 引用

不要写"详见 `scripts/distill/precheck_retrieval.py`"如果该脚本尚未实现。要写"WP-3 计划落地，见任务 #113"。文档撒谎比文档不全更有害——它让 reviewer 误判契约完整度。

### 九、历史文档不删

过去的设计稿即使过时也保留，加 banner 指向新契约。删除会让 git blame 链断裂、PR 评审失去比对参考、新人无法理解"为什么当初是那样设计"。
