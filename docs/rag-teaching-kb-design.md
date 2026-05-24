# RAG / PCM 教学知识库设计

> **⚠️ 历史设计稿 — 部分章节已超越 (since WP-0, 2026-05-24)**
>
> 本文档记录早期 RAG/PCM 设计与 Phase 1–3 的演进历史。**最新可执行契约以
> [`docs/retrieval-contract.md`](./retrieval-contract.md) 为准**。两者冲突时
> 以 retrieval-contract 为准。已超越的具体章节：
>
> - **旧 Chroma/PDF KB 三段回退**（`local_datasheet_v2 → kb_retrieval → local_fallback`
>   原描述）：`kb_retrieval` 中间层在 WP-0 已下线，agent 主链路不再可达。
>   现行流程仅 `local_datasheet_v2 → local_fallback`。
> - **`scene_id="exp_first_order_rc"` 隐含默认**：在 WP-1 全链路移除，
>   未解析的 topology 一律 fail-open，不再回落 RC。
>
> 仍然适用的部分：`teaching_scene` / `fault_case` / `circuit_kb` / `datasheet_v2`
> 四通道架构、`mrag_pack` 版本契约、`OpenVINOEmbeddingBackend` 设计、ingest
> 流水线等。修改本文档时请同步勾选/作废相应章节。

本项目里的 RAG 应定位为"实验调试助教"，而不是通用电子百科问答。

当前架构已经从单纯 RAG 继续推进到 PCM：

```text
validator_report_v2 / netlist_v2
-> RuntimeEvidence
-> ContextPack
-> teaching / fault / KB retrieval
-> verifier
```

RAG 的职责是补充教学知识和引用来源；PCM 的职责是按错误类型选择最小上下文和工具。

阶段 1 和阶段 2 只围绕一阶 RC 实验展开，来源资料为：

- `/Users/liusuan/Downloads/一阶RC电路的实验研究.pdf`

结构化知识放在：

- `knowledge/teaching_scenes/first_order_rc_experiment.json`
- `knowledge/fault_cases/rc/*.json`

## 为什么先按教学场景组织

原始 PDF 检索适合做引用，但它不知道学生正在搭哪一个实验。LabGuardian 已经有结构化电路事实：

```text
image -> components[].pins[] -> netlist_v2 -> validator_report_v2
```

因此检索和上下文推送顺序应是：

1. 当前 `RuntimeEvidence`。
2. 当前 `ContextPack` 中的 error family、allowed tools 和 pushed facts。
3. 一阶 RC 教学场景。
4. 一阶 RC 错误知识单元。
5. 面包板和仪器测量规则。
6. 原始 PDF 或 datasheet 片段。

## 知识类型

```text
teaching_scene
  实验级上下文：实验目标、正确电路、测量要求、常见错误。

fault_case
  错误解释和修复步骤，可映射 validator_report_v2 错误码。

measurement_rule
  仪器测量规则，例如示波器地线、探头 X1/X10、直流耦合、触发源。

component_knowledge
  电阻、电容等元件知识。阶段 2 只记录 RC 相关电阻/电容知识。

board_rule
  面包板连通规则、节点解释、布局约束。

datasheet_chunk
  原始 PDF 或器件手册片段，后续进入向量库。
```

## 当前一阶 RC 覆盖范围

### 教学场景

主要覆盖：

- 时间常数 `tau = RC`。
- 方波输入下的微分电路和积分电路现象。
- 示波器双通道测量、探头倍率、触发源、直流耦合。
- 信号源和示波器黑夹子必须接参考地。

### 错误知识单元

阶段 2 已建立：

- `rc_scope_ground_not_reference_ground`
- `rc_wrong_output_node_for_integrator`
- `rc_probe_x10_not_accounted`
- `rc_wrong_signal_offset`
- `rc_capacitor_value_mismatch`

如果出现 `NODE_MISMATCH` 或 `HOLE_MISMATCH`，RAG 应优先解释输出节点或面包板节点错误；如果用户问波形异常，应按参考地、输出节点、RC 数值、探头倍率、输入偏置的顺序排查。

## 回答格式

面向学生端回答：

```text
结论：
依据：
可能现象：
修改步骤：
引用：
```

面向教师端回答：

```text
工位状态：
诊断证据：
教学解释：
建议干预：
引用：
```

回答中应区分：

- 来自 `validator_report_v2` 的事实。
- 来自 `teaching_scene` 的实验规则。
- 来自 `fault_case` 的图文纠错知识。
- 来自 PDF/datasheet 的原文引用。

## 当前已实现

- `TeachingKbService`: 规则化教学场景与 fault case 检索。
- `MragService`: 生成 `mrag_pack_v1`。
- `VlmService`: template / openai-compatible / openvino-genai 解释边界。
- `app/agent/evidence.py`: station -> `RuntimeEvidence`。
- `app/agent/context_pack.py`: error code / tag -> `ContextPack`。
- `app/agent/tools.py`: fault case、board schema、netlist trace 工具骨架。
- `app/agent/verification.py`: Reflection / verifier node 第一版。

## M-RAG 知识包

`app/services/mrag_service.py` 不负责读取原始文件，而是把
`TeachingKbService` 检索到的场景和错误知识单元整理成稳定结构：

```json
{
  "pack_version": "mrag_pack_v1",
  "scene": {
    "scene_id": "exp_first_order_rc",
    "scene_name": "一阶 RC 电路的实验研究"
  },
  "query": "为什么积分电路输出波形不对",
  "error_tags": ["wrong_node_connection"],
  "structured_context": {
    "error_codes": ["NODE_MISMATCH"],
    "diagnostics": [],
    "risk_level": "warning",
    "circuit_snapshot": ""
  },
  "fault_cases": [],
  "references": {
    "texts": [],
    "images": [],
    "waveforms": [],
    "schematics": []
  },
  "fix_steps": []
}
```

这样 VLM 和前端都只需要消费一个知识包：

- `structured_context` 来自视觉/网表/规则层。
- `fault_cases` 来自本地教学知识库。
- `references.images / references.waveforms / references.schematics` 是后续双图对比的参考资产路径。
- `fix_steps` 可直接给前端展示。

当前仍不让 VLM 参与识别，也不让大模型重判事实。

## 轻量 VLM 解释边界

`app/services/vlm_service.py` 只负责解释，不负责检测、孔位定位、网表恢复。

输入边界：

```json
{
  "mrag_pack": "MragService 生成的 mrag_pack_v1",
  "current_image": "当前面包板图或示波器图，可为空",
  "reference_image": "参考图或标准波形图，可为空",
  "user_query": "学生问题"
}
```

输出边界：

```json
{
  "result_version": "vlm_explanation_v1",
  "provider": "template | openai_compatible | template_fallback",
  "status": "completed | vlm_call_failed",
  "inputs": {},
  "prompt": "",
  "answer": {
    "conclusion": "",
    "evidence": "",
    "fix_steps": []
  }
}
```

默认配置是 `VLM_PROVIDER=template`，不调用真实模型，便于离线开发和测试。

板端轻量 VLM 服务就绪后，可在 `.env` 中配置：

```text
VLM_PROVIDER=openai_compatible
VLM_BASE_URL=http://127.0.0.1:8001/v1
VLM_MODEL=local-vlm
VLM_TIMEOUT_S=30
```

该接口按 OpenAI-compatible `/chat/completions` 形式发送请求。若模型调用失败，会自动回退到模板解释，保证主链路不中断。

### OpenVINO GenAI 本地接入

Qwen2.5-VL-3B-Instruct 的 OpenVINO INT4 版本可以直接走本地 `openvino_genai` provider。

板端需要先准备 Python 依赖：

```bash
uv sync --no-install-project --extra edge
```

`.env` 示例：

```text
VLM_PROVIDER=openvino_genai
VLM_OPENVINO_MODEL_DIR=/models/Qwen2.5-VL-3B-Instruct-ov-int4
VLM_OPENVINO_DEVICE=CPU
VLM_OPENVINO_CACHE_DIR=artifacts/openvino_cache
VLM_MAX_NEW_TOKENS=256
```

当前实现使用 `openvino_genai.VLMPipeline(model_dir, device)`，并把本地图片或 data URL 转为 `openvino.Tensor`。如果模型目录不存在、依赖未安装或推理失败，会自动回退到模板解释。

板端 smoke test：

```bash
.venv/bin/python scripts/manual/tools/vlm/smoke_openvino_vlm.py \
  --model-dir /models/Qwen2.5-VL-3B-Instruct-ov-int4 \
  --device CPU \
  --image /path/to/current_scope_or_board.png \
  --reference-image /path/to/reference_waveform.png \
  --query "为什么我的波形不对？"
```

如果输出 `provider=template_fallback` 和 `status=openvino_call_failed`，说明服务层已兜底，但模型目录、OpenVINO 依赖或设备配置仍需检查。

## Datasheet KB v2（Phase 1 已落地）

Datasheet 检索从“PDF 全文 + 本地规则 fallback”升级为“多模态结构化 chunk + 统一引用契约”，目标是为板上离线部署铺路：

- **Schema**：在 `app/schemas/kb.py` 新增 `DatasheetDocument` / `DatasheetChunk` / `RetrievedChunk`。chunk 支持 `text / table / figure / schematic / waveform` 五种 modality，并强制 `chunk_id` 字段。
- **本地知识库**：`knowledge/datasheets/*.json` 存放手写或离线 parse 出来的结构化产物；图/原理图资产放 `knowledge/datasheets/assets/<doc_id>/`。当前样例:
  - `ne555.json` — text + figure + table 三种 modality
  - `lm324.json` — text + schematic
  - `passive_capacitor_polarity.json` — 纯 text，演示 `source_path=null` 的合法形态
- **检索服务**：`app/services/datasheet_kb_service.py` 的 `DatasheetKbService`，纯关键词 + part-number 别名打分，离线、确定性、可单测。`EmbeddingBackend` 抽象 (`app/services/embedding_backend.py`) 默认 `NullEmbeddingBackend`，Phase 3 可换成 OpenVINO INT8 实现，但 chunk 向量在开发机预计算成 `.npz`，板上仅做 query 编码。
- **工具入口**：`datasheet_lookup_tool` 当前为两级回退——`local_datasheet_v2` → `local_fallback`（带 `rule_id`）。  
  *⚠️ 原设计的中间层 `kb_retrieval`（旧 Chroma/PDF）已于 WP-0（2026-05-24）下线*，理由见 [`retrieval-contract.md`](./retrieval-contract.md) §2。`KbService` 类仍保留供 admin PDF 上传 API，但 agent 主链路、`RagService.build_context`、蒸馏脚本均不可达。
- **mrag_pack 版本契约**：`MragService.build_pack(..., retrieved=...)` 在带 `datasheet_chunks / figures / tables` 时输出 `pack_version="mrag_pack_v2"` + 顶层 `retrieved`；未传或为空仍输出 v1，旧 VLM/前端无感知。
- **Verifier 强约束**：`verify_draft_answer` 新增 `tool_results` 可选入参；当 `datasheet_lookup_tool` 命中 chunk 路径，回答必须引用至少一个 `chunk_id`；走 `local_fallback` 时必须引用至少一个 `rule_id`。

### Build-time ingest 流水线（Phase 2 已落地）

`scripts/ingest_datasheets.py` 把 PDF 解析成 `DatasheetDocument` JSON,**完全离线**,不上板。三个 backend 按优先级决起来用:

1. **`mineru`** — 完整多模态 parse（text / table / figure / formula / layout），需 `pip install '.[ingest]'` 并下载模型权重。
2. **`magic_pdf`** — 同一项目早期 API，兼容适配,代码已写好。
3. **`pypdf`** — 仅文本 fallback，runtime 已装,无外网下载、无 GPU。今天用它把 `/knowledge_base/` 三个 PDF 跑出真实文本 chunks。

CLI 示例:
```bash
.venv/bin/python scripts/ingest_datasheets.py \
  --pdf knowledge_base/C695838_555定时器-计时器_NE555DR_规格书_WJ1799212.PDF \
  --document-id ne555 \
  --title "NE555 单路定时器" \
  --part-numbers NE555 NE555DR 555 \
  --backend pypdf \
  --out knowledge/datasheets/ne555.json
```

合并语义(默认 `--overwrite` 关):
- 抽取出的 text chunks **覆盖**旧 text chunks。
- figure / table / schematic / waveform 等非 text chunks **保留**——Phase 1 手写的占位资产一直在,直到 MinerU 真跑出来才被替换。
- document-level metadata（title / part_numbers / source_path）优先用 CLI 参数,空则 fall back 到已存在 JSON。

边界条件:
- pypdf 对子集 CID / glyph-encoded PDF 无能为力(例如 SN74LS74A 前 6 页是 `/C0083/C0078/...`)。`PypdfBackend` 检测到这种 garbage 会跳过这些页;全部 garbage 时直接抛错并提示用 `--backend mineru`。
- 板上 runtime 永远不装 MinerU/Magic-PDF/PaddleOCR/PDF 解析库。`pyproject.toml` 的 `ingest` extras 只在开发机/CI 装,镜像打包必须排除。
- `document_id` 是稳定主键,重新解析时同名覆盖即可,板上代码契约不变。

### 混合检索（Phase 3 已落地）

板上 retrieval 现在支持"关键词 + 余弦"融合,模型权重和编码成本严格分两侧:

- **板上**:只用 `openvino` runtime(与 `vlm_service` 同源,无新加速栈)加载一个 INT8 IR 嵌入模型(推荐 `bge-small-zh-v1.5`, ~50–100MB)。`OpenVINOEmbeddingBackend` 懒加载,模型目录缺失/损坏时静默回退到关键词,板子永远不因缺模型而崩。**chunk 向量永远不在板上重新计算**——只编码用户 query(每次 1 句话)。
- **开发机/CI**:跑 `scripts/build_datasheet_embeddings.py` 一次,把每个 `DatasheetDocument` 的所有 chunk 文本批量编码,写出 `knowledge/datasheets/embeddings/<document_id>.npz`(两列:`chunk_ids` + L2-normalized `vectors`)。这些 `.npz` 跟 JSON / assets 一起进固件。

融合公式:把关键词得分 `k` 经 `k/(1+k)` 压到 [0,1),与 cosine `c ∈ [0,1]` 加权:
```
fused = (1 - w) * k/(1+k) + w * c           # w = DATASHEET_EMBEDDING_FUSION_WEIGHT, 默认 0.55
```
不在意 part 的 doc 仍然被乘以 0.35 derank,跟 Phase 2 行为一致;cosine 负值钉到 0。

启用步骤(开发机):
1. 装依赖:`pip install -e '.[embedding-build]'`(只在开发机,板子不需要)
2. 把 HF Sentence-Transformers 模型转 OpenVINO INT8 IR:
   ```bash
   optimum-cli export openvino \
     --model BAAI/bge-small-zh-v1.5 \
     --weight-format int8 \
     models/bge-small-zh-v1.5-int8-ov
   ```
   产物含 `openvino_model.xml` + `openvino_model.bin` + `tokenizer.json`,~50MB。
3. 预计算 chunk 向量:
   ```bash
   .venv/bin/python scripts/build_datasheet_embeddings.py \
     --backend openvino \
     --model-dir models/bge-small-zh-v1.5-int8-ov \
     --device CPU
   ```
   产物落到 `knowledge/datasheets/embeddings/<doc_id>.npz`,提交进仓库。

启用步骤(板上):
1. 板上 runtime 已带 `openvino`(VLM 在用),不需要装 `embedding-build` extras。
2. `.env` 配置:
   ```
   DATASHEET_EMBEDDING_BACKEND=openvino
   DATASHEET_EMBEDDING_MODEL_DIR=/models/bge-small-zh-v1.5-int8-ov
   DATASHEET_EMBEDDING_DEVICE=CPU   # 或 GPU,跟随 VLM 设置
   ```
3. 重启;`DatasheetKbService.has_embeddings` 为 True 时自动启用融合,否则保持 Phase 1/2 关键词路径。

回退保证:
- 模型目录不存在 → backend `is_active=False`,纯关键词
- `.npz` 缓存不存在 → cosine 路径跳过,纯关键词
- 配置 `DATASHEET_EMBEDDING_BACKEND=null`(默认)→ 关键词,板上零额外资源

### 路由（Phase 4 已落地）

`ContextPack` 不再用宽口袋的关键词集决定是否调 `datasheet_lookup_tool`,改成由 `app/agent/router.py:SemanticRouter` 读 `app/agent/routes/*.yaml`,按"utterances + 阈值"决策。三级回退:

1. **auto_fire**:query 命中 YAML 的 `auto_fire_part_numbers`(如 `ne555`/`555`/`lm324`/`74ls74`)→ 直接 fire。处理"随便讲讲 ne555"这种无datasheet 关键词但显然在问芯片的场景。
2. **embedding(可选)**:`DATASHEET_EMBEDDING_BACKEND=openvino` 时,bge-small-zh 启动期把 `utterances` 和 `negative_utterances` 各编码一次。query 编码后:
   ```
   score = max_cosine(query, positives) − max_cosine(query, negatives)
   fire 当 score > threshold(默认 0.30)
   ```
   负样本(`我电路里这根线接哪`、`示波器探头要夹哪个节点` 等)拥有真实否决权,把"问当前布线"的 query 排除掉。
3. **keyword 回退**:`NullEmbeddingBackend` 或 cosine 不足时,query 必须包含至少 `min_keyword_hits` 个 YAML 的 `keywords`(`datasheet`/`pinout`/`引脚`/`电气特性`等)才 fire。比之前的"任一关键词就 fire 再加一个 `return True` 兜底"严得多。

实测准确率(13 条真实风格 query,bge-small-zh INT8,threshold=0.30):**13/13**。
- 正例命中 auto_fire 5 个、embedding 2 个,覆盖中英文混杂问法。
- 反例包括"为什么这个电容方向反了"——表面上有"电容/方向",但语义偏布线 debug,被负样本余弦否决,不再误触 datasheet 工具。

`SemanticRouter` 共享 `DatasheetKbService` 的 OpenVINO embedding 后端(单进程一次 load),启用零额外资源。文件 `app/agent/routes/datasheet.yaml` 即唯一可调旋钮:`threshold` / `utterances` / `negative_utterances` / `keywords` / `auto_fire_part_numbers` 全可热替换、无需改代码。

## Future Plan

近期：

1. 给 `diagnostic_agent` 的 LangGraph 链路补 graph / agent metrics。
2. 给 `fault_cases/rc/*.json` 补真实参考图、标准波形图和接线图。
3. 增加 `datasheet_lookup_tool`，但保持本地 fallback。
4. 增加更多 agent golden tests，覆盖 short circuit、node mismatch 和 measurement error。

中期：

1. 在 feature flag 后接入可选 LLM `generate_draft` node。
2. 引入 OpenAI-compatible / OpenVINO 模型适配器。
3. 建立 PCM vs no-PCM、tool vs no-tool、verifier vs no-verifier 消融实验。
4. 将 fault cases、datasheet 和 PDF 证据统一进入可引用 ContextPack。
