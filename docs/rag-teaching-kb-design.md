# RAG / PCM 教学知识库设计

本项目里的 RAG 应定位为“实验调试助教”，而不是通用电子百科问答。

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
