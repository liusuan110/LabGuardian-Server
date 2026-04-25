# 教学知识库

这个目录存放 RAG/Agent 使用的结构化教学知识。它和原始 PDF 存储不是一回事：

- 原始 PDF、芯片手册后续可以进入向量库，作为引用来源。
- 教学场景负责描述当前实验、正确现象、常见错误和纠错步骤。
- 一阶 RC 第一阶段应先使用结构化教学知识，再补充 PDF 片段或 datasheet。

当前阶段只关注：

- `teaching_scenes/first_order_rc_experiment.json`
- `fault_cases/rc/*.json`

## 目录说明

```text
knowledge/
├── teaching_scenes/
│   └── first_order_rc_experiment.json
├── fault_cases/
│   └── rc/
│       ├── scope_ground_not_reference_ground.json
│       ├── wrong_output_node_for_integrator.json
│       ├── probe_x10_not_accounted.json
│       ├── wrong_signal_offset.json
│       └── capacitor_value_mismatch.json
└── reference_assets/
    └── README.md
```

## 教学场景字段

- `scene_id`：稳定场景 ID。
- `source_materials`：来源课件或资料。
- `learning_goals`：学生应理解的知识目标。
- `required_equipment`：仪器和元件。
- `circuit_principles`：正确现象背后的电路原理。
- `expected_measurements`：正确实验应观察到的测量结果。
- `common_faults`：场景内的常见错误摘要。
- `rag_queries`：该场景应能回答的典型问题。

## 错误知识单元字段

- `knowledge_id`：稳定知识单元 ID。
- `scene_id`：所属实验场景，目前固定为 `exp_first_order_rc`。
- `error_tags`：面向教学的错误标签。
- `related_error_codes`：可映射的 `validator_report_v2` 错误码。
- `reference_text`：可直接用于回答的依据说明。
- `reference_images`：参考接线图或仪器设置图路径。
- `reference_waveforms`：标准波形或错误波形示例路径。
- `reference_schematics`：参考原理图路径。
- `fix_steps`：可展示给学生的修复步骤。

当前 JSON 格式可以直接被 Python 标准库加载，不额外引入 YAML 依赖。

## M-RAG 知识包

阶段 3 增加了 `MragService`，它会把当前一阶 RC 场景、错误标签、错误知识单元、参考图路径和修复步骤整理为 `mrag_pack_v1`。

这个知识包是后续前端展示和 VLM 双图对比的输入边界。当前阶段只生成结构化数据，不调用 VLM。

## 轻量 VLM 接入

阶段 4 增加了 `VlmService`。它消费 `mrag_pack_v1`，并可附带当前实拍图和参考图路径。

默认 `VLM_PROVIDER=template`，只返回基于知识包的模板解释；板端模型服务就绪后，再切换到 `openai_compatible`。

VLM 的职责只包括：

- 对比解释当前图与参考图。
- 结合错误标签和知识包说明原因。
- 输出简短修改步骤。

VLM 不负责：

- 元件识别。
- 孔位级定位。
- `hole_id / node_id / netlist` 恢复。
