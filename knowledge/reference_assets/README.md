# 一阶 RC 参考图文资产目录

阶段 2 只建立图文知识单元的路径规范，不把临时截图或大体积图片直接塞进仓库。

后续真实资产建议放在：

```text
knowledge/reference_assets/rc/
├── rc_integrator_standard_waveform.png
├── rc_differentiator_x10_standard_waveform.png
├── rc_input_square_0_to_2v_standard.png
├── scope_ground_reference_connection.png
└── ...
```

命名规则：

- 标准波形：`rc_<实验类型>_standard_waveform.png`
- 错误示例：`rc_<错误点>_error_example.png`
- 接线参考：`rc_<对象>_connection.png`
- 原理图参考：`rc_<对象>_schematic.png`

这些路径会被 `knowledge/fault_cases/rc/*.json` 引用。真实图片补齐前，系统仍可使用文字说明和修正步骤完成第一版 RAG 闭环。
