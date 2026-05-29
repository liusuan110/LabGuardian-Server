# Local Student Eval Report

- model_dir: `D:\LabGuardian\LabGuardian-Server\models\labguardian-student-1p5-int4-ov`
- device: `GPU`
- available_devices: `CPU, GPU.0, GPU.1`
- questions: `1`
- max_new_tokens: `128`
- load_seconds: `24.45`
- avg_latency_s: `3.89`

---

## user_01 · custom_cli · custom

**Question**: 共射放大电路输出失真，最先该查什么？

**Latency**: `3.89s`

输出失真首先应检查反馈电阻和输入偏置电流是否匹配，其次检查晶体管是否饱和或截止。失真类型（如饱和、截止、负反馈）和具体参数（如静态工作点、增益、输入偏置电流）是诊断失真的关键。

引用依据：输出失真通常由静态工作点、增益、输入偏置电流等参数异常引起，具体应根据电路类型（如共射放大电路）和失真类型（如饱和、截止、负反馈）进行分析。

---

