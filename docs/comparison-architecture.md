# Logical Comparison Architecture

本文档描述当前后端重构后的逻辑比较架构。现在主链路已经从“用户手写完整参考 JSON + 严格 net role 对齐”切换为：

```text
Python DSL reference
-> logical_reference_v1 payload
-> reference graph

current netlist_v2
-> rail / port / alias / merge normalization
-> current graph

reference graph + current graph
-> topology-aware graph compare
-> validator_report_v2 compatible report
```

核心目标是让用户只标注输入/输出端口和电源轨，其余内部 signal、对称端口和无极性两脚器件顺序由系统推断。

## Module Map

### Reference DSL

参考电路源码位于 `knowledge/references/*.py`，使用 SKiDL 风格的 Python DSL 编写。

关键模块：

- `app/domain/dsl/core.py`
  - `Circuit`
  - `Net`
  - `Component`
  - `Pin`
  - `PinSelection`
- `app/domain/dsl/components.py`
  - `Resistor`
  - `Capacitor`
  - `CapacitorCeramic`
  - `CapacitorElectrolytic`
  - `LED`
  - `Diode`
  - `Transistor`
  - `Potentiometer`
  - `Wire`
  - `IC`
- `app/domain/dsl/compile.py`
  - `circuit_to_logical_reference(circuit)`
  - 输出仍是 `logical_reference_v1`
- `app/domain/dsl/loader.py`
  - 执行 reference `.py`
  - 读取 module-level `circuit` 或 `c`
  - 编译并缓存结果

示例：

```python
from app.domain.dsl import CapacitorCeramic, Circuit, Resistor

circuit = Circuit(
    reference_id="rc_first_order_v1",
    name="一阶 RC 带通滤波器逻辑参考电路",
)

VIN = circuit.input("VIN", label="UI1")
VLP = circuit.net("VLP")
VOUT = circuit.output("VOUT", label="UO1")
GND = circuit.ground()

R1 = Resistor("R1")
C1 = CapacitorCeramic("C1")
C2 = CapacitorCeramic("C2")
R2 = Resistor("R2")

R1[1, 2] += VIN, VLP
C1[1, 2] += VLP, GND
C2[1, 2] += VLP, VOUT
R2[1, 2] += VOUT, GND
```

DSL 是 authoring format，不是新的比较协议。比较器只消费编译后的 `logical_reference_v1`。

### Reference Service

`app/services/reference_service.py` 是参考电路加载入口。

行为：

- `knowledge/references/{reference_id}.py` 优先。
- 同名 `.json` 只作为兼容 fallback。
- 加载后统一校验 `logical_reference_v1`。
- DSL 编译产物带：

```json
{
  "format": "logical_reference_v1",
  "source": {
    "type": "dsl_python_v1"
  }
}
```

当前生产 reference 目录只保留 `.py` 文件。

### Graph Builders

`app/domain/logical_reference.py` 是 payload/netlist 到图模型的边界。

函数：

- `logical_reference_to_graph(payload)`
- `current_netlist_v2_to_graph(netlist_v2)`

图结构是 NetworkX 无向二分图：

```text
component node <-> net node
```

component node 关键属性：

- `kind="comp"`
- `ctype`
- `source_id`

net node 关键属性：

- `kind="net"`
- `role`
- `role_label`
- `role_source`
- `canonical_name`
- `aliases`
- `source_id`

edge 关键属性：

- `pin`
- `pin_role`
- `comp_type`

## Current Netlist Normalization

当前电路来自 S3 的 `netlist_v2`。进入 S4 之前会先做几类后处理。

### Rail Assignments

电源轨仍由 `rail_assignments` 设置。S3 topology 根据电源轨配置写入 `power_role` / 电源网络信息。

### Port Annotations

最小标注协议由 `app/schemas/pipeline.py` 定义：

```python
class PinSelector(BaseModel):
    hole_id: str | None = None
    component_id: str | None = None
    pin_name: str | None = None
    electrical_net_id: str | None = None
    electrical_node_id: str | None = None
    x_image: float | None = None
    y_image: float | None = None

class PortAnnotation(BaseModel):
    role: Literal["input", "output"]
    target: PinSelector
    label: str | None = None
    source: str = "port_annotation"
```

处理入口是 `app/pipeline/net_roles.py`：

```text
apply_net_role_assignments(
  netlist_v2,
  net_role_assignments,
  port_annotations=port_annotations,
)
```

应用顺序：

1. `port_annotations`
2. legacy `net_role_assignments`

普通前端流程应只提交 input/output `port_annotations`。不要再要求用户标注内部 `signal`。

### Alias / Merge Normalization

`app/domain/net_normalization.py` 负责：

- 应用手动 alias。
- 应用 net merge。
- 根据 reference 推断 current net 的 `canonical_name`。
- 生成 `logical_nets` 给报告层和前端消费。

这一步不会改变物理 `electrical_net_id`，只补稳定逻辑名和别名。

## Compare Package

旧的 `app/domain/graph_compare.py` 现在只保留兼容 re-export。真实实现位于 `app/domain/compare/`。

### orchestrator.py

顶层入口：

```python
compare_logical_graphs(
    reference_graph,
    current_graph,
    ref_payload=reference_circuit,
    cur_netlist_v2=current_netlist_v2,
)
```

比较级联：

1. 自动检测 reference symmetry。
2. 尝试完整图同构。
3. 如果失败，尝试从 reference 推断 current net role，再比较一次。
4. 如果 current 包含 reference，判定为正确但存在额外元件或连接。
5. 如果 current 是 reference 子图，判定为未完成。
6. 否则使用 graph edit distance / fallback 生成错接报告。

### matcher.py

负责节点、边、同构和对称匹配。

关键策略：

- `power` / `ground` 严格匹配。
- `input` / `output` 只在 reference 侧存在关键 label 时严格检查。
- current 端口 label 为空时允许推断。
- `signal` 内部 net 不靠 label 匹配，只靠拓扑。
- `Resistor` / `Capacitor` / `CapacitorCeramic` / `Wire` 忽略两脚顺序。
- `Capacitor` 与 `CapacitorCeramic` 视作非极性等价类型。
- `Transistor` / `Potentiometer` / `LED` / `Diode` / `CapacitorElectrolytic` 保持功能引脚严格。
- `auto_detect_symmetries(reference_graph)` 会把拓扑签名相同的非电源 net 加入可互换 label 集合。

### role_inference.py

当直接同构失败时，比较器尝试从 reference 向 current 推断 net role。

规则：

- 不覆盖 `role_source == "manual_role"` 的人工角色。
- 可覆盖默认或自动来源的 current signal。
- 多个同构映射给出相同 role 但不同 label 时，接受 role，label 留空。
- 多个映射 role 冲突时放弃推断。
- 使用 label 匹配度选择最佳映射：
  - 完全匹配加 1
  - 部分包含加 0.5

推断结果写回临时 graph/netlist，并附加到 report summary：

```json
{
  "role_inference_applied": true,
  "inferred_net_roles": [...]
}
```

### diff_report.py

负责将比较结果转换为前端和 Agent 消费的报告。

输出保持兼容：

```json
{
  "version": "validator_report_v2",
  "summary": {...},
  "items": [...]
}
```

注意：`validator_report_v2` 现在只是报告协议名，不代表旧 `CircuitValidator` 仍在主链路中。

报告层职责：

- 生成 component mapping。
- 生成 net mapping。
- 生成 missing / extra / wrong connection / role mismatch / short circuit 等 items。
- 计算 similarity / progress。
- 把 `canonical_name`、`role_label`、`power_role` 统一用于 role mismatch 判断，避免 normalization 后的 current net 被误报。

## Pipeline Integration

### Full Pipeline

入口：

- `app/pipeline/orchestrator.py`

主流程：

```text
S1 detect
-> S1.5 pin detect
-> S2 mapping
-> S3 topology
-> apply port/manual role assignments
-> normalize_current_netlist
-> S4 validate
-> S5 semantic analysis
```

### S4 Validate

入口：

- `app/pipeline/stages/s4_validate.py`

职责：

- 拒绝旧 reference 格式。
- 构建 reference graph。
- 构建 current graph。
- 调用 `compare_logical_graphs`。
- 将结果映射为 S4 response：
  - `is_correct`
  - `risk_level`
  - `diagnostics`
  - `comparison_report`

### API

主要接口：

- `POST /api/v1/pipeline/run`
- `POST /api/v1/pipeline/recompute-corrected`
- `POST /api/v1/pipeline/compare-netlist`
- `GET /api/v1/references`
- `GET /api/v1/references/{reference_id}`

前端推荐最小请求：

```json
{
  "rail_assignments": {
    "top_plus": "VCC",
    "top_minus": "VCC",
    "bot_plus": "GND",
    "bot_minus": "GND"
  },
  "port_annotations": [
    {
      "role": "input",
      "target": {
        "component_id": "R1",
        "pin_name": "pin1"
      }
    },
    {
      "role": "output",
      "target": {
        "component_id": "R2",
        "pin_name": "pin1"
      }
    }
  ]
}
```

## Compatibility Boundaries

保留：

- `from app.domain.graph_compare import compare_logical_graphs`
- `logical_reference_v1`
- `validator_report_v2` 报告协议名
- `.json` reference fallback
- legacy `net_role_assignments`

不再作为主链路：

- `CircuitValidator`
- `app/domain/validator.py`
- `labguardian_ref_v4`
- 直接把 `netlist_v2` 当 reference 格式
- 要求用户标完整 nets / pin-to-net / roles 的 reference JSON
- 要求前端手动标注内部 signal

## Validation Commands

核心比较测试：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/labguardian-pycache \
  .venv/bin/python -m pytest \
  tests/domain/test_graph_compare*.py \
  tests/pipeline/test_logical_reference_validate.py \
  tests/pipeline/test_logical_reference_validate_detailed.py -v
```

DSL / reference service 测试：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/labguardian-pycache \
  .venv/bin/python -m pytest \
  tests/domain/dsl \
  tests/domain/test_reference_service.py \
  tests/pipeline/test_logical_reference_by_reference_id.py -q
```

API / minimal annotation 测试：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/labguardian-pycache \
  .venv/bin/python -m pytest \
  tests/api/test_compare_netlist.py \
  tests/pipeline/test_recompute_corrected_net_roles.py -q
```

## Maintenance Rules

- 修改 DSL 语义时，同步更新 `tests/domain/dsl/`。
- 修改 graph matching 行为时，同步更新 `tests/domain/test_graph_compare*.py`。
- 修改报告 item / error code 时，同步更新 `docs/validator-error-codes.md`。
- 修改前端请求协议时，同步更新 `app/schemas/pipeline.py`、`docs/board-schema-format.md` 和前端类型定义。
- 新增 reference 时优先写 `knowledge/references/*.py`，不要新增手写 JSON。
