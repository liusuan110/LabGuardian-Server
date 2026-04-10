# Current Status

这份文档用于回答 3 个团队协作里的高频问题：

1. 当前迁移做到哪一步了？
2. 哪些链路已经可信，哪些仍然是兼容过渡层？
3. 后续新增功能应该落到哪里？

## 当前结论

项目已经从“demo 可跑”推进到了“后端主语义已经成型”的阶段。

目前建议默认以新链路为准：

```text
component detect
-> component ROI pin detect
component_id + pin_name + hole_id
-> electrical_node_id
-> electrical_net_id
-> netlist_v2
-> validator_report_v2
```

`topology_input.py` 已不再接受旧 `pin1_logic / pin2_logic` 直接建图。
旧字段目前仍可能出现在 S2 输出中，用于调试和过渡观察，但不再参与正式拓扑构建主链。
`circuit.py / validator.py / ic_models.py / polarity.py` 的内部主逻辑也已经切到
`ComponentInstance + pins[]`，`CircuitComponent` 兼容层已退出主流程。

## 完成进度

### 1. 领域模型与 board schema

已完成：

- `board_schema.py`
- `netlist_models.py`
- 默认比赛板 schema 加载
- 电源轨分段默认规则

仍待继续：

- 用实物再确认电源轨分段位置
- 如果比赛板型和当前默认板型有差异，补一份正式 schema JSON

### 2. Pipeline 主链路迁移

已完成：

- S1.5 `component ROI pin detect` 骨架已接入 orchestrator
- S2 开始输出 `components[].pins[]`
- `topology_input.py` 已切换为只接受结构化 `components[].pins[]`
- S3 / S4 使用统一 analyzer builder
- `circuit.py` 的拓扑图生成、描述导出、SPICE 导出已优先围绕 `component_instances`
- `validator.py` 的独立诊断已优先围绕 `component_instances`
- `ic_models.py` 只输出结构化 DIP-8 引脚位置
- `polarity.py` 只接收 `ComponentInstance`

仍待继续：

- `pin_model.py` 还是 stub，占位承接旧 detection hint
- S2 的 side-view observation 目前还是轻量占位
- 真正的多视图 pin 证据还要继续接上游视觉输出

### 3. Netlist 与 Validator

已完成：

- `export_netlist_v2()`
- `labguardian_ref_v4`
- `validator_report_v2`
- `error_code + suggested_action + evidence_refs`

仍待继续：

- 将 `evidence_refs` 更直接绑定到 netlist / topology 证据对象
- 为前端和 agent 增加更强的 evidence bundle 适配层

### 4. 回归与冒烟

已完成：

- `test_board_schema_default.py`
- `test_reference_v4_roundtrip.py`
- `test_validator_error_codes.py`

目前已经覆盖的核心 error code：

- `REFERENCE_NOT_SET`
- `COMPONENT_MISSING`
- `COMPONENT_EXTRA`
- `COMPONENT_INSTANCE_MISSING`
- `COMPONENT_SYMMETRY_GROUP_INCOMPLETE`
- `HOLE_MISMATCH`
- `NODE_MISMATCH`
- `POLARITY_REVERSED`
- `POLARITY_UNKNOWN`
- `PIN_MISSING`
- `PIN_EXTRA`
- `TOPOLOGY_VALID_SUBSET`
- `FLOATING_PIN`
- `COMPONENT_SHORTED_SAME_NET`
- `LED_SERIES_RESISTOR_MISSING`
- `MULTIPLE_DISCONNECTED_SUBGRAPHS`

## 当前推荐阅读顺序

1. `README.md`
2. `docs/backend-architecture.md`
3. `docs/current-status.md`
4. `app/pipeline/stages/s2_mapping.py`
5. `app/pipeline/topology_input.py`
6. `app/domain/circuit.py`
7. `app/domain/validator.py`

## 当前最重要的协作原则

### 不要再把旧字段重新接回主链

新逻辑应该尽量直接落到：

- `components[].pins[]`
- `netlist_v2`
- `validator_report_v2`

### 兼容层统一收口

如果必须兼容旧字段，优先放到：

- `app/pipeline/topology_input.py`

当前仍允许存在的 legacy 代码，应该只作为 `domain` 内部实现细节存在，
而不应该重新暴露回 S2 / S3 / S4 / reference / validator 主链。

不要让兼容逻辑散落到：

- API 层
- 服务层
- agent 层

### 文档和 fixture 必须跟代码一起更新

这类文件改动后建议同步更新：

- `docs/validator-error-codes.md`
- `tests/fixtures/validator_error_codes/`
- `tests/manual/smoke/`
