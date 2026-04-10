# Current Status

这份文档用于回答 3 个团队协作里的高频问题：

1. 当前迁移做到哪一步了？
2. 哪些链路已经可信，哪些仍然是兼容过渡层？
3. 后续新增功能应该落到哪里？

## 当前结论

项目已经从“demo 可跑”推进到了“后端主语义已经成型”的阶段。

目前建议默认以新链路为准：

```text
component_id + pin_name + hole_id
-> electrical_node_id
-> electrical_net_id
-> netlist_v2
-> validator_report_v2
```

旧链路仍然保留，只用于兼容：

```text
pin1_logic / pin2_logic
-> legacy analyzer
-> legacy netlist
```

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

- S2 开始输出 `components[].pins[]`
- `topology_input.py` 兼容旧/新结构
- S3 / S4 使用统一 analyzer builder

仍待继续：

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

### 不要再新增只服务旧结构的新逻辑

新逻辑应该尽量直接落到：

- `components[].pins[]`
- `netlist_v2`
- `validator_report_v2`

### 兼容层统一收口

如果必须兼容旧字段，优先放到：

- `app/pipeline/topology_input.py`

不要让兼容逻辑散落到：

- API 层
- 服务层
- agent 层

### 文档和 fixture 必须跟代码一起更新

这类文件改动后建议同步更新：

- `docs/validator-error-codes.md`
- `tests/fixtures/validator_error_codes/`
- `tests/manual/smoke/`
