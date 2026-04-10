# Validator Error Codes

`CircuitValidator` 当前输出的正式诊断格式版本为 `validator_report_v2`。

每条诊断项结构：

```json
{
  "error_code": "HOLE_MISMATCH",
  "category": "hole_errors",
  "severity": "warning",
  "message": "R1.pin1: 孔位不同，期望 B12，当前 A12",
  "suggested_action": "将 R1.pin1 插回参考孔位，或更新参考电路孔位定义。",
  "evidence_refs": [
    {"kind": "reference_component", "component_id": "R1"},
    {"kind": "pin", "component_id": "R1", "pin_name": "pin1"},
    {"kind": "expected_hole", "value": "B12"},
    {"kind": "actual_hole", "value": "A12"}
  ],
  "component_id": "R1",
  "current_component_id": "R1",
  "pin_name": "pin1",
  "expected": "B12",
  "actual": "A12"
}
```

新增字段约定：

- `suggested_action`
  - 给 guidance / agent 的单句修复建议
- `evidence_refs`
  - 给前端、RAG、agent 复用的证据索引列表
  - 常见 `kind`：
    - `reference_component`
    - `current_component`
    - `pin`
    - `expected_hole`
    - `actual_hole`
    - `expected_node`
    - `actual_node`
    - `expected_polarity`
    - `actual_polarity`
    - `net`
    - `component_type`
    - `pin_group`
    - `graph_node`
    - `diagnostic_code`
    - `diagnostic_category`

## Categories

- `topology_errors`
- `node_errors`
- `hole_errors`
- `polarity_errors`
- `component_errors`

## Codes

### Topology

- `REFERENCE_NOT_SET`
  - 未提供参考电路
- `TOPOLOGY_CHECK_FAILED`
  - 拓扑检查执行失败
- `TOPOLOGY_VALID_SUBSET`
  - 当前电路是参考电路的有效子集，但不完整
- `TOPOLOGY_MATCH_PIN_PLACEMENT_DIFFERS`
  - 拓扑一致，但孔位摆放不同
- `FLOATING_PIN`
  - 元件引脚疑似悬空
- `MULTIPLE_DISCONNECTED_SUBGRAPHS`
  - 电路存在多个独立连通分量

### Node

- `NODE_MISMATCH`
  - 目标引脚连接到了错误的静态导通节点
- `COMPONENT_SHORTED_SAME_NET`
  - 元件两脚落在同一导通组，疑似短路

### Hole

- `HOLE_MISMATCH`
  - 节点正确，但具体孔位与参考不一致

### Polarity

- `POLARITY_REVERSED`
  - 极性方向与参考相反
- `POLARITY_UNKNOWN`
  - 极性无法确定

### Component

- `COMPONENT_MISSING`
  - 元件数量少于参考
- `COMPONENT_EXTRA`
  - 元件数量多于参考
- `COMPONENT_INSTANCE_MISSING`
  - 某个参考组件没有找到对应实例
- `COMPONENT_SYMMETRY_GROUP_INCOMPLETE`
  - 对称引脚组不完整
- `PIN_MISSING`
  - 缺少参考定义的引脚
- `PIN_EXTRA`
  - 存在参考未定义的额外引脚
- `LED_SERIES_RESISTOR_MISSING`
  - LED 所在网络未检测到限流电阻

## Severity Convention

- `error`
  - 会直接影响电路正确性
- `warning`
  - 可能影响演示可靠性、教学解释或电路安全，但未必导致拓扑完全错误
