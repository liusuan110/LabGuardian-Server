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
    {"kind": "actual_hole", "value": "A12"},
    {
      "kind": "hole_candidate_ref",
      "component_id": "R1",
      "pin_name": "pin1",
      "current_hole_id": "A12",
      "target_hole_id": "B12"
    },
    {
      "kind": "validator_rule_ref",
      "error_code": "HOLE_MISMATCH",
      "category": "hole_errors"
    }
  ],
  "component_id": "R1",
  "current_component_id": "R1",
  "pin_name": "pin1",
  "current_pin_name": "pin1",
  "current_hole_id": "A12",
  "target_hole_id": "B12",
  "current_node_id": "ROW_12_L",
  "target_node_id": "ROW_12_L",
  "current_observation_refs": [
    {
      "kind": "pin_keypoint_ref",
      "component_id": "R1",
      "pin_name": "pin1",
      "view_id": "top",
      "keypoint": [10.0, 20.0]
    }
  ],
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
    - `component_bbox_ref`
    - `pin_keypoint_ref`
    - `hole_candidate_ref`
    - `node_trace_ref`
    - `validator_rule_ref`
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
- 定位字段
  - `current_component_id`: 当前检测到的元件实例 ID
  - `current_pin_name`: 当前引脚名；对称引脚组可为列表
  - `current_hole_id`: 当前孔位；对称引脚组或短路场景可为列表
  - `current_node_id`: 当前导通节点；对称引脚组或短路场景可为列表
  - `target_hole_id`: 参考孔位
  - `target_node_id`: 参考导通节点
  - `current_observation_refs`: 当前视觉观测引用，主要用于前端高亮和 VLM 解释

标准证据对象说明：

- `component_bbox_ref`: 指向当前元件检测框，可用于前端框选元件。
- `pin_keypoint_ref`: 指向某视角下的 pin keypoint，可用于高亮引脚。
- `hole_candidate_ref`: 指向当前孔位、目标孔位和候选孔位集合。
- `node_trace_ref`: 指向当前导通节点、目标导通节点和候选节点集合。
- `validator_rule_ref`: 指向触发该诊断的 validator 规则。
- `kb_reference_ref`: 预留给 RAG / 教学知识库引用，不由 validator 直接生成。

## Frontend Highlight Protocol

`validator_report_v2.summary.highlight_target_count` 记录可高亮目标数量。
`validator_report_v2.highlight_protocol` 是前端可直接消费的高亮协议：

```json
{
  "version": "labguardian_highlight_v1",
  "targets": [
    {
      "kind": "component_bbox_ref",
      "render": "box",
      "target_type": "component",
      "component_id": "R1",
      "view_id": "top",
      "bbox": [100, 100, 180, 140]
    },
    {
      "kind": "pin_keypoint_ref",
      "render": "point",
      "target_type": "pin",
      "component_id": "R1",
      "pin_name": "pin1",
      "view_id": "top",
      "keypoint": [10.0, 20.0],
      "radius_px": 8
    },
    {
      "kind": "hole_candidate_ref",
      "render": "hole",
      "target_type": "hole",
      "component_id": "R1",
      "pin_name": "pin1",
      "current_hole_ids": ["A12"],
      "target_hole_ids": ["B12"],
      "hole_ids": ["A12", "B12"]
    }
  ]
}
```

每条 diagnostic item 同时带 `highlight_targets`，用于前端在某条错误展开时只高亮
该错误相关目标。`AgentService mode="diagnostic_agent"` 会把同一份协议作为
`AngntEvidence(evidence_type="highlight_protocol")` 输出，方便前端从 Agent 回答页一键框元件、点引脚、亮孔位。

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
- `WIRE_ENDPOINT_UNCONNECTED`
  - 导线端点未接入其他元件，疑似导线-元件断开
- `WIRE_SELF_LOOP_OR_REDUNDANT`
  - 导线形成自环或重复连接同一节点对，疑似冗余布线
- `MULTIPLE_DISCONNECTED_SUBGRAPHS`
  - 电路存在多个独立连通分量
- `MISSING_REQUIRED_PATH`
  - `VCC` 到 `GND` 不存在有效连通路径，疑似元件间断路或关键连线缺失
- `MISSING_EXPECTED_ADJACENCY`
  - 参考电路中应连通的关键元件对在当前电路未连通

### Node

- `NODE_MISMATCH`
  - 目标引脚连接到了错误的静态导通节点
- `COMPONENT_SHORTED_SAME_NET`
  - 元件两脚落在同一导通组，疑似短路
- `POWER_RAIL_SHORT`
  - `VCC` 与 `GND` 落在同一导通网络，或单个导线/元件两端直接跨接 `VCC-GND`，属于电路级短路
- `UNEXPECTED_NET_BRIDGE`
  - 本应隔离的两网被意外桥接到同一导通网络

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
