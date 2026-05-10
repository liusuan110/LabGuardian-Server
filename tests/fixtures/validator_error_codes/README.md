# Validator Error Code Fixtures

这里放最小 `validator_report_v2` 回归样例。

覆盖的 error code：

- `REFERENCE_NOT_SET`
- `HOLE_MISMATCH`
- `NODE_MISMATCH`
- `POLARITY_REVERSED`
- `POLARITY_UNKNOWN`
- `FLOATING_PIN`
- `COMPONENT_SHORTED_SAME_NET`
- `POWER_RAIL_SHORT`
- `MISSING_REQUIRED_PATH`
- `WIRE_ENDPOINT_UNCONNECTED`
- `UNEXPECTED_NET_BRIDGE`
- `WIRE_SELF_LOOP_OR_REDUNDANT`
- `MISSING_EXPECTED_ADJACENCY`
- (absence) `FLOATING_PIN` on power rails (`mapped_power_pins_only.json`)
- (absence) `WIRE_ENDPOINT_UNCONNECTED` on rail pin (`mapped_wire_rail_pin_exempt.json`)
- `POWER_RAIL_SHORT` on direct component bridge (`mapped_component_direct_vcc_gnd_bridge.json`)
- `LED_SERIES_RESISTOR_MISSING`
- `COMPONENT_MISSING`
- `COMPONENT_EXTRA`
- `COMPONENT_INSTANCE_MISSING`
- `COMPONENT_SYMMETRY_GROUP_INCOMPLETE`
- `PIN_MISSING`
- `PIN_EXTRA`
- `TOPOLOGY_VALID_SUBSET`
- `MULTIPLE_DISCONNECTED_SUBGRAPHS`

约定：

- `reference_*.json` 是 `labguardian_ref_v4` 参考文件
- `mapped_*.json` 是当前输入组件列表
- `test_validator_error_codes.py` 会逐一加载这些样例并断言目标 code 存在
