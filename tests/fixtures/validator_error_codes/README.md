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
