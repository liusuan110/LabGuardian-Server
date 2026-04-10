# Manual Tests

当前仅保留两类手工测试:

- `smoke/`: 仍有价值的链路冒烟与单案例排障脚本
- `research/`: AOI / WinCLIP 等研究型验证脚本

维护原则:

- 新增手工测试优先放入 `smoke/` 或 `research/`
- 被替代或强耦合本地路径的脚本直接移除，不再长期保留
- `smoke/test_reference_v4_roundtrip.py` 用于验证 `netlist_v2` 参考文件读写与比较链路
- `smoke/test_validator_error_codes.py` 用于验证关键 `validator_report_v2` error code 回归
- `smoke/test_board_schema_default.py` 用于验证默认比赛板 schema 的孔位/轨道映射
