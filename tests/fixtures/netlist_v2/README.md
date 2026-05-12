# Netlist V2 Fixtures

这里放最小可回归的 `netlist_v2` 历史样例，目标是验证：

- `component_id + pin_name + hole_id` 比较逻辑
- `hole_id -> electrical_node_id -> electrical_net_id` 链路

`labguardian_ref_v4` 参考文件格式已经退役。新 S4 参考电路只使用
`logical_reference_v1`，这里的 v4 文件仅作历史 fixture。

当前样例：

- `reference_simple_v4.json`: 一个已退役的最小 v4 参考电路 fixture
- `mapped_components_simple.json`: 对应的 S2 风格输入
- `mapped_components_s01_correct.json`: 对齐 `reference_S01.json` 的结构化样例（应匹配）
- `mapped_components_s01_faulty.json`: 人工注入孔位/节点偏差的结构化样例（应报错）
