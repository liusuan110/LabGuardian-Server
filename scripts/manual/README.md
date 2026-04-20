# Manual Scripts

当前仅保留 `tools/` 目录，里面是仍有复用价值的联调和数据准备工具。

建议:

- 日常只维护 `tools/`
- 新的离线脚本优先做成可参数化工具，不再提交一次性实验脚本
- `tools/circuit/save_reference.py` 现在默认输出 `labguardian_ref_v4` 参考文件
- 如果脚本需要配合新网表链路，优先围绕 `netlist_v2`、`board_schema`、`validator_report_v2` 组织输入输出
