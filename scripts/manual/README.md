# Manual Scripts

当前仅保留 `tools/` 目录，里面是仍有复用价值的联调和数据准备工具。

建议:

- 日常只维护 `tools/`
- 新的离线脚本优先做成可参数化工具，不再提交一次性实验脚本
- `tools/circuit/save_reference.py` 现在默认输出 `labguardian_ref_v4` 参考文件
- 如果脚本需要配合新网表链路，优先围绕 `netlist_v2`、`board_schema`、`validator_report_v2` 组织输入输出

## Vision tools

当前 `tools/vision/` 里的脚本按职责分四类:

- 正式链路调试:
  - `run_official_pipeline_debug.py`
  - 只调用正式 `run_pipeline()`
  - 适合演示、联调、核对当前主链输出
- 数据准备:
  - `labelme_pose_dataset_utils.py`
  - `mine_component_roi_priors.py`
  - `build_pose_roi_dataset.py`
- 模型研究 / 实验:
  - `evaluate_full_image_pose_dataset.py`
  - `debug_full_image_pose.py`
  - 允许使用实验性 full-image pose 逻辑, 不代表正式主链
- 面包板网格诊断:
  - `debug_calibrator_grid.py`
  - 只诊断 calibrator, 不代表完整 pipeline 演示

协作约定:

- 对外演示或回归核对当前系统能力时, 优先使用 `run_official_pipeline_debug.py`。
- `debug_full_image_pose.py` 这类脚本只能用于研究候选方案, 不能当成正式链路结果解释给团队。
- 面包板网格化实现当前由队友继续推进，`debug_calibrator_grid.py` 只作为可视化/诊断入口保留。
- 新的 pin pose 模型评估优先输出到 `/tmp/...`，不要把一次性可视化结果提交进仓库。
- 模型资产清单统一维护在 `docs/vision-model-inventory.md`。
