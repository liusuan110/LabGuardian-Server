# Manual Scripts

当前仅保留 `tools/` 目录，里面是仍有复用价值的联调和数据准备工具。

建议:

- 日常只维护 `tools/`
- 新的离线脚本优先做成可参数化工具，不再提交一次性实验脚本
- `tools/circuit/save_reference.py` 现在默认输出 `labguardian_ref_v4` 参考文件
- 如果脚本需要配合新网表链路，优先围绕 `netlist_v2`、`board_schema`、`validator_report_v2` 组织输入输出

## Vision tools

当前 `tools/vision/` 里的文件分成两类:

- 可直接运行的入口脚本
- 被入口脚本复用的辅助模块

### 当前仍在使用的入口脚本

- 正式链路调试:
  - `run_official_pipeline_debug.py`
  - 只调用正式 `run_pipeline()`
  - 适合演示、联调、核对当前主链输出
- ROI / 数据集准备:
  - `mine_component_roi_priors.py`
  - `build_pose_roi_dataset.py`
- 模型研究 / 实验:
  - `evaluate_full_image_pose_dataset.py`
  - `debug_full_image_pose.py`
  - 当前主要用于单独研究 full-image pose 匹配细节
- OpenVINO 板端 smoke:
  - `run_openvino_pose_smoke.py`
  - 验证 OpenVINO 推理路径下 pose 结果与基线一致

### 辅助模块

- `labelme_pose_dataset_utils.py`
  - 不是独立入口脚本
  - 当前被 `mine_component_roi_priors.py` 和 `build_pose_roi_dataset.py` 复用

### 当前整理结论

- `tools/vision/` 目录里目前没有“高置信度可直接删除”的多余脚本
- 但要注意区分:
- `run_official_pipeline_debug.py` 才是当前视觉主链调试入口
  - `debug_full_image_pose.py` 是实验脚本, 但当前正式 S1.5 也已切到 full-image pose 主语义
  - `run_openvino_pose_smoke.py` 服务 OpenVINO 推理通路 smoke
  - `labelme_pose_dataset_utils.py` 是 helper, 不是一个给团队直接运行的主入口

协作约定:

- 对外演示或回归核对当前系统能力时, 优先使用 `run_official_pipeline_debug.py`。
- `debug_full_image_pose.py` 这类脚本只能用于研究候选方案, 不能当成正式链路结果解释给团队。
- 面包板网格化实现当前由队友继续推进。
- 新的 pin pose 模型评估优先输出到 `/tmp/...`，不要把一次性可视化结果提交进仓库。
- 模型资产清单统一维护在 `docs/vision-model-inventory.md`。

## VLM tools

- `tools/vlm/smoke_openvino_vlm.py`
  - 默认按 `GPU` 设备跑 `openvino_genai`
  - 用一阶 RC 的本地 MRAG 包做最小解释 smoke
  - 输出设备列表、时延和 `vlm_explanation_v1` 结果
- `tools/vlm/smoke_lang_vlm_integration.py`
  - 默认按 `GPU` 设备跑 `diagnostic_agent + MRAG + openvino_genai`
  - 内置 `short_circuit / floating / wrong_hole` 三个教学场景
  - 适合板端联调 LangGraph 白盒诊断和 VLM 解释闭环
