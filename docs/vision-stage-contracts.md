# Vision Stage Contracts

当前视觉主链固定为:

```text
S1 component detect
-> S1.5 full-image pose pin detect
-> S2 hole mapping
```

这三层的协议在模型训练完成前就应保持稳定。

当前默认视觉主路径：

- `S1`: `YOLO-Detect`
- `S1.5`: `YOLO-Pose`

其中 `OBB` 相关字段仅作为历史兼容保留，不是当前主方案。

## S1

阶段: `component_detect_v1`

职责:

- 只使用 `top` 视图建立全局 `component_id`
- 侧视图在当前版本不参与组件实例化
- 若 `top` 解码失败, S1 不产生检测结果

当前默认 backend:

- `detector_backend = "yolo_detect_component"`
- 若历史兼容权重为 `OBBModel`，才可能出现 `yolo_obb_component`

顶层字段:

- `interface_version`
- `detector_backend`
- `detector_contract`
- `detections`
- `supplemental_detections`
- `recall_mode`
- `primary_image_shape`
- `decoded_view_count`
- `available_view_ids`
- `dropped_view_ids`
- `decode_errors`
- `duration_ms`

每个 detection:

- `component_id`
- `input_detection_interface_version`
- `class_name`
- `component_type`
- `package_type`
- `pin_schema_id`
- `confidence`
- `bbox`
- `is_obb`（兼容字段，detect 主路径通常为 `false`）
- `orientation`
- `view_id`
- `source`
- `source_model_type`
- `wire_color`
- `obb_corners`（兼容字段，detect 主路径通常为 `null`）

每个 supplemental detection:

- `candidate_id`
- `class_name`
- `component_type`
- `package_type`
- `pin_schema_id`
- `confidence`
- `bbox`
- `is_obb`（兼容字段）
- `orientation`
- `view_id`
- `source`
- `source_model_type`
- `instance_status`
- `wire_color`
- `obb_corners`（兼容字段）

## S1.5

阶段: `component_pin_detect_v1`

顶层字段:

- `interface_version`
- `pin_detector_backend`
- `pin_detector_mode`
- `pin_detector_contract`
- `components`
- `decoded_view_count`
- `available_view_ids`
- `dropped_view_ids`
- `decode_errors`
- `duration_ms`

每个 component:

- `component_id`
- `component_type`
- `class_name`
- `package_type`
- `pin_schema_id`
- `input_detection_interface_version`
- `input_pin_detect_interface_version`
- `part_subtype`
- `symmetry_group`
- `bbox`
- `confidence`
- `orientation`
- `pins`
- `roi`
- `roi_by_view`
- `pin_detector`

当前主语义:

- `top` 整图 pose 是默认 pin 来源
- `roi` / `roi_by_view` 仅作为兼容外壳继续保留
- 若 `pin_detector_mode="full_image_model"`，则不应再把 `roi` 理解成真实裁切推理来源

每个 pin:

- `pin_id`
- `pin_name`
- `keypoints_by_view`
- `visibility_by_view`
- `score_by_view`
- `source_by_view`
- `confidence`
- `source`
- `metadata`

约定:

- `source="model"` 表示来自真实 `YOLO-Pose`
- `source="heuristic_fallback"` 表示来自 fallback
- fallback 可以继续存在, 但必须显式标记, 不得伪装成模型输出
- `roi_by_view` 现在主要作为兼容输出壳保留:
  - `top` 视图常见为 `source="full_image_pose"`
  - 侧视图若未参与默认判定，应明确标记为 `unavailable`
- 若未来重新启用 ROI 路径，才需要恢复 `associated_bbox_candidate / shared_bbox_fallback` 这类语义

## S2

阶段: `hole_mapping_v1`

顶层字段:

- `interface_version`
- `board_schema_id`
- `calibration`
- `decoded_view_count`
- `available_view_ids`
- `dropped_view_ids`
- `decode_errors`
- `components`
- `duration_ms`

`calibration`:

- `mode`
- `grid_ready`

每个 component:

- 继承 S1.5 的核心组件字段
- `pins`

每个 mapped pin:

- `pin_id`
- `pin_name`
- `logic_loc`
- `hole_id`
- `electrical_node_id`
- `confidence`
- `observations`
- `candidate_hole_ids`
- `candidate_node_ids`
- `candidate_count`
- `primary_visibility`
- `visible_view_ids`
- `observation_count`
- `is_ambiguous`
- `ambiguity_reasons`
- `is_anchor_pin`
- `source`
- `evidence_source`
- `decisive_view_id`
- `fusion_confidence`
- `fusion_margin`
- `cross_view_agreement`
- `snap_distance_px`
- `snap_confidence`
- `metadata`

约定:

- `source` 继承自 S1.5 pin 预测来源
- `metadata.mapping_interface_version` 固定写入 `hole_mapping_v1`
- `metadata.vote_scores` 记录每个候选 hole 的多视图投票分数
- `metadata.selected_by="multi_view_weighted_vote"` 表示最终 hole 来自多视图加权投票
- `calibration.mode="synthetic_fallback"` 时, 下游应将结果视为低可信校准

多视图融合元字段:

- `evidence_source ∈ {top, left_front, right_front, fused, explicit_or_fallback, none}`
  - `fused`: 至少两个可见视图的 top-1 与最终 hole 一致
  - `top` / `left_front` / `right_front`: 单一视图主导
  - `explicit_or_fallback`: 没有可信视图证据，由 explicit hole 或 fallback 兜底
- `decisive_view_id`: 对最终 hole 贡献最大的视图 id
- `fusion_confidence`: 赢家得分 / 总得分, 范围 [0,1]
- `fusion_margin`: (赢家 - 第二名) / 总得分, 范围 [0,1], 越大越稳
- `cross_view_agreement`: 各可见视图 top-1 与最终 hole 一致的比例
- `metadata.fusion`:
  - `per_view_contribution`: 每个视图贡献给赢家的分数
  - `per_view_top1`: 每个视图自己的 top-1 候选 hole
  - `occlusion_boost`: top 视图被遮挡时对各视图权重的动态加成因子

遮挡感知规则 (写在 `_compute_occlusion_boost`):

- top visibility ≥ 2 且 confidence > 0.3 → 无调整
- top visibility = 1 → side 视图 ×1.25
- top visibility = 0 或 confidence ≤ 0 → top ×0.4, side ×1.6

吸附质量字段 (孔洞吸附精准度):

- `observations[*].snap_distance_px`: 该视图候选 hole 离原始预测点的像素距离
- `observations[*].snap_normalized`: `snap_distance_px / pitch_px`, 范围 [0, ~1]
- `observations[*].snap_confidence`: 由距离换算的吸附置信度,
  公式 `max(0, 1 - (d/pitch)^2)`, 范围 [0, 1]
- `observations[*].pitch_px`: 当前 calibrator 的代表 grid pitch
- `observations[*].candidate_distances_px`: 每个候选孔的距离, 与 `candidate_hole_ids` 同序
- pin 级 `snap_distance_px` / `snap_confidence`: 取所有可见且支持最终 hole
  的视图中最优值, 作为该 pin 吸附质量的代理
- 吸附质量也参与多视图投票权重: `_snap_weight` 把 `snap_confidence` 映射到
  [0.4, 1.0]; 模型置信度高但远离孔的预测会被自动降权
- 当 pin 上的最优 `snap_confidence < 0.5`, `ambiguity_reasons` 中会出现
  `low_snap_confidence`

下游消费提示:

- validator / agent 引用 mapped pin 时, 优先用 `evidence_source` 和 `fusion_margin`
  判断证据强度, 不再直接读 `vote_scores` 原始值
- `cross_view_agreement < 1` 通常意味着 ambiguity, 应配合 `ambiguity_reasons`
  里的 `multi_view_vote_conflict` 一起用
- `low_snap_confidence` 表示几何吸附本身存疑 (pin keypoint 可能预测偏了),
  与 `multi_view_vote_conflict` 是两类独立的不确定性来源

协作边界:

- 面包板二维网格化 / `calibrator.py` 的内部拟合策略当前由队友继续推进
- S2 只依赖 `BreadboardCalibrator` 暴露的候选查询接口
- 上游视觉阶段不得直接写死孔位编号或绕过 S2 生成最终 `hole_id`
