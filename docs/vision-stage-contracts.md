# Vision Stage Contracts

当前视觉主链固定为:

```text
S1 component detect
-> S1.5 component ROI pin detect
-> S2 hole mapping
```

这三层的协议在模型训练完成前就应保持稳定。

## S1

阶段: `component_detect_v1`

职责:

- 只使用 `top` 视图建立全局 `component_id`
- 侧视图在当前版本不参与组件实例化
- 若 `top` 解码失败, S1 不产生检测结果

顶层字段:

- `interface_version`
- `detector_backend`
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
- `is_obb`
- `orientation`
- `view_id`
- `source`
- `source_model_type`
- `wire_color`
- `obb_corners`

每个 supplemental detection:

- `candidate_id`
- `class_name`
- `component_type`
- `package_type`
- `pin_schema_id`
- `confidence`
- `bbox`
- `is_obb`
- `orientation`
- `view_id`
- `source`
- `source_model_type`
- `instance_status`
- `wire_color`
- `obb_corners`

## S1.5

阶段: `component_pin_detect_v1`

顶层字段:

- `interface_version`
- `pin_detector_backend`
- `pin_detector_mode`
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
- `roi_by_view[view].source="detected_bbox"` 表示 top 视图使用真实检测框
- `roi_by_view[view].source="associated_bbox_candidate"` 表示侧视图 ROI 来自 side recall 候选关联
- `roi_by_view[view].source="shared_bbox_fallback"` 表示当前没有命中 side 关联, 仍暂时共用 top bbox 裁 ROI
- `roi_by_view[view].crop_source="package_profile_crop"` 表示 ROI 已按封装裁剪策略生成
- `roi_by_view[view].crop_profile` 表示当前采用的封装裁剪模板
- `roi_by_view[view].crop_bounds` 表示实际裁剪范围
- `roi_by_view[view].association` 表示侧视图 ROI 关联元数据
- 后续真实多视图关联完成后, 只需要替换 `shared_bbox_fallback` 这一路

当前 ROI 裁剪原则:

- 轴向 2-pin 器件沿主轴保留更多 lead 空间
- DIP 封装沿短轴保留更多 pin 排空间
- top 视图优先使用 OBB 主轴
- side 视图当前允许走封装驱动的 fallback crop, 但来源必须显式标记

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
- `metadata`

约定:

- `source` 继承自 S1.5 pin 预测来源
- `metadata.mapping_interface_version` 固定写入 `hole_mapping_v1`
- `metadata.vote_scores` 记录每个候选 hole 的多视图投票分数
- `metadata.selected_by="multi_view_weighted_vote"` 表示最终 hole 来自多视图加权投票
- `calibration.mode="synthetic_fallback"` 时, 下游应将结果视为低可信校准
