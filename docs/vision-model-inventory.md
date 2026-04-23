# Vision Model Inventory

这份文档用于整理当前工作区里的视觉模型产物，避免把旧模型、候选模型和当前主链模型混用。

当前视觉主线是：

```text
S1: YOLO-Detect component detection
S1.5: YOLO-Pose ROI pin detection
S2: pixel pin -> hole_id mapping
```

面包板网格化 / `calibrator.py` 当前由队友继续推进。本仓库这边优先保持接口稳定，并继续围绕组件检测、ROI 裁剪和 pin pose 模型评估收口。

## 当前推荐模型

### S1 组件检测

推荐权重：

```text
train_demo/detect_components/weights/best.pt
```

用途：

- 当前组件检测主模型
- 模型类型：`DetectionModel`
- 后端 backend：`yolo_detect_component`
- 标签体系已经对齐当前后端：
  - `capacitor_ceramic`
  - `capacitor_electrolytic`
  - `diode`
  - `jumper_wire`
  - `led`
  - `resistor`
  - `transistor_3pin`

训练表现：

- 最好 `Box mAP50-95`: `0.8142`
- 最后一轮 `Box mAP50-95`: `0.8124`

结论：

- 继续作为 S1 默认组件检测模型。

### S1.5 引脚检测

当前最值得继续评估的两个候选：

```text
train_demo/pose_roi_context_v12/weights/best.pt
train_demo/models/weights/best.pt
```

`pose_roi_context_v12`：

- 模型类型：`PoseModel`
- `kpt_shape = [3, 3]`
- 最好 `Pose mAP50-95`: `0.8815`
- 最后一轮 `Pose mAP50-95`: `0.8759`
- 之前在真实 ROI 链路里表现稳定，是当前稳妥候选。

`train_demo/models`：

- 模型类型：`PoseModel`
- `kpt_shape = [3, 3]`
- 训练数据：稍微扩大后的裁切小图
- 来源：原根目录 `pose_crop_by_box_v1-3`，已归档重命名到 `train_demo/models`
- 最好 `Pose mAP50-95`: `0.8745`
- 最后一轮 `Pose mAP50-95`: `0.8614`
- `Pose recall` 较高，真实复杂图上出点稳定，适合作为下一版候选主模型继续 A/B。

当前判断：

- 不建议只看 mAP 直接替换默认模型。
- 需要继续用真实图片比较：
  - pin 是否稳定来自 `model`
  - 是否出现同孔
  - 最终 `hole_id` 是否合理
  - ROI 是否包含完整元件和引脚活动区

## 历史模型

### `train_demo/pose_components/weights/best.pt`

用途：

- 旧 pose 模型。

训练表现：

- 最好 `Pose mAP50-95`: `0.8740`

当前判断：

- 指标不错，但在当前 ROI 链路里曾出现较多同孔问题。
- 保留用于对比，不建议作为默认 pin 模型。

### 已删除：旧根目录 `models/`

原用途：

- 旧检测模型。

标签体系：

- 包含旧标签，例如 `breadboard / ic / line_area / capacitor / potentiometer`。

训练表现：

- 最好 `Box mAP50-95`: `0.5307`

当前判断：

- 标签体系和当前后端不一致。
- 已从仓库删除，不再作为历史备份或默认 fallback。

## 当前调试输出

最近一次新版 `train_demo/models` 在真实图片上的可视化输出位于：

```text
/tmp/pose_crop_by_box_v1_3_camera_roll_visuals
/tmp/pose_crop_by_box_v1_3_complex_visuals
```

这些目录不属于仓库源码，仅用于本地观察模型效果。

## 协作约定

- 新模型不要直接覆盖旧路径，先放到独立目录并记录在本文档。
- 后端默认模型路径切换前，先跑真实图片 A/B。
- `S1/S1.5/S2` 的 JSON 协议优先保持稳定。
- 面包板网格化由 `calibrator.py` 对外提供接口，当前具体实现由队友继续推进；其他模块不要绕过 S2 直接猜 `hole_id`。
