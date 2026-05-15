# Vision Model Inventory

这份文档用于整理当前工作区里的视觉模型产物，避免把旧模型、候选模型和当前主链模型混用。

当前视觉主线是：

```text
S1: YOLO-Detect component detection
S1.5: YOLO-Pose full-image pin detection
S2: pixel pin -> hole_id mapping
```

面包板网格化 / `calibrator.py` 当前已经切到队友的 detected-hole 主路径。本仓库这边当前重点是继续围绕 full-image pose pin 质量和 hole 匹配稳定性收口。

## 当前推荐模型

### S1 组件检测

推荐权重：

```text
train_demo/merged_det_v2/weights/best.pt
```

用途：

- 当前组件检测主模型，合并基础元件、电位器、DIP-8/DIP-14 IC
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
  - `potentiometer`
  - `ic_8`
  - `ic_14`

训练表现：

- 最好 `Box mAP50-95`: `0.8359`
- 最后一轮 `Box mAP50-95`: `0.8300`

结论：

- 作为 S1 默认组件检测模型。

### S1.5 引脚检测

当前最值得继续评估的候选：

```text
train_demo/pose_components/weights/best.pt
```

`pose_components`：

- 模型类型：`PoseModel`
- `kpt_shape = [3, 3]`
- 用途：当前正式 S1.5 full-image pose 主模型
- 优势：整图直接输出元件框和 pin keypoint，便于与新 S2 hole mapping 主链直连
- 当前主要风险：部分器件的 keypoint 语义更接近引脚端点而不是孔中心

当前判断：

- 正式主链默认已切到 `pose_components`
- 单组件裁切小图后的 pin 识别链路已移除，后续 S1.5 只评估整图 pose 结果

## 历史模型

### `train_demo/pose_components/weights/best.pt`

用途：

- 旧 pose 模型。

训练表现：

- 最好 `Pose mAP50-95`: `0.8740`

当前判断：

- 指标不错，但 keypoint 落点仍需要结合 S2 hole mapping 继续评估。
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

## 协作约定

- 新模型不要直接覆盖旧路径，先放到独立目录并记录在本文档。
- 后端默认模型路径切换前，先跑真实图片 A/B。
- `S1/S1.5/S2` 的 JSON 协议优先保持稳定。
- 面包板网格化由 `calibrator.py` 对外提供接口，当前具体实现由队友继续推进；其他模块不要绕过 S2 直接猜 `hole_id`。
