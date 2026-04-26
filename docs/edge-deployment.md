# Edge Deployment Notes

这份文档记录第一阶段收口后的板端部署约定。目标是让开发环境、容器环境和后续
edge profile 使用同一组路径与运行元数据。

## Model Paths

统一入口为：

```text
LABGUARDIAN_MODEL_ROOT=/app/models
```

默认候选路径按角色查找：

```text
component detector:
  $LABGUARDIAN_MODEL_ROOT/component/best.pt
  $LABGUARDIAN_MODEL_ROOT/component_detector/best.pt
  $LABGUARDIAN_MODEL_ROOT/detect_components/best.pt
  $LABGUARDIAN_MODEL_ROOT/detect_components/weights/best.pt

pin detector:
  $LABGUARDIAN_MODEL_ROOT/pin/best.pt
  $LABGUARDIAN_MODEL_ROOT/pin_detector/best.pt
  $LABGUARDIAN_MODEL_ROOT/pose_roi_context_v12/best.pt
  $LABGUARDIAN_MODEL_ROOT/pose_roi_context_v12/weights/best.pt
  $LABGUARDIAN_MODEL_ROOT/models/best.pt
  $LABGUARDIAN_MODEL_ROOT/models/weights/best.pt
```

显式配置 `YOLO_MODEL_PATH` / `PIN_MODEL_PATH` 时仍然优先使用显式路径；若路径
不存在，会回退到上述候选路径，再回退到仓库内 `train_demo` 权重。

## Runtime Defaults

当前统一默认推理尺寸：

```text
YOLO_IMGSZ=960
PIPELINE_HIGH_RES_IMGSZ=960
PipelineRequest.imgsz=960
```

容器默认把 `./models` 只读挂载到 `/app/models`，并显式设置：

```text
YOLO_MODEL_PATH=/app/models/component/best.pt
PIN_MODEL_PATH=/app/models/pin/best.pt
```

如果这两个路径不存在，服务会继续尝试同一模型根目录下的其他候选布局。

## Pipeline Result Metadata

`run_pipeline()` 现在会在顶层返回 `runtime_metadata`，并由 `PipelineResult`
透出，用于后续 benchmark 和论文实验记录：

```json
{
  "code_version": "0.1.0",
  "model_version": "dev",
  "kb_version": "none",
  "rule_version": "dev",
  "model_root": "/app/models",
  "component_model_path": "/app/models/component/best.pt",
  "pin_model_path": "/app/models/pin/best.pt",
  "yolo_device": "cpu",
  "pin_model_device": "cpu",
  "conf": 0.25,
  "iou": 0.5,
  "imgsz": 960,
  "board_rows": 63,
  "board_cols_per_side": 5
}
```

## Board Schema

默认比赛板 schema 使用 63 行主区和 63 行电源轨，电源轨分段固定为：

```text
LP/LN/RP/RN 1-31  -> *_SEG1
LP/LN/RP/RN 32-63 -> *_SEG2
```

