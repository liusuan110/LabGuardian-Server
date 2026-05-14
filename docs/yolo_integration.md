# YOLO 组件检测模型集成

## 功能说明

本次集成为 LabGuardian Server 引入了一个专门训练的 YOLOv8 检测模型，在原有的通用元件检测能力（电阻、LED、电容等）基础上，新增了两类 IC 芯片和电位器的识别支持：

| 新增类别 | 后端类型 | 封装类型 | Pin Schema |
|----------|----------|----------|------------|
| IC-8（DIP-8 芯片） | `IC8` | `dip8` | `dip8_anchor_pair` |
| IC-14（DIP-14 芯片）| `IC14` | `dip14` | `dip14_anchor_pair` |
| 电位器 | `Potentiometer` | `potentiometer_3pin` | `fixed_pins` |

这三类元件经由完整的 S1→S1.5→S2→S3→S4 流水线，与既有的网表构建和电路校验逻辑无缝对接。

---

## 模型信息

### 基本参数

| 项目 | 值 |
|------|----|
| 框架 | Ultralytics YOLOv8 |
| 任务类型 | `detect`（水平检测框，非旋转框） |
| 权重路径 | `models/component/best.pt` |
| 文件大小 | ~6.1 MB |
| 输入尺寸 | 960×960（默认） |
| 运行设备 | CPU（默认）/ CUDA（需 `YOLO_DEVICE=cuda`） |

### 检测类别（模型原始标签 → 后端类型）

| 模型原始标签 | 后端 `component_type` | 说明 |
|-------------|----------------------|------|
| `IC-8` | `IC8` | DIP-8 封装集成电路 |
| `IC-14` | `IC14` | DIP-14 封装集成电路 |
| `potentiometer` | `Potentiometer` | 三脚旋转电位器 |
| `capacitor_ceramic` | `CapacitorCeramic` | 陶瓷电容 |
| `capacitor_electrolytic` | `CapacitorElectrolytic` | 电解电容 |
| `diode` | `Diode` | 二极管 |
| `jumper_wire` | `Wire` | 跳线 |
| `led` | `LED` | 发光二极管 |
| `resistor` | `Resistor` | 电阻 |
| `transistor_3pin` | `Transistor` | 三脚晶体管 |

标签映射由 `app/pipeline/vision/label_mapping.py` 统一维护，S1/S1.5/S2 不直接处理原始标签字符串。

### 模型性能（端到端测试）

| 场景 | 结果 |
|------|------|
| 含 IC8+IC14 的面包板图片 | 3 个目标，置信度 0.918–0.955 |
| 含 7 个电位器的面包板图片 | 7 个目标，置信度 0.896–0.946 |
| 全黑图（负样本） | 0 个目标（正常） |
| 单张推理耗时（CPU） | ~1.2–1.3 秒（960×960） |

### 训练数据

- 来源：`E:\inter\yolo8\`（本地采集的面包板俯拍图片）
- 测试集图片路径：`E:\inter\yolo8\test\images\`（13 张）
- 注意：训练数据不在本仓库内管理，模型权重通过 `models/component/best.pt` 单独交付

---

## API 使用说明

### 接口

```
POST /api/v1/pipeline/run
Content-Type: application/json
```

### 请求体

```json
{
  "station_id": "station_01",
  "images_b64": ["<base64 JPEG>"],
  "conf": 0.25,
  "iou": 0.5,
  "imgsz": 960
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `station_id` | string | ✓ | 工位 ID |
| `images_b64` | list[str] | ✓ | 1–3 张 base64 JPEG（top/left_front/right_front） |
| `conf` | float | | 置信度阈值，默认 0.25 |
| `iou` | float | | NMS IoU 阈值，默认 0.5 |
| `imgsz` | int | | 推理尺寸，默认 960 |

### 返回示例（detect 阶段）

```json
{
  "stages": [
    {
      "stage": "detect",
      "data": {
        "detector_backend": "yolo_detect_component",
        "detector_contract": {
          "task": "detect",
          "loaded": true
        },
        "detections": [
          {
            "component_id": "IC141",
            "component_type": "IC14",
            "package_type": "dip14",
            "pin_schema_id": "dip14_anchor_pair",
            "confidence": 0.955,
            "bbox": [239, 259, 288, 337],
            "orientation": 90.0
          },
          {
            "component_id": "IC81",
            "component_type": "IC8",
            "package_type": "dip8",
            "pin_schema_id": "dip8_anchor_pair",
            "confidence": 0.918,
            "bbox": [242, 102, 290, 147],
            "orientation": 90.0
          }
        ]
      }
    }
  ],
  "total_duration_ms": 1258.0
}
```

### Python 调用示例

```python
import base64, json, urllib.request

with open("board.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = json.dumps({
    "station_id": "demo",
    "images_b64": [b64],
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/api/v1/pipeline/run",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())

dets = next(s["data"]["detections"]
            for s in result["stages"] if s["stage"] == "detect")
for d in dets:
    print(d["component_type"], d["confidence"], d["bbox"])
```

---

## 部署步骤

### 方式一：本地直接运行

```bash
# 1. 放置模型权重
mkdir -p models/component
cp /path/to/best.pt models/component/best.pt

# 2. 安装依赖（项目 .venv 或 conda 环境）
pip install -e ".[dev]"

# 3. 启动 Redis
docker compose up -d redis

# 4. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. 验证
curl http://localhost:8000/health
python scripts/verify_yolo.py
```

### 方式二：Docker Compose（推荐）

```bash
# 1. 放置模型权重（挂载进容器）
mkdir -p models/component
cp /path/to/best.pt models/component/best.pt

# 2. 一键启动
docker compose up --build
```

### 方式三：仅构建镜像

```bash
docker build . -t labguardian:latest

# 运行（需挂载 models 目录）
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  labguardian:latest
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LABGUARDIAN_MODEL_ROOT` | `models/` | 模型根目录 |
| `YOLO_MODEL_PATH` | 自动发现 | 组件检测权重路径 |
| `YOLO_DEVICE` | `cpu` | `cpu` 或 `cuda` |
| `YOLO_CONF_THRESHOLD` | `0.25` | 检测置信度阈值 |
| `YOLO_IMGSZ` | `960` | 推理输入尺寸 |

模型路径自动发现优先级（`config.py`）：
```
models/component/best.pt          ← 当前生效
models/component_detector/best.pt
models/detect_components/best.pt
train_demo/detect_components/weights/best.pt
```

---

## FAQ

**Q: 模型权重为什么不在 git 仓库里？**

A: `models/` 已加入 `.gitignore`，二进制权重文件不适合用 git 管理（历史膨胀、diff 无意义）。权重文件通过项目内部渠道（共享网盘/S3/DVC）单独分发，放置到 `models/component/best.pt` 后服务自动加载。

---

**Q: 如何更新模型？**

A: 直接替换 `models/component/best.pt` 并重启服务（或 Docker 容器）。`config.py` 会在启动时重新检查路径。无需修改任何代码。

---

**Q: 新模型检测到了 IC，但类别是 IC8 还是 IC14 如何区分？**

A: 由模型本身在训练时区分。模型输出原始标签 `IC-8` 或 `IC-14`，`label_mapping.py` 负责映射为 `IC8`/`IC14`。两者的 `package_type`（`dip8`/`dip14`）和 `pin_schema_id`（`dip8_anchor_pair`/`dip14_anchor_pair`）均有独立配置，S1.5 会按对应 schema 进行 pin 检测。

---

**Q: 置信度阈值设多少合适？**

A: 默认 `0.25` 适合大多数场景。若误检多，可调高至 `0.35–0.45`；若漏检，调低至 `0.15–0.20`。通过请求体的 `conf` 字段或环境变量 `YOLO_CONF_THRESHOLD` 控制。

---

**Q: GPU 版镜像用的是 CUDA 12.8，但 PyTorch 没有 cu128 wheel？**

A: PyTorch 官方 pip 仓库当前最高支持 cu124。cu124 wheel 与 CUDA 12.8 驱动完全兼容（CUDA 向下兼容），Dockerfile 中先安装 cu124 wheel 再安装项目依赖，torch 不会被覆盖为 CPU 版。

---

**Q: 如何运行集成测试？**

```bash
# 冒烟测试（不需要运行中的服务）
python scripts/verify_yolo.py

# 端到端测试（需要服务已启动）
python scripts/e2e_test.py
```
