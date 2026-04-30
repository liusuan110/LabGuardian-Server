# 电路网表生成与拓扑分析实现计划

## 一、需求分析

### 1.1 目标
利用现有三维面包板图转二维面包板孔洞连接情况图，实现：
- 上传图片进行检测分析后，得到电路中各个元件引脚的二维定位（孔位映射）
- 根据孔位映射，判断电路各元件之间的电气连接情况
- 形成电路网表，给出网表信息，得到拓扑结构
- 提供前端可用的接口，便于将端口添加到前端

### 1.2 已有架构分析

现有代码已经实现了大部分功能，架构如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Orchestrator                       │
│  (run_pipeline: images_b64 → S1→S1.5→S2→S3→S4)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌─────────────────┐      ┌───────────────┐
│  S1: Detect   │      │ S1.5: Pin Detect│      │ S2: Mapping  │
│  (YOLO检测)   │ ───▶ │ (引脚ROI检测)   │ ───▶ │ (引脚→孔位)  │
└───────────────┘      └─────────────────┘      └───────┬───────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      S3: Topology (拓扑构建)                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ CircuitAnalyzer │  │  Union-Find网表   │  │ NetworkX图   │  │
│  │ (电路分析器)    │  │  (电气网络)       │  │ (拓扑图)     │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     S4: Validate (验证)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 已有接口

| 接口 | 路径 | 功能 | 状态 |
|------|------|------|------|
| `/pipeline/run` | POST | 同步执行Pipeline | ✅ 已实现 |
| `/pipeline/submit` | POST | 异步提交任务 | ✅ 已实现 |
| `/pipeline/status/{job_id}` | GET | 查询任务状态 | ✅ 已实现 |
| `/pipeline/analyze` | POST | **电路分析(核心接口)** | ✅ 已实现 |

---

## 二、数据模型结构

### 2.1 核心数据结构（已实现）

```python
# 元件引脚二维定位
PinLocation:
  - pin_id: int              # 引脚编号
  - pin_name: str            # 引脚名称
  - hole_id: str             # 面包板孔洞ID (如 "A1", "LP5")
  - logic_loc: tuple[str,str]# 逻辑坐标 (行, 列)
  - x_warp: float            # 校正后二维图中的X坐标
  - y_warp: float            # 校正后二维图中的Y坐标
  - x_image: float           # 原始图像中的X坐标
  - y_image: float           # 原始图像中的Y坐标
  - electrical_node_id: str  # 电气节点ID
  - electrical_net_id: str   # 电气网络ID

# 元件及其引脚
ComponentWithPins:
  - component_id: str        # 元件ID
  - component_type: str      # 元件类型
  - package_type: str        # 封装类型
  - polarity: str           # 极性
  - confidence: float       # 置信度
  - pins: list[PinLocation] # 引脚定位列表
  - bbox: list[float]       # 边界框坐标

# 电气网络
ElectricalNet:
  - electrical_net_id: str   # 网络ID (如 "NET_001")
  - power_role: str          # 电源角色 (VCC/GND)
  - member_node_ids: list[str]   # 成员节点ID列表
  - member_hole_ids: list[str]  # 成员孔洞ID列表
  - connected_components: list[str] # 连接到此网络的元件ID列表

# 元件连接关系
ComponentConnection:
  - component_id: str        # 元件ID
  - connected_to: list[str] # 直接连接的元件ID列表
  - is_isolated: bool       # 是否为孤立元件
  - connected_net_ids: list[str] # 连接的网络ID列表

# 电路分析结果（主返回结构）
CircuitAnalysisResult:
  - job_id: str
  - station_id: str
  - status: JobStatus
  - total_duration_ms: float
  - components: list[ComponentWithPins]  # 元件列表（含引脚定位）
  - component_count: int
  - nets: list[ElectricalNet]            # 电气网络列表
  - net_count: int
  - topology_graph: dict                 # 节点链接格式的拓扑图
  - circuit_description: str              # 电路文本描述
  - rail_assignments: dict               # 电源轨道分配
  - component_connections: list[ComponentConnection]  # 元件连接关系
  - isolated_components: list[str]       # 孤立元件ID列表
```

### 2.2 坐标映射关系

```
原始图像坐标 (x_image, y_image)
        │
        ▼ [BreadboardCalibrator.透视变换]
校正后图像坐标 (x_warp, y_warp)
        │
        ▼ [BreadboardCalibrator.frame_pixel_to_logic]
逻辑坐标 (row, col)  ←→  孔洞ID (hole_id)
        │
        ▼ [BoardSchema.hole_to_spec]
电气节点ID (electrical_node_id)
        │
        ▼ [CircuitAnalyzer.Union-Find]
电气网络ID (electrical_net_id)
```

---

## 三、API接口说明

### 3.1 主接口：电路分析

**Endpoint:** `POST /pipeline/analyze`

**请求体 (PipelineRequest):**
```json
{
  "station_id": "station_001",
  "images_b64": ["base64_image_1", "base64_image_2"],  // 1-3张图片
  "conf": 0.25,           // 置信度阈值
  "iou": 0.5,             // NMS IoU阈值
  "imgsz": 960,           // 推理尺寸
  "rail_assignments": {    // 电源轨道分配（可选）
    "top_plus": "VCC",
    "top_minus": "GND",
    "bot_plus": "VCC",
    "bot_minus": "GND"
  }
}
```

**响应体 (CircuitAnalysisResult):**
```json
{
  "job_id": "uuid",
  "station_id": "station_001",
  "status": "completed",
  "total_duration_ms": 1234.5,
  "components": [
    {
      "component_id": "R1",
      "component_type": "Resistor",
      "package_type": "AXIAL",
      "polarity": "none",
      "confidence": 0.95,
      "pins": [
        {
          "pin_id": 1,
          "pin_name": "1",
          "hole_id": "A5",
          "logic_loc": ["5", "a"],
          "x_warp": 150.2,
          "y_warp": 80.5,
          "x_image": 320.4,
          "y_image": 180.3,
          "electrical_node_id": "ROW_5_L",
          "electrical_net_id": "NET_001"
        },
        {
          "pin_id": 2,
          "pin_name": "2",
          "hole_id": "F5",
          "logic_loc": ["5", "f"],
          "x_warp": 650.8,
          "y_warp": 80.5,
          "x_image": 820.1,
          "y_image": 180.3,
          "electrical_node_id": "ROW_5_R",
          "electrical_net_id": "NET_002"
        }
      ]
    }
  ],
  "component_count": 5,
  "nets": [
    {
      "electrical_net_id": "NET_001",
      "power_role": "VCC",
      "member_node_ids": ["ROW_5_L", "LP5"],
      "member_hole_ids": ["A5", "LP5"],
      "connected_components": ["R1", "U1"]
    },
    {
      "electrical_net_id": "NET_002",
      "power_role": "GND",
      "member_node_ids": ["ROW_5_R", "RN10"],
      "member_hole_ids": ["F5", "RN10"],
      "connected_components": ["R1", "LED1"]
    }
  ],
  "net_count": 4,
  "topology_graph": {
    "nodes": [...],
    "links": [...]
  },
  "circuit_description": "电路概况: 共 5 个元件 (Resistor×2, LED×1, IC×1, Wire×1), 4 个电气网络\n\n元件连接:\n  R1 (Resistor): 1=A5(NET_001), 2=F5(NET_002)\n  ...",
  "rail_assignments": {
    "top_plus": "VCC",
    "top_minus": "GND",
    "bot_plus": "VCC",
    "bot_minus": "GND"
  },
  "component_connections": [
    {
      "component_id": "R1",
      "connected_to": ["U1", "LED1"],
      "is_isolated": false,
      "connected_net_ids": ["NET_001", "NET_002"]
    }
  ],
  "isolated_components": []
}
```

---

## 四、拓扑图格式（前端可视化）

### 4.1 topology_graph 节点链接格式

```json
{
  "nodes": [
    {
      "id": "NET_001",
      "kind": "net",
      "power": "VCC"
    },
    {
      "id": "R1",
      "kind": "comp",
      "ctype": "Resistor",
      "polarity": "none",
      "pins": 2
    },
    {
      "id": "U1",
      "kind": "comp",
      "ctype": "IC",
      "polarity": "none",
      "pins": 8
    }
  ],
  "links": [
    {
      "source": "R1",
      "target": "NET_001",
      "role": "1"
    },
    {
      "source": "R1",
      "target": "NET_002",
      "role": "2"
    },
    {
      "source": "U1",
      "target": "NET_001",
      "role": "VCC"
    }
  ]
}
```

### 4.2 前端端口映射示例

```json
{
  "ports": [
    {
      "component_id": "R1",
      "pin_id": 1,
      "position": {
        "x": 150.2,
        "y": 80.5
      },
      "hole_id": "A5",
      "net_id": "NET_001"
    },
    {
      "component_id": "R1",
      "pin_id": 2,
      "position": {
        "x": 650.8,
        "y": 80.5
      },
      "hole_id": "F5",
      "net_id": "NET_002"
    }
  ]
}
```

---

## 五、实现状态与增强建议

### 5.1 已有功能 ✅

| 功能 | 状态 | 说明 |
|------|------|------|
| 图像检测与元件识别 | ✅ | S1 (YOLO) |
| 引脚检测与定位 | ✅ | S1.5 (Pin ROI) |
| 引脚→孔位映射 | ✅ | S2 (Mapping) |
| 二维坐标计算 | ✅ | BreadboardCalibrator |
| 电气网络分析 | ✅ | CircuitAnalyzer |
| 网表生成 | ✅ | export_netlist_v2() |
| 拓扑图生成 | ✅ | to_node_link_data() |
| /pipeline/analyze接口 | ✅ | 完整返回CircuitAnalysisResult |
| SPICE网表导出 | ✅ | export_spice_netlist() |

### 5.2 建议增强项

#### 5.2.1 增强PinLocation坐标信息
```python
# 建议在PinLocation中添加更多前端可视化所需的字段
class PinLocation(BaseModel):
    # ... 现有字段 ...

    # 新增：更精确的二维坐标（用于前端渲染）
    x_2d: float | None = Field(None, description="面包板2D图中的X坐标")
    y_2d: float | None = Field(None, description="面包板2D图中的Y坐标")

    # 新增：所属网络信息
    network_name: str | None = Field(None, description="所属电气网络名称")
```

#### 5.2.2 创建专用的前端可视化数据结构
```python
class PortMappingItem(BaseModel):
    """前端端口映射项"""
    component_id: str
    pin_id: int
    x: float  # 2D坐标
    y: float  # 2D坐标
    hole_id: str
    net_id: str
    pin_name: str

class NetlistVisualization(BaseModel):
    """网表可视化数据结构"""
    ports: list[PortMappingItem]
    nets: list[dict]  # 网络连接信息
    bounding_box: dict  # 电路边界框
```

#### 5.2.3 新增专用前端接口
```python
@router.post("/visualize/ports", response_model=NetlistVisualization)
async def get_port_mapping(request: PipelineRequest):
    """获取前端端口映射数据"""
    result = pipeline_service.analyze_circuit(...)
    return NetlistVisualization.from_analysis_result(result)
```

---

## 六、执行计划

### 6.1 阶段一：验证现有功能（0.5天）
- [ ] 验证 `/pipeline/analyze` 接口完整性
- [ ] 验证返回数据结构的正确性
- [ ] 确认前端所需数据是否完备

### 6.2 阶段二：数据模型增强（1天）
- [ ] 增强 `PinLocation` 数据结构
- [ ] 创建 `PortMappingItem` 数据结构
- [ ] 创建 `NetlistVisualization` 数据结构
- [ ] 在 `CircuitAnalysisResult` 中添加转换方法

### 6.3 阶段三：API接口增强（0.5天）
- [ ] 新增 `/pipeline/visualize/ports` 接口
- [ ] 添加请求/响应模型
- [ ] 实现数据转换逻辑

### 6.4 阶段四：测试与文档（0.5天）
- [ ] 单元测试
- [ ] 接口文档更新
- [ ] 前端集成示例

---

## 七、关键代码位置

| 功能 | 文件路径 |
|------|----------|
| Pipeline主入口 | `app/pipeline/orchestrator.py` |
| 电路分析服务 | `app/services/pipeline_service.py` (analyze_circuit方法) |
| 电路分析Schema | `app/schemas/pipeline.py` (CircuitAnalysisResult) |
| 拓扑构建 | `app/pipeline/stages/s3_topology.py` |
| 电路分析器 | `app/domain/circuit.py` (CircuitAnalyzer) |
| 面包板校准器 | `app/pipeline/vision/calibrator.py` |
| 网表模型 | `app/domain/netlist_models.py` |
| API路由 | `app/api/v1/pipeline.py` |

---

## 八、技术亮点

1. **并查集算法**：使用 Union-Find 实现高效的电气网络合并，复杂度 O(α(n))
2. **透视变换**：BreadboardCalibrator 实现三维到二维的精确映射
3. **多视角融合**：S1.5 阶段融合多个视角的检测结果
4. **NetworkX图论**：使用 NetworkX 构建拓扑图，支持多种导出格式
