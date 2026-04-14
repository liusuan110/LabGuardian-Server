# LabGuardian Server

LabGuardian 的服务器端负责把视觉识别结果转换成可验证、可解释、可审计的电路诊断结果。

当前主线已经从早期 demo 形态迁移到新的结构化网表链路：

- `component_id + pin_name + hole_id -> electrical_node_id -> electrical_net_id -> netlist_v2`

后续比赛、RAG、agent、指导下发都会优先基于新链路继续演进。

基座模板：
- [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)
- [GregaVrbancic/fastapi-celery](https://github.com/GregaVrbancic/fastapi-celery)

## 当前状态

当前仓库已经完成的关键迁移：

- 服务层骨架已经建立：`pipeline_service / guidance_service / version_service / rag_service / agent_service`
- 新网表模型已经落地：`netlist_v2`
- 比赛板 `board_schema` 已接入默认加载流程
- pipeline 已切到 `S1 component detect -> S1.5 ROI pin detect -> S2 hole mapping`
- S1 现在固定由 `top` 视图建立全局 `component_id`
- S1 已支持 `side recall candidates` 输出，但 side 候选当前不直接进入主实例链
- S1.5 已支持多视图 ROI pin 检测结构，侧视图当前用显式 `shared_bbox_fallback`
- S1.5 的 ROI 裁剪已切到“按封装 + OBB 主轴”的策略，不再使用统一 margin
- S1.5 已预留侧视图 ROI 关联骨架，优先尝试使用 `side recall candidates`
- S2 开始原生输出 `components[].pins[]`
- S3 / S4 / validator 已开始消费新结构
- `circuit.py` 内部主逻辑已切到 `ComponentInstance + pins[]`
- `ic_models.py / polarity.py` 已切到新语义，不再依赖旧内部组件对象
- `validator_report_v2` 已支持结构化 `error_code + suggested_action + evidence_refs`
- 已补一组最小 regression fixture 与 smoke tests

建议先读这几份文档：

- [README.md](README.md)
- [backend-architecture.md](docs/backend-architecture.md)
- [current-status.md](docs/current-status.md)
- [board-schema-format.md](docs/board-schema-format.md)
- [vision-stage-contracts.md](docs/vision-stage-contracts.md)
- [validator-error-codes.md](docs/validator-error-codes.md)

## 架构概览

```text
┌──────────────────────────────────────────────────────────────┐
│ FastAPI / WebSocket API                                     │
│ classroom.py  pipeline.py  angnt.py  websocket.py           │
├──────────────────────────────────────────────────────────────┤
│ Services                                                    │
│ pipeline_service  guidance_service  rag_service             │
│ agent_service    version_service   classroom_state          │
├──────────────────────────────────────────────────────────────┤
│ Domain                                                      │
│ board_schema  circuit  netlist_models  validator            │
│ polarity      risk     ic_models                            │
├──────────────────────────────────────────────────────────────┤
│ Pipeline                                                    │
│ s1_detect -> s1b_pin_detect -> s2_mapping                    │
│              -> s3_topology -> s4_validate                  │
│ topology_input.py 负责结构化输入归一化                      │
├──────────────────────────────────────────────────────────────┤
│ Infra                                                       │
│ Redis  Celery  Docker Compose  手工 smoke fixtures          │
└──────────────────────────────────────────────────────────────┘
```

## 视觉检测流程

当前视觉链已经固定为“两阶段检测 + 一阶段映射”:

```text
top / left / right 图片
-> S1: component detect (YOLO-OBB)
-> S1.5: component ROI pin detect (YOLO-Pose 接口预留)
-> S2: pin keypoint -> hole_id / electrical_node_id
-> S3: topology / netlist_v2
-> S4: validator_report_v2 / risk
```

### S1: 组件检测

对应文件:

- [app/pipeline/stages/s1_detect.py](app/pipeline/stages/s1_detect.py)
- [app/pipeline/vision/detector.py](app/pipeline/vision/detector.py)

职责:

- 使用 `top` 视图建立主实例
- 生成全局 `component_id`
- 输出 `component_type / package_type / pin_schema_id / bbox / orientation`
- 输出 `side recall candidates` 作为侧视图补召回候选

当前约束:

- `top` 是主实例化入口
- `left_front / right_front` 当前只做候选补召回, 不直接进入主实例列表

### S1.5: 组件 ROI 引脚检测

对应文件:

- [app/pipeline/stages/s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py)
- [app/pipeline/vision/pin_model.py](app/pipeline/vision/pin_model.py)
- [app/pipeline/vision/roi_cropper.py](app/pipeline/vision/roi_cropper.py)
- [app/pipeline/vision/pin_schema.py](app/pipeline/vision/pin_schema.py)

职责:

- 根据 `component_id + bbox + package_type` 为每个元件建立 ROI
- ROI 裁剪按封装模板执行，不同封装使用不同覆盖范围
- 对每个视图分别执行 pin detector
- 输出 ordered `pins[]`
- 为每个 pin 保留:
  - `keypoints_by_view`
  - `visibility_by_view`
  - `score_by_view`
  - `source_by_view`
  - `roi_by_view`

当前状态:

- `top` 视图 ROI 来源为 `detected_bbox`
- 侧视图 ROI 现在优先尝试 `associated_bbox_candidate`
- 未命中侧视图候选时，才回退到 `shared_bbox_fallback`
- ROI 元数据会继续保留:
  - `crop_source`
  - `crop_profile`
  - `crop_bounds`
  - `association`
- `PinRoiDetector` 已有正式 `YOLO-Pose` 接口, 真实模型尚未接入时会走 `heuristic_fallback`

### S2: 孔位映射与多视图证据整理

对应文件:

- [app/pipeline/stages/s2_mapping.py](app/pipeline/stages/s2_mapping.py)
- [app/pipeline/vision/calibrator.py](app/pipeline/vision/calibrator.py)
- [app/pipeline/vision/image_io.py](app/pipeline/vision/image_io.py)

职责:

- 使用校准器把 pin keypoint 映射到 `hole_id`
- 进一步映射到 `electrical_node_id`
- 按多视图 observation 做 hole 加权投票
- 生成:
  - `candidate_hole_ids`
  - `candidate_node_ids`
  - `observations`
  - `is_ambiguous`
  - `ambiguity_reasons`
- 保留上游来源信息, 明确区分:
  - `model`
  - `heuristic_fallback`
  - `shared_bbox_fallback`
  - `synthetic_grid`
- 保留投票元数据:
  - `vote_scores`
  - `selected_by`

### S3: 拓扑与网表构建

对应文件:

- [app/pipeline/stages/s3_topology.py](app/pipeline/stages/s3_topology.py)
- [app/pipeline/topology_input.py](app/pipeline/topology_input.py)
- [app/domain/circuit.py](app/domain/circuit.py)
- [app/domain/board_schema.py](app/domain/board_schema.py)

职责:

- 消费 `components[].pins[]`
- 构建 `topology_graph`
- 导出 `netlist_v2`
- 保持主语义:

```text
component_id + pin_name + hole_id
-> electrical_node_id
-> electrical_net_id
```

### S4: 校验、诊断、风险分级

对应文件:

- [app/pipeline/stages/s4_validate.py](app/pipeline/stages/s4_validate.py)
- [app/domain/validator.py](app/domain/validator.py)
- [app/domain/risk.py](app/domain/risk.py)

职责:

- 与 `labguardian_ref_v4` 参考电路比较
- 输出 `validator_report_v2`
- 生成:
  - `error_code`
  - `suggested_action`
  - `evidence_refs`
  - `risk_level`
  - `risk_reasons`

## 完整工作流程

从一次请求进入到最终结果返回, 当前工作流可以按下面理解:

### 1. API / Service 入口

对应文件:

- [app/api/v1/pipeline.py](app/api/v1/pipeline.py)
- [app/services/pipeline_service.py](app/services/pipeline_service.py)
- [app/worker/tasks.py](app/worker/tasks.py)

职责:

- 接收 `station_id / images_b64 / rail_assignments / reference_circuit`
- 同步执行或异步提交
- 统一把原始阶段结果整理成 `PipelineResult`

### 2. Pipeline 编排

对应文件:

- [app/pipeline/orchestrator.py](app/pipeline/orchestrator.py)

职责:

- 为每次请求创建独立 `BreadboardCalibrator`
- 共享可复用模型对象:
  - `ComponentDetector`
  - `PinRoiDetector`
- 依次调度:
  - `run_detect()`
  - `run_pin_detect()`
  - `run_mapping()`
  - `run_topology()`
  - `run_validate()`

### 3. 结果落到服务层与课堂态

对应文件:

- [app/services/classroom_state.py](app/services/classroom_state.py)
- [app/services/guidance_service.py](app/services/guidance_service.py)
- [app/services/version_service.py](app/services/version_service.py)
- [app/services/rag_service.py](app/services/rag_service.py)
- [app/services/agent_service.py](app/services/agent_service.py)

职责:

- 更新课堂态、缩略图、指导历史
- 暴露 `/version`
- 为后续 RAG / agent 保留结构化证据入口

### 4. 面向前端 / agent 的最终结果

当前正式主输出已经收束为:

- `PipelineResult`
- `netlist_v2`
- `validator_report_v2`

这些结构是后续前端联调、指导下发、RAG / agent 的共同基础。

核心职责约定：

- `api/` 只做协议入口，不做领域推理
- `services/` 做编排、审计、下发、任务管理
- `domain/` 放稳定规则和核心模型
- `pipeline/` 只输出结构化事实，不直接生成教学话术

## 当前数据主线

### 当前主链

```text
S1 检测
-> S1.5 component ROI pin detect
-> S2 components[].pins[]
-> topology_input.normalize_components_for_topology()
-> CircuitAnalyzer(board_schema=...)
-> export_netlist_v2()
-> validator_report_v2
```

新链路里最重要的语义是：

```text
component_id + pin_name + hole_id
-> electrical_node_id
-> electrical_net_id
```

## 文件职责速查

如果团队成员只想快速定位改动入口, 可以直接按下面分工看文件:

| 目标 | 首先看哪些文件 |
|---|---|
| 接入 `YOLO-OBB` 组件检测模型 | [app/pipeline/vision/detector.py](app/pipeline/vision/detector.py), [app/pipeline/stages/s1_detect.py](app/pipeline/stages/s1_detect.py) |
| 接入 `YOLO-Pose` 引脚检测模型 | [app/pipeline/vision/pin_model.py](app/pipeline/vision/pin_model.py), [app/pipeline/stages/s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py) |
| 修改 ROI 裁剪或多视图 ROI 来源 | [app/pipeline/vision/roi_cropper.py](app/pipeline/vision/roi_cropper.py), [app/pipeline/stages/s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py) |
| 修改 pin schema / 封装默认规则 | [app/pipeline/vision/pin_schema.py](app/pipeline/vision/pin_schema.py) |
| 修改孔位映射 / ambiguity / calibration 输出 | [app/pipeline/stages/s2_mapping.py](app/pipeline/stages/s2_mapping.py), [app/pipeline/vision/calibrator.py](app/pipeline/vision/calibrator.py) |
| 修改 netlist / topology / board schema | [app/pipeline/topology_input.py](app/pipeline/topology_input.py), [app/domain/circuit.py](app/domain/circuit.py), [app/domain/board_schema.py](app/domain/board_schema.py) |
| 修改 compare / diagnose / error code | [app/domain/validator.py](app/domain/validator.py), [app/pipeline/stages/s4_validate.py](app/pipeline/stages/s4_validate.py) |
| 修改 API / worker / 结果封装 | [app/api/v1/pipeline.py](app/api/v1/pipeline.py), [app/services/pipeline_service.py](app/services/pipeline_service.py), [app/worker/tasks.py](app/worker/tasks.py) |

## 目录结构

```text
app/
├── main.py
├── core/
│   ├── config.py
│   ├── celery_app.py
│   └── deps.py
├── api/v1/
│   ├── angnt.py
│   ├── classroom.py
│   ├── pipeline.py
│   └── websocket.py
├── domain/
│   ├── board_schema.py
│   ├── circuit.py
│   ├── ic_models.py
│   ├── netlist_models.py
│   ├── polarity.py
│   ├── risk.py
│   ├── validator.py
│   └── data/board_schemas/
├── pipeline/
│   ├── orchestrator.py
│   ├── topology_input.py
│   ├── stages/
│   └── vision/
├── schemas/
├── services/
└── worker/

docs/
├── backend-architecture.md
├── board-schema-format.md
├── current-status.md
└── validator-error-codes.md

scripts/manual/
└── tools/
    └── circuit/

tests/
├── fixtures/
│   ├── netlist_v2/
│   └── validator_error_codes/
├── pipeline/
└── manual/
    └── smoke/
```

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 启动 Redis
docker compose up -d redis

# 3. 启动 Celery Worker
celery -A app.core.celery_app:celery_app worker -Q pipeline -c 1 --loglevel=info

# 4. 启动 FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 常用验证命令

```bash
# board schema 默认映射冒烟
python3 tests/manual/smoke/test_board_schema_default.py

# reference v4 读写与 compare 冒烟
python3 tests/manual/smoke/test_reference_v4_roundtrip.py

# validator error code regression
python3 tests/manual/smoke/test_validator_error_codes.py

# pipeline 合同与阶段级回归
python3 -m pytest tests/pipeline
```

## 团队协作约定

- 先看文档再下手改迁移链路，尤其是 `board_schema`、`topology_input`、`validator`
- 新逻辑优先写到新链路，不再新增只服务旧 `pin1/pin2` 的结构
- 不再为旧 `pin1_logic/pin2_logic` 新增兼容入口
- S1 / S1.5 / S2 的 JSON 契约优先保持稳定，模型训练完成后尽量只替换推理内核
- fallback 必须显式标记来源，不要伪装成真实模型输出
- 新增手工脚本放 `scripts/manual/tools/`
- 新增回归样例优先补到 `tests/fixtures/` 和 `tests/manual/smoke/`
- 阶段级回归优先补到 `tests/pipeline/`
- PCB / AOI 相关代码已彻底移除，后续不要再向仓库重新引入平行子系统

## 近期最值得关注的文件

- [s2_mapping.py](app/pipeline/stages/s2_mapping.py)
  - S2 从 pin 预测结果到 `hole_id` / `components[].pins[]` 的核心迁移点
- [s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py)
  - 组件 ROI pin 检测阶段骨架
- [pin_model.py](app/pipeline/vision/pin_model.py)
  - 第二个视觉模型的接入位置
- [topology_input.py](app/pipeline/topology_input.py)
  - 结构化 `components[].pins[]` 到 analyzer 的统一入口
- [circuit.py](app/domain/circuit.py)
  - `board_schema` + `netlist_v2` 核心落点
- [validator.py](app/domain/validator.py)
  - compare / diagnose / error code / guidance 证据入口

## 迁移对照表

| 原文件/目录 | 目标位置 | 当前状态 |
|---|---|---|
| `teacher/server.py` | `app/api/v1/classroom.py` | 已拆成 API 路由 |
| `teacher/classroom.py` | `app/services/classroom_state.py` | 已保留状态层 |
| `shared/models.py` | `app/schemas/*` | 已拆到 schema 层 |
| `shared/risk.py` | `app/domain/risk.py` | 已迁移 |
| `src_v2/vision/image_analyzer.py` | `app/pipeline/orchestrator.py` | 已拆成四阶段 |
| `src_v2/vision/*` | `app/pipeline/vision/*` | 继续复用 |
| `src_v2/logic/circuit.py` | `app/domain/circuit.py` | 已切到 `ComponentInstance + netlist_v2` 主语义 |
| `src_v2/logic/validator.py` | `app/domain/validator.py` | 已切到 `labguardian_ref_v4 + validator_report_v2` |
