# LabGuardian Server

LabGuardian 的服务器端负责把视觉识别结果转换成可验证、可解释、可审计的电路诊断结果。

> 新成员请先阅读 [HANDOFF.md](HANDOFF.md)，再按 [docs/README.md](docs/README.md)
> 进入具体设计文档。交接指南区分了 Git 内代码与需要单独交付的模型、数据和板端环境。

当前主线已经从早期 demo 形态迁移到新的结构化网表链路：

- `component_id + pin_name + hole_id -> electrical_node_id -> electrical_net_id -> netlist_v2`

后续比赛、RAG、PCM Agent、指导下发和论文实验都会优先基于这条事实链继续演进。

基座模板：
- [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)
- [GregaVrbancic/fastapi-celery](https://github.com/GregaVrbancic/fastapi-celery)

## 当前状态

当前仓库已经完成的关键迁移：

- 服务层骨架已经建立：`pipeline_service / guidance_service / version_service / rag_service / agent_service`
- 新网表模型已经落地：`netlist_v2`
- 比赛板 `board_schema` 已接入默认加载流程
- pipeline 已切到 `S1 component detect -> S1.5 full-image pose pin detect -> S2 hole mapping`
- 当前组件检测主路径已经收口为 `YOLO-Detect`，`OBB` 仅保留兼容解析能力
- S1 现在固定由 `top` 视图建立全局 `component_id`
- S1 已支持 `side recall candidates` 输出，但 side 候选当前不直接进入主实例链
- S1.5 正式主路径已切到 `full-image YOLO-Pose`
- ROI 裁切 pin 检测逻辑不再参与默认主流程，仅作为 legacy fallback 保留
- S2 开始原生输出 `components[].pins[]`
- S3 / S4 / validator 已开始消费新结构
- `circuit.py` 内部主逻辑已切到 `ComponentInstance + pins[]`
- `ic_models.py / polarity.py` 已切到新语义，不再依赖旧内部组件对象
- `validator_report_v2` 已支持结构化 `error_code + suggested_action + evidence_refs`
- 模型路径、默认 `imgsz=960`、board schema 分段和 `runtime_metadata` 已完成第一轮 edge 收口
- `app/agent/` 已落地 PCM Agent 第一版白盒基建：
  - `RuntimeEvidence`
  - `ContextPack`
  - error family 路由
  - deterministic tools
  - LangGraph state machine shell
  - Reflection / verifier node
- 已补一组最小 regression fixture 与 smoke tests

建议先读这几份文档：

- [README.md](README.md)
- [docs/README.md](docs/README.md)
- [comparison-architecture.md](docs/comparison-architecture.md)
- [development-roadmap.md](docs/development-roadmap.md)
- [vision-model-inventory.md](docs/vision-model-inventory.md)
- [board-schema-format.md](docs/board-schema-format.md)
- [edge-deployment.md](docs/edge-deployment.md)
- [pcm-agent-architecture.md](docs/pcm-agent-architecture.md)
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
│ Agent / Knowledge                                           │
│ RuntimeEvidence  ContextPack  tools  verifier  MRAG/VLM     │
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
-> S1: component detect (YOLO-Detect)
-> S1.5: full-image pose pin detect (YOLO-Pose, fallback 显式标记)
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

### S1.5: 全图引脚检测

对应文件:

- [app/pipeline/stages/s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py)
- [app/pipeline/vision/pin_model.py](app/pipeline/vision/pin_model.py)
- [app/pipeline/vision/roi_cropper.py](app/pipeline/vision/roi_cropper.py)
- [app/pipeline/vision/pin_schema.py](app/pipeline/vision/pin_schema.py)

职责:

- 在 `top` 整图上直接执行 `YOLO-Pose`
- 将 pose 实例按 `component_type + bbox` 几何关系关联回 S1 组件
- 输出 ordered `pins[]`
- 为每个 pin 保留:
  - `keypoints_by_view`
  - `visibility_by_view`
  - `score_by_view`
  - `source_by_view`
  - `roi_by_view`

当前状态:

- 当前主语义:
  - `top` 视图整图 pose 为唯一默认 pin 来源
  - `left_front / right_front` 暂不参与默认 pin 主判定
- `PinRoiDetector` 已能加载真实 `YOLO-Pose` 权重；无模型或测试 mock 时才显式走 legacy fallback

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

协作说明:

- 面包板网格化 / `calibrator.py` 当前由队友继续推进
- 本仓库其他阶段继续以 `calibrator` 提供的 `frame_pixel_to_logic_candidates()` 为边界消费结果
- 不建议在 S1/S1.5/S3/S4 中绕过 S2 直接推断 `hole_id`

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
- [app/domain/compare/](app/domain/compare)
- [app/domain/risk.py](app/domain/risk.py)

职责:

- 与 `logical_reference_v1` 参考电路做逻辑图比较
- 从参考电路和当前 netlist 构建逻辑图, 通过 `compare_logical_graphs` 输出比较结论
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
- [app/agent/contracts.py](app/agent/contracts.py)
- [app/agent/context_pack.py](app/agent/context_pack.py)
- [app/agent/tools.py](app/agent/tools.py)

职责:

- 更新课堂态、缩略图、指导历史
- 暴露 `/version`
- 将 pipeline 事实转成 RAG / PCM Agent 可消费的最小证据

### 4. 面向前端 / agent 的最终结果

当前正式主输出已经收束为:

- `PipelineResult`
- `netlist_v2`
- `validator_report_v2`
- `runtime_metadata`
- `RuntimeEvidence`
- `ContextPack`

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
-> S1.5 full-image pose pin detect
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
| 接入 `YOLO-Detect` 组件检测模型 | [app/pipeline/vision/detector.py](app/pipeline/vision/detector.py), [app/pipeline/stages/s1_detect.py](app/pipeline/stages/s1_detect.py) |
| 接入 `YOLO-Pose` 引脚检测模型 | [app/pipeline/vision/pin_model.py](app/pipeline/vision/pin_model.py), [app/pipeline/stages/s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py) |
| 修改 ROI 裁剪或多视图 ROI 来源 | [app/pipeline/vision/roi_cropper.py](app/pipeline/vision/roi_cropper.py), [app/pipeline/stages/s1b_pin_detect.py](app/pipeline/stages/s1b_pin_detect.py) |
| 修改 pin schema / 封装默认规则 | [app/pipeline/vision/pin_schema.py](app/pipeline/vision/pin_schema.py) |
| 修改孔位映射 / ambiguity / calibration 输出 | [app/pipeline/stages/s2_mapping.py](app/pipeline/stages/s2_mapping.py), [app/pipeline/vision/calibrator.py](app/pipeline/vision/calibrator.py) |
| 修改 netlist / topology / board schema | [app/pipeline/topology_input.py](app/pipeline/topology_input.py), [app/domain/circuit.py](app/domain/circuit.py), [app/domain/board_schema.py](app/domain/board_schema.py) |
| 修改 compare / diagnose / error code | [app/domain/compare/](app/domain/compare), [app/domain/logical_reference.py](app/domain/logical_reference.py), [app/pipeline/stages/s4_validate.py](app/pipeline/stages/s4_validate.py) |
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
│   ├── kb.py
│   ├── pipeline.py
│   └── websocket.py
├── agent/
│   ├── contracts.py
│   ├── context_pack.py
│   ├── evidence.py
│   ├── tools.py
│   └── verification.py
├── domain/
│   ├── board_schema.py
│   ├── circuit.py
│   ├── ic_models.py
│   ├── netlist_models.py
│   ├── polarity.py
│   ├── risk.py
│   ├── compare/
│   ├── dsl/
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
├── board-schema-format.md
├── comparison-architecture.md
├── development-roadmap.md
├── edge-deployment.md
├── pcm-agent-architecture.md
├── rag-teaching-kb-design.md
├── telemetry-protocol.md
├── vision-model-inventory.md
├── vision-stage-contracts.md
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
# 1. 创建锁定的开发环境（推荐）
uv sync --locked --extra dev

# 2. 复制本机配置；不要提交 .env
cp .env.example .env

# 3. 启动 Redis
docker compose up -d redis

# 4. 启动 Celery Worker（异步接口需要）
uv run celery -A app.core.celery_app:celery_app worker -Q pipeline -c 1 --loglevel=info

# 5. 启动 FastAPI
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

若未安装 `uv`，可使用 Python 3.11 和 `pip install -e ".[dev]"`，但该路径不会使用
`uv.lock` 固定的依赖版本。

## 常用验证命令

```bash
# board schema 默认映射冒烟
python3 tests/manual/smoke/test_board_schema_default.py

# 逻辑图比较与 S4 回归
python3 -m pytest tests/domain/test_graph_compare*.py tests/pipeline/test_s4_validate_logical.py -v

# DSL 参考电路加载与编译
python3 -m pytest tests/domain/dsl tests/domain/test_reference_service.py -q

# pipeline 合同与阶段级回归
python3 -m pytest tests/pipeline

# 默认自动测试（不包含必须加载真实 OpenVINO 模型的环境测试）
uv run pytest

# 在目标 OpenVINO 环境上额外执行
uv run pytest -m openvino_runtime
```

## 团队协作约定

- 先看文档再下手改迁移链路，尤其是 `board_schema`、`topology_input`、`validator`
- 新逻辑优先写到新链路，不再新增只服务旧 `pin1/pin2` 的结构
- 不再为旧 `pin1_logic/pin2_logic` 新增兼容入口
- S1 / S1.5 / S2 的 JSON 契约优先保持稳定，模型训练完成后尽量只替换推理内核
- fallback 必须显式标记来源，不要伪装成真实模型输出
- Agent / VLM 只消费结构化证据，不绕开 validator 重新判断事实
- PCM Agent 第一版优先保持规则化；LangGraph 壳子已接入，LLM 和 OpenVINO adapter 后续作为可选节点接入
- 新增手工脚本放 `scripts/manual/tools/`
- 新增回归样例优先补到 `tests/fixtures/` 和 `tests/manual/smoke/`
- 阶段级回归优先补到 `tests/pipeline/`
- PCB / AOI 相关代码已彻底移除，后续不要再向仓库重新引入平行子系统

## 下一步开发重点

- [app/services/agent_service.py](app/services/agent_service.py)
  - 已接入 `mode="diagnostic_agent"`，当前通过 LangGraph 壳子编排白盒链路
- [app/agent/](app/agent)
  - 下一步补 graph metrics、更多 golden tests 和可选 LLM node
- [app/services/rag_service.py](app/services/rag_service.py)
  - 与 PCM context pack 对齐 runtime / teaching / KB 三路检索
- [app/pipeline/stages/s2_mapping.py](app/pipeline/stages/s2_mapping.py)
  - 继续增强多视图证据和 ambiguity 元数据
- [app/pipeline/vision/calibrator.py](app/pipeline/vision/calibrator.py)
  - 继续优化实物板网格拟合和 pixel -> hole candidates
- [docs/development-roadmap.md](docs/development-roadmap.md)
  - 后续计划、论文实验和 edge benchmark 的总入口
