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
- S2 开始原生输出 `components[].pins[]`
- S3 / S4 / validator 已开始消费新结构
- `circuit.py` 内部主逻辑已切到 `ComponentInstance + pins[]`
- `ic_models.py / polarity.py` 已切到新语义，不再依赖旧内部组件对象
- `validator_report_v2` 已支持结构化 `error_code + suggested_action + evidence_refs`
- 已补一组最小 regression fixture 与 smoke tests

建议先读这几份文档：

- [README.md](/Users/liusuan/Desktop/LabGuardian-Server/README.md)
- [backend-architecture.md](/Users/liusuan/Desktop/LabGuardian-Server/docs/backend-architecture.md)
- [current-status.md](/Users/liusuan/Desktop/LabGuardian-Server/docs/current-status.md)
- [board-schema-format.md](/Users/liusuan/Desktop/LabGuardian-Server/docs/board-schema-format.md)
- [validator-error-codes.md](/Users/liusuan/Desktop/LabGuardian-Server/docs/validator-error-codes.md)

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
│ s1_detect -> s2_mapping -> s3_topology -> s4_validate       │
│ topology_input.py 负责结构化输入归一化                      │
├──────────────────────────────────────────────────────────────┤
│ Infra                                                       │
│ Redis  Celery  Docker Compose  手工 smoke fixtures          │
└──────────────────────────────────────────────────────────────┘
```

核心职责约定：

- `api/` 只做协议入口，不做领域推理
- `services/` 做编排、审计、下发、任务管理
- `domain/` 放稳定规则和核心模型
- `pipeline/` 只输出结构化事实，不直接生成教学话术

## 当前数据主线

### 当前主链

```text
S1 检测
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

tests/
├── fixtures/
│   ├── netlist_v2/
│   └── validator_error_codes/
└── manual/
    ├── research/
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
```

## 团队协作约定

- 先看文档再下手改迁移链路，尤其是 `board_schema`、`topology_input`、`validator`
- 新逻辑优先写到新链路，不再新增只服务旧 `pin1/pin2` 的结构
- 不再为旧 `pin1_logic/pin2_logic` 新增兼容入口
- 新增手工脚本放 `scripts/manual/tools/`
- 新增回归样例优先补到 `tests/fixtures/` 和 `tests/manual/smoke/`

## 近期最值得关注的文件

- [s2_mapping.py](/Users/liusuan/Desktop/LabGuardian-Server/app/pipeline/stages/s2_mapping.py)
  - S2 从像素结果到 `components[].pins[]` 的核心迁移点
- [topology_input.py](/Users/liusuan/Desktop/LabGuardian-Server/app/pipeline/topology_input.py)
  - 结构化 `components[].pins[]` 到 analyzer 的统一入口
- [circuit.py](/Users/liusuan/Desktop/LabGuardian-Server/app/domain/circuit.py)
  - `board_schema` + `netlist_v2` 核心落点
- [validator.py](/Users/liusuan/Desktop/LabGuardian-Server/app/domain/validator.py)
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
