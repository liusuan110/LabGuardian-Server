# LabGuardian Server

> 统一后端：检测 → 映射 → 拓扑 → 检错 四阶段 Pipeline

基座模板：
- [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template) (backend 结构)
- [GregaVrbancic/fastapi-celery](https://github.com/GregaVrbancic/fastapi-celery) (异步任务)

## 架构

```
┌───────────────────────────────────────────────────────────┐
│                     FastAPI Server                        │
│   api/v1/classroom.py   api/v1/pipeline.py   ws/station  │
├───────────────────────────────────────────────────────────┤
│                     Services                              │
│   classroom_state.py          pipeline orchestrator       │
├───────────────────────────────────────────────────────────┤
│                   Celery Workers                          │
│   S1:Detect → S2:Mapping → S3:Topology → S4:Validate     │
├───────────────────────────────────────────────────────────┤
│                      Domain                               │
│   circuit.py   validator.py   polarity.py   risk.py       │
├───────────────────────────────────────────────────────────┤
│               Pipeline / Vision                           │
│   detector   calibrator   wire_analyzer   pin_utils       │
└───────────────────────────────────────────────────────────┘
```

## 目录结构

```
app/
├── main.py                  # FastAPI 入口
├── core/
│   ├── config.py            # Pydantic Settings
│   ├── celery_app.py        # Celery 实例
│   └── deps.py              # FastAPI 依赖注入
├── api/v1/
│   ├── classroom.py         # 教室 REST API (← teacher/server.py)
│   ├── pipeline.py          # Pipeline 任务提交 API
│   └── websocket.py         # WebSocket 端点
├── schemas/
│   ├── classroom.py         # 数据模型 (← shared/models.py)
│   └── pipeline.py          # Pipeline 请求/响应
├── services/
│   └── classroom_state.py   # 课堂状态管理 (← teacher/classroom.py)
├── domain/
│   ├── risk.py              # 风险分级 (← shared/risk.py)
│   ├── circuit.py           # 电路建模 (← src_v2/logic/circuit.py)
│   ├── validator.py         # 电路验证 (← src_v2/logic/validator.py)
│   └── polarity.py          # 极性推断 (← src_v2/logic/polarity.py)
├── pipeline/
│   ├── orchestrator.py      # 四阶段调度 (← image_analyzer.py)
│   ├── stages/
│   │   ├── s1_detect.py     # YOLO 检测
│   │   ├── s2_mapping.py    # 坐标映射
│   │   ├── s3_topology.py   # 拓扑建模
│   │   └── s4_validate.py   # 检错诊断
│   └── vision/              # 视觉模块 (← src_v2/vision/)
│       ├── detector.py
│       ├── calibrator.py
│       ├── wire_analyzer.py
│       ├── pin_hole_detector.py
│       ├── pin_utils.py
│       └── stabilizer.py
└── worker/
    └── tasks.py             # Celery 任务定义
```

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 启动 Redis
docker compose up -d redis

# 3. 启动 Celery Worker
celery -A app.worker.tasks worker --loglevel=info

# 4. 启动 FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## 迁移对照表

| 原文件/目录 | 目标位置 | 处理方式 |
|---|---|---|
| `teacher/server.py` | `app/api/v1/classroom.py` | 拆分路由到 APIRouter |
| `teacher/classroom.py` | `app/services/classroom_state.py` | 直接迁移 |
| `shared/models.py` | `app/schemas/classroom.py` | 迁移 + OpenAPI 生成客户端 |
| `shared/risk.py` | `app/domain/risk.py` | 直接迁移 |
| `src_v2/vision/image_analyzer.py` | `app/pipeline/orchestrator.py` | 拆成四阶段调度 |
| `src_v2/vision/*` | `app/pipeline/vision/*` | 按模块复用 |
| `src_v2/logic/*` | `app/domain/*` | 拓扑+检错核心复用 |
