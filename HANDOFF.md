# LabGuardian Server 交接指南

## 1. 仓库边界

本仓库包含 FastAPI 服务、视觉 pipeline、面包板孔位映射、逻辑网表比较、
Agent/RAG 服务、测试和部分可直接运行的演示权重。配套单工位界面位于
`LabGuardian-Web`。

核心事实链：

```text
图片 -> 元件检测 -> 引脚检测 -> hole_id -> electrical_node_id
     -> electrical_net_id -> netlist_v2 -> 参考电路比较 -> 诊断与解释
```

## 2. 首次接手验收

建议使用 Python 3.11 与 `uv`：

```bash
git clone https://github.com/liusuan110/LabGuardian-Server.git
cd LabGuardian-Server
uv sync --locked --extra dev
cp .env.example .env
uv run pytest
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

另开终端检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/version
```

默认测试不包含必须成功加载真实 OpenVINO IR 的环境测试。目标 Intel 运行环境准备好后，
再执行 `uv run pytest -m openvino_runtime`。

## 3. 运行模式

| 模式 | Redis/Celery | 外部模型资产 | 用途 |
|---|---:|---:|---|
| API + 同步 `pipeline/run` | 否 | 视觉主链需要 | 本地联调、前端演示 |
| 异步 `pipeline/submit` | 是 | 视觉主链需要 | Worker 队列 |
| `AGENT_LLM_PROVIDER=template` | 否 | 否 | 可复现的规则化回答 |
| Ollama Agent | 否 | 需本机 Ollama 模型 | 大模型解释 |
| OpenVINO embedding / VLM | 否 | 需单独 IR 目录 | Intel 板端验证 |

## 4. Git 内已有与仓库外资产

Git 中已有可回退的演示视觉权重，主要位于 `train_demo/`。以下内容被 `.gitignore`
排除，不能假设新成员 clone 后自动拥有：

- `models/`：正式部署模型与 OpenVINO IR。
- `checkpoints/`：训练检查点。
- 大部分 `datasets/`：训练/评测数据。
- `knowledge/datasheets/embeddings/`：OpenVINO datasheet embedding 缓存。
- `outputs/demo_real/`：真实演示照片、网表、出图脚本和答辩图，属于交接证据而非临时缓存。
- `.env`：机器路径、端口和 provider 配置。
- 本机 Redis 数据、Ollama 模型、板卡驱动与 NPU/GPU runtime。

交接时应把这些资产放入团队网盘或对象存储，并附 SHA-256、来源、适用代码版本和
目标目录。不要把密钥写入资产清单。

2026-08-20 本机待外部交接资产约为：`datasets/` 469 MB、`models/` 24 MB、
`checkpoints/` 8.9 MB、`outputs/demo_real/` 20 MB、datasheet embeddings 152 KB。
这些目录均已审核为有用内容并保留，但不会随 Git clone 自动到达新成员机器。

Docker Compose 固定挂载 `./models:/app/models:ro`，因此 fresh clone 在执行
`docker compose up` 前必须先准备 `models/component/best.pt` 与
`models/pin/best.pt`，或者修改为团队实际采用的模型目录。

## 5. 代码入口

| 修改目标 | 入口 |
|---|---|
| 服务配置与模型选择 | `app/core/config.py` |
| HTTP / WebSocket API | `app/api/v1/` |
| S1-S4 编排 | `app/pipeline/orchestrator.py`, `app/pipeline/stages/` |
| 孔位、电气节点与网表 | `app/domain/board_schema.py`, `app/domain/circuit.py` |
| 参考电路与逻辑比较 | `app/domain/dsl/`, `app/domain/compare/` |
| Agent / 检索契约 | `app/agent/`, `docs/retrieval-contract.md` |
| 模型与部署说明 | `docs/vision-model-inventory.md`, `docs/edge-deployment.md` |

## 6. 配置与安全

- 只提交 `.env.example`；`.env` 已忽略。
- 交接前在共享平台重新签发 API key，不通过聊天或 Git 传递旧密钥。
- 默认 `CORS_ORIGINS=["*"]` 只适合内网演示；部署到公网前必须收窄。
- `/api/v1/kb/upload`、调试接口和课堂控制接口目前没有面向公网的认证边界，不能直接暴露。
- 当前仓库没有许可证文件；对外公开或商业复用前由项目负责人确定授权方式。

## 7. Git 与协作约定

1. 新成员从 `main` 建功能分支，不直接在演示机上改 `main`。
2. 协议变更必须同时更新类型、文档和测试；细则见 `docs/README.md`。
3. 模型替换必须记录文件哈希、训练数据版本、推理尺寸和验收样例。
4. 合并前至少运行 `uv run pytest`，前后端联调改动还要运行前端 `npm run build`。
5. GitHub 权限、分支保护、Issue/里程碑和大文件下载权限需由仓库所有者另行移交。

## 8. 大创阶段建议的第一批 Issue

- 固化一套不含个人路径的端到端验收样例及预期 JSON。
- 为模型/数据建立带哈希的版本清单和统一下载入口。
- 将 Intel 目标机 OpenVINO 测试接入专用 runner，而不是普通 Mac/云端单元测试。
- 建立 CI；当前仓库尚无 GitHub Actions 工作流。
- 分批清理 Ruff 历史基线（当前全仓检查有大量既存格式问题），不要一次机械改写全部文件。
- 明确公网部署的鉴权、CORS、上传大小与日志脱敏策略。
- 确认开源许可证、数据集授权与比赛资料可复用范围。

## 9. 2026-08-20 交接整理基线

- `uv lock --check`：通过。
- 默认自动测试：596 passed、6 skipped、2 deselected。
- `/health` 与 `/version`：均返回 HTTP 200。
- 两个 deselected 测试是 `openvino_runtime`，必须在兼容的 Intel/OpenVINO 环境复验。
- 全仓 Ruff 尚未达到 clean；这是既存技术债，不影响上述自动测试结果。
- 本机没有 Docker 命令，因此 `docker compose` 尚未实机验证；接手者需在装有 Docker 的机器上复验。
