# Telemetry Protocol (Phase 5)

实时硬件遥测协议，给 DK-2500 (Intel Core Ultra 5 + iGPU + NPU) 上跑 LabGuardian
时把 CPU / 内存 / iGPU / NPU 占用、功耗、当前 pipeline stage 推到任何 WebSocket
订阅者。

后端只负责采样与推流；前端 demo 页 / 论文截图 / dashboard 由前端独立维护。

## 端点

| 类型      | 路径                          | 用途                       |
|----------|-------------------------------|---------------------------|
| WS Push  | `/ws/telemetry/system`        | 5Hz 实时帧推送 (主通道)    |
| REST     | `/api/v1/telemetry/latest`    | 拉最新一帧 (smoke / curl)  |

WS 推流频率由配置项 `TELEMETRY_HZ` 决定 (默认 5.0 Hz)。

## 帧 Schema (`telemetry_frame_v1`)

```json
{
  "frame_version": "telemetry_frame_v1",
  "ts": 1715000000.123,

  "cpu_pct": 12.4,
  "mem_used_mb": 2048.7,
  "mem_total_mb": 16384.0,

  "igpu_pct": 5.1,
  "igpu_freq_mhz": 1100.0,

  "npu_pct": null,
  "npu_power_mw": null,

  "pipeline_stage": "S2",

  "sampler_status": {
    "cpu":  "ok",
    "igpu": "ok",
    "npu":  "unavailable"
  }
}
```

字段语义：

- `frame_version` — 协议版本，目前固定 `telemetry_frame_v1`
- `ts` — Unix 秒, float
- `cpu_pct` / `mem_*` — psutil 直接读取
- `igpu_pct` — 由 `/sys/class/drm/card0/engine/*/busy` 多引擎累计差分得到，可能 > 100% (多引擎并行)
- `igpu_freq_mhz` — `/sys/class/drm/card0/gt_cur_freq_mhz`
- `npu_pct` / `npu_power_mw` — Phase 5 仅探测 `/sys/class/intel_npu/`, DK-2500 NPU 验证后 (Phase 7+) 完善
- `pipeline_stage` — 由 `TelemetryService.mark_stage()` 写入，便于把 pipeline 阶段叠到功耗曲线上
- `sampler_status` — `ok | degraded | unavailable | error`，前端 UI 据此决定是否点亮该 lane

任意"硬件不存在"字段为 `null`，前端必须容忍。

## 配置

```env
TELEMETRY_ENABLED=true
TELEMETRY_HZ=5.0
TELEMETRY_RING_SECONDS=120
```

`TELEMETRY_ENABLED=false` 时：lifespan 不启动后台任务，WS 端点关闭连接 (code=1013, reason=`telemetry_disabled`)。

## 快速验证

```bash
# 1. 启动服务
TELEMETRY_ENABLED=true uvicorn app.main:app --port 8000

# 2. REST 拉最新一帧
curl -s http://localhost:8000/api/v1/telemetry/latest | jq

# 3. WebSocket 实时流 (用 websocat)
websocat ws://localhost:8000/ws/telemetry/system | head -10

# 4. WebSocket 实时流 (用 wscat)
wscat -c ws://localhost:8000/ws/telemetry/system
```

期望首帧含 `frame_version: "telemetry_frame_v1"` 和 `sampler_status` 三键。

## 与 Pipeline 集成

视觉/拓扑 pipeline 在节点边界调用 `mark_stage()` 即可把当前阶段名写入下一帧：

```python
from app.services.telemetry import get_telemetry_service

telemetry = get_telemetry_service()
telemetry.mark_stage("S2")
# ... run S2 mapping ...
telemetry.mark_stage("S3")
```

零异步、零阻塞，适合插在 orchestrator 的同步代码里。

## 降级行为

- 任何 sampler 内部异常 → 该字段返回 `None`，service 继续运行
- macOS / 无 `/sys` 容器 → `igpu` + `npu` 标 `unavailable`
- WebSocket 客户端慢 → 自身订阅队列 (maxsize=16) 满后丢最老帧，**不阻塞采样器**
- 10 秒无新帧推送 → 发送 `{"frame_version": "telemetry_frame_v1", "heartbeat": true}` 心跳，便于客户端检测连接活性

## 性能预算

- 5Hz 采样目标 CPU 自身占用 < 2% (DK-2500)
- 单订阅者 RAM 占用 ≈ 16 帧 × < 1KB = 可忽略
- 环形缓冲固定 `TELEMETRY_RING_SECONDS × TELEMETRY_HZ` 帧 (默认 600 帧 ≈ 600 KB)

## 后续 Phase 7+ 计划

- DK-2500 NPU 驱动验证后细化 `samplers/npu.py` 的 sysfs 路径与 OpenVINO Core 属性
- 加 `intel_gpu_top -J` fallback (需 CAP_PERFMON 容器权限)
- 加历史压缩 / Prometheus exporter 双通道 (短期 5Hz WS + 长期 1Hz Prom)
