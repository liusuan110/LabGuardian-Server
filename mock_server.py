#!/usr/bin/env python3
"""
LabGuardian Mock Server — 模拟所有 API 端点, 用于客户端联调测试
仅依赖标准库 (http.server + json), 无需安装任何包
端口 8000, 路由前缀 /api/v1/
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Mock 数据 ────────────────────────────────────────

STATIONS = {
    "S01": {
        "station_id": "S01",
        "student_name": "张三",
        "timestamp": time.time(),
        "component_count": 5,
        "net_count": 3,
        "components": [
            {"name": "R1", "type": "resistor", "polarity": "none", "pin1": ["A1"], "pin2": ["A5"], "pin3": [], "confidence": 0.95},
            {"name": "LED1", "type": "led", "polarity": "forward", "pin1": ["B1"], "pin2": ["B3"], "pin3": [], "confidence": 0.90},
        ],
        "progress": 0.72,
        "similarity": 0.85,
        "match_level": "good",
        "missing_components": ["C1"],
        "diagnostics": ["R1 阻值偏差 5%", "LED1 极性正确"],
        "risk_level": "warning",
        "risk_reasons": ["缺少电容 C1"],
        "circuit_snapshot": "",
        "fps": 15.0,
        "detector_ok": "ok",
        "llm_backend": "tinyllama",
        "ocr_backend": "paddle",
        "thumbnail_b64": "",
    },
    "S02": {
        "station_id": "S02",
        "student_name": "李四",
        "timestamp": time.time(),
        "component_count": 6,
        "net_count": 4,
        "components": [],
        "progress": 0.95,
        "similarity": 0.97,
        "match_level": "excellent",
        "missing_components": [],
        "diagnostics": ["全部元器件就位", "电路拓扑正确"],
        "risk_level": "safe",
        "risk_reasons": [],
        "circuit_snapshot": "",
        "fps": 20.0,
        "detector_ok": "ok",
        "llm_backend": "tinyllama",
        "ocr_backend": "paddle",
        "thumbnail_b64": "",
    },
}

JOBS = {}
JOB_COUNTER = [0]

GUIDANCE_QUEUE = {}  # station_id -> [messages]


def mock_ranking():
    entries = []
    for i, (sid, st) in enumerate(sorted(STATIONS.items(), key=lambda x: -x[1]["progress"]), 1):
        entries.append({
            "rank": i,
            "station_id": sid,
            "student_name": st["student_name"],
            "progress": st["progress"],
            "similarity": st["similarity"],
            "elapsed_s": 120.0 + i * 30,
            "risk_event_count": len(st["risk_reasons"]),
            "component_count": st["component_count"],
            "risk_level": st["risk_level"],
            "online": True,
        })
    return entries


def mock_stats():
    return {
        "total_stations": len(STATIONS),
        "online_count": len(STATIONS),
        "completed_count": sum(1 for s in STATIONS.values() if s["progress"] >= 0.9),
        "avg_progress": sum(s["progress"] for s in STATIONS.values()) / max(len(STATIONS), 1),
        "total_risk_events": sum(len(s["risk_reasons"]) for s in STATIONS.values()),
        "danger_count": sum(1 for s in STATIONS.values() if s["risk_level"] == "danger"),
        "error_histogram": {"missing_component": 1},
        "session_duration_s": 600.0,
    }


def mock_alerts():
    alerts = []
    for st in STATIONS.values():
        if st["risk_level"] in ("warning", "danger"):
            alerts.append({
                "station_id": st["station_id"],
                "student_name": st["student_name"],
                "risk_level": st["risk_level"],
                "risk_reasons": st["risk_reasons"],
                "diagnostics": st["diagnostics"],
                "progress": st["progress"],
            })
    return alerts


def mock_pipeline_result(job_id, station_id):
    return {
        "job_id": job_id,
        "station_id": station_id,
        "status": "completed",
        "stages": [
            {"stage": "detect", "status": "completed", "duration_ms": 320.0, "data": {}, "errors": []},
            {"stage": "mapping", "status": "completed", "duration_ms": 150.0, "data": {}, "errors": []},
            {"stage": "topology", "status": "completed", "duration_ms": 200.0, "data": {}, "errors": []},
            {"stage": "validate", "status": "completed", "duration_ms": 80.0, "data": {}, "errors": []},
        ],
        "total_duration_ms": 750.0,
        "component_count": 5,
        "net_count": 3,
        "progress": 0.72,
        "similarity": 0.85,
        "diagnostics": ["R1 检测正常", "LED1 极性正确", "缺少 C1"],
        "risk_level": "warning",
        "risk_reasons": ["缺少电容 C1"],
        "report": "电路分析完成, 进度 72%, 相似度 85%",
    }


# ── HTTP Handler ─────────────────────────────────────

class MockHandler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return json.loads(self.rfile.read(length))
        return {}

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")

        if path == "/health":
            return self._json({"status": "ok"})

        if path == "/api/v1/classroom/stations":
            return self._json(STATIONS)

        if path == "/api/v1/classroom/ranking":
            return self._json(mock_ranking())

        if path == "/api/v1/classroom/alerts":
            return self._json(mock_alerts())

        if path == "/api/v1/classroom/stats":
            return self._json(mock_stats())

        if path == "/api/v1/classroom/reference":
            return self._json({"components": [], "nets": []})

        # /api/v1/classroom/station/{id}
        if path.startswith("/api/v1/classroom/station/"):
            parts = path.split("/")
            if len(parts) >= 6:
                sid = parts[5]
                if len(parts) == 7 and parts[6] == "thumbnail":
                    return self._json({"thumbnail_b64": ""})
                if sid in STATIONS:
                    return self._json(STATIONS[sid])
                return self._json({"detail": "not found"}, 404)

        # /api/v1/pipeline/status/{job_id}
        if path.startswith("/api/v1/pipeline/status/"):
            job_id = path.split("/")[-1]
            if job_id in JOBS:
                job = JOBS[job_id]
                return self._json(job)
            return self._json({"detail": "not found"}, 404)

        # /docs - simple redirect info
        if path == "/docs":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>LabGuardian Mock Server</h1><p>API is running.</p>")
            return

        # openapi.json
        if path == "/api/v1/openapi.json":
            return self._json({"openapi": "3.1.0", "info": {"title": "LabGuardian Mock", "version": "0.1.0"}})

        self._json({"detail": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        body = self._read_body()

        if path == "/api/v1/classroom/heartbeat":
            sid = body.get("station_id", "")
            if sid:
                STATIONS[sid] = body
            return self._json({"status": "ok", "new_alerts": 0})

        if path == "/api/v1/classroom/broadcast":
            msg = body.get("message", "")
            print(f"📢 Broadcast: {msg}")
            return self._json({"status": "ok", "sent": len(STATIONS), "total": len(STATIONS)})

        if path == "/api/v1/classroom/reference":
            return self._json({"status": "ok"})

        if path == "/api/v1/classroom/reset":
            STATIONS.clear()
            return self._json({"status": "ok"})

        # /api/v1/classroom/station/{id}/guidance
        if "/guidance" in path and path.startswith("/api/v1/classroom/station/"):
            sid = path.split("/")[5]
            msg = body.get("message", "")
            print(f"💬 Guidance → {sid}: {msg}")
            GUIDANCE_QUEUE.setdefault(sid, []).append(body)
            return self._json({"status": "delivered"})

        # /api/v1/pipeline/submit
        if path == "/api/v1/pipeline/submit":
            JOB_COUNTER[0] += 1
            job_id = f"job-{JOB_COUNTER[0]:04d}"
            station_id = body.get("station_id", "unknown")

            # 先返回 pending, 3秒后模拟完成
            job_response = {
                "job_id": job_id,
                "status": "running",
                "current_stage": "detect",
                "result": None,
            }
            JOBS[job_id] = job_response

            def simulate_pipeline():
                import time as _time
                stages = ["detect", "mapping", "topology", "validate"]
                for stage in stages:
                    _time.sleep(0.8)
                    JOBS[job_id]["current_stage"] = stage
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["current_stage"] = None
                JOBS[job_id]["result"] = mock_pipeline_result(job_id, station_id)
                print(f"✅ Pipeline {job_id} completed for {station_id}")

            threading.Thread(target=simulate_pipeline, daemon=True).start()
            return self._json(job_response)

        self._json({"detail": "not found"}, 404)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


# ── Main ──────────────────────────────────────────────

if __name__ == "__main__":
    host, port = "0.0.0.0", 8000
    server = HTTPServer((host, port), MockHandler)
    print(f"🚀 LabGuardian Mock Server running at http://localhost:{port}")
    print(f"   GET  /health")
    print(f"   GET  /api/v1/classroom/stations")
    print(f"   GET  /api/v1/classroom/ranking")
    print(f"   GET  /api/v1/classroom/alerts")
    print(f"   GET  /api/v1/classroom/stats")
    print(f"   GET  /api/v1/classroom/station/{{id}}")
    print(f"   POST /api/v1/classroom/heartbeat")
    print(f"   POST /api/v1/pipeline/submit")
    print(f"   GET  /api/v1/pipeline/status/{{job_id}}")
    print(f"   Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.server_close()
