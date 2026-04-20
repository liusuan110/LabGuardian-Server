"""
课堂状态管理器 (← teacher/classroom.py)

维护全班工位的实时状态, 线程安全。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WebSocketConnection = Any


@dataclass
class StationState:
    """单个工位的完整状态"""

    heartbeat: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    websocket: Optional[WebSocketConnection] = None
    risk_event_count: int = 0
    peak_progress: float = 0.0
    guidance_history: List[Dict[str, str]] = field(default_factory=list)


class ClassroomState:
    """全班课堂状态 (内存存储, 单次实验会话)"""

    def __init__(self, online_timeout: float = 10.0):
        self._lock = threading.Lock()
        self._stations: Dict[str, StationState] = {}
        self._session_start: float = time.time()
        self._reference_circuit: Optional[Dict] = None
        self.online_timeout = online_timeout

    # ---- 心跳更新 ----

    def update_station(self, heartbeat: Dict[str, Any]) -> List[str]:
        """更新工位状态, 返回新产生的警报列表"""
        station_id = heartbeat.get("station_id", "unknown")
        now = time.time()
        new_alerts: List[str] = []

        with self._lock:
            if station_id not in self._stations:
                self._stations[station_id] = StationState(first_seen=now)
                logger.info(f"[Classroom] 新工位上线: {station_id}")

            station = self._stations[station_id]
            old_risk = station.heartbeat.get("risk_level", "safe")

            station.heartbeat = heartbeat
            station.last_seen = now

            progress = heartbeat.get("progress", 0.0)
            if progress > station.peak_progress:
                station.peak_progress = progress

            new_risk = heartbeat.get("risk_level", "safe")
            if new_risk == "danger":
                station.risk_event_count += 1
                if old_risk != "danger":
                    student = heartbeat.get("student_name", station_id)
                    reasons = heartbeat.get("risk_reasons", [])
                    reason_text = reasons[0] if reasons else "检测到危险电路"
                    new_alerts.append(f"{station_id} {student} — {reason_text}")

        return new_alerts

    # ---- 查询 ----

    def get_all_stations(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        result = {}
        with self._lock:
            for sid, state in self._stations.items():
                result[sid] = {
                    **state.heartbeat,
                    "online": (now - state.last_seen) < self.online_timeout,
                    "first_seen": state.first_seen,
                    "elapsed_s": now - state.first_seen,
                    "risk_event_count": state.risk_event_count,
                    "peak_progress": state.peak_progress,
                }
        return result

    def get_ranking(self) -> List[Dict[str, Any]]:
        stations = self.get_all_stations()
        ranking = []
        for sid, data in stations.items():
            ranking.append({
                "station_id": sid,
                "student_name": data.get("student_name", ""),
                "progress": data.get("progress", 0.0),
                "similarity": data.get("similarity", 0.0),
                "elapsed_s": data.get("elapsed_s", 0),
                "risk_event_count": data.get("risk_event_count", 0),
                "component_count": data.get("component_count", 0),
                "risk_level": data.get("risk_level", "safe"),
                "online": data.get("online", False),
            })
        ranking.sort(key=lambda x: (-x["progress"], x["elapsed_s"]))
        for i, entry in enumerate(ranking):
            entry["rank"] = i + 1
        return ranking

    def get_alerts(self) -> List[Dict[str, Any]]:
        stations = self.get_all_stations()
        alerts = []
        for sid, data in stations.items():
            if data.get("risk_level", "safe") != "safe" and data.get("online"):
                alerts.append({
                    "station_id": sid,
                    "student_name": data.get("student_name", ""),
                    "risk_level": data.get("risk_level", "warning"),
                    "risk_reasons": data.get("risk_reasons", []),
                    "diagnostics": data.get("diagnostics", []),
                    "progress": data.get("progress", 0.0),
                })
        alerts.sort(key=lambda x: (0 if x["risk_level"] == "danger" else 1))
        return alerts

    def get_stats(self) -> Dict[str, Any]:
        stations = self.get_all_stations()
        if not stations:
            return {
                "total_stations": 0,
                "online_count": 0,
                "completed_count": 0,
                "avg_progress": 0.0,
                "total_risk_events": 0,
                "danger_count": 0,
                "error_histogram": {},
                "session_duration_s": time.time() - self._session_start,
            }

        online_count = sum(1 for s in stations.values() if s.get("online"))
        completed = sum(1 for s in stations.values() if s.get("progress", 0) >= 1.0)
        progresses = [s.get("progress", 0.0) for s in stations.values()]
        total_risk = sum(s.get("risk_event_count", 0) for s in stations.values())

        error_counter: Counter = Counter()
        for s in stations.values():
            for diag in s.get("diagnostics", []):
                if ":" in diag:
                    error_type = diag.split(":", 1)[1].strip()[:30]
                else:
                    error_type = diag[:30]
                error_counter[error_type] += 1

        danger_count = sum(
            1 for s in stations.values() if s.get("risk_level") == "danger" and s.get("online")
        )

        return {
            "total_stations": len(stations),
            "online_count": online_count,
            "completed_count": completed,
            "avg_progress": sum(progresses) / len(progresses) if progresses else 0.0,
            "total_risk_events": total_risk,
            "danger_count": danger_count,
            "error_histogram": dict(error_counter.most_common(10)),
            "session_duration_s": time.time() - self._session_start,
        }

    # ---- WebSocket 管理 ----

    def register_websocket(self, station_id: str, ws: WebSocketConnection):
        now = time.time()
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].websocket = ws
                self._stations[station_id].last_seen = now
            else:
                self._stations[station_id] = StationState(
                    first_seen=now, last_seen=now, websocket=ws
                )

    def unregister_websocket(self, station_id: str):
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].websocket = None

    def touch_station(self, station_id: str):
        """更新 last_seen 保活 (WebSocket ping 时调用)"""
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].last_seen = time.time()

    def get_websocket(self, station_id: str) -> Optional[WebSocketConnection]:
        with self._lock:
            station = self._stations.get(station_id)
            return station.websocket if station else None

    def get_all_websockets(self) -> List[WebSocketConnection]:
        with self._lock:
            return [s.websocket for s in self._stations.values() if s.websocket is not None]

    def add_guidance_record(self, station_id: str, guidance: Dict[str, str]):
        with self._lock:
            if station_id in self._stations:
                self._stations[station_id].guidance_history.append(guidance)

    # ---- 参考电路 ----

    def set_reference(self, reference: Dict):
        with self._lock:
            self._reference_circuit = reference

    def get_reference(self) -> Optional[Dict]:
        with self._lock:
            return self._reference_circuit

    # ---- 会话管理 ----

    def reset(self):
        with self._lock:
            self._stations.clear()
            self._reference_circuit = None
            self._session_start = time.time()

    @property
    def station_count(self) -> int:
        with self._lock:
            return len(self._stations)
