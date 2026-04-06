"""
课堂指导与缩略图服务

负责:
- 工位缩略图缓存
- 单工位指导消息下发
- 全班广播下发
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict

from app.schemas.classroom import GuidanceAuditRecord
from app.services.classroom_state import ClassroomState


class GuidanceService:
    """封装指导消息发送与缩略图缓存."""

    def __init__(self) -> None:
        self._thumbnail_lock = threading.Lock()
        self._thumbnail_cache: dict[str, str] = {}
        self._audit_lock = threading.Lock()
        self._audit_records: list[GuidanceAuditRecord] = []

    def cache_thumbnail(self, station_id: str, thumbnail_b64: str) -> None:
        if not thumbnail_b64:
            return
        with self._thumbnail_lock:
            self._thumbnail_cache[station_id] = thumbnail_b64

    def get_thumbnail(self, station_id: str) -> str:
        with self._thumbnail_lock:
            return self._thumbnail_cache.get(station_id, "")

    def list_audit_records(self, *, station_id: str | None = None, limit: int = 100) -> list[GuidanceAuditRecord]:
        with self._audit_lock:
            records = self._audit_records
            if station_id:
                records = [record for record in records if record.target_id == station_id]
            return list(records[-limit:])

    def _persist_audit_record(
        self,
        *,
        target_type: str,
        target_id: str,
        delivery_status: str,
        delivery_reason: str,
        payload: Dict[str, Any],
    ) -> GuidanceAuditRecord:
        record = GuidanceAuditRecord(
            audit_id=str(uuid.uuid4()),
            target_type=target_type,
            target_id=target_id,
            delivery_status=delivery_status,
            delivery_reason=delivery_reason,
            payload=payload,
            created_at=time.time(),
        )
        with self._audit_lock:
            self._audit_records.append(record)
        return record

    async def send_guidance(
        self,
        classroom: ClassroomState,
        station_id: str,
        guidance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """向单个工位发送指导消息，并记录到课堂状态."""
        classroom.add_guidance_record(station_id, guidance_data)

        ws = classroom.get_websocket(station_id)
        if ws is None:
            record = self._persist_audit_record(
                target_type="station",
                target_id=station_id,
                delivery_status="queued",
                delivery_reason="ws_not_connected",
                payload=guidance_data,
            )
            return {"status": "queued", "reason": "ws_not_connected", "audit_id": record.audit_id}

        try:
            await ws.send_json(guidance_data)
        except Exception:
            record = self._persist_audit_record(
                target_type="station",
                target_id=station_id,
                delivery_status="queued",
                delivery_reason="ws_send_failed",
                payload=guidance_data,
            )
            return {"status": "queued", "reason": "ws_send_failed", "audit_id": record.audit_id}

        record = self._persist_audit_record(
            target_type="station",
            target_id=station_id,
            delivery_status="delivered",
            delivery_reason="",
            payload=guidance_data,
        )
        return {"status": "delivered", "audit_id": record.audit_id}

    async def broadcast(self, classroom: ClassroomState, payload: Dict[str, Any]) -> Dict[str, Any]:
        """向全班广播消息."""
        websockets = classroom.get_all_websockets()
        sent = 0
        for ws in websockets:
            try:
                await ws.send_json(payload)
                sent += 1
            except Exception:
                continue
        record = self._persist_audit_record(
            target_type="classroom",
            target_id="broadcast",
            delivery_status="delivered" if sent else "queued",
            delivery_reason="" if sent else "no_active_websocket",
            payload=payload,
        )
        return {"status": "ok", "sent": sent, "total": len(websockets), "audit_id": record.audit_id}
