from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.reference_service import ReferenceService

router = APIRouter(prefix="/references", tags=["references"])


@router.get("")
async def list_references() -> list[dict]:
    """列出所有可用的 logical_reference_v1 参考电路。"""
    service = ReferenceService()
    return service.list_references()


@router.get("/{reference_id}")
async def get_reference(reference_id: str) -> dict:
    """根据 reference_id 获取完整参考电路 JSON。"""
    service = ReferenceService()
    try:
        return service.load_reference(reference_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{reference_id}/summary")
async def get_reference_summary(reference_id: str) -> dict:
    """根据 reference_id 获取参考电路摘要信息。"""
    service = ReferenceService()
    try:
        payload = service.load_reference(reference_id)
        return {
            "reference_id": payload.get("reference_id", reference_id),
            "name": payload.get("name", ""),
            "description": payload.get("description", ""),
            "format": payload.get("format", ""),
            "component_count": len(payload.get("components", [])),
            "net_count": len(payload.get("nets", [])),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
