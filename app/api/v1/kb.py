from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.deps import get_kb_service
from app.schemas.kb import KbDocumentInfo, KbQueryRequest, KbQueryResponse, KbStatusResponse
from app.services.kb_service import KbService

router = APIRouter(prefix="/kb", tags=["kb"])


@router.get("/status", response_model=KbStatusResponse)
async def get_status(kb: KbService = Depends(get_kb_service)):
    return KbStatusResponse(**kb.get_status())


@router.get("/docs", response_model=list[KbDocumentInfo])
async def list_docs(kb: KbService = Depends(get_kb_service)):
    return kb.list_documents()


@router.post("/upload", response_model=KbDocumentInfo)
async def upload_pdf(
    file: UploadFile = File(...),
    kb: KbService = Depends(get_kb_service),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf is supported")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        return kb.ingest_pdf(content=content, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=KbQueryResponse)
async def query_kb(
    request: KbQueryRequest,
    kb: KbService = Depends(get_kb_service),
):
    try:
        answer, citations, evidence, used = kb.answer(query=request.query, top_k=request.top_k)
        return KbQueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            evidence=evidence,
            used_retrieval=used,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

