from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.schemas.angnt import AngntCitation, AngntEvidence
from app.schemas.kb import KbDocumentInfo


@dataclass(frozen=True)
class _KbDocRecord:
    doc_id: str
    filename: str
    sha256: str
    page_count: int
    chunk_count: int
    created_at: float


class KbService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._storage_dir = Path(settings.KB_STORAGE_DIR)
        self._docs_dir = self._storage_dir / "docs"
        self._chroma_dir = self._storage_dir / "chroma"
        self._manifest_path = self._storage_dir / "manifest.json"
        self._collection = settings.KB_COLLECTION
        self._vs: Chroma | None = None

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if not settings.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is required for embeddings")
        return OpenAIEmbeddings(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_EMBEDDING_MODEL,
        )

    def _get_vectorstore(self) -> Chroma:
        if self._vs is not None:
            return self._vs
        with self._lock:
            if self._vs is None:
                self._vs = Chroma(
                    collection_name=self._collection,
                    persist_directory=str(self._chroma_dir),
                    embedding_function=self._get_embeddings(),
                )
        return self._vs

    def _safe_filename(self, name: str) -> str:
        name = name.strip().replace("\\", "_").replace("/", "_")
        name = re.sub(r"[^0-9A-Za-z.\-_() ]+", "_", name)
        return name[:120] or "document.pdf"

    def _load_manifest(self) -> dict[str, _KbDocRecord]:
        if not self._manifest_path.exists():
            return {}
        payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        result: dict[str, _KbDocRecord] = {}
        for item in payload.get("docs", []):
            try:
                rec = _KbDocRecord(**item)
                result[rec.doc_id] = rec
            except TypeError:
                continue
        return result

    def _save_manifest(self, docs: dict[str, _KbDocRecord]) -> None:
        payload = {"docs": [asdict(v) for v in docs.values()]}
        self._manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_documents(self) -> list[KbDocumentInfo]:
        docs = self._load_manifest()
        return [
            KbDocumentInfo(
                doc_id=d.doc_id,
                filename=d.filename,
                sha256=d.sha256,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                created_at=d.created_at,
            )
            for d in sorted(docs.values(), key=lambda x: x.created_at, reverse=True)
        ]

    def get_status(self) -> dict[str, Any]:
        docs = list(self._load_manifest().values())
        return {
            "storage_dir": str(self._storage_dir),
            "collection": self._collection,
            "doc_count": len(docs),
            "chunk_count": sum(d.chunk_count for d in docs),
        }

    def ingest_pdf(self, *, content: bytes, filename: str) -> KbDocumentInfo:
        sha256 = hashlib.sha256(content).hexdigest()
        doc_id = sha256[:16]
        safe_name = self._safe_filename(filename)

        with self._lock:
            manifest = self._load_manifest()
            if doc_id in manifest:
                existing = manifest[doc_id]
                return KbDocumentInfo(
                    doc_id=existing.doc_id,
                    filename=existing.filename,
                    sha256=existing.sha256,
                    page_count=existing.page_count,
                    chunk_count=existing.chunk_count,
                    created_at=existing.created_at,
                )

            pdf_path = self._docs_dir / f"{doc_id}_{safe_name}"
            pdf_path.write_bytes(content)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        for idx, doc in enumerate(chunks):
            doc.metadata = dict(doc.metadata or {})
            doc.metadata.update(
                {
                    "doc_id": doc_id,
                    "filename": safe_name,
                    "source_path": str(pdf_path),
                    "chunk_index": idx,
                }
            )

        vs = self._get_vectorstore()
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        vs.add_documents(chunks, ids=ids)
        vs.persist()

        record = _KbDocRecord(
            doc_id=doc_id,
            filename=safe_name,
            sha256=sha256,
            page_count=len(pages),
            chunk_count=len(chunks),
            created_at=time.time(),
        )
        with self._lock:
            manifest = self._load_manifest()
            manifest[doc_id] = record
            self._save_manifest(manifest)

        return KbDocumentInfo(
            doc_id=record.doc_id,
            filename=record.filename,
            sha256=record.sha256,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            created_at=record.created_at,
        )

    def retrieve(self, *, query: str, top_k: int) -> list[tuple[dict[str, Any], str]]:
        q = (query or "").strip()
        if not q:
            return []

        vs = self._get_vectorstore()
        docs_with_score = vs.similarity_search_with_score(q, k=top_k)
        results: list[tuple[dict[str, Any], str]] = []
        for doc, score in docs_with_score:
            meta = dict(doc.metadata or {})
            page = meta.get("page")
            filename = meta.get("filename") or meta.get("source") or "datasheet"
            title = f"{filename}" + (f" p{int(page) + 1}" if isinstance(page, int) else "")
            snippet = (doc.page_content or "").strip().replace("\n", " ")
            snippet = snippet[:260]
            results.append(
                (
                    {
                        "title": title,
                        "score": float(score),
                        "metadata": meta,
                        "snippet": snippet,
                        "text": doc.page_content or "",
                    },
                    filename,
                )
            )
        return results

    def _get_llm(self) -> ChatOpenAI:
        if not settings.LLM_API_KEY or not settings.LLM_MODEL:
            raise RuntimeError("LLM_API_KEY and LLM_MODEL are required for answering")
        return ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.2,
        )

    def answer(self, *, query: str, top_k: int) -> tuple[str, list[AngntCitation], list[AngntEvidence], bool]:
        hits = self.retrieve(query=query, top_k=top_k)
        if not hits:
            return "知识库未命中相关内容。", [], [], False

        citations: list[AngntCitation] = []
        evidence: list[AngntEvidence] = []
        context_blocks: list[str] = []

        for i, (hit, _) in enumerate(hits, start=1):
            meta = hit["metadata"]
            page = meta.get("page")
            filename = meta.get("filename") or meta.get("source") or "datasheet"
            page_label = f"p{int(page) + 1}" if isinstance(page, int) else ""
            source_id = f'{meta.get("doc_id", "")}:{meta.get("chunk_index", i - 1)}'

            citations.append(
                AngntCitation(
                    source_type="datasheet_pdf",
                    source_id=source_id,
                    title=f"{filename} {page_label}".strip(),
                    snippet=hit["snippet"],
                )
            )
            evidence.append(
                AngntEvidence(
                    evidence_type="datasheet_chunk",
                    source_id=source_id,
                    summary=f"{filename} {page_label}".strip(),
                    payload={
                        "page": page,
                        "filename": filename,
                        "text": (hit["text"] or "")[:2400],
                    },
                )
            )
            context_blocks.append(
                f"[{i}] {filename} {page_label}\n{(hit['text'] or '').strip()}\n"
            )

        try:
            llm = self._get_llm()
            sys = SystemMessage(
                content=(
                    "你是芯片数据手册助教。只允许根据提供的资料片段回答。"
                    "如果资料片段没有明确答案，请说明无法从资料中确定，并给出需要查找的关键词/章节。"
                    "回答尽量用中文，必要时保留英文参数名。"
                    "回答最后给出引用编号，例如“引用：[1][3]”。"
                )
            )
            human = HumanMessage(
                content="问题：\n"
                + query.strip()
                + "\n\n资料片段：\n"
                + "\n".join(context_blocks)
            )
            msg = llm.invoke([sys, human])
            answer_text = str(getattr(msg, "content", "")).strip() or "已检索到相关资料，但生成回答失败。"
        except Exception:
            answer_text = "已检索到相关资料片段，但当前未配置可用的 LLM（或调用失败）。请查看引用内容。"

        return answer_text, citations, evidence, True
