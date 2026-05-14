from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pypdf import PdfReader

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
        self._bootstrap_dir = Path(getattr(settings, "KB_BOOTSTRAP_DIR", "") or "")
        if self._bootstrap_dir and not self._bootstrap_dir.is_absolute():
            self._bootstrap_dir = Path(__file__).resolve().parent.parent.parent / self._bootstrap_dir
        self._bootstrap_attempted = False
        self._bootstrap_scan_at = 0.0

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

    def _get_embeddings(self) -> Any:
        provider = str(getattr(settings, "KB_EMBEDDING_PROVIDER", "openai") or "openai").strip().lower()
        if provider == "ollama":
            base_url = str(getattr(settings, "AGENT_LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")
            model = str(getattr(settings, "KB_OLLAMA_EMBED_MODEL", "nomic-embed-text") or "nomic-embed-text").strip()

            class _OllamaEmbeddings:
                def __init__(self, *, base_url: str, model: str) -> None:
                    self._base_url = base_url
                    self._model = model

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    return [self._embed_one(text) for text in texts]

                def embed_query(self, text: str) -> list[float]:
                    return self._embed_one(text)

                def _embed_one(self, text: str) -> list[float]:
                    prompt = str(text or "")
                    for endpoint, payload in (
                        (
                            f"{self._base_url}/api/embeddings",
                            {"model": self._model, "prompt": prompt},
                        ),
                        (
                            f"{self._base_url}/api/embed",
                            {"model": self._model, "input": prompt},
                        ),
                    ):
                        try:
                            with httpx.Client(timeout=60.0, trust_env=False) as client:
                                resp = client.post(endpoint, json=payload)
                                resp.raise_for_status()
                                body = resp.json()
                            if isinstance(body, dict) and isinstance(body.get("embedding"), list):
                                return [float(x) for x in body["embedding"]]
                            if isinstance(body, dict) and isinstance(body.get("embeddings"), list):
                                first = body["embeddings"][0] if body["embeddings"] else []
                                if isinstance(first, list):
                                    return [float(x) for x in first]
                        except Exception:
                            continue
                    raise RuntimeError("ollama embeddings failed: no embedding returned")

            return _OllamaEmbeddings(base_url=base_url, model=model)

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("langchain-openai is required for KB embeddings") from exc

        if not settings.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is required for embeddings")
        return OpenAIEmbeddings(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_EMBEDDING_MODEL,
        )

    def _get_vectorstore(self):
        Chroma = None
        try:
            from langchain_chroma import Chroma as _Chroma  # type: ignore[import-not-found]

            Chroma = _Chroma
        except Exception:
            Chroma = None
        if Chroma is None:
            try:
                from langchain_community.vectorstores import Chroma as _Chroma  # type: ignore[import-not-found]

                Chroma = _Chroma
            except ImportError as exc:
                raise RuntimeError("langchain-community (or langchain-chroma) is required for KB vector search") from exc

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

    def _chip_hints_from_query(self, query: str) -> list[str]:
        msg = str(query or "").strip().lower()
        alias_groups = {
            "ne555": ("ne555", "na555", "sa555", "se555", "555", "定时器", "计时器"),
            "lm324": ("lm324", "lm2902", "运放", "运算放大器", "op amp", "operational amplifier"),
            "74ls74": (
                "sn74ls74",
                "sn74ls74a",
                "74ls74",
                "54ls74",
                "xd74ls74",
                "d触发器",
                "双d",
                "flip-flop",
                "flip flop",
            ),
        }
        hints: list[str] = []
        for canonical, aliases in alias_groups.items():
            if any(alias in msg for alias in aliases) and canonical not in hints:
                hints.append(canonical)
        return hints

    def _chip_aliases_for_hints(self, chip_hints: list[str]) -> list[str]:
        aliases_by_hint = {
            "ne555": ("ne555", "na555", "sa555", "se555", "555"),
            "lm324": ("lm324", "lm2902"),
            "74ls74": ("sn74ls74", "sn74ls74a", "74ls74", "54ls74", "xd74ls74"),
        }
        aliases: list[str] = []
        for hint in chip_hints:
            for alias in aliases_by_hint.get(hint, (hint,)):
                if alias not in aliases:
                    aliases.append(alias)
        return aliases

    def _filter_hits_by_chip(
        self,
        hits: list[tuple[dict[str, Any], str]],
        chip_hints: list[str],
    ) -> list[tuple[dict[str, Any], str]]:
        if not chip_hints:
            return hits
        aliases = self._chip_aliases_for_hints(chip_hints)
        filtered: list[tuple[dict[str, Any], str]] = []
        for hit, filename in hits:
            meta = hit.get("metadata", {}) or {}
            haystack = " ".join(
                [
                    str(filename or meta.get("filename") or ""),
                    str(hit.get("title") or ""),
                    str(hit.get("snippet") or ""),
                    str(hit.get("text") or "")[:1200],
                ]
            ).lower()
            if any(alias in haystack for alias in aliases):
                filtered.append((hit, filename))
        return filtered

    def _maybe_expand_datasheet_query(self, query: str) -> str:
        msg = str(query or "").strip()
        lowered = msg.lower()
        hints = ("pin", "引脚", "脚位", "管脚", "脚", "pinout", "reset", "clk", "clear", "preset")
        if any(h in lowered for h in hints) and not any(
            k in lowered
            for k in (
                "pin configuration",
                "pin diagram",
                "pin description",
                "pin functions",
                "引脚功能",
                "引脚描述",
                "引脚定义",
                "封装",
                "package",
            )
        ):
            return msg + " pin configuration pin diagram pin description pin functions pinning information terminal functions package 引脚功能 引脚定义 封装"
        if any(h in lowered for h in ("供电", "电源", "电压", "vcc", "vdd", "vee", "supply")):
            return msg + " recommended operating conditions supply voltage VCC VDD VEE power supply electrical characteristics"
        if any(h in lowered for h in ("真值表", "逻辑表", "功能表", "truth table", "function table")):
            return msg + " truth table function table logic table clear preset clock output"
        return msg

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

    def get_debug_status(self) -> dict[str, Any]:
        docs = list(self._load_manifest().values())
        provider = str(getattr(settings, "KB_EMBEDDING_PROVIDER", "openai") or "openai")
        embed_model = (
            str(getattr(settings, "KB_OLLAMA_EMBED_MODEL", "") or "")
            if provider.strip().lower() == "ollama"
            else str(getattr(settings, "LLM_EMBEDDING_MODEL", "") or "")
        )
        pdfs = self._iter_local_pdf_paths()
        return {
            "storage_dir": str(self._storage_dir),
            "docs_dir": str(self._docs_dir),
            "bootstrap_dir": str(self._bootstrap_dir) if self._bootstrap_dir else "",
            "bootstrap_pdf_count": len([p for p in pdfs if self._bootstrap_dir and p.is_relative_to(self._bootstrap_dir)]),
            "local_pdf_count": len(pdfs),
            "local_pdfs": [p.name for p in pdfs[:20]],
            "collection": self._collection,
            "doc_count": len(docs),
            "chunk_count": sum(d.chunk_count for d in docs),
            "embedding_provider": provider,
            "embedding_model": embed_model,
            "ollama_base_url": str(getattr(settings, "AGENT_LLM_OLLAMA_BASE_URL", "") or ""),
            "bootstrap_attempted": bool(self._bootstrap_attempted),
        }

    def test_embeddings(self, *, text: str = "ping") -> dict[str, Any]:
        provider = str(getattr(settings, "KB_EMBEDDING_PROVIDER", "openai") or "openai")
        embed_model = (
            str(getattr(settings, "KB_OLLAMA_EMBED_MODEL", "") or "")
            if provider.strip().lower() == "ollama"
            else str(getattr(settings, "LLM_EMBEDDING_MODEL", "") or "")
        )
        try:
            embeddings = self._get_embeddings()
            vec = embeddings.embed_query(str(text or "ping"))
            dim = len(vec) if isinstance(vec, list) else 0
            return {
                "ok": True,
                "provider": provider,
                "model": embed_model,
                "dim": dim,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": embed_model,
                "error": str(exc),
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

        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError("langchain-community and langchain-text-splitters are required for PDF ingestion") from exc

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

        chip_hints = self._chip_hints_from_query(q)
        requested_k = max(1, int(top_k or 6))
        base_k = max(requested_k, 8)
        if chip_hints:
            base_k = min(60, max(base_k * 8, 24))
        expanded = self._maybe_expand_datasheet_query(q)
        results: list[tuple[dict[str, Any], str]] = []
        try:
            self._ensure_bootstrap_ingested()
            vs = self._get_vectorstore()
            docs_with_score = vs.similarity_search_with_score(expanded, k=base_k)
        except Exception:
            docs_with_score = []
        for doc, score in docs_with_score:
            results.append(self._hit_from_document(doc=doc, score=float(score)))

        fallback = self._fallback_retrieve_from_pdfs(query=expanded, top_k=base_k)
        results = self._merge_hits(results + fallback)

        filtered = self._filter_hits_by_chip(results, chip_hints)
        if chip_hints and not filtered:
            return []
        results = filtered or results
        return self._rank_hits(query=expanded, hits=results, chip_hints=chip_hints)[:requested_k]

    def _hit_from_document(self, *, doc: Any, score: float) -> tuple[dict[str, Any], str]:
        meta = dict(doc.metadata or {})
        page = meta.get("page")
        filename = meta.get("filename") or meta.get("source") or "datasheet"
        title = f"{filename}" + (f" p{int(page) + 1}" if isinstance(page, int) else "")
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        snippet = snippet[:260]
        return (
            {
                "title": title,
                "score": score,
                "metadata": meta,
                "snippet": snippet,
                "text": doc.page_content or "",
            },
            filename,
        )

    def _merge_hits(
        self,
        hits: list[tuple[dict[str, Any], str]],
    ) -> list[tuple[dict[str, Any], str]]:
        merged: dict[tuple[str, int | str], tuple[dict[str, Any], str]] = {}
        content_seen: dict[str, tuple[dict[str, Any], str]] = {}
        for hit, filename in hits:
            meta = hit.get("metadata", {}) or {}
            text_key = re.sub(r"\s+", " ", str(hit.get("text") or "")[:1800]).strip().lower()
            if text_key:
                digest = hashlib.sha256(text_key.encode("utf-8", errors="ignore")).hexdigest()[:16]
                existing_by_text = content_seen.get(digest)
                if existing_by_text is not None:
                    existing_hit, existing_filename = existing_by_text
                    existing_meta = existing_hit.get("metadata", {}) or {}
                    if str(existing_meta.get("doc_id") or "") == "local_pdf":
                        content_seen[digest] = (hit, filename or existing_filename)
                    continue
            key = (
                str(meta.get("source_path") or meta.get("filename") or filename or "").lower(),
                meta.get("page") if meta.get("page") is not None else meta.get("chunk_index", ""),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = (hit, filename)
                if text_key:
                    content_seen[digest] = merged[key]
                continue
            existing_hit, existing_filename = existing
            if len(str(hit.get("text") or "")) > len(str(existing_hit.get("text") or "")):
                merged[key] = (hit, filename or existing_filename)
            if text_key:
                content_seen[digest] = merged[key]
        return list(content_seen.values()) if content_seen else list(merged.values())

    def _rank_hits(
        self,
        *,
        query: str,
        hits: list[tuple[dict[str, Any], str]],
        chip_hints: list[str],
    ) -> list[tuple[dict[str, Any], str]]:
        terms = self._query_terms(query)
        query_l = query.lower()
        pin_query = any(
            token in query_l
            for token in ("pin", "pinout", "引脚", "脚位", "管脚", "脚", "reset", "clear", "preset", "clk")
        )
        supply_query = any(token in query_l for token in ("供电", "电源", "电压", "supply", "vcc", "vdd", "vee"))
        truth_query = any(token in query_l for token in ("真值表", "功能表", "truth table", "function table"))
        aliases = self._chip_aliases_for_hints(chip_hints)

        def rank_key(item: tuple[dict[str, Any], str]) -> tuple[float, float]:
            hit, filename = item
            meta = hit.get("metadata", {}) or {}
            title = str(hit.get("title") or "")
            text = str(hit.get("text") or "")
            haystack = " ".join([str(filename or ""), title, str(hit.get("snippet") or ""), text[:5000]]).lower()
            filename_l = str(filename or meta.get("filename") or "").lower()
            lexical = 0.0
            if aliases and any(alias in filename_l for alias in aliases):
                lexical += 24.0
            elif aliases and any(alias in haystack for alias in aliases):
                lexical += 10.0
            for term in terms:
                count = haystack.count(term)
                if count:
                    lexical += 1.0 + min(4.0, count * 0.35)
                if term in filename_l:
                    lexical += 5.0
            if pin_query and any(
                phrase in haystack
                for phrase in (
                    "pin configuration",
                    "pin diagram",
                    "pin description",
                    "pin functions",
                    "pinning information",
                    "terminal functions",
                    "terminal assignments",
                    "connection diagram",
                    "引脚",
                )
            ):
                lexical += 16.0
                page = meta.get("page")
                if isinstance(page, int) and page < 10:
                    lexical += 5.0
            if supply_query and any(
                phrase in haystack
                for phrase in (
                    "recommended operating conditions",
                    "supply voltage",
                    "power supply",
                    "electrical characteristics",
                    "vcc",
                    "vdd",
                    "vee",
                )
            ):
                lexical += 14.0
            if truth_query and any(phrase in haystack for phrase in ("truth table", "function table", "logic table")):
                lexical += 16.0
            if any(
                phrase in haystack
                for phrase in (
                    "package option addendum",
                    "package materials information",
                    "tape and reel",
                    "changes from revision",
                    "封装选项附录",
                    "可订购器件",
                )
            ):
                lexical -= 30.0
            vector_score = float(hit.get("score") or 0.0)
            hit["rank_score"] = lexical
            return lexical, -vector_score

        return sorted(hits, key=rank_key, reverse=True)

    def _ensure_bootstrap_ingested(self) -> None:
        now = time.time()
        if self._bootstrap_attempted and (now - float(self._bootstrap_scan_at or 0.0)) < 30.0:
            return
        self._bootstrap_attempted = True
        self._bootstrap_scan_at = now
        if not (self._bootstrap_dir and self._bootstrap_dir.exists()):
            return
        pdfs = sorted(self._bootstrap_dir.glob("*.[pP][dD][fF]"))
        if not pdfs:
            return
        manifest = self._load_manifest()
        known_ids = set(manifest.keys())
        for path in pdfs:
            try:
                content = path.read_bytes()
                if not content:
                    continue
                sha = hashlib.sha256(content).hexdigest()
                doc_id = sha[:16]
                if doc_id in known_ids:
                    continue
                self.ingest_pdf(content=content, filename=path.name)
            except Exception:
                continue

    def _iter_local_pdf_paths(self) -> list[Path]:
        paths: list[Path] = []
        if self._bootstrap_dir and self._bootstrap_dir.exists():
            paths.extend(sorted(self._bootstrap_dir.glob("*.[pP][dD][fF]")))
        if self._docs_dir.exists():
            paths.extend(sorted(self._docs_dir.glob("*.[pP][dD][fF]")))
        uniq: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                uniq.append(path)
        return uniq

    def _query_terms(self, query: str) -> list[str]:
        text = str(query or "").lower()
        alias_terms: list[str] = []
        if any(token in text for token in ("供电", "电源", "电压")):
            alias_terms.extend(["supply", "supply voltage", "power supply", "vcc", "vdd", "vee"])
        if any(token in text for token in ("复位", "清零")):
            alias_terms.extend(["reset", "clear", "clr"])
        if "置位" in text:
            alias_terms.extend(["preset", "pre"])
        if "时钟" in text:
            alias_terms.extend(["clock", "clk"])
        if "输出" in text:
            alias_terms.extend(["output", "out", "q"])
        if any(token in text for token in ("真值表", "逻辑表", "功能表")):
            alias_terms.extend(["truth table", "function table", "logic table"])
        if any(token in text for token in ("引脚", "脚位", "管脚", "pinout")) or re.search(r"\d+\s*脚", text):
            alias_terms.extend(["pin", "pin configuration", "pin description", "pinning information"])

        split_parts = [
            p.strip().lower()
            for p in re.split(r"[\s,，。？?；;：:、/()（）]+", text)
            if p and len(p.strip()) >= 2
        ]
        alnum_parts = re.findall(r"[a-z]+[0-9]+[a-z0-9]*|[0-9]+[a-z]+[a-z0-9]*|[a-z]{2,}|\d+", text)
        parts = split_parts + alnum_parts + alias_terms
        deduped: list[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        return deduped[:24]

    def _fallback_retrieve_from_pdfs(self, *, query: str, top_k: int) -> list[tuple[dict[str, Any], str]]:
        terms = self._query_terms(query)
        if not terms:
            return []

        scored: list[tuple[float, dict[str, Any], str]] = []
        for pdf_path in self._iter_local_pdf_paths():
            try:
                reader = PdfReader(str(pdf_path))
            except Exception:
                continue
            try:
                pages = list(reader.pages)
            except Exception:
                continue
            for page_index, page in enumerate(pages):
                try:
                    text = str(page.extract_text() or "")
                except Exception:
                    text = ""
                if not text.strip():
                    continue
                lowered = text.lower()
                score = 0.0
                for term in terms:
                    if term in lowered:
                        score += 1.0 + min(3.0, float(lowered.count(term)) * 0.2)
                if score <= 0:
                    continue
                filename = pdf_path.name
                title = f"{filename} p{page_index + 1}"
                snippet = re.sub(r"\s+", " ", text).strip()[:260]
                scored.append(
                    (
                        score,
                        {
                            "title": title,
                            "score": score,
                            "metadata": {
                                "filename": filename,
                                "source_path": str(pdf_path),
                                "page": page_index,
                                "doc_id": "local_pdf",
                                "chunk_index": page_index,
                            },
                            "snippet": snippet,
                            "text": text,
                        },
                        filename,
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [(hit, filename) for _, hit, filename in scored[: max(1, int(top_k or 5))]]

    def _get_llm(self) -> ChatOpenAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("langchain-openai is required for KB answering") from exc

        if not settings.LLM_API_KEY or not settings.LLM_MODEL:
            raise RuntimeError("LLM_API_KEY and LLM_MODEL are required for answering")
        return ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.2,
        )

    def answer(self, *, query: str, top_k: int) -> tuple[str, list[AngntCitation], list[AngntEvidence], bool]:
        builtin = self._answer_from_builtin_datasheet(query=query)
        if builtin is not None:
            return builtin
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
            from langchain_core.messages import HumanMessage, SystemMessage

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
            answer_text = self._answer_with_ollama(query=query, context_blocks=context_blocks)

        return answer_text, citations, evidence, True

    def _answer_from_builtin_datasheet(
        self,
        *,
        query: str,
    ) -> tuple[str, list[AngntCitation], list[AngntEvidence], bool] | None:
        chip_hints = self._chip_hints_from_query(query)
        query_l = str(query or "").lower()
        pin_query = any(token in query_l for token in ("引脚", "脚位", "管脚", "pinout", "pin ", "pin:"))
        if "74ls74" not in chip_hints or not pin_query:
            return None
        text = (
            "74LS74 / SN74LS74A 常见 DIP-14 引脚为："
            "1=/CLR1（清零，低有效），2=1D，3=CLK1，4=/PRE1（置位，低有效），"
            "5=1Q，6=/1Q，7=GND，8=/2Q，9=2Q，10=/PRE2，11=CLK2，"
            "12=2D，13=/CLR2，14=VCC。"
            "不同厂商/封装的管脚图仍以对应 datasheet 和实物丝印方向为准。"
        )
        citation = AngntCitation(
            source_type="builtin_datasheet_fallback",
            source_id="builtin:74ls74:dip14-pinout",
            title="74LS74 / SN74LS74A DIP-14 pinout fallback",
            snippet=text,
        )
        evidence = AngntEvidence(
            evidence_type="datasheet_builtin_fallback",
            source_id=citation.source_id,
            summary=citation.title,
            payload={
                "part": "74LS74/SN74LS74A",
                "package": "DIP-14",
                "text": text,
            },
        )
        return f"{text}\n引用：[builtin:74ls74:dip14-pinout]", [citation], [evidence], True

    def _answer_with_ollama(self, *, query: str, context_blocks: list[str]) -> str:
        if not context_blocks:
            return "已检索到相关资料片段，但无法生成回答。"
        endpoint = f"{getattr(settings, 'AGENT_LLM_OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/chat"
        timeout_s = float(getattr(settings, "AGENT_LLM_OLLAMA_TIMEOUT_S", 120.0) or 120.0)
        model_name = getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "qwen3:4b")
        prompt = "\n".join(
            [
                "你是芯片数据手册助教。只能依据提供的资料片段回答；不得编造参数、引脚定义或极限值。",
                "如果资料片段没有明确答案，请说明无法从资料中确定，并给出需要查找的关键词/章节名。",
                "回答尽量用中文，必要时保留英文参数名。最后给出引用编号，例如“引用：[1][2]”。",
                f"问题：{(query or '').strip()}",
                "",
                "资料片段：",
                "\n".join(context_blocks),
            ]
        )
        payload = {
            "model": model_name,
            "stream": False,
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": "你是严谨的数据手册问答助手，严禁编造证据。"},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": 420,
            },
        }
        try:
            with httpx.Client(timeout=min(max(timeout_s, 20.0), 60.0), trust_env=False) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
            text = str(((body or {}).get("message") or {}).get("content") or "").strip()
            if text:
                return text
        except Exception:
            pass
        return "已检索到相关资料片段，但当前未配置可用的 LLM（或调用失败）。请查看引用内容。"
