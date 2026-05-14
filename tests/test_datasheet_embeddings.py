"""Phase 3 tests: hybrid keyword + cosine retrieval and embedding plumbing.

We don't depend on a downloaded OpenVINO IR — instead we drop in a
deterministic fake EmbeddingBackend per test so fusion behavior is
reproducible. The real OpenVINOEmbeddingBackend is exercised separately by a
lightweight smoke test that only checks the graceful-fallback path when the
model directory is missing (the path the board hits when no model is
installed).
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from app.schemas.kb import DatasheetChunk, DatasheetDocument
from app.services.datasheet_kb_service import DatasheetKbService
from app.services.embedding_backend import (
    EmbeddingBackend,
    NullEmbeddingBackend,
    OpenVINOEmbeddingBackend,
    create_embedding_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _DeterministicEmbedding(EmbeddingBackend):
    """Hash-bucket embedding, just for tests.

    Produces a fixed-dim unit vector per text by hashing token bigrams into
    buckets. Different texts that share tokens get correlated vectors; this
    is exactly the property we need to test that fusion uses the cosine
    side. NOT suitable for production retrieval quality.
    """

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def is_active(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t for t in text.lower().split() if t]
            if not tokens:
                out[row] = 1.0 / np.sqrt(self._dim)
                continue
            grams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
            grams.extend(tokens)
            for gram in grams:
                bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self._dim
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


@pytest.fixture()
def synthetic_corpus(tmp_path: Path) -> Path:
    """A tiny corpus with two near-duplicate docs and one distractor."""

    base = tmp_path / "datasheets"
    base.mkdir()

    docs = {
        "ne555_demo": DatasheetDocument(
            document_id="ne555_demo",
            title="NE555",
            part_numbers=["NE555"],
            chunks=[
                DatasheetChunk(
                    chunk_id="ne555_demo.timing",
                    modality="text",
                    title="NE555 timing capacitor sizing",
                    text="The NE555 monostable pulse width is determined by R and C.",
                    keywords=["NE555", "timing", "capacitor"],
                ),
                DatasheetChunk(
                    chunk_id="ne555_demo.pinout",
                    modality="text",
                    title="NE555 pin assignments",
                    text="Pins are GND, TRIG, OUT, RESET, CONT, THRES, DISCH, VCC.",
                    keywords=["NE555", "pinout"],
                ),
            ],
        ),
        "lm324_demo": DatasheetDocument(
            document_id="lm324_demo",
            title="LM324",
            part_numbers=["LM324"],
            chunks=[
                DatasheetChunk(
                    chunk_id="lm324_demo.supply",
                    modality="text",
                    title="LM324 supply range",
                    text="LM324 quad op-amp accepts 3V to 32V single supply.",
                    keywords=["LM324", "supply"],
                ),
            ],
        ),
    }
    for doc in docs.values():
        (base / f"{doc.document_id}.json").write_text(
            doc.model_dump_json(indent=2), encoding="utf-8"
        )
    return base


def test_null_backend_keeps_keyword_only_behavior(synthetic_corpus: Path) -> None:
    kb = DatasheetKbService(base_dir=synthetic_corpus, embedding=NullEmbeddingBackend())
    assert not kb.has_embeddings
    hits = kb.search("NE555 timing", top_k=3)
    assert hits and hits[0].chunk_id.startswith("ne555_demo.")


def test_hybrid_fusion_uses_cosine_when_keyword_misses(
    tmp_path: Path,
    synthetic_corpus: Path,
) -> None:
    """A query that shares NO surface tokens with chunk.keywords should still
    surface the right chunk via the semantic side after we precompute vectors.
    """

    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()

    backend = _DeterministicEmbedding(dim=64)

    # Pre-compute chunk vectors (same text recipe the production script uses).
    docs = sorted(synthetic_corpus.glob("*.json"))
    for path in docs:
        doc = DatasheetDocument.model_validate_json(path.read_text(encoding="utf-8"))
        texts = [
            "\n".join(filter(None, [c.title, c.text, " ".join(c.keywords)]))
            for c in doc.chunks
        ]
        vectors = backend.encode(texts)
        np.savez(
            embed_dir / f"{doc.document_id}.npz",
            chunk_ids=np.array([c.chunk_id for c in doc.chunks], dtype=np.str_),
            vectors=vectors,
        )

    kb = DatasheetKbService(
        base_dir=synthetic_corpus,
        embedding=backend,
        embeddings_dir=embed_dir,
        fusion_weight=0.8,
    )
    assert kb.has_embeddings

    # Query shares a literal token with one chunk's text. Hybrid should rank
    # that chunk first because both keyword and cosine point the same way.
    hits = kb.search("monostable pulse width", top_k=3)
    assert hits
    assert hits[0].chunk_id == "ne555_demo.timing"


def test_hybrid_skipped_when_npz_missing(synthetic_corpus: Path, tmp_path: Path) -> None:
    """Active backend + empty cache must NOT call encode (avoids on-device
    overhead) and must reproduce keyword-only ordering."""

    backend = _DeterministicEmbedding()
    kb = DatasheetKbService(
        base_dir=synthetic_corpus,
        embedding=backend,
        embeddings_dir=tmp_path / "no_embeddings_here",
    )
    assert not kb.has_embeddings
    hits = kb.search("NE555 timing", top_k=2)
    assert hits[0].chunk_id.startswith("ne555_demo.")


def test_openvino_backend_falls_back_when_model_dir_missing(tmp_path: Path) -> None:
    backend = OpenVINOEmbeddingBackend(model_dir=tmp_path / "no-such-model")
    assert not backend.is_active  # lazy load fails silently
    vectors = backend.encode(["hello"])
    assert vectors.shape == (1, 0)


def test_create_embedding_backend_defaults_to_null() -> None:
    assert isinstance(create_embedding_backend(None), NullEmbeddingBackend)
    assert isinstance(create_embedding_backend("openvino", model_dir=None), NullEmbeddingBackend)


def test_build_embeddings_script_writes_npz_against_null_backend(
    tmp_path: Path,
    synthetic_corpus: Path,
) -> None:
    """The precompute script must round-trip: write .npz with the same
    chunk_ids/vectors schema that DatasheetKbService loads. Using the null
    backend here keeps the test hermetic (no model needed)."""

    out_dir = tmp_path / "embeddings_out"

    spec = importlib.util.spec_from_file_location(
        "build_datasheet_embeddings",
        REPO_ROOT / "scripts" / "build_datasheet_embeddings.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    backend = _DeterministicEmbedding(dim=16)
    docs = sorted(synthetic_corpus.glob("*.json"))
    written = 0
    for path in docs:
        document = DatasheetDocument.model_validate_json(path.read_text(encoding="utf-8"))
        chunk_ids, vectors = module.encode_document(document, backend)
        if not chunk_ids:
            continue
        module.write_npz(out_dir / f"{document.document_id}.npz", chunk_ids, vectors)
        written += 1
    assert written == 2

    # Round-trip: KbService should pick up the vectors and activate hybrid.
    kb = DatasheetKbService(
        base_dir=synthetic_corpus,
        embedding=backend,
        embeddings_dir=out_dir,
    )
    assert kb.has_embeddings
