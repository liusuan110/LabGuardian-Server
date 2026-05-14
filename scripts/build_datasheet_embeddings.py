"""Build-time pre-compute of chunk embeddings.

Reads every ``knowledge/datasheets/*.json`` document, batches the chunk text
through an embedding backend, and writes one ``.npz`` per document into
``knowledge/datasheets/embeddings/``. The npz holds two arrays:

- ``chunk_ids`` — 1-D str array, one entry per row
- ``vectors``   — 2-D float32, L2-normalized

Runs **on the developer machine / CI** only. The board never reruns this — it
just ships the resulting ``.npz`` files inside the image. Encoding the entire
corpus once offline keeps the on-device cost to a single query encode at
search time.

Backends:

- ``openvino`` (default) — needs ``DATASHEET_EMBEDDING_MODEL_DIR`` pointing
  at a local OpenVINO IR directory (``openvino_model.xml`` +
  ``openvino_model.bin`` + ``tokenizer.json``). See the Phase 3 docs for
  the ``optimum-cli`` conversion command.
- ``null`` — produces zero-width vectors (useful for testing this script
  end-to-end without a real model).

Example
=======

::

    .venv/bin/python scripts/build_datasheet_embeddings.py \
        --backend openvino \
        --model-dir models/bge-small-zh-v1.5-int8-ov \
        --device CPU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.schemas.kb import DatasheetDocument  # noqa: E402
from app.services.embedding_backend import (  # noqa: E402
    EmbeddingBackend,
    create_embedding_backend,
)

logger = logging.getLogger("build_datasheet_embeddings")


def _chunk_text(chunk) -> str:
    parts = [chunk.title or "", chunk.text or ""]
    if chunk.keywords:
        parts.append(" ".join(chunk.keywords))
    return "\n".join(p for p in parts if p)


def encode_document(
    document: DatasheetDocument,
    backend: EmbeddingBackend,
    batch_size: int = 16,
) -> tuple[list[str], np.ndarray]:
    chunk_ids = [c.chunk_id for c in document.chunks]
    if not chunk_ids:
        return [], np.zeros((0, backend.dim or 0), dtype=np.float32)

    texts = [_chunk_text(c) for c in document.chunks]
    blocks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = backend.encode(batch)
        if vectors.size == 0:
            return chunk_ids, np.zeros((0, 0), dtype=np.float32)
        blocks.append(vectors.astype(np.float32, copy=False))
    return chunk_ids, np.vstack(blocks)


def write_npz(out_path: Path, chunk_ids: list[str], vectors: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        chunk_ids=np.array(chunk_ids, dtype=np.str_),
        vectors=vectors,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasheets-dir",
        type=Path,
        default=REPO_ROOT / "knowledge" / "datasheets",
        help="directory containing DatasheetDocument JSON files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "knowledge" / "datasheets" / "embeddings",
        help="output directory for <document_id>.npz files",
    )
    parser.add_argument(
        "--backend",
        default="openvino",
        choices=["openvino", "null"],
        help="embedding backend (null is for plumbing tests)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="OpenVINO IR directory (required for --backend openvino)",
    )
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--documents",
        nargs="*",
        default=[],
        help="restrict to specific document_id(s); default = all",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    backend = create_embedding_backend(
        args.backend,
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )
    if args.backend == "openvino" and not backend.is_active:
        logger.error(
            "openvino backend failed to load (model_dir=%s). "
            "Install with `pip install '.[embedding]'` and pass --model-dir "
            "pointing at an INT8 IR. See docs/rag-teaching-kb-design.md.",
            args.model_dir,
        )
        return 2

    json_paths = sorted(args.datasheets_dir.glob("*.json"))
    if not json_paths:
        logger.warning("no datasheet JSON files in %s", args.datasheets_dir)
        return 0

    selected = set(args.documents) if args.documents else None
    written = 0
    for json_path in json_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            document = DatasheetDocument.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip unreadable %s: %s", json_path.name, exc)
            continue
        if selected and document.document_id not in selected:
            continue

        chunk_ids, vectors = encode_document(document, backend, args.batch_size)
        if not chunk_ids:
            logger.info("%s: no chunks, skipping", document.document_id)
            continue
        if vectors.shape[1] == 0:
            logger.warning(
                "%s: backend returned empty vectors (is_active=%s); skipping",
                document.document_id,
                backend.is_active,
            )
            continue

        out_path = args.out_dir / f"{document.document_id}.npz"
        write_npz(out_path, chunk_ids, vectors)
        written += 1
        logger.info(
            "%s: wrote %d vectors (dim=%d) -> %s",
            document.document_id,
            vectors.shape[0],
            vectors.shape[1],
            out_path.relative_to(REPO_ROOT)
            if out_path.is_relative_to(REPO_ROOT)
            else out_path,
        )

    logger.info("done — wrote %d/%d documents", written, len(json_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
