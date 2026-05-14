"""Pluggable text embedding backends for Datasheet KB v2.

Phase 1 shipped only ``NullEmbeddingBackend`` — the board encodes nothing,
keyword retrieval handles everything. Phase 3 adds ``OpenVINOEmbeddingBackend``
which loads a local INT8 IR model (e.g. ``bge-small-zh-v1.5``) via the OpenVINO
runtime that is *already* on-device for ``vlm_service``. No new accelerator
stack is introduced.

On-device assumptions
=====================

- The model directory holds an OpenVINO IR pair (``openvino_model.xml`` +
  ``openvino_model.bin``) plus a HuggingFace ``tokenizer.json``. Conversion is
  done **offline** on a developer machine (see Phase 3 docs).
- Chunk-side vectors are pre-computed offline by
  ``scripts/build_datasheet_embeddings.py`` and stored under
  ``knowledge/datasheets/embeddings/<document_id>.npz``. The board only ever
  encodes the *query* — never the entire datasheet corpus — so the heavy
  one-time cost stays off the device.
- If the model directory is missing or invalid, the backend reports
  ``is_active=False`` and ``DatasheetKbService`` transparently falls back to
  keyword-only retrieval. The board never crashes for lack of a model.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Interface — ``encode`` must return one row per input text."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return 0


class NullEmbeddingBackend(EmbeddingBackend):
    """No-op backend used when the board has no embedding model.

    Returns an empty (N, 0) matrix so callers can shape-check without
    branching. ``is_active=False`` signals to ``DatasheetKbService`` that
    cosine fusion must be skipped.
    """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 0), dtype=np.float32)

    @property
    def is_active(self) -> bool:
        return False


class OpenVINOEmbeddingBackend(EmbeddingBackend):
    """OpenVINO INT8 IR embedding backend.

    Lazy-loaded: the model + tokenizer are constructed on the first
    ``encode`` call so a missing or broken model directory doesn't prevent
    process startup. The board uses the same ``openvino`` runtime as
    ``vlm_service`` (no new accelerator stack).

    Pooling defaults to **mean-pool over non-padding tokens** then L2
    normalize, which matches the sentence-bge-* family. Output dtype is
    ``float32``.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "CPU",
        max_length: int = 256,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._device = device
        self._max_length = max_length
        self._lock = threading.Lock()
        self._loaded = False
        self._load_failed = False
        self._compiled = None
        self._tokenizer = None
        self._dim = 0
        self._input_names: list[str] = []

    def _try_load(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                import openvino as ov
                from tokenizers import Tokenizer
            except Exception as exc:  # noqa: BLE001
                logger.warning("openvino/tokenizers unavailable: %s", exc)
                self._load_failed = True
                return False

            xml = self._model_dir / "openvino_model.xml"
            tok_json = self._model_dir / "tokenizer.json"
            if not xml.exists() or not tok_json.exists():
                logger.info(
                    "embedding model dir incomplete (%s): falling back to keyword-only",
                    self._model_dir,
                )
                self._load_failed = True
                return False

            try:
                core = ov.Core()
                model = core.read_model(str(xml))
                self._compiled = core.compile_model(model, self._device)
                self._tokenizer = Tokenizer.from_file(str(tok_json))
                self._tokenizer.enable_truncation(max_length=self._max_length)
                self._tokenizer.enable_padding(length=None)
                self._input_names = [
                    inp.get_any_name() for inp in self._compiled.inputs
                ]
                out_shape = self._compiled.outputs[0].get_partial_shape()
                # Last static dim is the hidden size for BGE / MiniLM.
                if out_shape.rank.is_static:
                    last = out_shape[out_shape.rank.get_length() - 1]
                    if last.is_static:
                        self._dim = int(last.get_length())
                self._loaded = True
                logger.info(
                    "loaded embedding model dir=%s device=%s dim=%s",
                    self._model_dir,
                    self._device,
                    self._dim or "?",
                )
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load embedding model %s: %s", self._model_dir, exc)
                self._load_failed = True
                return False

    @property
    def is_active(self) -> bool:
        return self._try_load()

    @property
    def dim(self) -> int:
        self._try_load()
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        if not self._try_load():
            return np.zeros((len(texts), 0), dtype=np.float32)
        encodings = self._tokenizer.encode_batch(list(texts))
        # Pad to max length in the batch — Tokenizer.enable_padding(None) pads
        # to the longest item.
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type = np.zeros_like(ids)

        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            lname = name.lower()
            if "mask" in lname:
                feed[name] = mask
            elif "token_type" in lname or "segment" in lname:
                feed[name] = token_type
            else:
                feed[name] = ids

        outputs = self._compiled(feed)
        # OpenVINO returns a dict keyed by Output objects; first output is the
        # token-level hidden states for BGE / MiniLM.
        first_key = next(iter(outputs))
        hidden = np.asarray(outputs[first_key], dtype=np.float32)

        if hidden.ndim == 3:
            # (batch, seq, dim) → mean-pool over non-padding positions.
            mask_f = mask.astype(np.float32)[..., None]
            summed = (hidden * mask_f).sum(axis=1)
            counts = np.maximum(mask_f.sum(axis=1), 1.0)
            pooled = summed / counts
        elif hidden.ndim == 2:
            # Already pooled by the model itself.
            pooled = hidden
        else:
            raise RuntimeError(f"unexpected embedding output rank: {hidden.shape}")

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-9)).astype(np.float32)


def create_embedding_backend(
    kind: str | None,
    *,
    model_dir: str | Path | None = None,
    device: str = "CPU",
    max_length: int = 256,
) -> EmbeddingBackend:
    """Factory used by services / scripts; never raises on misconfig."""

    kind = (kind or "null").lower()
    if kind == "openvino":
        if not model_dir:
            logger.info("openvino embedding requested without model_dir; using null")
            return NullEmbeddingBackend()
        return OpenVINOEmbeddingBackend(
            model_dir=model_dir, device=device, max_length=max_length
        )
    return NullEmbeddingBackend()
