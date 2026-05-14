from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingBackend(ABC):
    """Pluggable text embedding backend.

    Phase 1 ships only NullEmbeddingBackend — the board needs no model weights
    and DatasheetKbService falls back to deterministic keyword scoring. Phase 3
    will add an OpenVINOEmbeddingBackend that loads a local INT8 IR.
    """

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        return False


class NullEmbeddingBackend(EmbeddingBackend):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[] for _ in texts]

    @property
    def is_active(self) -> bool:
        return False
