"""
元件极性推断 (← src_v2/logic/polarity.py)

基于 OBB 几何信息 + 元件类型推断 LED 极性方向和引脚角色
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .circuit import (
    CircuitComponent,
    Polarity,
    PinRole,
    POLARIZED_TYPES,
    NON_POLAR_TYPES,
    norm_component_type,
)

logger = logging.getLogger(__name__)


class PolarityResolver:
    """元件极性解析器 (LED-only)"""

    def __init__(self, board_rows: int = 30):
        self.board_rows = board_rows
        self.stats = {"total": 0, "resolved": 0, "unknown": 0}

    def reset_stats(self):
        self.stats = {"total": 0, "resolved": 0, "unknown": 0}

    def enrich(
        self,
        comp: CircuitComponent,
        obb_corners: Optional[np.ndarray] = None,
        orientation_deg: float = 0.0,
    ) -> CircuitComponent:
        """填充极性和引脚角色 (原地修改)"""
        self.stats["total"] += 1
        norm_type = self._norm_type(comp.type)
        comp.orientation_deg = orientation_deg

        if norm_type in NON_POLAR_TYPES or norm_type == "UNKNOWN":
            comp.polarity = Polarity.NONE
            return comp

        if norm_type in POLARIZED_TYPES:
            self._resolve_diode_polarity(comp, obb_corners, orientation_deg)
        else:
            comp.polarity = Polarity.NONE

        return comp

    def _resolve_diode_polarity(
        self,
        comp: CircuitComponent,
        obb_corners: Optional[np.ndarray],
        orientation_deg: float,
    ):
        if comp.pin1_loc is None or comp.pin2_loc is None:
            comp.polarity = Polarity.UNKNOWN
            self.stats["unknown"] += 1
            return

        try:
            int(comp.pin1_loc[0])
            int(comp.pin2_loc[0])
        except (ValueError, TypeError):
            comp.polarity = Polarity.UNKNOWN
            self.stats["unknown"] += 1
            return

        if obb_corners is not None and len(obb_corners) == 4:
            comp.polarity = Polarity.FORWARD
        else:
            comp.polarity = Polarity.FORWARD

        self.stats["resolved"] += 1

    @staticmethod
    def _obb_long_axis_direction(corners: np.ndarray) -> float:
        p0, p1, p2, p3 = corners
        d01 = np.linalg.norm(p0 - p1)
        d12 = np.linalg.norm(p1 - p2)
        if d01 < d12:
            start = (p0 + p1) / 2
            end = (p2 + p3) / 2
        else:
            start = (p1 + p2) / 2
            end = (p3 + p0) / 2
        return float(end[1] - start[1])

    @staticmethod
    def _norm_type(t: str) -> str:
        return norm_component_type(t)
