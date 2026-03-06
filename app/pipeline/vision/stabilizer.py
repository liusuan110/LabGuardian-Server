"""
多帧检测稳定器 (← src_v2/vision/stabilizer.py)

滑动窗口投票, 减少单帧抖动和误检
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


class DetectionStabilizer:
    """检测结果稳定器 — 滑动窗口投票"""

    def __init__(self, window_size: int = 5, min_hits: int = 3):
        self.window_size = window_size
        self.min_hits = min_hits
        self._history: List[List[dict]] = []

    def update(self, detections: List[dict]) -> List[dict]:
        """输入当前帧检测, 输出稳定化后的检测"""
        self._history.append(detections)
        if len(self._history) > self.window_size:
            self._history.pop(0)

        # 投票: 同类型+相似位置在 >= min_hits 帧出现
        vote_map: Dict[str, int] = defaultdict(int)
        latest_det: Dict[str, dict] = {}

        for frame_dets in self._history:
            for det in frame_dets:
                key = self._detection_key(det)
                vote_map[key] += 1
                latest_det[key] = det

        return [
            latest_det[key]
            for key, count in vote_map.items()
            if count >= self.min_hits
        ]

    @staticmethod
    def _detection_key(det: dict) -> str:
        """生成检测结果的位置键 (用于投票去重)"""
        cls = det.get("class_name", "")
        bbox = det.get("bbox", (0, 0, 0, 0))
        cx = (bbox[0] + bbox[2]) // 2 // 20 * 20  # 量化到 20px 网格
        cy = (bbox[1] + bbox[3]) // 2 // 20 * 20
        return f"{cls}_{cx}_{cy}"

    def reset(self):
        self._history.clear()
