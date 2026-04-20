"""
面包板 schema 与孔位解析。

第一阶段先提供一个默认 schema，用来把旧的逻辑坐标 `(row, col)`
升级为显式 `hole_id / electrical_node_id`。后续接入正式
`board_schema.json` 时，可以在这里扩展真实板型、轨道分段和导通规则。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple


_GRID_RE = re.compile(r"^([A-J])(\d+)$")
_LEGACY_PLUS_RE = re.compile(r"^PWR_PLUS(?:_(\d+))?$")
_LEGACY_MINUS_RE = re.compile(r"^PWR_MINUS(?:_(\d+))?$")
_LEGACY_RAIL_RE = re.compile(r"^RAIL_([A-Z]+(?:_[A-Z]+)*)(?:_(\d+))?$")
_TRACK_RE = re.compile(r"^(LP|LN|RP|RN)(\d+)$")
_TRACK_TO_LEGACY_LOGIC = {
    "LP": "rail_top+",
    "LN": "rail_top-",
    "RP": "rail_bot+",
    "RN": "rail_bot-",
}
_LEGACY_LOGIC_TO_TRACK = {v: k for k, v in _TRACK_TO_LEGACY_LOGIC.items()}


@dataclass(frozen=True)
class HoleSpec:
    """单个孔位的静态定义。"""

    hole_id: str
    electrical_node_id: str
    group_type: str
    row: Optional[int] = None
    col: Optional[str] = None


@dataclass
class BoardSchema:
    """面包板孔位与静态导通规则。"""

    schema_id: str
    board_type: str
    aliases: Dict[str, str] = field(default_factory=dict)
    holes: Dict[str, HoleSpec] = field(default_factory=dict)

    @classmethod
    def default_breadboard(cls) -> "BoardSchema":
        default_path = Path(__file__).parent / "data" / "board_schemas" / "breadboard_legacy_v1.json"
        if default_path.exists():
            return cls.load_json(default_path)
        return cls(
            schema_id="breadboard_legacy_v1",
            board_type="legacy_breadboard",
            aliases={
                "PWR_PLUS": "PWR_PLUS",
                "PWR_MINUS": "PWR_MINUS",
            },
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BoardSchema":
        holes: Dict[str, HoleSpec] = {}
        for hole_id, item in (payload.get("holes") or {}).items():
            holes[str(hole_id).upper()] = HoleSpec(
                hole_id=str(item.get("hole_id") or hole_id).upper(),
                electrical_node_id=str(item.get("electrical_node_id") or f"NODE_{hole_id}"),
                group_type=str(item.get("group_type") or "custom"),
                row=int(item["row"]) if item.get("row") is not None else None,
                col=str(item["col"]) if item.get("col") is not None else None,
            )
        for group in payload.get("generated_groups") or []:
            for spec in cls._expand_generated_group(group):
                holes.setdefault(spec.hole_id, spec)
        return cls(
            schema_id=str(payload.get("schema_id") or "board_schema_unknown"),
            board_type=str(payload.get("board_type") or "custom_board"),
            aliases={str(k).upper(): str(v).upper() for k, v in (payload.get("aliases") or {}).items()},
            holes=holes,
        )

    @staticmethod
    def _expand_generated_group(group: Dict[str, Any]) -> list[HoleSpec]:
        kind = str(group.get("kind") or "").strip().lower()
        if kind == "main_strip":
            return BoardSchema._expand_main_strip(group)
        if kind == "track":
            return BoardSchema._expand_track(group)
        return []

    @staticmethod
    def _expand_main_strip(group: Dict[str, Any]) -> list[HoleSpec]:
        cols = [str(col).upper() for col in group.get("cols", [])]
        row_start = int(group.get("row_start", 1))
        row_end = int(group.get("row_end", row_start))
        side = str(group.get("side") or ("L" if cols and cols[0] in "ABCDE" else "R")).upper()
        node_template = str(group.get("electrical_node_template") or "ROW_{row}_{side}")
        group_type = str(group.get("group_type") or "main_grid")

        holes: list[HoleSpec] = []
        for row in range(row_start, row_end + 1):
            node_id = node_template.format(row=row, side=side)
            for col in cols:
                holes.append(
                    HoleSpec(
                        hole_id=f"{col}{row}",
                        electrical_node_id=node_id,
                        group_type=group_type,
                        row=row,
                        col=col,
                    )
                )
        return holes

    @staticmethod
    def _expand_track(group: Dict[str, Any]) -> list[HoleSpec]:
        track = str(group.get("track") or "").upper()
        if not track:
            return []
        group_type = str(group.get("group_type") or "track")
        node_template = str(group.get("electrical_node_template") or "TRACK_{track}{segment_suffix}")
        segments = group.get("segments") or [
            {
                "row_start": int(group.get("row_start", 1)),
                "row_end": int(group.get("row_end", group.get("row_start", 1))),
            }
        ]

        holes: list[HoleSpec] = []
        has_multiple_segments = len(segments) > 1
        for idx, segment in enumerate(segments, start=1):
            row_start = int(segment.get("row_start", 1))
            row_end = int(segment.get("row_end", row_start))
            suffix = str(segment.get("suffix") or (f"_SEG{idx}" if has_multiple_segments else ""))
            node_id = node_template.format(
                track=track,
                segment_index=idx,
                segment_suffix=suffix,
            )
            for row in range(row_start, row_end + 1):
                holes.append(
                    HoleSpec(
                        hole_id=f"{track}{row}",
                        electrical_node_id=node_id,
                        group_type=group_type,
                        row=row,
                        col=track,
                    )
                )
        return holes

    @classmethod
    def load_json(cls, path: str | Path) -> "BoardSchema":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "board_type": self.board_type,
            "aliases": dict(self.aliases),
            "holes": {
                hole_id: {
                    "hole_id": spec.hole_id,
                    "electrical_node_id": spec.electrical_node_id,
                    "group_type": spec.group_type,
                    "row": spec.row,
                    "col": spec.col,
                }
                for hole_id, spec in self.holes.items()
            },
        }

    def resolve_track_assignment_nodes(self, track_id: str) -> list[str]:
        """把 API/旧链路里的轨道名映射到 schema 内的静态节点。"""
        normalized = str(track_id or "").strip().lower()
        track = {
            "top_plus": "LP",
            "top_minus": "LN",
            "bot_plus": "RP",
            "bot_minus": "RN",
            "lp": "LP",
            "ln": "LN",
            "rp": "RP",
            "rn": "RN",
        }.get(normalized)
        if track is None:
            return []

        node_ids = {
            spec.electrical_node_id
            for spec in self.holes.values()
            if (spec.col or "").upper() == track
        }
        if node_ids:
            return sorted(node_ids)

        fallback = self.hole_to_spec(f"{track}1").electrical_node_id
        return [fallback]

    def normalize_hole_id(self, hole_id: str) -> str:
        value = str(hole_id or "").strip().upper()
        return self.aliases.get(value, value)

    def logic_loc_to_hole_id(self, loc: Tuple[str, str]) -> str:
        """旧 `(row, col)` 坐标到显式 hole_id 的兼容映射。"""
        row, col = str(loc[0]).strip(), str(loc[1]).strip()
        col_upper = col.upper()
        col_lower = col.lower()

        if len(col_upper) == 1 and col_upper in "ABCDEFGHIJ":
            return f"{col_upper}{row}"

        if col_upper in ("LP", "LN", "RP", "RN"):
            return f"{col_upper}{row}"

        if col in ("+", "plus", "P"):
            return f"PWR_PLUS_{row}"
        if col in ("-", "minus", "N", "GND"):
            return f"PWR_MINUS_{row}"

        if col_lower in _LEGACY_LOGIC_TO_TRACK:
            return f"{_LEGACY_LOGIC_TO_TRACK[col_lower]}{row}"
        if col.startswith("rail_"):
            suffix = col[5:].replace("+", "_plus").replace("-", "_minus").upper()
            return f"RAIL_{suffix}_{row}"

        return f"LEGACY_{col_upper}_{row}"

    def hole_id_to_logic_loc(self, hole_id: str) -> Optional[Tuple[str, str]]:
        """显式 hole_id 回退到旧 `(row, col)` 逻辑坐标，供旧链路兼容使用。"""
        normalized = self.normalize_hole_id(hole_id)

        m = _GRID_RE.match(normalized)
        if m:
            col, row = m.groups()
            return (row, col.lower())

        m = _LEGACY_PLUS_RE.match(normalized)
        if m:
            row = m.group(1) or "0"
            return (row, "+")

        m = _LEGACY_MINUS_RE.match(normalized)
        if m:
            row = m.group(1) or "0"
            return (row, "-")

        m = _LEGACY_RAIL_RE.match(normalized)
        if m:
            rail_key, row = m.groups()
            row = row or "0"
            col = f"rail_{rail_key.lower().replace('_plus', '+').replace('_minus', '-')}"
            return (row, col)

        m = _TRACK_RE.match(normalized)
        if m:
            track, row = m.groups()
            return (row, _TRACK_TO_LEGACY_LOGIC.get(track, track.lower()))

        return None

    def hole_to_spec(self, hole_id: str) -> HoleSpec:
        normalized = self.normalize_hole_id(hole_id)

        if normalized in self.holes:
            return self.holes[normalized]

        m = _GRID_RE.match(normalized)
        if m:
            col, row = m.groups()
            side = "L" if col in "ABCDE" else "R"
            return HoleSpec(
                hole_id=normalized,
                electrical_node_id=f"ROW_{row}_{side}",
                group_type="main_grid",
                row=int(row),
                col=col,
            )

        m = _TRACK_RE.match(normalized)
        if m:
            track, row = m.groups()
            return HoleSpec(
                hole_id=normalized,
                electrical_node_id=f"TRACK_{track}",
                group_type="track",
                row=int(row),
                col=track,
            )

        m = _LEGACY_PLUS_RE.match(normalized)
        if m:
            row = m.group(1)
            return HoleSpec(
                hole_id=normalized,
                electrical_node_id="PWR_PLUS",
                group_type="power",
                row=int(row) if row else None,
                col="+",
            )

        m = _LEGACY_MINUS_RE.match(normalized)
        if m:
            row = m.group(1)
            return HoleSpec(
                hole_id=normalized,
                electrical_node_id="PWR_MINUS",
                group_type="power",
                row=int(row) if row else None,
                col="-",
            )

        m = _LEGACY_RAIL_RE.match(normalized)
        if m:
            rail_key, row = m.groups()
            return HoleSpec(
                hole_id=normalized,
                electrical_node_id=f"RAIL_{rail_key}",
                group_type="rail",
                row=int(row) if row else None,
                col=rail_key,
            )

        return HoleSpec(
            hole_id=normalized,
            electrical_node_id=f"NODE_{normalized}",
            group_type="custom",
        )

    def resolve_hole_to_node(self, hole_id: str) -> str:
        return self.hole_to_spec(hole_id).electrical_node_id
