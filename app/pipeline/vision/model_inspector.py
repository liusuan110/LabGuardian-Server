"""
YOLO weight inspection helpers.

在没有真正加载 Ultralytics 模型之前, 先从权重文件里做一次轻量合同检查:
- 这是 detect / obb / pose 哪一类模型
- 模型里大概有哪些类名
- pose 是否带 keypoint shape
"""

from __future__ import annotations

from pathlib import Path
import re
import zipfile


LABEL_KEYWORDS = (
    "resistor",
    "capacitor",
    "diode",
    "led",
    "wire",
    "jumper",
    "transistor",
    "potentiometer",
    "ic",
    "breadboard",
    "line_area",
    "pinned",
)

KNOWN_MODEL_LABELS = (
    "capacitor_ceramic",
    "capacitor_electrolytic",
    "diode",
    "jumper_wire",
    "led",
    "resistor",
    "transistor_3pin",
    "capacitor",
    "wire",
    "potentiometer",
    "ic",
    "breadboard",
    "line_area",
    "pinned",
)


def _normalize_label_token(token: str) -> str | None:
    lower = token.lower()
    if lower in KNOWN_MODEL_LABELS:
        return lower
    if len(lower) > 1:
        stripped_once = lower[:-1]
        if stripped_once in KNOWN_MODEL_LABELS and lower[-1] in "rqpthxg_":
            return stripped_once
    return None


def inspect_yolo_weight(path: str | Path) -> dict[str, object]:
    weight_path = Path(path)
    meta: dict[str, object] = {
        "path": str(weight_path),
        "exists": weight_path.exists(),
        "task": "unknown",
        "model_class": "unknown",
        "names": [],
        "kpt_shape": None,
    }
    if not weight_path.exists():
        return meta

    try:
        with zipfile.ZipFile(weight_path, "r") as zf:
            data_name = next((name for name in zf.namelist() if name.endswith("data.pkl")), None)
            if data_name is None:
                return meta
            raw = zf.read(data_name)
    except Exception:
        return meta

    text = raw.decode("latin1", "ignore")
    if "PoseModel" in text:
        meta["task"] = "pose"
        meta["model_class"] = "PoseModel"
    elif "OBBModel" in text:
        meta["task"] = "obb"
        meta["model_class"] = "OBBModel"
    elif "DetectionModel" in text:
        meta["task"] = "detect"
        meta["model_class"] = "DetectionModel"

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,40}", text)
    names: list[str] = []
    seen = set()
    for tok in tokens:
        normalized = _normalize_label_token(tok)
        if normalized is not None:
            if normalized not in seen:
                seen.add(normalized)
                names.append(normalized)
            continue
        lower = tok.lower()
        if any(key in lower for key in LABEL_KEYWORDS):
            candidate = lower
            if len(lower) > 1 and lower[-1] in "rqpthxg_":
                stripped_once = lower[:-1]
                if any(key in stripped_once for key in LABEL_KEYWORDS):
                    candidate = stripped_once
            if candidate not in seen:
                seen.add(candidate)
                names.append(candidate)
    meta["names"] = names

    idx = raw.find(b"kpt_shape")
    if idx >= 0:
        tail = raw[idx : idx + 64]
        vals = []
        for i in range(len(tail) - 1):
            if tail[i : i + 1] == b"K":
                vals.append(int(tail[i + 1]))
                if len(vals) >= 2:
                    break
        if len(vals) >= 2:
            meta["kpt_shape"] = vals[:2]

    return meta
