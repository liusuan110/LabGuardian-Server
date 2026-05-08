"""
LabGuardian Server — 全局配置 (Pydantic Settings)

参考: fastapi/full-stack-fastapi-template 的 config 模式
环境变量 / .env 文件优先, dataclass 给出默认值
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 (LabGuardian-Server/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DEMO_DIR = PROJECT_ROOT / "train_demo"
DEFAULT_MODEL_ROOT_CANDIDATES = (
    Path("/opt/labguardian/models"),
    Path("/app/models"),
    PROJECT_ROOT / "models",
)


def _first_existing_path(*candidates: Path) -> str | None:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_model_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _prefer_existing_path(current: str | None, fallback: str | None) -> str | None:
    if current:
        try:
            if Path(current).exists():
                return current
        except Exception:
            pass
    return fallback


def _normalize_runtime_device(requested: str | None, default: str = "cpu") -> str:
    value = str(requested or default).strip()
    if not value:
        return default
    if value.lower() == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return value
    except Exception:
        pass
    return "cpu"


DEFAULT_MODEL_ROOT = _first_existing_path(*DEFAULT_MODEL_ROOT_CANDIDATES)
MODEL_ROOT = Path(DEFAULT_MODEL_ROOT) if DEFAULT_MODEL_ROOT else PROJECT_ROOT / "models"

DEFAULT_COMPONENT_MODEL_PATH = _first_existing_path(
    MODEL_ROOT / "component" / "best.pt",
    MODEL_ROOT / "component_detector" / "best.pt",
    MODEL_ROOT / "detect_components" / "best.pt",
    MODEL_ROOT / "detect_components" / "weights" / "best.pt",
    TRAIN_DEMO_DIR / "detect_components" / "weights" / "best.pt",
)
# Pin main path uses full-image YOLO-Pose. Do not auto-select legacy ROI weights here.
DEFAULT_PIN_MODEL_PATH = _first_existing_path(
    MODEL_ROOT / "pose_components" / "best.pt",
    MODEL_ROOT / "pose_components" / "weights" / "best.pt",
    TRAIN_DEMO_DIR / "pose_components" / "weights" / "best.pt",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Server ----
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    CODE_VERSION: str = "0.1.0"
    MODEL_VERSION: str = "dev"
    KB_VERSION: str = "none"
    RULE_VERSION: str = "dev"

    # ---- CORS ----
    CORS_ORIGINS: list[str] = ["*"]

    # ---- Redis / Celery ----
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ---- YOLO ----
    LABGUARDIAN_MODEL_ROOT: str = DEFAULT_MODEL_ROOT or str(MODEL_ROOT)
    YOLO_MODEL_PATH: str = DEFAULT_COMPONENT_MODEL_PATH or str(
        TRAIN_DEMO_DIR / "detect_components" / "weights" / "best.pt"
    )
    # 当前视觉主路径使用 YOLO-Detect。OBB 仅保留兼容位，不参与默认主流程。
    YOLO_OBB_MODEL_PATH: str | None = None
    YOLO_CONF_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.5
    YOLO_IMGSZ: int = 960
    YOLO_DEVICE: str = "cpu"
    # 当前 pin 主路径使用 full-image YOLO-Pose。ROI 训练权重仅保留兼容候选位。
    PIN_MODEL_PATH: str | None = DEFAULT_PIN_MODEL_PATH
    PIN_MODEL_DEVICE: str = "cpu"

    # ---- 面包板校准 ----
    BREADBOARD_ROWS: int = 63
    BREADBOARD_COLS_PER_SIDE: int = 5
    ROI_PADDING: int = 30

    # ---- Pipeline ----
    PIPELINE_HIGH_RES_IMGSZ: int = 960
    PIN_CANDIDATE_K: int = 5
    REFERENCE_CIRCUIT_PATH: str | None = None

    # ---- S2 引脚孔位吸附调参 (Tier 1) ----
    # 距离门：在 calibrator.board_point_to_logic_candidates 中限制候选孔范围
    # gate = min(min_dist + max(pitch * LOWER, 3.0), max(pitch * UPPER, 8.0))
    # 默认 (0.35, 0.95) 偏宽，把 0.95 收到 0.65 后相邻孔不再无条件入选；
    # 透视形变严重时 (>30°) 可放回 (0.30, 0.85)
    MAPPING_DIST_GATE_LOWER: float = 0.25
    MAPPING_DIST_GATE_UPPER: float = 0.65
    # 多视图加权投票时的 rank 衰减底数：base ** rank
    # 0.72 偏柔和易翻盘；0.45 让 top-1 几乎垄断票权
    MAPPING_VOTE_RANK_DECAY: float = 0.45
    # close_vote_margin 阈值：top-1 与 top-2 票数差小于该值即标 ambiguous
    MAPPING_AMBIGUOUS_MARGIN: float = 0.35

    # ---- 课堂 ----
    STATION_ONLINE_TIMEOUT: float = 10.0

    # ---- LLM (可选) ----
    LLM_API_KEY: str | None = None
    LLM_BASE_URL: str | None = None
    LLM_MODEL: str | None = None
    LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ---- Knowledge Base (Datasheet RAG) ----
    KB_STORAGE_DIR: str = str(PROJECT_ROOT / "artifacts" / "kb")
    KB_COLLECTION: str = "labguardian_kb"
    KB_DEFAULT_TOP_K: int = 6
    TEACHING_KB_DIR: str = str(PROJECT_ROOT / "knowledge" / "teaching_scenes")
    FAULT_CASE_KB_DIR: str = str(PROJECT_ROOT / "knowledge" / "fault_cases")

    # ---- Local VLM (optional, edge deployment) ----
    VLM_PROVIDER: str = "template"
    VLM_BASE_URL: str | None = None
    VLM_MODEL: str | None = None
    VLM_TIMEOUT_S: float = 30.0
    VLM_OPENVINO_MODEL_DIR: str | None = None
    VLM_OPENVINO_DEVICE: str = "CPU"
    VLM_OPENVINO_CACHE_DIR: str | None = None
    VLM_MAX_NEW_TOKENS: int = 256

    # ---- Diagnostic Agent ReAct loop (Phase 4) ----
    AGENT_LLM_PROVIDER: str = "template"  # template | openvino_genai_text (Phase 7+)
    REACT_MAX_ITERATIONS: int = 4
    AGENT_LLM_OPENVINO_MODEL_DIR: str | None = None
    AGENT_LLM_OPENVINO_DEVICE: str = "GPU"

    # ---- Hardware telemetry (Phase 5) ----
    TELEMETRY_ENABLED: bool = True
    TELEMETRY_HZ: float = 5.0
    TELEMETRY_RING_SECONDS: int = 120

    @field_validator("DEBUG", mode="before")
    @classmethod
    def _parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "off"}:
                return False
            if normalized in {"debug", "dev", "development", "on"}:
                return True
        return value


settings = Settings()
model_root = Path(settings.LABGUARDIAN_MODEL_ROOT)
component_candidates = (
    model_root / "component" / "best.pt",
    model_root / "component_detector" / "best.pt",
    model_root / "detect_components" / "best.pt",
    model_root / "detect_components" / "weights" / "best.pt",
)
pin_candidates = (
    model_root / "pose_components" / "best.pt",
    model_root / "pose_components" / "weights" / "best.pt",
)
settings.YOLO_MODEL_PATH = (
    _prefer_existing_path(_resolve_model_path(settings.YOLO_MODEL_PATH), None)
    or _first_existing_path(*component_candidates)
    or DEFAULT_COMPONENT_MODEL_PATH
    or settings.YOLO_MODEL_PATH
)
settings.PIN_MODEL_PATH = (
    _prefer_existing_path(_resolve_model_path(settings.PIN_MODEL_PATH), None)
    or _first_existing_path(*pin_candidates)
    or DEFAULT_PIN_MODEL_PATH
)
settings.YOLO_DEVICE = _normalize_runtime_device(settings.YOLO_DEVICE)
settings.PIN_MODEL_DEVICE = _normalize_runtime_device(settings.PIN_MODEL_DEVICE)
settings.REFERENCE_CIRCUIT_PATH = _prefer_existing_path(settings.REFERENCE_CIRCUIT_PATH, None)
