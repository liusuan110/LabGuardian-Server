"""
LabGuardian Server — 全局配置 (Pydantic Settings)

参考: fastapi/full-stack-fastapi-template 的 config 模式
环境变量 / .env 文件优先, dataclass 给出默认值
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 (LabGuardian-Server/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DEMO_DIR = PROJECT_ROOT / "train_demo"


def _first_existing_path(*candidates: Path) -> Optional[str]:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _prefer_existing_path(current: Optional[str], fallback: Optional[str]) -> Optional[str]:
    if current:
        try:
            if Path(current).exists():
                return current
        except Exception:
            pass
    return fallback


def _normalize_runtime_device(requested: Optional[str], default: str = "cpu") -> str:
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


DEFAULT_COMPONENT_MODEL_PATH = _first_existing_path(
    TRAIN_DEMO_DIR / "detect_components" / "weights" / "best.pt",
)
DEFAULT_PIN_MODEL_PATH = _first_existing_path(
    TRAIN_DEMO_DIR / "models" / "weights" / "best.pt",
    TRAIN_DEMO_DIR / "pose_roi_context_v12" / "weights" / "best.pt",
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
    YOLO_MODEL_PATH: str = DEFAULT_COMPONENT_MODEL_PATH or str(TRAIN_DEMO_DIR / "detect_components" / "weights" / "best.pt")
    # 当前视觉主路径使用 YOLO-Detect。OBB 仅保留兼容位，不参与默认主流程。
    YOLO_OBB_MODEL_PATH: Optional[str] = None
    YOLO_CONF_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.5
    YOLO_IMGSZ: int = 960
    YOLO_DEVICE: str = "cpu"
    PIN_MODEL_PATH: Optional[str] = DEFAULT_PIN_MODEL_PATH
    PIN_MODEL_DEVICE: str = "cpu"

    # ---- 面包板校准 ----
    BREADBOARD_ROWS: int = 63
    BREADBOARD_COLS_PER_SIDE: int = 5
    ROI_PADDING: int = 30

    # ---- Pipeline ----
    PIPELINE_HIGH_RES_IMGSZ: int = 1280
    PIN_CANDIDATE_K: int = 5
    REFERENCE_CIRCUIT_PATH: Optional[str] = None

    # ---- 课堂 ----
    STATION_ONLINE_TIMEOUT: float = 10.0

    # ---- LLM (可选) ----
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    LLM_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ---- Knowledge Base (Datasheet RAG) ----
    KB_STORAGE_DIR: str = str(PROJECT_ROOT / "artifacts" / "kb")
    KB_COLLECTION: str = "labguardian_kb"
    KB_DEFAULT_TOP_K: int = 6
    TEACHING_KB_DIR: str = str(PROJECT_ROOT / "knowledge" / "teaching_scenes")
    FAULT_CASE_KB_DIR: str = str(PROJECT_ROOT / "knowledge" / "fault_cases")

    # ---- Local VLM (optional, edge deployment) ----
    VLM_PROVIDER: str = "template"
    VLM_BASE_URL: Optional[str] = None
    VLM_MODEL: Optional[str] = None
    VLM_TIMEOUT_S: float = 30.0
    VLM_OPENVINO_MODEL_DIR: Optional[str] = None
    VLM_OPENVINO_DEVICE: str = "CPU"
    VLM_OPENVINO_CACHE_DIR: Optional[str] = None
    VLM_MAX_NEW_TOKENS: int = 256


settings = Settings()
settings.YOLO_MODEL_PATH = _prefer_existing_path(settings.YOLO_MODEL_PATH, DEFAULT_COMPONENT_MODEL_PATH) or settings.YOLO_MODEL_PATH
settings.PIN_MODEL_PATH = _prefer_existing_path(settings.PIN_MODEL_PATH, DEFAULT_PIN_MODEL_PATH)
settings.YOLO_DEVICE = _normalize_runtime_device(settings.YOLO_DEVICE)
settings.PIN_MODEL_DEVICE = _normalize_runtime_device(settings.PIN_MODEL_DEVICE)
settings.REFERENCE_CIRCUIT_PATH = _prefer_existing_path(settings.REFERENCE_CIRCUIT_PATH, None)
