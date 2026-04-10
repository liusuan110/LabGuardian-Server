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
MODELS_DIR = PROJECT_ROOT / "models"


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
    YOLO_MODEL_PATH: str = str(MODELS_DIR / "yolov8n.pt")
    YOLO_OBB_MODEL_PATH: Optional[str] = None   # 引脚检测 OBB 模型，训练完成后填写
    YOLO_CONF_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.5
    YOLO_IMGSZ: int = 960
    YOLO_DEVICE: str = "cpu"

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


settings = Settings()
