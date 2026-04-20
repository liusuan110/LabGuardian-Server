"""
版本信息服务

负责统一输出当前代码、模型、知识库与规则版本。
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import settings


class VersionService:
    """统一管理对外暴露的版本信息."""

    def get_version_info(self) -> dict[str, Any]:
        return {
            "service": "labguardian-server",
            "code_version": settings.CODE_VERSION,
            "model_version": settings.MODEL_VERSION,
            "kb_version": settings.KB_VERSION,
            "rule_version": settings.RULE_VERSION,
            "llm_model": settings.LLM_MODEL or "",
            "api_prefix": settings.API_V1_PREFIX,
            "timestamp": time.time(),
        }
