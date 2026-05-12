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
        agent_provider = str(getattr(settings, "AGENT_LLM_PROVIDER", "template") or "template")
        if agent_provider == "ollama":
            agent_model = str(getattr(settings, "AGENT_LLM_OLLAMA_MODEL", "") or "")
        elif agent_provider == "openvino_genai_text":
            agent_model = str(getattr(settings, "AGENT_LLM_OPENVINO_MODEL_DIR", "") or "")
        else:
            agent_model = "template"
        return {
            "service": "labguardian-server",
            "code_version": settings.CODE_VERSION,
            "model_version": settings.MODEL_VERSION,
            "kb_version": settings.KB_VERSION,
            "rule_version": settings.RULE_VERSION,
            "llm_model": settings.LLM_MODEL or "",
            "agent_llm_provider": agent_provider,
            "agent_llm_model": agent_model,
            "api_prefix": settings.API_V1_PREFIX,
            "timestamp": time.time(),
        }
