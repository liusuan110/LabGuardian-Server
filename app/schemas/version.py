"""
版本信息响应模型
"""

from __future__ import annotations

from pydantic import BaseModel


class VersionInfoResponse(BaseModel):
    service: str
    code_version: str
    model_version: str
    kb_version: str
    rule_version: str
    llm_model: str
    api_prefix: str
    timestamp: float
