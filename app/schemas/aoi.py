"""
AOI Pydantic Schemas — PCB 缺陷检测 API 数据模型
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AOITrainRequest(BaseModel):
    """训练请求 — 指定黄金样本目录进行训练"""
    golden_dir: str | None = Field(
        default=None,
        description="正常 PCB 图片目录路径 (留空则使用默认数据集目录)",
    )
    backbone: str = Field(
        default="wide_resnet50_2",
        description="特征提取骨干网络",
    )
    coreset_sampling_ratio: float = Field(
        default=0.1,
        description="核心集采样比例 (越小速度越快, 越大精度越高)",
    )


class AOITrainResponse(BaseModel):
    """训练响应"""
    status: str
    checkpoint: str | None = None
    golden_count: int = 0
    duration_ms: float = 0.0


class AOIInspectRequest(BaseModel):
    """检测请求 — 提交一张 PCB 图片"""
    image_b64: str = Field(
        ..., description="base64 编码的 PCB 图片 (JPEG/PNG)",
    )
    score_threshold: float = Field(
        default=0.5,
        description="异常判定阈值 [0, 1]: 分数高于此值判为缺陷",
    )


class AOIInspectResponse(BaseModel):
    """检测响应 — 包含异常分数和热力图"""
    anomaly_score: float = Field(description="图像级异常分数 [0, 1]")
    is_defective: bool = Field(description="是否判定为缺陷")
    anomaly_map_b64: str = Field(description="热力图叠加原图的 base64 JPEG")
    duration_ms: float = 0.0


class AOIUploadRequest(BaseModel):
    """上传黄金/缺陷样本"""
    image_b64: str = Field(..., description="base64 编码的图片")
    filename: str | None = Field(default=None, description="文件名 (可选)")
    category: str = Field(
        default="good",
        description="类别: 'good' (黄金样本) 或 'defect' (缺陷样本)",
    )


class AOIUploadResponse(BaseModel):
    """上传响应"""
    filename: str
    category: str


class AOIDatasetInfo(BaseModel):
    """数据集信息"""
    golden_count: int
    defect_count: int
    good_files: list[str]
    defect_files: list[str]
    model_trained: bool


class AOIStatusResponse(BaseModel):
    """AOI 模块状态"""
    model_trained: bool
    checkpoint_path: str | None = None
    golden_count: int = 0
    defect_count: int = 0
    backbone: str = "wide_resnet50_2"
    score_threshold: float = 0.5


# ── WinCLIP 零样本检测 ──


class WinCLIPInspectRequest(BaseModel):
    """WinCLIP 零样本检测请求"""
    image_b64: str = Field(
        ..., description="base64 编码的 PCB 图片 (JPEG/PNG)",
    )
    score_threshold: float = Field(
        default=0.5,
        description="异常判定阈值 [0, 1]",
    )


class WinCLIPInspectResponse(BaseModel):
    """WinCLIP 零样本检测响应"""
    anomaly_score: float = Field(description="图像级异常分数")
    is_defective: bool = Field(description="是否判定为缺陷")
    anomaly_map_b64: str = Field(description="热力图叠加原图的 base64 JPEG")
    duration_ms: float = 0.0


class WinCLIPStatusResponse(BaseModel):
    """WinCLIP 模块状态"""
    is_ready: bool
    class_name: str = "printed circuit board"
    k_shot: int = 0
    score_threshold: float = 0.5
