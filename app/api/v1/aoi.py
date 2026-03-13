"""
AOI (PCB 缺陷检测) API 路由

端点:
  POST /api/v1/pcb/train           — 训练 PatchCore 模型
  POST /api/v1/pcb/inspect         — PatchCore 检测单张 PCB 图片
  POST /api/v1/pcb/upload          — 上传黄金/缺陷样本
  GET  /api/v1/pcb/dataset         — 查看数据集信息
  GET  /api/v1/pcb/status          — 查看 AOI 模块状态
  DELETE /api/v1/pcb/dataset       — 清空数据集
  POST /api/v1/pcb/winclip/inspect — WinCLIP 零样本 PCB 缺陷检测
  GET  /api/v1/pcb/winclip/status  — WinCLIP 模块状态
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException

from app.core.config import settings, PROJECT_ROOT
from app.pipeline.aoi.data_manager import AOIDataManager
from app.pipeline.aoi.detector import PCBDefectDetector
from app.pipeline.aoi.winclip_detector import WinCLIPDetector
from app.schemas.aoi import (
    AOIDatasetInfo,
    AOIInspectRequest,
    AOIInspectResponse,
    AOIStatusResponse,
    AOITrainRequest,
    AOITrainResponse,
    AOIUploadRequest,
    AOIUploadResponse,
    WinCLIPInspectRequest,
    WinCLIPInspectResponse,
    WinCLIPStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pcb", tags=["pcb-aoi"])

# ── 线程安全单例 ──
_lock = threading.Lock()
_data_manager: AOIDataManager | None = None
_detector: PCBDefectDetector | None = None
_winclip_detector: WinCLIPDetector | None = None


def _get_data_manager() -> AOIDataManager:
    global _data_manager
    if _data_manager is None:
        with _lock:
            if _data_manager is None:
                _data_manager = AOIDataManager(
                    datasets_root=PROJECT_ROOT / settings.AOI_DATASET_ROOT,
                )
    return _data_manager


def _get_detector() -> PCBDefectDetector:
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                _detector = PCBDefectDetector(
                    model_dir=PROJECT_ROOT / settings.AOI_MODEL_DIR,
                    backbone=settings.AOI_BACKBONE,
                    coreset_sampling_ratio=settings.AOI_CORESET_RATIO,
                    num_neighbors=settings.AOI_NUM_NEIGHBORS,
                    image_size=(settings.AOI_IMAGE_SIZE, settings.AOI_IMAGE_SIZE),
                    score_threshold=settings.AOI_SCORE_THRESHOLD,
                )
    return _detector


def _get_winclip_detector() -> WinCLIPDetector:
    global _winclip_detector
    if _winclip_detector is None:
        with _lock:
            if _winclip_detector is None:
                _winclip_detector = WinCLIPDetector(
                    class_name="printed circuit board",
                    k_shot=0,
                    score_threshold=0.5,
                )
    return _winclip_detector


# ── 训练 ──


@router.post("/train", response_model=AOITrainResponse)
async def train_model(request: AOITrainRequest):
    """训练 PatchCore 模型 (仅需正常 PCB 图片)

    如果不提供 golden_dir, 则使用已上传的黄金样本集。
    """
    dm = _get_data_manager()
    detector = _get_detector()

    golden_dir = request.golden_dir or str(dm.good_dir)
    if dm.golden_count == 0 and request.golden_dir is None:
        raise HTTPException(
            status_code=400,
            detail="No golden samples found. Upload samples first via POST /pcb/upload",
        )

    if request.backbone != detector.backbone:
        detector.backbone = request.backbone
    if request.coreset_sampling_ratio != detector.coreset_sampling_ratio:
        detector.coreset_sampling_ratio = request.coreset_sampling_ratio

    try:
        result = detector.train(
            golden_dir=golden_dir,
            abnormal_dir=str(dm.defect_dir) if dm.defect_count > 0 else None,
        )
        return AOITrainResponse(
            status=result["status"],
            checkpoint=result["checkpoint"],
            golden_count=dm.golden_count,
            duration_ms=result["duration_ms"],
        )
    except Exception as exc:
        logger.exception("AOI train failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── 检测 ──


@router.post("/inspect", response_model=AOIInspectResponse)
async def inspect_pcb(request: AOIInspectRequest):
    """检测单张 PCB 图片是否存在缺陷

    score_threshold 作为参数传入 predict(), 不再修改全局 detector 状态。
    """
    detector = _get_detector()

    try:
        result = detector.predict(
            image_b64=request.image_b64,
            score_threshold=request.score_threshold,
        )
        return AOIInspectResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("AOI inspect failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── 数据集管理 ──


@router.post("/upload", response_model=AOIUploadResponse)
async def upload_sample(request: AOIUploadRequest):
    """上传训练样本图片"""
    dm = _get_data_manager()
    if request.category not in ("good", "defect"):
        raise HTTPException(status_code=400, detail="category must be 'good' or 'defect'")

    try:
        if request.category == "good":
            name = dm.upload_golden_sample(request.image_b64, request.filename)
        else:
            name = dm.upload_defect_sample(request.image_b64, request.filename)
        return AOIUploadResponse(filename=name, category=request.category)
    except Exception as exc:
        logger.exception("AOI upload failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/dataset", response_model=AOIDatasetInfo)
async def get_dataset_info():
    """查看当前 AOI 数据集信息"""
    dm = _get_data_manager()
    detector = _get_detector()
    samples = dm.list_samples()
    return AOIDatasetInfo(
        golden_count=len(samples["good"]),
        defect_count=len(samples["defect"]),
        good_files=samples["good"],
        defect_files=samples["defect"],
        model_trained=detector.is_trained,
    )


@router.delete("/dataset")
async def clear_dataset():
    """清空 AOI 数据集"""
    dm = _get_data_manager()
    dm.clear_dataset()
    return {"status": "ok", "message": "Dataset cleared"}


@router.get("/status", response_model=AOIStatusResponse)
async def get_aoi_status():
    """查看 AOI 模块状态"""
    dm = _get_data_manager()
    detector = _get_detector()
    return AOIStatusResponse(
        model_trained=detector.is_trained,
        checkpoint_path=detector.checkpoint_path,
        golden_count=dm.golden_count,
        defect_count=dm.defect_count,
        backbone=detector.backbone,
        score_threshold=detector.score_threshold,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WinCLIP 零样本检测 (无需训练!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/winclip/inspect", response_model=WinCLIPInspectResponse)
async def winclip_inspect(request: WinCLIPInspectRequest):
    """WinCLIP 零样本 PCB 缺陷检测

    score_threshold 作为参数传入 predict(), 不再修改全局 detector 状态。
    """
    detector = _get_winclip_detector()

    try:
        result = detector.predict(
            image_b64=request.image_b64,
            score_threshold=request.score_threshold,
        )
        return WinCLIPInspectResponse(
            anomaly_score=result["anomaly_score"],
            is_defective=result["is_defective"],
            anomaly_map_b64=result["anomaly_map_b64"],
            duration_ms=result["duration_ms"],
        )
    except Exception as exc:
        logger.exception("WinCLIP inspect failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/winclip/status", response_model=WinCLIPStatusResponse)
async def winclip_status():
    """查看 WinCLIP 零样本检测模块状态"""
    detector = _get_winclip_detector()
    return WinCLIPStatusResponse(
        is_ready=detector.is_ready,
        class_name=detector.class_name,
        k_shot=detector.k_shot,
        score_threshold=detector.score_threshold,
    )
