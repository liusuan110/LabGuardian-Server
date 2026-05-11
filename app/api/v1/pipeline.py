"""
Pipeline 任务提交 API

客户端提交图片 → 返回 job_id → 轮询/WebSocket 获取结果
参考: GregaVrbancic/fastapi-celery 的 task 提交模式
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import get_classroom, get_guidance_service, get_pipeline_service
from app.domain.graph_compare import compare_logical_graphs
from app.domain.logical_reference import current_netlist_v2_to_graph, logical_reference_to_graph
from app.schemas.pipeline import (
    CircuitAnalysisResult,
    CompareNetlistRequest,
    CompareNetlistResponse,
    CorrectedRecomputeRequest,
    JobStatusResponse,
    NetlistVisualization,
    PipelineRequest,
    PipelineResult,
)
from app.services.classroom_state import ClassroomState
from app.services.guidance_service import GuidanceService
from app.services.pipeline_service import PipelineService
from app.services.reference_service import ReferenceService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineResult)
async def run_pipeline_sync(
    request: PipelineRequest,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    同步执行 Pipeline (演示用) — 直接返回完整结果，无需 Celery/Redis
    """
    try:
        return pipeline_service.run_sync(
            request=request,
            classroom=classroom,
            guidance_service=guidance_service,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/recompute-corrected", response_model=PipelineResult)
async def recompute_corrected_pipeline(
    request: CorrectedRecomputeRequest,
    classroom: ClassroomState = Depends(get_classroom),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    应用前端手动孔位修正，并重跑 S3/S4，返回可被后续模块直接消费的正式结果。
    """
    try:
        result = pipeline_service.recompute_corrected(request)
        pipeline_service.sync_corrected_result_to_classroom(
            classroom=classroom,
            request=request,
            result=result,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/submit", response_model=JobStatusResponse)
async def submit_pipeline(
    request: PipelineRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    提交四阶段 Pipeline 任务 (异步，需要 Celery+Redis)
    演示阶段请使用 POST /pipeline/run (同步)
    """
    try:
        return pipeline_service.submit_async(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Celery unavailable: {exc}")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_pipeline_status(
    job_id: str,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """查询 Pipeline 任务状态"""
    return pipeline_service.get_job_status(job_id)


@router.post("/analyze", response_model=CircuitAnalysisResult)
async def analyze_circuit(
    request: PipelineRequest,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    电路分析接口 — 返回元件引脚二维定位和网表信息
    
    执行完整的图像处理和电路分析流程，返回：
    - 元件及其引脚的二维定位信息（包含逻辑坐标和图像坐标）
    - 电气连接关系（网表）
    - 电路拓扑图（用于前端可视化）
    
    输入：
    - images_b64: 1-3张面包板俯拍图 (base64 JPEG)
    - station_id: 工作站ID
    - conf: YOLO置信度阈值（可选，默认0.25）
    - iou: YOLO NMS IoU阈值（可选，默认0.5）
    - imgsz: YOLO推理尺寸（可选，默认960）
    - rail_assignments: 电源轨道分配（可选）
    - reference_circuit: 参考电路（可选）
    
    输出：
    - components: 元件列表，每个元件包含引脚定位信息
    - nets: 电气网络列表
    - topology_graph: 节点链接格式的拓扑图
    - circuit_description: 电路文本描述
    """
    try:
        return pipeline_service.analyze_circuit(
            request=request,
            classroom=classroom,
            guidance_service=guidance_service,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/compare-netlist", response_model=CompareNetlistResponse)
async def compare_netlist(request: CompareNetlistRequest):
    """
    调试接口 — 直接比较 reference 与 current_netlist_v2，不跑完整识别流程。
    """
    ref_payload = None
    if request.reference_id:
        ref_payload = ReferenceService().load_reference(request.reference_id)
    elif request.reference_circuit:
        ref_payload = request.reference_circuit

    if not ref_payload:
        raise HTTPException(status_code=400, detail="必须提供 reference_id 或 reference_circuit")

    if ref_payload.get("format") != "logical_reference_v1":
        raise HTTPException(
            status_code=400,
            detail="当前仅支持 format='logical_reference_v1' 的参考电路",
        )

    current_netlist_v2 = request.current_netlist_v2
    if not current_netlist_v2:
        raise HTTPException(status_code=400, detail="必须提供 current_netlist_v2")

    try:
        reference_graph = logical_reference_to_graph(ref_payload)
        current_graph = current_netlist_v2_to_graph(current_netlist_v2)
        result = compare_logical_graphs(
            reference_graph,
            current_graph,
            ref_payload=ref_payload,
            cur_netlist_v2=current_netlist_v2,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"比较失败: {exc}")

    report = result.get("report", {})
    diagnostics = [str(item.get("message")) for item in report.get("items", []) if item.get("message")]
    if not diagnostics:
        diagnostics = [str(result.get("message") or "电路逻辑连接与参考电路一致")]

    return CompareNetlistResponse(
        is_correct=bool(result.get("logic_correct", False)),
        similarity=float(result.get("similarity", 0.0)),
        progress=float(result.get("progress", 0.0)),
        diagnostics=diagnostics,
        risk_level="safe" if result.get("logic_correct") else "warning",
        comparison_report=report,
    )


@router.post("/visualize/ports", response_model=NetlistVisualization)
async def get_port_mapping(
    request: PipelineRequest,
    classroom: ClassroomState = Depends(get_classroom),
    guidance_service: GuidanceService = Depends(get_guidance_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    """
    前端端口映射接口 — 返回元件引脚的二维面包板坐标，便于检验结果准确性
    
    此接口专为前端网表可视化设计，返回：
    - 每个元件引脚的二维面包板行列坐标（行号、列名、孔洞ID）
    - 引脚所属的电气网络信息
    - 电源轨标识和电源角色
    
    输入：
    - images_b64: 1-3张面包板俯拍图 (base64 JPEG)
    - station_id: 工作站ID
    - conf: YOLO置信度阈值（可选，默认0.25）
    - iou: YOLO NMS IoU阈值（可选，默认0.5）
    - imgsz: YOLO推理尺寸（可选，默认960）
    - rail_assignments: 电源轨道分配（可选）
    
    输出示例（端口映射列表）：
    ```json
    {
      "ports": [
        {
          "component_id": "R1",
          "component_type": "Resistor",
          "pin_id": 1,
          "pin_name": "1",
          "row_number": 5,
          "col_name": "a",
          "hole_id": "A5",
          "logic_loc": ["5", "a"],
          "is_power_rail": false,
          "power_role": null,
          "net_id": "NET_001",
          "net_name": "VCC"
        }
      ],
      "nets": [...],
      "components": [...],
      "component_count": 5,
      "pin_count": 12,
      "net_count": 4
    }
    ```
    
    前端使用建议：
    1. 在网表表格中显示 row_number + col_name（如 "5a"）
    2. 使用 is_power_rail 标记电源轨上的引脚
    3. 根据 power_role 显示不同颜色（VCC=红色, GND=蓝色）
    4. 使用 net_id/net_name 检验连接关系是否正确
    """
    try:
        # 执行电路分析
        analysis = pipeline_service.analyze_circuit(
            request=request,
            classroom=classroom,
            guidance_service=guidance_service,
        )
        
        # 转换为可视化数据结构
        return NetlistVisualization.from_circuit_analysis(analysis)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
