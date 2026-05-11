"""
数据模型 — Pipeline 相关
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PipelineStage(StrEnum):
    """四阶段名称"""

    DETECT = "detect"
    PIN_DETECT = "pin_detect"
    MAPPING = "mapping"
    TOPOLOGY = "topology"
    VALIDATE = "validate"
    SEMANTIC_ANALYSIS = "semantic_analysis"


class JobStatus(StrEnum):
    """任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRequest(BaseModel):
    """Pipeline 任务提交请求"""

    station_id: str
    images_b64: list[str] = Field(
        ..., min_length=1, max_length=3,
        description="1-3 张面包板俯拍图 (base64 JPEG)",
    )
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 960
    reference_id: str | None = None
    reference_circuit: dict[str, Any] | None = None
    rail_assignments: dict[str, str] | None = Field(
        default=None,
        description=(
            "面包板电源轨道指定, 如 "
            '{"top_plus": "VCC", "top_minus": "VCC", '
            '"bot_plus": "GND", "bot_minus": "GND"}'
        ),
    )
    net_role_assignments: list[ManualNetRoleAssignment] = Field(default_factory=list)


class ManualCorrectionPatch(BaseModel):
    """前端手动修正后的单个 pin 孔位覆盖。"""

    component_id: str
    pin_name: str
    from_hole_id: str
    to_hole_id: str
    source: str = "manual_drag"


class ManualNetRoleAssignment(BaseModel):
    """前端手动指定的网络角色（输入/输出/正电/地）。"""

    role: str
    source: str = "manual_netlist_select"
    hole_id: str | None = None
    component_id: str | None = None
    pin_name: str | None = None
    electrical_net_id: str | None = None
    electrical_node_id: str | None = None
    x_image: float | None = None
    y_image: float | None = None


class ManualPinPolarityAssignment(BaseModel):
    """前端手动指定三极管引脚极性。"""

    component_id: str
    pin_name: str
    polarity: str
    source: str = "manual_pin_polarity_select"


class CorrectedRecomputeRequest(BaseModel):
    """基于前端手动修正重算 S3/S4 的请求。"""

    station_id: str
    job_id: str | None = None
    components: list[dict[str, Any]] = Field(default_factory=list)
    corrections: list[ManualCorrectionPatch] = Field(default_factory=list)
    rail_assignments: dict[str, str] | None = None
    net_role_assignments: list[ManualNetRoleAssignment] = Field(default_factory=list)
    pin_polarity_assignments: list[ManualPinPolarityAssignment] = Field(default_factory=list)
    reference_id: str | None = None
    reference_circuit: dict[str, Any] | None = None


class StageResult(BaseModel):
    """单阶段执行结果"""

    stage: PipelineStage
    status: JobStatus = JobStatus.COMPLETED
    duration_ms: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """完整 Pipeline 结果"""

    job_id: str
    station_id: str
    status: JobStatus = JobStatus.COMPLETED
    stages: list[StageResult] = Field(default_factory=list)
    total_duration_ms: float = 0.0

    # ---- 汇总 ----
    component_count: int = 0
    net_count: int = 0
    progress: float = 0.0
    similarity: float = 0.0
    diagnostics: list[str] = Field(default_factory=list)
    comparison_report: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "safe"
    risk_reasons: list[str] = Field(default_factory=list)
    report: str = ""
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_pipeline_run(
        cls,
        *,
        job_id: str,
        station_id: str,
        raw: dict[str, Any],
    ) -> PipelineResult:
        """将编排器原始输出标准化为统一的 PipelineResult.

        兼容两种输入:
        1. 编排器原始结果: {"stages": {...}, "total_duration_ms": ...}
        2. 已序列化的 PipelineResult dict
        """
        if isinstance(raw.get("stages"), list) and "status" in raw:
            payload = dict(raw)
            payload.setdefault("job_id", job_id)
            payload.setdefault("station_id", station_id)
            return cls(**payload)

        stages_raw = raw.get("stages", {})
        stages = [
            StageResult(
                stage=PipelineStage(stage_name),
                duration_ms=stage_data.get("duration_ms", 0),
                data={k: v for k, v in stage_data.items() if k != "duration_ms"},
            )
            for stage_name, stage_data in stages_raw.items()
        ]
        s3 = stages_raw.get(PipelineStage.TOPOLOGY.value, {})
        s4 = stages_raw.get(PipelineStage.VALIDATE.value, {})
        return cls(
            job_id=job_id,
            station_id=station_id,
            status=JobStatus.COMPLETED,
            stages=stages,
            total_duration_ms=raw.get("total_duration_ms", 0),
            component_count=s3.get("component_count", 0),
            net_count=len(s3.get("netlist_v2", {}).get("nets", [])),
            progress=s4.get("progress", 0.0),
            similarity=s4.get("similarity", 0.0),
            diagnostics=s4.get("diagnostics", []),
            comparison_report=s4.get("comparison_report", {}),
            risk_level=s4.get("risk_level", "safe"),
            risk_reasons=s4.get("risk_reasons", []),
            runtime_metadata=raw.get("runtime_metadata", {}),
        )


class CompareNetlistRequest(BaseModel):
    """直接比较网表调试请求"""

    reference_id: str | None = None
    reference_circuit: dict[str, Any] | None = None
    current_netlist_v2: dict[str, Any] = Field(..., description="当前识别的 netlist_v2")


class CompareNetlistResponse(BaseModel):
    """直接比较网表调试响应"""

    is_correct: bool = False
    similarity: float = 0.0
    progress: float = 0.0
    diagnostics: list[str] = Field(default_factory=list)
    risk_level: str = "safe"
    comparison_report: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    """任务状态查询响应"""

    job_id: str
    status: JobStatus
    current_stage: PipelineStage | None = None
    result: PipelineResult | None = None


# ============================================================
# 电路分析响应模型
# ============================================================

# 面包板坐标解析辅助函数
import re

_HOLE_ID_RE = re.compile(r"^([A-J])(\d+)$")
_PWR_PLUS_RE = re.compile(r"^PWR_PLUS(?:_(\d+))?$")
_PWR_MINUS_RE = re.compile(r"^PWR_MINUS(?:_(\d+))?$")
_TRACK_RE = re.compile(r"^(LP|LN|RP|RN)(\d+)$")


def _parse_hole_id(hole_id: str, logic_loc: tuple[str, str] | None) -> tuple[int | None, str | None, bool]:
    """解析面包板孔洞ID，提取行号、列名和是否为电源轨
    
    Returns:
        (row_number, col_name, is_power_rail)
    """
    row_number = None
    col_name = None
    is_power_rail = False
    
    # 优先从 logic_loc 解析
    if logic_loc and len(logic_loc) >= 2:
        try:
            row_number = int(logic_loc[0])
        except (ValueError, TypeError):
            pass
        col_name = str(logic_loc[1]).lower()
        if col_name.startswith("rail_"):
            is_power_rail = True
            col_name = col_name.replace("rail_", "")
        elif col_name in ("+", "-", "plus", "minus"):
            is_power_rail = True
    
    # 从 hole_id 解析
    if hole_id:
        # 标准网格孔位: A1, B5, J10 等
        m = _HOLE_ID_RE.match(hole_id.upper())
        if m:
            col_name = m.group(1).lower()
            try:
                row_number = int(m.group(2))
            except ValueError:
                pass
            return (row_number, col_name, False)
        
        # 电源轨孔位: LP1, LN5, RP10, RN20
        m = _TRACK_RE.match(hole_id.upper())
        if m:
            track_type = m.group(1)
            try:
                row_number = int(m.group(2))
            except ValueError:
                pass
            is_power_rail = True
            col_name_map = {
                "LP": "rail_top+",
                "LN": "rail_top-", 
                "RP": "rail_bot+",
                "RN": "rail_bot-"
            }
            col_name = col_name_map.get(track_type, track_type.lower())
            return (row_number, col_name, True)
        
        # 旧格式电源轨
        m = _PWR_PLUS_RE.match(hole_id.upper())
        if m:
            is_power_rail = True
            col_name = "+"
            if m.group(1):
                try:
                    row_number = int(m.group(1))
                except ValueError:
                    pass
            return (row_number, col_name, True)
        
        m = _PWR_MINUS_RE.match(hole_id.upper())
        if m:
            is_power_rail = True
            col_name = "-"
            if m.group(1):
                try:
                    row_number = int(m.group(1))
                except ValueError:
                    pass
            return (row_number, col_name, True)
    
    return (row_number, col_name, is_power_rail)


class PinLocation(BaseModel):
    """元件引脚的二维定位信息"""

    pin_id: int = Field(..., description="引脚编号")
    pin_name: str = Field(..., description="引脚名称")
    pin_display_name: str | None = Field(None, description="前端展示用引脚标签（如 E/B/C）")
    polarity_role: str | None = Field(None, description="三极管极性角色（高置信）")
    polarity_candidate_role: str | None = Field(None, description="三极管极性候选角色（低置信回退）")
    hole_id: str = Field(..., description="面包板孔洞ID")
    logic_loc: tuple[str, str] | None = Field(None, description="逻辑坐标 (行, 列)")
    x_warp: float | None = Field(None, description="校正后二维图中的X坐标")
    y_warp: float | None = Field(None, description="校正后二维图中的Y坐标")
    x_image: float | None = Field(None, description="原始图像中的X坐标")
    y_image: float | None = Field(None, description="原始图像中的Y坐标")
    electrical_node_id: str | None = Field(None, description="电气节点ID")
    electrical_net_id: str | None = Field(None, description="电气网络ID")

    # 前端可视化增强字段
    row_number: int | None = Field(None, description="面包板行号 (1-63)")
    col_name: str | None = Field(None, description="面包板列名 (a-j 或电源轨标识)")
    is_power_rail: bool = Field(False, description="是否在电源轨上")
    power_role: str | None = Field(None, description="电源角色 (VCC/GND)")
    net_name: str | None = Field(None, description="所属电气网络名称")


class ComponentWithPins(BaseModel):
    """包含引脚定位信息的元件"""

    component_id: str = Field(..., description="元件ID")
    component_type: str = Field(..., description="元件类型")
    package_type: str = Field(..., description="封装类型")
    polarity: str = Field("none", description="极性")
    confidence: float = Field(1.0, description="置信度")
    pins: list[PinLocation] = Field(default_factory=list, description="引脚定位列表")
    bbox: list[float] | None = Field(None, description="边界框坐标")


class ElectricalNet(BaseModel):
    """电气网络信息"""

    electrical_net_id: str = Field(..., description="网络ID")
    power_role: str = Field("", description="电源角色 (VCC/GND)")
    member_node_ids: list[str] = Field(default_factory=list, description="成员节点ID列表")
    member_hole_ids: list[str] = Field(default_factory=list, description="成员孔洞ID列表")
    connected_components: list[str] = Field(default_factory=list, description="连接到此网络的元件ID列表")


class ComponentConnection(BaseModel):
    """元件连接关系"""
    
    component_id: str = Field(..., description="元件ID")
    connected_to: list[str] = Field(default_factory=list, description="直接连接的元件ID列表")
    is_isolated: bool = Field(False, description="是否为孤立元件")
    connected_net_ids: list[str] = Field(default_factory=list, description="连接的网络ID列表")


# ============================================================
# 前端可视化专用数据结构
# ============================================================

class PortMappingItem(BaseModel):
    """前端端口映射项 - 用于显示元件引脚的二维面包板坐标"""
    
    component_id: str = Field(..., description="元件ID")
    component_type: str = Field(..., description="元件类型")
    pin_id: int = Field(..., description="引脚编号")
    pin_name: str = Field(..., description="引脚名称")
    
    # 二维面包板坐标（核心显示信息）
    row_number: int = Field(..., description="面包板行号 (1-63)")
    col_name: str = Field(..., description="面包板列名 (a-j 或电源轨标识)")
    hole_id: str = Field(..., description="面包板孔洞ID (如 A5, LP10)")
    logic_loc: tuple[str, str] = Field(..., description="逻辑坐标 (行号, 列名)")
    
    # 电源轨标识
    is_power_rail: bool = Field(False, description="是否在电源轨上")
    power_role: str | None = Field(None, description="电源角色 (VCC/GND)")
    
    # 电气连接信息
    electrical_node_id: str | None = Field(None, description="电气节点ID")
    net_id: str | None = Field(None, description="所属电气网络ID")
    net_name: str | None = Field(None, description="所属电气网络名称")
    
    # 可视化坐标（用于绘制）
    x_warp: float | None = Field(None, description="校正后二维图中的X坐标")
    y_warp: float | None = Field(None, description="校正后二维图中的Y坐标")


class NetlistVisualization(BaseModel):
    """网表可视化数据结构 - 专为前端渲染设计"""
    
    job_id: str = Field(..., description="任务ID")
    station_id: str = Field(..., description="工作站ID")
    
    # 端口映射列表（核心数据）
    ports: list[PortMappingItem] = Field(default_factory=list, description="所有元件引脚的端口映射")
    
    # 电气网络信息
    nets: list[dict] = Field(default_factory=list, description="电气网络列表")
    
    # 元件信息
    components: list[dict] = Field(default_factory=list, description="元件列表")
    
    # 电路边界框（用于画布尺寸计算）
    bounding_box: dict[str, float] = Field(default_factory=dict, description="电路边界框")
    
    # 统计信息
    component_count: int = Field(0, description="元件总数")
    pin_count: int = Field(0, description="引脚总数")
    net_count: int = Field(0, description="电气网络总数")
    
    @classmethod
    def from_circuit_analysis(cls, analysis: "CircuitAnalysisResult") -> "NetlistVisualization":
        """从CircuitAnalysisResult构建可视化数据结构"""
        ports = []
        bounding_box = {
            "min_x": float("inf"),
            "max_x": float("-inf"),
            "min_y": float("inf"),
            "max_y": float("-inf"),
        }
        
        # 构建网络名称映射
        net_name_map = {net.electrical_net_id: net.power_role or net.electrical_net_id 
                       for net in analysis.nets}
        
        for comp in analysis.components:
            for pin in comp.pins:
                # 更新边界框
                if pin.x_warp is not None:
                    bounding_box["min_x"] = min(bounding_box["min_x"], pin.x_warp)
                    bounding_box["max_x"] = max(bounding_box["max_x"], pin.x_warp)
                if pin.y_warp is not None:
                    bounding_box["min_y"] = min(bounding_box["min_y"], pin.y_warp)
                    bounding_box["max_y"] = max(bounding_box["max_y"], pin.y_warp)
                
                ports.append(PortMappingItem(
                    component_id=comp.component_id,
                    component_type=comp.component_type,
                    pin_id=pin.pin_id,
                    pin_name=pin.pin_name,
                    row_number=pin.row_number or 0,
                    col_name=pin.col_name or "",
                    hole_id=pin.hole_id,
                    logic_loc=pin.logic_loc or ("0", ""),
                    is_power_rail=pin.is_power_rail,
                    power_role=pin.power_role,
                    electrical_node_id=pin.electrical_node_id,
                    net_id=pin.electrical_net_id,
                    net_name=net_name_map.get(pin.electrical_net_id) if pin.electrical_net_id else None,
                    x_warp=pin.x_warp,
                    y_warp=pin.y_warp,
                ))
        
        # 转换边界框
        if bounding_box["min_x"] == float("inf"):
            bounding_box = {"min_x": 0, "max_x": 800, "min_y": 0, "max_y": 600}
        
        # 构建元件列表（简化版）
        components = []
        for comp in analysis.components:
            components.append({
                "component_id": comp.component_id,
                "component_type": comp.component_type,
                "package_type": comp.package_type,
                "polarity": comp.polarity,
                "confidence": comp.confidence,
                "pin_count": len(comp.pins),
            })
        
        # 构建网络列表（简化版）
        nets = []
        for net in analysis.nets:
            nets.append({
                "net_id": net.electrical_net_id,
                "power_role": net.power_role,
                "component_count": len(net.connected_components),
                "hole_count": len(net.member_hole_ids),
            })
        
        return cls(
            job_id=analysis.job_id,
            station_id=analysis.station_id,
            ports=ports,
            nets=nets,
            components=components,
            bounding_box=bounding_box,
            component_count=analysis.component_count,
            pin_count=len(ports),
            net_count=analysis.net_count,
        )


class CircuitAnalysisResult(BaseModel):
    """电路分析结果 — 包含元件引脚二维定位和网表信息"""

    job_id: str
    station_id: str
    status: JobStatus = JobStatus.COMPLETED
    total_duration_ms: float = 0.0

    # 元件及其引脚的二维定位
    components: list[ComponentWithPins] = Field(default_factory=list, description="元件列表（含引脚定位）")
    component_count: int = 0

    # 网表信息
    nets: list[ElectricalNet] = Field(default_factory=list, description="电气网络列表")
    net_count: int = 0

    # 电路拓扑图 (用于前端可视化)
    topology_graph: dict[str, Any] = Field(default_factory=dict, description="节点链接格式的拓扑图")

    # 电路描述
    circuit_description: str = Field("", description="电路文本描述")

    # 电源轨道分配
    rail_assignments: dict[str, str] = Field(default_factory=dict, description="电源轨道分配")

    # 元件连接关系分析
    component_connections: list[ComponentConnection] = Field(default_factory=list, description="元件连接关系列表")
    isolated_components: list[str] = Field(default_factory=list, description="孤立元件ID列表")

    @classmethod
    def from_pipeline_result(cls, job_id: str, station_id: str, pipeline_result: PipelineResult) -> "CircuitAnalysisResult":
        """从PipelineResult提取电路分析信息"""
        stages = {stage.stage.value: stage.data for stage in pipeline_result.stages}
        topology_data = stages.get("topology", {})

        # 提取元件引脚信息
        components = []
        normalized_components = topology_data.get("normalized_components", [])
        # 建立元件ID到元件的映射
        comp_by_id = {}
        # 建立元件ID到其连接的网络ID映射
        comp_to_nets: dict[str, set[str]] = {}
        
        # 获取网络到电源角色的映射
        net_power_map = {}
        netlist_v2 = topology_data.get("netlist_v2", {})
        for net in netlist_v2.get("nets", []):
            net_power_map[net.get("electrical_net_id", "")] = net.get("power_role", "")
        
        for comp in normalized_components:
            comp_id = comp.get("component_id", "")
            pins = []
            comp_nets = set()
            for pin in comp.get("pins", []):
                net_id = pin.get("electrical_net_id")
                if net_id:
                    comp_nets.add(net_id)
                
                hole_id = pin.get("hole_id", "")
                logic_loc = pin.get("logic_loc")
                
                # 解析面包板坐标
                row_number, col_name, is_power_rail = _parse_hole_id(hole_id, logic_loc)
                power_role = net_power_map.get(net_id) if net_id else None
                
                pins.append(PinLocation(
                    pin_id=pin.get("pin_id", 0),
                    pin_name=pin.get("pin_name", ""),
                    pin_display_name=pin.get("pin_display_name"),
                    polarity_role=pin.get("polarity_role"),
                    polarity_candidate_role=pin.get("polarity_candidate_role"),
                    hole_id=hole_id,
                    logic_loc=logic_loc,
                    electrical_node_id=pin.get("electrical_node_id"),
                    electrical_net_id=net_id,
                    row_number=row_number,
                    col_name=col_name,
                    is_power_rail=is_power_rail,
                    power_role=power_role,
                    net_name=power_role or net_id,
                ))
            component = ComponentWithPins(
                component_id=comp_id,
                component_type=comp.get("component_type", ""),
                package_type=comp.get("package_type", ""),
                polarity=comp.get("polarity", "none"),
                confidence=comp.get("confidence", 1.0),
                pins=pins,
            )
            components.append(component)
            comp_by_id[comp_id] = component
            comp_to_nets[comp_id] = comp_nets

        # 提取网表信息并建立网络到元件的映射
        nets = []
        netlist_v2 = topology_data.get("netlist_v2", {})
        # 建立网络ID到元件ID列表的映射
        net_to_comps: dict[str, list[str]] = {}
        
        for net in netlist_v2.get("nets", []):
            net_id = net.get("electrical_net_id", "")
            # 找出连接到此网络的元件
            connected_comps = []
            for comp_id, comp_nets in comp_to_nets.items():
                if net_id in comp_nets:
                    connected_comps.append(comp_id)
            
            nets.append(ElectricalNet(
                electrical_net_id=net_id,
                power_role=net.get("power_role", ""),
                member_node_ids=net.get("member_node_ids", []),
                member_hole_ids=net.get("member_hole_ids", []),
                connected_components=connected_comps,
            ))
            net_to_comps[net_id] = connected_comps

        # 计算元件连接关系
        component_connections = []
        isolated_components = []
        
        for comp_id, comp_nets in comp_to_nets.items():
            # 找出所有通过网络连接的元件
            connected_to = set()
            for net_id in comp_nets:
                for other_comp_id in net_to_comps.get(net_id, []):
                    if other_comp_id != comp_id:
                        connected_to.add(other_comp_id)
            
            is_isolated = len(connected_to) == 0
            if is_isolated:
                isolated_components.append(comp_id)
            
            component_connections.append(ComponentConnection(
                component_id=comp_id,
                connected_to=sorted(list(connected_to)),
                is_isolated=is_isolated,
                connected_net_ids=sorted(list(comp_nets)),
            ))

        return cls(
            job_id=job_id,
            station_id=station_id,
            status=pipeline_result.status,
            total_duration_ms=pipeline_result.total_duration_ms,
            components=components,
            component_count=len(components),
            nets=nets,
            net_count=len(nets),
            topology_graph=topology_data.get("topology_graph", {}),
            circuit_description=topology_data.get("circuit_description", ""),
            rail_assignments=topology_data.get("rail_assignments", {}),
            component_connections=component_connections,
            isolated_components=isolated_components,
        )
