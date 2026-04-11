"""
新一代网表领域模型。

目标是显式保留:
`component_id + pin_name + hole_id -> electrical_node_id -> electrical_net_id`
这条链路，供后续 validator / RAG / agent 直接复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PinObservation:
    """
    引脚在多视角下的单次视觉观测记录。

    记录引脚（或疑似引脚）在特定视角图像中的二维像素坐标以及检测置信度等元数据。
    主要用于 S1 阶段（检测）到 S2 阶段（映射关联）过程中的数据追溯与调试评估。

    Attributes:
        view_id: 观测到该引脚的相机或视角唯一标识符。
        keypoint: 引脚在图像中的二维像素坐标 (x, y)。如果完全不可见或被遮挡，则为 None。
        visibility: 可见性标识（例如：0=不可见/遮挡，1=部分可见，2=完全可见）。
        confidence: 目标检测或关键点定位的置信度得分 (0.0 ~ 1.0)。
    """
    view_id: str
    keypoint: Optional[tuple[float, float]] = None
    visibility: int = 0
    confidence: float = 0.0


@dataclass
class PinAssignment:
    """
    引脚的物理孔位映射与逻辑电气归属记录。

    该模型记录了单个物理引脚从视觉检测（S1）映射至面板物理孔位（S2），
    最终归属到高层电气网络（S3）的完整生命周期链路。
    
    层级映射关系追踪:
    pin -> hole -> electrical_node -> electrical_net

    Attributes:
        pin_id: 元件内部用于排序或内部索引的物理序号。
        pin_name: 统一的引脚功能或位置命名（如 'A', 'K', '1', '2' 等）。
        hole_id: S2（映射阶段）推断出该引脚插入物理板的孔位唯一标识符（如 'A1'）。
        electrical_node_id: S3（拓扑生成阶段）解析分配的底层电气节点 ID。
        electrical_net_id: S3（拓扑生成阶段）分配的高层连通网络 ID。
        observations: 该引脚在多视角下的视点观测轨迹，用于回溯追溯与异常排查。
        confidence: 综合推断该引脚被正确映射到当前 hole_id 的置信度 (0.0 ~ 1.0)。
        is_ambiguous: 标记该孔位映射是否存在多解歧义，供后续 Validator 告警决策使用。
        metadata: 用于存放大模型规则匹配特征、提示或调试信息的外部数据字典。
    """
    pin_id: int
    pin_name: str
    hole_id: str
    electrical_node_id: Optional[str] = None
    electrical_net_id: Optional[str] = None
    observations: List[PinObservation] = field(default_factory=list)
    confidence: float = 0.0
    is_ambiguous: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentInstance:
    """
    实体电路板上存在的单个元器件数字孪生。

    在物理电路中除自身类别与极性特征外，携带了所有引脚（PinAssignment）
    在空间及电气网表上的连接映射，构成物理到逻辑模型的完整映射核心节点。

    Attributes:
        component_id: 电路系统内该元件的全局唯一标识符（如 'R1_01', 'U1'）。
        component_type: 元件基础大类类别（如 'resistor', 'capacitor', 'IC'）。
        package_type: 物理元器件封装级规格（如 'DIP8', 'AXIAL', 'RADIAL'）。
        part_subtype: 元件的具体子类别型号或电性参数（如 '10k', 'NE555'）。
        polarity: 元件的极性识别状态标识（如 LED 的正负极属性）。默认通常无极性 ('none')。
        orientation: 元件被安插的物理朝向或旋转角度（单位：度，0~360）。
        symmetry_group: 对称或可互换状态的引脚列表集，组内引脚被视作拓扑校验中的可互换逻辑。
        pins: 挂载此组件上所有引脚的连通状态与多视角视觉追溯实例组。
        confidence: 大模型或目标检测中对该组件被正确识别的整体置信度评估 (0.0 ~ 1.0)。
        metadata: 用于扩展补充大环境语境信息的其他数据键值对。
    """
    component_id: str
    component_type: str
    package_type: str
    part_subtype: str = ""
    polarity: str = "none"
    orientation: float = 0.0
    symmetry_group: List[List[str]] = field(default_factory=list)
    pins: List[PinAssignment] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElectricalNet:
    """
    高层次电气网络 (Electrical Net)。

    将电路板上因物理连接（面板内部走线或外部飞线/跳线）导致具有
    等电位特征的一组底层物理孔位（Holes）或电气节点（Nodes）抽象为连通网。

    Attributes:
        electrical_net_id: 整个系统中唯一的高层网络标识（通常在 S3 阶段合成）。
        member_node_ids: 归属于该电气网络内部的所有底层电气节点 ID 集。
        member_hole_ids: 归属于该电气网络体系下物理面板上所有被占用的孔位集。
        power_role: 如果此组网络为电源体系的关键角色（如 'VCC', 'GND'），则在此标注。
        labels: 赋予该网络的其他功能定义标签或系统别名。
    """
    electrical_net_id: str
    member_node_ids: List[str]
    member_hole_ids: List[str]
    power_role: str = ""
    labels: List[str] = field(default_factory=list)


@dataclass
class NetlistV2:
    """
    V2 架构完整网表数据契约模型 (Global Digital Netlist)。

    作为核心检测流水线向验证引擎 (Validator)、检索增强代答 (RAG) 
    及顶层代理 (Agent) 交付的最顶层电气语义表示图（数据黑盒）。
    完整囊括当前场景检测与推断建立的所有元件集合及等电位拓扑网信息。

    Attributes:
        scene_id: 正在执行流水线物理场景的全局唯一识别码。
        board_schema_id: 该网表应用的底层面板基底拓扑结构规范版本标识。
        components: 被识别并解析连通关联的元气件（ComponentInstance）实体集合列表。
        nets: 体系中抽离提取出的全部连通电位网集簇列表 (ElectricalNet)。
        node_index: 维护网络底层 Node ID 映射至其涵盖 Hole IDs 集合的高速检索引擎。
    """
    scene_id: str
    board_schema_id: str
    components: List[ComponentInstance]
    nets: List[ElectricalNet]
    node_index: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
