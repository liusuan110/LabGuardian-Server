# CADx Phase 1 · 拓扑分类标签规范

> **Single source of truth**: `app/domain/topology/labels.py`。本文档为人类可读的对应描述。

## 标签集合（7 类）

| idx | label | 中文显示 | template_id | reference_id |
|---|---|---|---|---|
| 0 | `rc_first_order` | 一阶 RC 滤波器 | `rc_first_order_v1` | `rc_first_order_v1` |
| 1 | `common_emitter` | 共射放大电路 | `common_emitter_v1` | `ce_amp_fixed_bias_v1` |
| 2 | `differential_pair` | BJT 差分放大器 | `differential_pair_v1` | `diff_pair_current_source_ref_split_potentiometer` |
| 3 | `inverting_amp_ua741` | UA741 反相放大器 | `inverting_amp_ua741_v1` | `ua741_inverting_amp_gain10_v1` |
| 4 | `summing_amp_ua741` | UA741 反相加法器 | `summing_amp_ua741_v1` | `ua741_inverting_summing_amp_v1` |
| 5 | `integrator_ua741` | UA741 反相积分器 | `integrator_ua741_v1` | `ua741_integrator_v1` |
| 6 | `unknown` | 未识别拓扑 | (none) | (none) |

**索引契约**：idx 是 GNN-A 模型输出层的 softmax 维度索引。**不可重排**，新增标签**只能追加到末尾**，否则会让旧 ckpt 报错。

---

## 数据集 schema

每个训练样本：

```python
@dataclass
class TopologyDatasetSample:
    sample_id: str                 # 全局唯一 ID
    label: str                     # one of TOPOLOGY_LABELS
    label_index: int               # 0..6
    source: Literal[
        "canonical_reference",     # 6 个 reference DSL 本身
        "perturbation",            # canonical 派生的拓扑保持扰动
        "real_student",            # tests/fixtures/real_student/*.json
    ]
    netlist_v2: dict               # current_netlist_v2 shape
    metadata: dict                 # source-specific details
```

JSON 序列化格式（`dataset/v1/<split>/<label>/<sample_id>.json`）：

```json
{
  "sample_id": "inverting_amp_ua741__perturbed__0042",
  "label": "inverting_amp_ua741",
  "label_index": 3,
  "source": "perturbation",
  "netlist_v2": {
    "scene_id": "synthetic_v1",
    "board_schema_id": "logical_v1",
    "components": [...],
    "nets": [...]
  },
  "metadata": {
    "base_reference_id": "ua741_inverting_amp_gain10_v1",
    "perturbation_chain": ["rename_components", "add_decoration_wire"]
  }
}
```

---

## Train / Val / Test 划分

| Split | 比例 | 来源 | 用途 |
|---|---|---|---|
| `train` | 80% | 大部分 canonical + perturbation 样本 | 训练 |
| `val`   | 10% | canonical 各类至少 1 + 部分 perturbation | 早停 + checkpoint 选优 |
| `test`  | 10% | **全部 real_student fixtures** + 留出 perturbation | 最终评估，绝不参与训练 |

**关键约束**：`real_student` 来源的 7 个 fixture **全部进入 test split**，确保 test 是真实分布而非合成数据，避免乐观估计。

---

## 类别目标样本量

| 类别 | 最少 canonical | 推荐 perturbation | 总目标 |
|---|---|---|---|
| 每个非 unknown 类 | 1（既有 reference DSL） | 100-500 | 500/类 |
| `unknown` | 0（推理期阈值兜底） | 0 | 0 |

**总训练集**：6 类 × 500 = **3000 样本**。这个规模在 GraphSAGE + RTX 4090 上 < 15 分钟训完。

---

## Perturbation 策略（CADx-3）

**拓扑保持**（不改变 canonical 拓扑类别）：

1. `rename_components`：随机重命名 component_id（R1→R5、IC1→U7 等）
2. `rename_nets`：随机重命名 net_id（NET_001→NET_888）
3. `permute_pin_assignments_on_symmetric_components`：被动二端元件 pin1/pin2 交换
4. `add_decoration_wires`：在已存在的同一个 net 上添加冗余 Wire 跨接
5. `add_optional_component`：仅当模板声明 optional 时添加（如反相放大器加 R_p）
6. `remove_optional_component`：移除该拓扑的 optional 元件（如共射移除 C_E）
7. `vary_passive_count`：仅在 multiplicity 范围内变化（如加法器 2-5 输入）

**禁止**（会改变拓扑类别 — 这些是错误样本，不能作为正样本）：
- 改变反馈支路元件类型（R↔C 互换 — 这是 inverting↔integrator 的分界）
- 删除必需元件（如反相放大器删 R_f）
- 改变核心连接（如 IC pin2 改接 VOUT）

---

## Open-set 处理 (`unknown` 类)

**训练阶段**：**不**显式生成 unknown 样本。理由：
- 真实"非 6 类"分布无法穷举
- 显式负样本反而会让模型 overfit 到这些负样本的特征
- 推理时直接用 softmax 最大概率阈值判定

**推理阈值**：
```python
if max(softmax) < 0.4:
    predicted = "unknown"
else:
    predicted = argmax_label
```

阈值 0.4 来自经验，Phase 1 训练完后用 val set 调优。

---

## 与模板系统的协作契约

GNN-A 与符号模板**并行运行**（不互相阻塞）：

```
                        ┌──────────────────────────┐
student netlist  ─────► │  GNN-A topology classifier │ ─► (label, conf)
                        └──────────────────────────┘
                        ┌──────────────────────────┐
                  └───► │  template matcher (6x)   │ ─► top-K templates
                        └──────────────────────────┘
                                    ↓
                         融合层 (Phase 1 中段实现):
                         - 一致 → 高置信单一推荐
                         - 不一致 → 显示双方 + 由用户决定
```

**输出 API 形式**（API endpoint `/api/v1/topology/suggest` —— CADx-6）：

```json
{
  "gnn_predictions": [
    {"label": "integrator_ua741", "confidence": 0.91, ...},
    {"label": "inverting_amp_ua741", "confidence": 0.06, ...}
  ],
  "template_matches": [
    {"template_id": "integrator_ua741_v1", "confidence": 0.94, "variant": "with_leak_resistor"},
    {"template_id": "inverting_amp_ua741_v1", "confidence": 0.88}
  ],
  "consensus": {
    "agreed": true,
    "recommended_reference_id": "ua741_integrator_v1",
    "recommended_template_id": "integrator_ua741_v1"
  }
}
```

---

## 演进策略

| Phase | 新增 | 影响 |
|---|---|---|
| Phase 1 | 7 类 + GraphSAGE | 当前 |
| Phase 2 | 同相放大器、电压跟随器、Schmitt 触发器等 | 追加 idx 7-10，旧 ckpt 不可用，需重训 |
| Phase 3 | LM358 / TL082 等其他 IC 系列 | 同上 |

为减少 Phase 2 重训成本，可考虑：
- **MLflow 标记** ckpt 与标签版本号绑定
- **新加类用 0 类样本 + 拒识** 作为软切换（Phase 2 探索）
