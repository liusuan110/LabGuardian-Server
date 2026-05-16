# LabGuardian-Server · 图神经网络优化电路比较模块技术方案

## Context（背景与目标）

LabGuardian-Server 当前的电路比较器（[app/domain/compare/orchestrator.py](app/domain/compare/orchestrator.py)）是一个**纯规则 / 图论级联**：
`auto_detect_symmetries → full_isomorphism → role_inference fallback → subgraph check → GED fallback → enrich_report`。

它的优势是确定性、可解释、教学反馈友好；劣势是：
1. **GED fallback 在中等规模图（>20 节点）上指数爆炸**，复杂电路容易超时或退化为粗略相似度。
2. **同型器件多时（如 2×R + 2×C）GraphMatcher 候选爆炸**，匹配失败时无法定位"最像哪一个"。
3. **错误定位粒度粗** —— 只能基于规则枚举 missing / extra / wrong_connection，无法预测"学生最可能在哪个 net 上接错"。
4. **进度估计缺失** —— 学生搭一半的电路只会被判为 "fail"，没有"完成度 70%"的反馈。

**目标**：引入一个 GNN **辅助层**（learning-guided graph comparison），在不动现有规则比较器最终判定权的前提下，提供 candidate matching、similarity score、error hotspot、error type 与 progress score，用于 (a) 缩小 GraphMatcher 搜索空间、(b) 替代 GED fallback、(c) 给 validator_report_v2 增加细粒度教学反馈。

**核心原则**：GNN **永远不直接决定 pass/fail**。它只输出"建议 / 概率 / 候选"，最终判定回到规则层。

**MVP 头条任务（聚焦）**：**"哪根引脚接错了？应该接到哪里？"**

整个 GNN 模块的第一性目标，是对学生电路中的每一条 `(port, net)` 连接给出两项预测：
1. `P(edge_correct)` —— 这根线该不该这么接？低于阈值即标记为 **wrong-pin candidate**。
2. **suggested_target** —— 如果接错了，这个 pin **本应接到哪个 net（top-k）**？

其他 head（graph_similarity / error_type / progress）作为**轻量辅助**附在末端，不再做层次化 / motif-level / DAG-async 等重型设计。

**唯一核心参考：GNN-ACLP (arxiv 2504.10240v5, 2024)**

该工作在 SpiceNetlist（775 电路）上用 **port-level 节点 + SEAL enclosing subgraph + DRNL 标签 + DGCNN backbone** 做 analog circuit link prediction，5-fold CV 94.04%、跨数据集 92–99%。其范式与我们"判定每条线该不该这么接"的问题**几乎完全同构**。

| GNN-ACLP 思想 | 本方案如何借鉴 / 落地 |
|---|---|
| Port-level 节点 | 把每个 component pin 升级为独立图节点（component / port / net 三类异构图），让极性与 pin 角色进入节点而非边特征 |
| SEAL enclosing subgraph | 对每条候选 `(port, net)` 边抽 2-hop 邻居子图，作为 link-pred 输入 |
| DRNL 节点标签 | 在子图内对每个节点计算到目标边两端的最小距离对 `(d_u, d_v)`，one-hot 后拼到节点特征 —— 强结构归纳偏置 |
| DGCNN + SortPooling | 子图编码器：3 层 GCN + SortPooling (k=30) + 1-D Conv → P(edge_correct) |
| SpiceNetlist 预训练 | Branch A 先在 SpiceNetlist 上自监督预训练（masked-edge link prediction），再在我们的 3000 合成样本上 fine-tune；解决合成数据稀疏与泛化问题 |
| 跨数据集泛化评估 | test split 强制包含从未见过的 ref 拓扑，复用 GNN-ACLP 的评估口径 |
| **新增（GNN-ACLP 未直接做）**：suggested_target head | 对被判 `wrong` 的 port，枚举 cur 中其他候选 net 作为目标边，SEAL 评分，取 top-k 作为"建议接到这里" |

**不借鉴**：GNN-ACLP 的 Netlist Babel Fish 转换器（我们已有 DSL → netlist_v2 流水线）；**不做** CktGNN 风格的层次化 / motif-level / VAE / 生成式分支 —— MVP 不需要。

---

## 一、总体推荐架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                Reference DSL              Vision Pipeline            │
│                     │                            │                   │
│                     ▼                            ▼                   │
│         logical_reference_v1            netlist_v2                   │
│                     │                            │                   │
│                     └────────────┬───────────────┘                   │
│                                  ▼                                   │
│              app/domain/logical_reference.py                         │
│         (logical_reference_to_graph / current_netlist_v2_to_graph)   │
│                                  │                                   │
│                                  ▼                                   │
│                    NetworkX component-net graph                      │
│                                  │                                   │
│                                  ▼                                   │
│       app/domain/gnn/port_graph.py                                   │
│       (拆出 port 节点，得到 component+port+net 三元异构图)            │
│                                  │                                   │
│       ┌──────────────────────────┼──────────────────────────┐        │
│       ▼                          ▼                          ▼        │
│  Rule Comparator       app/domain/gnn/pyg_converter   (always run    │
│  (orchestrator)        → HeteroData (ref, cur)         in parallel    │
│       │                          │                     for warm cache)│
│       │                          ▼                                   │
│       │       CircuitMatchNet (GNN-ACLP-style SEAL link-pred)        │
│       │       ┌───────────────────────────────────────────┐          │
│       │       │ L1: HeteroConv on comp+port+net           │          │
│       │       │     → shared port / net embeddings        │          │
│       │       │                                           │          │
│       │       │ L2: 对 cur 中每条 (port,net) 边，         │          │
│       │       │     抽 2-hop SEAL subgraph + DRNL,        │          │
│       │       │     DGCNN → P(edge_correct)               │          │
│       │       │                                           │          │
│       │       │ L3: 对 P<阈值 的 port，枚举候选 net,       │          │
│       │       │     SEAL 评分 → suggested_target top-k    │          │
│       │       └───────────────────────────────────────────┘          │
│       │              ├─ port / net embeddings (shared)               │
│       │              └─ heads:                                       │
│       │                  · edge_correct_score    per cur edge        │
│       │                  · suggested_target_topk per wrong port      │
│       │                  · hotspot_score         per cur node        │
│       │                  · graph_similarity      scalar (aux)        │
│       │                  · error_type_logits     multi-label (aux)   │
│       │                          │                                   │
│       ▼                          ▼                                   │
│  should_use_gnn()  ──── GNNAdvice (mappings + scores + hotspots)     │
│       │                          │                                   │
│       └────────────► Orchestrator merge layer ◄──────────────────────┤
│                                  │                                   │
│             ┌────────────────────┼────────────────────┐              │
│             │                    │                    │              │
│   1) seed GraphMatcher    2) replace GED       3) enrich report      │
│      with top-k mapping      fallback when         (gnn block)       │
│                              规则失败                                │
│                                  │                                   │
│                                  ▼                                   │
│                       validator_report_v2 (+ gnn 字段)               │
└──────────────────────────────────────────────────────────────────────┘
```

**职责边界**：

| 阶段 | Rule | GNN | 谁说了算 |
|---|---|---|---|
| 全图同构判定 | ✓ | — | Rule |
| 候选节点排序 / 缩 GraphMatcher 搜索空间 | — | ✓ | GNN 给候选，Rule 验证 |
| 错误枚举（missing / extra / wrong） | ✓ | — | Rule |
| 错误"热点"高亮 / 教学提示 | — | ✓ | GNN |
| Progress score | — | ✓ | GNN |
| GED fallback（替换） | 备份 | ✓ | GNN（带 confidence） |
| Pass/Fail | ✓ | — | **永远 Rule** |
| 短路 / 极性 / 安全规则 | ✓ | — | **永远 Rule** |

**触发时机**：GNN 不在每次都跑。`should_use_gnn(compare_context)` 返回 True 时才推理（详见第七节）。推理失败 / 超时 → 静默 fallback 到原 GED 路径，**绝不阻塞主比较器**。

---

## 二、关键模块职责表

| 新增文件 | 职责 | 依赖 |
|---|---|---|
| `app/domain/gnn/__init__.py` | 暴露 `GNNAdvisor`、`should_use_gnn` | — |
| `app/domain/gnn/graph_schema.py` | 异构图节点 / 边 / 特征常量；类型枚举（ComponentType, NetRole, PinRole） | — |
| `app/domain/gnn/graph_builder.py` | 把 NetworkX 图（已由 logical_reference.py 产出）规范化为统一的中间结构 `HeteroCircuitGraph`（dict-of-tensors-friendly） | networkx |
| `app/domain/gnn/port_graph.py` | **GNN-ACLP port-level 表示**。把 component-net bipartite 拆成 (component, port, net) 三元节点图；port 节点继承 component_type + port_type (Drain/Gate/Source/Anode/Cathode/Pin1/Pin2/…) | networkx |
| `app/domain/gnn/seal_subgraph.py` | **GNN-ACLP SEAL + DRNL**。对任意目标边 `(port_u, net_v)` 抽 h-hop enclosing subgraph、计算 DRNL 标签，返回 PyG Data；同时支持 batched 抽取（一次性给出 cur 中所有边的子图） | networkx, torch_geometric |
| `app/domain/gnn/pyg_converter.py` | `to_pyg_data(hetero_circuit_graph) → HeteroData`；处理 categorical encoding、padding、reverse edge | torch_geometric |
| `app/domain/gnn/perturbation.py` | 对参考图施加 12 类扰动，**重点保证每条扰动都能精确标注 wrong-edge 与对应的 correct-edge（用于 suggested_target 监督）** | networkx |
| `app/domain/gnn/dataset_builder.py` | 遍历 `tests/fixtures/references/` + DSL，调用 perturbation，输出 `datasets/processed/pyg/*.pt` 和 `labels/*.json` | torch |
| `app/domain/gnn/model.py` | `CircuitMatchNet` 模型（共享 HeteroConv backbone + SEAL DGCNN head + 辅助 heads） | torch_geometric |
| `app/domain/gnn/train.py` | 训练循环、多任务 loss、ckpt 管理 | torch |
| `app/domain/gnn/inference.py` | `GNNAdvisor.advise(ref_graph, cur_graph) → GNNAdvice`；模型加载、超时、CPU/GPU 选择、异常吞掉 | torch |
| `app/domain/gnn/evaluator.py` | 离线评估（top-k acc, hotspot acc, false_pass_rate） | torch |
| `app/domain/compare/orchestrator.py`（**改**） | 在 `compare_logical_graphs` 中插入 GNN 调用钩子：(1) 在 `_find_isomorphism` 前把 candidate seed 喂给 GraphMatcher；(2) 在 GED fallback 处用 GNN similarity 替换；(3) 在 `_enrich_result` 中追加 `gnn` 字段 | gnn |
| `app/domain/compare/diff_report.py`（**改**） | `enrich` 接受 `gnn_advice` 参数，写入 report 的 `summary.gnn` 与 `items[*].gnn_hint` | — |

---

## 三、图 Schema 表（HeteroData）

**重要范式切换（GNN-ACLP-inspired）**：从 "component + net 二分图" 升级为 **"component + port + net 三元异构图"**。port 是真正承担电气语义的节点，component 仅是 port 的"宿主"容器、net 是 port 的"连接介质"。这一改动让我们可以：
- 在 port-level 抽 SEAL 子图（论文证明这是关键）；
- 让极性 / pin_role 进入节点而不是边特征，给消息传递更强信号；
- 直接对应面包板上的"哪根线插到哪个孔"。

### 节点

| 节点类型 | 特征字段 | 维度 | 编码方式 |
|---|---|---|---|
| `component` | `ctype`（Resistor/Capacitor/LED/Diode/BJT/IC/Wire/...） | 16 | one-hot |
|  | `package`（DIP8/DIP14/SMD/THT） | 6 | one-hot |
|  | `polarity_class`（none/two-polar/multi-asymmetric） | 3 | one-hot |
|  | `pin_count` | 1 | scalar (log) |
|  | `value_log10` (+ mask) | 2 | scalar+mask |
|  | `confidence` | 1 | scalar |
|  | `is_reference` | 1 | binary |
|  | **总计** | **30** | |
| `port` 🆕 | `port_type` —— P0.5 扩张到 23 类（含 op-amp INVERTING_INPUT / NON_INVERTING_INPUT / OUTPUT / OFFSET_NULL / NC / V_PLUS / V_MINUS） | **23** | one-hot |
|  | `parent_ctype`（16 类 ComponentType） | 16 | one-hot |
|  | `polarity_sensitive` | 1 | binary（该 port 是否极性关键，如 BJT base） |
|  | `is_power_port` / `is_ground_port` | 2 | binary |
|  | `is_floating`（P0.6 后真正生效：cur 侧未接任何 net 时为 True） | 1 | binary |
|  | `is_reference` | 1 | binary |
|  | `connection_policy` (REQUIRED / OPTIONAL / FORBIDDEN) —— **P0.6 新增** | **3** | one-hot |
|  | `has_pin_number` —— **P0.6 新增** | 1 | binary |
|  | `pin_number_log` ≈ log(1+n)/log(65) clip 0~1 —— **P0.6 新增** | 1 | scalar |
|  | `symmetry_class_size_inverse` = 1/k where k = 同 component 同互换类的 port 数 —— **P0.6 新增** | 1 | scalar |
|  | **总计** | **50** | |
|  | （演进史：P0 = 37 → P0.5 扩 PortType → 44 → P0.6 加 policy/pin_number/symmetry → **50**；运行时 self-check 守住一致性） | | |
| `net` | `role`（input/output/vcc/gnd/internal/unknown） | 6 | one-hot |
|  | `degree` | 1 | scalar (log) |
|  | `is_power_rail` | 1 | binary |
|  | `voltage_hint` (+ mask) | 2 | scalar+mask |
|  | `is_reference` | 1 | binary |
|  | **总计** | **11** | |

### 边

| 边类型 | 特征字段 | 维度 |
|---|---|---|
| `(component, has_port, port)` | 结构边，无属性 | 0 |
| `(port, connects, net)` 🔑 | `connection_confidence`（视觉侧 0~1） | 1 |
|  | `source_type`（dsl / vision / inferred） | 3 |
|  | `is_observed_in_cur`（cur 侧 1，ref 侧 1，预测候选边为 0） | 1 |
|  | **总计** | **5** |
| 所有反向边 | PyG `T.ToUndirected()` 自动添加 | — |

**关键差别 vs 旧 schema**：pin_index / pin_role 不再是 edge 特征，而是直接进 **port 节点**特征（**50 维**，含 P0.5 + P0.6 累积）—— 这是 GNN-ACLP 的核心 insight：port 是一等公民。

### 设计决策（已修订）

- **MVP 改为 component + port + net 三类节点**（推翻原"只 component + net"决策）。理由：(a) GNN-ACLP 实证 port-level 在 link prediction 上 94–99% acc，远高于 component-level；(b) 让 SEAL 抽子图时距离标签 DRNL 计算在 port-level 才有意义；(c) 节点数翻倍但 5 类 MVP 电路仍 < 100 节点，PyG 完全可承受。
- **无极性两脚器件（R, C 陶瓷）** —— 两个 port 都标 `port_type=Pin_symmetric`，训练时随机 swap port_id 作为正样本。
- **极性器件（LED, Diode, 电解 Cap, BJT, IC）** —— port_type 按器件定义严格区分（Anode/Cathode, Base/Collector/Emitter, Pin1..PinN）；`polarity_sensitive=1`。
- **VCC / GND / input / output 角色**：仍在 net 节点的 `role` one-hot；port 侧若 parent_ctype = VoltageSource，则 `is_power_port` 强信号。

### 三.6 SEAL Enclosing Subgraphs（GNN-ACLP 核心借鉴）

对**每条候选边** `e=(port_u, net_v)` 构造一个独立 PyG Data：

1. **抽 2-hop enclosing subgraph** `G_e`：包含 `port_u`、`net_v` 及 2-hop 邻居（沿 port↔net 边交替走）。
2. **DRNL 节点标签**：对 `G_e` 内每个节点 `w`，计算 `d_u(w) = dist(w, port_u)`、`d_v(w) = dist(w, net_v)`，按论文公式 `l(w) = 1 + min(d_u, d_v) + (d/2) * ((d/2) + (d%2) - 1)`（其中 `d=d_u+d_v`），结果做 one-hot（labels 0..15 + overflow bucket）= 17 维。
3. **拼接特征**：节点最终特征 = DRNL[17] ⊕ 原 port/net 特征（**50** 或 11，按节点类型；port 维度经 P0.5/P0.6 扩张）。
4. **目标边方向**：标记 `e` 的两端点为 `target=True`（额外 1 维），其余为 `target=False`。

**两类候选边**（决定 link-pred head 的两个任务）：
- **A. wrong_connection 检测**：枚举 cur 中**实际存在**的所有 `(port, net)` 边 → 让 SEAL 预测 P(该边应当存在)。低于阈值 → 标记为 `wrong_connection` 热点。
- **B. missing_connection 检测**：枚举 **ref 中存在但 cur 中可能缺失**的边（先用 GNN 匹配头给出 ref↔cur 节点映射，再把 ref 边映射到 cur 节点对查询是否存在）→ SEAL 预测 P(该边应当存在)。高于阈值且 cur 中缺失 → `missing_connection` 候选。

**复杂度控制**：每个电路 cur 边数 ≤ 50，每条 2-hop 子图 ≤ 30 节点，单次推理 < 30 ms（CPU）。

---

## 四、模型设计（CircuitMatchNet · MVP）

**单分支聚焦架构**：所有任务围绕 SEAL link-prediction 展开。共享 backbone 编码 port / net，主头判定每条线对错，次头建议正确目标，辅助头给图级反馈。

```
                       HeteroData (cur)           HeteroData (ref)
                            │                         │
                            ▼                         ▼
─────────────── L1 · Shared Backbone (comp + port + net) ─────────────
NodeEncoder.comp  [Nc, 30] → [Nc, 128]
NodeEncoder.port  [Np, 50] → [Np, 128]   # P0.5/P0.6 累积后 50 维（见 §三）
NodeEncoder.net   [Nn, 11] → [Nn, 128]
HeteroConv (SAGE) × 3   with edge_attr MLP, residual + LayerNorm
   edges traversed: comp↔port (structural), port↔net (electrical)
   → z_comp, z_port, z_net   （ref / cur 共享权重，分别前传）

─────────────── L2 · SEAL Head (主任务 · 哪根线接错了) ─────────────────
对 cur 中每条 (port_u, net_v) 边 e：
   1. 抽 2-hop enclosing subgraph  G_e   (≤30 nodes)
   2. 每个节点特征 = DRNL_label[17] ⊕ z_(port|net|comp)[i, 128] ⊕ target_flag[1]
   3. 3-layer DGCNN (hidden=64)
        ↓ SortPooling (k=30)
        ↓ 1-D Conv (output_channels=32 → 1)
        ↓ Sigmoid
   ⇒ P(edge_correct) ∈ [0, 1]

   聚合：hotspot_score[port_u] = 1 − min over edges touching port_u

─────────────── L3 · Suggested-Target Head (次任务 · 应该接到哪) ──────
对每个 P(edge_correct) < τ_wrong 的 port_u：
   1. 候选 net 集合 N_cand = cur 中所有 net  ∪  {<new_net>}   (≤ 20)
   2. 对每个 v ∈ N_cand 抽 SEAL(port_u, v)，复用同一 DGCNN backbone 评分
   3. 取 top-k (k=3) 作为 suggested_target，附带 confidence

─────────────── L4 · 辅助 Heads (轻量) ─────────────────────────────────
  · graph_similarity     mean-pool(z_port_ref) ⊕ mean-pool(z_port_cur) ⊕ |diff|
                         → 2-layer MLP → scalar∈[0,1]
  · error_type_logits    mean-pool(z_*_cur) → MLP → 12-d multi-label
  · progress_score       直接由 graph_similarity 与 #correct_edges / #ref_edges 估计
                         （MVP 不单独训练 head，按规则公式聚合 SEAL 输出）
```

### 各层 I/O

| 层 | 输入 | 输出 |
|---|---|---|
| L1 NodeEncoder.comp / port / net | [Nc,30] / [Np,**50**] / [Nn,11] | 三者全部 → 128 |
| L1 HeteroConv (×3) | (z, edge_index per type, edge_attr[5]) | [N, 128] + residual + LN |
| **L2 SEAL DGCNN** （主头） | per-edge enclosing subgraph (DRNL[17] ⊕ z[128] ⊕ flag[1]) | scalar P(edge_correct) |
| **L3 Suggested-Target** （次头） | 对每个 wrong port 枚举候选 net 子图 | top-k (net, score) |
| Hotspot aggregator | edge probs grouped by port / net | per-node hotspot [0,1] |
| Graph-similarity head | mean-pool(z_port_ref \|\| z_port_cur \|\| \|diff\|) | scalar [0,1] |
| Error-type head | mean-pool(z_*_cur) | 12-d multi-label |
| Progress | 公式：`α·sim + (1-α)·mean(edge_correct over ref-mapped edges)` | scalar [0,1] |

### Loss（多任务，聚焦头条任务）

```
L =  λ_seal · BCE(edge_correct, gt_edge_label)         # 主头
  +  λ_tgt  · CE(suggested_target, gt_correct_partner) # 次头（GNN-ACLP 范式延伸）
  +  λ_hot  · BCE(hotspot, gt_hotspot_mask)            # 由 edge 标签聚合
  +  λ_sim  · BCE(graph_sim, gt_equivalent)            # 辅助
  +  λ_err  · BCE(error_type, gt_error_type_multilabel)# 辅助
```

MVP 默认权重：`(seal=1.5, tgt=1.0, hot=0.5, sim=0.4, err=0.3)`。**SEAL 主头权重最高**，因为它直接服务"哪根引脚接错"，且有 SpiceNetlist 预训练打底，学得最快、最稳。次头共享 backbone，几乎零额外成本但能直接产出教学建议。

---

## 五、数据集 Schema 与构建（LabGuardian-CircuitCompare）

### 目录

```
datasets/
├── pretrain_spicenetlist/                ← 🆕 GNN-ACLP 公开数据集（775 标注电路）
│   ├── raw/                              ← 从 GNN-ACLP 项目下载 SPICE + JSON
│   ├── processed/seal_pairs/             ← (port_u, net_v, label) tuples + 2-hop subgraph
│   └── splits/                           ← 70/10/20 + 5-fold
└── circuit_compare/                      ← LabGuardian 自有合成数据集
    ├── raw/
    │   ├── references/                ← tests/fixtures/references/*.json + 5 个手写 DSL
    │   │   ├── voltage_divider.json
    │   │   ├── led_resistor.json
    │   │   ├── rc_lowpass.json
    │   │   ├── transistor_switch.json
    │   │   └── opamp_buffer.json
    │   └── generated_samples/<ref_id>/<sample_id>.json  ← NetworkX node-link
    ├── processed/
    │   ├── pyg/<ref_id>/<sample_id>.pt        ← HeteroData (comp+port+net)
    │   ├── seal_edges/<ref_id>/<sample_id>.pt ← list of SEAL subgraphs + edge labels
    │   └── labels/<ref_id>/<sample_id>.json
    └── splits/{train,val,test}.json
```

**预训练 → fine-tune 两阶段**：
- 阶段 1：在 `pretrain_spicenetlist/` 上自监督训练 SEAL 主头（随机 mask 一条边，预测其存在性），5-fold CV，目标 AUC ≥ 0.95（对齐 GNN-ACLP 论文 94–99% acc）。
- 阶段 2：载入 backbone 与 SEAL head 权重，在 `circuit_compare/` 上端到端 fine-tune 所有 heads（SEAL 主头 + suggested-target + hotspot + 辅助头）。

### 样本 label schema

```json
{
  "sample_id": "rc_lowpass__neg_pinrev_0042",
  "ref_id": "rc_lowpass",
  "is_equivalent": false,
  "perturbation_chain": ["pin_reversed:C1"],
  "component_mapping": {"R1": "U_R_3", "C1": "U_C_1"},
  "port_mapping": {                            // 🆕 GNN-ACLP port-level GT
    "R1.p1": "U_R_3.p1", "R1.p2": "U_R_3.p2",
    "C1.anode": "U_C_1.cathode",               // 极性反 → port 错配
    "C1.cathode": "U_C_1.anode"
  },
  "net_mapping": {"VIN": "n_07", "GND": "n_03", "VOUT": "n_05"},
  "edge_labels": [                             // SEAL 主头 GT
    {"src": "U_R_3.p1",       "dst": "n_07", "is_correct": 1},
    {"src": "U_C_1.cathode",  "dst": "n_05", "is_correct": 0, "error": "pin_reversed",
     "suggested_target": "n_03"},              // 次头 GT：本应接到 GND
    {"src": "U_C_1.anode",    "dst": "n_03", "is_correct": 0, "error": "pin_reversed",
     "suggested_target": "n_05"}
  ],
  "hotspot_nodes": ["cur_port:U_C_1.cathode", "cur_net:n_05"],
  "error_type": ["pin_reversed"],
  "progress_score": 0.85,
  "rule_report_baseline": { ... }
}
```

### 正样本生成（保拓扑等价的扰动）

| 扰动 | 实现 |
|---|---|
| 完全正确 + 视觉噪声 | 不动结构，加 `connection_confidence` 噪声 0.85~1.0 |
| 对称网络互换 | 调用现有 `auto_detect_symmetries`，按 orbit 重命名 |
| 无极性 pin swap | Resistor / 陶瓷 Cap 的 pin1 ↔ pin2 |
| net alias 变化 | 重命名 `VOUT → out1`，role 保持 |
| component id 改名 | `R1 → R_x`、`U_R_3` 等 |
| 内部 signal 名不同 | 拓扑不变，net id 全部 hash 化 |

### 负样本生成（12 类错误，对齐 error_type head）

| 扰动 | 实现要点 |
|---|---|
| missing_component | 删 1 个非关键 component + 它的所有边 |
| extra_component | 加一个孤立 / 旁挂 component |
| wrong_connection | 把一条 edge 的 net 端点替换为另一个随机 net |
| pin_reversed | 极性器件交换 pin_role（LED anode↔cathode） |
| power_swapped | VCC net ↔ GND net 整体交换 |
| input_output_swapped | input role net ↔ output role net |
| short_circuit | 把两个不同 net 合并为一 |
| floating_net | 删除某 net 上除一个 pin 外的所有边 |
| missing_resistor | 专门删除分压 / 限流电阻（语义关键） |
| wrong_resistor_position | 把 R1 的位置和 R2 互换（值不同时） |
| extra_wire_bridge | 在两个原本无连接的 net 间插入 0Ω wire |
| chained | 两到三种扰动叠加（提高难度） |

### Ground-truth mapping 生成

由于扰动**从已知的 ref 拷贝并改名**，`component_mapping` / `net_mapping` 在 perturbation 阶段直接保存（不需要后期对齐）。`hotspot_nodes` 即被扰动直接触碰的节点集合。

### 防泄漏与划分

- **train / val 共享 ref**，但样本（perturbation seed）不交叉。
- **test 用 held-out ref 电路**（取 5 个 MVP 中的 1 个完全留出 + 1 个全新拓扑），强制评估**对新拓扑的泛化**。
- **rule_report_baseline**：每个样本同步保存当前规则比较器的输出，便于评估"GNN 在哪些样本上比规则更优"。
- **目标规模**：每个 ref × 600 样本（300 正 / 300 负，按 12 类均匀） = 5 ref × 600 = **3000 样本** MVP，可在 1 张 RTX 3060 上 30 分钟内训练。

---

## 六、系统接入方案（Orchestrator 集成）

### `compare_logical_graphs` 改动伪码

```python
def compare_logical_graphs(reference_graph, current_graph, ref_payload, cur_netlist_v2):
    ctx = build_compare_context(reference_graph, current_graph)
    advice = None
    if should_use_gnn(ctx):
        try:
            advice = GNNAdvisor.get().advise(reference_graph, current_graph,
                                             timeout_ms=300)
        except Exception as e:
            logger.warning("gnn_advisor_failed", exc_info=e)  # 静默 fallback

    # ---- 既有规则流程 ----
    sym = auto_detect_symmetries(...)
    iso = _find_isomorphism(
        reference_graph, current_graph,
        seed_node_mapping=advice.top1_component_mapping if advice else None,
    )  # 把 GNN top-1 当作 GraphMatcher 的初始化候选
    ...
    if reached_ged_fallback:
        if advice and advice.graph_similarity_confidence > 0.85:
            similarity = advice.graph_similarity            # 替换 GED
            match_type = "gnn_assisted_similarity"
        else:
            similarity = _ged_similarity(...)               # 原 fallback

    report = _enrich_result(..., gnn_advice=advice)
    return report
```

### 冲突仲裁

| 情形 | 处理 |
|---|---|
| Rule 判 pass，GNN 判 fail | **以 Rule 为准**，但在 report.summary.gnn 加 `disagreement: true` 警告，severity=warning |
| Rule 判 fail，GNN 判 pass | **以 Rule 为准**（永远不让 GNN "救"错电路 → 守住 false_pass） |
| Rule 判 fail 且 fallback 到 GED，GNN confidence > 0.85 | 用 GNN similarity，但 match_type 标 `gnn_assisted` |
| GNN 推理超时 / 异常 | 静默忽略，走原路径 |

### Confidence 阈值（初始默认，按评估调）

- `MIN_MAPPING_CONFIDENCE = 0.7` —— 低于不喂给 GraphMatcher。
- `MIN_SIMILARITY_CONFIDENCE = 0.85` —— 低于不替换 GED。
- `MIN_HOTSPOT_CONFIDENCE = 0.6` —— 低于不在 report 中高亮。

### validator_report_v2 增强字段

```json
{
  "version": "validator_report_v2",
  "summary": {
    "logic_correct": true,
    "match_type": "gnn_assisted_similarity",
    "similarity": 0.92,
    "gnn": {
      "enabled": true,
      "model_version": "circuit_match_v0.1",
      "graph_similarity": 0.923,
      "graph_similarity_confidence": 0.88,
      "progress_score": 0.78,
      "predicted_error_types": [
        {"label": "pin_reversed", "score": 0.74},
        {"label": "wrong_connection", "score": 0.21}
      ],
      "component_mapping_topk": {
        "R1": [["U_R_3", 0.94], ["U_R_1", 0.05]],
        "C1": [["U_C_1", 0.91]]
      },
      "net_mapping_topk": { "VOUT": [["n_05", 0.89]] },
      "hotspots": [
        {"node": "cur_port:U_C_1.cathode", "score": 0.81, "hint": "Possible polarity reversal"}
      ],
      "edge_predictions": [
        {
          "edge": ["U_C_1.cathode", "n_05"],
          "p_correct": 0.18,
          "verdict": "likely_wrong",
          "suggested_target": [
            {"net": "n_03", "score": 0.84},
            {"net": "n_07", "score": 0.09}
          ],
          "hint": "U_C_1.cathode 当前接在 VOUT，模型建议接到 GND（极性可能反了）"
        }
      ],
      "disagreement_with_rule": false,
      "inference_ms": 47
    }
  },
  "items": [
    {
      "error_code": "PIN_REVERSED",
      ...,
      "gnn_hint": {
        "node": "cur_comp:U_C_1",
        "hotspot_score": 0.81,
        "alt_candidates": ["U_C_2"]
      }
    }
  ]
}
```

---

## 七、`should_use_gnn(compare_context)` 触发逻辑

```python
def should_use_gnn(ctx: CompareContext) -> bool:
    # 不用 GNN 的情况（早退）
    if ctx.node_count_total < 8:                       # 极简电路
        return False
    if ctx.has_safety_critical_check_pending:          # VCC/GND 短路等
        return False
    if ctx.deterministic_polarity_violation:           # 极性已被 rule 锁定
        return False

    # 触发 GNN 的情况
    triggers = [
        ctx.full_isomorphism_failed,
        ctx.match_type_so_far in {
            "current_subgraph_in_reference",
            "equivalent_with_extra",
            "graph_edit_distance_or_fallback",
        },
        ctx.has_repeated_component_types(min_count=2),     # 同型多于 2 个
        ctx.min_visual_confidence < 0.7,                   # 视觉低置信
        ctx.requested_features & {"hotspot", "progress"},  # 上层显式要求
        ctx.has_symmetric_or_repeated_substructure,
    ]
    return any(triggers)
```

---

## 八、评估指标体系

| 指标 | 说明 | MVP 目标 |
|---|---|---|
| ⭐ **SEAL edge prediction AUC** (SpiceNetlist 预训练) | 主头在公开基准对齐 | ≥ 0.95（对齐论文 94–99%） |
| ⭐ **SEAL edge prediction AUC** (LabGuardian fine-tune) | 主头在合成数据 fine-tune 后 | ≥ 0.92 |
| ⭐ **wrong_pin detect F1** | 主头识别 cur 中接错的 (port, net) 边 | ≥ 0.88 |
| ⭐ **suggested_target Top-1 acc** | 次头：对错 pin 给出的最优 net 命中率 | ≥ 0.70 |
| ⭐ **suggested_target Top-3 acc** | 次头：Top-3 内命中 | ≥ 0.85 |
| missing_connection detect F1 | 用 ref→cur net 映射枚举缺失边并 SEAL 评分 | ≥ 0.80 |
| Graph equivalence AUC | 二分类 ROC-AUC | ≥ 0.92 |
| Graph equivalence F1 | @最佳阈值 | ≥ 0.88 |
| Error type macro-F1 | 12 类多标签 macro-F1 | ≥ 0.65 |
| Error localization acc | 预测 hotspot ∩ gt_hotspot ≠ ∅ 的比例 | ≥ 0.75 |
| Progress score MAE | vs 合成的 progress label | ≤ 0.12 |
| **False pass rate ⚠️** | **GNN+Rule 联判错电路为对的比例** | **≤ 0.5%（最关键安全指标）** |
| False fail rate | 联判对电路为错的比例 | ≤ 5% |
| GraphMatcher runtime reduction | 用 GNN seed 后平均耗时下降 | ≥ 50% |
| End-to-end report quality | 人工抽样 100 条评分（1-5） | ≥ 4.0 |

**False pass rate 是不可妥协的红线**。评估时分别报：(a) 仅规则；(b) 规则+GNN；后者必须 ≤ 前者。这是引入 GNN 的硬约束。

---

## 九、MVP 里程碑（分阶段计划）

| Phase | 时长 | 交付物 | 关键文件 | 验收 |
|---|---|---|---|---|
| **P0 · Schema** | 3 天 | graph_schema.py（含 port 节点）+ HeteroCircuitGraph + 添加 torch / torch_geometric 到 pyproject extras `[gnn]` | `app/domain/gnn/graph_schema.py`、`graph_builder.py`、`port_graph.py` | unit test：现有 fixture 全部能转 component+port+net schema |
| **P0.5 · IC + Pot Port 语义** | 1.5 天 | PortType 16→23（含 op-amp 角色）、IC_PIN_MAPS、parallel-pin bypass | `graph_schema.py`、`port_graph.py`、`test_opamp_buffer_v1.json` | UA741 buffer 5 个连接 pin 全部拿到精细 PortType |
| **P0.6 · Package Port Materialization** | 2 天 | ConnectionPolicy / PinSpec / 全 pin materialize + 自动 symmetry 组 | `graph_schema.py`、`hetero_circuit.py`、`port_graph.py` | UA741 fixture 8 port 全 materialize；is_floating + connection_policy 正确；R/Pot symmetry 组成立 |
| **P0.7 · SEAL Subgraph Pipeline**（GNN-ACLP-inspired） | 4 天 | `seal_subgraph.py` 实现 2-hop enclosing subgraph 抽取 + DRNL labeling + batched 接口 | `seal_subgraph.py` | unit test：手算 DRNL 标签与论文公式一致；性能：50 条边 < 30 ms |
| **P0.8 · Label Builder + Alignment** | 3 天 | `alignment.py` + `label_builder.py`：ComponentAlignment / SealSample（含 task_type / candidate_edge / expected_edge / group_id）/ SealSampleGroup（query_origin: floating \| wrong_redirect）/ **8 类 LabelSource**（**含 WRONG_OBSERVED 强负 + HARD 占位**）/ TaskType / LabelStats / LabelBuildResult / **6 步算法**（WRONG_EDGE 点式 + MISSING_EDGE group [双触发: floating + wrong_redirect] + sym swap + **WRONG_OBSERVED 100% 覆盖** + FORBIDDEN 强负）+ JSON schema serialize/deserialize | `alignment.py`、`label_builder.py` | UA741 buffer 全套 LabelBuildResult < 80 ms；OPTIONAL 默认排除；R.pin1↔pin2 swap 双正样本；FORBIDDEN 默认 4 条合成负；MISSING_EDGE group 覆盖 floating + wrong_redirect；**cur 中每条非 OPTIONAL 非 ref-correct 边必有 WRONG_EDGE 负样本**；LabelStats.by_source/by_task_type 一致；serialize round-trip 等价 |
| **P1 · Synthetic Dataset** | 5 天 | 5 ref × 600 样本 = 3000 样本（PyG + seal_edges），label 含 `suggested_target` 监督 | `perturbation.py`、`dataset_builder.py` | label 抽样 50 条人工核对；edge_labels / suggested_target 自动校验 |
| **P2 · PyG Converter** | 3 天 | NetworkX → HeteroData + SEALEdgeBatch | `pyg_converter.py` | 往返一致性测试 |
| **P2.5 · SpiceNetlist 预训练**（GNN-ACLP 范式） | 5 天 | 下载 SpiceNetlist 775 电路 + 转 port-level + 自监督训练 SEAL 主头，5-fold CV | `pretrain_seal.py` | edge AUC ≥ 0.95；对齐 GNN-ACLP 论文报告 |
| **P3 · Train Full Model** | 7 天 | 载入 P2.5 backbone，端到端 fine-tune SEAL 主头 + suggested-target + 辅助头；含 ablation：(去预训练) / (去 DRNL) / (去 port 节点) | `model.py`、`train.py` | val: SEAL F1 ≥ 0.88，suggested-target top-3 ≥ 0.85；ablation 显示预训练 ≥+5%、DRNL ≥+3% |
| **P4 · Inference Integration** | 5 天 | GNNAdvisor + orchestrator 钩子 + `should_use_gnn` + report `gnn` 字段 | `inference.py`、改 `orchestrator.py` / `diff_report.py` | 全部既有 35+ test_graph_compare 用例不回归；新增 10 条 fallback 替换用例 |
| **P5 · Evaluate & Ablation** | 5 天 | 离线 evaluator + ablation table + 风险报告 | `evaluator.py` | false_pass ≤ 0.5%，GraphMatcher runtime −50% |

**MVP 不做**：层次化 motif-level DAG / DAG-async / motif matching head、hole node、IC 内部结构建模、跨 batch 训练真实视觉数据、GNN 直接判 pass/fail。

---

## 一句话技术路线（写入开题报告）

> 本工作以 **GNN-ACLP (Dong et al., 2024)** 提出的 *port-level 节点 + SEAL enclosing subgraph + DRNL labeling + DGCNN* 链路预测范式为核心，在 LabGuardian-Server 的规则比较器之外构建一个 learning-guided 辅助层：将面包板电路建模为 component / port / net 三元异构图，对学生电路中的每一条 `(port, net)` 连接抽取 2-hop 子图、计算 DRNL 标签后由 DGCNN 评分 `P(edge_correct)`，主头直接回答"**哪根引脚接错了**"，次头对低分 port 枚举候选 net 再次评分，回答"**它本应接到哪里**"；模型先在 SpiceNetlist 775 电路上自监督预训练（对齐论文 94–99% AUC），再在 LabGuardian 合成的 3000 扰动样本上 fine-tune，最终通过 GNNAdvisor 把 wrong-pin 列表与 suggested_target 注入 validator_report_v2，在守住零 false-pass 红线的前提下，为教学场景提供从"哪儿错了"到"应该怎么接"的细粒度反馈。

---

## 十、风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 数据不足（5 个电路 × 3000 样本仍稀疏） | 泛化差 | (a) **SpiceNetlist 775 电路预训练 Branch A**（GNN-ACLP 范式，已纳入 P2.5）；(b) chained perturbation 增强；(c) 节点 / 边随机 mask 自监督；(d) 后续真实学生 netlist 日志 fine-tune |
| 合成 vs 真实视觉分布差异 | 推理时退化 | (a) 在 edge_attr 中用 `connection_confidence` 注入"视觉噪声"通道训练；(b) Phase P5 后采集真实数据做 domain adaptation；(c) `source_type` edge feature 区分 dsl/vision |
| GNN 黑盒不可解释 | 教学反馈失效 | (a) 所有 GNN 输出必须落在 hotspot/error_type/mapping 三类**结构化字段**，禁止自由文本；(b) report 中始终保留 rule 路径作为 ground truth 解释；(c) 对每个 hotspot 用 attention rollout 给"为什么"的轻量归因 |
| GNN 误判导致错误教学建议 | 误导学生 | (a) 在 report 中所有 GNN 字段标 `confidence`；(b) UI 层对 confidence < 0.7 的提示弱化展示；(c) 永不让 GNN 改 `logic_correct` |
| 新电路泛化不足 | test set 上 acc 崩 | (a) test split 强制留出新拓扑；(b) 节点/边特征严格 type-agnostic（用 ctype one-hot 而非 component-id embedding）；(c) 监控 OOD —— 推理时若 max(component_match_score) < 0.4，advice 整体置 `low_confidence=true` 不参与决策 |
| 推理延迟（影响评测响应） | 用户体验 | (a) MVP 在 CPU 上 < 100 ms（5 ref 电路 < 50 节点）；(b) `inference.py` 设 300 ms hard timeout；(c) 模型 ≤ 2 MB ckpt 常驻内存 |
| 与规则系统冲突 | 教学反馈矛盾 | 第六节冲突仲裁表明确以 rule 为准；冲突写入 `summary.gnn.disagreement_with_rule`，便于离线追踪迭代 |

---

## 十一、关键既有文件（不要重复造轮子）

- 图构造：`app/domain/logical_reference.py:156` (`logical_reference_to_graph`)、`:215` (`current_netlist_v2_to_graph`)
- 比较入口：`app/domain/compare/orchestrator.py` `compare_logical_graphs`
- 对称检测：`app/domain/compare/matcher.py` `auto_detect_symmetries`
- 角色推断：`app/domain/compare/role_inference.py`
- net 角色启发式：`app/domain/net_normalization.py`
- Report 增强：`app/domain/compare/diff_report.py:13` (schema 常量)、`_enrich_result`
- 既有 fixture：`tests/fixtures/references/`、`tests/fixtures/netlist_v2/`、`tests/fixtures/validator_error_codes/`（35+ 错误用例可直接做 GNN ground truth）
- 测试范式：`tests/domain/test_graph_compare.py`、`test_graph_compare_detailed.py`

---

## 附录 A · P0 执行细则（Schema · 3 天）

### 目标
建立 GNN 模块的"地基"：包目录、依赖声明、schema 常量、中间数据结构、port-level 图构建器。**不写模型、不写训练、不写推理。** 所有代码必须能被现有 fixture 喂入并产出符合预期的中间结构。

### 新建文件清单

```
app/domain/gnn/
├── __init__.py            ← 暂仅占位 + 公开未来 API 名（暂时 raise NotImplementedError）
├── graph_schema.py        ← 枚举 + 特征常量 + 维度常量
├── hetero_circuit.py      ← @dataclass HeteroCircuitGraph（中间结构）
├── port_graph.py          ← 把 NetworkX bipartite → HeteroCircuitGraph
└── README.md              ← 模块说明（≤ 1 页）

tests/domain/gnn/
├── __init__.py
├── test_graph_schema.py
├── test_port_graph.py
└── conftest.py            ← 复用 tests/fixtures/* 的 helper
```

### 文件 1 · `app/domain/gnn/graph_schema.py`

仅包含**常量与枚举**，无运行逻辑。内容：

> ⚠️ **下表是 P0 初始版本**。P0.5 把 PortType 扩到 23 项、P0.6 加 ConnectionPolicy + pin_number + symmetry_class_size_inverse、PORT_FEAT_DIM 从 37 演进到 50。当前实现以 §三 节点 schema 表为准，本附录仅保留历史路径供回溯。

| 段 | 内容 |
|---|---|
| `ComponentType` enum | 16 类，与 `app/domain/circuit.py` 现有规范化值一一对齐：Resistor / Capacitor / CapacitorCeramic / CapacitorElectrolytic / Wire / LED / Diode / Transistor / IC / Potentiometer / OpAmp（IC 子类，预留）/ VoltageSource / CurrentSource / Switch / Sensor / UNKNOWN |
| `PortType` enum | ~~16 类~~（**P0.5 已扩到 23 类**，含 op-amp INVERTING_INPUT / NON_INVERTING_INPUT / OUTPUT / OFFSET_NULL / NC / V_PLUS / V_MINUS） |
| `NetRole` enum | 6 类：input / output / vcc / gnd / signal / unknown（与 `normalize_net_role` 对齐） |
| `PolarityClass` enum | 3 类：none / two_polar / multi_asymmetric |
| `SourceType` enum | 3 类：dsl / vision / inferred |
| 维度常量 | `COMPONENT_FEAT_DIM = 30`、~~`PORT_FEAT_DIM = 37`~~（**当前 50**）、`NET_FEAT_DIM = 11`、`PORT_NET_EDGE_FEAT_DIM = 5`、`DRNL_LABEL_DIM = 17` |
| 特征布局常量 | `COMPONENT_FEAT_LAYOUT: list[tuple[str, slice]]`、`PORT_FEAT_LAYOUT`、`NET_FEAT_LAYOUT` —— 让 encoder 后续按 slice 取子特征做调试 |
| 类型映射工具 | `CTYPE_TO_INDEX: dict[str, int]`、`PORT_TYPE_TO_INDEX`、`NET_ROLE_TO_INDEX`，索引用于 one-hot |
| 极性元数据 | `POLARITY_CLASS_OF: dict[ComponentType, PolarityClass]`，对齐 `circuit.py` 的 `POLARIZED_TYPES` / `NON_POLAR_TYPES` |

**禁止**在该文件 import torch / torch_geometric（保持纯 Python，便于在无 GPU 环境单元测试）。

### 文件 2 · `app/domain/gnn/hetero_circuit.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

NodeKind = Literal["component", "port", "net"]
Side = Literal["ref", "cur"]

@dataclass(frozen=True)
class ComponentNode:
    node_id: str               # "<side>_comp:<source_id>"
    side: Side
    source_id: str
    ctype: str                 # ComponentType.value
    package: str | None
    polarity_class: str        # PolarityClass.value
    pin_count: int
    value: float | None
    confidence: float

@dataclass(frozen=True)
class PortNode:
    node_id: str               # "<side>_port:<comp_source_id>.<port_key>"
    side: Side
    parent_component_id: str   # ComponentNode.node_id
    port_key: str              # 原始 pin_name 或 normalized pin_role
    port_type: str             # PortType.value
    parent_ctype: str          # 复制自 component
    polarity_sensitive: bool
    is_power_port: bool
    is_ground_port: bool
    is_floating: bool          # cur 侧若该 port 没连任何 net 则 True

@dataclass(frozen=True)
class NetNode:
    node_id: str               # "<side>_net:<source_id>"
    side: Side
    source_id: str
    role: str                  # NetRole.value
    role_label: str | None
    is_power_rail: bool
    voltage_hint: float | None
    aliases: tuple[str, ...] = ()

@dataclass(frozen=True)
class PortConnectsNetEdge:
    src_port_id: str
    dst_net_id: str
    connection_confidence: float
    source_type: str           # SourceType.value
    is_observed_in_cur: bool

@dataclass
class HeteroCircuitGraph:
    side: Side
    components: dict[str, ComponentNode] = field(default_factory=dict)
    ports: dict[str, PortNode] = field(default_factory=dict)
    nets: dict[str, NetNode] = field(default_factory=dict)
    port_of_component: dict[str, list[str]] = field(default_factory=dict)
    edges: list[PortConnectsNetEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict[str, int]:
        return {
            "n_components": len(self.components),
            "n_ports": len(self.ports),
            "n_nets": len(self.nets),
            "n_edges": len(self.edges),
        }
```

设计要点：所有节点 frozen（hash 友好，便于映射）；`HeteroCircuitGraph` 本身可变，便于增量构造；不依赖 torch（torch 转换在 P2 `pyg_converter.py`）。

### 文件 3 · `app/domain/gnn/port_graph.py`

主函数：`build_hetero_circuit_graph(nx_graph: nx.Graph, side: Side) -> HeteroCircuitGraph`。

**算法**（直接消费 `logical_reference_to_graph` / `current_netlist_v2_to_graph` 的输出）：

1. 遍历 `nx_graph.nodes(data=True)`：
   - `kind == "comp"` → 构造 `ComponentNode`，`pin_count` 通过该节点的度数推断（先放 0，第 3 步回填），`polarity_class` 用 `POLARITY_CLASS_OF[ctype]`，`package` 暂从 `data.get("package")` 取（缺则 None）
   - `kind == "net"` → 构造 `NetNode`，`is_power_rail = role in {"power","ground"}`
2. 遍历 `nx_graph.edges(data=True)` —— 每条 `(comp_node, net_node, attrs)`：
   - 抽出 `pin = attrs["pin"]`、`pin_role = attrs["pin_role"]`、`comp_type = attrs["comp_type"]`
   - **构造 PortNode**，node_id = `f"{side}_port:{comp.source_id}.{pin}"`（pin 为字符串，先做 slugify：`pin.replace(' ','_')`）
   - port_type 由 `_normalize_port_type(pin_role, comp_type)` 给出（小工具函数：先查 PinRole 直接映射，否则 fallback 到 PIN_SYMMETRIC / 数字 pin → PIN1/PIN2/PINn_generic）
   - `polarity_sensitive` 由 `POLARITY_CLASS_OF[comp_type] != "none"` 决定，且 port_type ∈ {ANODE, CATHODE, BASE, COLLECTOR, EMITTER, POSITIVE, NEGATIVE}
   - 加入 `port_of_component[comp_node_id]`
   - 构造 `PortConnectsNetEdge(src_port_id=port_id, dst_net_id=net_node_id, connection_confidence=1.0, source_type=...)` —— 参考侧 `source_type="dsl"`、当前侧 `"vision"`，`connection_confidence` 暂固定 1.0（cur 侧若 nx_graph 上携带 `confidence` 属性则采用之）
3. 回填 `ComponentNode.pin_count = len(port_of_component[comp_id])`
4. 对 `cur` 侧：扫描所有 ComponentInstance 的 declared pins（这一步**先不做**，留 P0 末尾用 TODO 注释标记 `is_floating` 永远为 False；P1 接 netlist_v2 原始数据时再回填）
5. 把 `nx_graph.graph["format"]` / `reference_id` 复制到 `HeteroCircuitGraph.metadata`
6. 返回

**便利函数**：
- `build_from_logical_reference(payload: dict) -> HeteroCircuitGraph`（内部调 `logical_reference_to_graph` 再调 `build_hetero_circuit_graph(_, "ref")`）
- `build_from_netlist_v2(netlist_v2: dict) -> HeteroCircuitGraph`（同理）

**禁止**：在 P0 阶段引入 torch；不做特征向量化（向量化是 P2 的事）。

### 文件 4 · `app/domain/gnn/__init__.py`

```python
from .graph_schema import (
    ComponentType, PortType, NetRole, PolarityClass, SourceType,
    COMPONENT_FEAT_DIM, PORT_FEAT_DIM, NET_FEAT_DIM,
)
from .hetero_circuit import (
    HeteroCircuitGraph, ComponentNode, PortNode, NetNode, PortConnectsNetEdge,
)
from .port_graph import (
    build_hetero_circuit_graph,
    build_from_logical_reference,
    build_from_netlist_v2,
)

__all__ = [
    "ComponentType", "PortType", "NetRole", "PolarityClass", "SourceType",
    "COMPONENT_FEAT_DIM", "PORT_FEAT_DIM", "NET_FEAT_DIM",
    "HeteroCircuitGraph", "ComponentNode", "PortNode", "NetNode",
    "PortConnectsNetEdge",
    "build_hetero_circuit_graph",
    "build_from_logical_reference", "build_from_netlist_v2",
]

# 推理 API（P4 实现）
class GNNAdvisor:
    @classmethod
    def get(cls):
        raise NotImplementedError("GNNAdvisor will be implemented in P4")

def should_use_gnn(ctx) -> bool:
    return False  # P0 stub；P4 替换为真实实现
```

### 文件 5 · `pyproject.toml` 改动

在 `[project.optional-dependencies]` 末尾新增：

```toml
gnn = [
    "torch>=2.2",
    "torch-geometric>=2.5",
]
```

**只声明，不安装**。本机如需调试可 `pip install -e ".[gnn]"`，但 P0 自身的单元测试不依赖 torch（schema + dataclass 是纯 Python）。

### 文件 6 · 测试

#### `tests/domain/gnn/conftest.py`

```python
import json
from pathlib import Path
import pytest

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures"

@pytest.fixture
def rc_reference_payload():
    return json.loads((FIXTURE_ROOT / "references" / "test_rc_v1.json").read_text())

@pytest.fixture
def all_reference_payloads():
    return {
        p.stem: json.loads(p.read_text())
        for p in (FIXTURE_ROOT / "references").glob("*.json")
    }

@pytest.fixture
def simple_netlist_v2():
    return json.loads((FIXTURE_ROOT / "netlist_v2" / "mapped_components_simple.json").read_text())
```

#### `tests/domain/gnn/test_graph_schema.py` · 至少 4 个用例

1. `test_component_type_enum_covers_circuit_module` —— 断言 `ComponentType` 的 value 集合是 `circuit.py` 中 `norm_component_type` 所有可能输出的超集。
2. `test_feature_dimensions_are_consistent` —— 断言 `COMPONENT_FEAT_DIM == sum(width for _, sl in COMPONENT_FEAT_LAYOUT)`，对 PORT/NET 同理。
3. `test_polarity_class_table_complete` —— `POLARITY_CLASS_OF` 必须覆盖所有 `ComponentType`。
4. `test_port_type_includes_all_pin_roles` —— `PortType` value 集合 ⊇ `PinRole` value 集合。

#### `tests/domain/gnn/test_port_graph.py` · 至少 6 个用例

1. `test_build_from_rc_reference_basic_shape(rc_reference_payload)`：
   - 期望 `summary()` = `{n_components: 2, n_ports: 4, n_nets: 3, n_edges: 4}`
   - R1 应该有 2 个 port（pin1/pin2），C1 同理
2. `test_port_node_ids_are_deterministic_and_unique`：node_id 全局唯一，命名形如 `ref_port:R1.1`、`ref_port:R1.2`
3. `test_polarity_sensitive_flag`：LED 电路中 LED 的 anode/cathode port `polarity_sensitive=True`；R 的两个 port `polarity_sensitive=False`
4. `test_metadata_propagated`：`hcg.metadata["format"] == "logical_reference_v1"`
5. `test_all_reference_fixtures_convert(all_reference_payloads)`：遍历 `tests/fixtures/references/*.json`，每个都能成功转换且 `n_ports >= n_components`、`n_edges == n_ports`
6. `test_build_from_netlist_v2_smoke(simple_netlist_v2)`：调 `current_netlist_v2_to_graph` → `build_hetero_circuit_graph(_, "cur")`，断言 side="cur" 且 node_id 前缀为 `cur_*`

### 验收清单（DoD）

- [ ] `pytest tests/domain/gnn/` 全绿
- [ ] `pytest tests/domain/test_graph_compare*.py` 仍全绿（**零回归**，没动现有 compare 模块）
- [ ] `python -c "from app.domain.gnn import build_from_logical_reference, HeteroCircuitGraph; print('ok')"` 不报错
- [ ] `ruff check app/domain/gnn/ tests/domain/gnn/` 无 violation
- [ ] `mypy app/domain/gnn/` 通过（仅本目录范围）
- [ ] `app/domain/gnn/README.md` ≤ 1 页，说明：模块定位、当前阶段、与 logical_reference.py 的关系图（一段文字即可）
- [ ] 新增文件总行数（不含测试）≤ 600，避免过早抽象

### 不在 P0 范围（明确推出）

- ❌ torch / torch_geometric 的实际 import 与使用
- ❌ HeteroData 转换（→ P2）
- ❌ SEAL 子图抽取（→ P0.7）
- ❌ DRNL 标签 / 任何特征向量化（→ P2）
- ❌ 模型、训练、推理（→ P3 / P4）
- ❌ 修改 `app/domain/compare/orchestrator.py` 或 `diff_report.py`（→ P4）

### 复用既有工具

- `app/domain/logical_reference.py:156` `logical_reference_to_graph(payload)` —— **不要重新实现**，直接调
- `app/domain/logical_reference.py:215` `current_netlist_v2_to_graph(netlist_v2)` —— 同上
- `app/domain/circuit.py` `norm_component_type`、`PinRole` enum、`POLARIZED_TYPES`、`NON_POLAR_TYPES`、`THREE_PIN_TYPES` —— P0 schema 必须与之对齐
- `app/domain/net_normalization.py` `normalize_net_role` —— net role 字符串来源
- 测试 fixture：`tests/fixtures/references/*.json`、`tests/fixtures/netlist_v2/*.json`

### 时间盒

| 子任务 | 预估 |
|---|---|
| 1. 起目录骨架 + `__init__.py` + pyproject extras | 0.5 h |
| 2. `graph_schema.py`（枚举 + 维度常量 + 类型映射） | 2 h |
| 3. `hetero_circuit.py` dataclasses | 1 h |
| 4. `port_graph.py` 构建器 + 工具函数 | 4 h |
| 5. conftest + 10 个单元测试 | 4 h |
| 6. README + ruff/mypy 通关 | 1 h |
| 7. 验收 / 调试余量 | 3.5 h |
| **合计** | **~16 h（≈ 2 个工作日）** |

完成后即可推进 **P0.5**（IC + Pot port 语义稳定化），再到 P0.7（SEAL 子图抽取）。

---

## 附录 A.5 · P0.5 执行细则（IC + Pot port 语义稳定化 · 2 天）

### 触发原因
P0 落地的 16-class `PortType` 对两类元件描述不足：
1. **IC（UA741 等运放）** —— 8 个 pin 全部归入 `PIN_N_GENERIC`，SEAL 拿到的 port 嵌入无法区分"反相输入 / 同相输入 / 输出 / 电源 / NC"。一旦在不稳的 port schema 上跑 P0.7（DRNL + SEAL）和 P1（合成数据），所有缓存的 `seal_edges/*.pt` 都得重做。
2. **Potentiometer** —— wiper 与 terminal_a/b 在结构上不可互换，但当前 `polarity_sensitive` 仅在 `POLARITY_SENSITIVE_PORT_TYPES` 列了 anode/cathode/base/collector/emitter/positive/negative，wiper 漏了。

P0.5 在动 SEAL 前一次性补齐。

### 改动清单

**1. `app/domain/gnn/graph_schema.py`** —— 扩 `PortType` 至 23 项：
- 新增 `INVERTING_INPUT` / `NON_INVERTING_INPUT` / `OUTPUT` / `OFFSET_NULL` / `NC` / `V_PLUS` / `V_MINUS`
- 与 ``app.domain.ic_models.UA741_PIN_ROLES`` 严格同步（`offset_null_1` / `offset_null_2` 在 GNN 层合并为 `OFFSET_NULL`，其它一一对齐）
- 重算 `PORT_FEAT_DIM = 23 + 16 + 5 = 44`（旧 37）。**运行时 self-check 会自动验证；不更新就 import 失败**
- 新增 `POLARITY_SENSITIVE_PORT_TYPES ⊇ {WIPER, INVERTING_INPUT, NON_INVERTING_INPUT, OUTPUT, V_PLUS, V_MINUS}`
- 新增 `POWER_PORT_TYPES ⊇ {V_PLUS, V_MINUS}`
- 新增 `IC_PIN_MAPS: dict[str, dict[str, str]]` 注册表：键是 `part_subtype.upper()`，值是 `pin_name_or_number → PortType.value`。首批支持：`UA741`、`LM358`、`NE555`（MVP 5 电路只用到 UA741；其余预留）
- 新增 `_OPAMP_PIN_ALIASES: dict[str, str]`：覆盖 DSL 用人话写引脚的常见写法（如 `"in-"` / `"inv"` / `"-in"` → `inverting_input`；`"v+"` / `"vcc"` 对 op-amp 上下文 → `v_plus`）
- 重构 `normalize_port_type(pin_role, *, component_type=None, part_subtype=None, pin_raw=None)`：
  1. 若 `component_type ∈ {IC, OpAmp}` 且 `(part_subtype, pin_raw)` 命中 IC_PIN_MAPS → 返回映射
  2. 否则查 `_OPAMP_PIN_ALIASES`
  3. 否则走原 PinRole 直查 / pin1/pin2 / 数字 → pin_n_generic / generic 兜底

**2. `app/domain/gnn/port_graph.py`** —— 让 subtype 流到 normalizer：
- `build_hetero_circuit_graph(nx_graph, side, *, subtype_by_source_id: dict[str, str] | None = None)` 新增 kwarg
- `build_from_logical_reference(payload)`：从 `payload["components"][*]["subtype"]` 抽 dict，传入
- `build_from_netlist_v2(netlist_v2)`：从 `netlist_v2["components"][*]["part_subtype"]` 抽 dict，传入
- 边循环时把 subtype + pin 原始名一并喂给 `normalize_port_type`
- ⚠️ 不动 `app/domain/logical_reference.py`（保持零侵入）；subtype 通过 payload 旁路传递

**3. 新增 fixture** —— `tests/fixtures/references/test_opamp_buffer_v1.json`：UA741 单位增益缓冲器（V+ / V- / IN+ → in / IN- → out / OUT → out / OFFSET ×2 / NC），8 个 pin 全部出场。P1 合成数据集会复用。

**4. 测试** —— 扩 `test_graph_schema.py`、新增 `tests/domain/gnn/test_ic_pot_semantics.py`：
- `test_port_type_extended_to_23_members`
- `test_port_feat_dim_updated`
- `test_ic_pin_maps_ua741_complete`（8 个 pin 全有映射，且每个映射值 ∈ PortType.values）
- `test_normalize_port_type_ua741_by_number`：`(pin_role="2", ctype="IC", subtype="UA741", pin_raw="2")` → `inverting_input`
- `test_normalize_port_type_opamp_alias`：`(pin_role="in-", ctype="OpAmp")` → `inverting_input`
- `test_wiper_is_polarity_sensitive`
- `test_build_opamp_buffer_fixture_ports`：fixture → HeteroCircuitGraph 后断言每个 pin 的 port_type 与 UA741_PIN_ROLES 对齐；输入 / 输出 pin polarity_sensitive=True
- `test_build_handcrafted_potentiometer_graph`：wiper port polarity_sensitive=True，terminal_a / terminal_b 各 polarity_sensitive=False
- 现有 RC / LED 测试不回归

### 复用既有模块（**不要造轮子**）
- `app/domain/ic_models.py:9` `UA741_PIN_ROLES` —— 直接 import，构造 IC_PIN_MAPS["UA741"]
- `app/domain/netlist_models.py:91` `ComponentInstance.part_subtype` —— cur 侧字段来源
- `app/domain/dsl/compile.py:67` —— 证明 DSL 已经在序列化 `subtype` 字段

### DoD
- [ ] `pytest tests/domain/gnn/` 全绿（含新测试）
- [ ] 既有 29 个 compare 测试零回归
- [ ] `ruff check` / `mypy` 仍然 clean
- [ ] PORT_FEAT_DIM 一致性 self-check 在 import 期通过
- [ ] UA741 fixture 通过完整 DSL → HeteroCircuitGraph 路径，8 个 port 各拿到正确 port_type

### 时间盒
| 子任务 | 预估 |
|---|---|
| 1. PortType 扩张 + 维度重算 + 自检 | 1 h |
| 2. IC_PIN_MAPS + alias 表 + normalize_port_type 重构 | 2 h |
| 3. port_graph subtype 旁路传递 | 1 h |
| 4. UA741 fixture（手写 JSON） | 0.5 h |
| 5. 测试（5 schema + 3 integration） | 3 h |
| 6. ruff / mypy / 调试余量 | 1.5 h |
| **合计** | **~9 h（≈ 1.5 个工作日）** |

完成后推进 **P0.6**（Package port materialization + symmetry policy），再到 P0.7（SEAL）。

---

## 附录 A.7 · P0.7 执行细则（SEAL Enclosing Subgraph + DRNL · 2 天）

### 目标
在 P0/0.5/0.6 已稳定的 `HeteroCircuitGraph` 之上，实现 **GNN-ACLP 范式的 SEAL 子图抽取与 DRNL 标签计算**。这是 SEAL link-prediction head 的输入流水线 —— 把整张电路图按"目标候选边"拆成一组独立的小子图。**不引入 torch**（向量化推到 P2 PyG converter）。

### 新建文件
```
app/domain/gnn/seal_subgraph.py
tests/domain/gnn/test_seal_subgraph.py
```

### 公开 API
```python
@dataclass(frozen=True)
class SealSubgraph:
    target_port_id: str
    target_net_id: str
    edge_present: bool                # 是 cur 中已存在 (wrong 检测) 还是缺连 (suggested_target)
    num_hops: int
    port_ids: tuple[str, ...]         # 包含的 port 节点，anchor 排第一
    net_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]   # (port, net) 边；候选边本身已排除
    drnl_labels: dict[str, int]
    is_target: dict[str, bool]

def extract_seal_subgraph(
    hcg, port_node_id, net_node_id, num_hops=2, *,
    edge_present: bool | None = None,
) -> SealSubgraph: ...

def extract_subgraphs_for_observed_edges(hcg, num_hops=2) -> list[SealSubgraph]: ...
def extract_subgraphs_for_floating_ports(
    hcg, num_hops=2, candidate_nets: list[str] | None = None,
    exclude_forbidden: bool = True,
) -> list[SealSubgraph]: ...
# ↑ ⚠️ 已修订（P0.7 收尾 audit）：`exclude_forbidden` 改为
#   `policies: frozenset[ConnectionPolicy] = frozenset({REQUIRED})`，
#   两个 batched 入口都接受 `include_same_component_edges` kwarg。
#   见模块当前实现 app/domain/gnn/seal_subgraph.py。
```

### 算法要点
1. **底图是 (port, net) 二分图**：component 节点不进 SEAL 子图，但每个 port 的 component 上下文已在 port 特征里（parent_ctype, polarity_class）。
2. **候选边必须在 BFS 与最终子图边集中都剔除**（SEAL 守则：模型不可窥见目标）。
3. **DRNL 公式**（Zhang & Chen 2018）：
   - 锚点 `port_u` / `net_v` → 标签 1
   - 不可达节点 → 标签 0
   - 其它：`d = d_u + d_v; d_half = d // 2; label = 1 + min(d_u, d_v) + d_half * (d_half + (d % 2) - 1)`
4. **节点顺序**：anchor port 首位、anchor net 首位，其余按 node_id 字典序，**确定性**（便于测试与缓存键）。
5. **`suggested_target` 候选 net 集合**：默认是 cur 中**所有 net**。
6. **`extract_subgraphs_for_floating_ports` 的 ConnectionPolicy 过滤策略**（**P0.7 收尾审计修订**）：默认仅 `policies={REQUIRED}` —— OPTIONAL pin（如 UA741 offset_null）允许保持 floating，把它打成"应该接"会给 P1 合成标签注入系统性噪声；FORBIDDEN（UA741 pin 8 NC）默认排除。需要审计 / 教学增强场景显式传 `policies={REQUIRED, OPTIONAL}` 或全集。
7. **`same_component_edges` 字段**（**P0.7 收尾审计预留**）：默认空。开启 `include_same_component_edges=True` 才填充"同片 IC / 同只 BJT 的兄弟 pin"成对边。**DRNL 距离始终在 bipartite (port, net) 底图上 BFS**，本字段只是 P2/P3 模型的可选结构信号，是否消费由后续阶段决定。

### 测试覆盖
1. DRNL 公式手算（du=1,dv=1→2；du=1,dv=2→3；du=2,dv=2→5；du=2,dv=3→7）
2. 锚点 → 1；不可达 → 0
3. 2-hop 边界正确（3-hop 节点不在子图）
4. 候选边在 `edges` 中已剔除；BFS 也用了去边图
5. UA741 fixture：为某条已存在边抽取，断言子图大小合理、anchors 正确
6. `extract_subgraphs_for_observed_edges`：边数恰好等于 hcg.edges
7. `extract_subgraphs_for_floating_ports`：FORBIDDEN 的 pin 8 不在候选；OPTIONAL/REQUIRED 的浮空 port 会出现
8. 性能：UA741 全图（≤ 50 边）抽取 < 30 ms

### DoD
- [ ] pytest 全绿，新增 ≥ 12 项 SEAL 测试
- [ ] ruff / mypy clean
- [ ] 既有 119+29 测试零回归
- [ ] README 新增 P0.7 段
- [ ] 不引入 torch / torch_geometric

### 时间盒
| 子任务 | 预估 |
|---|---|
| 1. SealSubgraph dataclass + BFS / DRNL 实现 | 3 h |
| 2. extract_seal_subgraph + 两个 batched helper | 3 h |
| 3. 测试 (≥ 12 项含手算 DRNL + UA741 集成) | 4 h |
| 4. README + ruff/mypy + 性能验证 | 2 h |
| **合计** | **~12 h（≈ 2 工作日）** |

完成后推进 **P0.8**（label builder + ref↔cur alignment），再到 P1。

---

## 附录 A.8 · P0.8 执行细则（Label Builder + Alignment · 2-2.5 天）

### 触发原因（外审 + GNN-ACLP 数据集对比）
P0.7 把抽图层做干净后，**打标签层**完全空白。直接进 P1 perturbation 会被迫在
perturbation 代码里穿插 label 逻辑，污染两个关心点。

GNN-ACLP（arxiv 2504.10240v5）的标签方式审计发现 3 处他们做错 / 没做、我们必须做对：

1. **k-fold 在 edge 上 split** —— 把 94 个电路合并成一张大图后 random
   split，test edge 与 train edge 来自同电路 → 隐性数据泄漏。**我们必须按
   ref 电路 split**（plan §五 早已定）。
2. **无 ConnectionPolicy** —— 随机非边都被采样为负，把 UA741 offset_null
   "可不接"的边也当负样本，标签语义模糊。**我们 OPTIONAL 默认排除**。
3. **无 symmetry** —— R.pin1↔pin2 swap 在他们框架里会被打成"wrong edge"。**我们
   必须把 sibling pin swap 识别为合法**（pin_symmetry_groups + symmetry_class_id
   在 P0.6 已落库，但 P0.7 没用过；P0.8 是首次消费）。

### 已决策（本次确认）

| 决策点 | 选择 |
|---|---|
| Symmetric swap 标签 | **Both edges = label 1**（cur 实际边 + sibling 合成边都作 positive） |
| FORBIDDEN pin 负样本 | **默认 4 条合成负** + violated 边各自一条 |
| missing_component perturbation | **Silently skip + log**（不在 edge level 表达，留给 graph-level error_type 头） |

### 新建文件

```
app/domain/gnn/alignment.py             ← ComponentAlignment + identity / explicit constructors
app/domain/gnn/label_builder.py         ← SealSample, LabelSource, build_seal_samples
tests/domain/gnn/test_alignment.py       ← ≥ 6 测试
tests/domain/gnn/test_label_builder.py   ← ≥ 15 测试
```

### 公开 API（含 P0.8 二轮 audit 修订）

```python
# alignment.py
@dataclass(frozen=True)
class ComponentAlignment:
    """ref ↔ cur 的 source_id 对齐表。component 与 net 各一份。"""
    ref_to_cur_component: dict[str, str]
    ref_to_cur_net: dict[str, str]
    cur_to_ref_component: dict[str, str]      # 反向缓存
    cur_to_ref_net: dict[str, str]
    notes: dict = field(default_factory=dict) # perturbation 写"做了什么改动"

    def map_ref_port_to_cur_port_id(
        self, ref_port_id: str, ref_hcg, cur_hcg
    ) -> str | None: ...
    def map_ref_net(self, ref_net_id: str) -> str | None: ...

def identity_alignment(ref, cur) -> ComponentAlignment: ...     # 同 source_id 自动对齐
def alignment_from_dicts(
    ref, cur, component_map: dict[str, str], net_map: dict[str, str]
) -> ComponentAlignment: ...

# label_builder.py

class TaskType(str, Enum):
    """一个 SealSample 服务哪个模型 head（决定 P3 loss 形态）。

    **WRONG_EDGE** 覆盖**所有**点式 P(edge_correct) 监督，**不仅是 cur 中实
    际存在的边**：含 ref-present 正样本、cur 实际错边（WRONG_OBSERVED 强
    负）、FORBIDDEN-violated、FORBIDDEN-合成、随机负、ref-absent-REQUIRED
    （正样本，告诉模型该有边但 cur 没接）。

    **MISSING_EDGE** 覆盖 per-port N-way ranking / softmax：给 REQUIRED-floating
    或 REQUIRED-wrong_redirect 的 port 在候选 net 集合中选正确目标。推理时
    suggested_target head 复用此训练。
    """
    WRONG_EDGE   = "wrong_edge"
    MISSING_EDGE = "missing_edge"

class LabelSource(str, Enum):
    REF_PRESENT          = "ref_present"
    REF_SYMMETRIC_SWAP   = "ref_symmetric_swap"
    REF_ABSENT_REQUIRED  = "ref_absent_required"
    WRONG_OBSERVED       = "wrong_observed"   # **关键负样本**：cur 中实际存在的、
                                              # 不在 ref-sym-aware 正集合中的边。
                                              # 必须 100% 标负，不能依赖 NEGATIVE_RANDOM
                                              # 偶然采到 —— 否则 WRONG_EDGE 主头被系统性弱监督
    FORBIDDEN_VIOLATED   = "forbidden_violated"
    FORBIDDEN_NEGATIVE   = "forbidden_negative"
    NEGATIVE_RANDOM      = "negative_random"
    NEGATIVE_HARD        = "negative_hard"    # 预留 slot；P0.8 不生成

@dataclass(frozen=True)
class SealSample:
    """单个 (subgraph, label) 训练样本。

    P0.8 二轮 audit 修订：candidate_edge / expected_edge / task_type / group_id
    全部上提到顶层，避免下游 dataloader 再去 unwrap subgraph。
    """
    subgraph: SealSubgraph
    label: int                                   # 0 or 1
    label_source: str                            # LabelSource.value
    task_type: str                               # TaskType.value
    candidate_edge: tuple[str, str]              # (cur_port_id, cur_net_id) — 该 sample 评分的对象
    expected_edge: tuple[str, str] | None = None # 该 port 本应连接的"对的" net；MISSING_EDGE 正样本时填
    ref_edge_origin: tuple[str, str] | None = None  # (ref port source_id, ref net source_id) 来源溯源
    confidence: float = 1.0
    is_symmetric_equivalent: bool = False        # sibling pin sym-swap 产生的合成正样本
    group_id: str | None = None                  # 同 group_id 的 sample 属于同一个 MISSING_EDGE ranking 组

@dataclass(frozen=True)
class SealSampleGroup:
    """多个候选 SealSample 构成的"互斥候选集"，用于 ranking 任务
    （MISSING_EDGE 与推理 suggested_target）。WRONG_EDGE 任务不需要。"""
    group_id: str
    task_type: str                              # 必为 TaskType.MISSING_EDGE
    query_port_id: str                          # 该组要回答 "这个 port 该接到哪个 net"
    sample_indices: tuple[int, ...]             # 指向 LabelBuildResult.samples 列表的 index
    correct_index: int | None                   # 哪个 sample.label==1（None 表示组内无正样本，训练时跳过）

@dataclass(frozen=True)
class LabelStats:
    """构造过程的可观测性指标 —— P1 dataset_builder 每个 perturbation 后
    应记录到日志 / 数据集 manifest，便于发现 silent failure（如某天
    FORBIDDEN_VIOLATED 突然降到 0）。"""
    total_samples: int
    n_positives: int
    n_negatives: int
    pos_neg_ratio: float                        # n_positives / max(1, n_negatives)
    by_source: dict[str, int]                   # LabelSource.value → count
    by_task_type: dict[str, int]                # TaskType.value → count
    n_groups: int                               # SealSampleGroup 数量（MISSING_EDGE only）
    n_groups_without_positive: int              # ref expected 但 cur 中 spec 限制无法 surface 的 group
    n_skipped_missing_component: int            # 决策 3：silently skip 的次数
    n_skipped_optional_pin: int
    n_skipped_forbidden_pin_no_violation: int
    n_skipped_extract_error: int
    n_unique_ports_covered: int
    n_unique_nets_covered: int

@dataclass(frozen=True)
class LabelBuildResult:
    samples: tuple[SealSample, ...]
    groups: tuple[SealSampleGroup, ...]
    stats: LabelStats

def build_seal_samples(
    ref_hcg, cur_hcg, alignment,
    *,
    negatives_per_positive: float = 1.0,
    include_optional: bool = False,
    forbidden_negative_samples: int = 4,
    missing_edge_group_size: int = 5,           # 每个 MISSING_EDGE 组内候选 net 数
    enable_hard_negative_mining: bool = False,  # P0.8 永远 False；预留
    seed: int = 0,
    num_hops: int = 2,
    include_same_component_edges: bool = False,
) -> LabelBuildResult: ...
```

### Hard-negative 策略（P3 启用前文档化，P0.8 不实现）

`NEGATIVE_HARD` enum slot 已留。未来 3 种候选策略，按优先级：

| 策略 | 生成规则 | 适用场景 |
|---|---|---|
| **same_port_wrong_net** | 对每个正 `(p, n_correct)`，采样 `(p, n_wrong)` where `n_wrong ∈ cur.nets \ correct_nets_for_p` | 教模型在 port 上分辨"该接哪个 net"；R-divider 极有效 |
| **same_net_wrong_port** | 对每个正 `(p_correct, n)`，采样 `(p_wrong, n)` where `p_wrong` 是其它 REQUIRED port | 教模型在 net 上分辨"该接哪个 port"；多 R 并联场景关键 |
| **1_hop_perturbation** | 把正边的一个端点替换为它在 bipartite 图中的 1-hop 邻居 | 通用 hard mining；保留拓扑相似但语义不同 |

P0.8 只暴露 `enable_hard_negative_mining` kwarg 占位；P3 训练曲线指导启用哪一种。

### 算法 5 步（含 group 构造与 stats 累积）

构建过程用一个 `_Builder` 上下文对象累积 samples / groups / counters，最后
冻结成 `LabelBuildResult`。

**Step 1 · WRONG_EDGE samples from ref edges（含 sym swap 展开 — 决策 1）**
```
for ref_edge in ref.edges:
    cur_port = alignment.map_ref_port(ref_edge.src, ref, cur)
    cur_net  = alignment.map_ref_net(ref_edge.dst)
    if cur_port is None:
        stats.n_skipped_missing_component += 1; continue        # 决策 3
    port = cur.ports[cur_port]
    if port.policy == OPTIONAL and not include_optional:
        stats.n_skipped_optional_pin += 1; continue
    if port.policy == FORBIDDEN:
        # ref 说应该连，但 cur policy 禁 → spec 矛盾，记 warn 后跳
        log.warning(...); continue

    actually_present = (cur_port, cur_net) in cur_edges
    siblings = sym_class_siblings(cur, cur_port)
    anchors = [cur_port] + list(siblings)
    for anchor in anchors:
        is_sym = (anchor != cur_port)
        edge_present = (anchor, cur_net) in cur_edges
        sg = extract_seal_subgraph(cur, anchor, cur_net, edge_present=edge_present, ...)
        source = (REF_SYMMETRIC_SWAP if is_sym else
                  REF_PRESENT if actually_present else REF_ABSENT_REQUIRED)
        builder.add_sample(SealSample(
            subgraph=sg, label=1, label_source=source.value,
            task_type=TaskType.WRONG_EDGE.value,
            candidate_edge=(anchor, cur_net),
            expected_edge=(cur_port, cur_net),
            ref_edge_origin=(ref_edge.src_source_id, ref_edge.dst_source_id),
            is_symmetric_equivalent=is_sym,
        ))
```

**Step 2 · MISSING_EDGE / suggested-target groups（统一两种来源）**

**两种触发场景**（来自二轮 audit）：
- `floating`：REQUIRED port 漏接 —— 对应 missing_connection 错误
- `wrong_redirect`：REQUIRED port 已接，但接到了**错误的 net** —— 对应
  wrong_connection 错误，suggested_target head 应推荐把它移到哪个 net。
  **同一个 port 也可能 wrong-edge + missing 同时存在**（学生接错了一个 net
  漏掉另一个 net），各自独立成 group。

```
for ref_edge in ref.edges:
    cur_port = alignment.map_ref_port(ref_edge.src, ref, cur)
    if cur_port is None:
        stats.n_skipped_missing_component += 1; continue
    port = cur.ports[cur_port]
    if port.policy != REQUIRED: continue          # MISSING 任务仅 REQUIRED

    cur_net_correct = alignment.map_ref_net(ref_edge.dst)
    if cur_net_correct is None: continue

    cur_nets_actual = {e.dst_net_id for e in cur.edges if e.src_port_id == cur_port}
    if cur_net_correct in cur_nets_actual:
        continue                                  # 已经接对了，无需 group

    # 决定 query_origin：cur 中该 port 是否完全没接
    query_origin = "floating" if not cur_nets_actual else "wrong_redirect"

    # 候选 net 集合：cur_net_correct 必入；其余从 cur.nets 中按种子采样
    # 对 wrong_redirect 还必入 cur 上已接的错 net（让模型在"留 / 移"间选择）
    must_include = {cur_net_correct, *cur_nets_actual}
    distractors = sample_candidate_nets(
        cur, exclude=must_include,
        k=missing_edge_group_size - len(must_include), rng=rng,
    )
    candidate_nets = list(must_include) + list(distractors)

    group_id = f"miss_{cur_port}_{ref_edge.src_source_id}_{ref_edge.dst_source_id}"
    group_sample_indices = []
    correct_idx = None
    for i, net_id in enumerate(candidate_nets):
        is_correct = (net_id == cur_net_correct)
        sg = extract_seal_subgraph(
            cur, cur_port, net_id,
            edge_present=(net_id in cur_nets_actual),    # ← 当前接到的错 net 也是 edge_present=True
        )
        idx = builder.add_sample(SealSample(
            subgraph=sg,
            label=int(is_correct),
            label_source=(REF_ABSENT_REQUIRED if is_correct
                          else NEGATIVE_RANDOM).value,
            task_type=TaskType.MISSING_EDGE.value,
            candidate_edge=(cur_port, net_id),
            expected_edge=(cur_port, cur_net_correct),
            ref_edge_origin=(ref_edge.src_source_id, ref_edge.dst_source_id),
            group_id=group_id,
        ))
        if is_correct: correct_idx = i
        group_sample_indices.append(idx)
    builder.add_group(SealSampleGroup(
        group_id=group_id, task_type=TaskType.MISSING_EDGE.value,
        query_port_id=cur_port,
        query_origin=query_origin,                # "floating" | "wrong_redirect"
        sample_indices=tuple(group_sample_indices),
        correct_index=correct_idx,
    ))
```

`SealSampleGroup` 多一个 `query_origin: str` 字段：

```python
@dataclass(frozen=True)
class SealSampleGroup:
    group_id: str
    task_type: str
    query_port_id: str
    query_origin: str                             # "floating" | "wrong_redirect"
    sample_indices: tuple[int, ...]
    correct_index: int | None
```

**Step 2.5 · WRONG_OBSERVED — cur 中实际存在但 ref-sym-aware 不期望的边（强负）**

这是 **WRONG_EDGE 主头的核心负样本**。**100% 覆盖**，不留给 NEGATIVE_RANDOM
偶然采样。**必须** 在 NEGATIVE_RANDOM 之前执行，否则比例计算错位。

```
sym_aware_correct_cur_edges = compute_sym_equivalent_correct_edge_set(ref, cur, alignment)

for e in cur.edges:
    cur_port_id, cur_net_id = e.src_port_id, e.dst_net_id
    if (cur_port_id, cur_net_id) in sym_aware_correct_cur_edges:
        continue                           # 已在 Step 1 标 positive
    port = cur.ports[cur_port_id]
    if port.policy == FORBIDDEN:
        continue                           # 留给 Step 3 FORBIDDEN_VIOLATED
    if port.policy == OPTIONAL and not include_optional:
        stats.n_skipped_optional_pin += 1; continue
    # REQUIRED 但接到了错 net（最常见的"哪根线接错了"场景）
    sg = extract_seal_subgraph(cur, cur_port_id, cur_net_id, edge_present=True, ...)
    # 该 port 在 ref 中应该接到哪个 net（如有）
    expected = infer_expected_net_for_port_via_alignment(cur_port_id, ref, alignment, cur)
    builder.add_sample(SealSample(
        subgraph=sg, label=0, label_source=LabelSource.WRONG_OBSERVED.value,
        task_type=TaskType.WRONG_EDGE.value,
        candidate_edge=(cur_port_id, cur_net_id),
        expected_edge=expected,            # (cur_port_id, cur_correct_net) 或 None
    ))
```

**Step 3 · FORBIDDEN_VIOLATED — cur 中实际接的 FORBIDDEN 边 → 强负 WRONG_EDGE**
```
for port in cur.ports.values():
    if port.policy != FORBIDDEN: continue
    has_violation = False
    for e in cur.edges:
        if e.src_port_id == port.node_id:
            has_violation = True
            sg = extract(cur, e.src_port_id, e.dst_net_id, edge_present=True, ...)
            builder.add_sample(SealSample(
                subgraph=sg, label=0, label_source=FORBIDDEN_VIOLATED.value,
                task_type=TaskType.WRONG_EDGE.value,
                candidate_edge=(e.src_port_id, e.dst_net_id),
                expected_edge=None,
            ))
    if not has_violation:
        stats.n_skipped_forbidden_pin_no_violation += 1  # not really skipped, but tracked
```

**Step 4 · FORBIDDEN_NEGATIVE — 每个 FORBIDDEN pin 合成 N 条非边（决策 2）**
```
for port in cur.ports.values():
    if port.policy != FORBIDDEN: continue
    already_paired_nets = {e.dst_net_id for e in cur.edges if e.src_port_id == port.node_id}
    candidate_nets = [n for n in cur.nets if n not in already_paired_nets]
    sampled = rng.sample(candidate_nets, k=min(forbidden_negative_samples, len(candidate_nets)))
    for net_id in sampled:
        sg = extract(cur, port.node_id, net_id, edge_present=False, ...)
        builder.add_sample(SealSample(
            subgraph=sg, label=0, label_source=FORBIDDEN_NEGATIVE.value,
            task_type=TaskType.WRONG_EDGE.value,
            candidate_edge=(port.node_id, net_id),
            expected_edge=None,
        ))
```

**Step 5 · NEGATIVE_RANDOM — 凑齐 negatives_per_positive 比例，避开正集合 + 已 emit 的负**

注意：此时 WRONG_OBSERVED + FORBIDDEN_VIOLATED + FORBIDDEN_NEGATIVE 已先全部
emit，已计入 negatives 总数。Random 只补差。

```
wrong_edge_positives = sum(1 for s in builder.samples
                            if s.task_type == WRONG_EDGE and s.label == 1)
wrong_edge_negatives = sum(1 for s in builder.samples
                            if s.task_type == WRONG_EDGE and s.label == 0)
# 含 WRONG_OBSERVED / FORBIDDEN_VIOLATED / FORBIDDEN_NEGATIVE
n_need = max(0, int(wrong_edge_positives * negatives_per_positive) - wrong_edge_negatives)

candidates = [
    (p_id, n_id)
    for p_id, p in cur.ports.items() if p.policy == REQUIRED
    for n_id in cur.nets
    if (p_id, n_id) not in sym_aware_positive_edges
    and (p_id, n_id) not in builder.already_emitted_pairs(task=WRONG_EDGE)
]
rng.shuffle(candidates)
for p_id, n_id in candidates[:n_need]:
    sg = extract(cur, p_id, n_id, edge_present=(p_id, n_id) in cur_edges, ...)
    builder.add_sample(SealSample(
        subgraph=sg, label=0, label_source=LabelSource.NEGATIVE_RANDOM.value,
        task_type=TaskType.WRONG_EDGE.value,
        candidate_edge=(p_id, n_id),
        expected_edge=None,
    ))

# (Hard-negative 留 enable_hard_negative_mining=True 触发；P0.8 保持 False)
```

**重要不变量**（在 builder 的 finalize 检查并写进 stats）：

```
assert all(
    (e.src_port_id, e.dst_net_id) in sym_aware_correct_cur_edges
    or any(s.candidate_edge == (e.src_port_id, e.dst_net_id)
           and s.task_type == WRONG_EDGE and s.label == 0
           for s in builder.samples)
    for e in cur.edges
    if cur.ports[e.src_port_id].policy != OPTIONAL or include_optional
), "every observed cur edge must be either ref-correct positive or labeled negative"
```

—— 即：**cur 中每条非 OPTIONAL 边要么是正样本要么是负样本，没有漏网**。

**Finalize**：builder 把累积的 counters 冻结为 LabelStats，返回 LabelBuildResult。

### 显式不在范围（推到 P0.9 / P1）

| 范围外项 | 影响 | 缓解 |
|---|---|---|
| **Net-level swap**（DSL `symmetry_groups` 中 VCC↔VEE） | 模型把 swapped supply 当 wrong | MVP 5 电路 DSL 不使用；文档化为已知限制 |
| **Component-level swap**（R1↔R2 in divider） | 同上 | perturbation 阶段不生成此类样本 |
| **跨电路 alignment**（学生用了不同的 component 命名） | alignment 需 fuzzy match | P1 perturbation 已知 ref，直接传 explicit map |
| **noisy labels**（视觉低置信时降权） | 训练偶尔被错样本污染 | SealSample.confidence 字段已留位，P3 实际写 loss 时再决定 |

### 复用既有

- `app/domain/gnn/seal_subgraph.py:extract_seal_subgraph` —— 每个 SealSample
  内部子图都通过它生成；本次 P0.7 收尾验证的 `include_same_component_edges`
  与 policies 设计直接被 label builder 调用
- `app/domain/gnn/graph_schema.py:ConnectionPolicy` + `IC_PIN_POLICIES`
- `app/domain/gnn/hetero_circuit.py:PortNode.symmetry_class_id` /
  `ComponentNode.pin_symmetry_groups`
- `app/domain/compare/orchestrator.py:compare_logical_graphs` —— **不直接调用**，
  但 P5 评估时会用 rule comparator 输出做 label 一致性 sanity check

### 测试覆盖（≥ 30 项，按层组织）

- `test_alignment.py`（≥ 6）
  - identity_alignment 同名 ref/cur → 完整对齐 + 反向索引正确
  - identity_alignment 部分 ref id 在 cur 缺失 → notes 记录
  - alignment_from_dicts 显式覆盖
  - map_ref_port 命中 / 缺失返回 None
  - map_ref_net 命中 / 缺失
  - alignment 序列化（dict-ish）便于 P1 perturbation log

- `test_label_builder.py`（≥ 24）
  - **TaskType + 字段（4 项）**
    - 每个 SealSample 有非空 `task_type` ∈ {wrong_edge, missing_edge}
    - 每个 WRONG_EDGE sample 的 `candidate_edge` 与 `subgraph.target_*` 一致
    - REF_PRESENT 的 sample `expected_edge` 等于 candidate（自指）
    - REF_SYMMETRIC_SWAP 的 sample `is_symmetric_equivalent=True` 且
      `expected_edge` 指向原 ref 对齐结果（不是 sibling）
  - **LabelSource 一一覆盖（10 项）**
    - REF_PRESENT：完美 cur copy
    - REF_ABSENT_REQUIRED：cur 缺一条 REQUIRED 边
    - REF_SYMMETRIC_SWAP：R.pin1↔pin2 swap 双正样本
    - **WRONG_OBSERVED（关键）**：cur 把 R1.pin1 接到 GND（应接 VIN）→ 必出现一条 `label=0, source=WRONG_OBSERVED, candidate_edge=(R1.pin1, GND), expected_edge=(R1.pin1, VIN)`，**不能**依赖 NEGATIVE_RANDOM
    - **WRONG_OBSERVED 全覆盖不变量**：cur.edges 中每条非 OPTIONAL、非 ref-sym-correct 的边，**必有**对应的 WRONG_EDGE 负样本（断言 `builder.assert_observed_edges_covered()`）
    - FORBIDDEN_VIOLATED：cur 接 pin 8 → GND（与 WRONG_OBSERVED 互斥：FORBIDDEN 走 Step 3，REQUIRED 错边走 Step 2.5）
    - FORBIDDEN_NEGATIVE：pin 8 floating + forbidden_negative_samples=3 → 3 条
    - 默认 forbidden_negative_samples=4 时数量正确
    - NEGATIVE_RANDOM：全部 port.policy == REQUIRED，不命中 sym-equivalent 正集合，**不重复覆盖 WRONG_OBSERVED 已 emit 的 (port, net) 对**
    - NEGATIVE_HARD enum slot 存在但 P0.8 不生成（断言计数 == 0）
  - **SealSampleGroup（6 项）**
    - 每个 REQUIRED-floating 且 ref 期望连接的 port 产出 1 个 group，`query_origin == "floating"`
    - **wrong_redirect**：cur 把 R1.pin1 接到错误 net 时，仍产生 group 且 `query_origin == "wrong_redirect"`；候选集合必含当前错 net + 正确 net
    - 同一 port 上同时 wrong + missing 多个 ref 边 → 多个 group 各自独立
    - group.sample_indices 全部指向有效 SealSample，task_type 都是 MISSING_EDGE
    - group.correct_index 不为 None 时，对应 sample.label==1
    - missing_edge_group_size=5 → 每个 group 内 ≤ 5 samples
  - **LabelStats（5 项）**
    - by_source 与按 source filter 数量一致
    - by_task_type 与按 task filter 数量一致
    - pos_neg_ratio = n_positives / max(1, n_negatives)
    - n_skipped_missing_component 在 cur 缺 R1 时 ≥ 1
    - n_unique_ports_covered / n_unique_nets_covered 单调正确
  - **行为 / 边界（3 项）**
    - 同 seed 两次 build → samples + groups + stats 完全一致
    - OPTIONAL 默认 → 计入 n_skipped_optional_pin，不进 samples
    - extract_seal_subgraph 抛 KeyError 时 → n_skipped_extract_error 递增，不 raise

- `test_label_serialization.py`（≥ 4）
  - serialize round-trip：`build → serialize → deserialize` 与原 result 完全相等
  - schema_version 字段存在且为 "1.0"
  - serialize 输出全部可被 `json.dumps` / `json.loads` 处理（无 enum/tuple 漏网）
  - 文件级不变量校验：`len(samples) == stats.total_samples`、`stats.by_source / by_task_type` 计数自洽、group.sample_indices 都是合法 index 且 correct_index 对应 label==1

### DoD
- [ ] pytest 全绿（既有 157 + 新 ≥ 36 = ≥ 193）
- [ ] 既有 29 比较器测试零回归
- [ ] ruff / mypy clean
- [ ] README 加 P0.8 段；含 LabelSource × TaskType 矩阵表 + LabelStats 字段表 + JSON schema 示例
- [ ] 不引入 torch / torch_geometric
- [ ] 一次构造 UA741 buffer 全套 LabelBuildResult（pos+neg+groups+stats）耗时 < 80 ms
- [ ] LabelStats.by_source / by_task_type 与按 filter 的 sample 计数严格对应（防 silent drift）
- [ ] `serialize → deserialize` round-trip 等价（hash-equal）
- [ ] **MISSING_EDGE group 覆盖 floating + wrong_redirect 两种来源**
- [ ] **WRONG_OBSERVED 全覆盖不变量**：cur.edges 中每条非 OPTIONAL、非 ref-sym-correct 的边必有 WRONG_EDGE 负样本（builder.assert_observed_edges_covered() 通过）

### 时间盒
| 子任务 | 预估 |
|---|---|
| 1. ComponentAlignment + 2 个 constructor + 测试 | 2 h |
| 2. SealSample (含 task_type/candidate/expected/group_id) + SealSampleGroup（query_origin） + LabelSource + TaskType + LabelStats + LabelBuildResult + dataclass 测试 | 2 h |
| 3. _Builder 上下文 + counters | 1.5 h |
| 4. build_seal_samples **6 步**算法（Step 1 ref-positive · Step 2 MISSING_EDGE group [floating + wrong_redirect] · **Step 2.5 WRONG_OBSERVED 强负 100% 覆盖** · Step 3 FORBIDDEN_VIOLATED · Step 4 FORBIDDEN_NEGATIVE · Step 5 NEGATIVE_RANDOM 补差） | 6.5 h |
| 5. symmetric sibling 展开 + 防去重 + sym-aware 正集合 | 2 h |
| 6. NEGATIVE_HARD 占位 + hard-neg 策略文档 | 0.5 h |
| 7. JSON schema serialize / deserialize + round-trip 测试 | 2 h |
| 8. label builder 主测试 (≥ 30 含 wrong_redirect + serialization) + cur fixture 手搓 | 6 h |
| 9. ruff / mypy / README / 调试余量 | 2 h |
| **合计** | **~24 h（≈ 3 工作日）** |

完成后 P1 perturbation 只需关心"如何生成 cur HeteroCircuitGraph + 一个
ComponentAlignment"，label 全权由 label_builder 接管。两层完全解耦。

### Label 落盘 JSON Schema（dataset_builder 在 P1 写盘契约）

P0.8 同时定下 LabelBuildResult 序列化为 JSON 的最终格式 —— P1 dataset_builder
按此写文件，P2 PyG converter 按此读文件。**所有数值字段保持 plain JSON 类型**
（int / float / string / list / dict）；torch 张量化推到 P2。

文件命名约定：`datasets/circuit_compare/processed/labels/<ref_id>/<sample_id>.json`
（与 plan §五 目录结构一致）。

```json
{
  "schema_version": "1.0",
  "sample_id": "rc_lowpass__neg_pinrev_0042",
  "ref_id": "rc_lowpass",
  "cur_metadata": {
    "perturbation_chain": ["pin_reversed:C1"],
    "alignment": {
      "ref_to_cur_component": {"R1": "U_R_3", "C1": "U_C_1"},
      "ref_to_cur_net": {"VIN": "n_07", "GND": "n_03", "VOUT": "n_05"},
      "notes": {"perturbation_seed": 42}
    }
  },
  "stats": {
    "total_samples": 28,
    "n_positives": 12,
    "n_negatives": 16,
    "pos_neg_ratio": 0.75,
    "by_source": {
      "ref_present": 8,
      "ref_symmetric_swap": 2,
      "ref_absent_required": 2,
      "wrong_observed": 6,
      "forbidden_violated": 0,
      "forbidden_negative": 0,
      "negative_random": 10,
      "negative_hard": 0
    },
    "by_task_type": {"wrong_edge": 24, "missing_edge": 4},
    "n_groups": 1,
    "n_groups_without_positive": 0,
    "n_skipped_missing_component": 0,
    "n_skipped_optional_pin": 0,
    "n_skipped_forbidden_pin_no_violation": 0,
    "n_skipped_extract_error": 0,
    "n_unique_ports_covered": 4,
    "n_unique_nets_covered": 3
  },
  "samples": [
    {
      "index": 0,
      "label": 1,
      "label_source": "ref_present",
      "task_type": "wrong_edge",
      "candidate_edge": ["cur_port:U_R_3.pin1", "cur_net:n_07"],
      "expected_edge": ["cur_port:U_R_3.pin1", "cur_net:n_07"],
      "ref_edge_origin": ["R1.pin1", "VIN"],
      "confidence": 1.0,
      "is_symmetric_equivalent": false,
      "group_id": null,
      "subgraph": {
        "target_port_id": "cur_port:U_R_3.pin1",
        "target_net_id": "cur_net:n_07",
        "edge_present": true,
        "num_hops": 2,
        "port_ids": ["cur_port:U_R_3.pin1", "cur_port:U_C_1.cathode"],
        "net_ids": ["cur_net:n_07", "cur_net:n_05"],
        "edges": [["cur_port:U_C_1.cathode", "cur_net:n_05"]],
        "same_component_edges": [],
        "drnl_labels": {
          "cur_port:U_R_3.pin1": 1,
          "cur_net:n_07": 1,
          "cur_port:U_C_1.cathode": 3,
          "cur_net:n_05": 2
        },
        "is_target": {
          "cur_port:U_R_3.pin1": true,
          "cur_net:n_07": true,
          "cur_port:U_C_1.cathode": false,
          "cur_net:n_05": false
        }
      }
    }
  ],
  "groups": [
    {
      "group_id": "miss_cur_port:U_C_1.cathode_C1.cathode_GND",
      "task_type": "missing_edge",
      "query_port_id": "cur_port:U_C_1.cathode",
      "query_origin": "wrong_redirect",
      "sample_indices": [24, 25, 26, 27],
      "correct_index": 0
    }
  ]
}
```

**关键不变量**（dataset_builder 必须验证）：

- `len(samples) == stats.total_samples`
- 每个 sample 的 `index` 字段等于它在 `samples` 列表中的位置
- `groups[*].sample_indices` 都是有效 index，且这些 sample 的 `task_type ==
  group.task_type` 且 `group_id == group.group_id`
- 对每个 group：`correct_index` 不为 null 时，`samples[group.sample_indices[correct_index]].label == 1`
- `stats.by_source` / `stats.by_task_type` 计数与 samples 实际分布一致

**版本管理**：`schema_version` 字段保证 P2/P3 演进时 dataset 可识别。
breaking change 必须 bump major（"2.0"）；纯加字段可 minor（"1.1"）。

### 序列化辅助函数（P0.8 内提供）

```python
# label_builder.py
def serialize_label_build_result(
    result: LabelBuildResult,
    *,
    sample_id: str,
    ref_id: str,
    cur_metadata: dict | None = None,
) -> dict: ...

def deserialize_label_build_result(payload: dict) -> LabelBuildResult: ...
```

两者**互逆**（round-trip 在 P0.8 测试中验证），P1 dataset_builder 调
serialize 写盘，P2 PyG converter 调 deserialize 读盘。

---

---

## 附录 A.6 · P0.6 执行细则（Package port materialization + symmetry policy · 2 天）

### 触发原因（外审反馈 + 内审补充）
P0.5 把"已连接 port 的语义"补齐了，但下列 5 项仍是 SEAL **次头（suggested_target）** 和 **missing_connection 检测** 的硬阻塞：

1. UA741 的 pin 1/5/8 等未连接（NC）pin 完全不在图中 —— 候选边集合连 pin 1 都生成不了
2. `PortNode.is_floating` 永远是 False（死代码）
3. R / Pot 等元件的可互换 pin 关系（symmetric policy）从 DSL / netlist_v2 全部丢弃
4. NC（必不接） vs OPTIONAL（可不接） vs REQUIRED（必接）三态未区分
5. `PortNode.pin_number` 没有显字段，下游需要再 str→int 解析

cur 侧另有一处对称性问题：`electrical_net_id=None` 的"学生没接"pin 被静默丢弃。

### 决策（已与用户对齐）
- UA741 pin 8 (NC) → `FORBIDDEN`
- `pin_number` 现在加（P0.6）
- `NetNode.swappable_with`（DSL 顶层 net swap）推迟到 P0.7 之后做

### 改动清单

**1. `graph_schema.py`**
- 新增 `class ConnectionPolicy(str, Enum)`：`REQUIRED` / `OPTIONAL` / `FORBIDDEN`
- 新增 `class PinSpec(NamedTuple)`：`pin_key, port_type, connection_policy, symmetry_class, pin_number`
- 新增 `PACKAGE_PIN_SPECS: dict[str, list[PinSpec]]` 覆盖 9 个非 IC component type（Resistor/Capacitor/CapacitorCeramic/CapacitorElectrolytic/Wire/LED/Diode/Transistor/Potentiometer）
- 新增 `IC_PIN_POLICIES: dict[str, dict[str, ConnectionPolicy]]` 覆盖 UA741（pin 1/5 = OPTIONAL，pin 8 = FORBIDDEN，其它 REQUIRED）
- 新增 `IC_PIN_SYMMETRY: dict[str, list[list[str]]]` 覆盖 UA741（`[["1","5"]]` —— offset_null_1↔2 可互换）
- 新增 `make_ic_pin_specs(subtype) -> list[PinSpec]` 从 IC_PIN_MAPS + POLICIES + SYMMETRY 合成
- 新增 `get_expected_pin_specs(ctype, subtype) -> list[PinSpec]`：调用方统一入口
- 扩 PORT_FEAT_LAYOUT 加 6 维 → `PORT_FEAT_DIM = 50`（44 + 3 policy_one_hot + 1 has_pin_number + 1 pin_number_log + 1 symmetry_class_size_inverse）
- self-check 守住一致性

**2. `hetero_circuit.py`** —— PortNode / ComponentNode 字段扩张：
- `PortNode` 新增 `pin_number: int | None` / `connection_policy: str` / `symmetry_class_id: int`
- `ComponentNode` 新增 `pin_symmetry_groups: tuple[tuple[str, ...], ...]`

**3. `port_graph.py`**
- 边路径扫完后新增 **materialize phase**：
  - 对每个 component 查 `get_expected_pin_specs(ctype, subtype)`
  - 对未出现的 expected pin，创建 `is_floating=True` 的 PortNode，填 spec 派生字段
  - 对已存在 port，**回填** pin_number / connection_policy / symmetry_class_id（从 spec 查；spec 没有则按 default `REQUIRED` / 唯一 symmetry class 派生）
- `_payload_raw_pin_edges_cur` **不再 skip** `electrical_net_id=None` 的 pin —— 而是把它们记录为"已观测但 floating"（保留 pin_raw / pin_role 但 net_source_id=None）；materialize phase 把这些 + IC spec 缺口合并
- ComponentNode.pin_symmetry_groups 在 materialize phase 末尾根据 symmetry_class_id 分组后回填（用 dataclasses.replace 风格）

**4. `__init__.py`**：导出 `ConnectionPolicy` / `PACKAGE_PIN_SPECS` / `get_expected_pin_specs`

**5. `README.md`** 重写为 P0 / P0.5 / P0.6 三段进度 + 一张 port lifecycle 图

**6. 测试** —— `tests/domain/gnn/test_package_materialization.py`，≥ 12 项：
- UA741 fixture：n_ports = **8**（5 connected + 3 floating），pin 8 policy=FORBIDDEN，pin 1/5 policy=OPTIONAL，pin 2/3/4/6/7 policy=REQUIRED
- UA741 offset_null pin 1/5 同 symmetry_class_id（且其它 pin 各自独立）
- Resistor 手搓：2 port 同 symmetry_class_id（互换组）
- Capacitor 陶瓷：2 port 同 class
- LED / Diode / 电解 Cap：anode/cathode 不同 class（极性）
- Transistor：3 个 pin 各自独立 class
- Pot：terminal_a/b 同 class，wiper 独立
- cur 侧 netlist_v2 floating pin：构造 `electrical_net_id=None` 测试 floating port 出现
- ComponentNode.pin_symmetry_groups 与 PortNode.symmetry_class_id 自洽
- 全部 floating port `is_floating=True`，全部 connected port `is_floating=False`
- pin_number：UA741 pin "3" → port.pin_number=3；R pin "pin1" → port.pin_number=1；anode → None
- 既有 65+29 测试零回归

### 复用既有模块
- `app/domain/netlist_models.py:94` `ComponentInstance.symmetry_group` —— cur 侧 pin 互换组来源（v1 仅作为参考校验，不覆盖 spec）
- `app/domain/ic_models.py:9` `UA741_PIN_ROLES` —— 仍是 IC 模板事实源
- P0.5 的 `IC_PIN_MAPS` —— make_ic_pin_specs 直接读

### DoD
- [ ] UA741 fixture：n_ports=8（先前 5 → 8）
- [ ] pin 8 connection_policy=FORBIDDEN，pin 1/5=OPTIONAL，其它=REQUIRED
- [ ] is_floating 在 3 个 NC pin 上为 True，其它为 False
- [ ] symmetry_class_id 正确划分（R/Pot/UA741 offset_null）
- [ ] PORT_FEAT_DIM=50 且 self-check 通过
- [ ] 既有 RC fixture 仍 4 ports（R/C 都 2 pin 全连，不应新增 floating）
- [ ] 既有 29+65 测试零回归
- [ ] ruff / mypy 仍 clean
- [ ] README 更新到 P0.6
- [ ] plan §九 MVP 表插入 P0.6 行

### 时间盒
| 子任务 | 预估 |
|---|---|
| 1. ConnectionPolicy + PinSpec + PACKAGE_PIN_SPECS + IC overlay | 2 h |
| 2. PortNode / ComponentNode 字段扩 + dim self-check | 1 h |
| 3. port_graph materialize phase | 3 h |
| 4. cur 侧 floating pin 检测 | 1 h |
| 5. 测试 (≥ 12 项) | 3 h |
| 6. README + ruff/mypy + 调试余量 | 2 h |
| **合计** | **~12 h（≈ 2 个工作日）** |

完成后所有 port 节点（含 NC 与 floating）齐全、symmetric / connection policy 显式标注，SEAL P0.7 可以放心生成完整候选边集合。

---

## 十二、验证（端到端）

1. `pytest tests/domain/test_graph_compare*.py` —— 集成 GNN 后**零回归**。
2. 新增 `tests/domain/gnn/test_advisor_integration.py`：mock GNNAdvisor 返回固定 advice，断言 orchestrator 行为（seed mapping 喂入、fallback 替换、disagreement 写入）。
3. 新增 `tests/domain/gnn/test_perturbation.py`：每类扰动产出至少 1 条样本，label 校验。
4. 新增 `scripts/gnn_eval.py`：跑全部 test split，输出指标 markdown 表，加入 CI nightly。
5. 人工：从真实学生历史 netlist_v2 抽 30 条，对比 v_old / v_gnn 报告差异，邀请教研评审 4.0+。

