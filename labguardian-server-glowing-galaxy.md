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
| `port` 🆕 | `port_type`（Drain/Source/Gate/Base/Collector/Emitter/Anode/Cathode/Pin1/Pin2/VCC/GND/IO/NC/…） | 16 | one-hot |
|  | `parent_ctype`（从 component 复制下来，方便 SEAL 局部使用） | 16 | one-hot |
|  | `polarity_sensitive` | 1 | binary（该 port 是否极性关键，如 BJT base） |
|  | `is_power_port` / `is_ground_port` | 2 | binary |
|  | `is_floating`（是否连了 net，cur 侧用以标记浮空 pin） | 1 | binary |
|  | `is_reference` | 1 | binary |
|  | **总计** | **37** | |
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

**关键差别 vs 旧 schema**：pin_index / pin_role 不再是 edge 特征，而是直接进 **port 节点**特征（37 维）—— 这是 GNN-ACLP 的核心 insight：port 是一等公民。

### 设计决策（已修订）

- **MVP 改为 component + port + net 三类节点**（推翻原"只 component + net"决策）。理由：(a) GNN-ACLP 实证 port-level 在 link prediction 上 94–99% acc，远高于 component-level；(b) 让 SEAL 抽子图时距离标签 DRNL 计算在 port-level 才有意义；(c) 节点数翻倍但 5 类 MVP 电路仍 < 100 节点，PyG 完全可承受。
- **无极性两脚器件（R, C 陶瓷）** —— 两个 port 都标 `port_type=Pin_symmetric`，训练时随机 swap port_id 作为正样本。
- **极性器件（LED, Diode, 电解 Cap, BJT, IC）** —— port_type 按器件定义严格区分（Anode/Cathode, Base/Collector/Emitter, Pin1..PinN）；`polarity_sensitive=1`。
- **VCC / GND / input / output 角色**：仍在 net 节点的 `role` one-hot；port 侧若 parent_ctype = VoltageSource，则 `is_power_port` 强信号。

### 三.6 SEAL Enclosing Subgraphs（GNN-ACLP 核心借鉴）

对**每条候选边** `e=(port_u, net_v)` 构造一个独立 PyG Data：

1. **抽 2-hop enclosing subgraph** `G_e`：包含 `port_u`、`net_v` 及 2-hop 邻居（沿 port↔net 边交替走）。
2. **DRNL 节点标签**：对 `G_e` 内每个节点 `w`，计算 `d_u(w) = dist(w, port_u)`、`d_v(w) = dist(w, net_v)`，按论文公式 `l(w) = 1 + min(d_u, d_v) + (d/2) * ((d/2) + (d%2) - 1)`（其中 `d=d_u+d_v`），结果做 one-hot（labels 0..15 + overflow bucket）= 17 维。
3. **拼接特征**：节点最终特征 = DRNL[17] ⊕ 原 port/net 特征（37 或 11，按节点类型）。
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
NodeEncoder.port  [Np, 37] → [Np, 128]
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
| L1 NodeEncoder.comp / port / net | [Nc,30] / [Np,37] / [Nn,11] | 三者全部 → 128 |
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
| **P0.7 · SEAL Subgraph Pipeline**（GNN-ACLP-inspired） | 4 天 | `seal_subgraph.py` 实现 2-hop enclosing subgraph 抽取 + DRNL labeling + batched 接口 | `seal_subgraph.py` | unit test：手算 DRNL 标签与论文公式一致；性能：50 条边 < 30 ms |
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

## 十二、验证（端到端）

1. `pytest tests/domain/test_graph_compare*.py` —— 集成 GNN 后**零回归**。
2. 新增 `tests/domain/gnn/test_advisor_integration.py`：mock GNNAdvisor 返回固定 advice，断言 orchestrator 行为（seed mapping 喂入、fallback 替换、disagreement 写入）。
3. 新增 `tests/domain/gnn/test_perturbation.py`：每类扰动产出至少 1 条样本，label 校验。
4. 新增 `scripts/gnn_eval.py`：跑全部 test split，输出指标 markdown 表，加入 CI nightly。
5. 人工：从真实学生历史 netlist_v2 抽 30 条，对比 v_old / v_gnn 报告差异，邀请教研评审 4.0+。

