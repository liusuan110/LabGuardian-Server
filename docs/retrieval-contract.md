# 检索契约（训练 ≡ 部署）

**状态**：自 WP-0（2026-05-24）起为权威契约。  
**读者**：在 `app/agent/**`、`app/services/rag_service.py`、`app/services/agent_service.py`、或 `scripts/distill/**` 下新增检索调用点的任何人。

本文档定义"什么算 production agent graph 和蒸馏管线中合法的知识"。**任何不在本契约内的检索源**要么 (a) 仅供开发/admin 工具使用、要么 (b) 是 bug。

---

## 1. 合法检索源（agent 主链路 + 蒸馏）

| 通道 | 实现服务 | 机制 | 数据位置 |
|---|---|---|---|
| `teaching_scene` | `TeachingKbService` | 规则打分跨 6 个 demo 场景 | `knowledge/teaching_scenes/*.json` |
| `fault_case` (亦称 `fault_case_pack`) | `TeachingKbService` + `MragService` | error_tag / scene_id / error_code 三路键控 | `knowledge/fault_cases/<scene>/*.json` |
| `datasheet_v2` | `DatasheetKbService` | 本地 OpenVINO 嵌入 + 结构化规则 | `knowledge/datasheets/*.json` + `models/bge-small-zh-v1.5-int8-ov/` |
| `circuit_kb` | `CircuitKbService` | 结构化电路知识查找 | `knowledge/circuits/*.json` |

外加 3 类**结构化事实通道**（不算检索，但会以 evidence 形式出现）：

- `station_state` — 实时工位状态
- `error_tags` / `pipeline_snapshot` / `circuit_snapshot` — pipeline 产物
- `reference_circuit` — 课堂参考电路

---

## 2. 禁用源（agent 主链路 + 蒸馏）

下列源**禁止**在 `app/agent/**`、`app/services/rag_service.py`、`app/services/agent_service.py`、`scripts/distill/**` 出现：

| 源 | 为什么禁 | 现仍存在于 |
|---|---|---|
| `KbService`（旧 PDF/Chroma） | (1) chunk 库以 NE555/74LS74/LM324 PDF 为主，与 6 demo 拓扑的核心器件不匹配；(2) 默认 `text-embedding-3-small`（云依赖），向量空间与端侧部署不一致；(3) 蒸馏数据中会引入"wrong-scene"污染 | 仅 `app/api/v1/kb.py` 的 admin 上传端点 |
| `RagService.answer_with_kb` | `KbService.answer` 的薄封装 | WP-0 已移除 |
| `AgentService` 中的 `mode == "rag"` 分支 | 用于调度到 `answer_with_kb` | WP-0 已移除 |

`KbService` 类**保留**在仓库中（`artifacts/kb/chroma/` 旧数据也保留）以供 admin/debug 工具，但类顶部已加 deprecation header，agent 主链路 import 边界会被代码评审挡掉。

---

## 3. 训练 ≡ 部署不变量

蒸馏数据生成**必须**走与 DK-2500 板上一致的检索契约。这意味着：

- 相同的 `DatasheetKbService` 配置（模型目录 hash、tokenizer hash、fusion 权重、intent 路由 YAML hash）
- 相同的 `.npz` chunk 向量覆盖度（`knowledge/datasheets/*.json` 必须 100% 覆盖）
- 相同的 `teaching_scene` + `fault_case` JSON 内容（带版本号）
- **零回落**到禁用源

任一前置缺失，蒸馏管线必须 **fail-closed**（拒绝生成数据），不得静默降级到关键词检索或旧 `KbService`。

**状态（2026-05-24）**：

- **WP-3** ✅ 已落地：
  - `DISTILL_MODE` 配置（`app/core/config.py`）+ `datasheet_lookup_tool` 在该模式下 v2 miss 返回 `status="skipped"`（不再回落 `LOCAL_DATASHEET_FALLBACKS`）。
  - [`scripts/distill/precheck_retrieval.py`](../scripts/distill/precheck_retrieval.py) —— 起飞前校验：`DISTILL_MODE` / `DATASHEET_EMBEDDING_BACKEND="openvino"` / 模型目录三件套（`openvino_model.xml/.bin` + `tokenizer.json`）/ **实际加载 OV 模型并探活一次 encode**（v2 新增）/ 每个 datasheet JSON 都有 `.npz` / chunk 无 orphan / 向量维度统一。任一不达标 exit 1，stderr 给清晰指引。
  - 全量 `.npz` 缓存已补齐：6 个 datasheet 文档共 66 个 chunk × 512 维，覆盖度 100%。
  - **跨芯片泄漏防御（v3 — 生产契约）**：`scene_resolver.SCENE_TO_ALLOWED_DATASHEETS` 定义 6 场景 → 允许 datasheet 白名单。`datasheet_lookup_tool` 在**任何模式下**（生产 + 蒸馏）只要 `scene_id` 设置就硬过滤候选文档集，防止 UA741 turn 关键词 "555" 召回 NE555 chunk。**v2 只在 DISTILL_MODE 启用会导致 train-test distribution shift**（学生训练时只见 UA741+passive，部署时可能见 BJT/NE555），v3 升级为对称契约消除该风险。Admin/debug 走"不传 scene_id"的入口（直接调 `DatasheetKbService.search()`）。
  - **工件复现**：[`scripts/distill/fetch_artifacts.sh`](../scripts/distill/fetch_artifacts.sh) —— 新 clone 一键拉 OV 模型 + 重建所有 `.npz`。
  - 运行 `.venv/bin/python -m scripts.distill.precheck_retrieval` 即可审计。
- **WP-2** ✅ 已落地：
  - [`scripts/distill/run_inference.py`](../scripts/distill/run_inference.py) —— 唯一授权的蒸馏入口。物理隔离：只 import `app.agent.graph` / `app.agent.contracts` / `app.agent.evidence` / `app.services.scene_resolver` / `app.services.error_tag_service` / `app.core.config`，**不** import `RagService` / `KbService` / `langchain_openai` / `langchain_community.vectorstores`。
  - **启动时硬校验**：`_verify_isolation()` 扫 `sys.modules`，任何 forbidden 模块被偷偷加载（哪怕通过 transitive import）即 abort。`_FORBIDDEN_MODULES` 是黑名单常量便于扩展。
  - **Precheck 闸门**：每次运行先调 `precheck_retrieval.run_all_checks()`，11 项任一不达标 exit 1（生产 5k 训练集前没人能绕过）。`--skip-precheck` 仅供单测，CLI 默认强制走。
  - **样本强校验**：`_validate_sample` 拒绝 (a) 空 query, (b) 4 个合法 intent 之外, (c) `diagnostic` / `mixed` 缺 scene_id, (d) `scene_id` 非 6 demo 之一。不合格样本入 audit 记 skip_reason，**绝不**用空/未知 scene 喂 agent。
  - **审计输出**：每条样本 JSONL 含 `final_answer` / `tool_results` / `evidence_resolved_scene_id` / `react_iterations` / `verification_passed` / `audit.distill_mode` / `audit.run_at_iso`，便于追溯 + 论文复现。
  - **错误隔离**：单条样本 raise 不中断整个 run，记 audit `skip_reason="exception: ..."`。
  - 运行命令：`.venv/bin/python -m scripts.distill.run_inference --questions <in.jsonl> --output <out.jsonl>`

---

## 4. Intent 路由（子路由）

4 个 agent intent（`concept_tutor` / `diagnostic` / `lab_guidance` / `mixed`，见 `app/agent/contracts.py::AgentIntent`）到检索源的映射如下。

| Intent | 主源 | 条件性子路由 |
|---|---|---|
| `concept_tutor` | `teaching_scene`（当前 `scene_id`） | 问题含引脚/供电/额定值/真值表/封装时启用 `datasheet_v2`；含电路结构术语时启用 `circuit_kb` |
| `diagnostic` | `error_tags` + `fault_case`（当前 `scene_id`） + `pipeline_snapshot` | 极性/引脚/额定参数/器件特性故障时启用 `datasheet_v2` |
| `lab_guidance` | `teaching_scene.expected_measurements` + `teaching_scene.measurement_notes` | 涉及具体器件参数的操作步骤时启用 `datasheet_v2` |
| `mixed` | `diagnostic` ∪ `concept_tutor` 的主源 | 同上 |

**特意没有** `chip_lookup` 这一顶层 intent。芯片/数据手册问题是 `concept_tutor` 下的子路由（并对其他 intent 也是条件启用）。

---

## 5. 蒸馏 gold-set 锚点

按 WP-4，每条 gold-set 行向本契约的通道 ID 锚定：

```yaml
- qid: <stable_id>
  query: <str>
  intent: concept_tutor | diagnostic | lab_guidance | mixed
  scene_id_expected: <teaching_scene_id>      # 6 个 demo 场景之一
  fault_case_ids_expected: [<fault_id>, ...]
  datasheet_chunk_ids_expected: [<chunk_id>, ...]
  forbidden_ids: [<id_or_glob>, ...]          # evidence 中不得出现
  required_sources: [teaching_scene, fault_case, datasheet_v2, structured_fact]
```

`forbidden_ids` 是编码"这道题不许 pull RC 内容"或"这道题不许走 KbService"的方式。配合 **Wrong-Scene Rate** 和 **Legacy Fallback Rate** 指标（WP-5），它就是蒸馏的硬门槛。

---

## 6. 蒸馏开跑前的硬门槛

| 门槛 | 阈值 | 责任 WP |
|---|---|---|
| Wrong-Scene Rate（evidence 含 scene ≠ expected） | **= 0** | WP-1 + WP-5 |
| Legacy Fallback Rate（evidence 来自 `KbService` / OpenAI） | **= 0** | WP-0 + WP-3 |
| Datasheet `.npz` 覆盖度 | **= 100%** v2 chunks | WP-3 |
| 60 题 gold set 上 Datasheet Recall@3 | **≥ 0.85**（目标值，待 baseline 验证） | WP-5 |

---

## 7. 变更日志

| 日期 | WP | 改动 |
|---|---|---|
| 2026-05-24 | WP-0 | 把 `KbService` 从 agent 主链路、`RagService.build_context`、`agent_service.mode=="rag"` 全部移除。新增本契约。 |
| 2026-05-24 | WP-1 (v1) | 移除 `rag_service.py` / `agent/tools.py` / `agent/tool_runner.py` / `agent/nodes/react_observe.py` / `agent/nodes/vlm_explain.py` 5 处 `scene_id="exp_first_order_rc"` 硬编码默认。引入 `app/services/scene_resolver.py`，把 7 个 topology label 映射到 6 个 demo `scene_id`，未解析时 fail-open 返回 `None`。`RuntimeEvidence` 新增 first-class 字段 `current_scene_id`。 |
| 2026-05-24 | WP-1 (v2) | 二轮审查后修 4 个残余漏洞：(P1-A) `ErrorTagService` 词汇去 RC 化（`missing_rc_component → missing_required_component` 等）；(P1-B) `react_observe._dispatch_tool` 不再接受 planner 提供的 `scene_id` override；(P2-A) `RagService.build_context` 在 topology 已解析时直接 `get_scene()`，避免 cross-scene ranker 截断；(P2-B) `TeachingKbService` 和 `MragService` 的库底层默认值从 `"exp_first_order_rc"` 改为 `""`，空输入返回空，杜绝静默 RC 复活。 |
| 2026-05-24 | WP-1 (v3) | 三轮审查后修 3 个深层洞：**(P0 关键)** `PipelineService` 在同步课堂态时调用 GNN-A 拓扑分类器，把 `topology_label` 写入 station —— 没这一步 v2 契约形同虚设（scene_resolver 永远读不到 topology hint）；**(P1-A)** `TeachingKbService.search_fault_cases` / `build_knowledge_pack` 新增 `error_codes` 参数，按 `related_error_codes` 字段打分（KB 与 validator 的标准桥）；**(P1-B)** `context_pack._allowed_tools_for_*` 新增 `has_scene: bool`，无场景时不把 `fault_case_lookup_tool` 加入 allowed_tools（避免必跳过工具消耗 ReAct 槽位）。 |
| 2026-05-24 | WP-1 (v4) | v3 加的 `error_codes` 召回**没接入** production agent graph。`FaultCaseLookupInput` 无 `error_codes` 字段；`tool_runner` / `react_observe` 调用工具时只传 `error_tags`。结果：场景解析正确 + validator 有明确 code 的情况下 `fault_case_pack` 仍为空（重命名后的 scene-agnostic tags 与 KB 词汇不匹配）。最坏的契约洞 —— **无 Wrong-Scene 泄漏，但 Fault Case Recall = 0**。v4 给 `FaultCaseLookupInput` 加 `error_codes`，在两条调度路径上从 `evidence.error_codes` 注入；planner 传的 codes 被过滤到 validator 真实子集（同 scene_id override 的防御）。 |
| 2026-05-24 | WP-1 (v5) | 两个 production 配置层洞：**(P1-A safety)** 默认 `REACT_MAX_ITERATIONS=4` + `circuit_lookup` 关键词 gate + short_circuit 家族 5 工具 = 6 allowed / 4 槽位 → 危险路径上 `safety_rule_lookup_tool`（必跑）被挤掉。蒸馏样本中 15-25% 危险场景会让学生学到"答危险路径不先讲安全"的危险习惯。修：`build_context_pack_node` 按 `react_cap_auto_expand` flag 动态扩 cap 到 `max(initial, len(allowed_tools))`；caller 显式传 cap 是硬上限（不扩）。telemetry `react_cap_initial/applied` 入 graph_metrics。**(P1-B audit)** 删除文档中对未实现脚本 `scripts/distill/*` 的 vapor 引用，§3 改为显式 "WP-2/3 pending" 状态。 |
| 2026-05-24 | WP-1 (v6) | 残余 fallback 修正 + 文档中文化。**(P2)** `RagService._build_mrag_pack` 在 `mrag_service is None` 的兼容路径上 `build_knowledge_pack` 漏传 `error_codes`，补上。production DI 总是提供 MragService，所以此路径少有触发，但旧测试 / 降级运行会漏证据。**(P3)** `docs/rag-teaching-kb-design.md` 仍描述已删除的 `local_datasheet_v2 → kb_retrieval → local_fallback` 三段回退，与本契约冲突。给该文档加 deprecation banner，更新具体段落标注"已下线 (WP-0)"，避免污染论文复现说明。本文档翻译为中文以匹配团队主语言；写作规范见 [`docs/README.md`](./README.md)。 |
| 2026-05-24 | WP-3 v1 | Datasheet v2 fail-closed 落地。**配置**：新增 `settings.DISTILL_MODE`（默认 False，环境变量 `DISTILL_MODE=true` 启用）。**工具行为**：`datasheet_lookup_tool` 在 `DISTILL_MODE` 下，v2 miss 不再回落 `LOCAL_DATASHEET_FALLBACKS`（那些"保守规则"是 dev 友好兜底，但会注入端侧 runtime 不会真正产出的合成证据），改为返回 `status="skipped"` + `provider="distill_fail_closed"` + `miss_reason="datasheet_v2_miss_distill_fail_closed"`。v2 真命中时行为不变。**缓存**：补全 `ua741` + `bjt_8050` 两个 datasheet 的 `.npz`，覆盖度从 54/66 → **66/66 (100%)**，512 维统一。**Precheck 脚本** `scripts/distill/precheck_retrieval.py`：起飞前校验 6 项契约前置（DISTILL_MODE / backend / 模型目录 + 三件套文件 / .npz 文件存在 / chunk 无 orphan / 维度统一），任一不达标 exit 1。stderr 给具体修复指令（如 "Run scripts/build_datasheet_embeddings.py --documents ua741"）。9 个 e2e 测试 pin: 正常模式回落规则、DISTILL_MODE miss skip、DISTILL_MODE 真命中不受影响、4 种 precheck 失败检测、precheck 全配齐通过、所有 datasheet JSON 的 .npz 静态覆盖度断言。 |
| 2026-05-24 | WP-3 v2 | 用户审查后三个风险全部修。**(R1 工件复现)** `.npz` 和 OV 模型在 `.gitignore`，新 clone 跑不出测试 → 新增 [`scripts/distill/fetch_artifacts.sh`](../scripts/distill/fetch_artifacts.sh) 一键 HuggingFace 拉模型 + 重建所有 .npz。precheck 报错文案补"Run scripts/distill/fetch_artifacts.sh"指引。**(R2a 加载探活)** v1 precheck 只检查文件存在，不真正 load — 损坏 IR / 缺 `openvino` runtime / 错维 tokenizer 都能"静态通过"，runtime 静默回落 keyword-only，破坏训练↔部署一致。新增 `_check_embedding_backend_active`：实例化 `OpenVINOEmbeddingBackend`，调 `.is_active`，再 `encode("probe")` 一次，确认 forward pass 工作 + dim 非零。`tmp_path` 假模型测试证明能拦住"文件名对、内容是 garbage"的情况。**(R2b 跨芯片泄漏)** v1 datasheet 检索未按场景过滤，UA741 turn 问 "555 pin" 可能召回 NE555 chunk（NE555/74LS74/LM324 不在 6 demo 核心范围内）。新增 `scene_resolver.SCENE_TO_ALLOWED_DATASHEETS` 映射（RC=passive、CE/diff=BJT_8050+passive、UA741 三场景=UA741+passive），`DatasheetKbService.search(allowed_document_ids=...)` 硬过滤。`DatasheetLookupInput` 加 `scene_id` 字段，`tool_runner` / `react_observe` 都从 `evidence.current_scene_id` 注入。**仅在 `DISTILL_MODE` 下硬过滤**（dev 模式保持灵活，admin 可随时查 NE555）。10 个跨芯片参数化测试 pin: 5 非 RC 场景 × 各种 forbidden chip 都不出现在 hits 中；dev 模式不过滤。新测试总数 9 → 16。 |
| 2026-05-24 | WP-3 v3 | v2 的"DISTILL_MODE only"过滤产生了 train-test distribution shift：学生训练时只见 UA741+passive 干净证据，部署时却可能见 BJT/NE555 噪声 chunk（dev/生产无过滤）。用户审查后判定该 shift 会让蒸馏失败风险显著上升，决定改回严格 train ≡ deploy。v3 将场景白名单从 distill profile 升级为**生产检索契约**：`datasheet_lookup_tool` 只要 `scene_id` 设置就**永远硬过滤**，与 DISTILL_MODE 无关。Admin/debug 入口走"不传 scene_id"的直接 `DatasheetKbService.search()` 调用（绕过 agent graph）。10 个跨芯片参数化测试改为"两种模式都过滤"双重断言；新增 `test_datasheet_no_scene_id_keeps_full_corpus_search` 锁定 admin 入口仍可用。 |
| 2026-05-24 | WP-3 v4 | 用户审查后三个深层洞全修。**(P0 heartbeat 抹场景)** 学生客户端每 2s 发 `StationHeartbeat`（schema 不含 `topology_label` / `netlist_v2`），`ClassroomState.update_station` 之前用整个 heartbeat **替换**已有 snapshot — 结果 pipeline 写入的场景标签秒级被抹掉，部署侧 `current_scene_id` 解析为空、`fault_case` 跳过、场景白名单失效、蒸馏部署再次错位。修：`update_station` 改为 **merge** 策略，新增 `_PIPELINE_DERIVED_FIELDS` 白名单（topology_label / scene_id / netlist_v2 / comparison_report / circuit_snapshot / semantic_analysis / runtime_metadata / missing_components / match_level），heartbeat 不显式携带这些字段时保活之前的值。**(P1 蒸馏 scene_id 强校验)** `DISTILL_MODE=True` 下 `datasheet_lookup_tool` 在空 / 未知 `scene_id` 时不再静默走全 corpus 搜索，而是 `status="skipped"` + `miss_reason="distill_no_scene_id"` 或 `"distill_invalid_scene_id"`。WP-2 蒸馏 entrypoint 落地前的 defense-in-depth — 一个 typo 的 scene_id 不会污染训练数据。**(P2 miss 路径对称)** v3 只对成功路径做了 train ≡ deploy，v2 miss 路径仍非对称（蒸馏 skip，部署回落 `LOCAL_DATASHEET_FALLBACKS`）。修：scene_id 设置时**任何模式**的 v2 miss 都 skip，`provider="scene_anchored_no_fallback"` / `miss_reason="datasheet_v2_miss_scene_anchored_no_fallback"`。`scene_id` 空时（admin / no-topo concept_tutor）legacy fallback 仍触发以保 LED 等通用安全规则可用。这条规则单源 ——"场景锚定 → 无 fallback；无场景 → fallback OK" —— 比 v2 的"distill 一回事、dev 另一回事"清晰得多。新增 7 个测试覆盖三个洞各自的修复 + 边界。 |
| 2026-05-24 | WP-3 v4 (known limitation) | 工件版本仍未硬 pin：`fetch_artifacts.sh` 用 `HF_REVISION="main"`（移动 ref），`precheck` 仅校验维度与可加载性，未校验 model.xml SHA。今天蒸馏与下次部署可能下到不同 revision 的同维度模型。**风险**：不同 BGE 模型生成的向量空间不一致，蒸馏时学到的"chunk 距离感"在部署时不再准确，召回率下降但不报错。**缓解**：在 contract §3 标注此为 known gap；下一轮（与 WP-2 一起）落 model SHA pin + precheck hash verification。当前 6 chunks × 512-dim 的存量缓存与本地模型一致，正式蒸馏前如未升级则无问题。 |
| 2026-05-24 | WP-2 v1 | 蒸馏唯一授权入口落地：[`scripts/distill/run_inference.py`](../scripts/distill/run_inference.py)。**物理隔离**：模块顶部 import 名单不含 `RagService` / `KbService` / `langchain_openai` / `langchain_community.vectorstores`；启动时 `_verify_isolation()` 扫 `sys.modules`，任何 forbidden 出现即 abort。**Precheck 闸门**：默认强制走 `precheck_retrieval.run_all_checks()`，11 项任一不达标 exit 1。**样本强校验**：拒绝空 query / 非 4 合法 intent / `diagnostic`+`mixed` 缺 scene_id / scene_id 非 6 demo。**审计 JSONL 输出**：每条样本含 `agent_output.final_answer` + `tool_results` + `evidence_resolved_scene_id` + `react_iterations` + `verification_passed` + `audit.{distill_mode, run_at_iso}`。 |
| 2026-05-24 | WP-2 v1.1 | 用户审查后三个洞修。**(P0 isolation 假成立)** v1 只在 main 入口前一次性 `_verify_isolation()`，但统一 graph 中的 `vlm_explain_node` 在样本运行**后**动态 `import app.core.deps`，而 deps 顶层 `import RagService, KbService` → forbidden 模块被偷进 `sys.modules`，物理隔离形同虚设。subprocess 探针实测复现："启动检查空 → 跑样本 → 检测到 `['app.services.kb_service', 'app.services.rag_service']`"。**用户决定 VLM 不进当前项目** → 删除 `app/agent/nodes/vlm_explain.py`、从 `graph.py` 摘除节点 + edge + 重路由 `_route_after_verification` 为 2-way、清 `nodes/__init__` 导出、删 `tests/test_vlm_provider_contract.py`。`VerificationReport.needs_micro_inspection` 字段保留作 schema 兼容但不再 branch graph。新 subprocess 测试 `test_run_sample_does_not_leak_forbidden_modules_after_e2e` 跑真实 diagnostic 样本后断言 `_FORBIDDEN_MODULES & sys.modules == ∅`。**(P1 precheck 可绕过)** `--skip-precheck` CLI 参数允许任何操作员一行命令旁路检索契约。**完全移除该参数**；argparse 现在拒收 `--skip-precheck` 为未知参数（exit 2）。`run_sample()` 函数级 API 保留供单测，但加 WARNING docstring 标注 "Internal API — production 必须走 main，否则跳过 precheck + isolation guardrail"。新测试 `test_skip_precheck_flag_does_not_exist` 锁定 CLI 拒收。**(P2 singleton 污染回归集)** `tests/test_wp2_distill_entrypoint.distill_env` fixture 通过真实 graph 初始化 `_DATASHEET_KB_SINGLETON` 为 OpenVINO-bound，但 `monkeypatch` 只回滚 settings；后续 datasheet 测试切回 null backend 仍用旧 singleton → 联合回归 6 fail / 单独均过。新增 `app.agent.tools._reset_datasheet_kb_singleton()` test-only helper + fixture teardown 调用之。新测试 `test_datasheet_singleton_reset_helper_exists` pin。**回归**：1035 passed, 0 new failures, deselect 5 → 3（删 VLM 测试）。 |
