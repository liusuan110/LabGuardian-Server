# LabGuardian 上下文诊断对话修复计划

## 问题诊断

### 前端核心问题
1. `useAgentChat.ts` 把系统提示词、pipeline 摘要、对话历史全部拼接到 `query` 字段发送给后端，导致后端直接把这段 prompt 当成用户问题，回答中暴露了内部信息。
2. `AgentAskRequest` 类型缺少 `job_id`、`user_message`、`chat_history`、`diagnosis_context`，但后端 schema 已定义这些字段。
3. `AgentChat.tsx` 使用 `<input>` 不支持多行/Shift+Enter，没有自动滚动到底部。
4. `demoReducer` 的 `run-success` 不清空 `chatMessages`，重新运行诊断后对话会串到旧 job。
5. 错误处理直接把原始错误消息拼到 assistant 气泡中（`Agent 暂时不可用：${action.error}`）。

### 后端核心问题
1. `agent_service.py` 的 `submit` 每次都生成新的 `job_id`（`uuid.uuid4()`），不使用 `request.job_id`。
2. `_run_diagnostic_agent_job` 只使用 `request.query`（即前端拼好的大段 prompt），不使用 `request.user_message`、`request.chat_history`、`request.diagnosis_context`。
3. `build_diagnostic_template_answer` 生成模板化回答，硬拼接了错误码、证据、工具结果、修改步骤等内部信息。
4. `repair_diagnostic_answer` 会进一步把 verification_issues 等调试信息拼接到回答中。
5. 后端没有基于 `chat_history` 做上下文追问。

---

## 改造方案

### 一、前端改造（LabGuardian-Web）

#### 1. `src/types/agent.ts`
- 扩展 `AgentAskRequest`，添加：
  - `job_id?: string`
  - `user_message?: string`
  - `chat_history?: { role: string; content: string }[]`
  - `diagnosis_context?: Record<string, unknown>`
  - `locale?: string`
- `AgentJobResult` 添加 `follow_up_suggestions?: string[]` 和 `debug?: Record<string, unknown>`（向后兼容）。
- `ChatMessage` 添加 `status?: "sending" | "sent" | "error"`。

#### 2. `src/api/agent.ts`
- `askAgent` 保持 `/api/v1/angnt/ask` POST + 轮询 `/api/v1/angnt/status/{jobId}` 机制不变。
- 确保请求体使用新的字段名。

#### 3. `src/features/demo/useAgentChat.ts`（重写核心逻辑）
- 移除 `buildContextualQuery`、`summarizePipeline`、`summarizeConversation`。
- `send(prompt)` 改为：
  - 从 `state.pipelineResult` 提取 `job_id`
  - 从 `state.chatMessages` 构造 `chat_history`（过滤掉 sending/error 状态的assistant消息，只保留完整 user/assistant 对）
  - 从 `state.pipelineResult` 构造 `diagnosis_context`（包含 components、nets、risk_level、error_codes 等结构化摘要，不拼成字符串）
  - 发送 `user_message: prompt`、`query: prompt`（兼容后端）、`mode: "diagnostic_agent"`
- 第一轮自动消息（来自 `usePipelineRun` 的 `onPipelineComplete`）同样走上述逻辑。

#### 4. `src/features/demo/demoReducer.ts`
- `run-success` 时清空 `chatMessages`、`agentResult`、`agentError`，确保重新运行诊断后对话隔离。
- `agent-error` 时：
  - 设置 `agentError`（用于 UI 可能的 toast）
  - 添加一个固定友好错误消息的 assistant 气泡：`"回答生成失败，请稍后重试。"`，不暴露原始错误。
- `agent-success` 保持从 `result.result.answer` 读取答案。

#### 5. `src/features/demo/usePipelineRun.ts`
- `onPipelineComplete` 调用 `send` 时传入的 prompt 改为 `"请根据当前诊断结果给出演示用诊断解释和下一步建议。"`（仍然自动发送第一轮，但走新的结构化请求）。

#### 6. `src/components/AgentChat.tsx`（重写 UI）
- 输入框从 `<input>` 改为 `<textarea>`，支持 Shift+Enter 换行、Enter 发送。
- 添加 `chatThreadRef`，用 `useEffect` 在 `messages` 或 `status` 变化时自动滚动到底部。
- 用户消息靠右、助手消息靠左的气泡样式。
- `actions` 不再直接渲染到气泡中（避免暴露内部 action 详情），可改为不展示或仅展示 label。
- 发送时显示 loading 指示器。
- 如果没有 `pipelineResult`（`canSend=false`），提示"请先运行完整诊断"。

#### 7. `src/styles/global.css`（追加 chat 样式）
- 追加 `.chat-thread`、`.chat-message.user`、`.chat-message.assistant`、`.chat-bubble`、`.chat-input` 等样式，确保气泡清晰、输入框固定在底部。

---

### 二、后端改造（LabGuardian-Server）

#### 1. `app/schemas/angnt.py`
- 已在 schema 中定义 `user_message`、`chat_history`、`diagnosis_context`、`locale`，无需新增模型字段。
- `AngntJobResult` 追加 `follow_up_suggestions: list[str] = Field(default_factory=list)` 和 `debug: AngntChatDebug | None = None`。

#### 2. `app/api/v1/angnt.py`
- 路由保持不变，继续接收 `AngntAskRequest` 并返回 `AngntJobAcceptedResponse`，轮询返回 `AngntJobStatusResponse`。

#### 3. `app/services/agent_service.py`（重写核心逻辑）
- `submit` 方法：
  - `job_id = request.job_id or str(uuid.uuid4())`
  - 如果 `request.job_id` 提供但 classroom 中找不到对应 station，且 `request.diagnosis_context` 为空，则返回失败状态 `"当前诊断结果不存在，请先运行完整诊断"`
- `_run_job`：
  - 当 `mode == "diagnostic_agent"` 时调用 `_run_diagnostic_agent_job`
  - 其他 mode 保持原有逻辑（但使用 `request.user_message or request.query`）
- `_run_diagnostic_agent_job`：
  - `user_message = request.user_message or request.query or ""`
  - 优先从 `classroom` 获取 station 数据；若 `classroom` 中无数据，使用 `request.diagnosis_context` 作为兜底构造 `RuntimeEvidence`
  - 将 `request.chat_history` 注入到 `RuntimeEvidence.history_facts` 中（格式：`chat_history:{role}:{content}`）
  - 调用 `run_diagnostic_graph` 时传入 `user_message=user_message`
  - 从 `graph_state.final_answer` 获取答案
  - 构造 `AngntJobResult` 时：
    - `answer = graph_state.final_answer`
    - `follow_up_suggestions = _build_follow_up_suggestions(evidence, context_pack)`
    - `debug = AngntChatDebug(job_id=job_id, used_context_refs=[...])`
    - `evidence` 中不再把 `runtime_evidence`、`context_pack`、`tool_results` 的完整 payload 放进 answer（这些已存在于 `evidence` 字段，用于 debug，但 answer 字段不包含）

#### 4. `app/agent/contracts.py`
- `DiagnosticState` 追加字段：
  - `user_message: str = ""`
  - `chat_history: list[dict[str, str]] = Field(default_factory=list)`

#### 5. `app/agent/graph.py`
- `run_diagnostic_graph` 签名增加 `user_message: str = "", chat_history: list[dict[str, str]] | None = None`
- `initial` 赋值时传入 `user_message` 和 `chat_history`
- `_generate_draft_node`：调用 `build_diagnostic_template_answer` 时传入 `user_message=state.user_message`（替代原 `query` 作为用户问题理解依据）

#### 6. `app/agent/answering.py`（重写回答生成）
- 重写 `build_diagnostic_template_answer`：
  - 不再硬拼接 "问题：... 错误码：... 证据：... 工具结果：... 修改步骤：..."
  - 改为基于 `user_message` 内容理解，生成不同风格的自然语言回答：
    - **元件清单类**（用户问"有什么元件"）：基于 `evidence.netlist_v2.components` 或 `context_pack` 中的元件信息，列出识别到的元件，并简要提及潜在问题（如悬空引脚）。
    - **原因解释类**（用户问"为什么"、"悬空"）：基于 `evidence.findings` 和 `evidence.evidence_refs`，用自然语言解释原因（如"CC1 的 2 脚只映射到自身，未形成有效参考连接"）。
    - **通用诊断类**：给出简洁的诊断结论、1-2 条关键发现、下一步建议。不暴露错误码、raw evidence、tool results。
  - 安全提示保留：若 `risk_level == "danger"`，在回答末尾自然提醒断电复查。
- 重写 `repair_diagnostic_answer`：
  - 不再拼接 `error_codes`、`evidence_refs`、`verification_issues` 到回答中。
  - 仅做安全提示补充（若缺失）。
- 更新 `build_diagnostic_citations` 和 `build_diagnostic_evidence`：保持原有逻辑，但这些只进入 `result.evidence` 和 `result.citations`（前端默认不展示）。
- 新增 `_build_follow_up_suggestions(evidence, context_pack) -> list[str]`：基于诊断内容生成 2-3 个追问建议。

#### 7. `app/agent/context_pack.py`
- `build_context_pack` 接收 `user_message` 参数，用于 pushed_facts 中的 `user_query`。

#### 8. `app/agent/tool_runner.py`
- `run_diagnostic_tools` 的 `query` 参数仍保留（用于 `fault_case_lookup_tool` 和 `datasheet_lookup_tool`），但传入 `user_message`。

---

### 三、测试更新

#### `tests/test_agent_answering.py`
- 更新 `test_build_verified_diagnostic_answer_passes_danger_short_circuit`：
  - 不再断言 `"COMPONENT_SHORTED_SAME_NET" in answer`
  - 保留断言 `"R1" in answer` 和 `"断电" in answer`（自然语言回答仍应提及元件和安全提示）

#### `tests/test_agent_graph.py`
- 更新 `test_diagnostic_graph_runs_white_box_short_circuit_path`：
  - 不再断言 `"COMPONENT_SHORTED_SAME_NET" in state.final_answer`
  - 保留 `"R1" in state.final_answer` 和 `"断电" in state.final_answer`
- 更新 `test_diagnostic_graph_routes_failed_verification_to_repair`：
  - 同样移除对 `"COMPONENT_SHORTED_SAME_NET"` 的断言
  - 保留 `"断电" in state.final_answer`

#### `tests/test_agent_service_diagnostic.py`
- 更新 `test_diagnostic_agent_mode_builds_template_answer_and_verifies`：
  - 不再断言 `"COMPONENT_SHORTED_SAME_NET" in status.result.answer`
  - 保留 `"R1" in status.result.answer` 和 `"断电" in status.result.answer`
- 其他历史相关测试（`test_diagnostic_agent_uses_history_for_repeated_error` 等）保留，因为 `history_summary` 仍以自然语言形式出现在回答中。

---

## 前后端最终统一契约

### 请求格式
```json
POST /api/v1/angnt/ask
{
  "job_id": "当前诊断任务ID（从 pipelineResult.job_id 获取）",
  "station_id": "LG-DEMO-01",
  "user_message": "用户当前输入",
  "query": "同 user_message，保留兼容",
  "mode": "diagnostic_agent",
  "chat_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "diagnosis_context": {
    "job_id": "...",
    "risk_level": "...",
    "components": [...],
    "nets": [...],
    "error_codes": [...],
    "findings": [...]
  },
  "locale": "zh-CN",
  "top_k": 5
}
```

### 响应格式（轮询 /api/v1/angnt/status/{job_id}）
```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "job_id": "...",
    "station_id": "...",
    "mode": "diagnostic_agent",
    "answer": "自然语言回答",
    "follow_up_suggestions": ["为什么判断这个引脚悬空？", "我应该如何修改？"],
    "citations": [...],
    "evidence": [...],
    "actions": [...],
    "used_retrieval": false,
    "created_at": 1234567890,
    "debug": {
      "job_id": "...",
      "used_context_refs": [...]
    }
  },
  "error": null
}
```

错误响应：
```json
{
  "job_id": "...",
  "status": "failed",
  "result": null,
  "error": "当前诊断结果不存在，请先运行完整诊断"
}
```

---

## 文件修改清单

### 前端文件
1. `src/types/agent.ts`
2. `src/api/agent.ts`
3. `src/features/demo/useAgentChat.ts`
4. `src/features/demo/demoReducer.ts`
5. `src/features/demo/usePipelineRun.ts`
6. `src/components/AgentChat.tsx`
7. `src/styles/global.css`

### 后端文件
1. `app/schemas/angnt.py`
2. `app/services/agent_service.py`
3. `app/agent/contracts.py`
4. `app/agent/graph.py`
5. `app/agent/answering.py`
6. `app/agent/context_pack.py`
7. `app/agent/tool_runner.py`
8. `tests/test_agent_answering.py`
9. `tests/test_agent_graph.py`
10. `tests/test_agent_service_diagnostic.py`

---

## 验证步骤

1. 后端：`python -m pytest tests/test_agent_answering.py tests/test_agent_graph.py tests/test_agent_service_diagnostic.py`
2. 前端：`cd E:\LabGuardian-Web && npm run typecheck && npm run build`
3. 手动验证：
   - 启动前后端
   - 上传图片运行诊断
   - 在聊天框输入："这个电路图中都有什么元件"
   - 确认显示用户气泡 + 助手气泡，助手回复为自然语言，无 raw JSON / prompt / 错误码
   - 追问："为什么判断 CC1 的 2 脚悬空？"
   - 确认基于同一个 job_id 和 chat_history 回答
   - 重新运行诊断，确认对话被清空
   - 断开后端网络，发送消息，确认 UI 只显示"回答生成失败，请稍后重试。"
