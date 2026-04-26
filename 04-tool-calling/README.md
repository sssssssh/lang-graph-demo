# 模块 04：Tool Calling

## 1. 本模块要解决什么问题

让 LLM **自己决定** 何时去查股价、何时去算数字。这是 ReAct 模式的核心机制——LLM 在回答途中输出"我要调 get_quote(NVDA)"，框架去执行，把结果再喂回 LLM，LLM 拿到结果继续回答。

InvestBot 进度：现在它能"动手"了。问"NVDA 多少钱"，它会自己调 `get_quote`；问"925 * 100 是多少"，它会自己调 `calculator`。

## 2. 核心概念：工具调用协议

OpenAI 兼容的 chat 模型（包括火山方舟、Doubao）都支持这套协议：

1. **客户端**告诉模型："你可以用这些工具"（json schema），靠 `bind_tools` 完成
2. **模型**返回的 `AIMessage` 中带有 `tool_calls` 字段：`[{name, args, id}]`
3. **客户端**实际执行工具，把结果包成 `ToolMessage(tool_call_id=id, content=result)` 加回对话
4. **模型**看到 `ToolMessage`，要么再调一次工具（多轮），要么给最终答复

LangGraph 把第 1、3 步分别封装为 `bind_tools` 和 `ToolNode`，第 4 步是再次调 `call_model`，循环直到模型不再返回 `tool_calls`。

```
            START
              │
              ▼
        ┌──────────┐
        │call_model│ ← LLM 回 AIMessage（可能带 tool_calls）
        └─────┬────┘
              │
        tools_condition
         ┌────┴────┐
         │         │
       "tools"  "__end__"
         │         │
         ▼         ▼
      ┌─────┐    END
      │tools│ ← ToolNode 执行所有 tool_calls
      └──┬──┘
         │
         └──→ 回 call_model（再让 LLM 看工具结果）
```

## 3. 关键 API

| API | 一句话 |
|---|---|
| `llm.bind_tools(tools)` | 把 list[BaseTool] 的 JSON schema 注入到 LLM 调用，返回新 Runnable |
| `from langgraph.prebuilt import ToolNode` | 节点：执行 state.messages[-1] 中的所有 tool_calls，返回 ToolMessage 列表 |
| `from langgraph.prebuilt import tools_condition` | 路由函数：返回 `"tools"` 或 `"__end__"`；mapping 不写时用默认 |
| `AIMessage(content="", tool_calls=[{name, args, id}])` | 测试时的"假工具调用"形态 |
| `ToolMessage(content, tool_call_id)` | 工具执行结果的消息形态（ToolNode 自动构造） |

## 4. 代码导读

- `TOOLS = [get_quote, calculator]`：本模块只用两个不依赖网络的工具
- `make_call_model(llm_with_tools)`：节点工厂——把 SYSTEM_BASE 拼前面再调 LLM，返回 AIMessage
- `g.add_node("tools", ToolNode(TOOLS))`：**节点名必须叫 "tools"**，因为 `tools_condition` 默认 mapping 把 `"tools"` 字符串映射到这个节点
- `g.add_conditional_edges("call_model", tools_condition)`：不传 mapping 用默认（`{"tools": "tools", "__end__": END}`）
- `g.add_edge("tools", "call_model")`：循环回模型——这是 ReAct "再思考"的环

## 5. 如何运行

```bash
# .env 须有 ARK_API_KEY + LLM_MODEL
uv run python 04-tool-calling/main.py
```

观察输出：你会看到 messages 列表里出现 `tool_calls` → `ToolMessage` → 最终 `AIMessage` 的完整链。这就是 LLM 在"读到"工具结果后才能给出准确价格。

## 6. 常见坑

1. **模型不调工具**：可能是 prompt 没让它意识到"该用工具"，或工具 docstring 写得太抽象。**docstring = 工具的 prompt**——`@tool` 把 docstring 暴露给模型作为描述
2. **`tool_calls` 是 list**：模型可以并行调多个工具（一次返回多个 tool_calls），ToolNode 会一次性全部执行
3. **节点名 "tools"**：写成 "tool"、"toolbox" 都不行（除非自己传 mapping）；保持默认更省事
4. **AIMessage(content="", tool_calls=[...])**：调工具时 content 通常为空（或简短解释）。测试时若忘记设 content="" 会让 tools_condition 误判
5. **死循环**：模型如果反复调同一个工具，会撞 `recursion_limit`（默认 25）。生产环境记得加上限保护或在 prompt 里说"如已查到结果，不要再重复调"
6. **bind_tools 返回的是新对象**：`llm_with_tools = llm.bind_tools(...)` 不修改 llm；忘记接返回值会导致工具未挂上
7. **ToolMessage 里的内容**：是 `str(tool_return_value)`，dict 会被序列化成 JSON 风格的 str；如果你工具返回复杂结构，模型可能看不懂——保持工具返回扁平化更安全
8. **测试时 fake_llm 不支持 bind_tools**：`FakeMessagesListChatModel` 默认 `bind_tools` 抛 `NotImplementedError`。本仓库测试里用一个子类把它重载成 `return self`（fake 的 responses 已预制好工具调用，不需要真注入 schema）

## 7. 小练习

1. 把 `search_web` 也加到 `TOOLS`，问"特斯拉最新财报"，看 LLM 是否会主动调网络搜索
2. 给 `tools_condition` 自己写一个 mapping，把 "tools" 改名成 "exec_tools"，看怎么改图才能跑通
3. 在 `call_model` 里加一行 `print(resp.tool_calls)`，可以直观看到模型每次"想干什么"
4. 故意让 `get_quote` 抛异常，看 ToolNode 的 ToolMessage 中怎么报告错误（默认会 catch 并塞 error 字段进 content）
