# 模块 05：ReAct Loop —— 高阶封装与低阶对比

## 1. 本模块要解决什么问题

模块 04 已经手写了完整的 ReAct 循环。LangGraph 把这个套路封装成 `create_react_agent`，**一行**就能搭出来。本模块同时提供两种构建方式：

- `build_high_level()`：一行，`create_react_agent(model, tools, prompt)`
- `build_low_level()`：手写，与 04 同结构（`call_model + ToolNode + tools_condition`）

放一起对比看，你就明白"高阶 API 帮你做了什么"——既能用快捷方式，又不被黑盒卡住。

InvestBot 进度：现在它的工具集是完整 5 个（包括真 Tavily 联网搜索 + 笔记落盘），具备一个能用的研究助手雏形。

## 2. 核心概念：create_react_agent 内部做了什么

```python
agent = create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_BASE)
```

等价于：

```python
llm_with_tools = llm.bind_tools(ALL_TOOLS)

def call_model(state):
    msgs = [SystemMessage(SYSTEM_BASE)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(msgs)]}

g = StateGraph(MessagesState)
g.add_node("agent", call_model)
g.add_node("tools", ToolNode(ALL_TOOLS))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
agent = g.compile()
```

读完这段你会发现：**它就是 04 模块的图**。区别只在节点叫 `"agent"` 而不是 `"call_model"`，仅此而已。

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.prebuilt import create_react_agent` | 一行 ReAct agent 工厂 |
| `create_react_agent(model, tools, prompt=...)` | model = BaseChatModel；tools = list；prompt = str / SystemMessage / Callable |
| `agent.invoke({"messages": [...]})` | 与手写图一样调用 |
| 高阶 vs 低阶 trade-off | 高阶：少 5 行；低阶：可以在循环中插自定义节点（重写、防御、统计） |

## 4. 代码导读

- `build_high_level`：8 行（含注释）。生产代码默认用这个就够
- `build_low_level`：把"高阶帮你做的事"逐行展开。**这两个版本的图结构是同构的**，节点名稍有差异（高阶叫 `"agent"`，低阶我们也命名为 `"agent"` 让对比直接）
- `run(mode=...)`：把入口拍平，方便 main 区分 demo

## 5. 如何运行

```bash
# .env 须有 ARK_API_KEY + LLM_MODEL；若想测真 Tavily 联网，再加 TAVILY_API_KEY
uv run python 05-react-loop/main.py
```

输出会有两个 section（high / low），分别打印各自的 messages 序列，方便肉眼对比。结构应当几乎一致。

## 6. 常见坑

1. **`prompt=` 参数旧名 `messages_modifier`**：早期 LangGraph 0.x 教程里你可能见过 `messages_modifier=...`，1.x 已改 `prompt=`。本仓库用 1.1.x，认准 `prompt`
2. **prompt 类型**：可以是 str（SystemMessage 形式）、SystemMessage、Callable（动态生成）。本模块用 str
3. **state schema 不同**：`create_react_agent` 默认用内置 `AgentState`（继承 MessagesState），不是 `InvestBotState`；如果你需要 `last_intent` 这类自定义字段，要传 `state_schema=` 参数
4. **何时用低阶手写**：①循环中要插"审核"节点（如内容过滤）；②工具结果想做后处理；③要并行多 agent；④要在循环里写 metrics。否则高阶更短
5. **真 LLM + 真 Tavily**：跑 `main.py` 时可能消耗 1-2 次 LLM 调用 + 1 次 Tavily 搜索。预算需要心里有数
6. **递归保护**：`create_react_agent` 出来的 graph 仍受默认 `recursion_limit=25` 约束；invoke 时传 `config={"recursion_limit": 50}` 可调高
7. **Deprecation 提示**：跑代码会看到 `LangGraphDeprecatedSinceV10: create_react_agent has been moved to langchain.agents`。LangGraph 计划在 2.0 移除 `langgraph.prebuilt.create_react_agent`，迁移到 `from langchain.agents import create_agent`（行为基本一致）。本教程保留旧路径以匹配大量现存教程；当你升级到 LangGraph 2.x 时，把 import 换掉、签名做小修改即可

## 7. 小练习

1. 把 `prompt` 改成一个 Callable：`lambda state: [SystemMessage("...")] + state["messages"][-3:]`，让 agent 只看最近 3 条上下文
2. 在 `build_low_level` 里加一个 `validate_node`，在 `call_model → ToolNode` 之间过一道——如果 LLM 想调 `save_note` 但 title 为空，跳过工具执行
3. 给 `create_react_agent` 加 `response_format=` 让它输出强类型 JSON，对比手写版要做多少改动
4. 对比 `agent.get_graph().draw_mermaid()` 和你手写图的可视化，确认结构一致
