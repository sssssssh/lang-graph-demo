# 模块 02：State 与 Reducer

## 1. 本模块要解决什么问题

模块 01 中节点返回的 `reply` 字段每次都被整体覆盖；但对话场景下我们要的是 messages 列表**追加**新条目而不是覆盖。LangGraph 用 **Reducer** 解决这个问题。

InvestBot 进度：现在它能"记住对话历史"——多轮调用同一个图，messages 会累加，每轮节点只需返回**新增的那一条**。

## 2. 核心概念

```
state 字段 ─┬─ 普通字段（无 reducer）  → 节点返回的值整体替换
            └─ Annotated[T, reducer]   → 节点返回的值经 reducer 合并到旧值
```

- **Reducer**：一个二元函数 `(old, update) -> new`，决定字段如何合并
- **`add_messages`**：LangGraph 内置的列表追加 reducer，会按 message id 智能去重 / 替换
- **`MessagesState`**：内置便捷 State（只含一个 `messages` 字段并带好 reducer），等价写法

## 3. 关键 API

| API | 一句话 |
|---|---|
| `Annotated[list[AnyMessage], add_messages]` | 给字段挂上 reducer |
| `from langgraph.graph.message import add_messages` | 内置 messages reducer |
| `from langgraph.graph import MessagesState` | 等价的便捷 State 基类 |

## 4. 代码导读

- `ChatState`：`messages` 带 reducer，`turn_count` 不带，留作对比
- `reply_node`：返回 `{"messages": [一条新 AIMessage]}`——只返回新增项，reducer 负责追加
- 主函数：连续调用三轮，把上一轮的 `messages` 喂给下一轮，体会"累加"

## 5. 如何运行

```bash
uv run python 02-state-and-reducer/main.py
```

观察输出：第 3 轮时 messages 列表里已经有 6 条（3 user + 3 ai），全是按顺序累加的。

## 6. 常见坑

1. **忘记 reducer**：消息字段不挂 `add_messages` 时，每次返回都会**整体覆盖** messages，等于把历史抹掉
2. **节点返回完整 messages 列表**：返回 `{"messages": full_list}` 也能用，但容易把"新旧合并"和"自己 append"两种心智模型混着用，最后调试困难。**始终只返回新增项**
3. **`turn_count` 这种数值字段**：没 reducer 是对的（你想"覆盖"为最新值）；如果用错了 reducer 会乱
4. **MessagesState vs 自定义 State**：只有 messages 一个字段时直接用 `MessagesState` 更简洁，要加别的字段就自己写 TypedDict
5. **`chat_once` 把整段 history 灌进 input，与"节点只返回新增项"是两件事**：本例为了演示累加效果，`chat_once` 每次都用 `history + [新 HumanMessage]` 作为完整 input；`add_messages` 按 message id 智能去重所以历史不会重复累加。换句话说，"input 含完整历史"和"节点只返回新增项"是叠加的两种机制。模块 06 引入 checkpointer 之后，state 由框架自动恢复，input 里就只需要传新消息了

## 7. 小练习

1. 把 `ChatState` 替换成 `from langgraph.graph import MessagesState`，看代码哪里要改、哪里不变
2. 给 `turn_count` 也加一个 reducer：`Annotated[int, lambda old, new: (old or 0) + new]`，观察行为变化
3. 试着在节点里返回 `{"messages": []}`（空列表），看 add_messages 行为
