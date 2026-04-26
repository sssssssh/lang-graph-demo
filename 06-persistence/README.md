# 模块 06：Persistence

## 1. 本模块要解决什么问题

到现在为止 InvestBot 的对话是"金鱼记忆"——每次 invoke 都从空白开始。LangGraph 的 checkpointer 让 state **自动持久化**：每次节点执行完都拍一张快照，下次传同一个 thread_id 就能接着上次跑。这是 LangGraph 比"裸调 OpenAI API"最大的工程价值之一。

InvestBot 进度：终于不再失忆，能开始多轮研究对话。

## 2. 核心概念

```
compile(checkpointer=MemorySaver()) ──┐
                                       │
            invoke(input, config={"configurable": {"thread_id": "X"}})
                                       │
            ┌─────────── thread "X" 的 state 历史 ────────────┐
            │ checkpoint 1 → checkpoint 2 → checkpoint 3 ... │
            └─────────────────────────────────────────────────┘
```

- **Checkpointer**：state 的存储后端。`MemorySaver`（dict 内存）/ `SqliteSaver`（sqlite 文件）/ `PostgresSaver`（生产）
- **thread_id**：一段对话的"身份证"。同 id = 接着上次；新 id = 空白开始
- **每个节点执行后自动 save**：你不需要手动调任何 save，框架自己做

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.checkpoint.memory import MemorySaver` | 进程内 dict，重启即丢 |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | sqlite 落盘；**是 context manager**，须 `with SqliteSaver.from_conn_string(...) as saver:` |
| `graph.compile(checkpointer=saver)` | 编译时挂载 |
| `app.invoke(input, config={"configurable": {"thread_id": "X"}})` | 调用时指定是哪段对话 |
| `app.get_state(config)` | 取当前 state 快照（含 messages） |
| `app.get_state_history(config)` | 列出所有历史 checkpoint，支持时间旅行 |

## 4. 代码导读

- `build_graph(checkpointer=None)`：把 checkpointer 作为参数；不传时图是无记忆的（与 04 行为一致）
- `sqlite_checkpointer(db_path)`：把 SqliteSaver 的 context-manager 用法包成 `@contextmanager`，调用方写 `with sqlite_checkpointer(...) as saver:` 拿到普通 saver
- `chat(app, user_text, thread_id)`：构造 `config={"configurable": {"thread_id": ...}}` 的便捷封装

## 5. 如何运行

```bash
uv run python 06-persistence/main.py
```

观察三段 demo 的输出。Demo 1 应当能让 AI"想起" NVDA；Demo 2 换 thread_id 后失忆；Demo 3 用 SqliteSaver 行为一致。

## 6. 常见坑

1. **忘记传 thread_id**：会报"checkpointer 已挂载但缺 thread_id"。`config["configurable"]["thread_id"]` 是必填
2. **SqliteSaver 不是普通构造**：`SqliteSaver(":memory:")` 不工作；必须 `with SqliteSaver.from_conn_string(":memory:") as saver: ...`
3. **persistence 不等于 messages 自己累加**：累加靠的是 `add_messages` reducer；persistence 是把整份 state 跨次保存。两者配合才有"对话记忆"
4. **`thread_id` 重名风险**：不同用户用同一 thread_id 会"撞车"。生产中常见做法是 `thread_id = f"{user_id}:{conversation_id}"`
5. **MemorySaver 在多进程下失效**：每个进程一份内存——多进程要用 SqliteSaver / PostgresSaver
6. **state 太大**：每个 checkpoint 都全量保存 state；如果 messages 列表越积越长，存储与序列化都会变慢，生产中要做 trim / summary

## 7. 小练习

1. 改 sqlite 的 db_path 为真实文件如 `./checkpoint.sqlite`，跑两次 main.py，体会"重启后还能继续"
2. 调 `app.get_state(config)` 看 state 形态；调 `app.get_state_history(config)` 看历史
3. 给 `chat` 加参数 `replay_from: int`，演示从历史第 N 个 checkpoint "重放"对话
