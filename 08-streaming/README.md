# 模块 08：Streaming

## 1. 本模块要解决什么问题

到现在为止 invoke 都是**整体阻塞**——发请求，等所有节点跑完才拿到结果。前端体验差。LangGraph 的 `stream()` 让你拿到**中间事件**：每个节点跑完就 emit 一次，可以接到 UI / 终端做实时展示。

InvestBot 进度：终于能"边想边说"——研究过程逐步显示，而不是憋到最后才输出。

## 2. 三种 stream_mode

```
graph.stream(input, stream_mode="updates")
  → 每步 yield 该步**增量更新**：dict[node_name, partial_state]
  → 适合"我想知道每一步在做什么"

graph.stream(input, stream_mode="values")
  → 每步 yield **完整 state 快照**
  → 适合"我想监控整体 state 演变"

app.astream(input, stream_mode="messages")
  → token 级流式（要 async）；yield (token_chunk, metadata) 元组
  → 适合给前端做打字机效果
```

## 3. 关键 API

| API | 一句话 |
|---|---|
| `for chunk in app.stream(...)` | 同步 generator |
| `async for chunk in app.astream(...)` | 异步 generator（前端集成必备） |
| `stream_mode="updates"` | dict[node_name, partial_dict] |
| `stream_mode="values"` | full state snapshot |
| `stream_mode="messages"` | (token, meta) 二元组，token 级 |
| `stream_mode=["updates", "values"]` | 多模式同时 emit（每个 chunk 第一项是 mode 名） |

## 4. 代码导读

- `stream_updates`：把 stream 的 generator 全部 collect 成 list，便于打印或测试
- `stream_values`：同上但用 values 模式，每步是完整快照
- 主函数：分两段演示，并在末尾留一段 async messages 的伪代码

## 5. 如何运行

```bash
uv run python 08-streaming/main.py
```

观察 updates 与 values 的差异——前者每步 dict 只有一个 key（节点名），后者是完整 state。

## 6. 常见坑

1. **stream 与 invoke 的关系**：`invoke = list(stream(...))` 的最后一项 + 同步等待；二者底层一致
2. **token 模式必须 async**：sync `stream(stream_mode="messages")` 不会按 token 切片；要 token 级请用 `astream`
3. **多 mode 时形态变化**：`stream_mode=["values", "updates"]` 时每个 chunk 是 `(mode, payload)` 元组，遍历时要解包
4. **流式的反压**：如果消费方慢，generator 会自然阻塞节点继续执行；这是好事但要意识到
5. **stream 不开 checkpointer 也能用**：流式只是"中间事件 emit"，与 persistence 是正交的两件事
6. **错误处理**：节点抛异常时 stream 会 raise；在 try/except 外包一层

## 7. 小练习

1. 写一个 async demo 用 `astream(stream_mode="messages")`，把 LLM 输出按 token 打印到终端
2. 实现 `stream_mode=["updates", "values"]` 的双模式，看每个 chunk 的形态
3. 在 stream 过程中按 Ctrl+C 中断，再用 thread_id + Command(resume) 续跑（要配 checkpointer）
