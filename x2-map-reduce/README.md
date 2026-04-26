# 独立小例 x2：Map-Reduce —— Send API

## 1. 本模块要解决什么问题

LangGraph 默认是顺序图——一个节点干完才到下一个。但有时候你需要 **并行**：比如同时查 10 只股票的报价，再汇总。`Send` API 让你在一条 conditional edge 上同时投放多份任务，框架并行执行，结果通过 reducer 自动汇总。

## 2. 核心概念

```
            START
              │
              ▼
       ┌──────────────┐
       │  fan_out     │ ← 返回 [Send("worker", {symbol:"NVDA"}), Send(..., AAPL), ...]
       └──────┬───────┘
              │ 并行调度
        ┌─────┼─────┬─────┐
        ▼     ▼     ▼     ▼
      worker worker worker worker
        │     │     │     │
        └──┬──┴──┬──┴─────┘
           ▼     ▼
       results: list (用 operator.add reducer 自动 merge)
              │
              ▼
          summary → END
```

- **`Send(node_name, arg)`**：fan-out 单元；arg 会作为该 worker 的输入 state
- **`operator.add` reducer**：list 之间相加 = concatenation；每个 worker 返回单元素 list，框架自动 concat 成完整 list
- **从 START 直接 fan-out**：用 `add_conditional_edges(START, fan_out, ["worker"])`

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.types import Send` | fan-out 单元 |
| `g.add_conditional_edges(START, fan_out_fn, [target_node_name])` | 第三个参数是可能去到的节点名列表 |
| `Annotated[list[T], operator.add]` | 列表自动累加的 reducer |

## 4. 代码导读

- `MapState`：`symbols`（输入）、`results`（每个 worker 写一项，靠 reducer concat）、`summary`（最终输出）
- `fan_out`：返回 list[Send]，每个 Send 触发一个 worker 并行执行
- `worker`：接收 Send 的 arg（`{"symbol": s}`）作为本地 state，调 `get_quote` 写 results
- `summary`：把 results 汇总成一段文字

## 5. 如何运行

```bash
uv run python x2-map-reduce/main.py
```

预期打印 5 只股票（含一只 ZZZZ 未知）的报价汇总。

## 6. 常见坑

1. **worker 的 state 不是父 state**：worker 接收的是 `Send` 的 arg；如果 worker 想读父 state 的其他字段，要在 Send arg 里显式传
2. **必须有 reducer**：`results` 字段必须挂 `operator.add` 之类的 reducer，否则后到达的 worker 会覆盖前面的
3. **顺序不保证**：worker 是并行的，results 列表里元素的顺序和 fan_out 时的顺序不一定一致
4. **conditional_edges 第三参数**：`["worker"]` 是 LangGraph 用来构图的提示——告诉它这条 edge 可能去到哪些节点；不传会报错
5. **错误处理**：某个 worker 抛异常，整个 graph 会 raise；生产中要在 worker 内部 catch，把 error 字段写进结果 list

## 7. 小练习

1. 加一个 `enrich` map 阶段：worker 之后再 fan-out 调 `get_fundamentals`，演示多级 map
2. 让 worker 调真 `search_web`，并行搜每只股票的新闻
3. 加 `top_k` 参数：summary 阶段只输出涨幅前 K 名
