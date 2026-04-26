# 模块 10：Production —— FastAPI + LangSmith + 错误处理

## 1. 本模块要解决什么问题

教程跑了一路，到这里 InvestBot 已经能"想 + 算 + 查 + 写 + 流"。但它还住在 Python REPL 里。生产化要解决三件事：①**对外服务化**（HTTP / SSE）；②**可观测性**（LangSmith trace）；③**容错**（重试 / 错误处理）。

## 2. 核心思路

```
       客户端
         │
   POST /chat         ── 阻塞返回最终回复
   POST /chat/stream  ── SSE 流式推 updates
         │
         ▼
   ┌──────────────────┐
   │  FastAPI app     │
   │  retry decorator │ ← 网络抖动兜底
   │  exception → 500 │
   └────────┬─────────┘
            │
            ▼
        LangGraph app
            │
       LangSmith trace（可选，env 开关）
```

## 3. 关键技术点

| 点 | 一句话 |
|---|---|
| FastAPI + Pydantic | 标准 Python 服务化 |
| `StreamingResponse` | SSE / 任意流式响应 |
| `app.astream(..., stream_mode="updates")` | 桥接到 SSE |
| `retry` 装饰器 | 指数退避，N 次后抛 |
| LangSmith | 设 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` 即开 |
| `lifespan` / `on_event("startup")` | 服务启动时预热 graph |

## 4. 代码导读

- `retry`：最简单的指数退避装饰器；生产中可换成 `tenacity`
- `get_app`：懒加载 + 简单缓存，避免每次请求重建 graph
- `_check_langsmith`：env 检查后只 log，不强制安装
- `/chat`：阻塞 invoke，错误转 500
- `/chat/stream`：SSE 流式；每个 update 推一行；末尾推 `[DONE]`

## 5. 如何运行

```bash
# 终端 1：启 server
uv run python 10-production/main.py

# 终端 2：调用
uv run python 10-production/client_demo.py
```

LangSmith trace（可选）：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_xxx
export LANGSMITH_PROJECT=investbot
uv run python 10-production/main.py
# 打开 https://smith.langchain.com/ 看 trace
```

## 6. 常见坑

1. **`on_event` deprecate**：FastAPI 新版推 `lifespan`；本模块用 `on_event` 是为了简洁，生产建议迁移到 `@asynccontextmanager` + `FastAPI(lifespan=...)`
2. **SSE 格式严格**：每行 `data: <json>\n\n`（双换行结尾），缺一个客户端就解析不到
3. **`astream` 在同步 endpoint 里报错**：要用 `async def` 端点 + `async for`
4. **重试装饰器与幂等**：retry 只对幂等操作安全；写盘 / 转账类操作要在装饰前考虑幂等键
5. **LangSmith 配置**：除了 KEY，还要设 `LANGSMITH_PROJECT` 否则 trace 会扔到 default project
6. **graph 缓存**：`_APP` 全局缓存方便 demo，但单测时要小心 monkeypatch
7. **uvicorn 启动方式**：`if __name__ == "__main__"` + `uvicorn.run` 适合脚本启动；生产用 `uvicorn 10-production.main:app --workers 4`
8. **TestClient 走同步路径**：`/chat/stream` 是 async endpoint，TestClient 也能调；如果换成纯 async 框架（如 starlette）请用 `httpx.AsyncClient`

## 7. 小练习

1. 加 `/chat/v2` 端点：接受 `thread_id`，返回带记忆的多轮对话（结合模块 06）
2. 把 retry 替换成 `tenacity`，加 jitter + 自定义 Exception 白名单
3. 加 Prometheus 指标：每个 endpoint 的请求数、延迟分位
4. 接入 OpenTelemetry trace，端到端串到 LangSmith
