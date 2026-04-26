"""模块 10：Production —— FastAPI + LangSmith trace + 错误处理

把 04 模块的 ReAct agent 暴露成 HTTP service：
- POST /chat       同步返回最终回复
- POST /chat/stream  SSE 流式返回中间 updates
- GET  /health     健康检查

LangSmith：仅当环境变量 LANGSMITH_API_KEY 与 LANGSMITH_TRACING=true 同时存在时才上报；
否则静默跳过。这样开发本地不需要 LangSmith 账号也能跑。
"""
import json
import logging
import os
import time
from functools import wraps

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote

log = logging.getLogger("investbot")
logging.basicConfig(level=logging.INFO)

TOOLS = [get_quote, calculator]


# ====== 重试装饰器：网络抖动场景兜底 ======

def retry(max_attempts: int = 3, delay: float = 0.5):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    log.warning(f"{fn.__name__} 失败第 {i+1} 次: {e}")
                    time.sleep(delay * (2 ** i))
            raise last_exc

        return wrapper

    return deco


# ====== 构造 graph（懒加载，方便测试时替换 llm） ======

_APP = None


def get_app(llm=None):
    """懒加载 + 简单缓存。生产中可以接入 lifespan 启动时构建。"""
    global _APP
    if _APP is not None and llm is None:
        return _APP

    use_default_cache = llm is None
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: InvestBotState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    g = StateGraph(InvestBotState)
    g.add_node("call_model", call_model)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    compiled = g.compile()
    if use_default_cache:
        _APP = compiled
    return compiled


# ====== LangSmith 提示（不强加依赖） ======

def _check_langsmith():
    if os.environ.get("LANGSMITH_TRACING") == "true" and os.environ.get("LANGSMITH_API_KEY"):
        log.info("LangSmith trace 已开启")
    else:
        log.info("LangSmith 未开启（设 LANGSMITH_TRACING=true + LANGSMITH_API_KEY 可启用）")


# ====== FastAPI app ======

app = FastAPI(title="InvestBot", version="0.1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.on_event("startup")
def _startup():
    _check_langsmith()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@retry(max_attempts=2, delay=0.3)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        graph = get_app()
        out = graph.invoke({"messages": [HumanMessage(content=req.message)]})
        return ChatResponse(reply=out["messages"][-1].content)
    except Exception as e:  # noqa: BLE001
        log.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE：每收到一个 update 推一行 data: ..."""

    async def event_gen():
        graph = get_app()
        try:
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                stream_mode="updates",
            ):
                yield f"data: {json.dumps({k: '...' for k in chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
