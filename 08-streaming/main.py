"""模块 08：Streaming

graph.stream(...) 让你拿到中间事件而不是只等最终 state。三种主要 mode：

- updates：每步只返回该步**新增的字段**（dict[node_name, partial_state]）
- values：每步返回**完整 state 快照**
- messages：token 级流式（需要 astream）—— 适合给前端做"打字机"效果

本模块演示前两种（同步），并附一段 messages 模式的注释示例（要 async）。
"""
import sys
from pathlib import Path

# 兼容按文件路径直接执行 `main.py` 时的导入路径。
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote

TOOLS = [get_quote, calculator]


def _make_call_model(llm_with_tools: BaseChatModel):
    def call_model(state: InvestBotState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    return call_model


def build_graph(llm: BaseChatModel | None = None):
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    g = StateGraph(InvestBotState)
    g.add_node("call_model", _make_call_model(llm_with_tools))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    return g.compile()


def stream_updates(app, user_text: str) -> list[dict]:
    """收集 stream_mode='updates' 产生的事件序列。"""
    events = []
    for chunk in app.stream(
        {"messages": [HumanMessage(content=user_text)]},
        stream_mode="updates",
    ):
        events.append(chunk)
    return events


def stream_values(app, user_text: str) -> list[dict]:
    """收集 stream_mode='values' 产生的快照序列。"""
    snapshots = []
    for chunk in app.stream(
        {"messages": [HumanMessage(content=user_text)]},
        stream_mode="values",
    ):
        snapshots.append(chunk)
    return snapshots


if __name__ == "__main__":
    app = build_graph()

    print("--- stream_mode='updates' ---")
    for ev in stream_updates(app, "查 NVDA 现价"):
        # ev 形如 {"call_model": {"messages": [...新增...]}}
        for node_name, partial in ev.items():
            print(f"  step '{node_name}': keys={list(partial.keys())}")

    print("\n--- stream_mode='values' ---")
    for i, snap in enumerate(stream_values(app, "查 AAPL 现价")):
        print(f"  snapshot {i}: messages_count={len(snap.get('messages', []))}")

    print("""
    # token 级流式（messages mode）需要 async：
    #     async for token, meta in app.astream(input, stream_mode="messages"):
    #         print(token.content, end="", flush=True)
    # 适合给前端 SSE / WebSocket 做打字机效果。
    """)
