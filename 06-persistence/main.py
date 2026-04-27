"""模块 06：Persistence

让 InvestBot 跨多次 invoke "记得"对话。

核心思路：compile(checkpointer=...) 让 LangGraph 在每个节点执行后自动保存 state；
再次 invoke 时，传同一个 thread_id 就会从最新 checkpoint 接着跑。

本模块演示 MemorySaver（进程内）+ SqliteSaver（落盘）两种 checkpointer。
"""
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 兼容按文件路径直接执行 `main.py` 时的导入路径。
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
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


def build_graph(llm: BaseChatModel | None = None, checkpointer=None):
    """构造带 checkpointer 的 ReAct 图。

    checkpointer 在 compile() 时挂载，决定 state 如何持久化：
    - None：每次 invoke 独立、不记忆
    - MemorySaver()：进程内字典存，重启就丢
    - SqliteSaver(...)：落盘 sqlite，跨进程可恢复
    """
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    g = StateGraph(InvestBotState)
    g.add_node("call_model", _make_call_model(llm_with_tools))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    return g.compile(checkpointer=checkpointer)


@contextmanager
def sqlite_checkpointer(db_path: str) -> Iterator[SqliteSaver]:
    """SqliteSaver 是 context manager；这里再包一层方便外部 with 调用。

    db_path 可写 ":memory:" 做临时演示，也可传文件路径做持久化。
    """
    with SqliteSaver.from_conn_string(db_path) as saver:
        yield saver


def chat(app, user_text: str, thread_id: str) -> dict:
    """对同一个 app 多次 invoke 的便捷封装：传同 thread_id 就会接着上次跑。"""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)


if __name__ == "__main__":
    # === 演示 1：MemorySaver，同 thread_id 多轮记忆 ===
    print("--- Demo 1: MemorySaver, same thread_id ---")
    app = build_graph(checkpointer=MemorySaver())
    out1 = chat(app, "我刚买了 NVDA 100 股", thread_id="demo")
    print("AI:", out1["messages"][-1].content[:80], "...")

    out2 = chat(app, "你还记得我买了什么吗？", thread_id="demo")
    print("AI:", out2["messages"][-1].content[:80], "...")
    print(f"messages 累计 = {len(out2['messages'])}\n")

    # === 演示 2：换 thread_id，记忆隔离 ===
    print("--- Demo 2: different thread_id ---")
    out3 = chat(app, "你还记得我买了什么吗？", thread_id="other")
    print("AI:", out3["messages"][-1].content[:80], "...")
    print(f"messages 累计 = {len(out3['messages'])}（仅本次的 1 user + 1 ai）\n")

    # === 演示 3：SqliteSaver，落盘可恢复 ===
    print("--- Demo 3: SqliteSaver in-memory db ---")
    with sqlite_checkpointer(":memory:") as saver:
        app2 = build_graph(checkpointer=saver)
        chat(app2, "我刚买了 AAPL 50 股", thread_id="t1")
        out = chat(app2, "你还记得我买了什么吗？", thread_id="t1")
        print("AI:", out["messages"][-1].content[:80], "...")
