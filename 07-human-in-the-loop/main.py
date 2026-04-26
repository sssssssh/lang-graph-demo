"""模块 07：Human-in-the-loop

在 LLM 想保存研究笔记前 interrupt() 暂停，让人决定 approve / reject。
关键 API：
- interrupt(value)：在节点里调，会抛出 Interrupt，把 value 暴露给调用方
- Command(resume=...)：调用方决定后，再 invoke 时把决策喂回去
- HITL 必须配 checkpointer——没有 checkpoint 就没"暂停点"
"""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from common.state import InvestBotState


def propose_save_node(state: InvestBotState) -> dict:
    """模拟 LLM 决定要保存笔记，把"建议保存"写进 messages。"""
    return {"messages": [AIMessage(content="我建议把今天的研究保存为笔记《NVDA 速览》。")]}


def confirm_node(state: InvestBotState) -> Command[Literal["execute_save", "abort"]]:
    """interrupt 节点：暂停图，把"是否保存"问题抛给人。

    返回 Command(goto=...) 显式指定下一个节点（基于人类决策）。
    """
    decision = interrupt({"action": "approve_save_note", "preview": "NVDA 速览"})
    # interrupt 返回值就是调用方 Command(resume=...) 传进来的 value
    if decision == "approve":
        return Command(goto="execute_save")
    return Command(goto="abort")


def execute_save_node(state: InvestBotState) -> dict:
    return {"messages": [AIMessage(content="✅ 笔记已保存（mock）。")]}


def abort_node(state: InvestBotState) -> dict:
    return {"messages": [AIMessage(content="❌ 已取消保存。")]}


def build_graph(checkpointer=None):
    """HITL 必须挂 checkpointer 才能在 interrupt 处暂停后重启。"""
    if checkpointer is None:
        checkpointer = MemorySaver()

    g = StateGraph(InvestBotState)
    g.add_node("propose_save", propose_save_node)
    g.add_node("confirm", confirm_node)
    g.add_node("execute_save", execute_save_node)
    g.add_node("abort", abort_node)

    g.add_edge(START, "propose_save")
    g.add_edge("propose_save", "confirm")
    # confirm_node 用 Command(goto=...) 自己决定下一步，无需额外 conditional edge
    g.add_edge("execute_save", END)
    g.add_edge("abort", END)

    return g.compile(checkpointer=checkpointer)


def run_until_interrupt(app, user_text: str, thread_id: str):
    """跑到 interrupt 处暂停，返回 (result, config)。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
    return result, config


def resume(app, decision: str, config: dict):
    """用 Command(resume=...) 把人类决策喂回去，继续跑到 END。"""
    return app.invoke(Command(resume=decision), config=config)


if __name__ == "__main__":
    # === 演示 1：approve ===
    print("--- Demo 1: approve ---")
    app = build_graph()
    paused, config = run_until_interrupt(app, "今天看了 NVDA", thread_id="t1")
    print("interrupt payload:", paused.get("__interrupt__"))
    final = resume(app, "approve", config)
    print("最终 message:", final["messages"][-1].content)

    # === 演示 2：reject ===
    print("\n--- Demo 2: reject ---")
    app2 = build_graph()
    paused2, config2 = run_until_interrupt(app2, "今天看了 AAPL", thread_id="t2")
    final2 = resume(app2, "reject", config2)
    print("最终 message:", final2["messages"][-1].content)
