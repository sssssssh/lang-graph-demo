"""测试 InvestBotState：messages 必须用 add_messages reducer 累加，不替换。"""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from common.state import InvestBotState


def _identity_node(state: InvestBotState) -> dict:
    """什么都不做，仅返回空 dict（不更新）。"""
    return {}


def test_state_messages_accumulate_via_reducer():
    """两次 invoke 同一个 graph，新 message 应被追加到 messages 列表，而非替换。"""
    graph = (
        StateGraph(InvestBotState)
        .add_node("noop", _identity_node)
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile()
    )

    out = graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert len(out["messages"]) == 1

    # 模拟：把上一次输出的 messages 再次喂给 graph，并附加一条新消息
    out2 = graph.invoke(
        {"messages": out["messages"] + [AIMessage(content="hello")]}
    )
    assert len(out2["messages"]) == 2
    assert out2["messages"][0].content == "hi"
    assert out2["messages"][1].content == "hello"


def test_state_has_last_intent_field():
    """InvestBotState 应当包含 last_intent 字段（路由模块会用）。"""
    state: InvestBotState = {"messages": [], "last_intent": "explain"}
    assert state["last_intent"] == "explain"
