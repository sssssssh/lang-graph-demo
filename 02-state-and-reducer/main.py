"""模块 02：State 与 Reducer

升级 InvestBot：用 messages 列表代替单字符串，体会 add_messages reducer 的累加语义；
再加一个普通字段 turn_count 演示"不带 reducer 的字段被覆盖"的对比。
仍然不调 LLM —— 节点是确定性 Python 函数。
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 1. State：messages 字段加 reducer，turn_count 不加
class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_count: int  # 没有 reducer：每次返回该字段都会被"整体替换"


# 2. 节点：模拟"每轮回复"——不调 LLM，按规则拼一句话
def reply_node(state: ChatState) -> dict:
    last = state["messages"][-1]
    user_text = last.content if isinstance(last, HumanMessage) else "(非用户消息)"
    answer = AIMessage(content=f"我听到了：{user_text}")
    # 注意：messages 只返回"新增的那一条"——add_messages 会替我们追加进 list
    return {"messages": [answer], "turn_count": state.get("turn_count", 0) + 1}


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("reply", reply_node)
    g.add_edge(START, "reply")
    g.add_edge("reply", END)
    return g.compile()


def chat_once(history: list[AnyMessage], user_text: str) -> dict:
    """一次对话：把 user_text 追加到 history，跑一遍图，返回新 state。"""
    app = build_graph()
    return app.invoke({"messages": history + [HumanMessage(content=user_text)], "turn_count": len(history) // 2})


if __name__ == "__main__":
    history: list[AnyMessage] = []
    for q in ["什么是夏普比率？", "PE 又是什么？", "谢谢"]:
        out = chat_once(history, q)
        history = out["messages"]   # 把累计 messages 带到下一轮
        print(f"--- 第 {out['turn_count']} 轮 ---")
        for m in history:
            print(f"  [{m.__class__.__name__}] {m.content}")
