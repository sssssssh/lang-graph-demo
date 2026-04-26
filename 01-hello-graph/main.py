"""模块 01：Hello Graph

最小可运行的 LangGraph：单节点图，把用户输入原样回显（不调用 LLM）。
学习重点：State 是什么、Node 是什么、START / END 是什么、compile() 与 invoke() 的区别。
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. 定义 State：图在执行过程中流动的"数据快照"
class HelloState(TypedDict):
    user_input: str
    reply: str


# 2. 定义 Node：一个普通 Python 函数，接收 state，返回 state 的部分更新
def echo_node(state: HelloState) -> dict:
    user = state["user_input"]
    # 注意：返回的 dict 只是"对 state 的局部更新"，LangGraph 会合并到完整 state
    return {"reply": f"InvestBot 收到你的提问：{user!r}"}


# 3. 组装图：StateGraph(State) → add_node → add_edge → compile
def build_graph():
    graph = StateGraph(HelloState)
    graph.add_node("echo", echo_node)
    graph.add_edge(START, "echo")  # START 是入口"虚拟节点"
    graph.add_edge("echo", END)    # END 是出口"虚拟节点"
    return graph.compile()         # compile 后才能 invoke


def run(user_input: str) -> dict:
    """暴露给 smoke test 的入口函数。"""
    app = build_graph()
    return app.invoke({"user_input": user_input, "reply": ""})


if __name__ == "__main__":
    out = run("什么是夏普比率？")
    print("=== 最终 state ===")
    print(out)
