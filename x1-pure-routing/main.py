"""独立小例 x1：纯路由 + 循环

这个图"猜数字"——每次给当前 guess 加上 step，直到 guess >= target，然后退出。
完全不调 LLM，专注演示：
- add_conditional_edges：根据 state 选择下一个节点
- 循环边：节点指回自己（或前节点）形成环
- recursion_limit：防失控
"""
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END


class GuessState(TypedDict):
    target: int
    guess: int
    step: int
    log: list[str]


def increment(state: GuessState) -> dict:
    """节点：每次把 guess 加上 step，并记录一条 log。"""
    new_guess = state["guess"] + state["step"]
    log_line = f"guess: {state['guess']} -> {new_guess}"
    # 注意：log 没挂 reducer，所以我们手动拼接 old + new
    return {"guess": new_guess, "log": state["log"] + [log_line]}


def route(state: GuessState) -> Literal["increment", "__end__"]:
    """条件路由：guess 到了 target 就出图，否则回去再加一次。"""
    if state["guess"] >= state["target"]:
        return "__end__"
    return "increment"


def build_graph():
    g = StateGraph(GuessState)
    g.add_node("increment", increment)
    g.add_edge(START, "increment")
    # 关键：从 increment 节点出来后，根据 route() 返回值决定去哪
    g.add_conditional_edges("increment", route, {"increment": "increment", "__end__": END})
    return g.compile()


def run(target: int, step: int) -> dict:
    app = build_graph()
    # recursion_limit 防失控：默认 25，演示时手动调大一点更直观
    return app.invoke(
        {"target": target, "guess": 0, "step": step, "log": []},
        config={"recursion_limit": 50},
    )


if __name__ == "__main__":
    out = run(target=10, step=3)
    print(f"final guess = {out['guess']}, took {len(out['log'])} steps")
    for line in out["log"]:
        print(" ", line)
