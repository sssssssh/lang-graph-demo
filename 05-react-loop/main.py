"""模块 05：ReAct Loop —— 高阶封装与低阶对比

create_react_agent(model, tools, prompt=...) 一行搭出 ReAct agent。
本文件同时给出"手写低阶版"（与 04 同结构）作为对比，让你明白封装内部到底做了什么。

为充分演示，本模块挂上 ALL_TOOLS（5 个工具，含 Tavily 真实联网）。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import ALL_TOOLS


# ====== 高阶版：create_react_agent ======

def build_high_level(llm: BaseChatModel | None = None):
    """一行版本。

    create_react_agent 内部做了：
    1. llm.bind_tools(tools)
    2. 节点 "agent"（即 04 的 call_model）
    3. 节点 "tools" = ToolNode(tools)
    4. add_conditional_edges("agent", tools_condition)
    5. add_edge("tools", "agent") 形成循环
    几乎就是 04 的图，prompt 参数自动拼到对话最前面。
    """
    if llm is None:
        llm = get_llm(temperature=0)
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_BASE)


# ====== 低阶版：手写（与 04 等价，只是工具集换成 ALL_TOOLS） ======

def build_low_level(llm: BaseChatModel | None = None):
    """手写版——与 build_high_level 行为一致，让你看清内部结构。"""
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def call_model(state: InvestBotState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    g = StateGraph(InvestBotState)
    g.add_node("agent", call_model)             # 对应 create_react_agent 内的 "agent" 节点
    g.add_node("tools", ToolNode(ALL_TOOLS))    # 对应 "tools" 节点
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ====== 共用入口 ======

def run(user_text: str, mode: str = "high", llm: BaseChatModel | None = None) -> dict:
    """mode='high' 用 create_react_agent；mode='low' 用手写版。两者输出 state 形态一致。"""
    if mode == "high":
        app = build_high_level(llm=llm)
    elif mode == "low":
        app = build_low_level(llm=llm)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    question = "帮我查 AAPL 的报价和基本面，然后把要点写成笔记保存。"
    for mode in ("high", "low"):
        print(f"\n========== mode = {mode} ==========")
        out = run(question, mode=mode)
        for m in out["messages"]:
            cls = m.__class__.__name__
            if hasattr(m, "tool_calls") and m.tool_calls:
                print(f"[{cls}] tool_calls={[(c['name'], c['args']) for c in m.tool_calls]}")
            else:
                content = m.content if isinstance(m.content, str) else str(m.content)
                print(f"[{cls}] {content[:200]}")
