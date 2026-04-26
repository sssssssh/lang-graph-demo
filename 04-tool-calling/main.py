"""模块 04：Tool Calling

让 InvestBot 学会"主动调工具"。本模块演示工具调用三件套：
- llm.bind_tools(ALL_TOOLS)：让 LLM 知道有哪些工具可用、参数 schema 长啥样
- ToolNode(ALL_TOOLS)：负责把 LLM 选中的工具实际执行掉
- tools_condition：检查上一条 AIMessage 是否要求调工具，决定循环还是退出

为聚焦机制本身，本模块只用 get_quote + calculator 两个不依赖网络的工具。
search_web / save_note / get_fundamentals 留到 05 用 create_react_agent 时一起放进来。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote


# 本模块用的工具子集（不依赖网络，便于本地跑通）
TOOLS = [get_quote, calculator]


def make_call_model(llm_with_tools: BaseChatModel):
    """节点工厂：调 LLM（已 bind_tools），把它的回复追加到 messages。"""

    def call_model(state: InvestBotState) -> dict:
        # 把 SystemMessage 拼到最前面（不写进 state，避免重复）
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        resp = llm_with_tools.invoke(msgs)
        return {"messages": [resp]}  # add_messages 会追加

    return call_model


def build_graph(llm: BaseChatModel | None = None):
    """构造工具调用循环图。

    结构：
        START → call_model → tools_condition →
                                  ├─ "tools" → ToolNode → call_model（回环）
                                  └─ END
    """
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    g = StateGraph(InvestBotState)
    g.add_node("call_model", make_call_model(llm_with_tools))
    g.add_node("tools", ToolNode(TOOLS))   # 节点名必须叫 "tools"，与 tools_condition 的 mapping 默认一致

    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)  # 有 tool_calls → "tools"，否则 → END
    g.add_edge("tools", "call_model")  # 工具执行完，回到模型让它"看到"工具结果再决定
    return g.compile()


def run(user_text: str, llm: BaseChatModel | None = None) -> dict:
    app = build_graph(llm=llm)
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    out = run("帮我查一下 NVDA 的股价，再帮我算 925.30 * 100 等于多少。")
    print("=== 全部 messages ===")
    for m in out["messages"]:
        cls = m.__class__.__name__
        if hasattr(m, "tool_calls") and m.tool_calls:
            print(f"[{cls}] tool_calls={[(c['name'], c['args']) for c in m.tool_calls]}")
        else:
            print(f"[{cls}] {m.content[:200]}")
