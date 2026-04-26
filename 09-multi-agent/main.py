"""模块 09：Multi-agent —— Subgraph + Supervisor

把 InvestBot 拆成两个专精子 agent：
- research_agent：能用 search_web / get_quote / get_fundamentals
- writer_agent：只能用 save_note（避免它"自己跑去查行情"）

再用一个 supervisor 节点根据用户意图路由到二者之一。
关键观察：编译后的 subgraph 可以直接作为节点 add_node("name", subgraph)。
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import get_fundamentals, get_quote, save_note, search_web


# ====== 通用：构造一个 ReAct 子图 ======

def _build_react_subgraph(llm: BaseChatModel, tools: list, name_hint: str):
    """把 04 的 ReAct 图模板化，传入工具列表即可。"""
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: InvestBotState) -> dict:
        sys_msg = SystemMessage(content=f"{SYSTEM_BASE}\n你当前的角色：{name_hint}")
        return {"messages": [llm_with_tools.invoke([sys_msg] + list(state["messages"]))]}

    g = StateGraph(InvestBotState)
    g.add_node("call_model", call_model)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    return g.compile()


# ====== Supervisor：决定下一步交给哪个子 agent ======

def make_supervisor(llm: BaseChatModel):
    """让 LLM 看用户问题，输出 'research' 或 'writer'。"""

    def supervisor(state: InvestBotState) -> Command[Literal["research", "writer", "__end__"]]:
        # 简化：取最后一条 HumanMessage 看关键词
        last_user = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_user is None:
            return Command(goto="__end__")

        prompt = [
            SystemMessage(
                content="判断下面这条用户消息属于哪类，仅输出 research / writer。\n"
                "research：查行情、看新闻、汇总基本面；\n"
                "writer：把内容整理成笔记保存。"
            ),
            HumanMessage(content=last_user.content),
        ]
        resp = llm.invoke(prompt)
        intent = (resp.content or "").strip().lower()
        if "writer" in intent:
            return Command(goto="writer", update={"last_intent": "writer"})
        return Command(goto="research", update={"last_intent": "research"})

    return supervisor


# ====== 主图：supervisor → research/writer 子图 ======

def build_graph(llm: BaseChatModel | None = None):
    if llm is None:
        llm = get_llm(temperature=0)

    research = _build_react_subgraph(
        llm, [search_web, get_quote, get_fundamentals], name_hint="研究员"
    )
    writer = _build_react_subgraph(llm, [save_note], name_hint="笔记整理员")

    g = StateGraph(InvestBotState)
    g.add_node("supervisor", make_supervisor(llm))
    g.add_node("research", research)  # 编译后的子图直接当节点用！
    g.add_node("writer", writer)
    g.add_edge(START, "supervisor")
    # supervisor 用 Command(goto=...) 自己跳，不用额外的 conditional edge
    g.add_edge("research", END)
    g.add_edge("writer", END)
    return g.compile()


def run(user_text: str, llm: BaseChatModel | None = None) -> dict:
    app = build_graph(llm=llm)
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    for q in ["NVDA 现在多少钱？", "把上面的研究整理成笔记保存"]:
        out = run(q)
        print(f"\n[Q] {q}")
        print(f"[intent] {out.get('last_intent')}")
        print(f"[final] {out['messages'][-1].content[:120]}")
