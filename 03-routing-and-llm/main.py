"""模块 03：Routing 与 LLM

第一次让 InvestBot 调用真 LLM。把用户的最新问题用 LLM 分类成 4 类，
再根据分类结果走 4 条分支，每条分支只回一段固定模板（暂不调工具，留给 04）。

学习重点：
- LLM 在节点里怎么调（同步 .invoke，传 messages 列表）
- 怎么把 LLM 的输出"提取"成结构化字段写进 state
- add_conditional_edges 怎么根据 state 字段挑分支
- InvestBotState 作为跨模块基类的扩展用法
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from common.llm import get_llm
from common.prompts import ROUTE_INSTRUCTIONS
from common.state import InvestBotState


# 1. State：直接复用 InvestBotState（含 messages + last_intent）
#    若需要扩展，写 class RoutingState(InvestBotState, total=False): ...
RoutingState = InvestBotState


# 2. 路由节点：调 LLM 让它输出 A/B/C/D 单字母
_LETTER_TO_INTENT = {"A": "explain", "B": "stock", "C": "sector", "D": "fallback"}


def make_route_node(llm: BaseChatModel):
    """工厂函数：把 llm 关进闭包，返回节点函数。

    用 closure 而不是把 llm 塞进 state，是因为 llm 不该序列化进 state（影响 checkpoint）。
    """

    def route_node(state: RoutingState) -> dict:
        # 取最后一条用户消息作为分类依据
        last_user = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_user is None:
            return {"last_intent": "fallback"}

        # 调 LLM。注意：传 SystemMessage + HumanMessage，与直接 OpenAI API 一致
        prompt = [
            SystemMessage(content=ROUTE_INSTRUCTIONS),
            HumanMessage(content=last_user.content),
        ]
        resp = llm.invoke(prompt)
        # 取首字符兜底（即使模型话多，也能拿到 A/B/C/D）
        letter = (resp.content or "").strip().upper()[:1]
        intent = _LETTER_TO_INTENT.get(letter, "fallback")
        return {"last_intent": intent}

    return route_node


# 3. 四个分支节点：暂时只回固定话术，演示路由分发
_DISCLAIMER = "（以上为研究信息汇总，仅供参考，不构成投资建议）"


def explain_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[explain 分支] 我会解释这个概念。" + _DISCLAIMER)]}


def stock_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[stock 分支] 我会查这只股票的公开信息。" + _DISCLAIMER)]}


def sector_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[sector 分支] 我会汇总这个板块的研究信息。" + _DISCLAIMER)]}


def fallback_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[fallback 分支] 这个问题超出我的范围，请换个投资相关的问题。")]}


# 4. 路由函数：从 state.last_intent 翻译成下一个节点名
def route(state: RoutingState) -> Literal["explain", "stock", "sector", "fallback"]:
    return state.get("last_intent", "fallback")  # type: ignore[return-value]


# 5. 组图
def build_graph(llm: BaseChatModel | None = None):
    """构造路由图。llm=None 时走真 get_llm()，测试时传入 FakeMessagesListChatModel。"""
    if llm is None:
        llm = get_llm(temperature=0)  # 路由要尽量确定性，温度调零

    g = StateGraph(RoutingState)
    g.add_node("route", make_route_node(llm))
    g.add_node("explain", explain_node)
    g.add_node("stock", stock_node)
    g.add_node("sector", sector_node)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        route,
        {
            "explain": "explain",
            "stock": "stock",
            "sector": "sector",
            "fallback": "fallback",
        },
    )
    for branch in ("explain", "stock", "sector", "fallback"):
        g.add_edge(branch, END)
    return g.compile()


def run(user_text: str, llm: BaseChatModel | None = None) -> dict:
    app = build_graph(llm=llm)
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    # 跑前确认 .env 已配 ARK_API_KEY + LLM_MODEL
    samples = [
        "什么是夏普比率？",          # → explain
        "NVDA 现在多少钱？",          # → stock
        "新能源板块最近怎么样？",     # → sector
        "今晚吃什么？",               # → fallback
    ]
    for q in samples:
        out = run(q)
        last_msg = out["messages"][-1].content
        print(f"Q: {q}\n  intent = {out['last_intent']}\n  reply  = {last_msg}\n")
