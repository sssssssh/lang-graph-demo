"""InvestBot 共享 State。

跨模块约定：
- 模块 01 / 02 / x1 出于教学目的，使用各自的自定义 State（HelloState / ChatState / GuessState），
  让学员先体会"State 就是任意 TypedDict"。
- 自模块 03 起统一以 InvestBotState 为基类扩展：
      class RoutingState(InvestBotState, total=False):
          extra_field: str
  这样跨模块的"对话历史"与"路由结果"约定一致。

字段含义：
- messages：用 add_messages reducer，新 messages 会被追加而不是替换整个列表
- last_intent：路由分类结果，由 03 模块写入；可能值 "explain" / "stock" / "sector" / "fallback"
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InvestBotState(TypedDict, total=False):
    # add_messages reducer：新 messages 会被追加，而不是替换整个列表
    messages: Annotated[list[AnyMessage], add_messages]
    # 路由结果，由 03 模块的路由节点写入；可能值："explain" / "stock" / "sector" / "fallback"
    last_intent: str
