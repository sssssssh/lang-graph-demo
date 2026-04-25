"""InvestBot 共享 State。

后续模块按需通过 TypedDict 继承或扩展，但 messages + last_intent 这两个字段是项目通用约定。
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InvestBotState(TypedDict, total=False):
    # add_messages reducer：新 messages 会被追加，而不是替换整个列表
    messages: Annotated[list[AnyMessage], add_messages]
    # 路由结果，由 03 模块的路由节点写入；可能值："explain" / "stock" / "sector" / "fallback"
    last_intent: str
