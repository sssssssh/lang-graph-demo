"""模块 09 smoke test：subgraph + supervisor 路由。"""
import sys
import importlib.util
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class FakeChatModelWithTools(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_supervisor_routes_to_research():
    """fake LLM 第一次回 'research'，子 agent 直接回最终答复。"""
    main = _load("09-multi-agent")
    fake = FakeChatModelWithTools(
        responses=[
            AIMessage(content="research"),  # supervisor 决策
            AIMessage(content="NVDA 现报 925"),  # research 子图直接给最终答
        ]
    )
    out = main.run("NVDA 多少钱", llm=fake)
    assert out["last_intent"] == "research"
    assert "NVDA" in out["messages"][-1].content


def test_supervisor_routes_to_writer():
    main = _load("09-multi-agent")
    fake = FakeChatModelWithTools(
        responses=[
            AIMessage(content="writer"),
            AIMessage(content="已写好笔记。"),
        ]
    )
    out = main.run("把它存成笔记", llm=fake)
    assert out["last_intent"] == "writer"
    assert "笔记" in out["messages"][-1].content
