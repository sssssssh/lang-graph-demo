"""模块 05 smoke test：验证 high / low 两种构建方式行为等价。"""
import sys
import importlib.util
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class FakeChatModelWithTools(FakeMessagesListChatModel):
    """补 bind_tools 空实现（fake responses 已预制好，无需真注入 schema）。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scripted_responses():
    """预制：先调 get_quote，再给最终答复。两个 mode 共用同一脚本。"""
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "c1"}],
        ),
        AIMessage(content="AAPL 当前 188.14 美元。"),
    ]


def test_high_level_runs_react_loop():
    main = _load("05-react-loop")
    fake = FakeChatModelWithTools(responses=_scripted_responses())
    out = main.run("AAPL 多少钱", mode="high", llm=fake)
    classes = [m.__class__.__name__ for m in out["messages"]]
    assert "ToolMessage" in classes
    assert out["messages"][-1].content == "AAPL 当前 188.14 美元。"


def test_low_level_runs_react_loop():
    main = _load("05-react-loop")
    fake = FakeChatModelWithTools(responses=_scripted_responses())
    out = main.run("AAPL 多少钱", mode="low", llm=fake)
    classes = [m.__class__.__name__ for m in out["messages"]]
    assert "ToolMessage" in classes
    assert out["messages"][-1].content == "AAPL 当前 188.14 美元。"


def test_high_and_low_produce_same_message_classes():
    """两个版本的 messages 序列在结构上应当一致。"""
    main = _load("05-react-loop")
    out_h = main.run(
        "AAPL 多少钱",
        mode="high",
        llm=FakeChatModelWithTools(responses=_scripted_responses()),
    )
    out_l = main.run(
        "AAPL 多少钱",
        mode="low",
        llm=FakeChatModelWithTools(responses=_scripted_responses()),
    )
    cls_h = [m.__class__.__name__ for m in out_h["messages"]]
    cls_l = [m.__class__.__name__ for m in out_l["messages"]]
    assert cls_h == cls_l
