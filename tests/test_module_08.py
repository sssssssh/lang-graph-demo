"""模块 08 smoke test：验证 stream_mode 产生事件序列。"""
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


def test_updates_mode_emits_per_node_events():
    main = _load("08-streaming")
    fake = FakeChatModelWithTools(responses=[AIMessage(content="ok")])
    app = main.build_graph(llm=fake)
    events = main.stream_updates(app, "hi")
    # 至少有一条来自 call_model
    node_names = []
    for ev in events:
        node_names.extend(ev.keys())
    assert "call_model" in node_names


def test_values_mode_emits_full_snapshots():
    main = _load("08-streaming")
    fake = FakeChatModelWithTools(responses=[AIMessage(content="ok")])
    app = main.build_graph(llm=fake)
    snapshots = main.stream_values(app, "hi")
    # 每个快照都含 messages 字段
    assert all("messages" in s for s in snapshots)
    # messages 数量单调不减
    counts = [len(s["messages"]) for s in snapshots]
    assert counts == sorted(counts)


def test_values_includes_tool_message_when_tool_is_called():
    main = _load("08-streaming")
    fake = FakeChatModelWithTools(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "get_quote", "args": {"symbol": "NVDA"}, "id": "c1"}]),
            AIMessage(content="NVDA 925"),
        ]
    )
    app = main.build_graph(llm=fake)
    snapshots = main.stream_values(app, "查 NVDA")
    # 末尾快照应含 ToolMessage
    last = snapshots[-1]
    classes = [m.__class__.__name__ for m in last["messages"]]
    assert "ToolMessage" in classes
