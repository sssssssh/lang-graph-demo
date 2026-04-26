"""模块 06 smoke test：验证 thread_id 的隔离 + 记忆累加。"""
import sys
import importlib.util
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


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


def test_same_thread_id_accumulates_messages():
    """同一 thread_id 跨两次 invoke，messages 应当累加。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="ack 1"), AIMessage(content="ack 2")]
    )
    app = main.build_graph(llm=fake, checkpointer=MemorySaver())

    out1 = main.chat(app, "first", thread_id="t1")
    out2 = main.chat(app, "second", thread_id="t1")

    assert len(out1["messages"]) == 2  # 1 user + 1 ai
    assert len(out2["messages"]) == 4  # 累加：2 user + 2 ai


def test_different_thread_ids_are_isolated():
    """不同 thread_id 互不影响。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="t1-1"), AIMessage(content="t2-1")]
    )
    app = main.build_graph(llm=fake, checkpointer=MemorySaver())

    out_t1 = main.chat(app, "in t1", thread_id="t1")
    out_t2 = main.chat(app, "in t2", thread_id="t2")

    assert len(out_t1["messages"]) == 2
    assert len(out_t2["messages"]) == 2  # 不会变成 4
    assert out_t2["messages"][0].content == "in t2"


def test_no_checkpointer_means_no_memory():
    """compile 不传 checkpointer 时，state 不会跨 invoke 持久化。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="ack 1"), AIMessage(content="ack 2")]
    )
    app = main.build_graph(llm=fake)  # checkpointer=None

    out1 = main.chat(app, "first", thread_id="t1")
    out2 = main.chat(app, "second", thread_id="t1")
    assert len(out1["messages"]) == 2
    assert len(out2["messages"]) == 2  # 没有累加


def test_sqlite_checkpointer_works_like_memory():
    """SqliteSaver 应当与 MemorySaver 行为一致。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="ack 1"), AIMessage(content="ack 2")]
    )
    with main.sqlite_checkpointer(":memory:") as saver:
        app = main.build_graph(llm=fake, checkpointer=saver)
        out1 = main.chat(app, "first", thread_id="t1")
        out2 = main.chat(app, "second", thread_id="t1")
    assert len(out2["messages"]) == 4
