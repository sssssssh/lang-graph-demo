"""模块 04 smoke test：用 FakeMessagesListChatModel 预制工具调用序列。"""
import sys
import importlib.util
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class FakeChatModelWithTools(FakeMessagesListChatModel):
    """FakeMessagesListChatModel 不实现 bind_tools，这里补一个空操作版本。
    fake 的 responses 已经预先定好（含 tool_calls），不需要真的把 tool schema 注入。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_tool_calls_ends_immediately():
    """模型直接回答，tools_condition 应当跳到 END。"""
    main = _load("04-tool-calling")
    fake = FakeChatModelWithTools(responses=[AIMessage(content="夏普比率衡量风险调整后收益。")])
    out = main.run("什么是夏普比率？", llm=fake)
    # 只有 1 条 Human + 1 条 AI = 2，没有 ToolMessage
    assert len(out["messages"]) == 2
    assert out["messages"][-1].content.startswith("夏普比率")


def test_one_tool_call_then_final_answer():
    """模型先调 get_quote，再用结果作答。"""
    main = _load("04-tool-calling")

    # 第一轮：返回带 tool_calls 的 AIMessage（content 必须为 ""，tool_calls 触发循环）
    first = AIMessage(
        content="",
        tool_calls=[{"name": "get_quote", "args": {"symbol": "NVDA"}, "id": "call_1"}],
    )
    # 第二轮：模型看到 ToolMessage 后给出最终答复
    second = AIMessage(content="NVDA 现报 925.30 美元。（仅供参考）")

    fake = FakeChatModelWithTools(responses=[first, second])
    out = main.run("NVDA 多少钱", llm=fake)

    msgs = out["messages"]
    classes = [m.__class__.__name__ for m in msgs]
    # Human → AI(tool_call) → Tool → AI(final)
    assert classes == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    # ToolMessage 内容应当包含 NVDA 的 mock price 925.3
    assert "925.3" in msgs[2].content
    assert msgs[-1].content == "NVDA 现报 925.30 美元。（仅供参考）"


def test_calculator_tool_executes():
    """模型选 calculator，ToolNode 应当真的算出来。"""
    main = _load("04-tool-calling")
    first = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expr": "1 + 2 * 3"}, "id": "call_1"}],
    )
    second = AIMessage(content="结果是 7。")
    fake = FakeChatModelWithTools(responses=[first, second])
    out = main.run("1+2*3", llm=fake)
    tool_msg = next(m for m in out["messages"] if m.__class__.__name__ == "ToolMessage")
    assert "7" in tool_msg.content
