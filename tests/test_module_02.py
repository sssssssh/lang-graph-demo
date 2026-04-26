"""模块 02 smoke test：验证 add_messages 累加、turn_count 覆盖。"""
import sys
import importlib.util
from pathlib import Path


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_messages_accumulate_across_turns():
    main = _load("02-state-and-reducer")
    history = []
    out1 = main.chat_once(history, "Q1")
    out2 = main.chat_once(out1["messages"], "Q2")

    # 第一轮：1 user + 1 ai = 2
    assert len(out1["messages"]) == 2
    # 第二轮：在前一轮基础上又加 1 user + 1 ai = 4
    assert len(out2["messages"]) == 4
    contents = [m.content for m in out2["messages"]]
    assert contents[0] == "Q1"
    assert contents[2] == "Q2"


def test_turn_count_is_replaced_not_accumulated():
    """turn_count 没有 reducer，节点返回的值会替换原 state 的值。

    给定 turn_count=100 跑一遍图，节点内部 100+1=101，节点返回 {"turn_count": 101}。
    若没有 reducer：最终 state.turn_count == 101（替换）。
    若误加了累加 reducer：最终会变成 100 + 101 = 201。
    """
    from langchain_core.messages import HumanMessage
    main = _load("02-state-and-reducer")
    app = main.build_graph()
    out = app.invoke({"messages": [HumanMessage(content="hi")], "turn_count": 100})
    assert out["turn_count"] == 101
