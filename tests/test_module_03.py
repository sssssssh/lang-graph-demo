"""模块 03 smoke test：用 FakeMessagesListChatModel 预制 LLM 响应，验证路由分流正确。"""
import sys
import importlib.util
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_llm(letter: str):
    """构造一个回 letter 字母的假 LLM。"""
    return FakeMessagesListChatModel(responses=[AIMessage(content=letter)])


def test_route_to_explain():
    main = _load("03-routing-and-llm")
    out = main.run("什么是夏普比率？", llm=_fake_llm("A"))
    assert out["last_intent"] == "explain"
    assert "[explain 分支]" in out["messages"][-1].content


def test_route_to_stock():
    main = _load("03-routing-and-llm")
    out = main.run("NVDA 现在多少钱？", llm=_fake_llm("B"))
    assert out["last_intent"] == "stock"
    assert "[stock 分支]" in out["messages"][-1].content


def test_route_to_sector():
    main = _load("03-routing-and-llm")
    out = main.run("新能源板块怎么样", llm=_fake_llm("C"))
    assert out["last_intent"] == "sector"


def test_route_to_fallback_when_unknown_letter():
    """模型若回 'Z' 这种未定义字母，应当兜底到 fallback。"""
    main = _load("03-routing-and-llm")
    out = main.run("今晚吃什么", llm=_fake_llm("Z"))
    assert out["last_intent"] == "fallback"


def test_route_extracts_first_letter_when_model_verbose():
    """模型可能会输出 'A. explain — ...'，节点应只取首字母。"""
    main = _load("03-routing-and-llm")
    out = main.run("什么是 PE", llm=_fake_llm("A. explain — 解释概念"))
    assert out["last_intent"] == "explain"
