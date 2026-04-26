"""模块 10 smoke test：FastAPI endpoints + retry 装饰器。

用 TestClient 不启 server；mock get_app 返回 fake graph，避免调真 LLM。
"""
import sys
import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient
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


def test_health_returns_ok():
    main = _load("10-production")
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_endpoint_returns_reply(monkeypatch):
    main = _load("10-production")
    fake = FakeChatModelWithTools(responses=[AIMessage(content="夏普比率衡量风险调整后收益。")])
    fake_app = main.get_app(llm=fake)
    monkeypatch.setattr(main, "get_app", lambda llm=None: fake_app)

    client = TestClient(main.app)
    r = client.post("/chat", json={"message": "什么是夏普比率"})
    assert r.status_code == 200
    assert "夏普比率" in r.json()["reply"]


def test_retry_decorator_eventually_succeeds():
    """重试装饰器：前两次失败，第三次成功。"""
    main = _load("10-production")
    calls = {"n": 0}

    @main.retry(max_attempts=3, delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_max_attempts():
    main = _load("10-production")

    @main.retry(max_attempts=2, delay=0.01)
    def always_fail():
        raise RuntimeError("nope")

    try:
        always_fail()
    except RuntimeError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("应该抛错的")
