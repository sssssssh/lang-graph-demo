"""测试公共 fixture。

autouse fixture 在每个测试前重置 common.tools._TAVILY_CLIENT，
避免某个测试 monkeypatch 的假 client 泄漏到下一个测试。
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_tavily_client():
    """每个测试前后都把 Tavily 全局 client 清成 None。"""
    import common.tools as tools_mod
    tools_mod._TAVILY_CLIENT = None
    yield
    tools_mod._TAVILY_CLIENT = None
