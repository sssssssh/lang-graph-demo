"""测试 InvestBot 工具集。

mock 工具是确定性 dict 数据，可直接测试。search_web 用 monkeypatch 屏蔽真实 Tavily。
"""
from common.tools import (
    get_quote,
    get_fundamentals,
    calculator,
    save_note,
    search_web,
    ALL_TOOLS,
)


# ---------- get_quote ----------

def test_get_quote_returns_known_symbol():
    res = get_quote.invoke({"symbol": "NVDA"})
    assert res["symbol"] == "NVDA"
    assert "price" in res
    assert isinstance(res["price"], (int, float))


def test_get_quote_unknown_symbol_returns_not_found():
    res = get_quote.invoke({"symbol": "ZZZZ"})
    assert "error" in res or res.get("symbol") == "ZZZZ" and res.get("price") is None


# ---------- get_fundamentals ----------

def test_get_fundamentals_returns_pe_pb():
    res = get_fundamentals.invoke({"symbol": "NVDA"})
    assert "pe" in res and "pb" in res


# ---------- calculator ----------

def test_calculator_simple_arithmetic():
    assert calculator.invoke({"expr": "1 + 2 * 3"}) == 7


def test_calculator_rejects_non_arithmetic():
    """禁止任意 Python 表达式（防注入）。"""
    res = calculator.invoke({"expr": "__import__('os').system('echo hacked')"})
    assert isinstance(res, str) and "error" in res.lower()


# ---------- save_note ----------

def test_save_note_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("INVESTBOT_NOTES_DIR", str(tmp_path))
    msg = save_note.invoke({"title": "my note", "content": "hello"})
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert "hello" in files[0].read_text()
    assert "my note" in msg  # 工具应返回提示信息


# ---------- search_web ----------

def test_search_web_uses_injected_client(monkeypatch):
    """测试时不调真实 Tavily：通过 monkeypatch 替换内部 client。"""
    fake_results = [
        {"title": "T1", "url": "https://x.com/1", "content": "snippet 1"},
        {"title": "T2", "url": "https://x.com/2", "content": "snippet 2"},
    ]

    class FakeClient:
        def search(self, query, max_results, **kwargs):
            return {"results": fake_results}

    import common.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_get_tavily_client", lambda: FakeClient())

    out = search_web.invoke({"query": "NVDA earnings"})
    assert len(out) == 2
    assert out[0]["title"] == "T1"


# ---------- ALL_TOOLS 列表 ----------

def test_all_tools_list_complete():
    names = {t.name for t in ALL_TOOLS}
    assert names == {
        "search_web",
        "get_quote",
        "get_fundamentals",
        "calculator",
        "save_note",
    }
