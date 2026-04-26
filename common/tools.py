"""InvestBot 的工具集。

- search_web：Tavily 真实联网（自模块 04 起使用，工厂函数 _get_tavily_client 便于测试 mock）
- get_quote / get_fundamentals：mock 数据，结构化返回
- calculator：受限算术求值（防注入）
- save_note：本地落盘成 markdown
"""
from __future__ import annotations

import ast
import operator as op
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


# ====== 1. search_web（Tavily） ======

_TAVILY_CLIENT = None


def _get_tavily_client():
    """惰性构造 Tavily 客户端。测试时通过 monkeypatch 替换本函数。"""
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY 未设置（自模块 04 起需要）。")
        _TAVILY_CLIENT = TavilyClient(api_key=api_key)
    return _TAVILY_CLIENT


@tool
def search_web(query: str) -> list[dict]:
    """在网络上搜索最近的新闻、研报、公告。
    返回最多 5 条 {title, url, content} 字典列表。
    """
    client = _get_tavily_client()
    raw = client.search(query, max_results=5, search_depth="basic")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in raw.get("results", [])
    ]


# ====== 2. get_quote（mock） ======

_QUOTE_DB = {
    "NVDA": {"symbol": "NVDA", "price": 925.30, "change_pct": 1.85, "high_52w": 974.0, "low_52w": 394.5},
    "AAPL": {"symbol": "AAPL", "price": 188.14, "change_pct": -0.42, "high_52w": 199.6, "low_52w": 164.1},
    "TSLA": {"symbol": "TSLA", "price": 174.20, "change_pct": 2.31, "high_52w": 299.3, "low_52w": 138.8},
    "MSFT": {"symbol": "MSFT", "price": 412.55, "change_pct": 0.65, "high_52w": 433.6, "low_52w": 309.5},
}


@tool
def get_quote(symbol: str) -> dict:
    """查询某只股票的最新报价与近 52 周高低（mock 数据，仅用于教学示例）。
    仅支持以下 symbol：NVDA、AAPL、TSLA、MSFT；其他 symbol 返回 error 字段。
    """
    sym = symbol.upper().strip()
    if sym in _QUOTE_DB:
        return _QUOTE_DB[sym]
    return {"error": f"unknown symbol: {sym}", "symbol": sym, "price": None}


# ====== 3. get_fundamentals（mock） ======

_FUNDAMENTALS_DB = {
    "NVDA": {"symbol": "NVDA", "pe": 68.2, "pb": 56.1, "rev_yoy": 1.22, "ni_yoy": 5.81},
    "AAPL": {"symbol": "AAPL", "pe": 29.1, "pb": 39.3, "rev_yoy": 0.06, "ni_yoy": 0.10},
    "TSLA": {"symbol": "TSLA", "pe": 51.7, "pb": 8.4, "rev_yoy": 0.02, "ni_yoy": -0.55},
    "MSFT": {"symbol": "MSFT", "pe": 35.4, "pb": 12.7, "rev_yoy": 0.17, "ni_yoy": 0.20},
}


@tool
def get_fundamentals(symbol: str) -> dict:
    """查询某只股票的基本面快照（PE/PB、营收同比、净利润同比；mock 数据）。
    仅支持以下 symbol：NVDA、AAPL、TSLA、MSFT；其他 symbol 返回 error 字段。
    """
    sym = symbol.upper().strip()
    if sym in _FUNDAMENTALS_DB:
        return _FUNDAMENTALS_DB[sym]
    return {"error": f"unknown symbol: {sym}", "symbol": sym}


# ====== 4. calculator（受限 AST 求值） ======

_ALLOWED_BIN_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                    ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow,
                    ast.FloorDiv: op.floordiv}
_ALLOWED_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        # 防 DoS：限制 ** 的右操作数为小常量，避免 9**9**9**9 之类卡死
        if isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and abs(node.right.value) <= 100):
                raise ValueError("** 右操作数必须是绝对值 ≤ 100 的常量")
        return _ALLOWED_BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expr: str) -> float | str:
    """安全计算简单算术表达式，仅支持 + - * / % ** //，不支持任何函数调用或变量引用。"""
    try:
        tree = ast.parse(expr, mode="eval")
        return _safe_eval(tree)
    except Exception as e:  # noqa: BLE001
        return f"calculator error: {e}"


# ====== 5. save_note（本地 markdown） ======

_FILENAME_SAFE = re.compile(r"[^\w一-龥\-]+")


@tool
def save_note(title: str, content: str) -> str:
    """把研究笔记保存为本地 markdown 文件。
    保存目录由环境变量 INVESTBOT_NOTES_DIR 控制，默认 ./notes。
    返回文件路径供 LLM 引用。
    """
    notes_dir = Path(os.environ.get("INVESTBOT_NOTES_DIR", "notes"))
    notes_dir.mkdir(parents=True, exist_ok=True)

    safe_title = _FILENAME_SAFE.sub("-", title.strip())[:60] or "note"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = notes_dir / f"{ts}-{safe_title}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"已保存笔记 {title} 到 {path}"


# ====== 工具清单（供 bind_tools / ToolNode） ======

ALL_TOOLS = [search_web, get_quote, get_fundamentals, calculator, save_note]
