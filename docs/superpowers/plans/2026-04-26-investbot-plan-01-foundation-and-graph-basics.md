# InvestBot Plan 1：脚手架 + 坐 1（图与状态骨架）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 LangGraph 学习项目的工程骨架立起来，并完成"坐 1：图与状态骨架"三个模块（01 hello-graph、02 state-and-reducer、x1 pure-routing），让学员第一次跑通 LangGraph 的核心抽象。

**Architecture:** 一个单仓 Python 项目，由 `uv` 管理依赖；公共能力（LLM 工厂、mock 工具、共享 State、合规 prompts）放在 `common/` 包下；每个学习模块是独立子目录 `NN-xxx/`，目录内含可独立运行的 `main.py` 与讲义 `README.md`；测试集中在 `tests/`，对公共模块用真正的 TDD 单元测试，对学习模块用 smoke test（运行入口函数，断言关键状态/输出）。

**Tech Stack:** Python 3.11+、`uv`、`langgraph` 1.0.x、`langchain-openai`、`tavily-python`、`python-dotenv`、`pytest`。

**Plan 范围（本份只覆盖 Plan 1）：**

- Phase 0：项目脚手架 + 公共模块 `common/`
- Phase 1：坐 1 三个学习模块（01 / 02 / x1）

模块 03–10、x2 不在本 plan，由后续 Plan 2/3/4 覆盖。Plan 1 完工后该仓库已具备：所有依赖装好、`.env` 模板就绪、`common/` 公共能力可被后续模块直接 `import`、坐 1 三个模块全部可单独运行并通过 smoke test。

**关于 TDD 在学习教程中的折衷（重要）：**

- 公共模块（`common/llm.py`、`common/tools.py`、`common/state.py`）：**严格 TDD**——先写测试看到红，再实现，再变绿，再 commit
- 学习模块（`NN-xxx/main.py`）：**Build-then-Verify**——先写完整 demo 代码（教学示例），跑一遍肉眼验收，再补一份 smoke test 锁定关键行为，最后 commit。理由：教程代码的"成品形态"是教学起点，先有代码再有测试更贴合学习节奏，但仍要测试以防后续重构破坏

---

## 文件结构

```
lang-graph-demo/
├── README.md                       # 仓库总览（Task F.1 创建）
├── pyproject.toml                  # 依赖（Task 0.1 创建）
├── uv.lock                         # uv 自动生成
├── .env.example                    # 环境变量模板（Task 0.2 创建）
├── .gitignore                      # 已存在
├── common/
│   ├── __init__.py                 # Task 0.3 创建（空）
│   ├── state.py                    # Task 0.3 创建：共享 State 基类
│   ├── prompts.py                  # Task 0.4 创建：合规 system prompt 片段
│   ├── llm.py                      # Task 0.5 创建：get_llm() 工厂
│   └── tools.py                    # Task 0.6 创建：search_web / get_quote / get_fundamentals / calculator / save_note
├── tests/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── test_state.py           # Task 0.3
│   │   ├── test_llm.py             # Task 0.5
│   │   └── test_tools.py           # Task 0.6
│   ├── test_module_01.py           # Task 1.1
│   ├── test_module_02.py           # Task 1.2
│   └── test_module_x1.py           # Task 1.3
├── 01-hello-graph/
│   ├── README.md
│   └── main.py
├── 02-state-and-reducer/
│   ├── README.md
│   └── main.py
└── x1-pure-routing/
    ├── README.md
    └── main.py
```

每个文件的职责：

- `common/state.py`：定义 `InvestBotState`（基于 `MessagesState` 扩展），其他模块按需扩展
- `common/prompts.py`：把"我是研究助手，不出投资建议"这类合规话术抽出来，避免散落
- `common/llm.py`：`get_llm()` 工厂，从环境变量读取，返回配置好的 `ChatOpenAI`
- `common/tools.py`：所有工具的实现 + 一个 `ALL_TOOLS` 列表，供 `bind_tools` / `ToolNode` 使用
- 学习模块 `main.py`：每个文件 ≤ 200 行，关键行附简短中文注释解释 **为什么**
- 学习模块 `README.md`：固定 7 节（问题 / 概念 / API / 代码导读 / 运行 / 常见坑 / 练习）

---

## 任务列表

---

### Task 0.1：项目脚手架（uv + pyproject）

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1：用 Write 工具直接创建 `pyproject.toml`**

不用 `uv init`，直接写整个文件，避免它生成多余的 `hello.py` 等样板再删除。

```toml
[project]
name = "lang-graph-demo"
version = "0.1.0"
description = "Step-by-step LangGraph tutorial — InvestBot research assistant"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=1.0,<2.0",
    "langchain-openai>=0.2",
    "langchain-core>=0.3",
    "langgraph-checkpoint-sqlite",
    "tavily-python>=0.5",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
# 让 `from common.xxx import ...` 在跑 pytest 时稳定可用，不依赖 cwd
pythonpath = ["."]

[tool.uv]
package = false
```

> 说明：
> - `tool.uv.package = false`：不把仓库当成可安装包，适合教程仓库
> - `tool.pytest.ini_options.pythonpath = ["."]`：把仓库根加入 sys.path，测试可直接 `from common.xxx import ...`
> - `langgraph-checkpoint-sqlite` 不锁定版本，让 uv resolver 与 langgraph 主版本协商出兼容版本

- [ ] **Step 2：装依赖并锁定**

```bash
uv sync
```

预期：生成 `.venv/` 与 `uv.lock`；终端最后类似 `Resolved 70+ packages in ...`，无 ERROR。如出现版本冲突，先看错误信息再决定调整哪个上下界。

- [ ] **Step 3：验证 LangGraph 能 import**

```bash
uv run python -c "import langgraph, langgraph.graph; print(langgraph.__version__)"
```

预期：打印 `1.0.x`（具体小版本号），无 ImportError。

- [ ] **Step 4：commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: scaffold uv project with langgraph 1.0 deps"
```

---

### Task 0.2：环境变量模板

**Files:**
- Create: `.env.example`

- [ ] **Step 1：写 `.env.example`**

```bash
# === 火山引擎方舟（默认 LLM 后端） ===
# 在 https://console.volcengine.com/ark 创建 API Key
ARK_API_KEY=your_volcengine_ark_api_key_here

# 方舟 OpenAI 兼容 endpoint（一般无需修改）
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 模型 ID：可填官方名（如 doubao-1-5-pro-32k-250115）或你创建的 endpoint id
LLM_MODEL=doubao-1-5-pro-32k-250115

# === Tavily 联网搜索（自模块 04 起需要） ===
# 在 https://tavily.com 注册，免费额度 1000 次/月
TAVILY_API_KEY=your_tavily_api_key_here

# === LangSmith（可选，自模块 10 演示） ===
# LANGSMITH_API_KEY=
# LANGSMITH_TRACING=true
# LANGSMITH_PROJECT=lang-graph-demo
```

- [ ] **Step 2：让 `.gitignore` 屏蔽真实 `.env`**

确认仓库根目录的 `.gitignore` 中已包含 `.env`（spec 阶段已写入）。如缺失则补。

```bash
grep -E "^\.env$" .gitignore || echo ".env" >> .gitignore
```

- [ ] **Step 3：commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env.example with ark / tavily slots"
```

---

### Task 0.3：`common/state.py` —— 共享 State 基类（TDD）

**Files:**
- Create: `common/__init__.py`、`common/state.py`、`tests/__init__.py`、`tests/common/__init__.py`、`tests/common/test_state.py`

- [ ] **Step 1：创建空 `__init__.py` 文件**

```bash
mkdir -p common tests/common
touch common/__init__.py tests/__init__.py tests/common/__init__.py
```

- [ ] **Step 2：写失败测试 `tests/common/test_state.py`**

```python
"""测试 InvestBotState：messages 必须用 add_messages reducer 累加，不替换。"""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from common.state import InvestBotState


def _identity_node(state: InvestBotState) -> dict:
    """什么都不做，仅返回空 dict（不更新）。"""
    return {}


def test_state_messages_accumulate_via_reducer():
    """两次 invoke 同一个 graph，新 message 应被追加到 messages 列表，而非替换。"""
    graph = (
        StateGraph(InvestBotState)
        .add_node("noop", _identity_node)
        .add_edge(START, "noop")
        .add_edge("noop", END)
        .compile()
    )

    out = graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert len(out["messages"]) == 1

    # 模拟：把上一次输出的 messages 再次喂给 graph，并附加一条新消息
    out2 = graph.invoke(
        {"messages": out["messages"] + [AIMessage(content="hello")]}
    )
    assert len(out2["messages"]) == 2
    assert out2["messages"][0].content == "hi"
    assert out2["messages"][1].content == "hello"


def test_state_has_last_intent_field():
    """InvestBotState 应当包含 last_intent 字段（路由模块会用）。"""
    state: InvestBotState = {"messages": [], "last_intent": "explain"}
    assert state["last_intent"] == "explain"
```

- [ ] **Step 3：跑测试，确认 fail**

```bash
uv run pytest tests/common/test_state.py -v
```

预期：`ModuleNotFoundError: No module named 'common.state'` 或 `ImportError: cannot import name 'InvestBotState'`。

- [ ] **Step 4：写最小实现 `common/state.py`**

```python
"""InvestBot 共享 State。

后续模块按需通过 TypedDict 继承或扩展，但 messages + last_intent 这两个字段是项目通用约定。
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InvestBotState(TypedDict, total=False):
    # add_messages reducer：新 messages 会被追加，而不是替换整个列表
    messages: Annotated[list[AnyMessage], add_messages]
    # 路由结果，由 03 模块的路由节点写入；可能值："explain" / "stock" / "sector" / "fallback"
    last_intent: str
```

> `total=False`：允许某些字段缺省（比如模块 01 还没用到 `last_intent`）。

- [ ] **Step 5：跑测试，确认 pass**

```bash
uv run pytest tests/common/test_state.py -v
```

预期：2 passed。

- [ ] **Step 6：commit**

```bash
git add common/__init__.py common/state.py tests/__init__.py tests/common/__init__.py tests/common/test_state.py
git commit -m "feat(common): add shared InvestBotState with add_messages reducer"
```

---

### Task 0.4：`common/prompts.py` —— 合规 system prompt 片段

**Files:**
- Create: `common/prompts.py`

> 这个文件只是字符串常量，不需要单元测试。后续模块直接 import 使用。

- [ ] **Step 1：写 `common/prompts.py`**

```python
"""集中管理跨模块复用的 prompt 片段。

把"合规话术 / 角色定位"抽出来，避免散落各模块。
"""

# 全局 system prompt：所有调 LLM 的模块都拼上这段
SYSTEM_BASE = """你是 InvestBot，一个面向散户投资者的"投资研究信息助手"。

你的职责：
- 解释金融/投资概念
- 汇总个股、行业、概念板块的公开信息
- 帮助用户整理研究笔记

你严格遵守的边界：
- 不出具买入/卖出/持有的具体投资建议
- 不预测股价涨跌、不给点位
- 在涉及标的的回答末尾，必须附"以上为研究信息汇总，仅供参考，不构成投资建议"

回答风格：
- 中文为主，专有名词保留英文
- 简洁、结构化，必要时用要点列出
"""

# 路由分类用的 prompt（03 模块用）
ROUTE_INSTRUCTIONS = """请把用户的最新问题归入下列四类之一，仅输出类别字母：

A. explain — 解释概念、定义、公式（不涉及具体标的）
B. stock — 询问某只具体股票/ETF/基金（涉及代码或公司名）
C. sector — 询问行业、板块、概念（多个标的或宏观）
D. fallback — 与投资无关，或问题不清

只输出单个字母 A / B / C / D，不要任何其他内容。
"""
```

- [ ] **Step 2：commit**

```bash
git add common/prompts.py
git commit -m "feat(common): add shared compliance system prompts"
```

---

### Task 0.5：`common/llm.py` —— LLM 工厂（TDD）

**Files:**
- Create: `common/llm.py`、`tests/common/test_llm.py`

- [ ] **Step 1：写失败测试 `tests/common/test_llm.py`**

```python
"""测试 get_llm()：从 env 读取配置，缺失时清晰报错。"""
import os
import pytest
from unittest.mock import patch

from common.llm import get_llm


def test_get_llm_raises_when_api_key_missing():
    """ARK_API_KEY 没设置时，应当 raise，且 message 中包含变量名提示用户。"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="ARK_API_KEY"):
            get_llm()


def test_get_llm_raises_when_model_missing():
    """LLM_MODEL 没设置时，应当 raise，且 message 中包含变量名。"""
    with patch.dict(os.environ, {"ARK_API_KEY": "fake"}, clear=True):
        with pytest.raises(RuntimeError, match="LLM_MODEL"):
            get_llm()


def test_get_llm_returns_chat_openai_when_env_present():
    """env 齐全时返回 ChatOpenAI 实例，且 base_url / model / api_key 配置正确。"""
    from langchain_openai import ChatOpenAI

    env = {
        "ARK_API_KEY": "fake-key",
        "LLM_MODEL": "doubao-test",
        "ARK_BASE_URL": "https://ark.example.com/v3",
    }
    with patch.dict(os.environ, env, clear=True):
        llm = get_llm(temperature=0.5)
    # 用 model_dump() 跨 LangChain 版本稳定地检查配置（不依赖具体属性名）
    assert isinstance(llm, ChatOpenAI)
    assert llm.temperature == 0.5
    dump_str = str(llm.model_dump())
    assert "doubao-test" in dump_str
    assert "ark.example.com" in dump_str


def test_get_llm_uses_default_base_url_when_not_set():
    """ARK_BASE_URL 未设置时，应当走默认的火山方舟 endpoint。"""
    env = {"ARK_API_KEY": "fake-key", "LLM_MODEL": "doubao-test"}
    with patch.dict(os.environ, env, clear=True):
        llm = get_llm()
    assert "ark.cn-beijing.volces.com" in str(llm.model_dump())
```

- [ ] **Step 2：跑测试，确认 fail**

```bash
uv run pytest tests/common/test_llm.py -v
```

预期：4 个测试全 fail（ImportError 或 NameError）。

- [ ] **Step 3：写最小实现 `common/llm.py`**

```python
"""统一的 LLM 客户端工厂。

走 OpenAI 兼容协议，默认指向火山引擎方舟。
切换其他厂商（DeepSeek / 通义 / 智谱）只需改 .env，代码不动。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 仓库根目录的 .env 自动加载（多次调用幂等）
load_dotenv()

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def get_llm(temperature: float = 0.3, **kwargs) -> ChatOpenAI:
    """构造 ChatOpenAI 实例。

    必需 env：ARK_API_KEY、LLM_MODEL
    可选 env：ARK_BASE_URL（默认走火山方舟）

    其他 ChatOpenAI 参数通过 **kwargs 透传，比如 streaming=True。
    """
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY 未设置。请把 .env.example 复制为 .env 并填入火山引擎 API Key。"
        )

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError(
            "LLM_MODEL 未设置。请在 .env 中填入模型名，例如 doubao-1-5-pro-32k-250115。"
        )

    base_url = os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )
```

- [ ] **Step 4：跑测试，确认 pass**

```bash
uv run pytest tests/common/test_llm.py -v
```

预期：4 passed。如有失败，先看是不是 `model_name` / `openai_api_base` 属性名不一致，改成 LangChain 当前实际暴露的属性名（必要时 `print(dir(llm))` 调试）。

- [ ] **Step 5：commit**

```bash
git add common/llm.py tests/common/test_llm.py
git commit -m "feat(common): add get_llm() factory targeting volcengine ark"
```

---

### Task 0.6：`common/tools.py` —— InvestBot 工具集（TDD）

**Files:**
- Create: `common/tools.py`、`tests/common/test_tools.py`

> Tavily 在测试中不真调网络，用 mock。其他工具是纯函数。

- [ ] **Step 1：写失败测试 `tests/common/test_tools.py`**

```python
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
```

- [ ] **Step 2：跑测试，确认 fail**

```bash
uv run pytest tests/common/test_tools.py -v
```

预期：全 fail（ImportError / 找不到符号）。

- [ ] **Step 3：写最小实现 `common/tools.py`**

```python
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
    """查询某只股票的最新报价与近 52 周高低（mock 数据，仅用于教学示例）。"""
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
    """查询某只股票的基本面快照（PE/PB、营收同比、净利润同比；mock 数据）。"""
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
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = notes_dir / f"{ts}-{safe_title}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"已保存笔记 {title} 到 {path}"


# ====== 工具清单（供 bind_tools / ToolNode） ======

ALL_TOOLS = [search_web, get_quote, get_fundamentals, calculator, save_note]
```

- [ ] **Step 4：跑测试，确认 pass**

```bash
uv run pytest tests/common/test_tools.py -v
```

预期：8 passed。

- [ ] **Step 5：commit**

```bash
git add common/tools.py tests/common/test_tools.py
git commit -m "feat(common): add InvestBot tool kit (search/quote/fundamentals/calc/note)"
```

---

### Task 1.1：模块 01 hello-graph

**Files:**
- Create: `01-hello-graph/main.py`、`01-hello-graph/README.md`、`tests/test_module_01.py`

**学习目标：** 第一次跑通 LangGraph，理解最小四件套：`StateGraph` / `add_node` / `add_edge` / `compile()` + `invoke()`。

- [ ] **Step 1：写 `01-hello-graph/main.py`**

```python
"""模块 01：Hello Graph

最小可运行的 LangGraph：单节点图，把用户输入原样回显（不调用 LLM）。
学习重点：State 是什么、Node 是什么、START / END 是什么、compile() 与 invoke() 的区别。
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. 定义 State：图在执行过程中流动的"数据快照"
class HelloState(TypedDict):
    user_input: str
    reply: str


# 2. 定义 Node：一个普通 Python 函数，接收 state，返回 state 的部分更新
def echo_node(state: HelloState) -> dict:
    user = state["user_input"]
    # 注意：返回的 dict 只是"对 state 的局部更新"，LangGraph 会合并到完整 state
    return {"reply": f"InvestBot 收到你的提问：{user!r}"}


# 3. 组装图：StateGraph(State) → add_node → add_edge → compile
def build_graph():
    graph = StateGraph(HelloState)
    graph.add_node("echo", echo_node)
    graph.add_edge(START, "echo")  # START 是入口"虚拟节点"
    graph.add_edge("echo", END)    # END 是出口"虚拟节点"
    return graph.compile()         # compile 后才能 invoke


def run(user_input: str) -> dict:
    """暴露给 smoke test 的入口函数。"""
    app = build_graph()
    return app.invoke({"user_input": user_input, "reply": ""})


if __name__ == "__main__":
    out = run("什么是夏普比率？")
    print("=== 最终 state ===")
    print(out)
```

- [ ] **Step 2：手动运行，肉眼验收**

```bash
uv run python 01-hello-graph/main.py
```

预期 stdout 包含：

```
=== 最终 state ===
{'user_input': '什么是夏普比率？', 'reply': "InvestBot 收到你的提问：'什么是夏普比率？'"}
```

- [ ] **Step 3：写 smoke test `tests/test_module_01.py`**

```python
"""模块 01 smoke test：确认 graph 能 invoke 且 reply 正确反映输入。"""
import sys
import importlib.util
from pathlib import Path


def _load_module_main(module_dir: str):
    """直接按文件路径加载模块的 main.py（避免学习模块目录名以数字开头无法 import）。"""
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_01_echoes_user_input():
    main = _load_module_main("01-hello-graph")
    out = main.run("hello langgraph")
    assert out["user_input"] == "hello langgraph"
    assert "hello langgraph" in out["reply"]
    assert "InvestBot" in out["reply"]
```

- [ ] **Step 4：跑 smoke test，确认 pass**

```bash
uv run pytest tests/test_module_01.py -v
```

预期：1 passed。

- [ ] **Step 5：写讲义 `01-hello-graph/README.md`**

````markdown
# 模块 01：Hello Graph

## 1. 本模块要解决什么问题

第一次接触 LangGraph，先**摸清楚最小可运行图的形状**：State 长什么样、节点是什么、START/END 干嘛的、compile 与 invoke 的关系。本模块刻意**不调 LLM**，把"图机制"和"模型推理"两件事拆开学。

InvestBot 进度：现在它只是个"复读机"，把用户问题原样回显。后续模块会逐步把它升级成能调 LLM、用工具、记忆、流式输出的真正研究助手。

## 2. 核心概念

```
        START
          │
          ▼
       ┌──────┐
       │ echo │   ← 一个 Node = 一个 Python 函数
       └──────┘
          │
          ▼
         END
```

- **State**（`HelloState`）：图在执行过程中流动的数据快照，用 `TypedDict` 描述结构
- **Node**：一个普通 Python 函数 `(state) -> dict`，返回的 dict 是对 state 的**局部更新**
- **Edge**：节点之间的连线，决定执行顺序
- **START / END**：虚拟入口 / 出口节点，用于把"第一个真节点"与"最后一个真节点"接上图

## 3. 关键 API

| API | 一句话 |
|---|---|
| `StateGraph(State)` | 用一个 TypedDict 创建图，State 描述数据结构 |
| `graph.add_node(name, fn)` | 注册一个节点 |
| `graph.add_edge(from, to)` | 注册一条有向边 |
| `graph.compile()` | 把"声明的图"编译成可执行 app |
| `app.invoke(initial_state)` | 同步执行一次，返回最终 state |

## 4. 代码导读

打开 `main.py`：

- `class HelloState(TypedDict)`：定义两个字段，`user_input` 与 `reply`
- `def echo_node(state) -> dict`：节点函数，**只返回要更新的字段**，不返回完整 state
- `build_graph()`：标准三步：创建 StateGraph → add_node → add_edge → compile
- `run()`：把 invoke 包了一层，方便测试调用

## 5. 如何运行

```bash
uv run python 01-hello-graph/main.py
```

预期输出（关键行）：

```
=== 最终 state ===
{'user_input': '什么是夏普比率？', 'reply': "InvestBot 收到你的提问：'什么是夏普比率？'"}
```

## 6. 常见坑

1. **节点函数返回完整 state vs 局部更新**：返回完整 state 也能跑，但写局部更新（只含变化字段）才是地道用法，未来加 reducer 时不会冲突
2. **忘记 `compile()`**：直接 `graph.invoke(...)` 会报错。`StateGraph` 是"声明"，`compile()` 之后的对象才是"可执行 app"
3. **TypedDict 的字段缺失**：`invoke({"user_input": "..."})` 不报错，但下游节点访问 `state["reply"]` 时会 KeyError。要么 `total=False`，要么传齐字段

## 7. 小练习

1. 给图加第二个节点 `polish_node`，把 `reply` 末尾加上"（仅供学习）"，然后串成 `START → echo → polish → END`
2. 把 `HelloState` 改成 `total=False`，去掉 `reply: ""` 这个初始字段，看 invoke 是否仍能正常工作
3. 试试 `app.get_graph().draw_ascii()`，把图结构打印出来
````

- [ ] **Step 6：commit**

```bash
git add 01-hello-graph/main.py 01-hello-graph/README.md tests/test_module_01.py
git commit -m "feat(01): hello-graph — minimal StateGraph with single echo node"
```

---

### Task 1.2：模块 02 state-and-reducer

**Files:**
- Create: `02-state-and-reducer/main.py`、`02-state-and-reducer/README.md`、`tests/test_module_02.py`

**学习目标：** 理解多字段 State、Reducer 概念、`add_messages` 的作用、为什么 messages 字段要被特殊对待。

- [ ] **Step 1：写 `02-state-and-reducer/main.py`**

```python
"""模块 02：State 与 Reducer

升级 InvestBot：用 messages 列表代替单字符串，体会 add_messages reducer 的累加语义；
再加一个普通字段 turn_count 演示"不带 reducer 的字段被覆盖"的对比。
仍然不调 LLM —— 节点是确定性 Python 函数。
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 1. State：messages 字段加 reducer，turn_count 不加
class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_count: int  # 没有 reducer：每次返回该字段都会被"整体替换"


# 2. 节点：模拟"每轮回复"——不调 LLM，按规则拼一句话
def reply_node(state: ChatState) -> dict:
    last = state["messages"][-1]
    user_text = last.content if isinstance(last, HumanMessage) else "(非用户消息)"
    answer = AIMessage(content=f"我听到了：{user_text}")
    # 注意：messages 只返回"新增的那一条"——add_messages 会替我们追加进 list
    return {"messages": [answer], "turn_count": state.get("turn_count", 0) + 1}


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("reply", reply_node)
    g.add_edge(START, "reply")
    g.add_edge("reply", END)
    return g.compile()


def chat_once(history: list[AnyMessage], user_text: str) -> dict:
    """一次对话：把 user_text 追加到 history，跑一遍图，返回新 state。"""
    app = build_graph()
    return app.invoke({"messages": history + [HumanMessage(content=user_text)], "turn_count": len(history) // 2})


if __name__ == "__main__":
    history: list[AnyMessage] = []
    for q in ["什么是夏普比率？", "PE 又是什么？", "谢谢"]:
        out = chat_once(history, q)
        history = out["messages"]   # 把累计 messages 带到下一轮
        print(f"--- 第 {out['turn_count']} 轮 ---")
        for m in history:
            print(f"  [{m.__class__.__name__}] {m.content}")
```

- [ ] **Step 2：手动运行，肉眼验收**

```bash
uv run python 02-state-and-reducer/main.py
```

预期：每轮都会打印当前 messages 完整列表，能看到 HumanMessage 与 AIMessage 交替累加，不被覆盖。

- [ ] **Step 3：写 smoke test `tests/test_module_02.py`**

```python
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
```

- [ ] **Step 4：跑 smoke test，确认 pass**

```bash
uv run pytest tests/test_module_02.py -v
```

预期：2 passed。

- [ ] **Step 5：写讲义 `02-state-and-reducer/README.md`**

````markdown
# 模块 02：State 与 Reducer

## 1. 本模块要解决什么问题

模块 01 中节点返回的 `reply` 字段每次都被整体覆盖；但对话场景下我们要的是 messages 列表**追加**新条目而不是覆盖。LangGraph 用 **Reducer** 解决这个问题。

InvestBot 进度：现在它能"记住对话历史"——多轮调用同一个图，messages 会累加，每轮节点只需返回**新增的那一条**。

## 2. 核心概念

```
state 字段 ─┬─ 普通字段（无 reducer）  → 节点返回的值整体替换
            └─ Annotated[T, reducer]   → 节点返回的值经 reducer 合并到旧值
```

- **Reducer**：一个二元函数 `(old, update) -> new`，决定字段如何合并
- **`add_messages`**：LangGraph 内置的列表追加 reducer，会按 message id 智能去重 / 替换
- **`MessagesState`**：内置便捷 State（只含一个 `messages` 字段并带好 reducer），等价写法

## 3. 关键 API

| API | 一句话 |
|---|---|
| `Annotated[list[AnyMessage], add_messages]` | 给字段挂上 reducer |
| `from langgraph.graph.message import add_messages` | 内置 messages reducer |
| `from langgraph.graph import MessagesState` | 等价的便捷 State 基类 |

## 4. 代码导读

- `ChatState`：`messages` 带 reducer，`turn_count` 不带，留作对比
- `reply_node`：返回 `{"messages": [一条新 AIMessage]}`——只返回新增项，reducer 负责追加
- 主函数：连续调用三轮，把上一轮的 `messages` 喂给下一轮，体会"累加"

## 5. 如何运行

```bash
uv run python 02-state-and-reducer/main.py
```

观察输出：第 3 轮时 messages 列表里已经有 6 条（3 user + 3 ai），全是按顺序累加的。

## 6. 常见坑

1. **忘记 reducer**：消息字段不挂 `add_messages` 时，每次返回都会**整体覆盖** messages，等于把历史抹掉
2. **节点返回完整 messages 列表**：返回 `{"messages": full_list}` 也能用，但容易把"新旧合并"和"自己 append"两种心智模型混着用，最后调试困难。**始终只返回新增项**
3. **`turn_count` 这种数值字段**：没 reducer 是对的（你想"覆盖"为最新值）；如果用错了 reducer 会乱
4. **MessagesState vs 自定义 State**：只有 messages 一个字段时直接用 `MessagesState` 更简洁，要加别的字段就自己写 TypedDict

## 7. 小练习

1. 把 `ChatState` 替换成 `from langgraph.graph import MessagesState`，看代码哪里要改、哪里不变
2. 给 `turn_count` 也加一个 reducer：`Annotated[int, lambda old, new: (old or 0) + new]`，观察行为变化
3. 试着在节点里返回 `{"messages": []}`（空列表），看 add_messages 行为
````

- [ ] **Step 6：commit**

```bash
git add 02-state-and-reducer/main.py 02-state-and-reducer/README.md tests/test_module_02.py
git commit -m "feat(02): state-and-reducer — add_messages vs plain field"
```

---

### Task 1.3：模块 x1 pure-routing（独立小例）

**Files:**
- Create: `x1-pure-routing/main.py`、`x1-pure-routing/README.md`、`tests/test_module_x1.py`

**学习目标：** 在不引入 LLM 不确定性的前提下，把"条件边"和"循环"讲透。这是一个不接主线的独立例子，输入一个数字，让图自己反复执行直到收敛。

- [ ] **Step 1：写 `x1-pure-routing/main.py`**

```python
"""独立小例 x1：纯路由 + 循环

这个图"猜数字"——每次给当前 guess 加上 step，直到 guess >= target，然后退出。
完全不调 LLM，专注演示：
- add_conditional_edges：根据 state 选择下一个节点
- 循环边：节点指回自己（或前节点）形成环
- recursion_limit：防失控
"""
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END


class GuessState(TypedDict):
    target: int
    guess: int
    step: int
    log: list[str]


def increment(state: GuessState) -> dict:
    """节点：每次把 guess 加上 step，并记录一条 log。"""
    new_guess = state["guess"] + state["step"]
    log_line = f"guess: {state['guess']} -> {new_guess}"
    # 注意：log 没挂 reducer，所以我们手动拼接 old + new
    return {"guess": new_guess, "log": state["log"] + [log_line]}


def route(state: GuessState) -> Literal["increment", "__end__"]:
    """条件路由：guess 到了 target 就出图，否则回去再加一次。"""
    if state["guess"] >= state["target"]:
        return "__end__"
    return "increment"


def build_graph():
    g = StateGraph(GuessState)
    g.add_node("increment", increment)
    g.add_edge(START, "increment")
    # 关键：从 increment 节点出来后，根据 route() 返回值决定去哪
    g.add_conditional_edges("increment", route, {"increment": "increment", "__end__": END})
    return g.compile()


def run(target: int, step: int) -> dict:
    app = build_graph()
    # recursion_limit 防失控：默认 25，演示时手动调大一点更直观
    return app.invoke(
        {"target": target, "guess": 0, "step": step, "log": []},
        config={"recursion_limit": 50},
    )


if __name__ == "__main__":
    out = run(target=10, step=3)
    print(f"final guess = {out['guess']}, took {len(out['log'])} steps")
    for line in out["log"]:
        print(" ", line)
```

- [ ] **Step 2：手动运行，肉眼验收**

```bash
uv run python x1-pure-routing/main.py
```

预期：

```
final guess = 12, took 4 steps
  guess: 0 -> 3
  guess: 3 -> 6
  guess: 6 -> 9
  guess: 9 -> 12
```

- [ ] **Step 3：写 smoke test `tests/test_module_x1.py`**

```python
"""模块 x1 smoke test：验证条件边收敛 + recursion_limit 触发。"""
import sys
import importlib.util
from pathlib import Path

import pytest


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_loop_converges_when_step_positive():
    main = _load("x1-pure-routing")
    out = main.run(target=10, step=3)
    assert out["guess"] >= 10
    assert len(out["log"]) == 4  # 0→3→6→9→12


def test_loop_one_step_when_target_immediately_reachable():
    main = _load("x1-pure-routing")
    out = main.run(target=1, step=5)
    assert out["guess"] == 5
    assert len(out["log"]) == 1


def test_recursion_limit_protects_against_infinite_loop():
    """step=0 时永远到不了 target，recursion_limit 应当抛错保护。"""
    main = _load("x1-pure-routing")
    with pytest.raises(Exception):  # langgraph.errors.GraphRecursionError
        main.run(target=10, step=0)
```

- [ ] **Step 4：跑 smoke test，确认 pass**

```bash
uv run pytest tests/test_module_x1.py -v
```

预期：3 passed。

- [ ] **Step 5：写讲义 `x1-pure-routing/README.md`**

````markdown
# 独立小例 x1：纯路由 + 循环

## 1. 本模块要解决什么问题

LangGraph 最容易让人"看不清楚"的两个特性是 **条件边** 和 **循环**。把它们和 LLM 混在一起讲，新手会被"输出不确定"搞糊涂。本例**完全不用 LLM**，用一个"猜数字到收敛"的小图把骨架吃透。

## 2. 核心概念

```
            ┌──────────────┐
            │   START      │
            └─────┬────────┘
                  ▼
            ┌──────────────┐
   ┌──────► │  increment   │
   │        └─────┬────────┘
   │              │
   │              ▼
   │     ┌────────────────┐
   │     │  route(state)  │   ← 一个普通函数，根据 state 返回下一步标签
   │     └─┬─────────────┬┘
   │       │             │
   │   "increment"     "__end__"
   │       │             │
   └───────┘             ▼
                        END
```

- **条件边（conditional edges）**：从某节点出来后，调用一个 Python 函数，函数返回值映射到下一个节点
- **循环**：边可以指回前面的节点，形成环；只要 state 在变，环就有意义
- **`recursion_limit`**：防止环写错变死循环；默认 25，超出会 raise `GraphRecursionError`

## 3. 关键 API

| API | 一句话 |
|---|---|
| `add_conditional_edges(from, fn, mapping)` | 从 `from` 节点出来后，调用 `fn(state)`，把返回值通过 `mapping` 翻成实际目标节点 |
| `mapping = {"a": "node_a", "__end__": END}` | 路由函数返回值到节点的映射 |
| `app.invoke(state, config={"recursion_limit": 50})` | 调高循环上限 |

## 4. 代码导读

- `route(state)`：返回 `"increment"`（继续循环）或 `"__end__"`（出图）
- `add_conditional_edges("increment", route, {...})`：从 increment 出来后由 `route` 决定下一步
- 注意 `log` 字段没有 reducer，所以节点内**手动拼接** `state["log"] + [new_line]`，否则会被覆盖

## 5. 如何运行

```bash
uv run python x1-pure-routing/main.py
```

把 `target` 和 `step` 改成不同组合，观察循环次数。

## 6. 常见坑

1. **路由函数必须返回 mapping 中存在的 key**，否则 LangGraph 会找不到目标节点
2. **`__end__` 字符串**：路由函数中写 `__end__`（带下划线），不是 `END`；`mapping` 字典里再把 `"__end__"` 映射到 `END` 常量
3. **死循环**：忘记让 state 朝收敛方向变化（比如 `step=0`）就会无限循环，被 `recursion_limit` 兜住
4. **`add_conditional_edges` vs `add_edge`**：前者返回值动态、后者死连接，混用时图结构会奇怪

## 7. 小练习

1. 加一个"上限保护"路由：当 guess 超过 100 也提前退出，无论是否到达 target
2. 改成"二分逼近"：每次 guess += (target - guess) / 2，看路由如何处理浮点收敛
3. 把 `route` 改成 lambda 内联写在 `add_conditional_edges` 里，体会哪种更可读
````

- [ ] **Step 6：commit**

```bash
git add x1-pure-routing/main.py x1-pure-routing/README.md tests/test_module_x1.py
git commit -m "feat(x1): pure-routing — conditional edges and loops without LLM"
```

---

### Task F.1：仓库总览 README

**Files:**
- Create: `README.md`

- [ ] **Step 1：写仓库根 `README.md`**

````markdown
# LangGraph 学习项目 —— InvestBot 投资研究助手

一份从 0 到生产级的 LangGraph 教程，主线场景是"投资研究信息助手"，逐模块演进。

> **定位声明：** InvestBot 仅做研究信息汇总，不出投资建议、不预测涨跌。

## 目录结构

```
common/         # 跨模块复用：LLM 工厂、mock 工具、共享 State、合规 prompts
NN-xxx/         # 主线学习模块（10 个），每个目录可独立运行
xN-xxx/         # 独立小例（避开主线的特性演示）
tests/          # 测试：common/ 用单元测试，学习模块用 smoke test
docs/           # spec 与 plans
```

## 环境准备

```bash
# 1. 安装 uv（若没装）：https://docs.astral.sh/uv/
# 2. 装依赖
uv sync

# 3. 配 .env
cp .env.example .env
# 编辑 .env，填入：
#   ARK_API_KEY      —— 火山引擎方舟 API Key
#   LLM_MODEL        —— 模型 ID，例如 doubao-1-5-pro-32k-250115
#   TAVILY_API_KEY   —— Tavily 搜索 key（自模块 04 起需要）
```

## 学习路径

按顺序学，每个模块约 30-90 分钟：

### 坐 1：图与状态骨架（不调 LLM）
- [01-hello-graph](01-hello-graph/) — StateGraph / Node / Edge / compile / invoke
- [02-state-and-reducer](02-state-and-reducer/) — `add_messages` reducer 与字段合并语义
- [x1-pure-routing](x1-pure-routing/) — 条件边与循环（独立小例）

### 坐 2：接入 LLM 与工具（Plan 2，待生成）
- 03-routing-and-llm
- 04-tool-calling
- 05-react-loop

### 坐 3：让 Agent 真正能用（Plan 3，待生成）
- 06-persistence
- 07-human-in-the-loop
- 08-streaming

### 坐 4：多 Agent 与生产化（Plan 4，待生成）
- 09-multi-agent
- x2-map-reduce
- 10-production

## 跑某个模块

```bash
uv run python 01-hello-graph/main.py
# 或
cd 01-hello-graph && uv run python main.py
```

## 跑全部测试

```bash
uv run pytest -v
```

## 关于默认 LLM 后端

默认走**火山引擎方舟**（OpenAI 兼容协议）。切换到 DeepSeek / 通义 / 智谱等其他 OpenAI 兼容厂商，只需改 `.env` 中三件：`ARK_API_KEY`、`ARK_BASE_URL`、`LLM_MODEL`，代码不动。

## 设计与计划

- 设计文档：`docs/superpowers/specs/`
- 实施计划：`docs/superpowers/plans/`
````

- [ ] **Step 2：commit**

```bash
git add README.md
git commit -m "docs: add repo overview README with learning path"
```

---

### Task F.2：Plan 1 全量验证 + 整体收尾

- [ ] **Step 1：跑全部测试一次**

```bash
uv run pytest -v
```

预期：
- `tests/common/test_state.py` 2 passed
- `tests/common/test_llm.py` 4 passed
- `tests/common/test_tools.py` 8 passed
- `tests/test_module_01.py` 1 passed
- `tests/test_module_02.py` 2 passed
- `tests/test_module_x1.py` 3 passed
- 合计 20 passed，0 failed

如有 fail：先停下定位根因，必要时回到对应 Task 修；不要打补丁绕过。

- [ ] **Step 2：跑三个学习模块的 main.py 各一次，肉眼检查输出**

```bash
uv run python 01-hello-graph/main.py
uv run python 02-state-and-reducer/main.py
uv run python x1-pure-routing/main.py
```

预期：每条命令均成功退出（exit code 0），stdout 输出与各自 README §5 一致。

- [ ] **Step 3：检查 git 状态干净**

```bash
git status
```

预期：`nothing to commit, working tree clean`。如有未提交内容，说明前面漏了 commit，补上。

- [ ] **Step 4：确认 Plan 1 范围内的所有 spec 项已被覆盖**

对照 `docs/superpowers/specs/2026-04-26-langgraph-investbot-tutorial-design.md`：

- §3 技术栈 → Task 0.1（依赖）+ Task 0.5（LLM 工厂）已实现
- §3.2 `.env` 规范 → Task 0.2 完成
- §6 目录结构 → `common/`、三个学习模块、`tests/`、`docs/` 全部就位
- §4.1 模块 01 / 02 → Task 1.1 / 1.2 完成
- §4.2 模块 x1 → Task 1.3 完成
- §5 每模块统一结构 → 三个学习模块各自有 README（7 节）+ main.py + 对应 smoke test

模块 03–10、x2 不在 Plan 1 范围，留给后续 plan。

- [ ] **Step 5：打 tag 标记 Plan 1 完成**

```bash
git tag plan-1-complete
git log --oneline | head -20
```

最近 commit 序列大致：

```
docs: add repo overview README with learning path
feat(x1): pure-routing — conditional edges and loops without LLM
feat(02): state-and-reducer — add_messages vs plain field
feat(01): hello-graph — minimal StateGraph with single echo node
feat(common): add InvestBot tool kit (search/quote/fundamentals/calc/note)
feat(common): add get_llm() factory targeting volcengine ark
feat(common): add shared compliance system prompts
feat(common): add shared InvestBotState with add_messages reducer
chore: add .env.example with ark / tavily slots
chore: scaffold uv project with langgraph 1.0 deps
docs: 落定 LangGraph 学习项目 spec —— InvestBot 投资研究助手
```

---

## Plan 1 完成后的状态

仓库具备：

- 完整的 `uv` 项目骨架，依赖锁定到 `uv.lock`
- `.env.example` 模板，按文档填好 key 即可全栈跑通
- `common/` 公共能力：LLM 工厂（火山方舟）、5 个工具（含真实 Tavily）、共享 State、合规 prompts
- 三个可独立运行的学习模块 + 三份讲义 + 对应 smoke test
- 仓库总览 README

学员到这里应当能：

- 回答"State / Node / Edge / START / END / compile / invoke 各是什么"
- 解释 reducer 为什么必要、`add_messages` 在干嘛
- 看着代码画出条件边的执行流程
- 自己改改循环条件而不会让图死循环

---

## 下一步：Plan 2（坐 2：接入 LLM 与工具）

Plan 1 完工并验证后，会接着生成 **Plan 2**，覆盖：

- 模块 03 routing-and-llm：第一次调火山方舟，LLM 做路由分类
- 模块 04 tool-calling：`@tool` / `bind_tools` / `ToolNode` / `tools_condition` 接入工具
- 模块 05 react-loop：用 `create_react_agent` 高阶封装，再手写一份低阶版本对比

到那时 InvestBot 才真正"活起来"——能联网、能算、能查行情。
