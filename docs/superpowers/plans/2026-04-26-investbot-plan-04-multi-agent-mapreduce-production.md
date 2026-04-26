# InvestBot Plan 4：坐 4 —— 多 Agent 与生产化

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 InvestBot 走完"从教程到产品"最后一公里——多 Agent 协作（subgraph + supervisor）、并行 map-reduce（Send API）、对外服务化（FastAPI + LangSmith trace）。覆盖三个学习模块（09 multi-agent、x2 map-reduce、10 production）。

**Prerequisites:** Plan 3 完成（tag `plan-3-complete`）；42 测试通过。

**Architecture:**

- **09 multi-agent**：把 04 模块的 ReAct agent 包成"研究子图"和"写笔记子图"两个独立子图，再用一个 supervisor 节点根据用户意图路由到不同子图。子图当节点用是 LangGraph 一等公民——`compile` 后的 graph 可以直接 `add_node("name", subgraph)`
- **x2 map-reduce**：用 `Send` API 实现 fan-out + fan-in。给定一组 symbols，并行调 `get_quote`，最后用 reducer 把结果汇成列表
- **10 production**：把 04 ReAct agent 用 FastAPI 暴露成 HTTP endpoint（含 SSE 流式），加上重试 / 错误处理装饰器，再讲 LangSmith trace 怎么开

**Tech Stack:** Python 3.11+、langgraph 1.1.x、`langgraph.types.Send`、subgraph composition；新增 dev 依赖 `fastapi`、`httpx`（FastAPI TestClient）。LangSmith 走环境变量 opt-in，不强加依赖。

**Plan 范围（本份只覆盖 Plan 4）：**

- Phase 0：补 dev 依赖（fastapi + httpx）
- Phase 1：模块 09 multi-agent
- Phase 2：模块 x2 map-reduce
- Phase 3：模块 10 production
- Phase F：收尾 + tag —— 至此 InvestBot 全部模块完成

---

## 文件结构

```
09-multi-agent/
├── README.md
└── main.py
x2-map-reduce/
├── README.md
└── main.py
10-production/
├── README.md
├── main.py             # FastAPI app + 装饰器
└── client_demo.py      # 调用示例
tests/
├── test_module_09.py
├── test_module_x2.py
└── test_module_10.py
```

---

## 任务列表

---

### Task 0.1：补 dev 依赖（fastapi + httpx）

**Files:** Modify: `pyproject.toml`

- [ ] **Step 1：在 `[dependency-groups].dev` 中加 fastapi 和 httpx**

```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "fastapi>=0.110",
    "httpx>=0.27",
]
```

- [ ] **Step 2：sync 依赖**

```bash
uv sync
```

预期：能装上 fastapi、starlette、httpx、anyio 等，无版本冲突。

- [ ] **Step 3：commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add fastapi + httpx to dev deps for plan 4"
```

---

### Task 1.1：模块 09 multi-agent

**Files:** Create: `09-multi-agent/main.py`、`09-multi-agent/README.md`、`tests/test_module_09.py`

**学习目标：**
- 用编译后的 subgraph 当一个普通 node：`g.add_node("name", subgraph_compiled)`
- 用 supervisor 节点 + `Command(goto=...)` 在子图之间路由
- 把 InvestBot 拆成两个能力专精的子 agent：研究 / 写笔记

- [ ] **Step 1：写 `09-multi-agent/main.py`**

```python
"""模块 09：Multi-agent —— Subgraph + Supervisor

把 InvestBot 拆成两个专精子 agent：
- research_agent：能用 search_web / get_quote / get_fundamentals
- writer_agent：只能用 save_note（避免它"自己跑去查行情"）

再用一个 supervisor 节点根据用户意图路由到二者之一。
关键观察：编译后的 subgraph 可以直接作为节点 add_node("name", subgraph)。
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import get_fundamentals, get_quote, save_note, search_web


# ====== 通用：构造一个 ReAct 子图 ======

def _build_react_subgraph(llm: BaseChatModel, tools: list, name_hint: str):
    """把 04 的 ReAct 图模板化，传入工具列表即可。"""
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: InvestBotState) -> dict:
        sys_msg = SystemMessage(content=f"{SYSTEM_BASE}\n你当前的角色：{name_hint}")
        return {"messages": [llm_with_tools.invoke([sys_msg] + list(state["messages"]))]}

    g = StateGraph(InvestBotState)
    g.add_node("call_model", call_model)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    return g.compile()


# ====== Supervisor：决定下一步交给哪个子 agent ======

def make_supervisor(llm: BaseChatModel):
    """让 LLM 看用户问题，输出 'research' 或 'writer'。"""

    def supervisor(state: InvestBotState) -> Command[Literal["research", "writer", "__end__"]]:
        # 简化：取最后一条 HumanMessage 看关键词
        last_user = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_user is None:
            return Command(goto="__end__")

        prompt = [
            SystemMessage(
                content="判断下面这条用户消息属于哪类，仅输出 research / writer。\n"
                "research：查行情、看新闻、汇总基本面；\n"
                "writer：把内容整理成笔记保存。"
            ),
            HumanMessage(content=last_user.content),
        ]
        resp = llm.invoke(prompt)
        intent = (resp.content or "").strip().lower()
        if "writer" in intent:
            return Command(goto="writer", update={"last_intent": "writer"})
        return Command(goto="research", update={"last_intent": "research"})

    return supervisor


# ====== 主图：supervisor → research/writer 子图 ======

def build_graph(llm: BaseChatModel | None = None):
    if llm is None:
        llm = get_llm(temperature=0)

    research = _build_react_subgraph(
        llm, [search_web, get_quote, get_fundamentals], name_hint="研究员"
    )
    writer = _build_react_subgraph(llm, [save_note], name_hint="笔记整理员")

    g = StateGraph(InvestBotState)
    g.add_node("supervisor", make_supervisor(llm))
    g.add_node("research", research)  # 编译后的子图直接当节点用！
    g.add_node("writer", writer)
    g.add_edge(START, "supervisor")
    # supervisor 用 Command(goto=...) 自己跳，不用额外的 conditional edge
    g.add_edge("research", END)
    g.add_edge("writer", END)
    return g.compile()


def run(user_text: str, llm: BaseChatModel | None = None) -> dict:
    app = build_graph(llm=llm)
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    for q in ["NVDA 现在多少钱？", "把上面的研究整理成笔记保存"]:
        out = run(q)
        print(f"\n[Q] {q}")
        print(f"[intent] {out.get('last_intent')}")
        print(f"[final] {out['messages'][-1].content[:120]}")
```

- [ ] **Step 2：手动运行（需要 ARK_API_KEY）**

```bash
uv run python 09-multi-agent/main.py
```

预期：第一个问题路由到 research，第二个路由到 writer。每次输出末尾打印对应子 agent 的回答。

- [ ] **Step 3：写 smoke test `tests/test_module_09.py`**

```python
"""模块 09 smoke test：subgraph + supervisor 路由。"""
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


def test_supervisor_routes_to_research():
    """fake LLM 第一次回 'research'，子 agent 直接回最终答复。"""
    main = _load("09-multi-agent")
    fake = FakeChatModelWithTools(
        responses=[
            AIMessage(content="research"),  # supervisor 决策
            AIMessage(content="NVDA 现报 925"),  # research 子图直接给最终答
        ]
    )
    out = main.run("NVDA 多少钱", llm=fake)
    assert out["last_intent"] == "research"
    assert "NVDA" in out["messages"][-1].content


def test_supervisor_routes_to_writer():
    main = _load("09-multi-agent")
    fake = FakeChatModelWithTools(
        responses=[
            AIMessage(content="writer"),
            AIMessage(content="已写好笔记。"),
        ]
    )
    out = main.run("把它存成笔记", llm=fake)
    assert out["last_intent"] == "writer"
    assert "笔记" in out["messages"][-1].content
```

- [ ] **Step 4：跑 smoke test**

```bash
uv run pytest tests/test_module_09.py -v
```

预期：2 passed。

- [ ] **Step 5：写 `09-multi-agent/README.md`**（按 7 节标准）

````markdown
# 模块 09：Multi-agent —— Subgraph + Supervisor

## 1. 本模块要解决什么问题

单个全能 agent 挂着所有工具，规模一大就出现两个问题：①工具集太大让 LLM 选择困难；②不同任务对 prompt 的要求不一致。**多 agent** 把"专业领域"切片给独立 sub-agent，再用 supervisor 调度。

InvestBot 进度：从单 agent 拆成研究员 + 笔记员两个角色，supervisor 决定派谁干。

## 2. 核心概念

```
                     START
                       │
                       ▼
                 ┌────────────┐
                 │ supervisor │ ← LLM 看 user_text，决定 research / writer
                 └─────┬──────┘
                  Command(goto=...)
                  ┌────┴─────┐
                  ▼          ▼
            ┌─────────┐  ┌────────┐
            │research │  │ writer │  ← 各自是一个完整 ReAct 子图
            │ (sub)   │  │ (sub)  │
            └────┬────┘  └────┬───┘
                 │            │
                 └─────┬──────┘
                       ▼
                      END
```

- **Subgraph**：`g.add_node("name", another_compiled_graph)`——子图就是节点
- **共享 state**：父图与子图用同一个 State schema，子图能读到父图写的字段
- **Supervisor 模式**：一个节点根据某种规则（LLM 推理 / 硬编码 / RL）决定 goto 哪个子 agent
- **`Command(goto=..., update=...)`**：节点同时跳转 + 更新 state

## 3. 关键 API

| API | 一句话 |
|---|---|
| `g.add_node("research", compiled_subgraph)` | 编译后的图直接作节点 |
| `Command(goto="research", update={"last_intent": "research"})` | 在 supervisor 里同时跳 + 写字段 |
| 子图 + 父图 State schema 必须兼容 | 通常都用 `InvestBotState` 系列 |

## 4. 代码导读

- `_build_react_subgraph(llm, tools, name_hint)`：04 ReAct 图的工厂化版本，给不同子 agent 分配不同工具集
- `make_supervisor(llm)`：调 LLM 做"研究 vs 笔记"的二分类，输出 `Command(goto=..., update=...)`
- `build_graph`：把 supervisor + 两个子图串起来；子图作为节点直接 add_node

## 5. 如何运行

```bash
uv run python 09-multi-agent/main.py
```

观察两个输入分别被路由到 research / writer 子图。

## 6. 常见坑

1. **子图必须 compile 后才能加进父图**：`g.add_node("x", uncompiled_graph)` 不会工作
2. **State schema 不兼容**：子图用了父图没有的字段会报 KeyError；父图也读不到子图的私有字段（除非显式声明）
3. **递归限制**：父图 + 子图各自的递归层级会叠加；`recursion_limit` 在 invoke 时统一控制
4. **子图的 START/END**：子图内部从 START 进、到 END 出；父图调用子图时 = "进了子图，从子图 END 处回来"
5. **Supervisor 选择逻辑**：纯硬编码 / few-shot LLM / 强化学习都可以；本模块用最简单的 LLM 二分类
6. **数据流向**：子图返回的 state 会合并回父图（通过 reducer）

## 7. 小练习

1. 加第三个子 agent：`calculator_agent`（只挂 calculator 工具），supervisor 改三分类
2. 让 supervisor 在路由前看 messages 历史而不只是最后一条用户输入
3. 给每个子图独立 checkpointer：父图与子图各自记忆
````

- [ ] **Step 6：commit**

```bash
git add 09-multi-agent/ tests/test_module_09.py
git commit -m "feat(09): multi-agent — subgraph + supervisor"
```

---

### Task 1.2：模块 x2 map-reduce

**Files:** Create: `x2-map-reduce/main.py`、`x2-map-reduce/README.md`、`tests/test_module_x2.py`

**学习目标：**
- 用 `Send` API 把一组 symbols fan-out 给并行 worker
- 用 reducer（`operator.add`）把 worker 结果 fan-in 成列表
- 理解 `Send` 与普通 conditional edge 的区别（前者并行调度多份，后者走一条）

- [ ] **Step 1：写 `x2-map-reduce/main.py`**

```python
"""独立小例 x2：Map-Reduce —— Send API

给一组股票 symbols，并行查每只的 quote，然后汇总。
"map" 阶段：fan_out 节点用 Send 把 N 个 worker 任务并行投出去；
"reduce" 阶段：worker 各自写 results，列表 reducer 把它们累加；
最后 summary 节点读 results 输出一段汇总文字。

完全不调 LLM，专注演示 Send API 的并行机制。
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from common.tools import get_quote


class MapState(TypedDict, total=False):
    symbols: list[str]
    # results 用 operator.add 作为 reducer：每个 worker 返回一份单元素 list，会被合并
    results: Annotated[list[dict], operator.add]
    summary: str


def fan_out(state: MapState) -> list[Send]:
    """conditional edge 函数：返回 list[Send]，每个 Send 触发一份并行 worker。"""
    return [Send("worker", {"symbol": s}) for s in state["symbols"]]


def worker(state: dict) -> dict:
    """worker 接收的 state 是 Send 的 arg —— 这里是 {"symbol": s}。"""
    quote = get_quote.invoke({"symbol": state["symbol"]})
    return {"results": [quote]}  # 单元素 list，会被 reducer 累加


def summary(state: MapState) -> dict:
    lines = []
    for r in state["results"]:
        if "error" in r:
            lines.append(f"- {r['symbol']}: 未知")
        else:
            lines.append(f"- {r['symbol']}: ${r['price']} ({r['change_pct']:+.2f}%)")
    return {"summary": "今日报价：\n" + "\n".join(lines)}


def build_graph():
    g = StateGraph(MapState)
    g.add_node("worker", worker)
    g.add_node("summary", summary)

    # START 用 conditional edge 直接 fan-out
    g.add_conditional_edges(START, fan_out, ["worker"])
    g.add_edge("worker", "summary")
    g.add_edge("summary", END)
    return g.compile()


def run(symbols: list[str]) -> dict:
    app = build_graph()
    return app.invoke({"symbols": symbols, "results": []})


if __name__ == "__main__":
    out = run(["NVDA", "AAPL", "TSLA", "MSFT", "ZZZZ"])
    print(out["summary"])
```

- [ ] **Step 2：手动运行**

```bash
uv run python x2-map-reduce/main.py
```

预期：打印一段"今日报价"列表，前 4 只有具体价格，最后 ZZZZ 是"未知"。

- [ ] **Step 3：写 smoke test `tests/test_module_x2.py`**

```python
"""模块 x2 smoke test：验证 Send fan-out + reducer fan-in。"""
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


def test_fan_out_collects_all_results():
    main = _load("x2-map-reduce")
    out = main.run(["NVDA", "AAPL", "MSFT"])
    # 三只股票，results 列表应有 3 项
    assert len(out["results"]) == 3
    syms = {r.get("symbol") for r in out["results"]}
    assert syms == {"NVDA", "AAPL", "MSFT"}


def test_summary_includes_all_symbols():
    main = _load("x2-map-reduce")
    out = main.run(["NVDA", "AAPL"])
    assert "NVDA" in out["summary"]
    assert "AAPL" in out["summary"]


def test_unknown_symbol_handled_gracefully():
    main = _load("x2-map-reduce")
    out = main.run(["NVDA", "ZZZZ"])
    assert "未知" in out["summary"]
```

- [ ] **Step 4：跑 smoke test**

```bash
uv run pytest tests/test_module_x2.py -v
```

预期：3 passed。

- [ ] **Step 5：写 `x2-map-reduce/README.md`**

````markdown
# 独立小例 x2：Map-Reduce —— Send API

## 1. 本模块要解决什么问题

LangGraph 默认是顺序图——一个节点干完才到下一个。但有时候你需要 **并行**：比如同时查 10 只股票的报价，再汇总。`Send` API 让你在一条 conditional edge 上同时投放多份任务，框架并行执行，结果通过 reducer 自动汇总。

## 2. 核心概念

```
            START
              │
              ▼
       ┌──────────────┐
       │  fan_out     │ ← 返回 [Send("worker", {symbol:"NVDA"}), Send(..., AAPL), ...]
       └──────┬───────┘
              │ 并行调度
        ┌─────┼─────┬─────┐
        ▼     ▼     ▼     ▼
      worker worker worker worker
        │     │     │     │
        └──┬──┴──┬──┴─────┘
           ▼     ▼
       results: list (用 operator.add reducer 自动 merge)
              │
              ▼
          summary → END
```

- **`Send(node_name, arg)`**：fan-out 单元；arg 会作为该 worker 的输入 state
- **`operator.add` reducer**：list 之间相加 = concatenation；每个 worker 返回单元素 list，框架自动 concat 成完整 list
- **从 START 直接 fan-out**：用 `add_conditional_edges(START, fan_out, ["worker"])`

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.types import Send` | fan-out 单元 |
| `g.add_conditional_edges(START, fan_out_fn, [target_node_name])` | 第三个参数是可能去到的节点名列表 |
| `Annotated[list[T], operator.add]` | 列表自动累加的 reducer |

## 4. 代码导读

- `MapState`：`symbols`（输入）、`results`（每个 worker 写一项，靠 reducer concat）、`summary`（最终输出）
- `fan_out`：返回 list[Send]，每个 Send 触发一个 worker 并行执行
- `worker`：接收 Send 的 arg（`{"symbol": s}`）作为本地 state，调 `get_quote` 写 results
- `summary`：把 results 汇总成一段文字

## 5. 如何运行

```bash
uv run python x2-map-reduce/main.py
```

预期打印 5 只股票（含一只 ZZZZ 未知）的报价汇总。

## 6. 常见坑

1. **worker 的 state 不是父 state**：worker 接收的是 `Send` 的 arg；如果 worker 想读父 state 的其他字段，要在 Send arg 里显式传
2. **必须有 reducer**：`results` 字段必须挂 `operator.add` 之类的 reducer，否则后到达的 worker 会覆盖前面的
3. **顺序不保证**：worker 是并行的，results 列表里元素的顺序和 fan_out 时的顺序不一定一致
4. **conditional_edges 第三参数**：`["worker"]` 是 LangGraph 用来构图的提示——告诉它这条 edge 可能去到哪些节点；不传会报错
5. **错误处理**：某个 worker 抛异常，整个 graph 会 raise；生产中要在 worker 内部 catch，把 error 字段写进结果 list

## 7. 小练习

1. 加一个 `enrich` map 阶段：worker 之后再 fan-out 调 `get_fundamentals`，演示多级 map
2. 让 worker 调真 `search_web`，并行搜每只股票的新闻
3. 加 `top_k` 参数：summary 阶段只输出涨幅前 K 名
````

- [ ] **Step 6：commit**

```bash
git add x2-map-reduce/ tests/test_module_x2.py
git commit -m "feat(x2): map-reduce — Send API for parallel fan-out"
```

---

### Task 1.3：模块 10 production

**Files:** Create: `10-production/main.py`、`10-production/client_demo.py`、`10-production/README.md`、`tests/test_module_10.py`

**学习目标：**
- 把 LangGraph 应用包成 FastAPI HTTP service
- SSE 流式 endpoint 桥接 LangGraph 的 `astream`
- 错误处理 + 简单重试装饰器
- LangSmith trace 怎么开（环境变量 opt-in）

- [ ] **Step 1：写 `10-production/main.py`**

```python
"""模块 10：Production —— FastAPI + LangSmith trace + 错误处理

把 04 模块的 ReAct agent 暴露成 HTTP service：
- POST /chat       同步返回最终回复
- POST /chat/stream  SSE 流式返回中间 updates
- GET  /health     健康检查

LangSmith：仅当环境变量 LANGSMITH_API_KEY 与 LANGSMITH_TRACING=true 同时存在时才上报；
否则静默跳过。这样开发本地不需要 LangSmith 账号也能跑。
"""
import json
import logging
import os
import time
from functools import wraps

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote

log = logging.getLogger("investbot")
logging.basicConfig(level=logging.INFO)

TOOLS = [get_quote, calculator]


# ====== 重试装饰器：网络抖动场景兜底 ======

def retry(max_attempts: int = 3, delay: float = 0.5):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    log.warning(f"{fn.__name__} 失败第 {i+1} 次: {e}")
                    time.sleep(delay * (2 ** i))
            raise last_exc

        return wrapper

    return deco


# ====== 构造 graph（懒加载，方便测试时替换 llm） ======

_APP = None


def get_app(llm=None):
    """懒加载 + 简单缓存。生产中可以接入 lifespan 启动时构建。"""
    global _APP
    if _APP is not None and llm is None:
        return _APP

    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: InvestBotState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    g = StateGraph(InvestBotState)
    g.add_node("call_model", call_model)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    compiled = g.compile()
    if llm is None:  # 仅默认 llm 时缓存
        _APP = compiled
    return compiled


# ====== LangSmith 提示（不强加依赖） ======

def _check_langsmith():
    if os.environ.get("LANGSMITH_TRACING") == "true" and os.environ.get("LANGSMITH_API_KEY"):
        log.info("LangSmith trace 已开启")
    else:
        log.info("LangSmith 未开启（设 LANGSMITH_TRACING=true + LANGSMITH_API_KEY 可启用）")


# ====== FastAPI app ======

app = FastAPI(title="InvestBot", version="0.1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.on_event("startup")
def _startup():
    _check_langsmith()
    get_app()  # 预热


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@retry(max_attempts=2, delay=0.3)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        graph = get_app()
        out = graph.invoke({"messages": [HumanMessage(content=req.message)]})
        return ChatResponse(reply=out["messages"][-1].content)
    except Exception as e:  # noqa: BLE001
        log.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE：每收到一个 update 推一行 data: ..."""

    async def event_gen():
        graph = get_app()
        try:
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                stream_mode="updates",
            ):
                yield f"data: {json.dumps({k: '...' for k in chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

- [ ] **Step 2：写 `10-production/client_demo.py`（调用示例）**

```python
"""调用示例（开两个终端，一个跑 main.py，另一个跑这个）。"""
import json
import sys

import httpx


def main():
    base = "http://127.0.0.1:8000"

    print("--- /health ---")
    print(httpx.get(f"{base}/health").json())

    print("\n--- /chat ---")
    r = httpx.post(f"{base}/chat", json={"message": "查 NVDA 现价"}, timeout=60)
    print(r.json())

    print("\n--- /chat/stream ---")
    with httpx.stream("POST", f"{base}/chat/stream", json={"message": "查 AAPL"}, timeout=60) as resp:
        for line in resp.iter_lines():
            if line:
                print(line)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3：手动验证（需要 ARK_API_KEY）**

```bash
# 终端 1：启 server
uv run python 10-production/main.py
# 终端 2：调用
uv run python 10-production/client_demo.py
```

预期：/health 返回 ok；/chat 返回 NVDA 报价的回复；/chat/stream 推几行 SSE 然后 [DONE]。

- [ ] **Step 4：写 smoke test `tests/test_module_10.py`**（用 FastAPI TestClient + monkeypatch get_app）

```python
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
```

- [ ] **Step 5：跑 smoke test**

```bash
uv run pytest tests/test_module_10.py -v
```

预期：4 passed。

- [ ] **Step 6：写 `10-production/README.md`**

````markdown
# 模块 10：Production —— FastAPI + LangSmith + 错误处理

## 1. 本模块要解决什么问题

教程跑了一路，到这里 InvestBot 已经能"想 + 算 + 查 + 写 + 流"。但它还住在 Python REPL 里。生产化要解决三件事：①**对外服务化**（HTTP / SSE）；②**可观测性**（LangSmith trace）；③**容错**（重试 / 错误处理）。

## 2. 核心思路

```
       客户端
         │
   POST /chat         ── 阻塞返回最终回复
   POST /chat/stream  ── SSE 流式推 updates
         │
         ▼
   ┌──────────────────┐
   │  FastAPI app     │
   │  retry decorator │ ← 网络抖动兜底
   │  exception → 500 │
   └────────┬─────────┘
            │
            ▼
        LangGraph app
            │
       LangSmith trace（可选，env 开关）
```

## 3. 关键技术点

| 点 | 一句话 |
|---|---|
| FastAPI + Pydantic | 标准 Python 服务化 |
| `StreamingResponse` | SSE / 任意流式响应 |
| `app.astream(..., stream_mode="updates")` | 桥接到 SSE |
| `retry` 装饰器 | 指数退避，N 次后抛 |
| LangSmith | 设 `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` 即开 |
| `lifespan` / `on_event("startup")` | 服务启动时预热 graph |

## 4. 代码导读

- `retry`：最简单的指数退避装饰器；生产中可换成 `tenacity`
- `get_app`：懒加载 + 简单缓存，避免每次请求重建 graph
- `_check_langsmith`：env 检查后只 log，不强制安装
- `/chat`：阻塞 invoke，错误转 500
- `/chat/stream`：SSE 流式；每个 update 推一行；末尾推 `[DONE]`

## 5. 如何运行

```bash
# 终端 1：启 server
uv run python 10-production/main.py

# 终端 2：调用
uv run python 10-production/client_demo.py
```

LangSmith trace（可选）：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_xxx
export LANGSMITH_PROJECT=investbot
uv run python 10-production/main.py
# 打开 https://smith.langchain.com/ 看 trace
```

## 6. 常见坑

1. **`on_event` deprecate**：FastAPI 新版推 `lifespan`；本模块用 `on_event` 是为了简洁，生产建议迁移
2. **SSE 格式严格**：每行 `data: <json>\n\n`（双换行结尾），缺一个客户端就解析不到
3. **`astream` 在同步 endpoint 里报错**：要用 `async def` 端点 + `async for`
4. **重试装饰器与幂等**：retry 只对幂等操作安全；写盘 / 转账类操作要在装饰前考虑幂等键
5. **LangSmith 配置**：除了 KEY，还要设 `LANGSMITH_PROJECT` 否则 trace 会扔到 default project
6. **graph 缓存**：`_APP` 全局缓存方便 demo，但单测时要小心 monkeypatch
7. **uvicorn 启动方式**：`if __name__ == "__main__"` + `uvicorn.run` 适合脚本启动；生产用 `uvicorn 10-production.main:app --workers 4`

## 7. 小练习

1. 加 `/chat/v2` 端点：接受 `thread_id`，返回带记忆的多轮对话（结合模块 06）
2. 把 retry 替换成 `tenacity`，加 jitter + 自定义 Exception 白名单
3. 加 Prometheus 指标：每个 endpoint 的请求数、延迟分位
4. 接入 OpenTelemetry trace，端到端串到 LangSmith
````

- [ ] **Step 7：commit**

```bash
git add 10-production/ tests/test_module_10.py
git commit -m "feat(10): production — FastAPI + LangSmith hooks + retry"
```

---

### Task F：整体收尾

- [ ] **Step 1：跑全量测试**

```bash
uv run pytest -v
```

预期：Plan 1+2+3 = 42 + 09(2) + x2(3) + 10(4) = 51 passed。

- [ ] **Step 2：更新 README**

把"### 坐 4"区块改成可点击：

```markdown
### 坐 4：多 Agent 与生产化
- [09-multi-agent](09-multi-agent/) — Subgraph + Supervisor 多 agent 模式
- [x2-map-reduce](x2-map-reduce/) — Send API 并行 fan-out/fan-in（独立小例）
- [10-production](10-production/) — FastAPI + LangSmith + 重试装饰器
```

- [ ] **Step 3：commit + tag + 庆祝**

```bash
git add README.md
git commit -m "docs: update README — link plan 4 modules"
git tag plan-4-complete
git tag investbot-v1.0  # 全部模块完成
git log --oneline | head -25
```

---

## Plan 4 完成后的状态

- 仓库具备 10 个主线模块 + 2 个独立小例（x1 / x2），全部跑通
- InvestBot 已经具备：
  - 图与 state 抽象（坐 1）
  - 真 LLM + 工具调用 + ReAct 循环（坐 2）
  - 持久化记忆 + 人机协作 + 流式输出（坐 3）
  - 多 agent 协作 + 并行 map-reduce + 服务化部署（坐 4）

到这里 InvestBot 教程结束。学员应当具备：
- 独立设计 LangGraph 应用的图结构
- 在不同模式（路由、ReAct、多 agent、map-reduce）之间选择
- 把应用包成生产可部署的 service
- 处理常见坑：state 累加、checkpointer、interrupt、SSE 格式
