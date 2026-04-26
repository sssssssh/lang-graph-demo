# InvestBot Plan 2：坐 2 —— 接入 LLM 与工具

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 InvestBot 从"复读机"升级成"会调用模型 + 会用工具"的真正研究助手，覆盖 LangGraph 学习路径中的"坐 2：接入 LLM 与工具"——三个学习模块（03 routing-and-llm、04 tool-calling、05 react-loop）。

**Prerequisites:** Plan 1 完成（tag `plan-1-complete`）。仓库已具备 `common/llm.py`（`get_llm()` 工厂）、`common/tools.py`（5 个 `@tool`）、`common/state.py`（`InvestBotState` 基类）、`common/prompts.py`（`SYSTEM_BASE` / `ROUTE_INSTRUCTIONS`），并通过 20 个测试。

**Architecture:** 沿用 Plan 1 的目录布局——每个学习模块一个独立子目录（`NN-xxx/main.py` + `README.md`），测试集中在 `tests/`。本 Plan 在 `tests/` 下新增 `conftest.py` 提供两件事：① autouse fixture 重置 `common.tools._TAVILY_CLIENT` 全局缓存以隔离测试；② 一个共享的"假 LLM 注入"思路（不强制工具，而是教学习模块用**依赖注入**：`build_graph(llm=None)` 默认调 `get_llm()`，测试时传入 `FakeMessagesListChatModel`）。

**Tech Stack:** Python 3.11+、`langgraph` 1.1.x（已锁 1.1.9）、`langchain-core`、`langchain-openai`、`pytest`。本 Plan 用到的 LangGraph API：`langgraph.prebuilt.ToolNode`、`tools_condition`、`create_react_agent`；`AIMessage.tool_calls` 字段；`langchain_core.tools.@tool` 已在 Plan 1 用过。

**Plan 范围（本份只覆盖 Plan 2）：**

- Phase 0：测试公共设施（`tests/conftest.py`）
- Phase 1：模块 03 routing-and-llm（第一次调真 LLM，做路由分类）
- Phase 2：模块 04 tool-calling（`bind_tools` / `ToolNode` / `tools_condition` 三件套）
- Phase 3：模块 05 react-loop（`create_react_agent` 高阶 + 手写低阶对比）
- Phase F：整体收尾 + tag

模块 06–10、x2 不在本 Plan，由 Plan 3/4 覆盖。Plan 2 完工后 InvestBot 已经能：①用 LLM 把用户问题路由到不同处理路径；②让 LLM 自己决定何时调用工具、调哪个工具；③用 `create_react_agent` 一行起飞，并理解它内部到底做了什么。

**TDD 折衷沿用 Plan 1：** `tests/conftest.py` 等公共设施仍按"先测后写"；学习模块 main.py 用 Build-then-Verify（先写完整 demo，跑一遍肉眼验收，再补 smoke test 锁定关键行为）。

**关于 LLM 测试的策略（重要）：**

本 Plan 三个学习模块都涉及真 LLM 调用，但测试**绝不联网**。统一做法：

1. 每个学习模块的 `build_graph()` 接受可选参数 `llm: BaseChatModel | None = None`，默认 `None` 时内部调 `get_llm()` 取真模型；测试传入 `FakeMessagesListChatModel(responses=[...])` 预制对话
2. `FakeMessagesListChatModel` 来自 `langchain_core.language_models.fake_chat_models`，支持 `tool_calls` 字段——预制 `AIMessage(content="", tool_calls=[{...}])` 即可模拟模型选择调用工具
3. 这种"依赖注入"既让测试隔离干净，又顺便教学员怎么把"模型来源"从图结构里解耦——这是生产化常见模式

---

## 文件结构

```
lang-graph-demo/
├── tests/
│   ├── conftest.py               # Task 0.1 创建：Tavily 缓存重置 + 共享 helper
│   ├── test_module_03.py         # Task 1.1
│   ├── test_module_04.py         # Task 1.2
│   └── test_module_05.py         # Task 1.3
├── 03-routing-and-llm/
│   ├── README.md
│   └── main.py
├── 04-tool-calling/
│   ├── README.md
│   └── main.py
└── 05-react-loop/
    ├── README.md
    └── main.py
```

每个文件的职责：

- `tests/conftest.py`：autouse fixture 在每个测试前清空 `common.tools._TAVILY_CLIENT`，避免上一个测试 monkeypatch 的假 client 泄漏到下一个测试
- `03-routing-and-llm/main.py`：第一次调火山方舟真 LLM；用 `ROUTE_INSTRUCTIONS` 让模型把用户问题分类成 A/B/C/D 四类；用 `add_conditional_edges` 路由到四个分支节点
- `04-tool-calling/main.py`：演示工具调用三件套——`llm.bind_tools(ALL_TOOLS)`、`ToolNode(ALL_TOOLS)`、`tools_condition` 控制 LLM ↔ ToolNode 循环；选用 `get_quote` + `calculator`（不依赖网络），演示完整 ReAct 雏形
- `05-react-loop/main.py`：用 `create_react_agent` 一行搭出 ReAct agent；并把 04 的手写图作为低阶对比版，标注每段对应高阶封装的什么部分

---

## 任务列表

---

### Task 0.1：`tests/conftest.py` —— 测试公共设施（TDD）

**Files:**
- Create: `tests/conftest.py`

> 目的：解决 Plan 1 final review 留下的 Important 问题——`common.tools._TAVILY_CLIENT` 是模块级全局变量，第一个测试 monkeypatch 后，第二个测试如果不再 monkeypatch 就会拿到旧的假 client。autouse fixture 一劳永逸。

- [ ] **Step 1：先写一个会暴露问题的测试**

把要新建的 `tests/conftest.py` 的"反例"作为锚点测试，加到 `tests/common/test_tools.py` 末尾（临时验证用，验证完保留）：

```python
def test_tavily_client_is_reset_between_tests(monkeypatch):
    """这个测试应当与 test_search_web_uses_injected_client 完全隔离。
    若 conftest.py 没有正确 reset _TAVILY_CLIENT，本测试可能会"复用"上一个测试 monkeypatch 的 FakeClient。
    """
    import common.tools as tools_mod
    # 进来时 _TAVILY_CLIENT 必须为 None（被 conftest 清掉）
    assert tools_mod._TAVILY_CLIENT is None, (
        "_TAVILY_CLIENT 在测试入口应为 None，否则跨测试状态会泄漏"
    )
```

- [ ] **Step 2：跑测试，确认 fail（如顺序不利）**

```bash
uv run pytest tests/common/test_tools.py -v
```

预期：在 `test_search_web_uses_injected_client` 之后跑的话，新测试会因为 `_TAVILY_CLIENT` 不为 None 而失败。如恰好顺序无害则 pass，但 conftest 仍要写——这是结构性问题。

- [ ] **Step 3：写 `tests/conftest.py`**

```python
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
```

- [ ] **Step 4：跑全部测试，确认 pass**

```bash
uv run pytest -v
```

预期：Plan 1 的 20 个 + 新增 1 个 = 21 passed，0 failed。

- [ ] **Step 5：commit**

```bash
git add tests/conftest.py tests/common/test_tools.py
git commit -m "test: add conftest to reset tavily client between tests"
```

---

### Task 1.1：模块 03 routing-and-llm

**Files:**
- Create: `03-routing-and-llm/main.py`、`03-routing-and-llm/README.md`、`tests/test_module_03.py`

**学习目标：**
- 第一次让 LangGraph 调用真 LLM（火山方舟），理解 `llm.invoke([SystemMessage, HumanMessage])` 的形态
- 用 LLM 输出做路由分类，把"分类结果"写进 state 的 `last_intent`
- 用 `add_conditional_edges` + 一个简单 `route()` 函数把图分到 4 个分支节点
- 第一次以 `InvestBotState` 为基类（per common/state.py 跨模块约定）

- [ ] **Step 1：写 `03-routing-and-llm/main.py`**

```python
"""模块 03：Routing 与 LLM

第一次让 InvestBot 调用真 LLM。把用户的最新问题用 LLM 分类成 4 类，
再根据分类结果走 4 条分支，每条分支只回一段固定模板（暂不调工具，留给 04）。

学习重点：
- LLM 在节点里怎么调（同步 .invoke，传 messages 列表）
- 怎么把 LLM 的输出"提取"成结构化字段写进 state
- add_conditional_edges 怎么根据 state 字段挑分支
- InvestBotState 作为跨模块基类的扩展用法
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from common.llm import get_llm
from common.prompts import ROUTE_INSTRUCTIONS, SYSTEM_BASE
from common.state import InvestBotState


# 1. State：直接复用 InvestBotState（含 messages + last_intent）
#    若需要扩展，写 class RoutingState(InvestBotState, total=False): ...
RoutingState = InvestBotState


# 2. 路由节点：调 LLM 让它输出 A/B/C/D 单字母
_LETTER_TO_INTENT = {"A": "explain", "B": "stock", "C": "sector", "D": "fallback"}


def make_route_node(llm: BaseChatModel):
    """工厂函数：把 llm 关进闭包，返回节点函数。

    用 closure 而不是把 llm 塞进 state，是因为 llm 不该序列化进 state（影响 checkpoint）。
    """

    def route_node(state: RoutingState) -> dict:
        # 取最后一条用户消息作为分类依据
        last_user = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_user is None:
            return {"last_intent": "fallback"}

        # 调 LLM。注意：传 SystemMessage + HumanMessage，与直接 OpenAI API 一致
        prompt = [
            SystemMessage(content=ROUTE_INSTRUCTIONS),
            HumanMessage(content=last_user.content),
        ]
        resp = llm.invoke(prompt)
        # 取首字符兜底（即使模型话多，也能拿到 A/B/C/D）
        letter = (resp.content or "").strip().upper()[:1]
        intent = _LETTER_TO_INTENT.get(letter, "fallback")
        return {"last_intent": intent}

    return route_node


# 3. 四个分支节点：暂时只回固定话术，演示路由分发
def explain_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[explain 分支] 我会解释这个概念。" + _DISCLAIMER)]}


def stock_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[stock 分支] 我会查这只股票的公开信息。" + _DISCLAIMER)]}


def sector_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[sector 分支] 我会汇总这个板块的研究信息。" + _DISCLAIMER)]}


def fallback_node(state: RoutingState) -> dict:
    return {"messages": [AIMessage(content="[fallback 分支] 这个问题超出我的范围，请换个投资相关的问题。")]}


_DISCLAIMER = "（以上为研究信息汇总，仅供参考，不构成投资建议）"


# 4. 路由函数：从 state.last_intent 翻译成下一个节点名
def route(state: RoutingState) -> Literal["explain", "stock", "sector", "fallback"]:
    return state.get("last_intent", "fallback")  # type: ignore[return-value]


# 5. 组图
def build_graph(llm: BaseChatModel | None = None):
    """构造路由图。llm=None 时走真 get_llm()，测试时传入 FakeMessagesListChatModel。"""
    if llm is None:
        llm = get_llm(temperature=0)  # 路由要尽量确定性，温度调零

    g = StateGraph(RoutingState)
    g.add_node("route", make_route_node(llm))
    g.add_node("explain", explain_node)
    g.add_node("stock", stock_node)
    g.add_node("sector", sector_node)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        route,
        {
            "explain": "explain",
            "stock": "stock",
            "sector": "sector",
            "fallback": "fallback",
        },
    )
    for branch in ("explain", "stock", "sector", "fallback"):
        g.add_edge(branch, END)
    return g.compile()


def run(user_text: str, llm: BaseChatModel | None = None) -> dict:
    app = build_graph(llm=llm)
    # SYSTEM_BASE 不强制塞，路由节点自己用 ROUTE_INSTRUCTIONS；分支节点也不调 LLM
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    # 跑前确认 .env 已配 ARK_API_KEY + LLM_MODEL
    samples = [
        "什么是夏普比率？",          # → explain
        "NVDA 现在多少钱？",          # → stock
        "新能源板块最近怎么样？",     # → sector
        "今晚吃什么？",               # → fallback
    ]
    for q in samples:
        out = run(q)
        last_msg = out["messages"][-1].content
        print(f"Q: {q}\n  intent = {out['last_intent']}\n  reply  = {last_msg}\n")
```

- [ ] **Step 2：手动运行，肉眼验收**

```bash
uv run python 03-routing-and-llm/main.py
```

预期：4 个样例分别被分到 explain / stock / sector / fallback，每条 reply 都带正确分支前缀。如某条分类不准确（比如"NVDA 现在多少钱？"被分到 explain），可以先记下来，**这是真 LLM 的不确定性**——后续可调 ROUTE_INSTRUCTIONS 或 examples，不在本任务范围内修。

- [ ] **Step 3：写 smoke test `tests/test_module_03.py`**

```python
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
```

- [ ] **Step 4：跑 smoke test，确认 pass**

```bash
uv run pytest tests/test_module_03.py -v
```

预期：5 passed。

- [ ] **Step 5：写讲义 `03-routing-and-llm/README.md`**

````markdown
# 模块 03：Routing 与 LLM

## 1. 本模块要解决什么问题

InvestBot 终于要调真 LLM 了。但我们刻意让它**只做一件事——分类**：把用户问题归到 explain / stock / sector / fallback 四类之一，再让图据此分流到不同分支。原因：

- 把"调 LLM"这件事讲透：消息怎么传、输出怎么取、温度怎么调
- 把"LLM 输出的不确定性"和"图结构的确定性"分开——LLM 只决定一个字段，图怎么走是确定的
- 为后续模块铺路（04 在 stock 分支接真工具）

## 2. 核心概念

```
                START
                  │
                  ▼
              ┌────────┐
              │ route  │  调 LLM 输出 A/B/C/D，写入 state.last_intent
              └────┬───┘
                   │
       add_conditional_edges(route_fn, mapping)
                   │
       ┌─────┬─────┼─────┬──────┐
       ▼     ▼     ▼     ▼      ▼
   explain stock sector fallback
       │     │     │     │
       └─────┴──┬──┴─────┘
                ▼
               END
```

- **`llm.invoke(messages)`**：同步调用，输入 `[SystemMessage, HumanMessage, ...]`，返回 `AIMessage`
- **依赖注入 build_graph(llm=None)**：让节点对"模型来源"不耦合，方便测试塞 fake、生产换厂商
- **routes via state field**：把分类结果写进 `state.last_intent`，路由函数只读这个字段——保持职责单一

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from common.llm import get_llm` | 火山方舟 LLM 工厂 |
| `llm.invoke([SystemMessage(...), HumanMessage(...)])` | 同步调用，返回 AIMessage |
| `add_conditional_edges(from, fn, mapping)` | 已在 x1 学过，本模块第一次结合 LLM 输出 |
| `InvestBotState` | 跨模块共享 State 基类，`messages` 自带 add_messages，`last_intent` 留给路由写 |

## 4. 代码导读

- `make_route_node(llm)`：工厂函数，把 llm 关进 closure。**为什么不让节点直接调 `get_llm()`？** 因为节点会被多次执行，每次都构造 ChatOpenAI 浪费；也不利于测试注入
- `route_node`：取最后一条用户消息 → 拼 `[ROUTE_INSTRUCTIONS, user_text]` → llm.invoke → 取首字母 → 写 last_intent
- 四个分支节点：暂时只回固定话术，留待 04 接工具
- `route(state)`：路由函数，只读 last_intent
- `build_graph(llm=None)`：依赖注入入口，默认走真 LLM

## 5. 如何运行

```bash
# 1. 先确认 .env 已配 ARK_API_KEY + LLM_MODEL（cp .env.example .env，填 key）
# 2. 跑
uv run python 03-routing-and-llm/main.py
```

预期：4 个样例分别被分到不同分支，输出形如：

```
Q: 什么是夏普比率？
  intent = explain
  reply  = [explain 分支] 我会解释这个概念。（以上为研究信息汇总，仅供参考，不构成投资建议）

Q: NVDA 现在多少钱？
  intent = stock
  reply  = [stock 分支] ...
...
```

> 真 LLM 偶尔会分错类（比如"巴菲特怎么看"可能分到 stock 也可能分到 explain）。这是不确定性，**不是 bug**。生产中可以加 few-shot 例子或换更强模型缓解。

## 6. 常见坑

1. **`temperature` 不调 0**：路由要确定性，默认 0.3 也偏高，本模块用 `get_llm(temperature=0)`
2. **节点里直接 `get_llm()`**：每次调用都会重建 ChatOpenAI 客户端；用 closure 工厂只构造一次
3. **路由函数返回值不在 mapping**：比如 `last_intent="other"`，会 KeyError；务必给 default 兜底
4. **Conditional edge 的"自然"目标节点**：mapping 的 key 要和 `route()` 返回值一致，value 是真实节点名；二者可同名也可不同名
5. **`HumanMessage` vs 字符串**：`llm.invoke("hi")` 在新版能跑，但 `llm.invoke([HumanMessage("hi")])` 才是地道写法（与 LangChain Message 抽象一致）
6. **测试如何隔离 LLM**：用 `FakeMessagesListChatModel(responses=[AIMessage(content="A")])` 预制；它和真 ChatOpenAI 都实现 `BaseChatModel` 接口，可直接互换

## 7. 小练习

1. 给 ROUTE_INSTRUCTIONS 加 2 个 few-shot 例子（比如"PE 是什么 → A"、"宁德时代 → B"），看分类准确率
2. 把 `route_node` 改用 `with_structured_output(SchemaWith Literal)`，让 LLM 输出强类型而不是字母
3. 实现一个 `unknown_node`：当 LLM 输出无法识别时（不只是字母不在 ABCD），路由走这里而不是 fallback
````

- [ ] **Step 6：commit**

```bash
git add 03-routing-and-llm/main.py 03-routing-and-llm/README.md tests/test_module_03.py
git commit -m "feat(03): routing-and-llm — first real LLM call for intent classification"
```

---

### Task 1.2：模块 04 tool-calling

**Files:**
- Create: `04-tool-calling/main.py`、`04-tool-calling/README.md`、`tests/test_module_04.py`

**学习目标：**
- 理解"工具调用"协议：LLM 返回 `AIMessage(content="", tool_calls=[{name, args, id}])` 是什么意思
- `llm.bind_tools(tools)` 把工具的 JSON schema 喂给模型
- `ToolNode(tools)` 自动执行 tool_calls，把结果包装成 `ToolMessage` 写回 messages
- `tools_condition` 检查最后一条 AIMessage 是否带 tool_calls，决定走 `"tools"` 还是 `"__end__"`
- LLM ↔ ToolNode 形成的循环就是 ReAct 的雏形

- [ ] **Step 1：写 `04-tool-calling/main.py`**

```python
"""模块 04：Tool Calling

让 InvestBot 学会"主动调工具"。本模块演示工具调用三件套：
- llm.bind_tools(ALL_TOOLS)：让 LLM 知道有哪些工具可用、参数 schema 长啥样
- ToolNode(ALL_TOOLS)：负责把 LLM 选中的工具实际执行掉
- tools_condition：检查上一条 AIMessage 是否要求调工具，决定循环还是退出

为聚焦机制本身，本模块只用 get_quote + calculator 两个不依赖网络的工具。
search_web / save_note / get_fundamentals 留到 05 用 create_react_agent 时一起放进来。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote


# 本模块用的工具子集（不依赖网络，便于本地跑通）
TOOLS = [get_quote, calculator]


def make_call_model(llm_with_tools: BaseChatModel):
    """节点工厂：调 LLM（已 bind_tools），把它的回复追加到 messages。"""

    def call_model(state: InvestBotState) -> dict:
        # 把 SystemMessage 拼到最前面（不写进 state，避免重复）
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        resp = llm_with_tools.invoke(msgs)
        return {"messages": [resp]}  # add_messages 会追加

    return call_model


def build_graph(llm: BaseChatModel | None = None):
    """构造工具调用循环图。

    结构：
        START → call_model → tools_condition →
                                  ├─ "tools" → ToolNode → call_model（回环）
                                  └─ END
    """
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    g = StateGraph(InvestBotState)
    g.add_node("call_model", make_call_model(llm_with_tools))
    g.add_node("tools", ToolNode(TOOLS))   # 节点名必须叫 "tools"，与 tools_condition 的 mapping 默认一致

    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)  # 有 tool_calls → "tools"，否则 → END
    g.add_edge("tools", "call_model")  # 工具执行完，回到模型让它"看到"工具结果再决定
    return g.compile()


def run(user_text: str, llm: BaseChatModel | None = None) -> dict:
    app = build_graph(llm=llm)
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    out = run("帮我查一下 NVDA 的股价，再帮我算 925.30 * 100 等于多少。")
    print("=== 全部 messages ===")
    for m in out["messages"]:
        cls = m.__class__.__name__
        if hasattr(m, "tool_calls") and m.tool_calls:
            print(f"[{cls}] tool_calls={[(c['name'], c['args']) for c in m.tool_calls]}")
        else:
            print(f"[{cls}] {m.content[:200]}")
```

- [ ] **Step 2：手动运行，肉眼验收**

```bash
uv run python 04-tool-calling/main.py
```

预期 stdout 大致：

```
=== 全部 messages ===
[HumanMessage] 帮我查一下 NVDA 的股价，再帮我算 925.30 * 100 等于多少。
[AIMessage] tool_calls=[('get_quote', {'symbol': 'NVDA'})]
[ToolMessage] {'symbol': 'NVDA', 'price': 925.3, ...}
[AIMessage] tool_calls=[('calculator', {'expr': '925.30 * 100'})]
[ToolMessage] 92530.0
[AIMessage] NVDA 当前价格 925.30，乘以 100 等于 92530.0... （以上为研究信息汇总，仅供参考...）
```

如果 LLM 一口气返回多个 tool_calls 也合理（并行调用），ToolNode 会一次性全部执行。

- [ ] **Step 3：写 smoke test `tests/test_module_04.py`**

```python
"""模块 04 smoke test：用 FakeMessagesListChatModel 预制工具调用序列。"""
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


def test_no_tool_calls_ends_immediately():
    """模型直接回答，tools_condition 应当跳到 END。"""
    main = _load("04-tool-calling")
    fake = FakeMessagesListChatModel(responses=[AIMessage(content="夏普比率衡量风险调整后收益。")])
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

    fake = FakeMessagesListChatModel(responses=[first, second])
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
    fake = FakeMessagesListChatModel(responses=[first, second])
    out = main.run("1+2*3", llm=fake)
    tool_msg = next(m for m in out["messages"] if m.__class__.__name__ == "ToolMessage")
    assert "7" in tool_msg.content
```

- [ ] **Step 4：跑 smoke test，确认 pass**

```bash
uv run pytest tests/test_module_04.py -v
```

预期：3 passed。

- [ ] **Step 5：写讲义 `04-tool-calling/README.md`**

````markdown
# 模块 04：Tool Calling

## 1. 本模块要解决什么问题

让 LLM **自己决定** 何时去查股价、何时去算数字。这是 ReAct 模式的核心机制——LLM 在回答途中输出"我要调 get_quote(NVDA)"，框架去执行，把结果再喂回 LLM，LLM 拿到结果继续回答。

InvestBot 进度：现在它能"动手"了。问"NVDA 多少钱"，它会自己调 `get_quote`；问"925 * 100 是多少"，它会自己调 `calculator`。

## 2. 核心概念：工具调用协议

OpenAI 兼容的 chat 模型（包括火山方舟、Doubao）都支持这套协议：

1. **客户端**告诉模型："你可以用这些工具"（json schema），靠 `bind_tools` 完成
2. **模型**返回的 `AIMessage` 中带有 `tool_calls` 字段：`[{name, args, id}]`
3. **客户端**实际执行工具，把结果包成 `ToolMessage(tool_call_id=id, content=result)` 加回对话
4. **模型**看到 `ToolMessage`，要么再调一次工具（多轮），要么给最终答复

LangGraph 把第 1、3 步分别封装为 `bind_tools` 和 `ToolNode`，第 4 步是再次调 `call_model`，循环直到模型不再返回 `tool_calls`。

```
            START
              │
              ▼
        ┌──────────┐
        │call_model│ ← LLM 回 AIMessage（可能带 tool_calls）
        └─────┬────┘
              │
        tools_condition
         ┌────┴────┐
         │         │
       "tools"  "__end__"
         │         │
         ▼         ▼
      ┌─────┐    END
      │tools│ ← ToolNode 执行所有 tool_calls
      └──┬──┘
         │
         └──→ 回 call_model（再让 LLM 看工具结果）
```

## 3. 关键 API

| API | 一句话 |
|---|---|
| `llm.bind_tools(tools)` | 把 list[BaseTool] 的 JSON schema 注入到 LLM 调用，返回新 Runnable |
| `from langgraph.prebuilt import ToolNode` | 节点：执行 state.messages[-1] 中的所有 tool_calls，返回 ToolMessage 列表 |
| `from langgraph.prebuilt import tools_condition` | 路由函数：返回 `"tools"` 或 `"__end__"`；mapping 不写时用默认 |
| `AIMessage(content="", tool_calls=[{name, args, id}])` | 测试时的"假工具调用"形态 |
| `ToolMessage(content, tool_call_id)` | 工具执行结果的消息形态（ToolNode 自动构造） |

## 4. 代码导读

- `TOOLS = [get_quote, calculator]`：本模块只用两个不依赖网络的工具
- `make_call_model(llm_with_tools)`：节点工厂——把 SYSTEM_BASE 拼前面再调 LLM，返回 AIMessage
- `g.add_node("tools", ToolNode(TOOLS))`：**节点名必须叫 "tools"**，因为 `tools_condition` 默认 mapping 把 `"tools"` 字符串映射到这个节点
- `g.add_conditional_edges("call_model", tools_condition)`：不传 mapping 用默认（`{"tools": "tools", "__end__": END}`）
- `g.add_edge("tools", "call_model")`：循环回模型——这是 ReAct "再思考"的环

## 5. 如何运行

```bash
# .env 须有 ARK_API_KEY + LLM_MODEL
uv run python 04-tool-calling/main.py
```

观察输出：你会看到 messages 列表里出现 `tool_calls` → `ToolMessage` → 最终 `AIMessage` 的完整链。这就是 LLM 在"读到"工具结果后才能给出准确价格。

## 6. 常见坑

1. **模型不调工具**：可能是 prompt 没让它意识到"该用工具"，或工具 docstring 写得太抽象。**docstring = 工具的 prompt**——`@tool` 把 docstring 暴露给模型作为描述
2. **`tool_calls` 是 list**：模型可以并行调多个工具（一次返回多个 tool_calls），ToolNode 会一次性全部执行
3. **节点名 "tools"**：写成 "tool"、"toolbox" 都不行（除非自己传 mapping）；保持默认更省事
4. **AIMessage(content="", tool_calls=[...])**：调工具时 content 为空（或简短解释）。测试时若忘记设 content="" 会让 tools_condition 误判
5. **死循环**：模型如果反复调同一个工具，会撞 `recursion_limit`（默认 25）。生产环境记得加上限保护或在 prompt 里说"如已查到结果，不要再重复调"
6. **bind_tools 返回的是新对象**：`llm_with_tools = llm.bind_tools(...)` 不修改 llm；忘记接返回值会导致工具未挂上
7. **ToolMessage 里的内容**：是 `str(tool_return_value)`，dict 会被序列化成 JSON 风格的 str；如果你工具返回复杂结构，模型可能看不懂——保持工具返回扁平化更安全

## 7. 小练习

1. 把 `search_web` 也加到 `TOOLS`，问"特斯拉最新财报"，看 LLM 是否会主动调网络搜索
2. 给 `tools_condition` 自己写一个 mapping，把 "tools" 改名成 "exec_tools"，看怎么改图才能跑通
3. 在 `call_model` 里加一行 `print(resp.tool_calls)`，可以直观看到模型每次"想干什么"
4. 故意让 `get_quote` 抛异常，看 ToolNode 的 ToolMessage 中怎么报告错误（默认会 catch 并塞 error 字段进 content）
````

- [ ] **Step 6：commit**

```bash
git add 04-tool-calling/main.py 04-tool-calling/README.md tests/test_module_04.py
git commit -m "feat(04): tool-calling — bind_tools / ToolNode / tools_condition loop"
```

---

### Task 1.3：模块 05 react-loop

**Files:**
- Create: `05-react-loop/main.py`、`05-react-loop/README.md`、`tests/test_module_05.py`

**学习目标：**
- 用 `create_react_agent` 一行搭出与 04 等价的 ReAct agent
- 把 04 的"手写图"作为低阶对比版（同个文件里两个 build 函数），逐行标注对应关系
- 理解 `prompt=` 参数（替代旧版 `messages_modifier`）
- 第一次接入完整 `ALL_TOOLS`（包括 search_web、save_note、get_fundamentals）

- [ ] **Step 1：写 `05-react-loop/main.py`**

```python
"""模块 05：ReAct Loop —— 高阶封装与低阶对比

create_react_agent(model, tools, prompt=...) 一行搭出 ReAct agent。
本文件同时给出"手写低阶版"（与 04 同结构）作为对比，让你明白封装内部到底做了什么。

为充分演示，本模块挂上 ALL_TOOLS（5 个工具，含 Tavily 真实联网）。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import ALL_TOOLS


# ====== 高阶版：create_react_agent ======

def build_high_level(llm: BaseChatModel | None = None):
    """一行版本。

    create_react_agent 内部做了：
    1. llm.bind_tools(tools)
    2. 节点 "agent"（即 04 的 call_model）
    3. 节点 "tools" = ToolNode(tools)
    4. add_conditional_edges("agent", tools_condition)
    5. add_edge("tools", "agent") 形成循环
    几乎就是 04 的图，prompt 参数自动拼到对话最前面。
    """
    if llm is None:
        llm = get_llm(temperature=0)
    return create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_BASE)


# ====== 低阶版：手写（与 04 等价，只是工具集换成 ALL_TOOLS） ======

def build_low_level(llm: BaseChatModel | None = None):
    """手写版——与 build_high_level 行为一致，让你看清内部结构。"""
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def call_model(state: InvestBotState) -> dict:
        from langchain_core.messages import SystemMessage
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    g = StateGraph(InvestBotState)
    g.add_node("agent", call_model)             # 对应 create_react_agent 内的 "agent" 节点
    g.add_node("tools", ToolNode(ALL_TOOLS))    # 对应 "tools" 节点
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ====== 共用入口 ======

def run(user_text: str, mode: str = "high", llm: BaseChatModel | None = None) -> dict:
    """mode='high' 用 create_react_agent；mode='low' 用手写版。两者输出 state 形态一致。"""
    if mode == "high":
        app = build_high_level(llm=llm)
    elif mode == "low":
        app = build_low_level(llm=llm)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return app.invoke({"messages": [HumanMessage(content=user_text)]})


if __name__ == "__main__":
    question = "帮我查 AAPL 的报价和基本面，然后把要点写成笔记保存。"
    for mode in ("high", "low"):
        print(f"\n========== mode = {mode} ==========")
        out = run(question, mode=mode)
        for m in out["messages"]:
            cls = m.__class__.__name__
            if hasattr(m, "tool_calls") and m.tool_calls:
                print(f"[{cls}] tool_calls={[(c['name'], c['args']) for c in m.tool_calls]}")
            else:
                content = m.content if isinstance(m.content, str) else str(m.content)
                print(f"[{cls}] {content[:200]}")
```

- [ ] **Step 2：手动运行，肉眼验收**

```bash
uv run python 05-react-loop/main.py
```

预期：两个 mode 都跑通，最终都会有 AIMessage 给出 AAPL 的报价 + 基本面 + "已保存笔记到 ..." 的总结。两个 mode 的 messages 序列结构应当**几乎一致**（细节可能因 LLM 不确定性略有不同）。

> 如果 ARK_API_KEY 配了但 TAVILY_API_KEY 没配，并且 LLM 选择调用 search_web 也没关系——`get_quote` + `get_fundamentals` + `save_note` 已足以回答这个问题，模型很可能不会调 search_web。

- [ ] **Step 3：写 smoke test `tests/test_module_05.py`**

```python
"""模块 05 smoke test：验证 high / low 两种构建方式行为等价。"""
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


def _scripted_responses():
    """预制：先调 get_quote，再给最终答复。两个 mode 共用同一脚本。"""
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "c1"}],
        ),
        AIMessage(content="AAPL 当前 188.14 美元。"),
    ]


def test_high_level_runs_react_loop():
    main = _load("05-react-loop")
    fake = FakeMessagesListChatModel(responses=_scripted_responses())
    out = main.run("AAPL 多少钱", mode="high", llm=fake)
    classes = [m.__class__.__name__ for m in out["messages"]]
    assert "ToolMessage" in classes
    assert out["messages"][-1].content == "AAPL 当前 188.14 美元。"


def test_low_level_runs_react_loop():
    main = _load("05-react-loop")
    fake = FakeMessagesListChatModel(responses=_scripted_responses())
    out = main.run("AAPL 多少钱", mode="low", llm=fake)
    classes = [m.__class__.__name__ for m in out["messages"]]
    assert "ToolMessage" in classes
    assert out["messages"][-1].content == "AAPL 当前 188.14 美元。"


def test_high_and_low_produce_same_message_classes():
    """两个版本的 messages 序列在结构上应当一致。"""
    main = _load("05-react-loop")
    out_h = main.run("AAPL 多少钱", mode="high", llm=FakeMessagesListChatModel(responses=_scripted_responses()))
    out_l = main.run("AAPL 多少钱", mode="low", llm=FakeMessagesListChatModel(responses=_scripted_responses()))
    cls_h = [m.__class__.__name__ for m in out_h["messages"]]
    cls_l = [m.__class__.__name__ for m in out_l["messages"]]
    assert cls_h == cls_l
```

- [ ] **Step 4：跑 smoke test，确认 pass**

```bash
uv run pytest tests/test_module_05.py -v
```

预期：3 passed。

- [ ] **Step 5：写讲义 `05-react-loop/README.md`**

````markdown
# 模块 05：ReAct Loop —— 高阶封装与低阶对比

## 1. 本模块要解决什么问题

模块 04 已经手写了完整的 ReAct 循环。LangGraph 把这个套路封装成 `create_react_agent`，**一行**就能搭出来。本模块同时提供两种构建方式：

- `build_high_level()`：一行，`create_react_agent(model, tools, prompt)`
- `build_low_level()`：手写，与 04 同结构（`call_model + ToolNode + tools_condition`）

放一起对比看，你就明白"高阶 API 帮你做了什么"——既能用快捷方式，又不被黑盒卡住。

InvestBot 进度：现在它的工具集是完整 5 个（包括真 Tavily 联网搜索 + 笔记落盘），具备一个能用的研究助手雏形。

## 2. 核心概念：create_react_agent 内部做了什么

```python
agent = create_react_agent(model=llm, tools=ALL_TOOLS, prompt=SYSTEM_BASE)
```

等价于：

```python
llm_with_tools = llm.bind_tools(ALL_TOOLS)

def call_model(state):
    msgs = [SystemMessage(SYSTEM_BASE)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(msgs)]}

g = StateGraph(MessagesState)
g.add_node("agent", call_model)
g.add_node("tools", ToolNode(ALL_TOOLS))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
agent = g.compile()
```

读完这段你会发现：**它就是 04 模块的图**。区别只在节点叫 `"agent"` 而不是 `"call_model"`，仅此而已。

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.prebuilt import create_react_agent` | 一行 ReAct agent 工厂 |
| `create_react_agent(model, tools, prompt=...)` | model = BaseChatModel；tools = list；prompt = str / SystemMessage / Callable |
| `agent.invoke({"messages": [...]})` | 与手写图一样调用 |
| 高阶 vs 低阶 trade-off | 高阶：少 5 行；低阶：可以在循环中插自定义节点（重写、防御、统计） |

## 4. 代码导读

- `build_high_level`：8 行（含注释）。生产代码默认用这个就够
- `build_low_level`：把"高阶帮你做的事"逐行展开。**这两个版本的图结构是同构的**，节点名稍有差异（`"agent"` vs `"agent"`，巧合一致）
- `run(mode=...)`：把入口拍平，方便 main 区分 demo

## 5. 如何运行

```bash
# .env 须有 ARK_API_KEY + LLM_MODEL；若想测真 Tavily 联网，再加 TAVILY_API_KEY
uv run python 05-react-loop/main.py
```

输出会有两个 section（high / low），分别打印各自的 messages 序列，方便肉眼对比。结构应当几乎一致。

## 6. 常见坑

1. **`prompt=` 参数旧名 `messages_modifier`**：早期 LangGraph 0.x 教程里你可能见过 `messages_modifier=...`，1.x 已改 `prompt=`。本仓库用 1.1.x，认准 `prompt`
2. **prompt 类型**：可以是 str（SystemMessage 形式）、SystemMessage、Callable（动态生成）。本模块用 str
3. **state schema 不同**：`create_react_agent` 默认用内置 `AgentState`（继承 MessagesState），不是 `InvestBotState`；如果你需要 `last_intent` 这类自定义字段，要传 `state_schema=` 参数
4. **何时用低阶手写**：①循环中要插"审核"节点（如内容过滤）；②工具结果想做后处理；③要并行多 agent；④要在循环里写 metrics。否则高阶更短
5. **真 LLM + 真 Tavily**：跑 `main.py` 时可能消耗 1-2 次 LLM 调用 + 1 次 Tavily 搜索。预算需要心里有数
6. **递归保护**：`create_react_agent` 出来的 graph 仍受默认 `recursion_limit=25` 约束；invoke 时传 `config={"recursion_limit": 50}` 可调高

## 7. 小练习

1. 把 `prompt` 改成一个 Callable：`lambda state: [SystemMessage("...")] + state["messages"][-3:]`，让 agent 只看最近 3 条上下文
2. 在 `build_low_level` 里加一个 `validate_node`，在 `call_model → ToolNode` 之间过一道——如果 LLM 想调 `save_note` 但 title 为空，跳过工具执行
3. 给 `create_react_agent` 加 `response_format=` 让它输出强类型 JSON，对比手写版要做多少改动
4. 对比 `agent.get_graph().draw_mermaid()` 和你手写图的可视化，确认结构一致
````

- [ ] **Step 6：commit**

```bash
git add 05-react-loop/main.py 05-react-loop/README.md tests/test_module_05.py
git commit -m "feat(05): react-loop — create_react_agent + hand-rolled equivalent"
```

---

### Task F.1：更新仓库总览 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1：把"学习路径"区块的坐 2 三个模块从"待生成"改成可点击的链接**

定位 `README.md` 中"### 坐 2"那块，替换为：

```markdown
### 坐 2：接入 LLM 与工具
- [03-routing-and-llm](03-routing-and-llm/) — 第一次调真 LLM，做意图路由
- [04-tool-calling](04-tool-calling/) — `bind_tools` / `ToolNode` / `tools_condition` 三件套
- [05-react-loop](05-react-loop/) — `create_react_agent` 一行起飞，并对比手写低阶版
```

- [ ] **Step 2：commit**

```bash
git add README.md
git commit -m "docs: update README — link plan 2 modules"
```

---

### Task F.2：Plan 2 全量验证 + 整体收尾

- [ ] **Step 1：跑全部测试一次**

```bash
uv run pytest -v
```

预期：
- Plan 1 部分（共 21 passed，含 conftest 引入的 1 个）
- `tests/test_module_03.py` 5 passed
- `tests/test_module_04.py` 3 passed
- `tests/test_module_05.py` 3 passed
- 合计 32 passed，0 failed

如有 fail：先停下定位根因。fake LLM 的 tool_calls 形态、`tools_condition` 的 mapping、`create_react_agent` 的 prompt 参数都是常见摔跤点。

- [ ] **Step 2：跑三个学习模块的 main.py 各一次（需要真 ARK_API_KEY）**

```bash
uv run python 03-routing-and-llm/main.py
uv run python 04-tool-calling/main.py
uv run python 05-react-loop/main.py
```

预期：每条命令均成功退出（exit code 0）。05 可能慢一些（多轮 LLM 调用 + Tavily 搜索）。

> 如果还没配 ARK key 就跳过这步，让 smoke test 替代验收（已 mock 不联网）。但**强烈建议**至少跑一次真 LLM，让你直观感受"真模型 + 真工具循环"的形态——这是教程的灵魂。

- [ ] **Step 3：检查 git 状态干净**

```bash
git status
```

预期：`nothing to commit, working tree clean`。

- [ ] **Step 4：确认 Plan 2 范围内的所有 spec 项已被覆盖**

对照 `docs/superpowers/specs/2026-04-26-langgraph-investbot-tutorial-design.md` 中"坐 2"部分：

- 模块 03（routing）✅ Task 1.1
- 模块 04（tool-calling）✅ Task 1.2
- 模块 05（react-loop）✅ Task 1.3
- 跨模块约定（InvestBotState 作基类）✅ 03 模块首次实践
- 测试隔离基础（conftest）✅ Task 0.1

模块 06–10、x2 不在 Plan 2 范围，留给 Plan 3 / 4。

- [ ] **Step 5：打 tag 标记 Plan 2 完成**

```bash
git tag plan-2-complete
git log --oneline | head -25
```

最近 commit 序列大致：

```
docs: update README — link plan 2 modules
feat(05): react-loop — create_react_agent + hand-rolled equivalent
feat(04): tool-calling — bind_tools / ToolNode / tools_condition loop
feat(03): routing-and-llm — first real LLM call for intent classification
test: add conftest to reset tavily client between tests
docs: address final review I-1/I-2/I-3 — clarify reducer/state/dotenv semantics
docs: add repo overview README with learning path
feat(x1): pure-routing — conditional edges and loops without LLM
feat(02): state-and-reducer — add_messages vs plain field
feat(01): hello-graph — minimal StateGraph with single echo node
... (Plan 1 部分)
```

---

## Plan 2 完成后的状态

仓库具备：

- 三个新增学习模块（03 / 04 / 05），各带 README + main.py + smoke test
- 测试公共设施（`tests/conftest.py`）解决 Tavily client 跨测试粘连
- "依赖注入 build_graph(llm=None)" 模式贯穿三个模块——既隔离测试，又顺便教学解耦
- 32 个测试全 passed
- InvestBot 已经能：
  - 用 LLM 做意图分类（4 类路由）
  - 让 LLM 自己决定何时 / 调哪个工具
  - 用 `create_react_agent` 一行搭起完整 ReAct agent

学员到这里应当能：

- 解释"工具调用协议"在 OpenAI/方舟接口层是怎么传递的（`tool_calls` 字段、`tool_call_id`、`ToolMessage`）
- 说出 `create_react_agent` 内部展开的 5 步骤
- 选择何时用高阶 / 何时手写（trade-off：拓展点 vs 简洁）
- 自己接入新工具：写 `@tool` 函数 → 加进 list → `bind_tools` → 跑通

---

## 下一步：Plan 3（坐 3：让 Agent 真正能用）

Plan 2 完工并验证后，会接着生成 **Plan 3**，覆盖：

- 模块 06 persistence：`MemorySaver` / `SqliteSaver` / `thread_id`，让对话能跨调用记忆
- 模块 07 human-in-the-loop：`interrupt()` / `Command(resume=...)`，让人类在敏感点（比如 save_note 前）介入审核
- 模块 08 streaming：`graph.stream()` / `astream()`，把"逐步生成"讲透——这是 InvestBot 走向产品体验的关键

到那时 InvestBot 才真正"能用"——不会忘事、能审核、能流式输出。
