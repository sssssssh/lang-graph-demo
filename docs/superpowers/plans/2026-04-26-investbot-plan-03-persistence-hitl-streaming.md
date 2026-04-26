# InvestBot Plan 3：坐 3 —— 让 Agent 真正能用

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 InvestBot 从"能跑"变成"能用"——记得对话历史（persistence）、关键动作前能让人类确认（human-in-the-loop）、回答时逐步流式输出（streaming）。覆盖三个学习模块（06 persistence、07 human-in-the-loop、08 streaming）。

**Prerequisites:** Plan 2 完成（tag `plan-2-complete`），32 测试通过；`common/` 公共模块、5 个工具、`InvestBotState` 已就绪。

**Architecture:** 沿用前两份 Plan 的目录布局。本 Plan 三个模块都要复用 04 模块手写的 ReAct 循环作为基础（"call_model + ToolNode + tools_condition"），叠加新特性来突出每一节的学习点：

- 06：在 04 ReAct 图基础上 `compile(checkpointer=...)` + `config={"configurable": {"thread_id": ...}}`
- 07：在 04 ReAct 图基础上插入一个 `confirm_node`，用 `interrupt()` 暂停，用 `Command(resume=...)` 恢复
- 08：在 04 ReAct 图基础上演示 `graph.stream(stream_mode="updates" | "values")`，再附一份 `astream(stream_mode="messages")` 演示 token 级流式

**Tech Stack:** Python 3.11+、langgraph 1.1.x、`langgraph.checkpoint.memory.MemorySaver`、`langgraph.checkpoint.sqlite.SqliteSaver`、`langgraph.types.interrupt / Command`。SqliteSaver 是 **context manager**（`with SqliteSaver.from_conn_string(...) as saver:`），不是构造器返回普通对象——这是常见摔跤点。

**Plan 范围（本份只覆盖 Plan 3）：**

- Phase 1：模块 06 persistence
- Phase 2：模块 07 human-in-the-loop
- Phase 3：模块 08 streaming
- Phase F：收尾 + tag

模块 09 / x2 / 10 不在 Plan 3，留给 Plan 4。

---

## 文件结构

```
06-persistence/
├── README.md
└── main.py
07-human-in-the-loop/
├── README.md
└── main.py
08-streaming/
├── README.md
└── main.py
tests/
├── test_module_06.py
├── test_module_07.py
└── test_module_08.py
```

---

## 任务列表

---

### Task 1.1：模块 06 persistence

**Files:** Create: `06-persistence/main.py`、`06-persistence/README.md`、`tests/test_module_06.py`

**学习目标：**
- 理解 LangGraph 的 checkpointer 机制：每次节点执行后 state 自动保存
- `MemorySaver`（进程内）vs `SqliteSaver`（落盘）——场景区别
- `config={"configurable": {"thread_id": "..."}}` 是怎么决定"是哪段对话"的
- 同 thread_id 跨多次 invoke 能记忆，不同 thread_id 互不影响

- [ ] **Step 1：写 `06-persistence/main.py`**

```python
"""模块 06：Persistence

让 InvestBot 跨多次 invoke "记得"对话。

核心思路：compile(checkpointer=...) 让 LangGraph 在每个节点执行后自动保存 state；
再次 invoke 时，传同一个 thread_id 就会从最新 checkpoint 接着跑。

本模块演示 MemorySaver（进程内）+ SqliteSaver（落盘）两种 checkpointer。
"""
from contextlib import contextmanager
from typing import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote

TOOLS = [get_quote, calculator]


def _make_call_model(llm_with_tools: BaseChatModel):
    def call_model(state: InvestBotState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    return call_model


def build_graph(llm: BaseChatModel | None = None, checkpointer=None):
    """构造带 checkpointer 的 ReAct 图。

    checkpointer 在 compile() 时挂载，决定 state 如何持久化：
    - None：每次 invoke 独立、不记忆
    - MemorySaver()：进程内字典存，重启就丢
    - SqliteSaver(...)：落盘 sqlite，跨进程可恢复
    """
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    g = StateGraph(InvestBotState)
    g.add_node("call_model", _make_call_model(llm_with_tools))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    return g.compile(checkpointer=checkpointer)


@contextmanager
def sqlite_checkpointer(db_path: str) -> Iterator[SqliteSaver]:
    """SqliteSaver 是 context manager；这里再包一层方便外部 with 调用。

    db_path 可写 ":memory:" 做临时演示，也可传文件路径做持久化。
    """
    with SqliteSaver.from_conn_string(db_path) as saver:
        yield saver


def chat(app, user_text: str, thread_id: str) -> dict:
    """对同一个 app 多次 invoke 的便捷封装：传同 thread_id 就会接着上次跑。"""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)


if __name__ == "__main__":
    # === 演示 1：MemorySaver，同 thread_id 多轮记忆 ===
    print("--- Demo 1: MemorySaver, same thread_id ---")
    app = build_graph(checkpointer=MemorySaver())
    out1 = chat(app, "我刚买了 NVDA 100 股", thread_id="demo")
    print("AI:", out1["messages"][-1].content[:80], "...")

    out2 = chat(app, "你还记得我买了什么吗？", thread_id="demo")
    print("AI:", out2["messages"][-1].content[:80], "...")
    print(f"messages 累计 = {len(out2['messages'])}\n")

    # === 演示 2：换 thread_id，记忆隔离 ===
    print("--- Demo 2: different thread_id ---")
    out3 = chat(app, "你还记得我买了什么吗？", thread_id="other")
    print("AI:", out3["messages"][-1].content[:80], "...")
    print(f"messages 累计 = {len(out3['messages'])}（仅本次的 1 user + 1 ai）\n")

    # === 演示 3：SqliteSaver，落盘可恢复 ===
    print("--- Demo 3: SqliteSaver in-memory db ---")
    with sqlite_checkpointer(":memory:") as saver:
        app2 = build_graph(checkpointer=saver)
        chat(app2, "我刚买了 AAPL 50 股", thread_id="t1")
        out = chat(app2, "你还记得我买了什么吗？", thread_id="t1")
        print("AI:", out["messages"][-1].content[:80], "...")
```

- [ ] **Step 2：手动运行（需要 ARK_API_KEY）**

```bash
uv run python 06-persistence/main.py
```

预期：Demo 1 第二轮的 AI 应当回答出 NVDA；Demo 2 换 thread_id 后 AI 不知道；Demo 3 用 SqliteSaver 行为与 MemorySaver 一致。

- [ ] **Step 3：写 smoke test `tests/test_module_06.py`**

```python
"""模块 06 smoke test：验证 thread_id 的隔离 + 记忆累加。"""
import sys
import importlib.util
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


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


def test_same_thread_id_accumulates_messages():
    """同一 thread_id 跨两次 invoke，messages 应当累加。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="ack 1"), AIMessage(content="ack 2")]
    )
    app = main.build_graph(llm=fake, checkpointer=MemorySaver())

    out1 = main.chat(app, "first", thread_id="t1")
    out2 = main.chat(app, "second", thread_id="t1")

    assert len(out1["messages"]) == 2  # 1 user + 1 ai
    assert len(out2["messages"]) == 4  # 累加：2 user + 2 ai


def test_different_thread_ids_are_isolated():
    """不同 thread_id 互不影响。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="t1-1"), AIMessage(content="t2-1")]
    )
    app = main.build_graph(llm=fake, checkpointer=MemorySaver())

    out_t1 = main.chat(app, "in t1", thread_id="t1")
    out_t2 = main.chat(app, "in t2", thread_id="t2")

    assert len(out_t1["messages"]) == 2
    assert len(out_t2["messages"]) == 2  # 不会变成 4
    assert out_t2["messages"][0].content == "in t2"


def test_no_checkpointer_means_no_memory():
    """compile 不传 checkpointer 时，state 不会跨 invoke 持久化。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="ack 1"), AIMessage(content="ack 2")]
    )
    app = main.build_graph(llm=fake)  # checkpointer=None

    out1 = main.chat(app, "first", thread_id="t1")
    out2 = main.chat(app, "second", thread_id="t1")
    assert len(out1["messages"]) == 2
    assert len(out2["messages"]) == 2  # 没有累加


def test_sqlite_checkpointer_works_like_memory():
    """SqliteSaver 应当与 MemorySaver 行为一致。"""
    main = _load("06-persistence")
    fake = FakeChatModelWithTools(
        responses=[AIMessage(content="ack 1"), AIMessage(content="ack 2")]
    )
    with main.sqlite_checkpointer(":memory:") as saver:
        app = main.build_graph(llm=fake, checkpointer=saver)
        out1 = main.chat(app, "first", thread_id="t1")
        out2 = main.chat(app, "second", thread_id="t1")
    assert len(out2["messages"]) == 4
```

- [ ] **Step 4：跑 smoke test**

```bash
uv run pytest tests/test_module_06.py -v
```

预期：4 passed。

- [ ] **Step 5：写 `06-persistence/README.md`**（按 7 节标准）

````markdown
# 模块 06：Persistence

## 1. 本模块要解决什么问题

到现在为止 InvestBot 的对话是"金鱼记忆"——每次 invoke 都从空白开始。LangGraph 的 checkpointer 让 state **自动持久化**：每次节点执行完都拍一张快照，下次传同一个 thread_id 就能接着上次跑。这是 LangGraph 比"裸调 OpenAI API"最大的工程价值之一。

InvestBot 进度：终于不再失忆，能开始多轮研究对话。

## 2. 核心概念

```
compile(checkpointer=MemorySaver()) ──┐
                                       │
            invoke(input, config={"configurable": {"thread_id": "X"}})
                                       │
            ┌─────────── thread "X" 的 state 历史 ────────────┐
            │ checkpoint 1 → checkpoint 2 → checkpoint 3 ... │
            └─────────────────────────────────────────────────┘
```

- **Checkpointer**：state 的存储后端。`MemorySaver`（dict 内存）/ `SqliteSaver`（sqlite 文件）/ `PostgresSaver`（生产）
- **thread_id**：一段对话的"身份证"。同 id = 接着上次；新 id = 空白开始
- **每个节点执行后自动 save**：你不需要手动调任何 save，框架自己做

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.checkpoint.memory import MemorySaver` | 进程内 dict，重启即丢 |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | sqlite 落盘；**是 context manager**，须 `with SqliteSaver.from_conn_string(...) as saver:` |
| `graph.compile(checkpointer=saver)` | 编译时挂载 |
| `app.invoke(input, config={"configurable": {"thread_id": "X"}})` | 调用时指定是哪段对话 |
| `app.get_state(config)` | 取当前 state 快照（含 messages） |
| `app.get_state_history(config)` | 列出所有历史 checkpoint，支持时间旅行 |

## 4. 代码导读

- `build_graph(checkpointer=None)`：把 checkpointer 作为参数；不传时图是无记忆的（与 04 行为一致）
- `sqlite_checkpointer(db_path)`：把 SqliteSaver 的 context-manager 用法包成 `@contextmanager`，调用方写 `with sqlite_checkpointer(...) as saver:` 拿到普通 saver
- `chat(app, user_text, thread_id)`：构造 `config={"configurable": {"thread_id": ...}}` 的便捷封装

## 5. 如何运行

```bash
uv run python 06-persistence/main.py
```

观察三段 demo 的输出。Demo 1 应当能让 AI"想起" NVDA；Demo 2 换 thread_id 后失忆；Demo 3 用 SqliteSaver 行为一致。

## 6. 常见坑

1. **忘记传 thread_id**：会报"checkpointer 已挂载但缺 thread_id"。`config["configurable"]["thread_id"]` 是必填
2. **SqliteSaver 不是普通构造**：`SqliteSaver(":memory:")` 不工作；必须 `with SqliteSaver.from_conn_string(":memory:") as saver: ...`
3. **persistence 不等于 messages 自己累加**：累加靠的是 `add_messages` reducer；persistence 是把整份 state 跨次保存。两者配合才有"对话记忆"
4. **`thread_id` 重名风险**：不同用户用同一 thread_id 会"撞车"。生产中常见做法是 `thread_id = f"{user_id}:{conversation_id}"`
5. **MemorySaver 在多进程下失效**：每个进程一份内存——多进程要用 SqliteSaver / PostgresSaver
6. **state 太大**：每个 checkpoint 都全量保存 state；如果 messages 列表越积越长，存储与序列化都会变慢，生产中要做 trim / summary

## 7. 小练习

1. 改 sqlite 的 db_path 为真实文件如 `./checkpoint.sqlite`，跑两次 main.py，体会"重启后还能继续"
2. 调 `app.get_state(config)` 看 state 形态；调 `app.get_state_history(config)` 看历史
3. 给 `chat` 加参数 `replay_from: int`，演示从历史第 N 个 checkpoint "重放"对话
````

- [ ] **Step 6：commit**

```bash
git add 06-persistence/ tests/test_module_06.py
git commit -m "feat(06): persistence — MemorySaver / SqliteSaver / thread_id"
```

---

### Task 1.2：模块 07 human-in-the-loop

**Files:** Create: `07-human-in-the-loop/main.py`、`07-human-in-the-loop/README.md`、`tests/test_module_07.py`

**学习目标：**
- 在敏感动作（如 `save_note` 写盘）前 `interrupt()` 暂停图
- 用 `Command(resume=value)` 把用户决策喂回图
- 理解 interrupt 与 checkpointer 的依赖关系（HITL 必须配 checkpointer）

- [ ] **Step 1：写 `07-human-in-the-loop/main.py`**

```python
"""模块 07：Human-in-the-loop

在 LLM 想保存研究笔记前 interrupt() 暂停，让人决定 approve / reject。
关键 API：
- interrupt(value)：在节点里调，会抛出 Interrupt，把 value 暴露给调用方
- Command(resume=...)：调用方决定后，再 invoke 时把决策喂回去
- HITL 必须配 checkpointer——没有 checkpoint 就没"暂停点"
"""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from common.state import InvestBotState


def propose_save_node(state: InvestBotState) -> dict:
    """模拟 LLM 决定要保存笔记，把"建议保存"写进 messages。"""
    return {"messages": [AIMessage(content="我建议把今天的研究保存为笔记《NVDA 速览》。")]}


def confirm_node(state: InvestBotState) -> Command[Literal["execute_save", "abort"]]:
    """interrupt 节点：暂停图，把"是否保存"问题抛给人。

    返回 Command(goto=...) 显式指定下一个节点（基于人类决策）。
    """
    decision = interrupt({"action": "approve_save_note", "preview": "NVDA 速览"})
    # interrupt 返回值就是调用方 Command(resume=...) 传进来的 value
    if decision == "approve":
        return Command(goto="execute_save")
    return Command(goto="abort")


def execute_save_node(state: InvestBotState) -> dict:
    return {"messages": [AIMessage(content="✅ 笔记已保存（mock）。")]}


def abort_node(state: InvestBotState) -> dict:
    return {"messages": [AIMessage(content="❌ 已取消保存。")]}


def build_graph(checkpointer=None):
    """HITL 必须挂 checkpointer 才能在 interrupt 处暂停后重启。"""
    if checkpointer is None:
        checkpointer = MemorySaver()

    g = StateGraph(InvestBotState)
    g.add_node("propose_save", propose_save_node)
    g.add_node("confirm", confirm_node)
    g.add_node("execute_save", execute_save_node)
    g.add_node("abort", abort_node)

    g.add_edge(START, "propose_save")
    g.add_edge("propose_save", "confirm")
    # confirm_node 用 Command(goto=...) 自己决定下一步，无需额外 conditional edge
    g.add_edge("execute_save", END)
    g.add_edge("abort", END)

    return g.compile(checkpointer=checkpointer)


def run_until_interrupt(app, user_text: str, thread_id: str):
    """跑到 interrupt 处暂停，返回 interrupt payload。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
    # __interrupt__ 在 result 中（v1.x）；也可以从 app.get_state(config).interrupts 取
    return result, config


def resume(app, decision: str, config: dict):
    """用 Command(resume=...) 把人类决策喂回去，继续跑到 END。"""
    return app.invoke(Command(resume=decision), config=config)


if __name__ == "__main__":
    # === 演示 1：approve ===
    print("--- Demo 1: approve ---")
    app = build_graph()
    paused, config = run_until_interrupt(app, "今天看了 NVDA", thread_id="t1")
    print("interrupt payload:", paused.get("__interrupt__"))
    final = resume(app, "approve", config)
    print("最终 message:", final["messages"][-1].content)

    # === 演示 2：reject ===
    print("\n--- Demo 2: reject ---")
    app2 = build_graph()
    paused2, config2 = run_until_interrupt(app2, "今天看了 AAPL", thread_id="t2")
    final2 = resume(app2, "reject", config2)
    print("最终 message:", final2["messages"][-1].content)
```

- [ ] **Step 2：手动运行（不需要 ARK_API_KEY，本模块不调真 LLM）**

```bash
uv run python 07-human-in-the-loop/main.py
```

预期：Demo 1 末尾打印"✅ 笔记已保存"；Demo 2 末尾打印"❌ 已取消保存"。

- [ ] **Step 3：写 smoke test `tests/test_module_07.py`**

```python
"""模块 07 smoke test：验证 interrupt 暂停 + Command(resume) 恢复。"""
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


def test_approve_path_executes_save():
    main = _load("07-human-in-the-loop")
    app = main.build_graph()
    _, config = main.run_until_interrupt(app, "go", thread_id="ta")
    final = main.resume(app, "approve", config)
    assert "已保存" in final["messages"][-1].content


def test_reject_path_aborts():
    main = _load("07-human-in-the-loop")
    app = main.build_graph()
    _, config = main.run_until_interrupt(app, "go", thread_id="tr")
    final = main.resume(app, "reject", config)
    assert "取消" in final["messages"][-1].content


def test_interrupt_actually_pauses_before_resume():
    """暂停时不应执行 execute_save / abort 节点；只跑到 confirm。"""
    main = _load("07-human-in-the-loop")
    app = main.build_graph()
    paused, _ = main.run_until_interrupt(app, "go", thread_id="tp")
    contents = [m.content for m in paused.get("messages", [])]
    # propose_save 写了一条；execute / abort 都没跑
    assert any("建议把今天的研究保存" in c for c in contents)
    assert not any("已保存" in c or "取消" in c for c in contents)
```

- [ ] **Step 4：跑 smoke test**

```bash
uv run pytest tests/test_module_07.py -v
```

预期：3 passed。

- [ ] **Step 5：写 `07-human-in-the-loop/README.md`**

````markdown
# 模块 07：Human-in-the-loop

## 1. 本模块要解决什么问题

LLM agent 自由"动手"（写文件、发消息、执行交易）很危险。HITL 让你在关键动作前**暂停**图，把决策权交还给人，用户 approve 之后再继续。这对投资场景尤为重要——任何写盘 / 发出动作都应当可控。

InvestBot 进度：保存笔记前会先问"要不要保存？"，得到 approve 才动手。

## 2. 核心概念

```
START → propose_save → confirm ─── interrupt() ⏸  暂停！
                          │
                  调用方传 Command(resume="approve")
                          │
              恢复后 confirm 取到 "approve"
                          │
                          ▼
                    Command(goto="execute_save") → END
```

- **`interrupt(value)`**：在节点里抛断点，**`value` 会被暴露给调用方**（一般是要审批的操作详情 / 预览）
- **`Command(resume=value)`**：调用方再次 `app.invoke(Command(resume=...))`，框架把 `value` 作为 `interrupt()` 的返回值塞回节点
- **必须配 checkpointer**：没有 checkpoint 就没法暂停（暂停 = 暂存当前 state，等以后再来）
- **`Command(goto=...)`**：节点也可以用 `Command` 而不是 dict 返回——goto 显式控制下一步去哪

## 3. 关键 API

| API | 一句话 |
|---|---|
| `from langgraph.types import interrupt, Command` | HITL 两件套 |
| `interrupt(value)` | 在节点里调；返回值就是 Command(resume=...) 传进来的内容 |
| `Command(resume=value)` | 用 invoke(Command(resume=...)) 喂回 |
| `Command(goto=node_name)` | 节点用 Command 返回时显式控制下一步 |
| `app.get_state(config).interrupts` | 取当前未处理的 interrupt 列表 |

## 4. 代码导读

- `propose_save_node`：模拟 LLM 决定要保存笔记
- `confirm_node`：调 `interrupt(...)` 暂停；恢复后根据 decision 用 `Command(goto=...)` 跳转
- `execute_save_node` / `abort_node`：终端节点，写一条 message
- `run_until_interrupt` + `resume`：把"调用 → 暂停 → 决策 → 继续"包装成两步入口

## 5. 如何运行

```bash
uv run python 07-human-in-the-loop/main.py
```

预期：Demo 1（approve）打印"已保存"；Demo 2（reject）打印"已取消"。

## 6. 常见坑

1. **忘记 checkpointer**：interrupt 会抛 "no checkpointer configured"——HITL 强依赖 checkpoint
2. **resume 后 thread_id 必须一致**：`Command(resume=...)` 通过 config 的 thread_id 找到暂停点；新 thread_id 等于"另一段对话"
3. **`__interrupt__` vs `app.get_state(config).interrupts`**：v1.x 的 `app.invoke()` 返回 dict 中含 `__interrupt__` 键；也可以从 state 里查；两者都是合法访问方式
4. **Command(resume=...) 的 value 是任意 Python 对象**：可以是 str、dict、复杂结构。设计时让 value schema 配合 interrupt(payload) schema
5. **interrupt 在循环节点里要小心**：每次循环都会暂停，体验差；通常只在"敏感动作前"加 interrupt
6. **静态 vs 动态 interrupt**：`compile(interrupt_before=["confirm"])` 是另一种用法（无需在节点内调 interrupt()），但灵活性差

## 7. 小练习

1. 把 `interrupt(value)` 的 `value` 从 dict 改成自定义 dataclass / pydantic model，让前端接收时类型更稳
2. 在 `confirm_node` 中加 timeout 逻辑：interrupt 超过 N 秒未恢复就走 abort（思路：用 `app.get_state(config).created_at` 算出已暂停多久）
3. 改用静态 interrupt：`compile(interrupt_before=["execute_save"])`，对比两种 HITL 写法的差异
````

- [ ] **Step 6：commit**

```bash
git add 07-human-in-the-loop/ tests/test_module_07.py
git commit -m "feat(07): human-in-the-loop — interrupt() + Command(resume)"
```

---

### Task 1.3：模块 08 streaming

**Files:** Create: `08-streaming/main.py`、`08-streaming/README.md`、`tests/test_module_08.py`

**学习目标：**
- `graph.stream(input, config, stream_mode="updates" | "values")` 的差异
- 怎么把"逐步输出"接到 UI / 终端
- `astream(stream_mode="messages")` 拿到 token 级流式（高级）

- [ ] **Step 1：写 `08-streaming/main.py`**

```python
"""模块 08：Streaming

graph.stream(...) 让你拿到中间事件而不是只等最终 state。三种主要 mode：

- updates：每步只返回该步**新增的字段**（dict[node_name, partial_state]）
- values：每步返回**完整 state 快照**
- messages：token 级流式（需要 astream）—— 适合给前端做"打字机"效果

本模块演示前两种（同步），并附一段 messages 模式的注释示例（要 async）。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from common.llm import get_llm
from common.prompts import SYSTEM_BASE
from common.state import InvestBotState
from common.tools import calculator, get_quote

TOOLS = [get_quote, calculator]


def _make_call_model(llm_with_tools: BaseChatModel):
    def call_model(state: InvestBotState) -> dict:
        msgs = [SystemMessage(content=SYSTEM_BASE)] + list(state["messages"])
        return {"messages": [llm_with_tools.invoke(msgs)]}

    return call_model


def build_graph(llm: BaseChatModel | None = None):
    if llm is None:
        llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    g = StateGraph(InvestBotState)
    g.add_node("call_model", _make_call_model(llm_with_tools))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    return g.compile()


def stream_updates(app, user_text: str) -> list[dict]:
    """收集 stream_mode='updates' 产生的事件序列。"""
    events = []
    for chunk in app.stream(
        {"messages": [HumanMessage(content=user_text)]},
        stream_mode="updates",
    ):
        events.append(chunk)
    return events


def stream_values(app, user_text: str) -> list[dict]:
    """收集 stream_mode='values' 产生的快照序列。"""
    snapshots = []
    for chunk in app.stream(
        {"messages": [HumanMessage(content=user_text)]},
        stream_mode="values",
    ):
        snapshots.append(chunk)
    return snapshots


if __name__ == "__main__":
    app = build_graph()

    print("--- stream_mode='updates' ---")
    for ev in stream_updates(app, "查 NVDA 现价"):
        # ev 形如 {"call_model": {"messages": [...新增...]}}
        for node_name, partial in ev.items():
            print(f"  step '{node_name}': keys={list(partial.keys())}")

    print("\n--- stream_mode='values' ---")
    for i, snap in enumerate(stream_values(app, "查 AAPL 现价")):
        print(f"  snapshot {i}: messages_count={len(snap.get('messages', []))}")

    print("""
    # token 级流式（messages mode）需要 async：
    #     async for token, meta in app.astream(input, stream_mode="messages"):
    #         print(token.content, end="", flush=True)
    # 适合给前端 SSE / WebSocket 做打字机效果。
    """)
```

- [ ] **Step 2：手动运行（需要 ARK_API_KEY）**

```bash
uv run python 08-streaming/main.py
```

预期：updates 模式打印 N 个增量；values 模式打印 N 个快照（messages_count 单调递增）。

- [ ] **Step 3：写 smoke test `tests/test_module_08.py`**

```python
"""模块 08 smoke test：验证 stream_mode 产生事件序列。"""
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


def test_updates_mode_emits_per_node_events():
    main = _load("08-streaming")
    fake = FakeChatModelWithTools(responses=[AIMessage(content="ok")])
    app = main.build_graph(llm=fake)
    events = main.stream_updates(app, "hi")
    # 至少有一条来自 call_model
    node_names = []
    for ev in events:
        node_names.extend(ev.keys())
    assert "call_model" in node_names


def test_values_mode_emits_full_snapshots():
    main = _load("08-streaming")
    fake = FakeChatModelWithTools(responses=[AIMessage(content="ok")])
    app = main.build_graph(llm=fake)
    snapshots = main.stream_values(app, "hi")
    # 每个快照都含 messages 字段
    assert all("messages" in s for s in snapshots)
    # messages 数量单调不减
    counts = [len(s["messages"]) for s in snapshots]
    assert counts == sorted(counts)


def test_values_includes_tool_message_when_tool_is_called():
    main = _load("08-streaming")
    fake = FakeChatModelWithTools(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "get_quote", "args": {"symbol": "NVDA"}, "id": "c1"}]),
            AIMessage(content="NVDA 925"),
        ]
    )
    app = main.build_graph(llm=fake)
    snapshots = main.stream_values(app, "查 NVDA")
    # 末尾快照应含 ToolMessage
    last = snapshots[-1]
    classes = [m.__class__.__name__ for m in last["messages"]]
    assert "ToolMessage" in classes
```

- [ ] **Step 4：跑 smoke test**

```bash
uv run pytest tests/test_module_08.py -v
```

预期：3 passed。

- [ ] **Step 5：写 `08-streaming/README.md`**

````markdown
# 模块 08：Streaming

## 1. 本模块要解决什么问题

到现在为止 invoke 都是**整体阻塞**——发请求，等所有节点跑完才拿到结果。前端体验差。LangGraph 的 `stream()` 让你拿到**中间事件**：每个节点跑完就 emit 一次，可以接到 UI / 终端做实时展示。

InvestBot 进度：终于能"边想边说"——研究过程逐步显示，而不是憋到最后才输出。

## 2. 三种 stream_mode

```
graph.stream(input, stream_mode="updates")
  → 每步 yield 该步**增量更新**：dict[node_name, partial_state]
  → 适合"我想知道每一步在做什么"

graph.stream(input, stream_mode="values")
  → 每步 yield **完整 state 快照**
  → 适合"我想监控整体 state 演变"

app.astream(input, stream_mode="messages")
  → token 级流式（要 async）；yield (token_chunk, metadata) 元组
  → 适合给前端做打字机效果
```

## 3. 关键 API

| API | 一句话 |
|---|---|
| `for chunk in app.stream(...)` | 同步 generator |
| `async for chunk in app.astream(...)` | 异步 generator（前端集成必备） |
| `stream_mode="updates"` | dict[node_name, partial_dict] |
| `stream_mode="values"` | full state snapshot |
| `stream_mode="messages"` | (token, meta) 二元组，token 级 |
| `stream_mode=["updates", "values"]` | 多模式同时 emit（每个 chunk 第一项是 mode 名） |

## 4. 代码导读

- `stream_updates`：把 stream 的 generator 全部 collect 成 list，便于打印或测试
- `stream_values`：同上但用 values 模式，每步是完整快照
- 主函数：分两段演示，并在末尾留一段 async messages 的伪代码

## 5. 如何运行

```bash
uv run python 08-streaming/main.py
```

观察 updates 与 values 的差异——前者每步 dict 只有一个 key（节点名），后者是完整 state。

## 6. 常见坑

1. **stream 与 invoke 的关系**：`invoke = list(stream(...))` 的最后一项 + 同步等待；二者底层一致
2. **token 模式必须 async**：sync `stream(stream_mode="messages")` 不会按 token 切片；要 token 级请用 `astream`
3. **多 mode 时形态变化**：`stream_mode=["values", "updates"]` 时每个 chunk 是 `(mode, payload)` 元组，遍历时要解包
4. **流式的反压**：如果消费方慢，generator 会自然阻塞节点继续执行；这是好事但要意识到
5. **stream 不开 checkpointer 也能用**：流式只是"中间事件 emit"，与 persistence 是正交的两件事
6. **错误处理**：节点抛异常时 stream 会 raise；在 try/except 外包一层

## 7. 小练习

1. 写一个 async demo 用 `astream(stream_mode="messages")`，把 LLM 输出按 token 打印到终端
2. 实现 `stream_mode=["updates", "values"]` 的双模式，看每个 chunk 的形态
3. 在 stream 过程中按 Ctrl+C 中断，再用 thread_id + Command(resume) 续跑（要配 checkpointer）
````

- [ ] **Step 6：commit**

```bash
git add 08-streaming/ tests/test_module_08.py
git commit -m "feat(08): streaming — stream_mode updates / values / messages"
```

---

### Task F：收尾

- [ ] **Step 1：跑全量测试**

```bash
uv run pytest -v
```

预期：32（Plan 1+2）+ 4 + 3 + 3 = 42 passed。

- [ ] **Step 2：更新仓库总览 README**

把"### 坐 3"区块改成可点击链接：

```markdown
### 坐 3：让 Agent 真正能用
- [06-persistence](06-persistence/) — `MemorySaver` / `SqliteSaver` / `thread_id`
- [07-human-in-the-loop](07-human-in-the-loop/) — `interrupt()` / `Command(resume)`
- [08-streaming](08-streaming/) — `stream_mode` updates / values / messages
```

- [ ] **Step 3：commit + tag**

```bash
git add README.md
git commit -m "docs: update README — link plan 3 modules"
git tag plan-3-complete
```
