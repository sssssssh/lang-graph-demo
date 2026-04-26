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
