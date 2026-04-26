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
