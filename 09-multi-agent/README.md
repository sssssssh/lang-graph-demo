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
