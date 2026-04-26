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
