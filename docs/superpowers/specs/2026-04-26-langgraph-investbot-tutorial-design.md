# LangGraph 学习项目设计 —— InvestBot 投资研究助手

- 日期：2026-04-26
- 学员：Python 熟练，做过简单的直接调 LLM API，没用过 LangChain / LangGraph
- 目标：从 0 到生产级，分模块递进，学完能独立用 LangGraph 设计并实现一个生产级 Agent
- 工作目录：`/Users/ssh/code-ai-agent/lang-graph-demo/`

---

## 1. 学习目标

完成本项目后，学员应当能够：

1. **概念**：清楚解释 StateGraph、Node、Edge、State、Reducer、条件边、子图、Send 等核心概念，看懂任意一份 LangGraph 代码
2. **构建**：独立设计并实现带工具调用、记忆、Human-in-the-Loop、流式输出的 Agent
3. **生产化**：知道持久化、错误处理、可观测性、超时/重试、部署形态如何接入
4. **对照**：在 LangGraph 官方文档中能准确定位 API，知道何时用何特性，并能与"自己写循环+OpenAI 函数调用"的土法做对比

完成自检：能在不抄代码的前提下，向同事讲清楚"为什么我的 Agent 要用 LangGraph 而不是手写循环"。

---

## 2. 主线场景：InvestBot 投资研究助手

### 2.1 定位

面向散户投资者的"投资研究信息助手"。

**支持的请求模式（多模式，由路由决定走哪条主分支）**：

| 模式 | 用户提问示例 | 助手主要动作 |
|---|---|---|
| 概念解释 | "什么是夏普比率？" | 直接 LLM 回答，无需工具 |
| 个股研究 | "看一下 NVDA 最近的情况" | 搜索新闻 → 取行情/财报 mock → 汇总 |
| 板块/概念研究 | "AI 算力板块最近怎么样" | 多次搜索 + 多标的对比 |
| 兜底 | 模糊或无关问题 | 礼貌请求澄清 |

LLM 在路由模块中根据问题类型分流到上述分支。

**横切能力**（任意模式下都可被 LLM 触发）：

- 笔记保存：调用 `save_note` 工具（自模块 06 起）
- 简单计算：调用 `calculator` 工具（自模块 05 起）

这两个工具不属于路由分支，而是 LLM 在生成回答时自主决定何时使用，体现"工具是横切能力"这一设计理念。

### 2.2 范围限制（合规与教学双重考量）

- **不输出**买入/卖出/持有建议；不预测涨跌
- 输出统一标注"仅供研究参考，不构成投资建议"，由专门节点附加
- 在 system prompt 中明示助手身份与边界

### 2.3 工具集

| 工具 | 实现 | 出现模块 |
|---|---|---|
| `search_web(query)` | Tavily 真实联网 | 04 起 |
| `get_quote(symbol)` | mock：返回结构化 dict（最新价、涨跌幅、52w 高低） | 04 起 |
| `get_fundamentals(symbol)` | mock：返回结构化 dict（PE、PB、营收、净利同比） | 05 起 |
| `calculator(expr)` | 安全 eval | 05 起 |
| `save_note(title, content)` | 写本地 `notes/` 目录 markdown | 06 起 |

mock 工具放在 `common/tools.py`，全局共享。

---

## 3. 技术栈

| 维度 | 选型 | 备注 |
|---|---|---|
| Python | 3.11+ | langgraph 要求 |
| 包管理 | `uv` | `uv init`、`uv add`、`uv run` |
| 图引擎 | `langgraph` | 主角 |
| 模型客户端 | `langchain-openai`（`ChatOpenAI`） | 国产模型走 OpenAI 兼容协议 |
| **默认 LLM 后端** | **火山引擎方舟（Ark）** | `base_url=https://ark.cn-beijing.volces.com/api/v3` |
| 默认模型 | 豆包系列（如 `doubao-1-5-pro-32k-...`） | 由 `.env` 中 `LLM_MODEL` 控制，可换 |
| 搜索工具 | `tavily-python` | 免费 1000 次/月，需注册 |
| 持久化 | `MemorySaver` / `SqliteSaver` | 模块 06 切换 |
| 环境变量 | `python-dotenv` | `.env` 管理 keys |
| 可观测性 | LangSmith（可选） | 模块 10 演示 |

### 3.1 LLM 客户端规范（关键）

`common/llm.py` 中统一导出 `get_llm()`，所有模块只用这个函数：

```python
# 示意，最终代码在 writing-plans 阶段定稿
from langchain_openai import ChatOpenAI
import os

def get_llm(temperature: float = 0.3, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["ARK_API_KEY"],
        base_url=os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        temperature=temperature,
        **kwargs,
    )
```

切换到通义/智谱/DeepSeek 时，只改 `.env` 中三件：`ARK_API_KEY`、`ARK_BASE_URL`、`LLM_MODEL`，代码不动。

### 3.2 `.env` 规范

```
# 火山引擎方舟（默认）
ARK_API_KEY=...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=doubao-1-5-pro-32k-250115

# Tavily 搜索
TAVILY_API_KEY=...

# 可选
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lang-graph-demo
```

`.env.example` 提交到 git，`.env` 走 `.gitignore`。

---

## 4. 模块划分

### 4.1 主线 10 模块

每个模块都基于 InvestBot 同一个场景，逐步加新能力。

| # | 模块 | 学习目标（核心 API） | InvestBot 在本模块的形态 |
|---|---|---|---|
| 01 | **hello-graph** | `StateGraph`、`TypedDict` State、`add_node`、`add_edge`、`START`、`END`、`graph.compile()`、`graph.invoke()` | 单节点：原样回显用户问题 |
| 02 | **state-and-reducer** | 多字段 State、`Annotated[..., reducer]`、`add_messages`、`MessagesState` | 累计对话历史 + 一个 `last_intent` 字段 |
| 03 | **routing-and-llm** | 接入 `ChatOpenAI`、`add_conditional_edges`、路由函数 | 用 LLM 把问题分流到 "概念解释 / 个股研究 / 兜底" 三条边 |
| 04 | **tool-calling** | `@tool`、`bind_tools`、`ToolNode`、`tools_condition` | 在"个股研究"分支接入 `search_web` + `get_quote` |
| 05 | **react-loop** | LLM ↔ Tool 循环、`create_react_agent`（高阶封装）、自定义 ReAct 节点（低阶版本） | 让助手自主多步搜索、计算、整理 |
| 06 | **persistence** | `MemorySaver`、`SqliteSaver`、`thread_id`、`graph.get_state()` | 不同用户独立记忆，可断线续聊；引入 `save_note` 工具 |
| 07 | **human-in-the-loop** | `interrupt()`、`Command(resume=...)`、断点 / 修改 State | 输出"风险提示"前由用户确认或编辑；演示"被叫停后改 State 再继续" |
| 08 | **streaming** | `graph.stream()` / `astream()`、模式 `values` / `updates` / `messages` | 边搜边写，分阶段渲染 |
| 09 | **multi-agent** | Subgraph、Supervisor 模式、子图与父图状态映射 | 研究员 Agent + 风控审查员 Agent + 总结员 Agent |
| 10 | **production** | 错误处理、`with_retry`、超时、结构化日志、LangSmith 接入、把图包成 FastAPI 服务 | 把 InvestBot 包成可调用 HTTP 服务 |

### 4.2 独立小例 2 个

主线之外，用最干净的场景演示某个特性。

| # | 模块 | 为什么独立 |
|---|---|---|
| **x1-pure-routing** | 不调 LLM，纯 Python 函数做条件分支，把"图结构 / 条件边 / 循环"讲透，不被 LLM 不确定性干扰 |
| **x2-map-reduce** | 用 `Send` API 并行处理多个标的，再汇总。生产场景常用，但塞进主线会让某个模块臃肿 |

### 4.3 模块依赖关系

- 每个模块的 `main.py` 可独立 `python main.py` 运行
- 仅依赖 `common/`（LLM 客户端、mock 工具、State 基类）
- **不依赖**前一模块的运行结果或 import 前一模块代码

这样学员可以挑任意模块上手，也方便 review。

---

## 5. 每个模块的统一结构

每个 `NN-xxx/` 目录包含：

### 5.1 `README.md`（讲义）

固定 6 节：

1. **本模块要解决什么问题**（场景描述，不是 API 罗列）
2. **核心概念**（2-4 个，每个配 ASCII 或 mermaid 图）
3. **关键 API**（函数签名 + 一句话解释 + 何时用）
4. **代码导读**（指向 `main.py` 的关键行号，解释为何这么写）
5. **如何运行**（命令 + 预期输出片段）
6. **常见坑**（3-5 条 LangGraph 容易翻车的点）
7. **小练习**（2-3 题，自己改代码体会）

### 5.2 `main.py`

- 单文件，可独立运行
- 关键行带简短中文注释，说明 **为什么这么写**（不是 What）
- 末尾有 `if __name__ == "__main__":` 跑一个小例子，stdout 输出可读

### 5.3 可选：`notes.md`

学员自己写思考、踩坑记录。仓库提供空模板。

---

## 6. 目录结构

```
lang-graph-demo/
├── README.md                       # 总览 + 环境准备 + 模块索引 + 学习路径
├── pyproject.toml                  # uv 管理依赖
├── uv.lock
├── .env.example                    # ARK_API_KEY / ARK_BASE_URL / LLM_MODEL / TAVILY_API_KEY
├── .gitignore
├── common/
│   ├── __init__.py
│   ├── llm.py                      # get_llm() 工厂
│   ├── tools.py                    # search_web / get_quote / get_fundamentals / calculator / save_note
│   ├── state.py                    # 共享 State 基类
│   └── prompts.py                  # 共享 system prompt 片段（合规声明等）
├── notes/                          # save_note 工具落盘目录（gitignore）
├── 01-hello-graph/
│   ├── README.md
│   └── main.py
├── 02-state-and-reducer/
├── 03-routing-and-llm/
├── 04-tool-calling/
├── 05-react-loop/
├── 06-persistence/
│   └── checkpoint.sqlite           # 运行后产生（gitignore）
├── 07-human-in-the-loop/
├── 08-streaming/
├── 09-multi-agent/
├── 10-production/
│   └── server.py                   # FastAPI 包装
├── x1-pure-routing/
├── x2-map-reduce/
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-26-langgraph-investbot-tutorial-design.md   # 本文件
```

---

## 7. 环境与运行约定

### 7.1 一次性环境准备

```bash
cd /Users/ssh/code-ai-agent/lang-graph-demo
uv sync                              # 装齐所有依赖
cp .env.example .env && vim .env     # 填 ARK_API_KEY / TAVILY_API_KEY 等
```

### 7.2 跑某个模块

```bash
cd 01-hello-graph
uv run python main.py
```

或在仓库根目录 `uv run python 01-hello-graph/main.py` 也行。

### 7.3 学习路径建议

- **第一坐**：模块 01 + 02 + x1（图与状态的"机械骨架"）
- **第二坐**：模块 03 + 04 + 05（接入 LLM，做出 Agent 雏形）
- **第三坐**：模块 06 + 07 + 08（让 Agent 在真实场景能用）
- **第四坐**：模块 09 + x2 + 10（多 Agent 与生产化）

每坐 1-2 小时，做完模块内"小练习"再前进。

---

## 8. 完成标准

学员能在不参考本仓库代码的前提下：

1. 用 LangGraph 从零搭建一个带至少 2 个工具的 ReAct Agent
2. 给该 Agent 加上持久化（SqliteSaver）和多用户隔离（thread_id）
3. 给该 Agent 增加一个 Human-in-the-Loop 断点
4. 用 stream 模式做流式响应
5. 把它包成一个可调用的 HTTP 服务，并加上基本错误处理和日志
6. 解释 `add_messages` reducer 为什么必要、`tools_condition` 与手写条件边的差别、`Send` 与普通 fan-out 的差别

---

## 9. 风险与备选

| 风险 | 缓解 |
|---|---|
| 火山引擎工具调用稳定性不及 OpenAI / Anthropic | 提供 `.env` 一键切换，模块 04+ 文档注明"若工具调用偶发不触发，可临时切到 DeepSeek" |
| Tavily 免费额度被刷完 | `common/tools.py` 中 `search_web` 加 LRU 缓存（写到本地 JSON），重复 query 不重复请求 |
| LangGraph API 在小版本间变动 | `pyproject.toml` 锁定 `langgraph` 次版本，README 注明本仓库验证过的版本 |
| 学员被合规话术干扰学习节奏 | 合规声明集中在 `common/prompts.py`，主线模块只关注图结构 |

---

## 10. 不在本项目范围

- 真实的金融数据接入（股价、财报使用 mock），避免引入复杂第三方 API key
- 前端 UI（CLI / 简单 HTTP 已足够说明问题）
- 训练自定义模型 / fine-tune
- 向量库与 RAG 完整链路（仅在模块 10 风险提示里点到为止；如学员想深入，可加 X3 模块作为后续扩展，本期不做）

---

## 11. 后续：实施计划

本设计文档（spec）通过后，进入 `superpowers:writing-plans` 技能，把上述 12 个模块拆成可逐个执行的实施计划，每个模块对应计划中的一个阶段，含验证标准。
