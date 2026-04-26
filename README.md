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

### 坐 2：接入 LLM 与工具
- [03-routing-and-llm](03-routing-and-llm/) — 第一次调真 LLM，做意图路由
- [04-tool-calling](04-tool-calling/) — `bind_tools` / `ToolNode` / `tools_condition` 三件套
- [05-react-loop](05-react-loop/) — `create_react_agent` 一行起飞，并对比手写低阶版

### 坐 3：让 Agent 真正能用
- [06-persistence](06-persistence/) — `MemorySaver` / `SqliteSaver` / `thread_id`
- [07-human-in-the-loop](07-human-in-the-loop/) — `interrupt()` / `Command(resume)`
- [08-streaming](08-streaming/) — `stream_mode` updates / values / messages

### 坐 4：多 Agent 与生产化
- [09-multi-agent](09-multi-agent/) — Subgraph + Supervisor 多 agent 模式
- [x2-map-reduce](x2-map-reduce/) — `Send` API 并行 fan-out/fan-in（独立小例）
- [10-production](10-production/) — FastAPI + LangSmith + 重试装饰器

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
