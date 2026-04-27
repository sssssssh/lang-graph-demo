# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目定位

教学型仓库——一份 LangGraph 从 0 到生产的渐进式 tutorial，主线场景是 **InvestBot 投资研究信息助手**。重点是把每个 LangGraph 概念压缩到一个 30-90 分钟可吃透的最小可运行模块。

**强约束**（贯穿所有 prompt 与回复）：InvestBot 只做研究信息汇总，**不出投资建议、不预测涨跌**，涉及标的的回答末尾必须附免责声明。改 prompts 时勿弱化此边界。

## 常用命令

```bash
uv sync                                  # 安装依赖（含 dev group）
uv run python NN-xxx/main.py             # 跑某个模块（也可 cd 进去再跑）
uv run pytest -v                         # 跑全部测试
uv run pytest tests/test_module_03.py -v # 跑单个模块的 smoke test
uv run pytest tests/common/ -v           # 只跑 common 单元测试
uv run pytest tests/test_module_03.py::test_route_to_explain  # 单个用例
uv run uvicorn 10-production.main:app --reload  # 跑模块 10 的 FastAPI（注意：模块名带连字符，需用 importlib 风格或直接 python main.py）
```

环境变量在 `.env`（拷自 `.env.example`）：必填 `ARK_API_KEY` / `LLM_MODEL`，模块 04 起需要 `TAVILY_API_KEY`。

## 架构关键点

### 目录约定
- `common/`：跨模块复用——`llm.py`（LLM 工厂）、`state.py`（共享 State 基类 `InvestBotState`）、`tools.py`（5 个工具）、`prompts.py`（合规话术 + 路由 prompt）
- `NN-xxx/`：主线 10 个模块，每个目录独立可运行的 `main.py` + 教学型 `README.md`
- `xN-xxx/`：独立小例（避开主线特性演示，如 `x1` 纯路由、`x2` map-reduce）
- `tests/test_module_NN.py`：每模块一份 smoke test，命名严格对应 `NN-xxx/`
- `tests/common/`：`common/` 模块的真单元测试
- `docs/superpowers/{specs,plans}/`：设计文档与实施计划

### 数字开头目录的 import 陷阱（重要）
模块目录名以数字开头（`01-hello-graph` 等），**Python 无法直接 `import`**。测试统一通过 `importlib.util.spec_from_file_location` 按文件路径加载 `main.py`——这是 `tests/test_module_*.py` 里那个 `_load(...)` helper 的由来。新写测试请沿用同样模式。

`pyproject.toml` 把 `pythonpath = ["."]` 写进 pytest 配置，这是为了让模块内 `from common.xxx import ...` 在测试 cwd 不确定时也稳定。

### State 演进约定
- 模块 01 / 02 / x1 教学目的，用各自自定义 State（`HelloState` / `ChatState` / `GuessState`），让学员先体会 "State = 任意 TypedDict"
- 自模块 03 起统一以 `InvestBotState`（含 `messages` + `last_intent`）为基类扩展：
  ```python
  class RoutingState(InvestBotState, total=False):
      extra_field: str
  ```
- `messages` 字段挂 `add_messages` reducer——节点返回 `{"messages": [new_msg]}` 是**追加**而非替换；不带 reducer 的字段是覆盖语义。

### LLM 注入模式（贯穿模块 03-10）
所有调 LLM 的模块都把 `build_graph()` / `run()` 设计成接受 `llm: BaseChatModel | None = None` 参数：
- 生产路径：`llm=None` → 走 `common.llm.get_llm()` → `ChatOpenAI` 指向方舟
- 测试路径：传入 `FakeMessagesListChatModel(responses=[...])` 预制响应，**不走真 API**

需要 `bind_tools` 的测试用 `FakeChatModelWithTools`（见 `tests/test_module_10.py`）——继承 fake 模型并把 `bind_tools` 实现成 no-op 返回 self。新写涉及 LLM 的测试请沿用此模式而非真调 API。

### LLM 后端与切换
默认走**火山引擎方舟**（OpenAI 兼容协议）。换 DeepSeek / 通义 / 智谱仅改 `.env` 的 `ARK_API_KEY` / `ARK_BASE_URL` / `LLM_MODEL`，**代码不动**。所以 `common/llm.py` 是单一来源，不要在模块里直接 `ChatOpenAI(...)`。

### `common/llm.py` 的 import 副作用
`load_dotenv()` 在 import 时执行——`from common.llm import get_llm` 这一行就会读 `.env` 改 `os.environ`。写隔离测试时用 `monkeypatch` 或 `patch.dict(os.environ, ..., clear=True)`，并注意 `load_dotenv()` 默认 `override=False` 不覆盖已有 env。

### Tavily client 全局缓存
`common/tools.py` 用 `_TAVILY_CLIENT` 模块级变量惰性缓存。测试不能让一个用例 monkeypatch 的 fake 泄漏到下一个——`tests/conftest.py` 有一个 `autouse=True` 的 fixture 在每个测试前后把它重置为 `None`。新加涉及 Tavily 的测试无需自己重置。

### 工具集
`common/tools.py::ALL_TOOLS` 是 `[search_web, get_quote, get_fundamentals, calculator, save_note]`。`get_quote` / `get_fundamentals` 是 mock 数据，仅认识 `NVDA / AAPL / TSLA / MSFT`，未知 symbol 返回 `{"error": ...}` 字段——LLM 测试 prompt 时要意识到这点。`calculator` 是 AST 受限求值（防注入，限制 `**` 右操作数 ≤100）。`save_note` 写入 `INVESTBOT_NOTES_DIR`（默认 `./notes`，已 gitignore）。

### checkpoint 文件
`**/checkpoint.sqlite` 已 gitignore——模块 06 和 07 演示 `SqliteSaver` 时会落盘，别 commit。

## 写代码规则

- 学习模块（`NN-xxx/main.py`）的注释密度比正常项目高，目的是教学——**保持这种风格**，新写或改时不要把注释剥光
- 每个学习模块的 `main.py` 必须暴露 `build_graph()` 和某种 `run(...)` 入口，smoke test 依赖这一对外 API
- 涉及标的的回复模板都附了 `_DISCLAIMER` 或 `SYSTEM_BASE` 里的合规约束——改流程时不要丢掉
- 跨模块复用从 `common/` 引；不要把 LLM/工具的硬编码塞进模块 `main.py`
