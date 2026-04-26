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
- **依赖注入 `build_graph(llm=None)`**：让节点对"模型来源"不耦合，方便测试塞 fake、生产换厂商
- **路由靠 state 字段**：把分类结果写进 `state.last_intent`，路由函数只读这个字段——保持职责单一

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
```

> 真 LLM 偶尔会分错类（比如"巴菲特怎么看"可能分到 stock 也可能分到 explain）。这是不确定性，**不是 bug**。生产中可以加 few-shot 例子或换更强模型缓解。

## 6. 常见坑

1. **`temperature` 不调 0**：路由要确定性，默认 0.3 也偏高，本模块用 `get_llm(temperature=0)`
2. **节点里直接 `get_llm()`**：每次调用都会重建 ChatOpenAI 客户端；用 closure 工厂只构造一次
3. **路由函数返回值不在 mapping**：比如 `last_intent="other"`，会 KeyError；务必给 default 兜底
4. **conditional edge 的目标节点**：mapping 的 key 要和 `route()` 返回值一致，value 是真实节点名；二者可同名也可不同名
5. **`HumanMessage` vs 字符串**：`llm.invoke("hi")` 在新版能跑，但 `llm.invoke([HumanMessage("hi")])` 才是地道写法（与 LangChain Message 抽象一致）
6. **测试如何隔离 LLM**：用 `FakeMessagesListChatModel(responses=[AIMessage(content="A")])` 预制；它和真 ChatOpenAI 都实现 `BaseChatModel` 接口，可直接互换

## 7. 小练习

1. 给 ROUTE_INSTRUCTIONS 加 2 个 few-shot 例子（比如"PE 是什么 → A"、"宁德时代 → B"），看分类准确率
2. 把 `route_node` 改用 `with_structured_output(SchemaWith Literal)`，让 LLM 输出强类型而不是字母
3. 实现一个 `unknown_node`：当 LLM 输出无法识别时（不只是字母不在 ABCD），路由走这里而不是 fallback
