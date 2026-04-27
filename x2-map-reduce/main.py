"""独立小例 x2：Map-Reduce —— Send API

给一组股票 symbols，并行查每只的 quote，然后汇总。
"map" 阶段：fan_out 节点用 Send 把 N 个 worker 任务并行投出去；
"reduce" 阶段：worker 各自写 results，列表 reducer 把它们累加；
最后 summary 节点读 results 输出一段汇总文字。

完全不调 LLM，专注演示 Send API 的并行机制。
"""
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# 兼容按文件路径直接执行 `main.py` 时的导入路径。
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from common.tools import get_quote


class MapState(TypedDict, total=False):
    symbols: list[str]
    # results 用 operator.add 作为 reducer：每个 worker 返回一份单元素 list，会被合并
    results: Annotated[list[dict], operator.add]
    summary: str


def fan_out(state: MapState) -> list[Send]:
    """conditional edge 函数：返回 list[Send]，每个 Send 触发一份并行 worker。"""
    return [Send("worker", {"symbol": s}) for s in state["symbols"]]


def worker(state: dict) -> dict:
    """worker 接收的 state 是 Send 的 arg —— 这里是 {"symbol": s}。"""
    quote = get_quote.invoke({"symbol": state["symbol"]})
    return {"results": [quote]}  # 单元素 list，会被 reducer 累加


def summary(state: MapState) -> dict:
    lines = []
    for r in state["results"]:
        if "error" in r:
            lines.append(f"- {r['symbol']}: 未知")
        else:
            lines.append(f"- {r['symbol']}: ${r['price']} ({r['change_pct']:+.2f}%)")
    return {"summary": "今日报价：\n" + "\n".join(lines)}


def build_graph():
    g = StateGraph(MapState)
    g.add_node("worker", worker)
    g.add_node("summary", summary)

    # START 用 conditional edge 直接 fan-out
    g.add_conditional_edges(START, fan_out, ["worker"])
    g.add_edge("worker", "summary")
    g.add_edge("summary", END)
    return g.compile()


def run(symbols: list[str]) -> dict:
    app = build_graph()
    return app.invoke({"symbols": symbols, "results": []})


if __name__ == "__main__":
    out = run(["NVDA", "AAPL", "TSLA", "MSFT", "ZZZZ"])
    print(out["summary"])
