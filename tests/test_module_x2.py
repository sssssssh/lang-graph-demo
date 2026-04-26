"""模块 x2 smoke test：验证 Send fan-out + reducer fan-in。"""
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


def test_fan_out_collects_all_results():
    main = _load("x2-map-reduce")
    out = main.run(["NVDA", "AAPL", "MSFT"])
    # 三只股票，results 列表应有 3 项
    assert len(out["results"]) == 3
    syms = {r.get("symbol") for r in out["results"]}
    assert syms == {"NVDA", "AAPL", "MSFT"}


def test_summary_includes_all_symbols():
    main = _load("x2-map-reduce")
    out = main.run(["NVDA", "AAPL"])
    assert "NVDA" in out["summary"]
    assert "AAPL" in out["summary"]


def test_unknown_symbol_handled_gracefully():
    main = _load("x2-map-reduce")
    out = main.run(["NVDA", "ZZZZ"])
    assert "未知" in out["summary"]
