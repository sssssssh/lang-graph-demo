"""模块 01 smoke test：确认 graph 能 invoke 且 reply 正确反映输入。"""
import sys
import importlib.util
from pathlib import Path


def _load_module_main(module_dir: str):
    """直接按文件路径加载模块的 main.py（避免学习模块目录名以数字开头无法 import）。"""
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_01_echoes_user_input():
    main = _load_module_main("01-hello-graph")
    out = main.run("hello langgraph")
    assert out["user_input"] == "hello langgraph"
    assert "hello langgraph" in out["reply"]
    assert "InvestBot" in out["reply"]
