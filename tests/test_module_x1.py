"""模块 x1 smoke test：验证条件边收敛 + recursion_limit 触发。"""
import sys
import importlib.util
from pathlib import Path

import pytest


def _load(module_dir: str):
    path = Path(__file__).parent.parent / module_dir / "main.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_loop_converges_when_step_positive():
    main = _load("x1-pure-routing")
    out = main.run(target=10, step=3)
    assert out["guess"] >= 10
    assert len(out["log"]) == 4  # 0→3→6→9→12


def test_loop_one_step_when_target_immediately_reachable():
    main = _load("x1-pure-routing")
    out = main.run(target=1, step=5)
    assert out["guess"] == 5
    assert len(out["log"]) == 1


def test_recursion_limit_protects_against_infinite_loop():
    """step=0 时永远到不了 target，recursion_limit 应当抛错保护。"""
    main = _load("x1-pure-routing")
    with pytest.raises(Exception):  # langgraph.errors.GraphRecursionError
        main.run(target=10, step=0)
