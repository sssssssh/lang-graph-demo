"""模块 07 smoke test：验证 interrupt 暂停 + Command(resume) 恢复。"""
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


def test_approve_path_executes_save():
    main = _load("07-human-in-the-loop")
    app = main.build_graph()
    _, config = main.run_until_interrupt(app, "go", thread_id="ta")
    final = main.resume(app, "approve", config)
    assert "已保存" in final["messages"][-1].content


def test_reject_path_aborts():
    main = _load("07-human-in-the-loop")
    app = main.build_graph()
    _, config = main.run_until_interrupt(app, "go", thread_id="tr")
    final = main.resume(app, "reject", config)
    assert "取消" in final["messages"][-1].content


def test_interrupt_actually_pauses_before_resume():
    """暂停时不应执行 execute_save / abort 节点；只跑到 confirm。"""
    main = _load("07-human-in-the-loop")
    app = main.build_graph()
    paused, _ = main.run_until_interrupt(app, "go", thread_id="tp")
    contents = [m.content for m in paused.get("messages", [])]
    # propose_save 写了一条；execute / abort 都没跑
    assert any("建议把今天的研究保存" in c for c in contents)
    assert not any("已保存" in c or "取消" in c for c in contents)
