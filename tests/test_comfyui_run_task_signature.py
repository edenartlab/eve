"""ComfyUI.run_task must accept every keyword the shared task handler passes.

eve.task._task_handler calls each handler with user/agent/session/message/
tool_call_id keywords. ComfyUI.run_task lives in comfyui.py (deployed as a
separate Modal app per workspace) and once lagged behind: it lacked `message`
and `tool_call_id`, so every ComfyUI task failed immediately with
"run_task() got an unexpected keyword argument 'message'".

comfyui.py cannot be imported in a unit test (it builds a Modal image at import
time), so both sides are compared through their syntax trees.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _handler_keywords() -> set:
    tree = ast.parse((ROOT / "eve" / "task.py").read_text())
    handler = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_task_handler"
    )
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "func"
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError("_task_handler no longer calls func(...)")


def _comfyui_run_task_params() -> tuple:
    tree = ast.parse((ROOT / "comfyui.py").read_text())
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ComfyUI"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "run_task"
    )
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    return names, fn.args.kwarg is not None


def test_comfyui_run_task_accepts_all_task_handler_keywords():
    passed = _handler_keywords()
    assert {"message", "tool_call_id"} <= passed  # guards the parser itself
    accepted, has_var_kwargs = _comfyui_run_task_params()
    missing = passed - accepted
    assert has_var_kwargs or not missing, (
        f"ComfyUI.run_task does not accept {sorted(missing)}; "
        "every ComfyUI task would fail with a TypeError"
    )
