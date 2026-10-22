"""Architecture assertions for the intentionally removed application and ledger."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def _control_depth(node: ast.AST) -> int:
    controls = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Match)
    children = (_control_depth(child) for child in ast.iter_child_nodes(node) if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef))
    return int(isinstance(node, controls)) + max(children, default=0)


def test_portfolio_functions_remain_small_and_shallow() -> None:
    root = Path(__file__).resolve().parents[2] / "src/portfolio"
    violations = []
    count = 0
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            count += 1
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            depth = _control_depth(node)
            if length > 150 or depth > 3:
                violations.append(f"{path.relative_to(root)}:{node.name}: {length} lines, {depth} nesting levels")
    assert not violations, "\n".join(violations)
    print(f"SANITY: all {count} portfolio functions fit 150 lines and at most 3 control-flow nesting levels.")


def test_no_streamlit_or_legacy_consumers() -> None:
    root = Path(__file__).resolve().parents[2]
    assert not (root / "app/app.py").exists()
    assert not list((root / "src/strategies").rglob("*.py"))
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any(dep.startswith("streamlit") for dep in project["project"]["dependencies"])
    forbidden = ("streamlit", "src.strategies", "src.portfolio.step_strategy_moves", "src.portfolio.step_portfolio", "src.portfolio.utils.blend")
    for path in (root / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            imports = (
                [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            )
            assert not any(name == f or name.startswith(f + ".") for name in imports for f in forbidden), str(path)
    lock = tomllib.loads((root / "poetry.lock").read_text(encoding="utf-8"))
    assert "streamlit" not in {p["name"] for p in lock["package"]}
    print("SANITY: app, Streamlit, old strategies and ledger consumers are absent; no orphaned runtime imports remain.")
