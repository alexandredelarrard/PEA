"""Architecture assertions for the intentionally removed application and ledger."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


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
