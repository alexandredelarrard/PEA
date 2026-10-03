"""Every place that NAMES a sleeve only names a registered one.

The app's sleeve picker, `portfolio.yml` and the `StepPortfolio` fallback list are read
statically (AST / YAML) so the check needs no Streamlit, DB or trained model. A name that is not
in `STRATEGY_REGISTRY` breaks at run time in three different ways: Streamlit rejects a multiselect
default that is not an option, `StepPortfolio` raises on an unknown sleeve, and
`StepStrategyMoves._sleeve_cfg` silently falls back to the portfolio fees.
"""

from __future__ import annotations

import ast
from pathlib import Path

from omegaconf import OmegaConf

from src.strategies import STRATEGY_REGISTRY

ROOT = Path(__file__).resolve().parents[2]


def _app_all_sleeves() -> list[str]:
    tree = ast.parse((ROOT / "app" / "app.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALL_SLEEVES" for t in node.targets):
            return [ast.literal_eval(elt) for elt in node.value.elts]  # type: ignore[attr-defined]
    raise AssertionError("ALL_SLEEVES not found in app/app.py")


def _portfolio_default_sleeves() -> list[str]:
    tree = ast.parse((ROOT / "src" / "portfolio" / "step_portfolio.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "sleeves"
            and len(node.args) == 2
        ):
            return list(ast.literal_eval(node.args[1]))
    raise AssertionError("no `.get('sleeves', [...])` default in step_portfolio.py")


def test_every_named_sleeve_is_registered() -> None:
    registered = set(STRATEGY_REGISTRY)
    sources = {
        "app/app.py ALL_SLEEVES": _app_all_sleeves(),
        "configs/portfolio.yml sleeves": [str(s) for s in OmegaConf.load(ROOT / "configs" / "portfolio.yml").portfolio.sleeves],
        "StepPortfolio default": _portfolio_default_sleeves(),
    }
    unknown = {where: sorted(set(names) - registered) for where, names in sources.items() if set(names) - registered}
    assert not unknown, f"sleeves named but not registered: {unknown}"
    for where, names in sources.items():
        assert names, f"{where} names no sleeve at all"

    print("\n=== SANITY CHECK: sleeve names vs STRATEGY_REGISTRY ===")
    print(f"registered: {sorted(registered)}")
    for where, names in sources.items():
        print(f"  {where:32s} -> {names}")
    print("Every sleeve the app, the config and the portfolio fallback can request is registered.")
