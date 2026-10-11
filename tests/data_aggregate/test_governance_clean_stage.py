"""
The governance cleaning stage (src/data_aggregate/utils/governance/clean.py).

  * a static guard: every impute / carry / repair / reconcile helper is called from the stage
    only, never from a feature module, the builder or the step;
  * re-cleaning a cleaned bundle is a no-op.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_aggregate.utils.governance.clean import GovernanceInputs, clean_governance_inputs

_ROOT = Path(__file__).resolve().parents[2]
_GOV = _ROOT / "src" / "data_aggregate" / "utils" / "governance"
_STEP = _ROOT / "src" / "data_aggregate" / "transformers" / "step_cube_governance.py"

#: The cleaning helpers. Only `clean.py` may call them in production code.
_HELPERS = frozenset(
    {
        "impute_def14a",
        "impute_exec_comp",
        "impute_director_comp",
        "fill_director_attributes",
        "board_aggregates",
        "merge_board_aggregates",
        "finalize_board_source",
        "repair_ownership_basis",
        "economic_ownership",
    }
)
_GUARDED = (
    _GOV / "panel.py",
    _GOV / "pay_features.py",
    _GOV / "provisions_features.py",
    _GOV / "directors.py",
    _GOV / "director_comp.py",
    _GOV / "vote_dissent_features.py",
    _STEP,
)


def _called_names(tree: ast.AST) -> list[tuple[str, str, int]]:
    """(enclosing function, called name, line) for every call to a plain or attribute name."""
    found: list[tuple[str, str, int]] = []

    def visit(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = owner
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
            if isinstance(child, ast.Call):
                fn = child.func
                called = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if called is not None:
                    found.append((owner, called, child.lineno))
            visit(child, name)

    visit(tree, "<module>")
    return found


def test_governance_clean_stage_owns_cleaning() -> None:
    """No helper is called outside `clean.py`; the step calls the stage exactly once."""
    offenders: list[str] = []
    step_calls = 0
    for path in _GUARDED:
        for owner, called, line in _called_names(ast.parse(path.read_text(encoding="utf-8"))):
            # a helper defined in directors.py / director_comp.py may call its own siblings
            if called in _HELPERS and owner not in _HELPERS:
                offenders.append(f"{path.name}:{line} {owner}() calls {called}")
            if path == _STEP and called == "clean_governance_inputs":
                step_calls += 1

    panel_defs = {n.name for n in ast.parse((_GOV / "panel.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    clean_defs = {n.name for n in ast.parse((_GOV / "clean.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    moved = {"repair_ownership_basis", "economic_ownership"}

    assert not offenders, "cleaning helpers called outside the stage:\n" + "\n".join(offenders)
    assert step_calls == 1, f"the step must call clean_governance_inputs exactly once, found {step_calls}"
    assert moved <= clean_defs and not (moved & panel_defs), "the ownership repair must live in clean.py only"

    print("\n=== SANITY CHECK: governance cleaning has one owner ===")
    print(f"  {len(_HELPERS)} helpers, {len(_GUARDED)} guarded modules: 0 calls outside clean.py;")
    print("  the step calls clean_governance_inputs once and the ownership repair moved out of panel.py. Validated.")


def _bundle() -> tuple[GovernanceInputs, pd.DataFrame]:
    """Two tickers, three proxies each, with gaps every helper fills."""
    as_of = ["2021-03-01", "2022-03-01", "2023-03-01"]
    def14a = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "accession_number": [f"A-{i}" for i in range(3)] + [f"B-{i}" for i in range(3)],
            "as_of": as_of * 2,
            "ceo_name_proxy": ["Jane Doe"] * 3 + ["John Roe"] * 3,
            "ceo_salary": [1.0e6, np.nan, 1.1e6, 9.0e5, 9.5e5, np.nan],
            "ceo_total_comp": [5.0e6, np.nan, 6.0e6, 4.0e6, 4.4e6, np.nan],
            "board_size": [10.0, np.nan, 11.0, 9.0, 9.0, 9.0],
            "avg_other_public_boards": [np.nan, 1.0, np.nan, 0.5, np.nan, 0.7],
            "insider_shares": [1.0e6, 1.2e6, np.nan, 5.0e6, 5.0e6, 5.0e6],
            "insider_ownership_pct": [np.nan, 0.012, np.nan, 0.40, np.nan, 0.05],
            "insider_voting_pct": [np.nan, np.nan, np.nan, 0.20, np.nan, 0.60],
            "dual_class_shares": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )
    exec_comp = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "accession_number": ["A-0", "A-0", "B-0"],
            "as_of": ["2021-03-01"] * 2 + ["2021-03-01"],
            "name": ["Jane Doe", "Bob Ray", "John Roe"],
            "fiscal_year": [2020, 2020, 2020],
            "salary": [1.0e6, 5.0e5, 9.0e5],
            "bonus": [1.0e6, np.nan, np.nan],
            "total": [np.nan, 8.0e5, np.nan],
        }
    )
    directors = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "accession_number": ["A-0", "A-0", "A-1", "A-1", "A-2", "A-2"],
            "as_of": ["2021-03-01", "2021-03-01", "2022-03-01", "2022-03-01", "2023-03-01", "2023-03-01"],
            "name": ["Ann Lee", "Sam Kim"] * 3,
            "age": [60.0, 55.0, np.nan, 56.0, 62.0, np.nan],
            "tenure_years": [3.0, 5.0, 4.0, 6.0, 5.0, 7.0],
            "is_independent": [1.0, 0.0] * 3,
            "other_public_company_boards": [1.0, 2.0, np.nan, 2.0, np.nan, np.nan],
        }
    )
    director_comp = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "accession_number": ["A-0", "A-0"],
            "as_of": ["2021-03-01"] * 2,
            "name": ["Ann Lee", "Sam Kim"],
            "fees_earned": [1.0e5, 9.0e4],
            "stock_awards": [1.5e5, np.nan],
            "total": [np.nan, 2.0e5],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "as_of": pd.to_datetime(["2020-12-31", "2022-12-31", "2020-12-31"]),
            "sharesOutstandingPit": [1.0e8, 1.0e8, 1.0e7],
        }
    )
    return GovernanceInputs(def14a=def14a, exec_comp=exec_comp, directors=directors, director_comp=director_comp), fundamentals


def _frames(b: GovernanceInputs) -> dict[str, pd.DataFrame | None]:
    return {k: getattr(b, k) for k in ("def14a", "exec_comp", "directors", "director_comp", "votes", "ownership")}


def test_clean_is_idempotent() -> None:
    """Re-cleaning a cleaned bundle returns it unchanged and reports no fills."""
    raw, fundamentals = _bundle()
    once, stats = clean_governance_inputs(raw, fundamentals)
    twice, again = clean_governance_inputs(once, fundamentals)

    assert stats, "the fixture must exercise the stage"
    assert once.cleaned and not raw.cleaned
    assert again == {}, f"a second pass reported fills: {again}"
    for name, frame in _frames(once).items():
        other = _frames(twice)[name]
        if frame is None:
            assert other is None, name
        else:
            pd.testing.assert_frame_equal(frame, other, check_exact=True, obj=name)

    print("\n=== SANITY CHECK: the cleaning stage is idempotent ===")
    print(f"  first pass: {len(stats)} stats keys; second pass: {len(again)} keys and every frame identical. Validated.")
