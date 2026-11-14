"""The growth-base kind table: which fiscal fields use `signed_pct` (negative bases exist) and which use plain `pct`."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fiscal_change_values

ROOT = Path(__file__).resolve().parents[2] / "src" / "data_aggregate" / "utils" / "fundamentals"
ENGINE_CALLS = {"fiscal_change_values", "fiscal_change_to_daily"}
EXPECTED_KINDS: dict[str, set[str]] = {
    "_basis_code": {"diff"},
    "_rpe": {"pct"},
    "capex": {"pct"},
    "depAmort": {"pct"},
    "dilutedShares": {"pct"},
    "dividendsPerShare": {"pct"},
    "freeCashflow": {"signed_pct"},
    "grossMargins": {"diff"},
    "intangibles": {"pct"},
    "netIncome": {"pct", "signed_pct"},
    "netIncome_q": {"signed_pct"},
    "operatingIncome": {"signed_pct"},
    "profitMargins": {"diff"},
    "revenue_q": {"pct"},
    "sellingGeneralAdmin": {"pct"},
    "sharesOutstanding": {"pct"},
    "totalRevenue": {"pct"},
}


def _declared_kinds() -> dict[str, set[str]]:
    """{field: kinds} of every constant-field growth call and every (name, field, kind) constant triple in the fundamentals builders."""
    kinds: dict[str, set[str]] = {}
    for module in ("fundamental_features.py", "employee_features.py"):
        for node in ast.walk(ast.parse((ROOT / module).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ENGINE_CALLS and len(node.args) > 1:
                field, kind = node.args[1], next((k.value for k in node.keywords if k.arg == "kind"), None)
                if isinstance(field, ast.Constant) and isinstance(kind, ast.Constant):
                    kinds.setdefault(str(field.value), set()).add(str(kind.value))
            if isinstance(node, ast.Tuple) and len(node.elts) == 3 and all(isinstance(e, ast.Constant) for e in node.elts):
                name, field, kind = (e.value for e in node.elts if isinstance(e, ast.Constant))
                if kind in ("pct", "signed_pct", "diff") and isinstance(field, str):
                    kinds.setdefault(field, set()).add(kind)
    return kinds


def _history(values: list[float]) -> pd.DataFrame:
    dates = pd.to_datetime(["2020-03-31", "2021-03-31", "2022-03-31"])
    return pd.DataFrame({"ticker": "AAA", "as_of": dates + pd.Timedelta(days=30), "fiscal_end": dates, "netIncome": values})


def test_negative_base_reads_the_right_way_only_with_signed_pct() -> None:
    history = _history([-10.0, -5.0, 5.0])
    pct = fiscal_change_values(history, "netIncome", kind="pct", periods=1, years=1)
    signed = fiscal_change_values(history, "netIncome", kind="signed_pct", periods=1, years=1)
    assert np.isclose(pct.iloc[1], -0.5), "plain pct reads an improvement from -10 to -5 as a 50% fall"
    assert np.isclose(signed.iloc[1], 0.5), "signed pct reads it as a 50% improvement"
    assert np.isclose(signed.iloc[2], 2.0), "from -5 to 5 is +200% of the base magnitude"
    print(f"SANITY: -10 -> -5 gives pct {pct.iloc[1]:+.2f} but signed_pct {signed.iloc[1]:+.2f}; -5 -> 5 gives {signed.iloc[2]:+.2f}.")


def test_the_kind_table_is_the_declared_one() -> None:
    kinds = _declared_kinds()
    assert kinds == EXPECTED_KINDS, (
        f"the growth kind table moved: {sorted(k for k in set(kinds) | set(EXPECTED_KINDS) if kinds.get(k) != EXPECTED_KINDS.get(k))}"
    )
    signed = sorted(field for field, declared in kinds.items() if "signed_pct" in declared)
    print(
        f"SANITY: signed_pct on {signed}; netIncome keeps one pct read (the PEGY growth leg, only positive values used); {len(kinds) - len(signed)} other fields keep pct or diff."
    )
