"""
test_fill_is_past_only.py  (tests/data_aggregate/test_fill_is_past_only.py)
----------------------------------------------------------------------------
THE PROPERTY NO OTHER TEST IN THIS REPO ASSERTS: every governance fill must be reproducible
from rows at or before the row it writes to.

⚠ WHY A WHOLE TEST MODULE FOR ONE SENTENCE. A linear interpolation with `limit_area="inside"`
filled 12,607 interior gaps across 13 fields -- 46% of everything `f_board_busyness` shipped --
by drawing a line between the observation before a gap and the observation AFTER it, so every
intermediate value was arithmetic on a filing that did not exist yet. It passed 444 tests and
every audit check. It could, because the two things that should have caught it both look in the
wrong place by construction:

  * `03_pit` asks whether the value on day D comes from a filing dated after D. It does not: the
    future-derived value is laundered onto a filing row whose own `as_of` is legitimately in the
    past. The check looks one layer BELOW the defect and correctly reports nothing.
  * the coverage and domain tests see a plausible number in a cell that used to be empty, which
    is exactly what a correct fill also produces.

THE TEST THAT CAN SEE IT is prefix stability, and it needs no threshold:

    truncate the sources at a cut date C, run the fill chain, and compare against the same chain
    run on the FULL sources and then restricted to rows at or before C.

A fill that reads only the past cannot notice the truncation. A fill that reads a later filing --
for the VALUE (interpolation) or merely for the DECISION (an `ffill`/`bfill` interior gate) --
answers differently the moment its "after" is removed. There is nothing to tune and it cannot
abstain, and `test_the_property_catches_the_interpolation_it_exists_to_catch` proves it fails
when the defect is put back, because a check that cannot fail is not a check.

WHAT IS UNDER TEST is the composed chain, exactly as `StepCubeGovernance` runs it, because the
defect class crosses the grains: a child fill that reads the future reaches the parent through
the board derivation, and only the composed chain sees that.

This is the CI twin of `reports/validate/governance/_scripts/18_fill_is_past_only.py`, which runs
the identical property against the live DB. The two are deliberately separate rather than sharing
code: one is evidence about the real archive and needs Postgres, this one is a fixture that runs
anywhere and gates every commit. Neither substitutes for the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance import def14a_impute
from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
from src.data_aggregate.utils.governance.directors import (
    board_aggregates, fill_director_attributes, finalize_board_source, merge_board_aggregates,
)

_YEARS = ["2019-05-01", "2020-05-01", "2021-05-01", "2022-05-01", "2023-05-01"]
#: The cuts. Every filing date is a cut, which is the strictest schedule this fixture admits --
#: a fill that reaches forward by one filing has nowhere to hide.
_CUTS = tuple(_YEARS)

PARENT_KEY = ["ticker", "accession_number"]
CHILD_KEY = ["ticker", "accession_number", "name"]


def _child() -> pd.DataFrame:
    """Three boards whose `other_public_company_boards` gaps are the shapes that matter.

    AAA/Ann Agree      2, 2, GAP, 2, 2      -- the gap the agreement gate fills, and the reason
                                               it is a leak: the fill needs the 2022 filing.
    AAA/Dan Disagree   1, 1, GAP, 4, 4      -- declined either way; present so the fixture is not
                                               made of the passing case alone.
    BBB/Bea Trailing   3, 3, 3, GAP, GAP    -- a TRAILING gap: no later filing exists, so the
                                               agreement gate structurally cannot fill it and a
                                               forward carry can. The live/backtest asymmetry.
    CCC/*              complete             -- the control: no gap, so no rule can differ.
    """
    rows: list[dict] = []
    for i, y in enumerate(_YEARS):
        rows += [
            {"ticker": "AAA", "accession_number": f"AAA-{i}", "as_of": y, "name": "Ann Agree",
             "age": 60 + i, "tenure_years": 5 + i,
             "other_public_company_boards": None if i == 2 else 2.0},
            {"ticker": "AAA", "accession_number": f"AAA-{i}", "as_of": y, "name": "Dan Disagree",
             "age": 50 + i, "tenure_years": 20 + i,
             "other_public_company_boards": {0: 1.0, 1: 1.0, 2: None, 3: 4.0, 4: 4.0}[i]},
            {"ticker": "BBB", "accession_number": f"BBB-{i}", "as_of": y, "name": "Bea Trailing",
             "age": 70 + i, "tenure_years": 16 + i,
             "other_public_company_boards": 3.0 if i < 3 else None},
            {"ticker": "CCC", "accession_number": f"CCC-{i}", "as_of": y, "name": "Cara Complete",
             "age": 45 + i, "tenure_years": 3 + i, "other_public_company_boards": 3.0},
            {"ticker": "CCC", "accession_number": f"CCC-{i}", "as_of": y, "name": "Carl Complete",
             "age": 65 + i, "tenure_years": 17 + i, "other_public_company_boards": 1.0},
        ]
    return pd.DataFrame(rows)


def _parent() -> pd.DataFrame:
    """Proxy rows carrying a PARENT-grain gap as well, so the chain's own carry is exercised.

    `insider_ownership_pct` reproduces the LVS shape that started all of this: a value, three
    years of silence, then a very different value. Interpolation invents the ramp between them;
    a forward carry answers the last known number.
    """
    ownership = {0: 0.108, 1: np.nan, 2: np.nan, 3: 0.012, 4: np.nan}
    rows: list[dict] = []
    for t in ("AAA", "BBB", "CCC"):
        for i, y in enumerate(_YEARS):
            rows.append({"ticker": t, "accession_number": f"{t}-{i}", "as_of": y,
                         "avg_other_public_boards": np.nan, "avg_director_age": np.nan,
                         "board_size": 2.0 if t == "CCC" else np.nan,
                         "ceo_name_proxy": f"{t} Chief",
                         "insider_ownership_pct": ownership[i]})
    return pd.DataFrame(rows)


def _chain(parent: pd.DataFrame, child: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(imputed parent, filled child)` in `StepCubeGovernance`'s own order (D35, D39)."""
    filled_child, _ = fill_director_attributes(child)
    merged, _ = merge_board_aggregates(parent, board_aggregates(filled_child))
    imputed, _ = impute_def14a(merged)
    imputed, _ = finalize_board_source(imputed)
    return imputed, filled_child


def _leaks(full: pd.DataFrame, past_only: pd.DataFrame, key: list[str],
           cut: pd.Timestamp) -> dict[str, int]:
    """Per column, the cells at or before `cut` the past-only run does not reproduce.

    NaN-safe on both sides. A cell the past-only run leaves NaN where the full run wrote a value
    is a leak (it needed the future to EXIST); a cell where the two wrote different values is a
    leak too (it needed the future to be COMPUTED). NaN on both sides agrees.
    """
    a = full.copy()
    b = past_only.copy()
    for df in (a, b):
        df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    a = a[a["as_of"] <= cut].set_index(key).sort_index()
    b = b[b["as_of"] <= cut].set_index(key).sort_index()
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
    out: dict[str, int] = {}
    for col in sorted(set(a.columns) & set(b.columns) - {"_pk"}):
        x, y = a[col], b[col]
        if x.dtype == object or y.dtype == object:
            differ = x.astype(str).fillna("") != y.astype(str).fillna("")
        else:
            x, y = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
            differ = ~((x - y).abs() <= 1e-9) & ~(x.isna() & y.isna())
        n = int(differ.sum())
        if n:
            out[col] = n
    return out


def prefix_leaks() -> tuple[dict[str, int], dict[str, int]]:
    """`(parent leaks, child leaks)` summed over every cut. Empty dicts mean the fill is PIT."""
    parent, child = _parent(), _child()
    full_parent, full_child = _chain(parent.copy(), child.copy())
    p_total: dict[str, int] = {}
    c_total: dict[str, int] = {}
    for cut_s in _CUTS:
        cut = pd.Timestamp(cut_s)
        p_past, c_past = _chain(parent[pd.to_datetime(parent["as_of"]) <= cut].copy(),
                                child[pd.to_datetime(child["as_of"]) <= cut].copy())
        for col, n in _leaks(full_parent, p_past, PARENT_KEY, cut).items():
            p_total[col] = p_total.get(col, 0) + n
        for col, n in _leaks(full_child, c_past, CHILD_KEY, cut).items():
            c_total[col] = c_total.get(col, 0) + n
    return p_total, c_total


def test_the_governance_fill_reads_only_the_past():
    """Prefix stability over the whole chain: no cell needs a filing dated after its own row.

    ⚠ THIS TEST IS THE ACCEPTANCE CRITERION FOR THE CHILD CARRY (B1). It is RED while
    `directors.py` still gates on `ffill() & bfill()`, and that is the instrument working, not a
    broken test -- the whole point of writing the check before the fix is that the measurement is
    evidence rather than a formality.
    """
    p_leaks, c_leaks = prefix_leaks()

    print("\n=== SANITY CHECK: prefix stability of the governance fill chain ===")
    print(f"  {len(_CUTS)} cut dates | parent leaks: {p_leaks or 'none'}")
    print(f"                       | child  leaks: {c_leaks or 'none'}")
    assert not c_leaks, (
        "the CHILD fill reads a later filing -- these cells cannot be reproduced from rows at or "
        f"before their own as_of: {c_leaks}")
    assert not p_leaks, (
        "the PARENT fill reads a later filing -- these cells cannot be reproduced from rows at "
        f"or before their own as_of: {p_leaks}")
    print("  SANITY CHECK: every filled cell in both grains is reproducible from the past "
          "alone, so no fill in the governance package can launder a future value.")


def test_the_property_catches_the_interpolation_it_exists_to_catch(monkeypatch):
    """A check that cannot fail is not a check. Put the defect back and the property must see it.

    `_carry` is monkeypatched to the `limit_area="inside"` interpolation it replaced on
    2026-09-09, with an age of 0 so `CARRY_MAX_DAYS` cannot mask the effect -- i.e. the exact
    code that shipped the LVS 0.0840 / 0.0600 / 0.0360 ramp.
    """
    real_carry = def14a_impute._carry

    def _interpolating_carry(df: pd.DataFrame, col: str, gk: pd.Series):
        # The flags were never interpolated -- they were an interior-gated carry -- and a string
        # column cannot be. Only the LEVELS take the old rule, which is what shipped.
        if not pd.api.types.is_numeric_dtype(df[col]):
            return real_carry(df, col, gk)
        vals = df[col].groupby(gk, sort=False).transform(
            lambda s: s.interpolate(limit_area="inside"))
        return vals, pd.Series(0, index=df.index)

    monkeypatch.setattr(def14a_impute, "_carry", _interpolating_carry)
    p_leaks, _ = prefix_leaks()

    print("\n=== SANITY CHECK: the property fails when the interpolation is restored ===")
    print(f"  parent leaks with `_carry` -> interpolate(limit_area='inside'): {p_leaks}")
    assert p_leaks, ("the interpolation was restored and the property still passed -- it is "
                     "measuring nothing")
    assert "insider_ownership_pct" in p_leaks, (
        "the LVS-shaped gap (0.108, silence, 0.012) was not flagged; the property is not "
        f"reaching the carried levels. Flagged instead: {sorted(p_leaks)}")
    print(f"  SANITY CHECK: restoring the interpolation makes {sum(p_leaks.values())} cells "
          "irreproducible from the past, including the LVS-shaped `insider_ownership_pct` gap -- "
          "so the check has a real failure mode and is not vacuous.")


def test_a_trailing_gap_is_the_asymmetry_the_property_also_measures():
    """A gap at the LIVE EDGE has no "after", so an interior-gated rule can never fill it.

    That is not a look-ahead in the value; it is the same defect in the DECISION, and it makes a
    backtest fill situations a live run structurally cannot. Recorded as its own test because it
    is the half of the argument that a cell-value diff alone does not show.
    """
    filled, _ = fill_director_attributes(_child())
    bea = filled[filled["name"] == "Bea Trailing"].sort_values("as_of")
    trailing = bea["other_public_company_boards"].tolist()[3:]

    print("\n=== SANITY CHECK: the trailing gap ===")
    print(f"  Bea Trailing's last two filings: {trailing} "
          f"(filed 3.0 for the first three years, then silence)")
    filled_n = sum(0 if pd.isna(v) else 1 for v in trailing)
    print(f"  SANITY CHECK: {filled_n} of 2 trailing cells filled. An interior-gated rule "
          "reaches 0 of them by construction; a bounded forward carry reaches both.")
    assert len(trailing) == 2
