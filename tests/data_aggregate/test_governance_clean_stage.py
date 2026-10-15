"""
The governance cleaning stage (src/data_aggregate/utils/governance/clean.py).

  * a static guard: every impute / carry / repair / reconcile helper is called from the stage
    only, never from a feature module, the builder or the step;
  * cleaning the same raw input twice gives identical frames, and re-cleaning keeps the filed snapshots;
  * the stats keys are stable `<stage>|<rule>` counts covering every stage (REQ-010).
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.clean import VOTE_SOURCE_PROXY, GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.def14a_impute import fill_kind_column
from src.data_aggregate.utils.governance.staleness import LEVEL_MAX_AGE_DAYS, expire_stale, source_date_column

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
        "estimate_insider_ownership",
        "sct_pay_growth",
        "backfill_ceo_pay",
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


def _fresh() -> tuple[GovernanceInputs, pd.DataFrame]:
    """`_bundle` plus filed start years and say-on-pay support (no 8-K votes, so every filed
    support becomes a proxy fallback vote)."""
    raw, fundamentals = _bundle()
    d = cast(pd.DataFrame, raw.def14a).assign(
        ceo_since_year=[2015.0, np.nan, 2015.0, np.nan, 2010.0, 2012.0], say_on_pay_support_pct=[0.9, 0.8, np.nan, 0.95, np.nan, 0.7]
    )
    return replace(raw, def14a=d), fundamentals


def test_clean_is_idempotent() -> None:
    """The same raw input cleans to identical frames twice; a cleaned bundle is returned as is; and
    cleaned frames re-wrapped as raw keep their filed `_raw` values and the vote `source`."""
    raw, fundamentals = _fresh()
    once, stats = clean_governance_inputs(raw, fundamentals)
    other, stats_other = clean_governance_inputs(_fresh()[0], fundamentals)
    assert stats and stats == stats_other
    for name, frame in _frames(once).items():
        twin = _frames(other)[name]
        if frame is None:
            assert twin is None, name
        else:
            pd.testing.assert_frame_equal(frame, twin, check_exact=True, obj=name)

    twice, again = clean_governance_inputs(once, fundamentals)
    assert again == {} and twice is once

    again_raw, _ = clean_governance_inputs(replace(once, cleaned=False), fundamentals)
    d = cast(pd.DataFrame, again_raw.def14a).set_index("accession_number")
    filed = cast(pd.DataFrame, raw.def14a).set_index("accession_number")["ceo_since_year"]
    pd.testing.assert_series_equal(d["ceo_since_year_raw"].reindex(filed.index), filed, check_names=False)
    votes, first_votes = cast(pd.DataFrame, again_raw.votes), cast(pd.DataFrame, once.votes)
    assert len(votes) == len(first_votes) == 4, "re-cleaning must not add fallback rows twice"
    assert (votes["source"] == VOTE_SOURCE_PROXY).all(), f"fallback rows relabelled: {votes['source'].tolist()}"

    print("\n=== SANITY CHECK: the cleaning stage is idempotent ===")
    print(f"  two fresh raw passes: {len(stats)} identical stats keys and identical frames; re-wrapped pass keeps _raw and {len(votes)} proxy votes")
    print("  CONCLUSION: cleaning is a pure function of the raw input, and a second pass cannot corrupt the filed snapshots.")


def _expired_daily(def14a: pd.DataFrame, col: str, days: list[str]) -> pd.Series:
    """`col` for the one fixture ticker on `days`, through the panel's real expiry path."""
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    daily = expire_stale(fundamentals_to_daily(def14a, col, idx), def14a, col, max_age_days=LEVEL_MAX_AGE_DAYS, feature=col)
    return daily.iloc[:, 0].reindex(pd.DatetimeIndex(days))


def test_carried_value_expires_at_its_true_age() -> None:
    """A 2016 carry retains the 2015 source clock; the 2017 proxy cannot carry it again."""
    raw = pd.DataFrame(
        {
            "ticker": ["CCC"] * 3,
            "accession_number": ["C-0", "C-1", "C-2"],
            "as_of": ["2015-03-01", "2016-03-01", "2017-03-01"],
            "ceo_name_proxy": ["Ann Chief"] * 3,
            "insider_ownership_pct": [0.05, np.nan, np.nan],
        }
    )
    cleaned, _ = clean_governance_inputs(GovernanceInputs(def14a=raw), None)
    d = cast(pd.DataFrame, cleaned.def14a)

    carried = d["insider_ownership_pct"].tolist()
    stamp = pd.to_datetime(d[source_date_column("insider_ownership_pct")]).tolist()
    kinds = d[fill_kind_column("insider_ownership_pct")].tolist()
    cells = _expired_daily(d, "insider_ownership_pct", ["2016-03-30", "2016-03-31", "2017-06-01"])

    assert carried[:2] == [0.05, 0.05] and np.isnan(carried[2]), carried
    assert stamp[:2] == [pd.Timestamp("2015-03-01")] * 2 and pd.isna(stamp[2]), stamp
    assert kinds[:2] == ["filed", "carry"], kinds
    assert cells.iloc[0] == 0.05, "day 395 of the original source remains current"
    assert cells.iloc[1:].isna().all(), "a carry cannot extend the source clock beyond day 395"

    print("\n=== SANITY CHECK: a carried value ages from the filing that disclosed it ===")
    print(f"  day 395 -> {cells.iloc[0]}; day 396 -> {cells.iloc[1]}; 2017 proxy -> {carried[2]}. Validated.")


def test_derived_value_takes_its_oldest_leg_date() -> None:
    """A total summed from a carried salary is dated at the salary's source filing."""
    comps = {"ceo_bonus": 1.0e6, "ceo_stock_awards": 2.0e6, "ceo_option_awards": 0.0, "ceo_non_equity_incentive": 5.0e5, "ceo_all_other_comp": 1.0e5}
    raw = pd.DataFrame(
        {
            "ticker": ["DDD"] * 2,
            "accession_number": ["D-0", "D-1"],
            "as_of": ["2020-03-01", "2021-03-01"],
            "ceo_name_proxy": ["Bo Lead"] * 2,
            "ceo_salary": [1.0e6, np.nan],
            "ceo_total_comp": [4.6e6, np.nan],
            **{c: [v, v] for c, v in comps.items()},
        }
    )
    cleaned, _ = clean_governance_inputs(GovernanceInputs(def14a=raw), None)
    d = cast(pd.DataFrame, cleaned.def14a)
    total_stamp = pd.to_datetime(d[source_date_column("ceo_total_comp")]).tolist()

    assert d["ceo_total_comp"].tolist() == [4.6e6, 4.6e6]
    assert d[fill_kind_column("ceo_salary")].tolist() == ["filed", "carry"]
    assert d[fill_kind_column("ceo_total_comp")].tolist() == ["filed", "identity"]
    assert total_stamp == [pd.Timestamp("2020-03-01")] * 2, f"the identity inherits the carried leg's date: {total_stamp}"

    print("\n=== SANITY CHECK: a derived value is as old as its oldest leg ===")
    print(f"  2021 total = salary carried from 2020 + filed components -> stamped {total_stamp[1].date()}, kind identity. Validated.")


def test_ownership_writes_reset_provenance() -> None:
    """The ownership blank clears the stamp and kind; the computed value is stamped at its proxy."""
    raw, fundamentals = _bundle()
    cleaned, _ = clean_governance_inputs(raw, fundamentals)
    d = cast(pd.DataFrame, cleaned.def14a).set_index("accession_number")
    src, kind = source_date_column("insider_ownership_pct"), fill_kind_column("insider_ownership_pct")

    blanked, computed, carried = d.loc["B-0"], d.loc["A-0"], d.loc["A-2"]
    assert pd.isna(blanked["insider_ownership_pct"]) and pd.isna(blanked[src]) and pd.isna(blanked[kind])
    assert computed["insider_ownership_pct"] == 0.01 and computed[kind] == "cross_source"
    assert pd.Timestamp(computed[src]) == pd.Timestamp("2021-03-01")
    assert carried[kind] == "carry" and pd.Timestamp(carried[src]) == pd.Timestamp("2022-03-01")

    print("\n=== SANITY CHECK: every ownership write resets its provenance ===")
    print("  B-0 blanked -> value, stamp and kind all empty; A-0 computed -> cross_source @ 2021-03-01; A-2 carry @ 2022-03-01. Validated.")


#: Every stage the cleaning stage reports under, in run order (REQ-010).
_STAGES = (
    "exec_comp",
    "directors",
    "director_comp",
    "board_merge",
    "impute_def14a",
    "board_source",
    "ownership",
    "ownership_estimate",
    "sct_growth",
    "pay_backfill",
    "votes_fallback",
)


def _tally_bundle() -> tuple[GovernanceInputs, pd.DataFrame]:
    """`_bundle` plus a CEO succession, stint constants, say-on-pay, votes and holder rows."""
    raw, fundamentals = _bundle()
    d = raw.def14a.copy()
    d["ceo_name_proxy"] = ["Jane Doe"] * 3 + ["John Roe", "John Roe", "Mary Poe"]
    d["ceo_since_year"] = [2015.0, np.nan, 2016.0, 2010.0, np.nan, np.nan]
    d["ceo_is_founder"] = [0.0, np.nan, 0.0, 1.0, np.nan, np.nan]
    d["ceo_is_board_chair"] = [1.0, np.nan, 1.0, 1.0, 1.0, np.nan]
    d["say_on_pay_support_pct"] = [0.90, 0.92, np.nan, 0.62, np.nan, 0.70]
    votes = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "accession_number": ["8K-A"],
            "proposal_seq": [1.0],
            "filing_date": pd.to_datetime(["2021-05-01"]),
            "proposal_type": ["say_on_pay"],
            "votes_for": [9.0e7],
            "votes_against": [1.0e7],
            "votes_abstain": [0.0],
        }
    )
    ownership = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "accession_number": ["A-2"] * 6,
            "holder_type": ["director_officer"] * 6,
            "shares": [2.0e5] * 6,
        }
    )
    bundle = GovernanceInputs(
        def14a=d, exec_comp=raw.exec_comp, directors=raw.directors, director_comp=raw.director_comp, votes=votes, ownership=ownership
    )
    return bundle, fundamentals


def test_clean_stage_tally_keys() -> None:
    """REQ-010: stats are `<stage>|<rule>` integer counts, from known stages only, identical on a
    fresh re-run, and they carry the filled / declined / overridden and stint-boundary counts."""
    raw, fundamentals = _tally_bundle()
    _, stats = clean_governance_inputs(raw, fundamentals)
    raw2, fundamentals2 = _tally_bundle()
    _, stats2 = clean_governance_inputs(raw2, fundamentals2)

    assert stats == stats2, "the tally is not stable across identical runs"
    assert all("|" in k for k in stats), [k for k in stats if "|" not in k]
    stages = {k.split("|", 1)[0] for k in stats}
    assert stages == set(_STAGES), f"stages unknown {sorted(stages - set(_STAGES))}, silent {sorted(set(_STAGES) - stages)}"
    assert all(isinstance(v, int) and v >= 0 for v in stats.values())

    rules = {k.split("|", 1)[1] for k in stats}
    required = {
        "person: ceo_since_year",  # stint-constant fills
        "person overrides a filed value: ceo_since_year",
        "declined (stint boundary): ceo_is_board_chair",  # carry refused at a stint boundary
        "ceo_total_comp filled from the filing's own SCT row",
    }
    assert required <= rules, f"missing REQ-010 rules: {sorted(required - rules)}"
    assert any(k.startswith("votes_fallback|say_on_pay: fallback vote rows ADDED") for k in stats)
    assert any(k.startswith("votes_fallback|say_on_pay: proxy value NOT used") for k in stats)
    assert any(k.startswith("impute_def14a|declined") for k in stats)

    print("\n=== SANITY CHECK: the cleaning-stage tally is stable and complete ===")
    print(f"  {len(stats)} keys over {len(stages)} stages ({', '.join(s for s in _STAGES if s in stages)}),")
    print("  identical on a re-run; stint fills, overrides, stint-boundary refusals and the vote fallback all counted. Validated.")
