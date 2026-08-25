"""
test_ledger.py  (tests/validate/fundamentals/)
--------------------------------------------------------------------------------------------
COMPARABILITY, on a synthetic ledger. No DB.

One property carries the whole loop: a drop in row count between two runs proves a fix ONLY
when the two runs looked at the same thing. Everything here exists to make that testable
rather than assumed -- a 54-ticker baseline differenced against a one-ticker re-validation
would report ~11,800 findings "closed", and every one of them would read as a triumph.
"""
from __future__ import annotations

import pandas as pd

from src.validate.fundamentals.ledger import Ledger
from src.validate.fundamentals.scope import RunScope


def _runs(*specs) -> pd.DataFrame:
    """`(run_id, run_date, scope_hash, tickers, roster)` -> a `fundamentals_check_run` frame."""
    return pd.DataFrame([{
        "run_id": run_id, "run_date": pd.Timestamp(run_date), "check_name": check,
        "scope_hash": scope_hash, "scope_roster": roster, "scope_tickers": tickers,
        "scope_ticker_list": "[]", "scope_fields": "[]", "scope_tiers": "[1, 2, 3]",
        "tier": 2, "substrate": "facts", "examined": 100, "queued": 5, "info": 0,
        "ceiling": 0.06, "abstained": False, "over_ceiling": False,
    } for run_id, run_date, scope_hash, tickers, roster in specs
        for check in ("peer_ratio", "trend_break")])


#: Three runs: two of ONE scope on different days, one of a narrower scope in between.
_LEDGER = Ledger(
    findings=pd.DataFrame([
        {"run_id": "wide0001", "run_date": pd.Timestamp("2026-08-20"),
         "cluster_id": "c1", "ticker": "AAPL", "field": "capex"},
        {"run_id": "wide0001", "run_date": pd.Timestamp("2026-08-20"),
         "cluster_id": "c2", "ticker": "MCD", "field": "capex"},
        {"run_id": "wide0002", "run_date": pd.Timestamp("2026-08-24"),
         "cluster_id": "c1", "ticker": "AAPL", "field": "capex"},
        {"run_id": "narrow01", "run_date": pd.Timestamp("2026-08-22"),
         "cluster_id": "c1", "ticker": "AAPL", "field": "capex"},
    ]),
    runs=_runs(("wide0001", "2026-08-20", "scopeAAA", 54, "in_sample"),
               ("narrow01", "2026-08-22", "scopeBBB", 1, None),
               ("wide0002", "2026-08-24", "scopeAAA", 54, "in_sample")),
    status=pd.DataFrame())


def test_only_runs_of_the_SAME_scope_are_comparable() -> None:
    """The narrower run is not a worse comparison. It is not a comparison."""
    peers = _LEDGER.comparable_runs("wide0002")

    print(f"\ncomparable to wide0002: {[p.run_id for p in peers]}")
    print("  SANITY: narrow01 ran in between and is EXCLUDED -- its scope hash differs, so "
          "differencing against it would report a cluster it never looked at as 'closed'.")
    assert [p.run_id for p in peers] == ["wide0001"]


def test_the_previous_comparable_run_is_STRICTLY_earlier() -> None:
    """Comparing a run to a peer that came AFTER it reports a fix as a regression."""
    previous = _LEDGER.previous_comparable("wide0002")
    first = _LEDGER.previous_comparable("wide0001")

    print(f"\nwide0002 -> {previous.label if previous else None}")
    print(f"wide0001 -> {first.label if first else None} (the earliest run has no predecessor)")
    assert previous is not None and previous.run_id == "wide0001"
    assert first is None


def test_a_run_with_no_comparable_predecessor_returns_nothing_rather_than_guessing() -> None:
    """The report renders "no delta" with a reason. A first run is not a trend."""
    assert _LEDGER.previous_comparable("narrow01") is None
    print("\nnarrow01 has no comparable predecessor -> no delta section, with a stated reason")


def test_cluster_history_counts_runs_of_ONE_scope_only() -> None:
    """"How long has this been broken?" is only answerable within a scope.

    A cluster absent from a NARROWER run did not close -- it was never looked at -- so
    counting that run would understate how long the defect has survived.
    """
    history = _LEDGER.cluster_history("scopeAAA").set_index("cluster_id")

    print(f"\n{history[['first_seen', 'last_seen', 'runs_open']].to_string()}")
    print("  SANITY: c1 spans BOTH wide runs; c2 was seen once and is absent from the latest, "
          "which is what `settled_clusters` reads as a closed defect.")
    assert history.loc["c1", "runs_open"] == 2
    assert history.loc["c2", "runs_open"] == 1
    assert history.loc["c1", "first_seen"] == pd.Timestamp("2026-08-20")


def test_the_scope_hash_ignores_the_roster_NAME_but_not_its_contents() -> None:
    """Renaming a roster in configs/ must not look like a scope change."""
    same = RunScope.build(tickers=["A", "B"], fields=None, tiers=[1], roster="old_name")
    renamed = RunScope.build(tickers=["A", "B"], fields=None, tiers=[1], roster="new_name")
    changed = RunScope.build(tickers=["A", "C"], fields=None, tiers=[1], roster="old_name")

    print(f"\nsame contents, renamed roster: {same.scope_hash} == {renamed.scope_hash}")
    print(f"different contents           : {same.scope_hash} != {changed.scope_hash}")
    print("  SANITY: a roster is a LABEL for a ticker list. Two runs covering the same "
          "tickers are comparable whether or not someone renamed it in between.")
    assert same.scope_hash == renamed.scope_hash
    assert same.scope_hash != changed.scope_hash


def test_an_empty_ledger_answers_every_question_without_raising() -> None:
    """Day one. A validator that crashes on its first run is not much of a validator."""
    empty = Ledger(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    print(f"\nrun={empty.run('x')}  comparable={empty.comparable_runs('x')}  "
          f"status={empty.status_map()}  history_rows={len(empty.cluster_history('x'))}")
    assert empty.run("x") is None
    assert empty.comparable_runs("x") == [] and empty.previous_comparable("x") is None
    assert empty.status_map() == {} and empty.cluster_history("x").empty
