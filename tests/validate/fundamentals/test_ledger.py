"""
test_ledger.py  (tests/validate/fundamentals/)
--------------------------------------------------------------------------------------------
COMPARABILITY, on a synthetic ledger. No DB.

One property carries the whole loop: a drop in row count between two runs proves a fix ONLY
when the two runs looked at the same thing. Everything here exists to make that testable
rather than assumed -- a 54-ticker baseline differenced against a one-ticker re-validation
would report ~11,800 findings "closed", and every one of them would read as a triumph.

The second half of this file pins the two READ SHAPES that carry the fix-recording design:
a waiver map that must be NESTED (a flat one silently loses the second waiver on a cluster),
and `qualifying_fix`, which is the settlement predicate and lives in exactly one place.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.validate.fundamentals.ledger import CLUSTER_WIDE, Ledger
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


# --------------------------------------------------------------------------- #
# waivers, fixes, and the settlement predicate                                 #
# --------------------------------------------------------------------------- #

def _status(*specs) -> pd.DataFrame:
    """`(cluster_id, check_name, findings_at_decision)` -> a `fundamentals_check_status` frame.

    `decided_at` is a `datetime.date`, deliberately: that is what Postgres hands back, and a
    fixture using `Timestamp` would hide the exact bug class this module normalises for.
    """
    return pd.DataFrame([{
        "cluster_id": cluster, "check_name": check, "ticker": "MCD", "field": "capex",
        "status": "wontfix", "note": f"{at_decision} benign finding(s) measured",
        "findings_at_decision": at_decision, "decided_at": dt.date(2026, 8, 25),
    } for cluster, check, at_decision in specs])


def _fixes(*specs) -> pd.DataFrame:
    """`(cluster, run_after, scope_hash, queued_before, queued_after, day)` -> a fix frame."""
    return pd.DataFrame([{
        "cluster_id": cluster, "run_id_after": after, "run_id_before": "before01",
        "scope_hash": scope, "ticker": "MCD", "field": "capex",
        "findings_before": before_q + 1, "findings_after": after_q + 1,
        "queued_before": before_q, "queued_after": after_q,
        "layer": "extraction", "root_cause": "route 1 took a sibling total",
        "evidence": '{"accessions": ["0000063908-18-000010"]}',
        "commit_sha": "2fb6ef2", "test_path": "tests/data_extract/test_x.py",
        "decided_at": dt.date(2026, 8, day),
    } for cluster, after, scope, before_q, after_q, day in specs])


def test_two_waivers_on_ONE_cluster_do_not_collide() -> None:
    """The nested map is the whole ripple of widening the status primary key.

    A flat `{cluster_id: row}` under the wider key lets the last row read win silently, so a
    `peer_ratio` waiver would quietly become a cluster-wide one depending on row order.
    """
    ledger = Ledger(pd.DataFrame(), pd.DataFrame(),
                    _status(("c1", CLUSTER_WIDE, 9), ("c1", "peer_ratio", 2),
                            ("c2", "series_shape", 1)))
    waivers = ledger.waivers_for("c1")
    sizes = {k: v["findings_at_decision"] for k, v in sorted(waivers.items())}

    print(f"\nc1 waivers: {sorted(waivers)}   ('' == the whole cluster)")
    print(f"  findings_at_decision per check: {sizes}")
    print(f"c2 waivers: {sorted(ledger.waivers_for('c2'))}")
    print("  SANITY: two entries survive on c1, each with its OWN findings_at_decision. "
          "Under the old flat map one of them would have been silently overwritten.")
    assert set(waivers) == {CLUSTER_WIDE, "peer_ratio"}
    assert waivers["peer_ratio"]["findings_at_decision"] == 2
    assert set(ledger.waivers_for("c2")) == {"series_shape"}


def test_decided_at_is_normalised_off_a_postgres_DATE() -> None:
    """Postgres DATE arrives as `datetime.date`. Normalised ONCE here, never in a caller.

    A parquet-cached harness hands back `Timestamp` and hides this entire bug class, so the
    fixtures use `datetime.date` on purpose and the real load path is exercised.
    """
    class _Store:
        def load(self, table, columns=None, where=None, optional=False):
            name = str(table)
            if name == "fundamentals_check_status":
                return _status(("c1", "peer_ratio", 2))
            if name == "fundamentals_check_fix":
                return _fixes(("c1", "after001", "scopeAAA", 55, 4, 25))
            return pd.DataFrame()

    ledger = Ledger.load(type("Ctx", (), {"store": _Store()})())

    print(f"\nfixture cell type       : {type(dt.date(2026, 8, 25)).__name__}")
    print(f"status.decided_at dtype : {ledger.status['decided_at'].dtype}")
    print(f"fixes.decided_at  dtype : {ledger.fixes['decided_at'].dtype}")
    print("  SANITY: both are datetime64[ns]. Left as datetime.date, an unguarded "
          "`frame['decided_at'] > some_timestamp` does something surprising instead.")
    assert ledger.status["decided_at"].dtype == "datetime64[ns]"
    assert ledger.fixes["decided_at"].dtype == "datetime64[ns]"


def test_a_twice_fixed_cluster_keeps_BOTH_rows_newest_first() -> None:
    """Append-only. The second fix did not un-happen the first, and dropping it hides the
    attempt that failed -- which is the most useful row in the table."""
    ledger = Ledger(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    _fixes(("c1", "after001", "scopeAAA", 55, 30, 20),
                           ("c1", "after002", "scopeAAA", 30, 4, 25)))
    history = ledger.fixes_for("c1")

    print(f"\n{len(history)} fix row(s) on c1:")
    for record in history:
        print(f"  {record.run_id_after}  {record.decided_at.date()}  {record.summary}")
    print("  SANITY: newest first, and BOTH survive. A single-row read would report the "
          "cluster as fixed once, from 30, and lose that it started at 55.")
    assert [f.run_id_after for f in history] == ["after002", "after001"]
    assert history[0].queued_before == 30 and history[1].queued_before == 55


def test_qualifying_fix_refuses_a_no_improvement_row_and_a_wrong_scope_row() -> None:
    """The settlement predicate, in the ONE place it is written down.

    Both refusals matter and they fail for different reasons: a wrong-scope row was never
    measured against this run, and a no-improvement row was measured and closed nothing.
    Neither is illegitimate as a RECORD -- only as a proof.
    """
    ledger = Ledger(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    _fixes(("wrong", "after001", "scopeBBB", 55, 4, 25),
                           ("flat", "after002", "scopeAAA", 4, 4, 25),
                           ("good", "after003", "scopeAAA", 55, 4, 25)))

    wrong = ledger.qualifying_fix("wrong", "scopeAAA")
    flat = ledger.qualifying_fix("flat", "scopeAAA")
    good = ledger.qualifying_fix("good", "scopeAAA")

    print(f"\nwrong scope (scopeBBB vs scopeAAA) -> {wrong}")
    print(f"no improvement (4 -> 4)            -> {flat}")
    print(f"55 -> 4 at the right scope         -> {good.summary if good else None}")
    print(f"  the flat row is STILL on record  : "
          f"{[f.summary for f in ledger.fixes_for('flat')]}")
    print("  SANITY: permissive to record, strict to settle. Both refused rows stay "
          "readable through `fixes_for`; only `qualifying_fix` declines them.")
    assert wrong is None and flat is None
    assert good is not None and good.run_id_after == "after003"
    assert len(ledger.fixes_for("flat")) == 1


def test_an_empty_ledger_answers_the_fix_questions_too() -> None:
    """Day one, extended. Nothing has ever been fixed, and that must not raise."""
    empty = Ledger(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    print(f"\nwaivers={empty.waivers_for('x')}  fixes={empty.fixes_for('x')}  "
          f"qualifying={empty.qualifying_fix('x', 'scopeAAA')}")
    print("  SANITY: a cluster nobody has touched settles nothing and raises nothing.")
    assert empty.waivers_for("x") == {} and empty.fixes_for("x") == []
    assert empty.qualifying_fix("x", "scopeAAA") is None
