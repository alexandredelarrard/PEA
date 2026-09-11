"""
test_fill_respects_the_staleness_horizon.py
--------------------------------------------------------------------------------
NO FILL MAY WRITE A VALUE OLDER THAN THE HORIZON ITS FEATURE IS ON (B3b).

⚠ THE CLOCK CANNOT TELL A FILLED CELL FROM A FILED ONE, and that is a defect in the mechanism,
not in any one call site. `staleness.expire_stale` dates a cell by the most recent filing that
CARRIES the field:

    h = h.dropna(subset=["ticker", "as_of", field])   # keep only rows that carry the field
    h["_produced_at"] = h["as_of"]                    # then date the cell by that row

That is correct for a filed value and wrong for an imputed one. A value the imputer copied onto a
2026 filing row is dated 2026 and reads as age 0, no matter how old the observation it came from.
Measured while sizing phase 7: **21,518 carryable cells, of which 10,400 (48%) copy an observation
more than 1,095 days old**, the worst at 10,984 days -- 30.1 years, on `ceo_ownership_pct`. The
interpolation that preceded the carry laundered the same way, reaching 10,593 days on
`avg_other_public_boards`.

So any fill upstream can push a stale value straight through the horizon phase 3 built, invisibly.

TWO WAYS TO CLOSE IT, and this module is the cheaper one:

  * **B3a** -- thread the true vintage: have the imputer emit `<field>_as_of` beside each filled
    value and have `expire_stale` prefer it over the row's own `as_of`. The honest fix. It touches
    the imputer, the panel and every family wrapper, and is recorded as the real fix rather than
    ridden along with a bugfix phase.
  * **B3b** -- assert the invariant instead, which is this module. It does not make the clock
    truthful; it makes the thing the clock cannot see impossible to introduce.

⚠ DOING NEITHER WAS NOT AN OPTION. `def14a_impute.CARRY_MAX_DAYS` is currently the ONLY thing
holding this closed, it is one edit from being removed by someone who does not know why it is
there, and nothing in the suite would have gone red. This module is what goes red.

THE FIXTURE IS BUILT SO THE CARRY IS THE ONLY MECHANISM AVAILABLE. `impute_def14a` also fills
from within-row identities (the CEO pay sum, `n_directors == board_size`, the pay-ratio triple)
and from the `ceo_age` accrual, and none of those has a "source age" -- they read the row they
write to. The fixture therefore omits every identity PARTNER, so a cell that goes from NaN to a
value can only have been carried, and its source age is exactly measurable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance import def14a_impute
from src.data_aggregate.utils.governance.def14a_impute import (
    CARRY_FORBIDDEN, CARRY_LEVELS, CARRY_MAX_DAYS, FLAGS, IDENTITY_GATED_CARRY, impute_def14a,
)
from src.data_aggregate.utils.governance.directors import fill_director_attributes
from src.data_aggregate.utils.governance.staleness import LEVEL_MAX_AGE_DAYS, horizon_for

#: Columns the fixture carries. Every one is in `CARRY_LEVELS`, `IDENTITY_GATED_CARRY` or
#: `FLAGS`; none is half of an identity `_reconcile_rows` could satisfy instead.
_CARRIED = ["insider_ownership_pct", "pct_independent_directors", "avg_other_public_boards",
            "say_on_pay_support_pct", "poison_pill", "majority_voting", "independent_chair",
            "ceo_salary"]

#: `(ticker, filing dates, the index of the row that DISCLOSES a value)`. Everything else is a
#: gap, so each ticker states one gap width in days.
#:   FRESH  2,192-day archive, disclosed at both ends of a 731-day hole  -> inside the cap
#:   STALE  disclosed once in 2010, silent until 2019                    -> 3,287 days, refused
#:   EDGE   disclosed at 1,095 days exactly                              -> the boundary itself
_ARCHIVES: dict[str, tuple[list[str], list[int]]] = {
    "FRESH": (["2019-05-01", "2020-05-01", "2021-05-01"], [0, 2]),
    "STALE": (["2010-05-01", "2019-05-01"], [0]),
    "EDGE":  (["2019-05-01", "2022-04-30"], [0]),          # 2019-05-01 -> 2022-04-30 = 1,095 d
}


def _parent() -> pd.DataFrame:
    rows: list[dict] = []
    for ticker, (dates, disclosed) in _ARCHIVES.items():
        for i, d in enumerate(dates):
            row = {"ticker": ticker, "accession_number": f"{ticker}-{i}", "as_of": d,
                   "ceo_name_proxy": f"{ticker} Chief"}
            for col in _CARRIED:
                row[col] = (0.5 if col not in ("poison_pill", "majority_voting",
                                               "independent_chair") else 1.0) \
                    if i in disclosed else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _source_age_days(raw: pd.DataFrame, filled: pd.DataFrame, col: str,
                     key: list[str], group: str) -> pd.Series:
    """For each row, `as_of - as_of of the last row whose RAW `col` was disclosed`, in days.

    The same payload trick `_carry` and `expire_stale` both use: carry the source DATE forward
    alongside the value, so "age" means the same thing in the fill and in the clock.
    """
    r = raw.copy()
    r["as_of"] = pd.to_datetime(r["as_of"], errors="coerce")
    r = r.sort_values([group, "as_of"])
    src = r["as_of"].where(r[col].notna()).groupby(r[group], sort=False).ffill()
    age = (r["as_of"] - src).dt.days
    age.index = pd.MultiIndex.from_frame(r[key])
    f = filled.copy()
    f["as_of"] = pd.to_datetime(f["as_of"], errors="coerce")
    return age.reindex(pd.MultiIndex.from_frame(f[key]))


def _carried_cells(raw: pd.DataFrame, filled: pd.DataFrame, col: str,
                   key: list[str], group: str) -> pd.Series:
    """The source ages of the cells this fill actually WROTE (NaN before, a value after)."""
    r = raw.set_index(pd.MultiIndex.from_frame(raw[key]))[col]
    f = filled.set_index(pd.MultiIndex.from_frame(filled[key]))[col]
    newly = r.reindex(f.index).isna() & f.notna()
    return _source_age_days(raw, filled, col, key, group)[newly.values]


def test_the_carry_cap_can_never_exceed_the_staleness_horizon():
    """The structural half: raising `CARRY_MAX_DAYS` above the level horizon must go red here.

    This is the guard that was missing. The cap is not a taste parameter -- it is the ONLY reason
    a carried value cannot be laundered past the horizon `expire_stale` enforces, because the
    clock dates the cell to the row it landed on and so cannot enforce it itself.
    """
    assert CARRY_MAX_DAYS <= LEVEL_MAX_AGE_DAYS, (
        f"CARRY_MAX_DAYS={CARRY_MAX_DAYS} exceeds LEVEL_MAX_AGE_DAYS={LEVEL_MAX_AGE_DAYS}. The "
        "imputer would then write values that `expire_stale` believes are fresh, because it "
        "dates a filled cell by the row it landed on. Raise the horizon or lower the cap; they "
        "cannot disagree.")
    strictest = min(horizon_for(f) for f in ("board_size", "f_sop_dissent"))
    print("\n=== SANITY CHECK: the cap versus the clock ===")
    print(f"  CARRY_MAX_DAYS={CARRY_MAX_DAYS} <= LEVEL_MAX_AGE_DAYS={LEVEL_MAX_AGE_DAYS} "
          f"(the event horizon is {strictest}d, and nothing here is carried onto it)")
    print("  SANITY CHECK: no carried value can outlive the horizon that is supposed to "
          "expire it.")


def test_no_carried_cell_is_older_than_the_cap():
    """The behavioural half, over every carried field: a fill's SOURCE must be within the cap."""
    raw = _parent()
    filled, stats = impute_def14a(raw.copy())
    key, group = ["ticker", "accession_number"], "ticker"

    fields = [c for c in CARRY_LEVELS + sorted(IDENTITY_GATED_CARRY) + FLAGS
              if c in raw.columns and c not in CARRY_FORBIDDEN]
    violations: dict[str, list[int]] = {}
    filled_ages: dict[str, list[int]] = {}
    for col in fields:
        ages = _carried_cells(raw, filled, col, key, group).dropna()
        filled_ages[col] = sorted(int(a) for a in ages)
        over = sorted(int(a) for a in ages if a > CARRY_MAX_DAYS)
        if over:
            violations[col] = over

    print("\n=== SANITY CHECK: the age of every carried cell ===")
    for col in fields:
        print(f"  {col:<28} filled at ages (days): {filled_ages[col] or 'nothing filled'}")
    assert not violations, (
        f"a carry wrote a value older than CARRY_MAX_DAYS={CARRY_MAX_DAYS}: {violations}. "
        "`expire_stale` cannot see this -- it dates the cell to the row it landed on.")
    # the fixture must actually exercise the rule, or the assertion above is vacuous
    assert any(filled_ages[c] for c in fields), "nothing was filled; the fixture proves nothing"
    assert stats, "the imputer reported no work at all"
    print(f"  SANITY CHECK: {sum(len(v) for v in filled_ages.values())} cells carried across "
          f"{len(fields)} fields, every one of them sourced within {CARRY_MAX_DAYS} days; the "
          "3,287-day STALE gap was refused.")


def test_the_invariant_is_load_bearing_and_not_vacuous(monkeypatch):
    """Raise the cap and the STALE ticker's 3,287-day gap gets filled -- so the guard bites.

    A test that would pass with the protection removed proves nothing about the protection.
    """
    raw = _parent()
    monkeypatch.setattr(def14a_impute, "CARRY_MAX_DAYS", 20_000)
    filled, _ = impute_def14a(raw.copy())
    ages = _carried_cells(raw, filled, "insider_ownership_pct",
                          ["ticker", "accession_number"], "ticker").dropna()
    oldest = int(ages.max()) if len(ages) else 0

    print("\n=== SANITY CHECK: the guard removed ===")
    print(f"  with CARRY_MAX_DAYS=20,000 the oldest carried source is {oldest} days "
          f"({oldest / 365.25:.1f} years) -- versus a {LEVEL_MAX_AGE_DAYS}-day horizon")
    assert oldest > LEVEL_MAX_AGE_DAYS, (
        "raising the cap did not produce a cell older than the horizon; the fixture no longer "
        "contains a gap wide enough to test the rule")
    print("  SANITY CHECK: the cap is the only thing preventing a 9-year-old observation from "
          "being dated to the row it lands on and reading as fresh.")


def test_no_child_fill_is_older_than_the_cap():
    """The same invariant one grain down, on `def14a_directors`.

    ⚠ RED UNTIL B1. The agreement gate this replaces has NO age bound at all: it fills whenever
    the same person reports the same count either side of a gap, however wide, so a 9-year-old
    count is written and then dated to the row it landed on. That is the same laundering as the
    parent's, and it is why the child fill needs the SAME cap rather than a rule of its own.
    """
    dates = ["2010-05-01", "2019-05-01", "2020-05-01"]
    rows = [{"ticker": "STALE", "accession_number": f"S-{i}", "as_of": d, "name": "Rip Winkle",
             "age": 55 + (i * 5), "tenure_years": 10 + i,
             "other_public_company_boards": 2.0 if i in (0, 2) else np.nan}
            for i, d in enumerate(dates)]
    raw = pd.DataFrame(rows)
    filled, _ = fill_director_attributes(raw.copy())
    key, group = ["ticker", "accession_number", "name"], "name"
    ages = _carried_cells(raw, filled, "other_public_company_boards", key, group).dropna()
    over = sorted(int(a) for a in ages if a > CARRY_MAX_DAYS)

    print("\n=== SANITY CHECK: the child grain's fill ages ===")
    print(f"  Rip Winkle discloses 2 boards in 2010, is silent in 2019, discloses 2 again in "
          f"2020. Cells filled at ages (days): {sorted(int(a) for a in ages) or 'none'}")
    assert not over, (
        f"the child fill wrote a value sourced {over} days back, past CARRY_MAX_DAYS="
        f"{CARRY_MAX_DAYS}. An agreement gate has no age bound: it only asks whether the two "
        "sides match, never how far apart they are.")
    print(f"  SANITY CHECK: no child cell is filled from an observation older than "
          f"{CARRY_MAX_DAYS} days.")
