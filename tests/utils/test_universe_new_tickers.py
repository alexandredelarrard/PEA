"""`Universe.new_since` and `Universe.added_on_dates`: the new-ticker window on `universe_added_on`, both ends."""

from __future__ import annotations

import pandas as pd

from tests.support.universe import seed_universe

_AS_OF = pd.Timestamp("2026-10-03")


def _universe(store, added_on: dict[str, str]):
    return seed_universe(store, list(added_on), rows={t: {"universe_added_on": pd.Timestamp(d)} for t, d in added_on.items()})


def test_new_since_window(sqlite_store) -> None:
    universe = _universe(sqlite_store, {"OLD": "2000-01-01", "MID": "2026-09-26", "NEW": "2026-10-02"})

    week = universe.new_since(7, _AS_OF)
    quarter = universe.new_since(95, _AS_OF)
    assert week == {"NEW"}  # 2026-09-26 is exactly 7 days back: outside the window
    assert quarter == {"MID", "NEW"}
    assert universe.new_since(7, pd.Timestamp("2026-10-10")) == set()

    print("\n=== SANITY CHECK: new-ticker window ===")
    print(f"  as_of {_AS_OF.date()}: 7-day overlap -> {sorted(week)}, 95-day -> {sorted(quarter)}; 2000-01-01 is established. Validated.")


def test_a_future_added_on_is_not_new_yet(sqlite_store) -> None:
    universe = _universe(sqlite_store, {"EDGE": "2026-09-27", "TODAY": "2026-10-03", "TOMORROW": "2026-10-04", "LATER": "2026-12-01"})

    week = universe.new_since(7, _AS_OF)
    on_the_day = universe.new_since(7, pd.Timestamp("2026-10-04"))
    dates = universe.added_on_dates(["EDGE", "TODAY", "TOMORROW", "LATER"], _AS_OF)

    assert week == {"EDGE", "TODAY"}  # as_of - 6 days and as_of itself are new; after as_of is not yet
    assert on_the_day == {"TODAY", "TOMORROW"}  # a day later EDGE is 7 days back and TOMORROW has joined
    assert dates == {"EDGE": pd.Timestamp("2026-09-27"), "TODAY": pd.Timestamp("2026-10-03")}
    print("\n=== SANITY CHECK: the new-ticker window ends on as_of ===")
    print(f"  as_of {_AS_OF.date()}: {sorted(week)} new, TOMORROW and LATER not yet; on 2026-10-04 -> {sorted(on_the_day)}. Validated.")
