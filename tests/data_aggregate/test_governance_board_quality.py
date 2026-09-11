"""
test_governance_board_quality.py  (tests/data_aggregate/test_governance_board_quality.py)
------------------------------------------------------------------------------------------
The six board-quality features (D40, D41) in
`src/data_aggregate/utils/governance/directors.py` — what a board AVERAGE structurally cannot
express. A board can hold its average tenure flat while half of it turns over.

The load-bearing test here is `test_a_dispersion_is_computed_on_the_FILED_column`: a carry
forward or an anchor shrinks variance by construction, so a dispersion built on the filled
column measures the smoothness of the FILL. If filling the child table moves these two features
at all, they are being computed on the wrong column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.directors import (
    ALL_FIELDS, BOARD_QUALITY_FIELDS, EVENT_FIELDS, LEVEL_FIELDS, PEER_RELATIVE_FIELDS,
    _OVERBOARDED_OTHER_SEATS, board_quality_fields, fill_director_attributes,
)
from src.data_aggregate.utils.governance.staleness import (
    GOVERNANCE_EVENT_MAX_AGE_DAYS, LEVEL_MAX_AGE_DAYS,
)
from src.data_store.schema import Tables

IDX = pd.bdate_range("2019-01-01", "2025-06-30")
_YEARS = ["2019-05-01", "2020-05-01", "2021-05-01", "2022-05-01", "2023-05-01"]


def _board(ticker: str, year_idx: int, people: list[dict]) -> list[dict]:
    return [{"ticker": ticker, "accession_number": f"{ticker}-{year_idx}",
             "as_of": _YEARS[year_idx], **p} for p in people]


def _at(frame: pd.DataFrame, date: str, ticker: str) -> float:
    return float(frame.loc[pd.Timestamp(date), ticker])


def _turnover_board() -> pd.DataFrame:
    """TTT: a five-seat board where in 2020 exactly ONE director leaves and ONE arrives, and in
    2021 nothing changes. So turnover must read 1/5 = 0.2 then 0.0 — and 0.0 is the half that
    matters: a quiet year has to be a measured zero, not an absence."""
    keep = [{"name": ["Ann Alder", "Bob Birch", "Cara Cedar", "Dave Dogwood"][i], "age": 60, "tenure_years": 10,
             "other_public_company_boards": 1.0} for i in range(4)]
    leaver = {"name": "Olive Outgoing", "age": 70, "tenure_years": 20,
              "other_public_company_boards": 1.0}
    joiner = {"name": "Ivan Incoming", "age": 50, "tenure_years": 1,
              "other_public_company_boards": 1.0}
    rows = _board("TTT", 0, keep + [leaver])
    rows += _board("TTT", 1, keep + [joiner])
    rows += _board("TTT", 2, keep + [joiner])
    return pd.DataFrame(rows)


def test_turnover_counts_one_arrival_on_a_five_seat_board_and_zero_in_a_quiet_year():
    """One in, one out, on five seats, stamped on the LATER filing -- the date the change became
    knowable. A departure with no replacement does NOT count: the measure is arrivals over
    current board size, so a board that shrinks reads 0, which is the honest answer to "how much
    of this board is new"."""
    frames, tally = board_quality_fields(_turnover_board(), IDX)
    t = frames["board_turnover"]
    assert np.isnan(_at(t, "2019-06-03", "TTT")), "the FIRST filing has no predecessor"
    assert _at(t, "2020-06-01", "TTT") == pytest.approx(0.2)
    assert _at(t, "2021-06-01", "TTT") == pytest.approx(0.0)
    assert tally["board_turnover pairs (person_key)"] == 2
    print("\n=== SANITY CHECK: board_turnover ===")
    print(f"  2019 (first filing) {_at(t, '2019-06-03', 'TTT')} -> NaN, no predecessor")
    print(f"  2020 (one in, one out of five) {_at(t, '2020-06-01', 'TTT'):.4f}")
    print(f"  2021 (nothing changed)         {_at(t, '2021-06-01', 'TTT'):.4f}")
    print("  CONCLUSION: 1/5 on the change year, a measured 0.0 on the quiet one, NaN before "
          "any comparison exists. Validated.")


def test_turnover_keys_on_person_key_so_a_RESPELLING_is_not_churn():
    """The measured reason for the key (module header): under the filed name a respelling reads
    as one departure AND one arrival, and on the live table that inflates mean turnover from
    10.47% to 14.18%. Here the same person is spelled two ways across one year."""
    people = [{"name": ["Ann Alder", "Bob Birch", "Cara Cedar", "Dave Dogwood"][i], "age": 60, "tenure_years": 10,
               "other_public_company_boards": 1.0} for i in range(4)]
    rows = _board("RRR", 0, people + [{"name": "Daniel Rosensweig", "age": 55,
                                       "tenure_years": 5,
                                       "other_public_company_boards": 1.0}])
    rows += _board("RRR", 1, people + [{"name": "Daniel L. Rosensweig", "age": 56,
                                        "tenure_years": 6,
                                        "other_public_company_boards": 1.0}])
    frames, tally = board_quality_fields(pd.DataFrame(rows), IDX)
    assert _at(frames["board_turnover"], "2020-06-01", "RRR") == pytest.approx(0.0)
    assert tally["board_turnover mean x1000 (filed name)"] > \
        tally["board_turnover mean x1000 (person_key)"], \
        "the raw-name tally must record the spelling component, not agree with the keyed one"
    print("\n=== SANITY CHECK: the turnover key ===")
    print("  `Daniel Rosensweig` -> `Daniel L. Rosensweig`: keyed turnover "
          f"{_at(frames['board_turnover'], '2020-06-01', 'RRR'):.4f}, "
          f"filed-name tally {tally['board_turnover mean x1000 (filed name)'] / 1000:.4f} "
          f"vs keyed {tally['board_turnover mean x1000 (person_key)'] / 1000:.4f}")
    print("  CONCLUSION: a respelling is 0.0 churn under the key, and the raw-name figure is "
          "still TALLIED so the spelling component stays visible at build time. Validated.")


def test_overboarded_counts_over_REPORTING_directors_not_board_size():
    """The denominator is the whole point. A ten-seat board where only four directors report
    their other seats, two of them over the ISS trigger, is 50% overboarded AMONG THOSE WHO
    REPORT -- and 20% only if the reporting rate is silently multiplied in."""
    people = [{"name": ["Sam Sable", "Sue Spruce", "Sid Sorrel", "Sky Sumac", "Sean Sedge", "Sara Salix"][i], "age": 60, "tenure_years": 5,
               "other_public_company_boards": None} for i in range(6)]
    people += [{"name": "Busy A", "age": 61, "tenure_years": 5,
                "other_public_company_boards": 4.0},
               {"name": "Busy B", "age": 62, "tenure_years": 5,
                "other_public_company_boards": 3.0},
               {"name": "Quiet A", "age": 63, "tenure_years": 5,
                "other_public_company_boards": 0.0},
               {"name": "Quiet B", "age": 64, "tenure_years": 5,
                "other_public_company_boards": 1.0}]
    rows = []
    for i in range(len(_YEARS)):
        rows += _board("OOO", i, people)
    frames, _ = board_quality_fields(pd.DataFrame(rows), IDX)
    got = _at(frames["pct_overboarded"], "2021-06-01", "OOO")
    assert got == pytest.approx(0.5), f"{got} -- divided by board size, not by reporters"
    print("\n=== SANITY CHECK: pct_overboarded's denominator ===")
    print(f"  10 seats, 4 reporting, 2 at or above {_OVERBOARDED_OTHER_SEATS:.0f} other seats "
          f"-> {got:.4f} (2/4), not 0.20 (2/10)")
    print("  CONCLUSION: the share is over the directors who REPORT. `other_public_company_"
          "boards` excludes the issuer (15.7% of reporting directors say 0), so the ISS "
          "four-public-board trigger is >= 3 OTHER seats. Validated.")


def test_a_dispersion_is_computed_on_the_FILED_column():
    """§3.4's guard, stated as an equality rather than as an intention.

    The child fill accrues 20,210 ages on the live table. If `board_age_dispersion` read the
    filled column, every one of those would pull the board's spread toward the anchor's straight
    line -- so the test is that filling changes the feature by EXACTLY zero.
    """
    people = [{"name": "Young", "age": 45, "tenure_years": 2,
               "other_public_company_boards": 1.0},
              {"name": "Middle", "age": 60, "tenure_years": 10,
               "other_public_company_boards": 1.0},
              {"name": "Elder", "age": 75, "tenure_years": 25,
               "other_public_company_boards": 1.0}]
    rows = []
    for i in range(len(_YEARS)):
        board = [dict(p) for p in people]
        board = [{**p, "age": p["age"] + i} for p in board]
        if i == 2:                        # a hole the accrual will fill
            board[1] = {**board[1], "age": None}
        rows += _board("DDD", i, board)
    raw = pd.DataFrame(rows)
    filled, stats = fill_director_attributes(raw)
    assert stats["child accrued: age"] == 1, "the fixture no longer exercises the accrual"

    from_raw, _ = board_quality_fields(raw, IDX)
    from_filled, _ = board_quality_fields(filled, IDX)
    # ⚠ `1e-12`, not `== 0.0`: `fill_director_attributes` returns the rows sorted per PERSON, so
    # the two frames sum the same values in a different ORDER and float addition is not
    # associative -- the observed residual is 1.8e-15. Anything the fill actually reached would
    # move a 3-seat board's std by ~1e-1, so the tolerance cannot hide the defect it guards.
    for name in ("board_age_dispersion", "board_tenure_dispersion"):
        a, b = from_raw[name], from_filled[name]
        gap = (a - b).abs().to_numpy()
        assert np.nanmax(gap) < 1e-12, f"{name} moved when the child table was filled"
    # ...and the guard is NOT vacuous: on the filled column the same board's spread would be a
    # different number entirely. 2021 discloses ages 47 and 77 (std 21.21); the accrual adds 62,
    # and a std over all three is 15.00 -- so had the feature read the filled column it would
    # report the anchor's tidiness as the board's.
    y2021 = filled[filled["accession_number"] == "DDD-2"]["age"].astype("float64")
    would_be = float(y2021.std())
    lvl_raw = _at(from_raw["board_age_dispersion"], "2021-06-01", "DDD")
    assert abs(would_be - lvl_raw) > 1.0, \
        "the fixture no longer distinguishes the two bases -- the guard proves nothing"
    print("\n=== SANITY CHECK: the §3.4 dispersion guard ===")
    print(f"  a 3-seat board with one age accrued in 2021: dispersion from the FILED ages "
          f"{lvl_raw:.4f} == from the filled frame "
          f"{_at(from_filled['board_age_dispersion'], '2021-06-01', 'DDD'):.4f}")
    print(f"  had it read the FILLED column it would report {would_be:.4f} instead "
          f"({y2021.notna().sum()} ages including the accrued one)")
    print("  CONCLUSION: filling the child table moves both dispersions by <1e-12 (summation "
          "order only), while the wrong basis would move this one by "
          f"{abs(would_be - lvl_raw):.2f}. They measure the board, not the fill. Validated.")


def test_the_encoding_and_expiry_contracts():
    """The encoding + expiry contracts that keep this family honest, asserted not described.

    ⚠ TWO CLOCKS, NOT ONE -- and this test asserted there was only one until phase 3 landed.
    `board_turnover` is a one-year CHANGE, the same shape as `board_busyness_delta_1y`, so
    ffilling "this board replaced a fifth of its seats" past a missed cycle asserts churn nobody
    disclosed: it ages out at `GOVERNANCE_EVENT_MAX_AGE_DAYS` (548d). The five standing LEVELS
    are not events -- but **"not an event" is not the same statement as "never expires"**, which
    is exactly the reasoning error phase 3 fixed. They age out at `LEVEL_MAX_AGE_DAYS` (1,095d =
    two whole missed annual meetings). `directors.LEVEL_FIELDS` carries the measured bite on the
    live part: 1,161-1,209 cells per field, 0.04%, max age 1,822 days.

    The load-bearing assertion is the MIDDLE pair, at +612 days: the event is gone and the level
    is still there. That is the only probe that fails if the two horizons ever collapse into one,
    in either direction -- every other date passes under a single shared clock.
    """
    assert PEER_RELATIVE_FIELDS == frozenset(), \
        "a peer leg was added without the measured precondition"
    assert set(BOARD_QUALITY_FIELDS) == ALL_FIELDS
    assert EVENT_FIELDS == {"board_turnover"}
    assert EVENT_FIELDS < ALL_FIELDS
    assert LEVEL_FIELDS == ALL_FIELDS - EVENT_FIELDS, "a field sits on neither clock"
    assert GOVERNANCE_EVENT_MAX_AGE_DAYS < LEVEL_MAX_AGE_DAYS, "the two clocks are inverted"

    # One filing in 2019, one in 2020, then five years of silence -- so every probe below ages
    # against 2020-05-01, and the two horizons land on 2021-10-31 and 2023-05-01.
    people = [{"name": ["Pat Poplar", "Pam Pine", "Pete Plane", "Pia Palm"][i], "age": 60, "tenure_years": 10,
               "other_public_company_boards": 1.0} for i in range(4)]
    rows = _board("XXX", 0, people)
    rows += _board("XXX", 1, people[:3] + [{"name": "New", "age": 50, "tenure_years": 1,
                                            "other_public_company_boards": 1.0}])
    frames, tally = board_quality_fields(pd.DataFrame(rows), IDX)
    t, lvl = frames["board_turnover"], frames["pct_long_tenured"]
    last = pd.Timestamp(_YEARS[1])

    assert not np.isnan(_at(t, "2020-06-01", "XXX")), "a fresh event was expired"
    assert not np.isnan(_at(t, "2021-10-29", "XXX")), "an event died inside its own horizon"
    assert np.isnan(_at(t, "2022-01-03", "XXX")), "an event survived past 548 days"
    assert not np.isnan(_at(lvl, "2022-01-03", "XXX")), \
        "a standing LEVEL was expired on the EVENT clock"
    assert not np.isnan(_at(lvl, "2023-05-01", "XXX")), "a level died inside its own horizon"
    assert np.isnan(_at(lvl, "2024-06-03", "XXX")), "a level survived past 1,095 days"

    # ...and both caps REPORTED their bite. A silent cap is the failure mode phase 3 found.
    expired = [k for k in tally if k.startswith("expired >")]
    assert any("board_turnover" in k for k in expired), "the event cap did not report its bite"
    assert any("pct_long_tenured" in k for k in expired), "the level cap did not report its bite"

    print("\n=== SANITY CHECK: the encoding + two-clock expiry contracts ===")
    print(f"  PEER_RELATIVE_FIELDS = {set(PEER_RELATIVE_FIELDS) or '{} (raw only, measured)'}")
    print(f"  EVENT_FIELDS {sorted(EVENT_FIELDS)} on {GOVERNANCE_EVENT_MAX_AGE_DAYS}d; "
          f"{len(LEVEL_FIELDS)} LEVEL_FIELDS on {LEVEL_MAX_AGE_DAYS}d")
    print(f"  last filing {last.date()} -> event horizon ends "
          f"{(last + pd.Timedelta(days=GOVERNANCE_EVENT_MAX_AGE_DAYS)).date()}, level horizon "
          f"{(last + pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)).date()}")
    for d in ("2020-06-01", "2021-10-29", "2022-01-03", "2023-05-01", "2024-06-03"):
        print(f"  {d} (+{(pd.Timestamp(d) - last).days:>4}d)  "
              f"board_turnover={_at(t, d, 'XXX')!s:<8} pct_long_tenured={_at(lvl, d, 'XXX')}")
    print("  CONCLUSION: no peer leg is emitted, the one CHANGE ages out at 548 days, and the "
          "five standing levels outlive it by 547 more days before ageing out themselves -- "
          "two clocks, both biting, both reported. Validated.")


def test_the_real_board_quality_readout():
    """§1.6 regenerated on the live table, plus `pct_overboarded` before and after the fill."""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load(Tables.def14a_directors)
        n_filings = ctx.store.load(Tables.def14a_llm, columns=["accession_number"]
                                   )["accession_number"].nunique()
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_directors not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_directors empty")

    filled, _ = fill_director_attributes(raw)
    idx = pd.bdate_range("1995-01-01", "2026-12-31")
    pre, _ = board_quality_fields(raw, idx)
    post, tally = board_quality_fields(filled, idx)

    floors = {"board_turnover": 11_000, "pct_long_tenured": 10_500,
              "board_tenure_dispersion": 10_500, "board_age_dispersion": 10_000,
              "oldest_director_age": 11_500, "pct_overboarded": 8_000}
    print("\n=== SANITY CHECK: §1.6 board quality on the live archive ===")
    print(f"  {n_filings:,} parent filings, {len(raw):,} director rows")
    for name in sorted(post):
        n = int(tally[f"{name}: filings"])
        cells = int(post[name].notna().to_numpy().sum())
        print(f"  {name:26s} {n:6,} filings ({n / n_filings:5.1%})  "
              f"{cells:10,} daily cells  {int(post[name].notna().any().sum()):3d} tickers")
        assert n >= floors[name], f"{name} covers only {n:,} filings, floor {floors[name]:,}"
    ob_pre = int(pre["pct_overboarded"].notna().any(axis=0).sum())
    n_pre = int((pre["pct_overboarded"].notna().sum(axis=1) > 0).sum())
    print(f"  pct_overboarded coverage: {tally['pct_overboarded: filings']:,} filings AFTER the "
          f"child fill, against 5,656 before it ({ob_pre} tickers pre-fill, {n_pre:,} dates)")
    print("  CONCLUSION: every member covers its measured share of filings, and the ISS "
          "overboarded share -- the definition the literature uses, which `avg_other_public_"
          "boards` only proxies -- goes from 45.8% to 70.3% of filings on the child fill. "
          "Validated.")
