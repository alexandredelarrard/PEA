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

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.directors import (
    _OVERBOARDED_OTHER_SEATS,
    ALL_FIELDS,
    BOARD_QUALITY_FIELDS,
    EVENT_FIELDS,
    LEVEL_FIELDS,
    PEER_RELATIVE_FIELDS,
    board_quality_fields,
    fill_director_attributes,
)
from src.data_aggregate.utils.governance.staleness import (
    GOVERNANCE_EVENT_MAX_AGE_DAYS,
    LEVEL_MAX_AGE_DAYS,
)
from src.data_store.schema import Tables

IDX = pd.bdate_range("2019-01-01", "2025-06-30")
_YEARS = ["2019-05-01", "2020-05-01", "2021-05-01", "2022-05-01", "2023-05-01"]


def _board(ticker: str, year_idx: int, people: list[dict]) -> list[dict]:
    return [{"ticker": ticker, "accession_number": f"{ticker}-{year_idx}", "as_of": _YEARS[year_idx], **p} for p in people]


def _at(frame: pd.DataFrame, date: str, ticker: str) -> float:
    return float(cast(float, frame.loc[pd.Timestamp(date), ticker]))


def _source_supported_board_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """Use numeric disclosures at preceding positions in stable source-date order."""
    col = "other_public_company_boards"
    work = raw[["ticker", "accession_number", "as_of", "name", col]].copy()
    work["as_of"] = pd.to_datetime(work["as_of"], errors="coerce")
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["ticker", "as_of", "name"]).sort_values("as_of", kind="stable")
    # Filed names are the attribute identity: an initial-only person key can merge two people.
    # Only actual numeric disclosures update this state. Same-day ties retain raw source order,
    # so a later sibling accession cannot supply an earlier gap or reset a carried source clock.
    latest: dict[tuple[str, str], tuple[pd.Timestamp, float]] = {}
    source_dates = []
    expected_values = []
    for ticker, name, date, value in work[["ticker", "name", "as_of", col]].itertuples(index=False, name=None):
        key = (ticker, name)
        if pd.notna(value):
            latest[key] = (date, float(value))
        source = latest.get(key)
        source_dates.append(pd.NaT if source is None else source[0])
        expected_values.append(np.nan if source is None or (date - source[0]).days > 395 else source[1])
    work["source_as_of"] = source_dates
    work["expected_seats"] = expected_values
    return work


def _turnover_board() -> pd.DataFrame:
    """TTT: a five-seat board where in 2020 exactly ONE director leaves and ONE arrives, and in
    2021 nothing changes. So turnover must read 1/5 = 0.2 then 0.0 — and 0.0 is the half that
    matters: a quiet year has to be a measured zero, not an absence."""
    keep = [
        {"name": ["Ann Alder", "Bob Birch", "Cara Cedar", "Dave Dogwood"][i], "age": 60, "tenure_years": 10, "other_public_company_boards": 1.0}
        for i in range(4)
    ]
    leaver = {"name": "Olive Outgoing", "age": 70, "tenure_years": 20, "other_public_company_boards": 1.0}
    joiner = {"name": "Ivan Incoming", "age": 50, "tenure_years": 1, "other_public_company_boards": 1.0}
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
    print("  CONCLUSION: 1/5 on the change year, a measured 0.0 on the quiet one, NaN before any comparison exists. Validated.")


def test_turnover_keys_on_person_key_so_a_respelling_is_not_churn():
    """The measured reason for the key (module header): under the filed name a respelling reads
    as one departure AND one arrival, and on the live table that inflates mean turnover from
    10.47% to 14.18%. Here the same person is spelled two ways across one year."""
    people = [
        {"name": ["Ann Alder", "Bob Birch", "Cara Cedar", "Dave Dogwood"][i], "age": 60, "tenure_years": 10, "other_public_company_boards": 1.0}
        for i in range(4)
    ]
    rows = _board("RRR", 0, people + [{"name": "Daniel Rosensweig", "age": 55, "tenure_years": 5, "other_public_company_boards": 1.0}])
    rows += _board("RRR", 1, people + [{"name": "Daniel L. Rosensweig", "age": 56, "tenure_years": 6, "other_public_company_boards": 1.0}])
    frames, tally = board_quality_fields(pd.DataFrame(rows), IDX)
    assert _at(frames["board_turnover"], "2020-06-01", "RRR") == pytest.approx(0.0)
    assert tally["board_turnover mean x1000 (filed name)"] > tally["board_turnover mean x1000 (person_key)"], (
        "the raw-name tally must record the spelling component, not agree with the keyed one"
    )
    print("\n=== SANITY CHECK: the turnover key ===")
    print(
        "  `Daniel Rosensweig` -> `Daniel L. Rosensweig`: keyed turnover "
        f"{_at(frames['board_turnover'], '2020-06-01', 'RRR'):.4f}, "
        f"filed-name tally {tally['board_turnover mean x1000 (filed name)'] / 1000:.4f} "
        f"vs keyed {tally['board_turnover mean x1000 (person_key)'] / 1000:.4f}"
    )
    print(
        "  CONCLUSION: a respelling is 0.0 churn under the key, and the raw-name figure is "
        "still TALLIED so the spelling component stays visible at build time. Validated."
    )


def test_overboarded_counts_over_reporting_directors_not_board_size():
    """The denominator is the whole point. A ten-seat board where only four directors report
    their other seats, two of them over the ISS trigger, is 50% overboarded AMONG THOSE WHO
    REPORT -- and 20% only if the reporting rate is silently multiplied in."""
    people = [
        {
            "name": ["Sam Sable", "Sue Spruce", "Sid Sorrel", "Sky Sumac", "Sean Sedge", "Sara Salix"][i],
            "age": 60,
            "tenure_years": 5,
            "other_public_company_boards": None,
        }
        for i in range(6)
    ]
    people += [
        {"name": "Busy A", "age": 61, "tenure_years": 5, "other_public_company_boards": 4.0},
        {"name": "Busy B", "age": 62, "tenure_years": 5, "other_public_company_boards": 3.0},
        {"name": "Quiet A", "age": 63, "tenure_years": 5, "other_public_company_boards": 0.0},
        {"name": "Quiet B", "age": 64, "tenure_years": 5, "other_public_company_boards": 1.0},
    ]
    rows = []
    for i in range(len(_YEARS)):
        rows += _board("OOO", i, people)
    frames, _ = board_quality_fields(pd.DataFrame(rows), IDX)
    got = _at(frames["pct_overboarded"], "2021-06-01", "OOO")
    assert got == pytest.approx(0.5), f"{got} -- divided by board size, not by reporters"
    print("\n=== SANITY CHECK: pct_overboarded's denominator ===")
    print(f"  10 seats, 4 reporting, 2 at or above {_OVERBOARDED_OTHER_SEATS:.0f} other seats -> {got:.4f} (2/4), not 0.20 (2/10)")
    print(
        "  CONCLUSION: the share is over the directors who REPORT. `other_public_company_"
        "boards` excludes the issuer (15.7% of reporting directors say 0), so the ISS "
        "four-public-board trigger is >= 3 OTHER seats. Validated."
    )


def test_a_dispersion_is_computed_on_the_filed_column():
    """§3.4's guard, stated as an equality rather than as an intention.

    The child fill accrues 20,210 ages on the live table. If `board_age_dispersion` read the
    filled column, every one of those would pull the board's spread toward the anchor's straight
    line -- so the test is that filling changes the feature by EXACTLY zero.
    """
    people = [
        {"name": "Young", "age": 45, "tenure_years": 2, "other_public_company_boards": 1.0},
        {"name": "Middle", "age": 60, "tenure_years": 10, "other_public_company_boards": 1.0},
        {"name": "Elder", "age": 75, "tenure_years": 25, "other_public_company_boards": 1.0},
    ]
    rows = []
    for i in range(len(_YEARS)):
        board = [dict(p) for p in people]
        board = [{**p, "age": p["age"] + i} for p in board]
        if i == 2:  # a hole the accrual will fill
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
    assert abs(would_be - lvl_raw) > 1.0, "the fixture no longer distinguishes the two bases -- the guard proves nothing"
    print("\n=== SANITY CHECK: the §3.4 dispersion guard ===")
    print(
        f"  a 3-seat board with one age accrued in 2021: dispersion from the FILED ages "
        f"{lvl_raw:.4f} == from the filled frame "
        f"{_at(from_filled['board_age_dispersion'], '2021-06-01', 'DDD'):.4f}"
    )
    print(f"  had it read the FILLED column it would report {would_be:.4f} instead ({y2021.notna().sum()} ages including the accrued one)")
    print(
        "  CONCLUSION: filling the child table moves both dispersions by <1e-12 (summation "
        "order only), while the wrong basis would move this one by "
        f"{abs(would_be - lvl_raw):.2f}. They measure the board, not the fill. Validated."
    )


def test_the_encoding_and_expiry_contracts() -> None:
    """All annual board measurements expire at their own last source plus 395 days."""
    assert PEER_RELATIVE_FIELDS == frozenset()
    assert set(BOARD_QUALITY_FIELDS) == ALL_FIELDS
    assert EVENT_FIELDS == {"board_turnover"}
    assert LEVEL_FIELDS == ALL_FIELDS - EVENT_FIELDS
    assert GOVERNANCE_EVENT_MAX_AGE_DAYS == LEVEL_MAX_AGE_DAYS == 395
    people = [
        {"name": name, "age": 60, "tenure_years": 10, "other_public_company_boards": 1.0}
        for name in ("Pat Poplar", "Pam Pine", "Pete Plane", "Pia Palm")
    ]
    rows = _board("XXX", 0, people)
    rows += _board("XXX", 1, people[:3] + [{"name": "New", "age": 50, "tenure_years": 1, "other_public_company_boards": 1.0}])
    frames, tally = board_quality_fields(pd.DataFrame(rows), IDX)
    last = pd.Timestamp(_YEARS[1])
    for name, frame in frames.items():
        age = (frame.index - last).days
        assert frame.loc[(age >= 0) & (age <= 395), "XXX"].notna().all(), name
        assert frame.loc[age > 395, "XXX"].isna().all(), name
        assert any(k.startswith("expired >395d:") and name in k for k in tally), name
    print("SANITY: all six board-quality measurements share the 395-day limit and report their expiry counts.")


def test_overboarded_support_keeps_zero_through_day_395_without_restarting_the_source_clock() -> None:
    """Known-truth boundary: prior numeric zero is evidence; a future count is not."""
    start = pd.Timestamp("2020-01-01")
    dates = [start + pd.Timedelta(days=offset) for offset in (-1, 0, 200, 395, 396, 500, 895, 896)]
    raw = pd.DataFrame(
        [
            {"ticker": "AAA", "accession_number": f"a{i}", "as_of": date, "name": "Ann Alder", "other_public_company_boards": value}
            for i, (date, value) in enumerate(zip(dates, [None, 0.0, None, None, None, 3.0, None, None], strict=True))
        ]
        + [{"ticker": "BBB", "accession_number": "b0", "as_of": start, "name": "Ann Alder", "other_public_company_boards": None}]
    )
    expected_values = pd.Series([np.nan, 0.0, 0.0, 0.0, np.nan, 3.0, 3.0, np.nan], index=[f"a{i}" for i in range(8)], name="seats")
    oracle = _source_supported_board_counts(raw).set_index("accession_number")
    filled, _ = fill_director_attributes(raw)
    actual = filled.set_index("accession_number")
    for frame, col in ((oracle, "expected_seats"), (actual, "other_public_company_boards")):
        pd.testing.assert_series_equal(frame.loc[expected_values.index, col].rename("seats"), expected_values, check_index_type=False)
        assert pd.isna(frame.loc["b0", col]), "another ticker cannot supply this person's missing disclosure"
    for accession in ("a1", "a2", "a3"):
        assert oracle.loc[accession, "source_as_of"] == start
        assert actual.loc[accession, "other_public_company_boards_source_as_of"] == start
    assert actual.loc["a6", "other_public_company_boards_source_as_of"] == dates[5]
    idx = pd.DatetimeIndex(dates)
    frames, tally = board_quality_fields(filled, idx)
    assert frames["pct_overboarded"].loc[dates[3], "AAA"] == pytest.approx(0.0)
    assert pd.isna(frames["pct_overboarded"].loc[dates[4], "AAA"])
    assert frames["pct_overboarded"].loc[dates[6], "AAA"] == pytest.approx(1.0)
    assert pd.isna(frames["pct_overboarded"].loc[dates[7], "AAA"])
    assert tally["pct_overboarded: filings"] == 5
    print(
        "SANITY: numeric zero and a later genuine disclosure each survive day 395, expire on day 396, and never borrow future or other-ticker evidence."
    )


@pytest.mark.parametrize("numeric_first", [False, True])
def test_overboarded_same_day_accessions_use_only_preceding_source_records(numeric_first: bool) -> None:
    """Source order, not the presence of a same-day sibling, determines prior evidence."""
    date = pd.Timestamp("2020-01-01")
    rows = [
        {"ticker": "AAA", "accession_number": "gap", "as_of": date, "name": "Ann Alder", "other_public_company_boards": None},
        {"ticker": "AAA", "accession_number": "numeric", "as_of": date, "name": "Ann Alder", "other_public_company_boards": 4.0},
    ]
    if numeric_first:
        rows.reverse()
    rows.append(
        {"ticker": "AAA", "accession_number": "later", "as_of": date + pd.Timedelta(days=1), "name": "Ann Alder", "other_public_company_boards": None}
    )
    raw = pd.DataFrame(rows)
    expected = pd.Series([4.0 if numeric_first else np.nan, 4.0, 4.0], index=["gap", "numeric", "later"], name="seats")
    oracle = _source_supported_board_counts(raw).set_index("accession_number")
    actual, _ = fill_director_attributes(raw)
    actual = actual.set_index("accession_number")
    for frame, col in ((oracle, "expected_seats"), (actual, "other_public_company_boards")):
        pd.testing.assert_series_equal(frame.loc[expected.index, col].rename("seats"), expected, check_index_type=False)
    print(f"SANITY: numeric-first={numeric_first}; same-day gaps use only preceding raw numeric records and the next day carries the genuine source.")


def test_the_real_board_quality_readout() -> None:
    """Live coverage must match independently reconstructed 395-day numeric-source support."""
    try:
        from src.context import get_config_context

        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        source_start, source_end = ctx.store.bounds(Tables.def14a_directors)
        parent_start, parent_end = ctx.store.bounds(Tables.def14a_llm)
        raw = ctx.store.load(
            Tables.def14a_directors,
            columns=["ticker", "accession_number", "as_of", "name", "age", "tenure_years", "other_public_company_boards"],
            since=source_start,
            until=source_end,
        )
        filings = ctx.store.load(Tables.def14a_llm, columns=["accession_number"], since=parent_start, until=parent_end)
        assert filings is not None
        n_filings = filings["accession_number"].nunique()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"def14a_directors not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_directors empty")

    support = _source_supported_board_counts(raw)
    keys = ["ticker", "accession_number", "as_of"]
    observed = support["other_public_company_boards"].notna()
    supported = support["expected_seats"].notna()
    n_observed = len(support.loc[observed, keys].drop_duplicates())
    expected_filings = support.loc[supported, keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    filled, _ = fill_director_attributes(raw)
    actual_filings = filled.loc[filled["other_public_company_boards"].notna(), keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_filings, expected_filings)
    idx = pd.bdate_range("1995-01-01", "2026-12-31")
    pre, pre_tally = board_quality_fields(raw, idx)
    post, tally = board_quality_fields(filled, idx)

    floors = {
        "board_turnover": 11_000,
        "pct_long_tenured": 10_500,
        "board_tenure_dispersion": 10_500,
        "board_age_dispersion": 10_000,
        "oldest_director_age": 11_500,
    }
    print("\n=== SANITY CHECK: §1.6 board quality on the live archive ===")
    print(f"  {n_filings:,} parent filings, {len(raw):,} director rows")
    for name in sorted(post):
        n = int(tally[f"{name}: filings"])
        cells = int(post[name].notna().to_numpy().sum())
        print(f"  {name:26s} {n:6,} filings ({n / n_filings:5.1%})  {cells:10,} daily cells  {int(post[name].notna().any().sum()):3d} tickers")
        if name == "pct_overboarded":
            assert n == len(expected_filings), f"{name} must cover exactly the raw source-supported filings"
        else:
            assert n >= floors[name], f"{name} covers only {n:,} filings, floor {floors[name]:,}"
    assert pre_tally["pct_overboarded: filings"] == n_observed
    assert len(expected_filings) > n_observed, "the archive must exercise bounded carry beyond raw disclosures"
    stale = ~observed & support["source_as_of"].notna() & ~supported
    no_prior = ~observed & support["source_as_of"].isna()
    assert stale.any() and no_prior.any(), "the archive must exercise stale and no-prior refusals"
    ob_pre = int(pre["pct_overboarded"].notna().any(axis=0).sum())
    n_pre = int((pre["pct_overboarded"].notna().sum(axis=1) > 0).sum())
    print(
        f"  pct_overboarded coverage: {tally['pct_overboarded: filings']:,} filings AFTER the "
        f"child fill, against {n_observed:,} before it ({ob_pre} tickers pre-fill, {n_pre:,} dates)"
    )
    print(
        f"  numeric-source support: {int(observed.sum()):,} raw rows, {int((supported & ~observed).sum()):,} carried rows, "
        f"{int(stale.sum()):,} stale refusals, {int(no_prior.sum()):,} no-prior refusals"
    )
    print(
        f"  CONCLUSION: pct_overboarded covers {n_observed / n_filings:.1%} raw and {len(expected_filings) / n_filings:.1%} "
        "source-supported filings, exactly matching preceding numeric disclosures within 395 days in source order; "
        "stale and absent source coverage stays visible. Other board-quality coverage floors remain satisfied."
    )
