"""Annual governance expiry, source provenance and bounded real-data controls."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from pandas._typing import Scalar

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
from src.data_aggregate.utils.governance.directors import (
    board_aggregates,
    fill_director_attributes,
    merge_board_aggregates,
)
from src.data_aggregate.utils.governance.staleness import (
    GOVERNANCE_EVENT_MAX_AGE_DAYS,
    LEGACY_EXEMPT_FROM_EXPIRY,
    LEVEL_HORIZON_FIELDS,
    LEVEL_MAX_AGE_DAYS,
    expire_event_fields,
    expire_level_fields,
    expire_stale,
    horizon_for,
)


def _daily(history: pd.DataFrame, field: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return fundamentals_to_daily(history, field, idx)


def _asof_value(frame: pd.DataFrame, when: pd.Timestamp, ticker: str) -> Scalar:
    """Select the latest fixture value at or before ``when`` as a scalar."""
    date = cast(pd.Timestamp, pd.DatetimeIndex(frame.index).asof(when))
    value = frame.loc[date, ticker]
    assert not isinstance(value, pd.Series)
    return value


def test_expiry_boundary_on_both_tiers() -> None:
    """Annual events and levels share the 395-day source-age limit."""
    filed = pd.Timestamp("2020-06-01")
    idx = pd.date_range(filed, periods=397)
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": filed, "sop_dissent": 0.11, "ceo_pay_growth": 0.11}])
    assert GOVERNANCE_EVENT_MAX_AGE_DAYS == LEVEL_MAX_AGE_DAYS == 395
    assert len(LEVEL_HORIZON_FIELDS) == 11
    assert LEGACY_EXEMPT_FROM_EXPIRY is LEVEL_HORIZON_FIELDS
    for field in ("sop_dissent", "ceo_pay_growth"):
        capped = expire_stale(_daily(hist, field, idx), hist, field)
        assert horizon_for(field) == 395
        assert capped.iloc[395, 0] == pytest.approx(0.11)
        assert pd.isna(capped.iloc[396, 0])
    print("SANITY: annual vote and pay observations survive day 395 and expire on day 396.")


def test_expiry_uses_the_filing_that_produced_the_cell():
    """Two filings: the clock restarts on the later one — and a NULL filing does not restart it."""
    idx = pd.bdate_range("2018-01-01", "2024-12-31")
    hist = pd.DataFrame(
        [
            {"ticker": "AAA", "as_of": "2019-05-01", "sop_dissent": 0.05},
            {"ticker": "AAA", "as_of": "2020-05-01", "sop_dissent": 0.09},
            # a LATER filing that disclosed nothing for this field: it must not reset the clock
            {"ticker": "AAA", "as_of": "2021-05-01", "sop_dissent": None},
            # a different ticker whose only filing is ancient -> fully expired
            {"ticker": "BBB", "as_of": "2018-03-01", "sop_dissent": 0.42},
        ]
    )
    daily = _daily(hist, "sop_dissent", idx)
    capped = expire_stale(daily, hist, "sop_dissent")

    def at(d: str, t: str):
        return _asof_value(capped, pd.Timestamp(d), t)

    # 2020-06-01 is 396d after the 2019 filing but only 31d after the 2020 one -> alive at 0.09
    assert at("2020-06-01", "AAA") == pytest.approx(0.09)
    # Disclosure silence does not restart the source clock: the old observation expires.
    assert at("2021-05-31", "AAA") == pytest.approx(0.09)
    assert pd.isna(at("2021-06-01", "AAA"))
    assert pd.isna(at("2022-01-01", "AAA"))
    # BBB's single 2018 filing is dead everywhere after its horizon
    assert at("2018-06-01", "BBB") == pytest.approx(0.42)
    assert pd.isna(at("2021-01-01", "BBB"))

    frames, stats = expire_event_fields({"sop_dissent": daily}, hist, {"sop_dissent"})
    expired, before = stats["sop_dissent"]

    print("\n=== SANITY CHECK: governance event expiry (provenance) ===")
    print("  AAA filed 2019-05-01 (0.05), 2020-05-01 (0.09), 2021-05-01 (NULL).")
    print("   2020-06-01 -> 0.09; 2021-05-31 -> 0.09 (395d); 2021-06-01 -> NaN (396d).")
    print("  BBB's lone 2018 filing is alive at +3mo and dead by 2021.")
    print(f"  expire_event_fields bite: {expired} of {before} non-null cells ({expired / before:.1%})")
    print("  CONCLUSION: the age is measured against the filing that PRODUCED the cell,")
    print("  which is the only reason a forward-filled frame can be expired at all.")


def test_chained_carries_do_not_restart_the_ultimate_source_clock():
    t0 = pd.Timestamp("2010-01-01")
    t1 = t0 + pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)
    t2 = t1 + pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)
    t3 = t2 + pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)
    child = pd.DataFrame(
        [
            {"ticker": "AAA", "accession_number": "a0", "as_of": t0, "name": "Ann Alder", "other_public_company_boards": 2.0},
            {"ticker": "AAA", "accession_number": "a1", "as_of": t1, "name": "Ann Alder", "other_public_company_boards": None},
            {"ticker": "BBB", "accession_number": "b0", "as_of": t0, "name": "Bob Birch", "other_public_company_boards": 1.0},
            {"ticker": "BBB", "accession_number": "b1", "as_of": t1, "name": "Bob Birch", "other_public_company_boards": 3.0},
        ]
    )
    parent = pd.DataFrame(
        [
            {"ticker": ticker, "accession_number": f"{ticker[0].lower()}{i}", "as_of": date, "avg_other_public_boards": None}
            for ticker in ("AAA", "BBB")
            for i, date in enumerate((t0, t1, t2))
        ]
    )

    filled, _ = fill_director_attributes(child)
    merged, _ = merge_board_aggregates(parent, board_aggregates(filled))
    history, _ = impute_def14a(merged)
    idx = pd.date_range(t0, t3, freq="D")
    daily = fundamentals_to_daily(history, "avg_other_public_boards", idx)
    capped = expire_stale(
        daily,
        history,
        "avg_other_public_boards",
        max_age_days=LEVEL_MAX_AGE_DAYS,
    )

    assert capped.loc[t1, "AAA"] == pytest.approx(2.0)
    assert pd.isna(capped.loc[t1 + pd.Timedelta(days=1), "AAA"])
    assert pd.isna(capped.loc[t2, "AAA"])
    assert capped.loc[t2, "BBB"] == pytest.approx(3.0)
    assert pd.isna(capped.loc[t2 + pd.Timedelta(days=1), "BBB"])
    print("\n=== SANITY CHECK: ultimate-source freshness across chained carry ===")
    print(f"  AAA's t0 value survives exactly through t0+{LEVEL_MAX_AGE_DAYS}, then expires despite child,")
    print("  aggregate and parent carry. BBB's genuine t1 filing reopens the same window.")
    print("  CONCLUSION: transformations preserve provenance; only observations restart freshness.")


def test_expiry_bite_on_real_def14a():
    """How hard does 548 days bite each proxy field, with the tier forced to the event one?"""
    try:
        from src.context import get_config_context

        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    from src.data_aggregate.utils.governance.def14a_impute import impute_def14a

    imp, _ = impute_def14a(raw)
    idx = pd.bdate_range("2011-01-03", "2026-09-04")  # the modern era only

    fields = [
        "say_on_pay_support_pct",
        "ceo_total_comp",
        "avg_other_public_boards",
        "board_size",
        "pct_independent_directors",
        "poison_pill",
        "majority_voting",
    ]
    rows = []
    for f in fields:
        if f not in imp.columns:
            continue
        daily = fundamentals_to_daily(imp, f, idx)
        if daily.empty:
            continue
        before = int(daily.notna().to_numpy().sum())
        # ⚠ `feature=` is passed a name that is deliberately NOT in the exemption, so this
        # measures the HORIZON, not the exemption. Two of these fields (`board_size`,
        # `pct_independent_directors`) share their name with a legacy FEATURE, so calling
        # `expire_stale(daily, imp, f)` bare would return them untouched and this table would
        # report 0.0% — reading as "a board size never goes stale" when it means "exempt".
        capped = expire_stale(daily, imp, f, feature=f"_measure_{f}")
        after = int(capped.notna().to_numpy().sum())
        exempt = f in LEVEL_HORIZON_FIELDS
        rows.append((f, before, before - after, (before - after) / before if before else 0.0, exempt))

    assert rows, "no field produced a daily frame"
    # An expiry can only remove cells, never add them.
    assert all(n >= 0 for _, _, n, _, _ in rows)
    # and a LEVEL name really does route to the looser tier rather than to 548 -- the
    # assertion that replaced `expire_stale(d, imp, f) is d`, which required exemption.
    for f, _, _, _, is_level in rows:
        if not is_level:
            continue
        d = fundamentals_to_daily(imp, f, idx)
        lvl = int(expire_stale(d, imp, f).notna().to_numpy().sum())
        evt = int(expire_stale(d, imp, f, feature=f"_measure_{f}").notna().to_numpy().sum())
        full = int(d.notna().to_numpy().sum())
        assert lvl < full, f"{f} is a level but nothing expired -- the D3 exemption is back"
        assert lvl > evt, f"{f} expired as hard as an event; the level tier is not applying"

    print("\n=== SANITY CHECK: what a 548-day horizon would cost each proxy field ===")
    print(f"  daily grid {idx[0].date()}..{idx[-1].date()} ({len(idx)} days), post-impute")
    print(f"  {'field':<28} {'non-null':>10} {'expired':>10} {'share':>8}  exempt?")
    for f, before, n, share, exempt in sorted(rows, key=lambda r: -r[3]):
        print(f"  {f:<28} {before:>10,} {n:>10,} {share:>7.1%}  {'EXEMPT (legacy)' if exempt else ''}")
    worst = max(rows, key=lambda r: r[3])
    print(f"  worst: {worst[0]} at {worst[3]:.1%}")
    print("  CONCLUSION: this is the EVENT horizon's bite, measured with the tier routing")
    print("  deliberately bypassed via `feature=` — the fields marked LEVEL would otherwise")
    print("  report their 1,095-day cost here and the column would mix two clocks.")
    print("  Every one of these IS expired in the tree as of phase 3; on the real panel the")
    print("  level horizon removes 3.37% of cells, spread 0.11% to 18.89% by field.")
    print("  Read each as the cost of a MISSED FILING CYCLE: >20% means the horizon is too")
    print("  tight for that field and belongs in the report, not in a silent retune.")


# --------------------------------------------------------------------------------------------
# Phase 3: the LEVEL horizon. Each shape is named for the real ticker that motivated it.
# --------------------------------------------------------------------------------------------


def test_a_jnj_shaped_history_expires_at_the_annual_horizon():
    """JNJ: two early filings carry a value, 29 later ones carry NULL.

    This is the case the exemption cost the most. Measured on the live archive,
    `f_insider_ownership_pct` for JNJ was 7,671 daily cells spanning 30.5 years built on 2 of
    its 31 proxies (1996 and 1997, both 0.01) -- so 29 of 31 proxies said nothing and the panel
    reported an insider-ownership level for three decades anyway.

    Annual freshness now expires this level after 395 days, despite later empty proxies.
    """
    idx = pd.bdate_range("1996-01-01", "2026-09-04")
    rows = [
        {"ticker": "JNJ", "as_of": "1996-04-25", "insider_ownership_pct": 0.01},
        {"ticker": "JNJ", "as_of": "1997-04-24", "insider_ownership_pct": 0.01},
    ]
    rows += [{"ticker": "JNJ", "as_of": f"{y}-04-24", "insider_ownership_pct": None} for y in range(1998, 2027)]
    hist = pd.DataFrame(rows)

    daily = _daily(hist, "insider_ownership_pct", idx)
    capped = expire_stale(daily, hist, "insider_ownership_pct")
    second = pd.Timestamp("1997-04-24")

    def at(days):
        return _asof_value(capped, second + pd.Timedelta(days=days), "JNJ")

    assert daily["JNJ"].notna().sum() > 7_000, "the unbounded ffill shape is not reproduced"
    assert at(395) == pytest.approx(0.01), "expired before the annual horizon"
    assert pd.isna(at(1000)), "an old annual observation survived"
    # the boundary, partitioned over the whole index by age -- see the routing test for why a
    # two-point `index.asof` probe is not enough near a weekend
    capped_index = pd.DatetimeIndex(capped.index)
    ages = pd.Series((capped_index - second).days, index=capped_index)
    inside = capped["JNJ"][(ages >= 0) & (ages <= LEVEL_MAX_AGE_DAYS)]
    outside = capped["JNJ"][ages > LEVEL_MAX_AGE_DAYS]
    assert inside.notna().all(), "a cell inside the level horizon was expired"
    assert (inside - 0.01).abs().max() < 1e-9, "the surviving cells are not the filed value"
    assert outside.isna().all(), "survived its own horizon"
    # and the 29 NULL filings did not restart the clock at any point
    assert pd.isna(_asof_value(capped, pd.Timestamp("2026-01-02"), "JNJ"))

    kept, before = int(capped["JNJ"].notna().sum()), int(daily["JNJ"].notna().sum())
    print("\n=== SANITY CHECK: the JNJ shape (2 of 31 proxies carry a value) ===")
    print(f"  unbounded ffill: {before:,} daily cells from 2 filings, 30.5 years")
    print(f"  level horizon:   {kept:,} kept, {before - kept:,} expired ({(before - kept) / before:.1%})")
    print(f"  alive at +395d, expired at +1000d and +{LEVEL_MAX_AGE_DAYS + 1}d after the 1997 filing")
    print("  CONCLUSION: what is removed is fiction, not information -- there was never a")
    print("  measurement of JNJ insider ownership in 2020 to lose.")


def test_a_ford_shaped_history_expires_the_zero_and_reopens_on_the_new_filing():
    """Ford: a 0 in 2011, ten NULL years, a healthy value in 2022.

    The compound case. Ford's `f_ceo_to_director_pay_ratio` was exactly 0 for 2,769 consecutive
    trading days because two independent failures were welded together by the unbounded fill:
    the CEO leg extracted as 0 in the 2011 proxy while the director leg was healthy, and then
    the director leg vanished for ten years while the CEO leg recovered.

    Both halves are asserted. Expiring the zero is only half a fix if the 2022 filing does not
    reopen the series on its own date -- an expiry that poisoned the ticker permanently would
    trade one wrong answer for another.
    """
    idx = pd.bdate_range("2010-01-01", "2026-09-04")
    rows = [{"ticker": "F", "as_of": "2011-04-01", "ceo_to_director_pay_ratio": 0.0}]
    rows += [{"ticker": "F", "as_of": f"{y}-04-01", "ceo_to_director_pay_ratio": None} for y in range(2012, 2022)]
    rows += [{"ticker": "F", "as_of": "2022-04-01", "ceo_to_director_pay_ratio": 62.4}]
    hist = pd.DataFrame(rows)

    daily = _daily(hist, "ceo_to_director_pay_ratio", idx)
    capped = expire_stale(daily, hist, "ceo_to_director_pay_ratio", max_age_days=LEVEL_MAX_AGE_DAYS)

    def at(d):
        return _asof_value(capped, pd.Timestamp(d), "F")

    filed = pd.Timestamp("2011-04-01")
    horizon = filed + pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)
    assert horizon == pd.Timestamp("2012-04-30"), horizon

    # A ZERO IS A VALUE, so every check here is on `isna`, never on truthiness.
    assert at("2012-04-30") == 0.0, "the zero expired before its horizon"
    assert pd.isna(at("2012-05-01")), "the zero outlived its horizon"
    assert pd.isna(at("2021-12-31")), "the ten NULL years did not stay NULL"
    # the second half: the new filing reopens the series on its own date, at its own value
    assert at("2022-04-01") == pytest.approx(62.4)
    assert at("2023-05-01") == pytest.approx(62.4), "the 2022 value expired too early"
    assert pd.isna(at("2023-05-02")), "the replacement value must also expire"

    zeros_before = int((daily["F"] == 0.0).sum())
    zeros_after = int((capped["F"] == 0.0).sum())
    assert zeros_after < zeros_before
    print("\n=== SANITY CHECK: the Ford shape (a 0, ten NULL years, then a real value) ===")
    print(f"  unbounded ffill: {zeros_before:,} consecutive trading days at exactly 0")
    print(f"  level horizon:   {zeros_after:,} days, the last of them {horizon.date()}")
    print("  CONCLUSION: the zero ends after 395 days; the 2022-04-01 filing reopens at 62.4.")


def test_a_normal_annual_filer_is_bit_identical():
    """The test that proves the override is NARROW: a regular filer loses nothing at all.

    Phase 3 overrides the D3 decision that no legacy value may change. That override is
    defensible only because the horizon replaces a STALE value with UNKNOWN and never touches
    an in-range one -- so on a filer who files every year, every cell must survive, and the
    frame must compare equal cell-for-cell rather than merely in count.
    """
    idx = pd.bdate_range("2012-01-01", "2026-09-04")
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": f"{y}-04-15", "board_size": 10.0 + (y % 3)} for y in range(2012, 2027)])

    for field in ("board_size", "insider_ownership_pct", "ceo_pay_ratio"):
        h = hist.rename(columns={"board_size": field})
        daily = _daily(h, field, idx)
        pd.testing.assert_frame_equal(expire_stale(daily, h, field), daily)

    gaps = hist["as_of"].map(pd.Timestamp).diff().dt.days.dropna()
    print("\n=== SANITY CHECK: a normal annual filer is untouched ===")
    print(f"  15 filings 2012-2026, gaps {int(gaps.min())}-{int(gaps.max())} days, every one")
    print(f"  under the {LEVEL_MAX_AGE_DAYS}-day horizon")
    print("  three level fields, each bit-identical by `assert_frame_equal` -- not merely")
    print("  equal in count. THIS is what makes the D3 override narrow: only stale cells move.")


def test_the_two_family_wrappers_enforce_annual_freshness():
    """Both family wrappers retain routing and apply the same annual source-age policy."""
    idx = pd.bdate_range("2018-01-01", "2026-09-04")
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": "2019-05-01", "sop_dissent": 0.11, "board_size": 9.0}])
    event = _daily(hist, "sop_dissent", idx)
    level = _daily(hist, "board_size", idx)

    # The event wrapper still skips legacy levels; their owning level pass must handle them.
    frames, stats = expire_event_fields({"sop_dissent": event, "board_size": level}, hist, {"sop_dissent", "board_size"})
    assert "sop_dissent" in stats and "board_size" not in stats
    assert frames["board_size"] is level, "a level was expired on the event clock"
    assert int(frames["sop_dissent"].notna().sum().sum()) < int(event.notna().sum().sum())

    # The LEVEL wrapper has no skip set: a family calling it has already decided.
    lframes, lstats = expire_level_fields({"board_size": level}, hist, {"board_size"})
    expired, before = lstats["board_size"]
    assert expired > 0 and before == int(level.notna().to_numpy().sum())
    assert int(lframes["board_size"].notna().sum().sum()) == before - expired

    # Partition by age across the full index so a weekend cannot weaken the boundary check.
    assert GOVERNANCE_EVENT_MAX_AGE_DAYS == 395
    filed = pd.Timestamp("2019-05-01")
    ev = frames["sop_dissent"]
    ev_index = pd.DatetimeIndex(ev.index)
    ages = pd.Series((ev_index - filed).days, index=ev_index)
    assert int(ages.max()) > GOVERNANCE_EVENT_MAX_AGE_DAYS, "the boundary is never crossed"
    inside = ev["AAA"][(ages >= 0) & (ages <= GOVERNANCE_EVENT_MAX_AGE_DAYS)]
    outside = ev["AAA"][ages > GOVERNANCE_EVENT_MAX_AGE_DAYS]
    assert inside.notna().all(), "an in-horizon event cell was expired"
    assert outside.isna().all(), "an out-of-horizon event cell survived"

    print("\n=== SANITY CHECK: the two wrappers route to their own clocks ===")
    print(f"  expire_event_fields  -> {GOVERNANCE_EVENT_MAX_AGE_DAYS}d, and SKIPS a level")
    print(f"  expire_level_fields  -> {LEVEL_MAX_AGE_DAYS}d, with no skip set")
    print(f"  board_size on the level clock: {expired:,} of {before:,} cells expired")
    print("  both wrappers enforce the 395-day annual policy.")


def test_the_director_pay_family_declares_its_levels():
    """`director_comp.EVENT_FIELDS = frozenset()` used to mean nothing there ever expired."""
    from src.data_aggregate.utils.governance import director_comp as dc

    assert dc.EVENT_FIELDS == frozenset(), "director pay is not an event; that claim stands"
    assert dc.LEVEL_FIELDS == dc.ALL_FIELDS, "a level set that can drift from ALL_FIELDS"
    assert len(dc.LEVEL_FIELDS) == 4

    print("\n=== SANITY CHECK: the director-pay family is on the level horizon ===")
    print(f"  EVENT_FIELDS = {set(dc.EVENT_FIELDS)}  (still empty, and still correct)")
    print(f"  LEVEL_FIELDS = ALL_FIELDS = {len(dc.LEVEL_FIELDS)} fields, on {LEVEL_MAX_AGE_DAYS}d")
    print("  before phase 3 an empty EVENT_FIELDS meant an UNBOUNDED ffill:")
    print("  f_log_median_director_pay held ONE value for 20.4 years on WMB, 18.5 on PSA.")


#: ⚠ THE TWELVE FIELDS THAT DELIBERATELY HAVE NO HORIZON, with the module that decided it.
#:
#: Each is a documented, per-field DECISION recorded beside its family's `EVENT_FIELDS`, not an
#: omission -- and the distinction matters, because from the outside the two look identical.
#: `director_comp` had an empty `EVENT_FIELDS` and NO stated reason, which is why phase 3 gave
#: it a horizon; these twelve state their reasons:
#:
#:   pay_features        "a CEO's package and their share of the top five are standing facts
#:                        between proxies"
#:   provisions_features "`board_busyness` and `ceo_is_board_chair` are standing facts between
#:                        proxies; `auditor_tenure` ACCRUES daily off a start date, so ageing it
#:                        out would delete a number that is more current than the filing;
#:                        `auditor_is_big4` and `auditor_tenure_censored` describe that standing
#:                        level rather than an act"
#:   directors           "`board_turnover` IS expired and the other five are not... Turnover is
#:                        a one-year CHANGE... The five levels are standing facts and must NOT
#:                        expire" (an explicit deviation from D40, with its own test)
#:
#: ⚠ **NOW EMPTY, ON THE USER'S EXPLICIT AUTHORISATION (2026-09-09).** All twelve joined the
#: 1,095-day LEVEL horizon. The reasons quoted above were right that these are levels and wrong
#: that being a level is a reason to keep a value forever: phase 3's own argument -- "a level
#: should not expire on an EVENT clock" justifies a DIFFERENT horizon, not the absence of one --
#: applies here exactly as it did to the legacy twelve.
#:
#: Measured on the live part before the change (each cell aged against its ticker's most recent
#: proxy, a LOWER bound because that proxy need not have carried the field):
#:
#:     ceo_pay_slice / log_ceo_total_comp    3,956 cells past 1,095d   0.15%   max 4,428d (12.1y)
#:     ceo_is_board_chair                    2,263                     0.08%   max 2,631d
#:     the other nine                        1,161-1,209 each          0.04%   max 1,822d (5.0y)
#:
#: So this is a CONTRACT-CONSISTENCY fix, not a data emergency -- `insider_ownership_pct`, the
#: field that motivated phase 3, was 18.89% and 29.5 years. Said plainly because the temptation
#: with a 0.04% finding is to oversell it.
#:
#: ⚠ `auditor_tenure` WAS THE ONE REAL OBJECTION and it is resolved by precedent, not overruled.
#: Its recorded reason -- that it ACCRUES daily off a start date, so ageing it out deletes a
#: number more current than the filing -- is a genuinely different argument from the other
#: eleven. But `ceo_tenure` is the identical shape and has been on this same horizon since
#: phase 3: an accruing clock is only valid while the RELATIONSHIP persists, and after two
#: missed proxy cycles the accrual is extrapolating "this is still the auditor", not merely
#: adding days to a known date. Consistency with `ceo_tenure` decides it.
#:
#: Kept as an empty set rather than deleted: it is the registry that made this question visible
#: in the first place, and a future field arriving with no horizon should land here and be
#: argued for, not pass unnoticed.
DELIBERATELY_UNEXPIRED: frozenset[str] = frozenset()


def test_every_governance_field_is_on_a_horizon_or_deliberately_exempt():
    """No field may fall through BOTH horizon sets without a recorded decision.

    An unbounded forward-fill is silent by construction -- it returns a plausible number
    forever -- so nothing but this test would catch a field added to `ALL_FIELDS` and to
    neither `EVENT_FIELDS` nor `LEVEL_FIELDS`. That is not hypothetical:
    `provisions_features.ALL_FIELDS` exists precisely because "the two fields that WERE
    produced sat in neither and silently skipped their expiry".

    The test is an EQUALITY, not a subset check, in both directions:
      * a new unexpired field fails, because it is not in `DELIBERATELY_UNEXPIRED`;
      * giving one of the twelve a horizon ALSO fails, so the allowlist cannot rot into a
        description of code that has moved on.
    """
    from src.data_aggregate.utils.governance import director_comp as dc
    from src.data_aggregate.utils.governance import directors as dr
    from src.data_aggregate.utils.governance import pay_features as pay
    from src.data_aggregate.utils.governance import provisions_features as pv
    from src.data_aggregate.utils.governance import vote_dissent_features as vd

    families = {"pay_features": pay, "provisions_features": pv, "directors": dr, "director_comp": dc, "vote_dissent_features": vd}
    rows, loose = [], set()
    for name, module in families.items():
        declared = set(getattr(module, "ALL_FIELDS", set())) or set(module.EVENT_FIELDS)
        events = set(module.EVENT_FIELDS)
        levels = set(getattr(module, "LEVEL_FIELDS", frozenset()))
        # a field on the panel's own legacy set is expired by `panel._expire`
        mine = declared - events - levels - LEVEL_HORIZON_FIELDS
        loose |= mine
        rows.append((name, len(declared), len(events), len(levels), sorted(mine)))
        # a field on BOTH clocks has a horizon that depends on call order, which is worse
        # than either answer
        both = sorted(events & levels)
        assert not both, f"{name}: declared as BOTH event and level: {both}"

    assert loose == DELIBERATELY_UNEXPIRED, (
        f"unexpired fields changed.\n"
        f"  newly unexpired (no recorded decision): {sorted(loose - DELIBERATELY_UNEXPIRED)}\n"
        f"  now expired (drop from the allowlist):  "
        f"{sorted(DELIBERATELY_UNEXPIRED - loose)}"
    )

    print("\n=== SANITY CHECK: every field is on a horizon, or exempt ON THE RECORD ===")
    print(f"  {'family':<24} {'declared':>8} {'event':>6} {'level':>6}  no horizon")
    for name, n_all, n_ev, n_lv, mine in rows:
        print(f"  {name:<24} {n_all:>8} {n_ev:>6} {n_lv:>6}  {len(mine)}")
    print(f"  plus the {len(LEVEL_HORIZON_FIELDS)} legacy levels, expired in `panel._expire`")
    print(f"  {len(DELIBERATELY_UNEXPIRED)} fields are deliberately unexpired, each with a")
    print("  stated reason beside its family's EVENT_FIELDS. `director_comp` had an empty")
    print("  EVENT_FIELDS and NO stated reason, which is the one phase 3 changed.")
    print("  ⚠ The allowlist keeps the question open; it does not settle it.")
