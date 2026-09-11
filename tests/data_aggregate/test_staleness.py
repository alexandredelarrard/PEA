"""
test_staleness.py  (tests/data_aggregate/test_staleness.py)
----------------------------------------------------------
The governance staleness horizons — `expire_stale`, on both of its two tiers: the 548-day
EVENT horizon (D21) and the 1,095-day LEVEL horizon (`LEVEL_MAX_AGE_DAYS`, phase 3).

Properties, most of them synthetic because they are about the boundary:

  * 17 months old SURVIVES, 19 months old EXPIRES — 548 days sits between them on purpose;
  * a LEVEL field expires at 1,095 days — not at 548, and not never. ⚠ This file asserted
    the opposite until 2026-09-08: the D3 guard required the twelve legacy features to be
    returned BY IDENTITY at any age, and phase 3 overrides that. See `LEVEL_MAX_AGE_DAYS` for
    what the exemption cost (JNJ: 7,671 cells over 30.5 years built on 2 of 31 proxies);
  * a field with two filings expires only once the LATER one has aged out — the age is
    measured against the filing that produced the cell, not the first one ever seen;
  * a NULL-carrying filing does not reset the clock: the age belongs to the last filing that
    actually disclosed a value, which is the one `fundamentals_to_daily` took the cell from.

Then the real-data BITE: what share of each def14a field's non-null daily cells the cap would
null, measured with the tier deliberately forced so the table reports a HORIZON rather than a
routing decision. On the real panel the level horizon removes 3.37% of cells overall, spread
0.11% (`pct_independent_directors`) to 18.89% (`insider_ownership_pct`) -- a factor of 172,
which is why `LEVEL_MAX_AGE_DAYS` records the per-field figures instead of one headline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.staleness import (
    GOVERNANCE_EVENT_MAX_AGE_DAYS, LEGACY_EXEMPT_FROM_EXPIRY, LEVEL_HORIZON_FIELDS,
    LEVEL_MAX_AGE_DAYS, expire_event_fields, expire_level_fields, expire_stale, horizon_for,
)


def _daily(history: pd.DataFrame, field: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return fundamentals_to_daily(history, field, idx)


def test_expiry_boundary_on_both_tiers():
    """17 months lives, 19 months dies -- and a LEVEL lives to 1,095 days, then dies."""
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": "2020-06-01",
                          "sop_dissent": 0.11, "ceo_pay_growth": 0.11}])

    daily = _daily(hist, "sop_dissent", idx)
    capped = expire_stale(daily, hist, "sop_dissent")

    filed = pd.Timestamp("2020-06-01")
    d17 = filed + pd.Timedelta(days=int(30.4 * 17))     # ~517 days -> inside the horizon
    d19 = filed + pd.Timedelta(days=int(30.4 * 19))     # ~577 days -> outside it
    assert (d17 - filed).days <= GOVERNANCE_EVENT_MAX_AGE_DAYS < (d19 - filed).days

    v17 = capped.loc[capped.index.asof(d17), "AAA"]
    v19 = capped.loc[capped.index.asof(d19), "AAA"]
    assert v17 == pytest.approx(0.11), "a 17-month-old event was expired"
    assert pd.isna(v19), "a 19-month-old event survived"

    # the exact boundary: still alive on day 548, gone on day 549
    assert capped.loc[capped.index.asof(filed + pd.Timedelta(days=548)), "AAA"] \
        == pytest.approx(0.11)
    assert pd.isna(capped.loc[capped.index.asof(filed + pd.Timedelta(days=549)), "AAA"])

    # --- the LEVEL tier: a legacy field is on 1,095 days, NOT exempt and NOT on 548 ---
    # ⚠ This block asserted the opposite until 2026-09-08. It required `expire_stale` to return
    # a legacy field BY IDENTITY (`kept is legacy`) -- the D3 guard, which phase 3 overrides on
    # the user's explicit authorisation. The test is inverted rather than deleted because the
    # boundary it guards is still the thing worth guarding; only the number moved.
    legacy = _daily(hist, "ceo_pay_growth", idx)
    kept = expire_stale(legacy, hist, "ceo_pay_growth")
    assert "ceo_pay_growth" in LEVEL_HORIZON_FIELDS
    assert len(LEVEL_HORIZON_FIELDS) == 12, "the twelve legacy features are the LEVEL tier"
    assert LEGACY_EXEMPT_FROM_EXPIRY is LEVEL_HORIZON_FIELDS, "the alias drifted from the set"
    assert horizon_for("ceo_pay_growth") == LEVEL_MAX_AGE_DAYS == 1095
    assert horizon_for("sop_dissent") == GOVERNANCE_EVENT_MAX_AGE_DAYS == 548

    def at(frame, days):
        return frame.loc[frame.index.asof(filed + pd.Timedelta(days=days)), "AAA"]

    # survives where an EVENT would already be dead -- that is what the second tier buys
    assert at(kept, 549) == pytest.approx(0.11), "a level expired on the EVENT horizon"
    assert at(kept, 1095) == pytest.approx(0.11), "a level died before its own horizon"
    assert pd.isna(at(kept, 1096)), "a level never expired at all (the pre-phase-3 defect)"

    print("\n=== SANITY CHECK: governance expiry, both tiers (boundary) ===")
    print(f"  EVENT horizon = {GOVERNANCE_EVENT_MAX_AGE_DAYS} days   "
          f"LEVEL horizon = {LEVEL_MAX_AGE_DAYS} days")
    print(f"  filed 2020-06-01: +17mo ({(d17 - filed).days}d) = {v17}  (survives)")
    print(f"                    +19mo ({(d19 - filed).days}d) = NaN   (expired)")
    print("  event boundary: alive on day 548, NaN on day 549.")
    print("  level boundary: `ceo_pay_growth` alive on day 549 AND on day 1095, NaN on 1096.")
    print(f"  the {len(LEVEL_HORIZON_FIELDS)} legacy features are the LEVEL tier. Before")
    print("  2026-09-08 they were EXEMPT — returned by identity, never expiring at any age.")


def test_expiry_uses_the_filing_that_produced_the_cell():
    """Two filings: the clock restarts on the later one — and a NULL filing does not restart it."""
    idx = pd.bdate_range("2018-01-01", "2024-12-31")
    hist = pd.DataFrame([
        {"ticker": "AAA", "as_of": "2019-05-01", "sop_dissent": 0.05},
        {"ticker": "AAA", "as_of": "2020-05-01", "sop_dissent": 0.09},
        # a LATER filing that disclosed nothing for this field: it must not reset the clock
        {"ticker": "AAA", "as_of": "2021-05-01", "sop_dissent": None},
        # a different ticker whose only filing is ancient -> fully expired
        {"ticker": "BBB", "as_of": "2018-03-01", "sop_dissent": 0.42},
    ])
    daily = _daily(hist, "sop_dissent", idx)
    capped = expire_stale(daily, hist, "sop_dissent")

    def at(d: str, t: str):
        return capped.loc[capped.index.asof(pd.Timestamp(d)), t]

    # 2020-06-01 is 396d after the 2019 filing but only 31d after the 2020 one -> alive at 0.09
    assert at("2020-06-01", "AAA") == pytest.approx(0.09)
    # 2021-10-01 is 518d after the 2020-05-01 filing that produced the cell -> still alive,
    # even though a 2021-05-01 filing exists: that filing carried no value, so it is not the
    # cell's producer and must not have restarted the clock.
    assert at("2021-10-01", "AAA") == pytest.approx(0.09)
    # 2022-01-01 is 611d after it -> expired
    assert pd.isna(at("2022-01-01", "AAA"))
    # BBB's single 2018 filing is dead everywhere after its horizon
    assert at("2018-06-01", "BBB") == pytest.approx(0.42)
    assert pd.isna(at("2021-01-01", "BBB"))

    frames, stats = expire_event_fields({"sop_dissent": daily}, hist, {"sop_dissent"})
    expired, before = stats["sop_dissent"]

    print("\n=== SANITY CHECK: governance event expiry (provenance) ===")
    print("  AAA filed 2019-05-01 (0.05), 2020-05-01 (0.09), 2021-05-01 (NULL).")
    print("   2020-06-01 -> 0.09 (31d old)   2021-10-01 -> 0.09 (518d old, the NULL filing")
    print("   did NOT restart the clock)     2022-01-01 -> NaN (611d old)")
    print("  BBB's lone 2018 filing is alive at +3mo and dead by 2021.")
    print(f"  expire_event_fields bite: {expired} of {before} non-null cells "
          f"({expired / before:.1%})")
    print("  CONCLUSION: the age is measured against the filing that PRODUCED the cell,")
    print("  which is the only reason a forward-filled frame can be expired at all.")


def test_expiry_bite_on_real_def14a():
    """How hard does 548 days bite each proxy field, with the tier forced to the event one?"""
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        raw = ctx.store.load("def14a_llm")
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_llm not reachable ({e})")
    if raw is None or raw.empty:
        pytest.skip("def14a_llm empty")

    from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
    imp, _ = impute_def14a(raw)
    idx = pd.bdate_range("2011-01-03", "2026-09-04")        # the modern era only

    fields = ["say_on_pay_support_pct", "ceo_total_comp", "avg_other_public_boards",
              "board_size", "pct_independent_directors", "poison_pill", "majority_voting"]
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
        rows.append((f, before, before - after,
                     (before - after) / before if before else 0.0, exempt))

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
        print(f"  {f:<28} {before:>10,} {n:>10,} {share:>7.1%}  "
              f"{'EXEMPT (legacy)' if exempt else ''}")
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

def test_a_jnj_shaped_history_expires_at_the_level_horizon_not_the_event_one():
    """JNJ: two early filings carry a value, 29 later ones carry NULL.

    This is the case the exemption cost the most. Measured on the live archive,
    `f_insider_ownership_pct` for JNJ was 7,671 daily cells spanning 30.5 years built on 2 of
    its 31 proxies (1996 and 1997, both 0.01) -- so 29 of 31 proxies said nothing and the panel
    reported an insider-ownership level for three decades anyway.

    The assertion that matters most is the NEGATIVE one: alive well past 548 days. A level must
    not be dragged onto the event clock as the price of expiring at all, or every annual filer
    who skips a single proxy loses a year of otherwise-good structure.
    """
    idx = pd.bdate_range("1996-01-01", "2026-09-04")
    rows = [{"ticker": "JNJ", "as_of": "1996-04-25", "insider_ownership_pct": 0.01},
            {"ticker": "JNJ", "as_of": "1997-04-24", "insider_ownership_pct": 0.01}]
    rows += [{"ticker": "JNJ", "as_of": f"{y}-04-24", "insider_ownership_pct": None}
             for y in range(1998, 2027)]
    hist = pd.DataFrame(rows)

    daily = _daily(hist, "insider_ownership_pct", idx)
    capped = expire_stale(daily, hist, "insider_ownership_pct")
    second = pd.Timestamp("1997-04-24")

    def at(days):
        return capped.loc[capped.index.asof(second + pd.Timedelta(days=days)), "JNJ"]

    assert daily["JNJ"].notna().sum() > 7_000, "the unbounded ffill shape is not reproduced"
    assert at(548) == pytest.approx(0.01), "expired on the EVENT horizon -- wrong tier"
    assert at(1000) == pytest.approx(0.01), "expired before its own horizon"
    # the boundary, partitioned over the whole index by age -- see the routing test for why a
    # two-point `index.asof` probe is not enough near a weekend
    ages = pd.Series((capped.index - second).days, index=capped.index)
    inside = capped["JNJ"][(ages >= 0) & (ages <= LEVEL_MAX_AGE_DAYS)]
    outside = capped["JNJ"][ages > LEVEL_MAX_AGE_DAYS]
    assert inside.notna().all(), "a cell inside the level horizon was expired"
    assert (inside - 0.01).abs().max() < 1e-9, "the surviving cells are not the filed value"
    assert outside.isna().all(), "survived its own horizon"
    # and the 29 NULL filings did not restart the clock at any point
    assert pd.isna(capped.loc[capped.index.asof(pd.Timestamp("2026-01-02")), "JNJ"])

    kept, before = int(capped["JNJ"].notna().sum()), int(daily["JNJ"].notna().sum())
    print("\n=== SANITY CHECK: the JNJ shape (2 of 31 proxies carry a value) ===")
    print(f"  unbounded ffill: {before:,} daily cells from 2 filings, 30.5 years")
    print(f"  level horizon:   {kept:,} kept, {before - kept:,} expired "
          f"({(before - kept) / before:.1%})")
    print(f"  alive at +548d and +1000d, NaN at +{LEVEL_MAX_AGE_DAYS + 1}d after the 1997 filing")
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
    rows += [{"ticker": "F", "as_of": f"{y}-04-01", "ceo_to_director_pay_ratio": None}
             for y in range(2012, 2022)]
    rows += [{"ticker": "F", "as_of": "2022-04-01", "ceo_to_director_pay_ratio": 62.4}]
    hist = pd.DataFrame(rows)

    daily = _daily(hist, "ceo_to_director_pay_ratio", idx)
    capped = expire_stale(daily, hist, "ceo_to_director_pay_ratio",
                          max_age_days=LEVEL_MAX_AGE_DAYS)

    def at(d):
        return capped.loc[capped.index.asof(pd.Timestamp(d)), "F"]

    filed = pd.Timestamp("2011-04-01")
    horizon = filed + pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)
    assert horizon == pd.Timestamp("2014-03-31"), horizon

    # A ZERO IS A VALUE, so every check here is on `isna`, never on truthiness.
    assert at("2014-03-31") == 0.0, "the zero expired before its horizon"
    assert pd.isna(at("2014-04-01")), "the zero outlived its horizon"
    assert pd.isna(at("2021-12-31")), "the ten NULL years did not stay NULL"
    # the second half: the new filing reopens the series on its own date, at its own value
    assert at("2022-04-01") == pytest.approx(62.4)
    assert at("2024-01-02") == pytest.approx(62.4), "the 2022 value expired too early"

    zeros_before = int((daily["F"] == 0.0).sum())
    zeros_after = int((capped["F"] == 0.0).sum())
    assert zeros_after < zeros_before
    print("\n=== SANITY CHECK: the Ford shape (a 0, ten NULL years, then a real value) ===")
    print(f"  unbounded ffill: {zeros_before:,} consecutive trading days at exactly 0")
    print(f"  level horizon:   {zeros_after:,} days, the last of them {horizon.date()}")
    print("  measured the same way on the LIVE archive: 2,769 -> 753 days, last zero")
    print("  2014-03-31 -- and the 2022-04-01 filing reopens the series at 62.4 that day.")


def test_a_normal_annual_filer_is_bit_identical():
    """The test that proves the override is NARROW: a regular filer loses nothing at all.

    Phase 3 overrides the D3 decision that no legacy value may change. That override is
    defensible only because the horizon replaces a STALE value with UNKNOWN and never touches
    an in-range one -- so on a filer who files every year, every cell must survive, and the
    frame must compare equal cell-for-cell rather than merely in count.
    """
    idx = pd.bdate_range("2012-01-01", "2026-09-04")
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": f"{y}-04-15", "board_size": 10.0 + (y % 3)}
                         for y in range(2012, 2027)])

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


def test_the_event_tier_is_unperturbed_and_the_two_wrappers_route_correctly():
    """548 still means 548, and neither wrapper can put a field on the other's clock."""
    idx = pd.bdate_range("2018-01-01", "2026-09-04")
    hist = pd.DataFrame([{"ticker": "AAA", "as_of": "2019-05-01",
                          "sop_dissent": 0.11, "board_size": 9.0}])
    event = _daily(hist, "sop_dissent", idx)
    level = _daily(hist, "board_size", idx)

    # The EVENT wrapper refuses to expire a level, so a family that mis-declares one in its
    # `EVENT_FIELDS` cannot silently downgrade it from 1,095 days to 548.
    frames, stats = expire_event_fields({"sop_dissent": event, "board_size": level}, hist,
                                        {"sop_dissent", "board_size"})
    assert "sop_dissent" in stats and "board_size" not in stats
    assert frames["board_size"] is level, "a level was expired on the event clock"
    assert int(frames["sop_dissent"].notna().sum().sum()) < int(event.notna().sum().sum())

    # The LEVEL wrapper has no skip set: a family calling it has already decided.
    lframes, lstats = expire_level_fields({"board_size": level}, hist, {"board_size"})
    expired, before = lstats["board_size"]
    assert expired > 0 and before == int(level.notna().to_numpy().sum())
    assert int(lframes["board_size"].notna().sum().sum()) == before - expired

    # and the event horizon itself is untouched by phase 3. Partitioned over the WHOLE index
    # by age rather than probed at two dates: `index.asof(filed + 549d)` walks back to the
    # previous business day, so on a horizon that lands near a weekend a two-point check can
    # test day 548 twice and pass while saying nothing.
    assert GOVERNANCE_EVENT_MAX_AGE_DAYS == 548
    filed = pd.Timestamp("2019-05-01")
    ev = frames["sop_dissent"]
    ages = pd.Series((ev.index - filed).days, index=ev.index)
    assert int(ages.max()) > GOVERNANCE_EVENT_MAX_AGE_DAYS, "the boundary is never crossed"
    inside = ev["AAA"][(ages >= 0) & (ages <= GOVERNANCE_EVENT_MAX_AGE_DAYS)]
    outside = ev["AAA"][ages > GOVERNANCE_EVENT_MAX_AGE_DAYS]
    assert inside.notna().all(), "an in-horizon event cell was expired"
    assert outside.isna().all(), "an out-of-horizon event cell survived"

    print("\n=== SANITY CHECK: the two wrappers route to their own clocks ===")
    print(f"  expire_event_fields  -> {GOVERNANCE_EVENT_MAX_AGE_DAYS}d, and SKIPS a level")
    print(f"  expire_level_fields  -> {LEVEL_MAX_AGE_DAYS}d, with no skip set")
    print(f"  board_size on the level clock: {expired:,} of {before:,} cells expired")
    print("  the 548-day boundary is bit-for-bit where it was before phase 3.")


def test_the_director_pay_family_declares_its_levels():
    """`director_comp.EVENT_FIELDS = frozenset()` used to mean nothing there ever expired."""
    from src.data_aggregate.utils.governance import director_comp as dc

    assert dc.EVENT_FIELDS == frozenset(), "director pay is not an event; that claim stands"
    assert dc.LEVEL_FIELDS == dc.ALL_FIELDS, "a level set that can drift from ALL_FIELDS"
    assert len(dc.LEVEL_FIELDS) == 4

    print("\n=== SANITY CHECK: the director-pay family is on the level horizon ===")
    print(f"  EVENT_FIELDS = {set(dc.EVENT_FIELDS)}  (still empty, and still correct)")
    print(f"  LEVEL_FIELDS = ALL_FIELDS = {len(dc.LEVEL_FIELDS)} fields, on "
          f"{LEVEL_MAX_AGE_DAYS}d")
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

    families = {"pay_features": pay, "provisions_features": pv, "directors": dr,
                "director_comp": dc, "vote_dissent_features": vd}
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
        f"{sorted(DELIBERATELY_UNEXPIRED - loose)}")

    print("\n=== SANITY CHECK: every field is on a horizon, or exempt ON THE RECORD ===")
    print(f"  {'family':<24} {'declared':>8} {'event':>6} {'level':>6}  no horizon")
    for name, n_all, n_ev, n_lv, mine in rows:
        print(f"  {name:<24} {n_all:>8} {n_ev:>6} {n_lv:>6}  {len(mine)}")
    print(f"  plus the {len(LEVEL_HORIZON_FIELDS)} legacy levels, expired in `panel._expire`")
    print(f"  {len(DELIBERATELY_UNEXPIRED)} fields are deliberately unexpired, each with a")
    print("  stated reason beside its family's EVENT_FIELDS. `director_comp` had an empty")
    print("  EVENT_FIELDS and NO stated reason, which is the one phase 3 changed.")
    print("  ⚠ The allowlist keeps the question open; it does not settle it.")
