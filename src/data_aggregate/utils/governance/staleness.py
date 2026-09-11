"""
staleness.py  (src/data_aggregate/utils/governance/staleness.py)
---------------------------------------------------------------
A TWO-TIER EXPIRY for governance features. `pit.fundamentals_to_daily` forward-fills the
last known value across the ENTIRE trading index, with no horizon, which is a claim the data
does not support in either of the two shapes governance comes in:

  * an annual EVENT -- a 2019 say-on-pay vote is not evidence about 2026, and a feature that
    keeps reporting it is asserting a shareholder opinion nobody expressed. 548 days.
  * a structural LEVEL -- a board does not stop having a size between proxies, so it is not
    on the event clock; but a board size from 2014 is not evidence about 2026 either. 1,095
    days, two whole missed annual cycles.

⚠ THE SECOND TIER DID NOT EXIST UNTIL 2026-09-08. Twelve levels were EXEMPT -- not on a
looser horizon, on none -- and `LEVEL_MAX_AGE_DAYS` carries the measured consequence.

WHY GOVERNANCE-LOCAL rather than `utils/common/`. 548 days encodes a governance disclosure
CADENCE -- annual meetings, annual proxies. Putting the knob in `common/` would invite the
fundamentals and sector builders to adopt a horizon that was never measured for their
cadence, and a quarterly filer expiring after 18 months means something entirely different.

WHAT IT DOES NOT DO. It does not detect a regime cliff and must not: `expire_stale` measures
the age of the filing that produced a cell, never the calendar. A field that is structurally
absent before 2011 has no filing to age, so it is already NaN and this function is a no-op on
it. See phase 2 §2 kind A -- no date literals anywhere in this package.
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily

#: How long a governance EVENT stays informative after the filing that disclosed it.
#:
#: Annual proxies and annual meetings run ~12 months apart. 548 days is a full year plus a
#: season of slack -- enough for a late filer or a meeting date that shifted -- and it expires
#: a value once a WHOLE cycle has been missed, which is the event the horizon exists to catch.
#:
#: Measured context (2026-09-07, `sec_8k_votes` since 2012, 6,732 proxy company-years): 95.9%
#: carry a director election, 94.5% an auditor vote, 91.3% a say-on-pay. So the cap bites on
#: the genuinely absent minority, not on the normal cadence.
GOVERNANCE_EVENT_MAX_AGE_DAYS = 548

#: How long a governance LEVEL stays informative. Distinct from the 548-day EVENT horizon above
#: because the two cadences differ in KIND: a shareholder vote is a dated event that expires once
#: a cycle is missed, while a board size persists between proxies by construction.
#:
#: 1,095 days = two whole missed annual cycles.
#:
#: ⚠ WHY THIS EXISTS AT ALL. Until 2026-09-08 the twelve fields below were EXEMPT -- not on a
#: looser horizon, on NONE -- so a single parsed filing was asserted as current for as long as
#: the trading index ran. Measured full-table against each cell's own producing `as_of`:
#:
#:     source column                cells      >548d     >1095d     >1825d    max age
#:     insider_ownership_pct    2,670,585    775,004    608,332    461,081   10,768 d  (29.5 y)
#:     ceo_is_founder           2,847,773    372,113    131,373     56,439    9,491 d  (26.0 y)
#:     ceo_since_year           2,868,837    361,721    165,612     79,334    7,301 d  (20.0 y)
#:     say_on_pay_support_pct   1,431,769    246,699    127,514     61,699    5,833 d
#:     avg_board_tenure         2,936,349    181,487     78,030     33,999    5,839 d
#:     ceo_pay_ratio              978,731     32,060     16,357     10,945    4,797 d
#:     ceo_equity_pay_pct       2,345,307    238,549    121,910     53,996    4,428 d
#:     ceo_total_comp           2,348,306    237,229    120,970     52,918    4,428 d
#:     pct_independent_directors 2,846,762   223,872     46,790      9,415    3,982 d
#:     pct_female_directors     2,993,174     22,543      4,837      2,343    4,428 d
#:     board_size               2,993,661     20,685      4,592      2,343    4,428 d
#:
#: Behind those tails: `insider_ownership_pct` for JNJ was 7,671 daily cells spanning 30.5 years
#: built on 2 of its 31 proxies (1996 and 1997, both 0.01), and XEL, VZ and NOC each rested on
#: ONE filing. The exemption's own argument -- that a LEVEL should not expire on an EVENT clock
#: -- is right, and it justifies a DIFFERENT horizon, not the absence of one.
#:
#: ⚠ THE TABLE ABOVE IS THE RAW-COLUMN UPPER BOUND, NOT THE BITE. It ages `def14a_llm` as
#: stored, and the panel builds on `impute_def14a`'s output instead -- which fills a value onto
#: more filings, shortens every forward-fill segment, and so expires far less. The two must not
#: be quoted interchangeably. Measured on the real panel (7,803-day index, 1995-09-01 ..
#: 2026-09-04, post-impute), the horizon removes 1,194,505 of 35,478,121 cells = **3.37%**:
#:
#:     insider_ownership_pct   18.89%      ceo_pay_ratio            0.60%
#:     log_median_director_pay  5.37%      ceo_tenure               0.57%
#:     director_cash_fee_pct    5.36%      founder_ceo              0.30%
#:     ceo_equity_pay_pct       5.23%      avg_board_tenure         0.16%
#:     ceo_to_director_pay_rat  4.43%      board_size               0.15%
#:     director_equity_pay_pct  4.14%      pct_female_directors     0.15%
#:     say_on_pay_support       3.54%      pct_independent_direct   0.11%
#:     ceo_pay_growth           3.14%
#:
#: ⚠ THE COST IS NOT UNIFORM, and the phase-3 plan's "0.17%" was `board_size` generalised to
#: twelve columns: the real spread is 0.11% to 18.89%, a factor of 172. `insider_ownership_pct`
#: is the one field anywhere near the >20% line `expire_event_fields` names below, and it is
#: REPORTED rather than retuned -- it is disclosed on so few proxies that most of its coverage
#: was forward-filled fiction, and a horizon loose enough to keep 18.89% of it would have to be
#: ~30 years, which is not a horizon.
LEVEL_MAX_AGE_DAYS = 1095

#: The twelve features that ship out of `cube_part_governance` as structural LEVELS, and are
#: therefore on `LEVEL_MAX_AGE_DAYS` rather than the event horizon. Two of them
#: (`ceo_pay_growth`, `ceo_pay_vs_revenue_growth`) are delta-shaped and would otherwise qualify
#: as events on every rule in this module; they are levels here because their PRODUCING cadence
#: is the annual proxy, which is what the horizon measures.
#:
#: Named on the FEATURE side (what `panel.py` emits), not the source-column side, because that
#: is the vocabulary a family module holds when it decides how to expire something.
#:
#: ⚠ THIS SET NO LONGER EXEMPTS ANYTHING. It was `LEGACY_EXEMPT_FROM_EXPIRY`, the D3 guard, and
#: membership meant "never expires". Phase 3 overrides D3 on the user's explicit authorisation
#: and membership now means "expires at 1,095 days instead of 548". The old name is kept as an
#: alias below because three test modules import it.
LEVEL_HORIZON_FIELDS: frozenset[str] = frozenset({
    "ceo_pay_growth", "ceo_pay_vs_revenue_growth", "ceo_pay_ratio", "ceo_equity_pay_pct",
    "ceo_tenure", "founder_ceo", "pct_independent_directors", "pct_female_directors",
    "board_size", "avg_board_tenure", "say_on_pay_support", "insider_ownership_pct",
})

#: ⚠ DEPRECATED NAME, kept because the set itself is still the right set -- only its MEANING
#: changed (see above). Read `LEVEL_HORIZON_FIELDS`; this alias exists so a stale import does
#: not break, and it should go once the tests naming it are next touched.
LEGACY_EXEMPT_FROM_EXPIRY: frozenset[str] = LEVEL_HORIZON_FIELDS


def horizon_for(feature: str, default: int = GOVERNANCE_EVENT_MAX_AGE_DAYS) -> int:
    """The staleness horizon a feature is on, in days.

    `LEVEL_MAX_AGE_DAYS` for a structural level, `default` (the 548-day EVENT horizon) for
    everything else. One function so the two-tier rule is stated once and every call site --
    the panel, the family modules, the tests -- reads the same answer.
    """
    return LEVEL_MAX_AGE_DAYS if feature in LEVEL_HORIZON_FIELDS else default


def expire_stale(daily: pd.DataFrame, history: pd.DataFrame, field: str,
                 max_age_days: int | None = None,
                 feature: str | None = None) -> pd.DataFrame:
    """NaN out cells of a ffilled daily frame that are older than `max_age_days`.

    Takes the ORIGINAL filing `history` so the age is measured against the real `as_of` that
    produced the cell. Once a value has been forward-filled its provenance is gone from the
    daily frame -- every day looks like the day it was filed -- so the age cannot be
    recovered downstream and has to be carried in alongside.

    `field` names the column in `history`; `feature` names the emitted feature when the two
    differ (`say_on_pay_support_pct` -> `say_on_pay_support`), and it is `feature` that picks
    the horizon.

    `max_age_days=None` means "ask `horizon_for`", which is the two-tier rule: a level gets
    1,095 days and everything else 548. Passing a number OVERRIDES that, which is what the
    family wrappers below do and what a measurement harness does when it wants one horizon
    applied uniformly. ⚠ Before phase 3, a level was returned unchanged here instead; nothing
    in this module exempts a field from expiry any more.
    """
    name = feature if feature is not None else field
    if max_age_days is None:
        max_age_days = horizon_for(name)
    if daily is None or daily.empty or history is None or history.empty:
        return daily
    if field not in history.columns or "as_of" not in history.columns:
        return daily

    # The producing `as_of`, pivoted with the SAME ffill as the value itself rather than a
    # second hand-written one: `_produced_at` is the filing date carried as a payload. Rows
    # where the field is NULL are dropped first, because `fundamentals_to_daily` aggregates
    # with `last` (which skips NaN) -- so a cell's value comes from the last NON-NULL filing,
    # and its age must be measured against that same filing, not against a later empty one.
    h = history[["ticker", "as_of", field]].copy()
    h["as_of"] = pd.to_datetime(h["as_of"], errors="coerce")
    h = h.dropna(subset=["ticker", "as_of", field])
    if h.empty:
        return daily
    h["_produced_at"] = h["as_of"].map(pd.Timestamp.toordinal).astype("float64")

    produced = fundamentals_to_daily(h, "_produced_at", daily.index)
    if produced.empty:
        return daily
    produced = produced.reindex(columns=daily.columns)

    today = pd.Series(daily.index.map(pd.Timestamp.toordinal), index=daily.index,
                      dtype="float64")
    age = produced.rsub(today, axis=0)
    # `> max_age` is False wherever the age is NaN (no filing yet), which is the right
    # answer: those cells are already NaN in `daily` and masking them changes nothing.
    return daily.mask(age > float(max_age_days))


def _expire_family(
    frames: dict[str, pd.DataFrame], history: pd.DataFrame,
    selected: frozenset[str] | set[str],
    sources: dict[str, str] | None,
    max_age_days: int,
    skip: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """The shared workhorse of the two wrappers below. `skip` is never expired at all."""
    out: dict[str, pd.DataFrame] = {}
    stats: dict[str, tuple[int, int]] = {}
    for name, frame in frames.items():
        if name not in selected or name in skip:
            out[name] = frame
            continue
        field = (sources or {}).get(name, name)
        before = int(frame.notna().to_numpy().sum())
        capped = expire_stale(frame, history, field, max_age_days=max_age_days, feature=name)
        after = int(capped.notna().to_numpy().sum())
        out[name] = capped
        stats[name] = (before - after, before)
    return out, stats


def expire_event_fields(
    frames: dict[str, pd.DataFrame], history: pd.DataFrame,
    event_fields: frozenset[str] | set[str],
    sources: dict[str, str] | None = None,
    max_age_days: int = GOVERNANCE_EVENT_MAX_AGE_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """Apply the EVENT horizon to every field in a family's `EVENT_FIELDS`.

    Each family module declares its own `EVENT_FIELDS` (phase 2 §3.3) and calls this once, so
    the horizon is applied in exactly one place per family and the BITE is measured in the
    same pass. `sources` maps a feature name to its `history` column where they differ.

    Returns `(frames, stats)` with `stats[feature] = (expired, non_null_before)` -- the second
    number is what makes the first readable. A cap nulling >20% of a modern-era field is a
    sign the horizon is too tight FOR THAT FIELD; report it, do not silently retune it.

    ⚠ A LEVEL IS STILL SKIPPED HERE, and for an unchanged reason: a field on
    `LEVEL_HORIZON_FIELDS` has a 1,095-day horizon, so letting a family put it on the 548-day
    event clock by declaring it in `EVENT_FIELDS` would be a silent downgrade. It expires
    through `expire_level_fields` instead. Before phase 3 the skip meant "never expires"; now
    it means "not on THIS clock".
    """
    return _expire_family(frames, history, event_fields, sources, max_age_days,
                          skip=LEVEL_HORIZON_FIELDS)


def expire_level_fields(
    frames: dict[str, pd.DataFrame], history: pd.DataFrame,
    level_fields: frozenset[str] | set[str],
    sources: dict[str, str] | None = None,
    max_age_days: int = LEVEL_MAX_AGE_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """Apply the LEVEL horizon to every field a family declares as a structural level.

    Same contract as `expire_event_fields` -- same `(frames, stats)` return, same `sources`
    mapping -- on the 1,095-day clock instead of the 548-day one, and with NO skip set: a
    family calling this has already decided its fields are levels, so there is nothing to
    protect them from.

    It exists because "not an event" was previously the same statement as "never expires", and
    `director_comp.EVENT_FIELDS = frozenset()` is what that cost: the director-pay family
    declared nothing as an event, so nothing aged out, and `f_log_median_director_pay` held one
    value for 20.4 years on WMB and 18.5 on PSA.
    """
    return _expire_family(frames, history, level_fields, sources, max_age_days)
