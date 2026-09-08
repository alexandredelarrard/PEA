"""
staleness.py  (src/data_aggregate/utils/governance/staleness.py)
---------------------------------------------------------------
An EXPIRY for governance EVENTS. `pit.fundamentals_to_daily` forward-fills the last known
value across the ENTIRE trading index, with no horizon. For a structural LEVEL that is
right -- a board does not stop having a size between proxies. For an annual EVENT it is a
claim the data does not support: a 2019 say-on-pay vote is not evidence about 2026, and a
feature that keeps reporting it is asserting a shareholder opinion nobody expressed.

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

#: ⚠ THE D3 GUARD -- these are the twelve features that ship TODAY out of `cube_part_extras`,
#: and the expiry must never touch them. Two of them (`ceo_pay_growth`,
#: `ceo_pay_vs_revenue_growth`) are delta-shaped and would otherwise qualify as events on
#: every rule in this module -- and expiring their tails would silently change the cells
#: feeding the live `comp_governance` composite and the monotone list. That is precisely what
#: D3 forbids, so the exemption is written down as a DECISION rather than left as an omission
#: for someone to later "tidy up".
#:
#: Named on the FEATURE side (what `panel.py` emits), not the source-column side, because that
#: is the vocabulary a family module holds when it decides whether to expire something.
LEGACY_EXEMPT_FROM_EXPIRY: frozenset[str] = frozenset({
    "ceo_pay_growth", "ceo_pay_vs_revenue_growth", "ceo_pay_ratio", "ceo_equity_pay_pct",
    "ceo_tenure", "founder_ceo", "pct_independent_directors", "pct_female_directors",
    "board_size", "avg_board_tenure", "say_on_pay_support", "insider_ownership_pct",
})


def expire_stale(daily: pd.DataFrame, history: pd.DataFrame, field: str,
                 max_age_days: int = GOVERNANCE_EVENT_MAX_AGE_DAYS,
                 feature: str | None = None) -> pd.DataFrame:
    """NaN out cells of a ffilled daily frame that are older than `max_age_days`.

    Takes the ORIGINAL filing `history` so the age is measured against the real `as_of` that
    produced the cell. Once a value has been forward-filled its provenance is gone from the
    daily frame -- every day looks like the day it was filed -- so the age cannot be
    recovered downstream and has to be carried in alongside.

    `field` names the column in `history`; `feature` names the emitted feature when the two
    differ (`say_on_pay_support_pct` -> `say_on_pay_support`), and it is `feature` that is
    checked against `LEGACY_EXEMPT_FROM_EXPIRY`. An exempt field is returned UNCHANGED and
    unwrapped -- not copied, not re-indexed -- so a caller cannot accidentally perturb it.
    """
    name = feature if feature is not None else field
    if name in LEGACY_EXEMPT_FROM_EXPIRY:
        return daily
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


def expire_event_fields(
    frames: dict[str, pd.DataFrame], history: pd.DataFrame,
    event_fields: frozenset[str] | set[str],
    sources: dict[str, str] | None = None,
    max_age_days: int = GOVERNANCE_EVENT_MAX_AGE_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    """Apply `expire_stale` to every EVENT field in a family's daily-frame dict.

    Each family module declares its own `EVENT_FIELDS` (phase 2 §3.3) and calls this once, so
    the horizon is applied in exactly one place per family and the BITE is measured in the
    same pass. `sources` maps a feature name to its `history` column where they differ.

    Returns `(frames, stats)` with `stats[feature] = (expired, non_null_before)` -- the second
    number is what makes the first readable. A cap nulling >20% of a modern-era field is a
    sign the horizon is too tight FOR THAT FIELD; report it, do not silently retune it.
    """
    out: dict[str, pd.DataFrame] = {}
    stats: dict[str, tuple[int, int]] = {}
    for name, frame in frames.items():
        if name not in event_fields or name in LEGACY_EXEMPT_FROM_EXPIRY:
            out[name] = frame
            continue
        field = (sources or {}).get(name, name)
        before = int(frame.notna().to_numpy().sum())
        capped = expire_stale(frame, history, field, max_age_days=max_age_days, feature=name)
        after = int(capped.notna().to_numpy().sum())
        out[name] = capped
        stats[name] = (before - after, before)
    return out, stats
