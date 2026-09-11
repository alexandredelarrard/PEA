"""
names.py  (src/data_aggregate/utils/governance/names.py)
--------------------------------------------------------------------------------
The governance-facing layer over `src/utils/names.py`: everything that is specific to
comparing the PERSON in a `def14a_llm.ceo_name_proxy` cell, and nothing that is not.

The key itself is deliberately NOT redefined here. `person_key` is shared with the extraction
that wrote `sec_8k_votes`' role-category columns, so this module only decides what STRING to
hand it -- which for a CEO cell means splitting a co-CEO listing first.

Used at exactly four places, and nowhere else a name is compared:
  * the CEO-turnover guard, comparing a filing's identity against the previous filing's;
  * `def14a_impute`'s identity gap fill, comparing the value before a gap against the one after;
  * the CEO<->NEO cross-check against `def14a_executive_comp.name`;
  * the election-nominee lookup against `sec_8k_votes.nominee_votes_json[].name` (diagnostic).

⚠ `None` means UNKNOWN, never a value. Two `None`s must never compare equal in a turnover or
agreement test, because "we do not know who ran this company either year" is not evidence that
the CEO did not change -- and reading it as evidence is exactly how a real transition slips
past the guard. Every caller compares with an explicit not-None precondition.
"""
from __future__ import annotations

import re

import pandas as pd

from src.utils.names import person_key

#: A cell naming more than one person. Co-CEOs are printed as `A. Jayson Adair; Jeffrey Liaw`
#: (CPRT) or joined by a standalone " and " / " & ". `\s+and\s+` needs the surrounding
#: whitespace: without it the pattern fires inside surnames such as "Anderson" or "Alexander".
_CO_NAME_SPLIT_RE = re.compile(r"\s*;\s*|\s+and\s+|\s*&\s*", re.I)


def split_co_names(value: object) -> list[str]:
    """Every person named in one cell, in the order the filer printed them. `[]` for a blank.

    Left UNSPLIT, `A. Jayson Adair; Jeffrey Liaw` keys `liaw|a` -- a chimera of the first
    person's initial and the second's surname, which matches neither of them and therefore
    reads as a turnover in BOTH directions, the year it appears and the year it stops.
    """
    if not isinstance(value, str):
        return []
    return [p for p in (part.strip() for part in _CO_NAME_SPLIT_RE.split(value)) if p]


def is_multi_name(value: object) -> bool:
    """True when a cell names two or more people. Counted and logged at build time so the
    affected population is a number rather than a guess -- there is no feature and no flag
    column behind it (D28)."""
    return len(split_co_names(value)) > 1


def ceo_identity(value: object) -> str | None:
    """The stable CEO identity for a `def14a_llm.ceo_name_proxy` cell, or None for UNKNOWN.

    `person_key` applied to the FIRST person listed (D28). Taking the first keeps pay growth
    computable for the continuing CEO across a co-CEO year, which is the question the
    compensation family actually asks: Adair is still CEO and his package is comparable year
    over year.

    ⚠ Accepted cost, stated because it is real: a genuine co-CEO transition
    (`Adair` -> `Adair; Liaw`) therefore reads as NO turnover, so the governance-structure
    change is invisible. Switching to the CEO-duality reading is this split rule plus one flag,
    not a redesign.
    """
    names = split_co_names(value)
    if not names:
        return None
    return person_key(names[0])


def ceo_identity_series(values: pd.Series) -> pd.Series:
    """`ceo_identity` over a column, as an object Series of keys and Nones.

    A plain `.map` would be enough on a numpy-backed column, but a pyarrow-backed string column
    turns a returned `None` into `pd.NA`, and `pd.NA == pd.NA` is `pd.NA` -- neither True nor
    False -- which silently poisons every agreement test downstream. Casting to `object` first
    keeps the Nones as Nones, so `!=` stays a plain boolean comparison.
    """
    return values.astype(object).map(ceo_identity)


def ceo_identity_changed(names: pd.Series, groups: pd.Series) -> pd.Series:
    """1.0 where the CEO CHANGED versus the previous row of the same `groups` value, 0.0 where
    it is the same person, NaN where either side's identity is UNKNOWN. Indexed like `names`.

    `names` are raw `ceo_name_proxy` cells and `groups` the ticker column beside them. The
    caller must already have sorted the frame chronologically WITHIN `groups`: the comparison
    is a plain `shift(1)` and cannot detect a frame handed to it out of order.

    ⚠ THE CALLER MUST EXPAND THIS FROM THE SAME ROWS AS THE VALUE IT GUARDS. Both consumers
    forward-fill onto a daily grid, and `fundamentals_to_daily` fills each column over the
    rows that CARRY it -- so a flag pivoted from a wider row set than its value can present a
    LATER filing's flag against an EARLIER filing's growth. That is the desync that leaked 515
    cells past the phase-0 insider-ownership gate before it was found; here it would silently
    un-guard a transition. `panel._ceo_pay_growth` pivots both legs out of one subset for
    exactly this reason.

    ⚠ NaN IS NOT "no change" -- see the module docstring. What each caller then DOES with an
    unknown is its own decision, and both are recorded: `pay_features._comp_history` nulls the
    growth, `panel._ceo_pay_growth` keeps it and counts it.
    """
    idents = ceo_identity_series(names)
    prev = idents.groupby(groups, sort=False).shift(1)
    return (idents != prev).astype("float64").where(idents.notna() & prev.notna())
