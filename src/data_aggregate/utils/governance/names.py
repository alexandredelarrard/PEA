"""
names.py  (src/data_aggregate/utils/governance/names.py)
--------------------------------------------------------------------------------
The governance-facing layer over `src/utils/names.py`: everything that is specific to
comparing the PERSON in a `def14a_llm.ceo_name_proxy` cell, and nothing that is not.

The key itself is deliberately NOT redefined here. `person_key` is shared with the extraction
that wrote `sec_8k_votes`' role-category columns, so this module only decides what STRING to
hand it -- which for a CEO cell means splitting a co-CEO listing first.

Used at exactly five places, and nowhere else a name is compared:
  * the CEO-turnover guard, comparing a filing's identity against the previous filing's;
  * `def14a_impute`'s identity gap fill, comparing the value before a gap against the one after;
  * the CEO<->NEO cross-check against `def14a_executive_comp.name`;
  * the election-nominee lookup against `sec_8k_votes.nominee_votes_json[].name` (diagnostic);
  * `ceo_stint_ids`, the CEO stints `def14a_impute` completes the start year and founder flag over.

⚠ `None` means UNKNOWN, never a value. Two `None`s must never compare equal in a turnover or
agreement test, because "we do not know who ran this company either year" is not evidence that
the CEO did not change -- and reading it as evidence is exactly how a real transition slips
past the guard. Every caller compares with an explicit not-None precondition.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils.names import clean_person_name, person_key

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


def key_at_last_value(keys: pd.Series, values: pd.Series, groups: pd.Series) -> pd.Series:
    """The key of the last row of each group, at or before each row, whose `values` is present
    (the CEO identity or stint that SOURCED a carried value). The caller sorts within `groups`."""
    return keys.where(values.notna()).groupby(groups, sort=False).ffill()


#: Leading honorifics; `person_key` keeps them as the first initial (`Dr. Sehat Sutardja` -> `sutardja|d`).
_HONORIFIC_RE = re.compile(r"^\s*(?:(?:dr|mr|mrs|ms|miss|mx|prof|professor|sir|dame|hon|rev)\.?\s+)+", re.I)
_GIVEN_DROP = frozenset({"jr", "sr", "ii", "iii", "iv"})


def strip_honorific(name: str) -> str:
    """`name` without its leading honorifics (Dr., Mr., Ms., Prof., Sir, ...)."""
    return _HONORIFIC_RE.sub("", name)


def given_names(value: object) -> tuple[str, str] | None:
    """`(first name, initials of the given names)` of the first person named in a cell, honorifics,
    post-nominals and generational suffixes dropped: `J.S. Tisch` -> `("j", "js")`. None when the
    cell names no given name. Tells apart relatives who share one `person_key`."""
    names = split_co_names(value)
    cleaned = clean_person_name(strip_honorific(names[0])) if names else None
    if not cleaned:
        return None
    tokens = [t for t in re.sub(r"[^a-z]+", " ", cleaned.split(",")[0].lower()).split() if t not in _GIVEN_DROP]
    given = tokens[:-1]
    if not given:
        return None
    return given[0], "".join(t[0] for t in given)


#: A generational suffix on a person cell. `person_key` strips it, so father and son
#: (`William R. Berkley` / `W. Robert Berkley, Jr.`) share one key; the stint reads it back.
_GENERATIONAL_RE = re.compile(r"(?:^|[\s,])(jr|sr|ii|iii|iv)\.?(?=$|[\s,])", re.I)


def generational_suffix(value: object) -> str:
    """The generational suffix of the first person named in a cell (`jr`, `iii`, ...), or ""."""
    names = split_co_names(value)
    if not names:
        return ""
    found = _GENERATIONAL_RE.findall(names[0])
    return found[-1].lower() if found else ""


#: the two proxies around a one-off foreign proxy are at most this far apart (one annual cycle)
_OFF_CYCLE_DAYS = 400


def _stint_person(value: object) -> tuple[str, str, str | None] | None:
    """`(key, generational suffix, first name)` of the first CEO named, honorifics stripped. None
    when no first name or initial is left (`Mr. Cherkasky`): a bare surname is an unknown CEO."""
    names = split_co_names(value)
    key = person_key(strip_honorific(names[0])) if names else None
    if key is None or "|" not in key:
        return None
    given = given_names(value)
    return key, generational_suffix(value), given[0] if given else None


def ceo_stint_ids(names: pd.Series, tickers: pd.Series, dates: pd.Series | None = None) -> pd.Series:
    """A stint number per row: consecutive rows of one ticker naming the same CEO. NaN where the
    CEO is UNKNOWN; such a row breaks no stint.

    The CEO is the honorific-stripped `person_key` of the first person named. A generational
    suffix change splits the stint only with a change of first name (`William R. Berkley` /
    `W. Robert Berkley, Jr.`). With `dates`, a single off-cycle proxy naming a CEO seen nowhere
    else in the ticker, between two proxies of one CEO, belongs to no stint (another company's
    proxy filed under the ticker). The caller sorts chronologically within `tickers`.
    """
    parts = names.astype(object).map(_stint_person)
    known = parts.notna()
    p = pd.DataFrame(parts[known].tolist(), index=parts.index[known], columns=["key", "suffix", "first"])
    tick = tickers[known].astype(str)
    if dates is not None and len(p):
        g = p["key"].groupby(tick, sort=False)
        t = pd.Series(pd.to_datetime(dates[known], errors="coerce"), index=p.index)
        span = (t.groupby(tick, sort=False).shift(-1) - t.groupby(tick, sort=False).shift(1)).dt.days
        once = p["key"].groupby([tick, p["key"]], sort=False).transform("size") == 1
        foreign = once & (g.shift(1) == g.shift(-1)) & (p["key"] != g.shift(1)) & (span <= _OFF_CYCLE_DAYS)
        p, tick = p.loc[~foreign], tick.loc[~foreign]
    prev = p.groupby(tick, sort=False).shift(1)
    renamed = (p["suffix"] != prev["suffix"]) & p["first"].notna() & prev["first"].notna() & (p["first"] != prev["first"])
    starts = (tick != tick.shift(1)) | (p["key"] != prev["key"]) | renamed
    stint = pd.Series(np.nan, index=names.index, dtype="float64")
    stint[p.index] = starts.cumsum().astype("float64")
    return stint
