"""
accrual.py  (src/data_aggregate/utils/governance/accrual.py)
-----------------------------------------------------------
ANCHOR-AND-ACCRUE for quantities that are CLOCKS, not levels: an age, a tenure, a
"years since". Interpolating one of these re-derives by regression a number that
arithmetic already determines, and it is measurably the wrong instrument.

    anchor_i = year(date_i) - value_i        # the implied birth year / start year
    anchor   = median(anchor_i) per entity   # consensus over the entity's own history
    value(d) = year(d) - anchor              # recompute at ANY date

Two things an interpolation cannot do, and this can:

  * it fills EDGE gaps. `interpolate(limit_area="inside")` refuses the leading and trailing
    gaps by design, which is right for a level and wrong for a clock -- an age before the
    first disclosure is not unknown, it is first_age - elapsed_years.
  * it is immune to per-filing re-derivation noise. Measured 2026-09-07 on `def14a_directors`
    (the largest sample of the same quantities available):

        | quantity                            | median slope/yr | within 0.75-1.25 |
        | age (92,086 person-pairs)           | 1.00            | 97.0%            |
        | tenure_years (91,330 person-pairs)  | 1.00            | 73.3%            |

    Age accrues exactly, so its anchor is near-exact. Tenure's spread (p10 0.00, p90 1.98)
    is NOT the person changing -- it is the extraction re-reading "director since YYYY"
    inconsistently from one proxy to the next. A median anchor over the whole history is
    precisely what averages that away; an interpolation between two noisy neighbours
    propagates it.

MEDIAN, not first or last: one mis-extracted age shifts every value on the series if the
anchor is taken from a single observation, and shifts nothing if it is taken from the median.

`accrual_dispersion` is the diagnostic, and since 2026-09-09 it also GATES: a spread of
+/-1 in the implied years is rounding (a birthday falls either side of the filing date),
while a spread of 10 means two different PEOPLE are sharing one entity key -- a defect in
the KEY, not in the data. It was measured and logged for two days before anything acted
on it, during which the median blended a father and a son into one impossible person
rather than declining to answer. See `ANCHOR_MAX_OUTLIER_SHARE`.

⚠ The entity key is the caller's business and it must not be a bare ticker. `ceo_age` is
anchored per (ticker, CEO identity): anchoring it per ticker is the D33 defect in a different
costume -- it would blend an outgoing and an incoming CEO into one impossible person.

Written generically (it takes a key column) so that the day board metrics are rebuilt per
director -- D29 says not today -- the same helper serves `age` and `tenure_years` with no
change.
"""
from __future__ import annotations

import pandas as pd


def _implied_year(obs: pd.DataFrame, value: str, key: str, date: str) -> pd.DataFrame:
    """(key, implied_year) for every observation that carries both a date and a value."""
    df = pd.DataFrame({
        "key": obs[key],
        "year": pd.to_datetime(obs[date], errors="coerce").dt.year,
        "value": pd.to_numeric(obs[value], errors="coerce"),
    }).dropna(subset=["key", "year", "value"])
    df["implied"] = df["year"] - df["value"]
    return df


#: How far ONE observation's implied year may sit from its entity's median before it is treated
#: as a mis-extraction and dropped from the anchor. +/-1 is a birthday falling either side of the
#: filing date, and two such roundings in one series reach 2.
ANCHOR_OUTLIER_YEARS = 2.0

#: What SHARE of an entity's observations may be dropped as outliers before the entity itself is
#: refused an anchor.
#:
#: ⚠ THIS IS THE STATISTIC THAT SEPARATES THE TWO CAUSES, and picking the wrong one costs real
#: fills. The obvious gate -- refuse when max(implied) - min(implied) is wide -- is destroyed by
#: a SINGLE bad row, which is precisely what a median anchor is designed to survive:
#:
#:     SYY  Ali Dibadj    2022:47, 2023:48, **2024:76**, 2025:50   -> implied 1975,1975,1948,1975
#:
#: One mis-extraction, spread 27, median dead right. A max-min gate refuses that series and loses
#: three good fills. An OUTLIER-SHARE gate drops the single 1948 and keeps the anchor, because
#: 1 in 4 is noise. What it refuses instead is the case where the observations fall into two
#: CLUSTERS, i.e. two people sharing one key -- overwhelmingly a father and a son:
#:
#:     PHM  William J. Pulte   spread 56  (the founder 1997-2008, then the grandson)
#:     BRO  brown|j            spread 30  (J. Hyatt Brown, then J. Powell Brown)
#:     WRB  berkley|w          spread 27  (William R. Berkley, then W. Robert Berkley Jr.)
#:
#: There a large fraction of the rows sits far from the median whichever cluster wins, so the
#: share crosses the bound and the entity is refused. That is the D33 defect on a new grain, and
#: it is exactly what `directors.py`'s header claims a per-person anchor makes "impossible by
#: construction" -- which it is not, when two people share a name.
#:
#: HOW IT WAS FOUND. `accrual_dispersion` was computed, logged as "the key-collision alarm", and
#: gated NOTHING. `18_fill_is_past_only` refits the anchor on rows at or before each cut date and
#: compares: 1,164 accrued cells moved by more than a year, the worst by **30 years**, because a
#: median over two people is not an estimate of either.
#:
#: COST, measured rather than assumed: 127 of 17,598 director series (0.72%) and 11 of 1,579 CEO
#: series (0.70%) have a spread wide enough to be candidates. Only ACCRUED cells are lost; every
#: disclosed age is untouched, because this module never overwrites a filed value.
ANCHOR_MAX_OUTLIER_SHARE = 0.25


def accrual_anchor(obs: pd.DataFrame, value: str, key: str = "pk", date: str = "as_of",
                   outlier_years: float | None = ANCHOR_OUTLIER_YEARS,
                   max_outlier_share: float = ANCHOR_MAX_OUTLIER_SHARE) -> pd.Series:
    """Per-entity anchor YEAR implied by every observation of an accruing quantity.

    Returns a Series indexed by the entity key. Recompute the quantity at any date `d` as
    `year(d) - anchor[key]`. Entities with no usable observation are simply absent from the
    index, so a `.map` over it yields NaN -- unknown stays unknown.

    TWO ROBUSTNESS STEPS, and they do different jobs:

      1. observations further than `outlier_years` from the entity's median are DROPPED, so one
         mis-extracted age cannot drag the anchor -- the median already resisted it, and this
         also stops it perturbing the anchor when refitted on a shorter history;
      2. an entity that loses more than `max_outlier_share` of its observations that way is
         REFUSED an anchor entirely, by the same mechanism as an unobserved one: it drops out of
         the index and every value it would have filled stays NaN. A large outlier share means
         the observations form two clusters, i.e. two PEOPLE share the key, and a median over
         two people is a number belonging to neither. Refusing to answer is the correct output.

    Pass `outlier_years=None` to measure the ungated behaviour, never to ship it.
    """
    if obs is None or obs.empty or value not in obs.columns:
        return pd.Series(dtype="float64")
    df = _implied_year(obs, value, key, date)
    if df.empty:
        return pd.Series(dtype="float64")
    if outlier_years is None:
        return df.groupby("key")["implied"].median()
    med = df.groupby("key")["implied"].median()
    outlier = (df["implied"] - df["key"].map(med)).abs() > outlier_years
    share = outlier.groupby(df["key"]).mean()
    anchor = df[~outlier].groupby("key")["implied"].median()
    return anchor[share.reindex(anchor.index) <= max_outlier_share]


def accrual_dispersion(obs: pd.DataFrame, value: str, key: str = "pk",
                       date: str = "as_of") -> pd.Series:
    """Per-entity spread (max - min) of the implied anchor years — the key-collision alarm.

    Indexed like `accrual_anchor`. Entities observed once have a dispersion of 0.0, which is
    an absence of evidence rather than agreement; count them separately when reporting.
    """
    if obs is None or obs.empty or value not in obs.columns:
        return pd.Series(dtype="float64")
    df = _implied_year(obs, value, key, date)
    if df.empty:
        return pd.Series(dtype="float64")
    g = df.groupby("key")["implied"]
    return g.max() - g.min()


def accrue(dates: pd.Series, keys: pd.Series, anchor: pd.Series) -> pd.Series:
    """`year(date) - anchor[key]`, aligned to `dates`' index. NaN where the key has no anchor."""
    years = pd.to_datetime(dates, errors="coerce").dt.year
    return years - keys.map(anchor)
