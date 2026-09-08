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

`accrual_dispersion` is the diagnostic that must be logged alongside: a spread of +/-1 in the
implied years is rounding (a birthday falls either side of the filing date), while a spread of
10 means two different PEOPLE are sharing one entity key -- which is a defect in the key, not
in the data, and is invisible unless it is measured.

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


def accrual_anchor(obs: pd.DataFrame, value: str, key: str = "pk",
                   date: str = "as_of") -> pd.Series:
    """Per-entity anchor YEAR implied by every observation of an accruing quantity.

    Returns a Series indexed by the entity key. Recompute the quantity at any date `d` as
    `year(d) - anchor[key]`. Entities with no usable observation are simply absent from the
    index, so a `.map` over it yields NaN -- unknown stays unknown.
    """
    if obs is None or obs.empty or value not in obs.columns:
        return pd.Series(dtype="float64")
    df = _implied_year(obs, value, key, date)
    if df.empty:
        return pd.Series(dtype="float64")
    return df.groupby("key")["implied"].median()


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
