"""
directors.py  (src/data_aggregate/utils/governance/directors.py)
-----------------------------------------------------------------
The PER-DIRECTOR grain of the proxy archive: `def14a_directors`, 134,490 rows over 488
tickers and 12,239 filings, and until this module read it, unread by the whole cube.

It does three jobs, and the first one is a REPAIR rather than a feature.

1. **The board averages become evidence.** `avg_other_public_boards` and `avg_director_age`
   were reaching the features through `impute_def14a`'s linear interpolation. Here they are
   DERIVED from the director rows they describe, after a per-person fill of the child table,
   with the filer's own scalar and then interpolation as the fallbacks (D35, D39).
2. **Board quality** — what a board AVERAGE structurally cannot express: turnover,
   entrenchment, dispersion, and the ISS-style overboarded share (D40, D41).
3. It leaves `pct_female_directors`, `pct_independent_directors`, `avg_board_tenure` and
   `board_size` ALONE. All four are live D3-protected features and re-deriving them would move
   cells the plan forbids moving (D36).

WHY THE RE-AGGREGATION IS NOT A NO-OP, which is the one thing to understand before reading the
code. Measured 2026-09-07, the parent scalar ALREADY IS the mean of its own children --
`avg_other_public_boards` derivable on 5,656 filings against 5,656 filed, correlation 1.000,
median absolute difference 0.00. The extraction computes the parent from the child rows, so a
plain re-aggregation reproduces it exactly and gains nothing. The gain comes from filling the
CHILD table first: `other_public_company_boards` is present on only 29.3% of director rows, and
the median filing carrying any of it covers just 67% of its own board.

THE NUMBER THAT JUSTIFIES THE MODULE. `board_busyness_delta_1y` (phase 5) rejects any pair with
an interpolated leg, and on the interpolated basis that was **65.5%** of pairs -- the feature was
substantially a measurement of the fill. Re-based on the derivation it is ~22.5%, and the clean
population multiplies by 2.25x. That is the whole case, and it does NOT depend on gaining a
single extra filing: on `avg_other_public_boards` the derivation reaches 8,673 filings, a SUBSET
of the 10,484 interpolation already reached. `avg_director_age` is the one that also pays a
coverage dividend (+250 filings neither filed nor interpolable).

⚠ THE PERSON KEY IS `(ticker, name)` AS FILED, and that is a MEASURED choice, not a default.
`person_key` buckets `lastname|firstinitial`, and measured on this table it collides on **261 of
12,239 boards** (263 rows) with people who are demonstrably different humans:

    ADM  1995   andreas|m   ['Martin L. Andreas', 'Michael D. Andreas']
    AON  2002   ryan|p      ['Patrick G. Ryan', 'Patrick G. Ryan, Jr.']
    AXON 2002-06 smith|p    ['Phillips W. Smith', 'Patrick W. Smith']

Merging those blends two people's directorship counts and two people's ages, which is the D33
defect on a new grain. So the ATTRIBUTE FILL keys on the filed name and accepts the opposite
cost: a respelling (`Daniel Rosensweig` / `Daniel L. Rosensweig`) reads as two series, whose
interior gap therefore becomes an edge and simply does not fill. Conservative in the direction
that cannot fabricate.

⚠ `board_turnover` GOES THE OTHER WAY, and for the same reason phase 1 keyed CEO turnover.
There the failure is inverted: under the filed name a respelling reads as one departure AND one
arrival, inflating turnover at both ends. Measured over 11,751 consecutive-filing pairs:

    | key        | median | mean   | p90    | arrivals | departures |
    |------------|--------|--------|--------|----------|------------|
    | filed name | 10.00% | 14.18% | 30.77% | 19,098   | 18,800     |
    | person_key |  8.33% | 10.47% | 25.00% | 14,290   | 13,981     |

The filed name reads HIGHER on 2,074 pairs and never lower, mean gap 3.71pp -- i.e. roughly a
quarter of all measured churn is spelling. So turnover keys on `person_key`, and pays the 261
colliding boards, where the roster SET loses one seat and both sides of the comparison lose it
together. Each half of the module picks the error that does not invent the quantity it measures;
they differ because the two quantities fail in opposite directions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.governance.def14a_impute import CARRY_MAX_DAYS
from src.data_aggregate.utils.governance.accrual import (
    accrual_anchor, accrual_dispersion, accrue,
)
from src.data_aggregate.utils.governance.staleness import (
    LEVEL_MAX_AGE_DAYS, expire_event_fields, expire_level_fields,
)
from src.utils.names import person_key

#: ⚠ The COLUMN PROJECTION for `def14a_directors` lives in `utils/common/sources.py`, which is
#: the single registry every cube step reads and which `test_cube_incremental` asserts against.
#: It is not restated here: two copies of a projection is exactly how a builder ends up needing
#: a column the read no longer fetches.

#: The child column filled by a BOUNDED FORWARD CARRY: the last value this director disclosed,
#: carried into a gap and refused once it is older than `CARRY_MAX_DAYS` (D37, revised).
#:
#: ⚠ THIS WAS AN AGREEMENT GATE UNTIL 2026-09-09 -- fill only when the same director reports the
#: SAME count either side of the gap -- and its argument was made before point-in-time was a
#: constraint. Both halves of that rule read a LATER filing: `bwd` supplied the value, and
#: `fwd.notna() & bwd.notna()` supplied the DECISION that the gap was worth filling. It is the
#: same defect `def14a_impute` closed one grain up on the same day, reached by the other path, so
#: the feature it feeds -- `avg_other_public_boards` -> `f_board_busyness` and
#: `f_board_busyness_delta_1y` -- was repaired at the parent and left broken at the child.
#:
#: THE MEASURED CASE, live table 2026-09-09 (135,162 rows, 20,015 people, 489 tickers, column
#: 29.5% filled):
#:
#:     | rule                            | PIT | cells filled |
#:     |---------------------------------|-----|--------------|
#:     | agreement gate (what was here)  | no  | 8,213        |
#:     | bounded forward carry (this)    | yes | 21,488       |
#:     | no fill at all                  | yes | 0            |
#:
#: The carry also gains **10,147 TRAILING cells** the old rule could not reach at any price: a gap
#: at the live edge has no "after", so a backtest filled situations a live run structurally
#: cannot. And it REFUSES 13,786 candidates as more than 1,095 days stale -- which the agreement
#: gate never did, because it asked only whether the two sides matched and never how far apart
#: they were. Measured on a synthetic 2010->2020 gap, the old rule wrote a value sourced 3,287
#: days (9.0 years) back and `expire_stale` then dated it to the row it landed on.
#:
#: ⚠ THE "46% WRONG" FIGURE IS A STALENESS STATISTIC AND WAS READ WRONGLY ONCE ALREADY, in this
#: docstring's own predecessor ("8,160 agree and 8,128 are declined ... the looser rules get
#: there by inventing the half that disagrees"). What it measures is how often a carried value
#: differs from the NEXT disclosure:
#:
#:     | carry age (days) | scorable | differs |    % |
#:     |------------------|---------:|--------:|-----:|
#:     | 0-200            |       70 |      19 | 27.1 |
#:     | 200-400          |    6,894 |   2,944 | 42.7 |
#:     | 400-600          |       61 |      26 | 42.6 |
#:     | 600-800          |    3,301 |   1,648 | 49.9 |
#:     | 800-1,095        |    1,008 |     530 | 52.6 |
#:     | 1,095-2,000      |    3,074 |   1,743 | 56.7 |
#:     | >2,000           |    1,966 |   1,257 | 63.9 |
#:     | **all**          |   16,381 |   8,168 | 49.9 |
#:
#: The same statistic on the PARENT board average is **91.5%**, and on prices it would be ~100%.
#: Every forward-fill in this repo differs from the next observation most of the time; that is
#: what a forward-fill IS. It is evidence that the carry is STALE, not that it is fabricated, and
#: the answer to staleness is the cap plus the `<col>_imputed` flag -- both of which are here --
#: not a rule that buys accuracy with the future. A tighter cap does not rescue the old argument
#: either: even over one missed annual cycle the value differs 42.7% of the time.
#:
#: ⚠ `ceo_name_proxy` IS A GENUINELY DIFFERENT CASE and stays in `def14a_impute.CARRY_FORBIDDEN`.
#: A wrong NAME corrupts an identity join and lets pay growth be computed straight across a CEO
#: succession; a stale COUNT is a stale number of exactly the kind the horizon already governs.
CARRY_GATED_CHILD: tuple[str, ...] = ("other_public_company_boards",)

#: The child column that is a CLOCK and therefore ANCHORED, not carried (D38, same instrument as
#: `ceo_age`). Measured on 92,086 person-pairs in this table, `age` accrues at a median slope of
#: exactly 1.00/yr and 97.0% of pairs fall in [0.75, 1.25] -- so a per-person median anchor is
#: near-exact, and it fills EDGE gaps that `limit_area="inside"` refuses. 81.4% -> 96.5%.
ACCRUED_CHILD: tuple[str, ...] = ("age",)

#: `tenure_years` is deliberately in NEITHER set. It accrues too (median slope 1.00/yr) but only
#: 73.3% of its person-pairs land in [0.75, 1.25]: the spread is the extraction re-reading
#: "director since YYYY" inconsistently between proxies. An anchor WOULD average that away, and
#: the reason not to reach for it here is `board_tenure_dispersion` -- anchoring every director
#: to a consensus start year shrinks the dispersion by construction, which is the one thing
#: §3.4's guard forbids. Tenure is read exactly as filed.
_RAW_ONLY_CHILD: tuple[str, ...] = ("tenure_years",)

#: `pct_long_tenured`'s cutoff. 15 years is the entrenchment threshold in the governance
#: literature and in ISS's own tenure policy, and it is a THRESHOLD ON A DISCLOSED NUMBER rather
#: than a definition invented here -- unlike a cutoff on a board AVERAGE, which is what phase 5
#: refused for `board_busyness`.
_LONG_TENURE_YEARS = 15.0

#: `pct_overboarded`'s cutoff: FOUR public boards, i.e. three OTHER seats besides this one, is
#: ISS's over-commitment trigger for a non-executive director. The column counts OTHER boards,
#: so the comparison is `>= 4` other seats only if the column includes this one -- it does not.
#: ⚠ Measured on the live column, `other_public_company_boards` EXCLUDES the issuer (median 1.0,
#: max 9.0, and 0 is by far the modal value), so the ISS trigger is `>= 3` OTHER seats.
_OVERBOARDED_OTHER_SEATS = 3.0

#: A standard deviation over one observation is 0.0, which reads as "this board is perfectly
#: uniform" when it means "we know one director". Two is the floor for a dispersion to exist.
_MIN_FOR_DISPERSION = 2

#: The three-valued provenance of a merged board average (D39). `filed` = the filer's own scalar
#: stood; `derived` = the director rows overrode or supplied it; `interpolated` = neither could,
#: and `impute_def14a`'s temporal fill got there last.
SOURCE_FILED = "filed"
SOURCE_DERIVED = "derived"
SOURCE_INTERPOLATED = "interpolated"

#: The parent columns this module re-derives, and the child column each is the mean of.
#: ⚠ `pct_female_directors` is NOT here and must not be added (D36): gender is already 100%
#: filled at the child level and 99.2% at the parent, so there is no coverage to gain, and it is
#: one of the twelve D3-protected live features. It is also 77.6% inferred from an honorific,
#: 18.7% filer-stated, 1.9% from a name and 1.8% from a pronoun -- so a derived basis would not
#: even be unambiguously stronger evidence than the scalar it replaced.
DERIVED_AGGREGATES: dict[str, str] = {
    "avg_other_public_boards": "other_public_company_boards",
    "avg_director_age": "age",
}

#: Board-quality features (D40). All LEVELS except the first.
#:
#: ⚠ `board_turnover` IS EXPIRED and the other five are not, which DEVIATES from D40's blanket
#: "every one is a level, not an event". The five really are standing descriptions of the board
#: that filed -- a board does not stop having a tenure dispersion between proxies. Turnover is a
#: ONE-YEAR CHANGE, the same shape as `board_busyness_delta_1y` and `ceo_comp_growth_1y`, and
#: forward-filling "this board replaced 10% of its seats" for six years asserts churn nobody
#: disclosed. D21's argument applies to it verbatim; being listed beside five levels is not an
#: argument that it is one. The cost is near-zero -- proxies are annual, so the 548-day horizon
#: bites only where a whole cycle was missed.
BOARD_QUALITY_FIELDS: tuple[str, ...] = (
    "board_turnover", "pct_long_tenured", "board_tenure_dispersion",
    "board_age_dispersion", "oldest_director_age", "pct_overboarded",
)
EVENT_FIELDS: frozenset[str] = frozenset({"board_turnover"})

#: The five structural LEVELS, on `LEVEL_MAX_AGE_DAYS` (1,095 days) rather than on NO horizon.
#:
#: ⚠ "NOT AN EVENT" IS NOT THE SAME STATEMENT AS "NEVER EXPIRES", and until this set existed
#: these five made it so: correctly kept off the 548-day event clock, then forward-filled with
#: no cutoff at all, so one parsed proxy was asserted as current for as long as the trading
#: index ran. That is the identical reasoning error phase 3 fixed for the twelve fields in
#: `LEVEL_HORIZON_FIELDS`; these are the residue it did not reach.
#:
#: Measured on the live part before the change (aged against each ticker's most recent proxy,
#: which is a LOWER bound because it does not require that proxy to have carried this field):
#: 1,161-1,209 cells past 1,095 days per field = **0.04%**, max age 1,822 days (5.0 years).
#: So the fix is a contract-consistency one, not a data emergency -- `insider_ownership_pct`,
#: the field that motivated phase 3, was 18.89% and 29.5 years. Stated plainly because the
#: temptation with a 0.04% finding is to oversell it.
LEVEL_FIELDS: frozenset[str] = frozenset(BOARD_QUALITY_FIELDS) - EVENT_FIELDS

#: Every field this module can emit -- the exhaustiveness anchor phase 4 learned to need.
ALL_FIELDS: frozenset[str] = frozenset(BOARD_QUALITY_FIELDS)

#: ⚠ EMPTY, and measured rather than assumed. D41 anticipated "D2's peer machinery gives each its
#: own two columns"; that language predates phases 3-5, each of which refuted peer legs on
#: measurement. The precondition is that a peer norm EXISTS -- a between-sector variance share
#: comparable to the four surviving legacy legs (7.7%-12.4%) -- and not one of these six clears
#: it. Measured 2026-09-08:
#:
#:     board_age_dispersion 7.20%   board_tenure_dispersion 6.50%   pct_long_tenured 5.48%
#:     pct_overboarded 4.71%        oldest_director_age 3.81%       board_turnover 2.91%
#:
#: BOARD COMPOSITION IS A FIRM-LEVEL FACT. `board_turnover` at 2.91% is the clearest case: who
#: left your board last year has essentially nothing to do with your sector. `board_age_dispersion`
#: is the near miss at 7.20%, and the 7.7% floor belongs to `insider_ownership_pct`, which was
#: kept only for being the most SIGN-STABLE signal in the panel -- a dispersion has no such record.
#: `pct_overboarded` is additionally a bounded share whose ZERO (nobody overboarded) is the thesis,
#: the same argument that kept phase 5's counts raw.
#:
#: ⚠ The director-PAY family that landed with this phase did earn two legs on the identical
#: measure (`director_comp.PEER_RELATIVE_FIELDS`, 11.94% and 8.17%) -- so the yardstick is not
#: rigged against new families, it simply separates a consultant-benchmarked pay mix from a board
#: roster.
PEER_RELATIVE_FIELDS: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- #
# the child-grain fill (D37, D38)                                              #
# --------------------------------------------------------------------------- #
def _prepared(df: pd.DataFrame) -> pd.DataFrame | None:
    """`def14a_directors` dated, keyed and sorted into per-person chronological series.

    Sorting here rather than trusting the caller: every primitive below reads `ffill` / `bfill`
    as "the previous / next filing for this person", which is only true in date order.
    """
    need = {"ticker", "as_of", "name"}
    if df is None or df.empty or not need <= set(df.columns):
        return None
    out = df.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    out = out.dropna(subset=["ticker", "as_of", "name"])
    if out.empty:
        return None
    # The filed name, per ticker -- see the module header for why this is not `person_key`.
    out["_pk"] = out["ticker"].astype(str) + "|" + out["name"].astype(str)
    return out.sort_values(["_pk", "as_of"]).reset_index(drop=True)


def _carry_gated_fill(out: pd.DataFrame, col: str, stats: dict[str, int]) -> None:
    """Carry the last value this director DISCLOSED forward into their gaps, bounded and counted.

    Reads only rows at or before each row: that is a property of `ffill`, not of a guard someone
    has to remember, and it is what the `ffill()`/`bfill()` pair here until 2026-09-09 did not
    have. Writes `<col>_imputed` provenance (1.0 where this wrote the value), which is what lets
    `board_tenure_dispersion` and the delta features reconstruct the RAW column.

    ⚠ THE AGE IS MEASURED AGAINST THE `as_of` THAT SOURCED THE VALUE, never the previous row --
    the same payload trick `def14a_impute._carry` and `staleness.expire_stale` both use, so the
    fill and the clock agree on what "age" means. Without it a person filing every year for a
    decade and then falling silent would look freshly disclosed forever.

    Every reason a candidate is NOT filled gets its own counter, because a fill count alone
    cannot distinguish "nothing was missing" from "everything was refused". The two reasons are
    mutually exclusive: a cell with no prior disclosure has no age to test.
    """
    gk = out["_pk"]
    fwd = out.groupby(gk, sort=False)[col].ffill()
    src = out["as_of"].where(out[col].notna()).groupby(gk, sort=False).ffill()
    age = (out["as_of"] - src).dt.days

    gaps = out[col].isna()
    candidate = gaps & fwd.notna()
    within = age <= CARRY_MAX_DAYS
    newly = candidate & within

    out.loc[newly, col] = fwd[newly]
    out[f"{col}_imputed"] = newly.astype("float64")
    stats[f"child gaps: {col}"] = int(gaps.sum())
    stats[f"child carried: {col}"] = int(newly.sum())
    stats[f"child declined (>{CARRY_MAX_DAYS}d stale): {col}"] = int((candidate & ~within).sum())
    stats[f"child declined (no prior disclosure): {col}"] = int((gaps & fwd.isna()).sum())


def _accrue_child(out: pd.DataFrame, col: str, stats: dict[str, int]) -> None:
    """Fill `col` from a per-(ticker, director) median anchor year. Stamps `<col>_imputed`.

    The anchor also reaches EDGE gaps, which is the point: a director's age before their first
    disclosed one is not unknown, it is `first_age - elapsed_years`.

    `accrual_dispersion` is reported beside it because it is the key-collision alarm,
    and since 2026-09-09 `accrual_anchor` GATES on the same evidence: a spread of +/-1
    implied year is a birthday falling either side of a filing date, while
    `PHM|William J. Pulte` spans 56 years across two generations. Such a series is
    refused an anchor entirely rather than accrued to a median belonging to neither
    person. `child anchors REFUSED` counts them; the raw spread is still reported
    beside it, because the two disagree exactly where a SINGLE age is mis-extracted.
    """
    obs = pd.DataFrame({"pk": out["_pk"], "as_of": out["as_of"], col: out[col]})
    anchor = accrual_anchor(obs, col, key="pk", date="as_of")
    if anchor.empty:
        out[f"{col}_imputed"] = 0.0
        stats[f"child skipped (no anchor): {col}"] = 1
        return
    implied = accrue(out["as_of"], out["_pk"], anchor)
    newly = out[col].isna() & implied.notna()
    out.loc[newly, col] = implied[newly]
    out[f"{col}_imputed"] = newly.astype("float64")
    spread = accrual_dispersion(obs, col, key="pk", date="as_of")
    stats[f"child accrued: {col}"] = int(newly.sum())
    stats[f"child anchors: {col}"] = int(len(anchor))
    stats[f"child anchors REFUSED (two people share a key): {col}"] = int(
        len(set(obs['pk'].dropna()) - set(anchor.index)))
    stats[f"child anchor spread > 2y (raw alarm): {col}"] = int((spread > 2).sum())


def fill_director_attributes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-DIRECTOR fill of `def14a_directors`. Returns `(copy, stats)`; the raw table is never
    mutated, and a value is only ever written where it is currently NaN.

    Two rules and no others, matching `def14a_impute`'s taxonomy exactly:
      * `other_public_company_boards` -- bounded forward carry (D37, kind B), sharing
        `def14a_impute.CARRY_MAX_DAYS` so the two grains cannot drift apart;
      * `age` -- anchor-and-accrue (D38, a clock).

    ⚠ NO temporal fill of `tenure_years` and NO fill of `is_independent`. The second is a
    measured opportunity DELIBERATELY not taken: it is 75.0% present with 13,566 interior gaps,
    so the same machinery would apply -- but `pct_independent_directors` is one of the twelve
    D3-protected features and its derived basis would move live cells (§4).

    Every filled column carries `<col>_imputed`, always present even when nothing was filled, so
    the dispersion features can reconstruct the RAW column from it (§3.4's guard) and the
    provenance of a board average stays stateable rather than assumed.
    """
    out = _prepared(df)
    if out is None:
        return (df if df is not None else pd.DataFrame()), {}
    stats: dict[str, int] = {"child rows": len(out),
                             "child person-series": int(out["_pk"].nunique())}
    for col in CARRY_GATED_CHILD:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            _carry_gated_fill(out, col, stats)
    for col in ACCRUED_CHILD:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            _accrue_child(out, col, stats)
    for col in _RAW_ONLY_CHILD:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop(columns=["_pk"]).reset_index(drop=True), stats


# --------------------------------------------------------------------------- #
# the re-aggregation and the precedence chain (D35, D39)                       #
# --------------------------------------------------------------------------- #
def _raw_leg(frame: pd.DataFrame, col: str) -> pd.Series:
    """`col` with every cell this module's fill invented removed -- the column as FILED.

    Reconstructed from the `<col>_imputed` stamp rather than kept as a second copy, so the two
    cannot disagree. An absent stamp means nothing was filled, which is the right reading for a
    frame that never went through `fill_director_attributes`.
    """
    flag = f"{col}_imputed"
    if flag not in frame.columns:
        return frame[col]
    return frame[col].where(pd.to_numeric(frame[flag], errors="coerce").fillna(0.0) <= 0)


def board_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-filing board averages DERIVED from the director rows, with their DENOMINATORS.

    Keyed `(ticker, accession_number, as_of)` and carrying, for each field in
    `DERIVED_AGGREGATES`:

        <field>                      the mean over the directors who report it (filled basis)
        n_reporting_<field>          how many directors that was
        n_reporting_<field>_filed    how many reported it BEFORE the child fill
        n_directors_total            the board size the filing lists

    ⚠ The coverage columns are returned rather than recomputed downstream, because D39's
    precedence is a comparison BETWEEN two coverages and computing each in a different place is
    exactly how the two drift apart.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    need = {"ticker", "accession_number", "as_of"}
    if not need <= set(df.columns):
        return pd.DataFrame()
    d = df.copy()
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    d = d.dropna(subset=["ticker", "accession_number", "as_of"])
    if d.empty:
        return pd.DataFrame()
    keys = ["ticker", "accession_number", "as_of"]
    g = d.groupby(keys, sort=False)
    out = g.size().rename("n_directors_total").to_frame()
    for field, child in DERIVED_AGGREGATES.items():
        if child not in d.columns:
            continue
        filled = pd.to_numeric(d[child], errors="coerce")
        out[field] = filled.groupby([d[k] for k in keys], sort=False).mean()
        out[f"n_reporting_{field}"] = filled.notna().groupby(
            [d[k] for k in keys], sort=False).sum().astype("int64")
        out[f"n_reporting_{field}_filed"] = _raw_leg(d, child).notna().groupby(
            [d[k] for k in keys], sort=False).sum().astype("int64")
    return out.reset_index()


def merge_board_aggregates(parent: pd.DataFrame,
                           derived: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply D39's precedence to the parent proxy rows. Returns `(copy, stats)`.

    ```
    1. derived from def14a_directors   where its board coverage BEATS the filer's own
    2. the filed parent scalar         (the filer's own figure)
    3. derived from def14a_directors   where the parent is NULL
    4. interpolation                   left to impute_def14a, as the LAST resort
    ```

    ⚠ Steps 1 and 3 are the SAME computation at different precedence against step 2, and writing
    them as one branch is the bug this ordering exists to prevent -- a derived value must not
    displace a COMPLETE filer-stated one merely because it exists.

    WHAT "COVERAGE BEATS" MEANS, concretely. §1.1 measured that the parent scalar IS the mean of
    its own children, to the decimal, on every filing where both exist -- the extraction derives
    it from the same rows. So the filed scalar's own denominator is `n_reporting_<field>_filed`,
    and the derived value is strictly better evidence exactly when the child fill added a
    reporting director. Anything else would be comparing a number against itself.

    Measured 2026-09-07 the override population is 781 on `avg_other_public_boards` (median move
    0.133) and 3,260 on `avg_director_age` (median move 1.000 year).

    The stamp is THREE-VALUED, not a boolean: `filed` and `derived` are written here, and
    `interpolated` is written by `finalize_board_source` after `impute_def14a` has run. "Not
    filed" and "not evidence" are different facts, and phase 5's delta gate needs to tell them
    apart.
    """
    if parent is None or parent.empty:
        return parent, {}
    out = parent.copy()
    stats: dict[str, int] = {}
    fields = [f for f in DERIVED_AGGREGATES if f in out.columns]
    for f in fields:
        out[f"{f}_source"] = np.where(out[f].notna(), SOURCE_FILED, None)
    if derived is None or derived.empty or "accession_number" not in out.columns:
        stats["skipped: no derived board aggregates (directors table absent)"] = 1
        return out, stats

    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    cols = ["ticker", "accession_number"]
    for f in fields:
        take = [c for c in (f, f"n_reporting_{f}", f"n_reporting_{f}_filed") if c in derived.columns]
        if f not in take:
            continue
        d = derived[cols + take].rename(columns={f: f"_d_{f}"})
        out = out.merge(d, on=cols, how="left")
        cand = pd.to_numeric(out[f"_d_{f}"], errors="coerce")
        n_all = pd.to_numeric(out[f"n_reporting_{f}"], errors="coerce")
        n_filed = pd.to_numeric(out[f"n_reporting_{f}_filed"], errors="coerce")
        better = cand.notna() & n_all.notna() & n_filed.notna() & (n_all > n_filed)

        override = out[f].notna() & better                    # step 1
        supply = out[f].isna() & cand.notna()                 # step 3
        moved = (cand[override] - out.loc[override, f]).abs()
        out.loc[override | supply, f] = cand[override | supply]
        out.loc[override | supply, f"{f}_source"] = SOURCE_DERIVED
        stats[f"{f}: filed and kept"] = int((out[f"{f}_source"] == SOURCE_FILED).sum())
        stats[f"{f}: derived OVERRODE a filed value"] = int(override.sum())
        stats[f"{f}: derived where the parent was NULL"] = int(supply.sum())
        stats[f"{f}: median |move| of an override (x1000)"] = (
            int(round(float(moved.median()) * 1000)) if len(moved) else 0)
        stats[f"{f}: still NULL -> left to interpolation"] = int(out[f].isna().sum())
        out = out.drop(columns=[c for c in out.columns
                               if c.startswith("_d_") or c.startswith("n_reporting_")])
    return out, stats


def finalize_board_source(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Complete D39's three-valued stamp AFTER `impute_def14a`: anything now non-null whose
    source is still unset got there by interpolation -- step 4 of the chain.

    Split out rather than folded into the merge because it can only be known once the last
    fallback has run, and inferring it earlier would label a cell by what was ABOUT to happen.
    """
    if df is None or df.empty:
        return df, {}
    out = df.copy()
    stats: dict[str, int] = {}
    for f in DERIVED_AGGREGATES:
        col = f"{f}_source"
        if f not in out.columns or col not in out.columns:
            continue
        late = out[f].notna() & out[col].isna()
        out.loc[late, col] = SOURCE_INTERPOLATED
        for label in (SOURCE_FILED, SOURCE_DERIVED, SOURCE_INTERPOLATED):
            stats[f"{f} source={label}"] = int((out[col] == label).sum())
        stats[f"{f} source=none (still NULL)"] = int(out[f].isna().sum())
    return out, stats


# --------------------------------------------------------------------------- #
# board quality (D40, D41)                                                     #
# --------------------------------------------------------------------------- #
def _per_filing(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """`(dated copy, group keys)` for the per-filing aggregations."""
    out = d.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    out = out.dropna(subset=["ticker", "as_of"])
    return out, ["ticker", "accession_number", "as_of"]


def _share_of_reporting(d: pd.DataFrame, keys: list[str], col: str,
                        threshold: float) -> pd.DataFrame | None:
    """`>= threshold` as a share of the directors who REPORT `col` -- never of board size.

    ⚠ The denominator is the whole point. `other_public_company_boards` is reported by 35.4% of
    director rows after the fill, so dividing by board size would report a 10-person board with
    4 reporting directors, 2 of them overboarded, as 20% overboarded rather than 50% -- a
    silent multiplication by the reporting rate.
    """
    if col not in d.columns:
        return None
    v = pd.to_numeric(d[col], errors="coerce")
    g = [d[k] for k in keys]
    n = v.notna().groupby(g, sort=False).sum()
    # ⚠ `astype("float64")` on the indicator: `(v >= t).where(v.notna())` is OBJECT dtype (True /
    # False / NaN mixed), and an object-dtype daily frame silently breaks arithmetic downstream --
    # `peer_relative`'s `min_count=1` sum returns None and the division raises `TypeError:
    # unsupported operand type(s) for /: 'NoneType' and 'float'`. Found by measuring, not by a
    # crash in production, because this family emits no peer leg today.
    hits = (v >= threshold).where(v.notna()).astype("float64").groupby(g, sort=False).sum()
    out = (hits / n.where(n > 0)).astype("float64").rename("value").reset_index()
    out.columns = [*keys, "value"]
    return out


def _dispersion(d: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame | None:
    """Sample std of `col` over a board, on the RAW (unfilled) column, `NaN` below two reports.

    ⚠ THE RAW COLUMN, and this is §3.4's guard rather than a preference. An anchor or a
    carry-forward shrinks variance by construction, so a dispersion computed on the filled
    column measures the smoothness of the FILL. `test_governance_board_quality` asserts that
    filling the child table moves these two features by exactly zero.
    """
    if col not in d.columns:
        return None
    v = _raw_leg(d, col).astype("float64")
    g = [d[k] for k in keys]
    n = v.notna().groupby(g, sort=False).sum()
    sd = v.groupby(g, sort=False).std()
    out = sd.where(n >= _MIN_FOR_DISPERSION).rename("value").reset_index()
    out.columns = [*keys, "value"]
    return out


def _turnover(d: pd.DataFrame, keys: list[str], tally: dict[str, int]) -> pd.DataFrame | None:
    """`|names(t) \\ names(t-1)| / |names(t)|` on consecutive filings, stamped on the LATER one.

    Keyed on `person_key`, against the filed name everywhere else in this module -- see the
    module header for the measurement that splits the two decisions. The raw-name count is
    tallied beside it so the spelling component stays visible at build time rather than only in
    a plan document.
    """
    if "name" not in d.columns:
        return None
    work = d[[*keys, "name"]].copy()
    work["pk"] = work["name"].astype(object).map(person_key)
    rows: list[dict] = []
    for key_col, label in (("pk", "person_key"), ("name", "filed name")):
        rosters = (work.dropna(subset=[key_col])
                   .groupby(keys, sort=False)[key_col].agg(frozenset)
                   .rename("roster").reset_index().sort_values(["ticker", "as_of"]))
        prev = rosters.groupby("ticker", sort=False)["roster"].shift(1)
        paired = prev.notna() & rosters["roster"].map(bool)
        arrivals = [len(c - p) if isinstance(p, frozenset) else np.nan
                    for c, p in zip(rosters["roster"], prev)]
        size = rosters["roster"].map(len)
        rate = pd.Series(arrivals, index=rosters.index) / size.where(size > 0)
        tally[f"board_turnover pairs ({label})"] = int(paired.sum())
        tally[f"board_turnover mean x1000 ({label})"] = int(
            round(float(rate[paired].mean()) * 1000)) if bool(paired.any()) else 0
        if key_col == "pk":
            rows = rosters.loc[paired, keys].assign(value=rate[paired]).to_dict("records")
    return pd.DataFrame(rows) if rows else None


def board_quality_fields(directors: pd.DataFrame, idx: pd.DatetimeIndex,
                         ) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """The six board-quality features (D40, D41) as daily wide frames, plus the tallies.

    Takes the FILLED child frame: levels read the filled column, dispersions reconstruct the
    filed one from the provenance stamp, so both bases come out of one input.

    ⚠ `pct_overboarded` ships BESIDE `avg_other_public_boards`, not instead of it (D41). It is
    the definition the literature and ISS policy actually use, where a board mean is the weaker
    proxy for it, and neither is a composite of the other -- the model selects, the plan does
    not pre-judge. Phase 5's `board_busyness` family is NOT reworked around it.
    """
    tally: dict[str, int] = {}
    if directors is None or directors.empty or "as_of" not in directors.columns:
        tally["skipped: no def14a_directors -> no board-quality family"] = 1
        return {}, tally
    d, keys = _per_filing(directors)
    if d.empty:
        tally["skipped: def14a_directors has no dated rows"] = 1
        return {}, tally

    per_filing: dict[str, pd.DataFrame | None] = {
        "board_turnover": _turnover(d, keys, tally),
        "pct_long_tenured": _share_of_reporting(d, keys, "tenure_years", _LONG_TENURE_YEARS),
        "pct_overboarded": _share_of_reporting(d, keys, "other_public_company_boards",
                                               _OVERBOARDED_OTHER_SEATS),
        "board_tenure_dispersion": _dispersion(d, keys, "tenure_years"),
        "board_age_dispersion": _dispersion(d, keys, "age"),
    }
    if "age" in d.columns:
        oldest = (pd.to_numeric(d["age"], errors="coerce")
                  .groupby([d[k] for k in keys], sort=False).max()
                  .rename("value").reset_index())
        oldest.columns = [*keys, "value"]
        per_filing["oldest_director_age"] = oldest

    frames: dict[str, pd.DataFrame] = {}
    hist: dict[str, pd.DataFrame] = {}
    for name, pf in per_filing.items():
        if pf is None or pf.empty or not pf["value"].notna().any():
            tally[f"skipped: no data for {name}"] = 1
            continue
        h = pf.rename(columns={"value": name})[["ticker", "as_of", name]]
        daily = fundamentals_to_daily(h, name, idx)
        if daily.empty or not daily.notna().any().any():
            tally[f"skipped: {name} empty on the daily grid"] = 1
            continue
        frames[name] = daily
        # EVERY field needs its history now, not just the event members: the five levels are
        # on the 1,095-day LEVEL horizon rather than on no horizon at all, and `_expire_family`
        # ages a cell against the `as_of` of the filing that produced it, which it can only
        # read from this frame.
        hist[name] = h
        tally[f"{name}: filings"] = int(pf["value"].notna().sum())

    if frames and hist:
        # Only the EVENT members need a history at all -- `expire_event_fields` looks a feature's
        # `as_of` up by column name, so one merged frame over just those is the whole input.
        merged_hist = None
        for h in hist.values():
            merged_hist = h if merged_hist is None else merged_hist.merge(
                h, on=["ticker", "as_of"], how="outer")
        capped, stats = expire_event_fields(frames, merged_hist, EVENT_FIELDS)
        for name, (expired, before) in stats.items():
            if expired:
                tally[f"expired >548d: {name}"] = expired
                tally[f"non-null before expiry: {name}"] = before
        capped, lvl_stats = expire_level_fields(capped, merged_hist, LEVEL_FIELDS)
        for name, (expired, before) in lvl_stats.items():
            if expired:
                tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {name}"] = expired
                tally[f"non-null before expiry: {name}"] = before
        frames = {k: v for k, v in capped.items() if not v.empty and v.notna().any().any()}

    undeclared = set(frames) - ALL_FIELDS
    if undeclared:
        raise AssertionError(f"board-quality fields not in ALL_FIELDS: {sorted(undeclared)}")
    return frames, tally
