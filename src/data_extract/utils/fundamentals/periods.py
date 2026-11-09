"""
periods.py (src/data_extract/utils/fundamentals/periods.py)
--------------------------------------------------------------------------------------
The period engine: turns one ticker's as-filed duration facts from `sec_fundamentals_facts` into
discrete quarters, trailing-twelve-month values and quarterly-grid instants, in memory only.

Every input is selected by its calendar WINDOW shape (`period_shape`), never by the filer's
`fiscal_period` label. Quarters are kept as reported or derived by the ladder `Q2 = YTD6 - Q1`,
`Q3 = YTD9 - YTD6`, `Q4 = FY - YTD9`, fallback `Q4 = FY - (Q1+Q2+Q3)`, each tagged with its `basis`.
The FY and YTD legs may come from different filings and concepts: a switch is allowed, recorded in
`concept_switch`, and gated by a scale test. A TTM is emitted only from four contiguous discrete
quarters, otherwise NULL (never a carried-forward annual). A refused window is simply not emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.fundamentals.kpi_catalogue import DEFAULT_CONFIG_DIR, FieldSpec
from src.utils.config import read_config


@dataclass(frozen=True)
class PeriodGuards:
    """The derivation guard thresholds, read from `configs.yml` (`data_extract.fundamentals_periods`)
    and injected so a known-truth test can state its own."""

    max_opposite_sign_ratio: float
    concept_switch_scale_max: float
    share_basis_max_ratio: float


def load_guards(config_dir: str | None = DEFAULT_CONFIG_DIR) -> PeriodGuards:
    """Read the guards from `configs.yml`, cached per resolved config directory (`resolve_config_dir`)."""
    return _guards_at(resolve_config_dir(config_dir))


@lru_cache(maxsize=4)
def _guards_at(config_dir: str) -> PeriodGuards:
    block = read_config(config_dir).data_extract.fundamentals_periods
    return PeriodGuards(
        max_opposite_sign_ratio=float(block.max_opposite_sign_q4_ratio),
        concept_switch_scale_max=float(block.q4_tag_mismatch_fy_max),
        share_basis_max_ratio=float(block.share_basis_max_ratio),
    )


# --------------------------------------------------------------------- period shapes ---

#: Day-count bands classifying a duration fact by SHAPE. Wide enough for 52/53-week quarters
#: (~112 days) and deliberately disjoint: a duration between bands is `other`, never rounded in.
_DURATION_BANDS: tuple[tuple[int, int, str], ...] = (
    (60, 120, "quarterly"),
    (150, 210, "ytd6"),
    (240, 300, "ytd9"),
    (330, 400, "annual"),
)
QUARTERLY, YTD6, YTD9, ANNUAL = (name for _, _, name in _DURATION_BANDS)
INSTANT = "instant"
OTHER_SHAPE = "other"


def period_shape(period_type: str, days: float | None) -> str:
    """The period's SHAPE: `instant`, one of the `_DURATION_BANDS` names, or `other`.

    `other` (a stub, or a missing day count) is a valid outcome and is carried as unusable.
    """
    if period_type == INSTANT:
        return INSTANT
    if days is None or pd.isna(days):
        return OTHER_SHAPE
    for low, high, name in _DURATION_BANDS:
        if low <= days <= high:
            return name
    return OTHER_SHAPE


# --------------------------------------------------------------------------- vocabulary ---

#: How a discrete quarter was obtained, carried on the value's own row so the Q4 footing check
#: can exclude `FY_MINUS_QUARTERS` rows (which foot by construction).
AS_REPORTED = "as_reported"
Q2_FROM_YTD6 = "ytd6_minus_q1"
Q3_FROM_YTD9 = "ytd9_minus_ytd6"
FY_MINUS_YTD9 = "fy_minus_ytd9"
FY_MINUS_QUARTERS = "fy_minus_q1q2q3"

#: TTM bases.
TTM_FOUR_QUARTERS = "sum_4q"
TTM_FOUR_QUARTER_MEAN = "mean_4q"
TTM_AS_REPORTED_ANNUAL = "as_reported_annual"

#: A trailing-twelve window is four quarters spanning the annual band, so 52/53-week years pass
#: and a year with a stub or missing quarter does not.
TTM_MIN_DAYS, TTM_MAX_DAYS = 330, 400
TTM_QUARTERS = 4

#: Two period ends this close are the SAME period tagged twice -- see `_latest_per_window`.
_SAME_PERIOD_DAYS = 7

_QUARTER_COLUMNS: tuple[str, ...] = (
    "ticker",
    "field",
    "period_start",
    "period_end",
    "period_days",
    "value",
    "basis",
    "known_from",
    "source_concept",
    "concept_switch",
    "fiscal_year",
    "fiscal_quarter",
)


# ------------------------------------------------------------------ selection helpers ---


def _inclusive_days(days):
    """Day counts counting BOTH endpoints, so abutting periods' day counts add up exactly
    for the share-day arithmetic (`period_days` is `end - start`, one short)."""
    return days + 1


_DAY = np.timedelta64(1, "D")


@dataclass(frozen=True)
class FieldFacts:
    """Duration facts as plain arrays, in their frame order: the period engine's working form.

    Built once per frame (`of`), so the replay can slice one field's visible facts by position (`take`).
    """

    duration_type: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    days: np.ndarray
    values: np.ndarray
    filed: np.ndarray
    concepts: np.ndarray

    @classmethod
    def of(cls, facts: pd.DataFrame) -> FieldFacts:
        """`facts`' engine columns; dates are coerced, so strings are never sorted lexicographically."""
        return cls(
            facts["duration_type"].to_numpy(dtype=object),
            pd.to_datetime(facts["period_start"]).to_numpy("datetime64[ns]"),
            pd.to_datetime(facts["period_end"]).to_numpy("datetime64[ns]"),
            facts["period_days"].to_numpy(dtype=float),
            facts["value"].to_numpy(dtype=float),
            pd.to_datetime(facts["filing_date"]).to_numpy("datetime64[ns]"),
            facts["source_concept"].to_numpy(dtype=object),
        )

    def take(self, positions: np.ndarray) -> FieldFacts:
        return FieldFacts(*(column[positions] for column in self._columns()))

    def _columns(self) -> tuple[np.ndarray, ...]:
        return (self.duration_type, self.starts, self.ends, self.days, self.values, self.filed, self.concepts)


def _sort_key(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(value, is_null)` lexsort keys putting NaN/NaT last, as `DataFrame.sort_values` does."""
    if values.dtype.kind == "M":
        null = np.isnat(values)
        return np.where(null, 0, values.view("i8")), null
    null = np.isnan(values)
    return np.where(null, 0.0, values), null


def _window_order(facts: FieldFacts) -> np.ndarray:
    """The stable sort by `(period_end, filing_date, period_days)` -- `_latest_per_window`'s contract: the row that
    sorts last in its window wins."""
    keys = [key for column in (facts.days, facts.filed, facts.ends) for key in _sort_key(column)]
    return np.lexsort(keys)


def _latest_per_window(ends: np.ndarray) -> np.ndarray:
    """One row per calendar window, the LATEST filing winning: positions into `ends`, sorted by `_window_order`.

    A window is identified by its END within `_SAME_PERIOD_DAYS` (filers nudge boundary days between filings), not
    by its exact (start, end): a row opens a new window when it ends more than `_SAME_PERIOD_DAYS` whole days after
    the previous row (a NaT gap never does). Latest-wins is point-in-time correct because the caller passes only
    facts with `filing_date <= as_of`.
    """
    if ends.size == 0:
        return np.empty(0, dtype=np.intp)
    gaps = ends[1:] - ends[:-1]
    known = ~np.isnat(gaps)
    opens = np.zeros(gaps.size, dtype=bool)
    opens[known] = gaps[known] // _DAY > _SAME_PERIOD_DAYS
    # Row i closes its window when row i + 1 opens the next one; the last row always closes.
    return np.flatnonzero(np.append(opens, True))


@dataclass(frozen=True)
class _Legs:
    """One shape's latest-per-window facts, in `_window_order`: bounds and values as arrays, and each row as the plain
    dict `_derived` reads."""

    starts: np.ndarray
    ends: np.ndarray
    values: np.ndarray
    days: np.ndarray
    filed: np.ndarray
    rows: list[dict]

    @classmethod
    def of(cls, ordered: FieldFacts, shape: str) -> _Legs:
        """The `shape` legs of facts already in `_window_order`."""
        legs = ordered.take(np.flatnonzero(ordered.duration_type == shape))
        legs = legs.take(_latest_per_window(legs.ends))
        rows = [
            {"value": v, "period_end": e, "period_days": d, "filing_date": f, "source_concept": c}
            for v, e, d, f, c in zip(legs.values.tolist(), legs.ends, legs.days.tolist(), legs.filed, legs.concepts, strict=True)
        ]
        return cls(legs.starts, legs.ends, legs.values, legs.days, legs.filed, rows)


#: Max gap between a nine-month cumulative's end and a fourth quarter's start for the two to be
#: contiguous pieces of one fiscal year (52/53-week filers drift by a day or two).
_CONTIGUOUS_DAYS = 4


def _ambiguous_durations(q_start: np.ndarray, q_value: np.ndarray, ends: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Per quarterly-shaped fact: is it really the whole fiscal year? Used only where no annual fact exists.

    True when a contiguous `ytd9` (given as `ends` / `values`) exists, is materially non-zero (>1% of the
    quarter), has the SAME sign, and is smaller in magnitude than the "quarter". The same-sign condition
    keeps a genuine loss quarter after a profitable nine months.
    """
    if ends.size == 0 or q_start.size == 0:
        return np.zeros(q_start.size, dtype=bool)
    gap = (q_start[:, None] - ends[None, :]) / _DAY
    # An opposite-sign cumulative means the year turned, not that the window is mislabelled.
    keep = (gap >= 0) & (gap <= _CONTIGUOUS_DAYS) & (values[None, :] * q_value[:, None] > 0)
    nine = np.where(keep, np.abs(values)[None, :], -np.inf).max(axis=1)
    quarter = np.abs(q_value)
    return ~np.isnat(q_start) & ~np.isnan(q_value) & keep.any(axis=1) & (nine > 0.01 * quarter) & (quarter > nine)


def _masquerading_quarters(facts: FieldFacts) -> np.ndarray:
    """Per fact: a quarterly-shaped fact whose value is the FULL YEAR (an annual figure tagged into a Q4 context).

    With an annual fact present, a quarter is flagged only when all hold: an annual fact ends within
    `_SAME_PERIOD_DAYS` of it, the values agree within 0.1%, and an interim cumulative inside that year
    is >1% of the annual (so a real Q4 that equals its year, after a zero nine months, survives).
    With no annual fact, `_ambiguous_durations` decides (refused, never reclassified).
    """
    shape = facts.duration_type
    quarter, annual, ytd9 = shape == QUARTERLY, shape == ANNUAL, shape == YTD9
    flagged = np.zeros(shape.size, dtype=bool)
    # `annual` may be empty: the no-annual-fact branch must still run.
    if not quarter.any() or not (annual.any() or ytd9.any()):
        return flagged
    q_start, q_end, q_value = facts.starts[quarter], facts.ends[quarter], facts.values[quarter]
    a_start, a_end, a_value = facts.starts[annual], facts.ends[annual], facts.values[annual]
    interim = ytd9 | (shape == YTD6)
    i_end, i_value = facts.ends[interim], facts.values[interim]
    # One row per quarter, one column per annual fact; NaN and NaT propagate exactly as a per-row scan's would.
    near = np.abs(a_end[None, :] - q_end[:, None]) <= np.timedelta64(_SAME_PERIOD_DAYS, "D")
    has_near = near.any(axis=1)
    drop = _ambiguous_durations(q_start, q_value, facts.ends[ytd9], facts.values[ytd9]) & ~has_near
    if a_end.size:
        scale = np.where(near, np.abs(a_value)[None, :], -np.inf).max(axis=1)
        closest = np.where(near, np.abs(a_value[None, :] - q_value[:, None]), np.inf).min(axis=1)
        for i in np.flatnonzero(has_near & ~((scale < 1) | (closest > 0.001 * scale))):
            # Scoped to the annual window: the nine-month cumulative ends where the quarter begins.
            year_start = a_start[near[i]].min()
            accumulated = (i_end > year_start) & (i_end <= q_end[i])
            drop[i] = bool((np.abs(i_value[accumulated]) > 0.01 * scale[i]).any())
    flagged[np.flatnonzero(quarter)[drop]] = True
    return flagged


def _window_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`frame`'s `(period_start, period_end)` as nanosecond arrays, taken once per candidate frame."""
    return frame["period_start"].to_numpy("datetime64[ns]"), frame["period_end"].to_numpy("datetime64[ns]")


def _same_start_before(candidates: _Legs, start: np.datetime64, end: np.datetime64) -> int | None:
    """The position in `candidates` of the latest leg sharing this window's START and ending strictly earlier, or None.

    Sharing the start is what makes a cumulative subtraction valid (never across a fiscal-year boundary).
    """
    if candidates.ends.size == 0 or np.isnat(start) or np.isnat(end):
        return None
    hit = (candidates.starts == start) & (candidates.ends < end)
    if not hit.any():
        return None
    # The latest end wins; on a tie, the last such row in frame order.
    positions = np.flatnonzero(hit)
    hit_ends = candidates.ends[positions]
    return int(positions[np.flatnonzero(hit_ends == hit_ends.max())[-1]])


# ---------------------------------------------------------------------------- guards ---


def _scale_agrees(total: float, total_days: float, part: float, part_days: float, guards: PeriodGuards, two_sided: bool) -> bool:
    """Are the two legs of a subtraction plausibly the SAME line, compared as per-day rates?

    One-sided (upper bound `concept_switch_scale_max`) for an additive flow; two-sided with
    `share_basis_max_ratio` for a non-additive share count, so a split between the legs is refused.
    """
    if total_days <= 0 or part_days <= 0 or part == 0:
        return False
    ratio = abs(float(total) / total_days) / abs(float(part) / part_days)
    # Share counts need a tighter bound than 2.0, where a 2-for-1 split sits on the threshold.
    bound = guards.share_basis_max_ratio if two_sided else guards.concept_switch_scale_max
    if ratio > bound:
        return False
    return not two_sided or ratio >= 1 / bound


def _is_coherent(derived: float, siblings: list[float], spec: FieldSpec, guards: PeriodGuards) -> bool:
    """Is a derived quarter consistent with the year's observed quarters?

    False for a negative value on a `non_negative` field. Otherwise True if it shares its sign with
    ANY sibling, if its sign flip is forced by the year's total, or if its magnitude is within
    `max_opposite_sign_ratio` of the largest sibling.
    """
    if spec.sign == "non_negative" and derived < 0:
        return False
    if not siblings:
        return True
    if any((derived >= 0) == (s >= 0) for s in siblings):
        return True
    # A large opposite-sign quarter is forced when it flips the sign of the year's own total.
    if ((derived + sum(siblings)) >= 0) != (sum(siblings) >= 0):
        return True
    largest = max(abs(s) for s in siblings)
    return largest == 0 or abs(derived) <= largest * guards.max_opposite_sign_ratio


# ----------------------------------------------------------------------- the ladder ---


def _derived(total: dict, subtrahend: dict, basis: str, spec: FieldSpec, siblings: list[float], guards: PeriodGuards) -> dict | None:
    """One guarded subtraction `total - subtrahend` -> a quarter row, or None when the guards refuse it.

    Refused: a concept-switched (or share-count) subtraction failing the scale test, or a value `_is_coherent` rejects.
    """
    value = float(total["value"]) - float(subtrahend["value"])
    switched = str(total["source_concept"]) != str(subtrahend["source_concept"])
    start = subtrahend["period_end"] + _DAY
    end = total["period_end"]

    # Scale test on a concept switch, and always for a share count (legs may sit on two split bases).
    if switched or not spec.is_additive:
        if not _scale_agrees(
            total["value"], total["period_days"], subtrahend["value"], subtrahend["period_days"], guards, two_sided=not spec.is_additive
        ):
            return None
    if not _is_coherent(value, siblings, spec, guards):
        return None
    # The later filing date; a NaT never compares later, so it wins only as the total's own date.
    known = subtrahend["filing_date"] if subtrahend["filing_date"] > total["filing_date"] else total["filing_date"]
    return {
        "period_start": start,
        "period_end": end,
        "period_days": float((end - start) // _DAY),
        "value": value,
        "basis": basis,
        "known_from": known,
        "source_concept": total["source_concept"],
        "concept_switch": switched,
    }


#: `_quarter_rows`' row keys, also the unlabelled `quarterize` columns.
_ROW_COLUMNS: tuple[str, ...] = ("period_start", "period_end", "period_days", "value", "basis", "known_from", "source_concept", "concept_switch")


def _quarter_rows(facts: FieldFacts, spec: FieldSpec, guards: PeriodGuards) -> list[dict]:
    """One field's duration facts -> its discrete quarters as `_ROW_COLUMNS` dicts, ascending and unique by `period_end`.

    Reported quarters are kept and win over a derived one for the same window; the rest come from the ladder. A
    non-additive field (weighted-average shares) is differenced in share-days (`average x days`) and converted back.
    """
    facts = facts.take(np.flatnonzero(~np.isnan(facts.values) & ~np.isnat(facts.starts) & ~np.isnat(facts.ends)))
    # Before the share-day transform, so the value comparison is on as-filed numbers.
    facts = facts.take(np.flatnonzero(~_masquerading_quarters(facts)))
    if not spec.is_additive:
        facts = replace(facts, values=facts.values * _inclusive_days(facts.days))
    # Sorted once for all four shape reads below -- see `_latest_per_window`.
    ordered = facts.take(_window_order(facts))
    quarters, y6, y9, annual = (_Legs.of(ordered, s) for s in (QUARTERLY, YTD6, YTD9, ANNUAL))
    rows: list[dict] = [
        {
            "period_start": start,
            "period_end": r["period_end"],
            "period_days": r["period_days"],
            "value": r["value"],
            "basis": AS_REPORTED,
            "known_from": r["filing_date"],
            "source_concept": r["source_concept"],
            "concept_switch": False,
        }
        for r, start in zip(quarters.rows, quarters.starts, strict=True)
    ]  # still in share-days if weighted
    rows.extend(_ladder(quarters, y6, y9, annual, spec, guards))
    # An as-reported quarter always beats a derived one for the same window, then the earliest known (NaT last); the
    # first such row wins a tie, as a stable sort would keep it.
    best: dict[np.datetime64, tuple[tuple, dict]] = {}
    for row in rows:
        known = row["known_from"]
        key = (row["basis"] != AS_REPORTED, bool(np.isnat(known)), 0 if np.isnat(known) else int(known.view("i8")))
        held = best.get(row["period_end"])
        if held is None or key < held[0]:
            best[row["period_end"]] = (key, row)
    kept = [best[end][1] for end in sorted(best)]
    if not spec.is_additive:
        for row in kept:
            row["value"] = row["value"] / _inclusive_days(row["period_days"])
    return kept


def quarterize(
    facts: pd.DataFrame,
    spec: FieldSpec,
    guards: PeriodGuards | None = None,
    year_ends: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """One (ticker, field)'s duration facts -> discrete quarters (`_QUARTER_COLUMNS`), with provenance (`_quarter_rows`).

    `known_from` is the latest filing date of the inputs. `year_ends` defaults to this field's own.
    """
    guards = guards or load_guards()
    if facts.empty:
        return pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    rows = _quarter_rows(FieldFacts.of(facts), spec, guards)
    if not rows:
        return pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    out = pd.DataFrame({column: [row[column] for row in rows] for column in _ROW_COLUMNS})
    out.insert(0, "field", spec.name)
    out.insert(0, "ticker", facts["ticker"].iloc[0])
    # Callers should pass the TICKER's calendar: a field's own sparse annual facts mislabel fiscal years.
    return label_fiscal_periods(out, fiscal_year_ends(facts[facts["period_start"].notna()]) if year_ends is None else year_ends)


def _ladder(quarters: _Legs, y6: _Legs, y9: _Legs, annual: _Legs, spec: FieldSpec, guards: PeriodGuards) -> list[dict]:
    """The Q2/Q3 decumulation rungs, then Q4 by `FY - YTD9` or else `FY - (Q1+Q2+Q3)`; a declined rung emits nothing."""
    out: list[dict] = []
    for cumulative, earlier, basis in ((y6, quarters, Q2_FROM_YTD6), (y9, y6, Q3_FROM_YTD9)):
        for row, start, end in zip(cumulative.rows, cumulative.starts, cumulative.ends, strict=True):
            prior = _same_start_before(earlier, start, end)
            if prior is None:
                continue
            derived = _derived(row, earlier.rows[prior], basis, spec, [float(earlier.values[prior])], guards)
            if derived:
                out.append(derived)

    q_starts, q_ends, q_values = quarters.starts, quarters.ends, quarters.values
    for fy_row, fy_start, fy_end in zip(annual.rows, annual.starts, annual.ends, strict=True):
        within = (q_starts >= fy_start) & (q_ends <= fy_end)
        # A discrete quarter ending on the fiscal year-end IS Q4 as reported.
        if (q_ends[within] == fy_end).any():
            continue
        siblings = [float(v) for v in q_values[within]]

        ytd9 = _same_start_before(y9, fy_start, fy_end)
        if ytd9 is not None:
            derived = _derived(fy_row, y9.rows[ytd9], FY_MINUS_YTD9, spec, siblings, guards)
            if derived:
                out.append(derived)
                continue

        # Fallback needs exactly the three preceding quarters; a gap, overlap or stub is ambiguous.
        if int(within.sum()) != TTM_QUARTERS - 1:
            continue
        # Quarter legs are unique and ascending by end, so the last one inside is the latest.
        last = quarters.rows[int(np.flatnonzero(within)[-1])]
        filed = quarters.filed[within]
        filed = filed[~np.isnat(filed)]
        derived = _derived(
            fy_row,
            {
                "value": float(_skipna_sum(q_values[within])),
                "period_end": last["period_end"],
                "period_days": float(_skipna_sum(quarters.days[within])),
                "filing_date": filed.max() if filed.size else np.datetime64("NaT", "ns"),
                "source_concept": last["source_concept"],
            },
            FY_MINUS_QUARTERS,
            spec,
            siblings,
            guards,
        )
        if derived:
            out.append(derived)
    return out


# ------------------------------------------------------------------ fiscal calendar ---


def fiscal_year_ends(facts: pd.DataFrame) -> list[pd.Timestamp]:
    """The issuer's fiscal year-end dates, ascending, from its ANNUAL-shaped facts' window ends.

    Missing years between two ends are interpolated, and one year is extrapolated past the last
    annual end so quarters filed since the latest 10-K still get labelled. Empty when no annual fact.
    """
    annual = facts[(facts["duration_type"] == ANNUAL) & facts["value"].notna()]
    ends = sorted(pd.Timestamp(e) for e in pd.to_datetime(annual["period_end"]).dropna().unique())
    if not ends:
        return []
    # Fill gaps (a gap is a scoping loss, not a calendar change), then extend exactly one year.
    filled = [ends[0]]
    for end in ends[1:]:
        previous = filled[-1]
        years = max(round((end - previous).days / 364), 1)
        step = (end - previous) / years
        filled.extend(previous + step * n for n in range(1, years))
        filled.append(end)
    last = filled[-1]
    span = (last - filled[-2]).days if len(filled) > 1 else 364
    return [*filled, last + pd.Timedelta(days=span)]


@lru_cache(maxsize=256)
def _bounds_of(year_ends: tuple[pd.Timestamp, ...]) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    """`_fiscal_bounds` cached on the calendar tuple; returns immutable tuples because callers share them."""
    ends = tuple(sorted(pd.Timestamp(e) for e in year_ends))
    starts = (ends[0] - pd.Timedelta(days=364), *(e + pd.Timedelta(days=1) for e in ends[:-1]))
    return ends, starts


def _fiscal_bounds(year_ends: list[pd.Timestamp]) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    """The fiscal years as `(ends, starts)`: each starts the day after the previous end, the first 364 days back.

    Shared by `label_fiscal_periods` and `fiscal_quarter_of_end` so both label a quarter identically.
    """
    return _bounds_of(tuple(year_ends))


def fiscal_quarter_of_end(end, year_ends: list[pd.Timestamp]) -> int | None:
    """Which fiscal quarter (1-4) a period ENDING on `end` sits in, or None if outside the calendar.

    The end-date mirror of `label_fiscal_periods`, for TTM and instant rows too: the covered span from
    the year start, divided by that year's quarter length and rounded.
    """
    if not year_ends or end is None or pd.isna(end):
        return None
    ends, starts = _fiscal_bounds(year_ends)
    end = pd.Timestamp(end)
    # side='left' puts a period ending exactly ON a year end into that year, where Q4 lives.
    position = int(pd.Series(ends).searchsorted(end, side="left"))
    if position >= len(ends):
        return None
    year_start, year_end = starts[position], ends[position]
    quarter_length = max((year_end - year_start).days + 1, 1) / TTM_QUARTERS
    covered = (end - year_start).days + 1
    return int(min(max(round(covered / quarter_length), 1), TTM_QUARTERS))


def label_fiscal_periods(quarters: pd.DataFrame, year_ends: list[pd.Timestamp]) -> pd.DataFrame:
    """Attach `fiscal_year` and `fiscal_quarter` and return `_QUARTER_COLUMNS`.

    The quarter is the offset of `period_start` into its fiscal year divided by that year's own quarter
    length, rounded (exact for 52/53-week years). `fiscal_year` is the calendar year of the year-end,
    matching SEC's `fy`. Quarters beyond the calendar stay NA.
    """
    out = quarters.copy()
    out["fiscal_year"] = pd.NA
    out["fiscal_quarter"] = pd.NA
    if not year_ends or out.empty:
        return out[list(_QUARTER_COLUMNS)]
    ends, starts = _fiscal_bounds(year_ends)
    bounds = pd.Series(ends)
    # side='left' puts a quarter ending exactly ON a year end into that year, where Q4 lives.
    slot = bounds.searchsorted(out["period_end"].values, side="left")
    inside = slot < len(ends)
    if inside.any():
        day = np.timedelta64(1, "D")
        position = slot[inside]
        year_end = np.array(ends, dtype="datetime64[ns]")[position]
        year_start = np.array(starts, dtype="datetime64[ns]")[position]
        quarter_length = np.maximum((year_end - year_start) // day + 1, 1) / TTM_QUARTERS
        offset = (out["period_start"].to_numpy("datetime64[ns]")[inside] - year_start) // day
        # np.round is half-to-even, like the builtin `round`; cells stay python ints in an object column.
        quarter = np.minimum(np.maximum(np.round(offset / quarter_length) + 1, 1), TTM_QUARTERS)
        years = np.array([end.year for end in ends])[position]
        fiscal_year = np.full(len(out), pd.NA, dtype=object)
        fiscal_quarter = np.full(len(out), pd.NA, dtype=object)
        fiscal_year[inside] = [int(y) for y in years]
        fiscal_quarter[inside] = [int(q) for q in quarter]
        out["fiscal_year"] = fiscal_year
        out["fiscal_quarter"] = fiscal_quarter
    return out[list(_QUARTER_COLUMNS)]


# ----------------------------------------------------------------- trailing twelve ---


def trailing_twelve(quarters: pd.DataFrame, spec: FieldSpec, annual: pd.DataFrame | None = None, guards: PeriodGuards | None = None) -> pd.DataFrame:
    """A TTM value at every quarter end, built only from FOUR contiguous discrete quarters (`_ttm_value`).

    Otherwise the row is NULL (never a carried-forward annual). A non-additive field is the share-day-weighted
    mean, or the as-reported annual figure where the window ends on a fiscal year end; a window straddling a
    split is NULL.
    """
    empty = pd.DataFrame(columns=["ticker", "field", "period_end", "value", "basis", "known_from", "n_quarters"])
    if quarters.empty:
        return empty
    guards = guards or load_guards()
    ordered = quarters.sort_values("period_end").reset_index(drop=True)
    # Only the non-additive branch reads the annual facts.
    reported_annual = {} if spec.is_additive or annual is None or annual.empty else _annual_by_end(FieldFacts.of(annual))
    ticker = ordered["ticker"].iloc[0]
    # Arrays taken once; each window below is a positional slice of them.
    ends = list(ordered["period_end"])
    values = ordered["value"].to_numpy(dtype=float)
    days_all = _inclusive_days(ordered["period_days"]).to_numpy(dtype=float)
    known = ordered["known_from"]
    known_ns = known.to_numpy() if is_datetime64_any_dtype(known) else None
    starts_ns, ends_ns = _window_bounds(ordered)
    rows = []
    for i, end in enumerate(ends):
        low = max(0, i - TTM_QUARTERS + 1)
        base = {"ticker": ticker, "field": spec.name, "period_end": end}
        fact = reported_annual.get(int(ends_ns[i].view("i8")))
        if fact is not None:
            rows.append({**base, "value": float(fact[0]), "basis": TTM_AS_REPORTED_ANNUAL, "known_from": pd.Timestamp(fact[1]), "n_quarters": 0})
            continue
        value = _ttm_value(i, starts_ns, ends_ns, values, days_all, spec, guards)
        if value is None:
            rows.append({**base, "value": None, "basis": None, "known_from": None, "n_quarters": i + 1 - low})
            continue
        if known_ns is not None:
            seen = known_ns[low : i + 1]
            seen = seen[~np.isnat(seen)]
            latest = pd.Timestamp(seen.max()) if seen.size else pd.NaT
        else:
            latest = known.iloc[low : i + 1].max()
        rows.append(
            {
                **base,
                "value": value,
                "basis": TTM_FOUR_QUARTERS if spec.is_additive else TTM_FOUR_QUARTER_MEAN,
                "known_from": latest,
                "n_quarters": TTM_QUARTERS,
            }
        )
    out = pd.DataFrame(rows, columns=list(empty.columns))
    # One null representation: mixed None/NaN in an object column breaks downstream null tests.
    return out.astype({"basis": "string"})


def _ttm_value(
    i: int, starts: np.ndarray, ends: np.ndarray, values: np.ndarray, days: np.ndarray, spec: FieldSpec, guards: PeriodGuards
) -> float | None:
    """The four-quarter TTM ending at quarter `i` (arrays ascending by end, `days` inclusive), or None when the
    window is short, not contiguous, or (for a share count) straddles a split."""
    low = max(0, i - TTM_QUARTERS + 1)
    if i + 1 - low < TTM_QUARTERS or not _bounds_are_contiguous(starts[low : i + 1], ends[low : i + 1]):
        return None
    window = values[low : i + 1]
    if not spec.is_additive and not _one_share_basis_values(window, guards):
        return None
    # A twelve-month weighted average is share-days over days, not the mean of four means.
    weights = days[low : i + 1]
    return float(_skipna_sum(window) if spec.is_additive else _skipna_sum(window * weights) / _skipna_sum(weights))


def _annual_by_end(facts: FieldFacts) -> dict[int, tuple[float, np.datetime64]]:
    """The as-reported annual `(value, filing_date)` per window end (nanoseconds), the latest filing winning."""
    annual = facts.take(np.flatnonzero(facts.duration_type == ANNUAL))
    annual = annual.take(_window_order(annual))
    annual = annual.take(_latest_per_window(annual.ends))
    return {
        int(end.view("i8")): (value, filed) for end, value, filed in zip(annual.ends, annual.values, annual.filed, strict=True) if not np.isnat(end)
    }


def newest_periods(facts: FieldFacts, spec: FieldSpec, guards: PeriodGuards) -> tuple[dict | None, dict | None]:
    """The newest TTM row and the newest discrete quarter of one field's duration facts, as `{period_end, value}`.

    The last rows of `trailing_twelve(quarterize(...))` and `quarterize(...)`, computed without building either
    frame; `(None, None)` for a non-`duration` field or one with no quarter.
    """
    if spec.kind != "duration":
        return None, None
    rows = _quarter_rows(facts, spec, guards)
    if not rows:
        return None, None
    last = rows[-1]
    quarter = {"period_end": last["period_end"], "value": last["value"]}
    fact = None if spec.is_additive else _annual_by_end(facts).get(int(last["period_end"].view("i8")))
    if fact is not None:
        return {"period_end": last["period_end"], "value": float(fact[0])}, quarter
    window = rows[-TTM_QUARTERS:]
    value = _ttm_value(
        len(window) - 1,
        np.array([row["period_start"] for row in window], dtype="datetime64[ns]"),
        np.array([row["period_end"] for row in window], dtype="datetime64[ns]"),
        np.array([row["value"] for row in window], dtype=float),
        _inclusive_days(np.array([row["period_days"] for row in window], dtype=float)),
        spec,
        guards,
    )
    return {"period_end": last["period_end"], "value": value}, quarter


def _one_share_basis(window: pd.DataFrame, guards: PeriodGuards) -> bool:
    """Are the window's share counts on ONE split basis (max/min within `share_basis_max_ratio`)?

    A window straddling a split mixes two incompatible units; it is refused, not repaired.
    """
    return _one_share_basis_values(window["value"].to_numpy(dtype=float), guards)


def _one_share_basis_values(values: np.ndarray, guards: PeriodGuards) -> bool:
    """`_one_share_basis` on a window's value array; NaN is skipped, an all-NaN window is refused."""
    present = np.abs(values[~np.isnan(values)])
    if present.size == 0:
        return False
    smallest = present.min()
    return bool(smallest > 0 and (present.max() / smallest) <= guards.share_basis_max_ratio)


def _skipna_sum(values: np.ndarray) -> np.float64:
    """`Series.sum()` on a float array: NaN counts as zero, the same `ndarray.sum` underneath (numpy division rules kept)."""
    return np.where(np.isnan(values), 0.0, values).sum()


def _window_is_contiguous(window: pd.DataFrame) -> bool:
    """True when the quarters abut (gaps of at most 1 day) and span `TTM_MIN_DAYS`..`TTM_MAX_DAYS`."""
    starts, ends = _window_bounds(window)
    return _bounds_are_contiguous(starts, ends)


def _bounds_are_contiguous(starts: np.ndarray, ends: np.ndarray) -> bool:
    """`_window_is_contiguous` on nanosecond bound arrays; whole days are floored, a NaT gap is not a break."""
    day = np.timedelta64(1, "D")
    gaps = starts[1:] - ends[:-1]
    known = ~np.isnat(gaps)
    if (np.abs(gaps[known] // day) > 1).any():
        return False
    span = ends[-1] - starts[0]
    if np.isnat(span):
        return False
    return bool(TTM_MIN_DAYS <= span // day <= TTM_MAX_DAYS)


# --------------------------------------------------------------------------- instants ---

#: An instant tagged with the fiscal YEAR is the year-END snapshot, which occupies the Q4 grid slot.
_YEAR_END_LABEL = {"FY": "Q4", "YTD12": "Q4"}


def instant_stock(facts: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time facts on the quarterly grid: balance-sheet levels, cover-page shares, headcount.

    Instants are rows with no `period_start` (else `duration_type == instant`); an `FY`/`YTD12` label on
    them is relabelled `Q4`. One row per (ticker, field, period_end), the latest filing winning.
    """
    if facts.empty:
        return facts
    if "period_start" in facts.columns:
        out = facts[facts["period_start"].isna()].copy()
    else:
        out = facts[facts["duration_type"] == INSTANT].copy()
    if out.empty:
        return out
    if "fiscal_period" in out.columns:
        out["fiscal_period"] = out["fiscal_period"].replace(_YEAR_END_LABEL)
    keys = [c for c in ("ticker", "field", "period_end") if c in out.columns]
    if keys and "filing_date" in out.columns:
        out = out.sort_values([*keys, "filing_date"]).drop_duplicates(subset=keys, keep="last")
    return out


class InstantLookup:
    """`instant_stock`'s output as one sorted `(period_end, value)` array pair per field, so a level's
    latest known value at `as_of` is a `np.searchsorted`.

    Equivalent to `build_history.carry_latest_known` (backward as-of, exact matches allowed).
    """

    __slots__ = ("_by_field",)

    #: The columns a lookup needs; `filing_date` only breaks a duplicate `period_end`.
    _COLUMNS: tuple[str, ...] = ("field", "period_end", "value", "filing_date")

    def __init__(self, instants: pd.DataFrame | None) -> None:
        self._by_field: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if instants is None or instants.empty or "field" not in instants.columns:
            return
        frame = instants[[c for c in self._COLUMNS if c in instants.columns]].copy()
        frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce").astype("datetime64[ns]")
        frame = frame.dropna(subset=["period_end"])
        if "filing_date" in frame.columns:
            frame = frame.sort_values(["period_end", "filing_date"]).drop_duplicates(subset=["field", "period_end"], keep="last")
        else:
            frame = frame.sort_values("period_end")
        values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
        frame = frame.assign(_value=values)
        # `searchsorted` silently misreads an unsorted array; groupby keeps the global order per group.
        ends = frame["period_end"].to_numpy("datetime64[ns]")
        assert bool(np.all(ends[:-1] <= ends[1:])), "InstantLookup: period_end is not ascending -- searchsorted would be wrong"
        for name, group in frame.groupby("field", sort=False):
            self._by_field[str(name)] = (group["period_end"].to_numpy(dtype="datetime64[ns]"), group["_value"].to_numpy(dtype=float))

    def value(self, field: str, as_of) -> float | None:
        """`field`'s latest value dated on or before `as_of`, or None when there is none (or it is NaN)."""
        entry = self._by_field.get(field)
        if entry is None or as_of is None or pd.isna(as_of):
            return None
        ends, values = entry
        index = int(np.searchsorted(ends, np.datetime64(pd.Timestamp(as_of), "ns"), side="right")) - 1
        if index < 0:
            return None
        value = values[index]
        return None if np.isnan(value) else float(value)


# ----------------------------------------------------------------------- entry point ---


def build_periods(
    facts: pd.DataFrame,
    catalogue,
    guards: PeriodGuards | None = None,
    year_ends: list[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every `duration` field's discrete quarters and TTM values, plus the instants, for one ticker's facts.

    Returns long frames `(quarters, ttm, instants)`; nothing is written to a table. `year_ends` is the ticker's fiscal calendar,
    derived from all annual-shaped facts when omitted and shared by every field.
    """
    # Resolved once here rather than per (event, field) in the callees.
    guards = guards or load_guards()
    if facts.empty:
        return (
            pd.DataFrame(columns=list(_QUARTER_COLUMNS)),
            trailing_twelve(pd.DataFrame(), catalogue.field(catalogue.extracted_fields[0]), guards=guards),
            facts,
        )
    durations = facts[is_duration(facts)]
    # One calendar for the whole ticker, built from every field's annual-shaped facts.
    if year_ends is None:
        year_ends = fiscal_year_ends(durations)
    all_quarters, all_ttm = [], []
    for name, group in durations.groupby("field", sort=True):
        spec = catalogue.field(name)
        if spec.kind != "duration":
            continue
        quarters = quarterize(group, spec, guards, year_ends)
        if quarters.empty:
            continue
        all_quarters.append(quarters)
        all_ttm.append(trailing_twelve(quarters, spec, annual=group, guards=guards))
    quarters = pd.concat(all_quarters, ignore_index=True) if all_quarters else pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    ttm = pd.concat(all_ttm, ignore_index=True) if all_ttm else pd.DataFrame()
    return quarters, ttm, instant_stock(facts)


def is_duration(facts: pd.DataFrame) -> pd.Series:
    """The rows `build_periods` treats as duration facts: every shape but `instant` and `other`."""
    return ~facts["duration_type"].isin([INSTANT, OTHER_SHAPE])
