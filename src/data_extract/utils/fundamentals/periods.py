"""
periods.py (src/data_extract/utils/fundamentals/periods.py)
--------------------------------------------------------------------------------------
The period engine: turns one ticker's as-filed duration facts from `fundamentals_facts` into
discrete quarters, trailing-twelve-month values and quarterly-grid instants, in memory only.

Every input is selected by its calendar WINDOW shape (`period_shape`), never by the filer's
`fiscal_period` label. Quarters are kept as reported or derived by the ladder `Q2 = YTD6 - Q1`,
`Q3 = YTD9 - YTD6`, `Q4 = FY - YTD9`, fallback `Q4 = FY - (Q1+Q2+Q3)`, each tagged with its `basis`.
The FY and YTD legs may come from different filings and concepts: a switch is allowed, recorded in
`concept_switch`, and gated by a scale test. A TTM is emitted only from four contiguous discrete
quarters, otherwise NULL with `insufficient_quarters`. Refused windows are reported through `refusals`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

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

#: `dc_code`s attached to a value this module refuses to emit. Every refusal travels through the
#: `refusals` out-parameter to `fundamentals_reason_codes`, the single source of truth for absences.
INSUFFICIENT_QUARTERS = "insufficient_quarters"
SPLIT_BASIS_MISMATCH = "split_basis_mismatch"

#: Quarter-level refusals made by `_derived` (the window existed but the arithmetic was refused):
#:   * `derived_basis_mismatch` -- the scale test refused a concept-switched subtraction (a share
#:     count on two split bases uses `split_basis_mismatch` instead).
#:   * `derived_sign_implausible` -- `_is_coherent` refused the value's sign.
DERIVED_BASIS_MISMATCH = "derived_basis_mismatch"
DERIVED_SIGN_IMPLAUSIBLE = "derived_sign_implausible"

#: A quarterly-context fact whose value is the whole fiscal year, where no annual-window fact exists
#: to compare it against (see `_is_ambiguous_duration`). Refused, never reclassified.
AMBIGUOUS_DURATION = "ambiguous_duration"

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


#: `_latest_per_window`'s sort order and therefore its contract: the row that sorts last wins.
_WINDOW_ORDER: list[str] = ["period_end", "filing_date", "period_days"]


def _latest_per_window(frame: pd.DataFrame, *, presorted: bool = False) -> pd.DataFrame:
    """One row per calendar window, the LATEST filing winning.

    A window is identified by its END within `_SAME_PERIOD_DAYS` (filers nudge boundary days
    between filings), not by its exact (start, end). Latest-wins is point-in-time correct because
    the caller passes only facts with `filing_date <= as_of`. `presorted=True` asserts the frame
    is already ordered by `_WINDOW_ORDER`; the bucketing depends on that sort.
    """
    if frame.empty:
        return frame
    ordered = frame if presorted else frame.sort_values(_WINDOW_ORDER)
    ends = pd.to_datetime(ordered["period_end"])
    bucket = (ends.diff().dt.days.fillna(0) > _SAME_PERIOD_DAYS).cumsum()
    return ordered[~bucket.duplicated(keep="last")]


def _shape(frame: pd.DataFrame, shape: str, *, presorted: bool = False) -> pd.DataFrame:
    return _latest_per_window(frame[frame["duration_type"] == shape], presorted=presorted)


#: Max gap between a nine-month cumulative's end and a fourth quarter's start for the two to be
#: contiguous pieces of one fiscal year (52/53-week filers drift by a day or two).
_CONTIGUOUS_DAYS = 4


def _is_ambiguous_duration(q, ends: np.ndarray, values: np.ndarray) -> bool:
    """Is this `quarterly`-shaped fact really the whole fiscal year? Used only where no annual fact exists.

    True when a contiguous `ytd9` (given as numpy `ends` / `values`) exists, is materially non-zero
    (>1% of the quarter), has the SAME sign, and is smaller in magnitude than the "quarter". The
    same-sign condition keeps a genuine loss quarter after a profitable nine months.
    """
    if ends.size == 0 or pd.isna(q.period_start) or pd.isna(q.value):
        return False
    gap = cast(Any, np.datetime64(cast(Any, q.period_start), "ns") - ends) / np.timedelta64(1, "D")
    quarter = float(cast(Any, q.value))
    # An opposite-sign cumulative means the year turned, not that the window is mislabelled.
    keep = (gap >= 0) & (gap <= _CONTIGUOUS_DAYS) & (values * quarter > 0)
    if not keep.any():
        return False
    nine = float(np.abs(values[keep]).max())
    return bool(nine > 0.01 * abs(quarter) and abs(quarter) > nine)


def _ambiguous_refusal(q) -> dict:
    """The `AMBIGUOUS_DURATION` refusal record of one dropped quarter row."""
    return {
        "period_start": q.period_start,
        "period_end": q.period_end,
        "period_days": q.period_days,
        "value": float(cast(Any, q.value)),
        "known_from": q.filing_date,
        "dc_code": AMBIGUOUS_DURATION,
        "source_concept": getattr(q, "source_concept", None),
    }


def _drop_annual_masquerading_as_quarter(
    frame: pd.DataFrame,
    refusals: list[dict] | None = None,
) -> pd.DataFrame:
    """Drop quarterly-shaped facts whose value is the FULL YEAR (an annual figure tagged into a Q4 context).

    With an annual fact present, a quarter is dropped only when all hold: an annual fact ends within
    `_SAME_PERIOD_DAYS` of it, the values agree within 0.1%, and an interim cumulative inside that year
    is >1% of the annual (so a real Q4 that equals its year, after a zero nine months, survives).
    With no annual fact, `_is_ambiguous_duration` decides; those drops are appended to `refusals`
    with `AMBIGUOUS_DURATION`.
    """
    quarters = frame[frame["duration_type"] == QUARTERLY]
    annual = frame[frame["duration_type"] == ANNUAL]
    interim = frame[frame["duration_type"].isin((YTD6, YTD9))]
    ytd9 = frame[frame["duration_type"] == YTD9]
    # `annual` may be empty: the no-annual-fact branch must still run.
    if quarters.empty or (annual.empty and ytd9.empty):
        return frame
    drop: list = []
    # Numpy arrays taken once; `quarterize` has already dropped rows with a null value, start or end.
    a_end = annual["period_end"].to_numpy("datetime64[ns]")
    a_start = annual["period_start"].to_numpy("datetime64[ns]")
    a_value = annual["value"].to_numpy(float)
    y9_end = ytd9["period_end"].to_numpy("datetime64[ns]")
    y9_value = ytd9["value"].to_numpy(float)
    i_end = interim["period_end"].to_numpy("datetime64[ns]")
    i_value = interim["value"].to_numpy(float)
    same_period = np.timedelta64(_SAME_PERIOD_DAYS, "D")
    for q in quarters.itertuples():
        q_end = np.datetime64(cast(Any, q.period_end), "ns")
        near = np.abs(a_end - q_end) <= same_period
        if not near.any():
            ambiguous = _is_ambiguous_duration(q, y9_end, y9_value)
            if ambiguous:
                drop.append(q.Index)
            if ambiguous and refusals is not None:
                refusals.append(_ambiguous_refusal(q))
            continue
        near_value = a_value[near]
        scale = float(np.abs(near_value).max())
        if scale < 1 or float(np.abs(near_value - float(cast(Any, q.value))).min()) > 0.001 * scale:
            continue
        # Scoped to the annual window: the nine-month cumulative ends where the quarter begins.
        year_start = a_start[near].min()
        accumulated = (i_end > year_start) & (i_end <= q_end)
        if (np.abs(i_value[accumulated]) > 0.01 * scale).any():
            drop.append(q.Index)
    return frame.drop(index=drop) if drop else frame


def _window_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`frame`'s `(period_start, period_end)` as nanosecond arrays, taken once per candidate frame."""
    return frame["period_start"].to_numpy("datetime64[ns]"), frame["period_end"].to_numpy("datetime64[ns]")


def _same_start_before(candidates: pd.DataFrame, start, end, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> pd.Series | None:
    """The latest candidate sharing this window's START and ending strictly earlier, or None.

    Sharing the start is what makes a cumulative subtraction valid (never across a fiscal-year boundary).
    `bounds` is `_window_bounds(candidates)`, passed by a caller that probes the same frame repeatedly.
    """
    if candidates.empty or pd.isna(start) or pd.isna(end):
        return None
    starts, ends = bounds if bounds is not None else _window_bounds(candidates)
    hit = (starts == np.datetime64(pd.Timestamp(start), "ns")) & (ends < np.datetime64(pd.Timestamp(end), "ns"))
    if not hit.any():
        return None
    # The latest end wins; on a tie, the last such row in frame order.
    positions = np.flatnonzero(hit)
    hit_ends = ends[positions]
    return candidates.iloc[int(positions[np.flatnonzero(hit_ends == hit_ends.max())[-1]])]


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


def _derived(
    total, subtrahend, basis: str, spec: FieldSpec, siblings: list[float], guards: PeriodGuards, refusals: list[dict] | None = None
) -> dict | None:
    """One guarded subtraction `total - subtrahend` -> a quarter row, or None when the guards refuse it.

    A refusal is appended to `refusals` with the would-be window, the refused value and its `dc_code`.
    For a non-additive field the recorded value is in share-days (diagnostic only).
    """
    value = float(total["value"]) - float(subtrahend["value"])
    switched = str(total["source_concept"]) != str(subtrahend["source_concept"])
    start = pd.Timestamp(subtrahend["period_end"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(total["period_end"])

    def refuse(code: str) -> None:
        if refusals is None:
            return
        refusals.append(
            {
                "period_start": start,
                "period_end": end,
                "period_days": (end - start).days,
                "value": value,
                "basis": basis,
                "known_from": max(pd.Timestamp(total["filing_date"]), pd.Timestamp(subtrahend["filing_date"])),
                "source_concept": total["source_concept"],
                "dc_code": code,
            }
        )

    # Scale test on a concept switch, and always for a share count (legs may sit on two split bases).
    if switched or not spec.is_additive:
        if not _scale_agrees(
            total["value"], total["period_days"], subtrahend["value"], subtrahend["period_days"], guards, two_sided=not spec.is_additive
        ):
            refuse(SPLIT_BASIS_MISMATCH if not spec.is_additive else DERIVED_BASIS_MISMATCH)
            return None
    if not _is_coherent(value, siblings, spec, guards):
        refuse(DERIVED_SIGN_IMPLAUSIBLE)
        return None
    return {
        "period_start": start,
        "period_end": end,
        "period_days": (end - start).days,
        "value": value,
        "basis": basis,
        "known_from": max(pd.Timestamp(total["filing_date"]), pd.Timestamp(subtrahend["filing_date"])),
        "source_concept": total["source_concept"],
        "concept_switch": switched,
    }


def quarterize(
    facts: pd.DataFrame,
    spec: FieldSpec,
    guards: PeriodGuards | None = None,
    year_ends: list[pd.Timestamp] | None = None,
    refusals: list[dict] | None = None,
) -> pd.DataFrame:
    """One (ticker, field)'s duration facts -> discrete quarters (`_QUARTER_COLUMNS`), with provenance.

    Reported quarters are kept and win over a derived one for the same window; the rest come from the
    ladder. A non-additive field (weighted-average shares) is differenced in share-days
    (`average x days`) and converted back. `known_from` is the latest filing date of the inputs.
    `refusals`, when given, collects every declined window; `year_ends` defaults to this field's own.
    """
    guards = guards or load_guards()
    if facts.empty:
        return pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    frame = facts[facts["value"].notna() & facts["period_start"].notna() & facts["period_end"].notna()].copy()
    # Coerce only if the caller did not, so strings are never sorted lexicographically below.
    for column in ("period_start", "period_end", "filing_date"):
        if not is_datetime64_any_dtype(frame[column]):
            frame[column] = pd.to_datetime(frame[column])
    # Before the share-day transform, so the value comparison is on as-filed numbers.
    frame = _drop_annual_masquerading_as_quarter(frame, refusals)
    if not spec.is_additive:
        frame["value"] = frame["value"] * _inclusive_days(frame["period_days"])

    # Sorted once for all four shape reads below -- see `_latest_per_window`.
    frame = frame.sort_values(_WINDOW_ORDER)

    quarters = _shape(frame, QUARTERLY, presorted=True)
    rows: list[dict] = [
        {
            "period_start": r.period_start,
            "period_end": r.period_end,
            "period_days": r.period_days,
            "value": float(cast(Any, r.value)),
            "basis": AS_REPORTED,
            "known_from": r.filing_date,
            "source_concept": r.source_concept,
            "concept_switch": False,
        }
        for r in quarters.itertuples()
    ]  # still in share-days if weighted

    y6, y9, annual = (_shape(frame, s, presorted=True) for s in (YTD6, YTD9, ANNUAL))
    rows.extend(_ladder(quarters, y6, y9, annual, spec, guards, refusals))

    out = pd.DataFrame(rows, columns=[c for c in _QUARTER_COLUMNS if c not in ("ticker", "field", "fiscal_year", "fiscal_quarter")])
    if out.empty:
        return pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    if not spec.is_additive:
        out["value"] = out["value"] / _inclusive_days(out["period_days"])
    # An as-reported quarter always beats a derived one for the same window.
    out["_rank"] = (out["basis"] != AS_REPORTED).astype(int)
    out = out.sort_values(["period_end", "_rank", "known_from"]).drop_duplicates(subset=["period_end"], keep="first").drop(columns="_rank")
    out.insert(0, "field", spec.name)
    out.insert(0, "ticker", facts["ticker"].iloc[0])
    # Callers should pass the TICKER's calendar: a field's own sparse annual facts mislabel fiscal years.
    return label_fiscal_periods(out, fiscal_year_ends(frame) if year_ends is None else year_ends)


def _ladder(
    quarters: pd.DataFrame,
    y6: pd.DataFrame,
    y9: pd.DataFrame,
    annual: pd.DataFrame,
    spec: FieldSpec,
    guards: PeriodGuards,
    refusals: list[dict] | None = None,
) -> list[dict]:
    """The Q2/Q3 decumulation rungs, then Q4 by `FY - YTD9` or else `FY - (Q1+Q2+Q3)`.

    A rung that declines a window records it once in `refusals`; a window with no input to difference
    is not a refusal and is not recorded.
    """
    out: list[dict] = []
    for cumulative, earlier, basis in ((y6, quarters, Q2_FROM_YTD6), (y9, y6, Q3_FROM_YTD9)):
        earlier_bounds = _window_bounds(earlier)
        for row in cumulative.itertuples():
            prior = _same_start_before(earlier, row.period_start, row.period_end, earlier_bounds)
            if prior is None:
                continue
            derived = _derived(cast(Any, row)._asdict(), prior, basis, spec, [float(cast(Any, prior["value"]))], guards, refusals)
            if derived:
                out.append(derived)

    y9_bounds = _window_bounds(y9)
    for fy in annual.itertuples():
        fy_row = cast(Any, fy)._asdict()
        inside = quarters[(quarters["period_start"] >= fy.period_start) & (quarters["period_end"] <= fy.period_end)]
        # A discrete quarter ending on the fiscal year-end IS Q4 as reported.
        if (inside["period_end"] == fy.period_end).any():
            continue
        siblings = [float(cast(Any, v)) for v in inside["value"]]

        ytd9 = _same_start_before(y9, fy.period_start, fy.period_end, y9_bounds)
        if ytd9 is not None:
            derived = _derived(fy_row, ytd9, FY_MINUS_YTD9, spec, siblings, guards, refusals)
            if derived:
                out.append(derived)
                continue

        # Fallback needs exactly the three preceding quarters; a gap, overlap or stub is ambiguous.
        if len(inside) != TTM_QUARTERS - 1:
            continue
        total = float(inside["value"].sum())
        last = inside.sort_values("period_end").iloc[-1]
        derived = _derived(
            fy_row,
            {
                "value": total,
                "period_end": last["period_end"],
                "period_days": float(inside["period_days"].sum()),
                "filing_date": inside["filing_date"].max(),
                "source_concept": last["source_concept"],
            },
            FY_MINUS_QUARTERS,
            spec,
            siblings,
            guards,
            refusals,
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
    for position, index in zip(slot, out.index, strict=False):
        if position >= len(ends):
            continue
        year_start, year_end = starts[position], ends[position]
        quarter_length = max((year_end - year_start).days + 1, 1) / TTM_QUARTERS
        offset = (pd.Timestamp(cast(Any, out.at[index, "period_start"])) - year_start).days
        out.at[index, "fiscal_year"] = year_end.year
        out.at[index, "fiscal_quarter"] = int(min(max(round(offset / quarter_length) + 1, 1), TTM_QUARTERS))
    return out[list(_QUARTER_COLUMNS)]


# ----------------------------------------------------------------- trailing twelve ---


def trailing_twelve(quarters: pd.DataFrame, spec: FieldSpec, annual: pd.DataFrame | None = None, guards: PeriodGuards | None = None) -> pd.DataFrame:
    """A TTM value at every quarter end, built only from FOUR contiguous discrete quarters.

    Otherwise the row is NULL with `insufficient_quarters` (never a carried-forward annual). A
    non-additive field is the share-day-weighted mean, or the as-reported annual figure where the
    window ends on a fiscal year end; a window straddling a split is NULL with `split_basis_mismatch`.
    """
    empty = pd.DataFrame(columns=["ticker", "field", "period_end", "value", "basis", "known_from", "n_quarters", "dc_code"])
    if quarters.empty:
        return empty
    guards = guards or load_guards()
    ordered = quarters.sort_values("period_end").reset_index(drop=True)
    # Only the non-additive branch reads the annual facts.
    reported_annual = {} if spec.is_additive else _annual_by_end(annual)
    rows = []
    for i in range(len(ordered)):
        window = ordered.iloc[max(0, i - TTM_QUARTERS + 1) : i + 1]
        end = window["period_end"].iloc[-1]
        base = {"ticker": ordered["ticker"].iloc[0], "field": spec.name, "period_end": end}
        if not spec.is_additive and end in reported_annual:
            fact = reported_annual[end]
            rows.append(
                {
                    **base,
                    "value": float(fact["value"]),
                    "basis": TTM_AS_REPORTED_ANNUAL,
                    "known_from": fact["filing_date"],
                    "n_quarters": 0,
                    "dc_code": None,
                }
            )
            continue
        if len(window) < TTM_QUARTERS or not _window_is_contiguous(window):
            rows.append({**base, "value": None, "basis": None, "known_from": None, "n_quarters": len(window), "dc_code": INSUFFICIENT_QUARTERS})
            continue
        if not spec.is_additive and not _one_share_basis(window, guards):
            rows.append({**base, "value": None, "basis": None, "known_from": None, "n_quarters": len(window), "dc_code": SPLIT_BASIS_MISMATCH})
            continue
        # A twelve-month weighted average is share-days over days, not the mean of four means.
        days = _inclusive_days(window["period_days"])
        value = window["value"].sum() if spec.is_additive else (window["value"] * days).sum() / days.sum()
        rows.append(
            {
                **base,
                "value": float(value),
                "basis": TTM_FOUR_QUARTERS if spec.is_additive else TTM_FOUR_QUARTER_MEAN,
                "known_from": window["known_from"].max(),
                "n_quarters": TTM_QUARTERS,
                "dc_code": None,
            }
        )
    out = pd.DataFrame(rows, columns=list(empty.columns))
    # One null representation: mixed None/NaN in an object column breaks downstream null tests.
    return out.astype({"basis": "string", "dc_code": "string"})


def _annual_by_end(annual: pd.DataFrame | None) -> dict:
    if annual is None or annual.empty:
        return {}
    latest = _latest_per_window(annual[annual["duration_type"] == ANNUAL])
    return {pd.Timestamp(cast(Any, r.period_end)): {"value": r.value, "filing_date": r.filing_date} for r in latest.itertuples()}


def _one_share_basis(window: pd.DataFrame, guards: PeriodGuards) -> bool:
    """Are the window's share counts on ONE split basis (max/min within `share_basis_max_ratio`)?

    A window straddling a split mixes two incompatible units; it is refused, not repaired.
    """
    values = window["value"].abs()
    smallest = values.min()
    return smallest > 0 and (values.max() / smallest) <= guards.share_basis_max_ratio


def _window_is_contiguous(window: pd.DataFrame) -> bool:
    """True when the quarters abut (gaps of at most 1 day) and span `TTM_MIN_DAYS`..`TTM_MAX_DAYS`."""
    starts = list(window["period_start"])
    ends = list(window["period_end"])
    for previous_end, next_start in zip(ends, starts[1:], strict=False):
        if abs((pd.Timestamp(next_start) - pd.Timestamp(previous_end)).days) > 1:
            return False
    span = (pd.Timestamp(ends[-1]) - pd.Timestamp(starts[0])).days
    return TTM_MIN_DAYS <= span <= TTM_MAX_DAYS


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
    refusals: list[dict] | None = None,
    year_ends: list[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every `duration` field's discrete quarters and TTM values, plus the instants, for one ticker's facts.

    Returns long frames `(quarters, ttm, instants)`; nothing is written to a table. `refusals`, when given,
    collects every refused window tagged with its `field`. `year_ends` is the ticker's fiscal calendar,
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
    durations = facts[~facts["duration_type"].isin([INSTANT, OTHER_SHAPE])]
    # One calendar for the whole ticker, built from every field's annual-shaped facts.
    if year_ends is None:
        year_ends = fiscal_year_ends(durations)
    all_quarters, all_ttm = [], []
    for name, group in durations.groupby("field", sort=True):
        spec = catalogue.field(name)
        if spec.kind != "duration":
            continue
        field_refusals: list[dict] = []
        quarters = quarterize(group, spec, guards, year_ends, field_refusals)
        if refusals is not None:
            refusals.extend({**r, "field": name} for r in field_refusals)
        if quarters.empty:
            continue
        all_quarters.append(quarters)
        all_ttm.append(trailing_twelve(quarters, spec, annual=group, guards=guards))
    quarters = pd.concat(all_quarters, ignore_index=True) if all_quarters else pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    ttm = pd.concat(all_ttm, ignore_index=True) if all_ttm else pd.DataFrame()
    return quarters, ttm, instant_stock(facts)
