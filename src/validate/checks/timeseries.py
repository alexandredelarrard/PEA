"""
timeseries.py  (src/validate/checks/timeseries.py)
--------------------------------------------------------------------------------------------
One ticker, one leg, along TIME: a step where the basis changed, a hole where a source dropped
a period, a stretch where the value does not move at all.

WHY THIS IS NOT A POOLED CHECK. A leg can have a clean profile, a clean null rate and a spread
cross-section and still be wrong on the time axis for one name, because a pooled p99 has no
idea which rows are adjacent. Every number here is computed over ONE ticker's own history and
over each leg's OWN support -- first to last non-null -- so a NaN before a name listed is not
a hole and a late-starting source costs nothing.

⚠ THE JUMP RULE HAS TWO GATES AND BOTH ARE LOAD-BEARING. The first run of `check_08` used the
modified z of the first difference alone and produced **296,926 "jumps" scoring up to 6.5e9**
on `cube_part_institutionals`. A decayed-event leg spends most of its life taking tiny decay
steps, so its MAD scale is near zero and ANY real move scores astronomically. The move must
therefore ALSO cover at least `jump_span_frac` of the leg's own p1-p99 range. And small-integer
legs are EXCLUDED rather than thresholded harder: on a leg whose whole range is {0, 1, 2, 3} a
single +1 step IS half the p1-p99 span by definition, so both gates fire on the feature
working. Jumps are reported at the `info` score and never turn a run red -- a real disclosure
legitimately moves these series.

⚠ FROZEN NEEDS A DECLARATION AND ABSTAINS WITHOUT ONE, AND `cadence: daily` IS NOT THAT
DECLARATION. Measured across six same-shaped cube parts, the mean number of legs frozen inside
a quarter runs from 0.1 of 28 on `cube_part_momentum` to **66.4 of 103** on
`cube_part_governance` -- governance is CORRECT, because it ffills quarterly proxy data onto a
daily grid. But "the table is daily, so every leg must move every day" is ALSO wrong, and
`cube_part_momentum` is the counter-example: every leg in that part is a cross-sectional
percentile (min 0.0020, max 1.0000, median 0.5012 on all of them), so a value only changes
when the name's RANK changes. Measured there, "every leg" produced 4,661 flat spells -- AAPL
holding `dollar_volume_63 == 1.0000` for 2,071 consecutive sessions because it was the largest
dollar volume in the universe on every one of them, MSFT holding `amihud_63 == 0.006135`
(rank 3 of 489) for 347. A stable rank is a stable value, and none of it is a stalled input.

    daily_legs declared          -> FROZEN runs on exactly those legs
    cadence: quarterly + horizon -> FROZEN runs on every leg above the declared ffill horizon
    anything else                -> FROZEN abstains; JUMP and HOLE still run, and it says so

⚠ AND IT READS PER TICKER FROM THE DB, NOT FROM THE SNAPSHOT. This is the one check that needs
one name's whole history contiguous and in order. The parquet snapshot is written in the
database's heap order, so every row group holds every ticker and a per-ticker read off disk
would either rescan the file 491 times or materialise the whole frame -- 6.6 GB on
`cube_part_fundamentals`. One ticker at a time from Postgres is 249 columns x ~7,800 rows.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import as_ts, feature_columns
from src.validate.result import CheckResult, Finding
from src.validate.spec import TableSpec, load_spec
from src.validate.utils.outliers import modified_zscore

log = logging.getLogger(__name__)

CHECK = "timeseries"

#: Non-null observations a (ticker, leg) series needs before it is judged at all. Below this
#: the median and MAD are decided by a handful of points, and `check_08` used the same 20.
_MIN_SUPPORT = 20

#: Findings filed per kind. The counts, and the worst examples, live in `metrics`.
_MAX_FINDINGS = 15

#: Examples carried in `metrics` per kind. Enough to act on, not a second copy of the table.
_MAX_EXAMPLES = 30

_WHY_FROZEN = (
    "no `daily_legs`, so there is no leg the builder promises to recompute every "
    "session, and a flat run cannot be told from a hold. `cadence: daily` does "
    "not settle it: every leg of cube_part_momentum is a cross-sectional "
    "percentile on a daily table, and a name that holds its rank holds its value "
    "-- 4,661 spells measured there, AAPL at 1.0000 for 2,071 sessions"
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """`(start, length)` of every True run in `mask`.

    Vectorised rather than a `groupby` over a cumsum key: this is called once per (ticker,
    leg) pair, which is 122,259 times on `cube_part_fundamentals`.
    """
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e - s)) for s, e in zip(edges[0::2], edges[1::2], strict=False)]


def _frozen_legs(spec: TableSpec, columns: list[str]) -> tuple[list[str], int | None, str]:
    """`(legs, max_allowed_flat_days, how)` for the FROZEN sub-check. See the module docstring.

    `max_allowed_flat_days is None` means "this sub-check abstains"; the caller reports that
    rather than reporting zero frozen spells."""
    if spec.daily_legs:
        legs = [c for c in columns if spec.is_daily_leg(c)]
        return (
            legs,
            spec.frozen_min_days,
            (f"the {len(legs)} leg(s) the table declares as rebuilt every session " f"(`daily_legs`), at >= {spec.frozen_min_days} identical days"),
        )
    if spec.cadence == "quarterly":
        if spec.ffill_horizon_days is None:
            return (
                [],
                None,
                (
                    "ABSTAINED -- the table declares `cadence: quarterly` but no "
                    "`ffill_horizon_days`, and a quarterly leg held flat between filings is the "
                    "feature working. An invented horizon turns the whole part green or the "
                    "whole part red, and neither is a result"
                ),
            )
        return (
            list(columns),
            spec.ffill_horizon_days,
            (f"every leg, at > {spec.ffill_horizon_days} identical days -- the declared ffill " f"horizon for this quarterly table"),
        )
    return [], None, f"ABSTAINED -- {_WHY_FROZEN}"


def _measure(
    series: pd.Series, dates: pd.DatetimeIndex, spec: TableSpec, *, frozen_limit: int | None, hole_eligible: np.ndarray | None = None
) -> dict[str, list[dict[str, Any]]]:
    """JUMP, HOLE and FROZEN for one (ticker, leg) series, over the leg's own support."""
    found: dict[str, list[dict[str, Any]]] = {"jump": [], "hole": [], "frozen": [], "explained": []}
    # `to_numeric`, not `astype`: an entirely-NULL `double precision` column comes back as
    # `object` (see `frame._is_leg`), and those are exactly the legs worth reaching.
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    present = np.isfinite(values)
    if present.sum() < _MIN_SUPPORT:
        return found

    # The leg's OWN support: a NaN before the first value is not a hole, it is a leg that did
    # not exist yet.
    lo, hi = int(np.argmax(present)), int(len(present) - 1 - np.argmax(present[::-1]))
    support, stamps = values[lo : hi + 1], dates[lo : hi + 1]
    live = support[np.isfinite(support)]

    # -- JUMP: modified z of the CHANGE, and the change must be material for this leg ----- #
    span = float(np.quantile(live, 0.99) - np.quantile(live, 0.01))
    if span > 0 and np.unique(live).size > spec.jump_min_distinct:
        change = np.diff(support, prepend=np.nan)
        score = np.nan_to_num(modified_zscore(change))
        material = np.abs(np.nan_to_num(change)) >= spec.jump_span_frac * span
        for k in np.flatnonzero((score > spec.jump_z) & material):
            found["jump"].append(
                {
                    "date": stamps[k],
                    "value": float(support[k]),
                    "previous": float(support[k - 1]) if k else float("nan"),
                    "change": float(change[k]),
                    "z": round(float(score[k]), 1),
                    "p1_p99_span": round(span, 6),
                    "change_over_span": round(abs(float(change[k])) / span, 2),
                }
            )

    # -- HOLE: a run of absent values strictly inside the support ------------------------- #
    missing = ~np.isfinite(support)
    eligible = np.asarray(hole_eligible[lo : hi + 1], dtype=bool) if hole_eligible is not None else np.ones(len(support), dtype=bool)
    for start, length in _runs(missing & eligible):
        if length >= spec.hole_min_days:
            found["hole"].append({"days": length, "start": stamps[start], "end": stamps[start + length - 1]})
    if hole_eligible is not None:
        for start, length in _runs(missing & ~eligible):
            if length >= spec.hole_min_days:
                found["explained"].append(
                    {
                        "days": length,
                        "start": stamps[start],
                        "end": stamps[start + length - 1],
                        "reason": "the declared economic precondition is inactive",
                    }
                )

    # -- FROZEN: a run of identical values, counted over the observations that exist ------ #
    #
    # ⚠ A RUN PINNED AT THE SERIES' OWN EXTREME IS A SATURATION, NOT A FREEZE, and separating
    # the two is the difference between 4,661 findings and a readable result. Measured on
    # `cube_part_momentum`: every leg there is a cross-sectional percentile (min 0.0020, max
    # 1.0000, median 0.5012 on all of them), so AAPL -- the largest dollar volume in the
    # universe for 2,071 consecutive sessions -- carries `dollar_volume_63 == 1.0000` on every
    # one of them. That is the rank working. A leg stuck at an INTERIOR value is the stalled
    # input FROZEN exists to find; a leg pinned at its own extreme is a question about the
    # ranking's resolution, which is `clip`'s tie mass. Both are recorded; only the interior
    # one turns a run red.
    if frozen_limit is not None:
        kept = np.flatnonzero(np.isfinite(support))
        observed = support[kept]
        low, high = float(observed.min()), float(observed.max())
        varied = np.unique(observed).size > 1
        for start, length in _runs(observed[1:] == observed[:-1]):
            flat = length + 1
            if flat >= frozen_limit:
                value = float(observed[start])
                found["frozen"].append(
                    {
                        "days": flat,
                        "value": value,
                        "at_extreme": bool(varied and value in (low, high)),
                        "start": stamps[kept[start]],
                        "end": stamps[kept[start + length]],
                    }
                )
    return found


def _hole_condition(frame: pd.DataFrame, leg: str, spec: TableSpec) -> np.ndarray | None:
    rule = spec.conditional_holes.get(leg)
    if rule is None:
        return None
    active_field = rule["active_field"]
    if active_field not in frame.columns:
        raise KeyError(f"{spec.table}.conditional_holes.{leg} requires live field {active_field!r}")
    values = pd.to_numeric(frame[active_field], errors="coerce").to_numpy(dtype="float64")
    active = np.isfinite(values)
    if "min_value" in rule:
        active &= values >= float(rule["min_value"])
    if "max_value" in rule:
        active &= values <= float(rule["max_value"])
    return active


def _known_ineligible_reason(
    spec: TableSpec,
    *,
    ticker: str,
    leg: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str | None:
    for rule in spec.known_ineligible:
        if ticker != rule["ticker"]:
            continue
        if not any(leg.startswith(pattern) or leg.endswith(pattern) for pattern in rule["fields"]):
            continue
        declared_start = pd.Timestamp(rule["start"]).normalize()
        declared_end = pd.Timestamp(rule["end"]).normalize()
        if declared_start <= pd.Timestamp(start).normalize() and pd.Timestamp(end).normalize() <= declared_end:
            return rule["reason"]
    return None


def _cluster(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Per leg: how many tickers, how many occurrences, and the worst one by `key`."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = out.setdefault(row["leg"], {"n": 0, "tickers": set(), "worst": None})
        entry["n"] += 1
        entry["tickers"].add(row["ticker"])
        if entry["worst"] is None or row[key] > entry["worst"][key]:
            entry["worst"] = row
    for entry in out.values():
        entry["tickers"] = sorted(entry["tickers"])
    return out


def check_timeseries(
    context: Context, table: Table | str, *, config: DictConfig, cache: Any = None, tickers: list[str] | None = None, **kwargs: Any
) -> CheckResult:
    """Per (ticker, leg): jumps, holes and frozen spells over the leg's own support."""
    spec_t = resolve(table)
    spec = load_spec(config, spec_t)
    live = context.store.columns(spec_t)
    date_col = spec_t.date_col
    ticker_col = spec_t.ticker_col if spec_t.ticker_col in live else None
    if date_col is None or ticker_col is None:
        return CheckResult.abstained(
            CHECK, spec_t.name, f"this is a (ticker x time) question and the table declares " f"date_col={date_col!r}, ticker_col={ticker_col!r}"
        )

    columns = feature_columns(context, spec_t)
    if not columns:
        return CheckResult.abstained(CHECK, spec_t.name, "no numeric leg to measure")

    names = [t.strip().upper() for t in tickers] if tickers else sorted(str(t).strip().upper() for t in context.store.distinct(spec_t, ticker_col))
    if not names:
        return CheckResult.abstained(CHECK, spec_t.name, "the table carries no ticker")

    frozen_legs, frozen_limit, frozen_how = _frozen_legs(spec, columns)
    frozen_set = set(frozen_legs)

    jumps: list[dict[str, Any]] = []
    holes: list[dict[str, Any]] = []
    explained_holes: list[dict[str, Any]] = []
    frozen: list[dict[str, Any]] = []
    rows = 0
    measured_tickers = 0
    for index, ticker in enumerate(names, start=1):
        frame = context.store.load(spec_t, columns=[date_col] + columns, where={ticker_col: ticker}, order_by=date_col, optional=True)
        if frame is None or frame.empty:
            continue
        measured_tickers += 1
        rows += len(frame)
        stamps = pd.DatetimeIndex(as_ts(frame[date_col]))
        for leg in columns:
            found = _measure(
                frame[leg], stamps, spec, frozen_limit=frozen_limit if leg in frozen_set else None, hole_eligible=_hole_condition(frame, leg, spec)
            )
            for kind, bucket in (("jump", jumps), ("hole", holes), ("frozen", frozen)):
                for row in found[kind]:
                    enriched = {"ticker": ticker, "leg": leg, **row}
                    if kind == "hole":
                        reason = _known_ineligible_reason(
                            spec,
                            ticker=ticker,
                            leg=leg,
                            start=row["start"],
                            end=row["end"],
                        )
                        if reason is not None:
                            explained_holes.append({**enriched, "reason": reason})
                            continue
                    bucket.append(enriched)
            for row in found["explained"]:
                explained_holes.append({"ticker": ticker, "leg": leg, **row})
        if index % 25 == 0 or index == len(names):
            log.info(
                "timeseries %s: %d/%d tickers, %s jump / %s hole / %s frozen",
                spec_t.name,
                index,
                len(names),
                f"{len(jumps):,}",
                f"{len(holes):,}",
                f"{len(frozen):,}",
            )

    if not measured_tickers:
        return CheckResult.abstained(CHECK, spec_t.name, f"none of the {len(names)} requested ticker(s) has a row")

    stalled = [row for row in frozen if not row["at_extreme"]]
    saturated = [row for row in frozen if row["at_extreme"]]
    by_leg_hole = _cluster(holes, "days")
    by_leg_explained = _cluster(explained_holes, "days")
    by_leg_frozen = _cluster(stalled, "days")
    by_leg_saturated = _cluster(saturated, "days")
    by_leg_jump = _cluster(jumps, "z")

    findings: list[Finding] = []
    for leg, entry in sorted(by_leg_hole.items(), key=lambda kv: -kv[1]["worst"]["days"])[:_MAX_FINDINGS]:
        worst = entry["worst"]
        findings.append(
            Finding.at(
                6,
                field=leg,
                ticker=worst["ticker"],
                observed=f"{entry['n']:,} gap(s) of >= {spec.hole_min_days} rows inside the leg's "
                f"own support across {len(entry['tickers'])} ticker(s); worst "
                f"{worst['days']:,} rows on {worst['ticker']} from "
                f"{pd.Timestamp(worst['start']).date()} to "
                f"{pd.Timestamp(worst['end']).date()}",
                expected="no gap inside a leg's own support. Absent rows before the first value "
                "are the source's start date and are not counted here; missing ROWS "
                "rather than missing values are `coverage`'s finding",
                n=entry["n"],
                tickers=entry["tickers"][:20],
                worst=worst,
            )
        )

    for leg, entry in sorted(by_leg_explained.items(), key=lambda kv: -kv[1]["worst"]["days"])[:_MAX_FINDINGS]:
        worst = entry["worst"]
        findings.append(
            Finding.at(
                3,
                field=leg,
                ticker=worst["ticker"],
                observed=f"{entry['n']:,} gap(s) across {len(entry['tickers'])} ticker(s) match "
                f"a declared conditional or source-ineligible interval; worst "
                f"{worst['days']:,} rows on {worst['ticker']} from "
                f"{pd.Timestamp(worst['start']).date()} to "
                f"{pd.Timestamp(worst['end']).date()}",
                expected="reported, not failed: NaN is the correct value while the economic "
                "precondition is inactive or the named source interval is ineligible. "
                "Any gap outside the exact declaration remains a score-6 hole",
                n=entry["n"],
                tickers=entry["tickers"][:20],
                worst=worst,
            )
        )

    for leg, entry in sorted(by_leg_frozen.items(), key=lambda kv: -kv[1]["worst"]["days"])[:_MAX_FINDINGS]:
        worst = entry["worst"]
        findings.append(
            Finding.at(
                7,
                field=leg,
                ticker=worst["ticker"],
                observed=f"{entry['n']:,} flat spell(s) across {len(entry['tickers'])} ticker(s); "
                f"worst {worst['days']:,} identical days at {worst['value']:g} on "
                f"{worst['ticker']} from {pd.Timestamp(worst['start']).date()} to "
                f"{pd.Timestamp(worst['end']).date()}",
                expected=f"FROZEN was applied to {frozen_how}. A leg that stops moving there is "
                f"a stalled input or a builder holding a stale value, not a cadence",
                n=entry["n"],
                tickers=entry["tickers"][:20],
                worst=worst,
            )
        )

    # Saturation is INFO by design, for the reason given in `_measure`: a top-ranked name
    # holding the top rank is the feature working, and it must be visible without being a
    # defect. Same contract as `coverage`'s declared exclusions.
    for leg, entry in sorted(by_leg_saturated.items(), key=lambda kv: -kv[1]["worst"]["days"])[:_MAX_FINDINGS]:
        worst = entry["worst"]
        findings.append(
            Finding.at(
                3,
                field=leg,
                ticker=worst["ticker"],
                observed=f"{entry['n']:,} flat spell(s) sitting on the series' OWN extreme across "
                f"{len(entry['tickers'])} ticker(s); worst {worst['days']:,} days at "
                f"{worst['value']:g} on {worst['ticker']} from "
                f"{pd.Timestamp(worst['start']).date()} to "
                f"{pd.Timestamp(worst['end']).date()}",
                expected="reported, not failed. A name that holds the top (or bottom) of a "
                "cross-section for years carries the same value for years, and that is "
                "the ranking working -- how much of the cross-section shares one value "
                "is `clip`'s tie mass, not a stalled input",
                n=entry["n"],
                tickers=entry["tickers"][:20],
                worst=worst,
            )
        )

    # Jumps are INFO by design: a real disclosure legitimately moves these series, so they are
    # ranked and reported and must not turn a run red. See the module docstring.
    for leg, entry in sorted(by_leg_jump.items(), key=lambda kv: -kv[1]["worst"]["z"])[:_MAX_FINDINGS]:
        worst = entry["worst"]
        findings.append(
            Finding.at(
                3,
                field=leg,
                ticker=worst["ticker"],
                observed=f"{entry['n']:,} step(s) clearing BOTH gates across "
                f"{len(entry['tickers'])} ticker(s); worst z = {worst['z']:,.1f} on "
                f"{worst['ticker']} at {pd.Timestamp(worst['date']).date()}, a change of "
                f"{worst['change']:,.6g} = {worst['change_over_span']:.2f}x the leg's own "
                f"p1-p99 span",
                expected=f"reported, not failed: a step is only a defect if no event explains it. "
                f"Both gates had to fire -- modified z of the change > {spec.jump_z} AND "
                f"the change >= {spec.jump_span_frac:.0%} of the leg's p1-p99 span -- "
                f"because the z gate alone produced 296,926 of these",
                n=entry["n"],
                tickers=entry["tickers"][:20],
                worst=worst,
            )
        )

    first_date, last_date = context.store.bounds(spec_t)
    scope = {
        "rows": rows,
        "tickers": measured_tickers,
        "first_date": first_date,
        "last_date": last_date,
        "legs": len(columns),
        "series": measured_tickers * len(columns),
        "jump_z": spec.jump_z,
        "jump_span_frac": spec.jump_span_frac,
        "jump_min_distinct": spec.jump_min_distinct,
        "hole_min_days": spec.hole_min_days,
        "min_support": _MIN_SUPPORT,
        "cadence": spec.cadence,
        "frozen_legs": len(frozen_legs),
        "frozen_limit_days": frozen_limit,
        "frozen_scope": frozen_how,
        "source": "db(per ticker)",
    }
    metrics = {
        "n_jumps": len(jumps),
        "n_holes": len(holes),
        "n_explained_holes": len(explained_holes),
        "n_frozen": len(stalled),
        "n_frozen_at_extreme": len(saturated),
        "legs_with_holes": sorted(by_leg_hole),
        "legs_with_frozen": sorted(by_leg_frozen),
        "legs_saturated": sorted(by_leg_saturated),
        "legs_with_jumps": sorted(by_leg_jump),
        "worst_holes": sorted(holes, key=lambda r: -r["days"])[:_MAX_EXAMPLES],
        "worst_explained_holes": sorted(explained_holes, key=lambda r: -r["days"])[:_MAX_EXAMPLES],
        "worst_frozen": sorted(stalled, key=lambda r: -r["days"])[:_MAX_EXAMPLES],
        "worst_saturated": sorted(saturated, key=lambda r: -r["days"])[:_MAX_EXAMPLES],
        "worst_jumps": sorted(jumps, key=lambda r: -r["z"])[:_MAX_EXAMPLES],
    }
    reason = "" if frozen_limit is not None else f"FROZEN {frozen_how}"
    return CheckResult.measured(CHECK, spec_t.name, findings, scope=scope, metrics=metrics, reason=reason)
