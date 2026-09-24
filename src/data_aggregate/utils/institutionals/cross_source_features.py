"""Availability-normalized institutional cross-source agreement features.

Each directional feature divides flagged families by families whose source and per-cell
dependencies are available. The ratio is emitted only with at least three available families;
its denominator is persisted as ``*_available_family_count`` so support stays observable.
The bullish side has four possible families and the bearish side has three.

The point is independent confirmation. A company officer, concentrated fund manager,
activist, or short-flow signal reaching the same conclusion is different from one source
being emphatic. Treating a source that did not yet exist as a zero would make identical
company behavior mean something different across time. Conflict is the conservative overlap
``min(bullish_ratio, bearish_ratio)`` when both exist.

Distinct bullish actors remain an exact trailing-window count. A NaN input contributes to
neither numerator nor denominator; an available zero remains a measured negative vote.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.xs import xs_rank_pct
from src.data_aggregate.utils.institutionals.decay import snap_to_grid
from src.data_aggregate.utils.institutionals.sink import (
    BEARISH_INPUTS,
    BULLISH_INPUTS,
    NEGATED_INPUTS,
    AvailableSignal,
)

logger = logging.getLogger(__name__)

#: Registry #83/#85: "above their 80th universe percentile".
PERCENTILE = 0.80

#: Trading days over which #84 counts distinct actors. Two quarters -- the disclosure cadence
#: of the slowest contributing family (13F), so a manager who acted last quarter is still a
#: live vote when this quarter's Form 4s land.
ACTOR_WINDOW = 126

#: A directional ratio over fewer than this many available families is not comparable.
MIN_FAMILIES = 3

EMISSION: dict[str, str] = {
    "ic_xs_bullish_family_ratio": "raw",
    "ic_xs_bullish_available_family_count": "raw",
    "ic_xs_bullish_actor_count": "raw+xs",
    "ic_xs_bearish_family_ratio": "raw",
    "ic_xs_bearish_available_family_count": "raw",
    "ic_xs_conflict_ratio": "raw",
}


def _direction(
    signals: dict[str, AvailableSignal],
    inputs: dict[str, str],
    idx: pd.DatetimeIndex,
    columns: pd.Index,
    percentile: float,
    label: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return `(flagged / available, available)` for one direction."""
    flags: list[pd.DataFrame] = []
    masks: list[pd.DataFrame] = []
    resolved, missing = [], []
    for role, name in inputs.items():
        signal = signals.get(name)
        if signal is None or signal.values.empty:
            missing.append(f"{role} ({name})")
            continue
        wide = signal.values.reindex(index=idx, columns=columns)
        available = signal.available.reindex(index=idx, columns=columns, fill_value=False)
        if role in NEGATED_INPUTS:
            wide = -wide
        rank = xs_rank_pct(wide.where(available))
        masks.append(available)
        flags.append((rank > percentile).astype("float64").where(available))
        resolved.append(role)
    if not flags:
        logger.warning("`ic_xs_%s_family_ratio` NOT built: no inputs resolved", label)
        return None, None
    available_count = masks[0].astype("int16")
    for mask in masks[1:]:
        available_count = available_count + mask.astype("int16")
    total = flags[0].fillna(0.0)
    for f in flags[1:]:
        total = total + f.fillna(0.0)
    ratio = total.divide(available_count.where(available_count > 0)).where(available_count >= MIN_FAMILIES)
    if len(flags) < MIN_FAMILIES:
        logger.warning(
            "`ic_xs_%s_family_ratio` has only %s of %s inputs resolved; denominator is "
            "persisted but the ratio cannot meet the per-cell minimum (missing %s)",
            label,
            len(flags),
            len(inputs),
            missing or "-",
        )
    else:
        logger.info(
            "`ic_xs_%s_family_ratio`: %s families resolved %s%s",
            label,
            len(resolved),
            resolved,
            f"; missing {missing}" if missing else "",
        )
    return ratio, available_count.astype("float64")


def _rolling_distinct_actors(events: pd.DataFrame, idx: pd.DatetimeIndex, columns: pd.Index, window: int) -> pd.DataFrame:
    """Distinct `actor` per ticker with an event in the trailing `window` TRADING DAYS.

    A sweep line over the grid, keeping a per-(ticker, actor) occurrence count: an actor joins
    when their event lands and leaves `window` days later, and the ticker's distinct count
    moves only when a counter crosses 0 <-> 1. That makes the whole pass O(events + days) and,
    more importantly, EXACT -- the alternative (an `ffill(limit=window)` over a
    `(ticker, actor)` column MultiIndex) is 7,800 x ~20,000 cells for the insider family
    alone, which is 1.2 GB for a count of at most a few dozen.

    NaN before a ticker's first event, never 0: see the module docstring.
    """
    out = np.full((len(idx), len(columns)), np.nan)
    if events.empty:
        return pd.DataFrame(out, index=idx, columns=columns)
    col_of = {t: i for i, t in enumerate(columns)}
    e = events.dropna(subset=["ticker", "date", "actor"]).copy()
    e["_grid"] = snap_to_grid(e["date"], idx)
    e = e.dropna(subset=["_grid"])
    e = e[e["ticker"].isin(col_of)]
    if e.empty:
        return pd.DataFrame(out, index=idx, columns=columns)

    pos = idx.get_indexer(pd.DatetimeIndex(e["_grid"]))
    col = e["ticker"].map(col_of).to_numpy(dtype="int64")
    # ONE integer code per (ticker, actor) pair, so the sweep's book-keeping is array indexing
    # rather than dict hashing on a string built per row.
    pair, _ = pd.factorize(e["ticker"].astype(str) + "|" + e["actor"].astype(str))
    order = np.argsort(pos, kind="stable")
    pos, col, pair = pos[order], col[order], pair[order]

    counts = np.zeros(len(columns), dtype="int64")  # distinct actors, per ticker
    active = np.zeros(int(pair.max()) + 1, dtype="int64")  # events live per (ticker, actor)
    seen = np.zeros(len(columns), dtype=bool)
    enter = leave = 0
    n = len(pos)
    for k in range(len(idx)):
        while enter < n and pos[enter] <= k:
            p = pair[enter]
            if active[p] == 0:
                counts[col[enter]] += 1
            active[p] += 1
            seen[col[enter]] = True
            enter += 1
        while leave < n and pos[leave] <= k - window:
            p = pair[leave]
            active[p] -= 1
            if active[p] == 0:
                counts[col[leave]] -= 1
            leave += 1
        out[k] = np.where(seen, counts, np.nan)
    return pd.DataFrame(out, index=idx, columns=columns)


def build_cross_source_panel(
    frames: PriceFrames,
    sink,
    *,
    percentile: float = PERCENTILE,
    actor_window: int = ACTOR_WINDOW,
) -> pd.DataFrame:
    """Long-format cross-source panel (`f_<name>` per `EMISSION`).

    `sink` is the `ConditioningSink` the source panels filled: its `signals` back the two
    family counts and its `events` (the `actor` column) back the distinct-actor count. Empty
    when neither direction resolves enough inputs.

    ⚠ `frames` RATHER THAN THREE UNPACKED FIELDS. `peer_dict`, `trading_index` and `universe`
    were all read off one `PriceFrames` at the call site. Collapsing them is not about the basis
    here -- this builder reads no wide price frame -- but about arity: three of the old
    parameters were one object at every call site, and unpacking them at 39 of those is what let
    them drift apart.

    ⚠ NO `frames.require(...)`: this builder dereferences no optional wide frame at all.
    `trading_index` and `peers` are non-Optional fields of `PriceFrames`, so requiring them
    would assert something the type already guarantees.

    The non-frame arguments are KEYWORD-ONLY. A positional slip between two same-typed
    `pd.DataFrame | None` neighbours is a silent wrong-frame bug that reads as a plausible
    call; the keyword form makes it unrepresentable.
    """
    peer_dict = frames.peers
    trading_index = frames.trading_index
    universe = pd.Index(frames.universe)
    # ⚠ NOT `to_day`, and deliberately so at all eight of these sites. `to_day` is for a
    # COLUMN -- it returns a Series -- while this normalizes an already-datetime INDEX, which
    # has no `.dt` accessor and must stay an index. The two are not interchangeable, so this
    # is not a site the `to_day` sweep missed.
    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    signals = getattr(sink, "signals", {}) or {}
    if idx.empty or (not signals and not getattr(sink, "actors", {})):
        return pd.DataFrame(columns=["date", "ticker"])

    if universe is not None and len(universe):
        columns = pd.Index(sorted(map(str, universe)), name="ticker")
    else:
        cols: set[str] = set()
        for signal in signals.values():
            cols |= set(map(str, signal.values.columns))
        columns = pd.Index(sorted(cols), name="ticker")
    if not len(columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields: dict[str, pd.DataFrame] = {}
    bull, bull_count = _direction(signals, BULLISH_INPUTS, idx, columns, percentile, "bullish")
    bear, bear_count = _direction(signals, BEARISH_INPUTS, idx, columns, percentile, "bearish")
    if bull_count is not None:
        fields["ic_xs_bullish_available_family_count"] = bull_count
    if bear_count is not None:
        fields["ic_xs_bearish_available_family_count"] = bear_count
    if bull is not None:
        fields["ic_xs_bullish_family_ratio"] = bull
    if bear is not None:
        fields["ic_xs_bearish_family_ratio"] = bear
    if bull is not None and bear is not None:
        fields["ic_xs_conflict_ratio"] = np.minimum(bull, bear).where(bull.notna() & bear.notna())

    # #84 -- distinct actors on the BULLISH side only. The bearish families have no comparable
    # actor axis: short flow has no identifiable actor at all (FINRA reports volume, not who
    # traded), so a bearish actor count would be three families wearing a four-family name.
    actors = getattr(sink, "actors", {}) or {}
    if actors:
        combined = pd.concat([e.assign(_fam=fam) for fam, e in actors.items()], ignore_index=True)
        # An actor id is only unique WITHIN its family (an owner CIK and a manager CIK are
        # both 10 digits), so the family prefixes the id before they are pooled.
        combined["actor"] = combined["_fam"].astype(str) + ":" + combined["actor"].astype(str)
        fields["ic_xs_bullish_actor_count"] = _rolling_distinct_actors(combined, idx, columns, actor_window)
        logger.info("`ic_xs_bullish_actor_count`: %s bullish acts across %s families %s", len(combined), combined["_fam"].nunique(), sorted(actors))

    for name in list(fields):
        frame = fields[name]
        if frame is None or frame.empty or not frame.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
