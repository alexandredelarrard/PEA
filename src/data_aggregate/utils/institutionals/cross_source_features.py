"""
cross_source_features.py  (src/data_aggregate/utils/institutionals/cross_source_features.py)
--------------------------------------------------------------------------------------------
CROSS-SOURCE AGREEMENT (`ic_xs_*`) -- registry section 8, features #83-#86.

    ic_xs_bullish_family_count  how many of four INDEPENDENT bullish families are flagging
    ic_xs_bullish_actor_count   how many distinct ACTORS acted bullishly in the window
    ic_xs_bearish_family_count  the same over four explicit NEGATIVE actions
    ic_xs_conflict              both sides at once: `min(bull, bear)` where both are >= 2

THE POINT IS INDEPENDENCE, WHICH IS WHY THESE ARE COUNTS AND NOT A SCORE. Four unrelated
parties -- a company officer, a concentrated fund manager, an activist, and the options/short
market -- reaching the same conclusion about one name is evidence of a different kind from any
one of them being emphatic. Averaging them into an `alignment_score` destroys exactly that
distinction, and the standing no-composites rule still governs SCORES (D5 relaxed the feature
COUNT for this family, not the principle). One conflict flag is the only interaction shipped.

⚠ TWO RULES THE REGISTRY IS EXPLICIT ABOUT, and both are easy to get wrong:

  1. **DISTINCT ACTORS, NOT ROWS** (#84, report section 3.1.6). Ten Form 4s from one CEO are
     one opinion, and a manager who files an amendment has not changed their mind twice. #84
     counts distinct `actor` ids (owner CIK / manager CIK / activist filer id) with an event
     in the trailing `ACTOR_WINDOW`, EXACTLY -- never a decayed count, because a decayed
     distinct count is not a distinct count.
  2. **ABSENCE OF BUYING IS NOT A SHORT SIGNAL** (#85, report section 3.6.2). Every bearish
     member is somebody DOING something: a discretionary sale, a full exit, a stake reduction,
     short flow confirmed by a falling price. "Nobody bought" is not in the list, and a NaN
     input never contributes to either count.

THE FLAG IS A PER-DATE UNIVERSE PERCENTILE (`PERCENTILE`, the registry's 80th), not a fixed
threshold: every input is a different quantity in different units, several of them decayed
intensities whose scale drifts with event frequency, and a fixed cut on any of them would mean
something different in 2013 than in 2026. `xs_rank_pct` ranks only the cells that exist on the
date, so a family with no coverage on a name is silent rather than negative.

⚠ A COUNT OF NOTHING IS NaN, NOT ZERO. On a ticker-day where none of the four inputs has a
value, the count is NaN: "no family is flagging" and "nothing is known about this name yet"
are different facts, which is the same rule `decay.py` states for a never-yet-seen event. Where
at least one input exists the count is a real 0.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.xs import xs_rank_pct
from src.data_aggregate.utils.institutionals.decay import snap_to_grid
from src.data_aggregate.utils.institutionals.sink import (
    BEARISH_INPUTS, BULLISH_INPUTS, NEGATED_INPUTS,
)
from src.data_aggregate.utils.common.price_frames import PriceFrames

logger = logging.getLogger(__name__)

#: Registry #83/#85: "above their 80th universe percentile".
PERCENTILE = 0.80

#: Trading days over which #84 counts distinct actors. Two quarters -- the disclosure cadence
#: of the slowest contributing family (13F), so a manager who acted last quarter is still a
#: live vote when this quarter's Form 4s land.
ACTOR_WINDOW = 126

#: A "family count" over fewer than this many resolved inputs is not a family count. The
#: builder refuses to emit that direction rather than shipping a column that silently measures
#: one family -- see the sink's note on renames.
MIN_FAMILIES = 2

#: Registry #86: the conflict flag fires only when BOTH sides have at least this many families.
CONFLICT_MIN = 2

EMISSION: dict[str, str] = {
    "ic_xs_bullish_family_count": "raw",      # an integer 0-4, comparable everywhere
    "ic_xs_bullish_actor_count":  "raw+xs",   # an unbounded count: scale drifts with coverage
    "ic_xs_bearish_family_count": "raw",
    "ic_xs_conflict":             "raw",
}


def _direction(signals: dict[str, pd.DataFrame], inputs: dict[str, str],
               idx: pd.DatetimeIndex, columns: pd.Index,
               percentile: float, label: str) -> pd.DataFrame | None:
    """The family count for one direction: how many resolved inputs sit above their own
    per-date universe percentile. NaN where none of them has a value."""
    flags: list[pd.DataFrame] = []
    present: list[pd.DataFrame] = []
    resolved, missing = [], []
    for role, name in inputs.items():
        frame = signals.get(name)
        if frame is None or frame.empty:
            missing.append(f"{role} ({name})")
            continue
        wide = frame.reindex(index=idx, columns=columns)
        if role in NEGATED_INPUTS:
            wide = -wide
        rank = xs_rank_pct(wide)
        present.append(wide.notna())
        flags.append((rank > percentile).astype("float64").where(wide.notna()))
        resolved.append(role)
    if len(flags) < MIN_FAMILIES:
        logger.warning("`ic_xs_%s_family_count` NOT built: only %s of %s inputs resolved "
                       "(missing %s)", label, len(flags), len(inputs), missing or "-")
        return None
    logger.info("`ic_xs_%s_family_count`: %s families resolved %s%s", label, len(resolved),
                resolved, f"; missing {missing}" if missing else "")
    any_input = present[0]
    for p in present[1:]:
        any_input = any_input | p
    total = flags[0].fillna(0.0)
    for f in flags[1:]:
        total = total + f.fillna(0.0)
    return total.where(any_input)


def _rolling_distinct_actors(events: pd.DataFrame, idx: pd.DatetimeIndex,
                             columns: pd.Index, window: int) -> pd.DataFrame:
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

    counts = np.zeros(len(columns), dtype="int64")            # distinct actors, per ticker
    active = np.zeros(int(pair.max()) + 1, dtype="int64")     # events live per (ticker, actor)
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
        for frame in signals.values():
            cols |= set(map(str, frame.columns))
        columns = pd.Index(sorted(cols), name="ticker")
    if not len(columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields: dict[str, pd.DataFrame] = {}
    bull = _direction(signals, BULLISH_INPUTS, idx, columns, percentile, "bullish")
    bear = _direction(signals, BEARISH_INPUTS, idx, columns, percentile, "bearish")
    if bull is not None:
        fields["ic_xs_bullish_family_count"] = bull
    if bear is not None:
        fields["ic_xs_bearish_family_count"] = bear
    if bull is not None and bear is not None:
        both = (bull >= CONFLICT_MIN) & (bear >= CONFLICT_MIN)
        # The magnitude is the WEAKER side: two families against four is a conflict of
        # strength two, not four. Zero (not NaN) where the condition fails -- "no conflict" is
        # an observation wherever both counts exist.
        fields["ic_xs_conflict"] = pd.DataFrame(
            np.where(both, np.minimum(bull, bear), 0.0), index=idx, columns=columns
        ).where(bull.notna() & bear.notna())

    # #84 -- distinct actors on the BULLISH side only. The bearish families have no comparable
    # actor axis: short flow has no identifiable actor at all (FINRA reports volume, not who
    # traded), so a bearish actor count would be three families wearing a four-family name.
    actors = getattr(sink, "actors", {}) or {}
    if actors:
        combined = pd.concat([e.assign(_fam=fam) for fam, e in actors.items()],
                             ignore_index=True)
        # An actor id is only unique WITHIN its family (an owner CIK and a manager CIK are
        # both 10 digits), so the family prefixes the id before they are pooled.
        combined["actor"] = combined["_fam"].astype(str) + ":" + combined["actor"].astype(str)
        fields["ic_xs_bullish_actor_count"] = _rolling_distinct_actors(
            combined, idx, columns, actor_window)
        logger.info("`ic_xs_bullish_actor_count`: %s bullish acts across %s families %s",
                    len(combined), combined["_fam"].nunique(), sorted(actors))

    for name in list(fields):
        frame = fields[name]
        if frame is None or frame.empty or not frame.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
