"""Availability support metadata for institutional source families.

The former agreement scores thresholded same-day cross-sectional percentiles, so their value
depended on which tickers happened to be present rather than on an interpretable economic
quantity. Only the observable family counts remain. They describe support, not alpha.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.sink import (
    BEARISH_INPUTS,
    BULLISH_INPUTS,
    AvailableSignal,
)

logger = logging.getLogger(__name__)

EMISSION: dict[str, str] = {
    "ic_xs_bullish_available_family_count": "raw",
    "ic_xs_bearish_available_family_count": "raw",
}


def _available_count(
    signals: dict[str, AvailableSignal],
    inputs: dict[str, str],
    idx: pd.DatetimeIndex,
    columns: pd.Index,
    label: str,
) -> pd.DataFrame | None:
    """Count source families whose inputs are observable in each cell."""
    masks: list[pd.DataFrame] = []
    resolved, missing = [], []
    for role, name in inputs.items():
        signal = signals.get(name)
        if signal is None or signal.available.empty:
            missing.append(f"{role} ({name})")
            continue
        available = signal.available.reindex(index=idx, columns=columns, fill_value=False)
        masks.append(available)
        resolved.append(role)
    if not masks:
        logger.warning("`ic_xs_%s_available_family_count` NOT built: no inputs resolved", label)
        return None
    available_count = masks[0].astype("int16")
    for mask in masks[1:]:
        available_count = available_count + mask.astype("int16")
    logger.info(
        "`ic_xs_%s_available_family_count`: %s of %s families resolved%s",
        label,
        len(resolved),
        len(inputs),
        f"; missing {missing}" if missing else "",
    )
    return available_count.astype("float64")


def build_cross_source_panel(
    frames: PriceFrames,
    sink,
) -> pd.DataFrame:
    """Long-format cross-source panel (`f_<name>` per `EMISSION`).

    `sink` is the `ConditioningSink` the source panels filled. Its signal-availability masks
    back the two support counts. Empty when neither direction resolves an input.

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
    if idx.empty or not signals:
        return pd.DataFrame(columns=["date", "ticker"])

    if universe is not None and len(universe):
        columns = pd.Index(sorted(map(str, universe)), name="ticker")
    else:
        cols: set[str] = set()
        for signal in signals.values():
            cols |= set(map(str, signal.available.columns))
        columns = pd.Index(sorted(cols), name="ticker")
    if not len(columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields: dict[str, pd.DataFrame] = {}
    bull_count = _available_count(signals, BULLISH_INPUTS, idx, columns, "bullish")
    bear_count = _available_count(signals, BEARISH_INPUTS, idx, columns, "bearish")
    if bull_count is not None:
        fields["ic_xs_bullish_available_family_count"] = bull_count
    if bear_count is not None:
        fields["ic_xs_bearish_available_family_count"] = bear_count
    for name in list(fields):
        frame = fields[name]
        if frame is None or frame.empty or not frame.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission, availability=frames.availability)
