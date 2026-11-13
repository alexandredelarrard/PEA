"""
panel.py  (src/data_aggregate/utils/common/panel.py)
---------------------------------------------------
The PEER-RELATIVE panel: turn a {name: daily wide frame} dict into the long feature
panel every cube part is made of, each characteristic expressed as

    f_<name>_vs_peers    z-score against the firm's direct competitors
    f_<name>_xs          percentile against the whole universe

Shared by all thirteen panel builders -- the single hottest symbol in the package.

The generic pieces this module used to also own now live beside it in `common/`: safe
division and inf-sanitizing in `frames.py`, and every per-day cross-sectional transform in
`xs.py` (which is where the five duplicate standardizers were merged). What is left here is
the one thing that is genuinely about PEERS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.xs import (
    HIST_MIN_PERIODS,
    HIST_WINDOW,
    Z_CLIP,
    self_history_z,
    xs_rank_pct,
    zscore,
)


def mask_to_availability(
    frame: pd.DataFrame,
    availability: pd.DataFrame | None,
) -> pd.DataFrame:
    """Mask a daily ticker frame to current own-price availability."""
    if availability is None or frame.empty:
        return frame
    active = availability.reindex(index=frame.index, columns=frame.columns, fill_value=False)
    return frame.where(active)


#: A peer statistic needs at least this many peers with a value on the day.
_MIN_PEERS = 3


def _peer_z(df_values: pd.DataFrame, ticker: str, peers: dict, floor: pd.Series) -> pd.Series | None:
    """Daily z-score of one ticker against its own peers; None when it has too few peers to ever score."""
    names = [peer for peer in peers if peer in df_values.columns]
    if ticker not in df_values.columns or len(names) < _MIN_PEERS:
        return None
    df_peers = df_values[names]
    weights = pd.Series({peer: float(peers[peer]) for peer in names}, dtype="float64")

    # Weight carried by the peers that report on each day: a missing peer drops out of both moments.
    present = df_peers.notna()
    day_weight = present.mul(weights, axis=1).sum(axis=1)
    usable = (present.sum(axis=1) >= _MIN_PEERS) & (day_weight > 0)
    day_weight = day_weight.where(usable)

    # Weighted mean and weighted standard deviation of the peers.
    mean = df_peers.mul(weights, axis=1).sum(axis=1).div(day_weight)
    variance = df_peers.sub(mean, axis=0).pow(2).mul(weights, axis=1).sum(axis=1).div(day_weight)
    std = np.sqrt(variance)

    # Peers that all report the same value cannot rank anything (NaN). The floor is applied after
    # that test, so only a negligible spread is floored, never an exactly-zero one.
    std = std.clip(lower=floor).where(std > 0)
    return zscore(df_values[ticker], mean, std)


def peer_relative(field_df: pd.DataFrame, peer_dict: dict, std_floor_ratio: float = 0.5) -> pd.DataFrame:
    """Score every ticker against its peers, per date: (X - weighted peer mean) / max(weighted peer std, `std_floor_ratio` x universe std), clipped at +-`Z_CLIP`.

    `field_df` is a (date x ticker) frame and `peer_dict` maps a ticker to {peer: weight}; the
    ticker itself is never in its own basket. The score is NaN when fewer than 3 peers have a value
    that day, or when the peers all share one value. Dividing by the peer std alone explodes when
    similar companies report near-identical values, so the std is floored at `std_floor_ratio` of
    the whole universe's std that day.

    Inputs are not trimmed; the output clip bounds a bad value's effect.
    """
    # The floor is a fraction of the day's universe dispersion (sample std, as in `xs.xs_z`).
    floor = field_df.std(axis=1) * std_floor_ratio

    # One score column per ticker that has a usable basket.
    df_z = pd.DataFrame(np.nan, index=field_df.index, columns=field_df.columns)
    for ticker, peers in peer_dict.items():
        z = _peer_z(field_df, ticker, peers, floor)
        if z is not None:
            df_z[ticker] = z

    # A non-finite score is missing, and is made so BEFORE the clip: a clip would turn an inf into a fabricated +-Z_CLIP.
    return df_z.replace([np.inf, -np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)


#: The emission modes a field may declare. The DEFAULT -- a field absent from the emission map
#: -- stays `_vs_peers` + `_xs` with no raw leg, which is what all thirteen existing builders
#: rely on. The three explicit modes exist for SPARSE / DECAYED signals, where a rank alone is
#: not a usable feature:
#:   "raw"        f_<name>                          the value itself, nothing else
#:   "raw+xs"     f_<name>, f_<name>_xs             + its universe percentile
#:   "raw+peers"  f_<name>, f_<name>_vs_peers       + its peer z-score
#:   "raw+hist"   f_<name>, f_<name>_vs_hist        + its trailing self-history z-score
#: Every explicit mode emits the raw leg: nothing ships as a rank alone (D27). A decayed
#: conviction weight in [0, ~n_managers] is a quantity LightGBM can split on directly, and
#: ranking it away throws that scale out.
_EMISSION_MODES = ("raw", "raw+xs", "raw+peers", "raw+hist")
_SEMANTIC_MODES = ("binary", "structural_zero")


@dataclass(frozen=True)
class SelfHistory:
    """Settings of the `raw+hist` leg and of the emitted date window.

    `prefix` is an earlier raw daily frame per field, used only as rolling context so an
    incremental caller can recompute a short current window; `output_since` then emits only
    that repaired tail.
    """

    window: int = HIST_WINDOW
    min_periods: int = HIST_MIN_PERIODS
    prefix: dict[str, pd.DataFrame] | None = None
    output_since: pd.Timestamp | None = None


@dataclass(frozen=True)
class PanelSettings:
    """Every declared difference between the callers of `build_peer_relative_panel`.

    `history` configures the `raw+hist` leg and the emitted date window. `peer_transform` maps a
    field to a variance-stabilising transform applied to the peer INPUT only (the raw leg keeps the
    value). `leg_order` replaces the default
    one-block-per-field layout by one block per leg kind, in the order given (`raw`, `vs_peers`,
    `xs`, `vs_hist`): each block drops its all-NaN legs and all-NaN rows, and the blocks are
    outer-merged on (date, ticker), which sorts the keys.
    """

    history: SelfHistory = field(default_factory=SelfHistory)
    peer_transform: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None
    leg_order: tuple[str, ...] | None = None


def _validate_declarations(fields: dict, emission: dict, semantics: dict) -> None:
    """Reject emission/semantics declarations that name an unknown mode or a field that does not exist."""
    unknown = {m for m in emission.values() if m not in _EMISSION_MODES}
    if unknown:
        raise ValueError(f"unknown emission mode(s) {sorted(unknown)}; expected one of {list(_EMISSION_MODES)}")

    # A name in the map that no field produces is a silent no-op: fail loudly rather than emit a
    # panel missing the column the caller thinks it declared.
    stray = set(emission) - set(fields)
    if stray:
        raise KeyError(f"emission declares field(s) not present in `fields`: {sorted(stray)}")
    unknown_semantics = {mode for mode in semantics.values() if mode not in _SEMANTIC_MODES}
    if unknown_semantics:
        raise ValueError(f"unknown semantic mode(s) {sorted(unknown_semantics)}; expected one of {list(_SEMANTIC_MODES)}")
    stray_semantics = set(semantics) - set(fields)
    if stray_semantics:
        raise KeyError(f"semantics declares field(s) not present in `fields`: {sorted(stray_semantics)}")


def _clean_field(fdf: pd.DataFrame | None, availability: pd.DataFrame | None) -> pd.DataFrame | None:
    """Numeric, availability-masked field frame; None when nothing usable is left."""
    if fdf is None or fdf.empty:
        return None

    # A stray Python None / object cell (a KPI absent for a name) must become NaN, or the
    # peer-z math and the `_xs` rank raise on it. Absent = NaN; a no-op on float frames.
    fdf = mask_to_availability(fdf.apply(pd.to_numeric, errors="coerce"), availability)
    if fdf.empty or not fdf.notna().any().any():
        return None
    return fdf


def _long_leg(frame: pd.DataFrame, column: str, output: pd.DataFrame | None = None) -> pd.Series:
    """Stack a (date x ticker) frame (optionally reindexed to `output`) into a float32 (date, ticker) column."""
    if output is not None:
        frame = frame.reindex(index=output.index, columns=output.columns)
    leg = frame.stack().astype("float32")
    leg.index.set_names(["date", "ticker"], inplace=True)
    return leg.rename(column)


def _peer_leg(
    comparable: pd.DataFrame,
    zero_state: pd.DataFrame | None,
    peer_dict: dict,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None,
) -> pd.DataFrame:
    """Peer z-score of the (optionally transformed) comparable values."""
    values = comparable if transform is None else transform(comparable)
    rel = peer_relative(values, peer_dict)
    return rel if zero_state is None else rel.where(~zero_state, 0.0)


def _xs_leg(fdf: pd.DataFrame, comparable: pd.DataFrame, zero_state: pd.DataFrame | None, binary: bool) -> pd.DataFrame:
    """Universe percentile of the comparable values; a binary field keeps its exact 0/1 state."""
    xs = fdf if binary else xs_rank_pct(comparable)
    return xs if zero_state is None else xs.where(~zero_state, 0.0)


def _hist_leg(name: str, fdf: pd.DataFrame, history: SelfHistory) -> pd.DataFrame:
    """Trailing self-history z-score, computed over the stored prefix (if any) followed by the current frame."""
    prefix = None if history.prefix is None else history.prefix.get(name)
    history_input = fdf
    if prefix is not None and not prefix.empty:
        history_input = pd.concat([prefix.apply(pd.to_numeric, errors="coerce"), fdf], axis=0).sort_index()
        history_input = history_input.loc[~history_input.index.duplicated(keep="last")]
    return self_history_z(
        history_input,
        window=history.window,
        min_periods=history.min_periods,
        output_since=history.output_since,
    )


def _field_legs(
    name: str,
    fdf: pd.DataFrame,
    mode: str | None,
    semantic: str | None,
    peer_dict: dict,
    settings: PanelSettings,
) -> list[tuple[str, pd.Series]]:
    """The (leg kind, long column) pairs of one cleaned field, in the order raw, vs_peers, xs, vs_hist."""
    history = settings.history
    output = fdf if history.output_since is None else fdf.loc[fdf.index >= pd.Timestamp(history.output_since)]
    if output.empty:
        return []

    # Structural zeros are a state: masked out of the comparable support, then restored as 0.
    zero_state = fdf.eq(0) if semantic == "structural_zero" else None
    comparable = fdf if zero_state is None else fdf.mask(zero_state)
    binary = semantic == "binary"

    # The raw leg is deliberately neither winsorized nor clipped: it is a quantity in its own units.
    legs: list[tuple[str, pd.Series]] = []
    if mode is not None:
        legs.append(("raw", _long_leg(output, f"f_{name}")))
    if mode is None or mode == "raw+peers":
        transform = (settings.peer_transform or {}).get(name)
        legs.append(("vs_peers", _long_leg(_peer_leg(comparable, zero_state, peer_dict, transform), f"f_{name}_vs_peers", output)))
    if mode is None or mode == "raw+xs":
        legs.append(("xs", _long_leg(_xs_leg(fdf, comparable, zero_state, binary), f"f_{name}_xs", output)))
    if mode == "raw+hist":
        legs.append(("vs_hist", _long_leg(_hist_leg(name, fdf, history), f"f_{name}_vs_hist", output)))
    return legs


def _merge_by_leg(feature_frame: pd.DataFrame, kinds: dict[str, str], leg_order: tuple[str, ...]) -> pd.DataFrame:
    """One block per leg kind (all-NaN legs and rows dropped), outer-merged on (date, ticker) in `leg_order`."""
    unlisted = set(kinds.values()) - set(leg_order)
    if unlisted:
        raise KeyError(f"leg_order does not list the emitted leg kind(s) {sorted(unlisted)}")

    # A block keeps the legs of one kind that hold a value, and the rows where one of them does.
    blocks: list[pd.DataFrame] = []
    for kind in leg_order:
        block = feature_frame.loc[:, [column for column in feature_frame.columns if kinds[column] == kind]]
        block = block.loc[:, block.notna().any()].dropna(how="all")
        if not block.empty:
            blocks.append(block.copy().reset_index())

    # The outer merge sorts the keys; a single block keeps its own order.
    if not blocks:
        return pd.DataFrame(columns=["date", "ticker"])
    merged = blocks[0]
    for block in blocks[1:]:
        merged = merged.merge(block, on=["date", "ticker"], how="outer")
    return merged


def _assemble(legs: list[tuple[str, pd.Series]], availability: pd.DataFrame | None, leg_order: tuple[str, ...] | None) -> pd.DataFrame:
    """Concat the legs, keep only available (date, ticker) keys, and turn the index into key columns."""
    # Concat keeps the independent float32 blocks (a `.copy()` would re-allocate every column), and
    # the key columns are built separately to avoid a fragmentation warning on a 100+ block frame.
    feature_frame = pd.concat([leg for _, leg in legs], axis=1)
    if availability is not None:
        active = availability.stack().reindex(feature_frame.index, fill_value=False).fillna(False)
        feature_frame = feature_frame.loc[active]
    if leg_order is not None:
        return _merge_by_leg(feature_frame, {str(leg.name): kind for kind, leg in legs}, leg_order)
    keys = feature_frame.index.to_frame(index=False)
    feature_frame.index = keys.index
    return pd.concat([keys, feature_frame], axis=1)


def build_peer_relative_panel(
    fields: dict,
    peer_dict: dict,
    emission: dict | None = None,
    semantics: dict | None = None,
    settings: PanelSettings | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn a {name: daily wide frame} dict into the long feature panel shared by every panel builder.

    By default each field becomes `f_<name>_vs_peers` and `f_<name>_xs`; `emission` overrides that
    per field with one of `_EMISSION_MODES`. `semantics` declares `binary` (exact 0/1 `_xs`) or
    `structural_zero` (zero kept as a state, only non-zero support compared). `settings`
    (`PanelSettings`) carries the history window, the per-field peer transform and the block
    layout, and `availability` masks both the input
    observations and the emitted (date, ticker) keys.
    """
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = emission or {}
    semantics = semantics or {}
    settings = settings or PanelSettings()
    _validate_declarations(fields, emission, semantics)

    # Clean each field, then build its legs.
    legs: list[tuple[str, pd.Series]] = []
    for name, raw_frame in fields.items():
        fdf = _clean_field(raw_frame, availability)
        if fdf is not None:
            legs += _field_legs(name, fdf, emission.get(name), semantics.get(name), peer_dict, settings)

    if not legs:
        return pd.DataFrame(columns=["date", "ticker"])
    return _assemble(legs, availability, settings.leg_order)
