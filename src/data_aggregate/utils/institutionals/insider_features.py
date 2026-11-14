"""Raw insider features on original-disclosure windows and public correction intervals.

Complete, corroborated inactivity can be zero; unsupported prices, quantities and account
balances mask only dependent fields. Owner ranks use strictly earlier visible disclosures.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.institutionals.insider_positions import cumulative_positions
from src.data_aggregate.utils.institutionals.insider_quality import (
    POP_C_SUITE,
    POP_MANAGEMENT,
    POP_TEN_PCT,
    account_other_flows,
    security_class,
)
from src.data_aggregate.utils.institutionals.short_flow_features import _proven_tenure_mask
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Tables

_log = logging.getLogger(__name__)

#: Trailing calendar-day windows of the dense features; the population features use `POPULATION_WINDOW` only.
WINDOW_60, WINDOW_120, WINDOW_180 = 60, 120, 180
WINDOWS = (WINDOW_60, WINDOW_120, WINDOW_180)
POPULATION_WINDOW = WINDOW_180

#: Window of the distinct-insider counts and of the breadth ratio built from them.
BREADTH_WINDOW = WINDOW_120

#: Disjoint insider populations (see `insider_quality.TITLE_POPULATIONS`): output suffix -> population code. Insiders in
#: none of them appear only in the all-insider features.
POPULATIONS: dict[str, int] = {"c_suite": POP_C_SUITE, "management": POP_MANAGEMENT, "ten_pct": POP_TEN_PCT}
#: Populations with a purchase_pct_prior feature: 10% owners mostly open zero-prior accounts, so their ratio is undefined.
PCT_POPULATIONS: dict[str, int] = {name: code for name, code in POPULATIONS.items() if name != "ten_pct"}

#: feature -> (population code or None for all insiders, trailing window days).
BUY_VALUE_FEATURES: dict[str, tuple[int | None, int]] = {
    **{f"ic_insider_buy_value_mcap_{days}d": (None, days) for days in WINDOWS},
    **{f"ic_insider_{name}_buy_mcap_{POPULATION_WINDOW}d": (code, POPULATION_WINDOW) for name, code in POPULATIONS.items()},
}
PURCHASE_PCT_FEATURES: dict[str, tuple[int | None, int]] = {
    **{f"ic_insider_purchase_pct_prior_{days}d": (None, days) for days in WINDOWS},
    **{f"ic_insider_{name}_purchase_pct_prior_{POPULATION_WINDOW}d": (code, POPULATION_WINDOW) for name, code in PCT_POPULATIONS.items()},
}
NET_BUY_RATIO_FEATURES: dict[str, int] = {f"ic_insider_net_buy_ratio_{days}d": days for days in WINDOWS}
DISTINCT_BUYERS = f"ic_insider_distinct_buyers_{BREADTH_WINDOW}d"
DISTINCT_SELLERS = f"ic_insider_distinct_sellers_{BREADTH_WINDOW}d"
BUYER_BREADTH = f"ic_insider_buyer_breadth_{BREADTH_WINDOW}d"

#: Insider magnitudes, ratios, counts, and event intensities stay in their economic units.
#: Same-day cross-sectional ranks would replace those units with universe-composition noise.
EMISSION: dict[str, str] = dict.fromkeys(
    [*BUY_VALUE_FEATURES, *PURCHASE_PCT_FEATURES, *NET_BUY_RATIO_FEATURES, DISTINCT_BUYERS, DISTINCT_SELLERS, BUYER_BREADTH], "raw"
)

#: No `_vs_peers` leg anywhere in this family, for the same reason as `ic_super_*` (D25): an
#: insider purchase is an event about ONE company, and expressing it relative to a peer
#: basket that mostly has no event at all measures the basket's emptiness, not the signal.

#: Hard cutoff: the first stored insider quarter. Emits NaN, never 0, before the date: a 0 would read as
#: "no insider bought", a claim the data cannot support for a year it does not cover.
INSIDER_FLOOR = pd.Timestamp("2006-01-03")


@dataclass(frozen=True)
class InsiderCoverage:
    """Public-source floor and the observation frontier: `complete_through` is the newest
    stored insider filing date (whole table, markers included); a quiet window after it is unknown."""

    complete_through: pd.Timestamp | None = None
    availability: InstitutionalAvailability | None = None
    insider_floor: pd.Timestamp = INSIDER_FLOOR
    identity_intervals: pd.DataFrame | None = None

    @classmethod
    def from_availability(
        cls, availability: InstitutionalAvailability | None, complete_through: pd.Timestamp | None, identity_intervals: pd.DataFrame | None = None
    ) -> InsiderCoverage:
        return cls(
            complete_through,
            availability,
            availability.source_date(Tables.insider_transactions) if availability is not None else INSIDER_FLOOR,
            identity_intervals,
        )


def build_insider_feature_panel(
    frames: PriceFrames,
    df_trades: pd.DataFrame,
    df_position_trades: pd.DataFrame,
    diagnostics: dict,
    df_mcap: pd.DataFrame | None,
    coverage: InsiderCoverage,
) -> pd.DataFrame:
    """Compute features from clean source-USD trades, separately rebased positions and dated coverage."""
    if "source_first" not in diagnostics:
        return _empty_panel()
    diag = diagnostics
    df_unpriced = diag.get("unpriced_events", pd.DataFrame(columns=["ticker", "day", "code"]))
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    if df_mcap is None or df_mcap.empty or not df_mcap.notna().any().any():
        _log.warning("Saved mcap_pit is unsupported -> active size-scaled insider contributions remain unknown; rebuild prices if evidence changed.")

    # Keep execution USD intact while preparing a separate action-date contribution.
    df_trades = _action_values(df_trades, df_mcap, frames, coverage.identity_intervals)
    # Reconstruct positions with disclosed shares even when execution dollars are unknown.
    # Trades flagged as disputed are left out of every sum and count, like the other excluded events.
    df_flagged = df_trades[df_trades["quality_uncertain"].fillna(False).astype(bool)]
    df_supported = df_trades.drop(index=df_flagged.index)
    df_buys = df_supported[df_supported["code"].eq("P")].copy()
    df_sells = df_supported[df_supported["code"].eq("S")].copy()
    # Missing dollars do not invalidate disclosed shares or post-trade holdings.
    df_other_flows = diag.get("other_flows", pd.DataFrame())
    df_positions = _daily_positions(df_position_trades, df_other_flows)

    # Calculate raw feature groups before applying source and identity masks.
    seen = _first_filing(df_buys, df_sells, idx)
    fields = {
        **_value_fields(df_buys, idx, df_mcap, seen),
        **_net_ratio_fields(df_buys, df_sells, idx, seen),
        **_breadth_fields(*_named_traders(df_buys, df_sells, df_unpriced), idx),
    }
    df_disputed = pd.concat(
        [diag.get("uncertain_events", pd.DataFrame()), diag.get("unsupported_events", pd.DataFrame()), diag.get("oversized_events", pd.DataFrame())],
        ignore_index=True,
    )
    for days in WINDOWS:
        names = {name: code for name, (code, window) in PURCHASE_PCT_FEATURES.items() if window == days}
        fields.update(cumulative_positions(df_position_trades, df_positions, df_other_flows, idx, days, names, _disputed_accounts(df_disputed, days)))
    fields = _gate_dates(fields, idx, coverage.insider_floor)

    # Establish observable quiet windows, preserving unsupported active disclosures; disputed trades carry no dollars.
    df_unsupported = diag.get("unsupported_events", pd.DataFrame())
    df_oversized = diag.get("oversized_events", pd.DataFrame())
    df_dollars = df_trades.assign(value=df_trades["value"].where(~df_trades.index.isin(df_flagged.index)))
    df_activity = pd.concat([df_dollars, df_unpriced, df_unsupported, df_oversized], ignore_index=True)
    _observe_quiet_windows(fields, df_activity, diag["source_first"], frames, coverage)

    # A window is unknown only where excluded events are active and no supported one is.
    sets = _support_sets(df_supported, df_flagged, df_unpriced, pd.concat([df_unsupported, df_oversized], ignore_index=True))
    _mask_unknown_windows(fields, _unknown_windows(sets, idx))

    # Emit the existing raw float32 fields with own-price availability.
    _log.info("insider panel: %s features from %s scoped transactions (%s buys, %s sells)", len(fields), len(df_trades), len(df_buys), len(df_sells))
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)


def _disputed_accounts(df_uncertain: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Direct accounts with a disputed or unsupported quantity and the interval of a `window_days` window it affects."""
    columns = ["ticker", "owner_cik", "start", "stop"]
    if df_uncertain.empty or "direct_indirect" not in df_uncertain:
        return pd.DataFrame(columns=columns)
    df_direct = df_uncertain[df_uncertain["direct_indirect"].eq("D") & df_uncertain["owner_cik"].ne("")]
    return pd.DataFrame(
        {
            "ticker": df_direct["ticker"],
            "owner_cik": df_direct["owner_cik"],
            "start": df_direct["visible_from"],
            "stop": _window_stop(df_direct, window_days),
        }
    )


def _gate_dates(fields: dict[str, pd.DataFrame], idx: pd.DatetimeIndex, insider_floor: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Preserve observed operands after their public source floor; inventory gates only inferred absence."""
    floor = pd.Series(idx >= insider_floor, index=idx)
    return {name: frame.where(floor, axis=0) for name, frame in fields.items() if frame is not None and not frame.empty}


def _listed(stock_close: pd.DataFrame | None, idx: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """True where the ticker has a close on the day; missing listing evidence is unknown."""
    if stock_close is not None and not stock_close.empty:
        return stock_close.reindex(index=idx, columns=columns).notna()
    return pd.DataFrame(False, index=idx, columns=columns)


def _net_buy_mask(
    idx: pd.DatetimeIndex,
    columns: pd.Index,
    listed: pd.DataFrame,
    insider_floor: pd.Timestamp,
    coverage: InsiderCoverage,
    window_days: int = WINDOW_180,
) -> pd.DataFrame:
    """Cells where an empty window is observable: listed, a full calendar window after the
    source floor, and on or before the table-wide frontier."""
    history_start = pd.Timestamp(insider_floor).normalize() + pd.Timedelta(days=window_days - 1)
    full_window = InstitutionalAvailability.date_mask(idx, columns, history_start)
    frontier = coverage.complete_through
    if frontier is None:
        observed_through = pd.DataFrame(False, index=idx, columns=columns)
    else:
        observed_through = InstitutionalAvailability.through_mask(idx, columns, frontier)
    if coverage.availability is not None:
        return coverage.availability.source_mask(Tables.insider_transactions, idx, columns, requirements=(listed, full_window, observed_through))
    source_started = InstitutionalAvailability.date_mask(idx, columns, insider_floor)
    return InstitutionalAvailability.combine(source_started, listed, full_window, observed_through)


def _observe_quiet_windows(
    fields: dict[str, pd.DataFrame],
    activity: pd.DataFrame,
    source_first: pd.Series,
    frames: PriceFrames,
    coverage: InsiderCoverage,
) -> None:
    """Fill only proven inactivity; unsupported active rows never become zero.

    Any issuer disclosure corroborates coverage prospectively. The table-wide frontier, a full
    named source window and current listing are also necessary.
    """
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    columns = pd.Index(frames.universe, name="ticker")
    listed = _listed(frames.close_split, idx, columns)
    frontier = coverage.complete_through
    seen = pd.DataFrame(True, index=idx, columns=columns)
    corroborated = pd.DataFrame({ticker: idx >= day for ticker, day in source_first.items()}, index=idx).reindex(columns=columns, fill_value=False)
    ratios = {*PURCHASE_PCT_FEATURES, *NET_BUY_RATIO_FEATURES, BUYER_BREADTH}
    for name, (window, selector) in _window_specifications(activity).items():
        active = _rolling(activity[selector], idx, window, None, seen).fillna(0).gt(0)
        if frontier is None:
            if name in fields:
                fields[name] = fields[name].reindex(index=idx, columns=columns).where(active)
            continue
        observed = _net_buy_mask(idx, columns, listed, coverage.insider_floor, coverage, window) & corroborated
        frame = fields.get(name, pd.DataFrame(index=idx)).reindex(index=idx, columns=columns)
        if name in NET_BUY_RATIO_FEATURES:
            bv = _rolling(activity[activity["code"].eq("P")], idx, window, "value", seen).fillna(0.0)
            sv = _rolling(activity[activity["code"].eq("S")], idx, window, "value", seen).fillna(0.0)
            verified = (bv - sv) / (bv + sv).where((bv + sv).gt(0))
            frame = frame.where(~observed, verified)
        fields[name] = (frame if name in ratios else frame.mask(observed & ~active, 0.0)).where(active | observed)


def _window_specifications(activity: pd.DataFrame) -> dict[str, tuple[int, pd.Series]]:
    """feature -> (window days, the activity rows that make its window observed)."""
    buy, sell = activity["code"].eq("P"), activity["code"].eq("S")
    traded = buy | sell
    specifications: dict[str, tuple[int, pd.Series]] = {}
    for name, (code, days) in BUY_VALUE_FEATURES.items():
        specifications[name] = (days, buy & _member(activity, code))
    for name, (code, days) in PURCHASE_PCT_FEATURES.items():
        specifications[name] = (days, traded & _member(activity, code))
    for name, days in NET_BUY_RATIO_FEATURES.items():
        specifications[name] = (days, traded)
    specifications[DISTINCT_BUYERS] = (BREADTH_WINDOW, buy)
    specifications[DISTINCT_SELLERS] = (BREADTH_WINDOW, sell)
    specifications[BUYER_BREADTH] = (BREADTH_WINDOW, traded)
    return specifications


def _member(rows: pd.DataFrame, code: int | None) -> pd.Series:
    """Rows of a population; every row for `None`."""
    if code is None:
        return pd.Series(True, index=rows.index)
    return rows["population"].eq(code)


# --------------------------------------------------------------------------- #
# dense features -- trailing calendar windows                                   #
# --------------------------------------------------------------------------- #


def _value_fields(buys: pd.DataFrame, idx: pd.DatetimeIndex, mcap: pd.DataFrame | None, seen: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Purchase dollars over market cap, summed over trailing calendar windows and sampled onto the trading grid,
    so a day only ever sees transactions already filed by it. A population is observed from its first purchase."""
    if mcap is None or mcap.empty or buys.empty:
        return {}
    out = {}
    for name, (code, days) in BUY_VALUE_FEATURES.items():
        selected = buys if code is None else buys[buys["population"].eq(code)]
        if selected.empty:
            continue
        first = selected.groupby("ticker")["visible_from"].min()
        eligible = seen.loc[:, first.index] & pd.DataFrame({ticker: idx >= day for ticker, day in first.items()}, index=idx)
        out[name] = _action_sum(selected, idx, days, eligible)
    return out


def _net_ratio_fields(buys: pd.DataFrame, sells: pd.DataFrame, idx: pd.DatetimeIndex, seen: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Buy minus sell dollars over their sum, bounded [-1, 1], per window.

    An unobserved opposite leg requires inventory proof; an actual 0/0 always stays undefined."""
    out = {}
    for name, days in NET_BUY_RATIO_FEATURES.items():
        bv = _rolling(buys, idx, days, "value", seen).where(_rolling(buys, idx, days, None, seen).gt(0))
        sv = _rolling(sells, idx, days, "value", seen).where(_rolling(sells, idx, days, None, seen).gt(0))
        denom = bv + sv
        out[name] = ((bv - sv) / denom.where(denom > 0)).replace([np.inf, -np.inf], np.nan)
    return out


def _first_filing(buys: pd.DataFrame, sells: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean (date x ticker): has this ticker filed ANY open-market Form 4 by this date?

    The coverage mask every windowed feature shares. It keys on purchases AND sales, so a name
    that only ever sold reads 0 (observed) on the buy features, not NaN (unknown); before a
    ticker's first visible filing every window is unknown.
    """
    df_both = pd.concat([buys[["visible_from", "ticker"]], sells[["visible_from", "ticker"]]])
    if df_both.empty:
        return pd.DataFrame(False, index=idx, columns=pd.Index([], name="ticker"))
    first = df_both.groupby("ticker")["visible_from"].min()
    df_grid = pd.DataFrame({t: idx >= d for t, d in first.items()}, index=idx)
    df_grid.columns.name = "ticker"
    return df_grid


def _plain(records: pd.DataFrame) -> pd.Series:
    """True for a record visible from its own anchor with no end: neither a restatement nor restated."""
    return records["visible_from"].eq(records["anchor"]) & records["visible_until"].isna()


def _grid_pos(dates: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first `idx` day on or after each date; `len(idx)` for NaT (never reached)."""
    pos = idx.searchsorted(pd.DatetimeIndex(dates), side="left")
    return np.where(dates.isna().to_numpy(), len(idx), pos)


def _window_stop(records: pd.DataFrame, window_days: int | None) -> pd.Series:
    """The exclusive last day a record counts in a trailing `window_days` window: `anchor + window`, cut at `visible_until`."""
    if window_days is None:
        return records["visible_until"]
    window_end = records["anchor"] + pd.Timedelta(days=window_days)
    return window_end.where(records["visible_until"].isna() | (window_end <= records["visible_until"]), records["visible_until"])


def _spread(start: np.ndarray, stop: np.ndarray, col: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each `[start, stop)` grid interval into its (row, col, value) cells."""
    length = np.clip(stop - start, 0, None)
    offset = np.arange(int(length.sum())) - np.repeat(np.cumsum(length) - length, length)
    rows = np.repeat(start, length) + offset
    cols = np.repeat(col, length)
    vals = np.repeat(values, length)
    return rows, cols, vals


def _rolling(txns: pd.DataFrame, idx: pd.DatetimeIndex, window_days: int | None, value_col: str | None, seen: pd.DataFrame) -> pd.DataFrame:
    """Trailing `window_days`-calendar-day sum per ticker, sampled onto `idx`.

    Plain records are zero-filled between filing days on a DAILY calendar and rolled, so the
    total counts the window's real length; every other record counts on the days `d` with
    `visible_from <= d < min(visible_until, anchor + window)`. Widened to every ticker in
    `seen` and masked before each one's first filing.
    """
    if txns.empty:
        return pd.DataFrame(np.nan, index=idx, columns=seen.columns)
    plain = _plain(txns) if window_days is not None else pd.Series(False, index=txns.index)
    df_summed = pd.DataFrame(0.0, index=idx, columns=seen.columns)
    if plain.any():
        df_plain = txns[plain]
        if value_col:
            df_daily = df_plain.groupby(["anchor", "ticker"])[value_col].sum().unstack("ticker")
        else:
            df_daily = df_plain.groupby(["anchor", "ticker"]).size().unstack("ticker")
        calendar = pd.date_range(min(df_daily.index.min(), idx.min()), max(df_daily.index.max(), idx.max()), freq="D")
        df_daily = df_daily.reindex(calendar).fillna(0.0).rolling(f"{window_days}D").sum()
        df_summed = df_daily.reindex(index=idx, columns=seen.columns).fillna(0.0)
    if not plain.all():
        df_summed = df_summed + _late_window_sum(txns[~plain], idx, window_days, value_col, seen.columns)
    return df_summed.where(seen)


def _late_window_sum(records: pd.DataFrame, idx: pd.DatetimeIndex, window_days: int | None, value_col: str | None, columns: pd.Index) -> pd.DataFrame:
    """Window sum of non-plain records: each counts on `visible_from <= d < min(visible_until, anchor + window)`."""
    stop = _window_stop(records, window_days)
    col = columns.get_indexer(pd.Index(records["ticker"].astype(str)))
    values = pd.to_numeric(records[value_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64") if value_col else np.ones(len(records))
    known = col >= 0
    rows, cols, vals = _spread(_grid_pos(records["visible_from"], idx)[known], _grid_pos(stop, idx)[known], col[known], values[known])
    out = np.zeros((len(idx), len(columns)), dtype="float64")
    np.add.at(out, (rows, cols), vals)
    return pd.DataFrame(out, index=idx, columns=columns)


def _named_traders(buys: pd.DataFrame, sells: pd.DataFrame, unpriced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buys and sells with an owner identity, unpriced ones included: a distinct owner needs no price."""
    traders = []
    for df_side, code in ((buys, "P"), (sells, "S")):
        df_all = pd.concat([df_side, unpriced[unpriced["code"].eq(code)]], ignore_index=True) if not unpriced.empty else df_side
        traders.append(df_all[df_all["owner_cik"].ne("")])
    return traders[0], traders[1]


def _support_sets(
    supported: pd.DataFrame, flagged: pd.DataFrame, unpriced: pd.DataFrame, unsupported: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per kind of feature, (events it leaves out, events it can use): dollars over cap, dollar ratios, distinct owners."""
    no_cap = supported[supported["action_value"].isna()]
    no_price = pd.concat([flagged, unpriced, unsupported], ignore_index=True)
    no_owner = pd.concat(
        [supported[supported["owner_cik"].eq("")], unpriced[unpriced["owner_cik"].eq("")]]
        if not unpriced.empty
        else [supported[supported["owner_cik"].eq("")]]
    )
    return {
        "value": (pd.concat([no_price, no_cap], ignore_index=True), supported[supported["action_value"].notna()]),
        "ratio": (no_price, supported),
        "owner": (
            pd.concat([flagged, unsupported, no_owner], ignore_index=True),
            pd.concat(
                [supported[supported["owner_cik"].ne("")], unpriced[unpriced["owner_cik"].ne("")]]
                if not unpriced.empty
                else [supported[supported["owner_cik"].ne("")]]
            ),
        ),
    }


def _unknown_windows(sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]], idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Feature windows that hold an excluded event and no supported one."""
    tickers = sorted({str(t) for pair in sets.values() for frame in pair if not frame.empty for t in frame["ticker"].unique()})
    if not tickers:
        return {}
    seen = pd.DataFrame(True, index=idx, columns=pd.Index(tickers, name="ticker"))
    groups: list[tuple[str, str, tuple[str, ...], int, int | None]] = [
        *((name, "value", ("P",), days, code) for name, (code, days) in BUY_VALUE_FEATURES.items()),
        *((name, "ratio", ("P", "S"), days, None) for name, days in NET_BUY_RATIO_FEATURES.items()),
        (DISTINCT_BUYERS, "owner", ("P",), BREADTH_WINDOW, None),
        (DISTINCT_SELLERS, "owner", ("S",), BREADTH_WINDOW, None),
        (BUYER_BREADTH, "owner", ("P", "S"), BREADTH_WINDOW, None),
    ]
    masks: dict[str, pd.DataFrame] = {}
    for name, kind, codes, days, code in groups:
        left_out, usable = sets[kind]
        if left_out.empty:
            continue
        left_out = left_out[left_out["code"].isin(codes) & _member(left_out, code)]
        if left_out.empty:
            continue
        usable = usable[usable["code"].isin(codes) & _member(usable, code)]
        masks[name] = _rolling(left_out, idx, days, None, seen).gt(0) & ~_rolling(usable, idx, days, None, seen).gt(0)
    return masks


def _mask_unknown_windows(fields: dict[str, pd.DataFrame], masks: dict[str, pd.DataFrame]) -> None:
    for name, unknown in masks.items():
        if name in fields:
            field = fields[name]
            aligned = unknown.reindex(index=field.index, columns=field.columns, fill_value=False).astype(bool)
            fields[name] = field.mask(aligned)


def _action_values(trades: pd.DataFrame, mcap: pd.DataFrame | None, frames: PriceFrames, identity_intervals: pd.DataFrame | None) -> pd.DataFrame:
    """Attach `action_value = value / mcap_pit` on each trade's action session; clean execution dollars are unchanged.

    The session is the last trading day at or before the action, at most 7 days earlier, with the action already
    public; a non-session action of an unclassed title also needs one continuous canonical identity across both
    dates. A classed title without proved identity uses the ticker's own cap (ratio 1) and is flagged `class_approximated`.
    """
    df_trades = trades.copy()
    df_trades["action_session"] = pd.NaT
    df_trades["action_mcap"] = np.nan
    df_trades["action_value"] = np.nan
    df_trades["class_approximated"] = False
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    if df_trades.empty or mcap is None or mcap.empty or idx.empty or frames.close_pit is None:
        return df_trades

    # Select the action session on the authoritative calendar.
    action = pd.to_datetime(df_trades["transaction_day"], errors="coerce").dt.normalize()
    public_day = pd.to_datetime(df_trades["visible_from"], errors="coerce").dt.normalize()
    positions = idx.searchsorted(pd.DatetimeIndex(action), side="right") - 1
    rows = np.clip(positions, 0, len(idx) - 1)
    chosen = pd.Series(idx.take(rows), index=df_trades.index).where(action.notna() & (positions >= 0))
    df_trades["action_session"] = chosen
    supported = chosen.notna() & (action - chosen).le(pd.Timedelta(days=7)) & action.le(public_day)

    # Read the saved cap and as-traded quote on that session; both must be positive.
    columns = mcap.columns.get_indexer(pd.Index(df_trades["ticker"]))
    capital = mcap.reindex(index=idx).to_numpy(dtype=float)[rows, np.maximum(columns, 0)]
    price = frames.close_pit.reindex(index=idx, columns=mcap.columns).to_numpy(dtype=float)[rows, np.maximum(columns, 0)]
    supported &= (columns >= 0) & np.isfinite(capital) & (capital > 0) & np.isfinite(price) & (price > 0)

    # A non-session action needs one continuous canonical identity; an unproved class is approximated, never dropped.
    klass = security_class(df_trades["security_title"])
    needs_identity = action.ne(chosen) | klass.ne("COMMON")
    continuous = pd.Series(False, index=df_trades.index)
    if needs_identity.any():
        continuous.loc[needs_identity] = _action_identity(df_trades.loc[needs_identity], chosen.loc[needs_identity], identity_intervals)
    supported &= action.eq(chosen) | continuous | klass.ne("COMMON")
    df_trades["class_approximated"] = klass.ne("COMMON") & ~continuous

    # Divide clean execution dollars by the session cap.
    values = pd.to_numeric(df_trades["value"], errors="coerce")
    supported &= np.isfinite(values) & values.gt(0)
    df_trades["action_mcap"] = pd.Series(capital, index=df_trades.index).where(supported)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        contribution = values / df_trades["action_mcap"]
    df_trades["action_value"] = contribution.where(np.isfinite(contribution))
    return df_trades


def _action_identity(trades: pd.DataFrame, sessions: pd.Series, master: pd.DataFrame | None) -> pd.Series:
    """Require one canonical interval across both dates and the shared global ownership/conflict policy."""
    supported = pd.Series(False, index=trades.index)
    required = {
        "canonical_company",
        "source",
        "security_id",
        "source_symbol",
        "security_class",
        "conversion_ratio",
        "lineage_role",
        "valid_from",
        "valid_to",
    }
    if master is None or master.empty or not required.issubset(master):
        return supported
    df_pairs = pd.DataFrame(
        {
            "ticker": trades["ticker"],
            "action": trades["transaction_day"],
            "session": sessions,
            "klass": security_class(trades["security_title"]),
            "row": np.arange(len(trades)),
        },
        index=trades.index,
    ).dropna(subset=["action", "session"])
    if df_pairs.empty:
        return supported
    dates = pd.date_range(df_pairs["session"].min(), df_pairs["action"].max())
    columns = pd.Index(df_pairs["ticker"].unique(), name="ticker")
    tenure = _proven_tenure_mask(dates, columns, master, None)
    gaps = np.vstack([np.zeros(len(columns), dtype=int), (~tenure).to_numpy().cumsum(axis=0)])
    col = columns.get_indexer(df_pairs["ticker"])
    continuous = gaps[dates.get_indexer(df_pairs["action"]) + 1, col] == gaps[dates.get_indexer(df_pairs["session"]), col]
    df_pairs = df_pairs.loc[continuous]
    df_lines = master.loc[master["source"].eq("ftd") & master["lineage_role"].isin(CANONICAL_ROLES)].copy()
    ratio = pd.to_numeric(df_lines["conversion_ratio"], errors="coerce")
    df_lines = df_lines.loc[np.isfinite(ratio) & ratio.gt(0) & df_lines["security_class"].fillna("").str.fullmatch(r"common|class_[A-Z]")]
    df_lines["start"] = pd.to_datetime(df_lines["valid_from"], errors="coerce").dt.normalize()
    df_lines["end"] = pd.to_datetime(df_lines["valid_to"], errors="coerce").dt.normalize()
    df_matches = df_pairs.merge(df_lines, left_on="ticker", right_on="canonical_company")
    matches = (
        df_matches["start"].le(df_matches["session"])
        & (df_matches["end"].isna() | df_matches["end"].gt(df_matches["action"]))
        & (df_matches["klass"].eq("COMMON") | df_matches["security_class"].eq("class_" + df_matches["klass"]))
    )
    accepted = df_matches.loc[matches].groupby("row")["security_id"].nunique().eq(1)
    supported.iloc[accepted.index[accepted]] = True
    return supported


def _action_sum(trades: pd.DataFrame, idx: pd.DatetimeIndex, window_days: int, seen: pd.DataFrame) -> pd.DataFrame:
    """Sum the supported contributions; a trade without one is left out (a window with none is masked by `_unknown_windows`)."""
    return _rolling(trades, idx, window_days, "action_value", seen)


# --------------------------------------------------------------------------- #
# cumulative purchase intensities and historical ranks                                  #
# --------------------------------------------------------------------------- #


def _breadth_fields(buys: pd.DataFrame, sells: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Distinct buyers and sellers in `BREADTH_WINDOW` days, and buyers over all distinct insiders who traded.

    Rolling distinct counts, deliberately not decayed: a decayed sum of events is not a count of people. Breadth is
    undefined (NaN) when nobody traded in the window."""
    buyers, sellers = _distinct_owners(buys, idx), _distinct_owners(sells, idx)
    insiders = _distinct_owners(pd.concat([buys, sells], ignore_index=True), idx)
    out = {}
    if buyers is not None:
        out[DISTINCT_BUYERS] = buyers
    if sellers is not None:
        out[DISTINCT_SELLERS] = sellers
    if insiders is not None:
        bought = buyers.reindex_like(insiders).fillna(0.0) if buyers is not None else insiders * 0.0
        out[BUYER_BREADTH] = bought / insiders.where(insiders.gt(0))
    return out


def _distinct_owners(records: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Per-ticker rolling distinct owner count of `records`, or None when there is nothing to count."""
    if records.empty or "owner_cik" not in records.columns:
        return None
    df_counted = records.assign(start=records["visible_from"], stop=_window_stop(records, BREADTH_WINDOW))
    df_per_owner = df_counted.drop_duplicates(["start", "stop", "ticker", "owner_cik"])
    frames = {tkr: _rolling_distinct(g, idx) for tkr, g in df_per_owner.groupby("ticker")}
    df_grid = pd.DataFrame({t: s for t, s in frames.items() if s is not None})
    return None if df_grid.empty else df_grid


def _rolling_distinct(g: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Distinct owners in the trailing `BREADTH_WINDOW` days, stepped onto `idx`.

    Exact rather than approximate. Each record counts on `[start, stop)` (its visible days
    inside the window after its anchor), so the count changes only on a start or a stop day;
    it is evaluated at their union and forward-filled between. Sweeping both edges with a
    single owner counter keeps it O(n) per ticker instead of re-counting a window per day.
    """
    start = g["start"].to_numpy("datetime64[ns]")
    stop = g["stop"].to_numpy("datetime64[ns]")
    owner = g["owner_cik"].astype(str).to_numpy()
    first = pd.Timestamp(start.min())
    counted = stop > start
    start, stop, owner = start[counted], stop[counted], owner[counted]
    by_start = np.argsort(start, kind="stable")
    by_stop = np.argsort(stop, kind="stable")
    add_day, add_owner = start[by_start], [str(o) for o in owner[by_start]]
    drop_day, drop_owner = stop[by_stop], [str(o) for o in owner[by_stop]]
    points = np.unique(np.concatenate([start, stop]))
    counts = np.empty(len(points), dtype="float64")

    live: dict[str, int] = {}
    distinct = i = j = 0
    for k, cp in enumerate(points):
        while i < len(add_day) and add_day[i] <= cp:
            live[add_owner[i]] = live.get(add_owner[i], 0) + 1
            distinct += live[add_owner[i]] == 1
            i += 1
        while j < len(drop_day) and drop_day[j] <= cp:
            live[drop_owner[j]] -= 1
            distinct -= live[drop_owner[j]] == 0
            j += 1
        counts[k] = distinct

    s = pd.Series(counts, index=pd.DatetimeIndex(points)).reindex(pd.DatetimeIndex(points).union(idx)).ffill().fillna(0.0).reindex(idx)
    # NaN before the ticker's first record: "nobody has ever traded this name" and "the
    # window has emptied" are different facts, and only the second is a zero.
    return s.where(idx >= first)


def _starting_balance(records: pd.DataFrame) -> float:
    """Reconstruct a uniquely supported daily lot path, independent of dataframe order."""
    after = records["shares_owned_after"].to_numpy(dtype=float)
    delta = np.where(records["code"].eq("P"), records["shares_n"], -records["shares_n"])
    if not np.isfinite(after).all() or (after < 0).any():
        return np.nan
    if len(np.unique(after)) == 1:
        return float(after[0] - delta.sum())
    before = np.round(after - delta, 6)
    ends = np.round(after, 6)
    if len(np.unique(before)) != len(before) or len(np.unique(ends)) != len(ends):
        return np.nan
    roots = set(before) - set(ends)
    tails = set(ends) - set(before)
    if len(roots) != 1 or len(tails) != 1:
        return np.nan
    current = next(iter(roots))
    start = current
    edges = dict(zip(before, ends, strict=True))
    visited = set()
    while current in edges and current not in visited:
        visited.add(current)
        current = edges[current]
    return float(start) if len(visited) == len(records) and current in tails else np.nan


def _daily_positions(trades: pd.DataFrame, other_flows: pd.DataFrame) -> pd.DataFrame:
    """One versioned direct account/class/economic-day position; unsupported flows abstain."""
    columns = [
        "ticker",
        "owner_cik",
        "klass",
        "transaction_day",
        "anchor",
        "visible_from",
        "visible_until",
        "starting",
        "net_shares",
        "population",
    ]
    if trades.empty or "shares_owned_after" not in trades:
        return pd.DataFrame(columns=columns)
    joint = pd.to_numeric(trades.get("n_reporting_owners", pd.Series(1, index=trades.index)), errors="coerce").fillna(1).gt(1)
    df_rows = trades[trades["direct_indirect"].eq("D") & trades["owner_cik"].ne("") & trades["transaction_day"].notna() & ~joint].copy()
    if df_rows.empty:
        return pd.DataFrame(columns=columns)
    df_rows["klass"] = security_class(df_rows["security_title"])
    df_rows["delta"] = df_rows["shares_n"].where(df_rows["code"].eq("P"), -df_rows["shares_n"])
    df_rows["missing_after"] = df_rows["shares_owned_after"].isna()
    keys = ["ticker", "owner_cik", "klass", "transaction_day"]
    df_rows["other_day"] = pd.NaT
    flow_groups = {}
    if not other_flows.empty:
        by_ticker = dict(tuple(other_flows.groupby("ticker", sort=False)))
        for (ticker, owner), df_account in df_rows.groupby(["ticker", "owner_cik"], sort=False):
            df_blocked = account_other_flows(by_ticker.get(ticker, other_flows.iloc[:0]), ticker, owner)
            flow_groups[(ticker, owner)] = df_blocked
            earliest = df_blocked.dropna(subset=["visible_from", "transaction_day"]).groupby("transaction_day")["visible_from"].min()
            df_rows.loc[df_account.index, "other_day"] = earliest.reindex(df_account["transaction_day"]).to_numpy()
    grouped = df_rows.groupby(keys, sort=False)
    df_days = grouped.agg(
        anchor=("anchor", "min"),
        visible_from=("visible_from", "min"),
        n_start=("visible_from", "nunique"),
        closed=("visible_until", "count"),
        n_after=("shares_owned_after", "nunique"),
        missing_after=("missing_after", "max"),
        terminal=("shares_owned_after", "first"),
        net_shares=("delta", "sum"),
        population=("population", "min"),
        other_day=("other_day", "min"),
    )
    fast = (
        df_days["n_start"].eq(1)
        & df_days["closed"].eq(0)
        & df_days["n_after"].eq(1)
        & ~df_days["missing_after"]
        & df_days["other_day"].isna()
        & df_days["terminal"].ge(0)
    )
    df_fast = df_days.loc[fast].copy()
    df_fast["starting"] = df_fast["terminal"] - df_fast["net_shares"]
    df_fast["visible_until"] = pd.NaT
    results = [df_fast.reset_index()[columns]]
    slow_keys = set(df_days.index[~fast])
    _log.info(
        "insider positions: %s account days, %s simple, %s sequences, %s initially blocked by other flows",
        len(df_days),
        int(fast.sum()),
        len(slow_keys),
        int(df_days["other_day"].le(df_days["visible_from"]).sum()),
    )
    if slow_keys:
        df_slow = df_rows[pd.MultiIndex.from_frame(df_rows[keys]).isin(slow_keys)]
        records_out = []
        for key, df_group in df_slow.groupby(keys, sort=False):
            account = dict(zip(keys, key, strict=True))
            df_flows = flow_groups.get(key[:2], pd.DataFrame(columns=["transaction_day", "visible_from", "visible_until"]))
            records_out.extend(_account_day_versions(df_group, account, df_flows))
        if records_out:
            results.append(pd.DataFrame(records_out)[columns])
    df_result = pd.concat(results, ignore_index=True)
    df_result = df_result[df_result["starting"].ge(0)]
    _log.info(
        "insider positions: %s supported account-day versions from %s direct rows; %s zero-prior versions",
        len(df_result),
        len(df_rows),
        int(df_result["starting"].eq(0).sum()),
    )
    return df_result


def _account_day_versions(df_group: pd.DataFrame, account: Mapping[str, object], df_flows: pd.DataFrame) -> list[dict[str, object]]:
    """Reconstruct one account/economic day's balances at its public version boundaries."""
    df_blocked = df_flows.loc[df_flows["transaction_day"].eq(account["transaction_day"])]
    points = sorted(
        set(df_group["visible_from"])
        | set(df_group["visible_until"].dropna())
        | set(df_blocked["visible_from"].dropna())
        | set(df_blocked["visible_until"].dropna())
    )
    anchor = df_group["anchor"].min()
    records: list[dict[str, object]] = []

    # Sweep disclosure/expiry boundaries, excluding publicly unsupported account flows.
    for i, point in enumerate(points):
        df_visible = df_group[df_group["visible_from"].le(point) & (df_group["visible_until"].isna() | df_group["visible_until"].gt(point))]
        blocked = df_blocked["visible_from"].le(point) & (df_blocked["visible_until"].isna() | df_blocked["visible_until"].gt(point))
        if df_visible.empty or blocked.any():
            continue
        start = _starting_balance(df_visible)
        if not np.isfinite(start) or start < 0:
            continue

        # Keep each supported balance visible only until the next public boundary.
        record = dict(account)
        record.update(
            anchor=anchor,
            visible_from=point,
            visible_until=points[i + 1] if i + 1 < len(points) else pd.NaT,
            starting=start,
            net_shares=float(df_visible["delta"].sum()),
            population=int(df_visible["population"].min()),
        )
        records.append(record)
    return records
