"""Raw insider features on original-disclosure windows and public correction intervals.

Complete, corroborated inactivity can be zero; unsupported prices, quantities and account
balances mask only dependent fields. Owner ranks use strictly earlier visible disclosures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel, mask_to_availability, sector_relative
from src.data_aggregate.utils.institutionals.insider_positions import account_conviction, cumulative_positions
from src.data_aggregate.utils.institutionals.insider_quality import POP_C_SUITE, POP_MANAGEMENT, security_class
from src.data_aggregate.utils.institutionals.short_flow_features import _proven_tenure_mask
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_store.schema import Tables

_log = logging.getLogger(__name__)

#: Trailing calendar-day windows of the dense features.
WINDOW_60, WINDOW_120, WINDOW_180 = 60, 120, 180
WINDOWS = (WINDOW_60, WINDOW_120, WINDOW_180)

#: Window of the distinct-insider counts and of the breadth ratio built from them.
BREADTH_WINDOW = WINDOW_120

#: Disjoint insider populations (see `insider_quality.TITLE_POPULATIONS`) merged into the one population leg: C-suite and
#: management. 10% owners and unclassed owners are left out.
CM_POPULATIONS = frozenset({POP_C_SUITE, POP_MANAGEMENT})
CM_BUY = f"ic_insider_cm_buy_mcap_{WINDOW_180}d"
CM_BUY_SECTOR = f"{CM_BUY}_vs_sector"

#: feature -> (population codes or None for all insiders, trailing window days).
BUY_VALUE_FEATURES: dict[str, tuple[frozenset[int] | None, int]] = {
    **{f"ic_insider_buy_value_mcap_{days}d": (None, days) for days in WINDOWS},
    CM_BUY: (CM_POPULATIONS, WINDOW_180),
}
#: feature -> trailing window days; all insiders pooled.
PURCHASE_PCT_FEATURES: dict[str, int] = {f"ic_insider_purchase_pct_prior_{days}d": days for days in (WINDOW_60, WINDOW_120)}
CONVICTION_FEATURES: dict[str, int] = {f"ic_insider_conviction_{days}d": days for days in (WINDOW_60, WINDOW_120)}
NET_BUY_RATIO_FEATURES: dict[str, int] = {f"ic_insider_net_buy_ratio_{days}d": days for days in WINDOWS}
DISTINCT_BUYERS = f"ic_insider_distinct_buyers_{BREADTH_WINDOW}d"
DISTINCT_SELLERS = f"ic_insider_distinct_sellers_{BREADTH_WINDOW}d"
BUYER_BREADTH = f"ic_insider_buyer_breadth_{BREADTH_WINDOW}d"

#: Insider magnitudes, ratios, counts, and event intensities stay in their economic units.
#: Same-day cross-sectional ranks would replace those units with universe-composition noise.
EMISSION: dict[str, str] = dict.fromkeys(
    [
        *BUY_VALUE_FEATURES,
        CM_BUY_SECTOR,
        *PURCHASE_PCT_FEATURES,
        *CONVICTION_FEATURES,
        *NET_BUY_RATIO_FEATURES,
        DISTINCT_BUYERS,
        DISTINCT_SELLERS,
        BUYER_BREADTH,
    ],
    "raw",
)
#: The pooled 120d ratio also ships its peer z-score: the same net purchase reads differently in a sector with a selling wave.
PEER_RELATIVE = "ic_insider_purchase_pct_prior_120d"
EMISSION[PEER_RELATIVE] = "raw+peers"

#: No other `_vs_peers` leg in this family, for the same reason as `ic_super_*` (D25): an
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
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Compute features from clean source-USD trades, separately rebased positions and dated coverage.

    `sector_map` (ticker -> GICS sector) scores the C-suite and management buy leg within its sector."""
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
    disputed_accounts = _disputed_accounts(df_disputed)
    for name, days in PURCHASE_PCT_FEATURES.items():
        fields[name] = cumulative_positions(df_position_trades, df_other_flows, idx, days, disputed_accounts)
    for name, days in CONVICTION_FEATURES.items():
        fields[name] = account_conviction(df_position_trades, df_other_flows, idx, days, disputed_accounts)
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
    if CM_BUY in fields:
        fields[CM_BUY_SECTOR] = sector_relative(mask_to_availability(fields[CM_BUY], frames.availability), sector_map)

    # Emit the existing raw float32 fields with own-price availability.
    _log.info("insider panel: %s features from %s scoped transactions (%s buys, %s sells)", len(fields), len(df_trades), len(df_buys), len(df_sells))
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)


def _disputed_accounts(df_uncertain: pd.DataFrame) -> pd.DataFrame:
    """Direct accounts (ticker, owner_cik) with a disputed or unsupported quantity, once each.

    The exclusion is whole-history by design: a disputed account leaves every position ratio on every date, not only
    the window its disputed trade touches, because one wrong quantity makes the account's balance chain unreliable."""
    columns = ["ticker", "owner_cik"]
    if df_uncertain.empty or "direct_indirect" not in df_uncertain:
        return pd.DataFrame(columns=columns)
    df_direct = df_uncertain[df_uncertain["direct_indirect"].eq("D") & df_uncertain["owner_cik"].ne("")]
    return df_direct[columns].drop_duplicates().reset_index(drop=True)


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
    window_days: int,
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
    for name, days in PURCHASE_PCT_FEATURES.items():
        specifications[name] = (days, traded)
    for name, days in CONVICTION_FEATURES.items():
        specifications[name] = (days, traded)
    for name, days in NET_BUY_RATIO_FEATURES.items():
        specifications[name] = (days, traded)
    specifications[DISTINCT_BUYERS] = (BREADTH_WINDOW, buy)
    specifications[DISTINCT_SELLERS] = (BREADTH_WINDOW, sell)
    specifications[BUYER_BREADTH] = (BREADTH_WINDOW, traded)
    return specifications


def _member(rows: pd.DataFrame, codes: frozenset[int] | None) -> pd.Series:
    """Rows whose population is in `codes`; every row for `None`."""
    if codes is None:
        return pd.Series(True, index=rows.index)
    return rows["population"].isin(codes)


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
        selected = buys[_member(buys, code)]
        if selected.empty:
            continue
        first = selected.groupby("ticker")["visible_from"].min()
        eligible = seen.loc[:, first.index] & pd.DataFrame({ticker: idx >= day for ticker, day in first.items()}, index=idx)
        # a trade without a supported contribution is left out; a window with none is masked by `_unknown_windows`
        out[name] = _rolling(selected, idx, days, "action_value", eligible)
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
    groups: list[tuple[str, str, tuple[str, ...], int, frozenset[int] | None]] = [
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
