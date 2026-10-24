"""Raw insider features on original-disclosure windows and public correction intervals.

Complete, corroborated inactivity can be zero; unsupported prices, quantities and account
balances mask only dependent fields. Owner ranks use strictly earlier visible disclosures.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.insider_positions import cumulative_positions, position_units
from src.data_aggregate.utils.institutionals.insider_quality import FLAG_PCT_SHARES_OUTSTANDING, clean_transactions, report_oversized, security_class
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis

_log = logging.getLogger(__name__)

#: Insider magnitudes, ratios, counts, and event intensities stay in their economic units.
#: Same-day cross-sectional ranks would replace those units with universe-composition noise.
EMISSION: dict[str, str] = {
    "ic_insider_buy_value_mcap_60d": "raw",
    "ic_insider_buy_value_mcap_180d": "raw",
    "ic_insider_distinct_buyers_120d": "raw",  # 98.1% ties: a 0-8 integer count
    "ic_insider_cluster_buy_120d": "raw",  # 98.3% ties: the same, gated
    "ic_insider_ceo_buy_mcap_180d": "raw",
    "ic_insider_cfo_buy_mcap_180d": "raw",
    "ic_insider_director_buy_mcap_180d": "raw",
    "ic_insider_purchase_pct_prior": "raw",
    "ic_insider_owner_surprise_120d": "raw",  # already a percentile in [0, 1]
    "ic_insider_net_buy_ratio_180d": "raw",  # bounded [-1, 1] by construction
    "ic_insider_discretionary_sell_mcap_60d": "raw",
    "ic_insider_planned_sell_mcap_60d": "raw",
    "ic_insider_management_position_change_60d": "raw",
}

#: No `_vs_peers` leg anywhere in this family, for the same reason as `ic_super_*` (D25): an
#: insider purchase is an event about ONE company, and expressing it relative to a peer
#: basket that mostly has no event at all measures the basket's emptiness, not the signal.

#: Trailing calendar-day windows for the dense features. 20d is deliberately absent.
WINDOW_60, WINDOW_120, WINDOW_180 = 60, 120, 180

#: Cluster definition: distinct buyers within `CLUSTER_WINDOW_DAYS`, at least `CLUSTER_MIN`.
CLUSTER_WINDOW_DAYS, CLUSTER_MIN = 120, 2

#: D16 hard cutoffs. Both are the measured first row of their own evidence, not a guess, and
#: both emit NaN -- never 0 -- before the date. A 0 would read as "no insider bought", which
#: is a claim the data cannot support for a year it does not cover.
INSIDER_FLOOR = pd.Timestamp("2006-01-03")
TEN_B5_1_FLOOR = pd.Timestamp("2023-04-01")

#: Legacy caller default; cumulative insider fields do not use temporal decay.
DEFAULT_DECAY_HALFLIFE = 63.0


def build_insider_feature_panel(
    frames: PriceFrames,
    insider: pd.DataFrame | None,
    *,
    shares_out_history: pd.DataFrame | None = None,
    decay_halflife: float = DEFAULT_DECAY_HALFLIFE,
    availability: InstitutionalAvailability | None = None,
    complete_through: pd.Timestamp | None = None,
    as_traded_quotes: pd.DataFrame | None = None,
    role_evidence: pd.DataFrame | None = None,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Point-in-time insider features, preserving source-complete quiet windows and unit evidence."""
    need = {"ticker", "filing_date", "transaction_code", "shares"}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return _empty_panel()

    # Clean source versions once and establish the complete calculation calendar.
    df_trades, diag = clean_transactions(insider, as_traded_quotes=as_traded_quotes, role_evidence=role_evidence, share_basis=share_basis)
    df_unpriced = diag.get("unpriced_events", pd.DataFrame(columns=["ticker", "day", "code"]))
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    # Prepare the distinct valuation and as-traded share denominators.
    df_stock_close = frames.close_split
    df_mcap = _market_cap(shares_out_history, df_stock_close, frames.level_factor)
    df_shares_out = (
        fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", idx, share_basis=share_basis)
        if shares_out_history is not None and not shares_out_history.empty
        else pd.DataFrame(index=idx)
    )
    _report_oversized(df_trades, df_shares_out)

    # Reconstruct positions with disclosed shares even when execution dollars are unknown.
    df_buys = df_trades[df_trades["code"].eq("P")].copy()
    df_sells = df_trades[df_trades["code"].eq("S")].copy()
    # Missing dollars do not invalidate disclosed shares or post-trade holdings.
    df_position_trades = position_units(pd.concat([df_trades, df_unpriced], ignore_index=True), frames, as_traded_quotes, share_basis)
    df_other_flows = diag.get("other_flows", pd.DataFrame())
    df_positions = _daily_positions(df_position_trades, df_other_flows)

    # Calculate raw feature groups before applying source and identity masks.
    insider_floor = availability.source_date(Tables.insider_transactions) if availability is not None else INSIDER_FLOOR
    ten_b5_floor = availability.source_date(Tables.insider_transactions, "is_10b5_1") if availability is not None else TEN_B5_1_FLOOR
    fields = {
        **_dense_fields(df_buys, df_sells, idx, df_mcap, ten_b5_floor),
        **_breadth_fields(df_buys, idx),
        **_sparse_fields(df_buys, idx, df_mcap),
        **cumulative_positions(df_position_trades, df_positions, df_other_flows, idx, WINDOW_60),
    }
    fields = _gate_dates(fields, idx, insider_floor, complete_through)

    # Establish observable quiet windows, preserving unsupported active disclosures.
    frontier = pd.Timestamp(complete_through) if complete_through is not None and pd.notna(complete_through) else None
    columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
    df_listed = _listed(df_stock_close, idx, columns)
    df_unsupported = diag.get("unsupported_events", pd.DataFrame())
    df_activity = pd.concat([df_trades, df_unpriced, df_unsupported], ignore_index=True)
    _observe_quiet_windows(
        fields, df_activity, diag["source_first"], idx, columns, df_listed, insider_floor, ten_b5_floor, frontier, availability, df_mcap
    )

    # Propagate uncertainty through dollar windows, account balances and prior-owner ranks.
    df_uncertain = pd.concat([diag.get("uncertain_events", pd.DataFrame()), df_unsupported], ignore_index=True)
    df_unknown = pd.concat([df_unpriced, df_uncertain], ignore_index=True)
    unpriced_masks = _unpriced_masks(df_unknown, idx)
    _add_position_uncertainty(unpriced_masks, df_uncertain, df_positions, idx)
    _add_owner_uncertainty(unpriced_masks, df_unknown, df_buys, idx)
    _mask_unknown_windows(fields, unpriced_masks)

    # Emit the existing raw float32 fields with own-price availability.
    _log.info("insider panel: %s features from %s scoped transactions (%s buys, %s sells)", len(fields), len(df_trades), len(df_buys), len(df_sells))
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)


def _add_position_uncertainty(masks: dict[str, pd.DataFrame], df_uncertain: pd.DataFrame, df_positions: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
    """Mask account ratios while their visible holdings depend on uncertain quantities."""
    keys = ["ticker", "owner_cik"]
    if df_uncertain.empty or df_positions.empty:
        return

    # Intersect account-position visibility with the uncertain disclosure's active window.
    df_unknown = df_uncertain[keys + ["visible_from", "visible_until"]].copy()
    df_unknown["visible_until"] = _window_stop(df_uncertain, WINDOW_60)
    df_dependencies = df_positions.merge(df_unknown, on=keys, suffixes=("", "_unknown"))
    df_dependencies["visible_from"] = df_dependencies[["visible_from", "visible_from_unknown"]].max(axis=1)
    df_dependencies["visible_until"] = df_dependencies[["visible_until", "visible_until_unknown"]].min(axis=1)
    df_dependencies = df_dependencies[df_dependencies["visible_until"].isna() | df_dependencies["visible_from"].lt(df_dependencies["visible_until"])]
    df_seen = pd.DataFrame(True, index=idx, columns=pd.Index(sorted(df_dependencies["ticker"].unique()), name="ticker"))

    # Extend each field's existing unknown mask without erasing other uncertainty.
    for name, selector in (
        ("ic_insider_purchase_pct_prior", df_dependencies["buy_shares"].gt(0)),
        ("ic_insider_management_position_change_60d", df_dependencies["is_management"]),
    ):
        df_affected = _rolling(df_dependencies[selector], idx, WINDOW_60, None, df_seen).gt(0)
        df_previous = masks.get(name, pd.DataFrame())
        masks[name] = df_previous.combine(df_affected, lambda a, b: a.fillna(False) | b.fillna(False))


def _add_owner_uncertainty(masks: dict[str, pd.DataFrame], df_unknown: pd.DataFrame, df_buys: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
    """Mask later ranks that depend on an earlier uncertain purchase by the same owner."""
    if df_unknown.empty or "owner_cik" not in df_unknown or "owner_cik" not in df_buys:
        return

    # Find earlier purchases actually visible when the later purchase was disclosed.
    df_pairs = df_buys[["ticker", "owner_cik", "anchor", "visible_from", "visible_until"]].merge(
        df_unknown.loc[df_unknown["code"].eq("P"), ["owner_cik", "anchor", "visible_from", "visible_until"]],
        on="owner_cik",
        suffixes=("", "_prior"),
    )
    affected = (
        df_pairs["owner_cik"].ne("")
        & df_pairs["anchor_prior"].lt(df_pairs["anchor"])
        & df_pairs["visible_from_prior"].le(df_pairs["visible_from"])
        & (df_pairs["visible_until_prior"].isna() | df_pairs["visible_until_prior"].gt(df_pairs["visible_from"]))
    )
    df_dependencies = df_pairs.loc[affected, ["ticker", "anchor", "visible_from", "visible_until"]].drop_duplicates()
    if df_dependencies.empty:
        return

    # Propagate the dependency through the same 120-day rank window.
    columns = pd.Index(sorted(df_dependencies["ticker"].unique()), name="ticker")
    df_seen = pd.DataFrame(True, index=idx, columns=columns)
    df_affected = _rolling(df_dependencies, idx, WINDOW_120, None, df_seen).gt(0)
    df_previous = masks.get("ic_insider_owner_surprise_120d", pd.DataFrame())
    masks["ic_insider_owner_surprise_120d"] = df_previous.combine(df_affected, lambda a, b: a.fillna(False) | b.fillna(False))


def _gate_dates(
    fields: dict[str, pd.DataFrame], idx: pd.DatetimeIndex, insider_floor: pd.Timestamp, complete_through: pd.Timestamp | None
) -> dict[str, pd.DataFrame]:
    """The non-empty fields, NaN before the source floor and after the complete frontier (when known)."""
    floor = pd.Series(idx >= insider_floor, index=idx)
    frontier = (
        pd.Series(idx <= pd.Timestamp(complete_through).normalize(), index=idx)
        if complete_through is not None and pd.notna(complete_through)
        else pd.Series(True, index=idx)
    )
    return {name: frame.where(floor & frontier, axis=0) for name, frame in fields.items() if frame is not None and not frame.empty}


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
    complete_through: pd.Timestamp,
    availability: InstitutionalAvailability | None,
    window_days: int = WINDOW_180,
) -> pd.DataFrame:
    """Cells where an empty window is observable: listed, a full calendar window after the
    source floor, and on or before the complete frontier."""
    history_start = pd.Timestamp(insider_floor).normalize() + pd.Timedelta(days=window_days - 1)
    full_window = InstitutionalAvailability.date_mask(idx, columns, history_start)
    observed_through = InstitutionalAvailability.through_mask(idx, columns, complete_through)
    if availability is not None:
        return availability.source_mask(Tables.insider_transactions, idx, columns, requirements=(listed, full_window, observed_through))
    source_started = InstitutionalAvailability.date_mask(idx, columns, insider_floor)
    return InstitutionalAvailability.combine(source_started, listed, full_window, observed_through)


def _observe_quiet_windows(
    fields: dict[str, pd.DataFrame],
    activity: pd.DataFrame,
    source_first: pd.Series,
    idx: pd.DatetimeIndex,
    columns: pd.Index,
    listed: pd.DataFrame,
    insider_floor: pd.Timestamp,
    plan_floor: pd.Timestamp,
    frontier: pd.Timestamp | None,
    availability: InstitutionalAvailability | None,
    mcap: pd.DataFrame | None,
) -> None:
    """Fill only proven inactivity; unsupported active rows never become zero.

    Any issuer disclosure corroborates coverage prospectively. A complete frontier, full
    named source window, current listing and required denominator/flag coverage are also
    necessary. Unknown officer/director attribution cannot establish role inactivity.
    """
    buy, sell = activity["code"].eq("P"), activity["code"].eq("S")
    role_unknown = activity["officer_title_clean"].eq("") & ~activity["is_officer"].eq(0)
    specifications = {
        "ic_insider_buy_value_mcap_60d": (WINDOW_60, buy),
        "ic_insider_buy_value_mcap_180d": (WINDOW_180, buy),
        "ic_insider_distinct_buyers_120d": (WINDOW_120, buy),
        "ic_insider_cluster_buy_120d": (WINDOW_120, buy),
        "ic_insider_ceo_buy_mcap_180d": (WINDOW_180, buy & (activity["role"].eq("CEO") | role_unknown)),
        "ic_insider_cfo_buy_mcap_180d": (WINDOW_180, buy & (activity["role"].eq("CFO") | role_unknown)),
        "ic_insider_director_buy_mcap_180d": (WINDOW_180, buy & ~activity["is_director"].eq(0)),
        "ic_insider_purchase_pct_prior": (WINDOW_60, buy),
        "ic_insider_owner_surprise_120d": (WINDOW_120, buy),
        "ic_insider_net_buy_ratio_180d": (WINDOW_180, buy | sell),
        "ic_insider_discretionary_sell_mcap_60d": (WINDOW_60, sell & ~activity["is_10b5_1"].eq(1) & ~activity["in_exercise_package"]),
        "ic_insider_planned_sell_mcap_60d": (WINDOW_60, sell & ~activity["is_10b5_1"].eq(0)),
        "ic_insider_management_position_change_60d": (WINDOW_60, (buy | sell) & (activity["is_management"] | role_unknown)),
    }
    seen = pd.DataFrame(True, index=idx, columns=columns)
    corroborated = pd.DataFrame({ticker: idx >= day for ticker, day in source_first.items()}, index=idx).reindex(columns=columns, fill_value=False)
    for name, (window, selector) in specifications.items():
        active = _rolling(activity[selector], idx, window, None, seen).fillna(0).gt(0)
        if frontier is None:
            if name in fields:
                fields[name] = fields[name].reindex(index=idx, columns=columns).where(active)
            continue
        floor = max(insider_floor, plan_floor) if "sell_mcap" in name else insider_floor
        observed = _net_buy_mask(idx, columns, listed, floor, frontier, availability, window) & corroborated
        if "mcap" in name:
            if mcap is None or mcap.empty:
                continue
            observed &= mcap.reindex(index=idx, columns=columns).gt(0)
        frame = fields.get(name, pd.DataFrame(index=idx)).reindex(index=idx, columns=columns)
        fields[name] = frame.mask(observed & ~active, 0.0).where(observed)
    unknown_roles = activity[role_unknown & buy]
    for name, days in (("ic_insider_ceo_buy_mcap_180d", WINDOW_180), ("ic_insider_cfo_buy_mcap_180d", WINDOW_180)):
        if name in fields:
            fields[name] = fields[name].mask(_rolling(unknown_roles, idx, days, None, seen).gt(0))
    unknown_flags = activity[sell & activity["is_10b5_1"].isna()]
    for name in ("ic_insider_discretionary_sell_mcap_60d", "ic_insider_planned_sell_mcap_60d"):
        if name in fields:
            fields[name] = fields[name].mask(_rolling(unknown_flags, idx, WINDOW_60, None, seen).gt(0))
    for name, selector, days in (
        ("ic_insider_director_buy_mcap_180d", buy & activity["is_director"].isna(), WINDOW_180),
        ("ic_insider_distinct_buyers_120d", buy & activity["owner_cik"].eq(""), WINDOW_120),
        ("ic_insider_cluster_buy_120d", buy & activity["owner_cik"].eq(""), WINDOW_120),
    ):
        if name in fields:
            fields[name] = fields[name].mask(_rolling(activity[selector], idx, days, None, seen).gt(0))


# --------------------------------------------------------------------------- #
# dense features -- trailing calendar windows                                   #
# --------------------------------------------------------------------------- #


def _dense_fields(
    buys: pd.DataFrame,
    sells: pd.DataFrame,
    idx: pd.DatetimeIndex,
    mcap: pd.DataFrame | None,
    ten_b5_floor: pd.Timestamp = TEN_B5_1_FLOOR,
) -> dict[str, pd.DataFrame]:
    """Dense trailing sums over calendar windows, sampled onto
    the trading grid, so a day only ever sees transactions already filed by it."""
    out: dict[str, pd.DataFrame] = {}
    seen = _first_filing(buys, sells, idx)
    buy_val_180 = _rolling(buys, idx, WINDOW_180, "value", seen)
    sell_val_180 = _rolling(sells, idx, WINDOW_180, "value", seen)

    if mcap is not None and not mcap.empty:
        out["ic_insider_buy_value_mcap_180d"] = _over(buy_val_180, mcap)

    bv, sv = buy_val_180.fillna(0.0), sell_val_180.fillna(0.0)
    denom = bv + sv

    # Leave 0/0 undefined here. The caller converts it to an observed neutral 0 only inside an
    # explicit source-complete window; without that coverage proof, no filing remains unknown.
    # The bounded [-1, 1] range is what keeps this one `raw`.
    out["ic_insider_net_buy_ratio_180d"] = ((bv - sv) / denom.where(denom > 0)).replace([np.inf, -np.inf], np.nan)

    if mcap is not None and not mcap.empty:
        planned = sells["is_10b5_1"]
        # Discretionary requires BOTH: not on a plan, and not the sell leg of an
        # exercise-and-sell package (35.0% of all `S` rows). A NaN plan flag is not a 0 --
        # before 2023-04-01 the field was not compulsory, and the floor removes that region
        # rather than letting "unknown" masquerade as "discretionary".
        disc = sells[planned.eq(0) & ~sells["in_exercise_package"].astype(bool)]
        for name, sub in (("ic_insider_discretionary_sell_mcap_60d", disc), ("ic_insider_planned_sell_mcap_60d", sells[planned.eq(1)])):
            frame = _over(_rolling(sub, idx, WINDOW_60, "value", seen), mcap)
            out[name] = frame.where(pd.Series(idx >= ten_b5_floor, index=idx), axis=0)
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


def _unpriced_masks(events: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Feature windows made unknown by an excluded, unpriced open-market transaction."""
    if events.empty:
        return {}
    groups = (
        (("P",), WINDOW_60, ("ic_insider_buy_value_mcap_60d",)),
        (("P",), WINDOW_180, ("ic_insider_ceo_buy_mcap_180d", "ic_insider_cfo_buy_mcap_180d", "ic_insider_director_buy_mcap_180d")),
        (
            ("P",),
            WINDOW_120,
            ("ic_insider_distinct_buyers_120d", "ic_insider_cluster_buy_120d", "ic_insider_owner_surprise_120d"),
        ),
        (
            ("P",),
            WINDOW_180,
            (
                "ic_insider_buy_value_mcap_180d",
                "ic_insider_buy_shares_so_180d",
            ),
        ),
        (("P", "S"), WINDOW_180, ("ic_insider_net_buy_ratio_180d",)),
        (("S",), WINDOW_60, ("ic_insider_discretionary_sell_mcap_60d", "ic_insider_planned_sell_mcap_60d")),
    )
    columns = pd.Index(sorted(events["ticker"].astype(str).unique()), name="ticker")
    seen = pd.DataFrame(True, index=idx, columns=columns)
    masks: dict[str, pd.DataFrame] = {}
    for codes, days, names in groups:
        affected_events = events[events["code"].isin(codes)]
        if affected_events.empty:
            continue
        affected = _rolling(affected_events, idx, days, None, seen).gt(0)
        masks.update(dict.fromkeys(names, affected))
    if "is_management" in events:
        quantity_unknown = events[events["direct_indirect"].eq("D") & events["quality_uncertain"].fillna(False)]
        if not quantity_unknown.empty:
            masks["ic_insider_purchase_pct_prior"] = _rolling(quantity_unknown, idx, WINDOW_60, None, seen).gt(0)
        management = events[events["is_management"] & events["direct_indirect"].eq("D") & events["quality_uncertain"].fillna(False)]
        if not management.empty:
            masks["ic_insider_management_position_change_60d"] = _rolling(management, idx, WINDOW_60, None, seen).gt(0)
    return masks


def _mask_unknown_windows(fields: dict[str, pd.DataFrame], masks: dict[str, pd.DataFrame]) -> None:
    for name, unknown in masks.items():
        if name in fields:
            field = fields[name]
            aligned = unknown.reindex(index=field.index, columns=field.columns, fill_value=False).astype(bool)
            fields[name] = field.mask(aligned)


def _over(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """`numerator / denominator` on the intersection, NaN where the denominator is absent or
    non-positive. Never zero-fills the denominator: a missing share count is unknown size,
    not infinite size."""
    cols = numerator.columns.intersection(denominator.columns)
    if cols.empty:
        return pd.DataFrame(index=numerator.index)
    den = denominator.reindex(index=numerator.index)[cols]
    return (numerator[cols] / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# cumulative purchase intensities and historical ranks                                  #
# --------------------------------------------------------------------------- #


def _sparse_fields(buys: pd.DataFrame, idx: pd.DatetimeIndex, mcap: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """Cumulative dollar intensities and a dollar-weighted historical rank."""
    if buys.empty:
        return {}
    seen = _first_filing(buys, buys.iloc[:0], idx)
    out = {}
    if mcap is not None and not mcap.empty:
        out["ic_insider_buy_value_mcap_60d"] = _over(_rolling(buys, idx, WINDOW_60, "value", seen), mcap)
        roles = {
            "ic_insider_ceo_buy_mcap_180d": buys["role"].eq("CEO"),
            "ic_insider_cfo_buy_mcap_180d": buys["role"].eq("CFO"),
            "ic_insider_director_buy_mcap_180d": buys["is_director"].eq(1),
        }
        for name, selector in roles.items():
            selected = buys[selector]
            if not selected.empty:
                role_seen = seen.loc[:, selected["ticker"].unique()]
                # A role's flow becomes observed only at its first eligible disclosure.
                first = selected.groupby("ticker")["visible_from"].min()
                role_seen = role_seen & pd.DataFrame({ticker: idx >= day for ticker, day in first.items()}, index=idx)
                out[name] = _over(_rolling(selected, idx, WINDOW_180, "value", role_seen), mcap)
    surprise = _owner_surprise(buys)
    if surprise is not None and not surprise.empty:
        scores = surprise.assign(_num=surprise["value"] * surprise["surprise"])
        numerator = _rolling(scores, idx, WINDOW_120, "_num", seen)
        denominator = _rolling(scores, idx, WINDOW_120, "value", seen)
        out["ic_insider_owner_surprise_120d"] = numerator / denominator.where(denominator.gt(0))
    return out


def _breadth_fields(buys: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """#31 distinct buyers in 120 days, and #32 the same count gated at `CLUSTER_MIN`.

    Rolling distinct counts, deliberately not decayed (a departure from D13): a decayed sum of
    events is not a count of people, and this family has no `_vs_peers` leg, so the sparse-flag
    failure D13 guards against cannot occur here.
    """
    if buys.empty or "owner_cik" not in buys.columns:
        return {}
    df_counted = buys.assign(start=buys["visible_from"], stop=_window_stop(buys, CLUSTER_WINDOW_DAYS))
    df_per_owner = df_counted.drop_duplicates(["start", "stop", "ticker", "owner_cik"])
    frames = {tkr: _rolling_distinct(g, idx) for tkr, g in df_per_owner.groupby("ticker")}
    df_grid = pd.DataFrame({t: s for t, s in frames.items() if s is not None})
    if df_grid.empty:
        return {}
    return {
        "ic_insider_distinct_buyers_120d": df_grid,
        # Gated, not masked: below the threshold the answer is "no cluster" = 0, which is
        # a fact, while before the first purchase it is unknown = NaN, inherited above.
        "ic_insider_cluster_buy_120d": df_grid.where(df_grid >= CLUSTER_MIN, 0.0).where(df_grid.notna()),
    }


def _rolling_distinct(g: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Distinct owners buying in the trailing `CLUSTER_WINDOW_DAYS`, stepped onto `idx`.

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
    # NaN before the ticker's first purchase: "nobody has ever bought this name" and "the
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
        "buy_shares",
        "value",
        "is_management",
    ]
    if trades.empty or "shares_owned_after" not in trades:
        return pd.DataFrame(columns=columns)
    joint = pd.to_numeric(trades.get("n_reporting_owners", pd.Series(1, index=trades.index)), errors="coerce").fillna(1).gt(1)
    df_rows = trades[trades["direct_indirect"].eq("D") & trades["owner_cik"].ne("") & trades["transaction_day"].notna() & ~joint].copy()
    if df_rows.empty:
        return pd.DataFrame(columns=columns)
    df_rows["klass"] = security_class(df_rows["security_title"])
    df_rows["delta"] = df_rows["shares_n"].where(df_rows["code"].eq("P"), -df_rows["shares_n"])
    df_rows["buy_shares"] = df_rows["shares_n"].where(df_rows["code"].eq("P"), 0.0)
    df_rows["buy_value"] = df_rows["value"].where(df_rows["code"].eq("P"), 0.0)
    df_rows["missing_after"] = df_rows["shares_owned_after"].isna()
    keys = ["ticker", "owner_cik", "klass", "transaction_day"]
    flow_keys = ["ticker", "owner_cik", "transaction_day"]
    df_rows["other_day"] = pd.NaT
    if not other_flows.empty:
        earliest = other_flows.dropna(subset=["day", "transaction_day"]).groupby(flow_keys)["day"].min()
        df_rows["other_day"] = earliest.reindex(pd.MultiIndex.from_frame(df_rows[flow_keys])).to_numpy()
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
        buy_shares=("buy_shares", "sum"),
        value=("buy_value", "sum"),
        is_management=("is_management", "max"),
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
    never_supported = df_days["other_day"].le(df_days["visible_from"])
    slow_keys = set(df_days.index[~fast & ~never_supported])
    _log.info(
        "insider positions: %s account days, %s simple, %s sequences, %s blocked by other flows",
        len(df_days),
        int(fast.sum()),
        len(slow_keys),
        int(never_supported.sum()),
    )
    if slow_keys:
        df_slow = df_rows[pd.MultiIndex.from_frame(df_rows[keys]).isin(slow_keys)]
        records_out = []
        for key, df_group in df_slow.groupby(keys, sort=False):
            account = dict(zip(keys, key, strict=True))
            records_out.extend(_account_day_versions(df_group, account))
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


def _account_day_versions(df_group: pd.DataFrame, account: Mapping[str, object]) -> list[dict[str, object]]:
    """Reconstruct one account/economic day's balances at its public version boundaries."""
    points = sorted(set(df_group["visible_from"]) | set(df_group["visible_until"].dropna()) | set(df_group["other_day"].dropna()))
    anchor = df_group["anchor"].min()
    records: list[dict[str, object]] = []

    # Sweep disclosure/expiry boundaries, excluding publicly unsupported account flows.
    for i, point in enumerate(points):
        df_visible = df_group[df_group["visible_from"].le(point) & (df_group["visible_until"].isna() | df_group["visible_until"].gt(point))]
        if df_visible.empty or (df_visible["other_day"].notna() & df_visible["other_day"].le(point)).any():
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
            buy_shares=float(df_visible["buy_shares"].sum()),
            value=float(df_visible["buy_value"].sum()),
            is_management=bool(df_visible["is_management"].any()),
        )
        records.append(record)
    return records


def _owner_surprise(ev: pd.DataFrame) -> pd.DataFrame | None:
    """Each purchase's percentile among that OWNER's own strictly earlier purchases (no look-ahead).

    The prior set of a record is that owner's earlier-anchored records visible on its own
    `visible_from`, so a later restatement never re-ranks it. An owner's first purchase has no
    prior distribution and is NaN, never 0.5.
    """
    if "owner_cik" not in ev.columns:
        return None
    df_purchases = ev.dropna(subset=["value"]).sort_values(["owner_cik", "anchor"], kind="stable")
    if df_purchases.empty:
        return None
    df_purchases = df_purchases[df_purchases["owner_cik"].notna() & df_purchases["owner_cik"].ne("")]
    df_purchases["surprise"] = _visible_prior_rank(df_purchases)
    return df_purchases.dropna(subset=["surprise"])


def _visible_prior_rank(d: pd.DataFrame) -> pd.Series:
    """Fraction of each record's prior set (see `_owner_surprise`) at or below its value; `d` is sorted by owner then anchor."""
    out = pd.Series(np.nan, index=d.index, dtype=float)
    for _, group in d.groupby("owner_cik", sort=False):
        if len(group) < 2:
            continue
        anchor = group["anchor"].to_numpy(dtype="datetime64[ns]")
        start = group["visible_from"].to_numpy(dtype="datetime64[ns]")
        stop = group["visible_until"].to_numpy(dtype="datetime64[ns]")
        value = group["value"].to_numpy(dtype=float)
        prior = (anchor[None, :] < anchor[:, None]) & (start[None, :] <= start[:, None])
        prior &= np.isnat(stop)[None, :] | (stop[None, :] > start[:, None])
        count = prior.sum(axis=1)
        below = (prior & (value[None, :] <= value[:, None])).sum(axis=1)
        out.loc[group.index] = below / np.where(count > 0, count, np.nan)
    return out


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


def _market_cap(shares_out_history: pd.DataFrame | None, stock_close: pd.DataFrame | None, level_factor: pd.DataFrame | None) -> pd.DataFrame | None:
    if shares_out_history is None or shares_out_history.empty or stock_close is None or stock_close.empty:
        _log.warning("No shares outstanding or close -> the seven size-scaled insider features are skipped.")
        return None
    mcap = daily_market_cap(shares_out_history, stock_close, level_factor=level_factor)
    if mcap.empty:
        # ⚠ THIS PATH USED TO BE SILENT, and that is what let a projection delete 10 features
        # across three builders with a clean-looking build log. `daily_market_cap` returns a
        # COLUMN-LESS frame when its input has no `sharesOutstanding` column, so the caller's
        # `if not mcap.empty` branch just never fires. The inputs are present -- the warning
        # above only covers their absence -- so the column is what has to be named.
        _log.warning(
            "daily_market_cap returned no columns (shares_out_history has %s; it "
            "needs `sharesOutstanding`, the VENDOR basis, not `sharesOutstandingPit`)"
            " -> the seven size-scaled insider features are skipped.",
            sorted(shares_out_history.columns),
        )
        return None
    return mcap


def _report_oversized(t: pd.DataFrame, shares_out: pd.DataFrame) -> None:
    """Log, never drop. See `insider_quality.report_oversized` for why a threshold cannot
    separate a fabricated filing from a genuine block disposal on this universe."""
    flagged = report_oversized(t, shares_out if not shares_out.empty else None)
    if flagged.empty:
        return
    top = flagged.head(5)
    _log.warning(
        "insider: %s transaction(s) above %.0f%% of shares outstanding, kept and reported: %s",
        len(flagged),
        100 * FLAG_PCT_SHARES_OUTSTANDING,
        ", ".join(f"{r.ticker} {r.day:%Y-%m-%d} {r.code} {r.pct_shares_outstanding:.1%}" for r in top.itertuples()),
    )
