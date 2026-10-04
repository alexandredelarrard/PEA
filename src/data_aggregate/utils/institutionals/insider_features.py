"""
insider_features.py  (src/data_aggregate/utils/institutionals/insider_features.py)
-------------------------------------------------------------------
Insider-trading signal from the SEC Insider Transactions Data Sets (Forms 3/4/5,
table `insider_transactions`). Distinct from the 13F institutional signal: this is the
issuer's OWN officers, directors and 5%/10% holders trading its stock. The durable alpha is
in OPEN-MARKET PURCHASES (`P`) -- especially CLUSTER buying -- while sales (`S`) are noisy,
because insiders sell for liquidity and diversification and, since 2023, mostly on a plan.

TAXONOMY IS THE FIRST FEATURE, NOT A DETAIL. `P` is **1.77%** of the 2,031,286 rows. `A`
grants (24.6%), `M` option exercises (24.2%), `F` tax withholding (10.0%), `G` gifts (2.6%),
`J` other acquisitions (2.3%) and `C` conversions (1.3%) are compensation mechanics, and
netting any of them into an insider number buries the 1.77% that carries information. The
scope cut lives in `insider_quality.clean_transactions`, with the derivative rows, the
non-common securities and the unpriced rows, and so does the repair of the filed prices --
read that module's docstring before trusting any dollar figure here.

POINT-IN-TIME. A Form 4 is due within ~2 business days of the trade, so every aggregate is
stamped on `filing_date`, never `transaction_date`, and every window is trailing. The price
consensus in `insider_quality` is trailing on that same publication clock; repaired values
reach the panel, so later filings must never enter the reference. A cleaned record counts on
date d only when `visible_from <= d < visible_until`, and ages from its `anchor` (the trade's
first disclosure): windows hold it while `d - window < anchor`, decayed legs weigh it by the
decay since the anchor, and an owner's surprise ranks it against that owner's records visible
on its own `visible_from`.

AVAILABILITY (D16, measured 2026-09-10):

    every feature      NaN before 2006-01-03   -- `insider_transactions` has no earlier row
    the two 10b5-1     NaN before 2023-04-01   -- the SEC reporting requirement begins then;
    features                                      2020/2021/2022, first non-null 2023-03-20,
                                                  74.1% for 2023 as a whole, then 99.6%
                                                  (2024), 99.6% (2025), 99.8% (2026)

`ic_insider_planned_sell_mcap_60d` is a CONTROL, not a signal: a sale executed under a plan
adopted months earlier should carry near-zero information, and shipping it beside the
discretionary leg is what makes the split testable rather than assumed. Its monotone
direction is 0 for that reason.

WINDOWS WERE CHOSEN BY MEASURED SPARSITY, not by preference (% of ticker-days non-zero,
2015-2026):

    window   any buy   >=2 distinct buyers   >=3 distinct buyers
      20d     4.2% x         1.2% x                0.6% x
      60d    10.7% v         3.4% x                1.7% x
     120d    18.5% v         6.9% v                3.5% x
     180d    24.8% v        10.6% v                5.6% v

There is no 20-day buy window: 4.2% is under the 5% coverage floor. A cluster is **>=2
distinct buyers in 120 days**, not the 3-in-20-days of the source research, which occupies
0.6% of ticker-days here -- this universe produces only ~2.0 distinct insider buyers per
ticker per year.

CLASS S FEATURES ARE DECAYED, NOT WINDOWED (D13) -- WITH TWO EXCEPTIONS. Eight of the
fourteen are sparse events; a raw flag through the peer panel is ~100% NaN, not a weak
feature (see `decay.py`). For the six that ARE decayed, the window named in the table below
is the NATURAL window the coverage floor is judged on, per D14 rule 1, and the emitted column
is the decayed intensity at `decay_halflife` trading days. The two exceptions are #31 and
#32, which are true rolling distinct counts: see `_breadth_fields` for why decay destroys a
count of PEOPLE, and why this family may depart from D13 where others may not.

Features -- registry #28-#41:

    #28 buy_value_mcap_60d          S  decayed  P value / market cap, per purchase
    #29 buy_value_mcap_180d         D  180d     sum P value / market cap
    #30 buy_shares_so_180d          D  180d     sum P shares / shares outstanding (PIT)
    #31 distinct_buyers_120d        S  120d     distinct owner_cik buying -- NOT decayed
    #32 cluster_buy_120d            S  120d     the same count, 0 below CLUSTER_MIN
    #33 ceo_buy_mcap_180d           S  decayed  as #28, CEO only
    #34 cfo_buy_mcap_180d           S  decayed  as #28, CFO only
    #35 director_buy_mcap_180d      S  decayed  as #28, is_director = 1
    #36 purchase_pct_prior          S  decayed  shares bought / shares held before
    #37 owner_surprise_120d         S  decayed  this purchase's percentile in that owner's
                                                OWN prior purchases, expanding window
    #38 net_buy_ratio_180d          D  180d     (P$ - S$) / (P$ + S$), in [-1, 1]
    #39 discretionary_sell_mcap_60d D  60d      S value, not on a plan and not an
                                                exercise-and-sell, / market cap
    #40 planned_sell_mcap_60d       D  60d      S value on a 10b5-1 plan / market cap

#36 and #37 are RATIOS, so their class-S treatment is a decay-WEIGHTED MEAN -- numerator and
denominator are decayed with the same half-life and divided -- not a decayed sum. A decayed
sum of percentiles would grow with the number of purchases and stop being a percentile.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.decay import decay_events
from src.data_aggregate.utils.institutionals.insider_quality import FLAG_PCT_SHARES_OUTSTANDING, asof_values, clean_transactions, report_oversized
from src.data_store.schema import Tables


def _absent(df: pd.DataFrame | None, need: set[str] | None = None) -> bool:
    """True when `df` cannot be built from: missing, empty, or short a required column.

    The three-part test is the D5 entry contract stated once. `need` is the set the builder
    dereferences unconditionally -- a column it only uses `if present` does NOT belong here,
    or an optional projection turns into an empty panel.
    """
    if df is None or df.empty:
        return True
    return bool(need) and not need.issubset(df.columns)


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

#: Half-life in TRADING days for the class-S decay, overridden from
#: `build_cube.institutionals.decay_halflife.insider`.
DEFAULT_DECAY_HALFLIFE = 63.0


def build_insider_feature_panel(
    frames: PriceFrames,
    insider: pd.DataFrame | None,
    *,
    shares_out_history: pd.DataFrame | None = None,
    decay_halflife: float = DEFAULT_DECAY_HALFLIFE,
    availability: InstitutionalAvailability | None = None,
    complete_through: pd.Timestamp | None = None,
    sink=None,
) -> pd.DataFrame:
    """Long-format insider feature panel (`f_<name>` per `EMISSION`).

    Empty when there are no usable transactions. `shares_out_history` + `stock_close` are
    what make the size-scaled features possible; without them only the scale-free
    ones (`distinct_buyers`, `cluster_buy`, `net_buy_ratio`, `purchase_pct_prior`,
    `owner_surprise`) are emitted, because a dollar flow that is not
    divided by the company's size is a market-cap proxy.

    `sink` is the optional `ConditioningSink` the price-conditioning and cross-source panels
    read. ⚠ IT IS FILLED FROM `buys` RATHER THAN FROM THE PANEL, and that is the reason it
    exists: `buys` is the output of `clean_transactions`, a scope-and-repair pass over 2M rows
    and the most expensive read in the step, so the conditioning layer must take the event
    dates from here rather than repeat it. The `value`/`shares` it carries are the REPAIRED
    value and the AS-FILED share count -- the split restatement the cost anchor needs belongs
    to the consumer, which is the only place the price basis is known.

    ⚠ `frames` RATHER THAN FOUR UNPACKED FIELDS. `peer_dict`, `trading_index`, `stock_close` and
    `level_factor` were all read off one `PriceFrames` at the call site. Naming the object makes
    the basis un-mistakable: there is one `close_split` and one `close_total` on it, and neither
    can arrive under the other's parameter name.

    ⚠ NO `frames.require(...)`, AND THAT IS MEASURED RATHER THAN FORGOTTEN. Every wide frame
    this builder reads sits behind an explicit `is None` guard, or is handed to a callee that
    documents `None` as a MEANING rather than an error -- `daily_market_cap`'s
    `level_factor=None` IS "S is 1.0 everywhere". `require` would turn each of those graceful
    degrades into a raise, which is exactly what its own docstring warns against.

    The non-frame arguments are KEYWORD-ONLY. A positional slip between two same-typed
    `pd.DataFrame | None` neighbours is a silent wrong-frame bug that reads as a plausible
    call; the keyword form makes it unrepresentable.
    """
    peer_dict = frames.peers
    trading_index = frames.trading_index
    stock_close = frames.close_split
    level_factor = frames.level_factor

    need = {"ticker", "filing_date", "transaction_code", "shares"}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return _empty_panel()

    t, diag = clean_transactions(insider)
    unpriced_events = diag.get("unpriced_events", pd.DataFrame(columns=["ticker", "day", "code"]))
    if t.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    mcap = _market_cap(shares_out_history, stock_close, level_factor)
    shares_out = (
        fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", idx)
        if shares_out_history is not None and not shares_out_history.empty
        else pd.DataFrame(index=idx)
    )
    _report_oversized(t, shares_out)

    buys = t[t["code"].eq("P")].copy()
    sells = t[t["code"].eq("S")].copy()
    buys["mcap"] = asof_values(mcap, buys["ticker"], buys["anchor"])
    buys["value_mcap"] = buys["value"] / buys["mcap"].where(buys["mcap"] > 0)

    fields: dict[str, pd.DataFrame] = {}
    insider_floor = availability.source_date(Tables.insider_transactions) if availability is not None else INSIDER_FLOOR
    ten_b5_floor = availability.source_date(Tables.insider_transactions, "is_10b5_1") if availability is not None else TEN_B5_1_FLOOR

    fields.update(_dense_fields(buys, sells, idx, mcap, ten_b5_floor))
    fields.update(_breadth_fields(buys, idx))
    fields.update(_sparse_fields(buys, idx, decay_halflife))

    floor = pd.Series(idx >= insider_floor, index=idx)
    frontier = (
        pd.Series(idx <= pd.Timestamp(complete_through).normalize(), index=idx)
        if complete_through is not None and pd.notna(complete_through)
        else pd.Series(True, index=idx)
    )
    for name, frame in list(fields.items()):
        if frame is None or frame.empty:
            fields.pop(name)
            continue
        fields[name] = frame.where(floor & frontier, axis=0)

    # Unlike an event-price anchor, the net-buy ratio has a defined neutral state: in a fully
    # observed 180-day window with neither purchases nor sales, it is 0. Broaden to the listed
    # universe only when the extraction layer supplied an explicit complete frontier; without
    # that proof, missing stays NaN. The extra 179-day gate prevents a partial first window from
    # being called empty.
    net_name = "ic_insider_net_buy_ratio_180d"
    net_mask: pd.DataFrame | None = None
    if complete_through is not None and pd.notna(complete_through) and net_name in fields:
        columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
        if stock_close is not None and not stock_close.empty:
            listed = stock_close.reindex(index=idx, columns=columns).notna()
        else:
            listed = pd.DataFrame(True, index=idx, columns=columns)
        history_start = pd.Timestamp(insider_floor).normalize() + pd.Timedelta(days=WINDOW_180 - 1)
        full_window = InstitutionalAvailability.date_mask(idx, columns, history_start)
        observed_through = InstitutionalAvailability.through_mask(idx, columns, pd.Timestamp(complete_through))
        if availability is not None:
            net_mask = availability.source_mask(
                Tables.insider_transactions,
                idx,
                columns,
                requirements=(listed, full_window, observed_through),
            )
        else:
            source_started = InstitutionalAvailability.date_mask(idx, columns, insider_floor)
            net_mask = InstitutionalAvailability.combine(source_started, listed, full_window, observed_through)
        fields[net_name] = fields[net_name].reindex(index=idx, columns=columns).fillna(0.0).where(net_mask)

    unpriced_masks = _unpriced_masks(unpriced_events, idx)
    _mask_unknown_windows(fields, unpriced_masks)

    if sink is not None:
        sink.set_frontier("insider", complete_through)
        # ⚠ PROJECT FIRST, THEN RENAME. `buys` carries BOTH the source `shares` column and
        # `clean_transactions`' numeric `shares_n`, so renaming `shares_n -> shares` on the
        # whole frame produces two columns of that name and every later `ev["shares"]` is a
        # DataFrame, not a Series. That is what broke the first full build, ~28 minutes in and
        # past six merged panels -- the unit fixtures only ever carried `shares_n`.
        # One event per trade, at its first disclosure with the record known then; a later
        # amendment of the same trade is not a new disclosure.
        first_disclosure = buys["visible_from"].eq(buys["anchor"])
        ev = buys.loc[first_disclosure, ["ticker", "anchor", "value", "shares_n"]].rename(columns={"anchor": "date", "shares_n": "shares"})
        sink.add_events("insider", ev)
        columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
        if stock_close is not None and not stock_close.empty:
            listed = stock_close.reindex(index=idx, columns=columns).notna()
        else:
            listed = pd.DataFrame(True, index=idx, columns=columns)
        signal_fields = dict(fields)
        signal_masks: dict[str, pd.DataFrame] = {}
        source_last = (
            pd.Timestamp(complete_through).normalize()
            if complete_through is not None and pd.notna(complete_through)
            else pd.to_datetime(insider["filing_date"], errors="coerce").max()
        )
        frontier_mask = (
            InstitutionalAvailability.through_mask(idx, columns, source_last)
            if pd.notna(source_last)
            else pd.DataFrame(False, index=idx, columns=columns)
        )
        for name in ("ic_insider_buy_value_mcap_180d", "ic_insider_net_buy_ratio_180d"):
            if name not in fields:
                continue
            requirements = [listed, frontier_mask]
            if name == "ic_insider_buy_value_mcap_180d":
                if mcap is None or mcap.empty:
                    continue
                requirements.append(mcap.reindex(index=idx, columns=columns).gt(0))
            if name == net_name and net_mask is not None:
                mask = net_mask
            else:
                mask = (
                    availability.source_mask(
                        Tables.insider_transactions,
                        idx,
                        columns,
                        requirements=tuple(requirements),
                    )
                    if availability is not None
                    else InstitutionalAvailability.combine(*requirements)
                )
            signal_masks[name] = mask
            signal_fields[name] = fields[name].reindex(index=idx, columns=columns).fillna(0.0).where(mask)
        _mask_unknown_windows(signal_fields, unpriced_masks)
        for name, unknown in unpriced_masks.items():
            if name in signal_masks:
                signal_masks[name] &= ~unknown.reindex(index=idx, columns=columns, fill_value=False)
        sink.keep_signals(signal_fields, signal_masks)

    _log.info("insider panel: %s features from %s scoped transactions (%s buys, %s sells)", len(fields), len(t), len(buys), len(sells))
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission, availability=frames.availability)


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

    It is the coverage mask every windowed feature shares, and it exists because the two
    answers a zero can mean are different facts. `BEAR` sold and never bought: its buying
    over the window is **0**, a real observation. `BEAR` before its first-ever Form 4: its
    buying is **unknown**. Keying the mask on the union of purchases and sales rather than on
    the feature's own leg is what separates them -- without it a name that only ever sold
    reads NaN on every buy feature, which understates coverage and silently drops the name
    from the cross-section on the very dates it is most informative.
    """
    both = pd.concat([buys[["visible_from", "ticker"]], sells[["visible_from", "ticker"]]])
    if both.empty:
        return pd.DataFrame(False, index=idx, columns=pd.Index([], name="ticker"))
    first = both.groupby("ticker")["visible_from"].min()
    grid = pd.DataFrame({t: idx >= d for t, d in first.items()}, index=idx)
    grid.columns.name = "ticker"
    return grid


def _plain(records: pd.DataFrame) -> pd.Series:
    """True for a record visible from its own anchor with no end: neither a restatement nor restated."""
    return records["visible_from"].eq(records["anchor"]) & records["visible_until"].isna()


def _grid_pos(dates: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first `idx` day on or after each date; `len(idx)` for NaT (never reached)."""
    pos = idx.searchsorted(pd.DatetimeIndex(dates), side="left")
    return np.where(dates.isna().to_numpy(), len(idx), pos)


def _spread(start: np.ndarray, stop: np.ndarray, col: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each `[start, stop)` grid interval into its (row, col, value) cells."""
    length = np.clip(stop - start, 0, None)
    offset = np.arange(int(length.sum())) - np.repeat(np.cumsum(length) - length, length)
    rows = np.repeat(start, length) + offset
    cols = np.repeat(col, length)
    vals = np.repeat(values, length)
    return rows, cols, vals


def _rolling(txns: pd.DataFrame, idx: pd.DatetimeIndex, window_days: int, value_col: str | None, seen: pd.DataFrame) -> pd.DataFrame:
    """Trailing `window_days`-calendar-day sum per ticker, sampled onto `idx`.

    Plain records are zero-filled between filing days on a DAILY calendar and rolled, so the
    total counts the window's real length; every other record counts on the days `d` with
    `visible_from <= d < min(visible_until, anchor + window)`. Widened to every ticker in
    `seen` and masked before each one's first filing.
    """
    if txns.empty:
        return pd.DataFrame(np.nan, index=idx, columns=seen.columns)
    plain = _plain(txns)
    summed = pd.DataFrame(0.0, index=idx, columns=seen.columns)
    if plain.any():
        df_plain = txns[plain]
        if value_col:
            piv = df_plain.groupby(["anchor", "ticker"])[value_col].sum().unstack("ticker")
        else:
            piv = df_plain.groupby(["anchor", "ticker"]).size().unstack("ticker")
        calendar = pd.date_range(min(piv.index.min(), idx.min()), max(piv.index.max(), idx.max()), freq="D")
        piv = piv.reindex(calendar).fillna(0.0).rolling(f"{window_days}D").sum()
        summed = piv.reindex(index=idx, columns=seen.columns).fillna(0.0)
    if not plain.all():
        summed = summed + _late_window_sum(txns[~plain], idx, window_days, value_col, seen.columns)
    return summed.where(seen)


def _late_window_sum(records: pd.DataFrame, idx: pd.DatetimeIndex, window_days: int, value_col: str | None, columns: pd.Index) -> pd.DataFrame:
    """Window sum of non-plain records: each counts on `visible_from <= d < min(visible_until, anchor + window)`."""
    window_end = records["anchor"] + pd.Timedelta(days=window_days)
    stop = window_end.where(records["visible_until"].isna() | (window_end <= records["visible_until"]), records["visible_until"])
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
                "ic_insider_ceo_buy_mcap_180d",
                "ic_insider_cfo_buy_mcap_180d",
                "ic_insider_director_buy_mcap_180d",
                "ic_insider_purchase_pct_prior",
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
# sparse features -- decayed event intensities                                  #
# --------------------------------------------------------------------------- #


def _sparse_fields(buys: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float) -> dict[str, pd.DataFrame]:
    """The six DECAYED features: four value-scaled intensities and two decay-weighted means.

    #31 and #32 are class S too but live in `_breadth_fields` -- see there for why a distinct
    count is the one class-S quantity that decay destroys.
    """
    out: dict[str, pd.DataFrame] = {}
    if buys.empty:
        return out
    ev = buys
    # ⚠ ALL FOUR value-scaled intensities are gated on a real market cap, and the guard is
    # not defensive padding. `decay_events` treats a NaN magnitude as an event of unknown
    # size and counts it 1.0 -- correct policy there, but with no market cap EVERY magnitude
    # is NaN and these four would silently become event counts under names that promise
    # dollars over market cap.
    priced = "value_mcap" in ev.columns and ev["value_mcap"].notna().any()
    if priced:
        out["ic_insider_buy_value_mcap_60d"] = _decay_visible(ev, idx, halflife, "value_mcap")

    roles = {
        "ic_insider_ceo_buy_mcap_180d": ev["role"].eq("CEO"),
        "ic_insider_cfo_buy_mcap_180d": ev["role"].eq("CFO"),
        "ic_insider_director_buy_mcap_180d": ev["is_director"].eq(1),
    }
    for name, mask in roles.items():
        sub = ev[mask]
        if priced and not sub.empty and sub["value_mcap"].notna().any():
            out[name] = _decay_visible(sub, idx, halflife, "value_mcap")

    pct_prior = _purchase_pct_prior(ev)
    if pct_prior is not None:
        out["ic_insider_purchase_pct_prior"] = _decay_weighted_mean(pct_prior, idx, halflife, "pct_prior", "value")
    surprise = _owner_surprise(ev)
    if surprise is not None:
        out["ic_insider_owner_surprise_120d"] = _decay_weighted_mean(surprise, idx, halflife, "surprise", "value")
    return out


def _decay_weighted_mean(ev: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float, value_col: str, weight_col: str) -> pd.DataFrame:
    """Weighted mean of `value_col` whose weights are `weight_col` times the decay.

    Both legs go through the same `decay_events`, so the NaN-before-first-event region and
    the trading-day clock are inherited rather than re-implemented, and the quotient stays in
    the units of `value_col` however many events have accumulated.
    """
    w = ev.copy()
    vals = pd.to_numeric(w[value_col], errors="coerce")
    w["_den"] = pd.to_numeric(w[weight_col], errors="coerce")
    w = w[(w["_den"] > 0) & vals.notna()]
    vals = vals.reindex(w.index)
    if w.empty:
        return pd.DataFrame(index=idx)
    # ⚠ THE OFFSET IS NOT COSMETIC. `decay_events` masks everything before a ticker's first
    # event with `cumsum(magnitude) > 0`, so a genuine event whose magnitude is exactly 0 --
    # a purchase that is the SMALLEST that owner has ever made, surprise 0.0 -- reads as "no
    # event has happened" and the whole feature disappears. Shifting the numerator above
    # zero and subtracting the shift back is exact, because decay is linear:
    #     decay(w(v + c)) / decay(w) - c  ==  decay(wv) / decay(w)
    offset = 1.0 + max(0.0, -float(vals.min()))
    w["_num"] = (vals + offset) * w["_den"]
    num = _decay_visible(w, idx, halflife, "_num")
    den = _decay_visible(w, idx, halflife, "_den")
    return ((num / den.where(den != 0)) - offset).replace([np.inf, -np.inf], np.nan)


def _decay_visible(records: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float, magnitude_col: str) -> pd.DataFrame:
    """`decay_events` of `magnitude_col` with each record counted while visible and aged from its anchor.

    Plain records are `decay_events` at the anchor. A record with an open interval enters at
    `visible_from` already decayed by the trading days since its anchor; a closed one is summed
    directly over its visible days, so nothing is left behind after `visible_until`. A ticker
    is NaN until its first visible record with a positive magnitude.
    """
    if _plain(records).all():
        return decay_events(records.assign(date=records["anchor"]), idx, halflife, magnitude_col=magnitude_col)
    grid = pd.DatetimeIndex(idx).normalize().unique().sort_values()
    magnitude = pd.to_numeric(records[magnitude_col], errors="coerce")
    pos_anchor = _grid_pos(records["anchor"], grid)
    pos_from = _grid_pos(records["visible_from"], grid)
    is_open = records["visible_until"].isna().to_numpy()

    entering = magnitude * 0.5 ** ((pos_from - pos_anchor) / halflife)
    df_open = records.loc[is_open, ["ticker", "visible_from"]].assign(date=records["visible_from"], _m=entering)
    opened = decay_events(df_open, grid, halflife, magnitude_col="_m")

    df_closed = records.loc[~is_open & magnitude.notna().to_numpy(), ["ticker"]]
    tickers = pd.Index(sorted(set(opened.columns.astype(str)) | set(df_closed["ticker"].astype(str))), name="ticker")
    closed = np.zeros((len(grid), len(tickers)), dtype="float64")
    started = np.zeros_like(closed, dtype=bool)
    if not df_closed.empty:
        keep = records.index.get_indexer(df_closed.index)
        col = tickers.get_indexer(pd.Index(df_closed["ticker"].astype(str)))
        start, stop = pos_from[keep], _grid_pos(records["visible_until"], grid)[keep]
        rows, cols, vals = _spread(start, stop, col, magnitude.to_numpy(dtype="float64")[keep])
        ages = rows - np.repeat(pos_anchor[keep], np.clip(stop - start, 0, None))
        np.add.at(closed, (rows, cols), vals * 0.5 ** (ages / halflife))
        first = (start < stop) & (magnitude.to_numpy()[keep] > 0)
        started[start[first], col[first]] = True
    has_event = opened.notna().reindex(index=grid, columns=tickers, fill_value=False).to_numpy() | np.maximum.accumulate(started, axis=0)
    frame = opened.reindex(index=grid, columns=tickers).fillna(0.0) + closed
    frame.index.name = None
    return frame.where(has_event)


def _breadth_fields(buys: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """#31 distinct buyers in 120 days, and #32 the same count gated at `CLUSTER_MIN`.

    ⚠ NEITHER IS DECAYED, AND THAT IS A DELIBERATE DEPARTURE FROM D13. Two reasons, and the
    first is decisive:

      * A DECAYED DISTINCT COUNT IS NOT A DISTINCT COUNT. `decay_events` sums events, so
        thirteen Horizon Kinetics accounts buying `TPL` every trading day since 2019
        accumulate to a "distinct buyer" reading of **91.4** -- the saturation value of a
        daily event stream at a 63-day half-life -- against 13 owners who have ever bought
        it. Gating that inflated number and decaying it again made #32 read **7,513**. The
        quantity the registry asks for is a count of PEOPLE, and only a rolling distinct
        count produces one.
      * D13 exists because a sparse flag through `peer_relative` is ~100% NaN on a
        zero-dispersion basket. This family emits NO `_vs_peers` leg (D25), so that failure
        mode cannot occur here, and the `raw` / `_xs` legs are both well defined on a
        mostly-zero column.

    Occupancy is 18.5% of ticker-days for any buy in 120 days and 6.9% for two or more, both
    above the 5% coverage floor, so the undecayed columns clear D14 rule 1 on their own.
    """
    if buys.empty or "owner_cik" not in buys.columns:
        return {}
    window_end = buys["anchor"] + pd.Timedelta(days=CLUSTER_WINDOW_DAYS)
    counted = buys.assign(
        start=buys["visible_from"],
        stop=window_end.where(buys["visible_until"].isna() | (window_end <= buys["visible_until"]), buys["visible_until"]),
    )
    per_owner = counted.drop_duplicates(["start", "stop", "ticker", "owner_cik"])
    frames = {tkr: _rolling_distinct(g, idx) for tkr, g in per_owner.groupby("ticker")}
    grid = pd.DataFrame({t: s for t, s in frames.items() if s is not None})
    if grid.empty:
        return {}
    return {
        "ic_insider_distinct_buyers_120d": grid,
        # Gated, not masked: below the threshold the answer is "no cluster" = 0, which is
        # a fact, while before the first purchase it is unknown = NaN, inherited above.
        "ic_insider_cluster_buy_120d": grid.where(grid >= CLUSTER_MIN, 0.0).where(grid.notna()),
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


def _purchase_pct_prior(ev: pd.DataFrame) -> pd.DataFrame | None:
    """`shares bought / shares held before the trade`, per purchase.

    DIRECT holdings only. An indirect row reports the whole vehicle's stake in
    `shares_owned_after` -- a trust, a family LP, a fund -- so the ratio would measure the
    vehicle, not the person's conviction. Measured on `P` rows: `shares_owned_after` is
    99.9% filled and the prior holding is positive on 92.5%, so the direct-only restriction
    is the binding one, not the arithmetic.
    """
    if "shares_owned_after" not in ev.columns:
        return None
    owned = pd.to_numeric(ev["shares_owned_after"], errors="coerce")
    prior = owned - ev["shares_n"]
    ok = prior > 0
    if "direct_indirect" in ev.columns:
        ok &= ev["direct_indirect"].astype(str).str.upper().eq("D")
    if not ok.any():
        return None
    out = ev[ok].copy()
    out["pct_prior"] = out["shares_n"] / prior[ok]
    return out


def _owner_surprise(ev: pd.DataFrame) -> pd.DataFrame | None:
    """Each purchase's percentile among that OWNER's own earlier purchases.

    ⚠ THE EASIEST PLACE IN THE WHOLE FAMILY TO LEAK THE FUTURE, so the window is expanding
    and strictly prior: the rank of purchase `i` counts only purchases `j < i` in filing
    order. A full-sample percentile would rank today's buy against buys that have not
    happened, and it would look almost identical -- which is why the test for this is a
    comparison against a deliberately-leaky version, not an eyeball.

    An owner's FIRST purchase has no prior distribution and is NaN, never 0.5: "unusually
    large for this person" is undefined before there is a person to compare against.

    The prior set of a record is that owner's earlier-anchored records visible on its own
    `visible_from`, so a later restatement never re-ranks it.
    """
    if "owner_cik" not in ev.columns:
        return None
    d = ev.dropna(subset=["value"]).sort_values(["owner_cik", "anchor"], kind="stable")
    if d.empty:
        return None
    restated = d["owner_cik"].isin(set(d.loc[~_plain(d), "owner_cik"]))
    d["surprise"] = np.nan
    if (~restated).any():
        g = d[~restated].groupby("owner_cik", sort=False)["value"]
        # `rank(pct=True)` over the expanding window includes the current row, so the first
        # observation is always 1.0 and every later one is inflated by 1/n. Subtracting the
        # self-contribution rescales to "fraction of PRIOR purchases at or below this one".
        n = g.cumcount()
        expanding_rank = g.expanding().apply(lambda s: (s.iloc[:-1] <= s.iloc[-1]).sum(), raw=False).reset_index(level=0, drop=True)
        d.loc[~restated, "surprise"] = (expanding_rank / n.where(n > 0)).astype("float64")
    if restated.any():
        d.loc[restated, "surprise"] = _visible_prior_rank(d[restated])
    return d.dropna(subset=["surprise"])


def _visible_prior_rank(d: pd.DataFrame) -> pd.Series:
    """Fraction of each record's prior set (see `_owner_surprise`) at or below its value; `d` is sorted by owner then anchor."""
    df_rec = d[["owner_cik", "value", "visible_from", "visible_until"]].assign(order=np.arange(len(d)))
    df_pair = df_rec.merge(df_rec, on="owner_cik", suffixes=("", "_prior"))
    prior = (
        (df_pair["order_prior"] < df_pair["order"])
        & (df_pair["visible_from_prior"] <= df_pair["visible_from"])
        & (df_pair["visible_until_prior"].isna() | (df_pair["visible_until_prior"] > df_pair["visible_from"]))
    )
    n_prior = prior.groupby(df_pair["order"]).sum().reindex(df_rec["order"], fill_value=0).to_numpy()
    n_at_or_below = (
        (prior & (df_pair["value_prior"] <= df_pair["value"])).groupby(df_pair["order"]).sum().reindex(df_rec["order"], fill_value=0).to_numpy()
    )
    return pd.Series(n_at_or_below / np.where(n_prior > 0, n_prior, np.nan), index=d.index, dtype="float64")


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


def _market_cap(shares_out_history: pd.DataFrame | None, stock_close: pd.DataFrame | None, level_factor: pd.DataFrame | None) -> pd.DataFrame | None:
    if shares_out_history is None or shares_out_history.empty or stock_close is None or stock_close.empty:
        _log.warning("No shares outstanding or close -> the 11 size-scaled insider features are skipped.")
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
            " -> the 11 size-scaled insider features are skipped.",
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
