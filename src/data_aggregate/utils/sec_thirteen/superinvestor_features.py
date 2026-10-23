"""Elite-manager whole-book conviction, existing comparison rules and decayed events."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.decay import decay_events
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.holdings_clean import (
    attach_tickers as attach_tickers,
)
from src.data_aggregate.utils.sec_thirteen.holdings_clean import (
    clean_holdings as clean_holdings,
)
from src.data_aggregate.utils.sec_thirteen.holdings_clean import (
    clean_legacy_manager_holdings,
    clean_public_holdings,
)
from src.data_aggregate.utils.sec_thirteen.holdings_clean import (
    legacy_manager_public_dates as _as_of_stamp,
)
from src.data_aggregate.utils.sec_thirteen.institutional_vintages import iter_public_books
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily
from src.data_store.schema import Tables
from src.utils.institutional_vintages import filing_public_dates
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

#: Ranks at or below this count as a "top holding" for #17/#18/#22.
_TOP_N = 10

#: Corporate-action guard for the aggregate share ratio, used ONLY where `prices_splits`
#: supplied no factor. A QoQ share ratio landing within `_SPLIT_TOL` of one of these (or its
#: reciprocal) is read as a split, not a trade, and the delta is nulled. Deliberately a short
#: list of the ratios issuers actually declare: a wider one would start eating real trades.
_SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
_SPLIT_TOL = 0.01

#: Fallback for `stale_quarters`: how long after a manager's last PUBLIC filing their book
#: keeps counting. A 13F filer who has not filed in a year has deregistered or dropped below
#: the $100M threshold; forward-filling their last book forever would leave a phantom holder
#: in every name they ever owned -- and the roster is 23 managers deep in exactly that
#: situation. A TUNABLE, so `build_cube.institutionals.superinvestor.stale_quarters` is the
#: real home; this constant is only what the function signatures default to when no config
#: reaches them (tests, notebooks).
_STALE_QUARTERS = 4

#: Elite-manager levels, changes, counts, scores, and event intensities remain in their
#: economic units. This family has no defensible sector-relative or universe-rank leg.
EMISSION: dict[str, str] = {
    "ic_super_holders": "raw",  # a SHARE of the eligible pool (D28)
    "ic_super_breadth_chg": "raw",  # 96% ties under top_k -- see below
    "ic_super_conviction_weight": "raw",  # a weighted-AVERAGE weight in [0,1] (D28)
    "ic_super_max_conviction": "raw",  # a portfolio weight in [0,1]
    "ic_super_conviction_chg": "raw",
    "ic_super_conviction_weight_yoy": "raw",
    "ic_super_holders_yoy": "raw",  # 97% ties under top_k
    "ic_super_top10_holders": "raw",  # 98% ties under top_k
    "ic_super_quarters_held": "raw",
    "ic_super_sp500_share": "raw",
    "ic_super_selection_score": "raw",  # a mean of `sel` in [0,1]; see #27
    "ic_super_shares_chg": "raw",
    "ic_super_new_top10": "raw",  # decayed intensities, all five
    "ic_super_rank_jump": "raw",
    "ic_super_initiations": "raw",
    "ic_super_full_exits": "raw",
}

#: The features above marked "decayed intensities" are the SPARSE (class-S) ones. An
#: event is 0 on almost every ticker-day, so each goes through `decay_events` before the
#: panel -- see `decay.py` for why an undecayed flag is an ABSENT feature, not a weak one.
#: They are produced by `_events` and named there; no second list is kept, because two
#: declarations of one membership rule is how a family drifts out of sync with its emission
#: map.


def _selection_ciks(roster: Iterable[str] | None) -> set[str]:
    """Padded CIKs from roster-map keys or an iterable of manager IDs."""
    if not roster:
        return set()
    return {c for m in roster if (c := pad_cik(m))}


def manager_quarter_state(holdings: pd.DataFrame) -> pd.DataFrame:
    """Calculate whole-book manager statistics and universe support per report quarter."""

    if holdings.empty:
        return pd.DataFrame()

    g = holdings.groupby(["cik", "period"], sort=True)
    state = g.agg(total_common_value=("value_usd", "sum"), n_positions=("cusip", "nunique")).reset_index()
    unknown = g["value_usd"].count().lt(g.size()).to_numpy()
    state.loc[unknown, "total_common_value"] = np.nan

    val = holdings[["cik", "period", "value_usd"]].copy()
    val = val.merge(state[["cik", "period", "total_common_value"]], on=["cik", "period"])
    tot = val["total_common_value"].where(val["total_common_value"] > 0)
    val["w"] = val["value_usd"] / tot

    conc = (
        val.groupby(["cik", "period"])["w"]
        .agg(
            eff_n=lambda s: 1.0 / float((s**2).sum()) if float((s**2).sum()) > 0 else np.nan,
            top1_weight="max",
            top5_weight=lambda s: float(s.nlargest(5).sum()),
            top10_weight=lambda s: float(s.nlargest(_TOP_N).sum()),
        )
        .reset_index()
    )
    state = state.merge(conc, on=["cik", "period"], how="left")
    state.loc[state["total_common_value"].isna(), ["eff_n", "top1_weight", "top5_weight", "top10_weight"]] = np.nan

    if "filing_date" in holdings.columns:
        lag = g["filing_date"].max().reset_index()
        lag["filing_lag_days"] = (lag["filing_date"] - lag["period"]).dt.days
        state = state.merge(lag[["cik", "period", "filing_date", "filing_lag_days"]], on=["cik", "period"], how="left")
    else:
        state["filing_date"], state["filing_lag_days"] = pd.NaT, np.nan

    # The S&P500 share of each book -- the coverage spread README Finding 2 measured, shipped
    # as feature #26 rather than left as an invisible bias.
    if "ticker" in holdings.columns:
        g_idx = holdings[holdings["ticker"].notna()].groupby(["cik", "period"])
        idx = g_idx.agg(sp500_value=("value_usd", "sum"), n_index_positions=("ticker", "nunique")).reset_index()
        state = state.merge(idx, on=["cik", "period"], how="left")
        state["sp500_value"] = state["sp500_value"].fillna(0.0)
        # `n_index_positions` is NOT a concentration measure -- it is the eligibility floor
        # `manager_selection` applies, because a manager holding two universe names cannot
        # move a consensus basket whatever their whole-book concentration says.
        state["n_index_positions"] = state["n_index_positions"].fillna(0).astype("int64")
        denom = state["total_common_value"].where(state["total_common_value"] > 0)
        state["sp500_share"] = state["sp500_value"] / denom
    return state


def manager_stock_conviction(holdings: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Compute whole-book portfolio weights and deterministic stock ranks."""
    if holdings.empty or state.empty:
        return pd.DataFrame()
    c = holdings.merge(state[["cik", "period", "total_common_value"]], on=["cik", "period"], how="left")
    denom = c["total_common_value"].where(c["total_common_value"] > 0)
    c["portfolio_weight"] = c["value_usd"] / denom
    # ⚠ THE TIE-BREAK IS `cusip`, NOT ROW ORDER. `rank(method="first")` numbers equal values
    # by their position in the frame, so two positions of identical `value_usd` swapped
    # ranks whenever anything upstream changed the row order -- measured at 2.06% of
    # manager-quarter-cusip rows, with pairs like 38<->39 at an identical 0.000089 weight.
    # Ranks feed `is_top10`, `new_top10` and `rank_jump`, so an order-coupled rank makes
    # those features non-reproducible. Sorting on (value desc, cusip asc) and counting gives
    # a strict total order that depends only on the filing's contents.
    c = c.sort_values(["cik", "period", "value_usd", "cusip"], ascending=[True, True, False, True])
    c["rank_in_book"] = c.groupby(["cik", "period"]).cumcount() + 1
    c["conviction_pct"] = c.groupby(["cik", "period"])["value_usd"].rank(pct=True)
    c["rank_in_book"] = c["rank_in_book"].where(denom.reindex(c.index).notna() & c["value_usd"].notna())
    c["conviction_pct"] = c["conviction_pct"].where(denom.reindex(c.index).notna())
    c["is_top10"] = c["rank_in_book"] <= _TOP_N
    return c


def _selection_series(state: pd.DataFrame, selection: pd.Series | Callable | None) -> pd.Series:
    """Align supplied manager-quarter weights, preserving intentional exclusions."""
    idx = pd.MultiIndex.from_frame(state[["cik", "period"]])
    if selection is None:
        return pd.Series(1.0, index=idx, name="sel")
    if callable(selection):
        selection = selection(state)
    sel = pd.Series(selection).reindex(idx).astype("float64").fillna(0.0)
    sel.name = "sel"
    return sel


def attach_split_factor(contrib: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """Restate preceding shares using genuine splits in the comparison interval."""
    out = contrib.copy()
    out["prev_shares_adj"] = out["prev_shares"]
    if splits is None or splits.empty or "prev_period" not in out.columns:
        return out
    s = splits.dropna(subset=["date", "ticker", "ratio"]).copy()
    # ⚠ BOTH SIDES FORCED TO ns. `prices_splits.date` is a Postgres TIMESTAMP and arrives as
    # `datetime64[us]`, while the periods built here are `datetime64[s]`; `merge_asof`
    # refuses to join two datetime resolutions rather than coercing them.
    s["date"] = to_day(s["date"]).astype("datetime64[ns]")
    s = s[s["ticker"].isin(out["ticker"].unique())].sort_values(["ticker", "date"])
    if s.empty:
        return out
    s["cum"] = s.groupby("ticker")["ratio"].cumprod()
    # `merge_asof` requires BOTH sides globally sorted on the `on` key, not merely sorted
    # within each `by` group -- it raises "keys must be sorted" otherwise.
    right = s[["ticker", "date", "cum"]].sort_values("date")
    factor = _cumulative_splits(out["ticker"], out["period"], right) / _cumulative_splits(out["ticker"], out["prev_period"], right)
    out["prev_shares_adj"] = out["prev_shares"] * factor
    n = int((factor != 1.0).sum())
    if n:
        logger.info("split restatement: %s manager-quarters had their prior share count rebased (a split fell between the two periods)", n)
    return out


def _cumulative_splits(tickers: pd.Series, dates: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """Align cumulative splits to positions; absent prior dates cannot carry shares."""
    left = pd.DataFrame({"ticker": tickers.to_numpy(), "_d": pd.to_datetime(dates).astype("datetime64[ns]")})
    left["_d"] = left["_d"].fillna(pd.Timestamp("1900-01-01"))
    left["_i"] = np.arange(len(left))
    merged = pd.merge_asof(left.sort_values("_d"), splits, left_on="_d", right_on="date", by="ticker", direction="backward")
    return merged.sort_values("_i")["cum"].fillna(1.0).to_numpy()


def _corporate_action_mask(ratio: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing ratio guard without changing economic units."""
    mask = pd.DataFrame(False, index=ratio.index, columns=ratio.columns)
    for f in _SPLIT_RATIOS:
        for target in (f, 1.0 / f):
            mask |= (ratio - target).abs() <= _SPLIT_TOL * target
    return mask & ratio.notna()


def public_state(state: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """Combine manager statistics and weights at their public observation dates."""
    st = state[["cik", "period"]].copy()
    st["sel"] = sel.to_numpy()
    st["avail"] = (state["avail"] if "avail" in state.columns else _as_of_stamp(state)).to_numpy()
    st = st.sort_values(["cik", "period"])
    superseded = st[::-1].groupby("cik")["avail"].cummin()[::-1].groupby(st["cik"]).shift(-1)
    st = st[st["avail"] < superseded.fillna(pd.Timestamp.max)].copy()
    st["prev_sel"] = st.groupby("cik")["sel"].shift(1)
    st["seq"] = st.groupby("cik").cumcount()
    return st


def _contributions(conv: pd.DataFrame, state: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """Build manager-stock history, observed-row persistence and comparison inputs."""
    c = conv[conv["ticker"].notna()].copy()
    if c.empty or state.empty:
        return pd.DataFrame()
    key = pd.MultiIndex.from_frame(c[["cik", "period"]])
    c["sel"] = sel.reindex(key).to_numpy()
    # one manager can hold two CUSIPs of one issuer (share classes): sum the position and
    # keep the BEST rank -- "is this a top-ten name for them" is about the issuer.
    c = c.groupby(["cik", "period", "ticker"], as_index=False).agg(
        sel=("sel", "first"),
        w=("portfolio_weight", lambda values: values.sum(skipna=False)),
        shares=("shares", "sum"),
        value_usd=("value_usd", lambda values: values.sum(skipna=False)),
        rank_in_book=("rank_in_book", "min"),
    )

    st = public_state(state, sel)

    # ⚠ DROP THE FILINGS THAT ARE NEVER THE MANAGER'S PUBLIC STATE. A manager's state on
    # date d is the LATEST PERIOD among the filings public by then, so a period `q` matters
    # only on `[avail(q), min over later periods of avail)`. A 13F back-filed for 2013 and
    # submitted in 2016 arrives after the 2014-2016 filings that supersede it, so that
    # window is empty and the row can never be read -- keeping it would let an old period
    # overwrite a newer one, a look-ahead in reverse.
    #
    # A running MAXIMUM of `avail` was the first attempt and it is subtly wrong in the other
    # direction: it also delays an on-time filing that happens to follow a late one, so
    # removing a future filing changed a past value. That is what the truncation test
    # catches, and it is why the rule is stated as "is this row ever the state" rather than
    # "force availability to be monotone".

    # the full (manager x their filed periods x names they ever held) frame
    pairs = c[["cik", "ticker"]].drop_duplicates()
    full = pairs.merge(st, on="cik", how="left")
    full = full.merge(c.drop(columns=["sel"]), on=["cik", "period", "ticker"], how="left")
    full["held"] = full["shares"].gt(0)
    for col in ("w", "shares", "value_usd"):
        full.loc[~full["held"], col] = 0.0
    full["is_top10"] = full["rank_in_book"].le(_TOP_N).where(full["w"].notna(), False)
    full = full.sort_values(["cik", "ticker", "seq"])

    g = full.groupby(["cik", "ticker"], sort=False)
    for col in ("sel", "held", "w", "shares", "value_usd", "rank_in_book", "is_top10"):
        full[f"prev_{col}"] = g[col].shift(1)
    full["w4"] = g["w"].shift(4)
    full["held4"] = g["held"].shift(4)
    full["sel4"] = g["sel"].shift(4)
    # consecutive quarters held, ending at this period: `(~held).cumsum()` labels each
    # unbroken streak, so a running count inside the label IS the streak length and a
    # repurchase after a gap correctly restarts at 1.
    full["run_len"] = (full.groupby(["cik", "ticker", (~full["held"]).cumsum()], sort=False).cumcount() + 1).where(full["held"])
    if "sp500_share" in state.columns:
        full = full.merge(state[["cik", "period", "sp500_share"]], on=["cik", "period"], how="left")
    full["prev_period"] = full.groupby(["cik", "ticker"], sort=False)["period"].shift(1)
    return full


def _effective(
    contrib: pd.DataFrame, column: str, grid: pd.DatetimeIndex, pairs: pd.MultiIndex, stale_after: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Project manager observations with the existing staleness and absence rules."""
    wide = (
        contrib.pivot_table(index="avail", columns=["cik", "ticker"], values=column, aggfunc="last", dropna=False)
        .reindex(index=grid, columns=pairs)
        .ffill()
    )
    known = (
        contrib.assign(_known=contrib[column].notna().astype(float))
        .pivot_table(index="avail", columns=["cik", "ticker"], values="_known", aggfunc="last", dropna=False)
        .reindex(index=grid, columns=pairs)
        .ffill()
    )
    wide = wide.where(known.eq(1.0))
    return wide.where(stale_after) if stale_after is not None else wide


def _project_positions(
    contrib: pd.DataFrame, columns: tuple[str, ...], stale_quarters: int
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, pd.Index]:
    """Project each required position field once using the effective row's expiry."""
    grid = pd.DatetimeIndex(sorted(contrib["avail"].dropna().unique()))
    pairs = pd.MultiIndex.from_frame(contrib[["cik", "ticker"]].drop_duplicates().sort_values(["cik", "ticker"]), names=["cik", "ticker"])
    tickers = pd.Index(sorted(contrib["ticker"].unique()), name="ticker")
    fresh = _effective(contrib.assign(_a=contrib["avail"]), "_a", grid, pairs)
    horizon = fresh + pd.DateOffset(months=3 * stale_quarters)
    live = horizon.ge(pd.Series(grid, index=grid), axis=0)
    return {column: _effective(contrib, column, grid, pairs, live) for column in columns}, grid, tickers


def _across_tickers(frame: pd.DataFrame, tickers: pd.Index, how: str = "sum") -> pd.DataFrame:
    """Reduce manager columns in the original grouping and ticker order."""
    agg = getattr(frame.T.groupby(level="ticker"), how)()
    return agg.T.reindex(columns=tickers)


def _current_super_levels(effective: dict[str, pd.DataFrame], tickers: pd.Index, pool: pd.Series) -> dict[str, pd.DataFrame]:
    """Calculate the level equations shared by both temporal views."""
    sel, held, weight = effective["sel"], effective["held"], effective["w"]
    # The eligible pool comes from manager states, never future ticker membership.
    held_mask = held.fillna(False).astype(bool)
    sp500 = effective["sp500_share"].where(held_mask)
    sp_num = _across_tickers(sel.where(held_mask) * sp500, tickers)
    sp_den = _across_tickers(sel.where(held_mask & sp500.notna()), tickers)
    levels = {
        "ic_super_holders": _across_tickers(sel * held, tickers).div(pool, axis=0),
        "ic_super_conviction_weight": _across_tickers(sel * weight, tickers).div(pool, axis=0),
        "ic_super_max_conviction": _across_tickers(weight.where(held.fillna(False)), tickers, "max"),
        "ic_super_top10_holders": _across_tickers(sel * effective["is_top10"], tickers),
        "ic_super_quarters_held": _across_tickers(effective["run_len"], tickers, "median"),
        "ic_super_sp500_share": sp_num / sp_den.where(sp_den > 0),
    }
    held_num = held.fillna(0.0)
    if float(np.nanstd(sel.to_numpy())) > 0:
        sel_den = _across_tickers(held_num, tickers)
        levels["ic_super_selection_score"] = _across_tickers(sel.fillna(0.0) * held_num, tickers) / sel_den.where(sel_den > 0)
    else:
        logger.info(
            "`sel` is flat -> `ic_super_selection_score` would be constant and is not emitted; it becomes live under a point-in-time selector."
        )
    unknown = _across_tickers((held_mask & weight.isna() & sel.gt(0)).astype(float), tickers).gt(0)
    for feature in ("ic_super_conviction_weight", "ic_super_max_conviction", "ic_super_top10_holders", "ic_super_sp500_share"):
        levels[feature] = levels[feature].mask(unknown)
    return levels


def _legacy_super_changes(effective: dict[str, pd.DataFrame], tickers: pd.Index, pool: pd.Series, prev_pool: pd.Series) -> dict[str, pd.DataFrame]:
    """Keep the legacy observation-based comparison equations and residual split guard."""
    sel, held, weight = effective["sel"], effective["held"], effective["w"]
    prev_sel, prev_held, prev_weight = effective["prev_sel"], effective["prev_held"], effective["prev_w"]
    holders = _across_tickers(sel * held, tickers).div(pool, axis=0)
    prev_holders = _across_tickers(prev_sel * prev_held, tickers).div(prev_pool, axis=0)
    conviction = _across_tickers(sel * weight, tickers).div(pool, axis=0)
    prev_conviction = _across_tickers(prev_sel * prev_weight, tickers).div(prev_pool, axis=0)
    conviction4 = _across_tickers(effective["sel4"] * effective["w4"], tickers).div(pool, axis=0)
    holders4 = _across_tickers(effective["sel4"] * effective["held4"], tickers).div(pool, axis=0)
    both = held.fillna(False).astype(bool) & prev_held.fillna(False).astype(bool)
    sh_now = _across_tickers((sel * effective["shares"]).where(both), tickers)
    sh_prev = _across_tickers((prev_sel * effective["prev_shares_adj"]).where(both), tickers)
    ratio = sh_now / sh_prev.where(sh_prev > 0)
    guard = _corporate_action_mask(ratio)
    if int(guard.to_numpy().sum()):
        logger.info(
            "share-change guard: %s ticker-dates nulled as residual corporate actions after the `prices_splits` restatement",
            int(guard.to_numpy().sum()),
        )
    unknown = _across_tickers((held.fillna(False).astype(bool) & weight.isna() & sel.gt(0)).astype(float), tickers).gt(0)
    unknown_previous = _across_tickers((prev_held.fillna(False).astype(bool) & prev_weight.isna() & prev_sel.gt(0)).astype(float), tickers).gt(0)
    unknown_year = _across_tickers(
        (effective["held4"].fillna(False).astype(bool) & effective["w4"].isna() & effective["sel4"].gt(0)).astype(float), tickers
    ).gt(0)
    return {
        "ic_super_breadth_chg": holders - prev_holders,
        "ic_super_conviction_chg": (conviction - prev_conviction).mask(unknown | unknown_previous),
        "ic_super_holders_yoy": holders - holders4,
        "ic_super_conviction_weight_yoy": (conviction - conviction4).mask(unknown | unknown_year),
        "ic_super_shares_chg": (ratio - 1.0).where(~guard),
    }


def _visible_super_levels(levels: dict[str, pd.DataFrame], seen_tickers: Iterable[str] | None) -> dict[str, pd.DataFrame]:
    """Hide names until their first supported holding."""
    holders = levels["ic_super_holders"]
    seen = (holders.fillna(0.0) > 0).cummax()
    if seen_tickers is not None:
        seen.loc[:, seen.columns.isin(seen_tickers)] = True
    return {name: frame.where(seen) for name, frame in _order_super_levels(levels).items()}


def _order_super_levels(levels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep the public column order after independent level and comparison reductions."""
    order = (
        "ic_super_holders",
        "ic_super_conviction_weight",
        "ic_super_max_conviction",
        "ic_super_top10_holders",
        "ic_super_breadth_chg",
        "ic_super_conviction_chg",
        "ic_super_holders_yoy",
        "ic_super_conviction_weight_yoy",
        "ic_super_shares_chg",
        "ic_super_quarters_held",
        "ic_super_sp500_share",
        "ic_super_selection_score",
    )
    return {name: levels[name] for name in order if name in levels} | {name: frame for name, frame in levels.items() if name not in order}


def _aggregate(
    contrib: pd.DataFrame,
    st: pd.DataFrame,
    stale_quarters: int = _STALE_QUARTERS,
    *,
    seen_tickers: Iterable[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Combine shared levels with the legacy comparison view."""
    columns = (
        "sel",
        "held",
        "w",
        "is_top10",
        "run_len",
        "sp500_share",
        "prev_sel",
        "prev_held",
        "prev_w",
        "sel4",
        "w4",
        "held4",
        "shares",
        "prev_shares_adj",
    )
    effective, grid, tickers = _project_positions(contrib, columns, stale_quarters)
    pool = _manager_pool(st, "sel", grid, stale_quarters)
    levels = _current_super_levels(effective, tickers, pool)
    levels.update(_legacy_super_changes(effective, tickers, pool, _manager_pool(st, "prev_sel", grid, stale_quarters)))
    return _visible_super_levels(levels, seen_tickers), grid


def _manager_pool(st: pd.DataFrame, column: str, grid: pd.DatetimeIndex, stale_quarters: int) -> pd.Series:
    """Aggregate the public manager pool under the existing expiry policy."""
    one = st[["cik", "avail", column]].copy()
    # `pivot_table` cannot aggregate a datetime column, so the freshness clock travels as an
    # integer nanosecond stamp and is turned back into a date after the forward-fill.
    one["_stamp"] = one["avail"].astype("datetime64[ns]").astype("int64")
    m = one.pivot_table(index="avail", columns="cik", values=column, aggfunc="last", dropna=False).reindex(index=grid).ffill()
    seen = one.pivot_table(index="avail", columns="cik", values="_stamp", aggfunc="last", dropna=False).reindex(index=grid).ffill()
    horizon = seen.apply(pd.to_datetime, unit="ns") + pd.DateOffset(months=3 * stale_quarters)
    live = horizon.ge(pd.Series(grid, index=grid), axis=0)
    total = m.where(live).sum(axis=1)
    return total.where(total > 0)


def _events(contrib: pd.DataFrame) -> pd.DataFrame:
    """Build existing entry, exit, rank and top-ten events from contribution rows."""
    held, prev_held = contrib["held"], contrib["prev_held"]
    jump = (contrib["prev_rank_in_book"] - contrib["rank_in_book"]).clip(lower=0)
    candidates = (
        ("ic_super_initiations", held & (prev_held == False), contrib["sel"] * contrib["w"]),  # noqa: E712
        ("ic_super_full_exits", (~held) & (prev_held == True), contrib["prev_sel"] * contrib["prev_w"]),  # noqa: E712
        ("ic_super_new_top10", contrib["is_top10"] & (contrib["prev_rank_in_book"] > _TOP_N), contrib["sel"]),
        ("ic_super_rank_jump", held & (jump > 0), contrib["sel"] * jump),
    )
    rows = [_event_rows(contrib, kind, mask, magnitude) for kind, mask, magnitude in candidates if mask.fillna(False).any()]
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "magnitude", "kind"])
    ev = pd.concat(rows, ignore_index=True).dropna(subset=["date", "magnitude"])
    return cast(pd.DataFrame, ev.groupby(["date", "ticker", "kind"], as_index=False)["magnitude"].sum())


def _event_rows(contrib: pd.DataFrame, kind: str, mask: pd.Series, magnitude: pd.Series) -> pd.DataFrame:
    """Keep the source order of supported event observations."""
    selected = mask.fillna(False).to_numpy(dtype=bool)
    return pd.DataFrame(
        {
            "date": contrib["avail"].to_numpy()[selected],
            "ticker": contrib["ticker"].to_numpy()[selected],
            "magnitude": magnitude.to_numpy()[selected],
            "kind": kind,
        }
    )


def _to_long(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """An availability-indexed `(date x ticker)` frame -> the `(ticker, as_of, <name>)` shape
    `fundamentals_to_daily` forward-fills onto the trading grid."""
    stacked = cast(pd.Series, frame.stack(future_stack=True))
    stacked.name = name
    long = stacked.reset_index()
    long.columns = ["as_of", "ticker", name]
    return long.dropna(subset=["as_of"])[["ticker", "as_of", name]]


def build_superinvestor_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame | None,
    roster: Iterable[str] | None,
    *,
    cusip_map: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    selection: pd.Series | Callable | None = None,
    decay_halflife: float = 63.0,
    stale_quarters: int = _STALE_QUARTERS,
    availability: InstitutionalAvailability | None = None,
    valuation_close: pd.DataFrame | None = None,
    value_basis: Mapping[str, float] | None = None,
    filings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adapt raw public or legacy inputs to the shared cleaning boundary."""
    need = {"cik", "period", "cusip", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns) or not _selection_ciks(roster):
        return pd.DataFrame(columns=["date", "ticker"])
    if filings is not None:
        df_holdings = clean_public_holdings(attach_tickers(holdings, cusip_map, frames.universe), valuation_close, value_basis)
    else:
        df_holdings = clean_legacy_manager_holdings(holdings, cusip_map, frames.universe, valuation_close, value_basis)
    return compute_superinvestor_feature_panel(
        frames,
        df_holdings,
        splits=splits,
        selection=selection,
        decay_halflife=decay_halflife,
        stale_quarters=stale_quarters,
        availability=availability,
        filings=filings,
    )


def compute_superinvestor_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame,
    *,
    splits: pd.DataFrame | None = None,
    selection: pd.Series | Callable | None = None,
    decay_halflife: float = 63.0,
    stale_quarters: int = _STALE_QUARTERS,
    availability: InstitutionalAvailability | None = None,
    filings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate elite features from cleaned books using the existing family view."""
    empty = pd.DataFrame(columns=["date", "ticker"])
    if holdings.empty:
        return empty
    if filings is not None:
        levels, state = _versioned_super_levels(frames, holdings, filings, splits, selection, stale_quarters, decay_halflife)
        if not levels:
            return empty
        family_start = min(frame.index.min() for frame in levels.values())
        events = pd.DataFrame()
    else:
        state = manager_quarter_state(holdings)
        conv = manager_stock_conviction(holdings, state)
        if state.empty or conv.empty:
            return empty
        state["avail"] = _as_of_stamp(state).to_numpy()
        sel = _selection_series(state, selection)
        st = public_state(state, sel)
        contrib = attach_split_factor(_contributions(conv, state, sel), splits)
        if contrib.empty:
            return empty
        family_start = pd.Timestamp(contrib["avail"].min()).normalize()
        levels, _ = _aggregate(contrib, st, stale_quarters)
        events = _events(contrib)
    if availability is not None:
        family_start = max(family_start, availability.source_date(Tables.sec13f_manager_holdings))
    fields = _project_super_fields(levels, events, frames, family_start, availability, decay_halflife, filings is not None)
    emission = {name: EMISSION[name] for name in fields}
    logger.info("elite 13F panel: %s features over %s managers / %s quarters", len(fields), state["cik"].nunique(), state["period"].nunique())
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)


def _project_super_fields(
    levels: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    frames: PriceFrames,
    family_start: pd.Timestamp,
    availability: InstitutionalAvailability | None,
    decay_halflife: float,
    versioned: bool,
) -> dict[str, pd.DataFrame]:
    """Project public levels and events with the existing decay and source floors."""
    trading_index = frames.trading_index
    fields = {name: nullable_value_to_daily(_to_long(frame, name), name, trading_index) for name, frame in levels.items()}
    event_features = ("ic_super_new_top10", "ic_super_rank_jump", "ic_super_initiations", "ic_super_full_exits")
    if versioned:
        for name in (name for name in event_features if name in fields):
            dates = pd.Series(np.arange(len(trading_index)), index=trading_index)
            publication = pd.Series(trading_index.get_indexer(levels[name].index), index=levels[name].index)
            age = dates - publication.reindex(trading_index).ffill()
            fields[name] = fields[name].mul(0.5 ** (age / decay_halflife), axis=0)
    for kind, df_events in events.groupby("kind") if not events.empty else []:
        df_daily = decay_events(df_events, trading_index, decay_halflife, magnitude_col="magnitude")
        if not df_daily.empty and df_daily.notna().any().any():
            fields[str(kind)] = df_daily
    for name, df_daily in fields.items():
        feature_start = family_start
        if availability is not None and name in ("ic_super_conviction_chg", "ic_super_full_exits"):
            feature_start = max(feature_start, availability.derived_date(name, dependencies=((Tables.sec13f_manager_holdings, None),)))
        fields[name] = df_daily.where(pd.Series(df_daily.index >= feature_start, index=df_daily.index), axis=0)
    return {name: df_daily for name, df_daily in fields.items() if df_daily is not None and not df_daily.empty}


def _snapshot_super_contributions(
    books: pd.DataFrame,
    states: pd.DataFrame,
    first_public: pd.Series,
    selection: pd.Series | Callable | None,
    day: pd.Timestamp,
    splits: pd.DataFrame | None,
    stale_quarters: int,
    observed_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use whole-book conviction and the latest public adjacent-quarter baseline."""
    common = books.loc[books.position_type.eq("common") | books.shares.isna()].copy()
    state = manager_quarter_state(common)
    keys = states.loc[states.book_known, ["cik", "period"]]
    state = keys.merge(state, on=["cik", "period"], how="left") if not state.empty else keys.copy()
    for col in (
        "total_common_value",
        "n_positions",
        "eff_n",
        "top1_weight",
        "top5_weight",
        "top10_weight",
        "sp500_share",
        "n_index_positions",
        "filing_lag_days",
    ):
        if col not in state:
            state[col] = np.nan
    state["n_positions"] = state.n_positions.fillna(0)
    state["avail"] = first_public.reindex(pd.MultiIndex.from_frame(state[["cik", "period"]])).to_numpy()
    sel = _selection_series(state, selection)
    conviction = manager_stock_conviction(common, state)
    observed = observed_pairs.merge(state[["cik", "period"]], on="cik")
    present = (
        pd.MultiIndex.from_frame(conviction[["cik", "period", "ticker"]])
        if not conviction.empty
        else pd.MultiIndex.from_tuples([], names=["cik", "period", "ticker"])
    )
    absent = observed.loc[~pd.MultiIndex.from_frame(observed[["cik", "period", "ticker"]]).isin(present)].copy()
    for column in ("shares", "value_usd", "portfolio_weight"):
        absent[column] = 0.0
    absent["rank_in_book"] = np.nan
    conviction = pd.concat([conviction, absent], ignore_index=True)
    contrib = _contributions(conviction, state, sel)
    if contrib.empty:
        return contrib, state, pd.DataFrame(), contrib
    contrib = attach_split_factor(contrib, splits)
    latest = states.loc[states.book_known].sort_values("period").drop_duplicates("cik", keep="last")
    active = contrib.merge(states[["cik", "period", "public_at", "comparison_eligible", "zero_eligible"]], on=["cik", "period"])
    prior = states[["cik", "period", "comparison_eligible", "zero_eligible"]].rename(
        columns={"period": "prior_period", "comparison_eligible": "prior_eligible", "zero_eligible": "prior_zero"}
    )
    active["prior_period"] = active.period.map(lambda q: (q.to_period("Q") - 1).end_time.normalize())
    active = active.merge(prior, on=["cik", "prior_period"], how="left")
    paired = active.comparison_eligible & active.prior_eligible.eq(True) & active.prev_period.eq(active.prior_period)
    current_rows = common.groupby(["cik", "period", "ticker"]).shares.agg(lambda amounts: amounts.sum(skipna=False))
    current_known = current_rows.reindex(pd.MultiIndex.from_frame(active[["cik", "period", "ticker"]])).notna().to_numpy()
    present = pd.MultiIndex.from_frame(active[["cik", "period", "ticker"]]).isin(current_rows.index)
    active["quantity_known"] = current_known | (~present & active.zero_eligible.to_numpy())
    missing_current = ~active.quantity_known
    prior_keys = pd.MultiIndex.from_frame(active[["cik", "prior_period", "ticker"]].rename(columns={"prior_period": "period"}))
    prior_known = current_rows.reindex(prior_keys).notna().to_numpy() | (~prior_keys.isin(current_rows.index) & active.prior_zero.eq(True).to_numpy())
    paired &= prior_known
    identity = common.loc[common.shares.gt(0)].groupby(["cik", "period", "ticker"]).cusip.agg(frozenset)
    current_identity = identity.reindex(pd.MultiIndex.from_frame(active[["cik", "period", "ticker"]])).reset_index(drop=True)
    prior_identity = identity.reindex(prior_keys).reset_index(drop=True)
    paired &= ~(current_identity.notna() & prior_identity.notna() & current_identity.ne(prior_identity)).to_numpy()
    active["shares"] = current_rows.reindex(pd.MultiIndex.from_frame(active[["cik", "period", "ticker"]])).to_numpy()
    active.loc[~present & active.zero_eligible, "shares"] = 0.0
    active.loc[missing_current, ["shares", "w", "value_usd"]] = np.nan
    active.loc[~paired, ["prev_sel", "prev_w", "prev_shares_adj", "prev_held", "prev_rank_in_book"]] = np.nan
    year_keys = pd.MultiIndex.from_arrays(
        [
            active.cik,
            active.period.map(lambda q: (q.to_period("Q") - 4).end_time.normalize()),
            active.ticker,
        ],
        names=["cik", "period", "ticker"],
    )
    past = active.set_index(["cik", "period", "ticker"]).reindex(year_keys)
    active["year_known"] = (
        active.comparison_eligible.to_numpy() & past.comparison_eligible.eq(True).to_numpy() & past.quantity_known.eq(True).to_numpy()
    )
    for old, current in (("w4", "w"), ("held4", "held"), ("sel4", "sel")):
        active[old] = past[current].to_numpy()
        active.loc[~active.year_known, old] = np.nan
    historical = active.copy()
    active = active.merge(latest[["cik", "period"]], on=["cik", "period"])
    active = active.loc[active.public_at + pd.DateOffset(months=3 * stale_quarters) >= day].copy()
    active["avail"] = day
    st = public_state(state, sel)
    st = st.merge(latest[["cik", "period"]], on=["cik", "period"])
    st = st.loc[st.cik.isin(active.cik)].copy()
    st["avail"] = day
    return active, state, st, historical


def _matched_super_changes(contrib: pd.DataFrame, day: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Compare only supported adjacent observations using the existing endpoint pools."""
    levels: dict[str, pd.DataFrame] = {}
    paired = contrib.prev_sel.notna() & contrib.prev_shares_adj.notna() & contrib.shares.notna()
    matched = contrib.loc[paired]
    groups = matched.groupby("ticker")
    now = (matched.sel * matched.shares).groupby(matched.ticker).sum(min_count=1)
    old = (matched.prev_sel * matched.prev_shares_adj).groupby(matched.ticker).sum(min_count=1)
    levels["ic_super_shares_chg"] = (now / old.where(old.gt(0)) - 1).to_frame(day).T
    pool = groups.sel.sum(min_count=1).where(lambda totals: totals.gt(0))
    prior_pool = groups.prev_sel.sum(min_count=1).where(lambda totals: totals.gt(0))
    breadth = (matched.sel * matched.held).groupby(matched.ticker).sum() / pool
    previous_breadth = (matched.prev_sel * matched.prev_held).groupby(matched.ticker).sum() / prior_pool
    levels["ic_super_breadth_chg"] = (breadth - previous_breadth).to_frame(day).T
    weight = (matched.sel * matched.w).where(matched.sel.gt(0), 0).groupby(matched.ticker).agg(lambda values: values.sum(skipna=False)) / pool
    prior_weight = (matched.prev_sel * matched.prev_w).where(matched.prev_sel.gt(0), 0).groupby(matched.ticker).agg(
        lambda values: values.sum(skipna=False)
    ) / prior_pool
    levels["ic_super_conviction_chg"] = (weight - prior_weight).to_frame(day).T
    return levels, paired


def _annual_super_changes(contrib: pd.DataFrame, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Keep the existing eligible year-endpoint comparisons."""
    levels: dict[str, pd.DataFrame] = {}
    annual = contrib.loc[contrib.quantity_known & contrib.year_known]
    annual_pool = annual.groupby("ticker").sel.sum(min_count=1).where(lambda totals: totals.gt(0))
    old_annual_pool = annual.groupby("ticker").sel4.sum(min_count=1).where(lambda totals: totals.gt(0))
    levels["ic_super_holders_yoy"] = (
        (
            (annual.sel * annual.held).groupby(annual.ticker).sum() / annual_pool
            - (annual.sel4 * annual.held4).groupby(annual.ticker).sum() / old_annual_pool
        )
        .to_frame(day)
        .T
    )
    levels["ic_super_conviction_weight_yoy"] = (
        (
            (annual.sel * annual.w).where(annual.sel.gt(0), 0).groupby(annual.ticker).agg(lambda values: values.sum(skipna=False)) / annual_pool
            - (annual.sel4 * annual.w4).where(annual.sel4.gt(0), 0).groupby(annual.ticker).agg(lambda values: values.sum(skipna=False))
            / old_annual_pool
        )
        .to_frame(day)
        .T
    )
    return levels


def _versioned_super_events(
    contrib: pd.DataFrame, historical: pd.DataFrame, paired: pd.Series, day: pd.Timestamp, trading_index: pd.DatetimeIndex, decay_halflife: float
) -> dict[str, pd.DataFrame]:
    """Rebuild decayed public events prospectively when source history changes."""
    levels: dict[str, pd.DataFrame] = {}
    # Unknown books and omitted legs cannot prove an initiation or an exit.
    events = _events(historical.loc[historical.quantity_known])
    for kind in ("ic_super_initiations", "ic_super_full_exits", "ic_super_new_top10", "ic_super_rank_jump"):
        subset = events.loc[events.kind.eq(kind)] if not events.empty else events
        score = pd.Series(0.0, index=contrib.ticker.unique())
        if not subset.empty:
            elapsed = trading_index.get_loc(day) - trading_index.get_indexer(pd.DatetimeIndex(subset.date))
            score = (
                (subset.magnitude * 0.5 ** (elapsed / decay_halflife)).groupby(subset.ticker).sum(min_count=1).reindex(score.index, fill_value=0.0)
            )
        unsupported = contrib.loc[~paired, "ticker"].unique()
        score.loc[score.index.isin(unsupported)] = np.nan
        levels[kind] = score.to_frame(day).T
    return levels


def _mask_unknown_quantities(levels: dict[str, pd.DataFrame], unknown: np.ndarray) -> None:
    """Clear every affected feature when a current quantity is unsupported."""
    for feature in levels:
        levels[feature].loc[:, levels[feature].columns.isin(unknown)] = np.nan


def _snapshot_super_levels(
    contrib: pd.DataFrame,
    st: pd.DataFrame,
    historical: pd.DataFrame,
    day: pd.Timestamp,
    trading_index: pd.DatetimeIndex,
    seen_tickers: Iterable[str],
    stale_quarters: int,
    decay_halflife: float,
) -> dict[str, pd.DataFrame]:
    """Compose levels, comparisons and events without calculating overwritten legacy changes."""
    columns = ("sel", "held", "w", "is_top10", "run_len", "sp500_share")
    effective, grid, tickers = _project_positions(contrib, columns, stale_quarters)
    pool = _manager_pool(st, "sel", grid, stale_quarters)
    levels = _visible_super_levels(_current_super_levels(effective, tickers, pool), seen_tickers)
    changes, paired = _matched_super_changes(contrib, day)
    levels.update(changes)
    unknown = contrib.loc[~contrib.quantity_known, "ticker"].unique()
    _mask_unknown_quantities(levels, unknown)
    levels.update(_annual_super_changes(contrib, day))
    levels.update(_versioned_super_events(contrib, historical, paired, day, trading_index, decay_halflife))
    _mask_unknown_quantities(levels, unknown)
    return _order_super_levels(levels)


def _versioned_super_levels(
    frames: PriceFrames,
    holdings: pd.DataFrame,
    filings: pd.DataFrame,
    splits: pd.DataFrame | None,
    selection: pd.Series | Callable | None,
    stale_quarters: int,
    decay_halflife: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Freeze each historical observation from its public source prefix, including revisions."""
    public = filing_public_dates(filings, frames.trading_index, settle_days=0)
    first = filings.assign(public_at=public).loc[filings.amendment_type.isin(["original", "restatement"])]
    first_public = first.groupby(["cik", "period"]).public_at.min()
    snapshots: dict[str, list[pd.DataFrame]] = {}
    state = pd.DataFrame()
    observed_pairs = pd.DataFrame(columns=["cik", "ticker"])
    for day, books, states in iter_public_books(
        filings, holdings, frames.trading_index, settle_days=0, history_quarters=None, expiry_quarters=stale_quarters
    ):
        observed = books.loc[books.position_type.eq("common") & books.shares.gt(0) & books.ticker.notna(), ["cik", "ticker"]]
        observed_pairs = pd.concat([observed_pairs, observed], ignore_index=True).drop_duplicates()
        contrib, state, st, historical = _snapshot_super_contributions(
            books,
            states,
            first_public,
            selection,
            day,
            splits,
            stale_quarters,
            observed_pairs,
        )
        if contrib.empty or st.empty:
            _append_unknown_snapshot(snapshots, day)
            continue
        levels = _snapshot_super_levels(
            contrib, st, historical, day, frames.trading_index, observed_pairs.ticker.unique(), stale_quarters, decay_halflife
        )
        for name, frame in levels.items():
            snapshots.setdefault(name, []).append(frame)
    return (
        {name: pd.concat(parts).sort_index() for name, parts in snapshots.items()},
        state,
    )


def _append_unknown_snapshot(snapshots: dict[str, list[pd.DataFrame]], day: pd.Timestamp) -> None:
    """Invalidate carried values when no current public contribution is supported."""
    for parts in snapshots.values():
        columns = parts[-1].columns
        parts.append(pd.DataFrame(np.nan, index=[day], columns=columns))
