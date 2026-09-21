"""
superinvestor_features.py  (src/data_aggregate/utils/institutionals/superinvestor_features.py)
----------------------------------------------------------------------------------------------
Elite-manager 13F features for the Dataroma superinvestor roster.

This family complements the all-filer institutional panel by measuring whether
selected, concentrated managers are accumulating a stock. Manager opinions are
combined using portfolio-conviction weights rather than simple holder counts.

Portfolio-weight denominator
----------------------------
Conviction must be calculated from `sec13f_manager_holdings`, which contains
each manager's complete common-equity book at CUSIP grain:

    portfolio_weight = value_usd / total_common_value

Do not derive these weights from `sec13f_hr`. That table is restricted to the
investment universe, so its denominator represents only the manager's
in-universe sleeve and can materially inflate conviction. Values produced from
the old filtered denominator are not comparable with this module's output.

Computation grain
-----------------
Manager-level statistics such as portfolio weight, position count, effective
number of positions, top-10 weight, and filing lag are intermediate values,
not emitted features. They are reduced to the `(ticker, date)` cube grain
through:

    _manager_quarter_state
        -> _manager_stock_conviction
        -> _ticker_quarter_panel

Manager selection
-----------------
Selection is a weight, not necessarily a filter. Each ticker aggregate has the
form:

    sum(sel(manager, period) * quantity)

where `sel` is in `[0, 1]` and is supplied through `selection`.

* `None` gives every roster manager weight `1.0`.
* Continuous selection reweights managers without removing them.
* Top-k selection can also change ticker coverage: a stock held only by
  excluded managers is absent rather than emitted as zero.

Selection is indexed by `(cik, period)`. Every prior-quarter term must therefore
use `sel(manager, q - 1)`, not the current quarter's weight. Otherwise, a
manager entering the selected set can create false accumulation in positions
they already owned.

Availability
------------
The module emits the history supported by `sec13f_manager_holdings`, which
starts at 2011-09-30. Any family-level cutoff required to handle the separate
all-filer coverage break is applied outside this module.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.constants.constants import SEC_13F_FILING_LAG_DAYS
from src.context import Context
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.decay import decay_events
from src.data_aggregate.utils.institutionals.holdings_clean import clean_holdings
from src.data_aggregate.utils.institutionals.value_basis import repair_value_basis
from src.data_store.schema import Tables
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

#: Columns read from the manager-book table. `position_type` is required -- the conviction
#: denominator is COMMON STOCK only, so the debt / call / put legs must be identifiable and
#: excluded rather than summed into `total_common_value`.
_HOLDINGS_COLS = ["cik", "period", "filing_date", "cusip", "position_type", "shares", "value_usd"]

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

#: Emission policy for elite-manager features (D25/D27).
#:
#: This family never emits `_vs_peers`: elite-manager conviction is an absolute
#: signal, and sparse ticker coverage would make peer-relative legs almost entirely
#: missing.
#:
#: `_xs` is an additional within-date percentile leg used only when it contributes
#: meaningful cross-date normalization. Pooled raw-vs-percentile correlation is not
#: sufficient: under manager selection, low-cardinality features create large tie
#: plateaus whose percentiles mostly reflect changing panel composition.
#:
#: Count-like and heavily tied features therefore emit `raw` only:
#: `top10_holders`, `holders`, `holders_yoy`, `breadth_chg`,
#: `selection_score`, `conviction_weight`, and `conviction_chg`.
#:
#: `shares_chg`, `sp500_share`, and `quarters_held` emit `raw+xs` because their
#: selected cross-sections remain sufficiently continuous. `max_conviction` emits
#: `raw` because its raw values already preserve essentially the same pooled ranking.
#:
#: The governing criterion is measured tie rate and cardinality, not whether a
#: feature is bounded, differenced, or evaluated under a particular `top_k`.
EMISSION: dict[str, str] = {
    "ic_super_holders": "raw",  # a SHARE of the eligible pool (D28)
    "ic_super_breadth_chg": "raw",  # 96% ties under top_k -- see below
    "ic_super_conviction_weight": "raw",  # a weighted-AVERAGE weight in [0,1] (D28)
    "ic_super_max_conviction": "raw",  # a portfolio weight in [0,1]
    "ic_super_conviction_chg": "raw",
    "ic_super_conviction_weight_yoy": "raw",
    "ic_super_holders_yoy": "raw",  # 97% ties under top_k
    "ic_super_top10_holders": "raw",  # 98% ties under top_k
    "ic_super_quarters_held": "raw+xs",  # measured rho 0.933
    "ic_super_sp500_share": "raw+xs",  # measured rho 0.942
    "ic_super_selection_score": "raw",  # a mean of `sel` in [0,1]; see #27
    "ic_super_shares_chg": "raw+xs",
    "ic_super_flow_to_mcap": "raw+xs",
    "ic_super_new_top10": "raw+xs",  # decayed intensities, all five
    "ic_super_rank_jump": "raw+xs",
    "ic_super_initiations": "raw+xs",
    "ic_super_full_exits": "raw+xs",
    "ic_super_exit_after_top10": "raw+xs",
}

#: The five features above marked "decayed intensities" are the SPARSE (class-S) ones. An
#: event is 0 on almost every ticker-day, so each goes through `decay_events` before the
#: panel -- see `decay.py` for why an undecayed flag is an ABSENT feature, not a weak one.
#: They are produced by `_events` and named there; no second list is kept, because two
#: declarations of one membership rule is how a family drifts out of sync with its emission
#: map.


def load_superinvestor_holdings(context: Context, roster: dict | list | set | None) -> pd.DataFrame | None:
    """The roster managers' COMPLETE quarterly books from `sec13f_manager_holdings`.

    `cik` is written by ONE producer with `pad_cik` already applied -- verified on the live
    table, where every one of the 342,501 rows is exactly 10 characters -- so this is a plain
    `where={"cik": [...]}` push-down. The `store.distinct` + every-stored-spelling dance the
    `sec13f_hr` reader needs exists only because that table takes `cik` straight from the SEC
    submission and holds the same manager both padded and unpadded; it is not needed here and
    reintroducing it would be cargo cult.

    Returns None when the roster resolves to no manager or the table is not populated.
    """
    ciks = sorted(_selection_ciks(roster))
    if not ciks:
        return None

    return context.store.load(Tables.sec13f_manager_holdings, _HOLDINGS_COLS, where={"cik": ciks}, optional=True)


def _selection_ciks(roster: dict | list | set | None) -> set[str]:
    """Padded CIKs from a roster in any of the shapes callers hold it in: the
    `{cik: manager_name}` map `roster_map_as_of` returns, a `{"cik_to_name": {...}}` wrapper,
    the legacy `{"managers": [{"cik": ...}]}` / bare list, or a plain set of CIK strings --
    which is what `roster_cik_union` returns and what the READ SCOPE must be.

    ⚠ THE READ SCOPE IS THE UNION, NOT TODAY'S ROSTER (plan D19/D20). Loading only today's
    83 managers drops the 19 culled managers that have a book
    """

    if roster is None:
        return set()
    if isinstance(roster, set | frozenset) or (isinstance(roster, list | tuple) and all(isinstance(m, str) for m in roster)):
        return {c for m in roster if (c := pad_cik(m))}
    if isinstance(roster, dict):
        mapping = roster.get("cik_to_name")
        if mapping is None and "managers" not in roster:
            mapping = {k: v for k, v in roster.items() if isinstance(v, str)}
        if mapping:
            return {c for k in mapping if (c := pad_cik(k))}
        managers = roster.get("managers", [])
    else:
        managers = roster or []
    return {c for m in managers if (c := pad_cik(m.get("cik")))}


def attach_tickers(holdings: pd.DataFrame, cusip_map: pd.DataFrame | None, universe: Sequence[str] | None = None) -> pd.DataFrame:
    """LEFT-join `cusip_ticker_map` onto the book, adding a nullable `ticker`.

    ⚠ A LEFT JOIN, AND THE UNMAPPED ROWS ARE KEPT ON PURPOSE. They are the rest of the
    manager's book -- foreign issuers, small caps, the names never resolved to a ticker --
    and they belong in `total_common_value`. Dropping them here would silently rebuild the
    S&P500-slice denominator this module exists to replace, from the other end.

    ⚠ `universe` NARROWS THE TICKER SIDE, AND OMITTING IT IS A MEASUREMENT BUG, not just a
    memory one. `cusip_ticker_map` resolves 20,561 tickers, so an unrestricted join marks
    almost the whole book as "has a ticker": the index share (#26) reads a median 98% instead
    of the measured ~52%, and the panel carries 20k columns of which the merge keeps ~500.
    Pass the analysis universe -- `PriceFrames.universe` -- so `ticker` means "in the
    strategy's universe" and nothing else.
    """
    h = holdings.copy()
    if cusip_map is None or cusip_map.empty:
        h["ticker"] = pd.NA
        return h
    m = cusip_map.dropna(subset=["cusip", "ticker"]).drop_duplicates("cusip")
    if universe is not None:
        m = m[m["ticker"].isin(set(universe))]
    return h.merge(m[["cusip", "ticker"]], on="cusip", how="left")


def manager_quarter_state(holdings: pd.DataFrame) -> pd.DataFrame:
    """Per `(cik, period)` portfolio state -- INTERMEDIATE, never a feature, never persisted.

    `total_common_value` is the conviction denominator. The concentration columns are what
    Phase 2.2b's selector ranks managers on; they are computed here rather than in the
    selector so both the features and the selection read one definition.

    ⚠ `n_positions` and `effective_n` are WHOLE-BOOK measures, which is a different statistic
    from the S&P500-slice concentration the +3.41%/yr top-15 basket was measured on. A manager
    with 8 index names out of 200 scores maximally on the slice and poorly here. Which basis
    selects better is an open question the plan requires be measured BOTH ways before the
    selector is wired; this function supplies the whole-book leg.
    """

    if holdings.empty:
        return pd.DataFrame()

    g = holdings.groupby(["cik", "period"], sort=True)
    state = g.agg(total_common_value=("value_usd", "sum"), n_positions=("cusip", "nunique")).reset_index()

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
    """Per `(cik, period, cusip)` conviction -- the second INTERMEDIATE.

    `portfolio_weight` is `value_usd / total_common_value`: the manager's real weight in
    their whole book. `rank_in_book` is 1 for their largest position, which is what makes
    "this manager just moved the name into their top ten" expressible."""
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
    c["is_top10"] = c["rank_in_book"] <= _TOP_N
    return c


def _selection_series(state: pd.DataFrame, selection: pd.Series | Callable | None) -> pd.Series:
    """`sel(m, q)` in [0, 1], indexed by `(cik, period)`.

    Three shapes, because the caller is in three different situations:

    * ``None`` -- flat 1.0, every roster manager counts the same. What the panel emitted
      before point-in-time selection existed, and what a test that does not care about
      selection wants.
    * a ``Series`` indexed by `(cik, period)` -- a selection computed elsewhere.
    * a ``callable`` taking `state` (with `avail` attached) and returning that Series --
      what the cube step passes, because `manager_selection` needs the state this function's
      caller has just built AND a roster accessor that only the step can close over.

    A manager-quarter the selector does not score gets 0.0, not 1.0: an unscored manager is
    one the selector excluded, and defaulting them back in would silently undo the selection.
    """
    idx = pd.MultiIndex.from_frame(state[["cik", "period"]])
    if selection is None:
        return pd.Series(1.0, index=idx, name="sel")
    if callable(selection):
        selection = selection(state)
    sel = pd.Series(selection).reindex(idx).astype("float64").fillna(0.0)
    sel.name = "sel"
    return sel


def attach_split_factor(contrib: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """Restate each manager's PRIOR-quarter share count onto the current quarter's basis.

    A 20-for-1 split multiplies every holder's share count by 20 with no trade taking place,
    so `shares(q) / shares(q-1)` reads +1,900% unless the prior count is restated first.
    `prices_splits` gives the ratio on its effective date; the factor between two periods is
    the ratio of the cumulative products at each end, which is exact -- and exactness is why
    this is done per `(cik, ticker, period)` rather than over the aggregate's date grid,
    where the interval between two availability dates is about a month and would not line up
    with the quarter the prior shares belong to.
    """
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
    floor = pd.Timestamp("1900-01-01")

    def cum_at(dates: pd.Series) -> np.ndarray:
        """Cumulative split product for each `(ticker, date)`; 1.0 before the first split.

        A NaT date (the manager's first filing has no previous period) is floored rather
        than dropped: it resolves to 1.0, and that row's `prev_shares` is NaN anyway, so the
        factor it receives can never reach a feature."""
        left = pd.DataFrame({"ticker": out["ticker"].to_numpy(), "_d": pd.to_datetime(dates).astype("datetime64[ns]")})
        left["_d"] = left["_d"].fillna(floor)
        left["_i"] = np.arange(len(left))
        merged = pd.merge_asof(left.sort_values("_d"), right, left_on="_d", right_on="date", by="ticker", direction="backward")
        return merged.sort_values("_i")["cum"].fillna(1.0).to_numpy()

    factor = cum_at(out["period"]) / cum_at(out["prev_period"])
    out["prev_shares_adj"] = out["prev_shares"] * factor
    n = int((factor != 1.0).sum())
    if n:
        logger.info("split restatement: %s manager-quarters had their prior share count " "rebased (a split fell between the two periods)", n)
    return out


def _corporate_action_mask(ratio: pd.DataFrame) -> pd.DataFrame:
    """True where a QoQ share ratio still looks like a corporate action after the
    `prices_splits` factor was applied -- the fallback guard for a split the price table
    does not carry. Within 1% of 2, 3, 4, 5, 10, 20 or any of their reciprocals."""
    mask = pd.DataFrame(False, index=ratio.index, columns=ratio.columns)
    for f in _SPLIT_RATIOS:
        for target in (f, 1.0 / f):
            mask |= (ratio - target).abs() <= _SPLIT_TOL * target
    return mask & ratio.notna()


def _as_of_stamp(state: pd.DataFrame) -> pd.Series:
    """`max(period + 45d, filing_date)` per `(cik, period)` -- when the filing became public.

    The 45-day statutory deadline alone is a LEAK for a late filer: a manager who files on
    day 60 did not tell the market anything on day 45. Taking the max of the two is the
    honest availability date and never earlier than the deadline."""
    deadline = state["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    filed = state.get("filing_date")
    if filed is None:
        return deadline
    return pd.concat([deadline, pd.to_datetime(filed)], axis=1).max(axis=1)


def public_state(state: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """`(cik, period, sel, avail, seq)` for the filings that are ever a manager's PUBLIC
    STATE, in period order.

    A manager's state on date `d` is the LATEST PERIOD among the filings public by then, so
    period `q` is readable only on `[avail(q), min over later periods of avail)`. A 13F
    back-filed for 2013 and submitted in 2016 arrives after the 2014-2016 filings that
    supersede it: that window is empty and the row can never be read. Dropping it here is
    what stops an old period overwriting a newer one -- a look-ahead in reverse.

    ⚠ A running MAXIMUM of `avail` was the first attempt and is wrong in the other
    direction: it also delays an ON-TIME filing that happens to follow a late one, so
    removing a future filing changed a past value. The truncation test catches that, which
    is why the rule is "is this row ever the state" and not "force availability monotone".
    """
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
    """One row per `(cik, ticker, period)` for EVERY period the manager filed, carrying that
    manager's own previous- and year-ago-period values for the name.

    ⚠ THE EMPTY ROWS ARE THE POINT. A manager who filed in `q` but no longer holds the name
    gets a row with `held=False`, and without it a full exit is invisible: the holding simply
    stops appearing and the aggregate would forward-fill the old position forever. The frame
    is built over the manager's filed periods, not over the ticker's, for the same reason.

    `avail` is the date the row became public, `max(period + 45d, filing_date)`. It is NOT
    forced monotone in period -- `public_state` instead DROPS the filings that are never the
    manager's public state, and the comment on that call below says why a running maximum is
    the wrong instrument. A back-filed 13F -- measured here at up to 1,311 days late -- is
    removed by that supersession test, not by flattening it onto its successor.
    """
    c = conv[conv["ticker"].notna()].copy()
    if c.empty or state.empty:
        return pd.DataFrame()
    key = pd.MultiIndex.from_frame(c[["cik", "period"]])
    c["sel"] = sel.reindex(key).to_numpy()
    # one manager can hold two CUSIPs of one issuer (share classes): sum the position and
    # keep the BEST rank -- "is this a top-ten name for them" is about the issuer.
    c = c.groupby(["cik", "period", "ticker"], as_index=False).agg(
        sel=("sel", "first"),
        w=("portfolio_weight", "sum"),
        shares=("shares", "sum"),
        value_usd=("value_usd", "sum"),
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
    full["held"] = full["w"].notna()
    for col in ("w", "shares", "value_usd"):
        full[col] = full[col].fillna(0.0)
    full["is_top10"] = full["rank_in_book"].le(_TOP_N).fillna(False)
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
    """`column` for each `(cik, ticker)` as it stood on each availability date -- the
    manager's most recent PUBLIC filing, forward-filled and then dropped once stale.

    This is the point-in-time join the whole module turns on. Stamping a ticker-quarter with
    the LAST contributing manager's filing date instead was measured to delay 45 of 60
    quarters past 90 days (median 189 days) because one back-filer drags the whole
    aggregate; stamping every quarter at the 45-day deadline leaks, because 16.5% of
    manager-quarters carrying 36.7% of book value are filed after it. Only a per-manager
    stamp is both timely and leak-free.
    """
    wide = (
        contrib.pivot_table(index="avail", columns=["cik", "ticker"], values=column, aggfunc="last", dropna=False)
        .reindex(index=grid, columns=pairs)
        .ffill()
    )
    return wide.where(stale_after) if stale_after is not None else wide


def _aggregate(contrib: pd.DataFrame, st: pd.DataFrame, stale_quarters: int = _STALE_QUARTERS) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Aggregate ACROSS managers on the availability grid -> `{feature: (date x ticker)}`."""

    grid = pd.DatetimeIndex(sorted(contrib["avail"].dropna().unique()))
    pairs = pd.MultiIndex.from_frame(contrib[["cik", "ticker"]].drop_duplicates().sort_values(["cik", "ticker"]), names=["cik", "ticker"])
    tickers = pd.Index(sorted(contrib["ticker"].unique()), name="ticker")

    # A filer who has gone quiet is not still holding: a manager's contribution expires
    # `stale_quarters` quarters after the filing that is currently effective. ⚠ MEASURED
    # FROM THE EFFECTIVE ROW, never from the manager's last-ever filing -- the latter is a
    # look-ahead (it needs to know when they stopped) and it made past values change when
    # future filings were removed.
    fresh = _effective(contrib.assign(_a=contrib["avail"]), "_a", grid, pairs)
    horizon = fresh + pd.DateOffset(months=3 * stale_quarters)
    live = horizon.ge(pd.Series(grid, index=grid), axis=0)

    def eff(col: str) -> pd.DataFrame:
        return _effective(contrib, col, grid, pairs, live)

    def across(frame: pd.DataFrame, how: str = "sum") -> pd.DataFrame:
        agg = getattr(frame.T.groupby(level="ticker"), how)()
        return agg.T.reindex(columns=tickers)

    sel_e, held_e = eff("sel"), eff("held")
    prev_sel_e, prev_held_e = eff("prev_sel"), eff("prev_held")
    holds, prev_holds = sel_e * held_e, prev_sel_e * prev_held_e

    # The eligible POOL: every effective manager, whether or not they hold this name --
    # dividing by it (D28) turns a headcount that jumped 41 -> 556 at the 2013 fetch break
    # into a share that did not move.
    #
    # ⚠ BUILT FROM THE MANAGER STATE, never from the ticker-level frame. A `(cik, ticker)`
    # column exists only for names the manager EVER held, so deriving the pool from it made
    # the denominator depend on which names a manager would go on to buy -- it MOVED when an
    # unrelated future filing was removed, which is what the truncation test caught.
    pool = _manager_pool(st, "sel", grid, stale_quarters)
    prev_pool = _manager_pool(st, "prev_sel", grid, stale_quarters)

    w_e, prev_w_e = eff("w"), eff("prev_w")
    holders = across(holds).div(pool, axis=0)  # #12
    prev_holders = across(prev_holds).div(prev_pool, axis=0)
    conviction = across(sel_e * w_e).div(pool, axis=0)  # #14
    prev_conviction = across(prev_sel_e * prev_w_e).div(prev_pool, axis=0)
    conviction4 = across(eff("sel4") * eff("w4")).div(pool, axis=0)
    holders4 = across(eff("sel4") * eff("held4")).div(pool, axis=0)

    # #24 -- the COMMON-MANAGER share change. Restricting to managers present in BOTH
    # quarters is what makes it "did they accumulate" rather than "did the holder set
    # change": on the full set an entrant against a tiny prior holder produced a +301,507
    # ratio on MDLZ 2012Q4. Entries and exits are already features (#20, #21).
    both = held_e.fillna(False).astype(bool) & prev_held_e.fillna(False).astype(bool)
    sh_now = across((sel_e * eff("shares")).where(both))
    sh_prev = across((prev_sel_e * eff("prev_shares_adj")).where(both))
    ratio = sh_now / sh_prev.where(sh_prev > 0)
    guard = _corporate_action_mask(ratio)
    if int(guard.to_numpy().sum()):
        logger.info(
            "share-change guard: %s ticker-dates nulled as residual corporate " "actions after the `prices_splits` restatement",
            int(guard.to_numpy().sum()),
        )

    # #26 is a property of the managers who HOLD the name right now, so both legs are
    # masked by `held_e`. Averaging over everyone who ever held it makes the value depend on
    # the whole history rather than on today's holder set.
    held_mask = held_e.fillna(False).astype(bool)
    sp_e = eff("sp500_share").where(held_mask)
    sp_num = across(sel_e.where(held_mask) * sp_e)
    sp_den = across(sel_e.where(held_mask & sp_e.notna()))

    out = {
        "ic_super_holders": holders,
        "ic_super_conviction_weight": conviction,
        "ic_super_max_conviction": across(w_e.where(held_e.fillna(False)), "max"),  # #15
        "ic_super_top10_holders": across(sel_e * eff("is_top10")),  # #17
        "ic_super_breadth_chg": holders - prev_holders,  # #13
        "ic_super_conviction_chg": conviction - prev_conviction,  # #16
        "ic_super_holders_yoy": holders - holders4,  # #27b
        "ic_super_conviction_weight_yoy": conviction - conviction4,  # #27a
        "ic_super_shares_chg": (ratio - 1.0).where(~guard),  # #24
        "ic_super_quarters_held": across(eff("run_len"), "median"),  # #23
        "ic_super_sp500_share": sp_num / sp_den.where(sp_den > 0),  # #26
        "_super_value_flow": across((sel_e * eff("value_usd")).fillna(0.0) - (prev_sel_e * eff("prev_value_usd")).fillna(0.0)),
    }

    # #27 -- the mean `sel` across this name's HOLDERS. It audits the selector from inside
    # the cube ("how concentrated are the managers backing this name") instead of leaving
    # the selection invisible inside a weight.
    #
    # ⚠ ONLY EMITTED WHEN `sel` ACTUALLY VARIES. Under the flat 1.0 default every holder
    # scores 1 and the column is a constant, which is not a feature -- it is 1.9M cells of
    # nothing that would still consume a cube column, a fingerprint row and a SHAP slot.
    held_num = held_e.fillna(0.0)
    if float(np.nanstd(sel_e.to_numpy())) > 0:
        sel_den = across(held_num)
        out["ic_super_selection_score"] = across(sel_e.fillna(0.0) * held_num) / sel_den.where(sel_den > 0)
    else:
        logger.info(
            "`sel` is flat -> `ic_super_selection_score` would be constant and is " "not emitted; it becomes live under a point-in-time selector."
        )
    # NaN until the name is FIRST HELD, a real number after -- "no elite manager has ever
    # held this" and "they all sold out in 2019" are different facts, and only the second is
    # evidence about the shareholder base.
    #
    # ⚠ THE TEST IS `> 0`, NOT `notna()`, AND THE DIFFERENCE IS A LEAK. `_contributions`
    # gives a manager a `held=False` row in every period they filed, for every name they
    # EVER hold -- so a name first bought in 2024 has a non-null 0 stretching back to 2011,
    # and whether the feature is present at all would depend on the future. Keying the mask
    # on the first date the name is actually held is causal: it is decidable from the past.
    seen = (holders.fillna(0.0) > 0).cummax()
    return {k: v.where(seen) for k, v in out.items()}, grid


def _manager_pool(st: pd.DataFrame, column: str, grid: pd.DatetimeIndex, stale_quarters: int) -> pd.Series:
    """Total `sel` across the managers whose filing is effective on each grid date.

    Manager-level by construction: one row per `(cik, period)`, so the answer cannot depend
    on which of that manager's names happens to sort first."""
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
    """The five class-S event logs, long: `(date, ticker, kind, magnitude)`.

    Stamped on each manager's OWN availability date, and every prior-quarter magnitude uses
    that manager's `sel(m, q-1)`. Reusing `sel(m, q)` on the `q-1` leg is the phantom
    accumulation the module docstring warns about: a manager entering the elite set would
    register as a purchase on every name they already held."""
    rows = []

    def add(kind: str, mask: pd.Series, magnitude: pd.Series) -> None:
        m = mask.fillna(False).to_numpy(dtype=bool)
        if not m.any():
            return
        rows.append(
            pd.DataFrame(
                {
                    "date": contrib["avail"].to_numpy()[m],
                    "ticker": contrib["ticker"].to_numpy()[m],
                    "magnitude": magnitude.to_numpy()[m],
                    "kind": kind,
                }
            )
        )

    held, prev_held = contrib["held"], contrib["prev_held"]
    add(
        "ic_super_initiations",
        held & (prev_held == False),  # noqa: E712
        contrib["sel"] * contrib["w"],
    )
    add(
        "ic_super_full_exits",
        (~held) & (prev_held == True),  # noqa: E712
        contrib["prev_sel"] * contrib["prev_w"],
    )
    add(
        "ic_super_exit_after_top10",
        (~held) & (prev_held == True) & (contrib["prev_is_top10"] == True),  # noqa: E712
        contrib["prev_sel"] * contrib["prev_w"],
    )
    add("ic_super_new_top10", contrib["is_top10"] & (contrib["prev_rank_in_book"] > _TOP_N), contrib["sel"])
    jump = (contrib["prev_rank_in_book"] - contrib["rank_in_book"]).clip(lower=0)
    add("ic_super_rank_jump", held & (jump > 0), contrib["sel"] * jump)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "magnitude", "kind"])
    ev = pd.concat(rows, ignore_index=True).dropna(subset=["date", "magnitude"])
    return ev.groupby(["date", "ticker", "kind"], as_index=False)["magnitude"].sum()


def _to_long(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """An availability-indexed `(date x ticker)` frame -> the `(ticker, as_of, <name>)` shape
    `fundamentals_to_daily` forward-fills onto the trading grid."""
    long = frame.stack(future_stack=True).rename(name).reset_index()
    long.columns = ["as_of", "ticker", name]
    return long.dropna(subset=["as_of"])[["ticker", "as_of", name]]


def build_superinvestor_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame | None,
    roster: dict | list | set | None,
    *,
    shares_out_history: pd.DataFrame | None = None,
    cusip_map: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    selection: pd.Series | Callable | None = None,
    decay_halflife: float = 63.0,
    stale_quarters: int = _STALE_QUARTERS,
    sink=None,
) -> pd.DataFrame:
    """Long-format elite-manager 13F panel -- `f_ic_super_*` (+ `_xs` where the scale drifts).

    `holdings` is the WHOLE-BOOK table from `load_superinvestor_holdings`; `cusip_map` is
    `cusip_ticker_map` and `universe` the analysis tickers, which together resolve the
    S&P500 leg without touching the denominator.
    `selection` is `sel(m, q)` indexed by `(cik, period)` -- None means flat 1.0 (Phase 2.2b
    supplies the point-in-time concentration score). `splits` is `prices_splits`, used to
    restate a prior quarter's share count before any QoQ ratio is taken. `sink` is the
    optional `ConditioningSink` the price-conditioning and cross-source panels read (see
    `_fill_sink`); passing nothing changes nothing.

    Empty when there are no holdings or the roster resolves to no manager.

    ⚠ `frames` RATHER THAN FIVE UNPACKED FIELDS. `peer_dict`, `trading_index`, `stock_close`,
    `level_factor` and `universe` were all read off one `PriceFrames` at the call site. Naming
    the object makes the basis un-mistakable: there is one `close_split` and one `close_total`
    on it, and neither can arrive under the other's parameter name.

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
    universe = frames.universe
    close_split = frames.close_split

    empty = pd.DataFrame(columns=["date", "ticker"])
    need = {"cik", "period", "cusip", "shares", "value_usd"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return empty
    if not _selection_ciks(roster):
        return empty

    holdings = attach_tickers(holdings, cusip_map, universe)
    h = clean_holdings(holdings, key=("cik", "period", "cusip"), numeric=("shares", "value_usd"), common_only=True, pad_ciks=True)
    h, _ = repair_value_basis(h, close_split)
    if h.empty:
        return empty

    state = manager_quarter_state(h)
    conv = manager_stock_conviction(h, state)
    if state.empty or conv.empty:
        return empty

    # `avail` is attached HERE rather than inside `public_state` because a callable
    # `selection` needs it: `manager_selection` ranks each manager against the peers who
    # were public when their filing landed, which is not answerable from `period` alone.
    state["avail"] = _as_of_stamp(state).to_numpy()
    sel = _selection_series(state, selection)
    st = public_state(state, sel)
    contrib = attach_split_factor(_contributions(conv, state, sel), splits)
    if contrib.empty:
        return empty
    levels, _grid = _aggregate(contrib, st, stale_quarters)

    flow = levels.pop("_super_value_flow", None)
    fields = {name: fundamentals_to_daily(_to_long(frame, name), name, trading_index) for name, frame in levels.items()}

    # #25 -- the size-scaled net dollar flow needs a point-in-time daily market cap.
    if flow is not None and shares_out_history is not None and not shares_out_history.empty and stock_close is not None and not stock_close.empty:
        mcap = daily_market_cap(shares_out_history, stock_close, level_factor=level_factor)
        if mcap.empty:
            # ⚠ NOT SILENT -- same trap as `institutional_features`: a `shares_out_history`
            # projected without `sharesOutstanding` (the VENDOR basis, not the PIT one)
            # returns a column-less frame and deletes the feature without a word.
            logger.warning(
                "daily_market_cap returned no columns (shares_out_history has %s; "
                "it needs `sharesOutstanding`, the VENDOR basis) -> "
                "ic_super_flow_to_mcap is skipped.",
                sorted(shares_out_history.columns),
            )
        else:
            daily = fundamentals_to_daily(_to_long(flow, "ic_super_flow_to_mcap"), "ic_super_flow_to_mcap", trading_index)
            f2m = (daily / mcap.where(mcap > 0)).replace([np.inf, -np.inf], np.nan)
            if f2m.notna().any().any():
                fields["ic_super_flow_to_mcap"] = f2m

    # The five class-S features, decayed onto the trading grid. Already stamped on each
    # manager's own availability date -- an event is not tradable before the filing that
    # discloses it.
    events = _events(contrib)
    if not events.empty:
        for kind, sub in events.groupby("kind"):
            frame = decay_events(sub, trading_index, decay_halflife, magnitude_col="magnitude")
            if not frame.empty and frame.notna().any().any():
                fields[str(kind)] = frame

    fields = {k: v for k, v in fields.items() if v is not None and not v.empty}
    _fill_sink(sink, contrib, fields)
    emission = {k: EMISSION[k] for k in fields if k in EMISSION}
    logger.info("elite 13F panel: %s features over %s managers / %s quarters", len(fields), state["cik"].nunique(), state["period"].nunique())
    return build_peer_relative_panel(fields, peer_dict, emission=emission)


def _fill_sink(sink, contrib: pd.DataFrame, fields: dict) -> None:
    """Hand the derived panels this family's event dates, bullish actors and signal frames.

    ⚠ THE TWO EVENT SETS ARE DIFFERENT AND THAT IS THE POINT. `events` is every disclosure by
    a selected manager who holds the name -- the date the conditioning layer measures its
    price path FROM, regardless of direction. `actors` is the bullish subset: a manager whose
    portfolio weight in the name ROSE (an initiation counts, since `prev_w` is absent), which
    is an act rather than a restatement. Counting disclosures as bullish would make
    `ic_xs_bullish_actor_count` a holder count.

    `avail` is already `max(period + 45d, filing_date)` per manager, so nothing here is
    visible before the filing that disclosed it.
    """
    if sink is None or contrib is None or contrib.empty:
        return
    live = contrib["held"].fillna(False).to_numpy(dtype=bool) & (contrib["sel"] > 0).to_numpy()
    disclosures = contrib.loc[live, ["ticker", "avail"]].rename(columns={"avail": "date"})
    sink.add_events("super", disclosures.drop_duplicates())
    added = live & (contrib["w"].fillna(0.0) > contrib["prev_w"].fillna(0.0)).to_numpy()
    sink.add_actors("super", contrib.loc[added, ["ticker", "avail", "cik"]].rename(columns={"avail": "date", "cik": "actor"}))
    sink.keep_signals(fields)
