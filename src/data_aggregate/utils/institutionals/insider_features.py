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
stamped on `filing_date`, never `transaction_date`, and every window is trailing. The
consensus price in `insider_quality` is the one place a forward-looking window is used, and
it is a data-quality reference that never reaches the panel.

AVAILABILITY (D16, measured 2026-09-10):

    every feature      NaN before 2006-01-03   -- `insider_transactions` has no earlier row
    the two 10b5-1     NaN before 2023-07-01   -- `is_10b5_1` is 0.00% filled on `S` rows in
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
    #38 days_since_last_buy         D  --       trading days since the last P filing
    #39 net_buy_ratio_180d          D  180d     (P$ - S$) / (P$ + S$), in [-1, 1]
    #40 discretionary_sell_mcap_60d D  60d      S value, not on a plan and not an
                                                exercise-and-sell, / market cap
    #41 planned_sell_mcap_60d       D  60d      S value on a 10b5-1 plan / market cap

#36 and #37 are RATIOS, so their class-S treatment is a decay-WEIGHTED MEAN -- numerator and
denominator are decayed with the same half-life and divided -- not a decayed sum. A decayed
sum of percentiles would grow with the number of purchases and stop being a percentile.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.institutionals.decay import decay_events
from src.data_aggregate.utils.institutionals.insider_quality import (
    FLAG_PCT_SHARES_OUTSTANDING, asof_values, clean_transactions, report_oversized)

_log = logging.getLogger(__name__)

#: Emission class per feature (D27), MEASURED on the live panel rather than assigned from the
#: shape of the formula. The discriminator is not boundedness: it is whether a within-date
#: percentile carries information, and the test that settles that is the TIE FRACTION. An
#: `_xs` leg over a cross-section that is 90%+ tied is a plateau -- almost every name shares
#: one rank, so the column moves on which names happen to be present, not on the signal.
#:
#:     feature                       nonnull   uniq/date   ties/date   rho(raw,xs)   emit
#:     buy_value_mcap_60d              56.3%         445        0.0%        0.9397   raw+xs
#:     purchase_pct_prior              52.8%         417        0.4%        0.9943   raw+xs
#:     director_buy_mcap_180d          54.4%         435        0.0%        0.9379   raw+xs
#:     ceo_buy_mcap_180d               24.2%         184        0.0%        0.8756   raw+xs
#:     cfo_buy_mcap_180d               18.9%         152        0.0%        0.7724   raw+xs
#:     planned_sell_mcap_60d            6.3%         149       50.8%        0.9021   raw+xs
#:     discretionary_sell_mcap_60d      9.7%         178       62.0%        0.8476   raw+xs
#:     buy_shares_so_180d              60.4%         124       72.5%        0.7886   raw+xs
#:     buy_value_mcap_180d             60.2%         124       72.5%        0.7883   raw+xs
#:     owner_surprise_120d             48.3%         350        9.8%             -   raw
#:     days_since_last_buy             56.3%         338       25.3%             -   raw
#:     net_buy_ratio_180d              56.2%         107       74.8%             -   raw
#:     cluster_buy_120d                56.3%           7       98.3%        0.4583   raw
#:     distinct_buyers_120d            56.3%           8       98.1%        0.7237   raw
#:
#: ⚠ THE TEST REMOVES LEGS, IT DOES NOT ADD THEM. `owner_surprise_120d` (9.8% ties) and
#: `days_since_last_buy` (25.3%) sit far below the threshold and are still `raw`, because a
#: low tie fraction is not a reason to rank something that is already comparable across
#: dates: a percentile of a percentile is a re-rank, and a day count means the same thing in
#: 2009 as in 2025. `net_buy_ratio_180d` is bounded [-1, 1] AND 74.8% tied -- its median is
#: exactly -1, so more than half the cross-section is one plateau -- which is two independent
#: reasons for the same answer.
#:
#: ⚠ `days_since_last_buy` IS THE ONE WORTH RE-EXAMINING IN 2.8. Early in the sample every
#: ticker's count is small because the history is short, so part of the raw level is calendar
#: rather than signal -- the one case here where `_xs` would remove a real drift. Left as the
#: registry declares it rather than changed on an argument that has not been measured.
EMISSION: dict[str, str] = {
    "ic_insider_buy_value_mcap_60d":          "raw+xs",
    "ic_insider_buy_value_mcap_180d":         "raw+xs",
    "ic_insider_buy_shares_so_180d":          "raw+xs",
    "ic_insider_distinct_buyers_120d":        "raw",     # 98.1% ties: a 0-8 integer count
    "ic_insider_cluster_buy_120d":            "raw",     # 98.3% ties: the same, gated
    "ic_insider_ceo_buy_mcap_180d":           "raw+xs",
    "ic_insider_cfo_buy_mcap_180d":           "raw+xs",
    "ic_insider_director_buy_mcap_180d":      "raw+xs",
    "ic_insider_purchase_pct_prior":          "raw+xs",
    "ic_insider_owner_surprise_120d":         "raw",     # already a percentile in [0, 1]
    "ic_insider_days_since_last_buy":         "raw",     # a day count, comparable as-is
    "ic_insider_net_buy_ratio_180d":          "raw",     # bounded [-1, 1] by construction
    "ic_insider_discretionary_sell_mcap_60d": "raw+xs",
    "ic_insider_planned_sell_mcap_60d":       "raw+xs",
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
TEN_B5_1_FLOOR = pd.Timestamp("2023-07-01")

#: Half-life in TRADING days for the class-S decay, overridden from
#: `build_cube.institutionals.decay_halflife.insider`.
DEFAULT_DECAY_HALFLIFE = 63.0


def build_insider_feature_panel(
    insider: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    shares_out_history: pd.DataFrame | None = None,
    stock_close: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
    decay_halflife: float = DEFAULT_DECAY_HALFLIFE,
) -> pd.DataFrame:
    """Long-format insider feature panel (`f_<name>` and `f_<name>_xs`, per `EMISSION`).

    Empty when there are no usable transactions. `shares_out_history` + `stock_close` are
    what make the **8 size-scaled** features possible; without them only the **6 scale-free**
    ones (`distinct_buyers`, `cluster_buy`, `days_since_last_buy`, `net_buy_ratio`,
    `purchase_pct_prior`, `owner_surprise`) are emitted, because a dollar flow that is not
    divided by the company's size is a market-cap proxy.
    """
    t, diag = clean_transactions(insider)
    if t.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if idx.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    mcap = _market_cap(shares_out_history, stock_close, level_factor)
    shares_out = (fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", idx)
                  if shares_out_history is not None and not shares_out_history.empty
                  else pd.DataFrame(index=idx))
    _report_oversized(t, shares_out)

    buys = t[t["code"].eq("P")].copy()
    sells = t[t["code"].eq("S")].copy()
    buys["mcap"] = asof_values(mcap, buys["ticker"], buys["day"])
    buys["value_mcap"] = buys["value"] / buys["mcap"].where(buys["mcap"] > 0)

    fields: dict[str, pd.DataFrame] = {}
    fields.update(_dense_fields(buys, sells, idx, mcap, shares_out))
    fields.update(_breadth_fields(buys, idx))
    fields.update(_sparse_fields(buys, idx, decay_halflife))

    floor = pd.Series(idx >= INSIDER_FLOOR, index=idx)
    for name, frame in list(fields.items()):
        if frame is None or frame.empty:
            fields.pop(name)
            continue
        fields[name] = frame.where(floor, axis=0)

    _log.info("insider panel: %s features from %s scoped transactions (%s buys, %s sells)",
              len(fields), len(t), len(buys), len(sells))
    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)


# --------------------------------------------------------------------------- #
# dense features -- trailing calendar windows                                   #
# --------------------------------------------------------------------------- #

def _dense_fields(buys: pd.DataFrame, sells: pd.DataFrame, idx: pd.DatetimeIndex,
                  mcap: pd.DataFrame | None,
                  shares_out: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The six class-D features. Each is a trailing sum over a calendar window, sampled onto
    the trading grid, so a day only ever sees transactions already filed by it."""
    out: dict[str, pd.DataFrame] = {}
    seen = _first_filing(buys, sells, idx)
    buy_val_180 = _rolling(buys, idx, WINDOW_180, "value", seen)
    sell_val_180 = _rolling(sells, idx, WINDOW_180, "value", seen)

    if mcap is not None and not mcap.empty:
        out["ic_insider_buy_value_mcap_180d"] = _over(buy_val_180, mcap)
    if not shares_out.empty:
        out["ic_insider_buy_shares_so_180d"] = _over(
            _rolling(buys, idx, WINDOW_180, "shares_n", seen), shares_out)

    bv, sv = buy_val_180.fillna(0.0), sell_val_180.fillna(0.0)
    denom = bv + sv
    # NaN, not 0, where nothing was filed in the window: "no insider traded" is not
    # "insiders were evenly split". The bounded [-1, 1] range is what keeps this one `raw`.
    out["ic_insider_net_buy_ratio_180d"] = ((bv - sv) / denom.where(denom > 0)
                                            ).replace([np.inf, -np.inf], np.nan)
    out["ic_insider_days_since_last_buy"] = _days_since(buys, idx)

    if mcap is not None and not mcap.empty:
        planned = sells["is_10b5_1"]
        # Discretionary requires BOTH: not on a plan, and not the sell leg of an
        # exercise-and-sell package (35.0% of all `S` rows). A NaN plan flag is not a 0 --
        # before 2023-07-01 the field is empty, and the floor below removes that region
        # rather than letting "unknown" masquerade as "discretionary".
        disc = sells[planned.eq(0) & ~sells["in_exercise_package"].astype(bool)]
        for name, sub in (("ic_insider_discretionary_sell_mcap_60d", disc),
                          ("ic_insider_planned_sell_mcap_60d", sells[planned.eq(1)])):
            frame = _over(_rolling(sub, idx, WINDOW_60, "value", seen), mcap)
            out[name] = frame.where(pd.Series(idx >= TEN_B5_1_FLOOR, index=idx), axis=0)
    return out


def _first_filing(buys: pd.DataFrame, sells: pd.DataFrame,
                  idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean (date x ticker): has this ticker filed ANY open-market Form 4 by this date?

    It is the coverage mask every windowed feature shares, and it exists because the two
    answers a zero can mean are different facts. `BEAR` sold and never bought: its buying
    over the window is **0**, a real observation. `BEAR` before its first-ever Form 4: its
    buying is **unknown**. Keying the mask on the union of purchases and sales rather than on
    the feature's own leg is what separates them -- without it a name that only ever sold
    reads NaN on every buy feature, which understates coverage and silently drops the name
    from the cross-section on the very dates it is most informative.
    """
    both = pd.concat([buys[["day", "ticker"]], sells[["day", "ticker"]]])
    if both.empty:
        return pd.DataFrame(False, index=idx, columns=pd.Index([], name="ticker"))
    first = both.groupby("ticker")["day"].min()
    grid = pd.DataFrame({t: idx >= d for t, d in first.items()}, index=idx)
    grid.columns.name = "ticker"
    return grid


def _rolling(txns: pd.DataFrame, idx: pd.DatetimeIndex, window_days: int,
             value_col: str | None, seen: pd.DataFrame) -> pd.DataFrame:
    """Trailing `window_days`-calendar-day sum per ticker, sampled onto `idx`.

    Zero-filled between filing days on a DAILY calendar first, so the rolling total counts
    the window's real length rather than the number of filing days inside it, then widened to
    every ticker in `seen` and masked before each one's first filing.
    """
    if txns.empty:
        return pd.DataFrame(np.nan, index=idx, columns=seen.columns)
    if value_col:
        piv = txns.groupby(["day", "ticker"])[value_col].sum().unstack("ticker")
    else:
        piv = txns.groupby(["day", "ticker"]).size().unstack("ticker")
    calendar = pd.date_range(min(piv.index.min(), idx.min()), max(piv.index.max(), idx.max()),
                             freq="D")
    piv = piv.reindex(calendar).fillna(0.0).rolling(f"{window_days}D").sum()
    return piv.reindex(index=idx, columns=seen.columns).fillna(0.0).where(seen)


def _over(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """`numerator / denominator` on the intersection, NaN where the denominator is absent or
    non-positive. Never zero-fills the denominator: a missing share count is unknown size,
    not infinite size."""
    cols = numerator.columns.intersection(denominator.columns)
    if cols.empty:
        return pd.DataFrame(index=numerator.index)
    den = denominator.reindex(index=numerator.index)[cols]
    return (numerator[cols] / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)


def _days_since(buys: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Trading days since the most recent purchase filing, NaN before a ticker's first.

    Counted in ROWS OF THE GRID, so a long weekend is one day, matching every other
    trading-day clock in the cube (`decay_events`, the momentum windows).
    """
    if buys.empty:
        return pd.DataFrame(index=idx)
    days = buys.groupby(["day", "ticker"]).size().unstack("ticker")
    # `side="left"`: a Form 4 filed on a non-trading day is first actionable on the NEXT
    # session, so it must not be credited to the previous one.
    pos = idx.searchsorted(days.index.to_numpy(), side="left")
    keep = pos < len(idx)
    grid = np.zeros((len(idx), days.shape[1]), dtype=bool)
    if keep.any():
        rows = pos[keep]
        vals = days.to_numpy()[keep] > 0
        np.logical_or.at(grid, rows, vals)
    ordinal = np.arange(len(idx))[:, None]
    last = np.maximum.accumulate(np.where(grid, ordinal, -1), axis=0)
    out = np.where(last >= 0, ordinal - last, np.nan).astype("float64")
    return pd.DataFrame(out, index=idx, columns=days.columns)


# --------------------------------------------------------------------------- #
# sparse features -- decayed event intensities                                  #
# --------------------------------------------------------------------------- #

def _sparse_fields(buys: pd.DataFrame, idx: pd.DatetimeIndex,
                   halflife: float) -> dict[str, pd.DataFrame]:
    """The six DECAYED features: four value-scaled intensities and two decay-weighted means.

    #31 and #32 are class S too but live in `_breadth_fields` -- see there for why a distinct
    count is the one class-S quantity that decay destroys.
    """
    out: dict[str, pd.DataFrame] = {}
    if buys.empty:
        return out
    ev = buys.rename(columns={"day": "date"})
    # ⚠ ALL FOUR value-scaled intensities are gated on a real market cap, and the guard is
    # not defensive padding. `decay_events` treats a NaN magnitude as an event of unknown
    # size and counts it 1.0 -- correct policy there, but with no market cap EVERY magnitude
    # is NaN and these four would silently become event counts under names that promise
    # dollars over market cap.
    priced = "value_mcap" in ev.columns and ev["value_mcap"].notna().any()
    if priced:
        out["ic_insider_buy_value_mcap_60d"] = decay_events(
            ev, idx, halflife, magnitude_col="value_mcap")

    roles = {"ic_insider_ceo_buy_mcap_180d": ev["role"].eq("CEO"),
             "ic_insider_cfo_buy_mcap_180d": ev["role"].eq("CFO"),
             "ic_insider_director_buy_mcap_180d": ev["is_director"].eq(1)}
    for name, mask in roles.items():
        sub = ev[mask]
        if priced and not sub.empty and sub["value_mcap"].notna().any():
            out[name] = decay_events(sub, idx, halflife, magnitude_col="value_mcap")

    pct_prior = _purchase_pct_prior(ev)
    if pct_prior is not None:
        out["ic_insider_purchase_pct_prior"] = _decay_weighted_mean(
            pct_prior, idx, halflife, "pct_prior", "value")
    surprise = _owner_surprise(ev)
    if surprise is not None:
        out["ic_insider_owner_surprise_120d"] = _decay_weighted_mean(
            surprise, idx, halflife, "surprise", "value")
    return out


def _decay_weighted_mean(ev: pd.DataFrame, idx: pd.DatetimeIndex, halflife: float,
                         value_col: str, weight_col: str) -> pd.DataFrame:
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
    num = decay_events(w, idx, halflife, magnitude_col="_num")
    den = decay_events(w, idx, halflife, magnitude_col="_den")
    return ((num / den.where(den != 0)) - offset).replace([np.inf, -np.inf], np.nan)


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
    per_owner = buys.drop_duplicates(["day", "ticker", "owner_cik"])
    frames = {tkr: _rolling_distinct(g, idx) for tkr, g in per_owner.groupby("ticker")}
    grid = pd.DataFrame({t: s for t, s in frames.items() if s is not None})
    if grid.empty:
        return {}
    return {"ic_insider_distinct_buyers_120d": grid,
            # Gated, not masked: below the threshold the answer is "no cluster" = 0, which is
            # a fact, while before the first purchase it is unknown = NaN, inherited above.
            "ic_insider_cluster_buy_120d": grid.where(grid >= CLUSTER_MIN, 0.0
                                                      ).where(grid.notna())}


def _rolling_distinct(g: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Distinct owners buying in the trailing `CLUSTER_WINDOW_DAYS`, stepped onto `idx`.

    Exact rather than approximate. The count changes on exactly two kinds of day -- a new
    purchase is filed, or an old one ages out -- so it is evaluated at the union of filing
    days and filing days + the window, and forward-filled between. Sweeping both edges with
    a single owner counter keeps it O(n) per ticker instead of re-counting a window per day.
    """
    day = g["day"].to_numpy("datetime64[ns]")
    owner = g["owner_cik"].astype(str).to_numpy()
    order = np.argsort(day, kind="stable")
    day, owner = day[order], owner[order]
    width = np.timedelta64(CLUSTER_WINDOW_DAYS, "D")
    points = np.unique(np.concatenate([day, day + width]))
    counts = np.empty(len(points), dtype="float64")

    live: dict[str, int] = {}
    distinct = i = j = 0
    for k, cp in enumerate(points):
        while i < len(day) and day[i] <= cp:
            live[owner[i]] = live.get(owner[i], 0) + 1
            distinct += live[owner[i]] == 1
            i += 1
        while j < len(day) and day[j] <= cp - width:
            live[owner[j]] -= 1
            distinct -= live[owner[j]] == 0
            j += 1
        counts[k] = distinct

    s = pd.Series(counts, index=pd.DatetimeIndex(points)).reindex(
        pd.DatetimeIndex(points).union(idx)).ffill().reindex(idx)
    # NaN before the ticker's first purchase: "nobody has ever bought this name" and "the
    # window has emptied" are different facts, and only the second is a zero.
    return s.where(idx >= pd.Timestamp(day[0]))


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
    """
    if "owner_cik" not in ev.columns:
        return None
    d = ev.dropna(subset=["value"]).sort_values(["owner_cik", "date"], kind="stable")
    if d.empty:
        return None
    g = d.groupby("owner_cik", sort=False)["value"]
    # `rank(pct=True)` over the expanding window includes the current row, so the first
    # observation is always 1.0 and every later one is inflated by 1/n. Subtracting the
    # self-contribution rescales to "fraction of PRIOR purchases at or below this one".
    n = g.cumcount()
    expanding_rank = g.expanding().apply(lambda s: (s.iloc[:-1] <= s.iloc[-1]).sum(),
                                         raw=False).reset_index(level=0, drop=True)
    d["surprise"] = (expanding_rank / n.where(n > 0)).astype("float64")
    return d.dropna(subset=["surprise"])


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #

def _market_cap(shares_out_history: pd.DataFrame | None, stock_close: pd.DataFrame | None,
                level_factor: pd.DataFrame | None) -> pd.DataFrame | None:
    if (shares_out_history is None or shares_out_history.empty
            or stock_close is None or stock_close.empty):
        _log.warning("No shares outstanding or close -> the 11 size-scaled insider features "
                     "are skipped.")
        return None
    mcap = daily_market_cap(shares_out_history, stock_close, level_factor=level_factor)
    return mcap if not mcap.empty else None


def _report_oversized(t: pd.DataFrame, shares_out: pd.DataFrame) -> None:
    """Log, never drop. See `insider_quality.report_oversized` for why a threshold cannot
    separate a fabricated filing from a genuine block disposal on this universe."""
    flagged = report_oversized(t, shares_out if not shares_out.empty else None)
    if flagged.empty:
        return
    top = flagged.head(5)
    _log.warning("insider: %s transaction(s) above %.0f%% of shares outstanding, kept and "
                 "reported: %s", len(flagged), 100 * FLAG_PCT_SHARES_OUTSTANDING,
                 ", ".join(f"{r.ticker} {r.day:%Y-%m-%d} {r.code} "
                           f"{r.pct_shares_outstanding:.1%}" for r in top.itertuples()))
