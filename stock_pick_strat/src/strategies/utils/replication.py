"""
replication.py  (src/strategies/utils/replication.py)
------------------------------------------------------
Mirror the elite 13F cohort's equity book with a FIXED pot of capital, long-only and
unlevered. Pure (DataFrame -> dict of frames); all IO stays in the step.

THE MIRROR RULE. On each day the disclosed book changes, hold the cohort's WEIGHTS:

    my_target_value_i = my_equity * (their_value_i / their_total_value)

This gives the intended proportional behaviour: if KO is 5% of my book and the cohort sells
10% of its KO stake, their KO weight falls ~10% and mine follows it down 0.5pp of portfolio.

It is deliberately a TARGET, not an increment. Trading `f * delta_shares` with
`f = my_equity / their_book_value` looks equivalent and is not: `f` moves over time, so the
buys and the sells of a position are scaled by different factors and a name the cohort fully
EXITS does not net back to zero. The leftover sliver then compounds forever — which is exactly
how a book mirroring Li Lu finished 94% in a Micron position he had already sold, off a 13x
run in a name he did not own. A weight target self-corrects: weight 0 means sell all of it.

Between filings the cohort's share counts are fixed, so their weights and mine drift
identically with prices — the target is already met and nothing trades. Turnover therefore
still comes only from their disclosed trades, not from daily rebalancing.

WHY IT CANNOT LEVER. The cohort's book is roughly self-financing (they sell to buy), so the
mirrored trades are too — but only roughly, and never exactly, because my execution prices and
my capital base differ from theirs. So the day is settled in the only order that can't overdraw:
sells execute first and credit cash, then buys are capped at the cash actually on hand and
scaled down pro-rata if the cohort bought more than I can fund. Positions therefore floor at
zero (13F reports long holdings only, so the mirror never goes short) and invested value never
exceeds equity. `max_leverage` in the returned diagnostics is the assertion of that.

POINT-IN-TIME. Flows are stamped on the filing date, and filings routinely land after the close,
so trading that same close would be look-ahead. Every flow is executed `execution_lag` trading
days later (default 1) at that day's close.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SEED_MIN_NAMES = 50    # the cohort holds 2 names on its first filing day and 271 by 2014-02-28;
                        # seeding on a 2-name book is not a replication of anything


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot_table(index="as_of", columns="ticker", values=col, aggfunc="last")


def _close_panel(prices: pd.DataFrame, index: pd.DatetimeIndex,
                 tickers: list[str]) -> pd.DataFrame:
    """Long [date, ticker, close] -> a wide close matrix on the panel's own calendar,
    forward-filled (a market holiday must not read as a missing price and silently drop a
    position from the book)."""
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p[p["ticker"].isin(tickers)]
    wide = p.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    wide = wide.reindex(wide.index.union(index)).ffill().reindex(index)
    # a 13F name with NO price history at all (e.g. GEHC) has no column in the pivot -- give it
    # an all-NaN one so the caller's "is this priceable" filter sees it instead of KeyError-ing
    return wide.reindex(columns=tickers)


def _seed_weights(init_cum: pd.Series, held: pd.Series, px: pd.Series) -> pd.Series:
    """Opening weights ∝ (cumulative init shares) x price, restricted to names the cohort
    STILL holds at the seed date. The cumulative init up to the seed date is the cohort's
    starting book; masking on `held` drops names that were initialized early and fully exited
    before we start, which would otherwise seed a position the cohort no longer owns."""
    basis = (init_cum.where(held > 0, 0.0) * px).fillna(0.0)
    basis = basis[basis > 0]
    return basis / basis.sum()


def replicate_superinvestors(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    capital: float,
    fee_bps: float,
    spread_bps: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    execution_lag: int = 1,
    seed_min_names: int = _SEED_MIN_NAMES,
) -> dict:
    """Run the mirror. Returns {returns, equity, weights, cash_weight, trades, diagnostics}."""
    cost_rate = (float(fee_bps) + float(spread_bps)) / 1e4

    shares_w = _wide(panel, "superinvestor_shares").sort_index()
    net_w = _wide(panel, "superinvestor_net_shares").sort_index().fillna(0.0)
    init_w = _wide(panel, "superinvestor_init_shares").sort_index().fillna(0.0)
    if start is not None:
        keep = shares_w.index >= pd.Timestamp(start)
        shares_w, net_w, init_w = shares_w[keep], net_w[keep], init_w[keep]
    if end is not None:
        keep = shares_w.index <= pd.Timestamp(end)
        shares_w, net_w, init_w = shares_w[keep], net_w[keep], init_w[keep]

    close = _close_panel(prices, shares_w.index, list(shares_w.columns))
    tradable = [t for t in shares_w.columns if close[t].notna().any()]
    shares_w, net_w, init_w, close = (shares_w[tradable], net_w[tradable],
                                      init_w[tradable], close[tradable])
    held_w = shares_w.fillna(0.0)

    # a holding disclosed on day t is only actionable at the close `execution_lag` days later
    held_lag = held_w.shift(execution_lag).fillna(0.0)
    # trade ONLY on days the disclosed book actually moved. Between filings the cohort's shares
    # are constant, so their weights and mine drift identically with prices and the target is
    # already met -- rebalancing daily would just churn fees on floating-point noise.
    moved = held_lag.diff().abs().sum(axis=1) > 0

    # Seed on the first day the book is diversified enough to replicate. The threshold guards
    # against seeding on the POOLED cohort's ramp-up (it holds 2 names on its first filing day
    # and 271 six months later), so it is a preference, not a gate: a genuinely concentrated
    # book -- one manager running 12 names, which is a strategy and not an artifact -- relaxes
    # to its first priced holding instead of failing. Raising here would make the whole sleeve
    # crash on any roster narrow enough to never reach the default.
    # measured on the LAGGED book, like every later trade: seeding off same-day holdings would
    # both look ahead by a day and start the book on a different basis than it is checked against
    n_names = (held_lag.gt(0) & close.notna()).sum(axis=1)
    eligible = n_names[n_names >= seed_min_names]
    if eligible.empty:
        eligible = n_names[n_names > 0]
        if eligible.empty:
            raise ValueError("no priced holdings on any date -> nothing to seed")
        logger.info("replication: book never holds %d priced names (max %d) -> seeding on its "
                    "first priced holding instead", seed_min_names, int(n_names.max()))
    t0 = eligible.index[0]
    dates = shares_w.index[shares_w.index >= t0]

    init_cum = init_w.loc[:t0].sum(axis=0)
    w0 = _seed_weights(init_cum, held_lag.loc[t0], close.loc[t0])
    shares = pd.Series(0.0, index=tradable)
    px0 = close.loc[t0]
    # establishing the book is itself a trade: buy only what the capital covers INCLUDING its
    # cost, or the very first day overdraws by exactly the fee and reads as 1.001x leverage
    seed_notional = capital / (1.0 + cost_rate)
    shares[w0.index] = (w0 * seed_notional / px0[w0.index]).fillna(0.0)
    spent = float((shares * px0.fillna(0.0)).sum())
    cash = float(capital - spent * (1.0 + cost_rate))

    equity, invested, cash_hist, rows, book = [], [], [], [], []
    for i, t in enumerate(dates):
        px = close.loc[t].fillna(0.0)
        if i > 0 and moved.loc[t]:                      # day 0 is the seed, already traded
            # TARGET the cohort's weights rather than accumulating their share increments.
            # Accumulating `f x delta_shares` looks equivalent but is not: `f` moves over time,
            # so a name the cohort fully EXITS leaves a residual (the buys were scaled at a
            # different `f` than the sells) which then compounds forever -- that is how a book
            # mirroring Li Lu ended up 94% in a Micron position he no longer held. A weight
            # target is self-correcting: weight 0 means sell all of it, always.
            tv = held_lag.loc[t] * px
            total = float(tv.sum())
            eq_pre = float((shares * px).sum()) + cash
            if total > 0 and eq_pre > 0:
                target = (eq_pre * (tv / total) / px.where(px > 0)).fillna(0.0)
                delta = target - shares
                # SELLS first: they fund the day. Never sell more than is actually held.
                sell = (-delta.clip(upper=0.0)).clip(upper=shares).where(px > 0, 0.0)
                proceeds = float((sell * px).sum())
                if proceeds > 0:
                    shares -= sell
                    cash += proceeds * (1.0 - cost_rate)
                # BUYS second, capped by the cash on hand -> leverage is impossible
                buy = delta.clip(lower=0.0).where(px > 0, 0.0)
                gross = float((buy * px).sum())
                capped = False
                if gross > 0:
                    affordable = max(cash, 0.0) / (1.0 + cost_rate)
                    if gross > affordable:              # cohort bought more than I can fund
                        buy *= (affordable / gross)
                        gross, capped = affordable, True
                    shares += buy
                    cash -= gross * (1.0 + cost_rate)
                # `capped` is recorded even when the cap scaled the buy to exactly zero -- a
                # fully-blocked purchase is the constraint biting hardest, not a non-event
                if proceeds > 0 or gross > 0 or capped:
                    rows.append({"date": t, "sold_usd": proceeds, "bought_usd": gross,
                                 "cost_usd": (proceeds + gross) * cost_rate,
                                 "buy_capped": capped})
        inv = float((shares * px).sum())
        equity.append(inv + cash)
        invested.append(inv)
        cash_hist.append(cash)
        book.append((shares * px).to_numpy(copy=True))          # $ position, priced at t

    eq = pd.Series(equity, index=dates, dtype=float)
    inv_s = pd.Series(invested, index=dates, dtype=float)
    cash_s = pd.Series(cash_hist, index=dates, dtype=float)
    # day 0 is measured against the capital handed in, NOT against itself: establishing the
    # book costs a fee, and a `pct_change` first value of 0 would hide that entry cost from the
    # return series (and let `(1+r).cumprod()*capital` drift above the real equity path)
    ret = eq.div(eq.shift(1).fillna(float(capital))) - 1.0
    book_val = pd.DataFrame(book, index=dates, columns=tradable)
    weights = book_val.div(eq, axis=0)

    # ORPHAN CHECK: value parked in a name the cohort does not hold. Every other invariant here
    # (no leverage, no shorts, cash >= 0) passed while a book mirroring Li Lu sat 94% in a
    # Micron position he had already sold -- none of them can see a position that simply should
    # not exist. Measured on VALUE, not shares: a delisted name with no price contributes
    # nothing to equity and cannot be sold, so flagging it would be a false positive.
    orphan = book_val.where(held_lag.loc[dates].fillna(0.0) <= 0, 0.0).abs().sum(axis=1)
    orphan_weight = (orphan / eq.where(eq > 0)).fillna(0.0)

    trades = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "sold_usd", "bought_usd", "cost_usd", "buy_capped"])
    diagnostics = {
        "seed_date": t0,
        "seed_names": int((w0 > 0).sum()),
        "max_leverage": float((inv_s / eq).max()),
        "min_cash": float(cash_s.min()),
        "final_equity": float(eq.iloc[-1]),
        "total_cost_usd": float(trades["cost_usd"].sum()) if not trades.empty else 0.0,
        "n_trade_days": int(len(trades)),
        # days the cohort bought more than this book could fund -- the no-leverage rule biting
        "n_days_buy_capped": int(trades["buy_capped"].sum()) if not trades.empty else 0,
        # share of equity sitting in names the cohort does not hold -- must be ~0
        "max_orphan_weight": float(orphan_weight.max()),
        "orphan_date": (orphan_weight.idxmax() if float(orphan_weight.max()) > 0 else None),
    }
    return {"returns": ret, "equity": eq, "invested": inv_s, "cash": cash_s,
            "cash_weight": (cash_s / eq), "trades": trades, "diagnostics": diagnostics,
            "weights": weights}
