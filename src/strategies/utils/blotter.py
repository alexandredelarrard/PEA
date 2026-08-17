"""
blotter.py  (src/strategies/utils/blotter.py)
---------------------------------------------
Per-day, per-instrument TRADE BLOTTER for a strategy sleeve + an Excel aggregator for the
portfolio. Each strategy turns its daily weight panel into a blotter (what was bought/sold each
trading day, and the fee + spread it cost); the portfolio writes one workbook with one sheet
per sleeve (+ a summary), so you can see exactly how much trading is needed to run the book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_COLS = ["date", "sleeve", "instrument", "side", "shares_traded", "price", "traded_usd",
         "shares_held", "position_usd", "fee_usd", "spread_usd", "cost_usd"]


def _melt(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df.copy()
    d.index.name = "date"
    return d.reset_index().melt(id_vars="date", var_name="instrument", value_name=name)


def trade_blotter(weights: pd.DataFrame, capital: float, fee_bps: float, spread_bps: float,
                  sleeve: str, prices: pd.DataFrame | None = None, floor_usd: float = 0.0) -> pd.DataFrame:
    """Per-(day, instrument) trade blotter from a date x instrument FRACTIONAL weight panel.

    position_usd = weight · capital. When a `prices` panel (date x instrument close/level) is given
    the blotter is SHARE-ACCURATE:
        shares_held  = position_usd / price
        shares_traded = shares_held − shares_held_yesterday   (day 1 = establish; new name = full BUY,
                        dropped name = full SELL — each is its own move with its own fee)
        traded_usd   = shares_traded · price                  (the $ actually transacted, price-aware)
    This correctly captures that maintaining a $-target through price moves, or replacing yesterday's
    holding with a different name today, requires selling one and buying the other. Without prices it
    falls back to the $-position delta (shares blank). Fee/spread = |traded_usd|·bps/1e4 per move
    (cash / price-less legs are not charged). Rows with |traded_usd| ≤ floor_usd are dropped."""
    if weights is None or weights.empty:
        return pd.DataFrame(columns=_COLS)
    w = weights.sort_index().fillna(0.0)
    pos = w * float(capital)

    if prices is not None:
        px = prices.reindex(index=w.index, columns=w.columns).astype(float)
        shares_held = (pos / px.replace(0.0, np.nan))                 # NaN where no price (e.g. cash)
        shares_traded = shares_held.diff()
        shares_traded.iloc[0] = shares_held.iloc[0]                   # day 1 establishes
        traded = shares_traded * px                                   # $ value of shares transacted
        pos_delta = pos.diff(); pos_delta.iloc[0] = pos.iloc[0]
        no_px = px.isna() | (px == 0.0)
        traded = traded.where(~no_px, pos_delta)                      # price-less legs: $ delta
    else:
        px = shares_held = shares_traded = None
        traded = pos.diff(); traded.iloc[0] = pos.iloc[0]

    tl = _melt(traded, "traded_usd")
    tl = tl[tl["traded_usd"].abs() > float(floor_usd)].dropna(subset=["traded_usd"])
    if tl.empty:
        return pd.DataFrame(columns=_COLS)
    tl = tl.merge(_melt(pos, "position_usd"), on=["date", "instrument"], how="left")
    if prices is not None:
        tl = tl.merge(_melt(px, "price"), on=["date", "instrument"], how="left")
        tl = tl.merge(_melt(shares_held, "shares_held"), on=["date", "instrument"], how="left")
        tl = tl.merge(_melt(shares_traded, "shares_traded"), on=["date", "instrument"], how="left")
    else:
        tl["price"] = np.nan; tl["shares_held"] = np.nan; tl["shares_traded"] = np.nan
    tl["sleeve"] = sleeve
    tl["side"] = np.where(tl["traded_usd"] > 0, "BUY", "SELL")
    tradeable = tl["price"].notna() if prices is not None else True   # don't charge fees on cash
    tl["fee_usd"] = np.where(tradeable, tl["traded_usd"].abs() * float(fee_bps) / 1e4, 0.0)
    tl["spread_usd"] = np.where(tradeable, tl["traded_usd"].abs() * float(spread_bps) / 1e4, 0.0)
    tl["cost_usd"] = tl["fee_usd"] + tl["spread_usd"]
    return tl[_COLS].sort_values(["date", "instrument"]).reset_index(drop=True)


def _summary(sleeve_trades: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, tr in sleeve_trades.items():
        if tr is None or tr.empty:
            rows.append({"sleeve": name, "n_trades": 0}); continue
        ndays = tr["date"].nunique()
        rows.append({"sleeve": name, "n_trades": len(tr), "trading_days": ndays,
                     "avg_trades_per_day": round(len(tr) / max(ndays, 1), 1),
                     "gross_traded_usd": round(float(tr["traded_usd"].abs().sum()), 2),
                     "total_fee_usd": round(float(tr["fee_usd"].sum()), 2),
                     "total_spread_usd": round(float(tr["spread_usd"].sum()), 2),
                     "total_cost_usd": round(float(tr["cost_usd"].sum()), 2)})
    return pd.DataFrame(rows)


def write_trades_excel(sleeve_trades: dict[str, pd.DataFrame], path) -> None:
    """Write one workbook: a `summary` sheet (per-sleeve trade counts + total fees/spread/cost)
    then one sheet PER SLEEVE with its full daily trade blotter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        _summary(sleeve_trades).to_excel(xl, sheet_name="summary", index=False)
        for name, tr in sleeve_trades.items():
            df = tr if (tr is not None and not tr.empty) else pd.DataFrame(columns=_COLS)
            df.to_excel(xl, sheet_name=str(name)[:31], index=False)
