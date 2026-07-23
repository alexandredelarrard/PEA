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

_COLS = ["date", "sleeve", "instrument", "side", "traded_usd", "position_usd",
         "fee_usd", "spread_usd", "cost_usd"]


def trade_blotter(weights: pd.DataFrame, capital: float, fee_bps: float, spread_bps: float,
                  sleeve: str, floor_usd: float = 0.0) -> pd.DataFrame:
    """Per-(day, instrument) trades from a date x instrument FRACTIONAL weight panel.

    position_usd = weight · capital ; traded_usd = Δposition day-over-day (first day = establish);
    fee/spread = |traded_usd| · bps/1e4. Only rows with |traded_usd| > `floor_usd` are kept."""
    if weights is None or weights.empty:
        return pd.DataFrame(columns=_COLS)
    w = weights.sort_index().fillna(0.0)
    pos = w * float(capital)
    traded = pos.diff()
    traded.iloc[0] = pos.iloc[0]                                   # day 1 establishes the position
    tl = (traded.stack().rename("traded_usd").reset_index())
    tl.columns = ["date", "instrument", "traded_usd"]
    tl = tl[tl["traded_usd"].abs() > float(floor_usd)]
    if tl.empty:
        return pd.DataFrame(columns=_COLS)
    pl = pos.stack().rename("position_usd").reset_index()
    pl.columns = ["date", "instrument", "position_usd"]
    tl = tl.merge(pl, on=["date", "instrument"], how="left")
    tl["sleeve"] = sleeve
    tl["side"] = np.where(tl["traded_usd"] > 0, "BUY", "SELL")
    tl["fee_usd"] = tl["traded_usd"].abs() * float(fee_bps) / 1e4
    tl["spread_usd"] = tl["traded_usd"].abs() * float(spread_bps) / 1e4
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
