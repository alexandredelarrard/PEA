"""
positions.py  (src/strategies/utils/positions.py)
-------------------------------------------------
Turn a per-move TRADE BLOTTER into a POSITION LEDGER: match every closing move against
the opening moves it closes (FIFO), so each round trip carries its entry price, its exit
price and its realized P&L net of both legs' fees.

Why FIFO lot matching rather than a running average cost: a sleeve rebalances, so a name
is bought, partly trimmed, topped up and finally exited over many days. Only lot matching
can answer "the shares I bought on 2 June left at what price, and earned what?" -- which
is the question the `strategy` table exists to answer.

Shorts are first-class. A lot is SIGNED, so the same code handles both directions:
    long  lot: opened by a BUY,  closed by a SELL -> pnl = (exit - entry) * shares
    short lot: opened by a SELL, closed by a BUY  -> pnl = (entry - exit) * shares
A move that flips the position (long -> short in one go) first closes every open long lot
and then opens a short lot with what is left over.

TWO P&L views, each independently summable -- read the one that answers your question:
  * `pnl`              realized net P&L of the position OPENED by this row. NULL while the
                       position is still open; filled on the day it is finally closed
                       (partial exits accumulate). `SUM(pnl)` = total realized P&L.
  * `pnl_closed_today` realized net P&L booked BY this move, i.e. on the closing row.
                       `SUM(pnl_closed_today)` = the same total, attributed to the day the
                       money was actually made.
They deliberately hold the same total on different rows; summing BOTH together would
double-count.

Fees are split pro-rata: a row's fee is charged to a round trip in proportion to the share
of that row's quantity the round trip consumed, so a partially-closed lot is only charged
for the part that closed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["LEDGER_COLUMNS", "round_trip_ledger"]

LEDGER_COLUMNS = [
    "trading_day", "sleeve", "ticker", "side", "run_time",
    "shares", "price", "amount_invested", "fee",
    "price_bought", "price_sold", "shares_closed", "closed_on",
    "pnl", "pnl_closed_today", "position_usd", "shares_held",
]

# The trading ledger (`Tables.strategy`): one row per (trading day, sleeve, ticker) move, with the
# FIFO-matched entry/exit price and realized P&L of each round trip.
STRATEGY_SIDE_BUY = "BUY"
STRATEGY_SIDE_SELL = "SELL"

@dataclass
class _Lot:
    """One open lot: `shares` at `price`, and which ledger row opened it."""
    shares: float                     # always positive; direction is on the parent side
    price: float
    row: int                          # index into the ledger rows being built
    long: bool = True


@dataclass
class _Row:
    """A mutable ledger row while matching is in progress."""
    trading_day: pd.Timestamp
    sleeve: str
    ticker: str
    side: str
    shares: float                     # absolute quantity traded by this move
    price: float
    amount_invested: float
    fee: float
    position_usd: float
    shares_held: float
    # filled by matching
    opened_shares: float = 0.0        # of `shares`, how many OPENED a new lot
    closed_shares: float = 0.0        # of `shares`, how many CLOSED an existing lot
    exit_notional: float = 0.0        # share-weighted exit price accumulator (opening side)
    entry_notional: float = 0.0       # share-weighted entry price accumulator (closing side)
    lot_closed_shares: float = 0.0    # how many of THIS row's opened shares were later closed
    closed_on: pd.Timestamp | None = None
    pnl: float = float("nan")         # this position's realized P&L (opening side)
    pnl_today: float = float("nan")   # P&L booked by this move (closing side)
    _pnl_acc: float = 0.0
    _pnl_today_acc: float = 0.0
    _has_pnl: bool = False
    _has_pnl_today: bool = False

    def fee_share(self, qty: float) -> float:
        """The part of this move's fee attributable to `qty` of its shares."""
        return 0.0 if self.shares <= 0 else self.fee * (qty / self.shares)


def _side(shares_signed: float) -> str:
    return STRATEGY_SIDE_BUY if shares_signed > 0 else STRATEGY_SIDE_SELL


def _match_one_name(moves: pd.DataFrame, rows: list[_Row]) -> None:
    """FIFO-match one (sleeve, ticker) in date order, mutating `rows` in place."""
    lots: deque[_Lot] = deque()
    for _, mv in moves.iterrows():
        row_idx = int(mv["_row"])
        row = rows[row_idx]
        qty = float(mv["_signed_shares"])          # + buy / - sell
        price = float(mv["price"])
        remaining = abs(qty)
        opening_long = qty > 0

        # 1. close opposite-direction lots first (FIFO)
        while remaining > 1e-12 and lots and lots[0].long != opening_long:
            lot = lots[0]
            closed = min(remaining, lot.shares)
            entry, exit_ = lot.price, price
            gross = (exit_ - entry) * closed if lot.long else (entry - exit_) * closed

            open_row = rows[lot.row]
            fees = open_row.fee_share(closed) + row.fee_share(closed)
            net = gross - fees

            # opening row learns its exit price + P&L; closing row learns its cost basis
            open_row.exit_notional += exit_ * closed
            open_row.lot_closed_shares += closed
            open_row.closed_on = mv["date"]        # blotter column; renamed to trading_day on output
            open_row._pnl_acc += net
            open_row._has_pnl = True

            row.entry_notional += entry * closed
            row.closed_shares += closed
            row._pnl_today_acc += net
            row._has_pnl_today = True

            lot.shares -= closed
            remaining -= closed
            if lot.shares <= 1e-12:
                lots.popleft()

        # 2. whatever is left OPENS a new lot in this move's direction
        if remaining > 1e-12:
            lots.append(_Lot(shares=remaining, price=price, row=row_idx, long=opening_long))
            row.opened_shares += remaining


def round_trip_ledger(trades: pd.DataFrame, run_time: pd.Timestamp | None = None,
                      instrument_col: str = "instrument") -> pd.DataFrame:
    """Blotter -> position ledger (`LEDGER_COLUMNS`), one row per (trading_day, sleeve, ticker).

    `trades` is a `blotter.trade_blotter` frame: date, sleeve, instrument, side, shares_traded,
    price, traded_usd, shares_held, position_usd, fee_usd, spread_usd, cost_usd. The fee charged
    is `cost_usd` (commission + spread) -- the full cost of the move, which is what a real
    ledger deducts.

    `run_time` stamps every row with when the pipeline produced it (defaults to now, UTC-naive
    to match the rest of the DB). Rows whose price is unknown (a cash leg has no price) are
    dropped: they are an accounting residual, not a trade you place with a broker."""
    empty = pd.DataFrame(columns=LEDGER_COLUMNS)
    if trades is None or trades.empty:
        return empty

    stamp = pd.Timestamp(run_time) if run_time is not None else pd.Timestamp.now().floor("s")
    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={instrument_col: "ticker"})
    # a price is required: shares, entry/exit prices and P&L are all undefined without one
    df = df[df["price"].notna() & (df["price"] > 0)]
    # shares_traded is authoritative when present; otherwise derive it from the $ traded
    shares = df.get("shares_traded")
    if shares is None:
        shares = df["traded_usd"] / df["price"]
    shares = pd.to_numeric(shares, errors="coerce")
    df["_signed_shares"] = np.where(shares.notna(), shares, df["traded_usd"] / df["price"])
    df = df[df["_signed_shares"].notna() & (df["_signed_shares"].abs() > 1e-12)]
    if df.empty:
        return empty

    # fee BEFORE the sort, so it travels with its row (charging the full cost of the move:
    # commission + spread, which is what a real ledger deducts)
    df["_fee"] = pd.to_numeric(
        df["cost_usd"] if "cost_usd" in df.columns else df.get("fee_usd", 0.0),
        errors="coerce").fillna(0.0)
    df = df.sort_values(["sleeve", "ticker", "date"]).reset_index(drop=True)
    df["_row"] = np.arange(len(df))

    rows: list[_Row] = [
        _Row(trading_day=r["date"], sleeve=str(r["sleeve"]), ticker=str(r["ticker"]),
             side=_side(r["_signed_shares"]), shares=abs(float(r["_signed_shares"])),
             price=float(r["price"]),
             amount_invested=abs(float(r["_signed_shares"])) * float(r["price"]),
             fee=float(r["_fee"]),
             position_usd=float(r.get("position_usd", np.nan)),
             shares_held=float(r.get("shares_held", np.nan)))
        for _, r in df.iterrows()
    ]

    for _, moves in df.groupby(["sleeve", "ticker"], sort=False):
        _match_one_name(moves, rows)

    out = []
    for row in rows:
        # A row's OWN price is one leg of every round trip it takes part in; the other leg is
        # the FIFO counterparty price -- which is a SELL price for a BUY row and vice versa,
        # whether this row opened the position (counterparty = the later exit) or closed one
        # (counterparty = the earlier entry). Summing both accumulators is therefore correct
        # even for a FLIP (a sell that closes longs and opens a short in one move): both of
        # its counterparty legs are buy prices.
        other_shares = row.closed_shares + row.lot_closed_shares
        other = (row.entry_notional + row.exit_notional) / other_shares if other_shares > 1e-12 else np.nan
        buying = row.side == STRATEGY_SIDE_BUY
        out.append({
            "trading_day": row.trading_day, "sleeve": row.sleeve, "ticker": row.ticker,
            "side": row.side, "run_time": stamp,
            "shares": row.shares, "price": row.price,
            "amount_invested": row.amount_invested, "fee": row.fee,
            "price_bought": row.price if buying else other,
            "price_sold": other if buying else row.price,
            "shares_closed": other_shares if other_shares > 1e-12 else np.nan,
            "closed_on": row.closed_on,
            "pnl": row._pnl_acc if row._has_pnl else np.nan,
            "pnl_closed_today": row._pnl_today_acc if row._has_pnl_today else np.nan,
            "position_usd": row.position_usd, "shares_held": row.shares_held,
        })

    led = pd.DataFrame(out, columns=LEDGER_COLUMNS)
    return led.sort_values(["trading_day", "sleeve", "ticker"]).reset_index(drop=True)
