"""FIFO round-trip matching for the `strategy` trading ledger
(src/strategies/utils/positions.py).

The ledger is what lets you ask "what did the shares I bought on 2 June earn, and at what
price did they leave?" -- so the matching has to be right on the cases a rebalancing sleeve
actually produces: partial exits, top-ups, SHORTS (the L/S sleeve is half short), and a move
that flips a position from long to short in one go.

Each test states the arithmetic it expects explicitly, so a regression shows up as a wrong
number rather than a vaguely different frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.utils.blotter import trade_blotter
from src.strategies.utils.positions import LEDGER_COLUMNS, round_trip_ledger


def _blotter_rows(rows: list[tuple], sleeve: str = "ls_equity",
                  fee_bps: float = 0.0) -> pd.DataFrame:
    """A minimal blotter: rows of (date, ticker, signed_shares, price)."""
    recs = []
    for date, ticker, shares, price in rows:
        traded = shares * price
        fee = abs(traded) * fee_bps / 1e4
        recs.append({"date": pd.Timestamp(date), "sleeve": sleeve, "instrument": ticker,
                     "side": "BUY" if shares > 0 else "SELL", "shares_traded": float(shares),
                     "price": float(price), "traded_usd": traded, "shares_held": np.nan,
                     "position_usd": np.nan, "fee_usd": fee, "spread_usd": 0.0,
                     "cost_usd": fee})
    return pd.DataFrame(recs)


def test_simple_long_round_trip_fills_pnl_on_the_day_it_is_sold():
    """Buy 100 @210, sell 100 @228.50 -> the BUY row learns price_sold + pnl; the SELL row
    learns the cost basis it closed."""
    led = round_trip_ledger(_blotter_rows([
        ("2026-06-02", "AAPL", 100, 210.00),
        ("2026-07-28", "AAPL", -100, 228.50),
    ]), run_time=pd.Timestamp("2026-07-28 06:00"))

    assert list(led.columns) == LEDGER_COLUMNS
    buy, sell = led.iloc[0], led.iloc[1]

    assert buy["side"] == "BUY" and sell["side"] == "SELL"
    assert buy["price_bought"] == 210.00
    assert buy["price_sold"] == 228.50            # filled the day the position was sold
    assert buy["closed_on"] == pd.Timestamp("2026-07-28")
    assert buy["shares_closed"] == 100
    assert buy["pnl"] == pytest.approx((228.50 - 210.00) * 100)
    assert np.isnan(buy["pnl_closed_today"])      # the buy booked nothing on its own day

    assert sell["price_bought"] == 210.00         # FIFO cost basis of what it closed
    assert sell["price_sold"] == 228.50
    assert sell["pnl_closed_today"] == pytest.approx((228.50 - 210.00) * 100)
    assert np.isnan(sell["pnl"])                  # the sell opened nothing

    assert (led["run_time"] == pd.Timestamp("2026-07-28 06:00")).all()

    print("\n=== SANITY CHECK: long round trip ===")
    print(f"  BUY  2026-06-02 100 @210.00 -> price_sold {buy['price_sold']:.2f} on "
          f"{buy['closed_on'].date()}, pnl ${buy['pnl']:+,.2f}")
    print(f"  SELL 2026-07-28 100 @228.50 -> price_bought {sell['price_bought']:.2f}, "
          f"pnl_closed_today ${sell['pnl_closed_today']:+,.2f}")
    print("  (228.50-210.00)*100 = +1,850.00. Validated.")


def test_still_open_position_has_no_exit_price_or_pnl():
    led = round_trip_ledger(_blotter_rows([("2026-06-02", "MSFT", 50, 400.0)]))
    row = led.iloc[0]
    assert row["price_bought"] == 400.0
    assert np.isnan(row["price_sold"]) and np.isnan(row["pnl"]) and np.isnan(row["shares_closed"])
    assert row["closed_on"] is None or pd.isna(row["closed_on"])
    print("\n=== SANITY CHECK: open position ===")
    print("  BUY with no matching sell -> price_sold / pnl / shares_closed all NULL, "
          "closed_on NULL. Validated.")


def test_fifo_partial_exits_and_topup():
    """Buy 100@100, buy 100@120, sell 150@130. FIFO: the 150 sold = all 100 of the first lot
    + 50 of the second, so pnl = 100*(130-100) + 50*(130-120) = 3000 + 500 = 3500."""
    led = round_trip_ledger(_blotter_rows([
        ("2026-01-05", "XOM", 100, 100.0),
        ("2026-02-05", "XOM", 100, 120.0),
        ("2026-03-05", "XOM", -150, 130.0),
    ]))
    lot1 = led[led["trading_day"] == pd.Timestamp("2026-01-05")].iloc[0]
    lot2 = led[led["trading_day"] == pd.Timestamp("2026-02-05")].iloc[0]
    sell = led[led["trading_day"] == pd.Timestamp("2026-03-05")].iloc[0]

    assert lot1["shares_closed"] == 100 and lot1["pnl"] == pytest.approx(100 * 30)
    assert lot2["shares_closed"] == 50 and lot2["pnl"] == pytest.approx(50 * 10)
    # the sell closed 150 shares whose weighted basis is (100*100 + 50*120)/150
    assert sell["shares_closed"] == 150
    assert sell["price_bought"] == pytest.approx((100 * 100 + 50 * 120) / 150)
    assert sell["pnl_closed_today"] == pytest.approx(3500.0)
    # each view sums to the same realized total, independently
    assert led["pnl"].sum() == pytest.approx(3500.0)
    assert led["pnl_closed_today"].sum() == pytest.approx(3500.0)

    print("\n=== SANITY CHECK: FIFO partial exit + top-up ===")
    print(f"  lot1 100@100 fully closed -> pnl ${lot1['pnl']:+,.0f} (100 x 30)")
    print(f"  lot2 100@120 half closed  -> shares_closed {lot2['shares_closed']:.0f}, "
          f"pnl ${lot2['pnl']:+,.0f} (50 x 10)")
    print(f"  sell 150@130 basis {sell['price_bought']:.2f} (FIFO weighted), "
          f"pnl_closed_today ${sell['pnl_closed_today']:+,.0f}")
    print(f"  SUM(pnl)={led['pnl'].sum():+,.0f} == SUM(pnl_closed_today)="
          f"{led['pnl_closed_today'].sum():+,.0f} == 3,500 (no double count). Validated.")


def test_short_round_trip_pnl_is_entry_minus_exit():
    """A SHORT is opened by a SELL and closed by a BUY: sell 80 @50, buy back 80 @42
    -> pnl = (50-42)*80 = +640. price_sold is the ENTRY, price_bought the exit."""
    led = round_trip_ledger(_blotter_rows([
        ("2026-04-01", "TSLA", -80, 50.0),
        ("2026-05-01", "TSLA", 80, 42.0),
    ]))
    open_sell = led[led["trading_day"] == pd.Timestamp("2026-04-01")].iloc[0]
    close_buy = led[led["trading_day"] == pd.Timestamp("2026-05-01")].iloc[0]

    assert open_sell["side"] == "SELL" and open_sell["price_sold"] == 50.0
    assert open_sell["price_bought"] == 42.0      # the buy-back that closed it
    assert open_sell["pnl"] == pytest.approx((50.0 - 42.0) * 80)
    assert close_buy["side"] == "BUY" and close_buy["price_bought"] == 42.0
    assert close_buy["price_sold"] == 50.0       # the short's entry
    assert close_buy["pnl_closed_today"] == pytest.approx(640.0)
    assert led["pnl"].sum() == pytest.approx(640.0)

    print("\n=== SANITY CHECK: short round trip ===")
    print(f"  SELL 2026-04-01 80 @50 (opens short) -> bought back @"
          f"{open_sell['price_bought']:.2f}, pnl ${open_sell['pnl']:+,.0f}")
    print(f"  BUY  2026-05-01 80 @42 (closes)      -> price_sold {close_buy['price_sold']:.2f}, "
          f"pnl_closed_today ${close_buy['pnl_closed_today']:+,.0f}")
    print("  short profits when it falls: (50-42)*80 = +640. Validated.")


def test_flip_long_to_short_closes_then_opens():
    """Long 100 @10, then sell 250 @12: closes the 100 long (pnl +200) AND opens a 150 short
    at 12. The later buy-back of 150 @9 realizes (12-9)*150 = +450."""
    led = round_trip_ledger(_blotter_rows([
        ("2026-01-02", "KO", 100, 10.0),
        ("2026-02-02", "KO", -250, 12.0),
        ("2026-03-02", "KO", 150, 9.0),
    ]))
    long_row = led[led["trading_day"] == pd.Timestamp("2026-01-02")].iloc[0]
    flip = led[led["trading_day"] == pd.Timestamp("2026-02-02")].iloc[0]
    cover = led[led["trading_day"] == pd.Timestamp("2026-03-02")].iloc[0]

    assert long_row["pnl"] == pytest.approx(100 * 2.0)          # 100 x (12-10)
    # the flip closed 100 longs AND opened a 150 short that later closed at 9
    assert flip["pnl_closed_today"] == pytest.approx(200.0)     # what it closed on its own day
    assert flip["pnl"] == pytest.approx(150 * 3.0)              # what the short it opened earned
    assert flip["price_sold"] == 12.0                           # its own price
    # counterparty prices it dealt with: the 100 long's 10.00 entry + the 150 short's 9.00 exit
    assert flip["price_bought"] == pytest.approx((100 * 10.0 + 150 * 9.0) / 250)
    assert cover["pnl_closed_today"] == pytest.approx(450.0)
    assert led["pnl"].sum() == pytest.approx(650.0)
    assert led["pnl_closed_today"].sum() == pytest.approx(650.0)

    print("\n=== SANITY CHECK: position flip (long -> short in one move) ===")
    print(f"  BUY  100 @10                     -> pnl ${long_row['pnl']:+,.0f} (closed by the flip @12)")
    print(f"  SELL 250 @12 (closes 100, opens 150 short) -> closed today "
          f"${flip['pnl_closed_today']:+,.0f}, its own short later earned ${flip['pnl']:+,.0f}")
    print(f"  BUY  150 @9  (covers)            -> ${cover['pnl_closed_today']:+,.0f}")
    print(f"  total realized {led['pnl'].sum():+,.0f} = 200 (long) + 450 (short). Validated.")


def test_fees_are_charged_pro_rata_to_the_closed_part():
    """A lot only half closed is charged half of its opening fee. 10bps on both legs:
    buy 200@100 (fee $20), sell 100@110 (fee $11) -> closed half the buy, so fees deducted
    = 20/2 + 11 = 21, and pnl = 100*(110-100) - 21 = 979."""
    led = round_trip_ledger(_blotter_rows([
        ("2026-01-02", "PG", 200, 100.0),
        ("2026-02-02", "PG", -100, 110.0),
    ], fee_bps=10.0))
    buy = led.iloc[0]
    sell = led.iloc[1]
    assert buy["fee"] == pytest.approx(200 * 100 * 10 / 1e4)     # $20 on the full move
    assert sell["fee"] == pytest.approx(100 * 110 * 10 / 1e4)    # $11
    assert buy["pnl"] == pytest.approx(1000.0 - (10.0 + 11.0))
    assert sell["pnl_closed_today"] == pytest.approx(979.0)

    print("\n=== SANITY CHECK: pro-rata fees ===")
    print(f"  BUY 200@100 fee ${buy['fee']:.2f}; only 100 shares closed -> $10.00 charged")
    print(f"  SELL 100@110 fee ${sell['fee']:.2f} charged in full")
    print(f"  pnl = 1000 gross - 21 fees = ${buy['pnl']:+,.2f}. Validated.")


def test_cash_legs_without_a_price_are_dropped():
    """The long_book sleeve trades a `cash` residual with no price. It is an accounting
    residual, not an order you place, and shares / entry / exit / P&L are undefined for it."""
    rows = _blotter_rows([("2026-01-02", "equity", 10, 100.0)], sleeve="long_book")
    cash = rows.iloc[0].copy()
    cash["instrument"] = "cash"; cash["price"] = np.nan; cash["shares_traded"] = np.nan
    cash["traded_usd"] = -1000.0
    led = round_trip_ledger(pd.concat([rows, cash.to_frame().T], ignore_index=True))
    assert set(led["ticker"]) == {"equity"}
    print("\n=== SANITY CHECK: price-less cash leg ===")
    print("  long_book 'cash' row (no price) dropped; only the priced 'equity' move is a "
          "ledger entry. Validated.")


def test_end_to_end_from_a_real_blotter():
    """Straight through `trade_blotter` -> ledger, so the column contract between the two is
    pinned: a 2-day 2-name weight panel produces matched round trips with sane totals."""
    idx = pd.bdate_range("2026-03-02", periods=3)
    weights = pd.DataFrame({"AAA": [0.5, 0.5, 0.0], "BBB": [-0.5, 0.0, 0.0]}, index=idx)
    prices = pd.DataFrame({"AAA": [100.0, 105.0, 110.0], "BBB": [50.0, 48.0, 47.0]}, index=idx)
    trades = trade_blotter(weights, capital=1_000_000, fee_bps=1.0, spread_bps=0.0,
                           sleeve="ls_equity", prices=prices)
    led = round_trip_ledger(trades)

    assert not led.empty and list(led.columns) == LEDGER_COLUMNS
    assert set(led["sleeve"]) == {"ls_equity"}
    # AAA: long 5000sh @100, trimmed to 4761.9sh @105 (the $ target holds as price rises),
    # exited @110. BBB: short 10000sh @50, covered @48.
    aaa = led[led["ticker"] == "AAA"].sort_values("trading_day")
    bbb = led[led["ticker"] == "BBB"].sort_values("trading_day")
    assert aaa.iloc[0]["side"] == "BUY" and aaa.iloc[-1]["side"] == "SELL"
    assert bbb.iloc[0]["side"] == "SELL" and bbb.iloc[-1]["side"] == "BUY"
    assert aaa["pnl"].sum() > 0                       # AAA rose while long
    assert bbb["pnl"].sum() > 0                       # BBB fell while short
    realized = led["pnl"].sum()
    assert realized == pytest.approx(led["pnl_closed_today"].sum())

    print("\n=== SANITY CHECK: blotter -> ledger end to end ===")
    print(led[["trading_day", "ticker", "side", "shares", "price", "price_bought",
               "price_sold", "pnl"]].to_string(index=False))
    print(f"  AAA long into a rising price and BBB short into a falling one both realize a "
          f"profit; total realized ${realized:+,.2f}, identical from both P&L views. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
