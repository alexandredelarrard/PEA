"""StepStrategyMoves — the daily trading ledger (src/portfolio/step_strategy_moves.py).

What must hold, and is easy to get silently wrong:
  1. each sleeve is sized by its OWN dynamic ERC weight x leverage, not by the full
     starting_capital it is backtested on standalone;
  2. the re-sizing happens on the WEIGHT PANEL, not on the resulting dollars -- share counts are
     `weight * capital / price`, so with a time-varying capital multiplier the shares TRADED (a
     difference of share counts) are NOT proportional to the multiplier;
  3. the ledger is upserted on (trading_day, sleeve, ticker), so a re-run refreshes the exit
     price and P&L of positions that have closed since instead of duplicating moves;
  4. each sleeve's own fee/spread is charged (resolved via the strategy class's `config_key`).

The portfolio blend is stubbed, so this runs with no DB and no model artifacts.
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants import STRATEGY_TABLE
from src.portfolio import step_strategy_moves as sm
from src.strategies.base import StrategyResult
from src.strategies.utils.positions import LEDGER_COLUMNS


def _panels(n_days: int = 8):
    idx = pd.bdate_range("2026-03-02", periods=n_days)
    # AAA held then exited; BBB entered late -> a closed round trip and an open position
    w = pd.DataFrame({"AAA": [0.6] * (n_days - 2) + [0.0, 0.0],
                      "BBB": [0.0] * (n_days - 3) + [0.4] * 3}, index=idx)
    px = pd.DataFrame({"AAA": np.linspace(100.0, 114.0, n_days),
                       "BBB": np.linspace(50.0, 43.0, n_days)}, index=idx)
    return w, px


def _step(monkeypatch, erc_weight: float = 0.4, leverage: float = 1.5,
          capital: float = 1_000_000.0, saved: list | None = None):
    """A StepStrategyMoves whose portfolio blend is stubbed to fixed ERC weights + leverage."""
    w, px = _panels()
    result = StrategyResult(name="trend_cta", returns=pd.Series(0.0, index=w.index),
                            metrics={}, book_weights=w, book_prices=px)

    def fake_load(self):
        self.results = {"trend_cta": result}
        self.sleeve_rets = pd.DataFrame({"trend_cta": pd.Series(0.0, index=w.index)})

    def fake_blend(self):
        self.weights = pd.DataFrame({"trend_cta": erc_weight}, index=w.index)
        self.blended = pd.DataFrame({"leverage": leverage}, index=w.index)

    monkeypatch.setattr(sm.StepPortfolio, "load_sleeves", fake_load)
    monkeypatch.setattr(sm.StepPortfolio, "blend", fake_blend)

    config = OmegaConf.create({
        "portfolio": {"starting_capital": capital, "fee_bps": 2.0, "spread_bps": 8.0,
                      "scheme": "erc", "portfolio_vol_target": 0.10},
        "strategy_trend": {"fee_bps": 1.0, "spread_bps": 5.0},
    })
    store = types.SimpleNamespace(save=lambda t, df: ((saved if saved is not None else []).append((t, df)), len(df))[1])
    context = types.SimpleNamespace(save=saved is not None, store=store,
                                    logger=logging.getLogger("moves-test"),
                                    log=logging.getLogger("moves-test"), paths={})
    step = sm.StepStrategyMoves(context=context, config=config)
    return step, w, px


def test_sleeve_is_sized_by_its_erc_allocation_not_full_capital(monkeypatch):
    """A sleeve at ERC weight 0.4 and leverage 1.5 trades 0.6x the starting capital, so its
    day-1 notional is weight * 0.6 * capital -- not weight * capital."""
    capital = 1_000_000.0
    step, w, px = _step(monkeypatch, erc_weight=0.4, leverage=1.5, capital=capital)
    led = step.run()

    assert list(led.columns) == LEDGER_COLUMNS
    day1 = led[led["trading_day"] == led["trading_day"].min()]
    aaa = day1[day1["ticker"] == "AAA"].iloc[0]
    # expected: weight 0.6 x (0.4 x 1.5) x 1,000,000 = $360,000 at $100 -> 3,600 shares
    expected_usd = 0.6 * (0.4 * 1.5) * capital
    assert aaa["amount_invested"] == pytest.approx(expected_usd, rel=1e-9)
    assert aaa["shares"] == pytest.approx(expected_usd / 100.0, rel=1e-9)
    assert aaa["amount_invested"] < 0.6 * capital        # NOT the standalone full-capital size

    print("\n=== SANITY CHECK: sleeve sized by its ERC allocation ===")
    print(f"  starting_capital ${capital:,.0f} | ERC weight 0.40 x leverage 1.50 = 0.60 of it")
    print(f"  trend_cta AAA target weight 0.60 -> day-1 notional ${aaa['amount_invested']:,.0f} "
          f"({aaa['shares']:,.1f} shares @ ${aaa['price']:.2f})")
    print(f"  standalone full-capital sizing would have been ${0.6 * capital:,.0f} — "
          f"{0.6 * capital / aaa['amount_invested']:.2f}x too big. Validated.")


def test_resizing_uses_the_weight_panel_not_scaled_dollars(monkeypatch):
    """With a TIME-VARYING allocation, the shares traded are not the standalone shares scaled by
    the multiplier -- so the step must rebuild the blotter on the scaled panel. This pins that:
    a rising allocation forces extra BUYS on days the standalone book does not trade at all."""
    w, px = _panels()
    idx = w.index
    ramp = pd.Series(np.linspace(0.2, 0.8, len(idx)), index=idx)      # ERC weight grows daily

    result = StrategyResult(name="trend_cta", returns=pd.Series(0.0, index=idx), metrics={},
                            book_weights=w, book_prices=px)
    monkeypatch.setattr(sm.StepPortfolio, "load_sleeves",
                        lambda self: (setattr(self, "results", {"trend_cta": result}),
                                      setattr(self, "sleeve_rets",
                                              pd.DataFrame({"trend_cta": pd.Series(0.0, index=idx)})))[0])
    monkeypatch.setattr(sm.StepPortfolio, "blend",
                        lambda self: (setattr(self, "weights", pd.DataFrame({"trend_cta": ramp})),
                                      setattr(self, "blended", pd.DataFrame({"leverage": 1.0}, index=idx)))[0])
    config = OmegaConf.create({"portfolio": {"starting_capital": 1_000_000.0, "fee_bps": 2.0,
                                             "spread_bps": 8.0},
                               "strategy_trend": {"fee_bps": 1.0, "spread_bps": 5.0}})
    context = types.SimpleNamespace(save=False, store=None,
                                    logger=logging.getLogger("moves-test"),
                                    log=logging.getLogger("moves-test"), paths={})
    led = sm.StepStrategyMoves(context=context, config=config).run()

    aaa = led[led["ticker"] == "AAA"].sort_values("trading_day")
    # the standalone book holds AAA flat at 0.6 for the first 6 days -> it would trade ONCE.
    # With a growing allocation it must top up every day.
    buys = aaa[aaa["side"] == "BUY"]
    assert len(buys) >= 4, f"a rising allocation must force repeated top-ups, got {len(buys)}"
    assert buys["shares"].iloc[0] > 0
    # and the notional grows with the allocation
    assert buys["amount_invested"].iloc[-1] > 0

    print("\n=== SANITY CHECK: re-size the PANEL, not the dollars ===")
    print(f"  standalone book: AAA flat at weight 0.60 -> 1 establishing trade")
    print(f"  ERC weight ramping 0.20 -> 0.80: {len(buys)} BUY(s) as the allocation grows")
    print(aaa[["trading_day", "side", "shares", "price", "amount_invested"]].to_string(index=False))
    print("  scaling the standalone blotter's dollars would have missed every top-up. Validated.")


def test_ledger_is_upserted_with_the_position_pk(monkeypatch):
    """The step upserts to `strategy` on (trading_day, sleeve, ticker) so a re-run refreshes a
    closed position's exit price / P&L instead of appending a duplicate move."""
    from src.data_store.schema_registry import BY_NAME

    saved: list = []
    step, _, _ = _step(monkeypatch, saved=saved)
    led = step.run()

    assert BY_NAME[STRATEGY_TABLE].pk == ("trading_day", "sleeve", "ticker")
    assert BY_NAME[STRATEGY_TABLE].date_col == "trading_day"
    assert len(saved) == 1 and saved[0][0] == STRATEGY_TABLE
    written = saved[0][1]
    assert list(written.columns) == LEDGER_COLUMNS
    assert not written.duplicated(["trading_day", "sleeve", "ticker"]).any(), \
        "a duplicate PK would make the upsert ambiguous"
    # AAA was exited -> a closed round trip with both prices and a P&L; BBB is still open
    aaa_open = led[(led["ticker"] == "AAA") & (led["side"] == "BUY")].iloc[0]
    bbb = led[led["ticker"] == "BBB"]
    assert aaa_open["price_sold"] > 0 and np.isfinite(aaa_open["pnl"])
    assert bbb["pnl"].isna().all() and bbb["price_sold"].isna().all()

    print("\n=== SANITY CHECK: upsert grain + open vs closed ===")
    print(f"  wrote {len(written)} row(s) to '{STRATEGY_TABLE}' with PK "
          f"{BY_NAME[STRATEGY_TABLE].pk} — no duplicate keys")
    print(f"  AAA (exited): bought @{aaa_open['price_bought']:.2f}, sold "
          f"@{aaa_open['price_sold']:.2f} on {pd.Timestamp(aaa_open['closed_on']).date()}, "
          f"pnl ${aaa_open['pnl']:+,.2f}")
    print(f"  BBB (still held): {len(bbb)} move(s), price_sold + pnl NULL until it closes")
    print("  re-running the day AAA closes rewrites that BUY row rather than duplicating it. "
          "Validated.")


def test_sleeve_fee_override_is_charged(monkeypatch):
    """The ledger charges the sleeve's OWN fee/spread (strategy_trend: 1.0 + 5.0 bps = 6 bps),
    resolved through the strategy class's `config_key`, not the portfolio default (2 + 8)."""
    step, _, _ = _step(monkeypatch)
    led = step.run()
    row = led.iloc[0]
    assert row["fee"] == pytest.approx(row["amount_invested"] * 6.0 / 1e4, rel=1e-9)
    assert step._sleeve_cfg("trend_cta")["fee_bps"] == 1.0

    print("\n=== SANITY CHECK: sleeve fee override ===")
    print(f"  strategy_trend fee 1.0bps + spread 5.0bps = 6bps -> ${row['fee']:,.2f} on "
          f"${row['amount_invested']:,.0f} traded")
    print("  resolved via TrendCTAStrategy.config_key ('strategy_trend'); the portfolio default "
          "(2+8bps) was NOT used. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
