"""Step 2 — Dividend / shareholder-yield features.

Checks: features are point-in-time; non-payers get a real 0 yield; rising
dividends -> positive growth; shareholder_yield adds buyback yield (up for
repurchasers, down for issuers); the built panel exposes the expected f_*
columns. (The dividend extractor's own parser, `_extract_dividends`, is
covered in test_da_capex_and_dividend_consolidation.py.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamentals.dividend_features import (
    _dividend_fields, build_dividend_feature_panel,
)

_YOY = 252


def _synth():
    dates = pd.bdate_range("2021-01-04", periods=3 * 252)      # ~3y
    payers = ["P0", "P1", "P2", "P3"]
    nonpayers = ["N0", "N1", "N2", "N3"]
    tickers = payers + nonpayers
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(dates), len(tickers))), axis=0),
                         index=dates, columns=tickers)

    # quarterly dividends; P0 GROWS its payout each year, others flat
    ex_dates = dates[::63]
    rows = []
    for i, d in enumerate(ex_dates):
        year = i // 4
        for p in payers:
            amt = 0.5 * (1.15 ** year if p == "P0" else 1.0)   # P0 grows 15%/yr
            rows.append({"date": d, "ticker": p, "dividend": amt})
    div_hist = pd.DataFrame(rows)

    # fundamentals: sharesOutstanding — P0 buys back, N0 issues, rest flat
    frows = []
    for q, aso in enumerate(dates[::63]):
        for t in tickers:
            base = 1e9
            shares = base * (0.9 ** (q / 4) if t == "P0" else
                             1.1 ** (q / 4) if t == "N0" else 1.0)
            frows.append({"ticker": t, "as_of": aso, "sharesOutstanding": shares})
    fund = pd.DataFrame(frows)
    return dates, tickers, close, div_hist, fund


def test_dividend_fields_economics_and_pit():
    dates, tickers, close, div_hist, fund = _synth()
    F = _dividend_fields(div_hist, close, fund)
    for k in ("dividend_yield", "dividend_growth", "dividend_payer", "shareholder_yield"):
        assert k in F, f"missing {k}"

    t = dates[-1]
    # non-payer -> 0 yield, flag 0; payer -> positive yield, flag 1
    assert F["dividend_yield"].loc[t, "N0"] == 0.0
    assert F["dividend_payer"].loc[t, "N0"] == 0.0
    assert F["dividend_yield"].loc[t, "P0"] > 0
    assert F["dividend_payer"].loc[t, "P0"] == 1.0
    # rising payout -> positive growth for P0, ~0 for a flat payer
    assert F["dividend_growth"].loc[t, "P0"] > 0.05
    assert abs(F["dividend_growth"].loc[t, "P1"]) < 0.02
    # shareholder yield: buyback firm > its dividend yield; issuer < its div yield
    assert F["shareholder_yield"].loc[t, "P0"] > F["dividend_yield"].loc[t, "P0"]
    assert F["shareholder_yield"].loc[t, "N0"] < F["dividend_yield"].loc[t, "N0"]

    # point-in-time: perturb everything AFTER t -> value at t unchanged
    div2 = pd.concat([div_hist, pd.DataFrame([{"date": dates[-1], "ticker": "P0",
                                               "dividend": 99.0}])], ignore_index=True)
    tm = dates[len(dates) // 2]
    F2 = _dividend_fields(div2, close, fund)
    assert np.isclose(F["dividend_yield"].loc[tm, "P0"], F2["dividend_yield"].loc[tm, "P0"])

    print("\n=== SANITY CHECK: dividend economics + point-in-time ===")
    print(f"  P0 yield={F['dividend_yield'].loc[t,'P0']:.3f} growth={F['dividend_growth'].loc[t,'P0']:.2f} "
          f"shareholder={F['shareholder_yield'].loc[t,'P0']:.3f} (>div, buyback); "
          f"N0 yield=0, shareholder<0 (issuer). Future dividend didn't change past value. Validated.")


def _synth_reconcile():
    """~6y so the 5y CAGR is defined. Three names:
      A      — pays via BOTH sources (per-share ex-dates AND `dividendsPaid`), grows 10%/yr,
      B_ONLY — pays via source B ONLY (no ex-date history; yfinance missed it), flat,
      N      — never pays."""
    dates = pd.bdate_range("2018-01-01", periods=6 * 252 + 20)
    tickers = ["A", "B_ONLY", "N"]
    close = pd.DataFrame(100.0, index=dates, columns=tickers)
    ex = dates[::63]                                        # quarterly ex-dates
    rows = [{"date": d, "ticker": "A", "dividend": 1.0 * (1.10 ** (i // 4))}
            for i, d in enumerate(ex)]                      # A grows 10%/yr per share
    div_hist = pd.DataFrame(rows)
    frows = []
    for q, aso in enumerate(dates[::63]):
        ttm = 4.0 * (1.10 ** (q // 4))                      # A's TTM per-share ~= 4 * quarterly
        frows += [
            {"ticker": "A", "as_of": aso, "sharesOutstanding": 1000.0,
             "dividendsPaid": ttm * 1000.0, "netIncome": 8000.0, "freeCashflow": 12000.0},
            {"ticker": "B_ONLY", "as_of": aso, "sharesOutstanding": 1000.0,
             "dividendsPaid": 2000.0, "netIncome": 8000.0, "freeCashflow": 1000.0},
            {"ticker": "N", "as_of": aso, "sharesOutstanding": 1000.0,
             "dividendsPaid": 0.0, "netIncome": 8000.0, "freeCashflow": 5000.0},
        ]
    return dates, close, div_hist, pd.DataFrame(frows)


def test_reconcile_sources_5y_growth_payout_coverage():
    dates, close, div_hist, fund = _synth_reconcile()
    F = _dividend_fields(div_hist, close, fund)
    t = dates[-1]
    for k in ("dividend_growth_5y", "dividend_payout_ratio", "dividend_coverage"):
        assert k in F, f"missing {k}"

    # RECONCILIATION: A's per-share (source A) yield agrees with its source-B yield
    # (dividendsPaid/mcap) — same cash two ways.
    src_b_yield_A = 4.0 * (1.10 ** 5) * 1000.0 / (1000.0 * 100.0)   # dividendsPaid / mcap
    assert abs(F["dividend_yield"].loc[t, "A"] - src_b_yield_A) < 0.03
    # RECONCILIATION gap-fill: B_ONLY has NO ex-dates, yield comes from source B = 2000/100000
    assert abs(F["dividend_yield"].loc[t, "B_ONLY"] - 0.02) < 1e-6
    assert F["dividend_payer"].loc[t, "B_ONLY"] == 1.0        # counted as a payer via source B
    assert F["dividend_yield"].loc[t, "N"] == 0.0             # true non-payer -> real 0

    # 5-YEAR GROWTH: A grew per-share ~10%/yr -> 5y CAGR ~= 0.10
    assert abs(F["dividend_growth_5y"].loc[t, "A"] - 0.10) < 0.02
    assert pd.isna(F["dividend_growth_5y"].loc[t, "N"])       # non-payer -> undefined

    # PAYOUT + COVERAGE (dividend safety): B_ONLY pays 2000 on 8000 NI (25%), FCF only
    # 1000 -> coverage 0.5 (<1, UNSAFE); A's FCF 12000 comfortably covers -> coverage > 1
    assert abs(F["dividend_payout_ratio"].loc[t, "B_ONLY"] - 0.25) < 1e-6
    assert abs(F["dividend_coverage"].loc[t, "B_ONLY"] - 0.5) < 1e-6
    assert F["dividend_coverage"].loc[t, "A"] > 1.0 > F["dividend_coverage"].loc[t, "B_ONLY"]

    print("\n=== SANITY CHECK: dividend source reconciliation + 5y growth + safety ===")
    print(f"  A yield (src A) {F['dividend_yield'].loc[t,'A']:.4f} ~= dividendsPaid/mcap "
          f"{src_b_yield_A:.4f} (two sources agree); B_ONLY yield {F['dividend_yield'].loc[t,'B_ONLY']:.3f} "
          f"gap-filled from source B; A 5y CAGR {F['dividend_growth_5y'].loc[t,'A']:.3f}~0.10; "
          f"B_ONLY payout {F['dividend_payout_ratio'].loc[t,'B_ONLY']:.2f}, coverage "
          f"{F['dividend_coverage'].loc[t,'B_ONLY']:.2f}<1 (unsafe) < A {F['dividend_coverage'].loc[t,'A']:.2f}. Validated.")


def test_panel_exposes_f_columns():
    dates, tickers, close, div_hist, fund = _synth()
    # give each ticker enough peers for the _vs_peers column to populate
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    panel = build_dividend_feature_panel(div_hist, peers, dates, stock_close=close,
                                         fundamentals_history=fund)
    for c in ("f_dividend_yield_xs", "f_dividend_growth_xs", "f_dividend_payer_xs",
              "f_shareholder_yield_xs", "f_dividend_yield_vs_peers"):
        assert c in panel.columns, f"{c} missing from panel"
    xs = panel["f_dividend_yield_xs"].dropna()
    assert xs.between(0, 1).all()
    print("\n=== SANITY CHECK: dividend panel columns ===")
    print(f"  panel has f_dividend_yield/growth/payer/shareholder_yield (_xs & _vs_peers); "
          f"xs rank in [0,1]. Rows={len(panel)}. Validated.")


if __name__ == "__main__":
    test_dividend_fields_economics_and_pit()
    test_panel_exposes_f_columns()
