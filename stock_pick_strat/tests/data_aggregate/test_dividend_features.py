"""Step 2 — Dividend / shareholder-yield features.

Checks: extractor parser is pure/clean; features are point-in-time; non-payers
get a real 0 yield; rising dividends -> positive growth; shareholder_yield adds
buyback yield (up for repurchasers, down for issuers); the built panel exposes
the expected f_* columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.fetch_dividends import _series_to_long
from src.data_aggregate.utils.dividend_features import (
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


def test_series_to_long_parser():
    s = pd.Series([0.5, 0.0, 0.6, np.nan], index=pd.to_datetime(
        ["2021-03-01", "2021-06-01", "2021-09-01", "2021-12-01"]))
    out = _series_to_long(s, "AAA")
    assert list(out.columns) == ["date", "ticker", "dividend"]
    assert len(out) == 2 and (out["dividend"] > 0).all()     # drops 0 and NaN
    assert _series_to_long(pd.Series(dtype=float), "AAA").empty
    print("\n=== SANITY CHECK: dividend parser ===")
    print(f"  kept {len(out)} positive ex-dates, dropped 0/NaN; empty series -> empty frame. Validated.")


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
    test_series_to_long_parser()
    test_dividend_fields_economics_and_pit()
    test_panel_exposes_f_columns()
