"""Step 6 — Short-selling-pressure features (FINRA RegSHO short volume).

Checks the RegSHO file parser, the short-volume-ratio features, the 1-trading-day
publication lag (point-in-time), and the built f_* panel columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.prices.fetch_short_interest import _parse_regsho
from src.data_aggregate.utils.short_interest_features import (
    _short_fields, build_short_interest_feature_panel,
)


def test_parse_regsho():
    txt = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
           "20230103|AAPL|500|0|1000|Q\n"
           "20230103|AAPL|100|0|200|N\n"      # second market center -> summed
           "20230103|MSFT|300|0|1200|Q\n")
    out = _parse_regsho(txt)
    aapl = out[out["ticker"] == "AAPL"].iloc[0]
    assert aapl["short_volume"] == 600 and aapl["total_volume"] == 1200   # summed
    assert out["date"].iloc[0] == pd.Timestamp("2023-01-03")
    assert _parse_regsho("").empty
    print("\n=== SANITY CHECK: RegSHO parser ===")
    print(f"  pipe file parsed; AAPL short/total summed across markets = "
          f"{int(aapl['short_volume'])}/{int(aapl['total_volume'])}; empty -> empty. Validated.")


def _synth(T=200, N=8, seed=0):
    dates = pd.bdate_range("2023-01-03", periods=T)
    tickers = [f"S{i}" for i in range(N)]
    rng = np.random.default_rng(seed)
    total = rng.uniform(1e6, 5e6, (T, N))
    frac = np.clip(rng.normal(0.4, 0.1, (T, N)), 0.05, 0.95)
    frac[:, 0] = 0.85                      # S0 persistently heavily shorted
    rows = []
    for j, t in enumerate(tickers):
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t,
                         "short_volume": total[i, j] * frac[i, j],
                         "total_volume": total[i, j]})
    return dates, tickers, pd.DataFrame(rows)


def test_short_ratio_and_pit_lag():
    dates, tickers, hist = _synth()
    F = _short_fields(hist, dates)
    assert {"short_vol_ratio", "short_vol_ratio_chg"}.issubset(F)

    # heavily-shorted S0 has the highest ratio cross-sectionally
    t = dates[120]
    assert F["short_vol_ratio"].loc[t].idxmax() == "S0"

    # 1-day publication lag: the ratio at date t must equal the RAW rolling mean
    # computed THROUGH t-1 (i.e. it lags by one trading day)
    daily = hist.pivot_table(index="date", columns="ticker", values="short_volume", aggfunc="sum") \
        / hist.pivot_table(index="date", columns="ticker", values="total_volume", aggfunc="sum")
    expected_tm1 = daily.rolling(21, min_periods=5).mean().loc[dates[119], "S0"]
    assert np.isclose(F["short_vol_ratio"].loc[dates[120], "S0"], expected_tm1), "lag broken"
    print("\n=== SANITY CHECK: short-vol ratio + 1-day publication lag ===")
    print(f"  S0 (85% shorted) tops short_vol_ratio at {t.date()}; the value at t "
          f"equals the ratio computed through t-1 (published next morning). Validated.")


def test_panel_columns():
    dates, tickers, hist = _synth()
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    panel = build_short_interest_feature_panel(hist, peers, dates)
    for c in ("f_short_vol_ratio_xs", "f_short_vol_ratio_chg_xs", "f_short_vol_ratio_vs_peers"):
        assert c in panel.columns, f"{c} missing"
    assert panel["f_short_vol_ratio_xs"].dropna().between(0, 1).all()
    print("\n=== SANITY CHECK: short-interest panel columns ===")
    print("  panel exposes f_short_vol_ratio(_chg) (_xs & _vs_peers), xs in [0,1]. Validated.")


if __name__ == "__main__":
    test_parse_regsho()
    test_short_ratio_and_pit_lag()
    test_panel_columns()
