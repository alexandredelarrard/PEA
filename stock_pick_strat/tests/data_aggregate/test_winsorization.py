"""The z-score features are winsorized to a per-day cross-sectional [1%, 99%]
band so a few extreme names can't dominate the standardized value / the model
(src/data_aggregate/utils/fundamental_features.py::_winsorize_xs, applied to the
peer-z `_vs_peers` and the self-history `_vs_hist` outputs). The rank `_xs`
features are already outlier-proof and are left untouched.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.panel import _winsorize_xs, _peer_relative, build_peer_relative_panel


def test_peer_panel_no_fragmentation_warning():
    """Regression: build_peer_relative_panel concats one block per feature column,
    so once the panel has 100+ columns the reset_index() insert used to emit
    pandas' 'DataFrame is highly fragmented' PerformanceWarning. The `.copy()`
    that consolidates the blocks must keep it silent."""
    dates = pd.bdate_range("2022-01-03", periods=30)
    tickers = [f"T{i}" for i in range(6)]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    rng = np.random.default_rng(0)
    # 60 fields -> 120 f_*_vs_peers/_xs columns (+ date,ticker) => well over 100 blocks
    fields = {f"feat{k}": pd.DataFrame(rng.standard_normal((len(dates), len(tickers))),
                                       index=dates, columns=tickers)
              for k in range(60)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = build_peer_relative_panel(fields, peers)
    frag = [w for w in caught if "fragmented" in str(w.message).lower()]

    assert not frag, f"fragmentation warning emitted: {[str(w.message)[:80] for w in frag]}"
    assert panel.shape[1] == 2 + 60 * 2 and {"date", "ticker"} <= set(panel.columns)
    print("\n=== SANITY CHECK: peer panel not fragmented ===")
    print(f"  built {panel.shape[1]-2} feature cols from 60 fields with ZERO "
          "'highly fragmented' warnings (blocks consolidated via .copy()). Validated.")


def test_winsorize_xs_clips_each_row_to_1_99():
    # one date, 100 tickers with values 0..99; extremes must be pulled to the 1/99 pct
    row = pd.DataFrame([{f"t{i}": float(i) for i in range(100)}], index=[pd.Timestamp("2024-01-02")])
    w = _winsorize_xs(row)
    assert w.iloc[0].max() == pytest.approx(row.iloc[0].quantile(0.99))
    assert w.iloc[0].min() == pytest.approx(row.iloc[0].quantile(0.01))
    # interior values unchanged; only the 1% tails clipped
    assert (w.iloc[0].sort_values().iloc[1:-1].to_numpy()
            == pytest.approx(row.iloc[0].sort_values().iloc[1:-1].to_numpy()))
    # all-NaN row -> no crash, stays NaN
    nanrow = pd.DataFrame([{f"t{i}": np.nan for i in range(5)}])
    assert _winsorize_xs(nanrow).isna().all().all()

    print("\n=== SANITY CHECK: cross-sectional 1/99 winsorize ===")
    print(f"  values 0..99 -> max clipped {row.iloc[0].max():.0f} -> {w.iloc[0].max():.2f} "
          f"(99th pct), min 0 -> {w.iloc[0].min():.2f} (1st pct); interior untouched. Validated.")


def test_peer_z_outlier_is_trimmed_but_rank_untouched():
    # 60 tickers, one field; ONE ticker is a wild outlier on the last date.
    dates = pd.bdate_range("2024-01-02", periods=3)
    tickers = [f"T{i}" for i in range(60)]
    rng = np.random.default_rng(0)
    field = pd.DataFrame(rng.normal(1.0, 0.05, size=(3, 60)), index=dates, columns=tickers)
    field.loc[dates[-1], "T0"] = 1e6                     # extreme outlier
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    d = dates[-1]
    raw = _peer_relative(field, peers)                    # un-winsorized peer-z
    wins = _winsorize_xs(raw)

    # T0's raw peer-z is extreme (hits the internal +-8 clip); winsorize then pulls
    # it down to the row's cross-sectional 99th percentile (< raw -> trimmed).
    assert raw.loc[d, "T0"] >= 7.9
    assert wins.loc[d, "T0"] == pytest.approx(raw.loc[d].quantile(0.99))
    assert wins.loc[d, "T0"] < raw.loc[d, "T0"]

    # the built panel reflects the winsorized value; the rank `_xs` is untouched (T0 top).
    last = build_peer_relative_panel({"metric": field}, peers)
    last = last[last["date"] == d].set_index("ticker")
    assert last.loc["T0", "f_metric_vs_peers"] == pytest.approx(wins.loc[d, "T0"])
    assert last.loc["T0", "f_metric_xs"] == pytest.approx(1.0, abs=1e-9)

    print("\n=== SANITY CHECK: peer-z outlier trim + rank intact ===")
    print(f"  T0 raw=1e6: peer-z {raw.loc[d,'T0']:.2f} (at +-8 clip) -> winsorized "
          f"{wins.loc[d,'T0']:.3f} (day 99th pct); f_metric_xs rank still 1.00. Validated.")
