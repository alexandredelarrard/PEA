"""Peer-relative panel construction: no fragmentation warning, and an outlier input is clipped
at the +-5 peer-z bound while the rank `_xs` feature is untouched.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.panel import build_peer_relative_panel, peer_relative


def test_peer_panel_no_fragmentation_warning(monkeypatch):
    """Regression: build_peer_relative_panel concats one block per feature column,
    so once the panel has 100+ columns the reset_index() insert used to emit
    pandas' 'DataFrame is highly fragmented' PerformanceWarning. Building the keys
    separately must keep it silent without consolidating the multi-gigabyte feature frame."""
    dates = pd.bdate_range("2022-01-03", periods=30)
    tickers = [f"T{i}" for i in range(6)]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    rng = np.random.default_rng(0)
    # 60 fields -> 120 f_*_vs_peers/_xs columns (+ date,ticker) => well over 100 blocks
    fields = {f"feat{k}": pd.DataFrame(rng.standard_normal((len(dates), len(tickers))), index=dates, columns=tickers) for k in range(60)}

    original_copy = pd.DataFrame.copy

    def reject_wide_eager_copy(frame, *args, **kwargs):
        if frame.shape[1] >= 100:
            raise AssertionError("peer panel attempted a full-width eager copy")
        return original_copy(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", reject_wide_eager_copy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = build_peer_relative_panel(fields, peers)
    frag = [w for w in caught if "fragmented" in str(w.message).lower()]

    assert not frag, f"fragmentation warning emitted: {[str(w.message)[:80] for w in frag]}"
    assert panel.shape[1] == 2 + 60 * 2 and {"date", "ticker"} <= set(panel.columns)
    print("\n=== SANITY CHECK: peer panel not fragmented ===")
    print(f"  built {panel.shape[1] - 2} feature cols from 60 fields with ZERO 'highly fragmented' warnings and no full-width eager copy. Validated.")


def test_peer_z_outlier_input_is_clipped_not_trimmed_and_rank_untouched():
    # 60 tickers, one field; ONE ticker is a wild outlier on the last date.
    dates = pd.bdate_range("2024-01-02", periods=3)
    tickers = [f"T{i}" for i in range(60)]
    rng = np.random.default_rng(0)
    field = pd.DataFrame(rng.normal(1.0, 0.05, size=(3, 60)), index=dates, columns=tickers)
    field.loc[dates[-1], "T0"] = 1e6  # extreme outlier
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    d = dates[-1]
    rel = peer_relative(field, peers)

    # The outlier's own score sits at the fixed +-5 clip; no input is trimmed, so a ticker whose basket
    # contains the outlier sees it in the basket mean (the other tickers are scored against it).
    assert rel.loc[d, "T0"] == 5.0
    assert (rel.abs().stack() <= 5.0).all()

    # the built panel ships the same score; the rank `_xs` is untouched (T0 top).
    last = build_peer_relative_panel({"metric": field}, peers)
    last = last[last["date"] == d].set_index("ticker")
    assert last.loc["T0", "f_metric_vs_peers"] == pytest.approx(5.0)
    assert last.loc["T0", "f_metric_xs"] == pytest.approx(1.0, abs=1e-9)

    print("\n=== SANITY CHECK: peer-z outlier clip + rank intact ===")
    print("  T0 raw=1e6: peer-z clipped to exactly 5.00, f_metric_xs rank still 1.00. Validated.")
