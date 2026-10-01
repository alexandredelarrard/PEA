"""Cross-source availability support metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.cross_source_features import EMISSION, build_cross_source_panel
from src.data_aggregate.utils.institutionals.sink import BEARISH_INPUTS, BULLISH_INPUTS, ConditioningSink
from tests.conftest import make_frames

IDX = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=5))
TICKERS = ["AAA", "BBB"]
PEERS = {ticker: {peer: 1.0 for peer in TICKERS if peer != ticker} for ticker in TICKERS}


def _register(sink: ConditioningSink, name: str, available: pd.DataFrame) -> None:
    values = pd.DataFrame(1.0, index=IDX, columns=TICKERS).where(available)
    sink.add_signal(name, values, available)


def test_only_raw_availability_counts_are_emitted() -> None:
    sink = ConditioningSink()
    all_available = pd.DataFrame(True, index=IDX, columns=TICKERS)
    bull_last_missing = all_available.copy()
    bull_last_missing.loc[IDX[-1], "BBB"] = False
    bear_unavailable = all_available.copy()
    bear_unavailable.loc[IDX[-1], "BBB"] = False

    bull_names = list(BULLISH_INPUTS.values())
    for name in bull_names[:-1]:
        _register(sink, name, all_available)
    _register(sink, bull_names[-1], bull_last_missing)
    for name in BEARISH_INPUTS.values():
        _register(sink, name, bear_unavailable)

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    emitted = {column for column in panel if column.startswith("f_")}
    day = panel.loc[panel["date"].eq(IDX[-1])].set_index("ticker")

    assert EMISSION == {
        "ic_xs_bullish_available_family_count": "raw",
        "ic_xs_bearish_available_family_count": "raw",
    }
    assert emitted == {f"f_{name}" for name in EMISSION}
    assert day.loc["AAA", "f_ic_xs_bullish_available_family_count"] == 4.0
    assert day.loc["BBB", "f_ic_xs_bullish_available_family_count"] == 3.0
    assert day.loc["AAA", "f_ic_xs_bearish_available_family_count"] == 3.0
    assert day.loc["BBB", "f_ic_xs_bearish_available_family_count"] == 0.0
    assert not any(column.endswith(("_xs", "_vs_peers")) for column in emitted)
    print("\n=== SANITY CHECK: cross-source output is support, not ranked alpha ===")
    print("  Exactly two raw availability counts remain; ranked ratios, conflict, and actor-count alpha are absent. Validated.")


def test_unresolved_families_still_report_observable_support() -> None:
    sink = ConditioningSink()
    available = pd.DataFrame(True, index=IDX, columns=TICKERS)
    _register(sink, next(iter(BULLISH_INPUTS.values())), available)
    for name in BEARISH_INPUTS.values():
        _register(sink, name, available)

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)

    assert panel["f_ic_xs_bullish_available_family_count"].eq(1.0).all()
    assert panel["f_ic_xs_bearish_available_family_count"].eq(3.0).all()
    print("\n=== SANITY CHECK: support survives partial family resolution ===")
    print("  One resolved bullish source reports support=1 while all three bearish sources report support=3. Validated.")


def test_a_family_changes_support_only_when_it_becomes_available() -> None:
    sink = ConditioningSink()
    available = pd.DataFrame(True, index=IDX, columns=TICKERS)
    names = list(BULLISH_INPUTS.values())
    for name in names[:3]:
        _register(sink, name, available)
    late = pd.DataFrame(False, index=IDX, columns=TICKERS)
    late.loc[IDX[-1], :] = True
    _register(sink, names[3], late)

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    support = panel.pivot(index="date", columns="ticker", values="f_ic_xs_bullish_available_family_count")

    assert support.loc[IDX[-2]].eq(3.0).all()
    assert support.loc[IDX[-1]].eq(4.0).all()
    print("\n=== SANITY CHECK: source support is point-in-time ===")
    print("  The fourth family changes support from 3 to 4 only on its first available date. Validated.")


def test_sink_rejects_available_but_unexplained_missing_values() -> None:
    values = pd.DataFrame(1.0, index=IDX, columns=TICKERS)
    values.loc[IDX[-1], "BBB"] = np.nan
    available = pd.DataFrame(True, index=IDX, columns=TICKERS)

    with np.testing.assert_raises_regex(ValueError, "available-but-unexplained"):
        ConditioningSink().add_signal(next(iter(BULLISH_INPUTS.values())), values, available)
    print("\n=== SANITY CHECK: construction holes cannot shrink support ===")
    print("  An available cell containing NaN raises at the sink contract. Validated.")
