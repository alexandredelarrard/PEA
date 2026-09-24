"""Cross-source agreement (`ic_xs_*`, registry section 8).

The two rules the registry is explicit about are the two that are easy to get wrong, so each
has its own test: ten filings from one actor are ONE opinion (#84), and the absence of buying
is not a short signal (#85). Plus the conflict flag's magnitude, the refusal to build a
"family count" out of one family, and declared-vs-emitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.cross_source_features import (
    ACTOR_WINDOW,
    EMISSION,
    PERCENTILE,
    build_cross_source_panel,
)
from src.data_aggregate.utils.institutionals.sink import (
    BEARISH_INPUTS,
    BULLISH_INPUTS,
    NEGATED_INPUTS,
    ConditioningSink,
)
from tests.conftest import make_frames

IDX = pd.DatetimeIndex(pd.bdate_range("2022-01-03", periods=300))
TICKERS = [f"T{i}" for i in range(10)]
PEERS = {t: {p: 1.0 for p in TICKERS if p != t} for t in TICKERS}


def _ramp(top: list[str]) -> pd.DataFrame:
    """A wide frame where the named tickers alone sit above the flag percentile.

    ⚠ THE NON-TOP TICKERS ARE ALL EQUAL, deliberately. A 0..9 ramp over ten names puts the
    SECOND-highest at the 0.9 percentile, which is also above the 0.8 threshold, so every
    count came out one too high and the test was measuring the fixture. Nine tied values rank
    at 0.5 (`rank` averages ties), which is unambiguously below the flag.
    """
    base = pd.DataFrame({t: 1.0 for t in TICKERS}, index=IDX, dtype="float64")
    for t in top:
        base[t] = 100.0
    return base


def _bearish_signal(role: str, top: list[str]) -> pd.DataFrame:
    """The bearish input for `role` with `top` flagging.

    ⚠ `activist_reduction` IS NEGATED BY THE BUILDER -- a stake REDUCTION is a negative change
    in percent-of-class -- so its flagging ticker must carry the most negative value, not the
    largest. The first version of this fixture ignored that and read as a missing family.
    """
    frame = _ramp(top)
    return -frame if role in {r for r, n in BEARISH_INPUTS.items() if r in NEGATED_INPUTS} else frame


def _register(
    sink: ConditioningSink,
    name: str,
    frame: pd.DataFrame,
    available: pd.DataFrame | None = None,
) -> None:
    mask = available if available is not None else pd.DataFrame(True, index=frame.index, columns=frame.columns)
    sink.add_signal(name, frame.where(mask), mask)


def test_family_ratios_use_the_per_cell_available_denominator():
    sink = ConditioningSink()
    roles = list(BULLISH_INPUTS)
    for role in roles[:3]:  # T0 tops three of the four bullish families
        _register(sink, BULLISH_INPUTS[role], _ramp(["T0"]))
    _register(sink, BULLISH_INPUTS[roles[3]], _ramp(["T9"]))
    for role, name in BEARISH_INPUTS.items():
        _register(sink, name, _bearish_signal(role, ["T9"]))

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    day = panel[panel["date"] == IDX[-1]].set_index("ticker")
    assert day.loc["T0", "f_ic_xs_bullish_family_ratio"] == 3.0 / 4.0
    assert day.loc["T9", "f_ic_xs_bullish_family_ratio"] == 1.0 / 4.0
    assert day.loc["T0", "f_ic_xs_bearish_family_ratio"] == 0.0
    assert day.loc["T9", "f_ic_xs_bearish_family_ratio"] == 1.0
    assert day.loc["T0", "f_ic_xs_bullish_available_family_count"] == 4.0
    assert day.loc["T0", "f_ic_xs_bearish_available_family_count"] == 3.0
    ratios = panel["f_ic_xs_bullish_family_ratio"].dropna()
    assert ratios.between(0, 1).all()
    print("\n=== SANITY CHECK: availability-normalized family ratios ===")
    print(
        f"  T0 tops 3/4 bullish families -> 0.75; T9 tops all 3 bearish families -> 1.00. "
        f"The {PERCENTILE:.0%} vote threshold and both denominators reconcile. Validated."
    )


def test_absence_of_evidence_is_never_bearish_and_a_count_of_nothing_is_nan():
    """#85 (report section 3.6.2). A ticker no bearish family has an opinion about must read
    NaN, not 0 -- and certainly not a positive count. 'Nobody sold' is not a sale."""
    sink = ConditioningSink()
    for name in BULLISH_INPUTS.values():
        _register(sink, name, _ramp(["T0"]))
    for role, name in BEARISH_INPUTS.items():
        frame = _bearish_signal(role, ["T9"])
        available = pd.DataFrame(True, index=frame.index, columns=frame.columns)
        available.loc[:, "T5"] = False
        _register(sink, name, frame, available)

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    day = panel[panel["date"] == IDX[-1]].set_index("ticker")
    assert np.isnan(day.loc["T5", "f_ic_xs_bearish_family_ratio"]), "a ticker with fewer than three available families read as a ratio"
    assert day.loc["T5", "f_ic_xs_bearish_available_family_count"] == 0.0
    assert day.loc["T1", "f_ic_xs_bearish_family_ratio"] == 0.0, "a ticker with inputs but no flag must read a real 0"
    assert np.isnan(day.loc["T5", "f_ic_xs_conflict_ratio"])
    print("\n=== SANITY CHECK: absence of evidence vs a measured zero ===")
    print(
        "  T5 has denominator 0 and ratio NaN; T1 has all 3 families available and a measured "
        "0.0 vote. Conflict inherits unavailable evidence. Validated."
    )


def test_distinct_actors_not_rows_and_the_window_releases():
    """#84 (report section 3.1.6). Ten Form 4s from ONE officer are one vote; three officers
    are three. And the count releases once the window passes, so it is a trailing distinct
    count rather than an ever-growing one."""
    sink = ConditioningSink()
    for name in BULLISH_INPUTS.values():
        _register(sink, name, _ramp(["T0"]))
    rows = [{"ticker": "T0", "date": IDX[10 + i], "actor": "OWNER-1"} for i in range(10)]
    rows += [{"ticker": "T1", "date": IDX[10], "actor": f"OWNER-{k}"} for k in (1, 2, 3)]
    sink.add_actors("insider", pd.DataFrame(rows))
    # the same actor id in ANOTHER family is a different actor -- an owner CIK and a manager
    # CIK are both 10 digits, so the family namespaces the id
    sink.add_actors("super", pd.DataFrame([{"ticker": "T1", "date": IDX[10], "actor": "OWNER-1"}]))

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    col = "f_ic_xs_bullish_actor_count"
    at = panel[panel["date"] == IDX[25]].set_index("ticker")[col]
    assert at["T0"] == 1.0, f"ten filings from one owner counted as {at['T0']}"
    assert at["T1"] == 4.0, "three owners + one manager should be four distinct actors"
    # NaN before the ticker's first act, never 0
    before = panel[panel["date"] == IDX[5]].set_index("ticker")[col]
    assert np.isnan(before["T0"]) and np.isnan(before["T1"])
    # and released after the window
    after = panel[panel["date"] == IDX[10 + ACTOR_WINDOW + 5]].set_index("ticker")[col]
    assert after["T1"] == 0.0, "the trailing window never released"
    print("\n=== SANITY CHECK: distinct ACTORS, not rows ===")
    print(
        f"  T0: 10 Form 4s from one owner -> {at['T0']:.0f}; T1: 3 owners + 1 manager "
        f"-> {at['T1']:.0f} (the family namespaces a repeated id); NaN before the first act; "
        f"back to 0 once the {ACTOR_WINDOW}-day window passes. Validated."
    )


def test_the_conflict_flag_is_the_weaker_side():
    sink = ConditioningSink()
    roles_b, roles_s = list(BULLISH_INPUTS), list(BEARISH_INPUTS)
    for role in roles_b[:3]:
        _register(sink, BULLISH_INPUTS[role], _ramp(["T0"]))
    _register(sink, BULLISH_INPUTS[roles_b[3]], _ramp(["T9"]))
    for role in roles_s[:2]:
        _register(sink, BEARISH_INPUTS[role], _bearish_signal(role, ["T0"]))
    for role in roles_s[2:]:
        _register(sink, BEARISH_INPUTS[role], _bearish_signal(role, ["T9"]))

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    day = panel[panel["date"] == IDX[-1]].set_index("ticker")
    # T0: 3 bullish families and 2 bearish -> a conflict of strength 2, not 3
    assert day.loc["T0", "f_ic_xs_bullish_family_ratio"] == 3.0 / 4.0
    assert day.loc["T0", "f_ic_xs_bearish_family_ratio"] == 2.0 / 3.0
    assert day.loc["T0", "f_ic_xs_conflict_ratio"] == 2.0 / 3.0
    assert day.loc["T9", "f_ic_xs_conflict_ratio"] == 1.0 / 4.0
    print("\n=== SANITY CHECK: continuous conflict ratio ===")
    print("  Conflict is exactly the weaker normalized side: min(3/4, 2/3) = 2/3. Validated.")


def test_it_persists_the_denominator_but_withholds_a_ratio_below_three_families():
    """A rename that the sink's declaration does not follow would silently leave a 'family
    count' measuring one family. One resolved input is not a count of families, and the
    builder drops the column rather than shipping a misnamed one."""
    sink = ConditioningSink()
    _register(sink, BULLISH_INPUTS[list(BULLISH_INPUTS)[0]], _ramp(["T0"]))
    for role, name in BEARISH_INPUTS.items():
        _register(sink, name, _bearish_signal(role, ["T9"]))
    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    assert "f_ic_xs_bullish_family_ratio" not in panel.columns
    assert "f_ic_xs_conflict_ratio" not in panel.columns
    assert panel["f_ic_xs_bullish_available_family_count"].eq(1.0).all()
    assert "f_ic_xs_bearish_family_ratio" in panel.columns
    print("\n=== SANITY CHECK: three-family minimum ===")
    print("  One resolved bullish family persists denominator=1 for audit, while its ratio and conflict are withheld. Validated.")


def test_declared_vs_emitted():
    sink = ConditioningSink()
    for name in BULLISH_INPUTS.values():
        _register(sink, name, _ramp(["T0"]))
    for role, name in BEARISH_INPUTS.items():
        _register(sink, name, _bearish_signal(role, ["T9"]))
    sink.add_actors("insider", pd.DataFrame([{"ticker": t, "date": IDX[20], "actor": "O1"} for t in TICKERS]))
    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    expected = set()
    for name, mode in EMISSION.items():
        expected.add(f"f_{name}")
        if mode == "raw+xs":
            expected.add(f"f_{name}_xs")
    emitted = {c for c in panel.columns if c.startswith("f_")}
    assert emitted == expected, f"missing {sorted(expected - emitted)}; " f"undeclared {sorted(emitted - expected)}"
    for col in sorted(emitted):
        assert panel[col].notna().any(), f"{col} is declared, emitted and ALL-NaN"
    print("\n=== SANITY CHECK: ic_xs_* declared vs emitted ===")
    print(f"  {len(EMISSION)} features -> {len(emitted)} legs, exact match, all non-empty. " "Validated.")


def test_an_unavailable_family_does_not_change_the_ratio_until_it_becomes_available():
    sink = ConditioningSink()
    names = list(BULLISH_INPUTS.values())
    for name in names[:3]:
        _register(sink, name, _ramp(["T0"]))
    fourth = _ramp(["T9"])
    available = pd.DataFrame(False, index=IDX, columns=TICKERS)
    available.loc[IDX[-1], :] = True
    _register(sink, names[3], fourth, available)

    panel = build_cross_source_panel(make_frames(IDX, PEERS, universe=pd.Index(TICKERS)), sink)
    earlier = panel[panel["date"] == IDX[-2]].set_index("ticker")
    later = panel[panel["date"] == IDX[-1]].set_index("ticker")
    assert earlier.loc["T0", "f_ic_xs_bullish_family_ratio"] == 1.0
    assert earlier.loc["T0", "f_ic_xs_bullish_available_family_count"] == 3.0
    assert later.loc["T0", "f_ic_xs_bullish_family_ratio"] == 3.0 / 4.0
    assert later.loc["T0", "f_ic_xs_bullish_available_family_count"] == 4.0
    print("\n=== SANITY CHECK: a new family changes numerator and denominator together ===")
    print("  T0 stays 3/3 while family four is unavailable, then becomes 3/4 on its first available date. Validated.")


def test_sink_rejects_available_but_unexplained_missing_values():
    frame = _ramp(["T0"])
    frame.loc[IDX[-1], "T5"] = np.nan
    available = pd.DataFrame(True, index=IDX, columns=TICKERS)
    with np.testing.assert_raises_regex(ValueError, "available-but-unexplained"):
        ConditioningSink().add_signal(next(iter(BULLISH_INPUTS.values())), frame, available)
    print("\n=== SANITY CHECK: construction holes cannot shrink denominators ===")
    print("  An available cell containing NaN raises at the sink contract. Validated.")
