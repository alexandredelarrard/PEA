"""Multi-class volume denominators (Q2e', plan addendum §13.1).

`ic_ftd_to_adv20` and `ic_shortvol_market_coverage` divide a class-summed numerator (fails and RegSHO
volume in reference shares, from extraction) by tape volume. The tape side now sums the issuer's
secondary classes too, each on the master's `secondary_class` dates and at its conversion ratio.
Known-truth fixtures: two classes at ratio 1, a BRK-style ratio step at a one-class split, a missing
class bar, a class bar outside its master interval, and a single-class issuer that must not move.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import DEFAULT_CONFIG_DIR
from src.data_aggregate.transformers import step_cube_institutionals as step_module
from src.data_aggregate.utils.common.level_basis import load_bugfix
from src.data_aggregate.utils.institutionals.short_flow_features import (
    _fails_fields,
    _shortvol_fields,
    build_short_flow_feature_panel,
    secondary_class_volume,
    with_class_volume,
)
from src.data_store.schema import Tables
from tests.conftest import make_frames

OPEN = pd.NaT


def _lines(*rows: tuple[str, str, float, str, object]) -> pd.DataFrame:
    """`security_master` secondary-class rows: (company, market symbol, ratio, valid_from, valid_to)."""
    return pd.DataFrame(rows, columns=["canonical_company", "market_symbol", "conversion_ratio", "valid_from", "valid_to"])


def _bars(symbol: str, days: pd.DatetimeIndex, volume: float | list[float]) -> pd.DataFrame:
    return pd.DataFrame({"ticker": symbol, "date": days, "volume": volume})


def _cell(frame: pd.DataFrame, day: str, column: str) -> float:
    return float(cast(Any, frame.loc[pd.Timestamp(day), column]))


def test_two_classes_at_ratio_one_sum_into_both_denominators() -> None:
    """LEN trades 1,000 a day and LEN-B 500: fails of 300 are 20% of ADV, not 30%; the single-class control is unchanged."""
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-14"), name="date")
    volume = pd.DataFrame({"LEN": 1_000.0, "MRK": 1_000.0}, index=idx)
    fails = pd.DataFrame([{"date": "2024-01-02", "ticker": t, "fails_quantity": 300.0, "period": "202401a"} for t in ("LEN", "MRK")])
    regsho = pd.DataFrame([{"date": d, "ticker": t, "short_volume": 100.0, "total_volume": 300.0} for d in idx for t in ("LEN", "MRK")])
    lines = _lines(("LEN", "LEN-B", 1.0, "1900-01-01", OPEN))
    class_vol, missing = secondary_class_volume(lines, _bars("LEN-B", idx, 500.0), idx)
    peers = {"LEN": {"MRK": 1.0}, "MRK": {"LEN": 1.0}}
    frames = make_frames(idx, peers, volume=volume)

    old = build_short_flow_feature_panel(frames, regsho, fails_history=fails).set_index(["date", "ticker"]).sort_index()
    new = build_short_flow_feature_panel(frames, regsho, fails_history=fails, class_volume=class_vol).set_index(["date", "ticker"]).sort_index()

    day = (pd.Timestamp("2024-01-30"), "LEN")
    assert old.loc[day, "f_ic_ftd_to_adv20"] == pytest.approx(0.3)
    assert new.loc[day, "f_ic_ftd_to_adv20"] == pytest.approx(0.2)
    assert new.loc[day, "f_ic_shortvol_market_coverage"] == pytest.approx(0.2)
    assert old.loc[day, "f_ic_shortvol_market_coverage"] == pytest.approx(0.3)
    assert not missing.to_numpy().any()
    pd.testing.assert_frame_equal(new.xs("MRK", level="ticker"), old.xs("MRK", level="ticker"))
    moved = [c for c in new.columns if not new[c].equals(old[c])]
    assert sorted(moved) == ["f_ic_ftd_to_adv20", "f_ic_shortvol_market_coverage"]
    print("\n=== SANITY CHECK: two classes at ratio 1 ===")
    print(
        f"  LEN fails/ADV {old.loc[day, 'f_ic_ftd_to_adv20']:.2f} -> {new.loc[day, 'f_ic_ftd_to_adv20']:.2f} and coverage "
        f"{old.loc[day, 'f_ic_shortvol_market_coverage']:.2f} -> {new.loc[day, 'f_ic_shortvol_market_coverage']:.2f} "
        f"(LEN-B's 500 a day joins LEN's 1,000); only those two columns moved; MRK identical. Validated."
    )


def test_brk_style_ratio_step_matches_the_as_traded_identity_across_a_one_class_split() -> None:
    """BRK-B splits 50:1 on 2010-01-21 and BRK-A does not, so A's ratio steps 30 -> 1,500 there.

    Yahoo restates BRK-B's earlier volume x50 and leaves BRK-A's alone, so on the restated tape one A share is
    1,500 B shares on both sides of the step. The as-traded truth: fails 160 against 2,000 B + 30 x 10 A before
    (6.96%), fails 2,500 against 100,000 B + 1,500 x 10 A after (2.17%); coverage 50% on both sides.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2009-12-01", "2010-03-31"), name="date")
    split = pd.Timestamp("2010-01-21")
    b_raw = np.where(idx < split, 2_000.0, 100_000.0)
    volume = pd.DataFrame({"BRK-B": np.where(idx < split, b_raw * 50, b_raw)}, index=idx)
    splits = pd.DataFrame([{"date": split, "ticker": "BRK-B", "ratio": 50.0}])
    lines = _lines(("BRK-B", "BRK-A", 30.0, "1900-01-01", "2010-01-21"), ("BRK-B", "BRK-A", 1500.0, "2010-01-21", OPEN))
    class_vol, missing = secondary_class_volume(lines, _bars("BRK-A", idx, 10.0), idx)
    tape = with_class_volume(volume, class_vol)

    assert _cell(tape, "2010-01-20", "BRK-B") == pytest.approx(50 * (2_000 + 30 * 10))
    assert _cell(tape, "2010-01-21", "BRK-B") == pytest.approx(100_000 + 1_500 * 10)
    fails_days = [d for d in idx if d <= pd.Timestamp("2009-12-14")] + [
        d for d in idx if pd.Timestamp("2010-02-01") <= d <= pd.Timestamp("2010-02-12")
    ]
    fails = pd.DataFrame(
        [
            {"date": d, "ticker": "BRK-B", "fails_quantity": 100 + 30 * 2 if d < split else 1_000 + 1_500 * 1, "period": f"{d:%Y%m}a"}
            for d in fails_days
        ]
    )
    ratio = _fails_fields(fails, idx, None, tape, splits)["ic_ftd_to_adv20"]
    regsho = pd.DataFrame(
        [{"date": d, "ticker": "BRK-B", "short_volume": 1.0, "total_volume": 1_000 + 30 * 5 if d < split else 50_000 + 1_500 * 5} for d in idx]
    )
    cov = _shortvol_fields(regsho, idx, None, None, tape, splits)["ic_shortvol_market_coverage"]

    assert _cell(ratio, "2009-12-30", "BRK-B") == pytest.approx(160 / 2_300)
    assert _cell(ratio, "2010-03-02", "BRK-B") == pytest.approx(2_500 / 115_000)
    assert cov["BRK-B"].dropna().round(12).eq(0.5).all() and cov["BRK-B"].notna().sum() > 50
    assert not missing.to_numpy().any()
    print("\n=== SANITY CHECK: BRK ratio step on the restated tape ===")
    print(
        f"  fails/ADV {_cell(ratio, '2009-12-30', 'BRK-B'):.4f} before (as-traded 160/2,300) and {_cell(ratio, '2010-03-02', 'BRK-B'):.4f} after "
        f"(2,500/115,000); coverage 0.50 on every day across the 50:1 one-class split. Validated."
    )


def test_a_missing_class_bar_contributes_nothing_and_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """LEN-B has no bar on 3 days and the delisted DISCK none at all: no volume from them, each company-day flagged and logged."""
    idx = pd.DatetimeIndex(pd.bdate_range("2022-03-01", "2022-04-29"), name="date")
    gaps = idx[[5, 6, 30]]
    lines = _lines(("LEN", "LEN-B", 1.0, "1900-01-01", OPEN), ("WBD", "DISCK", 1.0, "1900-01-01", "2022-04-11"))
    with caplog.at_level(logging.WARNING):
        class_vol, missing = secondary_class_volume(lines, _bars("LEN-B", idx.difference(gaps), 500.0), idx)

    assert class_vol.loc[gaps, "LEN"].eq(0.0).all() and class_vol.loc[idx.difference(gaps), "LEN"].eq(500.0).all()
    assert list(missing.index[missing["LEN"]]) == list(gaps)
    disck_days = idx[idx < pd.Timestamp("2022-04-11")]
    assert list(missing.index[missing["WBD"]]) == list(disck_days) and class_vol["WBD"].eq(0.0).all()
    text = caplog.text
    assert "LEN-B" in text and "DISCK" in text and f"{len(gaps) + len(disck_days)}" in text
    print("\n=== SANITY CHECK: missing class bars ===")
    print(f"  LEN-B's 3 bar-less days and DISCK's {len(disck_days)} active days add no volume, are flagged per company-day and logged. Validated.")


def test_a_class_bar_outside_its_master_interval_is_ignored_and_a_seam_day_counts_once() -> None:
    """Yahoo's GOOG before 2014-04-03 is the old single Google class: those bars must not enter GOOGL's tape.

    Two GOOG securities share the 2015-10-01 seam day in the master: the one Yahoo bar is counted once.
    """
    idx = pd.DatetimeIndex(pd.bdate_range("2014-03-20", "2015-10-15"), name="date")
    start = pd.Timestamp("2014-04-03")
    bars = _bars("GOOG", idx, np.where(idx < start, 1e9, 700.0).tolist())
    lines = _lines(("GOOGL", "GOOG", 1.0, "2014-04-03", "2015-10-02"), ("GOOGL", "GOOG", 1.0, "2015-10-01", OPEN))
    class_vol, missing = secondary_class_volume(lines, bars, idx)

    assert class_vol.loc[idx < start, "GOOGL"].eq(0.0).all()
    assert class_vol.loc[idx >= start, "GOOGL"].eq(700.0).all()
    assert _cell(class_vol, "2015-10-01", "GOOGL") == 700.0
    assert not missing.to_numpy().any()
    print("\n=== SANITY CHECK: class bars joined to master dates ===")
    print("  the pre-2014-04-03 GOOG bars (old Google, 1e9 a day) add nothing; the 2015-10-01 seam day adds 700 once. Validated.")


def test_a_single_class_issuer_is_bit_identical() -> None:
    """No class lines, or class lines of other companies only, leave every value of a single-class issuer unchanged."""
    idx = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=300), name="date")
    rng = np.random.default_rng(7)
    tickers = ["AAA", "BBB", "CCC"]
    volume = pd.DataFrame({t: rng.uniform(1e6, 3e6, len(idx)) for t in tickers}, index=idx)
    close = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0, 0.02, len(idx))) for t in tickers}, index=idx)
    regsho = pd.DataFrame([{"date": d, "ticker": t, "short_volume": 4e5, "total_volume": rng.uniform(5e5, 9e5)} for d in idx for t in tickers])
    keep = rng.random(len(idx) * len(tickers)) < 0.3
    fails = pd.DataFrame(
        {
            "date": np.repeat(idx.to_numpy(), len(tickers))[keep],
            "ticker": np.tile(np.array(tickers), len(idx))[keep],
            "fails_quantity": rng.lognormal(8, 1.2, int(keep.sum())).round(0),
        }
    )
    fails["period"] = pd.to_datetime(fails["date"]).map(lambda day: f"{day:%Y%m}{'a' if day.day <= 15 else 'b'}")
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}
    frames = make_frames(idx, peers, volume=volume, close_total=close)
    empty_vol, _ = secondary_class_volume(_lines(), _bars("X", idx[:0], []), idx)
    other_vol, _ = secondary_class_volume(_lines(("CCC", "CCC-B", 1.0, "1900-01-01", OPEN)), _bars("CCC-B", idx, 1e6), idx)

    base = build_short_flow_feature_panel(frames, regsho, fails_history=fails)
    none_lines = build_short_flow_feature_panel(frames, regsho, fails_history=fails, class_volume=empty_vol)
    others = build_short_flow_feature_panel(frames, regsho, fails_history=fails, class_volume=other_vol)

    pd.testing.assert_frame_equal(none_lines, base)
    single = base["ticker"].isin(["AAA", "BBB"])
    pd.testing.assert_frame_equal(others[others["ticker"].isin(["AAA", "BBB"])], base[single])
    assert with_class_volume(volume, None) is volume
    print("\n=== SANITY CHECK: single-class issuers ===")
    print("  empty class lines leave the whole panel bit-identical; a class of CCC leaves AAA and BBB bit-identical. Validated.")


def test_loader_reads_universe_secondary_classes_and_only_their_bars(sqlite_store: Any) -> None:
    """The class read takes `secondary_class` lines of universe companies and reads `prices` by exactly their symbols."""
    master = pd.DataFrame(
        {
            "security_id": ["C1", "C2", "C3", "C4"],
            "canonical_company": ["LEN", "LEN", "LEN", "XYZ"],
            "source": ["ftd"] * 4,
            "source_symbol": ["LENB", "LEN", "LENPRA", "XYZB"],
            "market_symbol": ["LEN-B", "LEN", "LENPRA", "XYZ-B"],
            "security_class": ["class_B", "class_A", "preferred", "class_B"],
            "conversion_ratio": [1.0, 1.0, 1.0, 1.0],
            "lineage_role": ["secondary_class", "canonical_current", "excluded", "secondary_class"],
            "valid_from": ["2009-06-26"] * 4,
            "valid_to": [None] * 4,
            "n_observations": [10, 10, 10, 10],
        }
    )
    sqlite_store.save(Tables.security_master, master)
    days = pd.bdate_range("2024-01-02", periods=3)
    sqlite_store.save(
        Tables.prices,
        pd.concat([_bars(s, days, 1.0).assign(close_split=1.0, close_total=1.0) for s in ("LEN", "MRK", "LEN-B", "XYZ-B", "BRK-A")]),
    )

    loaded = step_module.institutional_inputs.load_secondary_classes(sqlite_store, logging.getLogger(__name__), ["LEN", "MRK"])

    assert loaded is not None
    lines, bars = loaded
    assert lines["market_symbol"].tolist() == ["LEN-B"]
    assert sorted(set(bars["ticker"])) == ["LEN-B"] and len(bars) == 3
    assert step_module.institutional_inputs.load_secondary_classes(sqlite_store, logging.getLogger(__name__), ["MRK"]) is None
    print("\n=== SANITY CHECK: class loader ===")
    print("  of 4 master lines it kept LEN's secondary class only; of 5 symbols in prices it read LEN-B's 3 bars only. Validated.")


def test_loader_applies_the_registered_brk_a_volume_unit_repair(sqlite_store: Any) -> None:
    """P30: with the shipped price register, BRK-A's Yahoo volume before 2013-07-29 is divided by 100 on the class read;
    the boundary day and later bars are untouched, and without the register nothing changes."""
    master = pd.DataFrame(
        {
            "security_id": ["B1", "A1"],
            "canonical_company": ["BRK-B", "BRK-B"],
            "source": ["ftd", "ftd"],
            "source_symbol": ["BRKB", "BRKA"],
            "market_symbol": ["BRK-B", "BRK-A"],
            "security_class": ["class_B", "class_A"],
            "conversion_ratio": [1.0, 1500.0],
            "lineage_role": ["canonical_current", "secondary_class"],
            "valid_from": ["2009-06-26"] * 2,
            "valid_to": [None] * 2,
            "n_observations": [10, 10],
        }
    )
    sqlite_store.save(Tables.security_master, master)
    days = pd.bdate_range("2013-06-03", "2013-08-30")
    volume = [24_800.0 if day < pd.Timestamp("2013-07-29") else 300.0 for day in days]
    sqlite_store.save(Tables.prices, _bars("BRK-A", days, volume).assign(close_split=170_000.0, close_total=170_000.0))
    log = logging.getLogger(__name__)

    loaded = step_module.institutional_inputs.load_secondary_classes(sqlite_store, log, ["BRK-B"], bugfix=load_bugfix(DEFAULT_CONFIG_DIR))
    untouched = step_module.institutional_inputs.load_secondary_classes(sqlite_store, log, ["BRK-B"])

    assert loaded is not None and untouched is not None
    fixed = loaded[1].assign(date=lambda f: pd.to_datetime(f["date"])).set_index("date")["volume"]
    raw = untouched[1].assign(date=lambda f: pd.to_datetime(f["date"])).set_index("date")["volume"]
    assert fixed.loc["2013-07-26"] == pytest.approx(248.0) and fixed.loc[:"2013-07-26"].eq(248.0).all()
    assert fixed.loc["2013-07-29":].eq(300.0).all()
    assert raw.loc["2013-07-26"] == 24_800.0
    print("\n=== SANITY CHECK: BRK-A volume unit repair on the class read ===")
    print("  2013-07-26 24,800 -> 248 with the shipped register; 2013-07-29 on stays 300; no register -> raw 24,800. Validated.")


def test_loader_without_a_master_returns_none(sqlite_store: Any) -> None:
    assert step_module.institutional_inputs.load_secondary_classes(sqlite_store, logging.getLogger(__name__), ["LEN"]) is None
    print("\n=== SANITY CHECK: no master ===")
    print("  before `security_master` exists the class read returns None, so the features keep the canonical tape. Validated.")
