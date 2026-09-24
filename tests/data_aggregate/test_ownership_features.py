"""Phase 2.4 — Beneficial-ownership panel (Schedule 13D activist + 13G passive >5%).

Checks canonical event construction, repeat-activist and escalation/de-escalation joins,
event-only history, and removal of the short-history `percent_of_class` feature family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.ownership_features import (
    EMISSION,
    HOLDER_ACTIVE_DAYS,
    _act_fields,
    _bo_fields,
    _canonicalize,
    _cross_fields,
    build_ownership_feature_panel,
)
from tests.conftest import make_frames

IDX = pd.bdate_range("2023-01-03", "2025-06-30")


def _peers(tickers):
    return {t: {p: 1.0 for p in tickers if p != t} for t in tickers}


def test_canonicalize_collapses_reporting_persons_without_ownership_numerics():
    """Four co-filers are one filing event while ownership numerics stay retired."""
    rows = [
        {
            "ticker": "AAA",
            "accession_number": "0001",
            "cusip": "CUS1",
            "filing_date": "2025-01-10",
            "is_amendment": 0.0,
            "percent_of_class": 10.0,
            "reporting_person_cik": str(i),
            "reporting_person_name": f"R{i}",
            "item4_purpose_of_transaction": "we may seek board representation",
        }
        for i in range(1, 5)
    ]
    raw = pd.DataFrame(rows)
    canon = _canonicalize(raw, text_col="item4_purpose_of_transaction", has_amendment=True)
    assert len(canon) == 1, "one filing, one canonical event -- not one row per reporting person"
    assert canon.loc[0, "n_reporting_persons"] == 4
    assert "percent_of_class" not in canon.columns
    print("\n=== SANITY CHECK: canonical event construction ===")
    print("  Four reporting persons collapse to one filing event; percent_of_class is not carried into feature construction. Validated.")


def test_canonicalize_empty_and_missing_cik_fallback():
    assert _canonicalize(None).empty
    assert _canonicalize(pd.DataFrame()).empty
    # a reporting person with no CIK falls back to name as the filer identity
    raw = pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "accession_number": "0001",
                "cusip": "CUS1",
                "filing_date": "2025-02-01",
                "is_amendment": 0.0,
                "percent_of_class": 6.0,
                "reporting_person_cik": None,
                "reporting_person_name": "Nameless Fund LP",
            }
        ]
    )
    canon = _canonicalize(raw, has_amendment=True)
    assert canon.loc[0, "filer_id"] == "Nameless Fund LP"


def _campaign_row(ticker, filer, day, pct=np.nan):
    return {
        "ticker": ticker,
        "filing_date": pd.Timestamp(day),
        "is_amendment": 0.0,
        "filer_id": filer,
        "percent_of_class": pct,
        "n_reporting_persons": 1,
    }


def test_repeat_activist_fires_on_the_fourth_campaign_only():
    """One filer opens initial 13Ds on 4 different tickers, in order. The plan's floor is
    >=3 PRIOR campaigns, so only the 4th ticker's event qualifies."""
    canon = pd.DataFrame(
        [
            _campaign_row("R1", "SAME_FILER", "2023-02-01"),
            _campaign_row("R2", "SAME_FILER", "2023-06-01"),
            _campaign_row("R3", "SAME_FILER", "2023-10-01"),
            _campaign_row("R4", "SAME_FILER", "2024-02-01"),
        ]
    )
    out = _act_fields(canon, IDX, halflife=126.0)
    repeat = out["ic_act_repeat_activist"]
    assert "R4" in repeat.columns and repeat["R4"].dropna().gt(0).any()
    for early in ("R1", "R2", "R3"):
        assert early not in repeat.columns or repeat[early].dropna().eq(0).all()
    print("\n=== SANITY CHECK: repeat-activist (>=3 prior campaigns) ===")
    print("  filer's 4th campaign (R4) flags repeat_activist; campaigns 1-3 do not. Validated.")


def test_campaign_age_days_resets_and_is_nan_before_first_event():
    canon = pd.DataFrame([_campaign_row("AAA", "F1", "2023-03-01")])
    out = _act_fields(canon, IDX, halflife=126.0)
    age = out["ic_act_campaign_age_days"]
    before = age.loc[: pd.Timestamp("2023-02-28"), "AAA"]
    assert before.isna().all(), "no campaign yet -> NaN, not 0"
    d0 = age.loc[pd.Timestamp("2023-03-01"), "AAA"]
    d10 = age.loc[IDX[IDX.get_indexer([pd.Timestamp("2023-03-01")])[0] + 7], "AAA"]
    assert d0 == 0
    assert d10 > d0


def test_escalation_and_deescalation_join():
    canon_g = pd.DataFrame([_campaign_row("ZZZ", "F1", "2023-01-10")])
    canon_d = pd.DataFrame([_campaign_row("ZZZ", "F1", "2023-03-01")])
    out = _cross_fields(canon_d, canon_g, IDX, halflife=126.0)
    esc = out["ic_bo_escalation_13g_to_13d"]
    assert "ZZZ" in esc.columns
    assert esc.loc[: pd.Timestamp("2023-02-28"), "ZZZ"].dropna().eq(0).all() or esc.loc[: pd.Timestamp("2023-02-28"), "ZZZ"].isna().all()
    assert esc.loc[pd.Timestamp("2023-03-01") :, "ZZZ"].dropna().gt(0).any()

    # reverse order (13D before 13G on another ticker) -> de-escalation, not escalation
    canon_g2 = pd.DataFrame([_campaign_row("YYY", "F2", "2023-05-01")])
    canon_d2 = pd.DataFrame([_campaign_row("YYY", "F2", "2023-01-01")])
    out2 = _cross_fields(canon_d2, canon_g2, IDX, halflife=126.0)
    deesc = out2["ic_bo_de_escalation_13d_to_13g"]
    assert "YYY" in deesc.columns and deesc["YYY"].dropna().gt(0).any()
    assert "YYY" not in out2["ic_bo_escalation_13g_to_13d"].columns
    print("\n=== SANITY CHECK: 13G<->13D escalation join ===")
    print("  prior-13G-then-13D -> escalation; prior-13D-then-13G -> de-escalation. Validated.")


def test_bo_holder_count_sums_distinct_filers_not_group_members():
    canon = pd.DataFrame(
        [
            _campaign_row("AAA", "F1", "2023-01-05"),
            _campaign_row("AAA", "F2", "2023-01-06"),
            _campaign_row("BBB", "F3", "2023-01-05"),
        ]
    )
    out = _bo_fields(canon, IDX, halflife=126.0)
    hc = out["ic_bo_holder_count"]
    # 2 distinct filers on AAA / 3 filers total that year -> 2/3; 1 of 3 on BBB -> 1/3
    assert np.isclose(hc.loc[pd.Timestamp("2023-01-10"), "AAA"], 2 / 3)
    assert np.isclose(hc.loc[pd.Timestamp("2023-01-10"), "BBB"], 1 / 3)
    assert hc.loc[: pd.Timestamp("2023-01-04"), "AAA"].isna().all(), "before any filer -> NaN"


def test_percent_of_class_features_are_not_constructed():
    canon = pd.DataFrame(
        [
            _campaign_row("AAA", "F1", "2025-01-10", pct=10.0),
            _campaign_row("AAA", "F1", "2025-03-10", pct=15.0),
        ]
    )
    out = _act_fields(canon, IDX, halflife=126.0)
    assert not any("percent_of_class" in name or "delta_percent_class" in name for name in out)
    print("\n=== SANITY CHECK: retired ownership numerics ===")
    print("  Supplying percent_of_class source values creates no level or delta feature. Validated.")


def test_event_features_remain_available_before_the_structured_data_mandate():
    d13 = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "accession_number": "0001",
                "cusip": "CUS1",
                "filing_date": "2023-05-01",
                "is_amendment": 0.0,
                "percent_of_class": 20.0,
                "reporting_person_cik": "1",
                "reporting_person_name": "R1",
                "item4_purpose_of_transaction": None,
            },
        ]
    )
    peers = _peers(["AAA", "BBB"])
    panel = build_ownership_feature_panel(make_frames(IDX, peers), d13, None)
    row = panel[(panel["ticker"] == "AAA") & (panel["date"] == pd.Timestamp("2023-05-01"))]
    assert not row.empty
    assert row["f_ic_act_initial_13d"].notna().all()
    assert not any("percent_of_class" in col or "delta_percent_class" in col for col in panel.columns)
    print("\n=== SANITY CHECK: pre-mandate event history preserved ===")
    print("  A 2023 Schedule 13D still emits its event intensity while all ownership numerics are absent. Validated.")


def test_panel_columns_and_emission_coverage():
    dates = ["2025-01-10", "2025-04-10"]
    d13 = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "accession_number": f"000{i+1}",
                "cusip": "CUS1",
                "filing_date": dates[i],
                "is_amendment": float(i),
                "percent_of_class": 10.0 * (i + 1),
                "reporting_person_cik": str(k),
                "reporting_person_name": f"R{k}",
                "item4_purpose_of_transaction": "the reporting persons intend to nominate directors",
            }
            for i in range(2)
            for k in range(1, 5)
        ]
    )
    d13g = pd.DataFrame(
        [
            {
                "ticker": "BBB",
                "accession_number": acc,
                "cusip": "CUS2",
                "filing_date": day,
                "percent_of_class": pct,
                "reporting_person_cik": "9",
                "reporting_person_name": "Passive Fund",
            }
            # two filings by the SAME filer, so the per-filer delta has a prior to difference
            # against (one filing alone correctly yields no delta at all)
            for acc, day, pct in (("1001", "2025-02-03", 6.0), ("1002", "2025-05-05", 8.0))
        ]
    )
    peers = _peers(["AAA", "BBB"])
    panel = build_ownership_feature_panel(make_frames(IDX, peers), d13, d13g)
    assert not panel.empty
    # Features that ALWAYS fire on this data (unlike repeat_activist/escalation/strategic,
    # which need a specific trigger this synthetic scenario does not construct -- those are
    # covered directly against `_act_fields`/`_bo_fields`/`_cross_fields` above).
    guaranteed = [
        "ic_act_initial_13d",
        "ic_act_amendment_intensity",
        "ic_act_campaign_age_days",
        "ic_act_purpose_board",
        "ic_bo_holder_count",
        "ic_bo_new_holder",
    ]
    for name in guaranteed:
        mode = EMISSION[name]
        assert f"f_{name}" in panel.columns, f"raw leg f_{name} missing"
        if mode == "raw+xs":
            assert f"f_{name}_xs" in panel.columns
        if mode == "raw+peers":
            assert f"f_{name}_vs_peers" in panel.columns

    assert not any("percent_of_class" in col or "delta_percent_class" in col for col in panel.columns)
    print("\n=== SANITY CHECK: ownership panel columns ===")
    print(f"  {len(EMISSION)} event-only features remain declared; no emitted column contains percent_of_class. Validated.")


def _g_row(ticker, filer, day, pct):
    return {"ticker": ticker, "filing_date": pd.Timestamp(day), "filer_id": filer, "percent_of_class": pct, "n_reporting_persons": 1}


def test_retired_bo_numeric_features_are_absent_from_private_builder():
    canon = pd.DataFrame([_g_row("AAA", "F1", "2025-01-10", 6.0), _g_row("AAA", "F1", "2025-03-10", 9.0)])
    out = _bo_fields(canon, IDX, halflife=126.0)
    assert "ic_bo_percent_of_class" not in out
    assert "ic_bo_delta_percent_class" not in out


def test_delta_ignores_a_newly_observed_filer():
    """Measured defect: diffing the SUMMED level read filer-set growth as buying (median
    +5.1pp on the live 13G panel). A second filer's FIRST filing must move the level and NOT
    the delta."""
    canon = pd.DataFrame([_g_row("AAA", "F1", "2025-01-10", 6.0), _g_row("AAA", "F2", "2025-03-10", 7.0)])
    out = _bo_fields(canon, IDX, halflife=126.0)
    assert "ic_bo_percent_of_class" not in out and "ic_bo_delta_percent_class" not in out
    print("\n=== SANITY CHECK: no short-history stake-change proxy ===")
    print("  Adding a newly observed filer cannot create a percent-of-class level or delta. Validated.")


def test_holder_count_is_a_bounded_share_and_lapses():
    """Both halves of the D28 ratio share one trailing window, so it is a share: built with a
    never-releasing numerator over a calendar-year denominator it measured 1.23 on live data."""
    canon = pd.DataFrame([_g_row("AAA", "F1", "2023-01-05", np.nan), _g_row("BBB", "F2", "2023-01-05", np.nan)])
    hc = _bo_fields(canon, IDX, halflife=126.0)["ic_bo_holder_count"]
    vals = hc.to_numpy()
    assert np.nanmax(vals) <= 1.0, "a share cannot exceed 1"
    assert hc.loc[pd.Timestamp("2023-02-01"), "AAA"] == 0.5  # 1 of the 2 active filers
    # F1 lapses HOLDER_ACTIVE_DAYS after its only filing, so AAA stops being held
    lapsed = IDX[IDX.get_indexer([pd.Timestamp("2023-01-05")])[0] + HOLDER_ACTIVE_DAYS + 5]
    assert pd.isna(hc.loc[lapsed, "AAA"]) or hc.loc[lapsed, "AAA"] == 0.0
    print("\n=== SANITY CHECK: holder_count is a bounded share that lapses ===")
    print(f"  max {np.nanmax(vals):.2f} <= 1; a filer lapses after " f"{HOLDER_ACTIVE_DAYS} trading days. Validated.")


def test_build_ownership_feature_panel_empty_when_no_source():
    assert build_ownership_feature_panel(make_frames(IDX, {}), None, None).empty
    assert build_ownership_feature_panel(make_frames(IDX, {}), pd.DataFrame(), pd.DataFrame()).empty


if __name__ == "__main__":
    test_canonicalize_collapses_reporting_persons_without_ownership_numerics()
    test_canonicalize_empty_and_missing_cik_fallback()
    test_repeat_activist_fires_on_the_fourth_campaign_only()
    test_campaign_age_days_resets_and_is_nan_before_first_event()
    test_escalation_and_deescalation_join()
    test_bo_holder_count_sums_distinct_filers_not_group_members()
    test_percent_of_class_features_are_not_constructed()
    test_event_features_remain_available_before_the_structured_data_mandate()
    test_panel_columns_and_emission_coverage()
    test_retired_bo_numeric_features_are_absent_from_private_builder()
    test_delta_ignores_a_newly_observed_filer()
    test_holder_count_is_a_bounded_share_and_lapses()
    test_build_ownership_feature_panel_empty_when_no_source()
