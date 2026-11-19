"""Schedule 13D (active) and 13G (passive) filing counts: one per accession, 126-session window, universe z-score, clip, unknown cells."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.institutionals.ownership_features import (
    CONCENTRATED_MIN_FILINGS,
    COUNT_CLIP,
    COUNT_WINDOW,
    EMISSION,
    ScheduleSource,
    SubjectIdentity,
    _filings,
    build_ownership_feature_panel,
)
from src.data_store.schema import Tables
from tests.conftest import make_frames
from tests.data_extract.fake_context import extract_config

IDX = pd.bdate_range("2023-01-03", "2025-06-30")
TICKERS = [f"T{i:02d}" for i in range(6)]
D13 = "f_ic_act_13d_count_126d_z"
G13 = "f_ic_bo_13g_count_126d_z"


def _peers(tickers):
    return {t: {p: 1.0 for p in tickers if p != t} for t in tickers}


def _filing(ticker, day, accession=None):
    return {"ticker": ticker, "filing_date": pd.Timestamp(day), "accession_number": accession or f"{ticker}-{pd.Timestamp(day):%Y%m%d}"}


def _filings_on(ticker, days):
    return [_filing(ticker, IDX[d], f"{ticker}-{d}") for d in days]


def _panel(df_13d=None, df_13g=None, tickers=TICKERS, idx=IDX, **kwargs):
    return build_ownership_feature_panel(
        make_frames(idx, _peers(tickers)),
        (ScheduleSource(Tables.sec_13d, df_13d, None), ScheduleSource(Tables.sec_13g, df_13g, None)),
        extract_config(),
        **kwargs,
    )


def _series(panel, ticker, column):
    return panel[panel["ticker"] == ticker].set_index("date")[column]


def test_filings_are_counted_once_per_accession_whatever_the_reporting_persons_or_cusips():
    df_schedule = pd.DataFrame(
        [
            _filing("AAA", "2025-02-10", "a1"),
            _filing("AAA", "2025-02-10", "a1"),  # second reporting person of the same filing
            _filing("AAA", "2025-03-14", "a2"),  # amendment: its own filing
            _filing("AAA", "2025-03-14", "a3"),
        ]
    )
    df_filings = _filings(df_schedule).sort_values("accession_number")
    assert list(df_filings["accession_number"]) == ["a1", "a2", "a3"]
    print("\n=== SANITY CHECK: one count per accession ===")
    print("  Co-filer rows collapse to their filing; each amendment and each other accession counts. Validated.")


def test_a_filing_stays_in_the_count_for_exactly_the_trailing_window():
    panel = _panel(pd.DataFrame(_filings_on("T00", [10])))
    s, other = _series(panel, "T00", D13), _series(panel, "T01", D13)
    last = COUNT_WINDOW - 1
    assert s.loc[IDX[11]] > 0 and s.loc[IDX[11 + last]] > 0 and (other.loc[IDX[11] : IDX[11 + last]] < 0).all()
    assert s.loc[IDX[11 + COUNT_WINDOW]] != s.loc[IDX[11 + COUNT_WINDOW]], "after expiry the whole cross-section is 0: no dispersion, so unknown"
    print("\n=== SANITY CHECK: trailing window ===")
    print(f"  a filing lifts the ticker above the universe for exactly {COUNT_WINDOW} sessions. Validated.")


def test_value_is_the_universe_z_score_and_is_clipped():
    tickers = [f"U{i:02d}" for i in range(40)]
    rows = [r for d in range(20) for r in _filings_on("U00", [d])] + _filings_on("U01", [0, 1, 2])
    panel = _panel(pd.DataFrame(rows), tickers=tickers)
    at = panel[panel["date"] == IDX[30]].set_index("ticker")[D13]
    counts = pd.Series(0.0, index=tickers)
    counts["U00"], counts["U01"] = 20.0, 3.0
    expected = ((counts - counts.mean()) / counts.std()).clip(-COUNT_CLIP, COUNT_CLIP)
    assert np.isclose(at["U00"], COUNT_CLIP), "an extreme count is clipped"
    assert np.allclose(at.sort_index().to_numpy(), expected.sort_index().to_numpy())
    print("\n=== SANITY CHECK: z-score and clip ===")
    print(f"  the value equals (count - universe mean) / universe std and an outlier stops at {COUNT_CLIP}. Validated.")


def test_a_change_of_filing_regime_does_not_change_the_value():
    early = [r for t, n in (("T00", 4), ("T01", 2), ("T02", 2)) for r in _filings_on(t, range(10, 10 + n))]
    late = [r for t, n in (("T00", 2), ("T01", 1), ("T02", 1)) for r in _filings_on(t, range(300, 300 + n))]
    panel = _panel(pd.DataFrame(early + late))
    z_early, z_late = _series(panel, "T00", D13).loc[IDX[20]], _series(panel, "T00", D13).loc[IDX[320]]
    assert np.isclose(z_early, z_late)
    print("\n=== SANITY CHECK: stable across filing regimes ===")
    print("  halving every ticker's filing count leaves the standardised value unchanged. Validated.")


def test_weekend_filing_lands_on_the_next_session():
    saturday = _filing("T00", "2024-03-02")
    s = _series(_panel(pd.DataFrame([saturday])), "T00", D13)
    assert s.loc[pd.Timestamp("2024-03-04")] > 0 and np.isnan(s.loc[pd.Timestamp("2024-03-01")])


def test_13d_and_13g_are_separate_raw_columns():
    panel = _panel(pd.DataFrame(_filings_on("T00", [30])), pd.DataFrame(_filings_on("T01", [40, 50])))
    assert set(EMISSION) == {D13.removeprefix("f_"), G13.removeprefix("f_")} and set(EMISSION.values()) == {"raw"}
    assert _series(panel, "T00", D13).loc[IDX[60]] > 0 and _series(panel, "T01", G13).loc[IDX[60]] > 0
    assert _series(panel, "T00", G13).loc[IDX[60]] < 0
    assert not any(c.endswith(("_xs", "_vs_peers")) for c in panel.columns)


def test_every_ownership_output_stops_at_its_complete_frontier():
    idx = pd.bdate_range("2025-01-02", periods=90)
    close = pd.DataFrame(100.0, index=idx, columns=TICKERS)
    df_13g = pd.DataFrame([_filing("T00", idx[10], "g1")])
    df_13d = pd.DataFrame([_filing("T00", idx[20], "d1"), _filing("T00", idx[70], "late")])
    frontier = idx[55]
    panel = build_ownership_feature_panel(
        make_frames(idx, _peers(TICKERS), close_split=close),
        (ScheduleSource(Tables.sec_13d, df_13d, frontier), ScheduleSource(Tables.sec_13g, df_13g, frontier)),
        extract_config(),
    )
    dense = panel.set_index(["date", "ticker"]).reindex(pd.MultiIndex.from_product([idx, TICKERS], names=["date", "ticker"]))
    cols = [c for c in dense if c.startswith("f_")]
    assert cols and dense.loc[(slice(idx[57], None), slice(None)), cols].isna().all().all()
    assert dense.loc[(idx[56], slice(None)), cols].notna().all().all(), "inside the frontier every ticker has a value"
    print("\n=== SANITY CHECK: complete-through masking ===")
    print("  every leg is NaN after the inclusive source frontier (its publication session) and defined for every ticker inside it. Validated.")


def test_build_ownership_feature_panel_empty_when_no_source():
    assert build_ownership_feature_panel(
        make_frames(IDX, {}), (ScheduleSource(Tables.sec_13d, None, None), ScheduleSource(Tables.sec_13g, None, None)), extract_config()
    ).empty
    assert build_ownership_feature_panel(
        make_frames(IDX, {}),
        (ScheduleSource(Tables.sec_13d, pd.DataFrame(), None), ScheduleSource(Tables.sec_13g, pd.DataFrame(), None)),
        extract_config(),
    ).empty


def test_marker_filings_are_counted_once_next_to_parsed_ones(sqlite_store) -> None:
    from src.data_aggregate.utils.common import inputs as capital_inputs
    from src.data_store.schema import Tables

    day = pd.Timestamp("2024-01-02")
    rows = [
        # a1: parsed row plus a marker of the same accession; a2: marker only; a3: two parsed co-filers
        ("AAA", "a1", 0, day),
        ("AAA", "a1", -1, day),
        ("AAA", "a2", -1, day),
        ("AAA", "a3", 0, day),
        ("AAA", "a3", 1, day),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "accession_number", "rp_seq", "filing_date"])
    sqlite_store.save(Tables.sec_13d, df)
    parsed_only = capital_inputs.load_source(sqlite_store, Tables.sec_13d, ["AAA"])
    with_markers = capital_inputs.load_source(sqlite_store, Tables.sec_13d, ["AAA"], markers=True)
    assert parsed_only is not None and with_markers is not None
    assert len(_filings(parsed_only)) == 2 and len(_filings(with_markers)) == 3
    print("\n=== SANITY CHECK: marker filings ===")
    print("  Marker-only filing counted once, parsed+marker filing once, co-filers once: 3 of 3 (parsed-only read sees 2). Validated.")


def _identity():
    lineage = pd.DataFrame({"canonical_ticker": ["AAA", "BBB"], "cik": ["0000000001", "0000000002"], "role": ["cik_window", "cik_window"]})
    lines = pd.DataFrame(
        {"canonical_company": ["AAA", "BBB"], "cusip": ["AAAA00001", "BBBB00001"], "lineage_role": ["canonical_current", "canonical_current"]}
    )
    return SubjectIdentity.from_tables(lineage, lines)


def _row(ticker, accession, rp_seq=0, cusip="", rp_cik=""):
    return {
        "ticker": ticker,
        "accession_number": accession,
        "filing_date": pd.Timestamp("2025-02-10"),
        "rp_seq": rp_seq,
        "cusip": cusip,
        "reporting_person_cik": rp_cik,
    }


def test_filings_the_ticker_filed_about_others_are_dropped_and_its_subject_filings_kept():
    df = pd.DataFrame(
        [
            _row("AAA", "0000009999-25-000001", cusip="AAAA00001"),  # subject: its own CUSIP
            _row("AAA", "0000000001-25-000002", rp_seq=-1),  # marker submitted under AAA's own CIK: AAA filed it
            _row("AAA", "0000009999-25-000003", rp_seq=-1),  # marker by an unrelated submitter: kept
            _row("AAA", "0000009999-25-000004", cusip="BBBB00001"),  # about BBB ...
            _row("BBB", "0000009999-25-000004", cusip="BBBB00001"),  # ... listed under both: BBB is the subject
            _row("AAA", "0000009999-25-000005", rp_cik="0000000001"),  # legacy parsed row, AAA is the reporting person
            _row("AAA", "0000009999-25-000006", rp_cik="0000000555"),  # legacy parsed row, another reporting person: kept
        ]
    )
    kept = _filings(df, _identity())
    assert sorted(zip(kept["ticker"], kept["accession_number"], strict=True)) == [
        ("AAA", "0000009999-25-000001"),
        ("AAA", "0000009999-25-000003"),
        ("AAA", "0000009999-25-000006"),
        ("BBB", "0000009999-25-000004"),
    ]
    assert len(_filings(df)) == 7, "without an identity every (ticker, accession) pair counts"
    print("\n=== SANITY CHECK: filer-side filings dropped ===")
    print("  own-CIK prefix, other-ticker CUSIP subject and own reporting person drop; own CUSIP, unrelated marker and other person stay. Validated.")


def test_a_concentrated_institution_submitter_loses_its_markers_but_not_its_parsed_filings():
    hub = [_row("AAA", f"0000007777-25-{i:06d}", rp_seq=-1) for i in range(CONCENTRATED_MIN_FILINGS)]
    other = [_row("BBB", "0000007777-25-999999", rp_seq=-1), _row("AAA", "0000007777-25-777777", cusip="AAAA00001")]
    kept = _filings(pd.DataFrame(hub + other), _identity())
    assert sorted(zip(kept["ticker"], kept["accession_number"], strict=True)) == [("AAA", "0000007777-25-777777"), ("BBB", "0000007777-25-999999")]
    print("\n=== SANITY CHECK: concentrated submitter ===")
    print(
        f"  {CONCENTRATED_MIN_FILINGS} markers of one prefix under one ticker drop; its subject-CUSIP filing and a minority marker stay. Validated."
    )


def test_a_marker_heavy_filer_keeps_only_its_parsed_subject_filings_while_an_ordinary_ticker_keeps_its_markers():
    n = CONCENTRATED_MIN_FILINGS
    # AAA: 4n markers spread over four agent prefixes (no prefix reaches the concentrated test) + 5 parsed subject filings
    filer = [_row("AAA", f"000000{agent}-25-{i:06d}", rp_seq=-1) for agent in (1111, 2222, 3333, 4444) for i in range(n)]
    filer += [_row("AAA", f"0000009999-25-{i:06d}", cusip="AAAA00001") for i in range(5)]
    # BBB: 20 markers + 80 parsed subject filings, far below the thresholds
    ordinary = [_row("BBB", f"0000001111-24-{i:06d}", rp_seq=-1) for i in range(20)]
    ordinary += [_row("BBB", f"0000009998-24-{i:06d}", cusip="BBBB00001") for i in range(80)]
    kept = _filings(pd.DataFrame(filer + ordinary), _identity())
    counts = kept.groupby("ticker").size().to_dict()
    assert counts == {"AAA": 5, "BBB": 100}
    print("\n=== SANITY CHECK: marker-heavy filer ===")
    print(
        f"  AAA ({4 * n} markers of 4 agents, 5 parsed) keeps its 5 parsed subject filings; BBB (20 markers of 100) keeps all 100: {counts}. Validated."
    )
