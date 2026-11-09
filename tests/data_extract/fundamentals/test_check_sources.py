"""
`check_sources`: the read-only SEC vs Sharadar check on known-truth frames. One fixture, built so each finding has
one cause: `totalRevenue` is 5% high on every SEC row, `netIncome` is 10% off on exactly 1 of 100 matched rows, `cash`
(an expected fork) is 10% high on AAA only, `goodwill` exists in the SEC frame only, `totalAssets` agrees. CCC is in
Sharadar only, DDD in SEC only, ZZZ is outside the universe, BBB's Sharadar `secfilings` link names a foreign CIK, and
BF-B's vendor row is spelled `BF.B` and links a predecessor CIK its company owns.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data_extract.utils.common.identity_contract import entity_by_cik_map
from src.data_extract.utils.fundamentals.check_sources import (
    SHARADAR_GAP_EXPECTED_FIELDS,
    SourceCheck,
    check_sources,
    log_source_check,
)

_UNIVERSE = ["AAA", "BBB", "BF-B", "CCC", "DDD"]
_ROSTER = {"AAA": "1", "BBB": "0000000002", "BF-B": "0000000005", "CCC": "0000000003", "DDD": "0000000004"}
#: the published lineage rows: BF-B's company also holds its predecessor CIK 0000000055
_LINEAGE = pd.DataFrame(
    {
        "company_id": ["E1", "E2", "E5", "E5", "E3", "E4"],
        "cik": ["0000000001", "0000000002", "0000000005", "0000000055", "0000000003", "0000000004"],
    }
)
_QUARTERS = pd.date_range("2010-02-15", periods=50, freq="QS-FEB")
_LINK = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K"


def _rows(ticker: str, base: float) -> pd.DataFrame:
    n = len(_QUARTERS)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "as_of": _QUARTERS,
            "totalRevenue": base * (1 + pd.RangeIndex(n) / 100),
            "netIncome": base / 10,
            "cash": base / 4,
            "totalAssets": base * 3,
        }
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    sharadar = pd.concat([_rows("AAA", 1000.0), _rows("BBB", 500.0), _rows("CCC", 200.0), _rows("ZZZ", 50.0)], ignore_index=True)
    sec = pd.concat([_rows("AAA", 1000.0), _rows("BBB", 500.0), _rows("DDD", 80.0)], ignore_index=True)
    sec["totalRevenue"] *= 1.05
    sec.loc[(sec["ticker"] == "BBB") & (sec["as_of"] == _QUARTERS[7]), "netIncome"] *= 1.10
    sec.loc[sec["ticker"] == "AAA", "cash"] *= 1.10
    sec["goodwill"] = 1e9
    return sec, sharadar


def _vendor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BF.B", "CCC"],
            "secfilings": [
                _LINK.format(cik="0000000001"),
                _LINK.format(cik="0000000001"),
                _LINK.format(cik="0000000099"),
                _LINK.format(cik="55"),
                "N/A",
            ],
        }
    )


@pytest.fixture(scope="module")
def check() -> SourceCheck:
    sec, sharadar = _frames()
    entities = entity_by_cik_map(_LINEAGE)
    return check_sources(sec, sharadar, _vendor(), entities, _ROSTER, _UNIVERSE, gap_pct=0.03, field_share=0.05, top_tickers=3)


def test_a_field_shifted_5pct_on_every_row_is_listed_worst_first(check: SourceCheck) -> None:
    gaps = check.field_gaps.set_index("field")
    assert list(check.field_gaps["field"]) == ["totalRevenue", "cash"], check.field_gaps
    assert gaps.loc["totalRevenue", "share"] == 1.0 and gaps.loc["totalRevenue", "max_gap"] == pytest.approx(0.05)
    assert gaps.loc["totalRevenue", "rows"] == 100 and gaps.loc["totalRevenue", "top"] == [("AAA", 1.0), ("BBB", 1.0)]
    print("\n=== SANITY CHECK: a 5% shift on every row ===")
    print(f"  totalRevenue: share {gaps.loc['totalRevenue', 'share']:.0%} of {gaps.loc['totalRevenue', 'rows']} rows, max gap")
    print(f"  {gaps.loc['totalRevenue', 'max_gap']:.2%}, top {gaps.loc['totalRevenue', 'top']}; listed first. Validated.")


def test_a_field_off_on_1pct_of_rows_is_not_listed(check: SourceCheck) -> None:
    assert "netIncome" not in set(check.field_gaps["field"]) and "totalAssets" not in set(check.field_gaps["field"])
    assert {"netIncome", "totalAssets"} <= set(check.fields_compared)
    print("\n=== SANITY CHECK: 1 of 100 matched rows 10% off ===")
    print("  netIncome (1% of rows over 3%) and totalAssets (identical) are compared and not listed. Validated.")


def test_an_sec_only_field_is_not_value_checked(check: SourceCheck) -> None:
    assert check.sec_only == ["goodwill"] and "goodwill" not in check.fields_compared
    assert "goodwill" not in set(check.field_gaps["field"])
    print("\n=== SANITY CHECK: SEC-only field ===")
    print(f"  goodwill (1e9 on every SEC row, absent from Sharadar): sec_only {check.sec_only}, not compared. Validated.")


def test_an_expected_fork_is_tagged_and_still_shown(check: SourceCheck) -> None:
    gaps = check.field_gaps.set_index("field")
    assert "cash" in SHARADAR_GAP_EXPECTED_FIELDS and "totalRevenue" not in SHARADAR_GAP_EXPECTED_FIELDS
    assert bool(gaps.loc["cash", "expected"]) and not bool(gaps.loc["totalRevenue", "expected"])
    assert gaps.loc["cash", "share"] == 0.5 and gaps.loc["cash", "top"] == [("AAA", 1.0)]
    print("\n=== SANITY CHECK: expected fork ===")
    print(f"  cash: share {gaps.loc['cash', 'share']:.0%}, top {gaps.loc['cash', 'top']}, expected fork = True. Validated.")


def test_a_ticker_in_one_table_only_is_listed_within_the_universe(check: SourceCheck) -> None:
    assert check.only_sec == ["DDD"] and check.only_sharadar == ["CCC"] and check.in_both == ["AAA", "BBB"]
    assert check.matched_rows == 100 and check.sharadar_rows == 150 and check.sec_rows == 150
    print("\n=== SANITY CHECK: universe ===")
    print(f"  only SEC {check.only_sec}, only Sharadar {check.only_sharadar} (ZZZ outside the universe, not listed);")
    print(f"  exact (ticker, as_of) matches {check.matched_rows} of {check.sharadar_rows} Sharadar rows. Validated.")


def test_a_foreign_cik_and_a_missing_link_are_listed(check: SourceCheck) -> None:
    assert check.cik_mismatches == [("BBB", "0000000099", "0000000002")], check.cik_mismatches
    assert check.no_cik == ["CCC", "DDD"], check.no_cik
    print("\n=== SANITY CHECK: CIK alignment ===")
    print(f"  mismatches (ticker, Sharadar CIK, roster CIK) {check.cik_mismatches}; AAA's two vendor rows agree (roster '1' padded);")
    print("  vendor 'BF.B' matched as BF-B; its CIK 55 is its company's predecessor CIK -> not a mismatch, not missing;")
    print(f"  no parseable Sharadar CIK {check.no_cik} ('N/A' link, no vendor row). Validated.")


def test_the_log_is_short_and_findings_are_warnings(check: SourceCheck, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="src.data_extract.utils.fundamentals.check_sources"):
        log_source_check(check, seconds=1.25)
    lines = [r for r in caplog.records if r.name.endswith("check_sources")]
    text = "\n".join(r.getMessage() for r in lines)
    warned = [r.getMessage() for r in lines if r.levelno == logging.WARNING]
    assert len(lines) <= 60 and any("expected fork" in w and w.lstrip().startswith("cash") for w in warned), text
    assert any("DDD" in w for w in warned) and any("0000000099" in w for w in warned) and "1.2" in lines[-1].getMessage()
    print("\n=== SANITY CHECK: log ===")
    print(f"  {len(lines)} lines ({len(warned)} WARNING), last = {lines[-1].getMessage()!r}:")
    print(text)
    print("  Validated.")
