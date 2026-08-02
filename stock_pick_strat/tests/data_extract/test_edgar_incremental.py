"""Tests for the EDGAR incremental-extraction machinery
(sec_utils meta sidecar, list_filings `since` cutoff, per-ticker cutoffs, and the
skip-if-fresh guard). All network access is mocked, so these are fast and offline.

What matters: a re-run must fetch ONLY filings after the last date already parsed
(`D`), and must skip entirely when already built today.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.sec_utils import (
    load_extract_meta, save_extract_meta, today_iso,
)


# --------------------------------------------------------------------------- #
# Mock SEC submissions response                                                #
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _submissions(forms_dates):
    """Fake data.sec.gov submissions JSON from [(form, 'YYYY-MM-DD'), ...]."""
    n = len(forms_dates)
    return {
        "name": "TEST CO",
        "filings": {
            "recent": {
                "accessionNumber": [f"0000-{i}" for i in range(n)],
                "form": [f for f, _ in forms_dates],
                "filingDate": [d for _, d in forms_dates],
                "primaryDocument": [f"doc{i}.htm" for i in range(n)],
                "reportDate": [d for _, d in forms_dates],
            },
            "files": [],   # no older archive pages
        },
    }


# --------------------------------------------------------------------------- #
# 1. list_filings `since` = incremental window                                 #
# --------------------------------------------------------------------------- #
def test_list_filings_since_filters_to_after_D(monkeypatch):
    import src.data_extract.utils.common.edgar_fillings as ef

    payload = _submissions([
        ("10-K", "2020-02-15"),
        ("10-K", "2021-02-15"),
        ("10-K", "2022-02-15"),
        ("10-Q", "2022-05-15"),   # wrong form -> excluded
    ])
    monkeypatch.setattr(ef, "sec_get", lambda url, **k: _FakeResp(payload))

    # full window: every 10-K (10-Q excluded by form filter)
    allf = ef.list_filings("320193", ["10-K"], years=20)
    assert list(allf["filing_date"].dt.strftime("%Y-%m-%d")) == \
        ["2020-02-15", "2021-02-15", "2022-02-15"]

    # incremental: since D=2021-02-15 -> only filings STRICTLY after D
    inc = ef.list_filings("320193", ["10-K"], years=20, since="2021-02-15")
    assert list(inc["filing_date"].dt.strftime("%Y-%m-%d")) == ["2022-02-15"]

    print("\n=== SANITY CHECK: list_filings incremental `since` ===")
    print(f"  full window -> {len(allf)} 10-Ks;  since 2021-02-15 -> {len(inc)} "
          f"(only 2022-02-15, strictly after D). Validated.")


# --------------------------------------------------------------------------- #
# 2. Meta sidecar round-trip + freshness/universe skip guard                   #
# --------------------------------------------------------------------------- #
def test_extract_meta_roundtrip(tmp_path):
    pq = tmp_path / "fundamentals_history.parquet"
    assert load_extract_meta(pq) is None                     # nothing built yet

    save_extract_meta(pq, "2026-05-01", ticker_count=3, universe_size=5)
    meta = load_extract_meta(pq)

    assert (tmp_path / "fundamentals_history_meta.json").exists()
    assert meta["last_built"] == today_iso()
    assert meta["last_filing_date"] == "2026-05-01"
    assert meta["universe_size"] == 5

    print("\n=== SANITY CHECK: extract meta sidecar ===")
    print(f"  wrote/read fundamentals_history_meta.json: last_built={meta['last_built']}, "
          f"D={meta['last_filing_date']}, universe={meta['universe_size']}. Validated.")

# NOTE: the employee-specific skip-if-fresh guard (`_is_up_to_date`) and per-ticker
# `as_of` cutoff (`_last_asof_by_ticker`) tests were REMOVED with the machinery they
# covered. Employee headcount is now a `fundamentals_facts` field parsed from the
# same 10-K as the fundamentals (`fundamentals_employees.py`), so its incremental
# behaviour is `fetch_fundamentals_edgar`'s `done_accessions` skip -- one mechanism
# for every field, exercised by tests/data_extract/test_fundamentals_employees.py.
