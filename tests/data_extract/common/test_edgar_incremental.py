"""Tests for the EDGAR incremental-extraction machinery (`list_filings`'s `since`
cutoff -- the mechanism the manifest-driven EDGAR fetchers reuse, see
`run_manifest.py` and `tests/data_extract/utils/common/test_run_manifest.py` for
the manifest side). All network access is mocked, so these are fast and offline.

What matters: a re-run must fetch ONLY filings after the last date already parsed
(`D`).
"""
from __future__ import annotations

import pandas as pd


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
    monkeypatch.setattr(ef, "sec_get", lambda context, url, **k: _FakeResp(payload))

    # full window: every 10-K (10-Q excluded by form filter)
    allf = ef.list_filings(None, "320193", ["10-K"], years=20)
    assert list(allf["filing_date"].dt.strftime("%Y-%m-%d")) == \
        ["2020-02-15", "2021-02-15", "2022-02-15"]

    # incremental: since D=2021-02-15 -> only filings STRICTLY after D
    inc = ef.list_filings(None, "320193", ["10-K"], years=20, since="2021-02-15")
    assert list(inc["filing_date"].dt.strftime("%Y-%m-%d")) == ["2022-02-15"]

    print("\n=== SANITY CHECK: list_filings incremental `since` ===")
    print(f"  full window -> {len(allf)} 10-Ks;  since 2021-02-15 -> {len(inc)} "
          f"(only 2022-02-15, strictly after D). Validated.")

# NOTE: the bespoke `sec_utils` meta-sidecar (`meta_path`/`load_extract_meta`/
# `save_extract_meta`) was RETIRED -- `fetch_def14a_llm.py` was its only caller and
# now reads/writes the shared `run_manifest.py` checkpoint instead (see
# tests/data_extract/utils/common/test_run_manifest.py and
# test_def14a_incremental.py's manifest-driven window tests). The employee-specific
# skip-if-fresh guard (`_is_up_to_date`) and per-ticker `as_of` cutoff
# (`_last_asof_by_ticker`) tests were REMOVED with the machinery they covered.
# Employee headcount is now a `fundamentals_facts` field parsed from the same 10-K
# as the fundamentals (`fundamentals_employees.py`), so its incremental behaviour is
# the fundamentals fetcher's `done_accessions` skip -- one mechanism for every
# field, exercised by tests/data_extract/test_fundamentals_employees.py.
