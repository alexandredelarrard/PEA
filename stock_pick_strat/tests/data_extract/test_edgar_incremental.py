"""Tests for the EDGAR incremental-extraction machinery
(sec_utils meta sidecar, list_filings `since` cutoff, per-ticker cutoffs, and the
skip-if-fresh guard). All network access is mocked, so these are fast and offline.

What matters: a re-run must fetch ONLY filings after the last date already parsed
(`D`), must skip entirely when already built today, and the officer cutoff must
not be advanced by insider-only rows.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.sec_utils import (
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
    import src.data_extract.utils.edgar_fillings as ef

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
    pq = tmp_path / "employees_history.parquet"
    assert load_extract_meta(pq) is None                     # nothing built yet

    save_extract_meta(pq, "2026-05-01", ticker_count=3, universe_size=5)
    meta = load_extract_meta(pq)

    assert (tmp_path / "employees_history_meta.json").exists()
    assert meta["last_built"] == today_iso()
    assert meta["last_filing_date"] == "2026-05-01"
    assert meta["universe_size"] == 5

    print("\n=== SANITY CHECK: extract meta sidecar ===")
    print(f"  wrote/read employees_history_meta.json: last_built={meta['last_built']}, "
          f"D={meta['last_filing_date']}, universe={meta['universe_size']}. Validated.")


def test_is_up_to_date_skip_guard(tmp_path):
    from src.data_extract.utils import fetch_employees_edgar as fee

    pq = tmp_path / "employees_history.parquet"
    pd.DataFrame({"ticker": ["AAA"], "as_of": pd.to_datetime(["2021-02-01"]),
                  "employees": [100]}).to_parquet(pq)

    class _Ctx:
        paths = {"EMPLOYEES_HISTORY_PATH": pq}

    ctx, cik_map = _Ctx(), pd.DataFrame({"ticker": ["AAA", "BBB"]})

    assert fee._is_up_to_date(ctx, cik_map) is False          # no meta yet
    save_extract_meta(pq, "2021-02-01", ticker_count=2, universe_size=2)
    assert fee._is_up_to_date(ctx, cik_map) is True           # built today, universe covered
    save_extract_meta(pq, "2021-02-01", ticker_count=2, universe_size=1)
    assert fee._is_up_to_date(ctx, cik_map) is False          # universe grew -> stale

    print("\n=== SANITY CHECK: skip-if-fresh guard ===")
    print("  no meta -> run; built-today & universe covered -> skip; universe grew -> run. Validated.")


# --------------------------------------------------------------------------- #
# 3. Per-ticker incremental cutoff (`D`)                                       #
# --------------------------------------------------------------------------- #
def test_employees_cutoff_per_ticker():
    from src.data_extract.utils.fetch_employees_edgar import _last_asof_by_ticker

    existing = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "as_of": pd.to_datetime(["2020-02-01", "2021-02-01", "2019-03-01"]),
        "employees": [100, 110, 50],
        "accession_number": ["a1", "a2", "b1"],
    })
    cutoff = _last_asof_by_ticker(existing)

    assert cutoff["AAA"] == pd.Timestamp("2021-02-01")     # latest per ticker
    assert cutoff["BBB"] == pd.Timestamp("2019-03-01")
    assert "CCC" not in cutoff                              # new ticker -> full history

    print("\n=== SANITY CHECK: employees per-ticker cutoff ===")
    print(f"  AAA -> {cutoff['AAA'].date()}, BBB -> {cutoff['BBB'].date()}, "
          f"unseen ticker -> full history. Validated.")


def test_management_cutoff_ignores_insider_rows():
    from src.data_extract.utils.fetch_management_edgar import _last_officer_asof

    existing = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "as_of": pd.to_datetime(["2021-02-01", "2022-02-01", "2023-06-01"]),
        # last row is an insider-only row (no accession) with a LATER date
        "accession_number": ["a1", "a2", None],
    })
    cutoff = _last_officer_asof(existing)

    # the later insider-only row must NOT advance the officer cutoff, else we'd
    # wrongly skip re-fetching 10-K officer data filed between 2022 and 2023.
    assert cutoff["AAA"] == pd.Timestamp("2022-02-01")

    print("\n=== SANITY CHECK: management officer cutoff ignores insider rows ===")
    print(f"  officer cutoff -> {cutoff['AAA'].date()} (insider-only 2023-06 row ignored). Validated.")
