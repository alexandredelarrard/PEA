"""The local EDGAR filing index (`edgar_index.py`): `master.idx` parsing, the quarter refresh rule
(closed quarters once, the current one every run, the previous one early in a quarter) and the
filtered read. Offline: `sec_io.download` is replaced by a fake that serves fixture files."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_index
from tests.data_extract.edgar_fixtures import fake_context, master_text

#: 30 index lines: universe and non-universe CIKs, policy forms and forms the cache drops.
_LINES = [
    (320193, "Apple Inc.", "8-K", "2026-07-01", "0000320193-26-000071"),
    (320193, "Apple Inc.", "10-Q", "2026-07-31", "0000320193-26-000080"),
    (320193, "Apple Inc.", "4", "2026-08-02", "0001127602-26-000101"),
    (320193, "Apple Inc.", "SC 13G/A", "2026-08-05", "0000315066-26-000001"),
    (320193, "Apple Inc.", "424B2", "2026-08-06", "0001193125-26-000001"),
    (320193, "Apple Inc.", "S-8", "2026-08-07", "0001193125-26-000002"),
    (19617, "JPMORGAN CHASE & CO", "4", "2026-07-02", "0000019617-26-000001"),
    (19617, "JPMORGAN CHASE & CO", "SCHEDULE 13G", "2026-07-03", "0000019617-26-000002"),
    (19617, "JPMORGAN CHASE & CO", "13F-HR", "2026-08-14", "0000019617-26-000003"),
    (19617, "JPMORGAN CHASE & CO", "FWP", "2026-08-15", "0000019617-26-000004"),
    (19617, "JPMORGAN CHASE & CO", "10-Q", "2026-08-01", "0000019617-26-000005"),
    (19617, "JPMORGAN CHASE & CO", "DEF 14A", "2026-04-01", "0000019617-26-000006"),
    (1364742, "BlackRock Inc.", "SC 13G", "2026-07-10", "0001364742-26-000001"),
    (1364742, "BlackRock Inc.", "SC 13G/A", "2026-07-11", "0001364742-26-000002"),
    (1364742, "BlackRock Inc.", "8-K", "2026-07-12", "0001364742-26-000003"),
    (1364742, "BlackRock Inc.", "N-PX", "2026-08-12", "0001364742-26-000004"),
    (999999, "Tiny Fund LP", "SC 13D", "2026-07-20", "0000999999-26-000001"),
    (999999, "Tiny Fund LP", "SC 13D/A", "2026-08-20", "0000999999-26-000002"),
    (999999, "Tiny Fund LP", "D", "2026-08-21", "0000999999-26-000003"),
    (888888, "Other Corp", "10-K", "2026-07-25", "0000888888-26-000001"),
    (888888, "Other Corp", "10-K/A", "2026-08-25", "0000888888-26-000002"),
    (888888, "Other Corp", "DEFC14A", "2026-08-26", "0000888888-26-000003"),
    (888888, "Other Corp", "8-K12B", "2026-08-27", "0000888888-26-000004"),
    (888888, "Other Corp", "3", "2026-08-28", "0000888888-26-000005"),
    (888888, "Other Corp", "5/A", "2026-08-29", "0000888888-26-000006"),
    (777777, "Société Générale", "SC 13G", "2026-09-01", "0000777777-26-000001"),
    (777777, "Société Générale", "6-K", "2026-09-02", "0000777777-26-000002"),
    (777777, "Société Générale", "10-Q/A", "2026-09-03", "0000777777-26-000003"),
    (777777, "Société Générale", "DEF 14C", "2026-09-04", "0000777777-26-000004"),
    (777777, "Société Générale", "13F-HR/A", "2026-09-05", "0000777777-26-000005"),
]
_DROPPED_FORMS = {"424B2", "S-8", "FWP", "N-PX", "D", "6-K"}


def test_master_idx_parses_to_typed_policy_rows():
    raw = gzip.compress(master_text(_LINES).encode("latin-1"))
    df = edgar_index.parse_master(raw)

    assert len(_LINES) == 30 and len(df) == 30 - 6
    assert not set(df["form"]) & _DROPPED_FORMS
    assert df["cik"].dtype == "int64"
    first = df[df["accession"] == "0000320193-26-000071"].iloc[0]
    assert first["cik"] == 320193 and first["form"] == "8-K" and str(first["filed"]) == "2026-07-01"
    assert "Société Générale" in set(df["company"])  # latin-1 company names survive
    print("\n=== SANITY CHECK: master.idx parse ===")
    print(f"  30 lines -> {len(df)} rows of policy/13F forms; 6 other forms dropped; accession from the filename; CIK int64. Validated.")


@pytest.fixture
def served(monkeypatch) -> list[str]:
    """Every quarter URL `sec_io.download` was asked for; each answers with the fixture index."""
    requested: list[str] = []

    def download(context, url: str, path: Path, *, timeout: int = 300) -> int:
        requested.append(url)
        path.write_bytes(gzip.compress(master_text(_LINES).encode("latin-1")))
        return 200

    monkeypatch.setattr(edgar_index.sec_io, "download", download)
    return requested


def _quarters(urls: list[str]) -> list[str]:
    return [f"{url.split('/')[-3]}Q{url.split('/')[-2][-1]}" for url in urls]


def test_a_closed_quarter_is_downloaded_once_and_the_current_quarter_every_run(tmp_path, sqlite_store, served):
    ctx = fake_context(tmp_path, sqlite_store, ["AAPL"])

    first = edgar_index.refresh(ctx, pd.Timestamp("2026-10-20"), years_history=1)
    second = edgar_index.refresh(ctx, pd.Timestamp("2026-10-21"), years_history=1)
    early = edgar_index.refresh(ctx, pd.Timestamp("2027-01-05"), years_history=1)

    assert sorted(first) == ["2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4"]
    assert list(second) == ["2026Q4"]  # closed quarters are cached, the current one is not
    assert sorted(early) == ["2026Q4", "2027Q1"]  # day 5 of 2027Q1: the previous quarter again
    assert _quarters(served) == ["2025Q4", "2026Q1", "2026Q2", "2026Q3", "2026Q4", "2026Q4", "2026Q4", "2027Q1"]
    assert edgar_index.quarter_counts(ctx)["2026Q3"] == 24
    assert not list(edgar_index.index_dir(ctx).glob("*.gz")) and not list(edgar_index.index_dir(ctx).glob("*.part"))
    print("\n=== SANITY CHECK: index refresh rule ===")
    print(f"  run 1 downloads {len(first)} quarters; run 2 only the current one; on 2027-01-05 the previous quarter too: {sorted(early)}.")


def test_a_quarter_sec_does_not_serve_writes_nothing(tmp_path, sqlite_store, monkeypatch):
    ctx = fake_context(tmp_path, sqlite_store, ["AAPL"])
    monkeypatch.setattr(edgar_index.sec_io, "download", lambda context, url, path, **kw: 404)

    written = edgar_index.refresh(ctx, pd.Timestamp("2026-10-02"), years_history=0)

    assert written == {} and edgar_index.quarter_counts(ctx) == {}
    assert any("2026Q4 not served (HTTP 404)" in w for w in ctx.warnings)
    print("\nSANITY: a 404 quarter is warned and leaves the cache untouched.")


def test_entries_filter_ciks_forms_and_date(tmp_path, sqlite_store, served):
    ctx = fake_context(tmp_path, sqlite_store, ["AAPL"])
    edgar_index.refresh(ctx, pd.Timestamp("2026-08-01"), years_history=0)  # one quarter file: 2026Q3

    df = edgar_index.entries(ctx, {"0000320193", "19617"}, ["4", "8-K"], since=pd.Timestamp("2026-07-02"))

    assert df["accession"].tolist() == ["0000019617-26-000001", "0001127602-26-000101"]
    assert set(df["cik"]) == {"0000320193", "0000019617"} and df["filed"].dtype.kind == "M"
    assert edgar_index.entries(ctx, set(), ["4"]).empty
    print("\n=== SANITY CHECK: filtered read ===")
    print(f"  CIKs {{AAPL, JPM}} x forms {{4, 8-K}} from 2026-07-02 -> {df['accession'].tolist()}; CIKs padded, dates Timestamps.")
