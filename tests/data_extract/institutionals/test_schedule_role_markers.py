"""Schedule 13D/13G role resolution through the driver (D11): the local index lists a schedule under
every party, so a filing the key only FILED becomes one `rp_seq = -1` marker under the key and is
never listed again; a subject-role filing stores real rows; an SEC error page in place of the header
is a failed unit, never a marker. Offline: seeded index, fake filings, real `DataStore` (SQLite)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_driver
from src.data_extract.utils.common.edgar_driver import run_edgar_fetch
from src.data_extract.utils.institutionals.fetch_13g_edgar import SEC_13G_FETCH
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context, fake_filing, identity_for, patch_index_filings, seed_index

_AS_OF = pd.Timestamp("2026-09-30")
_ROSTER = {"JPM": "19617", "AAA": "1"}


def _header(subject_cik: str | None, filer_cik: str = "19617") -> SimpleNamespace:
    subjects = [SimpleNamespace(company_information=SimpleNamespace(cik=subject_cik))] if subject_cik else []
    filers = [SimpleNamespace(company_information=SimpleNamespace(cik=filer_cik, name="JPMORGAN CHASE & CO"))]
    return SimpleNamespace(text="<SEC-HEADER>", subject_companies=subjects, filers=filers, reporting_owners=None, issuer=None)


def _schedule(accession: str, cik: str, *, subject_cik: str | None, issuer_cik: str) -> SimpleNamespace:
    obj = SimpleNamespace(
        has_structured_data=True,
        issuer_info=SimpleNamespace(cik=issuer_cik, name="Issuer"),
        security_info=SimpleNamespace(cusip="000000AA1", title=""),
        date_of_event="09/01/2026",
        rule_designation="Rule 13d-1(b)",
        is_amendment=False,
        amendment_number=None,
        reporting_persons=[
            SimpleNamespace(
                name="JPMorgan Chase & Co.",
                cik="19617",
                no_cik=False,
                citizenship="DE",
                type_of_reporting_person="HC",
                member_of_group=None,
                comment=None,
                sole_voting_power=10,
                shared_voting_power=0,
                sole_dispositive_power=10,
                shared_dispositive_power=0,
                aggregate_amount=10,
                percent_of_class=5.5,
            )
        ],
    )
    return fake_filing(accession, cik, "2026-09-10", form="SC 13G", header=_header(subject_cik), obj=lambda: obj)


@pytest.fixture
def ctx(tmp_path, sqlite_store, monkeypatch) -> Any:
    context = fake_context(tmp_path, sqlite_store, list(_ROSTER), ciks=list(_ROSTER.values()))
    identity = identity_for(_ROSTER)
    monkeypatch.setattr(edgar_driver, "load_identity", lambda _context: identity)
    monkeypatch.setattr(edgar_driver, "load_registrants", lambda _config_dir: {})
    return context


def _run(ctx, as_of: pd.Timestamp = _AS_OF):
    return run_edgar_fetch(ctx, list(_ROSTER), 15, SEC_13G_FETCH, as_of=as_of, refresh_index=False, max_workers=1)


def test_a_13g_the_key_only_filed_is_one_marker_never_listed_again(ctx, sqlite_store, monkeypatch):
    seed_index(ctx, [(19617, "JPMORGAN CHASE & CO", "SC 13G", "2026-09-10", "jpm-filed")])
    patch_index_filings(monkeypatch, {"jpm-filed": _schedule("jpm-filed", "19617", subject_cik="0000999999", issuer_cik="0000999999")})

    first = _run(ctx)
    again = _run(ctx, _AS_OF + pd.Timedelta(days=1))

    shown = sqlite_store.load(Tables.sec_13g, markers=True)
    assert shown[["ticker", "accession_number", "rp_seq"]].to_dict("records") == [{"ticker": "JPM", "accession_number": "jpm-filed", "rp_seq": -1}]
    assert shown["cik"].iloc[0] == "0000019617" and shown["form"].iloc[0] == "SC 13G"
    assert sqlite_store.load(Tables.sec_13g, optional=True) is None  # consumers see nothing
    assert first.outcomes["JPM"].markers == 1 and again.work.size == 0
    print("\n=== SANITY CHECK: filer-role 13G ===")
    print("  JPM's 13G on a non-universe subject -> one rp_seq=-1 marker under JPM; the next run lists 0.")


def test_a_subject_role_13g_stores_real_rows(ctx, sqlite_store, monkeypatch):
    seed_index(ctx, [(1, "AAA Inc", "SC 13G", "2026-09-10", "aaa-subject"), (19617, "JPMORGAN CHASE & CO", "SC 13G", "2026-09-10", "aaa-subject")])
    patch_index_filings(monkeypatch, {"aaa-subject": _schedule("aaa-subject", "1", subject_cik="0000000001", issuer_cik="0000000001")})

    _run(ctx)

    shown = sqlite_store.load(Tables.sec_13g, markers=True).sort_values(["ticker", "rp_seq"])
    assert shown[["ticker", "rp_seq"]].to_dict("records") == [{"ticker": "AAA", "rp_seq": 0}, {"ticker": "JPM", "rp_seq": -1}]
    real = sqlite_store.load(Tables.sec_13g)
    assert real["percent_of_class"].tolist() == [5.5] and real["reporting_person_cik"].tolist() == ["19617"]
    print("\n=== SANITY CHECK: subject-role 13G ===")
    print("  listed under both: AAA (subject) stores the reporting person's row; JPM (filer) stores only its marker.")


def test_an_sec_error_page_header_is_a_failed_unit_not_a_marker(ctx, sqlite_store, monkeypatch):
    seed_index(ctx, [(19617, "JPMORGAN CHASE & CO", "SC 13G", "2026-09-10", "jpm-503")])
    filing = _schedule("jpm-503", "19617", subject_cik="0000999999", issuer_cik="0000999999")
    filing.header = SimpleNamespace(text="", subject_companies=None, filers=None, reporting_owners=None, issuer=None)  # edgartools' fallback
    patch_index_filings(monkeypatch, {"jpm-503": filing})

    summary = _run(ctx)

    assert not sqlite_store.exists(Tables.sec_13g)
    assert summary.missing == {"JPM": ["jpm-503"]}
    assert _run(ctx, _AS_OF + pd.Timedelta(days=1)).work.size == 1
    print("\n=== SANITY CHECK: empty header ===")
    print("  an empty SGML header (SEC error page) -> TransientReadError -> nothing stored, listed again the next night.")
