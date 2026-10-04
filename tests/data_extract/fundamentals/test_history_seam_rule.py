"""The seam rule of `build_history`: across a CIK seam, one filer per fiscal period.

GOOGL's register seam is 2015-10-02 (Google Inc 1288776 -> Alphabet 1652044). Both registrants filed
a 10-Q for the quarter ended 2015-09-30 on 2015-10-29, inside the 31-day listing margin, so both are
listed and stored. History keeps the filing of the CIK whose STATED window owns the period end; a
margin filing that reports a period nobody else reports is kept. Known-truth fixture, offline.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.fundamentals import build_history as mod
from src.data_extract.utils.fundamentals.build_history import FACT_COLUMNS, TickerHistory, keep_window_owner_filings
from src.data_store.schema import Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

GOOGLE, ALPHABET = "0001288776", "0001652044"
_IDENTITY = dated_identity(
    [("GOOGL", GOOGLE, "cik_window", SENTINEL, "2015-10-02"), ("GOOGL", ALPHABET, "cik_window", "2015-10-02", None)],
    {"GOOGL": ALPHABET},
)


def _facts() -> pd.DataFrame:
    """(accession, cik, form, filed, period) -- two rows per filing, as a filing carries several facts."""
    filings = [
        ("google-q2", GOOGLE, "10-Q", "2015-07-23", "2015-06-30"),
        ("google-q3", GOOGLE, "10-Q", "2015-10-29", "2015-09-30"),
        ("alphabet-q3", ALPHABET, "10-Q", "2015-10-29", "2015-09-30"),
        ("alphabet-stub", ALPHABET, "10-Q", "2015-10-29", "2015-08-31"),
        ("alphabet-fy", ALPHABET, "10-K", "2016-02-11", "2015-12-31"),
    ]
    return pd.DataFrame(
        [
            {
                "ticker": "GOOGL",
                "cik": cik,
                "accession_number": accession,
                "form": form,
                "filing_date": pd.Timestamp(filed),
                "period_of_report": pd.Timestamp(period),
                "field": field,
                "value": 1.0,
            }
            for accession, cik, form, filed, period in filings
            for field in ("totalRevenue", "totalAssets")
        ]
    )


def test_a_period_two_ciks_report_keeps_the_window_owner_only():
    facts = _facts()
    kept = keep_window_owner_filings(facts, _IDENTITY.filing_scope("GOOGL").windows)

    kept_accessions = list(dict.fromkeys(kept["accession_number"]))
    assert kept_accessions == ["google-q2", "google-q3", "alphabet-stub", "alphabet-fy"]
    periods = kept.drop_duplicates("accession_number").groupby("period_of_report")["cik"].nunique()
    assert (periods == 1).all(), "a fiscal period is reported by two CIKs"
    assert set(facts["period_of_report"]) == set(kept["period_of_report"]), "a fiscal period was lost"
    print("\n=== SANITY CHECK: GOOGL Q3-2015 seam rule ===")
    print("  2015-09-30 reported by Google Inc and Alphabet -> Google Inc's (its stated window owns the period end) kept")
    print("  Alphabet's 2015-08-31 stub (only it reports) and its FY2015 10-K kept; no period lost or doubled")


def test_a_filing_outside_its_filers_window_is_set_aside_even_alone_in_its_period():
    """F-001 (JCI/Tyco shape): the successor CIK's own pre-seam 10-K ends its 52/53-week year days away from the
    predecessor's, so no period is shared; the filing is still outside its filer's widened window and is set aside."""
    jci, tyco = "0000053669", "0000833444"
    identity = dated_identity([("JCI", jci, "cik_window", SENTINEL, "2016-09-02"), ("JCI", tyco, "cik_window", "2016-09-02", None)], {"JCI": tyco})
    filings = [
        ("tyco-fy2015", tyco, "2015-11-13", "2015-09-25"),
        ("jci-fy2015", jci, "2015-11-18", "2015-09-30"),
        ("tyco-margin", tyco, "2016-08-20", "2016-06-24"),
        ("jci-plc-fy2017", tyco, "2017-11-21", "2017-09-30"),
        ("no-cik", None, "2014-01-01", "2013-12-31"),
    ]
    facts = pd.DataFrame(
        [
            {
                "ticker": "JCI",
                "cik": c,
                "accession_number": a,
                "form": "10-K",
                "filing_date": pd.Timestamp(f),
                "period_of_report": pd.Timestamp(p),
                "field": "totalRevenue",
                "value": 1.0,
            }
            for a, c, f, p in filings
        ]
    )

    kept = keep_window_owner_filings(facts, identity.filing_scope("JCI").windows)

    assert list(kept["accession_number"]) == ["jci-fy2015", "tyco-margin", "jci-plc-fy2017", "no-cik"], list(kept["accession_number"])
    print("\n=== SANITY CHECK: F-001 JCI/Tyco filer window ===")
    print("  Tyco's FY2015 10-K (filed 10 months before its window) set aside; its margin filing, JCI's own FY2015 and the null-CIK row kept")


def test_a_single_window_or_cikless_frame_is_untouched():
    facts = _facts()
    assert keep_window_owner_filings(facts, _IDENTITY.filing_scope("GOOGL").windows[:1]).equals(facts)
    assert keep_window_owner_filings(facts.drop(columns="cik"), _IDENTITY.filing_scope("GOOGL").windows).equals(facts.drop(columns="cik"))
    print("\n=== SANITY CHECK: rule inert without a seam or a cik column ===")


def test_the_history_build_applies_the_rule_before_the_event_ladder(sqlite_store: Any, monkeypatch, caplog):
    stored = _facts().assign(duration_type="quarterly", period_end=lambda f: f["period_of_report"])
    stored = stored.assign(**{column: None for column in FACT_COLUMNS if column not in stored.columns})
    sqlite_store.save(Tables.fundamentals_facts, stored.assign(fiscal_year=2015, period_days=91.0, is_amendment=False))
    seen: dict[str, pd.DataFrame] = {}

    def spy_build(ticker, facts, **kwargs):
        seen[ticker] = facts
        return TickerHistory(pd.DataFrame(), pd.DataFrame())

    monkeypatch.setattr(mod, "build_ticker", spy_build)
    monkeypatch.setattr(mod, "load_identity", lambda context: _IDENTITY)
    monkeypatch.setattr(mod, "record_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "get_entry", lambda *args, **kwargs: None)  # no manifest: no scope-change rebuild
    context = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.seam_rule"), config_dir="./configs")
    with caplog.at_level(logging.INFO, logger="test.seam_rule"):
        mod.build_fundamentals_history(context, ["GOOGL"])

    assert "cik" in FACT_COLUMNS
    assert sorted(seen["GOOGL"]["accession_number"].unique()) == ["alphabet-fy", "alphabet-stub", "google-q2", "google-q3"]
    assert "seam rule set aside 1 filing(s)" in caplog.text and "alphabet-q3" in caplog.text
    print("\n=== SANITY CHECK: build_fundamentals_history wiring ===")
    print("  the replay read `cik`, asked the identity layer for GOOGL's windows and set aside alphabet-q3 before the ladder")
