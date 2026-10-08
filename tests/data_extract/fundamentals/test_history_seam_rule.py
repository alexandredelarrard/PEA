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


def test_a_cikless_frame_or_a_frame_with_no_window_is_untouched():
    facts = _facts()
    assert keep_window_owner_filings(facts, ()).equals(facts)
    assert keep_window_owner_filings(facts.drop(columns="cik"), _IDENTITY.filing_scope("GOOGL").windows).equals(facts.drop(columns="cik"))
    print("\n=== SANITY CHECK: rule inert without a window or a cik column ===")


def test_a_single_window_keeps_only_its_own_ciks_filings():
    """GOOGL cut to Google Inc's window alone: every Alphabet filing is from a CIK with no window, so it is set aside."""
    kept = keep_window_owner_filings(_facts(), _IDENTITY.filing_scope("GOOGL").windows[:1])
    assert list(dict.fromkeys(kept["accession_number"])) == ["google-q2", "google-q3"], list(kept["accession_number"])
    print("\n=== SANITY CHECK: one window ===")
    print("  Google Inc's two 10-Qs kept; Alphabet's three filings (no window in this frame) set aside")


#: PLD after the traded-security realignment: Prologis Inc (AMB) owns the ticker from the sentinel, old ProLogis is event-only.
PROLOGIS, OLD_PROLOGIS = "0001045609", "0000899881"
_PLD = dated_identity([("PLD", PROLOGIS, "cik_window", SENTINEL, None), ("PLD", OLD_PROLOGIS, "cik_event", SENTINEL, None)], {"PLD": PROLOGIS})


def _pld_facts() -> pd.DataFrame:
    filings = [
        ("old-prologis-q3-2010", OLD_PROLOGIS, "10-Q", "2010-11-09", "2010-09-30"),
        ("old-prologis-fy-2010", OLD_PROLOGIS, "10-K", "2011-02-28", "2010-12-31"),
        ("prologis-q2-2011", PROLOGIS, "10-Q", "2011-08-09", "2011-06-30"),
        ("prologis-fy-2011", PROLOGIS, "10-K", "2012-02-29", "2011-12-31"),
        ("no-cik", None, "10-Q", "2012-05-01", "2012-03-31"),
    ]
    return pd.DataFrame(
        [
            {
                "ticker": "PLD",
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


def test_an_event_only_cik_is_set_aside_for_a_single_window_ticker(sqlite_store: Any, monkeypatch, caplog):
    """H1 (AC-004): old ProLogis holds no window of PLD, so its 10-Q/10-K never feed PLD's history, although PLD has one window."""
    windows = _PLD.filing_scope("PLD").windows
    assert [w.cik for w in windows] == [PROLOGIS]

    kept = keep_window_owner_filings(_pld_facts(), windows)

    assert list(dict.fromkeys(kept["accession_number"])) == ["prologis-q2-2011", "prologis-fy-2011", "no-cik"], list(kept["accession_number"])
    stored = _pld_facts().assign(duration_type="quarterly", period_end=lambda f: f["period_of_report"])
    stored = stored.assign(**{column: None for column in FACT_COLUMNS if column not in stored.columns})
    sqlite_store.save(Tables.fundamentals_facts, stored.assign(fiscal_year=2011, period_days=91.0, is_amendment=False))
    seen: dict[str, pd.DataFrame] = {}

    def spy_build(ticker, facts, **kwargs):
        seen[ticker] = facts
        return TickerHistory(pd.DataFrame(), pd.DataFrame())

    monkeypatch.setattr(mod, "build_ticker", spy_build)
    monkeypatch.setattr(mod, "load_identity", lambda context: _PLD)
    context = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.seam_rule"), config_dir="./configs")
    with caplog.at_level(logging.INFO, logger="test.seam_rule"):
        mod.build_fundamentals_history(context, ["PLD"])

    assert set(seen["PLD"]["accession_number"]) == {"prologis-q2-2011", "prologis-fy-2011", "no-cik"}
    assert "seam rule set aside 2 filing(s)" in caplog.text and "old-prologis-q3-2010" in caplog.text
    print("\n=== SANITY CHECK: H1 event-only CIK, single window ===")
    print("  PLD (one window, Prologis Inc): old ProLogis's 2010 10-Q and 10-K set aside in the rule and in the history build;")
    print("  Prologis Inc's filings and the null-CIK row kept")


def test_an_ordinary_single_cik_tickers_facts_are_unchanged():
    """Regression: a ticker whose only filer is its window CIK keeps every row, whatever the filing date."""
    facts = _pld_facts()
    own = facts[facts["cik"].ne(OLD_PROLOGIS)].assign(filing_date=lambda f: f["filing_date"] - pd.DateOffset(years=30))

    kept = keep_window_owner_filings(own, _PLD.filing_scope("PLD").windows)

    assert kept.equals(own)
    print("\n=== SANITY CHECK: ordinary single-CIK ticker ===")
    print(f"  {len(own)} rows of the window CIK (filed 1981-1982, open window) and the null-CIK row: all kept, frame identical")


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
    context = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.seam_rule"), config_dir="./configs")
    with caplog.at_level(logging.INFO, logger="test.seam_rule"):
        mod.build_fundamentals_history(context, ["GOOGL"])

    assert "cik" in FACT_COLUMNS
    assert sorted(seen["GOOGL"]["accession_number"].unique()) == ["alphabet-fy", "alphabet-stub", "google-q2", "google-q3"]
    assert "seam rule set aside 1 filing(s)" in caplog.text and "alphabet-q3" in caplog.text
    print("\n=== SANITY CHECK: build_fundamentals_history wiring ===")
    print("  the replay read `cik`, asked the identity layer for GOOGL's windows and set aside alphabet-q3 before the ladder")


def test_f114_every_row_outside_its_window_leaves_an_empty_frame_with_its_columns():
    facts = _facts()
    late = facts[facts["accession_number"].eq("google-q2")].assign(filing_date=pd.Timestamp("2018-01-02"), accession_number="google-late")
    kept = keep_window_owner_filings(late, _IDENTITY.filing_scope("GOOGL").windows)
    assert kept.empty and list(kept.columns) == list(late.columns), list(kept.columns)
    assert kept["accession_number"].tolist() == []
    print("\n=== SANITY CHECK: F-114 all rows set aside ===")
    print("  Google Inc's 10-Q filed in 2018, past its window: no row kept, every column kept (no KeyError downstream)")


#: JCI after the realignment: Tyco (the survivor, renamed Johnson Controls International plc) owns the ticker from the
#: sentinel; old JCI, the accounting acquirer of the 2016-09-02 reverse acquisition, is event-only.
TYCO, OLD_JCI = "0000833444", "0000053669"
#: DD: Dow Chemical (accounting acquirer = the traded security) until 2017-08-31, then DowDuPont; not declared.
TDCC, DOWDUPONT = "0000029915", "0001666700"
_REVERSE = dated_identity(
    [
        ("JCI", TYCO, "cik_window", SENTINEL, None),
        ("JCI", OLD_JCI, "cik_event", SENTINEL, None),
        ("DD", TDCC, "cik_window", SENTINEL, "2017-08-31"),
        ("DD", DOWDUPONT, "cik_window", "2017-08-31", None),
    ],
    {"JCI": TYCO, "DD": DOWDUPONT},
)


def _comparative_facts() -> pd.DataFrame:
    """(ticker, accession, cik, form, filed, period_of_report, period_ends): one row per period a filing reports."""
    filings = [
        ("JCI", "tyco-fy2015", TYCO, "10-K", "2015-11-13", "2015-09-25", ("2015-09-25", "2014-09-26")),
        ("JCI", "old-jci-fy2015", OLD_JCI, "10-K", "2015-11-18", "2015-09-30", ("2015-09-30", "2014-09-30")),
        ("JCI", "jci-plc-fy2016", TYCO, "10-K", "2016-11-23", "2016-09-30", ("2016-09-30", "2015-09-30", "2014-09-30")),
        ("JCI", "tyco-fy2015-amendment", TYCO, "10-K/A", "2016-12-15", "2015-09-25", ("2015-09-25",)),
        ("JCI", "jci-plc-q1-2017", TYCO, "10-Q", "2017-02-08", "2016-12-31", ("2016-12-31", "2015-12-31")),
        ("DD", "tdcc-fy2016", TDCC, "10-K", "2017-02-10", "2016-12-31", ("2016-12-31", "2015-12-31")),
        ("DD", "dowdupont-fy2017", DOWDUPONT, "10-K", "2018-02-12", "2017-12-31", ("2017-12-31", "2016-12-31")),
    ]
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "cik": cik,
                "accession_number": accession,
                "form": form,
                "filing_date": pd.Timestamp(filed),
                "period_of_report": pd.Timestamp(period),
                "period_end": pd.Timestamp(end),
                "field": "totalRevenue",
                "value": 1.0,
            }
            for ticker, accession, cik, form, filed, period, ends in filings
            for end in ends
        ]
    )


def test_post_seam_comparatives_of_a_reverse_acquisition_are_ignored(sqlite_store: Any, monkeypatch, caplog):
    """REQ-014 / AC-017: JCI's post-seam 10-K/10-Q restate pre-seam periods with old JCI's numbers (the accounting
    acquirer), which are not the traded security's history; Tyco's own pre-seam filings and its pre-seam-period
    10-K/A filed after the seam are kept. DD is not a declared reverse acquisition, so its facts are unchanged."""
    facts = _comparative_facts()
    jci = facts[facts["ticker"].eq("JCI")]

    kept = mod.drop_reverse_acquisition_comparatives(jci, pd.Timestamp("2016-09-02"))
    dropped = jci.loc[~jci.index.isin(kept.index), ["accession_number", "period_end"]]
    assert sorted(map(tuple, dropped.astype(str).values)) == [
        ("jci-plc-fy2016", "2014-09-30"),
        ("jci-plc-fy2016", "2015-09-30"),
        ("jci-plc-q1-2017", "2015-12-31"),
    ], dropped

    stored = facts.assign(duration_type="annual")
    stored = stored.assign(**{column: None for column in FACT_COLUMNS if column not in stored.columns})
    sqlite_store.save(Tables.fundamentals_facts, stored.assign(fiscal_year=2016, period_days=364.0, is_amendment=False))
    seen: dict[str, pd.DataFrame] = {}

    def spy_build(ticker, frame, **kwargs):
        seen[ticker] = frame
        return TickerHistory(pd.DataFrame(), pd.DataFrame())

    monkeypatch.setattr(mod, "build_ticker", spy_build)
    monkeypatch.setattr(mod, "load_identity", lambda context: _REVERSE)
    context = SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.seam_rule"), config_dir="./configs")
    with caplog.at_level(logging.INFO, logger="test.seam_rule"):
        mod.build_fundamentals_history(context, ["JCI", "DD"])

    pairs = {(a, str(pd.Timestamp(e).date())) for a, e in zip(seen["JCI"]["accession_number"], seen["JCI"]["period_end"], strict=True)}
    assert pairs == {
        ("tyco-fy2015", "2015-09-25"),
        ("tyco-fy2015", "2014-09-26"),
        ("tyco-fy2015-amendment", "2015-09-25"),
        ("jci-plc-fy2016", "2016-09-30"),
        ("jci-plc-q1-2017", "2016-12-31"),
    }, pairs
    assert len(seen["DD"]) == int(facts["ticker"].eq("DD").sum()), "an undeclared ticker lost a fact"
    assert "reverse-acquisition rule dropped 3 pre-seam comparative fact(s)" in caplog.text
    assert "jci-plc-fy2016 (2)" in caplog.text and "jci-plc-q1-2017 (1)" in caplog.text
    print("\n=== SANITY CHECK: REQ-014 post-merger comparatives (JCI seam 2016-09-02) ===")
    print("  JCI plc FY2016 10-K: FY2015/FY2014 comparatives (old JCI's) dropped, its FY2016 kept; Q1-2017 10-Q: Dec-2015 quarter dropped")
    print("  Tyco's own FY2015 10-K and its 10-K/A filed after the seam (period of report 2015-09-25) kept; old JCI's 10-K set aside by H1")
    print(f"  DD (undeclared): all {len(seen['DD'])} facts kept, including DowDuPont's 2016 comparative")
