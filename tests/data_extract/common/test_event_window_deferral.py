"""P40: PLD, JCI and DD are exempt from P35's dated CIK windows for 8-K and 13D/13G (plan §13.10).

Their event filings union over every CIK of the entity, undated, as before P35 (Tyco's pre-merger 8-Ks stay JCI's),
in the listing, the schedule guards, the propagation purge, the validator and the gate; every other ticker keeps
the window rule. Offline: a real `Identity` over hand-written lineage rows (JCI exempt, MRK the control) and a
real SQLite `DataStore`; the exemption itself is read from the shipped config.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts import identity_regression_gate as gate
from src.constants.constants import SEC_8K_FORMS, SEC_13G_FORMS
from src.data_extract import identity_propagate as prop
from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp
from src.data_extract.utils.common.identity import Identity, build_identity, load_identity
from src.data_extract.utils.common.registrant import resolve_registrant_entries
from src.data_extract.utils.institutionals import schedule_rows
from src.data_extract.utils.institutionals.fetch_13g_edgar import SCHEDULE_13G
from src.data_store.schema import Tables
from src.validate.checks.identity import pending_removals

SENTINEL = "1900-01-01"
OLD_JCI, TYCO = "0000053669", "0000833444"  # JCI: Tyco's CIK is Johnson Controls International from 2016-09-02
MRK, OLD_MERCK = "0000310158", "0000064978"  # control: Schering-Plough's CIK is Merck & Co. from 2009-11-03
JCI_SEAM, MRK_SEAM = "2016-09-02", "2009-11-03"
ROSTER = {"JCI": TYCO, "MRK": MRK}
CHANGED = pd.Timestamp("2026-01-12 09:00")
RUN_DATE = pd.Timestamp("2026-01-13")
CONFIG_DIR = "configs"


def _lineage() -> pd.DataFrame:
    rows = [
        ("JCI", OLD_JCI, SENTINEL, JCI_SEAM),
        ("JCI", TYCO, JCI_SEAM, None),
        ("MRK", OLD_MERCK, SENTINEL, MRK_SEAM),
        ("MRK", MRK, MRK_SEAM, None),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": f"E{ROSTER[ticker]}",
                "canonical_ticker": ticker,
                "cik": cik,
                "role": "cik_window",
                "symbol": "",
                "valid_from": pd.Timestamp(start),
                "valid_to": pd.Timestamp(end) if end else pd.NaT,
                "status": "curated",
                "sources": "register",
                "oracle": "register",
                "confidence": None,
                "n_observations": 1,
                "evidence": "fixture",
                "scope_changed_at": CHANGED,
            }
            for ticker, cik, start, end in rows
        ]
    )


def _tenure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": t,
                "issuer_cik": c,
                "valid_from": pd.Timestamp("2000-01-01"),
                "valid_to": None,
                "n_filings": 5,
                "source": "form345",
                "evidence": "",
            }
            for t, c in ROSTER.items()
        ]
    )


def _roster() -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])


def _identity() -> Identity:
    return build_identity(lineage=_lineage(), tenure=_tenure(), roster=_roster(), undated_event_tickers=("JCI",))


def _index(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"cik": c, "company": "x", "form": f, "filed": d, "accession": a} for c, f, d, a in rows])


def _listed(identity: Identity, ticker: str, df_index: pd.DataFrame, forms: list[str]) -> list[str]:
    return resolve_registrant_entries(identity.filing_scope(ticker), df_index, forms)["accession"].tolist()


class _Context:
    """A weak-referenceable context (`load_identity` caches per context) over a real store and the shipped configs."""

    def __init__(self, store: Any, data_store: Path | None = None) -> None:
        self.store = store
        self.log = logging.getLogger("test.event_window_deferral")
        self.config_dir = CONFIG_DIR
        self.paths = {"DATA_STORE": data_store}
        self.config = types.SimpleNamespace(data_extract=types.SimpleNamespace(redundant_ticks=[], years_history=15))


# --------------------------------------------------------------------------- listing


def test_jci_lists_tycos_pre_merger_event_filings_and_mrk_keeps_the_window_rule() -> None:
    identity = _identity()
    df = _index(
        [
            (TYCO, "8-K", "2012-05-01", "tyco-2012"),
            (OLD_JCI, "8-K", "2015-05-01", "jci-2015"),
            (OLD_JCI, "8-K", "2018-05-01", "jci-late"),
            (TYCO, "10-K", "2012-11-15", "tyco-10k"),
            (OLD_JCI, "10-K", "2015-11-15", "jci-10k"),
            (MRK, "8-K", "2009-06-01", "sgp-2009"),
            (OLD_MERCK, "8-K", "2008-05-01", "merck-2008"),
        ]
    )
    assert identity.filing_scope("JCI").undated_events and not identity.filing_scope("MRK").undated_events
    assert _listed(identity, "JCI", df, SEC_8K_FORMS) == ["tyco-2012", "jci-2015", "jci-late"]
    assert _listed(identity, "JCI", df, ["10-K"]) == ["jci-10k"], "periodic forms keep their windows"
    assert _listed(identity, "MRK", df, SEC_8K_FORMS) == ["merck-2008"]
    print("\nsanity: JCI lists Tyco's 2012 and old JCI's 2018 8-Ks (undated union), its 10-Ks stay windowed; MRK's SGP pre-seam 8-K stays out")


def test_jcis_schedule_guards_accept_a_tyco_subject_before_the_merger(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    df = _index([(TYCO, "SC 13G", "2013-02-10", "g-tyco"), (OLD_MERCK, "SC 13G", "2011-02-11", "g-merck-late")])
    assert _listed(identity, "JCI", df, SEC_13G_FORMS) == ["g-tyco"]
    assert _listed(identity, "MRK", df, SEC_13G_FORMS) == []
    stamp = FilingStamp("g-tyco", "SC 13G", TYCO, pd.Timestamp("2013-02-10"), False, None, filing=types.SimpleNamespace())
    monkeypatch.setattr(schedule_rows, "header_subject_ciks", lambda filing: frozenset({TYCO}))
    monkeypatch.setattr(schedule_rows, "schedule_filing_rows", lambda stamp, spec: [{"cik": TYCO, "rp_seq": 0}])
    scope = EdgarScope(identity)
    assert schedule_rows.schedule_is_subject("JCI", TYCO, stamp, scope)
    assert [row["ticker"] for row in schedule_rows.schedule_ticker_rows("JCI", TYCO, stamp, scope, SCHEDULE_13G)] == ["JCI"]
    print("\nsanity: a 2013 13G on Tyco is JCI's (listed, subject and issuer guards accept); old Merck's 2011 13G is not MRK's")


def test_the_identity_reads_the_exemption_from_the_shipped_config(sqlite_store) -> None:
    sqlite_store.save(Tables.entity_lineage, _lineage())
    sqlite_store.save(Tables.symbol_tenure, _tenure().assign(evidence_period=""))
    sqlite_store.save(Tables.sp500_tickers, _roster().assign(name="x", sector="x", industry_group="x", sub_industry="x"))
    identity = load_identity(_Context(sqlite_store), refresh=True)
    assert identity.undated_event_tickers == frozenset({"PLD", "JCI", "DD"})
    assert identity.filing_scope("JCI").undated_events and not identity.filing_scope("MRK").undated_events
    print("\nsanity: load_identity takes PLD, JCI and DD from security_master_manual.json; MRK keeps its dated windows")


# --------------------------------------------------------------------------- stored rows


def _8k(ticker: str, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, "accession_number": a, "item": item, "cik": c, "filing_date": pd.Timestamp(f)}
            for a, c, f in rows
            for item in ("2.02", "9.01")
        ]
    )


JCI_8K = [("tyco-2012", TYCO, "2012-05-01"), ("jci-2015", OLD_JCI, "2015-05-01"), ("jci-late", OLD_JCI, "2018-05-01")]
MRK_8K = [("sgp-2009", MRK, "2009-06-01"), ("merck-2008", OLD_MERCK, "2008-05-01")]


def test_the_purge_and_the_validator_keep_jcis_undated_event_rows(sqlite_store, tmp_path, monkeypatch) -> None:
    context = _Context(sqlite_store, tmp_path)
    sqlite_store.save(Tables.sec_8k, pd.concat([_8k("JCI", JCI_8K), _8k("MRK", MRK_8K)], ignore_index=True))
    monkeypatch.setattr(prop, "_reparse_bulk", lambda context, changed, scope_ciks: {})
    monkeypatch.setattr(prop, "_refresh_insider", lambda context, identity, changed, dry_run: [])
    pending = pending_removals(context, _lineage(), list(ROSTER))

    result = prop.propagate_identity(context, list(ROSTER), identity=_identity(), as_of=RUN_DATE)

    kept = set(sqlite_store.load(Tables.sec_8k, columns=["accession_number"])["accession_number"])
    assert kept == {"tyco-2012", "jci-2015", "jci-late", "merck-2008"}
    removed = {(r.table, r.ticker, r.cik, r.rows) for r in result.removals.itertuples(index=False)}
    assert removed == {("sec_8k", "MRK", MRK, 2)}, removed
    assert {(r.table, r.ticker, r.cik, r.rows) for r in pending.itertuples(index=False)} == removed
    print("\nsanity: the purge and the validator keep Tyco's pre-merger and old JCI's post-merger 8-Ks; MRK's SGP pre-seam 8-K still goes")


def test_the_gate_reads_jcis_event_rows_as_undated() -> None:
    identity = _identity()
    scopes = {t: identity.filing_scope(t) for t in ROSTER}
    old = {t: frozenset(identity.filing_scope(t).event_ciks) for t in ROSTER}

    def frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=["ticker", "accession", "cik", "form", "filed"]).assign(period_end=None)
        out["filed"] = pd.to_datetime(out["filed"])
        return out

    before = frame([("JCI", "tyco-2012", TYCO, "8-K", "2012-05-01"), ("MRK", "sgp-2009", MRK, "8-K", "2009-06-01")])
    after = frame([("JCI", "tyco-2011", TYCO, "8-K", "2011-05-01")])
    diff = gate.filing_diff("sec_8k", before, after, scopes, old)
    reasons = {r.accession: r.reason for r in diff.itertuples(index=False)}
    assert reasons == {"tyco-2012": "", "sgp-2009": "outside_cik_window", "tyco-2011": "relisted_own_filing"}, reasons
    print(
        "\nsanity: a JCI Tyco 8-K leaving is unexplained (the exemption keeps it), one arriving is a relisted own filing; MRK's SGP removal is outside_cik_window"
    )
