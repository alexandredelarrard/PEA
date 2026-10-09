"""JCI under the traded-security view: Tyco's CIK owns JCI from the sentinel start, old Johnson Controls is an
event-only CIK, and 8-K / 13D / 13G follow the dated CIK windows like every other ticker (no undated exemption).

Offline: a real `Identity` over hand-written lineage rows (JCI as the realignment builds it, MRK the control) and a
real SQLite `DataStore`; the shipped config no longer declares a traded-security deferral.
"""

from __future__ import annotations

import json
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
from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.registrant import resolve_registrant_entries
from src.data_extract.utils.institutionals import schedule_rows
from src.data_extract.utils.institutionals.fetch_13g_edgar import SCHEDULE_13G
from src.data_store.schema import Tables
from src.validate.checks.identity import pending_removals

SENTINEL = "1900-01-01"
OLD_JCI, TYCO = "0000053669", "0000833444"  # JCI's prices are Tyco's; old Johnson Controls is the acquired target
MRK, OLD_MERCK = "0000310158", "0000064978"  # control: Schering-Plough's CIK is Merck & Co. from 2009-11-03
MRK_SEAM = "2009-11-03"
ROSTER = {"JCI": TYCO, "MRK": MRK}
CHANGED = pd.Timestamp("2026-01-12 09:00")
RUN_DATE = pd.Timestamp("2026-01-13")
CONFIG_DIR = Path("configs")
REPO = Path(__file__).resolve().parents[3]


def _lineage() -> pd.DataFrame:
    rows = [
        ("JCI", TYCO, "cik_window", SENTINEL, None),
        ("JCI", OLD_JCI, "cik_event", None, None),
        ("MRK", OLD_MERCK, "cik_window", SENTINEL, MRK_SEAM),
        ("MRK", MRK, "cik_window", MRK_SEAM, None),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": f"E{ROSTER[ticker]}",
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": "",
                "valid_from": pd.Timestamp(start) if start else pd.NaT,
                "valid_to": pd.Timestamp(end) if end else pd.NaT,
                "status": "curated",
                "sources": "register" if role == "cik_window" else "manual",
                "oracle": "register" if role == "cik_window" else "manual",
                "confidence": None,
                "n_observations": 1,
                "evidence": "fixture",
                "scope_changed_at": CHANGED,
            }
            for ticker, cik, role, start, end in rows
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
    return build_identity(lineage=_lineage(), tenure=_tenure(), roster=_roster())


def _index(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"cik": c, "company": "x", "form": f, "filed": d, "accession": a} for c, f, d, a in rows])


def _listed(identity: Identity, ticker: str, df_index: pd.DataFrame, forms: list[str]) -> list[str]:
    return resolve_registrant_entries(identity.filing_scope(ticker), df_index, forms)["accession"].tolist()


class _Context:
    """A weak-referenceable context over a real store and the shipped configs."""

    def __init__(self, store: Any, data_store: Path | None = None) -> None:
        self.store = store
        self.log = logging.getLogger("test.event_window_traded_view")
        self.config_dir = str(CONFIG_DIR)
        self.paths = {"DATA_STORE": data_store}
        self.config = types.SimpleNamespace(data_extract=types.SimpleNamespace(redundant_ticks=[], years_history=15))


# --------------------------------------------------------------------------- no deferral left


def test_the_manual_config_declares_no_traded_security_deferral() -> None:
    """No manual-config section or README line of the shipped security manual declares a traded-security deferral."""
    manual = json.loads((REPO / CONFIG_DIR / "sec" / "security_master_manual.json").read_text(encoding="utf-8"))
    declared = [key for key in manual if "deferred" in key] + [line for line in manual["_README"] if line.startswith("deferred")]
    print("\n=== SANITY CHECK: traded-security deferral removed from the config ===")
    print(f"  declarations: {declared or 'none'}; manual sections: {sorted(k for k in manual if not k.startswith('_'))}")
    assert declared == []
    print("  OK: no config section or README line declares a deferral.")


# --------------------------------------------------------------------------- listing


def test_jci_lists_tycos_event_filings_by_window_and_drops_old_jcis() -> None:
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
    jci_8k, jci_10k, mrk_8k = (
        _listed(identity, "JCI", df, SEC_8K_FORMS),
        _listed(identity, "JCI", df, ["10-K"]),
        _listed(identity, "MRK", df, SEC_8K_FORMS),
    )
    print("\n=== SANITY CHECK: JCI listing under the traded view ===")
    print(f"  JCI 8-K {jci_8k}; JCI 10-K {jci_10k}; MRK 8-K {mrk_8k}")
    assert jci_8k == ["tyco-2012"], "old JCI is event-only: its 8-Ks leave the ticker"
    assert jci_10k == ["tyco-10k"], "and so do its periodic forms"
    assert mrk_8k == ["merck-2008"]
    print("  OK: Tyco's filings are JCI's from the sentinel start; old JCI contributes none; MRK keeps its windows.")


def test_jcis_schedule_guards_accept_a_tyco_subject_and_refuse_old_jci(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    df = _index(
        [
            (TYCO, "SC 13G", "2013-02-10", "g-tyco"),
            (OLD_JCI, "SC 13G", "2013-02-11", "g-old-jci"),
            (OLD_MERCK, "SC 13G", "2011-02-11", "g-merck-late"),
        ]
    )
    assert _listed(identity, "JCI", df, SEC_13G_FORMS) == ["g-tyco"]
    assert _listed(identity, "MRK", df, SEC_13G_FORMS) == []
    scope = EdgarScope(identity)
    stamp = FilingStamp("g-tyco", "SC 13G", TYCO, pd.Timestamp("2013-02-10"), False, None, filing=types.SimpleNamespace())
    monkeypatch.setattr(schedule_rows, "header_subject_ciks", lambda filing: frozenset({TYCO}))
    monkeypatch.setattr(schedule_rows, "schedule_filing_rows", lambda stamp, spec: [{"cik": TYCO, "rp_seq": 0}])
    tyco_ok = schedule_rows.schedule_is_subject("JCI", TYCO, stamp, scope)
    tyco_rows = [row["ticker"] for row in schedule_rows.schedule_ticker_rows("JCI", TYCO, stamp, scope, SCHEDULE_13G)]
    old = FilingStamp("g-old-jci", "SC 13G", OLD_JCI, pd.Timestamp("2013-02-11"), False, None, filing=types.SimpleNamespace())
    monkeypatch.setattr(schedule_rows, "header_subject_ciks", lambda filing: frozenset({OLD_JCI}))
    old_ok = schedule_rows.schedule_is_subject("JCI", OLD_JCI, old, scope)
    print("\n=== SANITY CHECK: JCI schedule guards ===")
    print(f"  Tyco 2013 13G subject={tyco_ok} rows={tyco_rows}; old JCI 2013 13G subject={old_ok}")
    assert tyco_ok and tyco_rows == ["JCI"]
    assert not old_ok
    print("  OK: a 2013 13G on Tyco is JCI's; one on old JCI (no window) is not; old Merck's 2011 13G is not MRK's.")


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


def test_the_purge_and_the_validator_drop_old_jcis_event_rows(sqlite_store, tmp_path, monkeypatch) -> None:
    context = _Context(sqlite_store, tmp_path)
    sqlite_store.save(Tables.sec_8k, pd.concat([_8k("JCI", JCI_8K), _8k("MRK", MRK_8K)], ignore_index=True))
    monkeypatch.setattr(prop, "_reparse_bulk", lambda context, changed, scopes, co_registrants: {})
    monkeypatch.setattr(prop, "_refresh_insider", lambda context, identity, changed, dry_run: [])
    pending = pending_removals(context, _lineage(), list(ROSTER))

    result = prop.propagate_identity(context, list(ROSTER), identity=_identity(), as_of=RUN_DATE)

    kept = set(sqlite_store.load(Tables.sec_8k, columns=["accession_number"])["accession_number"])
    removed = {(r.table, r.ticker, r.cik, r.rows) for r in result.removals.itertuples(index=False)}
    print("\n=== SANITY CHECK: purge and validator under the dated rule ===")
    print(f"  kept {sorted(kept)}; removed {sorted(removed)}")
    assert kept == {"tyco-2012", "merck-2008"}
    assert removed == {("sec_8k", "JCI", OLD_JCI, 4), ("sec_8k", "MRK", MRK, 2)}, removed
    assert {(r.table, r.ticker, r.cik, r.rows) for r in pending.itertuples(index=False)} == removed
    print("  OK: Tyco's 8-K stays JCI's; old JCI's 8-Ks (event-only CIK) and MRK's SGP pre-seam 8-K go, and the validator agrees.")


def test_the_gate_reads_jcis_event_rows_as_dated() -> None:
    identity = _identity()
    scopes = {t: identity.filing_scope(t) for t in ROSTER}
    old = {t: frozenset(identity.filing_scope(t).event_ciks) for t in ROSTER}

    def frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=["ticker", "accession", "cik", "form", "filed"]).assign(period_end=None)
        out["filed"] = pd.to_datetime(out["filed"])
        return out

    before = frame([("JCI", "jci-2015", OLD_JCI, "8-K", "2015-05-01"), ("MRK", "sgp-2009", MRK, "8-K", "2009-06-01")])
    after = frame([("JCI", "tyco-2011", TYCO, "8-K", "2011-05-01")])
    diff = gate.filing_diff("sec_8k", before, after, scopes, old)
    reasons = {r.accession: r.reason for r in diff.itertuples(index=False)}
    print("\n=== SANITY CHECK: gate reasons under the dated rule ===")
    print(f"  {reasons}")
    assert reasons == {"jci-2015": "acquired_constituent_purge", "sgp-2009": "outside_cik_window", "tyco-2011": "relisted_own_filing"}, reasons
    print(
        "  OK: old JCI's 8-K leaving is an acquired-constituent purge, a Tyco 8-K arriving a relisted own filing, MRK's SGP removal outside its window."
    )
