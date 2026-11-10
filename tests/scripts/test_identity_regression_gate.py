"""The explained-difference regression gate (P22, `tests/support/identity_regression_gate.py`).

Known-truth fixtures on the real CIKs and seams: ACE / old Chubb (event-only acquired target), Merck /
Schering-Plough (outside its window), Tyco / Johnson Controls (register window and seam margin) and the co-registrant
Digital Realty LP. Every filing reason rule has a row; a row no rule explains fails the gate; a hypothesis that does
not hold fails the gate. The traded-security realignment (TDCC moving DOW -> DD) exercises an insider row following
its CIK's entity. No market tape is diffed: the FTD and FINRA lines are stored as filed (P4, P5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import CikWindow, FilingScope
from src.data_store.schema import Tables
from src.utils.tickers_universe.universe import Universe
from tests.data_extract.fake_context import extract_config
from tests.support import identity_regression_gate as gate
from tests.support.universe import seed_universe, universe_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
ACE, OLD_CHUBB = "0000896159", "0000020171"
OLD_MERCK, SGP = "0000064978", "0000310158"
BRK, FITB, XOM, WBD, APTV, NWSA, FOXA, GOOGL = (
    "0001067983",
    "0000035527",
    "0000034088",
    "0001437107",
    "0001521332",
    "0001564708",
    "0001754301",
    "0001652044",
)
TYCO, JCI_INC, FOREIGN = "0000833444", "0000053669", "0000999999"
DLR, DLR_LP = "0001297996", "0001494877"


# --------------------------------------------------------------------------- the before-state fixtures


def _old_lineage() -> pd.DataFrame:
    """The pre-cutover membership table: old Chubb joins ACE's entity, Schering-Plough (later Merck) joins old Merck's."""
    rows = [(ACE, f"E{ACE}"), (OLD_CHUBB, f"E{ACE}"), (OLD_MERCK, f"E{OLD_MERCK}"), (SGP, f"E{OLD_MERCK}")]
    return pd.DataFrame([{"cik": c, "company_id": e, "source": "register", "confidence": None, "evidence": ""} for c, e in rows])


def _roster() -> pd.DataFrame:
    ciks = {"CB": ACE, "MRK": SGP, "BRK-B": BRK, "FITB": FITB, "XOM": XOM, "WBD": WBD, "APTV": APTV, "NWSA": NWSA, "FOXA": FOXA, "GOOGL": GOOGL}
    return pd.DataFrame([{"ticker": t, "cik": c} for t, c in ciks.items()])


def _tenure() -> pd.DataFrame:
    rows = [
        ("CB", OLD_CHUBB, "2006-01-01", "2016-01-15"),
        ("ACE", ACE, "2006-01-01", "2016-01-15"),
        ("CB", ACE, "2016-01-15", None),
        ("SGP", SGP, "2006-01-01", "2009-11-04"),
        ("MRK", OLD_MERCK, "2006-01-01", "2009-11-04"),
        ("MRK", SGP, "2009-11-04", None),
        ("BRK-B", BRK, "2006-01-01", None),
        ("BRKA", BRK, "2006-01-01", None),
        ("GOOG", GOOGL, "2006-01-01", None),
        ("GOOGL", GOOGL, "2006-01-01", None),
        ("FITB", FITB, "2006-01-01", None),
        ("FITBP", FITB, "2006-01-01", None),
        ("XOM", XOM, "2006-01-01", None),
        ("XOMZZZZ", XOM, "2026-07-01", None),
        ("DISCA", WBD, "2008-09-18", None),
        ("WBD", WBD, "2022-04-08", None),
    ]
    return pd.DataFrame(
        [{"symbol": s, "issuer_cik": c, "valid_from": f, "valid_to": t, "n_filings": 9, "source": "form345", "evidence": ""} for s, c, f, t in rows]
    )


# --------------------------------------------------------------------------- filing lineage


def _window(cik: str, start: str | None, end: str | None, margin_before: bool = False, margin_after: bool = False) -> CikWindow:
    lo, hi = (pd.Timestamp(start) if start else None), (pd.Timestamp(end) if end else None)
    pad = pd.Timedelta(days=31)
    return CikWindow(cik, lo, hi, lo - pad if lo is not None and margin_before else lo, hi + pad if hi is not None and margin_after else hi)


SCOPES = {
    "JCI": FilingScope(
        "JCI",
        f"E{JCI_INC}",
        TYCO,
        (JCI_INC, TYCO),
        (_window(JCI_INC, None, "2016-09-02", margin_after=True), _window(TYCO, "2016-09-02", None, margin_before=True)),
    ),
    "CB": FilingScope("CB", f"E{ACE}", ACE, (ACE, OLD_CHUBB), (_window(ACE, None, None),)),
}
OLD_CIKS = {"JCI": frozenset({TYCO}), "CB": frozenset({ACE, OLD_CHUBB})}


def _filings(rows: list[tuple[str, str, str, str, str, str | None]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ticker", "accession", "cik", "form", "filed", "period_end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    return frame


def test_every_filing_rule_explains_its_accession_and_an_own_filing_lost_is_unexplained() -> None:
    before_8k = _filings(
        [
            ("CB", "a-foreign", FOREIGN, "8-K", "2020-01-02", None),
            ("CB", "a-own-lost", ACE, "8-K", "2020-02-03", None),
            ("BRK.B", "a-moved", BRK, "8-K", "2020-03-02", None),
            ("CB", "a-kept", ACE, "8-K", "2020-06-01", None),
        ]
    )
    after_8k = _filings(
        [
            ("CB", "a-kept", ACE, "8-K", "2020-06-01", None),
            ("BRK-B", "a-moved", BRK, "8-K", "2020-03-02", None),
            ("CB", "a-relisted", ACE, "8-K", "2019-05-01", None),
            ("CB", "a-new", ACE, "8-K", "2020-07-01", None),
        ]
    )
    diff = gate.filing_diff("sec_8k", before_8k, after_8k, SCOPES, OLD_CIKS)
    got = {(r.accession, r.canonical_company): r.reason for r in diff.itertuples(index=False)}
    assert got == {
        ("a-foreign", "CB"): "foreign_filer_purge",
        ("a-own-lost", "CB"): "",  # E30: an own filing inside its window disappeared -- no rule
        ("a-moved", "BRK.B"): "symbol_normalisation",
        ("a-moved", "BRK-B"): "symbol_normalisation",
        ("a-relisted", "CB"): "relisted_own_filing",
        ("a-new", "CB"): "new_filing",
    }
    before_facts = _filings([("JCI", "t-2015", TYCO, "10-Q", "2015-05-01", "2015-03-31")])
    after_facts = _filings(
        [
            ("JCI", "j-2015", JCI_INC, "10-Q", "2015-07-30", "2015-06-30"),
            ("JCI", "j-margin", JCI_INC, "10-K", "2016-09-20", "2016-06-30"),
        ]
    )
    facts = gate.filing_diff("sec_fundamentals_facts", before_facts, after_facts, SCOPES, OLD_CIKS)
    assert {r.accession: r.reason for r in facts.itertuples(index=False)} == {
        "t-2015": "register_window",
        "j-2015": "register_window",
        "j-margin": "seam_margin",
    }
    insider = gate.filing_diff(
        "insider_transactions",
        _filings([("DLR", "lp-4", DLR_LP, "4", "2020-01-02", None)]),
        _filings([]),
        {"DLR": FilingScope("DLR", f"E{DLR}", DLR, (DLR, DLR_LP), (_window(DLR, None, None),))},
        {"DLR": frozenset({DLR, DLR_LP})},
        co_registrants={DLR_LP},
    )
    assert insider["reason"].tolist() == ["co_registrant_purge"]
    assert list(diff.columns[:10]) == [
        "canonical_company",
        "cik",
        "accession",
        "form",
        "accepted_at",
        "period_end",
        "old_decision",
        "new_decision",
        "reason",
        "evidence_accession",
    ]
    print(
        "sanity: foreign purge, co-registrant purge, register window (out and in), seam margin, symbol normalisation, a relisted and a new own filing each explain their accession; a lost own filing is unexplained"
    )


def test_the_history_view_names_the_seam_rule_set_asides_with_the_owner_accession() -> None:
    scopes = {
        "APO": FilingScope(
            "APO",
            "E1",
            "0000000002",
            ("0000000001", "0000000002"),
            (_window("0000000001", None, "2022-01-01", margin_after=True), _window("0000000002", "2022-01-01", None, margin_before=True)),
        )
    }
    facts = _filings(
        [
            ("APO", "old-10k", "0000000001", "10-K", "2022-01-20", "2021-12-31"),
            ("APO", "new-10k", "0000000002", "10-K", "2022-01-25", "2021-12-31"),
            ("APO", "old-late", "0000000001", "10-Q", "2023-03-01", "2022-12-31"),
            ("APO", "new-10q", "0000000002", "10-Q", "2022-05-01", "2022-03-31"),
        ]
    )
    history = gate.history_diff(facts, facts, scopes, pd.DataFrame(columns=gate.FILING_COLUMNS))
    got = {r.accession: (r.reason, r.evidence_accession, r.new_decision) for r in history.itertuples(index=False)}
    assert got == {"new-10k": ("same_period_rule", "old-10k", "set_aside"), "old-late": ("register_window", "", "set_aside")}
    print(
        "sanity: the successor's 10-K for a period its predecessor owns and reports is set aside with the predecessor's accession; a filing outside its window is set aside by the window"
    )


def test_f116_a_null_cik_is_no_filer_in_the_history_view() -> None:
    windows = (_window("0000000001", None, "2022-01-01", margin_after=True), _window("0000000002", "2022-01-01", None, margin_before=True))
    scopes = {"APO": FilingScope("APO", "E1", "0000000002", ("0000000001", "0000000002"), windows)}
    facts = _filings(
        [
            ("APO", "old-late", "0000000001", "10-Q", "2023-03-01", "2022-12-31"),  # outside its window: the seam rule would set it aside
            ("APO", "null-cik", None, "10-Q", "2022-05-01", "2022-03-31"),
        ]
    )
    history = gate.history_diff(facts, facts, scopes, pd.DataFrame(columns=gate.FILING_COLUMNS))
    assert history.empty, history[["accession", "reason"]]
    print("sanity: one filer CIK plus null-CIK rows: the seam rule is not simulated (the build skips it), so nothing is set aside")


# --------------------------------------------------------------------------- insider, merged fundamentals, prices


def test_insider_rows_leave_canonical_history_only_by_role_or_purge() -> None:
    key = ["accession_number", "security_type", "row_sequence"]
    before = pd.DataFrame(
        [
            ("acc-chubb", "nonderivative", 1, "CB", OLD_CHUBB, "2014-03-01"),
            ("acc-lp", "nonderivative", 1, "DLR", DLR_LP, "2015-03-01"),
            ("acc-ace", "nonderivative", 1, "CB", ACE, "2015-04-01"),
            ("acc-lost", "nonderivative", 1, "CB", ACE, "2015-05-01"),
        ],
        columns=[*key, "ticker", "issuer_cik", "filing_date"],
    )
    after = pd.DataFrame(
        [
            ("acc-chubb", "nonderivative", 1, "CB", OLD_CHUBB, "2014-03-01", "acquired_constituent", "2014-02-27"),
            ("acc-ace", "nonderivative", 1, "CB", ACE, "2015-04-01", "canonical_current", "2015-03-30"),
            ("acc-new", "nonderivative", 1, "CB", ACE, "2026-01-02", "canonical_current", "2025-12-30"),
        ],
        columns=[*key, "ticker", "issuer_cik", "filing_date", "lineage_role", "economic_date"],
    )
    diff = gate.insider_diff(before, after, {"CB": frozenset({ACE, OLD_CHUBB}), "DLR": frozenset({DLR, DLR_LP})}, {DLR_LP})
    assert {r.accession_number: r.reason for r in diff.itertuples(index=False)} == {
        "acc-chubb": "acquired_constituent",
        "acc-lp": "co_registrant_purge",
        "acc-lost": "",
        "acc-new": "new_filing",
    }
    print(
        "sanity: old Chubb's row leaves canonical history by role, DLR LP's is purged as a co-registrant, a new filing is new, and a lost canonical row is unexplained"
    )


def test_merged_rows_change_only_inside_a_predecessor_window_with_a_stored_owner_series() -> None:
    def frame(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=["ticker", "as_of", "fiscal_end", "totalRevenue"])
        return out.assign(as_of=pd.to_datetime(out["as_of"]), fiscal_end=pd.to_datetime(out["fiscal_end"]))

    before = frame(
        [
            ("PLD", "2011-02-10", "2010-12-31", 164.5),
            ("PLD", "2011-08-09", "2011-06-30", 170.0),
            ("PLD", "2012-05-01", "2012-03-31", 400.0),
            ("PLD", "2012-08-01", "2012-06-30", 410.0),
            ("MRK", "2015-05-01", "2015-03-31", 9.0),
        ]
    )
    after = frame(
        [
            ("PLD", "2011-02-10", "2010-12-31", 238.8),
            ("PLD", "2011-08-09", "2011-06-30", 171.0),
            ("PLD", "2012-05-01", "2012-03-31", 400.0),
            ("PLD", "2012-08-01", "2012-06-30", 411.0),
            ("PLD", "2026-08-01", "2026-06-30", 900.0),
            ("MRK", "2015-05-01", "2015-03-31", 9.5),
        ]
    )
    windows = [gate.Window("PLD", "PLD1", None, pd.Timestamp("2011-06-03"))]
    diff = gate.merged_diff(before, after, windows)
    assert {(r.ticker, r.fiscal_end): r.reason for r in diff.itertuples(index=False)} == {
        ("PLD", "2010-12-31"): "predecessor_vendor_series",
        ("PLD", "2011-06-30"): "predecessor_vendor_series",  # trailing quarters after the window
        ("PLD", "2012-06-30"): "",  # a vendor change more than a year after the window: no rule
        ("PLD", "2026-06-30"): "new_period",
        ("MRK", "2015-03-31"): "",
    }
    print(
        "sanity: PLD's AMB-era quarter is replaced by old ProLogis's series, a quarter within a year after it changes through trailing figures, an unchanged row is not listed, a new period is explained, a vendor change elsewhere is not"
    )


def test_prices_may_only_gain_secondary_class_symbols_and_new_dates() -> None:
    def frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=["ticker", "date", "close_split"])
        return out.assign(date=pd.to_datetime(out["date"]))

    before = frame([("BRK-B", "2026-01-02", 480.0), ("BRK-B", "2026-01-05", 482.0)])
    after = frame(
        [
            ("BRK-B", "2026-01-02", 480.0),
            ("BRK-B", "2026-01-05", 481.0),
            ("BRK-B", "2026-01-06", 483.0),
            ("BRK-A", "2026-01-02", 720000.0),
            ("ZZZ", "2026-01-02", 1.0),
        ]
    )
    diff = gate.prices_diff(before, after, {"BRK-A"})
    assert {(r.ticker, r.date): r.reason for r in diff.itertuples(index=False)} == {
        ("BRK-B", "2026-01-05"): "",
        ("BRK-B", "2026-01-06"): "new_data",
        ("BRK-A", "2026-01-02"): "secondary_class_added",
        ("ZZZ", "2026-01-02"): "",
    }
    print(
        "sanity: BRK-A rows are a secondary-class addition and a later BRK-B date is new; a revised BRK-B close and an unknown symbol are unexplained"
    )


# --------------------------------------------------------------------------- hypotheses


def test_a_hypothesis_that_does_not_hold_fails_and_a_skipped_one_is_reported() -> None:
    filing = gate.filing_diff(
        "sec_fundamentals_facts", _filings([("JCI", "t-2015", TYCO, "10-Q", "2015-05-01", "2015-03-31")]), _filings([]), SCOPES, OLD_CIKS
    )
    filing["table"] = gate.HISTORY_TABLE
    hypotheses = [
        {"id": "tyco", "kind": "filing_lineage", "tickers": ["JCI"], "change": "removed", "expected": 1, "reason": "register_window"},
        {"id": "tyco_wrong", "kind": "filing_lineage", "tickers": ["JCI"], "change": "removed", "expected": 2, "reason": "register_window"},
        {"id": "tyco_any", "kind": "filing_lineage", "tickers": ["JCI"], "change": "removed", "expected": None, "reason": "register_window"},
        {"id": "dd", "kind": "filing_lineage", "tickers": ["DD"], "change": "added", "expected": 30, "reason": "register_window"},
    ]
    result = gate.check_hypotheses(hypotheses, filing, skip={"dd"})
    assert dict(zip(result["id"], zip(result["observed"], result["status"], strict=True), strict=True)) == {
        "tyco": (1, "pass"),
        "tyco_wrong": (1, "fail"),
        "tyco_any": (1, "pass"),
        "dd": (0, "skipped"),
    }
    shipped = [h for h in gate.load_hypotheses(CONFIGS) if h["kind"] not in gate.RETIRED_KINDS]
    assert all(h["reason"] in gate.REASONS for h in shipped), "every checked shipped hypothesis names a gate reason"
    print(f"sanity: a matching count passes, a wrong count fails, a skipped id is reported; the {len(shipped)} shipped hypotheses name gate reasons")


# --------------------------------------------------------------------------- end to end through the store


class _Context:
    def __init__(self, store: Any) -> None:
        self.store = store
        self.config_dir = str(CONFIGS)
        self.config = extract_config(data_extract={"years_history": 17})
        self.log = logging.getLogger("gate-test")

    @property
    def universe(self) -> Universe:
        """The stored `universe_tickers`, with GOOG as GOOGL's alias when GOOGL is in it."""
        tickers = self.store.load(Tables.universe_tickers, columns=["ticker"])["ticker"].tolist()
        return Universe(self.store, universe_config(tickers, {"GOOGL": ["GOOG"]} if "GOOGL" in tickers else None))


def _dated_lineage() -> pd.DataFrame:
    base = {
        "symbol": "",
        "valid_to": None,
        "status": "curated",
        "sources": "roster",
        "oracle": "fixture",
        "confidence": None,
        "n_observations": 1,
        "evidence": "",
        "scope_changed_at": pd.Timestamp("2026-10-01"),
    }
    rows = [("CB", ACE, "cik_window"), ("CB", OLD_CHUBB, "cik_event")] + [
        (t, c, "cik_window") for t, c in _roster().itertuples(index=False) if t != "CB"
    ]
    return pd.DataFrame(
        [
            {**base, "company_id": f"E{ACE}" if t == "CB" else f"E{c}", "canonical_ticker": t, "cik": c, "role": role, "valid_from": "1900-01-01"}
            for t, c, role in rows
        ]
    )


def _seed_before(store: Any) -> None:
    _old_lineage().to_sql(Tables.universe_cik_symbol.name, store.engine, index=False)  # the old shape: SQLite refuses the dated PK on it
    store.save(Tables.symbol_tenure, _tenure().assign(evidence_period=""))
    seed_universe(store, _roster()["ticker"].tolist(), rows={t: {"cik": c} for t, c in _roster().itertuples(index=False)})
    store.save(
        Tables.sec_8k,
        pd.DataFrame(
            {
                "ticker": ["CB", "CB"],
                "cik": [FOREIGN, ACE],
                "accession_number": ["a-foreign", "a-own"],
                "form": ["8-K", "8-K"],
                "filing_date": ["2020-01-02", "2020-02-03"],
                "item": ["2.02", "2.02"],
            }
        ),
    )


def _master() -> pd.DataFrame:
    """The after state's `universe_cusip` lines (the prices section reads its secondary-class symbols)."""
    rows = [
        ("C171232101", "acquired_constituent", "event_only_cik", 1.0, OLD_CHUBB, "2009-06-26", "2016-01-15"),
        ("C806605101", "acquired_constituent", "outside_window", 1.0, SGP, "2009-06-26", "2009-11-04"),
        ("C084670108", "secondary_class", "class_description", 1500.0, BRK, "2010-01-21", None),
        ("C38259P706", "secondary_class", "class_description", 1.0, GOOGL, "2014-04-03", None),
        ("C316773209", "excluded", "preferred", 1.0, FITB, "2009-06-26", None),
        ("C30233Q108", "excluded", "transition_placeholder", 1.0, XOM, "2026-07-01", "2026-07-02"),
        ("C25470F104", "excluded", "superseded", 1.0, WBD, "2022-04-07", "2022-04-12"),
        ("CG27823106", "canonical_current", "manual_boundary", 1.0, APTV, "2011-11-17", "2017-12-05"),
        ("C65249B109", "canonical_current", "ticker_symbol", 1.0, NWSA, "2013-07-01", None),
        ("C35137L204", "secondary_class", "class_description", 1.0, FOXA, "2019-03-19", None),
        ("C14040H105", "canonical_current", "ticker_symbol", 1.0, "0000927628", "2009-06-26", None),
    ]
    return pd.DataFrame(
        [
            {
                "security_id": s,
                "valid_from": f,
                "valid_to": t,
                "lineage_role": r,
                "lineage_reason": why,
                "conversion_ratio": x,
                "issuer_cik": c,
                "exchange": "NYSE",
                "market_symbol": "",
            }
            for s, r, why, x, c, f, t in rows
        ]
    )


def _make_after(store: Any, *, lose_own: bool) -> None:
    store.drop(Tables.universe_cik_symbol)
    store.save(Tables.universe_cik_symbol, _dated_lineage())
    master = _master().assign(
        canonical_company=None,
        source="ftd",
        source_symbol="",
        market_symbol="",
        cusip=lambda f: f["security_id"].str[1:],
        security_class="common",
        source_accession=None,
        evidence="",
        n_observations=1,
        scope_changed_at=pd.Timestamp("2026-10-01"),
    )
    store.save(Tables.universe_cusip, master)
    store.delete(Tables.sec_8k, {"accession_number": "a-foreign"})
    if lose_own:
        store.delete(Tables.sec_8k, {"accession_number": "a-own"})


@pytest.mark.parametrize("lose_own", [False, True])
def test_snapshot_reads_only_and_diff_exits_non_zero_on_an_unexplained_row(sqlite_store: Any, tmp_path: Path, lose_own: bool) -> None:
    _seed_before(sqlite_store)
    counts = gate.take_snapshot(gate.ReadOnlyStore(sqlite_store), tmp_path / "before")
    assert counts["filing_sec_8k"] == 2 and "ftd" not in counts, "the FTD lines carry no ticker grain to snapshot"
    _make_after(sqlite_store, lose_own=lose_own)
    skip = {h["id"] for h in gate.load_hypotheses(CONFIGS)}
    code = gate.run_diff(_Context(gate.ReadOnlyStore(sqlite_store)), tmp_path / "before", tmp_path / "out", skip=skip)
    filing = pd.read_csv(tmp_path / "out" / gate.OUTPUTS["filing"], dtype=str, keep_default_na=False)
    assert list(filing.columns) == gate.FILING_COLUMNS and "tape" not in gate.OUTPUTS
    assert (tmp_path / "out" / "gate_summary.txt").is_file() and (tmp_path / "out" / "gate_hypotheses.csv").is_file()
    sec_8k = filing[filing["table"].eq("sec_8k")]
    expected = {"a-foreign": "foreign_filer_purge", **({"a-own": ""} if lose_own else {})}
    assert dict(zip(sec_8k["accession"], sec_8k["reason"], strict=True)) == expected
    assert code == (1 if lose_own else 0)
    print(
        f"sanity: snapshot made no write; diff exit {code} with {'an unexplained own 8-K' if lose_own else 'every row explained'} (foreign 8-K purged; no stamped tape row)"
    )


def test_hypotheses_config_round_trips(tmp_path: Path) -> None:
    data = json.loads((CONFIGS / gate.HYPOTHESES_FILE).read_text(encoding="utf-8"))
    ids = [h["id"] for h in data["hypotheses"]]
    assert len(ids) == len(set(ids)) and all(h["kind"] in gate.HYPOTHESIS_KINDS or h["kind"] in gate.RETIRED_KINDS for h in data["hypotheses"])
    print(f"sanity: {len(ids)} hypotheses, unique ids, known kinds")


def test_the_cli_computes_only_the_requested_sections_and_reads_only(sqlite_store: Any, tmp_path: Path) -> None:
    _seed_before(sqlite_store)
    assert gate.main(["snapshot", str(tmp_path / "before")], context_factory=lambda c, u: _Context(sqlite_store)) == 0
    _make_after(sqlite_store, lose_own=True)
    ids = [h["id"] for h in gate.load_hypotheses(CONFIGS)]
    code = gate.main(
        ["diff", str(tmp_path / "before"), "-o", str(tmp_path / "out"), "--sections", "insider", "--skip-hypothesis", *ids],
        context_factory=lambda c, u: _Context(sqlite_store),
    )
    assert code == 0, "the lost own 8-K is in the filing section, which was not requested"
    assert sorted(p.name for p in (tmp_path / "out").glob("gate_*")) == ["gate_hypotheses.csv", "gate_insider_diff.csv", "gate_summary.txt"]
    hypotheses = pd.read_csv(tmp_path / "out" / "gate_hypotheses.csv")
    assert set(hypotheses["kind"]) == {"insider"} and set(hypotheses["status"]) == {"skipped"}
    with pytest.raises(PermissionError):
        gate.ReadOnlyStore(sqlite_store).save(Tables.prices, pd.DataFrame({"ticker": ["X"], "date": ["2026-01-02"]}))
    print(
        "sanity: the CLI diff of the insider section alone writes its file, the summary and only the insider hypotheses; the wrapped store refuses writes"
    )


def test_f112_the_cli_splits_commas_refuses_a_scope_outside_the_snapshot_and_skips_out_of_scope_hypotheses(sqlite_store: Any, tmp_path: Path) -> None:
    _seed_before(sqlite_store)
    factory = lambda c, u: _Context(sqlite_store)  # noqa: E731
    assert gate.main(["snapshot", str(tmp_path / "before"), "-t", "CB,MRK"], context_factory=factory) == 0
    meta = json.loads((tmp_path / "before" / "snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["tickers"] == ["CB", "MRK"], "-t CB,MRK is two tickers"
    _make_after(sqlite_store, lose_own=False)
    out = tmp_path / "out"
    assert gate.main(["diff", str(tmp_path / "before"), "-o", str(out), "--sections", "insider"], context_factory=factory) == 2, (
        "unscoped diff of a scoped snapshot"
    )
    assert gate.main(["diff", str(tmp_path / "before"), "-o", str(out), "-t", "CB,AAPL", "--sections", "insider"], context_factory=factory) == 2
    code = gate.main(["diff", str(tmp_path / "before"), "-o", str(out), "-t", "CB", "--sections", "insider"], context_factory=factory)
    hypotheses = pd.read_csv(out / "gate_hypotheses.csv")
    assert code != 2 and set(hypotheses.loc[hypotheses["tickers"].ne("CB"), "status"]) == {"skipped"}, hypotheses
    print("sanity: -t CB,MRK snapshots two tickers; a diff outside that scope exits 2; a CB diff skips every hypothesis naming another ticker")


# --------------------------------------------------------------------------- the traded-security realignment (stamped before view)

TDCC, DOW_INC, DOWDUPONT = "0000029915", "0001751788", "0001666700"


def test_an_insider_row_follows_its_cik_to_the_entity_that_now_holds_it() -> None:
    key = ["accession_number", "security_type", "row_sequence"]
    before = pd.DataFrame(
        [("acc-tdcc", "nonderivative", 1, "DOW", TDCC, "2015-03-01"), ("acc-dow", "nonderivative", 1, "DOW", DOW_INC, "2020-03-01")],
        columns=[*key, "ticker", "issuer_cik", "filing_date"],
    )
    after = pd.DataFrame(
        [
            ("acc-tdcc", "nonderivative", 1, "DD", TDCC, "2015-03-01", "canonical_predecessor", "2015-02-27"),
            ("acc-dow", "nonderivative", 1, "DD", DOW_INC, "2020-03-01", "canonical_current", "2020-02-27"),
        ],
        columns=[*key, "ticker", "issuer_cik", "filing_date", "lineage_role", "economic_date"],
    )
    own = {"DD": frozenset({DOWDUPONT, TDCC}), "DOW": frozenset({DOW_INC})}
    diff = gate.insider_diff(before, after, own, ())
    assert {r.accession_number: r.reason for r in diff.itertuples(index=False)} == {"acc-tdcc": "entity_moved", "acc-dow": ""}
    hypothesis = {"id": "x", "kind": "insider", "tickers": ["DOW"], "ciks": [TDCC], "change": "moved", "expected": 1, "reason": "entity_moved"}
    empty = pd.DataFrame(columns=gate.FILING_COLUMNS)
    result = gate.check_hypotheses([hypothesis], empty, insider=diff)
    assert result[["observed", "status"]].values.tolist() == [[1, "pass"]]
    print(
        "sanity: TDCC's row moves DOW -> DD with its CIK's entity (entity_moved, counted by an insider hypothesis); Dow Inc's own row moving is unexplained"
    )


def test_the_shipped_hypotheses_describe_the_traded_security_realignment_only() -> None:
    hypotheses = json.loads((CONFIGS / gate.HYPOTHESES_FILE).read_text(encoding="utf-8"))["hypotheses"]
    assert hypotheses and all(h["change_set"] == "traded_security_realignment" for h in hypotheses)
    named = {t for h in hypotheses for t in h["tickers"]}
    assert named == {"PLD", "JCI", "DD", "DOW"}, named
    checked = [h for h in hypotheses if h["kind"] not in gate.RETIRED_KINDS]
    assert checked and all(h["reason"] in gate.REASONS and h["kind"] in gate.HYPOTHESIS_KINDS for h in checked)
    by_kind = pd.Series([h["kind"] for h in hypotheses]).value_counts().to_dict()
    print(f"sanity: {len(hypotheses)} hypotheses {by_kind}, all under traded_security_realignment, naming only {sorted(named)}")
