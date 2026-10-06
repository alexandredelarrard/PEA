"""Tier C: the CIK-keyed bulk data sets resolve rows through the dated `entity_lineage` only.

`notes_num`/`notes_text` and `pension_facts` are CONSOLIDATING: a row belongs to a ticker when its
filer CIK's seam-widened window holds the filing date (`ticker_for_cik(..., "consolidating")`).
`insider_transactions` (Forms 3/4/5) are EVENTS: any CIK of the entity (`"event"`), no date filter.
No register JSON is read: the fixtures' config directory holds none, so a predecessor resolves only
through its lineage window.

Synthetic fixtures: these are resolution rules, not measurements.
"""

from __future__ import annotations

import ast
import io
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import fetch_financial_notes as fn
from src.data_extract.utils.fundamentals import fetch_financial_statements as fin
from src.data_extract.utils.institutionals.insider_common import screen_insider_rows
from src.data_store.schema import Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

#: GOOGL (Google Inc -> Alphabet, 2015-10-02), VTRS (Mylan -> Viatris, 2020-11-07), APA (Apache -> APA Corp, 2021-03-01).
IDENTITY = dated_identity(
    [
        ("GOOGL", "0001288776", "cik_window", SENTINEL, "2015-10-02"),
        ("GOOGL", "0001652044", "cik_window", "2015-10-02", None),
        ("VTRS", "0001623613", "cik_window", SENTINEL, "2020-11-07"),
        ("VTRS", "0001792044", "cik_window", "2020-11-07", None),
        ("APA", "0000006769", "cik_window", SENTINEL, "2021-03-01"),
        ("APA", "0001841666", "cik_window", "2021-03-01", None),
        ("AAPL", "0000320193", "cik_window", SENTINEL, None),
    ],
    {"GOOGL": "0001652044", "VTRS": "0001792044", "APA": "0001841666", "AAPL": "0000320193"},
)
ROSTER = pd.DataFrame({"ticker": ["GOOGL", "VTRS", "APA", "AAPL"], "cik": ["0001652044", "0001792044", "0001841666", "0000320193"]})
UNIVERSE = ["AAPL", "APA", "GOOGL", "VTRS"]

#: (accession, filer CIK, filed, expected ticker or None)
FILINGS = [
    ("g-inside", "1288776", "20150728", "GOOGL"),  # Google Inc inside its window
    ("g-margin", "1288776", "20151029", "GOOGL"),  # Google Inc 27 days past the seam: the margin admits it
    ("g-after", "1288776", "20151215", None),  # Google Inc past window + margin
    ("m-inside", "1623613", "20200807", "VTRS"),  # Mylan N.V. before the Viatris seam
    ("a-sub", "6769", "20220501", None),  # Apache Corp as a subsidiary, a year past its window
    ("a-parent", "1841666", "20220501", "APA"),
    ("aapl", "320193", "20241101", "AAPL"),
]
EXPECTED = {adsh: ticker for adsh, _, _, ticker in FILINGS if ticker is not None}

_SUB = ["adsh", "cik", "name", "form", "period", "fy", "fp", "filed"]
_NUM = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "dimn", "coreg", "value", "footnote"]
_TXT = ["adsh", "tag", "version", "ddate", "qtrs", "dimn", "coreg", "escaped", "txtlen", "footnote", "value"]


def _tsv(columns: list[str], rows: list[dict[str, str]]) -> str:
    return "\n".join(["\t".join(columns), *("\t".join(row.get(c, "") for c in columns) for row in rows)])


def _notes_zip() -> bytes:
    sub = [{"adsh": a, "cik": c, "name": a, "form": "10-Q", "fy": "2020", "fp": "Q2", "filed": f} for a, c, f, _ in FILINGS]
    num = [
        {"adsh": a, "tag": "DefinedBenefitPlanBenefitObligation", "ddate": f"{f[:4]}0630", "qtrs": "0", "uom": "USD", "dimn": "0", "value": "100"}
        for a, _, f, _ in FILINGS
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("sub.tsv", _tsv(_SUB, sub))
        archive.writestr("num.tsv", _tsv(_NUM, num))
        archive.writestr("txt.tsv", _tsv(_TXT, []))
    return buf.getvalue()


def _context(store: Any, tmp_path) -> SimpleNamespace:
    """A context whose config directory holds no registrant register."""
    return SimpleNamespace(
        store=store,
        config_dir=str(tmp_path / "no_register"),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(financial_notes="unused", financial_statements="unused"))),
    )


def _patch_bulk_io(module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Cache, clocks and bookkeeping stubbed; identity and roster from the fixtures above."""
    monkeypatch.setattr(module, "load_identity", lambda context: IDENTITY, raising=False)
    monkeypatch.setattr(module, "load_cik_mapping", lambda context: ROSTER.copy(), raising=False)
    monkeypatch.setattr(module, "cache_dir", lambda context, key: tmp_path)
    monkeypatch.setattr(module, "ensure_zip", lambda context, path, url, **kwargs: path)
    monkeypatch.setattr(module, "is_cached", lambda path: True)
    monkeypatch.setattr(module, "archive_available_at", lambda *args, **kwargs: date(2025, 1, 13))
    monkeypatch.setattr(module, "stored_period_clock", lambda *args, **kwargs: None)


def test_notes_rows_resolve_through_the_lineage_window_of_their_filer(sqlite_store, monkeypatch, tmp_path):
    """A predecessor's notes row filed inside its window (or its seam margin) is the ticker's; outside, nobody's."""
    period = "2020q3"
    (tmp_path / f"{period}_notes.zip").write_bytes(_notes_zip())
    _patch_bulk_io(fn, monkeypatch, tmp_path)
    monkeypatch.setattr(fn, "_repair_stored_clocks", lambda *args, **kwargs: {})
    monkeypatch.setattr(fn, "_notes_periods", lambda context, years_history, today=None: [period])

    fn.fetch_financial_notes(_context(sqlite_store, tmp_path), UNIVERSE, years_history=1)

    stored = sqlite_store.load(Tables.notes_num, columns=["adsh", "ticker"])
    got = dict(zip(stored["adsh"], stored["ticker"], strict=True))
    assert got == EXPECTED, f"notes_num tickers by accession: {got}"
    print("\n=== SANITY CHECK: notes rows by lineage window ===")
    print(f"  kept {sorted(got)}; Google Inc past its margin and Apache Corp's subsidiary-era filing resolve to no ticker. Validated.")


def test_pension_rows_resolve_through_the_lineage_window_of_their_filer(sqlite_store, monkeypatch, tmp_path):
    """Same consolidating rule for `pension_facts`."""
    facts = pd.DataFrame(
        [
            {
                "cik": c.zfill(10),
                "tag": "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
                "ddate": pd.Timestamp(f),  # one fact per filing: the pension key holds no accession
                "qtrs": 0,
                "uom": "USD",
                "value": 1.0,
                "adsh": a,
                "filed": pd.Timestamp(f),
                "form": "10-Q",
                "fy": "2020",
                "fp": "Q2",
            }
            for a, c, f, _ in FILINGS
        ]
    )
    _patch_bulk_io(fin, monkeypatch, tmp_path)
    monkeypatch.setattr(fin, "quarter_periods", lambda *args: ["2020q3"])
    monkeypatch.setattr(fin, "_read_pension_facts", lambda path: facts.copy())

    fin.fetch_financial_statements(_context(sqlite_store, tmp_path), UNIVERSE, years_history=1)

    stored = sqlite_store.load(Tables.pension_facts, columns=["adsh", "ticker"])
    got = dict(zip(stored["adsh"], stored["ticker"], strict=True))
    assert got == EXPECTED, f"pension_facts tickers by accession: {got}"
    print("\n=== SANITY CHECK: pension rows by lineage window ===")
    print(f"  kept {sorted(got)}; out-of-window predecessor filings dropped. Validated.")


def test_insider_rows_of_any_entity_cik_resolve_without_a_date_filter():
    """Forms 3/4/5 are events: Google Inc's and Apache Corp's filings belong to the ticker whatever the date."""
    df = pd.DataFrame(
        {
            "ticker": ["GOOG", None, "APA", None],
            "issuer_cik": ["0001288776", "0001288776", "0000006769", "0001623613"],
            "filing_date": pd.to_datetime(["2015-07-28", "2015-12-15", "2022-05-01", "2020-08-07"]),
        }
    )
    kept, rejected = screen_insider_rows(df, UNIVERSE, IDENTITY)

    assert kept["ticker"].tolist() == ["GOOGL", "GOOGL", "APA", "VTRS"]
    assert kept["claimed_ticker"].tolist()[0] == "GOOG"
    assert rejected.empty
    print("\n=== SANITY CHECK: insider rows by event CIK ===")
    print("  predecessor Form 4s (Google Inc, Mylan, Apache Corp post-seam) all keep their ticker. Validated.")


# --------------------------------------------------------------------------- #
# AC-027: one identity source for every fetcher                                #
# --------------------------------------------------------------------------- #
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
#: The identity layer itself: the accessor, the lineage/tenure builds, the register loader they read, the roster listing.
_ACCESSOR = {
    "src/data_extract/utils/common/identity.py",
    "src/data_extract/utils/common/entity_lineage.py",
    "src/data_extract/utils/common/symbol_tenure.py",
    "src/data_extract/utils/common/registrant.py",
    "src/data_extract/utils/common/sec_utils.py",
    "src/data_extract/utils/common/security_master.py",
}
#: The roster seeder writes `sp500_tickers`; 13F managers are listed by their own CIK, not an issuer (00-run N1).
_ROSTER_WRITER = "src/data_extract/utils/prices/fetch_tickers.py"
_MANAGER_LISTING = "src/data_extract/utils/institutionals/fetch_13f_managers.py"
_RETIRED = {"cik_to_ticker", "drop_rows_outside_segment", "entity_ticker", "resolve_symbol_rows", "resolve_symbol_ticker", "candidate_symbols"}
_REGISTER = {"load_registrants", "_registrants_at", "REGISTRANT_CONFIG_FILENAME"}


def _names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.FunctionDef | ast.alias):
            out.add(node.name)
    return out


def test_fetchers_resolve_identity_only_through_the_accessor():
    """AC-027: no retired CIK/symbol map anywhere in `src`; outside the identity layer no fetcher reads the
    register, `sp500_tickers` or builds `Company` from anything but `int(cik)`."""
    retired, register, roster, company = [], [], [], []
    for path in sorted((REPO / "src").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = _names(tree)
        retired += [f"{rel}: {name}" for name in sorted(names & _RETIRED)]
        if not rel.startswith("src/data_extract/utils/") or rel in _ACCESSOR:
            continue
        register += [f"{rel}: {name}" for name in sorted(names & _REGISTER)]
        if rel != _ROSTER_WRITER and any(isinstance(n, ast.Attribute) and n.attr == "sp500_tickers" for n in ast.walk(tree)):
            roster.append(rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "Company"
                and rel != _MANAGER_LISTING
            ):
                arg = node.args[0] if node.args else None
                if not (isinstance(arg, ast.Call) and getattr(arg.func, "id", None) == "int"):
                    company.append(f"{rel}:{node.lineno}")
    assert retired == [], f"retired identity maps still referenced: {retired}"
    assert register == [], f"fetchers reading the register directly: {register}"
    assert roster == [], f"fetchers reading sp500_tickers directly: {roster}"
    assert company == [], f"Company() built from a non-int argument: {company}"
    print("\n=== SANITY CHECK: AC-027 one identity source ===")
    print(f"  no {sorted(_RETIRED)} in src; outside {len(_ACCESSOR)} identity modules no register read, no sp500_tickers read,")
    print("  every issuer Company() takes int(cik). Validated.")
