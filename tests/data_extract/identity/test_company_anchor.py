"""Company anchor (REQ-002, AC-006): one frozen `company_id` per universe ticker, never re-keyed by later evidence.

Known-truth fixtures for the anchor read and the frozen id; a real-data check on the phase-0 fixture store when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.identity.company_anchor import anchor_company_ids, roster_ciks, sharadar_ciks
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.support.lineage_build import cik_history_of

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
#: The phase-0 fixture store of the identity run (read-only here); the worktree reads the main checkout's run dir.
FIXTURE = Path("reports/validate/2026-11-03-identity-security-master/_cache/fixture_store/identity.sqlite")


def _fixture_store() -> DataStore:
    """The fixture store opened read-only, or a skip when this checkout has no run dir."""
    for base in (ROOT, *ROOT.parents):
        path = base / FIXTURE
        if path.exists():
            return DataStore(create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"))
    pytest.skip(f"no fixture store at {FIXTURE} under {ROOT} or its parents")


def _tenure(rows: list[tuple[str, str, str, str | None, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "issuer_cik": c,
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(t) if t else pd.NaT,
                "n_filings": n,
                "source": "form345",
                "evidence": c,
            }
            for s, c, f, t, n in rows
        ]
    )


def _config(tmp_path: Path, manual: dict | None = None) -> str:
    sec = tmp_path / "configs" / "sec"
    sec.mkdir(parents=True, exist_ok=True)
    if manual is not None:
        (sec / "entity_lineage_manual.json").write_text(json.dumps(manual), encoding="utf-8")
    return str(tmp_path / "configs")


def test_the_anchor_reads_the_published_id_first_then_the_stored_lineage() -> None:
    """`universe_tickers.company_id` wins; a ticker with no published id takes its stored lineage id; a new ticker has none."""
    universe = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "company_id": ["E0000000100", None, None]})
    lineage = pd.DataFrame(
        {"canonical_ticker": ["AAA", "BBB", "BBB", None], "company_id": ["E0000000999", "E0000000050", "E0000000050", "E0000000777"]}
    )

    anchors = anchor_company_ids(universe, lineage, ["AAA", "BBB", "CCC"])

    assert anchors == {"AAA": "E0000000100", "BBB": "E0000000050"}
    assert anchor_company_ids(None, None, ["AAA"]) == {}
    print("\n=== SANITY CHECK: anchor read order ===")
    print(f"  published AAA -> {anchors['AAA']} (lineage said E0000000999); BBB from its lineage rows -> {anchors['BBB']}; CCC new -> minted later")
    print("  OK: the published id is frozen first, the stored lineage second, nothing else")


def test_the_sharadar_seed_joins_through_the_permaticker_newest_row() -> None:
    """BRK-B is spelled BRK.B by Sharadar, and a permaticker's stale row never wins (PSKY/SKYD)."""
    universe = pd.DataFrame({"ticker": ["BRK-B", "PSKY"], "permaticker": [196523, 198331]})
    sharadar = pd.DataFrame(
        {
            "permaticker": [196523, 198331, 198331],
            "secfilings": [
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001067983",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000813828",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=2041610",
            ],
            "lastupdated": ["2026-08-10", "2026-08-05", "2026-10-22"],
        }
    )
    seeds = sharadar_ciks(universe, sharadar)
    assert seeds == {"BRK-B": "0001067983", "PSKY": "0002041610"}
    print("\n=== SANITY CHECK: Sharadar CIK seed ===")
    print(f"  {seeds}; PSKY's 2026-08-05 row (CIK 813828) is older than the SKYD row and is not read")


def test_a_forced_older_predecessor_keeps_the_frozen_id(tmp_path: Path) -> None:
    """AC-006: a manual entry joining an OLDER CIK to a stored company changes nothing in its id, and nothing is backlogged."""
    tenure = _tenure([("AAA", "0000000200", "2012-06-01", None, 400), ("AAA", "0000000100", "2006-01-05", "2012-01-01", 400)])
    manual = {"AAA/old": {"same_entity": ["0000000100", "0000000200"], "evidence": "8-K 0000000000-12-000001: the old registrant"}}
    stored = pd.DataFrame({"canonical_ticker": ["AAA"], "company_id": ["E0000000200"], "cik": ["0000000200"]})
    owners = pd.DataFrame(columns=["issuer_cik", "owner_cik_raw"])

    frozen = cik_history_of(tenure, {"AAA": "0000000200"}, owners, _config(tmp_path, manual), existing=stored)
    cold = cik_history_of(tenure, {"AAA": "0000000200"}, owners, _config(tmp_path, manual))

    assert frozen.entity_of["0000000100"] == frozen.entity_of["0000000200"] == "E0000000200"
    assert set(frozen.rows["company_id"]) == {"E0000000200"} and frozen.backlog.empty
    assert cold.entity_of["0000000200"] == "E0000000100", "a first build mints E + the oldest CIK, as before"
    print("\n=== SANITY CHECK: frozen company_id (AC-006) ===")
    print(
        f"  stored AAA = E0000000200; manual joins the older 0000000100 -> {frozen.entity_of['0000000100']} for both CIKs, backlog {len(frozen.backlog)}"
    )
    print(f"  cold build (no stored id) -> {cold.entity_of['0000000200']}")
    print("  OK: later evidence adds CIKs to a company and never re-keys it")


def test_every_stored_company_keeps_its_id_on_the_fixture_store() -> None:
    """Real data: each universe ticker's anchor is the `company_id` of its stored lineage rows (XOM, TT, DD, DIS, WBD, PSKY)."""
    store = _fixture_store()
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik"])
    lineage = store.load(Tables.universe_cik_symbol, columns=["canonical_ticker", "company_id"])

    anchors = anchor_company_ids(universe, lineage, list(universe["ticker"]))

    expected = {"XOM": "E0000034088", "TT": "E0000836102", "DD": "E0000029915", "DIS": "E0001001039", "WBD": "E0001437107", "PSKY": "E0000813828"}
    assert {t: anchors.get(t) for t in expected} == expected
    assert len(anchors) == len(universe), f"{len(universe) - len(anchors)} universe ticker(s) without a stored id"
    print("\n=== SANITY CHECK: anchors on the fixture store ===")
    print(f"  {len(anchors)} of {len(universe)} universe tickers anchored; {expected}")
    print("  OK: every stored company keeps today's E + CIK id (PD-3); none is minted again")


def test_the_roster_cik_reads_the_stored_lineage_then_the_sec_list_then_sharadar() -> None:
    """U-49: AAA's newest stored CIK window wins over the SEC list (XOM-style holdco move); BBB is new and takes the SEC
    list; CCC is listed nowhere but on Sharadar (an ended company on a cold start); `universe_tickers.cik` is never read."""
    lineage = pd.DataFrame(
        {
            "canonical_ticker": ["AAA", "AAA", "AAA"],
            "cik": ["0000000100", "0000000200", "0000000200"],
            "role": ["cik_window", "cik_window", "symbol"],
            "valid_from": pd.to_datetime(["1900-01-01", "2025-01-02", "2025-01-02"]),
        }
    )
    sec = pd.DataFrame({"ticker": ["AAA", "BBB"], "cik": [100, 300]})

    roster = roster_ciks(["AAA", "BBB", "CCC"], lineage, sec, {"CCC": "0000000400", "BBB": "0000000999"})

    assert roster == {"AAA": "0000000200", "BBB": "0000000300", "CCC": "0000000400"}, roster
    assert roster_ciks(["AAA"], None, None, {}) == {}
    print("\n=== SANITY CHECK: roster CIK seed (U-49) ===")
    print(f"  {roster}: AAA newest stored window (SEC lists the older 100), BBB from the SEC list, CCC from Sharadar")


def test_a_cold_start_roster_equals_the_synced_cik_on_the_fixture_store() -> None:
    """Real data, cold start: with no stored lineage, the SEC list and the Sharadar seed give every universe ticker the CIK
    the old sync stored; with the stored lineage, the same (fixture: the newest window is that CIK for all 491)."""
    store = _fixture_store()
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik", "permaticker"])
    sec = store.load(Tables.sec_company_tickers, columns=["ticker", "cik"])
    lineage = store.load(Tables.universe_cik_symbol, columns=["canonical_ticker", "cik", "role", "valid_from"])
    seeds = sharadar_ciks(universe, store.load(Tables.sharadar_tickers, columns=["permaticker", "secfilings", "lastupdated"]))
    tickers = list(universe["ticker"])
    synced = dict(zip(universe["ticker"], universe["cik"], strict=True))

    cold = roster_ciks(tickers, None, sec, seeds)
    warm = roster_ciks(tickers, lineage, sec, seeds)

    assert cold == synced and warm == synced, sorted(set(cold.items()) ^ set(synced.items()))[:10]
    print("\n=== SANITY CHECK: roster seed on the fixture store (U-49) ===")
    print(f"  cold start: {len(cold)} of {len(tickers)} tickers seeded, all equal to the stored sync CIK; warm: the same")
