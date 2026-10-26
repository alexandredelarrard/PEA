"""Persisted same-filing holder reconciliation against recorded GE repair rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract, OwnershipExtract
from src.data_extract.utils.structure.def14a import fetch
from src.data_extract.utils.structure.def14a.holders import reconcile_holders
from src.data_extract.utils.structure.def14a.postprocess import append_answer, postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS, reservation_frame
from src.data_extract.utils.structure.def14a.work import load_archives, load_work, replay_candidates, reserve, save_archive, save_section
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.fake_context import extract_config


@pytest.fixture
def ge_holders() -> pd.DataFrame:
    fixture = Path(__file__).parent / "fixtures/def14a_ge2025_holders.json"
    df_holders = pd.DataFrame(json.loads(fixture.read_text())["rows"])
    df_holders["as_of"] = pd.to_datetime(df_holders["as_of"])
    return df_holders


def _stamp(df_holders: pd.DataFrame) -> dict[str, Any]:
    return df_holders.iloc[0][["ticker", "accession_number", "as_of", "cik"]].to_dict()


@pytest.mark.usefixtures("filing_store")
def test_recorded_ge_repair_reconciles_four_aliases_before_persisting(sqlite_store: DataStore, ge_holders: pd.DataFrame) -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = _stamp(ge_holders)
    sqlite_store.save(section.table, ge_holders)
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    filings = pd.DataFrame([{**stamp, "filing_date": stamp["as_of"]}])
    work = load_work(context, "GE", filings, pd.Timestamp("2025-01-01"))[0]
    raw = Def14AExtract.model_validate(
        {"ownership_holders": ge_holders.to_dict("records"), "governance": {"n_five_percent_holders": 99}}
    ).model_dump_json()

    original_raw = raw
    raw = append_answer(raw, section, OwnershipExtract(percent_unit="fraction", shares_scale="units"), "recorded ownership source")
    original_responses = json.loads(raw)["_def14a_split"]["responses"]

    # Replay the real previously paid labels twice through actual SQLite upserts and deletes.
    for _ in range(2):
        result = postprocess_section(section, Def14AExtract.model_validate_json(raw), stamp, work.saved, work.parent)
        save_section(context, work, result)
        facade_raw = refresh_archive(raw, section, result.legacy)
        facade = Def14AExtract.model_validate_json(facade_raw)
        assert json.loads(facade_raw)["_def14a_split"]["legacy_json"] == original_raw
        assert json.loads(facade_raw)["_def14a_split"]["responses"] == original_responses
        assert json.loads(original_raw)["governance"]["n_five_percent_holders"] == 99
        assert facade.governance is not None and facade.governance.n_five_percent_holders == 4
        assert len(result.legacy.ownership_holders) == len(result.frames[section.table]) == 11
        df_stored = sqlite_store.load(section.table, where={"ticker": "GE", "accession_number": stamp["accession_number"]})
        assert df_stored is not None and len(df_stored) == 11
        assert len(facade.ownership_holders) == 11
        assert set(df_stored.retry) == {2} and pd.to_datetime(df_stored.as_of).eq(stamp["as_of"]).all()
        assert work.parent["n_ownership_rows"] == 11 and work.parent["n_five_percent_holders"] == 4
    names = set(df_stored.holder_name)
    assert {"BlackRock, Inc.", "Capital International Investors", "Fidelity Management & Research", "The Vanguard Group"} <= names
    assert "H. Lawrence Culp, Jr." in names and "John Phillips, III" in names
    print("SANITY: recorded GE repair stays at 11 holders, including four institutions; repeat replay cannot recreate four address aliases.")


@pytest.mark.usefixtures("filing_store")
def test_complete_exhausted_holders_are_normalized_free_without_archive(sqlite_store: DataStore, ge_holders: pd.DataFrame) -> None:
    stamp = _stamp(ge_holders)
    df_complete = ge_holders.fillna({"percent_of_class": 0.01, "percent_of_voting_power": 0.01})
    sqlite_store.save(Tables.def14a_ownership, df_complete)
    parent = {**stamp, "insider_shares": 1, "insider_ownership_pct": 0.01, "insider_voting_pct": 0.01, "retry": 2}
    sqlite_store.ensure_columns(Tables.def14a_llm, pd.DataFrame([parent]))
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([parent]))
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    filings = pd.DataFrame([{**stamp, "filing_date": stamp["as_of"]}])
    works = load_work(context, "GE", filings, pd.Timestamp("2025-01-01"))
    load_archives(context, works)
    candidates = list(replay_candidates(works))
    assert len(candidates) == 1 and candidates[0][1].key == "ownership"
    work, section, answer = candidates[0]
    reserve(context, work, section, work.attempts[section.key])
    result = postprocess_section(section, answer, work.stamp, work.saved, work.parent)
    save_section(context, work, result)
    work.raw = refresh_archive(work.raw, section, result.legacy)
    save_archive(context, work)
    assert len(work.saved[section.table]) == 11 and work.attempts[section.key] == 2
    repeated = load_work(context, "GE", filings, pd.Timestamp("2025-01-01"))
    load_archives(context, repeated)
    assert list(replay_candidates(repeated)) == []
    assert repeated[0].saved[section.table].retry.eq(2).all()
    print("SANITY: complete retry=2 holders without any archived answer receive one free cleanup; the next run has no ownership replay candidate.")


def test_holder_repair_retains_partial_old_facts_and_distinct_entities() -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = dict(ticker="GE", accession_number="same", as_of=pd.Timestamp("2025-03-13"), cik="40545")
    df_saved = pd.DataFrame(
        [
            {
                **stamp,
                "holder_name": "Example Holdings, Inc., 100 Main Street, Boston, MA 02110",
                "holder_type": "5pct_holder",
                "shares": 123,
                "retry": 2,
            },
            {**stamp, "holder_name": "Example Holdings Advisors, Inc.", "holder_type": "5pct_holder", "shares": 456, "retry": 2},
            {**stamp, "holder_name": "John Example, Jr.", "holder_type": "director_officer", "shares": 1, "retry": 2},
            {**stamp, "holder_name": "John Example, Sr.", "holder_type": "director_officer", "shares": 2, "retry": 2},
        ]
    )
    answer = OwnershipExtract.model_validate(
        dict(
            percent_unit="fraction",
            shares_scale="units",
            holders=[dict(holder_name="Example Holdings, Inc.", holder_type="5pct_holder", shares=999, percent_of_class=0.1)],
        )
    )
    df_result = postprocess_section(section, answer, stamp, {section.table: df_saved}, {}).frames[section.table].set_index("holder_name")
    assert len(df_result) == 4 and df_result.loc["Example Holdings, Inc.", "shares"] == 123
    assert df_result.loc["Example Holdings, Inc.", "percent_of_class"] == 0.1
    assert set(df_result.index) == {"Example Holdings, Inc.", "Example Holdings Advisors, Inc.", "John Example, Jr.", "John Example, Sr."}
    assert df_result.retry.eq(2).all()
    print("SANITY: alias repair fills only the missing percentage, preserves saved shares, and keeps affiliates and Jr/Sr separate.")


@pytest.mark.usefixtures("filing_store")
def test_survivor_save_failure_does_not_delete_saved_holder_aliases(
    sqlite_store: DataStore, ge_holders: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    stamp = _stamp(ge_holders)
    sqlite_store.save(section.table, ge_holders)
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    work = load_work(context, "GE", pd.DataFrame([{**stamp, "filing_date": stamp["as_of"]}]), pd.Timestamp("2025-01-01"))[0]
    result = postprocess_section(section, OwnershipExtract(percent_unit="fraction", shares_scale="units"), stamp, work.saved, work.parent)
    original_save = sqlite_store.save

    def fail_save(table: Any, frame: pd.DataFrame, **kwargs: Any) -> int:
        if table is section.table:
            raise RuntimeError("survivor save interrupted")
        return original_save(table, frame, **kwargs)

    monkeypatch.setattr(sqlite_store, "save", fail_save)
    with pytest.raises(RuntimeError, match="survivor save interrupted"):
        save_section(context, work, result)
    assert sqlite_store.row_count(section.table) == 15
    monkeypatch.setattr(sqlite_store, "save", original_save)
    save_section(context, work, result)
    assert sqlite_store.row_count(section.table) == 11
    print("SANITY: an interrupted survivor save deletes no old facts; a successful restart saves survivors before removing exact obsolete keys.")


@pytest.mark.parametrize("difference", ["ticker", "accession_number", "as_of", "holder_type"])
def test_holder_identity_stays_with_exact_filing_date_and_type(difference: str) -> None:
    from_row = dict(
        ticker="GE", accession_number="same", as_of=pd.Timestamp("2025-03-13"), holder_type="5pct_holder", holder_name="Example, Inc.", shares=123
    )
    changed = {"ticker": "OTHER", "accession_number": "different", "as_of": pd.Timestamp("2025-03-14"), "holder_type": "director_officer"}
    df_rows = pd.DataFrame([from_row, {**from_row, difference: changed[difference], "holder_name": "Example Inc", "shares": 456}])
    df_result = reconcile_holders(df_rows, pd.DataFrame())
    assert len(df_result) == 2 and set(df_result.shares) == {123, 456}
    print(f"SANITY: punctuation equality cannot combine different {difference} scopes.")


def test_holder_identity_preserves_word_boundaries_and_legal_suffixes() -> None:
    names = ["A B Capital", "AB Capital", "Ann A", "Anna", "Example LP", "Example LLC"]
    rows = [
        dict(ticker="GE", accession_number="same", as_of="2025-03-13", holder_type="5pct_holder", holder_name=name, shares=index)
        for index, name in enumerate(names)
    ]
    df_result = reconcile_holders(pd.DataFrame(rows), pd.DataFrame())
    assert df_result.holder_name.tolist() == names and len(df_result) == 6
    print("SANITY: separate name tokens and legal suffixes remain distinct; no fuzzy or prefix matching.")


@pytest.mark.usefixtures("filing_store")
def test_public_daily_and_full_replay_ge_aliases_without_source_or_provider(
    sqlite_store: DataStore, ge_holders: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = _stamp(ge_holders)
    for section in SECTIONS:
        sqlite_store.save(section.table, reservation_frame(section.table, stamp, None, 2))
    sqlite_store.save(Tables.def14a_ownership, ge_holders)
    raw = Def14AExtract.model_validate(
        {"ownership_holders": ge_holders.to_dict("records"), "governance": {"n_five_percent_holders": 99}}
    ).model_dump_json()
    sqlite_store.ensure_columns(Tables.def14a_llm, pd.DataFrame([{**stamp, "def14a_json": raw}]))
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([{**stamp, "def14a_json": raw}]))
    context = cast(Context, SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.def14a.holders")))
    config = extract_config(data_extract={"years_history": 31})
    filings = pd.DataFrame([{**stamp, "filing_date": stamp["as_of"]}])
    identity = SimpleNamespace(filing_scope=lambda _: FilingScope.roster_only("GE", "40545"))
    monkeypatch.setattr(fetch, "load_cik_mapping", lambda *_: pd.DataFrame([dict(ticker="GE", name="GE")]))
    monkeypatch.setattr(fetch, "load_identity", lambda *_: identity)
    monkeypatch.setattr(fetch, "_listed_proxies", lambda *_: filings.copy())
    monkeypatch.setattr(fetch, "_section_extractors", lambda *_: {})
    monkeypatch.setattr(fetch, "_finalise_gender", lambda *_: None)
    monkeypatch.setattr(fetch, "_read_proxy", lambda *_: pytest.fail("exhausted cleanup must not fetch SEC source"))

    # Exercise the unchanged daily entry point twice; an empty extractor map forbids calls.
    for full in (False, True):
        fetch.fetch_def14a_llm(context, config, ["GE"], full=full)
        df_stored = sqlite_store.load(Tables.def14a_ownership, where={"ticker": "GE"})
        parent = sqlite_store.load(Tables.def14a_llm, where={"ticker": "GE"}).iloc[0]
        assert df_stored is not None and len(df_stored) == 11 and set(df_stored.retry) == {2}
        facade = json.loads(parent.def14a_json)
        assert len(facade["ownership_holders"]) == 11 and facade["_def14a_split"]["legacy_json"] == raw
        assert facade["governance"]["n_five_percent_holders"] == 4
        assert json.loads(raw)["governance"]["n_five_percent_holders"] == 99
        assert parent.n_ownership_rows == 11 and parent.n_five_percent_holders == 4 and parent.retry == 2
    print("SANITY: public daily/full extraction cleans recorded GE aliases with no source/provider access and keeps original paid JSON and retry=2.")
