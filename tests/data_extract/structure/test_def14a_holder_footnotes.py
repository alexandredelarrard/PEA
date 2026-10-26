"""Exact EOG source footnotes must not become a second economic holder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract, OwnershipExtract
from src.data_extract.utils.structure.def14a.holders import _holder_name, normalize_name, reconcile_holders
from src.data_extract.utils.structure.def14a.postprocess import postprocess_section, refresh_archive
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_extract.utils.structure.def14a.work import load_work, save_section
from src.data_store.schema import Tables
from src.data_store.store import DataStore

EOG_NAMES = (
    ("The Vanguard Group", "(b)", 34_810_950),
    ("BlackRock, Inc.", "(c)", 31_845_410),
    ("FMR LLC", "(d)", 29_869_058),
    ("Capital Research Global Investors", "(e)", 28_680_524),
)


@pytest.fixture
def eog_holders() -> pd.DataFrame:
    fixture = Path(__file__).parent / "fixtures/def14a_eog2015_holder_footnotes.json"
    document = json.loads(fixture.read_text())
    assert document["source_accession"] == "0001193125-15-098865" and document["source_html_table_ordinal"] == 20
    df_rows = pd.DataFrame(document["rows"])
    df_rows["as_of"] = pd.to_datetime(df_rows["as_of"])
    return df_rows


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("name", "footnote", "shares"), EOG_NAMES)
def test_recorded_eog_lettered_postal_aliases_and_terminal_markers(
    eog_holders: pd.DataFrame, name: str, footnote: str, shares: int, reverse: bool
) -> None:
    df_pair = eog_holders.loc[eog_holders.holder_name.str.startswith(name)]
    assert len(df_pair) == 2 and df_pair.shares.eq(shares).all()
    df_result = reconcile_holders(df_pair.iloc[::-1] if reverse else df_pair, pd.DataFrame())
    assert len(df_result) == 1 and df_result.iloc[0].holder_name == name and df_result.iloc[0].shares == shares
    assert normalize_name(_holder_name(name + footnote)) == normalize_name(name)
    print(f"SANITY: EOG2015 {name}{footnote} and its recorded postal label resolve to one holder with {shares:,} unchanged shares.")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Example Fund Series A", "Example Fund Series B"),
        ("John Example, Jr.", "John Example, Sr."),
        ("Example Capital LLC", "Example Capital Advisors LLC"),
        ("A B Capital", "AB Capital"),
    ],
)
def test_name_normalization_preserves_generations_affiliates_and_tokens(first: str, second: str) -> None:
    stamp = dict(ticker="EOG", accession_number="same", as_of="2015-03-20", holder_type="5pct_holder")
    df_rows = pd.DataFrame([{**stamp, "holder_name": first, "shares": 1}, {**stamp, "holder_name": second, "shares": 2}])
    df_result = reconcile_holders(df_rows, pd.DataFrame())
    assert df_result.holder_name.tolist() == [first, second] and len(df_result) == 2
    print(f"SANITY: shared normalized names preserve distinct tokens outside parentheses: {first}; {second}.")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  JOSÉ  ÁLVAREZ, Jr. (c)  ", "jose alvarez jr"),
        ("BlackRock, Inc.(c)", "blackrock inc"),
        ("Élodie (director) O'Neil-Smith", "elodie o neil smith"),
        ("Example Fund Class(a)", "example fund class"),
        ("Example Fund Class(b)", "example fund class"),
        ("Alpha (fund (class A) partner) Holdings", "alpha holdings"),
        ("The Vanguard Group (b) (address)", "the vanguard group"),
        (" A_B...Capital  ", "a b capital"),
    ],
)
def test_shared_name_normalizer_removes_parentheses_accents_case_and_punctuation(value: str, expected: str) -> None:
    assert normalize_name(value) == expected and normalize_name(expected) == expected
    print(f"SANITY: {value!r} normalizes idempotently to {expected!r}.")


@pytest.mark.parametrize("postal_only", [False, True])
@pytest.mark.usefixtures("filing_store")
def test_eog_letter_cleanup_persists_once_and_leaves_ge_facts_untouched(
    sqlite_store: DataStore, eog_holders: pd.DataFrame, postal_only: bool
) -> None:
    section = next(item for item in SECTIONS if item.key == "ownership")
    ge_fixture = Path(__file__).parent / "fixtures/def14a_ge2025_holders.json"
    df_ge = pd.DataFrame(json.loads(ge_fixture.read_text())["rows"])
    df_ge["as_of"] = pd.to_datetime(df_ge["as_of"])
    df_seed = eog_holders.iloc[1::2] if postal_only else eog_holders
    sqlite_store.save(section.table, pd.concat([df_seed, df_ge], ignore_index=True))
    df_ge_before = sqlite_store.load(section.table, where={"ticker": "GE"}).copy()
    stamp: dict[str, Any] = eog_holders.iloc[0][["ticker", "accession_number", "as_of", "cik"]].to_dict()
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    filings = pd.DataFrame([{**stamp, "filing_date": stamp["as_of"]}])
    work = load_work(context, "EOG", filings, pd.Timestamp("2015-01-01"))[0]
    raw = Def14AExtract.model_validate({"ownership_holders": eog_holders.to_dict("records")}).model_dump_json()

    # Replay the same already-paid labels through actual save/delete persistence twice.
    for _ in range(2):
        answer = OwnershipExtract(percent_unit="fraction", shares_scale="units")
        processed = postprocess_section(section, answer, stamp, work.saved, work.parent)
        save_section(context, work, processed)
        facade_raw = refresh_archive(raw, section, processed.legacy)
        document = json.loads(facade_raw)
        df_stored = sqlite_store.load(section.table, where={"ticker": "EOG"})
        assert df_stored is not None and len(df_stored) == 4 and df_stored.retry.eq(2).all()
        assert pd.to_datetime(df_stored.as_of).eq(stamp["as_of"]).all() and df_stored.cik.eq(stamp["cik"]).all()
        assert dict(zip(df_stored.holder_name, df_stored.shares, strict=True)) == {name: shares for name, _, shares in EOG_NAMES}
        assert len(document["ownership_holders"]) == 4 and document["_def14a_split"]["legacy_json"] == raw
        assert work.parent["n_ownership_rows"] == 4 and work.parent["n_five_percent_holders"] == 4
        pd.testing.assert_frame_equal(sqlite_store.load(Tables.def14a_ownership, where={"ticker": "GE"}), df_ge_before)
    print(
        "SANITY: EOG's eight saved labels become four stable holders at retry=2; exact shares, filing/entity clocks, archived originals and all 15 GE rows survive."
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("BlackRock, Inc.(c)", "BlackRock, Inc."),
        ("The Vanguard Group(b)(c)", "The Vanguard Group"),
        ("Capital Research Global Investors (e) (f)", "Capital Research Global Investors"),
        ("Example Fund (Class A)", "Example Fund (Class A)"),
        ("John Example, Jr.", "John Example, Jr."),
        ("Example (a) Capital", "Example (a) Capital"),
    ],
)
def test_display_cleanup_removes_only_terminal_letter_footnotes(value: str, expected: str) -> None:
    assert _holder_name(value) == expected and _holder_name(expected) == expected
    print(f"SANITY: holder display {value!r} becomes {expected!r}, retaining readable punctuation and substantive qualifiers.")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.usefixtures("filing_store")
def test_parenthetical_stock_label_keeps_issuer_quantity_in_either_order(sqlite_store: DataStore, reverse: bool) -> None:
    # Frozen EOG1996 order: issuer sole-power quantity, then separately disclosed Enron stock.
    stamp = dict(ticker="EOG", accession_number="0000950129-96-000428", as_of=pd.Timestamp("1996-03-25"), cik="0000821189")
    rows = [
        dict(**stamp, holder_name="Dennis M. Ulak", holder_type="director_officer", shares=52_863, retry=1),
        dict(**stamp, holder_name="Dennis M. Ulak (Enron Corp. common stock)", holder_type="director_officer", shares=4_822, retry=1),
    ]
    sqlite_store.save(Tables.def14a_ownership, pd.DataFrame(rows[::-1] if reverse else rows))
    context = cast(Context, SimpleNamespace(store=sqlite_store))
    work = load_work(context, "EOG", pd.DataFrame([{**stamp, "filing_date": stamp["as_of"]}]), pd.Timestamp("1996-01-01"))[0]
    section = next(item for item in SECTIONS if item.key == "ownership")
    answer = OwnershipExtract(percent_unit="fraction", shares_scale="units")
    processed = postprocess_section(section, answer, stamp, work.saved, work.parent)
    save_section(context, work, processed)
    df_stored = sqlite_store.load(section.table, where={"ticker": "EOG", "accession_number": stamp["accession_number"]})
    assert df_stored is not None and len(df_stored) == 1
    assert df_stored.iloc[0].holder_name == "Dennis M. Ulak" and df_stored.iloc[0].shares == 52_863 and df_stored.iloc[0].retry == 1
    print(
        "SANITY: in either EOG1996 row order, parenthesis normalization preserves the existing 52,863 issuer shares; the separate 4,822 Enron shares cannot overwrite them."
    )


def test_security_qualified_peer_never_fills_a_missing_issuer_quantity() -> None:
    stamp = dict(ticker="EOG", accession_number="same", as_of="1996-03-25", holder_type="director_officer")
    df_saved = pd.DataFrame(
        [
            {**stamp, "holder_name": "Dennis M. Ulak (Enron Corp. common stock)", "shares": 4_822},
            {**stamp, "holder_name": "Dennis M. Ulak", "shares": None},
        ]
    )
    df_result = reconcile_holders(df_saved, pd.DataFrame())
    assert len(df_result) == 1 and pd.isna(df_result.iloc[0].shares)
    assert df_result.iloc[0].holder_name == "Dennis M. Ulak"
    print("SANITY: a missing issuer quantity stays unknown; Enron's 4,822 shares cannot fill it.")


def test_new_issuer_row_displaces_only_the_foreign_security_peer() -> None:
    stamp = dict(ticker="EOG", accession_number="same", as_of="1996-03-25", holder_type="director_officer")
    df_saved = pd.DataFrame([{**stamp, "holder_name": "Dennis M. Ulak (Enron Corp. common stock)", "shares": 4_822}])
    df_new = pd.DataFrame([{**stamp, "holder_name": "Dennis M. Ulak", "shares": 52_863}])
    df_result = reconcile_holders(df_saved, df_new)
    assert len(df_result) == 1 and df_result.iloc[0].shares == 52_863
    assert df_result.iloc[0].holder_name == "Dennis M. Ulak"
    df_alone = reconcile_holders(df_saved, pd.DataFrame())
    assert len(df_alone) == 1 and df_alone.iloc[0].shares == 4_822
    print("SANITY: a source-comparable new issuer row displaces the foreign-security peer; a lone qualified row remains explicit and unchanged.")
