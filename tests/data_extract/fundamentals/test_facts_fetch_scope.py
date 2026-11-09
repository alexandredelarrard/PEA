"""`SEC_FUNDAMENTALS_FETCH` is a plain `EdgarFetch` constant: its parser (catalogue + GICS) is built per run through
`run_scope`, it plans like every other walk, and it lists the original 10-K and 10-Q only.

Offline: a seeded SQLite universe, the worktree's own `configs/` catalogue and a seeded local EDGAR index.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.data_extract.utils.edgar_extract.edgar_driver import plan_fetch
from src.data_extract.utils.fundamentals import fetch_fundamentals_sec as facts
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import SEC_FUNDAMENTALS_FETCH, parse_fundamentals
from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue
from src.data_store.schema import Resume, Tables
from tests.data_extract.edgar_fixtures import fake_context, roster_run, seed_index
from tests.support.universe import seed_universe

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
ROWS = {
    "XOM": {"name": "ExxonMobil", "cik": "0000034088", "sector": "Energy", "industry_group": "Energy", "sub_industry": "Integrated Oil & Gas"},
    "JPM": {"name": "JPMorgan", "cik": "0000019617", "sector": "Financials", "industry_group": "Banks", "sub_industry": "Diversified Banks"},
}
LEVELS = ("sector", "industry_group", "sub_industry")


def test_entering_the_run_scope_binds_the_catalogue_and_the_universe_gics(sqlite_store: Any) -> None:
    context: Any = SimpleNamespace(universe=seed_universe(sqlite_store, list(ROWS), rows=ROWS), config_dir=CONFIG_DIR)

    assert SEC_FUNDAMENTALS_FETCH.parse is None, "the constant declares no parser outside a run"
    assert SEC_FUNDAMENTALS_FETCH.run_scope is not None
    with SEC_FUNDAMENTALS_FETCH.run_scope(context, ["XOM"], pd.Timestamp("2026-11-08")) as prepared:
        parse = cast(Any, prepared.parse)

    assert parse.func is parse_fundamentals
    assert isinstance(parse.keywords["catalogue"], Catalogue) and parse.keywords["catalogue"].extracted_fields
    assert parse.keywords["gics_by_ticker"] == {t: {level: row[level] for level in LEVELS} for t, row in ROWS.items()}
    assert prepared.tables == (Tables.sec_fundamentals_facts,) and prepared.done is Tables.sec_fundamentals_facts
    print("\n=== SANITY CHECK: facts run scope ===")
    print(
        f"  constant parse=None; inside run_scope parse = parse_fundamentals with {len(parse.keywords['catalogue'].extracted_fields)} "
        f"catalogue fields and GICS for {sorted(parse.keywords['gics_by_ticker'])} (3 levels each, equal to the universe rows). Validated."
    )


def test_the_facts_contract_lists_originals_only_from_2009() -> None:
    resume = cast(Resume, Tables.sec_fundamentals_facts.resume)

    assert resume.forms == ("10-K", "10-Q") and resume.source_start == "2009-04-15"
    assert facts._ANNUAL_FORMS == ("10-K",), "the note-only guard follows the resume forms"
    print("\n=== SANITY CHECK: facts forms ===")
    print(f"  forms {resume.forms}, source start {resume.source_start}, annual forms {facts._ANNUAL_FORMS}; no /A form is listed. Validated.")


def test_plan_fetch_on_the_constant_lists_the_original_filings(tmp_path: Path, sqlite_store: Any) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], ["1"])
    seed_index(
        ctx,
        [
            (1, "Fixture Co", "10-K", "2026-02-20", "0000000001-26-000001"),
            (1, "Fixture Co", "10-K/A", "2026-03-02", "0000000001-26-000002"),
            (1, "Fixture Co", "10-Q", "2026-05-01", "0000000001-26-000003"),
            (1, "Fixture Co", "10-Q/A", "2026-05-20", "0000000001-26-000004"),
            (1, "Fixture Co", "8-K", "2026-06-01", "0000000001-26-000005"),
        ],
    )

    work = plan_fetch(ctx, SEC_FUNDAMENTALS_FETCH, roster_run(ctx, ["AAA"], as_of=pd.Timestamp("2026-09-30")))

    listed = work.units["AAA"].sort_values("accession")
    assert listed["accession"].tolist() == ["0000000001-26-000001", "0000000001-26-000003"], listed
    assert listed["form"].tolist() == ["10-K", "10-Q"]
    print("\n=== SANITY CHECK: plan_fetch on SEC_FUNDAMENTALS_FETCH ===")
    print(
        f"  index 10-K, 10-K/A, 10-Q, 10-Q/A, 8-K -> work list {listed['form'].tolist()} ({work.size} documents); no parser needed to plan. Validated."
    )
