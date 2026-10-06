"""AC-006 and AC-016 check 1 for `sec_8k_votes`: an Item 5.07 filing that was read and holds no vote
row (the LLM returned zero proposals, or the guard refused the text before any call) is stored as
one `proposal_seq = 0.0` marker and is never queued again; a failed LLM call writes nothing and is
queued on the next run. Runs on a real SQLite `DataStore` with a counting fake LLM (no spend)."""

from __future__ import annotations

import json
import logging
import pathlib
import re
import types
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.empty_markers import marker_frame
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract as _Def14AExtract
from src.data_extract.utils.schemas.vote_schema import Item507Extract as _Item507Extract
from src.data_extract.utils.structure.def14a.flatten import _result_frames as def14a_result_frames
from src.data_extract.utils.structure.votes import fetch as votes_fetch
from src.data_extract.utils.structure.votes.flatten import _marker_frame as votes_marker_frame
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import _frames_by_table
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from tests.data_extract.fake_context import extract_config

Item507Extract: Any = _Item507Extract
Def14AExtract: Any = _Def14AExtract

_SCHEMA_SQL = pathlib.Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
_FIXTURES = json.loads((pathlib.Path(__file__).parent / "fixtures" / "item507_texts.json").read_text(encoding="utf-8"))
_READ = ("aapl_2025", "jpm_2025", "aee_2026")
_SHORT_ACCESSION = "0000000009-25-000001"


class _CountingLLM:
    """`LLMExtractor` stand-in: answers every task with zero proposals (or an error for accessions in
    `failing`), saves through the caller's flatten like the real extractor, and counts the calls."""

    calls: list[str] = []
    failing: set[str] = set()

    def __init__(self, context: Any, config: Any, action: str | None = None, threads: int | None = None) -> None:
        self._context = context

    def run_extraction(self, tasks: Any, flatten: Any = None, group_key: Any = None) -> list[LlmResult]:
        results = []
        for task in tasks:
            accession = str(task.meta["filing"]["accession_number"])
            type(self).calls.append(accession)
            if accession in type(self).failing:
                results.append(LlmResult(seq=task.seq, task=task, parsed=None, error="APIError: 500"))
            else:
                results.append(LlmResult(seq=task.seq, task=task, parsed=Item507Extract(proposals=[])))
        for table, parts in _frames_by_table(results, flatten).items():
            self._context.store.save(table, pd.concat(parts, ignore_index=True))
        return results


def _source_row(key: str) -> dict[str, object]:
    row = dict(_FIXTURES[key])
    return {
        **row,
        "item": "5.07",
        "filing_date": pd.Timestamp(str(row["filing_date"])),
        "period_of_report": pd.Timestamp(str(row["period_of_report"])),
    }


def _context(sqlite_store, *, short: bool = True) -> Any:
    rows = [_source_row(key) for key in _READ]
    if short:
        rows.append({**_source_row("aapl_2025"), "accession_number": _SHORT_ACCESSION, "item_text": "Item 5.07 results to follow."})
    sqlite_store.save(Tables.sec_8k, pd.DataFrame(rows))
    return types.SimpleNamespace(store=sqlite_store, log=logging.getLogger("test.votes"), config=extract_config())


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> type[_CountingLLM]:
    _CountingLLM.calls, _CountingLLM.failing = [], set()
    monkeypatch.setattr(votes_fetch, "LLMExtractor", _CountingLLM)
    return _CountingLLM


def _tickers() -> list[str]:
    return sorted({str(_FIXTURES[key]["ticker"]) for key in _READ})


def test_zero_proposal_filing_is_queued_once(sqlite_store, llm) -> None:
    context = _context(sqlite_store)

    votes_fetch.fetch_8k_votes_llm(context, context.config, _tickers())
    first = len(llm.calls)
    votes_fetch.fetch_8k_votes_llm(context, context.config, _tickers())

    assert first == 3 and len(llm.calls) == 3, llm.calls
    markers = sqlite_store.load(Tables.sec_8k_votes, markers=True)
    assert sorted(markers["accession_number"]) == sorted([str(_FIXTURES[k]["accession_number"]) for k in _READ] + [_SHORT_ACCESSION])
    assert set(markers["proposal_seq"]) == {0.0}
    assert sqlite_store.load(Tables.sec_8k_votes, optional=True) is None
    print("\n=== SANITY CHECK: AC-006, zero-proposal and guard-rejected 5.07 filings ===")
    print(f"  run 1: {first} LLM calls, {len(markers)} markers (3 zero-proposal + 1 refused text); run 2: {len(llm.calls) - first} calls.")


def test_a_failed_llm_call_writes_nothing_and_is_queued_again(sqlite_store, llm) -> None:
    context = _context(sqlite_store, short=False)
    failed = str(_FIXTURES["jpm_2025"]["accession_number"])
    llm.failing = {failed}

    votes_fetch.fetch_8k_votes_llm(context, context.config, _tickers())
    stored = set(sqlite_store.distinct(Tables.sec_8k_votes, "accession_number"))
    llm.failing = set()
    votes_fetch.fetch_8k_votes_llm(context, context.config, _tickers())

    assert failed not in stored and len(stored) == 2
    assert llm.calls[3:] == [failed]
    print("\n=== SANITY CHECK: failed LLM call ===")
    print(f"  run 1 stored {len(stored)} markers and nothing for {failed}; run 2 queued only {llm.calls[3:]}.")


def test_a_votes_marker_reads_back_as_typed_nulls(sqlite_store, llm) -> None:
    context = _context(sqlite_store, short=False)
    real = {
        **{k: v for k, v in _source_row("ge_2024").items() if k not in ("item", "item_text")},
        "proposal_seq": 1.0,
        "votes_for": 10.0,
        "meeting_date": pd.Timestamp("2024-05-07"),
        "description": "Election of directors",
    }
    sqlite_store.save(Tables.sec_8k_votes, pd.DataFrame([real]))

    votes_fetch.fetch_8k_votes_llm(context, context.config, _tickers())

    shown = sqlite_store.load(Tables.sec_8k_votes, markers=True)
    row = shown[shown["proposal_seq"] == 0.0].iloc[0]
    assert pd.isna(row["votes_for"]) and isinstance(row["votes_for"], float)
    assert row["meeting_date"] is None or row["meeting_date"] is pd.NaT
    assert pd.isna(row["description"]) and not isinstance(row["description"], str)
    assert row["filing_date"] is not None and row["cik"] and row["form"] == "8-K"
    assert sqlite_store.load(Tables.sec_8k_votes)["proposal_seq"].tolist() == [1.0]
    print("\n=== SANITY CHECK: AC-016, sec_8k_votes marker ===")
    print(
        f"  proposal_seq=0.0; votes_for={row['votes_for']!r}, meeting_date={row['meeting_date']!r}, description={row['description']!r}; hidden by load."
    )


def _ddl_columns(name: str) -> set[str]:
    block = re.search(rf'CREATE TABLE IF NOT EXISTS "{name}" \((.*?)\n\);', _SCHEMA_SQL.read_text(encoding="utf-8"), re.S)
    assert block is not None, name
    return set(re.findall(r'^\s+"(\w+)"', block.group(1), re.M))


def test_the_llm_markers_carry_only_ddl_columns() -> None:
    filing = pd.Series({**_source_row("aapl_2025")})
    votes = votes_marker_frame("AAPL", filing)
    task = LlmTask(seq=0, payload="proxy", schema=Def14AExtract, table=Tables.def14a_llm, meta={"ticker": "AAPL", "filing": filing})
    proxy = def14a_result_frames(LlmResult(seq=0, task=task, parsed=Def14AExtract()))[Tables.def14a_llm]

    assert set(votes.columns) <= _ddl_columns("sec_8k_votes") and votes["proposal_seq"].tolist() == [0.0]
    assert set(proxy.columns) <= _ddl_columns("def14a_llm") and proxy["def14a_json"].tolist() == ["_empty"]
    assert (
        marker_frame(Tables.def14a_llm, {"ticker": "X", "accession_number": "a", "as_of": pd.Timestamp("2025-01-01")})["as_of"].dtype
        == "datetime64[ns]"
    )
    print("\n=== SANITY CHECK: AC-016, LLM marker columns ===")
    print(f"  sec_8k_votes marker {sorted(votes.columns)}; def14a_llm marker {sorted(proxy.columns)}; all declared in sql/schema.sql.")
