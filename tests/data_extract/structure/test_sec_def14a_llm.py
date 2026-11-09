"""The `sec-def14a-llm` step reads only `sec_def14a_text` (seeded here with the walk's own row builder) and the five
DEF 14A tables: closed sections make no call, rejection markers are never reserved, a `_none` proxy consumes its
attempts, the prompt is cut at the configured budget while postprocessing gets the stored uncut payload, and new
archives carry no `source`.

Offline: a real SQLite `DataStore` with the canonical DDL, the real `LLMExtractor` over a stub provider (no LLM call).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf
from sqlalchemy import create_engine

from src.data_extract.utils.edgar_extract.empty_markers import marker_frame
from src.data_extract.utils.edgar_extract.fetch_def14a_text import _section_rows
from src.data_extract.utils.structure.def14a import fetch
from src.data_extract.utils.structure.def14a.def14a_schema import Def14AExtract, ExecutivePayExtract
from src.data_extract.utils.structure.def14a.postprocess import append_answer, archived_answers, source_for_answer
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.utils.config import read_config
from tests.canonical_schema import create_canonical
from tests.data_extract.fake_context import extract_config
from tests.data_extract.structure.test_def14a_retry_daily import PAYLOADS, _complete_answers
from tests.support.universe import universe_of

BUDGET = int(read_config("configs").data_extract.llm_text_budget.def14a)
YESTERDAY = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
_FACT_TABLES = (Tables.def14a_llm, Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership, Tables.def14a_directors)
EXECUTIVE_PAY = next(section for section in SECTIONS if section.key == "executive_pay")


class Run:
    """One SQLite store, a stub provider that records `(schema, user)` and fails on a user prompt holding `fail_on`."""

    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = extract_config(
            data_extract={"years_history": 15, "llm_text_budget": OmegaConf.to_container(read_config("configs").data_extract.llm_text_budget)}
        )
        self.context = SimpleNamespace(
            store=DataStore(create_engine(f"sqlite:///{path}")),
            log=logging.getLogger("test.sec_def14a_llm"),
            config_dir="configs",
            config=config,
            universe=universe_of(["ZZ"]),
        )
        create_canonical(self.context.store.engine, [*_FACT_TABLES, Tables.sec_def14a_text])
        self.calls: list[tuple[str, str]] = []
        self.answers: dict[str, Any] = {}
        self.fail_on = "NEVER-PRESENT"
        run = self

        class Provider:
            name, model, structured = "stub", "stub", True

            def parse(self, schema, system, user):
                run.calls.append((schema.__name__, user))
                if run.fail_on in user:
                    raise RuntimeError("provider unavailable")
                return schema.model_validate(run.answers.get(schema.__name__, {"percent_unit": "percent"})), {"input_tokens": 1, "output_tokens": 1}

        monkeypatch.setattr(LLMExtractor, "initialize_client", lambda *_args, **_kwargs: Provider())

    def seed(self, accession: str, filed: pd.Timestamp, payloads: dict[str, str], period: str | None = "2025-12-31") -> None:
        """The walk's rows for one proxy: one per non-empty section, else the visible `_none` row."""
        stamped = {
            "cik": "0000000001",
            "accession_number": accession,
            "form": "DEF 14A",
            "filing_date": filed,
            "period": period,
            "primary_document": "p.htm",
        }
        self.context.store.save(Tables.sec_def14a_text, pd.DataFrame(_section_rows("ZZ", stamped, payloads)))

    def __call__(self, replay: bool = False) -> None:
        context = self.context
        fetch.fetch_def14a_llm(context, context.config, ["ZZ"], fetch.section_extractors(context, context.config), replay_stored_json=replay)

    def attempts(self) -> dict[str, list[int]]:
        out = {}
        for table in _FACT_TABLES:
            frame = self.context.store.load(table, markers=True, optional=True)
            out[table.name] = [] if frame is None else sorted(frame.retry.dropna().astype(int).unique().tolist())
        return out


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    return Run(tmp_path / "def14a.db", monkeypatch)


def test_closed_sections_make_no_call(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-003: a proxy with no key gap and one whose gaps are at retry=2 get no call and no payload read."""
    run.answers.update(_complete_answers())
    run.fail_on = "SPENT"
    run.seed("done", YESTERDAY - pd.Timedelta(days=2), PAYLOADS)
    run.seed("spent", YESTERDAY, {key: text + "\nSPENT" for key, text in PAYLOADS.items()})
    run()
    run()
    assert len(run.calls) == 15, "done: 5 answered calls; spent: 5 failed calls then 5 retries"
    loads: list[tuple[str, tuple[str, ...]]] = []
    original = run.context.store.load

    def tracked(table: Any, *args: Any, **kwargs: Any) -> Any:
        loads.append((getattr(table, "name", str(table)), tuple(kwargs.get("columns") or ())))
        return original(table, *args, **kwargs)

    monkeypatch.setattr(run.context.store, "load", tracked)
    run()
    assert len(run.calls) == 15
    assert not any(name == Tables.sec_def14a_text.name and "payload" in columns for name, columns in loads), loads
    print("\n=== SANITY: closed sections make no call (AC-003) ===")
    print("  a complete proxy (5 calls) and a spent one (5 + 5 failed calls, retry=2): third run 0 calls, 0 payload reads. Validated.")


def test_a_rejected_proxy_is_never_reserved(run: Run) -> None:
    rejected = marker_frame(
        Tables.sec_def14a_text, {"ticker": "ZZ", "accession_number": "rejected", "filing_date": YESTERDAY, "cik": "0000000001", "form": "DEF 14A"}
    )
    run.context.store.save(Tables.sec_def14a_text, rejected)
    run()
    assert len(run.context.store.load(Tables.sec_def14a_text, markers=True)) == 1
    assert run.calls == [] and all(run.context.store.row_count(table) == 0 for table in _FACT_TABLES)
    print("\n=== SANITY: subject-guard marker (section=_empty) ===")
    print("  one stored rejection marker: 0 calls, 0 reservations, 0 rows in the five DEF 14A tables. Validated.")


def test_a_walked_proxy_with_no_payload_consumes_its_attempts(run: Run) -> None:
    run.seed("blank", YESTERDAY, {})
    assert run.context.store.load(Tables.sec_def14a_text).section.tolist() == ["_none"]
    run()
    assert run.attempts() == {table.name: [1] for table in _FACT_TABLES}
    run()
    run()
    assert run.calls == [] and run.attempts() == {table.name: [2] for table in _FACT_TABLES}
    print("\n=== SANITY: `_none` proxy ===")
    print("  every domain empty: attempt 1, then 2, then nothing; 0 provider calls over three runs. Validated.")


def test_the_prompt_is_cut_and_postprocess_gets_the_uncut_payload(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """F-P0-3: no real payload reaches the budget, so a synthetic one does; a NULL period reads back as ''."""
    head = PAYLOADS["governance"] + "\n"
    long_payload = head + "a" * (BUDGET - len(head)) + "TAIL-BEYOND-THE-BUDGET"
    run.answers.update(_complete_answers())
    run.seed("long", YESTERDAY, {**PAYLOADS, "governance": long_payload}, period=None)
    seen: dict[str, tuple[str, Any]] = {}
    original = fetch.postprocess_section

    def spy(section: Any, answer: Any, stamp: dict, saved: Any, parent: dict, source_payload: str = "") -> Any:
        seen[section.key] = (source_payload, stamp["period"])
        return original(section, answer, stamp, saved, parent, source_payload)

    monkeypatch.setattr(fetch, "postprocess_section", spy)
    run()
    user = next(user for name, user in run.calls if name == "GovernanceExtract")
    assert long_payload[:BUDGET] in user and "TAIL-BEYOND-THE-BUDGET" not in user
    assert seen["governance"] == (long_payload, "")
    assert all(seen[key][0] == PAYLOADS[key] for key in PAYLOADS if key != "governance")
    print("\n=== SANITY: prompt cut vs postprocess input ===")
    print(f"  {len(long_payload):,}-char governance payload: prompt holds its first {BUDGET:,} chars (configs.yml), not the tail;")
    print("  postprocess got the stored uncut payload for all 5 sections; NULL period -> ''. Validated.")


def test_new_archive_has_no_source(run: Run) -> None:
    """AC-009: a new answer is archived without the payload it was read from."""
    run.answers.update(_complete_answers())
    run.seed("fresh", YESTERDAY, PAYLOADS)
    run()
    responses = json.loads(run.context.store.load(Tables.def14a_llm).iloc[0].def14a_json)["_def14a_split"]["responses"]
    entries = [entry for section in responses.values() for entry in section]
    assert len(entries) == 5 and all(set(entry) == {"response"} for entry in entries)
    print("\n=== SANITY: new archive ===")
    print(f"  {len(entries)} focused responses archived, none carries a `source`. Validated.")


def _old_archive(answer: Any, sources: list[str]) -> str:
    """A pre-split-change archive: one focused response per source, each with its embedded `source`."""
    entries = [{"response": json.loads(answer.model_dump_json()), "source": source} for source in sources]
    return json.dumps({"_def14a_split": {"version": 1, "legacy_json": None, "responses": {EXECUTIVE_PAY.key: entries}}})


def test_source_for_answer_keeps_its_gate() -> None:
    """D-07 amended: one matching focused response -> its embedded source, else the stored payload; otherwise ''."""
    answer = ExecutivePayExtract(currency="USD", amount_scale="units", compensation=[{"name": "Jane Roe", "fiscal_year": 2024, "salary": 1}])
    other = ExecutivePayExtract(currency="USD", amount_scale="units")
    legacy = Def14AExtract(ceo_name="Jane Roe").model_dump_json()
    legacy_answer = archived_answers(legacy)[EXECUTIVE_PAY.key][0]

    assert source_for_answer(legacy, EXECUTIVE_PAY, legacy_answer, "stored") == ""
    assert source_for_answer(_old_archive(answer, ["embedded"]), EXECUTIVE_PAY, answer, "stored") == "embedded"
    assert source_for_answer(append_answer(None, EXECUTIVE_PAY, answer), EXECUTIVE_PAY, answer, "stored") == "stored"
    assert source_for_answer(_old_archive(answer, ["one", "two"]), EXECUTIVE_PAY, answer, "stored") == ""
    assert source_for_answer(append_answer(None, EXECUTIVE_PAY, answer), EXECUTIVE_PAY, other, "stored") == ""
    print("\n=== SANITY: source_for_answer gate ===")
    print("  legacy whole-proxy -> ''; embedded source -> it; new answer -> stored payload; two sources or no match -> ''. Validated.")


def test_replay_over_an_old_archive_matches_the_new_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-009: the free replay of an archive with an embedded `source` and of one without give the same rows."""
    source = "Name\tYear\tSalary ($)\tBonus ($)\tOther Annual Compensation ($)\nJane Roe\t1995\t570,000\t\t69,721"
    answer = ExecutivePayExtract(currency="USD", amount_scale="units", compensation=[{"name": "Jane Roe", "fiscal_year": 1995, "salary": 570000}])
    stored = {}
    for label, raw in (("old", _old_archive(answer, [source])), ("new", append_answer(None, EXECUTIVE_PAY, answer))):
        run = Run(tmp_path / f"{label}.db", monkeypatch)
        stamp = {"ticker": "ZZ", "accession_number": "replayed", "as_of": YESTERDAY}
        run.context.store.save(Tables.def14a_llm, pd.DataFrame([{**stamp, "def14a_json": raw}]))
        run.context.store.save(
            Tables.def14a_executive_comp,
            pd.DataFrame([{**stamp, "cik": "1", "name": "Jane Roe", "fiscal_year": 1995, "salary": 570000, "bonus": 69721, "retry": 2}]),
        )
        run.seed("replayed", YESTERDAY, {EXECUTIVE_PAY.key: source})
        run(replay=True)
        assert run.calls == []
        stored[label] = run.context.store.load(Tables.def14a_executive_comp).drop(columns=["updated_at"], errors="ignore")
    pd.testing.assert_frame_equal(stored["old"], stored["new"])
    assert pd.isna(stored["new"].iloc[0].bonus) and stored["new"].iloc[0].retry == 2
    print("\n=== SANITY: replay over old vs new archive (AC-009) ===")
    print("  embedded-source archive and source-free archive replay to identical executive rows (bonus cleared, retry 2), 0 calls. Validated.")


def test_proxies_before_the_floor_are_not_sent(run: Run) -> None:
    run.answers.update(_complete_answers())
    run.seed("old", YESTERDAY - pd.DateOffset(years=20), PAYLOADS)
    run.seed("recent", YESTERDAY, PAYLOADS)
    run()
    assert len(run.calls) == 5 and run.context.store.load(Tables.def14a_llm).accession_number.tolist() == ["recent"]
    print("\n=== SANITY: window floor (M6) ===")
    print("  years_history=15: the 20-year-old stored proxy is not sent; the recent one gets its 5 calls. Validated.")
