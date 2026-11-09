"""The `sec-employees-llm` step reads only `sec_employees_text` (filled here by the real walk on a seeded index) and
decides each undecided filing date once; dates behind a filing the walk has not stored yet wait for it.

Offline: a real SQLite `DataStore`, a seeded local EDGAR index, filing stand-ins and a counting fake extractor (no
provider, no LLM call).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fundamentals_employees as mod
from src.data_extract.utils.common.sec_io import TransientReadError
from src.data_extract.utils.edgar_extract import edgar_driver
from src.data_extract.utils.edgar_extract.edgar_driver import run_edgar_fetch
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_store.schema import Tables
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from src.utils.string import run_date
from tests.data_extract.edgar_fixtures import fake_context, patch_index_filings, roster_run, seed_index

TODAY = run_date(None)


class TenK:
    """A 10-K family stand-in: `text()` returns `body`; the first `fail` reads raise a transient SEC error."""

    def __init__(self, accession: str, filed: str, body: str, *, form: str = "10-K", fail: int = 0) -> None:
        self.accession_number, self.cik, self.filing_date, self.form = accession, 1, filed, form
        self.company, self.period_of_report, self.primary_document = "Fixture Co", "2020-12-31", None
        self.attachments: list[object] = []
        self.body, self.fail = body, fail

    def html(self) -> str:
        if self.fail > 0:
            self.fail -= 1
            raise TransientReadError(f"{self.accession_number}: HTTP 503")
        return ""

    def text(self) -> str:
        return self.body

    def full_text_submission(self) -> str:
        return ""


class CountingExtractor:
    """No LLM: each task's accession is recorded and answered with its fixture `EmployeeAnswer`."""

    def __init__(self, answers: dict[str, mod.EmployeeAnswer]) -> None:
        self.answers = answers
        self.calls: list[str] = []
        self.payloads: list[str] = []

    def run(self, tasks: list[LlmTask]) -> list[LlmResult]:
        tasks = list(tasks)
        self.calls += [str(task.meta["stamp"].accession_number) for task in tasks]
        self.payloads += [task.payload for task in tasks]
        return [LlmResult(seq=task.seq, task=task, parsed=self.answers[str(task.meta["stamp"].accession_number)]) for task in tasks]


def answer(total: int, quote: str) -> mod.EmployeeAnswer:
    return mod.EmployeeAnswer(
        reason="fixture",
        status="found",
        total=total,
        total_quote=quote,
        full_time=None,
        full_time_quote=None,
        part_time=None,
        part_time_quote=None,
        is_fte=False,
        includes_contractors=False,
        measurement_period="year end",
        qualifier=None,
    )


def body(count: int) -> str:
    return f"As of December 31, we had {count:,} employees."


def accession(year: int, n: int = 1) -> str:
    return f"0000000001-{year % 100:02d}-{n:06d}"


@pytest.fixture
def world(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A context over ticker AAA (roster CIK 1, no identity layer), its driver retry waits skipped."""
    monkeypatch.setattr(edgar_driver, "_sleep", lambda seconds: None)
    monkeypatch.setattr(mod, "load_identity", lambda context: None)
    return lambda **data_extract: fake_context(
        tmp_path, sqlite_store, ["AAA"], fundamentals_workers=1, llm_text_budget={"employees": 60000}, **data_extract
    )


def seed(ctx: Any, monkeypatch: pytest.MonkeyPatch, filings: list[TenK]) -> None:
    seed_index(ctx, [(1, "AAA Inc", f.form, f.filing_date, f.accession_number) for f in filings])
    patch_index_filings(monkeypatch, {f.accession_number: f for f in filings})


def walk(ctx: Any, *, years_history: int = 15, no_cap: bool = True) -> None:
    run_edgar_fetch(ctx, EMPLOYEES_TEXT_FETCH, roster_run(ctx, ["AAA"], years_history, as_of=TODAY, no_cap=no_cap), max_workers=1)


def decide(ctx: Any, extractor: CountingExtractor, *, years_history: int = 15, full: bool = False) -> pd.DataFrame:
    mod.fetch_employees_llm(ctx, ["AAA"], years_history, extractor, full=full)  # type: ignore[arg-type]
    stored = ctx.store.load(Tables.fundamentals_employees, optional=True)
    return pd.DataFrame(columns=mod.FRAME_COLUMNS) if stored is None else stored.sort_values("as_of", ignore_index=True)


def test_decided_dates_make_no_call(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world()
    filings = [TenK(accession(year), f"{year}-03-01", body(1000 * year)) for year in (2022, 2023, 2024)]
    seed(ctx, monkeypatch, filings)
    walk(ctx)
    quiet = answer(0, "").model_copy(update={"status": "not_disclosed", "total": None, "total_quote": None})
    paraphrase = answer(50, "The company employs 50 people.")  # not in the text, and far from the earlier count: no recovery
    answers = {accession(2022): answer(2022000, body(2022000)), accession(2023): quiet, accession(2024): paraphrase}
    extractor = CountingExtractor(answers)

    first = decide(ctx, extractor)
    n_first = len(extractor.calls)
    second = decide(ctx, extractor)
    n_second = len(extractor.calls) - n_first
    decide(ctx, extractor, full=True)
    n_full = len(extractor.calls) - n_first - n_second

    assert n_first == 3 and first["status"].tolist() == ["found", "not_disclosed", "unsupported"]
    assert first["employees_total"].iloc[0] == 2022000 and first["employees_total"].iloc[1:].isna().all()
    assert n_second == 0 and len(second) == 3 and n_full == 3
    assert first["source_document"].eq("primary").all() and first["cik"].eq("0000000001").all()
    print("\n=== SANITY CHECK: decided dates make no call (AC-003) ===")
    print(f"  night one: {n_first} call(s) -> found, not_disclosed, unsupported rows; rerun: {n_second} call(s), so NULL-status")
    print(f"  rows count as decided; --full re-decided {n_full}.")


def test_a_capped_walk_splitting_a_same_day_pair_holds_the_date(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world(max_documents_per_run=1)
    older = TenK(accession(2023), "2023-03-01", body(40000))
    original = TenK(accession(2024, 1), "2024-03-01", body(41000))
    amendment = TenK(accession(2024, 2), "2024-03-01", body(41500), form="10-K/A")
    answers = {f.accession_number: answer(n, body(n)) for f, n in ((older, 40000), (original, 41000), (amendment, 41500))}
    extractor = CountingExtractor(answers)
    seed(ctx, monkeypatch, [older])
    walk(ctx, no_cap=False)
    seed(ctx, monkeypatch, [older, original, amendment])

    walk(ctx, no_cap=False)  # cap 1: the newest of the pair (the amendment) only
    held = decide(ctx, extractor)
    calls_held = list(extractor.calls)
    walk(ctx, no_cap=False)
    filled = decide(ctx, extractor)

    assert calls_held == [older.accession_number] and pd.to_datetime(held["as_of"]).tolist() == [pd.Timestamp("2023-03-01")]
    assert extractor.calls[1:] == [original.accession_number, amendment.accession_number]
    assert filled["employees_total"].tolist() == [40000, 41500]
    print("\n=== SANITY CHECK: same-day pair split by the cap (H1) ===")
    print("  the capped walk stored the 10-K/A but not its same-day 10-K: 2024-03-01 waited (no call, no row);")
    print("  once both were stored it was decided once, the amendment superseding the original (41,500).")


def test_an_older_unwalked_filing_holds_back_newer_dates(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world()
    counts = {2022: 30000, 2023: 31000, 2024: 32000}
    filings = [TenK(accession(year), f"{year}-03-01", body(n), fail=10 if year == 2022 else 0) for year, n in counts.items()]
    seed(ctx, monkeypatch, filings)
    extractor = CountingExtractor({accession(year): answer(n, body(n)) for year, n in counts.items()})

    walk(ctx)  # SEC 503s: the 2022 filing is not stored and stays listed
    waiting = decide(ctx, extractor)
    calls_waiting = len(extractor.calls)
    filings[0].fail = 0
    walk(ctx)
    decided = decide(ctx, extractor)

    assert calls_waiting == 0 and waiting.empty
    assert sorted(extractor.calls) == sorted(f.accession_number for f in filings) and len(decided) == 3
    assert any("2 date(s) held back" in line for line in ctx.infos), ctx.infos
    print("\n=== SANITY CHECK: older unwalked date (H1) ===")
    print("  2022 failed in the walk, so 2023 and 2024 waited behind it (0 calls, 2 dates held back); after the next")
    print(f"  walk all {len(decided)} dates were decided in filing order, so quote recovery sees the 2022 count first.")


def test_marker_rows_make_no_call_and_no_row(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world()
    blank = TenK(accession(2023), "2023-03-01", "   ")
    stated = TenK(accession(2024), "2024-03-01", body(41000))
    seed(ctx, monkeypatch, [blank, stated])
    walk(ctx)
    extractor = CountingExtractor({stated.accession_number: answer(41000, body(41000))})

    decided = decide(ctx, extractor)

    markers = ctx.store.load(Tables.sec_employees_text, markers=True)
    assert markers["source_document"].tolist().count("_empty") == 1
    assert extractor.calls == [stated.accession_number] and decided["accession_number"].tolist() == [stated.accession_number]
    print("\n=== SANITY CHECK: empty-filing marker (D-11) ===")
    print("  the blank 2023 filing is a walk marker: no call and no row; it is walked, so 2024 is not held back and is decided.")


def test_the_window_floor_bounds_the_worklist(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world()
    old = TenK(accession(2001), "2001-03-01", body(9000))
    recent = TenK(accession(2024), "2024-03-01", body(41000))
    seed(ctx, monkeypatch, [old, recent])
    walk(ctx, years_history=31)
    extractor = CountingExtractor({f.accession_number: answer(n, body(n)) for f, n in ((old, 9000), (recent, 41000))})

    decided = decide(ctx, extractor, years_history=15)

    stored = ctx.store.load(Tables.sec_employees_text)["accession_number"].tolist()
    assert sorted(stored) == sorted([old.accession_number, recent.accession_number])
    assert extractor.calls == [recent.accession_number] and len(decided) == 1
    print("\n=== SANITY CHECK: window floor (M6) ===")
    print(f"  both filings stored; a 15-year run (floor {(TODAY - pd.DateOffset(years=15)).date()}) sent only the 2024 one.")


INVALID_JSON = "ValidationError: 1 validation error for EmployeeAnswer\n  Invalid JSON: expected value at line 1 column 2 [type=json_invalid]"


class FlakyExtractor(CountingExtractor):
    """A failed result (its error, no answer) for each accession whose fixture answer is an error string."""

    def run(self, tasks: list[LlmTask]) -> list[LlmResult]:
        tasks = list(tasks)
        self.calls += [str(task.meta["stamp"].accession_number) for task in tasks]
        results = []
        for task in tasks:
            reply = self.answers[str(task.meta["stamp"].accession_number)]
            failed = isinstance(reply, str)
            results.append(LlmResult(seq=task.seq, task=task, parsed=None if failed else reply, error=reply if failed else None))
        return results


def test_one_invalid_llm_answer_fails_only_its_own_filing_date(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world()
    good = {year: 100 * year for year in range(2012, 2020)}
    filings = [TenK(accession(year), f"{year}-03-01", body(n)) for year, n in good.items()]
    filings += [
        TenK(accession(2020), "2020-03-01", body(9999)),
        TenK(accession(2021, 1), "2021-03-01", body(7000)),
        TenK(accession(2021, 2), "2021-03-01", body(7100), form="10-K/A"),
    ]
    seed(ctx, monkeypatch, filings)
    walk(ctx)
    answers: dict[str, Any] = {accession(year): answer(n, body(n)) for year, n in good.items()}
    answers |= {accession(2020): INVALID_JSON, accession(2021, 1): answer(7000, body(7000)), accession(2021, 2): INVALID_JSON}
    extractor = FlakyExtractor(answers)

    night_one = decide(ctx, extractor)
    sent_one = len(extractor.calls)
    decide(ctx, extractor)

    assert sent_one == 11 and pd.to_datetime(night_one["as_of"]).dt.year.tolist() == list(good)
    assert sorted(extractor.calls[sent_one:]) == sorted([accession(2020), accession(2021, 1), accession(2021, 2)])
    assert any("2 filing(s) not decided (LLM call failed), listed again next run" in line for line in ctx.warnings), ctx.warnings
    print("\n=== SANITY CHECK: one invalid answer ===")
    print(f"  {sent_one} calls, 2 invalid: the 8 good dates are stored; the failed filing and the one sharing its date get no row")
    print("  and are the only 3 filings re-sent the next night.")


def test_a_failed_ticker_is_logged_and_the_run_exits_cleanly(world: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = world()
    seed(ctx, monkeypatch, [TenK(accession(2024), "2024-03-01", body(41000))])
    walk(ctx)

    class DownExtractor(CountingExtractor):
        def run(self, tasks: list[LlmTask]) -> list[LlmResult]:
            raise RuntimeError("provider down")

    decided = decide(ctx, DownExtractor({}))

    assert decided.empty and any("1 ticker(s) not decided, listed again next run: AAA" in line for line in ctx.warnings), ctx.warnings
    print("\n=== SANITY CHECK: failed ticker ===")
    print("  a ticker whose run raised saves nothing, is named in the coverage log and the run returns; it has no row, so it retries.")
