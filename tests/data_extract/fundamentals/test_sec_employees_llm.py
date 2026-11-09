"""The `sec-employees-llm` step reads only `sec_employees_text` (filled here by the real walk on a seeded index) and
decides each undecided filing date once; dates behind a filing the walk has not stored yet wait for it.

Offline: a real SQLite `DataStore`, a seeded local EDGAR index, filing stand-ins and counting fake extractors (no
provider, no LLM call); the tickers run concurrently, one extractor each.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fundamentals_employees as mod
from src.data_extract.utils.edgar_extract import edgar_driver
from src.data_extract.utils.edgar_extract.edgar_driver import run_edgar_fetch
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_store.schema import Tables
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from src.utils.string import run_date
from tests.data_extract.edgar_fixtures import FakeEmployeesLLM, TenK, fake_context, patch_index_filings, roster_run, seed_index
from tests.data_extract.edgar_fixtures import employee_answer as answer

TODAY = run_date(None)


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


def decide(ctx: Any, extractor: FakeEmployeesLLM, *, years_history: int = 15, full: bool = False) -> pd.DataFrame:
    mod.fetch_employees_llm(ctx, ["AAA"], years_history, [extractor], full=full)  # type: ignore[list-item]
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
    extractor = FakeEmployeesLLM(answers)

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
    extractor = FakeEmployeesLLM(answers)
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
    extractor = FakeEmployeesLLM({accession(year): answer(n, body(n)) for year, n in counts.items()})

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
    extractor = FakeEmployeesLLM({stated.accession_number: answer(41000, body(41000))})

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
    extractor = FakeEmployeesLLM({f.accession_number: answer(n, body(n)) for f, n in ((old, 9000), (recent, 41000))})

    decided = decide(ctx, extractor, years_history=15)

    stored = ctx.store.load(Tables.sec_employees_text)["accession_number"].tolist()
    assert sorted(stored) == sorted([old.accession_number, recent.accession_number])
    assert extractor.calls == [recent.accession_number] and len(decided) == 1
    print("\n=== SANITY CHECK: window floor (M6) ===")
    print(f"  both filings stored; a 15-year run (floor {(TODAY - pd.DateOffset(years=15)).date()}) sent only the 2024 one.")


INVALID_JSON = "ValidationError: 1 validation error for EmployeeAnswer\n  Invalid JSON: expected value at line 1 column 2 [type=json_invalid]"


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
    extractor = FakeEmployeesLLM(answers)

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

    class DownExtractor(FakeEmployeesLLM):
        def run(self, tasks: list[LlmTask]) -> list[LlmResult]:
            raise RuntimeError("provider down")

    decided = decide(ctx, DownExtractor({}))

    assert decided.empty and any("1 ticker(s) not decided, listed again next run: AAA" in line for line in ctx.warnings), ctx.warnings
    print("\n=== SANITY CHECK: failed ticker ===")
    print("  a ticker whose run raised saves nothing, is named in the coverage log and the run returns; it has no row, so it retries.")


class GatedLLM(FakeEmployeesLLM):
    """Answers only once `barrier.parties` tickers are calling at the same time; a one-ticker-at-a-time run breaks it."""

    def __init__(self, answers: dict[str, Any], barrier: threading.Barrier) -> None:
        super().__init__(answers)
        self.barrier = barrier

    def run(self, tasks: Iterable[LlmTask]) -> list[LlmResult]:
        tasks = list(tasks)
        self.barrier.wait()
        return super().run(tasks)


def test_tickers_run_concurrently_one_extractor_each(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """ON-1: `len(extractors)` tickers are decided at once, each on its own extractor, and store what a serial run stores."""
    monkeypatch.setattr(edgar_driver, "_sleep", lambda seconds: None)
    monkeypatch.setattr(mod, "load_identity", lambda context: None)
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"], fundamentals_workers=2, llm_text_budget={"employees": 60000})
    lock = threading.Lock()  # the SQLite test store is one shared connection; production saves on pooled connections
    for name in ("load", "save"):
        original = getattr(ctx.store, name)

        def serialised(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            with lock:
                return _original(*args, **kwargs)

        monkeypatch.setattr(ctx.store, name, serialised)
    filings = {
        ticker: [TenK(f"000000000{cik}-{year % 100:02d}-000001", f"{year}-03-01", body(cik * 1000 + year), cik=cik) for year in (2023, 2024)]
        for ticker, cik in (("AAA", 1), ("BBB", 2))
    }
    every = [f for part in filings.values() for f in part]
    seed_index(ctx, [(f.cik, f"{ticker} Inc", f.form, f.filing_date, f.accession_number) for ticker, part in filings.items() for f in part])
    patch_index_filings(monkeypatch, {f.accession_number: f for f in every})
    run_edgar_fetch(ctx, EMPLOYEES_TEXT_FETCH, roster_run(ctx, ["AAA", "BBB"], 15, as_of=TODAY, no_cap=True), max_workers=1)
    answers = {f.accession_number: answer(int(f.body.split()[-2].replace(",", "")), f.body) for f in every}

    barrier = threading.Barrier(2, timeout=20)
    gated = [GatedLLM(answers, barrier), GatedLLM(answers, barrier)]
    mod.fetch_employees_llm(ctx, ["AAA", "BBB"], 15, gated)  # type: ignore[arg-type]
    concurrent = ctx.store.load(Tables.fundamentals_employees).sort_values(["ticker", "as_of"], ignore_index=True)
    serial = FakeEmployeesLLM(answers)
    mod.fetch_employees_llm(ctx, ["AAA", "BBB"], 15, [serial], full=True)  # type: ignore[list-item]
    again = ctx.store.load(Tables.fundamentals_employees).sort_values(["ticker", "as_of"], ignore_index=True)

    served = sorted(sorted({call[:10] for call in extractor.calls}) for extractor in gated)
    assert served == [["0000000001"], ["0000000002"]], served
    assert len(concurrent) == 4 and concurrent["employees_total"].notna().all()
    pd.testing.assert_frame_equal(concurrent, again)
    assert sorted(serial.calls) == sorted(f.accession_number for f in every)
    print("\n=== SANITY CHECK: tickers run concurrently (ON-1) ===")
    print(f"  2 extractors x 1 ticker each, both tickers calling at once (barrier of 2 passed); {len(concurrent)} rows stored,")
    print("  identical to a one-extractor serial --full re-decision of the same filings; per-ticker saves, no extra call.")
