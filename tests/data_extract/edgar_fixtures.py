"""Offline fixtures for the EDGAR document driver: a fake `Context`, a seeded local index, fake filings, and the
employees walk/LLM stand-ins (a 10-K, a fixture answer, a fake extractor).

The index cache is written through `edgar_index`'s own parser and writer, so tests read it exactly
as production does. Fake filings stand in for `edgar.Filing` by accession (`patch_index_filings`).
"""

from __future__ import annotations

import queue
import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import FilingScope, Identity, build_identity
from src.data_extract.utils.common.sec_io import TransientReadError
from src.data_extract.utils.edgar_extract import edgar_driver, edgar_index
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarRun
from src.data_extract.utils.fundamentals.fundamentals_employees import EmployeeAnswer, employee_task
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from src.utils.string import pad_cik
from tests.data_extract.fake_context import extract_config
from tests.fixtures.superinvestor_config import write_roster_config
from tests.support.universe import as_roster, seed_universe

MASTER_HEADER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    September 30, 2026
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/
Cloud HTTP:            https://www.sec.gov/Archives/




CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
"""


class TenK:
    """A 10-K family stand-in: `text()` returns `body` (no HTML, no exhibit); the first `fail` reads raise a transient
    SEC error; `_sgml` stands for edgartools' cached submission."""

    def __init__(
        self, accession: str, filed: str, body: str, *, cik: int = 1, form: str = "10-K", report: str | None = "2020-12-31", fail: int = 0
    ) -> None:
        self.accession_number, self.cik, self.filing_date, self.form = accession, cik, filed, form
        self.company, self.period_of_report, self.primary_document = "Fixture Co", report, None
        self.attachments: list[object] = []
        self._sgml: object | None = object()
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


def employee_answer(
    total: int | None,
    quote: str | None,
    *,
    full_time: int | None = None,
    full_time_quote: str | None = None,
    part_time: int | None = None,
    part_time_quote: str | None = None,
    is_fte: bool = False,
    includes_contractors: bool = False,
    status: str = "found",
    measurement_period: str | None = "year end",
) -> EmployeeAnswer:
    """A fake LLM answer: `total` with `quote`, plus optional full-/part-time components with their own quotes."""
    return EmployeeAnswer(
        status=status,  # type: ignore[arg-type]
        total=total,
        total_quote=quote,
        full_time=full_time,
        full_time_quote=full_time_quote,
        part_time=part_time,
        part_time_quote=part_time_quote,
        is_fte=is_fte,
        includes_contractors=includes_contractors,
        measurement_period=measurement_period,
        qualifier=None,
        reason="fixture",
    )


class FakeEmployeesLLM:
    """No LLM: each task is answered with its accession's fixture answer (a string answer is a failed call carrying
    that error); `tasks` records what was sent, `calls` their accessions."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.tasks: list[LlmTask] = []

    @property
    def calls(self) -> list[str]:
        return [str(task.meta["stamp"].accession_number) for task in self.tasks]

    def run(self, tasks: Iterable[LlmTask]) -> list[LlmResult]:
        tasks = list(tasks)
        self.tasks += tasks
        results = []
        for task in tasks:
            reply = self.answers[str(task.meta["stamp"].accession_number)]
            failed = isinstance(reply, str)
            results.append(LlmResult(seq=task.seq, task=task, parsed=None if failed else reply, error=reply if failed else None))
        return results


def lent(*extractors: Any) -> queue.SimpleQueue:
    """The extractor queue the employees per-ticker worker borrows from."""
    pool: queue.SimpleQueue = queue.SimpleQueue()
    for extractor in extractors:
        pool.put(extractor)
    return pool


def sent_excerpt(row: Any, limit: int) -> str:
    """The excerpt `employee_task` sends for a stored `sec_employees_text` row when `limit` characters are left after
    its prefix (the prefix length read from an unbounded task)."""
    probe = employee_task(0, "AAA", row, 10**9)
    prefix = len(probe.payload) - len(str(probe.meta["source_text"]))
    return str(employee_task(0, "AAA", row, limit + prefix).meta["source_text"])


def master_text(rows: Iterable[tuple[int | str, str, str, str, str]]) -> str:
    """A `master.idx` body for `(cik, company, form, filed, accession)` rows."""
    rows = list(rows)
    lines = [f"{int(cik)}|{company}|{form}|{filed}|edgar/data/{int(cik)}/{accession}.txt" for cik, company, form, filed, accession in rows]
    boundary = max(pd.Timestamp(row[3]) for row in rows).to_period("Q").end_time if rows else pd.Timestamp("2026-09-30")
    header = MASTER_HEADER.replace("September 30, 2026", boundary.strftime("%B %d, %Y"))
    return header + "\n".join(lines) + "\n"


def fake_context(tmp_path, store, tickers: list[str], ciks: list[str] | None = None, **data_extract: Any) -> Any:
    """A Context stand-in for the driver: `universe_tickers` seeded (name and GICS columns `<col>-<ticker>`) as its
    `universe`, and a `config_dir` holding empty roster overrides (`<config_dir>/superinvestors/overrides.json`)."""
    ciks = ciks or [str(i + 1) for i in range(len(tickers))]
    labels = ("name", "sector", "industry_group", "sub_industry")
    rows = {t: {"cik": pad_cik(c)} | {col: f"{col}-{t}" for col in labels} for t, c in zip(tickers, ciks, strict=True)}
    universe = seed_universe(store, tickers, rows=rows) if tickers else None
    warnings: list[str] = []
    errors: list[str] = []
    infos: list[str] = []
    ctx = types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=types.SimpleNamespace(
            info=lambda msg, *a: infos.append(msg % a if a else msg),
            warning=lambda msg, *a: warnings.append(msg % a if a else msg),
            error=lambda msg, *a: errors.append(msg % a if a else msg),
        ),
        config=extract_config(data_extract={"years_history": 15, **data_extract}),
        ensure_edgar_identity=lambda: None,
        config_dir=write_roster_config(tmp_path),
        universe=universe,
    )
    ctx.warnings, ctx.errors, ctx.infos = warnings, errors, infos
    return ctx


def seed_index(ctx: Any, rows: Iterable[tuple[int | str, str, str, str, str]]) -> pd.DataFrame:
    """Write `rows` into the local index cache, one Parquet file per quarter, through the production parser and
    writer."""
    directory = edgar_index.index_dir(ctx)
    for path in directory.glob("*.parquet"):
        path.unlink()
    quarters: dict[str, list[tuple[int | str, str, str, str, str]]] = {}
    for row in rows:
        quarters.setdefault(edgar_index.Quarter.of(pd.Timestamp(row[3])).name, []).append(row)
    frames = []
    for name, part in quarters.items():
        frame = edgar_index._parse_master_unit(master_text(part).encode("latin-1"))[0]
        edgar_index._write_atomic(frame, directory / f"{name}.parquet")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else edgar_index._parse_master_unit(master_text([]).encode("latin-1"))[0]


def patch_index_filings(monkeypatch: pytest.MonkeyPatch, filings: dict[str, Any]) -> None:
    """Make the driver read `filings[accession]` instead of building an `edgar.Filing` from the index row."""
    real = edgar_index.index_filing

    def _filing(cik: object, company: object, form: object, filed: object, accession: object) -> Any:
        return filings.get(str(accession)) or real(cik, company, form, filed, accession)

    monkeypatch.setattr(edgar_index, "index_filing", _filing)


def fake_filing(accession: str, cik: int | str, filed: str, form: str = "8-K", **attrs: Any) -> types.SimpleNamespace:
    """A filing stand-in carrying what `FilingStamp.of` reads, plus `attrs`."""
    return types.SimpleNamespace(accession_number=accession, cik=int(cik), form=form, filing_date=filed, company="Fixture Co", **attrs)


def identity_for(roster: dict[str, str]) -> Identity:
    """A real `Identity` where each roster ticker is its own single-CIK entity owning its symbol."""
    return build_identity(
        lineage=pd.DataFrame([{"cik": pad_cik(cik), "company_id": f"E{pad_cik(cik)}", "source": "roster"} for cik in roster.values()]),
        tenure=pd.DataFrame(
            [
                {
                    "symbol": t,
                    "issuer_cik": pad_cik(c),
                    "valid_from": pd.Timestamp("2000-01-01"),
                    "valid_to": None,
                    "n_filings": 10,
                    "source": "form345",
                }
                for t, c in roster.items()
            ]
        ),
        roster_cik=as_roster(pd.DataFrame([{"ticker": t, "cik": pad_cik(c)} for t, c in roster.items()])),
    )


@dataclass(frozen=True)
class RosterIdentity:
    """An `Identity` stand-in for walks that read only the filing scope: each ticker's roster CIK alone (two tickers may
    share a CIK, and a blank CIK scopes nothing), which a built `Identity` refuses."""

    roster: dict[str, str]

    def filing_scope(self, ticker: str) -> FilingScope:
        """`ticker`'s roster CIK as its only window."""
        return FilingScope.roster_only(ticker, self.roster[ticker])


def roster_identity(roster: dict[str, str]) -> Identity:
    """A `RosterIdentity` typed as the `Identity` the walks take."""
    return cast(Identity, RosterIdentity({ticker: pad_cik(cik) if cik else "" for ticker, cik in roster.items()}))


def roster_run(ctx: Any, tickers: list[str], years_history: int = 15, *, as_of: pd.Timestamp, full: bool = False, no_cap: bool = False) -> EdgarRun:
    """`EdgarRun.load` over a roster-only identity: each key's filing scope is its roster CIK alone."""
    roster = edgar_driver.universe_rows(ctx, tickers)
    identity = roster_identity(dict(zip(roster["ticker"], roster["cik"], strict=True)))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(edgar_driver, "load_identity", lambda context: identity)
        return EdgarRun.load(ctx, tickers, years_history, as_of=as_of, full=full, no_cap=no_cap)


def walk_call(fetch: EdgarFetch, run: EdgarRun) -> dict[str, Any]:
    """One driver call as the flat options it carries: the fetch, tickers, CIK mapping, window, run date, `full`, `no_cap`."""
    return {
        "fetch": fetch,
        "tickers": run.tickers,
        "cik_map": run.cik_map,
        "years_history": run.years_history,
        "as_of": run.run_date,
        "full": run.full,
        "no_cap": run.cap is None,
    }


def stub_edgar_step(monkeypatch: pytest.MonkeyPatch, driver: Any) -> None:
    """Make `StepExtractEdgar` constructible offline: the CIK mapping, the identity and the index refresh stubbed, each
    driver call handed to `driver(context, **walk_call(fetch, run))`."""
    from src.data_extract.transformers import step_extract_edgar

    monkeypatch.setattr(edgar_driver, "universe_rows", lambda ctx, names: pd.DataFrame({"ticker": names, "cik": ["0"] * len(names)}))
    monkeypatch.setattr(edgar_driver, "load_identity", lambda ctx: None)
    monkeypatch.setattr(step_extract_edgar, "refresh_index", lambda ctx, run, years_history: None)
    monkeypatch.setattr(edgar_index, "refreshed_today", lambda ctx, years_history: False)
    monkeypatch.setattr(step_extract_edgar, "run_edgar_fetch", lambda ctx, fetch, run: driver(ctx, **walk_call(fetch, run)))
