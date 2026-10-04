"""DEF 14A LLM: the incremental up-to-date check must be per-TICKER (not date+count),
and the new board-technology-maturity fields must flatten into the output row.
"""

from __future__ import annotations

import logging
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.schemas.def14a_schema import BeneficialOwner as _BeneficialOwner
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract as _Def14AExtract
from src.data_extract.utils.schemas.def14a_schema import DirectorCompensation as _DirectorCompensation
from src.data_extract.utils.schemas.def14a_schema import DirectorInfo as _DirectorInfo
from src.data_extract.utils.schemas.def14a_schema import ExecutiveCompensation as _ExecutiveCompensation
from src.data_extract.utils.schemas.def14a_schema import GovernanceProfile as _GovernanceProfile
from src.data_extract.utils.structure.def14a import flatten as flatten_mod
from src.data_extract.utils.structure.def14a.fetch import _is_up_to_date, _subject_is_accepted
from src.data_extract.utils.structure.def14a.flatten import _flatten, _result_frames
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.fake_context import extract_config

#: A stand-in identity whose every ticker is listed from its roster CIK alone.
_ROSTER_ONLY = SimpleNamespace(filing_scope=lambda ticker: FilingScope.roster_only(ticker, "1"))

BeneficialOwner: Any = _BeneficialOwner
Def14AExtract: Any = _Def14AExtract
DirectorCompensation: Any = _DirectorCompensation
DirectorInfo: Any = _DirectorInfo
ExecutiveCompensation: Any = _ExecutiveCompensation
GovernanceProfile: Any = _GovernanceProfile


def _completed_parent_row(ticker: str, accession: str, as_of: str) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in flatten_mod._DEF14A_EVIDENCE_COLUMNS}
    row.update({"ticker": ticker, "accession_number": accession, "as_of": as_of, "ceo_name_proxy": "Already extracted"})
    return row


def _ctx(tmp_path: Path, tickers: list[str], write_meta_today: bool = True) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    ds.save("def14a_llm", pd.DataFrame([_completed_parent_row(t, f"acc-{t}", "2024-04-01") for t in tickers]))
    ctx: Any = types.SimpleNamespace(store=ds, paths={"DATA_STORE": tmp_path}, config_dir="configs", config=extract_config())
    if write_meta_today:
        record_run(ctx, "def14a_llm", len(tickers), 0, is_full_rescan=True)
    return ctx


def test_up_to_date_is_per_ticker_not_date_count(tmp_path):
    ctx = _ctx(tmp_path, ["AAPL", "MSFT"])
    # every requested ticker present + built today -> up to date (skip)
    assert _is_up_to_date(ctx, ["AAPL", "MSFT"]) is True
    # a MISSING ticker must NOT be skipped, even though it was "built today" for 2
    # (this is the '~15 tickers then it stops' bug -> now fixed)
    assert _is_up_to_date(ctx, ["AAPL", "MSFT", "NVDA"]) is False

    # no meta today -> not up to date (re-scan, picks up new annual proxies)
    ctx2 = _ctx(tmp_path / "b", ["AAPL", "MSFT"], write_meta_today=False)
    assert _is_up_to_date(ctx2, ["AAPL", "MSFT"]) is False

    print("\n=== SANITY: DEF 14A incremental is per-ticker ===")
    print(
        "  all requested present -> skip; a missing ticker (NVDA) -> NOT skipped "
        "(re-processes it); no meta -> re-scan. date+count bug fixed. Validated."
    )


@pytest.mark.parametrize(
    "label, extract, expected",
    [
        ("empty", Def14AExtract(), False),
        ("metadata", Def14AExtract(company_name="ACME", fiscal_year=2025), False),
        (
            "inferred false defaults",
            Def14AExtract(governance=GovernanceProfile(classified_board=False, dual_class_shares=False)),
            False,
        ),
        ("classified true", Def14AExtract(governance=GovernanceProfile(classified_board=True)), True),
        ("dual class true", Def14AExtract(governance=GovernanceProfile(dual_class_shares=True)), True),
        ("explicit founder false", Def14AExtract(ceo_is_founder=False), True),
        (
            "explicit majority false",
            Def14AExtract(governance=GovernanceProfile(majority_voting_for_directors=False)),
            True,
        ),
        (
            "numeric zero",
            Def14AExtract(governance=GovernanceProfile(insider_ownership_pct=0.0)),
            True,
        ),
        (
            "ownership only",
            Def14AExtract(ownership_holders=[BeneficialOwner(holder_name="Example Fund", holder_type="5pct_holder")]),
            True,
        ),
        ("director only", Def14AExtract(directors=[DirectorInfo(name="Jane Director")]), True),
        (
            "executive compensation only",
            Def14AExtract(compensation=[ExecutiveCompensation(name="Jane CEO", title="CEO", fiscal_year=2025)]),
            True,
        ),
        (
            "director compensation only",
            Def14AExtract(director_compensation=[DirectorCompensation(name="Jane Director", fiscal_year=2025)]),
            True,
        ),
        ("ceo only", Def14AExtract(ceo_name="Jane CEO"), True),
        ("auditor only", Def14AExtract(governance=GovernanceProfile(auditor_name="Example Audit LLP")), True),
        ("board only", Def14AExtract(governance=GovernanceProfile(board_size=8)), True),
        ("blank scalar", Def14AExtract(ceo_name="   "), False),
    ],
)
def test_evidence_gate_parsed_stored_and_persistence_agree(label, extract, expected):
    filing = pd.Series(
        {
            "accession_number": f"acc-{label}",
            "filing_date": pd.Timestamp("2025-04-01"),
            "period_of_report": "2024-12-31",
        }
    )
    row = _flatten("ZZ", filing, extract)
    projected = {column: row.get(column) for column in flatten_mod._DEF14A_EVIDENCE_COLUMNS}
    task = LlmTask(seq=0, payload="proxy", schema=Def14AExtract, table=Tables.def14a_llm, meta={"ticker": "ZZ", "filing": filing})
    frames = _result_frames(LlmResult(seq=0, task=task, parsed=extract))

    assert flatten_mod._has_extract_evidence(extract) is expected, label
    assert flatten_mod._has_parent_evidence(projected) is expected, label
    assert (Tables.def14a_llm in frames) is expected, label

    print(f"\n=== SANITY: DEF 14A evidence gate — {label} ===")
    print(f"  parsed={expected}, stored={expected}, parent persisted={expected}. Validated.")


def _daily_context(tmp_path: Path) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        store=DataStore(create_engine(f"sqlite:///{tmp_path / 'daily.db'}")),
        log=logging.getLogger("test.def14a.daily"),
        paths={"DATA_STORE": tmp_path},
        config_dir="configs",
        config=extract_config(data_extract={"years_history": 15, "manifest_full_rescan_days": 30}),
    )


def _listed_filing(accession: str, filing_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "ZZ",
                "cik": "0000000001",
                "company_name": "Example Corp",
                "accession_number": accession,
                "doc_url": f"https://example.invalid/{accession}",
                "txt_url": f"https://example.invalid/{accession}.txt",
                "filing_date": filing_date,
                "period_of_report": str(filing_date.date()),
                "form": "DEF 14A",
            }
        ]
    )


def _save_parent(context: Any, accession: str, extract: Any, filing_date: pd.Timestamp) -> None:
    filing = _listed_filing(accession, filing_date).iloc[0]
    frame = pd.DataFrame([_flatten("ZZ", filing, extract)])
    for column in ("as_of", "period"):
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    context.store.save(Tables.def14a_llm, frame)


def _extractor_double(responses: list[BaseException | Any], tasked: list[str]):
    class _Extractor:
        def __init__(self, context, config, action=None, threads=None, methodes=None):
            del config, action, threads, methodes
            self._context = context

        def run_extraction(self, tasks, flatten=None, group_key=None):
            del group_key
            results = []
            for task in list(tasks):
                tasked.append(str(task.meta["filing"]["accession_number"]))
                answer = responses.pop(0)
                if isinstance(answer, BaseException):
                    results.append(LlmResult(seq=task.seq, task=task, parsed=None, error=f"{type(answer).__name__}: {answer}"))
                else:
                    results.append(LlmResult(seq=task.seq, task=task, parsed=answer))

            frames: dict[Any, list[pd.DataFrame]] = {}
            for result in results:
                if not result.ok:
                    continue
                for table, frame in (flatten(result) if flatten is not None else {}).items():
                    frames.setdefault(table, []).append(frame)
            for table, parts in frames.items():
                frame = pd.concat(parts, ignore_index=True)
                for column in ("as_of", "period"):
                    if column in frame:
                        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
                self._context.store.save(table, frame)
            return results

    return _Extractor


def _install_daily_fetch_doubles(monkeypatch, extractor, filing: pd.DataFrame, listed_since: list[pd.Timestamp | None]) -> None:
    from src.data_extract.utils.structure.def14a import fetch as mod

    def _list(context, scope, company, years, since):
        del context, scope, company, years
        listed_since.append(since)
        return filing.copy()

    monkeypatch.setattr(mod, "LLMExtractor", extractor)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *_: pd.DataFrame([{"ticker": "ZZ", "cik": "0000000001", "name": "Example Corp"}]))
    monkeypatch.setattr(mod, "load_identity", lambda context: _ROSTER_ONLY)
    monkeypatch.setattr(mod, "_list_scope_windows", _list)
    monkeypatch.setattr(mod, "_payload_for", lambda *_: "=== BOARD OF DIRECTORS ===\nJane Director")
    monkeypatch.setattr(mod, "_finalise_gender", lambda *_: None)


def test_provider_failure_is_retried_through_real_daily_manifest_gates(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    accession = "daily-provider-failure"
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    filing = _listed_filing(accession, yesterday)
    listed_since: list[pd.Timestamp | None] = []
    tasked: list[str] = []
    responses: list[BaseException | Any] = [
        TypeError("Responses.parse() got an unexpected keyword argument 'seed'"),
        Def14AExtract(ceo_name="Jane CEO", directors=[DirectorInfo(name="Jane Director", age=55)]),
    ]
    _install_daily_fetch_doubles(monkeypatch, _extractor_double(responses, tasked), filing, listed_since)

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")
    assert not context.store.exists(Tables.def14a_llm), "day-D provider failure must not create a parent"

    record_run(context, Tables.def14a_llm, ticker_count=1, rows_added=0, is_full_rescan=True, run_date=yesterday, tickers=["ZZ"])
    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    assert listed_since[-1] == yesterday - pd.Timedelta(days=1)
    assert tasked == [accession, accession]
    parent = context.store.load(Tables.def14a_llm)
    assert parent is not None and list(parent["accession_number"]) == [accession]

    print("\n=== SANITY: provider failure retries on D+1 ===")
    print(f"  day D saved no parent; D+1 listed since {listed_since[-1].date()} and queued {accession} again. Validated.")


def test_legacy_empty_parent_is_repaired_through_real_daily_manifest_gates(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    accession = "daily-empty-parent"
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    _save_parent(
        context,
        accession,
        Def14AExtract(governance=GovernanceProfile(classified_board=False, dual_class_shares=False)),
        yesterday,
    )
    record_run(context, Tables.def14a_llm, ticker_count=1, rows_added=0, is_full_rescan=True, run_date=yesterday, tickers=["ZZ"])

    listed_since: list[pd.Timestamp | None] = []
    tasked: list[str] = []
    response = Def14AExtract(
        ceo_name="Jane CEO",
        directors=[DirectorInfo(name="Jane Director", age=55)],
        governance=GovernanceProfile(board_size=1),
    )
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([response], tasked), _listed_filing(accession, yesterday), listed_since)

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    assert listed_since == [yesterday - pd.Timedelta(days=1)]
    assert tasked == [accession], "the legacy empty key must not count as completed"
    parent = context.store.load(Tables.def14a_llm)
    directors = context.store.load(Tables.def14a_directors)
    assert parent is not None and len(parent) == 1
    assert parent.iloc[0]["ceo_name_proxy"] == "Jane CEO" and float(parent.iloc[0]["board_size"]) == 1.0
    assert directors is not None and len(directors) == 1 and directors.iloc[0]["accession_number"] == accession

    print("\n=== SANITY: legacy empty parent repairs on D+1 ===")
    print(f"  {accession} was re-listed, re-extracted, and upserted to one evidenced parent plus one director. Validated.")


def test_valid_parent_is_relisted_but_not_reextracted_next_day(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    accession = "daily-valid-parent"
    yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    _save_parent(context, accession, Def14AExtract(ceo_name="Jane CEO", governance=GovernanceProfile(board_size=8)), yesterday)
    record_run(context, Tables.def14a_llm, ticker_count=1, rows_added=1, is_full_rescan=True, run_date=yesterday, tickers=["ZZ"])

    listed_since: list[pd.Timestamp | None] = []
    tasked: list[str] = []
    payloads: list[str] = []
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([], tasked), _listed_filing(accession, yesterday), listed_since)
    monkeypatch.setattr(mod, "_payload_for", lambda *_: payloads.append(accession) or "unused")

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    parent = context.store.load(Tables.def14a_llm)
    assert listed_since == [yesterday - pd.Timedelta(days=1)]
    assert tasked == [] and payloads == [], "semantic completion must filter the re-listed accession before token spend"
    assert parent is not None and len(parent) == 1

    print("\n=== SANITY: valid parent stays idempotent on D+1 ===")
    print(f"  {accession} was re-listed through the real daily window but produced zero payloads and zero LLM tasks. Validated.")


def test_subject_guard_rejects_only_known_disjoint_subjects(monkeypatch, caplog):
    """A dissident filer is valid when the proxy subject is the intended issuer."""
    import logging

    from src.data_extract.utils.structure.def14a import fetch as mod

    filing = pd.Series(
        {
            "cik": "0002041610",
            "company_name": "Paramount Skydance Corp",
            "form": "DEFC14A",
            "filing_date": pd.Timestamp("2026-02-17"),
            "accession_number": "0001104659-26-016573",
        }
    )
    identity_calls: list[bool] = []
    context = SimpleNamespace(
        log=logging.getLogger("test.def14a.subject"),
        ensure_edgar_identity=lambda: identity_calls.append(True),
    )
    accepted = frozenset({"0000813828", "0002041610"})
    caplog.set_level(logging.INFO, logger="test.def14a.subject")

    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0001437107"}))
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is False
    assert all(value in caplog.text for value in ("PSKY", filing["accession_number"], filing["cik"], "0001437107", "subject_cik_disjoint"))

    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0000813828"}))
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is True

    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset())
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is True

    def _unreadable(_):
        raise ValueError("no SGML header")

    monkeypatch.setattr(mod, "_filing_subject_ciks", _unreadable)
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is True
    assert len(identity_calls) == 4

    print("\n=== SANITY: DEF 14A subject guard ===")
    print("  WBD subject rejected; intended-issuer dissident and unknown headers retained")
    print("  OK: only known disjoint subjects are blocked before paid extraction")


def test_disjoint_subject_never_becomes_an_llm_task(tmp_path, monkeypatch):
    import logging

    from src.data_extract.utils.structure.def14a import fetch as mod

    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    context: Any = SimpleNamespace(
        store=store,
        log=logging.getLogger("test.def14a.subject.loop"),
        paths={"DATA_STORE": tmp_path},
        config_dir="configs",
        config=extract_config(data_extract={"years_history": 15}),
        ensure_edgar_identity=lambda: None,
    )
    filing = pd.DataFrame(
        [
            {
                "cik": "0002041610",
                "company_name": "Paramount Skydance Corp",
                "form": "DEFC14A",
                "filing_date": pd.Timestamp("2026-02-17"),
                "period_of_report": "2026-02-17",
                "accession_number": "0001104659-26-016573",
                "doc_url": "unused",
                "txt_url": "unused",
            }
        ]
    )
    llm_tasks: list[object] = []
    payload_calls: list[str] = []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def run_extraction(self, tasks, **kwargs):
            del kwargs
            llm_tasks.extend(tasks)
            return []

    monkeypatch.setattr(mod, "LLMExtractor", FakeLLM)
    monkeypatch.setattr(mod, "_is_up_to_date", lambda *_: False)
    monkeypatch.setattr(
        mod, "load_cik_mapping", lambda *_: pd.DataFrame([{"ticker": "PSKY", "cik": "0002041610", "name": "Paramount Skydance Corp"}])
    )
    # PSKY's register chain makes it a multi-window scope, so its proxies' subject headers are checked.
    psky = dated_identity(
        [("PSKY", "0000813828", "cik_window", SENTINEL, "2025-08-07"), ("PSKY", "0002041610", "cik_window", "2025-08-07", None)],
        {"PSKY": "0002041610"},
    )
    monkeypatch.setattr(mod, "load_identity", lambda context: psky)
    monkeypatch.setattr(mod, "_list_scope_windows", lambda *_: filing)
    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0001437107"}))
    monkeypatch.setattr(mod, "_payload_for", lambda *args: payload_calls.append(str(args[2]["accession_number"])))

    mod.fetch_def14a_llm(context, context.config, ["PSKY"], model="gpt-5-mini")

    assert payload_calls == [] and llm_tasks == []
    print("\n=== SANITY: wrong-subject filing stops before the LLM ===")
    print("  WBD accession produced zero payload fetches and zero LLM tasks")
    print("  OK: rejection precedes document carving and paid extraction")


def test_gap_fill_lists_full_window_and_skips_present(tmp_path, monkeypatch):
    """Gap-filling on a FRESH manifest (no recorded run yet -> full rescan, per
    `run_manifest.manifest_window`): the FULL window is listed (no `since` cutoff)
    and the LLM runs ONLY on filings whose accession is not already in the table —
    so a HOLE in the middle (2023 here) is filled while the present years (2022,
    2024) are skipped. Uses an in-memory SQLite store (no Postgres)."""
    import logging

    from src.data_extract.utils.structure.def14a import fetch as mod

    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    ds.save(
        "def14a_llm",
        pd.DataFrame(
            [  # 2022 + 2024 present; 2023 is a HOLE
                _completed_parent_row("ZZ", "a2022", "2022-04-01"),
                _completed_parent_row("ZZ", "a2024", "2024-04-01"),
            ]
        ),
    )
    ctx: Any = types.SimpleNamespace(
        store=ds,
        log=logging.getLogger("t"),
        paths={"DATA_STORE": tmp_path},
        config_dir="configs",
        config=extract_config(data_extract={"years_history": 15}),
    )

    listed_since, extracted = [], []

    def _fake_list(context, cik, forms, years, company_name="", since=None, cache_dir=None):  # mirrors edgar_fillings.list_filings EXACTLY
        # -- a stale stub binds `since` positionally
        listed_since.append(since)  # must be None now (full window)
        return pd.DataFrame(
            [
                {
                    "accession_number": a,
                    "doc_url": f"http://x/{a}",
                    "filing_date": pd.Timestamp(d),
                    "period_of_report": "2000-12-31",
                    "form": "DEF 14A",
                }
                for a, d in [("a2022", "2022-04-01"), ("a2023", "2023-04-01"), ("a2024", "2024-04-01"), ("a2025", "2025-04-01")]
            ]
        )

    class _FakeLLM:
        """Stands in for the whole extracter: records which accessions became LLM tasks,
        then writes the rows the real `flatten` would have produced.

        The frame is written with a STRING `as_of` because this test runs on SQLite, whose
        driver cannot bind a pandas Timestamp.
        """

        def __init__(self, context, config, action=None, threads=None, methodes=None):
            self._context = context

        def run_extraction(self, tasks, flatten=None, group_key=None):
            tasks = list(tasks)
            rows = []
            for t in tasks:
                f = t.meta["filing"]
                extracted.append(f["accession_number"])
                rows.append({"ticker": t.meta["ticker"], "accession_number": f["accession_number"], "as_of": f["filing_date"], "def14a_json": "{}"})
            if rows:
                df = pd.DataFrame(rows)
                df["as_of"] = pd.to_datetime(df["as_of"]).dt.strftime("%Y-%m-%d")
                self._context.store.save("def14a_llm", df)
            return [SimpleNamespace(ok=True, task=t, parsed=Def14AExtract(ceo_name="Fresh extraction"), error=None) for t in tasks]

    monkeypatch.setattr(mod, "list_filings", _fake_list)
    monkeypatch.setattr(mod, "load_identity", lambda context: _ROSTER_ONLY)
    monkeypatch.setattr(mod, "_payload_for", lambda context, ticker, f: "=== CARVED ===")
    monkeypatch.setattr(mod, "LLMExtractor", _FakeLLM)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda _c, _t=None: pd.DataFrame({"ticker": ["ZZ"], "cik": ["0000000001"], "name": ["Z"]}))
    monkeypatch.setattr(mod, "_is_up_to_date", lambda _c, _n: False)

    mod.fetch_def14a_llm(ctx, ctx.config, tickers=["ZZ"], model="gpt-5-mini")

    assert listed_since == [None], "must list the FULL window (no since cutoff) to find gaps"
    assert set(extracted) == {"a2023", "a2025"}, f"only missing filings should hit the LLM: {extracted}"
    stored = ds.load("def14a_llm")
    assert stored is not None
    accs = set(stored.query("ticker == 'ZZ'")["accession_number"])
    assert accs == {"a2022", "a2023", "a2024", "a2025"}

    print("\n=== SANITY: DEF 14A gap-filling incremental ===")
    print(
        f"  had 2022+2024, listed full window (since={listed_since[0]}) -> LLM ran ONLY on the "
        f"missing {sorted(set(extracted))} (2023 hole + new 2025); 2 present skipped. Validated."
    )


def test_manifest_narrows_since_on_routine_rerun(tmp_path, monkeypatch):
    """A ROUTINE re-run (manifest already has a recent run for this table, same
    ticker count, rescan not due) must list only from the manifest's last run date
    onward -- not the full `years_history` window -- per `run_manifest.manifest_window`.
    This is the narrow-window counterpart to the full-rescan case exercised by
    `test_gap_fill_lists_full_window_and_skips_present` above."""
    import logging

    from src.data_extract.utils.common.run_manifest import record_run
    from src.data_extract.utils.structure.def14a import fetch as mod

    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    ds.save(
        "def14a_llm",
        pd.DataFrame(
            [
                _completed_parent_row("ZZ", "a2024", "2024-04-01"),
            ]
        ),
    )
    ctx: Any = types.SimpleNamespace(
        store=ds,
        log=logging.getLogger("t"),
        paths={"DATA_STORE": tmp_path},
        config_dir="configs",
        config=extract_config(data_extract={"years_history": 15}),
    )
    # A prior run 10 days ago, one ticker -- same ticker count as this run, and well
    # inside the (default 30-day) self-heal window, so `manifest_window` must return
    # the narrow cutoff, not the full-rescan fallback.
    last_run = pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
    record_run(ctx, "def14a_llm", ticker_count=1, rows_added=1, is_full_rescan=True, run_date=last_run, tickers=["ZZ"])

    listed_since = []

    def _fake_list(context, cik, forms, years, company_name="", since=None, cache_dir=None):  # mirrors edgar_fillings.list_filings EXACTLY
        # -- a stale stub binds `since` positionally
        listed_since.append(since)
        return pd.DataFrame(columns=["accession_number", "doc_url", "filing_date", "period_of_report", "form"])

    class _FakeLLM:
        """This ticker lists no filings, so the extracter is built and never used."""

        def __init__(self, context, config, action=None, threads=None, methodes=None):
            pass

        def run_extraction(self, tasks, flatten=None, group_key=None):
            return []

    monkeypatch.setattr(mod, "list_filings", _fake_list)
    monkeypatch.setattr(mod, "load_identity", lambda context: _ROSTER_ONLY)
    monkeypatch.setattr(mod, "LLMExtractor", _FakeLLM)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda _c, _t=None: pd.DataFrame({"ticker": ["ZZ"], "cik": ["0000000001"], "name": ["Z"]}))
    monkeypatch.setattr(mod, "_is_up_to_date", lambda _c, _n: False)

    mod.fetch_def14a_llm(ctx, ctx.config, tickers=["ZZ"], model="gpt-5-mini")

    # list_filings' own `since` is STRICTLY AFTER the date passed, so the manifest's
    # last run date (inclusive) is passed as (last_run - 1 day).
    assert listed_since == [last_run - pd.Timedelta(days=1)], f"routine rerun must narrow to the manifest cutoff, got {listed_since}"

    print("\n=== SANITY: DEF 14A manifest narrows the window on a routine rerun ===")
    print(
        f"  prior run {last_run.date()}, same ticker count, rescan not due -> "
        f"listed since={listed_since[0]} (inclusive of the prior run date). Validated."
    )


def test_flatten_surfaces_the_auditor_block():
    """`n_technology_directors` / `technology_committee` were REMOVED -- they were an opinion,
    not an extraction (mean |delta| of 1.06 directors between consecutive filings of the same
    company, only 38.8% unchanged, and wrong by 7x on HUBB 2022 whose own matrix states
    "Cybersecurity and Technology 78%" of 9 directors).

    What replaced them is the auditor block, which is the opposite kind of field: the firm name
    is present in 98% of documents and was the WORST column in the retired edgar table at 2.05%
    fill."""
    extract = Def14AExtract(
        company_name="ACME",
        fiscal_year=2024,
        governance=GovernanceProfile(
            board_size=10,
            auditor_name="Ernst & Young LLP",
            auditor_since_year=1934,
            auditor_fees_usd=12_000_000.0,
            audit_fees_audit_usd=9_000_000.0,
            audit_fees_audit_related_usd=1_000_000.0,
            audit_fees_tax_usd=1_500_000.0,
            audit_fees_other_usd=500_000.0,
            auditor_fees_prior_usd=11_000_000.0,
        ),
    )
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"), "period_of_report": "2023-12-31", "accession_number": "a1"})
    row = _flatten("ACME", filing, extract)
    assert row["auditor_name"] == "Ernst & Young LLP"
    assert row["auditor_since_year"] == 1934
    assert row["auditor_fees"] == 12_000_000.0
    assert row["audit_fees_audit"] == 9_000_000.0
    assert row["audit_fees_tax"] == 1_500_000.0
    assert row["auditor_fees_prior"] == 11_000_000.0
    # the dropped fields must NOT come back
    for gone in ("n_technology_directors", "pct_technology_directors", "technology_committee"):
        assert gone not in row, f"{gone} reappeared in the flatten"
    # absent -> null (not a false 0)
    empty = _flatten("X", filing, Def14AExtract(governance=GovernanceProfile(board_size=8)))
    assert empty["auditor_name"] is None and empty["auditor_fees"] is None
    assert empty["audit_fees_audit"] is None

    print("\n=== SANITY: auditor block flattens; technology fields are gone ===")
    print("  auditor_name='Ernst & Young LLP', since=1934, fees=12,000,000 split 9.0M/1.0M/1.5M/0.5M, prior=11,000,000; absent -> null.")
    print("  n_technology_directors / pct_technology_directors / technology_committee are")
    print("  absent from the flatten -- they were an opinion, not an extraction. Validated.")


def _stub_proxy_listing(monkeypatch, mod, listed: dict[str, Any]) -> None:
    """Record each ticker's listing `since` (None = the whole `years_history` window); no proxy is listed."""

    def _list(context, scope, company, years, since):
        del context, company, years
        listed[scope.ticker] = since
        return pd.DataFrame()

    monkeypatch.setattr(mod, "_list_scope_windows", _list)
    monkeypatch.setattr(mod, "load_identity", lambda context: _ROSTER_ONLY)


def test_same_size_universe_swap_lists_the_new_ticker_over_the_full_window(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    last_run = pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
    record_run(context, Tables.def14a_llm, ticker_count=2, rows_added=0, is_full_rescan=True, run_date=last_run, tickers=["AA", "BB"])
    listed: dict[str, Any] = {}
    _stub_proxy_listing(monkeypatch, mod, listed)
    monkeypatch.setattr(mod, "LLMExtractor", _extractor_double([], []))
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *_: pd.DataFrame({"ticker": ["AA", "CC"], "cik": ["1", "2"], "name": ["A", "C"]}))

    mod.fetch_def14a_llm(context, context.config, ["AA", "CC"], model="gpt-5-mini")

    assert listed["CC"] is None, listed
    print("\n=== SANITY: DEF 14A same-size universe swap ===")
    print(f"  AA/BB -> AA/CC: CC listed with since={listed['CC']} (the whole 15y window), not the last run date {last_run.date()}. Validated.")


def test_a_ticker_whose_lineage_changed_since_the_last_run_is_relisted_over_the_full_window(tmp_path, monkeypatch):
    """F-005: a lineage expansion (new predecessor window) after the table's last run relists that ticker over
    the whole `years_history` window; an unchanged ticker keeps the manifest cutoff."""
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    today = pd.Timestamp.today().normalize()
    last_run = today - pd.Timedelta(days=10)
    record_run(context, Tables.def14a_llm, ticker_count=2, rows_added=0, is_full_rescan=True, run_date=last_run, tickers=["AA", "CC"])
    identity = dated_identity(
        [("AA", "0000000001", "cik_window", SENTINEL, None), ("CC", "0000000002", "cik_window", SENTINEL, None)],
        {"AA": "0000000001", "CC": "0000000002"},
        changed_at={"AA": today - pd.Timedelta(days=30), "CC": today - pd.Timedelta(days=2)},
    )
    listed: dict[str, Any] = {}
    _stub_proxy_listing(monkeypatch, mod, listed)
    monkeypatch.setattr(mod, "load_identity", lambda context: identity)
    monkeypatch.setattr(mod, "LLMExtractor", _extractor_double([], []))
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *_: pd.DataFrame({"ticker": ["AA", "CC"], "cik": ["1", "2"], "name": ["A", "C"]}))

    mod.fetch_def14a_llm(context, context.config, ["AA", "CC"], model="gpt-5-mini")

    assert listed == {"AA": last_run - pd.Timedelta(days=1), "CC": None}, listed
    print("\n=== SANITY: DEF 14A relists a ticker whose lineage changed ===")
    print(
        f"  CC's scope changed after the last run ({last_run.date()}) -> since=None (whole window); AA keeps since={listed['AA'].date()}. Validated."
    )


def test_llm_workers_default_to_config_gpt_threads(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    context.config = extract_config(data_extract={"years_history": 15, "manifest_full_rescan_days": 30}, gpt={"threads": 7})
    built: list[int] = []

    class _RecordingExtractor(LLMExtractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self.threads)

        def run_extraction(self, tasks, flatten=None, group_key=None):
            return []

    _stub_proxy_listing(monkeypatch, mod, {})
    monkeypatch.setattr(mod, "LLMExtractor", _RecordingExtractor)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *_: pd.DataFrame({"ticker": ["ZZ"], "cik": ["1"], "name": ["Z"]}))

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    assert built == [7]
    print("\n=== SANITY: DEF 14A LLM workers ===")
    print(f"  no `workers` passed -> the extractor ran {built[0]}-wide, i.e. config.gpt.threads (7 in this test config). Validated.")


def test_llm_proxies_are_listed_per_scope_window_with_the_seam_margin(monkeypatch):
    """AC-008 (DEF 14A LLM): proxies are listed by CIK for each window of the scope, never by roster CIK alone;
    each CIK keeps only what its widened window admits, and a multi-window scope checks subject headers."""
    from src.data_extract.utils.structure.def14a import fetch as mod

    identity = dated_identity(
        [("PSKY", "0000813828", "cik_window", SENTINEL, "2025-08-07"), ("PSKY", "0002041610", "cik_window", "2025-08-07", None)],
        {"PSKY": "0002041610"},
    )
    scope = identity.filing_scope("PSKY")
    listings = {
        "0000813828": ["2024-04-01", "2025-08-20", "2026-04-01"],
        "0002041610": ["2025-07-20", "2026-04-02"],
    }
    listed: list[str] = []

    def fake_list(context, cik, forms, years, company, since=None):
        listed.append(cik)
        return pd.DataFrame([{"cik": cik, "accession_number": f"{cik[-4:]}-{day}", "filing_date": pd.Timestamp(day)} for day in listings[cik]])

    monkeypatch.setattr(mod, "list_filings", fake_list)
    context: Any = SimpleNamespace(log=logging.getLogger("test.def14a.windows"))
    out = mod._list_scope_windows(context, scope, "Paramount", 15, None)

    assert listed == ["0000813828", "0002041610"]
    assert sorted(out["accession_number"]) == ["1610-2025-07-20", "1610-2026-04-02", "3828-2024-04-01", "3828-2025-08-20"]
    assert mod.accepted_subjects(scope) == frozenset({"0000813828", "0002041610"})
    assert mod.accepted_subjects(FilingScope.roster_only("AAPL", "320193")) == frozenset()
    print("\n=== SANITY: DEF 14A LLM per-window listing ===")
    print("  both PSKY CIKs listed; the predecessor's 2026 proxy (past its margin) dropped; margin proxies on both sides kept")
