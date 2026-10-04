"""DEF 14A LLM: every run lists each ticker's whole window and sends only proxies that are neither
stored with evidence nor marked; an evidence-free answer becomes one empty-filing marker (never
written over a stored parent row); a failed read or LLM call writes nothing and is listed again.
Also the flatten of the auditor block. LLM calls are fakes (no spend).
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

from src.data_extract.utils.schemas.def14a_schema import BeneficialOwner as _BeneficialOwner
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract as _Def14AExtract
from src.data_extract.utils.schemas.def14a_schema import DirectorCompensation as _DirectorCompensation
from src.data_extract.utils.schemas.def14a_schema import DirectorInfo as _DirectorInfo
from src.data_extract.utils.schemas.def14a_schema import ExecutiveCompensation as _ExecutiveCompensation
from src.data_extract.utils.schemas.def14a_schema import GovernanceProfile as _GovernanceProfile
from src.data_extract.utils.structure.def14a import flatten as flatten_mod
from src.data_extract.utils.structure.def14a.fetch import _subject_is_accepted
from src.data_extract.utils.structure.def14a.flatten import _flatten, _result_frames
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from tests.data_extract.fake_context import extract_config

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

    parent = frames[Tables.def14a_llm].iloc[0]
    assert flatten_mod._has_extract_evidence(extract) is expected, label
    assert flatten_mod._has_parent_evidence(projected) is expected, label
    assert (parent["def14a_json"] != "_empty") is expected, label
    if not expected:
        assert set(frames) == {Tables.def14a_llm} and list(frames[Tables.def14a_llm].columns) == [
            "ticker",
            "accession_number",
            "as_of",
            "def14a_json",
        ]

    print(f"\n=== SANITY: DEF 14A evidence gate — {label} ===")
    print(f"  parsed={expected}, stored={expected}, {'parent row' if expected else 'empty-filing marker, no child row'}. Validated.")


def _daily_context(tmp_path: Path) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        store=DataStore(create_engine(f"sqlite:///{tmp_path / 'daily.db'}")),
        log=logging.getLogger("test.def14a.daily"),
        paths={"DATA_STORE": tmp_path},
        config_dir="configs",
        config=extract_config(data_extract={"years_history": 15}),
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

    def _list(context, ticker, cik, company, years, since, cutovers):
        del context, ticker, cik, company, years, cutovers
        listed_since.append(since)
        return filing.copy()

    monkeypatch.setattr(mod, "LLMExtractor", extractor)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *_: pd.DataFrame([{"ticker": "ZZ", "cik": "0000000001", "name": "Example Corp"}]))
    monkeypatch.setattr(mod, "load_registrants", lambda config_dir: {})
    monkeypatch.setattr(mod, "_list_across_registrants", _list)
    monkeypatch.setattr(mod, "_payload_for", lambda *_: "=== BOARD OF DIRECTORS ===\nJane Director")
    monkeypatch.setattr(mod, "_finalise_gender", lambda *_: None)


def test_provider_failure_writes_nothing_and_is_retried_next_run(tmp_path, monkeypatch):
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

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    assert listed_since == [None, None]
    assert tasked == [accession, accession]
    parent = context.store.load(Tables.def14a_llm)
    assert parent is not None and list(parent["accession_number"]) == [accession]

    print("\n=== SANITY: provider failure retries on D+1 ===")
    print(f"  day D saved no parent and no marker; D+1 listed the whole window again and queued {accession} again. Validated.")


def test_legacy_empty_parent_is_repaired_on_the_next_run(tmp_path, monkeypatch):
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

    listed_since: list[pd.Timestamp | None] = []
    tasked: list[str] = []
    response = Def14AExtract(
        ceo_name="Jane CEO",
        directors=[DirectorInfo(name="Jane Director", age=55)],
        governance=GovernanceProfile(board_size=1),
    )
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([response], tasked), _listed_filing(accession, yesterday), listed_since)

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    assert listed_since == [None]
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

    listed_since: list[pd.Timestamp | None] = []
    tasked: list[str] = []
    payloads: list[str] = []
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([], tasked), _listed_filing(accession, yesterday), listed_since)
    monkeypatch.setattr(mod, "_payload_for", lambda *_: payloads.append(accession) or "unused")

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    parent = context.store.load(Tables.def14a_llm)
    assert listed_since == [None]
    assert tasked == [] and payloads == [], "semantic completion must filter the re-listed accession before token spend"
    assert parent is not None and len(parent) == 1

    print("\n=== SANITY: valid parent stays idempotent on D+1 ===")
    print(f"  {accession} was re-listed over the whole window but produced zero payloads and zero LLM tasks. Validated.")


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
    monkeypatch.setattr(
        mod, "load_cik_mapping", lambda *_: pd.DataFrame([{"ticker": "PSKY", "cik": "0002041610", "name": "Paramount Skydance Corp"}])
    )
    monkeypatch.setattr(mod, "_list_across_registrants", lambda *_: filing)
    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0001437107"}))
    monkeypatch.setattr(mod, "_payload_for", lambda *args: payload_calls.append(str(args[2]["accession_number"])))

    mod.fetch_def14a_llm(context, context.config, ["PSKY"], model="gpt-5-mini")

    assert payload_calls == [] and llm_tasks == []
    print("\n=== SANITY: wrong-subject filing stops before the LLM ===")
    print("  WBD accession produced zero payload fetches and zero LLM tasks")
    print("  OK: rejection precedes document carving and paid extraction")


def test_gap_fill_lists_full_window_and_skips_present(tmp_path, monkeypatch):
    """Every run lists the FULL window (no `since` cutoff) and the LLM runs ONLY on filings whose
    accession is not already in the table -- so a HOLE in the middle (2023 here) is filled while
    the present years (2022, 2024) are skipped. Uses a SQLite store (no Postgres)."""
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
    monkeypatch.setattr(mod, "_payload_for", lambda context, ticker, f: "=== CARVED ===")
    monkeypatch.setattr(mod, "LLMExtractor", _FakeLLM)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda _c, _t=None: pd.DataFrame({"ticker": ["ZZ"], "cik": ["0000000001"], "name": ["Z"]}))

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

    def _list(context, ticker, cik, company, years, since, cutovers):
        del context, cik, company, years, cutovers
        listed[ticker] = since
        return pd.DataFrame()

    monkeypatch.setattr(mod, "_list_across_registrants", _list)
    monkeypatch.setattr(mod, "load_registrants", lambda config_dir: {})


def test_every_ticker_is_listed_over_the_full_window_on_every_run(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    context.store.save(Tables.def14a_llm, pd.DataFrame([_completed_parent_row("AA", "acc-AA", "2024-04-01")]))
    listed: dict[str, Any] = {}
    _stub_proxy_listing(monkeypatch, mod, listed)
    monkeypatch.setattr(mod, "LLMExtractor", _extractor_double([], []))
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *_: pd.DataFrame({"ticker": ["AA", "CC"], "cik": ["1", "2"], "name": ["A", "C"]}))

    mod.fetch_def14a_llm(context, context.config, ["AA", "CC"], model="gpt-5-mini")
    first = dict(listed)
    mod.fetch_def14a_llm(context, context.config, ["AA", "CC"], model="gpt-5-mini")

    assert first == listed == {"AA": None, "CC": None}, (first, listed)
    print("\n=== SANITY: DEF 14A lists every ticker's whole window ===")
    print(f"  AA (stored) and CC (new): listed with since={listed} on two consecutive runs; no gate skips a ticker. Validated.")


def test_llm_workers_default_to_config_gpt_threads(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    context.config = extract_config(data_extract={"years_history": 15}, gpt={"threads": 7})
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


_EMPTY_ANSWER = Def14AExtract(governance=GovernanceProfile(classified_board=False, dual_class_shares=False))


def _seed_full_width_parent(context: Any) -> None:
    """One unrelated evidenced parent, so the SQLite table carries every column (Postgres adds them on save)."""
    _save_parent(context, "seed", Def14AExtract(ceo_name="Seed CEO"), pd.Timestamp("2020-04-01"))


def test_an_evidence_free_answer_is_marked_and_never_sent_again(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    _seed_full_width_parent(context)
    accession = "evidence-free"
    filed = pd.Timestamp("2025-04-01")
    tasked: list[str] = []
    listed_since: list[pd.Timestamp | None] = []
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([_EMPTY_ANSWER], tasked), _listed_filing(accession, filed), listed_since)

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")
    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    marker = context.store.load(Tables.def14a_llm, markers=True).set_index("accession_number").loc[accession]
    assert tasked == [accession], tasked
    assert marker["def14a_json"] == "_empty" and pd.isna(marker["ceo_name_proxy"]) and pd.isna(marker["board_size"])
    assert context.store.load(Tables.def14a_llm)["accession_number"].tolist() == ["seed"]
    assert not context.store.exists(Tables.def14a_directors)
    print("\n=== SANITY: DEF 14A evidence-free answer (D17) ===")
    print(f"  run 1: 1 LLM call -> marker def14a_json='_empty', no child row, hidden by load; run 2: {len(tasked) - 1} call.")


def test_a_failed_read_writes_nothing_and_is_sent_on_the_next_run(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    accession = "read-failed"
    tasked: list[str] = []
    answer = Def14AExtract(ceo_name="Jane CEO", governance=GovernanceProfile(board_size=8))
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([answer], tasked), _listed_filing(accession, pd.Timestamp("2025-04-01")), [])
    reads = iter([None, "=== BOARD OF DIRECTORS ===\nJane Director"])
    monkeypatch.setattr(mod, "_payload_for", lambda *_: next(reads))

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")
    exists_after_failure = context.store.exists(Tables.def14a_llm)
    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    assert not exists_after_failure and tasked == [accession]
    assert context.store.load(Tables.def14a_llm)["ceo_name_proxy"].tolist() == ["Jane CEO"]
    print("\n=== SANITY: DEF 14A failed read ===")
    print("  run 1: the proxy could not be read -> no row, no marker, no LLM call; run 2: read, sent once, parent stored.")


def test_full_sends_a_marked_proxy_again(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    _seed_full_width_parent(context)
    accession = "marked-then-full"
    tasked: list[str] = []
    answer = Def14AExtract(ceo_name="Jane CEO")
    _install_daily_fetch_doubles(
        monkeypatch, _extractor_double([_EMPTY_ANSWER, answer], tasked), _listed_filing(accession, pd.Timestamp("2025-04-01")), []
    )

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")
    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini", full=True)

    stored = context.store.load(Tables.def14a_llm, markers=True).set_index("accession_number").loc[accession]
    assert tasked == [accession, accession]
    assert stored["def14a_json"] != "_empty" and stored["ceo_name_proxy"] == "Jane CEO"
    print("\n=== SANITY: DEF 14A --full ===")
    print("  a marked proxy is sent again under --full; the evidenced answer replaces the marker.")


def test_an_evidence_free_answer_never_overwrites_a_stored_parent(tmp_path, monkeypatch):
    from src.data_extract.utils.structure.def14a import fetch as mod

    context = _daily_context(tmp_path)
    accession = "legacy-evidence-free"
    filed = pd.Timestamp("2025-04-01")
    _save_parent(context, accession, Def14AExtract(company_name="Legacy Co", governance=GovernanceProfile(classified_board=False)), filed)
    before = context.store.load(Tables.def14a_llm, markers=True).iloc[0]
    tasked: list[str] = []
    _install_daily_fetch_doubles(monkeypatch, _extractor_double([_EMPTY_ANSWER, _EMPTY_ANSWER], tasked), _listed_filing(accession, filed), [])

    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")
    mod.fetch_def14a_llm(context, context.config, ["ZZ"], model="gpt-5-mini")

    after = context.store.load(Tables.def14a_llm, markers=True)
    assert len(after) == 1 and after.iloc[0]["def14a_json"] == before["def14a_json"] != "_empty"
    assert after.iloc[0]["company_name"] == "Legacy Co"
    assert tasked == [accession, accession]
    print("\n=== SANITY: never mark over data (def14a_llm) ===")
    print("  a stored evidence-free parent keeps its row: the evidence-free answer's marker is dropped, so the proxy is sent again on each run.")
