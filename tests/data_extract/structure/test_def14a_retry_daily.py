"""Daily section attempts through the public fetcher, real store and stub provider."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.structure.def14a import fetch
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from tests.data_extract.fake_context import extract_config

PROXY = """<html><body><h1>Election of Directors</h1>
<p>Jane Example, age 54, Director since 2018. Jane Example is Chief Executive Officer.</p>
</body></html>"""


@pytest.fixture
def daily(tmp_path, monkeypatch):
    context = SimpleNamespace(
        store=DataStore(create_engine(f"sqlite:///{tmp_path / 'retry.db'}")),
        log=logging.getLogger("test.def14a.retry"),
        config_dir="configs",
        config=extract_config(data_extract={"years_history": 31}),
    )
    filed = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    filing = pd.DataFrame(
        [
            dict(
                ticker="ZZ",
                cik="0000000001",
                company_name="Example",
                accession_number="retry-proxy",
                filing_date=filed,
                period_of_report="2025-12-31",
                form="DEF 14A",
                doc_url="https://example.invalid/proxy",
            )
        ]
    )
    calls = []
    answers = {}

    class Provider:
        name, model, structured = "stub", "stub", True

        def parse(self, schema, system, user):
            calls.append((schema.__name__, system, user))
            answer = answers.get(schema.__name__, {"percent_unit": "percent"})
            if isinstance(answer, Exception):
                raise answer
            return schema.model_validate(answer), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(fetch, "load_cik_mapping", lambda *_: pd.DataFrame([dict(ticker="ZZ", name="Example")]))
    monkeypatch.setattr(fetch, "load_identity", lambda *_: SimpleNamespace(filing_scope=lambda _: FilingScope.roster_only("ZZ", "1")))
    monkeypatch.setattr(fetch, "_list_scope_windows", lambda *_: filing.copy())
    monkeypatch.setattr(fetch, "_fetch_filing_html", lambda *_: PROXY)
    monkeypatch.setattr(fetch, "_finalise_gender", lambda *_: None)
    monkeypatch.setattr(LLMExtractor, "initialize_client", lambda *_args, **_kwargs: Provider())
    return context, calls, answers, filing


def test_provider_failure_reserves_first_attempt(daily):
    context, calls, answers, _ = daily
    answers.update(Def14AExtract=RuntimeError("provider unavailable"), GovernanceExtract=RuntimeError("provider unavailable"))
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert "retry" in context.store.columns(Tables.def14a_directors)
    saved = context.store.load(Tables.def14a_directors, markers=True)
    assert saved is not None and saved.retry.tolist() == [1]
    assert context.store.load(Tables.def14a_directors, optional=True) is None
    print("SANITY: provider failure keeps attempt 1 durably and its empty child stays hidden.")


PAYLOADS = {
    "governance": "=== DIRECTOR NOMINEES ===\nName\tAge\tDirector Since\nJane Example\t54\t2018\nJane Example is Chief Executive Officer.",
    "executive_pay": "=== SUMMARY COMPENSATION TABLE ===\nName\tYear\tSalary\tTotal\nJane Example\t2025\t100\t100",
    "director_pay": "=== DIRECTOR COMPENSATION TABLE ===\nName\tFees Earned\tTotal\nAlice Director\t100\t100",
    "audit": "=== AUDIT FEE TABLE ===\nFees\t2025\t2024\nAudit fees\t100\t90\nAudit-related fees\t20\t10\nTax fees\t10\t5\nOther fees\t0\t0\nTotal\t130\t105\nKPMG LLP",
    "ownership": "=== INSIDER OWNERSHIP ===\nName\tNumber of Shares\tPercent of Class\nJane Example\t1000\t10%\nAll directors and executive officers as a group 1,000 10%",
}


def _complete_answers():
    money = dict(currency="USD", amount_scale="units")
    pay = dict(stock_awards=0, option_awards=0, non_equity_incentive=0, pension_change=0, other_compensation=0, total=100)
    return dict(
        GovernanceExtract=dict(
            percent_unit="percent",
            ceo_name="Jane Example",
            ceo_age=54,
            directors=[dict(name="Jane Example", age=54, director_since_year=2018, is_independent=False)],
        ),
        ExecutivePayExtract=dict(**money, compensation=[dict(**pay, name="Jane Example", title="CEO", fiscal_year=2025, salary=100, bonus=0)]),
        DirectorPayExtract=dict(**money, compensation=[dict(**pay, name="Alice Director", fiscal_year=2025, fees_earned=100)]),
        AuditExtract=dict(
            **money,
            auditor_name="KPMG LLP",
            auditor_fees=130,
            auditor_fees_prior=105,
            audit_fees_audit=100,
            audit_fees_audit_related=20,
            audit_fees_tax=10,
            audit_fees_other=0,
        ),
        OwnershipExtract=dict(
            percent_unit="percent",
            shares_scale="units",
            holders=[dict(holder_name="Jane Example", holder_type="director_officer", shares=1000, percent_of_class=10, percent_of_voting_power=10)],
            insider_shares=1000,
            insider_ownership_pct=10,
            insider_voting_pct=10,
        ),
    )


def _attempts(context):
    return {
        table.name: context.store.load(table, markers=True).retry.dropna().astype(int).unique().tolist()
        for table in (Tables.def14a_llm, Tables.def14a_directors, Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership)
    }


def test_five_matched_calls_then_only_source_present_gap_is_retried(daily, monkeypatch):
    context, calls, answers, _ = daily
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    answers.update(_complete_answers())
    answers["AuditExtract"]["audit_fees_tax"] = None
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert {name for name, _, _ in calls} == set(_complete_answers())
    assert len(calls) == 5
    for name, system, user in calls:
        assert name in user or name in system
    assert all(value == [1] for value in _attempts(context).values())

    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert [name for name, _, _ in calls[5:]] == ["AuditExtract"]
    assert _attempts(context)[Tables.def14a_llm.name] == [2]
    assert all(value == [1] for key, value in _attempts(context).items() if key != Tables.def14a_llm.name)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"], full=True)
    assert len(calls) == 6
    parent = context.store.load(Tables.def14a_llm)
    assert pd.isna(parent.iloc[0].audit_fees_tax) and parent.iloc[0].audit_fees_other == 0
    print("SANITY: five matched initial calls; only missing disclosed tax is retried; full preserves exhausted 2 and real zero.")


def test_two_failed_attempts_are_excluded_in_normal_and_full_runs(daily, monkeypatch):
    context, calls, answers, _ = daily
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    answers.update({name: RuntimeError("provider unavailable") for name in _complete_answers()})
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 5 and all(value == [1] for value in _attempts(context).values())
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 10 and all(value == [2] for value in _attempts(context).values())
    monkeypatch.setattr(fetch, "_read_proxy", lambda *_: pytest.fail("exhausted sections must not fetch source"))
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"], full=True)
    assert len(calls) == 10
    print("SANITY: all failed domains advance 1 -> 2 exactly once; later daily/full runs read no source and spend nothing.")


def test_complete_sparse_sections_do_not_recall_optional_absences(daily, monkeypatch):
    context, calls, answers, _ = daily
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    answers.update(_complete_answers())
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 5
    assert all(value == [1] for value in _attempts(context).values())
    parent = context.store.load(Tables.def14a_llm)
    assert parent.iloc[0].ceo_age == 54
    assert pd.isna(parent.iloc[0].ceo_is_founder) and pd.isna(parent.iloc[0].auditor_since_year)
    print("SANITY: sparse optional founder/start-year/ratio facts stay null without daily recall; CEO age remains available.")


def test_source_failure_does_not_consume_an_attempt(daily, monkeypatch):
    context, calls, _, _ = daily

    def unavailable(*_):
        raise OSError("source unavailable")

    monkeypatch.setattr(fetch, "_fetch_filing_html", unavailable)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert calls == []
    assert all(
        context.store.row_count(table) == 0
        for table in (Tables.def14a_llm, Tables.def14a_directors, Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership)
    )
    print("SANITY: unreadable source leaves all five counters unattempted and makes no provider call.")


def test_paid_answer_survives_postprocessing_failure_and_replays_free(daily, monkeypatch):
    context, calls, answers, _ = daily
    answers.update(_complete_answers())
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: {"governance": PAYLOADS["governance"]})
    original = fetch.postprocess_section

    def broken(*_):
        raise RuntimeError("postprocessing interrupted")

    monkeypatch.setattr(fetch, "postprocess_section", broken)
    with pytest.raises(RuntimeError, match="postprocessing interrupted"):
        fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    raw = context.store.load(Tables.def14a_llm, markers=True).iloc[0].def14a_json
    assert len(json.loads(raw)["_def14a_split"]["responses"]["governance"]) == 1
    assert len(calls) == 1
    monkeypatch.setattr(fetch, "postprocess_section", original)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 1
    directors = context.store.load(Tables.def14a_directors)
    assert directors.name.tolist() == ["Jane Example"] and directors.retry.tolist() == [1]
    print("SANITY: raw paid answer is durable before postprocessing and a restart restores facts with zero additional calls.")


def test_legacy_paid_json_restores_children_before_any_paid_repair(daily, monkeypatch):
    from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
    from src.data_extract.utils.structure.def14a.flatten import _flatten

    context, calls, _, filing = daily
    legacy = Def14AExtract.model_validate(
        dict(
            ceo_name="Jane Example",
            ceo_age=54,
            directors=[dict(name="Jane Example", age=54)],
            compensation=[dict(name="Jane Example", title="CEO", fiscal_year=2025, salary_usd=100)],
            director_compensation=[dict(name="Alice Director", fees_earned_usd=100)],
            ownership_holders=[dict(holder_name="Jane Example", holder_type="director_officer", shares=1000)],
            governance=dict(auditor_name="KPMG LLP", auditor_fees_usd=130),
        )
    )
    row = _flatten("ZZ", filing.iloc[0], legacy)
    original = row["def14a_json"]
    context.store.save(Tables.def14a_llm, pd.DataFrame([row]))
    monkeypatch.setattr(fetch, "_read_proxy", lambda *_: None)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert calls == []
    assert all(value == [1] for value in _attempts(context).values())
    assert context.store.load(Tables.def14a_executive_comp).name.tolist() == ["Jane Example"]
    raw = context.store.load(Tables.def14a_llm).iloc[0].def14a_json
    assert json.loads(raw)["_def14a_split"]["legacy_json"] == original
    print("SANITY: legacy all-domain attempt inherits 1; paid JSON restores all child tables for free and retains original bytes.")


def test_free_replay_preserves_two_on_retained_and_restored_rows(daily, monkeypatch):
    context, calls, answers, _ = daily
    answers.update(_complete_answers())
    answers["GovernanceExtract"]["directors"].append(dict(name="Second Director", age=60, director_since_year=2010))
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    roster = context.store.load(Tables.def14a_directors)
    context.store.save(Tables.def14a_directors, roster.assign(retry=2))
    context.store.delete(Tables.def14a_directors, where={"ticker": "ZZ", "name": "Second Director"})
    monkeypatch.setattr(fetch, "_read_proxy", lambda *_: None)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    restored = context.store.load(Tables.def14a_directors)
    assert set(restored.name) == {"Jane Example", "Second Director"}
    assert restored.retry.tolist() == [2, 2]
    assert len(calls) == 5
    print("SANITY: orphan partial roster replays all already-paid people; retained and restored rows both stay at attempt 2.")


def test_headers_bounds_and_absence_do_not_justify_a_retry(daily, monkeypatch):
    context, calls, _, _ = daily
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    empty_source = {
        "governance": "Name\tAge\tDirector Since\nJane Example\t-\t-",
        "executive_pay": "Name\tYear\tSalary\tTotal\nJane Example\t2025\t-\t-",
        "director_pay": "Name\tFees Earned\tTotal\nAlice Director\t-\t-",
        "audit": "Fees\t2025\t2024\nAudit fees\t-\t-",
        "ownership": "Name\tNumber of Shares\tPercent of Class\nJane Example\t-\t<1%",
    }
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: empty_source)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"], full=True)
    assert len(calls) == 5
    assert all(value == [1] for value in _attempts(context).values())
    from src.utils.filer_tables import def14a_evidence_mask

    assert not def14a_evidence_mask(context.store.load(Tables.def14a_llm, markers=True)).any()
    print("SANITY: headers, dashes and less-than bounds prove no exact missing value; empty answers create no governance observation.")


def test_interior_missing_filing_is_found_without_recalling_completed_filings(daily, monkeypatch):
    context, calls, answers, filing = daily
    answers.update(_complete_answers())
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    base = filing.iloc[0].copy()
    listed = pd.DataFrame(
        [
            {**base.to_dict(), "accession_number": "older", "filing_date": base.filing_date - pd.Timedelta(days=2)},
            {**base.to_dict(), "accession_number": "newer"},
        ]
    )
    monkeypatch.setattr(fetch, "_list_scope_windows", lambda *_: listed.copy())
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 10
    listed.loc[2] = {**base.to_dict(), "accession_number": "interior", "filing_date": base.filing_date - pd.Timedelta(days=1)}
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 15
    parent = context.store.load(Tables.def14a_llm)
    assert set(parent.accession_number) == {"older", "interior", "newer"}
    assert parent.retry.tolist() == [1, 1, 1]
    print("SANITY: the whole rolling filing list finds an interior hole; the two completed proxies incur zero extra calls.")


def test_split_parent_does_not_complete_unattempted_sibling_domains(daily, monkeypatch):
    from src.data_extract.utils.structure.def14a.postprocess import append_answer
    from src.data_extract.utils.structure.def14a.retry import SECTIONS

    context, calls, answers, filing = daily
    answers.update(_complete_answers())
    governance = next(section for section in SECTIONS if section.key == "governance")
    raw = append_answer(None, governance, governance.schema.model_validate(answers["GovernanceExtract"]), PAYLOADS["governance"])
    context.store.prepare_def14a_retries()
    context.store.save(
        Tables.def14a_llm, pd.DataFrame([dict(ticker="ZZ", accession_number="retry-proxy", as_of=filing.iloc[0].filing_date, def14a_json=raw)])
    )
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: PAYLOADS)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 4 and "GovernanceExtract" not in {name for name, _, _ in calls}
    assert all(value == [1] for value in _attempts(context).values())
    print("SANITY: a board answer replays free while all four unattempted siblings receive attempt 1.")


def test_retry_only_parent_is_hidden_from_cube_and_prior_ceo_roles(daily):
    from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
    from src.data_extract.utils.structure.votes.roles import _role_map, _role_source

    context, _, _, filing = daily
    context.store.prepare_def14a_retries()
    now = filing.iloc[0].filing_date
    context.store.save(
        Tables.def14a_llm,
        pd.DataFrame(
            [
                dict(ticker="ZZ", accession_number="usable", as_of=now - pd.Timedelta(days=2), ceo_name_proxy="Jane Example", retry=1),
                dict(ticker="ZZ", accession_number="reservation", as_of=now, ceo_name_proxy=None, retry=1),
            ]
        ),
    )
    step = object.__new__(StepCubeGovernance)
    step._store, step._log = context.store, context.log
    usable = step._load_def14a()
    assert usable is not None and usable.accession_number.tolist() == ["usable"]
    roles, _ = _role_map(_role_source(context, "ZZ"), now + pd.Timedelta(days=1))
    assert list(roles.values()) == ["ceo"]
    context.store.save(
        Tables.def14a_llm, pd.DataFrame([dict(ticker="ZZ", accession_number="real-audit", as_of=now + pd.Timedelta(days=1), audit_fees_other=0)])
    )
    assert set(step._load_def14a().accession_number) == {"usable", "real-audit"}
    roles, _ = _role_map(_role_source(context, "ZZ"), now + pd.Timedelta(days=2))
    assert roles == {}, "a genuine later audit observation keeps the existing latest-proxy semantics"
    print("SANITY: counter-only parents cannot hide a prior CEO or enter the cube; a genuine audit zero remains a real later observation.")


@pytest.mark.parametrize("missing", ["ceo_name", "ceo_since_year"])
def test_disclosed_ceo_identity_or_start_year_gap_is_retried(daily, monkeypatch, missing):
    context, calls, answers, _ = daily
    answers.update(_complete_answers())
    answers["GovernanceExtract"].update(ceo_since_year=2018, ceo_pay_ratio=100, say_on_pay_support_pct=95)
    answers["GovernanceExtract"][missing] = None
    payloads = {
        **PAYLOADS,
        "governance": PAYLOADS["governance"]
        + "\nJane Example is Chief Executive Officer since 2018.\nCEO pay ratio is 100 to 1; say-on-pay support was 95%.",
    }
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: payloads)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert [name for name, _, _ in calls[5:]] == ["GovernanceExtract"]
    assert _attempts(context)[Tables.def14a_directors.name] == [2]
    assert _attempts(context)[Tables.def14a_llm.name] == [1]
    print(f"SANITY: source-disclosed missing {missing} retries only board/CEO and leaves audit at attempt 1.")
