"""Daily section attempts through the public fetcher, real store and stub provider."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.structure.def14a import fetch
from src.data_extract.utils.structure.def14a.retry import SECTIONS
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore, _create_canonical
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
    _create_canonical(
        context.store.engine,
        [Tables.def14a_llm, Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership, Tables.def14a_directors],
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


@pytest.mark.parametrize("candidate", [True, False])
def test_complete_pay_source_replay_is_bounded_to_already_selected_filings(daily, monkeypatch, candidate):
    context, calls, answers, _ = daily
    payloads = {**PAYLOADS, "executive_pay": "Name\tYear\tSalary ($)\tTotal ($)\nJane Example\t2025\t100\t100"}
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: payloads)
    answers.update(_complete_answers())
    answers["GovernanceExtract"].update(ceo_since_year=2018, board_size=1, ceo_pay_ratio=1, say_on_pay_support_pct=99)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    for section in SECTIONS:
        frame = context.store.load(section.table, markers=True)
        frame["retry"] = 2
        if section.key == "executive_pay":
            frame["total"] = 999
        if section.key == "audit":
            frame["ceo_total_comp"] = 999
            if candidate:
                frame["audit_fees_tax"] = None
        context.store.save(section.table, frame)
    count = len(calls)
    monkeypatch.setattr(fetch, "_fetch_filing_html", lambda *_: pytest.fail("Exhausted source replay must not read SEC or call a provider"))
    for _ in range(2):
        fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    saved = context.store.load(Tables.def14a_executive_comp)
    assert saved.iloc[0].total == (100 if candidate else 999)
    assert saved.retry.eq(2).all() and len(calls) == count
    print(
        "SANITY: complete pay at retry2 reuses source only inside an already-selected filing; wholly complete filings are not scanned and no calls are added."
    )


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


@pytest.mark.parametrize(("cell", "repair"), [("Yes", True), ("No", False), ("Yes", None), ("?", None)])
def test_disclosed_independence_gap_repairs_only_governance_once(
    daily: tuple[SimpleNamespace, list[tuple[str, str, str]], dict[str, Any], pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    cell: str,
    repair: bool | None,
) -> None:
    context, calls, answers, _ = daily
    context = cast(Context, context)
    answers.update(_complete_answers())
    answers["GovernanceExtract"].update(ceo_since_year=2018, ceo_pay_ratio=100, say_on_pay_support_pct=95)
    answers["GovernanceExtract"]["directors"][0]["is_independent"] = None
    payloads = {**PAYLOADS, "governance": f"Name\tAge\tDirector Since\tIndependent\nJane Example\t54\t2018\t{cell}"}
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: payloads)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert len(calls) == 5 and _attempts(context)[Tables.def14a_directors.name] == [1]

    # A source-positive missing boolean gets one owned repair; ambiguity stays uncalled.
    answers["GovernanceExtract"]["directors"][0]["is_independent"] = repair
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    should_retry = cell in {"Yes", "No"}
    assert [name for name, _, _ in calls[5:]] == (["GovernanceExtract"] if should_retry else [])
    assert _attempts(context)[Tables.def14a_directors.name] == ([2] if should_retry else [1])
    assert all(value == [1] for key, value in _attempts(context).items() if key != Tables.def14a_directors.name)
    saved = context.store.load(Tables.def14a_directors)
    assert saved is not None
    assert pd.isna(saved.iloc[0].is_independent) if repair is None else saved.iloc[0].is_independent == repair

    # Retained False, exhausted missing output and ambiguous source never cause more calls.
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"], full=True)
    assert len(calls) == 5 + int(should_retry)
    print(f"SANITY: independence {cell}/{repair} repairs only its board attempt once; False is retained and missing retry=2 is excluded.")


@pytest.mark.parametrize("interrupt_core_save", [False, True])
def test_delayed_ceo_identity_refreshes_saved_pay_and_ownership_without_new_calls(
    daily: tuple[SimpleNamespace, list[tuple[str, str, str]], dict[str, Any], pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    interrupt_core_save: bool,
) -> None:
    context, calls, answers, _ = daily
    answers.update(_complete_answers())
    core = dict(answers["GovernanceExtract"], ceo_name="Bob Brown", ceo_age=50, directors=[dict(name="Bob Brown", age=50)])
    components = dict(bonus=0, stock_awards=0, option_awards=0, non_equity_incentive=0, pension_change=0, other_compensation=0)
    answers["ExecutivePayExtract"]["compensation"] = [
        dict(**components, name="Alice Adams", title="CEO", fiscal_year=2025, salary=100, total=100),
        dict(**{**components, "stock_awards": 40, "option_awards": 60}, name="Bob Brown", title="CEO", fiscal_year=2025, salary=100, total=200),
    ]
    answers["OwnershipExtract"].update(
        holders=[
            dict(holder_name="Alice Adams", holder_type="director_officer", shares=100, percent_of_class=10, percent_of_voting_power=15),
            dict(holder_name="Bob Brown", holder_type="director_officer", shares=200, percent_of_class=20, percent_of_voting_power=25),
        ],
        insider_shares=300,
        insider_ownership_pct=30,
        insider_voting_pct=40,
    )
    answers["GovernanceExtract"] = RuntimeError("core temporarily unavailable")
    payloads = {
        **PAYLOADS,
        "governance": "Name\tAge\tDirector Since\nBob Brown\t50\t2018\nBob Brown is Chief Executive Officer.",
        "executive_pay": "Name\tYear\tSalary\tStock Awards\tOption Awards\tTotal\nAlice Adams\t2025\t100\t0\t0\t100\nBob Brown\t2025\t100\t40\t60\t200",
        "ownership": "Name\tNumber of Shares\tPercent of Class\tVoting Power\nAlice Adams\t100\t10%\t15%\nBob Brown\t200\t20%\t25%",
    }
    monkeypatch.setattr(fetch, "prepare_def14a_payloads", lambda *_: payloads)
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    before = context.store.load(Tables.def14a_llm).iloc[0]
    assert len(calls) == 5 and before.ceo_total_comp == 100 and pd.isna(before.ceo_name_proxy)
    answers["GovernanceExtract"] = core
    reads: list[Table] = []
    original_load, original_save = context.store.load, fetch.save_section

    def tracked_load(table: Table, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        reads.append(table)
        return original_load(table, *args, **kwargs)

    def interrupted_save(context: Any, work: Any, processed: Any) -> None:
        if Tables.def14a_directors in processed.frames:
            raise OSError("core save interrupted")
        original_save(context, work, processed)

    monkeypatch.setattr(context.store, "load", tracked_load)
    if interrupt_core_save:
        monkeypatch.setattr(fetch, "save_section", interrupted_save)
        with pytest.raises(OSError, match="core save interrupted"):
            fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
        assert len(calls) == 6
        reads.clear()
        monkeypatch.setattr(fetch, "save_section", original_save)
        monkeypatch.setattr(fetch, "_read_proxy", lambda *_: pytest.fail("the paid core answer must replay without source"))
    fetch.fetch_def14a_llm(context, context.config, ["ZZ"])
    assert reads == [section.table for section in SECTIONS] + [Tables.def14a_llm]
    after = original_load(Tables.def14a_llm).iloc[0]
    assert after.ceo_name_proxy == "Bob Brown"
    assert (after.ceo_total_comp, after.ceo_equity_pay_pct, after.ceo_ownership_pct, after.ceo_voting_pct) == (200, 0.5, 0.2, 0.25)
    assert (after.n_neos, after.total_neo_comp, after.auditor_fees) == (2, 300, before.auditor_fees)
    assert len(calls) == 6 and calls[-1][0] == "GovernanceExtract"
    assert _attempts(context) == {table.name: [2 if table is Tables.def14a_directors else 1] for table in (section.table for section in SECTIONS)}
    raw = json.loads(after.def14a_json)["_def14a_split"]["responses"]
    assert raw["executive_pay"][0]["response"]["compensation"][0]["name"] == "Alice Adams"
    print(
        "SANITY: delayed core identity refreshes saved CEO pay/ownership immediately, including free restart replay; no extra reads/calls/counters."
    )
