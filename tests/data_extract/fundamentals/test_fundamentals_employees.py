"""Offline known-truth checks for SEC employee extraction and safe replay."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

import src.data_extract.cli as cli_mod
import src.data_extract.utils.fundamentals.fundamentals_employees as mod
from src.data_extract.utils.common.edgar_driver import IncompleteEdgarRunError
from src.data_extract.utils.common.registrant import Combine, combine_for
from src.gpt_extract.utils.schemas_gpt import LlmResult


class Filing:
    def __init__(self, accession: str, filed: str, text: str, *, form: str = "10-K", report: str | None = "2020-12-31") -> None:
        self.accession_number = accession
        self.filing_date = pd.Timestamp(filed).date()
        self.period_of_report = report
        self.form = form
        self.cik = "0000000001"
        self.body = text

    def html(self) -> str:
        return f"<p>{self.body}</p>"

    def text(self) -> str:
        return ""


def answer(
    count: int | None,
    quote: str | None,
    *,
    status: str = "found",
    measurement_period: str | None = "year end",
) -> mod.EmployeeAnswer:
    return mod.EmployeeAnswer(
        status=status,
        count=count,
        quote=quote,
        measurement_period=measurement_period,
        qualifier=None,
        reason="fixture",
    )


def build(monkeypatch, filings: list[Filing], answers: dict[str, mod.EmployeeAnswer]) -> mod.EmployeeTickerResult:
    def listing(ticker, forms, **kwargs):
        assert ticker == "AAA"
        assert "10-K405" in forms
        assert kwargs["identity"] is identity
        assert kwargs["roster_cik"] == "0000000001"
        assert kwargs["symbol_tenure"]["issuer_cik"].tolist() == ["0000000001"]
        return filings

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            self.tasks = []

        def submit(self, task):
            self.tasks.append(task)

        def run(self):
            return [LlmResult(seq=task.seq, task=task, parsed=answers[task.meta["filing"].accession_number]) for task in self.tasks]

    identity = SimpleNamespace(owns=lambda ticker, cik: ticker == "AAA" and str(cik).zfill(10) == "0000000001")
    monkeypatch.setattr(mod, "resolve_registrant_filings", listing)
    monkeypatch.setattr(mod, "LLMExtractor", FakeLLM)
    config = OmegaConf.create(
        {
            "gpt": {
                "default_api": "open_ai",
                "llm_model": {"open_ai": "gpt-6-sol", "open_ai_cheap": "gpt-6-luna"},
                "max_chars": {"employees": 60000},
            }
        }
    )
    context = SimpleNamespace(config=config, log=SimpleNamespace(info=lambda *args: None))
    return mod.build_ticker_employees(
        context,
        "AAA",
        "0000000001",
        since=None,
        done_accessions=frozenset(),
        registrants={},
        identity=identity,
        symbol_tenure=pd.DataFrame([{"symbol": "AAA", "issuer_cik": "0000000001"}]),
    )


def test_source_guard_rejects_bounds_and_unsupported_claims():
    exact = "As of December 31, 2023, we employed approximately 74,042 employees."
    bound = "We had over 300,000 employees."
    split = "We had 2,476 full-time employees and 37 part-time employees."
    assert mod.supported_employee_count(answer(74042, exact), exact) == 74042
    assert mod.supported_employee_count(answer(300000, bound), bound) is None
    assert mod.supported_employee_count(answer(300000, "300,000 employees"), bound) is None
    assert mod.supported_employee_count(answer(2476, split), split) == 2513
    assert mod.supported_employee_count(answer(99999, exact), exact) is None
    assert mod.supported_employee_count(answer(74042, "fabricated quote"), exact) is None
    assert mod.supported_employee_count(answer(None, None, status="image_only"), exact) is None
    print(
        "\nSANITY: exact text supports a count; full and shortened bounds, unsupported quote and image-only evidence abstain; an explicit split sums."
    )


def test_source_guard_recovers_punctuation_and_anchored_table_quote():
    apple_source = "Employees As of September 28, 2019 , the Company had approximately 137,000 full-time equivalent employees."
    apple_quote = "As of September 28, 2019, the Company had approximately 137,000 full-time equivalent employees."
    heading = (
        "The following tables set forth information about the Company's employees as of December 31, 2020. Number of Employees by Contract and Region"
    )
    total = "Total 17,163 20,412 1,513 39,088"
    table_quote = f"{heading} ... {total}"
    table_source = f"{heading} North America 8,196 10,227 270 18,693 EMEA 4,586 4,847 564 9,997 {total}"
    assert mod.supported_employee_count(answer(137000, apple_quote), apple_source) == 137000
    assert mod.supported_employee_count(answer(39088, table_quote), table_source) == 39088
    intro = heading.split(" Number of Employees", 1)[0]
    intro_quote = f"{intro} ... {total}"
    assert mod.supported_employee_count(answer(39088, intro_quote), table_source) == 39088
    assert mod.supported_employee_count(answer(39089, table_quote), table_source) is None
    assert mod.supported_employee_count(answer(39088, table_quote), total) is None
    assert mod.supported_employee_count(answer(39088, table_quote), f"{heading} {'other data ' * 300} {total}") is None
    assert mod.supported_employee_count(answer(39088, table_quote), f"{heading} [... filing gap ...] {total}") is None
    assert mod.supported_employee_count(answer(39088, f"{heading} [... filing gap ...] {total}"), f"{heading} [... filing gap ...] {total}") is None
    assert mod.supported_employee_count(answer(39088, table_quote), f"{heading} As of December 31, 2019, employees {total}") is None
    assert (
        mod.supported_employee_count(answer(39088, intro_quote), f"{intro} Number of Employees by Type Number of Employees by Region {total}") is None
    )
    assert (
        mod.supported_employee_count(answer(44043, "ADM employed approximately 44,000 people."), "ADM employed approximately 44,000 people.") is None
    )
    print(
        "\nSANITY: punctuation noise and a nearby real table heading/total recover supported counts; missing, distant, cross-gap, conflicting or imprecise claims abstain."
    )


def test_legacy_annual_form_uses_dated_registrant_scope():
    assert combine_for(mod.HEADCOUNT_FORMS) is Combine.SPLIT
    print("\nSANITY: 10-K405 joins the dated annual registrant scope used with entity lineage and symbol tenure.")


def test_legacy_form_table_total_measurement_period_and_image_null(monkeypatch):
    filings = [
        Filing("aapl-2001", "2001-12-21", "Apple and its subsidiaries worldwide had 9,603 employees.", form="10-K405", report="2001-09-29"),
        Filing("csco-2014", "2014-09-09", "United States 36,725 Rest of world 37,317 Total 74,042 employees.", report="2014-07-26"),
        Filing("peg-2023", "2024-02-26", "Workforce totals are shown in the following image.", report="2023-12-31"),
    ]
    answers = {
        "aapl-2001": answer(9603, "Apple and its subsidiaries worldwide had 9,603 employees.", measurement_period="September 29, 2001"),
        "csco-2014": answer(74042, "United States 36,725 Rest of world 37,317 Total 74,042 employees.", measurement_period="July 26, 2014"),
        "peg-2023": answer(None, None, status="image_only", measurement_period=None),
    }
    result = build(monkeypatch, filings, answers)
    assert result.frame["employees"].tolist() == [9603.0, 74042.0]
    assert result.frame["as_of"].tolist() == [pd.Timestamp("2001-12-21"), pd.Timestamp("2014-09-09")]
    assert {row["accession_number"]: row["status"] for row in result.outcomes} == {
        "aapl-2001": "saved",
        "csco-2014": "saved",
        "peg-2023": "no_headcount",
    }
    assert result.outcomes[0]["measurement_period"] == "September 29, 2001"
    assert result.outcomes[0]["model"] == "gpt-6-luna"
    assert result.unavailable_dates == frozenset({pd.Timestamp("2024-02-26")})
    print("\nSANITY: legacy annual form and table total use SEC filing dates; image-only PEG remains null with provenance.")


def test_same_day_amendment_uses_later_supported_count(monkeypatch):
    filings = [
        Filing("original", "2024-03-01", "We had 40,000 employees.", report="2023-12-31"),
        Filing("amendment", "2024-03-01", "We had 41,000 employees.", form="10-K/A", report="2023-12-31"),
    ]
    result = build(
        monkeypatch,
        filings,
        {
            "original": answer(40000, "We had 40,000 employees."),
            "amendment": answer(41000, "We had 41,000 employees."),
        },
    )
    assert result.frame["employees"].tolist() == [41000.0]
    assert [row["status"] for row in result.outcomes] == ["superseded", "saved"]
    print("\nSANITY: a same-date amendment replaces the original value at the one-row filing-date grain.")


def test_missing_report_period_keeps_filing_date_and_bad_cik_fails_closed(monkeypatch):
    filing = Filing("no-period", "2024-03-01", "We had 40,000 employees.", report=None)
    result = build(monkeypatch, [filing], {"no-period": answer(40000, "We had 40,000 employees.")})
    assert result.frame["as_of"].tolist() == [pd.Timestamp("2024-03-01")]
    assert result.outcomes[0]["report_date"] is None
    filing.cik = "0000000002"
    with pytest.raises(ValueError, match="outside the issuer lineage"):
        build(monkeypatch, [filing], {"no-period": answer(40000, "We had 40,000 employees.")})
    print("\nSANITY: optional report metadata does not block filing-date storage; reused alias CIKs fail closed.")


def test_full_replay_rechecks_saved_accession_and_clears_null(monkeypatch):
    deleted = []
    saved = []
    outcomes = []
    runs = []
    store = SimpleNamespace(save=lambda table, frame: saved.append(frame), delete=lambda table, where: deleted.append(where))
    config = SimpleNamespace(
        data_extract=SimpleNamespace(manifest_full_rescan_days=30, fundamentals_workers=1),
        gpt=SimpleNamespace(llm_model=SimpleNamespace(open_ai_cheap="gpt-6-luna")),
    )
    context = SimpleNamespace(
        store=store,
        config=config,
        config_dir="configs",
        ensure_edgar_identity=lambda: None,
        log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args: pd.DataFrame([{"ticker": "AAA", "cik": "0000000001"}]))
    monkeypatch.setattr(mod, "load_identity", lambda *args: SimpleNamespace(ciks_by_symbol={"AAA": {"0000000001"}}))
    monkeypatch.setattr(mod, "load_registrants", lambda *args: {})
    monkeypatch.setattr(mod, "identity_scope_fingerprint", lambda *args: "scope")
    monkeypatch.setattr(
        mod,
        "get_entry",
        lambda *args: {
            "coverage_complete": True,
            "filing_outcomes": [{"ticker": "AAA", "accession_number": "old", "status": "saved"}],
        },
    )
    monkeypatch.setattr(mod, "record_filing_outcomes", lambda *args: outcomes.extend(args[-1]))
    monkeypatch.setattr(mod, "record_run", lambda *args, **kwargs: runs.append(kwargs))
    monkeypatch.setattr(
        mod,
        "run_per_ticker",
        lambda mapping, worker, **kwargs: [worker("AAA", "0000000001")],
    )
    seen = {}

    def fake_build(*args, **kwargs):
        seen.update(kwargs)
        return mod.EmployeeTickerResult(
            pd.DataFrame(columns=["ticker", "as_of", "employees"]),
            [{"ticker": "AAA", "accession_number": "old", "status": "no_headcount"}],
            frozenset({pd.Timestamp("2024-02-26")}),
        )

    monkeypatch.setattr(mod, "build_ticker_employees", fake_build)
    mod.fetch_fundamentals_employees(context, ["AAA"], 15, full=True)
    assert seen["done_accessions"] == frozenset()
    assert deleted == [{"ticker": "AAA", "as_of": pd.Timestamp("2024-02-26")}]
    assert saved == []
    assert outcomes[0]["status"] == "no_headcount"
    assert len(runs) == 1
    deleted.clear()
    runs.clear()
    mod.fetch_fundamentals_employees(context, ["AAA"], 15)
    assert seen["done_accessions"] == frozenset()
    assert deleted == [{"ticker": "AAA", "as_of": pd.Timestamp("2024-02-26")}]
    assert len(runs) == 1
    print("\nSANITY: full replay and routine legacy migration revisit old decisions and clear only their now-null filing-date row.")


def test_new_null_amendment_preserves_skipped_supported_original(monkeypatch):
    deleted = []
    context = SimpleNamespace(
        store=SimpleNamespace(save=lambda *args: None, delete=lambda table, where: deleted.append(where)),
        config=SimpleNamespace(
            data_extract=SimpleNamespace(manifest_full_rescan_days=30, fundamentals_workers=1),
            gpt=SimpleNamespace(llm_model=SimpleNamespace(open_ai_cheap="gpt-6-luna")),
        ),
        config_dir="configs",
        ensure_edgar_identity=lambda: None,
        log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args: pd.DataFrame([{"ticker": "AAA", "cik": "0000000001"}]))
    monkeypatch.setattr(mod, "load_identity", lambda *args: SimpleNamespace(ciks_by_symbol={}))
    monkeypatch.setattr(mod, "load_registrants", lambda *args: {})
    monkeypatch.setattr(mod, "identity_scope_fingerprint", lambda *args: "scope")
    monkeypatch.setattr(mod, "changed_scope_tickers", lambda *args: set())
    monkeypatch.setattr(mod, "manifest_window", lambda *args, **kwargs: (pd.Timestamp("2024-03-01"), False))
    monkeypatch.setattr(
        mod,
        "get_entry",
        lambda *args: {
            "coverage_complete": True,
            "filing_outcomes": [
                {
                    "ticker": "AAA",
                    "accession_number": "original",
                    "status": "saved",
                    "model": "gpt-6-luna",
                    "filing_date": "2024-03-01",
                }
            ],
        },
    )
    monkeypatch.setattr(mod, "run_per_ticker", lambda mapping, worker, **kwargs: [worker("AAA", "0000000001")])
    monkeypatch.setattr(mod, "record_filing_outcomes", lambda *args: None)
    monkeypatch.setattr(mod, "record_run", lambda *args, **kwargs: None)
    seen = {}

    def fake_build(*args, **kwargs):
        seen.update(kwargs)
        return mod.EmployeeTickerResult(
            pd.DataFrame(columns=["ticker", "as_of", "employees"]),
            [{"ticker": "AAA", "accession_number": "amendment", "status": "no_headcount"}],
            frozenset({pd.Timestamp("2024-03-01")}),
        )

    monkeypatch.setattr(mod, "build_ticker_employees", fake_build)
    mod.fetch_fundamentals_employees(context, ["AAA"], 15)
    assert seen["done_accessions"] == frozenset({"original"})
    assert deleted == []
    print("\nSANITY: a new image-only amendment cannot erase the saved same-day original skipped by the routine frontier.")


def test_ambiguous_result_cannot_advance_complete_frontier(monkeypatch):
    # The guard may reject a paid answer; it must remain retryable and cannot certify coverage.
    saved_outcomes = []
    context = SimpleNamespace(
        store=SimpleNamespace(save=lambda *args: None, delete=lambda *args, **kwargs: None),
        config=SimpleNamespace(
            data_extract=SimpleNamespace(manifest_full_rescan_days=30, fundamentals_workers=1),
            gpt=SimpleNamespace(llm_model=SimpleNamespace(open_ai_cheap="gpt-6-luna")),
        ),
        config_dir="configs",
        ensure_edgar_identity=lambda: None,
        log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args: pd.DataFrame([{"ticker": "AAA", "cik": "0000000001"}]))
    monkeypatch.setattr(mod, "load_identity", lambda *args: SimpleNamespace(ciks_by_symbol={}))
    monkeypatch.setattr(mod, "load_registrants", lambda *args: {})
    monkeypatch.setattr(mod, "identity_scope_fingerprint", lambda *args: "scope")
    monkeypatch.setattr(mod, "get_entry", lambda *args: {})
    monkeypatch.setattr(mod, "run_per_ticker", lambda mapping, worker, **kwargs: [worker("AAA", "0000000001")])
    monkeypatch.setattr(
        mod,
        "build_ticker_employees",
        lambda *args, **kwargs: mod.EmployeeTickerResult(
            pd.DataFrame(columns=["ticker", "as_of", "employees"]),
            [{"ticker": "AAA", "accession_number": "uncertain", "status": "ambiguous", "model": "gpt-6-luna", "filing_date": "2001-12-21"}],
            frozenset(),
        ),
    )
    monkeypatch.setattr(mod, "record_filing_outcomes", lambda *args: saved_outcomes.extend(args[-1]))
    monkeypatch.setattr(mod, "record_run", lambda *args, **kwargs: pytest.fail("ambiguous result advanced the frontier"))
    with pytest.raises(IncompleteEdgarRunError, match="ambiguous"):
        mod.fetch_fundamentals_employees(context, ["AAA"], 15)
    assert saved_outcomes[0]["status"] == "ambiguous"
    monkeypatch.setattr(
        mod,
        "get_entry",
        lambda *args: {
            "coverage_complete": True,
            "last_run_date": "2025-01-01",
            "filing_outcomes": list(saved_outcomes),
        },
    )
    monkeypatch.setattr(mod, "manifest_window", lambda *args, **kwargs: pytest.fail("pending filing skipped by recent frontier"))

    def retry_build(*args, **kwargs):
        assert kwargs["since"] <= pd.Timestamp("2001-12-21")
        assert "uncertain" not in kwargs["done_accessions"]
        return mod.EmployeeTickerResult(
            pd.DataFrame(columns=["ticker", "as_of", "employees"]),
            list(saved_outcomes),
            frozenset(),
        )

    monkeypatch.setattr(mod, "build_ticker_employees", retry_build)
    with pytest.raises(IncompleteEdgarRunError, match="ambiguous"):
        mod.fetch_fundamentals_employees(context, ["AAA"], 15)
    print("\nSANITY: a historical ambiguous accession remains retryable on the next run despite an old recent frontier.")


def test_missing_roster_cik_cannot_certify_coverage(monkeypatch):
    context = SimpleNamespace(ensure_edgar_identity=lambda: None)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args: pd.DataFrame(columns=["ticker", "cik"]))
    with pytest.raises(ValueError, match="no roster CIK for AAA"):
        mod.fetch_fundamentals_employees(context, ["AAA"], 15)
    print("\nSANITY: an unresolved requested ticker cannot be reported as fully covered.")


def test_cli_still_dispatches_explicit_full_replay(monkeypatch):
    calls = []
    context = SimpleNamespace()
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=31))
    monkeypatch.setattr(cli_mod, "_ctx", lambda path: (config, context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    monkeypatch.setattr(
        cli_mod,
        "fetch_fundamentals_employees",
        lambda ctx, tickers, years_history, full=False: calls.append((tickers, years_history, full)),
    )
    result = CliRunner().invoke(cli_mod.cli, ["fundamentals-employees", "-t", "AAA", "--full"])
    assert result.exit_code == 0, result.output
    assert calls == [(["AAA"], 31, True)]
    print("\nSANITY: the existing CLI runs a ticker-scoped full employee replay when explicitly requested.")
