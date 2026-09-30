"""Offline contract tests for the standalone SEC employee-headcount extractor."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import FakeStore

import src.data_extract.cli as cli_mod
import src.data_extract.utils.fundamentals.fundamentals_employees as mod
from src.data_extract.utils.common.edgar_driver import IncompleteEdgarRunError
from src.data_extract.utils.common.run_manifest import get_entry, record_filing_outcomes
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config


class _Filing:
    def __init__(
        self,
        accession: str,
        filed: str,
        count: int | None,
        *,
        report: str = "2020-12-31",
        form: str = "10-K",
        error: Exception | None = None,
    ) -> None:
        self.accession_number = accession
        self.cik = "0000000001"
        self.form = form
        self.filing_date = pd.Timestamp(filed).date()
        self.period_of_report = report
        self.count = count
        self.error = error

    def html(self) -> str:
        if self.error:
            raise self.error
        return "<p>No workforce disclosure.</p>" if self.count is None else f"<p>We had approximately {self.count:,} employees.</p>"

    def text(self) -> str:
        return ""


def _build(monkeypatch, filings: list[_Filing], history: list[int], pending: list[dict] | None = None):
    monkeypatch.setattr(mod, "resolve_registrant_filings", lambda *args, **kwargs: filings)
    return mod.build_ticker_employees(
        "AAA",
        "0000000001",
        since=None,
        done_accessions=frozenset(),
        registrants={},
        history=history,
        pending_outcomes=pending or [],
    )


def test_two_consistent_outliers_establish_a_new_regime(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=mod.__name__)
    result = _build(
        monkeypatch,
        [_Filing("acc-1", "2021-02-01", 10_000), _Filing("acc-2", "2022-02-01", 11_000, report="2021-12-31")],
        [1_000, 1_100],
    )
    assert result.frame["employees"].tolist() == [10_000.0, 11_000.0]
    assert [row["status"] for row in result.outcomes] == ["saved", "saved"]
    assert result.outcomes[0]["cik"] == "0000000001"
    assert result.outcomes[0]["report_date"] == "2020-12-31"
    assert "reason=new_regime" in caplog.text
    assert "old_anchor=1100" in caplog.text
    assert "first=acc-1/10000" in caplog.text and "second=acc-2/11000" in caplog.text
    assert "first_ratio=" in caplog.text and "second_ratio=" in caplog.text
    print("\nSANITY: two mutually consistent observations establish and save the new regime.")


def test_pending_candidate_resumes_and_an_isolated_artifact_does_not_save(monkeypatch):
    first = _build(monkeypatch, [_Filing("acc-1", "2021-02-01", 10_000)], [1_000])
    assert first.frame.empty
    assert first.outcomes[0]["status"] == "pending_regime"
    assert first.outcomes[0]["candidate"] == 10_000

    resumed = _build(
        monkeypatch,
        [_Filing("acc-2", "2022-02-01", 11_000, report="2021-12-31")],
        [1_000],
        first.outcomes,
    )
    assert resumed.frame["employees"].tolist() == [10_000.0, 11_000.0]
    assert {row["accession_number"]: row["status"] for row in resumed.outcomes} == {"acc-1": "saved", "acc-2": "saved"}
    print("\nSANITY: a cached pending candidate is resumable and remains absent until corroborated.")


def test_body_retrieval_failure_is_not_a_committable_outcome(monkeypatch):
    filing = _Filing("acc-broken", "2021-02-01", None, error=RuntimeError("body unavailable"))
    monkeypatch.setattr(mod, "resolve_registrant_filings", lambda *args, **kwargs: [filing])
    with pytest.raises(RuntimeError, match="body unavailable"):
        mod.build_ticker_employees(
            "AAA",
            "0000000001",
            since=None,
            done_accessions=frozenset(),
            registrants={},
            history=[],
            pending_outcomes=[],
        )
    print("\nSANITY: body failure raises before any terminal/pending outcome can be committed.")


def test_same_day_amendment_cannot_self_confirm_a_new_regime(monkeypatch):
    result = _build(
        monkeypatch,
        [
            _Filing("acc-original", "2021-02-01", 10_000),
            _Filing("acc-amended", "2021-02-01", 11_000, form="10-K/A"),
        ],
        [1_000],
    )
    assert result.frame.empty
    assert {row["accession_number"]: row["status"] for row in result.outcomes} == {
        "acc-original": "rejected_outlier",
        "acc-amended": "pending_regime",
    }
    print("\nSANITY: same-day 10-K/10-K/A candidates normalize before continuity and cannot corroborate each other.")


def test_unmatched_pending_candidates_remain_pending(monkeypatch):
    result = _build(
        monkeypatch,
        [_Filing("acc-1", "2021-02-01", 10_000), _Filing("acc-2", "2022-02-01", 1_000_000)],
        [1_000],
    )
    assert result.frame.empty
    assert [row["status"] for row in result.outcomes] == ["pending_regime", "pending_regime"]
    print("\nSANITY: unrelated outliers remain pending for later annual evidence; neither is prematurely rejected.")


def test_continuity_anchor_uses_only_the_last_three_trusted_counts(monkeypatch):
    result = _build(monkeypatch, [_Filing("acc-current", "2021-02-01", 1_100)], [1] * 10 + [900, 1_000, 1_050])
    assert result.frame["employees"].tolist() == [1_100.0]
    assert result.outcomes[0]["status"] == "saved"
    print("\nSANITY: recent trusted observations, not an obsolete all-history median, anchor continuity.")


def test_incomplete_fetch_persists_successful_outcomes_but_not_frontier(monkeypatch, tmp_path):
    store = FakeStore({Tables.fundamentals_employees: pd.DataFrame(columns=["ticker", "as_of", "employees"])})
    log = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    context = SimpleNamespace(
        store=store,
        log=log,
        paths={"DATA_STORE": tmp_path},
        config_dir=tmp_path,
        config=extract_config(data_extract={"manifest_full_rescan_days": 30, "fundamentals_workers": 1}),
        ensure_edgar_identity=lambda: None,
    )
    monkeypatch.setattr(
        mod,
        "load_cik_mapping",
        lambda *args, **kwargs: pd.DataFrame({"ticker": ["AAA", "BBB"], "cik": ["0000000001", "0000000002"]}),
    )
    monkeypatch.setattr(mod, "load_registrants", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mod,
        "run_per_ticker",
        lambda mapping, worker, **kwargs: [worker(ticker, cik) for ticker, cik in mapping[["ticker", "cik"]].itertuples(index=False, name=None)],
    )
    saved_outcome = {
        "ticker": "AAA",
        "accession_number": "acc-ok",
        "cik": "0000000001",
        "form": "10-K",
        "filing_date": "2021-02-01",
        "report_date": "2020-12-31",
        "ordering": 0,
        "status": "saved",
    }

    def _build(ticker, *args, **kwargs):
        if ticker == "BBB":
            raise RuntimeError("body unavailable")
        return mod.EmployeeTickerResult(
            pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2021-02-01"), "employees": 1_000.0}]),
            [saved_outcome],
        )

    monkeypatch.setattr(mod, "build_ticker_employees", _build)
    with pytest.raises(IncompleteEdgarRunError, match="no run manifest was advanced"):
        mod.fetch_fundamentals_employees(context, ["AAA", "BBB"], 15)

    assert store.t[Tables.fundamentals_employees.name]["ticker"].tolist() == ["AAA"]
    entry = get_entry(context, Tables.fundamentals_employees)
    assert entry["filing_outcomes"] == [saved_outcome]
    assert "last_run_date" not in entry and "coverage_complete" not in entry
    print("\nSANITY: successful ticker progress is durable while an incomplete batch cannot advance coverage.")


def test_standalone_cli_dispatches_employee_fetch(monkeypatch):
    calls: list[tuple[list[str], bool, int]] = []
    context = SimpleNamespace()
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "_ctx", lambda path: (config, context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    monkeypatch.setattr(
        cli_mod,
        "fetch_fundamentals_employees",
        lambda ctx, tickers, years_history, full=False: calls.append((tickers, full, years_history)),
    )

    result = CliRunner().invoke(cli_mod.cli, ["fundamentals-employees", "-t", "AAA", "--full"])
    assert result.exit_code == 0, result.output
    assert calls == [(["AAA"], True, 15)]
    print("\nSANITY: fundamentals-employees owns a dedicated CLI dispatch and full flag.")


def test_full_reconsiders_parser_outcomes_but_not_saved_accessions(monkeypatch, tmp_path):
    store = FakeStore({Tables.fundamentals_employees: pd.DataFrame([{"ticker": "AAA", "as_of": pd.Timestamp("2020-02-01"), "employees": 1_000.0}])})
    context = SimpleNamespace(
        store=store,
        log=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        paths={"DATA_STORE": tmp_path},
        config_dir=tmp_path,
        config=extract_config(data_extract={"manifest_full_rescan_days": 30, "fundamentals_workers": 1}),
        ensure_edgar_identity=lambda: None,
    )
    outcomes = [
        {"ticker": "AAA", "accession_number": "acc-saved", "status": "saved"},
        {"ticker": "AAA", "accession_number": "acc-empty", "status": "no_headcount"},
        {"ticker": "AAA", "accession_number": "acc-rejected", "status": "rejected_outlier"},
        {"ticker": "AAA", "accession_number": "acc-pending", "status": "pending_regime", "candidate": 10_000},
    ]
    record_filing_outcomes(context, Tables.fundamentals_employees, outcomes)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda *args, **kwargs: pd.DataFrame({"ticker": ["AAA"], "cik": ["0000000001"]}))
    monkeypatch.setattr(mod, "load_registrants", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mod,
        "run_per_ticker",
        lambda mapping, worker, **kwargs: [worker("AAA", "0000000001")],
    )
    seen: dict[str, object] = {}

    def _build(*args, **kwargs):
        seen.update(kwargs)
        return mod.EmployeeTickerResult(pd.DataFrame(columns=["ticker", "as_of", "employees"]), [])

    monkeypatch.setattr(mod, "build_ticker_employees", _build)
    mod.fetch_fundamentals_employees(context, ["AAA"], 15, full=True)

    assert seen["done_accessions"] == frozenset({"acc-saved"})
    assert seen["pending_outcomes"] == []
    assert seen["history"] == [1_000]
    print("\nSANITY: full mode retries parser outcomes, skips saved accessions, and retains stored continuity history.")


def test_combined_rebuild_does_not_delete_employee_history(monkeypatch):
    deleted: list[object] = []
    store = SimpleNamespace(delete=lambda table, where: deleted.append(table))
    context = SimpleNamespace(store=store, log=SimpleNamespace(warning=lambda *args, **kwargs: None))
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=15))
    monkeypatch.setattr(cli_mod, "_ctx", lambda path: (config, context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])
    monkeypatch.setattr(cli_mod, "fetch_fundamentals_sec", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "build_fundamentals_history", lambda *args, **kwargs: None)

    result = CliRunner().invoke(cli_mod.cli, ["fundamentals", "-t", "AAA", "--rebuild"])
    assert result.exit_code == 0, result.output
    assert Tables.fundamentals_employees not in deleted
    assert set(deleted) == {
        Tables.fundamentals_facts,
        Tables.fundamentals_history_sec,
        Tables.fundamentals_reason_codes,
    }
    print("\nSANITY: the combined XBRL rebuild cannot delete independently owned employee history.")
