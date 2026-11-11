"""`run_per_ticker` guards each entity: a source failure is logged and becomes None, a
programming error aborts the pool."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data_extract.utils.edgar_extract.parallel_fetch import run_per_ticker

LOGGER = "src.data_extract.utils.edgar_extract.parallel_fetch"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


def test_run_per_ticker_turns_a_source_error_into_none_and_a_warning(caplog: pytest.LogCaptureFixture):
    scope = pd.DataFrame({"cik_label": ["Berkshire", "Muhlenkamp", "Akre"], "cik": ["0001067983", "0000915191", "0001112520"]})
    caplog.set_level(logging.WARNING, logger=LOGGER)

    def worker(label: str, cik: str) -> str:
        if label == "Muhlenkamp":
            raise ValueError("listing page malformed")
        return cik

    results = run_per_ticker(scope, worker, "13F manager books", max_workers=2, key_cols=("cik_label", "cik"))

    assert results == ["0001067983", None, "0001112520"]  # scope row order, failure as None
    assert _warnings(caplog) == ["13F manager books: Muhlenkamp failed (listing page malformed)"]

    print("\n=== SANITY CHECK: run_per_ticker source-error guard ===")
    print(f"  results={results}; warnings={_warnings(caplog)}")
    print("  -> A ValueError from one entity is logged (module logger) under its label and returns None; the others complete.")


def test_run_per_ticker_propagates_a_programming_error(caplog: pytest.LogCaptureFixture):
    scope = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "cik": ["0000320193", "0000789019"]})
    caplog.set_level(logging.WARNING, logger=LOGGER)

    def worker(ticker: str, cik: str) -> int:
        if ticker == "AAPL":
            raise KeyError("accession_number")
        return 1

    with pytest.raises(KeyError, match="accession_number"):
        run_per_ticker(scope, worker, "test")

    assert _warnings(caplog) == []

    print("\n=== SANITY CHECK: run_per_ticker programming-error abort ===")
    print(f"  KeyError escaped the pool; warnings logged: {len(_warnings(caplog))}.")
    print("  -> A repo defect fails the run instead of being logged once per ticker.")
