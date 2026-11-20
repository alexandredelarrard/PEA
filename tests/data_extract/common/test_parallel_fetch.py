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


@pytest.mark.parametrize("max_workers", [1, 2])
def test_run_per_ticker_turns_a_source_error_into_none_and_a_warning(caplog: pytest.LogCaptureFixture, max_workers: int) -> None:
    scope = pd.DataFrame({"cik_label": ["Berkshire", "Muhlenkamp", "Akre"], "cik": ["0001067983", "0000915191", "0001112520"]})
    caplog.set_level(logging.WARNING, logger=LOGGER)

    def worker(label: str, cik: str) -> str:
        if label == "Muhlenkamp":
            raise RuntimeError("listing page malformed")
        return cik

    results = run_per_ticker(scope, worker, "13F manager books", max_workers=max_workers, key_cols=("cik_label", "cik"))

    assert results == ["0001067983", None, "0001112520"]  # scope row order, failure as None
    assert _warnings(caplog) == ["13F manager books: Muhlenkamp failed (listing page malformed)"]

    print("\n=== SANITY CHECK: run_per_ticker source-error guard ===")
    print(f"  results={results}; warnings={_warnings(caplog)}")
    print(f"  -> With {max_workers} worker(s), a source RuntimeError returns None; the others complete in scope row order.")


def test_run_per_ticker_serial_stops_before_later_tickers() -> None:
    visited: list[str] = []

    def worker(ticker: str) -> str:
        visited.append(ticker)
        if ticker == "B":
            raise TypeError("broken vote contract")
        return ticker

    with pytest.raises(TypeError, match="broken vote contract"):
        run_per_ticker(pd.DataFrame({"ticker": ["A", "B", "C"]}), worker, "serial-proof", max_workers=1, key_cols=("ticker",))

    assert visited == ["A", "B"]
    print("\n=== SANITY CHECK: serial ticker stop order ===")
    print(f"  visited={visited}; the TypeError escapes before ticker C starts.")


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
