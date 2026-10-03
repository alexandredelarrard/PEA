"""`run_per_ticker` guards each entity: a source failure is logged and becomes None, a
programming error aborts the pool."""

from __future__ import annotations

import types

import pandas as pd
import pytest

from src.data_extract.utils.common.parallel_fetch import run_per_ticker


def _log() -> types.SimpleNamespace:
    warnings: list[str] = []
    return types.SimpleNamespace(warning=lambda msg, *args: warnings.append(msg % args), warnings=warnings)


def test_run_per_ticker_turns_a_source_error_into_none_and_a_warning():
    scope = pd.DataFrame({"cik_label": ["Berkshire", "Muhlenkamp", "Akre"], "cik": ["0001067983", "0000915191", "0001112520"]})
    log = _log()

    def worker(label: str, cik: str) -> str:
        if label == "Muhlenkamp":
            raise ValueError("listing page malformed")
        return cik

    results = run_per_ticker(scope, worker, "13F manager books", log=log, max_workers=2, key_cols=("cik_label", "cik"))

    assert results == ["0001067983", None, "0001112520"]  # scope row order, failure as None
    assert log.warnings == ["13F manager books: Muhlenkamp failed (listing page malformed)"]

    print("\n=== SANITY CHECK: run_per_ticker source-error guard ===")
    print(f"  results={results}; warnings={log.warnings}")
    print("  -> A ValueError from one entity is logged under its label and returns None; the others complete.")


def test_run_per_ticker_propagates_a_programming_error():
    scope = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "cik": ["0000320193", "0000789019"]})
    log = _log()

    def worker(ticker: str, cik: str) -> int:
        if ticker == "AAPL":
            raise KeyError("accession_number")
        return 1

    with pytest.raises(KeyError, match="accession_number"):
        run_per_ticker(scope, worker, "test", log=log)

    assert log.warnings == []

    print("\n=== SANITY CHECK: run_per_ticker programming-error abort ===")
    print(f"  KeyError escaped the pool; warnings logged: {len(log.warnings)}.")
    print("  -> A repo defect fails the run instead of being logged once per ticker.")
