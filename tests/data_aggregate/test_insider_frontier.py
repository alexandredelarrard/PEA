"""The insider completeness frontier: trusted only from the EDGAR run's manifest entry, and only
when that run covered exactly the analysis universe."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

LOG = logging.getLogger(__name__)
UNIVERSE = ["AAA", "BBB"]


def _context(tmp_path: Path) -> Any:
    return SimpleNamespace(paths={"DATA_STORE": tmp_path}, config=extract_config())


def _frontier(context: Any, universe: list[str]) -> pd.Timestamp | None:
    return schedule_complete_through(cast(Context, context), LOG, Tables.insider_transactions, expected_tickers=universe)


def test_a_complete_edgar_run_over_the_universe_is_the_frontier(tmp_path):
    context = _context(tmp_path)
    record_run(context, Tables.insider_transactions, 2, 7, run_date="2026-10-02", coverage_complete=True, tickers=["BBB", "AAA"])

    got = _frontier(context, UNIVERSE)

    assert got is not None and got == pd.Timestamp("2026-10-02")
    print(f"SANITY: a complete EDGAR run over exactly {UNIVERSE} on 2026-10-02 gives complete_through = {got.date()}.")


@pytest.mark.parametrize(
    ("label", "entry"),
    [
        ("no entry", None),
        ("no completeness proof", {"coverage_complete": False, "tickers": UNIVERSE}),
        ("subset of the universe", {"coverage_complete": True, "tickers": ["AAA"]}),
        ("same size, wrong member", {"coverage_complete": True, "tickers": ["AAA", "CCC"]}),
        ("superset of the universe", {"coverage_complete": True, "tickers": ["AAA", "BBB", "CCC"]}),
    ],
)
def test_a_missing_or_mismatched_run_leaves_the_frontier_unknown(tmp_path, label: str, entry: dict | None):
    context = _context(tmp_path)
    if entry is not None:
        record_run(context, Tables.insider_transactions, len(entry["tickers"]), 7, run_date="2026-10-02", **entry)

    got = _frontier(context, UNIVERSE)

    assert got is None, label
    print(f"SANITY: {label} -> complete_through is unknown (None), so absence is never emitted as zero.")
