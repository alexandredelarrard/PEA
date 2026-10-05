"""A filing whose `period_of_report` raises must not abort a DEF 14A or filing-text walk.

edgartools implements `Filing.period_of_report` as a property that can raise `TypeError` from
inside itself on old submissions. `TypeError` is in `PROGRAMMING_ERRORS`, which `run_per_ticker`
re-raises, so each builder must read the value through the guarded filing stamp and store None.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp
from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH, FILING_TEXT_MIN_CHARS
from src.data_store.schema import Tables

FIXTURES = Path(__file__).parent / "fixtures"


class _RaisingPeriodFiling:
    """A filing stand-in whose `period_of_report` raises exactly as edgartools' property does."""

    def __init__(self, *, form: str, accession: str, filing_date: str, cik: str) -> None:
        self.form = form
        self.accession_number = accession
        self.filing_date = pd.Timestamp(filing_date).date()
        self.cik = cik
        self.company = "TEST CO"

    @property
    def period_of_report(self) -> Any:
        broken: Any = None
        _, _, period = broken  # what edgartools' attachments.get_filing_dates() unpack does
        return period


def _stamp(filing: object, cik: str) -> FilingStamp:
    return FilingStamp.of(filing, cik)


def test_def14a_parse_stores_none_when_period_of_report_raises() -> None:
    facts = pd.read_parquet(FIXTURES / "ecd_ba_2025.parquet")
    filing = _RaisingPeriodFiling(form="DEF 14A", accession="0000012927-25-000001", filing_date="2025-03-07", cik="12927")
    filing.xbrl = lambda: SimpleNamespace(facts=SimpleNamespace(to_dataframe=lambda: facts))  # type: ignore[attr-defined]
    df = DEF14A_EDGAR_FETCH.parse("BA", "0000012927", _stamp(filing, "0000012927"), EdgarScope(None, {}))[Tables.def14a_edgar]

    assert len(df) == 1
    row = df.iloc[0]
    assert row["period_of_report"] is None
    assert row["cik"] == "0000012927"
    assert row["filing_date"] == pd.Timestamp("2025-03-07")
    print("\n=== SANITY: DEF 14A period_of_report guard ===")
    print(f"  1 ECD row kept, period_of_report={row['period_of_report']!r}, cik={row['cik']}: a raising property no longer aborts the walk.")


def test_filing_text_parse_stores_none_when_period_of_report_raises() -> None:
    body = "Revenue increased because demand grew. " * (FILING_TEXT_MIN_CHARS // 20)
    filing = _RaisingPeriodFiling(form="10-K", accession="0000320193-24-000010", filing_date="2024-05-03", cik="320193")
    filing.obj = lambda: SimpleNamespace(risk_factors=None, management_discussion=body)  # type: ignore[attr-defined]
    df = FILING_TEXT_FETCH.parse("AAPL", "0000320193", _stamp(filing, "0000320193"), EdgarScope(None, {}))[Tables.filing_risk_text]

    assert len(df) == 1
    row = df.iloc[0]
    assert row["period_of_report"] is None
    assert row["cik"] == "0000320193"
    assert row["filed"] == pd.Timestamp("2024-05-03")
    print("\n=== SANITY: filing-text period_of_report guard ===")
    print(f"  1 MD&A row kept, period_of_report={row['period_of_report']!r}: a raising property no longer aborts the walk.")
