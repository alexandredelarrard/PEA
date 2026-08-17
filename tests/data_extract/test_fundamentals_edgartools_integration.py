"""
Live-network integration test for the edgartools-based fundamentals pipeline,
across a fixed, reproducible 15-ticker list covering: large/small issuers,
financial/non-financial, non-calendar and 52/53-week fiscal years, a known
amendment, extension concepts, capex/current-liabilities presence, and the
confirmed REIT gap case (MAA).

Gated behind SEC_USER_AGENT (required for any live SEC call) AND an explicit
opt-in (RUN_LIVE_SEC_INTEGRATION=1), since a 15-ticker x ~15y live pull is slow
and must not run on a cold `pytest tests/ -v -s`. The report-building function
is shared with the standalone script `scripts/run_fundamentals_integration_report.py`
so the test and a manual rollout run never drift apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import pytest

from src.context import Context
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import (
    _configure_identity, build_ticker_facts_edgar,
)
from src.data_extract.utils.fundamentals.fundamentals_derive import (
    derive_fundamentals_history,
)
from src.data_extract.utils.fundamentals.fundamentals_validation import (
    reconcile_fundamentals_facts,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FLOW_TAGS, STOCK_TAGS,
)

# 15 tickers, each with a specific justification (see final report for detail):
#   JPM  - large bank; known 10-Q/A (2012 "London Whale" restatement)
#   PGR  - insurance underwriting KPIs
#   O    - REIT (general reference, complements MAA's specific gap case)
#   XOM  - large energy, extension concepts (no OilAndGasProperty* tags)
#   CSCO - non-calendar + 52/53-week fiscal year (FYE last Saturday in July)
#   NEE  - utilities, known revenue-tag era transition
#   LLY  - pharma, extension concept (Liabilities never tagged as one element)
#   AEP  - utilities, known PP&E-component reconstruction case
#   KR   - non-calendar + 52/53-week fiscal year (FYE Saturday nearest Jan 31), LIFO
#   VLO  - energy/refining, sector-specific revenue tag
#   PM   - large non-financial, calendar FYE, deliberate "plain" control
#   MAA  - the confirmed REIT capex/currentLiabilities gap case
#   ORCL - non-calendar fiscal year (FYE May 31), known share-count scale-error case
#   ERIE - smaller issuer, extension concept (large NCI-inclusive equity structure)
#   ROP  - known point-in-time edge case (as_of before fiscal_end)
INTEGRATION_TICKERS = [
    "JPM", "PGR", "O", "XOM", "CSCO", "NEE", "LLY", "AEP", "KR", "VLO",
    "PM", "MAA", "ORCL", "ERIE", "ROP",
]

_CORE_FIELDS = ["totalRevenue", "netIncome", "operatingIncome", "operatingCashFlow",
               "capex", "totalAssets", "currentLiabilities", "totalLiabilities",
               "stockholdersEquity"]
_ALL_FIELDS = {**FLOW_TAGS, **EXTRA_FLOW_TAGS, **STOCK_TAGS, **EXTRA_STOCK_TAGS}


@dataclass
class TickerIntegrationReport:
    ticker: str
    filings_discovered: dict[str, int] = field(default_factory=dict)
    filings_processed: int = 0
    filings_skipped: dict[str, int] = field(default_factory=dict)
    missing_normalized_fields: list[str] = field(default_factory=list)
    q4_reconciliation: dict = field(default_factory=dict)
    amendments_detected: list[dict] = field(default_factory=list)
    rows_written: dict[str, int] = field(default_factory=dict)
    exceptions: list[str] = field(default_factory=list)


def build_ticker_integration_report(context: Context, ticker: str,
                                    forms=("10-K", "10-K/A", "10-Q", "10-Q/A"),
                                    years: int = 15) -> TickerIntegrationReport:
    """Runs the real pipeline (retrieval -> fundamentals_facts -> derive ->
    fundamentals_history) for ONE ticker against live SEC data and summarizes the
    result. Shared by the pytest test below and the standalone rollout script.

    Scoped to `years` (default 15, matching production's `fundamentals_years_history`
    default) -- without a `since` cutoff, `build_ticker_facts_edgar` walks a ticker's
    ENTIRE available filing history; for a decades-old filer (e.g. JPM, filing since
    the 1990s) that is 3-4x the intended window and turns this "15-ticker" validation
    into a multi-hour run for no extra coverage of the behavior actually shipped."""
    report = TickerIntegrationReport(ticker=ticker)
    since = pd.Timestamp.today() - pd.DateOffset(years=years)
    try:
        facts = build_ticker_facts_edgar(ticker, forms=forms, since=since)
    except Exception as e:                          # noqa: BLE001
        report.exceptions.append(f"retrieval: {e}")
        return report

    if facts.empty:
        report.exceptions.append("retrieval produced zero rows")
        return report

    report.filings_discovered = facts.groupby("form")["accession_number"].nunique().to_dict()
    report.filings_processed = facts["accession_number"].nunique()
    report.missing_normalized_fields = sorted(set(_CORE_FIELDS) - set(facts["field"].unique()))
    report.amendments_detected = (
        facts[facts["is_amendment"] == 1.0][["accession_number", "amends_accession", "fiscal_year", "fiscal_period"]]
        .drop_duplicates().to_dict("records")
    )

    recon = reconcile_fundamentals_facts(facts)
    q4_checks = recon[recon["check"] == "q4_reconciliation_gap"]
    report.q4_reconciliation = {
        "quarters_checked": int((facts["fiscal_period"] == "Q4").sum()),
        "failures": q4_checks.to_dict("records"),
    }

    try:
        context.store.save("fundamentals_facts", facts,
                           pk=["ticker", "accession_number", "field", "fiscal_year",
                               "fiscal_period", "duration_type"])
        report.rows_written["fundamentals_facts"] = len(facts)
    except Exception as e:                          # noqa: BLE001
        report.exceptions.append(f"persist fundamentals_facts: {e}")
        return report

    try:
        hist = derive_fundamentals_history(context, ticker)
        report.rows_written["fundamentals_history"] = len(hist)
    except Exception as e:                          # noqa: BLE001
        report.exceptions.append(f"derive fundamentals_history: {e}")

    return report


@pytest.mark.skipif(not os.getenv("SEC_USER_AGENT"), reason="SEC EDGAR requires SEC_USER_AGENT")
@pytest.mark.skipif(os.getenv("RUN_LIVE_SEC_INTEGRATION") != "1",
                    reason="live 15-ticker x ~15y SEC pull -- opt in with RUN_LIVE_SEC_INTEGRATION=1")
def test_15_ticker_integration_report():
    from src.context import get_config_context
    _, context = get_config_context("./configs", use_cache=False, save=True)   # loads .env
    _configure_identity()

    reports = {t: build_ticker_integration_report(context, t) for t in INTEGRATION_TICKERS}

    print("\n=== SANITY CHECK: 15-ticker edgartools fundamentals integration ===")
    for t, r in reports.items():
        print(f"  {t:6s} filings={r.filings_discovered} processed={r.filings_processed} "
             f"missing={r.missing_normalized_fields} amendments={len(r.amendments_detected)} "
             f"rows={r.rows_written} exceptions={r.exceptions}")

    for t, r in reports.items():
        assert not r.exceptions, f"{t}: {r.exceptions}"
        assert r.rows_written.get("fundamentals_facts", 0) > 0, f"{t}: no fundamentals_facts rows"
        assert not r.q4_reconciliation.get("failures"), f"{t}: Q4 reconciliation failures"

    print("  Validated.")
