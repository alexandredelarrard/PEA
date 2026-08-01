"""
Standalone rollout script for the edgartools-based fundamentals pipeline's
15-ticker (or a custom --tickers list) integration report. Reuses the exact
same report-building function as
tests/data_extract/test_fundamentals_edgartools_integration.py, so the test and
a manual rollout run never drift apart.

Run:
    python -m scripts.run_fundamentals_integration_report
    python -m scripts.run_fundamentals_integration_report --tickers JPM,MAA,ORCL
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import click

from src.context import get_config_context
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity
from tests.data_extract.test_fundamentals_edgartools_integration import (
    INTEGRATION_TICKERS, build_ticker_integration_report,
)


@click.command()
@click.option("--tickers", default=None, help="Comma-separated tickers; default: the fixed 15-ticker list.")
@click.option("--config", "config_path", default="./configs")
def main(tickers: str | None, config_path: str) -> None:
    _, context = get_config_context(config_path, use_cache=False, save=True)   # loads .env
    _configure_identity()
    ticker_list = [t.strip().upper() for t in tickers.split(",")] if tickers else INTEGRATION_TICKERS

    for t in ticker_list:
        report = build_ticker_integration_report(context, t)
        print(f"\n{t}")
        print(f"  filings_discovered: {report.filings_discovered}")
        print(f"  filings_processed:  {report.filings_processed}")
        print(f"  missing_fields:     {report.missing_normalized_fields}")
        print(f"  amendments:         {len(report.amendments_detected)}")
        print(f"  q4_reconciliation:  {report.q4_reconciliation}")
        print(f"  rows_written:       {report.rows_written}")
        print(f"  exceptions:         {report.exceptions}")


if __name__ == "__main__":
    main()
