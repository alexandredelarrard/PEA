"""
fetch_8k_edgar.py (src/data_extract/utils/structure/fetch_8k_edgar.py)
------------------------------------------------------------------------
Extracts SEC Form 8-K filings into `sec_8k` using `edgartools` (`filing.obj()`).
Replaces legacy submission JSONs by capturing item codes along with typed 
`CurrentReport` metadata (`has_earnings`, `has_press_release`, `is_amendment`).

Data Grain:
- One row per (ticker, accession_number).

Scope Note:
- Exhibit Financial Parsing Out of Scope: Parsing financial statements directly 
  from attached earnings releases (`get_income_statement()`) is intentionally 
  excluded to avoid storing unstandardized, duplicate data that competes with 
  `fundamentals_facts`.
"""

from __future__ import annotations

import pandas as pd
from edgar import Company
import itertools

from src.constants.constants import SEC_8K_FORMS, SEC_8K_TABLE, SEC_8K_HIGH_SIGNAL_ITEMS
from src.context import Context
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity

_COLS = ["ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
        "n_items", "is_amendment", "has_earnings", "has_press_release",
        "primary_document", "item", "item_tag", "item_text"]


def _n_items(items: str) -> int:
    return sum(1 for x in str(items or "").split(",") if x.strip())


def _filing_row(ticker: str, cik : str, filing) -> dict:
    """One 8-K filing -> one `sec_8k` row. `has_earnings`/`has_press_release` come
    from edgartools' typed `CurrentReport` (`filing.obj()`) on a best-effort basis:
    a filing whose `.obj()` parse fails keeps its item-code row (still ~100%
    reliable, straight from the filing index), just with both flags null rather
    than losing the row entirely."""
    has_earnings = has_press_release = obj = None
    try:
        obj = filing.obj()
        has_earnings = float(bool(obj.has_earnings))
        has_press_release = float(bool(obj.has_press_release))
    except Exception:                                   # noqa: BLE001 -- best-effort only
        pass

    items = getattr(filing, "items", "") or ""
    item_list = [i.strip() for i in items.split(",") if i.strip()]

    base_row = {
        "ticker": ticker, 
        "cik": cik, 
        "accession_number": filing.accession_number,
        "form": filing.form, 
        "filing_date": pd.Timestamp(filing.filing_date),
        "period_of_report": filing.period_of_report,
        "n_items": _n_items(items),
        "is_amendment": 1.0 if str(filing.form).upper().endswith("/A") else 0.0,
        "has_earnings": has_earnings, "has_press_release": has_press_release,
        "primary_document": getattr(filing, "primary_document", None),
    }

    results = []
    for item_code in item_list:
        # Create an independent dictionary copy for each item
        row = base_row.copy()

        # 1. Map item code to human-readable tag safely
        row["item"] = item_code
        row["item_tag"] = SEC_8K_HIGH_SIGNAL_ITEMS.get(
            item_code, "other_unclassified_item"
        )

        # 2. Safely extract item text from CurrentReport (obj) or fallback
        item_text = None
        if obj is not None:
            try:
                item_text = obj["Item " + item_code]
            except Exception:
                item_text = None

        row["item_text"] = item_text or ""
        results.append(row)

    return results

def build_ticker_8k_edgar(ticker: str, cik :str, since: pd.Timestamp | None = None,
                         done_accessions: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Walks `Company(ticker).get_filings(form=SEC_8K_FORMS)`, skips accessions
    already in `done_accessions` or filed before `since`, and builds one row per
    filing via `_filing_row`."""

    company = Company(ticker)
    filings = company.get_filings(form=list(SEC_8K_FORMS))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    rows = [_filing_row(ticker, cik, f) for f in sorted_filings if f.accession_number not in done_accessions]
    rows = list(itertools.chain.from_iterable(rows))
    return pd.DataFrame(rows, columns=_COLS)


def fetch_8k_edgar(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Public entry point (mirrors `fetch_fundamentals_edgartools`'s conventions):
    per-ticker try/except so one bad ticker cannot abort the batch, incremental via
    `existing_filings` (dedup by accession) PLUS a `since` cutoff from the
    extraction manifest (see `run_manifest.py`) -- a routine run only relists
    filings from the last run's date onward, while a ticker-count change or the
    `manifest_full_rescan_days` self-heal window falls back to the FULL `years`
    window (falls back to `data_extract.years_history`, matching every sibling
    fetcher's default window), gap-filling instead of trusting the cutoff alone.
    Tickers are walked CONCURRENTLY on a thread pool (`run_per_ticker`) -- the walk
    is network I/O bound (SEC filing downloads via edgartools), not CPU bound, so
    this is a pure speed win with no change to the extracted rows (see
    parallel_fetch.py's module docstring)."""
    _configure_identity()
    years = int(years if years is not None else context.config.data_extract.years_history)
    full_since = pd.Timestamp.today() - pd.DateOffset(years=years)
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    rescan_days = int(getattr(context.config.data_extract, "manifest_full_rescan_days", 30))
    since, is_full_rescan = manifest_window(
        context, SEC_8K_TABLE, len(cik_map), fallback_since=full_since,
        full_rescan_days=rescan_days)

    seen = existing_filings(context, SEC_8K_TABLE)

    def _worker(ticker: str, cik: str) -> tuple[int, bool]:
        try:
            out = build_ticker_8k_edgar(ticker, cik, since=since, done_accessions=seen)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("fetch_8k_edgar: %s failed (%s)", ticker, e)
            return 0, False
        if not out.empty:
            context.store.save(SEC_8K_TABLE, out)
        return len(out), True

    results = run_per_ticker(cik_map, _worker, desc="8-K (edgartools)")
    total_rows = sum(n for n, _ in results)
    failed = sum(1 for _, ok in results if not ok)

    context.log.info("fetch_8k_edgar: +%d filings across %d/%d ticker(s) (%d failed) -> '%s'",
                     total_rows, len(results), len(cik_map), failed, SEC_8K_TABLE)
    record_run(context, SEC_8K_TABLE, len(cik_map), total_rows, is_full_rescan=is_full_rescan)
    return context.store.load(SEC_8K_TABLE)
