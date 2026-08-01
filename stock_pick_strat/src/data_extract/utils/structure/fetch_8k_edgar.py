"""
fetch_8k_edgar.py  (src/data_extract/utils/structure/fetch_8k_edgar.py)
------------------------------------------------------------------------
edgartools-based 8-K extraction -> `sec_8k`. Replaces `fetch_8k_items.py`'s
submissions-JSON approach (item codes only) the same way `fetch_fundamentals_edgar.py`
replaced the companyfacts-JSON fundamentals path: walk each filing directly via
`edgar.Company(ticker).get_filings(form=[...])`, which additionally exposes the
filing's typed `CurrentReport` object (`filing.obj()`) -- `has_earnings` /
`has_press_release` / `is_amendment` -- none of which the submissions JSON carries.

Kept OUT of scope deliberately: `CurrentReport.get_income_statement()` et al. (parsing
headline financials out of an attached earnings-release exhibit). That would duplicate
`fundamentals_facts` with a strictly less reliable source (exhibit XBRL is optional and
inconsistently tagged) -- a real "flash vs. official" feature is a separate, deliberate
design, not a byproduct of this table's refresh.

One row per (ticker, accession_number) -- an 8-K is one filing, one event; no
sub-grain needed the way SC 13D's multiple reporting persons requires one.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_8K_FORMS, SEC_8K_TABLE
from src.context import Context
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity

_TABLE = SEC_8K_TABLE
_COLS = ["ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
        "items", "n_items", "is_amendment", "has_earnings", "has_press_release",
        "primary_document"]


def _n_items(items: str) -> int:
    return sum(1 for x in str(items or "").split(",") if x.strip())


def _filing_row(ticker: str, filing) -> dict:
    """One 8-K filing -> one `sec_8k` row. `has_earnings`/`has_press_release` come
    from edgartools' typed `CurrentReport` (`filing.obj()`) on a best-effort basis:
    a filing whose `.obj()` parse fails keeps its item-code row (still ~100%
    reliable, straight from the filing index), just with both flags null rather
    than losing the row entirely."""
    has_earnings = has_press_release = None
    try:
        obj = filing.obj()
        has_earnings = float(bool(obj.has_earnings))
        has_press_release = float(bool(obj.has_press_release))
    except Exception:                                   # noqa: BLE001 -- best-effort only
        pass
    items = getattr(filing, "items", "") or ""
    return {
        "ticker": ticker, "cik": None, "accession_number": filing.accession_number,
        "form": filing.form, "filing_date": pd.Timestamp(filing.filing_date),
        "period_of_report": filing.period_of_report,
        "items": items, "n_items": _n_items(items),
        "is_amendment": 1.0 if str(filing.form).upper().endswith("/A") else 0.0,
        "has_earnings": has_earnings, "has_press_release": has_press_release,
        "primary_document": getattr(filing, "primary_document", None),
    }


def build_ticker_8k_edgar(ticker: str, *, since: pd.Timestamp | None = None,
                         done_accessions: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Walks `Company(ticker).get_filings(form=SEC_8K_FORMS)`, skips accessions
    already in `done_accessions` or filed before `since`, and builds one row per
    filing via `_filing_row`."""
    from edgar import Company

    company = Company(ticker)
    filings = company.get_filings(form=list(SEC_8K_FORMS))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    rows = [_filing_row(ticker, f) for f in sorted_filings if f.accession_number not in done_accessions]
    return pd.DataFrame(rows, columns=_COLS)


def fetch_8k_edgar(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Public entry point (mirrors `fetch_fundamentals_edgartools`'s conventions):
    per-ticker try/except so one bad ticker cannot abort the batch, incremental via
    `existing_filings` (dedup by accession + per-ticker resume cutoff), scoped by
    `years` (falls back to `data_extract.years_history`, matching every sibling
    fetcher's default window)."""
    _configure_identity()
    years = int(years if years is not None else context.config.data_extract.years_history)
    since = pd.Timestamp.today() - pd.DateOffset(years=years)
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen, last_by_ticker = existing_filings(context, _TABLE)
    all_frames: list[pd.DataFrame] = []
    touched, failed = 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="8-K (edgartools)"):
        ticker = r["ticker"]
        try:
            ticker_since = max(since, last_by_ticker[ticker]) if ticker in last_by_ticker else since
            out = build_ticker_8k_edgar(ticker, since=ticker_since, done_accessions=seen)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("fetch_8k_edgar: %s failed (%s)", ticker, e)
            failed += 1
            continue
        if not out.empty:
            context.store.save(_TABLE, out)
            seen.update(out["accession_number"].astype(str))
            all_frames.append(out)
        touched += 1

    context.log.info("fetch_8k_edgar: +%d filings across %d/%d ticker(s) (%d failed) -> '%s'",
                     sum(len(f) for f in all_frames), touched, len(cik_map), failed, _TABLE)
    return context.store.load(_TABLE)
