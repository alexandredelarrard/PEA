"""
fetch_8k_edgar.py (src/data_extract/utils/structure/fetch_8k_edgar.py)
------------------------------------------------------------------------
SEC Form 8-K filings -> `sec_8k`, one row per (ticker, accession, item code).
Item codes come from the filing index; `has_earnings` / `has_press_release` and
the per-item text come from edgartools' typed `CurrentReport` (`filing.obj()`).

Parsing financial statements out of an attached earnings release is deliberately
out of scope -- it would store unstandardized figures competing with
`fundamentals_facts`.
"""
from __future__ import annotations

import itertools

import pandas as pd

from src.constants.constants import SEC_8K_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_store.schema import Table, Tables

_COLS = ["ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
        "n_items", "is_amendment", "has_earnings", "has_press_release",
        "primary_document", "item", "item_tag", "item_text"]

# Curated leading distress/governance codes for the feature layer; any other code is
# tagged `other_unclassified_item` rather than dropped.
_HIGH_SIGNAL_ITEMS = {
    # 1: registrant's business and operations
    "1.01": "material_agreement_entered",
    "1.02": "material_agreement_terminated",
    "1.03": "bankruptcy_or_receivership",
    "1.04": "mine_safety_reporting",
    "1.05": "cybersecurity_incidents",
    # 2: financial information
    "2.01": "completion_acquisition_or_disposition",
    "2.02": "results_of_operations_and_financial_condition",
    "2.03": "creation_of_direct_financial_obligation",
    "2.04": "triggering_events_accelerating_financial_obligation",
    "2.05": "restructuring_costs",
    "2.06": "impairment",
    # 3: securities and trading markets
    "3.01": "delisting_or_covenant",
    "3.02": "unregistered_sales_of_equity",
    "3.03": "material_modification_to_security_rights",
    # 4: accountants and financial statements
    "4.01": "auditor_change",
    "4.02": "non_reliance_restatement",
    # 5: corporate governance and management
    "5.01": "change_in_control",
    "5.02": "exec_or_director_change",
    "5.03": "bylaw_change",
    "5.04": "employee_benefit_plan_trading_suspension",
    "5.05": "code_of_ethics_amendment_or_waiver",
    "5.06": "change_in_shell_company_status",
    "5.07": "vote_of_security_holders",
    "5.08": "shareholder_director_nominations",
    # 6: asset-backed securities
    "6.01": "abs_informational_computational_material",
    "6.02": "change_of_servicer_or_trustee",
    "6.03": "change_in_credit_enhancement",
    "6.04": "failure_to_make_required_distribution",
    "6.05": "securities_act_updating_disclosure",
    # 7: Regulation FD
    "7.01": "regulation_fd_disclosure",
    # 8: other events
    "8.01": "other_events",
    # 9: financial statements and exhibits
    "9.01": "financial_statements_and_exhibits",
}


def _filing_row(ticker: str, cik: str, filing) -> list[dict]:
    """One 8-K -> one row per item code. `has_earnings`/`has_press_release` are
    best-effort: a filing whose `.obj()` parse fails keeps its item rows (those come
    straight from the filing index) with both flags NaN.

    NaN rather than None: `store.ensure_table` infers column types from the first frame
    written to a cold table, so an all-None column would be created TEXT for good."""
    # Item codes come free off the filing index. Read them BEFORE `.obj()`: with no item codes
    # this filing yields no rows at all, so the parse would be thrown away.
    items = getattr(filing, "items", "") or ""
    item_list = [i.strip() for i in str(items).split(",") if i.strip()]
    if not item_list:
        return []

    has_earnings = has_press_release = float("nan")
    obj = None
    try:
        obj = filing.obj()
        has_earnings = float(bool(obj.has_earnings))
        has_press_release = float(bool(obj.has_press_release))
    except Exception:                                   # noqa: BLE001 -- best-effort only
        pass

    base = {
        "ticker": ticker,
        "cik": cik,
        "accession_number": filing.accession_number,
        "form": filing.form,
        "filing_date": pd.Timestamp(filing.filing_date),
        "period_of_report": filing.period_of_report,
        "n_items": len(item_list),
        "is_amendment": 1.0 if str(filing.form).upper().endswith("/A") else 0.0,
        "has_earnings": has_earnings,
        "has_press_release": has_press_release,
        "primary_document": getattr(filing, "primary_document", None),
    }

    rows = []
    for item_code in item_list:
        item_text = None
        if obj is not None:
            try:
                item_text = obj["Item " + item_code]
            except Exception:                           # noqa: BLE001 -- best-effort only
                item_text = None
        rows.append({**base,
                     "item": item_code,
                     "item_tag": _HIGH_SIGNAL_ITEMS.get(item_code, "other_unclassified_item"),
                     "item_text": item_text or ""})
    return rows


def build_ticker_8k_edgar(ticker: str, cik: str, *, since: pd.Timestamp | None = None,
                          done_accessions: frozenset[str] = frozenset(),
                          ) -> dict[Table, pd.DataFrame]:
    rows = itertools.chain.from_iterable(
        _filing_row(ticker, cik, f)
        for f in new_filings(ticker, SEC_8K_FORMS, since, done_accessions))
    df = pd.DataFrame(list(rows), columns=_COLS)
    # A filing repeating a code in its `items` string (two officer changes -> "5.02,5.02")
    # would make the upsert touch one PK row twice, which Postgres rejects outright.
    return {Tables.sec_8k: df.drop_duplicates(subset=list(Tables.sec_8k.pk), keep="last")}


def fetch_8k_edgar(context: Context, tickers: list[str], years_history: int) -> None:
    run_edgar_fetch(context, tickers, years_history,
                    tables=(Tables.sec_8k,), build=build_ticker_8k_edgar,
                    desc="8-K (edgartools)")
