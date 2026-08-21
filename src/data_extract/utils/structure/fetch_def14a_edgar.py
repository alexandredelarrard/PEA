"""
fetch_def14a_edgar.py (src/data_extract/utils/structure/fetch_def14a_edgar.py)
--------------------------------------------------------------------------------
Structured proxy-statement (DEF 14A) data via edgartools' typed `ProxyStatement`
(`filing.obj()`) -- SEC XBRL (ECD taxonomy) plus deterministic HTML-table parsing,
zero LLM cost. Complements `fetch_def14a_llm.py`, which covers director bios and
narrative text.

One filing-level table plus four detail tables:
  `sec_def14a`                  (ticker, accession) -- pay-vs-performance XBRL,
                                CEO pay ratio, audit fees, proposal counts
  `sec_def14a_executive_comp`   (ticker, accession, name, year)
  `sec_def14a_director_comp`    (ticker, accession, name)
  `sec_def14a_ownership`        (ticker, accession, holder_name, holder_type)
  `sec_def14a_votes`            (ticker, accession, proposal_number) -- the BOARD's
                                recommendation, not the vote outcome

Caveats: actual vote-result percentages have no structured tag anywhere in DEF 14A
or 8-K 5.07, so they stay with the LLM path; pay-vs-performance XBRL is filed
mainly by accelerated filers, so EGC/non-accelerated filers yield NaN there while
still populating the HTML-parsed compensation tables. Every parsed row goes
through `def14a_validate`, which repairs values edgartools reports WRONG rather
than absent (missed "(in thousands)" headers, a hardcoded 0.5 for a "*" percent).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.constants.constants import DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_extract.utils.structure.def14a_validate import (
    clean_person_name, repair_director_comp_rows, repair_exec_comp_rows, repair_main_row,
    repair_ownership_rows,
)
from src.data_store.schema import Table, Tables

_MAIN_COLS = [
    "ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
    "company_name", "has_xbrl", "has_individual_executive_data",
    "peo_name", "peo_total_comp", "peo_actually_paid_comp",
    "neo_avg_total_comp", "neo_avg_actually_paid_comp",
    "total_shareholder_return", "peer_group_tsr", "net_income",
    "company_selected_measure_name", "company_selected_measure_value",
    "insider_trading_policy_adopted", "award_timing_mnpi_considered",
    "award_dates_predetermined", "mnpi_disclosure_timed_for_comp_value",
    "ceo_pay_ratio_ceo_comp", "ceo_pay_ratio_median_employee_comp", "ceo_pay_ratio",
    "auditor_name", "audit_fiscal_year_current", "audit_fiscal_year_prior",
    "audit_fees_current", "audit_fees_prior",
    "audit_related_fees_current", "audit_related_fees_prior",
    "tax_fees_current", "tax_fees_prior", "other_fees_current", "other_fees_prior",
    "total_fees_current", "total_fees_prior",
    "n_voting_proposals", "n_say_on_pay_proposals", "n_director_election_proposals",
    "n_auditor_ratification_proposals", "n_equity_plan_proposals", "n_shareholder_proposals",
    "n_board_against_recommendations",
]
_EXEC_COMP_COLS = [
    "ticker", "cik", "accession_number", "filing_date", "name", "title", "year",
    "salary", "bonus", "stock_awards", "option_awards", "non_equity_incentive",
    "pension_change", "other_compensation", "total",
]
_DIRECTOR_COMP_COLS = [
    "ticker", "cik", "accession_number", "filing_date", "name",
    "fees_earned", "stock_awards", "option_awards", "non_equity_incentive",
    "pension_change", "other_compensation", "total",
]
_OWNERSHIP_COLS = [
    "ticker", "cik", "accession_number", "filing_date",
    "holder_name", "holder_type", "shares", "percent_of_class",
]
_VOTES_COLS = [
    "ticker", "cik", "accession_number", "filing_date",
    "proposal_number", "description", "board_recommendation", "proposal_type",
]

# Destination table -> its numeric columns. Doubles as the table list handed to the driver,
# so adding a child table is a single edit and the two can never disagree.
_NUMERIC_COLS: dict[Table, list[str]] = {
    Tables.def14a_edgar: [c for c in _MAIN_COLS if c not in (
        "ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
        "company_name", "peo_name", "company_selected_measure_name", "auditor_name",
    )],
    Tables.def14a_edgar_executive_comp: [
        "year", "salary", "bonus", "stock_awards", "option_awards",
        "non_equity_incentive", "pension_change", "other_compensation", "total"],
    Tables.def14a_edgar_director_comp: [
        "fees_earned", "stock_awards", "option_awards", "non_equity_incentive",
        "pension_change", "other_compensation", "total"],
    Tables.def14a_edgar_ownership: ["shares", "percent_of_class"],
    Tables.def14a_edgar_votes: ["proposal_number"],
}


def _bnum(x) -> float:
    """Bool -> 1.0/0.0 (numeric flag, repo convention); NaN (never None) when unknown, so the
    DB column stays float even when a whole batch lacks the field (see fetch_13d_edgar.py's
    `_num_or_null` precedent -- an all-None object column would be inferred as SQL TEXT)."""
    return float("nan") if x is None else float(bool(x))


def _num(x) -> float:
    """Coerce to float, NaN (never None) when absent/unparseable -- same reasoning as `_bnum`."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _get(proxy, attr):
    """Best-effort attribute read: several `ProxyStatement` properties are XBRL-backed and can
    raise on a malformed/partial filing rather than returning None."""
    try:
        return getattr(proxy, attr)
    except Exception:                                       # noqa: BLE001 -- best-effort only
        return None


def _main_row(ticker: str, cik: str, filing, proxy, proposals: list,
              filed: pd.Timestamp) -> dict:
    ratio = _get(proxy, "ceo_pay_ratio")            # CEOPayRatio dataclass or None
    audit = _get(proxy, "audit_fees")               # AuditFees dataclass or None
    n_by_type = Counter(p.proposal_type for p in proposals)
    n_against = sum(1 for p in proposals if (p.board_recommendation or "").upper() == "AGAINST")

    return {
        "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
        "form": str(filing.form), "filing_date": filed,
        # From the filing index, like every sibling fetcher. `ProxyStatement.fiscal_year_end`
        # was the previous source and never once resolved -- 0 of 329 stored rows had it.
        "period_of_report": filing.period_of_report,
        "company_name": _get(proxy, "company_name"),
        "has_xbrl": _bnum(_get(proxy, "has_xbrl")),
        "has_individual_executive_data": _bnum(_get(proxy, "has_individual_executive_data")),
        "peo_name": _get(proxy, "peo_name"),
        "peo_total_comp": _num(_get(proxy, "peo_total_comp")),
        "peo_actually_paid_comp": _num(_get(proxy, "peo_actually_paid_comp")),
        "neo_avg_total_comp": _num(_get(proxy, "neo_avg_total_comp")),
        "neo_avg_actually_paid_comp": _num(_get(proxy, "neo_avg_actually_paid_comp")),
        "total_shareholder_return": _num(_get(proxy, "total_shareholder_return")),
        "peer_group_tsr": _num(_get(proxy, "peer_group_tsr")),
        "net_income": _num(_get(proxy, "net_income")),
        "company_selected_measure_name": _get(proxy, "company_selected_measure"),
        "company_selected_measure_value": _num(_get(proxy, "company_selected_measure_value")),
        "insider_trading_policy_adopted": _bnum(_get(proxy, "insider_trading_policy_adopted")),
        "award_timing_mnpi_considered": _bnum(_get(proxy, "award_timing_mnpi_considered")),
        "award_dates_predetermined": _bnum(_get(proxy, "award_dates_predetermined")),
        "mnpi_disclosure_timed_for_comp_value": _bnum(_get(proxy, "mnpi_disclosure_timed_for_comp_value")),
        "ceo_pay_ratio_ceo_comp": _num(ratio.ceo_compensation) if ratio else float("nan"),
        "ceo_pay_ratio_median_employee_comp": _num(ratio.median_employee_compensation) if ratio else float("nan"),
        "ceo_pay_ratio": _num(ratio.ratio) if ratio else float("nan"),
        "auditor_name": (audit.auditor_name or None) if audit else None,
        # NOTE: edgartools' `AuditFees.current_year` / `.prior_year` are the fee table's YEAR
        # LABELS (2025, 2024) -- not fees. The dollar amounts are `audit_fees_current/_prior`
        # below. These columns were previously named `audit_fee_*_year`, which read as a fee.
        "audit_fiscal_year_current": _num(audit.current_year) if (audit and audit.current_year) else float("nan"),
        "audit_fiscal_year_prior": _num(audit.prior_year) if (audit and audit.prior_year) else float("nan"),
        "audit_fees_current": _num(audit.audit_fees_current) if audit else float("nan"),
        "audit_fees_prior": _num(audit.audit_fees_prior) if audit else float("nan"),
        "audit_related_fees_current": _num(audit.audit_related_current) if audit else float("nan"),
        "audit_related_fees_prior": _num(audit.audit_related_prior) if audit else float("nan"),
        "tax_fees_current": _num(audit.tax_fees_current) if audit else float("nan"),
        "tax_fees_prior": _num(audit.tax_fees_prior) if audit else float("nan"),
        "other_fees_current": _num(audit.other_fees_current) if audit else float("nan"),
        "other_fees_prior": _num(audit.other_fees_prior) if audit else float("nan"),
        "total_fees_current": _num(audit.total_current) if audit else float("nan"),
        "total_fees_prior": _num(audit.total_prior) if audit else float("nan"),
        "n_voting_proposals": _num(len(proposals)),
        "n_say_on_pay_proposals": _num(n_by_type.get("say_on_pay", 0)),
        "n_director_election_proposals": _num(n_by_type.get("director_election", 0)),
        "n_auditor_ratification_proposals": _num(n_by_type.get("auditor_ratification", 0)),
        "n_equity_plan_proposals": _num(n_by_type.get("equity_plan", 0)),
        "n_shareholder_proposals": _num(n_by_type.get("shareholder_proposal", 0)),
        "n_board_against_recommendations": _num(n_against),
    }


def _exec_comp_rows(ticker: str, cik: str, filing, proxy, filed: pd.Timestamp) -> list[dict]:
    df = _get(proxy, "summary_compensation_table")
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        name = (r.get("name") or "").strip()
        if not name or pd.isna(r.get("year")):
            continue
        rows.append({
            "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
            "filing_date": filed,
            "name": name, "title": r.get("title") or None, "year": _num(r.get("year")),
            "salary": _num(r.get("salary")), "bonus": _num(r.get("bonus")),
            "stock_awards": _num(r.get("stock_awards")), "option_awards": _num(r.get("option_awards")),
            "non_equity_incentive": _num(r.get("non_equity_incentive")),
            "pension_change": _num(r.get("pension_change")),
            "other_compensation": _num(r.get("other_compensation")),
            "total": _num(r.get("total")),
        })
    return rows


def _director_comp_rows(ticker: str, cik: str, filing, proxy, filed: pd.Timestamp) -> list[dict]:
    df = _get(proxy, "director_compensation_table")
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        name = (r.get("name") or "").strip()
        if not name:
            continue
        rows.append({
            "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
            "filing_date": filed, "name": name,
            "fees_earned": _num(r.get("fees_earned")), "stock_awards": _num(r.get("stock_awards")),
            "option_awards": _num(r.get("option_awards")),
            "non_equity_incentive": _num(r.get("non_equity_incentive")),
            "pension_change": _num(r.get("pension_change")),
            "other_compensation": _num(r.get("other_compensation")),
            "total": _num(r.get("total")),
        })
    return rows


def _ownership_rows(ticker: str, cik: str, filing, proxy, filed: pd.Timestamp) -> list[dict]:
    df = _get(proxy, "beneficial_ownership")
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        name = (r.get("holder_name") or "").strip()
        if not name:
            continue
        rows.append({
            "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
            "filing_date": filed,
            "holder_name": name, "holder_type": (r.get("holder_type") or None),
            "shares": _num(r.get("shares")), "percent_of_class": _num(r.get("percent_of_class")),
        })
    return rows


def _votes_rows(ticker: str, cik: str, filing, proposals: list, filed: pd.Timestamp) -> list[dict]:
    rows = []
    for p in proposals:
        rows.append({
            "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
            "filing_date": filed,
            "proposal_number": _num(p.number), "description": p.description,
            "board_recommendation": p.board_recommendation, "proposal_type": p.proposal_type,
        })
    return rows


def build_ticker_def14a_edgar(ticker: str, cik: str, *, since: pd.Timestamp | None = None,
                              done_accessions: frozenset[str] = frozenset(),
                              ) -> dict[Table, pd.DataFrame]:
    """Main + 4 child-table rows from each filing's typed `ProxyStatement`. A filing whose
    `.obj()` doesn't resolve to one -- a parse failure, or a DEF 14C, which edgartools'
    `PROXY_FORMS` dispatch omits so it falls through to a generic XBRL-only object -- is
    skipped here and stays covered by `fetch_def14a_llm.py`, which works off raw text."""
    main_rows: list[dict] = []
    exec_rows: list[dict] = []
    dir_rows: list[dict] = []
    own_rows: list[dict] = []
    vote_rows: list[dict] = []
    for f in new_filings(ticker, DEF14A_FORMS, since, done_accessions):
        try:
            proxy = f.obj()
        except Exception:                                   # noqa: BLE001 -- best-effort only
            continue
        if proxy is None or not hasattr(proxy, "voting_proposals"):
            continue                                         # not a ProxyStatement (see docstring)

        filed = pd.Timestamp(f.filing_date).normalize()
        proposals = _get(proxy, "voting_proposals") or []    # XBRL-backed: read once per filing

        # Repair PER FILING (not per table): re-typing an ownership row needs that same filing's
        # comp tables to know who its insiders are. See def14a_validate.py's module docstring.
        main = repair_main_row(_main_row(ticker, cik, f, proxy, proposals, filed))
        execs = repair_exec_comp_rows(_exec_comp_rows(ticker, cik, f, proxy, filed))
        directors = repair_director_comp_rows(_director_comp_rows(ticker, cik, f, proxy, filed))
        insiders = {n for n in (
            [clean_person_name(main.get("peo_name"))]
            + [r["name"] for r in execs] + [r["name"] for r in directors]
        ) if n}

        main_rows.append(main)
        exec_rows.extend(execs)
        dir_rows.extend(directors)
        own_rows.extend(repair_ownership_rows(
            _ownership_rows(ticker, cik, f, proxy, filed), insiders))
        vote_rows.extend(_votes_rows(ticker, cik, f, proposals, filed))

    frames = {
        Tables.def14a_edgar: pd.DataFrame(main_rows, columns=_MAIN_COLS),
        Tables.def14a_edgar_executive_comp: pd.DataFrame(exec_rows, columns=_EXEC_COMP_COLS),
        Tables.def14a_edgar_director_comp: pd.DataFrame(dir_rows, columns=_DIRECTOR_COMP_COLS),
        Tables.def14a_edgar_ownership: pd.DataFrame(own_rows, columns=_OWNERSHIP_COLS),
        Tables.def14a_edgar_votes: pd.DataFrame(vote_rows, columns=_VOTES_COLS),
    }
    # De-dup on each table's own PK FIRST (two filings in one batch can restate the same
    # NEO-year, and an upsert touching one PK row twice is an error in Postgres), then coerce --
    # coercing first would do the work on rows about to be dropped.
    return {table: _coerce_numeric(df.drop_duplicates(subset=list(table.pk), keep="last"),
                                   _NUMERIC_COLS[table])
            for table, df in frames.items()}


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_def14a_edgar(context: Context, tickers: list[str], years_history: int) -> None:
    run_edgar_fetch(context, tickers, years_history, tables=tuple(_NUMERIC_COLS),
                    build=build_ticker_def14a_edgar, desc="DEF 14A (edgartools)")
