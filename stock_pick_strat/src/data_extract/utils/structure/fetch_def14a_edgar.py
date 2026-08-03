"""
fetch_def14a_edgar.py (src/data_extract/utils/structure/fetch_def14a_edgar.py)
--------------------------------------------------------------------------------
Extracts structured proxy statement (DEF 14A) data via `edgartools` typed 
`ProxyStatement` objects (`filing.obj()`). Uses deterministic SEC XBRL (ECD 
taxonomy) and HTML-table parsing with zero LLM cost or hallucination risk. 
Complements `fetch_def14a_llm.py` (which covers director bios and narrative text).

Data Grain & Relational Tables:
- `def14a_edgar` (Filing Summary): One row per (ticker, accession_number). Tracks 
  pay-vs-performance XBRL tags, CEO pay ratio, audit fees, and proposal counts.
- `def14a_edgar_executive_comp`: One row per (ticker, accession, name, year). 
  Summary Compensation Table across Named Executive Officers (NEOs).
- `def14a_edgar_director_comp`: One row per (ticker, accession, name). Non-employee 
  Director Compensation Table (Reg S-K Item 402(k)).
- `def14a_edgar_ownership`: One row per (ticker, accession, holder_name, holder_type). 
  Beneficial ownership tables (5%+ holders and corporate insiders).
- `def14a_edgar_votes`: One row per (ticker, accession, proposal_number). Proposal 
  descriptions, classified proposal types, and board recommendations (FOR/AGAINST).

Key Guardrails & Caveats:
- Voting Outcomes: Actual shareholder voting result percentages lack structured 
  SEC tags in DEF 14A / 8-K 5.07 filings; these remain handled by `fetch_def14a_llm.py`.
- XBRL Coverage: Pay-vs-Performance XBRL tags apply primarily to accelerated 
  filers. Non-accelerated/EGC filers yield NaNs for XBRL fields but still populate 
  HTML-parsed compensation tables.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
from edgar import Company

from src.constants.constants import (
    DEF14A_EDGAR_DIRECTOR_COMP_TABLE, DEF14A_EDGAR_EXEC_COMP_TABLE, DEF14A_EDGAR_OWNERSHIP_TABLE,
    DEF14A_EDGAR_TABLE, DEF14A_EDGAR_VOTES_TABLE, DEF14A_FORMS,
)
from src.context import Context
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity

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
    "auditor_name", "audit_fee_current_year", "audit_fee_prior_year",
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

_MAIN_NUMERIC_COLS = [c for c in _MAIN_COLS if c not in (
    "ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
    "company_name", "peo_name", "company_selected_measure_name", "auditor_name",
)]
_EXEC_COMP_NUMERIC_COLS = ["year", "salary", "bonus", "stock_awards", "option_awards",
                          "non_equity_incentive", "pension_change", "other_compensation", "total"]
_DIRECTOR_COMP_NUMERIC_COLS = ["fees_earned", "stock_awards", "option_awards",
                              "non_equity_incentive", "pension_change", "other_compensation", "total"]
_OWNERSHIP_NUMERIC_COLS = ["shares", "percent_of_class"]
_VOTES_NUMERIC_COLS = ["proposal_number"]


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


def _main_row(ticker: str, cik: str, filing, proxy) -> dict:
    period_of_report = None
    fye = _get(proxy, "fiscal_year_end")
    if fye:
        try:
            period_of_report = pd.Timestamp(fye).normalize()
        except (TypeError, ValueError):
            period_of_report = None

    ratio = _get(proxy, "ceo_pay_ratio")            # CEOPayRatio dataclass or None
    audit = _get(proxy, "audit_fees")               # AuditFees dataclass or None
    proposals = _get(proxy, "voting_proposals") or []
    n_by_type = Counter(p.proposal_type for p in proposals)
    n_against = sum(1 for p in proposals if (p.board_recommendation or "").upper() == "AGAINST")

    return {
        "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
        "form": str(filing.form), "filing_date": pd.Timestamp(filing.filing_date).normalize(),
        "period_of_report": period_of_report,
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
        "audit_fee_current_year": _num(audit.current_year) if (audit and audit.current_year) else float("nan"),
        "audit_fee_prior_year": _num(audit.prior_year) if (audit and audit.prior_year) else float("nan"),
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


def _exec_comp_rows(ticker: str, cik: str, filing, proxy) -> list[dict]:
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
            "filing_date": pd.Timestamp(filing.filing_date).normalize(),
            "name": name, "title": r.get("title") or None, "year": _num(r.get("year")),
            "salary": _num(r.get("salary")), "bonus": _num(r.get("bonus")),
            "stock_awards": _num(r.get("stock_awards")), "option_awards": _num(r.get("option_awards")),
            "non_equity_incentive": _num(r.get("non_equity_incentive")),
            "pension_change": _num(r.get("pension_change")),
            "other_compensation": _num(r.get("other_compensation")),
            "total": _num(r.get("total")),
        })
    return rows


def _director_comp_rows(ticker: str, cik: str, filing, proxy) -> list[dict]:
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
            "filing_date": pd.Timestamp(filing.filing_date).normalize(), "name": name,
            "fees_earned": _num(r.get("fees_earned")), "stock_awards": _num(r.get("stock_awards")),
            "option_awards": _num(r.get("option_awards")),
            "non_equity_incentive": _num(r.get("non_equity_incentive")),
            "pension_change": _num(r.get("pension_change")),
            "other_compensation": _num(r.get("other_compensation")),
            "total": _num(r.get("total")),
        })
    return rows


def _ownership_rows(ticker: str, cik: str, filing, proxy) -> list[dict]:
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
            "filing_date": pd.Timestamp(filing.filing_date).normalize(),
            "holder_name": name, "holder_type": (r.get("holder_type") or None),
            "shares": _num(r.get("shares")), "percent_of_class": _num(r.get("percent_of_class")),
        })
    return rows


def _votes_rows(ticker: str, cik: str, filing, proxy) -> list[dict]:
    proposals = _get(proxy, "voting_proposals") or []
    rows = []
    for p in proposals:
        rows.append({
            "ticker": ticker, "cik": cik, "accession_number": filing.accession_number,
            "filing_date": pd.Timestamp(filing.filing_date).normalize(),
            "proposal_number": _num(p.number), "description": p.description,
            "board_recommendation": p.board_recommendation, "proposal_type": p.proposal_type,
        })
    return rows


def build_ticker_def14a_edgar(
    ticker: str, cik: str, since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walks `Company(ticker).get_filings(form=DEF14A_FORMS)`, skips accessions already in
    `done_accessions` or filed before `since`, and builds the main + 4 child-table rows via
    `filing.obj()`'s typed `ProxyStatement`. A filing whose `.obj()` doesn't resolve to a
    `ProxyStatement` (parse failure, or a DEF 14C -- edgartools' `PROXY_FORMS` dispatch list
    does not include DEF 14C, so it falls through to a generic/XBRL-only object with none of
    the proxy-specific properties) is skipped entirely for this deterministic path -- it
    remains covered by `fetch_def14a_llm.py`'s LLM extraction, which works off raw text."""
    company = Company(ticker)
    filings = company.get_filings(form=list(DEF14A_FORMS))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    main_rows: list[dict] = []
    exec_rows: list[dict] = []
    dir_rows: list[dict] = []
    own_rows: list[dict] = []
    vote_rows: list[dict] = []
    for f in sorted_filings:
        if f.accession_number in done_accessions:
            continue
        try:
            proxy = f.obj()
        except Exception:                                   # noqa: BLE001 -- best-effort only
            continue
        if proxy is None or not hasattr(proxy, "voting_proposals"):
            continue                                         # not a ProxyStatement (see docstring)

        main_rows.append(_main_row(ticker, cik, f, proxy))
        exec_rows.extend(_exec_comp_rows(ticker, cik, f, proxy))
        dir_rows.extend(_director_comp_rows(ticker, cik, f, proxy))
        own_rows.extend(_ownership_rows(ticker, cik, f, proxy))
        vote_rows.extend(_votes_rows(ticker, cik, f, proxy))

    return (pd.DataFrame(main_rows, columns=_MAIN_COLS),
            pd.DataFrame(exec_rows, columns=_EXEC_COMP_COLS),
            pd.DataFrame(dir_rows, columns=_DIRECTOR_COMP_COLS),
            pd.DataFrame(own_rows, columns=_OWNERSHIP_COLS),
            pd.DataFrame(vote_rows, columns=_VOTES_COLS))


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_def14a_edgar(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Public entry point (mirrors `fetch_8k_edgar`'s conventions): per-ticker try/except so one
    bad ticker cannot abort the batch, incremental via `existing_filings` (dedup by accession
    ONLY, keyed off the main `def14a_edgar` table -- every ticker's FULL `years` window is
    re-listed every run, gap-filling instead of resuming from a max-date cutoff), scoped by
    `years` (falls back to `data_extract.years_history`). Saves the main row + all 4 child
    tables per ticker. Tickers are walked CONCURRENTLY on a thread pool (`run_per_ticker`) --
    see parallel_fetch.py's module docstring."""
    _configure_identity()
    years = int(years if years is not None else context.config.data_extract.years_history)
    since = pd.Timestamp.today() - pd.DateOffset(years=years)
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen = existing_filings(context, DEF14A_EDGAR_TABLE)

    def _worker(ticker: str, cik: str) -> dict[str, int] | None:
        try:
            main_df, exec_df, dir_df, own_df, vote_df = build_ticker_def14a_edgar(
                ticker, cik, since=since, done_accessions=seen)
        except Exception as e:                              # noqa: BLE001
            context.log.warning("fetch_def14a_edgar: %s failed (%s)", ticker, e)
            return None

        totals = {"main": 0, "exec_comp": 0, "director_comp": 0, "ownership": 0, "votes": 0}
        if not main_df.empty:
            main_df = _coerce_numeric(main_df, _MAIN_NUMERIC_COLS)
            main_df = main_df.drop_duplicates(subset=["ticker", "accession_number"], keep="last")
            context.store.save(DEF14A_EDGAR_TABLE, main_df)
            totals["main"] += len(main_df)
        if not exec_df.empty:
            exec_df = _coerce_numeric(exec_df, _EXEC_COMP_NUMERIC_COLS)
            exec_df = exec_df.drop_duplicates(subset=["ticker", "accession_number", "name", "year"], keep="last")
            context.store.save(DEF14A_EDGAR_EXEC_COMP_TABLE, exec_df)
            totals["exec_comp"] += len(exec_df)
        if not dir_df.empty:
            dir_df = _coerce_numeric(dir_df, _DIRECTOR_COMP_NUMERIC_COLS)
            dir_df = dir_df.drop_duplicates(subset=["ticker", "accession_number", "name"], keep="last")
            context.store.save(DEF14A_EDGAR_DIRECTOR_COMP_TABLE, dir_df)
            totals["director_comp"] += len(dir_df)
        if not own_df.empty:
            own_df = _coerce_numeric(own_df, _OWNERSHIP_NUMERIC_COLS)
            own_df = own_df.drop_duplicates(
                subset=["ticker", "accession_number", "holder_name", "holder_type"], keep="last")
            context.store.save(DEF14A_EDGAR_OWNERSHIP_TABLE, own_df)
            totals["ownership"] += len(own_df)
        if not vote_df.empty:
            vote_df = _coerce_numeric(vote_df, _VOTES_NUMERIC_COLS)
            vote_df = vote_df.drop_duplicates(
                subset=["ticker", "accession_number", "proposal_number"], keep="last")
            context.store.save(DEF14A_EDGAR_VOTES_TABLE, vote_df)
            totals["votes"] += len(vote_df)
        return totals

    results = run_per_ticker(cik_map, _worker, desc="DEF 14A (edgartools)")
    failed = sum(1 for r in results if r is None)
    totals = {"main": 0, "exec_comp": 0, "director_comp": 0, "ownership": 0, "votes": 0}
    for r in results:
        if r is not None:
            for k in totals:
                totals[k] += r[k]

    context.log.info(
        "fetch_def14a_edgar: +%d filings, +%d exec-comp / +%d director-comp / +%d ownership / "
        "+%d vote rows across %d/%d ticker(s) (%d failed) -> '%s'",
        totals["main"], totals["exec_comp"], totals["director_comp"], totals["ownership"],
        totals["votes"], len(results), len(cik_map), failed, DEF14A_EDGAR_TABLE)
    return context.store.load(DEF14A_EDGAR_TABLE)
