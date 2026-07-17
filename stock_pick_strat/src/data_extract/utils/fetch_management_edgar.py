"""
fetch_management_edgar.py  (src/data_extract/fetch_management_edgar.py)
----------------------------------------------------------------------
FREE, full-history management & insider data from SEC EDGAR -- a historical
replacement for the yfinance snapshot (which had no history at all):

  * CEO name / age, founder flags, officer count & average age
        <- 10-K Part I "Information about our Executive Officers"
  * net_insider_buying (trailing-6m open-market shares)   [OFF by default]
        <- Form 4 XML transactions (code P buys - code S sells)

Output columns are a SUPERSET of what management_features.py reads
(heldPercentInsiders, heldPercentInstitutions, founder_present, family_owned,
net_insider_buying, ceo_age). The two that EDGAR does NOT give cleanly are
included as NaN placeholders so the feature builder runs unchanged:

  * heldPercentInstitutions -> needs 13F aggregation across all holders
  * heldPercentInsiders / family_owned -> need DEF 14A beneficial-ownership
    table parsing (fragile). Add later via EdgarTools proxy parsing if wanted.

SPEED / INCREMENTAL
-------------------
  * Skip entirely when already built today for the full universe (meta sidecar).
  * The 10-K officer path fetches ONLY filings after each ticker's last parsed
    `as_of` (`D`) -- the D..today window -- and skips already-seen accessions.
  * Per-ticker work runs in a ThreadPoolExecutor (rate-limited in sec_get).
  * Form 4 insiders are OFF by default (`with_insiders=False`): there are
    thousands per large-cap and the 182-day rolling needs the full window, so it
    is NOT incremental -- enable only when you need net_insider_buying (it
    recomputes that ticker's whole window each run).

Rows are point-in-time: officer fields dated by the 10-K filing date, insider
fields by the Form 4 filing date. Fields are left sparse (NaN off their own
as_of) so each is forward-filled independently downstream -- leak-free.

Run:
    python -m src.data_extract.fetch_management_edgar
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.sec_utils import (
    sec_get, load_cik_mapping, load_extract_meta, save_extract_meta, today_iso,
)
from src.data_extract.utils.edgar_fillings import list_filings
from src.data_extract.utils.edgar_extract import (
    html_to_text, extract_executive_officers,
    parse_form4, signed_open_market_shares, rolling_net_insider,
)

# columns management_features.py maps; placeholders kept so it never KeyErrors
_PLACEHOLDER_COLS = ["heldPercentInsiders", "heldPercentInstitutions", "family_owned"]
_OFFICER_FORMS = ["10-K"]
_INSIDER_FORMS = ["4"]
_MAX_WORKERS = 8                      # concurrent tickers (rate-limited in sec_get)


def _resolve_form4_xml(doc_url: str, primary_document: str) -> str:
    """Form 4 primaryDocument is sometimes the XSL-rendered HTML wrapper
    (path contains 'xsl'); the raw XML is the same file without that prefix."""
    if "xsl" in (primary_document or "").lower():
        base = primary_document.split("/")[-1]
        doc_url = doc_url.rsplit("/", 1)[0] + "/" + base
    return doc_url


def _officer_rows(context, ticker, cik, company, years, seen, since) -> list[dict]:
    """One ticker's NEW executive-officer rows (10-K), only after `since` (`D`)."""
    rows = []
    try:
        filings = list_filings(cik, _OFFICER_FORMS, years, company, since=since)
    except Exception as e:
        context.log.warning("%s: 10-K list failed (%s)", ticker, e)
        return rows
    for _, f in filings.iterrows():
        if f["accession_number"] in seen:
            continue
        try:
            info = extract_executive_officers(html_to_text(sec_get(f["doc_url"]).text))
        except Exception as e:
            context.log.warning("%s %s: officer parse failed (%s)", ticker, f["filing_date"].date(), e)
            continue
        if not info["officers"]:
            continue
        rows.append({
            "ticker": ticker, "as_of": f["filing_date"],
            "period": pd.to_datetime(f.get("period_of_report"), errors="coerce"),
            "form_type": f["form"], 
            "accession_number": f["accession_number"],
            "ceo_name": info["ceo_name"], 
            "ceo_age": info["ceo_age"],
            "founder_present": info["founder_present"], 
            "founder_ceo": info["founder_ceo"],
            "n_officers": info["n_officers"], 
            "avg_officer_age": info["avg_officer_age"],
        })
    return rows


def _insider_rows(context, ticker, cik, company, years) -> list[dict]:
    """Trailing-6m net open-market insider shares per Form 4 filing date. NOT
    incremental: the rolling window needs the full history, so this recomputes
    the whole window (only reached when with_insiders=True)."""
    try:
        filings = list_filings(cik, _INSIDER_FORMS, years, company)
    except Exception as e:
        context.log.warning("%s: Form 4 list failed (%s)", ticker, e)
        return []
    per_filing = []
    for _, f in filings.iterrows():
        try:
            url = _resolve_form4_xml(f["doc_url"], f["primary_document"])
            parsed = parse_form4(sec_get(url).content)
        except Exception:
            continue
        if not parsed["transactions"]:
            continue
        per_filing.append({"date": f["filing_date"].strftime("%Y-%m-%d"),
                           "net": signed_open_market_shares(parsed["transactions"])})
    return [{"ticker": ticker, "as_of": pd.Timestamp(r["as_of"]),
             "net_insider_buying": r["net_insider_shares"]}
            for r in rolling_net_insider(per_filing)]


def _rows_for_ticker(context, ticker, cik, company, years, seen, since,
                     with_insiders) -> list[dict]:
    """All new rows for one ticker (runs in a worker thread)."""
    rows = _officer_rows(context, ticker, cik, company, years, seen, since)
    if with_insiders:
        rows += _insider_rows(context, ticker, cik, company, years)
    return rows


def _seen_accessions(existing: pd.DataFrame | None) -> set:
    if existing is None or existing.empty or "accession_number" not in existing.columns:
        return set()
    return set(existing["accession_number"].dropna())


def _last_officer_asof(existing: pd.DataFrame | None) -> dict:
    """Max already-parsed 10-K filing date per ticker (rows carrying an
    accession) -> the incremental cutoff `D` for the officer path."""
    if existing is None or existing.empty or "accession_number" not in existing.columns:
        return {}
    s = existing[existing["accession_number"].notna()][["ticker", "as_of"]].copy()
    s["as_of"] = pd.to_datetime(s["as_of"], errors="coerce")
    s = s.dropna(subset=["as_of"])
    return s.groupby("ticker")["as_of"].max().to_dict()


def _is_up_to_date(context: Context, cik_map: pd.DataFrame) -> bool:
    """True when the history was already built today for the full universe."""
    path = context.paths["MANAGEMENT_HISTORY_PATH"]
    meta = load_extract_meta(path)
    if meta is None or meta.get("last_built") != today_iso() or not path.exists():
        return False
    return meta.get("universe_size", 0) >= len(cik_map)


def fetch_management_edgar(context: Context, tickers: list[str],
                           with_insiders: bool = False) -> pd.DataFrame:
    """Build/refresh the EDGAR management history. Incremental (officer path)
    and skips a same-day rebuild. Set `with_insiders=True` to also collect the
    Form 4 net-insider-buying series (slow, non-incremental; OFF by default)."""
    years = context.config.data_extract.years_history
    path = context.paths["MANAGEMENT_HISTORY_PATH"]

    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    existing = pd.read_parquet(path) if path.exists() else None
    if _is_up_to_date(context, cik_map):
        context.log.info("EDGAR management already up to date for %s — skipping (%d rows)",
                         today_iso(), 0 if existing is None else len(existing))
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    seen = _seen_accessions(existing)
    last_asof = _last_officer_asof(existing)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_rows_for_ticker, context, r["ticker"], r["cik"],
                      r.get("company_name", ""), years, seen,
                      last_asof.get(r["ticker"]), with_insiders): r["ticker"]
            for _, r in cik_map.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="EDGAR management"):
            rows.extend(fut.result())

    new_df = pd.DataFrame(rows)
    parts = [d for d in (existing, new_df) if d is not None and not d.empty]
    if not parts:
        save_extract_meta(path, None, 0, len(cik_map))
        return existing if existing is not None else pd.DataFrame(columns=["ticker", "as_of"])

    out = pd.concat(parts, ignore_index=True)
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    # merge officer- and insider-dated rows that share a date; keep values sparse
    out = out.groupby(["ticker", "as_of"], as_index=False).first()
    for c in _PLACEHOLDER_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out.sort_values(["ticker", "as_of"]).reset_index(drop=True)
    out.to_parquet(path, index=False)

    last_fd = out["as_of"].max()
    save_extract_meta(path, last_fd.date().isoformat() if pd.notna(last_fd) else None,
                      out["ticker"].nunique(), len(cik_map))
    context.log.info("EDGAR management: %d rows, %d tickers (founder_ceo=%d)",
                     len(out), out["ticker"].nunique(),
                     int(pd.to_numeric(out.get("founder_ceo"), errors="coerce").fillna(0).sum()))
    return out
