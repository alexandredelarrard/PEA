"""
fetch_employees_edgar.py  (src/data_extract/fetch_employees_edgar.py)
---------------------------------------------------------------------
FREE, full-history employee counts from SEC EDGAR 10-K (and optionally 10-Q)
body text -- a drop-in replacement for the FMP historical-employee-count fetch.

Output schema is IDENTICAL to fetch_employees.py, so employee_features.py works
unchanged:

    ticker | as_of (filing date) | period (report date) | employees | form_type

`as_of` is the SEC filing date (point-in-time / leak-free).

SPEED / INCREMENTAL
-------------------
Downloading a full 10-K per filing is the cost, so re-runs must avoid it:
  * Skip entirely when already built today for the full universe (meta sidecar).
  * Otherwise fetch ONLY filings after each ticker's last parsed `as_of` (`D`),
    i.e. the D..today window -- `list_filings(since=...)`. Already-parsed
    accessions are also skipped as a safety net.
  * Per-ticker work runs in a ThreadPoolExecutor; sec_get spaces request starts
    under SEC's 10 req/s limit, so the downloads overlap without breaching it.

Run:
    python -m src.data_extract.fetch_employees_edgar
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.constants.constants import HEADCOUNT_CONTINUITY_MAX, HEADCOUNT_CONTINUITY_MIN
from src.context import Context
from src.data_extract.utils.common.sec_utils import (
    sec_get, load_cik_mapping, load_extract_meta, save_extract_meta, seen_accessions,
    today_iso,
)
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.edgar_extract import html_to_text, extract_employee_count
from src.data_extract.utils.common.incremental import load_existing

_DATA_COLUMNS = ["ticker", "as_of", "period", "employees", "form_type"]
_FORMS = ["10-K"]                     # add "10-Q" for quarterly headcount refreshes
_MAX_WORKERS = 8                      # concurrent tickers (rate-limited in sec_get)


def _last_asof_by_ticker(existing: pd.DataFrame | None) -> dict:
    """Max already-parsed filing date per ticker -> the incremental cutoff `D`.
    Tickers absent here (new to the universe) get their full history fetched."""
    if existing is None or existing.empty:
        return {}
    s = existing[["ticker", "as_of"]].copy()
    s["as_of"] = pd.to_datetime(s["as_of"], errors="coerce")
    s = s.dropna(subset=["as_of"])
    return s.groupby("ticker")["as_of"].max().to_dict()


def _fiscal_state(existing: pd.DataFrame | None) -> dict[str, tuple[pd.Timestamp | None,
                                                                    pd.Timestamp]]:
    """Per ticker: (latest fiscal PERIOD end, latest FILING date) — the two anchors the
    cadence gate needs. `period` may be NaT for a legacy row, hence the Optional."""
    if existing is None or existing.empty:
        return {}
    s = existing[["ticker", "as_of", "period"]].copy()
    s["as_of"] = pd.to_datetime(s["as_of"], errors="coerce")
    s["period"] = pd.to_datetime(s["period"], errors="coerce")
    s = s.dropna(subset=["as_of"])
    out: dict[str, tuple[pd.Timestamp | None, pd.Timestamp]] = {}
    for ticker, g in s.groupby("ticker"):
        periods = g["period"].dropna()
        out[ticker] = (periods.max() if not periods.empty else None, g["as_of"].max())
    return out


def is_10k_due(state: tuple[pd.Timestamp | None, pd.Timestamp] | None,
               today: pd.Timestamp) -> bool:
    """Could this ticker plausibly have a 10-K we have not seen?

    A 10-K arrives ONCE per fiscal year, so on any given day almost no ticker can have a
    new one — but the fetcher used to list filings for all 498 every run, ~493 of those
    EDGAR requests guaranteed to return nothing.

    Anchored on the ticker's own FISCAL YEAR END, not on a gap since its last filing: a gap
    floor is unstable because once a ticker crosses it, it is polled daily until it files
    (~195 days for a February filer). We start looking the day the next fiscal year closes
    (a company cannot report a year before it ends) and keep looking until the filing lands.

    Deliberately NO upper bound: an overdue filer must keep being polled (SMCI once filed
    686 days after its fiscal year end), and being late is the safe direction to err in.
    A ticker with no history at all is always due.
    """
    if state is None:
        return True                                   # never seen -> full pull
    last_period, last_filed = state
    if last_period is None or pd.isna(last_period):   # legacy row without a period
        return (today - last_filed).days >= TENK_FILING_GAP_FALLBACK_DAYS
    next_fye = last_period + pd.DateOffset(years=1)
    return today >= next_fye + pd.Timedelta(days=TENK_WINDOW_OPENS_DAYS_AFTER_FYE)


def _history_by_ticker(existing: pd.DataFrame | None) -> dict[str, list[int]]:
    """Already-accepted headcounts per ticker -> the anchor `_is_continuous` compares a
    newly-parsed value against. Sorted by filing date so the median reflects the series,
    not the row order the DB happened to return."""
    if existing is None or existing.empty or "employees" not in existing.columns:
        return {}
    s = existing[["ticker", "as_of", "employees"]].copy()
    s["as_of"] = pd.to_datetime(s["as_of"], errors="coerce")
    s["employees"] = pd.to_numeric(s["employees"], errors="coerce")
    s = s.dropna(subset=["as_of", "employees"]).sort_values("as_of")
    return {t: g["employees"].astype(int).tolist() for t, g in s.groupby("ticker")}


def _is_up_to_date(context: Context, cik_map: pd.DataFrame) -> bool:
    """True when the history was already built today for the full universe."""
    path = context.paths["EMPLOYEES_HISTORY_PATH"]
    meta = load_extract_meta(path)
    if (meta is None or meta.get("last_built") != today_iso()
            or not context.store.exists("employees_history")):
        return False
    return meta.get("universe_size", 0) >= len(cik_map)


def _is_continuous(count: int, history: list[int]) -> bool:
    """Is `count` continuous with a ticker's own headcount history?

    Headcount is a SLOW-MOVING series: a real company does not multiply or divide its
    workforce by five between two annual filings, so this catches the text-extraction
    misses that survive every in-document heuristic. The 2026-07 audit measured 6.3% of
    year-over-year transitions at >2x or <0.5x, and the verification run caught CSGP
    picking up "2.3 million" (2,300,000) against a stored 1,155.

    Anchored on the MEDIAN of the accepted history, not the previous value: a median
    cannot be dragged by one bad reading, so a single wrong row does not then reject the
    correct ones after it (WRB's 4,502,942 would have done exactly that). A ticker's
    first filing has no anchor and is always accepted.
    """
    if not history:
        return True
    anchor = float(sorted(history)[len(history) // 2])
    if anchor <= 0:
        return True
    return HEADCOUNT_CONTINUITY_MIN <= count / anchor <= HEADCOUNT_CONTINUITY_MAX


def _employee_rows_for_ticker(context: Context, ticker: str, cik: str, company: str,
                              forms: list[str], years: int,
                              since: pd.Timestamp | None, seen: set,
                              history: list[int] | None = None) -> list[dict]:
    """One ticker's NEW employee-count rows (runs in a worker thread)."""
    try:
        filings = list_filings(cik, forms, years, company, since=since)
    except Exception as e:
        context.log.warning("%s: filing list failed (%s)", ticker, e)
        return []

    rows = []
    accepted = list(history or [])
    for _, f in filings.iterrows():
        if f["accession_number"] in seen:
            continue
        try:
            raw = sec_get(f["doc_url"]).text
            count = extract_employee_count(html_to_text(raw))
        except Exception as e:
            context.log.warning("%s %s: text fetch/parse failed (%s)",
                                ticker, f["filing_date"].date(), e)
            continue
        if count is None:
            continue
        if not _is_continuous(count, accepted):
            context.log.warning(
                "%s %s: headcount %d is discontinuous with its own history "
                "(median %d) — dropped as a parse artifact", ticker,
                f["filing_date"].date(), count, sorted(accepted)[len(accepted) // 2])
            continue
        accepted.append(count)
        rows.append({
            "ticker": ticker,
            "as_of": f["filing_date"],
            "period": pd.to_datetime(f.get("period_of_report"), errors="coerce"),
            "employees": count,
            "form_type": f["form"],
            "accession_number": f["accession_number"],
        })
    return rows


def fetch_employees_edgar(context: Context, tickers: list[str],
                          forms: list[str] | None = None, pause: float = 0.0) -> pd.DataFrame:
    """Build/refresh the EDGAR employee-count history. Incremental and skips a
    same-day rebuild. `pause` is accepted for backwards-compat but ignored --
    pacing is handled centrally by the rate limiter in sec_get."""
    forms = forms or _FORMS
    years = context.config.data_extract.years_history
    path = context.paths["EMPLOYEES_HISTORY_PATH"]

    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    existing = load_existing(context, "employees_history", date_col=None)
    if _is_up_to_date(context, cik_map):
        context.log.info("EDGAR employees already up to date for %s — skipping (%d rows)",
                         today_iso(), 0 if existing is None else len(existing))
        return existing if existing is not None else pd.DataFrame(columns=_DATA_COLUMNS)

    seen = seen_accessions(existing)
    last_asof = _last_asof_by_ticker(existing)
    history = _history_by_ticker(existing)

    # CADENCE GATE: skip the tickers whose next 10-K cannot exist yet, so a daily run spends
    # EDGAR requests only on names that could actually have filed (38 of 498 measured).
    fiscal = _fiscal_state(existing)
    today = pd.Timestamp.today().normalize()
    due = cik_map[[is_10k_due(fiscal.get(t), today) for t in cik_map["ticker"]]]
    context.log.info(
        "EDGAR employees: %d/%d tickers could have a new 10-K (annual cadence gate); "
        "skipping %d whose fiscal year has not closed since their last filing",
        len(due), len(cik_map), len(cik_map) - len(due))
    if due.empty:
        save_extract_meta(path, None, 0 if existing is None else len(existing), len(cik_map))
        return existing if existing is not None else pd.DataFrame(columns=_DATA_COLUMNS)
    cik_map = due

    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_employee_rows_for_ticker, context, r["ticker"], r["cik"],
                      r.get("company_name", ""), forms, years,
                      last_asof.get(r["ticker"]), seen,
                      history.get(r["ticker"], [])): r["ticker"]
            for _, r in cik_map.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="EDGAR employee counts"):
            new_rows.extend(fut.result())

    new_df = pd.DataFrame(new_rows)
    parts = [d for d in (existing, new_df) if d is not None and not d.empty]
    if not parts:
        save_extract_meta(path, None, 0, len(cik_map))
        return pd.DataFrame(columns=_DATA_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["employees"] = pd.to_numeric(out["employees"], errors="coerce")
    out = out.dropna(subset=["as_of", "employees"])
    out = out[out["employees"] > 0]
    out = (out.sort_values(["ticker", "as_of"])
              .drop_duplicates(subset=["ticker", "as_of"], keep="last")
              .reset_index(drop=True))
    context.store.save("employees_history", out)   # small table — full upsert

    last_fd = out["as_of"].max()
    save_extract_meta(path, last_fd.date().isoformat() if pd.notna(last_fd) else None,
                      out["ticker"].nunique(), len(cik_map))
    context.log.info("EDGAR employees: %d rows, %d tickers (%d new filings parsed)",
                     len(out), out["ticker"].nunique(), len(new_df))
    # feature builder only needs _DATA_COLUMNS; accession kept for incremental dedup
    return out
