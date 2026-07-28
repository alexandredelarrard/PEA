
import pandas as pd 
import re 
from pathlib import Path

from src.context import Context
from src.data_extract.utils.behavioral.utils_behavior import (
    _cache_dir, 
    _index_path,
    _load_index)
from src.constants.constants import (
    EARNINGS_CALL_REPORT_GRACE_DAYS,
    EARNINGS_REPORT_TO_QUARTER_LAG_DAYS,
    NO_EARNINGS_CALL_TICKERS,
    EARNINGS_CALL_SECTIONS_TABLE)

# --- quarter arithmetic (a fiscal quarter as a monotone integer index YYYY*4 + (Q-1)) ---
_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")


def _parse_quarter(q: str) -> tuple[int, int] | None:
    """'2025Q1' -> (2025, 1); None if malformed."""
    m = _QUARTER_RE.match(str(q))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _quarter_index(year: int, quarter: int) -> int:
    """Monotone quarter index so consecutive quarters differ by 1 (2024Q4 -> 2025Q1)."""
    return year * 4 + (quarter - 1)


def _index_to_quarter(idx: int) -> str:
    return f"{idx // 4}Q{idx % 4 + 1}"


def _quarters_between(start_idx: int, end_idx: int) -> list[str]:
    """Every quarter label from `start_idx` to `end_idx` inclusive (empty if start > end)."""
    return [_index_to_quarter(i) for i in range(start_idx, end_idx + 1)]


def _latest_expected_quarter_index(grace_days: int = EARNINGS_CALL_REPORT_GRACE_DAYS) -> int:
    """Quarter index of the newest call we EXPECT to exist today: the calendar quarter of
    (today - grace), so a just-ended quarter that has not been reported yet is not required."""
    end = pd.Timestamp.today() - pd.Timedelta(days=grace_days)
    return _quarter_index(end.year, end.quarter)


def _since_floor_index(since: str) -> int:
    """Quarter index of the `since` date floor — the fool gap start for a ticker HF doesn't cover."""
    ts = pd.Timestamp(since)
    return _quarter_index(ts.year, ts.quarter)


def _local_quarters(cache_dir: Path, ticker: str) -> set[str]:
    """Quarters ALREADY downloaded to disk for a ticker = the {quarter}.html files under
    data/call_transcripts/{ticker}/ (so a re-run never re-requests a cached transcript)."""
    d = cache_dir / ticker
    return {p.stem for p in d.glob("*.html")} if d.exists() else set()


def _db_quarters_by_ticker(context: Context) -> dict[str, set]:
    """{ticker: {quarters}} already in the sections table (ANY source, incl. HF). Empty when
    the DB is unavailable / the table is not created yet -> resume on disk + JSON coverage."""
    try:
        db = context.store.load(EARNINGS_CALL_SECTIONS_TABLE, columns=["ticker", "quarter"])
        if db is None or db.empty:
            return {}
        out: dict[str, set] = {}
        for tk, q in zip(db["ticker"], db["quarter"]):
            out.setdefault(str(tk), set()).add(str(q))
        return out
    except Exception:
        return {}


def _released_quarter_idx_by_ticker(
    context: Context, lag_days: int = EARNINGS_REPORT_TO_QUARTER_LAG_DAYS) -> dict[str, int]:
    """{ticker: index of the latest quarter it has ACTUALLY REPORTED}, from `earnings_surprises`
    (which carries the earnings report date per ticker). We take the most recent earnings_date that
    is <= today and map it back into the quarter it reported (shift by `lag_days`, since a report
    lands a few weeks after quarter-end). This replaces the blanket calendar guess with the real
    per-ticker release, so the gap logic never demands a quarter a ticker hasn't reported yet — and
    picks up an early reporter the calendar heuristic would miss. {} when the table is unavailable
    (callers then fall back to the calendar `end_idx`)."""
    try:
        es = context.store.load("earnings_surprises", columns=["ticker", "earnings_date"])
    except Exception:
        return {}
    if es is None or es.empty or not {"ticker", "earnings_date"}.issubset(es.columns):
        return {}
    d = pd.to_datetime(es["earnings_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    m = d.notna() & (d <= today)
    if not m.any():
        return {}
    rep = pd.DataFrame({"ticker": es["ticker"].astype(str)[m], "d": d[m]})
    latest = rep.groupby("ticker")["d"].max()
    out: dict[str, int] = {}
    for tk, dt in latest.items():
        q = (dt - pd.Timedelta(days=lag_days))            # shift into the reported quarter
        out[tk] = _quarter_index(q.year, q.quarter)
    return out


def _missing_for(tk: str, hf_latest: dict, floor_idx: int, end_idx: int, cache_dir: Path,
                 have_db: dict[str, set], have_json: dict[str, set],
                 released: dict[str, int] | None = None) -> set[str]:
    """The quarters still needed for `tk`: everything from the fool gap-start (the quarter AFTER the
    HF backbone's latest for `tk`, or the `since` floor when HF has none) up to the latest quarter
    the ticker has ACTUALLY REPORTED (`released[tk]` from earnings_surprises; falls back to the
    calendar `end_idx` when unknown), MINUS what's already on disk / in the DB / in the JSON index.
    Tickers that hold no earnings call (NO_EARNINGS_CALL_TICKERS, e.g. Berkshire) return {} so they
    are never fetched or flagged as missing. Shared by the MF quote-page discovery AND the Roic
    fallback so the 'what's missing' definition can't drift."""
    if tk in NO_EARNINGS_CALL_TICKERS:
        return set()
    hf = hf_latest.get(tk)
    gap_start = (_quarter_index(*hf) + 1) if hf else floor_idx
    tk_end = released.get(tk, end_idx) if released is not None else end_idx   # actual release, per ticker
    required = set(_quarters_between(gap_start, tk_end))
    have = _local_quarters(cache_dir, tk) | have_db.get(tk, set()) | have_json.get(tk, set())
    return required - have



def missing_quarters_by_ticker(context: Context, tickers: list[str] | None = None,
                               since: str = "2025-01-01",
                               grace_days: int = EARNINGS_CALL_REPORT_GRACE_DAYS,
                               ) -> dict[str, list[str]]:
    """{ticker: [missing quarter labels, oldest-first]} — the recent-gap quarters each ticker still
    needs after the HF backbone + whatever is already on disk / in the DB / JSON index. Empty entries
    are dropped. This is the SINGLE source of truth for 'what to fetch' shared by the Roic fallback
    and the Motley Fool discovery."""
    from src.data_extract.utils.behavioral.fetch_hf_transcripts import hf_latest_quarter_by_ticker

    universe = list(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    if tickers is not None:
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    cache_dir = _cache_dir(context)
    end_idx = _latest_expected_quarter_index(grace_days)
    floor_idx = _since_floor_index(str(since))
    hf_latest = hf_latest_quarter_by_ticker(context, tickers=universe)
    have_db = _db_quarters_by_ticker(context)
    released = _released_quarter_idx_by_ticker(context)   # latest ACTUALLY-reported quarter per ticker
    index = _load_index(_index_path(context))
    have_json: dict[str, set] = {}
    for r in index.values():
        have_json.setdefault(str(r["ticker"]), set()).add(str(r["quarter"]))

    out: dict[str, list[str]] = {}
    for tk in universe:
        miss = _missing_for(tk, hf_latest, floor_idx, end_idx, cache_dir, have_db, have_json, released)
        if miss:
            out[tk] = sorted(miss, key=lambda q: _quarter_index(*_parse_quarter(q)))
    return out