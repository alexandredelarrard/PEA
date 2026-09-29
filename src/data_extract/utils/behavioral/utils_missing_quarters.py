"""
utils_missing_quarters.py  (src/data_extract/utils/behavioral/utils_missing_quarters.py)
---------------------------------------------------------------------------------------
THE single definition of "which earnings-call quarters is each ticker still missing".

Every recent-gap source answers the same question before it spends a request, so the
answer is computed ONCE, here, and handed to each source in priority order:

    missing = missing_quarters_by_ticker(context)      <- one pass over HF + DB + disk + JSON
    rows, filled = fetch_roic_transcripts(context, missing=missing)   <- 1. clean JSON API
    missing = remaining_after(missing, filled)         <- drop what Roic just supplied
    build_transcript_index_by_ticker(context, missing=missing)        <- 2. fool HTML, last resort

Before this was centralised, `fetch_roic_transcripts` called `missing_quarters_by_ticker`
while the fool discovery re-derived the identical thing inline (its own universe load, HF
horizon read, DB read, released-quarter read and gap closure), and `fetch_earnings_calls`
carried a byte-level copy of all eleven helpers. Two consequences: the gap definition
could drift between the two sources, and `hf_latest_quarter_by_ticker` -- which scans the
1.8 GB HuggingFace parquet -- ran twice per pipeline run.

`gap = required - have`, where
  required : from the quarter AFTER the HF backbone's latest for the ticker (or the
             `since` floor when HF has nothing) up to the latest quarter that ticker has
             ACTUALLY REPORTED per `earnings_surprises` (falling back to the calendar
             quarter of today - grace when unknown), and
  have     : quarters already on disk, already in `earnings_call_sections` (any source),
             or already in the fool JSON index.
Names that hold no earnings call at all (NO_EARNINGS_CALL_TICKERS) are always empty.
"""

import re
from pathlib import Path
from typing import cast

import pandas as pd
from bs4 import BeautifulSoup

from src.constants.constants import EARNINGS_CALL_REPORT_GRACE_DAYS, EARNINGS_REPORT_TO_QUARTER_LAG_DAYS, NO_EARNINGS_CALL_TICKERS
from src.context import Context
from src.data_extract.utils.behavioral.fetch_hf_transcripts import hf_latest_quarter_by_ticker
from src.data_extract.utils.behavioral.utils_behavior import _index_path, _load_index
from src.data_extract.utils.behavioral.utils_split_qa import split_prepared_qa
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_store.schema import Tables
from src.utils.text_metrics import assess_earnings_call_sections

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
    return _quarter_index(int(end.year), int(end.quarter))


def _since_floor_index(since: str) -> int:
    """Quarter index of the `since` date floor — the fool gap start for a ticker HF doesn't cover."""
    ts = pd.Timestamp(since)
    return _quarter_index(int(ts.year), int(ts.quarter))


def _local_quarters(cache: Path, ticker: str) -> set[str]:
    """Only locally cached quarters whose parsed high-signal sections pass quality."""
    d = cache / ticker
    if not d.exists():
        return set()
    valid: set[str] = set()
    for path in d.glob("*.html"):
        try:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            div = soup.find("div", class_=lambda c: bool(c and ("transcript-content" in c or "article-body" in c)))
            text = div.get_text("\n", strip=True) if div else ""
            if assess_earnings_call_sections(split_prepared_qa(text)).valid:
                valid.add(path.stem)
        except OSError:
            continue
    return valid


def _db_quarters_by_ticker(context: Context) -> dict[str, set]:
    """{ticker: {quarters}} already in the sections table (ANY source, incl. HF). Empty when the
    table is not created yet -> resume on disk + JSON coverage."""
    db = context.store.load(Tables.earnings_call_sections, columns=["ticker", "quarter", "tag", "text"], optional=True)
    if db is None:
        return {}
    out: dict[str, set] = {}
    for (ticker, quarter), call in db.groupby(["ticker", "quarter"], sort=False):
        sections = dict(zip(call["tag"].astype(str), call["text"], strict=False))
        if assess_earnings_call_sections(sections).valid:
            out.setdefault(str(ticker), set()).add(str(quarter))
    return out


def _released_quarter_idx_by_ticker(context: Context, lag_days: int = EARNINGS_REPORT_TO_QUARTER_LAG_DAYS) -> dict[str, int]:
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
    latest = cast(pd.Series, rep.groupby("ticker")["d"].max())
    out: dict[str, int] = {}
    for tk, dt in latest.items():
        q = dt - pd.Timedelta(days=lag_days)  # shift into the reported quarter
        out[str(tk)] = _quarter_index(int(q.year), int(q.quarter))
    return out


def _missing_for(
    tk: str,
    hf_latest: dict,
    floor_idx: int,
    end_idx: int,
    cache: Path,
    have_db: dict[str, set],
    have_json: dict[str, set],
    released: dict[str, int] | None = None,
) -> set[str]:
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
    tk_end = released.get(tk, end_idx) if released is not None else end_idx  # actual release, per ticker
    required = set(_quarters_between(gap_start, tk_end))
    # An index URL is discovery state, not transcript coverage. Only valid parsed DB/disk
    # content closes the gap; malformed cached/indexed calls are retried.
    have = _local_quarters(cache, tk) | have_db.get(tk, set())
    return required - have


def sort_quarters(quarters) -> list[str]:
    """Quarter labels oldest-first. Malformed labels sort first rather than raising, so a
    stray value can never abort a whole run."""
    return sorted(quarters, key=lambda q: _quarter_index(*(_parse_quarter(q) or (0, 1))))


def remaining_after(missing: dict[str, list[str]], filled: dict[str, set[str]] | None) -> dict[str, list[str]]:
    """`missing` minus whatever an earlier source just supplied -> what the NEXT source
    should attempt. Tickers left with nothing are dropped, so the following source skips
    them entirely instead of spending a request to discover they are complete.

    This is what makes a single up-front gap computation safe to share across sources: the
    hand-off is explicit rather than relying on the next source re-reading the DB."""
    filled = filled or {}
    out: dict[str, list[str]] = {}
    for ticker, quarters in missing.items():
        left = set(quarters) - set(filled.get(ticker, ()))
        if left:
            out[ticker] = sort_quarters(left)
    return out


def missing_quarters_by_ticker(
    context: Context,
    tickers: list[str] | None = None,
    since: str = "2025-01-01",
    grace_days: int = EARNINGS_CALL_REPORT_GRACE_DAYS,
) -> dict[str, list[str]]:
    """{ticker: [missing quarter labels, oldest-first]} — the recent-gap quarters each ticker still
    needs after the HF backbone + whatever is already on disk / in the DB / JSON index. Empty entries
    are dropped.

    THE single source of truth for 'what to fetch', shared by the Roic API layer and the
    Motley Fool discovery. Call it ONCE per run and pass the result down (see the module
    docstring): it reads the HF parquet horizon, the sections table, the transcript cache
    and the fool JSON index, so re-deriving it per source is both slow and a chance for
    the two sources to disagree about what is missing."""
    roster = context.store.load("sp500_tickers", columns=["ticker"])
    assert roster is not None
    universe = list(cast(pd.Series, roster["ticker"]))
    if tickers is not None:
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    cache = cache_dir(context, context.config.local.paths.call_transcripts)
    end_idx = _latest_expected_quarter_index(grace_days)
    floor_idx = _since_floor_index(str(since))
    hf_latest = hf_latest_quarter_by_ticker(context, tickers=universe)
    have_db = _db_quarters_by_ticker(context)
    released = _released_quarter_idx_by_ticker(context)  # latest ACTUALLY-reported quarter per ticker
    index = _load_index(_index_path(context))
    have_json: dict[str, set] = {}
    for r in index.values():
        have_json.setdefault(str(r["ticker"]), set()).add(str(r["quarter"]))

    out: dict[str, list[str]] = {}
    for tk in universe:
        miss = _missing_for(tk, hf_latest, floor_idx, end_idx, cache, have_db, have_json, released)
        if miss:
            out[tk] = sort_quarters(miss)
    return out
