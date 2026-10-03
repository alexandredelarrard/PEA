"""The single definition of which earnings-call quarters each ticker is still missing.

Computed once per run and handed down the sources in priority order (Roic, then Motley Fool), with
`remaining_after` removing what each source supplied. `gap = required - have`: required runs from the quarter
after the HF backbone's latest (or the `since` floor) to the latest quarter the ticker has actually reported
per `earnings_surprises` (else the calendar quarter of today - grace); have is the quality-valid quarters on
disk or in `earnings_call_sections`. Quarters are fiscal labels `YYYYQn`. NO_EARNINGS_CALL_TICKERS are never missing.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd
from bs4 import BeautifulSoup

from src.constants.constants import (
    EARNINGS_CALL_REPORT_GRACE_DAYS,
    EARNINGS_CALL_SCORED_TAGS,
    EARNINGS_REPORT_TO_QUARTER_LAG_DAYS,
    NO_EARNINGS_CALL_TICKERS,
)
from src.context import Context
from src.data_extract.utils.behavioral.fetch_hf_transcripts import hf_latest_quarter_by_ticker
from src.data_extract.utils.behavioral.utils_split_qa import split_prepared_qa
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_store.schema import Tables
from src.utils.text_metrics import assess_earnings_call_sections

# --- quarter arithmetic (a fiscal quarter as a monotone integer index YYYY*4 + (Q-1)) ---
_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

#: Tickers per `earnings_call_sections` read; each read carries the scored sections' full text.
_SECTION_READ_CHUNK = 25


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


def stored_call_quarters(context: Context, tickers: Sequence[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """({ticker: quality-valid quarters}, {ticker: malformed quarters}) of the calls of `tickers`
    stored in `earnings_call_sections`, judged by the shared quality gate on the scored sections.

    Reads only the key, tag and text columns of the scored tags, `_SECTION_READ_CHUNK` tickers at
    a time; the gate needs the text itself.
    """
    valid: dict[str, set[str]] = {}
    malformed: dict[str, set[str]] = {}
    for start in range(0, len(tickers), _SECTION_READ_CHUNK):
        df_sections = context.store.load(
            Tables.earnings_call_sections,
            columns=["ticker", "quarter", "tag", "text"],
            where={"ticker": list(tickers[start : start + _SECTION_READ_CHUNK]), "tag": list(EARNINGS_CALL_SCORED_TAGS)},
            optional=True,
        )
        if df_sections is None:
            continue
        for (ticker, quarter), df_call in df_sections.groupby(["ticker", "quarter"], sort=False):
            sections = dict(zip(df_call["tag"].astype(str), df_call["text"], strict=False))
            target = valid if assess_earnings_call_sections(sections).valid else malformed
            target.setdefault(str(ticker), set()).add(str(quarter))
    return valid, malformed


def _released_quarter_idx_by_ticker(context: Context, lag_days: int = EARNINGS_REPORT_TO_QUARTER_LAG_DAYS) -> dict[str, int]:
    """{ticker: index of the latest quarter it has actually reported}: its latest `earnings_surprises` date
    <= today, shifted back by `lag_days` into the reported quarter. {} when the table is unavailable (callers
    then fall back to the calendar `end_idx`)."""
    try:
        es = context.store.load(Tables.earnings_surprises, columns=["ticker", "earnings_date"])
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
    malformed_db: dict[str, set] | None = None,
    released: dict[str, int] | None = None,
) -> set[str]:
    """The quarters still needed for `tk` per the module's gap rule, plus any malformed stored quarter up to its
    end; {} for NO_EARNINGS_CALL_TICKERS."""
    if tk in NO_EARNINGS_CALL_TICKERS:
        return set()
    hf = hf_latest.get(tk)
    gap_start = (_quarter_index(*hf) + 1) if hf else floor_idx
    tk_end = released.get(tk, end_idx) if released is not None else end_idx  # actual release, per ticker
    required = set(_quarters_between(gap_start, tk_end))
    # A malformed stored call stays recoverable even at or before the HF frontier.
    for quarter in (malformed_db or {}).get(tk, set()):
        parsed = _parse_quarter(quarter)
        if parsed is not None and _quarter_index(*parsed) <= tk_end:
            required.add(quarter)
    # Only valid parsed DB/disk content closes the gap; an index URL is not coverage.
    have = _local_quarters(cache, tk) | have_db.get(tk, set())
    return required - have


def sort_quarters(quarters) -> list[str]:
    """Quarter labels oldest-first. Malformed labels sort first rather than raising, so a
    stray value can never abort a whole run."""
    return sorted(quarters, key=lambda q: _quarter_index(*(_parse_quarter(q) or (0, 1))))


def remaining_after(missing: dict[str, list[str]], filled: dict[str, set[str]] | None) -> dict[str, list[str]]:
    """`missing` minus what an earlier source just supplied -> what the next source should attempt.
    Tickers left with nothing are dropped, so the next source never requests them."""
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
    needs after the HF backbone + whatever is already quality-valid on disk / in the DB. Empty entries
    are dropped.

    The single source of truth for what to fetch, shared by Roic and Motley Fool discovery; call it once
    per run and pass the result down (it scans the HF parquet, the sections table and the transcript cache)."""
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker"])
    assert roster is not None
    universe = list(cast(pd.Series, roster["ticker"]))
    if tickers is not None:
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    cache = cache_dir(context, context.config.local.paths.call_transcripts)
    end_idx = _latest_expected_quarter_index(grace_days)
    floor_idx = _since_floor_index(str(since))
    hf_latest = hf_latest_quarter_by_ticker(context, tickers=universe)
    try:
        have_db, malformed_db = stored_call_quarters(context, universe)
    except KeyError:  # the stored table lacks the declared `tag` / `text` columns
        have_db, malformed_db = {}, {}
    released = _released_quarter_idx_by_ticker(context)  # latest ACTUALLY-reported quarter per ticker
    out: dict[str, list[str]] = {}
    for tk in universe:
        miss = _missing_for(tk, hf_latest, floor_idx, end_idx, cache, have_db, malformed_db, released)
        if miss:
            out[tk] = sort_quarters(miss)
    return out
