"""
fetch_google_trends.py  (src/data_extract/utils/behavioral/fetch_google_trends.py)
----------------------------------------------------------------------------------
Google Trends search-interest per company (a RETAIL-ATTENTION proxy). Saved long
[date, ticker, search_interest] to the `google_trends` DB table.

Why this is not a thin `pytrends` wrapper:
  * ANTI-429. Google throttles the unofficial Trends API by TLS/JA3 fingerprint, so
    rotating User-Agent headers on top of `requests` (which always looks like Python)
    gets blocked within ~20 calls. We instead drive the API with `curl_cffi`, which
    IMPERSONATES a real Chrome TLS handshake — the single biggest lever against the
    fast 429 — plus a primed NID cookie, browser-like headers, jittered pacing,
    periodic session refresh, and exponential backoff on 429 (via call_with_retries).
  * WEEKLY over 15 YEARS. Trends returns weekly buckets only for windows of ~8 months
    to 5 years; a 15-year request silently degrades to MONTHLY. So we fetch overlapping
    ~4-year WEEKLY windows and stitch them into one continuous series, chain-scaling
    adjacent chunks by their overlap (Trends re-normalises each window 0-100
    independently), then rescale the whole series to 0-100.

CAVEATS (research-grade signal): the series is normalised within the requested window
and revised over time, so it is an attention proxy, not a precise point-in-time value;
the attention_features builder only uses within-name relative spikes (the robust part).

Network access is isolated in `_TrendsClient`; the stitching / windowing helpers
(`_weekly_windows`, `_stitch_chunks`, `_scale_to_reference`) are pure and unit-tested.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    GOOGLE_TRENDS_EXPLORE_URL,
    GOOGLE_TRENDS_HOME_URL,
    GOOGLE_TRENDS_MULTILINE_URL,
)
from src.context import Context
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.utils import crawler                    # authorized proxy-pool loader (PEA_SCRAPE_PROXIES / PEA_SCRAPE_PROXY)

try:                                            # TLS-impersonating transport (anti-429)
    from curl_cffi import requests as _cffi_requests
except Exception:                               # pragma: no cover - optional dependency
    _cffi_requests = None

logger = logging.getLogger(__name__)

# Env toggles: verify=0 only for sandboxes behind an SSL-inspecting proxy whose CA is
# absent from the venv trust store; the impersonation target can be pinned if Google
# starts flagging a specific Chrome build.
_VERIFY = os.getenv("TRENDS_VERIFY", "1") != "0"
_IMPERSONATE = os.getenv("TRENDS_IMPERSONATE", "chrome124")
# A ticker with at least this much stitched weekly history is treated as fully
# BACKFILLED (won't be re-backfilled): Google Trends often can't reach the full
# years_history for younger names / low-search keywords, so requiring the history to
# reach the deep_before floor made those tickers re-run the whole backfill EVERY time.
_MIN_BACKFILL_DAYS = 3 * 365

# Anti-429: rotate a realistic desktop User-Agent + full browser-like headers on top of
# the curl_cffi TLS impersonation (belt and suspenders).
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Weekly-resolution ceiling is ~5y; use 4y windows with 1y overlap so adjacent chunks
# share ~1y of weeks to chain-scale on.
_CHUNK_YEARS = 4
_OVERLAP_YEARS = 1


class TrendsRateLimited(RuntimeError):
    """Raised on an HTTP 429 from Google Trends (message carries '429' so
    call_with_retries treats it as rate-limited and backs off)."""


class TrendsError(RuntimeError):
    """Any other non-recoverable Google Trends response (bad status / unparseable)."""


def _random_header() -> dict:
    """Browser-like headers with a rotated User-Agent + Referer (harder to flag)."""
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8"),
        "Referer": "https://trends.google.com/trends/explore",
        "Connection": "keep-alive",
    }


class _TrendsClient:
    """Minimal Google Trends client over a curl_cffi (TLS-impersonating) session.

    Implements the two-step public flow: `explore` returns widget tokens, then
    `widgetdata/multiline` returns the interest-over-time series for the TIMESERIES
    widget's token. A primed NID cookie is required, so the home URL is fetched once
    per session. Raises `TrendsRateLimited` on 429 so callers can back off.
    """

    def __init__(self, verify: bool = True, impersonate: str = "chrome124",
                 timeout: int = 30, proxies: list[str] | None = None) -> None:
        if _cffi_requests is None:
            raise ImportError("curl_cffi is required for Google Trends extraction "
                              "(pip install curl_cffi)")
        self._verify = verify
        self._impersonate = impersonate
        self._timeout = timeout
        # authorized proxy pool (PEA_SCRAPE_PROXIES): rotate to a fresh IP on each refresh/block, so
        # a flagged exit IP is dropped. Empty -> direct (you can't rotate IPs you don't have).
        self._pool = list(proxies) if proxies is not None else crawler.load_proxy_pool()
        random.shuffle(self._pool)
        self._pi = 0
        self._session = None
        self.refresh(rotate=False)

    @property
    def n_proxies(self) -> int:
        return len(self._pool)

    def _current_proxy(self) -> str | None:
        return self._pool[self._pi % len(self._pool)] if self._pool else None

    def refresh(self, rotate: bool = True) -> None:
        """Start a fresh impersonated session and re-prime the NID cookie. When `rotate` (the
        default — this is the `call_with_retries` on_retry hook), MOVE to the next authorized proxy
        first so the retry session goes out on a FRESH IP. Also called periodically so no single
        session fingerprint accumulates throttling."""
        if rotate and self._pool:
            self._pi = (self._pi + 1) % len(self._pool)
        prox = self._current_proxy()
        proxies = {"http": prox, "https": prox} if prox else None
        self._session = _cffi_requests.Session(
            impersonate=self._impersonate, verify=self._verify, timeout=self._timeout,
            proxies=proxies)
        self._session.headers.update(_random_header())
        try:
            self._session.get(GOOGLE_TRENDS_HOME_URL)     # sets NID cookie (on the current IP)
        except Exception as e:                            # noqa: BLE001
            logger.debug("Trends cookie priming failed (continuing): %s", e)

    @staticmethod
    def _decode(text: str) -> dict:
        """Trends responses are JSON prefixed with an anti-JSON-hijack guard
        (")]}',\\n"); strip everything before the first brace and parse."""
        i = text.find("{")
        return json.loads(text[i:]) if i >= 0 else {}

    def _get(self, url: str, params: dict):
        resp = self._session.get(url, params=params) # type: ignore
        if resp.status_code == 429:
            raise TrendsRateLimited(f"429 Too Many Requests from {url}")
        if resp.status_code != 200:
            raise TrendsError(f"HTTP {resp.status_code} from {url}")
        return resp

    def interest_over_time(self, keyword: str, timeframe: str, geo: str = "") -> pd.DataFrame:
        """Return [date, search_interest] (weekly for ~8mo-5y windows). Empty frame
        when the term has no measurable interest for the window."""
        explore = self._get(GOOGLE_TRENDS_EXPLORE_URL, {
            "hl": "en-US", "tz": "0",
            "req": json.dumps({"comparisonItem": [{"keyword": keyword, "geo": geo,
                                                   "time": timeframe}],
                               "category": 0, "property": ""}),
        })
        widgets = self._decode(explore.text).get("widgets", [])
        ts = next((w for w in widgets if str(w.get("id", "")).startswith("TIMESERIES")), None)
        if ts is None:
            return _empty_series()
        data = self._get(GOOGLE_TRENDS_MULTILINE_URL, {
            "hl": "en-US", "tz": "0",
            "req": json.dumps(ts["request"]), "token": ts["token"],
        })
        timeline = self._decode(data.text).get("default", {}).get("timelineData", [])
        rows = [
            (pd.Timestamp(datetime.fromtimestamp(int(p["time"]), tz=timezone.utc)).tz_localize(None).normalize(),
             float(p["value"][0]))
            for p in timeline
            if p.get("value") and not p.get("isPartial")
        ]
        return pd.DataFrame(rows, columns=["date", "search_interest"])


def _empty_series() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "search_interest"])


# --------------------------------------------------------------------------- #
# Pure helpers: windowing, stitching, incremental re-scaling                   #
# --------------------------------------------------------------------------- #
def _weekly_windows(years: int, end: pd.Timestamp | None = None,
                    chunk_years: int = _CHUNK_YEARS,
                    overlap_years: int = _OVERLAP_YEARS) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Overlapping [start, end] windows (each <= `chunk_years`, so Trends returns
    WEEKLY data) covering `years` back from `end`. Adjacent windows overlap by
    `overlap_years` so their series can be chain-scaled onto a common level."""
    end = (end or pd.Timestamp.today()).normalize()
    start0 = end - pd.DateOffset(years=years)
    step = pd.DateOffset(years=chunk_years - overlap_years)
    windows, s = [], start0
    while True:
        e = min(end, s + pd.DateOffset(years=chunk_years))
        windows.append((s, e))
        if e >= end:
            break
        s = s + step
    return windows


def _stitch_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """Chain-scale independently-normalised overlapping weekly chunks into one
    continuous series, then rescale to 0-100.

    Chunks are ordered oldest->newest and anchored on the NEWEST (its scale = 1).
    Each older chunk is multiplied by the ratio of the newer chunk's values to its
    own over their shared weeks, so levels line up across the seams. In overlaps the
    newer chunk's (already-anchored) value wins.
    """
    chunks = [c.dropna(subset=["search_interest"]).sort_values("date")
              for c in chunks if c is not None and not c.empty]
    chunks = [c for c in chunks if not c.empty]
    if not chunks:
        return _empty_series()
    chunks.sort(key=lambda c: c["date"].min())
    n = len(chunks)
    scale = [1.0] * n
    for i in range(n - 2, -1, -1):
        a = chunks[i].set_index("date")["search_interest"]
        b = chunks[i + 1].set_index("date")["search_interest"]
        common = a.index.intersection(b.index)
        if len(common) >= 2:
            va, vb = a.loc[common], b.loc[common]
            mask = (va > 0) & (vb > 0)
            if int(mask.sum()) >= 2 and va[mask].mean() > 0:
                scale[i] = scale[i + 1] * (vb[mask].mean() / va[mask].mean())
                continue
        scale[i] = scale[i + 1]                 # no usable overlap -> carry level
    merged: dict[pd.Timestamp, float] = {}
    for i, c in enumerate(chunks):              # oldest->newest: newer overwrites overlaps
        for d, v in zip(c["date"], c["search_interest"]):
            merged[d] = v * scale[i]
    out = pd.DataFrame(sorted(merged.items()), columns=["date", "search_interest"])
    mx = out["search_interest"].max()
    if mx and mx > 0:
        out["search_interest"] = (out["search_interest"] / mx * 100).round(2).clip(0, 100)
    return out.reset_index(drop=True)


def _scale_to_reference(new: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Scale a freshly-fetched window (`new`, 0-100 within itself) onto the level of
    the already-stored series (`ref`) using their overlapping weeks, so appended
    weeks are continuous with history. No-op if there is too little overlap."""
    if new is None or new.empty or ref is None or ref.empty:
        return new
    a = new.set_index("date")["search_interest"]
    b = ref.set_index("date")["search_interest"]
    common = a.index.intersection(b.index)
    if len(common) >= 2:
        va, vb = a.loc[common], b.loc[common]
        mask = (va > 0) & (vb > 0)
        if int(mask.sum()) >= 2 and va[mask].mean() > 0:
            factor = vb[mask].mean() / va[mask].mean()
            new = new.copy()
            new["search_interest"] = (new["search_interest"] * factor).round(2)
    return new


def _append_and_renormalize(ref: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """Reconcile a freshly-fetched OVERLAPPING window (`recent`) onto the stored per-ticker
    series (`ref`) and return the FULL, coherent series to upsert:
      1. chain-scale `recent` onto ref's level over their shared weeks (`_scale_to_reference`),
      2. append only the weeks NEWER than ref's max (history is otherwise unchanged in shape),
      3. RE-NORMALISE the whole concatenated series to 0-100.
    Step 3 is what makes the rescale "make sense on the full trend": a recent spike above the old
    historical max would otherwise push appended weeks past 100, leaving a patchwork of
    window-local scales. Renormalising keeps ONE 0-100 scale across the entire per-ticker series
    (the peak = 100 wherever in history it falls). Returns `ref` unchanged when there is nothing
    new to append (so a stale-but-quiet ticker is not rewritten)."""
    if ref is None or ref.empty:
        return recent if recent is not None else _empty_series()
    if recent is None or recent.empty:
        return ref
    scaled = _scale_to_reference(recent, ref)
    mx = ref["date"].max()
    new_weeks = scaled[scaled["date"] > mx]
    if new_weeks.empty:
        return ref
    full = (pd.concat([ref[["date", "search_interest"]], new_weeks[["date", "search_interest"]]],
                      ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date").reset_index(drop=True))
    m = full["search_interest"].max()
    if m and m > 0:
        full["search_interest"] = (full["search_interest"] / m * 100).round(2).clip(0, 100)
    return full


def _fetch_weekly_history(client: _TrendsClient, keyword: str, years: int,
                          pause: float) -> pd.DataFrame:
    """Fetch overlapping weekly windows across `years` and stitch them (see module
    docstring). Each window retries with backoff on 429."""
    # with an authorized proxy pool a 429 rotates IP (via on_retry) -> short waits; else stay polite
    bw = 15.0 if client.n_proxies else 45.0
    chunks: list[pd.DataFrame] = []
    for start, end in _weekly_windows(years):
        timeframe = f"{start.date()} {end.date()}"
        try:
            df = call_with_retries(
                lambda tf=timeframe: client.interest_over_time(keyword, tf),
                retries=4, base_wait=bw, on_retry=client.refresh,
                label=f"trends {keyword} {timeframe}")
        except Exception as e:                  # noqa: BLE001 - skip a window, keep the rest
            logger.warning(f"trends {keyword} {timeframe}: window failed ({e})")
            df = None
        if df is not None and not df.empty:
            chunks.append(df)
        time.sleep(pause + random.uniform(1.0, 4.0))
    return _stitch_chunks(chunks)


def _load_existing(context: Context) -> pd.DataFrame | None:
    df = context.store.load("google_trends")
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _clean_name(raw: str) -> str:
    """Company display name -> search keyword: drop parenthetical suffixes/qualifiers."""
    import re
    return re.sub(r"\s*\([^)]*\)", "", str(raw)).strip()


def fetch_google_trends(context: Context, tickers: list[str] | None = None,
                        pause: float = 2.0, refetch_window_days: int = 7,
                        verify: bool | None = None,
                        impersonate: str | None = None) -> pd.DataFrame:
    """Download weekly search interest per company over `years_history` and upsert to
    the `google_trends` DB table, one ticker at a time.

    * WEEKLY / 15y: a ticker with no (or shallow) history gets a full chunked+stitched
      backfill; a ticker already deep and current is skipped; otherwise a fresh overlapping
      window is fetched, the new weeks are levelled onto the stored series and the FULL
      per-ticker trend is re-normalised to a single coherent 0-100 scale
      (`_append_and_renormalize`) before upsert.
    * ANTI-429: curl_cffi TLS impersonation + primed cookie + jitter + periodic session
      refresh + exponential backoff. Skips cleanly if curl_cffi is unavailable.
    """
    years = context.config.data_extract.years_history
    verify = _VERIFY if verify is None else verify
    impersonate = impersonate or _IMPERSONATE

    names = context.store.load("sp500_tickers")
    names["name"] = names["name"].apply(_clean_name)
    if tickers is not None:
        names = names[names["ticker"].isin(tickers)]

    existing = _load_existing(context)
    if existing is None:
        span: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    else:
        agg = existing.groupby("ticker")["date"].agg(["min", "max"])
        span = {t: (r["min"], r["max"]) for t, r in agg.iterrows()}
    
    today = pd.Timestamp.today().normalize()
    deep_before = today - pd.DateOffset(years=years - 1)      # history counts as "deep" if it reaches here

    try:
        client = _TrendsClient(verify=verify, impersonate=impersonate)
    except ImportError as e:
        context.log.warning("Google Trends extraction skipped: %s", e)
        return existing if existing is not None else _empty_long()

    total_new, touched, skipped = 0, 0, 0
    for i, (_, row) in enumerate(tqdm(list(names.iterrows()), desc="Google Trends")):
        tkr, keyword = row["ticker"], str(row["name"])
        mn, mx = span.get(tkr, (None, None))

        # "backfilled" = history reaches the full window OR already spans >= 3y (as deep
        # as Trends will give for this keyword) -> don't re-run the full backfill.
        deep = mn is not None and (mn <= deep_before or (mx - mn).days >= _MIN_BACKFILL_DAYS)
        current = mx is not None and (today - mx).days <= refetch_window_days
        if deep and current:
            skipped += 1
            continue

        try:
            if not deep: # full weekly backfill
                logger.info(f"Redo full history extract for {tkr}")
                series = _fetch_weekly_history(client, keyword, years, pause)
                n_new = len(series)
            else: # deep but stale -> fetch an overlapping recent window, reconcile onto history
                # Explicit [mx - 1y, today] window: < 5y so Trends returns WEEKLY buckets, and it
                # OVERLAPS the stored tail by ~1y so `_scale_to_reference` has real common weeks to
                # level on. (The old "today 1-y" relative timeframe is not a valid Trends unit and
                # errored out on every stale ticker.)
                win_start = (mx - pd.DateOffset(years=1)).normalize()
                timeframe = f"{win_start.date()} {today.date()}"
                logger.info(f"Append recent weeks for {tkr} ({timeframe})")
                recent = call_with_retries(
                    lambda tf=timeframe: client.interest_over_time(keyword, tf),
                    retries=4, base_wait=(15.0 if client.n_proxies else 45.0),
                    on_retry=client.refresh, label=f"trends {tkr} recent")
                ref = existing[existing["ticker"] == tkr][["date", "search_interest"]]
                series = _append_and_renormalize(ref, recent)   # FULL coherent 0-100 series
                n_new = max(0, len(series) - len(ref))          # weeks actually appended
        except Exception as e:                               # noqa: BLE001
            logger.warning("Trends fetch failed for %s (%s): %s", tkr, keyword, e)
            continue

        # save when there is genuinely new data: the full backfill, or the reconciled series with
        # >=1 appended week (upsert on (ticker,date) overwrites the renormalised history in place).
        if series is not None and not series.empty and n_new > 0:
            out = series.copy()
            out["ticker"] = tkr
            context.store.save("google_trends", out[["date", "ticker", "search_interest"]])
            total_new += n_new
            touched += 1

        if (i + 1) % 15 == 0:                                # periodic fresh fingerprint
            client.refresh()
        time.sleep(pause + random.uniform(2.0, 6.0))

    context.log.info("Google Trends: +%d rows across %d tickers (%d already current).",
                     total_new, touched, skipped)
    out = context.store.load("google_trends")
    return out if not out.empty else _empty_long()


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "search_interest"])
