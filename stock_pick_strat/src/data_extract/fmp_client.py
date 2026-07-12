"""
Shared Financial Modeling Prep (FMP) client: multi-key rotation + generic
incremental history fetch. Used by every FMP-backed extractor (employee counts,
analyst grades, analyst actions, executive compensation, estimates).

Key rotation
------------
FMP's free tier allows 250 requests/day *per key*. Any number of keys can be put
in .env as `FMP_API_KEY*` (e.g. FMP_API_KEY_alar, FMP_API_KEY_gardon, ...). Keys
are used one at a time; when one is exhausted (HTTP 429/403 or a "Limit Reach"
message) we roll to the next. If all are exhausted we stop early -- progress is
already persisted, so a later run resumes.

Incremental
-----------
Each endpoint returns the FULL history for a ticker in a single call, so there is
no pagination. We only (re)fetch a ticker if it was never pulled or was last
pulled more than `refetch_window_days` ago -- tracked via a `fetched_at` pull
date (NOT the filing date, since annual filings would otherwise look stale
forever and re-burn quota).
"""
from __future__ import annotations

import os
import time
from typing import Callable

import pandas as pd
import requests
from tqdm import tqdm

from src.context import Context

FMP_BASE = "https://financialmodelingprep.com/stable"

_RATE_LIMIT_MARKERS = (
    "limit reach", "reached your", "upgrade your plan", "daily request", "rate limit",
    "payment required", "special endpoint", "exclusive endpoint",
)
# FMP signals an exhausted key / plan limit with 402 (Payment Required), and
# 429/403 for throttling/auth -- all should roll to the next key, not skip.
_RATE_LIMIT_STATUS = (402, 403, 429)


class FMPRateLimitError(Exception):
    """Raised when a key has exhausted its quota -> triggers rotation."""


def collect_api_keys() -> list[tuple[str, str]]:
    """All non-empty `FMP_API_KEY*` env vars as (name, key), sorted by name for a
    deterministic rotation order."""
    keys = []
    for name in sorted(os.environ):
        if name.startswith("FMP_API_KEY"):
            val = (os.environ.get(name) or "").strip()
            if val:
                keys.append((name, val))
    return keys


def is_rate_limited(resp) -> bool:
    """True if the response signals an exhausted/blocked key (quota or auth)."""
    if resp.status_code in _RATE_LIMIT_STATUS:
        return True
    try:
        data = resp.json()
    except Exception:
        return False
    if isinstance(data, dict):
        msg = str(data.get("Error Message", data.get("message", ""))).lower()
        return any(m in msg for m in _RATE_LIMIT_MARKERS)
    return False


def plan_fetch(tickers: list[str], existing: pd.DataFrame | None,
               refetch_window_days: int) -> list[str]:
    """Tickers to fetch: never-pulled ones, plus those last pulled more than
    `refetch_window_days` ago (based on the PULL date, not the filing date)."""
    last_fetch: dict = {}
    if (existing is not None and not existing.empty
            and "fetched_at" in existing.columns):
        s = existing.dropna(subset=["fetched_at"])
        if not s.empty:
            last_fetch = s.groupby("ticker")["fetched_at"].max().to_dict()

    today = pd.Timestamp.today().normalize()
    plan = []
    for t in tickers:
        last = last_fetch.get(t)
        if last is None or (today - pd.Timestamp(last)).days > refetch_window_days:
            plan.append(t)
    return plan


def run_rotating_fetch(plan: list[str], keys: list[str], log,
                       download_one: Callable[[str, str], pd.DataFrame],
                       pause: float = 0.3, desc: str = "FMP history",
                       max_consecutive_dead: int = 3,
                       ) -> tuple[list[pd.DataFrame], bool]:
    """Download each ticker via `download_one(ticker, key)`, rotating keys ONLY on
    a real rate-limit error. Returns (frames, all_keys_exhausted).

    Rotation is conservative -- it does not blindly walk through every key:

    * We stick to the first live key and only advance when it rate-limits.
    * A key is marked *dead* (skipped for the rest of the run) only once a ticker
      that rate-limited on it then **succeeds on a later key** -- that proves the
      ticker itself is available, so the earlier key was genuinely out of quota.
    * If a ticker rate-limits on *every* live key, that's ambiguous: usually a
      premium / unavailable ticker rather than a simultaneous death of all keys.
      We skip that ticker (keeping the keys alive) instead of aborting the run.
    * We stop early (exhausted=True) only when all keys are confirmed dead, or
      when `max_consecutive_dead` tickers in a row fail on every live key (the
      signature of the last key actually running out of quota).
    """
    frames: list[pd.DataFrame] = []
    dead = [False] * len(keys)
    consecutive_full_cascade = 0

    def live_indices() -> list[int]:
        return [i for i in range(len(keys)) if not dead[i]]

    for tkr in tqdm(plan, desc=desc):
        live = live_indices()
        if not live:
            log.warning("All %d FMP keys confirmed exhausted; stopping at %s. "
                        "Progress saved -- re-run later to resume.", len(keys), tkr)
            return frames, True

        df = None
        rl_before_success: list[int] = []
        succeeded = False
        skipped = False
        for i in live:
            try:
                df = download_one(tkr, keys[i])
                succeeded = True
                # Keys that rate-limited on this (now-proven-available) ticker are
                # genuinely out of quota -> retire them for the rest of the run.
                for j in rl_before_success:
                    if not dead[j]:
                        dead[j] = True
                        log.info("FMP key #%d confirmed out of quota (key #%d served "
                                 "%s); retiring it for this run.", j + 1, i + 1, tkr)
                break
            except FMPRateLimitError:
                rl_before_success.append(i)
                continue
            except Exception as e:  # noqa: BLE001 - per-ticker network/parse issue
                log.warning("%s: fetch failed (%s)", tkr, e)
                skipped = True
                break

        if succeeded:
            consecutive_full_cascade = 0
            if df is not None and not df.empty:
                frames.append(df)
        elif skipped:
            consecutive_full_cascade = 0
        else:
            # Rate-limited on every live key: premium/unavailable ticker, or the
            # last key just died. Skip the ticker; only conclude exhaustion if it
            # keeps happening across consecutive tickers.
            consecutive_full_cascade += 1
            log.warning("%s: rate-limited on all %d live key(s); skipping "
                        "(premium/unavailable ticker, or quota running out).",
                        tkr, len(live))
            if consecutive_full_cascade >= max_consecutive_dead:
                log.warning("%d consecutive tickers failed on every live key; "
                            "assuming quota exhausted and stopping. Progress saved.",
                            consecutive_full_cascade)
                return frames, True

        if pause:
            time.sleep(pause)
    return frames, False


def fetch_incremental(
    context: Context,
    tickers: list[str],
    *,
    endpoint: str,
    normalize: Callable[[list, str], pd.DataFrame],
    dedup_keys: list[str],
    path_key: str,
    params: dict | None = None,
    refetch_window_days: int = 30,
    pause: float = 0.3,
    session: requests.Session | None = None,
    desc: str = "FMP history",
) -> pd.DataFrame:
    """Generic incremental FMP history fetch with key rotation.

    `normalize(records, ticker)` turns the endpoint's JSON list into a tidy
    DataFrame; rows are deduplicated on `dedup_keys` (freshly fetched rows win),
    stamped with a `fetched_at` pull date, and persisted to `paths[path_key]`.
    """
    log = context.log
    path = context.paths[path_key]
    existing = pd.read_parquet(path) if path.exists() else None
    if existing is not None and not existing.empty and "fetched_at" in existing.columns:
        existing["fetched_at"] = pd.to_datetime(existing["fetched_at"]).dt.normalize()

    empty = pd.DataFrame(columns=dedup_keys + ["fetched_at"])
    keys_named = collect_api_keys()
    if not keys_named:
        log.warning("No FMP_API_KEY* found in environment. Add e.g. FMP_API_KEY_alar=... "
                    "to .env (free key, 250 req/day). Skipping %s.", endpoint)
        return existing if existing is not None else empty

    keys = [k for _, k in keys_named]
    log.info("[%s] Loaded %d FMP key(s): %s (rotating on daily-limit)",
             endpoint, len(keys), [n for n, _ in keys_named])

    plan = plan_fetch(tickers, existing, refetch_window_days)
    log.info("[%s] %d/%d tickers to fetch (%d already current)",
             endpoint, len(plan), len(tickers), len(tickers) - len(plan))

    own_session = session is None
    session = session or requests.Session()

    def download_one(ticker: str, api_key: str) -> pd.DataFrame:
        p = {"symbol": ticker, **(params or {}), "apikey": api_key}
        resp = session.get(f"{FMP_BASE}/{endpoint}", params=p, timeout=20)
        if is_rate_limited(resp):
            raise FMPRateLimitError()
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):  # non-limit error payload
            raise ValueError(str(data)[:200])
        if not isinstance(data, list):
            raise ValueError(f"unexpected FMP payload type: {type(data).__name__}")
        return normalize(data, ticker)

    try:
        frames, exhausted = run_rotating_fetch(plan, keys, log, download_one, pause, desc)
    finally:
        if own_session:
            session.close()

    today = pd.Timestamp.today().normalize()
    for df in frames:
        df["fetched_at"] = today

    parts = [d for d in ([existing] if existing is not None else []) + frames
             if d is not None and not d.empty]
    if not parts:
        log.warning("[%s] No data available (nothing fetched, no cache).", endpoint)
        return existing if existing is not None else empty

    out = pd.concat(parts, ignore_index=True)
    if "fetched_at" not in out.columns:
        out["fetched_at"] = pd.NaT
    # keep="last" with a stable order index so freshly fetched rows (appended
    # after `existing`) win over cached duplicates on the same dedup keys.
    out["_order"] = range(len(out))
    out = (out.sort_values(dedup_keys + ["_order"])
              .drop_duplicates(subset=dedup_keys, keep="last")
              .drop(columns="_order")
              .reset_index(drop=True))
    out.to_parquet(path, index=False)
    log.info("[%s] Saved %d rows for %d tickers to %s%s",
             endpoint, len(out), out["ticker"].nunique(), path,
             " (stopped early: keys exhausted)" if exhausted else "")
    return out
