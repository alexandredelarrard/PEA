"""
fetch_cusip_map.py  (src/data_extract/utils/fetch_cusip_map.py)
---------------------------------------------------------------
Map the CUSIPs that appear in 13F filings to tickers via the free OpenFIGI
mapping API (idType=ID_CUSIP -> ticker). Cached to parquet so the (rate-limited)
lookup runs once. OpenFIGI cannot emit CUSIP from a ticker (CUSIP is licensed),
but it accepts a CUSIP as INPUT and returns the ticker -- exactly our direction.

Network is isolated in `_openfigi_request`; the response parser
(`_parse_openfigi`) is pure and unit-tested.
"""
from __future__ import annotations

import time
import os
import pandas as pd
import requests
from tqdm import tqdm
import logging 

from src.data_store.schema import Tables
from src.constants.constants import CUSIP_TICKER_OVERRIDES
from src.context import Context

logger = logging.getLogger(__name__)

_URL = "https://api.openfigi.com/v3/mapping"
_BATCH = 95          # OpenFIGI allows up to 100 jobs per request (no key)


def normalize_cusip(cusip) -> str | None:
    """Canonical CUSIP: uppercased, stripped, and left zero-padded to 9 chars."""
    if cusip is None:
        return None
    s = str(cusip).strip().upper()
    if not s or s in ("NAN", "NONE", "<NA>"):
        return None
    return s.zfill(9)


def _parse_openfigi(results: list[dict], cusips: list[str]) -> dict[str, str]:
    """Align OpenFIGI's per-job results to the input CUSIPs -> {cusip: ticker}.
    Jobs with a warning / no data are skipped. Pure."""
    out: dict[str, str] = {}
    for cusip, job in zip(cusips, results or []):
        data = (job or {}).get("data") or []
        if data and data[0].get("ticker"):
            out[cusip] = str(data[0]["ticker"]).replace("/", "-")
    return out


def _openfigi_request(cusips: list[str], api_key: str | None) -> list[dict]:
    """Network call for one batch, isolated for mocking."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in cusips]
    r = requests.post(_URL, json=jobs, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def build_cusip_ticker_map(context: Context, cusips: list[str],
                           pause: float = 6.0) -> pd.DataFrame:
    """Return + cache a [cusip, ticker] map for the given CUSIPs (deduplicated).
    Reuses the cache and only looks up CUSIPs not already mapped."""

    api_key = os.getenv("OPENFIGI_API_KEY")
    # `optional=True` yields None on a cold table -- the first-ever build's own case, so branch
    # on `is None` (repo convention) rather than assuming a frame.
    cached = context.store.load(Tables.cusip_ticker_map, optional=True)
    cached = (pd.DataFrame(columns=["cusip", "ticker"]) if cached is None else
              cached.assign(cusip=cached["cusip"].map(normalize_cusip))
                    .dropna(subset=["cusip"]).drop_duplicates("cusip", keep="last"))

    # Curated CINS overrides, applied to the cache IMMEDIATELY -- before the `todo` short-circuit
    # below, which returns early when every requested cusip is already known. A miss is cached as
    # a NULL ticker and never retried, so an identifier OpenFIGI cannot resolve stays broken for
    # ever: measured on the live DB, 15,404 letter-prefixed rows mapped to ZERO tickers, hiding
    # ~30 Irish / Bermudan / Swiss / Dutch S&P 500 names from sec13f_hr and the
    # superinvestor sleeve. `keep="last"` puts the override ahead of any cached row for the same
    # identifier. See CUSIP_TICKER_OVERRIDES for how each was recovered from the 13F INFOTABLE.
    overrides = pd.DataFrame({"cusip": [normalize_cusip(c) for c in CUSIP_TICKER_OVERRIDES],
                              "ticker": list(CUSIP_TICKER_OVERRIDES.values())})
    overrides = overrides.dropna(subset=["cusip"])
    if not overrides.empty:
        corrected = overrides.merge(cached, on="cusip", how="left", suffixes=("", "_cached"))
        stale = corrected[corrected["ticker_cached"].isna()]["cusip"].tolist()
        cached = (pd.concat([cached, overrides], ignore_index=True)
                    .drop_duplicates("cusip", keep="last"))
        if stale:
            context.store.save(Tables.cusip_ticker_map, overrides)   # repair the cached misses
            logger.info(f"CUSIP overrides: {len(overrides)} curated identifiers applied "
                        f"({len(stale)} were unmapped in the cache, e.g. {stale[:5]})")

    def _mapped_only(df: pd.DataFrame) -> pd.DataFrame:
        """Real mappings only (drop the recorded misses) -> feeds the ticker merge."""
        return df[df["ticker"].notna() & (df["ticker"].astype("string").str.strip() != "")]

    known = set(cached["cusip"])
    # compare on the SAME canonical form on both sides -> the skip actually skips
    todo = sorted({n for c in cusips if (n := normalize_cusip(c)) and n not in known})
    if not todo:
        return _mapped_only(cached)
    
    mapped: dict[str, str] = {}
    attempted: list[str] = []      # cusips whose OpenFIGI batch RESPONDED (a miss is permanent)
    for i in tqdm(range(0, len(todo), _BATCH)):
        batch = todo[i:i + _BATCH]
        try:
            mapped.update(_parse_openfigi(_openfigi_request(batch, api_key), batch))
            attempted.extend(batch)          # responded (map or genuine no-match) -> record it
        except Exception as e:               # network / rate error -> leave for a later run
            logger.warning(f"OpenFIGI batch {i // _BATCH} failed: {e}")

        if not api_key:
            time.sleep(pause)                     # unauthenticated OpenFIGI is ~25 req/min
        else:
            time.sleep(pause//4.5) 
            
    # Persist EVERY responded cusip (mapped -> ticker, no-match -> None). Recording the
    # large UNMAPPABLE tail (bonds / options / warrants / delisted / foreign lines) is
    # what stops the whole rate-limited lookup being re-run every time -> the "takes
    # ages" bug: those cusips never got a ticker, so were never stored, so were re-
    # queried forever. Transient (network) failures are NOT recorded, so they retry.
    new = pd.DataFrame({"cusip": attempted, "ticker": [mapped.get(c) for c in attempted]})
    context.store.save(Tables.cusip_ticker_map, new)

    # overrides last again, so a fresh OpenFIGI no-match cannot re-bury a curated identifier
    out = (pd.concat([cached, new, overrides], ignore_index=True)
             .drop_duplicates("cusip", keep="last"))
    n_mapped = int(out["ticker"].notna().sum())
    logger.info(f"CUSIP->ticker map: {n_mapped} mapped / {len(out)} attempted "
          f"({len(mapped)} newly mapped of {len(attempted)} attempted) -> DB 'cusip_ticker_map'")
    return _mapped_only(out)      # only real mappings feed the holdings<->ticker merge
