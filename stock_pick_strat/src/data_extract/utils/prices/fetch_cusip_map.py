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

from src.context import Context

_URL = "https://api.openfigi.com/v3/mapping"
_BATCH = 90          # OpenFIGI allows up to 100 jobs per request (no key)


def normalize_cusip(cusip) -> str | None:
    """Canonical CUSIP: uppercased, stripped, and left zero-padded to 9 chars.

    A CUSIP is ALWAYS 9 characters, but filers (and any int-coercing reader) drop
    the leading zero on all-digit CUSIPs -- so the SAME security appears as
    '037833100' and '37833100'. Without a single canonical form, the incremental
    'already mapped?' check (`c not in known`) and the holdings<->ticker merge both
    miss, so the map is rebuilt (and the rate-limited OpenFIGI lookup re-run) every
    time. Returns None for blank / NaN so those are skipped."""
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
    
    cached = context.store.load("cusip_ticker_map")
    if cached.empty:
        cached = pd.DataFrame(columns=["cusip", "ticker"])
    else:
        # normalize the STORED cusips too, so a legacy row saved before this fix
        # (or a differently-zero-padded one) still counts as 'already mapped'.
        cached = (cached.assign(cusip=cached["cusip"].map(normalize_cusip))
                  .dropna(subset=["cusip"]).drop_duplicates("cusip", keep="last"))

    def _mapped_only(df: pd.DataFrame) -> pd.DataFrame:
        """Real mappings only (drop the recorded misses) -> feeds the ticker merge."""
        return df[df["ticker"].notna() & (df["ticker"].astype("string").str.strip() != "")]

    known = set(cached["cusip"])
    # compare on the SAME canonical form on both sides -> the skip actually skips
    todo = sorted({n for c in cusips if (n := normalize_cusip(c)) and n not in known})
    if not todo:
        return _mapped_only(cached)

    api_key = os.getenv("OPENFIGI_API_KEY")
    mapped: dict[str, str] = {}
    attempted: list[str] = []      # cusips whose OpenFIGI batch RESPONDED (a miss is permanent)
    for i in tqdm(range(0, len(todo), _BATCH)):
        batch = todo[i:i + _BATCH]
        try:
            mapped.update(_parse_openfigi(_openfigi_request(batch, api_key), batch))
            attempted.extend(batch)          # responded (map or genuine no-match) -> record it
        except Exception as e:               # network / rate error -> leave for a later run
            print(f"OpenFIGI batch {i // _BATCH} failed: {e}")

        if not api_key:
            time.sleep(pause)                     # unauthenticated OpenFIGI is ~25 req/min
        else:
            time.sleep(pause//3) 
            
    # Persist EVERY responded cusip (mapped -> ticker, no-match -> None). Recording the
    # large UNMAPPABLE tail (bonds / options / warrants / delisted / foreign lines) is
    # what stops the whole rate-limited lookup being re-run every time -> the "takes
    # ages" bug: those cusips never got a ticker, so were never stored, so were re-
    # queried forever. Transient (network) failures are NOT recorded, so they retry.
    new = pd.DataFrame({"cusip": attempted, "ticker": [mapped.get(c) for c in attempted]})
    if not new.empty:
        context.store.save("cusip_ticker_map", new)
    out = pd.concat([cached, new], ignore_index=True).drop_duplicates("cusip", keep="last")
    n_mapped = int(out["ticker"].notna().sum())
    print(f"CUSIP->ticker map: {n_mapped} mapped / {len(out)} attempted "
          f"({len(mapped)} newly mapped of {len(attempted)} attempted) -> DB 'cusip_ticker_map'")
    return _mapped_only(out)      # only real mappings feed the holdings<->ticker merge
