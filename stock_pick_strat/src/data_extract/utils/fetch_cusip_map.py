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

import pandas as pd
import requests

from src.context import Context

_URL = "https://api.openfigi.com/v3/mapping"
_BATCH = 100          # OpenFIGI allows up to 100 jobs per request (no key)


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
    import os
    path = context.paths["CUSIP_MAP_PATH"]
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["cusip", "ticker"])
    known = set(cached["cusip"])
    todo = sorted({c for c in cusips if c and c not in known})
    if not todo:
        return cached

    api_key = os.getenv("OPENFIGI_API_KEY")
    mapped: dict[str, str] = {}
    for i in range(0, len(todo), _BATCH):
        batch = todo[i:i + _BATCH]
        try:
            mapped.update(_parse_openfigi(_openfigi_request(batch, api_key), batch))
        except Exception as e:
            print(f"OpenFIGI batch {i // _BATCH} failed: {e}")
        time.sleep(pause)          # unauthenticated OpenFIGI is ~25 req/min

    new = pd.DataFrame({"cusip": list(mapped), "ticker": list(mapped.values())})
    out = pd.concat([cached, new], ignore_index=True).drop_duplicates("cusip", keep="last")
    out.to_parquet(path, index=False)
    print(f"CUSIP->ticker map: {len(out)} entries ({len(new)} new) -> {path}")
    return out
