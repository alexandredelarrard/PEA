"""
client.py  (src/data_extract/utils/fundamentals_sharadar/client.py)
-------------------------------------------------------------------
HTTP layer for the Sharadar DIRECT API (`api.sharadar.com`): one paged request per
(table, filter set), with response-header validation and 403-as-ENTITLEMENT classification.

WHY NOT `nasdaqdatalink` / `quandl`: those libraries speak to `data.nasdaq.com`, which names
the filing-date column `datekey` and does not ship `fiscalperiod`. That column name sits
inside `fundamentals_sharadar`'s primary key, so the two channels are not interchangeable
(decisions D1/D2). Nothing here is a new dependency -- the transport is `src/utils/polite_http`.

Two traps this module exists to close:

1. **403 is a routine answer, not a failure.** Every ticker outside the subscription returns
   `403 {"error":"Exceeds free tier"}`. `polite_http.http_get` treats 403 as rate-limiting and
   retries 4 times with exponential backoff, so a loop over the S&P 500 would spend minutes
   per non-entitled ticker achieving nothing. We classify the status ourselves off a single
   no-retry GET and reserve the retry path for statuses that might actually change.

2. **`fields=` silently drops an unavailable field** rather than erroring, so a typo yields a
   missing column and no warning. Every response header is validated against the expected
   contract; a missing column raises.
"""
from __future__ import annotations

import io
import os

import pandas as pd

from src.constants.constants import SHARADAR_BASE_URL
from src.context import Context
from src.utils.polite_http import get_once, http_get

# Rows per request. Measured 2026-08-26: the API honours a `limit` above this (20000 returned
# 17,826 rows in one call), so paging is belt-and-braces rather than strictly required -- but
# an undocumented server-side cap is exactly the kind of silent truncation that costs a
# re-extraction to discover, and `offset` demonstrably works.
_PAGE_LIMIT = 10_000
_TIMEOUT = 60
SHARADAR_API_KEY_ENV = "SHARADAR_API_KEY"

# The 7 NON-NUMERIC columns. EVERYTHING ELSE IN SF1 IS A VALUE COLUMN AND MUST BE CAST TO
# float64 BEFORE THE FIRST WRITE -- `ensure_table` infers SQL types from the FIRST frame it
# sees, so a column the first ticker never populates lands as an all-None object column,
# becomes TEXT, and every later ticker's real number is then stored as a string. Measured
# live on `minorityInterest` / `restrictedCash`: VRT created them TEXT and APA's values came
# back as '1997000000.0'.
SHARADAR_ID_COLUMNS = ("ticker", "dimension", "calendardate", "date", "reportperiod",
                       "fiscalperiod", "lastupdated")

class NotEntitled(RuntimeError):
    """HTTP 403 -- the subscription does not cover this ticker/table.

    An exception rather than a `None` return because `None` already means "no data / the
    request failed", and a caller that cannot tell those apart cannot report the entitlement
    summary the run is required to end with.
    """


def _api_key() -> str:
    """The Sharadar key from the environment.

    Reads ONLY the correctly-spelled variable. There is deliberately no fallback to the
    historical misspelling `SHARDAR_API_KEY`: a silent fallback is how a typo survives forever.
    """
    key = os.getenv(SHARADAR_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{SHARADAR_API_KEY_ENV} is not set. Add it to .env (note the spelling: "
            f"SHAR-A-DAR, not SHARDAR) -- the Sharadar fetchers cannot run without it.")
    return key


def _parse_csv(text: str, *, keep_default_na: bool) -> pd.DataFrame:
    """One CSV response body -> DataFrame. Pure.

    `keep_default_na=False` is REQUIRED for `actions` / `sp500`, where the literal string
    "N/A" is a real value of `contraticker` and `contraticker` is a primary-key member. The
    default NA list contains "N/A", so the default would turn a PK value into NULL.
    """
    if not text or not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), keep_default_na=keep_default_na)


def _validate_header(context: Context, table: str, df: pd.DataFrame,
                     expect_columns: tuple[str, ...]) -> None:
    """Raise unless the response header IS the expected contract, both ways."""
    got = tuple(df.columns)
    if got == tuple(expect_columns):
        return
    missing = [c for c in expect_columns if c not in got]
    extra = [c for c in got if c not in expect_columns]
    if missing or extra:
        raise RuntimeError(
            f"Sharadar {table}: response header disagrees with the expected contract "
            f"({len(got)} columns received, {len(expect_columns)} expected). "
            f"Missing: {missing or 'none'}. Unexpected: {extra or 'none'}. "
            f"`fields=` drops an unavailable field silently, so this is a real change in "
            f"the feed, not a transient error.")
    # Same set, different order: harmless, but say so rather than hide it.
    context.log.warning("Sharadar %s: column ORDER changed vs the stored contract "
                        "(same %d columns); reindexing to the contract.", table, len(got))


def _page(context: Context, url: str, params: dict) -> str | None:
    """One page. A single no-retry GET, so a 403 costs ONE request; only a status that could
    plausibly change on a second attempt falls through to the retrying path."""
    resp = get_once(url, params=params, timeout=_TIMEOUT)
    code = getattr(resp, "status_code", None) if resp is not None else None
    if code == 200:
        return resp.text
    if code == 403:
        raise NotEntitled(str(params.get("ticker") or url))
    # Transport error, 5xx, 429, 404 -- worth exactly one retrying call.
    context.log.debug("Sharadar %s -> %s; falling back to the retrying GET", url, code)
    retried = http_get(url, params=params, timeout=_TIMEOUT, retries=3)
    return retried.text if retried is not None else None


def sharadar_get(context: Context, table: str, /, *,
                 expect_columns: tuple[str, ...] | None = None,
                 keep_default_na: bool = True,
                 **filters) -> pd.DataFrame | None:
    """`GET {SHARADAR_BASE_URL}/data/{table}` with `filters`, paged, as a DataFrame.

    `None` means the request failed; an EMPTY frame means the filters matched no rows.
    `NotEntitled` is raised on 403.

    The caller MUST pass an explicit `date.gte` for any table with a date column: the API
    defaults `from` to "1 year ago" and `sort` to `date.desc`, so omitting either silently
    truncates history to the last year. Dotted filter names go in as
    `sharadar_get(..., **{"date.gte": "2021-01-01"})`.

    `table` is POSITIONAL-ONLY (the `/`) because the `tickers` endpoint has a FILTER of its
    own called `table` -- `sharadar_get(ctx, "tickers", **{"table": "fundamentals"})` is a
    legitimate call, and without the `/` it would raise "got multiple values for argument".
    """
    url = f"{SHARADAR_BASE_URL}/data/{table}"
    key = _api_key()
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {"api_key": key, "limit": _PAGE_LIMIT, "offset": offset, **filters}
        text = _page(context, url, params)
        if text is None:
            return pd.concat(frames, ignore_index=True) if frames else None
        page = _parse_csv(text, keep_default_na=keep_default_na)
        if page.empty:
            break
        if expect_columns is not None:
            _validate_header(context, table, page, expect_columns)
            page = page.reindex(columns=list(expect_columns))
        frames.append(page)
        if len(page) < _PAGE_LIMIT:
            break
        offset += _PAGE_LIMIT
    if not frames:
        return pd.DataFrame(columns=list(expect_columns) if expect_columns else None)
    return pd.concat(frames, ignore_index=True)


def cast_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hard-cast every SF1 column that is NOT an identifier to `float64`.

    THIS IS NOT OPTIONAL AND IT MUST HAPPEN BEFORE THE FIRST WRITE. `store.ensure_table`
    infers SQL types from the FIRST DataFrame it ever sees for a table, and `ddl.sql_type`
    falls through to TEXT for an object dtype. A column the first ticker never populates
    arrives as an all-None object column, becomes a TEXT column, and every later ticker's
    real number is then stored as a string -- measured live on `minorityInterest` /
    `restrictedCash`, where VRT created them TEXT and APA's values came back as
    `'1997000000.0'`. Casting here makes the column DOUBLE PRECISION on creation whether or
    not the first ticker happens to report it.
    """
    out = df.copy()
    for col in out.columns:
        if col in SHARADAR_ID_COLUMNS:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def coerce_date_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """`columns` -> datetime64, blanks -> NaT.

    Needed because `keep_default_na=False` (see `_parse_csv`) leaves an absent date as the
    empty string, which Postgres rejects for a DATE column.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out
