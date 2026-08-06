"""
sources.py  (src/data_aggregate/utils/common/sources.py)
-----------------------------------------------------
Per-source COLUMN PROJECTION for the cube sub-steps: load ONLY the columns the consuming
builder(s) actually read.

This is a memory fix, not a tidiness one. `sec13f_hr` is ~21.7M rows; reading it in full
alongside another tall source is what OOM-killed the aggregation task. A table ABSENT from
this map loads in FULL, which is the right default for the small ones
(`fundamentals_history` ~27k rows, def14a / earnings / dividends all tiny) where a
projection saves nothing, and for `earnings_call_sections` whose `text` column IS the
payload the incremental scoring pass needs.

The projection MUST cover every column its builder requires --
`tests/data_aggregate/test_cube_incremental.py` asserts exactly that, so a projection that
drops a needed column fails there rather than silently emptying a feature.
"""
from __future__ import annotations

SOURCE_COLUMNS: dict[str, list[str]] = {
    # institutional_features + superinvestor_features (the ~21.7M-row table)
    "sec13f_hr": ["cik", "period", "ticker", "shares", "value_usd",
                  "call_value", "put_value", "filing_date"],
    # insider_features
    "insider_transactions": ["ticker", "filing_date", "transaction_code", "value_usd"],
    # short_interest_features: RegSHO short/total volume + reported short interest / ADV
    "short_interest": ["date", "ticker", "short_volume", "total_volume",
                       "short_interest", "avg_daily_volume"],
    "fails_to_deliver": ["date", "ticker", "fails_quantity"],
    # attention_features
    "wiki_pageviews": ["date", "ticker", "pageviews"],
    "google_trends": ["date", "ticker", "search_interest"],
}
