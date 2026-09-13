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

BUT it must also tolerate a column the builder treats as OPTIONAL and the live table does not
have, because `DataStore.read_table` resolves columns via `tbl.c[name]` and raises `KeyError`
for an absent one -- so an over-declared projection kills the read instead of degrading the
feature. Use `project_existing` rather than indexing `SOURCE_COLUMNS` directly.

`sec_short_interest` was the case that bit: the projection listed `short_interest` /
`avg_daily_volume` unconditionally for a `ic_shortvol_days_to_cover` feature, the live table
has only `date, ticker, short_volume, total_volume`, and the whole family died. That feature is
now gone for a better reason than the missing columns -- days-to-cover needs FINRA's twice
-monthly short-INTEREST positions, which this repo does not fetch, and short VOLUME cannot
approximate a position (see `short_flow_features`). The projection no longer asks for them.

⚠ THE KEYS ARE PHYSICAL TABLE NAMES (`Table.name`), NOT REGISTRY ATTRIBUTE NAMES. Five
registry entries differ between the two -- `Tables.short_interest` is the table
`sec_short_interest`, and `dividends`, `sharadar_fundamentals`, `def14a_edgar`,
`filing_risk_text` are the others. This map was keyed on `"short_interest"`, which matches no
table: `store.exists` returned False, `_load_source` returned None, and the three
`ic_shortvol_*` features were absent from the cube while 956,640 rows sat in the table. Call
sites pass `table.name`, so a future entry must be keyed the same way.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SOURCE_COLUMNS: dict[str, list[str]] = {
    # institutional_features + superinvestor_features (the ~21.7M-row table)
    "sec13f_hr": ["cik", "period", "ticker", "shares", "value_usd",
                  "call_value", "put_value", "filing_date"],
    # insider_features + insider_quality. Wider than it looks, and every column earns it:
    # `security_type`/`security_title` scope the read to common stock (a preferred row at par
    # put BAC's reference price at $57.80), `price_per_share` + `shares` are what the
    # consensus screen repairs `value_usd` from, `accession_number`/`transaction_date` link an
    # exercise-and-sell package, and `shares_owned_after`/`direct_indirect` are the two legs
    # of #36. Dropping any of them does not degrade a feature, it deletes it.
    "insider_transactions": ["accession_number", "ticker", "owner_cik", "owner_name",
                             "filing_date", "transaction_date", "transaction_code", "shares",
                             "price_per_share", "value_usd", "shares_owned_after",
                             "security_type", "security_title", "direct_indirect",
                             "officer_title", "is_director", "is_officer",
                             "is_ten_pct_owner", "is_10b5_1"],
    # short_flow_features: RegSHO short/total volume, and nothing else -- the table has only
    # these four columns and short VOLUME is all it reports.
    # Keyed on the PHYSICAL name -- `Tables.short_interest.name` is `sec_short_interest`.
    "sec_short_interest": ["date", "ticker", "short_volume", "total_volume"],
    "sec_fails_to_deliver": ["date", "ticker", "fails_quantity"],
    # ownership_features: canonical-event construction (max per accession+cusip, never sum)
    # needs the group-membership + ownership-number columns; `is_amendment` and
    # `item4_purpose_of_transaction` are 13D-only concepts (13G has neither).
    "sec_13d": ["ticker", "accession_number", "cusip", "filing_date", "is_amendment",
               "percent_of_class", "reporting_person_cik", "reporting_person_name",
               "item4_purpose_of_transaction"],
    "sec_13g": ["ticker", "accession_number", "cusip", "filing_date",
               "percent_of_class", "reporting_person_cik", "reporting_person_name"],
    # NOTE `sec_13d_transactions` is deliberately ABSENT, same reasoning as `wiki_pageviews`
    # below: the registry marks it "not used" for this feature set (18 tickers total) and a
    # projection with no reader is not free. Add it back if a future feature reads it.
    # NOTE `wiki_pageviews` / `google_trends` are deliberately ABSENT. Their only consumer
    # was the attention panel, which is deleted; both tables are still EXTRACTED and still
    # sit in the DB, so a future consumer adds its projection back here. A projection with
    # no reader is not free -- it is a claim that some builder needs those columns.
    # ---- the two PER-PERSON DEF 14A children (StepCubeGovernance) ----
    # 134,490 + 80,252 rows is a real read beside the 21.7M-row 13F table in the same build,
    # and both are WIDE with columns the governance builders never touch (`cik`, `gender_basis`,
    # `reconciles`, `fiscal_year`). Projected here rather than inlined in the step so
    # `test_cube_incremental` can assert the list covers what the builders require.
    #
    # ⚠ `def14a_directors` needs `tenure_years` for `pct_long_tenured` and
    # `board_tenure_dispersion` (D40) -- the six-column list that omits it predates the
    # board-quality family. `gender` is deliberately absent: `pct_female_directors` is
    # D3-protected and stays on the parent scalar (D36).
    "def14a_directors": ["ticker", "accession_number", "as_of", "name",
                         "age", "tenure_years", "is_independent",
                         "other_public_company_boards"],
    # ⚠ SIX components, and the first is `fees_earned` -- Item 402(k) has no `salary` and no
    # `bonus` line. `impute_director_comp` sums exactly these into a NULL `total`.
    "def14a_director_comp": ["ticker", "accession_number", "as_of", "name", "total",
                             "fees_earned", "stock_awards", "option_awards",
                             "non_equity_incentive", "pension_change", "other_compensation"],
}

# Columns a builder uses only IF present, so projecting them must not hard-fail when the live
# table predates them. Each entry is `table -> the optional columns of its projection`.
OPTIONAL_SOURCE_COLUMNS: dict[str, frozenset[str]] = {
    # institutional_features zero-fills the option legs when they are absent
    "sec13f_hr": frozenset({"call_value", "put_value", "filing_date"}),
}


def project_existing(available: list[str] | None, table: str) -> list[str] | None:
    """The projection for `table`, narrowed to the columns that actually EXIST.

    `available` is the table's real column list (None -> unknown, so project nothing and let
    the caller read in full). A REQUIRED column that is missing is still an error worth
    surfacing, so it is logged loudly; an OPTIONAL one is dropped quietly, matching the
    builder's own `issubset` guard.
    """
    wanted = SOURCE_COLUMNS.get(table)
    if wanted is None or available is None:
        return wanted
    have = set(available)
    keep = [c for c in wanted if c in have]
    missing = [c for c in wanted if c not in have]
    optional = OPTIONAL_SOURCE_COLUMNS.get(table, frozenset())
    required_missing = [c for c in missing if c not in optional]
    if required_missing:
        logger.warning("%s is missing REQUIRED column(s) %s -> the features that need them "
                       "will be empty", table, required_missing)
    elif missing:
        logger.info("%s has no %s (optional) -> those features are skipped", table, missing)
    return keep or None
