"""
analyze_history.py  (src/utils/analyze_history.py)
----------------------------------------------------
Read-only diagnostic tool for auditing `fundamentals_facts`: per (ticker,
field) level-outlier detection (Modified Z-score, MAD-based, Iglewicz &
Hoaglin) plus cross-duration_type SOURCE_TAG MISALIGNMENT -- a filer's ANNUAL
10-K resolving a DIFFERENT underlying XBRL concept than its own quarterly
10-Qs for the SAME logical field. Confirmed root cause of several real bugs
found this session (JPM's negative totalRevenue: FY picked `us-gaap:Revenues`
while quarters picked `us-gaap:RevenuesNetOfInterestExpense`; MAA's
dimensioned-slice Assets/Revenue) -- this tool is the reusable way to surface
that PATTERN across the whole table, not just re-discover one ticker at a time.

Not a fetcher -- purely analyzes whatever is already persisted; never writes
to the DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ordinal position of each fiscal-period label -- mirrors
# `fundamentals_periods.FISCAL_PERIOD_ORDER`, duplicated (not imported) since
# this is a generic `src/utils/` diagnostic, not part of the data_extract
# package, and the mapping is tiny/stable.
_FISCAL_PERIOD_ORDER: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}


def _latest_per_period(sub: pd.DataFrame) -> pd.DataFrame:
    """Collapse to ONE row per (fiscal_year, fiscal_period, duration_type):
    the LATEST-filed value (point-in-time "current best knowledge", same
    convention as `fundamentals_derive._resolve_latest_per_period`) -- an
    amendment coexisting as its own row must not look like a second,
    disagreeing observation of the same period."""
    return (sub.sort_values("filing_date")
           .drop_duplicates(subset=["fiscal_year", "fiscal_period", "duration_type"], keep="last"))


def _chronological_sort(sub: pd.DataFrame) -> pd.DataFrame:
    """Order strictly by (fiscal_year, fiscal_period-ordinal) -- NOT
    filing_date, which is only a proxy and can be out of step with the true
    fiscal sequence (a late amendment, a delayed filing). This is what makes
    the 4-quarter YoY lag and level-outlier comparison meaningful."""
    sub = sub.copy()
    sub["_fp_ord"] = sub["fiscal_period"].map(_FISCAL_PERIOD_ORDER).fillna(99)
    return sub.sort_values(["fiscal_year", "_fp_ord"]).drop(columns="_fp_ord")


def detect_level_outliers(
    df: pd.DataFrame,
    ticker: str,
    field: str,
    *,
    duration_type: str = "quarterly",
    threshold: float = 3.5,
    check_yoy: bool = True,
) -> pd.DataFrame:
    """Modified Z-score (MAD-based) level-outlier + YoY-shift-anomaly
    detection for ONE (ticker, field)'s time series, scoped to a SINGLE
    duration_type (quarterly figures and the annual total are different
    scales -- never mixed in one statistical pass; call twice for both).

    Fixes a false-positive bug found in the original version of this
    function: the YoY check filled the first (up to) 4 periods' undefined
    4-quarter-lag diff with 0 before scoring it, which could itself register
    as a large, spurious "YoY Shift Anomaly" -- those periods are now excluded
    from the YoY mask entirely (there is nothing to compare them against).

    Does NOT flag source_tag mismatches -- a first version of this function
    compared each row's tag against a single GLOBAL mode over the whole
    series, which cannot distinguish a clean, permanent taxonomy transition
    (e.g. `SalesRevenueNet` -> `RevenueFromContractWithCustomerExcludingAssessed
    Tax` post-ASC 606 -- every ticker with enough history has one) from a
    genuine anomaly: it flagged every single pre-transition (or post-
    transition, whichever era has fewer rows) period as a "mismatch", which is
    noise, not signal. `detect_source_tag_misalignment` is the correct,
    era-aware tool for tag-consistency questions (it compares WITHIN one
    fiscal year, where a clean transition's period-end and interim tags still
    agree) -- use that instead.

    Returns one row per PERIOD (already deduplicated to the latest-filed
    value per fiscal_year+fiscal_period via `_latest_per_period`) with
    outlier flags, so it can be filtered/aggregated across many (ticker,
    field) pairs by a caller such as `run_audit`.
    """
    cols = ["ticker", "field", "fiscal_year", "fiscal_period", "duration_type", "filing_date",
           "value", "source_tag", "is_amendment", "derived", "is_level_outlier",
           "level_z_score", "is_yoy_outlier"]
    sub = df.loc[
        (df["ticker"] == ticker) & (df["field"] == field) & (df["duration_type"] == duration_type)
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub = _latest_per_period(sub)
    sub = _chronological_sort(sub).reset_index(drop=True)
    if len(sub) < 3:
        return pd.DataFrame(columns=cols)

    vals = sub["value"].astype(float).values
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    if mad > 0:
        modified_z = 0.6745 * np.abs(vals - median) / mad
    else:
        mean_abs_dev = np.mean(np.abs(vals - median))
        modified_z = (0.6745 * np.abs(vals - median) / mean_abs_dev
                      if mean_abs_dev > 0 else np.zeros_like(vals))
    level_outlier = modified_z > threshold

    yoy_outlier = np.zeros(len(sub), dtype=bool)
    if check_yoy and len(sub) >= 5:
        yoy_change = sub["value"].astype(float).diff(4)
        has_yoy = yoy_change.notna()          # first (up to) 4 periods excluded -- nothing to lag against
        yoy_vals = yoy_change[has_yoy].values
        if len(yoy_vals) >= 3:
            yoy_med = np.median(yoy_vals)
            yoy_mad = np.median(np.abs(yoy_vals - yoy_med))
            if yoy_mad > 0:
                yoy_z = 0.6745 * np.abs(yoy_change - yoy_med) / yoy_mad
                yoy_outlier = (has_yoy & (yoy_z > threshold)).values

    out = sub.copy()
    out["is_level_outlier"] = level_outlier
    out["level_z_score"] = modified_z
    out["is_yoy_outlier"] = yoy_outlier
    return out[cols]


_PERIOD_END_LABELS = ("FY", "Q4")
_INTERIM_LABELS = ("Q1", "Q2", "Q3")


def detect_source_tag_misalignment(df: pd.DataFrame, ticker: str, field: str) -> pd.DataFrame:
    """Per fiscal_year, compare the PERIOD-END row(s) (fiscal_period 'FY'/'Q4'
    -- covers both a FLOW field's separate 'annual' duration_type row and an
    INSTANT field's year-end balance-sheet snapshot, which never has a
    separate 'annual' bucket at all) against the INTERIM quarters' (Q1-Q3)
    source_tags for that SAME year (and the interim quarters against each
    other). By fiscal_period rather than duration_type so this is ONE check
    that works for both flow and instant/balance-sheet fields.

    Flags exactly the pattern the JPM/Agilent-style bugs share: a filer's
    year-end filing resolving a genuinely different XBRL concept than its own
    interim filings for the same field/year -- as opposed to a clean taxonomy
    transition (e.g. `SalesRevenueNet` -> `RevenueFromContractWithCustomer-
    ExcludingAssessedTax` post-ASC 606), which shows up as a single,
    permanent, all-periods-agreeing cutover year and is NOT flagged since
    period-end and interim still agree WITHIN that year.

    Derived rows (source_tag=None by design -- always true of a FLOW field's
    Q4 now that it is unconditionally derived, see `fundamentals_periods.
    decumulate_quarterly_flow`) are excluded, so an instant field's own
    as-reported Q4 snapshot is still compared even though a flow field's
    never is."""
    cols = ["ticker", "field", "fiscal_year", "period_end_source_tag", "interim_source_tags",
           "mismatch_period_end_vs_interim", "mismatch_within_interim"]
    sub = df.loc[(df["ticker"] == ticker) & (df["field"] == field)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    as_reported = sub[sub.get("derived", 0.0) != 1.0]
    as_reported = _latest_per_period(as_reported)

    rows: list[dict] = []
    for fy, grp in as_reported.groupby("fiscal_year"):
        period_end = grp[grp["fiscal_period"].isin(_PERIOD_END_LABELS)]
        interim = grp[grp["fiscal_period"].isin(_INTERIM_LABELS)]
        period_end_tag = period_end["source_tag"].dropna().iloc[0] if not period_end["source_tag"].dropna().empty else None
        interim_tags = sorted(t for t in interim["source_tag"].dropna().unique().tolist())
        mismatch_pvi = bool(period_end_tag) and any(t != period_end_tag for t in interim_tags)
        mismatch_wi = len(interim_tags) > 1
        if mismatch_pvi or mismatch_wi:
            rows.append({
                "ticker": ticker, "field": field, "fiscal_year": fy,
                "period_end_source_tag": period_end_tag, "interim_source_tags": interim_tags,
                "mismatch_period_end_vs_interim": mismatch_pvi,
                "mismatch_within_interim": mismatch_wi,
            })
    return pd.DataFrame(rows, columns=cols)


# The raw, TAG-GRAIN fields these checks can run on -- i.e. present in `fundamentals_facts`
# with a real `source_tag`, as opposed to a column like `ebitda`/`freeCashflow`/`revenue_q`/
# `netIncome_q` that only exists post-hoc inside `fundamentals_history` (computed in
# `_derive_history`, never itself tagged, so there is no `source_tag` for a tag-switch or
# misalignment check to key on). Shared with `fundamentals_audit.py` so the internal and
# external (Tiingo/Yahoo) audits stay in sync on which raw fields are in scope.
DEFAULT_AUDIT_FIELDS = (
    'totalRevenue', 'costOfRevenue', 'sellingGeneralAdmin',
    'operatingIncome', 'pretaxIncome', 'netIncome', 'incomeTaxExpense',
    'interestExpense', 'epsBasic', 'epsDiluted', 'cash',
    'cashInclRestricted', 'shortTermInvestments', 'currentAssets',
    'totalAssets', 'ppeNet', 'goodwill', 'intangiblesGross',
    'currentLiabilities', 'totalLiabilities', 'shortTermDebt',
    'longTermDebt', 'longTermDebtTotal', 'stockholdersEquity',
    'operatingCashFlow', 'capex', 'depAmort', 'stockBasedComp',
    'changeInInventory', 'changeInReceivables', 'changeInPayables',
    'sharesOutstanding', 'basicShares', 'dilutedShares',
    'dividendsPerShare',
    # Added alongside the Tiingo/Yahoo TIINGO_FIELD_MAP/YAHOO_FIELD_MAP extension for
    # downstream-consumed fields -- these four ARE tag-resolved (unlike ebitda/
    # freeCashflow/revenue_q/netIncome_q, see the module-level note above).
    'researchAndDevelopment', 'employees', 'minorityInterest', 'nciIncome',
)

_ALL_DURATION_TYPES = ("quarterly", "annual", "instant")


def analyze_field(df: pd.DataFrame, ticker: str, field: str, threshold: float = 3.5) -> dict[str, pd.DataFrame]:
    """One (ticker, field)'s full diagnostic: level outliers across every
    duration_type this field could possibly use (a field only ever
    populates ONE of quarterly/annual/instant -- checking all three is a
    harmless no-op for the other two, and avoids having to first classify
    the field as flow vs instant), and cross-period_end/interim source_tag
    misalignment."""
    outlier_results = {
        dt: detect_level_outliers(df, ticker, field, duration_type=dt, threshold=threshold,
                                  check_yoy=(dt != "annual"))
        for dt in _ALL_DURATION_TYPES
    }
    return {
        "outliers": {dt: res for dt, res in outlier_results.items()},
        "tag_misalignment": detect_source_tag_misalignment(df, ticker, field),
    }


def run_audit(
    df: pd.DataFrame,
    tickers: list[str],
    fields: list[str],
    threshold: float = 3.5,
) -> dict[str, pd.DataFrame]:
    """Run the full diagnostic across every (ticker, field) pair and
    concatenate into 2 tidy DataFrames (outliers, tag_misalignment) --
    the shape a caller filters/groups/`.to_csv()`s to review hundreds of
    combinations at once instead of one ticker at a time."""
    outlier_frames: list[pd.DataFrame] = []
    tag_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        for field in fields:
            result = analyze_field(df, ticker, field, threshold=threshold)
            for res in result["outliers"].values():
                if not res.empty:
                    outlier_frames.append(res[res["is_level_outlier"] | res["is_yoy_outlier"]])
            if not result["tag_misalignment"].empty:
                tag_frames.append(result["tag_misalignment"])

    outliers = (pd.concat(outlier_frames, ignore_index=True) if outlier_frames
               else pd.DataFrame(columns=["ticker", "field", "fiscal_year", "fiscal_period"]))
    tag_misalignment = (pd.concat(tag_frames, ignore_index=True) if tag_frames
                        else pd.DataFrame(columns=["ticker", "field", "fiscal_year"]))
    return {"outliers": outliers, "tag_misalignment": tag_misalignment}


def _check_table(df, ticker, fiscal_period, field) -> list[str]:
    (df.
        loc[(df['field'] == field)&
            (df['ticker']==ticker)&
            (df['fiscal_period'].isin(fiscal_period))]
        [['filing_date', 'value', 'source_tag']]
        .sort_values('filing_date')
        .set_index('filing_date')
        .plot()
    )

def identify_missing_quarters(facts):
    all_quarters = {'Q1', 'Q2', 'Q3', 'Q4'}
    
    # 1. Filter unique existing records
    df_unique = facts.loc[
        facts['fiscal_period'].isin(all_quarters), 
        ['ticker', 'fiscal_year', 'fiscal_period']
    ].drop_duplicates()

    # 2. Define function to evaluate missing quarters per (ticker, year)
    def get_missing_quarters(series):
        _, fiscal_year = series.name
        # 2026 only expects Q1 and Q2; other years expect Q1-Q4
        expected = {'Q1', 'Q2', 'Q3', 'Q4'}
        if fiscal_year == '2026': ##### no later than today
            expected = {'Q1', 'Q2'}
        if fiscal_year == '2011': #### 15 years extraction so Q1 skipped, might also need to skip Q2
            expected = {'Q2', 'Q3', 'Q4'}
        return sorted(set(expected) - set(series))

    # 3. Apply missing logic
    missing = (
        df_unique.groupby(['ticker', 'fiscal_year'])['fiscal_period']
        .apply(get_missing_quarters)
        .reset_index(name='missing_quarters')
    )

    # 4. Keep only rows where at least one expected quarter is missing
    missing_quarters_df = missing[missing['missing_quarters'].str.len() > 0]

    # per ticker 
    missing_quarters_df = missing_quarters_df.explode('missing_quarters')
    missing_quarters_df['time_q'] = missing_quarters_df['missing_quarters'].astype(str) + '-' + missing_quarters_df['fiscal_year'].astype(str)
    agg = missing_quarters_df.groupby('ticker')['time_q'].apply(set)

    return agg 

def identify_missing_fields(facts, FIELDS):
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    years = list(range(2011, 2026 + 1))
    tickers = facts['ticker'].unique()

    # 1. Generate full Cartesian Product (Years x Quarters x Fields x Tickers)
    multi_idx = pd.MultiIndex.from_product(
        [tickers, years, quarters, FIELDS], 
        names=['ticker', 'fiscal_year', 'fiscal_period', 'field']
    )
    df_grid = pd.DataFrame(index=multi_idx).reset_index()
    df_grid['fiscal_year'] = df_grid['fiscal_year'].astype(str)

    # 2. Filter raw facts for target FIELDS and non-null values
    df_facts = facts.loc[
        facts['field'].isin(FIELDS) & facts['value'].notna(), 
        ['ticker', 'fiscal_year', 'fiscal_period', 'field', 'value']
    ]

    # 3. Merge the Cartesian Grid with extracted facts
    merged = pd.merge(
        df_grid, 
        df_facts, 
        on=['ticker', 'fiscal_year', 'fiscal_period', 'field'], 
        how='left'
    )

    # 4. Count missing quarters per (ticker, field) pair
    # Total expected quarters per field per ticker = len(years) * len(quarters)
    total_quarters = len(years) * len(quarters)
    
    missing_counts = (
        merged['value']
        .isna()
        .groupby([merged['ticker'], merged['field']])
        .sum()
        .reset_index(name='missing_quarter_count')
    )

    # 5. Filter for fields where ALL quarters are missing (missing_count == total_quarters)
    completely_missing = missing_counts[
        missing_counts['missing_quarter_count'] == total_quarters
    ]

    # 6. Group by ticker and aggregate missing fields into a set
    result = (
        completely_missing
        .groupby('ticker')['field']
        .apply(set)
        .reset_index(name='field_missing')
    )

    return result

if __name__ == "__main__":
    from src.context import get_config_context
    from src.utils.fundamentals_tag_ledger import write_tag_ledger

    _, context = get_config_context(config_path="./configs", use_cache=True, save=False)
    facts = context.store.load("fundamentals_facts")
    TICKERS = sorted(facts["ticker"].unique().tolist())
    FIELDS = list(DEFAULT_AUDIT_FIELDS)

    #############################################
    ##################  1. identify missing quarters
    missing_quarters = identify_missing_quarters(facts)
    missing_quarters.to_csv("data/gaps/missing_quarters.csv")

    #############################################
    ##################  2. identify missing variables -> expected or not
    missing_fields = identify_missing_fields(facts, FIELDS)
    missing_fields.to_csv("data/gaps/missing_fields.csv")

    #############################################
    ##################  3. identify outlier values
    audit = run_audit(facts, TICKERS, FIELDS)
    audit["outliers"].to_csv("data/gaps/fundamentals_facts_outliers.csv", index=False)
    audit["tag_misalignment"].to_csv("data/gaps/fundamentals_facts_tag_misalignment.csv", index=False)

    #############################################
    ##################  4. tag-era ledger + cross-year measure splices
    # Complements check 3: `detect_source_tag_misalignment` compares a fiscal year's
    # period-end tag against its own interim quarters and deliberately ignores a clean
    # CROSS-YEAR cutover, which is where a permanent measure change hides.
    write_tag_ledger(context, facts)
   
    # import matplotlib.pyplot as plt 
    # for field in FIELDS:
    #     _check_table(facts, TICKERS[1], ["Q1","Q2","Q3","Q4"], field)
    #     plt.title(field + TICKERS[1])
    #     plt.show()
    