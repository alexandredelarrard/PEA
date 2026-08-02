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


if __name__ == "__main__":
    from src.context import get_config_context

    _, context = get_config_context(config_path="./configs", use_cache=True, save=False)
    facts = context.store.load("fundamentals_facts")

    TICKERS = sorted(facts["ticker"].unique().tolist())
    # "totalIncome" isn't a literal field name in fundamentals_tags.py (no
    # exact match) -- interpreted as `pretaxIncome` (income before tax), the
    # one income-statement concept between totalRevenue and netIncome not
    # otherwise in this list; flag/replace if a different field was meant.
    FIELDS = ["totalRevenue", "pretaxIncome", "netIncome", "depAmort", "operatingIncome",
             "costOfRevenue", "sellingGeneralAdmin", "totalAssets", "totalLiabilities", "cash",
             "epsDiluted"]

    import matplotlib.pyplot as plt 
    for field in FIELDS:
        _check_table(facts, TICKERS[5], ["Q1","Q2","Q3","Q4"], field)
        plt.title(field + TICKERS[5])
        plt.show()

    audit = run_audit(facts, TICKERS, FIELDS)
    audit["outliers"].to_csv("data/output/diagnostics/fundamentals_facts_outliers.csv", index=False)
    audit["tag_misalignment"].to_csv("data/output/diagnostics/fundamentals_facts_tag_misalignment.csv", index=False)
    print(f"outliers: {len(audit['outliers'])} rows across "
         f"{audit['outliers'][['ticker', 'field']].drop_duplicates().shape[0] if not audit['outliers'].empty else 0} (ticker, field) pairs")
    print(f"tag_misalignment: {len(audit['tag_misalignment'])} rows across "
         f"{audit['tag_misalignment'][['ticker', 'field']].drop_duplicates().shape[0] if not audit['tag_misalignment'].empty else 0} (ticker, field) pairs")

# AFL, no depAmort, no CostOfRevenue, no operatingIncome after 2024, revenue and Income up and downs since 2023 ?? 
# CB -> no depAmort, no operatingIncome, no costOfRevenue
# DTE -> cash pickes at 2021 Quarter ?? -> weird , no values for SellingGeneralAdmin
# MCD, no totalLiabilities, depAmort huge drop early 2021 ? No pretaxIncome
# RF -> totalRevenue = 0 for all quarters, preTax income missing betwween 2-2- and 2024-Q2, no operatingIncome, no CostofRevenue, no SellingGeneralAdmin,
# MET -> no costOfRevenue, OperatingIncome only since 2025-05, depAmort only since 2024-04, huge drop in Revenue in 2019 -> almost 0 since ??
# REG -> drop of Revenue 2018-2019 ? operatingIncome, only since 2025-05, No CostOfRevenue, cash only up to 2019, epsDiluted only until 2015

