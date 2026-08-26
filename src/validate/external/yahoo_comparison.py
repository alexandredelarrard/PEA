"""
yahoo_comparison.py  (src/validate/external/yahoo_comparison.py)
------------------------------------------------------
External ground-truth cross-check for `fundamentals_history_sec`: compares our stored
figures against Yahoo Finance and reports where they disagree. Never writes to the DB.

THE ONLY EXTERNAL ADAPTER, and it excludes no ticker subset -- its audit covers the whole
universe it is handed. For an independent fundamentals cross-source WITH point-in-time
depth, use `fundamentals_sharadar`; this module cannot supply that (see the depth caveat).

Uses the `yfinance` library directly (already a pinned project dependency, not a new one)
-- `Ticker.quarterly_income_stmt` / `quarterly_balance_sheet` / `quarterly_cashflow` --
rather than scraping.

⚠ DEPTH CAVEAT (load-bearing -- read before trusting a "no discrepancy" result here):
yfinance's quarterly statements typically expose only ~4-5 TRAILING quarters of
CURRENT-RESTATED values. There is **no as-filed point-in-time depth**, nowhere near
`fundamentals_history_sec`'s 2007-2026 span. This module can therefore validate only the
MOST RECENT few quarters, and it can NEVER check a historical as-filed value -- which is
precisely the property the validator cares about most. A clean result here is weak
evidence, not a clearance. For as-filed history, cross-check against
`fundamentals_sharadar` instead.

STRUCTURE: field map -> bucket a/b/c -> `ratio_outlier_check` (reusing
`outliers.detect_level_outliers`). Buckets partition fields by whether Yahoo's DEFINITION
matches ours: (a) same basis, so a gap is a real discrepancy and is scored against a
tolerance; (b) known basis difference, so only a CHANGE in the ratio is meaningful;
(c) unmapped.

FIELD MAP CAVEAT: `YAHOO_FIELD_MAP`'s `yahoo_row` labels are yfinance's own statement row
names, which have shifted across yfinance releases and are NOT re-verified here against a
live pull at authoring time. `fetch_yahoo_statements` returns whatever rows exist and
`_row_value` returns `None` (never raises) for a row that is not present, so a stale label
degrades to "no comparison for this field" rather than a crash -- expect to correct a
handful of labels against real output on the first live run. Bucket-b entries should only
ever be added on live dollar evidence, never on suspicion.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.constants.constants import (
    YAHOO_CACHE_DIRNAME, YAHOO_COMPARISON_FILENAME,
    YAHOO_EXACT_MATCH_TOLERANCE_FLOW, YAHOO_EXACT_MATCH_TOLERANCE_LEVEL,
    YAHOO_RATIO_OUTLIERS_FILENAME,
)
from src.context import Context
from src.data_store.schema import Tables
from src.validate.outliers import detect_level_outliers

_LOG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "YAHOO_FIELD_MAP", "BUCKET_B_OVERRIDES", "BUCKET_B_FIELDS", "classify_bucket",
    "fetch_yahoo_statements", "build_comparison_frame", "ratio_outlier_check",
    "alignment_summary", "run_yahoo_audit",
]

# `kind` vocabulary:
#   "flow"     -- TTM sum (4 trailing yfinance quarters) vs our TTM figure, same sign.
#   "flow_abs" -- same, but yfinance carries the OPPOSITE sign convention.
#   "instant"  -- single-quarter balance-sheet compare, no summing.
#   "latest_q" -- single-quarter compare (weighted-average share counts, or the
#                 latest-quarter-only companions to a TTM field).
#   None       -- no yfinance row for this field at all (bucket "c").
YAHOO_FIELD_MAP: dict[str, dict[str, str | None]] = {
    "totalRevenue": {"statement": "income", "yahoo_row": "Total Revenue", "kind": "flow"},
    "costOfRevenue": {"statement": "income", "yahoo_row": "Cost Of Revenue", "kind": "flow"},
    "sellingGeneralAdmin": {"statement": "income",
                            "yahoo_row": "Selling General And Administration", "kind": "flow"},
    "operatingIncome": {"statement": "income", "yahoo_row": "Operating Income", "kind": "flow"},
    "pretaxIncome": {"statement": "income", "yahoo_row": "Pretax Income", "kind": "flow"},
    "netIncome": {"statement": "income", "yahoo_row": "Net Income", "kind": "flow"},
    "incomeTaxExpense": {"statement": "income", "yahoo_row": "Tax Provision", "kind": "flow"},
    "interestExpense": {"statement": "income", "yahoo_row": "Interest Expense", "kind": "flow_abs"},
    "epsBasic": {"statement": "income", "yahoo_row": "Basic EPS", "kind": "flow"},
    "epsDiluted": {"statement": "income", "yahoo_row": "Diluted EPS", "kind": "flow"},
    "cash": {"statement": "balance", "yahoo_row": "Cash And Cash Equivalents", "kind": "instant"},
    "currentAssets": {"statement": "balance", "yahoo_row": "Current Assets", "kind": "instant"},
    "totalAssets": {"statement": "balance", "yahoo_row": "Total Assets", "kind": "instant"},
    "ppeNet": {"statement": "balance", "yahoo_row": "Net PPE", "kind": "instant"},
    "goodwill": {"statement": "balance", "yahoo_row": "Goodwill", "kind": "instant"},
    "currentLiabilities": {"statement": "balance", "yahoo_row": "Current Liabilities",
                          "kind": "instant"},
    "totalLiabilities": {"statement": "balance",
                         "yahoo_row": "Total Liabilities Net Minority Interest",
                         "kind": "instant"},
    "shortTermDebt": {"statement": "balance", "yahoo_row": "Current Debt", "kind": "instant"},
    "longTermDebt": {"statement": "balance", "yahoo_row": "Long Term Debt", "kind": "instant"},
    "stockholdersEquity": {"statement": "balance", "yahoo_row": "Stockholders Equity",
                          "kind": "instant"},
    "operatingCashFlow": {"statement": "cashflow", "yahoo_row": "Operating Cash Flow",
                          "kind": "flow"},
    "capex": {"statement": "cashflow", "yahoo_row": "Capital Expenditure", "kind": "flow_abs"},
    "depAmort": {"statement": "cashflow", "yahoo_row": "Depreciation And Amortization",
                "kind": "flow"},
    "stockBasedComp": {"statement": "cashflow", "yahoo_row": "Stock Based Compensation",
                      "kind": "flow"},
    "sharesOutstanding": {"statement": "balance", "yahoo_row": "Share Issued", "kind": "instant"},
    "dividendsPerShare": {"statement": None, "yahoo_row": None, "kind": None},
    "freeCashflow": {"statement": "cashflow", "yahoo_row": "Free Cash Flow", "kind": "flow"},
    "ebitda": {"statement": "income", "yahoo_row": "EBITDA", "kind": "flow"},
    "researchAndDevelopment": {"statement": "income",
                              "yahoo_row": "Research And Development", "kind": "flow"},
    "revenue_q": {"statement": "income", "yahoo_row": "Total Revenue", "kind": "latest_q"},
    "netIncome_q": {"statement": "income", "yahoo_row": "Net Income", "kind": "latest_q"},
    "employees": {"statement": None, "yahoo_row": None, "kind": None},
    # Yahoo carries a distinct balance-sheet NCI line, which most aggregators fold into
    # equity -- so this field is genuinely checkable here.
    "minorityInterest": {"statement": "balance", "yahoo_row": "Minority Interest",
                        "kind": "instant"},
    "nciIncome": {"statement": None, "yahoo_row": None, "kind": None},
}

# Bucket "b": whole-field conventions Yahoo's OWN row naming already tells us differ,
# flagged provisionally by that naming alone -- NOT yet dollar-reconciled on two
# independent tickers, which is the bar a bucket-b entry must clear. Treat
# these as "expected to need bucket b" pending confirmation on the first live run, not
# settled evidence; demote to bucket "a" if the live run shows no discrepancy at all.
BUCKET_B_FIELDS: frozenset[str] = frozenset({
    # Yahoo's own label states this INCLUDES minority interest; our `totalLiabilities`
    # coalesce does not carve NCI out of liabilities either, so this may in fact match --
    # named here only because the label is a plausible convention gap, not confirmed one.
    "totalLiabilities",
})
BUCKET_B_OVERRIDES: dict[tuple[str, str], str] = {}


def fetch_yahoo_statements(
    ticker: str, *, cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame] | None:
    """One `yfinance.Ticker(ticker)` pull -> `{"income": df, "balance": df, "cashflow":
    df}`, each yfinance's own shape (rows = statement line labels, columns = quarter-end
    Timestamps, most-recent first). Returns `None` (logs, never raises) when yfinance has
    nothing for this ticker -- delisted/wrong symbol/transport failure are all
    indistinguishable from "not covered" from this module's point of view.

    Cached under `cache_dir/{ticker}_{income,balance,cashflow}.parquet` when given
    (no TTL -- delete to force a refresh)."""
    if cache_dir is not None:
        paths = {k: cache_dir / f"{ticker}_{k}.parquet" for k in ("income", "balance", "cashflow")}
        if all(p.exists() for p in paths.values()):
            try:
                return {k: pd.read_parquet(p) for k, p in paths.items()}
            except (OSError, ValueError) as e:
                _LOG.warning("yahoo_comparison: %s cache unreadable (%s), re-fetching", ticker, e)

    try:
        tk = yf.Ticker(ticker)
        income, balance, cashflow = (
            tk.quarterly_income_stmt, tk.quarterly_balance_sheet, tk.quarterly_cashflow,
        )
    except Exception as e:  # yfinance raises a mix of transport/parsing errors on a bad ticker
        _LOG.info("yahoo_comparison: no data for %s (%s)", ticker, e)
        return None
    if income is None or income.empty:
        _LOG.info("yahoo_comparison: no data for %s (empty statements)", ticker)
        return None

    out = {"income": income, "balance": balance, "cashflow": cashflow}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for k, df in out.items():
            (df if df is not None else pd.DataFrame()).to_parquet(cache_dir / f"{ticker}_{k}.parquet")
    return out


# Position-finding helpers. Small, generic, stable positional logic kept local to this
# module rather than hoisted -- the same rationale `outliers.py` documents for its own
# duplicated mapping.
def _yf_index(df: pd.DataFrame | None) -> pd.DataFrame:
    """Transpose to date-ascending rows / line-item columns, the shape the
    trailing-4-quarter and nearest-date logic below expects."""
    if df is None or df.empty:
        return pd.DataFrame()
    idx = df.T
    idx.index = pd.to_datetime(idx.index)
    return idx.sort_index()


def _nearest_pos(idx: pd.DataFrame, date: pd.Timestamp) -> int | None:
    if idx.empty:
        return None
    pos = idx.index.get_indexer([date], method="nearest", tolerance=pd.Timedelta("10D"))[0]
    return None if pos == -1 else pos


def _trailing4_sum(idx: pd.DataFrame, date: pd.Timestamp, row: str) -> float | None:
    if row not in idx.columns:
        return None
    pos = _nearest_pos(idx, date)
    if pos is None or pos < 3:
        return None
    window = idx.iloc[pos - 3: pos + 1][row]
    return None if window.isna().any() else float(window.sum())


def _single_value(idx: pd.DataFrame, date: pd.Timestamp, row: str) -> float | None:
    if row not in idx.columns:
        return None
    pos = _nearest_pos(idx, date)
    if pos is None:
        return None
    v = idx.iloc[pos][row]
    return None if pd.isna(v) else float(v)


def _compare_one(our_val: float | None, statements: dict[str, pd.DataFrame],
                 fiscal_end: pd.Timestamp, kind: str | None, statement: str | None,
                 row: str | None) -> tuple[float | None, str]:
    if kind is None or row is None or statement is None:
        return None, "no Yahoo equivalent for this field"
    idx = _yf_index(statements.get(statement))
    if kind in ("flow", "flow_abs"):
        yv = _trailing4_sum(idx, fiscal_end, row)
        if yv is not None and kind == "flow_abs":
            yv = abs(yv)
    else:  # "instant" or "latest_q"
        yv = _single_value(idx, fiscal_end, row)
    if our_val is None or yv is None:
        return yv, "missing value (our DB or Yahoo, or outside Yahoo's ~4-5Q covered window)"
    return yv, ""


def build_comparison_frame(
    context: Context, tickers: list[str], *, field_map: dict = YAHOO_FIELD_MAP,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Per (ticker, field, fiscal_end): `our_value`/`yahoo_value`/`delta_pct`,
    kind-dispatched per `YAHOO_FIELD_MAP`. This frame IS the contract the ranked
    findings artifact consumes; keep the column list stable."""
    cols = ["ticker", "field", "quarter", "our_value", "yahoo_value", "delta_pct",
           "kind", "bucket", "note"]
    hist = context.store.load(
        Tables.fundamentals_history_sec,
        columns=["ticker", "fiscal_end"] + list(field_map.keys()),
        where={"ticker": list(tickers)}, optional=True,
    )
    if hist is None:
        return pd.DataFrame(columns=cols)
    hist = hist.dropna(subset=["fiscal_end"]).copy()
    hist["fiscal_end"] = pd.to_datetime(hist["fiscal_end"])

    rows: list[dict] = []
    for ticker, sub in hist.groupby("ticker"):
        statements = fetch_yahoo_statements(ticker, cache_dir=cache_dir)
        if not statements:
            continue
        # only the LAST ~4-5 quarters exist on Yahoo -- restrict to fiscal_ends actually
        # reachable rather than iterating the whole 2007-2026 history for nothing.
        reachable = _yf_index(statements.get("income")).index
        if reachable.empty:
            continue
        recent = sub[sub["fiscal_end"] >= reachable.min() - pd.Timedelta("100D")]
        for _, r in recent.iterrows():
            fe = r["fiscal_end"]
            for field, spec in field_map.items():
                our_val = r.get(field)
                our_val = None if pd.isna(our_val) else float(our_val)
                yv, note = _compare_one(our_val, statements, fe, spec["kind"],
                                        spec["statement"], spec["yahoo_row"])
                delta = None
                if our_val is not None and yv is not None:
                    if yv == 0:
                        note = note or f"zero Yahoo base (abs diff={our_val:,.2f})"
                    else:
                        delta = (our_val - yv) / abs(yv) * 100
                bucket = classify_bucket(ticker, field, spec["kind"])
                rows.append({"ticker": ticker, "field": field, "quarter": fe.date(),
                            "our_value": our_val, "yahoo_value": yv, "delta_pct": delta,
                            "kind": spec["kind"], "bucket": bucket, "note": note})
    return pd.DataFrame(rows, columns=cols)


def classify_bucket(ticker: str, field: str, kind: str | None) -> str:
    """"a" (same-definition, exact-match target), "b" (a naming-implied or confirmed
    definitional difference), or "c" (no Yahoo equivalent at all)."""
    if kind is None:
        return "c"
    if field in BUCKET_B_FIELDS or (ticker, field) in BUCKET_B_OVERRIDES:
        return "b"
    return "a"


def ratio_outlier_check(
    comparison_frame: pd.DataFrame, *, threshold: float = 3.5,
) -> pd.DataFrame:
    """Reuses `outliers.detect_level_outliers` UNMODIFIED on the ratio series
    (our_value / yahoo_value) per (ticker, field), including decision 60's log-change kernel:
    a permanent step flags only its boundary quarter, and a one-quarter spike flags
    twice (in and back out).

    With only ~4-5 quarters of Yahoo history per ticker, most series here are too short
    for the level-outlier machinery at all -- `detect_level_outliers` needs >= 3 points
    and the log-change kernel spends one of them on the undefined first period, so it
    needs 4. This mainly catches an outright order-of-magnitude miss, not a subtle
    drift, and an outright SIGN flip is now invisible rather than caught: no log ratio
    exists across zero. Both are structural limits of the depth Yahoo exposes and of a
    scale-free statistic, not bugs in the check itself."""
    cols = ["ticker", "field", "fiscal_year", "fiscal_period", "value",
           "is_level_outlier", "level_z_score", "is_yoy_outlier"]
    usable = comparison_frame[
        comparison_frame["yahoo_value"].notna() & (comparison_frame["yahoo_value"] != 0)
        & comparison_frame["our_value"].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame(columns=cols)

    usable["quarter"] = pd.to_datetime(usable["quarter"])
    usable["value"] = usable["our_value"] / usable["yahoo_value"]
    usable["fiscal_year"] = usable["quarter"].dt.year
    usable["fiscal_period"] = "Q" + (((usable["quarter"].dt.month - 1) // 3) + 1).astype(str)
    usable["duration_type"] = "quarterly"
    usable["filing_date"] = usable["quarter"]
    usable["source_tag"] = None
    usable["is_amendment"] = 0.0
    usable["derived"] = 0.0

    frames = []
    for (ticker, field), _ in usable.groupby(["ticker", "field"]):
        res = detect_level_outliers(usable, ticker, field, duration_type="quarterly",
                                    threshold=threshold, check_yoy=False)
        if not res.empty:
            frames.append(res[res["is_level_outlier"] | res["is_yoy_outlier"]])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def alignment_summary(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket-"a" alignment %, per field and pooled."""
    a = comparison_frame[(comparison_frame["bucket"] == "a")
                        & comparison_frame["delta_pct"].notna()].copy()
    if a.empty:
        return pd.DataFrame(columns=["field", "n", "pct_within_tolerance", "tolerance"])
    tol = np.where(a["kind"].isin(["flow", "flow_abs"]),
                  YAHOO_EXACT_MATCH_TOLERANCE_FLOW * 100, YAHOO_EXACT_MATCH_TOLERANCE_LEVEL * 100)
    a["within_tolerance"] = a["delta_pct"].abs() <= tol
    a["tolerance"] = tol
    rows = []
    for field, grp in a.groupby("field"):
        rows.append({"field": field, "n": len(grp),
                     "pct_within_tolerance": grp["within_tolerance"].mean() * 100,
                     "tolerance": grp["tolerance"].iloc[0]})
    rows.append({"field": "__all__", "n": len(a),
                "pct_within_tolerance": a["within_tolerance"].mean() * 100, "tolerance": None})
    return pd.DataFrame(rows)


def run_yahoo_audit(
    context: Context, tickers: list[str], *, field_map: dict = YAHOO_FIELD_MAP,
    threshold: float = 3.5, cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """`run_audit`-shaped orchestrator: comparison frame + ratio outliers + alignment,
    over whatever tickers it is handed.

    Pass the FULL universe: no subset is reserved for another adapter."""
    comparison = build_comparison_frame(context, list(tickers), field_map=field_map,
                                       cache_dir=cache_dir)
    ratio_outliers = ratio_outlier_check(comparison, threshold=threshold)
    return {"comparison": comparison, "ratio_outliers": ratio_outliers,
           "alignment": alignment_summary(comparison)}


if __name__ == "__main__":
    from src.context import get_config_context
    from src.utils.universe import load_universe_tickers

    _, context = get_config_context(config_path="./configs", use_cache=True, save=False)
    # The FULL universe -- no subset is excluded, so the most liquid and best-covered
    # names are checked too.
    universe = load_universe_tickers(context)
    cache_dir = context.paths["DATA_STORE"] / "gaps" / YAHOO_CACHE_DIRNAME

    result = run_yahoo_audit(context, universe, cache_dir=cache_dir)

    out_dir = context.paths["DATA_STORE"] / "gaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["comparison"].to_csv(out_dir / YAHOO_COMPARISON_FILENAME, index=False)
    result["ratio_outliers"].to_csv(out_dir / YAHOO_RATIO_OUTLIERS_FILENAME, index=False)

    print("\n=== SANITY CHECK: Yahoo cross-validation ===")
    print(result["alignment"].to_string(index=False))
    n_ratio_flags = len(result["ratio_outliers"])
    print(f"  bucket-b/c ratio-outlier flags: {n_ratio_flags}")
    print("  Validated." if n_ratio_flags == 0 else "  Review flagged rows before calling it validated.")
