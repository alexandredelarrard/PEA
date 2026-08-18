"""
tiingo_comparison.py  (src/utils/tiingo_comparison.py)
--------------------------------------------------------
External ground-truth cross-check for `fundamentals_history`: compares our
SEC-EDGAR-derived figures against Tiingo's `/tiingo/fundamentals/{ticker}/statements`
API, so a Tiingo disagreement is a THIRD, independent signal alongside the
same-source diagnostics in `analyze_history.py`/`fundamentals_tag_ledger.py`/
`fundamentals_validation.py` (all of which can only ever check our own pipeline
against itself).

Lives in `src/utils/` beside those three, for the same reason stated in
`fundamentals_tag_ledger.py`: a read-only diagnostic over an already-persisted
table must not make `src/utils` import from `src/data_extract`. Not a fetcher --
purely reads `fundamentals_history` and an external API; never writes to the DB.

Two kinds of fields, requiring two different checks (never conflate them):

  * SAME-DEFINITION fields (bucket "a"): totalRevenue, netIncome, totalAssets, ...
    Most of `fundamentals_history`'s flow fields are TTM SUMS (`ttm_a` in
    `fetch_fundamentals.py::_derive_history`), while Tiingo's `statementData`
    reports the DISCRETE quarter -- so a flow field is compared against the sum
    of Tiingo's matched quarter + its 3 PRECEDING quarters, not the bare quarter
    value (`kind="flow"`/`"flow_abs"` below). An instant/balance-sheet field
    compares directly, single quarter vs single quarter (`kind="instant"`). These
    are graded against an EXACT-MATCH tolerance (`TIINGO_EXACT_MATCH_TOLERANCE_*`).

  * KNOWN-DEFINITIONAL-DIFFERENCE fields/ticker pairs (bucket "b": confirmed this
    session -- AXP's total revenue runs the card-issuer "net of interest expense"
    convention while its netIncome/totalAssets match Tiingo to 0.000% on the same
    rows; GS's capex uses a narrower financial-sector definition; HON's share
    counts sit at Tiingo's ~2.00x, plausibly a retroactive split-adjustment
    Tiingo applies that our as-filed XBRL data never does) never get scored
    against the exact-match bar. Instead `ratio_outlier_check` reuses
    `analyze_history.detect_level_outliers`' Modified-Z-score machinery on the
    RATIO series (our_value / tiingo_value) per (ticker, field): a stable
    structural gap keeps a flat ratio and stays quiet, while a NEW discrepancy
    (or a break in a previously-stable ratio) shows up as a level/YoY outlier --
    "stays close to its own history", not "matches Tiingo exactly".

  * Fields with NO Tiingo dataCode at all (bucket "c": `goodwill`,
    `intangiblesGross` -- bundled into one combined `intangibles` line on
    Tiingo's side --, `changeInInventory`/`changeInReceivables`/
    `changeInPayables` -- no working-capital-delta dataCodes exist in Tiingo's
    cash-flow taxonomy --, `dividendsPerShare` -- Tiingo only has the aggregate
    $ `payDiv`, no per-share figure) are marked `tiingo_code=None` and simply
    skipped by both checks above; they already get `analyze_history.py`'s plain
    (non-ratio) `detect_level_outliers` treatment on our own series.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants.constants import (
    DOW_30_TICKERS, TIINGO_CACHE_DIRNAME, TIINGO_COMPARISON_FILENAME,
    TIINGO_EXACT_MATCH_TOLERANCE_FLOW, TIINGO_EXACT_MATCH_TOLERANCE_LEVEL,
    TIINGO_RATIO_OUTLIERS_FILENAME, TIINGO_STATEMENTS_URL_TEMPLATE,
)
from src.context import Context
from src.validate.analyze_history import detect_level_outliers
from src.utils.polite_http import get_json

_LOG: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "TIINGO_FIELD_MAP", "BUCKET_B_OVERRIDES", "BUCKET_B_FIELDS", "classify_bucket",
    "fetch_tiingo_statements", "build_comparison_frame", "ratio_outlier_check",
    "alignment_summary", "run_tiingo_audit",
]

# `kind`:
#   "flow"     -- TTM sum (4 trailing Tiingo quarters) vs our TTM figure, same sign.
#   "flow_abs" -- same, but Tiingo carries the OPPOSITE sign convention (e.g. capex is
#                 a negative outflow on Tiingo, a positive outflow-magnitude here).
#   "instant"  -- single-quarter balance-sheet compare, no summing.
#   "latest_q" -- single-quarter compare for a WEIGHTED-AVERAGE share count (summing an
#                 average across 4 quarters would not be the TTM average -- it is
#                 whatever the single matched quarter reports).
#   None       -- no Tiingo dataCode exists for this field at all (bucket "c").
TIINGO_FIELD_MAP: dict[str, dict[str, str | None]] = {
    "totalRevenue": {"tiingo_code": "revenue", "kind": "flow"},
    "costOfRevenue": {"tiingo_code": "costRev", "kind": "flow"},
    "sellingGeneralAdmin": {"tiingo_code": "sga", "kind": "flow"},
    "operatingIncome": {"tiingo_code": "opinc", "kind": "flow"},
    "pretaxIncome": {"tiingo_code": "ebt", "kind": "flow"},
    "netIncome": {"tiingo_code": "netIncComStock", "kind": "flow"},
    "incomeTaxExpense": {"tiingo_code": "taxExp", "kind": "flow"},
    "interestExpense": {"tiingo_code": "intexp", "kind": "flow_abs"},
    "epsBasic": {"tiingo_code": "eps", "kind": "flow"},
    "epsDiluted": {"tiingo_code": "epsDil", "kind": "flow"},
    "cash": {"tiingo_code": "cashAndEq", "kind": "instant"},
    "cashInclRestricted": {"tiingo_code": "cashAndEq", "kind": "instant"},
    "shortTermInvestments": {"tiingo_code": "investmentsCurrent", "kind": "instant"},
    "currentAssets": {"tiingo_code": "assetsCurrent", "kind": "instant"},
    "totalAssets": {"tiingo_code": "totalAssets", "kind": "instant"},
    "ppeNet": {"tiingo_code": "ppeq", "kind": "instant"},
    "goodwill": {"tiingo_code": None, "kind": None},
    "intangiblesGross": {"tiingo_code": None, "kind": None},
    "currentLiabilities": {"tiingo_code": "liabilitiesCurrent", "kind": "instant"},
    "totalLiabilities": {"tiingo_code": "totalLiabilities", "kind": "instant"},
    "shortTermDebt": {"tiingo_code": "debtCurrent", "kind": "instant"},
    "longTermDebt": {"tiingo_code": "debtNonCurrent", "kind": "instant"},
    "longTermDebtTotal": {"tiingo_code": "debtNonCurrent", "kind": "instant"},
    "stockholdersEquity": {"tiingo_code": "equity", "kind": "instant"},
    "operatingCashFlow": {"tiingo_code": "ncfo", "kind": "flow"},
    "capex": {"tiingo_code": "capex", "kind": "flow_abs"},
    "depAmort": {"tiingo_code": "depamor", "kind": "flow"},
    "stockBasedComp": {"tiingo_code": "sbcomp", "kind": "flow"},
    "changeInInventory": {"tiingo_code": None, "kind": None},
    "changeInReceivables": {"tiingo_code": None, "kind": None},
    "changeInPayables": {"tiingo_code": None, "kind": None},
    "sharesOutstanding": {"tiingo_code": "sharesBasic", "kind": "instant"},
    "basicShares": {"tiingo_code": "shareswa", "kind": "latest_q"},
    "dilutedShares": {"tiingo_code": "shareswaDil", "kind": "latest_q"},
    "dividendsPerShare": {"tiingo_code": None, "kind": None},

    # Added for downstream-consumed fields in fundamental_features.py (EV bridge, EBITDA
    # yield, R&D moat, latest-quarter momentum) that had NO cross-check at all before this
    # pass. Confirmed live against Tiingo's own `/statements` response (2026-08 probe,
    # AAPL) that these dataCodes exist -- unlike `researchAndDevelopment`/`ebitda`/
    # `freeCashflow`, no bucket-b evidence has been gathered yet for these three, so they
    # start in bucket "a" (the default) and must be watched on the first live run:
    # `ebitda` in particular is not a standardized GAAP concept and Tiingo's own add-back
    # methodology may differ from ours, same caution as any other bucket-a candidate that
    # hasn't been reconciled to the dollar yet.
    "freeCashflow": {"tiingo_code": "freeCashFlow", "kind": "flow"},
    "ebitda": {"tiingo_code": "ebitda", "kind": "flow"},
    "researchAndDevelopment": {"tiingo_code": "rnd", "kind": "flow"},
    # Single latest-quarter (not TTM-summed) companions to `totalRevenue`/`netIncome`,
    # feeding fundamental_features.py's `q_rev_growth`/`q_earnings_growth` block. Same
    # Tiingo dataCode as the TTM field, but `kind="latest_q"` (single matched quarter, no
    # trailing-4 sum) -- same reasoning as `basicShares`/`dilutedShares` above.
    "revenue_q": {"tiingo_code": "revenue", "kind": "latest_q"},
    "netIncome_q": {"tiingo_code": "netIncComStock", "kind": "latest_q"},
    # No Tiingo equivalent at all (bucket "c"): headcount is not a financial-statement
    # line item, and Tiingo's `nonControllingInterests` dataCode lives only on the INCOME
    # STATEMENT (the NCI share of net income) -- there is no separate balance-sheet
    # minority-interest/NCI equity line in Tiingo's schema to compare `minorityInterest`
    # against (confirmed via the full AAPL dataCode dump, 2026-08 probe).
    "employees": {"tiingo_code": None, "kind": None},
    "minorityInterest": {"tiingo_code": None, "kind": None},
    # NCI's share of net income -- the one income-statement-side NCI figure Tiingo does
    # expose. Flow, TTM-summed like every other income-statement line.
    "nciIncome": {"tiingo_code": "nonControllingInterests", "kind": "flow"},
}

# Bucket "b": CONFIRMED (ticker, field) pairs whose definition genuinely differs from
# Tiingo's -- never scored against the exact-match tolerance, only against
# `ratio_outlier_check`'s "stayed close to its own history" bar. Add an entry ONLY
# with evidence (a stable, explainable, one-directional gap measured across multiple
# quarters), the same standard `FIELD_TAG_DENYLIST` holds itself to -- this is not a
# place to silence an inconvenient result.
BUCKET_B_OVERRIDES: dict[tuple[str, str], str] = {
    ("AXP", "totalRevenue"): "card-issuer revenue net of interest expense (Tiingo's "
                             "convention differs from our GAAP-tag coalesce)",
    ("GS", "capex"): "financial-sector capex definition narrower than our "
                     "PaymentsToAcquirePropertyPlantAndEquipment coalesce",
    ("HON", "sharesOutstanding"): "~2.00x Tiingo -- plausibly a retroactive split-"
                                  "adjustment Tiingo applies that our as-filed data does not",
    ("HON", "basicShares"): "~2.00x Tiingo -- see sharesOutstanding note",
    ("HON", "dilutedShares"): "~2.00x Tiingo -- see sharesOutstanding note",
}

# Bucket "b" at the FIELD level (not per-ticker): a GENERAL, confirmed convention
# difference that recurs on any ticker material enough for it to show, rather than one
# issuer's own quirk. Confirmed by exact-dollar reconciliation on two independent
# tickers each:
#
#   * `longTermDebt`/`longTermDebtTotal` (-> Tiingo `debtNonCurrent`) and
#     `shortTermDebt` (-> `debtCurrent`): Tiingo normalizes "debt" to include ASC-842
#     lease liabilities, which this schema deliberately tracks as their OWN fields
#     (`financeLeaseLiability(Current|Noncurrent)`, `operatingLeaseLiability(Current|
#     Noncurrent)`) rather than folding into the bond/loan-only debt concept -- see
#     `fundamentals_tags.py`'s own EXTRA_STOCK_TAGS comments on why leases are kept
#     separate (flexible leverage/EV construction in `utils/capital.py`). Verified
#     exactly: WMT 2024-07-31 longTermDebt $35,364M + financeLeaseLiabilityNoncurrent
#     $6,161M + operatingLeaseLiabilityNoncurrent $12,811M = $54,336M = Tiingo's
#     debtNonCurrent to the dollar; MSFT 2026-06-30 longTermDebt $31,067M +
#     operatingLeaseLiabilityNoncurrent $16,532M (no finance lease) = $47,599M,
#     Tiingo's debtNonCurrent to the dollar. NOT a fixable comparison adjustment:
#     which of `longTermDebt`'s own candidate tags won (`LongTermDebtNoncurrent`,
#     bond-only, vs `LongTermDebtAndCapitalLeaseObligations`, ALREADY lease-inclusive)
#     varies by filer, so blindly adding lease fields back in would DOUBLE-COUNT for
#     a filer like HD whose `longTermDebt` already resolved the capital-lease-
#     inclusive tag (confirmed: HD's longTermDebt $45,917M + operatingLeaseLiability-
#     Noncurrent $7,668M = $53,585M = Tiingo exactly, WITHOUT adding finance lease,
#     which would overshoot).
#   * `ppeNet` (-> Tiingo `ppeq`): Tiingo bundles the operating-lease ROU asset into
#     net PP&E; we keep `operatingLeaseRouAsset` separate for the same
#     asset-turnover/leverage reasons. Verified exactly: CRM 2024-04-30 ppeNet
#     $3,506M + operatingLeaseRouAsset $2,255M = $5,761M = Tiingo's ppeq to the
#     dollar.
#   * `cashInclRestricted`: mapped to Tiingo's `cashAndEq` only as an approximation
#     (Tiingo has no separate restricted-cash line at all) -- this was a
#     classification mistake to ever treat as bucket "a"; a discrepancy here is
#     expected by construction, not a signal of anything.
BUCKET_B_FIELDS: frozenset[str] = frozenset({
    "longTermDebt", "longTermDebtTotal", "shortTermDebt", "ppeNet", "cashInclRestricted",
})


def fetch_tiingo_statements(
    ticker: str, *, api_key: str, cache_dir: Path | None = None,
) -> list[dict] | None:
    """One GET to Tiingo's `/statements` endpoint, restricted to QUARTERLY entries
    (`quarter` 1-4; the annual/10-K entry, `quarter=0`, is dropped -- flow fields
    build their own TTM by summing 4 trailing quarterlies, so the annual entry
    would only ever be a redundant, differently-derived duplicate).

    Returns `None` (logs, never raises) when Tiingo has no data for this ticker --
    this is the SAME mechanism that self-discovers plan entitlement: a ticker
    outside the current plan's coverage (confirmed live 2026-08:
    `{"detail": "Error: Free and Power plans are limited to the DOW 30..."}`, and that
    whitelist is itself the PRE-2024 Dow roster -- AMZN/NVDA/SHW, the 2024
    reconstitution's additions, fail too) just comes back empty, with no
    ticker-eligibility list hardcoded anywhere. `run_tiingo_audit` defaults to the full
    analysis universe and lets each ticker self-discover coverage this way; `fundamentals_
    audit.py`'s `run_universe_audit` is what routes anything left uncovered to the Yahoo
    fallback.

    Cached under `cache_dir / f"{ticker}.json"` when a `cache_dir` is given (no
    TTL -- fundamentals restate slowly and this is a periodic audit, not a
    pipeline step; delete the file to force a refresh, same convention as the raw
    SEC filing caches under `SEC_13F_INSIDERS_DIR`)."""
    cache_path = cache_dir / f"{ticker}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return [e for e in data if e.get("quarter") in (1, 2, 3, 4)]
        except (OSError, ValueError, json.JSONDecodeError) as e:
            _LOG.warning("tiingo_comparison: %s cache unreadable (%s), re-fetching", ticker, e)

    url = TIINGO_STATEMENTS_URL_TEMPLATE.format(ticker=ticker)
    data = get_json(url, params={"token": api_key}, impersonate=False, retries=1, log_missing=False)
    if data is None:
        _LOG.info("tiingo_comparison: no data for %s (plan coverage or transport failure)", ticker)
        return None

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    return [e for e in data if e.get("quarter") in (1, 2, 3, 4)]


def _flatten_entry(entry: dict) -> dict:
    out: dict[str, float] = {}
    for items in entry.get("statementData", {}).values():
        for item in items:
            out[item["dataCode"]] = item.get("value")
    return out


def _tiingo_index(entries: list[dict]) -> pd.DataFrame:
    """One row per quarter-end `date`, every dataCode flattened into columns,
    sorted ascending -- the shape `_trailing4_sum`/`_single_value` index into."""
    rows = []
    for e in sorted(entries, key=lambda x: x["date"]):
        row = {"date": pd.Timestamp(e["date"])}
        row.update(_flatten_entry(e))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()


def _nearest_pos(idx: pd.DataFrame, date: pd.Timestamp) -> int | None:
    if idx.empty:
        return None
    pos = idx.index.get_indexer([date], method="nearest", tolerance=pd.Timedelta("10D"))[0]
    return None if pos == -1 else pos


def _trailing4_sum(idx: pd.DataFrame, date: pd.Timestamp, code: str) -> float | None:
    if code not in idx.columns:
        return None
    pos = _nearest_pos(idx, date)
    if pos is None or pos < 3:
        return None
    window = idx.iloc[pos - 3: pos + 1][code]
    return None if window.isna().any() else float(window.sum())


def _single_value(idx: pd.DataFrame, date: pd.Timestamp, code: str) -> float | None:
    if code not in idx.columns:
        return None
    pos = _nearest_pos(idx, date)
    if pos is None:
        return None
    v = idx.iloc[pos][code]
    return None if pd.isna(v) else float(v)


def _compare_one(our_val: float | None, idx: pd.DataFrame, fiscal_end: pd.Timestamp,
                 kind: str | None, code: str | None) -> tuple[float | None, str]:
    """One (our_value, tiingo_value) pair -> (tiingo_value, note). `note` explains
    a None on either side; empty when both resolved."""
    if kind is None or code is None:
        return None, "no Tiingo equivalent for this field"
    if kind in ("flow", "flow_abs"):
        tv = _trailing4_sum(idx, fiscal_end, code)
        if tv is not None and kind == "flow_abs":
            tv = abs(tv)
    else:                                       # "instant" or "latest_q"
        tv = _single_value(idx, fiscal_end, code)
    if our_val is None or tv is None:
        return tv, "missing value (our DB or Tiingo, or outside Tiingo's covered window)"
    return tv, ""


def build_comparison_frame(
    context: Context, tickers: list[str], *, field_map: dict = TIINGO_FIELD_MAP,
    api_key: str, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Per (ticker, field, fiscal_end): `our_value`/`tiingo_value`/`delta_pct`, kind-
    dispatched per `TIINGO_FIELD_MAP`. One row per (ticker, field, fiscal_end) --
    `fiscal_end IS NULL` rows in `fundamentals_history` are dropped (nothing to
    align a Tiingo quarter against)."""
    cols = ["ticker", "field", "quarter", "our_value", "tiingo_value", "delta_pct",
           "kind", "bucket", "note"]
    hist = context.store.load(
        "fundamentals_history",
        columns=["ticker", "fiscal_end"] + list(field_map.keys()),
        where={"ticker": list(tickers)}, optional=True,
    )
    if hist is None:
        return pd.DataFrame(columns=cols)
    hist = hist.dropna(subset=["fiscal_end"]).copy()
    hist["fiscal_end"] = pd.to_datetime(hist["fiscal_end"])

    rows: list[dict] = []
    for ticker, sub in hist.groupby("ticker"):
        entries = fetch_tiingo_statements(ticker, api_key=api_key, cache_dir=cache_dir)
        if not entries:
            continue
        idx = _tiingo_index(entries)
        if idx.empty:
            continue
        for _, r in sub.iterrows():
            fe = r["fiscal_end"]
            for field, spec in field_map.items():
                our_val = r.get(field)
                our_val = None if pd.isna(our_val) else float(our_val)
                tv, note = _compare_one(our_val, idx, fe, spec["kind"], spec["tiingo_code"])
                delta = None
                if our_val is not None and tv is not None:
                    if tv == 0:
                        note = note or f"zero Tiingo base (abs diff={our_val:,.2f})"
                    else:
                        delta = (our_val - tv) / abs(tv) * 100
                bucket = classify_bucket(ticker, field, spec["kind"])
                rows.append({"ticker": ticker, "field": field, "quarter": fe.date(),
                            "our_value": our_val, "tiingo_value": tv, "delta_pct": delta,
                            "kind": spec["kind"], "bucket": bucket, "note": note})
    return pd.DataFrame(rows, columns=cols)


def classify_bucket(ticker: str, field: str, kind: str | None) -> str:
    """"a" (same-definition, exact-match target), "b" (known definitional
    difference -- either a whole-field convention gap in `BUCKET_B_FIELDS`, e.g.
    Tiingo's lease-inclusive debt/PP&E, or a per-ticker case confirmed in
    `BUCKET_B_OVERRIDES`), or "c" (no Tiingo equivalent at all)."""
    if kind is None:
        return "c"
    if field in BUCKET_B_FIELDS or (ticker, field) in BUCKET_B_OVERRIDES:
        return "b"
    return "a"


def ratio_outlier_check(
    comparison_frame: pd.DataFrame, *, threshold: float = 3.5,
) -> pd.DataFrame:
    """Reuses `analyze_history.detect_level_outliers` UNMODIFIED on the ratio series
    (our_value / tiingo_value) per (ticker, field) -- a flag means the ratio moved
    off its OWN historical level, which is the right test for a field whose
    definition may legitimately differ from Tiingo's by a stable structural gap
    (AXP's steady revenue premium, GS's capex convention, HON's ~2.00x share
    count) but should not drift or suddenly jump.

    Caveat (documented, not fixed here): a genuine one-time step -- a real stock
    split moving the ratio from 1.0x to 2.0x permanently -- will register as an
    outlier right at the boundary, same as any other level shift. A human
    spot-check of whether flags cluster at one date and then stay flat (benign)
    vs. scatter continuously (worth investigating) is the right read; building a
    second, era-aware detector is out of scope for this pass."""
    cols = ["ticker", "field", "fiscal_year", "fiscal_period", "value",
           "is_level_outlier", "level_z_score", "is_yoy_outlier"]
    usable = comparison_frame[
        comparison_frame["tiingo_value"].notna() & (comparison_frame["tiingo_value"] != 0)
        & comparison_frame["our_value"].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame(columns=cols)

    usable["quarter"] = pd.to_datetime(usable["quarter"])
    usable["value"] = usable["our_value"] / usable["tiingo_value"]
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
    """Bucket-"a" alignment %, per field and pooled -- the "95%+" metric. Bucket
    "b"/"c" are reported separately (via `ratio_outlier_check` and plain coverage
    counts) and never folded into this denominator."""
    a = comparison_frame[(comparison_frame["bucket"] == "a") & comparison_frame["delta_pct"].notna()].copy()
    if a.empty:
        return pd.DataFrame(columns=["field", "n", "pct_within_tolerance", "tolerance"])
    tol = np.where(a["kind"].isin(["flow", "flow_abs"]),
                  TIINGO_EXACT_MATCH_TOLERANCE_FLOW * 100, TIINGO_EXACT_MATCH_TOLERANCE_LEVEL * 100)
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


def run_tiingo_audit(
    context: Context, tickers: list[str] | None = None, *, field_map: dict = TIINGO_FIELD_MAP,
    threshold: float = 3.5, api_key: str, cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """`run_audit`-shaped orchestrator: builds the comparison frame across every
    ticker, then the ratio-outlier check on top of it. One bad ticker (Tiingo
    HTTP failure, no coverage) never blocks the others -- `build_comparison_frame`
    already skips a ticker with no usable Tiingo data via `fetch_tiingo_statements`
    returning `None`.

    `tickers=None` resolves to the full analysis universe (`load_universe_tickers`) --
    Tiingo's plan-coverage gate (confirmed live: Free/Power = the PRE-2024 Dow roster
    only) is discovered per-ticker at fetch time, not predicted from `DOW_30_TICKERS`,
    so there is no reason to default to that constant anymore; pass it explicitly for a
    cheap smoke test instead. `fundamentals_audit.py`'s `run_universe_audit` is the
    caller that adds the Yahoo fallback for whatever this leaves uncovered."""
    if tickers is None:
        from src.utils.universe import load_universe_tickers
        tickers = load_universe_tickers(context)
    comparison = build_comparison_frame(context, list(tickers), field_map=field_map,
                                       api_key=api_key, cache_dir=cache_dir)
    ratio_outliers = ratio_outlier_check(comparison, threshold=threshold)
    return {"comparison": comparison, "ratio_outliers": ratio_outliers,
           "alignment": alignment_summary(comparison)}


if __name__ == "__main__":
    import os

    from src.context import get_config_context

    _, context = get_config_context(config_path="./configs", use_cache=True, save=False)
    api_key = os.getenv("TIIGO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TIIGO_API_KEY is not set. Add it to your .env file to run the Tiingo "
            "cross-validation audit."
        )
    cache_dir = context.paths["DATA_STORE"] / "gaps" / TIINGO_CACHE_DIRNAME

    result = run_tiingo_audit(context, api_key=api_key, cache_dir=cache_dir)

    out_dir = context.paths["DATA_STORE"] / "gaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["comparison"].to_csv(out_dir / TIINGO_COMPARISON_FILENAME, index=False)
    result["ratio_outliers"].to_csv(out_dir / TIINGO_RATIO_OUTLIERS_FILENAME, index=False)

    print("\n=== SANITY CHECK: Tiingo cross-validation (full universe) ===")
    print(result["alignment"].to_string(index=False))
    n_ratio_flags = len(result["ratio_outliers"])
    print(f"  bucket-b/c ratio-outlier flags: {n_ratio_flags}")
    print("  Validated." if n_ratio_flags == 0 else "  Review flagged rows before calling it validated.")
