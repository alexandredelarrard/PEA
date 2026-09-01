"""
prices.py  (src/validate/prices.py)
--------------------------------------------------------------------------------------------
The price / share-count ADJUSTMENT-BASIS validator: the invariant that would have caught the
2026-09-01 market-cap defect, run on demand and cheap enough to run nightly.

The bug it exists to prevent survived for one reason -- NOTHING CHECKED. `src/validate/` had
no price validator, and `tests/` contained zero occurrences of `auto_adjust`, `adj_close`,
`unadjusted` or any split-ratio assertion. Meanwhile the check was free the whole time:
Sharadar publishes `marketcap` on exactly the basis `close_split x sharesOutstanding` is
supposed to reproduce, so the identity is verifiable on every joined row.

READ-ONLY, always. Sharadar's own `marketcap` is internally inconsistent for spinoff names
(see invariant 1), so an auto-correction here would propagate a second vendor's error into
the repo's own numbers.

## What blocks, and why only one of them

`gate()` blocks the cube build on INVARIANT 3 ALONE. That is a measured decision, not a
timid one: invariant 1 fails on 12.6% of joined rows even on a CORRECT table, because Yahoo
back-adjusts prices for SPINOFFS while Sharadar's `sharesbas` does not -- so for HON, DD, GE,
FDX, BDX and ~220 others the reference itself is the inconsistent side. Blocking there would
block every build for someone else's defect. Invariant 3's failures have no such ambiguity:
an unexplained >50% round-trip with no split on the books is always a data fault, and it is
the mechanism that silently re-corrupts the table every time a stock splits.

## The three invariants

1. THE MARKET-CAP IDENTITY (reported, not blocking -- see above)
       | close_split(d) x sharesOutstanding(d) / sharadar.marketcap(d) - 1 |  <  1%
   Two independent vendors, one arithmetic identity. Both legs carry the same retroactive
   SPLIT restatement, so the future-split factor cancels and what is left is the true
   historical market cap.

2. PRICE VINTAGE FRESHNESS
       | close_split(d) / sharadar.price(d) - 1 |  <  0.5%     on filing dates
   Yahoo's `Close` and Sharadar's `price` are the same basis on two independent feeds, so a
   disagreement is a STALE ADJUSTMENT VINTAGE in one of them. This is the check that would
   have flagged MNST in July 2026 instead of an audit finding it in September.

3. SPIKE-AND-REVERT
   A >50% jump whose LEVEL comes back within a few bars, with no corroborating row in
   `prices_splits`, is two adjustment vintages meeting inside one ticker -- not a market
   event. Genuine moves must pass: 2020-03-09's oil crash (APA/OXY/FANG/TRGP), PCG's
   bankruptcy, CVNA 2022.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd

from src.context import Context
from src.data_store.schema import Tables

#: Invariant 1's band. 1% absorbs the as-of join (a filing date is often not a trading day)
#: and Sharadar's four-significant-figure rounding, and is an order of magnitude tighter than
#: any real basis error -- those are integer split factors (2x, 4x, 20x) or a dividend factor
#: that reached 0.62 in 2003.
MCAP_TOLERANCE = 0.01
#: Invariant 2's band. Tighter than invariant 1 because it compares two PRICES with no share
#: count in between, so only vendor rounding and the as-of join separate them.
PRICE_TOLERANCE = 0.005
#: Invariant 3, matching `scripts/basis_baseline.py`: a >50% jump whose level returns to
#: within 10% of the pre-jump close inside 7 bars. A vintage seam is a PLATEAU, not a one-day
#: tick -- MNST held the new basis for six bars before flipping back, so a strict next-day
#: test finds nothing while the table is visibly corrupt.
SPIKE_THRESHOLD = 0.50
SPIKE_REVERT_BARS = 7
SPIKE_REVERT_BAND = 0.10
#: How near a split event has to be to excuse a jump. Vendors disagree by a day or two on
#: whether an ex-date is the record or the trading date.
SPLIT_MATCH_DAYS = 3
#: Days either side of a filing date within which the last price bar is accepted.
ASOF_TOLERANCE_DAYS = 5


@dataclass
class InvariantResult:
    """One invariant's outcome, clustered BY TICKER.

    Per-ticker, not per-row, on purpose: one badly-adjusted ticker produces sixty failing
    rows, and sixty findings for one cause is how a report becomes unreadable and stops being
    read. The row counts stay in the summary.
    """
    name: str
    rows: int = 0
    failed: int = 0
    tickers: int = 0
    failing_tickers: dict[str, dict] = dataclass_field(default_factory=dict)
    detail: list[dict] = dataclass_field(default_factory=list)

    @property
    def share(self) -> float:
        return self.failed / self.rows if self.rows else 0.0

    def summary(self) -> str:
        return (f"{self.name}: {self.rows - self.failed:,}/{self.rows:,} pass "
                f"({1 - self.share:.2%}); {len(self.failing_tickers)} of {self.tickers} "
                f"tickers affected")


def _as_ns(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Parse a date column to NANOSECOND resolution.

    Postgres DATE round-trips as `datetime64[s]` and TIMESTAMP as `datetime64[us]`, and
    `merge_asof` refuses to join two keys of different resolution. `fundamentals_sharadar`
    dates are DATE, `prices.date` is TIMESTAMP."""
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    return frame


# --------------------------------------------------------------------------- #
# loading                                                                     #
# --------------------------------------------------------------------------- #
def load_panel(context: Context, tickers: list[str] | None = None,
               since: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (ticker, filing date): both vendors' prices and both share counts.

    An AS-OF join, not an equality join: a filing date is frequently a weekend or a holiday,
    and the market cap a filing row implies is the one from the last bar at or before it --
    which is exactly what `pit.daily_market_cap` computes after its forward-fill."""
    where = {"ticker": tickers} if tickers else None

    prices = context.store.load(Tables.prices, columns=["ticker", "date", "close_split"],
                                where=where)
    prices = _as_ns(prices, "date").dropna(subset=["close_split"])

    vendor = context.store.load(
        Tables.sharadar_fundamentals,
        columns=["ticker", "date", "dimension", "price", "sharesbas", "marketcap"],
        where={**(where or {}), "dimension": "ARQ"}, since=since)
    vendor = _as_ns(vendor, "date")

    merged = context.store.load(Tables.fundamentals_history,
                                columns=["ticker", "as_of", "sharesOutstanding"], where=where)
    merged = _as_ns(merged, "as_of")

    panel = vendor.merge(merged, left_on=["ticker", "date"], right_on=["ticker", "as_of"],
                         how="left")
    panel = pd.merge_asof(panel.sort_values("date"), prices.sort_values("date"),
                          on="date", by="ticker", direction="backward",
                          tolerance=pd.Timedelta(days=ASOF_TOLERANCE_DAYS))
    return panel


# --------------------------------------------------------------------------- #
# the three invariants                                                        #
# --------------------------------------------------------------------------- #
def _cluster(frame: pd.DataFrame, bad: pd.Series, ratio_col: str) -> dict[str, dict]:
    """Failing rows -> one entry per ticker, with enough evidence to act on."""
    out: dict[str, dict] = {}
    for ticker, group in frame[bad].groupby("ticker"):
        ratios = group[ratio_col]
        out[str(ticker)] = {
            "rows": int(len(group)),
            "median_ratio": round(float(ratios.median()), 4),
            "min_ratio": round(float(ratios.min()), 4),
            "max_ratio": round(float(ratios.max()), 4),
            "first_date": str(group["date"].min().date()),
            "last_date": str(group["date"].max().date()),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["rows"]))


def invariant_market_cap(panel: pd.DataFrame) -> InvariantResult:
    """INVARIANT 1 -- `close_split x sharesOutstanding == sharadar.marketcap`.

    ⚠ A FAILURE IS NOT AUTOMATICALLY THIS REPO'S FAULT, and the validator must never
    auto-correct on it. Sharadar's `marketcap` is `price x sharesbas`, and its `price` is
    adjusted for splits ONLY while Yahoo's `close_split` is adjusted for splits AND SPINOFFS.
    `sharesbas` carries the split restatement and no spinoff one, so for a ticker with a
    spinoff in its history the two vendors legitimately disagree and it is SHARADAR whose
    product is internally inconsistent: HON's `sharesbas` is unchanged across its 2026-06-29
    spinoff (316,826,560 -> 316,940,010) while its `price` drops 428.68 -> 246.27.

    So read a cluster here as "these two vendors disagree about a corporate action", check
    which one restated, and record the answer -- do not widen the tolerance."""
    frame = panel.dropna(subset=["close_split", "sharesOutstanding", "marketcap"]).copy()
    frame = frame[frame["marketcap"] > 0]
    frame["ratio"] = frame["close_split"] * frame["sharesOutstanding"] / frame["marketcap"]
    bad = (frame["ratio"] - 1).abs() > MCAP_TOLERANCE
    return InvariantResult(
        name="market_cap_identity", rows=int(len(frame)), failed=int(bad.sum()),
        tickers=int(frame["ticker"].nunique()),
        failing_tickers=_cluster(frame, bad, "ratio"))


def invariant_price_vintage(panel: pd.DataFrame) -> InvariantResult:
    """INVARIANT 2 -- the two vendors' split-adjusted prices must agree on filing dates.

    Both are split-adjusted-only, so they agree to the cent when both are current (AAPL
    2020-07-31: 106.26 and 106.26; KO 2004-02-27: 24.98 and 24.98). A disagreement means one
    feed has not applied a restatement the other has -- which is a STALE VINTAGE, and the
    thing that goes wrong silently and retroactively.

    Live example this catches: MNST reads exactly 2.0 against Sharadar in every year from
    2015 to 2026, because Yahoo never back-adjusted it for the 2:1 split its OWN splits feed
    reports on 2026-08-11."""
    frame = panel.dropna(subset=["close_split", "price"]).copy()
    frame = frame[frame["price"] > 0]
    frame["ratio"] = frame["close_split"] / frame["price"]
    bad = (frame["ratio"] - 1).abs() > PRICE_TOLERANCE
    return InvariantResult(
        name="price_vintage", rows=int(len(frame)), failed=int(bad.sum()),
        tickers=int(frame["ticker"].nunique()),
        failing_tickers=_cluster(frame, bad, "ratio"))


def invariant_spike_revert(context: Context, tickers: list[str] | None = None) -> InvariantResult:
    """INVARIANT 3 -- a big jump that comes straight back, with no split to explain it.

    Corroboration is the whole point: a real 2:1 split DOES halve the quote, so the test is
    not "did the price move a lot" but "did it move a lot, come back, and is there no event
    on the books". Reads the full price history, so it is the expensive one."""
    where = {"ticker": tickers} if tickers else None
    px = context.store.load(Tables.prices, columns=["ticker", "date", "close_split"],
                            where=where)
    px = _as_ns(px, "date").sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker")["close_split"].pct_change(fill_method=None)
    pre_jump = px.groupby("ticker")["close_split"].shift(1)
    ahead = [(px.groupby("ticker")["close_split"].shift(-i) / pre_jump - 1).abs()
             for i in range(1, SPIKE_REVERT_BARS + 1)]
    px["revert_gap"] = pd.concat(ahead, axis=1).min(axis=1)
    hit = px[(px["ret"].abs() > SPIKE_THRESHOLD) & (px["revert_gap"] < SPIKE_REVERT_BAND)]

    splits = context.store.load(Tables.prices_splits, columns=["ticker", "date"],
                                where=where, optional=True)
    known: set[tuple[str, pd.Timestamp]] = set()
    if splits is not None and not splits.empty:
        splits = _as_ns(splits, "date")
        for ticker, when in zip(splits["ticker"], splits["date"]):
            for offset in range(-SPLIT_MATCH_DAYS, SPLIT_MATCH_DAYS + 1):
                known.add((str(ticker), when + pd.Timedelta(days=offset)))

    detail, failing = [], {}
    for row in hit.sort_values(["date", "ticker"]).itertuples():
        corroborated = (str(row.ticker), row.date) in known
        record = {"ticker": str(row.ticker), "date": row.date.strftime("%Y-%m-%d"),
                  "ret": round(float(row.ret), 4),
                  "revert_gap": round(float(row.revert_gap), 4),
                  "corroborated_by_split": corroborated}
        detail.append(record)
        if not corroborated:
            failing.setdefault(str(row.ticker), {"rows": 0, "dates": []})
            failing[str(row.ticker)]["rows"] += 1
            failing[str(row.ticker)]["dates"].append(record["date"])

    return InvariantResult(
        name="spike_and_revert", rows=int(len(px)),
        failed=sum(v["rows"] for v in failing.values()),
        tickers=int(px["ticker"].nunique()), failing_tickers=failing, detail=detail)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class PricesReport:
    """All three invariants, plus the one number a gate can read."""
    invariants: list[InvariantResult]

    def worst_share(self) -> float:
        """The largest failing-row share across the invariants -- what the DoD gate reads."""
        return max((r.share for r in self.invariants), default=0.0)

    def to_markdown(self) -> str:
        lines = ["# Prices adjustment-basis validation", "",
                 "Read-only. Sharadar's own `marketcap` is internally inconsistent across "
                 "spinoffs, so a finding here names a vendor DISAGREEMENT to settle, not a "
                 "value to overwrite.", ""]
        for res in self.invariants:
            lines += [f"## {res.name}", "", res.summary(), ""]
            if not res.failing_tickers:
                lines += ["No failures.", ""]
                continue
            first = next(iter(res.failing_tickers.values()))
            if "median_ratio" in first:
                lines += ["| ticker | rows | median | min | max | from | to |",
                          "|---|---|---|---|---|---|---|"]
                lines += [f"| {t} | {v['rows']} | {v['median_ratio']} | {v['min_ratio']} "
                          f"| {v['max_ratio']} | {v['first_date']} | {v['last_date']} |"
                          for t, v in list(res.failing_tickers.items())[:40]]
            else:
                lines += ["| ticker | jumps | dates |", "|---|---|---|"]
                lines += [f"| {t} | {v['rows']} | {', '.join(v['dates'][:6])} |"
                          for t, v in res.failing_tickers.items()]
            lines.append("")
        return "\n".join(lines)


#: The BLOCKING threshold for invariant 3, measured on the post-fix table (2026-09-01):
#: 10 unexplained jumps in 3,263,459 rows, i.e. 3e-6, and every one of them a named,
#: understood case (MNST's live Yahoo defect plus four genuine GFC/2021 round-trips).
#: 1e-4 is ~30x that headroom -- enough that a handful of new vendor hiccups warn rather than
#: halt the nightly, tight enough that a systemic re-corruption cannot slip through.
SPIKE_BLOCK_SHARE = 1e-4
#: Invariant 1 does NOT block, and the reason is measured, not squeamish: 12.6% of joined
#: rows fail it on a CORRECT table, because Yahoo back-adjusts prices for SPINOFFS and
#: Sharadar's `sharesbas` does not -- so `sharadar.marketcap` is itself internally
#: inconsistent for those names (HON, DD, GE, FDX, BDX...). Blocking on it would block every
#: build for a defect in the reference, not in the repo. It is reported and clustered so the
#: disagreements get settled one ticker at a time. Revisit only when that cluster is closed.
MCAP_BLOCK_SHARE = None


def gate(report: "PricesReport") -> tuple[bool, str]:
    """`(ok, reason)` for the pre-cube-build gate.

    Only invariant 3 blocks. It is the one whose failures are unambiguous -- an unexplained
    >50% round-trip with no split on the books is always a data fault -- and the one that
    catches the mechanism that re-corrupts the table every time a stock splits."""
    by_name = {r.name: r for r in report.invariants}
    spike = by_name.get("spike_and_revert")
    if spike is None:
        return True, "spike scan skipped -- nothing to gate on"
    if spike.share > SPIKE_BLOCK_SHARE:
        return False, (
            f"{spike.failed} unexplained spike-and-revert rows "
            f"({spike.share:.2e} > {SPIKE_BLOCK_SHARE:.0e}) across "
            f"{len(spike.failing_tickers)} ticker(s): "
            f"{', '.join(sorted(spike.failing_tickers)[:8])}. A >50% jump that comes back "
            f"with no corroborating row in `prices_splits` is two adjustment vintages inside "
            f"one ticker -- re-pull those tickers with `price-history --full` before building.")
    return True, f"{spike.failed} unexplained spike(s), within the {SPIKE_BLOCK_SHARE:.0e} budget"


def run_prices_validation(context: Context, tickers: list[str] | None = None,
                          since: str | pd.Timestamp | None = None,
                          skip_spike: bool = False) -> PricesReport:
    """Run all three invariants and return the report. Writes nothing."""
    panel = load_panel(context, tickers=tickers, since=since)
    results = [invariant_market_cap(panel), invariant_price_vintage(panel)]
    if not skip_spike:
        results.append(invariant_spike_revert(context, tickers=tickers))
    return PricesReport(invariants=results)
