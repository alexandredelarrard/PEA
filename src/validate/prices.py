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

READ-ONLY, always -- `S(d)` is computed here IN MEMORY and never written. The stored copy
lives on `cube_part_prices`, produced by `StepCubePrices`; this module recomputes it from the
same two tables through the same function so the validator can run against a database whose
cube has not been rebuilt yet, and so a validator run can never be the thing that changes a
number it is about to judge.

## The 12.6% was NOT the vendor's fault, and this file used to say it was

⚠ This docstring previously argued that invariant 1 fails on 12.6% of rows "even on a CORRECT
table, because Sharadar is the inconsistent side". **That was wrong**, and it cost a month:
the failure was real, it was ours, and it was measurable. The decomposition, taken on the live
panel, puts it beyond doubt:

    leg_shares  (sharesOutstanding / sharesbas)     100.00%   the share leg is exact
    leg_vendor  (sharadar.price x sharesbas / mcap)  99.82%   the vendor is self-consistent
    leg_price   (close_split / sharadar.price)       87.59%   OURS is the leg that moves

The missing factor is `S(d)` -- the SPINOFF adjustment Yahoo applies to `Close` and nobody
applies to a share count. Multiplying the price leg by it takes invariant 1 from **87.44% to
98.30%** and invariant 2 from **87.33% to 98.18%**, and breaks 3 rows while fixing 5,516. See
`src/data_aggregate/utils/common/level_basis.py` for the derivation.

`configs/prices/yf_price_bugfix.json` then takes them to **99.03%** and **98.88%**: nine
tickers where the defect is in Yahoo's own data and no event feed expresses it, so `S` is
structurally blind to them. Applied here too -- the register is a SOURCE, not a cube artifact
-- and every entry is re-measured against Sharadar before it fires.

The lesson is the reusable part: "the reference is inconsistent" is the most comfortable
possible explanation for a failing invariant, and it must be the LAST one accepted, after the
identity has been decomposed leg by leg.

## What blocks, and why only one of them

`gate()` blocks the cube build on INVARIANT 3 ALONE, and that is still right -- but for a
different reason than the one written here before. Invariant 1's residual is no longer a
12.6% wall; it is ~1.0%, and the largest single contributor left is JCI (82 rows), whose
Yahoo series is corrupt in a way no factor can express. MNST's vintage and the stock-dividend
names are repaired by the register; Visa's multi-class `sharesbas` and as-of join noise are
the rest.
Raising a gate on it is now a defensible decision rather than an impossible one -- but it is a
SEPARATE decision, so `MCAP_BLOCK_SHARE` stays `None`. Invariant 3 keeps blocking because its
failures have no ambiguity: an unexplained >50% round-trip with no split on the books is
always a data fault, and it is the mechanism that silently re-corrupts the table every time a
stock splits.

## The three invariants

1. THE MARKET-CAP IDENTITY (reported, not blocking -- see above)
       | close_split(d) x S(d) x sharesOutstanding(d) / sharadar.marketcap(d) - 1 |  <  1%
   Two independent vendors, one arithmetic identity. Both legs carry the same retroactive
   SPLIT restatement so the split factor cancels; `S(d)` supplies the SPINOFF factor that does
   not. Reported BOTH raw and S-adjusted, so the size of the wedge stays visible instead of
   being absorbed into a headline rate.

2. PRICE VINTAGE FRESHNESS
       | close_split(d) x S(d) / sharadar.price(d) - 1 |  <  0.5%     on filing dates
   With `S` applied the two feeds are on the same basis, so what remains is a genuine STALE
   ADJUSTMENT VINTAGE rather than a convention difference. This is the check that would have
   flagged MNST in July 2026 instead of an audit finding it in September.

   ⚠ For the eight tickers carrying a registered LEVEL wedge this is partly SELF-REFERENTIAL,
   because the wedge was measured against the same `sharadar.price` it scores against. The
   independent alternative is the spun-off child's own price via `sharadar_actions
   .contraticker`, which needs history for securities outside the universe.

3. SPIKE-AND-REVERT
   ⚠ THE ONLY INVARIANT THAT STILL SCORES THE RAW TABLE, deliberately. It is the tripwire for
   vendor defects nobody has looked at yet, so it must keep firing on MNST's six 2026 jumps
   even though the register repairs them downstream -- that firing is the observation the
   register's own re-verification depends on. A cluster here means "the raw feed is broken",
   not "the cube is wrong"; cross-check `yf_price_bugfix.json` before acting on one.

   A >50% jump whose LEVEL comes back within a few bars, with no corroborating row in
   `prices_splits`, is two adjustment vintages meeting inside one ticker -- not a market
   event. Genuine moves must pass: 2020-03-09's oil crash (APA/OXY/FANG/TRGP), PCG's
   bankruptcy, CVNA 2022.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.context import Context
import logging

from src.data_aggregate.utils.common.level_basis import (
    apply_level_bugfix, apply_return_seams, apply_split_vintage, genuine_splits, level_factor,
    load_bugfix)
from src.data_store.schema import Tables

#: Invariant 1's band. 1% absorbs the as-of join (a filing date is often not a trading day)
#: and Sharadar's four-significant-figure rounding, and is an order of magnitude tighter than
#: any real basis error -- those are integer split factors (2x, 4x, 20x) or a dividend factor
#: that reached 0.62 in 2003.
logger = logging.getLogger(__name__)

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
    #: Failures WITHOUT `S(d)` applied, i.e. what this invariant scored before the spinoff
    #: level fix. Reported beside the corrected number on purpose: a single headline rate
    #: would hide the size of the wedge, and hiding it is how "12.6% is just the vendor"
    #: survived for a month. `None` on invariants that never read `S`.
    raw_failed: int | None = None

    @property
    def share(self) -> float:
        return self.failed / self.rows if self.rows else 0.0

    @property
    def raw_share(self) -> float | None:
        if self.raw_failed is None or not self.rows:
            return None
        return self.raw_failed / self.rows

    def summary(self) -> str:
        line = (f"{self.name}: {self.rows - self.failed:,}/{self.rows:,} pass "
                f"({1 - self.share:.2%}); {len(self.failing_tickers)} of {self.tickers} "
                f"tickers affected")
        if self.raw_share is not None:
            line += f" [without S(d): {1 - self.raw_share:.2%}]"
        return line


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

    prices = _repair_registered(context, prices, vendor)

    merged = context.store.load(Tables.fundamentals_history,
                                columns=["ticker", "as_of", "sharesOutstanding"], where=where)
    merged = _as_ns(merged, "as_of")

    panel = vendor.merge(merged, left_on=["ticker", "date"], right_on=["ticker", "as_of"],
                         how="left")
    panel = pd.merge_asof(panel.sort_values("date"), prices.sort_values("date"),
                          on="date", by="ticker", direction="backward",
                          tolerance=pd.Timedelta(days=ASOF_TOLERANCE_DAYS))
    panel["level_factor"] = _level_factor_for(context, panel, where)
    return panel


def _repair_registered(context: Context, prices: pd.DataFrame,
                       vendor: pd.DataFrame) -> pd.DataFrame:
    """Apply the two RETURN-MOVING entries of `configs/prices/yf_price_bugfix.json`.

    ⚠ THIS DOES NOT COMPROMISE THE INDEPENDENCE ABOVE. The register is a SOURCE -- a curated
    third statement about the vendors, re-verified against Sharadar on every use -- in exactly
    the way `split_events`' corroboration rules are, and this module already shares those. What
    it must not do is read a number back out of `cube_part_prices`, and it still does not.

    Without it the validator scores a `prices` table nothing downstream reads, and reports IP
    and MNST as broken when the cube has them right -- which is the worse failure, because it
    trains the next reader to ignore the report.

    Only the registered tickers are pivoted (ten of them), so this costs nothing next to the
    ~400 MB a full wide pivot would.
    """
    blob = load_bugfix(context.config_dir)
    named = sorted({*(blob.get("split_vintage") or {}), *(blob.get("return_seams") or {})})
    named = [t for t in named if t in set(prices["ticker"])]
    if not named:
        return prices

    slice_ = prices[prices["ticker"].isin(named)]
    wide = {"close_split": slice_.pivot(index="date", columns="ticker",
                                        values="close_split").sort_index()}
    apply_split_vintage(wide, blob, vendor[["ticker", "date", "price"]], logger.info)
    apply_return_seams(wide, blob, logger.info)

    repaired = (wide["close_split"].stack(future_stack=True).rename("close_split")
                .reset_index().dropna(subset=["close_split"]))
    return pd.concat([prices[~prices["ticker"].isin(named)], repaired], ignore_index=True)


def _level_factor_for(context: Context, panel: pd.DataFrame,
                      where: dict | None) -> pd.Series:
    """`S(d)` for each panel row, computed IN MEMORY. Writes nothing, ever (D6).

    Deliberately NOT read from `cube_part_prices.level_factor`, even though that column
    exists and holds the same numbers. Two reasons, both about what a validator is for:

      * it must run on a database whose cube is a build behind -- which is exactly the state
        a validator is most useful in;
      * reading the value the cube computed would make this a check that the cube agrees with
        itself. Recomputing from `prices_splits` + `sharadar_actions` through the same shared
        function keeps it a check against the two SOURCES.

    Wide-then-lookup rather than a per-row loop: `level_factor` is vectorised over a
    (date x ticker) grid, and the panel is ~52k rows over ~500 tickers.
    """
    yf_splits = context.store.load(Tables.prices_splits,
                                   columns=["ticker", "date", "ratio"],
                                   where=where, optional=True)
    actions = context.store.load(
        Tables.sharadar_actions, columns=["ticker", "date", "action", "value"],
        where={**(where or {}), "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]},
        optional=True)

    tickers = sorted(panel["ticker"].astype(str).unique())
    idx = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()), name="date")
    if idx.empty:
        return pd.Series(1.0, index=panel.index)
    wide = level_factor(idx, tickers, yf_splits, genuine_splits(actions, yf_splits))
    # The registered LEVEL wedges are the cases `S` is structurally blind to -- Yahoo adjusted
    # the price and its splits feed never said so. ⚠ For those tickers invariant 2 is partly
    # SELF-REFERENTIAL, because the wedge was measured against the same `sharadar.price` the
    # invariant scores against. It is 8 tickers and it is stated in every one of their
    # register entries; the alternative is scoring a basis the cube does not use.
    close_wide = panel.pivot_table(index="date", columns="ticker", values="close_split")
    wide = apply_level_bugfix(wide, load_bugfix(context.config_dir),
                              panel[["ticker", "date", "price"]],
                              close_wide.reindex(index=idx, columns=tickers), logger.info)

    flat = wide.stack(future_stack=True).rename("level_factor")
    flat.index = flat.index.set_names(["date", "ticker"])
    keys = pd.MultiIndex.from_arrays([panel["date"], panel["ticker"].astype(str)])
    # `fillna(1.0)`: a row whose (date, ticker) fell outside the grid gets NO adjustment,
    # never a NaN -- a NaN here would silently drop the row from every invariant below.
    return flat.reindex(keys).fillna(1.0).to_numpy()


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

    Sharadar's `marketcap` is `price x sharesbas`, and its `price` is adjusted for splits
    ONLY while Yahoo's `close_split` is adjusted for splits AND SPINOFFS. `sharesbas` carries
    the split restatement and no spinoff one, so the price leg carries a factor nothing else
    does. `S(d)` supplies it, and with it the identity closes: 87.44% -> 98.30%.

    ⚠ A SURVIVING FAILURE IS STILL NOT AUTOMATICALLY THIS REPO'S FAULT, and the validator
    must never auto-correct on one. Visa's cluster is a genuine vendor defect (`sharesbas` is
    Class A while `marketcap` is as-converted). So read a cluster as "these two vendors
    disagree about a corporate action", check which one restated, and record the answer -- do
    not widen the tolerance to make it pass."""
    frame = panel.dropna(subset=["close_split", "sharesOutstanding", "marketcap"]).copy()
    frame = frame[frame["marketcap"] > 0]
    base = frame["close_split"] * frame["sharesOutstanding"] / frame["marketcap"]
    frame["ratio"] = base * frame["level_factor"].fillna(1.0)
    bad = (frame["ratio"] - 1).abs() > MCAP_TOLERANCE
    return InvariantResult(
        name="market_cap_identity", rows=int(len(frame)), failed=int(bad.sum()),
        tickers=int(frame["ticker"].nunique()),
        failing_tickers=_cluster(frame, bad, "ratio"),
        raw_failed=int(((base - 1).abs() > MCAP_TOLERANCE).sum()))


def invariant_price_vintage(panel: pd.DataFrame) -> InvariantResult:
    """INVARIANT 2 -- the two vendors' split-adjusted prices must agree on filing dates.

    ONCE `S(d)` HAS PUT THEM ON THE SAME BASIS they agree to the cent (AAPL 2020-07-31:
    106.26 and 106.26; KO 2004-02-27: 24.98 and 24.98). Without it the check conflates two
    completely different things -- a spinoff CONVENTION difference, which is expected and
    harmless, and a STALE VINTAGE, which is a fault. Applying `S` leaves only the second,
    which is the whole point: 87.33% -> 98.18%.

    Live example this catches: MNST reads exactly 2.0 against Sharadar in every year from
    2015 to 2026, because Yahoo never back-adjusted it for the 2:1 split its OWN splits feed
    reports on 2026-08-11. `S` does NOT rescue that one and must not -- the event is in both
    event sets, so it cancels to 1.0, and the residual is the genuine stale vintage."""
    frame = panel.dropna(subset=["close_split", "price"]).copy()
    frame = frame[frame["price"] > 0]
    base = frame["close_split"] / frame["price"]
    frame["ratio"] = base * frame["level_factor"].fillna(1.0)
    bad = (frame["ratio"] - 1).abs() > PRICE_TOLERANCE
    return InvariantResult(
        name="price_vintage", rows=int(len(frame)), failed=int(bad.sum()),
        tickers=int(frame["ticker"].nunique()),
        failing_tickers=_cluster(frame, bad, "ratio"),
        raw_failed=int(((base - 1).abs() > PRICE_TOLERANCE).sum()))


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
                 "Read-only. Invariants 1 and 2 apply `S(d)`, the spinoff LEVEL factor, to "
                 "the price leg IN MEMORY -- nothing is written. Each is reported twice: the "
                 "corrected rate, and in brackets what it scored WITHOUT `S`, so the size of "
                 "the spinoff wedge stays visible rather than being absorbed into a "
                 "headline.", "",
                 "A surviving finding names a vendor DISAGREEMENT to settle, not a value to "
                 "overwrite.", ""]
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
#: Invariant 1 does NOT block. ⚠ The reason USED to be "12.6% fails on a correct table
#: because Sharadar is the inconsistent side", and that reason was wrong -- see the module
#: docstring. With `S(d)` applied the identity closes at **98.30%** (measured 2026-09-01,
#: 50,762 joined rows), and the residual ~1.8% is four NAMED clusters:
#:
#:     MNST  122 rows  ratio 2.0     Yahoo never applied its own published 2026-08-11 split
#:     V      74 rows  ratio 0.94    `sharesbas` is Class A, `marketcap` is as-converted
#:     APA/SJM/HBAN/ORCL ~79 rows    stock dividends Sharadar's `price` ignores
#:     ~20 small names, <=102 rows   as-of join noise, median 2 rows/ticker
#:
#: So a gate here is now a DEFENSIBLE decision rather than an impossible one. It is still
#: `None`, because choosing the threshold is a separate decision with its own evidence, and
#: because two of those four clusters are open work whose row counts will move. Set it only
#: with a measured number and a stated reason -- never to make a cluster pass.
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
