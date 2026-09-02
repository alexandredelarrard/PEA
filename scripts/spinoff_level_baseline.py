"""
spinoff_level_baseline.py  (scripts/)
--------------------------------------------------------------------------------------------
Freeze the SPINOFF LEVEL-BASIS measurements as ONE rerunnable script, before and after the
fix, so every claim in the plan is checkable rather than asserted.

## The gap this measures

`scripts/basis_baseline.py` put price and share count on one basis for SPLITS. It could not
do so for SPINOFFS, and the residual has a closed form:

    S(d)  =  PROD{ prices_splits.ratio    : date > d }
           /  PROD{ split_events(...).value : date > d }

Yahoo back-adjusts `Close` for a spinoff (correct for RETURNS -- it keeps the series
continuous); Sharadar's `price` does not (correct for LEVELS -- it is what the stock traded
at); and a spinoff does not change the parent's share count, so nothing cancels.
`close_split x sharesOutstanding` UNDERSTATES market cap by exactly `S(d)`.

A RATIO OF TWO PRODUCTS, not a set subtraction. HON carries an event on 2026-06-29 in both
sources with different values (yfinance 0.9535, Sharadar 0.5) and only the ratio is right.

## What it emits

`before.json` / `before.md` (or `after-*`) with the invariant pass rates raw AND S-adjusted,
the per-ticker factor for both cohorts, the FDX-style market-cap table against Sharadar's own
`marketcap`, and the two OPEN QUESTIONS the plan refuses to guess at:

  * `dividend_yield` -- is yfinance's `Dividends` column back-adjusted for spinoffs the way
    the quote is? If it is, the two legs cancel and `dividend_features` needs no change.
  * `fwd_eps_yield` -- is yfinance's earnings history on the level basis or the adjusted one?

Both are answered by a RATIO AGAINST SHARADAR on the spinoff cohort versus the control
cohort: ~1.0 on both means the legs cancel; ~S on the spinoff cohort alone means they do not.

Read-only; touches no table.

    "$PY" scripts/spinoff_level_baseline.py [--out DIR] [--tag before]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.context import get_config_context
from src.data_extract.utils.fundamentals_sharadar.field_map import split_events
from src.data_store.schema import Tables
from src.validate.prices import (MCAP_TOLERANCE, PRICE_TOLERANCE, load_panel)

#: Names with a spinoff in their history and no `sharadar_actions` row to explain it -- the
#: cohort the whole plan exists for. Each one's `S` is a yfinance-only price factor.
SPINOFF_COHORT = ("FDX", "GE", "DD", "T", "HPQ", "EXC", "RTX", "NI", "BAX", "EQT")
#: Never spun anything off. `S` must be EXACTLY 1.0 here, and every number this script
#: reports for them must be bit-identical before and after the fix. This is the cohort that
#: proves the change is targeted rather than broad.
CONTROL_COHORT = ("AAPL", "KO", "JNJ", "MSFT", "PG", "XOM")
#: Dates the market-cap table is sampled at, one per pre/post era. Fixed so a rerun in 2027
#: diffs the same rows rather than silently sliding forward.
MCAP_SAMPLE_DATES = ("2005-06-30", "2012-06-29", "2018-06-29", "2021-06-30")
#: |S - 1| below which two event sets are the same set and the factor is exactly 1.0. Float
#: noise from multiplying and dividing the same ratios is ~1e-16; a genuine factor is >1e-3.
LEVEL_SNAP_TOL = 1e-12
#: Trading days in a dividend TTM window, matching `dividend_features._YOY`.
TTM_DAYS = 365
#: |S - 1| above which a row SEPARATES the two hypotheses of `_cohort_verdict`. At 10% the
#: "legs cancel" and "off by S" predictions are 10% apart, well clear of the few-percent
#: vendor-definition wedge; below it the two are indistinguishable and the row is only noise.
STRONG_FACTOR = 0.10
#: Strongly-affected rows below which the A/B test is not called at all. A verdict that
#: rewrites a feature must not rest on a handful of rows.
MIN_VERDICT_ROWS = 100
DEFAULT_OUT = ROOT / "reports/planning/active-tasks/2026-09-01-spinoff-level-basis"


# --------------------------------------------------------------------------- #
# the reference implementation of S(d)                                        #
# --------------------------------------------------------------------------- #
def level_factor(tickers: pd.Series, dates: pd.Series, yf_splits: pd.DataFrame,
                 genuine: pd.DataFrame) -> pd.Series:
    """`S(d)` per row: the price adjustment Yahoo applied that the share count did not.

    THE REFERENCE. `src/data_aggregate/utils/common/level_basis.py` computes the same thing
    wide (date x ticker) for the cube; this long form exists so the two can be diffed on the
    live panel and so Phase 0 can measure the gap before any of that code is written.

    Both products run over events dated STRICTLY AFTER the row's own date, matching
    `field_map.forward_split_factor`: a factor applied on date e restates every bar BEFORE e
    and leaves e itself alone.
    """
    factor = pd.Series(1.0, index=dates.index)
    stamps = pd.to_datetime(dates, errors="coerce")
    tick = tickers.astype(str)

    for frame, column, power in ((yf_splits, "ratio", 1.0), (genuine, "value", -1.0)):
        if frame is None or frame.empty:
            continue
        for _, event in frame.iterrows():
            value = float(event[column])
            if not np.isfinite(value) or value <= 0:
                continue
            hit = (tick == str(event["ticker"])) & (stamps < pd.Timestamp(event["date"]))
            factor.loc[hit] = factor.loc[hit] * (value ** power)

    # Snap, so a ticker whose two event sets AGREE is bit-identical to today rather than
    # 1.0000000000000002 -- which is what makes the control cohort's digests comparable.
    factor.loc[(factor - 1.0).abs() < LEVEL_SNAP_TOL] = 1.0
    return factor


def _digest(series: pd.Series) -> str:
    """Order-independent SHA-256 over a float series, rounded to 6 significant figures so an
    unrelated pandas last-ulp change does not read as a basis change. Same recipe as
    `scripts/basis_baseline.py`, so the two scripts' digests are comparable."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    return hashlib.sha256(",".join(f"{v:.6g}" for v in s).encode()).hexdigest()[:16]


def _as_ns(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Postgres DATE round-trips as `datetime64[s]`, TIMESTAMP as `datetime64[us]`, and
    `merge_asof` refuses to join two keys of different resolution."""
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    return frame


# --------------------------------------------------------------------------- #
# loads                                                                       #
# --------------------------------------------------------------------------- #
def load_events(store) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(yfinance splits, genuine splits)` -- the numerator and denominator of `S`.

    The denominator comes from `field_map.split_events`, the SAME function the extract layer
    de-adjusts share counts with. That is deliberate and is what stops the two legs drifting:
    a row that counts as a genuine split there must cancel here."""
    yf = store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"])
    yf = _as_ns(yf, "date").dropna(subset=["ratio"])
    yf = yf[yf["ratio"] > 0]

    actions = store.load(Tables.sharadar_actions, columns=["ticker", "date", "action", "value"],
                         where={"action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]})
    actions = _as_ns(actions, "date")
    return yf, split_events(actions, yf)


# --------------------------------------------------------------------------- #
# the invariants, raw and S-adjusted                                          #
# --------------------------------------------------------------------------- #
def invariant_rates(panel: pd.DataFrame) -> dict:
    """Invariants 1 and 2 from `src/validate/prices.py`, each scored twice: as the validator
    scores it today, and with `S` applied to the price leg.

    The pair is the whole point. Reporting only the adjusted rate would hide the size of the
    wedge; reporting only the raw one is what let 12.6% be written off as "the vendor's
    fault" for a month."""
    out = {}
    for name, ref, tol, legs in (
            ("market_cap_identity", "marketcap", MCAP_TOLERANCE,
             ["close_split", "sharesOutstanding", "marketcap"]),
            ("price_vintage", "price", PRICE_TOLERANCE, ["close_split", "price"])):
        frame = panel.dropna(subset=legs + ["level_factor"])
        frame = frame[frame[ref] > 0]
        if frame.empty:
            out[name] = {"rows": 0}
            continue
        raw = frame["close_split"] * frame.get("sharesOutstanding", 1.0) / frame[ref] \
            if name == "market_cap_identity" else frame["close_split"] / frame[ref]
        adj = raw * frame["level_factor"]
        raw_ok, adj_ok = (raw - 1).abs() <= tol, (adj - 1).abs() <= tol
        out[name] = {
            "rows": int(len(frame)),
            "raw_pass": int(raw_ok.sum()), "raw_rate": round(float(raw_ok.mean()), 4),
            "adj_pass": int(adj_ok.sum()), "adj_rate": round(float(adj_ok.mean()), 4),
            # The number the plan promises to keep at <=1: rows the fix BREAKS. A fix that
            # lifts the aggregate while quietly failing rows that used to pass is not a fix.
            "newly_failing": int((raw_ok & ~adj_ok).sum()),
            "newly_failing_tickers": sorted(frame.loc[raw_ok & ~adj_ok, "ticker"]
                                            .astype(str).unique().tolist())[:20],
            "newly_passing": int((~raw_ok & adj_ok).sum()),
        }
    return out


def residual_clusters(panel: pd.DataFrame, top: int = 20) -> dict:
    """Which tickers still fail invariant 1 AFTER `S`, biggest first.

    The plan names four out-of-scope clusters (MNST, V, the stock-dividend names, the as-of
    join noise). A FIFTH appearing here means something in the plan is wrong, so this table
    is the falsifier rather than a decoration."""
    frame = panel.dropna(subset=["close_split", "sharesOutstanding", "marketcap",
                                 "level_factor"])
    frame = frame[frame["marketcap"] > 0]
    ratio = (frame["close_split"] * frame["sharesOutstanding"] * frame["level_factor"]
             / frame["marketcap"])
    bad = frame[(ratio - 1).abs() > MCAP_TOLERANCE].assign(ratio=ratio)
    counts = bad.groupby("ticker").agg(rows=("ratio", "size"),
                                       median_ratio=("ratio", "median"))
    counts = counts.sort_values("rows", ascending=False).head(top)
    return {str(t): {"rows": int(r.rows), "median_ratio": round(float(r.median_ratio), 4)}
            for t, r in counts.iterrows()}


def cohort_factors(panel: pd.DataFrame) -> dict:
    """Per-ticker `S` for both cohorts. The control cohort's entries must all read
    `distinct=1, max=1.0, exactly_one=true` -- anything else and the snap is broken."""
    out = {}
    for label, cohort in (("spinoff", SPINOFF_COHORT), ("control", CONTROL_COHORT)):
        block = {}
        for ticker in cohort:
            s = panel.loc[panel["ticker"] == ticker, "level_factor"].dropna()
            if s.empty:
                block[ticker] = {"rows": 0}
                continue
            block[ticker] = {
                "rows": int(s.size),
                "distinct": int(s.round(6).nunique()),
                "max": round(float(s.max()), 6),
                "min": round(float(s.min()), 6),
                # `is` equality to 1.0, not approx: a control ticker at 1.0000000000000002
                # would move a digest, which is exactly the failure mode this catches.
                "exactly_one": bool((s == 1.0).all()),
                "rows_not_one": int((s != 1.0).sum()),
            }
        out[label] = block
    return out


def market_cap_table(panel: pd.DataFrame) -> dict:
    """Ours vs Sharadar's `marketcap` at four fixed dates for the spinoff cohort.

    An as-of pick (the last filing row at or before the sample date), because a filing rarely
    lands on a chosen calendar date and the cube's own market cap is forward-filled the same
    way."""
    out: dict[str, list[dict]] = {}
    for ticker in ("FDX", "GE", "DD", "T", "HPQ", "EXC", "RTX"):
        rows = panel[panel["ticker"] == ticker].dropna(
            subset=["close_split", "sharesOutstanding", "marketcap", "level_factor"])
        rows = rows[rows["marketcap"] > 0].sort_values("date")
        picks = []
        for when in MCAP_SAMPLE_DATES:
            hit = rows[rows["date"] <= pd.Timestamp(when)]
            if hit.empty:
                continue
            r = hit.iloc[-1]
            ours = float(r["close_split"] * r["sharesOutstanding"])
            picks.append({
                "asked": when, "date": str(pd.Timestamp(r["date"]).date()),
                "S": round(float(r["level_factor"]), 6),
                "ours_bn": round(ours / 1e9, 3),
                "fixed_bn": round(ours * float(r["level_factor"]) / 1e9, 3),
                "sharadar_bn": round(float(r["marketcap"]) / 1e9, 3),
                "err_today": round(ours / float(r["marketcap"]) - 1.0, 4),
                "err_fixed": round(ours * float(r["level_factor"])
                                   / float(r["marketcap"]) - 1.0, 4)})
        out[ticker] = picks
    return out


def fdx_landmark(panel: pd.DataFrame) -> dict:
    """The plan's headline row, verbatim: FDX at its 2020-12-17 filing. $62.4bn today,
    $77.5bn after. If this one number does not move, nothing else in the report matters."""
    rows = panel[(panel["ticker"] == "FDX")
                 & (panel["date"] == pd.Timestamp("2020-12-17"))]
    if rows.empty:
        return {"found": False}
    r = rows.iloc[0]
    ours = float(r["close_split"] * r["sharesOutstanding"])
    return {"found": True, "date": "2020-12-17",
            "close_split": round(float(r["close_split"]), 4),
            "sharadar_price": round(float(r["price"]), 4),
            "shares": int(r["sharesOutstanding"]),
            "S": round(float(r["level_factor"]), 6),
            "ours_bn": round(ours / 1e9, 3),
            "fixed_bn": round(ours * float(r["level_factor"]) / 1e9, 3),
            "sharadar_bn": round(float(r["marketcap"]) / 1e9, 3)}


# --------------------------------------------------------------------------- #
# the two open questions                                                      #
# --------------------------------------------------------------------------- #
def dividend_leg_question(store, panel: pd.DataFrame) -> dict:
    """OPEN QUESTION 1 -- does yfinance back-adjust `Dividends` for spinoffs?

    `dividend_features` computes `ttm_ps / close_split`. If Yahoo divides BOTH legs by the
    spinoff factor the yield is unchanged and there is nothing to fix; if it divides only the
    quote, the yield is `S` times too high.

    The discriminator is Sharadar's own `dps / price`, which is unambiguously on the LEVEL
    basis. `ratio = (ttm_ps/close_split) / (dps/price)` reads ~1.0 on the control cohort by
    construction. On the spinoff cohort ~1.0 means the legs cancel (no change needed) and ~S
    means they do not.
    """
    div = store.load(Tables.dividends, columns=["ticker", "date", "dividends"], optional=True)
    vendor = store.load(Tables.sharadar_fundamentals,
                        columns=["ticker", "date", "dimension", "dps", "price"],
                        where={"dimension": "ART"})
    if div is None or div.empty or vendor is None or vendor.empty:
        return {"skipped": "prices_dividends or the ART dimension is empty"}
    div = _as_ns(div, "date")
    vendor = _as_ns(vendor, "date")
    vendor = vendor[(vendor["dps"] > 0) & (vendor["price"] > 0)]

    # TTM per share at each filing date: the ex-date sum over the preceding 365 days, which
    # is what `_ttm_dividends`' 252-bar rolling window approximates on the trading grid.
    px = panel[["ticker", "date", "close_split", "level_factor"]].dropna()
    keys = px.merge(vendor[["ticker", "date", "dps", "price"]], on=["ticker", "date"],
                    how="inner")
    if keys.empty:
        return {"skipped": "no (ticker, date) overlap between the panel and the ART frame"}

    ttm = []
    by_ticker = {t: g.sort_values("date") for t, g in div.groupby("ticker")}
    for row in keys.itertuples():
        g = by_ticker.get(row.ticker)
        if g is None:
            ttm.append(np.nan)
            continue
        window = g[(g["date"] > row.date - pd.Timedelta(days=TTM_DAYS))
                   & (g["date"] <= row.date)]
        ttm.append(float(window["dividends"].sum()) if not window.empty else np.nan)
    keys["ttm_ps"] = ttm
    keys = keys[keys["ttm_ps"] > 0]
    keys["ours"] = keys["ttm_ps"] / keys["close_split"]
    keys["theirs"] = keys["dps"] / keys["price"]
    keys["ratio"] = keys["ours"] / keys["theirs"]
    return _cohort_verdict(keys, "dividend_yield (yfinance ttm_ps/close_split) / "
                                 "(sharadar dps/price)")


def earnings_leg_question(store, panel: pd.DataFrame) -> dict:
    """OPEN QUESTION 2 -- is yfinance's earnings history on the level basis?

    `earnings_features` computes `eps / close_split`. The EPS leg cannot be spinoff-adjusted
    by anything downstream of the vendor, so the test is whether the vendor did it. Scored as
    a YIELD ratio, exactly like the dividend leg, so the same A/B test applies:

        ours   = eps_actual / close_split          (what the cube computes)
        theirs = sharadar.epsdil / sharadar.price  (unambiguously the LEVEL basis)

    ⚠ Noisier than the dividend leg by construction: yfinance's `eps_actual` is the
    consensus-comparable (often non-GAAP) figure while `epsdil` is GAAP, so the two disagree
    by a definitional wedge on ~2/3 of rows in EVERY cohort. That wedge is basis-independent,
    which is why the test reads the MEDIAN and the affected-vs-unaffected DIFFERENCE rather
    than an agreement rate.
    """
    earn = store.load(Tables.earnings_surprises,
                      columns=["ticker", "earnings_date", "eps_actual"], optional=True)
    vendor = store.load(Tables.sharadar_fundamentals,
                        columns=["ticker", "date", "dimension", "epsdil"],
                        where={"dimension": "ARQ"})
    if earn is None or earn.empty:
        return {"skipped": "earnings_surprises is empty"}
    earn = _as_ns(earn, "earnings_date").rename(columns={"earnings_date": "date"})
    earn = earn.dropna(subset=["eps_actual"])
    vendor = _as_ns(vendor, "date")
    vendor = vendor[vendor["epsdil"] > 0.05]   # a near-zero or negative EPS makes it noise

    base = panel[["ticker", "date", "close_split", "price", "level_factor"]].dropna()
    base = base[base["price"] > 0].drop_duplicates(subset=["ticker", "date"])
    base = base.merge(vendor[["ticker", "date", "epsdil"]], on=["ticker", "date"], how="inner")

    # The earnings date and the filing date are the same event a few days apart, so the
    # earnings row is carried ONTO the filing row rather than the other way round -- the
    # filing row is where both a price and a share count exist.
    j = pd.merge_asof(base.sort_values("date"), earn.sort_values("date"),
                      on="date", by="ticker", direction="nearest",
                      tolerance=pd.Timedelta(days=10)).dropna(subset=["eps_actual"])
    j = j[j["eps_actual"] > 0.05]
    if j.empty:
        return {"skipped": "no earnings row could be matched to a Sharadar quarter"}
    j["ratio"] = (j["eps_actual"] / j["close_split"]) / (j["epsdil"] / j["price"])
    return _cohort_verdict(j, "fwd/trailing EPS yield (yfinance eps / close_split) / "
                              "(sharadar epsdil / price)")


def _cohort_verdict(frame: pd.DataFrame, what: str) -> dict:
    """Score a yield ratio on five cohorts and CALL IT, so Phase 3 reads a verdict rather
    than a table it has to interpret.

    THE A/B TEST. Both consumers compute `numerator / close_split`, and `close_split` is the
    true price divided by `S`. Two hypotheses, and they are cleanly separable:

      A  the vendor back-adjusted the numerator too -> the legs cancel  -> ratio == 1
      B  it did not                                 -> ours is S too big -> ratio == S

    So the discriminator is `median|log ratio|` against `median|log(ratio/S)|`, measured on
    the STRONGLY-AFFECTED rows (`|S-1| > 10%`) where the two hypotheses are far apart. Any
    definitional wedge between the two vendors (GAAP vs non-GAAP EPS, a TTM window that
    straddles an ex-date) inflates BOTH distances equally and cannot flip the call -- which
    is why a raw agreement rate is the wrong statistic here and the ratio of the two
    distances is the right one.
    """
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
    frame = frame[frame["ratio"] > 0]
    cohorts = {
        "control": frame["ticker"].isin(CONTROL_COHORT),
        "spinoff_cohort": frame["ticker"].isin(SPINOFF_COHORT),
        "affected": frame["level_factor"] != 1.0,
        "strongly_affected": (frame["level_factor"] - 1.0).abs() > STRONG_FACTOR,
        "unaffected": frame["level_factor"] == 1.0,
    }
    stats = {}
    for name, mask in cohorts.items():
        s = frame.loc[mask, "ratio"]
        if s.empty:
            stats[name] = {"n": 0}
            continue
        stats[name] = {"n": int(s.size), "median": round(float(s.median()), 4),
                       "p25": round(float(s.quantile(0.25)), 4),
                       "p75": round(float(s.quantile(0.75)), 4),
                       "within_2pct": round(float(((s - 1).abs() < 0.02).mean()), 4)}

    hit = frame[cohorts["strongly_affected"]]
    detail = {"n": int(len(hit))}
    verdict = (f"INDETERMINATE -- only {len(hit)} rows with |S-1| > {STRONG_FACTOR:.0%}, "
               "too few to separate the hypotheses")
    if len(hit) >= MIN_VERDICT_ROWS:
        d_cancel = float(np.abs(np.log(hit["ratio"])).median())
        d_broken = float(np.abs(np.log(hit["ratio"] / hit["level_factor"])).median())
        detail |= {"median_S": round(float(hit["level_factor"].median()), 4),
                   "median_ratio": round(float(hit["ratio"].median()), 4),
                   "dist_to_1": round(d_cancel, 4), "dist_to_S": round(d_broken, 4)}
        if d_cancel < d_broken:
            verdict = (f"LEGS CANCEL -- on the {len(hit):,} strongly-affected rows the ratio "
                       f"sits {d_broken / max(d_cancel, 1e-9):.1f}x closer to 1.0 than to S "
                       f"(median ratio {detail['median_ratio']} vs median S "
                       f"{detail['median_S']}). The vendor back-adjusted BOTH legs. "
                       "NO CHANGE NEEDED.")
        else:
            verdict = (f"LEGS DO NOT CANCEL -- on the {len(hit):,} strongly-affected rows the "
                       f"ratio sits {d_cancel / max(d_broken, 1e-9):.1f}x closer to S than to "
                       f"1.0 (median ratio {detail['median_ratio']} vs median S "
                       f"{detail['median_S']}). This consumer IS distorted and needs the "
                       "level factor.")
    return {"what": what, "cohorts": stats, "discriminator": detail, "verdict": verdict}


def cross_sectional_impact(panel: pd.DataFrame, buckets: int = 10) -> dict:
    """How many rows change RANK BUCKET once `S` is applied -- the number that sizes the
    model effect.

    A cross-sectional model does not read a market cap, it reads where a name sits relative
    to its peers ON THAT DATE. So the question is not "how much did the level move" (24% on
    FDX) but "how many names crossed a decile boundary", which is what actually changes a
    fitted split. Both are reported: the aggregate, and the share among the AFFECTED rows,
    because the second is what an affected name's own history experiences.

    Ranked WITHIN each `as_of` date, never pooled: a pooled decile would rank calendar time,
    since market caps grow.
    """
    frame = panel.dropna(subset=["close_split", "sharesOutstanding", "level_factor"]).copy()
    frame["before"] = frame["close_split"] * frame["sharesOutstanding"]
    frame["after"] = frame["before"] * frame["level_factor"]
    frame = frame[frame["before"] > 0]
    # A single-name date has no cross-section to move within.
    frame = frame[frame.groupby("as_of")["ticker"].transform("size") >= buckets]
    if frame.empty:
        return {"rows": 0}

    def bucket(column: str) -> pd.Series:
        return frame.groupby("as_of")[column].transform(
            lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))

    moved = bucket("before") != bucket("after")
    hit = frame["level_factor"] != 1.0
    return {"rows": int(len(frame)), "buckets": buckets,
            "changed_bucket": int(moved.sum()),
            "changed_share": round(float(moved.mean()), 4),
            "affected_rows": int(hit.sum()),
            "changed_share_among_affected": (
                round(float(moved[hit].mean()), 4) if hit.any() else 0.0),
            "changed_share_among_unaffected": (
                round(float(moved[~hit].mean()), 4) if (~hit).any() else 0.0)}


# --------------------------------------------------------------------------- #
# controls                                                                    #
# --------------------------------------------------------------------------- #
def return_controls(store) -> dict:
    """Digests that MUST NOT MOVE. `S` multiplies a LEVEL and never a RETURN, so any change
    here means the factor leaked into the return path.

    Taken from `prices` rather than `cube_part_prices` because the part table is a build
    behind (it still carries the pre-fix `close` column), and a control has to be measurable
    on both sides of the change. `cube_part_prices` is digested too when its columns exist."""
    px = store.load(Tables.prices, columns=["ticker", "date", "close_total"])
    px = _as_ns(px, "date").sort_values(["ticker", "date"])
    ret = px.groupby("ticker")["close_total"].pct_change(fill_method=None)

    out = {"source": "prices",
           "rows": int(len(px)),
           "close_total_digest": _digest(px["close_total"]),
           "ret_from_close_total_digest": _digest(ret)}

    probe = store.load(Tables.cube_part_prices, limit=1, optional=True)
    have = set(probe.columns) if probe is not None else set()
    part_cols = [c for c in ("close_split", "close_total", "ret", "volume") if c in have]
    if part_cols:
        part = store.load(Tables.cube_part_prices, columns=["ticker", "date"] + part_cols)
        out["cube_part_prices"] = {"rows": int(len(part)),
                                   "columns": sorted(have),
                                   **{f"{c}_digest": _digest(part[c]) for c in part_cols}}
    else:
        out["cube_part_prices"] = {"stale": True, "columns": sorted(have),
                                   "note": "no close_split/close_total/ret column yet -- the "
                                           "part table predates the 2026-09-01 basis fix"}
    return out


def factor_population(panel: pd.DataFrame, yf: pd.DataFrame,
                      genuine: pd.DataFrame) -> dict:
    """How much of the table `S` touches, and the biggest factors. `top` is ranked by
    `|log S|` so a 0.5 and a 2.0 are equally interesting."""
    s = panel[["ticker", "level_factor"]].dropna()
    off = s[s["level_factor"] != 1.0]
    per_ticker = off.groupby("ticker")["level_factor"].max()
    ranked = per_ticker.reindex(per_ticker.map(lambda v: abs(np.log(v)))
                                .sort_values(ascending=False).index).head(15)
    return {"panel_rows": int(len(s)), "rows_off_one": int(len(off)),
            "share_off_one": round(float(len(off) / len(s)), 4) if len(s) else 0.0,
            "tickers_off_one": int(off["ticker"].nunique()),
            "panel_tickers": int(s["ticker"].nunique()),
            "yf_split_rows": int(len(yf)), "genuine_split_rows": int(len(genuine)),
            "top_by_abs_log_S": {str(t): round(float(v), 6) for t, v in ranked.items()}}


# --------------------------------------------------------------------------- #
# report                                                                      #
# --------------------------------------------------------------------------- #
def to_markdown(blob: dict) -> str:
    env, inv = blob["env"], blob["invariants"]
    L = [f"# Spinoff level-basis baseline -- `{blob['tag']}`", "",
         f"Generated {blob['generated_utc']} from the live `pea` database by "
         "`scripts/spinoff_level_baseline.py`.", "",
         "`S(d) = PROD(prices_splits.ratio after d) / PROD(split_events(...).value after d)` "
         "-- the price adjustment Yahoo applied that the share count did not.", "",
         f"**Environment**: {env['panel_rows']:,} joined filing rows / "
         f"{env['panel_tickers']} tickers; {env['yf_split_rows']} yfinance split rows, "
         f"{env['genuine_split_rows']} genuine ones.", "",
         "## invariants -- raw vs S-adjusted", "",
         "| invariant | rows | raw pass | S-adjusted pass | newly passing | newly FAILING |",
         "|---|---|---|---|---|---|"]
    for name, v in inv.items():
        if not v.get("rows"):
            continue
        L.append(f"| `{name}` | {v['rows']:,} | {v['raw_rate']:.2%} | {v['adj_rate']:.2%} "
                 f"| +{v['newly_passing']:,} | {v['newly_failing']} "
                 f"({', '.join(v['newly_failing_tickers']) or 'none'}) |")

    f = blob["fdx_landmark"]
    if f.get("found"):
        L += ["", "## FDX 2020-12-17 -- the landmark row", "",
              f"`close_split` {f['close_split']}, Sharadar `price` {f['sharadar_price']}, "
              f"S = {f['S']}, shares {f['shares']:,}.", "",
              f"| ours today | ours x S | Sharadar |", "|---|---|---|",
              f"| ${f['ours_bn']}bn | **${f['fixed_bn']}bn** | ${f['sharadar_bn']}bn |"]

    L += ["", "## market cap vs Sharadar, spinoff cohort", "",
          "| ticker | date | S | ours ($bn) | ours x S | Sharadar | err today | err fixed |",
          "|---|---|---|---|---|---|---|---|"]
    for ticker, picks in blob["market_cap_table"].items():
        for p in picks:
            L.append(f"| {ticker} | {p['date']} | {p['S']} | {p['ours_bn']} "
                     f"| {p['fixed_bn']} | {p['sharadar_bn']} | {p['err_today']:.2%} "
                     f"| {p['err_fixed']:.2%} |")

    L += ["", "## per-ticker S", ""]
    for label, block in blob["cohort_factors"].items():
        L += [f"### {label} cohort", "",
              "| ticker | rows | rows S!=1 | min S | max S | exactly 1.0 |",
              "|---|---|---|---|---|---|"]
        for t, v in block.items():
            if not v.get("rows"):
                L.append(f"| {t} | 0 | - | - | - | - |")
                continue
            L.append(f"| {t} | {v['rows']} | {v['rows_not_one']} | {v['min']} | {v['max']} "
                     f"| {'YES' if v['exactly_one'] else '**NO**'} |")
        L.append("")

    p = blob["factor_population"]
    L += ["## how much S touches", "",
          f"{p['rows_off_one']:,} of {p['panel_rows']:,} panel rows ({p['share_off_one']:.2%}) "
          f"across {p['tickers_off_one']} of {p['panel_tickers']} tickers.", "",
          "| ticker | max S |", "|---|---|"]
    L += [f"| {t} | {v} |" for t, v in p["top_by_abs_log_S"].items()]

    L += ["", "## residual after S -- invariant 1's biggest remaining clusters", "",
          "The plan scopes out four: MNST, V, the stock-dividend names (APA/HBAN/ORCL) and "
          "the as-of join noise. **A fifth here means the plan is wrong.**", "",
          "| ticker | rows | median ratio |", "|---|---|---|"]
    L += [f"| {t} | {v['rows']} | {v['median_ratio']} |"
          for t, v in blob["residual_clusters"].items()]

    L += ["", "## the two open questions", ""]
    for key in ("dividend_leg", "earnings_leg"):
        q = blob[key]
        if q.get("skipped"):
            L += [f"### `{key}` -- SKIPPED: {q['skipped']}", ""]
            continue
        L += [f"### `{key}` -- {q['what']}", "",
              f"**{q['verdict']}**", "",
              f"discriminator on the rows with `|S-1| > {STRONG_FACTOR:.0%}`: "
              f"`{q['discriminator']}`", "",
              "| cohort | n | median | p25 | p75 | within 2% of 1.0 |",
              "|---|---|---|---|---|---|"]
        for c, v in q["cohorts"].items():
            if not v.get("n"):
                L.append(f"| {c} | 0 | - | - | - | - |")
                continue
            L.append(f"| {c} | {v['n']:,} | {v['median']} | {v['p25']} | {v['p75']} "
                     f"| {v['within_2pct']:.2%} |")
        L.append("")

    x = blob["cross_sectional_impact"]
    if x.get("rows"):
        L += ["", "## cross-sectional impact -- what the MODEL sees", "",
              f"A cross-sectional model reads a name's RANK, not its level. Of "
              f"{x['rows']:,} scored rows, **{x['changed_bucket']:,} "
              f"({x['changed_share']:.2%})** change size decile once `S` is applied: "
              f"**{x['changed_share_among_affected']:.2%}** of the {x['affected_rows']:,} "
              f"rows with `S != 1`, and {x['changed_share_among_unaffected']:.2%} of the "
              f"rest (which move only because their peers did).", ""]

    c = blob["return_controls"]
    L += ["## return controls -- MUST NOT MOVE", "",
          "| digest | value |", "|---|---|",
          f"| `prices.close_total` | `{c['close_total_digest']}` |",
          f"| `ret` from `close_total` | `{c['ret_from_close_total_digest']}` |"]
    part = c["cube_part_prices"]
    if part.get("stale"):
        L += [f"", f"> `cube_part_prices` is a build behind -- columns "
                   f"`{', '.join(part['columns'])}`. {part['note']}.", ""]
    else:
        L += [f"| `cube_part_prices.{k[:-7]}` | `{v}` |"
              for k, v in part.items() if k.endswith("_digest")]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="directory for the report pair")
    ap.add_argument("--tag", default="before", help="label, e.g. before / after / after-p1")
    args = ap.parse_args()

    _, context = get_config_context("./configs", use_cache=False, save=False)
    store = context.store

    yf, genuine = load_events(store)
    panel = load_panel(context)
    panel["level_factor"] = level_factor(panel["ticker"], panel["date"], yf, genuine)

    blob = {
        "tag": args.tag,
        # The ONLY nondeterministic field. Every measurement below must be byte-identical
        # across two runs of the same code against the same tables.
        "generated_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "env": {"panel_rows": int(len(panel)),
                "panel_tickers": int(panel["ticker"].nunique()),
                "yf_split_rows": int(len(yf)),
                "genuine_split_rows": int(len(genuine))},
        "invariants": invariant_rates(panel),
        "fdx_landmark": fdx_landmark(panel),
        "market_cap_table": market_cap_table(panel),
        "cohort_factors": cohort_factors(panel),
        "factor_population": factor_population(panel, yf, genuine),
        "residual_clusters": residual_clusters(panel),
        "cross_sectional_impact": cross_sectional_impact(panel),
        "dividend_leg": dividend_leg_question(store, panel),
        "earnings_leg": earnings_leg_question(store, panel),
        "return_controls": return_controls(store),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    (out / f"{args.tag}.md").write_text(to_markdown(blob), encoding="utf-8")

    inv = blob["invariants"]
    m, v = inv["market_cap_identity"], inv["price_vintage"]
    ctrl = blob["cohort_factors"]["control"]
    dirty = [t for t, x in ctrl.items() if x.get("rows") and not x["exactly_one"]]

    print(f"\nwrote {out / (args.tag + '.json')} and {out / (args.tag + '.md')}")
    print(f"invariant 1  raw {m['raw_rate']:.2%} -> S-adjusted {m['adj_rate']:.2%}  "
          f"(+{m['newly_passing']:,} pass, {m['newly_failing']} newly FAIL)")
    print(f"invariant 2  raw {v['raw_rate']:.2%} -> S-adjusted {v['adj_rate']:.2%}  "
          f"(+{v['newly_passing']:,} pass, {v['newly_failing']} newly FAIL)")
    print(f"S != 1 on {blob['factor_population']['rows_off_one']:,} rows / "
          f"{blob['factor_population']['tickers_off_one']} tickers")
    print(f"FDX 2020-12-17: ${blob['fdx_landmark'].get('ours_bn')}bn today -> "
          f"${blob['fdx_landmark'].get('fixed_bn')}bn fixed, "
          f"Sharadar ${blob['fdx_landmark'].get('sharadar_bn')}bn")
    print(f"\ndividend_leg: {blob['dividend_leg'].get('verdict', blob['dividend_leg'])}")
    print(f"earnings_leg: {blob['earnings_leg'].get('verdict', blob['earnings_leg'])}")
    if dirty:
        print(f"\n[FAIL] control cohort has S != 1.0 on {dirty} -- the snap is broken and "
              "the change is NOT targeted. STOP.")
    else:
        print(f"\n[OK] S == 1.0 exactly on all {len(ctrl)} control tickers "
              "-- the factor is confined to the spinoff names.")


if __name__ == "__main__":
    main()
