"""
basis_baseline.py  (scripts/)
--------------------------------------------------------------------------------------------
Freeze the price / shares adjustment-basis measurements as ONE rerunnable script.

Every number the 2026-09-01 research measured, recomputed from the EXTRACT layer only --
`prices`, `fundamentals_history`, `fundamentals_sharadar`, `fundamentals_history_sec`,
`prices_macro`. Nothing here reads the cube: `cube_mcap` below REPLICATES the
`daily_market_cap` formula (close x sharesOutstanding, as-of-joined onto the filing date)
rather than reading a `marketCap` column, so the script runs whether or not a cube exists.

Rerun after each phase and diff the JSON:
  * `macro_equity_tr_digest` must be IDENTICAL after P1 -- the macro leg is total-return and
    must not flip basis with the equity leg.
  * `option_overhang_digest` must be IDENTICAL after P3 -- optionOverhang is split-invariant
    (both legs carry the same factor), so it is the control proving the shares change touched
    only what it should.

Read-only. Writes `baseline.json` + `baseline.md` next to the plan.

    "$PY" scripts/basis_baseline.py [--out DIR] [--tag before]
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

from src.context import get_config_context
from src.data_store.schema import Tables

#: Years the research tabulated. Kept explicit so a rerun in 2027 still diffs against the
#: same rows rather than silently gaining a column.
REPORT_YEARS = (1995, 1998, 2003, 2013, 2021, 2026)
#: |move| beyond which a reverting jump reads as an adjustment-vintage seam, not a market
#: event. 55% clears 2020-03-09's oil crash (APA/OXY/FANG/TRGP), PCG's bankruptcy and CVNA
#: 2022 -- every genuine move checked in the research.
SPIKE_THRESHOLD = 0.55
#: Trading bars allowed for the level to come BACK. A vintage seam is a plateau, not a
#: one-day tick: MNST sat at the new basis for six bars (2026-07-23 -> 07-31) before
#: flipping back, so a strict next-day test finds nothing. 7 bars catches all three seams.
SPIKE_REVERT_BARS = 7
#: How close the level must return to the PRE-jump close to count as reverted. 10% is wide
#: enough for the drift accumulated over those bars and tight enough that a genuine
#: round-trip has to be almost exact -- only FITB 2009 and HIG 2008 (GFC) qualify.
SPIKE_REVERT_BAND = 0.10
#: Agreement band for a share count against the SEC cover page. 3% absorbs the weeks between
#: the cover-page date and the filing date; a missing split is off by an integer factor.
SEC_AGREEMENT_TOL = 0.03
DEFAULT_OUT = ROOT / "reports/planning/active-tasks/2026-09-01-price-shares-basis-fix"


def _digest(series: pd.Series) -> str:
    """Order-independent SHA-256 over a float series, NaN-stable and float-repr-stable.

    Rounded to 6 significant figures before hashing: a bit-exact hash of float64 would flag a
    last-ulp difference from an unrelated pandas upgrade as a basis change."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    payload = ",".join(f"{v:.6g}" for v in s)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(s.size), "median": round(float(s.median()), 6),
            "mean": round(float(s.mean()), 6), "p05": round(float(s.quantile(0.05)), 6),
            "min": round(float(s.min()), 6), "max": round(float(s.max()), 6)}


def _has_column(store, table: str, column: str) -> bool:
    """Whether a column exists yet, so the same script runs before AND after the migration."""
    probe = store.load(table, limit=1)
    return column in probe.columns


def _as_ns(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Parse a date column to NANOSECOND resolution.

    Not cosmetic: Postgres DATE round-trips as `datetime64[s]` and TIMESTAMP as
    `datetime64[us]`, and `merge_asof` REFUSES to join two keys of different resolution.
    `as_of` is a DATE, `prices.date` a TIMESTAMP, so every join below needs this."""
    frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    return frame


# --------------------------------------------------------------------------- #
# loads                                                                       #
# --------------------------------------------------------------------------- #
def load_frames(store) -> dict[str, pd.DataFrame]:
    """The five extract tables, projected. `prices` is the only large read (3.3M rows), and
    it is needed whole -- the spike scan and the forward-12m leg both span the full history."""
    price_col = "close_split" if _has_column(store, "prices", "close_split") else "close"
    prices = store.load(Tables.prices, columns=["ticker", "date", price_col])
    prices = prices.rename(columns={price_col: "close"})
    prices = _as_ns(prices, "date")

    merged_cols = ["ticker", "as_of", "sharesOutstanding", "optionOverhang"]
    if _has_column(store, "fundamentals_history", "sharesOutstandingPit"):
        merged_cols.append("sharesOutstandingPit")
    merged = store.load(Tables.fundamentals_history, columns=merged_cols)
    merged = _as_ns(merged, "as_of")

    vendor = store.load(Tables.sharadar_fundamentals,
                        columns=["ticker", "date", "dimension", "sharesbas",
                                 "marketcap", "price"],
                        where={"dimension": "ARQ"})
    vendor = _as_ns(vendor, "date")

    sec = store.load(Tables.fundamentals_history_sec,
                     columns=["ticker", "as_of", "sharesOutstanding"])
    sec = _as_ns(sec, "as_of")

    macro = store.load(Tables.prices_macro, columns=["ticker", "date", "close"],
                       where={"ticker": "equity_tr"})
    return {"prices": prices, "merged": merged, "vendor": vendor, "sec": sec, "macro": macro}


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per filing: the merged share count, the vendor's, the price on both bases, and
    the forward-12m return -- the substrate every measurement below reads.

    The price leg is an AS-OF join, not an equality join: a filing date is often a weekend or
    holiday, and `daily_market_cap` ffills fundamentals onto the daily price grid, so the
    price a filing row sees is the last bar at or before it."""
    merged, vendor, prices = frames["merged"], frames["vendor"], frames["prices"]

    panel = merged.merge(vendor, left_on=["ticker", "as_of"], right_on=["ticker", "date"],
                         how="inner", suffixes=("", "_v"))
    px = prices.sort_values("date")
    panel = pd.merge_asof(panel.sort_values("as_of"), px.rename(columns={"date": "px_date"}),
                          left_on="as_of", right_on="px_date", by="ticker",
                          direction="backward")

    # forward 12m on the STORED price series -- whatever basis is in force. The cohort test
    # only needs a return, and both bases give the same one up to the dividend leg.
    fwd = px[["ticker", "date", "close"]].rename(
        columns={"date": "fwd_date", "close": "close_fwd"})
    panel["target_date"] = panel["as_of"] + pd.DateOffset(months=12)
    panel = pd.merge_asof(panel.sort_values("target_date"), fwd.sort_values("fwd_date"),
                          left_on="target_date", right_on="fwd_date", by="ticker",
                          direction="backward", tolerance=pd.Timedelta(days=10))

    panel["fwd_12m"] = panel["close_fwd"] / panel["close"] - 1.0
    panel["cube_mcap"] = panel["close"] * panel["sharesOutstanding"]
    panel["mcap_error"] = panel["cube_mcap"] / panel["marketcap"].replace(0, np.nan)
    panel["split_part"] = panel["sharesOutstanding"] / panel["sharesbas"].replace(0, np.nan)
    panel["dividend_part"] = panel["close"] / panel["price"].replace(0, np.nan)
    panel["residual"] = panel["mcap_error"] / (panel["split_part"] * panel["dividend_part"])
    panel["year"] = panel["as_of"].dt.year
    return panel


# --------------------------------------------------------------------------- #
# measurements                                                                #
# --------------------------------------------------------------------------- #
def mcap_error_by_year(panel: pd.DataFrame) -> dict:
    out = {}
    for year, g in panel.groupby("year"):
        e = g["mcap_error"].dropna()
        if e.empty:
            continue
        out[str(int(year))] = {"n": int(e.size), "median": round(float(e.median()), 4),
                               "p05": round(float(e.quantile(0.05)), 4),
                               "min": round(float(e.min()), 4),
                               "off_by_10pct": int(((e - 1).abs() > 0.10).sum())}
    return out


def error_decomposition(panel: pd.DataFrame) -> dict:
    """`mcap_error == split_part x dividend_part` must hold EXACTLY. The residual is the
    plan's foundation: if it is not 1.0000 in every year the decomposition is wrong."""
    out = {}
    for year, g in panel.groupby("year"):
        g = g.dropna(subset=["mcap_error", "split_part", "dividend_part"])
        if g.empty:
            continue
        out[str(int(year))] = {
            "n": int(len(g)),
            "split_part": round(float(g["split_part"].median()), 4),
            "dividend_part": round(float(g["dividend_part"].median()), 4),
            "product": round(float((g["split_part"] * g["dividend_part"]).median()), 4),
            "mcap_error": round(float(g["mcap_error"].median()), 4),
            "residual": round(float(g["residual"].median()), 4)}
    return out


def split_part_cohorts(panel: pd.DataFrame) -> dict:
    """The leak test: `split_part < 1` means "this stock splits AFTER the observation date",
    strictly future information sitting in the mcap denominator."""
    g = panel.dropna(subset=["split_part", "fwd_12m"])
    cohorts = {"lt_1": g["split_part"] < 0.9999,
               "eq_1": g["split_part"].between(0.9999, 1.0001),
               "gt_1": g["split_part"] > 1.0001}
    return {name: {"n": int(mask.sum()),
                   "mean_fwd_12m": round(float(g.loc[mask, "fwd_12m"].mean()), 4),
                   "median_fwd_12m": round(float(g.loc[mask, "fwd_12m"].median()), 4)}
            for name, mask in cohorts.items() if mask.any()}


def _xs_quintile(g: pd.DataFrame, column: str) -> pd.Series:
    """CROSS-SECTIONAL quintile within each as_of date. Cross-sectional, not pooled, because
    both `dividend_part` and `mcap_error` decay monotonically with age -- a pooled quintile
    would rank calendar time, not the defect."""
    return g.groupby("as_of")[column].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1 if s.size >= 5 else np.nan)


def dividend_part_quintiles(panel: pd.DataFrame) -> dict:
    g = panel.dropna(subset=["dividend_part", "fwd_12m"]).copy()
    g["q"] = _xs_quintile(g, "dividend_part")
    out = {}
    for q, sub in g.dropna(subset=["q"]).groupby("q"):
        out[f"Q{int(q)}"] = {"n": int(len(sub)),
                             "mean_dividend_part": round(float(sub["dividend_part"].mean()), 4),
                             "mean_fwd_12m": round(float(sub["fwd_12m"].mean()), 4)}
    return out


def combined_error_quintiles(panel: pd.DataFrame) -> dict:
    """The U-shape the research reported: the two defects push opposite ways, which is exactly
    why an AGGREGATE IC check hid both."""
    g = panel.dropna(subset=["mcap_error", "fwd_12m"]).copy()
    g["q"] = _xs_quintile(g, "mcap_error")
    return {f"Q{int(q)}": {"n": int(len(sub)),
                           "mean_fwd_12m": round(float(sub["fwd_12m"].mean()), 4)}
            for q, sub in g.dropna(subset=["q"]).groupby("q")}


def deadjusted_rows(panel: pd.DataFrame) -> dict:
    """How much of the table the de-adjustment actually touches, and in which direction.
    Upward (reverse splits) overstates mcap 8-20x; downward understates it 4-500x."""
    g = panel.dropna(subset=["split_part"])
    down, up = g["split_part"] < 0.9999, g["split_part"] > 1.0001
    return {"total_rows": int(len(g)), "deadjusted": int((down | up).sum()),
            "deadjusted_down": int(down.sum()), "deadjusted_up": int(up.sum()),
            "share": round(float((down | up).mean()), 4),
            "distinct_tickers": int(g.loc[down | up, "ticker"].nunique()),
            "tickers_down": int(g.loc[down, "ticker"].nunique()),
            "tickers_up": int(g.loc[up, "ticker"].nunique()),
            "total_tickers": int(g["ticker"].nunique())}


def sec_cover_page_agreement(frames: dict, panel: pd.DataFrame, column: str) -> dict:
    """The only place a point-in-time truth exists: the SEC cover-page share count.

    `column` selects which merged column is being judged, so the same function scores
    `sharesOutstanding` before P3 and `sharesOutstandingPit` after it."""
    if column not in panel.columns:
        return {"skipped": f"{column} not present"}
    sec = frames["sec"].rename(columns={"sharesOutstanding": "sec_shares"})
    j = panel[["ticker", "as_of", column]].merge(sec, on=["ticker", "as_of"], how="inner")
    j = j.dropna(subset=[column, "sec_shares"])
    j = j[j["sec_shares"] > 0]
    if j.empty:
        return {"column": column, "rows": 0}
    j["ratio"] = j[column] / j["sec_shares"]
    agree = (j["ratio"] - 1).abs() <= SEC_AGREEMENT_TOL
    # ANY failing row condemns the ticker. A median would hide ANET, whose 2021 split IS
    # de-adjusted and whose 2024 one is not -- a PARTIAL de-adjustment is the worst shape
    # there is, because the residual factor is neither the PIT count nor the vendor's.
    j["bad"] = ~agree
    per_ticker = j.groupby("ticker").agg(bad=("bad", "sum"), median_ratio=("ratio", "median"))
    failing = per_ticker[per_ticker["bad"] > 0].sort_values("median_ratio")
    return {"column": column, "rows": int(len(j)), "agree": int(agree.sum()),
            "too_high": int((j["ratio"] > 1 + SEC_AGREEMENT_TOL).sum()),
            "too_low": int((j["ratio"] < 1 - SEC_AGREEMENT_TOL).sum()),
            "tickers": int(len(per_ticker)), "failing_tickers": int(len(failing)),
            "failing": {t: {"bad_rows": int(r.bad),
                            "median_ratio": round(float(r.median_ratio), 4)}
                        for t, r in failing.iterrows()}}


def spike_revert_scan(prices: pd.DataFrame) -> dict:
    """Jumps of >55% whose LEVEL comes back within 7 bars -- two adjustment vintages meeting
    inside one ticker, not a market event.

    A LEVEL test, not a next-day-return test. A vintage seam is a plateau: MNST held the new
    basis for six bars before flipping back, so `|ret|>55% and |next_ret|>55%` finds nothing
    while the table is visibly corrupt. Measured on the pre-fix table this returns exactly
    the three MNST seams post-2020, plus FITB 2009-02-06 and HIG 2008-11-03 -- both genuine
    GFC round-trips, and both pre-2020, so the post-2020 count is the one to gate on."""
    px = prices.sort_values(["ticker", "date"]).copy()
    px["ret"] = px.groupby("ticker")["close"].pct_change(fill_method=None)
    pre_jump = px.groupby("ticker")["close"].shift(1)
    ahead = [(px.groupby("ticker")["close"].shift(-i) / pre_jump - 1).abs()
             for i in range(1, SPIKE_REVERT_BARS + 1)]
    px["revert_gap"] = pd.concat(ahead, axis=1).min(axis=1)

    hit = px[(px["ret"].abs() > SPIKE_THRESHOLD) & (px["revert_gap"] < SPIKE_REVERT_BAND)]
    events = [{"ticker": row.ticker, "date": row.date.strftime("%Y-%m-%d"),
               "ret": round(float(row.ret), 4),
               "revert_gap": round(float(row.revert_gap), 4)}
              for row in hit.sort_values(["date", "ticker"]).itertuples()]
    post2020 = [e for e in events if e["date"] >= "2020-01-01"]
    return {"n": len(events), "n_post_2020": len(post2020),
            "tickers_post_2020": sorted({e["ticker"] for e in post2020}), "events": events}


def mnst_window(prices: pd.DataFrame) -> list[dict]:
    """The live corruption, verbatim: MNST split ~2026-07-20 and the table interleaves the
    two bases because nothing ever re-pulls history."""
    w = prices[(prices["ticker"] == "MNST")
               & prices["date"].between("2026-07-15", "2026-08-15")].sort_values("date")
    return [{"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 4)}
            for d, c in zip(w["date"], w["close"])]


# --------------------------------------------------------------------------- #
# report                                                                      #
# --------------------------------------------------------------------------- #
def to_markdown(blob: dict) -> str:
    env = blob["env"]
    L = [f"# Basis baseline -- `{blob['tag']}`", "",
         f"Generated {blob['generated_utc']} from the live `pea` database by "
         "`scripts/basis_baseline.py`.", "",
         "> The merged `cube` table did not exist when this ran, so **no pre-fix model "
         "metrics exist**. `cube_mcap` below is the `daily_market_cap` FORMULA recomputed "
         "in-script, not a column read from anywhere. A future session must not hunt for a "
         "'before' IC or Sharpe column -- there never was one.", "",
         f"**Environment**: {env['prices_rows']:,} price rows "
         f"({env['prices_min']} -> {env['prices_max']}), "
         f"{env['panel_rows']:,} joined filing rows / {env['panel_tickers']} tickers. "
         f"`cube_part_prices`: {env['cube_part_prices']}.", ""]

    L += ["## mcap error by year", "",
          "| year | n | median | p05 | min | rows off >10% |", "|---|---|---|---|---|---|"]
    for y, v in sorted(blob["mcap_error_by_year"].items()):
        if int(y) in REPORT_YEARS:
            L.append(f"| {y} | {v['n']} | {v['median']} | {v['p05']} | {v['min']} "
                     f"| {v['off_by_10pct']} |")

    L += ["", "## error decomposition -- `mcap_error = split_part x dividend_part`", "",
          "**The residual must be 1.0000 in every year.** It is the plan's foundation.", "",
          "| year | n | split_part | dividend_part | product | mcap_error | residual |",
          "|---|---|---|---|---|---|---|"]
    for y, v in sorted(blob["error_decomposition"].items()):
        if int(y) in REPORT_YEARS:
            L.append(f"| {y} | {v['n']} | {v['split_part']} | {v['dividend_part']} "
                     f"| {v['product']} | {v['mcap_error']} | {v['residual']} |")

    c = blob["split_part_cohorts"]
    labels = {"lt_1": "`split_part < 1` (a split WILL occur)", "eq_1": "`split_part == 1`",
              "gt_1": "`split_part > 1` (reverse split will occur)"}
    L += ["", "## split_part cohorts -- the leak test", "",
          "| cohort | n | mean fwd 12m | median |", "|---|---|---|---|"]
    L += [f"| {labels[k]} | {c[k]['n']} | {c[k]['mean_fwd_12m']:.2%} "
          f"| {c[k]['median_fwd_12m']:.2%} |" for k in ("lt_1", "eq_1", "gt_1") if k in c]

    L += ["", "## dividend_part quintiles (cross-sectional)", "",
          "| quintile | n | mean dividend_part | mean fwd 12m |", "|---|---|---|---|"]
    for q, v in sorted(blob["dividend_part_quintiles"].items()):
        L.append(f"| {q} | {v['n']} | {v['mean_dividend_part']} | {v['mean_fwd_12m']:.2%} |")

    L += ["", "## combined mcap_error quintiles (the U-shape)", "",
          "| quintile | n | mean fwd 12m |", "|---|---|---|"]
    for q, v in sorted(blob["combined_error_quintiles"].items()):
        L.append(f"| {q} | {v['n']} | {v['mean_fwd_12m']:.2%} |")

    d = blob["deadjusted_rows"]
    L += ["", "## de-adjusted rows", "",
          f"{d['deadjusted']:,} of {d['total_rows']:,} rows ({d['share']:.1%}) across "
          f"{d['distinct_tickers']} of {d['total_tickers']} tickers. "
          f"DOWN (forward splits, mcap understated 4-500x): {d['deadjusted_down']:,} rows / "
          f"{d['tickers_down']} tickers. UP (reverse splits, mcap OVERstated 8-20x): "
          f"{d['deadjusted_up']} rows / {d['tickers_up']} tickers."]

    s = blob["sec_cover_page_agreement"]
    if s.get("rows"):
        L += ["", "## SEC cover-page agreement", "",
              f"Column `{s['column']}`: {s['agree']:,} of {s['rows']:,} rows agree within "
              f"+/-3%; {s['too_high']} too high, {s['too_low']} too low. "
              f"**{s['failing_tickers']} of {s['tickers']} tickers fail.**", "",
              "| ticker | bad rows | median ratio merged/SEC |", "|---|---|---|"]
        L += [f"| {t} | {v['bad_rows']} | {v['median_ratio']} |"
              for t, v in s["failing"].items()]

    sp = blob["spike_revert_scan"]
    L += ["", "## spike-and-revert scan", "",
          f"{sp['n']} events, {sp['n_post_2020']} post-2020 "
          f"({', '.join(sp['tickers_post_2020']) or 'none'}).", "",
          "| ticker | date | ret | revert gap |", "|---|---|---|---|"]
    L += [f"| {e['ticker']} | {e['date']} | {e['ret']:.2%} | {e['revert_gap']:.2%} |"
          for e in sp["events"]]

    L += ["", "## MNST 2026-07-15 -> 2026-08-15", "", "| date | close |", "|---|---|"]
    L += [f"| {r['date']} | {r['close']} |" for r in blob["mnst_window"]]

    L += ["", "## control digests", "",
          "| digest | value | must be identical after |", "|---|---|---|",
          f"| `macro_equity_tr_digest` | `{blob['macro_equity_tr_digest']}` | **P1** |",
          f"| `option_overhang_digest` | `{blob['option_overhang_digest']}` | **P3** |", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="directory for baseline.{json,md}")
    ap.add_argument("--tag", default="before", help="label, e.g. before / after-p3")
    args = ap.parse_args()

    _, context = get_config_context("./configs", use_cache=False, save=False)
    store = context.store

    frames = load_frames(store)
    panel = build_panel(frames)
    prices = frames["prices"]

    cube_probe = store.load("cube_part_prices", columns=["ticker"], limit=1, optional=True)
    # After P3 the PIT semantics live in their own column; before it, `sharesOutstanding`
    # still carries them. Score whichever one currently claims to be point-in-time.
    pit_col = ("sharesOutstandingPit" if "sharesOutstandingPit" in panel.columns
               else "sharesOutstanding")

    blob = {
        "tag": args.tag,
        # UTC, second resolution: the ONLY nondeterministic field, and it is metadata --
        # every measurement below must be byte-identical across two runs.
        "generated_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "env": {
            "prices_rows": int(len(prices)),
            "prices_min": str(prices["date"].min().date()),
            "prices_max": str(prices["date"].max().date()),
            "panel_rows": int(len(panel)),
            "panel_tickers": int(panel["ticker"].nunique()),
            "cube_part_prices": "present" if cube_probe is not None else "absent",
        },
        "mcap_error_by_year": mcap_error_by_year(panel),
        "error_decomposition": error_decomposition(panel),
        "split_part_cohorts": split_part_cohorts(panel),
        "dividend_part_quintiles": dividend_part_quintiles(panel),
        "combined_error_quintiles": combined_error_quintiles(panel),
        "deadjusted_rows": deadjusted_rows(panel),
        "sec_cover_page_agreement": sec_cover_page_agreement(frames, panel, pit_col),
        "spike_revert_scan": spike_revert_scan(prices),
        "mnst_window": mnst_window(prices),
        "macro_equity_tr_digest": _digest(frames["macro"]["close"]),
        "option_overhang_digest": _digest(frames["merged"]["optionOverhang"]),
        "mcap_error_overall": _stats(panel["mcap_error"]),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "baseline" if args.tag == "before" else f"baseline-{args.tag}"
    (out / f"{stem}.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(blob), encoding="utf-8")

    dec = blob["error_decomposition"]
    bad = {y: v["residual"] for y, v in dec.items() if abs(v["residual"] - 1.0) > 1e-4}
    print(f"\nwrote {out / (stem + '.json')} and {out / (stem + '.md')}")
    print(f"panel: {blob['env']['panel_rows']:,} rows / {blob['env']['panel_tickers']} tickers")
    print(f"macro_equity_tr_digest = {blob['macro_equity_tr_digest']}")
    print(f"option_overhang_digest = {blob['option_overhang_digest']}")
    if bad:
        print(f"\n[FAIL] error decomposition residual != 1.0000 in {len(bad)} years: "
              f"{dict(list(bad.items())[:8])}\n  -> the multiplicative decomposition is the "
              "plan's foundation. STOP and re-open the research.")
    else:
        print(f"\n[OK] decomposition residual == 1.0000 in all {len(dec)} years "
              "-- mcap_error = split_part x dividend_part holds exactly.")


if __name__ == "__main__":
    main()
