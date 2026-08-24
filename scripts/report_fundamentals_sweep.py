"""
report_fundamentals_sweep.py (scripts/)
---------------------------------------
Turn the `sweep_fundamentals_resolution.py` ledgers into Phase 4c's acceptance report.

Offline: reads only the per-ticker parquet the sweep already paid for, so every number here
is re-derivable in seconds and none of it needs EDGAR again. That is the point -- every
acceptance figure in Phases 3b, 4 and 4b came from scratchpad scripts that no longer exist,
so not one of them is reproducible by anyone else today.

Sections, each answering one of Phase 4c's own questions:

  1. route mix, and the `tag_fallback` rate against the 20% architecture gate
  2. **4c.1 before/after on the same join key** -- route-changed % and value-agreed % per
     fiscal year, with every material disagreement named. This is the 3c.1 protocol, and it
     is the acceptance criterion for the statement-role test
  3. the statement-role guard's own census: what it withheld, and where it had to give in
  4. **4c.3** duplicate-fact census -- the population Phase 5b's `duplicate_fact` inherits
  5. **4c.4** coverage by (regime, field), the input to widening the exception register
  6. **4c.2** `longTermDebt`'s basis census -- which concept actually wins, per ticker
  7. **4c.7** AXP's revenue legs -- does banning the post-provision concept gain a basis?

    "$PY" scripts/report_fundamentals_sweep.py [--in DIR] [--roster in_sample|both]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "fundamentals_sweep"

from src.constants.constants import (                                  # noqa: E402
    FUNDAMENTALS_CATALOGUE_SUBDIR, FUNDAMENTALS_ROSTERS_FILENAME)

#: The join key a value is compared on across the two resolution settings. It is the FACT's
#: identity, not the row's: same filing, same field, same period shape, same window. Anything
#: coarser pools two periods; anything finer (the route, the concept) is what we are measuring.
JOIN_KEY = ["ticker", "accession_number", "field", "duration_type",
            "period_start", "period_end"]

#: A relative difference above this is "material" and must be named rather than counted.
MATERIAL = 0.02

#: The architecture gate route 6 is measured against (v1 Phase 3).
TAG_FALLBACK_GATE = 0.20


def load_ledger(in_dir: Path, tickers: set[str] | None) -> pd.DataFrame:
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no sweep ledgers in {in_dir}")
    frames = []
    for path in files:
        if tickers is not None and path.stem not in tickers:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no ledgers in {in_dir} matched the requested roster")
    out = pd.concat(frames, ignore_index=True)
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce")
    return out


def _adjustment_flag(series: pd.Series, key: str) -> pd.Series:
    """Is `key` present in the `adjustment` JSON blob? Vectorised over the column rather
    than parsed per row -- a substring test is exact here because the keys are literal and
    the blob is machine-written."""
    return series.fillna("").astype(str).str.contains(f'"{key}"', regex=False)


def section_route_mix(strict: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("1. ROUTE MIX  (strict resolution, i.e. what production now does)")
    print("=" * 78)
    valued = strict[strict["value"].notna()]
    mix = valued["resolution_method"].value_counts()
    total = int(mix.sum())
    for method, n in mix.items():
        print(f"  {method:20s} {n:8,d}  {n / total:6.2%}")
    fallback = mix.get("tag_fallback", 0) / total if total else 0.0
    verdict = "PASS" if fallback < TAG_FALLBACK_GATE else "FAIL"
    print(f"  {'-' * 44}")
    print(f"  valued rows {total:,} | tag_fallback {fallback:.2%} "
          f"vs the {TAG_FALLBACK_GATE:.0%} architecture gate -> {verdict}")
    print("  (the gate applies to `tag_fallback` alone; `tag_primary` is the catalogue's own")
    print("   first choice with no linkbase arc, which is expected for any non-roll-up leaf)")


def section_before_after(ledger: pd.DataFrame) -> pd.DataFrame:
    """4c.1's acceptance: the same fact resolved with and without the statement-role test."""
    print("\n" + "=" * 78)
    print("2. 4c.1 BEFORE/AFTER  (prefer_structure=False -> True, same join key)")
    print("=" * 78)
    strict = ledger[ledger["prefer_structure"]]
    lax = ledger[~ledger["prefer_structure"]]
    cols = JOIN_KEY + ["fiscal_year", "resolution_method", "source_concept", "value",
                       "dc_code"]
    merged = strict[cols].merge(lax[cols], on=JOIN_KEY, suffixes=("_after", "_before"),
                                how="outer", indicator=True)
    both = merged[merged["_merge"] == "both"].copy()
    print(f"  facts present in both resolutions : {len(both):,}")
    print(f"  present only AFTER the guard      : {(merged['_merge'] == 'left_only').sum():,}")
    print(f"  present only BEFORE the guard     : "
          f"{(merged['_merge'] == 'right_only').sum():,}  <- the guard's coverage cost")

    both["route_changed"] = both["resolution_method_after"] != both["resolution_method_before"]
    both["concept_changed"] = both["source_concept_after"] != both["source_concept_before"]
    scale = both[["value_after", "value_before"]].abs().max(axis=1)
    both["relative"] = ((both["value_after"] - both["value_before"]).abs()
                        / scale.where(scale > 0))
    both["value_agreed"] = both["relative"].fillna(0) <= 1e-9

    print(f"\n  route changed on {both['route_changed'].mean():.3%} of shared facts "
          f"({int(both['route_changed'].sum()):,} rows)")
    print(f"  value agreed to the dollar on {both['value_agreed'].mean():.3%}")
    moved = both[~both["value_agreed"]]
    print(f"  values that MOVED: {len(moved):,}  "
          f"(material, >{MATERIAL:.0%}: {int((moved['relative'] > MATERIAL).sum()):,})")

    print("\n  by fiscal year:")
    print(f"    {'year':6s} {'facts':>8s} {'route chg':>10s} {'value agree':>12s} "
          f"{'moved':>7s} {'material':>9s}")
    # `fiscal_year` is not in JOIN_KEY, so the merge suffixed it; the two sides agree by
    # construction (same fact) and the `_after` copy is the one production produced.
    both["fiscal_year"] = pd.to_numeric(both["fiscal_year_after"], errors="coerce")
    for year, group in both.dropna(subset=["fiscal_year"]).groupby(
            lambda i: int(both.loc[i, "fiscal_year"]), sort=True):
        mv = group[~group["value_agreed"]]
        print(f"    {int(year):<6d} {len(group):8,d} {group['route_changed'].mean():9.2%} "
              f"{group['value_agreed'].mean():11.3%} {len(mv):7,d} "
              f"{int((mv['relative'] > MATERIAL).sum()):9,d}")
    return both


def section_material_disagreements(both: pd.DataFrame) -> None:
    print("\n  every MATERIAL disagreement, by (ticker, field) -- these must each be named:")
    moved = both[both["relative"] > MATERIAL]
    if moved.empty:
        print("    none")
        return
    grouped = (moved.groupby(["ticker", "field"])
               .agg(rows=("relative", "size"), median_rel=("relative", "median"),
                    max_rel=("relative", "max"),
                    after=("source_concept_after", lambda s: s.dropna().iloc[0]
                           if s.notna().any() else None),
                    before=("source_concept_before", lambda s: s.dropna().iloc[0]
                            if s.notna().any() else None))
               .sort_values("rows", ascending=False))
    for (ticker, field), row in grouped.iterrows():
        print(f"    {ticker:6s} {field:22s} {int(row['rows']):4d} rows  "
              f"median {row['median_rel']:8.2%}  max {row['max_rel']:9.2%}")
        print(f"           before: {row['before']}")
        print(f"           after : {row['after']}")


def section_guard_census(strict: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("3. 4c.1's OWN CENSUS -- the two halves, counted separately")
    print("=" * 78)
    rejected = _adjustment_flag(strict["adjustment"], "role_rejected")
    retained = _adjustment_flag(strict["adjustment"], "role_only_retained")
    undeclared = _adjustment_flag(strict["adjustment"], "undeclared_rejected")
    print("  half 1 -- the note-role test (`is_note_only`):")
    print(f"    withheld a candidate on {int(rejected.sum()):,} rows "
          f"({rejected.mean():.3%} of {len(strict):,})")
    print(f"    resolved ONLY after giving in on {int(retained.sum()):,} rows "
          f"({retained.mean():.3%})  <- flagged, never lost")
    print("  half 2 -- the DECLAREDNESS test (an undeclared tag loses to route 3b):")
    print(f"    reordered onto the leaf sum on {int(undeclared.sum()):,} rows "
          f"({undeclared.mean():.3%})")
    if undeclared.any():
        hits = strict[undeclared]
        print("    by (ticker, field):")
        for (ticker, field), n in (hits.groupby(["ticker", "field"]).size()
                                   .sort_values(ascending=False).items()):
            withheld = sorted({c for blob in hits[(hits["ticker"] == ticker)
                                                  & (hits["field"] == field)]["adjustment"]
                               for c in json.loads(blob).get("undeclared_rejected", [])})
            print(f"      {ticker:6s} {field:20s} {int(n):5d} rows  withheld "
                  f"{', '.join(withheld)}")
    if rejected.any():
        print("\n  which candidates were withheld, and from which field:")
        hits = strict[rejected]
        pairs: dict[tuple[str, str], set[str]] = {}
        for field, blob in zip(hits["field"], hits["adjustment"]):
            for concept in json.loads(blob).get("role_rejected", []):
                pairs.setdefault((field, concept), set())
        counts = (hits.assign(_n=1).groupby("field")["_n"].sum().sort_values(ascending=False))
        for field, n in counts.items():
            concepts = sorted({c for (f, c) in pairs if f == field})
            print(f"    {field:22s} {int(n):5d} rows   {', '.join(concepts)[:110]}")
        print("\n  tickers affected:")
        by_ticker = hits.groupby("ticker").size().sort_values(ascending=False)
        print("    " + ", ".join(f"{t}({n})" for t, n in by_ticker.items()))


def section_duplicates(strict: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("4. 4c.3 DUPLICATE-FACT CENSUS  (finer `decimals` wins; disagreements recorded)")
    print("=" * 78)
    flag = _adjustment_flag(strict["adjustment"], "duplicate_fact")
    print(f"  rows where one filing tagged a (concept, period) twice at TWO DIFFERENT "
          f"values: {int(flag.sum()):,} ({flag.mean():.4%})")
    if not flag.any():
        print("  none -- the tie-break never had to discriminate on this ledger")
        return
    hits = strict[flag]
    rows = []
    for ticker, field, blob in zip(hits["ticker"], hits["field"], hits["adjustment"]):
        for d in json.loads(blob).get("duplicate_fact", []):
            kept, dropped = abs(float(d["kept"])), abs(float(d["dropped"]))
            scale = max(kept, dropped)
            rows.append({"ticker": ticker, "field": field, "concept": d["concept"],
                         "kept": d["kept"], "dropped": d["dropped"],
                         "kept_decimals": d["kept_decimals"],
                         "dropped_decimals": d["dropped_decimals"],
                         "relative": (abs(kept - dropped) / scale) if scale else 0.0})
    frame = pd.DataFrame(rows)
    print(f"  distinct duplicate facts: {len(frame):,}  |  median disagreement "
          f"{frame['relative'].median():.4%}  max {frame['relative'].max():.4%}")
    print("\n  by (ticker, field), worst first:")
    grouped = (frame.groupby(["ticker", "field"])
               .agg(n=("relative", "size"), max_rel=("relative", "max"))
               .sort_values("max_rel", ascending=False).head(20))
    for (ticker, field), row in grouped.iterrows():
        print(f"    {ticker:6s} {field:22s} {int(row['n']):4d}  worst {row['max_rel']:8.4%}")
    print("\n  a sample, showing which precision won:")
    for r in frame.sort_values("relative", ascending=False).head(6).itertuples():
        print(f"    {r.ticker:6s} {r.field:18s} {r.concept}")
        print(f"           kept {r.kept:>18,.0f} (decimals {r.kept_decimals})  "
              f"dropped {r.dropped:>18,.0f} (decimals {r.dropped_decimals})")


#: The six balance-sheet DETAIL fields 4c.4 must scope by regime. Reg S-X 5-02 requires the
#: caption only "when appropriate", so a bank, insurer, broker-dealer or REIT that omits them
#: is compliant, not incomplete -- and 346 in-sample "holes" are exactly this.
DETAIL_FIELDS = ["accountsPayable", "accountsReceivable", "ppeGross",
                 "accumulatedDepreciation", "intangiblesExGoodwill", "minorityInterest"]


def section_regime_coverage(strict: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("5. 4c.4 COVERAGE OF THE SIX BALANCE-SHEET DETAIL FIELDS, BY REGIME")
    print("=" * 78)
    print("  share of a regime's TICKERS with at least one valued row for the field.")
    print("  Measured off filing.xbrl() (this ledger), never companyfacts -- which drops")
    print("  dimensioned facts and publishes no company-extension taxonomy at all.\n")
    frame = strict[strict["field"].isin(DETAIL_FIELDS)]
    if frame.empty:
        print("  no rows for these fields in the ledger")
        return
    regimes = sorted(strict["regime"].dropna().unique())
    tickers_per_regime = (strict.dropna(subset=["regime"]).groupby("regime")["ticker"]
                          .nunique())
    print(f"    {'field':26s} " + " ".join(f"{r[:11]:>12s}" for r in regimes))
    print(f"    {'(tickers)':26s} "
          + " ".join(f"{int(tickers_per_regime.get(r, 0)):>12d}" for r in regimes))
    for field in DETAIL_FIELDS:
        cells = []
        for regime in regimes:
            n_total = int(tickers_per_regime.get(regime, 0))
            if not n_total:
                cells.append(f"{'-':>12s}")
                continue
            have = frame[(frame["field"] == field) & (frame["regime"] == regime)
                         & frame["value"].notna()]["ticker"].nunique()
            cells.append(f"{have}/{n_total} {have / n_total:5.0%}".rjust(12))
        print(f"    {field:26s} " + " ".join(cells))
    print("\n  a 0% cell in a non-industrial regime is the structural absence 4c.4 registers;")
    print("  a partial cell is a per-filer question, not a regime one.")


def section_long_term_debt(strict: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("6. 4c.2 `longTermDebt` BASIS CENSUS")
    print("=" * 78)
    print("  The catalogue lists `us-gaap:LongTermDebt` at priority 2 against a field")
    print("  DEFINED ex-current, and that concept INCLUDES the current portion. Which")
    print("  concept actually wins, per ticker, decides whether to demote or to subtract.\n")
    frame = strict[strict["field"] == "longTermDebt"]
    if frame.empty:
        print("  no longTermDebt rows in the ledger")
        return
    valued = frame[frame["value"].notna()]
    print(f"  {'ticker':7s} {'rows':>6s} {'valued':>7s}  concepts (share of valued rows)")
    for ticker, group in frame.groupby("ticker"):
        v = group[group["value"].notna()]
        if v.empty:
            codes = group["dc_code"].dropna().value_counts().to_dict()
            print(f"  {ticker:7s} {len(group):6d} {0:7d}  NEVER RESOLVES -- {codes}")
            continue
        mix = v["source_concept"].value_counts(normalize=True)
        parts = ", ".join(f"{c.split(':')[-1]} {s:.0%}" for c, s in mix.head(3).items())
        print(f"  {ticker:7s} {len(group):6d} {len(v):7d}  {parts}")
    hits = valued[valued["source_concept"].astype(str).str.endswith(":LongTermDebt")]
    print(f"\n  rows resolved on the CONTAMINATED concept `us-gaap:LongTermDebt`: "
          f"{len(hits):,} across {hits['ticker'].nunique()} ticker(s)")
    if len(hits):
        print("    " + ", ".join(f"{t}({n})" for t, n
                                 in hits.groupby("ticker").size().items()))
    print("\n  tickers with NO longTermDebt value at all (the periodicity/absence question):")
    never = sorted(set(frame["ticker"]) - set(valued["ticker"]))
    print(f"    {never or 'none'}")


#: The post-provision revenue concept 4c.7 proposes to ban for the bank regime, and the two
#: Rule 9-04 legs that must BOTH be present for the ban to gain a comparable basis rather
#: than three-quarters of a top line.
AXP_BANNED = "TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses"
NINE_OH_FOUR_LEGS = ("InterestIncomeExpenseNet", "NoninterestIncome")


def section_axp(strict: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("7. 4c.7 AXP's REVENUE BASIS -- does banning the post-provision concept help?")
    print("=" * 78)
    print("  3c.4 worked for MTB because MTB tags BOTH Rule 9-04 legs, so 96 of 110")
    print("  post-provision rows moved onto a comparable basis. If AXP does not tag both,")
    print("  the ban turns rows into reason-coded nulls instead. Measure first.\n")
    revenue = strict[strict["field"] == "totalRevenue"]
    for ticker in ("AXP", "MTB"):
        group = revenue[revenue["ticker"] == ticker]
        if group.empty:
            print(f"  {ticker:6s} not in this ledger")
            continue
        valued = group[group["value"].notna()]
        mix = valued["source_concept"].value_counts()
        regime = group["regime"].dropna().unique()
        print(f"  {ticker:6s} regime={list(regime)} rows={len(group)} valued={len(valued)}")
        for concept, n in mix.items():
            mark = "  <- the post-provision basis" if AXP_BANNED in str(concept) else ""
            print(f"           {n:4d}  {concept}{mark}")
        # Are both Rule 9-04 legs actually reported by this filer, anywhere?
        every = strict[strict["ticker"] == ticker]
        legs = {leg: int(every["source_concept"].astype(str)
                         .str.endswith(f":{leg}").sum()) for leg in NINE_OH_FOUR_LEGS}
        print(f"           Rule 9-04 legs used as a source_concept anywhere: {legs}")
    print("\n  NOTE: a leg used as a `roll_up` child shows up in `roll_up_children`, not in")
    print("  `source_concept`, so a zero above is not proof of absence -- check the bank")
    print("  regime's roll_up rows below.")
    bank_sum = strict[(strict["field"] == "totalRevenue")
                      & (strict["resolution_method"] == "linkbase_sum")]
    if not bank_sum.empty:
        print("\n  totalRevenue rows resolved by linkbase_sum (the two-leg bank roll-up):")
        for ticker, group in bank_sum.groupby("ticker"):
            children = group["roll_up_children"].dropna().unique()[:1]
            print(f"    {ticker:6s} {len(group):4d} rows  {children[0] if len(children) else ''}")


def section_ambiguous_duration(strict):
    """4c.8's fire rate. Runs the period engine over the ledger -- offline, no network --
    because `ambiguous_duration` is a refusal made inside `quarterize`, not a stored column.
    """
    from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
    from src.data_extract.utils.fundamentals.periods import build_periods

    print("\n" + "=" * 78)
    print("8. 4c.8 D1b FIRE RATE  (`ambiguous_duration`), both rosters")
    print("=" * 78)
    print("  Known population from the plan: 1 period (ORCL fiscal 2020 revenue).")
    print("  A wider fire means the nine-month comparison is too loose.\n")
    catalogue = load_catalogue("./configs")
    refusals = []
    quarters = []
    for ticker, group in strict.groupby("ticker", sort=True):
        per_ticker = []
        try:
            q, _ttm, _inst = build_periods(group, catalogue, refusals=per_ticker)
        except Exception as exc:                                        # noqa: BLE001
            print(f"  {ticker:6s} build_periods FAILED {type(exc).__name__}: {exc}")
            continue
        refusals.extend({**r, "ticker": ticker} for r in per_ticker)
        if not q.empty:
            quarters.append(q)
    frame = pd.DataFrame(refusals)
    print(f"  total D1b refusals: {len(frame)}")
    if not frame.empty:
        for r in frame.sort_values(["ticker", "field", "period_end"]).itertuples():
            print(f"    {r.ticker:6s} {r.field:16s} {str(r.period_end)[:10]}  "
                  f"{float(r.value) / 1e6:>12,.1f}M  known_from {str(r.known_from)[:10]}")
        print(f"\n  distinct (ticker, field, period): "
              f"{frame.groupby(['ticker', 'field', 'period_end']).ngroups}")
        print(f"  tickers affected: {sorted(frame['ticker'].unique())}")
    if not quarters:
        return pd.DataFrame()
    allq = pd.concat(quarters, ignore_index=True)
    print(f"\n  quarters produced: {len(allq):,} | basis mix:")
    for basis, n in allq["basis"].value_counts().items():
        print(f"    {basis:20s} {n:8,d}  {n / len(allq):6.2%}")
    return allq

def section_form_coverage(strict):
    """Is the 10-Q half of the ledger actually there, and does any field carry a DIFFERENT
    basis in a 10-Q than in a 10-K?

    The second question is the one 4c.1 makes necessary. A 10-Q's calculation linkbase is
    smaller than a 10-K's, so a filer that declares its cash-flow arcs annually but not
    quarterly would send route 1's undeclared candidate back into play for three filings out
    of four -- one field, two bases, split by FORM instead of by era. That is the same defect
    class as MCD's 35.6x era step and no cross-vintage test would see it, because both
    vintages would agree within their own form.
    """
    print("\n" + "=" * 78)
    print("9. FORM COVERAGE, AND THE 10-K vs 10-Q BASIS SPLIT")
    print("=" * 78)
    print(f"  {'form':10s} {'filings':>8s} {'rows':>9s} {'valued':>9s}")
    for form, group in strict.groupby("form", sort=True):
        valued = int(group["value"].notna().sum())
        print(f"  {str(form):10s} {group['accession_number'].nunique():8,d} "
              f"{len(group):9,d} {valued:9,d}")

    valued = strict[strict["value"].notna()].copy()
    valued["is_q"] = valued["form"].astype(str).str.startswith("10-Q")
    # A (ticker, field) is SPLIT when the set of routes it uses in 10-Qs is disjoint from
    # the set it uses in 10-Ks. Overlapping sets are era variation, which cross-vintage
    # checks already cover; disjoint sets mean the FORM decides the basis.
    # The test is on the CONCEPT, not the route label. `tag_primary` and `linkbase_total`
    # are the same value on the same concept and differ only in whether the filer happened
    # to declare a calculation arc for it -- which a 10-Q routinely omits. Flagging that
    # would bury the real cases in noise.
    rows = []
    for (ticker, field), group in valued.groupby(["ticker", "field"]):
        k_c = set(group.loc[~group["is_q"], "source_concept"].dropna())
        q_c = set(group.loc[group["is_q"], "source_concept"].dropna())
        if not k_c or not q_c or (k_c & q_c):
            continue
        k = set(group.loc[~group["is_q"], "resolution_method"])
        q = set(group.loc[group["is_q"], "resolution_method"])
        rows.append({"ticker": ticker, "field": field,
                     "in_10k": f"{','.join(sorted(k))} {sorted(k_c)}",
                     "in_10q": f"{','.join(sorted(q))} {sorted(q_c)}",
                     "n_10k": int((~group["is_q"]).sum()), "n_10q": int(group["is_q"].sum())})
    print(f"\n  (ticker, field) pairs whose 10-K and 10-Q CONCEPTS are disjoint: {len(rows)}")
    for r in sorted(rows, key=lambda d: -(d["n_10k"] + d["n_10q"]))[:25]:
        print(f"    {r['ticker']:6s} {r['field']:20s} 10-K[{r['n_10k']:4d}] {r['in_10k']:19s}"
              f" | 10-Q[{r['n_10q']:4d}] {r['in_10q']}")
    if rows:
        print("  Each of these must be named: a disjoint route set means the FORM chose the")
        print("  basis, which is a step the growth features would carry as signal.")


#: A quarter end and an annual end this close are the same fiscal year end -- a 52/53-week
#: filer's year moves by up to a week against the calendar.
SAME_YEAR_END_DAYS = 7

#: Q4 bases that make the footing identity TAUTOLOGICAL rather than a test. `fy_minus_q1q2q3`
#: is exact by construction; `fy_minus_ytd9` is exact whenever Q2 and Q3 also came off the
#: YTD ladder, which is the normal case. Only an AS-REPORTED Q4 is independent evidence.
DERIVED_Q4 = ("fy_minus_ytd9", "fy_minus_q1q2q3")


def section_annual_footing(strict, quarters):
    """**Q1+Q2+Q3+Q4 == FY, over every ticker's full history.** The user's check 3, and the
    single most informative validation available without external data.

    Two properties make it honest rather than decorative:

      * it runs on **10-Qs and 10-Ks together** -- three of the four quarters can only come
        from 10-Qs, so a ledger without them cannot even attempt this; and
      * it is **split by whether Q4 is independent**. Where Q4 came from `FY - YTD9` and Q2/Q3
        came off the same YTD ladder, the identity is exact by construction and proves
        nothing; the old stack's version of this check passed 99.73% that way. Only an
        AS-REPORTED Q4 -- the filer publishing its own discrete fourth quarter -- makes the
        sum an independent test of our arithmetic against the filer's own annual.

    Non-additive fields are excluded: a weighted-average share count is differenced in
    share-days and summing four of them is meaningless, not wrong.
    """
    from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue

    print("\n" + "=" * 78)
    print("10. Q1+Q2+Q3+Q4 vs FY -- full history, all 52 tickers")
    print("=" * 78)
    catalogue = load_catalogue("./configs")
    additive = {n for n in catalogue.extracted_fields if catalogue.field(n).is_additive}

    # The filer's own annual facts, FIRST-FILED, so a later restatement does not masquerade
    # as a derivation error. `cross_vintage` is the check that owns restatements.
    # LAST-filed, to match `quarterize`: its `_latest_per_window` keeps the newest fact for
    # each window, so the quarters are a latest-vintage view and comparing them against a
    # FIRST-filed annual measures the restatement rate rather than our arithmetic. Measured
    # both ways on this ledger, first-filed cost ~1.5pp of the within-2% rate for exactly
    # that reason.
    annual = strict[(strict["duration_type"] == "annual") & strict["value"].notna()].copy()
    annual["filing_date"] = pd.to_datetime(annual["filing_date"])
    annual = (annual.sort_values("filing_date")
              .drop_duplicates(subset=["ticker", "field", "period_end"], keep="last"))

    q = quarters[quarters["field"].isin(additive) & quarters["value"].notna()].copy()
    q["period_end"] = pd.to_datetime(q["period_end"])
    rows = []
    for (ticker, field, year), grp in q.groupby(["ticker", "field", "fiscal_year"]):
        if len(grp) != 4 or grp["fiscal_quarter"].nunique() != 4:
            continue
        year_end = grp["period_end"].max()
        cand = annual[(annual["ticker"] == ticker) & (annual["field"] == field)]
        if cand.empty:
            continue
        gap = (pd.to_datetime(cand["period_end"]) - year_end).abs()
        near = cand[gap <= pd.Timedelta(days=SAME_YEAR_END_DAYS)]
        if near.empty:
            continue
        reported = float(near["value"].iloc[0])
        summed = float(grp["value"].sum())
        scale = max(abs(reported), abs(summed))
        q4 = grp.sort_values("period_end").iloc[-1]
        rows.append({"ticker": ticker, "field": field, "fiscal_year": year,
                     "summed": summed, "reported": reported,
                     "relative": abs(summed - reported) / scale if scale else 0.0,
                     "q4_basis": q4["basis"],
                     # A sum and an annual of equal magnitude and opposite sign is a SIGN
                     # CONVENTION defect, not a footing failure, and pooling the two hides
                     # both. Phase 5b's `sign_convention` owns this population.
                     "sign_flip": (summed * reported < 0
                                   and abs(abs(summed) - abs(reported)) <= 0.02 * scale),
                     "independent": q4["basis"] not in DERIVED_Q4})
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  no complete four-quarter years at all -- Phase 4 is NOT done")
        return frame
    print(f"  complete four-quarter years found: {len(frame):,} "
          f"across {frame['ticker'].nunique()} tickers and {frame['field'].nunique()} fields")

    flips = frame[frame.sign_flip]
    if not flips.empty:
        print(f"\nSIGN-CONVENTION cases excluded from the rates below: {len(flips)} "
              f"(equal magnitude, opposite sign -- a different defect class)")
        for r in flips.sort_values(["ticker", "field"]).itertuples():
            print(f"    {r.ticker:6s} {r.field:20s} FY{int(r.fiscal_year)}  "
                  f"summed {r.summed / 1e6:>13,.1f}M  filer {r.reported / 1e6:>13,.1f}M")
    frame = frame[~frame.sign_flip]
    for label, sub in (("INDEPENDENT (Q4 as-reported by the filer)", frame[frame.independent]),
                       ("tautological (Q4 derived from the identity)", frame[~frame.independent]),
                       ("pooled", frame)):
        if sub.empty:
            print(f"\n  {label}: 0 points")
            continue
        r = sub["relative"]
        print(f"\n  {label}: {len(sub):,} points")
        print(f"    exact to the dollar {(r < 1e-9).mean():7.2%} | within 0.1% "
              f"{(r < 0.001).mean():7.2%} | within 0.5% {(r < 0.005).mean():7.2%} | "
              f"within 1% {(r < 0.01).mean():7.2%} | within 2% {(r < 0.02).mean():7.2%}")
        print(f"    median error {r.median():.6%}   worst {r.max():.2%}")

    independent = frame[frame.independent]
    if not independent.empty:
        print("\n  INDEPENDENT set, per field (the three weakest fields on every other")
        print("  measure are totalRevenue, operatingIncome and incomeTaxExpense):")
        per_field = (independent.groupby("field")["relative"]
                     .agg(n="size", within_2pc=lambda s: (s < 0.02).mean(),
                          median="median", worst="max")
                     .sort_values("within_2pc"))
        print(f"    {'field':24s} {'n':>5s} {'within 2%':>10s} {'median':>10s} {'worst':>9s}")
        for field, row in per_field.iterrows():
            print(f"    {field:24s} {int(row['n']):5d} {row['within_2pc']:10.2%} "
                  f"{row['median']:10.4%} {row['worst']:9.2%}")

        print("\n  INDEPENDENT set, per ticker, worst first:")
        per_ticker = (independent.groupby("ticker")["relative"]
                      .agg(n="size", within_2pc=lambda s: (s < 0.02).mean(), worst="max")
                      .sort_values(["within_2pc", "worst"], ascending=[True, False]))
        for ticker, row in per_ticker.head(15).iterrows():
            print(f"    {ticker:6s} n={int(row['n']):4d}  within 2% {row['within_2pc']:7.2%}"
                  f"  worst {row['worst']:8.2%}")

        print("\n  every INDEPENDENT failure beyond 2%, named -- each needs a mechanism:")
        bad = independent[independent["relative"] > 0.02].sort_values("relative",
                                                                     ascending=False)
        print(f"    {len(bad)} of {len(independent)} ({len(bad) / len(independent):.2%})")
        for r in bad.head(40).itertuples():
            print(f"    {r.ticker:6s} {r.field:20s} FY{int(r.fiscal_year)}  "
                  f"summed {r.summed / 1e6:>13,.1f}M  filer {r.reported / 1e6:>13,.1f}M  "
                  f"{r.relative:8.2%}  q4={r.q4_basis}")
    print("\n  A restatement is the EXPECTED cause of a residual here and is not a defect:")
    print("  we compare against the FIRST-FILED annual, and 4.53% of annual windows move")
    print("  more than 2% between a filer's first and last filing of the same year.")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_dir", default=str(DEFAULT_IN))
    parser.add_argument("-c", "--config-dir", default="./configs")
    parser.add_argument("--roster", default="all",
                        help="in_sample | out_of_sample | both | all (default: whatever "
                             "ledgers exist)")
    args = parser.parse_args()

    tickers: set[str] | None = None
    if args.roster != "all":
        path = (Path(args.config_dir) / FUNDAMENTALS_CATALOGUE_SUBDIR
                / FUNDAMENTALS_ROSTERS_FILENAME)
        blob = json.loads(path.read_text(encoding="utf-8"))
        names = (["in_sample", "out_of_sample"] if args.roster == "both"
                 else [args.roster])
        tickers = {t for name in names for t in blob[name]}

    ledger = load_ledger(Path(args.in_dir), tickers)
    strict = ledger[ledger["prefer_structure"]]
    print(f"ledger: {len(ledger):,} rows | {ledger['ticker'].nunique()} tickers | "
          f"{ledger['accession_number'].nunique():,} filings | "
          f"fiscal {int(ledger['fiscal_year'].min())}-{int(ledger['fiscal_year'].max())}")

    section_route_mix(strict)
    both = section_before_after(ledger)
    section_material_disagreements(both)
    section_guard_census(strict)
    section_duplicates(strict)
    section_regime_coverage(strict)
    section_long_term_debt(strict)
    section_axp(strict)
    section_form_coverage(strict)
    quarters = section_ambiguous_duration(strict)
    section_annual_footing(strict, quarters)
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
