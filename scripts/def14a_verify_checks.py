"""
def14a_verify_checks.py  (scripts/)
--------------------------------------------------------------------------------------------
Value-level checks over the sample `def14a_verify_sample.py` persisted. Reads parquet only --
no LLM, no network, no database -- so it can be re-run on every prompt or schema change without
spending a token.

The point is the distinction between a populated cell and a CORRECT one. Fill rates and
"0 rows above $1e9" are shape checks: they cannot tell a right number from a plausible wrong
one. Each check below is an IDENTITY the filing itself must satisfy, so a failure localises the
defect instead of merely flagging a row:

  C1  SCT components sum to the reported total          (the filer's own arithmetic)
  C2  audit fee categories sum to the reported total    (same, for Item 9)
  C3  `ceo_total_comp` equals the CEO's own SCT row      (parent scalar vs child table)
  C4  pay ratio == ceo_total_comp / median_employee_pay  (a disclosed triplet, any leg checks)
  C5  director rows within a turnover window of board_size (scalar vs child table; NOT equality
      -- the child table legitimately includes directors who left during the year)
  C6  director ages and tenures are humanly possible
  C7  ownership percents are in range and do not exceed 100% in aggregate
  C8  say-on-pay support is a fraction, not a percentage
  C9  every director carrying a gender carries a gender_basis
  C10 no duplicate primary keys in any child table

A check that cannot be evaluated (both legs absent) is reported as such and never counted as a
pass -- an unevaluated check silently inflating a pass rate is the failure mode this file
exists to avoid.

    "$PY" scripts/def14a_verify_checks.py [--dir reports/.../verify] [--strict]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_extract.utils.structure.def14a_gender import person_key

DEFAULT_DIR = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/verify"

#: Item 402(c)'s seven components. `total` is column (j).
_SCT_COMPONENTS = ["salary", "bonus", "stock_awards", "option_awards", "non_equity_incentive",
                   "pension_change", "other_compensation"]
_DIR_COMPONENTS = ["fees_earned", "stock_awards", "option_awards", "non_equity_incentive",
                   "pension_change", "other_compensation"]
_FEE_PARTS = ["audit_fees_audit", "audit_fees_audit_related", "audit_fees_tax",
              "audit_fees_other"]
#: A filer rounds to the dollar; $10 absorbs that without absorbing a missing column.
_TOL_USD = 10.0
#: Pay ratio and fee identities are disclosed to fewer significant figures.
_TOL_PCT = 0.02


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.rows: list[tuple[str, str, int, int, int]] = []

    def add(self, cid: str, what: str, ok: int, bad: int, na: int,
            detail: list[str] | None = None) -> None:
        self.rows.append((cid, what, ok, bad, na))
        total = ok + bad
        rate = f"{100 * ok / total:.1f}%" if total else "n/a"
        print(f"  {cid:<5}{what:<52}{ok:>5} ok {bad:>4} bad {na:>4} n/a   {rate}")
        self.lines += [f"### {cid} — {what}", "",
                       f"- **{ok} pass / {bad} fail** ({rate}); {na} could not be evaluated", ""]
        if detail:
            self.lines += detail + [""]
        for d in detail or []:
            print(f"        {d}")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return pd.to_numeric(df[col], errors="coerce")


def _sum_identity(df: pd.DataFrame, parts: list[str], total: str,
                  tol: float) -> tuple[int, int, int, list[str]]:
    """Rows where the parts sum to the total.

    A row with only SOME components present cannot be expected to sum -- the shortfall is the
    missing column, not a wrong value -- so it is only a FAILURE when the partial sum already
    EXCEEDS the total, which is arithmetically impossible. Counting a partial row as a failure
    is what made this check report 7 fabricated failures on A's director table, where the filer
    discloses two fee columns and the check demanded six.
    """
    if df.empty:
        return 0, 0, 0, []
    tot = _num(df, total)
    present = [p for p in parts if p in df.columns]
    comp = pd.concat([_num(df, p) for p in present], axis=1) if present else None
    if comp is None:
        return 0, 0, len(df), []
    n_present = comp.notna().sum(axis=1)
    complete = n_present == len(present)
    evaluable = tot.notna() & (n_present > 0)
    summed = comp.sum(axis=1, min_count=1)
    diff = (summed - tot).abs()
    over = (summed - tot) > tol                      # impossible regardless of completeness
    ok = int((evaluable & (diff <= tol)).sum())
    bad_mask = evaluable & ((complete & (diff > tol)) | over)
    detail = []
    for _, r in df[bad_mask].head(8).iterrows():
        label = " ".join(str(r.get(k, "")) for k in ("ticker", "name", "fiscal_year") if k in r)
        d = float(diff[r.name])
        detail.append(f"- `{label.strip()}`: components sum off by **{d:,.0f}** "
                      f"(total {float(tot[r.name]):,.0f}, "
                      f"{int(n_present[r.name])}/{len(present)} components present)")
    n_bad = int(bad_mask.sum())
    return ok, n_bad, int(len(df) - ok - n_bad), detail


def main() -> None:
    ap = argparse.ArgumentParser(description="Value-level checks over the persisted sample.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--strict", action="store_true", help="exit 1 on any failing check")
    args = ap.parse_args()

    d = Path(args.dir) / "parquet"
    if not d.exists():
        sys.exit(f"no sample at {d} — run scripts/def14a_verify_sample.py first")

    def load(name: str) -> pd.DataFrame:
        p = d / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    parent = load("def14a_llm")
    execs = load("def14a_executive_comp")
    dcomp = load("def14a_director_comp")
    own = load("def14a_ownership")
    dirs = load("def14a_directors")

    print(f"\n=== SANITY: value-level checks on {len(parent)} filings ===")
    print(f"  exec_comp {len(execs)} rows | director_comp {len(dcomp)} | "
          f"ownership {len(own)} | directors {len(dirs)}\n")
    rep = Report()

    # C1 / C2 -- the filer's own arithmetic
    rep.add("C1", "SCT components sum to `total` (+/- $10)",
            *_sum_identity(execs, _SCT_COMPONENTS, "total", _TOL_USD))
    rep.add("C1b", "director-comp components sum to `total` (+/- $10)",
            *_sum_identity(dcomp, _DIR_COMPONENTS, "total", _TOL_USD))
    rep.add("C2", "audit fee categories sum to `auditor_fees` (+/- $10)",
            *_sum_identity(parent, _FEE_PARTS, "auditor_fees", _TOL_USD))

    # C3 -- parent scalar vs child table, the two places the same number is stored
    ok = bad = na = 0
    detail = []
    if not parent.empty and not execs.empty:
        for _, p in parent.iterrows():
            scalar = pd.to_numeric(pd.Series([p.get("ceo_total_comp")]), errors="coerce").iloc[0]
            name = str(p.get("ceo_name_proxy") or "").strip().lower()
            sub = execs[execs["accession_number"] == p["accession_number"]]
            if pd.isna(scalar) or not name or sub.empty:
                na += 1
                continue
            sub = sub.copy()
            sub["fy"] = pd.to_numeric(sub["fiscal_year"], errors="coerce")
            # `lastname|firstinitial`, the same key the gender consensus groups on. A last-token
            # match fails on PFE, whose ceo_name_proxy is "Albert Bourla, DVM, Ph.D." (last token
            # "Ph.D.") while its SCT row reads "A. Bourla" -- the two tables legitimately use
            # different name FORMS for the same person, which is exactly why person_key exists.
            key = person_key(p.get("ceo_name_proxy"))
            cand = sub[sub["name"].map(person_key) == key] if key else sub.iloc[0:0]
            cand = cand[cand["fy"] == cand["fy"].max()]
            if cand.empty:
                na += 1
                detail.append(f"- `{p['ticker']}`: CEO `{p.get('ceo_name_proxy')}` has no SCT row")
                continue
            child = pd.to_numeric(cand["total"], errors="coerce").iloc[0]
            if pd.isna(child):
                na += 1
            elif abs(child - scalar) <= _TOL_USD:
                ok += 1
            else:
                bad += 1
                detail.append(f"- `{p['ticker']}`: scalar {scalar:,.0f} vs SCT row "
                              f"{child:,.0f} (off by {abs(child - scalar):,.0f})")
    rep.add("C3", "`ceo_total_comp` == the CEO's own SCT row total", ok, bad, na, detail[:8])

    # C4 -- the pay-ratio triplet
    ok = bad = na = 0
    detail = []
    if not parent.empty:
        tot = _num(parent, "ceo_total_comp")
        med = _num(parent, "median_employee_pay")
        ratio = _num(parent, "ceo_pay_ratio")
        for i in parent.index:
            if pd.isna(tot[i]) or pd.isna(med[i]) or pd.isna(ratio[i]) or med[i] == 0:
                na += 1
                continue
            implied = tot[i] / med[i]
            if abs(implied - ratio[i]) <= max(ratio[i] * _TOL_PCT, 1.0):
                ok += 1
            else:
                bad += 1
                detail.append(f"- `{parent.at[i, 'ticker']}`: disclosed ratio {ratio[i]:,.0f} vs "
                              f"{tot[i]:,.0f}/{med[i]:,.0f} = {implied:,.0f}")
    rep.add("C4", "ceo_pay_ratio == ceo_total_comp / median_employee_pay", ok, bad, na, detail[:8])

    # C5 -- board_size against the director rows actually extracted.
    #
    # This check used to demand EQUALITY (+/-1) and reported AEE 12-vs-16, PFE 12-vs-14 and
    # PG 12-vs-15 as failures. They are not: the two counts measure different populations.
    # `board_size` is the CURRENT board -- AEE and PG each state "12 director nominees" in their
    # own words three times over -- while the director rows are everyone the proxy DESCRIBES,
    # which includes those who served during the year and have since left. PG's own footnote
    # says it: "Mr. Lundgren and Ms. Woertz retired from the Board effective October 14, 2025"
    # and "Ms. Lee is not standing for reelection"; AEE's says "former director James C.
    # Johnson, who did not stand for re-election". Those people appear in the DIRECTOR
    # COMPENSATION table because they earned fees, and are absent from the bios at any carve
    # budget -- so demanding equality asks the extraction to DROP disclosed directors.
    #
    # The gap is therefore one-sided, and measured over 22 filings it is: 17 exact, excess
    # +1..+4 on four, and -1 on one (TSCO). So the DEFICIT is the informative direction -- it
    # means the roster was truncated and directors are MISSING -- and the excess is bounded
    # rather than forbidden. The bound is one above the largest explained case; with 22 filings
    # behind it that is a weak bound, but it still catches a roster that swallowed a block of
    # non-directors (NEOs bleeding in from the SCT is the shape to fear).
    ok = bad = na = 0
    detail = []
    excess_max = 5
    if not parent.empty and not dirs.empty:
        counts = dirs.groupby("accession_number").size()
        bs = _num(parent, "board_size")
        for i in parent.index:
            n = counts.get(parent.at[i, "accession_number"])
            if pd.isna(bs[i]) or n is None:
                na += 1
                continue
            gap = n - bs[i]
            if -1 <= gap <= excess_max:
                ok += 1
            else:
                bad += 1
                why = ("roster TRUNCATED -- directors missing" if gap < 0
                       else "more rows than a year of turnover explains")
                detail.append(f"- `{parent.at[i, 'ticker']}`: board_size {bs[i]:.0f} vs "
                              f"{n} director rows ({gap:+.0f}, {why})")
    rep.add("C5", f"director rows within (board_size -1 .. +{excess_max})", ok, bad, na, detail[:8])

    # C6 -- humanly possible ages and tenures
    ok = bad = na = 0
    detail = []
    if not dirs.empty:
        age, ten = _num(dirs, "age"), _num(dirs, "tenure_years")
        for i in dirs.index:
            if pd.isna(age[i]) and pd.isna(ten[i]):
                na += 1
                continue
            bad_age = pd.notna(age[i]) and not (25 <= age[i] <= 95)
            bad_ten = pd.notna(ten[i]) and (ten[i] < 0 or ten[i] > 60)
            impossible = (pd.notna(age[i]) and pd.notna(ten[i]) and ten[i] > age[i] - 20)
            if bad_age or bad_ten or impossible:
                bad += 1
                detail.append(f"- `{dirs.at[i, 'ticker']}` {dirs.at[i, 'name']}: "
                              f"age={age[i]}, tenure={ten[i]}")
            else:
                ok += 1
    rep.add("C6", "director age 25-95, tenure 0-60 and <= age-20", ok, bad, na, detail[:8])

    # C7 -- ownership percents
    ok = bad = na = 0
    detail = []
    if not own.empty:
        pct = _num(own, "percent_of_class")
        for i in own.index:
            if pd.isna(pct[i]):
                na += 1
            elif 0 < pct[i] <= 1.0:
                ok += 1
            else:
                bad += 1
                detail.append(f"- `{own.at[i, 'ticker']}` {own.at[i, 'holder_name']}: "
                              f"percent_of_class={pct[i]}")
        agg = own.assign(p=pct).groupby("accession_number")["p"].sum(min_count=1)
        for acc, s in agg.items():
            if pd.notna(s) and s > 1.0:
                detail.append(f"- accession `{acc}`: holder percents sum to {s:.2f} (> 1.0)")
    rep.add("C7", "percent_of_class in (0, 1] (a fraction, not a percentage)",
            ok, bad, na, detail[:8])

    # C8 -- say-on-pay is a fraction
    ok = bad = na = 0
    detail = []
    if not parent.empty:
        sop = _num(parent, "say_on_pay_support_pct")
        for i in parent.index:
            if pd.isna(sop[i]):
                na += 1
            elif 0 < sop[i] <= 1.0:
                ok += 1
            else:
                bad += 1
                detail.append(f"- `{parent.at[i, 'ticker']}`: say_on_pay={sop[i]}")
    rep.add("C8", "say_on_pay_support_pct in (0, 1]", ok, bad, na, detail[:8])

    # C9 -- gender provenance
    ok = bad = na = 0
    if not dirs.empty:
        g = dirs["gender"] if "gender" in dirs else pd.Series(dtype=object)
        b = dirs["gender_basis"] if "gender_basis" in dirs else pd.Series(dtype=object)
        has_g = g.notna() & (g.astype(str).str.strip() != "")
        has_b = b.notna() & (b.astype(str).str.strip() != "")
        ok = int((has_g & has_b).sum())
        bad = int((has_g & ~has_b).sum())
        na = int((~has_g).sum())
    rep.add("C9", "every director with a gender has a gender_basis", ok, bad, na)

    # C10 -- primary keys
    pks = {"def14a_executive_comp": (execs, ["ticker", "accession_number", "name", "fiscal_year"]),
           "def14a_director_comp": (dcomp, ["ticker", "accession_number", "name"]),
           "def14a_ownership": (own, ["ticker", "accession_number", "holder_name",
                                      "holder_type"]),
           "def14a_directors": (dirs, ["ticker", "accession_number", "name"])}
    ok = bad = 0
    detail = []
    for name, (df, pk) in pks.items():
        if df.empty:
            continue
        cols = [c for c in pk if c in df.columns]
        dups = int(df.duplicated(subset=cols).sum())
        if dups:
            bad += 1
            detail.append(f"- `{name}`: {dups} duplicate rows on {cols}")
        else:
            ok += 1
    rep.add("C10", "no duplicate primary keys in any child table", ok, bad, 0, detail)

    total_ok = sum(r[2] for r in rep.rows)
    total_bad = sum(r[3] for r in rep.rows)
    total_na = sum(r[4] for r in rep.rows)
    print(f"\n  {total_ok} pass / {total_bad} fail / {total_na} not evaluable "
          f"({100 * total_ok / max(total_ok + total_bad, 1):.1f}% of evaluable)")
    print("  An identity failure is diagnostic: it names the row and the size of the gap, so it")
    print("  points at a column, not merely at a filing.")

    md = ["# Value-level checks on the DEF 14A sample", "",
          "Fill rates cannot tell a right number from a plausible wrong one. Every check here is "
          "an **identity the filing itself must satisfy**, so a failure localises the defect.", "",
          f"**{total_ok} pass / {total_bad} fail / {total_na} not evaluable** "
          f"({100 * total_ok / max(total_ok + total_bad, 1):.1f}% of evaluable).", "",
          "| check | what | pass | fail | n/a |", "|---|---|---|---|---|"]
    md += [f"| {c} | {w} | {o} | {b} | {n} |" for c, w, o, b, n in rep.rows]
    md += ["", "A check whose two legs are both absent is counted as **not evaluable**, never as "
           "a pass — an unevaluated check inflating a pass rate is the failure mode this file "
           "exists to avoid.", ""] + rep.lines
    out = Path(args.dir) / "CHECKS.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"  written -> {out}")

    if args.strict and total_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
