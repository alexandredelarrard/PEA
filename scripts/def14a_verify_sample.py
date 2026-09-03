"""
def14a_verify_sample.py  (scripts/)
--------------------------------------------------------------------------------------------
Build an INSPECTABLE sample of the DEF 14A extraction: run the real model over one cached
filing per ticker, PERSIST the parent row and all four child tables as parquet, render them as
markdown you can read next to the actual proxy, and cross-check the CEO's total compensation
against the filer's own XBRL tag.

Why this exists. The Phase-3 probe printed row COUNTS and threw the rows away, so "100% fill"
and "0 rows above $1e9" were the only claims it could support -- shape checks, not correctness.
An absent NULL is not a correct value. This script produces the artifact needed to judge the
values themselves, and adds the one strong automated check available:

    `def14a_llm.ceo_total_comp`      <- the LLM reading a Summary Compensation Table in PROSE
    `sec_def14a.peo_total_comp`      <- `ecd:PeoTotalCompAmt`, tagged by the FILER in inline XBRL

Two entirely unrelated code paths reading the same filing. When they agree to the dollar, the
prose extraction is right; when they disagree, one of them is wrong and the gap says which
(a 1000x gap is a units bug, a ~2x gap is usually a co-PEO year, a small gap is a rounding or
a "Total Without Change in Pension Value" column confusion).

Writes nothing to the database. Reads filings from `data/cache/def14a_probe` -- no SEC requests
for the LLM half. The ECD half needs `filing.xbrl()` (one archive fetch per filing) unless you
pass `--ecd-parquet` from an earlier `def14a_ecd_acceptance.py --out` run.

    "$PY" scripts/def14a_verify_sample.py [-c ./configs] [--tickers A,AAPL] [--year 2026]
                                          [--out DIR] [--ecd-parquet FILE] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.fetch_def14a_llm import (
    _CHILD_SPEC, _child_frames, _flatten, prepare_def14a_sections,
)

BASELINE = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/baseline"
CACHE_DIR = ROOT / "data/cache/def14a_probe"
DEFAULT_OUT = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/verify"

#: Agreement bands for the LLM-vs-XBRL cross-check on CEO total compensation.
_EXACT_USD = 1.0          # same number, modulo float noise
_CLOSE_PCT = 0.01         # within 1% -- rounding or a footnote-level difference

#: Scalars worth eyeballing per filing, in reading order.
_HEADLINE = [
    "company_name", "fiscal_year_extract", "board_size", "n_directors",
    "pct_female_directors", "pct_gender_stated", "n_women_directors_vs_inferred",
    "avg_director_age", "avg_board_tenure", "pct_independent_directors",
    "ceo_name_proxy", "ceo_age", "ceo_since_year", "ceo_salary", "ceo_bonus",
    "ceo_stock_awards", "ceo_option_awards", "ceo_non_equity_incentive",
    "ceo_all_other_comp", "ceo_total_comp", "ceo_equity_pay_pct",
    "n_neos", "sct_years", "total_neo_comp",
    "insider_ownership_pct", "ceo_ownership_pct", "n_five_percent_holders",
    "say_on_pay_support_pct", "ceo_pay_ratio", "median_employee_pay",
    "independent_chair", "lead_independent_director", "classified_board",
    "dual_class_shares", "poison_pill", "majority_voting",
    "auditor_name", "auditor_since_year", "auditor_fees", "audit_fees_audit",
    "audit_fees_audit_related", "audit_fees_tax", "audit_fees_other", "auditor_fees_prior",
]


def _read_cached(filing: pd.Series) -> str:
    """The filing markup from disk. A pre-2001 filing with an empty `primaryDocument` must come
    from the `.txt` full submission -- its cached `.htm` is a directory INDEX."""
    order = (("cache_txt", "cache_htm") if bool(filing.get("pre_2001_empty_primary"))
             else ("cache_htm", "cache_txt"))
    for col in order:
        name = filing.get(col)
        if isinstance(name, str) and name:
            p = CACHE_DIR / name
            if p.exists() and p.stat().st_size > 0:
                return p.read_bytes().decode("utf-8", "replace")
    raise FileNotFoundError(f"{filing['ticker']} {filing['filing_date']}: nothing cached")


def _pick(index: pd.DataFrame, ticker: str, year: int | None) -> pd.Series | None:
    m = index[index["ticker"] == ticker]
    if year is not None:
        m = m[pd.to_datetime(m["filing_date"]).dt.year == year]
    return None if m.empty else m.sort_values("filing_date").iloc[-1]


def extract_one(filing: pd.Series, extractor: LLMExtractor | None) -> tuple[dict, dict, int]:
    raw = _read_cached(filing)
    focused = prepare_def14a_sections(raw, html_to_text(raw))
    if extractor is None:
        return {}, {}, len(focused)
    ex = extractor.extract(Def14AExtract, focused)
    ticker = filing["ticker"]
    return _flatten(ticker, filing, ex), _child_frames(ticker, filing, ex), len(focused)


# --------------------------------------------------------------------------- #
# markdown rendering                                                          #
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _table(rows: list[dict], cols: list[str], limit: int = 40) -> list[str]:
    cols = [c for c in cols if any(c in r for r in rows)]
    if not rows or not cols:
        return ["_(no rows)_", ""]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows[:limit]:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    if len(rows) > limit:
        out.append(f"| _… {len(rows) - limit} more_ |" + " |" * (len(cols) - 1))
    return out + [""]


def render(results: list[dict]) -> str:
    md = ["# DEF 14A extraction — inspectable sample", "",
          "One cached filing per ticker, extracted with the production model and prompt. "
          "**Every number below came out of the LLM**; the `doc_url` on each section is the "
          "filing it was read from, so any cell can be checked against the source.", "",
          "Nothing here was written to the database.", ""]
    for r in results:
        row, ch = r["row"], r["children"]
        md += [f"## {r['ticker']} — filed {r['filing_date']}", "",
               f"- source: <{r['doc_url']}>",
               f"- accession: `{r['accession_number']}`",
               f"- carve payload: {r['payload_chars']:,} chars", ""]
        md += ["### Filing-level scalars", "", "| field | value |", "|---|---|"]
        md += [f"| `{k}` | {_fmt(row.get(k))} |" for k in _HEADLINE if k in row]
        md += ["", "### Summary Compensation Table (`def14a_executive_comp`)", ""]
        md += _table(ch["def14a_executive_comp"],
                     ["name", "title", "fiscal_year", "salary", "bonus", "stock_awards",
                      "option_awards", "non_equity_incentive", "pension_change",
                      "other_compensation", "total", "reconciles"])
        md += ["### Director compensation (`def14a_director_comp`)", ""]
        md += _table(ch["def14a_director_comp"],
                     ["name", "fiscal_year", "fees_earned", "stock_awards", "option_awards",
                      "non_equity_incentive", "pension_change", "other_compensation", "total",
                      "reconciles"])
        md += ["### Beneficial ownership (`def14a_ownership`)", ""]
        md += _table(ch["def14a_ownership"],
                     ["holder_name", "holder_type", "shares", "percent_of_class"])
        md += ["### Directors (`def14a_directors`)", ""]
        md += _table(ch["def14a_directors"],
                     ["name", "age", "tenure_years", "is_independent", "gender", "gender_basis",
                      "other_public_company_boards"])
        md += ["---", ""]
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# the cross-check that actually validates a VALUE                             #
# --------------------------------------------------------------------------- #
def crosscheck(parent: pd.DataFrame, ecd: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """LLM `ceo_total_comp` vs the filer's `ecd:PeoTotalCompAmt`, joined on ACCESSION.

    Joining on the accession rather than the ticker+year is what makes this a real check: both
    numbers then come from the SAME document, so a disagreement cannot be a vintage artefact.
    """
    left = parent[["ticker", "accession_number", "ceo_name_proxy", "ceo_total_comp",
                   "fiscal_year_extract"]].copy()
    right = ecd[["accession_number", "peo_name", "peo_total_comp", "n_peos",
                 "ecd_period_end"]].copy()
    m = left.merge(right, on="accession_number", how="inner", suffixes=("_llm", "_xbrl"))
    m["llm"] = pd.to_numeric(m["ceo_total_comp"], errors="coerce")
    m["xbrl"] = pd.to_numeric(m["peo_total_comp"], errors="coerce")
    m["abs_diff"] = (m["llm"] - m["xbrl"]).abs()
    m["rel_diff"] = m["abs_diff"] / m["xbrl"].abs()

    def verdict(r) -> str:
        if pd.isna(r["llm"]) or pd.isna(r["xbrl"]):
            return "no comparison"
        if r["abs_diff"] <= _EXACT_USD:
            return "EXACT"
        if r["rel_diff"] <= _CLOSE_PCT:
            return "within 1%"
        if 900 <= r["rel_diff"] <= 1100 or 0.0009 <= r["rel_diff"] <= 0.0011:
            return "1000x — UNITS BUG"
        return "DISAGREE"

    m["verdict"] = m.apply(verdict, axis=1)
    comparable = m[m["verdict"] != "no comparison"]
    agree = int(m["verdict"].isin(["EXACT", "within 1%"]).sum())

    md = ["# Cross-check — LLM prose extraction vs the filer's own XBRL tag", "",
          "`def14a_llm.ceo_total_comp` is read by the model out of a Summary Compensation Table "
          "in prose. `sec_def14a.peo_total_comp` is `ecd:PeoTotalCompAmt`, tagged by the filer "
          "in inline XBRL and parsed deterministically. **The two paths share no code.** Joined "
          "on the accession number, so both numbers come from the same document.", "",
          f"**{agree} of {len(comparable)} comparable filings agree** "
          f"({100 * agree / max(len(comparable), 1):.1f}%).", "",
          "A disagreement is diagnostic, not just a miss: ~1000x is a units bug, ~2x is usually "
          "a co-PEO year (`n_peos > 1`, where the two paths legitimately pick different people, "
          "and the XBRL is per-individual while the SCT row is the CEO), and a few percent is "
          "normally a `Total Without Change in Pension Value` column confusion.", "",
          "| ticker | CEO (LLM) | PEO (XBRL) | LLM total | XBRL total | diff | rel | n_peos | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.sort_values(["verdict", "ticker"]).iterrows():
        md.append(f"| {r['ticker']} | {_fmt(r['ceo_name_proxy'])} | {_fmt(r['peo_name'])} | "
                  f"{_fmt(r['llm'])} | {_fmt(r['xbrl'])} | {_fmt(r['abs_diff'])} | "
                  f"{'' if pd.isna(r['rel_diff']) else f'{r["rel_diff"]:.3%}'} | "
                  f"{_fmt(r['n_peos'])} | **{r['verdict']}** |")
    return m, "\n".join(md + [""])


def main() -> None:
    ap = argparse.ArgumentParser(description="Persist and render an inspectable DEF 14A sample.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--tickers", default=None, help="default: every ticker in the Phase-0 cache")
    ap.add_argument("--year", type=int, default=None, help="filing year (default: most recent)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--ecd-parquet", default=None,
                    help="rows from def14a_ecd_acceptance.py --out (skips the XBRL fetch)")
    ap.add_argument("--dry-run", action="store_true", help="carve only, spend nothing")
    args = ap.parse_args()

    config, context = get_config_context(args.config, use_cache=False, save=False)
    index = pd.read_parquet(BASELINE / "filings.parquet")
    tickers = (args.tickers.split(",") if args.tickers
               else sorted(index["ticker"].dropna().unique().tolist()))

    extractor = None
    if not args.dry_run:
        model = config.gpt.llm_model[config.gpt.default_api]
        print(f"model: {model} — {len(tickers)} filings, {len(tickers)} LLM calls")
        extractor = LLMExtractor(context, config, action="def14a")

    results = []
    for ticker in tickers:
        filing = _pick(index, ticker, args.year)
        if filing is None:
            print(f"  SKIP {ticker}: no cached filing")
            continue
        try:
            row, children, n_chars = extract_one(filing, extractor)
        except Exception as e:                                  # noqa: BLE001
            print(f"  FAIL {ticker}: {type(e).__name__}: {e}")
            continue
        n_sct = len(children.get("def14a_executive_comp", []))
        print(f"  {ticker:<6}{str(filing['filing_date'])[:10]}  {n_chars:>7,} chars  "
              f"sct={n_sct:<3} dcomp={len(children.get('def14a_director_comp', [])):<3} "
              f"own={len(children.get('def14a_ownership', [])):<3} "
              f"dirs={len(children.get('def14a_directors', [])):<3}")
        results.append(dict(ticker=ticker, filing_date=str(filing["filing_date"])[:10],
                            accession_number=filing["accession_number"],
                            doc_url=filing["doc_url"], payload_chars=n_chars,
                            row=row, children=children))

    if args.dry_run or not results:
        print(f"\n  {len(results)} filings carved (dry run — nothing extracted or written).")
        return

    out = Path(args.out)
    (out / "parquet").mkdir(parents=True, exist_ok=True)
    parent = pd.DataFrame([r["row"] for r in results])
    parent.to_parquet(out / "parquet" / "def14a_llm.parquet", index=False)
    for name in _CHILD_SPEC:
        rows = [row for r in results for row in r["children"][name]]
        pd.DataFrame(rows).to_parquet(out / "parquet" / f"{name}.parquet", index=False)
        print(f"  {name:<24}{len(rows):>6} rows")

    (out / "VERIFY.md").write_text(render(results), encoding="utf-8")
    print(f"\n  written -> {out / 'VERIFY.md'}")

    if args.ecd_parquet and Path(args.ecd_parquet).exists():
        ecd = pd.read_parquet(args.ecd_parquet)
        m, md = crosscheck(parent, ecd)
        (out / "CROSSCHECK.md").write_text(md, encoding="utf-8")
        m.to_parquet(out / "parquet" / "crosscheck.parquet", index=False)
        print("\n=== SANITY: LLM prose vs the filer's XBRL tag (same accession) ===")
        for v, n in m["verdict"].value_counts().items():
            print(f"  {v:<20}{n:>4}")
        print(f"  written -> {out / 'CROSSCHECK.md'}")
    else:
        print("  (no --ecd-parquet given, so the cross-check was skipped)")


if __name__ == "__main__":
    main()
