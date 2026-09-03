"""
def14a_probe_extract.py  (scripts/)
--------------------------------------------------------------------------------------------
Phase-3 paid probe: run the REAL model over a handful of ALREADY-CACHED filings and assert the
things the free replay structurally cannot.

Why this cannot be skipped. `scripts/def14a_replay_flatten.py` proves the flatten and the four
row builders are correct, but it replays *stored* `def14a_json` blobs -- and those predate this
phase, so two of the four child tables come back with **zero rows**:
`def14a_director_compensation` and `def14a_ownership_holders` are new arrays on the Pydantic
contract. Nothing has yet shown that the model actually populates them. Discovering that after
Phase 6's full paid backfill would mean the `>= 90% of post-2008 proxies carry a director-comp
row` gate fails with the tokens already spent.

Reads filings from `data/cache/def14a_probe` (Phase 0 put them there) -- **no SEC requests**.
The only cost is the LLM call per filing, one per probe target.

    "$PY" scripts/def14a_probe_extract.py [-c ./configs] [--targets CAT:2026,PFE:2026] [--dry-run]
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
    _child_frames, _flatten, prepare_def14a_sections,
)

BASELINE = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/baseline"
CACHE_DIR = ROOT / "data/cache/def14a_probe"

#: (ticker, filing year) -> what this filing is here to prove. The plan names MS / TSLA for the
#: 1000x fee cases and HUBB 2022 for gender, but NONE of the three is among Phase 0's 23 baseline
#: tickers, so none is cached; probing them would cost 3 SEC fetches plus a cache miss on every
#: later phase. The substitutes below test the same PROPERTY on filings already on disk.
DEFAULT_TARGETS = (
    ("CAT", 2026, "director comp present (4 of 4 hand-checked values were in the carve)"),
    ("PFE", 2026, "director comp present"),
    ("GE", 2026, "director comp present + the $-in-own-<td> fee/total table"),
    ("AAPL", 2026, "multi-year SCT: NEOs x years, not N rows all on the latest year"),
    ("JPM", 2026, "audit fees in whole USD + the say-on-pay revolt survives"),
    ("XOM", 2026, "ownership holders + audit fees"),
    ("PG", 2026, "ownership holders (the footnote-digit share-count ticker)"),
    ("A", 2000, "pre-2001 ASCII proxy: gender_basis must degrade honestly, not silently"),
)

#: Audit fees outside this band are a units failure, not a disclosure. An S&P 500 audit fee below
#: $100k means a "(in thousands)" note was ignored; above $200M means it was applied twice.
FEE_MIN_USD, FEE_MAX_USD = 1e5, 2e8


def _pick(index: pd.DataFrame, ticker: str, year: int) -> pd.Series | None:
    """The cached filing for (ticker, year) -- latest one if the year has several."""
    m = index[(index["ticker"] == ticker)
              & (pd.to_datetime(index["filing_date"]).dt.year == year)]
    return None if m.empty else m.sort_values("filing_date").iloc[-1]


def _read_cached(filing: pd.Series) -> str:
    """The filing markup from disk.

    Order matters and is not cosmetic: on a pre-2001 filing with an empty `primaryDocument`,
    `doc_url` is a bare DIRECTORY url, so the cached `.htm` is a 1,480-char folder INDEX, not a
    proxy. Reading it first would probe the very bug Phase 1 fixed and report an honest-looking
    "no directors found". Those filings must come from the `<accession>.txt` full submission.
    """
    order = (("cache_txt", "cache_htm") if bool(filing.get("pre_2001_empty_primary"))
             else ("cache_htm", "cache_txt"))
    for col in order:
        name = filing.get(col)
        if isinstance(name, str) and name:
            p = CACHE_DIR / name
            if p.exists() and p.stat().st_size > 0:
                return p.read_bytes().decode("utf-8", "replace")
    raise FileNotFoundError(f"{filing['ticker']} {filing['filing_date']}: nothing cached")


def probe(filing: pd.Series, extractor: LLMExtractor | None) -> dict:
    """One filing end to end: cached markup -> carve -> model -> parent row + child frames."""
    raw = _read_cached(filing)
    text = html_to_text(raw)
    focused = prepare_def14a_sections(raw, text)
    out = {"ticker": filing["ticker"], "filing_date": str(filing["filing_date"])[:10],
           "payload_chars": len(focused)}
    if extractor is None:                       # --dry-run: carve only, no tokens spent
        return out
    extract = extractor.extract(Def14AExtract, focused)
    row = _flatten(filing["ticker"], filing, extract)
    children = _child_frames(filing["ticker"], filing, extract)
    out.update(row=row, children=children)
    return out


def _sct_shape(rows: list[dict]) -> tuple[int, int]:
    """(distinct NEOs, distinct fiscal years) -- the multi-year SCT check."""
    if not rows:
        return 0, 0
    return len({r["name"] for r in rows}), len({r["fiscal_year"] for r in rows})


def report(results: list[dict]) -> bool:
    ok = True
    print("\n=== SANITY: the real model on cached filings ===")
    print(f"{'filing':<18}{'payload':>8}{'NEOs':>6}{'yrs':>5}{'dcomp':>7}{'own':>6}"
          f"{'dirs':>6}  auditor / fees")
    for r in results:
        if "row" not in r:
            print(f"{r['ticker']+' '+r['filing_date']:<18}{r['payload_chars']:>8,}  (dry run)")
            continue
        row, ch = r["row"], r["children"]
        n_neo, n_yr = _sct_shape(ch["def14a_executive_comp"])
        fee = row.get("auditor_fees")
        name = (row.get("auditor_name") or "-")[:34]
        print(f"{r['ticker']+' '+r['filing_date']:<18}{r['payload_chars']:>8,}{n_neo:>6}{n_yr:>5}"
              f"{len(ch['def14a_director_comp']):>7}{len(ch['def14a_ownership']):>6}"
              f"{len(ch['def14a_directors']):>6}  {name}"
              f"{'' if fee is None else f' / ${fee:,.0f}'}")

    live = [r for r in results if "row" in r]
    if not live:
        print("\n  dry run -- no assertions evaluated.")
        return True

    # --- 1. the two NEW arrays must actually be populated by the model -------------------- #
    post2008 = [r for r in live if int(r["filing_date"][:4]) >= 2009]
    with_dcomp = [r for r in post2008 if r["children"]["def14a_director_comp"]]
    with_own = [r for r in live if r["children"]["def14a_ownership"]]
    print(f"\n  director_comp rows on {len(with_dcomp)}/{len(post2008)} post-2008 filings "
          f"(the array the stored blobs could not exercise; Phase 6 gates at > 90%)")
    print(f"  ownership rows on {len(with_own)}/{len(live)} filings")
    ok &= len(with_dcomp) == len(post2008) and len(with_own) >= len(live) - 1

    # --- 2. multi-year SCT ---------------------------------------------------------------- #
    modern = [r for r in live if int(r["filing_date"][:4]) >= 2012]
    years = [_sct_shape(r["children"]["def14a_executive_comp"])[1] for r in modern]
    print(f"  SCT distinct fiscal years per modern filing: {years} "
          f"(Item 402(c) requires 3; 1 means the year column was collapsed)")
    ok &= all(y >= 2 for y in years)

    # --- 3. fee units --------------------------------------------------------------------- #
    fees = [(r["ticker"], r["row"]["auditor_fees"]) for r in live
            if r["row"].get("auditor_fees") is not None]
    bad_fees = [(t, f) for t, f in fees if not FEE_MIN_USD <= f <= FEE_MAX_USD]
    print(f"  audit fees in whole USD: {len(fees) - len(bad_fees)}/{len(fees)} inside "
          f"${FEE_MIN_USD:,.0f}-${FEE_MAX_USD:,.0f}" + (f"  OUTSIDE: {bad_fees}" if bad_fees else ""))
    ok &= not bad_fees

    # --- 4. auditor_name is a NAME, not the sentence around it ---------------------------- #
    names = [(r["ticker"], r["row"]["auditor_name"]) for r in live if r["row"].get("auditor_name")]
    too_long = [(t, n) for t, n in names if len(n) > 60]
    print(f"  auditor_name present on {len(names)}/{len(live)}, all <= 60 chars: "
          f"{not too_long}" + (f"  LONG: {too_long}" if too_long else ""))
    ok &= not too_long

    # --- 5. gender provenance, and its honest degradation on ASCII proxies ---------------- #
    print("\n  gender_basis by filing (the upgrade is only real if provenance is real):")
    for r in live:
        dirs = r["children"]["def14a_directors"]
        if not dirs:
            print(f"    {r['ticker']:<6}{r['filing_date']}  no director rows")
            continue
        dist = pd.Series([d["gender_basis"] for d in dirs]).value_counts(dropna=False).to_dict()
        blank = sum(d["gender"] is not None and not d["gender_basis"] for d in dirs)
        vs = r["row"].get("n_women_directors_vs_inferred")
        print(f"    {r['ticker']:<6}{r['filing_date']}  n={len(dirs):<3} {dist}"
              f"  vs_inferred={vs}")
        ok &= blank == 0

    print(f"\n  {'PASS' if ok else 'FAIL'}: {len(live)} paid filings probed from the on-disk "
          f"cache; 0 SEC requests.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Paid Phase-3 probe over cached DEF 14A filings.")
    ap.add_argument("-c", "--configs", default="./configs")
    ap.add_argument("--targets", default=None,
                    help="comma-separated TICKER:YEAR overriding the defaults")
    ap.add_argument("--dry-run", action="store_true", help="carve only, spend nothing")
    args = ap.parse_args()

    config, context = get_config_context(args.configs, use_cache=False, save=False)
    index = pd.read_parquet(BASELINE / "filings.parquet")

    if args.targets:
        targets = [(t.split(":")[0], int(t.split(":")[1]), "requested")
                   for t in args.targets.split(",")]
    else:
        targets = list(DEFAULT_TARGETS)

    extractor = None
    if not args.dry_run:
        model = config.gpt.llm_model[config.gpt.default_api]
        extractor = LLMExtractor(context, config, action="def14a")
        print(f"model: {model}  ({len(targets)} filings -> {len(targets)} LLM calls)")

    results = []
    for ticker, year, why in targets:
        filing = _pick(index, ticker, year)
        if filing is None:
            print(f"  SKIP {ticker} {year}: not in the cache index")
            continue
        print(f"  {ticker} {year}: {why}")
        try:
            results.append(probe(filing, extractor))
        except Exception as e:
            print(f"    FAILED ({type(e).__name__}: {e})")

    sys.exit(0 if report(results) else 1)


if __name__ == "__main__":
    main()
