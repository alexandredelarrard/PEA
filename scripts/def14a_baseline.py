"""
def14a_baseline.py  (scripts/)
--------------------------------------------------------------------------------------------
Freeze TODAY's six DEF 14A tables as an immutable baseline artifact, and cache the raw filings
of the 23 baseline tickers to disk.

Why it must run FIRST: the DEF 14A and 8-K backfills are live (`sec_def14a` moved 26 -> 151
tickers between the research pass and this plan). A baseline read at cutover time would be a
DIFFERENT population than a baseline read today, so the before/after comparison would measure
backfill progress rather than the parser change.

The filing cache is what makes every later phase cheap: the table-anchoring recall harness and
the flatten replay both run off `data/cache/def14a_probe/` with ZERO network and ZERO LLM cost,
so they can be re-run on every edit.

Read-only against Postgres. Writes parquet + `manifest.json` under the plan directory, and raw
filing bytes under `data/cache/def14a_probe/`.

    "$PY" scripts/def14a_baseline.py [-c ./configs] [--out DIR] [--tag baseline] [--no-cache-filings]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants.constants import DEF14A_FORMS, SEC_ARCHIVES_BASE_URL
from src.context import get_config_context
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.sec_utils import load_cik_mapping, sec_get
from src.data_store.schema import Tables

#: NEVER change -- the baseline is keyed to it. Phase 6 must resolve the identical 23 tickers
#: even though `def14a_llm` will have grown, which is why the resolved list is also persisted
#: to `manifest.json` and read back from there when it exists.
RANDOM_SEED = 20260901

#: 13 tickers each pinned to a NAMED defect the research measured, so the comparison proves
#: "the known-wrong values are gone", not merely "fill went up".
DEFECT_TICKERS = ["A", "AMAT", "PG", "SBUX", "BA", "NKE", "CAT", "PFE",
                  "GE", "T", "XOM", "JPM", "AAPL"]
N_RANDOM = 10

#: The six tables the plan compares, plus the Item 5.07 corpus Phase 5 works from. Snapshotting
#: 5.07 is not redundant even though it yields zero vote rows today: it freezes the `item_text`
#: corpus so Phase 5 can prove its fabrication guard's "emit nothing" cases were genuinely
#: tally-free rather than skipped.
SNAPSHOT_TABLES = {
    "def14a_llm": Tables.def14a_llm,
    "sec_def14a": Tables.def14a_edgar,
    "sec_def14a_executive_comp": Tables.def14a_edgar_executive_comp,
    "sec_def14a_director_comp": Tables.def14a_edgar_director_comp,
    "sec_def14a_ownership": Tables.def14a_edgar_ownership,
    "sec_def14a_votes": Tables.def14a_edgar_votes,
}
#: `item_text` is the only wide column and Phase 5 needs it, so the projection is explicit
#: rather than a full read (AGENTS.md: never read a large table unprojected).
SEC_8K_COLS = ["ticker", "cik", "accession_number", "form", "filing_date",
               "period_of_report", "is_amendment", "primary_document", "item", "item_text"]

DEFAULT_OUT = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix"
CACHE_DIR = ROOT / "data/cache/def14a_probe"


def resolve_tickers(context, out_dir: Path) -> list[str]:
    """The 23 baseline tickers: 13 pinned defect cases + 10 drawn from `def14a_llm` under a
    fixed seed. Re-reads a previously resolved list from `manifest.json` so a Phase-6 rerun
    compares the SAME population even after the backfill has added tickers.

    Drawn from `def14a_llm` and not `sp500_tickers` on purpose -- a ticker with no baseline
    rows gives nothing to compare against.
    """
    manifest = out_dir / "baseline" / "manifest.json"
    if manifest.exists():
        pinned = json.loads(manifest.read_text(encoding="utf-8")).get("tickers")
        if pinned:
            return list(pinned)

    universe = sorted(set(context.store.load(Tables.def14a_llm, columns=["ticker"])["ticker"].dropna()))
    pool = [t for t in universe if t not in DEFECT_TICKERS]
    drawn = random.Random(RANDOM_SEED).sample(pool, N_RANDOM)
    return sorted(set(DEFECT_TICKERS) | set(drawn))


def snapshot_tables(context, tickers: list[str], out_dir: Path) -> dict[str, dict]:
    """One parquet per table, restricted server-side to the 23 tickers. Returns the per-table
    summary that goes into the manifest (rows / tickers / accessions / date range)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for name, table in SNAPSHOT_TABLES.items():
        df = context.store.load(table, where={"ticker": tickers}, optional=True)
        df = pd.DataFrame() if df is None else df
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        summary[name] = _describe(df)

    votes = context.store.load(Tables.sec_8k, columns=SEC_8K_COLS,
                               where={"ticker": tickers, "item": "5.07"}, optional=True)
    votes = pd.DataFrame(columns=SEC_8K_COLS) if votes is None else votes
    votes.to_parquet(out_dir / "sec_8k_item507.parquet", index=False)
    summary["sec_8k_item507"] = _describe(votes)
    return summary


def _describe(df: pd.DataFrame) -> dict:
    """Row / ticker / accession counts and the date range, for the manifest and the printout."""
    if df.empty:
        return {"rows": 0, "tickers": 0, "accessions": 0, "date_min": None, "date_max": None}
    date_col = next((c for c in ("as_of", "filing_date", "date") if c in df.columns), None)
    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "accessions": int(df["accession_number"].nunique()) if "accession_number" in df.columns else 0,
        "date_min": None if dates.isna().all() else str(dates.min().date()),
        "date_max": None if dates.isna().all() else str(dates.max().date()),
    }


def _cache_name(ticker: str, filing: pd.Series, suffix: str) -> str:
    date = pd.Timestamp(filing["filing_date"]).strftime("%Y-%m-%d")
    return f"{ticker}_{date}_{filing['accession_number']}{suffix}"


def cache_filings(context, tickers: list[str], years: int, out_dir: Path) -> pd.DataFrame:
    """Download every DEF 14A / DEF 14C of the 23 tickers once and keep the bytes on disk.

    Idempotent by design: an existing file is never re-fetched, so a second run makes zero
    HTTP requests and every later phase re-reads from disk. ~23 x 26 filings at 9 req/s.

    Pre-2001 filings carry `primaryDocument == ""`, so `_doc_url` builds a bare DIRECTORY url
    that returns a ~10 KB EDGAR folder index. BOTH variants are cached -- the folder index under
    `.htm` and the `<accession>.txt` full submission under `.txt` -- because Phase 1's fix is
    precisely "fetch the .txt instead", and proving it needs the before and the after.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cik_map = load_cik_mapping(context, tickers)
    rows, n_fetched, n_hit = [], 0, 0

    for _, r in cik_map.iterrows():
        ticker, cik = r["ticker"], r["cik"]
        try:
            filings = list_filings(context, cik, DEF14A_FORMS, years, r.get("company_name", ""))
        except Exception as e:                              # a dead CIK must not stop the snapshot
            print(f"  ! {ticker}: filing list failed ({e})")
            continue

        for _, f in filings.iterrows():
            primary = (f.get("primary_document") or "").strip()
            acc_nodash = f["accession_number"].replace("-", "")
            txt_url = f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{acc_nodash}/{f['accession_number']}.txt"
            targets = [(f["doc_url"], _cache_name(ticker, f, ".htm"))]
            if not primary:
                targets.append((txt_url, _cache_name(ticker, f, ".txt")))

            saved = {}
            for url, name in targets:
                path = CACHE_DIR / name
                if path.exists():
                    n_hit += 1
                else:
                    try:
                        path.write_bytes(sec_get(context, url).content)
                        n_fetched += 1
                    except Exception as e:
                        print(f"  ! {ticker} {f['accession_number']}: {url} failed ({e})")
                        continue
                saved[path.suffix] = path.name

            rows.append({
                "ticker": ticker, "cik": cik, "form": f["form"],
                "filing_date": f["filing_date"], "period_of_report": f.get("period_of_report"),
                "accession_number": f["accession_number"], "primary_document": primary,
                "doc_url": f["doc_url"], "txt_url": txt_url,
                "cache_htm": saved.get(".htm"), "cache_txt": saved.get(".txt"),
                "pre_2001_empty_primary": not primary,
            })

    index = pd.DataFrame(rows)
    index.to_parquet(out_dir / "filings.parquet", index=False)
    print(f"\nFiling cache: {len(index)} filings — {n_fetched} downloaded, {n_hit} already on disk")
    print(f"  empty `primaryDocument` (pre-2001 shape): {int(index['pre_2001_empty_primary'].sum())}")
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot the DEF 14A tables + cache the filings.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tag", default="baseline", help="subdirectory under --out (use 'pre-cutover' at Phase 6)")
    ap.add_argument("--no-cache-filings", action="store_true", help="snapshot the tables only")
    args = ap.parse_args()

    config, context = get_config_context(args.config, use_cache=False, save=False)
    out_root = Path(args.out)
    out_dir = out_root / args.tag

    tickers = resolve_tickers(context, out_root)
    print(f"Baseline tickers ({len(tickers)}): {', '.join(tickers)}")

    summary = snapshot_tables(context, tickers, out_dir)
    print(f"\nSnapshot -> {out_dir}")
    print(f"{'table':<28}{'rows':>8}{'tickers':>9}{'accessions':>12}  date range")
    for name, s in summary.items():
        span = f"{s['date_min']} .. {s['date_max']}" if s["rows"] else "-"
        print(f"{name:<28}{s['rows']:>8}{s['tickers']:>9}{s['accessions']:>12}  {span}")

    n_filings = None
    if not args.no_cache_filings:
        n_filings = len(cache_filings(context, tickers, config.data_extract.years_history, out_dir))

    (out_dir / "manifest.json").write_text(json.dumps({
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "defect_tickers": DEFECT_TICKERS,
        "tickers": tickers,
        "years_history": config.data_extract.years_history,
        "tables": summary,
        "filings_cached": n_filings,
    }, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
