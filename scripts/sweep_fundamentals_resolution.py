"""
sweep_fundamentals_resolution.py (scripts/)
-------------------------------------------
The 52-ticker resolution sweep, as a COMMITTED command instead of a scratchpad script.

Every acceptance number in Phases 3b, 4 and 4b of the fundamentals rebuild was produced by
uncommitted session scripts that no longer exist, so not one of those figures is reproducible
by anyone else -- and Phase 4c's own acceptance is a *before/after on the same join key*,
which needs the sweep run twice. This is that instrument.

**One network pass, two resolutions.** `filing.xbrl()` costs 1.4-5.8 s and dominates
everything else by three orders of magnitude, so each filing is parsed once and resolved
BOTH with and without the 4c.1 statement-role test (`prefer_structure`). The two ledgers land in
one parquet per ticker, tagged by `prefer_structure`, which makes "which rows did 4c.1 move, and
did the value agree?" a groupby rather than a second 35-minute sweep.

Per-ticker parquet, so an interrupted sweep resumes: re-running skips any ticker whose file
already exists unless `--refresh` is passed.

    "$PY" scripts/sweep_fundamentals_resolution.py --roster both
    "$PY" scripts/sweep_fundamentals_resolution.py --roster in_sample -t MCD --refresh

Reads nothing from Postgres. GICS comes from `sp500_tickers` when a DB is reachable and
from the roster file's own `gics` block otherwise, so the sweep runs on a laptop with no
container up -- the regime drives `never_use` and the roll-up overrides, so it cannot be
skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                    # noqa: E402
from dotenv import load_dotenv                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from src.constants.constants import (                                  # noqa: E402
    FUNDAMENTALS_CATALOGUE_SUBDIR, FUNDAMENTALS_FORMS, FUNDAMENTALS_ROSTERS_FILENAME)
from src.data_extract.utils.fundamentals.cik_cutover import (              # noqa: E402
    cutover_filings, load_cutovers)
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import (  # noqa: E402
    rows_from_xbrl)
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue  # noqa: E402

#: Where the ledgers land. Under `data/` because it is a cache of a paid-for network walk,
#: not a source: deleting it costs wall-clock and nothing else.
DEFAULT_OUT = ROOT / "data" / "fundamentals_sweep"

#: The rosters, and WHY each ticker is on its list -- the property that made both of them
#: useful. Owned by `configs/fundamentals/fundamentals_rosters.json`; keys prefixed with `_`
#: are the per-ticker rationale blocks and are not roster names.


def rosters(config_dir: Path) -> dict[str, list[str]]:
    """The roster lists, by name. Raises rather than defaulting: a typo in `--roster` must
    not silently sweep a different 26 tickers than the one the report claims."""
    path = config_dir / FUNDAMENTALS_CATALOGUE_SUBDIR / FUNDAMENTALS_ROSTERS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"roster config missing: {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {name: [t for t in block if not t.startswith("_")]
            for name, block in blob.items() if not name.startswith("_")}


def gics_lookup(tickers: list[str]) -> dict[str, dict[str, str | None]]:
    """GICS for the swept tickers, off `sp500_tickers` when a DB is reachable.

    Returns {} when it is not, and the caller then resolves with `regime=None`. That is a
    real degradation and is printed rather than swallowed: the regime selects `never_use`
    entries and roll-up overrides, so a bank read against Article 5 resolves revenue on a
    different basis. A sweep is only comparable with another sweep that had the same answer
    here.
    """
    levels = ["sector", "industry_group", "sub_industry"]
    try:
        from src.context import get_config_context
        from src.data_store.schema import Tables
        _, context = get_config_context("./configs", use_cache=False, save=False)
        universe = context.store.load(Tables.sp500_tickers, columns=["ticker", *levels],
                                      optional=True)
    except Exception as exc:                                            # noqa: BLE001
        print(f"  ! no GICS ({type(exc).__name__}: {exc}) -- resolving with regime=None")
        return {}
    if universe is None:
        print("  ! sp500_tickers is empty -- resolving with regime=None")
        return {}
    return {row.ticker: {lvl: getattr(row, lvl) for lvl in levels}
            for row in universe.itertuples() if row.ticker in set(tickers)}


def sweep_ticker(ticker: str, catalogue, gics: dict | None,
                 cutovers: dict | None = None) -> pd.DataFrame:
    """One ticker's whole filing history, resolved BOTH ways off one parse per filing.

    Honours `fundamentals_cik_cutover.json` exactly as `build_ticker_fundamentals` does. It
    has to: `Company(ticker)` sees only the current registrant, so without it APA arrives
    with 22 filings instead of ~62 and every rate measured off this ledger would describe a
    pipeline nobody runs.
    """
    from edgar import Company

    company = Company(ticker)
    cik = str(getattr(company, "cik", "")).zfill(10)
    cutover = (cutovers or {}).get(ticker)
    if cutover is not None:
        filings = cutover_filings(cutover, FUNDAMENTALS_FORMS, None, frozenset())
    else:
        filings = sorted(company.get_filings(form=list(FUNDAMENTALS_FORMS)),
                         key=lambda f: pd.Timestamp(f.filing_date))
    frames: list[pd.DataFrame] = []
    for filing in filings:
        try:
            xbrl = filing.xbrl()
        except Exception:                                               # noqa: BLE001
            continue
        if xbrl is None:
            continue
        filing_cik = (cutover.cik_for(filing.filing_date) if cutover is not None else cik)
        for strict in (True, False):
            rows = rows_from_xbrl(ticker, filing_cik, filing, xbrl, catalogue, gics,
                                  prefer_structure=strict)
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            frame["prefer_structure"] = strict
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="both",
                        help="in_sample | out_of_sample | amendment_pair | both | all")
    parser.add_argument("-t", "--ticker", action="append", default=[],
                        help="sweep only these tickers (repeatable)")
    parser.add_argument("-c", "--config-dir", default="./configs")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh", action="store_true",
                        help="re-sweep tickers whose parquet already exists")
    parser.add_argument("--limit", type=int, default=0,
                        help="sweep at most N tickers then exit (0 = all). Drive this in a "
                             "shell loop: edgartools' per-filing caches are not released "
                             "inside one process and an all-52 run reached 14.7 GB RSS "
                             "before it was killed. Process exit is the only reclaim.")
    args = parser.parse_args()

    if not os.getenv("SEC_USER_AGENT", "").strip():
        print("SEC_USER_AGENT unset -- EDGAR refuses anonymous traffic. Aborting.")
        return 2
    from edgar import set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))

    config_dir = Path(args.config_dir)
    all_rosters = rosters(config_dir)
    if args.ticker:
        wanted = {"ad_hoc": list(args.ticker)}
    elif args.roster == "both":
        wanted = {k: all_rosters[k] for k in ("in_sample", "out_of_sample")}
    elif args.roster == "all":
        wanted = all_rosters
    else:
        wanted = {args.roster: all_rosters[args.roster]}

    catalogue = load_catalogue(str(config_dir))
    cutovers = load_cutovers(str(config_dir))
    tickers = [t for names in wanted.values() for t in names]
    gics = gics_lookup(tickers)
    affected = sorted(set(cutovers) & set(tickers))
    if affected:
        print(f"  CIK cutovers in play: "
              + ", ".join(f"{t}@{cutovers[t].cutover_date.date()}" for t in affected))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sweeping {len(tickers)} tickers "
          f"({', '.join(f'{k}={len(v)}' for k, v in wanted.items())}) "
          f"-> {out_dir}  workers={args.workers}")
    todo = [t for t in tickers
            if args.refresh or not (out_dir / f"{t}.parquet").exists()]
    cached = len(tickers) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"  {cached} already cached, {len(todo)} to sweep this pass")
    if not todo:
        print("nothing to do")
        return 0

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(sweep_ticker, t, catalogue, gics.get(t), cutovers): t
                   for t in todo}
        for future in as_completed(futures):
            ticker = futures[future]
            done += 1
            try:
                frame = future.result()
            except Exception as exc:                                    # noqa: BLE001
                print(f"  [{done}/{len(todo)}] {ticker:6s} FAILED "
                      f"{type(exc).__name__}: {exc}")
                continue
            if frame.empty:
                print(f"  [{done}/{len(todo)}] {ticker:6s} no rows")
                continue
            frame.to_parquet(out_dir / f"{ticker}.parquet", index=False)
            filings = frame["accession_number"].nunique()
            print(f"  [{done}/{len(todo)}] {ticker:6s} {len(frame):7,d} rows  "
                  f"{filings:4d} filings  {(time.time() - started) / 60:5.1f} min elapsed")

    files = sorted(out_dir.glob("*.parquet"))
    print(f"\ndone in {(time.time() - started) / 60:.1f} min -- "
          f"{len(files)} ticker ledgers in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
