"""
The Phase 0 safety net for the fundamentals replay refactor (Phases 1-4 of
`reports/planning/active-tasks/2026-08-27-refactor-fundamentals/`): freeze a sample of
`fundamentals_facts`, replay `build_ticker` over the frozen parquet, and diff one replay
against another CELL-EXACT.

`_latest_per_window` (`periods.py:196`) picks the LATEST-FILED vintage per window, so
`build_ticker` genuinely returns different values at different events once a restatement
has arrived. An optimisation that gets the visible-set boundary wrong changes history
silently, and a tolerance in the comparison below would forgive exactly that class of bug --
so there is none. `compare()` hard-asserts the 69-column contract and every dtype, then
reports every differing VALUE for the caller to judge.

Importable by `test_replay_equality.py`, and runnable as a script:
    python -m tests.data_extract.fundamentals.replay_equality freeze <out_dir> [--cap N] TICKER...
    python -m tests.data_extract.fundamentals.replay_equality snapshot <frozen_dir> <out_dir> TICKER...
    python -m tests.data_extract.fundamentals.replay_equality compare <before_dir> <after_dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data_extract.utils.fundamentals.build_history import (
    FACT_COLUMNS, TickerHistory, build_ticker, diff_against_stored,
)
from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue, load_catalogue
from src.data_extract.utils.fundamentals.periods import PeriodGuards, load_guards

ROOT = Path(__file__).resolve().parents[3]


def head_sha(root: Path | None = None) -> str:
    """Current git HEAD sha, or `"unknown"` off a machine with no git. `git` is affordable
    here: the harness runs a handful of times per phase, never inside a per-event loop."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root or ROOT,
                             capture_output=True, text=True, timeout=15, check=False)
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def truncate_by_accession(facts: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """The first `cap` distinct `accession_number`s, in filing-date order, and every fact
    row those accessions carry. Never a row-count truncation -- that would cut one filing in
    half and the replay would see a filing that reported 3 fields instead of 40."""
    if cap is None or facts.empty:
        return facts
    order = (facts[["accession_number", "filing_date"]]
             .drop_duplicates("accession_number")
             .sort_values("filing_date"))
    keep = set(order["accession_number"].head(cap))
    return facts[facts["accession_number"].isin(keep)].copy()


# --------------------------------------------------------------------------- freeze ---

def freeze_inputs(context, tickers: list[str], out_dir: Path, *,
                  filing_cap: int | None = None) -> dict:
    """One projected read per ticker of `Tables.fundamentals_facts`, optionally truncated to
    `filing_cap` filings, written to `out_dir/<ticker>.parquet`. Writes and returns the
    manifest `verify_live_matches_manifest` checks later -- HEAD, the cap, and per-ticker
    row/filing counts and filing-date range -- so a re-freeze that silently reads different
    facts is visible rather than producing fabricated diffs downstream."""
    from src.data_store.schema import Tables            # local: avoids a package cycle

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"head": head_sha(), "filing_cap": filing_cap, "tickers": {}}
    for ticker in tickers:
        facts = context.store.load(Tables.fundamentals_facts, columns=list(FACT_COLUMNS),
                                   where={"ticker": ticker}, optional=True)
        if facts is None:
            raise ValueError(f"{ticker}: no rows in fundamentals_facts -- not a valid "
                             "replay-sample member")
        facts = truncate_by_accession(facts, filing_cap)
        facts.to_parquet(out_dir / f"{ticker}.parquet", index=False)
        manifest["tickers"][ticker] = {
            "rows": int(len(facts)),
            "filings": int(facts["accession_number"].nunique()),
            "min_filing_date": str(pd.to_datetime(facts["filing_date"]).min().date()),
            "max_filing_date": str(pd.to_datetime(facts["filing_date"]).max().date()),
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_live_matches_manifest(context, frozen_dir: Path,
                                 tickers: list[str]) -> list[str]:
    """The moving-target guard for `--source db` mode: a fresh, UNCAPPED row-count read of
    each sample ticker, checked against the frozen manifest. Returns the tickers whose live
    count has moved since the freeze; empty means the freeze is still valid. Read-only --
    safe to run alongside the in-flight walk, never mid-phase re-freezing."""
    from src.data_store.schema import Tables

    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("filing_cap") is not None:
        raise ValueError("verify_live_matches_manifest needs an UNCAPPED (tier B) freeze -- "
                         "a capped manifest's row count is not the ticker's live row count")
    moved = []
    for ticker in tickers:
        live = context.store.load(Tables.fundamentals_facts, columns=["ticker"],
                                  where={"ticker": ticker}, optional=True)
        live_rows = 0 if live is None else len(live)
        if live_rows != manifest["tickers"][ticker]["rows"]:
            moved.append(ticker)
    return moved


# --------------------------------------------------------------------------- replay ---

def replay(frozen_dir: Path, tickers: list[str], *, catalogue: Catalogue | None = None,
          guards: PeriodGuards | None = None) -> dict[str, TickerHistory]:
    """`build_ticker` per ticker off its frozen parquet -- nothing else touches the DB."""
    catalogue = catalogue or load_catalogue()
    guards = guards or load_guards()
    return {ticker: build_ticker(ticker, pd.read_parquet(frozen_dir / f"{ticker}.parquet"),
                                 catalogue=catalogue, guards=guards)
           for ticker in tickers}


def snapshot(results: dict[str, TickerHistory], out_dir: Path) -> None:
    """`<ticker>__history.parquet` + `<ticker>__codes.parquet` per replayed ticker."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, built in results.items():
        built.history.to_parquet(out_dir / f"{ticker}__history.parquet", index=False)
        built.reason_codes.to_parquet(out_dir / f"{ticker}__codes.parquet", index=False)


# -------------------------------------------------------------------------- compare ---

@dataclass
class ComparisonReport:
    """The gate's verdict. `ok` is the single bool everything else exists to explain."""

    tickers: list[str] = field(default_factory=list)
    rows_before: dict[str, int] = field(default_factory=dict)
    rows_after: dict[str, int] = field(default_factory=dict)
    cells_differing: dict[str, int] = field(default_factory=dict)
    first_10_diffs: list[tuple] = field(default_factory=list)
    codes_added: dict[str, int] = field(default_factory=dict)
    codes_removed: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (sum(self.cells_differing.values()) == 0
               and sum(self.codes_added.values()) == 0
               and sum(self.codes_removed.values()) == 0)

    def summary(self) -> str:
        lines = [f"{'OK' if self.ok else 'FAIL'}: {len(self.tickers)} ticker(s)"]
        for t in self.tickers:
            lines.append(f"  {t}: rows {self.rows_before[t]}->{self.rows_after[t]}, "
                        f"{self.cells_differing[t]} cell(s) differing, "
                        f"codes +{self.codes_added[t]}/-{self.codes_removed[t]}")
        return "\n".join(lines)


def compare(before_dir: Path, after_dir: Path) -> ComparisonReport:
    """The actual gate. Cell-exact, not "close enough" -- see the module docstring."""
    report = ComparisonReport()
    tickers = sorted({p.name.split("__")[0] for p in before_dir.glob("*__history.parquet")})
    report.tickers = tickers
    for ticker in tickers:
        before = pd.read_parquet(before_dir / f"{ticker}__history.parquet")
        after = pd.read_parquet(after_dir / f"{ticker}__history.parquet")
        report.rows_before[ticker] = len(before)
        report.rows_after[ticker] = len(after)
        # the 69-column contract
        assert list(before.columns) == list(after.columns), (
            f"{ticker}: the column contract moved: {list(before.columns)} "
            f"vs {list(after.columns)}")
        # TEXT-vs-float64 drift (the VRT/APA bug this harness exists to catch)
        moved_dtype = before.dtypes[before.dtypes != after.dtypes]
        assert moved_dtype.empty, (
            f"{ticker}: dtype drift on {list(moved_dtype.index)}: "
            f"{dict(moved_dtype)} vs {dict(after.dtypes[moved_dtype.index])}")

        diffs = _cell_diffs(before, after)
        report.cells_differing[ticker] = len(diffs)
        report.first_10_diffs.extend((ticker, *d) for d in diffs[:10])

        before_codes = pd.read_parquet(before_dir / f"{ticker}__codes.parquet")
        after_codes = pd.read_parquet(after_dir / f"{ticker}__codes.parquet")
        # reason codes are a SET per (ticker, as_of, field): row order after a groupby
        # rewrite is not part of the contract, so compare as sets, not positionally. NaN is
        # normalised to `None` first -- two NaN `rejected_value` cells are the SAME absence,
        # but `float("nan") != float("nan")`, which would otherwise report every unchanged
        # NaN-carrying code row as both added and removed.
        before_set = _code_rowset(before_codes)
        after_set = _code_rowset(after_codes)
        report.codes_added[ticker] = len(after_set - before_set)
        report.codes_removed[ticker] = len(before_set - after_set)
    return report


def _code_rowset(codes: pd.DataFrame) -> set[tuple]:
    normalised = codes.astype(object).where(codes.notna(), None)
    return set(map(tuple, normalised.itertuples(index=False, name=None)))


def _cell_diffs(before: pd.DataFrame, after: pd.DataFrame) -> list[tuple]:
    """(as_of, column, before, after) for every cell that changed, keyed on `as_of` -- the
    replay's grain (`_assert_grain`: one row per `as_of`, no duplicates). Row order is
    explicitly not part of the contract, so both frames are re-indexed before comparing."""
    left = before.set_index("as_of").sort_index()
    right = after.set_index("as_of").sort_index()
    out: list[tuple] = []
    only_before = left.index.difference(right.index)
    only_after = right.index.difference(left.index)
    out.extend((as_of, "<row>", "present", "missing") for as_of in only_before)
    out.extend((as_of, "<row>", "missing", "present") for as_of in only_after)
    shared = left.index.intersection(right.index)
    l, r = left.loc[shared], right.loc[shared]
    for column in l.columns:
        a, b = l[column], r[column]
        both_na = a.isna() & b.isna()
        changed = ~both_na & ((a.isna() != b.isna()) | (a != b))
        out.extend((as_of, column, a.loc[as_of], b.loc[as_of])
                  for as_of in a.index[changed])
    return out


def compare_against_stored(context, frozen_dir: Path,
                          tickers: list[str]) -> dict[str, pd.DataFrame]:
    """`--source db` mode: replay off the frozen parquet, diff against what is actually
    STORED in `fundamentals_history_sec` via the production `diff_against_stored` (which
    already normalises the Postgres DATE-vs-`Timestamp` round trip a parquet-only harness
    would hide). Read-only. An empty frame per ticker means no drift; run once per phase."""
    from src.data_store.schema import Tables

    catalogue = load_catalogue()
    guards = load_guards()
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        rebuilt = build_ticker(
            ticker, pd.read_parquet(frozen_dir / f"{ticker}.parquet"),
            catalogue=catalogue, guards=guards).history
        stored = context.store.load(Tables.fundamentals_history_sec,
                                    where={"ticker": ticker}, optional=True)
        out[ticker] = diff_against_stored(stored, rebuilt)
    return out


# ------------------------------------------------------------------------------ CLI ---

def _load_sample(cli_tickers: list[str]) -> list[str]:
    if cli_tickers:
        return cli_tickers
    sample_path = Path(__file__).with_name("replay_sample.json")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    return [row["ticker"] for row in sample["tickers"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("out_dir", type=Path)
    p_freeze.add_argument("--cap", type=int, default=None)
    p_freeze.add_argument("tickers", nargs="*")

    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("frozen_dir", type=Path)
    p_snap.add_argument("out_dir", type=Path)
    p_snap.add_argument("tickers", nargs="*")

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("before_dir", type=Path)
    p_cmp.add_argument("after_dir", type=Path)

    args = parser.parse_args()

    if args.action == "compare":
        report = compare(args.before_dir, args.after_dir)
        print(report.summary())
        if not report.ok:
            print("\nfirst diffs:", report.first_10_diffs[:10])
        raise SystemExit(0 if report.ok else 1)

    from src.context import get_config_context   # local: only the CLI path needs a real DB

    _, context = get_config_context("./configs", use_cache=False, save=False)
    tickers = _load_sample(args.tickers)
    if args.action == "freeze":
        manifest = freeze_inputs(context, tickers, args.out_dir, filing_cap=args.cap)
        print(f"froze {len(tickers)} ticker(s) to {args.out_dir} (cap={args.cap}): "
             f"{ {t: v['rows'] for t, v in manifest['tickers'].items()} }")
    elif args.action == "snapshot":
        results = replay(args.frozen_dir, tickers)
        snapshot(results, args.out_dir)
        print(f"replayed + snapshotted {len(tickers)} ticker(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
