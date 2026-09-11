"""
source_coverage_report.py (scripts/source_coverage_report.py)
---------------------------------------------------------------
Emit the ownership/insider source-coverage tables on demand, to
`reports/<date>/source_coverage.md`.

WHY THIS IS A SCRIPT AND NOT A ONE-OFF QUERY. Two facts about these sources existed nowhere in
the repo until they were measured during planning: `sec13f_hr` has a ~13.6x coverage break at
2013-06-30, and three of the five ownership sources start years after the price history does.
Every hard cutoff the feature layer applies is derived from those numbers, so they must be
reproducible on demand rather than pasted from a session that has since scrolled away. Run it at
the end of any backfill: a walk that silently fails to EXTEND coverage shows up here immediately,
where a row count alone would look like progress.

Two tables:
  * TICKER COVERAGE BY YEAR -- distinct tickers present per year per source, against the current
    universe size. Answers "do we have the whole index back to 1995" (we do not).
  * MANAGERS PER TICKER PER QUARTER -- `sec13f_hr` only. This is the table that finds the
    structural break: the counts look equally valid on both sides of it, so nothing short of a
    time series of the coverage itself reveals that pre-2013 quarters describe THE FETCH.

All reads go through `context.store`; there is no SQL in this file, matching `scripts/dod`.
Two shapes are used deliberately:
  * the year table STREAMS (`iter_load`) a two-column projection and accumulates
    `{year: {ticker}}` -- bounded by years x tickers however large the table is;
  * the 13F break table loads ONE PERIOD AT A TIME (~140k rows) rather than streaming 22.5M,
    because distinct-manager-per-ticker cannot be reduced chunk-by-chunk without holding every
    (period, ticker, cik) triple in memory.

Usage:
    python scripts/source_coverage_report.py [-c ./configs] [--since-year 1995]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.constants.constants import DEFAULT_CONFIG_DIR
from src.context import Context, get_config_context
from src.data_store.schema import Tables

#: (label, Table, date column). The sources the institutionals feature families read, plus
#: `prices` and the fundamentals history as reference denominators -- a source that starts late
#: only matters relative to how far the price history goes.
_SOURCES = [
    ("prices", Tables.prices, "date"),
    ("13F", Tables.sec13f_hr, "period"),
    ("13F_mgr", Tables.sec13f_manager_holdings, "period"),
    ("insider", Tables.insider_transactions, "filing_date"),
    ("13D", Tables.sec_13d, "filing_date"),
    ("13G", Tables.sec_13g, "filing_date"),
    ("shortvol", Tables.short_interest, "date"),
    ("FTD", Tables.sec_fails_to_deliver, "date"),
    ("fundamentals", Tables.fundamentals_history_sec, "as_of"),
]

#: `sec13f_manager_holdings` has NO ticker column by design (see its schema docstring), so it is
#: counted by distinct manager CIK. Reported under its own label rather than silently mixing two
#: units into one column.
_COUNT_UNIT = {"sec13f_manager_holdings": "cik"}


def _unit(table) -> str:
    return _COUNT_UNIT.get(table.name, "ticker")


def ticker_coverage(context: Context, since_year: int) -> pd.DataFrame:
    """Distinct tickers (or manager CIKs) present per calendar year, one column per source."""
    since = pd.Timestamp(f"{since_year}-01-01")
    columns: dict[str, pd.Series] = {}
    for label, table, date_col in _SOURCES:
        if not context.store.exists(table):
            continue
        unit = _unit(table)
        if unit not in context.store.columns(table):
            continue
        per_year: dict[int, set] = {}
        for chunk in context.store.iter_load(table, columns=[unit, date_col],
                                             date_col=date_col, since=since):
            years = pd.to_datetime(chunk[date_col], errors="coerce").dt.year
            for year, values in chunk.assign(_y=years).dropna(subset=["_y"]).groupby("_y")[unit]:
                per_year.setdefault(int(year), set()).update(values.dropna().unique())
        if per_year:
            columns[label] = pd.Series({y: len(v) for y, v in per_year.items()})
    if not columns:
        return pd.DataFrame()
    out = pd.DataFrame(columns).sort_index()
    out.index.name = "year"
    return out.astype("Int64")


def managers_per_ticker(context: Context) -> pd.DataFrame:
    """Mean distinct 13F managers per ticker, per quarter -- the break detector.

    Reported beside the ticker count on purpose: managers-per-ticker jumping while the ticker
    count stays flat is a FETCH change, which is exactly what 2013-06-30 was."""
    if not context.store.exists(Tables.sec13f_hr):
        return pd.DataFrame()
    periods = sorted(context.store.distinct(Tables.sec13f_hr, "period"))
    rows = []
    for period in periods:
        df = context.store.load(Tables.sec13f_hr, columns=["ticker", "cik"],
                                where={"period": period}, optional=True)
        if df is None or df.empty:
            continue
        per_ticker = df.groupby("ticker")["cik"].nunique()
        rows.append({"period": pd.Timestamp(period).date(),
                     "managers_per_ticker": round(float(per_ticker.mean()), 1),
                     "tickers": int(per_ticker.size)})
    return pd.DataFrame(rows)


def coverage_breaks(mpt: pd.DataFrame, factor: float = 2.0) -> pd.DataFrame:
    """Quarters whose managers-per-ticker is >= `factor` x the previous quarter's."""
    if mpt.empty or len(mpt) < 2:
        return pd.DataFrame()
    prev = mpt["managers_per_ticker"].shift(1)
    out = mpt.assign(previous=prev,
                     ratio=(mpt["managers_per_ticker"] / prev).round(1)).dropna(subset=["previous"])
    return out[out["ratio"] >= factor]


def _md_table(df: pd.DataFrame, index_label: str | None = None) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    body = df.reset_index() if index_label else df
    if index_label:
        body = body.rename(columns={body.columns[0]: index_label})
    header = "| " + " | ".join(str(c) for c in body.columns) + " |"
    rule = "|" + "|".join("---" for _ in body.columns) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in body.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows]) + "\n"


def build_report(context: Context, since_year: int) -> str:
    universe = context.store.row_count(Tables.sp500_tickers)
    coverage = ticker_coverage(context, since_year)
    mpt = managers_per_ticker(context)
    breaks = coverage_breaks(mpt)

    return "\n".join([
        f"# Source coverage — {pd.Timestamp.today().date()}",
        "",
        f"Universe: **{universe}** tickers. "
        "Generated by `scripts/source_coverage_report.py`.",
        "",
        "## 1. Distinct tickers present per year, by source",
        "",
        "`13F_mgr` counts distinct MANAGER CIKs — that table has no ticker column by design. "
        "Every other column is distinct tickers.",
        "",
        _md_table(coverage, "year"),
        "",
        "## 2. `sec13f_hr` managers per ticker, per quarter",
        "",
        "A jump here with a flat ticker count is a change in the FETCH, not in the market. "
        "The numbers look equally valid on both sides, which is why this table exists.",
        "",
        _md_table(mpt),
        "",
        f"## 3. Coverage breaks (≥2x quarter-on-quarter)",
        "",
        _md_table(breaks) if not breaks.empty else "_None detected._\n",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Source-coverage baseline report.")
    ap.add_argument("-c", "--config-path", default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--since-year", type=int, default=1995)
    ap.add_argument("--out", default=None,
                    help="Override the output path (default reports/<date>/source_coverage.md)")
    args = ap.parse_args()

    _, context = get_config_context(args.config_path, use_cache=False, save=False)
    report = build_report(context, args.since_year)

    out = Path(args.out) if args.out else (
        Path("reports") / str(pd.Timestamp.today().date()) / "source_coverage.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    # The report is written as UTF-8 and read from the file, never echoed: it carries characters
    # (>=, arrows, the section dashes) that a cp1252 Windows console cannot encode, and printing
    # it raised UnicodeEncodeError AFTER a successful write -- a failure that looks like the
    # report was not produced when it was. `print` is also banned by AGENTS.md; the path is the
    # only thing a caller needs.
    context.log.info("source coverage: %d line(s) -> %s", report.count("\n") + 1, out)


if __name__ == "__main__":
    main()
