"""Recover named table-less Item 5.07 rows from their SEC primary documents.

Dry-run by default. A write requires both an explicit accession list and an existing
database-backup path, then upserts only the selected ``sec_8k`` Item 5.07 rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from edgar import Company

from src.constants.constants import SEC_8K_FORMS
from src.context import Context, get_config_context
from src.data_extract.utils.institutionals.fetch_8k_edgar import _recover_item_507_from_primary
from src.data_store.schema import Tables


def _split(value: str | None) -> list[str]:
    return [part.strip().upper() for part in (value or "").split(",") if part.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="./configs")
    parser.add_argument("--tickers", help="Comma-separated scan scope.")
    parser.add_argument("--accessions", help="Comma-separated exact accession scope; required with --write.")
    parser.add_argument("--backup", type=Path, help="Existing pre-write database backup; required with --write.")
    parser.add_argument("--write", action="store_true", help="Upsert recoverable rows. Default is dry-run.")
    return parser


def _source(context: Context, tickers: list[str], accessions: list[str]) -> pd.DataFrame:
    where: dict[str, object] = {"item": "5.07"}
    if tickers:
        where["ticker"] = tickers
    if accessions:
        where["accession_number"] = accessions
    source = context.store.load(Tables.sec_8k, where=where, optional=True)
    if source is None:
        return pd.DataFrame()
    return source


def _filings_by_accession(context: Context, source: pd.DataFrame) -> dict[str, object]:
    context.ensure_edgar_identity()
    wanted = set(source["accession_number"].astype(str))
    resolved: dict[str, object] = {}
    for cik in source["cik"].dropna().astype(str).unique():
        for filing in Company(cik.zfill(10)).get_filings(form=SEC_8K_FORMS):
            if filing.accession_number in wanted:
                resolved[filing.accession_number] = filing
    return resolved


def _recover(context: Context, source: pd.DataFrame) -> pd.DataFrame:
    filings = _filings_by_accession(context, source)
    recovered: list[dict[str, object]] = []
    for _, row in source.sort_values(["ticker", "filing_date"]).iterrows():
        accession = str(row["accession_number"])
        filing = filings.get(accession)
        if filing is None:
            context.log.warning("%s: accession not found in the stored CIK's 8-K listing", accession)
            continue
        before = str(row.get("item_text") or "")
        after = _recover_item_507_from_primary(filing, before)
        if after == before:
            continue
        replacement = row.to_dict()
        replacement["item_text"] = after
        recovered.append(replacement)
        context.log.info("%s %s: Item 5.07 %d -> %d chars", row["ticker"], accession, len(before), len(after))
    return pd.DataFrame(recovered, columns=source.columns)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tickers = _split(args.tickers)
    accessions = _split(args.accessions)
    if not tickers and not accessions:
        raise SystemExit("provide --tickers for a dry scan or --accessions for an exact scope")
    if args.write and not accessions:
        raise SystemExit("--write requires explicit --accessions")
    if args.write and (args.backup is None or not args.backup.is_file()):
        raise SystemExit("--write requires --backup pointing to an existing pre-write database backup")

    _, context = get_config_context(args.config, use_cache=False, save=False)
    source = _source(context, tickers, accessions)
    recovered = _recover(context, source) if not source.empty else pd.DataFrame()
    recovered_accessions = set(recovered.get("accession_number", pd.Series(dtype=str)).astype(str))
    missing = set(accessions) - recovered_accessions
    if args.write and missing:
        raise SystemExit(f"refusing partial write; requested accession(s) were not recoverable: {sorted(missing)}")

    context.log.info(
        "Item 5.07 recovery: %d source row(s), %d recoverable row(s), write=%s",
        len(source),
        len(recovered),
        args.write,
    )
    if not recovered.empty:
        context.log.info("recoverable accessions: %s", ",".join(recovered["accession_number"].astype(str)))
    if args.write:
        written = context.store.save(Tables.sec_8k, recovered)
        context.log.info("SANITY: upserted %d exact Item 5.07 row(s); no other table was written", written)
    else:
        context.log.info("SANITY: dry-run only; no row was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
