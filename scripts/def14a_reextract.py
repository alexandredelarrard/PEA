"""Re-extract a NAMED SET of DEF 14A filings through the current schema, and nothing else.

`fetch_def14a_llm` is year-incremental by design: `existing_filings` builds a `seen` set of
accessions and no accession in it is ever sent to the LLM again. That is the right default --
the tokens are already paid -- but it means a SCHEMA change cannot reach the archive. The rows
extracted under the old schema stay as they were, forever, and the only lever the production
fetcher offers is "delete rows and re-run everything", which on 12,343 filings is ~$284.

This script is the targeted lever. It resolves a scope to a set of accessions, prices the work
BEFORE spending anything, and re-extracts only those, reusing production's own carve
(`_payload_for`), schema (`Def14AExtract`) and flatten (`_result_frames`) so the rows it writes
are indistinguishable from rows production would write.

    --scope dual-class      the filings whose ownership table can print TWO percent columns
                            (`dual_class_shares > 0`), which is the only population that can
                            carry the voting-power-as-ownership swap
    --scope no-director-comp  filings that yielded ZERO Item 402(k) director-compensation rows
    --scope both            the UNION, deduplicated -- a filing in both scopes is extracted
                            ONCE, because one call fills the whole schema
    --scope accessions      an explicit list via --accessions

Dry run by default: it lists the work and prints the cost at the measured $0.023/filing and
does not call the model. `--write` is the only thing that spends money.

⚠ CHILD ROWS ARE DELETED BEFORE THE RE-INSERT (and BACKED UP to `BACKUP` first), and that is
deliberate. The four child tables
key on (ticker, accession_number, name); `save` upserts, so a director the new extract spells
differently, or drops, would leave the OLD row behind and the filing would appear to have both.
A re-extraction replaces a filing's children rather than merging with them. The parent row
keys on (ticker, accession_number) and is upserted in place.

Usage:
    PY=...python.exe
    rtk "$PY" scripts/def14a_reextract.py --scope dual-class --limit 20        # price + list
    rtk "$PY" scripts/def14a_reextract.py --scope dual-class --limit 20 --write
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import pandas as pd

from src.constants.constants import DEF14A_FORMS
from src.context import Context, get_config_context
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.def14a.fetch import _payload_for
from src.data_extract.utils.structure.def14a.flatten import _CHILD_TABLES, _result_frames
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmTask

#: MEASURED, not quoted from a price list: $0.023 per filing on the live archive, where output
#: tokens are 87% of the bill. Size an estimate on filings LISTED, never on rows stored -- a
#: filing that yields 40 child rows and one that yields 3 cost the same to read.
USD_PER_FILING = 0.023

#: Child rows are backed up here before a re-extraction deletes them (see `flush`). Recovery is
#: a plain `COPY ... FROM` of the CSV, and the files are small enough to keep.
BACKUP = Path("reports/validate/governance/_cache/reextract_child_backup")

SCOPES = {
    # The only population that CAN carry the swap: a single-class proxy prints one percent
    # column, so there is no second column to take by mistake.
    "dual-class": """
        SELECT ticker, accession_number, as_of::date AS as_of
        FROM   def14a_llm
        WHERE  dual_class_shares > 0
        ORDER  BY ticker, as_of
    """,
    # Zero Item 402(k) rows. `n_director_comp_rows` is the parent's own count of what the
    # extract produced, so this reads the recall failure off the row that recorded it.
    #
    # ⚠ THE 2007 CUTOFF IS THE WHOLE SCOPE AND IT IS A REGULATORY FACT, NOT A HEURISTIC. Item
    # 402(k) came in with the 2006 Reg S-K amendments and applies to fiscal years ending on or
    # after 2006-12-15, so the first proxy that can carry a Director Compensation Table is the
    # 2007 season. The archive shows that cliff to the year:
    #
    #     2004  97.2% zero    2006  83.1% zero    2008  19.6% zero
    #     2005  95.4% zero    2007  24.4% zero    2012  13.6% zero
    #
    # Without the cutoff this scope returns 4,399 filings over 435 tickers ($101 at
    # $0.023/filing); 3,395 of those are pre-2007 and have no table to find, so ~$78 of that
    # would buy nothing. With it: 1,097 of 8,808 filings (12.45%) over 272 tickers, $25 -- the
    # addressable population, and the only one where a zero row count is evidence of a defect.
    "no-director-comp": """
        SELECT l.ticker, l.accession_number, l.as_of::date AS as_of
        FROM   def14a_llm l
        WHERE  COALESCE(l.n_director_comp_rows, 0) = 0
          AND  l.as_of >= '2007-01-01'
        ORDER  BY l.ticker, l.as_of
    """,
    # THE DEFECT POPULATION ITSELF, and the right sample to validate the schema change on
    # before paying for the other 968. These are the 59 filings whose stored "insider
    # ownership" is above 0.90 -- 58 of them dual-class, so the number is voting power, and one
    # (LVS 2005-04-29 at 0.91) where Adelson genuinely held ~88% of a single-class company.
    # $1.36 buys a verdict on whether the model now puts each column in its own field, which is
    # the risk the two-column schema introduces: the same confusion MIRRORED.
    "high-ownership": """
        SELECT ticker, accession_number, as_of::date AS as_of
        FROM   def14a_llm
        WHERE  insider_ownership_pct >= 0.90
        ORDER  BY insider_ownership_pct DESC, ticker
    """,
}
SCOPES["both"] = f"""
    SELECT ticker, accession_number, min(as_of) AS as_of FROM (
        ({SCOPES['dual-class'].replace('ORDER  BY ticker, as_of', '')})
        UNION
        ({SCOPES['no-director-comp'].replace('ORDER  BY l.ticker, l.as_of', '')})
    ) u GROUP BY ticker, accession_number ORDER BY ticker, as_of
"""

logger = logging.getLogger("def14a_reextract")


def _work(ctx: Context, args: argparse.Namespace) -> pd.DataFrame:
    """The (ticker, accession_number, as_of) rows this run would re-extract."""
    if args.scope == "accessions":
        accs = [a.strip() for a in (args.accessions or "").split(",") if a.strip()]
        if not accs:
            raise SystemExit("--scope accessions needs --accessions A,B,C")
        quoted = ", ".join(f"'{a}'" for a in accs)
        sql = ("SELECT ticker, accession_number, as_of::date AS as_of FROM def14a_llm "
               f"WHERE accession_number IN ({quoted}) ORDER BY ticker, as_of")
    else:
        sql = SCOPES[args.scope]

    work = pd.read_sql(sql, ctx.store.engine)
    if args.tickers:
        keep = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        work = work[work["ticker"].isin(keep)]
    if args.limit:
        # ONE filing per ticker first, then fill. A 20-filing sample drawn straight off an
        # ORDER BY would be 20 filings of two companies and would say nothing about the other
        # 102 -- the same mistake the carve diagnostic made when it sampled six tickers.
        #
        # ⚠ `groupby().head()`, NOT `groupby().apply(head)`. The apply form DROPS the grouping
        # column from the result on current pandas, so `work['ticker']` raised a bare
        # `KeyError: 'ticker'` from the summary print -- after the scope query had already run.
        work = work.groupby("ticker", as_index=False, sort=False) \
                   .head(max(1, args.per_ticker)) \
                   .head(args.limit)
    return work.reset_index(drop=True)


def _tasks_for_ticker(ctx: Context, ticker: str, cik: str, company: str,
                      want: set[str]) -> list[tuple[str, dict]]:
    """Fetch + carve this ticker's target filings on THIS thread; `(payload, meta)` per readable
    one.

    Returns the ingredients rather than `LlmTask`s because `LlmTask` is `frozen=True` and its
    `seq` has to be the position within the BATCH -- which is not known until the batch closes,
    several tickers later. `run_extraction` returns results in `seq` order, so a duplicated seq
    across tickers would not corrupt the saves (they group on `meta["ticker"]`) but would make
    the failure log unreadable.
    """
    try:
        # The SAME listing call production makes, so `doc_url` / `txt_url` are the production
        # URLs. `years` is the full configured window: a target accession can be 20 years old
        # and a manifest-shaped window would simply not list it.
        filings = list_filings(ctx, cik, DEF14A_FORMS,
                               ctx.config.data_extract.years_history, company)
    except Exception as e:                              # noqa: BLE001 -- one ticker, not the run
        logger.warning("%s: DEF 14A filing list failed (%s)", ticker, e)
        return []

    todo = filings[filings["accession_number"].isin(want)]
    missing = want - set(todo["accession_number"])
    if missing:
        logger.warning("%s: %d target accession(s) not in the EDGAR listing: %s",
                       ticker, len(missing), sorted(missing)[:3])

    out: list[tuple[str, dict]] = []
    for _, f in todo.iterrows():
        payload = _payload_for(ctx, ticker, f)
        if payload:
            out.append((payload, {"ticker": ticker, "filing": f}))
    return out


def _extract(ctx: Context, config, work: pd.DataFrame, workers: int) -> tuple[int, int]:
    """Re-extract `work`, batching ACROSS tickers so the pool stays full.

    ⚠ BATCHED ACROSS TICKERS, NOT ONE TICKER AT A TIME, and the difference is hours. Production
    calls `run_extraction` once per ticker because it persists per ticker, and a DEF 14A is an
    annual filing so a routine run adds one or two per name -- fine there. Here the median
    ticker contributes ~6 target filings, so a per-ticker batch would leave half the 12 workers
    idle on every wave: 1,986 filings at ~2 minutes a call is ~5.5 hours with a full pool and
    ~13 without. `run_extraction` already groups its SAVES by `group_key`, so batching the
    calls costs nothing in crash safety -- each ticker's five frames are still written together,
    as soon as that ticker's last call in the batch returns.
    """
    cik_map = load_cik_mapping(ctx, sorted(work["ticker"].unique()))
    extractor = LLMExtractor(ctx, config, action="def14a", threads=workers)
    ok, failed = 0, 0
    batch: list[tuple[str, dict]] = []
    done_tickers = 0

    def flush(pending: list[tuple[str, dict]]) -> tuple[int, int]:
        """Delete the children of the filings in `pending`, then extract and save them."""
        if not pending:
            return 0, 0
        # ⚠ DELETE THE CHILDREN OF THE FILINGS WE ARE ABOUT TO REWRITE, and only those. Done
        # AFTER the carve so a filing whose HTML cannot be read keeps the rows it has, and
        # BEFORE the save so the new extract replaces rather than merges with the old one.
        accs = [str(meta["filing"]["accession_number"]) for _, meta in pending]
        # ⚠ BACK THE CHILDREN UP BEFORE DELETING THEM. The delete has to precede the save (the
        # save upserts, so a director the new extract spells differently would otherwise leave
        # the old row behind), but a call that FAILS after the delete would take real rows with
        # it -- turning a filing that had 11 director-pay rows into one that has none, which is
        # exactly the defect this run exists to fix. The backup makes that recoverable instead
        # of permanent, and it costs one CSV per batch.
        for table in _CHILD_TABLES.values():
            existing = ctx.store.load(table, where={"accession_number": accs}, optional=True)
            if existing is not None and len(existing):
                BACKUP.mkdir(parents=True, exist_ok=True)
                stamp = f"{table.name}_{accs[0]}_{len(accs)}"
                existing.to_csv(BACKUP / f"{stamp}.csv", index=False)
            n = ctx.store.delete(table, {"accession_number": accs})
            if n:
                logger.info("cleared %d row(s) from %s (backed up)", n, table.name)
        tasks = [LlmTask(seq=i, payload=payload, schema=Def14AExtract,
                         table=Tables.def14a_llm, meta=meta)
                 for i, (payload, meta) in enumerate(pending)]
        results = extractor.run_extraction(tasks, flatten=_result_frames,
                                           group_key=lambda t: str(t.meta["ticker"]))
        n_ok = sum(1 for x in results if x.ok)
        logger.info("batch: %d/%d filing(s) re-extracted", n_ok, len(tasks))
        return n_ok, len(results) - n_ok

    for _, r in cik_map.iterrows():
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        want = set(work.loc[work["ticker"] == ticker, "accession_number"])
        if not want:
            continue
        batch.extend(_tasks_for_ticker(ctx, ticker, cik, company, want))
        done_tickers += 1
        # a full pool plus one wave of slack, so the last worker is never waiting for a carve
        if len(batch) >= workers * 2:
            a, b = flush(batch)
            ok, failed = ok + a, failed + b
            batch = []
            logger.info("progress: %d ticker(s) resolved, %d filing(s) done, %d failed",
                        done_tickers, ok, failed)

    a, b = flush(batch)
    return ok + a, failed + b


def main(argv: list[str] | None = None) -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", choices=[*SCOPES, "accessions"], default="dual-class")
    ap.add_argument("--accessions", default=None, help="comma-separated, with --scope accessions")
    ap.add_argument("--tickers", default=None, help="restrict the scope to these tickers")
    ap.add_argument("--limit", type=int, default=0, help="cap the filing count (a sample)")
    ap.add_argument("--per-ticker", type=int, default=1,
                    help="with --limit, filings per ticker before filling (default 1)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default=None, help="pin a model without touching config")
    ap.add_argument("--write", action="store_true", help="SPEND MONEY and write the rows")
    args = ap.parse_args(argv)

    config, ctx = get_config_context("./configs", use_cache=True, save=False)
    work = _work(ctx, args)

    print(f"scope           {args.scope}")
    print(f"filings         {len(work)}")
    print(f"tickers         {work['ticker'].nunique()}")
    if len(work):
        print(f"as_of span      {work['as_of'].min()} .. {work['as_of'].max()}")
    print(f"estimated cost  ${len(work) * USD_PER_FILING:,.2f} "
          f"at ${USD_PER_FILING}/filing (measured)")
    if not len(work):
        return 0

    if not args.write:
        print("\nDRY RUN -- no LLM call made, nothing written. Re-run with --write to spend.")
        print(work.head(25).to_string(index=False))
        return 0

    config = with_gpt_overrides(config, "def14a", model=args.model)
    ok, failed = _extract(ctx, config, work, args.workers)
    print(f"\nre-extracted    {ok} filing(s)")
    print(f"failed          {failed} filing(s)")
    print(f"actual cost     ~${ok * USD_PER_FILING:,.2f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
