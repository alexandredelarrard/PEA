"""READ-ONLY, NO LLM: what the paid DEF 14A leg would cost after the registrant register.

Sizing a paid extraction on filings LISTED (not rows stored) is the only correct basis: the
bill is one call per filing sent, whatever comes back. Every proxy this prints is a filing the
register newly exposes and `def14a_llm` does not yet hold.

⚠ This performs an EDGAR walk (one submissions listing per segment CIK). The rate limiter is
per-PROCESS, so it must NOT run beside another walk.

    rtk "$PY" scripts/def14a_registrant_price.py
"""
from __future__ import annotations

import pandas as pd

from src.context import get_config_context
from src.constants.constants import DEF14A_FORMS
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.registrant import load_registrants
from src.data_store.schema import Tables

#: Measured 2026-09-05 over a full run: ~$0.023 a filing, of which output tokens are ~87%.
USD_PER_FILING = 0.023


def main() -> None:
    config, context = get_config_context("./configs", use_cache=False, save=False)
    context.ensure_edgar_identity()
    years = int(config.data_extract.years_history)
    regs = load_registrants()

    stored = frozenset(str(a) for a in
                       context.store.distinct(Tables.def14a_llm, "accession_number"))
    print(f"`def14a_llm` already holds {len(stored):,} accessions\n")
    print("| ticker | segment cik | segment window | proxies listed | NOT yet stored |")
    print("|---|---|---|---:|---:|")

    grand_new = 0
    per_ticker: dict[str, int] = {}
    for ticker, entry in sorted(regs.items()):
        for segment in entry.segments:
            part = list_filings(context, segment.cik, list(DEF14A_FORMS), years, ticker)
            if part is None or part.empty:
                listed = new = 0
            else:
                filed = pd.to_datetime(part["filing_date"])
                part = part[filed.map(segment.covers)]        # the DATED SPLIT, not a union
                listed = len(part)
                new = int((~part["accession_number"].astype(str).isin(stored)).sum())
            grand_new += new
            per_ticker[ticker] = per_ticker.get(ticker, 0) + new
            lo = segment.valid_from.date() if segment.valid_from is not None else "start"
            hi = segment.valid_to.date() if segment.valid_to is not None else "now"
            mark = f"**{new}**" if new else "0"
            print(f"| {ticker} | {segment.cik} | {lo} .. {hi} | {listed} | {mark} |")

    print(f"\n**{grand_new} filing(s) the register newly exposes** "
          f"-> ~${grand_new * USD_PER_FILING:,.2f} at ${USD_PER_FILING}/filing")
    movers = {t: n for t, n in per_ticker.items() if n}
    print(f"tickers that would gain rows: {movers or 'NONE'}")


if __name__ == "__main__":
    main()
