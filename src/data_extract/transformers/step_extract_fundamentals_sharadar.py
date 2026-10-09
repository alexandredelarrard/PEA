"""
step_extract_fundamentals_sharadar.py
  (src/data_extract/transformers/step_extract_fundamentals_sharadar.py)
-----------------------------------------------------------------------
Sharadar fundamentals extraction -> the four vendor-shaped tables.

A SIBLING of `StepExtractFundamentals` (the SEC layer) rather than a part of it, and it runs
BEFORE it: the two producers are independent, and the merged `fundamentals_history` that
consumes both is field-block precedence -- Sharadar owns a declared set of columns for all
history, SEC owns the rest, and no column ever switches source mid-series (D11/D14).

The four fetchers run in DEPENDENCY order, which is not negotiable: the fundamentals fetch
reads `currency` out of `sharadar_tickers` to enforce the USD assertion (D20) and raises if
that table is empty.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import (
    fetch_sharadar_actions,
    fetch_sharadar_fundamentals,
    fetch_sharadar_sp500,
    fetch_sharadar_tickers,
    predecessor_vendor_tickers,
)
from src.data_extract.utils.fundamentals_sharadar.merge_history import build_merged_history
from src.utils.step import Step


class StepExtractFundamentalsSharadar(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str], *, full: bool = False, config_dir: str | None = None, as_of: pd.Timestamp | None = None) -> None:
        """The five stages, in dependency order. `full` re-pulls the whole configured window
        instead of resuming, and makes the merge DELETE before it rebuilds.

        The CLI's `fundamentals-sharadar` command calls this rather than restating the order.
        `config_dir` defaults to `self._config_dir` (the CLI's `-c`, resolved once by `Context`),
        so `main.py`'s pipeline path, which passes no `config_dir`, still honours `-c`.
        """
        config_dir = str(config_dir or self._config_dir)
        years = int(self._config.data_extract.sharadar_years_history)

        # 1. The entity dimension FIRST -- `permaticker`, `currency`, `category`. A full
        #    refresh: `isdelisted` / `lastquarter` mutate, so an append-only view goes stale.
        fetch_sharadar_tickers(self._context)

        # 2. SF1, every column as delivered, on the three AS-REPORTED dimensions. Its own
        #    history knob (`sharadar_years_history`), separate from `years_history`, because
        #    the two sources are limited by different things -- the SEC walk by patience,
        #    Sharadar by subscription tier. A ticker outside the subscription returns 403,
        #    costs one request and is counted, never retried.
        #    The vendor tickers carrying a register predecessor CIK's own series (BHI, STE1, ...) are fetched
        #    too and stored under their own ticker; the merge reads them inside the predecessor's window.
        #    They are delisted, so they are read over the whole window: a rowless key resumes from a recent date.
        fetch_sharadar_fundamentals(self._context, tickers=tickers, years_history=years, full=full, as_of=as_of)
        predecessors = [t for t in predecessor_vendor_tickers(self._context, tickers) if t not in set(tickers)]
        if predecessors:
            self._log.info("Sharadar SF1: + %d predecessor vendor ticker(s): %s", len(predecessors), ", ".join(predecessors))
            fetch_sharadar_fundamentals(self._context, tickers=predecessors, years_history=years, full=True, as_of=as_of)

        # 3. Corporate actions: dividends, splits, spinoffs, acquisitions, relations.
        fetch_sharadar_actions(self._context, years_history=years, full=full, as_of=as_of)

        # 4. S&P 500 membership events. Ingested only -- `src/utils/universe.py` resolves
        #    the universe from `sp500_tickers`; nothing consumes this table yet.
        fetch_sharadar_sp500(self._context, full=full)

        # 5. The MERGED `fundamentals_history` -- Sharadar's declared column block plus the
        #    SEC-owned ones, joined backward as of each publication date.
        #
        #    LAST, and never on its own schedule, for the same reason the SEC step already
        #    documents: a snapshot is only as fresh as the rows it reads. Running this beside
        #    the fetchers rather than after them would publish a table that silently lags its
        #    own inputs by one run, and every row it wrote would look perfectly normal.
        #
        #    It reads `fundamentals_history_sec` too, which THIS step does not produce -- so
        #    the SEC-owned block is as fresh as the last `StepExtractFundamentals` run, not as
        #    this one. That is the stated coverage/freshness asymmetry, not a bug.
        build_merged_history(self._context, tickers=tickers, full=full, config_dir=config_dir)
