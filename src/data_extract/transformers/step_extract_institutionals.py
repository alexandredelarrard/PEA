"""
step_extract_institutionals.py (src/data_extract/transformers/step_extract_institutionals.py)
----------------------------------------------------------------------------------------------
WHO owns, trades and bets against each name -- every source that answers that question, in one
step, mirroring the cube's `institutionals` part on the aggregation side:

  * 13F institutional holdings (+ the OpenFIGI CUSIP->ticker map they are keyed through)
  * the Dataroma superinvestor roster -- the elite-manager subset of those same 13F filers
  * those managers' COMPLETE books (every security, not just the S&P 500 slice `sec13f_hr`
    keeps), which is the only honest denominator for a portfolio weight
  * insider transactions (Forms 3/4/5)
  * SC 13D activist stakes and SC 13G passive 5%+ stakes -- the two halves of the
    beneficial-ownership disclosure channel, and what an escalation between them means
  * 8-K corporate events
  * short interest (FINRA RegSHO) and SEC fails-to-deliver -- the SHORT side of the same
    question, and the other half of what `cube_part_institutionals` consumes

The window is resolved once here and passed INTO every fetcher rather than read from config
inside them, matching the other extract steps.

ORDER MATTERS in two places:
  * the CUSIP map is built inside `fetch_13f`, which is why no separate call appears below;
  * `fetch_8k_edgar` must run BEFORE `StepExtractStructure`, whose `fetch_8k_votes_llm` reads
    the `sec_8k` Item 5.07 narratives this step stores. `StepExtractAllData` orders the two
    steps accordingly -- running structure first leaves the vote parser reading yesterday's
    8-Ks.
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_superinvestors import upsert_roster_snapshot
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_13f_managers
from src.data_extract.utils.institutionals.fetch_insider_transactions import (
    fetch_insider_transactions)
from src.data_extract.utils.institutionals.fetch_13d_edgar import fetch_13d_edgar
from src.data_extract.utils.institutionals.fetch_13g_edgar import fetch_13g_edgar
from src.data_extract.utils.institutionals.fetch_8k_edgar import fetch_8k_edgar
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import fetch_fails_to_deliver


class StepExtractInstitutionals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run(self, tickers: list[str]) -> None:

        years_history = int(self.config.data_extract.years_history)

        # 13F institutional holdings (edgartools by filing date + OpenFIGI cusip map). Resumes
        # from max(filing_date) in sec13f_hr, so a routine run reads only the new filings.
        fetch_13f(self._context, tickers=tickers, years_history=years_history)

        # Superinvestors roster: curated top managers (Dataroma) -> one dated snapshot row per
        # manager in `superinvestor_roster`, so membership stays point-in-time. AFTER the 13F
        # pull: resolution settles an ambiguous manager name on which candidate CIK actually
        # has rows in `sec13f_hr`.
        upsert_roster_snapshot(self._context)

        # The roster managers' COMPLETE books, at CUSIP grain and unfiltered. AFTER the snapshot
        # above: its scope is the union of every CIK the roster has ever carried, so a manager
        # added today must already be in the table. Takes the window but NOT `tickers` -- having
        # no universe filter is the entire point of it.
        fetch_13f_managers(self._context, years_history=years_history)

        # insider transactions
        fetch_insider_transactions(self._context, tickers=tickers, years_history=years_history)

        # activist stakes (SC 13D) then the passive ones (SC 13G), 13D first because it is
        # ~7x cheaper and a failure there is the cheaper one to discover. Same grain and column
        # names, so the escalation join across them is a plain union.
        fetch_13d_edgar(self._context, tickers=tickers, years_history=years_history)
        fetch_13g_edgar(self._context, tickers=tickers, years_history=years_history)

        # corporate events (8-K)
        fetch_8k_edgar(self._context, tickers=tickers, years_history=years_history)

        # the short side: FINRA RegSHO short volume, then SEC settlement fails (mid-2009 on)
        fetch_short_interest(self._context, tickers=tickers, years_history=years_history)
        fetch_fails_to_deliver(self._context, tickers=tickers, years_history=years_history)
