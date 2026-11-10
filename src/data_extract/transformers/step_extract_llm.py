"""LLM step: the three extractions that send filing text to a provider.

Order: DEF 14A sections (reads the stored `sec_def14a_text`), 8-K Item 5.07 votes (reads the stored `sec_8k`
narratives and the DEF 14A tables, so it runs after DEF 14A), employee counts (reads the stored `sec_employees_text`).
The DEF 14A section extractors and the votes extractor are built once here, the employees extractors (their cheaper
provider, one per `data_extract.fundamentals_workers` concurrent ticker, one call at a time each) on the first employees
run; each is reused by every later run. The init raises `OSError` when a provider these runs use has no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals.fundamentals_employees import EMPLOYEES_PROVIDER, fetch_employees_llm
from src.data_extract.utils.structure.def14a.fetch import fetch_def14a_llm, section_extractors
from src.data_extract.utils.structure.votes import VOTES_ACTION, fetch_8k_votes_llm

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_provider
from src.utils.step import Step


@dataclass(frozen=True)
class LlmOptions:
    """The LLM commands' options: the ticker subset (None: the universe) and `full` (employees re-decide every date)."""

    tickers: list[str] | None = None
    full: bool = False


class StepExtractLlm(Step):
    def __init__(self, context: Context, config: DictConfig, options: LlmOptions | None = None) -> None:
        """Build the DEF 14A section, votes and employee extractors once; raise `OSError` when a provider these runs
        use has no API key."""
        super().__init__(context=context, config=config)
        options = options or LlmOptions()
        self._requested = options.tickers
        self._full = options.full
        self._years_history = int(config.data_extract.years_history)
        self._def14a = section_extractors(context, config)
        self._votes = LLMExtractor(context, config, action=VOTES_ACTION)
        for provider in (self._votes.default_api, EMPLOYEES_PROVIDER):
            self._votes._require_key(provider)

    @cached_property
    def _tickers(self) -> list[str]:
        """The `-t` subset, else the universe (read once, on first use)."""
        return self._requested or self._universe.tickers

    @cached_property
    def _employees(self) -> list[LLMExtractor]:
        """One employees extractor per concurrent ticker (built on first use)."""
        config = with_provider(self._config, EMPLOYEES_PROVIDER)
        workers = int(self._config.data_extract.fundamentals_workers)
        return [LLMExtractor(self._context, config, action="employees", threads=1) for _ in range(workers)]

    def run_def14a_llm(self) -> None:
        """Extract the open DEF 14A sections from the stored `sec_def14a_text` payloads."""
        fetch_def14a_llm(self._context, self._config, self._tickers, self._def14a)

    def run_8k_votes(self) -> None:
        """Parse the stored 8-K Item 5.07 narratives into vote rows."""
        fetch_8k_votes_llm(self._context, self._tickers, self._votes)

    def run_employees_llm(self) -> None:
        """Decide the employee counts of the stored `sec_employees_text` filings with no row yet (`full`: all of them)."""
        fetch_employees_llm(self._context, self._tickers, self._years_history, self._employees, full=self._full)

    def run(self) -> None:
        """The DEF 14A pass, then the 8-K votes, then the employees."""
        self.run_def14a_llm()
        self.run_8k_votes()
        self.run_employees_llm()
