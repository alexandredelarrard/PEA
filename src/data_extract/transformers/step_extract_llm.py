"""LLM step: the three extractions that send filing text to a provider.

Order: DEF 14A sections (lists and reads each ticker's proxies itself), 8-K Item 5.07 votes (reads the stored `sec_8k`
narratives and the DEF 14A tables, so it runs after DEF 14A), employee counts (lists and reads each ticker's annual
filings itself). The DEF 14A section extractors and the votes extractor are built once here and reused by every run;
the init raises `OSError` when a provider these runs use has no API key. The employee extractor is still built per
ticker inside `fundamentals_employees`: its per-ticker workers run concurrently and one extractor is not re-entrant.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals.fundamentals_employees import EMPLOYEES_PROVIDER, fetch_fundamentals_employees
from src.data_extract.utils.structure.def14a import fetch_def14a_llm, section_extractors
from src.data_extract.utils.structure.votes import VOTES_ACTION, fetch_8k_votes_llm

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.utils.step import Step
from src.utils.universe import load_universe_tickers


@dataclass(frozen=True)
class LlmOptions:
    """The LLM commands' options: the ticker subset (None: the universe) and `full` (employees re-decide every date)."""

    tickers: list[str] | None = None
    full: bool = False


class StepExtractLlm(Step):
    def __init__(self, context: Context, config: DictConfig, options: LlmOptions | None = None) -> None:
        super().__init__(context=context, config=config)
        options = options or LlmOptions()
        self._tickers = options.tickers or load_universe_tickers(context)
        self._full = options.full
        self._years_history = int(config.data_extract.years_history)
        self._def14a = section_extractors(context, config)
        self._votes = LLMExtractor(context, config, action=VOTES_ACTION)
        for provider in (self._votes.default_api, EMPLOYEES_PROVIDER):
            self._votes._require_key(provider)

    def run_def14a(self) -> None:
        fetch_def14a_llm(self._context, self._config, self._tickers, self._def14a)

    def run_8k_votes(self) -> None:
        fetch_8k_votes_llm(self._context, self._tickers, self._votes)

    def run_employees(self) -> None:
        fetch_fundamentals_employees(self._context, self._tickers, self._years_history, full=self._full)

    def run(self) -> None:
        self.run_def14a()
        self.run_8k_votes()
        self.run_employees()
