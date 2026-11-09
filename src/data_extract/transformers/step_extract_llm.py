"""LLM step: the three extractions that send filing text to a provider.

Order: DEF 14A sections (lists and reads each ticker's proxies itself), 8-K Item 5.07 votes (reads the stored `sec_8k`
narratives and the DEF 14A tables, so it runs after DEF 14A), employee counts (reads the stored `sec_employees_text`).
The DEF 14A section extractors, the votes extractor and the employees extractor (its cheaper provider,
`data_extract.fundamentals_workers` concurrent calls) are built once here and reused by every run; the init raises
`OSError` when a provider these runs use has no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals.fundamentals_employees import EMPLOYEES_PROVIDER, fetch_employees_llm
from src.data_extract.utils.structure.def14a import fetch_def14a_llm, section_extractors
from src.data_extract.utils.structure.votes import VOTES_ACTION, fetch_8k_votes_llm

# `gpt_extract` is a shared service like `src/utils/`, the sanctioned cross-import (model, keys, prompts, thread pool).
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.utils.step import Step


@dataclass(frozen=True)
class LlmOptions:
    """The LLM commands' options: the ticker subset (None: the universe) and `full` (employees re-decide every date)."""

    tickers: list[str] | None = None
    full: bool = False


class StepExtractLlm(Step):
    def __init__(self, context: Context, config: DictConfig, options: LlmOptions | None = None) -> None:
        super().__init__(context=context, config=config)
        options = options or LlmOptions()
        self._requested = options.tickers
        self._full = options.full
        self._years_history = int(config.data_extract.years_history)
        self._def14a = section_extractors(context, config)
        self._votes = LLMExtractor(context, config, action=VOTES_ACTION)
        employees_config = with_gpt_overrides(config, "employees", provider=EMPLOYEES_PROVIDER)
        self._employees = LLMExtractor(context, employees_config, action="employees", threads=int(config.data_extract.fundamentals_workers))
        for provider in (self._votes.default_api, EMPLOYEES_PROVIDER):
            self._votes._require_key(provider)

    @cached_property
    def _tickers(self) -> list[str]:
        """The `-t` subset, else the universe (read once, on first use)."""
        return self._requested or self._universe.tickers

    def run_def14a(self) -> None:
        fetch_def14a_llm(self._context, self._config, self._tickers, self._def14a)

    def run_8k_votes(self) -> None:
        fetch_8k_votes_llm(self._context, self._tickers, self._votes)

    def run_employees_llm(self) -> None:
        fetch_employees_llm(self._context, self._tickers, self._years_history, self._employees, full=self._full)

    def run(self) -> None:
        self.run_def14a()
        self.run_8k_votes()
        self.run_employees_llm()
