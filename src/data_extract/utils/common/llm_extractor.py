"""
llm_extractor.py  (src/data_extract/utils/common/llm_extractor.py)
------------------------------------------------------------
Generic structured-output extractor backed by the OpenAI Responses API
(openai >= 2.45). Accepts any Pydantic BaseModel as the output schema.

Usage:
    extractor = LLMExtractor(model="gpt-4o-mini")
    result: MySchema = extractor.extract(MySchema, raw_text)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial document parser specializing in SEC proxy statements (DEF 14A). "
    "Extract the requested structured information precisely from the provided text. "
    "Only include values explicitly stated in the document. "
    "Use null for any field not found or not clearly stated in the text."
)


class LLMExtractor:
    """Structured-output extractor using the OpenAI Responses API.

    `cache=True` sends a stable `prompt_cache_key` so OpenAI can reuse the cached prompt prefix
    (the shared system instructions + schema) across calls, lowering latency and cost.

    Extraction is NOT deterministic. `temperature` used to be accepted here and was never sent
    -- `extract` builds a kwargs dict of `model` / `input` / `instructions` / `text_format` /
    `prompt_cache_key` only -- so the old "temperature=0 makes it deterministic" claim was
    false, and `gpt-5-mini` (a reasoning model) does not accept the parameter at all. Measured:
    the same filing and the same carve returned `n_neos` = 1 on one run and 6 on another. Plan
    for that variance where it matters (a marginal anchor is the amplifier, so the fix is a
    tighter carve, not a sampling knob).
    """

    def __init__(self, model: str = "gpt-4o-mini", max_chars: int = 100_000,
                 cache: bool = True) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_chars = max_chars
        self._cache = cache
        #: Token usage of the LAST call, and the running totals for this extractor. The API
        #: reports them per response and nothing else in the repo captured them, so a backfill's
        #: real bill could only be estimated after the fact. `cached_input_tokens` is what makes
        #: `prompt_cache_key` measurable rather than assumed.
        self.last_usage: dict[str, int] | None = None
        self.totals: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                       "cached_input_tokens": 0}
        # `totals` is a read-modify-write from every caller, and one extractor is deliberately
        # SHARED across a thread pool so `prompt_cache_key` keeps hitting the same cached
        # prefix. Without the lock the call count silently under-reports under concurrency.
        self._usage_lock = threading.Lock()

    def extract(self, schema: Type[T], text: str, instructions: str | None = None) -> T:
        """Extract structured data from text according to the Pydantic schema.

        The text is truncated to max_chars before being sent. For long documents
        (e.g. full DEF 14A filings), pre-slice the relevant sections upstream via
        prepare_def14a_sections() to avoid truncating important tables. Pass
        `instructions` to override the generic system prompt with a task-tailored one
        (more accurate, and cached per (model, schema) so it stays cheap).
        """
        truncated = text[: self._max_chars]
        kwargs: dict = {
            "model": self._model,
            "input": truncated,
            "instructions": instructions or _SYSTEM_PROMPT,
            "text_format": schema
        }
        if self._cache:
            # stable key -> OpenAI reuses the cached prompt prefix for this
            # (model, schema) combination across every filing
            kwargs["prompt_cache_key"] = f"{self._model}:{schema.__name__}"
        response = self._client.responses.parse(**kwargs)
        self._record_usage(getattr(response, "usage", None))
        return response.output_parsed

    def _record_usage(self, usage: object) -> None:
        """Capture the response's token counts into `last_usage` / `totals`.

        Defensive by design: the usage object is an SDK model, so a field that moves or goes
        absent must not take down an extraction whose tokens are already paid for.
        """
        if usage is None:
            self.last_usage = None
            return
        details = getattr(usage, "input_tokens_details", None)
        self.last_usage = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        }
        with self._usage_lock:
            self.totals["calls"] += 1
            for k, v in self.last_usage.items():
                self.totals[k] += v
