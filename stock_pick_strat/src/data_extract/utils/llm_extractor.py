"""
llm_extractor.py  (src/data_extract/utils/llm_extractor.py)
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
    """Structured-output extractor using the OpenAI Responses API."""

    def __init__(self, model: str = "gpt-4o-mini", max_chars: int = 100_000) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_chars = max_chars

    def extract(self, schema: Type[T], text: str) -> T:
        """Extract structured data from text according to the Pydantic schema.

        The text is truncated to max_chars before being sent. For long documents
        (e.g. full DEF 14A filings), pre-slice the relevant sections upstream via
        prepare_def14a_sections() to avoid truncating important tables.
        """
        truncated = text[: self._max_chars]
        response = self._client.responses.parse(
            model=self._model,
            input=truncated,
            instructions=_SYSTEM_PROMPT,
            text_format=schema,
        )
        return response.output_parsed
