"""
providers.py  (src/gpt_extract/utils/providers.py)
--------------------------------------------------
One LLM endpoint behind one method. `parse` is the whole contract: a Pydantic schema goes
in, a filled instance comes out, with the call's token counts alongside it.

Structured output is the PROVIDER's job, not the caller's. OpenAI compiles the schema into
a constrained decoder, so a response that violates the schema is unrepresentable rather
than merely unlikely. A provider with no such mode is expected to fall back to
`RobustJSONParser` and to say so in `structured`, so a caller can log the difference
instead of discovering it in the data.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def usage_dict(usage: object) -> dict[str, int]:
    """An SDK usage object -> plain counts.

    Defensive by design: the usage object is an SDK model, so a field that moves or goes
    absent must not take down an extraction whose tokens are already paid for.
    """
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
    }


class _Provider(Protocol):
    """One LLM endpoint bound to one API key."""

    name: str
    model: str
    structured: bool

    def parse(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, int]]: ...

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]: ...


class OpenAIProvider:
    """The OpenAI Responses API, one client per API key.

    Extraction is NOT deterministic. Measured: the same filing and the same carve returned
    `n_neos` = 1 on one run and 6 on another. Plan for that variance where it matters -- a
    marginal anchor is the amplifier, so the fix is a tighter carve, not a sampling knob.
    """

    name = "open_ai"
    structured = True

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_token: int | None = None,
        cache: bool = True,
        reasoning: bool = False,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_token = max_token
        self.cache = cache
        #: Reasoning models REJECT `temperature` and `seed` with a 400. Resolved from the
        #: configured list by the caller rather than pattern-matched on the model name, so
        #: a new model is a config change and not a 400 in the middle of a paid run.
        self.reasoning = reasoning
        self._client = client if client is not None else OpenAI(
            api_key=api_key, **({"base_url": base_url} if base_url else {})
        )

    def request_kwargs(self, schema: type[T], system: str, user: str) -> dict[str, Any]:
        """Exactly what `parse` will send. Split out so the parameter rules are assertable
        without making a call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": user,
            "instructions": system,
            "text_format": schema,
        }
        if self.cache:
            # A stable key -> OpenAI reuses the cached prompt prefix for this
            # (model, schema) pair across every filing. That is the ~10x input-token
            # discount, and it only holds while the payload stays LAST in the prompt.
            kwargs["prompt_cache_key"] = f"{self.model}:{schema.__name__}"
        if self.max_token:
            kwargs["max_output_tokens"] = self.max_token
        if not self.reasoning:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.seed is not None:
                kwargs["seed"] = self.seed
        return kwargs

    def parse(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, int]]:
        response = self._client.responses.parse(**self.request_kwargs(schema, system, user))
        return response.output_parsed, usage_dict(getattr(response, "usage", None))

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        response = self._client.embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in response.data]


class GeminiProvider:
    """Placeholder for Google Gemini.

    Would use Gemini's native `response_schema` controlled generation, which gives the same
    decode-time guarantee as OpenAI strict mode. Deliberately unimplemented: adding it means
    a `google-genai` dependency, and the seam exists so that is a new file rather than a
    rewrite.
    """

    name = "google"
    structured = True

    def __init__(self, model: str, api_key: str | None = None, **_: Any) -> None:
        self.model = model
        self._api_key = api_key

    def parse(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, int]]:
        raise NotImplementedError("GeminiProvider needs the google-genai dependency")

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        raise NotImplementedError("GeminiProvider needs the google-genai dependency")
