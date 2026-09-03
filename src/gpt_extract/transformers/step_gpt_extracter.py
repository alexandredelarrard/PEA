"""
step_gpt_extracter.py  (src/gpt_extract/transformers/step_gpt_extracter.py)
---------------------------------------------------------------------------
`GptExtracter`: text in, Pydantic schema filled, token bill recorded.

Everything domain-specific -- which text to send, how to flatten the answer, which table it
lands in -- belongs to the caller. This class owns only the parts every LLM extraction
shares: configuration, API keys, provider clients, the `.md` prompt templates and the usage
accounting. A new extraction action is two `.md` files and a schema class, not a new client.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from src.context import Context
from src.gpt_extract.utils.providers import GeminiProvider, OpenAIProvider, _Provider
from src.gpt_extract.utils.usage import UsageTracker
from src.utils.step import Step

T = TypeVar("T", bound=BaseModel)

#: Environment-variable fragments that identify a provider's key. A FRAGMENT, not an exact
#: name, so several keys for one provider (`OPENAI_API_KEY_2`) are all found and rotated.
_KEY_PATTERNS: dict[str, tuple[str, ...]] = {
    "open_ai": ("OPENAI_API_KEY", "OPEN_AI_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
}

#: Providers that need no API key at all. `local` is an LM Studio server on this machine,
#: so demanding a key for it is how `initialize_client("local")` used to be an IndexError.
_KEYLESS: frozenset[str] = frozenset({"local"})

_LOCAL_BASE_URL = "http://localhost:1234/v1"

_PROVIDER_CLASSES: dict[str, type] = {
    "open_ai": OpenAIProvider,
    "local": OpenAIProvider,
    "google": GeminiProvider,
}


class GptExtracter(Step):
    """Fill a Pydantic schema from text, through whichever provider is configured."""

    def __init__(self, context: Context, config: DictConfig, action: str | None = None):
        super().__init__(context=context, config=config)

        gpt = self._config.gpt
        self.default_api: str = gpt.default_api
        self.llm_model: Mapping[str, str] = gpt.llm_model
        self.temperature = gpt.get("temperature")
        self.seed = gpt.get("seed")
        self.max_token = gpt.get("max_token")
        self.threads = int(gpt.get("threads") or 1)
        self.cache = bool(gpt.get("cache", True))
        self.reasoning_models = set(OmegaConf.to_container(gpt.get("reasoning_models"))
                                    if gpt.get("reasoning_models") is not None else [])
        self.max_chars: Mapping[str, int] = gpt.get("max_chars") or {}
        self.embedding: Mapping[str, Any] = gpt.get("embedding") or {}

        self.action = action
        self.prompt_path = Path(__file__).resolve().parents[1] / "prompt_templates"
        self.system_prompt: str = ""
        self.user_prompt: str = ""
        if action:
            self.read_prompts(action)

        self.api_keys: dict[str, list[str]] = self.get_api_keys()
        self.api_keys_index: dict[str, int] = {k: 0 for k in self.llm_model}
        self._key_lock = Lock()

        #: One tracker for the whole run, shared by every worker (see UsageTracker).
        self.usage = UsageTracker()

    # ------------------------------------------------------------------ prompts --- #

    def read_prompt_file(self, path: Path) -> str:
        """The file's text.

        `encoding="utf-8"` is not optional: the default is cp1252 on Windows and the
        prompt files contain em-dashes and `>=`, which is the same mojibake class the
        DEF 14A path already had to fix once.
        """
        if os.path.isdir(path):
            raise FileNotFoundError(f"Provided path {path} is a directory, not a file")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing prompt file {path}")
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()

    def read_prompts(self, action: str) -> tuple[str, str]:
        """`{action}_system_prompt.md` + `{action}_prompt.md`."""
        self.action = action
        self.system_prompt = self.read_prompt_file(
            self.prompt_path / f"{action}_system_prompt.md")
        self.user_prompt = self.read_prompt_file(self.prompt_path / f"{action}_prompt.md")
        return self.system_prompt, self.user_prompt

    def build_prompt(self, payload: str, schema: type[BaseModel] | None = None) -> tuple[str, str]:
        """(system, user) with the templates filled.

        `str.replace`, never `str.format`: the payload is filing text and a stray brace in
        it -- or in a template -- must not raise in the middle of a paid run.

        `{_format}` resolves to a LINE NAMING the schema, not a JSON Schema dump. The
        provider compiles the schema into a constrained decoder, so restating it in the
        prompt buys nothing and would add ~4,200 tokens per call on `Def14AExtract`.
        """
        schema_note = (f"the `{schema.__name__}` schema, which the API enforces on the "
                       "response" if schema is not None else "the requested schema")
        system = self.system_prompt.replace("{_format}", schema_note)
        user = self.user_prompt.replace("{query}", payload)
        return system, user

    def truncate(self, payload: str, action: str | None = None) -> str:
        """Cut the payload to this action's `config.gpt.max_chars` budget."""
        limit = self.max_chars.get(action or self.action or "")
        return payload[:int(limit)] if limit else payload

    # -------------------------------------------------------------------- keys --- #

    def get_api_keys(self) -> dict[str, list[str]]:
        """Every API key in the environment, grouped by provider and de-duplicated.

        Values are de-duplicated because `OPENAI_API_KEY` and `OPEN_AI_API_KEY` are commonly
        aliases of one secret, and rotating between two copies of the same key is not
        rotation.
        """
        keys: dict[str, list[str]] = {provider: [] for provider in self.llm_model}
        for name, value in os.environ.items():
            if not value:
                continue
            for provider, patterns in _KEY_PATTERNS.items():
                if provider not in keys:
                    continue
                if any(p in name for p in patterns) and value not in keys[provider]:
                    keys[provider].append(value)
        return keys

    def available_methodes(self) -> list[str]:
        """The providers that actually have a usable key, so a run can degrade to one
        provider instead of raising."""
        return [m for m in self.llm_model
                if m in _KEYLESS or self.api_keys.get(m)]

    def _require_key(self, methode: str) -> None:
        """Raise unless THIS provider has a key.

        The check this replaces tested `len(self.api_keys) == 0` on a dict pre-seeded with
        one entry per provider, so it could never be true and a totally missing key raised
        nothing.
        """
        if methode not in _KEYLESS and not self.api_keys.get(methode):
            raise EnvironmentError(
                f"No API key for provider '{methode}'. Add one to the .env file "
                f"(any variable whose name contains {' or '.join(_KEY_PATTERNS.get(methode, ()))})."
            )

    def next_key_index(self, methode: str) -> int:
        """The next key to use for this provider, advancing the round-robin."""
        keys = self.api_keys.get(methode) or []
        if not keys:
            return 0
        with self._key_lock:
            index = self.api_keys_index.get(methode, 0) % len(keys)
            self.api_keys_index[methode] = (index + 1) % len(keys)
        return index

    def initialize_client(self, methode: str | None = None,
                          key_index: int | None = None) -> _Provider:
        """One provider bound to one API key.

        `key_index` is USED, not merely computed: the version this replaces advanced a
        rotation index and then always read `api_keys[methode][0]`, so multi-key rotation
        never happened.
        """
        methode = methode or self.default_api
        if methode not in _PROVIDER_CLASSES:
            raise ValueError(
                f"No adapter for provider '{methode}'. Implement `_Provider` and register "
                f"it; known providers are {sorted(_PROVIDER_CLASSES)}."
            )
        self._require_key(methode)

        keys = self.api_keys.get(methode) or []
        if key_index is None:
            key_index = self.next_key_index(methode)
        api_key = keys[key_index % len(keys)] if keys else None

        model = self.llm_model[methode]
        provider = _PROVIDER_CLASSES[methode](
            model=model,
            api_key=api_key,
            temperature=self.temperature,
            seed=self.seed,
            max_token=self.max_token,
            cache=self.cache,
            reasoning=model in self.reasoning_models,
            base_url=_LOCAL_BASE_URL if methode == "local" else None,
        )
        self._log.info("initialized client=%s model=%s key=%d/%d",
                       methode, model, key_index + 1, max(len(keys), 1))
        return provider

    # ----------------------------------------------------------------- extract --- #

    def extract(self, schema: type[T], payload: str, provider: _Provider | None = None,
                action: str | None = None) -> T:
        """Fill `schema` from `payload`. The single-shot path -- no queues, no threads."""
        provider = provider or self.initialize_client()
        system, user = self.build_prompt(self.truncate(payload, action), schema)
        parsed, usage = provider.parse(schema, system, user)
        self.usage.record(usage)
        return parsed
