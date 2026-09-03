"""
gpt_extract  (src/gpt_extract/)
-------------------------------
"Text in, Pydantic schema filled, rows out." One class owns the model, the keys, the
prompts and the token bill; everything domain-specific stays with its caller.

`gpt_extract` is a SHARED SERVICE, like `src/utils/`: `data_aggregate` and `data_peers`
import its embedding mode directly. That is a deliberate, documented exception to the
no-cross-imports rule -- the alternative was two more copies of an OpenAI client, which is
exactly what this package replaced (one of them still carried a stale 8,000-char cap).

Adding an action = two `.md` files in `prompt_templates/` + a schema class + a table.
Adding a provider = implement `_Provider`, add a key pattern, add an `llm_model` entry.
"""
from src.gpt_extract.utils.embeddings import (
    EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_CHARS, EMBEDDING_MODEL, cosine, embed_texts,
    openai_api_key,
)

__all__ = [
    "EMBEDDING_BATCH_SIZE", "EMBEDDING_MAX_CHARS", "EMBEDDING_MODEL",
    "cosine", "embed_texts", "openai_api_key",
]
