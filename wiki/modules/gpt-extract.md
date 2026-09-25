---
title: GPT extraction
description: Reusable structured-output, prompt, provider, concurrency, usage, and embedding services.
type: module
tags:
  - wiki
  - module
---
# GPT extraction

## Summary

`src/gpt_extract` is the shared service for text-to-schema extraction and embeddings. It owns provider clients, prompt assembly, API-key rotation, concurrency, ordering, usage accounting, and the main-thread save discipline; domain packages still own input selection, result flattening, and destination tables. Its extension contract is documented in [src/gpt_extract/README.md](../../src/gpt_extract/README.md).

## Responsibilities

- Load paired system/user prompt templates by action name.
- Constrain supported providers to Pydantic schemas where available.
- Run parallel calls while returning results in submission order.
- Save grouped results on the main thread so workers never race table creation.
- Track tokens, cached-input share, and estimated cost.
- Provide shared embedding and cosine helpers.

## Public API / entry points

- `GptExtracter` in [step_gpt_extracter.py](../../src/gpt_extract/transformers/step_gpt_extracter.py).
- `LLMExtractor.run_extraction()` in [gpt_getter.py](../../src/gpt_extract/transformers/gpt_getter.py).
- `LlmTask` and `LlmResult` in [schemas_gpt.py](../../src/gpt_extract/utils/schemas_gpt.py).
- `OpenAIProvider` and the provider protocol in [providers.py](../../src/gpt_extract/utils/providers.py).

## Key files

- [prompt_templates](../../src/gpt_extract/prompt_templates/) contains action-specific prompt pairs for DEF 14A and Item 5.07 extraction.
- [usage.py](../../src/gpt_extract/utils/usage.py) accounts for request usage.
- [embeddings.py](../../src/gpt_extract/utils/embeddings.py) wraps embedding batches.
- [customed_parser.py](../../src/gpt_extract/utils/customed_parser.py) is the fallback parser for providers without structured output.

## Dependencies

The module uses OpenAI, Pydantic, pandas, NumPy, and [Context](./runtime-and-shared-utils.md). [Data extraction](./data-extract.md), [peer deduction](./data-peers.md), and text feature code consume it as a sanctioned cross-package shared service.

## Participates in

The primary end-to-end path is [SEC and LLM extraction](../flows/sec-llm-extraction.md). Peer descriptions also use the embedding service before [peer-relative features](../concepts/peer-relative-features.md) are built.

## Related

- [GPT configuration](./constants-and-configuration.md)
- [Add a data source](../guides/add-a-data-source.md)
- [Data sources](../reference/data-sources.md)
