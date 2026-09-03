# `src/gpt_extract` — text in, Pydantic schema filled, rows out

One place owns the model, the API keys, the prompts, the thread pool and the token bill.
Everything domain-specific — which text to send, how to flatten the answer, which table it
lands in — stays with the caller.

```
gpt_extract/
  prompt_templates/           <action>_system_prompt.md + <action>_prompt.md
  transformers/
    step_gpt_extracter.py     GptExtracter(Step)  — config, keys, prompts, extract(), embed()
    gpt_getter.py             LLMExtractor(GptExtracter) — two queues, ordered results
  utils/
    providers.py              _Provider protocol + OpenAIProvider + GeminiProvider stub
    schemas_gpt.py            LlmTask / LlmResult
    usage.py                  UsageTracker — counts, cached share, spend estimate
    embeddings.py             embed_texts / cosine / openai_api_key
    customed_parser.py        RobustJSONParser — fallback for a provider with no
                              structured-output mode; no provider uses it today
```

## Adding an action

Three things, no code in this package:

1. **Two prompt files** in `prompt_templates/`, named for the action:
   `myaction_system_prompt.md` (may contain `{_format}`) and `myaction_prompt.md` (must end
   with `{query}` — see "the payload goes last" below).
2. **A budget** in `configs/gpt.yml` under `gpt.max_chars.myaction`.
3. **A schema class and a table** at the call site:

```python
extractor = LLMExtractor(context, config, action="myaction")
tasks = [LlmTask(seq=i, payload=text, schema=MySchema, table=Tables.my_table,
                 meta={"ticker": t}) for i, (t, text) in enumerate(inputs)]
extractor.run_extraction(tasks, group_key=lambda task: str(task.meta["ticker"]))
```

`flatten` is optional: omit it and each answer becomes one row in `task.table`. Pass one to
fan a single answer out to several tables (the DEF 14A extract writes five).

## Adding a provider

Implement `_Provider` — `parse(schema, system, user) -> (filled, usage)` and `embed` — then
register it in `_PROVIDER_CLASSES`, add its key pattern to `_KEY_PATTERNS`, and add an entry
under `gpt.llm_model` in `configs/gpt.yml`. `GeminiProvider` is the worked stub.

Say in `structured` whether the provider constrains decoding to the schema. OpenAI does, so
an out-of-schema response is unrepresentable rather than merely unlikely; a provider without
that mode should fall back to `RobustJSONParser` and report `structured = False` so the
difference is logged rather than discovered in the data.

## Four rules that are load-bearing

- **The payload goes LAST in the prompt.** `prompt_cache_key` reuses the cached prefix
  across every call for a (model, schema) pair, and cached input is ~10x cheaper. Anything
  after the payload breaks the prefix and the bill goes up accordingly.
- **Reasoning models reject `temperature` and `seed`.** They are listed in
  `gpt.reasoning_models` rather than pattern-matched, so a new model name is a config change
  and not a 400 in the middle of a paid run.
- **Workers never write.** They receive a task and a provider, never a `Context`.
  `store.ensure_table` is a check-then-create with no lock, so concurrent writers on a cold
  table can each see "absent" and silently lose rows. `run_extraction` saves on the main
  thread, once per `group_key`, which is also what makes an interrupted run cost at most one
  group's tokens.
- **Tables are saved in the order `flatten` yields them.** A caller with parent/child tables
  should yield children first, so a crash between the two leaves a recoverable orphan rather
  than a parent claiming children it does not have.

## The sanctioned cross-import

AGENTS.md forbids cross-imports between `src/` subfolders. `gpt_extract` is the documented
exception: it is a shared service like `src/utils/`, and `data_extract`, `data_aggregate`
and `data_peers` all import it. The alternative was three OpenAI clients, which is what this
package replaced — and one of them still carried a stale 8,000-char embedding cap.
