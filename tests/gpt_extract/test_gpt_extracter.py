"""
test_gpt_extracter.py  (tests/gpt_extract/test_gpt_extracter.py)
------------------------------------------------------------------
The `GptExtracter` core: the parameter rules a reasoning model imposes, the cached-prefix
contract, utf-8 prompts, key rotation and lock-safe usage accounting.

Every test is synthetic and offline. The parameter tests assert on the kwargs the provider
WOULD send (`request_kwargs`), which is the only way to pin a recorded 400 without paying
for one.
"""
from __future__ import annotations

import threading

import pytest
from omegaconf import OmegaConf

from src.gpt_extract.transformers.step_gpt_extracter import GptExtracter
from src.gpt_extract.utils.providers import OpenAIProvider
from src.gpt_extract.utils.usage import UsageTracker
from tests.gpt_extract.fakes import Answer, StubProvider, fake_context, gpt_config


def _extracter(monkeypatch, keys=("k1",), **overrides) -> GptExtracter:
    for name in [n for n in dict(__import__("os").environ) if "API_KEY" in n]:
        monkeypatch.delenv(name, raising=False)
    for i, key in enumerate(keys):
        monkeypatch.setenv(f"OPENAI_API_KEY{'' if i == 0 else f'_{i}'}", key)
    return GptExtracter(fake_context(), gpt_config(**overrides))


# --------------------------------------------------------------------------- #
# The parameters a reasoning model rejects                                      #
# --------------------------------------------------------------------------- #
def test_a_reasoning_model_is_never_sent_temperature_or_seed():
    """gpt-5-mini is a reasoning model and 400s on `temperature`. Recorded, not guessed."""
    reasoning = OpenAIProvider("gpt-5-mini", client=object(), temperature=0.2, seed=1234,
                               reasoning=True)
    plain = OpenAIProvider("gpt-4o-mini", client=object(), temperature=0.2, seed=1234,
                           reasoning=False)

    r_kwargs = reasoning.request_kwargs(Answer, "sys", "user")
    p_kwargs = plain.request_kwargs(Answer, "sys", "user")

    assert "temperature" not in r_kwargs and "seed" not in r_kwargs
    assert p_kwargs["temperature"] == 0.2 and p_kwargs["seed"] == 1234

    print("\n=== SANITY: reasoning-model parameter guard ===")
    print(f"  gpt-5-mini  sends {sorted(r_kwargs)}")
    print(f"  gpt-4o-mini sends {sorted(p_kwargs)}")
    print("  temperature/seed suppressed for the reasoning model. Validated.")


def test_max_token_null_means_no_cap_is_sent():
    """`max_token: null` must send NO cap: 4096 truncates a large proxy."""
    uncapped = OpenAIProvider("gpt-5-mini", client=object(), max_token=None)
    capped = OpenAIProvider("gpt-5-mini", client=object(), max_token=4096)

    assert "max_output_tokens" not in uncapped.request_kwargs(Answer, "s", "u")
    assert capped.request_kwargs(Answer, "s", "u")["max_output_tokens"] == 4096

    print("\n=== SANITY: output cap ===")
    print("  max_token=None -> no max_output_tokens; 4096 -> sent. Validated.")


def test_prompt_cache_key_is_stable_per_model_and_schema():
    """The ~10x input-token discount the budget assumes."""
    provider = OpenAIProvider("gpt-5-mini", client=object(), cache=True)
    first = provider.request_kwargs(Answer, "sys", "payload one")
    second = provider.request_kwargs(Answer, "sys", "payload two")

    assert first["prompt_cache_key"] == second["prompt_cache_key"] == "gpt-5-mini:Answer"
    assert "prompt_cache_key" not in OpenAIProvider(
        "gpt-5-mini", client=object(), cache=False).request_kwargs(Answer, "s", "u")

    print("\n=== SANITY: prompt_cache_key ===")
    print(f"  two different payloads -> one key {first['prompt_cache_key']!r}; "
          "cache=False sends none. Validated.")


# --------------------------------------------------------------------------- #
# Prompt building                                                              #
# --------------------------------------------------------------------------- #
def test_the_payload_is_the_last_thing_in_the_prompt(monkeypatch):
    """The cached prefix is only stable while the payload stays LAST."""
    ext = _extracter(monkeypatch)
    ext.read_prompts("def14a")
    _, user = ext.build_prompt("THE-PAYLOAD-MARKER", Answer)

    assert user.rstrip().endswith("THE-PAYLOAD-MARKER")

    print("\n=== SANITY: payload position ===")
    print(f"  user prompt ends with the payload; tail = {user.rstrip()[-30:]!r}. Validated.")


def test_a_brace_in_the_payload_does_not_raise(monkeypatch):
    """Filing text contains braces; `str.format` would raise on them mid-run."""
    ext = _extracter(monkeypatch)
    ext.read_prompts("def14a")
    payload = 'a {brace} and a {"json": "fragment"} and a lone {'
    _, user = ext.build_prompt(payload, Answer)

    assert payload in user

    print("\n=== SANITY: braces survive ===")
    print(f"  {payload!r} passed through build_prompt untouched (str.replace). Validated.")


def test_the_format_placeholder_names_the_schema_not_its_json(monkeypatch):
    """Structured output means the schema is enforced at decode time, so restating a
    16,861-char JSON Schema in the prompt would buy nothing and cost ~4,215 tokens/call."""
    ext = _extracter(monkeypatch)
    ext.read_prompts("def14a")
    system, _ = ext.build_prompt("payload", Answer)

    assert "{_format}" not in system
    assert "Answer" in system
    assert len(system) < 2_000

    print("\n=== SANITY: {_format} resolution ===")
    print(f"  system prompt is {len(system)} chars and names the schema, not its JSON. Validated.")


def test_prompt_files_are_read_as_utf8(monkeypatch, tmp_path):
    """cp1252 is the Windows default and the prompts carry em-dashes and >=."""
    ext = _extracter(monkeypatch)
    fixture = tmp_path / "utf8_prompt.md"
    text = "an em-dash — and a ≥ sign"
    fixture.write_text(text, encoding="utf-8")

    assert ext.read_prompt_file(fixture) == text

    print("\n=== SANITY: prompt encoding ===")
    print(f"  round-tripped {text!r} through read_prompt_file. Validated.")


def test_truncate_uses_the_action_budget(monkeypatch):
    ext = _extracter(monkeypatch)
    long_payload = "x" * 200_000

    assert len(ext.truncate(long_payload, "def14a")) == 130_000
    assert len(ext.truncate(long_payload, "sec8k_votes")) == 40_000

    print("\n=== SANITY: per-action truncation ===")
    print("  def14a 130,000 chars / sec8k_votes 40,000 chars, from config. Validated.")


# --------------------------------------------------------------------------- #
# API keys                                                                     #
# --------------------------------------------------------------------------- #
def test_api_keys_rotate_across_clients(monkeypatch):
    """3 keys, 6 clients, each key twice. The version this replaces advanced an index and
    then always read `api_keys[methode][0]`."""
    ext = _extracter(monkeypatch, keys=("k1", "k2", "k3"))
    assert ext.api_keys["open_ai"] == ["k1", "k2", "k3"]

    used = [ext.next_key_index("open_ai") for _ in range(6)]

    assert used == [0, 1, 2, 0, 1, 2]
    print("\n=== SANITY: API-key round-robin ===")
    print(f"  3 keys over 6 clients -> indices {used}; each key used twice. Validated.")


def test_duplicate_key_values_are_not_treated_as_two_keys(monkeypatch):
    """OPENAI_API_KEY and OPEN_AI_API_KEY are usually aliases of ONE secret."""
    for name in [n for n in dict(__import__("os").environ) if "API_KEY" in n]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "same-secret")
    monkeypatch.setenv("OPEN_AI_API_KEY", "same-secret")
    ext = GptExtracter(fake_context(), gpt_config())

    assert ext.api_keys["open_ai"] == ["same-secret"]

    print("\n=== SANITY: alias de-duplication ===")
    print("  both env aliases of one secret -> 1 key, not 2. Validated.")


def test_a_missing_key_for_the_selected_provider_raises(monkeypatch):
    """The check this replaces read `len(self.api_keys) == 0` on a dict pre-seeded with one
    entry per provider, so it could never fire."""
    for name in [n for n in dict(__import__("os").environ) if "API_KEY" in n]:
        monkeypatch.delenv(name, raising=False)
    ext = GptExtracter(fake_context(), gpt_config())

    assert ext.api_keys["open_ai"] == []
    with pytest.raises(EnvironmentError, match="No API key for provider 'open_ai'"):
        ext.initialize_client("open_ai")

    print("\n=== SANITY: missing-key check ===")
    print("  no OPENAI key in the environment -> EnvironmentError, not an IndexError. Validated.")


def test_local_needs_no_key(monkeypatch):
    """`api_keys['local']` is never populated, so demanding one was an IndexError."""
    for name in [n for n in dict(__import__("os").environ) if "API_KEY" in n]:
        monkeypatch.delenv(name, raising=False)
    ext = GptExtracter(fake_context(), gpt_config())

    assert "local" in ext.available_methodes()
    ext._require_key("local")            # must not raise

    print("\n=== SANITY: keyless provider ===")
    print(f"  available with no keys set: {ext.available_methodes()}. Validated.")


# --------------------------------------------------------------------------- #
# Usage accounting                                                             #
# --------------------------------------------------------------------------- #
def test_usage_totals_are_lock_safe():
    """One tracker is shared across the pool, so the counter is a read-modify-write from
    every worker. Without the lock the call count silently under-reports."""
    tracker = UsageTracker()
    n = 200

    def record():
        tracker.record({"input_tokens": 10, "output_tokens": 1, "cached_input_tokens": 4})

    threads = [threading.Thread(target=record) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.totals["calls"] == n
    assert tracker.totals["input_tokens"] == 10 * n

    print("\n=== SANITY: concurrent usage accounting ===")
    print(f"  {n} threads recorded -> calls={tracker.totals['calls']}, "
          f"input={tracker.totals['input_tokens']}. Validated.")


def test_spend_estimate_treats_cached_tokens_as_a_subset():
    """`cached_input_tokens` is a SUBSET of `input_tokens`, not a separate bucket."""
    tracker = UsageTracker()
    tracker.record({"input_tokens": 1_000_000, "output_tokens": 0,
                    "cached_input_tokens": 1_000_000})
    all_cached = tracker.spend_estimate()

    fresh = UsageTracker()
    fresh.record({"input_tokens": 1_000_000, "output_tokens": 0, "cached_input_tokens": 0})

    assert all_cached == pytest.approx(0.025)
    assert fresh.spend_estimate() == pytest.approx(0.25)
    assert fresh.spend_estimate() == pytest.approx(10 * all_cached)

    print("\n=== SANITY: spend estimate ===")
    print(f"  1M input tokens: ${fresh.spend_estimate():.3f} fresh vs ${all_cached:.3f} "
          "fully cached -- the 10x the budget assumes. Validated.")


def test_extract_records_usage_through_a_stub_provider(monkeypatch):
    ext = _extracter(monkeypatch)
    ext.read_prompts("def14a")
    parsed = ext.extract(Answer, "some filing text", provider=StubProvider(), action="def14a")

    assert isinstance(parsed, Answer)
    assert ext.usage.totals["calls"] == 1
    assert ext.usage.totals["cached_input_tokens"] == 40

    print("\n=== SANITY: single-shot extract ===")
    print(f"  filled {type(parsed).__name__} and recorded {ext.usage.totals}. Validated.")


if __name__ == "__main__":
    for model in ("gpt-5-mini", "gpt-4o-mini"):
        provider = OpenAIProvider(model, client=object(), temperature=0.2, seed=1234,
                                  reasoning=model.startswith("gpt-5"))
        print(f"{model:<14} -> {sorted(provider.request_kwargs(Answer, 's', 'u'))}")
