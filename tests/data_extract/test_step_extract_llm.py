"""`StepExtractLlm` builds its DEF 14A and votes extractors once, raises in its init when a provider key is missing,
runs DEF 14A, votes and employees in that order, and each `run_*` fails alone. Offline: the three fetchers are stubbed
and no provider client is ever built."""

from __future__ import annotations

import os
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

import src.data_extract.cli as cli_mod
from src.data_extract import step_extract_all_data as all_data_module
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_extract.transformers import step_extract_llm as step_mod
from src.data_extract.transformers.step_extract_llm import LlmOptions, StepExtractLlm
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import GptExtracter
from tests.data_extract.fake_context import extract_config
from tests.support.universe import universe_of

RUNS = ("run_def14a", "run_8k_votes", "run_employees")
COMMANDS = {"def14a": "run_def14a", "sec-8k-votes": "run_8k_votes", "fundamentals-employees": "run_employees"}
#: The fragments `GptExtracter.get_api_keys` matches for the OpenAI providers this step uses.
KEY_FRAGMENTS = ("OPENAI_API_KEY", "OPEN_AI_API_KEY")


def _config() -> Any:
    return extract_config(data_extract={"years_history": 15})


class _Context(SimpleNamespace):
    """A stub context whose universe read is recorded by `on_read`."""

    @property
    def universe(self) -> Any:
        self.on_read()
        return self.universe_tickers


def _context(config: Any, fetchers: Fetchers | None = None) -> Any:
    on_read = (lambda: fetchers.call("universe")) if fetchers is not None else (lambda: None)
    return _Context(config=config, config_dir="configs", on_read=on_read, universe_tickers=universe_of(["AAA", "BBB"]))


def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in os.environ if any(fragment in name for fragment in KEY_FRAGMENTS)]:
        monkeypatch.delenv(name)


class Fetchers:
    """The three fetchers, stubbed on the step module: each records its call and the extractor objects it received."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, boom: str | None = None) -> None:
        self.order: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.boom = boom
        monkeypatch.setattr(
            step_mod, "fetch_def14a_llm", lambda context, config, tickers, extractors: self.call("run_def14a", tickers=tickers, extractors=extractors)
        )
        monkeypatch.setattr(
            step_mod, "fetch_8k_votes_llm", lambda context, tickers, extractor: self.call("run_8k_votes", tickers=tickers, extractor=extractor)
        )
        monkeypatch.setattr(
            step_mod,
            "fetch_fundamentals_employees",
            lambda context, tickers, years_history, full=False: self.call("run_employees", tickers=tickers, years=years_history, full=full),
        )

    def call(self, name: str, **kwargs: Any) -> None:
        self.order.append(name)
        self.calls.append({"name": name, **kwargs})
        if name == self.boom:
            raise RuntimeError(f"simulated {name} failure")


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Counts each extractor setup call; the wrapped methods still run, and `initialize_client` (a provider) must not."""
    counts: Counter = Counter()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for owner, name in (
        (LLMExtractor, "__init__"),
        (GptExtracter, "read_prompts"),
        (GptExtracter, "get_api_keys"),
        (GptExtracter, "initialize_client"),
    ):
        original = getattr(owner, name)

        def wrapped(self: Any, *args: Any, _original: Any = original, _label: str = f"{owner.__name__}.{name}", **kwargs: Any) -> Any:
            counts[_label] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)
    return counts


def test_the_extractors_are_built_once_and_reused_by_every_run(monkeypatch: pytest.MonkeyPatch, counted: Counter) -> None:
    fetchers = Fetchers(monkeypatch)
    config = _config()
    step = StepExtractLlm(_context(config, fetchers), config)
    built = dict(counted)
    assert "universe" not in fetchers.order, "constructing the step reads no universe (REQ-014)"
    for _ in range(2):
        step.run_def14a()
        step.run_8k_votes()

    # One extractor per action: the five DEF 14A sections and the Item 5.07 votes; each reads its prompts and scans the keys once.
    assert built == {"LLMExtractor.__init__": 6, "GptExtracter.read_prompts": 6, "GptExtracter.get_api_keys": 6}, built
    assert dict(counted) == built, "a run_* call builds nothing"
    def14a = [call["extractors"] for call in fetchers.calls if call["name"] == "run_def14a"]
    votes = [call["extractor"] for call in fetchers.calls if call["name"] == "run_8k_votes"]
    assert def14a[0] is def14a[1] and votes[0] is votes[1] and len(def14a[0]) == 5
    assert {extractor.threads for extractor in def14a[0].values()} == {2} and votes[0].threads == int(config.gpt.threads)
    assert fetchers.order.count("universe") == 1
    print("\n=== SANITY: StepExtractLlm setup (AC-010) ===")
    print(
        f"  init: {built}; 2x run_def14a + 2x run_8k_votes added nothing and reused the same 6 extractors; no provider built; universe read once, on first use."
    )


def test_a_missing_key_raises_in_the_init_and_nothing_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_keys(monkeypatch)
    fetchers = Fetchers(monkeypatch)
    config = _config()

    with pytest.raises(OSError, match="No API key for provider 'open_ai'"):
        StepExtractLlm(_context(config, fetchers), config)

    assert [name for name in fetchers.order if name != "universe"] == []
    print("\n=== SANITY: missing key (D-11) ===")
    print("  no OPENAI key in the environment -> StepExtractLlm() raised OSError; no fetcher was called.")


def test_all_data_is_built_without_a_key_and_builds_the_llm_step_only_in_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_keys(monkeypatch)
    built: list[str] = []
    for name in (
        "StepExtractPrices",
        "StepExtractInstitutionals",
        "StepExtractFundamentalsSharadar",
        "StepExtractFundamentals",
        "StepExtractBehavioral",
    ):
        monkeypatch.setattr(all_data_module, name, lambda context, config, _name=name: built.append(_name))
    monkeypatch.setattr(all_data_module, "StepExtractLlm", lambda *args: pytest.fail("the LLM step is built lazily, inside run()"))
    config = _config()

    StepExtractAllData(_context(config), config)

    assert len(built) == 5
    print("\n=== SANITY: lazy LLM step (D-11) ===")
    print("  StepExtractAllData built its 5 eager sub-steps with no API key; StepExtractLlm was not constructed.")


def test_run_is_the_three_extractions_in_order_over_the_declared_scope(monkeypatch: pytest.MonkeyPatch, counted: Counter) -> None:
    fetchers = Fetchers(monkeypatch)
    config = _config()

    StepExtractLlm(_context(config), config, LlmOptions(tickers=["ZZ"], full=True)).run()

    assert fetchers.order == list(RUNS), fetchers.order
    assert all(call["tickers"] == ["ZZ"] for call in fetchers.calls)
    assert fetchers.calls[-1]["years"] == 15 and fetchers.calls[-1]["full"] is True
    print("\n=== SANITY: StepExtractLlm.run order ===")
    print(f"  {' -> '.join(fetchers.order)} over ['ZZ'] (no universe read); employees got 15 years and full=True.")


@pytest.mark.parametrize("boom", RUNS)
def test_a_failing_run_raises_from_its_own_run_only(monkeypatch: pytest.MonkeyPatch, counted: Counter, boom: str) -> None:
    fetchers = Fetchers(monkeypatch, boom=boom)
    config = _config()
    step = StepExtractLlm(_context(config), config, LlmOptions(tickers=["ZZ"]))

    with pytest.raises(RuntimeError, match=boom):
        getattr(step, boom)()
    for name in (n for n in RUNS if n != boom):
        getattr(step, name)()
    alone = list(fetchers.order)
    fetchers.order.clear()
    with pytest.raises(RuntimeError, match=boom):
        step.run()

    assert alone == [boom, *[n for n in RUNS if n != boom]]
    assert fetchers.order == list(RUNS[: RUNS.index(boom) + 1]), "run() is a plain sequence: it stops at the failure"
    print(f"\n=== SANITY: {boom} failure ===")
    print(f"  {boom} raised alone; the 2 other run_* methods ran alone; run() stopped after {len(fetchers.order)} extraction(s).")


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_llm_commands_call_their_step_operation(monkeypatch: pytest.MonkeyPatch, counted: Counter, command: str) -> None:
    fetchers = Fetchers(monkeypatch)
    config = _config()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, _context(config)))

    result = CliRunner().invoke(cli_mod.cli, [command, "-t", "bbb"])

    assert result.exit_code == 0, result.output
    assert fetchers.order == [COMMANDS[command]] and fetchers.calls[0]["tickers"] == ["BBB"]
    print(f"\n=== SANITY: `{command}` -> StepExtractLlm.{COMMANDS[command]} ===")
    print(f"  one call over ['BBB'], extractors built once ({counted['LLMExtractor.__init__']}).")
