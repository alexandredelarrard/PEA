"""`StepExtractPrices` runs prices, macro, then earnings surprises (moved from `StepExtractFundamentals`, REQ-008);
`run_earnings_surprises` alone reaches only its fetcher with the configured window; the `earnings-surprises` command
calls it. No source is read: every fetcher is a recorder."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from src.data_extract import cli as cli_mod
from src.data_extract.transformers import step_extract_prices as mod
from src.data_extract.transformers.step_extract_prices import StepExtractPrices
from tests.support.universe import universe_of

ORDER = ["price_actions", "dividends", "prices", "macro", "earnings_surprises"]


def _step(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict[str, Any]]], *, boom: str | None = None) -> StepExtractPrices:
    def recorder(label: str):
        def fn(context: Any, *args: Any, **kwargs: Any) -> None:
            calls.append((label, kwargs))
            if label == boom:
                raise RuntimeError(f"simulated {label} failure")

        return fn

    monkeypatch.setattr(mod, "fetch_price_actions", recorder("price_actions"))
    monkeypatch.setattr(mod, "fetch_dividends", recorder("dividends"))
    monkeypatch.setattr(mod, "fetch_prices", recorder("prices"))
    monkeypatch.setattr(mod, "fetch_macro", recorder("macro"))
    monkeypatch.setattr(mod, "fetch_earnings_surprises", recorder("earnings_surprises"))
    config = OmegaConf.create({"data_extract": {"years_history": 15, "macro_years_history": 30}})
    context = SimpleNamespace(config=config, log=logging.getLogger("test"), config_dir="configs", universe=universe_of(["AAPL"]))
    return StepExtractPrices(context=context, config=config)


def test_run_ends_with_earnings_surprises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    _step(monkeypatch, calls).run(tickers=["AAPL"])

    assert [label for label, _ in calls] == ORDER
    assert calls[4][1] == {"tickers": ["AAPL"], "years_history": 15, "full": False}
    print("\n=== SANITY CHECK: StepExtractPrices order ===")
    print(f"  {' -> '.join(ORDER)}; earnings surprises get the equity window 15 (not the macro 30). Validated.")


def test_run_earnings_surprises_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    _step(monkeypatch, calls).run_earnings_surprises(["MSFT"])

    assert calls == [("earnings_surprises", {"tickers": ["MSFT"], "years_history": 15, "full": False})]
    print("\n=== SANITY CHECK: run_earnings_surprises alone ===\n  one fetcher call, no price or macro download. Validated.")


def test_a_failing_source_aborts_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """No per-source try/except (moved from `test_step_extract_fundamentals`): a failure propagates and the rest never run."""
    calls: list[tuple[str, dict[str, Any]]] = []
    with pytest.raises(RuntimeError, match="simulated macro failure"):
        _step(monkeypatch, calls, boom="macro").run(tickers=["AAPL"])

    assert [label for label, _ in calls] == ["price_actions", "dividends", "prices", "macro"]
    print("\n=== SANITY CHECK: StepExtractPrices direct calls, no isolation ===")
    print("  macro raised -> earnings surprises never ran. Validated.")


def test_the_cli_command_calls_the_prices_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    step = _step(monkeypatch, calls)
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (step.config, step._context))
    monkeypatch.setattr(cli_mod, "_tickers", lambda ctx, names: ["AAA"])

    result = CliRunner().invoke(cli_mod.cli, ["earnings-surprises", "-t", "aaa"])

    assert result.exit_code == 0, result.output
    assert calls == [("earnings_surprises", {"tickers": ["AAA"], "years_history": 15, "full": False})]
    print("\n=== SANITY CHECK: earnings-surprises command ===\n  calls StepExtractPrices.run_earnings_surprises once. Validated.")
