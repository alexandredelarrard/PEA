"""`StepExtractEdgar` declares its window, run date, tickers and CIK mapping once, runs the six EDGAR walks in order
through the shared driver, and each `run_*` fails alone. Offline: the driver, the universe and the CIK mapping are stubbed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

import src.data_extract.cli as cli_mod
from src.data_extract.transformers import step_extract_edgar as step_mod
from src.data_extract.transformers.step_extract_edgar import EdgarOptions, StepExtractEdgar

RUNS = ("run_insider", "run_13d", "run_13g", "run_8k", "run_filing_text", "run_def14a_ecd")
FETCHES = {
    "run_insider": step_mod.INSIDER_EDGAR_FETCH,
    "run_13d": step_mod.SEC_13D_FETCH,
    "run_13g": step_mod.SEC_13G_FETCH,
    "run_8k": step_mod.SEC_8K_FETCH,
    "run_filing_text": step_mod.FILING_TEXT_FETCH,
    "run_def14a_ecd": step_mod.DEF14A_EDGAR_FETCH,
}
COMMANDS = {
    "insider-edgar": "run_insider",
    "sec-13d": "run_13d",
    "sec-13g": "run_13g",
    "sec-8k-items": "run_8k",
    "filing-text": "run_filing_text",
    "def14a-edgar": "run_def14a_ecd",
}
CIK_MAP = pd.DataFrame({"ticker": ["AAA", "BBB"], "cik": ["0000000001", "0000000002"]})


class Driver:
    """Records every declared read and driver call; `boom` names the fetch whose walk raises."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, boom: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reads: list[str] = []
        self.boom = FETCHES[boom] if boom else None
        monkeypatch.setattr(step_mod, "load_universe_tickers", self.universe)
        monkeypatch.setattr(step_mod, "load_cik_mapping", self.cik_mapping)
        monkeypatch.setattr(step_mod, "run_edgar_fetch", self.fetch)

    def universe(self, context: Any) -> list[str]:
        self.reads.append("universe")
        return ["AAA", "BBB"]

    def cik_mapping(self, context: Any, tickers: list[str]) -> pd.DataFrame:
        self.reads.append("cik_map")
        return CIK_MAP[CIK_MAP["ticker"].isin(tickers)]

    def fetch(self, context: Any, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if kwargs["fetch"] is self.boom:
            raise RuntimeError(f"simulated {kwargs['fetch'].done.desc} failure")

    @property
    def order(self) -> list[str]:
        return [call["fetch"].done.desc for call in self.calls]


def _config() -> SimpleNamespace:
    return SimpleNamespace(data_extract=SimpleNamespace(years_history=15))


def _step(options: EdgarOptions | None = None) -> StepExtractEdgar:
    config = _config()
    return StepExtractEdgar(SimpleNamespace(config=config, config_dir="configs"), config, options)  # type: ignore[arg-type]


def test_run_walks_the_six_fetches_in_order_on_state_declared_once(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Driver(monkeypatch)
    _step(EdgarOptions(as_of=pd.Timestamp("2026-10-31 15:00"))).run()

    assert driver.order == [FETCHES[name].done.desc for name in RUNS]
    assert driver.reads == ["universe", "cik_map"], "the universe and the CIK mapping are read once, in the init"
    first = driver.calls[0]
    assert first["cik_map"]["ticker"].tolist() == ["AAA", "BBB"] and first["tickers"] == ["AAA", "BBB"]
    assert all(call["cik_map"] is first["cik_map"] and call["tickers"] is first["tickers"] for call in driver.calls)
    assert {(call["years_history"], call["as_of"], call["full"], call["no_cap"]) for call in driver.calls} == {
        (15, pd.Timestamp("2026-10-31"), False, False)
    }
    print("\n=== SANITY: StepExtractEdgar.run order ===")
    print(f"  {' -> '.join(driver.order)}; one universe read and one CIK mapping shared by all 6 walks; window 15y, run date 2026-10-31.")


def test_options_reach_every_walk_and_a_subset_skips_the_universe_read(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Driver(monkeypatch)
    _step(EdgarOptions(tickers=["BBB"], years=3, full=True, no_cap=True)).run()

    assert driver.reads == ["cik_map"]
    assert {(tuple(call["tickers"]), call["years_history"], call["full"], call["no_cap"]) for call in driver.calls} == {(("BBB",), 3, True, True)}
    assert driver.calls[0]["cik_map"]["ticker"].tolist() == ["BBB"]
    print("\n=== SANITY: EdgarOptions ===")
    print("  -t BBB, -y 3, -F and --no-cap reach all 6 walks; no universe read when tickers are given.")


@pytest.mark.parametrize("boom", RUNS)
def test_a_failing_walk_raises_from_its_own_run_only(monkeypatch: pytest.MonkeyPatch, boom: str) -> None:
    driver = Driver(monkeypatch, boom=boom)
    step = _step()

    with pytest.raises(RuntimeError, match="simulated"):
        getattr(step, boom)()
    for name in RUNS:
        if name != boom:
            getattr(step, name)()
    alone = list(driver.order)
    driver.calls.clear()
    with pytest.raises(RuntimeError, match="simulated"):
        step.run()

    assert alone == [FETCHES[name].done.desc for name in (boom, *[n for n in RUNS if n != boom])]
    assert driver.order == [FETCHES[name].done.desc for name in RUNS[: RUNS.index(boom) + 1]], "run() is a plain sequence: it stops at the failure"
    print(f"\n=== SANITY: {boom} failure ===")
    print(f"  {boom} raised alone; the 5 other run_* methods ran when called alone; run() stopped after {len(driver.order)} walk(s).")


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_edgar_commands_call_their_step_operation(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    driver = Driver(monkeypatch)
    config = _config()
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, SimpleNamespace(config=config, config_dir="configs")))

    result = CliRunner().invoke(cli_mod.cli, [command, "-t", "bbb", "--as-of", "2026-09-30"])

    assert result.exit_code == 0, result.output
    assert driver.order == [FETCHES[COMMANDS[command]].done.desc] and driver.reads == ["cik_map"]
    assert driver.calls[0]["tickers"] == ["BBB"] and driver.calls[0]["as_of"] == pd.Timestamp("2026-09-30")
    print(f"\n=== SANITY: `{command}` -> StepExtractEdgar.{COMMANDS[command]} ===")
    print(f"  one walk ({driver.order[0]}) over ['BBB'] on 2026-09-30, CIK mapping read once.")
