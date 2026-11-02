"""`StepExtractEdgar` declares its window, run date, tickers, CIK mapping, identity scope, cap and retry policy once,
runs the six EDGAR walks in order through the shared driver, and each `run_*` fails alone. Offline: the driver, the
universe, the CIK mapping and the identity are stubbed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

import src.data_extract.cli as cli_mod
from src.data_extract.transformers import step_extract_edgar as step_mod
from src.data_extract.transformers.step_extract_edgar import EdgarOptions, StepExtractEdgar
from src.data_extract.utils.edgar_extract import edgar_driver, edgar_index
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarRun
from tests.data_extract.edgar_fixtures import walk_call

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

    def __init__(self, monkeypatch: pytest.MonkeyPatch, boom: str | None = None, refreshed_today: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.refreshes: list[int] = []
        self.events: list[str] = []
        self.refreshed_today = refreshed_today
        self.reads: list[str] = []
        self.boom = FETCHES[boom] if boom else None
        self.runs: list[EdgarRun] = []
        monkeypatch.setattr(step_mod, "load_universe_tickers", self.universe)
        monkeypatch.setattr(edgar_driver, "load_cik_mapping", self.cik_mapping)
        monkeypatch.setattr(edgar_driver, "load_added_on", lambda ctx, day: {})
        monkeypatch.setattr(edgar_driver, "load_identity", self.identity)
        monkeypatch.setattr(step_mod, "run_edgar_fetch", self.fetch)
        monkeypatch.setattr(step_mod, "refresh_index", self.refresh)
        monkeypatch.setattr(edgar_index, "refreshed_today", lambda context, years_history: self.refreshed_today)

    def refresh(self, context: Any, run: EdgarRun, years_history: int) -> None:
        self.refreshes.append(years_history)
        self.events.append(f"refresh {years_history}y")

    def universe(self, context: Any) -> list[str]:
        self.reads.append("universe")
        return ["AAA", "BBB"]

    def cik_mapping(self, context: Any, tickers: list[str]) -> pd.DataFrame:
        self.reads.append("cik_map")
        return CIK_MAP[CIK_MAP["ticker"].isin(tickers)]

    def identity(self, context: Any) -> None:
        self.reads.append("identity")

    def fetch(self, context: Any, fetch: EdgarFetch, run: EdgarRun) -> None:
        self.calls.append(walk_call(fetch, run))
        self.runs.append(run)
        self.events.append(fetch.done.desc)
        if fetch is self.boom:
            raise RuntimeError(f"simulated {fetch.done.desc} failure")

    @property
    def order(self) -> list[str]:
        return [call["fetch"].done.desc for call in self.calls]


def _config() -> SimpleNamespace:
    return SimpleNamespace(data_extract=SimpleNamespace(years_history=15, max_documents_per_run=100))


def _step(options: EdgarOptions | None = None) -> StepExtractEdgar:
    config = _config()
    return StepExtractEdgar(SimpleNamespace(config=config, config_dir="configs"), config, options)  # type: ignore[arg-type]


def test_run_walks_the_six_fetches_in_order_on_state_declared_once(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Driver(monkeypatch)
    _step(EdgarOptions(as_of=pd.Timestamp("2026-10-31 15:00"))).run()

    assert driver.order == [FETCHES[name].done.desc for name in RUNS]
    assert driver.reads == ["universe", "cik_map", "identity"], "the universe, the CIK mapping and the identity are read once, in the init"
    first = driver.calls[0]
    assert first["cik_map"]["ticker"].tolist() == ["AAA", "BBB"] and first["tickers"] == ["AAA", "BBB"]
    assert all(run is driver.runs[0] for run in driver.runs), "every walk receives the one declared EdgarRun"
    assert driver.runs[0].cap == 100 and driver.runs[0].retry.rounds == 3
    assert {(call["years_history"], call["as_of"], call["full"], call["no_cap"]) for call in driver.calls} == {
        (15, pd.Timestamp("2026-10-31"), False, False)
    }
    print("\n=== SANITY: StepExtractEdgar.run order ===")
    print(
        f"  {' -> '.join(driver.order)}; one universe, CIK mapping and identity read, one EdgarRun (cap 100) shared by all 6 walks; window 15y, run date 2026-10-31."
    )


def test_options_reach_every_walk_and_a_subset_skips_the_universe_read(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Driver(monkeypatch)
    _step(EdgarOptions(tickers=["BBB"], years=3, full=True, no_cap=True)).run()

    assert driver.reads == ["cik_map", "identity"]
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
    assert driver.order == [FETCHES[COMMANDS[command]].done.desc] and driver.reads == ["cik_map", "identity"]
    assert driver.calls[0]["tickers"] == ["BBB"] and driver.calls[0]["as_of"] == pd.Timestamp("2026-09-30")
    print(f"\n=== SANITY: `{command}` -> StepExtractEdgar.{COMMANDS[command]} ===")
    print(f"  one walk ({driver.order[0]}) over ['BBB'] on 2026-09-30, CIK mapping and identity read once.")


#: The insider walk certifies its cohorts on the index inventory, so its window reaches the table's source start.
INSIDER_YEARS = 2026 - pd.Timestamp(str(step_mod.INSIDER_EDGAR_FETCH.done.resume.source_start)).year + 1


def test_run_refreshes_the_index_once_over_the_widest_window_before_the_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = Driver(monkeypatch)
    _step(EdgarOptions(as_of=pd.Timestamp("2026-10-31"))).run()

    assert driver.refreshes == [max(15, INSIDER_YEARS)], "one refresh per run(), over the insider walk's source-start window"
    assert driver.events[0].startswith("refresh") and len(driver.events) == 7
    print("\n=== SANITY: one index refresh per run() ===")
    print(f"  {driver.events[0]} then the 6 walks; edgar_index.refresh 6 -> 1.")


@pytest.mark.parametrize("name", RUNS)
def test_a_walk_alone_refreshes_its_own_window_once(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    driver = Driver(monkeypatch)
    step = _step(EdgarOptions(as_of=pd.Timestamp("2026-10-31")))
    getattr(step, name)()
    getattr(step, name)()

    expected = max(15, INSIDER_YEARS) if name == "run_insider" else 15
    assert driver.refreshes == [expected], "a lone walk refreshes once per step, over its own window"
    print(f"\n=== SANITY: {name} alone ===")
    print(f"  called twice on one step: one refresh over {expected}y, two walks.")


@pytest.mark.parametrize("name", RUNS)
def test_a_walk_alone_skips_a_refresh_already_done_today(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    driver = Driver(monkeypatch, refreshed_today=True)
    getattr(_step(), name)()
    backdated = Driver(monkeypatch, refreshed_today=True)
    getattr(_step(EdgarOptions(as_of=pd.Timestamp("2026-10-31"))), name)()

    assert driver.refreshes == [] and len(driver.calls) == 1, "today's nightly refresh covers a lone walk, the certifying insider walk included"
    assert backdated.refreshes != [], "a backdated run never relies on today's refresh"
    print(f"\n=== SANITY: {name} after the nightly edgar-index ===")
    print(f"  refreshes today: {driver.refreshes} (the insider walk certifies from the quarter footers); backdated run refreshes.")


def test_edgar_index_command_refreshes_the_widest_walk_window(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(data_extract=SimpleNamespace(years_history=3))
    context = SimpleNamespace(config=config, ensure_edgar_identity=lambda: None)
    refreshes: list[tuple[pd.Timestamp, int, bool]] = []
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))
    monkeypatch.setattr(edgar_index, "refresh", lambda ctx, as_of, years, *, rebuild=False: refreshes.append((as_of, years, rebuild)))
    monkeypatch.setattr(edgar_index, "quarter_counts", lambda ctx: {})

    result = CliRunner().invoke(cli_mod.cli, ["edgar-index", "--as-of", "2026-10-31"])

    assert result.exit_code == 0, result.output
    assert refreshes == [(pd.Timestamp("2026-10-31"), INSIDER_YEARS, False)], (
        "the nightly refresh reaches the insider source start, not only years_history"
    )
    print("\n=== SANITY: edgar-index window ===")
    print(f"  years_history 3 -> one refresh over {INSIDER_YEARS}y (insider source start), the window run() would refresh.")
