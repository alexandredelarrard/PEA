"""
test_threaded_runner.py  (tests/gpt_extract/test_threaded_runner.py)
----------------------------------------------------------------------
`LLMExtractor`: ordering, failure isolation, deterministic shutdown, key rotation, and the
structural rule that a worker never touches the store.

All synthetic, all through a `StubProvider`: no network, no key, no spend.
"""
from __future__ import annotations

import threading

import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from tests.gpt_extract.conftest import Answer, StubProvider, fake_context, gpt_config


class RecordingStore:
    """A store that remembers WHICH THREAD touched it. The whole point of Decision 2."""

    def __init__(self) -> None:
        self.saves: list[tuple[str, int]] = []
        self.threads: set[str] = set()

    def save(self, table, df: pd.DataFrame, pk=None) -> int:
        self.threads.add(threading.current_thread().name)
        self.saves.append((str(table), len(df)))
        return len(df)

    def __getattr__(self, item):                 # any other access is recorded too
        self.threads.add(threading.current_thread().name)
        raise AttributeError(item)


def _runner(monkeypatch, n_threads=4, keys=("k1",), provider_factory=None, store=None):
    for name in [n for n in dict(__import__("os").environ) if "API_KEY" in n]:
        monkeypatch.delenv(name, raising=False)
    for i, key in enumerate(keys):
        monkeypatch.setenv(f"OPENAI_API_KEY{'' if i == 0 else f'_{i}'}", key)

    runner = LLMExtractor(fake_context(store), gpt_config(), action="def14a",
                          threads=n_threads)
    factory = provider_factory or (lambda methode=None, key_index=None: StubProvider())
    monkeypatch.setattr(runner, "initialize_client", factory)
    return runner


def _tasks(n: int, payload=lambda i: f"payload-{i}") -> list[LlmTask]:
    return [LlmTask(seq=i, payload=payload(i), schema=Answer, table=Tables.def14a_llm,
                    meta={"ticker": f"T{i % 3}"}) for i in range(n)]


# --------------------------------------------------------------------------- #
# Ordering                                                                     #
# --------------------------------------------------------------------------- #
def test_results_come_back_in_submission_order(monkeypatch):
    """Completion order is the REVERSE of submission order here; the caller zips results
    against its own filing list, so submission order is the contract."""
    import time

    n = 8
    # task i sleeps (n - i) units, so the LAST submitted finishes FIRST
    delays = {f"payload-{i}": 0.02 * (n - i) for i in range(n)}
    completed: list[str] = []

    class Sleeper(StubProvider):
        def parse(self, schema, system, user):
            # the payload is LAST in the prompt, so endswith identifies the task
            tail = user.rstrip()
            payload = next(p for p in delays if tail.endswith(p))
            time.sleep(delays[payload])
            completed.append(payload)
            return super().parse(schema, system, user)

    runner = _runner(monkeypatch, n_threads=n,
                     provider_factory=lambda methode=None, key_index=None: Sleeper())
    for task in _tasks(n):
        runner.submit(task)
    results = runner.run()

    assert [r.seq for r in results] == list(range(n))
    assert all(r.ok for r in results)
    assert len(completed) == n
    assert completed == [f"payload-{i}" for i in reversed(range(n))], \
        f"the stub must complete in REVERSE order or ordering is untested: {completed}"

    print("\n=== SANITY: submission ordering ===")
    print(f"  completion order {completed}")
    print(f"  returned order   {[r.seq for r in results]} -- submission order. Validated.")


# --------------------------------------------------------------------------- #
# Failure isolation                                                            #
# --------------------------------------------------------------------------- #
def test_one_failing_task_does_not_lose_the_others(monkeypatch):
    """A dead worker with tasks still queued is a hang; a failure must become a result."""
    runner = _runner(monkeypatch, n_threads=4,
                     provider_factory=lambda methode=None, key_index=None:
                     StubProvider(fail_on=("payload-3",)))
    for task in _tasks(10):
        runner.submit(task)
    results = runner.run()

    assert len(results) == 10
    failed = [r for r in results if not r.ok]
    assert len(failed) == 1 and failed[0].seq == 3
    assert "stub failure" in failed[0].error
    assert sum(r.ok for r in results) == 9

    print("\n=== SANITY: failure isolation ===")
    print(f"  task 3 of 10 raised -> 9 parsed + 1 error ({failed[0].error!r}); "
          "the run terminated. Validated.")


def test_a_raising_task_returns_its_client_to_the_queue(monkeypatch):
    """The deadlock guard: the client goes back in `finally`, or every worker ends up
    blocked on an empty client queue."""
    runner = _runner(monkeypatch, n_threads=2,
                     provider_factory=lambda methode=None, key_index=None:
                     StubProvider(fail_on=("payload-0", "payload-1")))
    for task in _tasks(6):
        runner.submit(task)
    results = runner.run()                       # must not hang

    assert len(results) == 6
    assert runner._clients.qsize() == 2, "both clients must be back in the queue"

    print("\n=== SANITY: client returned on failure ===")
    print(f"  2 raising tasks, {runner._clients.qsize()} clients back in the queue, "
          "run terminated. Validated.")


# --------------------------------------------------------------------------- #
# Writes stay on the main thread                                               #
# --------------------------------------------------------------------------- #
def test_no_worker_ever_touches_the_store(monkeypatch):
    """`store.ensure_table` is a check-then-create with no lock, so a concurrent writer
    against a COLD table can silently lose rows. Workers get a task and a provider, never
    a Context -- this pins that structurally."""
    store = RecordingStore()
    runner = _runner(monkeypatch, n_threads=6, store=store)
    results = runner.run_extraction(_tasks(12))

    assert len(results) == 12
    assert store.threads == {"MainThread"}, f"store touched from {store.threads}"

    print("\n=== SANITY: store is main-thread only ===")
    print(f"  12 tasks over 6 workers; every store access came from {store.threads}. Validated.")


def test_saves_happen_once_per_group(monkeypatch):
    """One save per ticker per table -- an interrupted run then costs at most one ticker."""
    store = RecordingStore()
    runner = _runner(monkeypatch, n_threads=4, store=store)
    tasks = [LlmTask(seq=i, payload=f"p{i}", schema=Answer, table=Tables.def14a_llm,
                     meta={"ticker": f"T{i // 5}"}) for i in range(15)]

    runner.run_extraction(tasks, group_key=lambda t: str(t.meta["ticker"]))

    assert len(store.saves) == 3, store.saves
    assert [n for _, n in store.saves] == [5, 5, 5]

    print("\n=== SANITY: one save per group ===")
    print(f"  3 tickers x 5 filings -> {len(store.saves)} saves of {[n for _, n in store.saves]} "
          "rows. Validated.")


def test_the_default_flatten_writes_one_row_per_result(monkeypatch):
    store = RecordingStore()
    runner = _runner(monkeypatch, n_threads=2, store=store)
    runner.run_extraction(_tasks(4))

    assert store.saves == [(str(Tables.def14a_llm), 4)]

    print("\n=== SANITY: default 1:1 flatten ===")
    print(f"  4 answers -> {store.saves}. Validated.")


def test_a_fanout_flatten_writes_every_table(monkeypatch):
    """One answer fans out to several tables -- the DEF 14A extract writes five."""
    store = RecordingStore()
    runner = _runner(monkeypatch, n_threads=2, store=store)

    def flatten(result: LlmResult):
        return {Tables.def14a_llm: pd.DataFrame([{"a": 1}]),
                Tables.def14a_directors: pd.DataFrame([{"b": 1}, {"b": 2}])}

    runner.run_extraction(_tasks(3), flatten=flatten)

    saved = dict(store.saves)
    assert saved[str(Tables.def14a_llm)] == 3
    assert saved[str(Tables.def14a_directors)] == 6

    print("\n=== SANITY: fan-out flatten ===")
    print(f"  3 answers -> {store.saves}; one call, two tables. Validated.")


# --------------------------------------------------------------------------- #
# Keys and shutdown                                                            #
# --------------------------------------------------------------------------- #
def test_clients_rotate_over_every_key(monkeypatch):
    """M keys shared by N workers without partitioning them."""
    seen: list[int | None] = []

    def factory(methode=None, key_index=None):
        seen.append(key_index)
        return StubProvider(api_key=f"k{key_index}")

    runner = _runner(monkeypatch, n_threads=12, keys=("k1", "k2", "k3"),
                     provider_factory=factory)
    for task in _tasks(24):
        runner.submit(task)
    runner.run()

    assert sorted(set(seen)) == [0, 1, 2], seen

    print("\n=== SANITY: key rotation across the pool ===")
    print(f"  3 keys, 12 workers -> client key indices {sorted(set(seen))}; all used. Validated.")


@pytest.mark.parametrize("attempt", range(20))
def test_shutdown_is_not_racy(monkeypatch, attempt):
    """`Queue.qsize()` is advisory and workers concurrently put clients back, so sizing the
    shutdown on it is a race. One sentinel per worker is not."""
    runner = _runner(monkeypatch, n_threads=12)
    for task in _tasks(50):
        runner.submit(task)
    results = runner.run()

    assert len(results) == 50
    assert all(r.ok for r in results)

    if attempt == 19:
        print("\n=== SANITY: deterministic shutdown ===")
        print("  50 tasks over 12 workers, 20 consecutive runs, every one terminated. Validated.")


def test_an_empty_task_list_is_not_an_error(monkeypatch):
    runner = _runner(monkeypatch, n_threads=4)
    assert runner.run() == []
    assert runner.run_extraction([]) == []

    print("\n=== SANITY: empty run ===")
    print("  no tasks -> no threads, no clients, no saves, empty list. Validated.")


if __name__ == "__main__":
    import types

    class _Cfg:
        pass

    print("run via pytest -s for the sanity output")
