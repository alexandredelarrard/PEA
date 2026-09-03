"""
gpt_getter.py  (src/gpt_extract/transformers/gpt_getter.py)
------------------------------------------------------------
`LLMExtractor`: fill N schemas concurrently, return them in submission order, and write
only from the main thread.

Concurrency is not an optimisation here, it is what makes a backfill possible: measured
serially, one modern proxy takes ~94s (a 130k-char payload on a reasoning model), so 656
filings would be ~17h. The work is entirely network-bound on an API that is fine with
parallel requests.
"""
from __future__ import annotations

from queue import Queue
from threading import Lock, Thread
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.context import Context
from src.data_store.schema import Table
from src.gpt_extract.transformers.step_gpt_extracter import GptExtracter
from src.gpt_extract.utils.providers import _Provider
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

#: What `flatten` returns: the frames one answer produces, keyed by destination table.
FlattenFn = Callable[[LlmResult], Mapping[Table, pd.DataFrame]]
GroupKeyFn = Callable[[LlmTask], str]


class LLMExtractor(GptExtracter):
    """Threaded schema-filling over a list of payloads.

    Two queues. `clients` holds initialised providers so N workers rotate across every API
    key available -- that is what the queue is FOR, and with a single key every slot holds
    the same client object, which is correct because an OpenAI client is thread-safe and
    pools its connections. `tasks` holds the payloads.

    Results come back in SUBMISSION order regardless of completion order, because the
    caller zips them against its own filing list.

    **Workers never write.** They receive a task and a provider, and never a `Context`.
    Two properties depend on that, both earned the hard way:

    1. `store.ensure_table` is a check-then-create with no lock, so concurrent writers
       against a COLD table can each see "absent", each create, and silently lose rows.
    2. An interrupted run must lose no paid tokens: one save per ticker after that
       ticker's filings are all parsed means a Ctrl-C costs at most one ticker's calls.
    """

    def __init__(self, context: Context, config: DictConfig, action: str | None = None,
                 threads: int | None = None, methodes: Sequence[str] | None = None):
        super().__init__(context=context, config=config, action=action)

        self.threads = int(threads or self.threads)
        self.methodes = list(methodes) if methodes else None
        self._tasks: Queue = Queue()
        self._clients: Queue = Queue()
        self._results: dict[int, LlmResult] = {}
        self._results_lock = Lock()
        self._submitted = 0

    # ------------------------------------------------------------------- queues --- #

    def submit(self, task: LlmTask) -> None:
        self._tasks.put(task)
        self._submitted += 1

    def initialize_queue_clients(self, n_slots: int) -> list[_Provider]:
        """Fill the client queue with `n_slots` entries drawn from every (provider, key)
        pair available, so M keys are shared by N workers without partitioning them."""
        methodes = self.methodes or [m for m in (self.default_api,)
                                     if m in self.available_methodes()]
        if not methodes:
            methodes = [self.default_api]

        distinct: list[_Provider] = []
        for methode in methodes:
            keys = self.api_keys.get(methode) or []
            for index in range(len(keys) or 1):
                distinct.append(self.initialize_client(methode, key_index=index))

        for slot in range(n_slots):
            self._clients.put(distinct[slot % len(distinct)])
        return distinct

    def close_queue_clients(self, n_workers: int) -> None:
        """One sentinel per worker.

        Not `qsize()` sentinels: `Queue.qsize()` is advisory on every platform and workers
        are concurrently putting clients back, so sizing the shutdown on it is a race.
        """
        for _ in range(n_workers):
            self._tasks.put(None)

    # ------------------------------------------------------------------ workers --- #

    def _worker(self, progress: tqdm | None = None) -> None:
        """Call and parse. Never touch the store, never let an exception escape.

        A dead worker with tasks still queued is a hang, so every failure becomes an
        `LlmResult` carrying its error rather than propagating out of the thread.
        """
        while True:
            task = self._tasks.get()
            if task is None:                       # sentinel -> this worker is done
                self._tasks.task_done()
                return

            provider = self._clients.get()
            try:
                system, user = self.build_prompt(self.truncate(task.payload), task.schema)
                parsed, usage = provider.parse(task.schema, system, user)
                self.usage.record(usage)
                result = LlmResult(seq=task.seq, task=task, parsed=parsed, usage=usage)
            except Exception as exc:               # noqa: BLE001 -- one filing must not
                result = LlmResult(seq=task.seq, task=task, parsed=None,  # kill the run
                                   error=f"{type(exc).__name__}: {exc}")
            finally:
                # In `finally` or a raising task drains the client queue and the run
                # deadlocks with every worker blocked on `clients.get()`.
                self._clients.put(provider)

            with self._results_lock:
                self._results[task.seq] = result
                if progress is not None:
                    progress.update(1)
            self._tasks.task_done()

    def run(self) -> list[LlmResult]:
        """Drain the queue and return every result in SUBMISSION order."""
        total = self._submitted
        if total == 0:
            return []

        n_workers = max(1, min(self.threads, total))
        self.initialize_queue_clients(n_workers)

        with tqdm(total=total, desc=f"llm:{self.action or 'extract'}", unit="call") as bar:
            workers = [Thread(target=self._worker, args=(bar,), daemon=True)
                       for _ in range(n_workers)]
            for worker in workers:
                worker.start()
            self.close_queue_clients(n_workers)
            for worker in workers:
                worker.join()

        # From the results DICT, not the queue: a raising task must still occupy its slot.
        results = [self._results[seq] for seq in range(total)]
        self._submitted = 0
        self._results = {}
        self._log.info("%d call(s), %s, $%.2f, cached input %.0f%%",
                       self.usage.totals["calls"], self.usage.totals,
                       self.usage.spend_estimate(), 100 * self.usage.cached_share)
        return results

    # ---------------------------------------------------------------- the entry --- #

    @staticmethod
    def _default_flatten(result: LlmResult) -> Mapping[Table, pd.DataFrame]:
        """The simple 1:1 case: one answer, one row, in the task's own table."""
        if result.parsed is None or result.task.table is None:
            return {}
        return {result.task.table: pd.DataFrame([result.parsed.model_dump()])}

    def run_extraction(
        self,
        tasks: Iterable[LlmTask],
        flatten: FlattenFn | None = None,
        group_key: GroupKeyFn | None = None,
    ) -> list[LlmResult]:
        """Fill every task's schema, then save from the MAIN thread, once per group.

        `flatten` is a callable rather than a single table because one answer can fan out
        to several tables -- the DEF 14A extract writes five.
        """
        for task in tasks:
            self.submit(task)

        results = self.run()
        flatten = flatten or self._default_flatten

        groups: dict[str, list[LlmResult]] = {}
        for result in results:
            key = group_key(result.task) if group_key else "_all"
            groups.setdefault(key, []).append(result)

        for key, group in groups.items():
            frames: dict[Table, list[pd.DataFrame]] = {}
            for result in group:
                if not result.ok:
                    continue
                for table, frame in (flatten(result) or {}).items():
                    if frame is None or len(frame) == 0:
                        continue
                    frames.setdefault(table, []).append(frame)
            for table, parts in frames.items():
                self._context.store.save(table, pd.concat(parts, ignore_index=True))

        failed = [r for r in results if not r.ok]
        if failed:
            self._log.warning("%d of %d call(s) failed: %s", len(failed), len(results),
                              {r.seq: r.error for r in failed[:5]})
        return results
