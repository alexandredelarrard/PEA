"""`ensure_table` serializes the CREATE of a cold table across threads.

File-backed SQLite so every thread gets its own connection. `table_exists` is patched so each
thread's FIRST existence check waits on a barrier: all eight threads see the table missing at
the same moment, which is the cold-table race every from-scratch fetcher run starts in.
"""

from __future__ import annotations

import threading

import pandas as pd
from sqlalchemy import create_engine, event

from src.data_store import schema
from src.data_store import store as store_module
from src.data_store.schema import Table
from src.data_store.store import DataStore

_N_THREADS = 8
_T_PROBE = Table("lock_probe", ("ticker",), date_col=None)


def test_ensure_table_creates_a_cold_table_once_under_concurrent_saves(tmp_path, monkeypatch):
    monkeypatch.setitem(schema.BY_NAME, _T_PROBE.name, _T_PROBE)
    engine = create_engine(f"sqlite:///{tmp_path / 'lock.db'}", connect_args={"timeout": 30, "check_same_thread": False})
    creates: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _count_creates(conn, cursor, statement, parameters, context, executemany) -> None:
        if statement.lstrip().upper().startswith("CREATE TABLE"):
            creates.append(statement)

    real_table_exists = store_module.table_exists
    barrier = threading.Barrier(_N_THREADS, timeout=30)
    seen = threading.local()

    def racing_table_exists(eng, name: str) -> bool:
        answer = real_table_exists(eng, name)
        if not getattr(seen, "first_done", False):
            seen.first_done = True
            barrier.wait()
        return answer

    monkeypatch.setattr(store_module, "table_exists", racing_table_exists)
    store = DataStore(engine)
    errors: list[BaseException] = []

    def _save(i: int) -> None:
        try:
            store.save(_T_PROBE, pd.DataFrame({"ticker": [f"TK{i}"], "value": [float(i)]}))
        except BaseException as exc:  # noqa: BLE001 -- collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_save, args=(i,)) for i in range(_N_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    monkeypatch.setattr(store_module, "table_exists", real_table_exists)
    stored = store.load(_T_PROBE)
    engine.dispose()

    assert errors == []
    assert sorted(stored["ticker"]) == sorted(f"TK{i}" for i in range(_N_THREADS))
    assert len(creates) == 1, f"{len(creates)} CREATE TABLE statements ran for one cold table"

    print("\n=== SANITY CHECK: ensure_table cold-table lock ===")
    print(f"  {_N_THREADS} threads all saw the table missing at once; CREATE ran {len(creates)}x, errors: {len(errors)}, rows stored: {len(stored)}.")
    print("  -> The re-check under the lock lets exactly one writer create the table; every row lands.")
