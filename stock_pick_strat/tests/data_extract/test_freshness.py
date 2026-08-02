"""
Data-freshness / gap gate (src/data_extract/utils/common/freshness.py::check_data_freshness).

Verifies the cadence logic: a daily source must have ~day -1 (else STALE/RED), a quarterly/yearly
source is fine being weeks/months old (within its lag), a missing table is flagged, and the overall
`ok` flips to False as soon as one source is not as expected. The DB is faked with an in-memory
engine-like object so no Postgres is needed.
"""
from __future__ import annotations

import types

import pandas as pd

from src.constants.constants import DATA_FRESHNESS_MAX_AGE_DAYS, DATA_FRESHNESS_SOURCES
from src.data_extract.utils.common import freshness as fr


class _FakeConn:
    """Answers the three query shapes check_data_freshness issues, from an in-memory map
    {table: {col: latest_date}}. A table absent from the map = missing_table."""

    def __init__(self, data: dict[str, dict[str, pd.Timestamp | None]]):
        self._data = data

    def execute(self, clause, params=None):
        sql = str(clause)
        if "information_schema.tables" in sql:
            return _Res(1 if params["t"] in self._data else None)
        if "information_schema.columns" in sql:
            cols = self._data.get(params["t"], {})
            return _Res(1 if params["c"] in cols else None)
        # MAX("col") FROM "table"  -> pull table + col out of the SQL text
        table = sql.split('FROM "')[1].split('"')[0]
        col = sql.split('MAX("')[1].split('"')[0]
        return _Res(self._data.get(table, {}).get(col))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Res:
    def __init__(self, value):
        self._value = value

    def first(self):
        return (self._value,) if self._value is not None else None

    def scalar(self):
        return self._value


def _ctx(data: dict) -> types.SimpleNamespace:
    engine = types.SimpleNamespace(connect=lambda: _FakeConn(data))
    return types.SimpleNamespace(store=types.SimpleNamespace(engine=engine))


def test_freshness_flags_stale_daily_and_tolerates_lagged_quarterly():
    today = pd.Timestamp("2026-07-24")                    # a Friday
    # prices FRESH (yesterday), macro STALE (10 days -> > daily 4), short_interest FRESH,
    # a quarterly source normally-lagged (~80d, within 140) -> ok, a yearly source missing table.
    data = {
        "prices":               {"date": today - pd.Timedelta(days=1)},
        "macro":                {"date": today - pd.Timedelta(days=10)},
        "macro_asset_prices":   {"date": today - pd.Timedelta(days=1)},
        "short_interest":       {"date": today - pd.Timedelta(days=1)},
        "wiki_pageviews":       {"date": today - pd.Timedelta(days=2)},
        "google_trends":        {"date": today - pd.Timedelta(days=6)},
        "fails_to_deliver":     {"date": today - pd.Timedelta(days=16)},
        "notes_num":            {"filed": today - pd.Timedelta(days=16)},
        "notes_text":           {"filed": today - pd.Timedelta(days=16)},
        "insider_transactions": {"filing_date": today - pd.Timedelta(days=30)},
        "fundamentals_history": {"as_of": today - pd.Timedelta(days=80)},
        "fundamentals_facts":   {"filing_date": today - pd.Timedelta(days=80)},
        "earnings_surprises":   {"earnings_date": today - pd.Timedelta(days=80)},
        "sec13f_hr":            {"period": today - pd.Timedelta(days=120)},
        "pension_facts":        {"filed": today - pd.Timedelta(days=80)},
        "earnings_call_sections": {"as_of": today - pd.Timedelta(days=40)},
        # def14a_llm table intentionally ABSENT -> missing_table
    }
    report = fr.check_data_freshness(_ctx(data), today=today, track_new_fundamentals=False)

    assert report["ok"] is False                          # macro stale + def14a missing
    assert report["sources"]["prices"]["status"] == "ok"
    assert report["sources"]["prices"]["age_days"] == 1
    assert report["sources"]["macro"]["status"] == "stale"
    assert report["sources"]["fundamentals_history"]["status"] == "ok"      # 80d < 140 quarterly
    assert report["sources"]["fundamentals_facts"]["status"] == "ok"        # 80d < 140 quarterly
    assert report["sources"]["def14a_llm"]["status"] == "missing_table"
    assert set(report["stale"]) == {"macro", "def14a_llm"}

    print("\n=== SANITY CHECK: data-freshness / gap gate ===")
    print(f"  as_of {report['as_of']} overall_ok={report['ok']}")
    for label in ("prices", "macro", "fundamentals_history", "fundamentals_facts", "def14a_llm"):
        s = report["sources"][label]
        print(f"    {label:<22} cadence={s['cadence']:<9} latest={s['latest']} "
              f"age={s['age_days']} (<= {s['max_age_days']}) -> {s['status']}")
    print(f"  NOT up to date -> {report['stale']}")
    print("  CONCLUSION: day-1 enforced for daily (macro's 10d gap = RED), quarterly/yearly "
          "tolerate their filing lag, missing table flagged; overall_ok=False gates the DAG. Validated.")


def test_freshness_all_ok_when_current():
    today = pd.Timestamp("2026-07-24")
    data = {}
    for label, (table, col, cadence) in DATA_FRESHNESS_SOURCES.items():
        # every source exactly at the edge of its allowed age -> all ok, overall ok
        age = DATA_FRESHNESS_MAX_AGE_DAYS[cadence]
        data[table] = {col: today - pd.Timedelta(days=age)}
    report = fr.check_data_freshness(_ctx(data), today=today, track_new_fundamentals=False)
    assert report["ok"] is True and report["stale"] == []
    assert all(v["status"] == "ok" for v in report["sources"].values())
    print("\n=== SANITY CHECK: all sources current ===")
    print(f"  {len(report['sources'])} sources each at their max allowed age -> all ok, "
          f"overall_ok={report['ok']}. Validated.")


def _fund_ctx(rows, tmp_path):
    df = pd.DataFrame(rows, columns=["ticker", "as_of"])
    store = types.SimpleNamespace(load=lambda table, columns=None: df)
    return types.SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path})


def test_new_fundamentals_filings_delta(tmp_path):
    # run 1 — first ever run: establish the baseline, nothing reported as new
    ctx1 = _fund_ctx([("AAA", "2026-03-31"), ("BBB", "2026-03-31")], tmp_path)
    d1 = fr._new_fundamentals_filings(ctx1, fr.logger)
    assert d1["baseline"] is True and d1["new_count"] == 0 and d1["tracked"] == 2

    # run 2 — AAA reports Q2 (new earnings) and CCC appears; BBB unchanged
    ctx2 = _fund_ctx([("AAA", "2026-06-30"), ("BBB", "2026-03-31"), ("CCC", "2026-06-30")], tmp_path)
    d2 = fr._new_fundamentals_filings(ctx2, fr.logger)
    assert d2["baseline"] is False and d2["new_count"] == 2 and d2["tracked"] == 3
    assert set(d2["new"]) == {"AAA", "CCC"}
    assert d2["new"]["AAA"] == {"from": "2026-03-31", "to": "2026-06-30"}
    assert d2["new"]["CCC"] == {"from": None, "to": "2026-06-30"}

    # run 3 — nothing changed since run 2 -> no new filings
    d3 = fr._new_fundamentals_filings(ctx2, fr.logger)
    assert d3["new_count"] == 0 and d3["new"] == {}

    print("\n=== SANITY CHECK: new fundamentals filings since last run ===")
    print(f"  run1 baseline: tracked {d1['tracked']} tickers, 0 new")
    print(f"  run2 deltas: {[(tk, v['from'], v['to']) for tk, v in d2['new'].items()]}")
    print(f"  run3 (no change): {d3['new_count']} new")
    print("  CONCLUSION: snapshot-diff reports exactly the tickers whose latest earnings period "
          "advanced (or first appeared) since the previous run -> goes to XCom. Validated.")


if __name__ == "__main__":
    test_freshness_flags_stale_daily_and_tolerates_lagged_quarterly()
    test_freshness_all_ok_when_current()
    import tempfile
    from pathlib import Path
    test_new_fundamentals_filings_delta(Path(tempfile.mkdtemp()))
