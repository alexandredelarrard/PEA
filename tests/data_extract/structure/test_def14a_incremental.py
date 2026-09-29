"""DEF 14A LLM: the incremental up-to-date check must be per-TICKER (not date+count),
and the new board-technology-maturity fields must flatten into the output row.
"""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from sqlalchemy import create_engine

from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract as _Def14AExtract
from src.data_extract.utils.schemas.def14a_schema import GovernanceProfile as _GovernanceProfile
from src.data_extract.utils.structure.def14a.fetch import _is_up_to_date, _subject_is_accepted
from src.data_extract.utils.structure.def14a.flatten import _flatten
from src.data_store.store import DataStore
from tests.data_extract.fake_context import extract_config

Def14AExtract: Any = _Def14AExtract
GovernanceProfile: Any = _GovernanceProfile


def _ctx(tmp_path: Path, tickers: list[str], write_meta_today: bool = True) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    ds.save("def14a_llm", pd.DataFrame([{"ticker": t, "accession_number": f"acc-{t}", "as_of": "2024-04-01"} for t in tickers]))
    ctx: Any = types.SimpleNamespace(store=ds, paths={"DATA_STORE": tmp_path}, config=extract_config())
    if write_meta_today:
        record_run(ctx, "def14a_llm", len(tickers), 0, is_full_rescan=True)
    return ctx


def test_up_to_date_is_per_ticker_not_date_count(tmp_path):
    ctx = _ctx(tmp_path, ["AAPL", "MSFT"])
    # every requested ticker present + built today -> up to date (skip)
    assert _is_up_to_date(ctx, ["AAPL", "MSFT"]) is True
    # a MISSING ticker must NOT be skipped, even though it was "built today" for 2
    # (this is the '~15 tickers then it stops' bug -> now fixed)
    assert _is_up_to_date(ctx, ["AAPL", "MSFT", "NVDA"]) is False

    # no meta today -> not up to date (re-scan, picks up new annual proxies)
    ctx2 = _ctx(tmp_path / "b", ["AAPL", "MSFT"], write_meta_today=False)
    assert _is_up_to_date(ctx2, ["AAPL", "MSFT"]) is False

    print("\n=== SANITY: DEF 14A incremental is per-ticker ===")
    print(
        "  all requested present -> skip; a missing ticker (NVDA) -> NOT skipped "
        "(re-processes it); no meta -> re-scan. date+count bug fixed. Validated."
    )


def test_subject_guard_rejects_only_known_disjoint_subjects(monkeypatch, caplog):
    """A dissident filer is valid when the proxy subject is the intended issuer."""
    import logging

    from src.data_extract.utils.structure.def14a import fetch as mod

    filing = pd.Series(
        {
            "cik": "0002041610",
            "company_name": "Paramount Skydance Corp",
            "form": "DEFC14A",
            "filing_date": pd.Timestamp("2026-02-17"),
            "accession_number": "0001104659-26-016573",
        }
    )
    identity_calls: list[bool] = []
    context = SimpleNamespace(
        log=logging.getLogger("test.def14a.subject"),
        ensure_edgar_identity=lambda: identity_calls.append(True),
    )
    accepted = frozenset({"0000813828", "0002041610"})
    caplog.set_level(logging.INFO, logger="test.def14a.subject")

    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0001437107"}))
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is False
    assert all(value in caplog.text for value in ("PSKY", filing["accession_number"], filing["cik"], "0001437107", "subject_cik_disjoint"))

    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0000813828"}))
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is True

    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset())
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is True

    def _unreadable(_):
        raise ValueError("no SGML header")

    monkeypatch.setattr(mod, "_filing_subject_ciks", _unreadable)
    assert _subject_is_accepted(context, "PSKY", filing, accepted) is True
    assert len(identity_calls) == 4

    print("\n=== SANITY: DEF 14A subject guard ===")
    print("  WBD subject rejected; intended-issuer dissident and unknown headers retained")
    print("  OK: only known disjoint subjects are blocked before paid extraction")


def test_disjoint_subject_never_becomes_an_llm_task(tmp_path, monkeypatch):
    import logging

    from src.data_extract.utils.structure.def14a import fetch as mod

    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    context: Any = SimpleNamespace(
        store=store,
        log=logging.getLogger("test.def14a.subject.loop"),
        paths={"DATA_STORE": tmp_path},
        config=extract_config(data_extract={"years_history": 15}),
        ensure_edgar_identity=lambda: None,
    )
    filing = pd.DataFrame(
        [
            {
                "cik": "0002041610",
                "company_name": "Paramount Skydance Corp",
                "form": "DEFC14A",
                "filing_date": pd.Timestamp("2026-02-17"),
                "period_of_report": "2026-02-17",
                "accession_number": "0001104659-26-016573",
                "doc_url": "unused",
                "txt_url": "unused",
            }
        ]
    )
    llm_tasks: list[object] = []
    payload_calls: list[str] = []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def run_extraction(self, tasks, **kwargs):
            del kwargs
            llm_tasks.extend(tasks)
            return []

    monkeypatch.setattr(mod, "LLMExtractor", FakeLLM)
    monkeypatch.setattr(mod, "_is_up_to_date", lambda *_: False)
    monkeypatch.setattr(
        mod, "load_cik_mapping", lambda *_: pd.DataFrame([{"ticker": "PSKY", "cik": "0002041610", "company_name": "Paramount Skydance Corp"}])
    )
    monkeypatch.setattr(mod, "_list_across_registrants", lambda *_: filing)
    monkeypatch.setattr(mod, "_filing_subject_ciks", lambda _: frozenset({"0001437107"}))
    monkeypatch.setattr(mod, "_payload_for", lambda *args: payload_calls.append(str(args[2]["accession_number"])))

    mod.fetch_def14a_llm(context, context.config, ["PSKY"], model="gpt-5-mini")

    assert payload_calls == [] and llm_tasks == []
    print("\n=== SANITY: wrong-subject filing stops before the LLM ===")
    print("  WBD accession produced zero payload fetches and zero LLM tasks")
    print("  OK: rejection precedes document carving and paid extraction")


def test_gap_fill_lists_full_window_and_skips_present(tmp_path, monkeypatch):
    """Gap-filling on a FRESH manifest (no recorded run yet -> full rescan, per
    `run_manifest.manifest_window`): the FULL window is listed (no `since` cutoff)
    and the LLM runs ONLY on filings whose accession is not already in the table —
    so a HOLE in the middle (2023 here) is filled while the present years (2022,
    2024) are skipped. Uses an in-memory SQLite store (no Postgres)."""
    import logging

    from src.data_extract.utils.structure.def14a import fetch as mod

    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    ds.save(
        "def14a_llm",
        pd.DataFrame(
            [  # 2022 + 2024 present; 2023 is a HOLE
                {"ticker": "ZZ", "accession_number": "a2022", "as_of": "2022-04-01"},
                {"ticker": "ZZ", "accession_number": "a2024", "as_of": "2024-04-01"},
            ]
        ),
    )
    ctx: Any = types.SimpleNamespace(
        store=ds, log=logging.getLogger("t"), paths={"DATA_STORE": tmp_path}, config=extract_config(data_extract={"years_history": 15})
    )

    listed_since, extracted = [], []

    def _fake_list(context, cik, forms, years, company_name="", since=None, cache_dir=None):  # mirrors edgar_fillings.list_filings EXACTLY
        # -- a stale stub binds `since` positionally
        listed_since.append(since)  # must be None now (full window)
        return pd.DataFrame(
            [
                {
                    "accession_number": a,
                    "doc_url": f"http://x/{a}",
                    "filing_date": pd.Timestamp(d),
                    "period_of_report": "2000-12-31",
                    "form": "DEF 14A",
                }
                for a, d in [("a2022", "2022-04-01"), ("a2023", "2023-04-01"), ("a2024", "2024-04-01"), ("a2025", "2025-04-01")]
            ]
        )

    class _FakeLLM:
        """Stands in for the whole extracter: records which accessions became LLM tasks,
        then writes the rows the real `flatten` would have produced.

        The frame is written with a STRING `as_of` because this test runs on SQLite, whose
        driver cannot bind a pandas Timestamp.
        """

        def __init__(self, context, config, action=None, threads=None, methodes=None):
            self._context = context

        def run_extraction(self, tasks, flatten=None, group_key=None):
            tasks = list(tasks)
            rows = []
            for t in tasks:
                f = t.meta["filing"]
                extracted.append(f["accession_number"])
                rows.append({"ticker": t.meta["ticker"], "accession_number": f["accession_number"], "as_of": f["filing_date"], "def14a_json": "{}"})
            if rows:
                df = pd.DataFrame(rows)
                df["as_of"] = pd.to_datetime(df["as_of"]).dt.strftime("%Y-%m-%d")
                self._context.store.save("def14a_llm", df)
            return [SimpleNamespace(ok=True, task=t, parsed=object(), error=None) for t in tasks]

    monkeypatch.setattr(mod, "list_filings", _fake_list)
    monkeypatch.setattr(mod, "_payload_for", lambda context, ticker, f: "=== CARVED ===")
    monkeypatch.setattr(mod, "LLMExtractor", _FakeLLM)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda _c, _t=None: pd.DataFrame({"ticker": ["ZZ"], "cik": ["0000000001"], "company_name": ["Z"]}))
    monkeypatch.setattr(mod, "_is_up_to_date", lambda _c, _n: False)

    mod.fetch_def14a_llm(ctx, ctx.config, tickers=["ZZ"], model="gpt-5-mini")

    assert listed_since == [None], "must list the FULL window (no since cutoff) to find gaps"
    assert set(extracted) == {"a2023", "a2025"}, f"only missing filings should hit the LLM: {extracted}"
    stored = ds.load("def14a_llm")
    assert stored is not None
    accs = set(stored.query("ticker == 'ZZ'")["accession_number"])
    assert accs == {"a2022", "a2023", "a2024", "a2025"}

    print("\n=== SANITY: DEF 14A gap-filling incremental ===")
    print(
        f"  had 2022+2024, listed full window (since={listed_since[0]}) -> LLM ran ONLY on the "
        f"missing {sorted(set(extracted))} (2023 hole + new 2025); 2 present skipped. Validated."
    )


def test_manifest_narrows_since_on_routine_rerun(tmp_path, monkeypatch):
    """A ROUTINE re-run (manifest already has a recent run for this table, same
    ticker count, rescan not due) must list only from the manifest's last run date
    onward -- not the full `years_history` window -- per `run_manifest.manifest_window`.
    This is the narrow-window counterpart to the full-rescan case exercised by
    `test_gap_fill_lists_full_window_and_skips_present` above."""
    import logging

    from src.data_extract.utils.common.run_manifest import record_run
    from src.data_extract.utils.structure.def14a import fetch as mod

    ds = DataStore(create_engine(f"sqlite:///{tmp_path / 'd.db'}"))
    ds.save(
        "def14a_llm",
        pd.DataFrame(
            [
                {"ticker": "ZZ", "accession_number": "a2024", "as_of": "2024-04-01"},
            ]
        ),
    )
    ctx: Any = types.SimpleNamespace(
        store=ds, log=logging.getLogger("t"), paths={"DATA_STORE": tmp_path}, config=extract_config(data_extract={"years_history": 15})
    )
    # A prior run 10 days ago, one ticker -- same ticker count as this run, and well
    # inside the (default 30-day) self-heal window, so `manifest_window` must return
    # the narrow cutoff, not the full-rescan fallback.
    last_run = pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
    record_run(ctx, "def14a_llm", ticker_count=1, rows_added=1, is_full_rescan=True, run_date=last_run)

    listed_since = []

    def _fake_list(context, cik, forms, years, company_name="", since=None, cache_dir=None):  # mirrors edgar_fillings.list_filings EXACTLY
        # -- a stale stub binds `since` positionally
        listed_since.append(since)
        return pd.DataFrame(columns=["accession_number", "doc_url", "filing_date", "period_of_report", "form"])

    class _FakeLLM:
        """This ticker lists no filings, so the extracter is built and never used."""

        def __init__(self, context, config, action=None, threads=None, methodes=None):
            pass

        def run_extraction(self, tasks, flatten=None, group_key=None):
            return []

    monkeypatch.setattr(mod, "list_filings", _fake_list)
    monkeypatch.setattr(mod, "LLMExtractor", _FakeLLM)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda _c, _t=None: pd.DataFrame({"ticker": ["ZZ"], "cik": ["0000000001"], "company_name": ["Z"]}))
    monkeypatch.setattr(mod, "_is_up_to_date", lambda _c, _n: False)

    mod.fetch_def14a_llm(ctx, ctx.config, tickers=["ZZ"], model="gpt-5-mini")

    # list_filings' own `since` is STRICTLY AFTER the date passed, so the manifest's
    # last run date (inclusive) is passed as (last_run - 1 day).
    assert listed_since == [last_run - pd.Timedelta(days=1)], f"routine rerun must narrow to the manifest cutoff, got {listed_since}"

    print("\n=== SANITY: DEF 14A manifest narrows the window on a routine rerun ===")
    print(
        f"  prior run {last_run.date()}, same ticker count, rescan not due -> "
        f"listed since={listed_since[0]} (inclusive of the prior run date). Validated."
    )


def test_flatten_surfaces_the_auditor_block():
    """`n_technology_directors` / `technology_committee` were REMOVED -- they were an opinion,
    not an extraction (mean |delta| of 1.06 directors between consecutive filings of the same
    company, only 38.8% unchanged, and wrong by 7x on HUBB 2022 whose own matrix states
    "Cybersecurity and Technology 78%" of 9 directors).

    What replaced them is the auditor block, which is the opposite kind of field: the firm name
    is present in 98% of documents and was the WORST column in the retired edgar table at 2.05%
    fill."""
    extract = Def14AExtract(
        company_name="ACME",
        fiscal_year=2024,
        governance=GovernanceProfile(
            board_size=10,
            auditor_name="Ernst & Young LLP",
            auditor_since_year=1934,
            auditor_fees_usd=12_000_000.0,
            audit_fees_audit_usd=9_000_000.0,
            audit_fees_audit_related_usd=1_000_000.0,
            audit_fees_tax_usd=1_500_000.0,
            audit_fees_other_usd=500_000.0,
            auditor_fees_prior_usd=11_000_000.0,
        ),
    )
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"), "period_of_report": "2023-12-31", "accession_number": "a1"})
    row = _flatten("ACME", filing, extract)
    assert row["auditor_name"] == "Ernst & Young LLP"
    assert row["auditor_since_year"] == 1934
    assert row["auditor_fees"] == 12_000_000.0
    assert row["audit_fees_audit"] == 9_000_000.0
    assert row["audit_fees_tax"] == 1_500_000.0
    assert row["auditor_fees_prior"] == 11_000_000.0
    # the dropped fields must NOT come back
    for gone in ("n_technology_directors", "pct_technology_directors", "technology_committee"):
        assert gone not in row, f"{gone} reappeared in the flatten"
    # absent -> null (not a false 0)
    empty = _flatten("X", filing, Def14AExtract(governance=GovernanceProfile(board_size=8)))
    assert empty["auditor_name"] is None and empty["auditor_fees"] is None
    assert empty["audit_fees_audit"] is None

    print("\n=== SANITY: auditor block flattens; technology fields are gone ===")
    print("  auditor_name='Ernst & Young LLP', since=1934, fees=12,000,000 split 9.0M/1.0M/1.5M/0.5M, prior=11,000,000; absent -> null.")
    print("  n_technology_directors / pct_technology_directors / technology_committee are")
    print("  absent from the flatten -- they were an opinion, not an extraction. Validated.")
