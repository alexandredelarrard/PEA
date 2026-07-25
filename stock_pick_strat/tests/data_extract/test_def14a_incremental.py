"""DEF 14A LLM: the incremental up-to-date check must be per-TICKER (not date+count),
and the new board-technology-maturity fields must flatten into the output row.
"""
from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.data_store.store import DataStore
from src.data_extract.utils.common.sec_utils import save_extract_meta, today_iso
from src.data_extract.utils.structure.fetch_def14a_llm import _is_up_to_date, _flatten
from src.data_extract.utils.structure.def14a_schema import Def14AExtract, GovernanceProfile


def _ctx(tmp_path: Path, tickers: list[str], write_meta_today: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = DataStore(create_engine(f"sqlite:///{tmp_path/'d.db'}"))
    ds.save("def14a_llm", pd.DataFrame([
        {"ticker": t, "accession_number": f"acc-{t}", "as_of": "2024-04-01"} for t in tickers
    ]))
    path = tmp_path / "def14a_llm.parquet"
    if write_meta_today:
        save_extract_meta(path, today_iso(), len(tickers), len(tickers))
    return types.SimpleNamespace(store=ds, paths={"DEF14A_LLM_PATH": path})


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
    print("  all requested present -> skip; a missing ticker (NVDA) -> NOT skipped "
          "(re-processes it); no meta -> re-scan. date+count bug fixed. Validated.")


def test_gap_fill_lists_full_window_and_skips_present(tmp_path, monkeypatch):
    """Gap-filling: the FULL window is listed (no `since` cutoff) and the LLM runs ONLY on filings
    whose accession is not already in the table — so a HOLE in the middle (2023 here) is filled while
    the present years (2022, 2024) are skipped. Uses an in-memory SQLite store (no Postgres)."""
    import logging
    from omegaconf import OmegaConf
    from src.data_extract.utils.structure import fetch_def14a_llm as mod

    ds = DataStore(create_engine(f"sqlite:///{tmp_path/'d.db'}"))
    ds.save("def14a_llm", pd.DataFrame([                       # 2022 + 2024 present; 2023 is a HOLE
        {"ticker": "ZZ", "accession_number": "a2022", "as_of": "2022-04-01"},
        {"ticker": "ZZ", "accession_number": "a2024", "as_of": "2024-04-01"},
    ]))
    ctx = types.SimpleNamespace(store=ds, log=logging.getLogger("t"),
                                paths={"DEF14A_LLM_PATH": tmp_path / "m.parquet"},
                                config=OmegaConf.create({"data_extract": {"years_history": 15}}))

    listed_since, extracted = [], []

    def _fake_list(cik, forms, years, company="", since=None):
        listed_since.append(since)                            # must be None now (full window)
        return pd.DataFrame([
            {"accession_number": a, "doc_url": f"http://x/{a}", "filing_date": pd.Timestamp(d),
             "period_of_report": "2000-12-31", "form": "DEF 14A"}
            for a, d in [("a2022", "2022-04-01"), ("a2023", "2023-04-01"),
                         ("a2024", "2024-04-01"), ("a2025", "2025-04-01")]])

    def _fake_process(ticker, f, extractor):
        extracted.append(f["accession_number"])
        return {"ticker": ticker, "accession_number": f["accession_number"],
                "as_of": f["filing_date"], "def14a_json": "{}"}

    def _fake_save(context, rows):                            # string as_of (SQLite can't bind Timestamp)
        df = pd.DataFrame(rows)
        df["as_of"] = pd.to_datetime(df["as_of"]).dt.strftime("%Y-%m-%d")
        return context.store.save("def14a_llm", df)

    class _FakeLLM:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(mod, "list_filings", _fake_list)
    monkeypatch.setattr(mod, "_process_filing", _fake_process)
    monkeypatch.setattr(mod, "_save_ticker_rows", _fake_save)
    monkeypatch.setattr(mod, "LLMExtractor", _FakeLLM)
    monkeypatch.setattr(mod, "load_cik_mapping", lambda _c: pd.DataFrame(
        {"ticker": ["ZZ"], "cik": ["0000000001"], "company_name": ["Z"]}))
    monkeypatch.setattr(mod, "_is_up_to_date", lambda _c, _n: False)

    mod.fetch_def14a_llm(ctx, tickers=["ZZ"])

    assert listed_since == [None], "must list the FULL window (no since cutoff) to find gaps"
    assert set(extracted) == {"a2023", "a2025"}, f"only missing filings should hit the LLM: {extracted}"
    accs = set(ds.load("def14a_llm").query("ticker == 'ZZ'")["accession_number"])
    assert accs == {"a2022", "a2023", "a2024", "a2025"}

    print("\n=== SANITY: DEF 14A gap-filling incremental ===")
    print(f"  had 2022+2024, listed full window (since={listed_since[0]}) -> LLM ran ONLY on the "
          f"missing {sorted(set(extracted))} (2023 hole + new 2025); 2 present skipped. Validated.")


def test_flatten_surfaces_board_technology_maturity():
    extract = Def14AExtract(
        company_name="ACME", fiscal_year=2024,
        governance=GovernanceProfile(board_size=10, n_technology_directors=3,
                                     technology_committee=True),
    )
    filing = pd.Series({"filing_date": pd.Timestamp("2024-04-01"),
                        "period_of_report": "2023-12-31", "accession_number": "a1"})
    row = _flatten("ACME", filing, extract)
    assert row["n_technology_directors"] == 3
    assert abs(row["pct_technology_directors"] - 0.30) < 1e-9      # 3 / 10
    assert row["technology_committee"] == 1.0                       # bool -> numeric flag
    # absent -> null (not a false 0)
    empty = _flatten("X", filing, Def14AExtract(governance=GovernanceProfile(board_size=8)))
    assert empty["n_technology_directors"] is None
    assert empty["pct_technology_directors"] is None
    assert empty["technology_committee"] is None

    print("\n=== SANITY: board technology-maturity fields flatten ===")
    print("  n_technology_directors=3, pct=0.30 (3/10 board), technology_committee=1.0; "
          "absent -> null. Validated.")
