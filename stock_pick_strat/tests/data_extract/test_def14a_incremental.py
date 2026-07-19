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
