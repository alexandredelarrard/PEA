"""Share basis of a replaced predecessor window (P31): inside the window the owner's share counts are multiplied and
its per-share figures divided by the cited exchange ratio times the canonical ticker's splits after the seam, so
price x shares is on the canonical basis; `sharesOutstandingPit` stays the owner's as-filed count. Known-truth
fixtures, `FakeStore`, no network, no database.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.fundamentals_sharadar import merge_history
from src.data_extract.utils.fundamentals_sharadar.field_map import load_field_map
from src.data_store.schema import Tables
from src.utils import cutover_continuity as cc
from tests.conftest import FakeStore

LOGGER = "test.predecessor_share_basis"
URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=&dateb=&owner=include&count=40"
REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs"
PLD_RATIO, DD_RATIO = 0.4464, 1.282
EXCHANGES = [
    {
        "ticker": "PLD",
        "predecessor_cik": "0000899881",
        "seam_date": "2011-06-03",
        "ratio": PLD_RATIO,
        "source": "0000950123-11-057596",
        "evidence": "fixture",
    },
    {
        "ticker": "DD",
        "predecessor_cik": "0000030554",
        "seam_date": "2017-08-31",
        "ratio": DD_RATIO,
        "source": "0001193125-17-274840",
        "evidence": "fixture",
    },
]


def _config(tmp_path: Path, exchanges: list[dict[str, Any]]) -> str:
    """A config dir holding the Sharadar and fundamentals maps of the repo and a manual with `exchanges` only."""
    for sub in ("sharadar", "fundamentals"):
        shutil.copytree(REPO_CONFIGS / sub, tmp_path / sub)
    (tmp_path / "sec").mkdir()
    (tmp_path / "sec" / "security_master_manual.json").write_text(json.dumps({"exchange_ratios": exchanges}), encoding="utf-8")
    return str(tmp_path)


def _lineage() -> pd.DataFrame:
    rows = [
        ("PLD", "0000899881", "1900-01-01", "2011-06-03"),
        ("PLD", "0001045609", "2011-06-03", None),
        ("DD", "0000030554", "1900-01-01", "2017-08-31"),
        ("DD", "0001666700", "2017-08-31", None),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": f"E-{t}",
                "canonical_ticker": t,
                "cik": c,
                "role": "cik_window",
                "symbol": "",
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(e) if e else pd.NaT,
                "sources": "register",
            }
            for t, c, f, e in rows
        ]
    )


def _vendor_tickers() -> pd.DataFrame:
    rows = [
        ("PLD1", "0000899881", "2011-03-31"),
        ("PLD", "0001045609", "2026-06-30"),
        ("DD1", "0000030554", "2017-06-30"),
        ("DD", "0001666700", "2026-06-30"),
    ]
    return pd.DataFrame(
        {"ticker": [r[0] for r in rows], "secfilings": [URL.format(r[1]) for r in rows], "lastquarter": pd.to_datetime([r[2] for r in rows])}
    )


def _arq(ticker: str, labels: list[str], **values: float) -> pd.DataFrame:
    """Full vendor ARQ rows (every mapped column 1.0), filed 40 days after each quarter end, with `values` set."""
    field_map = load_field_map(str(REPO_CONFIGS))
    vendor_columns = {str(s.source) for s in field_map.direct.values()} | set(field_map.extras)
    vendor_columns |= set(cc.SHARE_COUNT_COLUMNS) | set(cc.PER_SHARE_COLUMNS) | {"marketcap"}
    ends = [pd.Period(q, freq="Q").end_time.normalize() for q in labels]
    frame = pd.DataFrame({column: 1.0 for column in sorted(vendor_columns)}, index=range(len(labels)))
    frame = frame.assign(
        ticker=ticker,
        dimension="ARQ",
        calendardate=[e.date() for e in ends],
        reportperiod=[e.date() for e in ends],
        date=[(e + pd.Timedelta(days=40)).date() for e in ends],
        fiscalperiod=[f"{q[:4]}-Q{q[-1]}" for q in labels],
    )
    for column, value in values.items():
        frame[column] = value
    return frame


def _store(vendor: pd.DataFrame, actions: list[tuple[str, str, float]], yf: list[tuple[str, str, float]]) -> FakeStore:
    return FakeStore(
        {
            Tables.entity_lineage: _lineage(),
            Tables.sharadar_tickers: _vendor_tickers(),
            Tables.sharadar_fundamentals: vendor,
            Tables.sharadar_actions: pd.DataFrame(
                [{"ticker": t, "date": pd.Timestamp(d), "action": "split", "value": v} for t, d, v in actions],
                columns=["ticker", "date", "action", "value"],
            ),
            Tables.prices_splits: pd.DataFrame(
                [{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in yf], columns=["ticker", "date", "ratio"]
            ),
            Tables.fundamentals_employees: pd.DataFrame(columns=["ticker", "as_of", "employees"]),
            # one NULL SEC row per ticker, so the backward join lands the SEC-owned columns (all NULL here)
            Tables.fundamentals_history_sec: pd.DataFrame(
                {"ticker": ["PLD", "DD"], "as_of": pd.Timestamp("1990-01-01")}
                | {c: float("nan") for c in load_field_map(str(REPO_CONFIGS)).sec_owned if c != "employees"}
            ),
        }
    )


def _merged(
    store: FakeStore, ticker: str, config_dir: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(the ARQ frame handed to build_frame, the merged frame build_frame returns from it)`, both keyed by quarter."""
    seen: dict[str, Any] = {}
    real = merge_history.build_frame

    def capture(*args: Any, **kwargs: Any) -> pd.DataFrame:
        seen["arq"] = args[0]
        seen["out"] = real(*args, **kwargs)
        return seen["out"].iloc[:0]

    monkeypatch.setattr(merge_history, "build_frame", capture)
    context = SimpleNamespace(store=store, log=logging.getLogger(LOGGER))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        merge_history.build_merged_history(context, [ticker], config_dir=config_dir)
    arq = (
        seen["arq"].assign(quarter=lambda f: [str(pd.Period(pd.Timestamp(d), freq="Q")) for d in f["reportperiod"]]).set_index("quarter").sort_index()
    )
    out = seen["out"].assign(quarter=lambda f: [str(pd.Period(pd.Timestamp(d), freq="Q")) for d in f["fiscal_end"]]).set_index("quarter").sort_index()
    return arq, out


def test_convert_share_basis_counts_times_per_share_divided_totals_untouched() -> None:
    owner = pd.DataFrame(
        {
            "sharesbas": [572e6],
            "shareswa": [570e6],
            "shareswadil": [575e6],
            "eps": [0.05],
            "dps": [0.1125],
            "price": [16.0],
            "marketcap": [9.152e9],
            "revenue": [238.8e6],
        }
    )
    got = cc.convert_share_basis(owner, PLD_RATIO).iloc[0]
    print("\n=== SANITY CHECK: share-basis conversion (old ProLogis -> Prologis at 0.4464) ===")
    print(got.to_string())
    assert got["sharesbas"] == pytest.approx(572e6 * PLD_RATIO) and got["shareswadil"] == pytest.approx(575e6 * PLD_RATIO)
    assert (
        got["price"] == pytest.approx(16.0 / PLD_RATIO)
        and got["dps"] == pytest.approx(0.1125 / PLD_RATIO)
        and got["eps"] == pytest.approx(0.05 / PLD_RATIO)
    )
    assert got["marketcap"] == 9.152e9 and got["revenue"] == 238.8e6
    assert got["price"] * got["sharesbas"] == pytest.approx(16.0 * 572e6)
    nulled = cc.convert_share_basis(owner, None).iloc[0]
    assert nulled[["sharesbas", "shareswa", "shareswadil", "eps", "dps", "price"]].isna().all() and nulled["revenue"] == 238.8e6
    print("  OK: counts x ratio, per-share / ratio, price x shares and totals unchanged; no ratio nulls the share block only.")


def test_reverse_merger_window_is_converted_and_pit_stays_as_filed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """PLD: AMB (the legal acquirer) owns the ticker's price history, so old ProLogis rows go to AMB shares at 0.4464."""
    canonical = pd.concat(
        [_arq("PLD", ["2010Q3", "2010Q4"], sharesbas=168e6, price=35.0), _arq("PLD", ["2011Q3"], sharesbas=458e6, price=29.7)], ignore_index=True
    )
    owner = _arq(
        "PLD1", ["2010Q3", "2010Q4", "2011Q1"], sharesbas=572e6, shareswa=570e6, shareswadil=570e6, price=16.0, dps=0.1125, marketcap=9.152e9
    )
    store = _store(pd.concat([canonical, owner], ignore_index=True), actions=[], yf=[])
    arq, out = _merged(store, "PLD", _config(tmp_path, EXCHANGES), monkeypatch, caplog)
    print("\n=== SANITY CHECK: reverse merger (PLD) ===")
    print(arq[["ticker", "sharesbas", "price", "dps", "marketcap"]].to_string())
    print(out[["sharesOutstanding", "sharesOutstandingPit"]].to_string())
    inside = ["2010Q3", "2010Q4", "2011Q1"]
    assert arq.loc[inside, "sharesbas"].tolist() == pytest.approx([572e6 * PLD_RATIO] * 3)
    assert arq.loc[inside, "price"].tolist() == pytest.approx([16.0 / PLD_RATIO] * 3)
    assert arq.loc[inside, "marketcap"].eq(9.152e9).all()
    assert arq.loc["2011Q3", "sharesbas"] == 458e6 and arq.loc["2011Q3", "price"] == 29.7
    assert out.loc[inside, "sharesOutstanding"].tolist() == pytest.approx([572e6 * PLD_RATIO] * 3)
    assert out.loc[inside, "sharesOutstandingPit"].tolist() == pytest.approx([572e6] * 3)
    assert out.loc["2011Q3", "sharesOutstandingPit"] == pytest.approx(458e6)
    assert "0.4464" in " ".join(r.getMessage() for r in caplog.records)
    print(
        "  OK: inside the window shares x 0.4464 and price / 0.4464 (market cap kept); the PIT count is old ProLogis' own; after the seam untouched."
    )


def test_forward_merger_takes_the_canonical_later_splits_and_the_owner_own_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """DD: old DuPont rows go to today's DD shares at 1.282 x the 1:3 reverse split after the seam. The acquirer's
    pre-seam 3:1 split (2000) no longer de-adjusts DuPont rows; DuPont's own 2:1 (1997) does."""
    canonical = pd.concat(
        [
            _arq("DD", ["1997Q1", "1999Q4", "2017Q1", "2017Q2"], sharesbas=135e6, price=595.0),
            _arq("DD", ["2018Q1"], sharesbas=257e6, price=580.0),
        ],
        ignore_index=True,
    )
    owner = pd.concat(
        [
            _arq("DD1", ["1997Q1"], sharesbas=1130e6, price=60.0),
            _arq("DD1", ["1999Q4"], sharesbas=1040e6, price=60.0),
            _arq("DD1", ["2017Q1", "2017Q2"], sharesbas=866e6, price=82.0, dps=0.38),
        ],
        ignore_index=True,
    )
    store = _store(
        pd.concat([canonical, owner], ignore_index=True),
        actions=[("DD", "2000-06-19", 3.0), ("DD", "2019-06-03", 1 / 3), ("DD1", "1997-06-13", 2.0)],
        yf=[("DD", "2000-06-19", 3.0), ("DD", "2019-06-03", 1 / 3)],
    )
    arq, out = _merged(store, "DD", _config(tmp_path, EXCHANGES), monkeypatch, caplog)
    factor = DD_RATIO / 3
    print("\n=== SANITY CHECK: forward merger with later canonical splits (DD) ===")
    print(arq[["ticker", "sharesbas", "price", "dps"]].to_string())
    print(out[["sharesOutstanding", "sharesOutstandingPit", "dividendsPerShare"]].to_string())
    assert arq.loc[["2017Q1", "2017Q2"], "sharesbas"].tolist() == pytest.approx([866e6 * factor] * 2)
    assert arq.loc[["2017Q1", "2017Q2"], "price"].tolist() == pytest.approx([82.0 / factor] * 2)
    assert arq.loc[["2017Q1", "2017Q2"], "dps"].tolist() == pytest.approx([0.38 / factor] * 2)
    assert arq.loc["2018Q1", "sharesbas"] == 257e6
    assert out.loc["2017Q2", "sharesOutstandingPit"] == pytest.approx(866e6)
    assert out.loc["1999Q4", "sharesOutstandingPit"] == pytest.approx(1040e6)
    assert out.loc["1997Q1", "sharesOutstandingPit"] == pytest.approx(1130e6 / 2)
    assert out.loc["2018Q1", "sharesOutstandingPit"] == pytest.approx(257e6 * 3)
    print("  OK: factor 1.282/3 inside the window; PIT = DuPont's as-filed count (its own 1997 split, not Dow's 2000 one); after the seam unchanged.")


def test_a_window_without_a_cited_ratio_nulls_the_share_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    canonical = pd.concat([_arq("PLD", ["2010Q4"], sharesbas=168e6), _arq("PLD", ["2011Q3"], sharesbas=458e6)], ignore_index=True)
    owner = _arq("PLD1", ["2010Q4"], sharesbas=572e6, revenue=260e6)
    store = _store(pd.concat([canonical, owner], ignore_index=True), actions=[], yf=[])
    arq, _ = _merged(store, "PLD", _config(tmp_path, []), monkeypatch, caplog)
    assert np.isnan(arq.loc["2010Q4", "sharesbas"]) and np.isnan(arq.loc["2010Q4", "price"]) and arq.loc["2010Q4", "revenue"] == 260e6
    assert arq.loc["2011Q3", "sharesbas"] == 458e6
    assert any(r.levelno == logging.WARNING and "PLD1" in r.getMessage() and "exchange ratio" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY CHECK: no cited exchange ratio ===")
    print("  OK: the owner's share block is NULL inside the window (never a mixed basis), other columns kept, one WARNING.")


def test_the_shipped_config_cites_one_exchange_ratio_per_replaced_window() -> None:
    exchanges = {x.ticker: x for x in sm.load_security_manual(str(REPO_CONFIGS)).exchanges}
    print("\n=== SANITY CHECK: shipped exchange ratios ===")
    for x in exchanges.values():
        print(f"  {x.ticker} {x.predecessor_cik} {x.seam_date.date()} x{x.ratio}")
    assert set(exchanges) == {"LIN", "EVRG", "BKR", "STE"}, "PLD and DD follow their traded security: no replacement, no ratio"
    assert all(exchanges[t].ratio == 1 for t in ("LIN", "EVRG", "BKR", "STE"))
    with pytest.raises(sm.SecurityManualError):
        sm.parse_security_manual({"exchange_ratios": [{"ticker": "X", "predecessor_cik": "1", "seam_date": "2020-01-01", "ratio": 2.0}]})
    print("  OK: four cited entries, each ratio 1 (PLD/DD removed with the traded-security view); an entry without a source is refused.")
