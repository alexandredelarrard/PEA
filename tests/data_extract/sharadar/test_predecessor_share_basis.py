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
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar, merge_history
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


def _config(tmp_path: Path, exchanges: list[dict[str, Any]], overrides: list[dict[str, Any]] | None = None) -> str:
    """A config dir holding the Sharadar and fundamentals maps of the repo and a manual with `exchanges` (and `overrides`) only."""
    for sub in ("sharadar", "fundamentals"):
        shutil.copytree(REPO_CONFIGS / sub, tmp_path / sub)
    (tmp_path / "sec").mkdir()
    manual = {"exchange_ratios": exchanges, "vendor_series_overrides": overrides or []}
    (tmp_path / "sec" / "security_master_manual.json").write_text(json.dumps(manual), encoding="utf-8")
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
    assert set(exchanges) == {"LIN", "EVRG", "BKR", "STE", "JCI"}, "PLD and DD follow their traded security: no replacement, no ratio"
    assert all(exchanges[t].ratio == 1 for t in ("LIN", "EVRG", "BKR", "STE"))
    assert (exchanges["JCI"].predecessor_cik, exchanges["JCI"].seam_date, exchanges["JCI"].ratio) == ("0000833444", pd.Timestamp("2016-09-02"), 0.955)
    with pytest.raises(sm.SecurityManualError):
        sm.parse_security_manual({"exchange_ratios": [{"ticker": "X", "predecessor_cik": "1", "seam_date": "2020-01-01", "ratio": 2.0}]})
    print("  OK: five cited entries: LIN/EVRG/BKR/STE ratio 1, JCI <- Tyco 0.955 (the consolidation split_events drops); PLD/DD have none;")
    print("  an entry without a source is refused.")


# --------------------------------------------------------------------------- JCI <- TYC (vendor_series_overrides)

JCI_SEAM = pd.Timestamp("2016-09-02")
SPIN_2012 = 2.011667672500503
#: Yahoo's JCI split events (prices_splits, live 2026-10-07): Tyco's own splits, the 2012 ADT/Pentair spin and the 2016 consolidation.
JCI_YF = [
    ("JCI", "1995-11-15", 2.0),
    ("JCI", "1997-10-23", 2.0),
    ("JCI", "1999-10-22", 2.0),
    ("JCI", "2007-07-02", 0.25),
    ("JCI", "2012-10-01", SPIN_2012),
    ("JCI", "2016-09-06", 0.955),
]
#: sharadar_actions splits and spinoffs (live 2026-10-07): old JCI's own splits under JCI, Tyco's under TYC.
JCI_ACTIONS = [
    ("JCI", "2004-01-05", "split", 2.0),
    ("JCI", "2007-10-03", "split", 3.0),
    ("JCI", "2016-10-31", "spinoff", 0.1),
    ("TYC", "1999-10-22", "split", 2.0),
    ("TYC", "2007-07-02", "spinoff", 1.0),
    ("TYC", "2007-07-02", "split", 0.25),
    ("TYC", "2012-10-01", "spinoff", 0.5),
    ("TYC", "2012-10-01", "spinoff", 0.23994),
]
#: Real Sharadar TYC ARQ rows: (date, reportperiod, calendardate, sharesbas, shareswa, price, marketcap, dps, JCI close_split on `date`).
TYC_ROWS = [
    ("1997-02-13", "1996-12-31", "1996-12-31", 156679751, 157445000, 59.00, 9244105309, 0.050, 30.710890),
    ("2006-05-09", "2006-03-31", "2006-03-31", 509150440, 504750000, 111.60, 56821189048, 0.400, 58.090427),
    ("2011-01-27", "2010-12-24", "2010-12-31", 473753233, 488000000, 44.74, 21195719644, 0.210, 23.288223),
    ("2012-07-31", "2012-06-29", "2012-06-30", 459875217, 463000000, 54.94, 25265544422, 0.250, 28.597565),
    ("2012-11-16", "2012-09-28", "2012-09-30", 465717368, 463000000, 26.77, 12467253941, 0.150, 28.031414),
    ("2016-07-29", "2016-06-24", "2016-06-30", 426224367, 426000000, 45.57, 19423044404, 0.205, 47.717278),
]
JCI_OVERRIDE = {
    "ticker": "JCI",
    "vendor_ticker": "TYC",
    "cik": "0000833444",
    "valid_from": None,
    "valid_to": "2016-09-02",
    "source": "0001104659-16-143068",
    "evidence": "fixture",
}
JCI_EXCHANGE = {"ticker": "JCI", "predecessor_cik": "0000833444", "seam_date": "2016-09-02", "ratio": 0.955, "source": "0001104659-16-143068"}


def _tyc() -> pd.DataFrame:
    rows = []
    for date, period, calendar, sharesbas, shareswa, price, marketcap, dps, _ in TYC_ROWS:
        row = _arq("TYC", [str(pd.Period(calendar, freq="Q"))], sharesbas=sharesbas, shareswa=shareswa, price=price, marketcap=marketcap, dps=dps)
        rows.append(row.assign(date=pd.Timestamp(date).date(), reportperiod=pd.Timestamp(period).date(), calendardate=pd.Timestamp(calendar).date()))
    return pd.concat(rows, ignore_index=True)


def _jci_store() -> FakeStore:
    """Old JCI's own rows inside the window (another company's shares), one post-seam JCI row, Tyco's real rows under TYC."""
    canonical = pd.concat(
        [_arq("JCI", ["2006Q1", "2010Q4", "2012Q2"], sharesbas=590e6, price=30.0), _arq("JCI", ["2016Q4"], sharesbas=935e6, price=41.0)],
        ignore_index=True,
    )
    return FakeStore(
        {
            Tables.entity_lineage: _lineage(),
            Tables.sharadar_tickers: _vendor_tickers(),
            Tables.sharadar_fundamentals: pd.concat([canonical, _tyc()], ignore_index=True),
            Tables.sharadar_actions: pd.DataFrame(
                [{"ticker": t, "date": pd.Timestamp(d), "action": a, "value": v} for t, d, a, v in JCI_ACTIONS],
                columns=["ticker", "date", "action", "value"],
            ),
            Tables.prices_splits: pd.DataFrame(
                [{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in JCI_YF], columns=["ticker", "date", "ratio"]
            ),
            Tables.fundamentals_employees: pd.DataFrame(columns=["ticker", "as_of", "employees"]),
            Tables.fundamentals_history_sec: pd.DataFrame(
                {"ticker": ["JCI"], "as_of": pd.Timestamp("1990-01-01")}
                | {c: float("nan") for c in load_field_map(str(REPO_CONFIGS)).sec_owned if c != "employees"}
            ),
        }
    )


def test_a_vendor_series_override_declares_jci_from_tyc(tmp_path: Path) -> None:
    manual = sm.parse_security_manual({"vendor_series_overrides": [JCI_OVERRIDE]})
    expected = cc.PredecessorSeries("JCI", "TYC", "0000833444", None, JCI_SEAM)
    shipped = sm.load_security_manual(str(REPO_CONFIGS)).vendor_series
    context = SimpleNamespace(store=_jci_store(), log=logging.getLogger(LOGGER), config_dir=_config(tmp_path, [JCI_EXCHANGE], [JCI_OVERRIDE]))
    print("\n=== SANITY CHECK: vendor_series_overrides ===")
    print(f"  parsed {manual.vendor_series}; shipped {shipped}")
    assert manual.vendor_series == (expected,) and shipped == (expected,)
    assert fetch_sharadar.load_predecessor_series(context, ["JCI", "AAPL"]) == (expected,)
    assert fetch_sharadar.predecessor_vendor_tickers(context, ["JCI"]) == ["TYC"]
    assert fetch_sharadar.predecessor_vendor_tickers(context, ["AAPL"]) == []
    with pytest.raises(sm.SecurityManualError):
        sm.parse_security_manual({"vendor_series_overrides": [{**JCI_OVERRIDE, "source": ""}]})
    print("  OK: JCI -> TYC (CIK 0000833444, open start, to 2016-09-02) from the cited entry; JCI has no register window yet the fetch")
    print("  lists TYC; a ticker outside the override is unchanged; an entry without a source is refused.")


def test_price_only_events_are_the_ticker_events_the_vendor_shows_as_a_spinoff_without_a_split() -> None:
    actions = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "action": a, "value": v} for t, d, a, v in JCI_ACTIONS])
    yf = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in JCI_YF])
    jci = cc.PredecessorSeries("JCI", "TYC", "0000833444", None, JCI_SEAM)
    got = cc.price_only_events(yf, actions, [(jci, JCI_SEAM)])
    # the existing windows' in-window ticker events are genuine vendor splits (PX1 2003-12-16, STE1 1998-08-25): none price-only
    lin = cc.PredecessorSeries("LIN", "PX1", "0000884905", None, pd.Timestamp("2018-10-31"))
    ste = cc.PredecessorSeries("STE", "STE1", "0000815065", None, pd.Timestamp("2015-11-02"))
    others = cc.price_only_events(
        pd.DataFrame({"ticker": ["LIN", "STE"], "date": pd.to_datetime(["2003-12-16", "1998-08-25"]), "ratio": [2.0, 2.0]}),
        pd.DataFrame({"ticker": ["PX1", "STE1"], "date": pd.to_datetime(["2003-12-16", "1998-08-25"]), "action": "split", "value": [2.0, 2.0]}),
        [(lin, pd.Timestamp("2018-10-31")), (ste, pd.Timestamp("2015-11-02"))],
    )
    print("\n=== SANITY CHECK: price-only events inside a window ===")
    print(got.to_string())
    assert got[["ticker", "date", "value"]].values.tolist() == [["JCI", pd.Timestamp("2012-10-01"), SPIN_2012]]
    assert others.empty
    print("  OK: only 2012-10-01 (TYC spinoffs ADT1/PNR, no split); 2007-07-02 has a TYC split, 1995/1997/1999 no TYC spinoff,")
    print("  2016-09-06 is after the seam; LIN's and STE's in-window events are vendor splits, so their windows are unchanged.")


def test_jci_rows_before_the_seam_are_tyco_on_jci_basis_per_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    arq, out = _merged(_jci_store(), "JCI", _config(tmp_path, [JCI_EXCHANGE], [JCI_OVERRIDE]), monkeypatch, caplog)
    truth = pd.DataFrame(
        TYC_ROWS, columns=["date", "reportperiod", "calendardate", "sharesbas", "shareswa", "price", "marketcap", "dps", "close_split"]
    )
    truth = truth.assign(quarter=[str(pd.Period(d, freq="Q")) for d in truth["calendardate"]]).set_index("quarter")
    factor = pd.Series([0.955 * SPIN_2012 if pd.Timestamp(d) < pd.Timestamp("2012-10-01") else 0.955 for d in truth["date"]], index=truth.index)
    print("\n=== SANITY CHECK: JCI <- TYC on JCI's share basis ===")
    shown = arq.loc[truth.index, ["ticker", "date", "sharesbas", "price", "marketcap"]].assign(factor=factor, close_split=truth["close_split"])
    print(shown.to_string())
    print(out[["sharesOutstanding", "sharesOutstandingPit"]].to_string())
    assert arq.loc[truth.index, "marketcap"].tolist() == truth["marketcap"].tolist(), "the inside rows are Tyco's"
    assert arq.loc[truth.index, "sharesbas"].tolist() == pytest.approx((truth["sharesbas"] * factor).tolist())
    assert arq.loc[truth.index, "dps"].tolist() == pytest.approx((truth["dps"] / factor).tolist())
    # known truth: Tyco's price on JCI's basis is JCI's own close_split, so close_split x shares is Tyco's market cap
    assert arq.loc[truth.index, "price"].tolist() == pytest.approx(truth["close_split"].tolist(), rel=1e-6)
    implied = truth["close_split"] * arq.loc[truth.index, "sharesbas"] / truth["marketcap"]
    assert implied.tolist() == pytest.approx([1.0] * len(truth), rel=1e-6)
    assert arq.loc["2016Q4", "sharesbas"] == 935e6 and arq.loc["2016Q4", "price"] == 41.0
    # PIT: Tyco's as-filed count (its own 1999 2:1 and 2007 1:4 undone), never the price-only spin or old JCI's splits
    assert out.loc["2006Q1", "sharesOutstandingPit"] == pytest.approx(509150440 / 0.25)
    assert out.loc["2010Q4", "sharesOutstandingPit"] == pytest.approx(473753233)
    assert out.loc["2012Q2", "sharesOutstandingPit"] == pytest.approx(459875217)
    assert out.loc["2012Q3", "sharesOutstandingPit"] == pytest.approx(465717368)
    assert out.loc["2016Q2", "sharesOutstandingPit"] == pytest.approx(426224367)
    assert out.loc["2016Q4", "sharesOutstandingPit"] == pytest.approx(935e6)
    print("  OK: inside the window Tyco's real rows at 0.955 x 2.0117 before 2012-10-01 and 0.955 after (no 0.25, no 0.955^2):")
    print("  converted price = JCI close_split and close_split x shares = Tyco's market cap; PIT = Tyco's as-filed count; after the seam untouched.")
