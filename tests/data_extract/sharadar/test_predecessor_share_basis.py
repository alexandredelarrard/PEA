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

from src.data_aggregate.utils.common.level_basis import genuine_splits, level_factor
from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar, gap_check, merge_history
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
    assert (exchanges["JCI"].predecessor_cik, exchanges["JCI"].seam_date, exchanges["JCI"].ratio) == ("0000833444", pd.Timestamp("2016-09-02"), 1)
    with pytest.raises(sm.SecurityManualError):
        sm.parse_security_manual({"exchange_ratios": [{"ticker": "X", "predecessor_cik": "1", "seam_date": "2020-01-01", "ratio": 2.0}]})
    print("  OK: five cited entries, each ratio 1 (JCI's 0.955 consolidation is price-only for split_events, so it lives in S); PLD/DD have none;")
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
JCI_EXCHANGE = {"ticker": "JCI", "predecessor_cik": "0000833444", "seam_date": "2016-09-02", "ratio": 1, "source": "0001104659-16-143068"}


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


def _tyco_genuine_events() -> pd.DataFrame:
    """The genuine split events of JCI as the cube's S needs them once its actions are Tyco's before the seam:
    TYC's own sharadar_actions relabelled to JCI (old JCI's 2004/2007-10 splits left out), with JCI's prices_splits."""
    actions = pd.DataFrame(
        [
            {"ticker": "JCI", "date": pd.Timestamp(d), "action": a, "value": v}
            for t, d, a, v in JCI_ACTIONS
            if t == "TYC" or pd.Timestamp(d) >= JCI_SEAM
        ]
    )
    yf = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in JCI_YF])
    return genuine_splits(actions, yf)


def test_jci_rows_before_the_seam_are_tyco_and_agree_with_the_cube_level_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shares carry only genuine share events; the 2012 spin and the 2016 consolidation (rejected by split_events)
    live in S(d), so close_split x S x shares and raw price x PIT shares are both Tyco's market cap."""
    arq, out = _merged(_jci_store(), "JCI", _config(tmp_path, [JCI_EXCHANGE], [JCI_OVERRIDE]), monkeypatch, caplog)
    truth = pd.DataFrame(
        TYC_ROWS, columns=["date", "reportperiod", "calendardate", "sharesbas", "shareswa", "price", "marketcap", "dps", "close_split"]
    )
    truth = truth.assign(quarter=[str(pd.Period(d, freq="Q")) for d in truth["calendardate"]], date=pd.to_datetime(truth["date"])).set_index(
        "quarter"
    )
    yf = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in JCI_YF])
    level = level_factor(pd.DatetimeIndex(truth["date"]), ["JCI"], yf, _tyco_genuine_events())["JCI"]
    s = pd.Series(level.to_numpy(), index=truth.index)
    yahoo_after = pd.Series([float(np.prod([v for _, d, v in JCI_YF if pd.Timestamp(d) > day] or [1.0])) for day in truth["date"]], index=truth.index)
    shares = arq.loc[truth.index, "sharesbas"]
    cube = truth["close_split"] * s * shares / truth["marketcap"]
    pit = truth["close_split"] * yahoo_after * out.loc[truth.index, "sharesOutstandingPit"] / truth["marketcap"]
    print("\n=== SANITY CHECK: JCI <- TYC against the cube's level factor ===")
    print(pd.DataFrame({"date": truth["date"].dt.date, "sharesbas": shares, "S": s, "cube_mc/vendor": cube, "pit_mc/vendor": pit}).to_string())
    assert arq.loc[truth.index, "marketcap"].tolist() == truth["marketcap"].tolist(), "the inside rows are Tyco's"
    assert shares.tolist() == truth["sharesbas"].tolist(), "no price-only factor in the share count"
    assert arq.loc[truth.index, "price"].tolist() == truth["price"].tolist()
    assert s["2006Q1"] == pytest.approx(SPIN_2012 * 0.955) and s["2012Q3"] == pytest.approx(0.955)
    assert cube.tolist() == pytest.approx([1.0] * len(truth), rel=1e-6)
    # 1996Q4 is filed before Tyco's 1997-10-23 2:1, which TYC's sharadar_actions do not carry, so its PIT is not as-filed
    assert pit.drop("1996Q4").tolist() == pytest.approx([1.0] * (len(truth) - 1), rel=1e-6)
    assert arq.loc["2016Q4", "sharesbas"] == 935e6 and out.loc["2016Q4", "sharesOutstandingPit"] == pytest.approx(935e6)
    print("  OK: Tyco's sharesbas unchanged (ratio 1, no genuine JCI split after the seam); S = 2.0117 x 0.955 before 2012-10-01")
    print("  and 0.955 after; close_split x S x shares and raw price x PIT both equal Tyco's market cap; after the seam untouched.")


def test_old_jci_split_actions_make_the_cube_level_factor_wrong_before_2007_10() -> None:
    """Why S reads the window owner's actions: JCI's own sharadar_actions before the seam are old JCI's (2004 x2,
    2007-10 x3), and split_events accepts them as genuine for ticker JCI, so S from them is off by 1/6 and 1/3 before
    2007-10-03 (`level_basis.level_actions` swaps in TYC's; see `test_level_predecessor_actions`)."""
    actions = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "action": a, "value": v} for t, d, a, v in JCI_ACTIONS if t == "JCI"])
    yf = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": v} for t, d, v in JCI_YF])
    days = pd.DatetimeIndex(["2003-05-01", "2006-05-09", "2011-01-27"])
    live = level_factor(days, ["JCI"], yf, genuine_splits(actions, yf))["JCI"]
    right = level_factor(days, ["JCI"], yf, _tyco_genuine_events())["JCI"]
    print("\n=== SANITY CHECK: S(d) with old JCI's split actions ===")
    print(pd.DataFrame({"live_actions": live, "tyco_actions": right, "ratio": live / right}).to_string())
    assert (live / right).tolist() == pytest.approx([1 / 6, 1 / 3, 1.0])
    print("  OK (pinned defect): old JCI's 2004 and 2007-10 splits divide S by 6 and 3 before 2007-10-03; the cube reads TYC's actions instead.")


def test_the_gap_check_compares_the_sec_history_with_the_swapped_vendor_series(tmp_path: Path) -> None:
    """F-008: JCI's SEC history before the seam is Tyco's, so the gap check must read TYC's vendor rows there, as the merge
    does. Old JCI's own rows are published on Tyco's dates here, so comparing them is a gap on every pre-seam quarter."""
    store = _jci_store()
    vendor = store.t[str(Tables.sharadar_fundamentals)]
    tyc = vendor[vendor["ticker"].eq("TYC")].set_index("calendardate")
    inside = vendor["ticker"].eq("JCI") & vendor["calendardate"].isin(tyc.index)
    vendor.loc[inside, "date"] = vendor.loc[inside, "calendardate"].map(tyc["date"])
    fields = gap_check.comparable_fields(load_field_map(str(REPO_CONFIGS)))
    # Tyco's SEC rows on Tyco's publication dates, and JCI's own after the seam
    sec = pd.DataFrame({"ticker": "JCI", "as_of": pd.to_datetime([*tyc.loc[vendor.loc[inside, "calendardate"], "date"], "2017-02-09"])})
    sec = sec.assign(**{f: float("nan") for f in fields}).assign(sharesOutstanding=[*tyc.loc[vendor.loc[inside, "calendardate"], "sharesbas"], 935e6])
    store.t[str(Tables.fundamentals_history_sec)] = sec
    config = SimpleNamespace(data_extract=SimpleNamespace(sharadar_gap_floor={"money": 1e6, "shares": 1e5, "ratio": 0.005}))
    config_dir = _config(tmp_path, [JCI_EXCHANGE], [JCI_OVERRIDE])
    context = SimpleNamespace(store=store, log=logging.getLogger(LOGGER), config=config, config_dir=config_dir)
    gaps = gap_check.measure_gaps(context, ["JCI"], config_dir=config_dir).set_index(["ticker", "field"])
    shares = gaps.loc[("JCI", "sharesOutstanding")]
    print("\n=== SANITY CHECK: gap check through the JCI <- TYC vendor series ===")
    print(gaps[["n_dates", "n_flagged", "median_pct_gap"]].to_string())
    assert shares["n_dates"] == int(inside.sum()) + 1, "every pre-seam Tyco quarter and the post-seam row are compared"
    assert shares["n_flagged"] == 0, "old JCI's raw rows were compared with Tyco's SEC history"
    assert gaps["n_flagged"].sum() == 0
    print(f"  OK: {int(inside.sum())} pre-seam quarters compared with TYC's rows (0 flagged), the post-seam JCI row with JCI's own.")
