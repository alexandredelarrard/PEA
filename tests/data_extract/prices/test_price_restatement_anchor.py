"""A restated adjustment basis is detected from the data and re-pulled (AC-001, AC-002, AC-008).

CTVA's spinoff left `prices` on two bases because the incremental fetch re-pulls a ticker only for an
event dated after its last stored bar. The fix compares a per-ticker ANCHOR (the bars of its first
stored calendar year) with Yahoo's fresh `close_split` and re-pulls on a mismatch.

Known-truth fixtures: a synthetic deterministic price path per ticker, a spinoff modelled as a factor
`_FACTOR` on every bar before the ex-date (volume on the same basis), a fake `yf.download` that honours
tickers, dates and the `actions` flag and records its calls. No network. Everything is driven through the
public `fetch_prices_and_actions`; an anchor download is any call with `actions=False` (the resume and the
re-pull downloads carry `actions=True`). `anchor_restated` is the pure comparison.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_extract.utils.prices import fetch_dividends as fd
from src.data_extract.utils.prices import fetch_prices as fp
from src.data_extract.utils.prices.anchor_check import anchor_restated
from src.data_store.schema import Tables
from src.data_store.store import DataStore

_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
_ACTIONS = ["Dividends", "Stock Splits"]
_PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close"]
_FACTOR = 0.15  # CTVA: 1 / 6.665
_LAST_STORED = pd.Timestamp("2024-06-14")  # last stored bar; the overlap window starts 2024-06-07
_AS_OF = pd.Timestamp("2024-06-19")  # last completed session: Tuesday 2024-06-18
_FIRST = pd.Timestamp("2023-01-02")
_UNIVERSE = ["CTVA", "QUIET"]


def _bars(base: float, start: str = "2023-01-02", end: str = "2024-06-18") -> pd.DataFrame:
    """Deterministic daily bars on the current basis; Adj Close sits 3% under Close (a dividend-like gap)."""
    idx = pd.bdate_range(start, end, name="Date")
    t = np.asarray((idx - _FIRST).days, dtype=float)
    close = base + 0.05 * t + 3.0 * np.sin(t / 7.0)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Adj Close": close * 0.97,
            "Volume": 1_000_000.0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


def _rescale(bars: pd.DataFrame, before: str, factor: float) -> pd.DataFrame:
    """The same series on another basis before `before`: prices x factor, volume / factor (what a spinoff does)."""
    out = bars.copy()
    mask = out.index < pd.Timestamp(before)
    out.loc[mask, _PRICE_FIELDS] = out.loc[mask, _PRICE_FIELDS] * factor
    out.loc[mask, "Volume"] = out.loc[mask, "Volume"] / factor
    return out


def _with_event(bars: pd.DataFrame, day: str, *, split: float = 0.0, dividend: float = 0.0) -> pd.DataFrame:
    out = bars.copy()
    out.loc[pd.Timestamp(day), "Stock Splits"] = split
    out.loc[pd.Timestamp(day), "Dividends"] = dividend
    return out


class FakeYahoo:
    """`yf.download` over per-ticker bar frames; records every call and can answer an anchor download with nothing."""

    def __init__(self) -> None:
        self.response: dict[str, pd.DataFrame] = {}
        self.anchor_empty = False
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tickers: list[str], start: str, end: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append({"tickers": list(tickers), "start": pd.Timestamp(start), "end": pd.Timestamp(end)} | kwargs)
        if not kwargs.get("actions") and self.anchor_empty:
            return pd.DataFrame()
        keep = [*_FIELDS, *(_ACTIONS if kwargs.get("actions") else [])]
        parts = {}
        for ticker in tickers:
            frame = self.response.get(ticker)
            if frame is None:
                continue
            rows = (frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))
            parts[ticker] = frame.loc[rows, keep]
        return pd.concat(parts, axis=1, sort=True) if parts else pd.DataFrame()

    @property
    def resume(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["actions"]]

    @property
    def anchor(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if not c["actions"]]

    def repulls(self, ticker: str) -> list[dict[str, Any]]:
        """The `actions=True` downloads after the first (resume) one that name `ticker`."""
        return [c for c in self.resume[1:] if ticker in c["tickers"]]


@pytest.fixture
def yahoo(monkeypatch: pytest.MonkeyPatch) -> FakeYahoo:
    fake = FakeYahoo()
    monkeypatch.setattr(fp.yf, "download", fake)
    monkeypatch.setattr(fp.time, "sleep", lambda seconds: None)
    return fake


def _context(store: DataStore) -> Any:
    config = OmegaConf.create({"data_extract": {"years_history": 1, "macro_years_history": 1}})
    return SimpleNamespace(store=store, log=logging.getLogger("test.anchor"), config=config, config_dir=".")


def _seed(
    store: DataStore,
    yahoo: FakeYahoo,
    stored: dict[str, pd.DataFrame],
    fresh: dict[str, pd.DataFrame],
    mutate: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> None:
    """Store `stored` (through `_LAST_STORED`, dividends all zero) then make Yahoo answer with `fresh`."""
    yahoo.response = stored
    first = min(frame.index.min() for frame in stored.values())
    bars = fp.download_ohlcv(list(stored), first, _LAST_STORED, pause=0.0, auto_adjust=False, actions=False)
    if mutate is not None:
        bars = mutate(bars)
    store.save(Tables.prices, bars)
    store.save(Tables.dividends, fd._extract_dividends(bars.assign(dividends=0.0)))
    yahoo.response = fresh
    yahoo.calls.clear()


def _fetch(store: DataStore, *, full: bool = False, universe: list[str] | None = None) -> None:
    fp.fetch_prices_and_actions(_context(store), universe or _UNIVERSE, years_history=1, full=full, as_of=_AS_OF, pause=0.0)


def _stored(store: DataStore, ticker: str, column: str = "close_split") -> pd.Series:
    df = cast(pd.DataFrame, store.load(Tables.prices, where={"ticker": [ticker]})).sort_values("date")
    return df.set_index(pd.to_datetime(df["date"]))[column]


def _max_step(series: pd.Series) -> float:
    return float(series.pct_change().abs().max())


def _assert_basis_repaired(store: DataStore, truth: pd.DataFrame) -> None:
    got = _stored(store, "CTVA")
    expected = truth["Close"].reindex(got.index)
    assert float((got / expected - 1).abs().max()) < 1e-9, "CTVA must be stored on Yahoo's current basis over its whole history"
    assert _max_step(got) < 0.5, "no one-bar basis step may remain"


def _stale_ctva(event: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(truth on today's basis, the same series stored on the old basis before `event`)."""
    truth = _bars(60.0)
    return truth, _rescale(truth, event, 1.0 / _FACTOR)


def test_a_ctva_shape_event_inside_the_overlap_is_re_pulled_in_full(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth, stale = _stale_ctva("2024-06-12")
    fresh_ctva = _with_event(truth, "2024-06-12", split=1.0 / _FACTOR)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})
    assert _max_step(_stored(sqlite_store, "CTVA")) > 0.5, "fixture: the stored CTVA series carries the old basis before the ex-date"

    _fetch(sqlite_store)

    repulls = yahoo.repulls("CTVA")
    assert repulls, f"no full re-pull although the stored basis is stale; downloads: {[(c['tickers'], c['start'].date()) for c in yahoo.calls]}"
    assert repulls[0]["start"] <= _FIRST, "the re-pull must start at the earliest stored bar"
    assert not yahoo.repulls("QUIET"), "the quiet ticker must not be re-pulled"
    _assert_basis_repaired(sqlite_store, truth)
    print("\n=== SANITY CHECK: (a) CTVA shape ===")
    print("  ex-date 2024-06-12 <= last bar 2024-06-14, only the 7-day tail restated -> CTVA re-pulled from the first stored bar,")
    print("  QUIET untouched, stored history on one basis (no one-bar step above 50%).")


def test_b_event_registered_after_the_overlap_is_caught_by_the_anchor_alone(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth, stale = _stale_ctva("2024-03-01")
    fresh_ctva = _with_event(truth, "2024-03-01", split=1.0 / _FACTOR)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})

    _fetch(sqlite_store)

    recent = _stored(sqlite_store, "CTVA").loc["2024-06-03":"2024-06-14"]
    expected = truth["Close"].reindex(recent.index)
    assert float((recent / expected - 1).abs().max()) < 1e-9, "fixture: every overlap bar agrees with Yahoo"
    repulls = yahoo.repulls("CTVA")
    assert repulls, (
        f"overlap bars agree but the first-year anchor does not: a re-pull was expected; downloads: {[(c['tickers'], c['start'].date()) for c in yahoo.calls]}"
    )
    assert not yahoo.repulls("QUIET")
    _assert_basis_repaired(sqlite_store, truth)
    print("\n=== SANITY CHECK: (b) late-registered event ===")
    print("  ex-date 2024-03-01 far before the overlap, overlap bars all agree, first-year bars do not -> CTVA re-pulled in full.")


@pytest.mark.parametrize("nan_close_split", [False, True], ids=["dividend-only", "dividend-only-nan-anchor"])
def test_c_a_dividend_only_change_does_not_trigger(sqlite_store: DataStore, yahoo: FakeYahoo, nan_close_split: bool) -> None:
    truth = _bars(60.0)
    stored_ctva = truth.copy()
    stored_ctva["Adj Close"] = np.where(stored_ctva.index < pd.Timestamp("2024-06-12"), truth["Adj Close"] * 1.03, truth["Adj Close"])
    fresh_ctva = _with_event(truth, "2024-06-12", dividend=0.5)
    if nan_close_split:
        fresh_ctva.loc[fresh_ctva.index[:2], "Close"] = np.nan  # Yahoo omits Close on two anchor bars; Adj Close still moves
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stored_ctva, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})

    _fetch(sqlite_store)

    assert len(yahoo.resume) == 1, f"a dividend-only restatement moved close_total, not close_split: no re-pull expected, got {len(yahoo.resume) - 1}"
    print("\n=== SANITY CHECK: (c) dividend-only ===")
    print(
        f"  close_total restated, close_split identical{' (NaN on two fresh anchor bars)' if nan_close_split else ''} -> a single resume download, no re-pull."
    )


def test_d_an_anchor_equal_to_storage_with_a_stale_event_row_does_not_re_pull(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth = _bars(60.0)
    fresh_ctva = _with_event(truth, "2024-06-10", split=1.0 / _FACTOR)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": truth, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})
    sqlite_store.save(Tables.prices_splits, pd.DataFrame({"date": [pd.Timestamp("2024-06-10")], "ticker": ["CTVA"], "ratio": [1.0 / _FACTOR]}))

    _fetch(sqlite_store)

    assert len(yahoo.resume) == 1, "storage already equals Yahoo: the registered event must not cost a re-pull"
    print("\n=== SANITY CHECK: (d) AC-002 quiet night ===")
    print("  anchor equals storage, split row 2024-06-10 already registered and dated before the last bar -> no re-pull.")


def test_e_an_empty_anchor_response_triggers_nothing(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth, stale = _stale_ctva("2024-03-01")
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})
    yahoo.anchor_empty = True
    before = _stored(sqlite_store, "CTVA").loc[:"2024-03-01"].copy()

    _fetch(sqlite_store)

    assert len(yahoo.resume) == 1, "an empty anchor response is missing data, never evidence of a restatement"
    pd.testing.assert_series_equal(_stored(sqlite_store, "CTVA").loc[:"2024-03-01"], before)
    print("\n=== SANITY CHECK: (e) Yahoo outage on the anchor call ===")
    print("  anchor download returns nothing -> no trigger, no crash, stored history untouched.")


def test_f_a_full_run_downloads_no_anchor(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth, stale = _stale_ctva("2024-03-01")
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})

    _fetch(sqlite_store, full=True)

    assert yahoo.anchor == [], f"full=True already pulls the whole window; anchor downloads seen: {yahoo.anchor}"
    assert len(yahoo.resume) == 1
    print("\n=== SANITY CHECK: (f) full=True ===")
    print("  one whole-window download, no anchor call, no extra re-pull.")


def test_g_sparse_first_year_still_triggers_and_a_missing_first_year_does_not(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    # CTVA listed 2023-12-28: its first calendar year holds only 2 stored bars (< 3 common bars).
    truth = _bars(60.0, start="2023-12-28")
    stale = _rescale(truth, "2024-03-01", 1.0 / _FACTOR)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})

    _fetch(sqlite_store)

    assert yahoo.repulls("CTVA"), "two common bars, both restated: the sparse-year rule must still re-pull"
    assert not yahoo.repulls("QUIET")
    print("\n=== SANITY CHECK: (g1) sparse first year ===")
    print("  2 first-year bars, both on the old basis -> max-deviation rule fires, CTVA re-pulled.")


def test_g_zero_common_bars_triggers_nothing(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    stale = _rescale(_bars(60.0, start="2023-12-28"), "2024-01-01", 1.0 / _FACTOR)  # only the two December bars differ
    served = _bars(60.0, start="2024-01-02")  # Yahoo no longer serves CTVA's two December bars
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": served, "QUIET": quiet})

    _fetch(sqlite_store)

    assert len(yahoo.resume) == 1, "no common anchor date means no evidence: never trigger"
    print("\n=== SANITY CHECK: (g2) zero common bars ===")
    print("  stored first-year dates absent from the fresh anchor -> no trigger.")


@pytest.mark.parametrize("pattern", ["mixed", "all-bad"])
def test_h_a_stored_zero_or_nan_close_split_neither_triggers_nor_crashes(sqlite_store: DataStore, yahoo: FakeYahoo, pattern: str) -> None:
    truth = _bars(60.0, start="2023-12-26")
    quiet = _bars(150.0)
    bad = [0.0, np.nan, 0.0, np.nan] if pattern == "all-bad" else [0.0, np.nan, None, None]

    def corrupt(bars: pd.DataFrame) -> pd.DataFrame:
        out = bars.copy()
        rows = out.index[(out["ticker"] == "CTVA") & (out["date"] <= pd.Timestamp("2023-12-29"))]
        for row, value in zip(rows, bad, strict=True):
            if value is not None:
                out.loc[row, "close_split"] = value
        return out

    _seed(sqlite_store, yahoo, {"CTVA": truth, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet}, mutate=corrupt)

    _fetch(sqlite_store)

    assert len(yahoo.resume) == 1, "stored zero/NaN close_split is not a restatement: no re-pull"
    print(f"\n=== SANITY CHECK: (h) stored zero/NaN close_split ({pattern}) ===")
    print("  unusable stored anchor bars are filtered before the ratio -> no trigger, no crash.")


def test_i_a_key_already_re_pulled_by_the_event_rule_is_not_anchor_downloaded(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth, stale = _stale_ctva("2024-06-17")
    fresh_ctva = _with_event(truth, "2024-06-17", split=1.0 / _FACTOR)  # ex-date after the last stored bar
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})

    _fetch(sqlite_store)

    assert yahoo.repulls("CTVA"), "the event rule must still re-pull a split after the last bar"
    assert all("CTVA" not in call["tickers"] for call in yahoo.anchor), f"CTVA is re-pulled in full already: {yahoo.anchor}"
    print("\n=== SANITY CHECK: (i) no double work ===")
    print("  split after the last bar -> event-rule re-pull; CTVA is absent from every anchor download.")


def _unserved_first_year(yahoo: FakeYahoo, served_from: str) -> None:
    """Yahoo no longer serves any bar of CTVA before `served_from` (the PSKY / WBD shape)."""
    yahoo.response["CTVA"] = yahoo.response["CTVA"].loc[lambda f: f.index >= pd.Timestamp(served_from)]


def test_j_unserved_first_year_falls_back_to_a_later_served_year_and_re_pulls(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth, stale = _stale_ctva("2024-06-03")
    fresh_ctva = _with_event(truth, "2024-06-03", split=1.0 / _FACTOR)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})
    _unserved_first_year(yahoo, "2024-01-02")  # 2023 unserved; 2024 served and restated

    _fetch(sqlite_store)

    assert yahoo.repulls("CTVA"), (
        f"first year unserved, 2024 served and restated: re-pull expected; anchors: {[(c['tickers'], c['start'].date()) for c in yahoo.anchor]}"
    )
    assert not yahoo.repulls("QUIET")
    print("\n=== SANITY CHECK: (j) fallback anchor year ===")
    print("  2023 not served, 2024 served and restated -> CTVA re-pulled; QUIET untouched.")


def test_k_unserved_first_year_with_an_agreeing_later_year_is_silent(
    sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture
) -> None:
    truth = _bars(60.0)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": truth, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})
    _unserved_first_year(yahoo, "2024-01-02")

    with caplog.at_level(logging.WARNING):
        _fetch(sqlite_store)

    assert len(yahoo.resume) == 1, "the fallback year agrees with storage: no re-pull"
    assert not [r for r in caplog.records if r.name.endswith("anchor_check") and "CTVA" in r.getMessage() and r.levelno >= logging.WARNING], (
        "an anchored ticker is not warned about"
    )
    print("\n=== SANITY CHECK: (k) fallback agrees ===")
    print("  2023 not served, 2024 served and equal -> no re-pull, no warning.")


def test_l_a_ticker_served_in_no_year_is_named_in_one_warning(sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture) -> None:
    stale = _rescale(_bars(60.0), "2024-03-01", 1.0 / _FACTOR)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": stale, "QUIET": quiet})
    _unserved_first_year(yahoo, "2030-01-01")  # nothing of CTVA is served in any year
    before = _stored(sqlite_store, "CTVA").copy()

    with caplog.at_level(logging.WARNING):
        _fetch(sqlite_store)

    named = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name.endswith("anchor_check") and "CTVA" in r.getMessage() and "QUIET" not in r.getMessage()
    ]
    assert len(named) == 1, f"exactly one WARNING must name the unanchorable ticker, got {[r.getMessage() for r in caplog.records]}"
    assert len(yahoo.resume) == 1, "missing data is never evidence: no re-pull"
    pd.testing.assert_series_equal(_stored(sqlite_store, "CTVA"), before)
    print("\n=== SANITY CHECK: (l) unanchorable ticker ===")
    print("  no year served -> one WARNING naming CTVA, no trigger, no crash, storage untouched.")


def test_m_a_ticker_served_in_its_first_year_is_downloaded_once(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    truth = _bars(60.0)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": truth, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})

    _fetch(sqlite_store)

    assert len([c for c in yahoo.anchor if "CTVA" in c["tickers"]]) == 1, f"served in its first year: one anchor download, got {yahoo.anchor}"
    assert len(yahoo.anchor) == 1
    print("\n=== SANITY CHECK: (m) no extra calls ===")
    print("  both tickers served in 2023 -> exactly one grouped anchor download.")


def _anchor_records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.endswith("anchor_check") and r.levelno == level]


def _breadth_run(store: DataStore, yahoo: FakeYahoo, n_tickers: int, n_stale: int) -> list[str]:
    """`n_tickers` tickers served in their first year, the first `n_stale` stored on the old basis; returns them."""
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    truth = {t: _bars(50.0 + i) for i, t in enumerate(tickers)}
    stored = {t: _rescale(frame, "2024-03-01", 1.0 / _FACTOR) if i < n_stale else frame for i, (t, frame) in enumerate(truth.items())}
    _seed(store, yahoo, stored, truth)
    _fetch(store, universe=tickers)
    return tickers


def test_n_a_few_triggered_tickers_are_re_pulled_and_their_factor_logged(
    sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        tickers = _breadth_run(sqlite_store, yahoo, n_tickers=34, n_stale=1)

    assert yahoo.repulls("T00") and not any(yahoo.repulls(t) for t in tickers[1:]), "1 of 34 triggered (3%): it is re-pulled alone"
    warned = [m for m in _anchor_records(caplog, logging.WARNING) if "disagrees" in m]
    assert warned and "T00 x0.1500" in warned[0], f"the WARNING must give the median fresh/stored factor: {warned}"
    assert not _anchor_records(caplog, logging.ERROR)
    print("\n=== SANITY CHECK: (n) breadth guard, 3% ===")
    print(f"  1 of 34 compared tickers restated -> re-pulled; WARNING: {warned[0][-40:]}")


def test_n_a_market_wide_trigger_re_pulls_none_and_logs_an_error(sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        tickers = _breadth_run(sqlite_store, yahoo, n_tickers=20, n_stale=6)

    assert len(yahoo.resume) == 1, f"6 of 20 triggered (30%): no anchor re-pull, got {[c['tickers'] for c in yahoo.resume[1:]]}"
    errors = _anchor_records(caplog, logging.ERROR)
    assert len(errors) == 1 and "price-history" in errors[0] and all(t in errors[0] for t in tickers[:6]), errors
    print("\n=== SANITY CHECK: (n) breadth guard, 30% ===")
    print(f"  6 of 20 restated -> none re-pulled, one ERROR: {errors[0][:90]}...")


def test_n_a_small_run_is_not_guarded(sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        tickers = _breadth_run(sqlite_store, yahoo, n_tickers=4, n_stale=2)

    assert yahoo.repulls("T00") and yahoo.repulls("T01") and not yahoo.repulls("T02"), "fewer than 20 compared: no breadth guard"
    assert not _anchor_records(caplog, logging.ERROR)
    print("\n=== SANITY CHECK: (n) small run ===")
    print(f"  2 of {len(tickers)} restated (50%) on a 4-ticker run -> both re-pulled, no ERROR.")


def test_o_a_re_pull_served_from_a_later_date_warns_about_the_stale_prefix(
    sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture
) -> None:
    truth, stale = _stale_ctva("2024-06-17")
    fresh_ctva = _with_event(truth, "2024-06-17", split=1.0 / _FACTOR).loc["2023-07-03":]  # Yahoo serves less history
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": stale, "QUIET": quiet}, {"CTVA": fresh_ctva, "QUIET": quiet})

    with caplog.at_level(logging.WARNING):
        _fetch(sqlite_store)

    assert yahoo.repulls("CTVA"), "the event rule still re-pulls"
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "CTVA" in r.getMessage() and "old basis" in r.getMessage()]
    assert len(warned) == 1 and "2023-01-02" in warned[0] and "2023-07-03" in warned[0], [r.getMessage() for r in caplog.records]
    assert float(_stored(sqlite_store, "CTVA").loc[:"2023-06-30"].iloc[0]) > 100, "nothing is deleted or rescaled: the old prefix stays"
    print("\n=== SANITY CHECK: (o) shorter served history ===")
    print(f"  re-pull served from 2023-07-03, stored from 2023-01-02 -> one WARNING: {warned[0][:100]}...")


def test_p_a_failing_anchor_download_is_not_checked_and_takes_no_fallback(
    sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = _bars(60.0)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": truth, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})

    def throttled(tickers: list[str], start: str, end: str, **kwargs: Any) -> pd.DataFrame:
        if not kwargs.get("actions"):
            yahoo.calls.append({"tickers": list(tickers), "start": pd.Timestamp(start), "end": pd.Timestamp(end)} | kwargs)
            raise RuntimeError("429 Too Many Requests")
        return yahoo(tickers, start, end, **kwargs)

    monkeypatch.setattr(fp.yf, "download", throttled)
    with caplog.at_level(logging.WARNING):
        _fetch(sqlite_store)

    years = {c["start"].year for c in yahoo.anchor}
    warned = _anchor_records(caplog, logging.WARNING)
    assert years == {2023}, f"a failed download is not evidence of an unserved year: no fallback, got years {years}"
    assert len(warned) == 1 and "CTVA" in warned[0] and "QUIET" in warned[0], warned
    assert len(yahoo.resume) == 1
    print("\n=== SANITY CHECK: (p) throttled anchor ===")
    print(f"  anchor download raises -> {len(yahoo.anchor)} attempts on 2023 only, one WARNING: {warned[0][:80]}...")


def test_q_the_anchor_log_counts_only_tickers_actually_compared(sqlite_store: DataStore, yahoo: FakeYahoo, caplog: pytest.LogCaptureFixture) -> None:
    truth = _bars(60.0)
    quiet = _bars(150.0)
    _seed(sqlite_store, yahoo, {"CTVA": truth, "QUIET": quiet}, {"CTVA": truth, "QUIET": quiet})
    _unserved_first_year(yahoo, "2024-01-02")

    with caplog.at_level(logging.INFO):
        _fetch(sqlite_store)

    first = [m for m in _anchor_records(caplog, logging.INFO) if "year 2023" in m]
    assert first and "1 ticker(s) compared" in first[0], first
    print("\n=== SANITY CHECK: (q) compared count ===")
    print(f"  CTVA unserved in 2023, QUIET served -> {first[0]}")


# --- the pure comparison -----------------------------------------------------------------------------------


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)), dtype=float)


_FLAT = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]


def test_anchor_restated_flags_a_uniform_factor_and_ignores_an_equal_series() -> None:
    stored = _series(_FLAT)
    assert anchor_restated(stored, stored * _FACTOR, 0.03) is True
    assert anchor_restated(stored, stored.copy(), 0.03) is False
    print("\n=== SANITY CHECK: anchor_restated basics ===")
    print("  every bar off by x0.15 -> restated; identical bars -> not restated.")


def test_anchor_restated_uses_the_median_so_one_bad_bar_does_not_trigger() -> None:
    stored = _series(_FLAT)
    fresh = stored.copy()
    fresh.iloc[2] = fresh.iloc[2] * 1.5
    assert anchor_restated(stored, fresh, 0.03) is False
    print("\n=== SANITY CHECK: median rule ===")
    print("  1 of 6 bars off by 50% -> median deviation 0 -> not restated.")


def test_anchor_restated_uses_the_max_below_three_common_bars() -> None:
    stored = _series([100.0, 100.0])
    assert anchor_restated(stored, pd.Series([100.0, 104.0], index=stored.index), 0.03) is True  # median 2% would pass, max 4% fails
    three = _series([100.0, 100.0, 100.0])
    assert anchor_restated(three, pd.Series([100.0, 100.0, 104.0], index=three.index), 0.03) is False  # three bars: median 0
    print("\n=== SANITY CHECK: sparse rule ===")
    print("  deviations [0, 4%] with tol 3% -> restated (max); [0, 0, 4%] -> not (median).")


def test_anchor_restated_has_no_opinion_without_common_or_valid_bars() -> None:
    stored = _series(_FLAT)
    elsewhere = pd.Series(_FLAT, index=stored.index + pd.Timedelta(days=400))
    assert anchor_restated(stored, elsewhere, 0.03) is False
    assert anchor_restated(stored, stored.iloc[0:0], 0.03) is False
    bad = stored.copy()
    bad.iloc[0], bad.iloc[1] = 0.0, np.nan
    assert anchor_restated(bad, stored, 0.03) is False, "zero and NaN stored bars are dropped, the rest agree"
    all_bad = pd.Series([0.0, np.nan, 0.0], index=stored.index[:3])
    assert anchor_restated(all_bad, stored * _FACTOR, 0.03) is False
    fresh_bad = stored.copy()
    fresh_bad.iloc[:3] = [np.nan, 0.0, np.nan]
    assert anchor_restated(stored, fresh_bad, 0.03) is False
    print("\n=== SANITY CHECK: no evidence, no trigger ===")
    print("  no common dates, empty fresh, zero/NaN on either side -> False, no exception.")


def test_anchor_restated_sees_an_event_inside_the_first_year() -> None:
    stored = pd.Series(np.linspace(100.0, 130.0, 260), index=pd.bdate_range("2023-01-02", periods=260))
    fresh = stored.copy()
    fresh.iloc[:40] = fresh.iloc[:40] / 2  # an event after bar 40 restated only the earlier bars
    assert anchor_restated(stored, fresh, 1e-4) is True
    print("\n=== SANITY CHECK: earliest anchor bars ===")
    print("  260-bar year, first 40 bars restated /2 -> the earliest 20 common bars disagree -> restated.")
