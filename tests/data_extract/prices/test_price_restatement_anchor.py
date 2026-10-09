"""A restated adjustment basis is detected from the data and re-pulled (AC-001, AC-002, AC-008).

CTVA's spinoff left `prices` on two bases because the incremental fetch re-pulls a ticker only for an
event dated after its last stored bar. The fix compares a per-ticker ANCHOR (the bars of its first
stored calendar year) with Yahoo's fresh `close_split` and re-pulls on a mismatch.

Known-truth fixtures: a synthetic deterministic price path per ticker, a spinoff modelled as a factor
`_FACTOR` on every bar before the ex-date (volume on the same basis), a fake `yf.download` that honours
tickers, dates and the `actions` flag and records its calls. No network. Everything is driven through the
public `fetch_prices_and_actions`; an anchor download is any call with `actions=False` (the resume and the
re-pull downloads carry `actions=True`).

Cases (a) and (b) are the genuine RED on the base code (no re-pull); (c)-(i) are guards that must stay
GREEN once the anchor rule exists. The pure comparison `anchor_restated(stored, fresh, tol)` (Series of
`close_split` indexed by date -> bool) lives in `anchor_check` and is imported inside its tests so that the
fetch-level cases still run before that module exists.
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


def _fetch(store: DataStore, *, full: bool = False) -> None:
    fp.fetch_prices_and_actions(_context(store), _UNIVERSE, years_history=1, full=full, as_of=_AS_OF, pause=0.0)


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
    stale = _rescale(_bars(60.0, start="2023-12-28"), "2024-03-01", 1.0 / _FACTOR)
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
    for ticker, frame in list(yahoo.response.items()):
        if ticker == "CTVA":
            yahoo.response[ticker] = frame.loc[frame.index >= pd.Timestamp(served_from)]


def _anchor_calls_naming(yahoo: FakeYahoo, ticker: str) -> list[dict[str, Any]]:
    return [c for c in yahoo.anchor if ticker in c["tickers"]]


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

    assert len(_anchor_calls_naming(yahoo, "CTVA")) == 1, f"served in its first year: one anchor download, got {yahoo.anchor}"
    assert len(yahoo.anchor) == 1
    print("\n=== SANITY CHECK: (m) no extra calls ===")
    print("  both tickers served in 2023 -> exactly one grouped anchor download.")


# --- the pure comparison -----------------------------------------------------------------------------------


def _anchor_restated() -> Callable[..., bool]:
    """Imported lazily: the module does not exist before P2 and the fetch-level cases above must still run."""
    from src.data_extract.utils.prices.anchor_check import anchor_restated

    return anchor_restated


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=len(values)), dtype=float)


_FLAT = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]


def test_anchor_restated_flags_a_uniform_factor_and_ignores_an_equal_series() -> None:
    f = _anchor_restated()
    stored = _series(_FLAT)
    assert f(stored, stored * _FACTOR, 0.03) is True
    assert f(stored, stored.copy(), 0.03) is False
    print("\n=== SANITY CHECK: anchor_restated basics ===")
    print("  every bar off by x0.15 -> restated; identical bars -> not restated.")


def test_anchor_restated_uses_the_median_so_one_bad_bar_does_not_trigger() -> None:
    f = _anchor_restated()
    stored = _series(_FLAT)
    fresh = stored.copy()
    fresh.iloc[2] = fresh.iloc[2] * 1.5
    assert f(stored, fresh, 0.03) is False
    print("\n=== SANITY CHECK: median rule ===")
    print("  1 of 6 bars off by 50% -> median deviation 0 -> not restated.")


def test_anchor_restated_uses_the_max_below_three_common_bars() -> None:
    f = _anchor_restated()
    stored = _series([100.0, 100.0])
    assert f(stored, pd.Series([100.0, 104.0], index=stored.index), 0.03) is True  # median 2% would pass, max 4% fails
    three = _series([100.0, 100.0, 100.0])
    assert f(three, pd.Series([100.0, 100.0, 104.0], index=three.index), 0.03) is False  # three bars: median 0
    print("\n=== SANITY CHECK: sparse rule ===")
    print("  deviations [0, 4%] with tol 3% -> restated (max); [0, 0, 4%] -> not (median).")


def test_anchor_restated_has_no_opinion_without_common_or_valid_bars() -> None:
    f = _anchor_restated()
    stored = _series(_FLAT)
    elsewhere = pd.Series(_FLAT, index=stored.index + pd.Timedelta(days=400))
    assert f(stored, elsewhere, 0.03) is False
    assert f(stored, stored.iloc[0:0], 0.03) is False
    bad = stored.copy()
    bad.iloc[0], bad.iloc[1] = 0.0, np.nan
    assert f(bad, stored, 0.03) is False, "zero and NaN stored bars are dropped, the rest agree"
    all_bad = pd.Series([0.0, np.nan, 0.0], index=stored.index[:3])
    assert f(all_bad, stored * _FACTOR, 0.03) is False
    fresh_bad = stored.copy()
    fresh_bad.iloc[:3] = [np.nan, 0.0, np.nan]
    assert f(stored, fresh_bad, 0.03) is False
    print("\n=== SANITY CHECK: no evidence, no trigger ===")
    print("  no common dates, empty fresh, zero/NaN on either side -> False, no exception.")
