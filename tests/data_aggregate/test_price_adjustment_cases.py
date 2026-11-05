"""Issuer-backed controls evaluated against bounded, read-only stored observations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context, get_config_context
from src.data_aggregate.transformers import step_cube_prices as price_step
from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.data_aggregate.utils.institutionals.insider_positions import position_units
from src.data_aggregate.utils.price.inputs import PRICE_COLS
from src.data_aggregate.utils.sec_thirteen.institutional_features import _daily_owned_shares
from src.data_store.schema import Tables
from src.utils.predecessor_series import load_level_actions as level_actions
from src.utils.price.adjustments import ShareBasis, load_bugfix, resolve_share_events
from src.utils.price.basis import load_share_basis
from src.utils.price.capitalization import project_security_shares
from src.utils.price.levels import apply_level_bugfix, as_traded_close, level_factor, security_basis_frame
from tests.data_aggregate.fundamental_pit_reference import legacy_fundamentals_to_daily
from tests.support.universe import universe_of


@pytest.fixture(scope="module")
def source_case() -> tuple[Context, pd.DataFrame, ShareBasis, dict]:
    _, context = get_config_context("./configs", use_cache=False, save=False)
    names = ["IVZ", "HBAN", "APA", "SJM"]
    shares = context.store.load(
        Tables.sharadar_fundamentals,
        columns=["ticker", "date", "dimension", "reportperiod", "sharesbas", "price"],
        where={"ticker": names, "dimension": "ARQ"},
        since="1995-01-01",
        until="2008-12-31",
    )
    actions = context.store.load(
        Tables.sharadar_actions, columns=["ticker", "date", "action", "value"], where={"ticker": names, "action": ["split", "spinoff"]}
    )
    yahoo = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names})
    assert shares is not None and actions is not None and yahoo is not None
    blob = load_bugfix(context.config_dir)
    basis = resolve_share_events(actions, yahoo, blob, shares=shares, tickers=names)
    return context, shares, basis, blob


@pytest.mark.parametrize(
    ("ticker", "stamp", "expected"),
    [
        ("HBAN", "1996-02-22", 137192253.0),
        ("HBAN", "1996-05-15", 132891485.0),
        ("APA", "2001-11-13", 124627521.0),
        ("SJM", "2002-03-11", 24858003.0),
        ("IVZ", "2006-06-23", 818106899.0),
        ("IVZ", "2007-11-09", 836680132.0),
    ],
)
def test_original_ordinary_share_controls(source_case: tuple, ticker: str, stamp: str, expected: float) -> None:
    _, shares, basis, _ = source_case
    row = shares.loc[shares["ticker"].eq(ticker) & pd.to_datetime(shares["date"]).eq(pd.Timestamp(stamp))]
    assert len(row) == 1
    actual = row["sharesbas"].iloc[0] / basis.factor(row["ticker"], row["date"], "vendor").iloc[0]
    assert actual == pytest.approx(expected, abs=1.0, rel=0)
    print(f"SANITY: {ticker} {stamp} vendor count restores {actual:,.3f} original ordinary shares (within one share).")


def test_real_ivz_level_quote_and_legacy_ordinary_carry(source_case: tuple) -> None:
    context, shares, basis, _ = source_case
    raw = context.store.load(
        Tables.prices, columns=["ticker", "date", "close_split"], where={"ticker": "IVZ"}, since="2006-01-01", until="2007-12-31"
    )
    assert raw is not None
    raw["date"] = pd.to_datetime(raw["date"])
    close = raw.pivot(index="date", columns="ticker", values="close_split")
    factor = level_factor(close.index, ["IVZ"], basis.price, basis)
    restored = as_traded_close(close, factor, basis)
    assert restored is not None
    assert factor["IVZ"].eq(1.0).all()
    for stamp, quote in [("2006-06-23", 17.17), ("2007-04-10", 23.16), ("2007-11-09", 27.05)]:
        assert restored.at[pd.Timestamp(stamp), "IVZ"] == pytest.approx(quote, abs=0.00002)
    filing = shares.loc[shares["ticker"].eq("IVZ") & pd.to_datetime(shares["date"]).eq(pd.Timestamp("2007-11-09"))]
    history = filing[["ticker", "date"]].rename(columns={"date": "as_of"}).assign(sharesOutstandingPit=836680132.0)
    index = pd.DatetimeIndex(["2007-11-09", "2007-12-03", "2007-12-04", "2007-12-05"])
    daily = legacy_fundamentals_to_daily(history, "sharesOutstandingPit", index, share_basis=basis)
    assert daily["IVZ"].tolist() == [836680132.0, 836680132.0, 418340066.0, 418340066.0]
    assert pd.Timestamp(history["as_of"].iloc[0]) == pd.Timestamp("2007-11-09")
    assert history["sharesOutstandingPit"].iloc[0] == 836680132.0
    stale = legacy_fundamentals_to_daily(history, "sharesOutstandingPit", index, max_age_days=20, share_basis=basis)
    assert stale.loc[pd.Timestamp("2007-12-04"), "IVZ"] != stale.loc[pd.Timestamp("2007-12-04"), "IVZ"]
    print("SANITY: IVZ level S=1 preserves ADS quotes; carried ordinary count converts December4 without refreshing its filing age.")


@pytest.mark.parametrize(
    ("ticker", "start", "boundary", "end", "expected_count"),
    [
        ("BLDR", "2009-12-14", "2009-12-17", "2010-01-05", 9),
        ("CCI", "2002-05-24", "2002-05-29", "2002-07-03", 23),
    ],
)
def test_real_level_boundaries(source_case: tuple, ticker: str, start: str, boundary: str, end: str, expected_count: int) -> None:
    context, _, _, blob = source_case
    raw = context.store.load(Tables.prices, columns=["ticker", "date", "close_split"], where={"ticker": ticker}, since="1995-01-01", until=end)
    vendor = context.store.load(
        Tables.sharadar_fundamentals, columns=["ticker", "date", "price"], where={"ticker": ticker, "dimension": "ARQ"}, until=end
    )
    assert raw is not None and vendor is not None
    raw["date"] = pd.to_datetime(raw["date"])
    vendor["date"] = pd.to_datetime(vendor["date"])
    close = raw.pivot(index="date", columns="ticker", values="close_split")
    factor = apply_level_bugfix(pd.DataFrame(1.0, index=close.index, columns=close.columns), blob, vendor, close)
    assert factor.loc[factor.index >= pd.Timestamp(boundary), ticker].eq(1.0).all()
    old_boundary = "2009-12-31" if ticker == "BLDR" else "2002-06-30"
    assert int(close.loc[(close.index >= boundary) & (close.index < old_boundary), ticker].notna().sum()) == expected_count
    print(f"SANITY: {ticker} ends its hidden wedge at {boundary}; {expected_count} real sessions avoid the extra multiplier.")


def test_actual_quantity_normalization_does_not_borrow_vendor_factor(source_case: tuple) -> None:
    _, shares, basis, _ = source_case
    trades = pd.DataFrame(
        {
            "ticker": ["HBAN"],
            "transaction_day": pd.to_datetime(["1996-02-22"]),
            "security_title": ["Common Stock"],
            "shares_n": [100.0],
            "shares_owned_after": [1000.0],
        }
    )
    normalized = position_units(trades, QuantityBasis.from_context(basis, pd.DatetimeIndex(shares["date"])))
    assert normalized["units_known"].all()
    assert normalized["shares_n"].iloc[0] == pytest.approx(100.0 * 1.1**5)
    assert normalized["shares_owned_after"].iloc[0] == pytest.approx(1000.0 * 1.1**5)
    assert basis.factor(trades["ticker"], trades["transaction_day"], "vendor").iloc[0] == 1.0
    assert trades["shares_n"].iloc[0] == 100.0
    print("SANITY: real HBAN stock dividends normalize actual positions by1.61051 while vendor PIT factor remains1; source quantities stay intact.")


@pytest.mark.parametrize("changed", ["source", "anchor"])
def test_ordinary_price_builder_refuses_changed_share_policy_before_write(
    source_case: tuple, monkeypatch: pytest.MonkeyPatch, changed: str, sqlite_store: Any, tmp_path: Path
) -> None:
    context, shares, _, _ = source_case
    raw = context.store.load(Tables.prices, columns=PRICE_COLS, where={"ticker": "IVZ"}, since="2007-11-01", until="2007-12-10")
    yahoo = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": "IVZ"})
    actions = context.store.load(
        Tables.sharadar_actions, columns=["ticker", "date", "action", "value"], where={"ticker": "IVZ", "action": ["split", "spinoff"]}
    )
    assert raw is not None and yahoo is not None and actions is not None
    if changed == "source":
        actions.loc[pd.to_datetime(actions["date"]).eq(pd.Timestamp("2007-11-01")), "value"] = 0.25
    vendor = shares.copy()
    if changed == "anchor":
        vendor.loc[vendor["ticker"].eq("IVZ"), "sharesbas"] *= 2
    sqlite_store.save(Tables.prices, raw)
    sqlite_store.save(Tables.prices_splits, yahoo)
    sqlite_store.save(Tables.sharadar_actions, actions.assign(contraticker=""))
    sqlite_store.save(Tables.sharadar_fundamentals, vendor)
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": raw["date"], "close": 1.0}))
    peer_path = tmp_path / "peers.json"
    peer_path.write_text(json.dumps({"IVZ": {"IVZ": 1.0}}), encoding="utf-8")
    isolated = SimpleNamespace(
        store=sqlite_store,
        config=context.config,
        config_dir=context.config_dir,
        log=context.log,
        paths={"SECTOR_PEERS_PATH": peer_path},
        universe=universe_of(["IVZ"]),
    )
    writes: list[object] = []
    monkeypatch.setattr(sqlite_store, "replace", lambda *args, **kwargs: writes.append(args))
    with pytest.raises(ValueError, match="changed sharadar|share anchor"):
        price_step.StepCubePrices(isolated, context.config).run(full=True)
    assert not writes
    print(f"SANITY: actual StepCubePrices.run refuses changed {changed} evidence before any part write.")


def test_carried_ownership_rebases_both_quantity_legs() -> None:
    events = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-06-01"]), "value": [0.5]})
    basis = ShareBasis(events, events, events, {"TEST": {}})
    index = pd.DatetimeIndex(["2020-05-28", "2020-05-29", "2020-06-01", "2020-06-02"])
    book = pd.DataFrame(
        {"ticker": ["TEST"], "period": pd.to_datetime(["2020-03-31"]), "as_of": pd.to_datetime(["2020-05-29"]), "inst_shares": [10.0]}
    )
    filing = book[["ticker", "as_of"]].assign(sharesOutstandingPit=100.0)
    owned = _daily_owned_shares(book, index, security_basis_frame(basis, pd.bdate_range("2020-03-31", index[-1]), pd.Index(["TEST"])))
    observations = filing.assign(capitalization_factor=1.0, capitalization_available=filing["as_of"])
    outstanding, _ = project_security_shares(observations, basis, security_basis_frame(basis, index, pd.Index(["TEST"])))
    assert pd.isna(owned.iloc[0, 0])
    assert owned["TEST"].iloc[1:].tolist() == [10.0, 5.0, 5.0]
    assert (owned / outstanding)["TEST"].iloc[1:].tolist() == [0.1, 0.1, 0.1]
    assert book["as_of"].iloc[0] == pd.Timestamp("2020-05-29")
    print("SANITY: a configured reverse split halves both carried ownership legs; ownership stays10% and the book never appears before publication.")


def test_level_diagnostic_counts_only_observed_quote_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.DataFrame({"TEST": [float("nan"), float("nan"), 5.0, 5.0]}, index=index)
    vendor = pd.DataFrame({"ticker": ["TEST"], "date": [index[-1]], "price": [10.0]})
    blob = {"level_factor": {"TEST": {"segments": [{"before": "2020-02-01", "factor": 2.0}]}}}
    messages: list[str] = []
    monkeypatch.setattr("src.utils.price.levels.logger.info", lambda msg, *args: messages.append(msg % args))
    factor = apply_level_bugfix(close * 0 + 1, blob, vendor, close)
    assert "APPLIED to 2 bar(s)" in messages[-1]
    assert factor["TEST"].dropna().eq(2.0).all()
    print("SANITY: a hidden wedge logs two observed quotes, excluding two prelisting NaN grid positions.")


def test_single_ticker_sjm_accepts_verified_empty_yahoo_source(source_case: tuple) -> None:
    context, _, _, blob = source_case
    actions = level_actions(context, ["SJM"])
    yahoo = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": ["SJM"]}, optional=True)
    assert yahoo is None and context.store.exists(Tables.prices_splits)
    basis = load_share_basis(context, ["SJM"], blob, actions, yahoo)
    value = basis.factor(pd.Series(["SJM"]), pd.Series(pd.to_datetime(["2002-03-11"])), "vendor").iloc[0]
    assert value == 0.945
    print("SANITY: real single-ticker SJM resolves its verified empty Yahoo feed; available table/absent rows is distinct from unavailable source.")
