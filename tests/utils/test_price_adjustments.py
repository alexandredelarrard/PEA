"""Known-truth contracts for independent price, quantity and quote-unit views."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.predecessor_series import load_level_actions
from src.utils.price_adjustments import ShareBasis, resolve_share_events, validate_bugfix
from src.utils.price_basis import load_share_basis


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(columns=["ticker", "date", "action", "value"]), pd.DataFrame(columns=["ticker", "date", "ratio"])


def _policy(event: dict, shares: float = 100.0, **extra: object) -> dict:
    return {
        "_APPROVED": {"date": "2026-10-14"},
        "share_events": {
            "TEST": {
                "anchors": [{"date": "2020-01-02", "ordinary_shares": shares, "rtol": 0.0002, "source": "known-truth original"}],
                "events": [event],
                **extra,
            }
        },
    }


def _factor(basis: ShareBasis, kind: str, dates: list[str]) -> list[float]:
    return basis.factor(pd.Series(["TEST"] * len(dates)), pd.Series(pd.to_datetime(dates)), kind).tolist()


@pytest.mark.parametrize("missing", [None, "yahoo", "sharadar"])
def test_actual_facade_distinguishes_absent_rows_from_missing_sources(sqlite_store: DataStore, missing: str | None) -> None:
    other = pd.DataFrame({"ticker": ["OTHER"], "date": pd.to_datetime(["2020-06-01"])})
    if missing != "sharadar":
        sqlite_store.save(Tables.sharadar_actions, other.assign(action="split", value=2.0, contraticker="OTHER2"))
    if missing != "yahoo":
        sqlite_store.save(Tables.prices_splits, other.assign(ratio=2.0))
    context = cast(Any, SimpleNamespace(store=sqlite_store, config_dir="./configs", log=SimpleNamespace(info=lambda *args: None)))
    actions = load_level_actions(context, ["TEST"])
    yahoo = sqlite_store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": ["TEST"]}, optional=True)
    assert yahoo is None
    event = {
        "expect": {"yahoo": None, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-06-01", "ratio": 2.0},
        "evidence": "known-truth absent feed rows",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"][0]["quoted_price"] = 10.0
    shares = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "sharesbas": [200.0]})
    quotes = shares[["ticker", "date"]].assign(close_split=5.0)
    if missing is None:
        basis = load_share_basis(context, ["TEST"], blob, actions, yahoo, shares=shares, quotes=quotes)
        assert _factor(basis, "vendor", ["2020-01-02"]) == [2.0]
        assert _factor(basis, "price", ["2020-01-02"]) == [2.0]
    else:
        with pytest.raises(ValueError, match=f"{missing} evidence unavailable"):
            load_share_basis(context, ["TEST"], blob, actions, yahoo, shares=shares, quotes=quotes)
    print(
        f"SANITY: actual isolated DataStore facade: missing={missing}; verified row absence can authorize an anchored event, missing tables refuse."
    )


@pytest.mark.parametrize("repair", ["legacy", "vintage"])
def test_quote_anchor_loader_uses_repaired_close_and_bounded_history(repair: str) -> None:
    actions, yahoo = _frames()
    event = {
        "expect": {"yahoo": None, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-06-01", "ratio": 2.0},
        "evidence": "known-truth missing factor on a separately repaired quote series",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"][0]["quoted_price"] = 10.0
    blob["return_seams"] = {"TEST": [{"date": "2020-01-06", "step": 0.5, "evidence": "known-truth legacy seam"}]}
    if repair == "vintage":
        blob.pop("return_seams")
        blob["split_vintage"] = {"TEST": [{"before": "2020-01-06", "ratio": 2.0, "expect_wedge": 1.0}]}
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    prices = pd.DataFrame({"ticker": "TEST", "date": dates, "close_split": [10.0 if stamp < pd.Timestamp("2020-01-06") else 5.0 for stamp in dates]})
    shares = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "sharesbas": [200.0]})
    reads: list[dict[str, Any]] = []

    class AnchorStore:
        def load(self, table: Any, **kwargs: Any) -> pd.DataFrame:
            reads.append(kwargs)
            assert kwargs["where"]["ticker"] == ["TEST"]
            if table == Tables.prices:
                assert pd.Timestamp(kwargs["until"]) >= pd.Timestamp("2020-01-06")
                return prices.copy()
            assert table == Tables.sharadar_fundamentals
            return pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "price": [10.0]})

    context = cast(Any, SimpleNamespace(store=AnchorStore(), log=SimpleNamespace(info=lambda *args: None)))
    first = load_share_basis(context, ["TEST"], blob, actions, yahoo, shares=shares)
    assert _factor(first, "price", ["2020-01-02"]) == [2.0]
    covered = prices.loc[prices["date"].eq(pd.Timestamp("2020-01-02"))].assign(close_split=5.0)
    second = load_share_basis(context, ["TEST"], blob, actions, yahoo, shares=shares, quotes=covered)
    pd.testing.assert_frame_equal(first.price, second.price)
    assert len(reads) == (2 if repair == "vintage" else 1)
    print(f"SANITY: {repair}: fetched and covered anchors verify the repaired quote; a tail build reads named history through its repair boundary.")


def test_unverified_hidden_wedge_cannot_authorize_a_price_override() -> None:
    actions, yahoo = _frames()
    event = {
        "expect": {"yahoo": None, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-06-01", "ratio": 2.0},
        "evidence": "known truth",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"][0]["quoted_price"] = 10.0
    blob["level_factor"] = {"TEST": {"segments": [{"before": "2020-03-01", "factor": 2.0}]}}
    shares = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "sharesbas": [200.0]})
    quotes = shares[["ticker", "date"]].assign(close_split=2.5)
    with pytest.raises(ValueError, match="hidden level"):
        resolve_share_events(actions, yahoo, blob, shares=shares, quotes=quotes)
    blob["share_events"]["CONTROL"] = deepcopy(blob["share_events"]["TEST"])
    included_shares = shares.assign(ticker="CONTROL")
    included_quotes = quotes.assign(ticker="CONTROL", close_split=5.0)
    accepted = resolve_share_events(actions, yahoo, blob, shares=included_shares, quotes=included_quotes, tickers=["CONTROL"])
    assert set(accepted.policies) == {"CONTROL"}
    print("SANITY: a declared hidden wedge cannot self-authorize a price-factor override without independent verification.")


def test_missing_adjusted_event_is_inserted_once() -> None:
    actions, yahoo = _frames()
    event = {
        "expect": {"yahoo": None, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-06-01", "ratio": 2.0},
        "evidence": "both histories already restated",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"][0]["quoted_price"] = 10.0
    shares = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "sharesbas": [200.0]})
    quotes = shares[["ticker", "date"]].assign(close_split=5.0)
    first = resolve_share_events(actions, yahoo, blob, shares=shares, quotes=quotes, tickers=["TEST"])
    second = resolve_share_events(actions, yahoo, blob, shares=shares, quotes=quotes, tickers=["TEST"], base_events=first.actual)
    assert len(first.actual) == len(second.actual) == 1
    for kind in ("actual", "vendor", "price"):
        assert _factor(first, kind, ["2020-01-02", "2020-06-01"]) == [2.0, 1.0]
    with pytest.raises(ValueError, match="price anchor"):
        resolve_share_events(actions, yahoo, blob, shares=shares, quotes=quotes.assign(close_split=10.0), tickers=["TEST"])
    print("SANITY: missing feed rows on adjusted bars resolve once; unadjusted bars fail the independent quote anchor.")


def test_vendor_exclusion_keeps_actual_stock_dividend(caplog: pytest.LogCaptureFixture) -> None:
    actions, _ = _frames()
    yahoo = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-06-01"]), "ratio": [1.1]})
    event = {
        "expect": {"yahoo": {"date": "2020-06-01", "ratio": 1.1}, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 1.1},
        "vendor": {"date": "2020-06-01", "ratio": 1.0},
        "evidence": "original count unchanged",
    }
    shares = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "sharesbas": [100.0]})
    snapshot = yahoo.copy(deep=True)
    with caplog.at_level("INFO", logger="src.utils.price_adjustments"):
        basis = resolve_share_events(actions, yahoo, _policy(event), shares=shares, tickers=["TEST"])
    assert "TEST VERIFIED configured overrides" in caplog.text
    assert _factor(basis, "actual", ["2020-01-02"]) == [1.1]
    assert _factor(basis, "vendor", ["2020-01-02"]) == [1.0]
    pd.testing.assert_frame_equal(yahoo, snapshot)
    with pytest.raises(ValueError, match="share anchor"):
        resolve_share_events(actions, yahoo, _policy(event), shares=shares.assign(sharesbas=110.0), tickers=["TEST"])
    with pytest.raises(ValueError, match="unavailable"):
        resolve_share_events(None, yahoo, _policy(event), shares=shares, tickers=["TEST"])
    print("SANITY: an unincorporated stock dividend changes actual units, not vendor PIT division; drift and unavailable evidence refuse.")


@pytest.mark.parametrize("healed", [False, True])
def test_moved_event_and_quote_units_have_independent_dates(healed: bool) -> None:
    _, yahoo = _frames()
    actions = pd.DataFrame(
        {"ticker": ["TEST"], "date": pd.to_datetime(["2020-06-01" if healed else "2020-05-01"]), "action": ["split"], "value": [0.5]}
    )
    event = {
        "expect": {"yahoo": None, "sharadar": {"date": "2020-05-01", "ratio": 0.5}},
        "actual": {"date": "2020-06-01", "ratio": 0.5},
        "evidence": "ordinary consolidation; quoted unit conversion",
    }
    blob = _policy(event, quote_units={"date": "2020-06-01", "before": 2.0, "after": 1.0, "quoted_titles": ["American Depositary Shares"]})
    blob["share_events"]["TEST"]["anchors"].append(
        {"date": "2020-05-15", "ordinary_shares": 100.0, "rtol": 0.0002, "source": "original between-date quantity"}
    )
    shares = pd.DataFrame({"ticker": ["TEST", "TEST"], "date": pd.to_datetime(["2020-01-02", "2020-05-15"]), "sharesbas": [50.0, 50.0]})
    basis = resolve_share_events(actions, yahoo, blob, shares=shares, tickers=["TEST"])
    assert len(basis.actual) == 1
    assert _factor(basis, "vendor", ["2020-05-15", "2020-06-01"]) == [0.5, 1.0]
    assert _factor(basis, "units", ["2020-05-15", "2020-06-01"]) == [2.0, 1.0]
    assert _factor(basis, "price", ["2020-05-15"]) == [1.0]
    print("SANITY: moved and exactly healed events replace once; quote units and Yahoo dates remain independent.")


@pytest.mark.parametrize("mutation", ["unknown", "negative", "duplicate"])
def test_new_contract_rejects_invalid_metadata(mutation: str) -> None:
    event = {"expect": {"yahoo": None, "sharadar": None}, "actual": {"date": "2020-06-01", "ratio": 2.0}, "evidence": "known truth"}
    blob = _policy(event)
    if mutation == "unknown":
        event["typo"] = 1
    elif mutation == "negative":
        event["actual"]["ratio"] = -1
    else:
        blob["share_events"]["TEST"]["events"].append(deepcopy(event))
    with pytest.raises(ValueError):
        validate_bugfix(blob)
    print(f"SANITY: {mutation} event metadata is refused before source reads.")


def test_missing_event_uses_observation_scope_without_explicit_tickers() -> None:
    actions, yahoo = _frames()
    event = {"expect": {"yahoo": None, "sharadar": None}, "actual": {"date": "2020-06-01", "ratio": 2.0}, "evidence": "missing action"}
    shares = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-01-02"]), "sharesbas": [200.0]})
    basis = resolve_share_events(actions, yahoo, _policy(event), shares=shares)
    assert _factor(basis, "vendor", ["2020-01-02"]) == [2.0]
    print("SANITY: an observed ticker with no source events still receives its missing-event policy.")


def test_price_override_requires_an_anchor_on_its_affected_prefix() -> None:
    event = {
        "expect": {"yahoo": None, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-07-01", "ratio": 2.0},
        "evidence": "independent quote date",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"][0].update(date="2020-08-01", quoted_price=10.0)
    with pytest.raises(ValueError, match="prefix"):
        validate_bugfix(blob)
    print("SANITY: post-event quotes cannot authorize an adjustment to earlier bars.")


def test_price_date_change_requires_an_anchor_between_the_dates() -> None:
    event = {
        "expect": {"yahoo": {"date": "2020-05-01", "ratio": 2.0}, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-07-01", "ratio": 2.0},
        "evidence": "moved price factor",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"][0]["quoted_price"] = 10.0
    with pytest.raises(ValueError, match="affected"):
        validate_bugfix(blob)
    print("SANITY: an anchor before both price cutoffs cannot verify their difference.")


@pytest.mark.parametrize("healed", [False, True])
def test_price_date_can_heal_independently_and_competing_rows_refuse(healed: bool) -> None:
    actions, _ = _frames()
    yahoo = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2020-07-01" if healed else "2020-05-01"]), "ratio": [2.0]})
    event = {
        "expect": {"yahoo": {"date": "2020-05-01", "ratio": 2.0}, "sharadar": None},
        "actual": {"date": "2020-06-01", "ratio": 2.0},
        "price": {"date": "2020-07-01", "ratio": 2.0},
        "evidence": "independent dates",
    }
    blob = _policy(event)
    blob["share_events"]["TEST"]["anchors"].append(
        {"date": "2020-05-15", "ordinary_shares": 100.0, "rtol": 0.0002, "source": "share count between vendor dates"}
    )
    blob["share_events"]["TEST"]["anchors"].append(
        {"date": "2020-06-15", "quoted_price": 10.0, "rtol": 0.0002, "source": "independent quote without an SF1 filing"}
    )
    blob["share_events"]["TEST"]["anchors"].append(
        {"date": "2020-06-20", "ordinary_shares": 100.0, "rtol": 0.0002, "source": "share count before the healed price date"}
    )
    shares = pd.DataFrame(
        {
            "ticker": ["TEST"] * 4,
            "date": pd.to_datetime(["2020-01-02", "2020-05-15", "2020-06-15", "2020-06-20"]),
            "sharesbas": [200.0, 200.0, 100.0, 100.0],
        }
    )
    quotes = shares[["ticker", "date"]].assign(close_split=5.0)
    shares = shares.loc[shares["date"].ne(pd.Timestamp("2020-06-15"))]
    basis = resolve_share_events(actions, yahoo, blob, shares=shares, quotes=quotes)
    assert _factor(basis, "price", ["2020-06-15", "2020-07-01"]) == [2.0, 1.0]
    assert _factor(basis, "actual", ["2020-06-15"]) == [1.0]
    competing = pd.concat([yahoo, yahoo.assign(date=pd.Timestamp("2020-05-01" if healed else "2020-07-01"))], ignore_index=True)
    with pytest.raises(ValueError, match="changed yahoo"):
        resolve_share_events(actions, competing, blob, shares=shares, quotes=quotes)
    print("SANITY: independent price dates replace/heal once; competing old/new feed rows refuse.")


def test_declared_quote_units_need_positive_title_evidence() -> None:
    events = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(
        events,
        events,
        events,
        {
            "TEST": {
                "quote_units": {
                    "date": "2020-06-01",
                    "before": 2.0,
                    "after": 1.0,
                    "quoted_titles": ["American Depositary Shares"],
                    "ordinary_titles": ["Ordinary Shares"],
                }
            }
        },
    )
    tickers = pd.Series(["TEST"] * 4)
    dates = pd.Series(pd.to_datetime(["2020-05-01"] * 3 + ["2020-06-01"]))
    titles = pd.Series([" american   depositary SHARES ", "Ordinary Shares", "Common Stock", "Common Stock"])
    units = basis.declared_units(tickers, dates, titles)
    assert units.iloc[:2].tolist() == [2.0, 1.0]
    assert pd.isna(units.iloc[2])
    assert units.iloc[3] == 1.0
    print("SANITY: ADS/ordinary titles resolve explicitly; generic common class cannot establish a nonunitary title's units.")


def test_quantity_rebase_uses_actual_events_only() -> None:
    df_actual = pd.DataFrame(
        {
            "ticker": ["FORWARD", "REVERSE", "MULTIPLE", "MULTIPLE"],
            "date": pd.to_datetime(["2024-06-10"] * 3 + ["2024-07-01"]),
            "value": [10, 0.125, 2, 3],
        }
    )
    df_price = pd.DataFrame({"ticker": ["PRICE_ONLY"], "date": pd.to_datetime(["2024-06-10"]), "value": [1.5]})
    basis = ShareBasis(df_actual, df_actual.assign(value=7), df_price, {})
    quantities = pd.Series([5.0, 80, 5, 5, 5, -5, 0, 5, 5], index=list("abcdefghi"))
    tickers = pd.Series(
        ["FORWARD", "REVERSE", "MULTIPLE", "FORWARD", "FORWARD", "FORWARD", "FORWARD", "PRICE_ONLY", "UNCHANGED"], index=quantities.index
    )
    source = pd.Series(pd.to_datetime(["2024-06-07"] * 3 + ["2024-06-10", "2024-06-11"] + ["2024-06-07"] * 4), index=quantities.index)
    target = pd.Series(pd.to_datetime(["2024-07-02"] * 4 + ["2024-06-07"] + ["2024-07-02"] * 4), index=quantities.index)
    result = basis.rebase(quantities, tickers, source, target)
    pd.testing.assert_series_equal(result, pd.Series([50.0, 10, 30, 5, 0.5, -50, 0, 5, 5], index=quantities.index))
    pd.testing.assert_series_equal(basis.rebase(result, tickers, target, target), result)
    assert quantities.iloc[0] == 5
    print("SANITY: actual forward/reverse/successive splits, post-split ex-date and identity agree; vendor/price-only factors never create shares.")


def test_quantity_rebase_preserves_unknowns_and_boundaries() -> None:
    basis = resolve_share_events(None, None, {})
    index = pd.Index([9, 4, 4, 8, 2, 1, 7, 3, 6])
    quantities = pd.Series([5.0, 5, 0, 0, np.inf, np.nan, 5, -5, 0], index=index)
    tickers = pd.Series(["TEST"] * 7 + [None, ""], index=index)
    source = pd.Series(pd.to_datetime(pd.Series(["2024-06-07"] * 3 + [None] + ["2024-06-07"] * 5)).to_numpy(), index=index)
    target = pd.Series(pd.to_datetime(["2024-06-07", "2024-06-10", "2024-06-10"] + ["2024-06-07"] * 6), index=index)
    result = basis.rebase(quantities, tickers, source, target)
    expected = pd.Series([5.0, np.nan, 0, np.nan, np.nan, np.nan, 5, np.nan, np.nan], index=index)
    pd.testing.assert_series_equal(result, expected)
    df_empty = pd.DataFrame(columns=["ticker", "date", "ratio"])
    known = resolve_share_events(None, df_empty, {})
    assert known.rebase(quantities, tickers, source, target).iloc[1] == 5
    assert not np.isinf(result.to_numpy()).any()
    for invalid_tickers in (pd.Series([123]), pd.Series([True]), pd.Series(["TEST", 123])):
        values = pd.Series(0.0, index=invalid_tickers.index)
        dates = pd.Series(pd.Timestamp("2024-06-07"), index=invalid_tickers.index)
        assert pd.isna(basis.rebase(values, invalid_tickers, dates, dates).iloc[-1])
    with pytest.raises(ValueError, match="aligned"):
        basis.rebase(quantities, tickers.reset_index(drop=True), source, target)
    print(
        "SANITY: missing event sources preserve valid identities and zero; unknown nonzero intervals, invalid inputs and misalignment never invent a factor."
    )


@pytest.mark.parametrize("factor", [0.0, -2.0, np.inf, np.nan])
def test_quantity_rebase_refuses_invalid_actual_evidence(factor: float) -> None:
    df_actual = pd.DataFrame({"ticker": ["TEST"], "date": pd.to_datetime(["2024-06-10"]), "value": [factor]})
    basis = ShareBasis(df_actual, df_actual, df_actual, {})
    with pytest.raises(ValueError, match="actual.*positive|positive.*actual"):
        basis.rebase(pd.Series([5.0]), pd.Series(["TEST"]), pd.Series(pd.to_datetime(["2024-06-07"])), pd.Series(pd.to_datetime(["2024-06-11"])))
    print(f"SANITY: invalid genuine split factor {factor} is refused before quantity arithmetic.")
