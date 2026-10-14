"""Known-truth contracts for independent price, quantity and quote-unit views."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.utils.price_adjustments import resolve_share_events, validate_bugfix


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


def _factor(basis: object, kind: str, dates: list[str]) -> list[float]:
    return basis.factor(pd.Series(["TEST"] * len(dates)), pd.Series(pd.to_datetime(dates)), kind).tolist()


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


def test_vendor_exclusion_keeps_actual_stock_dividend() -> None:
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
    basis = resolve_share_events(actions, yahoo, _policy(event), shares=shares, tickers=["TEST"])
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
        {"date": "2020-06-15", "ordinary_shares": 100.0, "quoted_price": 10.0, "rtol": 0.0002, "source": "quote between old/new dates"}
    )
    shares = pd.DataFrame(
        {"ticker": ["TEST", "TEST", "TEST"], "date": pd.to_datetime(["2020-01-02", "2020-05-15", "2020-06-15"]), "sharesbas": [200.0, 200.0, 100.0]}
    )
    quotes = shares[["ticker", "date"]].assign(close_split=5.0)
    basis = resolve_share_events(actions, yahoo, blob, shares=shares, quotes=quotes)
    assert _factor(basis, "price", ["2020-06-15", "2020-07-01"]) == [2.0, 1.0]
    assert _factor(basis, "actual", ["2020-06-15"]) == [1.0]
    competing = pd.concat([yahoo, yahoo.assign(date=pd.Timestamp("2020-05-01" if healed else "2020-07-01"))], ignore_index=True)
    with pytest.raises(ValueError, match="changed yahoo"):
        resolve_share_events(actions, competing, blob, shares=shares, quotes=quotes)
    print("SANITY: independent price dates replace/heal once; competing old/new feed rows refuse.")
