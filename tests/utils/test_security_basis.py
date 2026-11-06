"""Known-truth controls for explicit security units, money and dated cap support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import get_config_context
from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.price.capitalization import capitalization_observations, project_security_shares
from src.data_aggregate.utils.price.features import as_traded_volume, security_basis_frame
from src.data_store.schema import Tables
from src.utils.price.adjustments import ShareBasis
from src.utils.price.levels import convert_quantities, rebase_security_quantities, vendor_observation_factor


def _basis(ratio: float, before: float, after: float) -> ShareBasis:
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2024-01-05"]), "value": [ratio]})
    policy = {"AAA": {"quote_units": {"date": "2024-01-05", "before": before, "after": after}}}
    return ShareBasis(df_events, df_events.assign(value=7.0), df_events.assign(value=3.0), policy)


@pytest.mark.parametrize(
    ("ratio", "before", "after", "source_units", "expected"),
    [
        (4.0, 1.0, 1.0, 1.0, 40.0),
        (0.25, 1.0, 1.0, 1.0, 2.5),
        (1.1, 1.0, 1.0, 1.0, 11.0),
        (1.0, 2.0, 4.0, 2.0, 5.0),
        (1.0, 2.0, 4.0, 1.0, 2.5),
        (0.5, 2.0, 1.0, 2.0, 10.0),
    ],
)
def test_declared_units_preserve_money_and_do_not_use_vendor_price_factors(
    ratio: float, before: float, after: float, source_units: float, expected: float
) -> None:
    basis = _basis(ratio, before, after)
    quantity = pd.Series([10.0])
    ticker = pd.Series(["AAA"])
    source = pd.Series(pd.to_datetime(["2024-01-04"]))
    target = pd.Series(pd.to_datetime(["2024-01-05"]))
    units = pd.Series([source_units])
    rebased = rebase_security_quantities(basis, quantity, ticker, source, target, units)
    assert rebased.iloc[0] == pytest.approx(expected)
    roundtrip = rebase_security_quantities(basis, rebased, ticker, target, source, pd.Series([after]))
    assert roundtrip.iloc[0] == pytest.approx(10.0 * source_units / before)
    print("SANITY: genuine event and declared quote units convert the quantity; vendor F=7 and price Y=3 never become quantities.")


def test_unknown_units_and_unavailable_cross_date_evidence_remain_unknown() -> None:
    basis = _basis(4, 1, 1)
    ticker = pd.Series(["AAA"] * 3)
    source = pd.Series(pd.to_datetime(["2024-01-04"] * 3))
    target = pd.Series(pd.to_datetime(["2024-01-05"] * 3))
    result = rebase_security_quantities(basis, pd.Series([10.0, 10.0, 10.0]), ticker, source, target, pd.Series([np.nan, 0.0, -1.0]))
    assert result.isna().all()
    absent = ShareBasis(basis.actual, basis.vendor, basis.price, {}, actual_available=False)
    result = rebase_security_quantities(absent, pd.Series([10.0]), ticker.iloc[:1], source.iloc[:1], target.iloc[:1], pd.Series([1.0]))
    assert result.isna().all()
    print("SANITY: unknown/invalid units and unavailable nonzero cross-date evidence fail closed.")


def test_successive_and_future_splits_preserve_historical_ownership() -> None:
    df_events = pd.DataFrame({"ticker": ["AAA", "AAA"], "date": pd.to_datetime(["2024-01-05", "2024-01-10"]), "value": [4.0, 0.5]})
    basis = ShareBasis(df_events, df_events, df_events, {})
    quantity = pd.Series([10.0, 100.0])
    ticker = pd.Series(["AAA", "AAA"])
    source = pd.Series(pd.to_datetime(["2024-01-04"] * 2))
    target = pd.Series(pd.to_datetime(["2024-01-10"] * 2))
    units = pd.Series([1.0, 1.0])
    result = rebase_security_quantities(basis, quantity, ticker, source, target, units)
    assert result.tolist() == [20.0, 200.0]
    future = pd.concat([df_events, pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2027-01-05"]), "value": [10.0]})], ignore_index=True)
    later = ShareBasis(future, future, future, {})
    pd.testing.assert_series_equal(result, rebase_security_quantities(later, quantity, ticker, source, target, units), check_exact=True)
    assert result.iloc[0] / result.iloc[1] == 0.1
    with pytest.raises(ValueError, match="index-aligned"):
        rebase_security_quantities(basis, quantity, ticker, source, target, units.rename(index={0: 3}))
    print(
        "SANITY: a forward/reverse sequence preserves 10% ownership; a future split leaves historical quantities exactly invariant; misaligned units refuse."
    )


def test_capitalization_uses_dated_exact_lineage_and_unknown_successor_ends_support() -> None:
    basis = _basis(1, 1, 1)
    basis = ShareBasis(basis.actual, basis.actual, basis.price, basis.policies)
    index = pd.bdate_range("2024-01-01", "2024-01-12")
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA"] * 2,
            "as_of": pd.to_datetime(["2024-01-02", "2024-01-08"]),
            "fiscal_end": pd.to_datetime(["2023-09-30", "2023-12-31"]),
            "sharesOutstanding": [100.0, 120.0],
            "sharesOutstandingPit": [100.0, 120.0],
        }
    )
    df_vendor = pd.DataFrame(
        {
            "ticker": ["AAA"] * 2,
            "date": df_history["as_of"],
            "reportperiod": df_history["fiscal_end"],
            "sharesbas": [100.0, 999.0],
            "sharefactor": [1.5, 1.7],
            "lastupdated": pd.to_datetime(["2024-01-03", "2024-01-08"]),
            "marketcap": [7.0, 7.0],
        }
    )
    original = df_history.copy(deep=True)
    observations = capitalization_observations(df_history, df_vendor, basis)
    shares, scale = project_security_shares(observations, basis, security_basis_frame(basis, index, pd.Index(["AAA"])))
    assert scale.loc[:"2024-01-01", "AAA"].isna().all()
    assert scale.loc["2024-01-02":"2024-01-05", "AAA"].eq(1.5).all()
    assert scale.loc["2024-01-08":, "AAA"].isna().all()
    assert shares.at[pd.Timestamp("2024-01-08"), "AAA"] == 120.0
    pd.testing.assert_frame_equal(df_history, original, check_exact=True)
    print(
        "SANITY: K is public on its filing date (a later lastupdated does not delay it); "
        "mismatched successor stops old K, preserves actual count, and no reference cap fits the coefficient."
    )


def test_real_sharadar_class_scope_and_public_support() -> None:
    _, context = get_config_context("./configs", use_cache=False, save=False)
    names = ["BRK-B", "V"]
    df_history = context.store.load(
        Tables.fundamentals_history,
        columns=["ticker", "as_of", "fiscal_end", "sharesOutstanding", "sharesOutstandingPit"],
        where={"ticker": names},
        since="2026-01-01",
        until="2026-10-26",
    )
    df_vendor = context.store.load(
        Tables.sharadar_fundamentals,
        columns=["ticker", "date", "reportperiod", "sharesbas", "sharefactor", "lastupdated", "price", "marketcap"],
        where={"ticker": names, "dimension": "ARQ"},
        since="2026-01-01",
        until="2026-10-26",
    )
    df_calendar = context.store.load(
        Tables.prices_macro, columns=["date"], where={"ticker": MACRO_MARKET_SERIES}, since="2026-01-01", until="2026-10-26"
    )
    assert df_history is not None and df_vendor is not None and df_calendar is not None
    calendar = pd.DatetimeIndex(pd.to_datetime(df_calendar["date"]).sort_values())
    df_empty = pd.DataFrame({"ticker": pd.Series(dtype="str"), "date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})
    basis = ShareBasis(df_empty, df_empty, df_empty, {})
    observations = capitalization_observations(df_history, df_vendor, basis)
    shares, scale = project_security_shares(observations, basis, security_basis_frame(basis, calendar, pd.Index(names)))
    df_latest = observations.sort_values("as_of").groupby("ticker").tail(1)
    for ticker in names:
        row = df_latest.loc[df_latest["ticker"].eq(ticker)].iloc[0]
        native = df_vendor.loc[df_vendor["ticker"].eq(ticker) & pd.to_datetime(df_vendor["date"]).eq(row["as_of"])].iloc[-1]
        calculated = native["price"] * native["sharesbas"] * native["sharefactor"]
        tolerance = 0.0005 / native["sharefactor"] + 0.005 / native["price"] + 0.5 / native["marketcap"]
        assert abs(calculated / native["marketcap"] - 1) <= tolerance
        public = min(day for day in calendar if day >= row["as_of"])
        assert scale.loc[calendar < row["as_of"], ticker].ne(native["sharefactor"]).all()
        assert scale.at[public, ticker] == native["sharefactor"]
        assert shares.at[public, ticker] == row["sharesOutstandingPit"]
        assert scale.at[public, ticker] > 1
    print(
        "SANITY: real BRK-B/V issuer-cap equations agree within source rounding; the class factor is public on its filing session "
        "and never inflates the saved class count."
    )


def test_unmatched_vendor_vintages_require_corroborated_original_counts() -> None:
    basis = _basis(4, 1, 1)
    basis = ShareBasis(basis.actual, basis.actual, basis.price, basis.policies)
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "as_of": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "fiscal_end": pd.to_datetime(["2023-09-30"] * 2),
            "sharesOutstanding": [100.0, 100.0],
            "sharesOutstandingPit": [100.0, 50.0],
        }
    )
    df_raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": df_history["as_of"],
            "reportperiod": df_history["fiscal_end"],
            "sharesbas": [400.0, 400.0],
            "sharefactor": [1.0, 1.0],
            "lastupdated": pd.to_datetime(["2024-01-08"] * 2),
        }
    )
    observations = capitalization_observations(df_history, df_raw, basis)
    assert observations["sharesOutstandingPit"].iloc[0] == 100.0
    assert pd.isna(observations["sharesOutstandingPit"].iloc[1])
    assert observations["capitalization_factor"].isna().all()
    print(
        "SANITY: current raw400/F4 corroborates original100 despite old merged100; neither vendor observation corroborates cached50, so it remains unknown."
    )


def test_identity_vendor_event_needs_no_yahoo_price_factor() -> None:
    from src.data_aggregate.utils.price.checks import verified_price_factor_support

    basis = _basis(1, 1, 1)
    empty = basis.actual.iloc[:0]
    basis = ShareBasis(basis.actual, basis.actual, empty, {})
    df_close = pd.DataFrame({"AAA": [50.0, 50.0]}, index=pd.DatetimeIndex(["2024-01-04", "2024-01-05"]))
    assert verified_price_factor_support(df_close, basis).all().all()
    print("SANITY: a no-op vendor event does not turn a known quote into missing data.")


def test_vendor_observation_factor_requires_an_evidenced_vintage() -> None:
    basis = _basis(4, 1, 1)
    names = pd.Series(["AAA"] * 4)
    dates = pd.Series(pd.to_datetime(["2024-01-04"] * 4))
    vintages = pd.Series(pd.to_datetime(["2024-01-04", "2024-01-05", None, "2024-01-03"]))
    observed = vendor_observation_factor(basis, names, dates, vintages)
    assert observed.iloc[:2].tolist() == [1.0, 7.0]
    assert observed.iloc[2:].isna().all()
    with pytest.raises(ValueError, match="index-aligned"):
        vendor_observation_factor(basis, names, dates, vintages.rename(index={0: 9}))
    malformed = ShareBasis(basis.actual, basis.vendor.assign(value=-1.0), basis.price, {})
    with pytest.raises(ValueError, match="positive finite"):
        vendor_observation_factor(malformed, names, dates, vintages)
    print(
        "SANITY: vendor F7 enters only from its effective-day representation; missing/reversed vintages and malformed factors never become evidence."
    )


def test_cancelled_future_events_do_not_certify_an_unmatched_cached_count() -> None:
    df_events = pd.DataFrame({"ticker": ["AAA"] * 2, "date": pd.to_datetime(["2024-01-03", "2024-01-10"]), "value": [2.0, 0.5]})
    basis = ShareBasis(df_events, df_events, df_events, {})
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "as_of": pd.to_datetime(["2024-01-02"]),
            "fiscal_end": pd.to_datetime(["2023-09-30"]),
            "sharesOutstanding": [200.0],
            "sharesOutstandingPit": [200.0],
        }
    )
    df_raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": df_history["as_of"],
            "reportperiod": df_history["fiscal_end"],
            "sharesbas": [100.0],
            "sharefactor": [1.0],
            "lastupdated": pd.to_datetime(["2024-01-05"]),
        }
    )
    observations = capitalization_observations(df_history, df_raw, basis)
    assert observations["sharesOutstandingPit"].isna().all()
    assert observations["capitalization_factor"].isna().all()
    print("SANITY: cancelling future2× and0.5× events do not prove the representation of an unmatched cached count.")


def test_count_vintage_remains_supported_without_capitalization_coefficient() -> None:
    df_events = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2024-01-05"]), "value": [4.0]})
    basis = ShareBasis(df_events, df_events, df_events, {})
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "as_of": pd.to_datetime(["2024-01-02"]),
            "fiscal_end": pd.to_datetime(["2023-09-30"]),
            "sharesOutstanding": [400.0],
            "sharesOutstandingPit": [100.0],
        }
    )
    df_raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": df_history["as_of"],
            "reportperiod": df_history["fiscal_end"],
            "sharesbas": [400.0],
            "lastupdated": pd.to_datetime(["2024-01-08"]),
        }
    )
    observations = capitalization_observations(df_history, df_raw, basis)
    assert observations["sharesOutstandingPit"].iloc[0] == 100.0
    assert observations["capitalization_factor"].isna().all()
    print("SANITY: missing issuer-cap coefficient keeps supported count restoration and unknown capitalization independent.")


@pytest.mark.parametrize("sentinel", [0.0, -0.5])
def test_nonpositive_counts_cannot_corroborate_a_positive_cache(sentinel: float) -> None:
    df_events = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(df_events, df_events, df_events, {})
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "as_of": pd.to_datetime(["2024-01-02"]),
            "fiscal_end": pd.to_datetime(["2023-09-30"]),
            "sharesOutstanding": [sentinel],
            "sharesOutstandingPit": [0.5],
        }
    )
    df_raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": df_history["as_of"],
            "reportperiod": df_history["fiscal_end"],
            "sharesbas": [sentinel],
            "sharefactor": [1.0],
            "lastupdated": df_history["as_of"],
        }
    )
    observations = capitalization_observations(df_history, df_raw, basis)
    assert observations["sharesOutstandingPit"].isna().all()
    print("SANITY: one-share rounding never lets a zero/negative source sentinel establish a positive outstanding count.")


@pytest.mark.parametrize("period", ["invalid", None])
def test_missing_fiscal_keys_cannot_establish_or_extend_capitalization_support(period: str | None) -> None:
    df_empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(df_empty, df_empty, df_empty, {})
    calendar = pd.bdate_range("2024-01-01", "2024-01-12")
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "as_of": pd.to_datetime(["2024-01-02", "2024-01-08"]),
            "fiscal_end": ["2023-09-30", period],
            "sharesOutstanding": [100.0, 120.0],
            "sharesOutstandingPit": [100.0, 120.0],
        }
    )
    df_raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": df_history["as_of"],
            "reportperiod": df_history["fiscal_end"],
            "sharesbas": [100.0, 120.0],
            "sharefactor": [1.5, 1.7],
            "lastupdated": pd.to_datetime(["2024-01-02", "2024-01-08"]),
        }
    )
    observations = capitalization_observations(df_history, df_raw, basis)
    shares, scale = project_security_shares(observations, basis, security_basis_frame(basis, calendar, pd.Index(["AAA"])))
    assert scale.loc["2024-01-03":"2024-01-05", "AAA"].eq(1.5).all()
    assert scale.loc["2024-01-08":, "AAA"].isna().all()
    assert shares.loc["2024-01-08":, "AAA"].eq(120.0).all()
    print(
        "SANITY: invalid fiscal keys never match through NaT; the malformed successor ends oldK while its independently known count remains usable."
    )


def _events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame({"ticker": [r[0] for r in rows], "date": pd.to_datetime([r[1] for r in rows]), "value": [r[2] for r in rows]})


# Filed split events (effective first session on the new basis) and the IVZ ADS-to-common conversion.
_FILED = [
    ("GOOGL", "2014-04-03", 2.002),
    ("GOOGL", "2022-07-18", 20.0),
    ("AMZN", "1998-06-02", 2.0),
    ("AMZN", "1999-01-05", 3.0),
    ("AMZN", "1999-09-02", 2.0),
    ("AMZN", "2022-06-06", 20.0),
    ("IVZ", "2007-12-04", 0.5),
]
_IVZ_UNITS = {"IVZ": {"quote_units": {"date": "2007-12-04", "before": 2.0, "after": 1.0, "quoted_titles": ["American Depositary Shares"]}}}


def test_forward_basis_steps_on_effective_dates_and_folds_quote_units() -> None:
    df_events = _events(_FILED)
    basis = ShareBasis(df_events, df_events, df_events, _IVZ_UNITS)
    index = pd.DatetimeIndex(
        ["1998-01-02", "1999-01-04", "1999-01-05", "2007-12-03", "2007-12-04", "2014-04-02", "2014-04-03", "2022-07-15", "2022-07-18"]
    )
    df_basis = security_basis_frame(basis, index, pd.Index(["AMZN", "GOOGL", "IVZ", "AAA"]))
    assert df_basis.iloc[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert df_basis.at[pd.Timestamp("2022-07-15"), "GOOGL"] == 2.002 and df_basis.at[pd.Timestamp("2022-07-18"), "GOOGL"] == 2.002 * 20.0
    assert df_basis.loc[["1999-01-04", "1999-01-05", "2022-07-18"], "AMZN"].tolist() == [2.0, 6.0, 12.0 * 20.0]
    assert df_basis["IVZ"].eq(1.0).all() and df_basis["AAA"].eq(1.0).all()
    quantity = convert_quantities(100.0, df_basis.at[pd.Timestamp("2022-07-15"), "GOOGL"], df_basis.at[pd.Timestamp("2022-07-18"), "GOOGL"])
    assert quantity == pytest.approx(2000.0, rel=1e-12)
    unknown = ShareBasis(df_events, df_events, df_events, {}, unknown_units=frozenset({"IVZ"}))
    assert security_basis_frame(unknown, index, pd.Index(["IVZ"]))["IVZ"].isna().all()
    print(
        "SANITY: F jumps x20 on GOOGL 2022-07-18 and x3 on AMZN 1999-01-05 (F=240 after 2022), IVZ 0.5 consolidation x ADS 2->1 is continuous, "
        "no-event names stay 1, unknown units are null; 100 pre-split GOOGL shares are 2,000 after."
    )


def test_forward_basis_history_is_unchanged_by_a_later_event() -> None:
    index = pd.bdate_range("2022-06-01", "2022-08-31")
    df_events = _events(_FILED)
    before = security_basis_frame(ShareBasis(df_events, df_events, df_events, {}), index, pd.Index(["GOOGL", "AMZN"]))
    df_later = _events([*_FILED, ("GOOGL", "2022-08-01", 3.0)])
    after = security_basis_frame(ShareBasis(df_later, df_later, df_later, {}), index, pd.Index(["GOOGL", "AMZN"]))
    prefix = index < pd.Timestamp("2022-08-01")
    pd.testing.assert_frame_equal(before.loc[prefix], after.loc[prefix], check_exact=True)
    assert after.loc[~prefix, "GOOGL"].div(before.loc[~prefix, "GOOGL"]).eq(3.0).all()
    print(f"SANITY: adding a later event leaves all {int(prefix.sum())} earlier F rows bit-identical and multiplies only rows from its date.")


def test_asof_lookup_uses_last_grid_date_and_never_fills_a_null() -> None:
    index = pd.DatetimeIndex(["2022-07-14", "2022-07-15", "2022-07-18"])
    df_values = pd.DataFrame({"GOOGL": [1.0, 1.0, 20.0], "AAA": [3.0, np.nan, 4.0]}, index=index)
    tickers = pd.Series(["GOOGL", "GOOGL", "GOOGL", "AAA", "AAA", "ZZZ"])
    dates = pd.Series(pd.to_datetime(["2022-07-16", "2022-07-17", "2022-07-13", "2022-07-16", "2022-07-18", "2022-07-18"]))
    result = lookup_asof(df_values, tickers, dates)
    assert result.iloc[:2].tolist() == [1.0, 1.0] and result.iloc[2:4].isna().all() and result.iloc[4] == 4.0 and np.isnan(result.iloc[5])
    print(
        "SANITY: a weekend reads Friday's F (not Monday's x20); a null Friday stays null (no fill from Thursday); before-grid and unknown names are null."
    )


def test_volume_pit_is_as_traded_and_null_where_the_tape_is_restated_differently() -> None:
    index = pd.bdate_range("2022-07-13", "2022-07-20")
    df_volume = pd.DataFrame({"AAA": 1_000.0, "BBB": 1_000.0}, index=index)
    df_actual = _events([("AAA", "2022-07-18", 20.0), ("BBB", "2022-07-18", 20.0)])
    df_vendor = _events([("AAA", "2022-07-18", 20.0), ("BBB", "2022-07-18", 10.0)])
    basis = ShareBasis(df_actual, df_vendor, df_actual, {})
    df_basis = security_basis_frame(basis, index, pd.Index(["AAA", "BBB"]))
    result = as_traded_volume(df_volume, df_basis, basis)
    split = index >= pd.Timestamp("2022-07-18")
    assert result.loc[~split, "AAA"].eq(50.0).all() and result.loc[split, "AAA"].eq(1_000.0).all()
    assert result.loc[~split, "BBB"].isna().all() and result.loc[split, "BBB"].eq(1_000.0).all()
    print(
        "SANITY: a x20-restated tape returns to 50 as-traded shares before the split; a tape restated by a different vendor event is null before it."
    )
