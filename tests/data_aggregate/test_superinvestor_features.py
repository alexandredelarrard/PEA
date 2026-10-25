"""Known-truth behavioral controls through the shared production science engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import BookSupport, PublicBook, PublicBookSnapshot
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import (
    EMISSION,
    SuperinvestorFeatureHistory,
    compute_superinvestor_feature_history,
    project_superinvestor_features,
)
from src.data_aggregate.utils.sec_thirteen.superinvestor_metrics import BehaviorMarket
from src.utils.price_adjustments import ShareBasis


def _book(cik: str, period: str, positions: dict[str, tuple[float, float]], known: bool = True) -> PublicBook:
    rows = [
        {
            "cusip": ticker,
            "ticker": ticker,
            "shares": shares,
            "value_usd": value,
            "position_type": "common",
            "class_evidence": '["common"]',
            "value_basis_shares_source": "filed",
            "value_basis_inverse_unit_unverified": False,
        }
        for ticker, (shares, value) in positions.items()
    ]
    columns = [
        "cusip",
        "ticker",
        "shares",
        "value_usd",
        "position_type",
        "class_evidence",
        "value_basis_shares_source",
        "value_basis_inverse_unit_unverified",
    ]
    origin = pd.Timestamp(period) + pd.Timedelta(days=46)
    support = BookSupport(book_known=known, zero_eligible=known, comparison_eligible=known, restricted=False, resolution_status="complete")
    return PublicBook(
        cik=cik,
        period=pd.Timestamp(period),
        public_at=origin,
        first_public_at=origin,
        df_positions=pd.DataFrame(rows, columns=columns),
        support=support,
    )


def _snapshot(books: list[PublicBook], day: str = "2024-08-15") -> PublicBookSnapshot:
    archive = {(book.cik, book.period): book for book in books}
    latest = {}
    for book in sorted(books, key=lambda item: item.period):
        latest[book.cik] = book
    return PublicBookSnapshot(
        day=pd.Timestamp(day),
        common_period=max(book.period for book in books),
        latest_period_by_manager={cik: book.period for cik, book in latest.items()},
        level_books=latest,
        archive=archive,
    )


def _market(split: bool = False) -> BehaviorMarket:
    index = pd.bdate_range("2022-01-03", "2025-01-31")
    quotes = pd.DataFrame(10.0, index=index, columns=["AAA", "BBB", "OUT"])
    actions = (
        pd.DataFrame([{"ticker": "AAA", "date": pd.Timestamp("2024-05-01"), "value": 2.0}])
        if split
        else pd.DataFrame(columns=["ticker", "date", "value"])
    )
    basis = ShareBasis(actual=actions, vendor=actions, price=actions, policies={})
    return BehaviorMarket(index, quotes, basis)


def _compute(snapshot: PublicBookSnapshot, market: BehaviorMarket | None = None, topk: frozenset[str] | None = None) -> SuperinvestorFeatureHistory:
    roster = frozenset(snapshot.latest_period_by_manager)
    cohorts = {"": roster, "topk": roster if topk is None else topk}
    return compute_superinvestor_feature_history((snapshot,), cohorts, ("AAA", "BBB"), market or _market())


def _value(history: SuperinvestorFeatureHistory, metric: str, ticker: str = "AAA", cohort: str = "") -> float:
    prefix = f"ic_super_{cohort}_" if cohort else "ic_super_"
    return float(history.levels[prefix + metric].iloc[-1][ticker])


def test_equal_manager_weight_and_pooled_weight_have_distinct_denominators() -> None:
    books = [_book("1", "2024-06-30", {"AAA": (1, 10), "BBB": (9, 90)}), _book("2", "2024-06-30", {"AAA": (1.8, 18), "BBB": (88.2, 882)})]
    history = _compute(_snapshot(books))
    assert _value(history, "holders") == 2
    assert _value(history, "conviction_weight") == pytest.approx(0.06)
    assert _value(history, "pooled_weight") == pytest.approx(0.028)
    assert _value(history, "max_conviction") == pytest.approx(0.10)
    print("SANITY: two unequal books give6% equal-manager mean,2.8% pooled allocation and10% maximum weight.")


def test_identical_masks_have_all_13_identical_values_and_nulls() -> None:
    history = _compute(_snapshot([_book("1", "2024-06-30", {"AAA": (1, 10)})]))
    assert len(EMISSION) == 26
    for name, values in history.levels.items():
        if not name.startswith("ic_super_topk_"):
            pd.testing.assert_frame_equal(values, history.levels[name.replace("ic_super_", "ic_super_topk_")])
    print("SANITY: identical raw-CIK masks produce exact equality for every metric and unknown.")


def test_prices_do_not_create_trades_or_change_fixed_price_allocations() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000), "BBB": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (100, 2000), "BBB": (100, 1000)})
    history = _compute(_snapshot([prior, current]))
    for metric in ("shares_chg", "breadth_chg", "relative_position_chg", "pooled_weight_chg"):
        assert _value(history, metric) == pytest.approx(0)
    assert np.isnan(_value(history, "trade_balance"))
    assert _value(history, "conviction_weight") == pytest.approx(2 / 3)
    print("SANITY: a pure price double changes reported weight only; supported behavior is zero and no trader vote exists.")


def test_genuine_split_rebases_prior_shares_and_reference_price_inversely() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000), "BBB": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (300, 1500), "BBB": (100, 1000)})
    history = _compute(_snapshot([prior, current]), _market(split=True))
    assert _value(history, "shares_chg") == pytest.approx(0.5)
    assert _value(history, "relative_position_chg") == pytest.approx(0.25)
    assert _value(history, "pooled_weight_chg") == pytest.approx(0.10)
    assert _value(history, "trade_balance") == 1
    print("SANITY:2:1 split plus50% active share growth gives25pp relative activity and10pp pooled allocation change.")


def test_disappearing_manager_is_excluded_from_both_endpoints() -> None:
    books = [
        _book("1", "2024-03-31", {"AAA": (100, 1000)}),
        _book("2", "2024-03-31", {"AAA": (1000, 10000)}),
        _book("1", "2024-06-30", {"AAA": (110, 1100)}),
    ]
    snapshot = _snapshot(books)
    snapshot = replace(snapshot, level_books={"1": books[-1]})
    history = _compute(snapshot)
    assert _value(history, "holders") == 1
    assert _value(history, "shares_chg") == pytest.approx(0.1)
    assert history.diagnostics.quarter_pairs.tolist() == [1, 1]
    print("SANITY: the absent manager's1000shares never become a sale; the one genuine pair grows10%.")


def test_complete_empty_book_proves_share_exit_but_not_zero_pooled_denominator() -> None:
    history = _compute(_snapshot([_book("1", "2024-03-31", {"AAA": (100, 1000)}), _book("1", "2024-06-30", {})]))
    assert _value(history, "shares_chg") == -1
    assert _value(history, "trade_balance") == -1
    assert _value(history, "holders") == 0
    assert _value(history, "top10_holders") == 0
    assert np.isnan(_value(history, "pooled_weight"))
    assert np.isnan(_value(history, "pooled_weight_chg"))
    print("SANITY: a complete empty book supports a100% exit; its zero whole-book denominator remains unknown.")


def test_unknown_or_partial_book_never_manufactures_a_sale() -> None:
    books = [_book("1", "2024-03-31", {"AAA": (100, 1000)}), _book("1", "2024-06-30", {}, known=False)]
    history = _compute(_snapshot(books))
    assert np.isnan(_value(history, "holders"))
    assert np.isnan(_value(history, "shares_chg"))
    print("SANITY: unusable current books remain unknown rather than zero holdings or sales.")


def test_unselected_estimated_quantities_do_not_contaminate_topk() -> None:
    books = [
        _book("1", "2024-03-31", {"AAA": (100, 1000)}),
        _book("1", "2024-06-30", {"AAA": (110, 1100)}),
        _book("2", "2024-03-31", {"AAA": (100, 1000)}),
        _book("2", "2024-06-30", {"AAA": (999, 9990)}),
    ]
    books[-1].df_positions["value_basis_shares_source"] = "value_estimated"
    history = _compute(_snapshot(books), topk=frozenset({"1"}))
    assert _value(history, "shares_chg", cohort="topk") == pytest.approx(0.1)
    assert _value(history, "shares_chg") == pytest.approx(0.1)
    print("SANITY: estimated shares exclude their manager/ticker comparison, not the selected known pair.")


def test_same_quarter_watermark_does_not_reuse_old_topk_changes() -> None:
    books = [
        _book("1", "2023-12-31", {"AAA": (100, 1000)}),
        _book("1", "2024-03-31", {"AAA": (110, 1100)}),
        _book("2", "2024-06-30", {"AAA": (100, 1000)}),
    ]
    history = _compute(_snapshot(books), topk=frozenset({"1"}))
    assert _value(history, "holders", cohort="topk") == 1
    assert np.isnan(_value(history, "shares_chg", cohort="topk"))
    print("SANITY: carried old books may supply levels but cannot pose as the activated common report quarter.")


def test_annual_comparison_uses_exact_lag4_and_tenure_resets_on_a_gap() -> None:
    books = [
        _book("1", "2023-06-30", {"AAA": (100, 1000)}),
        _book("1", "2023-12-31", {"AAA": (150, 1500)}),
        _book("1", "2024-03-31", {"AAA": (180, 1800)}),
        _book("1", "2024-06-30", {"AAA": (200, 2000)}),
    ]
    history = _compute(_snapshot(books))
    assert _value(history, "shares_chg_yoy") == 1
    assert _value(history, "holders_yoy") == 0
    assert _value(history, "quarters_held") == 3
    print("SANITY: annual shares double from exact q-4; a missing intermediate quarter resets verified tenure to3.")


def test_off_universe_position_is_included_in_fixed_price_whole_book() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000), "OUT": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (150, 1500), "OUT": (100, 1000)})
    history = _compute(_snapshot([prior, current]))
    assert _value(history, "relative_position_chg") == pytest.approx(0.25)
    assert _value(history, "pooled_weight_chg") == pytest.approx(0.10)
    print("SANITY: off-universe holdings remain in both whole-book denominators; output scope never shrinks the book.")


def test_missing_reference_excludes_whole_book_only_and_never_fills() -> None:
    prior = _book("1", "2024-03-31", {"AAA": (100, 1000), "OUT": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (150, 1500), "OUT": (100, 1000)})
    market = _market()
    assert market.as_traded_close is not None
    market.as_traded_close.loc[pd.Timestamp("2024-03-29"), "OUT"] = np.nan
    history = _compute(_snapshot([prior, current]), market)
    assert _value(history, "shares_chg") == pytest.approx(0.5)
    assert np.isnan(_value(history, "relative_position_chg"))
    assert np.isnan(_value(history, "pooled_weight_chg"))
    print("SANITY: one missing exact prior-session union quote excludes whole-book rates while supported ticker share growth survives.")


def test_different_manager_capital_growth_can_move_pooled_allocation() -> None:
    books = [
        _book("1", "2024-03-31", {"AAA": (5, 50), "BBB": (5, 50)}),
        _book("2", "2024-03-31", {"BBB": (10, 100)}),
        _book("1", "2024-06-30", {"AAA": (10, 100), "BBB": (10, 100)}),
        _book("2", "2024-06-30", {"BBB": (10, 100)}),
    ]
    history = _compute(_snapshot(books))
    assert _value(history, "relative_position_chg") == pytest.approx(0)
    assert _value(history, "conviction_weight") == pytest.approx(0.25)
    assert _value(history, "pooled_weight_chg") == pytest.approx(1 / 3 - 1 / 4)
    print("SANITY: manager-specific proportional expansion has zero relative activity but changes pooled capital weights by8.333pp.")


def test_daily_projection_has_no_decay_and_explicit_unknown_clears_state() -> None:
    snapshot = _snapshot([_book("1", "2024-06-30", {"AAA": (100, 1000)})])
    expired = replace(snapshot, day=pd.Timestamp("2024-08-20"), level_books={})
    market = _market()
    history = compute_superinvestor_feature_history((snapshot, expired), {"": frozenset({"1"}), "topk": frozenset({"1"})}, ("AAA", "BBB"), market)
    dates = pd.bdate_range("2024-08-15", "2024-08-23")
    prices = pd.DataFrame(10.0, index=dates, columns=["AAA", "BBB"])
    frames = PriceFrames(dates, ("AAA", "BBB"), {}, close_split=prices)
    full = project_superinvestor_features(history, frames, pd.Timestamp("2013-01-01")).set_index(["date", "ticker"])
    assert len(full.columns) == 26
    assert full.loc[(pd.Timestamp("2024-08-19"), "AAA"), "f_ic_super_holders"] == 1
    assert full.loc[[(pd.Timestamp("2024-08-20"), "AAA")]].isna().all().all()
    tail_frames = replace(frames, trading_index=dates[-2:], close_split=prices.tail(2))
    tail = project_superinvestor_features(history, tail_frames, pd.Timestamp("2013-01-01")).set_index(["date", "ticker"])
    pd.testing.assert_frame_equal(tail, full.loc[tail.index])
    print("SANITY: no event means stable levels; explicit expiry clears all26fields, including an entirely unknown incremental tail.")


def test_breadth_uses_matched_manager_proportion_and_entries_have_no_prior_rate() -> None:
    books = [
        _book("1", "2024-03-31", {"AAA": (100, 1000)}),
        _book("1", "2024-06-30", {"AAA": (100, 1000)}),
        _book("2", "2024-03-31", {"BBB": (100, 1000)}),
        _book("2", "2024-06-30", {"AAA": (100, 1000)}),
    ]
    history = _compute(_snapshot(books))
    assert _value(history, "breadth_chg") == 0.5
    assert _value(history, "shares_chg") == 1
    assert _value(history, "trade_balance") == 1
    assert _value(history, "relative_position_chg") == 0
    entry = _compute(_snapshot(books[2:]))
    assert np.isnan(_value(entry, "shares_chg"))
    assert np.isnan(_value(entry, "relative_position_chg"))
    print("SANITY: one new holder among two matched books adds50pp breadth; a pure entry has no prior-positive rate but casts a buyer vote.")


@pytest.mark.parametrize("lag", [1, 4])
def test_changed_security_identity_is_unsupported_for_both_lags(lag: int) -> None:
    prior_period = "2024-03-31" if lag == 1 else "2023-06-30"
    prior = _book("1", prior_period, {"AAA": (100, 1000)})
    current = _book("1", "2024-06-30", {"AAA": (200, 2000)})
    current.df_positions["cusip"] = "UNRESOLVED_NEW_CUSIP"
    history = _compute(_snapshot([prior, current]))
    assert np.isnan(_value(history, "shares_chg" if lag == 1 else "shares_chg_yoy"))
    print("SANITY: same ticker with unresolved changed CUSIP cannot become a reported trade, for quarterly or annual comparisons.")


def test_consensus_is_signed_and_excludes_unchanged_managers() -> None:
    books = []
    for cik, current in (("1", 110), ("2", 110), ("3", 90), ("4", 100)):
        books.extend([_book(cik, "2024-03-31", {"AAA": (100, 1000)}), _book(cik, "2024-06-30", {"AAA": (current, current * 10)})])
    history = _compute(_snapshot(books))
    assert _value(history, "trade_balance") == pytest.approx(1 / 3)
    print("SANITY: two buyers, one seller and one unchanged manager give signed consensus+1/3.")
