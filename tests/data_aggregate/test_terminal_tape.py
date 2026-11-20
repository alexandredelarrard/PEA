"""Known-truth fixtures for the terminal-unit Yahoo volume rule: ledger spin-off price factors create no shares."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import SHARADAR_ACTION_SPLIT
from src.data_aggregate.utils.price.features import as_traded_volume, security_basis_frame, terminal_tape
from src.utils.price.adjustments import ShareBasis, resolve_share_events

RAW = 4_000_000.0
LATER = 3_870_000.0


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame({"ticker": [t for t, _, _ in rows], "date": pd.to_datetime([d for _, d, _ in rows]), "value": [v for _, _, v in rows]})


def _basis(actual: list[tuple[str, str, float]], ledger: list[tuple[str, str, float]]) -> ShareBasis:
    df_actual = _frame(actual)
    return ShareBasis(df_actual, df_actual, _frame(ledger), {})


def _tape(ticker: str, event: str, yahoo_ratio: float, *, raw: float = RAW) -> pd.DataFrame:
    """Yahoo's tape for one ticker: raw traded shares x its ledger ratio before the event, raw shares after."""
    days = pd.bdate_range(pd.Timestamp(event) - pd.Timedelta(days=60), pd.Timestamp(event) + pd.Timedelta(days=45))
    before = [raw * yahoo_ratio if day < pd.Timestamp(event) else LATER for day in days]
    return pd.DataFrame({ticker: before}, index=pd.DatetimeIndex(days, name="date"))


def _before(frame: pd.DataFrame, ticker: str, event: str) -> pd.Series:
    return frame.loc[frame.index < pd.Timestamp(event), ticker]


def _after(frame: pd.DataFrame, ticker: str, event: str) -> pd.Series:
    return frame.loc[frame.index >= pd.Timestamp(event), ticker]


def test_hon_spinoff_with_a_codated_reverse_split_keeps_only_the_split() -> None:
    """Sharadar's 1-for-2 and Yahoo's 0.9535 spin-off factor share 2026-06-29; the production ratio-conflict rule keeps 0.5."""
    df_actions = pd.DataFrame(
        {"ticker": ["HON"], "date": pd.to_datetime(["2026-06-29"]), "action": [SHARADAR_ACTION_SPLIT], "value": [0.5], "contraticker": ["N/A"]}
    )
    df_ledger = pd.DataFrame({"ticker": ["HON"], "date": pd.to_datetime(["2026-06-29"]), "ratio": [0.9535]})
    basis = resolve_share_events(df_actions, df_ledger, {}, tickers=["HON"])
    df_volume = _tape("HON", "2026-06-29", 0.9535)

    out = terminal_tape(df_volume, basis)

    assert _before(out, "HON", "2026-06-29").to_numpy() == pytest.approx(RAW * 0.5, rel=1e-12)
    assert (_before(out, "HON", "2026-06-29") / _before(df_volume, "HON", "2026-06-29")).iloc[0] == pytest.approx(0.5243838489774515, rel=1e-12)
    assert _after(out, "HON", "2026-06-29").eq(LATER).all()
    print(
        f"\nSANITY: HON pre-event Yahoo {RAW * 0.9535:,.0f} -> {RAW * 0.5:,.0f} = raw x the reverse split 0.5 (factor 0.5/0.9535); later bars untouched."
    )


def test_ctva_pure_spinoff_undoes_the_whole_ledger_ratio() -> None:
    basis = _basis([], [("CTVA", "2026-10-01", 6.665)])
    df_volume = _tape("CTVA", "2026-10-01", 6.665, raw=4_243_700.0)

    out = terminal_tape(df_volume, basis)

    ratio = (_before(out, "CTVA", "2026-10-01") / _before(df_volume, "CTVA", "2026-10-01")).to_numpy()
    assert ratio == pytest.approx(0.15003750937734434, rel=1e-12)
    assert _before(out, "CTVA", "2026-10-01").to_numpy() == pytest.approx(4_243_700.0, rel=1e-12)
    assert _after(out, "CTVA", "2026-10-01").eq(LATER).all()
    print("\nSANITY: CTVA pre-event bars x 1/6.665 = 0.15003750937734434 return to raw traded shares; no genuine event, later bars untouched.")


@pytest.mark.parametrize(("ticker", "event", "ratio"), [("DD", "2025-11-03", 2.39), ("DELL", "2021-11-02", 1.973), ("EQT", "2018-11-13", 1.837)])
def test_pure_spinoffs_are_restated_by_one_over_the_ledger_ratio(ticker: str, event: str, ratio: float) -> None:
    basis = _basis([], [(ticker, event, ratio)])
    df_volume = _tape(ticker, event, ratio)

    out = terminal_tape(df_volume, basis)

    assert _before(out, ticker, event).to_numpy() == pytest.approx(RAW, rel=1e-12)
    assert _after(out, ticker, event).eq(LATER).all()
    print(f"\nSANITY: {ticker} {event} ledger x{ratio} spin-off: pre-event bars divided by {ratio} back to raw {RAW:,.0f}; later bars untouched.")


def test_a_genuine_split_with_equal_ratios_leaves_volume_unchanged() -> None:
    basis = _basis([("AAA", "2022-07-18", 20.0)], [("AAA", "2022-07-18", 20.0)])
    df_volume = _tape("AAA", "2022-07-18", 20.0)

    out = terminal_tape(df_volume, basis)

    pd.testing.assert_frame_equal(out, df_volume, check_exact=True)
    print("\nSANITY: a 20:1 split Yahoo and the genuine basis both date gives factor 1; the tape is bit-identical.")


def test_a_yahoo_only_split_shaped_event_is_genuine_and_unchanged() -> None:
    df_ledger = pd.DataFrame({"ticker": ["BBB"], "date": pd.to_datetime(["2023-03-06"]), "ratio": [3.0]})
    basis = resolve_share_events(None, df_ledger, {}, tickers=["BBB"])
    assert basis.actual["value"].tolist() == [3.0], "the production rule keeps a Yahoo-only split-shaped event as a genuine split"
    df_volume = _tape("BBB", "2023-03-06", 3.0)

    out = terminal_tape(df_volume, basis)

    pd.testing.assert_frame_equal(out, df_volume, check_exact=True)
    print("\nSANITY: a Yahoo-only 3:1 ledger event that resolve_share_events keeps as genuine leaves the tape bit-identical.")


def test_two_ledger_events_compound_by_date() -> None:
    basis = _basis([], [("CCC", "2020-03-02", 1.3), ("CCC", "2021-06-01", 2.39)])
    days = pd.DatetimeIndex(pd.bdate_range("2019-12-02", "2021-09-01"), name="date")
    raw = 1_000.0
    volume = np.where(days < pd.Timestamp("2020-03-02"), raw * 1.3 * 2.39, np.where(days < pd.Timestamp("2021-06-01"), raw * 2.39, raw))
    df_volume = pd.DataFrame({"CCC": volume}, index=days)

    out = terminal_tape(df_volume, basis)

    assert out["CCC"].to_numpy() == pytest.approx(raw, rel=1e-12)
    print(
        "\nSANITY: a 1.3 spin-off then a 2.39 spin-off: bars before both /(1.3 x 2.39), bars between /2.39, bars after untouched; the series is flat at raw."
    )


def test_an_event_on_the_first_bar_restates_nothing_and_on_the_last_bar_restates_all_earlier_bars() -> None:
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=5), name="date")
    df_volume = pd.DataFrame({"FIRST": 100.0, "LAST": 100.0}, index=days)
    basis = _basis([], [("FIRST", str(days[0].date()), 2.39), ("LAST", str(days[-1].date()), 2.39)])

    out = terminal_tape(df_volume, basis)

    assert out["FIRST"].eq(100.0).all()
    assert out["LAST"].iloc[:-1].to_numpy() == pytest.approx(100.0 / 2.39, rel=1e-12) and out["LAST"].iloc[-1] == 100.0
    print(
        "\nSANITY: an event dated on the first bar restates no bar (the event restates bars BEFORE its date); one on the last bar restates the other four."
    )


def test_no_ledger_and_tickers_without_events_are_untouched_and_the_input_is_not_mutated() -> None:
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=5), name="date")
    df_volume = pd.DataFrame({"HIT": 100.0, "MISS": [1.0, 2.0, np.nan, 4.0, 5.0]}, index=days)
    original = df_volume.copy()

    empty = terminal_tape(df_volume, _basis([], []))
    out = terminal_tape(df_volume, _basis([], [("HIT", str(days[-1].date()), 2.39), ("OTHER", str(days[2].date()), 2.0)]))

    pd.testing.assert_frame_equal(empty, original, check_exact=True)
    pd.testing.assert_series_equal(out["MISS"], original["MISS"], check_exact=True)
    assert out["HIT"].iloc[0] == pytest.approx(100.0 / 2.39)
    pd.testing.assert_frame_equal(df_volume, original, check_exact=True)
    print(
        "\nSANITY: an empty ledger returns the tape unchanged; the ticker without events and a ledger ticker absent from the tape are bit-identical; the input frame is not mutated."
    )


def test_the_restated_tape_is_as_traded_through_the_pit_conversion(caplog: pytest.LogCaptureFixture) -> None:
    """HON: terminal tape x the reverse split, `volume_pit` then returns the raw traded shares before the event."""
    basis = _basis([("HON", "2026-06-29", 0.5)], [("HON", "2026-06-29", 0.9535)])
    df_volume = _tape("HON", "2026-06-29", 0.9535)
    caplog.set_level(logging.INFO)

    tape = terminal_tape(df_volume, basis)
    df_basis = security_basis_frame(basis, pd.DatetimeIndex(tape.index), pd.Index(["HON"]))
    pit = as_traded_volume(tape, df_basis, basis)

    assert _before(pit, "HON", "2026-06-29").to_numpy() == pytest.approx(RAW, rel=1e-12)
    assert _after(pit, "HON", "2026-06-29").eq(LATER).all()
    assert any("HON 2026-06-29" in message for message in caplog.messages), caplog.messages
    print(
        f"\nSANITY: HON volume_pit before the event = raw {RAW:,.0f} traded shares (supported, non-null), after = {LATER:,.0f}; one log line names the event."
    )
