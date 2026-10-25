"""Known-truth whole-book packet boundaries over the production ordered reader."""

from __future__ import annotations

import weakref
from collections.abc import Generator
from contextlib import closing
from functools import partial
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import inputs
from src.data_aggregate.utils.sec_thirteen import institutional_books as books
from src.data_aggregate.utils.sec_thirteen import value_basis as values
from src.data_aggregate.utils.sec_thirteen.holdings_clean import attach_tickers, clean_public_holdings
from src.data_aggregate.utils.sec_thirteen.institutional_vintages import iter_public_books
from src.data_store.schema import Tables


def _payload() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
    packets = (
        ("0000000001", "2023-12-31", "original", 3),
        ("0000000001", "2023-12-31", "restatement", 2),
        ("0000000001", "2024-03-31", "empty", 0),
        ("0000000002", "2023-12-31", "second", 1),
        ("0000000002", "2024-03-31", "last", 2),
    )
    for cik, period, accession, count in packets:
        base = {
            "cik": cik,
            "period": pd.Timestamp(period),
            "accession_number": accession,
            "filing_date": pd.Timestamp("2024-05-15"),
            "accepted_at": pd.Timestamp("2024-05-15T10:00:00Z"),
            "amendment_type": "restatement" if accession == "restatement" else "original",
            "amendment_number": int(accession == "restatement"),
            "report_type": "13F HOLDINGS REPORT",
            "restricted": False,
            "parsed_row_count": count,
            "position_type": "common",
            "class_evidence": '["common"]',
            "unneeded_payload": "not projected",
            **dict.fromkeys(numeric, 0.0),
            **{"raw_" + column: "0" for column in numeric if "value" in column},
        }
        rows.append({**base, "ticker": Tables.sec13f_hr.empty_marker[0], "cusip": Tables.sec13f_hr.empty_marker[1], "parse_status": "complete"})
        for number in range(count):
            rows.append(
                {**base, "ticker": "ABC", "cusip": f"{number:09d}", "parse_status": "position", "shares": float(number + 1), "value_usd": 10.0}
            )
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


@pytest.mark.parametrize("chunksize", [1, 2, 200_000])
def test_ordered_packets_cross_chunks_and_flush_empty_and_last_books(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, chunksize: int) -> None:
    payload = _payload()
    sqlite_store.save(Tables.sec13f_hr, payload)
    original = sqlite_store.iter_load
    calls: list[dict[str, Any]] = []

    def capture(table: Any, **kwargs: Any) -> Generator[pd.DataFrame]:
        assert table == Tables.sec13f_hr
        calls.append(kwargs)
        yield from original(table, chunksize=chunksize, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", capture)
    packets = inputs.iter_filing_vintages(sqlite_store, pd.Timestamp("2024-05-16"))
    assert packets is not None
    with closing(packets):
        result = list(packets)

    assert [(filings.cik.iloc[0], filings.period.iloc[0], len(filings), len(holdings)) for filings, holdings in result] == [
        ("0000000001", pd.Timestamp("2023-12-31"), 2, 5),
        ("0000000001", pd.Timestamp("2024-03-31"), 1, 0),
        ("0000000002", pd.Timestamp("2023-12-31"), 1, 1),
        ("0000000002", pd.Timestamp("2024-03-31"), 1, 2),
    ]
    assert len(calls) == 1
    assert calls[0]["order_by"] == ("cik", "period", "accession_number")
    assert calls[0]["markers"] is True
    assert calls[0]["where"] == {"parse_status": ["complete", "position"]}
    assert calls[0]["date_col"] == "filing_date" and calls[0]["until"] == pd.Timestamp("2024-05-16")
    assert "unneeded_payload" not in calls[0]["columns"]
    assert sum(len(holdings) for _, holdings in result) == payload.parse_status.eq("position").sum()
    print(
        f"SANITY: chunk size {chunksize} preserves complete multi-accession CIK-quarters, a verified empty book and EOF through one projected stream."
    )


def test_completed_packet_frames_are_released(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_store.save(Tables.sec13f_hr, _payload())
    monkeypatch.setattr(sqlite_store, "iter_load", partial(sqlite_store.iter_load, chunksize=2))
    packets = inputs.iter_filing_vintages(sqlite_store, pd.Timestamp("2024-05-16"))
    assert packets is not None
    references: list[weakref.ReferenceType[pd.DataFrame]] = []
    alive: list[int] = []
    with closing(packets):
        for filings, holdings in packets:
            references.extend((weakref.ref(filings), weakref.ref(holdings)))
            alive.append(sum(reference() is not None for reference in references))
            del filings, holdings
    assert len(references) == 8 and max(alive) <= 2
    assert all(reference() is None for reference in references)
    print("SANITY: consumed packet frames are released; retained raw history does not grow with completed CIK-quarters.")


@pytest.mark.parametrize("consumer_fails", [False, True])
def test_packet_stream_closes_on_processing_failure(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, consumer_fails: bool) -> None:
    sqlite_store.save(Tables.sec13f_hr, _payload())
    original = sqlite_store.iter_load
    closed: list[bool] = []

    def capture(table: Any, **kwargs: Any) -> Generator[pd.DataFrame]:
        try:
            yield from original(table, chunksize=2, **kwargs)
        finally:
            closed.append(True)

    def fail_partition(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("packet processing failed")

    monkeypatch.setattr(sqlite_store, "iter_load", capture)
    if not consumer_fails:
        monkeypatch.setattr(inputs, "_partition_packet", fail_partition)
    packets = inputs.iter_filing_vintages(sqlite_store, pd.Timestamp("2024-05-16"))
    assert packets is not None
    with pytest.raises(RuntimeError, match="packet processing failed"), closing(packets):
        next(packets)
        raise RuntimeError("packet processing failed")
    assert closed == [True]
    print(f"SANITY: the real source cursor closes after {'consumer' if consumer_fails else 'packet'} processing fails.")


def test_streamed_value_repair_reuses_exact_age_bounded_quotes() -> None:
    close = pd.DataFrame({"ABC": [10.0, 11.0]}, index=pd.to_datetime(["2024-05-10", "2024-05-20"]))
    policy = {"max_price_age_days": 7}
    raw = _payload().loc[lambda frame: frame.parse_status.eq("position")].iloc[:1].copy()
    raw["shares"], raw["value_usd"] = 2.0, 20.0
    periods = pd.to_datetime(["2024-05-09", "2024-05-10", "2024-05-12", "2024-05-18", "2024-05-25", "2024-05-28"])
    expected = [clean_public_holdings(raw.assign(period=period), close, policy) for period in periods]

    with patch.object(values, "_period_close", wraps=values._period_close) as lookup:
        quotes = values.prepare_public_value_quotes(close, policy)
        assert lookup.call_count == 1
    with patch.object(values, "_period_close", side_effect=AssertionError("quote history was rebuilt per book")):
        actual = [clean_public_holdings(raw.assign(period=period), close, policy, period_quotes=quotes) for period in periods]

    for before, after in zip(expected, actual, strict=True):
        pd.testing.assert_frame_equal(before, after)
    print("SANITY: one quote preparation preserves repair values/provenance before prices, on weekends, at expiry and after the final quote.")


def _version(cik: str, period: str, accession: str, filed: str, rows: list[dict[str, Any]], kind: str = "original") -> pd.DataFrame:
    base = _payload().loc[lambda frame: frame.parse_status.eq("position")].iloc[0].to_dict()
    base.update(
        cik=cik,
        period=pd.Timestamp(period),
        accession_number=accession,
        filing_date=pd.Timestamp(filed),
        accepted_at=pd.Timestamp(filed, tz="UTC") + pd.Timedelta(hours=10),
        amendment_type=kind,
        amendment_number=int(kind != "original"),
        parsed_row_count=len(rows),
        shares=0.0,
        value_usd=0.0,
    )
    marker = {**base, "ticker": Tables.sec13f_hr.empty_marker[0], "cusip": Tables.sec13f_hr.empty_marker[1], "parse_status": "complete"}
    return pd.DataFrame([marker, *({**base, **row, "parse_status": "position"} for row in rows)])


def _compare_compact_replay(sqlite_store: Any, payload: pd.DataFrame, mapping: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    grid = pd.bdate_range("2024-01-02", "2025-03-31")
    universe = tuple(mapping.ticker.unique())
    close = pd.DataFrame(10.0, index=grid, columns=list(universe))
    sqlite_store.save(Tables.sec13f_hr, payload)
    packet = inputs.load_filing_vintages(sqlite_store, grid.max())
    assert packet is not None
    manifests, raw = packet
    prepared = clean_public_holdings(attach_tickers(raw, mapping, universe), close, None)
    expected = list(iter_public_books(manifests, prepared, grid, settle_days=3))
    stream = inputs.iter_filing_vintages(sqlite_store, grid.max())
    assert stream is not None
    history = books.prepare_institutional_book_history(stream, grid, mapping, universe, close, None, settle_days=3)
    assert history is not None
    actual = list(books.iter_institutional_books(*history))

    assert [day for day, _, _ in actual] == [day for day, _, _ in expected]
    for (_, df_compact, states), (_, raw_books, expected_states) in zip(actual, expected, strict=True):
        expected_compact = books.compact_institutional_books(raw_books)
        pd.testing.assert_frame_equal(df_compact.sort_index().reset_index(), expected_compact.sort_index().reset_index())
        pd.testing.assert_frame_equal(
            states.sort_values(["cik", "period"]).reset_index(drop=True), expected_states.sort_values(["cik", "period"]).reset_index(drop=True)
        )
    return actual


def test_compact_history_keeps_global_floor_stale_days_and_empty_replacements(sqlite_store: Any) -> None:
    payload = pd.concat(
        [
            _version("0000000001", "2024-03-31", "a1", "2024-05-14", [{"cusip": "111111111", "shares": 10.0, "value_usd": 100.0}]),
            _version("0000000001", "2024-06-30", "a2", "2024-08-14", [{"cusip": "111111111", "shares": 20.0, "value_usd": 200.0}]),
            _version(
                "0000000001", "2024-03-31", "a1-correction", "2024-09-20", [{"cusip": "111111111", "shares": 12.0, "value_usd": 120.0}], "restatement"
            ),
            _version(
                "0000000002",
                "2024-09-30",
                "b3",
                "2024-11-14",
                [
                    {"cusip": "222222222", "shares": 30.0, "value_usd": 300.0},
                    {"cusip": "333333333", "position_type": "call", "call_shares": 5.0, "call_value": 50.0},
                    {"cusip": "444444444", "shares": float("nan"), "value_usd": float("nan")},
                    {"cusip": "555555555"},
                    {"cusip": "999999999", "shares": 40.0, "value_usd": 400.0},
                ],
            ),
            _version("0000000001", "2024-06-30", "a2-empty", "2024-12-17", [], "restatement"),
            _version(
                "0000000001", "2024-03-31", "a1-stale", "2025-01-03", [{"cusip": "111111111", "shares": 15.0, "value_usd": 150.0}], "restatement"
            ),
            _version("0000000003", "2024-12-31", "orphan", "2025-02-14", [{"cusip": "111111111", "shares": 50.0, "value_usd": 500.0}], "additions"),
            _version("0000000002", "2024-09-30", "b3-empty", "2025-02-24", [], "restatement"),
        ],
        ignore_index=True,
    )
    mapping = pd.DataFrame(
        {"cusip": ["111111111", "222222222", "333333333", "444444444", "555555555"], "ticker": ["AAA", "BBB", "OPT", "UNK", "ZER"]}
    )
    snapshots = _compare_compact_replay(sqlite_store, payload, mapping)
    november = next(frame for day, frame, _ in snapshots if day.month == 11)
    quarter = pd.Timestamp("2024-09-30")
    assert november.loc[(quarter, "0000000002", "OPT"), "call_value"] == 50.0
    assert november.loc[(quarter, "0000000002", "ZER"), "shares"] == 0.0
    assert pd.isna(november.loc[(quarter, "0000000002", "UNK"), "shares"])
    assert pd.isna(november.loc[(quarter, "0000000002", "OPT"), "_cusips"])
    assert not snapshots[-2][1].empty and snapshots[-2][2].loc[lambda frame: frame.book_known, "period"].max() == quarter
    assert snapshots[-1][1].empty and snapshots[-1][2].loc[lambda frame: frame.book_known, "period"].max() == quarter
    assert len(snapshots) == 8 and snapshots[-3][0].month == 1
    print(
        "SANITY: compact replay preserves global-quarter eviction, stale-only publication days, unknown/zero/options, prior corrections, orphan barriers and empty replacements."
    )


def test_unmapped_positions_still_establish_copied_majority(sqlite_store: Any) -> None:
    original = [{"cusip": f"{number:09d}", "shares": float(number), "value_usd": float(10 * number)} for number in range(1, 7)]
    amendment = [{**row, "shares": 20.0, "value_usd": 200.0} if row["cusip"] == "000000001" else row for row in original[:5]]
    payload = pd.concat(
        [
            _version("0000000001", "2024-03-31", "original", "2024-05-14", original),
            _version("0000000001", "2024-03-31", "copied", "2024-06-14", amendment, "additions"),
        ],
        ignore_index=True,
    )
    snapshots = _compare_compact_replay(sqlite_store, payload, pd.DataFrame({"cusip": ["000000001"], "ticker": ["AAA"]}))
    assert snapshots[-1][1].shares.iloc[0] == 20.0
    assert snapshots[-1][2].resolution_status.iloc[0] == "inferred_position_patch" and not snapshots[-1][2].zero_eligible.iloc[0]
    print("SANITY: off-universe repeated positions still establish copied-majority inference before mapped compact reduction.")
