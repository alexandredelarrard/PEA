"""Exact role sentiment, fiscal comparisons, identity and daily lifetime mathematics."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

import src.data_aggregate.utils.text.earnings_call_features as ec
from src.constants.constants import EARNINGS_CALL_FEATURES
from src.context import Context
from src.data_aggregate.utils.common.incremental import PartWindow, write_part
from src.data_aggregate.utils.text.earnings_call_features import (
    _daily_frame,
    attach_issuer_identity,
    build_earnings_call_feature_panel,
    feature_history_scope,
    prepare_earnings_call_kpis,
    sentiment_kpis,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore

EARNINGS_CALL_SENTIMENT_CACHE_MODEL = "fixture/tone:speaker-clean-v2"

_QDATE = {"2023Q1": "2023-02-01", "2023Q2": "2023-05-01", "2023Q3": "2023-08-01"}


def _canonical_tenure(roster: pd.DataFrame) -> pd.DataFrame:
    return roster.rename(columns={"ticker": "symbol", "cik": "issuer_cik"}).assign(valid_from="1900-01-01", valid_to=None)


def test_canonical_calls_require_a_consolidating_issuer_window() -> None:
    calls = pd.DataFrame(
        {
            "ticker": ["CEG", "CEG", "NEW", "NEW", "NEW", "EVENT"],
            "as_of": pd.to_datetime(["2008-01-30", "2024-05-09", "2021-02-01", "2022-01-01", "2024-02-01", "2024-02-01"]),
        }
    )
    roster = pd.DataFrame({"ticker": ["CEG", "NEW", "EVENT"], "cik": ["1", "3", "4"]})
    lineage = pd.DataFrame(
        {
            "cik": ["1", "2", "3", "4"],
            "entity_id": ["E1", "E2", "E2", "E4"],
            "role": ["cik_window", "cik_window", "cik_window", "cik_event"],
            "valid_from": pd.to_datetime(["2022-01-01", "1900-01-01", "2023-01-01", "1900-01-01"]),
            "valid_to": pd.to_datetime(pd.Series([None, "2022-01-01", None, None])),
        }
    )
    identified = attach_issuer_identity(calls, roster, lineage, _canonical_tenure(roster))
    assert identified.index.tolist() == [1, 2, 4]
    assert identified["issuer_id"].tolist() == ["E1", "E2", "E2"]
    assert identified["as_of"].tolist() == calls.loc[[1, 2, 4], "as_of"].tolist()
    print(
        "SANITY: legacy CEG, the half-open predecessor end/gap and event-only history are excluded; supported predecessor/current calls keep actual dates."
    )


def test_canonical_ticker_keeps_calls_before_the_symbol_change() -> None:
    calls = pd.DataFrame({"ticker": ["ECHO", "ECHO", "UNKNOWN"], "as_of": pd.to_datetime(["2008-05-13", "2026-08-03", "2026-08-03"])})
    roster = pd.DataFrame({"ticker": ["ECHO"], "cik": ["1"]})
    lineage = pd.DataFrame({"cik": ["1"], "entity_id": ["E1"]}).assign(role="cik_window", valid_from="1900-01-01", valid_to=None)
    identified = attach_issuer_identity(calls, roster, lineage, _canonical_tenure(roster))
    assert identified["issuer_id"].tolist() == ["E1", "E1"]
    assert identified["as_of"].tolist() == calls["as_of"].iloc[:2].tolist()
    print("SANITY: validated canonical ECHO keeps pre-rename history; unknown destinations remain excluded; dates unchanged.")


def test_reused_alias_requires_dated_own_support_but_preserves_uncontested_history() -> None:
    calls = pd.DataFrame(
        {
            "ticker": ["CEG", "CEG", "ECHO", "FOXA", "FOXA", "CRH"],
            "as_of": pd.to_datetime(["2008-01-30", "2024-05-09", "2015-05-13", "2013-04-30", "2024-02-01", "2008-02-01"]),
        }
    )
    roster = pd.DataFrame({"ticker": ["CEG", "ECHO", "FOXA", "CRH"], "cik": ["1", "2", "3", "4"]})
    lineage = pd.DataFrame({"cik": ["1", "2", "3", "4"], "entity_id": ["E1", "E2", "E3", "E4"]}).assign(
        role="cik_window", valid_from="1900-01-01", valid_to=None
    )
    tenure = pd.DataFrame(
        {
            "symbol": ["CEG", "CEG", "ECHO", "SATS", "FOX", "FOX", "CRH"],
            "issuer_cik": ["9", "1", "8", "2", "7", "3", "4"],
            "valid_from": ["2006-01-01", "2022-01-28", "2009-10-01", "2008-01-02", "2013-07-02", "2019-02-05", "2023-09-25"],
            "valid_to": ["2012-03-15", None, "2021-11-24", None, "2019-03-21", None, None],
        }
    )
    identified = attach_issuer_identity(calls, roster, lineage, tenure)
    assert identified.index.tolist() == [1, 2, 4, 5]
    print("SANITY: reused CEG/FOX aliases reject unsupported legacy dates, SATS preserves canonical ECHO, and uncontested CRH keeps pre-US history.")


@pytest.mark.parametrize(("variant", "expected"), [("overlap", [1, 3]), ("same_entity", [0, 1, 3]), ("alternate", [0, 1, 3]), ("unknown", [1, 3])])
def test_canonical_support_requires_unambiguous_alias_inside_its_cik_window(variant: str, expected: list[int]) -> None:
    roster = pd.DataFrame({"ticker": ["NEW"], "cik": ["2"]})
    lineage = pd.DataFrame(
        {
            "cik": ["1", "2", "3"],
            "entity_id": ["E1", "E1", "E1" if variant == "same_entity" else "E2"],
            "valid_from": ["1900-01-01", "2020-01-01", "1900-01-01"],
            "valid_to": ["2020-01-01", None, None],
            "role": "cik_window",
        }
    )
    tenure = pd.DataFrame(
        {
            "symbol": ["OLD", "OLD", "NEW"],
            "issuer_cik": ["1", "9" if variant == "unknown" else "3", "2"],
            "valid_from": ["2000-01-01", "2000-01-01", "2030-01-01"],
            "valid_to": ["2025-01-01", "2015-01-01", None],
        }
    )
    if variant == "alternate":
        tenure = pd.concat(
            [tenure, pd.DataFrame({"symbol": ["SAFE"], "issuer_cik": ["1"], "valid_from": ["2010-01-01"], "valid_to": ["2011-01-01"]})],
            ignore_index=True,
        )
    if variant == "same_entity":
        tenure = pd.concat(
            [tenure, pd.DataFrame({"symbol": ["OLD"], "issuer_cik": ["9"], "valid_from": ["1990-01-01"], "valid_to": ["1995-01-01"]})],
            ignore_index=True,
        )
    calls = pd.DataFrame({"ticker": "NEW", "as_of": pd.to_datetime(["2010-05-01", "2015-01-01", "2024-05-01", "2030-01-01"])}, index=[10, 20, 30, 40])
    attached = attach_issuer_identity(calls, roster, lineage, tenure)
    assert attached.index.tolist() == calls.index[expected].tolist()
    assert attached["as_of"].tolist() == calls.iloc[expected]["as_of"].tolist()
    print(
        f"SANITY: {variant} support respects entity-grain ownership, its own CIK window and half-open ends; canonical dates and indices stay intact."
    )


@pytest.mark.parametrize(
    ("call_date", "expected"),
    [
        ("2024-05-06", "2024-05-08"),
        ("2024-05-02", "2024-05-06"),
        ("2024-05-03", "2024-05-06"),
        ("2024-05-04", "2024-05-06"),
        ("2024-05-24", "2024-05-28"),
        ("2024-05-26", "2024-05-28"),
    ],
)
def test_calendar_day_buffer_applies_once_to_all_ten_features(call_date: str, expected: str) -> None:
    calendar = pd.bdate_range("2024-05-01", "2024-10-31").difference(pd.DatetimeIndex(["2024-05-27"]))
    call = pd.DataFrame({"ticker": ["A"], "as_of": [pd.Timestamp(call_date)], **{name: [float(i)] for i, name in enumerate(EARNINGS_CALL_FEATURES)}})
    panel = ec._feature_panel_from_prepared(call, calendar, None, availability_lag_days=2, since=None)
    assert len(panel.filter(like="f_ec_").columns) == 10
    for name in EARNINGS_CALL_FEATURES:
        live = panel.loc[panel[f"f_{name}"].notna(), "date"]
        assert live.min() == pd.Timestamp(expected) and len(live) == 66
    print(f"SANITY: {call_date}+2 calendar days -> {expected}, including holiday/weekend rollover, all10 features for66 sessions.")


def _number(value: object) -> float:
    """Narrow a scalar selected from a numeric fixture frame."""
    assert not isinstance(value, pd.Series | pd.DataFrame)
    return float(cast(float, value))


def _row(tkr: str, q: str, tag: str, pos: float, neg: float, words: int) -> dict[str, object]:
    return dict(
        ticker=tkr,
        quarter=q,
        tag=tag,
        as_of=_QDATE[q],
        sent_pos=pos,
        sent_neg=neg,
        sent_neu=round(1 - pos - neg, 6),
        n_words=words,
        model=EARNINGS_CALL_SENTIMENT_CACHE_MODEL,
    )


def _sentiment_frame() -> pd.DataFrame:
    rows = []
    for q, prep, answer, question in (
        ("2023Q1", (0.6, 0.1), (0.4, 0.1), (0.2, 0.5)),
        ("2023Q2", (0.7, 0.05), (0.5, 0.1), (0.3, 0.4)),
        ("2023Q3", (0.55, 0.15), (0.45, 0.2), (0.1, 0.6)),
    ):
        for tag, values in (("prepared_remarks", prep), ("questions", question), ("answers", answer)):
            rows.append(_row("A", q, tag, *values, 1000 if tag == "prepared_remarks" else 500))
    for i, ticker in enumerate(("B", "C", "D", "E"), 1):
        for q in _QDATE:
            for tag, pos, neg in (("prepared_remarks", 0.3 + 0.1 * i, 0.1), ("questions", 0.1, 0.15 + 0.1 * i), ("answers", 0.25 + 0.1 * i, 0.12)):
                rows.append(_row(ticker, q, tag, pos, neg, 500))
    return pd.DataFrame(rows)


_QUESTION = "Can you talk about revenue growth and the margin guidance for next year?"
_USEFUL = "revenue growth customer demand margin guidance cash flow outlook " * 12


def test_three_role_levels_and_independent_deltas() -> None:
    rows = []
    for quarter, date, nets in (("2024Q1", "2024-02-01", (0.5, -0.2, 0.1)), ("2024Q2", "2024-05-01", (0.3, 0.4, -0.3))):
        for tag, net in zip(("prepared_remarks", "questions", "answers"), nets, strict=True):
            rows.append(
                dict(
                    ticker="A",
                    quarter=quarter,
                    as_of=date,
                    tag=tag,
                    sent_pos=(1 + net) / 2,
                    sent_neg=(1 - net) / 2,
                    sent_neu=0.0,
                    n_words=200,
                    model="m:speaker-clean-v3",
                )
            )
    actual = prepare_earnings_call_kpis(sentiment_kpis(pd.DataFrame(rows), "m:speaker-clean-v3")).set_index("quarter")
    for role, level, delta in (("prep", 0.3, -0.2), ("questions", 0.4, 0.6), ("answers", -0.3, -0.4)):
        assert actual.loc["2024Q2", f"ec_{role}_tone"] == pytest.approx(level)
        assert actual.loc["2024Q2", f"ec_{role}_tone_delta"] == pytest.approx(delta)
        assert pd.isna(actual.loc["2024Q1", f"ec_{role}_tone_delta"])
    print("SANITY: three role levels and three independent adjacent-quarter deltas match exact positive-minus-negative arithmetic.")


def test_per_call_kpi_arithmetic() -> None:
    per = prepare_earnings_call_kpis(sentiment_kpis(_sentiment_frame().query("ticker == 'A'"), EARNINGS_CALL_SENTIMENT_CACHE_MODEL)).set_index(
        "quarter"
    )
    for role, first, second in (("prep", 0.5, 0.65), ("questions", -0.3, -0.1), ("answers", 0.3, 0.4)):
        assert per.loc["2023Q1", f"ec_{role}_tone"] == pytest.approx(first)
        assert per.loc["2023Q2", f"ec_{role}_tone_delta"] == pytest.approx(second - first)
        assert pd.isna(per.loc["2023Q1", f"ec_{role}_tone_delta"])
    print("SANITY: exact FinBERT role levels and adjacent-quarter deltas; word counts do not mix analyst and management tone.")


def test_sentiment_kpis_equals_batch(sqlite_store: DataStore) -> None:
    sent = _sentiment_frame()
    sqlite_store.save(Tables.earnings_call_sentiment, sent)
    stored = sqlite_store.load(Tables.earnings_call_sentiment)
    assert stored is not None
    actual = sentiment_kpis(stored, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)
    expected = pd.concat([sentiment_kpis(group, EARNINGS_CALL_SENTIMENT_CACHE_MODEL) for _, group in sent.groupby("ticker")], ignore_index=True)
    actual["as_of"] = actual["as_of"].astype("datetime64[ns]")
    expected["as_of"] = expected["as_of"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    print("SANITY: one saved metric frame produces exactly the same per-issuer arithmetic; no paragraph inputs.")


def test_malformed_or_incomplete_cached_call_is_missing() -> None:
    scores = _sentiment_frame().query("ticker == 'A' and quarter == '2023Q1'")
    valid = ec.valid_sentiment_rows(scores, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)
    assert ec.completed_sentiment_calls(valid) == {("A", "2023Q1")}
    assert not ec.completed_sentiment_calls(valid.head(1))
    assert not ec.completed_sentiment_calls(valid.assign(n_words=1))
    invalid = scores.copy()
    invalid.loc[invalid["tag"].eq("questions"), "sent_pos"] = np.nan
    partial = sentiment_kpis(invalid, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)
    assert len(partial) == 1 and partial["ec_questions_tone"].isna().all() and partial["ec_answers_tone"].notna().all()
    assert sentiment_kpis(scores.assign(model="legacy"), EARNINGS_CALL_SENTIMENT_CACHE_MODEL).empty
    placeholder = scores.copy()
    mask = placeholder["tag"].eq("answers")
    placeholder.loc[mask, ["sent_pos", "sent_neg", "sent_neu"]] = np.nan
    placeholder.loc[mask, "n_words"] = 0
    partial = sentiment_kpis(placeholder, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)
    assert len(partial) == 1 and partial["ec_answers_tone"].isna().all() and partial["ec_prep_tone"].notna().all()
    print("SANITY: incomplete scoring stays pending while valid partner roles remain usable; genuine absence stays null.")


def test_question_words_do_not_establish_completion() -> None:
    rows = _sentiment_frame().query("ticker == 'A' and quarter == '2023Q1'").copy()
    management = rows["tag"].ne("questions")
    rows.loc[management, "n_words"] = 0
    rows.loc[management, ["sent_pos", "sent_neg", "sent_neu"]] = np.nan
    rows.loc[~management, "n_words"] = 1000
    valid = ec.valid_sentiment_rows(rows, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)
    assert len(valid) == 3 and not ec.completed_sentiment_calls(valid)
    print("SANITY: three valid canonical roles plus 1,000 analyst words cannot satisfy the 100-management-word eligibility floor.")


def test_panel_columns_lifetime_and_missingness():
    idx = pd.bdate_range("2023-01-02", "2023-09-29")
    panel = build_earnings_call_feature_panel(sentiment_kpis(_sentiment_frame(), EARNINGS_CALL_SENTIMENT_CACHE_MODEL), idx, availability_lag_days=2)
    assert not panel.empty
    expected = {"date", "ticker", *[f"f_{name}" for name in EARNINGS_CALL_FEATURES]}
    assert set(panel.columns) == expected
    assert not any(c.endswith(("_xs", "_vs_peers")) for c in panel.columns)

    panel["date"] = pd.to_datetime(panel["date"])
    a = panel[panel["ticker"] == "A"]
    first_call = pd.Timestamp("2023-02-01")
    # The raw call date remains unchanged; feature eligibility waits two calendar days.
    a_tone = a[a["f_ec_prep_tone"].notna()]
    assert a_tone["date"].min() > first_call
    assert a_tone["date"].min() == pd.Timestamp("2023-02-03")
    # 66 trading sessions are inclusive; session 67 expires rather than becoming a fake zero.
    one = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-01"], "ec_prep_tone": [0.0]})
    daily = _daily_frame(one, "ec_prep_tone", idx, availability_lag_days=2)
    live = daily["A"].dropna()
    assert len(live) == 66
    assert (live == 0.0).all(), "a genuine zero must survive"
    assert pd.isna(_number(daily.loc[live.index[-1] + pd.offsets.BDay(1), "A"])), "session 67 must be NaN"

    weekend = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-04"], "ec_prep_tone": [0.2]})
    weekend_daily = _daily_frame(weekend, "ec_prep_tone", idx, availability_lag_days=2)
    assert weekend_daily.loc[pd.Timestamp("2023-02-06"), "A"] == 0.2

    print("\n=== SANITY CHECK: earnings-call features ===")
    print(f"  panel {panel.shape[0]} rows; exactly 10 raw EC features and no peer/cross-sectional variants.")
    print(
        f"  leak-free: ticker A first tone signal at {a_tone['date'].min().date()} "
        "(call 2023-02-01 + 2 calendar days); genuine zero survives for 66 sessions and "
        "session 67 is NaN."
    )


def test_full_calendar_late_refresh_and_rerun_are_bit_exact() -> None:
    """Exercise the real tail writer against a late correction and unchanged rerun."""
    calendar = pd.bdate_range("2021-01-04", "2023-09-29")
    assert len(calendar) >= 500
    old = build_earnings_call_feature_panel(
        sentiment_kpis(_sentiment_frame(), EARNINGS_CALL_SENTIMENT_CACHE_MODEL), calendar, availability_lag_days=2
    )
    revised_sentiment = _sentiment_frame()
    mask = (revised_sentiment["ticker"] == "A") & (revised_sentiment["quarter"] == "2023Q1")
    revised_sentiment.loc[mask, "sent_pos"] += 0.05
    revised_sentiment.loc[mask, "sent_neu"] -= 0.05
    revised = build_earnings_call_feature_panel(
        sentiment_kpis(revised_sentiment, EARNINGS_CALL_SENTIMENT_CACHE_MODEL), calendar, availability_lag_days=2
    )

    class _Store:
        def __init__(self, rows: pd.DataFrame):
            self.rows = rows.copy()

        def columns(self, _table):
            return list(self.rows.columns)

        def replace(self, _table, rows):
            self.rows = rows.copy()
            return len(rows)

        def append_tail(self, _table, tail, cutoff, *, inclusive):
            keep = self.rows["date"] < cutoff if inclusive else self.rows["date"] <= cutoff
            self.rows = pd.concat([self.rows.loc[keep], tail], ignore_index=True)
            return len(tail)

    feature_columns = [column for column in old if str(column).startswith("f_")]
    store = _Store(old.dropna(subset=feature_columns, how="all"))
    last = pd.Timestamp(old["date"].max())
    default_refresh = calendar[-130]
    late_refresh = pd.Timestamp("2023-02-02")
    window = PartWindow(last=last, since=calendar[0], refresh_from=default_refresh)
    assert window.refresh_from is not None
    effective_refresh = min(late_refresh, window.refresh_from)
    write_part(cast(DataStore, store), Tables.cube_part_text, revised, window, refresh_from=effective_refresh, drop_empty=True)
    expected = revised.dropna(subset=feature_columns, how="all").sort_values(["date", "ticker"]).reset_index(drop=True)
    actual = store.rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_exact=True)
    first = actual.copy()
    write_part(cast(DataStore, store), Tables.cube_part_text, revised, window, refresh_from=effective_refresh, drop_empty=True)
    pd.testing.assert_frame_equal(first, store.rows.sort_values(["date", "ticker"]).reset_index(drop=True), check_exact=True)
    print("\n=== SANITY CHECK: earnings-call full/tail equivalence ===")
    print(
        f"  a correction behind the default tail refreshed from {late_refresh.date()} and matched "
        f"the {len(actual)}-row full rebuild bit-for-bit; unchanged rerun was idempotent. Validated."
    )


def test_identity_excludes_unknown_and_ambiguous_rows_and_preserves_deltas() -> None:
    calls = pd.DataFrame(
        {
            "ticker": ["NEW", "NEW"],
            "quarter": ["2023Q4", "2024Q1"],
            "as_of": pd.to_datetime(["2023-11-01", "2024-02-01"]),
            "ec_prep_tone": [0.2, 0.5],
            "total_words": [1000.0, 2000.0],
        }
    )
    lineage = pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E1"]}).assign(role="cik_window", valid_from="1900-01-01", valid_to=None)
    roster = pd.DataFrame({"ticker": ["NEW"], "cik": ["2"]})
    prepared = prepare_earnings_call_kpis(attach_issuer_identity(calls, roster, lineage, _canonical_tenure(roster)))
    newest = prepared.iloc[-1]
    assert math.isclose(float(newest["ec_prep_tone_delta"]), 0.3)

    ambiguous = pd.concat(
        [
            roster,
            pd.DataFrame({"ticker": ["NEW"], "cik": ["3"]}),
        ],
        ignore_index=True,
    )
    ambiguous_lineage = pd.concat([lineage, pd.DataFrame({"cik": ["3"], "entity_id": ["E2"]})], ignore_index=True)
    assert attach_issuer_identity(calls.tail(1), ambiguous, ambiguous_lineage, _canonical_tenure(ambiguous)).empty
    unresolved = pd.concat(
        [roster, pd.DataFrame({"ticker": ["NEW"], "cik": ["9"]})],
        ignore_index=True,
    )
    assert attach_issuer_identity(calls.tail(1), unresolved, lineage, _canonical_tenure(unresolved)).empty
    assert attach_issuer_identity(calls.tail(1).assign(ticker="UNKNOWN"), roster, lineage, _canonical_tenure(roster)).empty
    assert attach_issuer_identity(calls, roster, lineage, None).empty
    assert attach_issuer_identity(calls, roster, lineage, pd.DataFrame()).empty
    assert attach_issuer_identity(calls, roster, lineage, _canonical_tenure(roster).assign(issuer_cik="9")).empty
    print("\n=== SANITY CHECK: strict issuer identity ===")
    print("  half-open tenures preserve OLD->NEW deltas; unknown or ambiguous mappings are excluded. Validated.")


def test_issuer_identity_reads_roster_and_distinct_lineage_pairs(sqlite_store) -> None:
    """The dated `entity_lineage` has several rows per CIK and `symbol_tenure` gains `dei` rows; the attach must not move."""
    calls = pd.DataFrame({"ticker": ["NEW", "NEW"], "quarter": ["2023Q4", "2024Q1"], "as_of": pd.to_datetime(["2023-11-01", "2024-02-01"])})
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["NEW"], "cik": ["2"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame(
            {
                "cik": ["1", "1", "2", "2"],
                "entity_id": ["E1"] * 4,
                "role": ["cik_window", "symbol", "cik_window", "symbol"],
                "symbol": ["", "OLD", "", "NEW"],
                "valid_from": ["1900-01-01", "2020-01-01", "2024-01-01", "2024-01-01"],
                "valid_to": ["2024-01-01", "2024-01-01", None, None],
            }
        ),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            {
                "symbol": ["OLD", "NEW", "NEW"],
                "issuer_cik": ["1", "2", "9"],
                "valid_from": ["2020-01-01", "2024-01-01", "2023-06-01"],
                "valid_to": ["2024-01-01", None, None],
                "n_filings": [10, 10, 2],
                "source": ["form345", "manual", "dei"],
                "evidence_period": ["", "", "2025q1"],
            }
        ),
    )

    roster, lineage, tenure = ec.load_issuer_identity(cast(Context, SimpleNamespace(store=sqlite_store)))
    identified = attach_issuer_identity(calls, roster, lineage, tenure)

    assert roster is not None and lineage is not None
    assert roster["cik"].astype(str).tolist() == ["2"] and len(lineage) == 4
    assert identified["issuer_id"].tolist() == ["E1", "E1"]
    print("\n=== SANITY CHECK: issuer identity inputs ===")
    print("  four lineage rows keep predecessor/current windows; canonical NEW reads roster CIK2 and retains both historical calls.")
    print("  dated symbol reuse and DEI source rows cannot reinterpret already validated canonical keys. Validated.")


def test_new_call_with_missing_kpi_terminates_the_previous_signal() -> None:
    calendar = pd.bdate_range("2024-01-01", periods=100)
    calls = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "quarter": ["2023Q4", "2024Q1"],
            "as_of": pd.to_datetime(["2024-01-05", "2024-02-02"]),
            "ec_prep_tone": [0.1, 0.2],
            "ec_answers_tone": [0.3, np.nan],
            "total_words": [1000.0, 1100.0],
        }
    )
    panel = build_earnings_call_feature_panel(calls, calendar, availability_lag_days=2)
    observed = panel[panel["ticker"].eq("AAA")].set_index("date")
    assert observed.loc[pd.Timestamp("2024-02-02"), "f_ec_answers_tone"] == 0.3
    assert pd.isna(observed.loc[pd.Timestamp("2024-02-05"), "f_ec_answers_tone"])
    assert observed.loc[pd.Timestamp("2024-02-05"), "f_ec_prep_tone"] == np.float32(0.2)
    print("\n=== SANITY CHECK: missing KPI supersedes stale signal ===")
    print("  the newer valid call preserves its observed tone but resets its absent Q&A gap to NaN. Validated.")


def test_each_role_delta_requires_its_own_adjacent_quarter() -> None:
    calls = pd.DataFrame(
        dict(
            ticker="A",
            issuer_id="E1",
            quarter=["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q2"],
            as_of=pd.date_range("2020-02-01", periods=5, freq="3MS"),
            ec_prep_tone=[0.1, 0.2, 0.3, 0.5, 0.6],
            ec_answers_tone=[np.nan, 0.7, np.nan, 0.4, 0.1],
            ec_questions_tone=[0.1, np.nan, 0.3, 0.6, 0.5],
        )
    )
    result = prepare_earnings_call_kpis(calls)
    assert result["ec_prep_tone_delta"].iloc[1:4].tolist() == pytest.approx([0.1, 0.1, 0.2])
    assert result["ec_answers_tone_delta"].isna().all()
    assert result["ec_questions_tone_delta"].iloc[3] == pytest.approx(0.3)
    assert result.filter(like="tone_delta").iloc[4].isna().all()
    print("SANITY: unrelated missing roles preserve prepared deltas; a missing role or fiscal quarter cannot bridge to an older observation.")


def test_tail_daily_calculation_preserves_original_expiry() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=200)
    calls = pd.DataFrame({"ticker": ["A"], "as_of": [calendar[10]], "ec_prep_tone": [0.2]})
    full = _daily_frame(calls, "ec_prep_tone", calendar, 2)
    for cutoff in (calendar[30], calendar[77], calendar[100]):
        tail = _daily_frame(calls, "ec_prep_tone", calendar, 2, since=cutoff)
        pd.testing.assert_frame_equal(tail, full.loc[cutoff:], check_exact=True)
    assert _daily_frame(calls, "ec_prep_tone", calendar, 2, since=calendar[100]).isna().all().all()
    print("SANITY: daily arrays start at the requested tail; the 66-session signal never restarts at its cutoff.")


def test_history_scope_is_bounded_and_keeps_fiscal_predecessors() -> None:
    periods = pd.period_range("2015Q1", "2026Q4", freq="Q")
    dates = periods.to_timestamp(how="end").normalize() + pd.Timedelta(days=30)
    calls = pd.DataFrame({"ticker": "A", "issuer_id": "E1", "quarter": [f"{period.year}Q{period.quarter}" for period in periods], "as_of": dates})
    calls.loc[calls["quarter"].eq("2026Q1"), "ticker"] = "OLD_A"
    expired = pd.DataFrame({"ticker": ["B"], "issuer_id": ["E2"], "quarter": ["2026Q1"], "as_of": [pd.Timestamp("2026-03-31")]})
    calls = pd.concat([calls, expired], ignore_index=True)
    calendar = pd.bdate_range("2010-01-01", "2026-09-15")
    cutoff = pd.Timestamp("2026-08-15")
    scope, comparisons = feature_history_scope(calls, calendar, cutoff, 2)
    assert len(scope) < len(calls)
    assert len(scope) == 2
    assert scope["as_of"].max() <= calendar[-1]
    assert set(scope["issuer_id"]) == {"E1"}
    assert comparisons == {("OLD_A", "2026Q1"), ("A", "2026Q2")}
    print("SANITY: bounded history excludes issuers with no active emitter and retains the actual same-issuer predecessor across a symbol change.")
