"""
Earnings-call sentiment/text FEATURES (src/data_aggregate/utils/earnings_call_features.py).

Validates the pure feature layer on synthetic cache rows and synthetic `earnings_call_sections`
paragraph rows split by `src/utils/earnings_call_split.py` (no model/GPU):
  * the smart per-call KPI arithmetic (length-weighted tone, Q&A gap, uncertainty,
    tone delta vs prior call, disclosure-length delta),
  * the exact raw and issuer-history feature contract,
  * POINT-IN-TIME / leak-free alignment with a 66-session signal lifetime.
"""

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
def test_calendar_day_buffer_applies_once_to_all_twelve_features(call_date: str, expected: str) -> None:
    calendar = pd.bdate_range("2024-05-01", "2024-10-31").difference(pd.DatetimeIndex(["2024-05-27"]))
    call = pd.DataFrame({"ticker": ["A"], "as_of": [pd.Timestamp(call_date)], **{name: [float(i)] for i, name in enumerate(EARNINGS_CALL_FEATURES)}})
    panel = ec._feature_panel_from_prepared(call, calendar, None, availability_lag_days=2, since=None)
    assert len(panel.filter(like="f_ec_").columns) == 12
    for name in EARNINGS_CALL_FEATURES:
        live = panel.loc[panel[f"f_{name}"].notna(), "date"]
        assert live.min() == pd.Timestamp(expected) and len(live) == 66
    print(f"SANITY: {call_date}+2 calendar days -> {expected}, including holiday/weekend rollover, all12 features for66 sessions.")


def _number(value: object) -> float:
    """Narrow a scalar selected from a numeric fixture frame."""
    assert not isinstance(value, pd.Series | pd.DataFrame)
    return float(cast(float, value))


def _row(tkr, q, tag, pos, neg, words, unc):
    return {
        "ticker": tkr,
        "quarter": q,
        "tag": tag,
        "as_of": _QDATE[q],
        "sent_pos": pos,
        "sent_neg": neg,
        "sent_neu": round(1 - pos - neg, 6),
        "n_words": words,
        "uncertainty_ratio": unc,
        "model": EARNINGS_CALL_SENTIMENT_CACHE_MODEL,
    }


def _sentiment_frame() -> pd.DataFrame:
    rows = [
        # ticker A — the arithmetic-checked name
        _row("A", "2023Q1", "prepared_remarks", 0.60, 0.10, 1000, 0.02),
        _row("A", "2023Q1", "qa", 0.40, 0.10, 500, 0.05),
        _row("A", "2023Q2", "prepared_remarks", 0.70, 0.05, 1200, 0.01),
        _row("A", "2023Q2", "qa", 0.50, 0.10, 600, 0.04),
        _row("A", "2023Q3", "prepared_remarks", 0.55, 0.15, 1100, 0.03),
        _row("A", "2023Q3", "qa", 0.45, 0.20, 550, 0.06),
    ]
    # B..E: distinct tone/uncertainty so the cross-section & peer basket are non-degenerate
    for i, tkr in enumerate(["B", "C", "D", "E"], start=1):
        for q in _QDATE:
            base = 0.30 + 0.1 * i
            rows.append(_row(tkr, q, "prepared_remarks", min(0.9, base), 0.10, 900 + 50 * i, 0.02 + 0.005 * i))
            rows.append(_row(tkr, q, "qa", min(0.9, base - 0.05), 0.12, 450 + 20 * i, 0.03 + 0.005 * i))
    return pd.DataFrame(rows)


_QUESTION = "Can you talk about revenue growth and the margin guidance for next year?"
_USEFUL = "revenue growth customer demand margin guidance cash flow outlook " * 12


def test_per_call_kpi_arithmetic() -> None:
    df_sentiment = _sentiment_frame().query("ticker == 'A'")
    per = prepare_earnings_call_kpis(sentiment_kpis(df_sentiment, EARNINGS_CALL_SENTIMENT_CACHE_MODEL))
    a = per[per["ticker"] == "A"].set_index("quarter")

    # length-weighted tone (net = pos-neg), Q1: (.5*1000 + .3*500)/1500
    assert abs(_number(a.loc["2023Q1", "ec_tone"]) - (0.5 * 1000 + 0.3 * 500) / 1500) < 1e-9
    # Q&A gap = qa_net - prepared_net = .3 - .5
    assert abs(_number(a.loc["2023Q1", "ec_qa_gap"]) - (0.3 - 0.5)) < 1e-9
    # length-weighted uncertainty, Q1: (.02*1000 + .05*500)/1500
    assert abs(_number(a.loc["2023Q1", "ec_uncertainty"]) - (0.02 * 1000 + 0.05 * 500) / 1500) < 1e-9
    # tone delta Q2 vs Q1
    tone_q1 = (0.5 * 1000 + 0.3 * 500) / 1500
    tone_q2 = (0.65 * 1200 + 0.40 * 600) / 1800
    assert abs(_number(a.loc["2023Q2", "ec_tone_delta"]) - (tone_q2 - tone_q1)) < 1e-9
    assert math.isnan(_number(a.loc["2023Q1", "ec_tone_delta"]))  # first call -> no prior
    # disclosure-length delta Q2 = log(1800/1500)
    assert abs(_number(a.loc["2023Q2", "ec_length_delta"]) - math.log(1800 / 1500)) < 1e-9
    print("SANITY: exact cleaned section lengths and uncertainty counts preserve weighted tone, Q&A gap and consecutive-quarter deltas.")


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
    assert sentiment_kpis(scores.head(1), EARNINGS_CALL_SENTIMENT_CACHE_MODEL).empty
    assert sentiment_kpis(scores.assign(n_words=1), EARNINGS_CALL_SENTIMENT_CACHE_MODEL).empty
    invalid = scores.copy()
    invalid.loc[invalid.index[0], "sent_pos"] = np.nan
    assert sentiment_kpis(invalid, EARNINGS_CALL_SENTIMENT_CACHE_MODEL).empty
    assert sentiment_kpis(scores.assign(model="legacy"), EARNINGS_CALL_SENTIMENT_CACHE_MODEL).empty
    placeholder = scores.copy()
    mask = placeholder["tag"].eq("qa")
    placeholder.loc[mask, ["sent_pos", "sent_neg", "sent_neu", "uncertainty_ratio"]] = np.nan
    placeholder.loc[mask, "n_words"] = 0
    partial = sentiment_kpis(placeholder, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)
    assert len(partial) == 1 and partial["ec_qa_gap"].isna().all() and partial["coverage"].eq(1).all()
    print("SANITY: invalid/incomplete/short/legacy caches are excluded; genuine absence is distinct from failed inference.")


def test_panel_columns_lifetime_and_missingness():
    idx = pd.bdate_range("2023-01-02", "2023-09-29")
    panel = build_earnings_call_feature_panel(sentiment_kpis(_sentiment_frame(), EARNINGS_CALL_SENTIMENT_CACHE_MODEL), idx, availability_lag_days=2)
    assert not panel.empty
    expected = {
        "date",
        "ticker",
        "f_ec_tone",
        "f_ec_tone_vs_hist",
        "f_ec_qa_gap",
        "f_ec_qa_gap_vs_hist",
        "f_ec_uncertainty",
        "f_ec_uncertainty_vs_hist",
        "f_ec_qa_coherence_mean",
        "f_ec_qa_coherence_mean_vs_hist",
        "f_ec_tone_delta",
        "f_ec_length_delta",
        "f_ec_qa_qq_distance",
        "f_ec_prep_qq_distance",
    }
    assert set(panel.columns) == expected
    assert not any(c.endswith(("_xs", "_vs_peers")) for c in panel.columns)

    panel["date"] = pd.to_datetime(panel["date"])
    a = panel[panel["ticker"] == "A"]
    first_call = pd.Timestamp("2023-02-01")
    # The raw call date remains unchanged; feature eligibility waits two calendar days.
    a_tone = a[a["f_ec_tone"].notna()]
    assert a_tone["date"].min() > first_call
    assert a_tone["date"].min() == pd.Timestamp("2023-02-03")
    # 66 trading sessions are inclusive; session 67 expires rather than becoming a fake zero.
    one = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-01"], "ec_tone": [0.0]})
    daily = _daily_frame(one, "ec_tone", idx, availability_lag_days=2)
    live = daily["A"].dropna()
    assert len(live) == 66
    assert (live == 0.0).all(), "a genuine zero must survive"
    assert pd.isna(_number(daily.loc[live.index[-1] + pd.offsets.BDay(1), "A"])), "session 67 must be NaN"

    weekend = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-04"], "ec_tone": [0.2]})
    weekend_daily = _daily_frame(weekend, "ec_tone", idx, availability_lag_days=2)
    assert weekend_daily.loc[pd.Timestamp("2023-02-06"), "A"] == 0.2

    print("\n=== SANITY CHECK: earnings-call features ===")
    print(f"  panel {panel.shape[0]} rows; exactly 12 raw/issuer-history EC features and no peer/cross-sectional variants.")
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


def test_issuer_history_is_prior_only_and_requires_four_observations() -> None:
    from src.data_aggregate.utils.text import earnings_call_features as ec

    history = getattr(ec, "_issuer_history_zscore", None)
    assert history is not None, "prior-only issuer-history normalization is missing"
    calls = pd.DataFrame(
        {
            "ticker": ["A"] * 5,
            "quarter": [f"202{i}Q1" for i in range(5)],
            "as_of": pd.date_range("2020-01-01", periods=5, freq="365D"),
            "ec_tone": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )
    got = history(calls, "ec_tone")
    assert got.iloc[:4].isna().all()
    expected = (100.0 - 2.5) / np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert math.isclose(float(got.iloc[4]), expected)
    print("\n=== SANITY CHECK: issuer history ===")
    print("  first four calls are NaN; fifth uses only the prior four with sample std. Validated.")


def test_issuer_history_survives_symbol_and_cik_change() -> None:
    from src.data_aggregate.utils.text.earnings_call_features import _issuer_history_zscore

    calls = pd.DataFrame(
        {
            "ticker": ["NEW"] * 5,
            "as_of": pd.to_datetime(["2020-02-01", "2021-02-01", "2022-02-01", "2023-02-01", "2024-02-01"]),
            "ec_tone": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    lineage = pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E1"]}).assign(role="cik_window", valid_from="1900-01-01", valid_to=None)
    roster = pd.DataFrame({"ticker": ["NEW"], "cik": ["2"]})
    identified = attach_issuer_identity(calls, roster, lineage, _canonical_tenure(roster))
    score = _issuer_history_zscore(identified, "ec_tone")
    assert identified["issuer_id"].eq("E1").all()
    assert score.iloc[:4].isna().all() and pd.notna(score.iloc[4])
    print("\n=== SANITY CHECK: issuer lineage ===")
    print("  OLD/CIK1 -> NEW/CIK2 remains one issuer history through symbol and CIK change. Validated.")


def test_identity_excludes_unknown_and_ambiguous_rows_and_preserves_deltas() -> None:
    calls = pd.DataFrame(
        {
            "ticker": ["NEW", "NEW"],
            "quarter": ["2023Q4", "2024Q1"],
            "as_of": pd.to_datetime(["2023-11-01", "2024-02-01"]),
            "ec_tone": [0.2, 0.5],
            "total_words": [1000.0, 2000.0],
        }
    )
    lineage = pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E1"]}).assign(role="cik_window", valid_from="1900-01-01", valid_to=None)
    roster = pd.DataFrame({"ticker": ["NEW"], "cik": ["2"]})
    prepared = prepare_earnings_call_kpis(attach_issuer_identity(calls, roster, lineage, _canonical_tenure(roster)))
    newest = prepared.iloc[-1]
    assert math.isclose(float(newest["ec_tone_delta"]), 0.3)
    assert math.isclose(float(newest["ec_length_delta"]), math.log(2.0))

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
            "ec_tone": [0.1, 0.2],
            "ec_qa_gap": [0.3, np.nan],
            "total_words": [1000.0, 1100.0],
        }
    )
    panel = build_earnings_call_feature_panel(calls, calendar, availability_lag_days=2)
    observed = panel[panel["ticker"].eq("AAA")].set_index("date")
    assert observed.loc[pd.Timestamp("2024-02-02"), "f_ec_qa_gap"] == 0.3
    assert pd.isna(observed.loc[pd.Timestamp("2024-02-05"), "f_ec_qa_gap"])
    assert observed.loc[pd.Timestamp("2024-02-05"), "f_ec_tone"] == np.float32(0.2)
    print("\n=== SANITY CHECK: missing KPI supersedes stale signal ===")
    print("  the newer valid call preserves its observed tone but resets its absent Q&A gap to NaN. Validated.")


def test_partial_coverage_controls_deltas_and_prior_history() -> None:
    calls = pd.DataFrame(
        {
            "ticker": ["A"] * 8,
            "issuer_id": ["E1"] * 8,
            "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4"],
            "as_of": pd.date_range("2020-02-01", periods=8, freq="3MS"),
            "coverage": [1, 3, 1, 3, 1, 3, 1, 1],
            "ec_tone": [0.1, 0.9, 0.2, 0.9, 0.3, 0.9, 0.4, 0.5],
            "ec_uncertainty": [0.01, 0.9, 0.02, 0.9, 0.03, 0.9, 0.04, 0.05],
            "total_words": [1000] * 7 + [2000],
        }
    )
    result = prepare_earnings_call_kpis(calls)
    assert result["ec_tone_delta"].iloc[:7].isna().all()
    assert result["ec_length_delta"].iloc[:7].isna().all()
    assert result["ec_tone_delta"].iloc[7] == pytest.approx(0.1)
    assert result["ec_length_delta"].iloc[7] == pytest.approx(np.log(2))
    assert pd.isna(result["ec_tone_vs_hist"].iloc[6])
    expected = (0.5 - 0.25) / np.std([0.1, 0.2, 0.3, 0.4], ddof=1)
    assert result["ec_tone_vs_hist"].iloc[7] == pytest.approx(expected)
    assert result["ec_uncertainty_vs_hist"].iloc[7] == pytest.approx(expected)
    print("SANITY: mixed section sets cannot become tone/length changes; five-year history requires four prior matching-coverage observations.")


def test_tail_daily_calculation_preserves_original_expiry() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=200)
    calls = pd.DataFrame({"ticker": ["A"], "as_of": [calendar[10]], "ec_tone": [0.2]})
    full = _daily_frame(calls, "ec_tone", calendar, 2)
    for cutoff in (calendar[30], calendar[77], calendar[100]):
        tail = _daily_frame(calls, "ec_tone", calendar, 2, since=cutoff)
        pd.testing.assert_frame_equal(tail, full.loc[cutoff:], check_exact=True)
    assert _daily_frame(calls, "ec_tone", calendar, 2, since=calendar[100]).isna().all().all()
    print("SANITY: daily arrays start at the requested tail; the 66-session signal never restarts at its cutoff.")


def test_history_scope_is_bounded_and_keeps_fiscal_predecessors() -> None:
    from src.data_aggregate.utils.text.earnings_call_features import feature_history_scope

    periods = pd.period_range("2015Q1", "2026Q4", freq="Q")
    dates = periods.to_timestamp(how="end").normalize() + pd.Timedelta(days=30)
    calls = pd.DataFrame({"ticker": "A", "issuer_id": "E1", "quarter": [f"{period.year}Q{period.quarter}" for period in periods], "as_of": dates})
    calendar = pd.bdate_range("2010-01-01", "2026-09-15")
    cutoff = pd.Timestamp("2026-08-15")
    scope, comparisons = feature_history_scope(calls, calendar, cutoff, 2)
    assert len(scope) < len(calls)
    assert scope["as_of"].min() >= pd.Timestamp("2026-07-30") - pd.DateOffset(years=5)
    assert scope["as_of"].max() <= calendar[-1]
    assert comparisons == {("A", "2026Q1"), ("A", "2026Q2")}
    print("SANITY: active emitter plus actual predecessor keeps bounded five-year history; old archive/future calls are excluded.")


def test_cached_uncertainty_recovers_exact_word_hits() -> None:
    rows = _sentiment_frame().query("ticker == 'A' and quarter == '2023Q1'").copy()
    rows["n_words"] = 507
    rows["uncertainty_ratio"] = round(87 / 507, 6)
    value = sentiment_kpis(rows, EARNINGS_CALL_SENTIMENT_CACHE_MODEL)["ec_uncertainty"].iloc[0]
    assert value == 87 / 507
    print("SANITY: legacy six-decimal uncertainty recovers the exact integer-word ratio without re-reading text.")


if __name__ == "__main__":
    test_per_call_kpi_arithmetic()
    test_panel_columns_lifetime_and_missingness()
    print("\n=== SANITY CHECK: earnings-call features ===")
    print(
        "  length-weighted tone / Q&A-gap / uncertainty / tone-delta / length-delta "
        "arithmetic correct; panel emits exactly 12 raw/history fields without peer/xs variants; "
        "features are leak-free and expire after 66 trading sessions. Validated."
    )
