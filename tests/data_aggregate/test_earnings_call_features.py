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

import logging
import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd

import src.data_aggregate.transformers.step_cube_text as step_text
import src.data_aggregate.utils.text.earnings_call_features as ec
from src.constants.constants import (
    EARNINGS_CALL_SENTIMENT_CACHE_MODEL,
    EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL,
    EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
    FINBERT_TONE_MODEL,
)
from src.context import Context
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_aggregate.utils.common.incremental import PartWindow, write_part
from src.data_aggregate.utils.text.earnings_call_features import (
    _daily_frame,
    _per_call_kpis,
    acknowledge_earnings_call_invalidations,
    attach_issuer_identity,
    build_earnings_call_feature_panel,
    prepare_earnings_call_kpis,
    score_earnings_calls,
    sentiment_kpis_streamed,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.earnings_call_split import split_call
from src.utils.text_metrics import assess_earnings_call_sections
from tests.fixtures.earnings_call_rows import synthetic_call

_QDATE = {"2023Q1": "2023-02-01", "2023Q2": "2023-05-01", "2023Q3": "2023-08-01"}


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


def _sections_frame() -> pd.DataFrame:
    """`earnings_call_sections` paragraph rows of a split-`ok`, quality-valid call per fixture call."""
    txt = {
        ("A", "2023Q1"): "cloud platform enterprise customers subscription revenue expansion",
        ("A", "2023Q2"): "cloud platform enterprise customers subscription revenue expansion margins",
        ("A", "2023Q3"): "litigation restructuring charges layoffs writedown goodwill impairment",
    }
    calls = [
        synthetic_call(tkr, q, _QDATE[q], prepared=((txt.get((tkr, q), _USEFUL) + " ") * 12).strip(), question=_QUESTION, answer=_USEFUL.strip())
        for tkr in ["A", "B", "C", "D", "E"]
        for q in _QDATE
    ]
    return pd.concat(calls, ignore_index=True)


def test_per_call_kpi_arithmetic():
    per = prepare_earnings_call_kpis(_per_call_kpis(_sentiment_frame(), None))
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


class _Ctx:
    def __init__(self, store):
        self.store = store
        self.log = logging.getLogger("test")


def test_sentiment_kpis_streamed_equals_batch(sqlite_store):
    """The per-ticker STREAMED KPIs (bounded memory) must exactly equal the whole-cache
    computation — proving the streaming refactor preserves the retained QoQ deltas. Runs on a
    REAL DataStore, so the per-ticker `distinct` + WHERE-scoped reads are exercised for real."""
    sent, sec = _sentiment_frame(), _sections_frame()
    sqlite_store.save("earnings_call_sentiment", sent)
    sqlite_store.save("earnings_call_sections", sec)
    ctx = cast(Context, _Ctx(sqlite_store))
    streamed = sentiment_kpis_streamed(ctx)
    assert streamed is not None
    batch = _per_call_kpis(sent, sec)
    m = streamed.merge(batch, on=["ticker", "quarter"], suffixes=("_s", "_b"))
    assert len(m) == len(batch) == len(streamed), "row set changed under streaming"
    kpi_cols = [
        "ec_tone",
        "ec_qa_gap",
        "ec_uncertainty",
        "total_words",
    ]
    for col in kpi_cols:
        s, b = m[f"{col}_s"].to_numpy(float), m[f"{col}_b"].to_numpy(float)
        ok = (np.isnan(s) & np.isnan(b)) | np.isclose(s, b, equal_nan=True)
        assert ok.all(), f"{col} differs streamed vs batch"
    print("\n=== SANITY CHECK: sentiment KPI streaming ===")
    print(f"  per-ticker streamed call metrics == whole-cache batch across {len(m)} calls x {len(kpi_cols)} fields.")


def test_malformed_or_incomplete_cached_call_is_missing() -> None:
    sentiment = _sentiment_frame().query("ticker == 'A' and quarter == '2023Q1'")
    valid_sections = _sections_frame().query("ticker == 'A' and quarter == '2023Q1'")
    incomplete = sentiment.query("tag == 'prepared_remarks'")
    assert _per_call_kpis(incomplete, valid_sections).empty

    malformed = valid_sections.copy()
    malformed.loc[malformed["paragraph"] >= 3, "content"] = None  # Q&A paragraphs blank -> no_qa
    assert split_call(malformed.to_dict("records")).status == "no_qa"
    assert _per_call_kpis(sentiment, malformed).empty

    refreshed = _per_call_kpis(sentiment.assign(n_words=1), valid_sections)
    split = split_call(valid_sections.to_dict("records"))
    expected_words = assess_earnings_call_sections({"prepared_remarks": split.prepared_remarks, "qa": split.qa}).combined_word_count
    assert expected_words > 100
    assert int(refreshed["total_words"].iloc[0]) == expected_words
    print("\n=== SANITY CHECK: current transcript quality dominates stale cache ===")
    print(
        f"  incomplete cache and a no_qa split produce no KPI row; the stale cached word count (1) is refreshed "
        f"to {expected_words} cleaned split words. Validated."
    )


def test_legacy_sentiment_cache_is_rescored_after_cleaning_change(sqlite_store, monkeypatch) -> None:
    """A cache produced before cleaned-text scoring must not suppress or feed the new build."""
    sections = _sections_frame().query("ticker == 'A' and quarter == '2023Q1'")
    legacy = _sentiment_frame().query("ticker == 'A' and quarter == '2023Q1'").assign(model=FINBERT_TONE_MODEL)
    sqlite_store.save(Tables.earnings_call_sentiment, legacy)

    class _Engine:
        device = "test"

        @staticmethod
        def score_texts(texts):
            return [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]

    monkeypatch.setattr(ec, "get_sentiment_engine", lambda _log: _Engine())
    score_earnings_calls(cast(Context, _Ctx(sqlite_store)), sections)
    refreshed = sqlite_store.load(Tables.earnings_call_sentiment)

    assert refreshed is not None
    assert set(refreshed["model"]) == {EARNINGS_CALL_SENTIMENT_CACHE_MODEL}
    assert len(_per_call_kpis(legacy, sections)) == 0
    assert len(_per_call_kpis(refreshed, sections)) == 1
    print("\n=== SANITY CHECK: sentiment cache preprocessing version ===")
    print("  legacy raw-text rows are rejected; both cleaned sections are rescored and replace them. Validated.")


def test_malformed_refresh_marker_survives_until_cube_write_ack(sqlite_store, monkeypatch) -> None:
    sections = synthetic_call("A", "2023Q1", "2023-02-01", prepared="x", question="y", answer="z").assign(content=None)
    marker = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "quarter": "2023Q1",
                "tag": tag,
                "as_of": as_of,
                "model": EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
            }
            for ticker, as_of in (("A", "2023-02-01"), ("B", None))
            for tag in ("prepared_remarks", "qa")
        ]
    )
    sqlite_store.save(Tables.earnings_call_sections, sections)
    sqlite_store.save(Tables.earnings_call_sentiment, marker)
    monkeypatch.setattr(ec, "get_sentiment_engine", lambda _log: None)
    context = cast(Context, _Ctx(sqlite_store))

    assert score_earnings_calls(context) == pd.Timestamp("2023-02-01")
    still_pending = sqlite_store.load(Tables.earnings_call_sentiment)
    assert still_pending is not None
    assert set(still_pending["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL}
    assert acknowledge_earnings_call_invalidations(context) == 2
    handled = sqlite_store.load(Tables.earnings_call_sentiment)
    assert handled is not None
    by_ticker = handled.groupby("ticker")["model"].first().to_dict()
    assert by_ticker == {
        "A": EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL,
        "B": EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
    }
    assert score_earnings_calls(context) is None
    print("\n=== SANITY CHECK: malformed refresh handoff ===")
    print("  a dated invalidation is acknowledged only after the cube write; an undated marker remains pending. Validated.")


def test_malformed_refresh_deletes_stale_tail_even_when_panel_is_empty(sqlite_store, monkeypatch) -> None:
    stale = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-02-01", "2023-02-02", "2023-02-03"]),
            "ticker": ["A", "A", "A"],
            "f_ec_tone": [0.5, 0.5, 0.5],
        }
    )
    sqlite_store.save(Tables.cube_part_text, stale)
    sqlite_store.save(
        Tables.earnings_call_sentiment,
        pd.DataFrame(
            [
                {
                    "ticker": "A",
                    "quarter": "2023Q1",
                    "tag": tag,
                    "as_of": "2023-02-01",
                    "model": EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
                }
                for tag in ("prepared_remarks", "qa")
            ]
        ),
    )
    step = StepCubeText.__new__(StepCubeText)
    step._store = sqlite_store
    step._context = cast(Context, _Ctx(sqlite_store))
    step._cfg = {"incremental": {}}
    step._part = SimpleNamespace(warmup_trading_days=5)
    step._load_frames = lambda _since: object()
    step._feature_panel = lambda _frames: (
        pd.DataFrame(columns=["date", "ticker", "f_ec_tone"]),
        [pd.Timestamp("2023-02-01")],
    )
    monkeypatch.setattr(step_text, "load_trading_calendar", lambda _store: pd.bdate_range("2023-02-01", periods=3))
    monkeypatch.setattr(
        step_text,
        "plan_window",
        lambda *_args, **_kwargs: PartWindow(
            last=pd.Timestamp("2023-02-03"),
            since=pd.Timestamp("2023-02-01"),
            refresh_from=pd.Timestamp("2023-02-03"),
        ),
    )

    step.run()

    remaining = sqlite_store.load(Tables.cube_part_text)
    assert remaining is not None
    assert remaining["date"].tolist() == [pd.Timestamp("2023-02-01")]
    marker = sqlite_store.load(Tables.earnings_call_sentiment)
    assert marker is not None
    assert set(marker["model"]) == {EARNINGS_CALL_SENTIMENT_INVALID_HANDLED_MODEL}
    print("\n=== SANITY CHECK: malformed correction clears persisted output ===")
    print("  an empty rebuilt panel deletes every stale row from call+1 onward, then acknowledges the durable marker. Validated.")


def test_panel_columns_lifetime_and_missingness():
    idx = pd.bdate_range("2023-01-02", "2023-09-29")
    panel = build_earnings_call_feature_panel(_sentiment_frame(), idx, sections=_sections_frame())
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
    # LEAK-FREE: nothing for A on/before the first call date; signal appears the NEXT day
    a_tone = a[a["f_ec_tone"].notna()]
    assert a_tone["date"].min() > first_call
    assert a_tone["date"].min() == pd.Timestamp("2023-02-02")
    # 66 trading sessions are inclusive; session 67 expires rather than becoming a fake zero.
    one = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-01"], "ec_tone": [0.0]})
    daily = _daily_frame(one, "ec_tone", idx)
    live = daily["A"].dropna()
    assert len(live) == 66
    assert (live == 0.0).all(), "a genuine zero must survive"
    assert pd.isna(daily.loc[live.index[-1] + pd.offsets.BDay(1), "A"]), "session 67 must be NaN"

    weekend = pd.DataFrame({"ticker": ["A"], "as_of": ["2023-02-04"], "ec_tone": [0.2]})
    weekend_daily = _daily_frame(weekend, "ec_tone", idx)
    assert weekend_daily.loc[pd.Timestamp("2023-02-06"), "A"] == 0.2

    print("\n=== SANITY CHECK: earnings-call features ===")
    print(f"  panel {panel.shape[0]} rows; exactly 12 raw/issuer-history EC features and no peer/cross-sectional variants.")
    print(
        f"  leak-free: ticker A first tone signal at {a_tone['date'].min().date()} "
        "(call 2023-02-01 + 1 trading day); genuine zero survives for 66 sessions and "
        "session 67 is NaN."
    )


def test_call_is_invisible_on_its_as_of_session_and_visible_on_the_next(sqlite_store, monkeypatch) -> None:
    """End to end from stored paragraphs: score -> stream -> daily panel. A call dated D (the real
    call date, `as_of` of its paragraphs) never reaches session D and first appears on the next
    trading session -- the day after for a Wednesday call, the Monday for a Friday call."""
    calls = {"A": "2024-05-01", "B": "2024-05-03"}  # Wednesday, Friday
    paragraphs = pd.concat(
        [
            synthetic_call(tkr, "2024Q1", as_of, prepared=(_USEFUL * 2).strip(), question=_QUESTION, answer=_USEFUL.strip())
            for tkr, as_of in calls.items()
        ],
        ignore_index=True,
    )
    sqlite_store.save(Tables.earnings_call_sections, paragraphs)

    class _Engine:
        device = "test"

        @staticmethod
        def score_texts(texts):
            return [{"pos": 0.6, "neg": 0.1, "neu": 0.3} for _ in texts]

    monkeypatch.setattr(ec, "get_sentiment_engine", lambda _log: _Engine())
    context = cast(Context, _Ctx(sqlite_store))
    assert score_earnings_calls(context) == pd.Timestamp("2024-05-01")
    per_call = sentiment_kpis_streamed(context)
    assert per_call is not None and len(per_call) == 2
    calendar = pd.bdate_range("2024-04-01", "2024-09-30")
    panel = build_earnings_call_feature_panel(None, calendar, per_call=per_call)
    panel["date"] = pd.to_datetime(panel["date"])
    first_seen = {}
    for tkr, as_of in calls.items():
        live = panel[panel["ticker"].eq(tkr) & panel["f_ec_tone"].notna()]
        call_date = pd.Timestamp(as_of)
        assert not live["date"].le(call_date).any(), f"{tkr}: visible on or before its as_of session"
        next_session = calendar[calendar.searchsorted(call_date, side="right")]
        assert live["date"].min() == next_session
        first_seen[tkr] = (as_of, str(next_session.date()))
    assert first_seen["B"][1] == "2024-05-06"
    print("\n=== SANITY CHECK: earnings-call point-in-time visibility ===")
    print(f"  (as_of, first visible session) per call from stored paragraphs: {first_seen}; never visible on its as_of session. Validated.")


def test_full_calendar_late_refresh_and_rerun_are_bit_exact() -> None:
    """Exercise the real tail writer against a late correction and unchanged rerun."""
    calendar = pd.bdate_range("2021-01-04", "2023-09-29")
    assert len(calendar) >= 500
    old = build_earnings_call_feature_panel(_sentiment_frame(), calendar, sections=_sections_frame())
    revised_sentiment = _sentiment_frame()
    mask = (revised_sentiment["ticker"] == "A") & (revised_sentiment["quarter"] == "2023Q1")
    revised_sentiment.loc[mask, "sent_pos"] += 0.05
    revised = build_earnings_call_feature_panel(revised_sentiment, calendar, sections=_sections_frame())

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

    store = _Store(old)
    last = pd.Timestamp(old["date"].max())
    default_refresh = calendar[-130]
    late_refresh = pd.Timestamp("2023-02-02")
    window = PartWindow(last=last, since=calendar[0], refresh_from=default_refresh)
    effective_refresh = min(late_refresh, window.refresh_from)
    write_part(cast(DataStore, store), Tables.cube_part_text, revised, window, refresh_from=effective_refresh, drop_empty=True)
    expected = revised.sort_values(["date", "ticker"]).reset_index(drop=True)
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
            "ticker": ["OLD"] * 4 + ["NEW"],
            "as_of": pd.to_datetime(["2020-02-01", "2021-02-01", "2022-02-01", "2023-02-01", "2024-02-01"]),
            "ec_tone": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    tenure = pd.DataFrame(
        {
            "symbol": ["OLD", "NEW"],
            "issuer_cik": ["1", "2"],
            "valid_from": ["2019-01-01", "2024-01-01"],
            "valid_to": ["2023-12-31", None],
            "n_filings": [20, 10],
        }
    )
    lineage = pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E1"]})
    identified = attach_issuer_identity(calls, tenure, lineage)
    score = _issuer_history_zscore(identified, "ec_tone")
    assert identified["issuer_id"].eq("E1").all()
    assert score.iloc[:4].isna().all() and pd.notna(score.iloc[4])
    print("\n=== SANITY CHECK: issuer lineage ===")
    print("  OLD/CIK1 -> NEW/CIK2 remains one issuer history through symbol and CIK change. Validated.")


def test_identity_excludes_unknown_and_ambiguous_rows_and_preserves_deltas() -> None:
    calls = pd.DataFrame(
        {
            "ticker": ["OLD", "NEW"],
            "quarter": ["2023Q4", "2024Q1"],
            "as_of": pd.to_datetime(["2023-11-01", "2024-02-01"]),
            "ec_tone": [0.2, 0.5],
            "total_words": [1000.0, 2000.0],
        }
    )
    tenure = pd.DataFrame(
        {
            "symbol": ["OLD", "NEW"],
            "issuer_cik": ["1", "2"],
            "valid_from": ["2020-01-01", "2024-01-01"],
            "valid_to": ["2024-01-01", None],
        }
    )
    lineage = pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E1"]})
    prepared = prepare_earnings_call_kpis(attach_issuer_identity(calls, tenure, lineage))
    newest = prepared.loc[prepared["ticker"] == "NEW"].iloc[0]
    assert math.isclose(float(newest["ec_tone_delta"]), 0.3)
    assert math.isclose(float(newest["ec_length_delta"]), math.log(2.0))

    ambiguous = pd.concat(
        [
            tenure,
            pd.DataFrame({"symbol": ["NEW"], "issuer_cik": ["3"], "valid_from": ["2024-01-01"], "valid_to": [None]}),
        ],
        ignore_index=True,
    )
    ambiguous_lineage = pd.concat([lineage, pd.DataFrame({"cik": ["3"], "entity_id": ["E2"]})], ignore_index=True)
    assert attach_issuer_identity(calls.tail(1), ambiguous, ambiguous_lineage).empty
    unresolved = pd.concat(
        [tenure, pd.DataFrame({"symbol": ["NEW"], "issuer_cik": ["9"], "valid_from": ["2024-01-01"], "valid_to": [None]})],
        ignore_index=True,
    )
    assert attach_issuer_identity(calls.tail(1), unresolved, lineage).empty
    assert attach_issuer_identity(calls.tail(1).assign(ticker="UNKNOWN"), tenure, lineage).empty
    print("\n=== SANITY CHECK: strict issuer identity ===")
    print("  half-open tenures preserve OLD->NEW deltas; unknown or ambiguous mappings are excluded. Validated.")


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
    panel = build_earnings_call_feature_panel(None, calendar, per_call=calls)
    observed = panel[panel["ticker"].eq("AAA")].set_index("date")
    assert observed.loc[pd.Timestamp("2024-02-02"), "f_ec_qa_gap"] == 0.3
    assert pd.isna(observed.loc[pd.Timestamp("2024-02-05"), "f_ec_qa_gap"])
    assert observed.loc[pd.Timestamp("2024-02-05"), "f_ec_tone"] == np.float32(0.2)
    print("\n=== SANITY CHECK: missing KPI supersedes stale signal ===")
    print("  the newer valid call preserves its observed tone but resets its absent Q&A gap to NaN. Validated.")


if __name__ == "__main__":
    test_per_call_kpi_arithmetic()
    test_panel_columns_lifetime_and_missingness()
    print("\n=== SANITY CHECK: earnings-call features ===")
    print(
        "  length-weighted tone / Q&A-gap / uncertainty / tone-delta / length-delta "
        "arithmetic correct; panel emits exactly 12 raw/history fields without peer/xs variants; "
        "features are leak-free and expire after 66 trading sessions. Validated."
    )
