from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.constants.constants import EARNINGS_CALL_FEATURES
from src.context import Context
from src.data_store.schema import Tables
from src.validate.checks.earnings_calls import _coverage, check_earnings_calls
from tests.fixtures.earnings_call_rows import synthetic_call

_USEFUL = "Revenue growth and margin guidance remained strong for customers this quarter. " * 12
_QUESTION = "Can you talk about revenue growth and the margin guidance for next year?"


def _valid_call(ticker: str, quarter: str, as_of: str | None) -> pd.DataFrame:
    return synthetic_call(ticker, quarter, as_of, prepared=_USEFUL.strip(), question=_QUESTION, answer=_USEFUL.strip())


def _short_call(ticker: str, quarter: str, as_of: str) -> pd.DataFrame:
    """Splits `ok` but fails the quality gate: well under the minimum cleaned words."""
    return synthetic_call(
        ticker,
        quarter,
        as_of,
        prepared="Revenue grew this quarter on strong demand across all of our segments and regions worldwide.",
        question="How is demand trending?",
        answer="Demand is trending well across segments.",
    )


def _config() -> object:
    return OmegaConf.merge(OmegaConf.load("configs/validate.yml"), OmegaConf.create({"train": {"end_date": "2022-01-01"}}))


def _cube(tickers: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-05-16", periods=120)
    cube = pd.DataFrame({"date": np.repeat(dates, len(tickers)), "ticker": tickers * len(dates)})
    for position, name in enumerate(EARNINGS_CALL_FEATURES):
        cube[f"f_{name}"] = np.sin(np.arange(len(cube)) / (7.0 + position))
    cube["f_ec_uncertainty"] = cube["f_ec_uncertainty"].abs().clip(upper=1)
    cube["f_ec_qa_coherence_mean"] = cube["f_ec_qa_coherence_mean"].clip(-1, 1)
    cube["f_ec_qa_qq_distance"] = cube["f_ec_qa_qq_distance"].abs().clip(upper=2)
    cube["f_ec_prep_qq_distance"] = cube["f_ec_prep_qq_distance"].abs().clip(upper=2)
    return cube


def test_earnings_call_validator_reports_coverage_schema_and_quality(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BBB"]}))
    sqlite_store.save(
        Tables.earnings_surprises,
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "earnings_date": pd.to_datetime(["2024-05-15", "2024-05-15"]),
                "eps_actual": [1.0, 1.0],
            }
        ),
    )
    paragraphs = pd.concat([_valid_call("AAA", "2024Q1", "2024-05-15"), _short_call("BBB", "2024Q1", "2024-05-15")], ignore_index=True)
    sqlite_store.save(Tables.earnings_call_sections, paragraphs)
    sqlite_store.save(Tables.cube_part_text, _cube(["AAA", "BBB"]))

    context = cast(Context, SimpleNamespace(store=sqlite_store))
    result = check_earnings_calls(context, Tables.cube_part_text, config=_config())
    coverage = cast(dict, result.metrics["coverage"])
    assert result.status == "pass", result.findings
    assert coverage["coverage_100pct"] == 1
    assert coverage["coverage_lt50pct"] == 1
    assert coverage["malformed_calls"] == 1
    assert coverage["rejected_by_reason"] == {"below minimum cleaned words": 1}
    assert coverage["split_status_counts"] == {"ok": 2, "no_qa": 0, "no_prepared": 0, "empty": 0}
    assert coverage["split_ok_rate"] == 1.0
    assert coverage["prepared_share_quantiles"]["q05"] >= 0.15
    assert coverage["grain"]["rows"] == 10 and coverage["grain"]["calls"] == 2
    assert coverage["coverage_by_calendar_quarter"] == [
        {"quarter": "2024Q2", "tickers_with_call": 2, "tickers_with_valid_call": 1, "roster_measured": 2, "share_with_call": 1.0}
    ]
    assert coverage["as_of_vs_release_days"]["same_day"] == 1
    assert result.scope["feature_columns"] == 12
    print("\n=== SANITY CHECK: dedicated earnings-call validator ===")
    print(
        "  exact 12-column schema measured from paragraph rows; both calls split ok (rate 1.0, prepared share q05 "
        f"{coverage['prepared_share_quantiles']['q05']:.2f}); the valid call is 100% covered, the short one fails the gate "
        "and stays <50%; 2024Q2 has 2/2 roster tickers with a call, 1 valid; as_of matches the release day. Validated."
    )


def test_grain_and_split_defects_fail_the_validator(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]}))
    sqlite_store.save(Tables.earnings_surprises, pd.DataFrame({"ticker": ["AAA"], "earnings_date": pd.to_datetime(["2024-05-15"])}))
    undated = _valid_call("AAA", "2024Q1", "2024-05-15")
    undated.loc[undated["paragraph"] == 5, "as_of"] = None
    no_qa = _valid_call("BBB", "2024Q1", "2024-05-15")
    no_qa = no_qa[no_qa["paragraph"] <= 2]  # prepared remarks only: no Q&A start
    gap = _valid_call("CCC", "2024Q1", "2024-05-15")
    gap = gap[gap["paragraph"] != 1]  # starts at paragraph 2
    sqlite_store.save(Tables.earnings_call_sections, pd.concat([undated, no_qa, gap], ignore_index=True))
    sqlite_store.save(Tables.cube_part_text, _cube(["AAA", "BBB", "CCC"]))

    result = check_earnings_calls(cast(Context, SimpleNamespace(store=sqlite_store)), Tables.cube_part_text, config=_config())
    coverage = cast(dict, result.metrics["coverage"])
    fields = {finding.field: finding.score for finding in result.findings}
    assert result.status == "fail"
    assert coverage["grain"]["null_as_of_rows"] == 1 and fields["null_as_of_rows"] == 9
    assert coverage["grain"]["calls_missing_first_paragraph"] == 1 and fields["calls_missing_first_paragraph"] == 6
    assert coverage["split_status_counts"]["no_qa"] == 1 and fields["split_status"] == 7
    assert coverage["rejected_by_reason"]["split status no_qa"] == 1
    print("\n=== SANITY CHECK: earnings-call source grain and split quality ===")
    print(
        f"  a null as_of (score 9), a call without paragraph 1 (score 6) and a no_qa split (ok rate "
        f"{coverage['split_ok_rate']:.2f} < 0.985, score 7) each file a finding; status={result.status}. Validated."
    )


def test_two_calls_on_one_ticker_date_fail_the_validator(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BBB"]}))
    sqlite_store.save(Tables.earnings_surprises, pd.DataFrame({"ticker": ["AAA"], "earnings_date": pd.to_datetime(["2024-05-15"])}))
    # AAA: a fiscal relabel left 2023Q4 and 2024Q4 on the same call date. BBB: two calls, two dates.
    paragraphs = pd.concat(
        [
            _valid_call("AAA", "2023Q4", "2024-05-15"),
            _valid_call("AAA", "2024Q4", "2024-05-15"),
            _valid_call("AAA", "2024Q1", "2024-08-15"),
            _valid_call("BBB", "2024Q1", "2024-05-15"),
            _valid_call("BBB", "2024Q2", "2024-08-15"),
        ],
        ignore_index=True,
    )
    sqlite_store.save(Tables.earnings_call_sections, paragraphs)
    sqlite_store.save(Tables.cube_part_text, _cube(["AAA", "BBB"]))

    result = check_earnings_calls(cast(Context, SimpleNamespace(store=sqlite_store)), Tables.cube_part_text, config=_config())
    coverage = cast(dict, result.metrics["coverage"])
    fields = {finding.field: finding.score for finding in result.findings}

    assert coverage["grain"]["ticker_dates_with_multiple_calls"] == 1
    assert fields["ticker_dates_with_multiple_calls"] == 8 and result.status == "fail"
    print("\n=== SANITY CHECK: one call per (ticker, as_of) ===")
    print(f"  AAA 2023Q4 + 2024Q4 on 2024-05-15 -> ticker_dates_with_multiple_calls={coverage['grain']['ticker_dates_with_multiple_calls']}")
    print(
        f"  BBB two calls on two dates -> not counted; finding score {fields['ticker_dates_with_multiple_calls']}; status={result.status}. Validated."
    )


def test_coverage_uses_point_in_time_lineage_and_separates_no_call_names(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["NEW", "BRK-B"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E2"], "role": "cik_window", "symbol": "", "valid_from": "1900-01-01"}),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            {
                "symbol": ["OLD", "OLD", "NEW"],
                "issuer_cik": ["1", "2", "1"],
                "valid_from": ["2020-01-01", "2024-04-01", "2024-01-01"],
                "valid_to": ["2024-04-01", None, None],
                "n_filings": [10, 10, 10],
                "source": ["form345"] * 3,
                "evidence_period": [""] * 3,
            }
        ),
    )
    events = [
        ("OLD", "2023-11-15", "2023Q4"),
        ("NEW", "2024-02-15", "2024Q1"),
        ("OLD", "2024-08-15", "2024Q3"),  # reused OLD now belongs to E2, not NEW/E1
    ]
    sqlite_store.save(
        Tables.earnings_surprises,
        pd.DataFrame({"ticker": [event[0] for event in events], "earnings_date": [event[1] for event in events]}),
    )
    sqlite_store.save(
        Tables.earnings_call_sections,
        pd.concat([_valid_call(ticker, quarter, date) for ticker, date, quarter in events], ignore_index=True),
    )

    summary, ratios, valid_dates = _coverage(cast(Context, SimpleNamespace(store=sqlite_store)))

    assert summary["roster_tickers"] == 2
    assert summary["structural_no_call_tickers"] == ["BRK-B"]
    assert summary["tickers_measured"] == 1
    assert ratios["NEW"] == 1.0
    assert len(valid_dates["NEW"]) == 2
    assert [row["quarter"] for row in summary["coverage_by_calendar_quarter"]] == ["2023Q4", "2024Q1"]
    print("\n=== SANITY CHECK: point-in-time coverage identity ===")
    print(
        "  OLD history follows E1 into NEW, reused OLD/E2 events are excluded (also from the per-quarter table), "
        "and BRK-B is structural, not a source failure. Validated."
    )


def test_coverage_reads_distinct_lineage_pairs_and_only_form345_manual_tenure(sqlite_store) -> None:
    """The dated `entity_lineage` holds several rows per CIK and `symbol_tenure` gains `dei` rows; coverage must not move."""
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["NEW"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame(
            {
                "cik": ["1", "1", "1", "2", "2"],
                "entity_id": ["E1", "E1", "E1", "E2", "E2"],
                "role": ["cik_window", "symbol", "symbol", "cik_window", "symbol"],
                "symbol": ["", "OLD", "NEW", "", "OLD"],
                "valid_from": ["1900-01-01", "2020-01-01", "2024-01-01", "1900-01-01", "2024-04-01"],
            }
        ),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            {
                "symbol": ["OLD", "OLD", "NEW", "NEW"],
                "issuer_cik": ["1", "2", "1", "9"],
                "valid_from": ["2020-01-01", "2024-04-01", "2024-01-01", "2023-06-01"],
                "valid_to": ["2024-04-01", None, None, None],
                "n_filings": [10, 10, 10, 2],
                "source": ["form345", "form345", "manual", "dei"],
                "evidence_period": ["", "", "", "2025q1"],
            }
        ),
    )
    events = [("OLD", "2023-11-15", "2023Q4"), ("NEW", "2024-02-15", "2024Q1"), ("OLD", "2024-08-15", "2024Q3")]
    sqlite_store.save(
        Tables.earnings_surprises,
        pd.DataFrame({"ticker": [event[0] for event in events], "earnings_date": [event[1] for event in events]}),
    )
    sqlite_store.save(
        Tables.earnings_call_sections,
        pd.concat([_valid_call(ticker, quarter, date) for ticker, date, quarter in events], ignore_index=True),
    )

    summary, ratios, valid_dates = _coverage(cast(Context, SimpleNamespace(store=sqlite_store)))

    assert summary["tickers_measured"] == 1
    assert ratios["NEW"] == 1.0
    assert [str(date.date()) for date in valid_dates["NEW"]] == ["2023-11-15", "2024-02-15"]
    print("\n=== SANITY CHECK: dated lineage and dei rows ===")
    print(
        "  five lineage rows over two CIKs read as two (cik, entity) pairs; a dei NEW row under a foreign CIK is "
        "not read, so NEW keeps its entity and its OLD history (2 valid calls, coverage 1.0). Validated."
    )
