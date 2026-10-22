from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants import EARNINGS_CALL_FEATURES
from src.context import Context
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.validate.checks.earnings_calls import _coverage, _identity, _linked, check_earnings_calls
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


def _seed_identity(store: DataStore, tickers: list[str]) -> None:
    roster = pd.DataFrame({"ticker": tickers, "cik": [str(i) for i in range(1, len(tickers) + 1)]})
    store.save(Tables.sp500_tickers, roster)
    store.save(
        Tables.entity_lineage,
        roster[["cik"]].assign(entity_id="E" + roster["cik"], role="cik_window", symbol="", valid_from="1900-01-01", valid_to=None),
    )
    store.save(
        Tables.symbol_tenure,
        roster.rename(columns={"ticker": "symbol", "cik": "issuer_cik"}).assign(
            valid_from="1900-01-01", valid_to=None, source="manual", evidence_period=""
        ),
    )


def test_coverage_keeps_ingestion_validated_canonical_history_before_rename(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["ECHO"], "cik": ["1"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame({"cik": ["1"], "entity_id": ["E1"], "role": ["cik_window"], "symbol": [""], "valid_from": ["1900-01-01"], "valid_to": [None]}),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            {
                "symbol": ["SATS", "ECHO"],
                "issuer_cik": ["1", "1"],
                "valid_from": ["2008-01-01", "2026-06-24"],
                "valid_to": ["2026-06-24", None],
                "source": ["manual", "manual"],
                "evidence_period": ["", ""],
            }
        ),
    )
    dates = ["2008-05-13", "2026-08-03"]
    sqlite_store.save(Tables.earnings_surprises, pd.DataFrame({"ticker": ["SATS", "ECHO"], "earnings_date": dates}))
    sqlite_store.save(
        Tables.earnings_call_sections,
        pd.concat([_valid_call("ECHO", quarter, date) for quarter, date in zip(["2008Q1", "2026Q2"], dates, strict=True)], ignore_index=True),
    )
    summary, _, valid = _coverage(
        cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[]))))
    )
    assert valid["ECHO"] == list(pd.to_datetime(dates))
    assert summary["valid_calls"] == 2
    print("SANITY: validator retains canonical ECHO history before the market rename; raw alias release dates remain dated.")


def test_coverage_excludes_canonical_calls_outside_consolidating_windows(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["CEG"], "cik": ["1"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame({"cik": ["1"], "entity_id": ["E1"], "role": ["cik_window"], "symbol": [""], "valid_from": ["2022-01-01"], "valid_to": [None]}),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            {"symbol": ["CEG"], "issuer_cik": ["1"], "valid_from": ["2022-01-01"], "valid_to": [None], "source": ["manual"], "evidence_period": [""]}
        ),
    )
    dates = ["2008-01-30", "2024-05-09"]
    sqlite_store.save(Tables.earnings_surprises, pd.DataFrame({"ticker": ["CEG"], "earnings_date": [dates[1]]}))
    sqlite_store.save(
        Tables.earnings_call_sections,
        pd.concat([_valid_call("CEG", quarter, date) for quarter, date in zip(["2007Q4", "2024Q1"], dates, strict=True)], ignore_index=True),
    )
    summary, _, valid = _coverage(
        cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[]))))
    )
    assert valid["CEG"] == [pd.Timestamp(dates[1])]
    assert [row["quarter"] for row in summary["coverage_by_calendar_quarter"]] == ["2024Q2"]
    print("SANITY: validator excludes stored canonical CEG from2008 before its consolidating window;2024 call remains usable.")


def test_canonical_coverage_checks_alias_lifetime_without_rejecting_supported_renames() -> None:
    roster = pd.DataFrame({"ticker": ["ECHO", "FOXA", "CRH"], "cik": ["1", "2", "3"]})
    lineage = pd.DataFrame({"cik": ["1", "2", "3"], "entity_id": ["E1", "E2", "E3"]}).assign(
        role="cik_window", valid_from="1900-01-01", valid_to=None
    )
    tenure = pd.DataFrame(
        {
            "symbol": ["ECHO", "SATS", "FOX", "FOX", "CRH"],
            "issuer_cik": ["9", "1", "8", "2", "3"],
            "valid_from": ["2009-10-01", "2008-01-02", "2013-07-02", "2019-02-05", "2023-09-25"],
            "valid_to": ["2021-11-24", None, "2019-03-21", None, None],
        }
    )
    identity = _identity(tenure, lineage, roster["ticker"].tolist(), roster)
    dates = {
        "ECHO": [pd.Timestamp("2015-05-13")],
        "FOXA": [pd.Timestamp("2013-04-30"), pd.Timestamp("2024-02-01")],
        "CRH": [pd.Timestamp("2008-02-01")],
    }
    linked = {ticker: _linked(dates, {ticker}, identity.roster_entity[ticker], identity, canonical_ticker=ticker) for ticker in dates}
    assert linked == {"ECHO": dates["ECHO"], "FOXA": dates["FOXA"][1:], "CRH": dates["CRH"]}
    print(
        "SANITY: canonical coverage preserves SATS-supported ECHO and uncontested pre-US CRH; reused FOX excludes even a date before its first foreign filing."
    )


def test_canonical_coverage_excludes_missing_and_contradictory_issuer_identity() -> None:
    roster = pd.DataFrame({"ticker": ["NEW"], "cik": ["1"]})
    lineage = pd.DataFrame({"cik": ["1"], "entity_id": ["E1"]}).assign(role="cik_window", valid_from="1900-01-01", valid_to=None)
    tenure = pd.DataFrame({"symbol": ["NEW"], "issuer_cik": ["1"], "valid_from": ["2020-01-01"], "valid_to": [None]})
    date = pd.Timestamp("2024-02-01")
    contradictory = pd.concat([lineage, lineage.assign(entity_id=None)], ignore_index=True)
    for evidence in (None, contradictory):
        identity = _identity(tenure, evidence, ["NEW"], roster)
        assert not _linked({"NEW": [date]}, {"NEW"}, identity.roster_entity.get("NEW"), identity, canonical_ticker="NEW")
    identity = _identity(tenure.assign(issuer_cik="9"), lineage, ["NEW"], roster)
    assert not _linked({"NEW": [date]}, {"NEW"}, identity.roster_entity.get("NEW"), identity, canonical_ticker="NEW")
    print("SANITY: missing lineage and a null/known contradictory CIK membership cannot certify a canonical call.")


@pytest.mark.parametrize(("variant", "expected"), [("overlap", [1, 3]), ("same_entity", [0, 1, 3]), ("alternate", [0, 1, 3]), ("unknown", [1, 3])])
def test_canonical_validator_requires_unambiguous_alias_inside_its_cik_window(variant: str, expected: list[int]) -> None:
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
    dates = list(pd.to_datetime(["2010-05-01", "2015-01-01", "2024-05-01", "2030-01-01"]))
    identity = _identity(tenure, lineage, ["NEW"], roster)
    linked = _linked({"NEW": dates}, {"NEW"}, identity.roster_entity["NEW"], identity, canonical_ticker="NEW")
    assert linked == [dates[position] for position in expected]
    print(
        f"SANITY: independent {variant} admission rejects ambiguous or wrong-CIK support while retaining half-open and valid alternate-alias dates."
    )


def test_earnings_call_validator_reports_coverage_schema_and_quality(sqlite_store) -> None:
    _seed_identity(sqlite_store, ["AAA", "BBB"])
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

    context = cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[]))))
    result = check_earnings_calls(context, Tables.cube_part_text, config=_config())
    coverage = cast(dict, result.metrics["coverage"])
    assert result.status == "fail"
    assert any(finding.field == "usable_call_rate" for finding in result.findings)
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
    _seed_identity(sqlite_store, ["AAA", "BBB", "CCC"])
    sqlite_store.save(Tables.earnings_surprises, pd.DataFrame({"ticker": ["AAA"], "earnings_date": pd.to_datetime(["2024-05-15"])}))
    undated = _valid_call("AAA", "2024Q1", "2024-05-15")
    undated.loc[undated["paragraph"] == 5, "as_of"] = None
    no_qa = _valid_call("BBB", "2024Q1", "2024-05-15")
    no_qa = no_qa[no_qa["paragraph"] <= 2]  # prepared remarks only: no Q&A start
    gap = _valid_call("CCC", "2024Q1", "2024-05-15")
    gap = gap[gap["paragraph"] != 1]  # starts at paragraph 2
    sqlite_store.save(Tables.earnings_call_sections, pd.concat([undated, no_qa, gap], ignore_index=True))
    sqlite_store.save(Tables.cube_part_text, _cube(["AAA", "BBB", "CCC"]))

    result = check_earnings_calls(
        cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[])))),
        Tables.cube_part_text,
        config=_config(),
    )
    coverage = cast(dict, result.metrics["coverage"])
    fields = {finding.field: finding.score for finding in result.findings}
    assert result.status == "fail"
    assert coverage["grain"]["null_as_of_rows"] == 1 and fields["null_as_of_rows"] == 9
    assert coverage["grain"]["calls_missing_first_paragraph"] == 1 and fields["calls_missing_first_paragraph"] == 6
    assert coverage["split_status_counts"]["no_qa"] == 1 and fields["usable_call_rate"] == 7
    assert "split status no_qa" not in coverage["rejected_by_reason"]
    print("\n=== SANITY CHECK: earnings-call source grain and split quality ===")
    print(
        f"  a null as_of (score 9), a call without paragraph 1 (score 6) and insufficient usable coverage (raw full-split rate "
        f"{coverage['split_ok_rate']:.2f} < 0.985, score 7) are reported; valid partial calls stay accepted; status={result.status}. Validated."
    )


def test_two_calls_on_one_ticker_date_fail_the_validator(sqlite_store) -> None:
    _seed_identity(sqlite_store, ["AAA", "BBB"])
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

    result = check_earnings_calls(
        cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[])))),
        Tables.cube_part_text,
        config=_config(),
    )
    coverage = cast(dict, result.metrics["coverage"])
    fields = {finding.field: finding.score for finding in result.findings}

    assert coverage["grain"]["ticker_dates_with_multiple_calls"] == 1
    assert fields["ticker_dates_with_multiple_calls"] == 8 and result.status == "fail"
    print("\n=== SANITY CHECK: one call per (ticker, as_of) ===")
    print(f"  AAA 2023Q4 + 2024Q4 on 2024-05-15 -> ticker_dates_with_multiple_calls={coverage['grain']['ticker_dates_with_multiple_calls']}")
    print(
        f"  BBB two calls on two dates -> not counted; finding score {fields['ticker_dates_with_multiple_calls']}; status={result.status}. Validated."
    )


@pytest.mark.parametrize("has_call", [False, True])
def test_coverage_uses_point_in_time_lineage_and_provider_presence(sqlite_store: Any, has_call: bool) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["NEW", "BRK-B"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame(
            {
                "cik": ["1", "2", "3"],
                "entity_id": ["E1", "E2", "E3"],
                "role": "cik_window",
                "symbol": "",
                "valid_from": "1900-01-01",
                "valid_to": None,
            }
        ),
    )
    sqlite_store.save(
        Tables.symbol_tenure,
        pd.DataFrame(
            {
                "symbol": ["OLD", "OLD", "NEW", "BRK-B"],
                "issuer_cik": ["1", "2", "1", "3"],
                "valid_from": ["2020-01-01", "2024-04-01", "2024-01-01", "2020-01-01"],
                "valid_to": ["2024-04-01", None, None, None],
                "n_filings": [10] * 4,
                "source": ["form345"] * 4,
                "evidence_period": [""] * 4,
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
        pd.DataFrame({"ticker": [event[0] for event in events] + ["BRK-B"], "earnings_date": [event[1] for event in events] + ["2024-02-15"]}),
    )
    sqlite_store.save(
        Tables.earnings_call_sections,
        pd.concat(
            [_valid_call(ticker, quarter, date) for ticker, date, quarter in events]
            + ([_valid_call("BRK-B", "2024Q1", "2024-02-15")] if has_call else []),
            ignore_index=True,
        ),
    )

    summary, ratios, valid_dates = _coverage(
        cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[]))))
    )

    assert summary["roster_tickers"] == 2
    assert summary["tickers_measured"] == 2
    assert ratios["NEW"] == 1.0
    assert ratios["BRK-B"] == float(has_call)
    assert len(valid_dates["BRK-B"]) == int(has_call)
    assert len(valid_dates["NEW"]) == 2
    assert [row["quarter"] for row in summary["coverage_by_calendar_quarter"]] == ["2023Q4", "2024Q1"]
    print("\n=== SANITY CHECK: point-in-time coverage identity ===")
    print(
        "  OLD history follows E1 into NEW, reused OLD/E2 events are excluded (also from the per-quarter table), "
        f"and BRK-B is measured from provider presence (coverage={ratios['BRK-B']}). Validated."
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
                "valid_to": [None] * 5,
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

    summary, ratios, valid_dates = _coverage(
        cast(Context, SimpleNamespace(store=sqlite_store, config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[]))))
    )

    assert summary["tickers_measured"] == 1
    assert ratios["NEW"] == 1.0
    assert sorted(str(date.date()) for date in valid_dates["NEW"]) == ["2023-11-15", "2024-02-15"]
    print("\n=== SANITY CHECK: dated lineage and dei rows ===")
    print(
        "  five lineage rows over two CIKs read as two (cik, entity) pairs; a dei NEW row under a foreign CIK is "
        "not read, so NEW keeps its entity and its OLD history (2 valid calls, coverage 1.0). Validated."
    )
