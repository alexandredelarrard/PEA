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
    useful = "Revenue growth and margin guidance remained strong for customers this quarter. " * 12
    sections = pd.DataFrame(
        [
            {"ticker": ticker, "quarter": "2024Q1", "as_of": "2024-05-15", "tag": tag, "text": text}
            for ticker, text in (("AAA", useful), ("BBB", "Thanks."))
            for tag in ("prepared_remarks", "qa")
        ]
    )
    sqlite_store.save(Tables.earnings_call_sections, sections)

    dates = pd.bdate_range("2024-05-16", periods=120)
    cube = pd.DataFrame({"date": np.repeat(dates, 2), "ticker": ["AAA", "BBB"] * len(dates)})
    for position, name in enumerate(EARNINGS_CALL_FEATURES):
        cube[f"f_{name}"] = np.sin(np.arange(len(cube)) / (7.0 + position))
    cube["f_ec_uncertainty"] = cube["f_ec_uncertainty"].abs().clip(upper=1)
    cube["f_ec_qa_coherence_mean"] = cube["f_ec_qa_coherence_mean"].clip(-1, 1)
    cube["f_ec_qa_qq_distance"] = cube["f_ec_qa_qq_distance"].abs().clip(upper=2)
    cube["f_ec_prep_qq_distance"] = cube["f_ec_prep_qq_distance"].abs().clip(upper=2)
    sqlite_store.save(Tables.cube_part_text, cube)

    context = cast(Context, SimpleNamespace(store=sqlite_store))
    config = OmegaConf.merge(OmegaConf.load("configs/validate.yml"), OmegaConf.create({"train": {"end_date": "2022-01-01"}}))
    result = check_earnings_calls(context, Tables.cube_part_text, config=config)
    coverage = cast(dict, result.metrics["coverage"])
    assert result.status == "pass"
    assert coverage["coverage_100pct"] == 1
    assert coverage["coverage_lt50pct"] == 1
    assert coverage["malformed_calls"] == 1
    assert result.scope["feature_columns"] == 12
    print("\n=== SANITY CHECK: dedicated earnings-call validator ===")
    print("  exact 12-column schema measured; one valid call is 100% covered and one malformed call remains <50%. Validated.")


def test_coverage_uses_point_in_time_lineage_and_separates_no_call_names(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["NEW", "BRK-B"]}))
    sqlite_store.save(
        Tables.entity_lineage,
        pd.DataFrame({"cik": ["1", "2"], "entity_id": ["E1", "E2"]}),
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
    useful = "Revenue growth demand margin guidance cash flow customer outlook remained strong. " * 12
    sqlite_store.save(
        Tables.earnings_call_sections,
        pd.DataFrame(
            [
                {"ticker": ticker, "quarter": quarter, "as_of": date, "tag": tag, "text": useful}
                for ticker, date, quarter in events
                for tag in ("prepared_remarks", "qa")
            ]
        ),
    )

    summary, ratios, valid_dates = _coverage(cast(Context, SimpleNamespace(store=sqlite_store)))

    assert summary["roster_tickers"] == 2
    assert summary["structural_no_call_tickers"] == ["BRK-B"]
    assert summary["tickers_measured"] == 1
    assert ratios["NEW"] == 1.0
    assert len(valid_dates["NEW"]) == 2
    print("\n=== SANITY CHECK: point-in-time coverage identity ===")
    print("  OLD history follows E1 into NEW, reused OLD/E2 events are excluded, and BRK-B is structural—not a source failure. Validated.")
