"""Independent known-truth model assessment checks; no source database or trained model needed."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.context import Context
from src.validate.checks.modelling import _live_path_checks, _reconcile_cohorts, check_modelling, independent_daily, independent_paths
from src.validate.result import Finding


def test_independent_rank_ic_skips_undefined_dates() -> None:
    dates = pd.to_datetime(["2022-01-03"] * 4 + ["2022-01-04"] * 4)
    rows = pd.DataFrame({"date": dates, "ticker": list("ABCD") * 2, "y": [4, 3, 2, 1] * 2, "lgbm": [1, 2, 3, 4, 5, 5, 5, 5]})
    got = independent_daily(rows, "lgbm")
    assert len(got) == 1 and got.iloc[0] == -1
    print("SANITY: independent SciPy rank IC gives -1 on a reverse ordering and excludes a constant-score date.")


def test_independent_return_path_requires_all_future_bars() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=5)
    returns = np.array([0.0, 0.1, 0.2, -0.1, 0.05])
    prices = pd.DataFrame({"date": calendar, "ticker": "A", "ret": returns, "close_total": 100 * np.cumprod(1 + returns)})
    prices.loc[0, "ret"] = np.nan  # Today's return is not one of the three future bars.
    first = independent_paths(prices, calendar, 3).iloc[0]
    assert first.valid and abs(first.realized_return - 0.188) < 1e-12
    prices.loc[2, "ret"] = np.nan
    masked = independent_paths(prices, calendar, 3).iloc[0]
    assert not masked.valid and pd.isna(masked.realized_return)
    prices.loc[2, "ret"] = 0.5
    assert not independent_paths(prices, calendar, 3).iloc[0].valid
    print("SANITY: independent prefix replay matches the hand-computed 18.8% and rejects an interior hole or inconsistent seam.")


def test_incomplete_assessment_abstains_without_store_access(tmp_path: Path) -> None:
    class NoRead:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(f"Unexpected store read {name}")

    context = cast(Context, SimpleNamespace(store=NoRead()))
    config = cast(DictConfig, OmegaConf.create({"model": {"ensemble": ["lgbm"]}}))
    result = check_modelling(context, out=tmp_path, config=config)
    assert result.status == "abstain" and result.scope["missing"]
    print("SANITY: incomplete modelling evidence abstains explicitly; it cannot produce a false pass or read the database.")


@pytest.fixture
def cohort_artifacts(tmp_path: Path) -> Path:
    output = tmp_path / "eval"
    output.mkdir()
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    scores = pd.DataFrame({"date": dates.repeat(3), "ticker": list("ABC") * 2, "lgbm": [1, 2, 3] * 2, "rf": [3, 2, 1] * 2})
    cohorts = pd.DataFrame({"date": dates.repeat(2), "ticker": ["C", "A"] * 2, "model": ["lgbm", "rf"] * 2, "weight": 1.0})
    outcomes = scores[["date", "ticker"]].assign(realized_return=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], valid=True)
    market = pd.DataFrame({"date": dates, "ticker": "equity_tr", "realized_return": [0.02, np.nan], "valid": [True, False]})
    matched = cohorts[["date", "model"]].assign(
        cohort_return=[0.3, 0.1, 0.6, 0.4], market_return=[0.02, 0.02, np.nan, np.nan], market_valid=[True, True, False, False]
    )
    matched["excess_return"] = matched.cohort_return - matched.market_return
    for name, frame in [
        ("scores", scores),
        ("cohorts", cohorts),
        ("return_windows", outcomes),
        ("market_windows", market),
        ("matched_returns", matched),
    ]:
        frame.to_parquet(output / f"{name}.parquet", index=False)
    return tmp_path


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("none", None),
        ("invalid_finite", None),
        ("market_and_excess", "Matched SPY returns or validity differ"),
        ("validity", "Matched SPY returns or validity differ"),
        ("invalid_unmasked", "Matched SPY returns or validity differ"),
        ("missing_date", "Matched SPY returns or validity differ"),
        ("duplicate_date", "Market window keys differ"),
        ("wrong_ticker", "Market window keys differ"),
    ],
)
def test_matched_market_returns_are_bound_to_validated_windows(cohort_artifacts: Path, mutation: str, expected: str | None) -> None:
    output = cohort_artifacts / "eval"
    matched = pd.read_parquet(output / "matched_returns.parquet")
    market = pd.read_parquet(output / "market_windows.parquet")
    if mutation == "market_and_excess":
        matched["market_return"] += 0.03
        matched["excess_return"] -= 0.03
    elif mutation == "validity":
        matched.loc[0, "market_valid"] = False
    elif mutation == "invalid_unmasked":
        matched.loc[2, "market_return"] = 0.04
        matched.loc[2, "excess_return"] = 0.56
    elif mutation == "invalid_finite":
        market.loc[1, "realized_return"] = 0.04
    elif mutation == "missing_date":
        market = market.iloc[:1]
    elif mutation == "duplicate_date":
        market = pd.concat([market, market.iloc[:1]], ignore_index=True)
    elif mutation == "wrong_ticker":
        market.loc[0, "ticker"] = "other"
    matched.to_parquet(output / "matched_returns.parquet", index=False)
    market.to_parquet(output / "market_windows.parquet", index=False)
    assert np.allclose(matched.cohort_return - matched.market_return, matched.excess_return, equal_nan=True)
    findings: list[Finding] = []
    assert _reconcile_cohorts(cohort_artifacts, ["lgbm", "rf"], findings) == 4
    if expected is None:
        assert not findings
    else:
        assert any(finding.observed == expected for finding in findings)
    print(f"SANITY: matched benchmark linkage handles {mutation}; arithmetic alone cannot certify the stored SPY comparison.")


@pytest.mark.parametrize("duplicate", [False, True])
def test_source_market_replay_reports_duplicate_keys(cohort_artifacts: Path, duplicate: bool) -> None:
    output = cohort_artifacts / "eval"
    calendar = pd.bdate_range("2022-01-03", periods=62)
    predictions = pd.DataFrame({"date": calendar[:2], "ticker": "A", "fold": 0, "label_end": calendar[60:], "y": 0.0})
    labels = predictions[["date", "ticker"]].assign(target_zscore_h60=0.0)
    prices = pd.DataFrame({"date": calendar, "ticker": "A", "close_total": 100.0, "ret": 0.0})
    macro = pd.DataFrame({"date": calendar, "ticker": "equity_tr", "close": 100.0})
    windows = independent_paths(prices, calendar, 60)
    market = windows.assign(ticker="equity_tr")
    if duplicate:
        market = pd.concat([market, market.iloc[:1]], ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    windows.to_parquet(output / "return_windows.parquet", index=False)
    market.to_parquet(output / "market_windows.parquet", index=False)
    (output / "result.json").write_text(json.dumps({"calendar_end": calendar[-1].isoformat()}), encoding="utf-8")
    (cohort_artifacts / "selection.json").write_text(
        json.dumps({"outcome_cutoff": "2021-12-31", "controls": {"assessment": {"evaluation_start": "2022-01-03"}}}), encoding="utf-8"
    )
    store = SimpleNamespace(distinct=Mock(return_value=calendar), iter_load=Mock(side_effect=[[labels], [prices]]), load=Mock(return_value=macro))
    context = cast(Context, SimpleNamespace(store=store))
    findings: list[Finding] = []
    assert _live_path_checks(context, cohort_artifacts, findings, 10) == 2
    assert [finding.observed for finding in findings] == (["Source SPY path replay differs"] if duplicate else [])
    print(
        f"SANITY: source replay completes with duplicate={duplicate}; malformed benchmark keys become a finding rather than an uncaught merge error."
    )
