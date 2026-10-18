"""Known-truth assessment arithmetic: fixed populations, pruning and date-balanced samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.modelling.utils.assessment import (
    assess_candidate,
    audit_features,
    balanced_sample_positions,
    frame_digest,
    json_value,
    prune_pool,
    set_horizon_features,
    summarize_ic,
)


def test_pooled_ir_uses_all_dates_instead_of_mean_fold_ir() -> None:
    values = pd.Series([0.1, 0.2, -0.2, 0.1, 0.3])
    result = summarize_ic(values, 60)
    expected = values.mean() / values.to_numpy().std() * np.sqrt(252 / 60)
    assert result["ic_ir"] == pytest.approx(expected)
    assert result["n_days"] == 5
    print("SANITY: pooled IC-IR uses the full daily series with population dispersion and h60 adjustment.")


def _population() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": np.repeat(pd.bdate_range("2000-01-03", periods=3), 6), "ticker": list("ABCDEF") * 3, "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 3}
    )


def test_a_candidate_cannot_win_by_losing_keys_or_constant_score_dates() -> None:
    expected = _population()
    predicted = expected[["date", "ticker"]].copy()
    predicted["pred"] = [0.0, 2.0, 1.0, 4.0, 3.0, 5.0, 2.0, 0.0, 1.0, 3.0, 5.0, 4.0, 4.0, 3.0, 5.0, 1.0, 0.0, 2.0]
    baseline, series = assess_candidate(expected, predicted, 60)
    assert baseline["eligible"] and len(series) == 3
    missing, _ = assess_candidate(expected, predicted.iloc[6:], 60)
    assert not missing["eligible"] and missing["missing_prediction_rows"] == 6
    predicted.loc[predicted["date"] == expected["date"].iloc[0], "pred"] = 0.0
    constant, _ = assess_candidate(expected, predicted, 60)
    assert not constant["eligible"] and constant["undefined_dates"] == 1
    print("SANITY: every candidate uses the same expected keys/dates; missing keys or a newly undefined date make it ineligible.")


def test_pruning_is_zero_first_then_bottom_twenty_percent_and_never_empty() -> None:
    features = list("abcdef")
    kept, removed, reason = prune_pool(features, dict(zip(features, [3.0, 2.0, 0.0, 1.0, 0.0, 4.0], strict=True)), 0.2)
    assert kept == list("abdf") and removed == ["c", "e"] and reason == "zero_importance"
    kept, removed, reason = prune_pool(kept, {"a": 3.0, "b": 2.0, "d": 1.0, "f": 4.0}, 0.2)
    assert kept == list("abf") and removed == ["d"] and reason == "bottom_fraction"
    assert len(prune_pool(["a", "b"], {"a": 0.0, "b": 0.0}, 0.2)[0]) == 1
    assert prune_pool(["a"], {"a": 0.0}, 0.2)[1] == []
    with pytest.raises(ValueError, match="importance"):
        prune_pool(features, {"a": 1.0}, 0.2)
    print("SANITY: zero removal precedes ceil(20%) pruning; ties are deterministic and at least one input survives.")


def test_date_balanced_sample_is_reproducible_and_feature_independent() -> None:
    panel = pd.DataFrame({"date": np.repeat(pd.bdate_range("2000-01-03", periods=5), [10, 20, 40, 70, 100])})
    positions = balanced_sample_positions(panel, 17, 4325)
    counts = panel.iloc[positions].groupby("date").size()
    assert len(positions) == 17 and counts.max() - counts.min() <= 1
    assert np.array_equal(positions, balanced_sample_positions(panel.assign(unused=np.nan), 17, 4325))
    assert len(balanced_sample_positions(panel, 0, 4325)) == len(panel)
    print("SANITY: the same 17 row keys are sampled across candidates, balanced by date despite changing breadth.")


def test_streamed_audit_distinguishes_sparse_support_bad_values_and_dead_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("1999-01-01", periods=6),
            "ticker": ["AAA"] * 6,
            "sector": [10, 10, 20, 20, 20, 10],
            "valid": ["1.25", None, "2.5", None, "3.0", None],
            "bad": ["oops", "1", None, None, None, None],
            "dead": [None] * 6,
            "constant": [2.0] * 6,
            "overflow": [1e100] + [1.0] * 5,
        }
    )
    features = ["sector", "valid", "bad", "dead", "constant", "overflow"]
    result, sample = audit_features(
        [frame.iloc[:3], frame.iloc[3:]], features, list(frame.columns), ["sector"], {"date": "key", "ticker": "key"}, 3, 4325
    )
    ledger = {row["feature"]: row for row in result["features"]}
    assert ledger["valid"]["role"] == "numeric" and ledger["valid"]["observed"] == 3
    assert ledger["bad"]["reason"] == "unresolved_numeric_text"
    assert ledger["dead"]["reason"] == "all_null_development"
    assert ledger["constant"]["reason"] == "constant_development"
    assert ledger["overflow"]["reason"] == "nonfinite_model_input"
    assert result["rows"] == 6 and len(sample) == 3
    valid_coverage = [row for row in result["coverage"] if row["feature"] == "valid"]
    assert sum(row["observed"] for row in valid_coverage) == 3 and sum(row["rows"] for row in valid_coverage) == 6
    assert json_value({"bad": np.nan, "infinite": np.inf, "number": np.int64(3)}) == {"bad": None, "infinite": None, "number": 3}
    print("SANITY: sparse features remain eligible; malformed, nonfinite, dead and constant inputs have distinct measured exclusions.")


def test_input_identity_is_row_order_independent_and_value_sensitive() -> None:
    panel = _population()
    panel["a"] = np.arange(len(panel), dtype="float32")
    before = frame_digest(panel, ["a", "y"])
    assert frame_digest(panel.iloc[::-1], ["y", "a"]) == before
    panel.loc[0, "a"] = 1.0
    assert frame_digest(panel, ["a", "y"]) != before
    print("SANITY: normalized keyed input identity ignores row/column order and rejects changed values.")


@pytest.mark.parametrize("old_override", ["", "  columns_by_horizon:\n    30: [keep]\n", "  columns_by_horizon:\n    30: [keep]\n    60: []\n"])
def test_h60_yaml_edit_preserves_other_horizons_parameters_and_comments(old_override: str) -> None:
    original = "# preserve comment\nlgbm:\n  learning_rate: 0.01\n" + old_override + "  monotonic:\n    enabled: true\n    features: {old: 1}\n"
    updated = set_horizon_features(original, "lgbm", "columns_by_horizon", ["beta_USD/EUR 1", 'quoted"feature'])
    updated = set_horizon_features(updated, "lgbm", "categoricals_by_horizon", [])
    before, after = yaml.safe_load(original)["lgbm"], yaml.safe_load(updated)["lgbm"]
    assert after["learning_rate"] == before["learning_rate"] and after["monotonic"] == before["monotonic"]
    assert after["columns_by_horizon"][60] == ["beta_USD/EUR 1", 'quoted"feature']
    assert after["categoricals_by_horizon"][60] == []
    if old_override:
        assert after["columns_by_horizon"][30] == ["keep"]
    assert "# preserve comment" in updated
    print("SANITY: only h60 feature overrides change; h30, hyperparameters, monotonic map and unrelated comments survive.")
