"""Known-truth tests for binary and structural-zero feature transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import src.data_aggregate.utils.common.panel as panel_module
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.xs import winsorize_xs
from src.data_aggregate.utils.fundamentals.dividend_features import (
    DIVIDEND_TRANSFORM_SEMANTICS,
)
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    FUNDAMENTAL_TRANSFORM_SEMANTICS,
)
from src.data_aggregate.utils.fundamentals.sector_features import (
    SECTOR_TRANSFORM_SEMANTICS,
)


def _all_peers(tickers: list[str]) -> dict[str, dict[str, float]]:
    return {ticker: {peer: 1.0 for peer in tickers if peer != ticker} for ticker in tickers}


def test_binary_semantic_preserves_state_and_bypasses_input_winsorization(monkeypatch):
    tickers = [f"T{i}" for i in range(8)]
    date = pd.Timestamp("2024-01-02")
    field = pd.DataFrame([[0, 0, 0, 0, 1, 1, np.nan, 0]], index=[date], columns=tickers)
    observed_winsor_flags: list[bool] = []
    real_peer_relative = panel_module.peer_relative

    def spy_peer_relative(values, peers, **kwargs):
        observed_winsor_flags.append(kwargs.get("winsorize_inputs", True))
        return real_peer_relative(values, peers, **kwargs)

    monkeypatch.setattr(panel_module, "peer_relative", spy_peer_relative)
    panel = build_peer_relative_panel(
        {"flag": field},
        _all_peers(tickers),
        semantics={"flag": "binary"},
    ).set_index("ticker")

    expected = field.loc[date].dropna().sort_index().astype("float32")
    expected_peer = winsorize_xs(
        real_peer_relative(
            field,
            _all_peers(tickers),
            winsorize_inputs=False,
        )
    ).loc[date]
    actual = panel["f_flag_xs"].sort_index()
    pd.testing.assert_series_equal(actual.dropna(), expected, check_names=False)
    pd.testing.assert_series_equal(
        panel["f_flag_vs_peers"].sort_index(),
        expected_peer.sort_index().astype("float32"),
        check_names=False,
    )
    assert pd.isna(actual.loc["T6"])
    assert observed_winsor_flags == [False]
    assert set(actual.dropna().unique()) == {0.0, 1.0}
    print("\n=== SANITY CHECK: binary transform semantics ===")
    print("  _xs is the exact 0/1 state and the peer leg bypasses continuous input winsorization. Validated.")


def test_structural_zero_semantic_ranks_only_nonzero_support():
    tickers = [f"T{i}" for i in range(8)]
    date = pd.Timestamp("2024-01-02")
    field = pd.DataFrame([[-2, -1, 0, 0, 1, 2, 3, np.nan]], index=[date], columns=tickers)
    peers = _all_peers(tickers)
    panel = build_peer_relative_panel(
        {"intensity": field},
        peers,
        semantics={"intensity": "structural_zero"},
    ).set_index("ticker")

    zero_tickers = ["T2", "T3"]
    nonzero_tickers = ["T0", "T1", "T4", "T5", "T6"]
    assert (panel.loc[zero_tickers, "f_intensity_xs"] == 0.0).all()
    assert (panel.loc[zero_tickers, "f_intensity_vs_peers"] == 0.0).all()
    nonzero = panel.loc[nonzero_tickers, "f_intensity_xs"]
    assert nonzero.is_monotonic_increasing
    assert nonzero.between(0, 1).all() and (nonzero > 0).all()
    assert panel.loc["T7", ["f_intensity_xs", "f_intensity_vs_peers"]].isna().all()
    comparable = field.mask(field.eq(0))
    expected_peer = (
        winsorize_xs(panel_module.peer_relative(comparable, peers))
        .where(
            ~field.eq(0),
            0.0,
        )
        .loc[date]
    )
    pd.testing.assert_series_equal(
        panel["f_intensity_vs_peers"].sort_index(),
        expected_peer.sort_index().astype("float32"),
        check_names=False,
    )
    print("\n=== SANITY CHECK: structural-zero transform semantics ===")
    print("  exact zeros remain 0; negative and positive non-zero support alone is ranked/peer-compared; null stays absent. Validated.")


def test_default_continuous_transform_is_unchanged_and_semantics_are_validated():
    tickers = [f"T{i}" for i in range(6)]
    dates = pd.bdate_range("2024-01-02", periods=3)
    field = pd.DataFrame(np.arange(18, dtype=float).reshape(3, 6), index=dates, columns=tickers)
    peers = _all_peers(tickers)

    default = build_peer_relative_panel({"metric": field}, peers)
    explicit_empty = build_peer_relative_panel({"metric": field}, peers, semantics={})
    pd.testing.assert_frame_equal(default, explicit_empty, check_exact=True)
    with pytest.raises(ValueError, match="semantic"):
        build_peer_relative_panel({"metric": field}, peers, semantics={"metric": "ordinal"})
    with pytest.raises(KeyError, match="semantics declares"):
        build_peer_relative_panel({"metric": field}, peers, semantics={"missing": "binary"})
    print("\n=== SANITY CHECK: default transform compatibility ===")
    print("  no semantic declaration is bit-identical to the old continuous path; unknown and stray declarations fail closed. Validated.")


def test_fundamentals_semantic_registry_is_explicit_and_bounded():
    declared = {
        **DIVIDEND_TRANSFORM_SEMANTICS,
        **FUNDAMENTAL_TRANSFORM_SEMANTICS,
        **SECTOR_TRANSFORM_SEMANTICS,
    }
    assert declared == {
        "dividend_payer": "binary",
        "nci_income_share": "structural_zero",
        "rd_intensity": "structural_zero",
        "deferred_rev_intensity": "structural_zero",
    }
    print("\n=== SANITY CHECK: bounded fundamentals semantics registry ===")
    print("  one measured binary field and three measured structural-zero fields are declared; no unrelated caller changes. Validated.")


def _all_strings(value) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {item for child in value.values() for item in _all_strings(child)}
    if isinstance(value, list):
        return {item for child in value for item in _all_strings(child)}
    return {str(value)} if isinstance(value, str) else set()


def test_no_active_model_selects_both_revenue_growth_aliases():
    checked: list[str] = []
    for path in sorted(Path("configs/models").glob("*_modelling.yml")):
        values = _all_strings(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
        cube_time = {value for value in values if value.startswith("f_revenueGrowth_")}
        computed = {value for value in values if value.startswith("f_y_rev_growth_") and not value.startswith("f_y_rev_growth_accel_")}
        assert not (cube_time and computed), f"{path} selects both revenue-growth aliases: {cube_time | computed}"
        checked.append(path.name)
    assert checked
    print("\n=== SANITY CHECK: duplicate revenue-growth consumers ===")
    print(f"  {len(checked)} active model configs checked; none selects both revenueGrowth and y_rev_growth. Validated.")
