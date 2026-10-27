"""Actual-builder controls for PIT units, publication and historical replacement."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.utils.price.features import pit_fields
from src.data_aggregate.utils.price.frames import load_price_frames
from src.data_aggregate.utils.price.inputs import PriceInputs
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries
from src.utils.price.adjustments import ShareBasis


def seed_prices(store: Any, directory: Path) -> SimpleNamespace:
    """A known four-for-one split with unchanged economic capitalization."""
    index = pd.bdate_range("2020-01-02", periods=760)
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAA", "BBB"]}))
    df_bars = pd.DataFrame(
        [
            {
                "date": day,
                "ticker": ticker,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close_split": price,
                "close_total": price,
                "volume": 1000.0,
            }
            for day in index
            for ticker, price in (("AAA", 25.0), ("BBB", 10.0))
        ]
    )
    store.save(Tables.prices, df_bars)
    store.save(Tables.prices_macro, pd.DataFrame({"ticker": MACRO_MARKET_SERIES, "date": index, "close": 100.0}))
    store.save(Tables.prices_splits, pd.DataFrame({"ticker": ["AAA"], "date": [index[5]], "ratio": [4.0]}))
    store.save(
        Tables.sharadar_actions, pd.DataFrame({"ticker": ["AAA"], "date": [index[5]], "action": ["split"], "value": [4.0], "contraticker": [""]})
    )
    store.save(
        Tables.fundamentals_history,
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "as_of": [index[0]] * 2,
                "fiscal_end": pd.to_datetime(["2019-09-30"] * 2),
                "sharesOutstanding": [400.0, 300.0],
                "sharesOutstandingPit": [100.0, 300.0],
            }
        ),
    )
    store.save(
        Tables.sharadar_fundamentals,
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "date": [index[0]] * 2,
                "dimension": ["ARQ"] * 2,
                "reportperiod": pd.to_datetime(["2019-09-30"] * 2),
                "sharesbas": [400.0, 300.0],
                "sharefactor": [1.0, 1.0],
                "price": [25.0, 10.0],
                "lastupdated": [index[6]] * 2,
            }
        ),
    )
    peer_path = directory / "peers.json"
    peer_path.write_text(json.dumps({"AAA": {"BBB": 1.0}, "BBB": {"AAA": 1.0}}), encoding="utf-8")
    config = OmegaConf.create({"build_cube": {}, "data_extract": {"redundant_ticks": []}})
    return SimpleNamespace(store=store, config=config, config_dir=directory, log=logging.getLogger(__name__), paths={"SECTOR_PEERS_PATH": peer_path})


def test_actual_builder_publishes_quoted_prices_and_split_carried_counts(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    StepCubePrices(context, context.config).run(full=True)
    df_prices = sqlite_store.load(Tables.cube_part_prices)
    assert {"close_pit", "shares_outstanding_pit", "mcap_pit"}.issubset(df_prices)
    df_a = df_prices.loc[df_prices["ticker"].eq("AAA")].sort_values("date")
    assert df_a["close_pit"].iloc[:5].eq(100.0).all()
    assert df_a["close_pit"].iloc[5:].eq(25.0).all()
    assert df_a["shares_outstanding_pit"].iloc[:5].eq(100.0).all()
    assert df_a["shares_outstanding_pit"].iloc[5:].eq(400.0).all()
    assert df_a["mcap_pit"].eq(10_000.0).all()
    assert df_a["ret"].dropna().eq(0).all()
    fields = ("close_pit", "shares_outstanding_pit", "mcap_pit")
    frames = load_price_frames(sqlite_store, peers={}, fields=fields)
    assert frames.close_split is None
    for field in fields:
        assert getattr(frames, field).dtypes.eq("float64").all()
    print(
        "SANITY: actual price build stores quoted100→25, public count100→400, constant$10,000cap, adjusted zero returns and projected float64 PIT fields."
    )


def test_historical_prefix_invalidation_reaches_before_old_warmup(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    df_raw = sqlite_store.load(Tables.prices)
    day = sorted(pd.to_datetime(df_raw["date"]).unique())[20]
    changed = pd.to_datetime(df_raw["date"]).eq(day) & df_raw["ticker"].eq("AAA")
    df_raw.loc[changed, ["close_split", "close_total"]] = 51.0
    sqlite_store.replace(Tables.prices, df_raw)
    step.run(full=False)
    df_incremental = sqlite_store.load(Tables.cube_part_prices)
    actual = df_incremental.loc[pd.to_datetime(df_incremental["date"]).eq(day) & df_incremental["ticker"].eq("AAA"), "close_split"].iloc[0]
    assert actual == 51.0, "A changed historical prefix must be replaced even before the old520-session warmup"
    step.run(full=True)
    df_full = sqlite_store.load(Tables.cube_part_prices)
    pd.testing.assert_frame_equal(df_incremental, df_full, check_exact=True)
    print("SANITY: a source correction740sessions before the frontier is rewritten; actual incremental and full outputs match exactly.")


def test_builder_reads_sources_once_and_promotes_old_schema_before_payload(
    sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    df_existing = sqlite_store.load(Tables.cube_part_prices)
    legacy = df_existing.drop(columns=[field for field in ("close_pit", "shares_outstanding_pit", "mcap_pit") if field in df_existing])
    sqlite_store.replace(Tables.cube_part_prices, legacy)
    calls: list[tuple[Any, dict]] = []
    original = sqlite_store.load

    def capture(table: Any, **kwargs: Any) -> Any:
        calls.append((table, kwargs))
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "load", capture)
    step.run(full=False)
    price_calls = [kwargs for table, kwargs in calls if table == Tables.prices]
    assert len(price_calls) == 1
    assert price_calls[0].get("since") is None
    for table in {item[0] for item in calls}:
        assert sum(item[0] == table for item in calls) == 1, table
    assert {"close_pit", "shares_outstanding_pit", "mcap_pit"}.issubset(sqlite_store.columns(Tables.cube_part_prices))
    print(
        "SANITY: old schema promotes before the large read; each source table is read once and the completed three-field candidate is published together."
    )


def test_invalid_candidate_never_reaches_the_writer(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    df_raw = sqlite_store.load(Tables.prices)
    df_raw.loc[df_raw.index[0], "close_total"] = -1.0
    sqlite_store.replace(Tables.prices, df_raw)
    writes: list[Any] = []
    monkeypatch.setattr(sqlite_store, "replace", lambda *args, **kwargs: writes.append(args))
    with pytest.raises(ValueError, match="positive|invalid"):
        StepCubePrices(context, context.config).run(full=True)
    assert not writes
    print("SANITY: an invalid total-return input is refused before any price-part write.")


def test_zero_open_is_null_only_in_the_candidate(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    df_expected = sqlite_store.load(Tables.cube_part_prices)
    df_source = sqlite_store.load(Tables.prices)
    row = df_source.index[0]
    day, ticker = df_source.loc[row, ["date", "ticker"]]
    df_source.loc[row, "open"] = 0.0
    sqlite_store.replace(Tables.prices, df_source)
    step.run(full=True)
    df_actual = sqlite_store.load(Tables.cube_part_prices)
    selected = pd.to_datetime(df_expected["date"]).eq(pd.Timestamp(day)) & df_expected["ticker"].eq(ticker)
    df_expected.loc[selected, "open"] = float("nan")
    pd.testing.assert_frame_equal(df_actual, df_expected, check_exact=True)
    pd.testing.assert_frame_equal(sqlite_store.load(Tables.prices), df_source, check_exact=True)
    print("SANITY: a zero opening quote becomes one candidate null; every other output and every raw source cell is unchanged.")


def test_missing_yahoo_price_factor_is_not_assumed_already_correct(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    sqlite_store.delete(Tables.prices_splits, where={"ticker": "AAA"})
    StepCubePrices(context, context.config).run(full=True)
    df_prices = sqlite_store.load(Tables.cube_part_prices)
    df_a = df_prices.loc[df_prices["ticker"].eq("AAA")].sort_values("date")
    assert df_a["close_pit"].iloc[:5].isna().all()
    assert df_a["mcap_pit"].iloc[:5].isna().all()
    assert df_a["close_pit"].iloc[5:].eq(25.0).all()
    assert df_a["shares_outstanding_pit"].iloc[:5].eq(100.0).all()
    assert df_prices.loc[df_prices["ticker"].eq("BBB"), "close_pit"].eq(10.0).all()
    print(
        "SANITY: a Sharadar-only restatement cannot certify an ambiguous pre-event Yahoo quote; count evidence and unaffected securities remain available."
    )


def test_removed_history_clears_an_empty_replacement_tail_and_retries_exactly(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    dates = sqlite_store.distinct(Tables.prices, "date")
    removed = sorted(dates)[-20:]
    sqlite_store.delete(Tables.prices, where={"date": removed})
    step.run(full=False)
    df_first = sqlite_store.load(Tables.cube_part_prices)
    assert len(df_first) == 740 * 2
    assert pd.to_datetime(df_first["date"]).max() < pd.Timestamp(removed[0])
    step.run(full=False)
    pd.testing.assert_frame_equal(df_first, sqlite_store.load(Tables.cube_part_prices), check_exact=True)
    step.run(full=True)
    pd.testing.assert_frame_equal(df_first, sqlite_store.load(Tables.cube_part_prices), check_exact=True)
    print("SANITY: removed saved keys are deleted even with no replacement rows after the cutoff; repeat/full builds remain exactly equal.")


def test_existing_empty_price_schema_is_a_full_cold_build(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    df_before = sqlite_store.load(Tables.cube_part_prices)
    sqlite_store.delete(Tables.cube_part_prices, where={"date": sqlite_store.NOT_NULL})
    step.run(full=False)
    pd.testing.assert_frame_equal(df_before, sqlite_store.load(Tables.cube_part_prices), check_exact=True)
    print("SANITY: an existing empty price schema promotes to a complete candidate without a late reload.")


def test_interrupted_nonatomic_tail_write_recovers_on_retry(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    df_before = sqlite_store.load(Tables.cube_part_prices)
    original = sqlite_store.append_tail

    def interrupted(table: Any, frame: pd.DataFrame, cutoff: Any, **kwargs: Any) -> int:
        original(table, frame.iloc[:0], cutoff, **kwargs)
        raise RuntimeError("interrupted after tail delete")

    with monkeypatch.context() as patch:
        patch.setattr(sqlite_store, "append_tail", interrupted)
        with pytest.raises(RuntimeError, match="interrupted after tail delete"):
            step.run(full=False)
    assert sqlite_store.row_count(Tables.cube_part_prices) < len(df_before)
    step.run(full=False)
    pd.testing.assert_frame_equal(df_before, sqlite_store.load(Tables.cube_part_prices), check_exact=True)
    print("SANITY: the existing writer is non-atomic; an actual delete/insert interruption is fully repaired by a retry from unchanged sources.")


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan")])
def test_invalid_yahoo_adjustment_evidence_refuses_publication(
    sqlite_store: Any, tmp_path: Path, ratio: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    df_splits = sqlite_store.load(Tables.prices_splits)
    df_splits["ratio"] = ratio
    sqlite_store.replace(Tables.prices_splits, df_splits)
    writes: list[Any] = []
    monkeypatch.setattr(sqlite_store, "replace", lambda *args, **kwargs: writes.append(args))
    with pytest.raises(ValueError, match="Invalid Yahoo"):
        StepCubePrices(context, context.config).run(full=True)
    assert not writes
    print("SANITY: malformed Yahoo adjustment ratios cannot silently become identity factors or reach the writer.")


def test_new_public_share_observation_preserves_the_earlier_prefix(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    step = StepCubePrices(context, context.config)
    step.run(full=True)
    df_before = sqlite_store.load(Tables.cube_part_prices)
    public = pd.Timestamp(sorted(sqlite_store.distinct(Tables.prices, "date"))[400])
    fiscal = pd.Timestamp("2020-12-31")
    sqlite_store.save(
        Tables.fundamentals_history,
        pd.DataFrame({"ticker": ["AAA"], "as_of": [public], "fiscal_end": [fiscal], "sharesOutstanding": [440.0], "sharesOutstandingPit": [440.0]}),
    )
    sqlite_store.save(
        Tables.sharadar_fundamentals,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": [public],
                "dimension": ["ARQ"],
                "reportperiod": [fiscal],
                "sharesbas": [440.0],
                "sharefactor": [1.0],
                "price": [25.0],
                "lastupdated": [public],
            }
        ),
    )
    step.run(full=False)
    df_after = sqlite_store.load(Tables.cube_part_prices)
    prior = pd.to_datetime(df_before["date"]).lt(public)
    pd.testing.assert_frame_equal(df_before.loc[prior].reset_index(drop=True), df_after.loc[prior].reset_index(drop=True), check_exact=True)
    later = df_after["ticker"].eq("AAA") & pd.to_datetime(df_after["date"]).ge(public)
    assert df_after.loc[later, "shares_outstanding_pit"].eq(440.0).all()
    assert df_after.loc[later, "mcap_pit"].eq(11_000.0).all()
    print(
        "SANITY: a newly public count changes only its own public-date suffix, despite historical recomputation; issuer cap follows the same eligible observation."
    )


def test_count_restoration_uses_matched_vendor_vintage_not_stale_cache(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    df_history = sqlite_store.load(Tables.fundamentals_history)
    df_history.loc[df_history["ticker"].eq("AAA"), "sharesOutstandingPit"] = 50.0
    sqlite_store.replace(Tables.fundamentals_history, df_history)
    StepCubePrices(context, context.config).run(full=True)
    df_a = sqlite_store.load(Tables.cube_part_prices, where={"ticker": "AAA"}).sort_values("date")
    assert df_a["shares_outstanding_pit"].iloc[:5].eq(100.0).all()
    assert df_a["shares_outstanding_pit"].iloc[5:].eq(400.0).all()
    assert df_a["mcap_pit"].eq(10_000.0).all()
    assert sqlite_store.load(Tables.fundamentals_history, where={"ticker": "AAA"})["sharesOutstandingPit"].eq(50.0).all()
    print("SANITY: the new count restores the corroborated vendor vintage, rather than trusting a stale cachedPIT value; the source is unchanged.")


def test_vendor_catch_up_preserves_original_counts_and_quoted_cap(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    df_history = sqlite_store.load(Tables.fundamentals_history)
    df_vendor = sqlite_store.load(Tables.sharadar_fundamentals)
    selected = df_history["ticker"].eq("AAA")
    df_history.loc[selected, "sharesOutstanding"] = 100.0
    df_history.loc[selected, "sharesOutstandingPit"] = 25.0
    df_vendor.loc[df_vendor["ticker"].eq("AAA"), "sharesbas"] = 100.0
    df_vendor.loc[df_vendor["ticker"].eq("AAA"), "price"] = 100.0
    df_vendor.loc[df_vendor["ticker"].eq("AAA"), "lastupdated"] = df_history.loc[selected, "as_of"].iloc[0]
    sqlite_store.replace(Tables.fundamentals_history, df_history)
    sqlite_store.replace(Tables.sharadar_fundamentals, df_vendor)
    StepCubePrices(context, context.config).run(full=True)
    df_before = sqlite_store.load(Tables.cube_part_prices)
    df_a = df_before.loc[df_before["ticker"].eq("AAA")].sort_values("date")
    assert df_a["shares_outstanding_pit"].iloc[:5].eq(100.0).all()
    assert df_a["shares_outstanding_pit"].iloc[5:].eq(400.0).all()
    assert df_a["mcap_pit"].eq(10_000.0).all()
    pd.testing.assert_frame_equal(sqlite_store.load(Tables.fundamentals_history), df_history, check_exact=True)
    df_history.loc[selected, "sharesOutstanding"] = 400.0
    df_vendor.loc[df_vendor["ticker"].eq("AAA"), "sharesbas"] = 400.0
    df_vendor.loc[df_vendor["ticker"].eq("AAA"), "price"] = 25.0
    df_vendor.loc[df_vendor["ticker"].eq("AAA"), "lastupdated"] = df_a["date"].iloc[6]
    sqlite_store.replace(Tables.fundamentals_history, df_history)
    sqlite_store.replace(Tables.sharadar_fundamentals, df_vendor)
    StepCubePrices(context, context.config).run(full=True)
    pd.testing.assert_frame_equal(sqlite_store.load(Tables.cube_part_prices), df_before, check_exact=True)
    print("SANITY: a vendor catch-up changes its representation, preserving every historical quote/count/cap; the stale cache is never rewritten.")


def test_pit_fields_refuse_an_unproven_predecessor_count_owner() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    df_close = pd.DataFrame({"AAA": [10.0] * 3}, index=index)
    df_events = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = ShareBasis(df_events, df_events, df_events, {})
    df_history = pd.DataFrame(
        {
            "ticker": ["AAA"] * 2,
            "as_of": index[:2],
            "fiscal_end": index[:2],
            "sharesOutstanding": [100.0, 200.0],
            "sharesOutstandingPit": [100.0, 200.0],
        }
    )
    df_vendor = df_history.rename(columns={"as_of": "date", "fiscal_end": "reportperiod", "sharesOutstanding": "sharesbas"}).assign(
        lastupdated=index[:2], sharefactor=1.0
    )
    inputs = PriceInputs(
        bars=pd.DataFrame(),
        market=pd.Series(100.0, index=index),
        peers={},
        vendor=df_vendor,
        shares=df_history,
        actions=None,
        splits=df_events.rename(columns={"value": "ratio"}),
        predecessors=(),
        unsupported_units=frozenset(),
    )
    expected = pit_fields(df_close, df_close * 0 + 1, basis, inputs, {}, None)
    inputs = replace(inputs, predecessors=(PredecessorSeries("AAA", "OLD", "0000000001", None, index[1]),))
    actual = pit_fields(df_close, df_close * 0 + 1, basis, inputs, {}, None)
    pd.testing.assert_frame_equal(actual["close_pit"], expected["close_pit"], check_exact=True)
    assert pd.isna(actual["shares_outstanding_pit"].iloc[0, 0])
    assert pd.isna(actual["mcap_pit"].iloc[0, 0])
    assert actual["shares_outstanding_pit"].iloc[1:, 0].eq(200.0).all()
    assert actual["mcap_pit"].iloc[1:, 0].eq(2000.0).all()
    print(
        "SANITY: canonical ticker/date equality cannot certify a different predecessor owner; quotes and the effective-day successor remain available."
    )
