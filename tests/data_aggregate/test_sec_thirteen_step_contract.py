"""Independent broad and elite 13F merge ownership."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import SecThirteenResources, StepCubeSecThirteen
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.incremental import PartWindow
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.inputs import iter_filing_vintages
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import EMISSION, SuperinvestorFeatureHistory
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.institutional_vintages import prepare_book_vintages
from src.utils.price_adjustments import resolve_share_events
from tests.data_aggregate.test_institutional_step_contract import context, seed_prices
from tests.data_aggregate.test_superinvestor_books import _packet


def _manager_batch_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, packets: list[tuple[pd.DataFrame, pd.DataFrame]]
) -> tuple[StepCubeSecThirteen, pd.DatetimeIndex, list[bool], Mock]:
    """Actual public owner with only resource/source/persistence boundaries supplied."""
    store = Mock(spec=DataStore)
    store.columns.return_value = []
    store.distinct.return_value = []
    ctx = context(store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    grid = pd.bdate_range("2024-01-02", "2024-12-31")
    df_close = pd.DataFrame(100.0, index=grid, columns=["AAA"])
    frames = PriceFrames(trading_index=grid, universe=("AAA",), peers={"AAA": {"AAA": 1.0}}, close_split=df_close, level_factor=df_close / 100.0)
    ciks = frozenset(cik for filings, holdings in packets for df_rows in (filings, holdings) for cik in df_rows.cik)
    resources = SecThirteenResources(frames, None, resolve_share_events(None, None, {}), None, None, {"": ciks})
    closed: list[bool] = []

    def source(*args: Any, **kwargs: Any) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
        try:
            yield from packets
        finally:
            closed.append(True)

    monkeypatch.setattr(step, "_load_resources", lambda: resources)
    monkeypatch.setattr(step, "_institutional_features", lambda resources: None)
    monkeypatch.setattr(step_module.thirteen_inputs, "iter_filing_vintages", source)
    monkeypatch.setattr(step_module, "load_trading_calendar", lambda store: grid)
    monkeypatch.setattr(step_module, "plan_window", lambda *args, **kwargs: PartWindow(None, None))
    save = Mock()
    monkeypatch.setattr(step_module, "write_part", save)
    return step, grid, closed, save


@pytest.mark.parametrize("first_state", ["public", "unpublished", "failed", "orphan"])
@pytest.mark.parametrize("same_manager", [False, True], ids=["different-manager", "different-quarter"])
def test_manager_duplicate_accession_across_batches_refuses_before_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first_state: str, same_manager: bool
) -> None:
    first = _packet("0000000001", "2024-03-31", "2024-12-31" if first_state == "unpublished" else "2024-05-15", {"111111111": 10})
    second = _packet("0000000001" if same_manager else "0000000002", "2024-06-30", "2024-08-14", {"222222222": 20})
    for df_rows in (*first, *second):
        df_rows["accession_number"] = "same-accession"
    if first_state == "failed":
        first[0]["parse_status"] = "failed"
    elif first_state == "orphan":
        first[0]["amendment_type"] = "additions"
    packets = [first, second]
    step, grid, closed, save = _manager_batch_owner(monkeypatch, tmp_path, packets)
    message = "^13F manifest accessions must be unique$"
    with pytest.raises(ValueError, match=message):
        prepare_book_vintages(pd.concat([first[0], second[0]]), pd.concat([first[1], second[1]]), grid, settle_days=0)
    with pytest.raises(ValueError, match=message):
        step.run(full=True)
    assert closed == [True] and save.call_count == 0
    print(f"SANITY: {first_state=} {same_manager=}; duplicate accession keeps the old exact refusal, closes upstream and saves nothing.")


@pytest.mark.parametrize("orphan_first", [False, True])
def test_manager_duplicate_holding_vintage_across_quarters_refuses_before_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, orphan_first: bool
) -> None:
    first = _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 10})
    second = _packet("0000000001", "2024-06-30", "2024-08-14", {"111111111": 20})
    for df_rows in (*first, *second):
        df_rows["accession_number"] = "same-accession"
    packets = [(first[0].iloc[:0] if orphan_first else first[0], first[1]), (second[0].iloc[:0], second[1])]
    step, grid, closed, save = _manager_batch_owner(monkeypatch, tmp_path, packets)
    message = "^13F holding vintages must be unique$"
    with pytest.raises(ValueError, match=message):
        prepare_book_vintages(pd.concat([packet[0] for packet in packets]), pd.concat([packet[1] for packet in packets]), grid, settle_days=0)
    with pytest.raises(ValueError, match=message):
        step.run(full=True)
    assert closed == [True] and save.call_count == 0
    print(f"SANITY: {orphan_first=} repeated CIK/accession/CUSIP across quarters keeps the exact holding-key refusal before any save.")


def test_manager_disjoint_orphan_holding_keys_remain_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 10})
    second = _packet("0000000001", "2024-06-30", "2024-08-14", {"222222222": 20})
    for df_rows in (*first, *second):
        df_rows["accession_number"] = "same-accession"
    packets = [first, (second[0].iloc[:0], second[1])]
    before = [tuple(df_rows.copy(deep=True) for df_rows in packet) for packet in packets]
    step, grid, closed, save = _manager_batch_owner(monkeypatch, tmp_path, packets)
    prepare_book_vintages(pd.concat([packet[0] for packet in packets]), pd.concat([packet[1] for packet in packets]), grid, settle_days=0)
    step.run(full=True)
    assert closed == [True] and save.call_count == 1
    for old, actual in zip(before, packets, strict=True):
        for expected, df_rows in zip(old, actual, strict=True):
            pd.testing.assert_frame_equal(expected, df_rows)
    print("SANITY: distinct orphan CUSIPs retain the old acceptance and one checked save; accession-only holding rejection would be too broad.")


def test_manager_validation_releases_prior_manager_holding_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    packets = [
        _packet("0000000001", "2024-03-31", "2024-05-15", {"111111111": 10, "222222222": 20}),
        _packet("0000000002", "2024-03-31", "2024-05-15", {"111111111": 30, "222222222": 40, "333333333": 50}),
        _packet("0000000003", "2024-03-31", "2024-05-15", {}),
    ]
    step, _, closed, save = _manager_batch_owner(monkeypatch, tmp_path, packets)
    with caplog.at_level("INFO", logger="src.data_aggregate.utils.sec_thirteen.superinvestor_books"):
        step.run(full=True)
    assert closed == [True] and save.call_count == 1
    assert "peak trailing-manager holding keys=3" in caplog.text
    print("SANITY: unique books and complete empty markers save once; validation retains at most3keys for one manager rather than5across managers.")


def test_sec_thirteen_executes_broad_and_elite_once(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    broad = Mock(
        return_value=pd.DataFrame({"period": [pd.Timestamp("2023-09-30")], "as_of": [days[0]], "ticker": ["AAA"], "ic_inst_concentration": [0.4]})
    )
    levels = {name: pd.DataFrame({"AAA": [float("nan")]}, index=days[:1]) for name in EMISSION}
    levels["ic_super_holders"].iloc[0, 0] = 1.0
    elite = Mock(return_value=SuperinvestorFeatureHistory(levels, pd.DataFrame()))
    monkeypatch.setattr(step, "_institutional_features", broad)
    monkeypatch.setattr(step, "_superinvestor_history", elite)
    result = step.build_panel()
    assert step._store is sqlite_store and step._part.table == Tables.cube_part_sec_thirteen
    assert broad.call_count == elite.call_count == 1
    assert set(result.columns) == {"date", "ticker", "f_ic_inst_concentration"} | {"f_" + name for name in EMISSION}
    assert result.f_ic_inst_concentration.dropna().eq(pd.Series([0.4], dtype="float32").iloc[0]).all()
    assert result.f_ic_super_holders.dropna().eq(1.0).all()
    assert not result.duplicated(["date", "ticker"]).any()
    print("SANITY: SEC thirteen activates broad and elite 13F exactly once with a dedicated part owner.")


def test_completed_accession_markers_and_positions_share_one_read() -> None:
    calls = []
    table = Tables.sec13f_hr
    assert table.empty_marker is not None
    manifest = {
        "accession_number": "original",
        "cik": "0000000001",
        "period": "2023-12-31",
        "filing_date": "2024-02-14",
        "parse_status": "complete",
        "cusip": table.empty_marker[1],
    }
    position = {**manifest, "parse_status": "position", "cusip": "123456789", "shares": 10.0, "value_usd": 100.0}

    class Store:
        def columns(self, table: Any) -> list[str]:
            return [
                "accession_number",
                "cik",
                "period",
                "filing_date",
                "accepted_at",
                "amendment_type",
                "amendment_number",
                "report_type",
                "restricted",
                "parsed_row_count",
                "parse_status",
            ]

        def iter_load(self, table: Any, *, columns: list[str], **kwargs: Any) -> Any:
            calls.append((table, kwargs))
            yield pd.DataFrame([manifest, position]).reindex(columns=columns)

    result = iter_filing_vintages(cast(DataStore, Store()), pd.Timestamp("2024-02-20"), ["1"])
    assert result is not None
    [(filings, holdings)] = list(result)
    assert len(filings) == len(holdings) == 1 and holdings.iloc[0]["shares"] == 10.0
    assert len(calls) == 1 and calls[0][1]["markers"] is True
    assert calls[0][1]["where"] == {"parse_status": ["complete", "position"], "cik": ["0000000001"]}
    print("SANITY: completed empty-book evidence and whole-book positions come from one projected marker-inclusive stream.")


def test_manager_accessions_use_one_projected_stream() -> None:
    table = Tables.sec13f_manager_holdings
    assert table.empty_marker is not None
    required = [
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "accepted_at",
        "amendment_type",
        "amendment_number",
        "report_type",
        "restricted",
        "parsed_row_count",
        "parse_status",
    ]
    rows = pd.DataFrame(
        [
            {
                "accession_number": "original",
                "cik": "0000000001",
                "period": "2023-12-31",
                "filing_date": "2024-02-14",
                "parse_status": "complete",
                "cusip": table.empty_marker[1],
            },
            {
                "accession_number": "original",
                "cik": "0000000001",
                "period": "2023-12-31",
                "filing_date": "2024-02-14",
                "parse_status": "position",
                "cusip": "123456789",
                "shares": 10.0,
                "value_usd": 100.0,
            },
        ]
    )
    calls = []

    class Store:
        def columns(self, actual: Any) -> list[str]:
            assert actual == table
            return required

        def iter_load(self, actual: Any, *, columns: list[str], **kwargs: Any) -> Any:
            calls.append((actual, columns, kwargs))
            yield rows.reindex(columns=columns)

        def load(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Manager history must use bounded source iteration")

    result = iter_filing_vintages(cast(DataStore, Store()), pd.Timestamp("2024-02-20"), ["1"], table=table)
    assert result is not None
    [(filings, holdings)] = list(result)
    assert len(filings) == len(holdings) == 1 and holdings.shares.iloc[0] == 10.0
    assert len(calls) == 1 and calls[0][0] == table
    assert calls[0][2]["markers"] is True
    assert calls[0][2]["chunksize"] == 500_000
    assert calls[0][2]["order_by"] == ("cik", "period", "accession_number")
    assert calls[0][2]["where"] == {"parse_status": ["complete", "position"], "cik": ["0000000001"]}
    assert calls[0][2]["date_col"] == "filing_date" and calls[0][2]["until"] == pd.Timestamp("2024-02-20")
    print("SANITY: manager markers and positions use one scoped projected500k stream with the same public cutoff.")


def test_schema_change_reuses_the_computed_panel(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    previous = pd.DataFrame({"date": days, "ticker": "AAA", "f_previous": 1.0})
    sqlite_store.save(Tables.cube_part_sec_thirteen, previous)
    panel = previous.drop(columns="f_previous").assign(f_current=2.0)
    calls = []

    def build(window: PartWindow | None = None, *, existing_columns: Sequence[str] | None = None) -> pd.DataFrame:
        assert window is not None and existing_columns is not None
        calls.append(window.is_full)
        return panel

    monkeypatch.setattr(step, "build_panel", build)
    monkeypatch.setattr(step_module, "load_trading_calendar", lambda store: days)
    step.run()
    assert calls == [False]
    persisted = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert persisted is not None
    pd.testing.assert_frame_equal(persisted.sort_values(["date", "ticker"]).reset_index(drop=True), panel.reset_index(drop=True), check_dtype=False)
    print("SANITY: a changed feature schema replaces the part once using the already computed full-history panel.")
