"""Independent broad and elite 13F merge ownership."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.transformers import step_cube_sec_thirteen as step_module
from src.data_aggregate.transformers.step_cube_sec_thirteen import SecThirteenResources, StepCubeSecThirteen
from src.data_aggregate.utils.common.incremental import PartWindow
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.holdings_clean import attach_tickers
from src.data_aggregate.utils.sec_thirteen.inputs import iter_filing_vintages
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import EMISSION, SuperinvestorFeatureHistory
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.institutional_vintages import prepare_book_vintages
from tests.data_aggregate.test_institutional_step_contract import context, fixed_sec_thirteen_projection, seed_prices
from tests.data_aggregate.test_superinvestor_books import _packet
from tests.support.sec_thirteen_panel import unsaved_panel


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
    basis = pd.DataFrame(1.0, index=pd.bdate_range("2023-01-02", "2024-12-31"), columns=["AAA"])
    frames = PriceFrames(
        trading_index=grid, universe=("AAA",), peers={"AAA": {"AAA": 1.0}}, close_split=df_close, level_factor=df_close / 100.0, security_basis=basis
    )
    ciks = frozenset(cik for filings, holdings in packets for df_rows in (filings, holdings) for cik in df_rows.cik)
    resources = SecThirteenResources(frames, {"": ciks})
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
    monkeypatch.setattr(step_module, "load_peers_or_raise", lambda ctx: {})
    broad = Mock(
        return_value=pd.DataFrame({"period": [pd.Timestamp("2023-09-30")], "as_of": [days[0]], "ticker": ["AAA"], "ic_inst_concentration": [0.4]})
    )
    levels = {name: pd.DataFrame({"AAA": [float("nan")]}, index=days[:1]) for name in EMISSION}
    levels["ic_super_holders"].iloc[0, 0] = 1.0
    elite = Mock(return_value=SuperinvestorFeatureHistory(levels, pd.DataFrame()))
    monkeypatch.setattr(step, "_institutional_features", broad)
    monkeypatch.setattr(step, "_superinvestor_history", elite)
    result = unsaved_panel(step)
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

        def load(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Whole roster books use bounded source iteration")

    result = iter_filing_vintages(cast(DataStore, Store()), pd.Timestamp("2024-02-20"), ["1"])
    assert result is not None
    [(filings, holdings)] = list(result)
    assert len(filings) == len(holdings) == 1 and holdings.iloc[0]["shares"] == 10.0
    assert len(calls) == 1 and calls[0][0] == table and calls[0][1]["markers"] is True
    assert calls[0][1]["where"] == {"parse_status": ["complete", "position"], "cik": ["0000000001"]}
    assert calls[0][1]["chunksize"] == 500_000 and calls[0][1]["order_by"] == ("cik", "period", "accession_number")
    assert calls[0][1]["date_col"] == "filing_date" and calls[0][1]["until"] == pd.Timestamp("2024-02-20")
    print(
        "SANITY: completed empty-book evidence and whole roster books come from one projected 500k marker-inclusive stream on the literal CIK list."
    )


def test_schema_change_reuses_the_computed_panel(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeSecThirteen(ctx, ctx.config)
    previous = pd.DataFrame({"date": days, "ticker": "AAA", "f_previous": 1.0})
    sqlite_store.save(Tables.cube_part_sec_thirteen, previous)
    panel = previous.drop(columns="f_previous").assign(f_current=2.0)
    project = fixed_sec_thirteen_projection(step, monkeypatch, panel)
    monkeypatch.setattr(step_module, "load_trading_calendar", lambda store: days)
    step.run()
    # The incremental window projection, then one whole-history reprojection reused by the schema fallback.
    assert project.call_count == 2 and len(project.call_args_list[-1].args[2]) == len(days)
    persisted = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert persisted is not None
    pd.testing.assert_frame_equal(persisted.sort_values(["date", "ticker"]).reset_index(drop=True), panel.reset_index(drop=True), check_dtype=False)
    print("SANITY: a changed feature schema replaces the part once using the already computed full-history panel.")


def _hr_book(cik: str, period: str, filed: str, rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """One `sec13f_hr` accession as extraction stores it: (cusip, ticker, shares, value_usd) positions + completion marker."""
    empty = cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1]
    base = {
        "cik": cik,
        "period": pd.Timestamp(period),
        "filing_date": pd.Timestamp(filed),
        "accession_number": f"{cik}-{period}",
        "accepted_at": pd.Timestamp(filed, tz="UTC") + pd.Timedelta(hours=10),
        "amendment_type": "original",
        "amendment_number": 0,
        "report_type": "13F HOLDINGS REPORT",
        "restricted": False,
        "parsed_row_count": len(rows),
        "position_type": "common",
        "class_evidence": '["COM"]',
        **dict.fromkeys(("call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value"), 0.0),
        **dict.fromkeys(("raw_call_value", "raw_put_value", "raw_debt_value", "raw_other_value"), 0.0),
    }
    marker = {**base, "cusip": empty, "ticker": empty, "parse_status": "complete", "shares": 0.0, "value_usd": 0.0}
    positions = [
        {**base, "cusip": cusip, "ticker": ticker, "parse_status": "position", "shares": shares, "value_usd": value, "raw_value_usd": value}
        for cusip, ticker, shares, value in rows
    ]
    return pd.DataFrame([marker, *positions])


def _roster_history(
    store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, quotes: dict[str, float], books: list[pd.DataFrame]
) -> SuperinvestorFeatureHistory:
    """The step's own roster read and superinvestor path over `sec13f_hr` books; only market and roster loaders are supplied."""
    store.save(Tables.sec13f_hr, pd.concat(books, ignore_index=True))
    ctx = context(store, tmp_path)
    ctx.config.build_cube.institutionals.value_basis = {"min_securities": 1}
    grid = pd.bdate_range("2024-01-02", "2024-12-31")
    close = pd.DataFrame([quotes] * len(grid), index=grid)
    basis = pd.DataFrame(1.0, index=pd.bdate_range("2023-01-02", "2024-12-31"), columns=list(quotes))
    frames = PriceFrames(
        trading_index=grid, universe=tuple(quotes), peers={}, close_split=close, close_pit=close, level_factor=close / close, security_basis=basis
    )
    ciks = sorted({str(cik) for book in books for cik in book.cik})
    monkeypatch.setattr(step_module, "load_price_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(step_module, "load_peers_or_raise", lambda context: {})
    monkeypatch.setattr(step_module.capital_inputs, "load_source", lambda store, table: pd.DataFrame({"cik": ciks}))
    monkeypatch.setattr(step_module, "filer_ciks", lambda ciks, config_dir: set(ciks))
    monkeypatch.setattr(step_module, "load_manager_cohorts", lambda config_dir, roster, k: {"": frozenset(roster), "topk": frozenset(roster)})
    step = StepCubeSecThirteen(ctx, ctx.config)
    history = step._superinvestor_history(step._load_resources())
    assert history is not None, "the roster books stored in sec13f_hr produce no superinvestor history"
    return history


def _on(history: SuperinvestorFeatureHistory, name: str, day: str) -> pd.Series:
    """The event row of `name` in force on `day`."""
    df_level = history.levels[name]
    return df_level.loc[df_level.index[df_level.index <= pd.Timestamp(day)].max()]


def test_roster_books_read_merged_company_rows_from_sec13f_hr(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Known truth: one BRK-B row merges 1 BRK-A (x1,500) + 1,000 BRK-B at extraction; the class lines never meet in the cube."""
    brk_q1, brk_q2 = ("084670702", "BRK-B", 2_500.0, 1_000_000.0), ("084670702", "BRK-B", 3_000.0, 1_200_000.0)
    aaa, unmapped = ("AAA000001", "AAA", 10_000.0, 1_000_000.0), ("999999991", "_empty", 1_000.0, 2_000_000.0)
    q1 = _hr_book("0000000001", "2024-03-31", "2024-05-10", [brk_q1, aaa, unmapped])
    q2 = _hr_book("0000000001", "2024-06-30", "2024-08-12", [brk_q2, aaa, (*unmapped[:3], 1_800_000.0)])
    history = _roster_history(sqlite_store, monkeypatch, tmp_path, {"AAA": 100.0, "BRK-B": 400.0}, [q1, q2])
    assert _on(history, "ic_super_conviction_weight", "2024-06-28")["BRK-B"] == pytest.approx(0.25)
    q2_day = "2024-08-20"
    assert _on(history, "ic_super_holders", q2_day)["BRK-B"] == 1 and _on(history, "ic_super_top10_holders", q2_day)["BRK-B"] == 1
    assert _on(history, "ic_super_conviction_weight", q2_day)["BRK-B"] == pytest.approx(1_200_000 / 4_000_000)
    assert _on(history, "ic_super_shares_chg", q2_day)["BRK-B"] == pytest.approx(0.2)
    assert _on(history, "ic_super_topk_shares_chg", q2_day)["AAA"] == pytest.approx(0.0)
    print("SANITY: BRK-B merged row (2,500 -> 3,000 B-equivalent shares) gives weight .25 -> .30 of the 4m whole book and a known +20% change.")


def test_unmapped_rows_stay_in_the_whole_book_without_a_ticker(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Rows stored on the `_empty` ticker sentinel are positions with no ticker: they count in book totals and ranks only."""
    first, second, third = (("99999999" + str(n), "_empty", 10.0) for n in (1, 2, 3))
    q1 = _hr_book(
        "0000000001", "2024-03-31", "2024-05-10", [("AAA000001", "AAA", 10_000.0, 1_000_000.0), (*first, 1_500_000.0), (*second, 1_500_000.0)]
    )
    q2 = _hr_book(
        "0000000001", "2024-06-30", "2024-08-12", [("AAA000001", "AAA", 12_000.0, 1_200_000.0), (*second, 1_400_000.0), (*third, 1_400_000.0)]
    )
    assert attach_tickers(q1.loc[q1.parse_status.eq("position")], None)["ticker"].tolist() == ["AAA", None, None]
    history = _roster_history(sqlite_store, monkeypatch, tmp_path, {"AAA": 100.0}, [q1, q2])
    assert _on(history, "ic_super_conviction_weight", "2024-06-28")["AAA"] == pytest.approx(0.25)
    assert _on(history, "ic_super_conviction_weight", "2024-08-20")["AAA"] == pytest.approx(0.3)
    assert _on(history, "ic_super_shares_chg", "2024-08-20")["AAA"] == pytest.approx(0.2)
    assert _on(history, "ic_super_breadth_chg", "2024-08-20")["AAA"] == 0
    print("SANITY: two unmapped positions (3m of a 4m book) dilute AAA to .25/.30; their CUSIP swap leaves the AAA +20% known change intact.")


def test_a_company_cusip_change_is_not_an_evidenced_trade(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A company row moving to a new canonical CUSIP keeps its holder but never reads as a share change."""
    q1 = _hr_book("0000000001", "2024-03-31", "2024-05-10", [("OLD000001", "AAA", 10_000.0, 1_000_000.0), ("CCC000001", "CCC", 1_000.0, 100_000.0)])
    q2 = _hr_book("0000000001", "2024-06-30", "2024-08-12", [("NEW000001", "AAA", 15_000.0, 1_500_000.0), ("CCC000001", "CCC", 1_500.0, 150_000.0)])
    history = _roster_history(sqlite_store, monkeypatch, tmp_path, {"AAA": 100.0, "CCC": 100.0}, [q1, q2])
    day = "2024-08-20"
    assert _on(history, "ic_super_holders", day)["AAA"] == 1
    for name in ("ic_super_shares_chg", "ic_super_breadth_chg", "ic_super_trade_balance"):
        assert pd.isna(_on(history, name, day)["AAA"]), name
    assert _on(history, "ic_super_shares_chg", day)["CCC"] == pytest.approx(0.5)
    print("SANITY: AAA OLD->NEW CUSIP keeps 1 holder but no share change, breadth or balance; unchanged CCC still reads +50%.")
