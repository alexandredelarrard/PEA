"""Known-truth controls for changed-book manager preparation and public replay."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_aggregate.utils.sec_thirteen import superinvestor_features as features
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filing_dates
from src.data_aggregate.utils.sec_thirteen.institutional_vintages import iter_public_books
from tests.conftest import make_frames


def _packet(
    accession: str, filed: str, amounts: dict[str, float], *, kind: str = "original", period: str = "2024-06-30"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = {"cik": "0000000001", "period": pd.Timestamp(period), "filing_date": pd.Timestamp(filed), "accession_number": accession}
    numeric = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
    rows = []
    for cusip, shares in amounts.items():
        row = {**identity, "cusip": cusip, "position_type": "common", "class_evidence": '["common"]', **dict.fromkeys(numeric, 0.0)}
        row.update({"shares": float(shares), "value_usd": float(shares * 10)})
        row.update({"raw_" + column: row[column] for column in numeric if "value" in column})
        rows.append(row)
    columns = [*identity, "cusip", "position_type", "class_evidence", *numeric, *("raw_" + column for column in numeric if "value" in column)]
    manifest = {**identity, "amendment_type": kind, "amendment_number": 0, "report_type": "13F HOLDINGS REPORT", "restricted": False}
    return pd.DataFrame([{**manifest, "parse_status": "complete", "parsed_row_count": len(rows)}]), pd.DataFrame(rows, columns=columns)


def _inputs() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.bdate_range("2024-01-01", "2026-01-30")
    packets = [
        _packet("prior-one", "2024-08-14", {"111111111": 100, "999999999": 900}),
        _packet("prior-two", "2024-08-16", {"111111111": 200, "999999999": 800}),
        _packet("current-one", "2024-11-14", {"111111111": 120, "222222222": 80, "999999999": 800}, period="2024-09-30"),
        _packet("current-two", "2024-11-18", {"111111111": 220, "222222222": 80, "999999999": 700}, period="2024-09-30"),
        _packet("prior-fix", "2024-12-01", {"111111111": 90, "999999999": 910}, kind="restatement"),
        _packet("current-empty", "2024-12-05", {}, period="2024-09-30", kind="restatement"),
    ]
    for index in (1, 3):
        for frame in packets[index]:
            frame["cik"] = "0000000002"
    filings = pd.concat([item[0] for item in packets], ignore_index=True)
    holdings = pd.concat([item[1] for item in packets], ignore_index=True)
    filings, holdings = clean_filing_dates(filings, holdings)
    cusip_map = pd.DataFrame({"cusip": ["111111111", "222222222"], "ticker": ["AAA", "BBB"]})
    return grid, filings, holdings, cusip_map


def _panel(grid: pd.DatetimeIndex, filings: pd.DataFrame, holdings: pd.DataFrame, cusip_map: pd.DataFrame) -> pd.DataFrame:
    close = pd.DataFrame(10.0, index=grid, columns=["AAA", "BBB"])
    frames = make_frames(grid, {}, universe=["AAA", "BBB"], close_split=close)
    return features.build_superinvestor_feature_panel(
        frames, holdings, {"0000000001", "0000000002"}, cusip_map=cusip_map, filings=filings, valuation_close=close, value_basis={"min_securities": 1}
    )


def test_manager_statistics_and_ranks_only_revisit_changed_books(monkeypatch: pytest.MonkeyPatch) -> None:
    grid, filings, holdings, cusip_map = _inputs()
    statistics = Mock(wraps=features.manager_quarter_state)
    ranks = Mock(wraps=features.manager_stock_conviction)
    monkeypatch.setattr(features, "manager_quarter_state", statistics)
    monkeypatch.setattr(features, "manager_stock_conviction", ranks)
    panel = _panel(grid, filings, holdings, cusip_map).set_index(["date", "ticker"])

    calls = {"statistics": [len(call.args[0]) for call in statistics.call_args_list], "ranks": [len(call.args[0]) for call in ranks.call_args_list]}
    assert calls == {"statistics": [2, 2, 3, 3, 2, 0], "ranks": [2, 2, 3, 3, 2, 0]}, calls
    assert panel.loc[(pd.Timestamp("2024-11-15"), "BBB"), "f_ic_super_holders"] == 0.5
    assert panel.loc[(pd.Timestamp("2024-11-19"), "BBB"), "f_ic_super_holders"] == 1.0
    assert panel.loc[(pd.Timestamp("2024-12-06"), "BBB"), "f_ic_super_holders"] == 0.5
    print("SANITY: six changed books each prepare statistics/ranks once; new-ticker past zeros and an empty replacement preserve holder support.")


def test_public_manager_state_is_computed_once_per_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    grid, filings, holdings, cusip_map = _inputs()
    snapshot_count = len(list(iter_public_books(filings, holdings, grid, settle_days=0, history_quarters=None, expiry_quarters=4)))
    public_state = Mock(wraps=features.public_state)
    monkeypatch.setattr(features, "public_state", public_state)
    panel = _panel(grid, filings, holdings, cusip_map)

    assert not panel.empty
    assert public_state.call_count == snapshot_count, (public_state.call_count, snapshot_count)
    print(f"SANITY: {snapshot_count} public/expiry snapshots build manager public state once each, including expiry without a filing.")


def test_last_manager_book_can_be_replaced_by_a_complete_empty_book() -> None:
    grid, _, _, cusip_map = _inputs()
    original = _packet("only-original", "2024-08-14", {"111111111": 100, "999999999": 900})
    empty = _packet("only-empty", "2024-12-05", {}, kind="restatement")
    filings = pd.concat([original[0], empty[0]], ignore_index=True)
    holdings = pd.concat([original[1], empty[1]], ignore_index=True)
    filings, holdings = clean_filing_dates(filings, holdings)

    panel = _panel(grid, filings, holdings, cusip_map).set_index(["date", "ticker"])

    assert panel.loc[(pd.Timestamp("2024-08-15"), "AAA"), "f_ic_super_holders"] == 1.0
    assert panel.loc[(pd.Timestamp("2024-12-06"), "AAA"), "f_ic_super_holders"] == 0.0
    assert "f_ic_super_full_exits" not in panel.columns
    print("SANITY: replacing the only resolved book with an empty original-quarter restatement clears holders without inventing a quarterly exit.")
