"""Known-truth distinctions between observed operands, proved absence and undefined division."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common import part_status
from src.data_aggregate.utils.institutionals import inputs
from src.data_aggregate.utils.institutionals import insider_features as insider
from src.data_aggregate.utils.institutionals import short_flow_features as short
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.conftest import make_frames
from tests.data_aggregate.insider_pit_reference import build_prepared_insider_panel
from tests.data_aggregate.test_insider_features import _txn


def test_quiet_proof_never_changes_undefined_insider_ratios_to_zero() -> None:
    idx = pd.bdate_range("2026-01-01", periods=20)
    frames = cast(Any, SimpleNamespace(trading_index=idx, universe=["AAA"], close_split=pd.DataFrame(1.0, index=idx, columns=["AAA"])))
    columns = [
        "code",
        "officer_title_clean",
        "is_officer",
        "role",
        "role_unknown",
        "is_director",
        "is_10b5_1",
        "in_exercise_package",
        "is_management",
        "owner_cik",
        "security_title",
        "security_type",
        "direct_indirect",
    ]
    activity = pd.DataFrame(columns=columns).astype({"in_exercise_package": bool, "is_management": bool, "role_unknown": bool})
    names = [
        "ic_insider_net_buy_ratio_180d",
        "ic_insider_owner_surprise_120d",
        "ic_insider_purchase_pct_prior",
        "ic_insider_management_position_change_60d",
        "ic_insider_distinct_buyers_120d",
    ]
    fields = {name: pd.DataFrame(np.nan, index=idx, columns=["AAA"]) for name in names}
    coverage = insider.InsiderCoverage.from_availability(None, pd.Series({"AAA": idx.max()}))
    insider._observe_quiet_windows(fields, activity, pd.Series({"AAA": pd.Timestamp("2020-01-01")}), frames, coverage)
    for name in names[:-1]:
        assert fields[name].isna().all().all(), f"Undefined ratio {name} must remain NULL under a certified quiet window"
    assert fields[names[-1]].eq(0).all().all()
    print("SANITY: certified quiet counts are zero; net0/0, no scored buys and no eligible prior-position ratios remain NULL.")


def test_insider_boundary_uses_full_per_ticker_cohort_before_selection(sqlite_store: DataStore) -> None:
    rows = pd.DataFrame(
        {
            "accession_number": ["a", "b", "c"],
            "security_type": ["nonderiv", "_empty", "nonderiv"],
            "row_sequence": [1, 0, 1],
            "ticker": ["AAA", "AAA", "BBB"],
            "lineage_role": ["canonical_current", None, "canonical_current"],
            "filing_date": pd.to_datetime(["2026-01-01"] * 3),
            "fetched_at": pd.to_datetime(pd.Series(["2026-07-01", None, "2026-05-31"])),
        }
    )
    sqlite_store.save(Tables.insider_transactions, rows)
    source = inputs.load_insider_transactions(sqlite_store, ["AAA", "BBB"])
    assert source is not None
    assert pd.isna(source.complete_through.loc["AAA"]) and source.complete_through.loc["BBB"] == pd.Timestamp("2026-05-31")
    print("SANITY: an unfetched valid-empty accession blocks only its ticker, including rows excluded by economic selection.")


def test_ftd_other_ticker_observation_cannot_create_zero() -> None:
    idx = pd.bdate_range("2026-07-01", periods=3)
    hist = pd.DataFrame({"ticker": ["AAA"], "date": [idx[0]], "fails_quantity": [12.0], "period": ["202607a"]})
    coverage = hist[["date", "period"]]
    fails, _ = short._settlement_fails(hist, coverage, idx, pd.Index(["AAA", "BBB"]), None)
    assert fails.loc[idx[0], "AAA"] == 12.0 and fails["BBB"].isna().all()
    print("SANITY: a valid explicit legacy FTD observation survives; another ticker's served date cannot fabricate an absent leg.")


def test_insider_boundary_is_the_latest_fetch_day_of_the_ticker(sqlite_store: DataStore) -> None:
    rows = pd.DataFrame(
        {
            "accession_number": ["a", "b"],
            "security_type": ["nonderiv", "nonderiv"],
            "row_sequence": [1, 1],
            "ticker": ["AAA", "AAA"],
            "lineage_role": ["canonical_current"] * 2,
            "filing_date": pd.to_datetime(["2024-01-01", "2026-01-01"]),
            "fetched_at": pd.to_datetime(["2026-06-01", "2026-07-01"]),
        }
    )
    sqlite_store.save(Tables.insider_transactions, rows)
    source = inputs.load_insider_transactions(sqlite_store, ["AAA"])
    assert source is not None and source.complete_through.loc["AAA"] == pd.Timestamp("2026-07-01")
    print("SANITY: a ticker read on several days is complete through its latest fetch day.")


def test_cube_readiness_cannot_infer_global_completeness_from_latest_filing(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = pd.DataFrame(
        {
            "accession_number": ["a", "b"],
            "security_type": ["nonderiv", "_empty"],
            "row_sequence": [1, 0],
            "ticker": ["AAA", "BBB"],
            "filing_date": pd.to_datetime(["2026-07-01"] * 2),
            "fetched_at": pd.to_datetime(["2026-07-02"] * 2),
            "inventory_checked_through": pd.to_datetime(pd.Series(["2026-07-01", None])),
        }
    )
    sqlite_store.save(Tables.insider_transactions, rows)
    sqlite_store.save(Tables.cube_part_prices, pd.DataFrame({"ticker": ["AAA", "BBB"], "date": pd.to_datetime(["2026-07-01"] * 2)}))
    monkeypatch.setattr(part_status, "load_universe_tickers", lambda context: ["AAA", "BBB"])
    status = part_status._insider_source_status(cast(Any, SimpleNamespace(store=sqlite_store)), "2026-07-01", 4)
    assert status["complete_through"] is None and not status["ok"]
    print("SANITY: one recent filing or one verified issuer cannot certify cube readiness while another requested issuer is unverified.")


def test_finra_metrics_use_their_own_defined_paired_operands() -> None:
    idx = pd.bdate_range("2026-01-01", periods=21)
    numerator = pd.DataFrame(1.0, index=idx, columns=["AAA"])
    denominator = pd.DataFrame(2.0, index=idx, columns=["AAA"])
    numerator.iloc[0] = 1000.0
    denominator.iloc[0] = np.nan
    shares = pd.DataFrame(100.0, index=idx, columns=["AAA"])
    ratio, _ = short._shortvol_window_fields(numerator, denominator, numerator.notna() & denominator.notna(), idx, shares)
    assert ratio[20].iloc[20, 0] == 0.5
    quiet = numerator * 0.0
    missing_total = denominator * np.nan
    ratio, turnover = short._shortvol_window_fields(quiet, missing_total, quiet.notna() & missing_total.notna(), idx, shares)
    assert ratio[20].isna().all().all() and turnover is not None and turnover.iloc[-1, 0] == 0.0
    _, undefined = short._shortvol_window_fields(quiet, missing_total, quiet.notna() & missing_total.notna(), idx, shares * 0.0)
    assert undefined is not None and undefined.isna().all().all()
    print(
        "SANITY: legacy paired short/total observations remain usable; an unpaired numerator cannot enter the ratio, and checked short0/positive shares gives turnover0 while denominator0 stays NULL."
    )


def test_ftd_no_event_count_does_not_impute_a_zero_zscore() -> None:
    idx = pd.bdate_range("2024-01-01", periods=180)
    basis = pd.DataFrame({"ZERO": np.zeros(len(idx)), "POSITIVE": np.ones(len(idx)), "MISSING": np.full(len(idx), np.nan)}, index=idx)
    zscore = short.self_history_z(basis, window=short.Z_WINDOW, min_periods=short.Z_MIN_PERIODS)
    result = short._ftd_settlement_persistence(basis, pd.Series(True, index=idx), idx)
    assert zscore.isna().all().all()
    assert result["ZERO"].iloc[-1] == 0.0 and result[["POSITIVE", "MISSING"]].isna().all().all()
    print(
        "SANITY: constant-zero/positive histories both have undefined z=0/0; independently observed all-zero no-event flags support count0, while constant-positive and missing-basis counts remain NULL."
    )


def test_ftd_zero_adv_remains_undefined_and_positive_adv_supports_zero() -> None:
    idx = pd.bdate_range("2025-12-01", periods=70)
    fails = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    settlement = short.FtdSettlement(fails, pd.Series(True, index=idx))
    coverage = pd.DataFrame({"date": idx, "period": [day.strftime("%Y%m") + ("a" if day.day <= 15 else "b") for day in idx]})
    available = {str(period): group["date"].max() + pd.Timedelta(days=16) for period, group in coverage.groupby("period")}
    publication = short.FtdPublication(coverage, available)
    zero = short._fails_fields(settlement, short.ShortFlowCapital(None, fails.copy()), publication, None)["ic_ftd_to_adv20"]
    defined = short._fails_fields(settlement, short.ShortFlowCapital(None, fails + 10.0), publication, None)["ic_ftd_to_adv20"]
    assert zero.isna().all().all() and defined.iloc[-1, 0] == 0.0
    print("SANITY: FTD0/ADV0 remains NULL; FTD0/positive independently observed ADV is a defined zero.")


def test_physically_absent_legacy_proof_preserves_observed_operands_only(sqlite_store: DataStore) -> None:
    transactions = pd.DataFrame(
        [
            _txn(accession_number="buy", row_sequence=1, issuer_cik="0000000001", lineage_role="canonical_current", value_usd=100.0),
            _txn(
                accession_number="sell",
                row_sequence=1,
                issuer_cik="0000000001",
                lineage_role="canonical_current",
                transaction_code="S",
                value_usd=50.0,
            ),
        ]
    )
    sqlite_store.save(Tables.insider_transactions, transactions)
    dates = pd.bdate_range("2015-01-01", "2015-12-31")
    sqlite_store.save(Tables.short_interest, pd.DataFrame({"ticker": "AAA", "date": dates[:21], "short_volume": 0.0, "total_volume": 2.0}))
    sqlite_store.save(
        Tables.sec_fails_to_deliver, pd.DataFrame({"ticker": ["AAA"], "date": dates[:1], "fails_quantity": [12.0], "period": ["201501a"]})
    )
    source = inputs.load_insider_transactions(sqlite_store, ["AAA", "BBB"])
    assert source is not None and source.complete_through.isna().all()
    close = pd.DataFrame(100.0, index=dates, columns=["AAA", "BBB"])
    frames = make_frames(dates, {"AAA": {"AAA": 1.0}, "BBB": {"BBB": 1.0}}, close_split=close)
    shares = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "as_of": pd.to_datetime(["2014-01-01"] * 2), "sharesOutstanding": [1e6] * 2, "sharesOutstandingPit": [1e6] * 2}
    )
    panel = build_prepared_insider_panel(
        frames, source.transactions, shares_out_history=shares, coverage=insider.InsiderCoverage.from_availability(None, source.complete_through)
    ).set_index(["date", "ticker"])
    ratio = panel["f_ic_insider_net_buy_ratio_180d"]
    assert ratio.loc[(pd.Timestamp("2015-06-01"), "AAA")] == pytest.approx(1.0 / 3.0)
    assert pd.isna(ratio.loc[(pd.Timestamp("2015-12-01"), "AAA")]) and ratio.xs("BBB", level="ticker").isna().all()
    short_rows = sqlite_store.load(Tables.short_interest, project=True)
    assert short_rows is not None
    short_volume = short_rows.pivot(index="date", columns="ticker", values="short_volume")
    total = short_rows.pivot(index="date", columns="ticker", values="total_volume")
    ratios, _ = short._shortvol_window_fields(short_volume, total, total.notna(), pd.DatetimeIndex(short_volume.index), None)
    assert ratios[20].iloc[-1, 0] == 0.0
    fails_rows, coverage = inputs.load_fails(sqlite_store, ["AAA", "BBB"])
    assert fails_rows is not None and coverage is not None
    fails, _ = short._settlement_fails(fails_rows, coverage, dates, pd.Index(["AAA", "BBB"]), None)
    assert fails.loc[dates[0], "AAA"] == 12.0 and fails["BBB"].isna().all()
    for table, proof in (
        (Tables.insider_transactions, "inventory_checked_through"),
        (Tables.short_interest, "source_checked_at"),
        (Tables.sec_fails_to_deliver, "source_checked_at"),
    ):
        assert proof not in sqlite_store.columns(table)
    print(
        "SANITY: physically legacy schemas keep explicit paired insider/FINRA operands and explicit FTD quantities, create no quiet or absent-ticker zeros, and execute no read-triggered schema evolution."
    )
