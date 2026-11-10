"""The cube's FTD series from the as-filed security lines (P4, REQ-007, AC-005): CUSIP -> company through the dated
`universe_cusip` lines, quantity x conversion ratio summed per (ticker, settlement date), explicit zeros on served dates,
uncertain dates left out; then through `prepare_fails_quantities`, the frame the short-flow builder receives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals import inputs
from src.data_aggregate.utils.institutionals.short_flow_features import prepare_fails_quantities
from src.data_store.schema import Tables
from tests.canonical_schema import create_canonical
from tests.conftest import _store

BRK_A, BRK_B, AAPL = "084670108", "084670702", "037833100"
#: The `universe_cusip` columns the institutionals step reads once.
LINE_READ = [
    "canonical_company",
    "market_symbol",
    "conversion_ratio",
    "valid_from",
    "valid_to",
    "security_id",
    "source",
    "source_symbol",
    "security_class",
    "lineage_role",
    "cusip",
]


@pytest.fixture
def sqlite_store(sqlite_store):
    create_canonical(sqlite_store.engine, [Tables.sec_fails_to_deliver_security, Tables.sharadar_actions, Tables.prices_splits])
    return sqlite_store


def _lines() -> pd.DataFrame:
    """BRK-A (class A, 1,500 B shares each, line ends on trade date 2025-03-04), BRK-B and AAPL canonical."""
    rows = [
        ("BRK-B", "BRK-A", 1500.0, "2010-01-01", "2025-03-04", f"C{BRK_A}", "ftd", "BRKA", "class_A", "secondary_class", BRK_A),
        ("BRK-B", "BRK-B", 1.0, "2010-01-01", None, f"C{BRK_B}", "ftd", "BRKB", "class_B", "canonical_current", BRK_B),
        ("AAPL", "AAPL", 1.0, "2010-01-01", None, f"C{AAPL}", "ftd", "AAPL", "common", "canonical_current", AAPL),
    ]
    frame = pd.DataFrame(rows, columns=LINE_READ)
    return frame.assign(valid_from=pd.to_datetime(frame["valid_from"]), valid_to=pd.to_datetime(frame["valid_to"]))


def _stored(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "cusip", "source_symbol", "fails_quantity", "price"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.assign(
        trade_date=frame["date"] - pd.offsets.BDay(1),
        description="",
        fails_value=frame["fails_quantity"] * frame["price"],
        period="202503a",
    )


def test_class_lines_sum_times_ratio_with_supported_zeros_and_an_uncertain_date(sqlite_store):
    """03-04: BRK-A 2 + BRK-B 100 = 3,100 B shares; 03-05: no Berkshire line -> 0; 03-06: a BRK-A line after its line ended
    is unresolved, so the date is unknown for BRK-B, never 0 or 3,100."""
    sqlite_store.save(
        Tables.sec_fails_to_deliver_security,
        _stored(
            [
                ("2025-03-04", BRK_A, "BRKA", 2.0, 700_000.0),
                ("2025-03-04", BRK_B, "BRKB", 100.0, 470.0),
                ("2025-03-04", AAPL, "AAPL", 10.0, 240.0),
                ("2025-03-05", AAPL, "AAPL", 5.0, 241.0),
                ("2025-03-06", AAPL, "AAPL", 7.0, 242.0),
                ("2025-03-06", BRK_A, "BRKA", 1.0, 700_000.0),
            ]
        ),
    )

    fails, coverage = inputs.load_fails(sqlite_store, ["AAPL", "BRK-B"], _lines(), {})

    got = {(t, f"{d:%m-%d}"): q for t, d, q in fails[["ticker", "date", "fails_quantity"]].itertuples(index=False)}
    assert got == {("AAPL", "03-04"): 10.0, ("AAPL", "03-05"): 5.0, ("AAPL", "03-06"): 7.0, ("BRK-B", "03-04"): 3_100.0, ("BRK-B", "03-05"): 0.0}
    assert fails.loc[fails["ticker"].eq("BRK-B"), "fails_value"].tolist() == [2 * 700_000.0 + 100 * 470.0, 0.0]
    assert pd.to_datetime(coverage["date"]).dt.strftime("%m-%d").tolist() == ["03-04", "03-05", "03-06"]

    idx = pd.DatetimeIndex(pd.to_datetime(["2025-03-04", "2025-03-05", "2025-03-06"]))
    settlement = prepare_fails_quantities(fails, coverage, idx, pd.Index(["AAPL", "BRK-B"], name="ticker"), None)
    assert settlement.quantities["BRK-B"].tolist()[:2] == [3_100.0, 0.0] and np.isnan(settlement.quantities["BRK-B"].iloc[2])
    print("\nSANITY: BRK-B 03-04 = 2 x 1,500 + 100 = 3,100 (dollars summed as filed); 03-05 a supported zero; 03-06 unknown")
    print(f"  (an unresolved BRK-A line); AAPL {settlement.quantities['AAPL'].tolist()}; the builder frame carries the same values.")


def test_real_stored_lines_reach_the_builder_as_company_series() -> None:
    """Real rows (read-only): the stored lines of each multi-class company sum x ratio per settlement date, through the
    loader and `prepare_fails_quantities`."""
    store = _store()
    lines = store.load(Tables.universe_cusip, columns=LINE_READ, where={"source": "ftd"}, optional=True)
    if (
        lines is None
        or not store.exists(Tables.sec_fails_to_deliver_security)
        or "security_id" in store.columns(Tables.sec_fails_to_deliver_security)
    ):
        pytest.skip("no universe_cusip or no as-filed sec_fails_to_deliver_security: run the FTD ingest and identity first")
    summed = lines.loc[lines["lineage_role"].isin(["canonical_current", "canonical_predecessor", "secondary_class"])]
    multi = sorted(summed.groupby("canonical_company")["cusip"].nunique().loc[lambda n: n.gt(1)].index)
    bugfix = {}

    fails, coverage = inputs.load_fails(store, multi, lines, bugfix)

    assert not fails.duplicated(["ticker", "date"]).any() and set(fails["ticker"]) <= set(multi)
    assert set(pd.to_datetime(fails["date"])) <= set(pd.to_datetime(coverage["date"])), "a series row lies on a served date"
    raw = store.load(
        Tables.sec_fails_to_deliver_security, columns=["date", "trade_date", "cusip", "fails_quantity"], where={"cusip": sorted(summed["cusip"])}
    )
    assert raw is not None
    day = pd.to_datetime(raw.groupby("date")["cusip"].nunique().idxmax())
    # each line on `day` with the master row covering its trade date (BRK-A is 30 before 2010-01-21, 1,500 after)
    own = raw.loc[pd.to_datetime(raw["date"]).eq(day)].merge(
        summed[["cusip", "canonical_company", "conversion_ratio", "valid_from", "valid_to"]], on="cusip"
    )
    trade = pd.to_datetime(own["trade_date"])
    own = own.loc[trade.ge(pd.to_datetime(own["valid_from"])) & (trade.lt(pd.to_datetime(own["valid_to"])) | own["valid_to"].isna())]
    expected = (own["fails_quantity"] * own["conversion_ratio"]).groupby(own["canonical_company"]).sum()
    on_day = fails.loc[pd.to_datetime(fails["date"]).eq(day)].set_index("ticker")["fails_quantity"]
    checked = [t for t in expected.index if t in on_day.index]
    assert checked and np.allclose(on_day.reindex(checked), expected.reindex(checked), rtol=1e-9)

    idx = pd.DatetimeIndex(sorted(set(pd.to_datetime(coverage["date"]))))
    settlement = prepare_fails_quantities(fails, coverage, idx, pd.Index(multi, name="ticker"), None)
    assert np.allclose(settlement.quantities.loc[day, checked], expected.reindex(checked), rtol=1e-9)
    print(f"\nSANITY (real rows): {len(fails)} series rows for {multi} over {len(coverage)} served dates; on {day.date()} the")
    print(f"  builder frame holds {settlement.quantities.loc[day, checked].round(0).to_dict()} = sum(qty x ratio) of the stored lines.")
