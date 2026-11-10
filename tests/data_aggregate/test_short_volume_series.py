"""The cube's short-volume series from the as-filed FINRA lines (P5, REQ-007, AC-005): symbol -> company through the
dated `universe_cusip` FTD keys, else the lineage-window fallback `S<cik>:<symbol>` from `universe_cik_symbol`; volume x
conversion ratio summed per (ticker, date); a served date a company's symbol supports but with no line of it is a
supported absence (short volume 0, total unknown)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals import inputs
from src.data_store.schema import Tables
from src.utils.security_series import finra_key, finra_marker_kind, short_volume_series
from tests.canonical_schema import create_canonical
from tests.conftest import _store
from tests.data_extract.institutionals.test_short_volume_security import MASTER_ROWS, _lineage, _master
from tests.data_extract.institutionals.test_short_volume_security import UNIVERSE as RULE_UNIVERSE

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
#: The `universe_cik_symbol` columns the institutionals step reads once.
LINEAGE_READ = ["company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]
UNIVERSE = ["AAPL", "AAA", "BRK-B"]


@pytest.fixture
def sqlite_store(sqlite_store):
    create_canonical(sqlite_store.engine, [Tables.sec_short_volume_security])
    return sqlite_store


def _lines() -> pd.DataFrame:
    """AAPL common; BRK-A (class A, 1,500 B shares each from 2010-01-21) and BRK-B. AAA has no FTD line."""
    rows = [
        ("AAPL", "AAPL", 1.0, "2009-06-26", None, "C037833100", "ftd", "AAPL", "common", "canonical_current", "037833100"),
        ("BRK-B", "BRK-A", 1500.0, "2010-01-21", None, "C084670108", "ftd", "BRKA", "class_A", "secondary_class", "084670108"),
        ("BRK-B", "BRK-B", 1.0, "2010-01-19", None, "C084670702", "ftd", "BRKB", "class_B", "canonical_current", "084670702"),
    ]
    frame = pd.DataFrame(rows, columns=LINE_READ)
    return frame.assign(valid_from=pd.to_datetime(frame["valid_from"]), valid_to=pd.to_datetime(frame["valid_to"]))


def _red_lineage() -> pd.DataFrame:
    """Each company's open window on its roster CIK and its listed symbols; AAA's symbol from 2019."""
    rows = [
        ("E0000320193", "AAPL", "0000320193", "cik_window", "", "1900-01-01", "register"),
        ("E0000320193", "AAPL", "0000320193", "symbol", "AAPL", "1980-12-12", "form345,roster"),
        ("E0001067983", "BRK-B", "0001067983", "cik_window", "", "1900-01-01", "register"),
        ("E0001067983", "BRK-B", "0001067983", "symbol", "BRK-B", "1996-05-09", "form345,roster"),
        ("E0000000100", "AAA", "0000000100", "cik_window", "", "1900-01-01", "roster"),
        ("E0000000100", "AAA", "0000000100", "symbol", "AAA", "2019-01-02", "form345,roster"),
    ]
    frame = pd.DataFrame(rows, columns=["company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "sources"])
    return frame.assign(valid_from=pd.to_datetime(frame["valid_from"]), valid_to=pd.NaT, status="corroborated")[LINEAGE_READ]


def _stored(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """FINRA lines as filed: (day, symbol, short volume, total volume)."""
    frame = pd.DataFrame(rows, columns=["date", "source_symbol", "short_volume", "total_volume"])
    return frame.assign(date=pd.to_datetime(frame["date"]), market="Q,N", short_exempt_volume=0.0)


def test_classes_sum_times_ratio_with_a_supported_absence_and_a_lineage_fallback(sqlite_store):
    """01-02: BRK/A 2 + BRK/B 100 = 3,100 B shares short; AAA (no FTD line) maps through its window CIK as
    `S0000000100:AAA`. 01-03 is served (AAPL's line) with no Berkshire or AAA line: a supported absence, 0 short and an
    unknown total."""
    sqlite_store.save(
        Tables.sec_short_volume_security,
        _stored(
            [
                ("2020-01-02", "AAPL", 50.0, 100.0),
                ("2020-01-02", "BRK/A", 2.0, 10.0),
                ("2020-01-02", "BRK/B", 100.0, 400.0),
                ("2020-01-02", "AAA", 30.0, 300.0),
                ("2020-01-03", "AAPL", 60.0, 120.0),
            ]
        ),
    )

    short, finra = inputs.load_short_volume(sqlite_store, UNIVERSE, _lines(), _red_lineage())

    assert short is not None and list(short.columns) == ["date", "ticker", "short_volume", "total_volume"]
    assert pd.api.types.is_datetime64_any_dtype(short["date"]) and not short.duplicated(["ticker", "date"]).any()
    got = {(t, f"{d:%m-%d}"): (s, v) for d, t, s, v in short.itertuples(index=False)}
    assert got[("BRK-B", "01-02")] == (100.0 + 1_500 * 2.0, 400.0 + 1_500 * 10.0), got
    assert got[("BRK-B", "01-03")][0] == 0.0 and np.isnan(got[("BRK-B", "01-03")][1]), "a supported absence"
    assert got[("AAA", "01-02")] == (30.0, 300.0) and got[("AAA", "01-03")][0] == 0.0
    assert got[("AAPL", "01-03")] == (60.0, 120.0)
    assert finra is not None and finra[["ticker", "security_id", "lineage_role", "security_class"]].drop_duplicates().values.tolist() == [
        ["AAA", "S0000000100:AAA", "canonical_current", "common"]
    ]
    assert pd.to_datetime(finra["date"]).dt.strftime("%m-%d").tolist() == ["01-02"]
    print("\nSANITY: BRK-B 01-02 = 100 + 1,500 x 2 = 3,100 short (15,400 total); 01-03 a supported absence (0, total NaN);")
    print("  AAA, never in FTD, maps through its window CIK as S0000000100:AAA (30 / 300); the builders' four columns kept.")


# --------------------------------------------------------------------------- known-truth rules (ported from the extraction stamp)

# The real Q2a master rows and the hand-written E25 lineage of `test_short_volume_security`.


def _series(day_rows: dict[str, list[tuple[str, float, float]]], master: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Run the rules on FINRA lines `{day: [(symbol, short, total)]}`; returns `{(ticker, day): (short, total)}` and the fallback."""
    lines = _stored([(day, symbol, short, total) for day, rows in day_rows.items() for symbol, short, total in rows])
    series, fallback = short_volume_series(lines, _master() if master is None else master, _lineage(), RULE_UNIVERSE)
    return {(t, f"{d:%Y-%m-%d}"): (s, v) for d, t, s, v in series.itertuples(index=False)}, fallback


@pytest.mark.parametrize(
    ("symbol", "key", "kind"),
    [
        ("BRK/A", "BRKA", None),
        ("LEN/B", "LENB", None),
        ("BACpB", "BACPRB", "preferred"),
        ("BACPB", "BACPB", None),
        ("BACpY/CL", "BACPRYCL", "preferred"),
        ("BAC/WS/A", "BACWSA", "warrant"),
        ("AAC/U", "AACU", "unit"),
        ("ABCr", "ABCRT", "right"),
        ("ABCw", "ABCWI", "when_issued"),
    ],
)
def test_finra_symbology_is_read_before_upper_casing(symbol, key, kind):
    assert finra_key(symbol) == key
    assert finra_marker_kind(symbol) == kind
    print(f"\nSANITY: FINRA symbology {symbol} -> key {key} (the FTD spelling), marker kind {kind}")


def test_classes_are_summed_by_ratio_and_preferreds_and_unknown_symbols_never_summed():
    got, _ = _series(
        {
            "2020-01-02": [
                ("BRK/A", 2.0, 10.0),
                ("BRK/B", 1000.0, 4000.0),
                ("BACpB", 50.0, 100.0),
                ("BAC", 7000.0, 30000.0),
                ("GOOG", 300.0, 900.0),
                ("GOOGL", 400.0, 1000.0),
                ("LEN/B", 5.0, 20.0),
                ("LEN", 60.0, 200.0),
                ("ZZZZ", 1.0, 2.0),
            ]
        }
    )
    day = "2020-01-02"
    assert got[("BRK-B", day)] == (1000.0 + 1500 * 2.0, 4000.0 + 1500 * 10.0)
    assert got[("BAC", day)] == (7000.0, 30000.0), "the BACpB preferred contributes 0"
    assert got[("GOOGL", day)] == (700.0, 1900.0) and got[("LEN", day)] == (65.0, 220.0)
    assert {t for t, _ in got} == {"BRK-B", "BAC", "GOOGL", "LEN", "DOC", "AAA"}, "ZZZZ maps to no company"
    assert got[("DOC", day)][0] == 0.0 and got[("AAA", day)][0] == 0.0, "PEAK and AAA's predecessor OLDA support absences"
    print("\nSANITY: BRK-B = B + 1,500 x A; GOOGL = A + C; LEN = A + B; BACpB a preferred, never summed; ZZZZ no company")


def test_acquired_constituent_is_never_summed_and_the_seam_moves_the_symbol():
    got, _ = _series({"2023-01-03": [("DOC", 900.0, 3000.0), ("PEAK", 400.0, 1500.0)], "2024-03-05": [("DOC", 800.0, 2000.0)]})
    assert got[("DOC", "2023-01-03")] == (400.0, 1500.0), "Physicians Realty's DOC (acquired) is not Healthpeak's"
    assert got[("DOC", "2024-03-05")] == (800.0, 2000.0), "Healthpeak's DOC after the seam"
    print("\nSANITY: 2023 DOC is the acquired constituent (0 in DOC), PEAK counts; DOC counts after 2024-03-01")


def test_e14_a_same_key_collision_is_unresolved_and_blocks_the_date():
    got, _ = _series({"2020-01-02": [("BACpB", 50.0, 100.0), ("BACPB", 5.0, 10.0), ("BAC", 7.0, 9.0)]})
    assert got[("BAC", "2020-01-02")] == (7.0, 9.0), "BACPB is another key: not BAC's"
    clash, _ = _series({"2020-01-02": [("BACpB", 50.0, 100.0), ("BACPRB", 5.0, 10.0), ("BAC", 7.0, 9.0)]})
    assert clash[("BAC", "2020-01-02")] == (7.0, 9.0), "the two preferred spellings clash; neither is summed, and a preferred blocks nothing"
    print("\nSANITY: BACpB -> BACPRB and BACPB stay different keys; BACpB with a BACPRB on one date is a clash, never summed")


def test_e25_a_symbol_never_in_ftd_maps_only_through_a_window_cik_inside_its_window():
    got, fallback = _series(
        {
            "2020-06-01": [("OLDA", 10.0, 100.0), ("ACQ", 20.0, 200.0), ("BAC", 1.0, 2.0)],
            "2021-01-05": [("OLDA", 11.0, 110.0), ("AAA", 30.0, 300.0), ("AAApA", 1.0, 2.0)],
        }
    )
    assert got[("AAA", "2020-06-01")] == (10.0, 100.0) and got[("AAA", "2021-01-05")] == (30.0, 300.0)
    stamps = {(f"{d:%Y-%m-%d}", i, r) for d, _, i, r, _ in fallback.itertuples(index=False)}
    assert stamps == {("2020-06-01", "S0000000099:OLDA", "canonical_predecessor"), ("2021-01-05", "S0000000100:AAA", "canonical_current")}, stamps
    print("\nSANITY: OLDA inside its window CIK's window and AAA on the open-window CIK map as S securities; OLDA after the")
    print("  window, the event-only ACQ and the AAApA preferred do not")


def test_a_known_key_outside_every_master_interval_leaves_the_date_unknown():
    got, _ = _series({"2015-01-05": [("GOOG", 3.0, 4.0), ("BAC", 1.0, 2.0)]})
    assert ("GOOGL", "2015-01-05") not in got and got[("BAC", "2015-01-05")] == (1.0, 2.0)
    print("\nSANITY: GOOG in 2015 (before Alphabet's class C line) is unresolved: GOOGL's date is unknown, never 0")


def test_f107_on_a_cusip_change_day_the_single_summed_security_takes_the_finra_line():
    old = ("526057999", "LEN", "0000920760", "LEN", "common", 1.0, "acquired_constituent", "2019-12-31", "2020-01-03", "outside_window")
    got, _ = _series({"2020-01-02": [("LEN", 60.0, 200.0)]}, _master([*MASTER_ROWS, old]))
    assert got[("LEN", "2020-01-02")] == (60.0, 200.0)
    both, _ = _series({"2020-01-02": [("LEN", 60.0, 200.0)]}, _master([*MASTER_ROWS, (*old[:6], "canonical_predecessor", *old[7:])]))
    assert ("LEN", "2020-01-02") not in both, "two summed securities on one day stay a conflict"
    print("\nSANITY: LEN under an acquired line and the canonical line on one day: the canonical takes it; two summed lines stay a conflict")


def test_real_stored_lines_reach_the_builder_as_company_series() -> None:
    """Real rows (read-only): each multi-class company's stored FINRA lines sum x ratio on a day, through the loader."""
    store = _store()
    lines = store.load(Tables.universe_cusip, columns=LINE_READ, where={"source": "ftd"}, optional=True)
    lineage = store.load(Tables.universe_cik_symbol, columns=LINEAGE_READ, optional=True)
    columns = store.columns(Tables.sec_short_volume_security) if store.exists(Tables.sec_short_volume_security) else []
    if lines is None or lineage is None or not columns or "ticker" in columns:
        pytest.skip("no universe_cusip / universe_cik_symbol or no as-filed sec_short_volume_security: run identity and short-interest first")
    summed = lines.loc[lines["lineage_role"].isin(["canonical_current", "canonical_predecessor", "secondary_class"])]
    multi = sorted(summed.groupby("canonical_company")["cusip"].nunique().loc[lambda n: n.gt(1)].index)

    short, _ = inputs.load_short_volume(store, multi, lines, lineage)

    assert short is not None and not short.duplicated(["ticker", "date"]).any() and set(short["ticker"]) <= set(multi)
    raw = store.load(Tables.sec_short_volume_security, columns=["date", "source_symbol", "short_volume"])
    assert raw is not None
    raw = raw.assign(date=pd.to_datetime(raw["date"]), key=raw["source_symbol"].str.replace("/", "", regex=False))
    day = raw["date"].max()
    own = raw.loc[raw["date"].eq(day)].merge(summed.assign(key=summed["source_symbol"]), on="key")
    own = own.loc[own["date"].ge(pd.to_datetime(own["valid_from"])) & (own["date"].lt(pd.to_datetime(own["valid_to"])) | own["valid_to"].isna())]
    expected = (own["short_volume"] * own["conversion_ratio"]).groupby(own["canonical_company"]).sum()
    on_day = short.loc[short["date"].eq(day)].set_index("ticker")["short_volume"]
    checked = [t for t in expected.index if t in on_day.index]
    assert checked and np.allclose(on_day.reindex(checked), expected.reindex(checked), rtol=1e-9)
    print(f"\nSANITY (real rows): {len(short)} series rows for {multi}; on {day.date()} the builder frame holds")
    print(f"  {on_day.reindex(checked).round(0).to_dict()} = sum(short volume x ratio) of the stored FINRA lines.")
