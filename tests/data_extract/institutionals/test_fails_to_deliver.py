"""SEC Fails-to-Deliver: parse + semi-monthly period logic, the full-rebuild gates, the FTD capture (`identity --only capture-evidence`) and
the FTD feature (fails/volume, publication-lagged so it's leak-free)."""

from __future__ import annotations

import logging
import re
import shutil
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pytest

from src.data_extract.utils.common.bulk_cache import read_zip_text
from src.data_extract.utils.common.security_lines import SecurityManual
from src.data_extract.utils.identity import cusip_history as sm
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from tests.canonical_schema import create_canonical
from tests.conftest import DATA, make_frames
from tests.data_aggregate.test_short_flow_features import build_short_flow_feature_panel
from tests.support.cusip_build import write_ftd_zips
from tests.support.store_identity import store_identity
from tests.support.universe import seed_universe


@pytest.fixture
def sqlite_store(sqlite_store):
    create_canonical(
        sqlite_store.engine,
        [
            Tables.universe_cik_symbol,
            Tables.symbol_tenure,
            Tables.universe_cusip,
            Tables.sec_fails_to_deliver_security,
            Tables.sec_fails_to_deliver,
            Tables.sec_short_volume_security,
            Tables.short_interest,
        ],
    )
    return sqlite_store


def _context(sqlite_store, tmp_path, universe: tuple[str, ...] = ("AAPL",)) -> Any:
    return SimpleNamespace(
        store=sqlite_store,
        log=logging.getLogger("test.ftd"),
        paths={"DATA_STORE": tmp_path},
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(fails_deliver="sec_fails_to_deliver"))),
        universe=seed_universe(sqlite_store, universe),
        _identity_resolver=store_identity(sqlite_store),
    )


def test_periods_semimonthly_bounded():
    ps = ftd.ftd_periods(1, today=pd.Timestamp("2024-03-10"))
    assert ps[:2] == ["202301a", "202301b"]  # a=1-15, b=16-end
    assert ps[-2:] == ["202403a", "202403b"]  # up to the current month only
    assert "202312b" in ps
    full = ftd.ftd_periods(50, today=pd.Timestamp("2024-03-10"))
    assert full[:2] == ["200907a", "200907b"]
    assert "200906b" not in full
    # 15y reaches into the legacy era -> full history, not just 2017+
    assert "201001a" in ftd.ftd_periods(15, today=pd.Timestamp("2024-03-10"))
    print("\n=== SANITY: FTD semi-monthly periods ===")
    print(
        f"  years_history=1 @2024-03 -> {len(ps)} files 202301a..202403b; 15y reaches back to 2010 "
        "(legacy era); full history starts at 200907a and skips unavailable January-June 2009 files. Validated."
    )


def test_period_urls_legacy_vs_modern_boundary():
    """<= 2017-06a -> FOIA legacy path (first); >= 2017-06b -> current path; the other
    path is the fallback. The switch is the 2nd half of June 2017."""
    _foia = "frequently-requested-foia"
    leg = ftd._period_urls("201301a")
    assert _foia in leg[0] and "cnsfails201301a.zip" in leg[0] and _foia not in leg[1]
    assert _foia in ftd._period_urls("201706a")[0]  # boundary: last legacy period
    assert _foia not in ftd._period_urls("201706b")[0]  # boundary: first modern period
    mod = ftd._period_urls("202401a")
    assert _foia not in mod[0] and "fails-deliver-data/cnsfails202401a" in mod[0] and _foia in mod[1]
    print("\n=== SANITY: FTD legacy/modern URL selection ===")
    print("  <=201706a -> FOIA legacy path first (modern fallback); >=201706b -> current path (legacy fallback). Boundary at 2017-06b. Validated.")


def test_period_urls_end_with_the_other_root():
    urls = ftd._period_urls("202610a")
    other = "https://www.sec.gov/files/data/other/fails-deliver-data/cnsfails202610a.zip"
    assert urls[-1] == other and urls[0] == ftd.SEC_FTD_URL_TEMPLATE.format(period="202610a")
    print(f"SANITY: 202610a tries {len(urls)} SEC roots, the current path first and the /other/ root last.")


def test_parse_ftd_lines_strips_the_trailer_and_checks_its_count():
    body = f"{_HEADER}20240102|037833100|AAPL|10|APPLE INC|10\n20240103|037833100|AAPL|5|APPLE INC|.\n"
    lines = ftd._parse_ftd_lines(body + "Trailer record count 2\nTrailer total quantity of shares 15\n")
    assert lines["fails_quantity"].tolist() == [10.0, 5.0]
    with pytest.raises(ValueError, match="trailer"):
        ftd._parse_ftd_lines(body + "Trailer record count 3\nTrailer total quantity of shares 15\n")
    print("SANITY: the two SEC trailer lines are dropped; a record count that disagrees with the data lines refuses the file.")


def test_parse_ftd_lines_keeps_a_pipe_inside_the_description():
    # cnsfails202104b line 52725, as filed by the SEC: the description holds a '|'.
    lines = ftd._parse_ftd_lines(f"{_HEADER}20210428|23344K110|DMYQWS|351|DMY TECHNOLOGY GROUP INC IV | |1.49\n")
    row = lines.iloc[0]
    assert (row["cusip"], row["fails_quantity"], row["price"], row["description"]) == ("23344K110", 351.0, 1.49, "DMY TECHNOLOGY GROUP INC IV |")
    print(f"SANITY: a 7-field SEC line keeps quantity {row['fails_quantity']:.0f} and price {row['price']}; the extra '|' stays in the description.")


def test_real_202609a_parses_its_trailer_count_on_a_copy(tmp_path):
    source = DATA / "sec_fails_to_deliver" / "cnsfails202609a.zip"
    if not source.exists():
        pytest.skip("cached FTD 202609a zip absent")
    cache = tmp_path / "sec_fails_to_deliver"
    cache.mkdir()
    path = cache / source.name
    shutil.copyfile(source, path)
    raw = read_zip_text(path)
    assert raw is not None
    trailer = int(re.search(r"Trailer record count (\d+)", raw).group(1))  # type: ignore[union-attr]
    parsed = ftd.read_ftd_period(SimpleNamespace(), cache, "202609a", download=False)  # type: ignore[arg-type]
    assert parsed is not None and len(parsed) == trailer and path.exists()
    print(f"SANITY: real FTD 202609a (copy) parses {len(parsed):,} lines = its trailer record count {trailer:,}; the cached file is kept.")


def test_invalid_ftd_content_keeps_the_cached_file(tmp_path):
    cache = tmp_path / "sec_fails_to_deliver"
    cache.mkdir()
    path = cache / "cnsfails202607a.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("cnsfails202607a.txt", f"{_HEADER}20260701|526057104|LEN|bad|LEN|10\n")
    assert ftd.read_ftd_period(SimpleNamespace(), cache, "202607a", download=False) is None  # type: ignore[arg-type]
    assert path.exists()
    print("SANITY: an FTD file with invalid content is skipped and left in the cache; only a corrupt archive is deleted.")


def test_parse_ftd_math_and_na_price():
    raw = (
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
        "20240102|X|AAPL|1000|APPLE INC|180.50\n"
        "20240102|Y|MSFT|500|MICROSOFT|.\n"  # PRICE '.' = N/A
        "20240102|B|BRK/B|1|BERKSHIRE|300.00\n"
        "20240103|Z|AAPL|200|APPLE INC|181.00\n"
    )
    df = ftd._parse_ftd_lines(raw)
    a = df[(df["source_symbol"] == "AAPL") & (df["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert a["fails_quantity"] == 1000.0 and abs(a["fails_value"] - 180_500.0) < 1e-6
    m = df[df["source_symbol"] == "MSFT"].iloc[0]
    assert m["fails_quantity"] == 500.0 and pd.isna(m["price"]) and pd.isna(m["fails_value"])  # '.' price -> NULL dollars
    assert set(df["source_symbol"]) == {"AAPL", "MSFT", "BRK/B"} and len(df) == 4
    print("\n=== SANITY: FTD parse ===")
    print("  AAPL 1000@180.5 -> fails_value $180.5k; MSFT price '.' -> NULL; BRK/B kept as filed. Validated.")


def test_parse_ftd_matches_real_legacy_and_modern_samples():
    """Real rows pulled from SEC's actual cnsfails200907a.zip (legacy path) and
    cnsfails202401a.zip (current path): both eras share the identical column layout
    and units (verified live during this refactor), so one parser handles both --
    unlike 13F, FTD has no $thousands-vs-$ones split to guard against."""
    raw = (
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
        "20090701|037833100|AAPL|32975|APPLE INC;COM NPV|142.43\n"  # legacy era
        "20240102|037833100|AAPL|516|APPLE INC;COM NPV|192.53\n"
    )  # modern era
    df = ftd._parse_ftd_lines(raw)
    legacy = df[df["date"] == pd.Timestamp("2009-07-01")].iloc[0]
    modern = df[df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert legacy["fails_quantity"] == 32975.0 and abs(legacy["fails_value"] - 4_696_629.25) < 1e-2
    assert modern["fails_quantity"] == 516.0 and abs(modern["fails_value"] - 99_345.48) < 1e-2
    print("\n=== SANITY CHECK: FTD real legacy vs. modern sample ===")
    print(
        f"  2009-07-01 AAPL 32975@142.43 -> ${legacy['fails_value']:,.2f}; "
        f"2024-01-02 AAPL 516@192.53 -> ${modern['fails_value']:,.2f}. Same units both eras. Validated."
    )


def _master(rows: list[tuple[str, str, str, str, str | None]]) -> pd.DataFrame:
    """(cusip, company, symbol, valid_from, valid_to) canonical common lines, ratio 1."""
    frame = pd.DataFrame(rows, columns=["cusip", "canonical_company", "source_symbol", "valid_from", "valid_to"])
    frame = frame.assign(
        security_id="C" + frame["cusip"],
        issuer_cik="0000000001",
        source=sm.SOURCE_FTD,
        market_symbol=frame["source_symbol"],
        exchange=None,
        security_class="common",
        conversion_ratio=1.0,
        lineage_role=sm.CANONICAL_CURRENT,
        lineage_reason="fixture",
        source_accession=None,
        evidence="fixture",
        n_observations=1,
        scope_changed_at=pd.Timestamp("2026-10-04"),
    )
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame[list(sm.TABLE_COLUMNS)]


_REUSE_MASTER = [
    ("30303M102", "META", "FB", "2012-05-18", "2022-06-09"),
    ("30303M102", "META", "META", "2022-06-09", None),
    ("G47791101", "TT", "IR", "2009-01-01", "2020-03-02"),
    ("45687V106", "IR", "IR", "2020-03-02", None),
    ("G96629103", "WTW", "WLTW", "2016-01-05", None),
]


def test_full_rebuild_keys_on_cusip_excludes_a_reused_symbol_and_is_repeatable(sqlite_store, monkeypatch, tmp_path):
    """FB and META are one CUSIP (one META key); IR before 2020 is the TT entity's CUSIP; the Weight Watchers
    2015 WTW line is a CUSIP the master does not hold, so it is never stored under Willis Towers Watson."""
    ctx = _context(sqlite_store, tmp_path, ("META", "TT", "IR", "WTW"))
    sqlite_store.replace(Tables.universe_cusip, _master(_REUSE_MASTER))
    raw_by_period = {
        "201501a": (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20150102|30303M102|FB|100|FACEBOOK INC CL A|10\n"
            "20150102|G47791101|IR|50|INGERSOLL RAND PLC|10\n"
            "20150102|948626106|WTW|70|WEIGHT WATCHERS|10\n"
        ),
        "202207b": "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20220720|30303M102|META|200|META PLATFORMS INC|10\n",
    }
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: set(raw_by_period))
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: raw_by_period[path.stem.removeprefix("cnsfails")])
    replace_calls: list[str] = []
    original_replace = ftd.replace_fails_source

    def _replace(
        store: DataStore, frames: dict[Table, pd.DataFrame], scopes: dict[Table, dict[str, object] | None], *args: Any, **kwargs: Any
    ) -> dict[Table, int]:
        replace_calls.extend(table.name for table in frames)
        return original_replace(store, frames, scopes, *args, **kwargs)

    monkeypatch.setattr(ftd, "replace_fails_source", _replace)
    universe = ["META", "TT", "IR", "WTW"]
    ftd.fetch_fails_to_deliver(ctx, universe, full=True)
    first = sqlite_store.load(Tables.sec_fails_to_deliver).sort_values(["ticker", "date"]).reset_index(drop=True)
    ftd.fetch_fails_to_deliver(ctx, universe, full=True)
    second = sqlite_store.load(Tables.sec_fails_to_deliver).sort_values(["ticker", "date"]).reset_index(drop=True)

    assert replace_calls == [Tables.sec_fails_to_deliver_security.name, Tables.sec_fails_to_deliver.name] * 4
    positive = first.loc[first["fails_quantity"].gt(0)]
    assert list(zip(positive["ticker"], positive["fails_quantity"], strict=True)) == [("META", 100.0), ("META", 200.0), ("TT", 50.0)]
    assert set(first.loc[first["fails_quantity"].eq(0), "ticker"]) == {"IR", "WTW"}
    assert "948626106" not in set(sqlite_store.load(Tables.sec_fails_to_deliver_security)["cusip"])
    pd.testing.assert_frame_equal(first.drop(columns="source_checked_at"), second.drop(columns="source_checked_at"))
    print("\n=== SANITY CHECK: FTD CUSIP full rebuild ===")
    print("  FB and META one CUSIP -> META; pre-2020 IR -> TT by its CUSIP; the Weight Watchers WTW line never stored")
    print("  repeated economics match; each source period replaces its raw/derived pair atomically and actual served dates support mapped zeros")


def test_a_scoped_full_run_upserts_its_company_and_never_replaces_the_tables(sqlite_store, monkeypatch, tmp_path):
    """`zips-ingest --only ftd -t META -F` re-reads the cache for META's CUSIPs only and upserts; TT's stored row survives."""
    ctx = _context(sqlite_store, tmp_path, ("META", "TT"))
    sqlite_store.replace(Tables.universe_cusip, _master(_REUSE_MASTER))
    sqlite_store.save(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {"ticker": ["TT"], "date": pd.to_datetime(["2015-01-02"]), "fails_quantity": [7.0], "fails_value": [7.0], "period": ["201501a"]}
        ),
    )
    raw = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20220720|30303M102|META|200|META PLATFORMS INC|10\n20220720|G47791101|IR|9|X|1\n"
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: {"202207b"})
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: raw)
    monkeypatch.setattr(sqlite_store, "replace", lambda *a, **k: pytest.fail("a scoped -F must never replace a table"))

    ftd.fetch_fails_to_deliver(ctx, ["META"], full=True)

    stored = sqlite_store.load(Tables.sec_fails_to_deliver).set_index("ticker")
    assert stored.loc["TT", "fails_quantity"] == 7.0 and stored.loc["META", "fails_quantity"] == 200.0
    assert set(sqlite_store.load(Tables.sec_fails_to_deliver_security)["cusip"]) == {"30303M102"}
    print("\n=== SANITY CHECK: FTD -t META -F ===")
    print("  META's CUSIP re-read and upserted (200); TT's stored row kept, the IR/TT line not read; no table replaced. Validated.")


def test_a_scoped_ingest_resolves_the_share_basis_of_its_companies_only(sqlite_store, monkeypatch, tmp_path):
    """X96 (X77's class on the ingest path): `zips-ingest --only ftd -t META -F` resolves the share basis over META's
    master rows only, so another company's share policy (HBAN on the fixture) never blocks it."""
    ctx = _context(sqlite_store, tmp_path, ("META", "TT"))
    sqlite_store.replace(Tables.universe_cusip, _master(_REUSE_MASTER))
    raw = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20220720|30303M102|META|200|META PLATFORMS INC|10\n"
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: {"202207b"})
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: raw)
    seen: list[set[str]] = []
    real = ftd.load_fails_basis
    monkeypatch.setattr(
        ftd, "load_fails_basis", lambda context, master: seen.append(set(master["canonical_company"].dropna())) or real(context, master)
    )

    ftd.fetch_fails_to_deliver(ctx, ["META"], full=True)

    assert seen == [{"META"}], seen
    print("\n=== SANITY CHECK: FTD -t META -F share basis (X96) ===")
    print(f"  the basis was resolved over the master rows of {sorted(seen[0])} only; TT's policy is never read. Validated.")


def test_full_rebuild_unreadable_cached_period_aborts_before_replace(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    sqlite_store.replace(Tables.universe_cusip, _master([("037833100", "AAPL", "AAPL", "2009-01-01", None)]))
    seeded = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": pd.to_datetime(["2024-01-02"]),
            "fails_quantity": [1.0],
            "fails_value": [2.0],
            "period": ["202401a"],
        }
    )
    sqlite_store.replace(Tables.sec_fails_to_deliver, seeded)
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: {"202401a"})
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: None)

    with pytest.raises(ValueError, match="cannot read cached period 202401a"):
        ftd.fetch_fails_to_deliver(ctx, ["AAPL"], full=True)
    stored = sqlite_store.load(Tables.sec_fails_to_deliver).reset_index(drop=True)
    assert len(stored) == 1 and stored.iloc[0]["ticker"] == "AAPL"
    assert pd.Timestamp(stored.iloc[0]["date"]) == pd.Timestamp("2024-01-02")
    assert stored.iloc[0]["fails_quantity"] == 1.0

    print("\n=== SANITY CHECK: FTD replacement gate ===")
    print("  unreadable required cache period raises before replace; stored row survives")
    print("  OK: a partial cache can never become a complete-looking replacement")


def test_full_rebuild_refuses_when_a_stored_period_left_the_cache(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    sqlite_store.replace(Tables.universe_cusip, _master([("037833100", "AAPL", "AAPL", "2009-01-01", None)]))
    lines = ftd._parse_ftd_lines("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20231215|037833100|AAPL|5|APPLE INC|190\n")
    unstamped = lines.assign(period="202312a", security_id=None, ticker=None, lineage_role=None, security_class=None)
    sqlite_store.save(Tables.sec_fails_to_deliver_security, unstamped[list(ftd.SECURITY_COLUMNS)])
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: {"202401a"})
    monkeypatch.setattr(ftd, "read_zip_text", lambda *a, **k: pytest.fail("no zip is read before the gate"))

    with pytest.raises(FileNotFoundError, match="202312a"):
        ftd.fetch_fails_to_deliver(ctx, ["AAPL"], full=True)
    assert sqlite_store.row_count(Tables.sec_fails_to_deliver_security) == 1
    print("\n=== SANITY CHECK: FTD rebuild reproduces every stored period ===")
    print("  a stored period missing from the cache aborts the rebuild before any read or write")


def test_ftd_download_first_successful_http_response_caches_the_zip_once(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    requested: list[str] = []

    class _Response:
        status_code = 200

        @staticmethod
        def iter_content(chunk_size):
            yield b"downloaded archive"

    def _get(url, **kwargs):
        requested.append(url)
        return _Response()

    ctx.sec_session = SimpleNamespace(get=_get)
    monkeypatch.setattr(ftd, "ftd_periods", lambda *a, **k: ["202609a"])

    assert ftd.download_fails_to_deliver(ctx, years_history=1) == 1
    assert ftd.download_fails_to_deliver(ctx, years_history=1) == 1
    assert len(requested) == 1, "a cached ZIP is never downloaded again"
    assert (ftd.cache_dir(ctx, "sec_fails_to_deliver") / "cnsfails202609a.zip").is_file()
    assert sqlite_store.row_count(Tables.sec_fails_to_deliver_security) == 0, "the download writes no table"
    print("\n=== SANITY CHECK: first successful FTD HTTP response ===")
    print("  SEC HTTP 200 cached the period ZIP once; the rerun found it cached and sent no request; no table written.")


def test_ftd_feature_ranks_high_fails_and_is_leak_free():
    idx = pd.DatetimeIndex(pd.bdate_range("2023-12-01", "2024-02-14"))
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-01-15"))
    fails = pd.concat(
        [
            pd.DataFrame({"date": days, "ticker": "HI", "fails_quantity": 1e5}),
            pd.DataFrame({"date": days, "ticker": "MID", "fails_quantity": 1e4}),
            pd.DataFrame({"date": days, "ticker": "LO", "fails_quantity": 1e2}),
        ],
        ignore_index=True,
    )
    fails["period"] = "202401a"
    volume = pd.DataFrame({t: 1e6 for t in ("HI", "MID", "LO")}, index=idx)
    peers = {"HI": {"MID": 1.0, "LO": 1.0}, "MID": {"HI": 1.0, "LO": 1.0}, "LO": {"HI": 1.0, "MID": 1.0}}

    panel = build_short_flow_feature_panel(make_frames(idx, peers, volume=volume), None, fails_history=fails)
    assert "f_ic_ftd_to_adv20" in panel.columns

    # The entire January-a ZIP becomes visible under the estimated 21-calendar-day policy.
    d = pd.Timestamp("2024-02-05")
    row = panel[panel["date"] == d].set_index("ticker")
    assert row["f_ic_ftd_to_adv20"]["HI"] > row["f_ic_ftd_to_adv20"]["LO"]
    assert row["f_ic_ftd_to_adv20"]["HI"] > row["f_ic_ftd_to_adv20"]["LO"]

    # leak-free: no row from the ZIP is visible before its shared publication date.
    early = panel[panel["date"] == pd.Timestamp("2024-02-02")]
    assert early.empty or early["f_ic_ftd_to_adv20"].isna().all()

    print("\n=== SANITY: FTD feature (fails/ADV20, ZIP-publication dated) ===")
    print(
        f"  HI fails/ADV20 {row['f_ic_ftd_to_adv20']['HI']:.4f} ranks above LO "
        f"{row['f_ic_ftd_to_adv20']['LO']:.6f} on estimated Feb 5; "
        f"the Feb 2 prefix is absent. Validated."
    )


# --------------------------------------------------------------------------- Q2a: FTD evidence of the CUSIP history

_HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"


def test_parse_ftd_lines_keeps_every_source_line_raw():
    raw = _HEADER + (
        "20170906|000111AAA|ABC|100|ABC CORP|10.00\n"
        "20170907|000111AAA|ABC|200|ABC CORP|.\n"
        "20240529|000111BBB|ABC/PR|5|ABC CORP PFD|25.5\n"
        "20240529|000111BBB|ABC/PR|5|ABC CORP PFD|25.5\n"
    )
    lines = ftd._parse_ftd_lines(raw)
    assert lines.columns.tolist() == list(ftd.SECURITY_COLUMNS[:8])
    assert len(lines) == 3, "an exact duplicate source line is one row; nothing else is summed"
    first = lines.iloc[0]
    assert (first["source_symbol"], first["description"], first["price"], first["fails_quantity"], first["fails_value"]) == (
        "ABC",
        "ABC CORP",
        10.0,
        100.0,
        1000.0,
    )
    assert lines["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2017-09-01", "2017-09-05", "2024-05-28"]
    assert pd.isna(lines.iloc[1]["price"]) and pd.isna(lines.iloc[1]["fails_value"])
    assert lines.iloc[2]["source_symbol"] == "ABC/PR"
    print("\n=== SANITY CHECK: raw FTD lines ===")
    print("  description and PRICE kept, '.' -> NULL dollars, symbol as filed, trade date by the T+3/T+2/T+1 cycle; no summing")


def _scope_lineage() -> pd.DataFrame:
    rows = [
        ("E0000000777", "ABC", "0000000777", "cik_window", "", "1900-01-01", None, "single_source", "roster"),
        ("E0000000777", "ABC", "0000000777", "symbol", "ABC", "2006-01-04", None, "corroborated", "form345,roster"),
    ]
    columns = ["company_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame


def test_ftd_evidence_reads_symbols_then_the_voted_cusip6_from_cache_and_stores_nothing(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    cache = write_ftd_zips(
        tmp_path / "ftd",
        {
            "201501a": _HEADER + "20150105|000111CCC|ABCWS|7|ABC CORP WT|1.0\n20150105|999999ZZZ|ZZZ|9|OTHER CO|3.0\n",
            "201501b": _HEADER + "20150120|000111AAA|ABC|100|ABC CORP|10.0\n20150120|000111BBB|ABCPRA|5|ABC CORP PFD|25.0\n",
        },
    )
    calls: list[str] = []
    read = ftd.read_zip_text
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: calls.append(path.stem.removeprefix("cnsfails")) or read(path, log=log))
    scope = sm.ftd_scope(_scope_lineage(), {}, SecurityManual.empty())
    assert scope.symbols == frozenset({"ABC"}) and not scope.prefixes

    lines = sm.read_ftd_evidence(ctx, cache, scope).sort_values("cusip").reset_index(drop=True)
    assert lines["cusip"].tolist() == ["000111AAA", "000111BBB", "000111CCC"], lines
    assert lines.columns.tolist() == list(sm.OBSERVATION_COLUMNS)
    assert not lines.duplicated(["cusip", "date"]).any()
    assert calls.count("201501a") == 2 and calls.count("201501b") == 2, "both periods hold the voted CUSIP-6 000111: each is read again"
    assert sqlite_store.row_count(Tables.sec_fails_to_deliver_security) == 0 and sqlite_store.row_count(Tables.sec_fails_to_deliver) == 0
    print("\n=== SANITY CHECK: FTD evidence of the CUSIP history ===")
    print("  pass 1 keeps the lineage symbol ABC; its vote adds CUSIP-6 000111, so the warrant and preferred lines are re-read from cache")
    print(f"  {len(lines)} line(s), observation columns only; unrelated ZZZ never kept; nothing written to either FTD table")
