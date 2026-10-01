"""SEC Fails-to-Deliver: parse + semi-monthly period logic + the FTD feature
(fails/volume, publication-lagged so it's leak-free)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals.short_flow_features import build_short_flow_feature_panel
from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.institutionals import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from tests.conftest import make_frames


def _identity() -> Identity:
    lineage = pd.DataFrame(
        [
            {"cik": "0000000001", "entity_id": "E_META", "source": "roster"},
            {"cik": "0000000002", "entity_id": "E_TT", "source": "roster"},
            {"cik": "0000000003", "entity_id": "E_IR", "source": "roster"},
            {"cik": "0000000004", "entity_id": "E_WTW", "source": "roster"},
            {"cik": "0000000005", "entity_id": "E_MSFT", "source": "roster"},
            {"cik": "0000000006", "entity_id": "E_AAPL", "source": "roster"},
        ]
    )
    tenure = pd.DataFrame(
        [
            ("FB", "0000000001", "2012-01-01", "2022-06-01", 100),
            ("META", "0000000001", "2022-06-01", None, 100),
            ("IR", "0000000002", "2009-01-01", "2020-03-01", 100),
            ("IR", "0000000003", "2020-03-05", None, 100),
            ("TT", "0000000002", "2020-03-01", None, 100),
            ("WTW", "0000000099", "2009-01-01", "2019-04-18", 100),
            ("WTW", "0000000004", "2019-04-18", None, 100),
            ("MSFT", "0000000005", "2009-01-01", None, 100),
            ("AAPL", "0000000006", "2009-01-01", None, 100),
        ],
        columns=["symbol", "issuer_cik", "valid_from", "valid_to", "n_filings"],
    )
    tenure["source"] = "form345"
    tenure["evidence"] = "test"
    roster = pd.DataFrame(
        [
            ("META", "0000000001"),
            ("TT", "0000000002"),
            ("IR", "0000000003"),
            ("WTW", "0000000004"),
            ("MSFT", "0000000005"),
            ("AAPL", "0000000006"),
        ],
        columns=["ticker", "cik"],
    )
    return build_identity(lineage, tenure, roster)


def _context(sqlite_store, tmp_path) -> Any:
    return SimpleNamespace(
        store=sqlite_store,
        log=logging.getLogger("test.ftd"),
        paths={"DATA_STORE": tmp_path},
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(fails_deliver="sec_fails_to_deliver"))),
    )


def test_periods_semimonthly_bounded():
    ps = ftd._periods(1, today=pd.Timestamp("2024-03-10"))
    assert ps[:2] == ["202301a", "202301b"]  # a=1-15, b=16-end
    assert ps[-2:] == ["202403a", "202403b"]  # up to the current month only
    assert "202312b" in ps
    assert all(int(p[:4]) >= ftd.SEC_FTD_FIRST_YEAR for p in ftd._periods(50, today=pd.Timestamp("2024-03-10")))
    # 15y reaches into the legacy era (FIRST_YEAR=2009) -> full history, not just 2017+
    assert "201001a" in ftd._periods(15, today=pd.Timestamp("2024-03-10"))
    print("\n=== SANITY: FTD semi-monthly periods ===")
    print(
        f"  years_history=1 @2024-03 -> {len(ps)} files 202301a..202403b; 15y reaches back to 2010 "
        f"(legacy era, FIRST_YEAR={ftd.SEC_FTD_FIRST_YEAR}). Validated."
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


def test_parse_ftd_math_and_na_price():
    raw = (
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
        "20240102|X|AAPL|1000|APPLE INC|180.50\n"
        "20240102|Y|MSFT|500|MICROSOFT|.\n"  # PRICE '.' = N/A
        "20240102|B|BRK/B|1|BERKSHIRE|300.00\n"
        "20240103|Z|AAPL|200|APPLE INC|181.00\n"
    )
    df = ftd._parse_ftd(raw)
    a = df[(df["source_symbol"] == "AAPL") & (df["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert a["fails_quantity"] == 1000.0 and abs(a["fails_value"] - 180_500.0) < 1e-6
    m = df[df["source_symbol"] == "MSFT"].iloc[0]
    assert m["fails_quantity"] == 500.0 and pd.isna(m["fails_value"])  # '.' price -> value NaN
    assert set(df["source_symbol"]) == {"AAPL", "MSFT", "BRK-B"} and len(df) == 4
    print("\n=== SANITY: FTD parse ===")
    print("  AAPL 1000@180.5 -> fails_value $180.5k; MSFT price '.' -> NaN; BRK/B -> BRK-B. Validated.")


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
    df = ftd._parse_ftd(raw)
    legacy = df[df["date"] == pd.Timestamp("2009-07-01")].iloc[0]
    modern = df[df["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert legacy["fails_quantity"] == 32975.0 and abs(legacy["fails_value"] - 4_696_629.25) < 1e-2
    assert modern["fails_quantity"] == 516.0 and abs(modern["fails_value"] - 99_345.48) < 1e-2
    print("\n=== SANITY CHECK: FTD real legacy vs. modern sample ===")
    print(
        f"  2009-07-01 AAPL 32975@142.43 -> ${legacy['fails_value']:,.2f}; "
        f"2024-01-02 AAPL 516@192.53 -> ${modern['fails_value']:,.2f}. Same units both eras. Validated."
    )


def test_fetch_skips_done_periods_and_upserts_without_duplicating(sqlite_store, monkeypatch, tmp_path):
    """Resume contract: an already-ingested period is never re-fetched while the universe
    is stable; a universe change re-parses cached periods, but the upsert on (ticker, date)
    must not duplicate rows already stored."""
    # `cache_dir(context, context.config.local.paths.fails_deliver)` -- the fetcher reads its
    # cache subdirectory out of config, so the double has to carry it (value from
    # configs/paths.yml).
    ctx = _context(sqlite_store, tmp_path)
    identity = _identity()

    def _raw_for(path) -> str:
        period = path.stem.removeprefix("cnsfails")
        yyyymm, day = period[:6], ("01" if period.endswith("a") else "16")
        return (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            f"{yyyymm}{day}|037833100|AAPL|100|APPLE INC|190.00\n"
            f"{yyyymm}{day}|055555555|MSFT|50|MICROSOFT|300.00\n"
        )

    requested: list[str] = []

    def _fake_ensure_zip(context, path, urls, *, label, timeout, log, on_download):
        requested.append(label)
        return path

    monkeypatch.setattr(ftd, "ensure_zip", _fake_ensure_zip)
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: _raw_for(path))
    monkeypatch.setattr(ftd, "_periods", lambda years_history, today=None: ["202401a", "202401b"])
    monkeypatch.setattr(ftd, "record_run", lambda *a, **k: None)

    # period 202401a already ingested, universe already converged on AAPL
    sqlite_store.replace(
        "sec_fails_to_deliver",
        pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": pd.to_datetime(["2024-01-01"]),
                "fails_quantity": [100.0],
                "fails_value": [19000.0],
                "period": ["202401a"],
            }
        ),
    )
    cache = ftd.cache_dir(ctx, "sec_fails_to_deliver")
    ftd.save_processed_universe(cache, ftd.Tables.sec_fails_to_deliver, set(identity.candidate_symbols(frozenset({"AAPL"}))) | {ftd._POLICY_MARKER})

    # 1) stable universe -> the already-done period is skipped, only the new one is fetched
    saved = ftd.fetch_fails_to_deliver(ctx, tickers=["AAPL"], years_history=1, identity=identity)
    assert requested == ["FTD 202401b"], f"already-done period was re-fetched: {requested}"
    assert saved == 1
    stored = sqlite_store.load("sec_fails_to_deliver")
    assert len(stored) == 2  # seeded 202401a row + new 202401b row

    # 2) an identity-policy revision replays both cached periods even with a stable universe
    requested.clear()
    monkeypatch.setattr(ftd, "_POLICY_MARKER", "__point_in_time_symbol_identity_v3__")
    assert ftd.fetch_fails_to_deliver(ctx, tickers=["AAPL"], years_history=1, identity=identity) == 2
    assert requested == ["FTD 202401a", "FTD 202401b"]

    # 3) universe grows (MSFT) -> both cached periods are re-parsed, but the upsert on
    #    (ticker, date) must not duplicate the 202401a/202401b AAPL rows already stored
    requested.clear()
    saved2 = ftd.fetch_fails_to_deliver(ctx, tickers=["AAPL", "MSFT"], years_history=1, identity=identity)
    assert requested == ["FTD 202401a", "FTD 202401b"]
    stored2 = sqlite_store.load("sec_fails_to_deliver")
    assert len(stored2) == 4  # AAPL+MSFT x 2 periods, no duplicates
    assert saved2 == 4

    print("\n=== SANITY CHECK: FTD resume + universe-growth reparse ===")
    print(
        f"  stable universe -> 202401a skipped; policy revision and universe growth each reparse both periods; "
        f"{len(stored2)} distinct rows stored "
        "(upsert, no duplicates). Validated."
    )


def test_full_rebuild_relabels_reuse_excludes_prior_holder_and_aggregates(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    identity = _identity()
    raw_by_period = {
        "201501a": (
            "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
            "20150102|A|FB|100|FACEBOOK|10\n"
            "20150102|B|FB|200|META ALIAS|10\n"
            "20150102|C|IR|50|INGERSOLL RAND PLC|10\n"
            "20150102|D|WTW|70|WEIGHT WATCHERS|10\n"
        ),
    }
    monkeypatch.setattr(ftd, "_periods", lambda *a, **k: ["201501a"])
    monkeypatch.setattr(ftd, "_cached_periods", lambda cache: {"201501a"})
    monkeypatch.setattr(ftd, "ensure_zip", lambda context, path, urls, **kwargs: path)
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: raw_by_period[path.stem.removeprefix("cnsfails")])
    monkeypatch.setattr(ftd, "record_run", lambda *a, **k: None)

    sqlite_store.replace(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {
                "ticker": ["WTW"],
                "date": pd.to_datetime(["2015-01-02"]),
                "fails_quantity": [999.0],
                "fails_value": [999.0],
                "period": ["201501a"],
            }
        ),
    )
    replace_calls = 0
    original_replace = sqlite_store.replace

    def _replace(table, frame, *args, **kwargs):
        nonlocal replace_calls
        replace_calls += 1
        return original_replace(table, frame, *args, **kwargs)

    monkeypatch.setattr(sqlite_store, "replace", _replace)
    universe = ["META", "TT", "IR", "WTW"]
    ftd.fetch_fails_to_deliver(ctx, universe, full=True, identity=identity)
    first = sqlite_store.load(Tables.sec_fails_to_deliver).sort_values("ticker").reset_index(drop=True)
    ftd.fetch_fails_to_deliver(ctx, universe, full=True, identity=identity)
    second = sqlite_store.load(Tables.sec_fails_to_deliver).sort_values("ticker").reset_index(drop=True)

    assert replace_calls == 2
    assert set(first["ticker"]) == {"META", "TT"}
    assert first.set_index("ticker").loc["META", "fails_quantity"] == 300.0
    assert first.set_index("ticker").loc["TT", "fails_quantity"] == 50.0
    assert not first.duplicated(["ticker", "date"]).any()
    pd.testing.assert_frame_equal(first, second)

    print("\n=== SANITY CHECK: FTD point-in-time full rebuild ===")
    print("  FB+META -> one META key (quantity 300); historical IR -> TT")
    print("  Weight Watchers under WTW excluded; second full run is byte-equivalent")
    print("  OK: renamed rows recovered, reused rows moved/removed, unique PK preserved")


def test_full_rebuild_unreadable_cached_period_aborts_before_replace(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    identity = _identity()
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
    monkeypatch.setattr(ftd, "_periods", lambda *a, **k: ["202401a"])
    monkeypatch.setattr(ftd, "_cached_periods", lambda cache: {"202401a"})
    monkeypatch.setattr(ftd, "ensure_zip", lambda context, path, urls, **kwargs: path)
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: None)

    with pytest.raises(ValueError, match="cannot read cached period 202401a"):
        ftd.fetch_fails_to_deliver(ctx, ["AAPL"], full=True, identity=identity)
    stored = sqlite_store.load(Tables.sec_fails_to_deliver).reset_index(drop=True)
    assert len(stored) == 1 and stored.iloc[0]["ticker"] == "AAPL"
    assert pd.Timestamp(stored.iloc[0]["date"]) == pd.Timestamp("2024-01-02")
    assert stored.iloc[0]["fails_quantity"] == 1.0

    print("\n=== SANITY CHECK: FTD replacement gate ===")
    print("  unreadable required cache period raises before replace; stored row survives")
    print("  OK: a partial cache can never become a complete-looking replacement")


def test_ftd_vintage_estimate_and_observed_dates():
    historical = ftd._vintage_frame(["202401a", "202405b"], {})
    assert historical["available_date"].tolist() == [pd.Timestamp("2024-01-30"), pd.Timestamp("2024-06-17")]
    assert set(historical["availability_basis"]) == {"estimated"}

    fetched = {
        "202608b": ("https://www.sec.gov/new.zip", datetime(2026, 9, 16, 14, tzinfo=UTC)),
        "201501a": ("https://www.sec.gov/old.zip", datetime(2026, 9, 16, 14, tzinfo=UTC)),
    }
    vintages = ftd._vintage_frame(["202608b", "201501a"], fetched).set_index("period")
    assert vintages.loc["202608b", "available_date"] == pd.Timestamp("2026-09-16")
    assert vintages.loc["202608b", "availability_basis"] == "observed"
    assert vintages.loc["201501a", "available_date"] == pd.Timestamp("2015-01-30")
    assert vintages.loc["201501a", "availability_basis"] == "estimated"
    print("\n=== SANITY CHECK: FTD ZIP availability ===")
    print("  Jan 1-15 -> Jan 30; weekend Jun 15 -> Mon Jun 17; current HTTP 200 -> observed; old re-download -> estimated.")


def test_ftd_vintage_metadata_backfill_preserves_observed(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    identity = _identity()
    sqlite_store.replace(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-16"]),
                "fails_quantity": [1.0, 2.0],
                "period": ["202401a", "202401b"],
            }
        ),
    )
    sqlite_store.save(
        Tables.sec_ftd_vintages,
        pd.DataFrame(
            {
                "period": ["202401b"],
                "available_date": [pd.Timestamp("2024-02-16")],
                "availability_basis": ["observed"],
                "first_seen_at": [pd.Timestamp("2024-02-16")],
                "source_url": ["https://www.sec.gov/observed.zip"],
            }
        ),
    )
    cache = ftd.cache_dir(ctx, "sec_fails_to_deliver")
    ftd.save_processed_universe(cache, Tables.sec_fails_to_deliver, set(identity.candidate_symbols(frozenset({"AAPL"}))) | {ftd._POLICY_MARKER})
    monkeypatch.setattr(ftd, "_periods", lambda *a, **k: ["202401a", "202401b"])
    monkeypatch.setattr(ftd, "ensure_zip", lambda *a, **k: pytest.fail("stored periods must not download"))
    monkeypatch.setattr(ftd, "record_run", lambda *a, **k: None)

    for _ in range(2):
        assert ftd.fetch_fails_to_deliver(ctx, ["AAPL"], identity=identity) == 0
    vintages = sqlite_store.load(Tables.sec_ftd_vintages).set_index("period")
    assert len(vintages) == 2
    assert pd.Timestamp(vintages.loc["202401a", "available_date"]) == pd.Timestamp("2024-01-30")
    assert vintages.loc["202401a", "availability_basis"] == "estimated"
    assert pd.Timestamp(vintages.loc["202401b", "available_date"]) == pd.Timestamp("2024-02-16")
    assert vintages.loc["202401b", "availability_basis"] == "observed"
    print("\n=== SANITY CHECK: FTD metadata-only backfill ===")
    print("  One missing vintage backfilled from distinct stored periods; no ZIP parsed and observed date survives rerun.")


def test_ftd_first_successful_http_response_sets_observed_date(sqlite_store, monkeypatch, tmp_path):
    ctx = _context(sqlite_store, tmp_path)
    identity = _identity()
    requested: list[str] = []

    class _Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 30, 15, tzinfo=UTC)

    class _Response:
        status_code = 200

        @staticmethod
        def iter_content(chunk_size):
            yield b"downloaded archive"

    def _get(url, **kwargs):
        requested.append(url)
        return _Response()

    ctx.sec_session = SimpleNamespace(get=_get)
    monkeypatch.setattr(bulk_cache, "datetime", _Clock)
    monkeypatch.setattr(ftd, "_periods", lambda *a, **k: ["202609a"])
    monkeypatch.setattr(
        ftd,
        "read_zip_text",
        lambda path, log=None: "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260902|037833100|AAPL|100|APPLE INC|190.00\n",
    )
    monkeypatch.setattr(ftd, "record_run", lambda *a, **k: None)

    assert ftd.fetch_fails_to_deliver(ctx, ["AAPL"], identity=identity) == 1
    assert ftd.fetch_fails_to_deliver(ctx, ["AAPL"], identity=identity) == 0
    vintage = sqlite_store.load(Tables.sec_ftd_vintages).iloc[0]
    assert len(requested) == 1
    assert vintage["period"] == "202609a"
    assert pd.Timestamp(vintage["available_date"]) == pd.Timestamp("2026-09-30")
    assert vintage["availability_basis"] == "observed"
    assert pd.Timestamp(vintage["first_seen_at"]) == pd.Timestamp("2026-09-30 15:00:00")
    assert vintage["source_url"] == requested[0]
    print("\n=== SANITY CHECK: first successful FTD HTTP response ===")
    print("  SEC HTTP 200 stamped 2026-09-30; rerun reused stored observed date without a second request.")


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
    volume = pd.DataFrame({t: 1e6 for t in ("HI", "MID", "LO")}, index=idx)
    peers = {"HI": {"MID": 1.0, "LO": 1.0}, "MID": {"HI": 1.0, "LO": 1.0}, "LO": {"HI": 1.0, "MID": 1.0}}

    panel = build_short_flow_feature_panel(make_frames(idx, peers, volume=volume), None, fails_history=fails)
    assert "f_ic_ftd_to_adv20" in panel.columns

    # The entire historical January-a ZIP becomes visible 15 days after its period end.
    d = pd.Timestamp("2024-01-30")
    row = panel[panel["date"] == d].set_index("ticker")
    assert row["f_ic_ftd_to_adv20"]["HI"] > row["f_ic_ftd_to_adv20"]["LO"]
    assert row["f_ic_ftd_to_adv20"]["HI"] > row["f_ic_ftd_to_adv20"]["LO"]

    # leak-free: no row from the ZIP is visible before its shared publication date.
    early = panel[panel["date"] == pd.Timestamp("2024-01-29")]
    assert early.empty or early["f_ic_ftd_to_adv20"].isna().all()

    print("\n=== SANITY: FTD feature (fails/ADV20, ZIP-publication dated) ===")
    print(
        f"  HI fails/ADV20 {row['f_ic_ftd_to_adv20']['HI']:.4f} ranks above LO "
        f"{row['f_ic_ftd_to_adv20']['LO']:.6f} on estimated Jan 30; "
        f"the Jan 29 prefix is absent. Validated."
    )
