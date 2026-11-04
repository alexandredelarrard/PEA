"""CIK history (REQ-003, AC-004): the SEC succession oracle from `sec_cik_profile`, the Sharadar CIK seed, the register's
precedence, and the named companies on the phase-0 fixture store.

Known-truth fixtures for the oracle rules; the named cases (TT, XOM, WBD, HUBB, CHTR, UAL, DD, DIS, PSKY) on real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.common.registrant import load_registrants
from src.data_extract.utils.identity.cik_history import SENTINEL_START, succession_pairs
from src.data_extract.utils.identity.company_anchor import sharadar_ciks
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.support.lineage_build import cik_history_of

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
RUN = Path("reports/validate/2026-11-03-identity-security-master/_cache")
CONFIG_DIR = "./configs"
NO_OWNERS = pd.DataFrame(columns=["issuer_cik", "owner_cik_raw"])


def _tenure(rows: list[tuple[str, str, str, str | None, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "issuer_cik": c,
                "valid_from": pd.Timestamp(f),
                "valid_to": pd.Timestamp(t) if t else pd.NaT,
                "n_filings": n,
                "source": "form345",
                "evidence": c,
            }
            for s, c, f, t, n in rows
        ]
    )


def _profile(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """`sec_cik_profile` rows from (cik, column, form, filed)."""
    by_cik: dict[str, dict[str, list[dict[str, str]]]] = {}
    for cik, column, form, filed in rows:
        by_cik.setdefault(cik, {}).setdefault(column, []).append({"form": form, "filed": filed, "accession": f"{cik}-26-000001"})
    columns = ("form_25", "form_15", "form_8k12b")
    return pd.DataFrame([{"cik": cik, **{c: json.dumps(cols[c]) if c in cols else None for c in columns}} for cik, cols in by_cik.items()])


def _config(tmp_path: Path, register: dict | None = None) -> str:
    sec = tmp_path / "configs" / "sec"
    sec.mkdir(parents=True, exist_ok=True)
    if register is not None:
        (sec / "registrant_cutover.json").write_text(json.dumps(register), encoding="utf-8")
    return str(tmp_path / "configs")


def _windows(history) -> dict[str, tuple[pd.Timestamp, pd.Timestamp | None, str]]:
    rows = history.rows[history.rows["role"].eq("cik_window")]
    return {
        r.cik: (pd.Timestamp(r.valid_from), None if pd.isna(r.valid_to) else pd.Timestamp(r.valid_to), r.sources)
        for r in rows.itertuples(index=False)
    }


#: An unregistered holdco reorganisation, XOM's shape: the old registrant files under XYZ to the seam, the new one after it.
HOLDCO_TENURE = [("XYZ", "0000000100", "2006-01-05", "2026-07-01", 400), ("XYZ", "0000000200", "2026-07-02", None, 12)]


def test_an_8k12b_paired_with_a_form_25_joins_and_dates_an_unregistered_holdco(tmp_path: Path) -> None:
    """X7: the successor's 8-K12B and the predecessor's 25-NSE within the tolerance join the two CIKs and date the seam."""
    tenure = _tenure(HOLDCO_TENURE)
    profile = _profile([("0000000200", "form_8k12b", "8-K12B", "2026-07-01"), ("0000000100", "form_25", "25-NSE", "2026-07-02")])

    joined = cik_history_of(tenure, {"XYZ": "0000000200"}, NO_OWNERS, _config(tmp_path), profile=profile)
    alone = cik_history_of(tenure, {"XYZ": "0000000200"}, NO_OWNERS, _config(tmp_path))

    assert joined.entity_of["0000000100"] == joined.entity_of["0000000200"] == "E0000000100"
    assert joined.oracle_of["0000000100"] == "sec_succession"
    assert _windows(joined) == {
        "0000000100": (SENTINEL_START, pd.Timestamp("2026-07-01"), "sec_profile"),
        "0000000200": (pd.Timestamp("2026-07-01"), None, "sec_profile"),
    }
    assert joined.backlog.empty
    assert "0000000100" not in set(alone.rows["cik"]), "without the SEC profile nothing links the two CIKs"
    print("\n=== SANITY CHECK: SEC succession oracle ===")
    print(joined.rows[["company_id", "cik", "role", "valid_from", "valid_to", "sources", "oracle"]].to_string(index=False))
    print("  OK: 8-K12B on 200 (2026-07-01) + 25-NSE on 100 (2026-07-02) -> one company, seam at the 8-K12B date; no profile -> no link")


def test_a_far_form_25_and_a_pair_of_universe_tickers_are_not_successions(tmp_path: Path) -> None:
    """A Form 25 outside the tolerance pairs nothing; a pair of two roster CIKs is refused like any merge."""
    tenure = _tenure(HOLDCO_TENURE)
    far = _profile([("0000000200", "form_8k12b", "8-K12B", "2026-07-01"), ("0000000100", "form_25", "25-NSE", "2026-09-30")])
    assert succession_pairs(far, {"XYZ": {"0000000100", "0000000200"}}).empty

    both = _profile([("0000000200", "form_8k12b", "8-K12B", "2026-07-01"), ("0000000100", "form_15", "15-12B", "2026-07-03")])
    two = pd.concat([tenure, _tenure([("OLD", "0000000100", "2006-01-05", None, 5)])], ignore_index=True)
    history = cik_history_of(two, {"XYZ": "0000000200", "OLD": "0000000100"}, NO_OWNERS, _config(tmp_path), profile=both)
    assert history.entity_of["0000000100"] != history.entity_of["0000000200"]
    assert history.blocked["proposed_by"].tolist() == ["sec_succession[XYZ]"]
    print("\n=== SANITY CHECK: succession refusals ===")
    print("  25-NSE 91 days after the 8-K12B -> no pair; a pair joining two universe tickers -> blocked, two companies")


def test_the_register_beats_the_sec_succession(tmp_path: Path) -> None:
    """DIS shape: the register seam (2019-02-06) stands although the 8-K12B pair dates the switch 2019-03-20."""
    register = {
        "ABC": {
            "kind": "reorganisation",
            "segments": [{"cik": "100", "valid_to": "2019-02-06", "evidence": "old"}, {"cik": "200", "valid_from": "2019-02-06", "evidence": "new"}],
        }
    }
    tenure = _tenure([("ABC", "0000000100", "2006-01-05", "2019-03-20", 400), ("ABC", "0000000200", "2019-03-21", None, 50)])
    profile = _profile([("0000000200", "form_8k12b", "8-K12B", "2019-03-20"), ("0000000100", "form_25", "25-NSE", "2019-03-20")])

    history = cik_history_of(tenure, {"ABC": "0000000200"}, NO_OWNERS, _config(tmp_path, register), profile=profile)

    assert _windows(history)["0000000200"][:2] == (pd.Timestamp("2019-02-06"), None)
    assert history.oracle_of["0000000100"] == history.oracle_of["0000000200"] == "register"
    print("\n=== SANITY CHECK: register over SEC succession ===")
    print("  register seam 2019-02-06 kept; the 8-K12B pair (2019-03-20) neither moves it nor claims the CIKs")


def test_the_sharadar_cik_is_a_candidate_seed(tmp_path: Path) -> None:
    """REQ-003 seeds: a Sharadar `secfilings` CIK that never typed the symbol is still scored against the roster CIK."""
    tenure = _tenure([("ABC", "0000000200", "2012-01-03", None, 50)])
    owners = pd.DataFrame([(cik, f"{k:010d}") for cik in ("0000000100", "0000000200") for k in range(10)], columns=["issuer_cik", "owner_cik_raw"])

    seeded = cik_history_of(tenure, {"ABC": "0000000200"}, owners, _config(tmp_path), sharadar_cik={"ABC": "0000000100"})
    unseeded = cik_history_of(tenure, {"ABC": "0000000200"}, owners, _config(tmp_path))

    assert seeded.entity_of["0000000100"] == seeded.entity_of["0000000200"] and seeded.oracle_of["0000000100"] == "owner_overlap"
    assert "0000000100" not in unseeded.entity_of
    print("\n=== SANITY CHECK: Sharadar CIK seed ===")
    print("  seeded: 0000000100 shares 10 owners with the roster CIK -> joined by owner_overlap; unseeded: never a candidate")


def test_a_missing_predecessor_window_is_modelled_and_closed_by_a_register_entry(tmp_path: Path) -> None:
    """WBD shape: the tape starts 2005-07-08 under a predecessor that typed another symbol; the roster CIK files from
    2008-09-18. Uncurated, the roster window alone covers the gap; the curated register entry (phase 8) dates it."""
    tenure = _tenure([("DISCA", "0001320482", "2005-07-08", "2008-09-18", 60), ("WBD", "0001437107", "2008-09-18", None, 900)])
    register = {
        "WBD": {
            "kind": "reorganisation",
            "segments": [
                {"cik": "1320482", "valid_to": "2008-09-18", "evidence": "Discovery Holding Co"},
                {"cik": "1437107", "valid_from": "2008-09-18", "evidence": "Discovery Communications"},
            ],
        }
    }
    roster = {"WBD": "0001437107"}

    gap = cik_history_of(tenure, roster, NO_OWNERS, _config(tmp_path / "gap"))
    curated = cik_history_of(tenure, roster, NO_OWNERS, _config(tmp_path / "curated", register))

    assert _windows(gap) == {"0001437107": (SENTINEL_START, None, "roster")}
    assert _windows(curated) == {
        "0001320482": (SENTINEL_START, pd.Timestamp("2008-09-18"), "register"),
        "0001437107": (pd.Timestamp("2008-09-18"), None, "register"),
    }
    print("\n=== SANITY CHECK: WBD missing window (modelled) ===")
    print("  uncurated: 1437107 alone from the open start, the 2005-07-08..2008-09-17 registrant is unseen")
    print("  with the dated register entry: 1320482 to 2008-09-18, 1437107 from it -- the curation phase 8 adds")


# --------------------------------------------------------------------------- #
# Real data: the phase-0 fixture store                                        #
# --------------------------------------------------------------------------- #
def _run_cache() -> Path:
    for base in (ROOT, *ROOT.parents):
        if (base / RUN / "fixture_store" / "identity.sqlite").exists():
            return base / RUN
    pytest.skip(f"no fixture store under {RUN}")


@pytest.fixture(scope="module")
def fixture_history():
    """`run_cik_history` over the fixture store (read-only) and the owner pairs the parity script cached from the zips."""
    cache = _run_cache()
    owners_path = cache / "parity" / "owner_pairs.parquet"
    if not owners_path.exists():
        pytest.skip(f"no cached owner pairs at {owners_path} (run _scripts/parity_check.py first)")
    store = DataStore(create_engine(f"sqlite:///file:{(cache / 'fixture_store' / 'identity.sqlite').as_posix()}?mode=ro&uri=true"))
    tenure = store.load(Tables.symbol_tenure, project=True, where={"source": ["form345", "manual"]})
    dei = store.load(Tables.symbol_tenure, project=True, where={"source": "dei"})
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik", "permaticker"])
    sharadar = store.load(Tables.sharadar_tickers, columns=["permaticker", "secfilings", "lastupdated"])
    profile = store.load(
        Tables.sec_cik_profile,
        columns=["cik", "form_25", "form_15", "form_8k12b"],
        where=[{"form_8k12b": store.NOT_NULL}, {"form_25": store.NOT_NULL}, {"form_15": store.NOT_NULL}],
    )
    existing = store.load(Tables.universe_cik_symbol, columns=["canonical_ticker", "company_id"])
    tenure = tenure.assign(valid_from=pd.to_datetime(tenure["valid_from"]), valid_to=pd.to_datetime(tenure["valid_to"]))
    roster = dict(zip(universe["ticker"], universe["cik"].astype(str).str.zfill(10), strict=True))
    return cik_history_of(
        tenure,
        roster,
        pd.read_parquet(owners_path),
        CONFIG_DIR,
        dei=dei,
        existing=existing,
        profile=profile,
        sharadar_cik=sharadar_ciks(universe, sharadar),
    )


#: ticker: ({window CIK: (valid_from, valid_to)}, {event-only CIKs}) as the register, the manual and the roster give them.
NAMED = {
    "TT": ({"0001160497": ("1900-01-01", "2009-07-01"), "0001466258": ("2009-07-01", None)}, {"0000836102"}),
    "XOM": ({"0000034088": ("1900-01-01", "2026-07-01"), "0002115436": ("2026-07-01", None)}, set()),
    "DD": ({"0000029915": ("1900-01-01", "2017-08-31"), "0001666700": ("2017-08-31", None)}, {"0000030554"}),
    "DIS": ({"0001001039": ("1900-01-01", "2019-02-06"), "0001744489": ("2019-02-06", None)}, set()),
    "PSKY": ({"0000813828": ("1900-01-01", "2025-08-07"), "0002041610": ("2025-08-07", None)}, set()),
    "WBD": ({"0001437107": ("1900-01-01", None)}, set()),
    "HUBB": ({"0000048898": ("1900-01-01", None)}, set()),
    "CHTR": ({"0001091667": ("1900-01-01", None)}, set()),
    "UAL": ({"0000100517": ("1900-01-01", None)}, set()),
}
#: the frozen ids today's lineage carries (PD-3).
NAMED_IDS = {"TT": "E0000836102", "XOM": "E0000034088", "DD": "E0000029915", "DIS": "E0001001039", "PSKY": "E0000813828", "WBD": "E0001437107"}


def test_the_named_companies_on_the_fixture_store(fixture_history) -> None:
    """TT (two spin-offs, IR refused), XOM (holdco reorganisation, one company), DD (register decision), DIS (2019 new
    registrant), PSKY, WBD, HUBB, CHTR and UAL resolve to the dated windows and frozen ids the curation and roster give."""
    rows = fixture_history.rows
    print("\n=== SANITY CHECK: named companies on the fixture store ===")
    for ticker, (windows, events) in NAMED.items():
        mine = rows[rows["canonical_ticker"].eq(ticker)]
        got = {
            r.cik: (str(pd.Timestamp(r.valid_from).date()), None if pd.isna(r.valid_to) else str(pd.Timestamp(r.valid_to).date()))
            for r in mine[mine["role"].eq("cik_window")].itertuples(index=False)
        }
        got_events = set(mine.loc[mine["role"].eq("cik_event"), "cik"])
        print(f"  {ticker:5s} {sorted(set(mine['company_id']))} windows {got} events {sorted(got_events)}")
        assert (got, got_events) == (windows, events), ticker
        if ticker in NAMED_IDS:
            assert set(mine["company_id"]) == {NAMED_IDS[ticker]}, ticker
    entity = fixture_history.entity_of
    assert entity["0001699150"] != entity["0001466258"], "today's IR (Ingersoll Rand Inc) stays out of TT"
    print("  OK: register, manual and roster windows as curated; IR (0001699150) is its own company")


def test_every_sharadar_cik_lies_in_its_company_and_the_succession_pairs_match_the_register(fixture_history) -> None:
    """AC-004 and X7 on real data: the Sharadar CIK of every universe ticker is one of its company's CIKs (XOM through the
    register), and each 8-K12B pair the profile yields joins CIKs the register or the manual already joins."""
    rows = fixture_history.rows
    entity = fixture_history.entity_of
    cache = _run_cache()
    store = DataStore(create_engine(f"sqlite:///file:{(cache / 'fixture_store' / 'identity.sqlite').as_posix()}?mode=ro&uri=true"))
    universe = store.load(Tables.universe_tickers, columns=["ticker", "cik", "permaticker"])
    seeds = sharadar_ciks(universe, store.load(Tables.sharadar_tickers, columns=["permaticker", "secfilings", "lastupdated"]))
    roster = dict(zip(universe["ticker"], universe["cik"].astype(str).str.zfill(10), strict=True))
    outside = sorted(t for t, cik in seeds.items() if entity.get(cik) != entity.get(roster[t]))
    assert not outside, outside
    assert seeds["XOM"] == "0000034088" and entity["0000034088"] == entity[roster["XOM"]]

    register = {c for entry in load_registrants(CONFIG_DIR).values() for c in entry.all_ciks()}
    succession = rows[rows["oracle"].eq("sec_succession")]
    assert succession.empty, f"a succession the curation does not hold: {succession[['canonical_ticker', 'cik']].to_dict('records')}"
    print("\n=== SANITY CHECK: Sharadar CIKs and SEC successions on the fixture store ===")
    print(f"  {len(seeds)} Sharadar CIK seeds, {len(outside)} outside their company; XOM's 0000034088 joins through the register")
    print(f"  0 rows decided by sec_succession: every 8-K12B pair is already held by the register ({len(register)} CIKs) or the manual")
