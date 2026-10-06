"""FTD per security (Q2b): raw lines stamped from `security_master`, the ticker-grain table rebuilt from them.

The cached-ZIP lines below are real (copied verbatim from the SEC files of the named periods) and so are the
master rows (the Q2a build over the whole cache) and the lineage rows (the q1 lineage) of the companies they
pin: P21 (AC-104) and the edge cases E6-E11, E15, E31, E32, AC-106 and AC-108.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.institutionals import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

REPO = Path(__file__).resolve().parents[3]
BUILT = pd.Timestamp("2026-10-04 12:00:00")
#: The run date: LATER falls inside the 7-day re-check window before it, BUILT does not.
RUN_DATE = pd.Timestamp("2026-10-12")
LATER = pd.Timestamp("2026-10-06 09:00:00")
HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE"
UNIVERSE = ["CB", "DOC", "GM", "MRK", "PLD", "JCI", "XOM", "APTV", "EXE"]

# Real lines of the cached SEC FTD files (one string per period file).
REAL_LINES: dict[str, list[str]] = {
    "200907a": [
        "20090701|165167107|CHK|1625|CHESAPEAKE ENERGY CORP|19.83",
    ],
    "200907b": [
        "20090715|370442105|GMGMQ|15884302|GEN MOTORS CORP;COM USD1 2/3|1.15",
        "20090717|62010A105|MTLQQ|15460598|MOTORS LIQUIDATION CO COM STK |0.39",
    ],
    "200909b": [
        "20090916|02520N106|APO|500|AMERICAN COMMUNITY PPTYS TST|7.20",
    ],
    "200911a": [
        "20091102|589331107|MRK|12005|MERCK & CO INC;COM USD0.01|30.93",
        "20091103|589331107|MRK|3647|MERCK & CO INC;COM USD0.01|31.26",
        "20091103|806605101|SGP|17093|SCHERING PLOUGH CORP|28.40",
        "20091104|589331107|MRK|804174|MERCK & CO INC;COM USD0.01|30.67",
        "20091104|806605101|SGP|4163|SCHERING PLOUGH CORP|28.15",
        "20091105|58933Y105|MRK|373679|MERCK & CO INC;COM USD0.01|32.64",
    ],
    "201001a": [
        "20100104|62010A105|MTLQQ|277775|MOTORS LIQUIDATION CO COM STK |0.47",
    ],
    "201106a": [
        "20110601|00163T109|AMB|2147|AMB PROPERTY CORP|36.99",
        "20110601|743410102|PLD|2017|PROLOGIS|16.56",
        "20110603|00163T109|AMB|6174|AMB PROPERTY CORP|34.07",
        "20110603|743410102|PLD|50574|PROLOGIS|15.21",
        "20110606|74340W103|PLD|619884|PROLOGIS, INC COM|34.00",
    ],
    "201506a": [
        "20150601|171232101|CB|185|CHUBB CORPORATION|97.50",
        "20150602|H0023R105|ACE|293|ACE LIMITED (SWITZERLAND)|106.37",
        "20150602|171232101|CB|2368|CHUBB CORPORATION|96.99",
    ],
    "201601a": [
        "20160104|H0023R105|ACE|1408|ACE LIMITED (SWITZERLAND)|116.85",
        "20160104|171232101|CB|23|CHUBB CORPORATION|132.64",
    ],
    "201601b": [
        "20160119|H0023R105|ACE|48359|ACE LIMITED (SWITZERLAND)|111.02",
        "20160119|H1467J104|CB|284787|CHUBB LTD COM|109.38",
        "20160119|171232101|CBXXXX|78225|CHUBB CORPORATION|127.26",
    ],
    "201608b": [
        "20160815|G91442106|TYC|815|TYCO INTL PLC COM SHS (IRL)|43.81",
        "20160815|478366107|JCI|129|JOHNSON CONTROLS INC|44.10",
    ],
    "201609a": [
        "20160902|G91442106|TYC|5|TYCO INTL PLC COM SHS (IRL)|45.01",
        "20160902|478366107|JCI|48898|JOHNSON CONTROLS INC|45.04",
        "20160906|G51502105|JCIZZZZ|1189973|JOHNSON CTLS INTL PLC SHS (IRL|47.74",
        "20160906|G91442106|TYC|938|TYCO INTL PLC COM SHS (IRL)|45.59",
        "20160907|G51502105|JCI|314448|JOHNSON CTLS INTL PLC SHS (IRL|48.90",
        "20160907|G91442106|TYC|1791|TYCO INTL PLC COM SHS (IRL)|45.59",
    ],
    "201712a": [
        "20171204|G27823106|DLPH|19475|DELPHI AUTOMOTIVE PLC|103.51",
        "20171205|G27823106|DLPH|37966|DELPHI AUTOMOTIVE PLC|104.30",
        "20171206|G27823106|3106PS|74454|DELPHI AUTOMOTIVE PLC|104.30",
        "20171206|G6095L109|APTV|2863|APTIV PLC SHS (JEY)|88.77",
        "20171207|G27823106|3106PS|558|DELPHI AUTOMOTIVE PLC|104.30",
        "20171207|G6095L109|APTV|706207|APTIV PLC SHS (JEY)|85.36",
    ],
    "202102a": [
        "20210211|165167743|CHKAQ|1564892|CHESAPEAKE ENERGY CORP COM|3.00",
        "20210212|165167735|CHK|248956|CHESAPEAKE ENERGY CORP COM|42.80",
        "20210212|165167743|CHKAQ|1564903|CHESAPEAKE ENERGY CORP COM|3.00",
    ],
    "202402b": [
        "20240223|71943U104|DOC|58897|PHYSICIANS RLTY TR COM|11.33",
    ],
    "202403a": [
        "20240301|42250P103|PEAK|651072|HEALTHPEAK PPTYS INC COM (MD) |16.75",
        "20240301|71943U104|DOC|4546|PHYSICIANS RLTY TR COM|11.23",
        "20240304|42250P103|PEAK|6321311|HEALTHPEAK PPTYS INC COM (MD) |17.10",
        "20240304|71943U104|DOCXXXX|7544|PHYSICIANS RLTY TR COM|11.23",
    ],
    "202607a": [
        "20260701|30231G102|XOM|302|EXXON MOBIL CORPORATION|136.72",
        "20260702|30231G102|XOM|60803|EXXON MOBIL CORPORATION|136.28",
        "20260702|30233Q108|XOMZZZZ|560223|EXXONMOBIL HLDGS CORPORATION|0.01",
        "20260706|30233Q108|XOM|22242|EXXONMOBIL HLDGS CORPORATION|137.09",
    ],
}

# Real `security_master` rows (Q2a build): cusip, company, issuer CIK, symbol, class, ratio, role, valid_from, valid_to, reason.
MASTER_ROWS = [
    ("171232101", "CB", "0000020171", "CB", "common", 1.0, "acquired_constituent", "2009-06-26", "2016-01-14", "event_only_cik"),
    ("171232101", "CB", "0000020171", "CBXXXX", "common", 1.0, "excluded", "2016-01-14", "2016-01-28", "transition_placeholder"),
    ("H0023R105", "CB", "0000896159", "ACE", "common", 1.0, "canonical_current", "2009-06-26", "2016-01-14", "tape_symbol"),
    ("H0023R105", "CB", "0000896159", "ACE", "common", 1.0, "excluded", "2016-01-14", "2016-01-16", "superseded"),
    ("H1467J104", "CB", "0000896159", "CB", "common", 1.0, "canonical_current", "2016-01-14", None, "ticker_symbol"),
    ("42250P103", "DOC", "0000765880", "PEAK", "common", 1.0, "canonical_current", "2019-11-01", "2024-03-01", "ticker_symbol"),
    ("42250P103", "DOC", "0000765880", "DOC", "common", 1.0, "canonical_current", "2024-03-01", None, "ticker_symbol"),
    ("71943U104", "DOC", "0001574540", "DOC", "common", 1.0, "acquired_constituent", "2013-07-19", "2024-03-01", "event_only_cik"),
    ("71943U104", "DOC", "0001574540", "DOCXXXX", "common", 1.0, "excluded", "2024-02-29", "2024-03-01", "transition_placeholder"),
    ("62010A105", "GM", "0000040730", "MTLQQ", "unclassified", 1.0, "acquired_constituent", "2009-07-14", "2011-05-31", "event_only_cik"),
    ("37045V100", "GM", "0001467858", "GM", "common", 1.0, "canonical_current", "2010-11-18", None, "ticker_symbol"),
    ("589331107", "MRK", "0000064978", "MRK", "common", 1.0, "canonical_predecessor", "2009-06-26", "2009-11-02", "ticker_symbol"),
    ("58933Y105", "MRK", "0000310158", "MRK", "common", 1.0, "canonical_current", "2009-11-02", None, "ticker_symbol"),
    ("806605101", "MRK", "0000310158", "SGP", "common", 1.0, "acquired_constituent", "2009-06-26", "2009-10-31", "outside_window"),
    ("00163T109", "PLD", "0001045609", "AMB", "common", 1.0, "acquired_constituent", "2009-06-26", "2011-06-01", "outside_window"),
    ("743410102", "PLD", "0000899881", "PLD", "common", 1.0, "canonical_predecessor", "2009-06-26", "2011-06-01", "ticker_symbol"),
    ("74340W103", "PLD", "0001045609", "PLD", "common", 1.0, "canonical_current", "2011-06-01", None, "ticker_symbol"),
    ("478366107", "JCI", "0000053669", "JCI", "common", 1.0, "canonical_predecessor", "2009-06-26", "2016-09-02", "ticker_symbol"),
    ("G51502105", "JCI", "0000833444", "JCI", "common", 1.0, "canonical_current", "2016-09-02", None, "ticker_symbol"),
    ("G51502105", "JCI", "0000833444", "JCIZZZZ", "common", 1.0, "excluded", "2016-09-01", "2016-09-02", "transition_placeholder"),
    ("G91442106", "JCI", "0000833444", "TYC", "common", 1.0, "acquired_constituent", "2014-11-13", "2016-09-03", "outside_window"),
    ("30231G102", "XOM", "0000034088", "XOM", "common", 1.0, "canonical_predecessor", "2009-06-26", "2026-07-02", "manual_boundary"),
    ("30231G102", "XOM", "0000034088", "XOM", "common", 1.0, "canonical_predecessor", "2026-07-02", "2026-07-03", "company_line"),
    ("30233Q108", "XOM", "0002115436", "XOM", "common", 1.0, "canonical_current", "2026-07-03", None, "manual_boundary"),
    ("30233Q108", "XOM", "0002115436", "XOMZZZZ", "common", 1.0, "excluded", "2026-07-01", "2026-07-02", "transition_placeholder"),
    ("G27823106", "APTV", "0001521332", "DLPH", "common", 1.0, "canonical_current", "2011-11-17", "2017-12-05", "manual_boundary"),
    ("G27823106", "APTV", "0001521332", "3106PS", "unclassified", 1.0, "excluded", "2017-12-04", "2017-12-07", "transition_placeholder"),
    ("G6095L109", "APTV", "0001521332", "APTV", "common", 1.0, "excluded", "2017-12-04", "2017-12-05", "superseded"),
    ("G6095L109", "APTV", "0001521332", "APTV", "common", 1.0, "canonical_current", "2017-12-05", "2024-12-18", "ticker_symbol"),
    ("165167107", "EXE", "0000895126", "CHK", "common", 1.0, "canonical_current", "2009-06-26", "2020-04-14", "manual_boundary"),
    ("165167743", "EXE", "0000895126", "CHKAQ", "common", 1.0, "canonical_current", "2020-06-30", "2021-02-10", "manual_boundary"),
    ("165167743", "EXE", "0000895126", "CHKAQ", "common", 1.0, "excluded", "2021-02-10", "2021-02-18", "cancelled_security"),
    ("165167735", "EXE", "0000895126", "CHK", "common", 1.0, "canonical_current", "2021-02-10", "2024-10-02", "manual_boundary"),
]

ROSTER = {
    "CB": "0000896159",
    "DOC": "0000765880",
    "GM": "0001467858",
    "MRK": "0000310158",
    "PLD": "0001045609",
    "JCI": "0000833444",
    "XOM": "0000034088",
    "APTV": "0001521332",
    "EXE": "0000895126",
}
ENTITY = {"CB": "0000020171", "DOC": "0000765880", "GM": "0000040730", "MRK": "0000064978", "PLD": "0000899881", "JCI": "0000053669"}
# q1 lineage (window, event and tape symbol rows): ticker, cik, role, symbol, valid_from, valid_to.
LINEAGE_ROWS = [
    ("CB", "0000020171", "cik_event", "", None, None),
    ("CB", "0000896159", "cik_window", "", None, None),
    ("CB", "0000020171", "symbol", "CB", "2006-01-04", "2016-02-04"),
    ("CB", "0000896159", "symbol", "ACE", "2006-01-06", "2015-12-30"),
    ("CB", "0000896159", "symbol", "CB", "2016-01-19", None),
    ("GM", "0000040730", "cik_event", "", None, None),
    ("GM", "0001467858", "cik_window", "", None, None),
    ("GM", "0000040730", "symbol", "GM", "2006-01-03", "2009-06-06"),
    ("GM", "0000040730", "symbol", "MTLQQ", "2009-10-23", "2019-06-05"),
    ("GM", "0001467858", "symbol", "GM", "2010-11-24", None),
    ("JCI", "0000053669", "cik_window", "", None, "2016-09-02"),
    ("JCI", "0000833444", "cik_window", "", "2016-09-02", None),
    ("JCI", "0000053669", "symbol", "JCI", "2006-01-04", "2016-09-08"),
    ("JCI", "0000833444", "symbol", "JCI", "2016-09-07", None),
    ("JCI", "0000833444", "symbol", "TYC", "2006-01-17", "2016-08-19"),
    ("MRK", "0000064978", "cik_window", "", None, "2009-11-03"),
    ("MRK", "0000310158", "cik_window", "", "2009-11-03", None),
    ("MRK", "0000064978", "symbol", "MRK", "2006-01-04", "2011-01-21"),
    ("MRK", "0000310158", "symbol", "MRK", "2009-11-05", None),
    ("MRK", "0000310158", "symbol", "SGP", "2006-01-04", "2009-11-05"),
    ("DOC", "0001574540", "cik_event", "", None, None),
    ("DOC", "0000765880", "cik_window", "", None, None),
    ("DOC", "0000765880", "symbol", "DOC", "2024-03-04", None),
    ("DOC", "0000765880", "symbol", "PEAK", "2019-11-05", "2024-03-02"),
    ("DOC", "0001574540", "symbol", "DOC", "2013-07-19", "2024-03-02"),
    ("PLD", "0000899881", "cik_window", "", None, "2011-06-03"),
    ("PLD", "0001045609", "cik_window", "", "2011-06-03", None),
    ("PLD", "0000899881", "symbol", "PLD", "2006-01-03", "2011-06-08"),
    ("PLD", "0001045609", "symbol", "AMB", "2006-01-04", "2011-06-04"),
    ("PLD", "0001045609", "symbol", "PLD", "2011-06-07", None),
]


# --------------------------------------------------------------------------- fixtures


def _master(rows: list[tuple] = MASTER_ROWS, stamp: pd.Timestamp = BUILT) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "cusip",
            "canonical_company",
            "issuer_cik",
            "source_symbol",
            "security_class",
            "conversion_ratio",
            "lineage_role",
            "valid_from",
            "valid_to",
            "lineage_reason",
        ],
    )
    frame["security_id"] = "C" + frame["cusip"]
    frame["source"] = sm.SOURCE_FTD
    frame["market_symbol"] = frame["source_symbol"]
    frame["exchange"] = None
    frame["source_accession"] = None
    frame["evidence"] = "fixture"
    frame["n_observations"] = 1
    frame["valid_from"] = pd.to_datetime(frame["valid_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    frame["scope_changed_at"] = stamp
    return frame[list(sm.TABLE_COLUMNS)]


def _lineage() -> pd.DataFrame:
    rows = []
    for ticker, cik, role, symbol, start, end in LINEAGE_ROWS:
        rows.append(
            {
                "entity_id": f"E{ENTITY.get(ticker, ROSTER[ticker])}",
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": symbol,
                "valid_from": pd.Timestamp(start or "1900-01-01"),
                "valid_to": pd.Timestamp(end) if end else pd.NaT,
                "status": "corroborated" if role == "symbol" else "curated",
                "sources": "form345,roster" if role == "symbol" else "register",
                "scope_changed_at": BUILT,
            }
        )
    for ticker in ("XOM", "APTV", "EXE"):
        cik = ROSTER[ticker]
        base = {"entity_id": f"E{cik}", "canonical_ticker": ticker, "cik": cik, "valid_to": pd.NaT, "scope_changed_at": BUILT}
        rows.append({**base, "role": "cik_window", "symbol": "", "valid_from": pd.Timestamp("1900-01-01"), "status": "curated", "sources": "roster"})
        rows.append(
            {
                **base,
                "role": "symbol",
                "symbol": ticker,
                "valid_from": pd.Timestamp("2006-01-03"),
                "status": "corroborated",
                "sources": "form345,roster",
            }
        )
    return pd.DataFrame(rows)


def _identity() -> Any:
    tenure = pd.DataFrame(
        [
            {"symbol": t, "issuer_cik": c, "valid_from": pd.Timestamp("2006-01-01"), "valid_to": None, "n_filings": 5, "source": "form345"}
            for t, c in ROSTER.items()
        ]
    )
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])
    return build_identity(_lineage(), tenure, roster, master=_master())


def _context(store, tmp_path: Path) -> Any:
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.ftd_security"),
        paths={"DATA_STORE": tmp_path},
        config=extract_config(data_extract={"years_history": 15}),
        config_dir=str(REPO / "configs"),
    )


def _serve(monkeypatch: pytest.MonkeyPatch, lines: dict[str, list[str]]) -> list[str]:
    """Serve `lines` as the cached FTD files; returns the periods read."""
    read: list[str] = []

    def _read(path: Path, log: Any = None) -> str:
        period = path.stem.removeprefix("cnsfails")
        read.append(period)
        return "\n".join([HEADER, *lines[period]]) + "\n"

    monkeypatch.setattr(ftd, "_cached_periods", lambda cache: set(lines))
    monkeypatch.setattr(ftd, "_periods", lambda *a, **k: sorted(lines))
    monkeypatch.setattr(ftd, "ensure_zip", lambda context, path, urls, **kwargs: path)
    monkeypatch.setattr(ftd, "read_zip_text", _read)
    monkeypatch.setattr(ftd, "load_identity", lambda context, refresh=False: _identity(), raising=False)
    return read


def _rebuild(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lines: dict[str, list[str]] = REAL_LINES, master: pd.DataFrame | None = None
) -> Any:
    """A full rebuild from `lines` as the whole cache, over a fresh raw table."""
    store.drop(Tables.sec_fails_to_deliver_security)
    store.replace(Tables.security_master, _master() if master is None else master)
    _serve(monkeypatch, lines)
    context = _context(store, tmp_path)
    ftd.fetch_fails_to_deliver(context, tickers=[*UNIVERSE, "BRK-B"], full=True)
    return context


def _grain(store) -> dict[tuple[str, str], float]:
    frame = store.load(Tables.sec_fails_to_deliver)
    return {(str(t), str(pd.Timestamp(d).date())): float(q) for t, d, q in zip(frame["ticker"], frame["date"], frame["fails_quantity"], strict=True)}


def _security(store) -> pd.DataFrame:
    frame = store.load(Tables.sec_fails_to_deliver_security)
    frame["day"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame


def _role(security: pd.DataFrame, cusip: str, day: str) -> tuple[Any, Any]:
    row = security[security["cusip"].eq(cusip) & security["day"].eq(day)].iloc[0]
    return row["ticker"], row["lineage_role"]


# --------------------------------------------------------------------------- AC-104 (P21)


def test_ac104_p21_acquired_and_event_only_lines_never_reach_the_canonical_ticker(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    grain = _grain(sqlite_store)
    assert ("CB", "2015-06-01") not in grain and grain.get(("CB", "2015-06-02")) == 293.0, (
        f"old Chubb counted for CB: {grain.get(('CB', '2015-06-02'))}"
    )
    security = _security(sqlite_store)

    # CB: ACE `H0023R105` in, old Chubb `171232101` out before 2016-01-14, then Chubb Ltd `H1467J104`.
    assert ("CB", "2015-06-01") not in grain, "old Chubb's CB line alone that day"
    assert grain[("CB", "2015-06-02")] == 293.0, "ACE only, not ACE + old Chubb (2,661)"
    assert grain[("CB", "2016-01-04")] == 1408.0
    assert grain[("CB", "2016-01-19")] == 284787.0, "Chubb Ltd only: ACE's tail is superseded, CBXXXX a placeholder"
    assert _role(security, "171232101", "2015-06-02") == ("CB", "acquired_constituent")
    # DOC: PEAK `42250P103` in, Physicians Realty `71943U104` out before 2024-03.
    assert ("DOC", "2024-02-23") not in grain
    assert grain[("DOC", "2024-03-01")] == 651072.0 and grain[("DOC", "2024-03-04")] == 6321311.0
    assert _role(security, "71943U104", "2024-03-01") == ("DOC", "acquired_constituent")
    # Old GM: MTLQQ `62010A105` kept raw as acquired; `370442105` (GMGMQ) is no master security, never stored.
    assert not any(t == "GM" for t, _ in grain)
    assert _role(security, "62010A105", "2010-01-04") == ("GM", "acquired_constituent")
    assert "370442105" not in set(security["cusip"])
    # SGP and AMB are acquired constituents of MRK and PLD before their mergers.
    assert grain[("MRK", "2009-11-03")] == 3647.0 and grain[("MRK", "2009-11-04")] == 804174.0 and grain[("MRK", "2009-11-05")] == 373679.0
    assert _role(security, "806605101", "2009-11-03") == ("MRK", "acquired_constituent")
    assert grain[("PLD", "2011-06-01")] == 2017.0 and grain[("PLD", "2011-06-03")] == 50574.0 and grain[("PLD", "2011-06-06")] == 619884.0
    assert _role(security, "00163T109", "2011-06-03") == ("PLD", "acquired_constituent")
    # TYC out of JCI before 2016-09-02.
    assert grain[("JCI", "2016-08-15")] == 129.0 and grain[("JCI", "2016-09-02")] == 48898.0 and grain[("JCI", "2016-09-07")] == 314448.0
    assert ("JCI", "2016-09-06") not in grain, "JCIZZZZ placeholder and Tyco only"
    assert _role(security, "G91442106", "2016-09-07") == ("JCI", "acquired_constituent")
    print("\n=== SANITY CHECK: AC-104 P21 on real FTD lines ===")
    print("  CB 2015-06-02 = ACE 293 (old Chubb 2,368 kept raw, acquired); DOC = PEAK only; old GM never counted;")
    print("  SGP, AMB and Tyco stored as acquired_constituent and out of MRK, PLD and JCI. Validated.")


def test_p21_tape_symbol_resolves_only_through_window_ciks_inside_their_windows():
    identity = _identity()

    def tape(symbol: str, day: str) -> str | None:
        hit = identity.tape_interval(symbol, day)
        return None if hit is None else identity.ticker_by_entity.get(hit.entity)

    assert tape("ACE", "2015-06-02") == "CB"
    assert tape("CB", "2015-06-02") is None, "old Chubb: event-only CIK"
    assert tape("CB", "2016-03-01") == "CB"
    assert tape("DOC", "2023-01-03") is None, "Physicians Realty: event-only CIK"
    assert tape("PEAK", "2023-01-03") == "DOC"
    assert tape("MTLQQ", "2010-01-04") is None
    assert tape("SGP", "2009-10-01") is None, "window CIK before its window"
    assert tape("AMB", "2011-05-02") is None
    assert tape("TYC", "2016-08-15") is None
    assert tape("MRK", "2009-10-01") == "MRK"
    assert {"TYC", "MTLQQ"}.isdisjoint(identity.universe_symbols(frozenset(UNIVERSE))), "no day inside a window CIK's window"
    print("\n=== SANITY CHECK: P21 symbol fallback ===")
    print("  tape symbols resolve only through the roster CIK or a cik_window CIK inside its window; event-only CIKs never")


# --------------------------------------------------------------------------- edge cases


def test_e6_same_day_cusip_change_placeholder_is_excluded(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    grain, security = _grain(sqlite_store), _security(sqlite_store)
    assert grain[("XOM", "2026-07-02")] == 60803.0 and grain[("XOM", "2026-07-06")] == 22242.0
    assert _role(security, "30233Q108", "2026-07-02") == ("XOM", "excluded")
    value = sqlite_store.load(Tables.sec_fails_to_deliver, where={"ticker": "XOM"})
    day = value[pd.to_datetime(value["date"]).eq(pd.Timestamp("2026-07-02"))].iloc[0]
    assert day["fails_value"] == pytest.approx(60803 * 136.28)
    print("\n=== SANITY CHECK: E6 XOM 2026-07-02 ===")
    print("  old CUSIP 60,803 canonical; XOMZZZZ 560,223 @ $0.01 on the new CUSIP excluded (transition placeholder)")


def test_e7_the_role_boundary_compares_trade_dates_not_settlement_dates(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    security = _security(sqlite_store)
    # MRK seam 2009-11-02 (trade): settled 2009-11-04 but traded 2009-10-30 (T+3) -> the predecessor line still.
    assert _role(security, "589331107", "2009-11-04") == ("MRK", "canonical_predecessor")
    assert _role(security, "806605101", "2009-11-04") == ("MRK", "acquired_constituent")
    assert _role(security, "58933Y105", "2009-11-05") == ("MRK", "canonical_current")
    # T+2 and T+1 eras on a synthetic boundary at trade date 2020-03-04 / 2025-03-04.
    master = _master(
        [
            ("000000AA1", "CB", "0000896159", "OLD", "common", 1.0, "canonical_current", "2019-01-01", "2020-03-04", "x"),
            ("000000AA1", "CB", "0000896159", "OLD", "common", 1.0, "acquired_constituent", "2020-03-04", "2025-03-04", "x"),
            ("000000AA1", "CB", "0000896159", "OLD", "common", 1.0, "excluded", "2025-03-04", None, "x"),
        ]
    )
    lines = {
        "202003a": ["20200305|000000AA1|OLD|10|OLD CO|1.0", "20200306|000000AA1|OLD|20|OLD CO|1.0"],
        "202503a": ["20250304|000000AA1|OLD|30|OLD CO|1.0", "20250305|000000AA1|OLD|40|OLD CO|1.0"],
    }
    _rebuild(sqlite_store, tmp_path, monkeypatch, lines=lines, master=master)
    security = _security(sqlite_store)
    assert _role(security, "000000AA1", "2020-03-05")[1] == "canonical_current", "settled after the boundary, traded 2020-03-03 (T+2)"
    assert _role(security, "000000AA1", "2020-03-06")[1] == "acquired_constituent"
    assert _role(security, "000000AA1", "2025-03-04")[1] == "acquired_constituent", "traded 2025-03-03 (T+1)"
    assert _role(security, "000000AA1", "2025-03-05")[1] == "excluded"
    print("\n=== SANITY CHECK: E7 settlement vs trade date ===")
    print("  roles switch on the trade date (T+3, T+2, T+1); a row settling after a seam but traded before keeps the old role")


def test_e8_half_open_market_boundary(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    grain = _grain(sqlite_store)
    assert grain[("APTV", "2017-12-04")] == 19475.0 and grain[("APTV", "2017-12-05")] == 37966.0
    assert ("APTV", "2017-12-06") not in grain, "3106PS placeholder and APTV's superseded first day"
    assert grain[("APTV", "2017-12-07")] == 706207.0
    master = _master(
        [
            ("G27823106", "APTV", "0001521332", "DLPH", "common", 1.0, "canonical_current", "2011-11-17", "2017-12-05", "manual_boundary"),
            ("G6095L109", "APTV", "0001521332", "APTV", "common", 1.0, "canonical_current", "2017-12-05", None, "ticker_symbol"),
        ]
    )
    lines = {
        "201712a": [
            "20171206|G27823106|DLPH|1|DELPHI AUTOMOTIVE PLC|1.0",
            "20171207|G27823106|DLPH|2|DELPHI AUTOMOTIVE PLC|1.0",
            "20171207|G6095L109|APTV|5|APTIV PLC SHS (JEY)|1.0",
        ]
    }
    _rebuild(sqlite_store, tmp_path, monkeypatch, lines=lines, master=master)
    grain = _grain(sqlite_store)
    assert grain == {("APTV", "2017-12-06"): 1.0, ("APTV", "2017-12-07"): 5.0}, "DLPH traded 2017-12-04 in, 2017-12-05 out"
    print("\n=== SANITY CHECK: E8 APTV [2011-11-17, 2017-12-05) ===")
    print("  DLPH traded 2017-12-04 counts; traded 2017-12-05 it is out and Aptiv's line takes over")


def test_e9_e10_open_start_and_bankruptcy_overlap(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    grain, security = _grain(sqlite_store), _security(sqlite_store)
    assert grain[("EXE", "2009-07-01")] == 1625.0, "CHK covered back to the first FTD row"
    assert grain[("EXE", "2021-02-11")] == 1564892.0
    assert grain[("EXE", "2021-02-12")] == 248956.0, "old CHKAQ never summed with the new CHK"
    assert _role(security, "165167743", "2021-02-12") == ("EXE", "excluded")
    print("\n=== SANITY CHECK: E9/E10 EXE ===")
    print("  CHK from 2009-07-01; CHKAQ ends at the 2021-02-10 cancellation, its later rows excluded, never added to new CHK")


def test_e11_a_symbol_reused_by_another_issuer_is_not_stored(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    security = _security(sqlite_store)
    assert {"02520N106", "370442105"}.isdisjoint(set(security["cusip"]))
    assert not any(t == "APO" for t, _ in _grain(sqlite_store))
    print("\n=== SANITY CHECK: E11 symbol reuse ===")
    print("  APO 2009 (American Community Properties, 02520N106) has no master security: not stored, never counted")


def test_e15_a_summed_line_without_price_makes_the_ticker_day_dollars_null(sqlite_store, tmp_path, monkeypatch):
    master = _master(
        [
            ("000000BB1", "CB", "0000896159", "CB", "common", 1.0, "canonical_current", "2019-01-01", None, "x"),
            ("000000BB2", "CB", "0000896159", "CBB", "class_B", 1.0, "secondary_class", "2019-01-01", None, "x"),
            ("000000BB3", "CB", "0000896159", "CBPRA", "preferred", 1.0, "excluded", "2019-01-01", None, "preferred"),
        ]
    )
    lines = {
        "202003a": [
            "20200303|000000BB1|CB|100|CB CORP|10.0",
            "20200303|000000BB2|CBB|50|CB CORP CL B|.",
            "20200304|000000BB1|CB|100|CB CORP|10.0",
            "20200304|000000BB3|CBPRA|7|CB CORP PFD|.",
        ]
    }
    _rebuild(sqlite_store, tmp_path, monkeypatch, lines=lines, master=master)
    rows = sqlite_store.load(Tables.sec_fails_to_deliver).sort_values("date").reset_index(drop=True)
    assert rows["fails_quantity"].tolist() == [150.0, 100.0]
    assert pd.isna(rows.loc[0, "fails_value"]), "a summed line with fails and no PRICE -> NULL dollars (D-Q2-4)"
    assert rows.loc[1, "fails_value"] == pytest.approx(1000.0), "an excluded preferred without price changes nothing"
    raw = _security(sqlite_store)
    assert raw.loc[raw["cusip"].eq("000000BB2"), "price"].isna().all() and raw.loc[raw["cusip"].eq("000000BB2"), "fails_value"].isna().all()
    print("\n=== SANITY CHECK: E15 missing PRICE ===")
    print("  2020-03-03: 150 shares, NULL dollars (class B has no price); 2020-03-04: $1,000, the priceless preferred is not summed")


def test_ac106_brk_a_enters_in_b_equivalent_shares_and_dollars_take_no_ratio(sqlite_store, tmp_path, monkeypatch):
    master = _master(
        [
            ("084670207", "BRK-B", "0001067983", "BRKB", "class_B", 1.0, "canonical_current", "2009-06-29", "2010-01-19", "ticker_symbol"),
            ("084670702", "BRK-B", "0001067983", "BRKB", "class_B", 1.0, "canonical_current", "2010-01-19", None, "ticker_symbol"),
            ("084670108", "BRK-B", "0001067983", "BRKA", "class_A", 30.0, "secondary_class", "2009-06-26", "2010-01-21", "class_description"),
            ("084670108", "BRK-B", "0001067983", "BRKA", "class_A", 1500.0, "secondary_class", "2010-01-21", None, "class_description"),
        ]
    )
    lines = {
        "201001a": [
            "20100104|084670207|BRKB|100|BERKSHIRE HATHWY INC(HLDG CO)B|3300.0",
            "20100104|084670108|BRKA|2|BERKSHIRE HATHWY INC(HLDG CO)A|99000.0",
        ],
        "201002a": [
            "20100201|084670702|BRKB|100|BERKSHIRE HATHWY INC(HLDG CO)B|70.0",
            "20100201|084670108|BRKA|2|BERKSHIRE HATHWY INC(HLDG CO)A|105000.0",
        ],
    }
    _rebuild(sqlite_store, tmp_path, monkeypatch, lines=lines, master=master)
    rows = sqlite_store.load(Tables.sec_fails_to_deliver).sort_values("date").reset_index(drop=True)
    assert rows["ticker"].tolist() == ["BRK-B", "BRK-B"]
    assert rows["fails_quantity"].tolist() == [100 + 30 * 2, 100 + 1500 * 2]
    assert rows["fails_value"].tolist() == pytest.approx([100 * 3300 + 2 * 99000, 100 * 70 + 2 * 105000])
    print("\n=== SANITY CHECK: AC-106 BRK units ===")
    print("  B-equivalent shares B + 30*A before 2010-01-21 and B + 1,500*A after; dollars are each line's own quantity x PRICE")


def test_ac108_stored_rows_equal_the_zip_lines_and_the_key_holds(sqlite_store, tmp_path, monkeypatch):
    _rebuild(sqlite_store, tmp_path, monkeypatch)
    security = _security(sqlite_store)
    assert not security.duplicated(["cusip", "date"]).any()
    stored = {
        (r.day.replace("-", ""), r.cusip, r.source_symbol, int(r.fails_quantity), r.description, float(r.price))
        for r in security[security["period"].eq("201601b")].itertuples()
    }
    expected = {(d, c, s, int(q), desc.strip(), float(p)) for d, c, s, q, desc, p in (line.split("|") for line in REAL_LINES["201601b"])}
    assert stored == expected
    print("\n=== SANITY CHECK: AC-108 raw rows ===")
    print("  201601b: every in-scope line stored as filed (symbol, CUSIP, quantity, description, price); PK (cusip, date) unique")


def test_the_ticker_grain_table_keeps_its_consumer_schema():
    ddl = (REPO / "sql" / "schema.sql").read_text(encoding="utf-8")
    block = re.search(r'CREATE TABLE IF NOT EXISTS "sec_fails_to_deliver" \((.*?)\);', ddl, re.S)
    assert block is not None
    columns = re.findall(r'^\s*"(\w+)"', block.group(1), re.M)
    assert columns == ["ticker", "date", "fails_quantity", "fails_value", "period"] == list(ftd.TICKER_COLUMNS)
    assert Tables.sec_fails_to_deliver.pk == ("ticker", "date")
    assert Tables.sec_fails_to_deliver.read_columns == ("date", "ticker", "fails_quantity", "period")
    print("\n=== SANITY CHECK: consumer schema ===")
    print("  sec_fails_to_deliver keeps PK (ticker, date) and its five columns; consumers read it unchanged")


# --------------------------------------------------------------------------- incremental stamp, E31, E32


def _seed_unstamped(store, lines: dict[str, list[str]]) -> None:
    frames = [ftd._parse_ftd_lines("\n".join([HEADER, *rows]) + "\n").assign(period=period) for period, rows in lines.items()]
    raw = pd.concat(frames, ignore_index=True).assign(security_id=None, ticker=None, lineage_role=None, security_class=None)
    store.save(Tables.sec_fails_to_deliver_security, raw[list(ftd.SECURITY_COLUMNS)])


def test_incremental_run_stamps_only_unstamped_rows_and_builds_their_periods(sqlite_store, tmp_path, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    _seed_unstamped(sqlite_store, {p: REAL_LINES[p] for p in ("201506a", "200907b")})
    context = _context(sqlite_store, tmp_path)
    monkeypatch.setattr(ftd, "read_zip_text", lambda *a, **k: pytest.fail("the incremental run reads no zip"))
    monkeypatch.setattr(ftd, "load_identity", lambda context, refresh=False: _identity(), raising=False)

    assert ftd.fetch_fails_to_deliver(context, tickers=UNIVERSE, as_of=RUN_DATE) > 0
    security = _security(sqlite_store)
    assert security["security_id"].notna().all() and "370442105" not in set(security["cusip"]), "GMGMQ left scope: deleted"
    assert _grain(sqlite_store) == {("CB", "2015-06-02"): 293.0}

    writes: list[str] = []
    for name in ("save", "delete", "replace"):
        original = getattr(sqlite_store, name)
        monkeypatch.setattr(sqlite_store, name, lambda *a, _n=name, _o=original, **k: writes.append(_n) or _o(*a, **k))
    assert ftd.fetch_fails_to_deliver(context, tickers=UNIVERSE, as_of=RUN_DATE) == 0
    assert writes == [], "nothing unstamped and an unchanged master: no write (E32)"
    print("\n=== SANITY CHECK: incremental FTD stamp ===")
    print("  only NULL-stamped rows are read; the out-of-scope GMGMQ line is deleted; a rerun with an unchanged master writes nothing")


def _stamped_store(store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    context = _rebuild(store, tmp_path, monkeypatch)
    return context


def test_e31_a_moved_boundary_restamps_exactly_the_affected_rows(sqlite_store, tmp_path, monkeypatch):
    context = _stamped_store(sqlite_store, tmp_path, monkeypatch)
    before = _security(sqlite_store)
    master = _master()
    old = master["cusip"].eq("806605101")
    master.loc[old, "valid_to"] = pd.Timestamp("2009-10-30")  # SGP's acquired line ends one day earlier
    master.loc[master["canonical_company"].eq("MRK"), "scope_changed_at"] = LATER
    master = pd.concat(
        [
            master,
            _master([("806605101", "MRK", "0000310158", "SGP", "common", 1.0, "canonical_current", "2009-10-30", "2009-10-31", "x")], stamp=LATER),
        ]
    )
    sqlite_store.replace(Tables.security_master, master)
    saved: list[pd.DataFrame] = []
    original = sqlite_store.save
    monkeypatch.setattr(
        sqlite_store,
        "save",
        lambda table, frame, *a, **k: saved.append(frame.assign(_table=getattr(table, "name", table))) or original(table, frame, *a, **k),
    )

    records = ftd.restamp_fails(context, companies=None, tickers=UNIVERSE, as_of=RUN_DATE)

    after = _security(sqlite_store)
    changed = before.merge(after, on=["cusip", "date"], suffixes=("_b", "_a"))
    changed = changed[changed["lineage_role_b"].ne(changed["lineage_role_a"])]
    assert changed[["cusip", "day_a"]].values.tolist() == [["806605101", "2009-11-04"]], "SGP traded 2009-10-30 only"
    raw_saves = [f for f in saved if f["_table"].iloc[0] == Tables.sec_fails_to_deliver_security.name]
    assert sum(len(f) for f in raw_saves) == 1
    assert _grain(sqlite_store)[("MRK", "2009-11-04")] == 804174.0 + 4163.0
    assert records == []
    print("\n=== SANITY CHECK: E31 master role flip ===")
    print("  SGP's boundary moved one day: exactly one raw row re-stamped and MRK 2009-11-04 rebuilt; nothing downloaded")


def test_e32_an_unchanged_master_restamps_nothing(sqlite_store, tmp_path, monkeypatch):
    context = _stamped_store(sqlite_store, tmp_path, monkeypatch)
    for name in ("save", "delete", "replace", "load"):
        monkeypatch.setattr(sqlite_store, name, lambda *a, _n=name, **k: pytest.fail(f"unchanged master must not {_n}"))
    assert ftd.restamp_fails(context, companies=None, tickers=UNIVERSE, master=_master(), as_of=RUN_DATE) == []
    print("\n=== SANITY CHECK: E32 unchanged master ===")
    print("  every company's stamp predates the last run: no read, no re-stamp, no purge, no write")


def _with_universe(context: Any, store) -> Any:
    """`context` over a stored `sp500_tickers` universe, so a run on fewer tickers is a scoped (`-t`) run."""
    store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": [*UNIVERSE, "BRK-B"]}))
    context.config = extract_config(data_extract={"years_history": 15, "redundant_ticks": []})
    return context


def test_f103_a_scoped_incremental_run_stamps_only_its_own_lines_and_saves_stamps_after_the_rebuild(sqlite_store, tmp_path, monkeypatch):
    sqlite_store.replace(Tables.security_master, _master())
    _seed_unstamped(sqlite_store, {"201506a": REAL_LINES["201506a"]})
    context = _with_universe(_context(sqlite_store, tmp_path), sqlite_store)
    monkeypatch.setattr(ftd, "read_zip_text", lambda *a, **k: pytest.fail("the incremental run reads no zip"))

    ftd.fetch_fails_to_deliver(context, tickers=["MRK"], as_of=RUN_DATE)
    security = _security(sqlite_store)
    assert security.loc[security["cusip"].eq("H0023R105"), "security_id"].isna().all(), "ACE's line stays pending for an unscoped run"
    assert not sqlite_store.exists(Tables.sec_fails_to_deliver) or ("CB", "2015-06-02") not in _grain(sqlite_store)

    original = sqlite_store.save

    def _fail_grain(table, frame, *a, **k):
        if getattr(table, "name", table) == Tables.sec_fails_to_deliver.name:
            raise RuntimeError("crash while rebuilding the ticker rows")
        return original(table, frame, *a, **k)

    monkeypatch.setattr(sqlite_store, "save", _fail_grain)
    with pytest.raises(RuntimeError):
        ftd.fetch_fails_to_deliver(context, tickers=[*UNIVERSE, "BRK-B"], as_of=RUN_DATE)
    assert _security(sqlite_store).loc[lambda f: f["cusip"].eq("H0023R105"), "security_id"].isna().all(), "a failed rebuild leaves the lines pending"
    monkeypatch.setattr(sqlite_store, "save", original)
    ftd.fetch_fails_to_deliver(context, tickers=[*UNIVERSE, "BRK-B"], as_of=RUN_DATE)
    assert _grain(sqlite_store)[("CB", "2015-06-02")] == 293.0
    print("\n=== SANITY CHECK: F-103 scoped incremental FTD ===")
    print("  a MRK-scoped run leaves ACE's 201506a line pending; a crashed rebuild stamps nothing; the next unscoped run builds CB 2015-06-02 = 293")


def test_f101_a_scoped_restamp_never_touches_another_companys_lines(sqlite_store, tmp_path, monkeypatch):
    context = _with_universe(_stamped_store(sqlite_store, tmp_path, monkeypatch), sqlite_store)
    before, grain_before = _security(sqlite_store), _grain(sqlite_store)
    master = _master()
    master.loc[master["cusip"].eq("806605101"), "valid_to"] = pd.Timestamp("2009-10-30")
    master.loc[master["canonical_company"].eq("MRK"), "scope_changed_at"] = LATER
    master = pd.concat(
        [
            master,
            _master([("806605101", "MRK", "0000310158", "SGP", "common", 1.0, "canonical_current", "2009-10-30", "2009-10-31", "x")], stamp=LATER),
        ]
    )
    sqlite_store.replace(Tables.security_master, master)

    assert ftd.restamp_fails(context, companies=None, tickers=["CB"], as_of=RUN_DATE) == []

    pd.testing.assert_frame_equal(
        _security(sqlite_store).sort_values(["cusip", "date"], ignore_index=True), before.sort_values(["cusip", "date"], ignore_index=True)
    )
    assert _grain(sqlite_store) == grain_before
    ftd.restamp_fails(context, companies=None, tickers=[*UNIVERSE, "BRK-B"], as_of=RUN_DATE)
    assert _grain(sqlite_store)[("MRK", "2009-11-04")] == 804174.0 + 4163.0
    print("\n=== SANITY CHECK: F-101 scoped FTD re-stamp ===")
    print("  MRK's master moved; a CB-scoped re-stamp leaves every line and ticker row; the unscoped one re-stamps MRK 2009-11-04")
