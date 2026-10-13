"""
test_13f_fast_path.py (tests/data_extract/institutionals/test_13f_fast_path.py)
------------------------------------------------------------------------------
The XML fast path (`_fast_xml_vintage`) stores exactly what the strict XML parse stores, and declines
to the strict parse whenever EdgarTools' information table does not agree with the cover. Known-truth
fakes for every decline rule; real SEC accessions (live, read-only, sequential) for equality.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import httpx
import pandas as pd
import pytest
from edgar import Filing
from edgar.thirteenf.parsers.infotable_xml import parse_infotable_xml

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.sec_io import TransientReadError
from src.data_extract.utils.institutionals import fetch_13f as f13

#: Real XML-era accessions: (cik, form, filing date, accession). 2014Q2/2024Q2 are the P0 bench sample.
_SAMPLE = [
    (1569515, "13F-HR", "2013-08-14", "0000921895-13-001739"),
    (1279342, "13F-HR", "2013-08-14", "0001140361-13-032445"),
    (1056488, "13F-HR", "2013-08-05", "0001056488-13-000004"),
    (1541496, "13F-HR", "2013-08-05", "0001541496-13-000011"),
    (1386929, "13F-HR", "2013-07-10", "0001386929-13-000003"),
    (1009212, "13F-HR", "2014-05-15", "0001009212-14-000003"),
    (1332905, "13F-HR", "2014-05-12", "0001332905-14-000009"),
    (1388168, "13F-HR", "2014-05-07", "0001388168-14-000006"),
    (1206792, "13F-HR", "2014-05-05", "0001144204-14-027585"),
    (1593191, "13F-HR", "2014-04-28", "0001172661-14-000907"),
    (1566414, "13F-HR", "2014-04-17", "0001062993-14-002243"),
    (1631664, "13F-HR", "2019-05-15", "0001398344-19-008980"),
    (49096, "13F-HR", "2019-05-15", "0000049096-19-000009"),
    (1389212, "13F-HR", "2019-05-14", "0001085146-19-001504"),
    (1758163, "13F-HR", "2019-04-09", "0001758163-19-000005"),
    (1730960, "13F-HR", "2019-04-04", "0001144204-19-018164"),
    (1966482, "13F-HR", "2024-05-20", "0001966482-24-000002"),
    (1764807, "13F-HR", "2024-05-13", "0001855205-24-000025"),
    (1616667, "13F-HR", "2024-05-10", "0000894189-24-003271"),
    (2012239, "13F-HR", "2024-05-03", "0002012239-24-000002"),
    (2009396, "13F-HR", "2024-04-05", "0001104659-24-044062"),
    (1673385, "13F-HR/A", "2024-04-03", "0001104659-24-042961"),
    (1009012, "13F-HR", "2014-04-25", "0001009012-14-000010"),
]

#: Cover 2,860 vs EdgarTools' VALUE sum 2,860,056,000 (scale x1000): a ratio-1000 cover the fast path must decline.
_EXPECT_STRICT = {"0001009012-14-000010"}


def _info_xml(lines: list[tuple[str, int, int]]) -> str:
    root = ElementTree.Element("informationTable")
    for cusip, value, shares in lines:
        entry = ElementTree.SubElement(root, "infoTable")
        for tag, text in (("nameOfIssuer", f"ISSUER {cusip}"), ("titleOfClass", "COM"), ("cusip", cusip), ("value", str(value))):
            ElementTree.SubElement(entry, tag).text = text
        amount = ElementTree.SubElement(entry, "shrsOrPrnAmt")
        ElementTree.SubElement(amount, "sshPrnamt").text = str(shares)
        ElementTree.SubElement(amount, "sshPrnamtType").text = "SH"
    return ElementTree.tostring(root, encoding="unicode")


def _cover(period: str, entries: int, total: float, schema: str = "") -> str:
    return (
        f"<edgarSubmission><schemaVersion>{schema}</schemaVersion><submissionType>13F-HR</submissionType>"
        f"<isAmendment>false</isAmendment><reportCalendarOrQuarter>{period}</reportCalendarOrQuarter>"
        "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
        f"<tableEntryTotal>{entries}</tableEntryTotal><tableValueTotal>{total}</tableValueTotal></edgarSubmission>"
    )


_LINES = [("037833100", 145_000_000, 800_000), ("594918104", 523_000, 1_300), ("02079K305", 0, 10)]


def _fake(
    *,
    filed: str = "2024-05-10",
    period: str = "2024-03-31",
    lines: list[tuple[str, int, int]] = _LINES,
    scale: int = 1,
    entries: int | None = None,
    holdings: int | None = None,
    total: float | None = None,
    edgar_total: float | None = None,
    **drop: bool,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    """`(report, filing)`: the report's `infotable` is EdgarTools' own parse of the info XML, its VALUE
    multiplied by `scale` (EdgarTools' per-filing unit guess), and `total_value` the cover total times
    `scale` unless `edgar_total` is given; the cover declares `entries`/`total`."""
    info = _info_xml(lines)
    table = parse_infotable_xml(info)
    table["Value"] = table["Value"] * scale
    entries = len(lines) if entries is None else entries
    total = float(sum(v for _, v, _ in lines) if total is None else total)
    report = SimpleNamespace(
        primary_form_information=object(),
        infotable=table,
        total_holdings=entries if holdings is None else holdings,
        total_value=total * scale if edgar_total is None else edgar_total,
        infotable_xml=info,
    )
    for name in drop:
        delattr(report, name)
    cover = _cover(period, entries, total)
    filing = SimpleNamespace(
        cik="1616667", filing_date=filed, period_of_report=period, form="13F-HR", accession_number="0000000001-24-000001", xml=lambda: cover
    )
    return report, filing


def _spy_strict(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts the strict parses of an information table (EdgarTools' parser called from fetch_13f)."""
    calls: list[int] = []

    def _parse(xml: str) -> pd.DataFrame:
        calls.append(1)
        return parse_infotable_xml(xml)

    monkeypatch.setattr(f13, "parse_infotable_xml", _parse)
    return calls


def _read(monkeypatch: pytest.MonkeyPatch, report: Any, filing: Any) -> Any:
    monkeypatch.setattr(f13, "filing_obj", lambda _: report)
    monkeypatch.setattr(f13, "sec_call", lambda fn, *args, **_: fn(*args))
    return f13.read_filing_vintage(FilingStamp.of(filing, "1"))


def _strict(filing: Any, info: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stamp = FilingStamp.of(filing, "1")
    return f13.xml_filing_vintage(
        filing.xml(),
        info,
        cik=stamp.cik,
        accession_number=stamp.accession_number,
        period=stamp.period_of_report,
        filing_date=stamp.filed,
        source=stamp.doc_url,
        expected_form=stamp.form,
    )


def _assert_same(fast: tuple[pd.DataFrame, pd.DataFrame], strict: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    pd.testing.assert_frame_equal(fast[0].drop(columns="completed_at"), strict[0].drop(columns="completed_at"))
    pd.testing.assert_frame_equal(fast[1].reset_index(drop=True), strict[1].reset_index(drop=True))


@pytest.mark.parametrize(
    ("filed", "scale", "total"),
    [("2024-05-10", 1, None), ("2014-05-10", 1000, None), ("2024-05-10", 1, 145_522_000)],
    ids=["dollars", "thousands", "rounded_cover"],
)
def test_fast_path_accepts_an_agreeing_report_and_equals_strict(monkeypatch, filed, scale, total):
    """`rounded_cover`: the cells sum to 145,523,000 and the cover rounds to 145,522,000 (0.0007%)."""
    report, filing = _fake(filed=filed, period=pd.Timestamp(filed).to_period("Q").start_time.strftime("%Y-%m-%d"), scale=scale, total=total)
    filing.period_of_report = filing.xml().split("<reportCalendarOrQuarter>")[1].split("<")[0]
    strict = _strict(filing, report.infotable_xml)
    calls = _spy_strict(monkeypatch)
    fast = _read(monkeypatch, report, filing)
    assert isinstance(fast, tuple), fast
    assert not calls, "the fast path re-parsed the information table"
    _assert_same(fast, strict)
    assert fast[1].raw_value_usd.tolist() == strict[1].raw_value_usd.tolist()
    print(
        f"\nSANITY: {filed} edgartools scale x{scale}, cover {fast[0].declared_value_total.iloc[0]:,.0f} -> fast path, 0 strict parses, rows identical to strict."
    )


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("count_mismatch", {"holdings": 4}),
        ("cover_count_mismatch", {"entries": 4, "holdings": 3}),
        ("value_beyond_half_percent", {"total": 145_523_000 * 1.006}),
        ("ratio_1000_cover_0001009012_14_000010_x1", {"filed": "2014-05-15", "period": "2014-03-31", "total": 145_523, "scale": 1}),
        ("ratio_1000_cover_0001009012_14_000010_x1000", {"filed": "2014-05-15", "period": "2014-03-31", "total": 145_523, "scale": 1000}),
        ("unit_disagreement_2024_x1000", {"scale": 1000, "total": 145_523_000}),
        ("unit_disagreement_2014_x1", {"filed": "2014-05-15", "period": "2014-03-31", "scale": 1}),
        (
            "non_integral_raw_value",
            {"filed": "2014-05-15", "period": "2014-03-31", "lines": [("037833100", 1500, 10)], "total": 1.5, "edgar_total": 1500},
        ),
        ("zero_cover_total", {"lines": [("037833100", 0, 10)]}),
        ("no_primary_form_information", {"primary_form_information": True}),
        ("no_total_holdings", {"total_holdings": True}),
        ("no_total_value", {"total_value": True}),
        ("no_infotable", {"infotable": True}),
    ],
)
def test_fast_path_declines_to_strict(monkeypatch, case, kwargs):
    """Every decline reads through the strict parse, which then decides alone (stored or failed)."""
    report, filing = _fake(**kwargs)
    stamp = FilingStamp.of(filing, "1")
    monkeypatch.setattr(f13, "sec_call", lambda fn, *args, **_: fn(*args))
    assert f13._fast_xml_vintage(report, stamp, filing.xml(), report.infotable_xml) is None
    try:
        strict: Any = _strict(filing, report.infotable_xml)
    except ValueError as e:
        strict = f13.ReadFailure(False, f"ValueError: {e}")
    calls = _spy_strict(monkeypatch)
    result = _read(monkeypatch, report, filing)
    assert len(calls) == 1, "a declined filing must be parsed strictly exactly once"
    if isinstance(strict, f13.ReadFailure):
        assert result == strict
    else:
        _assert_same(result, strict)
    outcome = strict.reason[:60] if isinstance(strict, f13.ReadFailure) else f"stored by strict, unit x{strict[0].value_unit_factor.iloc[0]:g}"
    print(f"\nSANITY: {case} -> fast path declined, 1 strict parse, result identical to strict ({outcome}).")


def test_fast_path_legacy_columns_and_odd_infotables_decline(monkeypatch):
    """The one-walk fakes' legacy `VALUE` columns, a None table and a raising table all decline."""
    report, filing = _fake()
    stamp = FilingStamp.of(filing, "1")
    monkeypatch.setattr(f13, "sec_call", lambda fn, *args, **_: fn(*args))
    legacy = report.infotable.rename(columns={"Value": "VALUE"})

    class _Raising:
        primary_form_information = object()
        total_holdings = 3

        @property
        def infotable(self) -> pd.DataFrame:
            raise ValueError("malformed information table")

    for odd in (SimpleNamespace(**{**vars(report), "infotable": legacy}), SimpleNamespace(**{**vars(report), "infotable": None}), _Raising()):
        assert f13._fast_xml_vintage(odd, stamp, filing.xml(), report.infotable_xml) is None
    print("\nSANITY: legacy VALUE columns, a None infotable and a raising infotable all fall back to strict.")


def test_fast_path_transient_error_stays_transient(monkeypatch):
    """A throttle while building EdgarTools' table is a transient read, never a silent fallback."""
    report, filing = _fake()

    class _Throttled(SimpleNamespace):
        @property
        def infotable(self) -> pd.DataFrame:
            raise TransientReadError("429")

    throttled = _Throttled(**{k: v for k, v in vars(report).items() if k != "infotable"})
    result = _read(monkeypatch, throttled, filing)
    assert isinstance(result, f13.ReadFailure) and result.transient
    print(f"\nSANITY: a transient error inside the fast path -> ReadFailure(transient=True, {result.reason!r}).")


def test_fast_path_cover_validation_error_is_not_swallowed(monkeypatch):
    """A cover that disagrees with the accession still fails, exactly as on the strict path."""
    report, filing = _fake()
    filing.period_of_report = "2023-12-31"
    result = _read(monkeypatch, report, filing)
    assert isinstance(result, f13.ReadFailure) and "report period" in result.reason
    print(f"\nSANITY: cover/accession period conflict -> {result.reason!r} on the fast path too.")


@pytest.fixture(scope="module")
def _sec_identity() -> None:
    from src.context import get_config_context
    from src.data_extract.utils.common.sec_io import configure

    _, ctx = get_config_context(str(Path.cwd() / "configs"), use_cache=False, save=False)
    ctx.ensure_edgar_identity()
    configure(ctx)


@pytest.mark.live
def test_fast_equals_strict_sample(_sec_identity, monkeypatch):
    """Real XML accessions, read sequentially through the EdgarTools disk cache: the walk's reader
    against a forced strict read of the same documents, on every manifest and book column."""
    sends: list[str] = []
    real_send = httpx.Client.send

    def _counted(self: httpx.Client, request: httpx.Request, **kw: Any) -> httpx.Response:
        sends.append(str(request.url))
        return real_send(self, request, **kw)

    monkeypatch.setattr(httpx.Client, "send", _counted)
    calls = _spy_strict(monkeypatch)
    real_fast, fast_hits = f13._fast_xml_vintage, []

    def _fast(*args: Any) -> Any:
        out = real_fast(*args)
        fast_hits.append(out is not None)
        return out

    monkeypatch.setattr(f13, "_fast_xml_vintage", _fast)
    rows, multi = [], 0
    for cik, form, filed, accession in _SAMPLE:
        try:
            report = Filing(cik, "", form, filed, accession).obj()  # warms the disk cache for both timed reads
            sends.clear()
            calls.clear()
            start = time.perf_counter()
            fast = f13.read_filing_vintage(FilingStamp.of(Filing(cik, "", form, filed, accession), ""))
            fast_s = time.perf_counter() - start
        except httpx.HTTPError as e:  # pragma: no cover -- offline
            pytest.skip(f"SEC unreachable: {e}")
        assert isinstance(fast, tuple), f"{accession}: {fast}"
        fast_sends, fast_parses, used_fast = len(sends), len(calls), fast_hits[-1]
        multi += int(used_fast and len(report.holdings) != report.total_holdings)
        monkeypatch.setattr(f13, "_fast_xml_vintage", lambda *_: None)
        start = time.perf_counter()
        strict = f13.read_filing_vintage(FilingStamp.of(Filing(cik, "", form, filed, accession), ""))
        strict_s = time.perf_counter() - start
        monkeypatch.setattr(f13, "_fast_xml_vintage", _fast)
        assert isinstance(strict, tuple), f"{accession}: {strict}"
        _assert_same(fast, strict)
        assert fast_sends == 1, f"{accession}: {fast_sends} HTTP sends"
        assert fast_parses == (0 if used_fast else 1)
        assert not (used_fast and accession in _EXPECT_STRICT), f"{accession} must decline to strict"
        rows.append((accession, used_fast, len(fast[1]), fast_s, strict_s))
        print(
            f"  {accession} {'fast' if used_fast else 'strict'} rows={len(fast[1]):5d} read fast={fast_s:.3f}s strict={strict_s:.3f}s sends={fast_sends}"
        )
    n_fast = sum(r[1] for r in rows)
    fast_time, strict_time = sum(r[3] for r in rows if r[1]), sum(r[4] for r in rows if r[1])
    assert len(rows) >= 20 and multi >= 1
    print(
        f"\nSANITY: {len(rows)} real XML accessions, {n_fast} fast / {len(rows) - n_fast} fallback, {multi} multi-manager "
        f"(holdings != total_holdings) on the fast path; every manifest (bar completed_at) and book column identical to strict; "
        f"1 HTTP send each; fast-path reads {fast_time:.2f}s vs strict {strict_time:.2f}s ({fast_time / strict_time:.0%})."
    )
