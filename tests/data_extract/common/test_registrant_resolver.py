"""`resolve_registrant_filings` -- which CIKs a ticker's filings are listed from, and how they combine.

Listing reads only the ticker's `FilingScope` (`entity_lineage`): event forms UNION every event CIK,
consolidating forms SPLIT by the CIK windows, widened 31 days at a seam. No symbol, current or
historical, ever reaches `edgar.Company`: a reused symbol resolves to whoever holds it TODAY, which is
how ~1,750 foreign 8-Ks reached `sec_8k`. A filing whose CIK is outside the listed CIKs is skipped and
counted, never raised.

Offline: a stub `Company`, no network.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.entity_lineage import derive_entity_lineage
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.common.registrant import FORM_POLICY, Combine, combine_for, resolve_registrant_filings
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity, filing, patch_company

LOGGER = "src.data_extract.utils.common.registrant"
REPO = Path(__file__).resolve().parents[3]

#: XOM's register shape: 34088 to the 2026-07-01 seam, 2115436 after it.
_XOM = dated_identity(
    [("XOM", "0000034088", "cik_window", SENTINEL, "2026-07-01"), ("XOM", "0002115436", "cik_window", "2026-07-01", None)],
    {"XOM": "0002115436"},
)


def _list(identity, ticker: str, forms: list[str], **kwargs) -> list[str]:
    kwargs.setdefault("since", None)
    kwargs.setdefault("done_accessions", frozenset())
    return [f.accession_number for f in resolve_registrant_filings(identity.filing_scope(ticker), forms, **kwargs)]


# --------------------------------------------------------------------------- #
# The policy table                                                             #
# --------------------------------------------------------------------------- #
def test_an_undeclared_form_raises_and_the_message_names_it():
    """Fail closed: a new form family must not inherit a combination rule silently."""
    with pytest.raises(ValueError, match="NT 10-K"):
        combine_for(["NT 10-K"])
    print("\n=== SANITY CHECK: an undeclared form raises ===")
    print("  'NT 10-K' has no FORM_POLICY entry -> ValueError naming it. Validated.")


def test_a_mixed_policy_list_raises_at_the_call_site():
    with pytest.raises(ValueError, match="mix"):
        combine_for(["8-K", "10-K"])
    print("\n=== SANITY CHECK: a mixed forms list raises ===")
    print("  ['8-K', '10-K'] mixes union and split -> refused. Validated.")


@pytest.mark.parametrize(
    ("forms", "expected"),
    [
        (["8-K", "8-K/A"], Combine.UNION),
        (["SC 13D", "SCHEDULE 13D/A"], Combine.UNION),
        (["SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"], Combine.UNION),
        (["3", "4", "5"], Combine.UNION),
        (["10-K", "10-K/A", "10-Q", "10-Q/A"], Combine.SPLIT),
        (["10-K", "10-Q"], Combine.SPLIT),
        (["DEF 14A", "DEF 14C", "DEFC14A"], Combine.SPLIT),
    ],
)
def test_every_fetched_form_family_has_the_policy_its_pipeline_needs(forms, expected):
    assert combine_for(forms) is expected
    print(f"\n=== SANITY CHECK: {forms[0]}... -> {expected.value} ===")


def test_both_spellings_of_the_renamed_schedules_are_declared():
    for pair in (("SC 13D", "SCHEDULE 13D"), ("SC 13D/A", "SCHEDULE 13D/A"), ("SC 13G", "SCHEDULE 13G"), ("SC 13G/A", "SCHEDULE 13G/A")):
        assert all(f in FORM_POLICY for f in pair), pair
    print("\n=== SANITY CHECK: both form-string eras are declared ===")
    print("  SC 13D/G and SCHEDULE 13D/G, base and /A, all present. Validated.")


# --------------------------------------------------------------------------- #
# AC-001 / AC-002 / AC-004: CIK only, never a symbol                           #
# --------------------------------------------------------------------------- #
#: Non-issuer `Company(...)` call sites allowed a string: 13F managers are listed by their own CIK, not a symbol (00-run N1).
_COMPANY_ALLOWLIST = {"src/data_extract/utils/institutionals/fetch_13f_managers.py"}


def test_no_issuer_listing_builds_company_from_anything_but_an_int_cik():
    """AC-001: every `Company(...)` under `src/data_extract` takes `int(...)`, and no alias walk remains."""
    offenders: list[str] = []
    alias_sites: list[str] = []
    for path in sorted((REPO / "src" / "data_extract").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        source = path.read_text(encoding="utf-8")
        if "alias" in source.lower():
            alias_sites.append(rel)
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name != "Company" or rel in _COMPANY_ALLOWLIST:
                continue
            arg = node.args[0] if node.args else None
            if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "int"):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], f"Company() built from a non-int argument: {offenders}"
    assert alias_sites == [], f"alias code under src/data_extract: {alias_sites}"
    print("\n=== SANITY CHECK: AC-001 source search ===")
    print(f"  every Company() call under src/data_extract takes int(cik); allowlisted (manager CIKs): {sorted(_COMPANY_ALLOWLIST)}; no alias code")


def test_an_alias_held_by_a_foreign_cik_yields_no_accession(monkeypatch):
    """AC-002: ALB's old Form 4 typo `AB` is AllianceBernstein today. Listing never asks for `AB`,
    so none of its filings can arrive, whatever CIK they carry."""
    identity = dated_identity(
        [("ALB", "0000915913", "cik_window", SENTINEL, None)],
        {"ALB": "0000915913"},
        symbols=(("ALB", "0000915913", "2000-01-01"), ("AB", "0000915913", "2003-01-01")),
    )
    built = patch_company(
        monkeypatch,
        {
            915913: [filing("albemarle-8k", "2024-02-15", 915913)],
            "AB": [filing("alliancebernstein-8k", "2024-02-12", 825313), filing("ab-no-cik", "2024-03-01")],
            "ALB": [filing("albemarle-8k", "2024-02-15", 915913)],
        },
    )
    for forms in (["8-K"], ["10-K"]):
        assert _list(identity, "ALB", forms) == ["albemarle-8k"]
    assert built == [915913, 915913], f"Company() built for {built}"
    print("\n=== SANITY CHECK: AC-002 alias held by a foreign CIK ===")
    print(f"  Company() built for {built} only; 0 accessions from AB (AllianceBernstein), with or without a CIK on the filing")


@pytest.mark.parametrize(("ticker", "cik", "old_era"), [("ZBH", "0001136869", "2005-03-01"), ("META", "0001326801", "2013-02-01")])
def test_a_renamed_issuer_keeps_its_old_symbol_era_through_its_cik(monkeypatch, ticker, cik, old_era):
    """AC-004: ZMH-era ZBH and FB-era META filings sit under the same CIK, so CIK listing finds them with no alias."""
    identity = dated_identity([(ticker, cik, "cik_window", SENTINEL, None)], {ticker: cik})
    built = patch_company(monkeypatch, {int(cik): [filing("old-era", old_era, int(cik)), filing("today", "2025-02-01", int(cik))]})
    assert _list(identity, ticker, ["10-K"]) == ["old-era", "today"]
    assert _list(identity, ticker, ["8-K"]) == ["old-era", "today"]
    assert all(isinstance(key, int) for key in built)
    print(f"\n=== SANITY CHECK: AC-004 {ticker} ===")
    print(f"  old-symbol era ({old_era}) and today listed from CIK {int(cik)} alone; no symbol walked")


# --------------------------------------------------------------------------- #
# AC-009: sibling CIKs on event forms                                          #
# --------------------------------------------------------------------------- #
def test_a_co_registrant_sibling_is_listed_for_events_but_never_for_consolidating_forms(monkeypatch):
    """AC-009 (TMUS): T-Mobile USA co-files 8-Ks with T-Mobile US. Its event CIK is unioned (a co-indexed
    accession once); its own 10-Ks are a subsidiary's accounts and are not listed."""
    identity = dated_identity(
        [("TMUS", "0001283699", "cik_window", SENTINEL, None), ("TMUS", "0001330849", "cik_event", SENTINEL, None)],
        {"TMUS": "0001283699"},
    )
    shared = filing("joint-8k", "2024-05-01", 1283699)
    built = patch_company(
        monkeypatch,
        {
            1283699: [filing("parent-8k", "2023-01-05", 1283699), shared, filing("parent-10k", "2024-02-01", 1283699)],
            1330849: [shared, filing("sub-8k", "2024-09-01", 1330849), filing("sub-10k", "2024-02-02", 1330849)],
        },
    )
    assert _list(identity, "TMUS", ["8-K"]) == ["parent-8k", "parent-10k", "sub-10k", "joint-8k", "sub-8k"]
    assert sorted(set(built)) == [1283699, 1330849]
    built.clear()
    assert _list(identity, "TMUS", ["10-K"]) == ["parent-8k", "parent-10k", "joint-8k"]
    assert built == [1283699]
    print("\n=== SANITY CHECK: AC-009 TMUS sibling ===")
    print("  8-K: both CIKs listed, the joint accession once; 10-K: the parent's window only (stub returns every form)")


def test_a_predecessor_keeps_its_events_after_the_seam_but_not_its_accounts(monkeypatch):
    """AC-009 (APA): Apache files 8-Ks and 10-Ks for years after APA Corp became the parent. Events
    union; the subsidiary's consolidating filings beyond the 31-day margin are not the group's."""
    identity = dated_identity(
        [("APA", "0000006769", "cik_window", SENTINEL, "2021-03-01"), ("APA", "0001841666", "cik_window", "2021-03-01", None)],
        {"APA": "0001841666"},
    )
    patch_company(
        monkeypatch,
        {
            6769: [filing("apache-2020", "2020-02-27", 6769), filing("apache-sub-2023", "2023-02-22", 6769)],
            1841666: [filing("apa-2022", "2022-02-23", 1841666)],
        },
    )
    assert _list(identity, "APA", ["8-K"]) == ["apache-2020", "apa-2022", "apache-sub-2023"]
    assert _list(identity, "APA", ["10-K"]) == ["apache-2020", "apa-2022"]
    print("\n=== SANITY CHECK: AC-009 APA ===")
    print("  8-K keeps Apache's 2023 filing (event); 10-K drops it (a subsidiary's accounts two years past the seam)")


# --------------------------------------------------------------------------- #
# AC-005 (amended): register windows plus the listed margin filings            #
# --------------------------------------------------------------------------- #
def test_register_windows_list_todays_set_plus_the_margin_filings(monkeypatch):
    """XOM: today's split set, plus each side's filings within 31 days of the seam (P6 margin)."""
    patch_company(
        monkeypatch,
        {
            34088: [
                filing("pre-eve", "2026-06-30", 34088),
                filing("pre-margin", "2026-07-20", 34088),
                filing("pre-late", "2026-08-07", 34088),
            ],
            2115436: [
                filing("suc-early", "2026-06-15", 2115436),
                filing("suc-day", "2026-07-01", 2115436),
                filing("suc-too-early", "2026-05-01", 2115436),
            ],
        },
    )
    out = _list(_XOM, "XOM", ["10-Q"])
    today = ["pre-eve", "suc-day"]
    assert out == ["suc-early", "pre-eve", "suc-day", "pre-margin"]
    assert set(today) < set(out) and "pre-late" not in out and "suc-too-early" not in out
    print("\n=== SANITY CHECK: AC-005 register windows + margin ===")
    print(f"  today's {today} plus margin filings {sorted(set(out) - set(today))}; 'pre-late' (37 days after) and 'suc-too-early' excluded")


def test_a_joint_filing_inside_the_margin_goes_to_the_window_that_owns_its_date(monkeypatch):
    shared_after = filing("joint-after", "2026-07-10", None)
    shared_before = filing("joint-before", "2026-06-20", None)
    patch_company(monkeypatch, {34088: [shared_after, shared_before], 2115436: [shared_after, shared_before]})
    stats: dict[str, int] = {}
    out = resolve_registrant_filings(_XOM.filing_scope("XOM"), ["10-Q"], since=None, done_accessions=frozenset(), stats=stats)
    assert [f.accession_number for f in out] == ["joint-before", "joint-after"]
    print("\n=== SANITY CHECK: a joint filing inside the margin is listed once ===")


def test_an_event_filing_listed_by_two_ciks_is_stamped_with_the_cik_whose_window_owns_its_date(monkeypatch):
    """F-006 (XOM shape): a holdco's listing also returns its predecessor's history under the holdco CIK.
    When the holdco CIK sorts first, the event union must still stamp each accession with its window owner."""
    holdco, predecessor = "0000000001", "0000000002"
    identity = dated_identity(
        [("HC", predecessor, "cik_window", SENTINEL, "2026-07-01"), ("HC", holdco, "cik_window", "2026-07-01", None)], {"HC": holdco}
    )
    patch_company(
        monkeypatch,
        {
            1: [filing("old-8k", "1996-05-01", 1), filing("new-8k", "2026-08-01", 1)],
            2: [filing("old-8k", "1996-05-01", 2)],
        },
    )
    out = resolve_registrant_filings(identity.filing_scope("HC"), ["8-K"], since=None, done_accessions=frozenset())
    stamps = {f.accession_number: FilingStamp.of(f, holdco).cik for f in out}
    assert stamps == {"old-8k": predecessor, "new-8k": holdco}, stamps
    print("\n=== SANITY CHECK: F-006 event stamp follows the window owner ===")
    print(f"  {stamps}: the 1996 8-K carries the predecessor CIK although the holdco CIK is walked first")


def test_split_over_a_five_window_chain_assigns_each_filing_once(monkeypatch):
    """PSKY-like: five registrants, four seams; a year-old filing of the first CIK stays out of later windows."""
    identity = dated_identity(
        [
            ("PSKY", "0000000001", "cik_window", SENTINEL, "2000-01-01"),
            ("PSKY", "0000000002", "cik_window", "2000-01-01", "2006-01-01"),
            ("PSKY", "0000000003", "cik_window", "2006-01-01", "2019-12-04"),
            ("PSKY", "0000000004", "cik_window", "2019-12-04", "2025-08-07"),
            ("PSKY", "0000000005", "cik_window", "2025-08-07", None),
        ],
        {"PSKY": "0000000005"},
    )
    patch_company(
        monkeypatch,
        {
            1: [filing("s1", "1995-06-01", 1), filing("s1-late", "2010-01-01", 1)],
            2: [filing("s2", "2003-06-01", 2)],
            3: [filing("s3", "2010-06-01", 3)],
            4: [filing("s4", "2021-06-01", 4)],
            5: [filing("s5", "2026-01-01", 5)],
        },
    )
    kept = _list(identity, "PSKY", ["10-K"])
    assert kept == ["s1", "s2", "s3", "s4", "s5"]
    print("\n=== SANITY CHECK: a 5-window chain, split ===")
    print(f"  {kept}; the oldest registrant's 2010 filing is outside its window and its margin. Validated.")


# --------------------------------------------------------------------------- #
# AC-006 (amended): uncurated multi-CIK consolidating scope                     #
# --------------------------------------------------------------------------- #
def test_an_uncurated_extra_cik_warns_lists_the_roster_cik_and_is_backlogged(tmp_path, monkeypatch, caplog):
    """No register entry, D3 off: the lineage build warns and backlogs the extra CIK; consolidating
    forms list the roster CIK alone, event forms both."""
    (tmp_path / "configs" / "sec").mkdir(parents=True)
    tenure = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "issuer_cik": "0000000100",
                "valid_from": pd.Timestamp("2006-01-05"),
                "valid_to": pd.Timestamp("2015-07-01"),
                "n_filings": 300,
            },
            {"symbol": "ABC", "issuer_cik": "0000000200", "valid_from": pd.Timestamp("2015-07-01"), "valid_to": pd.NaT, "n_filings": 100},
        ]
    ).assign(source="form345", evidence="fixture")
    dei = tenure.assign(
        source="dei",
        valid_from=[pd.Timestamp("2010-03-01"), pd.Timestamp("2015-07-10")],
        valid_to=[pd.Timestamp("2015-06-16"), pd.Timestamp("2024-11-02")],
    )
    roster = pd.DataFrame([{"ticker": "ABC", "cik": "0000000200"}])
    with caplog.at_level(logging.WARNING):
        build = derive_entity_lineage(
            tenure, roster, pd.DataFrame(columns=["issuer_cik", "owner_cik_raw"]), str(tmp_path / "configs"), dei=dei, auto_windows=False
        )
    backlog = build.backlog[build.backlog["kind"].eq("multi_cik_no_window")]
    assert len(backlog) == 1 and backlog["cik"].iloc[0] == "0000000100"
    assert "curation backlog" in caplog.text and "multi_cik_no_window" in caplog.text

    identity = build_identity(lineage=build.rows, tenure=tenure, roster=roster)
    built = patch_company(monkeypatch, {100: [filing("pred", "2012-02-01", 100)], 200: [filing("succ", "2020-02-01", 200)]})
    assert _list(identity, "ABC", ["10-K"]) == ["succ"]
    assert built == [200]
    assert _list(identity, "ABC", ["8-K"]) == ["pred", "succ"]
    print("\n=== SANITY CHECK: AC-006 uncurated multi-CIK ===")
    print(f"  build WARNING + backlog row ({backlog['detail'].iloc[0][:70]}...); 10-K lists roster CIK 200 only; 8-K lists both")


# --------------------------------------------------------------------------- #
# The guard, dead CIKs, since / done                                           #
# --------------------------------------------------------------------------- #
def test_a_filing_from_outside_the_scope_is_skipped_and_counted_never_raised(monkeypatch, caplog):
    """AC-008 at the resolver: a listing that returns another filer's accession drops it, counts it, warns once."""
    identity = dated_identity([("ALB", "0000915913", "cik_window", SENTINEL, None)], {"ALB": "0000915913"})
    patch_company(
        monkeypatch,
        {915913: [filing("own", "2024-02-15", 915913), filing("foreign-1", "2024-02-12", 825313), filing("foreign-2", "2024-03-12", 825313)]},
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = _list(identity, "ALB", ["8-K"], stats=stats)
    assert out == ["own"]
    assert stats == {"skipped_existing": 0, "foreign_skipped": 2}
    assert [r.getMessage() for r in caplog.records if r.name == LOGGER] == [
        "ALB: skipped filing(s) from CIK 0000825313, outside the filing scope (first: foreign-1)"
    ]
    print("\n=== SANITY CHECK: guard skip-and-count ===")
    print(f"  2 foreign accessions skipped and counted {stats}; one warning; nothing raised")


def test_a_dead_scope_cik_costs_only_its_own_filings(monkeypatch, caplog):
    patch_company(monkeypatch, {34088: [filing("pred", "2026-05-01", 34088)]}, dead=frozenset({2115436}))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _list(_XOM, "XOM", ["8-K"]) == ["pred"]
    assert "XOM: CIK 0002115436 could not be resolved" in caplog.text
    print("\n=== SANITY CHECK: a dead CIK does not kill the walk ===")


@pytest.mark.parametrize("forms", [["8-K"], ["10-K"]])
def test_since_and_done_accessions_apply_under_both_policies(monkeypatch, forms):
    listed = [filing("old", "2020-01-01", 34088), filing("stored", "2026-05-01", 34088), filing("new", "2026-06-01", 34088)]
    patch_company(monkeypatch, {34088: listed, 2115436: []})
    stats: dict[str, int] = {}
    assert _list(_XOM, "XOM", forms, since=pd.Timestamp("2026-01-01"), done_accessions=frozenset({"stored"}), stats=stats) == ["new"]
    assert stats["skipped_existing"] == 1
    print(f"\n=== SANITY CHECK: since + done_accessions under {combine_for(forms).value} ===")
    print("  'old' before `since`, 'stored' already held -> only 'new'. Validated.")
