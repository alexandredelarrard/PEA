"""
test_report.py  (tests/validate/fundamentals/)
--------------------------------------------------------------------------------------------
The RENDERER, on a synthetic model. No DB, no CLI.

What is pinned here is not the prose -- it is the four properties that make the report safe to
act on, each of which was absent from the report this one replaces:

  * the check-health gate renders ABOVE the rankings, with a banner, whenever a check abstained
    or fired over its ceiling. A cluster list from a mis-calibrated run reads as authoritative
    regardless, and this is the only thing that stops it;
  * a run with no comparable predecessor says so IN WORDS instead of omitting the section --
    a first run presented as a trend is a lie;
  * the `wontfix` footer is NEVER omitted, which is half of what replaces the deleted register;
  * the table is capped but the TOTAL is always stated. Capping is right -- agent B works one
    cluster from a menu of five -- but the old renderer capped the FILE at 25 rows per severity
    while its docstring claimed the markdown was untruncated, so run 2's report showed 57 of
    10,898 findings and said nothing about the other 10,841.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.validate.fundamentals import report as report_module
from src.validate.fundamentals.clusters import WONTFIX, build_clusters, build_families
from src.validate.fundamentals.finding import cluster_id
from src.validate.fundamentals.clusters import SettledCluster
from src.validate.fundamentals.report import MENU_SIZE, ReportModel, render, render_json


def _findings(n_clusters: int = 8) -> pd.DataFrame:
    return pd.DataFrame([{
        "run_date": pd.Timestamp("2026-08-24"), "run_id": "abc123def456",
        "cluster_id": cluster_id(f"T{i}", "capex"), "check_name": f"check_{j}",
        "ticker": f"T{i}", "field": "capex", "period_key": f"20{10 + j}-06-30",
        "finding_id": f"f{i}{j}", "tier": 2, "severity": "high", "substrate": "facts",
        "edgar_url": f"https://sec.gov/T{i}", "detail": '{"why": "the stated mechanism"}',
    } for i in range(n_clusters) for j in range(3)])


def _health(*, abstained: bool = False, over_ceiling: bool = False) -> pd.DataFrame:
    return pd.DataFrame([
        {"run_id": "abc123def456", "run_date": pd.Timestamp("2026-08-24"),
         "check_name": "peer_ratio", "tier": 2, "substrate": "facts",
         "examined": 0 if abstained else 1000, "queued": 0 if abstained else 24,
         "info": 0, "ceiling": 0.03, "abstained": abstained, "over_ceiling": False},
        {"run_id": "abc123def456", "run_date": pd.Timestamp("2026-08-24"),
         "check_name": "cross_identity", "tier": 1, "substrate": "history",
         "examined": 1000, "queued": 78, "info": 0, "ceiling": 0.03,
         "abstained": False, "over_ceiling": over_ceiling},
    ])


class _Fix:
    """The minimum of a `FixRecord` the renderers read. See `test_clusters._Fix`."""

    def __init__(self, queued_before: int = 55, queued_after: int = 4) -> None:
        self.layer, self.root_cause = "extraction", "route 1 took a sibling total"
        self.commit_sha, self.test_path = "2fb6ef2", "tests/data_extract/test_x.py"
        self.queued_before, self.queued_after = queued_before, queued_after
        self.findings_before, self.findings_after = 55, 4
        self.run_id_before, self.run_id_after = "3df52ae9af75", "725bae7bf8ed"
        self.evidence_json, self.decided_at = {"accessions": ["0000063908-18-000010"]}, None


def _model(**kwargs) -> ReportModel:
    findings = kwargs.pop("findings", _findings())
    health = kwargs.pop("health", _health())
    waivers = kwargs.pop("waivers", None)
    clusters = build_clusters(findings, waivers=waivers, roster_size=8)
    return ReportModel(
        run_id="abc123def456", run_date=pd.Timestamp("2026-08-24"), findings=findings,
        health=health, clusters=clusters,
        families=build_families(clusters, roster_size=8),
        roster="in_sample", tickers=tuple(f"T{i}" for i in range(8)), tiers=(1, 2, 3),
        **kwargs)


# --------------------------------------------------------------------------- #
# the gate                                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("abstained,over_ceiling,expect_banner", [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_the_health_gate_banners_and_renders_above_the_rankings(
        abstained, over_ceiling, expect_banner) -> None:
    """A ranked list from a mis-calibrated run reads as authoritative regardless."""
    text = render(_model(health=_health(abstained=abstained, over_ceiling=over_ceiling)))
    banner = "MAY BE INFLATED" in text
    health_at, clusters_at = text.index("## check health"), text.index("## clusters")

    print(f"abstained={abstained} over_ceiling={over_ceiling} -> banner={banner}")
    assert banner is expect_banner
    assert health_at < clusters_at, "the gate must render ABOVE the rankings"


def test_an_abstention_is_never_reported_as_a_clean_pass() -> None:
    """`examined == 0` is not 0%. It is "this went unchecked"."""
    text = render(_model(health=_health(abstained=True)))
    row = next(l for l in text.split("\n") if "`peer_ratio`" in l and "|" in l)

    print(f"\n{row.strip()}")
    print("  SANITY: a check that examined nothing found nothing because it LOOKED at "
          "nothing. Rendering that as 0.00% ok is how a roster silently loses a whole check.")
    assert "ABSTAINED" in row and "0.00%" not in row


# --------------------------------------------------------------------------- #
# the delta                                                                    #
# --------------------------------------------------------------------------- #

def test_no_comparable_run_says_so_rather_than_omitting_the_section() -> None:
    """A first run must never render as a trend."""
    text = render(_model(no_delta_reason="no earlier run of this exact scope is on record"))

    print("\n" + "\n".join(l for l in text.split("\n")
                           if "delta" in l.lower())[:400])
    assert "## delta" in text and "No delta" in text
    assert "no earlier run of this exact scope is on record" in text


def test_a_comparable_run_reports_the_settled_clusters_as_the_proof() -> None:
    """`55 -> 4` is the evidence, and the report prints the BASIS beside every settlement."""
    clean = SettledCluster(cluster_id=cluster_id("T9", "capex"), ticker="T9", field="capex",
                           fix=_Fix())
    waived = SettledCluster(cluster_id=cluster_id("T10", "capex"), ticker="T10",
                            field="capex", findings_after=4, waived_findings=3,
                            waived_checks=("peer_ratio", "series_shape"), fix=_Fix())
    text = render(_model(previous_label="2026-08-20 (8 tickers, in_sample)",
                         settled=[clean, waived]))

    print(f"\nclean settlement rendered as : {clean.basis}")
    print(f"waived settlement rendered as: {waived.basis}")
    print("  SANITY: the waived count is printed beside the cluster, so the basis of a "
          "settlement is never invisible. A settlement whose waivers are not shown is a "
          "suppression with better manners.")
    assert "## delta vs 2026-08-20 (8 tickers, in_sample)" in text
    assert "2 cluster(s) SETTLED" in text
    assert "(clean)" in text and "(3 finding(s) waived across 2 check(s))" in text
    assert "route 1 took a sibling total" in text and "55 -> 4" in text


def test_a_settled_cluster_leaves_the_MENU_but_keeps_its_findings() -> None:
    """A settled cluster still carries rows now. Without the SETTLED stamp it would keep its
    score and outrank real work on agent B's menu indefinitely."""
    target = cluster_id("T0", "capex")
    findings = _findings()
    model = _model(findings=findings, previous_label="2026-08-20 (8 tickers, in_sample)",
                   settled=[SettledCluster(cluster_id=target, ticker="T0", field="capex",
                                           findings_after=3, waived_findings=3,
                                           waived_checks=("check_0",), fix=_Fix())])
    ledger_rows = len(model.findings)

    print(f"\nmenu: {[c.cluster_id for c in model.menu]}")
    print(f"  {target} status: "
          f"{ {c.cluster_id: c.status for c in model.clusters}[target] }")
    print(f"  ledger rows still present for it: "
          f"{int((findings['cluster_id'] == target).sum())} of {ledger_rows}")
    print("  SANITY: off the work list, still on the ledger. Nothing was subtracted.")
    assert target not in {c.cluster_id for c in model.menu}
    assert int((findings["cluster_id"] == target).sum()) == 3


def test_the_fix_history_section_names_what_was_actually_done() -> None:
    """The gap this table was added to close: a fix whose only record was a commit sha."""
    target = cluster_id("T0", "capex")
    text = render(_model(previous_label="2026-08-20 (8 tickers, in_sample)",
                         settled=[SettledCluster(cluster_id=target, ticker="T0",
                                                 field="capex", findings_after=3,
                                                 fix=_Fix())]))
    section = text.split("## recorded fixes")[1]

    print(f"\n{chr(10).join(section.strip().splitlines()[:6])}")
    print("  SANITY: layer, before -> after, commit and test are all on the page. "
          "`validate fix show` carries root_cause and evidence.")
    assert "## recorded fixes" in text
    assert "extraction" in section and "2fb6ef2" in section and "55 -> 4" in section
    assert "never subtracts a row" in section


# --------------------------------------------------------------------------- #
# the wontfix footer -- half of what replaces the deleted register             #
# --------------------------------------------------------------------------- #

def test_the_wontfix_footer_is_never_omitted() -> None:
    """Even with nothing on file. A section that vanishes is a section nobody audits."""
    text = render(_model())
    print("\nempty-state footer present:", "## `wontfix` clusters" in text)
    assert "## `wontfix` clusters" in text and "None on file" in text


def test_a_wontfix_leaves_the_menu_and_enters_the_footer() -> None:
    """A wontfix cluster is not work -- but it stays VISIBLE, which is the whole point."""
    target = cluster_id("T0", "capex")
    waivers = {target: {"": {"cluster_id": target, "check_name": "", "status": WONTFIX,
                             "note": "measured at $0 over 3 periods",
                             "findings_at_decision": 3}}}
    model = _model(waivers=waivers)
    text = render(model)

    print(f"\nmenu: {[c.cluster_id for c in model.menu]}")
    print(f"footer carries {target}: {target in text.split('## `wontfix`')[1]}")
    print("  SANITY: suppressed from the WORK LIST, never from the report. A wontfix that "
          "stops being listed is a suppression.")
    assert target not in {c.cluster_id for c in model.menu}
    assert target in text.split("## `wontfix`")[1]
    assert "one more finding reopens it" in text


# --------------------------------------------------------------------------- #
# truncation, and the JSON contract                                            #
# --------------------------------------------------------------------------- #

def test_the_table_is_capped_but_the_TOTAL_is_always_stated() -> None:
    """The old renderer capped the FILE and its docstring denied it: 57 of 10,898, silently.

    Capping is fine — agent B works one cluster from a menu of five, so rows 26 onward are
    read by nobody. Capping SILENTLY is not: what a reader needs from the tail is its size.
    """
    model = _model(findings=_findings(n_clusters=40))
    text = render(model, packets=5, clusters=25)
    listed = sum(1 for c in model.clusters if f"`{c.cluster_id}`" in text)

    print(f"\n{len(model.clusters)} clusters, table capped at 25 -> {listed} listed")
    print("  " + next(l.strip() for l in text.split("\n") if "not listed" in l)[:150])
    assert "top 25 of 40" in text
    assert "15 further cluster(s) carry work and are not listed" in text
    assert listed == 25 and text.count("### ") == 5


def test_the_whole_list_is_still_reachable_for_a_caller_that_wants_it() -> None:
    """Capped by DEFAULT, not by construction. `clusters=None` still lists every one."""
    model = _model(findings=_findings(n_clusters=40))
    text = render(model, packets=2, clusters=None)
    listed = sum(1 for c in model.clusters if f"`{c.cluster_id}`" in text)

    print(f"\nclusters=None -> {listed} of {len(model.clusters)} listed")
    assert listed == len(model.clusters) and "not listed" not in text


def test_the_terminal_view_is_capped_harder_and_says_how_many_it_hid() -> None:
    model = _model(findings=_findings(n_clusters=40))
    text = render(model, packets=2, clusters=10)

    print("\n" + next(l.strip() for l in text.split("\n") if "not listed" in l)[:120])
    assert "top 10 of 40" in text and "30 further cluster(s)" in text


def test_the_json_carries_the_handoff_contract_and_the_health_caveat() -> None:
    """6.0. Agent B parses this; a missing field means B must stop and say so."""
    payload = json.loads(render_json(_model(health=_health(over_ceiling=True))))
    required = {"cluster_id", "ticker", "field", "score", "findings", "checks_agreeing",
                "severity_mix", "tier_mix", "period_range", "routing_hint", "family_breadth",
                "edgar_url", "why", "run_id"}
    assert payload["weights"]["corroboration_bonus_per_extra_check"]

    print(f"\nmenu size: {len(payload['menu'])} (MENU_SIZE={MENU_SIZE})")
    print(f"health warning: {payload['check_health']['warning'] is not None}")
    print(f"weights published: {payload['weights']['tier']} {payload['weights']['severity']}")
    assert required <= set(payload["clusters"][0])
    assert len(payload["menu"]) == MENU_SIZE
    assert payload["check_health"]["healthy"] is False
    assert payload["check_health"]["warning"]
    assert payload["weights"]["tier"] and payload["weights"]["severity"]


def test_the_report_is_pure_ascii() -> None:
    """The console logger is cp1252 here, and a UnicodeEncodeError loses the WHOLE report.

    It happened: a warning-sign glyph in the health banner raised inside `logging`, the
    terminal got a traceback instead of the report, and only the files were readable.
    """
    text = render(_model(health=_health(abstained=True, over_ceiling=True),
                         previous_label="2026-08-20 (8 tickers)",
                         settled=[SettledCluster(cluster_id="abc", ticker="T0",
                                                 field="capex", findings_after=4,
                                                 waived_findings=3,
                                                 waived_checks=("peer_ratio", "series_shape"),
                                                 fix=_Fix())]))
    offenders = sorted({c for c in text if ord(c) > 127})

    print(f"\nnon-ASCII characters in a fully-populated report: {offenders}")
    print("  SANITY: a report that cannot be printed is not a report.")
    assert not offenders
    text.encode("cp1252")


def test_the_routing_hint_admits_when_it_is_not_discriminating() -> None:
    """48 of 50 families came back identical on the real roster. That is noise, not evidence."""
    text = render(_model())
    hints = {f.routing_hint for f in _model().families if f.total_score > 0}

    print(f"\nhints present: {hints}")
    print(f"degeneracy note rendered: {'NOT discriminating' in text}")
    assert len(hints) == 1 and "NOT discriminating" in text
