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


def _model(**kwargs) -> ReportModel:
    findings = kwargs.pop("findings", _findings())
    health = kwargs.pop("health", _health())
    status = kwargs.pop("status", None)
    clusters = build_clusters(findings, status=status, roster_size=8)
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
    """`14 -> 0` is the evidence. A fix with no measured drop is not a fix."""
    closed = [cluster_id("T9", "capex"), cluster_id("T10", "capex")]
    text = render(_model(previous_label="2026-08-20 (8 tickers, in_sample)", settled=closed))

    print(f"\nsettled clusters rendered: {[c in text for c in closed]}")
    assert "## delta vs 2026-08-20 (8 tickers, in_sample)" in text
    assert "2 cluster(s) SETTLED" in text and all(c in text for c in closed)


# --------------------------------------------------------------------------- #
# the wontfix footer -- half of what replaces the deleted register             #
# --------------------------------------------------------------------------- #

def test_the_wontfix_footer_is_never_omitted() -> None:
    """Even with nothing on file. A section that vanishes is a section nobody audits."""
    text = render(_model())
    print("\nempty-state footer present:", "## `wontfix` clusters" in text)
    assert "## `wontfix` clusters" in text and "None on file" in text


def test_a_wontfix_leaves_the_menu_and_enters_the_footer() -> None:
    """A settled cluster is not work -- but it stays VISIBLE, which is the whole point."""
    target = cluster_id("T0", "capex")
    status = {target: {"status": WONTFIX, "note": "measured at $0 over 3 periods",
                       "findings_at_decision": 3}}
    model = _model(status=status)
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
                         previous_label="2026-08-20 (8 tickers)", settled=["abc"]))
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
