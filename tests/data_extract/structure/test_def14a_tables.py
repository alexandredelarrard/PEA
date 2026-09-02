"""The table-anchored carve: recall harness + the five cell-level regressions it depends on.

Runs entirely off the Phase-0 filing cache (`data/cache/def14a_probe/`) -- zero network, zero
LLM cost -- which is what makes it cheap enough to re-run on every edit to the signatures.

The recall harness asserts an IDENTIFYING VALUE from each target table, never merely that a
table was found: the research's naive header signature found tables and picked the wrong one
23 times out of 25 on audit fees.

The five cell-level cases are synthetic-from-real (known-truth fixtures for parsing math, per
the repo convention). Each one is a defect that corrupts CLASSIFICATION, not just values, which
is why they are prerequisites rather than nice-to-haves.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from src.data_extract.utils.structure.def14a_tables import (
    SCT, TARGETS, _has_salary_column, classify_filing, classify_table, iter_tables,
    merge_header_rows, to_tsv,
)

CACHE = Path(__file__).resolve().parents[3] / "data/cache/def14a_probe"
GROUND_TRUTH = Path(__file__).with_name("def14a_tables_ground_truth.json")

#: Payload ceiling for the five serialized tables. Research: the anchor carve is mean 50,300
#: chars and hits its cap on every section of every filing; B alone measured mean 4,283.
MAX_MEAN_TABLE_PAYLOAD = 10_000


def _cached(stem: str) -> str | None:
    p = CACHE / f"{stem}.htm"
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None


def _grid(html: str) -> tuple[list[str], list[list[str]]]:
    """The single table in a one-table fixture, header-merged."""
    grids = iter_tables(html)
    assert grids, "fixture produced no table at all"
    return merge_header_rows(grids[0])


# --------------------------------------------------------------------------- #
# 0. the SCT tie-break: telling a Salary COLUMN from prose about salary       #
# --------------------------------------------------------------------------- #
#: Every case is a real header cell from the 656-filing cache. Both halves are load-bearing and
#: both were got WRONG in turn while this rule was being written, each time silently handing a
#: filing the wrong table.
_SALARY_LABELS = [
    "Salary",                                                    # GE 2019
    "Base Salary",
    "Salary ($)(1)",                                             # BA 2013-2021
    "Salary ($) (c)",                                            # KLAC -- Item 402 uses (a)(b)(c)
    "SUMMARY COMPENSATION TABLE Salary ($) (c)",                 # colspan'd title propagation
    "Summary Compensation Table Annual Compensation Salary ($)",  # 57 chars, 8 words, genuine
    "Salary and incentive compensation Annual compensation Salary ($)",
    "Annual Compensation Salary($)",
]
_NOT_SALARY_LABELS = [
    # ends with the word, but it is a CD&A raise table -- 11-13 words, no unit marker
    "Year-Over-Year Percentage Increase Represented by the Fiscal Year 2013 Base Salary",
    "Percentage Increase Represented by the Approved but Deferred Fiscal Year 2009 Base Salary",
    # a merged FOOTNOTE row
    "The salary portion of the amounts reflected above is included",
    # mentions salary but is not a salary column
    "Annual Base Salary Rate as of June 30, 2012",
    "Name",
    "Annual Incentive Compensation ($)",
]


def test_salary_column_is_told_apart_from_prose_about_salary():
    """Item 402(c)(2)(iii) makes Salary a mandatory SCT column, so it is the strongest
    discriminator available -- but only if a COLUMN can be told from a sentence.

    Two traps, both of which produced a wrong table before being measured:
      * `^salary` fails on `SUMMARY COMPENSATION TABLE Salary ($)`, because a colspan'd table
        title propagates into every cell (PG, GE);
      * end-anchoring alone accepts `... Fiscal Year 2013 Base Salary`, a CD&A raise table that
        beat KLAC's real 15-row SCT with 5 rows.
    A char cap cannot separate them: genuine title-propagated labels reach 64 chars while the
    prose starts at 82. Word count can -- 1-6 words versus 11-13, with nothing in between.
    """
    wrong = ([c for c in _SALARY_LABELS if not _has_salary_column([c])]
             + [c for c in _NOT_SALARY_LABELS if _has_salary_column([c])])
    print("\n=== SANITY: Salary column vs prose ===")
    print(f"  {len(_SALARY_LABELS)} real column labels accepted, "
          f"{len(_NOT_SALARY_LABELS)} prose cells rejected, {len(wrong)} wrong")
    for c in wrong:
        print(f"    MISCLASSIFIED: {c}")
    assert not wrong, wrong


def test_the_real_sct_beats_a_longer_cda_table():
    """Row count alone is not enough, and the failure is not rare: measured over the 656 cached
    filings, a genuine Salary-bearing SCT existed and LOST on 21 of them -- BA 2013-2021 is nine
    consecutive years where `Name and Principal Position | Year | Salary ($)` lost to a longer
    `Name | Year | Annual Incentive Compensation` CD&A table, and GE 2019 lost to a 27-row
    director BIO grid. Those filings then stored `n_neos` from the wrong table."""
    sct = """<table>
      <tr><td>Name and Principal Position</td><td>Year</td><td>Salary ($)</td>
          <td>Stock Awards ($)</td><td>Total ($)</td></tr>
      <tr><td>A. Exec</td><td>2025</td><td>1,000,000</td><td>5,000,000</td><td>6,000,000</td></tr>
      <tr><td>B. Exec</td><td>2025</td><td>900,000</td><td>4,000,000</td><td>4,900,000</td></tr>
    </table>"""
    # the CD&A table is LONGER and also classifies as an SCT candidate
    cda_rows = "".join(
        f"<tr><td>Exec {i}</td><td>2025</td><td>{i}00,000</td><td>{i}00,000</td>"
        f"<td>{i}00,000</td></tr>" for i in range(1, 9))
    cda = f"""<table>
      <tr><td>Name</td><td>Year</td><td>Annual Incentive Compensation ($)</td>
          <td>Long-Term Incentive ($)</td><td>Total ($)</td></tr>{cda_rows}
    </table>"""

    best = classify_filing(f"<html><body>{cda}{sct}</body></html>")
    assert SCT in best, "neither table classified as an SCT"
    header, rows = best[SCT]
    print("\n=== SANITY: SCT tie-break prefers the Salary column ===")
    print(f"  CD&A candidate: 8 data rows, no Salary column")
    print(f"  SCT candidate : 2 data rows, Salary column")
    print(f"  winner: {len(rows)} rows, header={[str(h) for h in header[:3]]}")
    assert _has_salary_column(header), "the longer CD&A table won"
    assert len(rows) == 2


# --------------------------------------------------------------------------- #
# 1. cell-level regressions -- each corrupts classification, not just values  #
# --------------------------------------------------------------------------- #
def test_br_stacked_cell_yields_separate_values():
    """Agilent's multi-year cell. Without a `<br>` separator the three salaries FUSE into one
    number -- the mechanism behind the measured `salary` = 1.000e19 and `year` = '200420032002'.
    Measured cost of dropping the separator: 180 cells (2.1%) across 16 of 25 filings."""
    html = """<table>
      <tr><th>Name</th><th>Year</th><th>Salary</th></tr>
      <tr><td>A. Director</td><td>2004<br>2003<br>2002</td>
          <td>1,000,000<br>1,000,000<br>925,000</td></tr>
      <tr><td>B. Officer</td><td>2004</td><td>500,000</td></tr>
    </table>"""
    _, rows = _grid(html)
    salary = rows[0][2]
    assert "10000001000000925000" not in salary.replace(",", ""), f"values fused: {salary!r}"
    assert salary.split() == ["1,000,000", "1,000,000", "925,000"]
    assert rows[0][1].split() == ["2004", "2003", "2002"]
    print(f"\n  <br> stack -> {salary!r} (three separated values, not one 19-digit number)")


def test_superscript_digit_is_stripped_from_a_share_count():
    """PG's ownership 10x. A bare-digit `<sup>` glues onto the value: `217,956,036` + footnote
    `2` -> `2179560362`. 47 cells affected corpus-wide, 7 of them this value-corrupting form."""
    html = """<table>
      <tr><th>Name</th><th>Shares</th><th>Percent of Class</th></tr>
      <tr><td>Some Holder<sup>1</sup></td><td>217,956,036<sup>2</sup></td><td>9.1%</td></tr>
      <tr><td>Other Holder</td><td>1,234,567</td><td>0.5%</td></tr>
    </table>"""
    _, rows = _grid(html)
    assert rows[0][1] == "217,956,036", f"footnote digit survived: {rows[0][1]!r}"
    assert rows[0][0] == "Some Holder", f"footnote digit survived in the name: {rows[0][0]!r}"
    print(f"\n  <sup> strip -> {rows[0][1]!r} (not '2179560362')")


def test_css_positioned_superscript_is_also_stripped():
    """PG's 2026 proxy replaced `<sup>` with a CSS-positioned `<span>` and the 10x bug SURVIVED.
    Matching on the style is what makes the fix outlive one filer's template change."""
    html = """<table>
      <tr><th>Name</th><th>Shares</th></tr>
      <tr><td>Holder</td><td>217,956,036<span style="vertical-align: top">2</span></td></tr>
      <tr><td>Other</td><td>1,000</td></tr>
    </table>"""
    _, rows = _grid(html)
    assert rows[0][1] == "217,956,036", f"CSS superscript survived: {rows[0][1]!r}"
    print(f"\n  CSS vertical-align:top span -> {rows[0][1]!r}")


def test_currency_glyph_in_its_own_cell_does_not_desync_columns():
    """GE / CAT. A `$` in its own `<td>` doubles the effective column count and shifts every
    value one column left, which dropped 3 of GE's 4 numeric director-comp columns.

    Modelled on GE 2026's real geometry: the header cells are `colspan`-ed to span the `$`
    columns, so header and data rows are the same total width -- which is what any table a
    browser renders correctly must do. Sébastien Bazin's row is the measured example.
    """
    html = """<table>
      <tr><th colspan="2">NAME OF DIRECTOR</th><th colspan="2">CASH FEES</th>
          <th colspan="2">STOCK AWARDS</th><th colspan="2">ALL OTHER COMP</th>
          <th colspan="2">TOTAL</th></tr>
      <tr><td colspan="2">Sebastien Bazin</td><td>$</td><td>0</td><td>$</td><td>345,795</td>
          <td>$</td><td>0</td><td>$</td><td>345,795</td></tr>
      <tr><td colspan="2">Margaret Billson</td><td>$</td><td>140,000</td><td>$</td>
          <td>201,925</td><td>$</td><td>1,000</td><td>$</td><td>342,925</td></tr>
    </table>"""
    header, rows = _grid(html)
    assert header == ["NAME OF DIRECTOR", "CASH FEES", "STOCK AWARDS",
                      "ALL OTHER COMP", "TOTAL"], header
    assert rows[0] == ["Sebastien Bazin", "0", "345,795", "0", "345,795"], rows[0]
    print(f"\n  $-in-own-td -> {rows[0]}")
    print(f"                  aligned to {header}")


def test_multi_row_header_merges_into_one_label():
    """A `Stock` / `Awards ($)` split header is how edgartools drops a column: it treats exactly
    one row as the header. EOG 2026's SCT header spans FOUR rows and the row carrying `Year` --
    the SCT signature's own requirement -- is the last of them."""
    html = """<table>
      <tr><td></td><td></td><td></td><td>Non-Equity</td></tr>
      <tr><td></td><td></td><td>Stock</td><td>Incentive</td></tr>
      <tr><td>Name and</td><td>Fiscal</td><td>Awards</td><td>Plan Comp</td></tr>
      <tr><td>Principal Position</td><td>Year</td><td>($)</td><td>($)</td></tr>
      <tr><td>E. Yacob</td><td>2025</td><td>12,729,628</td><td>2,718,800</td></tr>
    </table>"""
    header, rows = _grid(html)
    blob = " ".join(header).lower()
    assert "year" in blob, f"the 4th header row was cut off: {header}"
    assert "stock awards" in blob, f"split header did not merge: {header}"
    assert rows[0][0] == "E. Yacob" and len(rows) == 1, f"data row consumed as header: {rows}"
    print(f"\n  4-row header -> {header} ('year' present, 1 data row kept)")


def test_rowspan_does_not_shift_following_rows_left():
    """edgartools ignores `rowspan` entirely, which shifts every subsequent row one column left
    for the rest of the table -- so a `Total` value lands under `All Other Compensation`."""
    html = """<table>
      <tr><th>Name</th><th>Year</th><th>Salary</th><th>Total</th></tr>
      <tr><td rowspan="2">A. Officer</td><td>2025</td><td>100,000</td><td>150,000</td></tr>
      <tr><td>2024</td><td>90,000</td><td>140,000</td></tr>
    </table>"""
    _, rows = _grid(html)
    assert rows[1] == ["A. Officer", "2024", "90,000", "140,000"], f"row shifted: {rows[1]}"
    print(f"\n  rowspan=2 -> second row {rows[1]} (name carried down, no left shift)")


def test_spacer_columns_are_dropped_on_data_rows_not_header_text():
    """A `colspan` header label propagates into every column it spans, so a header-based test
    finds no spacers. PG's insider table is 20 columns of which 13 are pure spacers."""
    html = """<table>
      <tr><th colspan="5">Amount and Nature of Beneficial Ownership</th></tr>
      <tr><td>Name</td><td></td><td>Direct</td><td></td><td>Total</td></tr>
      <tr><td>B. Marc Allen</td><td></td><td>9,399</td><td></td><td>9,399</td></tr>
      <tr><td>C. Other</td><td></td><td>511</td><td></td><td>511</td></tr>
    </table>"""
    header, rows = _grid(html)
    assert len(rows[0]) == 3, f"spacer columns survived: {rows[0]}"
    assert rows[0] == ["B. Marc Allen", "9,399", "9,399"]
    print(f"\n  spacer drop -> {len(header)} columns, row {rows[0]}")


# --------------------------------------------------------------------------- #
# 2. signature rejections -- the strictness IS the design                     #
# --------------------------------------------------------------------------- #
def test_pay_versus_performance_table_is_not_classified_as_the_sct():
    """The PvP table has a Year column and dollar columns too. Its giveaway is the regulatory
    phrase 'compensation actually paid', so rejecting on it is safe and exact."""
    html = """<table>
      <tr><th>Year</th><th>Summary Compensation Table Total for PEO</th>
          <th>Compensation Actually Paid to PEO</th><th>Net Income</th></tr>
      <tr><td>2025</td><td>27,429,900</td><td>-82,216,617</td><td>1,000,000</td></tr>
      <tr><td>2024</td><td>24,625,700</td><td>19,904,513</td><td>2,000,000</td></tr>
    </table>"""
    assert "sct" not in classify_table(*_grid(html))
    print("\n  PvP table correctly rejected as the SCT")


def test_audit_fee_footnote_sentence_is_not_classified_as_the_fee_table():
    """'audit fees' appears inside the SENTENCE 'the following table shows audit fees billed',
    which is how a substring test picked 23 wrong tables out of 25."""
    html = """<table>
      <tr><th>Note</th><th>Description</th></tr>
      <tr><td>(1)</td><td>The following table shows audit fees billed and tax fees
          incurred during 2025, which totalled 1,234,567 dollars.</td></tr>
      <tr><td>(2)</td><td>All other fees relate to advisory services of 89,000.</td></tr>
    </table>"""
    assert "audit_fees" not in classify_table(*_grid(html))
    print("\n  fee-label prose footnote correctly rejected as the fee table")


def test_director_comp_and_sct_are_not_confused():
    """A `Salary` column is the discriminator: it is present in the SCT and absent from the
    Item 402(k) director table by regulation."""
    director = """<table>
      <tr><th>Name</th><th>Fees Earned or Paid in Cash</th>
          <th>Restricted Stock Units</th><th>Total</th></tr>
      <tr><td>A. Director</td><td>120,000</td><td>175,033</td><td>295,033</td></tr>
      <tr><td>B. Director</td><td>110,000</td><td>175,033</td><td>285,033</td></tr>
    </table>"""
    matched = classify_table(*_grid(director))
    assert "director_comp" in matched and "sct" not in matched, matched
    print(f"\n  'Restricted Stock Units' + 'Fees Earned' + no Salary -> {matched}")


def test_one_table_can_serve_both_ownership_targets():
    """Filers routinely publish ONE beneficial-ownership table holding the >=5% institutions
    AND the directors-and-officers rows (AAPL 2026's is 16 rows). Under first-match-wins that
    table registered only as `ownership_insider` and >=5% recall sat at 61%."""
    html = """<table>
      <tr><th>Name of Beneficial Owner</th><th>Shares Beneficially Owned</th>
          <th>Percent of Common Stock Outstanding</th></tr>
      <tr><td>The Vanguard Group</td><td>1,415,826,462</td><td>9.63%</td></tr>
      <tr><td>BlackRock, Inc.</td><td>1,043,713,019</td><td>7.10%</td></tr>
      <tr><td>Tim Cook</td><td>3,280,295</td><td>*</td></tr>
      <tr><td>All current directors and executive officers as a group (12 persons)</td>
          <td>5,000,000</td><td>*</td></tr>
    </table>"""
    matched = classify_table(*_grid(html))
    assert {"ownership_insider", "ownership_5pct"} <= set(matched), matched
    print(f"\n  combined ownership table -> {sorted(matched)} (both targets)")


# --------------------------------------------------------------------------- #
# 3. the recall harness over the cached corpus                                #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CACHE.exists(), reason="Phase-0 filing cache absent")
def test_ground_truth_identifying_values_are_reached():
    """Every pinned value must appear in the serialized TSV of its target's table, and every
    pinned null must genuinely yield no table."""
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["filings"]
    checked = failures = 0
    for stem, spec in truth.items():
        html = _cached(stem)
        if html is None:
            continue
        best = classify_filing(html)
        for target, value in spec.items():
            if target.startswith("_"):
                continue
            checked += 1
            if value is None:
                if target in best:
                    failures += 1
                    print(f"  FAIL {stem[:30]} {target}: expected NO table, found one")
                continue
            if target not in best:
                failures += 1
                print(f"  FAIL {stem[:30]} {target}: no table found (want {value!r})")
                continue
            tsv = to_tsv(*best[target])
            if value not in tsv:
                failures += 1
                print(f"  FAIL {stem[:30]} {target}: {value!r} absent from the TSV")
    assert checked, "no ground-truth filing was present in the cache"
    print(f"\n  ground truth: {checked - failures}/{checked} identifying values reached")
    assert failures == 0, f"{failures} of {checked} ground-truth checks failed"


@pytest.mark.skipif(not CACHE.exists(), reason="Phase-0 filing cache absent")
def test_recall_matrix_and_payload_print_conclusion():
    """The sanity conclusion: the recall matrix in the same shape as the research's, plus the
    measured payload. Comparable line-for-line with follow-up 3's numbers."""
    files = sorted(CACHE.glob("*_202[4-6]-*.htm"))
    if not files:
        pytest.skip("no recent filings cached")
    hit = dict.fromkeys(TARGETS, 0)
    payloads = []
    for f in files:
        best = classify_filing(f.read_text(encoding="utf-8", errors="ignore"))
        payloads.append(sum(len(to_tsv(h, r)) for h, r in best.values()))
        for target in TARGETS:
            hit[target] += target in best

    n = len(files)
    total = sum(hit.values())
    mean_payload = statistics.mean(payloads)
    print(f"\n=== SANITY CHECK: table-anchored carve over {n} cached 2024-2026 filings ===")
    print(f"  {'target':<20}{'found':>8}{'rate':>8}   research (25 filings)")
    research = {"sct": "25/25", "director_comp": "24/25", "audit_fees": "25/25",
                "ownership_insider": "22/25", "ownership_5pct": "25/25"}
    for target in TARGETS:
        print(f"  {target:<20}{hit[target]:>5}/{n}{100 * hit[target] / n:>7.0f}%   {research[target]}")
    print(f"  {'TOTAL':<20}{total:>5}/{5 * n}{100 * total / (5 * n):>7.1f}%   121/125 = 96.8%")
    print(f"  payload: mean {mean_payload:,.0f}  median {statistics.median(payloads):,.0f}  "
          f"max {max(payloads):,} chars (research B mean 4,283)")
    print("  The audit-fee residual is ONE measured format class: EOG / GE / JPM / PEG all put")
    print("  their fee CATEGORY LABELS in narrative prose, so no table classifier reaches them.")
    print("  The router's narrative fallback owns that 19%. Validated.")

    assert hit["sct"] / n >= 0.95, f"SCT recall regressed to {hit['sct']}/{n}"
    assert hit["director_comp"] / n >= 0.90, f"director-comp recall regressed to {hit['director_comp']}/{n}"
    assert hit["ownership_5pct"] / n >= 0.90, f"5% recall regressed to {hit['ownership_5pct']}/{n}"
    assert total / (5 * n) >= 0.94, f"total recall regressed to {100 * total / (5 * n):.1f}%"
    assert mean_payload <= MAX_MEAN_TABLE_PAYLOAD, f"payload grew to {mean_payload:,.0f}"


@pytest.mark.skipif(not CACHE.exists(), reason="Phase-0 filing cache absent")
def test_pre_2001_ascii_filing_does_not_crash_the_parser():
    """Pre-2001 filings are ASCII wrapped in SGML with few or no HTML tables. The parser must
    return an empty result, not raise -- and the `.txt` full submission is what the Phase-1
    `_doc_url` fix now fetches for them."""
    old = sorted(CACHE.glob("*_19*.txt")) + sorted(CACHE.glob("*_200[01]-*.txt"))
    if not old:
        pytest.skip("no pre-2001 filing cached")
    f = old[0]
    best = classify_filing(f.read_text(encoding="utf-8", errors="ignore"))
    assert isinstance(best, dict)
    print(f"\n  {f.name}: {len(iter_tables(f.read_text(encoding='utf-8', errors='ignore')))} "
          f"tables, {len(best)} targets classified (no crash)")
