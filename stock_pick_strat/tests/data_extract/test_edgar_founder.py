"""Regression tests for founder / founder-CEO detection in the 10-K executive
officers parser (edgar_extract.extract_executive_officers).

Before the fix `founder_ceo` was 0 for every filing because it only matched the
literal substring "founder" in the 80-char position line -- missing the far more
common "founded" / "co-founded" phrasing in the bio, and missing comma-separated
prose blurbs entirely. These cases lock in the fix and guard the main false
positive (a non-founder CEO whose bio merely states when the company was founded).
"""
from __future__ import annotations

from src.data_extract.utils.common.edgar_extract import extract_executive_officers, _is_founder


# --- realistic 10-K "Information about our Executive Officers" formats -------- #
_TABLE_FOUNDER_IN_POSITION = """
Information about our Executive Officers
The following table sets forth information regarding our executive officers.
Name Age Position
Andrew Chen 58 Founder, Chairman and Chief Executive Officer
Robert Lee 55 Senior Vice President and Chief Financial Officer
"""

_TABLE_PLUS_COFOUNDED_BIO = """
Information about our Executive Officers
Jensen Huang 61 President and Chief Executive Officer
Mr. Huang co-founded the Company in 1993 and has served as its President and
Chief Executive Officer since inception.
Colette Kress 56 Executive Vice President and Chief Financial Officer
"""

_APPLE_NON_FOUNDER = """
Information about our Executive Officers
Timothy D. Cook 63 Chief Executive Officer
Luca Maestri 60 Senior Vice President and Chief Financial Officer
"""

_PROSE_COMMAS_COFOUNDED = """
Information about our Executive Officers
Reed Hastings, 63, co-founded the Company in 1997 and has served as our Chairman
and Chief Executive Officer.
Spencer Neumann, 54, has served as our Chief Financial Officer since 2019.
"""

_NON_FOUNDER_MENTIONS_FOUNDING_YEAR = """
Information about our Executive Officers
Sarah Brown 55 Chief Executive Officer
Ms. Brown has served as Chief Executive Officer since 2018. The Company was
founded in 1902 and is headquartered in Ohio.
David Kim 50 Chief Financial Officer
"""


def test_founder_ceo_detection_matrix():
    t1 = extract_executive_officers(_TABLE_FOUNDER_IN_POSITION)
    t2 = extract_executive_officers(_TABLE_PLUS_COFOUNDED_BIO)
    t3 = extract_executive_officers(_APPLE_NON_FOUNDER)
    t4 = extract_executive_officers(_PROSE_COMMAS_COFOUNDED)
    t5 = extract_executive_officers(_NON_FOUNDER_MENTIONS_FOUNDING_YEAR)

    # "Founder" literally in the position line (already worked) stays correct
    assert t1["founder_ceo"] == 1 and t1["ceo_name"] == "Andrew Chen"

    # bio says "co-founded" -> now detected (was 0 before: "founded" != "founder")
    assert t2["founder_ceo"] == 1 and t2["ceo_name"] == "Jensen Huang"

    # ordinary non-founder CEO -> 0
    assert t3["founder_ceo"] == 0 and t3["ceo_name"] == "Timothy D. Cook"

    # prose with commas -> officers now parsed AND founder-CEO detected
    assert t4["n_officers"] == 2, "comma-separated 'Name, Age,' blurb was not parsed"
    assert t4["founder_ceo"] == 1 and t4["ceo_name"] == "Reed Hastings"

    # CEO bio only mentions the COMPANY's founding year -> NOT a founder
    assert t5["founder_ceo"] == 0, "passive 'was founded in 1902' wrongly flagged a founder"
    assert t5["ceo_name"] == "Sarah Brown"

    print("\n=== SANITY CHECK: founder-CEO detection ===")
    print(f"  Founder-in-position: {t1['ceo_name']} -> founder_ceo={t1['founder_ceo']} (1)")
    print(f"  'co-founded' in bio:  {t2['ceo_name']} -> founder_ceo={t2['founder_ceo']} (1, was 0)")
    print(f"  non-founder CEO:      {t3['ceo_name']} -> founder_ceo={t3['founder_ceo']} (0)")
    print(f"  prose+commas:         {t4['ceo_name']} -> founder_ceo={t4['founder_ceo']} (1, was 0; "
          f"n_officers={t4['n_officers']})")
    print(f"  passive founding yr:  {t5['ceo_name']} -> founder_ceo={t5['founder_ceo']} (0, false-positive guarded)")
    print("  founder_ceo now fires on real 'founded/co-founded' phrasing; passive company-history excluded. Validated.")


def test_is_founder_helper_word_boundaries():
    # active founder phrasings -> True
    assert _is_founder("co-founded the Company in 1993")
    assert _is_founder("our Founder and Chief Executive Officer")
    assert _is_founder("he founded the business in 1980")
    assert _is_founder("a co-founder of the firm")
    # non-founder / false-positive traps -> False
    assert not _is_founder("Chief Executive Officer")
    assert not _is_founder("the Company was founded in 1902")          # passive company history
    assert not _is_founder("established a strong foundation for growth")  # 'foundation' != founder
    assert not _is_founder("has a profound impact on strategy")          # 'profound' != founder

    print("\n=== SANITY CHECK: _is_founder word boundaries ===")
    print("  matches co-founded/founder/founded(active); rejects 'was founded', "
          "'foundation', 'profound'. Validated.")
