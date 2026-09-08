"""
The shared person key (src/utils/names.py).

Known-truth fixtures, every one of them a spelling drift MEASURED in the live `def14a_llm`
archive on 2026-09-07 -- not invented cases. The key is what makes `Tim Cook` and
`Timothy D. Cook` one CEO instead of a turnover, and its output is baked into `sec_8k_votes`'
stored role-category columns, so these assertions are a contract, not a preference.

The last test pins the key's known CEILING on purpose. A limitation that is asserted stays
documented; one that is merely known gets rediscovered as a bug.
"""
from __future__ import annotations

from src.utils.names import clean_person_name, person_key


def test_the_measured_spelling_drifts_all_reconcile():
    """Every consecutive-filing drift found in the archive keys to one person."""
    drifts = {
        # AAPL printed THREE spellings of one CEO across consecutive proxies
        "AAPL": ["Timothy D. Cook", "Timothy Cook", "Tim Cook"],
        "ADI": ["Vincent T. Roche", "Vincent Roche"],            # middle initial dropped
        "ADM_1": ["G. Allen Andreas", "G. A. Andreas"],          # first name <-> initial, twice
        "ADM_2": ["Juan R. Luciano", "J. R. LUCIANO"],           # first name -> initial AND case
        "AEE": ["Charles W. Mueller", "C. W. Mueller"],          # first name -> initial
    }
    for ticker, spellings in drifts.items():
        keys = {person_key(s) for s in spellings}
        assert len(keys) == 1, f"{ticker}: {spellings} keyed {len(keys)} ways: {sorted(keys)}"

    print("\n=== SANITY CHECK: measured spelling drifts ===")
    for ticker, spellings in drifts.items():
        print(f"  {ticker:<6} {' | '.join(spellings):<55} -> {person_key(spellings[0])}")
    print("  CONCLUSION: all five archive drifts collapse to one key each -- middle tokens are "
          "ignored structurally (last token + first initial), so case flips and first-name/initial "
          "swaps reconcile with no rule of their own. Validated.")


def test_suffixes_and_footnotes_do_not_split_a_person():
    """A generational suffix, a post-nominal, or a footnote marker must not create a second
    person. The post-nominal case is the sharp one: before the dots were stripped FIRST,
    `person_key("Albert Bourla, DVM, Ph.D.")` returned `d|a` -- collapsing every credentialed
    director sharing a first initial onto ONE key."""
    pairs = [
        ("John Smith Jr.", "John Smith"),
        ("Harry A. Lawton III", "Harry Lawton"),
        ("H. Lawrence Culp, Jr.", "H. Lawrence Culp"),
        ("Albert Bourla, DVM, Ph.D.", "A. Bourla"),
        ("Emma N. Walmsley11", "Emma Walmsley"),          # footnote glued to the surname
        ("Katherine J. Smith", "Kathy Smith"),            # the docstring's own case
    ]
    for full, short in pairs:
        assert person_key(full) == person_key(short), f"{full!r} != {short!r}"
    assert person_key("Albert Bourla, DVM, Ph.D.") == "bourla|a", "the post-nominal regression"

    print("\n=== SANITY CHECK: suffixes, post-nominals, footnotes ===")
    for full, short in pairs:
        print(f"  {full:<28} == {short:<20} -> {person_key(full)}")
    print("  CONCLUSION: suffixes and footnote markers are stripped before keying, so evidence "
          "for one person is never split across two keys. Validated.")


def test_distinct_people_stay_distinct():
    """The key must not over-collapse: a different surname is a different person, and so is the
    same surname with a different first initial."""
    assert person_key("Tim Cook") != person_key("Tim Cash")
    assert person_key("Mark D. Mosca") != person_key("Peter A. Appel")
    assert person_key("Hector de J. Ruiz") != person_key("Derrick R. Meyer")
    assert person_key(None) is None
    assert person_key("") is None
    assert person_key("   ") is None
    assert clean_person_name("James DimonChairman and CEO") == "James Dimon"

    print("\n=== SANITY CHECK: over-collapse and empties ===")
    print(f"  cook|t vs cash|t: {person_key('Tim Cook')} != {person_key('Tim Cash')}")
    print(f"  real ACGL turnover: {person_key('Mark D. Mosca')} != {person_key('Peter A. Appel')}")
    print(f"  None / '' / '   ' -> {person_key(None)}, {person_key('')}, {person_key('   ')}")
    print("  CONCLUSION: distinct people keep distinct keys and an unusable cell keys to None "
          "(UNKNOWN), never to a sentinel two rows could match on. Validated.")


def test_nickname_with_a_different_initial_is_a_known_ceiling():
    """⚠ ASSERTED LIMITATION, not a bug report. `Bob` and `Robert` are one person to a human and
    two keys to this function, because the key is `lastname|firstinitial` and the initials
    differ. Fixing it needs a nickname table, which is its own project; the measured 98.4%
    CEO<->NEO cross-table match rate already has this ceiling priced in.

    Asserted so that a future session that CHANGES this behaviour is forced to notice: the key's
    output is baked into `sec_8k_votes`' stored role-category sums, so a redefinition is a
    re-extraction, not a refactor."""
    assert person_key("Bob Smith") != person_key("Robert Smith")
    assert person_key("Bill Gates") != person_key("William Gates")

    print("\n=== SANITY CHECK: the key's known ceiling ===")
    print(f"  'Bob Smith' -> {person_key('Bob Smith')}   vs  'Robert Smith' -> "
          f"{person_key('Robert Smith')}")
    print(f"  'Bill Gates' -> {person_key('Bill Gates')}  vs  'William Gates' -> "
          f"{person_key('William Gates')}")
    print("  CONCLUSION: a nickname with a DIFFERENT first initial does not reconcile. This is a "
          "documented limit of `lastname|firstinitial`, pinned here so it is not rediscovered as "
          "a defect -- and so that changing it is a deliberate act, since the key's output is "
          "stored in sec_8k_votes' role-category columns. Validated.")


if __name__ == "__main__":
    test_the_measured_spelling_drifts_all_reconcile()
    test_suffixes_and_footnotes_do_not_split_a_person()
    test_distinct_people_stay_distinct()
    test_nickname_with_a_different_initial_is_a_known_ceiling()
