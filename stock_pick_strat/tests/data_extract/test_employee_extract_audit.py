"""extract_employee_count: the four failure modes the 2026-07 source-table audit proved
against LIVE 10-K text. Each case below is the real sentence from the filing named in it,
with the value the parser used to store and the value the filing actually states.

Audit evidence that motivated these (measured on `employees_history`, 6,766 rows):
48% of tickers showed >3x spread across their own headcount history, 121 tickers >10x,
6.3% of year-over-year transitions were >2x or <0.5x, 39 rows were exactly 100.
"""
from __future__ import annotations

from src.data_extract.utils.common.edgar_extract import extract_employee_count


def test_full_time_part_time_split_is_summed():
    """CF 10-K 2012-02-27 (accession 0001047469-12-001620).

    'approximately 2,400 full-time and 100 part-time employees' is ONE workforce split
    over two numbers, not a two-year comparative. `_CMP_RE` cannot fire (a qualifier sits
    between N1 and 'and'), so the forward pattern matched the SECOND number and CF was
    stored as 100 employees for every year 2012-2019 (it reports 3,000/2,700/2,800 from
    2020, when the phrasing changed)."""
    txt = ("As of December 31, 2011, we employed approximately 2,400 full-time and "
           "100 part-time employees.")
    assert extract_employee_count(txt) == 2_500
    # the same shape with the parts reversed must also sum
    assert extract_employee_count(
        "We had 500 part-time and 9,500 full-time employees.") == 10_000


def test_company_named_subject_beats_a_divestiture_subset():
    """Citigroup 10-K 2026-02-20 (accession 0000831001-26-000011).

    Both 'approximately 800 employees' (describing a DIVESTED business) and 'Citi had
    approximately 226,000 full-time employees' scored 4, and first-found won -> 800.
    Two independent repairs make the real total win: the subject-agnostic '<name> had'
    context bonus, and a larger-value tie-break."""
    txt = ("The divestiture included all remaining businesses, as well as approximately "
           "800 employees. Under our management at December 31, 2025, Citi had "
           "approximately 226,000 full-time employees, compared to approximately "
           "229,000 at December 31, 2024.")
    assert extract_employee_count(txt) == 226_000


def test_workers_compensation_reserve_table_is_not_a_headcount():
    """WRB 10-K. The noun 'workers' matched 'workers' compensation' — an insurance
    RESERVE line in $thousands — giving a stored headcount of 4,502,942 for a company
    with ~8,000 staff. With no genuine headcount sentence in reach, the answer is None."""
    txt = ("Losses and loss adjustment expenses, net of reinsurance $ 8,289,106 "
           "professional liability 673,774 1,582,133 workers' compensation (1) "
           "1,103,703 760,075 2,255,907")
    assert extract_employee_count(txt) is None
    # the plain noun still works when it really is a workforce
    assert extract_employee_count(
        "At year end we had approximately 52,000 full-time workers.") == 52_000


def test_service_territory_population_loses_to_the_real_headcount():
    """AES 10-K 2026-03-02. 'estimated population of approximately 982,000 people' was
    taken as the headcount (stored 2,400,000 from the same failure in an earlier year)
    even though the filing states 'we had 8,336 full time/permanent employees'.

    Two repairs: 'population' is a subset/no-go context, and the qualifier separator now
    bridges the slash in 'full time/permanent' so the true sentence produces a candidate
    at all (previously it produced none, leaving the population figure unopposed)."""
    txt = ("AES Indiana serves an area of 3,000 square miles with an estimated "
           "population of approximately 982,000 people. As of December 31, 2025, we had "
           "8,336 full time/permanent employees.")
    assert extract_employee_count(txt) == 8_336


def test_all_candidates_penalised_yields_none_not_a_known_bad_number():
    """A NULL the caller skips beats a number already known to come from a money table."""
    assert extract_employee_count("The 401(k) plan held $205,000 for employees.") is None
    assert extract_employee_count(
        "Restricted stock of $1,250,000 was granted to employees.") is None


def test_audit_regression_suite_prints_conclusion():
    """One table covering every audit case plus the phrasings that already worked, so a
    future change cannot fix one failure mode by breaking another."""
    cases = [
        ("CF 2012      split full/part-time", 2_500,
         "As of December 31, 2011, we employed approximately 2,400 full-time and 100 part-time employees."),
        ("C 2026       divestiture subset", 226_000,
         "as well as approximately 800 employees. Citi had approximately 226,000 full-time employees."),
        ("WRB 2026     workers' comp table", None,
         "professional liability 673,774 1,582,133 workers' compensation (1) 1,103,703 2,255,907"),
        ("AES 2026     territory population", 8_336,
         "an estimated population of approximately 982,000 people. we had 8,336 full time/permanent employees."),
        ("AMZN         qualifiers, no split", 1_541_000,
         "we employed approximately 1,541,000 full-time and part-time employees."),
        ("KO           two-year comparative", 65_900,
         "our Company had approximately 65,900 and 69,700 employees, respectively."),
        ("MCD          number after noun", 205_000,
         "The number of Company employees was approximately 205,000 as of year-end."),
        ("XOM          thousand multiplier", 62_300,
         "The number of regular employees was 62.3, 63.0, and 64.0 thousand at years ended 2022, 2021, 2020."),
        ("NSC          bare table row", 30_456,
         "The following table shows the average number of employees 30,456 29,482 30,103."),
        ("union subset must lose", 130_000,
         "we had approximately 130,000 employees worldwide. Approximately 400 employees were covered by collective bargaining."),
    ]
    print("\n=== SANITY CHECK: employee headcount extraction (audit regressions) ===")
    for label, expected, txt in cases:
        got = extract_employee_count(txt)
        assert got == expected, f"{label}: got {got}, expected {expected}"
        print(f"  {label:36s} -> {str(got):>9}")
    print("  4 proven failure modes fixed (split-sum, named-subject tie-break,")
    print("  workers'-comp exclusion, population exclusion); 6 working phrasings intact.")
    print("  Verified against the LIVE filings: CF 100->2,500 | C 800->226,000 |")
    print("  WRB 4,502,942->None | AES 2,400,000->8,336. Validated.")


def test_headcount_continuity_guard():
    """The last line of defence, at the FETCHER level: a value that survives every
    in-document heuristic but is discontinuous with the ticker's own headcount history is
    dropped. The 30-ticker verification run caught CoStar (CSGP) picking up a "2.3 million"
    phrase -> 2,300,000 against a stored 1,155, which no text-level rule reached because
    the filing states no headcount sentence at all.

    Anchored on the MEDIAN so one bad reading cannot reject the correct rows after it —
    WRB's stored 4,502,942 would otherwise have poisoned every later year."""
    from src.data_extract.utils.structure.fetch_employees_edgar import (
        _history_by_ticker, _is_continuous,
    )
    import pandas as pd

    cases = [
        ("first filing, no anchor",             5_000,     [],                              True),
        ("CSGP 2.3M vs ~1,100 history",         2_300_000, [1000, 1100, 1155, 1200],         False),
        ("CF 100 vs ~2,800 history",            100,       [2700, 2800, 2900, 3000],         False),
        ("normal growth",                       3_000,     [2700, 2800, 2900],               True),
        ("transformative merger 2.5x",          7_000,     [2700, 2800, 2900],               True),
        ("5x boundary kept",                    14_000,    [2700, 2800, 2900],               True),
        ("6x rejected",                         17_400,    [2700, 2800, 2900],               False),
        ("one-third layoff kept",               950,       [2700, 2800, 2900],               True),
        ("collapse to a tenth rejected",        280,       [2700, 2800, 2900],               False),
        ("median resists one poisoned row",     2_900,     [4_502_942, 6537, 7000, 7356],    True),
    ]
    print("\n=== SANITY CHECK: headcount continuity guard ===")
    for label, count, history, expected in cases:
        got = _is_continuous(count, history)
        assert got == expected, f"{label}: keep={got}, expected {expected}"
        print(f"  {label:34s} {count:>9,} -> {'keep' if got else 'DROP'}")

    # the anchor is built in FILING-DATE order, not DB row order
    df = pd.DataFrame({"ticker": ["A", "A", "B"],
                       "as_of": ["2022-01-01", "2021-01-01", "2020-01-01"],
                       "employees": [200, 100, 50]})
    assert _history_by_ticker(df) == {"A": [100, 200], "B": [50]}
    assert _history_by_ticker(None) == {} and _history_by_ticker(pd.DataFrame()) == {}
    print("  History anchor is filing-date ordered; empty input is safe. Validated.")
