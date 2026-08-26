"""extract_employee_count: robust headcount parsing from 10-K text. Covers the phrasings
that the strict '<number> employees' form missed (real cases: AMZN/MCD/XOM/KO/QCOM/DVA/CAT)."""
from __future__ import annotations

from src.data_extract.utils.common.edgar_extract import extract_employee_count


def test_employee_count_phrasings():
    cases = {
        # forward, simple
        "As of Oct 31, 2020, we had approximately 21,400 employees worldwide.": 21400,
        # forward with intervening qualifiers (AMZN)
        "As of December 31, 2022, we employed approximately 1,541,000 full-time and part-time employees.": 1_541_000,
        # number AFTER the noun (MCD)
        "The number of Company employees, including restaurant employees, was approximately 205,000 as of year-end.": 205_000,
        # thousand multiplier, list form, noun-before-number (XOM)
        "The number of regular employees was 62.3, 63.0, and 64.0 thousand at years ended 2022, 2021, and 2020.": 62_300,
        # two-year comparative -> take the CURRENT (first) number (KO)
        "As of December 31, 2025 and 2024, our Company had approximately 65,900 and 69,700 employees, respectively.": 65_900,
        # 'workers' noun with 'temporary' qualifier (QCOM)
        "Human Capital. At September 28, 2025, we had approximately 52,000 full-time, part-time and temporary workers.": 52_000,
        # 'teammates' noun (DVA)
        "As of December 31, 2025, we had approximately 76,000 teammates.": 76_000,
        # 'full-time persons' (CAT)
        "We employed about 118,000 full-time persons at year-end.": 118_000,
        # 'staff members' (AMGN and many biotechs)
        "Human Resources. As of December 31, 2017, Amgen had approximately 20,800 staff members.": 20_800,
        # 'number of employees N' table row, no connector (NSC)
        "The following table shows the average number of employees 30,456 29,482 30,103.": 30_456,
    }
    print("\n=== SANITY CHECK: employee-count phrasings ===")
    for text, expected in cases.items():
        got = extract_employee_count(text)
        assert got == expected, f"{text!r} -> {got}, expected {expected}"
        print(f"  {expected:>9} <- {text[:58]}...")
    print("  All phrasings parsed correctly. Validated.")


def test_employee_count_rejects_subsets_and_noise():
    """Union/pension SUBSETS and benefit-plan money are not the headcount; the real total
    must win over them, and pure-noise text yields None."""
    # subset distractor present alongside the real total -> total wins
    txt = ("As of December 31, 2024, we had approximately 130,000 employees worldwide. "
           "Approximately 400 employees in North America were covered by collective bargaining.")
    assert extract_employee_count(txt) == 130_000

    # benefit-plan money near 'employees' must not be taken as a headcount
    assert extract_employee_count("The 401(k) plan held $205,000 for employees.") is None
    # no headcount anywhere
    assert extract_employee_count("We originated approximately 55,900 residential mortgage loans.") is None
    print("\n=== SANITY CHECK: subset / noise rejection ===")
    print("  union-subset (400) rejected for the 130,000 total; benefit money & loan counts -> None. Validated.")
