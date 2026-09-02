# DEF 14A extraction — inspectable sample

One cached filing per ticker, extracted with the production model and prompt. **Every number below came out of the LLM**; the `doc_url` on each section is the filing it was read from, so any cell can be checked against the source.

Nothing here was written to the database.

## A — filed 2026-02-06

- source: <https://www.sec.gov/Archives/edgar/data/1090872/000119312526040286/a-20260203.htm>
- accession: `0001193125-26-040286`
- carve payload: 41,033 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Agilent |
| `fiscal_year_extract` | 2025 |
| `board_size` | 11 |
| `n_directors` | 12 |
| `pct_female_directors` | 0.333 |
| `pct_gender_stated` | 0.5 |
| `n_women_directors_vs_inferred` | — |
| `avg_director_age` | 66 |
| `avg_board_tenure` | 7.9 |
| `pct_independent_directors` | 0.909 |
| `ceo_name_proxy` | Padraig McDonnell |
| `ceo_age` | 54 |
| `ceo_since_year` | 2024 |
| `ceo_salary` | 1,141,345 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 8,169,850 |
| `ceo_option_awards` | 2,017,759 |
| `ceo_non_equity_incentive` | 0 |
| `ceo_all_other_comp` | 61,473 |
| `ceo_total_comp` | 12,827,236 |
| `ceo_equity_pay_pct` | 0.794 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 31,806,094 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.89 |
| `ceo_pay_ratio` | 155 |
| `median_employee_pay` | 82,873 |
| `independent_chair` | 1 |
| `lead_independent_director` | — |
| `classified_board` | 1 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | PricewaterhouseCoopers LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 5,409,205 |
| `audit_fees_audit` | 5,203,000 |
| `audit_fees_audit_related` | 200,000 |
| `audit_fees_tax` | 4,205 |
| `audit_fees_other` | 2,000 |
| `auditor_fees_prior` | 5,769,918 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Padraig McDonnell | President; Chief Executive Officer | 2025 | 1,141,345 | 0 | 8,169,850 | 2,017,759 | 0 | 1,436,809 | 61,473 | 12,827,236 | 1 |
| Padraig McDonnell | President; Chief Executive Officer | 2024 | 869,962 | 0 | 5,981,631 | 1,504,289 | 0 | 669,309 | 21,314 | 9,046,505 | 1 |
| Padraig McDonnell | President; Chief Executive Officer | 2023 | 606,231 | 0 | 1,777,128 | 436,092 | 0 | 316,363 | 137,084 | 3,272,898 | 1 |
| Rodney Gonsalves | Vice President; Controller; Principal Accounting Officer; Interim Chief Financial Officer | 2025 | 513,559 | 0 | 1,687,786 | 175,921 | 0 | 291,638 | 26,883 | 2,695,787 | 1 |
| Simon May | Senior Vice President, Agilent; President Life Sciences and Diagnostics Market Group | 2025 | 635,385 | 500,000 | 1,801,424 | 444,959 | 0 | 472,395 | 17,406 | 3,871,569 | 1 |
| Simon May | Senior Vice President, Agilent; President Life Sciences and Diagnostics Market Group | 2024 | 276,923 | 500,000 | 2,264,042 | 564,837 | 0 | 92,086 | 13,846 | 3,711,734 | 1 |
| Angelica Riemann | Senior Vice President, Agilent; President Agilent Cross-Lab Group | 2025 | 591,346 | 0 | 1,675,932 | 413,890 | 0 | 520,454 | 12,956 | 3,214,578 | 1 |
| Bret DiMarco | Senior Vice President, Chief Legal Officer | 2025 | 679,808 | 0 | 1,990,228 | 491,517 | 0 | 475,466 | 21,574 | 3,658,593 | 1 |
| Robert McMahon | Former Senior Vice President; Former Chief Financial Officer | 2025 | 665,000 | 0 | 4,000,968 | 827,780 | 0 | 0 | 44,583 | 5,538,331 | 1 |
| Robert McMahon | Former Senior Vice President; Former Chief Financial Officer | 2024 | 756,654 | 0 | 6,869,915 | 845,947 | 0 | 324,587 | 84,037 | 8,881,140 | 1 |
| Robert McMahon | Former Senior Vice President; Former Chief Financial Officer | 2023 | 726,769 | 0 | 2,999,006 | 735,873 | 0 | 259,599 | 89,008 | 4,810,255 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Mala Anand | 2025 | 105,000 | 220,426 | — | — | — | — | 325,426 | 1 |
| Otis W. Brawley, M.D. | 2025 | 105,000 | 220,426 | — | — | — | — | 325,426 | 1 |
| Judy Gawlik Brown | 2025 | 81,986 | 182,642 | — | — | — | — | 269,149 | 0 |
| Mikael Dolsten, M.D., Ph.D. | 2025 | 105,000 | 220,426 | — | — | — | — | 335,426 | 0 |
| Koh Boon Hwee | 2025 | 105,000 | 220,426 | — | — | — | — | 480,426 | 0 |
| Heidi Kunz | 2025 | 105,000 | 220,426 | — | — | — | — | 325,426 | 1 |
| Daniel K. Podolsky, M.D. | 2025 | 105,000 | 220,426 | — | — | — | — | 335,426 | 0 |
| Sue H. Rataj | 2025 | 105,000 | 220,426 | — | — | — | — | 335,426 | 0 |
| George A. Scangos, Ph.D. | 2025 | 105,000 | 220,426 | — | — | — | — | 345,426 | 0 |
| Pascal Soriot | 2025 | 81,986 | 182,642 | — | — | — | — | 264,628 | 1 |
| Dow R. Wilson | 2025 | 105,000 | 220,426 | — | — | — | — | 350,426 | 0 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 33,446,526 | 0.118 |
| BlackRock, Inc. | 5pct_holder | 23,626,346 | 0.084 |
| Mala Anand | director_officer | 14,547 | — |
| Otis Brawley, M.D. | director_officer | 10,807 | — |
| Judy Gawlik Brown | director_officer | 1,677 | — |
| Bret DiMarco | director_officer | 5,106 | — |
| Mikael Dolsten, M.D., PhD | director_officer | 4,973 | — |
| Rodney Gonsalves | director_officer | 32,546 | — |
| Koh Boon Hwee | director_officer | 61,647 | — |
| Simon May | director_officer | 7,100 | — |
| Padraig McDonnell | director_officer | 54,582 | — |
| Robert W. McMahon | director_officer | 116,707 | — |
| Daniel K. Podolsky, M.D. | director_officer | 35,954 | — |
| Sue H. Rataj | director_officer | 21,459 | — |
| Angelica Riemann | director_officer | 30,493 | — |
| George A. Scangos, PhD | director_officer | 38,499 | — |
| Pascal Soriot | director_officer | 1,173 | — |
| Dow R. Wilson | director_officer | 18,176 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Judy Gawlik Brown | 57 | 1 | 1 | female | honorific | 1 |
| Sue H. Rataj | 69 | 10 | 1 | female | honorific | 0 |
| George A. Scangos, Ph.D. | 76 | 14 | 1 | male | pronoun | 1 |
| Dow R. Wilson | 66 | 7 | 1 | male | honorific | 1 |
| Mala Anand | 58 | 6 | 1 | female | honorific | 0 |
| Koh Boon Hwee | 75 | 22 | 1 | male | honorific | 1 |
| Padraig McDonnell | 54 | 1 | 0 | male | honorific | 0 |
| Daniel K. Podolsky, M.D. | 72 | 10 | 1 | male | name | 0 |
| Otis W. Brawley, M.D. | 66 | 4 | 1 | male | name | 3 |
| Mikael Dolsten, M.D., Ph.D. | 67 | 4 | 1 | male | name | 1 |
| Pascal Soriot | — | — | 1 | male | name | — |
| Heidi Kunz | — | — | 1 | female | name | — |

---

## AAPL — filed 2026-01-08

- source: <https://www.sec.gov/Archives/edgar/data/320193/000130817926000008/aapl014016-def14a.htm>
- accession: `0001308179-26-000008`
- carve payload: 40,396 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Apple Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 8 |
| `n_directors` | 8 |
| `pct_female_directors` | 0.5 |
| `pct_gender_stated` | 0.625 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 69.125 |
| `avg_board_tenure` | 11.375 |
| `pct_independent_directors` | 0 |
| `ceo_name_proxy` | Tim Cook |
| `ceo_age` | 65 |
| `ceo_since_year` | 2011 |
| `ceo_salary` | 3,000,000 |
| `ceo_bonus` | — |
| `ceo_stock_awards` | 57,535,293 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 12,000,000 |
| `ceo_all_other_comp` | 1,759,518 |
| `ceo_total_comp` | 74,294,811 |
| `ceo_equity_pay_pct` | 0.774 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 193,356,600 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.92 |
| `ceo_pay_ratio` | 533 |
| `median_employee_pay` | 139,483 |
| `independent_chair` | 1 |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | Ernst & Young |
| `auditor_since_year` | — |
| `auditor_fees` | 34,277,000 |
| `audit_fees_audit` | 24,703,000 |
| `audit_fees_audit_related` | 2,274,000 |
| `audit_fees_tax` | 4,533,000 |
| `audit_fees_other` | 2,767,000 |
| `auditor_fees_prior` | 30,271,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tim Cook | Chief Executive Officer | 2025 | 3,000,000 | — | 57,535,293 | — | 12,000,000 | — | 1,759,518 | 74,294,811 | 1 |
| Tim Cook | Chief Executive Officer | 2024 | 3,000,000 | — | 58,088,946 | — | 12,000,000 | — | 1,520,856 | 74,609,802 | 1 |
| Tim Cook | Chief Executive Officer | 2023 | 3,000,000 | — | 46,970,283 | — | 10,713,450 | — | 2,526,112 | 63,209,845 | 1 |
| Kevan Parekh | Senior Vice President, Chief Financial Officer | 2025 | 891,519 | — | 18,433,135 | — | 3,120,317 | — | 22,338 | 22,467,309 | 1 |
| Kate Adams | Senior Vice President, General Counsel and Secretary | 2025 | 1,000,000 | — | 22,009,766 | — | 4,000,000 | — | 22,482 | 27,032,248 | 1 |
| Kate Adams | Senior Vice President, General Counsel and Secretary | 2024 | 1,000,000 | — | 22,157,075 | — | 4,000,000 | — | 22,182 | 27,179,257 | 1 |
| Kate Adams | Senior Vice President, General Counsel and Secretary | 2023 | 1,000,000 | — | 22,323,641 | — | 3,571,150 | — | 46,914 | 26,941,705 | 1 |
| Sabih Khan | Senior Vice President, Chief Operating Officer | 2025 | 1,000,000 | — | 22,009,766 | — | 4,000,000 | — | 21,905 | 27,031,671 | 1 |
| Luca Maestri | Former Senior Vice President, Chief Financial Officer | 2025 | 819,231 | — | 13,003,031 | — | 1,638,462 | — | 22,204 | 15,482,928 | 1 |
| Luca Maestri | Former Senior Vice President, Chief Financial Officer | 2024 | 1,000,000 | — | 22,157,075 | — | 4,000,000 | — | 22,182 | 27,179,257 | 1 |
| Luca Maestri | Former Senior Vice President, Chief Financial Officer | 2023 | 1,000,000 | — | 22,323,641 | — | 3,571,150 | — | 41,092 | 26,935,883 | 1 |
| Deirdre O’Brien | Senior Vice President, Retail + People | 2025 | 1,000,000 | — | 22,009,766 | — | 4,000,000 | — | 37,867 | 27,047,633 | 1 |
| Deirdre O’Brien | Senior Vice President, Retail + People | 2024 | 1,000,000 | — | 22,157,075 | — | 4,000,000 | — | 22,182 | 27,179,257 | 1 |
| Deirdre O’Brien | Senior Vice President, Retail + People | 2023 | 1,000,000 | — | 22,323,641 | — | 3,571,150 | — | 42,219 | 26,937,010 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Wanda Austin | 2025 | 100,000 | 310,035 | — | — | — | 2,815 | 412,850 | 1 |
| Alex Gorsky | 2025 | 100,000 | 310,035 | — | — | — | 6,457 | 416,492 | 1 |
| Andrea Jung | 2025 | 140,000 | 310,035 | — | — | — | 7,985 | 458,020 | 1 |
| Art Levinson | 2025 | 275,000 | 274,956 | — | — | — | 7,275 | 557,231 | 1 |
| Monica Lozano | 2025 | 100,000 | 310,035 | — | — | — | 2,921 | 412,956 | 1 |
| Ron Sugar | 2025 | 145,000 | 310,035 | — | — | — | 16,248 | 471,283 | 1 |
| Sue Wagner | 2025 | 135,000 | 310,035 | — | — | — | 338 | 445,373 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 1,415,826,462 | 0.0963 |
| BlackRock, Inc. | 5pct_holder | 1,043,713,019 | 0.071 |
| Kate Adams | director_officer | 175,408 | — |
| Wanda Austin | director_officer | 2,843 | — |
| Tim Cook | director_officer | 3,280,295 | — |
| Alex Gorsky | director_officer | 6,794 | — |
| Andrea Jung | director_officer | 77,664 | — |
| Sabih Khan | director_officer | 1,074,404 | — |
| Art Levinson | director_officer | 4,126,689 | — |
| Monica Lozano | director_officer | 9,862 | — |
| Luca Maestri | director_officer | 91,304 | — |
| Deirdre O’Brien | director_officer | 136,687 | — |
| Kevan Parekh | director_officer | 8,765 | — |
| Ron Sugar | director_officer | 110,566 | — |
| Sue Wagner | director_officer | 69,788 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Art Levinson | 75 | 25 | — | male | name | 0 |
| Tim Cook | 65 | 14 | 0 | male | honorific | 1 |
| Wanda Austin | 71 | 1 | — | female | name | 2 |
| Alex Gorsky | 65 | 4 | — | male | honorific | 2 |
| Andrea Jung | 67 | 17 | — | female | honorific | 1 |
| Monica Lozano | 69 | 4 | — | female | honorific | 2 |
| Ron Sugar | 77 | 15 | — | male | name | 1 |
| Sue Wagner | 64 | 11 | — | female | honorific | 2 |

---

## AEE — filed 2026-03-31

- source: <https://www.sec.gov/Archives/edgar/data/1002910/000110465926037756/tm261401-1_def14a.htm>
- accession: `0001104659-26-037756`
- carve payload: 42,241 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Ameren Corporation |
| `fiscal_year_extract` | 2025 |
| `board_size` | 12 |
| `n_directors` | 16 |
| `pct_female_directors` | 0.375 |
| `pct_gender_stated` | 0.812 |
| `n_women_directors_vs_inferred` | — |
| `avg_director_age` | 62.75 |
| `avg_board_tenure` | 7.583 |
| `pct_independent_directors` | 0.917 |
| `ceo_name_proxy` | Martin J. Lyons, Jr. |
| `ceo_age` | 59 |
| `ceo_since_year` | 2022 |
| `ceo_salary` | 1,325,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 8,205,550 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 3,330,000 |
| `ceo_all_other_comp` | 610,563 |
| `ceo_total_comp` | 14,056,510 |
| `ceo_equity_pay_pct` | 0.584 |
| `n_neos` | 7 |
| `sct_years` | 3 |
| `total_neo_comp` | 34,916,953 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 5 |
| `say_on_pay_support_pct` | — |
| `ceo_pay_ratio` | 112 |
| `median_employee_pay` | 125,070 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | PricewaterhouseCoopers LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 5,441,000 |
| `audit_fees_audit` | 5,134,000 |
| `audit_fees_audit_related` | 100,000 |
| `audit_fees_tax` | 205,000 |
| `audit_fees_other` | 2,000 |
| `auditor_fees_prior` | 4,953,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Martin J. Lyons, Jr. | Chairman, President and Chief Executive Officer, Ameren | 2025 | 1,325,000 | 0 | 8,205,550 | — | 3,330,000 | 585,397 | 610,563 | 14,056,510 | 1 |
| Martin J. Lyons, Jr. | Chairman, President and Chief Executive Officer, Ameren | 2024 | 1,275,000 | 0 | 5,209,678 | — | 2,412,000 | 657,183 | 177,169 | 9,731,030 | 1 |
| Martin J. Lyons, Jr. | Chairman, President and Chief Executive Officer, Ameren | 2023 | 1,200,000 | 0 | 5,121,903 | — | 1,750,000 | 763,434 | 174,094 | 9,009,431 | 1 |
| Michael L. Moehn | Group President, Ameren Utilities, Ameren | 2025 | 895,000 | 0 | 3,491,892 | — | 1,492,400 | 248,193 | 255,709 | 6,383,194 | 1 |
| Michael L. Moehn | Group President, Ameren Utilities, Ameren | 2024 | 860,000 | 0 | 2,330,333 | — | 1,106,300 | 447,911 | 115,437 | 4,859,981 | 1 |
| Michael L. Moehn | Group President, Ameren Utilities, Ameren | 2023 | 825,000 | 0 | 7,788,803 | — | 887,900 | 508,537 | 114,614 | 10,124,854 | 1 |
| Leonard P. Singh | Executive Vice President and Chief Financial Officer, Ameren | 2025 | 650,000 | 0 | 1,730,861 | — | 879,700 | 171,913 | 159,268 | 3,591,742 | 1 |
| Leonard P. Singh | Executive Vice President and Chief Financial Officer, Ameren | 2024 | 625,000 | 0 | 1,129,033 | — | 723,800 | 172,700 | 77,337 | 2,727,870 | 1 |
| Leonard P. Singh | Executive Vice President and Chief Financial Officer, Ameren | 2023 | 585,000 | 250,000 | 1,086,882 | — | 565,700 | 110,328 | 104,772 | 2,702,682 | 1 |
| Shawn E. Schukar | Chairman and President, Ameren Transmission Company of Illinois | 2025 | 545,000 | 0 | 1,046,265 | — | 691,500 | 268,669 | 94,766 | 2,646,200 | 1 |
| Mark C. Lindgren | Executive Vice President, Communications and Chief Human Resources Officer, Ameren Services | 2025 | 482,500 | 0 | 776,898 | — | 520,500 | 109,588 | 81,774 | 1,971,260 | 1 |
| Mark C. Birk | Former Chairman and President, Ameren Missouri | 2025 | 634,646 | 0 | 1,883,848 | — | 0 | 263,679 | 78,045 | 2,860,218 | 1 |
| Mark C. Birk | Former Chairman and President, Ameren Missouri | 2024 | 650,000 | 0 | 1,174,177 | — | 787,000 | 290,634 | 72,006 | 2,973,817 | 1 |
| Mark C. Birk | Former Chairman and President, Ameren Missouri | 2023 | 610,000 | 0 | 1,225,254 | — | 617,900 | 369,238 | 70,235 | 2,892,627 | 1 |
| Fadi M. Diya | Former Senior Vice President and Chief Nuclear Officer, Ameren Missouri | 2025 | 434,473 | 0 | 1,567,769 | — | 0 | 241,861 | 1,163,726 | 3,407,829 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Cynthia J. Brinkley | — | 145,000 | 170,060 | — | — | — | — | 315,060 | 1 |
| Catherine S. Brune | — | 145,000 | 170,060 | — | — | — | — | 315,060 | 1 |
| Ward H. Dickson | — | 145,000 | 170,060 | — | — | — | — | 315,060 | 1 |
| Noelle K. Eder | — | 60,417 | 170,060 | — | — | — | — | 230,477 | 1 |
| Ellen M. Fitzsimmons | — | 160,000 | 170,060 | — | — | — | — | 330,060 | 1 |
| Rafael Flores | — | 145,000 | 170,060 | — | — | — | — | 315,060 | 1 |
| Kimberly J. Harris | — | 31,250 | 170,060 | — | — | — | — | 201,310 | 1 |
| Richard J. Harshman | — | 145,000 | 170,060 | — | — | — | — | 315,060 | 1 |
| Craig S. Ivey | — | 125,000 | 170,060 | — | — | — | — | 295,060 | 1 |
| James C. Johnson | — | 44,097 | 170,060 | — | — | — | — | 214,157 | 1 |
| Steven H. Lipstein | — | 137,889 | 170,060 | — | — | — | — | 307,949 | 1 |
| Leo S. Mackay, Jr. | — | 125,000 | 170,060 | — | — | — | — | 295,060 | 1 |
| Steven O. Vondran | — | 125,000 | 170,060 | — | — | — | — | 295,060 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 32,289,721 | 0.1228 |
| T. Rowe Price Associates, Inc. | 5pct_holder | 21,790,077 | 0.081 |
| BlackRock, Inc. | 5pct_holder | 21,878,148 | 0.081 |
| T. Rowe Price Investment Management, Inc. | 5pct_holder | 14,362,684 | 0.052 |
| State Street Corporation | 5pct_holder | 14,047,510 | 0.052 |
| Mark C. Birk | director_officer | 30,136 | — |
| Cynthia J. Brinkley | director_officer | 13,494 | — |
| Catherine S. Brune | director_officer | 26,628 | — |
| Ward H. Dickson | director_officer | 18,946 | — |
| Fadi M. Diya | director_officer | 57,372 | — |
| Jamie L. Engstrom | director_officer | 1,686 | — |
| Ellen M. Fitzsimmons | director_officer | 53,026 | — |
| Rafael Flores | director_officer | 13,953 | — |
| Richard J. Harshman | director_officer | 22,135 | — |
| Craig S. Ivey | director_officer | 16,597 | — |
| Mark C. Lindgren | director_officer | 39,898 | — |
| Steven H. Lipstein | director_officer | 37,862 | — |
| Martin J. Lyons, Jr. | director_officer | 212,055 | — |
| Leo S. Mackay, Jr. | director_officer | 11,834 | — |
| Michael L. Moehn | director_officer | 132,375 | — |
| Timothy S. Rausch | director_officer | 1,251 | — |
| Shawn E. Schukar | director_officer | 50,793 | — |
| Leonard P. Singh | director_officer | 16,748 | — |
| Steven O. Vondran | director_officer | 3,597 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Cynthia J. Brinkley | 66 | 7 | 1 | female | honorific | 1 |
| Catherine S. Brune | — | — | 1 | female | honorific | — |
| Ward H. Dickson | 63 | 8 | 1 | male | honorific | 1 |
| Noelle K. Eder | — | — | 1 | female | name | — |
| Ellen M. Fitzsimmons | 65 | 17 | 1 | female | honorific | 0 |
| Rafael Flores | 70 | 11 | 1 | male | honorific | 0 |
| Kimberly J. Harris | — | — | 1 | female | name | — |
| Richard J. Harshman | 69 | 13 | 1 | male | honorific | 1 |
| Craig S. Ivey | 63 | 8 | 1 | male | honorific | 0 |
| James C. Johnson | — | — | 1 | male | name | — |
| Steven H. Lipstein | 70 | 16 | 1 | male | honorific | 0 |
| Leo S. Mackay, Jr. | 64 | 6 | 1 | male | honorific | 1 |
| Jamie L. Engstrom | 48 | 0 | 1 | female | honorific | 0 |
| Timothy S. Rausch | 61 | 0 | 1 | male | honorific | 0 |
| Steven O. Vondran | 55 | 1 | 1 | male | honorific | 1 |
| Martin J. Lyons, Jr. | 59 | 4 | 0 | male | honorific | 0 |

---

## AMAT — filed 2026-01-28

- source: <https://www.sec.gov/Archives/edgar/data/6951/000119312526027307/d71992ddef14a.htm>
- accession: `0001193125-26-027307`
- carve payload: 38,287 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Applied Materials, Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 10 |
| `n_directors` | 10 |
| `pct_female_directors` | 0.2 |
| `pct_gender_stated` | 0.5 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 64 |
| `avg_board_tenure` | 10.571 |
| `pct_independent_directors` | 0.9 |
| `ceo_name_proxy` | Gary E. Dickerson |
| `ceo_age` | 68 |
| `ceo_since_year` | 2013 |
| `ceo_salary` | 1,030,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 26,944,995 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 1,506,375 |
| `ceo_all_other_comp` | 167,982 |
| `ceo_total_comp` | 29,649,352 |
| `ceo_equity_pay_pct` | 0.909 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 60,019,694 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.91 |
| `ceo_pay_ratio` | 330 |
| `median_employee_pay` | 89,744 |
| `independent_chair` | 1 |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | KPMG LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 8,484,000 |
| `audit_fees_audit` | 7,516,000 |
| `audit_fees_audit_related` | 93,000 |
| `audit_fees_tax` | 802,000 |
| `audit_fees_other` | 73,000 |
| `auditor_fees_prior` | 8,965,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gary E. Dickerson | President and Chief Executive Officer | 2025 | 1,030,000 | 0 | 26,944,995 | — | 1,506,375 | 0 | 167,982 | 29,649,352 | 1 |
| Gary E. Dickerson | President and Chief Executive Officer | 2024 | 1,030,000 | 0 | 24,861,142 | — | 1,754,734 | 0 | 153,336 | 27,799,212 | 1 |
| Gary E. Dickerson | President and Chief Executive Officer | 2023 | 1,030,000 | 0 | 23,951,048 | — | 1,631,520 | 0 | 241,976 | 26,854,544 | 1 |
| Brice Hill | Senior Vice President, Chief Financial Officer and Global Information Services | 2025 | 750,000 | 0 | 6,730,519 | — | 987,188 | 0 | 37,614 | 8,505,321 | 1 |
| Brice Hill | Senior Vice President, Chief Financial Officer and Global Information Services | 2024 | 744,616 | 0 | 5,851,542 | — | 1,149,947 | 0 | 63,641 | 7,809,746 | 1 |
| Brice Hill | Senior Vice President, Chief Financial Officer and Global Information Services | 2023 | 708,846 | 0 | 5,530,849 | — | 1,019,304 | 0 | 324,136 | 7,583,135 | 1 |
| Prabu G. Raja | President, Semiconductor Products Group | 2025 | 800,000 | 0 | 7,584,138 | — | 1,053,000 | 6,268 | 19,993 | 9,463,399 | 1 |
| Prabu G. Raja | President, Semiconductor Products Group | 2024 | 792,308 | 0 | 6,968,612 | — | 1,229,580 | 8,265 | 19,303 | 9,018,068 | 1 |
| Prabu G. Raja | President, Semiconductor Products Group | 2023 | 740,000 | 0 | 6,636,826 | — | 1,091,475 | 12,409 | 18,073 | 8,498,783 | 1 |
| Timothy M. Deane | Senior Vice President, Applied Global Services | 2025 | 734,616 | 0 | 4,771,545 | — | 859,219 | 0 | 20,421 | 6,385,801 | 1 |
| Timothy M. Deane | Senior Vice President, Applied Global Services | 2024 | 642,308 | 0 | 3,776,822 | — | 956,313 | 0 | 18,855 | 5,394,298 | 1 |
| Timothy M. Deane | Senior Vice President, Applied Global Services | 2023 | 574,947 | 0 | 3,097,266 | — | 733,590 | 0 | 17,761 | 4,423,564 | 1 |
| Omkaram Nalamasu | Senior Vice President, Chief Technology Officer | 2025 | 665,000 | 0 | 4,570,802 | — | 758,100 | 1,654 | 20,265 | 6,015,821 | 1 |
| Omkaram Nalamasu | Senior Vice President, Chief Technology Officer | 2024 | 659,616 | 0 | 4,255,476 | — | 866,608 | 2,944 | 14,440 | 5,799,084 | 1 |
| Omkaram Nalamasu | Senior Vice President, Chief Technology Officer | 2023 | 625,385 | 0 | 4,037,503 | — | 742,203 | 5,200 | 3,718 | 5,414,009 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| James R. Anderson | — | 30,522 | 152,624 | — | — | — | — | 183,146 | 1 |
| Rani Borkar | — | 122,500 | 237,118 | — | — | — | — | 359,618 | 1 |
| Judy Bruner | — | 165,357 | 237,118 | — | — | — | — | 402,475 | 1 |
| Xun (Eric) Chen | — | 122,500 | 237,118 | — | — | — | — | 359,618 | 1 |
| Aart J. de Geus | — | 110,000 | 237,118 | — | — | — | — | 347,118 | 1 |
| Thomas J. Iannotti | — | 282,500 | 237,118 | — | — | — | 4,000 | 523,618 | 1 |
| Alexander A. Karsner | — | 122,500 | 237,118 | — | — | — | — | 359,618 | 1 |
| Kevin P. March | — | 144,643 | 237,118 | — | — | — | — | 381,761 | 1 |
| Scott A. McGregor | — | 147,500 | 237,118 | — | — | — | — | 384,618 | 1 |
| Yvonne McGill | — | 118,681 | 237,118 | — | — | — | — | 355,799 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| BlackRock, Inc. | 5pct_holder | 76,811,130 | 0.0969 |
| The Vanguard Group | 5pct_holder | 74,116,511 | 0.0935 |
| James R. Anderson | director_officer | 806 | — |
| Rani Borkar | director_officer | 9,199 | — |
| Judy Bruner | director_officer | 34,141 | — |
| Xun (Eric) Chen | director_officer | 45,229 | — |
| Aart J. de Geus | director_officer | 111,042 | — |
| Thomas J. Iannotti | director_officer | 50,568 | — |
| Alexander A. Karsner | director_officer | 16,108 | — |
| Kevin P. March | director_officer | 5,939 | — |
| Scott A. McGregor | director_officer | 23,836 | — |
| Gary E. Dickerson | director_officer | 1,398,872 | — |
| Brice Hill | director_officer | 58,062 | — |
| Prabu G. Raja | director_officer | 406,740 | — |
| Timothy M. Deane | director_officer | 80,037 | — |
| Omkaram Nalamasu | director_officer | 122,402 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| James R. Anderson | 53 | 0 | 1 | male | honorific | 1 |
| Aart J. de Geus | 71 | 18 | 1 | male | pronoun | 1 |
| Kevin P. March | — | — | 1 | male | name | — |
| Rani Borkar | 64 | 5 | 1 | female | honorific | 0 |
| Gary E. Dickerson | 68 | 12 | 0 | male | honorific | 0 |
| Scott A. McGregor | — | — | 1 | male | name | — |
| Judy Bruner | 67 | 9 | 1 | female | honorific | 2 |
| Thomas J. Iannotti | 69 | 20 | 1 | male | honorific | 1 |
| Xun (Eric) Chen | 56 | 10 | 1 | male | pronoun | 0 |
| Alexander A. Karsner | — | — | 1 | male | name | — |

---

## BA — filed 2026-03-06

- source: <https://www.sec.gov/Archives/edgar/data/12927/000119312526096787/d39411ddef14a.htm>
- accession: `0001193125-26-096787`
- carve payload: 41,255 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | The Boeing Company |
| `fiscal_year_extract` | 2025 |
| `board_size` | 12 |
| `n_directors` | 12 |
| `pct_female_directors` | 0.333 |
| `pct_gender_stated` | 0.833 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 62.222 |
| `avg_board_tenure` | 5.889 |
| `pct_independent_directors` | 0.75 |
| `ceo_name_proxy` | Robert K. Ortberg |
| `ceo_age` | — |
| `ceo_since_year` | 2024 |
| `ceo_salary` | 1,500,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 7,874,818 |
| `ceo_option_awards` | 9,624,959 |
| `ceo_non_equity_incentive` | 3,930,000 |
| `ceo_all_other_comp` | 651,612 |
| `ceo_total_comp` | 23,581,389 |
| `ceo_equity_pay_pct` | 0.742 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 80,331,481 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | — |
| `ceo_pay_ratio` | 166 |
| `median_employee_pay` | 141,933 |
| `independent_chair` | 1 |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | 0 |
| `majority_voting` | 1 |
| `auditor_name` | Deloitte & Touche LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 39,100,000 |
| `audit_fees_audit` | 39,100,000 |
| `audit_fees_audit_related` | 4,500,000 |
| `audit_fees_tax` | 0 |
| `audit_fees_other` | 0 |
| `auditor_fees_prior` | 37,400,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Robert K. Ortberg | President and Chief Executive Officer | 2025 | 1,500,000 | 0 | 7,874,818 | 9,624,959 | 3,930,000 | 0 | 651,612 | 23,581,389 | 1 |
| Robert K. Ortberg | President and Chief Executive Officer | 2024 | 525,000 | 1,250,000 | 7,999,900 | 7,999,946 | 0 | 0 | 613,783 | 18,388,629 | 1 |
| Jesus Malave, Jr. | Chief Financial Officer | 2025 | 399,808 | 8,500,000 | 4,999,976 | 4,499,911 | 1,650,600 | 0 | 132,706 | 20,183,001 | 1 |
| Stephanie F. Pope | President and Chief Executive Officer, Commercial Airplanes | 2025 | 1,200,000 | 0 | 4,499,949 | 5,499,977 | 2,625,240 | 139,961 | 416,699 | 14,381,826 | 1 |
| Stephanie F. Pope | President and Chief Executive Officer, Commercial Airplanes | 2024 | 1,191,539 | 0 | 7,791,487 | 0 | 87,213 | 0 | 891,052 | 9,961,291 | 1 |
| Stephanie F. Pope | President and Chief Executive Officer, Commercial Airplanes | 2023 | 959,231 | 0 | 6,437,750 | 0 | 1,368,500 | 118,720 | 772,022 | 9,656,223 | 1 |
| Brett C. Gerry | Chief Legal Officer | 2025 | 955,154 | 0 | 2,024,848 | 2,474,974 | 1,257,600 | 72,285 | 167,429 | 6,952,290 | 1 |
| Brett C. Gerry | Chief Legal Officer | 2024 | 900,000 | 0 | 3,116,479 | 0 | 171,000 | 0 | 225,198 | 4,412,677 | 1 |
| Brett C. Gerry | Chief Legal Officer | 2023 | 900,000 | 0 | 6,437,750 | 0 | 850,500 | 68,500 | 304,729 | 8,561,479 | 1 |
| Jeffrey S. Shockey | Executive Vice President of Government Operations, Global Public Policy and Corporate Strategy | 2025 | 651,846 | 0 | 3,624,847 | 1,374,958 | 763,475 | 0 | 69,900 | 6,485,026 | 1 |
| Brian J. West | Former Chief Financial Officer | 2025 | 1,038,308 | 0 | 2,699,858 | 3,299,938 | 1,498,640 | 0 | 211,205 | 8,747,949 | 1 |
| Brian J. West | Former Chief Financial Officer | 2024 | 1,000,000 | 0 | 4,674,622 | 0 | 209,000 | 0 | 302,808 | 6,186,430 | 1 |
| Brian J. West | Former Chief Financial Officer | 2023 | 1,000,000 | 0 | 9,364,000 | 0 | 1,089,000 | 0 | 488,638 | 11,941,638 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Robert A. Bradway | — | 155,000 | 200,000 | — | — | — | 31,000 | 386,000 | 1 |
| Mortimer J. Buckley | — | 135,000 | 200,000 | — | — | — | 0 | 335,000 | 1 |
| Lynne M. Doughtie | — | 160,000 | 200,000 | — | — | — | 31,000 | 391,000 | 1 |
| David L. Gitlin | — | 135,000 | 200,000 | — | — | — | 31,000 | 366,000 | 1 |
| Lynn J. Good | — | 155,000 | 200,000 | — | — | — | 31,000 | 386,000 | 1 |
| Stayce D. Harris | — | 135,000 | 200,000 | — | — | — | 31,000 | 366,000 | 1 |
| Akhil Johri | — | 155,000 | 200,000 | — | — | — | 31,000 | 386,000 | 1 |
| David L. Joyce | — | 185,000 | 200,000 | — | — | — | 31,000 | 416,000 | 1 |
| Steven M. Mollenkopf | — | 385,000 | 200,000 | — | — | — | 29,500 | 614,500 | 1 |
| John M. Richardson | — | 150,000 | 200,000 | — | — | — | 31,000 | 381,000 | 1 |
| Sabrina Soussan | — | 42,651 | 63,187 | — | — | — | 0 | 105,838 | 1 |
| Bradley D. Tilden | — | 10,639 | 0 | — | — | — | 31,000 | 41,639 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group 100 Vanguard Boulevard Malvern, Pennsylvania 19355 | 5pct_holder | 70,989,325 | 0.09 |
| FMR LLC 245 Summer Street Boston, Massachusetts 02210 | 5pct_holder | 54,979,044 | 0.07 |
| BlackRock, Inc. 50 Hudson Yards New York, New York 10001 | 5pct_holder | 52,979,795 | 0.068 |
| Robert A. Bradway | director_officer | 15,449 | — |
| Mortimer J. Buckley | director_officer | 4,348 | — |
| Lynne M. Doughtie | director_officer | 5,390 | — |
| David L. Gitlin | director_officer | 6,894 | — |
| Lynn J. Good | director_officer | 17,028 | — |
| Stayce D. Harris | director_officer | 8,276 | — |
| Akhil Johri | director_officer | 11,041 | — |
| David L. Joyce | director_officer | 9,173 | — |
| Steven M. Mollenkopf | director_officer | 16,761 | — |
| John M. Richardson | director_officer | 6,658 | — |
| Bradley D. Tilden | director_officer | 629 | — |
| Brett C. Gerry | director_officer | 107,948 | — |
| Jesus Malave, Jr. | director_officer | 40,221 | — |
| Robert K. Ortberg | director_officer | 140,944 | — |
| Stephanie F. Pope | director_officer | 99,390 | — |
| Jeffrey S. Shockey | director_officer | 27,738 | — |
| Brian J. West | director_officer | 137,294 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Robert A. Bradway | 63 | 10 | 1 | male | honorific | 1 |
| Mortimer J. Buckley | 56 | 1 | 1 | male | honorific | 1 |
| Lynne M. Doughtie | 63 | 5 | 1 | female | honorific | 2 |
| David L. Gitlin | 56 | 4 | 1 | male | honorific | 1 |
| Lynn J. Good | 66 | 11 | 1 | female | honorific | 1 |
| Stayce D. Harris | 66 | 5 | 1 | female | pronoun | — |
| Akhil Johri | 64 | 6 | 1 | male | honorific | 1 |
| David L. Joyce | 69 | 5 | 1 | male | honorific | 0 |
| Steven M. Mollenkopf | 57 | 6 | 1 | male | honorific | 1 |
| John M. Richardson | — | — | — | male | honorific | — |
| Sabrina Soussan | — | — | — | female | name | — |
| Bradley D. Tilden | — | — | — | male | honorific | — |

---

## CAT — filed 2026-04-30

- source: <https://www.sec.gov/Archives/edgar/data/18230/000130817926000358/cat014496-def14a.htm>
- accession: `0001308179-26-000358`
- carve payload: 41,173 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Caterpillar Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 8 |
| `n_directors` | 8 |
| `pct_female_directors` | 0.5 |
| `pct_gender_stated` | 0.125 |
| `n_women_directors_vs_inferred` | — |
| `avg_director_age` | 63.125 |
| `avg_board_tenure` | 3.75 |
| `pct_independent_directors` | 0.875 |
| `ceo_name_proxy` | JOSEPH E. CREED |
| `ceo_age` | 50 |
| `ceo_since_year` | 2025 |
| `ceo_salary` | 1,381,892 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 9,538,484 |
| `ceo_option_awards` | 3,101,117 |
| `ceo_non_equity_incentive` | 2,476,800 |
| `ceo_all_other_comp` | 509,784 |
| `ceo_total_comp` | 17,008,077 |
| `ceo_equity_pay_pct` | 0.743 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 69,495,214 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | 0.94 |
| `ceo_pay_ratio` | 196 |
| `median_employee_pay` | 89,253 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | — |
| `auditor_since_year` | — |
| `auditor_fees` | 38,100,000 |
| `audit_fees_audit` | 35,500,000 |
| `audit_fees_audit_related` | 2,200,000 |
| `audit_fees_tax` | 300,000 |
| `audit_fees_other` | 100,000 |
| `auditor_fees_prior` | 39,300,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D. JAMES UMPLEBY III Executive | Executive Chairman | 2025 | 1,414,583 | 0 | 13,080,611 | 4,227,829 | 2,509,300 | 0 | 959,173 | 22,191,496 | 1 |
| D. JAMES UMPLEBY III Executive | Executive Chairman | 2024 | 1,811,250 | 0 | 14,079,975 | 4,125,037 | 4,363,800 | 0 | 881,234 | 25,261,296 | 1 |
| D. JAMES UMPLEBY III Executive | Executive Chairman | 2023 | 1,752,500 | 0 | 8,499,949 | 8,500,034 | 5,733,900 | 0 | 1,343,949 | 25,830,332 | 1 |
| JOSEPH E. CREED | CEO | 2025 | 1,381,892 | 0 | 9,538,484 | 3,101,117 | 2,476,800 | 0 | 509,784 | 17,008,077 | 1 |
| JOSEPH E. CREED | CEO | 2024 | 1,115,000 | 0 | 5,973,437 | 1,749,968 | 1,887,700 | 0 | 404,460 | 11,130,565 | 1 |
| JOSEPH E. CREED | CEO | 2023 | 819,092 | 0 | 2,199,975 | 2,199,962 | 1,771,700 | 0 | 253,746 | 7,244,475 | 1 |
| ANDREW R. J. BONFIELD | CFO | 2025 | 991,575 | 0 | 3,662,325 | 1,183,771 | 1,320,200 | 0 | 289,828 | 7,447,699 | 1 |
| ANDREW R. J. BONFIELD | CFO | 2024 | 953,450 | 0 | 4,095,869 | 1,200,047 | 1,434,400 | 0 | 291,731 | 7,975,497 | 1 |
| ANDREW R. J. BONFIELD | CFO | 2023 | 916,800 | 0 | 2,650,027 | 2,649,989 | 1,902,000 | 0 | 241,123 | 8,359,939 | 1 |
| BOB DE LANGE | Group President | 2025 | 925,525 | 0 | 2,765,492 | 893,866 | 1,147,100 | 0 | 288,692 | 6,020,675 | 1 |
| BOB DE LANGE | Group President | 2024 | 889,950 | 0 | 5,157,307 | 924,982 | 1,299,200 | 0 | 285,416 | 8,556,855 | 1 |
| BOB DE LANGE | Group President | 2023 | 855,700 | 0 | 2,199,975 | 2,199,962 | 1,651,200 | 0 | 305,254 | 7,212,091 | 1 |
| DENISE C. JOHNSON | Group President | 2025 | 985,825 | 0 | 2,765,492 | 893,866 | 1,201,900 | 0 | 305,268 | 6,152,351 | 1 |
| DENISE C. JOHNSON | Group President | 2024 | 947,900 | 0 | 5,157,307 | 924,982 | 1,374,500 | 0 | 311,808 | 8,716,497 | 1 |
| DENISE C. JOHNSON | Group President | 2023 | 911,450 | 0 | 2,199,975 | 2,199,962 | 1,874,200 | 0 | 312,957 | 7,498,544 | 1 |
| CHRISTY M. PAMBIANCHI CHRO | CHRO | 2025 | 566,667 | 450,000 | 7,907,120 | 692,171 | 660,500 | 0 | 398,458 | 10,674,916 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| DANIEL M. DICKINSON | 2025 | 82,843 | 0 | — | — | — | 1,500 | 84,343 | 1 |
| JAMES C. FISH, JR. | 2025 | 163,874 | 175,033 | — | — | — | 5,000 | 343,907 | 1 |
| GERALD JOHNSON | 2025 | 150,000 | 175,033 | — | — | — | 5,000 | 330,033 | 1 |
| NAZZIC S. KEENE | 2025 | 150,000 | 175,033 | — | — | — | 0 | 325,033 | 1 |
| JUDITH F. MARKS | 2025 | 150,000 | 175,033 | — | — | — | 5,000 | 330,033 | 1 |
| DAVID W. MACLENNAN | 2025 | 180,550 | 175,033 | — | — | — | 5,000 | 360,583 | 1 |
| DEBRA L. REED-KLAGES | 2025 | 225,000 | 175,033 | — | — | — | 0 | 400,033 | 1 |
| SUSAN C. SCHWAB | 2025 | 150,000 | 175,033 | — | — | — | 0 | 325,033 | 1 |
| RAYFORD WILKINS, JR. | 2025 | 175,000 | 175,033 | — | — | — | 5,000 | 355,033 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| BlackRock, Inc. | 5pct_holder | 33,509,816 | 0.066 |
| State Street Corporation | 5pct_holder | 37,741,566 | 0.074 |
| The Vanguard Group | 5pct_holder | 46,835,122 | 0.1006 |
| ANDREW R. J. BONFIELD | director_officer | 43,272 | — |
| JOSEPH E. CREED | director_officer | 132,092 | — |
| BOB DE LANGE | director_officer | 246,003 | — |
| DANIEL M. DICKINSON | director_officer | 4,732 | — |
| JAMES C. FISH, JR. | director_officer | 3,490 | — |
| LYNN J. GOOD | director_officer | 163 | — |
| DENISE C. JOHNSON | director_officer | 92,014 | — |
| GERALD JOHNSON | director_officer | 3,197 | — |
| NAZZIC S. KEENE | director_officer | 30 | — |
| DAVID W. MACLENNAN | director_officer | 7,467 | — |
| JUDITH F. MARKS | director_officer | 1,239 | — |
| CHRISTY M. PAMBIANCHI | director_officer | — | — |
| DEBRA L. REED-KLAGES | director_officer | 12,464 | — |
| SUSAN C. SCHWAB | director_officer | 5,479 | — |
| D. JAMES UMPLEBY III | director_officer | 807,651 | — |
| RAYFORD WILKINS, JR. | director_officer | 7,689 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| JOSEPH E. CREED | 50 | 1 | 0 | male | pronoun | 0 |
| JAMES C. FISH, JR. | 63 | 3 | 1 | male | pronoun | 1 |
| Lynn J. Good | 67 | 0 | 1 | female | honorific | 2 |
| GERALD JOHNSON | 63 | 5 | 1 | male | pronoun | 1 |
| NAZZIC S. KEENE | 65 | 2 | 1 | female | pronoun | 2 |
| DAVID W. MACLENNAN | 66 | 5 | 1 | male | pronoun | 1 |
| JUDITH F. MARKS | 62 | 3 | 1 | female | pronoun | 1 |
| DEBRA L. REED-KLAGES | 69 | 11 | 1 | female | pronoun | 2 |

---

## ECL — filed 2026-03-20

- source: <https://www.sec.gov/Archives/edgar/data/31462/000110465926032777/tm2530816-3_def14a.htm>
- accession: `0001104659-26-032777`
- carve payload: 41,712 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Ecolab Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 15 |
| `n_directors` | 15 |
| `pct_female_directors` | 0.4 |
| `pct_gender_stated` | 0.667 |
| `n_women_directors_vs_inferred` | — |
| `avg_director_age` | 60.3 |
| `avg_board_tenure` | 5.6 |
| `pct_independent_directors` | 0.933 |
| `ceo_name_proxy` | Christophe Beck |
| `ceo_age` | 58 |
| `ceo_since_year` | 2021 |
| `ceo_salary` | 1,390,081 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 7,228,101 |
| `ceo_option_awards` | 4,566,916 |
| `ceo_non_equity_incentive` | 3,383,183 |
| `ceo_all_other_comp` | 402,785 |
| `ceo_total_comp` | 17,404,935 |
| `ceo_equity_pay_pct` | 0.678 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 36,301,525 |
| `insider_ownership_pct` | 0.005 |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 0 |
| `say_on_pay_support_pct` | 0.9 |
| `ceo_pay_ratio` | 326 |
| `median_employee_pay` | 53,462 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | PricewaterhouseCoopers LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 18,291,483 |
| `audit_fees_audit` | 14,041,298 |
| `audit_fees_audit_related` | 348,185 |
| `audit_fees_tax` | 3,900,000 |
| `audit_fees_other` | 2,000 |
| `auditor_fees_prior` | 17,999,300 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Christophe Beck | Chairman and Chief Executive Officer | 2025 | 1,390,081 | 0 | 7,228,101 | 4,566,916 | 3,383,183 | 433,869 | 402,785 | 17,404,935 | 1 |
| Christophe Beck | Chairman and Chief Executive Officer | 2024 | 1,346,923 | 0 | 6,148,112 | 4,006,711 | 4,276,924 | 166,519 | 445,735 | 16,390,924 | 1 |
| Christophe Beck | Chairman and Chief Executive Officer | 2023 | 1,243,750 | 0 | 5,874,624 | 4,055,664 | 3,731,250 | 195,936 | 445,831 | 15,547,055 | 1 |
| Scott D. Kirkland | Chief Financial Officer | 2025 | 856,727 | 0 | 1,746,815 | 1,103,656 | 1,257,219 | 99,402 | 129,014 | 5,192,833 | 1 |
| Scott D. Kirkland | Chief Financial Officer | 2024 | 829,785 | 0 | 1,639,545 | 1,068,434 | 1,729,108 | 50,496 | 150,924 | 5,468,293 | 1 |
| Scott D. Kirkland | Chief Financial Officer | 2023 | 787,500 | 0 | 1,697,159 | 1,171,641 | 1,570,453 | 64,270 | 47,165 | 5,338,188 | 1 |
| Darrell R. Brown | President and Chief Operating Officer | 2025 | 856,727 | 0 | 1,867,368 | 1,179,802 | 1,257,222 | 183,249 | 302,976 | 5,647,343 | 1 |
| Darrell R. Brown | President and Chief Operating Officer | 2024 | 829,785 | 0 | 1,697,952 | 1,106,628 | 1,729,108 | 153,440 | 293,733 | 5,810,646 | 1 |
| Darrell R. Brown | President and Chief Operating Officer | 2023 | 787,500 | 0 | 1,762,406 | 1,216,719 | 1,570,453 | 59,090 | 279,330 | 5,675,498 | 1 |
| Gregory B. Cook | Executive Vice President and President − Institutional Group | 2025 | 632,100 | 0 | 963,623 | 608,955 | 841,721 | 179,046 | 130,534 | 3,355,979 | 1 |
| Gregory B. Cook | Executive Vice President and President − Institutional Group | 2024 | 570,477 | 0 | 848,976 | 553,314 | 962,332 | 18,266 | 95,710 | 3,049,075 | 1 |
| Margeaux M. King | Executive Vice President, Human Resources | 2025 | 545,769 | 47,000 | 2,652,242 | 883,784 | 534,054 | 14,778 | 22,808 | 4,700,435 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Judson B. Althoff | 2025 | 131,033 | 135,000 | 57,777 | — | — | 0 | 323,810 | 1 |
| Shari L. Ballard | 2025 | 135,000 | 135,000 | 57,777 | — | — | 0 | 327,777 | 1 |
| Michel D. Doukeris | 2025 | 114,122 | 101,250 | 68,762 | — | — | 0 | 284,134 | 1 |
| Eric M. Green | 2025 | 145,000 | 135,000 | 57,777 | — | — | 0 | 337,777 | 1 |
| Marion K. Gross | 2025 | 81,387 | 67,500 | 57,777 | — | — | 0 | 206,664 | 1 |
| Arthur J. Higgins | 2025 | 44,299 | 47,843 | 0 | — | — | 0 | 92,143 | 1 |
| Michael Larson | 2025 | 145,000 | 135,000 | 57,777 | — | — | 0 | 337,777 | 1 |
| David W. MacLennan | 2025 | 185,000 | 135,000 | 57,777 | — | — | 0 | 377,777 | 1 |
| Tracy B. McKibben | 2025 | 125,000 | 135,000 | 57,777 | — | — | 0 | 317,777 | 1 |
| Lionel L. Nowell III | 2025 | 150,000 | 135,000 | 57,777 | — | — | 5,051 | 347,828 | 1 |
| Victoria J. Reich | 2025 | 135,000 | 135,000 | 57,777 | — | — | 15,401 | 343,178 | 1 |
| Suzanne M. Vautrinot | 2025 | 155,000 | 135,000 | 57,777 | — | — | 1,598 | 349,375 | 1 |
| Julie P. Whalen | 2025 | 54,293 | 33,750 | 0 | — | — | 0 | 88,043 | 1 |
| John J. Zillmer | 2025 | 125,000 | 135,000 | 57,777 | — | — | 0 | 317,777 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| Christophe Beck (Chairman and Chief Executive Officer) | director_officer | 518,078 | — |
| Scott D. Kirkland (Chief Financial Officer) | director_officer | 117,231 | — |
| Darrell R. Brown | director_officer | 194,023 | — |
| Gregory B. Cook | director_officer | 68,043 | — |
| Margeaux M. King | director_officer | 3,126 | — |
| Judson B. Althoff | director_officer | 2,921 | — |
| Shari L. Ballard | director_officer | 18,028 | — |
| Michel Doukeris | director_officer | 1,402 | — |
| Eric M. Green | director_officer | 7,383 | — |
| Marion K. Gross | director_officer | 1,138 | — |
| Michael Larson | director_officer | 32,058 | — |
| David W. MacLennan | director_officer | 39,643 | — |
| Tracy B. McKibben | director_officer | 23,567 | — |
| Lionel L. Nowell III | director_officer | 15,397 | — |
| Victoria J. Reich | director_officer | 37,385 | — |
| Suzanne M. Vautrinot | director_officer | 23,700 | — |
| Julie P. Whalen | director_officer | 204 | — |
| John J. Zillmer | director_officer | 61,222 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Judson B. Althoff | 53 | 1 | 1 | male | honorific | — |
| Shari L. Ballard | 59 | 7 | 1 | female | honorific | — |
| Christophe Beck | 58 | 5 | 0 | male | honorific | — |
| Michel D. Doukeris | 52 | 0 | 1 | male | honorific | — |
| Eric M. Green | 56 | 3 | 1 | male | honorific | — |
| Marion K. Gross | 65 | 0 | 1 | female | honorific | — |
| Arthur J. Higgins | — | — | 1 | male | name | — |
| Michael Larson | 66 | 13 | 1 | male | honorific | — |
| David W. MacLennan | 66 | 10 | 1 | male | honorific | — |
| Tracy B. McKibben | 57 | 10 | 1 | female | honorific | — |
| Lionel L. Nowell III | 71 | 7 | 1 | male | honorific | — |
| Victoria J. Reich | — | — | 1 | female | name | — |
| Suzanne M. Vautrinot | — | — | 1 | female | name | — |
| Julie P. Whalen | — | — | 1 | female | name | — |
| John J. Zillmer | — | — | 1 | male | name | — |

---

## EOG — filed 2026-03-27

- source: <https://www.sec.gov/Archives/edgar/data/821189/000110465926036125/eog-20260520xdef14a.htm>
- accession: `0001104659-26-036125`
- carve payload: 40,653 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | EOG Resources, Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 10 |
| `n_directors` | 10 |
| `pct_female_directors` | 0.3 |
| `pct_gender_stated` | 0 |
| `n_women_directors_vs_inferred` | — |
| `avg_director_age` | — |
| `avg_board_tenure` | — |
| `pct_independent_directors` | 0.9 |
| `ceo_name_proxy` | Ezra Y. Yacob |
| `ceo_age` | — |
| `ceo_since_year` | 2021 |
| `ceo_salary` | 1,434,615 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 12,729,628 |
| `ceo_option_awards` | 0 |
| `ceo_non_equity_incentive` | 2,718,800 |
| `ceo_all_other_comp` | 693,593 |
| `ceo_total_comp` | 17,576,636 |
| `ceo_equity_pay_pct` | 0.724 |
| `n_neos` | 4 |
| `sct_years` | 3 |
| `total_neo_comp` | 35,637,207 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 5 |
| `say_on_pay_support_pct` | — |
| `ceo_pay_ratio` | 78 |
| `median_employee_pay` | 225,792 |
| `independent_chair` | — |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | Deloitte |
| `auditor_since_year` | — |
| `auditor_fees` | — |
| `audit_fees_audit` | 270,170 |
| `audit_fees_audit_related` | 321,000 |
| `audit_fees_tax` | 0 |
| `audit_fees_other` | 6,954 |
| `auditor_fees_prior` | — |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EZRA Y. YACOB | Chairman of the Board and Chief Executive Officer | 2025 | 1,434,615 | 0 | 12,729,628 | 0 | 2,718,800 | 0 | 693,593 | 17,576,636 | 1 |
| EZRA Y. YACOB | Chairman of the Board and Chief Executive Officer | 2024 | 1,326,923 | 0 | 11,439,317 | 0 | 2,936,250 | 0 | 512,135 | 16,214,625 | 1 |
| EZRA Y. YACOB | Chairman of the Board and Chief Executive Officer | 2023 | 1,169,231 | 0 | 10,496,402 | 0 | 2,520,000 | 0 | 373,139 | 14,558,772 | 1 |
| ANN D. JANSSEN | Executive Vice President and Chief Financial Officer | 2025 | 725,385 | 0 | 3,415,207 | 0 | 925,000 | 0 | 136,969 | 5,202,561 | 1 |
| ANN D. JANSSEN | Executive Vice President and Chief Financial Officer | 2024 | 638,077 | 0 | 4,341,547 | 0 | 841,725 | 0 | 104,349 | 5,925,698 | 1 |
| JEFFREY R. LEITZELL | Executive Vice President and Chief Operating Officer | 2025 | 771,692 | 0 | 4,656,969 | 0 | 990,000 | 0 | 200,313 | 6,618,974 | 1 |
| JEFFREY R. LEITZELL | Executive Vice President and Chief Operating Officer | 2024 | 653,077 | 0 | 4,159,706 | 0 | 957,000 | 0 | 163,811 | 5,933,594 | 1 |
| JEFFREY R. LEITZELL | Executive Vice President and Chief Operating Officer | 2023 | 524,231 | 0 | 2,900,940 | 0 | 775,000 | 0 | 142,578 | 4,342,749 | 1 |
| MICHAEL P. DONALDSON | Executive Vice President and Chief Legal Officer | 2025 | 784,462 | 0 | 4,242,968 | 0 | 912,600 | 0 | 299,006 | 6,239,036 | 1 |
| MICHAEL P. DONALDSON | Executive Vice President and Chief Legal Officer | 2024 | 760,385 | 0 | 3,119,782 | 0 | 998,325 | 0 | 287,345 | 5,165,837 | 1 |
| MICHAEL P. DONALDSON | Executive Vice President and Chief Legal Officer | 2023 | 728,846 | 0 | 3,148,795 | 0 | 926,100 | 0 | 293,771 | 5,097,512 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| John D. Chandler | 2025 | 0 | 87,498 | — | — | — | 0 | 87,498 | 1 |
| Janet F. Clark | 2025 | 115,000 | 209,891 | — | — | — | 100,000 | 424,891 | 1 |
| Charles R. Crisp | 2025 | 105,000 | 209,891 | — | — | — | 103,160 | 418,051 | 1 |
| Robert P. Daniels | 2025 | 125,000 | 209,891 | — | — | — | 103,160 | 438,051 | 1 |
| Lynn A. Dugle | 2025 | 115,000 | 209,891 | — | — | — | 100,000 | 424,891 | 1 |
| C. Christopher Gaut | 2025 | 125,000 | 209,891 | — | — | — | 100,360 | 435,251 | 1 |
| Michael T. Kerr | 2025 | 115,000 | 209,891 | — | — | — | 100,000 | 424,891 | 1 |
| Julie J. Robertson | 2025 | 115,000 | 209,891 | — | — | — | 19,650 | 344,541 | 1 |
| Donald F. Textor | 2025 | 41,667 | 0 | — | — | — | 200,000 | 241,667 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 54,643,522 | 0.102 |
| BlackRock, Inc. | 5pct_holder | 40,956,544 | 0.0764 |
| Capital World Investors | 5pct_holder | 40,823,482 | 0.0762 |
| State Street Corporation | 5pct_holder | 35,456,435 | 0.0662 |
| JPMorgan Chase & Co. | 5pct_holder | 29,540,610 | 0.0551 |
| John D. Chandler | director_officer | 826 | — |
| Janet F. Clark | director_officer | 47,370 | — |
| Charles R. Crisp | director_officer | 63,210 | — |
| Robert P. Daniels | director_officer | 32,700 | — |
| Michael P. Donaldson | director_officer | 186,645 | — |
| Lynn A. Dugle | director_officer | 6,026 | — |
| C. Christopher Gaut | director_officer | 21,241 | — |
| Ann D. Janssen | director_officer | 144,035 | — |
| Michael T. Kerr | director_officer | 189,213 | — |
| Jeffrey R. Leitzell | director_officer | 149,109 | — |
| Julie J. Robertson | director_officer | 15,609 | — |
| Ezra Y. Yacob | director_officer | 440,429 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| John D. Chandler | — | — | 1 | male | name | — |
| Janet F. Clark | — | — | 1 | female | name | — |
| Charles R. Crisp | — | — | 1 | male | name | — |
| Robert P. Daniels | — | — | 1 | male | name | — |
| Lynn A. Dugle | — | — | 1 | female | name | — |
| C. Christopher Gaut | — | — | 1 | male | name | — |
| Michael T. Kerr | — | — | 1 | male | name | — |
| Julie J. Robertson | — | — | 1 | female | name | — |
| Donald F. Textor | — | — | 1 | male | name | — |
| Ezra Y. Yacob | — | — | 0 | male | name | — |

---

## GE — filed 2026-03-12

- source: <https://www.sec.gov/Archives/edgar/data/40545/000004054526000018/ge-20260312.htm>
- accession: `0000040545-26-000018`
- carve payload: 40,368 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | GE Aerospace |
| `fiscal_year_extract` | 2025 |
| `board_size` | 9 |
| `n_directors` | 9 |
| `pct_female_directors` | 0.333 |
| `pct_gender_stated` | 0.333 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 64.667 |
| `avg_board_tenure` | 5.222 |
| `pct_independent_directors` | 0.889 |
| `ceo_name_proxy` | H. Lawrence Culp, Jr |
| `ceo_age` | 62 |
| `ceo_since_year` | 2022 |
| `ceo_salary` | 2,000,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 27,564,925 |
| `ceo_option_awards` | 4,575,005 |
| `ceo_non_equity_incentive` | 7,520,000 |
| `ceo_all_other_comp` | 742,962 |
| `ceo_total_comp` | 45,616,160 |
| `ceo_equity_pay_pct` | 0.705 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 83,404,755 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | 0.71 |
| `ceo_pay_ratio` | 486 |
| `median_employee_pay` | 93,873 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | 0 |
| `majority_voting` | 1 |
| `auditor_name` | Deloitte |
| `auditor_since_year` | — |
| `auditor_fees` | 19,900,000 |
| `audit_fees_audit` | 18,200,000 |
| `audit_fees_audit_related` | 1,300,000 |
| `audit_fees_tax` | 300,000 |
| `audit_fees_other` | 100,000 |
| `auditor_fees_prior` | 20,900,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H. Lawrence Culp, Jr | Chairman & CEO | 2025 | 2,000,000 | 0 | 27,564,925 | 4,575,005 | 7,520,000 | 3,213,268 | 742,962 | 45,616,160 | 1 |
| H. Lawrence Culp, Jr | Chairman & CEO | 2024 | 2,250,000 | 0 | 78,281,883 | 0 | 6,781,250 | 1,325,765 | 315,688 | 88,954,586 | 1 |
| H. Lawrence Culp, Jr | Chairman & CEO | 2023 | 2,500,000 | 0 | 4,999,987 | 0 | 5,625,000 | 1,002,278 | 571,020 | 14,698,285 | 1 |
| Rahul Ghai | SVP, CFO | 2025 | 955,000 | 0 | 6,219,038 | 1,800,027 | 2,154,000 | 0 | 142,678 | 11,270,743 | 1 |
| Rahul Ghai | SVP, CFO | 2024 | 916,042 | 0 | 5,265,201 | 1,427,982 | 1,887,000 | 0 | 111,934 | 9,608,159 | 1 |
| Rahul Ghai | SVP, CFO | 2023 | 900,000 | 0 | 2,529,166 | 0 | 1,374,332 | 0 | 102,994 | 4,906,492 | 1 |
| Russell Stokes | SVP, Former CEO, Commercial Engines & Services | 2025 | 1,400,000 | 0 | 6,307,750 | 1,500,036 | 2,408,000 | 1,545,493 | 161,963 | 13,323,242 | 1 |
| Russell Stokes | SVP, Former CEO, Commercial Engines & Services | 2024 | 1,400,000 | 0 | 5,971,540 | 2,347,391 | 2,226,000 | 3,788 | 138,645 | 12,087,364 | 1 |
| Russell Stokes | SVP, Former CEO, Commercial Engines & Services | 2023 | 1,400,000 | 0 | 1,745,057 | 1,499,995 | 2,086,000 | 1,414,057 | 115,832 | 8,260,941 | 1 |
| Mohamed Ali | SVP, CEO, Commercial Engines & Services | 2025 | 790,890 | 0 | 3,188,483 | 1,079,984 | 1,504,000 | 661,283 | 60,952 | 7,285,592 | 1 |
| John Phillips, III | SVP, General Counsel & Secretary | 2025 | 800,000 | 0 | 2,439,369 | 1,050,009 | 1,504,000 | 0 | 115,640 | 5,909,018 | 1 |
| John Phillips, III | SVP, General Counsel & Secretary | 2024 | 800,000 | 0 | 2,433,881 | 1,066,821 | 1,400,000 | 0 | 69,604 | 5,770,306 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Stephen Angel | 2025 | 0 | 359,040 | — | — | — | 0 | 359,040 | 1 |
| Sébastien Bazin | 2025 | 0 | 345,795 | — | — | — | 0 | 345,795 | 1 |
| Margaret Billson | 2025 | 140,000 | 201,925 | — | — | — | 1,000 | 342,925 | 1 |
| Wesley Bush | 2025 | 11,793 | 75,574 | — | — | — | 0 | 87,367 | 1 |
| Thomas Enders | 2025 | 140,000 | 201,925 | — | — | — | 0 | 341,925 | 1 |
| Edward Garden | 2025 | 140,000 | 201,925 | — | — | — | 0 | 341,925 | 1 |
| Isabella Goren | 2025 | 170,000 | 201,925 | — | — | — | 5,000 | 376,925 | 1 |
| Thomas Horton | 2025 | 191,467 | 201,925 | — | — | — | 0 | 393,392 | 1 |
| Catherine Lesjak | 2025 | 160,367 | 201,925 | — | — | — | 5,000 | 367,292 | 1 |
| Darren McDew | 2025 | 147,500 | 201,925 | — | — | — | 0 | 349,425 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 88,439,179 | 0.084 |
| BlackRock, Inc. | 5pct_holder | 82,447,476 | 0.079 |
| Fidelity Management & Research | 5pct_holder | 66,923,455 | 0.064 |
| H. Lawrence Culp, Jr. | director_officer | 1,612,480 | — |
| Rahul Ghai | director_officer | 140,216 | — |
| Russell Stokes | director_officer | 659,867 | — |
| Mohamed Ali | director_officer | 48,480 | — |
| John Phillips, III | director_officer | 7,685 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| H. Lawrence Culp, Jr | 62 | 8 | 0 | male | name | 1 |
| Thomas Horton | 64 | 8 | 1 | male | name | 2 |
| Isabella Goren | 65 | 4 | 1 | female | honorific | 1 |
| Catherine Lesjak | 67 | 7 | 1 | female | honorific | 1 |
| Darren McDew | 65 | 3 | 1 | male | name | 2 |
| Sébastien Bazin | 64 | 10 | 1 | male | name | 1 |
| Margaret Billson | 64 | 3 | 1 | female | name | 0 |
| Wesley Bush | 64 | 1 | 1 | male | honorific | 0 |
| Thomas Enders | 67 | 3 | 1 | male | name | 1 |

---

## INCY — filed 2026-04-28

- source: <https://www.sec.gov/Archives/edgar/data/879169/000087916926000031/incy-20260428.htm>
- accession: `0000879169-26-000031`
- carve payload: 42,468 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Incyte Corporation |
| `fiscal_year_extract` | 2025 |
| `board_size` | 9 |
| `n_directors` | 9 |
| `pct_female_directors` | 0.333 |
| `pct_gender_stated` | 0.444 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 66.25 |
| `avg_board_tenure` | 8.375 |
| `pct_independent_directors` | 0.778 |
| `ceo_name_proxy` | William J. Meury |
| `ceo_age` | 58 |
| `ceo_since_year` | 2025 |
| `ceo_salary` | 647,260 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 27,509,657 |
| `ceo_option_awards` | 2,573,998 |
| `ceo_non_equity_incentive` | 1,300,000 |
| `ceo_all_other_comp` | 55,037 |
| `ceo_total_comp` | 32,085,952 |
| `ceo_equity_pay_pct` | 0.938 |
| `n_neos` | 7 |
| `sct_years` | 3 |
| `total_neo_comp` | 70,129,310 |
| `insider_ownership_pct` | 0.162 |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 5 |
| `say_on_pay_support_pct` | — |
| `ceo_pay_ratio` | 110.47 |
| `median_employee_pay` | 295,914 |
| `independent_chair` | 0 |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | Ernst & Young LLP |
| `auditor_since_year` | 1991 |
| `auditor_fees` | 3,915,000 |
| `audit_fees_audit` | 3,868,000 |
| `audit_fees_audit_related` | 47,000 |
| `audit_fees_tax` | 0 |
| `audit_fees_other` | — |
| `auditor_fees_prior` | 3,742,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| William J. Meury | Chief Executive Officer | 2025 | 647,260 | 0 | 27,509,657 | 2,573,998 | 1,300,000 | — | 55,037 | 32,085,952 | 1 |
| Hervé Hoppenot | Former President and Chief Executive Officer | 2025 | 1,392,499 | 0 | 1,971,975 | 1,694,034 | 1,026,072 | — | 295,945 | 6,380,525 | 1 |
| Hervé Hoppenot | Former President and Chief Executive Officer | 2024 | 1,344,707 | 0 | 10,543,489 | 3,315,223 | 2,202,829 | — | 53,298 | 17,459,546 | 1 |
| Hervé Hoppenot | Former President and Chief Executive Officer | 2023 | 1,291,929 | 0 | 9,334,654 | 4,425,053 | 1,554,703 | — | 53,187 | 16,659,526 | 1 |
| Pablo J. Cagnoni | President and Global Head of Research and Development | 2025 | 966,516 | 0 | 4,952,378 | 1,815,159 | 1,007,027 | — | 97,881 | 8,838,961 | 1 |
| Pablo J. Cagnoni | President and Global Head of Research and Development | 2024 | 933,344 | 0 | 4,463,962 | 1,404,291 | 1,042,470 | — | 60,232 | 7,904,299 | 1 |
| Pablo J. Cagnoni | President and Global Head of Research and Development | 2023 | 517,808 | 737,362 | 14,089,823 | 450,377 | 735,750 | — | 39,183 | 16,570,303 | 1 |
| Mohamed Issa | Executive Vice President and Head of U.S. Commercial | 2025 | 690,411 | 1,000,000 | 7,870,118 | 450,983 | 533,610 | — | 47,660 | 10,592,782 | 1 |
| Steven H. Stein | Executive Vice President, Chief Medical Officer and Head of Late-stage Development | 2025 | 835,067 | 0 | 3,466,675 | 1,270,592 | 696,054 | — | 35,784 | 6,304,172 | 1 |
| Steven H. Stein | Executive Vice President, Chief Medical Officer and Head of Late-stage Development | 2024 | 806,406 | 0 | 3,124,735 | 1,295,829 | 720,553 | — | 32,305 | 5,979,828 | 1 |
| Steven H. Stein | Executive Vice President, Chief Medical Officer and Head of Late-stage Development | 2023 | 775,580 | 0 | 2,925,757 | 1,386,951 | 508,549 | — | 31,256 | 5,628,093 | 1 |
| Christiana Stamoulis | Former Executive Vice President and Chief Financial Officer | 2025 | 520,839 | 0 | 3,136,468 | 1,149,588 | 0 | — | 129,655 | 4,936,550 | 1 |
| Christiana Stamoulis | Former Executive Vice President and Chief Financial Officer | 2024 | 712,242 | 0 | 2,827,129 | 1,172,420 | 636,414 | — | 49,552 | 5,397,757 | 1 |
| Christiana Stamoulis | Former Executive Vice President and Chief Financial Officer | 2023 | 684,288 | 0 | 2,647,095 | 1,254,863 | 449,166 | — | 49,073 | 5,084,485 | 1 |
| Thomas Tray | Vice President, Chief Accounting Officer and Principal Financial Officer | 2025 | 358,189 | 0 | 305,897 | 154,623 | 149,423 | — | 22,236 | 990,368 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Julian C. Baker | 2025 | 0 | 308,436 | 264,702 | — | — | — | 573,138 | 1 |
| Jean-Jacques Bienaimé | 2025 | 95,000 | 176,436 | 264,702 | — | — | — | 536,138 | 1 |
| Otis W. Brawley | 2025 | 70,000 | 176,436 | 264,702 | — | — | — | 511,138 | 1 |
| Paul J. Clancy | 2025 | 0 | 273,436 | 264,702 | — | — | — | 538,138 | 1 |
| Jacqualyn A. Fouse | 2025 | 83,500 | 176,436 | 264,702 | — | — | — | 524,638 | 1 |
| Edmund P. Harrigan | 2025 | 0 | 274,936 | 264,702 | — | — | — | 539,638 | 1 |
| Katherine A. High | 2025 | 70,000 | 176,436 | 264,702 | — | — | — | 511,138 | 1 |
| Susanne Schaffert | 2025 | 82,000 | 176,436 | 264,702 | — | — | — | 523,138 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| Felix J. Baker | 5pct_holder | 31,225,572 | 0.156 |
| Baker Bros. Advisors LP and affiliated entities | 5pct_holder | 30,865,077 | 0.154 |
| The Vanguard Group and affiliates | 5pct_holder | 22,519,381 | 0.113 |
| Dodge & Cox | 5pct_holder | 16,090,421 | 0.081 |
| BlackRock, Inc. | 5pct_holder | 15,242,171 | 0.076 |
| William J. Meury | director_officer | — | — |
| Hervé Hoppenot | director_officer | 1,056,787 | — |
| Christiana Stamoulis | director_officer | 57,534 | — |
| Thomas Tray | director_officer | 70,942 | — |
| Pablo J. Cagnoni | director_officer | 84,555 | — |
| Steven H. Stein | director_officer | 313,303 | — |
| Mohamed Issa | director_officer | — | — |
| Julian C. Baker | director_officer | 31,223,155 | 0.156 |
| Jean-Jacques Bienaimé | director_officer | 138,988 | — |
| Otis W. Brawley | director_officer | 59,313 | — |
| Paul J. Clancy | director_officer | 141,888 | — |
| Jacqualyn A. Fouse | director_officer | 141,223 | — |
| Edmund P. Harrigan | director_officer | 86,157 | — |
| Katherine A. High | director_officer | 75,301 | — |
| Susanne Schaffert | director_officer | 47,167 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| William J. Meury | 58 | 0 | 0 | male | honorific | 0 |
| Julian C. Baker | 59 | 24 | 0 | male | honorific | 3 |
| Jean-Jacques Bienaimé | 72 | 10 | 1 | male | honorific | 3 |
| Otis W. Brawley | 66 | 4 | 1 | male | name | 3 |
| Paul J. Clancy | 64 | 10 | 1 | male | honorific | 1 |
| Jacqualyn A. Fouse | 64 | 8 | 1 | female | name | 2 |
| Edmund P. Harrigan | 73 | 6 | 1 | male | name | 1 |
| Katherine A. High | 74 | 5 | 1 | female | name | 1 |
| Susanne Schaffert | — | — | — | female | name | — |

---

## JPM — filed 2026-04-06

- source: <https://www.sec.gov/Archives/edgar/data/19617/000001961726000096/jpm-20260402.htm>
- accession: `0000019617-26-000096`
- carve payload: 42,585 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | JPMorgan Chase & Co. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 12 |
| `n_directors` | 12 |
| `pct_female_directors` | 0.5 |
| `pct_gender_stated` | 0.083 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 64.727 |
| `avg_board_tenure` | 7 |
| `pct_independent_directors` | 0.917 |
| `ceo_name_proxy` | James Dimon |
| `ceo_age` | 70 |
| `ceo_since_year` | 2005 |
| `ceo_salary` | 1,500,000 |
| `ceo_bonus` | 5,000,000 |
| `ceo_stock_awards` | 32,500,000 |
| `ceo_option_awards` | 0 |
| `ceo_non_equity_incentive` | 0 |
| `ceo_all_other_comp` | 1,587,852 |
| `ceo_total_comp` | 40,632,724 |
| `ceo_equity_pay_pct` | 0.8 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 168,480,771 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.91 |
| `ceo_pay_ratio` | 363 |
| `median_employee_pay` | 111,905 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | PricewaterhouseCoopers LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 135,000,000 |
| `audit_fees_audit` | 91,000,000 |
| `audit_fees_audit_related` | 38,100,000 |
| `audit_fees_tax` | 5,900,000 |
| `audit_fees_other` | — |
| `auditor_fees_prior` | 119,000,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| James Dimon | Chairman and CEO | 2025 | 1,500,000 | 5,000,000 | 32,500,000 | 0 | 0 | 44,872 | 1,587,852 | 40,632,724 | 1 |
| James Dimon | Chairman and CEO | 2024 | 1,500,000 | 5,000,000 | 29,500,000 | 0 | 0 | 50,826 | 1,632,636 | 37,683,462 | 1 |
| James Dimon | Chairman and CEO | 2023 | 1,500,000 | 5,000,000 | 28,000,000 | 0 | 0 | 40,185 | 1,420,074 | 35,960,259 | 1 |
| Mary Callahan Erdoes | CEO, AWM | 2025 | 1,000,000 | 12,000,000 | 16,500,000 | 0 | 0 | 28,934 | 0 | 29,528,934 | 1 |
| Mary Callahan Erdoes | CEO, AWM | 2024 | 1,000,000 | 11,000,000 | 15,750,000 | 0 | 0 | 19,779 | 37,181 | 27,806,960 | 1 |
| Mary Callahan Erdoes | CEO, AWM | 2023 | 750,000 | 10,500,000 | 14,850,000 | 0 | 0 | 29,183 | 5,000 | 26,134,183 | 1 |
| Troy Rohrbaugh | Co-CEO, CIB | 2025 | 1,000,000 | 10,600,000 | 13,800,000 | 0 | 0 | 7,218 | 8,676 | 25,415,894 | 1 |
| Troy Rohrbaugh | Co-CEO, CIB | 2024 | 1,000,000 | 9,200,000 | 13,350,000 | 0 | 0 | 4,530 | 5,373 | 23,559,903 | 1 |
| Douglas Petno | Co-CEO, CIB | 2025 | 1,000,000 | 10,600,000 | 11,400,000 | 0 | 0 | 41,597 | 31,825 | 23,073,422 | 1 |
| Jeremy Barnum | Chief Financial Officer | 2025 | 1,000,000 | 7,400,000 | 9,795,000 | 0 | 0 | 17,926 | 0 | 18,212,926 | 1 |
| Jeremy Barnum | Chief Financial Officer | 2024 | 1,000,000 | 6,530,000 | 8,550,000 | 0 | 0 | 10,582 | 5,000 | 16,095,582 | 1 |
| Jeremy Barnum | Chief Financial Officer | 2023 | 750,000 | 5,700,000 | 6,750,000 | 0 | 0 | 16,257 | 5,000 | 13,221,257 | 1 |
| Daniel Pinto | Vice Chair; Former President and COO | 2025 | 1,500,000 | 5,000,000 | 25,000,000 | 0 | 0 | 0 | 116,871 | 31,616,871 | 1 |
| Daniel Pinto | Vice Chair; Former President and COO | 2024 | 1,500,000 | 5,000,000 | 23,500,000 | 0 | 0 | 0 | 72,089 | 30,072,089 | 1 |
| Daniel Pinto | Vice Chair; Former President and COO | 2023 | 1,500,000 | 5,000,000 | 22,000,000 | 0 | 0 | 0 | 97,037 | 28,597,037 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Stephen B. Burke | 2025 | 165,000 | 265,000 | — | — | — | 60,000 | 490,000 | 1 |
| Linda B. Bammann | 2025 | 160,000 | 265,000 | — | — | — | 20,000 | 445,000 | 1 |
| Michele G. Buck | 2025 | 102,861 | 0 | — | — | — | 15,833 | 118,694 | 1 |
| Todd A. Combs | 2025 | 121,522 | 265,000 | — | — | — | 26,195 | 412,717 | 1 |
| Alicia Boler Davis | 2025 | 130,000 | 265,000 | — | — | — | 20,000 | 415,000 | 1 |
| Alex Gorsky | 2025 | 130,000 | 265,000 | — | — | — | 20,000 | 415,000 | 1 |
| Mellody Hobson | 2025 | 150,000 | 265,000 | — | — | — | 27,500 | 442,500 | 1 |
| Phebe N. Novakovic | 2025 | 130,000 | 265,000 | — | — | — | 20,000 | 415,000 | 1 |
| Virginia M. Rometty | 2025 | 111,304 | 265,000 | — | — | — | 30,000 | 406,304 | 1 |
| Brad D. Smith | 2025 | 119,667 | 0 | — | — | — | 18,889 | 138,556 | 1 |
| Mark A. Weinberger | 2025 | 160,000 | 265,000 | — | — | — | 20,000 | 445,000 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 265,758,185 | 0.0986 |
| BlackRock, Inc. | 5pct_holder | 192,831,104 | 0.0715 |
| Stephen B. Burke | director_officer | 107,334 | — |
| Linda B. Bammann | director_officer | 46,486 | — |
| Jeremy Barnum | director_officer | 23,804 | — |
| Michele G. Buck | director_officer | 5 | — |
| Alicia Boler Davis | director_officer | 285 | — |
| James Dimon | director_officer | 6,266,647 | — |
| Mary Callahan Erdoes | director_officer | 613,405 | — |
| Alex Gorsky | director_officer | 88 | — |
| Mellody Hobson | director_officer | 129,574 | — |
| Phebe N. Novakovic | director_officer | 545 | — |
| Douglas Petno | director_officer | 435,285 | — |
| Troy Rohrbaugh | director_officer | 111,279 | — |
| Virginia M. Rometty | director_officer | 280 | — |
| Brad D. Smith | director_officer | 9,898 | — |
| Mark A. Weinberger | director_officer | 500 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Stephen B. Burke | 67 | 21 | 1 | male | name | — |
| Linda B. Bammann | 70 | 12 | 1 | female | name | — |
| Michele G. Buck | 64 | 0 | 1 | female | name | — |
| Todd A. Combs | — | — | 1 | male | name | — |
| Alicia Boler Davis | 57 | 2 | 1 | female | name | — |
| Alex Gorsky | 65 | 3 | 1 | male | name | — |
| Mellody Hobson | 57 | 7 | 1 | female | name | — |
| Phebe N. Novakovic | 68 | 5 | 1 | female | name | — |
| Virginia M. Rometty | 68 | 5 | 1 | female | name | — |
| Brad D. Smith | 62 | 0 | 1 | male | name | — |
| Mark A. Weinberger | 64 | 1 | 1 | male | name | — |
| James Dimon | 70 | 21 | 0 | male | honorific | — |

---

## KLAC — filed 2025-09-23

- source: <https://www.sec.gov/Archives/edgar/data/319201/000119312525213294/d913108ddef14a.htm>
- accession: `0001193125-25-213294`
- carve payload: 40,893 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | KLA Corporation |
| `fiscal_year_extract` | 2025 |
| `board_size` | 10 |
| `n_directors` | 10 |
| `pct_female_directors` | 0.3 |
| `pct_gender_stated` | 1 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 63.3 |
| `avg_board_tenure` | 9.5 |
| `pct_independent_directors` | 0.9 |
| `ceo_name_proxy` | Richard Wallace |
| `ceo_age` | 65 |
| `ceo_since_year` | 2006 |
| `ceo_salary` | 1,190,481 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 20,147,371 |
| `ceo_option_awards` | 0 |
| `ceo_non_equity_incentive` | 3,704,317 |
| `ceo_all_other_comp` | 50,085 |
| `ceo_total_comp` | 25,092,254 |
| `ceo_equity_pay_pct` | 0.803 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 55,176,699 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.925 |
| `ceo_pay_ratio` | 257 |
| `median_employee_pay` | 97,645 |
| `independent_chair` | — |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | PricewaterhouseCoopers LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 7,100,000 |
| `audit_fees_audit` | 6,100,000 |
| `audit_fees_audit_related` | — |
| `audit_fees_tax` | 1,000,000 |
| `audit_fees_other` | 0 |
| `auditor_fees_prior` | 6,600,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Richard Wallace | President & Chief Executive Officer | 2025 | 1,190,481 | 0 | 20,147,371 | 0 | 3,704,317 | 0 | 50,085 | 25,092,254 | 1 |
| Richard Wallace | President & Chief Executive Officer | 2024 | 1,129,905 | 0 | 19,695,857 | 0 | 1,978,769 | 0 | 28,434 | 22,832,965 | 1 |
| Richard Wallace | President & Chief Executive Officer | 2023 | 1,000,000 | 0 | 23,492,301 | 0 | 2,124,000 | 0 | 28,058 | 26,644,359 | 1 |
| Bren Higgins | Executive Vice President & Chief Financial Officer | 2025 | 750,000 | 0 | 5,494,737 | 0 | 1,433,077 | 0 | 34,988 | 7,712,802 | 1 |
| Bren Higgins | Executive Vice President & Chief Financial Officer | 2024 | 758,174 | 0 | 5,855,158 | 0 | 810,569 | 0 | 28,649 | 7,452,550 | 1 |
| Bren Higgins | Executive Vice President & Chief Financial Officer | 2023 | 643,462 | 0 | 9,737,890 | 0 | 875,741 | 0 | 27,660 | 11,284,753 | 1 |
| Ahmad Khan | President, Semiconductor Products and Customers | 2025 | 750,000 | 0 | 5,494,737 | 0 | 1,433,077 | 0 | 32,630 | 7,710,444 | 1 |
| Ahmad Khan | President, Semiconductor Products and Customers | 2024 | 758,174 | 0 | 5,855,158 | 0 | 810,569 | 0 | 27,509 | 7,451,410 | 1 |
| Ahmad Khan | President, Semiconductor Products and Customers | 2023 | 650,000 | 0 | 9,975,137 | 0 | 920,400 | 0 | 26,853 | 11,572,390 | 1 |
| Oreste Donzella | Former Executive Vice President and Chief Strategy Officer | 2025 | 362,558 | 0 | 915,789 | 0 | 466,720 | 0 | 4,790,950 | 6,536,017 | 1 |
| Oreste Donzella | Former Executive Vice President and Chief Strategy Officer | 2024 | 501,251 | 0 | 1,809,895 | 0 | 407,809 | 0 | 4,955,816 | 7,674,771 | 1 |
| Oreste Donzella | Former Executive Vice President and Chief Strategy Officer | 2023 | 477,385 | 0 | 4,013,343 | 0 | 441,795 | 0 | 965,126 | 5,897,649 | 1 |
| Brian Lorig | Executive Vice President, KLA Global Services | 2025 | 543,654 | 0 | 3,663,158 | 0 | 775,855 | 0 | 13,486 | 4,996,153 | 1 |
| Brian Lorig | Executive Vice President, KLA Global Services | 2024 | 521,251 | 0 | 2,661,344 | 0 | 584,100 | 0 | 10,802 | 3,777,497 | 1 |
| Brian Lorig | Executive Vice President, KLA Global Services | 2023 | 496,731 | 0 | 4,368,618 | 0 | 506,012 | 0 | 10,578 | 5,381,939 | 1 |
| Mary Beth Wilkinson | Executive Vice President, Chief Legal Officer & Corporate Secretary | 2025 | 555,558 | 0 | 1,923,158 | 0 | 619,842 | 0 | 30,471 | 3,129,029 | 1 |
| Mary Beth Wilkinson | Executive Vice President, Chief Legal Officer & Corporate Secretary | 2024 | 525,000 | 0 | 2,395,612 | 0 | 495,600 | 0 | 30,132 | 3,446,344 | 1 |
| Mary Beth Wilkinson | Executive Vice President, Chief Legal Officer & Corporate Secretary | 2023 | 525,000 | 0 | 3,894,917 | 0 | 545,160 | 0 | 28,528 | 4,993,605 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Robert Calderoni | 2025 | 187,500 | 289,963 | — | — | 0 | 3,526 | 480,989 | 1 |
| Jeneanne Hanley | 2025 | 112,500 | 234,381 | — | — | 0 | 2,859 | 349,740 | 1 |
| Emiko Higashi | 2025 | 115,000 | 234,381 | — | — | 0 | 2,859 | 352,240 | 1 |
| Kevin Kennedy | 2025 | 152,500 | 234,381 | — | — | 0 | 2,859 | 389,740 | 1 |
| Michael McMullen | 2025 | 112,500 | 234,381 | — | — | 0 | 2,859 | 349,740 | 1 |
| Gary Moore | 2025 | 140,000 | 234,381 | — | — | 0 | 2,859 | 377,240 | 1 |
| Marie Myers | 2025 | 76,250 | — | — | — | 0 | 2,859 | 79,109 | 1 |
| Victor Peng | 2025 | 112,500 | 234,381 | — | — | 0 | 2,859 | 349,740 | 1 |
| Robert Rango | 2025 | 115,000 | 234,381 | — | — | 0 | 3,456 | 352,837 | 1 |
| Jamie Samath | 2025 | 28,750 | 122,268 | — | — | 0 | — | 151,018 | 1 |
| Susan Taylor | 2025 | 28,750 | 116,753 | — | — | 0 | — | 145,503 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group, Inc. | 5pct_holder | 13,499,554 | 0.103 |
| BlackRock, Inc. | 5pct_holder | 11,584,889 | 0.088 |
| Richard Wallace | director_officer | 34,503 | — |
| Robert Calderoni | director_officer | 14,963 | — |
| Jeneanne Hanley | director_officer | 3,897 | — |
| Emiko Higashi | director_officer | 14,766 | — |
| Kevin Kennedy | director_officer | 7,832 | — |
| Michael McMullen | director_officer | 975 | — |
| Gary Moore | director_officer | 15,028 | — |
| Victor Peng | director_officer | 5,612 | — |
| Jamie Samath | director_officer | 174 | — |
| Susan Taylor | director_officer | 166 | — |
| Oreste Donzella | director_officer | 7,580 | — |
| Bren Higgins | director_officer | 12,990 | — |
| Ahmad Khan | director_officer | 1,091 | — |
| Brian Lorig | director_officer | 576.102 | — |
| Mary Beth Wilkinson | director_officer | 6.934 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Robert Calderoni | 65 | 18 | 1 | male | honorific | 1 |
| Jeneanne Hanley | 52 | 6 | 1 | female | honorific | 1 |
| Emiko Higashi | 66 | 15 | 1 | female | honorific | 2 |
| Kevin Kennedy | 69 | 18 | 1 | male | honorific | 2 |
| Michael McMullen | 64 | 2 | 1 | male | honorific | 1 |
| Gary Moore | 76 | 11 | 1 | male | honorific | 0 |
| Victor Peng | 65 | 6 | 1 | male | honorific | 1 |
| Jamie Samath | 55 | 0 | 1 | male | honorific | 0 |
| Susan Taylor | 56 | 0 | 1 | female | honorific | 1 |
| Richard Wallace | 65 | 19 | 0 | male | honorific | 1 |

---

## LMT — filed 2026-03-26

- source: <https://www.sec.gov/Archives/edgar/data/936468/000093646826000004/lmt-20260326.htm>
- accession: `0000936468-26-000004`
- carve payload: 40,870 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Lockheed Martin Corporation |
| `fiscal_year_extract` | 2025 |
| `board_size` | 9 |
| `n_directors` | 9 |
| `pct_female_directors` | 0.444 |
| `pct_gender_stated` | 0.778 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 66.667 |
| `avg_board_tenure` | 6.889 |
| `pct_independent_directors` | 0.889 |
| `ceo_name_proxy` | James D. Taiclet |
| `ceo_age` | 65 |
| `ceo_since_year` | 2020 |
| `ceo_salary` | 1,751,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 13,667,645 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 5,496,300 |
| `ceo_all_other_comp` | 2,538,363 |
| `ceo_total_comp` | 23,453,308 |
| `ceo_equity_pay_pct` | 0.583 |
| `n_neos` | 6 |
| `sct_years` | 3 |
| `total_neo_comp` | 60,458,427 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | 0.92 |
| `ceo_pay_ratio` | 180 |
| `median_employee_pay` | 130,614 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | EY |
| `auditor_since_year` | — |
| `auditor_fees` | 27,045,000 |
| `audit_fees_audit` | 24,750,000 |
| `audit_fees_audit_related` | 95,000 |
| `audit_fees_tax` | 2,200,000 |
| `audit_fees_other` | 0 |
| `auditor_fees_prior` | 24,273,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| James D. Taiclet | Chairman, President and Chief Executive Officer | 2025 | 1,751,000 | 0 | 13,667,645 | — | 5,496,300 | 0 | 2,538,363 | 23,453,308 | 1 |
| James D. Taiclet | Chairman, President and Chief Executive Officer | 2024 | 1,751,000 | 0 | 13,046,594 | — | 6,566,900 | 0 | 2,389,420 | 23,753,914 | 1 |
| James D. Taiclet | Chairman, President and Chief Executive Officer | 2023 | 1,751,000 | 0 | 13,008,681 | — | 6,655,900 | 0 | 1,398,194 | 22,813,775 | 1 |
| Evan T. Scott | Chief Financial Officer | 2025 | 783,144 | 0 | 637,810 | — | 1,097,975 | 75,478 | 174,877 | 2,769,284 | 1 |
| Frank A. St. John | Chief Operating Officer | 2025 | 1,089,423 | 0 | 5,386,762 | — | 2,348,800 | 418,238 | 617,938 | 9,861,161 | 1 |
| Frank A. St. John | Chief Operating Officer | 2024 | 1,069,615 | 0 | 5,139,703 | — | 2,817,500 | 0 | 616,995 | 9,643,813 | 1 |
| Frank A. St. John | Chief Operating Officer | 2023 | 1,064,039 | 0 | 5,081,551 | — | 2,914,700 | 395,731 | 495,840 | 9,951,861 | 1 |
| Timothy S. Cahill | President, Missiles and Fire Control | 2025 | 995,000 | 0 | 3,618,238 | — | 1,756,000 | 267,432 | 202,419 | 6,839,089 | 1 |
| Timothy S. Cahill | President, Missiles and Fire Control | 2024 | 1,015,570 | 0 | 3,400,048 | — | 1,543,500 | 0 | 169,582 | 6,128,700 | 1 |
| Timothy S. Cahill | President, Missiles and Fire Control | 2023 | 1,008,905 | 0 | 3,414,828 | — | 1,758,600 | 252,214 | 734,879 | 7,169,426 | 1 |
| Kevin J. O’Connor | Senior Vice President, General Counsel and Corporate Secretary | 2025 | 800,961 | 300,000 | 10,199,879 | — | 1,077,700 | 0 | 318,180 | 12,696,720 | 1 |
| Jesus Malave | Former Chief Financial Officer | 2025 | 343,269 | 0 | 4,422,188 | — | 0 | 0 | 73,408 | 4,838,865 | 1 |
| Jesus Malave | Former Chief Financial Officer | 2024 | 1,014,808 | 0 | 4,269,880 | — | 3,012,600 | 0 | 235,326 | 8,532,614 | 1 |
| Jesus Malave | Former Chief Financial Officer | 2023 | 984,808 | 0 | 4,065,426 | — | 1,434,500 | 0 | 174,522 | 6,659,256 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| John C. Aquilino | 2025 | 170,000 | 170,000 | — | — | — | 0 | 340,000 | 1 |
| David B. Burritt | 2025 | 170,000 | 170,000 | — | — | — | 357 | 340,357 | 1 |
| Bruce A. Carlson | 2025 | 60,247 | 56,667 | — | — | — | 755 | 117,669 | 1 |
| John M. Donovan | 2025 | 200,000 | 170,000 | — | — | — | 625 | 370,625 | 1 |
| Joseph F. Dunford, Jr. | 2025 | 195,000 | 170,000 | — | — | — | 0 | 365,000 | 1 |
| Thomas J. Falk | 2025 | 250,000 | 170,000 | — | — | — | 0 | 420,000 | 1 |
| Vicki A. Hollub | 2025 | 170,000 | 170,000 | — | — | — | 1,000 | 341,000 | 1 |
| Debra L. Reed-Klages | 2025 | 170,000 | 170,000 | — | — | — | 3,041 | 343,041 | 1 |
| Heather A. Wilson | 2025 | 170,000 | 170,000 | — | — | — | 1,045 | 341,045 | 1 |
| Patricia E. Yarrington | 2025 | 205,000 | 170,000 | — | — | — | 0 | 375,000 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| State Street Corporation | 5pct_holder | 37,056,708 | 0.149 |
| The Vanguard Group | 5pct_holder | 22,098,899 | 0.089 |
| BlackRock, Inc. | 5pct_holder | 18,292,313 | 0.074 |
| John C. Aquilino | director_officer | 674 | — |
| David B. Burritt | director_officer | 31,193 | — |
| Timothy S. Cahill | director_officer | 25,347 | — |
| John M. Donovan | director_officer | 6,778 | — |
| Joseph F. Dunford, Jr. | director_officer | 2,936 | — |
| Thomas J. Falk | director_officer | 20,704 | — |
| Vicki A. Hollub | director_officer | 6,409 | — |
| Jesus Malave | director_officer | 224 | — |
| Kevin J. O'Connor | director_officer | 18,038 | — |
| Debra L. Reed-Klages | director_officer | 3,194 | — |
| Evan T. Scott | director_officer | 6,433 | — |
| Frank A. St. John | director_officer | 16,186 | — |
| James D. Taiclet | director_officer | 110,349 | — |
| Heather A. Wilson | director_officer | 895 | — |
| Patricia E. Yarrington | director_officer | 2,228 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| James D. Taiclet | 65 | 7 | 0 | male | honorific | 0 |
| John C. Aquilino | 64 | 1 | 1 | male | pronoun | 0 |
| David B. Burritt | 70 | 17 | 1 | male | honorific | 0 |
| John M. Donovan | 65 | 4 | 1 | male | honorific | 1 |
| Thomas J. Falk | 67 | 15 | 1 | male | honorific | 1 |
| Vicki A. Hollub | 66 | 7 | 1 | female | honorific | 1 |
| Debra L. Reed-Klages | 69 | 6 | 1 | female | honorific | 2 |
| Heather A. Wilson | 65 | 1 | 1 | female | pronoun | 0 |
| Patricia E. Yarrington | 69 | 4 | 1 | female | honorific | 0 |

---

## NKE — filed 2026-07-15

- source: <https://www.sec.gov/Archives/edgar/data/320187/000032018726000089/nke-20260715.htm>
- accession: `0000320187-26-000089`
- carve payload: 60,911 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | NIKE, Inc. |
| `fiscal_year_extract` | 2026 |
| `board_size` | 11 |
| `n_directors` | 11 |
| `pct_female_directors` | 0.455 |
| `pct_gender_stated` | 0.727 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 57.222 |
| `avg_board_tenure` | 7.889 |
| `pct_independent_directors` | 0.727 |
| `ceo_name_proxy` | Elliott Hill |
| `ceo_age` | 62 |
| `ceo_since_year` | — |
| `ceo_salary` | 1,500,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 28,137,798 |
| `ceo_option_awards` | 3,967,113 |
| `ceo_non_equity_incentive` | 2,220,000 |
| `ceo_all_other_comp` | 515,965 |
| `ceo_total_comp` | 36,340,876 |
| `ceo_equity_pay_pct` | 0.883 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 87,006,439 |
| `insider_ownership_pct` | 0.011 |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 6 |
| `say_on_pay_support_pct` | 0.94 |
| `ceo_pay_ratio` | 746 |
| `median_employee_pay` | 48,695 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 1 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | PricewaterhouseCoopers LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 24,800,000 |
| `audit_fees_audit` | 22,800,000 |
| `audit_fees_audit_related` | 600,000 |
| `audit_fees_tax` | 0 |
| `audit_fees_other` | 1,400,000 |
| `auditor_fees_prior` | 22,200,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Elliott Hill | President and Chief Executive Officer | 2026 | 1,500,000 | 0 | 28,137,798 | 3,967,113 | 2,220,000 | — | 515,965 | 36,340,876 | 1 |
| Elliott Hill | President and Chief Executive Officer | 2025 | 951,923 | 4,000,000 | 14,887,893 | 5,832,678 | 0 | — | 345,574 | 26,018,068 | 1 |
| Matthew Friend | Executive Vice President and Chief Financial Officer | 2026 | 1,250,000 | 0 | 10,680,353 | 2,047,547 | 1,170,000 | — | 17,500 | 15,165,400 | 1 |
| Matthew Friend | Executive Vice President and Chief Financial Officer | 2025 | 1,250,000 | 0 | 9,788,426 | 3,048,046 | 0 | — | 17,250 | 14,103,722 | 1 |
| Matthew Friend | Executive Vice President and Chief Financial Officer | 2024 | 1,298,077 | 0 | 5,221,473 | 2,878,629 | 975,000 | 975,000 | 17,331 | 10,390,510 | 0 |
| Venkatesh Alagirisamy | Executive Vice President, Chief Operating Officer | 2026 | 960,096 | 0 | 8,071,579 | 1,564,777 | 900,900 | — | 17,500 | 11,514,852 | 1 |
| Treasure Heinle | Executive Vice President, Chief People Officer | 2026 | 984,615 | 0 | 8,626,384 | 1,407,691 | 936,000 | — | 17,500 | 11,972,190 | 1 |
| Robert Leinwand | Executive Vice President, Chief Legal Officer | 2026 | 1,005,769 | 0 | 8,626,384 | 1,407,691 | 959,400 | — | 13,877 | 12,013,121 | 1 |
| Robert Leinwand | Executive Vice President, Chief Legal Officer | 2025 | 795,749 | 0 | 3,875,030 | 2,162,574 | 0 | — | 21,128 | 6,854,481 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Cathleen Benko | 2026 | 27,473 | 0 | — | — | — | 14,756 | 42,229 | 1 |
| Timothy Cook | 2026 | 165,000 | 192,758 | — | — | — | 20,000 | 377,758 | 1 |
| Thasunda Duckett | 2026 | 100,000 | 192,758 | — | — | — | 20,000 | 312,758 | 1 |
| Mónica Gil | 2026 | 100,000 | 192,758 | — | — | — | 0 | 292,758 | 1 |
| Maria Henry | 2026 | 105,000 | 192,758 | — | — | — | 20,000 | 317,758 | 1 |
| Peter Henry | 2026 | 105,000 | 192,758 | — | — | — | 0 | 297,758 | 1 |
| Travis Knight | 2026 | 100,000 | 192,758 | — | — | — | 0 | 292,758 | 1 |
| Jørgen Vig Knudstorp | 2026 | 72,802 | 385,517 | — | — | — | 0 | 458,319 | 1 |
| Michelle Peluso | 2026 | 125,000 | 192,758 | — | — | — | 20,000 | 337,758 | 1 |
| John Rogers, Jr. | 2026 | 100,000 | 192,758 | — | — | — | 0 | 292,758 | 1 |
| Robert Swan | 2026 | 135,000 | 192,758 | — | — | — | 20,000 | 347,758 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| Timothy Cook | director_officer | 130,480 | — |
| Thasunda Duckett | director_officer | 13,589 | — |
| Mónica Gil | director_officer | 8,893 | — |
| Maria Henry | director_officer | 8,767 | — |
| Peter Henry | director_officer | 11,099 | — |
| Elliott Hill | director_officer | 225,951 | — |
| Travis Knight | director_officer | 9,533,940 | 0.008 |
| Jørgen Vig Knudstorp | director_officer | 21,388 | — |
| Mark Parker | director_officer | 1,924,163 | 0.002 |
| Michelle Peluso | director_officer | 32,814 | — |
| John Rogers, Jr. | director_officer | 41,022 | — |
| Robert Swan | director_officer | 55,558 | — |
| Matthew Friend | director_officer | 460,050 | — |
| Venkatesh Alagirisamy | director_officer | 189,634 | — |
| Treasure Heinle | director_officer | 90,608 | — |
| Robert Leinwand | director_officer | 216,591 | — |
| Sojitz Corporation of America | 5pct_holder | 300,000 | 1 |
| Philip Knight | 5pct_holder | 27,479,487 | 0.098 |
| Philip Knight | 5pct_holder | 35,815,174 | 0.029 |
| Swoosh, LLC | 5pct_holder | 221,750,000 | 0.788 |
| Swoosh, LLC | 5pct_holder | 221,750,000 | 0.156 |
| Travis A. Knight 2009 Irrevocable Trust II | 5pct_holder | 24,681,369 | 0.088 |
| Travis A. Knight 2009 Irrevocable Trust II | 5pct_holder | 24,681,369 | 0.02 |
| Vanguard Capital Management | 5pct_holder | 89,476,687 | 0.075 |
| Vanguard Capital Management | 5pct_holder | 89,476,687 | 0.075 |
| State Street Corporation | 5pct_holder | 59,588,679 | 0.05 |
| State Street Corporation | 5pct_holder | 59,588,679 | 0.05 |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Cathleen Benko | — | — | — | female | name | — |
| Timothy Cook | 65 | 21 | 1 | male | honorific | 1 |
| Thasunda Duckett | 52 | 7 | — | female | honorific | 0 |
| Mónica Gil | 54 | 4 | — | female | honorific | 0 |
| Maria Henry | 59 | 3 | — | female | honorific | 2 |
| Peter Henry | 56 | 8 | — | male | pronoun | 2 |
| Travis Knight | 52 | 11 | — | male | honorific | 0 |
| Jørgen Vig Knudstorp | 57 | 1 | — | male | honorific | 1 |
| Michelle Peluso | 54 | 12 | — | female | honorific | 0 |
| John Rogers, Jr. | — | — | — | male | name | — |
| Robert Swan | 66 | 4 | — | male | honorific | 2 |

---

## PEG — filed 2026-03-12

- source: <https://www.sec.gov/Archives/edgar/data/788784/000119312526104270/peg-20260310.htm>
- accession: `0001193125-26-104270`
- carve payload: 43,725 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | — |
| `fiscal_year_extract` | 2025 |
| `board_size` | 11 |
| `n_directors` | 11 |
| `pct_female_directors` | 0.455 |
| `pct_gender_stated` | 0.273 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 65.273 |
| `avg_board_tenure` | 5.1 |
| `pct_independent_directors` | 0.909 |
| `ceo_name_proxy` | Ralph A. LaRossa |
| `ceo_age` | 62 |
| `ceo_since_year` | 2022 |
| `ceo_salary` | 1,386,000 |
| `ceo_bonus` | — |
| `ceo_stock_awards` | 9,000,153 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 2,563,400 |
| `ceo_all_other_comp` | 183,181 |
| `ceo_total_comp` | 13,866,735 |
| `ceo_equity_pay_pct` | 0.649 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 28,855,470 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | 0.938 |
| `ceo_pay_ratio` | 79 |
| `median_employee_pay` | 174,510 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | Deloitte |
| `auditor_since_year` | — |
| `auditor_fees` | 9,400,000 |
| `audit_fees_audit` | 8,500,000 |
| `audit_fees_audit_related` | 0 |
| `audit_fees_tax` | 900,000 |
| `audit_fees_other` | 0 |
| `auditor_fees_prior` | 8,900,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ralph A. LaRossa | Chair of the Board, President & CEO | 2025 | 1,386,000 | — | 9,000,153 | — | 2,563,400 | 734,000 | 183,181 | 13,866,735 | 1 |
| Ralph A. LaRossa | Chair of the Board, President & CEO | 2024 | 1,345,600 | — | 8,500,035 | — | 2,470,500 | 0 | 51,825 | 12,367,961 | 1 |
| Ralph A. LaRossa | Chair of the Board, President & CEO | 2023 | 1,293,800 | — | 8,000,097 | — | 1,833,300 | 611,000 | 40,666 | 11,778,863 | 1 |
| Daniel J. Cregg | EVP & CFO | 2025 | 854,000 | — | 2,500,112 | — | 1,053,000 | 381,000 | 31,003 | 4,819,115 | 1 |
| Daniel J. Cregg | EVP & CFO | 2024 | 825,000 | — | 2,400,086 | — | 1,009,800 | 0 | 42,938 | 4,277,824 | 1 |
| Daniel J. Cregg | EVP & CFO | 2023 | 780,500 | — | 3,950,148 | — | 680,600 | 410,000 | 31,835 | 5,853,083 | 1 |
| Kim C. Hanemann | President & COO (PSE&G) | 2025 | 802,100 | — | 2,000,089 | — | 1,036,300 | 473,000 | 39,185 | 4,350,674 | 1 |
| Kim C. Hanemann | President & COO (PSE&G) | 2024 | 775,000 | — | 1,800,019 | — | 869,600 | 146,000 | 30,526 | 3,621,145 | 1 |
| Kim C. Hanemann | President & COO (PSE&G) | 2023 | 671,600 | — | 1,400,115 | — | 569,200 | 405,000 | 76,356 | 3,122,271 | 1 |
| Grace H. Park | EVP & General Counsel | 2025 | 700,000 | — | 1,400,046 | — | 767,200 | 65,000 | 36,797 | 2,969,043 | 1 |
| Grace H. Park | EVP & General Counsel | 2024 | 0 | — | 0 | — | 0 | 0 | 0 | 0 | 1 |
| Grace H. Park | EVP & General Counsel | 2023 | 0 | — | 0 | — | 0 | 0 | 0 | 0 | 1 |
| Charles V. McFeaters | President & Chief Nuclear Officer | 2025 | 700,000 | — | 1,300,106 | — | 724,500 | 96,000 | 29,297 | 2,849,903 | 1 |
| Charles V. McFeaters | President & Chief Nuclear Officer | 2024 | 676,000 | — | 1,200,075 | — | 684,500 | 82,000 | 29,169 | 2,671,744 | 1 |
| Charles V. McFeaters | President & Chief Nuclear Officer | 2023 | 571,910 | — | 1,000,088 | — | 411,600 | 70,000 | 28,185 | 2,081,783 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Willie A. Deese | 2025 | 145,000 | 180,053 | 0 | 0 | 0 | 8,650 | 333,703 | 1 |
| Jamie M. Gentoso | 2025 | 120,000 | 180,053 | 0 | 0 | 0 | 7,650 | 307,703 | 1 |
| Barry H. Ostrowsky | 2025 | 150,000 | 180,053 | 0 | 0 | 0 | 1,150 | 331,203 | 1 |
| Ricardo G. Pérez | 2025 | 120,000 | 180,053 | 0 | 0 | 0 | 1,150 | 301,203 | 1 |
| Valerie A. Smith | 2025 | 120,000 | 180,053 | 0 | 0 | 0 | 8,650 | 308,703 | 1 |
| Scott G. Stephenson | 2025 | 145,000 | 180,053 | 0 | 0 | 0 | 8,650 | 333,703 | 1 |
| Laura A. Sugg | 2025 | 145,000 | 180,053 | 0 | 0 | 0 | 150 | 325,203 | 1 |
| John P. Surma | 2025 | 150,000 | 180,053 | 0 | 0 | 0 | 8,650 | 338,703 | 1 |
| Kenneth Y. Tanji | 2025 | 120,000 | 180,053 | 0 | 0 | 0 | 8,650 | 308,703 | 1 |
| Susan Tomasky | 2025 | 160,000 | 180,053 | 0 | 0 | 0 | 1,150 | 341,203 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| Willie A. Deese | director_officer | 29,974 | — |
| Jamie M. Gentoso | director_officer | 11,142 | — |
| Barry H. Ostrowsky | director_officer | 23,204 | — |
| Ricardo G. Pérez | director_officer | 5,059 | — |
| Valerie A. Smith | director_officer | 11,142 | — |
| Scott G. Stephenson | director_officer | 17,162 | — |
| Laura A. Sugg | director_officer | 20,034 | — |
| John P. Surma | director_officer | 18,615 | — |
| Kenneth Y. Tanji | director_officer | 7,145 | — |
| Susan Tomasky | director_officer | 50,532 | — |
| Geisha J. Williams | director_officer | 0 | — |
| Ralph A. LaRossa | director_officer | 351,589 | — |
| Daniel J. Cregg | director_officer | 179,870 | — |
| Kim C. Hanemann | director_officer | 81,753 | — |
| Grace H. Park | director_officer | 14,995 | — |
| Charles V. McFeaters | director_officer | 31,483 | — |
| BlackRock, Inc. | 5pct_holder | 46,597,183 | 0.093 |
| State Street Corporation | 5pct_holder | 30,417,859 | 0.0609 |
| Vanguard Group, Inc. | 5pct_holder | 67,498,797 | 0.1352 |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Ralph A. LaRossa | 62 | 3 | 0 | male | honorific | 0 |
| Susan Tomasky | 73 | 13 | 1 | female | name | — |
| Willie A. Deese | 70 | 9 | 1 | male | name | 1 |
| Jamie M. Gentoso | 49 | 3 | 1 | female | honorific | 0 |
| Ricardo G. Pérez | 66 | 1 | 1 | male | name | 0 |
| Valerie A. Smith | 70 | 3 | 1 | female | name | 0 |
| Scott G. Stephenson | 68 | 5 | 1 | male | name | 1 |
| Laura A. Sugg | 65 | 6 | 1 | female | name | 2 |
| John P. Surma | 71 | 6 | 1 | male | honorific | 2 |
| Kenneth Y. Tanji | 60 | 2 | 1 | male | name | 1 |
| Geisha J. Williams | 64 | — | 1 | female | name | 2 |

---

## PFE — filed 2026-03-12

- source: <https://www.sec.gov/Archives/edgar/data/78003/000007800326000033/pfe-20260312.htm>
- accession: `0000078003-26-000033`
- carve payload: 41,139 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Pfizer Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 12 |
| `n_directors` | 14 |
| `pct_female_directors` | 0.286 |
| `pct_gender_stated` | 0.143 |
| `n_women_directors_vs_inferred` | — |
| `avg_director_age` | 64 |
| `avg_board_tenure` | 7.9 |
| `pct_independent_directors` | — |
| `ceo_name_proxy` | Albert Bourla, DVM, Ph.D. |
| `ceo_age` | 64 |
| `ceo_since_year` | 2019 |
| `ceo_salary` | 1,800,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 9,442,705 |
| `ceo_option_awards` | 8,998,141 |
| `ceo_non_equity_incentive` | 5,400,000 |
| `ceo_all_other_comp` | 1,940,441 |
| `ceo_total_comp` | 27,585,301 |
| `ceo_equity_pay_pct` | 0.669 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 65,797,025 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | 0.547 |
| `ceo_pay_ratio` | — |
| `median_employee_pay` | — |
| `independent_chair` | — |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | KPMG LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 26,747,000 |
| `audit_fees_audit` | 24,154,000 |
| `audit_fees_audit_related` | 763,000 |
| `audit_fees_tax` | 1,830,000 |
| `audit_fees_other` | — |
| `auditor_fees_prior` | 43,258,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Bourla | Chairman and Chief Executive Officer | 2025 | 1,800,000 | 0 | 9,442,705 | 8,998,141 | 5,400,000 | 4,014 | 1,940,441 | 27,585,301 | 1 |
| A. Bourla | Chairman and Chief Executive Officer | 2024 | 1,800,000 | 0 | 4,838,694 | 9,993,969 | 7,020,000 | 0 | 996,064 | 24,648,727 | 1 |
| A. Bourla | Chairman and Chief Executive Officer | 2023 | 1,787,500 | 0 | 8,745,187 | 8,761,683 | 0 | 8,440 | 2,259,254 | 21,562,064 | 1 |
| D. Denton | Chief Financial Officer, EVP | 2025 | 1,388,964 | 0 | 2,367,893 | 2,499,485 | 2,800,000 | 0 | 613,840 | 9,670,182 | 1 |
| D. Denton | Chief Financial Officer, EVP | 2024 | 1,346,925 | 0 | 1,130,214 | 2,447,055 | 2,963,400 | 0 | 236,948 | 8,124,542 | 1 |
| D. Denton | Chief Financial Officer, EVP | 2023 | 1,296,875 | 0 | 1,341,079 | 2,190,418 | 0 | 0 | 451,164 | 5,279,536 | 1 |
| C. Boshoff | Chief Scientific Officer and President, Research & Development | 2025 | 1,400,000 | 0 | 1,808,655 | 2,999,378 | 3,150,000 | 0 | 648,006 | 10,006,039 | 1 |
| C. Boshoff | Chief Scientific Officer and President, Research & Development | 2024 | 1,200,000 | 0 | 896,297 | 2,249,857 | 2,880,000 | 0 | 627,830 | 7,853,984 | 1 |
| A. Malik | Chief U.S. Commercial Officer, EVP | 2025 | 1,386,817 | 0 | 2,281,630 | 2,249,535 | 2,900,000 | 0 | 611,245 | 9,429,227 | 1 |
| A. Malik | Chief U.S. Commercial Officer, EVP | 2024 | 1,344,825 | 0 | 1,127,148 | 2,475,933 | 3,093,200 | 0 | 253,164 | 8,294,270 | 1 |
| A. Malik | Chief U.S. Commercial Officer, EVP | 2023 | 1,294,800 | 0 | 1,336,257 | 2,190,418 | 0 | 0 | 599,259 | 5,420,734 | 1 |
| D. Lankler | Chief Legal Officer, EVP | 2025 | 1,411,787 | 0 | 1,976,390 | 2,249,535 | 2,675,000 | 257,976 | 535,588 | 9,106,276 | 1 |
| D. Lankler | Chief Legal Officer, EVP | 2024 | 1,244,588 | 0 | 913,346 | 1,935,758 | 2,464,400 | 11,244 | 244,779 | 6,814,115 | 1 |
| D. Lankler | Chief Legal Officer, EVP | 2023 | 1,198,313 | 0 | 1,817,208 | 1,703,661 | 0 | 196,724 | 561,892 | 5,477,798 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Ronald E. Blaylock | — | 155,000 | 205,000 | — | — | — | 0 | 360,000 | 1 |
| Mortimer J. Buckley | — | 155,000 | 205,000 | — | — | — | 0 | 360,000 | 1 |
| Susan Desmond-Hellmann, MD, M.P.H. | — | 175,522 | 205,000 | — | — | — | 0 | 380,522 | 1 |
| Joseph J. Echevarria | — | 185,000 | 205,000 | — | — | — | 10,372 | 400,372 | 1 |
| Scott Gottlieb, MD | — | 185,000 | 205,000 | — | — | — | 3,500 | 393,500 | 1 |
| Helen H. Hobbs, MD | — | 58,448 | 0 | — | — | — | 0 | 58,448 | 1 |
| Susan Hockfield, Ph.D. | — | 155,000 | 205,000 | — | — | — | 12,000 | 372,000 | 1 |
| Dan R. Littman, MD, Ph.D. | — | 155,000 | 205,000 | — | — | — | 20,000 | 380,000 | 1 |
| Shantanu Narayen | — | 205,000 | 205,000 | — | — | — | 0 | 410,000 | 1 |
| Suzanne Nora Johnson | — | 185,000 | 205,000 | — | — | — | 20,000 | 410,000 | 1 |
| James Quincey | — | 155,000 | 205,000 | — | — | — | 20,000 | 380,000 | 1 |
| James C. Smith | — | 185,000 | 205,000 | — | — | — | 0 | 390,000 | 1 |
| Cyrus Taraporevala | — | 155,000 | 205,000 | — | — | — | 20,000 | 380,000 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 506,479,807 | 0.0897 |
| BlackRock, Inc. | 5pct_holder | 434,748,255 | 0.077 |
| State Street Corporation State Street Financial Center One Congress Street, Suite | 5pct_holder | 287,875,814 | 0.051 |
| Ronald E. Blaylock | director_officer | 32,457 | — |
| Albert Bourla, DVM, Ph.D. | director_officer | 378,551 | — |
| Chris Boshoff, MD, FRCP, FMedSci, Ph.D. | director_officer | 114,055 | — |
| Mortimer J. Buckley | director_officer | — | — |
| David M. Denton | director_officer | 37,919 | — |
| Susan Desmond-Hellmann, MD, M.P.H. | director_officer | 3,408 | — |
| Joseph J. Echevarria | director_officer | — | — |
| Scott Gottlieb, MD | director_officer | 10,000 | — |
| Susan Hockfield, Ph.D. | director_officer | — | — |
| Douglas M. Lankler | director_officer | 161,363 | — |
| Dan R. Littman, MD, Ph.D. | director_officer | — | — |
| Aamir Malik | director_officer | 29,548 | — |
| Shantanu Narayen | director_officer | — | — |
| Suzanne Nora Johnson | director_officer | 10,000 | — |
| James Quincey | director_officer | — | — |
| James C. Smith | director_officer | 3,542 | — |
| Cyrus Taraporevala | director_officer | 10,000 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Ronald E. Blaylock | 66 | 8 | — | male | name | — |
| Albert Bourla, DVM, Ph.D. | 64 | 7 | — | male | pronoun | 0 |
| Mortimer J. Buckley | 56 | 1 | — | male | pronoun | — |
| Susan Desmond-Hellmann, MD, M.P.H. | 68 | 5 | — | female | pronoun | — |
| Joseph J. Echevarria | 69 | 10 | — | male | pronoun | — |
| Scott Gottlieb, MD | 53 | 6 | — | male | pronoun | — |
| Helen H. Hobbs, MD | — | — | — | female | name | — |
| Susan Hockfield, Ph.D. | — | — | — | female | pronoun | — |
| Dan R. Littman, MD, Ph.D. | 73 | 7 | — | male | pronoun | — |
| Shantanu Narayen | 62 | 12 | — | male | pronoun | — |
| Suzanne Nora Johnson | 68 | 18 | — | female | honorific | — |
| James Quincey | 61 | 5 | — | male | honorific | — |
| James C. Smith | — | — | — | male | name | — |
| Cyrus Taraporevala | — | — | — | male | name | — |

---

## PG — filed 2026-08-28

- source: <https://www.sec.gov/Archives/edgar/data/80424/000119312526372211/pg-20260826.htm>
- accession: `0001193125-26-372211`
- carve payload: 44,007 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | The Procter & Gamble Company |
| `fiscal_year_extract` | 2026 |
| `board_size` | 12 |
| `n_directors` | 15 |
| `pct_female_directors` | 0.333 |
| `pct_gender_stated` | 1 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 59.625 |
| `avg_board_tenure` | 4.875 |
| `pct_independent_directors` | 0.467 |
| `ceo_name_proxy` | Shailesh G. Jejurikar |
| `ceo_age` | 59 |
| `ceo_since_year` | 2026 |
| `ceo_salary` | 1,387,500 |
| `ceo_bonus` | 1,634,720 |
| `ceo_stock_awards` | 11,035,209 |
| `ceo_option_awards` | 3,500,010 |
| `ceo_non_equity_incentive` | 0 |
| `ceo_all_other_comp` | 390,303 |
| `ceo_total_comp` | 18,976,742 |
| `ceo_equity_pay_pct` | 0.766 |
| `n_neos` | 8 |
| `sct_years` | 3 |
| `total_neo_comp` | 79,810,348 |
| `insider_ownership_pct` | 0.0022 |
| `ceo_ownership_pct` | 0.06 |
| `n_five_percent_holders` | — |
| `say_on_pay_support_pct` | 0.922 |
| `ceo_pay_ratio` | — |
| `median_employee_pay` | — |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | Deloitte & Touche LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 30,527,000 |
| `audit_fees_audit` | 27,827,000 |
| `audit_fees_audit_related` | 2,166,000 |
| `audit_fees_tax` | 189,000 |
| `audit_fees_other` | 345,000 |
| `auditor_fees_prior` | 30,862,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shailesh G. Jejurikar | President and Chief Executive Officer | 2026 | 1,387,500 | 1,634,720 | 11,035,209 | 3,500,010 | 0 | 1,029,000 | 390,303 | 18,976,742 | 1 |
| Shailesh G. Jejurikar | President and Chief Executive Officer | 2025 | 1,162,500 | 873,730 | 3,640,460 | 3,432,016 | 0 | 401,000 | 81,981 | 9,591,687 | 1 |
| Shailesh G. Jejurikar | President and Chief Executive Officer | 2024 | 1,106,250 | 1,867,613 | 3,477,569 | 3,150,023 | 0 | 280,000 | 76,632 | 9,958,087 | 1 |
| Jon R. Moeller | Executive Chairman of the Board | 2026 | 1,425,000 | 1,532,550 | 10,182,573 | 5,625,025 | 0 | 0 | 308,761 | 19,073,909 | 1 |
| Jon R. Moeller | Executive Chairman of the Board | 2025 | 1,637,500 | 1,887,600 | 11,519,645 | 6,562,515 | 0 | 0 | 302,556 | 21,909,816 | 1 |
| Jon R. Moeller | Executive Chairman of the Board | 2024 | 1,600,000 | 4,086,400 | 11,301,824 | 5,600,006 | 0 | 0 | 375,651 | 22,963,881 | 1 |
| Andre Schulten | Chief Financial Officer | 2026 | 1,095,000 | 767,177 | 4,696,981 | 2,574,031 | 0 | 113,000 | 97,833 | 9,344,022 | 1 |
| Andre Schulten | Chief Financial Officer | 2025 | 1,037,500 | 690,690 | 3,378,353 | 3,120,022 | 0 | 193,000 | 95,941 | 8,515,506 | 1 |
| Andre Schulten | Chief Financial Officer | 2024 | 980,000 | 1,468,550 | 4,569,186 | 1,406,270 | 0 | 143,000 | 108,831 | 8,675,837 | 1 |
| Gary A. Coombe | CEO - Grooming | 2026 | 1,042,500 | 505,943 | 2,849,231 | 1,549,895 | 0 | 0 | 1,078,932 | 7,026,501 | 1 |
| Marc S. Pritchard | Chief Brand Officer | 2026 | 1,055,000 | 605,207 | 1,924,013 | 1,626,505 | 0 | 0 | 95,127 | 5,305,852 | 1 |
| Sundar G. Raman | CEO - Fabric and Home Care | 2026 | 1,005,000 | 624,824 | 2,588,298 | 2,283,007 | 0 | 100,000 | 573,401 | 7,174,530 | 1 |
| Sundar G. Raman | CEO - Fabric and Home Care | 2025 | 912,500 | 737,153 | 2,289,194 | 2,095,524 | 0 | 67,000 | 1,082,180 | 7,183,551 | 1 |
| Jennifer L. Davis | Former CEO - Health Care | 2026 | 1,000,000 | 651,771 | 2,583,671 | 2,283,007 | 0 | 0 | 89,983 | 6,608,432 | 1 |
| Jennifer L. Davis | Former CEO - Health Care | 2025 | 895,000 | 750,602 | 2,248,178 | 2,057,435 | 0 | 0 | 88,831 | 6,040,046 | 1 |
| Ma. Fatima D. Francisco | Former CEO - Baby, Feminine and Family Care | 2026 | 1,068,750 | 599,118 | 2,391,428 | 2,066,505 | 0 | 70,000 | 104,559 | 6,300,360 | 1 |
| Ma. Fatima D. Francisco | Former CEO - Baby, Feminine and Family Care | 2025 | 1,037,500 | 591,373 | 2,250,472 | 2,025,275 | 0 | 298,000 | 118,243 | 6,320,863 | 1 |
| Ma. Fatima D. Francisco | Former CEO - Baby, Feminine and Family Care | 2024 | 975,000 | 1,490,688 | 2,317,734 | 2,027,011 | 0 | 370,000 | 112,275 | 7,292,708 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| B. Marc Allen | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Craig Arnold | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| M. Brett Biggs | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Sheila Bonini | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Amy L. Chang | — | 140,000 | 220,000 | — | — | — | — | 360,000 | 1 |
| Joseph Jimenez | — | 190,000 | 220,000 | — | — | — | — | 410,000 | 1 |
| Christopher Kempczinski | — | 142,147 | 220,000 | — | — | — | — | 362,147 | 1 |
| Debra L. Lee | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Terry J. Lundgren | — | 39,524 | — | — | — | — | — | 39,524 | 1 |
| Christine M. McCarthy | — | 150,000 | 220,000 | — | — | — | — | 370,000 | 1 |
| Ashley McEvoy | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Robert Portman | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Rajesh Subramaniam | — | 120,000 | 220,000 | — | — | — | — | 340,000 | 1 |
| Patricia A. Woertz | — | 34,565 | — | — | — | — | — | 34,565 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| B. Marc Allen | director_officer | — | — |
| Craig Arnold | director_officer | — | — |
| Brett Biggs | director_officer | — | — |
| Sheila Bonini | director_officer | — | — |
| Amy L. Chang | director_officer | — | — |
| Gary A. Coombe | director_officer | 523,152 | 0.06 |
| Jennifer L. Davis | director_officer | 297,883 | 0.06 |
| Ma. Fatima D. Francisco | director_officer | 374,452 | 0.06 |
| Shailesh G. Jejurikar | director_officer | 834,315 | 0.06 |
| Joseph Jimenez | director_officer | 12,468 | 0.06 |
| Christopher Kempczinski | director_officer | — | — |
| Debra L. Lee | director_officer | — | — |
| Christine M. McCarthy | director_officer | — | — |
| Ashley McEvoy | director_officer | 1,413 | 0.06 |
| Jon R. Moeller | director_officer | 865,233 | 0.06 |
| Robert J. Portman | director_officer | — | — |
| Marc S. Pritchard | director_officer | 795,365 | 0.06 |
| Sundar G. Raman | director_officer | 347,380 | 0.06 |
| Andre Schulten | director_officer | 272,898 | 0.06 |
| Rajesh Subramaniam | director_officer | — | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Sheila Bonini | 62 | 3 | 1 | female | honorific | — |
| M. Brett Biggs | 58 | 3 | 1 | male | honorific | — |
| Amy L. Chang | 49 | 9 | 1 | female | honorific | — |
| Joseph Jimenez | 66 | 8 | 1 | male | honorific | — |
| Christopher Kempczinski | 57 | 5 | 1 | male | honorific | — |
| Shailesh G. Jejurikar | 59 | 1 | 0 | male | honorific | — |
| Christine M. McCarthy | 71 | 7 | 1 | female | honorific | — |
| Ashley McEvoy | 55 | 3 | 1 | female | honorific | — |
| B. Marc Allen | — | — | — | — | — | — |
| Craig Arnold | — | — | — | male | honorific | — |
| Debra L. Lee | — | — | — | — | — | — |
| Terry J. Lundgren | — | — | — | — | — | — |
| Robert Portman | — | — | — | — | — | — |
| Rajesh Subramaniam | — | — | — | — | — | — |
| Patricia A. Woertz | — | — | — | — | — | — |

---

## REG — filed 2026-03-25

- source: <https://www.sec.gov/Archives/edgar/data/910606/000119312526122880/d939595ddef14a.htm>
- accession: `0001193125-26-122880`
- carve payload: 38,771 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Regency Centers Corporation |
| `fiscal_year_extract` | 2025 |
| `board_size` | 11 |
| `n_directors` | 11 |
| `pct_female_directors` | 0.364 |
| `pct_gender_stated` | 0.909 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 62.727 |
| `avg_board_tenure` | 8.545 |
| `pct_independent_directors` | 0.818 |
| `ceo_name_proxy` | Lisa Palmer |
| `ceo_age` | 58 |
| `ceo_since_year` | 2020 |
| `ceo_salary` | 1,050,000 |
| `ceo_bonus` | — |
| `ceo_stock_awards` | 7,099,098 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 2,925,000 |
| `ceo_all_other_comp` | 18,334 |
| `ceo_total_comp` | 11,092,432 |
| `ceo_equity_pay_pct` | 0.64 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 23,473,092 |
| `insider_ownership_pct` | 0.01 |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 5 |
| `say_on_pay_support_pct` | — |
| `ceo_pay_ratio` | 72 |
| `median_employee_pay` | 154,922 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | 0 |
| `majority_voting` | 1 |
| `auditor_name` | KPMG LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 2,898,130 |
| `audit_fees_audit` | 2,380,000 |
| `audit_fees_audit_related` | 0 |
| `audit_fees_tax` | 518,130 |
| `audit_fees_other` | 0 |
| `auditor_fees_prior` | 2,560,437 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Martin E. Stein, Jr. Executive | Executive Chairman of the Board | 2025 | 500,000 | — | 851,905 | — | 0 | — | 40,828 | 1,392,733 | 1 |
| Martin E. Stein, Jr. Executive | Executive Chairman of the Board | 2024 | 500,000 | — | 712,882 | — | 0 | — | 40,628 | 1,253,510 | 1 |
| Martin E. Stein, Jr. Executive | Executive Chairman of the Board | 2023 | 500,000 | — | 1,034,858 | — | 0 | — | 40,728 | 1,575,586 | 1 |
| Lisa Palmer | President and Chief Executive Officer | 2025 | 1,050,000 | — | 7,099,098 | — | 2,925,000 | — | 18,334 | 11,092,432 | 1 |
| Lisa Palmer | President and Chief Executive Officer | 2024 | 1,030,000 | — | 5,702,941 | — | 2,805,000 | — | 21,322 | 9,559,263 | 1 |
| Lisa Palmer | President and Chief Executive Officer | 2023 | 1,000,000 | — | 5,536,318 | — | 2,712,500 | — | 20,217 | 9,269,035 | 1 |
| Michael J. Mas | Executive Vice President, Chief Financial Officer | 2025 | 640,000 | — | 2,226,325 | — | 1,537,500 | — | 15,574 | 4,419,399 | 1 |
| Michael J. Mas | Executive Vice President, Chief Financial Officer | 2024 | 620,000 | — | 1,805,910 | — | 1,470,000 | — | 15,630 | 3,911,540 | 1 |
| Michael J. Mas | Executive Vice President, Chief Financial Officer | 2023 | 600,000 | — | 1,759,195 | — | 1,116,000 | — | 17,078 | 3,492,273 | 1 |
| Alan T. Roth East Region | East Region President and Chief Operating Officer | 2025 | 625,000 | — | 1,703,810 | — | 937,500 | — | 21,438 | 3,287,748 | 1 |
| Alan T. Roth East Region | East Region President and Chief Operating Officer | 2024 | 600,000 | — | 1,330,713 | — | 900,000 | — | 14,270 | 2,844,983 | 1 |
| Alan T. Roth East Region | East Region President and Chief Operating Officer | 2023 | 500,000 | — | 1,134,858 | — | 775,000 | — | 17,126 | 2,426,984 | 1 |
| Nicholas A. Wibbenmeyer West Region | West Region President and Chief Investment Officer | 2025 | 625,000 | — | 1,703,810 | — | 937,500 | — | 14,470 | 3,280,780 | 1 |
| Nicholas A. Wibbenmeyer West Region | West Region President and Chief Investment Officer | 2024 | 600,000 | — | 1,330,713 | — | 900,000 | — | 13,580 | 2,844,293 | 1 |
| Nicholas A. Wibbenmeyer West Region | West Region President and Chief Investment Officer | 2023 | 500,000 | — | 1,134,858 | — | 775,000 | — | 13,680 | 2,423,538 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Gary E. Anderson | 2025 | 100,000 | 125,009 | — | — | — | — | 225,009 | 1 |
| Bryce Blair | 2025 | 120,000 | 125,009 | — | — | — | — | 245,009 | 1 |
| C. Ronald Blankenship | 2025 | 136,745 | 135,019 | — | — | — | — | 271,764 | 1 |
| Kristin A. Campbell | 2025 | 104,766 | 125,009 | — | — | — | — | 229,775 | 1 |
| Deirdre J. Evens | 2025 | 115,000 | 125,009 | — | — | — | — | 240,009 | 1 |
| Thomas W. Furphy | 2025 | 105,000 | 125,009 | — | — | — | — | 230,009 | 1 |
| Karin M. Klein | 2025 | 120,000 | 125,009 | — | — | — | — | 245,009 | 1 |
| Peter D. Linneman | 2025 | 100,000 | 125,009 | — | — | — | — | 225,009 | 1 |
| David P. O’Connor | 2025 | 52,734 | 0 | — | — | — | — | 52,734 | 1 |
| James H. Simmons, III | 2025 | 105,000 | 125,009 | — | — | — | — | 230,009 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group, Inc. | 5pct_holder | 28,058,356 | 0.1519 |
| BlackRock, Inc. | 5pct_holder | 18,420,218 | 0.101 |
| Norges Bank P.O. Box 1179 Sentrum NO 0107 Oslo Norway | 5pct_holder | 17,018,543 | 0.0921 |
| State Street Corporation One Lincoln Street Boston, MA 02111 | 5pct_holder | 12,892,463 | 0.0698 |
| JPMorgan Chase & Co. | 5pct_holder | 10,843,223 | 0.059 |
| Martin E. Stein, Jr. | director_officer | 595,295 | — |
| Gary E. Anderson | director_officer | 1,421 | — |
| Bryce Blair | director_officer | 31,196 | — |
| C. Ronald Blankenship | director_officer | 112,351 | — |
| Kristin A. Campbell | director_officer | 4,990 | — |
| Deirdre J. Evens | director_officer | 20,732 | — |
| Thomas W. Furphy | director_officer | 12,509 | — |
| Karin M. Klein | director_officer | 23,198 | — |
| Peter D. Linneman | director_officer | 53,600 | — |
| Lisa Palmer | director_officer | 157,942 | — |
| Mark J. Parrell | director_officer | 0 | — |
| James H. Simmons, III | director_officer | 6,242 | — |
| Michael J. Mas | director_officer | 54,849 | — |
| Alan T. Roth | director_officer | 25,519 | — |
| Nicholas A. Wibbenmeyer | director_officer | 40,995 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Gary E. Anderson | 60 | 2 | 1 | male | honorific | 0 |
| Bryce Blair | 67 | 12 | 1 | male | honorific | 1 |
| Kristin A. Campbell | 64 | 3 | 1 | female | stated | 0 |
| Deirdre J. Evens | 62 | 8 | 1 | female | stated | 0 |
| Thomas W. Furphy | 59 | 7 | 1 | male | honorific | 0 |
| Karin M. Klein | 54 | 7 | 1 | female | stated | 0 |
| Peter D. Linneman | 75 | 9 | 1 | male | pronoun | 0 |
| Lisa Palmer | 58 | 8 | 0 | female | stated | 0 |
| Mark J. Parrell | 59 | 0 | 1 | male | honorific | 1 |
| James H. Simmons, III | 59 | 5 | 1 | male | honorific | 1 |
| Martin E. Stein, Jr. | 73 | 33 | 0 | male | honorific | 1 |

---

## SBUX — filed 2026-01-26

- source: <https://www.sec.gov/Archives/edgar/data/829224/000121390026007780/ea0270973-01.htm>
- accession: `0001213900-26-007780`
- carve payload: 39,981 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Starbucks Corporation |
| `fiscal_year_extract` | 2025 |
| `board_size` | 11 |
| `n_directors` | 11 |
| `pct_female_directors` | 0.364 |
| `pct_gender_stated` | 0.444 |
| `n_women_directors_vs_inferred` | 1 |
| `avg_director_age` | 56.2 |
| `avg_board_tenure` | 2.7 |
| `pct_independent_directors` | 0.727 |
| `ceo_name_proxy` | Brian Niccol |
| `ceo_age` | 52 |
| `ceo_since_year` | 2024 |
| `ceo_salary` | 1,599,998 |
| `ceo_bonus` | 5,000,000 |
| `ceo_stock_awards` | 19,881,585 |
| `ceo_option_awards` | — |
| `ceo_non_equity_incentive` | 1,971,000 |
| `ceo_all_other_comp` | 2,540,190 |
| `ceo_total_comp` | 30,992,773 |
| `ceo_equity_pay_pct` | 0.641 |
| `n_neos` | 7 |
| `sct_years` | 3 |
| `total_neo_comp` | 84,547,353 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 3 |
| `say_on_pay_support_pct` | — |
| `ceo_pay_ratio` | 1,794 |
| `median_employee_pay` | 17,279 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | Deloitte & Touche LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 10,119,000 |
| `audit_fees_audit` | 9,765,000 |
| `audit_fees_audit_related` | 6,000 |
| `audit_fees_tax` | 117,000 |
| `audit_fees_other` | 231,000 |
| `auditor_fees_prior` | 9,680,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Brian Niccol | chairman and chief executive officer | 2025 | 1,599,998 | 5,000,000 | 19,881,585 | — | 1,971,000 | — | 2,540,190 | 30,992,773 | 1 |
| Brian Niccol | chairman and chief executive officer | 2024 | 61,538 | 5,000,000 | 90,291,772 | — | 30,295 | — | 418,071 | 95,801,676 | 1 |
| Cathy Smith | executive vice president, chief financial officer | 2025 | 462,498 | 2,500,000 | 12,459,305 | — | 328,697 | — | 15,523 | 15,766,023 | 1 |
| Brady Brewer | chief executive officer, Starbucks International | 2025 | 775,008 | 0 | 9,647,346 | — | 424,313 | — | 101,144 | 10,947,811 | 1 |
| Brady Brewer | chief executive officer, Starbucks International | 2024 | 767,218 | 0 | 4,814,831 | — | 242,833 | — | 116,245 | 5,941,127 | 1 |
| Mike Grams | executive vice president, chief operating officer | 2025 | 457,692 | 500,000 | 7,286,612 | — | 307,442 | — | 114,878 | 8,666,624 | 1 |
| Sara Kelly | executive vice president, chief partner officer | 2025 | 664,424 | 0 | 8,255,917 | — | 364,298 | — | 123,167 | 9,407,806 | 1 |
| Sara Kelly | executive vice president, chief partner officer | 2024 | 631,827 | 0 | 2,757,442 | — | 244,292 | — | 70,993 | 3,562,568 | 0 |
| Sara Kelly | executive vice president, chief partner officer | 2023 | 537,211 | 0 | 1,605,272 | — | 616,082 | — | 25,447 | 2,784,012 | 1 |
| Rachel Ruggeri | former executive vice president, chief financial officer | 2025 | 494,312 | 0 | 4,366,852 | — | 0 | — | 1,921,552 | 6,782,716 | 1 |
| Rachel Ruggeri | former executive vice president, chief financial officer | 2024 | 913,335 | 0 | 5,385,429 | — | 311,738 | — | 16,506 | 6,627,008 | 1 |
| Rachel Ruggeri | former executive vice president, chief financial officer | 2023 | 882,182 | 0 | 4,584,049 | — | 1,267,364 | — | 16,545 | 6,750,140 | 1 |
| Val Bauduin | interim chief financial officer | 2025 | 524,992 | 0 | 1,263,902 | — | 165,309 | — | 29,397 | 1,983,600 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Ritch Allison | — | 0 | 329,921 | 0 | — | — | — | 329,921 | 1 |
| Andy Campion | — | 0 | 339,987 | 0 | — | — | — | 339,987 | 1 |
| Beth Ford | — | 0 | 329,921 | 0 | — | — | — | 329,921 | 1 |
| Mellody Hobson | — | 0 | 0 | 0 | — | — | — | 0 | 1 |
| Jørgen Vig Knudstorp | — | 0 | 514,964 | 0 | — | — | — | 514,964 | 1 |
| Marissa Mayer | — | 0 | 220,784 | 0 | — | — | — | 220,784 | 1 |
| Neal Mohan | — | 0 | 309,985 | 0 | — | — | — | 309,985 | 1 |
| Dambisa Moyo | — | 92,599 | 128,142 | 0 | — | — | — | 220,741 | 1 |
| Daniel Servitje | — | 0 | 309,985 | 0 | — | — | — | 309,985 | 1 |
| Mike Sievert | — | 0 | 309,985 | 0 | — | — | — | 309,985 | 1 |
| Wei Zhang | — | 0 | 309,985 | 0 | — | — | — | 309,985 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| Ritch Allison | director_officer | 10,000 | — |
| Val Bauduin | director_officer | 288 | — |
| Andy Campion | director_officer | 25,870 | — |
| Brady Brewer | director_officer | 46,368 | — |
| Beth Ford | director_officer | 10,634 | — |
| Mike Grams | director_officer | — | — |
| Mellody Hobson | director_officer | 730,255 | — |
| Sara Kelly | director_officer | 25,216 | — |
| Jørgen Vig Knudstorp | director_officer | 102,525 | — |
| Marissa Mayer | director_officer | 2,359 | — |
| Neal Mohan | director_officer | 7,473 | — |
| Dambisa Moyo | director_officer | 1,350 | — |
| Brian Niccol | director_officer | 69,382 | — |
| Rachel Ruggeri | director_officer | 30,258 | — |
| Daniel Servitje | director_officer | 7,473 | — |
| Mike Sievert | director_officer | 7,211 | — |
| Cathy Smith | director_officer | 2,885 | — |
| Wei Zhang | director_officer | 8,655 | — |
| Capital Research Global Investors | 5pct_holder | 76,686,152 | 0.067 |
| Capital World Investors | 5pct_holder | 75,625,746 | 0.066 |
| The Vanguard Group | 5pct_holder | 113,888,463 | 0.1 |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Ritch Allison | 58 | 6 | 1 | male | pronoun | 1 |
| Andy Campion | 54 | 6 | 1 | male | pronoun | 2 |
| Beth Ford | 61 | 2 | 1 | female | honorific | — |
| Jørgen Vig Knudstorp | 57 | 8 | 1 | male | pronoun | 1 |
| Marissa Mayer | 50 | 0 | 1 | female | pronoun | 3 |
| Neal Mohan | 52 | 1 | 1 | male | honorific | — |
| Dambisa Moyo | 57 | 0 | 1 | female | name | 1 |
| Brian Niccol | 52 | 1 | 0 | male | honorific | 1 |
| Daniel Servitje | 66 | 1 | 1 | male | honorific | 1 |
| Mike Sievert | — | — | — | — | — | — |
| Wei Zhang | 55 | 2 | — | — | — | — |

---

## T — filed 2026-03-23

- source: <https://www.sec.gov/Archives/edgar/data/732717/000119312526119888/d919223ddef14a.htm>
- accession: `0001193125-26-119888`
- carve payload: 40,694 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | AT&T Inc. |
| `fiscal_year_extract` | 2025 |
| `board_size` | 11 |
| `n_directors` | 11 |
| `pct_female_directors` | 0.364 |
| `pct_gender_stated` | 0 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | — |
| `avg_board_tenure` | — |
| `pct_independent_directors` | 0.818 |
| `ceo_name_proxy` | John T. Stankey |
| `ceo_age` | — |
| `ceo_since_year` | — |
| `ceo_salary` | 2,400,000 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 19,500,012 |
| `ceo_option_awards` | 0 |
| `ceo_non_equity_incentive` | 6,272,000 |
| `ceo_all_other_comp` | 237,937 |
| `ceo_total_comp` | 29,906,872 |
| `ceo_equity_pay_pct` | 0.652 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 80,774,129 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.907 |
| `ceo_pay_ratio` | 215 |
| `median_employee_pay` | 139,026 |
| `independent_chair` | — |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | 1 |
| `auditor_name` | Ernst & Young LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 34,200,000 |
| `audit_fees_audit` | 34,200,000 |
| `audit_fees_audit_related` | 1,300,000 |
| `audit_fees_tax` | 3,400,000 |
| `audit_fees_other` | 0 |
| `auditor_fees_prior` | 40,000,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| John T. Stankey | CEO | 2025 | 2,400,000 | 0 | 19,500,012 | 0 | 6,272,000 | 1,496,923 | 237,937 | 29,906,872 | 1 |
| John T. Stankey | CEO | 2024 | 2,400,000 | 0 | 16,499,998 | 0 | 5,992,000 | 1,205,340 | 313,507 | 26,410,845 | 1 |
| John T. Stankey | CEO | 2023 | 2,400,000 | 0 | 16,500,000 | 0 | 6,440,000 | 750,966 | 359,191 | 26,450,157 | 1 |
| Pascal Desroches | Sr. Exec. Vice Pres. & CFO | 2025 | 1,250,000 | 0 | 8,500,005 | 0 | 3,080,000 | 35,462 | 814,684 | 13,680,151 | 1 |
| Pascal Desroches | Sr. Exec. Vice Pres. & CFO | 2024 | 1,250,000 | 0 | 10,499,992 | 0 | 2,942,500 | 0 | 784,627 | 15,477,119 | 1 |
| Pascal Desroches | Sr. Exec. Vice Pres. & CFO | 2023 | 1,250,000 | 0 | 7,500,000 | 0 | 3,162,500 | 34,220 | 713,321 | 12,660,041 | 1 |
| Lori Lee | GMO & Sr. Exec. Vice Pres. - International | 2025 | 875,000 | 0 | 5,599,999 | 0 | 1,680,000 | 436,916 | 285,429 | 8,877,344 | 1 |
| Lori Lee | GMO & Sr. Exec. Vice Pres. - International | 2024 | 750,000 | 0 | 7,124,998 | 0 | 1,444,500 | 322,312 | 248,093 | 9,889,903 | 1 |
| Lori Lee | GMO & Sr. Exec. Vice Pres. - International | 2023 | 750,000 | 250,000 | 5,125,020 | 0 | 1,552,500 | 0 | 159,739 | 7,837,259 | 1 |
| David R. McAtee II | Sr. Exec. Vice Pres. & General Counsel | 2025 | 1,300,000 | 300,000 | 7,000,004 | 0 | 3,080,000 | 433,980 | 254,039 | 12,368,023 | 1 |
| David R. McAtee II | Sr. Exec. Vice Pres. & General Counsel | 2024 | 1,300,000 | 0 | 9,000,006 | 0 | 2,889,000 | 197,076 | 251,189 | 13,637,271 | 1 |
| David R. McAtee II | Sr. Exec. Vice Pres. & General Counsel | 2023 | 1,300,000 | 0 | 7,000,000 | 0 | 3,105,000 | 315,467 | 671,096 | 12,391,563 | 1 |
| Jeffery S. McElfresh | Chief Operating Officer | 2025 | 1,250,000 | 0 | 11,000,018 | 0 | 3,080,000 | 329,647 | 282,074 | 15,941,739 | 1 |
| Jeffery S. McElfresh | Chief Operating Officer | 2024 | 1,250,000 | 0 | 10,999,999 | 0 | 2,942,500 | 262,347 | 165,759 | 15,620,605 | 1 |
| Jeffery S. McElfresh | Chief Operating Officer | 2023 | 1,208,333 | 0 | 9,000,000 | 0 | 3,162,500 | 239,520 | 219,303 | 13,829,656 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| SCOTT T. FORD | — | 140,000 | 220,000 | — | — | — | 250,000 | 610,000 | 1 |
| KELLY J. GRIER | — | 105,000 | 154,301 | — | — | — | 11,972 | 271,273 | 1 |
| GLENN H. HUTCHINS | — | 0 | 0 | — | — | — | 262,698 | 262,698 | 1 |
| WILLIAM E. KENNARD | — | 200,000 | 220,000 | — | — | — | 11,312 | 431,312 | 1 |
| STEPHEN J. LUCZO | — | 171,250 | 220,000 | — | — | — | 0 | 391,250 | 1 |
| MARISSA A. MAYER | — | 140,000 | 220,000 | — | — | — | 10,429 | 370,429 | 1 |
| MICHAEL B. MCCALLISTER | — | 140,000 | 220,000 | — | — | — | 0 | 360,000 | 1 |
| BETH E. MOONEY | — | 170,000 | 220,000 | — | — | — | 10,174 | 400,174 | 1 |
| MATTHEW K. ROSE | — | 167,083 | 220,000 | — | — | — | 11,557 | 398,640 | 1 |
| CYNTHIA B. TAYLOR | — | 180,000 | 220,000 | — | — | — | 0 | 400,000 | 1 |
| LUIS A. UBIÑAS | — | 140,000 | 220,000 | — | — | — | 0 | 360,000 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| BLACKROCK, INC. 50 Hudson Yards, New York, NY 10001 | 5pct_holder | 533,538,337 | 0.075 |
| THE VANGUARD GROUP 100 Vanguard Blvd., Malvern, PA 19355 | 5pct_holder | 622,382,246 | 0.087 |
| KELLY J. GRIER | director_officer | 723 | — |
| WILLIAM E. KENNARD | director_officer | 0 | — |
| STEPHEN J. LUCZO | director_officer | 562,500 | — |
| MARISSA A. MAYER | director_officer | 0 | — |
| MICHAEL B. MCCALLISTER | director_officer | 69,076 | — |
| BETH E. MOONEY | director_officer | 28,700 | — |
| MATTHEW K. ROSE | director_officer | 98,100 | — |
| CYNTHIA B. TAYLOR | director_officer | 5,718 | — |
| LUIS A. UBIÑAS | director_officer | 0 | — |
| JOHN T. STANKEY | director_officer | 1,305,185 | — |
| PASCAL DESROCHES | director_officer | 952,372 | — |
| LORI LEE | director_officer | 561,552 | — |
| DAVID R. MCATEE II | director_officer | 936,291 | — |
| JEFFERY S. MCELFRESH | director_officer | 674,503 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| SCOTT T. FORD | — | — | — | male | name | — |
| KELLY J. GRIER | — | — | — | female | name | — |
| GLENN H. HUTCHINS | — | — | — | male | name | — |
| WILLIAM E. KENNARD | — | — | — | male | name | — |
| STEPHEN J. LUCZO | — | — | — | male | name | — |
| MARISSA A. MAYER | — | — | — | female | name | — |
| MICHAEL B. MCCALLISTER | — | — | — | male | name | — |
| BETH E. MOONEY | — | — | — | female | name | — |
| MATTHEW K. ROSE | — | — | — | male | name | — |
| CYNTHIA B. TAYLOR | — | — | — | female | name | — |
| LUIS A. UBIÑAS | — | — | — | male | name | — |

---

## TDG — filed 2026-01-23

- source: <https://www.sec.gov/Archives/edgar/data/1260221/000126022126000009/tdg-20260122.htm>
- accession: `0001260221-26-000009`
- carve payload: 40,964 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | TransDigm Group Incorporated |
| `fiscal_year_extract` | 2025 |
| `board_size` | 10 |
| `n_directors` | 10 |
| `pct_female_directors` | 0.2 |
| `pct_gender_stated` | 1 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 62 |
| `avg_board_tenure` | 12.5 |
| `pct_independent_directors` | 0.8 |
| `ceo_name_proxy` | Kevin M. Stein |
| `ceo_age` | — |
| `ceo_since_year` | — |
| `ceo_salary` | 1,462,650 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | — |
| `ceo_option_awards` | 21,119,926 |
| `ceo_non_equity_incentive` | 2,585,054 |
| `ceo_all_other_comp` | 20,800 |
| `ceo_total_comp` | 25,188,430 |
| `ceo_equity_pay_pct` | 0.838 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 128,257,454 |
| `insider_ownership_pct` | 0.032 |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 4 |
| `say_on_pay_support_pct` | 0.944 |
| `ceo_pay_ratio` | 369 |
| `median_employee_pay` | 68,247 |
| `independent_chair` | 0 |
| `lead_independent_director` | 1 |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | Ernst & Young LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 9,350,000 |
| `audit_fees_audit` | 7,795,000 |
| `audit_fees_audit_related` | 105,000 |
| `audit_fees_tax` | 1,440,000 |
| `audit_fees_other` | 10,000 |
| `auditor_fees_prior` | 10,274,000 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kevin M. Stein | President, Chief Executive Officer, and Director | 2025 | 1,462,650 | 0 | — | 21,119,926 | 2,585,054 | — | 20,800 | 25,188,430 | 1 |
| Kevin M. Stein | President, Chief Executive Officer, and Director | 2024 | 1,452,500 | 0 | — | 15,822,774 | 2,511,495 | — | 1,645,600 | 21,432,369 | 1 |
| Kevin M. Stein | President, Chief Executive Officer, and Director | 2023 | 1,372,500 | 0 | — | 20,179,574 | 2,275,000 | — | 18,300 | 23,845,374 | 1 |
| Sarah L. Wynne | Chief Financial Officer | 2025 | 721,375 | 0 | — | 4,452,015 | 849,961 | — | 9,827,955 | 15,851,306 | 1 |
| Sarah L. Wynne | Chief Financial Officer | 2024 | 712,500 | 0 | — | 18,131,840 | 825,775 | — | 2,014,405 | 21,684,520 | 1 |
| Sarah L. Wynne | Chief Financial Officer | 2023 | 570,208 | 81,152 | — | 1,578,426 | 541,017 | — | 332,675 | 3,103,479 | 1 |
| Michael J. Lisman | Co-Chief Operating Officer | 2025 | 768,140 | 0 | — | 5,347,848 | 905,062 | — | 26,856,822 | 33,877,872 | 1 |
| Michael J. Lisman | Co-Chief Operating Officer | 2024 | 762,750 | 0 | — | 17,126,019 | 879,308 | — | 6,048,850 | 24,816,927 | 1 |
| Michael J. Lisman | Co-Chief Operating Officer | 2023 | 720,000 | 0 | — | 9,336,999 | 764,400 | — | 2,176,190 | 12,997,589 | 1 |
| Joel B. Reiss | Co-Chief Operating Officer | 2025 | 768,140 | 0 | — | 8,442,541 | 905,062 | — | 29,380,469 | 39,496,212 | 1 |
| Joel B. Reiss | Co-Chief Operating Officer | 2024 | 747,750 | 0 | — | 12,037,422 | 879,308 | — | 7,760,250 | 21,424,730 | 1 |
| Joel B. Reiss | Co-Chief Operating Officer | 2023 | 568,333 | 79,885 | — | 1,578,426 | 532,567 | — | 837,350 | 3,596,562 | 1 |
| Patrick J. Murphy | Co-Chief Operating Officer | 2025 | 551,400 | 0 | — | 0 | 472,354 | — | 12,819,880 | 13,843,634 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| David A. Barr | 2025 | 1,500 | 93,500 | 244,318 | — | — | 19,500 | 358,818 | 1 |
| Jane M. Cronin | 2025 | 1,289 | 73,711 | 244,318 | — | — | 0 | 319,318 | 1 |
| Michael Graff | 2025 | 1,289 | 73,711 | 244,318 | — | — | 19,500 | 338,818 | 1 |
| Sean P. Hennessy | 2025 | 1,500 | 93,500 | 244,318 | — | — | 19,500 | 358,818 | 1 |
| W. Nicholas Howley | 2025 | 0 | 0 | 0 | — | — | 0 | 0 | 1 |
| Gary E. McCullough | 2025 | 85,000 | 0 | 244,318 | — | — | 19,500 | 348,818 | 1 |
| Michele L. Santana | 2025 | 1,289 | 73,711 | 244,318 | — | — | 19,500 | 338,818 | 1 |
| Robert J. Small | 2025 | 1,769 | 113,231 | 244,318 | — | — | 19,500 | 378,818 | 1 |
| Jorge L. Valladares III | 2025 | 1,584 | 59,542 | 244,318 | — | — | 0 | 305,444 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| Capital International Investors | 5pct_holder | 6,489,193 | 0.115 |
| The Vanguard Group, Inc. | 5pct_holder | 6,790,800 | 0.12 |
| Capital World Investors | 5pct_holder | 3,938,687 | 0.07 |
| BlackRock, Inc. | 5pct_holder | 2,918,533 | 0.052 |
| David A. Barr | director_officer | 41,348 | — |
| Jane M. Cronin | director_officer | 2,180 | — |
| Michael Graff | director_officer | 18,259 | — |
| Sean P. Hennessy | director_officer | 38,259 | — |
| W. Nicholas Howley | director_officer | 630,539 | 0.0111 |
| Gary E. McCullough | director_officer | 10,125 | — |
| Peter J. Palmer | director_officer | 40,055 | — |
| Michele L. Santana | director_officer | 7,551 | — |
| Robert J. Small | director_officer | 504,661 | — |
| Kevin M. Stein | director_officer | 101,725 | — |
| Sarah L. Wynne | director_officer | 56,490 | — |
| Michael J. Lisman | director_officer | 163,163 | — |
| Joel B. Reiss | director_officer | 148,030 | — |
| Patrick J. Murphy | director_officer | 72,635 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| David A. Barr | 62 | 9 | 1 | male | honorific | — |
| Jane M. Cronin | 58 | 5 | 1 | female | honorific | 1 |
| Michael Graff | 74 | 23 | 1 | male | honorific | — |
| Sean P. Hennessy | 68 | 20 | 1 | male | honorific | 1 |
| W. Nicholas Howley | 73 | 33 | 0 | male | honorific | 1 |
| Michael J. Lisman | 43 | 1 | 0 | male | honorific | 0 |
| Gary E. McCullough | 67 | 9 | 1 | male | honorific | 1 |
| Peter J. Palmer | 61 | 1 | 1 | male | honorific | 0 |
| Michele L. Santana | 55 | 8 | 1 | female | honorific | — |
| Robert J. Small | 59 | 16 | 1 | male | honorific | — |

---

## TSCO — filed 2026-03-26

- source: <https://www.sec.gov/Archives/edgar/data/916365/000119312526126620/d16327ddef14a.htm>
- accession: `0001193125-26-126620`
- carve payload: 40,824 chars

### Filing-level scalars

| field | value |
|---|---|
| `company_name` | Tractor Supply Company |
| `fiscal_year_extract` | 2025 |
| `board_size` | 10 |
| `n_directors` | 9 |
| `pct_female_directors` | 0.5 |
| `pct_gender_stated` | 1 |
| `n_women_directors_vs_inferred` | 0 |
| `avg_director_age` | 59.375 |
| `avg_board_tenure` | 7.375 |
| `pct_independent_directors` | 0.889 |
| `ceo_name_proxy` | Harry A. Lawton III |
| `ceo_age` | — |
| `ceo_since_year` | — |
| `ceo_salary` | 1,344,231 |
| `ceo_bonus` | 0 |
| `ceo_stock_awards` | 26,937,383 |
| `ceo_option_awards` | 2,311,740 |
| `ceo_non_equity_incentive` | 1,635,837 |
| `ceo_all_other_comp` | 48,003 |
| `ceo_total_comp` | 32,277,194 |
| `ceo_equity_pay_pct` | 0.906 |
| `n_neos` | 5 |
| `sct_years` | 3 |
| `total_neo_comp` | 42,947,312 |
| `insider_ownership_pct` | — |
| `ceo_ownership_pct` | — |
| `n_five_percent_holders` | 2 |
| `say_on_pay_support_pct` | 0.93 |
| `ceo_pay_ratio` | 1,324 |
| `median_employee_pay` | 24,376 |
| `independent_chair` | 1 |
| `lead_independent_director` | — |
| `classified_board` | 0 |
| `dual_class_shares` | 0 |
| `poison_pill` | — |
| `majority_voting` | — |
| `auditor_name` | Ernst & Young LLP |
| `auditor_since_year` | — |
| `auditor_fees` | 1,564,012 |
| `audit_fees_audit` | 1,562,012 |
| `audit_fees_audit_related` | 0 |
| `audit_fees_tax` | 0 |
| `audit_fees_other` | 2,000 |
| `auditor_fees_prior` | 1,400,720 |

### Summary Compensation Table (`def14a_executive_comp`)

| name | title | fiscal_year | salary | bonus | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Harry A. Lawton III | President, Chief Executive Officer, and Director | 2025 | 1,344,231 | 0 | 26,937,383 | 2,311,740 | 1,635,837 | 0 | 48,003 | 32,277,194 | 1 |
| Harry A. Lawton III | President, Chief Executive Officer, and Director | 2024 | 1,294,231 | 0 | 6,562,207 | 2,187,455 | 1,689,691 | 0 | 43,879 | 11,773,463 | 0 |
| Harry A. Lawton III | President, Chief Executive Officer, and Director | 2023 | 1,298,077 | 0 | 6,374,641 | 2,124,967 | 1,535,547 | 0 | 41,434 | 11,374,666 | 1 |
| Kurt D. Barton | Executive Vice President – Chief Financial Officer and Treasurer | 2025 | 788,462 | 0 | 1,424,956 | 474,836 | 549,318 | 0 | 45,504 | 3,283,076 | 1 |
| Kurt D. Barton | Executive Vice President – Chief Financial Officer and Treasurer | 2024 | 697,692 | 0 | 1,199,619 | 399,974 | 454,917 | 0 | 44,337 | 2,796,539 | 1 |
| Kurt D. Barton | Executive Vice President – Chief Financial Officer and Treasurer | 2023 | 703,846 | 0 | 1,124,795 | 374,962 | 417,669 | 0 | 41,820 | 2,663,092 | 1 |
| Robert D. Mills | Executive Vice President – Chief Technology, Digital Commerce, and Strategy Officer | 2025 | 741,346 | 0 | 1,049,948 | 349,880 | 454,399 | 0 | 45,194 | 2,640,767 | 1 |
| Robert D. Mills | Executive Vice President – Chief Technology, Digital Commerce, and Strategy Officer | 2024 | 670,846 | 0 | 981,628 | 324,965 | 438,670 | 0 | 43,308 | 2,459,917 | 0 |
| Robert D. Mills | Executive Vice President – Chief Technology, Digital Commerce, and Strategy Officer | 2023 | 661,385 | 0 | 1,017,597 | 274,948 | 392,486 | 0 | 41,303 | 2,387,719 | 1 |
| J. Seth Estep | Executive Vice President – Chief Merchandising Officer | 2025 | 729,615 | 0 | 974,979 | 324,891 | 448,341 | 0 | 42,610 | 2,520,436 | 1 |
| J. Seth Estep | Executive Vice President – Chief Merchandising Officer | 2024 | 647,462 | 0 | 749,792 | 249,955 | 422,423 | 0 | 44,889 | 2,114,521 | 1 |
| J. Seth Estep | Executive Vice President – Chief Merchandising Officer | 2023 | 650,077 | 0 | 599,956 | 199,968 | 385,729 | 0 | 41,890 | 1,877,620 | 1 |
| John P. Ordus | Executive Vice President – Chief Stores Officer | 2025 | 672,115 | 0 | 824,933 | 274,901 | 408,959 | 0 | 44,931 | 2,225,839 | 1 |
| John P. Ordus | Executive Vice President – Chief Stores Officer | 2024 | 647,462 | 0 | 674,745 | 224,971 | 422,423 | 0 | 43,545 | 2,013,146 | 1 |
| John P. Ordus | Executive Vice President – Chief Stores Officer | 2023 | 650,077 | 0 | 599,956 | 199,968 | 385,729 | 0 | 40,930 | 1,876,660 | 1 |

### Director compensation (`def14a_director_comp`)

| name | fiscal_year | fees_earned | stock_awards | option_awards | non_equity_incentive | pension_change | other_compensation | total | reconciles |
|---|---|---|---|---|---|---|---|---|---|
| Joy Brown | — | 122,000 | 164,975 | — | — | — | 0 | 286,975 | 1 |
| Rick Cardenas | — | 142,000 | 164,975 | — | — | — | 0 | 306,975 | 1 |
| Meg Ham | — | 105,000 | 164,975 | — | — | — | 0 | 269,975 | 1 |
| André Hawaux | — | 122,000 | 164,975 | — | — | — | 0 | 286,975 | 1 |
| Denise L. Jackson | — | 125,000 | 164,975 | — | — | — | 0 | 289,975 | 1 |
| Ramkumar Krishnan | — | 105,000 | 164,975 | — | — | — | 0 | 269,975 | 1 |
| Edna K. Morris | — | 195,000 | 264,997 | — | — | — | 0 | 459,997 | 1 |
| Mark J. Weikel | — | 137,000 | 164,975 | — | — | — | 0 | 301,975 | 1 |

### Beneficial ownership (`def14a_ownership`)

| holder_name | holder_type | shares | percent_of_class |
|---|---|---|---|
| The Vanguard Group | 5pct_holder | 64,022,615 | 0.122 |
| BlackRock, Inc. | 5pct_holder | 48,193,100 | 0.092 |
| Kurt D. Barton | director_officer | 542,262 | — |
| Joy Brown | director_officer | 15,588 | — |
| Rick Cardenas | director_officer | 35,285 | — |
| J. Seth Estep | director_officer | 126,738 | — |
| Meg Ham | director_officer | 13,013 | — |
| André Hawaux | director_officer | 9,573 | — |
| Denise L. Jackson | director_officer | 31,123 | — |
| Ramkumar Krishnan | director_officer | 61,069 | — |
| Harry A. Lawton III | director_officer | 1,549,684 | — |
| Robert D. Mills | director_officer | 160,570 | — |
| Edna K. Morris | director_officer | 340,561 | — |
| John P. Ordus | director_officer | 204,110 | — |
| Sonia Syngal | director_officer | 0 | — |
| Mark J. Weikel | director_officer | 43,163 | — |

### Directors (`def14a_directors`)

| name | age | tenure_years | is_independent | gender | gender_basis | other_public_company_boards |
|---|---|---|---|---|---|---|
| Joy Brown | 47 | 5 | 1 | female | honorific | 1 |
| Ricardo Cardenas | 58 | 7 | 1 | male | honorific | 1 |
| Meg Ham | 59 | 3 | 1 | female | honorific | 0 |
| André Hawaux | 65 | 4 | 1 | male | honorific | 2 |
| Denise L. Jackson | 61 | 8 | 1 | female | honorific | 0 |
| Ramkumar Krishnan | 55 | 10 | 1 | male | honorific | 0 |
| Edna K. Morris | 74 | 22 | 1 | female | honorific | 0 |
| Sonia Syngal | 56 | 0 | 1 | female | honorific | 1 |
| Mark J. Weikel | — | — | — | — | — | — |

---
