# Value-level checks on the DEF 14A sample

Fill rates cannot tell a right number from a plausible wrong one. Every check here is an **identity the filing itself must satisfy**, so a failure localises the defect.

**1133 pass / 11 fail / 400 not evaluable** (99.0% of evaluable).

| check | what | pass | fail | n/a |
|---|---|---|---|---|
| C1 | SCT components sum to `total` (+/- $10) | 304 | 4 | 0 |
| C1b | director-comp components sum to `total` (+/- $10) | 225 | 0 | 7 |
| C2 | audit fee categories sum to `auditor_fees` (+/- $10) | 19 | 2 | 1 |
| C3 | `ceo_total_comp` == the CEO's own SCT row total | 22 | 0 | 0 |
| C4 | ceo_pay_ratio == ceo_total_comp / median_employee_pay | 19 | 1 | 2 |
| C5 | board_size == director rows (+/- 1) | 19 | 3 | 0 |
| C6 | director age 25-95, tenure 0-60 and <= age-20 | 188 | 0 | 55 |
| C7 | percent_of_class in (0, 1] (a fraction, not a percentage) | 84 | 0 | 320 |
| C8 | say_on_pay_support_pct in (0, 1] | 16 | 0 | 6 |
| C9 | every director with a gender has a gender_basis | 234 | 0 | 9 |
| C10 | no duplicate primary keys in any child table | 3 | 1 | 0 |

A check whose two legs are both absent is counted as **not evaluable**, never as a pass — an unevaluated check inflating a pass rate is the failure mode this file exists to avoid.

### C1 — SCT components sum to `total` (+/- $10)

- **304 pass / 4 fail** (98.7%); 0 could not be evaluated

- `NKE Matthew Friend 2024`: components sum off by **975,000** (total 10,390,510, 7/7 components present)
- `SBUX Sara Kelly 2024`: components sum off by **141,986** (total 3,562,568, 5/7 components present)
- `TSCO Harry A. Lawton III 2024`: components sum off by **4,000** (total 11,773,463, 7/7 components present)
- `TSCO Robert D. Mills 2024`: components sum off by **500** (total 2,459,917, 7/7 components present)

### C1b — director-comp components sum to `total` (+/- $10)

- **225 pass / 0 fail** (100.0%); 7 could not be evaluated

### C2 — audit fee categories sum to `auditor_fees` (+/- $10)

- **19 pass / 2 fail** (90.5%); 1 could not be evaluated

- `BA`: components sum off by **4,500,000** (total 39,100,000, 4/4 components present)
- `T`: components sum off by **4,700,000** (total 34,200,000, 4/4 components present)

### C3 — `ceo_total_comp` == the CEO's own SCT row total

- **22 pass / 0 fail** (100.0%); 0 could not be evaluated

### C4 — ceo_pay_ratio == ceo_total_comp / median_employee_pay

- **19 pass / 1 fail** (95.0%); 2 could not be evaluated

- `CAT`: disclosed ratio 196 vs 17,008,077/89,253 = 191

### C5 — board_size == director rows (+/- 1)

- **19 pass / 3 fail** (86.4%); 0 could not be evaluated

- `AEE`: board_size 12 but 16 director rows
- `PFE`: board_size 12 but 14 director rows
- `PG`: board_size 12 but 15 director rows

### C6 — director age 25-95, tenure 0-60 and <= age-20

- **188 pass / 0 fail** (100.0%); 55 could not be evaluated

### C7 — percent_of_class in (0, 1] (a fraction, not a percentage)

- **84 pass / 0 fail** (100.0%); 320 could not be evaluated

- accession `0000320187-26-000089`: holder percents sum to 2.44 (> 1.0)

### C8 — say_on_pay_support_pct in (0, 1]

- **16 pass / 0 fail** (100.0%); 6 could not be evaluated

### C9 — every director with a gender has a gender_basis

- **234 pass / 0 fail** (100.0%); 9 could not be evaluated

### C10 — no duplicate primary keys in any child table

- **3 pass / 1 fail** (75.0%); 0 could not be evaluated

- `def14a_ownership`: 5 duplicate rows on ['ticker', 'accession_number', 'holder_name', 'holder_type']
