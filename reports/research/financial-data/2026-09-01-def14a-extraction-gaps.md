# Research: DEF 14A / DEF 14C data extraction — gaps in edgartools and in our codebase

**Date**: 2026-09-01 12:05 EDT
**Research Phase**: 1 of 3 (FIC Workflow)
**Next Phase**: Planning (`/plan`)
**Spec**: `specs/2026-09-01/investigate-def-14a.md`
**Library under audit**: edgartools **5.51.0** (the prior repo audit was against 5.44.1)

> **Planning a fix? Read [Consolidated conclusions](#consolidated-conclusions--read-this-first-when-planning) at the END of this document first.**
> §1–§15 are the original audit. Three follow-ups extend it and, in one case, **correct it**:
> 1. **Bulk & third-party vote sources** — EFTS/FSDS/N-PX/vendors.
> 2. **`def14a_llm` correctness + 8-K 5.07 parse feasibility** — ⚠ contains a correction to the
>    Test Coverage section: the `SAY_ON_PAY_MIN_SUPPORT = 0.50` guard deletes 61 **correct** rows.
> 3. **How to fix the section carve** — a measured head-to-head of three carving strategies.

## Research Question

Review the DEF 14A / DEF 14C extraction in the codebase, identify every data gap, and attribute each
gap either to edgartools or to our own code. Specific questions on `sec_def14a` fill rate,
director/executive comp components, the ticker `A` billion-dollar salary, `Former` names, missing
titles, missing `board_recommendation`, vote counts, format drift over time, the `edgar.core` SGML
warnings, and whether these data are trustworthy enough to build alphas on.

## Summary

The five `sec_def14a*` tables are sparse for **three independent reasons that must not be
conflated**:

1. **The backfill is 5% complete.** `sec_def14a` holds 26 of 500 `sp500_tickers` — an alphabetical
   run A…AME, still actively inserting during measurement. Nothing about parsing explains this.
2. **A regulatory cliff at the 2023 filing season.** Eight columns of `sec_def14a` come *exclusively*
   from the ECD inline-XBRL block, which **did not exist in any DEF 14A before 2023**. Measured
   universe-wide: 0 of 1,641 S&P 500 proxies 2019–2022 carry inline XBRL; the SEC's monthly XBRL
   RSS feed goes 0 (2022-04) → 782 (2023-03). Those columns are 0.00% filled for every year
   1995–2022 and 81–100% filled 2023–2026. This is correct behaviour, not a defect.
3. **edgartools' HTML parser is silently wrong.** All six defects recorded in the 5.44.1 audit are
   still present in 5.51.0, plus five newly-measured ones — several of which **fabricate values**
   rather than omitting them.

The most important single finding: **nothing ever raises.** All 90 `filing.obj()` calls across the
15-ticker × 6-year probe returned a `ProxyStatement`; every failure is a silent empty table or a
silently wrong number. Only **11 of 90 cells (12%)** extract fully.

Against the trust question: the ECD XBRL block is sound if dimension-filtered; the HTML-parsed
child tables are not currently at alpha-grade. Detail in **Trustworthiness** below.

---

## Detailed Findings

### 1. Coverage — the dominant term, and it is ours

Live DB snapshot 2026-09-01 14:55 UTC, taken as TEMP-table copies in one psql session.

| table | rows | tickers | accessions | date range |
|---|---|---|---|---|
| `sec_def14a` | 684 | **26** | 684 | 1995-09-13 → 2026-05-06 |
| `sec_def14a_director_comp` | 3,119 | 25 | 312 | 2006-02-21 → 2026-04-24 |
| `sec_def14a_executive_comp` | 2,378 | 25 | 343 | 2001-02-16 → 2026-04-24 |
| `sec_def14a_ownership` | 3,477 | 23 | 294 | 2000-03-03 → 2026-04-24 |
| `sec_def14a_votes` | 1,851 | 26 | 435 | 1995-09-21 → 2026-04-24 |
| `def14a_llm` | 8,667 | **497** | 8,667 | 1995-09-13 → 2026-08-17 |
| `sp500_tickers` | 500 | — | — | — |

- **26 / 500 = 5.2%**, alphabetical A…AME. Row counts moved during measurement (525→684 over 8
  minutes), so a backfill is running.
- Per-ticker filing counts are 6 (ABNB) → 35 (ADSK); only 5 tickers have interior year gaps. Within
  the 26 covered tickers the *history* is essentially complete.
- Overlap with `def14a_llm` on `(ticker, accession)`: both **527**, llm-only **8,140**,
  edgar-only **157**. **471 of 497 llm tickers are absent from `sec_def14a` entirely.** No key-format
  mismatch (joining on accession alone gives the identical 527).
- `docs/database.md:125-127` already records this: *"`sec_def14a` covers only 23 of 500 tickers …
  Any `f_ceo_*` / governance feature built off `sec_def14a` will be ~95% NaN."*

**Nothing downstream reads these five tables.** Exhaustive grep over `src/data_aggregate/`,
`src/data_peers/`, `src/modelling/`, `src/validate/`, `app/`, `backtest/`, `stock_pick_strat/`
returns zero consumers outside the fetcher, its tests, `schema.py` and docs. The cube's `governance:`
feature group (`configs/build_cube.yml:197-200`) is fed by **`def14a_llm`** via
`src/data_aggregate/transformers/step_cube_extras.py:155` → `governance_features.py`.

### 2. The 2023 cliff — why `sec_def14a` looks empty

`sec_def14a` non-null rate: 100% on the keys, `form`, `has_xbrl`, `has_individual_executive_data`
and all 7 `n_*` counters; 97.37% `period_of_report`; **everything else ≤ 29.4%**.

`has_xbrl` × `peo_total_comp` is phi = **−0.982**: all 588 rows with `has_xbrl=0` have NULL
`peo_total_comp`; of the 96 rows with `has_xbrl=1`, 93 are non-null. `company_name` NULL ⇒
`peo_name` NULL with **zero exceptions** (phi 0.926). The set where `has_xbrl=1` is *identical* to
the set where `company_name` is non-null.

`company_name`, `peo_name`, `peo_total_comp`, `peo_actually_paid_comp`, `neo_avg_total_comp`,
`total_shareholder_return`, `net_income`, `has_xbrl` are **exactly 0 for every year 1995–2022** and
populate 2023–2026 (2023: 21/26; 2024: 26/26; 2025: 26/26). `ceo_pay_ratio` starts 2018 — matching
Item 402(u), FY2017.

Live-probe confirmation: 0 `ecd:` tags in **24 of 24** filings dated 2017–2022 for AAPL/JPM/CAT/KO;
first appearances JPM 2023-04-04 (28 tags), KO 2023-03-10 (27), CAT 2023-05-01 (24). In the main
grid, 15/15 of the 2026 cells carry 25–46 distinct `ecd:` tags; **0/15 at 2021 and earlier**.

Apple is the instructive exception: its **2023-01-12** proxy has 0 ecd tags because its FY2022 ended
2022-09-24, before the 2022-12-16 threshold. Its first tagged proxy is 2024-01-11.

**2023 S&P 500 ECD coverage = 435/468 (93%)**, and the 30 gaps decompose cleanly: **27 have
non-December fiscal year ends**; exactly 3 (APP, BSX, GDDY) published a PVP table with no XBRL.
2025: 489/493 (99.2%).

### 3. `company_name` — a one-line gap that is entirely ours

`edgar/proxy/core.py:193-196`:
```python
@property
def company_name(self) -> Optional[str]:
    return self._get_concept_value('dei:EntityRegistrantName')
```
XBRL-only. **No fallback to `self._filing.company`**, which is always populated from the filing
index. DEF 14A has no mandatory cover-page iXBRL requirement, so this is None for every pre-2023
filing and for any filer omitting the dei cover tags. `__str__` substitutes the literal
`"Unknown Company"` (`core.py:806-809`).

`fetch_def14a_edgar.py` already sources `period_of_report` from `filing.period_of_report` rather
than the library property, for exactly this class of reason (see its inline note: `fiscal_year_end`
"never once resolved — 0 of 329 stored rows had it"). Measured here: `ProxyStatement.fiscal_year_end`
(`dei:DocumentPeriodEndDate`) is **None on 4/4 live large-cap proxies**.

### 4. `peo_name` / PvP — the ECD block is read without dimension filtering

`_get_concept_value` (`core.py:105-131`) and `_get_concept_series` (`core.py:133-151`) filter on
`concept ==` only. Neither filters `dim_ecd_IndividualAxis` or `dim_ecd_ExecutiveCategoryAxis`;
they sort by `period_end` and take `.iloc[0]` / `.iloc[-1]`, with
`drop_duplicates(keep='first')` on a stable sort — so **document order decides the winner**.

Measured consequences on live filings:

| filing | what happened |
|---|---|
| **BA 2026** | FY2024 has two PEOs, no undimensioned total. Ortberg 18,388,629 kept; **Calhoun 15,050,812 dropped**. CAP: Ortberg **+19,904,513** kept, Calhoun **−23,875,735** dropped. |
| **NKE 2026** | FY2025 keeps Hill 26,018,068, drops Donahoe 28,442,712 (CAP **−10,924,243**). |
| **SBUX** | Filer tags a full individual×year matrix with 0.0 in non-applicable cells. `peo_total_comp` = **0.0** and `peo_actually_paid_comp` = **0.0** for FY2023, FY2024 *and* FY2025, while `peo_name` = `'Brian Niccol'`. |

`ecd:PeoName` is **the tag filers use for every named executive**, discriminated only by the axis the
library ignores. AAPL's instance carries **26 `ecd:PeoName` facts, only 5 of them `ecd:PeoMember`**;
the rest are NEOs. AAPL happens to return `"Mr. Cook"` (display text, not a full legal name); a filer
emitting NEO names first for the latest year returns an NEO.

Note our repair layer's `peo_total_comp == 0 → NaN` rule (`def14a_validate.py`) already neutralises
the SBUX shape, but as a NULL rather than the correct value.

**Units are not normalised anywhere.** `edgar/xbrl/parsers/instance.py:430-436` is
`float(value)` verbatim; grep for `sign`/`scale` in the instance parser returns nothing. Measured
`net_income`: AAPL FY2025 = `112010000000.0`; **SBUX FY2025 = `1856.4`** (raw `value='1856.4'`,
`decimals='1'`, `unit_ref='usd'` — tagged in $ millions). Our
`DEF14A_NET_INCOME_MIN_PLAUSIBLE` guard NULLs this rather than rescaling, which the module docstring
justifies (nothing in the row disambiguates millions from billions).

### 5. Negative `peo_actually_paid_comp` — legitimate, not a defect

11 of 93 non-null rows (11.83%), **all with positive `peo_total_comp`**; zero negative
`peo_total_comp`. Distribution: min −585,324,767 / p25 7,505,424 / p50 21,518,114 / max 303,463,125.

| ticker | yr | accession | peo_name | total | actually paid |
|---|---|---|---|---|---|
| ABNB | 2023 | 0001193125-23-109099 | Brian Chesky | 311,233 | **−585,324,767** |
| AMD | 2023 | 0001193125-23-088096 | Lisa Su | 30,219,921 | −160,142,075 |
| ABNB | 2026 | 0001193125-26-175062 | Brian Chesky | 242,122 | −89,229,878 |
| ALGN | 2023 | 0001097149-23-000029 | Joseph Hogan | 18,684,044 | −61,348,849 |

Compensation Actually Paid subtracts prior-year unvested-award fair value, so a share-price decline
makes it negative. Measured base rates: **28.5% of 2023** and **33.7% of 2025** S&P 500 proxies
report ≥1 negative `ecd:PeoActuallyPaidCompAmt`. Intel FY2024 reconciles exactly:
27,429,900 − 24,625,700 + 0 + 0 − 1,775,113 − 83,245,704 = **−82,216,617**. Others: TSLA FY2022
−9,703,000,000; MRNA −306,219,630; PYPL −87,002,457 against a +21,957,922 SCT total.

There is no sign flip in edgartools: `float()` → `Decimal(str(value))`, no `abs()`, no `-1 *`.
Signs arrive correct because the parser reads the SEC-extracted `*_htm.xml` instance (where the sign
is already applied), not the inline document.

### 6. Ticker `A` — `<br>`-separated multi-year cells concatenated

**The spec attributes this to director comp `salary`; both attributions are wrong.**
`sec_def14a_director_comp` has no `salary` column (it has `fees_earned`), and ticker A's director
rows are entirely sane (max fees 270,000 / stock 366,277 / total 507,471 over 156 rows). The
corruption is in **`sec_def14a_executive_comp.salary`**, 9 rows across **two accessions**:

| accession | filed | name | salary | other_comp |
|---|---|---|---|---|
| `0001193125-05-003365` | 2005-01-10 | Edward W. Barnholt | **1.000e19** | 820,080,008,000 |
| `0001193125-04-001673` | 2004-01-07 | Edward W. Barnholt | 1.000e18 | 800,080,006,800 |

Root cause, from the raw source (`A_2005_ddef14a.htm` byte 231,312):
```html
<TD ...><FONT ...>1,000,000</FONT><br><FONT ...>1,000,000</FONT><br><FONT ...>925,000</FONT></TD>
```
One `<TD>` holds three fiscal years stacked on `<br>`. The grid extractor takes `text_content()` per
cell (`html_extractor.py:661-677`), which drops `<br>` **without inserting a separator**, so
FY2004+FY2003+FY2002 concatenate to `1,000,0001,000,000925,000` → comma-stripped →
`10000001000000925000` = 1.0e19. The same cell shape produced `year` = `200420032002` (truncated to
2004) and `other_comp` = `820080008000`. Not a footnote digit, merged cell, or aggregate row.

**This is not confined to old filings.** 109 rows across the DB exceed $1e9 — **all in
`executive_comp`, zero in `director_comp`** (4.58% of exec rows). Worst offenders by ticker:
AMAT 33/109 rows, AIZ 13/78, AIG 12/98, AEP 11/94, ADSK 10/43, A 9/37. The largest single value is
**AMAT / Gary E. Dickerson = 3.527e23** (`0001193125-22-018340`), and AMAT/Dickerson exceeds 1e23 in
**nine consecutive filings, 2017 → 2026**. 32 accessions carry ≥1 such row (full list in the
measurement notes).

Global exec-comp maxima: salary 1.082e20, bonus 1.121e20, stock_awards 3.250e20, non_equity 2.480e20,
pension 3.233e20, total 3.527e23. Only `option_awards` (max 14,774,465) is clean.
`def14a_validate.py` has **no magnitude bound on comp components** — its only numeric guards are the
audit-fee rescale, the net-income floor, and the components-vs-total reconciliation, and the latter
is inert here because `total` is NULL on all nine A rows.

### 7. Director comp components — a parser miss in 4 of 4 cases opened

Non-null: `total` 99.65% · `fees_earned` 80.47% · `stock_awards` 48.96% · `other_compensation`
46.75% · `option_awards` 14.68% · `pension_change` 8.75% · `non_equity_incentive` **0.99%**.
By year, `option_awards` is **0.0% from 2022 onward** and `non_equity_incentive` **0.0% from 2010**.

- `total` NOT NULL but `stock_awards` NULL: **1,582 / 3,108 = 50.9%**.
- **Rows with all six components non-null: 0.** The user's "total should be the sum of all fields"
  test cannot be evaluated as posed anywhere in the table.
- Coalesce-0 residual (`total − Σ present`) is within $1 on 1,596 / 3,108 = 51.4%; p25 = −185,000.
- 100%-affected tickers: AMAT 208/208, A 156/156, ADP 133/133, AIG 61/61.

The live probe opened the source HTML for four cases. **In none was the value absent.**

**CAT 2026** — parser returned `stock_awards=None`, `total=343907`:
```
['Director','Fees Earned or Paid in Cash','Restricted Stock Units (1)','All Other Compensation (2)','Total']
['JAMES C. FISH, JR.', '$','163,874', '$','175,033', '$','5,000', '$','343,907']
```
**PFE 2026** — `stock_awards` null on **13 of 13 rows** while every row carries 205,000 under a header
labelled exactly `Stock Awards ($)`:
```
['Name','Fees Earned or Paid in Cash ($)','Stock Awards ($) (1)','All Other Compensation ($) (2)','Total ($)']
['Ronald E. Blaylock', '155,000', '205,000', '—', '360,000']
```
**GE 2026** — 3 of 4 numeric columns dropped (`fees=None, stock=None, other=None, total=345795`):
```
['NAME OF DIRECTOR','CASH FEES','STOCK AWARDS','ALL OTHER COMP','TOTAL']
['Sébastien Bazin', '$','0', '$','345,795', '$','0', '$','345,795']
```
**GE 2016** is the mirror shape (no standalone `$` cells) — stock/other/total parse, `fees_earned` is
null on 0/8 rows while the source plainly reads `'$110,000'`.

Two mechanisms:
- **(a) A `$` glyph in its own `<td>`** doubles the effective column count, desynchronising
  `col_map` from the data row.
- **(b) An unrecognised header label.** The director-comp synonyms (`html_extractor.py:1102-1111`)
  map `stock_awards` ← `stock\s+award` and `fees_earned` ← `fees?\s+earned|fees?\s+paid|retainer`.
  `Restricted Stock Units` and `Cash Fees` match neither. An unmapped column index never enters
  `col_map` and `_get_dollar_from_row` returns `None` (`html_extractor.py:820-822`) — **silently**.

Also relevant: `_split_header_data` (680-707) scans only `grid[:4]` and treats **exactly one row** as
the header — multi-row headers (`Stock` / `Awards ($)`) are never merged, which drops the column the
same way. And `rowspan` is **not handled at all** in `_extract_table_grid`, shifting every subsequent
row left by one column.

### 8. Executive comp — `Former`, missing titles, missing years

**`Former` (42 rows, 1.77%, 18 distinct values).** The pathology is a **title fragment landing in the
`name` column**: `name='Former'` with `title='CFO'` (ABNB `0001193125-24-102960`), `title='CTO'`
(ABNB `0001193125-26-175062`), `title='Executive Vice'` (ACGL `0001047469-13-003619`). Also
`'Corporate Officer andFormer'`, `'Executive Chairand Former ChiefExecutive Officer'` (ADP),
`'David L. WhiteFormer'`, `'Howard I. Smith Former Vice'` (AIG `0000950117-06-001639`).

Source: `_split_name_title` (`html_extractor.py:567-601`) does `\xa0`→space and
`re.sub(r'\s+',' ')` at 570-571, which **destroys the `\n` separator before** the
`for sep in [',', '\n']` loop at 574. Newline-separated name/title cells therefore never take the
clean path and fall through to a hardcoded keyword scan at 590:
`['Chief ', 'President', 'Senior Vice', 'Executive Vice', 'General Counsel']`, **in that order**.
Because `'Chief '` is tested first, live AAPL output is
`name='Kevan Parekh Senior Vice President'`, `title='CFO'` and
`name='Deirdre O�Brien Senior Vice'`, `title='President, Retail + People'`.
Our `def14a_validate._ORPHAN_MODIFIER_RE` already repairs the `"… Group" + "President"` shape; the
`Former` cases are the same class, differently split.

**Title NULL on 1,299 / 2,378 = 54.6%**, no trend (100% in 2001 → 43.2% in 2026).
**Backfill is mostly infeasible**: of 737 distinct `(ticker, name)` pairs, 488 have ≥1 NULL title but
only **108 (22.1%)** have a non-null title anywhere else. Row-level, **383 of 1,299 (29.5%)** are
backfillable. Title *stability* is not the obstacle — 77.9% of pairs that have any title have exactly
one distinct title, so the edge case the spec worried about (title changes for the same person) is
rare; the obstacle is that 380 pairs have no title anywhere.

**Years per accession** (a Summary Compensation Table normally carries 3): 1 year → **162 accessions
(47.2%)**, 2 → 61, 3 → 120. Rows per accession: median 6, mean 6.93, versus ~15 expected. Distinct
names per accession is correct (median 5) — **the year dimension is what is lost.**

Root cause, measured on AAPL's real SCT (grid has 15 rows / 12 exec-years; the library returned
**6 rows, all `year=2025`**): continuation rows have no name cell and are therefore one column short.
```
header:      [... 'Year' @2, ... 'Salary($)' @4, ...]
Cook 2025:   ['Tim\xa0Cook\nChief Executive Officer','','2025','','3,000,000',...]
    (blank): ['','2024','','3,000,000','','58,088,946',...]
```
For the continuation row `row[2] == ''` → `int('')` raises → the `year_col+1` fallback
(`html_extractor.py:886-896`) reads `'3,000,000'` → `'3000000'[:4]` → **`year = 3000`** → dropped by
the `year > 2100` guard at line 898. No warning.

**Exec ↔ PEO agreement.** Only **53** accessions are joinable to a non-null `peo_total_comp`;
**31 (58.5%)** agree within $1. `peo_name` is found verbatim in `executive_comp.name` for the same
accession in only **17 of 49 (34.7%)** — consistent with §4 (wrong PEO selected) and §8 (name
mangling). Exec ↔ director cross-reference is effectively empty: exactly **1** row shares
`(ticker, accession, name)`, and it disagrees (ADP, Carlos A. Rodriguez, exec 13,301,036 vs director
10,000 — a CEO who also drew a director retainer). **The two tables are disjoint on person by
design**; the spec's "does CEO total comp align with director comp total" has no population to test.

**Encoding:** 168 of 2,378 rows (**7.06%**) carry U+0097 in `name`/`title` — a cp1252 em-dash
mis-decode. Zero such rows in the other three child tables.

**Pre-2007 row explosion:** JPM_2006 returns 15 rows for 5 executives, with title fragments
(`'Chairman of the Board'`) as separate rows and the *year* in the `title` column. Same for MSFT_2006.
`total` is **0.0% populated for 2001–2005** — the pre-Reg-S-K SCT has a different column set entirely.

### 9. Votes — board recommendation is a text regex that never reads a table

`sec_def14a_votes` by type, with `board_recommendation` fill:

| proposal_type | n | % of rows | board_rec non-null |
|---|---|---|---|
| company_proposal | 621 | 33.6% | 50.4% |
| director_election | 362 | 19.6% | 51.1% |
| auditor_ratification | 361 | 19.5% | 74.2% |
| say_on_pay | 243 | 13.1% | **67.1%** |
| shareholder_proposal | 180 | 9.7% | 80.6% |
| equity_plan | 63 | 3.4% | 52.4% |
| say_on_pay_frequency | 21 | 1.1% | 57.1% |

Overall 60.45% non-null. Only two values ever occur: **FOR 890 (48.1%), AGAINST 229 (12.4%)**,
NULL 732 (39.6%). No improving trend (2020→2026: 62.9 / 67.6 / 61.8 / 68.6 / 73.2 / 63.0 / 59.0).
`n_voting_proposals` is 0 on **249 of 684 (36.4%)** and always equals the actual child row count.

**`extract_voting_proposals` is a pure regex over `filing.text()` — it never looks at a table.**
Anchor (`html_extractor.py:134-137`):
```
(?:proposal\s+(?:no\.?\s*)?|item\s+)(\d+)(?!\.\d)\s*[:\-—–.\s]+([^\n]{10,200})
```
Then a chain of silent drops (number 0 / duplicate / >30; description <15 chars; description starting
with a digit or a stopword; duplicate on `description.lower()[:40]`), and finally **line 252-253: if
the lowest surviving number is not 1, the entire list is discarded and `[]` returned** — which is how
36.4% of filings end up with zero proposals.

**The recommendation text is present in 100% of the misses.** Two independent probes agree:
- Probe A: 129 / 422 proposals (30.6%) None, flat across time (21.6% in 2001 → 28.0% in 2026).
- Probe B: 39 / 123 (31.7%) None across 22 proxies; **text present in 39/39**. Plus 2/22 proxies
  return **zero** proposals (T 2021 `0001193125-21-077769`, T 2026 `0001193125-26-119888`).

**In 4 of 4 failures opened, the recommendation lives in a TABLE**, which the parser cannot see:
```
DIS 2021 | PROPOSAL | FOR MORE INFORMATION | BOARD RECOMMENDATION |
AAPL 2026 | Proposal | Board Recommendation | Page Reference |
PG 2016  | Voting Matter | Vote Standard | Board Vote Recommendation | See Page |
T 2026   | Management Proposals: | Board Recommendation | Page |
```
T never writes "Proposal 1" — it writes `Management Proposals:` with rows `1.`–`8.`, which is exactly
why the regex finds nothing. **DIS 2021's recommendation exists ONLY in the table**; its prose hits
are all committee-recommends-to-board boilerplate.

Phrasing frequency across 22 proxies: `(Your|The) Board recommends` 22 · `recommends a vote AGAINST`
21 · `recommends a vote FOR` 19 · `recommends that you/shareholders vote` 13 · `FOR EACH NOMINEE` 13 ·
`Board Recommendation` header 13 · `unanimously recommends` 3. A regex for `recommends a vote FOR`
returns **zero** matches in JPMorgan's 2025 proxy (their idiom is `recommends you vote FOR`).

**Wrong values, not just missing ones.** Three independent instances:
- `_RECOMMENDATION_PATTERNS[4]` is **`\b(FOR)\s+Against\s+Abstain\b`** (`html_extractor.py:82`) —
  that is the **proxy card's column header**, and `m.group(1)` is the literal word `FOR`. Any
  proposal whose 5,000-char window contains a proxy card is assigned `FOR` regardless of the board's
  actual position. **This fabricates values into the 890 `FOR` rows.**
- KO 2021 proposal 3 returned **AGAINST** where the document says *"Board recommends a vote FOR the
  ratification"*. KO 2021 also emitted a bogus proposal numbered 14 described as *"What if I am a
  beneficial owner and do not give voting instructions to my broker"*.
- XOM 2026 assigns `FOR` to two shareholder proposals the board opposes.

**Hard limits on any parser.** Apple's recommendation cell is a **1,665-byte blue check-mark JPEG
with `alt=""`**. Intel's 2026 proxy embeds its proxy card as **two 792×1024 JPEGs**. Board
recommendations have **no XBRL tag and no prescribed wording** — `recommend` appears zero times in
Rules 14a-4 and 14a-19.

`proposal_type` classification (`_classify_proposal`, 66-71) runs on the *cleaned, truncated*
description, so truncation destroys the signal: AAPL's proposal 4 (an equity plan) is classified
`company_proposal` because `_clean_description` cut it to `'Approval of the Apple Inc.'` at the
period in "Apple Inc." `company_proposal` has **no trigger — it is the hardcoded default at line 71**,
which is why it is the largest bucket at 33.6%.

### 10. Vote tallies — obtainable, and the substrate is already in our DB

Vote counts are in **Form 8-K Item 5.07** and nowhere else. Item 5.07 began **March 2010**
(Rel. 33-9089 moved the disclosure out of 10-Q Part II Item 4 / 10-K Part I Item 4); efts shows 0
hits before 2010-03-01 and 738 in Mar–Apr 2010. **No vote number is ever in XBRL**: Apple's 2025 8-K
FilingSummary is `isOnlyDei="true"` with `<UnitCount>0</UnitCount>`; live probes returned 46/49/31
facts, **100% `dei:` cover-page tags**.

**We already fetch it.** `fetch_8k_edgar.py:57` tags `_HIGH_SIGNAL_ITEMS["5.07"] =
"vote_of_security_holders"`. Live DB: `sec_8k WHERE item='5.07'` = **5,537 rows, 5,492 with
`item_text` > 200 chars (99.2%), 336 tickers, 2010-03-01 → 2026-08-19**. The narrative with the
numbers is stored; nothing parses it. Discovery is free — the SEC submissions JSON `items` array
carries the codes, no download needed; 7 tickers × 2019-2026 = 57 filings, one per meeting, zero gaps.

edgartools gives `.items == ['Item 5.07']`, `.content_type == 'shareholder_vote'`,
`.sections['item_507']`, and `o['Item 5.07']` → a **plain `str`**. **There is no vote DataFrame
anywhere in the library** — the only DataFrames route through `EarningsRelease` (EX-99.1), and
`o.earnings` is `None` on a 5.07.

Parse difficulty, measured across 34 filings 2010–2026:
- 59% one table per proposal, **41% combined**; 85% have per-nominee director rows.
- **12 distinct header vocabularies for 34 filings.** GE alone switches `Non-Votes` (2019) →
  `Broker Non-Votes` (2024).
- **20.0% use "Withheld" not "Against"**; 25.4% carry say-on-frequency (4 buckets); 75.2% mention
  broker non-votes, encoded three ways when absent (column omitted / `N/A` / `0`).
- Numbers are **always shares**; percentages are an extra (21.1%).
- **8.8% have no HTML vote table at all** (HUM `0000049071-23-000042` is a legitimate tally-free Item
  5.07(d) board-response filing; ES 2010 and IBKR 2012 narrate in prose).

Three same-quarter examples show three incompatible geometries: AAPL `0001140361-25-005876` is
per-nominee 4-col `For|Against|Abstained|Broker Non-Vote` with the auditor table dropping to 3
columns; JPM `0000019617-25-000485` stacks a `91.45 | % | 8.14 | %` row *inside* the table and writes
`N/A`; XOM `0000034088-25-000030` interleaves `% For`/`% Against` and **transposes** non-director
proposals (`Votes Cast For: | 3,495,486,371 | 96.8 %`).

**The biggest trap: in contested elections the first 8-K is PRELIMINARY.** DIS
`0000950157-24-000595` states *"estimated preliminary voting results reported by … Innisfree … do not
include shares voted on the blue proxy card distributed by Trian"*; the 8-K/A `-000623` carries the
Inspector of Election's final results (and switches to For/Withhold). Same for XOM 2021 / Engine
No. 1. **224 / 5,537 (4.0%) are 8-K/A; 167 / 5,367 meetings (3.1%) have >1 filing** — but only 14
filings say "preliminary", so it is not signposted.

**Alternative sources, all settled empirically**: efts returns `"items":["5.07"]` but is capped at
10,000 results and yields pointers, never numbers. **FSDS carries no proxies** — 2025q1.zip (127 MB),
`sub.txt` = 6,231 submissions, **0 DEF 14A, 4 8-K**. **N-PX** is voter-level: SSgA
`0001021408-25-004923` has 2,869 records, every row `SECTION 14A SAY-ON-PAY VOTES` (13F managers
report only say-on-pay under Rule 14Ad-1), and it voted 1,478,942 PG shares against ~2.34bn
outstanding (**0.06%**) — an institutional sample, never the certified tally.

**Prior-year say-on-pay narrative in the proxy: present in 8/8**, but lossy and trap-laden. Merck
says "approximately 94%" where its 8-K says **93.50%**. XOM 2026 puts it in a *table*
(`Say-on-Pay | Votes "For" | 92% 92% 91%`), so sentence-regex scores 7/8. False positives are
material: T quotes a *proponent's* 42% claim, DIS quotes a Harvard Law Forum statistic. The two
sources do reconcile — AAPL's proxy "92% of votes cast" vs its 5.07 For/(For+Against) = **92.4%** —
but XOM states abstentions are not counted as votes cast in New Jersey, so the denominator convention
varies by state of incorporation.

### 11. Ownership

3,477 rows. `holder_type`: `director_officer` 2,976 (85.6%), `5pct_holder` 501 (14.4%).
`shares` non-null 74.3%. `percent_of_class` non-null **17.3% overall but 97.8% for 5pct_holder vs
6.3% for director_officer** — it is effectively a 5%-holder-only field.

**The repair layer is working.** **0 rows equal 0.5** in the DB, against **119 of 189 (63%)** of
raw library outputs in the live probe (9/12 in 2001, 32/42 in 2026). `_parse_percent` still hardcodes
`return 0.5` at `html_extractor.py:959-960`, and inconsistently: `'*'`→0.5 and `'less than 1%'`→0.5,
but `'**'`→None, `'<1%'`→None, `'Less than 1 percent'`→None.

**PG 10× — reproduced, still broken in 5.51.0, and it is not a 10× scale error.**
`0001193125-25-191749` returns Vanguard **2,249,200,352 @ 9.60%**; the filed source is
**224,920,035** with a footnote marker. Raw HTML:
```html
217,956,036<sup style="font-size:85%; vertical-align:top">2</sup>   →   2179560362
```
It is **decimal concatenation of the footnote digit**. `_FOOTNOTE_RE` strips only *parenthesised*
markers; `_parse_shares` (943-953) has no decimal support and no leading-text tolerance
(`'1,234,567 shares'` → **None**). **12/12 rows across 6 consecutive proxies (2021–2026), both
holders, 100% affected.** Two useful properties: `percent_of_class` is **correct in every case** (the
pct cell carries no marker), so shares-vs-percent diverge by ~10× — a **detectable invariant** given
shares outstanding; and the 2026 proxy uses a CSS-positioned `<span>` rather than `<sup>` and the bug
survives, so it lives in text flattening, not `<sup>` handling.

Other measured ownership defects: **MSFT_2006** puts share counts in `holder_name`
(`'957,499,336'`) with `shares` None on all 14 rows; **JPM_2026** puts addresses in `holder_name`;
**GE_2026** types CEO `H. Lawrence Culp, Jr.` as `5pct_holder` and a literal `'Total'` row as
`director_officer`. `beneficial_ownership` has **no subtotal filter at all** (contrast SCT line 882
and director comp line 1173), and the holder-type cascade's terminal fallback (line 1078) is a
fabricated `'5pct_holder'` label. `\xa0` is normalised for director comp (1169) but **not** for
ownership (1046) — live AAPL returns `'The\xa0Vanguard\xa0Group'`.

DB-side, 5 rows have `percent_of_class > 100`, all AIG `0000950117-03-001327`, with `shares` NULL —
share counts written into the percent column (Greenberg 36,637,500).

**Structured substitutes**: institutional share counts ← 13F (quarter-end); 5%-holder percent-of-class
← SC 13D/G; **insider/director holdings ← Forms 3/4/5, not 13F** (confirmed live: PG has 97 Form 3,
3,709 Form 4, 22 Form 5; `Form4.to_dataframe()` carries `Remaining Shares` = the post-transaction
level). **Not recoverable from any structured source**: the "directors and officers as a group"
subtotal, the 60-day-exercisable-options overlay, footnote attributions (trusts/family/401k), the
record date, retail ownership. As-of dates never align.

### 12. Auditor

`auditor_name` is the worst-filled column in the parent table at **2.05%**, and the live probe
explains why: `_extract_auditor_name` **returns `''` on failure** (`html_extractor.py:1293`) — an
empty string, not None. Across 41 cells that returned an `AuditFees` object, `auditor_name` is `''`
in **40 of 41**; the single non-empty value is a **148-character sentence**, not a firm name.

`audit_fees` returns `None` for AAPL entirely: `_find_section_table` anchored on `audit fees` and
selected the **footnote table** (`[['(1)','Audit fees relate to…'],['(3)','Tax fees … in 2025 … in
2024 …'],…]`), which scored ≥3 because `_YEAR_RE` matched inside the prose.
`_detect_year_columns` then returned `None` (both years live in the *same* cell and `.search` yields
only the first). The real fee table was never reached.

**Units are not normalised.** The same column carries three scales across 19 non-None cells:
CAT_2016 `=32` and MMM_2026 `=18` ($ millions); KO_2026 `=30587` and PG_2016 `=30937` ($ thousands);
AAPL_2021 `=17568300` and PFE_2026 `=24154000` ($). **PG alone flips scale between anchors**
(30937 → 28468000). Our `_rescale_block` handles the thousands case but its floor
(`DEF14A_AUDIT_FEE_MIN_PLAUSIBLE = 1e5`) rescales `=32` by 1e6, which is right only if the units note
is "millions".

The 5.44.1 note that `_detect_multiplier` "misses (in thousands)" is **not the real mechanism**.
`_THOUSANDS_RE` matches all common phrasings (`'(in thousands)'`, `'($ in thousands)'`,
`'dollars in thousands'` → 1000). The load-bearing limitation is at **line 1341-1343**: the text
searched is `table_el.getparent().text_content()[:2000]` — the **immediate parent only, first 2,000
chars**. A units note in a sibling paragraph or a different ancestor is never seen.

**`dei:AuditorName` IS a structured alternative for the auditor's identity** — `dei:AuditorName` +
`AuditorFirmId` (PCAOB ID) + `AuditorLocation`, in the **10-K**, measured 5/5 tickers from filings
2022 onward and 0/5 in 2019 and 2021 (rule: FY ending ≥ 2021-12-15). Identity only.

**There is no XBRL tag for audit fees anywhere. Definitive**: `us-gaap-2025.xsd` = 17,335 elements,
**zero** containing "Audit" (control: the `ProfessionalFees` family does resolve); `dei-2025.xsd` =
237 names, **zero** containing "Fee"; `ecd-2025.xsd` = 125 names, zero matching Audit/Fee/Vote/
Proposal. AAPL's 10-K exposes 384 concepts, zero "fee".

### 13. The SGML warning — cosmetic

```
edgar.core - WARNING - _filings.py - SGML fetch failed for 0000874761-15-000021,
falling back to homepage: SEC returned HTML or XML content instead of expected SGML filing data.
```

Emitted at `edgar/_filings.py:2013-2016` inside `Filing.sgml()`. The accession is **AES Corp**
(CIK 874761), not Halliburton.

**It is transient and server-side, not a vintage or format problem.** Fetching the canonical URL
directly returns **HTTP 200 / 2,146,805 bytes** of well-formed `<SEC-DOCUMENT>` SGML that
`detect_format` accepts at `sgml_parser.py:263`. On a clean fetch the warning **does not fire at
all** (0 warnings, 0 log lines). The trigger is `_raise_sec_html_error` (`sgml_parser.py:181-218`)
receiving an EDGAR/CloudFront interstitial that contains neither "Not Found" nor "404".

To settle the impact deterministically, `FilingSGML.from_filing` was monkeypatched to raise the
transient error and both accessions re-run:

| | AAPL 2026 normal | forced fallback | AES 2015 normal | forced fallback |
|---|---|---|---|---|
| summary_compensation_table | (6,11) | **(6,11)** | (0,0) | (0,0) |
| director_compensation_table | (7,8) | **(7,8)** | (0,0) | (0,0) |
| beneficial_ownership | (16,4) | **(16,4)** | (0,0) | (0,0) |
| voting_proposals | 5 | **5** | 8 | 8 |
| peo_name / peo_total_comp | Mr. Cook / 74294811.0 | **identical** | None | None |
| ceo_pay_ratio | ratio=533 | **identical** | None | None |

**Every extracted field is byte-identical.** The only measurable difference is attachment count
(AAPL 66→58). AES 2015 returns empty tables on the happy path too, so **the warning is not even
diagnostic of the data loss it appears next to.**

The fallback (`sgml_common.py:547-568`) is nonetheless a *partial* recovery: it builds
`FilingHeader(text="", filing_metadata={})`, so `filing.sgml().header` is empty and
`period_of_report`, filer data and SIC-based standardization silently degrade to `None`, and
`filing.filing_summary` / `.reports` / `.statements` become `None`. `filing.html()` still works via
`homepage.primary_html_document.download()` (one extra GET), and `xbrl()` works via per-linkbase
downloads (~6 extra GETs). Caching caveat: `from_source` only retries with `bypass_cache` when the
body is <50 bytes (`sgml_common.py:494-501`), so a cached HTML error page re-fails identically until
it expires.

Note what does **not** fall back and propagates instead (`_filings.py:2004-2011`):
`SECIdentityError`, `FilingNotFoundError`, `IdentityNotSetError`, and all httpx/httpcore network
errors.

### 14. Form scope and structural blind spots

`DEF14A_FORMS = ["DEF 14A", "DEF 14C"]` (`src/constants/constants.py:86`); `years_history: 31`
(`configs/configs.yml:4`).

edgartools' `PROXY_FORMS` (`edgar/proxy/models.py:27-39`) is **26 form strings** including `DEFA14A`,
`DEFM14A`, `DEFR14A`, `PRE 14A` and `PX14A6G`. `ProxyStatement.from_filing` **never returns `None`**
(`core.py:68-71`); the constructor's only guard is an `assert` (line 60-63), which **vanishes under
`python -O`**. `fetch_def14a_edgar.build_ticker_def14a_edgar`'s `hasattr(proxy, "voting_proposals")`
check therefore never fires for a DEF 14C — the docstring's claim that DEF 14C "falls through to a
generic XBRL-only object" does not match 5.51.0, where DEF 14C is not in `PROXY_FORMS` at all and
`matches_form` fails, so `.obj()` returns something else entirely.

Structural blind spots for a DEF 14A-only fetcher, all measured:
- **DEF 14C carries the tables and the XBRL** — Erie Indemnity's 2025 DEF 14C has 134 ecd facts;
  Meta `0001326801-22-000128` has a full SCT. 13.4% of DEF 14C EDGAR-wide contain one.
- **Some issuers file no proxy at all.** Blackstone puts full Item 402 + pay ratio in the 10-K
  (cover: `DOCUMENTS INCORPORATED BY REFERENCE: None`) and has **zero Pay Versus Performance**,
  because 402(v) reaches only proxy/information statements. MLPs (EPD, MPLX, WES) never file a proxy.
- **4,801 10-K/A filings 2015–2025 contain an SCT** (issuers that missed the 120-day General
  Instruction G(3) window).
- **FPIs file zero DEF 14A** (Rule 3a12-3(b), confirmed 0 across 9 issuers / ~30 years); 20-F Item
  6.B prescribes **no table at all**; Canadian comp arrives on 6-K under 51-102F6 **in C$**.
- **Special-meeting DEF 14As have no Item 402 content.** 35 S&P 500 company-years have >1 DEF 14A;
  sampled extras (TTD, MRNA, KKR) all measured SCT-absent, PVP-absent, 0 ix facts. Our DB has
  **12 `(ticker, year)` pairs with 2 filings** (AAPL 2017, ADSK 2005/2007/2011/2013, AES 2003/2015…).
- **DEFR14A can be the tagged copy**: Intel's DEF 14A `0000050863-26-000061` (2026-03-23) has SCT +
  PVP but **zero XBRL**; the DEFR14A filed the next day carries it.

### 15. Regulatory timeline (the "date threshold" question, answered)

| Threshold | Date | Effect |
|---|---|---|
| EDGAR phase-in complete | 1996-05-06 | before this, proxies may not be on EDGAR |
| **HTML first PERMITTED** | **1999-06-28** (EDGAR 6.50, Rel. 33-7684) | every DEF 14A filed 1994–1999 is ASCII; no HTML tables exist |
| Images allowed in HTML | 2000-05-30 (Rel. 33-7855) | |
| Auditor fees, 3 categories | proxies filed after 2001-02-05 (Rel. 33-7919) | |
| Auditor fees, 4 modern categories | FY ending after 2003-12-15 (Rel. 33-8183) | Audit / Audit-Related / Tax / All Other |
| **Item 402 overhaul** | FY ending ≥ 2006-12-15, **proxies spring 2007** (Rel. 33-8732A) | SCT columns change wholesale; **Director Comp Table and CD&A born** |
| SCT equity-basis flip | FY ending ≥ 2009-12-20, proxies spring 2010 (Rel. 33-9089) | ASC-718 expense → grant-date fair value |
| Vote results relocate | 2010-02-28 (Rel. 33-9089) | 10-Q/10-K Item 4 → **8-K Item 5.07** |
| Say-on-pay | meetings ≥ 2011-01-21 (SRCs 2013-01-21) | Dodd-Frank §951, self-executing |
| CEO pay ratio | FY2017, proxies 2018 (Rel. 33-9877) | **narrative only, forever** |
| **PVP + first XBRL ever in a proxy** | **FY ending ≥ 2022-12-16, proxies spring 2023** (Rel. 34-95607) | `ecd:` inline XBRL |
| Item 402(x) award timing | FY2024, proxies 2025 | `ecd:AwardTmg*` |

**ASCII→HTML has no cliff.** Measured on 146 filings from 10 mega-caps: 0% HTML 1996–99 → **40% in
2000** → 40/40/70/70/80% → **100% in 2006**. GE filed plain ASCII on 2004-03-02; Merck and J&J in
March 2005. **HTML was never mandated for Schedule 14A** (Rel. 33-10322 explicitly exempts proxies;
ASCII is still legal today). **File extension is not a format test** — CAT 2000's HTML lives in a
`.txt`.

Live probe corroboration: 9 grid cells have an ASCII `.txt` primary document with no `<html>` tag
(8 at the 2001 anchor + T_2006), and **all 9 returned (0,0) for all three tables**. The
director-compensation table is 0/15 at the 2001 anchor and only becomes reliable from the 2008
season; CAT returns (0,0) in all six years 2004–2009.

Additional pre-2001 finding: for filings ≤ 2000, `filing.document.url` does not resolve to the
document — it returns a ~10 KB EDGAR *folder index* page. GE_2001 and T_2001 `document.url` **HTTP
404** while the sibling `<accession>.txt` returns 200 / 144,271 bytes.

**Corrections to tag names assumed in the brief**: `ecd:NetIncLossAmt` **does not exist** — PVP
column (h) is `us-gaap:NetIncomeLoss`, wired `priority="-1"` so filers may override it (this is what
`fetch_def14a_edgar` already reads). `ecd:AwardsCloseToMNPIDisclOsdTableTextBlock` →
`ecd:AwardsCloseToMnpiDiscTableTextBlock`. `ecd:AggtAvailAmt` → `ecd:TrdArrSecuritiesAggAvailAmt`.
`ecd:RestatementDeterminationDateAxis` → `ecd:RestatementDateAxis`. **"Filed mainly by accelerated
filers" is not the rule** — acceleration status is irrelevant to 402(v); EGC/RIC/FPI are exempt, and
SRCs get a scaled set and tag only from their **third filing** (a filing count, not a date).

---

## Code References

| Reference | What is there |
|---|---|
| `src/data_extract/utils/structure/fetch_def14a_edgar.py:1-321` | The whole edgar path; `_MAIN_COLS` and 4 child column lists; `_NUMERIC_COLS` doubles as the table list |
| `fetch_def14a_edgar.py:229-235` | `hasattr(proxy, "voting_proposals")` guard — never fires for DEF 14C in 5.51.0 |
| `src/data_extract/utils/structure/def14a_validate.py` | The repair layer; `never fabricate` rule |
| `def14a_validate.py:64-71` | `DEF14A_*` sanity bounds — **no magnitude bound on comp components** |
| `def14a_validate.py` `_reconcile_components` | Duplicated-Total repair + single-missing-component residual |
| `src/data_extract/utils/common/edgar_driver.py:40-52` | `new_filings` — accession dedup + `since` filter |
| `edgar_driver.py:96-105` | `create_lock` guarding the `ensure_table` race |
| `src/data_extract/utils/structure/fetch_def14a_llm.py:63` | `_FORM = DEF14A_FORMS`; gpt-4o-mini, 7 sections, ~51k chars |
| `fetch_def14a_llm.py:331-365` | `prepare_def14a_sections` — densest-window slicing |
| `src/data_extract/utils/structure/fetch_8k_edgar.py:57` | `_HIGH_SIGNAL_ITEMS["5.07"] = "vote_of_security_holders"` |
| `src/data_store/schema.py:436-464` | All 6 table registrations, PKs and date cols |
| `src/data_extract/transformers/step_extract_structure.py:26-36` | Both DEF 14A paths run unconditionally |
| `src/data_extract/cli.py:394-400, 434-441` | `def14a` and `def14a_edgar` commands |
| `src/constants/constants.py:86` | `DEF14A_FORMS = ["DEF 14A", "DEF 14C"]` |
| `configs/build_cube.yml:197-200` | `governance:` composite — fed by `def14a_llm`, not `sec_def14a` |
| `docs/data_sources.md:201-207` | "edgartools' proxy HTML parser is silently wrong, not absent" |
| `docs/database.md:125-127` | Records the 23/500 coverage and the ~95% NaN consequence |

**Table PKs** (`src/data_store/schema.py:436-464`):

| Attr | SQL name | PK | date_col |
|---|---|---|---|
| `def14a_llm` | `def14a_llm` | `(ticker, accession_number)` | `as_of` |
| `def14a_edgar` | `sec_def14a` | `(ticker, accession_number)` | `filing_date` |
| `def14a_edgar_executive_comp` | `sec_def14a_executive_comp` | `(ticker, accession_number, name, year)` | `filing_date` |
| `def14a_edgar_director_comp` | `sec_def14a_director_comp` | `(ticker, accession_number, name)` | `filing_date` |
| `def14a_edgar_ownership` | `sec_def14a_ownership` | `(ticker, accession_number, holder_name, holder_type)` | `filing_date` |
| `def14a_edgar_votes` | `sec_def14a_votes` | `(ticker, accession_number, proposal_number)` | `filing_date` |

**Library references** (site-packages, edgartools 5.51.0):

| Reference | What is there |
|---|---|
| `edgar/proxy/core.py:105-151` | `_get_concept_value` / `_get_concept_series` — **no dimension filter** |
| `edgar/proxy/core.py:193-196` | `company_name` ← `dei:EntityRegistrantName`, no fallback |
| `edgar/proxy/core.py:214-217` | `peo_name` ← `ecd:PeoName` (which tags every NEO) |
| `edgar/proxy/core.py:219-287` | The 9 PvP properties and their tags |
| `edgar/proxy/core.py:778-804` | `named_executives` — hardcodes `role='PEO'` as the default |
| `edgar/proxy/html_extractor.py:82` | `\b(FOR)\s+Against\s+Abstain\b` — matches the proxy-card header |
| `edgar/proxy/html_extractor.py:134-137` | `_PROPOSAL_PATTERN`; 252-253 discards the list if proposal 1 is missing |
| `edgar/proxy/html_extractor.py:270-296` | `_parse_dollar_amount` — divides >$200M by 10 |
| `edgar/proxy/html_extractor.py:430-457` | Pay-ratio swap / two-largest-dollars / strip-last-digit repairs |
| `edgar/proxy/html_extractor.py:567-601` | `_split_name_title` — `\s+` collapse kills the `\n` split |
| `edgar/proxy/html_extractor.py:661-677` | `_extract_table_grid` — colspan expanded, **rowspan ignored**, `<br>` dropped with no separator |
| `edgar/proxy/html_extractor.py:680-707` | `_split_header_data` — exactly one header row, scans `grid[:4]` |
| `edgar/proxy/html_extractor.py:818-830` | `_get_dollar_from_row` — the empty-cell → `row[idx+1]` fallback |
| `edgar/proxy/html_extractor.py:886-898` | The `year_col+1` fallback that yields `year=3000` |
| `edgar/proxy/html_extractor.py:943-969` | `_parse_shares` (no footnote-digit strip) and `_parse_percent` (hardcoded 0.5 at 959-960) |
| `edgar/proxy/html_extractor.py:1102-1111` | Director-comp header synonyms |
| `edgar/proxy/html_extractor.py:1293` | `_extract_auditor_name` returns `''` on failure |
| `edgar/proxy/html_extractor.py:1341-1343` | Multiplier searched in `getparent().text_content()[:2000]` only |
| `edgar/_filings.py:2013-2016` | The SGML warning |
| `edgar/sgml/sgml_common.py:547-568` | The homepage fallback (empty `FilingHeader`) |
| `edgar/xbrl/parsers/instance.py:396-411, 430-436` | Prefix resolution; `float(value)` with no scale/sign handling |

---

## Architecture Documentation

Two independent DEF 14A paths, both running unconditionally inside `StepExtractStructure.run`:

- **`fetch_def14a_edgar`** — zero LLM cost, one HTTP request per filing (`Filing.sgml()`; every
  property afterwards is an in-memory slice, all `@cached_property`). Writes 1 parent + 4 child
  tables. Every row passes `def14a_validate` before persisting.
- **`fetch_def14a_llm`** — gpt-4o-mini over 7 section-narrowed slices (~51k of ~122k chars),
  accession-level dedup, per-ticker upsert. Writes the flat 45-column `def14a_llm` plus a
  `def14a_json` dump. **This is the path with real coverage and the only one the cube consumes.**

The `sec_*` naming and shape come from `specs/2028-08-19/2026-08-19_refactor_sec_structure.md`.
The repair layer's guiding rule (`def14a_validate.py` docstring) is **never fabricate**: write a value
only when deterministically recoverable, else NaN. That rule is doing real work — the DB has **0**
rows at the fabricated `0.5` percent against **63%** of raw library outputs.

`fetch_def14a_edgar.py` already applies the "prefer filing metadata over the library property"
pattern for `period_of_report`; `company_name` is the same shape of problem, unaddressed.

## Key Data Flows

```
Company(ticker).get_filings(form=DEF14A_FORMS)
  └─ new_filings()  → accession dedup + `since`          [edgar_driver.py:40]
      └─ filing.obj() → ProxyStatement                    [never None; never raises]
          ├─ XBRL block  → filing.xbrl() → facts df       [2023+ only; no dim filter; no scale]
          │    → peo_*, neo_*, tsr, net_income, ecd flags, company_name
          ├─ HTML block  → lxml tree over filing.html()
          │    → SCT, director comp, ownership, audit fees
          └─ text block  → filing.text()
               → voting_proposals, ceo_pay_ratio
      └─ repair_main_row / repair_*_rows                  [def14a_validate.py]
      └─ drop_duplicates(table.pk) → _coerce_numeric      [fetch_def14a_edgar.py:262]
      └─ store.save(table, df)                            [via edgar_driver._save]

def14a_llm ──> step_cube_extras._governance_panel ──> governance_features ──> cube_part_extras
sec_def14a* ──> (no consumers)
sec_8k[item='5.07'].item_text ──> (no consumers; 5,492 unparsed vote narratives)
```

## Dependencies

- `edgartools` 5.51.0 — `ProxyStatement`, `CurrentReport`/`EightK`, `XBRL`, `FilingSGML`
- `lxml.html` (via edgartools), `pandas`, `openai` (LLM path only)
- Internal: `src/data_extract/utils/common/{edgar_driver, parallel_fetch, run_manifest, sec_utils}`,
  `src/data_store/{schema, store}`, `src/constants/constants.py`

## Test Coverage

| File | What it asserts |
|---|---|
| `tests/data_extract/structure/test_def14a_validate.py` | Repair-layer units. **Fixtures are real observed defects** — KO thousands rescale, PG net_income, CAT pension-duplication, Apple 0.5 placeholder, JPM address-as-holder |
| `tests/data_extract/structure/test_fetch_def14a_edgar.py` | Pure-synthetic `SimpleNamespace` fakes of `ProxyStatement`; row builders, incremental dedup, `since` cutoff, DEF 14C skip |
| `tests/data_extract/structure/test_def14a_llm.py` | Schema roundtrip, section-anchor regexes, `_flatten`, mocked extractor, live-Apple (needs `OPENAI_API_KEY`), Postgres upsert, incremental gap-fill |
| `tests/data_extract/structure/test_def14a_incremental.py` | `_is_up_to_date` per-ticker semantics, manifest window narrowing |
| `tests/data_extract/structure/test_def14a_nul.py` | `_strip_nul` (Postgres TEXT rejects `\x00`) |
| `tests/data_aggregate/test_def14a_impute.py` | Live-DB non-destructiveness of `impute_def14a` |
| `tests/data_aggregate/test_def14a_say_on_pay.py` | `SAY_ON_PAY_MIN_SUPPORT = 0.50` floor. ⚠ **Premise measured FALSE** (see follow-up 2): 14/14 sampled sub-0.50 values are CORRECT, including all three the docstring cites as errors. The floor nulls **61 correct rows** |

Gaps: **no test covers the `sec_def14a*` tables against real filings** — `test_fetch_def14a_edgar.py`
is entirely synthetic, so every defect in this report is invisible to CI. There is **no SQL PASS/FAIL
check set for DEF 14A** (the "16 checks" referenced in an earlier note is not in the tree; the only
PASS/FAIL validator artifacts are fundamentals-domain).

## Trustworthiness for alpha construction

| Block | Verdict | Basis |
|---|---|---|
| ECD XBRL (PvP, TSR, governance flags, pay ratio) | **Trustworthy 2023+ IF dimension-filtered** | Filer-tagged, machine-readable. But co-PEO years silently drop a PEO (BA, NKE), zero-matrix filers return 0.0 (SBUX), and units are unnormalised (SBUX net_income 1856.4) |
| `ceo_pay_ratio` | **Usable 2018+** | 15/15 cells at 2021 and 2026. But narrative-only forever, and the extractor has 3 value-inventing repairs (swap, two-largest-dollars, strip-last-digit) |
| Director comp `total` | **Usable** | 99.65% filled, and the docstring's judgement holds — it is the number the filer printed |
| Director comp components | **Not usable** | 0 rows have all six; `option_awards` 0% since 2022; the value is in the source in 4/4 cases opened |
| Executive comp | **Not usable** | 109 rows >$1e9; 47% of accessions carry 1 year instead of 3; title 54.6% NULL; 7.06% encoding corruption |
| `board_recommendation` | **Not usable** | 39.6% NULL, and the `FOR Against Abstain` proxy-card pattern **fabricates `FOR`** into an unknown share of the 890 `FOR` rows |
| Ownership `percent_of_class` | **Usable for 5% holders** (97.8% filled there) | 0.5 placeholder neutralised by our repair layer |
| Ownership `shares` | **Not usable as-is** | PG 12/12 rows wrong across 6 years; detectable via the shares-vs-percent invariant |
| `auditor_name` | **Not usable** (2.05%) | Library returns `''` on failure, 40/41 cells empty |
| Audit fees | **Not usable as-is** | Three scales in one column; PG flips scale between its own filings |

The load-bearing asymmetry for a quant use: these defects are **not random noise**. They are
**ticker-persistent** (AMAT is wrong in 9 consecutive years; PG in 6; A in 2) and
**layout-correlated**, so they survive cross-sectional ranking and z-scoring as a fixed per-issuer
bias rather than averaging out. Several are also **silently wrong rather than missing**, so a
NULL-rate check passes while the values are fiction.

## Related Documentation

- `docs/data_sources.md:201-207` — DEF 14A source quirks
- `docs/data_schema.md:102-120` — both tables' PKs/columns and the repair list
- `docs/database.md:92, 100-104, 125-127` — measured coverage and the ~95% NaN warning
- `specs/2026-09-01/investigate-def-14a.md` — the spec driving this research
- `specs/2028-08-19/2026-08-19_refactor_sec_structure.md` — origin of the `sec_*` naming/shape
- Scratchpad measurement notes: `db_measurements.md`, `sec_format_timeline.md` (2,582 lines),
  `live_probe_comp.md` (1,100+ lines), `vote_and_ownership_sources.md` (888 lines), plus
  `probe_results.json`, `frames.pkl` (270 DataFrames) and 130+ cached raw filings

## Open Questions for Planning Phase

1. **Backfill vs parser.** The 26/500 coverage is the largest term by far and is independent of every
   parsing question. Sequencing matters: fixing parsers first means re-fetching; backfilling first
   means TRUNCATE + re-fetch later, since `existing_filings` dedups on accession and would skip
   everything.
2. **Two paths, one consumer.** `def14a_llm` has 497-ticker coverage and feeds the cube;
   `sec_def14a*` has 26 and feeds nothing. Whether the edgar path is meant to *replace*, *validate*,
   or *complement* the LLM path is not recorded anywhere and changes what "fixed" means.
3. **Where repairs belong.** Several defects are only fixable *above* edgartools (dimension-filtering
   the ECD facts, `company_name` ← `filing.company`); others need the raw HTML, which the current
   architecture reads only inside the library (the `<br>` concatenation, the `$`-in-own-`<td>`
   shape, header synonyms, the `<sup>` footnote digit).
4. **Detectable invariants already available**: shares × percent_of_class vs shares outstanding (PG);
   comp components vs `total`; audit-fee scale vs a sibling filing's overlapping year; year values
   outside [1990, today+1]. There is currently **no magnitude bound on comp components** at all.
5. **Vote tallies.** 5,492 Item 5.07 narratives are already stored and unparsed. The blockers are
   12 header vocabularies, 41% combined-table layouts, 20% For/Withheld, and the **4% preliminary
   8-K/A problem, which is not signposted in the text**.
6. **`board_recommendation` substrate.** The recommendation is in a **table** in 4/4 failures opened,
   and the current extractor never reads a table. Ceiling: Apple's cell is an image with `alt=""`.
7. **Form scope.** `DEF14A_FORMS` includes DEF 14C, but DEF 14C is not in edgartools' `PROXY_FORMS`,
   so `.obj()` does not return a `ProxyStatement` and the guard in `build_ticker_def14a_edgar`
   behaves differently than its docstring states. DEFR14A can be the only tagged copy (Intel 2026).
   Special-meeting DEF 14As have no Item 402 content but do occupy a PK.
8. **Pre-2000 filings.** `filing.document.url` resolves to a folder index, and two probed filings
   404. Any pre-2001 ambition needs the `<accession>.txt` path instead — but those years are ASCII,
   where 9/9 probed cells returned (0,0) on all three tables.
9. **Test substrate.** Every existing edgar-path test is synthetic, so none of this is CI-visible.

---

## Follow-up Research — bulk & third-party vote-outcome sources (2026-09-01 16:40)

Extends §10. All endpoints tested live 2026-09-01 with a declared User-Agent.
**`WebFetch` gets HTTP 403 from every `sec.gov` host** — SEC requires a User-Agent, so SEC probes
must go through `curl`.

### EDGAR full-text search (EFTS)

The correct endpoint is **`https://efts.sec.gov/LATEST/search-index`**, confirmed by reading the
search UI's own JS (`https://www.sec.gov/edgar/search/js/edgar_full_text_search.js`, which hardcodes
it). The legacy `cgi-bin/srqsb` and `cgi-srv/srqsb` endpoints both **404** with no successor.

- **Coverage starts 2001.** Measured: a 1994→2000 window for `"annual meeting of shareholders"`
  returns **8** hits (misdated strays); 2001-Q1 alone returns **6,079**; an 8-K query restricted to
  calendar 2000 returns **0**.
- **`forms=8-K` is honored.** So are `q`, `dateRange`/`startdt`/`enddt`, `ciks`, `entityName`,
  `sics`, `fileType`, `sort`, `from`.
- **`items=` is SILENTLY IGNORED — the critical gotcha.** The UI JS constructs and sends it, but the
  production API discards it. Measured on one day (2024-05-15, `q=meeting&forms=8-K`):

  | `items` value | hits |
  |---|---|
  | *(omitted)* | 244 |
  | `5.07` | 244 |
  | `9.99` (nonexistent) | 244 |
  | `NONSENSE` (not a number) | 244 |

  Identical `"took"` values confirm the same cached ES response. **Item filtering must be done
  client-side on `_source.items`**, which every hit does carry (EDGAR's authoritative tagging).
- **The 10,000-result cap is a hard error**, not silent truncation: `from=9990` returns
  `search_phase_execution_exception … Result window is too large`. Page size 100, max `from` 9900,
  no scroll cursor — you must window by date and stitch. Rate limit **10 req/s**.
- Under load the endpoint returns **plausible-but-wrong totals** (the agent had to discard a
  rate-limited run that showed differentiated item counts).

**Measured Item 5.07 volume**, full client-side sweep of all 8,682 8-Ks in May 2024:
**1,605 accessions from 1,580 issuers** carry item `5.07` — 18.5% of the month's 8-Ks, about 1 per
issuer. Ground-truthed against `full-index/2024/QTR2/form.idx` (8,685 8-K rows; EFTS returns 8,682).

**A better bulk route than EFTS**: `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip`
(**1.56 GB**, rebuilt daily) carries a per-filing `items` field per CIK
(`"items":["5.07,9.01"]`). No 10k cap, coverage back to 1994. The quarterly `form.idx` files do
**not** carry an items column.

### Financial Statement (and Notes) Data Sets — definitively no vote data

Downloaded and inspected `2025q2.zip` (79 MB). `sub.txt` form values: 10-Q 5,224 · 10-K 624 ·
20-F 538 · 6-K 86 · **8-K 13 + 8-K/A 1**. Those 8-Ks are a trap — they appear only because they
carried XBRL-tagged recast financials (`0001558370-25-006838` has `Assets`, `Goodwill`,
`EarningsPerShareBasic`). Zero vote content.

Scanning the whole quarter's `tag.txt` for vote concepts yields only share *voting rights*
(`CommonStockVotePerShare`, `PreferredStockVotesPerShare`, …) and proxy-fight *expenses*
(`ProxySolicitationCosts`, `ContestedProxyAndRelatedMattersNet`, `ShareholderActivismCosts`).
**No meeting-outcome data of any kind.**

**No SEC dataset for Item 5.07 exists**, and the root cause is measurable: Apple's 2025 annual-meeting
8-K (`0001140361-25-005876`) carries **21 XBRL facts, all `dei:` cover-page tags**. The
For/Against/Abstained/Broker-Non-Vote figures are free-form HTML cells. There is nothing for DERA to
extract. The only SEC-published structured route is **metadata identification** — which filings carry
5.07 — never the tallies.

### Form N-PX — schema detail

Authoritative field list from the XSD (`edgar-form-n-px-xml-technical-specification-31.zip`,
`eis_NPX_PROXY_VOTING_RECORD.xsd`): `issuerName` · `cusip` · `isin` · `figi` · `meetingDate` ·
`voteDescription` · `voteCategories/voteCategory/categoryType` · `voteSource` · `sharesVoted` ·
`sharesOnLoan` · `vote/voteRecord/(howVoted, sharesVoted, managementRecommendation)` ·
`voteManager/otherManagers` · `voteSeries` · `voteOtherInfo`.

Enumerations: `howVoted` in **FOR · AGAINST · ABSTAIN · WITHHOLD**; `managementRecommendation` in
FOR · AGAINST · NONE; `voteSource` in ISSUER · SECURITY HOLDER · N/A; `categoryType` has **14** values.

**There is no proposal number** (only free-text `voteDescription`), **no broker-non-vote concept**,
**no shares-outstanding denominator**, and **no total-votes-cast**.

Four measured reasons aggregation cannot reconstruct a tally:
1. **13F managers report say-on-pay ONLY** (Rule 14Ad-1). For director elections, auditor
   ratification or shareholder proposals the universe collapses to registered funds.
2. Registered funds hold **~33% of US corporate equity** (ICI 2024 Fact Book).
3. **~51% of N-PX filings contain zero vote records** — 2025 season: 5,542 notice reports vs 5,351
   voting reports.
4. **The largest filers are the empty ones.** Vanguard Group's own N-PX (`0001104659-25-083826`) has
   only `primary_doc.xml`, `<reportType>INSTITUTIONAL MANAGER NOTICE REPORT</reportType>` with
   `<noticeExplanation>ALL VOTES BY OTHER PERSONS</noticeExplanation>`. Any aggregation must resolve
   the `otherManagers`/`voteManager` graph or it will double-count and under-count simultaneously.

Scale contrast: **2,684 N-PX filings in 2025 mention Apple's CUSIP `037833100`** — that is the
parsing bill to *partially* reconstruct one meeting. Apple's own 8-K gives the complete tally in one
document.

**No SEC bulk N-PX download exists** (the EDGAR APIs page offers only `submissions.zip` and
`companyfacts.zip`). Harvest off `full-index` (10,874 N-PX in 2025 Q3; the Aug 31 deadline
concentrates them). **Parsing trap: the vote-table filename is not standardized** —
`ProxyVotingTable.xml`, `proxytable.xml`, `BRDWLB_0001086364_2025.xml` all observed. The stable
identifier is `<TYPE>PROXY VOTING RECORD` in the submission `.txt`.

### Third-party datasets

**Dead**: ProxyDemocracy.org (fund-level, 2003–2017; domain parked then TLS-dead; no dump, no GitHub
mirror). Si2 (shut down; `siinstitute.org` now 301s to `blancomuseum.com` — cite Wayback only; its
database survives free at the **US SIF Proxy Proposal Archive**, 2010–2024, 7,200+ proposals with
outcomes, **web UI only, no export**). ICCR EthVest (site search returns 0).

**Paid, issuer-level tallies**: **ISS Voting Analytics** is the de-facto standard — Russell 3000 from
2003; via WRDS as `iss_va_vote_us.vavoteresults` (**2002–2024**) with
`votedfor`/`votedagainst`/`votedabstain`/`votedwithheld` **plus a `base` field encoding the counting
rule** (F+A / F+A+AB / Outstanding / Capital Represented) — get `base` wrong and every support
percentage is wrong. FactSet (meetings from 2005, proposals from 2008, ~46,000 proposal results/yr).
Diligent Market Intelligence (ex-Proxy Insight/Insightia). ESGAUGE (6,000+ issuers, ~10 yrs, API).
Ideagen Audit Analytics (auditor ratification, all US issuers since 2010). Equilar (say-on-pay from
2011). **sec-api.io is the only vendor publishing prices** ($49/mo personal, $199/mo business) and
**explicitly does not parse the tallies** — its own docs say analysts "must manually parse Items
5.02, 5.03, 5.07"; its structured N-PX API is **2024 onward only**. Glass Lewis notably does *not*
sell historical issuer tallies (its products are client-vote-centric).

**Free, issuer-level, with export — exactly one**: **ProxyMonitor.org** (Manhattan Institute),
verified live and current (relaunched 2025-03-25, `dateModified` 2025-10-28). Shareholder proposals
at the Fortune 250 **2006–2024**, say-on-pay from 2011, extended to the full S&P 500 from 2025;
includes vote totals, flags the abstention-counting rule, **CSV export**. Limitation: 250–500 large
caps, shareholder proposals + say-on-pay only, not all management proposals. The old
`Forms/Findings.aspx` endpoint now 404s, so legacy scrapers are broken.

**Open datasets: essentially nothing.** Harvard Dataverse — one fund-level hit
(`doi:10.7910/DVN/SVZWLD`); `"shareholder proposal"` returns zero. Zenodo zero. Hugging Face zero.
**GitHub has no 8-K Item 5.07 vote parser or dataset at all**; the only N-PX repos are three tiny
unlicensed ones (`Marlowe97/fundscrape` 1 star last pushed 2020, `jacob187/npx-aggregator` 0 stars,
`Kamran06/proxy-voting-analysis`).

### Bottom line on granularity

Three objects are routinely conflated:
- **The issuer's official tally** (8-K Item 5.07) — free, complete, machine-*identifiable* via EDGAR
  item tags, but the numbers are unstructured HTML and **nobody has published a parse of them**.
- **Vendor-normalized percent support** (ISS/FactSet/Diligent) — and even they describe it as
  *issuer-disclosed*, not reconstructed, so broker non-votes and the abstention `base` rule are the
  fields most likely to be missing.
- **The fund-level ballot** (N-PX) — genuinely structured XML from 2024, but ~51% of filings are
  empty, 13F managers file say-on-pay only, funds hold ~33% of US equity, and there is no bulk
  download.

---

## Follow-up Research 2 — `def14a_llm` correctness + 8-K 5.07 parse feasibility (2026-09-01 18:20)

Two questions the first pass left open: (a) is the LLM path's poor fill a prompt, schema or carve
problem, and (b) does 8-K Item 5.07 need an LLM at all. Evidence base: **193 real DEF 14A documents**
re-fetched and re-carved with the repo's own `html_to_text` / `prepare_def14a_sections`, and **60
8-K 5.07 filings with 690 hand-read vote rows**.

### ⚠ Correction to this report: the say-on-pay floor is destroying correct data

`SAY_ON_PAY_MIN_SUPPORT = 0.50` in `drop_implausible_def14a` nulls **61 correct rows**, and the three
counter-examples its docstring cites as proof of extraction error are all **real disclosures**:

- JPM 2023 — *"the **31% support** we received for last year's say-on-pay resolution"*
- INTC 2023 — *"received **only 34% support**"*
- SPG 2024 — *"**11.1% of the votes cast** favored our Say-on-Pay"*

**14 of 14** sampled sub-0.50 values are correct. These are shareholder revolts — the
highest-signal events in the table — and the guard deletes exactly them. The Test Coverage table
above has been corrected accordingly.

### The bottleneck is the CARVE, not the LLM

Section **hit rates are near-perfect** (DIRECTOR / COMP / OWNERSHIP 100%, AUDITOR 98%, SAY ON PAY
96%, PAY RATIO 80%) — the anchors find their section. **Recall is what fails** (PAY RATIO 82%,
MEDIAN PAY 82%, SAY ON PAY 88%), for three measured mechanical reasons:

1. **`_find_content_section` applies the 5% TOC skip only on the anchor-fallback path.** NEM, ROK and
   SYK carves landed at 3.0% / 2.0% / 3.7% of the document while the pay ratio sat at 49–92%.
2. **`AUDITOR FEES` uses `last_occurrence=True`** and lands *past* the table — COP at 95% of the
   document; CINF's fee table starts 1,742 chars *before* the carve begins.
3. **Every section sits at its char cap.** EG 2018's say-on-pay result is **29 chars** past the carve
   end; HSIC's pay ratio **229 chars** past.

Consequence for the comp work: **`n_neos = 1` on 21.7% of 2012+ rows** because the 7,000-char
`EXECUTIVE COMPENSATION` budget cannot hold a 5-NEO × 3-year SCT. Widening that budget is a
prerequisite for the exec-comp flatten, not an optimisation.

### A fetch bug: 401 rows extracted from an EDGAR directory listing

`submissions/CIK*.json` has `primaryDocument = ""` for pre-2001 filings, so `_doc_url()` builds a
bare **directory URL** and the LLM was handed a folder index. **401 of 422 pre-2001 rows are fully
NULL, vs 18 of 8,245 after.** The carve works fine on those filings once the right bytes are fetched.
(Consistent with §15's finding that `filing.document.url` does not resolve for filings ≤ 2000.)

### Per-field correctness (8 populated + 8 NULL per field, opened against the filing)

| field | populated correct | populated wrong | NULL = missed | NULL = genuinely absent |
|---|---|---|---|---|
| `ceo_pay_ratio` | 8/8 | 0 | **8/8** | 0 |
| `median_employee_pay` | 8/8 | 0 | 7/8 | 0 |
| `say_on_pay_support_pct` | 8/8 (+14/14 on the sub-0.50 tail) | 0 | 6/8 | 2 |
| `insider_ownership_pct` | 6/8 | 2 (read a dual-class *voting-power* column) | 0/8 | **8/8 (all `*`)** |
| `ceo_ownership_pct` | 8/8 | 0 | 0/8 | 6/8 (`*`) |
| `auditor_fees` | 8/8 | 0 | 7/8 | 0 |
| `lead_independent_director` | 8/8 | 0 | 4/8 | 4 |
| `n_technology_directors` | 1/8 verifiable | 1 wrong by 7× | n/a | n/a |

**When the LLM populates a field it is almost always right; the losses are recall.** Two exceptions:
`insider_ownership_pct` reads a dual-class voting-power column instead of the economic stake, and
`n_technology_directors` is an opinion, not an extraction (see below).

Year coverage confirms the regulatory reading: **pay-ratio 0% before 2018 is CORRECT** (86–94%
in-era), so the "44.9% filled" headline is an artefact of averaging across an era where the
disclosure did not exist. Say-on-pay plateaus flat at 67–74% for 14 years — that is the carve
ceiling, not a format era. Falling ownership fill tracks real mega-cap insider stakes dropping below
the `*` threshold the schema tells the model to null.

### Schema gaps, ranked by cost to close

"Already carved" = the value is inside the text the LLM is *currently* being shown, so adding the
field costs a schema line and nothing else.

| item | present in doc | already carved |
|---|---|---|
| **auditor NAME** | 98% | **89% — free win** |
| **audit fee breakdown** (audit / audit-related / tax / other × current/prior) | 93% | **91% — free win** |
| committee memberships | 99% | 83% |
| per-holder ownership rows | ~100% | 95–97% |
| auditor tenure / since-year | 78% | 64% |
| **director compensation** | 92% | **21% — needs a new carve** |
| director attendance | 86% | 26% |
| say-on-pay frequency | 24% | 46% |
| board meeting counts | 40% | 23% |
| equity plan reserves | 23% | 12% |

Also absent from the contract: the SCT **Change in Pension Value** column (the missing seventh
component identified in §8's residual test), director-since-year, clawback/hedging policy, vote
counts, the PvP table, peer group, burn rate.

### Units

All `_pct` columns are consistent 0–1 fractions (zero values > 1.0). `ceo_total_comp = 1` rows are
legitimate $1-salary CEOs, not errors. The pay-ratio identity `total / median = ratio` holds on
**95.0% of 3,838 rows**; only 2 of the 32 values > 2,000 are wrong (GOOGL 2018/19 stored median pay
in the ratio field).

**One real 1000× bug**: `auditor_fees` — 8 of the 10 smallest values ignore the table's
"(in thousands)" / "($ in millions)" header (MS 57.6 = $57.6M; TSLA 10,919 = $10.9M *and* the wrong
year column). 27 rows are < $500k since 2005.

### The `directors[]` array is the most trustworthy thing in the table

**99.74% of names appear verbatim in the source** (no hallucination); 93% of ages and 98% of tenures
confirmable; a full hand-check of HUBB 2022 was **27/27 correct**, including public-vs-private board
judgements. Caveats for any downstream board-composition feature:

- `gender` is 97.9% filled but **only 17.4% of proxies disclose it** — it is a first-name inference.
- `is_independent` is 78% filled, 86% concordant with the filing's own independent-director count.
- **`avg_other_public_boards` is biased upward**: 37.2% of filings mix null and 0, and the nulls are
  systematically the zero-board directors.
- `n_technology_directors` / `pct_technology_directors` are **opinions** — mean |Δ| of 1.06 directors
  between consecutive filings (only 38.8% unchanged), and wrong by 7× on HUBB 2022, where the
  filing's own matrix states "Cybersecurity and Technology 78%" of 9 directors.
- `majority_voting` flips **21.2% year-over-year** on a bylaw that does not change.
- `poison_pill` is TRUE in 0.1% of rows — degenerate.

### 8-K Item 5.07: regex vs LLM, measured

**The stored substrate is usable — no HTML re-fetch needed.** `item_text` contains zero HTML tags
(0/5,984 rows) but is edgartools' *rendered* view and **column alignment survives**: 84.6% carry a
`U+2500` rule under the header, the rest are whitespace-aligned, one filing row per text line.

Two upstream defects no regex can fix: the renderer **drops the header row** on 21 of 60 sampled
filings, and item slicing **truncates** — HWM 2019 stores 508 chars of narrative where the live
filing has 10,922 chars and four tables, because the filer numbered proposals `Item 1.` / `Item 2.`
and the splitter cut there (62 rows, 1.0%, corpus-wide).

60 filings / 690 hand-read rows:

| | deterministic parser | gpt-4o-mini |
|---|---|---|
| rows fully correct | 82.8% | 82.5% |
| **silently wrong** | **62 (9.0%)** | **81** |
| missed | 39 | **1** |
| spurious / fabricated | 7 | 4 |
| precision / recall | 0.868 / 0.828 | 0.821 / 0.825 |
| **filings fully clean** | **40%** | **65%** |
| cost | free | **$0.00063/filing → $3.91 for all 6,251** |

Parser-breaking layouts, by filings hit: dropped header → positional guess (21 attempted, 5 wrong —
DVN 2026 mislabels broker-non-vote as abstain on all 11 nominees); a year in a proposal title
absorbed as `for` (6); scrambled multi-line headers producing a **column permutation** (AVY, HBAN,
ADI 2016, FDS, DVN); narrative years/quorum/zip read as vote rows (5); line-wrapped prose (4);
vertical `For: 1,234` layout — **100% miss** (4); sub-1,000 cells without a comma shifting a column.

LLM-specific failures: it **fabricated an entire table** for the truncated HWM 2019 — "John Doe" /
"Jane Smith", 250,000,000 votes — plus a fake row for BSX; 3 silent digit corruptions
(`23,859,241 → 2,385,941`); fractional votes truncated 1000× (CVNA); and a systematic,
prompt-fixable `broker_non_votes → votes_withheld` mislabel on ~62 rows.

**No validation gate is available, unlike compensation.** A vote table prints **no independent
total**, and the dominant parser error is a *column permutation*, which is **invariant under sums**.
Measured: (a) total ≤ shares outstanding is computable on only 9/57 (16%); (b) per-nominee totals
computable 96%, hold 91%; (c) meeting-level totals computable 100%, hold 81%. Combined gate:
**recall 0.56 / precision 0.56 — 7 of 16 known-bad filings pass clean.**

What substitutes for a gate: the two extractors fail on **disjoint** inputs (parser dies on layout,
LLM on arithmetic and on empty input). Every one of the 62 silent parser errors and 81 silent LLM
errors falls inside the disagreement set; on the 26 filings where the parser was flawless the two
agreed on 291/319 rows.

**Amendments: "latest wins" would destroy data.** 190 multi-filing meetings on
`(ticker, period_of_report)`. In **135 (71%) the amendment carries no vote numbers at all**, so
"latest wins" is correct on only 33/190 (17%). **"Union the group" is correct on 173/190 (91%).** Of
the 17 genuine restatements, 8 resolve on wording (explicit preliminary→final pairs — BBY 2012 is
textbook — or only the later saying "final"/"certified"); **9 of 190 (4.7%) have no signal beyond
the form type**.

Concurrency note: `sec_8k` item 5.07 grew **5,965 → 6,251 rows during the measurement session**, the
same live-backfill caveat that applies to `sec_def14a`.

---

## Follow-up Research 3 — how to fix the section carve (2026-09-01 19:30)

Question: the `def14a_llm` recall failures were traced to the carve (follow-up 2). Is the fix a
bigger budget, a better anchor, or a different substrate? Measured head-to-head on **25 DEF 14A
filings** (20 from 2025-26 including every named failure case, 5 from 2011-2016), with ground truth
established by opening each document and recording each target's character offset. **8,628 tables
parsed; zero contain a nested `<table>`.** No LLM calls — this is purely about which text is selected.

Three strategies:
- **A — current**: the repo's own `prepare_def14a_sections`, imported not reimplemented.
  Anchor + fixed char budget.
- **B — table-anchored**: parse the HTML, enumerate every `<table>`, build a cell grid, classify by
  header signature, serialize the winning table as TSV.
- **C — boundary-aware narrative**: carve from anchor to the next section boundary instead of a
  fixed budget, TOC skip applied on all paths.

### Recall (of 25)

| target | GT exists | A | B |
|---|---|---|---|
| Summary Compensation Table | 24 | 20 | **25** |
| **Director compensation table** | 25 | **2** | **24** |
| Insider / group ownership | 24 | 22 | 22 |
| ≥5% holders | 22 | 21 | **25** |
| Audit fee table | 23 | 20 | **25** |

**B = 121/125 (96.8%)**, 1 false positive (MS: picked page 2 of a paginated table). B never picked a
semantically different table.

**The director-comp table is A's structural blind spot**, and the reason is geometric: it sits at
**23–36% of the document**, in the gap between A's `DIRECTOR NOMINEES` window (ends ~17–21%) and
`EXECUTIVE COMPENSATION` (starts ~42–64%). Median miss distance **70,761 chars**. It was never
reachable by widening either window.

Narrative targets, A vs C: auditor name 25 → **20**; pay ratio 9 → **11**; median pay 13 → **14**;
say-on-pay 11 → 11. **C is a net loss (−2) and is rejected.**

### Payload — table-anchoring makes the input SMALLER

| strategy | mean | median | max |
|---|---|---|---|
| A | 50,300 | 51,000 | 51,000 |
| B | 4,283 | 3,786 | 14,169 |
| C | 6,261 | 6,127 | 11,000 |
| **B + C** | **10,544** | **9,334** | 21,603 |

B+C is **21% of A**. A hits its cap on every section of every filing. Real SCT text averages
**1,686 chars** (1,886 as TSV) — about **25%** of the 7k window A spends to sometimes reach it.

B+C does not cover bios or governance, so the realistic hybrid (B + narrative + A's 20k bios + 6k
governance) is **mean 36,544 = 73% of today's payload**, with materially better recall.

### SCT rows — the `n_neos=1` diagnosis confirmed

354 NEO×year rows exist across the 24 in-table SCTs. **B captures 354 (100%).** A holds the whole
SCT in 20/24 and *none* of it in 4 (WMT, TSLA, T-2011, A-2016). Critically, when A's anchor lands,
7k **never truncates** (`frac_in_A = 1.00` in all 20 hits). So the 17% total-miss rate — which
matches the DB's 21.7% `n_neos = 1` — is an **anchor failure, not budget truncation**. Widening the
budget would not have fixed it.

### Anchor vs budget, quantified

Only **5 of 25** narrative misses are within ~3,000 chars of the slice end (HSIC +123, PFE +190,
WMT +430, PFE-median +206, XOM say-on-pay +2,057). **Everything else is 10k–500k chars away.**
Raising budgets buys exactly those five.

Two anchor defects, both bidirectional:
- **The 5% TOC floor is itself the killer in some cases** — A-2016's say-on-pay result sits at
  **4.2%** of the document. Applying the TOC skip universally would make this case worse.
- **`last_occurrence=True` overshoots in both directions** — it puts WMT's auditor carve 3,066 chars
  *before* the fee table, and HUBB / ROK **149k / 165k** chars away.

### Two prerequisites, quantified over 8,580 ground-truth cells

- **`<br>` / block-boundary must emit a separator.** Dropping it fuses **180 cells (2.1%) across
  16 of 25 filings** (`All othercompensation`, `James DimonChairman and CEO`). This is load-bearing
  for **classification**, not just for values: **JPM's SCT was invisible to the classifier** until
  block `<div>` boundaries emitted a separator. Same root cause as §6's Agilent 1.0e19.
- **`<sup>` must be stripped.** 47 cells affected (0.5%, 5 filings), **7 of them the bare-digit form
  that corrupts a value** (`$1,587,852` + `⁶` → `1,5878526`). Same root cause as §11's PG 10×.

### Header signatures must be strict — the naive version fails badly

Scored against the same ground truth, a naive signature gets: SCT 25/25 (but admits 2–5 candidates
in **8/25** — JPM's CD&A "Annual compensation" table passes and only loses on a row-count
tie-break), director comp 24/25, insider ownership **14/25**, 5% holders **3/25**, audit fees
**2/25 (23 wrong picks)**.

The three fixes that reach 96.8%:
1. Require ≥1 SEC-mandated SCT column **and reject "compensation actually paid"** (which is the PvP
   table, not the SCT).
2. Match **exact** fee-category row labels rather than substring hits.
3. **Split ownership into two sub-targets** — the insider table often has **no percent column**, so
   one signature cannot serve both.

### B's residual failures — 5 real defects in 125

| cause | n | note |
|---|---|---|
| Disclosure genuinely absent | 4 | not a defect |
| Header carries no signature | 2 | MMM-2011, T-2011 ("Stock"/"Total", no %, group total in a footnote) |
| **Page is a JPG** | 1 | XOM — text hidden in a white-on-white `<p>`; unreachable by any text method |
| **Numbers live in .gif charts** | 1 | SYK fees — absent from the HTML entirely; A cannot reach it either |
| Table paginated across two `<table>`s | 1 | MS — loses 15 of 19 holders |
| Table with no numbers | 1 | TSLA director fees all em-dash |
| Prose, not a table | 1 | T-2011 fees |

### Recommendation

**A router, not a single strategy:**

| target | strategy |
|---|---|
| SCT, director comp, ≥5% holders, audit fee table | **B** |
| Insider / group ownership | **B**, with A's `SECURITY OWNERSHIP` window as fallback |
| Summary Comp Table on an image page | prose fallback (XOM) |
| Director bios, corporate governance | **keep A** — B does not cover them |
| Auditor **name** | anchor the narrative slice on **B's fee table** (the firm is often cell `[0][0]`) |
| Pay ratio, say-on-pay | **keep A, fix the anchor** — not the budget; lower or drop the 5% floor for say-on-pay |

**Do not adopt C.** Its only sound element is applying the TOC skip on paths that currently lack it.

Combined measured outcome: payload **mean 36,544 (73% of today)** with recall **25/25 SCT, 24/25
director comp, 25/25 5% holders, 25/25 fee table, 24/25 insider ownership**.

---

## Consolidated conclusions — read this first when planning

The five `sec_def14a*` tables are sparse for three independent reasons (§1, §2, §7-§12): a backfill
that is **26/500 tickers** complete, a **regulatory cliff at the 2023 filing season** below which the
ECD XBRL columns cannot exist, and an edgartools HTML parser that is **silently wrong rather than
absent**. Only the third is a defect to fix in parsing.

### What to keep, what to retire

- **KEEP the ECD XBRL reader** in `fetch_def14a_edgar.py` — these are filer-tagged machine-readable
  facts and an LLM would be strictly worse. **But fix its dimension bug** (§4): `_get_concept_value`
  ignores `dim_ecd_IndividualAxis`, so co-PEO years silently drop a PEO (BA drops Calhoun's
  −23.9M CAP) and zero-matrix filers return 0.0 (SBUX, three consecutive years).
- **RETIRE the HTML-parsed block** — exec comp, director comp, ownership, audit fees, board
  recommendations. Every one is measured broken (§6–§12).
- **MOVE `ceo_pay_ratio` to the LLM path** — it is narrative-only forever and edgartools extracts it
  with three value-inventing repairs (§12, `html_extractor.py:430-457`).

### Executive compensation — already 92% extracted, never flattened

**34,741 per-executive rows across 497 tickers** sit unqueryable inside `def14a_llm.def14a_json`
(keys are `_usd`-suffixed). Against `sec_def14a_executive_comp`'s 2,378 rows on 25 tickers: title
100% vs 45.4%, stock awards 93.8% vs 45.4%, option awards 88.0% vs 27.5%, and **2 rows > $1e9
vs 109**.

Three steps: **flatten** (free — the tokens are already paid for); **add `pension_change_usd`** to
`ExecutiveCompensation` (the residual is **positive on 97.5%** of non-reconciling rows, median
**$319,367** — the signature of a missing column, not misread values, and the post-2006 SCT has
seven components where the schema models six); **gate on `sum(components) == total`** within $1.
For 2023+, `ecd:PeoTotalCompAmt` is an independent filer-tagged check on the CEO row.

Prerequisite: the 7,000-char comp budget causes **`n_neos = 1` on 21.7% of 2012+ rows**. Fix the
carve (follow-up 3) before re-running.

### Director compensation — the genuine gap

Absent from the Pydantic contract entirely; the 87,984 `directors[]` rows are **biographical only**.
The edgar path's 3,119 rows cover 25 tickers with components that are a **measured parser miss —
0 of 4 cases opened were genuinely absent from the source** (§7). Needs a new model + B-carving.
Exists only from the **2008 proxy season** (Reg S-K 2006).

### Votes — two workstreams, not one

- **Board recommendation** (`sec_def14a_votes`): 39.6% NULL, text present in **100%** of the misses,
  and in **4/4 failures opened the recommendation is in a TABLE** the text-regex extractor never
  reads. An unknown share of the 890 `FOR` values are **fabricated** by a pattern that matches the
  proxy card's column header (§9). Ceiling: Apple's cell is a JPEG with `alt=""`.
- **Vote tallies** (8-K Item 5.07): **6,251 rows already stored, `item_text` usable as-is with column
  alignment intact — no HTML re-fetch needed.** Run the deterministic parser and the LLM together and
  auto-accept agreement: they are tied on rows (82.8% vs 82.5%) but fail on **disjoint inputs**, and
  **no validation gate exists** because a vote table prints no independent total and the dominant
  error is a column permutation that is invariant under sums (measured gate: recall 0.56 /
  precision 0.56). Cost **$3.91** for the corpus. Two hard rules: **emit nothing when `item_text`
  has no comma-grouped number** (the fabrication trap — "John Doe" / 250,000,000 votes on a
  truncated filing), and **never "latest wins" on amendments** (correct 17% of the time; "union the
  group" is 91%).

### Ownership — do not rebuild

Redundant with structured sources already available: institutional positions ← 13F; 5%-holder
positions ← SC 13D/G; insider holdings ← Forms 3/4/5. The proxy-only field is the
directors-and-officers **group aggregate**, which the LLM already extracts and the cube already
consumes. Work worth doing: lift `insider_ownership_pct` from 57.3% fill — not reconstruct the
table.

### Free wins

**`auditor_name`** (98% present, **89% already inside today's carve**) and the **audit fee
breakdown** (93% / 91%) cost one schema line each. `auditor_name` is currently the worst column in
`sec_def14a` at **2.05%**, because edgartools returns `''` on failure (§12).

### Bugs to fix regardless of architecture

1. **The say-on-pay 0.50 floor deletes 61 correct rows** and its docstring's three counter-examples
   are all real disclosures (JPM 2023 genuinely received 31% support). Highest-signal events in the
   table.
2. **401 rows were LLM-extracted from an EDGAR directory-listing page** — pre-2001 filings have
   `primaryDocument = ""`, so `_doc_url()` builds a bare directory URL.
3. **`auditor_fees` 1000× scale error** on 8 of its 10 smallest values.
4. **`company_name` has no fallback to `filing.company`** — a one-line fix worth the whole 14% fill
   rate (§3).

### Do not trust downstream

`n_technology_directors` (mean |Δ| 1.06 directors between consecutive filings; wrong by 7× on HUBB
2022), `gender` (a first-name inference — only 17.4% of proxies disclose it),
`avg_other_public_boards` (biased upward; nulls are systematically the zero-board directors),
`majority_voting` (flips 21.2% year-over-year on a bylaw that does not change), `poison_pill`
(TRUE in 0.1% of rows).

### Sequencing constraint

`existing_filings` dedups on accession, so a re-fetch after a parser fix requires TRUNCATE + refetch,
not an incremental run. Flattening the existing JSON is free and independent — it can land first.
