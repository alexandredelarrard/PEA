# Research: extracting fundamentals from Sharadar as a new source of truth

**Date**: 2026-08-26
**Research Phase**: 1 of 3 (FIC workflow)
**Next Phase**: Planning (`/plan`)
**Spec**: [specs/2026-08-26/leverage-shadar-v1.md](../../../specs/2026-08-26/leverage-shadar-v1.md)
**Branch at time of research**: `bugfix/cluster_876ab8a57bd8`

---

## Research Question

Can Sharadar Core US Fundamentals replace the SEC/edgartools extraction as the primary fundamentals
substrate — consistently, for the S&P 500 today and the Russell 1000 soon, with history back to
~1999 — and how should a `fundamentals_shardar` step be organised so that `fundamentals_history`
becomes "Sharadar first, our SEC extraction second"?

Three corrections carried from the spec: the vendor spells itself **Sharadar**; the schema file is
`sql/schema.sql`, not `schemas.sql`; and the SEC fundamentals code lives at
`src/data_extract/utils/fundamentals/`, not `src/data_extract/fundamentals`.

---

## Summary — the seven findings that decide the plan

1. **⚠ The licence is a gating decision, not a detail.** Sharadar Direct ($19/mo entry) is sold under
   [Personal Use License Terms](https://sharadar.com/terms) whose §2 forbids use *"for professional,
   commercial, institutional, or organizational purposes of any kind — including for yourself as a
   professional, for an employer, client, ... firm, fund ... Prohibited uses include, without
   limitation, professional research, analysis, trading or investing on behalf of others,
   **consulting**, ... technology development for a business"*. §3 forbids entities outright and
   forbids using an individual key to obtain data for use *within* an entity. §8 forbids publishing
   conclusions about the data without written approval. §10 requires deletion within 30 days of
   termination (derived models and backtest results may be kept). The FAQ confirms Personal Use
   *does* cover an individual's own research, backtesting and automated trading of their own account
   with no external clients or money managed for others. The documented institutional route is the
   **Nasdaq Data Link** channel, whose SF1 pricing could not be read (JS-rendered behind Incapsula,
   unreadable live and via Wayback). **This needs deciding before a plan is written.**

2. **Sharadar zero-fills missing values; the spec requires NULL.** 41 of 112 SF1 indicators carry the
   sentence *"Where this item is not contained on the company consolidated financial statements and
   cannot otherwise be imputed **the value of 0 is used**."* — including `capex`, `cor`, `revenue`,
   `debt`, `debtc`, `debtnc`, `depamor`, `sgna`, `intexp`, `inventory`, `receivables`, `deposits`,
   `cashneq`, `dps`, `divyield`, `accoci`. So "not reported" and "genuinely zero" are
   indistinguishable. That is the exact inverse of this repo's contract, where every null carries a
   `fundamentals_reason_codes` row and `unexplained_null` is a **zero-ceiling critical** check.

3. **Acceptance check #3 (Q4 = FY − 9M) becomes tautological on Sharadar.** SF1's ARQ does carry a Q4
   row whose `datekey` is the 10-K filing date — but a 10-K contains no standalone Q4 income
   statement, so that row is *constructed* at 10-K time. Sharadar's only documented period-arithmetic
   method is annual-minus-the-reported-quarters, shown in the legacy Quandl doc producing a
   **negative quarter** (ABT 2011 Q4 revenue −$7.1bn) with the comment *"the negative quarter has
   been calculated by Sharadar ... in order to ensure that the quarterly and annual financials are
   aligned"*. This repo already learned that lesson on its own data —
   [tier3_internal.py:86](../../../src/validate/fundamentals/checks/tier3_internal.py#L86):
   *"Where our Q4 was DERIVED as `FY - YTD9`, footing the four quarters back to FY is an identity: it
   passes on any numbers at all, including wrong ones."* Sharadar also documents that
   **ART ≠ the sum of the prior four ARQ**.

4. **Acceptance check #2 (no kicks from a changed tag/definition) cannot be ported at all, and
   Sharadar manufactures kinks by design.** SF1 ships **no provenance columns** — no source concept,
   no resolution method, no linkbase, no accession, no role URI. The repo's two detectors for exactly
   this, `tag_switch_break` and `basis_step`, both fire on *"a level step at the exact boundary where
   `source_concept` / `resolution_method` changed"*
   ([tier2_series.py:249](../../../src/validate/fundamentals/checks/tier2_series.py#L249),
   [:272](../../../src/validate/fundamentals/checks/tier2_series.py#L272)) and go dark. Meanwhile
   Sharadar deliberately dumps a whole year's restatement into one MRQ quarter, and three of its
   metric definitions (`invcap`, `invcapavg`, `roic`) carry *"Please note this calculation method is
   subject to change."*

5. **Sharadar disagrees with this repo on eight of its most load-bearing definitions** — not by
   error, by design (§3.3). The sharpest: Sharadar's `revenue` for financial institutions is *"net of
   interest expense **and provision for credit losses**"*, which is precisely the basis this repo
   *bans* with `never_use` entries after measuring MTB 110 rows ~32% understated and AXP 91 rows.
   Also: Sharadar `ebitda` is bottom-up (`netinc + taxexp + intexp + depamor`), the repo's is
   top-down (`operatingIncome + depAmort`); Sharadar `equity` is parent-only, the repo's is incl-NCI;
   Sharadar bundles goodwill and other intangibles into one `intangibles` column that the repo
   splits; and Sharadar's `de` is *liabilities*/equity despite being named "Debt to Equity Ratio".

6. **16 of the repo's 60 value columns have no Sharadar equivalent**, 12 of them read downstream. All
   six regime top-line legs are gone (`premiumsEarned`, `netInterestIncome`, `noninterestIncome`,
   `netInvestmentIncome`, `realizedInvestmentGains`, `rentalIncome`), as are `goodwill` /
   `intangiblesExGoodwill` (bundled), `ppeGross` / `accumulatedDepreciation`, `minorityInterest`,
   both lease liabilities, and `employees`. **But Sharadar also revives eight cube inputs that are
   dead today** (`deferredrev`, `deposits`, `ncfdiv`, `ncfcommon`, `accoci`, `netincnci`, `netincdis`,
   `marketcap`). Net column count is roughly flat; the *composition* changes materially. Full map §3.

7. **The codebase is ready, and the DOW-30 PoC harness already exists.**
   [src/validate/external/](../../../src/validate/external/) holds a complete Tier-4
   external-source comparison architecture (`tiingo_comparison.py` 491 lines,
   `yahoo_comparison.py` 395 lines) with an a/b/c field-bucket classification, TTM-vs-discrete-quarter
   alignment and ratio-drift detection — built and capped to `DOW_30_TICKERS` because Tiingo's free
   plan allows nothing else. Sharadar's free tier is also DJIA-only. Adding a table is a 2-file
   minimum; the cube's *hard* contract is only 6 columns.

**One thing worth saying plainly.** The recorded reason Tier 4 was deferred bears directly on
"Sharadar as source of truth"
([src/validate/external/__init__.py](../../../src/validate/external/__init__.py)): Boritz & No (2020)
measure as-reported XBRL matching the filed 10-K to within **0.01%** while **aggregators disagree
with the 10-K at 6.5–7.7%** — but 48–63% of XBRL line items are absent from those aggregators.
Sharadar is an aggregator, and nowhere claims to parse XBRL (§2.7). The trade is *far better
completeness and cross-ticker consistency* for *a measurable step away from the filed statement*.
That is a real trade, and it is yours; this document measures it rather than resolving it.

---

## Research graph

```
                     ┌──────────────────────────────────────┐
                     │  DECISION 0: which channel/licence   │
                     │  Direct (personal use only, $19/mo)  │
                     │  vs Nasdaq Data Link (institutional, │
                     │      price not readable)             │
                     └───────────────┬──────────────────────┘
                                     │
        ┌────────────────────────────┴─────────────────────────────┐
        │                                                          │
   SHARADAR SF1                                            SEC / edgartools
   112 cols; 1 row per                                      (existing, messy)
   (ticker, dimension,                                      fundamentals_facts
    datekey, reportperiod)                                   26 cols, as-filed,
   17,825 tickers                                            per-filing, linkbase-
   (5,529 live / 12,295 delisted)                            resolved, reason-coded
        │                                                          │
        │ ARQ / ARY / ART  (AR = point-in-time, immutable)          │
        │ MRQ / MRY / MRT  = NOT point-in-time, overwritten         │
        ▼                                                          ▼
  ┌──────────────────┐                                    ┌──────────────────┐
  │ fundamentals_    │  NEW TABLE                         │ fundamentals_    │
  │ shardar          │  vendor-shaped, as delivered       │ facts            │
  │ (ticker,dim,     │  + lastupdated for incremental     │ (existing)       │
  │  datekey,        │                                    └────────┬─────────┘
  │  reportperiod)   │                                             │
  └────────┬─────────┘                                             │
           │                                                       │
           │  map + regime-aware basis translation                 │  build_history.py
           │  (§3 field map; §3.3 basis conflicts)                 │  publication-event
           ▼                                                       ▼  replay
        ┌──────────────────────────────────────────────────────────────┐
        │  fundamentals_history  (69 cols, PK ticker+as_of)            │
        │  PRECEDENCE: Sharadar first, SEC second, per (field, cell)   │
        │  + provenance saying WHICH source won      ← §5              │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
                     ┌──────────────┴───────────────┐
                     ▼                              ▼
             data_aggregate (cube)          validate (35 checks)
             hard contract = 6 cols          ~10 portable, 8 dead,
             §6 impact map                   3 tautological   ← §4
```

---

## Part 1 — What Sharadar actually is

### 1.1 Two live channels, with different column names

Sharadar launched a **direct API on 2026-07-27** while keeping the Nasdaq Data Link channel alive
([launch post](https://sharadar.com/blog/posts/sharadar-launches-direct)). They must not be mixed —
[llms.txt](https://sharadar.com/llms.txt): *"This is the direct Sharadar API:
`https://api.sharadar.com/v1.0`. It is not Nasdaq Data Link. Do not call `data.nasdaq.com` or use the
`nasdaqdatalink` / `quandl` libraries with a sharadar.com key."*

| | Sharadar Direct | Nasdaq Data Link |
|---|---|---|
| base | `https://api.sharadar.com/v1.0` | `https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1` |
| table name | `fundamentals` (alias `SF1`) | `SF1` |
| **filing-date column** | **`date`** | **`datekey`** |
| primary key | `(ticker, dimension, date, reportperiod)` | `(ticker, dimension, datekey, reportperiod)` |
| `fiscalperiod` column | present | **absent** |
| filters | `ticker, dimension, calendardate, lastupdated` | `ticker, dimension, calendardate, datekey, reportperiod, lastupdated` |
| Python client | **none published** | `nasdaqdatalink` |
| licence | Personal Use only | institutional, per the launch post |

Sources: <https://sharadar.com/docs/fundamentals>,
`https://api.sharadar.com/v1.0/schema/fundamentals?format=postgres` (header "As of 2026-08-18"),
`https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1/metadata.json`.

**This report uses `datekey` throughout** and flags the rename where it matters. A plan must pick a
channel *first*, because the column name is inside the primary key.

Every `data.nasdaq.com` **web** page (`/databases/SF1`, `/databases/SF1/documentation`) is
JS-rendered behind Incapsula and yields no extractable text, including via Wayback (checked
snapshots 20230608211010 and 20201114165804). The Nasdaq **API** metadata endpoint is open and
unauthenticated and served as the primary source for that channel.

### 1.2 What a Fundamentals subscription includes

The `Included in` field on each table's own doc page is definitive. A **Fundamentals-only**
subscription gets **6 tables plus the dictionary**, not sold separately:

| table | legacy | grain | history | in Fundamentals? |
|---|---|---|---|---|
| `fundamentals` | **SF1** | ticker+dimension+datekey+reportperiod | Jan 1998 | ✅ |
| `tickers` | TICKERS | **table+permaticker+ticker** | Jun 1990 | ✅ |
| `actions` | ACTIONS | date+ticker+name+action | Jan 1998 | ✅ |
| `events` | EVENTS | ticker+date | Jan 2004 | ✅ |
| `sp500` | SP500 | date+ticker+action | Jan 1998 | ✅ |
| `daily` | DAILY | ticker+date | Dec 1998 | ✅ |
| `descriptions` | INDICATORS | table+indicator | n/a | ✅ |
| `stocks` | SEP (prices) | ticker+date | Jan 1998 | ❌ Prices/Bundle |
| `metrics` | METRICS | ticker+date | Sep 1996 | ❌ Prices/Bundle |
| `funds` | SFP | ticker+date | Jan 1998 | ❌ Prices/Bundle |
| `insiders` | SF2 | — | Jan 2008 | ❌ Investors/Bundle |
| `holdings` (+`_ticker`, `_investor`) | SF3/A/B | — | Jun 2013 | ❌ Investors/Bundle |

This matches the spec's intent: prices come from Yahoo, 13F/insiders from SEC — and neither is in the
Fundamentals plan anyway. **`tickers` and `actions` come free with it**, and both are needed
(§1.5). Two bonuses worth carrying into the plan:

- **`sp500`** gives current *and historical* S&P 500 membership — quarter-end full snapshots back to
  1998-03-31 plus an add/remove event log, with *"the effective membership date (not the announcement
  date)"*. The repo's `sp500_tickers` table has **no date column at all**
  ([sql/schema.sql:10](../../../sql/schema.sql#L10)), so this is the first available fix for
  survivorship bias in the universe itself.
- **`events`** is SEC 8-K event codes, overlapping the repo's own `sec_8k` table.

Sources: <https://sharadar.com/docs/fundamentals>, `/docs/tickers`, `/docs/actions`, `/docs/events`,
`/docs/sp500`, `/docs/daily`, `/docs/stocks`, `/docs/metrics`, `/docs/insiders`, `/docs/holdings`,
<https://sharadar.com/subscribe>. Table index: `https://api.sharadar.com/v1.0/schema` returns
`"count":14`.

### 1.3 The dimension system — six values, only three point-in-time

|  | AS REPORTED (immutable) | MOST-RECENT REPORTED (overwritten) |
|---|---|---|
| Annual | `ARY` | `MRY` |
| Quarterly | `ARQ` | `MRQ` |
| Trailing 12M | `ART` | `MRT` |

**AR\*, verbatim** (<https://sharadar.com/docs/fundamentals>): *"excludes restatements / point-in-time
view with data time-indexed to the date the form 10 regulatory filing was submitted to the SEC /
presents data for the latest reporting period at that filing date / **may include multiple
observations in a quarter if more than one filing is made during the quarter** / on limited occassion
may not have any observations in a particular quarter. Sometimes companies are delayed in reporting
for up to 18 months. On such occassions they may report multiple documents on the same date to catch
up, in which case these datasets will only provide date for the most recent reporting period. /
**typically suitable for back-testing**"* (spelling is the source's own).

**MR\*, verbatim**: *"includes restatements / time indexed to the financial/report period / presents
the most recently reported data for that reporting period / typically suitable for assessing business
performance after restatements for mergers/divestitures"*.

**The FAQ is the crispest statement** (<https://sharadar.com/docs/faqs>): *"As-Reported (ARQ, ARY,
ART) is a point-in-time view, time-indexed to the SEC form 10 filing date, and excludes restatements.
**That view is not rewritten when a later filing restates a prior period; the later filing creates a
new observation.** ... Most-Recent Reported (MRQ, MRY, MRT) is time-indexed to the report period and
**is updated when a company restates prior periods.**"*

**⚠ Only AR\* is usable here.** For MR dimensions `datekey` *is* `reportperiod` — measured across all
14 AAPL MRQ rows, `date == reportperiod` in every row. So `WHERE datekey <= D` on MRQ is **not** a
point-in-time filter. And MR\* rows mutate in place, which would violate the repo's
`diff_against_stored` immutability guarantee
([build_history.py:994](../../../src/data_extract/utils/fundamentals/build_history.py#L994)) on every
restatement.

**Three structural coverage facts:**

- **Quarterly dimensions are US-domestic-only** — *"Quarterly (Q): Quarterly observations of
  quarterly duration (**available only for US domestic companies, unavailable for foreign
  companies**)"*. ADR (form 20) and Canadian (form 40) filers have **no ARQ/MRQ at all**, only
  `ARY`/`ART`. `tickers.category` (`Domestic`/`Canadian`/`ADR`) identifies them. For the Russell 1000
  target this is a real hole and it is not fixable from Sharadar. *(Could not be verified empirically
  — the free DJIA sample contains no ADR or Canadian filer. The documented statement is
  unambiguous.)*
- **In ART/MRT, balance-sheet items are the instantaneous period-end value, not an average.**
  Verified on AAPL at `datekey` 2026-07-31: `assets` and `equity` are byte-identical between ARQ and
  ART, while `revenue` is exactly the sum of four trailing ARQ quarters
  (109,417 + 111,184 + 143,756 + 102,466 = 466,823). Separate `assetsavg` / `equityavg` / `invcapavg`
  columns exist for the ratios that need an average. *(Not stated in the docs — established by the
  existence of those columns plus measurement.)*
- **`assetsavg`, `equityavg`, `invcapavg`, `assetturnover` are NULL in ARQ/MRQ**, populated in
  ARY/ART. So `roe`/`roa`/`roic` do not exist quarterly.

**Measured** on 12 DJIA names including banks (JPM, GS) and insurers (TRV, UNH): all six dimensions
present for all 12, no dimension missing for any name. AR row counts exceed MR by exactly one per
window — an artefact of AR being indexed to the later filing date.

### 1.4 The date columns

| column | definition, verbatim from `descriptions` |
|---|---|
| `datekey` / `date` | *"The Date Key represents the SEC filing date for AR dimensions (ARQ;ART;ARY); and the [REPORTPERIOD] for MR dimensions (MRQ;MRT;MRY). In addition; **this is the observation date used for [Price] based data such as [MarketCap]; [Price] and [PE]**."* |
| `reportperiod` | *"The Report Period represents the end date of the fiscal period."* |
| `calendardate` | *"the normalized [ReportPeriod] ... if the report period is 2015-09-26; the calendar date will be 2015-09-30 for quarterly and trailing-twelve-month dimensions (ARQ;MRQ;ART;MRT); and **2015-12-31 for annual dimensions (ARY;MRY)**. We also employ **offsets in order to maximise comparability** ... consider two companies: one with a quarter ending on 2018-07-24; and the other on 2018-06-28. A naive normalization process would assign these to differing calendar quarters ... However; **we assign these both to the 2018-06-30 calendar quarter** because this maximises the overlap in the report periods in question."* |
| `fiscalperiod` | *"expressed as follows: 2024-Q2; 2024-Q3; 2024-FY etc. Note that companies can have different fiscal periods for the report period due to different year end dates."* (Direct-only; `unittype` is mislabelled `date (YYYY-MM-DD)` but values are strings and the schema types it `text`) |
| `lastupdated` | *"the last date that this database entry was updated; which is useful to users when updating their local records"* |

**Note the second sentence of the `datekey` definition.** `marketcap`, `price`, `pe`, `pe1`, `ps1`,
`pb`, `ev`, `sps`, `divyield` inside SF1 are all observed **as of the date key** — so in MR
dimensions those price-based columns are priced as of the *fiscal period end*, not any publication
date. Measured: AAPL 2024-09-28 has `pe` 35.946 in ARQ and 36.948 in MRQ.

**`calendardate` is the answer to the fiscal-alignment problem the spec cares about**, and it is
better than nearest-quarter rounding — the snap maximises report-period overlap, deliberately
assigning a period ending 2018-07-24 *backward* to 2018-06-30. But note the measured consequence for
AAPL (September fiscal year-end):

| dimension | `reportperiod` | `calendardate` |
|---|---|---|
| ARY | 2025-09-27 | **2025-12-31** |
| ARQ | 2025-09-27 | **2025-09-30** |

**The same fiscal period carries two different `calendardate` values depending on dimension.**
`calendardate` is *not* in the primary key, and because AR admits multiple filings per quarter,
`(ticker, dimension, calendardate)` is **not unique**. `fiscalperiod` disagrees with `calendardate`
too (AAPL's period ending 2023-12-30 is `fiscalperiod=2024-Q1`, `calendardate=2023-12-31`). The
three must not be used interchangeably.

**⚠ Amended filings (10-K/A, 10-Q/A) are entirely undocumented.** Grepped all 17 doc pages, all
product pages, `llms.txt` and the full `descriptions` dictionary for `amend`, `10-K/A`, `/A`: the only
hits are unrelated 8-K event codes. What *can* be said from the documented model: the AR rule *"the
later filing creates a new observation"* plus the four-column PK means an amendment filed on a new
date is a new row; an amendment filed on the *same* date for the same reportperiod and dimension
would collide on the PK and could only be an overwrite. The nearest documented analogue is the
delayed-filer rule — same-date multiple filings are collapsed to the most recent reporting period.
**How `lastupdated` behaves on an amendment is also undocumented.** This matters because the repo's
whole amendment grain (`MAX_AMENDMENT_LAG_DAYS = 365`, `_amended_fields`, `amended_fiscal_end`,
`amended_fields`) is built on knowing.

### 1.5 Identifiers — and the two joins you cannot avoid

- **`ticker` is not stable, and history is retroactively rewired.** FAQ: *"**When a ticker changes we
  rewire the history for the updated ticker.** When a company is delisted and its ticker is later
  reused by a different company, the active company keeps the ticker and the delisted company gets a
  number appended."* The `actions` table records historical ticker changes.
- **`permaticker` is the stable id — and it is NOT a column in SF1.** Verified against the schema.
  *"This field is not included in other tables in order to maximise data scalability in those tables;
  join from those tables via ticker to TICKERS (filter on the table field as needed)."*
- **⚠ There is no `cik` column in ANY Sharadar table.** Grepped all 14 tables' descriptions
  (~350 indicator rows): zero. CIK exists only inside a URL: `tickers.secfilings` = *"The URL pointing
  to the SEC filings which also contains the Central Index Key (CIK)."* Measured for AAPL:
  `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193` — zero-padded to 10
  digits, **must be regex-extracted**. This matters directly: `cik` is how the repo's SEC layer keys
  everything, and the Sharadar-first/SEC-second merge needs a reliable join key.
- **`tickers` has one row per ticker PER TABLE.** Measured: AAPL returns three rows (`stocks`,
  `fundamentals`, `insiders`). **Filter `table=fundamentals` or triple-count.**
- Also in `tickers`: `cusips` (AAPL `037833100`), `figi` (`BBG000B9XRY4`), `currency`, `category`,
  `isdelisted`, `firstquarter`, `relatedtickers`, `sector`/`industry`, `scalemarketcap`, `exchange`.

**Two `tickers` caveats:**
- `sector`/`industry` are *"based on SIC codes **in a format which approximates to GICS**"* — not
  GICS. The repo's regime assignment and peer grouping are GICS-based
  ([kpi_catalogue.py:377](../../../src/data_extract/utils/fundamentals/kpi_catalogue.py#L377)), so
  Sharadar's classification is not a substitute for `sp500_tickers`.
- `scalemarketcap` *"categorises the company according to it's maximum observed market cap"* and is
  flagged *"experimental and subject to change"* — **maximum observed** is forward-looking, so it is
  not a point-in-time size classification. Do not use it as a feature.
- No historical listing-venue history: *"Not at present, we are working on it. The exchange in the
  tickers table is the latest primary listing venue."*

### 1.6 Coverage, survivorship and the Russell 1000 question

**Measured** (`tickers`, `table=fundamentals`, paged): **17,825 rows — 5,529 `isdelisted=N`,
12,295 `isdelisted=Y`** (1 row unparseable, embedded comma in a name), corroborating the documented
*"nearly 18,000 active and delisted US public companies ... including 12,000 delisted"*.

FAQ: *"we estimate our data is **99% free of survivorship bias** ... We're continuously working on
that last 1%, with the most common issue holding us back being **malformed historic electronic
records** that we are improving our ability to process."*

**⚠ Russell 1000: NOT FOUND IN PRIMARY SOURCE.** The word "Russell" appears nowhere in Sharadar's
documentation. There is no Russell membership table — only `sp500`. A Russell 1000 universe must be
sourced elsewhere (the repo's single entry point is `sp500_tickers`,
[src/utils/universe.py:15](../../../src/utils/universe.py#L15)) and joined to Sharadar by ticker.
5,529 live tickers makes Russell 1000 comfortably in scope by *count*; the gaps will be ADR/Canadian
names (no quarterly data) and secondary share classes.

**⚠ SF1 covers *"the primary class of common stock securities"* only.** Secondary classes (GOOG vs
GOOGL, BRK.A vs BRK.B) live in `stocks`/SEP, not `fundamentals`. The repo already drops
`["GOOG","FOX","NWS","EA"]` via `data_extract.redundant_ticks`, so this is largely aligned — but
`sharefactor` is Sharadar's mechanism for multi-class economics and **whether `sharesbas` sums all
classes is NOT documented**. That is the same trap the repo solved painfully for 36 multi-class
tickers by summing cover-page `dei:EntityCommonStockSharesOutstanding`.

### 1.7 History depth — the docs contradict themselves

| stated start | where |
|---|---|
| **January 1998** | <https://sharadar.com/docs/fundamentals> — the *table-level* doc, most specific |
| January 1997 | <https://sharadar.com/fundamentals> |
| December 1997 | <https://sharadar.com/bundle> |
| **1999** | <https://sharadar.com/blog/posts/sharadar-launches-direct> — the only source for the spec's figure |
| 1997 | legacy Quandl fact sheet (PDF) |

Also measured: `tickers.firstquarter` for AAPL is **1992-12-31**, described as *"The first financial
quarter available in the dataset"* — earlier than every stated start date. The free tier caps history
at ~5 years so this cannot be settled by measurement without a paid key. **Practically: assume 1998,
verify on day one.** Note `data_extract.years_history` is `15` today
([configs/configs.yml](../../../configs/configs.yml)); 1998→2026 is 28 years.

### 1.8 Access mechanics, and what is undocumented

- Row API: `GET /v1.0/data/fundamentals?api_key=…&ticker=…&dimension=ARQ&calendardate.gte=…`.
  Operators: `=`, `.gt=`, `.gte=`, `.lt=`, `.lte=` only — no `.in`, no `.ne`.
- **⚠ Defaults that silently truncate**: `format=csv`, `limit=10000`, `sort=date.desc`, and **`from`
  defaults to "1 year ago"**, `to` to "prior day". A loader that omits `from` gets one year. `sort`
  applies to one field only: *"Only the first sort field is applied if multiple are provided."*
- **Bulk is the documented backfill path**: `?years=5|10|full` → HTTP 302 to a time-limited presigned
  zip. `?status=True` reports size; the full `fundamentals` zip is **620.8 MB**.
  [llms.txt](https://sharadar.com/llms.txt): *"For an entire table, use bulk ... **Do not page a full
  table with `limit` and `skip` / `offset`.**"*
- Incremental: `lastupdated.gte=YYYY-MM-DD`, documented as *"can be used to retrieve recently updated
  records"*, with its own btree index.
- Delivery: **twice daily, 17h30 and 23h30 US Eastern**; reporting lag *"< 1 day"*. Not intraday.
  Nasdaq reports `"update_frequency":"CONTINUOUS"`, `"status":"ON TIME"`.
- Public schema endpoint, **no key required**: `GET /v1.0/schema/{table}?format=postgres|sqlite|mysql`.
- **⚠ NO RATE LIMITS ARE DOCUMENTED ANYWHERE.** Grepped all 17 doc pages, 5 product pages and
  `llms.txt` for `rate limit` / `throttle` / `429` / `requests per` / `concurrent`: zero hits. Only
  the row `limit` is documented. Contrast the SEC path, where ~9 req/s is a hard designed-around
  constraint ([parallel_fetch.py:6](../../../src/data_extract/utils/common/parallel_fetch.py#L6)).
- **⚠ No Python client for the Direct API**, and `llms.txt` forbids using `nasdaqdatalink`/`quandl`
  with a sharadar.com key. A fetcher would use the repo's existing
  [src/utils/polite_http.py](../../../src/utils/polite_http.py) (`get_json` / `http_get`, curl_cffi
  with retry + backoff) — the same pattern as `fetch_roic_transcripts.py` and `fetch_cusip_map.py`.
  There is no new dependency to add.

**⚠ Two documented traps that cost a day each if missed:**

1. **Units differ between two tables in the same subscription.** SF1 money columns are **actual
   units** (typed `bigint`; AAPL `marketcap = 4508288143800`), while the `daily` table declares
   **USD millions** (same day: `marketcap = 4522736.4`) — a 10⁶ factor. `actions.value` for
   `bankruptcyliquidation` is also millions.
2. **Only 8 columns are USD-converted**: `revenueusd`, `netinccmnusd`, `equityusd`, `debtusd`,
   `ebitdausd`, `ebitusd`, `epsusd`, `cashnequsd` (plus the `fxusd` rate). **`assets`, `liabilities`,
   `netinc`, `ncfo`, `capex`, `gp`, `opinc`, `inventory`, `ppnenet` and everything else are in the
   filer's reporting currency with no USD twin** — while `marketcap`/`price` are always USD. So `pb`
   mixes units for a non-USD filer, in the same row. The docs never flag this; it is established by
   the schema plus `tickers.currency` = *"The company **functional reporting currency** for the
   fundamentals table"*.

Also: ratio columns are **decimal fractions, not percentages**, despite the 2019 dictionary typing
them `%` (measured: `grossmargin` 0.501, `roe` 1.379, `divyield` 0.003). And `evebit` is typed
`bigint` while `evebitda` is `double precision` — `evebit` returns integer-truncated (`"29"` vs
`26.94`).

### 1.9 Pricing, as read 2026-08-26

| plan | price | tagline |
|---|---|---|
| **Fundamentals** | **from $19/mo** | *"Financials, actions, events"* |
| Prices | from $9/mo | *"Stocks, funds, metrics"* |
| Investors | from $19/mo | *"Insiders and holdings"* |
| Bundle | from $29/mo | *"Every table in one plan"* |

Note **"from"**. History depth is itself a paid tier: [the subscription-management
post](https://sharadar.com/blog/posts/upgrade-pause-resume) documents *"Upgrade your subscription
history (**5Y → 10Y → Full**)"*, matching the bulk parameter's `years=5|10|full` and the FAQ's
*"depending on your preference and subscription permissions"*. So $19/mo is the **Fundamentals 5-year
monthly** entry price, and **the 10Y and Full prices are NOT FOUND IN PRIMARY SOURCE** — the subscribe
page renders the picker client-side and `/v1.0/plans`, `/products`, `/pricing` all return
`{"error":"Not found."}`. **Since the spec asks for full history since 1999, the relevant price is
the Full tier, which is unknown.** Nasdaq Data Link SF1 pricing is likewise unreadable.

---

## Part 2 — Sharadar SF1, column by column

### 2.1 It is 112 columns, not 150

Marketing says *"150+ financial indicators"* (that figure spans the whole Fundamentals bundle) and
the table doc says *"More than 100"*. The schema endpoint and the `descriptions` table both return
exactly **112**, and the two sets are identical — no column lacks a description, no description
lacks a column. Historically: 73 in the 2015 Quandl doc, 124 in the 2019 `indicators.txt` (including
14 since **removed**: `DILUTIONRATIO`, `EPSGROWTH1YR`, `EPSDILGROWTH1YR`, `NETINCGROWTH1YR`,
`NCFOGROWTH1YR`, `REVENUEGROWTH1YR`, `SHARESWAGROWTH1YR`, `LEVERAGERATIO`, `INTERESTBURDEN`,
`TAXEFFICIENCY`, `FILINGDATE`, `FILINGTYPE`, `EVENT`, `DATEKEY`), one added (`fiscalperiod`).

**That the growth indicators were removed is worth noting** — the repo's `revenueGrowth` /
`earningsGrowth` columns cannot be sourced from Sharadar either.

### 2.2 The 112 columns by group

**Metadata / entity (11):** `ticker`, `dimension`, `date`(`datekey`), `reportperiod`, `calendardate`,
`fiscalperiod`, `lastupdated`, `price`, `sharesbas`, `sharefactor` — plus `fxusd` (grouped with
metrics by Sharadar).

**Income statement (25):** `revenue`, `revenueusd`, `cor`, `gp`, `opex`, `sgna`, `rnd`, `opinc`,
`intexp`, `ebit`, `ebitusd`, `taxexp`, `consolinc`, `netincnci`, `netinc`, `prefdivis`, `netinccmn`,
`netinccmnusd`, `netincdis`, `eps`, `epsdil`, `epsusd`, `dps`, `shareswa`, `shareswadil`.

**Balance sheet (28):** `assets`, `assetsc`, `assetsnc`, `cashneq`, `cashnequsd`, `investments`,
`investmentsc`, `investmentsnc`, `receivables`, `inventory`, `intangibles`, `ppnenet`, `taxassets`,
`liabilities`, `liabilitiesc`, `liabilitiesnc`, `debt`, `debtc`, `debtnc`, `debtusd`, `deferredrev`,
`payables`, `deposits`, `taxliabilities`, `equity`, `equityusd`, `retearn`, `accoci`.

**Cash flow (13):** `ncfo`, `depamor`, `sbcomp`, `ncfi`, `capex`, `ncfbus`, `ncfinv`, `ncff`,
`ncfcommon`, `ncfdebt`, `ncfdiv`, `ncfx`, `ncf`.

**Metrics / ratios (35) — every one Sharadar-computed:** `marketcap`, `ev`, `ebitda`, `ebitdausd`,
`ebt`, `fcf`, `fcfps`, `tangibles`, `workingcapital`, `invcap`, `invcapavg`, `assetsavg`,
`equityavg`, `roe`, `roa`, `roic`, `ros`, `grossmargin`, `netmargin`, `ebitdamargin`,
`assetturnover`, `currentratio`, `de`, `bvps`, `tbvps`, `sps`, `pe`, `pe1`, `ps`, `ps1`, `pb`,
`evebit`, `evebitda`, `divyield`, `payoutratio`.

**Split: 51 Sharadar-computed / 60 taken from the filing / 1 rate input.** Computed = `calendardate`,
`fiscalperiod`, `assetsnc`, `gp`, `opinc`, `ebit`, all 8 `*usd` conversions, all 35 metrics,
plus `sharefactor`. `eps`/`epsdil` are the strongest as-filed case — *"as calculated and reported by
the company"*.

### 2.3 The 41 zero-filled indicators

Carrying the sentence *"Where this item is not contained on the company consolidated financial
statements and cannot otherwise be imputed the value of 0 is used"*:

`revenue`, `revenueusd`, `cor`, `sgna`, `rnd`, `intexp`, `taxexp`, `netincnci`, `prefdivis`,
`netincdis`, `dps`, `cashneq`, `cashnequsd`, `investments`, `investmentsc`, `investmentsnc`,
`receivables`, `inventory`, `intangibles`, `ppnenet`, `taxassets`, `debt`, `debtc`, `debtnc`,
`debtusd`, `deferredrev`, `payables`, `deposits`, `taxliabilities`, `accoci`, `depamor`, `ncfi`,
`capex`, `ncfbus`, `ncfinv`, `ncff`, `ncfcommon`, `ncfdebt`, `ncfdiv`, `ncfx`, `divyield`.

**This set was cross-validated**: it is *exactly* the set whose `NA Value` column read `0` in the
official 2019 `indicators.txt` — a perfect match in both directions, so the sentence is a real
marker, not editorial boilerplate.

**A second, different policy applies to classified-balance-sheet fields**, which are described as
conditional rather than zero-filled — *"reported if a company operates a classified balance sheet
that segments current and non-current"*: `assetsc`, `assetsnc`, `liabilitiesc`, `liabilitiesnc`,
`debtc`, `debtnc`, `investmentsc`, `investmentsnc`, and `currentratio`. **⚠ `debtc`/`debtnc`/
`investmentsc`/`investmentsnc` appear in BOTH lists** — the docs never reconcile the two policies,
and there is no consolidated statement of which field is in which class. Measured on JPM (a bank,
unclassified sheet): `assetsc`, `liabilitiesc`, `currentratio`, `workingcapital` are **NULL**, while
`cor = 0`, `intexp = 0`, `inventory = 0` are **zero-filled**. So both behaviours are live in one row.

Every SF1 indicator column is nullable in the schema — only `ticker`, `dimension`, `date`,
`reportperiod` are `NOT NULL`. NULL is representable; they simply do not use it for those 41.

### 2.4 Field availability by dimension

The 2015 Quandl doc states the rule: *"the 6 different dimensions are **not available for every
single indicator** ... Balance Sheet values are a point in time measurement and it is unnecessary to
have different annual and quarterly dimensions ... several valuation metrics ... are necessarily
based on annual duration and we therefore only present dimensions for Trailing Twelve Months."*

| statement | 2015 doc | 2019 `indicators.txt` |
|---|---|---|
| Income statement | all 6 | all 6 |
| Cash flow | all 6 | all 6 |
| Balance sheet | `ARQ,MRQ` | `ARQ,MRQ,ARY,MRY` |
| Metrics | `ART,MRT` | `ART,MRT` |
| Metrics (BS-shaped) | `ARQ,MRQ` | `ARQ,MRQ,ARY,MRY` |
| `sharesbas`, `price`, `marketcap` | no dimension | blank |

**Measured against the live API, this is now only partly true and the difference matters:**
- Balance-sheet fields *are* populated in ART and ARY, carrying the period-end instant. The 2019
  "not available" was about meaningfulness, not NULLs — the flat table repeats them.
- `roe`, `assetsavg`, `equityavg`, `invcapavg`, `assetturnover` **are genuinely NULL in ARQ/MRQ**.
- `pe`, `currentratio`, `bvps`, `workingcapital`, `sps`, `divyield`, `marketcap`, `sharesbas` are
  populated in **all** dimensions including ARQ — contradicting the 2019 "ART,MRT only" for
  `pe`/`sps`.

### 2.5 Sign conventions — undocumented, measured

**No sign-convention statement exists anywhere.** Direction is signalled only by phrasing: `capex`,
`ncfi`, `ncff`, `ncfbus`, `ncfinv`, `ncfcommon`, `ncfdebt`, `ncfx` are all *"the net cash **inflow
(outflow)**"*; `netincdis` is *"Amount of **loss (income)**"*; `netincnci` is *"subtracted from
[ConsolInc] in order to obtain [NetInc]"*; `prefdivis` is *"Subtracted from ... [NetInc] to obtain
... [NetIncCmn]"*.

**Measured (AAPL):** `capex` is **negative** (`-2,455,000,000`), `taxexp` **positive**
(`6,478,000,000`), `accoci` negative, `depamor` positive, `revenue`/`netinc`/`ncfo` positive. The
documented `fcf = ncfo − capex` only reconciles with the observed `fcf` if `capex` is read as the
signed outflow: AAPL 2026-Q3 `ncfo` 34,369 − |`capex`| 2,455 → `fcf` 31,914 ✔.

**⚠ This is a direct conflict with the repo, where `capex` is `sign: non_negative` — a positive
magnitude.** A loader must negate.

One sign behaviour *is* documented: *"**Negative P/E Ratios** — Where a company reports negative
earnings it's calculated PE (or PE1) ratio will be negative - please be aware of this when filtering
for low P/E ratios."*

### 2.6 Financial-sector, REIT and insurance treatment

**The docs contain exactly ONE explicit financial-sector accounting rule**, inside `revenue`:

> *"The amount of Revenue recognised from goods sold; services rendered; insurance premiums; or other
> activities that constitute an earning process. **Interest income for financial institutions is
> reported net of interest expense and provision for credit losses.**"*

`deposits` is the only bank-specific field. Beyond that, banks/insurers are handled implicitly by
the classified-balance-sheet conditionality (§2.3).

**Measured on JPM, ARQ, 2024-09-30** — three behaviours the docs do **not** warn about:
- `cor = 0` (zero-filled) and therefore **`gp` = `revenue` exactly** (both 42,654,000,000). So
  `gp` and `grossmargin` are structurally *meaningless but non-null* for a bank, rather than absent.
  This repo makes `grossProfit` `regime_gated: true` with `expected_absent` in 7 of 8 regimes
  precisely so that it *is* absent for a bank.
- `intexp = 0` and `inventory = 0`, both zero-filled.
- `invcap` is still computed (~$4.56tn), silently treating the NULL `liabilitiesc` as absent — so
  `roic` is produced for a bank on an incoherent denominator.

**⚠ REITs: NOT FOUND IN PRIMARY SOURCE.** The strings "REIT", "FFO", "AFFO", "NAV" appear nowhere in
`/docs/fundamentals`, `/docs/descriptions`, `/docs/faqs`, the legacy Quandl documentation, or
`indicators.txt`. No REIT-specific field, no REIT-specific NULL set, no caveat.

**Insurers**: only the "insurance premiums" clause above. No reserves, no float, no combined ratio,
no premiums-written, no DAC.

**Consequence for the plan.** Sharadar cannot supply *any* of the repo's regime-specific machinery.
The repo's `fundamentals_regimes.json` has 8 regimes, 4 `regime_gated` fields, 5 fields with
per-regime concept overrides and 6 tier-0 regime top-line legs. **All of that must continue to come
from the SEC layer** — which is exactly the split the spec proposes, and it is the right one.

### 2.7 Provenance and QA — what Sharadar claims, and what it does not

**Claims** (<https://sharadar.com/about>, and the stronger legacy Quandl *Publisher* section):
> *"Sharadar specializes in **extraction, standardization and organization of financial data from
> company filings**. **We combine people, software and rigorous review processes to generate accurate,
> professional grade data** for professional investors and analysts."*

**Does not claim — and these are commonly assumed:**
- **"XBRL" appears in no Sharadar or Quandl source.** Sharadar nowhere states that it parses XBRL.
- **"EDGAR" likewise appears nowhere.** The docs say *"the form 10 regulatory filing ... submitted to
  the SEC"* and *"company filings"*, never EDGAR or a specific ingestion mechanism.
- **No claim of human audit or third-party audit.** The strongest wording is *"rigorous review
  processes"* — self-described, unaudited.

Provenance breadth: *"The data are originally curated from SEC filings via forms 10; 20; 40; S-1;
S-4; F-1; F-40; and 6-K."*

**The one operational QA statement is negative in scope** (legacy Quandl doc, *Company Errors*):
> *"Where a company makes an error in their filing, **we do not correct for it and report it as
> filed. Should the error be material, the company will restate it, and we will update our
> dataset.**"*

**⚠ There is no errata or known-issues list.** "known issue", "errata", "erratum" appear nowhere. The
nearest thing to a changelog, <https://sharadar.com/blog/posts/latest-updates>, promises *"We'll list
recent updates to the datasets, and we'll keep adding to this same post"* — and its **most recent
entry is 2019-01-15**. It has not been maintained. Corrections are communicated only through
`lastupdated`, and through `actions` for ticker changes.

**Method-instability warning** in three definitions (`invcap`, `invcapavg`, `roic`): *"**Please note
this calculation method is subject to change.**"*

**No warranty** (<https://sharadar.com/terms> §12): *"The Services Data may be delayed, inaccurate or
contain errors or omissions, and Sharadar and its third-party suppliers will have no liability with
respect thereto."*

### 2.8 The incremental-refresh contract, and the trap in it

Documented idiom: `lastupdated.gte=2023-01-01`, *"This can be used to retrieve recently updated
records."*

**⚠ Measured: `lastupdated` does not behave as a per-row change stamp in SF1.** Every AAPL ARQ row
from 2021-09-25 through 2026-06-27 carries the **same** `lastupdated = 2026-07-31`. By contrast the
`daily` and `tickers` tables *do* vary per row (`tickers`: 2026-08-25 for AAPL's `stocks` row vs
2026-08-20 for its `fundamentals` row). Whether the uniform SF1 value is a property of the free
sample snapshot or of the paid table **cannot be determined without a paid key, and is not
documented either way.**

Three consequences for a loader:
1. A `lastupdated.gte` delta may return a wide swathe of history, not just the newly filed quarter →
   **upsert on the primary key, never append.** The repo's `store.save` is already
   `INSERT … ON CONFLICT (pk) DO UPDATE`
   ([store.py:274](../../../src/data_store/store.py#L274)), so this is free.
2. `lastupdated` is a **date, not a timestamp**, and delivery is twice daily. A same-day pull can
   miss the 23h30 batch; `.gte` on the prior day is the safe filter.
3. Because AR *"may include multiple observations in a quarter"*, `(ticker, dimension, calendardate)`
   is not unique — `datekey` is load-bearing in the key.

---

## Part 3 — The field map: Sharadar 112 → this repo's 60

### 3.1 How to read this

Left side is the repo's `fundamentals_history` value columns in statement order
([kpi_catalogue.py:199-230](../../../src/data_extract/utils/fundamentals/kpi_catalogue.py#L199-L230),
`HISTORY_STATEMENT_ORDER`). `Basis` column: **✅** definitions align; **⚠** a real definitional
difference; **∅** no Sharadar equivalent. `Cube` = whether anything in `src/data_aggregate` reads it.

### 3.2 The map

| # | repo column | Sharadar | Basis | Cube | note |
|---|---|---|---|---|---|
| 1 | `totalRevenue` | `revenue` | ⚠ | ✅ | bank basis is post-provision; see §3.3(a) |
| 2 | `premiumsEarned` | — | ∅ | ✅ | insurer top-line leg; no equivalent |
| 3 | `netInterestIncome` | — | ∅ | ✅ | bank leg; no equivalent |
| 4 | `noninterestIncome` | — | ∅ | ✅ | bank leg; no equivalent |
| 5 | `netInvestmentIncome` | — | ∅ | ✅ | insurer leg; no equivalent |
| 6 | `realizedInvestmentGains` | — | ∅ | ✅ | insurer leg; feeds core-earnings adjustment |
| 7 | `rentalIncome` | — | ∅ | ✅ | REIT leg; no equivalent |
| 8 | `costOfRevenue` | `cor` | ⚠ | ✅ | zero-filled when absent |
| 9 | `grossProfit` | `gp` | ⚠ | ✅ | Sharadar always computes `revenue − cor`; repo is regime-gated |
| 10 | `grossMargins` | `grossmargin` | ✅ | ✅ | `gp/revenue`; fraction not % |
| 11 | `sellingGeneralAdmin` | `sgna` | ✅ | ✅ | zero-filled |
| 12 | `researchAndDevelopment` | `rnd` | ⚠ | ✅ | repo basis is aggregate incl. acquired IPR&D; Sharadar's inclusion undocumented |
| 13 | `depAmort` | `depamor` | ✅ | ✅ | both are the cash-flow DD&A add-back |
| 14 | `stockBasedComp` | `sbcomp` | ✅ | ✅ | **not** zero-filled |
| 15 | `operatingIncome` | `opinc` | ⚠ | ✅ | Sharadar computes `gp − opex` always; repo takes `OperatingIncomeLoss` as filed and deliberately does *not* derive |
| 16 | `operatingMargins` | `opinc/revenue` | ⚠ | ✅ | inherits `opinc` |
| 17 | `ebitda` | `ebitda` | ⚠⚠ | ✅ | **bottom-up vs top-down**; see §3.3(b) |
| 18 | `interestExpense` | `intexp` | ⚠ | ✅ | repo has a bank Rule 9-04 chain; Sharadar has one field, zero-filled |
| 19 | `pretaxIncome` | `ebt` | ⚠ | ✅ | Sharadar `netinc + taxexp` (bottom-up, after NCI); repo uses continuing-ops-before-tax-and-equity-method |
| 20 | `incomeTaxExpense` | `taxexp` | ✅ | ✅ | zero-filled |
| 21 | `effectiveTaxRate` | `taxexp/ebt` | ⚠ | ❌ | read by nothing downstream |
| 22 | `netIncome` | **`consolinc`** | ✅ | ✅ | repo = `ProfitLoss` **incl. NCI** = Sharadar `consolinc`, **not** `netinc`; see §3.3(c) |
| 23 | `profitMargins` | `netmargin` | ⚠ | ✅ | Sharadar uses `netinccmn`; repo uses incl-NCI income |
| 24 | `epsDiluted` | `epsdil` | ⚠ | ✅ | Sharadar is the filer's own EPS; repo is TTM income / TTM diluted shares |
| 25 | `revenue_q` | ARQ `revenue` | ✅ | ✅ | |
| 26 | `netIncome_q` | ARQ `consolinc` | ✅ | ✅ | |
| 27 | `operatingCashFlow` | `ncfo` | ✅ | ✅ | continuing + discontinued, matches repo |
| 28 | `capex` | `capex` | ⚠ | ✅ | **sign flip**; one definition vs the repo's 5 regime variants + per-filer leaves |
| 29 | `freeCashflow` | `fcf` | ✅ | ✅ | `ncfo − capex` both sides |
| 30 | `cash` | `cashneq` (+`investmentsc`) | ⚠ | ✅ | repo = cash + restricted + short-term investments (decision #5); Sharadar `cashneq` is cash only, and there is no restricted-cash field |
| 31 | `restrictedCash` | — | ∅ | ❌ | read by nothing |
| 32 | `shortTermInvestments` | `investmentsc` | ⚠ | ✅ | approximate; Sharadar's is "current investments" |
| 33 | `accountsReceivable` | `receivables` | ⚠ | ✅ | Sharadar is trade **and non-trade**; repo is trade only |
| 34 | `inventory` | `inventory` | ✅ | ✅ | zero-filled |
| 35 | `currentAssets` | `assetsc` | ✅ | ✅ | NULL (not 0) for unclassified sheets — matches repo's `regime_gated` |
| 36 | `ppeGross` | — | ∅ | ✅ | kills `implied_useful_life`, `useful_life_change`, `asset_age` |
| 37 | `accumulatedDepreciation` | — | ∅ | ✅ | kills `asset_age` |
| 38 | `ppeNet` | `ppnenet` | ⚠⚠ | ✅ | Sharadar **includes** operating ROU assets; repo **excludes** finance-lease ROU where detectable |
| 39 | `goodwill` | — (in `intangibles`) | ∅ | ✅ | **bundled**; see §3.3(d) |
| 40 | `intangiblesExGoodwill` | — (in `intangibles`) | ∅ | ✅ | **bundled** |
| 41 | `totalAssets` | `assets` | ✅ | ✅ | |
| 42 | `accountsPayable` | `payables` | ⚠ | ✅ | trade **and non-trade** |
| 43 | `currentLiabilities` | `liabilitiesc` | ✅ | ✅ | NULL for unclassified |
| 44 | `shortTermDebt` | `debtc` | ⚠ | ✅ | Sharadar `debt*` **includes** operating-lease obligations; repo `shortTermDebt` excludes them |
| 45 | `shortTermBorrowingsOnly` | — | ∅ | ❌ | read by nothing |
| 46 | `longTermDebt` | `debtnc` | ⚠ | ✅ | same lease-inclusion difference |
| 47 | `longTermDebtCurrentOnly` | — | ∅ | ❌ | read by nothing |
| 48 | `operatingLeaseLiability` | — (inside `debt`) | ∅ | ✅ | not separable |
| 49 | `financeLeaseLiability` | — (inside `debt`) | ∅ | ✅ | not separable |
| 50 | `totalDebt` | **`debt`** | ✅ | ✅ | **the repo's most unusual choice happens to match**: both are gross debt + finance leases + operating leases |
| 51 | `totalLiabilities` | `liabilities` | ✅ | ✅ | as-reported both sides |
| 52 | `retainedEarnings` | `retearn` | ✅ | ✅ | ⚠ *"May only be reported annually by certain companies; rather than quarterly."* |
| 53 | `minorityInterest` | — | ∅ | ✅ | Sharadar has income-statement `netincnci` but **no balance-sheet NCI** |
| 54 | `stockholdersEquity` | `equity` | ⚠⚠ | ✅ | Sharadar is **parent-only**; repo is **incl-NCI**, and Sharadar has no incl-NCI equity field |
| 55 | `returnOnEquity` | `roe` | ⚠ | ✅ | Sharadar `netinccmn/equityavg`; repo TTM income / instant equity. **NULL in ARQ** |
| 56 | `debtToEquity` | `de` | ⚠⚠ | ✅ | Sharadar `de` is **liabilities/equity** despite its name; repo is `totalDebt/stockholdersEquity` |
| 57 | `basicShares` | `shareswa` | ✅ | ❌ | read by nothing |
| 58 | `dilutedShares` | `shareswadil` | ✅ | ✅ | |
| 59 | `sharesOutstanding` | `sharesbas` | ⚠ | ✅ | cover-page count both sides ✅, but Sharadar is **split-adjusted** and the repo is as-filed; multi-class summation undocumented |
| 60 | `optionOverhang` | `shareswadil/shareswa − 1` | ✅ | ✅ | computable |
| — | `employees` (side table) | — | ∅ | ❌* | no headcount field anywhere in Sharadar. *(*cube reads `employees` off the wrong frame today — §6.2)* |

### 3.3 The definitional conflicts, in order of consequence

**(a) Bank revenue — the one that is a measured, closed question in this repo.**
Sharadar: *"Interest income for financial institutions is reported net of interest expense **and
provision for credit losses**."* The repo's catalogue explicitly bans that basis:
`totalRevenue.regimes.bank.never_use` lists `InterestIncomeExpenseAfterProvisionForLoanLoss` and
`TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses`, with the recorded reason that the
former is *"GROSS interest income. Using it inflates bank revenue by the entire interest-EXPENSE
leg — the single largest available error in this field"* and the measured cost being MTB 110 rows
~32% understated and AXP 91 rows, both closed by `never_use`. The repo's bank top line is
`RevenuesNetOfInterestExpense` = NII + noninterest income, **pre**-provision. Adopting Sharadar means
adopting the post-provision basis for ~16 bank tickers plus 6 broker-dealers.

**(b) EBITDA — bottom-up vs top-down.**
Sharadar: `ebit = netinc + taxexp + intexp`, then `ebitda = ebit + depamor`. The repo:
`ebitda = operatingIncome + depAmort`. These differ by **every non-operating item** — equity-method
income, FX, gains on disposal, impairments below the operating line. For a diversified industrial the
gap is small; for a filer with large non-operating items it is not. `ebitda` feeds `ebitda_to_ev`,
`net_debt_to_ebitda`, `interest_coverage`, `adjusted_ebitda_margin` and `fixed_cost_coverage_margin`
in the cube. Sharadar also tags `ebt` as `[Metrics]`, not `[Income Statement]`.

**(c) Net income — three candidates, and the obvious map is wrong.**
Sharadar ships `consolinc` (incl. NCI) → `netincnci` → `netinc` (after NCI) → `prefdivis` →
`netinccmn` (after preferred). The repo's `netIncome` is `ProfitLoss` — **including** NCI — a decision
its catalogue flags as *"A VALIDATED EXISTING DECISION — do not 'fix' it toward NetIncomeLoss"*,
taken because revenue and total assets have no parent-only US-GAAP concept and a high-NCI filer's
ratios were otherwise built from two different companies (IBKR's parent takes 22.6% of income;
`sales_yield` was ~3.8× too high). **So the correct map is `netIncome ← consolinc`, not `netinc`.**
But note the knock-on: *all* of Sharadar's own ratios (`roe`, `roa`, `pe`, `netmargin`,
`payoutratio`) use `netinccmn`, so Sharadar's shipped ratios sit on a different basis than the
repo's. Either recompute them or accept the basis switch.

**(d) Goodwill is not separable.** Sharadar `intangibles` = *"the carrying amounts of all intangible
assets **and goodwill**"*. The repo splits `goodwill` and `intangiblesExGoodwill`. Downstream this
kills `roic_ex_goodwill`, `goodwill_roic_drag`, `goodwill_to_equity`, `goodwill_growth` and
`goodwill_intangibles_to_assets`' numerator split.

**(e) Equity basis.** Sharadar `equity` is *"attributable to the parent"*. The repo's
`stockholdersEquity` prefers `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`
and its whole consolidated-basis decision rests on that. Since Sharadar has neither an incl-NCI
equity field nor a balance-sheet NCI field, **the repo's consolidated basis cannot be reconstructed
from Sharadar at all**. This also removes the input to the repo's `derived_identity` /
`derived_identity_nci_assumed_zero` reason codes.

**(f) Lease treatment runs in opposite directions.** Sharadar `ppnenet` *"Includes Operating Right of
Use Assets"* and `debt` *"Includes ... operating lease obligations"*. The repo's `ppeNet` **subtracts**
`FinanceLeaseRightOfUseAsset` wherever separately detectable (44% of the 417 tagging tickers), and its
`shortTermDebt`/`longTermDebt` **subtract** lease legs while `totalDebt` **adds them back**. Net
effect: `totalDebt` matches ✅, `shortTermDebt`/`longTermDebt`/`ppeNet` do not.

**(g) `cash` is the repo's own widening, and Sharadar does not share it.** Repo decision #5:
`cash` = cash-and-equivalents **+ restricted cash + short-term investments**. Sharadar `cashneq` is
*"currency on hand as well as demand deposits"* only. Best available reconstruction is
`cashneq + investmentsc`, with restricted cash simply absent. This propagates to `liquid_assets`,
EV, `net_debt`, `cash_to_debt`, `refinancing_risk` and `invested_capital`.

**(h) Trade vs total receivables/payables.** Sharadar `receivables`/`payables` are *"trade **and
non-trade**"*; the repo uses `AccountsReceivableNetCurrent`/`AccountsPayableCurrent`. Affects `dso`,
`dpo`, `cash_conversion_cycle`, `beneish_m_score`'s DSRI leg.

### 3.4 What Sharadar ADDS — including eight currently-dead cube inputs

The cube asks for ~75 columns that no longer exist in `fundamentals_history` (the *"~170 columns whose
inputs left with the rebuild"*, [sql/schema.sql:104](../../../sql/schema.sql#L104)). Every one of
those reads silently returns an empty frame. **Sharadar can supply eight of them directly:**

| dead cube input | Sharadar | revives |
|---|---|---|
| `deferredRevenue` | `deferredrev` | `deferred_rev_intensity`, `rpo_coverage` (partly) |
| `deposits` | `deposits` | `deposit_stickiness` |
| `dividendsPaid` | `ncfdiv` | dividend reconciliation in `dividend_features.py` |
| `buybacks` | `ncfcommon` | `shareholder_yield` buyback leg, `sbc_to_buyback` |
| `accumulatedOCI` | `accoci` | `aoci_to_equity` |
| `nciIncome` | `netincnci` | `nci_income_share` |
| `discontinuedOps` | `netincdis` | (declared, no live consumer) |
| `marketCap` | `marketcap` | the `factors.py:81` fallback path |

Plus genuinely new material with no repo counterpart: `assetsnc`, `liabilitiesnc`, `investments`,
`investmentsnc`, `taxassets`, `taxliabilities`, `opex`, `prefdivis`, `netinccmn`, `dps`, and the full
cash-flow decomposition `ncfi`/`ncff`/`ncfbus`/`ncfinv`/`ncfdebt`/`ncfx`/`ncf` — which the repo's
`fundamentals_history` does not carry at all. `dps` in particular gives a per-share dividend the
repo's `tiingo_comparison.py` docstring records as absent from Tiingo.

### 3.5 The gap register, scored

**16 repo columns with no Sharadar equivalent.** Of these:

- **4 are read by nothing downstream** — `restrictedCash`, `shortTermBorrowingsOnly`,
  `longTermDebtCurrentOnly`, and (`basicShares` maps fine but is also unread). Free losses.
- **6 are the regime top-line legs** — `premiumsEarned`, `netInterestIncome`, `noninterestIncome`,
  `netInvestmentIncome`, `realizedInvestmentGains`, `rentalIncome`. All are `tier: 0` calculation
  inputs whose only consumers are `sector_features.py`'s bank/insurer/REIT KPIs — **which are already
  all-NaN today** because the GICS gate fails closed (§6.2). So switching to Sharadar loses nothing
  that currently works, but it does foreclose ever fixing them from Sharadar.
- **6 are real, live losses**: `goodwill`, `intangiblesExGoodwill`, `ppeGross`,
  `accumulatedDepreciation`, `minorityInterest`, and the two lease liabilities (counted as one loss
  since `totalDebt` still works). These kill `roic_ex_goodwill`, `goodwill_roic_drag`,
  `goodwill_to_equity`, `goodwill_growth`, `implied_useful_life`, `useful_life_change`, `asset_age`,
  and the EV minority-interest addition.
- **`employees` has no Sharadar source at all** — it must stay on the SEC 10-K text parse.

---

## Part 4 — The three acceptance checks, tested against Sharadar

The spec's three checks map onto existing named checks in `CHECK_REGISTRY`. Here is what each becomes
on a Sharadar substrate.

### 4.1 Check #1 — "extracted for all quarters of the same ticker until today"

| existing check | file | portable? | why |
|---|---|---|---|
| `coverage_quarters` | [tier1_value.py:525](../../../src/validate/fundamentals/checks/tier1_value.py#L525) | ✅ | Rule is `gap > 2 × median(own filing gaps)`, ≥4 filings required. Needs only `ticker` + a filing-date series → `datekey`. Works. |
| `coverage_field` | [tier1_value.py:589](../../../src/validate/fundamentals/checks/tier1_value.py#L589) | ⚠ | Rule is `null_rate ≥ 0.5` **and** a peer verdict of UNIVERSAL/MIXED, grouped by `regime`. **Zero-filling breaks the null-rate numerator** and there is no `regime` column. |
| `series_shape` | [tier2_series.py:334](../../../src/validate/fundamentals/checks/tier2_series.py#L334) | ⚠ | The `complete`/`sparse`/`interior_gap`/`late_start`/`early_stop` classifier is source-agnostic, but its **severity ladder keys on `dc_code`** and its `late_start` excusal keys on `catalogue.regime_break_effective`. Shape survives; severity does not. |
| `coverage_universe` | [tier1_value.py:504](../../../src/validate/fundamentals/checks/tier1_value.py#L504) | ✅ | Set difference on tickers. |
| `filing_continuity` | [tier1_value.py:1104](../../../src/validate/fundamentals/checks/tier1_value.py#L1104) | ⚠ | `FILINGS_PER_YEAR_BAND = (3.0, 5.5)` assumes 4 US quarterly filings + 1 annual. **Invalid for ADR/Canadian filers**, who have annual-only data. Reports `distinct_ciks`, which Sharadar has none of. |

**⚠ One gap against the literal wording, and it already exists today.** No check in the package
anchors on the calendar — `grep` for `today|Timestamp.now|utcnow|date.today` across
`src/validate/fundamentals/checks/*.py` returns **nothing**. Continuity is measured *inside* the
filer's own data, so a series that stopped two years ago surfaces only as an `early_stop` shape, not
as staleness against today. **"Until today" is not currently measured for either source.**

**⚠ And one that Sharadar makes structurally impossible.** Quarterly dimensions are US-domestic-only.
For an ADR or Canadian filer in the Russell 1000 there are no quarters to be complete about. That is
a coverage floor, not a bug to fix.

### 4.2 Check #2 — "no weird kicks due to change of definition or field tag modified"

**This is the check that does not survive the move, and it is the repo's strongest asset.**

| existing check | portable? | why |
|---|---|---|
| `tag_switch_break` | ❌ **dead** | Fires on *"a `source_concept` change coinciding with a level step"*. Sharadar has no source concept. |
| `basis_step` | ❌ **dead** | Fires on *"a level step at the exact boundary where `resolution_method` changes"*. Sharadar has no resolution method. |
| `trend_break` | ✅ | 3× vs trailing median of 8. Value-only. |
| `level_outlier` | ✅ | Modified-Z 3.5 on the QoQ log change. Value-only. |
| `scale` | ✅ | 10× vs the field's own median absolute level. Value-only. |
| `cross_vintage` | ⚠ **repurposed** | Today it compares two filings of the same period. On Sharadar the natural analogue is **ARQ vs MRQ for the same `reportperiod`** — which is a genuinely better restatement detector than the SEC version, because Sharadar ships both vintages side by side. |
| `dimensional_scope` | ❌ dead | XBRL `Member|Axis|Domain` regex. |
| `leaf_vs_total`, `derived_vs_asreported`, `adjustment_unguarded`, `catalogue_exclusion_cost`, `catalogue_override_coverage` | ❌ dead | All read resolver provenance or the KPI catalogue. |

Both surviving detectors for #2 are **pure-magnitude** — 3×, MAD-Z 3.5, 10×. The measured base rate
of `tag_switch_break` on the 54-ticker roster is 0.67–0.71% (68 findings), and `basis_step` 0.15%
(43 findings) at a **1.5×** threshold. Those are *small* level steps that only became findings because
a provenance change coincided with them. **A 1.5× step with no provenance signal is invisible to a
3×/10× detector.** That sensitivity is lost.

**Three ways Sharadar itself introduces definitional drift you cannot detect from the data:**
1. The restatement-into-final-MRQ rule (§3.1 below) creates a real, intentional spike.
2. `invcap`, `invcapavg`, `roic` carry *"this calculation method is subject to change"*.
3. 14 indicators were silently removed between the 2019 and 2026 dictionaries, including all six
   growth measures. There is no maintained changelog (last entry 2019-01-15).

### 4.3 Check #3 — "Q4 from FY − (Q1+Q2+Q3) should align with the previous quarter"

**This is the finding with the largest consequence for the plan.**

The repo has a three-check family, all Tier 3, all `high`, all ceiling 0.02:

| check | rule | tolerance | measured today |
|---|---|---|---|
| `holdout_q4` [tier3_internal.py:183](../../../src/validate/fundamentals/checks/tier3_internal.py#L183) | force `Q4 = FY − YTD9` where the filer *also* published a discrete Q4, compare | `0.01` | 591 of 752 cases testable; **98.73% / 98.99% within 1%** |
| `q4_footing` [:86](../../../src/validate/fundamentals/checks/tier3_internal.py#L86) | `Q1+Q2+Q3+Q4 == FY`, **only where the filer tagged all four itself** | `0.02` | 190 findings |
| `annual_footing` [:135](../../../src/validate/fundamentals/checks/tier3_internal.py#L135) | filer's own YTD9 + own Q4 == own FY | `0.02` | 178 findings |

**All three need a duration type Sharadar does not publish.** SF1 has quarterly, annual and TTM —
**no YTD6, no YTD9**. So `holdout_q4` and `annual_footing`, which read the `YTD9` constant imported
from `periods.py`, cannot run at all.

**And `q4_footing` becomes vacuous.** Its own docstring states the trap:

> *"Where our Q4 was DERIVED as `FY - YTD9`, footing the four quarters back to FY is an identity: it
> passes on any numbers at all, including wrong ones. The check therefore runs only on years where
> the filer tagged all four quarters itself."*

On Sharadar, **that guard cannot be satisfied** — SF1 exposes no flag saying whether a quarter was
filer-reported or vendor-derived. And the evidence says Sharadar's Q4 *is* derived:

- ARQ carries a Q4 row whose `datekey` is the 10-K filing date (AAPL: `fiscalperiod=2024-Q4`,
  `reportperiod=2024-09-28`, `date=2024-11-01`, `revenue=94,930,000,000`), and a 10-K contains no
  standalone Q4 income statement. The row is necessarily constructed at 10-K time.
- Sharadar's only documented period-arithmetic method is annual-minus-reported-quarters. The legacy
  Quandl doc shows it explicitly for ABT 2011: `MRY 21.4bn − 9.8 − 9.6 − 9.0 = MRQ 2011-12-31
  (−7.1bn)`, with *"the negative quarter has been calculated by Sharadar by utilising the most
  recently reported information ... in order to ensure that the quarterly and annual financials are
  aligned"* and *"Argubly it would be more representative that it was spread amongst the other
  quarters of the year, however this calculation would be arbitrary unless performed and reported by
  the company itself."*
- **There is no general statement that Q4 is derived by subtraction.** The mechanism is documented
  only through the restatement worked example. **NOT FOUND IN PRIMARY SOURCE** as a general rule.

**And a fourth fact removes the obvious workaround.** Sharadar documents that
**ART ≠ the sum of the prior four ARQ**:

> *"datapoints under the ART dimension ... **do not necessarily reflect the sum of the prior four
> ARQ** datapoints ... if an earlier reporting period is restated it will not be reported as a
> quarterly ARQ datapoint ... However, this restated quarter will still be considered in the
> calculation of a corresponding Trailing Twelve Month datapoint."*

So you cannot cross-check ARQ against ART either.

**Net: check #3 is not portable, and the repo's own SEC pipeline is the only place it can be run.**
That is an argument for keeping the SEC layer alive as a *validator* of Sharadar even after Sharadar
becomes the primary source — which is a different and more useful role than "the messy fallback".

### 4.4 The 35 checks, partitioned

**Portable as-is (10):** `grain`, `pit_leak`, `coverage_universe`, `coverage_quarters`,
`impossible_value`, `trend_break`, `level_outlier`, `scale`, `peer_ratio`, `duplicate_fact`.

**Portable with modification (7):** `coverage_field` (null-rate broken by zero-fill; needs a regime
source), `series_shape` (shape yes, severity ladder no), `filing_continuity` (band invalid for
foreign filers), `cross_identity` (works — `assets = liabilities + equity` is checkable, though the
NCI leg is unavailable), `cross_vintage` / `restatement_ledger` (repurpose to ARQ-vs-MRQ),
`peer_ratio_abstentions`, `filing_lag` (`datekey − reportperiod`).

**Dead (11):** `basis_step`, `tag_switch_break`, `dimensional_scope`, `leaf_vs_total`,
`derived_vs_asreported`, `adjustment_unguarded`, `catalogue_exclusion_cost`,
`catalogue_override_coverage`, `code_vocabulary`, `unexplained_null`, `expected_absent_drift`.

**Tautological or impossible (3):** `q4_footing`, `annual_footing`, `holdout_q4`.

**Not applicable (4):** `column_contract`, `amendment_ledger`, `same_day_collapse`, `filing_lag`
already counted. (`column_contract` becomes a new contract against the Sharadar table.)

**Mechanically, the port is cheap.** No check names a table — every check reads a `Substrates`
dataclass, and table names appear in exactly one function,
[substrate.py:197-216](../../../src/validate/fundamentals/substrate.py#L197-L216). Checks receive no
`store` and *cannot* read a table. `Substrates` is a plain dataclass already built by hand in
`tests/validate/` with no DB and no network. **But the column names inside each check are literals**
(`period_end`, `duration_type`, `resolution_method`, `source_concept`, `dc_code`, `regime`,
`period_of_report`, `accession_number`), and `FACTS_COLUMNS` is a fixed 24-column tuple pinned by
`tests/validate/fundamentals/test_substrate_contract.py` via an AST scan.

### 4.5 Why this matters: the error volume the spec is reacting to

Measured, run `8090b885946f`, 2026-08-25, 54 tickers, all tiers
([reports/validate/2026-08-25/54_tickers_revalidation.md](../../2026-08-25/../validate/2026-08-25/54_tickers_revalidation.md)):

- **12,769 findings, 2,332 clusters, 1,947 clusters with work, 49 field families**
- health gate `false` — `coverage_field` at 26.31% vs its own 25% ceiling
- **3 clusters settled** between 2026-08-24 and 2026-08-25
- biggest families: `incomeTaxExpense` (961 findings, 53 of 54 tickers), `netIncome` (754, 53),
  `pretaxIncome` (741, 53), `basicShares` (725, 53), `dilutedShares` (707, 52)
- routing hint non-discriminating: 48 of 49 families read `likely-check-or-catalogue`

**At three clusters a day, 1,947 clusters is ~2 years, not months.** The spec's premise is
well-founded. Note also that 14 families touch all 54 tickers — a systematic pattern, which the
report's own routing hint says points at the *check or catalogue*, not at filers.

---

## Part 5 — `fundamentals_history` as Sharadar-first, SEC-second

### 5.1 What "first" has to mean, precisely

The spec says "shardar first (source of truth) then our sec extraction (still messy)". Three
different things that could mean, and they have very different plans:

| interpretation | what it does | cost |
|---|---|---|
| **(A) Row-level precedence** | a `(ticker, as_of)` row comes entirely from Sharadar; SEC rows fill dates Sharadar has no filing for | simplest; but mixes bases *between* rows, so a time series can silently switch definition — exactly the `basis_step` failure the repo built a check for |
| **(B) Cell-level precedence** | per `(ticker, as_of, field)`, take Sharadar if present, else SEC | preserves maximum coverage; but a *ratio* can then be built from two sources on two bases. The repo already has a hard rule against this: *"Never compare two independently forward-filled columns"* ([docs/data_sources.md](../../../docs/data_sources.md)) |
| **(C) Field-block precedence** | Sharadar owns a declared set of fields; SEC owns the rest (regime legs, goodwill split, PP&E gross/accum, NCI, leases, employees) | each field has exactly one basis for all history; the split is declared and auditable; no field ever switches source mid-series |

**(C) is the only one consistent with the repo's existing invariants**, and it happens to match the
natural gap structure: Sharadar covers the universal fields well and has *nothing* for the
regime-specific ones. I am flagging this as a decision for the planning phase rather than choosing
it — but the other two carry a documented, measured failure mode.

### 5.2 What the merge needs that does not exist today

1. **A source-provenance column.** `fundamentals_history` has 4 provenance columns
   (`publication_form`, `is_amendment`, `amended_fiscal_end`, `amended_fields`) and **none of them
   records which source produced a cell**. Without one, "Sharadar first, SEC second" is unauditable
   and `basis_step` cannot be re-pointed at source boundaries. Note the column contract is asserted
   by list equality including order (`column_contract`, ceiling 0.0), so adding one is a deliberate
   contract change plus a `sql/schema.sql` edit plus the fingerprint question in §6.4.
2. **A join key.** Sharadar has no `cik`; the repo keys the SEC layer on `cik` and handles CIK
   cutovers explicitly (`fundamentals_cik_cutover.json`, 3 entries: APA, GOOGL, ETN — reorganisations
   and domestications where the CIK changed mid-history). Sharadar's ticker-rewiring means its
   history is keyed to the *current* symbol. **These two conventions will disagree on exactly the
   tickers the cutover register exists for.**
3. **An `as_of` reconciliation.** Both sources index on the filing date — repo `as_of` is verbatim
   `filing_date` ([build_history.py:182](../../../src/data_extract/utils/fundamentals/build_history.py#L182)),
   Sharadar `datekey` is *"the SEC filing date for AR dimensions"*. That is a genuinely good
   alignment. But the repo *collapses same-day filings by form precedence*
   (`FORM_PRECEDENCE = ("10-K","10-K/A","10-Q","10-Q/A")`) and Sharadar has **no form column** in SF1
   (`FILINGTYPE` was one of the 14 removed indicators). So a same-day 10-K + 10-Q cannot be resolved
   the same way. Measured: `same_day_collapse` fires on 9 ticker-years in the 54-ticker roster.
4. **A TTM decision.** The repo's `duration` fields are stored as **TTM sums of four discrete
   quarters, never a carried-forward annual** (`insufficient_quarters` otherwise), with a 45-day
   staleness cap. Sharadar ships `ART` directly — but ART is documented **not** to equal the sum of
   four ARQ. Using `ART` is cheaper and probably better; it is also a different number from what the
   column has always held.

### 5.3 The reason-code contract is the hard part

`fundamentals_reason_codes` is dense — *one row per null-or-qualified cell at every publication
event* — and `unexplained_null` is a **zero-ceiling critical** check with a one-line LEFT JOIN as its
gate. Its vocabulary is a closed set of 21–22 codes
([reason_codes.py:176](../../../src/data_extract/utils/fundamentals/reason_codes.py#L176)), asserted
on every write.

**Sharadar supplies no reason for anything, and zero-fills 41 indicators.** So either:
- the contract is relaxed for Sharadar-sourced cells (and `unexplained_null` stops being a universal
  gate), or
- new codes are minted for the vendor cases — at minimum `vendor_zero_filled` (indistinguishable
  zero), `vendor_not_supplied` (field absent from SF1 entirely), `vendor_dimension_unavailable`
  (ARQ absent for a foreign filer), `vendor_derived_quarter` (a Q4 known to be constructed).

The second preserves the gate and is more work. This is a planning decision; I note only that the
first quietly deletes the property the repo spent five phases building.

### 5.4 Where the code would go

The spec says "should not be under fundamentals but an entire new folder called
fundamentals_shardar". Two shapes exist in this repo, and the choice has a visible consequence:

**(i) A sub-domain of `data_extract`** — `src/data_extract/utils/fundamentals_shardar/` plus
`src/data_extract/transformers/step_extract_fundamentals_shardar.py`, wired into
`StepExtractAllData` and given commands in `src/data_extract/cli.py`. This matches every existing
domain (`prices/`, `fundamentals/`, `structure/`, `behavioral/`) and yields
`python -m src data_extract fundamentals-shardar`.

**(ii) A top-level package** — `src/fundamentals_shardar/` with its own `cli.py` and `step_*.py`.
[src/cli.py:9-20](../../../src/cli.py#L9-L20) auto-discovers any `src/<dir>/cli.py`, so this yields
`python -m src fundamentals_shardar <command>` for free with no registration. But `AGENTS.md`'s code
map assigns `data_extract/` the job of producing raw tables, and a top-level sibling would be the
first exception.

**(i) is the smaller change and matches the documented architecture.** (ii) gives stronger isolation
and its own CLI namespace. Flagging for the plan.

### 5.5 The new-table checklist, from a worked example

The most recent single-table addition is `fundamentals_check_fix` (commit `f484cc6`, 23 files).
Mandatory minimum is **two files**:

1. **[src/data_store/schema.py](../../../src/data_store/schema.py)** — one `Table(...)` in the
   `Tables` namespace class. `ALL`/`MANAGED`/`BY_NAME` are comprehensions over `vars(Tables)`, so
   **declaration order *is* the `sql/schema.sql` emission order** and there is no second list.
   For a Sharadar table:
   `Table("fundamentals_shardar", ("ticker","dimension","datekey","reportperiod"), date_col="datekey", date_type_cols=("datekey","reportperiod","calendardate","lastupdated"), freshness="quarterly", read_columns=(...))`
   — `read_columns` matters here because 112 columns × ~28 years × 1,000 tickers is a wide table and
   the repo's rule is "never read a large table unprojected".
2. **[sql/schema.sql](../../../sql/schema.sql)** — a block in the generated format:
   `-- [extract] name  (pk: a, b)` (two spaces before `(pk:`), blank line,
   `CREATE TABLE IF NOT EXISTS "name" (...)` with `NOT NULL` on PK members and
   `PRIMARY KEY (...)` last, then `CREATE INDEX IF NOT EXISTS ix_{table}_{col}`.

Both are **declared risk zones** in `AGENTS.md` and need explicit sign-off.

Then, for a fetcher-backed table: `src/constants/constants.py` (base URL, table name, dimension
constants, the field map), `configs/configs.yml` (a `years_history` sibling — 1998 is 28 years, not
15), the fetcher itself, wiring in the step + CLI, a consumer, tests, and doc rows in
`docs/data_schema.md` / `docs/database.md` / `docs/data_sources.md`.

**⚠ Three runtime facts that will bite:**
1. **`sql/schema.sql` is never applied to your live DB.** It is mounted as a Postgres initdb script
   ([docker-compose.yml:64](../../../docker-compose.yml#L64)), which runs only on an empty data
   directory. On a long-lived volume, tables are created by `store.ensure_table`.
2. **`ensure_table` infers column types from the FIRST DataFrame it sees.** An all-`None` object
   column becomes `TEXT` and every later ticker's real number is stored as a string — measured live:
   VRT created `minorityInterest`/`restrictedCash` as TEXT and APA's values came back as
   `'1997000000.0'`. **A Sharadar loader must hard-cast every value column to `float64` before its
   first write**, even for fields the first ticker never populates. Pinned for the SEC path by
   `tests/data_extract/test_build_history.py:352`.
3. **`ensure_table` is a check-then-create with no lock.** Threaded writers on a *cold* table race
   the `CREATE` and the losers silently lose their ticker's rows. The workaround already exists —
   [edgar_driver.py:94-107](../../../src/data_extract/utils/common/edgar_driver.py#L94-L107) uses a
   `threading.Lock` + a `created` set to serialise the first write per table.

### 5.6 Live DB state — narrower than the spec assumes

From [docs/database.md](../../../docs/database.md) (snapshot 2026-08-17, fundamentals section
refreshed to 2026-08-25):

| table | rows | cols | tickers | range |
|---|---|---|---|---|
| `fundamentals_facts` | **317,036** | 26 | **54** | 2009-07-31 → 2026-08-10 |
| `fundamentals_history` | **3,267** | **69** | **54** | 2009-07-31 → 2026-08-10 |
| `fundamentals_reason_codes` | 76,004 | 5 | 54 | same |
| `fundamentals_employees` | 745 | 3 | 54 | 2002-03-20 → 2026-07-29 |
| `fundamentals_check` | 23,656 | 23 | 54 | 2 runs |
| `fundamentals_facts_legacy` | 2.3M | 19 | — | *"Do not read, do not extend"* |

**Scope is 54 tickers, not 500 — deliberately** (the Phase 5 rebuild scope). The earlier 491-ticker /
239-column tables were dropped and rebuilt on 2026-08-24. Nulls are 36.7% of value cells and every
one carries a reason-code row.

**⚠ `prices` is missing entirely from the live DB**, which already blocks the cube build and
`tests/conftest.py::real_frames`. So does `cube`, every `cube_part_*`, `predictions*`, `strategy`,
`sec_13d`, `notes_embedding`, `ticker_descriptions`.

**Consequence for the spec's phrasing.** "Complement Sharadar with our SEC extraction" today means
complementing with **54 tickers'** worth of SEC facts. If the field-block split (§5.1(C)) puts the
regime legs, goodwill split, PP&E gross/accum, NCI and leases on the SEC side, those columns will be
populated for 54 tickers and NULL for the other ~450 until the SEC layer is widened. That is a
knowable, statable outcome — not a surprise — but it should be in the plan explicitly.

---

## Part 6 — Downstream impact on `data_aggregate`

*(The spec says not to fix `data_aggregate`. This section only measures the impact, as asked.)*

### 6.1 The hard contract is 6 columns

Only `ticker`, `as_of`, `sharesOutstanding`, `netIncome`, `freeCashflow`, `stockholdersEquity` can
*raise* — they are forced by one non-optional projected read at
[step_cube_target.py:106](../../../src/data_aggregate/transformers/step_cube_target.py#L106)
(`store.load` without `optional=True`, and `build_select` raises `KeyError` on an unknown column
before reaching the DB). Everything else routes through
[pit.py:62-63](../../../src/data_aggregate/utils/common/pit.py#L62) —
`if field not in fundamentals_history.columns: return empty frame` — or
[capital.py:51-56](../../../src/data_aggregate/utils/common/capital.py#L51).

**So a missing field is silently indistinguishable from an all-NaN field.** Good for a phased
cutover; bad for noticing a mapping mistake. `step_assemble_cube.py:66-78` warns explicitly that a
missing part leaves *"its columns in place, entirely NULL, and the cube's column set still looks
perfect"*.

One non-silent consequence: `write_part` returns `COLUMNS_CHANGED` when the built column set differs
from what is stored, and the sub-step self-recurses with `full=True`. **A fundamentals column change
therefore forces a full cube-part rebuild on the next incremental run.**

### 6.2 Three things already broken, independent of Sharadar

These matter because they change what "impact" means — several features the Sharadar gap would kill
are *already dead*:

1. **The GICS sector gate fails closed.** `sector_gates.row_gate` reads `sector`/`industry_group` as
   columns *of the frame passed in*, which is `fundamentals_history`
   ([step_cube_fundamentals.py:160](../../../src/data_aggregate/transformers/step_cube_fundamentals.py#L160)) —
   and those columns were removed in the rebuild ([sql/schema.sql:98-99](../../../sql/schema.sql#L98)).
   It *"fail[s] CLOSED"* — all-False. **So every sector-gated KPI (bank, insurance, REIT, pharma,
   energy, utilities) is currently all-NaN** and `family_tickers` returns the empty set. GICS does
   reach the cube by another route (`gics.load_gics_maps` off `sp500_tickers`), just not this one.
2. **The employee panel is empty.** `employee_features.py:40` reads `"employees"` off the
   `fundamentals_history` frame; headcount now lives in `fundamentals_employees`, which
   `src/data_aggregate` **never reads**. `revenue_per_employee`, `headcount_elasticity`,
   `ceo_pay_vs_revenue_growth` are gone.
3. **`revenueGrowth`/`earningsGrowth` don't exist, and neither does the documented mechanism.** Both
   [sql/schema.sql:100](../../../sql/schema.sql#L100) and `docs/data_schema.md` say they are computed
   at cube time *"on a fixed 365-day as_of offset"*. There is **no date-offset growth function in
   `pit.py`** — the only `365` is inside `infer_yoy_periods`, which converts a median filing gap into
   a **row** offset capped at 4. All four reads return empty frames. (Sharadar cannot supply these
   either — its six growth indicators were removed after 2019.)

### 6.3 Feature families that a Sharadar cutover would kill

Given §3.5's six real losses:

| lost column(s) | features that go all-NaN |
|---|---|
| `goodwill`, `intangiblesExGoodwill` | `roic_ex_goodwill`, `goodwill_roic_drag`, `goodwill_to_equity`, `goodwill_growth`, `goodwill_intangibles_to_assets` |
| `ppeGross`, `accumulatedDepreciation` | `implied_useful_life`, `useful_life_change`, `asset_age` |
| `minorityInterest` | the EV minority-interest addition → shifts `ebitda_to_ev`, `fcf_to_ev`, `implied_cap_rate`, `ebitdax_to_ev` *silently* (a missing EV component contributes 0, [fundamental_features.py:196-200](../../../src/data_aggregate/utils/fundamentals/fundamental_features.py#L196)) |
| `operatingLeaseLiability`, `financeLeaseLiability` | the lease leg of `total_debt` and `invested_capital` — but `totalDebt` maps ✅, so this is a *decomposition* loss, not a level loss |
| the 6 regime legs | `loss_ratio`, `expense_ratio`, `combined_ratio`, `investment_income_ratio`, `net_interest_margin`, `nii_growth`, `efficiency_ratio`, `bank_operating_margin`, `rental_margin` — **all already all-NaN** per §6.2(1) |
| `employees` | already all-NaN per §6.2(2) |

**Basis changes that shift values without breaking anything** (the more dangerous category, because
nothing surfaces): `ebitda` (bottom-up), `stockholdersEquity` (parent-only), `cash` (narrower),
`ppeNet` (ROU-inclusive), `shortTermDebt`/`longTermDebt` (lease-inclusive), `receivables`/`payables`
(trade + non-trade), `pretaxIncome`, `epsDiluted`, `debtToEquity`. Every ratio built on these moves.

### 6.4 The fingerprint baseline will not notice

`tests/data_aggregate/aggregate_fingerprint_baseline.json` (36 keys) is pinned to
`aggregate_fingerprint_fundamentals.parquet` — a **pre-rebuild 237-column** slice with `_meta`
recording `fundamentals_rows: 1160`, `fundamentals_cols: 237`, 22 tickers, 2019-01-02 → 2026-06-30.
It still carries `sector`, `industry_group`, `employees`, `revenueGrowth`, `earningsGrowth`,
`deposits`, `loans`, `tier1CapitalRatio`, `premiumsWritten` and the rest of the dropped columns, and
it is written once then read verbatim — **it does not track the live table**. So it will neither
detect nor block a Sharadar cutover. Regeneration is gated: *"the baseline may be regenerated ONLY in
a commit that touches no `src/` file, or in a PR that is exclusively a declared numeric change"*.
Two of its own tests are already stale (`DECLARED_DRIFT` predates commit `0053dc3`; the baseline
carries `prim.price_column_returns` while the test requires `prim.macro_factor_returns`).

---

## Part 7 — The DOW-30 proof of concept

### 7.1 It needs no subscription and no signup

`api_key=test-api-key` is a **documented public key**: *"use the test-api-key to query any table for
AAPL data"* (<https://sharadar.com/docs/auth>). Measured behaviour of the free tier, on the same
endpoints:

| request | result |
|---|---|
| `ticker=MSFT` / `IBM` / `KO` / `CAT` / `NVDA` / `JPM` / `GS` / `TRV` / `UNH` / `VZ` / `MMM` / `BA` | data returned |
| `ticker=TSLA` | `{"error":"Exceeds free tier","description":"Please sign up at /subscribe."}` |
| **no `ticker` param** | **silently returns AAPL only** (90 rows, 1 distinct ticker) |
| `dimension=ARQ&from=1990-01-01` (AAPL) | 19 rows, earliest 2021-10-29 — the documented ~5-year cap |
| `tickers` table, unfiltered | non-DJIA rows returned freely |
| `sp500` table | AAPL rows only |

Sample scope: *"more than 150+ indicators for all 30 Dow Jones Industrial Average stocks"*,
`History: Last 5 years`, tables `descriptions`/`fundamentals`/`stocks`/`funds`/`actions`/`sp500`/
`tickers`/`daily`. **No time-limited trial is documented** — it is an ongoing free tier.

**⚠ The silent AAPL-only default is a trap for a PoC**: a loader that forgets `ticker=` gets one
company and no error.

### 7.2 The harness already exists, and it is already DOW-30-shaped

[src/validate/external/tiingo_comparison.py](../../../src/validate/external/tiingo_comparison.py)
(491 lines) and `yahoo_comparison.py` (395 lines) implement exactly this comparison, and were built
DOW-30-capped because *"Free and Power plans are limited to the DOW 30"*. `DOW_30_TICKERS` is already
a constant ([constants.py:769](../../../src/constants/constants.py#L769)). Its architecture is
directly reusable:

- **The a/b/c bucket classification.** (a) same-definition fields graded against an exact-match
  tolerance (`TIINGO_EXACT_MATCH_TOLERANCE_FLOW = 0.02`, `_LEVEL = 0.01`); (b) known-definitional-
  difference pairs, never scored against the exact bar, instead checked for *drift* in the ratio
  `our_value / their_value` via `outliers.detect_level_outliers` — *"a stable structural gap keeps a
  flat ratio and stays quiet, while a NEW discrepancy shows up as a level/YoY outlier"*; (c) fields
  the vendor has no code for, skipped.
- **The TTM-vs-discrete-quarter alignment** already implemented: a flow field is compared against the
  sum of the matched quarter + its 3 preceding quarters, `kind="flow"`/`"flow_abs"`/`"instant"`.
- **The recorded bucket-b evidence pattern** — AXP's card-issuer net-of-interest revenue, GS's
  narrower capex, HON's 2.00× share counts — is exactly the register §3.3's eight conflicts need.

**§3.3 is essentially a pre-populated bucket-b list for Sharadar**, and §3.5 a pre-populated bucket-c
list. The `outliers.py` MAD kernel is source-agnostic and already shared with `scripts/dod/`.

### 7.3 What the PoC can and cannot settle

**Can settle** (DJIA 30, 5 years, free):
- The field map in §3.2 — every mapping, measured, per ticker.
- The eight basis conflicts in §3.3 — quantified as ratio distributions, not asserted.
- Zero-fill prevalence: how often each of the 41 indicators is `0` for a DJIA filer, and whether
  that `0` is real. JPM already demonstrates `cor = 0 → gp = revenue`.
- Whether `datekey` really equals the 10-K/10-Q filing date, against the repo's own
  `fundamentals_facts.filing_date` for the 54-ticker overlap (DJIA ∩ rosters includes AAPL, CSCO,
  JPM, BAC-adjacent, GS, MSFT, NVDA, WMT, CAT, MCD, UNH, JNJ, CVS-adjacent…).
- Sign conventions per field.
- `calendardate` behaviour for the non-calendar filers in the DJIA (AAPL Sept, MSFT June, CSCO July,
  HD/WMT Jan-Feb, NKE May, DIS Sept) — the highest-value part, because it is the spec's
  quarter-to-quarter comparability requirement.
- Whether ARQ Q4 is derived: compare ARQ Q4 against `ARY − Σ(Q1..Q3)` and see whether it is an exact
  identity. **If it is exact to the cent for every ticker-year, that settles the undocumented
  question in §4.3.**
- `sharesbas` vs the repo's cover-page multi-class sum — the DJIA has no dual-class name, so this is
  the one thing the DJIA sample *cannot* test. (BRK-B is in the repo's `in_sample` roster, not the
  DJIA.)

**Cannot settle** without a paid key:
- History before ~2021 — so nothing about the 1997/1998/1999 question, the depth of coverage, or
  pre-2021 definitional drift.
- Whether `lastupdated` is a real per-row stamp (§2.8).
- ADR/Canadian filers (none in the DJIA), hence the domestic-only-quarterly restriction.
- Delisted-ticker behaviour and survivorship.
- Rate limits under real load.
- The Full-history price.

### 7.4 A no-LLM, no-DB diagnostic first

Consistent with the recorded lesson about validating slices with cheap diagnostics before spending on
the expensive path: the entire §7.3 "can settle" list is answerable with **curl + pandas against
`test-api-key`, writing parquet to the scratchpad**, with no DB write, no new table, no schema change
and no subscription. That is the cheapest possible way to test the field map before committing to a
plan — and it is the shape `scripts/sweep_fundamentals_resolution.py` +
`scripts/report_fundamentals_sweep.py` already established for the SEC path (one network pass, then
offline reporting).

---

## Part 8 — Edge-case register

Consolidated, each marked **DOC** (documented) or **MEAS** (measured, undocumented) or
**UNKNOWN** (not in primary sources).

| # | edge case | status | consequence |
|---|---|---|---|
| 1 | 41 indicators zero-filled instead of NULL | DOC | breaks the spec's NULL requirement and `unexplained_null` |
| 2 | classified-BS fields conditional, not zero-filled; 4 fields in *both* lists | DOC (unreconciled) | two null policies live in one row |
| 3 | `gp = revenue` for zero-`cor` banks | MEAS | a meaningless-but-non-null gross margin for every financial |
| 4 | `invcap`/`roic` computed for unclassified sheets | MEAS | incoherent bank ROIC |
| 5 | Q dimensions US-domestic-only | DOC | ADR/Canadian names have no quarters at all |
| 6 | ARQ Q4 constructed at 10-K time | MEAS + DOC-by-example | check #3 becomes an identity |
| 7 | restatement dumped entirely into final MRQ | DOC | intentional single-quarter spike; negative quarters possible |
| 8 | ART ≠ Σ four ARQ | DOC | no cross-check available |
| 9 | `calendardate` differs by dimension for the same period | MEAS | `(ticker,dimension,calendardate)` not unique; do not join on it |
| 10 | annual `calendardate` snaps to Dec-31 | DOC | AAPL FY2025 → 2025-12-31 |
| 11 | quarter snap maximises overlap, not nearest date | DOC | 2018-07-24 → 2018-06-30 |
| 12 | MR `datekey` == `reportperiod` | DOC + MEAS | MR is not point-in-time |
| 13 | AR may have 0 or >1 observations per quarter | DOC | delayed filers up to 18 months |
| 14 | amended filings entirely undocumented | UNKNOWN | the repo's whole amendment grain has no vendor analogue |
| 15 | `lastupdated` uniform across AAPL history | MEAS | delta pulls return wide history → upsert only |
| 16 | no `cik` column anywhere; only in a URL | MEAS | the SEC join key must be regex-extracted |
| 17 | no `permaticker` in SF1 | DOC | `tickers` join mandatory |
| 18 | `tickers` = one row per ticker *per table* | MEAS | triple-counts without `table=fundamentals` |
| 19 | ticker history retroactively rewired | DOC | conflicts with the repo's CIK-cutover register |
| 20 | SF1 actual units, `daily` millions | DOC | 10⁶ factor in one subscription |
| 21 | only 8 USD-converted columns | DOC (by schema) | non-USD filers mix units inside one row |
| 22 | `capex` negative | MEAS | repo wants a positive magnitude |
| 23 | ratios are fractions despite 2019 `%` labels | MEAS | 100× error if trusted |
| 24 | `evebit` integer-truncated (`bigint`) | MEAS | precision loss |
| 25 | `de` is liabilities/equity | DOC | name/metric mismatch |
| 26 | `invcap`/`invcapavg`/`roic` "method subject to change" | DOC | undetectable definitional drift |
| 27 | 14 indicators removed since 2019, incl. all 6 growth measures | MEAS (dictionary diff) | no changelog since 2019-01-15 |
| 28 | `scalemarketcap` uses *maximum observed* market cap | DOC | forward-looking; never a feature |
| 29 | `sector`/`industry` SIC-derived, "approximates to GICS" | DOC | not a GICS substitute |
| 30 | `retearn` may be annual-only for some filers | DOC | interim gaps are structural |
| 31 | `assetsavg`/`equityavg`/`invcapavg`/`roe`/`assetturnover` NULL in ARQ | MEAS | no quarterly ROE/ROA/ROIC |
| 32 | no REIT concepts whatsoever | UNKNOWN/absent | FFO/AFFO must come from SEC |
| 33 | no insurance concepts beyond "insurance premiums" in `revenue` | DOC | reserves/combined ratio must come from SEC |
| 34 | no headcount field | absent | `employees` stays on the SEC text parse |
| 35 | IPO / <4-quarter ART computation | UNKNOWN | how ART/ARY behave at IPO is unstated |
| 36 | transition periods, FY changes, 52/53-week retailers | UNKNOWN | the repo handles these explicitly (`_inclusive_days`, KR's 16-week Q1); Sharadar says nothing |
| 37 | multi-class `sharesbas` summation | UNKNOWN | the repo's hardest-won share-count fix has no vendor analogue |
| 38 | no documented rate limits | UNKNOWN | cannot size a backfill's request budget |
| 39 | no Python client; `nasdaqdatalink` forbidden with a Direct key | DOC | use `polite_http` |
| 40 | `from` defaults to 1 year ago; no-`ticker` returns AAPL only | DOC / MEAS | silent truncation |
| 41 | form 8-K precedes form 10 by days or weeks | DOC | residual PIT gap even in AR — *"the information may have been separately disclosed to the market days (or on rare occassion - weeks) earlier"* |
| 42 | filer errors passed through uncorrected | DOC | *"we do not correct for it and report it as filed"* |
| 43 | no errata list, no maintained changelog | DOC-by-absence | corrections visible only via `lastupdated` |
| 44 | no warranty; no accuracy guarantee | DOC | terms §12 |

---

## Trade-offs requiring your decision

These are the points where I could not choose for you, ranked by how much of the plan they change.

**1. Licence and channel.** Direct at $19/mo entry is Personal-Use-only and its §2 names *consulting*
and *"technology development for a business"* as prohibited; §8 restricts publishing conclusions.
Nasdaq Data Link is the documented institutional route and its price is unreadable. **Options:**
(a) Direct, personal-account, personal use only; (b) contact Nasdaq/Sharadar for institutional
pricing; (c) treat Sharadar as evaluation-only via the free DJIA tier and decide after the PoC.
Everything else in this document is conditional on this.

**2. History-depth tier.** The spec wants 1998/1999 onward. That is the **Full** tier, whose price is
not published. The $19/mo figure is the 5-year tier. 5Y would not meet the spec.

**3. Zero-fill vs NULL.** Sharadar's 41 zero-filled indicators conflict head-on with the spec's
"Null value if the value is not existing at all". **Options:** (a) mint `vendor_zero_filled` reason
codes and treat `0` as unknown for those 41 fields — safe, loses real zeros; (b) accept `0` as a
value — preserves real zeros, silently corrupts every ratio whose denominator is a zero-filled field;
(c) cross-check against the SEC layer per cell to disambiguate — correct, and only possible for the
54 tickers the SEC layer currently covers.

**4. Merge precedence shape.** §5.1: row-level (A), cell-level (B), or field-block (C). I recommend
**(C)** because it is the only one that guarantees a field never switches basis mid-series, which is
the property the repo's `basis_step`/`tag_switch_break` checks exist to protect — but it means
declaring the split explicitly and accepting that SEC-owned fields stay at 54-ticker coverage until
the SEC layer is widened.

**5. Definitional forks.** Eight conflicts in §3.3. For each: adopt Sharadar's basis (comparable with
every other Sharadar user, breaks continuity with the repo's measured decisions), keep the repo's
basis and reconstruct from Sharadar where possible (only `cash` and the receivables/payables cases
are reconstructible; `equity` incl-NCI, the goodwill split, `ebitda` top-down and the lease
decomposition are **not**), or carry both. The bank-revenue one is the sharpest, because the repo has
filing-level measured evidence that Sharadar's basis understates by ~32% on MTB.

**6. What the SEC layer becomes.** The spec frames it as "still messy" and second in precedence. But
§4.3 shows the Q4 hold-out check — the strongest validator in the repo, 98.7–99.0% within 1% — can
*only* run on SEC data. There is a case for keeping the SEC layer as the **independent validator of
Sharadar** rather than as a fallback source. That reframes the 1,947 open clusters: they stop being a
blocker to shipping and become a backlog on the validation layer.

**7. Universe and survivorship.** Sharadar's `sp500` table offers historical membership back to
1998-03-31 with effective dates, which the repo's date-less `sp500_tickers` cannot express. Taking it
would fix survivorship bias but changes the single-entry-point universe contract in
`src/utils/universe.py`. There is no Russell 1000 membership in Sharadar at all.

---

## Open Questions for the Planning Phase

**Answerable from the free DJIA tier, before any spend** (§7.3):
1. Is ARQ Q4 exactly `ARY − Σ(Q1..Q3)` to the cent? (Settles §4.3.)
2. Does `datekey` equal the repo's `fundamentals_facts.filing_date` for the DJIA ∩ roster overlap?
3. How prevalent is zero-fill per field, and how often is a `0` demonstrably wrong?
4. Quantify each of the eight basis conflicts as a ratio distribution.
5. Does `calendardate` behave correctly for the six non-calendar DJIA filers?

**Require a paid key:**
6. Real earliest `datekey` per ticker — 1997, 1998 or 1999?
7. Is `lastupdated` a per-row stamp in the paid table?
8. What are the actual rate limits, and what does a 1,000-ticker × 28-year backfill cost?
9. How does an amended filing appear — new row, overwrite, or `lastupdated` bump only?
10. Does `sharesbas` sum multiple share classes? (Test on BRK-B, GOOGL, FOX, NWS.)
11. ADR/Canadian coverage in the Russell 1000: how many names, and are they annual-only?

**Design decisions with no external dependency:**
12. Package shape: `data_extract/utils/fundamentals_shardar/` (matches the code map) or
    `src/fundamentals_shardar/` (free CLI namespace via auto-discovery).
13. Store SF1 as-delivered (all 112 columns, all 6 dimensions) or ARQ/ARY/ART-only projected to the
    fields the map needs? As-delivered is ~28× the current facts table but is the only shape that
    lets a mapping mistake be re-derived without a refetch.
14. Does the merged `fundamentals_history` gain a source-provenance column? (Contract change,
    asserted by `column_contract` at ceiling 0.0.)
15. TTM: use Sharadar's `ART` directly, or keep the repo's four-discrete-quarter sum for continuity?
16. New reason codes for the vendor cases, or relax `unexplained_null`?
17. Does `data_extract.years_history: 15` get widened, or does Sharadar get its own knob? (1998 is
    28 years.)

---

## Code References

**Sharadar side (none — this is all new):** no `sharadar`/`nasdaqdatalink`/`quandl` reference exists
anywhere in the repo except the spec file itself. No new Python dependency is needed;
`src/utils/polite_http.py` covers the HTTP surface.

**SEC extraction pipeline:**
- [src/data_extract/transformers/step_extract_fundamentals.py](../../../src/data_extract/transformers/step_extract_fundamentals.py) — the 5-fetcher chain, no per-source isolation
- [src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py:867](../../../src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py#L867) — network layer, writes `fundamentals_facts` + `fundamentals_employees`
- [src/data_extract/utils/fundamentals/build_history.py:1037](../../../src/data_extract/utils/fundamentals/build_history.py#L1037) — the publication-event replay; `as_of = filing_date` at [:182](../../../src/data_extract/utils/fundamentals/build_history.py#L182); immutability at [:994](../../../src/data_extract/utils/fundamentals/build_history.py#L994)
- [src/data_extract/utils/fundamentals/periods.py:614-669](../../../src/data_extract/utils/fundamentals/periods.py#L614-L669) — `_ladder`, both Q4 rungs; [:793](../../../src/data_extract/utils/fundamentals/periods.py#L793) `trailing_twelve`
- [src/data_extract/utils/fundamentals/kpi_catalogue.py:199-230](../../../src/data_extract/utils/fundamentals/kpi_catalogue.py#L199-L230) — `HISTORY_STATEMENT_ORDER`, the 60 value columns
- [src/data_extract/utils/fundamentals/reason_codes.py:176](../../../src/data_extract/utils/fundamentals/reason_codes.py#L176) — the closed 22-code vocabulary
- [src/data_extract/utils/common/edgar_driver.py:59](../../../src/data_extract/utils/common/edgar_driver.py#L59) — the reusable generic fetch driver; cold-table lock at [:94](../../../src/data_extract/utils/common/edgar_driver.py#L94)
- [src/data_extract/utils/common/run_manifest.py:73](../../../src/data_extract/utils/common/run_manifest.py#L73) — `manifest_window`, the incremental-window control

**Store / schema:**
- [src/data_store/schema.py:90](../../../src/data_store/schema.py#L90) — the `Tables` namespace; `fundamentals_facts` entry at [:200](../../../src/data_store/schema.py#L200)
- [src/data_store/store.py:274](../../../src/data_store/store.py#L274) — `upsert_dataframe`; [:43](../../../src/data_store/store.py#L43) `ensure_table` (dtype inference, no lock)
- [sql/schema.sql:106-189](../../../sql/schema.sql#L106-L189) — `fundamentals_history` DDL and its 69-column contract
- [tests/data_store/test_store_boundary.py](../../../tests/data_store/test_store_boundary.py) — the grep-level no-SQL-outside-data_store gate

**Validation:**
- [src/validate/fundamentals/checks/__init__.py:72](../../../src/validate/fundamentals/checks/__init__.py#L72) — `CheckSpec` / `CHECK_REGISTRY`
- [src/validate/fundamentals/substrate.py:197-216](../../../src/validate/fundamentals/substrate.py#L197-L216) — **the single coupling point** to table names
- [src/validate/fundamentals/checks/tier2_series.py:293](../../../src/validate/fundamentals/checks/tier2_series.py#L293) — `_provenance_break`, the engine behind check #2
- [src/validate/fundamentals/checks/tier3_internal.py:86-236](../../../src/validate/fundamentals/checks/tier3_internal.py#L86-L236) — the Q4 family
- [src/validate/external/tiingo_comparison.py](../../../src/validate/external/tiingo_comparison.py) — **the reusable PoC harness**, a/b/c buckets + ratio drift
- [src/constants/constants.py:769](../../../src/constants/constants.py#L769) — `DOW_30_TICKERS`

**Aggregation:**
- [src/data_aggregate/transformers/step_cube_target.py:106](../../../src/data_aggregate/transformers/step_cube_target.py#L106) — the only hard 6-column read
- [src/data_aggregate/utils/common/pit.py:52-90](../../../src/data_aggregate/utils/common/pit.py#L52-L90) — `fundamentals_to_daily`, `daily_market_cap`, the lookahead mechanism
- [src/data_aggregate/utils/common/sector_gates.py:54-87](../../../src/data_aggregate/utils/common/sector_gates.py#L54-L87) — the gate that fails closed today
- [tests/data_aggregate/test_aggregate_regression.py](../../../tests/data_aggregate/test_aggregate_regression.py) — the fingerprint gate that will not notice

---

## Related Documentation

- [reports/research/financial-data/2026-08-21-fundamentals-extraction.md](2026-08-21-fundamentals-extraction.md) — §3.4 "How the vendors actually solve this" is the direct predecessor of this document: Compustat/FactSet/Worldscope/Morningstar/Bloomberg templates, `*_DC` reason codes, vintage instability, and the Boritz & No / Lyle-Siano-Yohn measurements
- [reports/research/financial-data/2026-08-21-fundamentals-extraction-part2.md](2026-08-21-fundamentals-extraction-part2.md) — the 17 UNVERIFIED fields, since closed
- [reports/research/financial-data/2026-08-22-fundamentals-validator-methodology.md](2026-08-22-fundamentals-validator-methodology.md)
- [src/validate/README.md](../../../src/validate/README.md) — the operating manual for the 35 checks
- [reports/validate/2026-08-25/54_tickers_revalidation.md](../../validate/2026-08-25/54_tickers_revalidation.md) — the measured error volume in §4.5

**Doc-sync items found incidentally** (not fixed, per the no-changes instruction):
`docs/data_sources.md` points at `fundamentals/fetch_fundamentals_edgar.py` (now
`fetch_fundamentals_sec.py`), `src/utils/fundamentals_tag_ledger.py` (deleted) and
`common/sector_gates.py` (moved to `data_aggregate/utils/common/`); it also still says headcount lands
in `fundamentals_history.employees`. `docs/data_schema.md` records the **old** `fundamentals_facts` PK
and says "48 tables: 40 managed" where the registry now has 52 / 45. `src/validate/README.md:365` and
`tier3_internal.py:31` reference `configs/fundamentals/fundamentals_baselines.json`, which does not
exist. `fundamentals_rosters.json` has no `random_cold` roster despite `src/validate/cli.py:183`
offering it.

---

## Sources

**Sharadar primary (all read 2026-08-26):**
<https://sharadar.com/docs/fundamentals> · `/docs/descriptions` · `/docs/faqs` · `/docs/auth` ·
`/docs/getting-started` · `/docs/tickers` · `/docs/actions` · `/docs/events` · `/docs/sp500` ·
`/docs/daily` · `/docs/stocks` · `/docs/metrics` · `/docs/insiders` · `/docs/holdings` ·
<https://sharadar.com/fundamentals> · `/bundle` · `/prices` · `/investors` · `/sample` · `/subscribe`
· `/about` · <https://sharadar.com/terms> · <https://sharadar.com/llms.txt> ·
<https://sharadar.com/blog/posts/sharadar-launches-direct> · `/blog/posts/upgrade-pause-resume` ·
`/blog/posts/latest-updates`

**Sharadar machine-readable:**
`https://api.sharadar.com/v1.0/schema` · `/schema/fundamentals?format=postgres` (header "As of
2026-08-18") · `/data/descriptions?api_key=test-api-key&format=csv` (the official
indicator-descriptions document) · `/data/fundamentals?api_key=test-api-key&…` (AAPL, JPM, and 12
DJIA names — all measurements labelled MEAS above) · `/data/tickers?…` · `/data/sp500?…` ·
`/data/daily?…`

**Nasdaq Data Link:**
`https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1/metadata.json` and the same endpoint for
`TICKERS`, `EVENTS`, `INDICATORS`, `SEP`, `SFP`, `SF2`, `SF3`, `SF3A`, `SF3B`, `ACTIONS`, `DAILY`,
`SP500`, `METRICS`

**Legacy / archived (used because the modern pages are JS-rendered):**
<https://web.archive.org/web/20150411012856/https://www.quandl.com/data/SF1/documentation/about> —
the only server-rendered SF1 narrative doc; source of the ABT Q4-subtraction worked example, the
per-statement dimension-availability rule and the *Company Errors* policy ·
<https://web.archive.org/web/20190121115719if_/http://www.sharadar.com/meta/indicators.txt> — the 2019
official indicator document with the `Available Dimensions` and `NA Value` columns the current
dictionary no longer publishes ·
<https://resources.quandl.com/a/res-hub/Sharadar_Datasheet_final.pdf> — the official fact sheet

**Not retrievable:** `data.nasdaq.com/databases/SF1` and `/databases/SF1/documentation` (JS-rendered
behind Incapsula; also unreadable in Wayback snapshots 20230608211010 and 20201114165804), and all
Nasdaq Data Link pricing.

**Academic, via the prior research report:** Boritz & No (2020) *JIS* — as-reported XBRL vs
aggregators; Lyle, Siano & Yohn — Compustat vintage instability. Note the prior report's ⚠ correction
that Du, Huddart & Jiang "Lost in standardization" (*JAE* 2023) is **retracted** and must not be
cited.

---

## Follow-up Research — 2026-08-26, measured against the live DB + free tier

Three questions from the user, answered by measurement rather than inference. Method: Sharadar free
tier (`api_key=test-api-key`, DJIA-only, ~5y) vs the live `pea` DB, on the 14 tickers present in
both. No subscription, no writes.

### F.1 Does `as_of` match between Sharadar and SEC? — YES, 99.64%

14 tickers × 5 years, Sharadar `ARQ.datekey` vs `fundamentals_history.as_of`, restricted per ticker
to the overlapping window:

| ticker | n_shar | n_sec | matched | shar-only | sec-only | detail |
|---|---|---|---|---|---|---|
| AAPL, AXP, BA, CAT, CSCO, JNJ, JPM, MCD, MSFT, PG, UNH, WMT | 20 | 20 | 20 | 0 | 0 | **exact** |
| NVDA | 19 | 19 | 19 | 0 | 0 | **exact** |
| **GS** | 20 | **21** | 20 | 0 | **1** | `sec-only: 2024-02-28 (amendment)` |

**Totals: matched 279, sharadar-only 0, sec-only 1 → 99.64% of the union.**
**`reportperiod == fiscal_end` on 279/279 matched pairs = 100.00%.**

So `datekey` (AR dimensions) is the same SEC filing date the repo stores as `as_of`, and Sharadar's
`reportperiod` is the same date the repo stores as `fiscal_end`. Both merge keys align.
**Zero Sharadar-only dates** — Sharadar never invents a publication event the SEC layer lacks.

Covers four fiscal calendars: AAPL (Sep), MSFT/PG (Jun), CSCO (Jul), WMT/NVDA (Jan), rest (Dec).

### F.2 The single mismatch class is amendment-only events

The one discrepancy is systematic, not noise. GS filed a **10-K/A on 2024-02-28** (accession
`0000886982-24-000012`, `period_of_report` 2023-12-31). The repo emits a publication event with
`is_amendment=t`, `amended_fiscal_end=2023-12-31` and 27 fields in `amended_fields`.
**Sharadar has no row at that date in ARQ or ARY** — it goes 2024-02-23 → 2024-05-03.

Population of this class in the live DB (54 tickers, full history): **18 tickers carry amendments**
(SWKS 7 filings, SMCI 4, ADM 4, SPG 3, then MTB/DUK/GS/KR at 2, and MSFT/LLY/PG/PLD/JPM/EQIX/DE/VRT/
WFC/WMT at 1), and **9 same-day multi-accession cases** exist (ADM, JPM, MTB, SMCI ×4, SPG, SWKS).

**⚠ And the repo's amendment row is the worse of the two here.** At `as_of=2024-02-28` the repo has
`totalRevenue`, `netIncome` and `incomeTaxExpense` identical to the original 10-K, but
`totalAssets`, `stockholdersEquity` and `totalLiabilities` are **NULL** — three balance-sheet totals
the 10-K itself carried. Sharadar ignoring the 10-K/A produced the *more correct* series on this
example. One data point, not a law, but relevant to how much the loss costs.

### F.3 Does Sharadar record edits (10-K/A, 10-Q/A)? — NO event, only an effect

**Measured, no.** There is no amendment row, no form column and no amendment flag. `FILINGTYPE` and
`FILINGDATE` were two of the 14 indicators **removed** from SF1 after 2019, so a row cannot even be
attributed to a form.

Sharadar's only record-of-edits mechanism is **AR vs MR divergence on the same `reportperiod`**.
Measured for GS `reportperiod=2023-12-31`:

| source | revenue | netinc | assets | equity | liabilities | taxexp |
|---|---|---|---|---|---|---|
| Sharadar ARY | 46,254 | 8,516 | 1,641,594 | 116,905 | 1,524,689 | 2,223 |
| Sharadar MRY | 46,254 | 8,516 | 1,641,594 | 116,905 | 1,524,689 | 2,223 |
| repo @ 2024-02-23 | 46,254 | 8,516 | 1,641,594 | 116,905 | 1,524,689 | 2,223 |

(all $M). **ARY == MRY exactly**, i.e. Sharadar asserts nothing was restated — and all three agree
with the original 10-K to the dollar. Note also that Sharadar *does* flag restatements elsewhere:
`insiders.formtype` is *"Prepended by RESTATED in the event that the filing is subsequently
restated"*. SF1 has no equivalent.

**Consequence.** You can detect *that* a period's value changed (diff ARQ vs MRQ on `reportperiod`),
but never *when*, *by which filing*, or *what form*. The four provenance columns
`publication_form`, `is_amendment`, `amended_fiscal_end`, `amended_fields` become unfillable from
Sharadar, and `MAX_AMENDMENT_LAG_DAYS` / `FORM_PRECEDENCE` / `_collapse_same_day` have no analogue.

### F.4 `netIncome` maps to `consolinc`, confirmed on live data

Predicted in §3.3(c), now measured. JPM, 11 ART dates: the repo's `netIncome` equals Sharadar
**`consolinc`** on all 11 (49,552 / 50,349 / 54,026 / 53,773 / 58,471 / 59,695 / 56,533 / 58,028 /
57,048 / 58,899 / 65,067 $M) and differs from `netinc` on all 11. `assets` and `equity` match
exactly on all 11. For AXP, `netinc == consolinc` (no NCI) and both match the repo exactly.

### F.5 ⚠ CORRECTION to §3.3(a): the bank-revenue conflict is filer-specific, not universal

My earlier statement that Sharadar's post-provision clause conflicts with the repo's bank basis was
too broad. Measured on the two banks available free:

**JPM — no conflict.** Sharadar ART `revenue` matches the repo's `totalRevenue` **exactly on all 11
dates** (158,104 / 161,689 / 170,582 / 173,362 / 177,556 / 180,932 / 175,644 / 179,417 / 182,447 /
186,973 / 199,408 $M). 177,556 is the very figure the repo's own catalogue quotes as JPM's CY2024
`Revenues`, and the repo's `netInterestIncome` 92,583 + `noninterestIncome` 84,973 = 177,556. Both
sides are on `RevenuesNetOfInterestExpense`, **pre**-provision.

**AXP — conflict confirmed, and quantified.** All 11 dates, Sharadar lower:

| datekey | Sharadar `revenue` | repo `totalRevenue` | gap | gap % |
|---|---|---|---|---|
| 2024-02-09 | 55,592 | 60,515 | 4,923 | 8.1% |
| 2025-02-07 | 60,764 | 65,949 | 5,185 | 7.9% |
| 2026-07-24 | 70,914 | 75,950 | 5,036 | 6.6% |

The repo's own legs foot to its total (13,134 + 47,381 = 60,515 ✓), and **the gap equals AXP's
provision for credit losses** (~$4.9bn for 2023). Sharadar `intexp = 0` for both banks — zero-filled.

**The corrected statement.** Sharadar takes **the filer's own top-line caption**. JPM tags a clean
`RevenuesNetOfInterestExpense`, so Sharadar is pre-provision. AXP tags
`TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses` — the concept the repo bans **by name** —
so Sharadar is post-provision. **The inconsistency is therefore between banks, not between vendors**,
which is precisely the "equivalent between tickers" property the spec requires. The repo's
`never_use` register exists to force one basis across the cohort; Sharadar has no such register.
Measured today: 2 of 2 testable banks disagree with each other by ~8% of revenue.

### F.6 Incidental API traps found while measuring

- **`fields=` silently drops an unavailable field** rather than erroring: requesting
  `...,fiscalperiod,...` returned a CSV with no `fiscalperiod` column and no warning.
- **MR dimensions need a `from`/`to` window on `reportperiod`, not on the filing date.** Querying
  `dimension=MRY&from=2024-01-01` returned an empty body for a `reportperiod=2023-12-31` row, because
  MR's `date` *is* the report period. Easy to read as "no data".
- `lastupdated` is uniform per ticker but **differs between tickers** (AAPL 2026-07-31, GS/JPM
  2026-08-04) — consistent with a per-ticker reprocessing stamp, not a per-row change stamp.
