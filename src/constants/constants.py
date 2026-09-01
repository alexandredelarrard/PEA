"""
constants.py  (src/constants/constants.py)
-------------------------------------------
Project-wide constants. Import these instead of hardcoding the same date
formats or SEC endpoints across modules, so a change happens in one place.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Date formats                                                                #
# --------------------------------------------------------------------------- #
DATE_FORMAT = "%Y-%m-%d"          # ISO day — as_of / filing / query dates
DATE_FORMAT_COMPACT = "%Y%m%d"    # SEC / FINRA daily-file name stamps

# --------------------------------------------------------------------------- #
# Config directory                                                            #
# --------------------------------------------------------------------------- #
# THE one declaration. `context.py` resolves it into `Context.config_dir`; every fundamentals
# loader's default parameter, and the CLI's `-c` default, import it from here rather than
# re-declaring their own copy -- five independent copies is what let the CLI's `-c` flag be
# silently discarded by `context.py` for years (see `Context.config_dir`'s docstring).
DEFAULT_CONFIG_DIR = "./configs"

# --------------------------------------------------------------------------- #
# HEADER for extract                                                          #
# --------------------------------------------------------------------------- #
_HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36; contact@example.com")}

# Tickers EXCLUDED from the modelling universe for INSUFFICIENT HISTORY (< 4 years of price
# data): recent IPOs / spin-offs that can't support the cube's multi-year look-backs.
INSUFFICIENT_HISTORY_TICKERS: frozenset[str] = frozenset({
    "HONA",   # Honeywell Aerospace spin-off (2026)
    "FDXF",   # FedEx Freight spin-off (2026)
    "Q",      # recent listing (2025)
    "SNDK",   # Sandisk / Western Digital spin-off (2025)
    "GEV",    # GE Vernova spin-off (2024)
    "SOLV",   # Solventum / 3M spin-off (2024)
    "VLTO",   # Veralto / Danaher spin-off (2023)
    "KVUE",   # Kenvue / J&J spin-off (2023)
    "GEHC",   # GE HealthCare spin-off (2022, ~3.6y -- closest to the 4y cutoff)
})

# --------------------------------------------------------------------------- #
# SEC EDGAR endpoints (free, no key; require a descriptive User-Agent)         #
# --------------------------------------------------------------------------- #
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# A 13F-HR must be filed within 45 days of quarter-end, so a quarter's positions only
# become knowable ~45 days later. Both 13F feature builders stamp
# `as_of = quarter_end + this` to stay leak-free; it was duplicated as a private
# `_FILING_LAG_DAYS = 45` in each of them.
SEC_13F_FILING_LAG_DAYS = 45
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
# EDGAR company-name search (atom): the authoritative NAME -> CIK lookup. Filtered to
# 13F-HR filers so a fund name resolves to its institutional-manager CIK. {company}
# must be URL-quoted. Response: one <company-info> block per match with <cik> +
# <conformed-name> (tags are lower-case).
SEC_EDGAR_COMPANY_SEARCH_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={company}"
    "&type=13F-HR&dateb=&owner=include&count=10&output=atom")

# 8-K events -> `sec_8k`, one row per item code (see fetch_8k_edgar.py
SEC_8K_FORMS = ["8-K", "8-K/A"]

# SC 13D activist filings (>5% stake WITH intent to influence) + amendments — the event-driven
# EDGAR renamed the form type at the structured-XML mandate: filings through 2024-12-16 are
# "SC 13D", filings from 2024-12-17 are "SCHEDULE 13D". `get_filings(form=...)` matches EXACTLY,
# so dropping either pair silently truncates the table at the changeover -- measured: 461 filings
# across 91 S&P 500 tickers were invisible until both pairs were listed.
SEC_13D_FORMS = ["SC 13D", "SC 13D/A",   # activist (13G = passive is deliberately excluded)
                 "SCHEDULE 13D", "SCHEDULE 13D/A"]

# 13F institutional holdings, walked per-filing-date via edgartools (fetch_13f.py). 
SEC_13F_FORMS = ["13F-HR", "13F-HR/A"]

# Fundamentals (financial-statement) via edgartools -> `fundamentals_facts` / `fundamentals_history_sec`.
FUNDAMENTALS_FORMS = ["10-K", "10-K/A", "10-Q", "10-Q/A"]

# DEF 14A proxy + the DEF 14C information-statement equivalent that CONTROLLED companies file
DEF14A_FORMS = ["DEF 14A", "DEF 14C"]

# The three JSON files that ARE the fundamentals contract -- one entry per KPI (tier, kind,
# sign, unit, definition, primary-source authority, and how to resolve it), the regime ->
# statement-template map, and the measured (regime, field) expected-absence register. JSON
# rather than YAML because they are machine-generated-adjacent, deeply nested and diff-heavy;
# see docs/config.md. Loaded and validated once by
# `data_extract/utils/fundamentals/kpi_catalogue.py`.
FUNDAMENTALS_CATALOGUE_SUBDIR = "fundamentals"
FUNDAMENTALS_KPIS_FILENAME = "fundamentals_kpis.json"
FUNDAMENTALS_REGIMES_FILENAME = "fundamentals_regimes.json"
FUNDAMENTALS_EXCEPTIONS_FILENAME = "fundamentals_exceptions.json"
FUNDAMENTALS_CIK_CUTOVER_FILENAME = "fundamentals_cik_cutover.json"
FUNDAMENTALS_ROSTERS_FILENAME = "fundamentals_rosters.json"

# --------------------------------------------------------------------------- #
# Sharadar SF1 fundamentals (Sharadar DIRECT, api.sharadar.com).               #
# NOT data.nasdaq.com and NOT the nasdaqdatalink / quandl client libraries:    #
# the Direct channel names the filing-date column `date` (Nasdaq Data Link     #
# calls it `datekey`) and ships `fiscalperiod`, which Nasdaq Data Link omits.  #
# The column name is inside the primary key, so the two channels are not       #
# interchangeable. See reports/research/financial-data/                        #
# 2026-08-26-sharadar-fundamentals.md.                                         #
# --------------------------------------------------------------------------- #
SHARADAR_BASE_URL = "https://api.sharadar.com/v1.0"

# The response-header CONTRACT, in delivered order (captured live 2026-08-26). Every response
# is validated against this: `fields=` SILENTLY DROPS an unavailable field rather than
# erroring, so a typo yields a missing column and no warning. A header that disagrees with
# this tuple is an error, not a shrug.
SHARADAR_SF1_COLUMNS: tuple[str, ...] = (
    # identifiers (7)
    "ticker", "dimension", "calendardate", "date", "reportperiod", "fiscalperiod",
    "lastupdated",
    # the 105 value columns, alphabetical as Sharadar delivers them
    "accoci", "assets", "assetsavg", "assetsc", "assetsnc", "assetturnover", "bvps",
    "capex", "cashneq", "cashnequsd", "cor", "consolinc", "currentratio", "de", "debt",
    "debtc", "debtnc", "debtusd", "deferredrev", "depamor", "deposits", "divyield", "dps",
    "ebit", "ebitda", "ebitdamargin", "ebitdausd", "ebitusd", "ebt", "eps", "epsdil",
    "epsusd", "equity", "equityavg", "equityusd", "ev", "evebit", "evebitda", "fcf",
    "fcfps", "fxusd", "gp", "grossmargin", "intangibles", "intexp", "invcap", "invcapavg",
    "inventory", "investments", "investmentsc", "investmentsnc", "liabilities",
    "liabilitiesc", "liabilitiesnc", "marketcap", "ncf", "ncfbus", "ncfcommon", "ncfdebt",
    "ncfdiv", "ncff", "ncfi", "ncfinv", "ncfo", "ncfx", "netinc", "netinccmn",
    "netinccmnusd", "netincdis", "netincnci", "netmargin", "opex", "opinc", "payables",
    "payoutratio", "pb", "pe", "pe1", "ppnenet", "prefdivis", "price", "ps", "ps1",
    "receivables", "retearn", "revenue", "revenueusd", "rnd", "roa", "roe", "roic", "ros",
    "sbcomp", "sgna", "sharefactor", "sharesbas", "shareswa", "shareswadil", "sps",
    "tangibles", "taxassets", "taxexp", "taxliabilities", "tbvps", "workingcapital",
)

# The 41 indicators Sharadar documents as ZERO-FILLED: "Where this item is not contained on
# the company consolidated financial statements and cannot otherwise be imputed the value of
# 0 is used". Cross-validated -- exactly the set whose `NA Value` read 0 in the official 2019
# indicators.txt, a perfect match in both directions.
#
# A 0 here can mean "genuinely not applicable" (a bank has no inventory) or "absent, and we
# wrote a zero anyway" -- and the two are indistinguishable in the payload. Measured over 279
# ARQ rows: `intexp = 0` for JPM and GS, which is provably false. So the rule must be
# PER-FIELD and measured, never global. Phase 1 only records the list; phase 2 measures its
# prevalence and phase 3 acts on it.
SHARADAR_ZERO_FILLED_FIELDS: frozenset[str] = frozenset({
    "revenue", "revenueusd", "cor", "sgna", "rnd", "intexp", "taxexp", "netincnci",
    "prefdivis", "netincdis", "dps", "cashneq", "cashnequsd", "investments", "investmentsc",
    "investmentsnc", "receivables", "inventory", "intangibles", "ppnenet", "taxassets",
    "debt", "debtc", "debtnc", "debtusd", "deferredrev", "payables", "deposits",
    "taxliabilities", "accoci", "depamor", "ncfi", "capex", "ncfbus", "ncfinv", "ncff",
    "ncfcommon", "ncfdebt", "ncfdiv", "ncfx", "divyield",
})

# --------------------------------------------------------------------------- #
# Sharadar phase-2 DIAGNOSTICS -- the vocabulary the read-only gate runner      #
# needs to interpret SF1 without fooling itself.                                #
#                                                                               #
# ⚠ THIS IS NOT THE SEC CHECK SCHEME. Nothing here registers a check, is read   #
# by `src/validate/`, or writes a `fundamentals_check` row (D25). These names    #
# exist for `data_extract/utils/fundamentals_sharadar/diagnostics.py` and for    #
# the phase-3 field map that consumes its output.                               #
# --------------------------------------------------------------------------- #
#: Where the human-approved per-field zero rule lives. Its own subdirectory rather than
#: `configs/fundamentals/`, because that directory is the SEC KPI catalogue's contract and
#: `kpi_catalogue.py` validates every file it finds against a schema this one does not have.
SHARADAR_CONFIG_SUBDIR = "sharadar"
SHARADAR_ZERO_RULES_FILENAME = "sharadar_zero_rules.json"

# --------------------------------------------------------------------------- #
# Sharadar phase-3 FIELD MAP -- SF1's 112 vendor columns -> the repo's          #
# `HISTORY_STATEMENT_ORDER` vocabulary, plus the two registers that clean the   #
# vendor frame BEFORE the rename.                                               #
# --------------------------------------------------------------------------- #
#: The map itself. Data, not code: `field_map.py` is deterministic given this file.
SHARADAR_FIELD_MAP_FILENAME = "sharadar_field_map.json"

#: The per-(field, ticker) correction register. Separate from the zero rules because none of
#: the three defects it exists for IS a zero -- a positive `capex`, a net-basis `intexp`, a
#: split-adjusted share count -- so the zero rule's per-FIELD grain cannot reach any of them.
SHARADAR_CORRECTIONS_FILENAME = "sharadar_corrections.json"

#: The block a human adds to a machine-PROPOSED register to approve it. Both registers are
#: refused without one: a regenerated proposal is byte-identical to a reviewed decision
#: otherwise, and the whole point of the files is that a human looked at the entries.
SHARADAR_APPROVAL_KEY = "_APPROVED"

#: Keys in either register that are DOCUMENTATION, not entries. Prose lives beside the
#: decision it justifies so the two cannot drift into separate files.
SHARADAR_REGISTER_DOC_PREFIX = "_"

#: The correction register's CLOSED action vocabulary. Closed on purpose: a free-form
#: expression field would be code in a config file, and `apply_corrections` raises on an
#: action it does not know rather than skipping it silently.
SHARADAR_CORRECTION_ACTIONS: frozenset[str] = frozenset({
    "null", "null_if_positive", "null_if_negative",
})

#: The field map's CLOSED vocabularies, for the same reason.
SHARADAR_MAP_KINDS: frozenset[str] = frozenset({"direct", "derived", "sec", "null"})
SHARADAR_MAP_OPS: frozenset[str] = frozenset({"sum", "ratio", "ratio_minus_one", "quarter"})
SHARADAR_MAP_SPLIT_BASES: frozenset[str] = frozenset({"count", "per_share"})

#: The only `negate` spelling the map accepts. `true` was the plan's original and phase 2
#: killed it: 13 of 1,346 stored rows carry a POSITIVE `capex`, so an unconditional flip
#: writes a NEGATIVE into a column the SEC catalogue declares `non_negative`. This spelling
#: flips where the convention holds and NULLs where it does not, and the NULLs are counted.
SHARADAR_NEGATE_IF_NON_POSITIVE = "if_non_positive"

# --------------------------------------------------------------------------- #
# Sharadar SPLIT DE-ADJUSTMENT.                                                 #
#                                                                               #
# SF1's whole per-share and share-count block is RETROACTIVELY SPLIT-ADJUSTED:  #
# `sharesbas`, `shareswa`, `shareswadil`, `eps`, `epsdil` and `dps` all report  #
# a pre-split quarter on the POST-split basis, and `sharefactor` is 1.0 on      #
# every one of those rows and does not flag it. Measured 2026-08-26: NVDA's     #
# 2021 rows carry 25.0bn shares against the ~2.5bn then outstanding (10-for-1,  #
# June 2024), WMT 3x, AMZN 20x. De-adjusted, `sharesbas` matches the SEC cover  #
# page EXACTLY on 10 of 10 WMT rows and 10 of 11 NVDA rows (the 11th differs by #
# Sharadar's own 4-significant-figure rounding).                                #
# --------------------------------------------------------------------------- #
#: The `sharadar_actions.action` value naming a share split, and the one naming a spinoff.
#:
#: ⚠ A `split` row is NOT always a share split, and reading it as one is a 100%-error trap.
#: HON carries `split` = 0.5 dated 2026-06-29 CO-DATED with `spinoff` = 1 and
#: `spinoffdividend` = 221.01 (Honeywell Aerospace): it is the spinoff's PRICE adjustment
#: factor, not a share-count event. HON's own as-filed cover page proves it -- `sharesbas` is
#: 316,826,560 on 2026-04-23 and 316,940,010 on 2026-07-23, unchanged across the date. So a
#: split candidate counts only when NO spinoff row shares its (ticker, date).
SHARADAR_ACTION_SPLIT = "split"
SHARADAR_ACTION_SPINOFF = "spinoff"

#: The 4 zero-filled fields `Tables.sharadar_fundamentals.read_columns` deliberately omits --
#: three `*usd` conversions and one vendor ratio, all excluded from `fundamentals_history` by
#: D21. The diagnostic still has to MEASURE them: every one of the 41 documented zero-filled
#: fields must get a rule, and a field nobody projected would otherwise be silently missing.
SHARADAR_DIAGNOSTIC_EXTRA_COLUMNS = ("revenueusd", "cashnequsd", "debtusd", "divyield")

#: DURATION fields: a value covering the PERIOD, so ARQ rows of one fiscal year sum to the
#: ARY row. This is what makes the Q4 identity measurable at all -- summing a balance-sheet
#: field across four quarters is meaningless, and comparing that sum to ARY would manufacture
#: a "deviation" out of arithmetic that was never supposed to hold.
#:
#: Per-share (`eps`, `epsdil`, `dps`), weighted-average share counts (`shareswa`,
#: `shareswadil`) and ratios (`divyield`) are EXCLUDED even though they are period figures:
#: they are averages and quotients, not sums, so ΣQ != FY for them by construction.
SHARADAR_FLOW_FIELDS: frozenset[str] = frozenset({
    # income statement
    "revenue", "revenueusd", "cor", "gp", "opex", "sgna", "rnd", "opinc", "intexp", "ebit",
    "ebitda", "ebt", "taxexp", "consolinc", "netincnci", "netinc", "prefdivis", "netinccmn",
    "netincdis",
    # cash flow
    "ncfo", "depamor", "sbcomp", "ncfi", "capex", "ncfbus", "ncfinv", "ncff", "ncfcommon",
    "ncfdebt", "ncfdiv", "ncfx", "ncf", "fcf",
})

#: Fields where a NEGATIVE value is an artefact, never a fact. This is the ABT failure mode:
#: Sharadar CONSTRUCTS Q4 as `ARY - Σ(Q1..Q3)`, and the legacy Quandl documentation shows that
#: construction yielding ABT 2011 Q4 revenue of -$7.1bn, annotated as intentional "to ensure
#: that the quarterly and annual financials are aligned".
#:
#: Deliberately EXCLUDED because a negative is real there: `equity` (buybacks put MCD, HD and
#: BA below zero), `retearn`, `accoci`, `workingcapital`, `gp` (a pre-revenue biotech),
#: `taxexp` (a tax benefit), every `netinc*`, `ebit`/`ebitda`, `fcf`, and all `ncf*` --
#: those are signed by design. `capex` is excluded too: Sharadar stores it NEGATIVE, and its
#: sign is asserted separately by `confirm_sign_conventions`.
SHARADAR_NON_NEGATIVE_FIELDS: frozenset[str] = frozenset({
    "revenue", "revenueusd", "cor", "sgna", "rnd", "opex", "intexp",
    "assets", "assetsc", "assetsnc", "cashneq", "cashnequsd", "investments", "investmentsc",
    "investmentsnc", "receivables", "inventory", "intangibles", "ppnenet", "taxassets",
    "liabilities", "liabilitiesc", "liabilitiesnc", "debt", "debtc", "debtnc", "debtusd",
    "deferredrev", "payables", "deposits", "taxliabilities",
    "depamor", "sbcomp", "prefdivis", "dps", "divyield",
    "shareswa", "shareswadil", "sharesbas",
})

#: Sharadar field -> (the `fundamentals_history_sec` columns that make up its counterpart,
#: how comparable the two bases are). The SECOND element is the whole point of this map: a
#: zero-fill verdict is only as strong as the basis behind it, and this repo has already
#: measured what happens when a comparison is made across two definitions that were never the
#: same thing.
#:
#:   "exact"     -- the same accounting line on both sides. A disagreement is a real defect.
#:   "sec_wider" -- the SEC column is a documented SUPERSET, so "SEC is non-zero while
#:                  Sharadar is zero" is EVIDENCE, never proof: the difference may live
#:                  entirely in the components Sharadar's column does not carry.
#:
#: The three `sec_wider` entries and why: `cash` is cash + restricted cash + short-term
#: investments (SEC decision #5) against Sharadar's `cashneq` alone; `totalDebt` includes
#: finance- and operating-lease liabilities (SEC decision #4) against Sharadar's lease-free
#: `debt`; `intangibles` is Sharadar's single goodwill-inclusive line against two SEC columns.
#:
#: 20 of the 41 zero-filled fields appear here. The other 21 -- `deposits`, `accoci`,
#: `taxassets`, `taxliabilities`, `prefdivis`, `netincnci`, `netincdis`, `deferredrev`,
#: `divyield`, `dps`, `investments`, `investmentsnc` and the 9 remaining `ncf*` legs -- have
#: NO counterpart in the 60-field SEC catalogue, so their zeros can only be judged on
#: Sharadar-internal evidence. Say so in the report rather than implying a check ran.
SHARADAR_SEC_COUNTERPART: dict[str, tuple[tuple[str, ...], str]] = {
    "revenue":      (("totalRevenue",), "exact"),
    "revenueusd":   (("totalRevenue",), "exact"),
    "cor":          (("costOfRevenue",), "exact"),
    "sgna":         (("sellingGeneralAdmin",), "exact"),
    "rnd":          (("researchAndDevelopment",), "exact"),
    "intexp":       (("interestExpense",), "exact"),
    "taxexp":       (("incomeTaxExpense",), "exact"),
    "depamor":      (("depAmort",), "exact"),
    "capex":        (("capex",), "exact"),
    "cashneq":      (("cash",), "sec_wider"),
    "cashnequsd":   (("cash",), "sec_wider"),
    "investmentsc": (("shortTermInvestments",), "exact"),
    "receivables":  (("accountsReceivable",), "exact"),
    "inventory":    (("inventory",), "exact"),
    "intangibles":  (("goodwill", "intangiblesExGoodwill"), "sec_wider"),
    "ppnenet":      (("ppeNet",), "exact"),
    "debt":         (("totalDebt",), "sec_wider"),
    "debtusd":      (("totalDebt",), "sec_wider"),
    "debtc":        (("shortTermDebt",), "exact"),
    "debtnc":       (("longTermDebt",), "exact"),
    "payables":     (("accountsPayable",), "exact"),
}

#: Fields describing a DISCRETE EVENT rather than a continuing state or flow. A zero here is a
#: fact -- "no business was acquired this quarter", "no debt was issued" -- and the mixed-ticker
#: heuristic below is exactly WRONG for them: of course a company that acquired something in Q2
#: reports zero in Q1. Only a SEC contradiction can null one of these.
#:
#: `ncfdiv` and `dps` are deliberately NOT here even though they look similar. Every DJIA
#: constituent pays a dividend every quarter, so a zero in those is a gap, not a quiet quarter.
SHARADAR_EVENT_FIELDS: frozenset[str] = frozenset({
    "ncfbus", "ncfinv", "ncfcommon", "ncfdebt", "ncfx", "netincdis",
})

#: Share of the zeros THE SEC LAYER COULD ACTUALLY JUDGE (a non-null counterpart) that must be
#: contradicted before the diagnostic proposes `"null"`. A rate over the judged cells, not over
#: all zeros: a field where 4 of 4 checkable zeros are wrong is a broken column, and dividing
#: those 4 by the 140 zeros nobody could check would hide it.
#:
#: Low on purpose. The question is "can a 0 in this column be read as a real 0?", and once any
#: measurable slice is provably wrong with nothing in the payload distinguishing those cells
#: from the rest, the answer is no for the whole column. Nulling also moves Sharadar TOWARDS
#: the SEC layer's own treatment: where the SEC path has no value it stores NULL, never 0.
SHARADAR_ZERO_RULE_CONTRADICTION_SHARE = 0.02

#: How many zeros the SEC layer must be able to judge before its verdict is used at all. Three
#: cells is not a rate, it is an anecdote.
SHARADAR_ZERO_RULE_MIN_CHECKED = 3

#: Share of a field's zeros sitting in tickers that report the SAME field non-zero in another
#: quarter. This is the Sharadar-INTERNAL signal, and it is the only one available for the 21
#: fields with no SEC counterpart: a bank has no inventory in any quarter (structural, and
#: `"keep"` is right), but a bank with 4.2bn of interest expense in Q2 and 0 in Q1 is a fill,
#: not a fact. Not applied to `SHARADAR_EVENT_FIELDS`.
SHARADAR_ZERO_RULE_MIXED_SHARE = 0.5

# --------------------------------------------------------------------------- #
# Sharadar phase-4 MERGE -- the Sharadar TTM frame + the SEC-owned block ->     #
# `fundamentals_history`, plus the gap check that proposes overrides and the    #
# register that records the human decision (D14/D22/D23).                       #
# --------------------------------------------------------------------------- #
#: The override register. Machine-PROPOSED, human-APPROVED: `merge_history` only READS it and
#: never decides at runtime, so the merge stays deterministic given this file. Same
#: `_APPROVED` refusal as the other two registers.
SHARADAR_SOURCE_OVERRIDES_FILENAME = "sharadar_source_overrides.json"

#: The only `source` an override entry may name. There is exactly one legal direction: a
#: Sharadar-owned column may be moved to SEC for a named ticker. The reverse is not an
#: override but a field-block change (D14), which belongs in the field map, and an
#: open vocabulary here would let one become the other silently.
SHARADAR_OVERRIDE_SOURCE_SEC = "sec"

#: The key whose presence -- not its truthiness -- distinguishes a REVIEWED entry from a
#: freshly PROPOSED one. `--propose` writes `null`; a human writes the date they adjudicated
#: it. An unapproved proposal must never change data.
SHARADAR_OVERRIDE_APPROVED_KEY = "approved"

#: `as_of` is Sharadar's `date`, the SEC block is joined BACKWARD onto it (never forward --
#: that is the leak), and this caps how stale the carried SEC snapshot may be. One year: the
#: SEC block is quarterly, so anything past four quarters means the SEC producer has stopped
#: covering that ticker and a carried value would be a fabricated present, not a lag.
SHARADAR_SEC_ASOF_TOLERANCE_DAYS = 370

#: How a duplicate publication date is resolved, in MERGED names (the vendor columns behind
#: them are `date` and `reportperiod`). Sharadar documents its AR dimensions as possibly
#: carrying several observations in one quarter and ships NO form column, so
#: `FORM_PRECEDENCE` has no analogue here: the vendor's own rule is to keep the GREATEST
#: period end, i.e. the most recent period the filer published that day.
SHARADAR_COLLAPSE_KEY: tuple[str, ...] = ("ticker", "as_of")
SHARADAR_COLLAPSE_ORDER = "fiscal_end"

#: The gap check's RELATIVE threshold (D23): |shar - sec| / |sec|. Paired with an absolute
#: floor from `configs.yml`, because either alone is useless -- a 3% relative test fires on
#: every rounding difference in a small number, and an absolute floor alone fires on every
#: large one.
SHARADAR_GAP_RELATIVE_THRESHOLD = 0.03

#: Share of a `(ticker, field)`'s shared dates that must be flagged before the gap is called
#: SYSTEMATIC -- i.e. a BASIS conflict rather than a one-off restatement. AXP `totalRevenue`
#: was 6.6-8.1% low on ALL 11 dates; that persistence is the whole signal. A gap on 1 of 11
#: dates is not an override candidate, and only a systematic gap is ever proposed.
SHARADAR_GAP_SYSTEMATIC_SHARE = 0.8

#: How many shared dates a `(ticker, field)` needs before "most dates" means anything.
SHARADAR_GAP_MIN_DATES = 4

#: The basis forks phase 3 DESIGNED IN. Each of these gaps is expected, is explained in
#: `sharadar_field_map.json`, and is NOT an override candidate -- the report names them so
#: they do not drown the real finding, which is anything gapping that is not on this list.
SHARADAR_GAP_EXPECTED_FIELDS: frozenset[str] = frozenset({
    "stockholdersEquity", "ppeNet", "shortTermDebt", "longTermDebt", "accountsReceivable",
    "accountsPayable", "cash", "ebitda",
})

# --------------------------------------------------------------------------- #
# Earnings-call transcripts (The Motley Fool — free, full text, no API key)    #
# and local FinBERT-tone sentiment scoring of the parsed sections.             #
# --------------------------------------------------------------------------- #
FOOL_BASE = "https://www.fool.com"

# Motley Fool politeness: base inter-request pause (seconds) for the quote-page discovery
# AND the transcript HTML download. Deliberately slow — fool.com sits behind Cloudflare and
# throttles (429) after a short burst; the per-host slowdown in polite_http then ratchets
# this up further. Reporting-lag grace (days): the just-ended calendar quarter is not required
# until this many days after quarter-end, so a not-yet-reported quarter never forces a request.
EARNINGS_CALL_REQUEST_PAUSE = 2
EARNINGS_CALL_REPORT_GRACE_DAYS = 50
# Map an earnings REPORT date (from earnings_surprises) back into the fiscal quarter it reports:
# a report lands ~4-8 weeks after quarter-end, so shifting the report date back this many days puts
# it inside the reported quarter (Feb report -> prior Q4, late-Apr report -> Q1). Used to demand
# only quarters a ticker has ACTUALLY released, instead of a blanket calendar guess.
EARNINGS_REPORT_TO_QUARTER_LAG_DAYS = 45
# Tickers that hold NO earnings call, so a transcript can never be downloaded -> skipped entirely by
# the earnings-call gap logic (no wasted request, not flagged as "missing"). Berkshire Hathaway is
# the classic case (Buffett publishes a letter + holds the annual meeting, but no quarterly call).
# Extend as other no-call names surface.
NO_EARNINGS_CALL_TICKERS: frozenset[str] = frozenset({"BRK-B", "BRK-A"})

# Roic AI earnings-call transcripts API — the PRIMARY recent-gap source (after the HF backbone,
# before Motley Fool): a clean JSON API covering ~2y of history on the FREE tier (5 req/min). Auth
# is the `apikey` QUERY param (not a header). `list` returns the available (year, quarter, date) per
# ticker; `transcript` returns {symbol, year, quarter, date, content} for one fiscal quarter.
ROIC_EARNINGS_LIST_URL = "https://api.roic.ai/v2/company/earnings-calls/list/{ticker}"
ROIC_EARNINGS_TRANSCRIPT_URL = "https://api.roic.ai/v2/company/earnings-calls/transcript/{ticker}"
ROIC_REQUEST_PAUSE = 12.5              # free tier = 5 req/min -> >= 12s between calls

# per-turn `tag` values in EARNINGS_CALL_EMBEDDING_TABLE
EARNINGS_CALL_TAG_QUESTION = "question"      # a sell-side analyst turn (asks)
EARNINGS_CALL_TAG_ANSWER = "answer"          # a management turn answering the current question
EARNINGS_CALL_TAG_PREPARED = "prepared"      # a prepared-remarks (scripted) management turn
# Sections we score for tone (the high-signal prose); 'participants'/'full' are skipped
# for KPIs ('full' stays in the sections table as a format-proof fallback).
EARNINGS_CALL_SCORED_TAGS = ("prepared_remarks", "qa")

# SENTIMENT ANALYSIS
FINBERT_TONE_MODEL = "yiyanghkust/finbert-tone"

# --------------------------------------------------------------------------- #
# CUSIP / CINS -> ticker overrides for the 13F reconciliation                   #
# --------------------------------------------------------------------------- #
# 13F reports holdings by CUSIP, so a name whose identifier we cannot resolve is INVISIBLE in
# `sec13f_hr` (and therefore in the superinvestor sleeve too). `cusip_ticker_map`
# is built from OpenFIGI and records a miss PERMANENTLY, so an unresolved identifier is never
# retried -- measured on the live DB: 15,404 letter-prefixed rows in the map, ZERO resolved to
# a ticker, and 34 of the 500 universe names entirely absent from sec13f_hr.
#
# The cause is domicile, not fuzzy matching: a foreign-domiciled issuer is identified by a CINS
# (a CUSIP whose first character is a LETTER encoding the country -- G Ireland/UK, H Switzerland,
# N Netherlands, V Liberia, Y Singapore), and OpenFIGI does not resolve these from the 13F feed.
# Nearly every S&P 500 name registered in Ireland / Bermuda / Jersey / Switzerland lands here.
#
# Every entry below was RECOVERED FROM THE DATA -- the `NAMEOFISSUER` + `CUSIP` pair in the
# cached 13F INFOTABLE, ranked by how many filers report it -- never typed from memory: a wrong
# identifier silently attributes another issuer's holdings to your ticker. Applied as an
# override so it also corrects a miss already cached in `cusip_ticker_map`.
CUSIP_TICKER_OVERRIDES: dict[str, str] = {
    "G0450A105": "ACGL",    # ARCH CAPITAL GROUP LTD          (Bermuda)
    "G1151C101": "ACN",     # ACCENTURE PLC                   (Ireland)
    "G0176J109": "ALLE",    # ALLEGION PLC                    (Ireland)
    "G0250X107": "AMCR",    # AMCOR PLC                       (Jersey)
    "G0403H108": "AON",     # AON PLC                         (Ireland)
    "G3265R107": "APTV",    # APTIV PLC                       (Jersey)
    "H11356104": "BG",      # BUNGE GLOBAL SA                 (Switzerland)
    "H1467J104": "CB",      # CHUBB LIMITED                   (Switzerland)
    "143658300": "CCL",     # CARNIVAL CORP                   (US-listed pair of Carnival plc)
    "G25508105": "CRH",     # CRH PLC                         (Ireland)
    "26614N102": "DD",      # DUPONT DE NEMOURS INC           (US - absent from the map, not a CINS)
    "G3223R108": "EG",      # EVEREST GROUP LTD               (Bermuda)
    "G29183103": "ETN",     # EATON CORP PLC                  (Ireland)
    "Y2573F102": "FLEX",    # FLEX LTD                        (Singapore)
    "H2906T109": "GRMN",    # GARMIN LTD                      (Switzerland)
    "438516106": "HON",     # HONEYWELL INTL INC              (US - absent from the map)
    "G51502105": "JCI",     # JOHNSON CONTROLS INTERNATIONAL  (Ireland)
    "G54950103": "LIN",     # LINDE PLC                       (Ireland)
    "N53745100": "LYB",     # LYONDELLBASELL INDUSTRIES NV    (Netherlands)
    "G5960L103": "MDT",     # MEDTRONIC PLC                   (Ireland)
    "G66721104": "NCLH",    # NORWEGIAN CRUISE LINE HLDGS     (Bermuda)
    "N6596X109": "NXPI",    # NXP SEMICONDUCTORS NV           (Netherlands)
    "G7S00T104": "PNR",     # PENTAIR PLC                     (Ireland)
    "V7780T103": "RCL",     # ROYAL CARIBBEAN GROUP           (Liberia)
    "G8473T100": "STE",     # STERIS PLC                      (Ireland)
    "G7997R103": "STX",     # SEAGATE TECHNOLOGY HLDGS PLC    (Ireland)
    "G8267P108": "SW",      # SMURFIT WESTROCK PLC            (Ireland)
    "G87052109": "TEL",     # TE CONNECTIVITY PLC             (Switzerland/Ireland)
    "G8994E103": "TT",      # TRANE TECHNOLOGIES PLC          (Ireland)
    "G96629103": "WTW",     # WILLIS TOWERS WATSON PLC        (Ireland)
    "30231G102": "XOM",     # EXXON MOBIL CORP                (US - absent from the map)
    # DELIBERATELY NOT MAPPED — resolve these before adding:
    #  * IVZ  — the recovery scan's top hit for "INVESCO" was 46090E103 = the INVESCO QQQ TRUST
    #    ETF, not Invesco Ltd the asset manager. 13F filers hold QQQ enormously, so filer-count
    #    ranking prefers the ETF; mapping it to IVZ would book QQQ's holdings as Invesco Ltd.
    #    Invesco Ltd is Bermuda-domiciled, so its identifier is a G-prefixed CINS.
    #  * FDXF / HONA — not real tickers (FedEx is FDX, Honeywell is HON, both already present).
    #    These look like corrupt rows in `sp500_tickers`, not a mapping gap.
}

# --------------------------------------------------------------------------- #
# Multi-asset trend (time-series-momentum) sleeve — StepTrendAssetClass output #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Daily prediction + live trading ledger (the `strat_prediction` DAG)          #
# --------------------------------------------------------------------------- #
# `Tables.predictions_latest` is LONG-format: one row per (as-of date, ticker, horizon, model), so
# each row can carry its OWN `predicts_for` -- the h30 and h90 predictions made on the same day
# target different future dates, which a wide pred_h30/pred_h60 layout cannot express.
# `model` values: one per ensemble member, plus these two aggregates.
PREDICTION_MODEL_ENSEMBLE = "ensemble"      # the per-horizon average of that horizon's members
PREDICTION_MODEL_BLENDED = "blended"        # the IR-weighted blend ACROSS horizons

# --------------------------------------------------------------------------- #
# Data-freshness cadence THRESHOLDS -- declarative metadata, no consumer today. #
# --------------------------------------------------------------------------- #
# WHICH tables carry a cadence, and on which date column, comes from
# `schema.freshness_tables()` (`Table.freshness` / `Table.freshness_col`).
# The automated gate that read these was removed; they now only document each
# source's expected refresh rate. Wire a new consumer to `freshness_tables()`
# rather than reintroducing a parallel table list here.
DATA_FRESHNESS_MAX_AGE_DAYS: dict[str, int] = {
    "daily": 4, "weekly": 10, "biweekly": 20, "monthly": 45,
    "quarterly": 140, "yearly": 460,
}
# cadence tiers from the tightest to the loosest (daily -> yearly)
DATA_FRESHNESS_CADENCE_ORDER: tuple[str, ...] = (
    "daily", "weekly", "biweekly", "monthly", "quarterly", "yearly")

# --------------------------------------------------------------------------- #
# Incremental cube-part builds (Airflow data_aggregation DAG)                  #
# --------------------------------------------------------------------------- #

# The join keys every feature panel carries; everything else in a panel is a feature column.
PANEL_KEYS = ["date", "ticker"]

# The cube + the ad-hoc intermediate `cube_part_*` tables the sub-steps hand to each other are
# declared in src/data_store/schema.py (`Tables.cube`, `Tables.cube_part_*`), the parts with
# `managed=False`: they carry a declared PK and date column but own their own DDL (inferred from
# the frame each owning step writes) and stay out of sql/schema.sql. `parts.py` adds only the
# build ORCHESTRATION on top (CLI command, warm-up window, binding look-backs).

# --------------------------------------------------------------------------- #
# GICS sectors / industry groups (values as stored in `sp500_tickers`, carried  #
# onto every `fundamentals_history_sec` row by the extractor)                  #
# --------------------------------------------------------------------------- #
GICS_SECTOR_ENERGY = "Energy"
GICS_SECTOR_FINANCIALS = "Financials"
GICS_SECTOR_REAL_ESTATE = "Real Estate"
GICS_SECTOR_UTILITIES = "Utilities"

GICS_GROUP_BANKS = "Banks"
GICS_GROUP_FINANCIAL_SERVICES = "Financial Services"
GICS_GROUP_INSURANCE = "Insurance"
GICS_GROUP_EQUITY_REITS = "Equity Real Estate Investment Trusts (REITs)"
GICS_GROUP_REAL_ESTATE_MGMT = "Real Estate Management & Development"
GICS_GROUP_PHARMA_BIOTECH = "Pharmaceuticals, Biotechnology & Life Sciences"

# The GICS scope each sector-KPI FAMILY is DEFINED for, as (level, accepted values).
# `sector_gates.py` masks every sector KPI with this instead of asking "did the filer
# report tag X", which mis-fired in BOTH directions:
#   * a tag that is not sector-exclusive leaked the KPI into the wrong sector --
#     `InterestIncomeExpenseNet` is used by 59 non-Financials, so bank NIM / ROA /
#     operating margin were computed for industrials & health care; `OperatingLease-
#     LeaseIncome` did the same for FFO on utilities and IT names;
#   * a tag that IS sector-exclusive but rarely tagged starved it -- only 3 of 21
#     Energy names tag `OilAndGasProperty*`, so EBITDAX / DD&A intensity were empty
#     for 86% of the sector.
# `fundamentals_history_sec` carries sector + industry_group only (no sub-industry), so
# `energy` is scoped at sector level: services / refiners simply report no exploration
# expense or oil&gas property, leaving those KPIs NaN as before.
SECTOR_KPI_SCOPE: dict[str, tuple[str, tuple[str, ...]]] = {
    "bank":       ("industry_group", (GICS_GROUP_BANKS,)),
    "insurance":  ("industry_group", (GICS_GROUP_INSURANCE,)),
    "financials": ("sector",         (GICS_SECTOR_FINANCIALS,)),
    "reit":       ("industry_group", (GICS_GROUP_EQUITY_REITS,)),
    "energy":     ("sector",         (GICS_SECTOR_ENERGY,)),
    "utilities":  ("sector",         (GICS_SECTOR_UTILITIES,)),
    "pharma":     ("industry_group", (GICS_GROUP_PHARMA_BIOTECH,)),
}

# --------------------------------------------------------------------------- #
# VALIDATOR FIX RECORDING (src/validate/, fundamentals_check_fix)                         #
# --------------------------------------------------------------------------- #
# A fix to a validator cluster is an EVENT, recorded once and never revised. These two
# constants are what `validate fix record` refuses on, so they are a closed vocabulary
# rather than a suggestion -- the same discipline as `reason_codes.ALL_CODES`, and for the
# same reason: an unpoliced free-text field stops being queryable within a month.

# What the fix DID, in four coarse terms. Defined by the EFFECT of the edit, never by which
# file it lives in -- otherwise every judgement call becomes an argument about directories.
#
#   check       the check was wrong. The data was fine and the finding was a false positive
#   catalogue   the FIELD SPECIFICATION was wrong (configs/fundamentals/*.json)
#   extraction  any code that PRODUCES a value -- xbrl_linkbase, build_history, periods
#   rows        the code was already right and the STORED DATA was stale; a refetch fixed it
#
# `root_cause` carries the precision. This is grouping, not a taxonomy: four terms stay
# countable in a `GROUP BY`, and a fifth would need a measured reason to exist.
FIX_LAYERS: frozenset[str] = frozenset({"check", "catalogue", "extraction", "rows"})

# The `evidence` JSON keys each layer must supply. They differ because the layers cite
# different KINDS of proof, and a universal requirement would force the wrong one.
#
# An `extraction` / `rows` / `catalogue` fix changed how a FILING is read, so it must name
# the filings: `accessions`. A `check` fix changed a THRESHOLD or a predicate, and there is
# no filing at fault -- demanding an accession would make it cite an irrelevant one. Its
# evidence is the false-positive population it was measured against: how many findings were
# `examined` and how many of those were `benign`.
FIX_EVIDENCE_KEYS: dict[str, frozenset[str]] = {
    "extraction": frozenset({"accessions"}),
    "rows": frozenset({"accessions"}),
    "catalogue": frozenset({"accessions"}),
    "check": frozenset({"examined", "benign"}),
}
