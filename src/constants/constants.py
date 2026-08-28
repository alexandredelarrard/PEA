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

# SEC bulk quarterly structured data sets (free TSV zips; {quarter} = e.g. "2024q1").
# insider = Forms 3/4/5 officer/director transactions; finstmt = primary-statement
# XBRL facts (num/sub) incl. the balance-sheet net pension liability.
SEC_INSIDER_URL_TEMPLATE = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
    "{quarter}_form345.zip")
SEC_FINSTMT_URL_TEMPLATE = (
    "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip")
SEC_INSIDER_FIRST_YEAR = 2011      
SEC_FINSTMT_FIRST_YEAR = 2009     

# SEC "Financial Statement AND Notes" data sets: like finstmt but ALSO carry the
# NOTES (footnote) facts — numeric (num.tsv, incl. footnote PBO / plan-asset /
# funded-status detail) AND the narrative TEXT blocks (txt.tsv, for embedding /
# sentiment). Files are .tsv (not .txt). {period} is either quarterly "YYYYqQ" OR
# monthly "YYYY_MM": the SEC now consolidates months into a quarter after ~1 year,
# so at any time only the last ~12 months exist as monthly and older data as
# quarterly (the fetcher probes both and skips 404s). ~300-450MB per file.
SEC_FINNOTES_URL_TEMPLATE = (
    "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/"
    "{period}_notes.zip")
SEC_FINNOTES_FIRST_YEAR = 2009     # earliest notes data set (2009q1)

# 8-K events -> `sec_8k`, one row per item code (see fetch_8k_edgar.py, which owns the
# high-signal item-code -> tag map).
SEC_8K_FORMS = ["8-K", "8-K/A"]

# SC 13D activist filings (>5% stake WITH intent to influence) + amendments — the event-driven
# catalyst signal, read via edgartools' typed Schedule13D object (reporting persons, CUSIP,
# ownership -- see fetch_13d_edgar.py). One row PER REPORTING PERSON per filing.
SEC_13D_FORMS = ["SC 13D", "SC 13D/A"]   # activist (13G = passive is deliberately excluded)

# 13F institutional holdings, walked per-filing-date via edgartools (fetch_13f.py). 13F-NT is
# excluded on purpose: a notice filing reports that holdings appear in ANOTHER manager's filing,
# so it carries no info table of its own.
SEC_13F_FORMS = ["13F-HR", "13F-HR/A"]

# Fundamentals (financial-statement) forms walked per-filing via edgartools -> `fundamentals_facts`
# / `fundamentals_history_sec`. Amendments included explicitly (never inferred from a form-filter
# default) so a 10-K/A or 10-Q/A restatement is always discovered as its own accession.
FUNDAMENTALS_FORMS = ["10-K", "10-K/A", "10-Q", "10-Q/A"]

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

# DEF 14A proxy + the DEF 14C information-statement equivalent that CONTROLLED companies file
# instead. Centralized here so the form-dispatch registry (form_registry.py) has one source of
# truth, matching SEC_8K_FORMS / SEC_13D_FORMS above.
DEF14A_FORMS = ["DEF 14A", "DEF 14C"]

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
SHARADAR_API_KEY_ENV = "SHARADAR_API_KEY"

# The AS-REPORTED dimensions only. Sharadar also publishes MRQ/MRY/MRT ("most recent
# reported"), which RESTATE IN PLACE when a filer amends -- a row's numbers change under a key
# that did not, so an append-only store cannot tell an amendment from a bug and
# `diff_against_stored` would fire forever. AR* rows are point-in-time and immutable.
SHARADAR_DIMENSIONS = ("ARQ", "ARY", "ART")

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

# The 7 NON-NUMERIC columns. EVERYTHING ELSE IN SF1 IS A VALUE COLUMN AND MUST BE CAST TO
# float64 BEFORE THE FIRST WRITE -- `ensure_table` infers SQL types from the FIRST frame it
# sees, so a column the first ticker never populates lands as an all-None object column,
# becomes TEXT, and every later ticker's real number is then stored as a string. Measured
# live on `minorityInterest` / `restrictedCash`: VRT created them TEXT and APA's values came
# back as '1997000000.0'.
SHARADAR_ID_COLUMNS = ("ticker", "dimension", "calendardate", "date", "reportperiod",
                       "fiscalperiod", "lastupdated")

# History floor for `sharadar_sp500`. Membership events are only useful at FULL depth -- the
# survivorship-bias fix needs the whole series, and the entire table is ~3.3k rows (earliest
# event measured 1992-01-02), so the cold pull is a single request. There is no years-history
# knob for it for that reason.
SHARADAR_SP500_FIRST_DATE = "1990-01-01"

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
# Google Trends (unofficial API — retail-attention proxy). The explore call    #
# returns widget tokens; the multiline call returns the interest-over-time     #
# series for a token. Priming the home URL first sets the required NID cookie.  #
# --------------------------------------------------------------------------- #
GOOGLE_TRENDS_HOME_URL = "https://trends.google.com/?geo=US"
GOOGLE_TRENDS_EXPLORE_URL = "https://trends.google.com/trends/api/explore"
GOOGLE_TRENDS_MULTILINE_URL = "https://trends.google.com/trends/api/widgetdata/multiline"

# --------------------------------------------------------------------------- #
# Earnings-call transcripts (The Motley Fool — free, full text, no API key)    #
# and local FinBERT-tone sentiment scoring of the parsed sections.             #
# --------------------------------------------------------------------------- #
MOTLEY_FOOL_BASE_URL = "https://www.fool.com"
MOTLEY_FOOL_TRANSCRIPT_INDEX_URL = "https://www.fool.com/earnings-call-transcripts/"

# raw transcript HTML + link-index cache, relative to DATA_STORE (non-tabular artifact)
EARNINGS_CALL_CACHE_DIR = "call_transcripts"

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

# HuggingFace backbone: clean S&P 500 earnings-call transcripts 2005-2025 (MIT license,
# 33k+ transcripts / 685 companies, full verbatim `content` + speaker-segmented
# `structured_content`). Downloaded ONCE as a single ~1.8 GB parquet, cached under the
# call_transcripts dir; the Motley Fool crawl then only fills the recent gap past its cut.
HF_TRANSCRIPTS_DATASET = "kurry/sp500_earnings_transcripts"
HF_TRANSCRIPTS_PARQUET_URL = (
    "https://huggingface.co/datasets/kurry/sp500_earnings_transcripts/"
    "resolve/main/parquet_files/part-0.parquet")
HF_TRANSCRIPTS_CACHE = "hf_sp500_transcripts.parquet"
# The HF backbone is a ONE-TIME historical load (2005 .. ~2025Q1). Once earnings_call_sections
# already spans that range, re-scanning the 1.8 GB parquet only to find every (ticker, quarter)
# already ingested is pure waste (minutes of "nothing happens"). So ingest_hf_transcripts skips the
# scan when the table's quarter coverage reaches back to EARLY and forward to LATE. Quarters are
# fixed-width "YYYYQN", so a plain string MIN/MAX compares chronologically.
HF_BACKBONE_EARLY_QUARTER = "2005Q4"   # table min quarter must be <= this (deep history is present)
HF_BACKBONE_LATE_QUARTER = "2025Q1"    # table max quarter must be >= this (HF's ~2025 cut is reached)

# Roic AI earnings-call transcripts API — the PRIMARY recent-gap source (after the HF backbone,
# before Motley Fool): a clean JSON API covering ~2y of history on the FREE tier (5 req/min). Auth
# is the `apikey` QUERY param (not a header). `list` returns the available (year, quarter, date) per
# ticker; `transcript` returns {symbol, year, quarter, date, content} for one fiscal quarter.
ROIC_EARNINGS_LIST_URL = "https://api.roic.ai/v2/company/earnings-calls/list/{ticker}"
ROIC_EARNINGS_TRANSCRIPT_URL = "https://api.roic.ai/v2/company/earnings-calls/transcript/{ticker}"
ROIC_REQUEST_PAUSE = 12.5              # free tier = 5 req/min -> >= 12s between calls


EARNINGS_CALL_EMBED_MODEL = "text-embedding-3-small"     # cheap, 1536-dim; cost-efficient default
# per-turn `tag` values in EARNINGS_CALL_EMBEDDING_TABLE
EARNINGS_CALL_TAG_QUESTION = "question"      # a sell-side analyst turn (asks)
EARNINGS_CALL_TAG_ANSWER = "answer"          # a management turn answering the current question
EARNINGS_CALL_TAG_PREPARED = "prepared"      # a prepared-remarks (scripted) management turn
# Sections we score for tone (the high-signal prose); 'participants'/'full' are skipped
# for KPIs ('full' stays in the sections table as a format-proof fallback).
EARNINGS_CALL_SCORED_TAGS = ("prepared_remarks", "qa")

# FinBERT-tone: finance-domain tone classifier (positive / neutral / negative),
# trained on analyst reports & earnings text. ~440MB, runs locally on GPU (fits 6GB)
# or CPU; free (HuggingFace). Sections longer than the 512-token BERT window are
# chunked and length-weighted (see src/utils/nlp_sentiment.py).
FINBERT_TONE_MODEL = "yiyanghkust/finbert-tone"
FINBERT_MAX_TOKENS = 512

# --------------------------------------------------------------------------- #
# SEC footnote NARRATIVE (`notes_text`) -> risk/compliance features (data_aggregate).
# The raw high-signal TextBlocks are embedded (OpenAI) + NLP-scored into per-filing,
# per-theme rows. These tables ARE extracted and populated, but nothing in the cube reads
# them yet: the module that was meant to turn them into peer-relative features (narrative
# drift, risk-anchor similarity, tone/litigious density, disclosure-length dynamics) was
# never wired into any panel and has been removed, so no cube consumer exists today.
# --------------------------------------------------------------------------- #       
NOTES_EMBED_MODEL = "text-embedding-3-small"       # cheap, 1536-dim (shared with the earnings-call layer)
# risk/compliance THEME <- the footnote TextBlock tags that carry it (see fetch_financial_notes
# `_NOTES_TEXT_TAGS`). Drift/tone/length are tracked per tag and aggregated to the theme.
NOTES_THEME_TAGS: dict[str, tuple[str, ...]] = {
    "litigation": ("CommitmentsAndContingenciesDisclosureTextBlock",
                   "LegalMattersAndContingenciesTextBlock"),
    "going_concern": ("SubstantialDoubtAboutGoingConcernTextBlock",),
    "revenue_rec": ("RevenueFromContractWithCustomerTextBlock",
                    "RevenueRecognitionPolicyTextBlock", "RevenueRecognitionTextBlock"),
    # `UseOfEstimates` intentionally EXCLUDED: it is mostly a canned boilerplate paragraph and its
    # apparent drift is dominated by filers re-tagging content (e.g. ASC 606) -> not a risk signal.
    "critical_estimates": ("SignificantAccountingPoliciesTextBlock",
                           "OrganizationConsolidationAndPresentationOfFinancialStatements"
                           "DisclosureAndSignificantAccountingPoliciesTextBlock"),
    "concentration": ("ConcentrationRiskDisclosureTextBlock",),
}

# Named RISK / COMPLIANCE archetypes: each note embedding is scored by cosine to these anchor
# phrases (feature B) -> "how close is this disclosure to a known risk pattern", trackable over time.
NOTES_RISK_ANCHORS: dict[str, str] = {
    "litigation_loss": ("It is probable that the company will incur a material adverse loss from "
                        "pending litigation, and it recorded a charge or accrual for legal "
                        "settlements, damages, fines or penalties."),
    "regulatory_action": ("The company is subject to a government or regulatory investigation, "
                          "subpoena, consent decree, or enforcement action alleging violations."),
    "going_concern": ("There is substantial doubt about the company's ability to continue as a "
                      "going concern due to recurring losses and liquidity problems."),
    "covenant_breach": ("The company was not in compliance with its debt covenants and obtained a "
                        "waiver or amendment from its lenders to avoid default."),
    "impairment": ("The company recognized a goodwill or long-lived asset impairment charge because "
                   "expected future cash flows and fair value declined."),
    "control_weakness": ("A material weakness was identified in the company's internal control over "
                         "financial reporting."),
    "restatement": ("The company restated previously issued financial statements to correct a "
                    "material misstatement or accounting error."),
    "customer_concentration": ("A substantial portion of the company's revenue or credit exposure is "
                               "concentrated in a single large customer, counterparty, supplier or region."),
}

# Super investors TODO: in config
SUPERINVESTORS_JSON = "superinvestors/superinvestors.json"

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
# MACRO / MARKET series registry -- everything in `prices_macro`               #
# --------------------------------------------------------------------------- #
# ONE registry for the single long table (date, ticker, close). It replaced two wide
# tables (`macro`, FRED features on a 16y window; `macro_asset_prices`, allocation legs
# on 31y) that double-stored yield_10y and vix from two source paths at two depths, plus
# the non-equity rows that used to sit in `prices`. The invariant is now: every series
# exists exactly ONCE, from exactly one source. Breaking it is what made "which gold is
# this?" a real question.
#
# yfinance symbol -> series name. CLOSE only: `fetch_macro` calls `download_ohlcv` and
# drops OHLV+volume, so nothing here ever reaches the `prices` table (which is the equity
# universe and nothing else). Auto-adjusted, so each price leg is a total-return proxy.
#   SPY   = S&P 500 total return (since 1993)     ^VIX = CBOE VIX (since 1990)
#   CL=F  = WTI front future                      GC=F = COMEX gold front future (2000)
#   XLE   = Energy Select SPDR (1998), the "commodity via ENERGY EQUITIES" leg (no futures):
#           the rate/inflation-shock diversifier that was +~60% in the 2022 selloff
MACRO_PRICE_SERIES = {
    "SPY": "equity_tr",
    "^VIX": "vix",
    "CL=F": "oil",
    "GC=F": "gold",
    "XLE": "energy",
}

# FRED series id -> series name. LEVELS ONLY; the spreads below are derived from these.
# No DGS3MO: it is the coupon-equivalent quote of the same 3-month bill as DTB3, so the
# pipeline was fetching one instrument twice. cash_rate (DTB3) is the survivor -- it has a
# real consumer (allocation.py's cash leg) and drives the freshness gate.
# FRED no longer serves a broad daily S&P (SP500 is license-truncated to ~10y) or ANY gold
# series (the London fixes were removed ~2025), which is why those legs are yfinance above.
MACRO_FRED_SERIES = {
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",           # -> bond_10y_tr
    "DGS30": "yield_30y",
    "DTB3": "cash_rate",            # 3-month T-bill secondary market rate (cash leg)
    "BAA10Y": "baa_credit_spread",  # Moody's Baa over 10Y: one consistently-defined series
    "T10YIE": "breakeven_10y",      # 10Y breakeven inflation (since 2003)
    # FX from FRED, not Yahoo's USDEUR=X: DEXUSEU starts 1999-01 (the euro's own first
    # quote) where Yahoo starts 2003-12, and it is already quoted USD per EUR -- the
    # convention every consumer uses -- so no reciprocal to invert on ingest. Yahoo also
    # carried stale 2008 bars (2008-12-08 read 1.49 against a real 1.29).
    # COST: DEXUSEU rides the WEEKLY H.10 release, so FX trails the calendar by up to a
    # week where Yahoo was same-day. Hence its absence from MACRO_CORE_LEVEL_SERIES -- it
    # must not hold the freshness gate open -- and the newest ~3 trading days carry no FX.
    # Consumers see NaN there (never a stale ffill), which is the safe direction.
    "DEXUSEU": "fx_usdeur",
}
# Derived spread -> (minuend, subtrahend). FRED's own T10Y2Y IS DGS10-DGS2, so deriving it
# is numerically identical; deriving BOTH is what makes every FRED leg a same-cadence level
# and retires the freshness bug where a same-day-publishing computed spread marked the table
# current while the 1-BDay-lagged level block was still stale.
# CAVEAT: yield_curve_10y3m now differs from FRED's T10Y3M by the ~5bp discount-vs-coupon
# basis between DTB3 and DGS3MO. Free here -- the series swings hundreds of bp and has no
# consumer; if it gains one it will be as a CHANGE, where a constant offset differences out.
MACRO_SPREAD_SERIES = {
    "yield_curve_10y2y": ("yield_10y", "yield_2y"),
    "yield_curve_10y3m": ("yield_10y", "cash_rate"),
}
# Reconstructed 10Y total-return index + its maturity assumption (build_bond_total_return).
MACRO_BOND_TR_SERIES = "bond_10y_tr"
MACRO_BOND_MATURITY_YEARS = 10
# CORE daily level series the freshness gate keys on (all lag ~1 business day). Judging
# freshness on the overall max let a fast series mask a stale level block.
MACRO_CORE_LEVEL_SERIES = ("equity_tr", "yield_10y", "cash_rate", "vix")

# The market series: the cube's beta/epsilon reference and every sleeve's benchmark. Named,
# not configured -- it identifies a row in `prices_macro`, not a tunable.
MACRO_MARKET_SERIES = "equity_tr"
# Cube factor-panel column -> prices_macro series. The panel KEYS are preserved from when
# these came out of `prices` via cube_part_market, so no beta/feature name changes
# downstream; only USD/EUR actually remaps. `energy` is deliberately absent: wiring it in
# would add a factor and a beta column, which is a modelling decision, not a refactor.
MACRO_CUBE_FACTORS = {"oil": "oil", "gold": "gold", "USD/EUR": "fx_usdeur"}

# Macro level -> daily-change factor name. ONLY daily-moving series belong here.
# NOTE: cpi_yoy_pct (monthly) and fed_balance_sheet (weekly) are deliberately
# EXCLUDED -- their daily change is ~always zero. Inflation risk is captured by
# the daily breakeven instead.
DAILY_MACRO_LEVELS = {
    "yield_10y": "d_yield_10y",
    "yield_curve_10y2y": "d_yield_curve",
    "vix": "d_vix",
    "breakeven_10y": "d_breakeven_10y",
    "baa_credit_spread": "d_baa_credit_spread",
}

# Every series name written to `prices_macro`, derived from the registries above so there is
# no second list to drift. Used by the freshness gate, the tests and the sanity prints.
MACRO_ALL_SERIES = (tuple(MACRO_PRICE_SERIES.values()) + tuple(MACRO_FRED_SERIES.values())
                    + tuple(MACRO_SPREAD_SERIES) + (MACRO_BOND_TR_SERIES,))

# --------------------------------------------------------------------------- #
# Multi-asset trend (time-series-momentum) sleeve — StepTrendAssetClass output #
# --------------------------------------------------------------------------- #

# model artifact (params + vol-target calibration) under paths["MODELS_DIR"]
TREND_ASSET_MODEL_FILE = "trend_asset_model.json"

# --------------------------------------------------------------------------- #
# Daily prediction + live trading ledger (the `strat_prediction` DAG)          #
# --------------------------------------------------------------------------- #
# `Tables.predictions_latest` is LONG-format: one row per (as-of date, ticker, horizon, model), so
# each row can carry its OWN `predicts_for` -- the h30 and h90 predictions made on the same day
# target different future dates, which a wide pred_h30/pred_h60 layout cannot express.
# `model` values: one per ensemble member, plus these two aggregates.
PREDICTION_MODEL_ENSEMBLE = "ensemble"      # the per-horizon average of that horizon's members
PREDICTION_MODEL_BLENDED = "blended"        # the IR-weighted blend ACROSS horizons
# The trading ledger (`Tables.strategy`): one row per (trading day, sleeve, ticker) move, with the
# FIFO-matched entry/exit price and realized P&L of each round trip.
STRATEGY_SIDE_BUY = "BUY"
STRATEGY_SIDE_SELL = "SELL"

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
# NOTE: the old single global `CUBE_INCREMENTAL_WARMUP_TRADING_DAYS = 1400` is gone. The
# warm-up is PER PART now (`parts.py::CubePart.warmup_trading_days`, checked against each
# part's binding look-backs by tests/data_aggregate/test_part_registry.py), and the global
# had no remaining reader.

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
# DATA-PLAUSIBILITY BANDS                                                      #
# --------------------------------------------------------------------------- #
# Added after the source-table sanity audit (2026-07-28). Every band below was
# calibrated on the LIVE table, and each one separates a proven extraction defect
# from legitimate data — none of them clips a real value. See the per-constant
# notes for the observed evidence.

# `sharesOutstanding` for an S&P 500 name. 1.3% of fundamentals_history_sec rows sat
# outside this: 57 rows above 2e10 (ORCL 2012 stored 4.819e15 vs a true 4.819e9 —
# exactly 1e6x), 147 rows in 1..1e6 and 166 zeros. The real maximum in the table
# among plausible rows is ~1.6e10 (BAC/T era), so 2e10 is a safe ceiling and 1e6 a
# safe floor (no S&P 500 constituent has fewer than a million shares outstanding).
SHARES_OUTSTANDING_MIN = 1_000_000.0
SHARES_OUTSTANDING_MAX = 2e10

# Per-share figures. Diluted EPS outside ±10,000 is never real (BRK.A, the largest
# legitimate EPS in the universe, is ~4,000). 21 rows breached it, e.g. ICE 2016
# eps = 1.2e8 = the diluted SHARE COUNT captured into the EPS field.
EPS_ABS_MAX = 10_000.0
# Dividends per share: 19 rows exceeded 100 (ROK 3.88e6, STX 2.8e6 = the dollar
# dividend TOTAL, 1e6x the per-share figure). The largest real DPS here is ~35.
DIVIDEND_PER_SHARE_ABS_MAX = 1_000.0

# Derived ratios. `grossMargins` already had GROSS_MARGIN_MIN/MAX; these are its
# missing siblings, all blown up by a near-zero denominator rather than by a bad
# input: returnOnEquity reached 5.52e7 (168 rows |ROE|>10), debtToEquity 9.69e7
# (39 rows >100), operatingMargins -209..81.7 (63 rows), profitMargins -148.7..45.
# Bands are wide enough to keep genuine distress (negative equity, loss-making
# quarters) and only null arithmetic artefacts.
RETURN_ON_EQUITY_ABS_MAX = 10.0
DEBT_TO_EQUITY_ABS_MAX = 100.0
OPERATING_MARGIN_ABS_MAX = 5.0
PROFIT_MARGIN_ABS_MAX = 5.0
# A ratio is only trustworthy when its denominator is a meaningful fraction of the
# firm's scale; below this share of |totalRevenue| (or |totalAssets| for equity)
# the quotient is noise and the ratio is nulled instead of clipped.
RATIO_DENOMINATOR_MIN_FRACTION = 1e-3

# Balance-sheet scale check. Stub/registration-era filings (spin-off S-4s, a first
# 10-Q) carry an internally consistent but wrongly-scaled balance sheet — LUV 2011
# totalAssets 1.788e4 for a real $17.88bn, KMB 1.9e4, SW 108, AMCR 130. A real
# operating company never reports total assets smaller than this fraction of its
# own revenue, so the balance-sheet block is dropped for those rows.
BALANCE_SHEET_MIN_ASSETS_TO_REVENUE = 1e-3
# |TA - (TL + SE)| / |TA| above this means the totals did not come from one statement.
# Deliberately LOOSE. Two effects make a tight bound wrong here:
#   * filers split non-controlling interests either inside or outside
#     `stockholdersEquity`, so the identity is tested BOTH ways and the better fit wins
#     (adding NCI unconditionally breaks rows it should not -- ERIE's `minorityInterest`
#     is the Erie Insurance Exchange's equity, larger than Erie Indemnity's own assets);
#   * `_assemble_base` carries balance-sheet LEVELS forward up to 4 quarters, so two
#     totals on one row can legitimately come from different quarter-ends.
# Measured on the live table: 3,060 rows breach 2% but only 1,928 survive the NCI
# alternative, and of those 1,479 sit in 2-10% -- ffill drift, not a broken statement.
# The genuine breaks (SW 7.3e7, ARES 2.3e7, AMCR 5.3e5, LIN 1,613, ICE 24.8, ERIE 5.5)
# are orders of magnitude away, so 0.5 separates them with room to spare.
BALANCE_SHEET_IDENTITY_TOLERANCE = 0.5

# --------------------------------------------------------------------------- #
# FUNDAMENTALS_FACTS RECONCILIATION (edgartools per-filing pipeline)          #
# --------------------------------------------------------------------------- #
# Q4 = FY - (Q1+Q2+Q3) is a SAME-TAG arithmetic identity (four pieces of one filer's own
# reported number), not a three-concept accounting identity with genuine classification
# ambiguity like BALANCE_SHEET_IDENTITY_TOLERANCE above -- so it must be far tighter.
# 2% flat, chosen as the tighter half of `BALANCE_SHEET_IDENTITY_TOLERANCE`'s reasoning
# rather than inherited from anything: the `_TO_COMMON_TOL` in `fetch_fundamentals.py`
# this comment used to cite as its precedent does not exist, and neither does that module.
# A genuine reconciliation failure must be FLAGGED (`src/validate/` owns the checks),
# never silently corrected.
Q4_RECONCILIATION_TOLERANCE = 0.02

# A DERIVED Q4 (blank source_tag) is arithmetically forced to satisfy the identity above by
# construction, so `q4_reconciliation_gap` can never catch the case where the ANNUAL fact it
# was derived against is itself wrong (e.g. a dimensioned/non-consolidated slice slipping
# through as if it were the whole-company total). This is a SEPARATE, flag-only signal for
# that blind spot: how much of the derived Q4's own magnitude the fiscal-year total implies.
# Confirmed live: BA's FY2025 `OperatingIncomeLoss` (+$4.28B) against its own Q3-2025
# quarterly fact under the IDENTICAL tag (-$4.78B) derives a Q4 of +$8.78B -- 2.05x the FY
# total. Deliberately loose (not a rejection threshold): `_q4_is_coherent`'s own
# "arithmetically forced sign-flip" branch (fundamentals_periods.py) already accepts an
# UNCONDITIONAL magnitude for two confirmed-real cases pinned by a regression test
# (Citigroup FY2017, Corning FY2017), both of which this ratio would also flag at 2.78x and
# ~3x respectively -- so this constant can only ever ADVISE (severity="info" in
# `reconcile_fundamentals_facts`), never null or reject, matching that function's existing
# "surface, never silently correct" philosophy.
SIGNED_Q4_FY_DOMINANCE_FLAG_RATIO = 1.5

# --------------------------------------------------------------------------- #
# EMBEDDING INPUT LIMITS                                                       #
# --------------------------------------------------------------------------- #
# text-embedding-3-small accepts 8,191 TOKENS. English prose runs ~3.6 chars per
# token, so ~29,000 chars is the real ceiling; 28,000 keeps a safety margin for
# token-dense text (tables, tickers). The previous 8,000-CHAR cap truncated 22.4%
# of prepared-remarks turns (max 74,550 chars), so the quarter-to-quarter drift
# feature only ever compared each turn's opening fragment.
EMBEDDING_MAX_CHARS = 28_000
# A turn shorter than this is boilerplate ("Thank you.", "Yes.") — 17,281 Q&A turns
# qualify. Embedding them and taking a cosine against the question is pure noise,
# so they are excluded from the coherence KPI (they stay in the cache).
EMBEDDING_MIN_TURN_CHARS = 30

# --------------------------------------------------------------------------- #
# HEADCOUNT CONTINUITY                                                         #
# --------------------------------------------------------------------------- #
# Employee counts come from 10-K PROSE, so a residue of mis-picked numbers survives
# every in-document heuristic. Headcount is a slow-moving series, which makes a
# ticker's own history the strongest remaining check: no real company multiplies or
# divides its workforce by five between two annual filings. The 2026-07 audit measured
# 6.3% of year-over-year transitions at >2x or <0.5x, and the 30-ticker verification
# caught CoStar picking up a "2.3 million" phrase (2,300,000) against a stored 1,155.
# The band is deliberately generous so a genuine transformative merger still passes;
# it is anchored on the MEDIAN of accepted values, so one bad reading cannot reject the
# correct ones that follow it.
HEADCOUNT_CONTINUITY_MIN = 0.2
HEADCOUNT_CONTINUITY_MAX = 5.0

# --------------------------------------------------------------------------- #
# FUNDAMENTALS QoQ DISCONTINUITY (flag, never auto-fix)                        #
# --------------------------------------------------------------------------- #
# Same shape/reasoning as HEADCOUNT_CONTINUITY_MIN/MAX above: a >5x or <0.2x quarter-
# over-quarter move is unusual enough to flag for review (a large M&A, a genuine
# one-off, or a mis-mapped concept/period), but NOT automatically wrong -- a real
# transformative event legitimately produces one. `reconcile_fundamentals_facts`
# reports it as a diagnostic; it never nulls or rescales the underlying value.
FUNDAMENTALS_DISCONTINUITY_MIN = 0.2
FUNDAMENTALS_DISCONTINUITY_MAX = 5.0

# --------------------------------------------------------------------------- #
# XBRL TAG-SWITCH LEDGER (flag, never auto-fix)                                #
# --------------------------------------------------------------------------- #
# A field's resolved `source_tag` CHANGING mid-history is normal and expected, so the mere
# switch is not the signal. Measured on the live `fundamentals_facts`, 84.4% of
# (ticker, field) pairs use exactly ONE tag across 15 years, and nearly every switch in the
# remaining 15.6% is a US-GAAP TAXONOMY MIGRATION that moved every filer in the same window:
# `leaseMaturity*` OperatingLeasesFutureMinimumPaymentsDue* -> LesseeOperatingLease-
# LiabilityPaymentsDue* (ASC 842, old tag through 2020-12-31 and new from 2019-03-31 across
# all tickers), `cashPeriodChange` (ASU 2016-18 restricted cash), `interestPaid` /
# `incomeTaxesPaid` (X -> XNet deprecations), `netChargeOffs` / `provisionForCreditLosses` /
# `allowanceCreditLosses` (CECL, all banks at once). Those are the SAME measure under a new
# element name and the series is continuous across them.
#
# What is NOT benign is a switch where the LEVEL jumps at the boundary: the two tags are then
# two different MEASURES spliced into one series, which fabricates a regime break for a
# cross-sectional model. Calibrated on DTE `shortTermDebt`, which shows both shapes: its
# benign fiscal-2015 -> 2016 switch moves the level $465M -> $499M (1.07x) while the harmful
# fiscal-2012 -> 2013 switch moves it $240M -> $694M (2.9x) because the second tag is the
# long-term-debt FOOTNOTE deduction row, not a balance-sheet line. 1.5 sits between the two
# with room on both sides.
TAG_SWITCH_LEVEL_BREAK_RATIO = 1.5
# Periods pooled either side of a boundary before comparing levels. Comparing the two
# BOUNDARY values alone is unusable on a volatile balance -- DTE's short-term borrowings
# legitimately swing $0 -> $1,131M quarter-over-quarter WITHIN one tag -- so each side is
# reduced to a median over up to this many periods (4 = one fiscal year).
TAG_SWITCH_BASELINE_PERIODS = 4
# Maximum hole (in days) between the end of one tag era and the start of the next before the
# level comparison is abandoned. Across a longer gap the two levels are separated by missing
# periods as well as by the tag change, so a break cannot be attributed to the switch.
TAG_SWITCH_MAX_BOUNDARY_GAP_DAYS = 100

# Say-on-pay support below this is dropped by `def14a_impute` (see
# `_drop_implausible_say_on_pay`). Real votes cluster 0.85-0.99; the 2026-07 audit found
# 125 of 4,785 values (2.6%) under 0.60, steady at 1-4% every year since 2011, with
# spot-checks proving them wrong (JPM 2023 stored 0.31 against ~89% actual, SPG 2024
# 0.111 against ~93%, INTC 2023 0.34). Set at 0.50 rather than 0.60 to keep the genuine
# shareholder revolts, which do reach the low 50s, while clearing the clear errors. NOTE
# the field holds a FRACTION (0-1) despite the `_pct` name -- the live max is exactly 1.0.
SAY_ON_PAY_MIN_SUPPORT = 0.50

# Effective tax rate (`EffectiveIncomeTaxRateContinuingOperations`, 481 of 500 tickers).
# A RATIO, so a near-zero pre-tax income makes it explode: the raw field spans -56.6 to
# +43.1 while 89.4% of values sit inside 0..0.60 and the median is 0.218 (correct for
# post-TCJA US corporates). The band is asymmetric on purpose -- a genuine tax BENEFIT
# year (loss carry-back, valuation-allowance release) is real signal and goes negative,
# but not by 50x.
EFFECTIVE_TAX_RATE_MIN = -1.0
EFFECTIVE_TAX_RATE_MAX = 1.0

# `ppeNet` is rebuilt from (ppeGross - accumulatedDepreciation) when it falls below this
# share of that roll-forward. Utilities tag their rate base as
# `PublicUtilitiesPropertyPlantAndEquipment{Transmission,Distribution,GenerationOrProcessing}`
# and leave `PropertyPlantAndEquipmentNet` holding only a minor non-utility component --
# AEP reports $0.71bn there against $120bn of gross PP&E and $114bn of total assets, a 99%
# understatement of the asset base behind asset turnover, capex intensity and Altman Z.
# 0.20 is far below any real net/gross ratio (even a fully-depreciated base stays well
# above it), so a genuine old asset base is never rewritten.
PPE_NET_MIN_SHARE_OF_ROLLFORWARD = 0.20

# Diluted weighted-average shares may never fall below basic -- dilution only adds shares.
# 415 of 31,580 rows (1.31%) broke this because the diluted count arrived in a different
# UNIT (T 2010: basic 5.908e9 vs diluted 5,938; GLW: 1.568e9 vs 1,591; ICE: diluted 0),
# confirmed by `epsDiluted > epsBasic` on only 10.7% of them. The tolerance absorbs genuine
# rounding (14.2% of the violations are under 0.1% of basic) while catching the unit errors,
# which are all >= 90% shortfalls.
DILUTED_SHARES_MIN_SHARE_OF_BASIC = 0.99

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
