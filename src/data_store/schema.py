"""
THE table registry: every table described exactly once -- name, pk, incremental date column,
freshness cadence, SQL date columns, vector column, default read projection.

Replaces five drifted declaration sites: `schema_registry.py`, the `*_TABLE` literals in
constants.py, `DATA_FRESHNESS_SOURCES` (whose date_col duplicated TableSpec's),
`sources.py`'s projections (which existed twice and had diverged), and the `cube_part_*`
tables, which were in no registry at all -- that last one is why a whole second store
implementation (`PartStore`) had to exist.

Call sites pass the `Table` object, not a string, so the pk / date column / projection travel
with the name. Pure data: imports nothing from `data_store`, which is what lets `store.py`
and `ddl.py` import it at module level instead of lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from src.data_store.errors import UnknownTableError

# Table kinds. `reference` is a dimension, `extract` is raw fetched data, `aggregate` is a
# pipeline output, `part` is private plumbing between cube sub-steps.
# NOTE: this is DDL grouping only. It is NOT `parts.py`'s `PartKind`
# ("prices"/"market"/"features"/"targets"/"betas"), which is aggregation semantics and
# stays there -- it drives FEATURE_PARTS and part_status.py's `never_behind` set.
KIND_REFERENCE = "reference"
KIND_EXTRACT = "extract"
KIND_AGGREGATE = "aggregate"
KIND_PART = "part"

# Columns that are zero-padded string identifiers, never numeric -- forcing them to TEXT
# preserves leading zeros (SEC CIK "0000320193" would lose them as BIGINT).
_TEXT_IDENTIFIER_COLS = {"cik"}


@dataclass(frozen=True, slots=True)
class Table:
    """One database table, described once."""

    name: str
    pk: tuple[str, ...]
    kind: str = KIND_EXTRACT

    # The incremental time column: what `since=`/`until=` filter on and `max_date()`
    # reads by default. None for tables with no time grain (reference / embeddings).
    date_col: str | None = None
    # The entity column; None when the table has no per-ticker grain (macro).
    ticker_col: str | None = "ticker"
    # Columns to force to SQL DATE (they arrive as strings, and TIMESTAMP would be a lie).
    date_type_cols: tuple[str, ...] = ()

    # Collapse a scalar vector family (e0..eN) into one float8[] column.
    vector_col: str | None = None
    vector_prefix: str | None = None

    # False -> the owning step creates the table from its own frame's dtypes, and it is
    # EXCLUDED from sql/schema.sql. Also selects DROP-and-recreate over DELETE in
    # `replace` (see the warning on `is_unmanaged` below).
    managed: bool = True

    # Freshness cadence key (see constants.DATA_FRESHNESS_MAX_AGE_DAYS). None -> the table
    # is not freshness-checked.
    freshness: str | None = None
    # The column freshness measures, when it differs from `date_col`. Four tables publish
    # on a different clock than the period they describe -- for those, freshness must watch
    # WHEN the fact was filed, not the quarter it covers.
    freshness_date_col: str | None = None

    # Default projection for `load(project=True)`. A memory control: `sec13f_hr` is ~21.7M
    # rows and reading it unprojected OOM-killed the aggregation task. A consumer needing a
    # NARROWER set passes `columns=` rather than widening this.
    read_columns: tuple[str, ...] = ()
    # Of `read_columns`, those a builder uses only if present -- projecting a column the live
    # table lacks raises KeyError, so these are dropped quietly.
    optional_columns: frozenset[str] = field(default_factory=frozenset)

    def __str__(self) -> str:                     # so f"{Tables.prices}" is the name
        return self.name

    @property
    def is_unmanaged(self) -> bool:
        """True -> `replace` must DROP and recreate, not DELETE (see DataStore.replace)."""
        return not self.managed

    @property
    def freshness_col(self) -> str | None:
        return self.freshness_date_col or self.date_col


class Tables:
    """Every table, grouped by kind. Attribute name == table name throughout."""

    # ----------------------------------------------------------------- #
    # Reference / dimension                                             #
    # ----------------------------------------------------------------- #
    # The ticker universe: ticker, name, cik and GICS sector / industry_group /
    # sub_industry. Also the single source of truth for ticker->CIK resolution (see
    # sec_utils.load_cik_mapping); the old separate `cik_mapping` table was dropped as a
    # redundant duplicate.
    sp500_tickers = Table("sp500_tickers", ("ticker",), KIND_REFERENCE)

    # ----------------------------------------------------------------- #
    # Extract -- prices & market data                                   #
    # ----------------------------------------------------------------- #
    prices = Table("prices", ("ticker", "date"), date_col="date", freshness="daily")
    dividends = Table("prices_dividends", ("ticker", "date"), date_col="date")
    short_interest = Table(
        "sec_short_interest", ("ticker", "date"), date_col="date", freshness="daily",
        # short_interest_features: RegSHO short/total volume + reported short interest / ADV.
        # `short_interest` and `avg_daily_volume` are OPTIONAL -- the builder only adds
        # `days_to_cover` when BOTH are present, and the live table has only
        # date/ticker/short_volume/total_volume. Demanding them unconditionally is what
        # killed the read instead of degrading it.
        read_columns=("date", "ticker", "short_volume", "total_volume",
                      "short_interest", "avg_daily_volume"),
        optional_columns=frozenset({"short_interest", "avg_daily_volume"}))
    # SEC Fails-to-Deliver: settlement fails per ticker x date. Same grain as
    # short_interest but a separate table -> its semi-monthly, ~2-month-lagged files don't
    # pollute short_interest's global-max-date incremental; combined at the feature layer.
    sec_fails_to_deliver = Table(
        "sec_fails_to_deliver", ("ticker", "date"), date_col="date",
        date_type_cols=("date",), freshness="biweekly",
        read_columns=("date", "ticker", "fails_quantity"))
    # Unified macro / market series, LONG: one close per (series, date). Replaced the two
    # wide tables `macro` (FRED features, 16y) and `macro_asset_prices` (allocation legs,
    # 31y) -- which double-stored yield_10y and vix from two sources at two depths -- and
    # took over the non-equity tickers that used to sit in `prices`. That last move is what
    # lets `prices` be the equity universe and nothing else, which in turn is what let
    # `cube_part_market` (a firewall against macro tickers leaking into cross-sectional
    # ranks) disappear entirely.
    # `ticker` holds the SERIES name (equity_tr, vix, yield_10y, bond_10y_tr, ...), not the
    # source symbol, so the wide pivot reproduces the column vocabulary its consumers had.
    # Long, not wide: the legs start on different dates (gold 2000, breakeven 2003) and a
    # wide layout paid for that with a NaN block per series. See fetch_macro.py.
    prices_macro = Table("prices_macro", ("ticker", "date"), date_col="date",
                         date_type_cols=("date",), freshness="daily",
                         read_columns=("date", "ticker", "close"))
    cusip_ticker_map = Table("cusip_ticker_map", ("cusip",), ticker_col="ticker")

    # ----------------------------------------------------------------- #
    # Extract -- fundamentals                                           #
    # ----------------------------------------------------------------- #
    # PUBLICATION-EVENT grain: `as_of` is a FILING DATE, one row per date on which >=1
    # extracted value became newly public, each row a COMPLETE snapshot of every column's
    # latest-known value. Append-only -- an earlier row keeps its as-filed numbers forever,
    # which is where the no-leakage property lives. Exactly 69 columns, enumerated by
    # `Catalogue.history_columns`. See fundamentals/build_history.py.
    fundamentals_history = Table(
        "fundamentals_history", ("ticker", "as_of"), date_col="as_of",
        date_type_cols=("as_of", "fiscal_end", "amended_fiscal_end"),
        freshness="quarterly")
    # WHY a `fundamentals_history` cell is null, or why its value is off-basis. DENSE -- one
    # row per null-or-qualified cell at every publication event -- so the
    # zero-unexplained-nulls gate is a LEFT JOIN on (ticker, as_of, field) rather than a
    # reconstruction. The code vocabulary is closed and lives in fundamentals/reason_codes.py.
    #
    # Two PAYLOAD columns, both NULL for every code but the one that owns them:
    # `combined_into` names the destination field for `combined_into`, and `rejected_value`
    # carries the number a `failed_hard_guard` refused (plan-5b decision 46). The second one
    # exists because a nulled DERIVED value -- a TTM, a `derived_identity` total -- has no
    # fact row anywhere, so without it the rejected number is simply lost and "what did the
    # guard actually throw away?" becomes unanswerable after the fact. That question is the
    # whole lesson of the 745 correct rows an over-strict guard once nulled.
    fundamentals_reason_codes = Table(
        "fundamentals_reason_codes", ("ticker", "as_of", "field", "dc_code"),
        date_col="as_of", date_type_cols=("as_of",), freshness="quarterly")
    # Headcount, parsed from 10-K BODY TEXT. Its own table because the source is prose: in the
    # wide table one failed regex would fail the whole snapshot. Annual, so `as_of` is a 10-K
    # filing date and consumers forward-fill (`build_history.carry_latest_known`).
    fundamentals_employees = Table(
        "fundamentals_employees", ("ticker", "as_of"), date_col="as_of",
        date_type_cols=("as_of",), freshness="quarterly")
    # Accession-grain, amendment-aware fundamentals facts: one row per catalogue FIELD per
    # period per filing, resolved from the filer's own XBRL calculation linkbase (see
    # data_extract/utils/fundamentals/xbrl_linkbase.py) rather than from a priority-ordered
    # candidate-tag list.
    #
    # STRICTLY AS-FILED. Every row carries a number the filer actually tagged, on the
    # period shape it tagged it with; nothing here is derived. Q4 = FY - YTD9 and the YTD
    # decumulation happen in memory during the history build, so this table stays a
    # faithful record of what was published -- which is what makes the publication-event
    # grain and the no-leakage property of `fundamentals_history` provable rather than
    # asserted. ORIGINAL and AMENDED (10-K/A, 10-Q/A) filings coexist as separate rows and
    # are never overwritten, so "what was knowable on date D" is answerable by filtering
    # `filing_date <= D`.
    #
    # THE PK IS THE CALENDAR WINDOW, not the filer's fiscal LABEL. edgartools tags a 10-K's
    # current year and its first comparative with the SAME `fiscal_year`, so a PK keyed on
    # (fiscal_year, fiscal_period, duration_type) collided them and the upsert dedup silently
    # dropped one -- with frame order deciding which. Measured over 337,190 swept facts:
    # **18,604 rows (5.5%) lost, 16,340 of the collisions holding two DIFFERENT values, and
    # 1,522 ANNUAL facts** gone, concentrated in the non-calendar filers (KR 25%, COST 24.5%,
    # CSCO 21%, JNJ 20%, AAPL 18%). AAPL's FY2025 10-K kept FY2023 and FY2024 and dropped
    # FY2025 -- the FY fact that is Phase 4's PRIMARY Q4 input. Keyed on `period_end` the same
    # sweep loses 3 rows of 337,190 (0.001%), and those three are the one case
    # `_latest_per_window` already exists to collapse: one window tagged twice with a nudged
    # start date. `fiscal_year` / `fiscal_period` stay as PAYLOAD, which is what periods.py
    # already assumes -- "Nothing in this module reads `fiscal_period`"; every input there is
    # selected by its calendar window.
    fundamentals_facts = Table(
        "fundamentals_facts",
        ("ticker", "accession_number", "field", "duration_type", "period_end"),
        date_col="filing_date",
        date_type_cols=("filing_date", "period_start", "period_end", "period_of_report"),
        freshness="quarterly")
    earnings_surprises = Table("earnings_surprises", ("ticker", "earnings_date"),
                               date_col="earnings_date", freshness="quarterly")
    # SEC Financial Statement Data Sets (num/sub): curated pension facts per
    # company/tag/period-end (`ddate`) / duration (`qtrs`).
    pension_facts = Table("pension_facts", ("cik", "tag", "ddate", "qtrs"),
                          date_col="ddate", date_type_cols=("ddate", "filed"),
                          freshness="quarterly", freshness_date_col="filed")
    # SEC Financial Statement AND NOTES Data Sets -- footnote NUMERIC facts (consolidated /
    # undimensioned, curated tag set: PBO, plan assets, funded status, service cost,
    # employer contributions, discount rate). Grain = one fact per filing (`adsh`) / tag /
    # period-end (`ddate`) / duration (`qtrs`).
    notes_num = Table("notes_num", ("adsh", "tag", "ddate", "qtrs"), date_col="ddate",
                      date_type_cols=("ddate", "filed"), freshness="biweekly",
                      freshness_date_col="filed")
    # SEC notes NARRATIVE TEXT blocks (high-signal notes only), stored raw for later
    # embedding / sentiment. Same grain as notes_num; `value` is the text.
    notes_text = Table("notes_text", ("adsh", "tag", "ddate", "qtrs"), date_col="ddate",
                       date_type_cols=("ddate", "filed"), freshness="biweekly",
                       freshness_date_col="filed")
    # NOTE: `employees_history` was RETIRED, and so was the `fundamentals_history."employees"`
    # column that briefly replaced it. Headcount now has its own `fundamentals_employees`
    # table above (decision 35) -- one producer, `fundamentals_employees.py`, parsing the
    # 10-K prose inside the walk the fundamentals fetch already performs.

    # ----------------------------------------------------------------- #
    # Extract -- fundamentals (Sharadar)                                #
    # ----------------------------------------------------------------- #
    # Sharadar SF1, ALL 112 COLUMNS AS DELIVERED. Stored vendor-shaped and unmapped on
    # purpose: the repo-camelCase field map is a separate layer, and a mapping mistake must
    # be re-derivable without refetching (decision D7).
    #
    # `dimension` is in the PK because the same (ticker, date) is published on three
    # as-reported bases -- ARQ (discrete quarter), ARY (annual), ART (trailing twelve
    # months). Only the AR* three are ever written: MRQ/MRY/MRT restate in place, so their
    # rows mutate under an unchanged key and `diff_against_stored` could not tell an
    # amendment from a bug (D8).
    #
    # ⚠ `date` HERE IS THE FILING DATE, not the period end -- that is `reportperiod`. The
    # Sharadar DIRECT channel names it `date`; Nasdaq Data Link calls the same column
    # `datekey`. Both are in the PK, so the two channels are not interchangeable.
    #
    # `read_columns` is REQUIRED: 112 columns x 3 dimensions is the widest extract table in
    # the schema. The projection is the 7 identifiers plus the line items the field map
    # consumes, and deliberately EXCLUDES the 35 vendor-computed ratios (`pe`, `roe`, `de`,
    # `ev`, `marketcap`, ...) and the 8 `*usd` conversions -- those live in the raw table
    # only and never reach `fundamentals_history` (D21). A consumer needing one passes
    # `columns=` explicitly.
    sharadar_fundamentals = Table(
        "fundamentals_sharadar", ("ticker", "dimension", "date", "reportperiod"),
        date_col="date",
        date_type_cols=("date", "reportperiod", "calendardate", "lastupdated"),
        freshness="quarterly",
        read_columns=(
            "ticker", "dimension", "calendardate", "date", "reportperiod", "fiscalperiod",
            "lastupdated",
            # income statement
            "revenue", "cor", "gp", "opex", "sgna", "rnd", "opinc", "intexp", "ebit",
            "ebitda", "ebt", "taxexp", "consolinc", "netincnci", "netinc", "prefdivis",
            "netinccmn", "netincdis", "eps", "epsdil", "dps",
            # share counts
            "shareswa", "shareswadil", "sharesbas", "sharefactor",
            # balance sheet
            "assets", "assetsc", "assetsnc", "cashneq", "investments", "investmentsc",
            "investmentsnc", "receivables", "inventory", "intangibles", "ppnenet",
            "taxassets", "liabilities", "liabilitiesc", "liabilitiesnc", "debt", "debtc",
            "debtnc", "deferredrev", "payables", "deposits", "taxliabilities", "equity",
            "retearn", "accoci",
            # cash flow
            "ncfo", "depamor", "sbcomp", "ncfi", "capex", "ncfbus", "ncfinv", "ncff",
            "ncfcommon", "ncfdebt", "ncfdiv", "ncfx", "ncf", "fcf",
        ))
    # Sharadar's own ticker dimension, filtered to `table=fundamentals` (17,826 rows
    # measured 2026-08-26). Kept vendor-shaped and SEPARATE from `sp500_tickers`: this one
    # carries `permaticker` -- Sharadar's stable entity id, which survives a ticker change
    # and is NOT a column in SF1 -- plus `currency`, which the fundamentals fetch reads to
    # enforce the USD assertion (D20). There is no `cik` column in any Sharadar table.
    #
    # ⚠ `table` is a COLUMN here (the Sharadar table the row describes) and it is in the PK.
    # It is also SQL-reserved-ish, so every call must pass this `Table` object rather than a
    # string literal, and the store must quote it.
    sharadar_tickers = Table(
        "sharadar_tickers", ("table", "permaticker", "ticker"), KIND_REFERENCE,
        date_type_cols=("firstadded", "firstpricedate", "lastpricedate", "firstquarter",
                        "lastquarter", "lastupdated"),
        read_columns=("table", "permaticker", "ticker", "name", "exchange", "isdelisted",
                      "category", "currency", "sector", "industry", "siccode", "location",
                      "firstquarter", "lastquarter"))
    # Corporate actions: dividends, splits, spinoffs, acquisitions, name/SIC changes and
    # `relation` (the link from a common share to a sibling security).
    #
    # ⚠ `contraticker` -- the OTHER side of the action -- IS A PK MEMBER, and the plan's
    # original (date, ticker, name, action) was not unique. A company links several sibling
    # securities on one day: measured over 1,927 live rows, GS emitted three `relation` rows
    # dated 2026-08-25 differing only in `contraticker` (GS-PD / GS-PA / GS-PC, its
    # preferred series D / A / C), and JPM eight more. That PK collapsed 11 rows into 3 on
    # upsert, silently. With `contraticker` the same 1,927 rows have zero duplicates.
    # `contraticker` is the literal string "N/A" when there is no other side, never NULL --
    # which is what makes it safe in a PK, and why the reader must not let pandas coerce
    # "N/A" to NaN (see fetch_sharadar.py's `keep_default_na=False`).
    sharadar_actions = Table(
        "sharadar_actions", ("date", "ticker", "action", "contraticker"), date_col="date",
        date_type_cols=("date",),
        read_columns=("date", "action", "ticker", "name", "value", "contraticker",
                      "contraname"))
    # S&P 500 index membership events (added / removed / historical), back to 1992. Ingested
    # for the survivorship-bias fix, which is a SEPARATE task: `src/utils/universe.py` still
    # resolves the universe from `sp500_tickers` and is deliberately not touched here (D27).
    sharadar_sp500 = Table(
        "sharadar_sp500", ("date", "ticker", "action"), date_col="date",
        date_type_cols=("date",),
        read_columns=("date", "action", "ticker", "name", "contraticker", "contraname",
                      "note"))

    # ----------------------------------------------------------------- #
    # Extract -- ownership & institutional                              #
    # ----------------------------------------------------------------- #
    # Renamed from `institutional_holdings` to match the form-dispatch registry's logical
    # name for 13F-HR. 13F is an all-filers pull walked by filing date (fetch_13f.py), so the
    # grain stays one row per manager x security x period -- no accession_number.
    sec13f_hr = Table(
        "sec13f_hr", ("cik", "period", "ticker", "cusip"), date_col="period",
        freshness="quarterly",
        # institutional_features + superinvestor_features. THE reason projections exist:
        # ~21.7M rows.
        read_columns=("cik", "period", "ticker", "shares", "value_usd",
                      "call_value", "put_value", "filing_date"),
        # institutional_features zero-fills the option legs when they are absent
        optional_columns=frozenset({"call_value", "put_value", "filing_date"}))
    # SEC Insider Transactions Data Sets (Forms 3/4/5): one row per reported transaction
    # (non-derivative + derivative), keyed by accession + table + SK.
    insider_transactions = Table(
        "insider_transactions",
        ("accession_number", "security_type", "transaction_sk"),
        date_col="transaction_date",
        date_type_cols=("transaction_date", "filing_date", "period_of_report"),
        freshness="quarterly", freshness_date_col="filing_date",
        read_columns=("ticker", "filing_date", "transaction_code", "value_usd"))
    # SC 13D activist filings + amendments: one row PER REPORTING PERSON per filing, keyed
    # (ticker, accession, rp_seq) -- a single 13D can have multiple co-filers (e.g. a fund
    # + its GP), and `rp_seq` is used rather than CIK since a reporting person without an
    # assigned CIK is common. Numeric ownership fields are NULL, not 0, whenever
    # `has_structured_data` is false: SC 13D has no XBRL-grade schema, the parser defaults
    # those fields to 0 when it can't find structured content, and publishing that as if
    # real would claim false 0% stakes.
    sec_13d = Table("sec_13d", ("ticker", "accession_number", "rp_seq"),
                    date_col="filing_date",
                    date_type_cols=("filing_date", "date_of_event"))
    # Item 5(c) 60-day transaction log: one row PER DISCLOSED TRADE, keyed (ticker,
    # accession, trade_seq) -- an independent grain from `sec_13d` (no rp_seq
    # relationship). Parsed from each filing's "TRADING DATA" exhibit; the exhibit number
    # varies by filer so it is identified by table content ("Trade Date" header).
    sec_13d_transactions = Table(
        "sec_13d_transactions", ("ticker", "accession_number", "trade_seq"),
        date_col="filing_date", date_type_cols=("filing_date", "trade_date"))

    # ----------------------------------------------------------------- #
    # Extract -- governance (DEF 14A) & events                          #
    # ----------------------------------------------------------------- #
    def14a_llm = Table("def14a_llm", ("ticker", "accession_number"), date_col="as_of",
                       freshness="yearly")
    # Deterministic complement to def14a_llm: structured DEF 14A data via edgartools' typed
    # ProxyStatement (SEC XBRL ECD taxonomy + deterministic HTML-table parsing), zero LLM
    # cost. Filing-level row + four one-to-many detail tables below.
    def14a_edgar = Table("sec_def14a", ("ticker", "accession_number"),
                         date_col="filing_date",
                         date_type_cols=("filing_date", "period_of_report"))
    # Summary Compensation Table: one row per NEO per fiscal year (edgartools typically
    # recovers 3 years per filing) -- richer multi-year history than def14a_llm's single
    # most-recent-year CEO fields.
    def14a_edgar_executive_comp = Table(
        "sec_def14a_executive_comp",
        ("ticker", "accession_number", "name", "year"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # Non-employee Director Compensation Table (Item 402(k)): one row per director/filing.
    def14a_edgar_director_comp = Table(
        "sec_def14a_director_comp", ("ticker", "accession_number", "name"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # Beneficial ownership table (5%+ holders + insiders, Reg S-K Item 403).
    def14a_edgar_ownership = Table(
        "sec_def14a_ownership",
        ("ticker", "accession_number", "holder_name", "holder_type"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # Ballot items: one row per proposal, carrying the BOARD's recommendation (not the
    # shareholder vote OUTCOME -- see fetch_def14a_edgar.py) + classified type.
    def14a_edgar_votes = Table(
        "sec_def14a_votes", ("ticker", "accession_number", "proposal_number"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # 8-K events: one row per ITEM CODE of a filing, keyed (ticker, accession, item) -- an
    # 8-K reports 1..n items and ~75% report more than one. `item` is in the PK because
    # keying on the accession alone made every extra item upsert onto the same row, silently
    # keeping only the last (95,785 accessions stored for 196,875 item rows built).
    # `has_earnings`/`has_press_release` come from edgartools' typed `CurrentReport`
    # (best-effort -- NaN, not False, when that parse fails).
    sec_8k = Table("sec_8k", ("ticker", "accession_number", "item"), date_col="filing_date",
                   date_type_cols=("filing_date", "period_of_report"))
    # 10-K Item 1A (Risk Factors) + Item 7 (MD&A) raw text; one row per
    # (ticker, accession, section). Feeds the embedding/drift feature layer.
    filing_risk_text = Table("sec_filing_text",
                             ("ticker", "accession_number", "section"), date_col="filed",
                             date_type_cols=("filed", "period_of_report"))

    # ----------------------------------------------------------------- #
    # Extract -- behavioral / text / embeddings                         #
    # ----------------------------------------------------------------- #
    google_trends = Table("google_trends", ("ticker", "date"), date_col="date",
                          freshness="weekly",
                          read_columns=("date", "ticker", "search_interest"))
    wiki_pageviews = Table("wiki_pageviews", ("ticker", "date"), date_col="date",
                           freshness="daily",
                           read_columns=("date", "ticker", "pageviews"))
    # FREE earnings-call transcripts (Motley Fool), split into high-signal sections
    # (prepared_remarks / qa / participants). One row per ticker / fiscal quarter /
    # section; `as_of` = call date, `text` = the prose. NOT projected: `text` IS the
    # payload the incremental scoring pass needs.
    earnings_call_sections = Table("earnings_call_sections", ("ticker", "quarter", "tag"),
                                  date_col="as_of", date_type_cols=("as_of",),
                                  freshness="quarterly")
    # Per-call sentiment / text-metrics cache (FinBERT-tone + LM lexicon), one row per
    # ticker / fiscal quarter / section. Holds the EXPENSIVE, call-intrinsic scores (tone
    # probs, word count, uncertainty ratio) so the GPU pass runs once; the cross-call KPIs
    # are derived cheaply at cube-build time. Same grain as sections.
    earnings_call_sentiment = Table("earnings_call_sentiment",
                                    ("ticker", "quarter", "tag"), date_col="as_of",
                                    date_type_cols=("as_of",))
    # OpenAI earnings-call embeddings: one row PER SPEAKER TURN, each with its own
    # float8[] `embedding` + raw `text` + `person` + `tag` + `exchange_idx` (links a
    # question to its answer turns) + model/run stamp + `as_of` (call date). The
    # Q&A-coherence + QoQ-distance cube features are DERIVED from these turns at build
    # time. The projection omits `text` -- the KPI pass never reads it, and it is the bulk
    # of the table.
    earning_calls_embedding = Table(
        "earning_calls_embedding", ("ticker", "quarter", "seq"),
        date_col=None, vector_col="embedding", vector_prefix="e",
        read_columns=("ticker", "quarter", "as_of", "section", "tag", "exchange_idx",
                      "embedding"))
    # OpenAI embeddings of SEC footnote NARRATIVE (`notes_text`): one mean-pooled vector
    # PER (ticker, filing `adsh`, TextBlock `tag`), with its `theme`, `filed`/`ddate` for
    # point-in-time ordering, `txtlen` and chunk count. Populated by extraction but NOT yet
    # consumed by the cube: the narrative-drift builder was never wired into a panel, so
    # these rows currently have no downstream reader.
    notes_embedding = Table("notes_embedding", ("ticker", "adsh", "tag"), date_col=None,
                            date_type_cols=("as_of", "ddate"), vector_col="embedding",
                            vector_prefix="e")
    ticker_descriptions = Table("ticker_descriptions", ("ticker",), date_col=None)
    ticker_embeddings = Table("ticker_embeddings", ("ticker",), date_col=None,
                              vector_col="embedding", vector_prefix="e")

    # ----------------------------------------------------------------- #
    # Aggregate -- pipeline outputs                                     #
    # ----------------------------------------------------------------- #
    cube = Table("cube", ("ticker", "date", "target_horizon"), KIND_AGGREGATE,
                 date_col="date")
    cube_signal = Table("cube_signal", ("ticker", "date"), KIND_AGGREGATE, date_col="date")
    predictions = Table("predictions", ("ticker", "date"), KIND_AGGREGATE, date_col="date")
    # LIVE predictions in LONG form, one row per (as-of date, ticker, horizon, model): each
    # row carries its OWN `predicts_for` (as-of + horizon business days), because the h30
    # and h90 predictions made on the same day are about different future dates. `model` is
    # an ensemble member name, or 'ensemble' (that horizon's member average) / 'blended'
    # (the IR-weighted blend across horizons). `predicted_at` is when the run produced the
    # row -- deliberately distinct from `date`, the as-of date of the features.
    predictions_latest = Table("predictions_latest",
                               ("date", "ticker", "horizon", "model"), KIND_AGGREGATE,
                               date_col="date",
                               date_type_cols=("date", "predicts_for"))
    # Multi-asset trend sleeve daily NET returns (one row per date, no ticker) -- a
    # directional cross-asset time-series-momentum book blended with the equity alpha +
    # SPY in the backtest.
    trend_asset_returns = Table("trend_asset_returns", ("date",), KIND_AGGREGATE,
                                date_col="date", ticker_col=None)
    # The TRADING LEDGER: one row per (trading day, sleeve, ticker) move the portfolio
    # would place, with its FIFO-matched entry/exit price and realized P&L. Upserted (not
    # replaced) so a BUY row written weeks ago gains its `price_sold` / `pnl` on the day
    # its position closes.
    strategy = Table("strategy", ("trading_day", "sleeve", "ticker"), KIND_AGGREGATE,
                     date_col="trading_day",
                     date_type_cols=("trading_day", "closed_on"))

    # ----------------------------------------------------------------- #
    # Validate -- the finding ledger                                    #
    # ----------------------------------------------------------------- #
    # One row per FINDING per RUN: the fundamentals validator's append-only queue (plan-5b
    # decision 42). `src/validate/` writes this and mutates nothing else -- the nightly
    # build of `fundamentals_facts` / `fundamentals_history` runs to completion whatever
    # lands here, because nothing gates (decision 45).
    #
    # `run_date` IS IN THE KEY, so a re-run appends rather than overwriting: "did this check
    # fire yesterday?" stays answerable, and a check whose threshold moved leaves both
    # verdicts on the record. What survives across runs is `finding_id`, a deterministic
    # hash of (check_name, ticker, field, period_key), so a finding keeps its identity even
    # though its rows do not -- which is what makes differencing two runs meaningful.
    #
    # `period_key` IS TEXT AND POLYMORPHIC, by grain: the `as_of` for a history-grain check,
    # the `period_end` for a facts-grain one, `''` for a ticker-level check
    # (`filing_continuity`), and a `start..end` range for a series-grain one
    # (`series_shape`). One key column rather than three nullable ones,
    # because a PK cannot contain a NULL in Postgres and a sentinel date would be a lie.
    #
    # `cluster_id` is a hash of (ticker, field) ALONE -- the DEFECT, of which every row here
    # is one witness. It is not in the PK and is not meant to be unique: MCD `capex` trips
    # nine checks over dozens of periods and that is ONE thing to fix. `run_id` records which
    # scope produced the row, so two runs can be differenced honestly. See
    # `fundamentals_check_run` below.
    #
    # NOTHING IS SUBTRACTED ON THE WAY IN. Every finding of every run is written, including
    # ones a human has already settled, because a ledger that suppresses rows makes its own
    # row count meaningless -- a drop would be ambiguous between "fixed" and "hidden". A
    # `wontfix` is recorded in `fundamentals_check_status` and applied when the report is
    # RENDERED, never when the row is written.
    #
    # The payload is decision 47's SELF-CONTAINED INVESTIGATION PACKET: identity, observed
    # vs expected, the full provenance the fact row carried, the EDGAR URL, and a
    # check-specific `detail` JSON. Deliberately denormalised -- a Tier-2/3 finding on a
    # DERIVED value (a TTM, a `derived_identity` total) has no single fact row to join back
    # to, so an identity-only row plus an on-demand join cannot reconstruct it at all.
    # `run_id` IS IN THE KEY, and that was learned the hard way rather than designed in.
    # Without it the key is (run_date, check_name, ticker, field, period_key), so TWO RUNS OF
    # DIFFERENT SCOPE ON ONE DAY collide on every ticker they share: a `-t MCD` run wrote 270
    # rows, a 54-ticker roster run an hour later upserted over 269 of them, and the first run
    # was left claiming 35 checks and one surviving finding. Every delta computed against it
    # would have been nonsense. Two runs that looked at different things must be able to
    # coexist, so the run is part of the row's identity.
    fundamentals_check = Table(
        "fundamentals_check",
        ("run_date", "run_id", "check_name", "ticker", "field", "period_key"),
        KIND_AGGREGATE, date_col="run_date",
        date_type_cols=("run_date", "as_of"))

    # One row per (run_id, check_name): WHAT THE RUN LOOKED AT, and what each check did with
    # it. Without this table a row-count drop in `fundamentals_check` is ambiguous between
    # "the fix worked" and "the second run was scoped to fewer tickers", so the whole
    # fix-then-measure loop rests on it.
    #
    # `run_id` is a hash of (run_date, tickers, fields, tiers); `scope_hash` is the same hash
    # WITHOUT the date. Two runs are COMPARABLE iff their `scope_hash` matches -- that is the
    # single equality test `ledger.comparable_runs` makes, rather than a fragile three-column
    # text comparison.
    #
    # The scope columns repeat on every check row. Denormalised deliberately: it matches this
    # repo's flat-table convention and keeps "what did this run cover, and did any check
    # abstain?" a single unprojected read of a table with ~35 rows per run.
    #
    # `abstained` and `over_ceiling` are STORED rather than recomputed on read. They are the
    # check-health gate the report renders ABOVE its rankings, and a gate that has to be
    # re-derived from a ceiling that has since moved would answer a different question than
    # the one the run asked.
    fundamentals_check_run = Table(
        "fundamentals_check_run", ("run_id", "check_name"),
        KIND_AGGREGATE, date_col="run_date", ticker_col=None,
        date_type_cols=("run_date",))

    # `wontfix`, keyed on `(cluster_id, check_name)` -- a TOLERANCE for one `(ticker, field)`
    # defect. The ONLY mutable state in the validator, and the replacement for the deleted
    # JSON register.
    #
    # `check_name` IS IN THE KEY and `''` means the WHOLE cluster. Keyed on `cluster_id`
    # alone a waiver is all-or-nothing: MCD `capex` retains two benign `peer_ratio` findings
    # on a documented blind spot, and tolerating those at cluster grain would also silence
    # the eight other checks still live on the same defect. Per-check is the narrowest
    # tolerance expressible, so it is the one stored.
    #
    # `open` and `settled` are NOT stored: they are DERIVED from the ledger, because a status
    # column that says `settled` while the check still fires is exactly the suppression list
    # the register became. The only thing a human can assert here is "I have looked at this,
    # it is real, and it is not worth repairing" -- and `findings_at_decision` makes even that
    # self-expiring: the entry REOPENS automatically the moment it grows past the size that
    # was actually assessed.
    #
    # Waiving every check still does NOT settle a cluster. Settlement additionally requires a
    # `fundamentals_check_fix` row that measurably reduced the queue; without that rule the
    # suppression list is simply reassembled one check at a time.
    fundamentals_check_status = Table(
        "fundamentals_check_status", ("cluster_id", "check_name"),
        KIND_AGGREGATE, date_col="decided_at",
        date_type_cols=("decided_at",))

    # An INTERVENTION, keyed `(cluster_id, run_id_after)`. A DIFFERENT KIND OF THING from the
    # table above: a fix is an EVENT that happened, and a waiver is a STATE that persists. So
    # this table is append-only and nothing here is ever revised -- two fixes of one cluster
    # are two rows, because the second did not un-happen the first.
    #
    # NO RENDERER MAY FILTER FINDINGS USING THIS TABLE. It records what was done and what it
    # measurably closed; it never subtracts a row from `fundamentals_check`. That separation
    # is the entire reason a fix is stored apart from a waiver, and it is what keeps a
    # row-count drop usable as proof.
    #
    # `run_id_after` is in the key because it is the run that PROVED the fix. Both runs must
    # share a `scope_hash` or the before/after counts are not a comparison at all -- the same
    # test `fundamentals_check_run` exists to make.
    #
    # `layer` is a CLOSED four-term vocabulary (`constants.FIX_LAYERS`) defined by what the
    # edit DOES, never by which file it lives in: `check` = the check was wrong;
    # `catalogue` = the field specification was wrong; `extraction` = any code that PRODUCES
    # a value (xbrl_linkbase, build_history, periods); `rows` = the code was already right
    # and the stored data was stale. Coarse grouping; `root_cause` carries the precision.
    #
    # `evidence` is JSON, never prose, and its required keys vary BY LAYER
    # (`constants.FIX_EVIDENCE_KEYS`): a `check` fix has no filing to cite, so demanding an
    # accession would force it to name an irrelevant one. Its evidence is the false-positive
    # population it was measured against.
    fundamentals_check_fix = Table(
        "fundamentals_check_fix", ("cluster_id", "run_id_after"),
        KIND_AGGREGATE, date_col="decided_at",
        date_type_cols=("decided_at",))

    # ----------------------------------------------------------------- #
    # Parts -- private plumbing between the cube sub-steps               #
    # ----------------------------------------------------------------- #
    # Rebuilt wholesale by their owning step, so each carries its OWN DDL inferred from the
    # frame it writes and is excluded from sql/schema.sql (`managed=False`). The pk and
    # date_col are still DECLARED: `date_col` is what `since=` and `max_date()` resolve
    # against, and the pk is the append/dedup grain.
    #
    # The BUILD-ORCHESTRATION facet of these tables -- CLI sub-command, warm-up trading
    # days, per-group binding look-backs -- deliberately stays in
    # `data_aggregate/utils/common/parts.py`. That is aggregation policy, not schema.
    # NO cube_part_market: it existed only to keep the market/commodity/FX tickers out of
    # the equity frame the cross-sectional ranks are computed on. Once those series live in
    # `prices_macro` and never in `prices`, there is nothing to separate -- StepCubeTarget
    # reads them straight from `prices_macro`, and the trading calendar (which this part
    # used to define) comes off cube_part_prices' own dates.
    cube_part_prices = Table("cube_part_prices", ("date", "ticker"), KIND_PART,
                             date_col="date", managed=False)
    # NOT ("date","ticker"): `_labels_to_long` (utils/assemble/cube.py) stamps
    # `target_horizon` per horizon and concatenates, so the grain is three-part. Declaring
    # the narrower key would let an upsert path collapse the horizons into one row -- and
    # there IS such a path: `copy_load` falls back to an upsert on the registry PK for
    # frames carrying list-valued cells.
    cube_part_targets = Table("cube_part_targets", ("date", "ticker", "target_horizon"),
                              KIND_PART, date_col="date", managed=False)
    cube_part_betas = Table("cube_part_betas", ("date", "ticker"), KIND_PART,
                            date_col="date", managed=False)
    cube_part_fundamentals = Table("cube_part_fundamentals", ("date", "ticker"), KIND_PART,
                                   date_col="date", managed=False)
    cube_part_momentum = Table("cube_part_momentum", ("date", "ticker"), KIND_PART,
                               date_col="date", managed=False)
    cube_part_text = Table("cube_part_text", ("date", "ticker"), KIND_PART,
                           date_col="date", managed=False)
    cube_part_extras = Table("cube_part_extras", ("date", "ticker"), KIND_PART,
                             date_col="date", managed=False)


def _collect() -> tuple[Table, ...]:
    """Every `Table` declared on `Tables`, in declaration order.

    Read off the class rather than hand-listed, so adding a table to `Tables` is the only
    step -- there is no second list to forget. `vars()` preserves definition order.
    """
    return tuple(v for v in vars(Tables).values() if isinstance(v, Table))


ALL: tuple[Table, ...] = _collect()
BY_NAME: dict[str, Table] = {t.name: t for t in ALL}

# `ddl.py` iterates this: the parts own their own DDL and must not reach sql/schema.sql.
MANAGED: tuple[Table, ...] = tuple(t for t in ALL if t.managed)
PARTS: tuple[Table, ...] = tuple(t for t in ALL if t.kind == KIND_PART)


def resolve(table: Table | str) -> Table:
    """Accept either a `Table` or its name and return the `Table`.

    The string form is not a back-compat shim and is expected to stay: `part_status`
    iterates names, `CubePart` exposes a name, and reflection returns strings.
    """
    if isinstance(table, Table):
        return table
    try:
        return BY_NAME[table]
    except KeyError:
        raise UnknownTableError(
            f"{table!r} is not in src/data_store/schema.py. Add a Table for it rather "
            f"than creating it implicitly.") from None


def name_of(table: Table | str) -> str:
    """The table name, without requiring registration.

    Used on the few paths that must tolerate an unregistered name -- e.g. dropping a
    superseded table during a migration, where insisting on a registry entry for something
    being deleted would be backwards.
    """
    return table.name if isinstance(table, Table) else str(table)


def by_kind(kind: str) -> tuple[Table, ...]:
    return tuple(t for t in ALL if t.kind == kind)


def freshness_tables() -> tuple[Table, ...]:
    """The tables declaring a refresh cadence, in registry order.

    Currently has NO caller: the gate that consumed it was removed. Kept as the single
    entry point for any future staleness check, so a new consumer reads the cadence off
    the registry instead of reintroducing a hand-maintained table list (the mistake
    `constants.DATA_FRESHNESS_SOURCES` made -- every label there equalled its own table
    name, duplicating `date_col` plus a cadence the spec already declared).
    """
    return tuple(t for t in ALL if t.freshness is not None)


def projection(table: Table | str,
               available: Sequence[str] | None) -> list[str] | None:
    """`table`'s read projection, narrowed to the columns that actually EXIST.

    `available` is the table's real column list; None means unknown, so project nothing and
    let the caller read in full. A REQUIRED column that is missing is still worth
    surfacing, so it is reported; an OPTIONAL one is dropped quietly, matching the
    builder's own `issubset` guard.

    Use `projection_report` instead when you want to log what was dropped.
    """
    
    cols, _, _ = projection_report(table, available)
    return cols


def projection_report(
    table: Table | str, available: Sequence[str] | None
) -> tuple[list[str] | None, list[str], list[str]]:
    """`projection` plus what was dropped, so the caller can log it at the right level."""
    spec = resolve(table)
    wanted = list(spec.read_columns)
    if not wanted or available is None:
        return (wanted or None), [], []
    have = set(available)
    keep = [c for c in wanted if c in have]
    missing = [c for c in wanted if c not in have]
    required_missing = [c for c in missing if c not in spec.optional_columns]
    optional_missing = [c for c in missing if c in spec.optional_columns]
    return (keep or None), required_missing, optional_missing
