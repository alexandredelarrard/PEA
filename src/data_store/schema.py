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
    dividends = Table("dividends", ("ticker", "date"), date_col="date")
    short_interest = Table(
        "short_interest", ("ticker", "date"), date_col="date", freshness="daily",
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
    fails_to_deliver = Table(
        "fails_to_deliver", ("ticker", "date"), date_col="date",
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
    fundamentals_history = Table(
        "fundamentals_history", ("ticker", "as_of"), date_col="as_of",
        date_type_cols=("as_of", "fiscal_end"), freshness="quarterly")
    # Accession-grain, amendment-aware raw fundamentals facts (edgartools per-filing XBRL
    # walk). One row per (ticker, accession, field, fiscal period, duration shape);
    # ORIGINAL and AMENDED (10-K/A, 10-Q/A) filings coexist as separate rows -- never
    # overwritten -- so `fundamentals_derive` can answer "what was known as of date D"
    # without ever exposing an amendment's value before its own filing date.
    # `fundamentals_history` above is DERIVED from this table.
    fundamentals_facts = Table(
        "fundamentals_facts",
        ("ticker", "accession_number", "field", "fiscal_year", "fiscal_period",
         "duration_type"),
        date_col="filing_date",
        date_type_cols=("filing_date", "period_start", "period_end"),
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
    # NOTE: `employees_history` was RETIRED. Employee headcount is a `fundamentals_facts`
    # field now (10-K body text, see fundamentals_employees.py) and is read as
    # fundamentals_history."employees".

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
    def14a_edgar = Table("def14a_edgar", ("ticker", "accession_number"),
                         date_col="filing_date",
                         date_type_cols=("filing_date", "period_of_report"))
    # Summary Compensation Table: one row per NEO per fiscal year (edgartools typically
    # recovers 3 years per filing) -- richer multi-year history than def14a_llm's single
    # most-recent-year CEO fields.
    def14a_edgar_executive_comp = Table(
        "def14a_edgar_executive_comp",
        ("ticker", "accession_number", "name", "year"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # Non-employee Director Compensation Table (Item 402(k)): one row per director/filing.
    def14a_edgar_director_comp = Table(
        "def14a_edgar_director_comp", ("ticker", "accession_number", "name"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # Beneficial ownership table (5%+ holders + insiders, Reg S-K Item 403).
    def14a_edgar_ownership = Table(
        "def14a_edgar_ownership",
        ("ticker", "accession_number", "holder_name", "holder_type"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # Ballot items: one row per proposal, carrying the BOARD's recommendation (not the
    # shareholder vote OUTCOME -- see fetch_def14a_edgar.py) + classified type.
    def14a_edgar_votes = Table(
        "def14a_edgar_votes", ("ticker", "accession_number", "proposal_number"),
        date_col="filing_date", date_type_cols=("filing_date",))
    # 8-K events: one row per 8-K filing. `items` is the raw comma-separated item-code
    # string (structured, ~100% fill); `has_earnings`/`has_press_release` come from
    # edgartools' typed `CurrentReport` (best-effort -- null, not False, when that fails).
    sec_8k = Table("sec_8k", ("ticker", "accession_number"), date_col="filing_date",
                   date_type_cols=("filing_date", "period_of_report"))
    # 10-K Item 1A (Risk Factors) + Item 7 (MD&A) raw text; one row per
    # (ticker, accession, section). Feeds the embedding/drift feature layer.
    filing_risk_text = Table("filing_risk_text",
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
