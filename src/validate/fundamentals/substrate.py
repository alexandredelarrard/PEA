"""
substrate.py  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
`Substrates`: every frame a check can read, loaded ONCE and passed down the tiers.

## Why this type exists at all

Phase 10 names it as the validator's efficiency risk in so many words: a validator that
re-reads its tables per check does 30-odd unprojected reads of `fundamentals_facts`, which is
~28M rows universe-wide. Handing every check the SAME already-loaded object makes that
structurally impossible instead of merely discouraged -- a check has no `store` to re-read
from, because it is never given one.

The second reason is testability. A check takes a `Substrates`, so `tests/validate/` builds
one from synthetic frames and exercises the entire validator with **no DB and no CLI**. That
is the bar the plan sets for every check: "a check that cannot be planted cannot be trusted."

## The four frames, and what each is for

  `history`  `fundamentals_history_sec` -- the wide publication-event table, 69 columns. Read
             ONLY by the six Tier-1 CONTRACT checks (`grain`, `column_contract`,
             `code_vocabulary`, `unexplained_null`, `pit_leak`, `coverage_universe`).
  `codes`    `fundamentals_reason_codes` -- dense, one row per null-or-qualified cell. The
             thing that makes a NULL checkable rather than merely visible.
  `facts`    `fundamentals_facts` -- accession-grain, strictly as-filed. EVERY VALUE AND
             COVERAGE CHECK, all three tiers.
  `employees` -- separate table, annual, read only by the coverage checks.

## Why the value checks all read `facts` now

Because a finding without an accession cannot be acted on. `fundamentals_history_sec` carries no
`accession_number` and no `cik`, and `Finding.edgar_url` is built from exactly that pair --
so every one of the 1,437 Tier-1 findings on the last history-based run had a NULL URL,
against 100% on Tier 3. An agent handed such a finding cannot open the filing that caused it,
which is the first move the triage loop requires.

The move costs nothing in evidence and gains some: measured on the live 54-ticker table, the
balance-sheet identity fails on the SAME seven filers either way, but facts exposes 4,763
testable statements against history's 3,229 (a filing carries comparatives, and each is a
separate published claim), so it finds 144 breaks where history found 64.

The SIX contract checks stay on history deliberately. They test properties `facts` does not
have -- a 69-column ordered contract (facts is long), a null CELL (in facts a missing fact is
an absent ROW), the reason-code vocabulary, and the no-leakage snapshot grain. Porting them
would delete them rather than relocate them, and they are the tripwires for the one defect
class that is genuinely history's own: a bug in `build_history`. ETN's 2012 row is the
specimen -- `totalLiabilities` of -$8,237,223,652 against `totalAssets` of $4,776,348, tagged
`derived_identity`, with no counterpart anywhere in `facts` because no filer ever tagged it.

## Date columns are `datetime.date`, and that is a trap this repo has already paid for

Postgres DATE round-trips to Python as `datetime.date`, never `Timestamp`. An unguarded
`frame["as_of"] > some_timestamp` comparison then does something surprising, and a
parquet-cached test harness hides the entire bug class because parquet gives you `Timestamp`.
So every date column is normalised to `datetime64[ns]` HERE, once, on load -- not in each
check, where the second author to forget it reintroduces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

import pandas as pd

from src.data_extract.utils.fundamentals.kpi_catalogue import (
    HISTORY_KEYS, HISTORY_PROVENANCE, HISTORY_REGIME, Catalogue)
from src.data_store.schema import Tables

#: The `fundamentals_facts` columns the checks actually read. A projection, not a
#: convenience: the table is ~28M rows universe-wide, and AGENTS.md forbids an unprojected
#: read of a table this size outright.
#:
#: `adjustment` IS READ -- by `adjustment_unguarded` (tier1_value.py), which audits whether an
#: adjustment fired on positive evidence or on silence. This comment previously asserted the
#: opposite ("read by no check"), which was false, and the check consequently returned early
#: on every run for want of the column. It reported ABSTAINED, which looks like a clean
#: abstention rather than a broken check, so nothing surfaced it.
#:
#: THE RULE, not the exception: every column any check subscripts off `sub.facts` must be
#: here. `test_substrate_contract.py` pins that generally rather than pinning `adjustment`,
#: because a test that named one column would not catch the next instance of this class.
#:
#: `unit` and `decimals` remain excluded, and those two really are read by nothing.
FACTS_COLUMNS: tuple[str, ...] = (
    "ticker", "accession_number", "field", "duration_type", "period_end", "period_start",
    "period_days", "fiscal_year", "fiscal_period", "cik", "form", "filing_date",
    "is_amendment", "period_of_report", "regime", "value", "resolution_method",
    "source_concept", "roll_up_children", "root_anchor", "role_uri", "is_extension",
    "dc_code", "adjustment")

#: Date columns per frame, forced to `datetime64[ns]` on load. See the module docstring.
_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "history": ("as_of", "fiscal_end", "amended_fiscal_end"),
    "codes": ("as_of",),
    "facts": ("filing_date", "period_end", "period_start", "period_of_report"),
    "employees": ("as_of",),
}


@dataclass
class Substrates:
    """Everything the checks read, plus the catalogue that says what the columns MEAN.

    Built by `load()` in production and by hand in tests. `tickers` is the roster the run was
    scoped to -- carried explicitly because "0 findings" and "0 tickers loaded" must never
    look the same in a report.
    """

    catalogue: Catalogue
    history: pd.DataFrame
    codes: pd.DataFrame
    facts: pd.DataFrame
    employees: pd.DataFrame | None = None
    tickers: tuple[str, ...] = ()

    #: Counts every check reports its fire rate against. Populated lazily; see `denominator`.
    _denominators: dict[str, int] = dataclass_field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ derived views ---
    @property
    def value_columns(self) -> list[str]:
        """The 60-odd VALUE columns of `fundamentals_history_sec` -- everything that is not a key,
        the regime, or provenance. The denominator of every coverage claim."""
        skip = {*HISTORY_KEYS, HISTORY_REGIME, *HISTORY_PROVENANCE}
        return [c for c in self.history.columns if c not in skip]

    @property
    def facts_fields(self) -> list[str]:
        """The catalogue fields that actually appear in `fundamentals_facts` -- the
        denominator of every coverage claim now that the coverage checks read facts.

        Deliberately NOT `value_columns`. Twelve of history's 69 columns exist nowhere in
        facts because they are DERIVED there -- `grossMargins`, `operatingMargins`,
        `profitMargins`, `effectiveTaxRate`, `returnOnEquity`, `debtToEquity`,
        `optionOverhang`, `freeCashflow`, `ebitda`, `epsDiluted`, `revenue_q`, `netIncome_q`.
        A ported check must not ask facts for a column facts does not have and then report
        the absence as a coverage defect.
        """
        if self.facts.empty or "field" not in self.facts.columns:
            return []
        return sorted(self.facts["field"].dropna().unique())

    @property
    def regime_by_ticker(self) -> dict[str, str]:
        """Each ticker's LATEST regime -- from FACTS where any were loaded, history otherwise.

        Facts first, because Tier 1's value and coverage checks moved there: a run scoped to
        facts alone must still know what regime a filer is in, and `fundamentals_facts.regime`
        is fully populated (316,136 of 316,136 rows on the live table). History remains the
        fallback so the six contract checks still work on a facts-less run.

        Latest and not modal: a filer that redomiciled or was reclassified is in its current
        regime now, and `peer_ratio` compares it to the peers it has today.
        """
        if not self.facts.empty and "regime" in self.facts.columns:
            latest = (self.facts.sort_values("filing_date")
                      .groupby("ticker")["regime"].last())
            from_facts = {t: r for t, r in latest.items() if isinstance(r, str)}
            if from_facts:
                return from_facts
        if self.history.empty or HISTORY_REGIME not in self.history.columns:
            return {}
        latest = self.history.sort_values("as_of").groupby("ticker")[HISTORY_REGIME].last()
        return {t: r for t, r in latest.items() if isinstance(r, str)}

    def cik_for(self, ticker: str) -> str | None:
        """The ticker's CIK, for building an EDGAR URL. From the FACTS table, which is where
        it is stored; None when this run loaded no facts for the ticker."""
        if self.facts.empty or "cik" not in self.facts.columns:
            return None
        rows = self.facts.loc[self.facts["ticker"] == ticker, "cik"].dropna()
        return str(rows.iloc[-1]) if len(rows) else None

    def coded(self) -> set[tuple[Any, Any, Any]]:
        """`{(ticker, as_of, field)}` that carry at least one reason code.

        A set, because the zero-unexplained-nulls gate is a membership test over ~200k cells
        and a per-cell `.loc` lookup on the codes frame turns that gate into minutes.
        """
        if self.codes.empty:
            return set()
        return set(zip(self.codes["ticker"], self.codes["as_of"], self.codes["field"]))

    def denominator(self, key: str, value: int | None = None) -> int:
        """Record (or read back) how many units a check EXAMINED.

        The fire rate is findings / this, and every check that declares an
        `expected_fire_rate_ceiling` has to say what it divided by -- a rate whose denominator
        is unstated cannot be compared to a ceiling, and "2% of rows" and "2% of cells" differ
        by two orders of magnitude on this table.
        """
        if value is not None:
            self._denominators[key] = int(value)
        return self._denominators.get(key, 0)


def load(context, catalogue: Catalogue, tickers: list[str] | None,
         *, since=None, need_facts: bool = True) -> Substrates:
    """Read every substrate ONCE, projected and ticker-scoped. The only DB access in the package.

    `need_facts` exists for a run that reads only the six Tier-1 CONTRACT checks. It is no
    longer implied by "Tier 1": the eight value and coverage checks in that tier read facts,
    so `validator.py` requests them for any tier at all. Left in place because the contract
    checks genuinely do not need the largest read in this function.
    `since` scopes the FACTS read to tickers that received a filing recently, which is
    decision 53's nightly shape -- a series can only change where a filing landed.
    """
    where = {"ticker": tickers} if tickers else None
    history = context.store.load(Tables.fundamentals_history_sec, columns=None, where=where)
    codes = context.store.load(Tables.fundamentals_reason_codes, columns=None, where=where,
                               optional=True)
    facts = (context.store.load(Tables.fundamentals_facts, columns=list(FACTS_COLUMNS),
                                where=where, since=since, optional=True)
             if need_facts else None)
    employees = context.store.load(Tables.fundamentals_employees, columns=None, where=where,
                                   optional=True)

    frames = {"history": history, "codes": codes, "facts": facts, "employees": employees}
    for name, frame in frames.items():
        frames[name] = _normalise(frame, _DATE_COLUMNS[name])

    loaded = tuple(sorted(frames["history"]["ticker"].unique())) if len(frames["history"]) \
        else tuple(tickers or ())
    return Substrates(catalogue=catalogue, history=frames["history"], codes=frames["codes"],
                      facts=frames["facts"], employees=frames["employees"], tickers=loaded)


def _normalise(frame: pd.DataFrame | None, date_columns: tuple[str, ...]) -> pd.DataFrame:
    """An empty frame for a missing table, and every date column as `datetime64[ns]`.

    An EMPTY frame rather than None so a check never has to branch on `is None` before it can
    ask a question. A check that genuinely needs to know whether a table was populated asks
    `.empty`, which is the same answer without the extra shape.
    """
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    for column in date_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out
