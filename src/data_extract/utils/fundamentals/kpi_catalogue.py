"""
kpi_catalogue.py
----------------
Loads, validates and exposes typed accessors over the three JSON files that ARE the fundamentals
contract, under `configs/fundamentals/`:
  * `fundamentals_kpis.json`       -- per field: tier, kind, sign, unit, definition, authority, resolution.
  * `fundamentals_regimes.json`    -- which statement template a filing is read against (role URI, then GICS).
  * `fundamentals_exceptions.json` -- (regime/ticker, field) -> is a missing value structural or a regression.

Every field must carry an `authority` citing a primary source (`UNVERIFIED` is the only placeholder and
requires an `authority_note`), optionally `authority_caveat` or `authority_inherits_from`. Validation is
strict and runs at load; the catalogue is built once per resolved config directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

from src.constants.constants import (
    DEFAULT_CONFIG_DIR,
    FUNDAMENTALS_CATALOGUE_SUBDIR,
    FUNDAMENTALS_EXCEPTIONS_FILENAME,
    FUNDAMENTALS_KPIS_FILENAME,
    FUNDAMENTALS_REGIMES_FILENAME,
)
from src.data_extract.utils.common.config_paths import resolve_config_dir

# `DEFAULT_CONFIG_DIR` is re-exported: sibling fundamentals modules import it from here.

#: Keys with this prefix are inline documentation in the JSONs, not data; the loader skips them.
_DOC_PREFIX = "_"

#: The placeholder that means "the research could not establish a primary source".
UNVERIFIED = "UNVERIFIED"

Kind = Literal["instant", "duration", "ratio", "derived"]
Sign = Literal["non_negative", "non_positive", "any"]

#: Every `kind` a value can be EXTRACTED as. `derived` and `ratio` fields are computed from
#: other fields and never resolved against a concept, so they carry no fallback list.
EXTRACTED_KINDS: frozenset[str] = frozenset({"instant", "duration"})

#: tier 0 = a calculation input: carried because a chosen definition requires it, never
#: z-scored and never peer-ranked. 1-3 = a scored KPI.
INPUT_TIER = 0
SCORED_TIERS: frozenset[int] = frozenset({1, 2, 3})


@dataclass(frozen=True)
class FieldSpec:
    """One field's contract. Mirrors its JSON entry, with the keys the resolver needs
    promoted to attributes and everything else reachable through `raw`."""

    name: str
    tier: int
    kind: Kind
    sign: Sign
    unit: str
    definition: str
    authority: str
    raw: dict[str, Any]

    @property
    def is_scored(self) -> bool:
        """Does this field enter z-scores and peer ranks? False for calculation inputs."""
        return self.tier in SCORED_TIERS

    @property
    def is_extracted(self) -> bool:
        """Is this field resolved against XBRL concepts? False for computed and text-sourced fields."""
        if self.is_text_sourced:
            return False
        return self.kind in EXTRACTED_KINDS

    @property
    def is_text_sourced(self) -> bool:
        """Is the value parsed from narrative text (`source` starting `text`, e.g. `employees`) rather than XBRL?

        Such fields are kept out of the wide history table and written to their own side table.
        """
        return str(self.raw.get("source", "")).startswith("text")

    @property
    def regime_gated(self) -> bool:
        """True where the field is only DEFINED for some regimes, so absence elsewhere is
        `not_applicable_for_regime` rather than a coverage finding."""
        return bool(self.raw.get("regime_gated", False))

    @property
    def authority_inherits_from(self) -> list[str]:
        """Field(s) whose cited authority justifies carrying this calculation input, which has no
        primary source of its own. Distinct from UNVERIFIED (definition unestablished)."""
        return list(self.raw.get("authority_inherits_from", []))

    @property
    def is_additive(self) -> bool:
        """Can four discrete quarters be summed to a TTM? False for weighted-average share
        counts, ratios and per-share amounts -- summing those is meaningless."""
        return self.is_extracted and not self.raw.get("not_additive", False)

    def fallback_concepts(self, regime: str | None = None) -> list[str]:
        """Priority-ordered concepts for the `tag_fallback` branch, with the regime's
        override taking precedence over the field-level list when one exists."""
        override = self._regime_block(regime).get("fallback_concepts")
        if override is not None:
            return list(override)
        return list(self.raw.get("fallback_concepts", []))

    def total_concept(self, regime: str | None = None) -> str | None:
        """The concept to prefer when the filer declares it as a linkbase PARENT."""
        block = self._regime_block(regime)
        return block.get("total_concept", self.raw.get("total_concept"))

    def roll_up(self, regime: str | None = None) -> list[str]:
        """The children the linkbase result is CHECKED AGAINST -- never a substitute for reading the linkbase."""
        block = self._regime_block(regime)
        spec = block.get("roll_up", self.raw.get("roll_up")) or {}
        return list(spec.get("sum", []))

    @cached_property
    def _never_use_by_regime(self) -> dict[str | None, MappingProxyType]:
        """`never_use`'s per-regime memo (an instance dict: `raw` makes this frozen dataclass unhashable)."""
        return {}

    def never_use(self, regime: str | None = None) -> MappingProxyType:
        """Read-only map concept -> why it must NEVER resolve this field (field-level merged with the regime's).

        Part of the contract: a resolver MUST consult it, not merely log it.
        """
        cached = self._never_use_by_regime.get(regime)
        if cached is None:
            merged = dict(self.raw.get("never_use", {}))
            merged.update(self._regime_block(regime).get("never_use", {}))
            cached = self._never_use_by_regime[regime] = MappingProxyType(merged)
        return cached

    def _regime_block(self, regime: str | None) -> dict[str, Any]:
        if not regime:
            return {}
        return self.raw.get("regimes", {}).get(regime, {})


#: `fundamentals_history_sec`'s key columns. `fiscal_quarter` labels which quarter of the issuer's own
#: fiscal year `fiscal_end` closes, on every row including TTM and instant values. Sector / industry
#: are not carried: they join from `sp500_tickers`.
HISTORY_KEYS: tuple[str, ...] = ("ticker", "as_of", "fiscal_end", "fiscal_quarter")

#: The value columns in STATEMENT order (income statement, cash flow, balance sheet, share counts), each
#: ratio right after the line it is computed from. Declared, not derived; `history_columns` asserts it
#: against the catalogue so a new field fails loudly.
HISTORY_STATEMENT_ORDER: tuple[str, ...] = (
    # -- revenue: the general top line, then the regime-specific ones that replace it
    "totalRevenue",
    "premiumsEarned",
    "netInterestIncome",
    "noninterestIncome",
    "netInvestmentIncome",
    "realizedInvestmentGains",
    "rentalIncome",
    # -- cost of sales and gross result
    "costOfRevenue",
    "grossProfit",
    "grossMargins",
    # -- operating expense
    "sellingGeneralAdmin",
    "researchAndDevelopment",
    "depAmort",
    "stockBasedComp",
    # -- operating result
    "operatingIncome",
    "operatingMargins",
    "ebitda",
    # -- below the operating line, down to the bottom line
    "interestExpense",
    "pretaxIncome",
    "incomeTaxExpense",
    "effectiveTaxRate",
    "netIncome",
    "profitMargins",
    "epsDiluted",
    # -- the two single-quarter slices, next to the TTM lines they are cut from
    "revenue_q",
    "netIncome_q",
    # -- cash flow
    "operatingCashFlow",
    "capex",
    "freeCashflow",
    # -- assets, in Reg S-X current-then-long-lived order
    "cash",
    "restrictedCash",
    "shortTermInvestments",
    "accountsReceivable",
    "inventory",
    "currentAssets",
    "ppeGross",
    "accumulatedDepreciation",
    "ppeNet",
    "goodwill",
    "intangiblesExGoodwill",
    "totalAssets",
    # -- liabilities and debt, current then long-term, components before the roll-ups
    "accountsPayable",
    "currentLiabilities",
    "shortTermDebt",
    "shortTermBorrowingsOnly",
    "longTermDebt",
    "longTermDebtCurrentOnly",
    "operatingLeaseLiability",
    "financeLeaseLiability",
    "totalDebt",
    "totalLiabilities",
    # -- equity, and the two ratios that read off it
    "retainedEarnings",
    "minorityInterest",
    "stockholdersEquity",
    "returnOnEquity",
    "debtToEquity",
    # -- share counts last: the denominators, not the statements
    "basicShares",
    "dilutedShares",
    "sharesOutstanding",
    "optionOverhang",
)

#: Publication-event provenance, scalar per row: `publication_form` is the highest-precedence form filed
#: that day (`10-K` > `10-K/A` > `10-Q` > `10-Q/A`), `is_amendment` an OR, `amended_fiscal_end` the latest
#: restated period, `amended_fields` the union. Accession detail stays in `fundamentals_facts`.
HISTORY_PROVENANCE: tuple[str, ...] = ("publication_form", "is_amendment", "amended_fiscal_end", "amended_fields")

#: The filing's resolution regime, as stamped per filing on `fundamentals_facts`.
HISTORY_REGIME = "regime"

#: Declared columns computed at CUBE time, not by the history build: year-on-year ratios need a 365-day
#: `as_of` offset, which the publication-event grain (amendment rows) cannot give as a row offset.
CUBE_TIME_COLUMNS: frozenset[str] = frozenset({"revenueGrowth", "earningsGrowth"})


@dataclass(frozen=True)
class Catalogue:
    """The three loaded files, validated. `fields` is eager; every derived view is memoised on first use."""

    fields: dict[str, FieldSpec]
    derived_columns: dict[str, str]
    regimes: dict[str, Any]
    regime_exceptions: dict[str, Any]
    force_regime_by_sub_industry: dict[str, str]
    ticker_exceptions: dict[str, Any]
    ticker_periodicity: dict[str, Any]

    @cached_property
    def all_column_names(self) -> frozenset[str]:
        """Every name the contract declares (catalogue fields plus computed columns), the set `feeds` /
        `components` references resolve against. Wider than `history_columns` (includes side-table fields)."""
        return frozenset(self.fields) | frozenset(self.derived_columns)

    # ------------------------------------------------- the history contract --- #
    @cached_property
    def side_table_fields(self) -> list[str]:
        """Catalogue fields the wide history table does not carry because they are text-sourced
        (`employees` -> `fundamentals_employees`)."""
        return sorted(n for n, s in self.fields.items() if s.is_text_sourced)

    @cached_property
    def history_fields(self) -> list[str]:
        """The catalogue fields `fundamentals_history_sec` carries, ordered tier then name.

        One column per field under its bare name: the TTM for a `duration` field, the latest instant for an
        `instant` one.
        """
        side = set(self.side_table_fields)
        return sorted((n for n in self.fields if n not in side), key=lambda n: (self.fields[n].tier, n))

    @cached_property
    def history_derived_columns(self) -> list[str]:
        """The computed columns the history build owns: all declared minus `CUBE_TIME_COLUMNS`."""
        return sorted(set(self.derived_columns) - CUBE_TIME_COLUMNS)

    @cached_property
    def history_columns(self) -> list[str]:
        """The `fundamentals_history_sec` column contract in table order: keys, `HISTORY_STATEMENT_ORDER`,
        `regime`, provenance (4 + 60 + 1 + 4 = 69). `build_history` builds its frame from it.

        Asserts `HISTORY_STATEMENT_ORDER` matches the history fields plus derived columns exactly.
        """
        fields = [*self.history_fields, *self.history_derived_columns]
        missing = sorted(set(fields) - set(HISTORY_STATEMENT_ORDER))
        stale = sorted(set(HISTORY_STATEMENT_ORDER) - set(fields))
        assert not missing and not stale, (
            f"HISTORY_STATEMENT_ORDER is out of step with the catalogue: unordered {missing}, ordered-but-absent {stale}"
        )
        return [*HISTORY_KEYS, *HISTORY_STATEMENT_ORDER, HISTORY_REGIME, *HISTORY_PROVENANCE]

    # ---------------------------------------------------------------- fields --- #
    def field(self, name: str) -> FieldSpec:
        """One field's spec. Raises KeyError rather than returning None: a field outside the contract is a bug."""
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(f"{name!r} is not in the KPI catalogue ({len(self.fields)} fields declared)") from None

    @cached_property
    def _by_tier(self) -> dict[int, list[str]]:
        """Every tier's sorted field list, built in one pass over `fields`."""
        out: dict[int, list[str]] = {}
        for name in sorted(self.fields):
            out.setdefault(self.fields[name].tier, []).append(name)
        return out

    def by_tier(self, tier: int) -> list[str]:
        return list(self._by_tier.get(tier, ()))

    @cached_property
    def scored_fields(self) -> list[str]:
        return sorted(n for n, s in self.fields.items() if s.is_scored)

    @cached_property
    def input_fields(self) -> list[str]:
        return sorted(n for n, s in self.fields.items() if s.tier == INPUT_TIER)

    @cached_property
    def extracted_fields(self) -> list[str]:
        """Fields resolved against XBRL, i.e. everything the facts layer must produce."""
        return sorted(n for n, s in self.fields.items() if s.is_extracted)

    @cached_property
    def unverified_fields(self) -> list[str]:
        """Fields whose `authority` is still the placeholder; the schema test asserts on this list."""
        return sorted(n for n, s in self.fields.items() if s.authority == UNVERIFIED)

    # --------------------------------------------------------------- regimes --- #
    @cached_property
    def regime_names(self) -> list[str]:
        return sorted(self.regimes)

    def regime_for_gics(self, sector: str | None = None, industry_group: str | None = None, sub_industry: str | None = None) -> str | None:
        """The GICS-implied regime, or None (never a guess) when nothing matches.

        Most specific wins: a forced sub-industry override (`force_regime`), then the regimes' declared
        sub-industry, industry-group, then sector membership.
        """
        forced = self.force_regime_by_sub_industry.get(sub_industry or "")
        if forced:
            return forced
        for level, value in (("sub_industry", sub_industry), ("industry_group", industry_group), ("sector", sector)):
            if not value:
                continue
            for name, spec in self.regimes.items():
                declared = spec.get("gics", {}).get(level, [])
                if value in (declared if isinstance(declared, list) else [declared]):
                    return name
        return None

    def regime_for_role_uris(self, role_uris: list[str]) -> str | None:
        """The regime implied by the filer's statement role URIs (case-insensitive `role_patterns` match).

        Evidence from the filing itself, so it outranks GICS. Returns None, not the default, when no role matches.
        """
        blob = " ".join(role_uris).lower()
        for name, spec in self.regimes.items():
            for pattern in spec.get("role_patterns", []):
                if str(pattern).lower() in blob:
                    return name
        return None

    def regime_for(self, gics: dict[str, str | None] | None, role_uris: list[str]) -> str | None:
        """The filing's regime: role URI first, then GICS, then the `industrial` (Article 5) default.

        A ticker with no GICS row is unclassified and returns None (skip, never default) unless a role URI matched.
        """
        from_role = self.regime_for_role_uris(role_uris)
        if from_role:
            return from_role
        if not gics:
            return None
        return self.regime_for_gics(**gics) or self.default_regime()

    def default_regime(self) -> str:
        """The regime marked `is_default` (Article 5, Reg S-X's general case). Raises ValueError if none is."""
        for name, spec in self.regimes.items():
            if spec.get("is_default"):
                return name
        raise ValueError("no regime is marked is_default in the regimes config")

    # ------------------------------------------------------------ exceptions --- #
    def expected_absent(self, regime: str, field: str) -> bool:
        """Is a missing (regime, field) value STRUCTURAL rather than a regression?

        Defaults to False: an unjustified absence must stay a finding."""
        block = self.regime_exceptions.get(regime, {}).get(field)
        return bool(block.get("expected_absent", False)) if isinstance(block, dict) else False

    @cached_property
    def _filer_leaves_memo(self) -> dict[tuple[str | None, str], tuple]:
        """`filer_leaves`' memo (an instance dict, as for `FieldSpec._never_use_by_regime`)."""
        return {}

    def filer_leaves(self, ticker: str | None, field: str) -> tuple[tuple[tuple[str, ...], ...], frozenset[str]]:
        """One filer's DECLARED company-extension leaves for a field, as `(leaf_groups, not_leaves)`, memoised.

        No structural rule identifies an extension leaf, so it is declared per filer or the filer stays
        reason-coded. `leaf_groups` has the shape of `roll_up.any_of` (alternatives within a group, summed
        across groups) and is appended to it; `not_leaves` names the node's extensions that are NOT this field.
        """
        key = (ticker, field)
        cached = self._filer_leaves_memo.get(key)
        if cached is None:
            block = self.ticker_exceptions.get(ticker or "", {}).get(field)
            if isinstance(block, dict):
                cached = (tuple(tuple(g) for g in block.get("leaves", [])), frozenset(block.get("not_leaves", [])))
            else:
                cached = ((), frozenset())
            self._filer_leaves_memo[key] = cached
        return cached

    def periodicity_shapes(self, ticker: str | None, field: str) -> list[str] | None:
        """The period shapes a filer tags this field on, where structurally limited (e.g. annual-only);
        None where nothing is declared. Such a gap is structural, not a regression."""
        block = self.ticker_periodicity.get(ticker or "", {}).get(field)
        return list(block.get("shapes", [])) if isinstance(block, dict) else None

    def combined_into(self, regime: str | None, ticker: str | None, field: str) -> str | None:
        """The field this one is folded into for this filer or regime, or None. A ticker cell wins over a regime cell."""
        for register, key in ((self.ticker_exceptions, ticker), (self.regime_exceptions, regime)):
            block = register.get(key or "", {}).get(field)
            if isinstance(block, dict) and block.get("combined_into"):
                return str(block["combined_into"])
        return None

    def regime_break_effective(self, field: str) -> pd.Timestamp | None:
        """The date a definitional discontinuity (`regime_break.effective`, e.g. ASC 842) took effect for
        `field`, or None. Values on either side are real but not comparable."""
        block = self.field(field).raw.get("regime_break") or {}
        effective = block.get("effective")
        return pd.Timestamp(effective) if effective else None

    def measured_absent_rate(self, regime: str, field: str) -> float | None:
        """The share of the regime's tickers with no fact for this field, or None where not measured.

        A frozen historical record from a dropped source table: it cannot be re-derived from `fundamentals_facts`.
        """
        block = self.regime_exceptions.get(regime, {}).get(field)
        return block.get("measured_absent_rate") if isinstance(block, dict) else None


# --------------------------------------------------------------------------- #
# loading + validation                                                         #
# --------------------------------------------------------------------------- #
def _read_json(config_dir: Path, filename: str) -> dict[str, Any]:
    path = config_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"KPI catalogue file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _data_items(blob: dict[str, Any]) -> dict[str, Any]:
    """The blob's real entries, with the inline documentation keys dropped."""
    return {k: v for k, v in blob.items() if not k.startswith(_DOC_PREFIX)}


def _build_field(name: str, entry: dict[str, Any]) -> FieldSpec:
    missing = [k for k in ("tier", "kind", "sign", "unit", "definition", "authority") if k not in entry]
    if missing:
        raise ValueError(f"{name}: missing mandatory key(s) {missing}")
    if entry["authority"] == UNVERIFIED and "authority_note" not in entry:
        raise ValueError(f"{name}: authority is {UNVERIFIED} with no `authority_note` saying what IS known and which document would close it")
    return FieldSpec(
        name=name,
        tier=int(entry["tier"]),
        kind=entry["kind"],
        sign=entry["sign"],
        unit=entry["unit"],
        definition=entry["definition"],
        authority=entry["authority"],
        raw=entry,
    )


def load_catalogue(config_dir: str | None = DEFAULT_CONFIG_DIR) -> Catalogue:
    """The validated catalogue, built once per (process, resolved config directory).

    Deliberately NOT cached itself: caching lives only on `_catalogue_at`, keyed on the resolved path,
    so different spellings of one directory share one object and a `cache_clear()` cannot leave stale copies.
    """
    return _catalogue_at(resolve_config_dir(config_dir))


@cache
def _catalogue_at(config_dir: str) -> Catalogue:
    """`load_catalogue`, keyed on a resolved absolute path. Raises ValueError / FileNotFoundError on a
    malformed or missing contract, so a bad config fails at the first call."""
    root = Path(config_dir) / FUNDAMENTALS_CATALOGUE_SUBDIR
    kpis_blob = _read_json(root, FUNDAMENTALS_KPIS_FILENAME)
    regimes_blob = _read_json(root, FUNDAMENTALS_REGIMES_FILENAME)
    exceptions_blob = _read_json(root, FUNDAMENTALS_EXCEPTIONS_FILENAME)

    kpis = _data_items(kpis_blob)
    fields = {name: _build_field(name, entry) for name, entry in kpis.items()}
    derived_columns = _data_items(kpis_blob.get("_derived_columns", {}))

    # An inherited authority must name a field that exists.
    for name, spec in fields.items():
        for parent in spec.authority_inherits_from:
            if parent not in fields:
                raise ValueError(f"{name}: authority_inherits_from names unknown field {parent!r}")

    regimes = _data_items(regimes_blob.get("regimes", {}))
    if not regimes:
        raise ValueError("regimes config declares no regimes")

    force: dict[str, str] = {}
    for sub_industry, block in _data_items(regimes_blob.get("exceptions", {}).get("force_regime", {})).items():
        force[sub_industry] = block["regime"]

    regime_exceptions = {regime: _data_items(block) for regime, block in _data_items(exceptions_blob.get("by_regime", {})).items()}

    ticker_exceptions = {ticker: _data_items(block) for ticker, block in _data_items(exceptions_blob.get("by_ticker", {})).items()}
    periodicity = {ticker: _data_items(block) for ticker, block in _data_items(exceptions_blob.get("by_ticker_periodicity", {})).items()}

    # An exception-register field missing from the catalogue is a typo that would excuse nothing.
    for regime, block in regime_exceptions.items():
        unknown = sorted(set(block) - set(fields))
        if unknown:
            raise ValueError(f"exceptions[{regime}] names unknown field(s) {unknown}")
    for label, register in (("by_ticker", ticker_exceptions), ("by_ticker_periodicity", periodicity)):
        ticker: str | None = None
        block: dict = {}
        for ticker, block in register.items():
            unknown = sorted(set(block) - set(fields))
            if unknown:
                raise ValueError(f"exceptions.{label}[{ticker}] names unknown field(s) {unknown}")
        if ticker is None:
            continue
        # A concept declared both leaf and not-leaf is a contradiction; fail loudly.
        for field, entry in block.items():
            if not isinstance(entry, dict):
                continue
            leaves = {c for g in entry.get("leaves", []) for c in g}
            both = sorted(leaves & set(entry.get("not_leaves", [])))
            if both:
                raise ValueError(f"exceptions.by_ticker[{ticker}][{field}]: {both} is declared both a leaf and not a leaf")
            if leaves and "evidence" not in entry:
                raise ValueError(
                    f"exceptions.by_ticker[{ticker}][{field}]: declares extension leaves "
                    "with no `evidence` key. A per-filer override with no written evidence "
                    "is exactly the guess this register exists to replace."
                )

    # A regime-keyed override in the KPI catalogue must name a real regime.
    known_regimes = set(regimes)
    for name, spec in fields.items():
        unknown = sorted(set(spec.raw.get("regimes", {})) - known_regimes)
        if unknown:
            raise ValueError(f"{name}: regimes override names unknown regime(s) {unknown}")

    return Catalogue(
        fields=fields,
        derived_columns=derived_columns,
        regimes=regimes,
        regime_exceptions=regime_exceptions,
        force_regime_by_sub_industry=force,
        ticker_exceptions=ticker_exceptions,
        ticker_periodicity=periodicity,
    )
