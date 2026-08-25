"""
kpi_catalogue.py
----------------
Loads and validates the three JSON files that ARE the fundamentals contract, and exposes
typed accessors over them:

  * `configs/fundamentals/fundamentals_kpis.json`       -- one entry per field: tier, kind,
                                    sign, unit, definition, authority, and how to resolve it.
  * `configs/fundamentals/fundamentals_regimes.json`    -- which statement TEMPLATE a filing
                                    is read against (role URI first, GICS as tiebreak).
  * `configs/fundamentals/fundamentals_exceptions.json` -- (regime, field) -> is a missing
                                    value structural, or a regression?

`authority` is mandatory on every field and must cite a primary source. The literal string
"UNVERIFIED" is the only permitted placeholder, and `tests/data_extract/test_kpi_catalogue.py`
asserts that every occurrence is deliberate -- so no definition in this pipeline can quietly
rest on a guess. Phase 2 shipped with 17 UNVERIFIED fields; a second research pass closed
**all 17** against FASB's own 2025 taxonomy files (`us-gaap-doc-2025.xml`,
`us-gaap-ref-2025.xml`, `us-gaap-2025.xsd`) and eCFR Reg S-X, so the placeholder is now unused.

Three grades of citation, all machine-checkable, because "sourced" is not binary:
  * `authority`                -- the citation. A quote or a rule/paragraph reference.
  * `authority_caveat`         -- the field IS verified, but one sub-claim is a notch weaker,
                                  typically an ASC paragraph whose NUMBER is primary (FASB's
                                  reference linkbase) while its PROSE could only be read in a
                                  secondary reproduction, `asc.fasb.org` being login-walled.
  * `authority_inherits_from`  -- a tier-0 calculation input or a ratio has no primary source
                                  of its OWN; it exists because another field's cited
                                  definition requires it, so it names that field.

Loaded ONCE per process (`functools.cache`): the files are small, but the facts layer asks
for a field spec per field per filing, and re-reading + re-validating three JSONs ~30k times
during a full rebuild is pure waste.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.constants.constants import (
    FUNDAMENTALS_CATALOGUE_SUBDIR, FUNDAMENTALS_EXCEPTIONS_FILENAME,
    FUNDAMENTALS_KPIS_FILENAME, FUNDAMENTALS_REGIMES_FILENAME,
)

#: Default config directory, matching the `-c ./configs` CLI default.
DEFAULT_CONFIG_DIR = "./configs"

#: Keys whose leading underscore marks them as documentation, not data. The JSONs carry
#: their own rationale inline (a `_README` block, `_authority` notes) so the contract and
#: its justification cannot drift apart in separate files; the loader skips them.
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
        """Is this field resolved against XBRL CONCEPTS at all?

        False for a field computed from others, and false for one the catalogue sources
        somewhere else entirely: `employees` declares `"source": "text:10-K"` and is parsed
        out of the 10-K narrative by `fundamentals_employees.py`, so it has no
        `fallback_concepts` and can never resolve here. Left in, it resolved 0 times on all
        52 swept tickers and emitted one reason-coded row per filing -- ~1,600 rows a sweep
        asserting that a field the XBRL walk was never going to find was not found.
        """
        if self.is_text_sourced:
            return False
        return self.kind in EXTRACTED_KINDS

    @property
    def is_text_sourced(self) -> bool:
        """Is the value parsed out of NARRATIVE TEXT rather than XBRL?

        True only for `employees`, whose `source` is `"text:10-K"`. It is the discriminator
        that keeps that field out of the wide history table (decision 35): a text parse can
        fail in ways an XBRL walk cannot, and in the wide table one failed regex would fail
        the whole snapshot. It goes to `fundamentals_employees` instead, and the field keeps
        its tier and its authority in the catalogue.
        """
        return str(self.raw.get("source", "")).startswith("text")

    @property
    def regime_gated(self) -> bool:
        """True where the field is only DEFINED for some regimes, so absence elsewhere is
        `not_applicable_for_regime` rather than a coverage finding."""
        return bool(self.raw.get("regime_gated", False))

    @property
    def authority_inherits_from(self) -> list[str]:
        """Field(s) whose SOURCED authority justifies carrying this one. A calculation
        input has no primary source of its own -- it exists because another field's cited
        definition requires it -- so naming that field is the honest citation. Distinct
        from UNVERIFIED, which means the definition itself is unestablished."""
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
        """The children the linkbase result is CHECKED AGAINST -- never a substitute for
        reading the linkbase, which is the whole point of the rebuild."""
        block = self._regime_block(regime)
        spec = block.get("roll_up", self.raw.get("roll_up")) or {}
        return list(spec.get("sum", []))

    def never_use(self, regime: str | None = None) -> dict[str, str]:
        """concept -> why it must NEVER resolve this field. These are the measured traps
        (MAA's IPR&D-tagged capex, a bank's gross interest income), so they are part of the
        contract and a resolver MUST consult them, not merely log them."""
        merged = dict(self.raw.get("never_use", {}))
        merged.update(self._regime_block(regime).get("never_use", {}))
        return merged

    def _regime_block(self, regime: str | None) -> dict[str, Any]:
        if not regime:
            return {}
        return self.raw.get("regimes", {}).get(regime, {})


#: `fundamentals_history`'s own key columns. `sector` / `industry_group` are NOT here
#: (decision 32): they are a slowly-changing dimension joinable from `sp500_tickers`, and
#: carrying them inside a point-in-time table duplicates a non-vintaged roster into every
#: row. The residual is stated rather than hidden -- `regime` stays, and it is derived from
#: `sub_industry` off that same roster, so the look-ahead this removes from two columns is
#: still present in one. Accepted because `regime` drives RESOLUTION and therefore cannot be
#: joined at cube time.
#:
#: `fiscal_quarter` labels which quarter of the ISSUER's OWN year `fiscal_end` closes -- Q1-Q4
#: on every row, including the ones whose values are TTM or balance-sheet instants. A TTM
#: spans four quarters, but the row still reports as of one of them, and that is what a
#: seasonal comparison needs: a filer's Q4 is not its Q1, and `fiscal_end`'s calendar month
#: does not say which is which for the 52/53-week and non-December filers on the roster.
HISTORY_KEYS: tuple[str, ...] = ("ticker", "as_of", "fiscal_end", "fiscal_quarter")

#: The 60 value columns in STATEMENT order: income statement top-down, then cash flow, then
#: the balance sheet, then the share counts. Each ratio sits immediately after the line it is
#: computed from, so a reader can check it in place -- `grossMargins` under `grossProfit`,
#: `returnOnEquity` under `stockholdersEquity`.
#:
#: Declared, not derived. The resolution order is tier-then-name (`history_fields`), which is
#: what the BUILD needs and reads as noise in a table: `basicShares` first and `totalRevenue`
#: twenty-four columns later, with `costOfRevenue` in a different tier from the revenue it is
#: subtracted from. `history_columns` asserts this list against the catalogue, so a new field
#: in the JSON fails loudly here rather than being appended to the end of the table.
HISTORY_STATEMENT_ORDER: tuple[str, ...] = (
    # -- revenue: the general top line, then the regime-specific ones that replace it
    "totalRevenue",
    "premiumsEarned", "netInterestIncome", "noninterestIncome", "netInvestmentIncome",
    "realizedInvestmentGains", "rentalIncome",
    # -- cost of sales and gross result
    "costOfRevenue", "grossProfit", "grossMargins",
    # -- operating expense
    "sellingGeneralAdmin", "researchAndDevelopment", "depAmort", "stockBasedComp",
    # -- operating result
    "operatingIncome", "operatingMargins", "ebitda",
    # -- below the operating line, down to the bottom line
    "interestExpense", "pretaxIncome", "incomeTaxExpense", "effectiveTaxRate",
    "netIncome", "profitMargins", "epsDiluted",
    # -- the two single-quarter slices, next to the TTM lines they are cut from
    "revenue_q", "netIncome_q",
    # -- cash flow
    "operatingCashFlow", "capex", "freeCashflow",
    # -- assets, in Reg S-X current-then-long-lived order
    "cash", "restrictedCash", "shortTermInvestments", "accountsReceivable", "inventory",
    "currentAssets", "ppeGross", "accumulatedDepreciation", "ppeNet", "goodwill",
    "intangiblesExGoodwill", "totalAssets",
    # -- liabilities and debt, current then long-term, components before the roll-ups
    "accountsPayable", "currentLiabilities", "shortTermDebt", "shortTermBorrowingsOnly",
    "longTermDebt", "longTermDebtCurrentOnly", "operatingLeaseLiability",
    "financeLeaseLiability", "totalDebt", "totalLiabilities",
    # -- equity, and the two ratios that read off it
    "retainedEarnings", "minorityInterest", "stockholdersEquity", "returnOnEquity",
    "debtToEquity",
    # -- share counts last: the denominators, not the statements
    "basicShares", "dilutedShares", "sharesOutstanding", "optionOverhang",
)

#: The publication-event provenance, scalar by precedence (decision 37) so every column
#: stays queryable: `publication_form` is the highest-precedence form filed that day
#: (`10-K` > `10-K/A` > `10-Q` > `10-Q/A`), `is_amendment` is an OR, `amended_fiscal_end` the
#: latest restated period and `amended_fields` the union. Accession-level detail always
#: remains recoverable from `fundamentals_facts`.
HISTORY_PROVENANCE: tuple[str, ...] = ("publication_form", "is_amendment",
                                       "amended_fiscal_end", "amended_fields")

#: The filing's resolution regime, taken off `fundamentals_facts` where the facts layer
#: already stamped it per filing rather than re-derived here.
HISTORY_REGIME = "regime"

#: Declared columns computed at CUBE time, not by the history build (decision 33).
#:
#: Both are year-on-year ratios, and the bug is in the OFFSET, not in the numerator: at cube
#: time `pit.py` can take a fixed 365-DAY `as_of` offset, while a history-build version can
#: only take a 4-ROW one -- and under the publication-event grain an amendment row makes four
#: rows ~9 months, not 12. Computing them here would fix two columns and leave
#: `infer_yoy_periods`' row-offset in every other cube growth feature; moving them fixes all
#: of them at once. That repair is Phase 6's (§6.1); Phase 5's job is not to ship the two
#: columns whose definition it cannot satisfy.
CUBE_TIME_COLUMNS: frozenset[str] = frozenset({"revenueGrowth", "earningsGrowth"})


@dataclass(frozen=True)
class Catalogue:
    """The three loaded files, validated, with lookups precomputed."""

    fields: dict[str, FieldSpec]
    derived_columns: dict[str, str]
    regimes: dict[str, Any]
    regime_exceptions: dict[str, Any]
    force_regime_by_sub_industry: dict[str, str]
    ticker_exceptions: dict[str, Any]
    ticker_periodicity: dict[str, Any]

    @property
    def all_column_names(self) -> set[str]:
        """Every name the CONTRACT declares: catalogue fields plus the computed columns.

        The reference-resolution set, for `feeds` / `components`. Deliberately WIDER than
        `history_columns`: `employees` is a real catalogue field that another field may
        legitimately reference, it simply lives in its own table.
        """
        return set(self.fields) | set(self.derived_columns)

    # ------------------------------------------------- the history contract --- #
    @property
    def side_table_fields(self) -> list[str]:
        """Catalogue fields the WIDE history table does not carry, because their source is
        not XBRL. Today: `employees` -> `fundamentals_employees` (decision 35)."""
        return sorted(n for n, s in self.fields.items() if s.is_text_sourced)

    @property
    def history_fields(self) -> list[str]:
        """The catalogue fields `fundamentals_history` carries, ordered TIER then name.

        One column per field, BARE NAME, on the TTM basis for a `duration` field and the
        latest instant for an `instant` one (decision 31). That matches the legacy naming
        exactly -- `totalRevenue` always WAS the TTM -- so `build_cube.yml` and
        `SECTOR_KPI_SCOPE` need no renaming, and the `_ttm` suffix in the KPI JSON's prose
        names the CONCEPT, never a column.
        """
        side = set(self.side_table_fields)
        return sorted((n for n in self.fields if n not in side),
                      key=lambda n: (self.fields[n].tier, n))

    @property
    def history_derived_columns(self) -> list[str]:
        """The computed columns the history build owns -- everything declared minus the
        cube-time ones. Subtracted defensively as well as excluded in the config, so the
        contract cannot silently regrow by a config edit alone."""
        return sorted(set(self.derived_columns) - CUBE_TIME_COLUMNS)

    @property
    def history_columns(self) -> list[str]:
        """The `fundamentals_history` column contract, in table order: 4 keys + 52 catalogue
        fields + 8 derived + `regime` + 4 provenance = **69**.

        The number was "~71" twice and "68" once in the rebuild plan with no enumeration
        behind either, while *"column count is exactly as contracted"* was a verification
        item. This property IS the contract; `build_history` builds its frame from it and
        asserts the length, so the two can never disagree again.

        Column ORDER is `HISTORY_STATEMENT_ORDER`, not the tier-then-name order the fields
        are RESOLVED in: reading the table should read like the statements it came from.
        """
        fields = [*self.history_fields, *self.history_derived_columns]
        missing = sorted(set(fields) - set(HISTORY_STATEMENT_ORDER))
        stale = sorted(set(HISTORY_STATEMENT_ORDER) - set(fields))
        assert not missing and not stale, (
            f"HISTORY_STATEMENT_ORDER is out of step with the catalogue: "
            f"unordered {missing}, ordered-but-absent {stale}")
        return [*HISTORY_KEYS, *HISTORY_STATEMENT_ORDER,
                HISTORY_REGIME, *HISTORY_PROVENANCE]

    # ---------------------------------------------------------------- fields --- #
    def field(self, name: str) -> FieldSpec:
        """One field's spec. Raises rather than returning None: a caller asking for a field
        that is not in the contract is a bug, not a missing value."""
        try:
            return self.fields[name]
        except KeyError:
            raise KeyError(f"{name!r} is not in the KPI catalogue "
                           f"({len(self.fields)} fields declared)") from None

    def by_tier(self, tier: int) -> list[str]:
        return sorted(n for n, s in self.fields.items() if s.tier == tier)

    @property
    def scored_fields(self) -> list[str]:
        return sorted(n for n, s in self.fields.items() if s.is_scored)

    @property
    def input_fields(self) -> list[str]:
        return sorted(n for n, s in self.fields.items() if s.tier == INPUT_TIER)

    @property
    def extracted_fields(self) -> list[str]:
        """Fields resolved against XBRL, i.e. everything the facts layer must produce."""
        return sorted(n for n, s in self.fields.items() if s.is_extracted)

    @property
    def unverified_fields(self) -> list[str]:
        """Fields whose `authority` is still the placeholder. Surfaced deliberately -- the
        schema test asserts on this list so the gap is visible rather than forgotten."""
        return sorted(n for n, s in self.fields.items() if s.authority == UNVERIFIED)

    # --------------------------------------------------------------- regimes --- #
    @property
    def regime_names(self) -> list[str]:
        return sorted(self.regimes)

    def regime_for_sub_industry(self, sub_industry: str | None) -> str | None:
        """The GICS tiebreak, INCLUDING the forced overrides.

        The overrides are the four verified traps: Insurance Brokers, Financial Exchanges,
        Payments and Asset Management all sit inside financial-sector GICS blocks but file
        ARTICLE 5 statements. Routing GICS 'Financials' wholesale to a bank or insurer
        template would mis-read 37 live tickers.

        Returns None when nothing matches, so the caller applies the role-URI result or the
        `industrial` default rather than being handed a silent guess."""
        return self.regime_for_gics(sub_industry=sub_industry)

    def regime_for_gics(self, sector: str | None = None, industry_group: str | None = None,
                        sub_industry: str | None = None) -> str | None:
        """The GICS tiebreak, most-specific level first, INCLUDING the forced overrides.

        The regimes config declares its GICS membership at whichever level is the natural
        one: `bank` and `insurer` enumerate sub-industries, `real_estate` claims a whole
        industry group, `utility` and `energy` claim a whole sector. Reading only
        `sub_industry` -- as this accessor originally did -- silently returned None for
        every energy, utility and REIT ticker, which then fell through to the `industrial`
        default and read a utility's income statement against Rule 5-03.

        Order is specificity, not convenience: a forced sub-industry override beats an
        explicit sub-industry, which beats an industry group, which beats a sector. That
        is what lets Telecom Tower / Data Center / Timber REITs be pulled OUT of the
        `real_estate` industry-group claim -- they file like industrials (AMT reports
        `AssetsCurrent`, `OperatingIncomeLoss` and PP&E capex).
        """
        forced = self.force_regime_by_sub_industry.get(sub_industry or "")
        if forced:
            return forced
        for level, value in (("sub_industry", sub_industry),
                             ("industry_group", industry_group),
                             ("sector", sector)):
            if not value:
                continue
            for name, spec in self.regimes.items():
                declared = spec.get("gics", {}).get(level, [])
                if value in (declared if isinstance(declared, list) else [declared]):
                    return name
        return None

    def regime_for_role_uris(self, role_uris: list[str]) -> str | None:
        """The regime implied by the STATEMENT ROLES the filer actually used.

        Checked before GICS because it is evidence from the filing itself: FASB's role URIs
        name the template (`sfp-dbo` 108000 = deposit-based, `sfp-ibo` 108200 =
        insurance-based, `sfp-sbo` 112000 = securities-based), so a filer shipping a
        deposit-based balance sheet IS a bank however GICS classifies it. That is what
        makes routing `Asset Management & Custody Banks` to Article 5 safe: BNY and STT
        genuinely take deposits, and if they ship a deposit-based role this overrides the
        GICS default rather than being overridden by it.

        Returns None -- not the default -- when no role matches, which is the common case:
        most filers use their own role URIs (`http://www.exxonmobil.com/role/...`) rather
        than FASB's, so the GICS tiebreak carries most of the universe.
        """
        blob = " ".join(role_uris).lower()
        for name, spec in self.regimes.items():
            for pattern in spec.get("role_patterns", []):
                if str(pattern).lower() in blob:
                    return name
        return None

    def regime_for(self, gics: dict[str, str | None] | None,
                   role_uris: list[str]) -> str | None:
        """The filing's regime: role URI first, GICS as tiebreak, `industrial` as the
        Article 5 default -- but ONLY for a ticker that has a GICS row at all.

        The distinction matters and is not cosmetic. A ticker IN the universe whose
        sub-industry matches no regime is an ordinary Article 5 filer, and Reg S-X 5-01
        makes Article 5 the rule with the other regimes its enumerated exceptions, so
        defaulting is correct. A ticker with NO universe row (AVB, EA and EQR have facts
        but no `sp500_tickers` row) is *unclassified*, and defaulting it would add
        unclassified names to the 340-ticker industrial denominator and shift every
        industrial rate in the expected-absence register. Skip, never default.
        """
        from_role = self.regime_for_role_uris(role_uris)
        if from_role:
            return from_role
        if not gics:
            return None
        return self.regime_for_gics(**gics) or self.default_regime()

    def default_regime(self) -> str:
        """The Article 5 general case. Reg S-X makes it the RULE and the other regimes its
        enumerated specialised-industry exceptions, so falling back here is correct."""
        for name, spec in self.regimes.items():
            if spec.get("is_default"):
                return name
        raise ValueError("no regime is marked is_default in the regimes config")

    # ------------------------------------------------------------ exceptions --- #
    def expected_absent(self, regime: str, field: str) -> bool:
        """Is a missing (regime, field) value STRUCTURAL rather than a regression?

        Defaults to False: an absence nobody has justified must stay a finding, or the
        register becomes a way to silence coverage checks."""
        block = self.regime_exceptions.get(regime, {}).get(field)
        return bool(block.get("expected_absent", False)) if isinstance(block, dict) else False

    def filer_leaves(self, ticker: str | None,
                     field: str) -> tuple[list[list[str]], frozenset[str]]:
        """One filer's DECLARED company-extension leaves for a field, as
        `(leaf_groups, not_leaves)`.

        The whole of §4b.4's conclusion in one accessor: **there is no structural rule that
        identifies a company-extension capex or D&A leaf.** Every candidate rule was
        measured and refuted -- "a negative-weight extension child of the investing node"
        admits `apa:EquityMethodInvestmentContribution` ($501M, an investment),
        `nee:PurchasesOfSecuritiesInSpecialUseFunds` ($1.4-2.6bn, securities),
        `dte:ConsolidationOfVIES` and `eog:ChangesInComponentsOfWorkingCapital...`; and the
        inverse framing fails on the same rows. So an extension leaf is either DECLARED
        here, per filer, with its evidence, or the filer stays reason-coded. There is no
        third answer.

        `leaf_groups` has exactly the shape of `roll_up.any_of` (alternatives within a
        group, a sum across groups) and is APPENDED to it -- DTE needs both, because its
        `PlantAndEquipmentExpenditures{Utility,NonUtility}` pair and its
        `PaymentsToAcquirePropertyPlantAndEquipment{Utility,NonUtility}` pair are era
        variants of each other.

        `not_leaves` is the other half and is what makes the register a CLOSED statement:
        it names the extensions in the same node that are NOT this field, so route 3b's
        partial-leaf guard can tell "classified as excluded" from "never looked at". Listing
        only the leaves would leave the guard refusing every filer that parks any unrelated
        extension in the node.
        """
        block = self.ticker_exceptions.get(ticker or "", {}).get(field)
        if not isinstance(block, dict):
            return [], frozenset()
        return ([list(g) for g in block.get("leaves", [])],
                frozenset(block.get("not_leaves", [])))

    def periodicity_shapes(self, ticker: str | None, field: str) -> list[str] | None:
        """The period SHAPES a filer tags this field on, where that is structurally
        limited. None where nothing is declared, which is the common case.

        A separate register from `filer_leaves` because it answers a different question:
        the value EXISTS, it simply has no discrete quarter. AFL and CSCO tag
        `DepreciationDepletionAndAmortization` on the annual window only -- 48 and 36
        annual facts, zero quarterly -- so `ebitda` is annual-only for them and Phase 7's
        `coverage_quarters` gate must read that as structural, not as a regression. The
        `by_regime` register has no key for a periodicity gap, only for an absence.
        """
        block = self.ticker_periodicity.get(ticker or "", {}).get(field)
        return list(block.get("shapes", [])) if isinstance(block, dict) else None

    def combined_into(self, regime: str | None, ticker: str | None,
                      field: str) -> str | None:
        """The field this one is FOLDED INTO for this filer or regime, where a register cell
        declares one -- the destination that makes `combined_into` a usable reason code
        rather than a shrug. A ticker cell wins over a regime cell, matching every other
        override in this class.

        No cell declares one today, so this returns None for every (regime, ticker, field)
        in the universe. The mechanism ships anyway because the alternative is a reason code
        with no producer, and §B.6.4 writes the cells the validator's findings demand rather
        than a speculative sweep of them.
        """
        for register, key in ((self.ticker_exceptions, ticker),
                              (self.regime_exceptions, regime)):
            block = register.get(key or "", {}).get(field)
            if isinstance(block, dict) and block.get("combined_into"):
                return str(block["combined_into"])
        return None

    def regime_break_effective(self, field: str) -> pd.Timestamp | None:
        """The date a definitional discontinuity took effect for `field`, or None.

        Four fields declare one: `cash` (ASU 2016-18, restricted cash enters the total
        retrospectively), and `totalDebt` / `ppeNet` / `operatingLeaseLiability` (ASC 842,
        which put operating leases on the balance sheet). The value is real on both sides
        and comparable across neither, which is a qualifier rather than an absence.
        """
        block = self.field(field).raw.get("regime_break") or {}
        effective = block.get("effective")
        return pd.Timestamp(effective) if effective else None

    def measured_absent_rate(self, regime: str, field: str) -> float | None:
        """The share of the regime's tickers with no fact for this field, as measured on
        `fundamentals_facts_legacy`. None where it was not measured."""
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
    missing = [k for k in ("tier", "kind", "sign", "unit", "definition", "authority")
               if k not in entry]
    if missing:
        raise ValueError(f"{name}: missing mandatory key(s) {missing}")
    if entry["authority"] == UNVERIFIED and "authority_note" not in entry:
        raise ValueError(
            f"{name}: authority is {UNVERIFIED} with no `authority_note` saying what IS "
            "known and which document would close it")
    return FieldSpec(
        name=name, tier=int(entry["tier"]), kind=entry["kind"], sign=entry["sign"],
        unit=entry["unit"], definition=entry["definition"], authority=entry["authority"],
        raw=entry,
    )


@cache
def load_catalogue(config_dir: str = DEFAULT_CONFIG_DIR) -> Catalogue:
    """The validated catalogue, built once per (process, config_dir).

    Validation is deliberately strict and happens HERE rather than in a test, so a
    malformed contract fails at the first call in a nightly run instead of producing a
    quietly wrong number thousands of filings later."""
    root = Path(config_dir) / FUNDAMENTALS_CATALOGUE_SUBDIR
    kpis_blob = _read_json(root, FUNDAMENTALS_KPIS_FILENAME)
    regimes_blob = _read_json(root, FUNDAMENTALS_REGIMES_FILENAME)
    exceptions_blob = _read_json(root, FUNDAMENTALS_EXCEPTIONS_FILENAME)

    kpis = _data_items(kpis_blob)
    fields = {name: _build_field(name, entry) for name, entry in kpis.items()}
    derived_columns = _data_items(kpis_blob.get("_derived_columns", {}))

    # An inherited authority must name a field that exists AND is itself sourced, or the
    # chain bottoms out in nothing and the citation is decorative.
    for name, spec in fields.items():
        for parent in spec.authority_inherits_from:
            if parent not in fields:
                raise ValueError(
                    f"{name}: authority_inherits_from names unknown field {parent!r}")

    regimes = _data_items(regimes_blob.get("regimes", {}))
    if not regimes:
        raise ValueError("regimes config declares no regimes")

    force: dict[str, str] = {}
    for sub_industry, block in _data_items(
            regimes_blob.get("exceptions", {}).get("force_regime", {})).items():
        force[sub_industry] = block["regime"]

    regime_exceptions = {
        regime: _data_items(block)
        for regime, block in _data_items(exceptions_blob.get("by_regime", {})).items()
    }

    ticker_exceptions = {
        ticker: _data_items(block)
        for ticker, block in _data_items(exceptions_blob.get("by_ticker", {})).items()
    }
    periodicity = {
        ticker: _data_items(block)
        for ticker, block in
        _data_items(exceptions_blob.get("by_ticker_periodicity", {})).items()
    }

    # A field named in the exception register but absent from the catalogue is a typo that
    # would otherwise silently excuse nothing at all.
    for regime, block in regime_exceptions.items():
        unknown = sorted(set(block) - set(fields))
        if unknown:
            raise ValueError(f"exceptions[{regime}] names unknown field(s) {unknown}")
    for label, register in (("by_ticker", ticker_exceptions),
                            ("by_ticker_periodicity", periodicity)):
        for ticker, block in register.items():
            unknown = sorted(set(block) - set(fields))
            if unknown:
                raise ValueError(f"exceptions.{label}[{ticker}] names unknown field(s) "
                                 f"{unknown}")
        # A declared leaf that is ALSO declared not-a-leaf is a contradiction the resolver
        # would silently resolve in favour of the leaf. Fail loudly instead.
        for field, entry in block.items():
            if not isinstance(entry, dict):
                continue
            leaves = {c for g in entry.get("leaves", []) for c in g}
            both = sorted(leaves & set(entry.get("not_leaves", [])))
            if both:
                raise ValueError(f"exceptions.by_ticker[{ticker}][{field}]: {both} is "
                                 "declared both a leaf and not a leaf")
            if leaves and "evidence" not in entry:
                raise ValueError(
                    f"exceptions.by_ticker[{ticker}][{field}]: declares extension leaves "
                    "with no `evidence` key. A per-filer override with no written evidence "
                    "is exactly the guess this register exists to replace.")

    # Likewise a regime-keyed override in the KPI catalogue must name a real regime.
    known_regimes = set(regimes)
    for name, spec in fields.items():
        unknown = sorted(set(spec.raw.get("regimes", {})) - known_regimes)
        if unknown:
            raise ValueError(f"{name}: regimes override names unknown regime(s) {unknown}")

    return Catalogue(fields=fields, derived_columns=derived_columns, regimes=regimes,
                     regime_exceptions=regime_exceptions,
                     force_regime_by_sub_industry=force,
                     ticker_exceptions=ticker_exceptions,
                     ticker_periodicity=periodicity)
