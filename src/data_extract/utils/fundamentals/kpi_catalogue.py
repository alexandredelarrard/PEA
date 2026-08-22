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

#: Subdirectory of the config directory holding the three files. TODO(rebuild Phase 3): move
#: these four literals to `src/constants/constants.py` alongside the other `*_FILENAME`
#: entries, as part of the batched risk-zone diff -- the plan directs it there and
#: `constants/` needs approval.
CATALOGUE_SUBDIR = "fundamentals"
KPIS_FILENAME = "fundamentals_kpis.json"
REGIMES_FILENAME = "fundamentals_regimes.json"
EXCEPTIONS_FILENAME = "fundamentals_exceptions.json"

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
        """Is this field resolved against XBRL concepts at all, or computed from others?"""
        return self.kind in EXTRACTED_KINDS

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


@dataclass(frozen=True)
class Catalogue:
    """The three loaded files, validated, with lookups precomputed."""

    fields: dict[str, FieldSpec]
    derived_columns: dict[str, str]
    regimes: dict[str, Any]
    regime_exceptions: dict[str, Any]
    force_regime_by_sub_industry: dict[str, str]

    @property
    def all_column_names(self) -> set[str]:
        """Every name `fundamentals_history` can carry: extracted fields plus the columns
        it computes. A `feeds` reference may name either."""
        return set(self.fields) | set(self.derived_columns)

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
        if not sub_industry:
            return None
        forced = self.force_regime_by_sub_industry.get(sub_industry)
        if forced:
            return forced
        for name, spec in self.regimes.items():
            if sub_industry in spec.get("gics", {}).get("sub_industry", []):
                return name
        return None

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
    root = Path(config_dir) / CATALOGUE_SUBDIR
    kpis_blob = _read_json(root, KPIS_FILENAME)
    regimes_blob = _read_json(root, REGIMES_FILENAME)
    exceptions_blob = _read_json(root, EXCEPTIONS_FILENAME)

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

    # A field named in the exception register but absent from the catalogue is a typo that
    # would otherwise silently excuse nothing at all.
    for regime, block in regime_exceptions.items():
        unknown = sorted(set(block) - set(fields))
        if unknown:
            raise ValueError(f"exceptions[{regime}] names unknown field(s) {unknown}")

    # Likewise a regime-keyed override in the KPI catalogue must name a real regime.
    known_regimes = set(regimes)
    for name, spec in fields.items():
        unknown = sorted(set(spec.raw.get("regimes", {})) - known_regimes)
        if unknown:
            raise ValueError(f"{name}: regimes override names unknown regime(s) {unknown}")

    return Catalogue(fields=fields, derived_columns=derived_columns, regimes=regimes,
                     regime_exceptions=regime_exceptions,
                     force_regime_by_sub_industry=force)
