"""
field_map.py  (src/data_extract/utils/fundamentals_sharadar/field_map.py)
------------------------------------------------------------------------------------
SF1's 112 vendor columns -> the repo's `HISTORY_STATEMENT_ORDER` vocabulary.

A PURE TRANSFORM. Nothing here reads or writes a table: every function takes a frame and
returns one, so the whole map is testable without a database and phase 4 owns the I/O. The
map itself is `configs/sharadar/sharadar_field_map.json` -- data, not code -- and this module
is deterministic given that file plus its two registers.

## The order, and why it is that order

    apply_zero_rules -> apply_corrections -> rename/negate -> [build_ttm]
                                                             -> deadjust_splits
                                                             -> apply_derived

Zero rules and corrections run FIRST, on the vendor frame, BEFORE anything is summed. A cell
Sharadar zero-filled is unknown, not zero, and a zero that survives into a TTM sum contributes
silently -- it does not propagate as a NULL and nobody can tell afterwards which quarters were
real. Every derived formula runs LAST, on the TTM frame, because a ratio of two TTM levels is
not the TTM of a ratio (decision 31, the basis the SEC path already uses).

⚠ `deadjust_splits` is on the TTM frame too, and that is a CORRECTION: it used to run on the
discrete quarters, which put two split bases inside one four-quarter window and overstated
`epsDiluted` by up to 3.5x for the three filings after every split. See its own docstring.

## Three defects the zero rule cannot reach, and the register that can

`sharadar_zero_rules.json` is keyed by FIELD and matches only `0.0`. Phase 2 found three
defects that are neither:

  * `capex` is POSITIVE on 13 of 1,346 stored rows (11 of them GS). The cells are positive,
    not zero, so an unconditional sign flip would write a negative into a column the SEC
    catalogue declares `non_negative`.
  * `intexp` is on a NET basis for NKE -- 14 negative quarters and 0 zeros, so the null rule
    never inspects a single NKE row.
  * the whole share-count and per-share block is retroactively SPLIT-ADJUSTED. Those cells are
    correct numbers on the WRONG BASIS, which no value test can see.

The first two are `sharadar_corrections.json`, keyed by (field, ticker) with a closed action
vocabulary and a mandatory `evidence` field. The third is `deadjust_splits`, which reads the
split events out of `sharadar_actions`.

## Both registers are REFUSED without an `_APPROVED` block

Not a formality. A regenerated proposal is byte-identical to a reviewed decision, so without
the check "human-approved" is a sentence in a docstring, and the one thing these files exist
to guarantee is that a human looked at the entries.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field as dataclass_field
from functools import cache, cached_property
from pathlib import Path

from fractions import Fraction

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT, SHARADAR_APPROVAL_KEY,
    SHARADAR_CONFIG_SUBDIR, SHARADAR_CORRECTION_ACTIONS, SHARADAR_CORRECTIONS_FILENAME,
    SHARADAR_FIELD_MAP_FILENAME, SHARADAR_FLOW_FIELDS, SHARADAR_MAP_KINDS, SHARADAR_MAP_OPS,
    SHARADAR_MAP_SPLIT_BASES, SHARADAR_NEGATE_IF_NON_POSITIVE, SHARADAR_REGISTER_DOC_PREFIX,
    SHARADAR_SF1_COLUMNS, SHARADAR_ZERO_FILLED_FIELDS, SHARADAR_ZERO_RULES_FILENAME,
)
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    DEFAULT_CONFIG_DIR, HISTORY_STATEMENT_ORDER, Catalogue, load_catalogue,
    resolve_config_dir)

log = logging.getLogger(__name__)

#: The three TTM bases a mapped column can carry. `mean` exists for the weighted-average
#: share counts: four quarterly averages SUM to four times the year's average, so summing
#: them is not a trailing twelve, it is a four-fold overstatement.
DURATION, INSTANT, MEAN = "duration", "instant", "mean"

#: The vendor identifier columns every stage carries through untouched. `date` is the FILING
#: date on the Direct channel (Nasdaq Data Link calls it `datekey`); `reportperiod` is the
#: period end. Both are needed downstream and neither is a value.
KEY_COLUMNS: tuple[str, ...] = ("ticker", "dimension", "calendardate", "date",
                                "reportperiod", "fiscalperiod")

#: Applied to `0.0` only, never to a small number. Sharadar writes a LITERAL zero where it
#: has nothing, so an approximate test would null real values that happen to round small.
_EXACT_ZERO = 0.0


@dataclass(frozen=True)
class ColumnSpec:
    """One output column's contract, resolved from the JSON and the KPI catalogue."""

    name: str
    kind: str
    source: str | None = None
    negate: str | None = None
    split_basis: str | None = None
    op: str | None = None
    inputs: tuple[str, ...] = ()
    formula: str | None = None
    basis: str | None = None


@dataclass(frozen=True)
class FieldMap:
    """The loaded, validated map plus both registers. Everything the transform needs, with
    nothing it can reach around: a caller cannot smuggle in an unapproved rule."""

    columns: dict[str, ColumnSpec]
    added: dict[str, ColumnSpec]
    extras: dict[str, ColumnSpec]
    excluded: frozenset[str]
    zero_rules: dict[str, str]
    corrections: dict[str, dict[str, dict]]

    # `cached_property` rather than `property`: these are read inside per-field loops, and
    # rebuilding an 88-key dict on every access made `measure_gaps` do it ~120 times per run.
    # It writes through `__dict__`, which a frozen dataclass still permits.
    @cached_property
    def outputs(self) -> dict[str, ColumnSpec]:
        """Every column the transform emits: the 60 contract names, the 3 added ones, and
        the Sharadar extras under their own names."""
        return {**self.columns, **self.added, **self.extras}

    @cached_property
    def direct(self) -> dict[str, ColumnSpec]:
        return {n: s for n, s in self.outputs.items() if s.kind == "direct"}

    @cached_property
    def derived(self) -> dict[str, ColumnSpec]:
        return {n: s for n, s in self.outputs.items() if s.kind == "derived"}

    @cached_property
    def sec_owned(self) -> list[str]:
        """The columns the SEC layer owns (D18). NaN here; phase 4 merges them in."""
        return sorted(n for n, s in self.outputs.items() if s.kind == "sec")


@dataclass
class TranslationReport:
    """What the transform REMOVED, counted. Every branch that can destroy a value reports
    here rather than doing it quietly -- `negate: if_non_positive` in particular exists to
    null cells, and a silent null is indistinguishable from a column the vendor never sent.
    """

    rows_in: int = 0
    zero_nulled: dict[str, int] = dataclass_field(default_factory=dict)
    corrected: dict[str, int] = dataclass_field(default_factory=dict)
    negation_nulled: dict[str, int] = dataclass_field(default_factory=dict)
    split_deadjusted: dict[str, int] = dataclass_field(default_factory=dict)
    splits_applied: list[str] = dataclass_field(default_factory=list)
    splits_rejected: list[str] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        """One block a human can read in a log or a test's sanity print."""
        return "\n".join([
            f"rows in                 : {self.rows_in}",
            f"zero-rule NULLs         : {sum(self.zero_nulled.values())} "
            f"over {len(self.zero_nulled)} field(s) {self.zero_nulled or '{}'}",
            f"correction NULLs        : {sum(self.corrected.values())} "
            f"{self.corrected or '{}'}",
            f"sign-guard NULLs        : {sum(self.negation_nulled.values())} "
            f"{self.negation_nulled or '{}'}",
            f"split de-adjusted cells : {sum(self.split_deadjusted.values())} "
            f"{self.split_deadjusted or '{}'}",
            f"splits applied          : {self.splits_applied or 'none'}",
            f"splits rejected         : {self.splits_rejected or 'none'}",
        ])


# --------------------------------------------------------------------------- #
# loading and validation                                                       #
# --------------------------------------------------------------------------- #
def _entries(raw: dict) -> dict:
    """The register's real entries -- documentation keys skipped, as both files declare."""
    return {k: v for k, v in raw.items() if not k.startswith(SHARADAR_REGISTER_DOC_PREFIX)}


def _require_approval(raw: dict, path: Path) -> None:
    """Refuse a register a human has not signed.

    The check is the whole governance model. Without it a re-run of the machine proposer
    produces a file indistinguishable from a reviewed decision, and every downstream claim
    that these rules were approved becomes unfalsifiable.
    """
    block = raw.get(SHARADAR_APPROVAL_KEY)
    if not isinstance(block, dict) or not block.get("on") or not block.get("scope"):
        raise RuntimeError(
            f"{path} carries no usable `{SHARADAR_APPROVAL_KEY}` block (needs `on` and "
            f"`scope`). It is a PROPOSAL until a human approves it, and phase 3 refuses to "
            f"run against a proposal -- see the file's own _README.")


def load_zero_rules(config_dir: str = DEFAULT_CONFIG_DIR) -> dict[str, str]:
    """The per-field zero rule, approved. Every one of the 41 documented zero-filled fields
    must carry a rule: a field nobody ruled on would otherwise be silently kept."""
    path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_ZERO_RULES_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_approval(raw, path)
    rules = {name: block["rule"] for name, block in _entries(raw).items()}
    missing = sorted(SHARADAR_ZERO_FILLED_FIELDS - set(rules))
    if missing:
        raise RuntimeError(f"{path} has no rule for {len(missing)} zero-filled field(s): "
                           f"{missing}")
    unknown = sorted(set(rules.values()) - {"null", "keep"})
    if unknown:
        raise RuntimeError(f"{path} uses unknown rule(s) {unknown}; only `null` and `keep` "
                           f"exist")
    return rules


def load_corrections(config_dir: str = DEFAULT_CONFIG_DIR) -> dict[str, dict[str, dict]]:
    """The per-(field, ticker) correction register, approved.

    `evidence` is required on every entry and not merely conventional: this repo has been
    burned by fallbacks with no stated authority, and a correction whose justification is
    "it looked wrong" cannot be re-checked when the roster widens.
    """
    path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_CORRECTIONS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the field map refuses to run without "
                                f"the correction register (phase-3 §0)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_approval(raw, path)
    register = _entries(raw)
    for field, by_ticker in register.items():
        for ticker, entry in by_ticker.items():
            where = f"{path}: {field}/{ticker}"
            action = entry.get("action")
            if action not in SHARADAR_CORRECTION_ACTIONS:
                raise RuntimeError(f"{where} has action {action!r}; the vocabulary is closed "
                                   f"to {sorted(SHARADAR_CORRECTION_ACTIONS)}")
            for key in ("reason", "evidence"):
                if not str(entry.get(key, "")).strip():
                    raise RuntimeError(f"{where} has no `{key}`. Every correction states what "
                                       f"was measured or which filing was read.")
    return register


def _basis_for(name: str, source: str, catalogue: Catalogue) -> str:
    """A direct column's TTM basis, taken from the CATALOGUE where it has an opinion.

    The catalogue is the authority for the 60 contract names, so the basis cannot drift from
    the SEC path's. It has no opinion on one case -- `freeCashflow`, which the catalogue calls
    `derived` (it computes it) while this map takes it straight from Sharadar's `fcf`. There
    the Sharadar flow set decides, and `fcf` is in it.
    """
    spec = catalogue.fields.get(name)
    if spec is not None and spec.kind == "instant":
        return INSTANT
    if spec is not None and spec.kind == "duration":
        return DURATION if spec.is_additive else MEAN
    return DURATION if source in SHARADAR_FLOW_FIELDS else INSTANT


def _spec_from(name: str, entry: dict, *, catalogue, path: Path,
               basis: str | None = None) -> ColumnSpec:
    """One JSON entry -> a validated `ColumnSpec`."""
    kind = entry.get("kind")
    if kind not in SHARADAR_MAP_KINDS:
        raise RuntimeError(f"{path}: {name} has kind {kind!r}; the vocabulary is closed to "
                           f"{sorted(SHARADAR_MAP_KINDS)}")
    source = entry.get("from")
    split_basis = entry.get("split_basis")
    if split_basis is not None and split_basis not in SHARADAR_MAP_SPLIT_BASES:
        raise RuntimeError(f"{path}: {name} has split_basis {split_basis!r}; expected one of "
                           f"{sorted(SHARADAR_MAP_SPLIT_BASES)}")
    negate = entry.get("negate")
    if negate is not None and negate != SHARADAR_NEGATE_IF_NON_POSITIVE:
        raise RuntimeError(
            f"{path}: {name} has negate {negate!r}. The only accepted spelling is "
            f"{SHARADAR_NEGATE_IF_NON_POSITIVE!r} -- `true` flips unconditionally, and 13 of "
            f"1,346 stored rows carry a positive `capex`, so it writes a negative into a "
            f"`non_negative` column.")

    if kind == "direct":
        if source not in set(SHARADAR_SF1_COLUMNS):
            raise RuntimeError(f"{path}: {name} maps from {source!r}, which SF1 does not "
                               f"deliver ({len(SHARADAR_SF1_COLUMNS)} columns)")
        return ColumnSpec(name=name, kind=kind, source=source, negate=negate,
                          split_basis=split_basis,
                          basis=basis or _basis_for(name, source, catalogue))
    if kind == "derived":
        op = entry.get("op")
        if op not in SHARADAR_MAP_OPS:
            raise RuntimeError(f"{path}: {name} has op {op!r}; the vocabulary is closed to "
                               f"{sorted(SHARADAR_MAP_OPS)}")
        inputs = tuple(entry.get("inputs", ()))
        formula = entry.get("formula")
        _assert_formula_matches(name, op, inputs, formula, path)
        return ColumnSpec(name=name, kind=kind, op=op, inputs=inputs, formula=formula)
    return ColumnSpec(name=name, kind=kind, basis=basis)


def _assert_formula_matches(name: str, op: str, inputs: tuple[str, ...], formula: str | None,
                            path: Path) -> None:
    """The prose `formula` is for a reader; `op` + `inputs` is what runs. Asserting they
    agree keeps the config honest -- a formula string nobody executes is a comment that drifts.
    """
    if op == "quarter":
        expected = f"the DISCRETE quarter's {inputs[0]}" if len(inputs) == 1 else None
    elif op == "sum":
        expected = " + ".join(inputs)
    elif op == "ratio":
        expected = " / ".join(inputs) if len(inputs) == 2 else None
    else:
        expected = f"{inputs[0]} / {inputs[1]} - 1" if len(inputs) == 2 else None
    if expected is None:
        raise RuntimeError(f"{path}: {name} op {op!r} has the wrong arity for inputs {inputs}")
    if formula != expected:
        raise RuntimeError(f"{path}: {name} declares formula {formula!r} but `op`/`inputs` "
                           f"compute {expected!r}")


def load_field_map(config_dir: str | None = DEFAULT_CONFIG_DIR) -> FieldMap:
    """The validated map, built once per (process, config DIRECTORY). It was the only loader
    in the family with no cache at all, while calling the cached `load_catalogue` inside
    itself -- so every caller re-read and re-validated both registers."""
    return _field_map_at(config_dir)


@cache
def _field_map_at(config_dir: str) -> FieldMap:
    """Load and validate the map, both registers and the contract they must satisfy.

    FAILS LOUDLY, and the two failures worth naming: a `HISTORY_STATEMENT_ORDER` name with no
    entry (the merged table would carry a column nothing fills, and the contract asserts by
    LIST EQUALITY, so it would pass), and a `from` naming a column SF1 does not deliver
    (`fields=` silently drops an unavailable field rather than erroring, so a typo yields a
    missing column and no warning).
    """
    path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_FIELD_MAP_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    catalogue = load_catalogue(config_dir)

    columns = {n: _spec_from(n, e, catalogue=catalogue, path=path)
               for n, e in raw["columns"].items()}
    added = {n: _spec_from(n, e, catalogue=catalogue, path=path)
             for n, e in raw["added_columns"].items()}
    # An extra is keyed by its VENDOR name and emitted under its repo one. `to` is required:
    # D16 says "where no repo counterpart exists, keep Sharadar's own name", and that was read
    # as "keep Sharadar's own SPELLING" -- which left `ncfx`, `prefdivis` and `accoci` sitting
    # in a table whose other 63 columns are camelCase. The vendor spelling stays the KEY, so
    # `sharadar_fundamentals` is still the thing this file maps FROM.
    extras = {e["to"]: ColumnSpec(name=e["to"], kind="direct", source=n, basis=e["basis"],
                                  split_basis=e.get("split_basis"))
              for n, e in raw["extras"].items()}

    unmapped = [n for n in HISTORY_STATEMENT_ORDER if n not in columns]
    if unmapped:
        raise RuntimeError(f"{path} leaves {len(unmapped)} contract column(s) unmapped: "
                           f"{unmapped}")
    stray = sorted(set(columns) - set(HISTORY_STATEMENT_ORDER))
    if stray:
        raise RuntimeError(f"{path} maps {stray}, which are not in HISTORY_STATEMENT_ORDER. "
                           f"A column beyond the 60 belongs in `added_columns`.")
    # ⚠ Checked on the SOURCE, not on the emitted name. An extra is now keyed by its vendor
    # column and emitted under a repo one, so testing the OUTPUT against `SHARADAR_SF1_COLUMNS`
    # would reject every correctly renamed extra and accept a `to` that shadows a contract
    # column -- exactly backwards on both counts.
    for name, spec in extras.items():
        if spec.source not in set(SHARADAR_SF1_COLUMNS):
            raise RuntimeError(f"{path}: extra {name!r} reads {spec.source!r}, which is not an "
                               f"SF1 column")
        if name in set(HISTORY_STATEMENT_ORDER) or name in columns or name in added:
            raise RuntimeError(f"{path}: extra {spec.source!r} renames to {name!r}, which is "
                               f"already a contract column. Two sources would write one "
                               f"column and the later one would win silently.")
        if spec.basis not in (DURATION, INSTANT):
            raise RuntimeError(f"{path}: extra {name!r} has basis {spec.basis!r}; expected "
                               f"{DURATION!r} or {INSTANT!r}")
    collisions = sorted(n for n in extras if sum(1 for e in raw["extras"].values()
                                                 if e["to"] == n) > 1)
    if collisions:
        raise RuntimeError(f"{path}: {collisions} is the `to` of more than one extra; the "
                           f"dict would silently keep only the last.")

    field_map = FieldMap(columns=columns, added=added, extras=extras,
                         excluded=frozenset(raw["excluded"]),
                         zero_rules=load_zero_rules(config_dir),
                         corrections=load_corrections(config_dir))
    _assert_derived_inputs_resolve(field_map, path)
    return field_map


def _assert_derived_inputs_resolve(field_map: FieldMap, path: Path) -> None:
    """Every derived input must be a column the transform actually produces, and no derived
    column may depend on another -- the formulas are one pass over the TTM frame, and a
    hidden dependency would evaluate against a column that is still NaN."""
    outputs, derived = field_map.outputs, field_map.derived
    for name, spec in derived.items():
        for source in spec.inputs:
            if source not in outputs:
                raise RuntimeError(f"{path}: {name} reads {source!r}, which the map does not "
                                   f"produce")
            if source in derived:
                raise RuntimeError(f"{path}: {name} reads the derived column {source!r}. "
                                   f"Formulas run in ONE pass, so a derived input would be "
                                   f"read before it is computed.")


# --------------------------------------------------------------------------- #
# the vendor-frame cleaning stages                                             #
# --------------------------------------------------------------------------- #
def apply_zero_rules(frame: pd.DataFrame, rules: dict[str, str], *,
                     report: TranslationReport | None = None) -> pd.DataFrame:
    """Replace `0.0` with NaN for every field ruled `"null"`.

    Runs on the VENDOR frame, before any sum. A zero that survives into a TTM contributes
    silently and is unrecoverable afterwards; a NaN propagates, which is the honest answer for
    a cell whose value Sharadar never had.

    Fails loudly on a zero-filled field present in the frame with no rule -- the register is
    the decision record, and a field nobody ruled on must not default to "keep".
    """
    ungoverned = sorted((SHARADAR_ZERO_FILLED_FIELDS & set(frame.columns)) - set(rules))
    if ungoverned:
        raise RuntimeError(f"no zero rule for {ungoverned}; every zero-filled field in the "
                           f"frame needs one")
    out = frame.copy()
    for name, rule in rules.items():
        if rule != "null" or name not in out.columns:
            continue
        hit = out[name] == _EXACT_ZERO
        count = int(hit.sum())
        if count:
            out.loc[hit, name] = np.nan
            if report is not None:
                report.zero_nulled[name] = report.zero_nulled.get(name, 0) + count
    return out


def apply_corrections(frame: pd.DataFrame, corrections: dict[str, dict[str, dict]], *,
                      report: TranslationReport | None = None) -> pd.DataFrame:
    """Apply the (field, ticker) register to the VENDOR frame.

    Before the field map, so `fundamentals_sharadar` stays a faithful record of what the
    vendor sent (D7) and every correction is one auditable, reversible step rather than an
    `if ticker == "GS"` scattered through the rename.
    """
    out = frame.copy()
    for field, by_ticker in corrections.items():
        if field not in out.columns:
            continue
        for ticker, entry in by_ticker.items():
            rows = out["ticker"] == ticker
            action = entry["action"]
            if action == "null":
                hit = rows & out[field].notna()
            elif action == "null_if_positive":
                hit = rows & (out[field] > 0)
            else:
                hit = rows & (out[field] < 0)
            count = int(hit.sum())
            if count:
                out.loc[hit, field] = np.nan
                if report is not None:
                    key = f"{field}/{ticker}:{action}"
                    report.corrected[key] = report.corrected.get(key, 0) + count
    return out


# --------------------------------------------------------------------------- #
# the split de-adjustment                                                      #
# --------------------------------------------------------------------------- #
#: Calendar days within which a Sharadar and a yfinance event are the SAME event. Vendors
#: disagree by a day or two on whether an ex-date is the record or the trading date; no
#: ticker in the universe has ever split twice inside a week.
SPLIT_MATCH_DAYS = 7
#: How exactly a ratio must reproduce a small-integer fraction to read as a split. Tight on
#: purpose: a spinoff factor lands NEAR a simple fraction without being one (BDX 1.272 is
#: 14/11 to 7e-4), and a loose tolerance would readmit exactly what this test exists to reject.
SPLIT_INTEGER_TOL = 1e-6
#: Largest denominator a genuine split ratio may have. Real splits are ratios of SMALL
#: integers -- 2:1, 3:2, 4:3, 5:4, 1:5, 1:10, 1:20 -- and stock dividends are 21/20 or 11/10.
#: Corporate-action artefacts are not: BDX's 1.025 is 41/40, SJM's 0.945 is 189/200.
SPLIT_MAX_DENOMINATOR = 20


def _is_split_shaped(ratio: float) -> bool:
    """Whether `ratio` has the shape of a share split rather than a corporate-action factor.

    A share split is declared as "n new shares for every m old", so its ratio is a fraction
    of SMALL integers: 2/1, 3/2, 4/3, 5/4, 20/1, 50/1, and the reverse cases 1/2, 1/5, 1/10,
    1/20. Stock dividends sit at the edge of the same family (21/20 = a 5% stock dividend,
    11/10 = 10%) and ARE genuine share-count events, so they belong in.

    What does NOT reproduce such a fraction is a price-adjustment factor from a spinoff,
    merger or exchange offer. Those are dividend-yield-like numbers that merely land near a
    fraction: BDX 1.025 (41/40) and 1.272, CMCSA 1.067 (16/15), SJM 0.945 (189/200), WTW
    0.3775, CCL 0.0012. They must never reach a share count -- BDX's two factors compound to
    1.304 and pushed its whole PIT series 23% off the SEC cover page.

    ⚠ This is applied to BOTH vendors, not just Sharadar. yfinance's `Stock Splits` column
    carries spinoff factors too, which is how BDX and CMCSA broke when the union rule trusted
    it unconditionally.
    """
    if not ratio or ratio <= 0 or not np.isfinite(ratio):
        return False
    frac = Fraction(float(ratio)).limit_denominator(SPLIT_MAX_DENOMINATOR)
    return abs(float(frac) - ratio) < SPLIT_INTEGER_TOL


def split_events(actions: pd.DataFrame, yf_splits: pd.DataFrame | None = None, *,
                 report: TranslationReport | None = None) -> pd.DataFrame:
    """The GENUINE share splits, as `(ticker, date, value)`, from BOTH vendors.

    ⚠ A `sharadar_actions` `split` row is not always a share split, and reading it as one is
    a 100%-error trap. HON carries `split` = 0.5 dated 2026-06-29 co-dated with `spinoff` = 1
    and `spinoffdividend` = 221.01 (Honeywell Aerospace): it is the SPINOFF'S PRICE
    ADJUSTMENT, not a share-count event, and HON's own cover page proves it -- `sharesbas`
    reads 316,826,560 on 2026-04-23 and 316,940,010 on 2026-07-23, unchanged across the date.
    Applying it would have DOUBLED every HON share count in the history. So a Sharadar
    candidate counts only when no `spinoff` row shares its `(ticker, date)`.

    ⚠ `sharadar_actions` is also INCOMPLETE, which is worse than being wrong, because it is
    wrong PER TICKER: it misses GOOGL 2022 x20, NVDA 2021 x4, TSLA 2022 x3, AVGO 2024 x10,
    CMG 2024 x50, ANET 2024 x4, BKNG 2026 x25, MNST 2023 and 2026 x2, and AMCR 2026 x0.2 --
    so AAPL is de-adjusted and GOOGL is not, and the same column sits on different bases in
    one cross-section. yfinance (`prices_splits`) has every one of those events, so the two
    sources cross-validate:

      * in BOTH                         -> keep, on the YFINANCE date. That is the ex-date
                                           Yahoo adjusted its own prices on, so the share
                                           factor and `close_split` step on the same day.
                                           WTW 2016-01-05 x0.3775 is genuine and lands here
                                           despite its odd ratio.
      * yfinance ONLY, split-shaped     -> keep. This is the nine-hole fix.
      * yfinance ONLY, not split-shaped -> DROP. yfinance's `Stock Splits` column also
                                           carries SPINOFF factors: BDX 2022-04-01 x1.025
                                           and 2026-02-10 x1.272 compound to 1.304 and put
                                           BDX's whole PIT series 23% off the SEC cover page.
      * Sharadar ONLY, split-shaped     -> keep, and WARN. Real but uncorroborated.
      * Sharadar ONLY, not split-shaped -> DROP. The false-positive signature: SJM
                                           2002-05-30 x0.945 is a merger share-issuance
                                           factor, absent from yfinance; CCL's 0.0012
                                           compounds into a 1000x error.

    `yf_splits=None` falls back to Sharadar alone, so a cold `prices_splits` degrades to the
    previous behaviour rather than emptying the list.
    """
    # An EMPTY frame still needs a datetime `date`: the union subtracts two date columns, and
    # on a bare `pd.DataFrame(columns=...)` that column is `object` and the subtraction raises
    # TypeError -- turning "this ticker has no Sharadar rows" into a crash.
    empty = pd.DataFrame({"ticker": pd.Series(dtype="object"),
                          "date": pd.Series(dtype="datetime64[ns]"),
                          "value": pd.Series(dtype="float64"),
                          "label": pd.Series(dtype="object")})
    if actions is None or actions.empty:
        sharadar = empty
    else:
        frame = actions.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        spinoffs = set(map(tuple, frame.loc[frame["action"] == SHARADAR_ACTION_SPINOFF,
                                            ["ticker", "date"]].to_numpy()))
        candidates = frame[frame["action"] == SHARADAR_ACTION_SPLIT]
        kept, spinoff_dropped = [], []
        for _, row in candidates.iterrows():
            label = f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']}"
            target = spinoff_dropped if (row["ticker"], row["date"]) in spinoffs else kept
            target.append({"ticker": row["ticker"], "date": row["date"],
                           "value": float(row["value"]), "label": label})
        if spinoff_dropped:
            log.warning("rejected %d `split` row(s) co-dated with a spinoff (price "
                        "adjustment, not a share-count event): %s", len(spinoff_dropped),
                        ", ".join(r["label"] for r in spinoff_dropped))
            if report is not None:
                report.splits_rejected.extend(r["label"] for r in spinoff_dropped)
        sharadar = (pd.DataFrame(kept, columns=["ticker", "date", "value", "label"])
                    if kept else empty)

    yf = pd.DataFrame(columns=["ticker", "date", "value"])
    if yf_splits is not None and not yf_splits.empty:
        yf = yf_splits.rename(columns={"ratio": "value"})[["ticker", "date", "value"]].copy()
        yf["date"] = pd.to_datetime(yf["date"], errors="coerce")
        yf = yf.dropna(subset=["date", "value"])
        yf = yf[yf["value"] != 0.0]

    return union_split_sources(sharadar, yf, report=report)


def union_split_sources(sharadar: pd.DataFrame, yf: pd.DataFrame, *,
                        report: TranslationReport | None = None) -> pd.DataFrame:
    """Apply the four-case corroboration rule to two already-cleaned event lists.

    Split out from `split_events` so the rule is testable on a synthetic fixture with no DB
    and no network -- it is the part that decides whether a share count is right."""
    matched_sharadar: set[int] = set()
    events: list[dict] = []
    kept_both = kept_yf = kept_sharadar = 0

    for _, row in yf.iterrows():
        near = sharadar.index[(sharadar["ticker"] == row["ticker"])
                              & ((sharadar["date"] - row["date"]).abs()
                                 <= pd.Timedelta(days=SPLIT_MATCH_DAYS))]
        matched_sharadar.update(near)
        # The yfinance DATE and RATIO win on a match: Yahoo adjusted its own `close_split` on
        # that day, and the whole point of the fix is that the share factor and the price
        # factor cancel -- which they only do if they step together.
        # Corroboration overrides shape: an event BOTH vendors report is genuine whatever
        # its ratio (WTW's 0.3775). Uncorroborated, the shape test is all there is.
        if len(near):
            kept_both += 1
        elif _is_split_shaped(float(row["value"])):
            kept_yf += 1
        else:
            if report is not None:
                report.splits_rejected.append(
                    f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']} "
                    "(yfinance-only, not split-shaped)")
            continue
        events.append({"ticker": row["ticker"], "date": row["date"],
                       "value": float(row["value"])})

    uncorroborated = []
    for idx, row in sharadar.iterrows():
        if idx in matched_sharadar:
            continue
        label = row["label"] if "label" in row else (
            f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']}")
        if _is_split_shaped(float(row["value"])):
            events.append({"ticker": row["ticker"], "date": row["date"],
                           "value": float(row["value"])})
            uncorroborated.append(label)
            kept_sharadar += 1
        elif report is not None:
            report.splits_rejected.append(f"{label} (uncorroborated, not split-shaped)")

    if uncorroborated:
        log.warning("%d split event(s) in sharadar_actions but NOT in yfinance, kept because "
                    "the ratio is split-shaped -- review: %s",
                    len(uncorroborated), ", ".join(uncorroborated))

    out = (pd.DataFrame(events, columns=["ticker", "date", "value"])
           .drop_duplicates(subset=["ticker", "date"])
           .sort_values(["ticker", "date"])
           .reset_index(drop=True))
    log.info("split events: %d corroborated, %d yfinance-only, %d sharadar-only -> %d total",
             kept_both, kept_yf, kept_sharadar, len(out))
    if report is not None:
        report.splits_applied.extend(
            f"{r.ticker} {pd.Timestamp(r.date).date()} x{r.value}" for r in out.itertuples())
    return out


def forward_split_factor(tickers: pd.Series, dates: pd.Series,
                         splits: pd.DataFrame) -> pd.Series:
    """The product of every genuine split dated STRICTLY AFTER each row's own filing date.

    That product is exactly the factor Sharadar applied retroactively, so dividing a count by
    it (or multiplying a per-share figure) recovers the as-filed basis. A row filed after all
    of a ticker's splits gets 1.0 and is untouched.
    """
    factor = pd.Series(1.0, index=dates.index)
    if splits.empty:
        return factor
    stamps = pd.to_datetime(dates, errors="coerce")
    for _, split in splits.iterrows():
        hit = (tickers == split["ticker"]) & (stamps < split["date"])
        factor.loc[hit] = factor.loc[hit] * float(split["value"])
    return factor


def deadjust_splits(frame: pd.DataFrame, field_map: FieldMap, actions: pd.DataFrame | None,
                    yf_splits: pd.DataFrame | None = None, *,
                    report: TranslationReport | None = None) -> pd.DataFrame:
    """Undo Sharadar's RETROACTIVE split adjustment on the columns that carry it.

    SF1 reports a pre-split quarter on the POST-split basis across its whole share block --
    `sharesbas`, `shareswa`, `shareswadil`, `eps`, `epsdil` and `dps` -- and `sharefactor` is
    1.0 on every affected row, so nothing in the payload flags it. That makes those columns
    NOT POINT-IN-TIME: anything multiplying one by an as-filed price is wrong by the split
    factor for every date before the split.

    Counts are DIVIDED and per-share figures MULTIPLIED. Verified against the SEC cover page:
    WMT matches on 10 of 10 pre-split rows exactly, NVDA on 10 of 11 (the 11th differs only by
    Sharadar's own 4-significant-figure rounding).

    ⚠ RUNS ON THE **TTM** FRAME, after the four-quarter aggregation -- not on the discrete
    quarters, which is where it used to run and where it was WRONG.

    Sharadar stores the whole series on ONE basis (today's), so `as_filed = adjusted / F`
    where `F` is the factor at the row's own filing date. De-adjusting each QUARTER first put
    four numbers on two different bases into one window whenever that window straddled a
    split, and the mean of those is on no basis at all. Measured on the 3 splits this roster
    has: NVDA's 2024-08-28 `dilutedShares` came out 8.08bn -- the mean of three de-adjusted
    2.49bn quarters and one 24.9bn one -- against a true 24.9bn, and `epsDiluted` read 6.56
    against an as-filed ~2.16. AMZN was worse, at 3.48x.

    Aggregating FIRST and de-adjusting the RESULT once fixes it, because every quarter in the
    window shares the vendor's single basis, so the mean is coherent; dividing it by `F` at
    the row's date then maps it to the basis in force at that date -- which is exactly what
    the filer does when it restates comparatives. Where a window does NOT straddle a split
    the two orders are algebraically identical (`mean(x)/F == mean(x/F)`), which is why only
    9 of 59 rows moved.

    It still keys on each output column's declared `split_basis`, and the TTM frame still
    carries `ticker` and `date`, so nothing in the body had to change.
    """
    targets = {n: s for n, s in field_map.outputs.items() if s.split_basis}
    if not targets:
        return frame
    splits = split_events(actions, yf_splits, report=report)
    if splits.empty:
        log.warning("no genuine split events available -- the share block stays on Sharadar's "
                    "retroactively adjusted basis, which is NOT point-in-time")
        return frame
    out = frame.copy()
    factor = forward_split_factor(out["ticker"], out["date"], splits)
    touched = factor != 1.0
    for name, spec in targets.items():
        if name not in out.columns:
            continue
        hit = touched & out[name].notna()
        if not hit.any():
            continue
        out.loc[hit, name] = (out.loc[hit, name] / factor[hit] if spec.split_basis == "count"
                              else out.loc[hit, name] * factor[hit])
        if report is not None:
            report.split_deadjusted[name] = int(hit.sum())
    return out


# --------------------------------------------------------------------------- #
# the rename                                                                   #
# --------------------------------------------------------------------------- #
def translate(frame: pd.DataFrame, field_map: FieldMap, *,
              report: TranslationReport | None = None) -> pd.DataFrame:
    """A vendor ARQ frame -> the repo-named ARQ frame, still on the DISCRETE-quarter grain.

    Zero rules, then corrections, then the direct renames with their sign guard. Derived
    formulas are NOT evaluated here: they run on the TTM frame, which `build_ttm` produces
    (decision 31).

    ⚠ NEITHER is the split de-adjustment, and it used to be. It moved into `build_ttm`, AFTER
    the four-quarter aggregation -- see `deadjust_splits` for the measurement that forced the
    move. Taking `actions=` here would now be a silently ignored argument, so the parameter is
    gone rather than deprecated.

    `sec` and `null` columns are emitted as all-NaN. They are part of the contract -- which
    asserts by LIST EQUALITY -- and phase 4 fills the 15 SEC-owned ones from
    `fundamentals_history_sec`. The 3 `null` ones have no source anywhere and stay empty.
    """
    report = report if report is not None else TranslationReport()
    report.rows_in = len(frame)

    missing = [s.source for s in field_map.direct.values() if s.source not in frame.columns]
    if missing:
        raise RuntimeError(f"the vendor frame is missing {len(missing)} mapped column(s): "
                           f"{sorted(set(missing))}. `fields=` silently drops an unavailable "
                           f"field, so this is a projection or a typo, never an empty column.")

    cleaned = apply_zero_rules(frame, field_map.zero_rules, report=report)
    cleaned = apply_corrections(cleaned, field_map.corrections, report=report)

    # Accumulated then concatenated ONCE: ~90 single-column inserts refragment the block
    # manager on every assignment and make pandas warn about it.
    columns: dict[str, pd.Series] = {}
    for name, spec in field_map.direct.items():
        values = cleaned[spec.source].astype("float64")
        if spec.negate == SHARADAR_NEGATE_IF_NON_POSITIVE:
            values = _negate_if_non_positive(values, name, report)
        columns[name] = values
    for name, spec in field_map.outputs.items():
        if spec.kind in ("sec", "null"):
            columns[name] = pd.Series(np.nan, index=cleaned.index)

    keys = cleaned[[c for c in KEY_COLUMNS if c in cleaned.columns]]
    return pd.concat([keys, pd.DataFrame(columns, index=cleaned.index)], axis=1)


def _negate_if_non_positive(values: pd.Series, name: str,
                            report: TranslationReport) -> pd.Series:
    """Flip the sign where Sharadar's convention holds; NULL the cells where it does not.

    `capex` is the only user. The repo declares it `sign: non_negative` and Sharadar stores it
    negative, but the convention is NOT universal -- 13 of 1,346 stored rows are positive (11
    of them GS, plus BA, CVX and IBM). An unconditional flip turns each of those into a
    negative the column cannot hold, so the exceptions are nulled and COUNTED.
    """
    violations = values > 0
    count = int(violations.sum())
    out = -values
    if count:
        out[violations] = np.nan
        report.negation_nulled[name] = report.negation_nulled.get(name, 0) + count
        log.warning("%s: %d row(s) violate Sharadar's sign convention and were NULLed rather "
                    "than flipped into a negative", name, count)
    return out


# --------------------------------------------------------------------------- #
# the derived formulas -- LAST, on the TTM frame                               #
# --------------------------------------------------------------------------- #
def apply_derived(frame: pd.DataFrame, field_map: FieldMap,
                  only: set[str] | None = None) -> pd.DataFrame:
    """Evaluate every `derived` column on the TTM frame.

    A ratio of two TTM levels, never the TTM of a ratio: `profitMargins` is TTM net income
    over TTM revenue (decision 31). `op` == `quarter` is skipped here -- `build_ttm` owns the
    discrete-quarter columns, because only it holds the un-summed quarter.

    A zero DENOMINATOR yields NaN, not an infinity. `x / 0` survives every plausibility check
    downstream and then poisons a z-score.

    `only` restricts the pass to a NAMED subset. Phase 4 needs it: after the merge, the
    columns whose inputs actually changed must be recomputed (`stockholdersEquityInclNci`'s
    NCI leg is SEC-owned and does not exist until then) while every OTHER derived column must
    be left exactly as built -- a blanket re-run would quietly undo a registered override.
    One evaluator, two callers, so the formulas cannot drift apart.
    """
    computed: dict[str, pd.Series] = {}
    for name, spec in field_map.derived.items():
        if spec.op == "quarter" or (only is not None and name not in only):
            continue
        missing = [c for c in spec.inputs if c not in frame.columns]
        if missing:
            raise RuntimeError(f"{name} needs {missing}, absent from the TTM frame")
        parts = [frame[c].astype("float64") for c in spec.inputs]
        if spec.op == "sum":
            computed[name] = sum(parts[1:], start=parts[0])
        else:
            numerator, denominator = parts[0], parts[1].replace(0.0, np.nan)
            values = numerator / denominator
            computed[name] = values - 1.0 if spec.op == "ratio_minus_one" else values
    if not computed:
        return frame.copy()
    # every derived name either replaces an existing column or is new; assign in one pass
    return frame.assign(**computed)
