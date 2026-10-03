"""Pure transform from SF1 vendor columns to the repo's `HISTORY_STATEMENT_ORDER` vocabulary.

No table I/O: every function takes and returns a frame. The map is `configs/sharadar/sharadar_field_map.json`
plus two human-approved registers (zero rules, per-(field, ticker) corrections); a register without its
`_APPROVED` block is refused. Order: zero rules -> corrections -> rename/negate on the vendor frame (before
any sum, so a zero-filled cell becomes NaN, not a silent 0 in a TTM), then on the TTM frame `deadjust_splits`
and `apply_derived` (a ratio of TTM levels, never the TTM of a ratio).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from fractions import Fraction
from functools import cache, cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_ACTION_SPINOFF,
    SHARADAR_ACTION_SPLIT,
    SHARADAR_APPROVAL_KEY,
    SHARADAR_CONFIG_SUBDIR,
    SHARADAR_CORRECTION_ACTIONS,
    SHARADAR_CORRECTIONS_FILENAME,
    SHARADAR_FIELD_MAP_FILENAME,
    SHARADAR_FLOW_FIELDS,
    SHARADAR_MAP_KINDS,
    SHARADAR_MAP_OPS,
    SHARADAR_MAP_SPLIT_BASES,
    SHARADAR_NEGATE_IF_NON_POSITIVE,
    SHARADAR_REGISTER_DOC_PREFIX,
    SHARADAR_SF1_COLUMNS,
    SHARADAR_ZERO_FILLED_FIELDS,
    SHARADAR_ZERO_RULES_FILENAME,
)
from src.data_extract.utils.common.config_paths import resolve_config_dir
from src.data_extract.utils.fundamentals.kpi_catalogue import (
    DEFAULT_CONFIG_DIR,
    HISTORY_STATEMENT_ORDER,
    Catalogue,
    load_catalogue,
)

log = logging.getLogger(__name__)

#: The three TTM bases a mapped column can carry; `mean` is for weighted-average share counts, which must not be summed.
DURATION, INSTANT, MEAN = "duration", "instant", "mean"

#: Vendor identifier columns carried through untouched; `date` is the FILING date (`datekey`), `reportperiod` the period end.
KEY_COLUMNS: tuple[str, ...] = ("ticker", "dimension", "calendardate", "date", "reportperiod", "fiscalperiod")

#: Zero rules match a literal `0.0` only, never an approximate small value.
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
    """The loaded, validated map plus both approved registers -- everything the transform needs."""

    columns: dict[str, ColumnSpec]
    added: dict[str, ColumnSpec]
    extras: dict[str, ColumnSpec]
    excluded: frozenset[str]
    zero_rules: dict[str, str]
    corrections: dict[str, dict[str, dict]]

    # cached: read inside per-field loops; a frozen dataclass still permits the `__dict__` write.
    @cached_property
    def outputs(self) -> dict[str, ColumnSpec]:
        """Every emitted column: contract names, added columns and the renamed Sharadar extras."""
        return {**self.columns, **self.added, **self.extras}

    @cached_property
    def direct(self) -> dict[str, ColumnSpec]:
        return {n: s for n, s in self.outputs.items() if s.kind == "direct"}

    @cached_property
    def derived(self) -> dict[str, ColumnSpec]:
        return {n: s for n, s in self.outputs.items() if s.kind == "derived"}

    @cached_property
    def sec_owned(self) -> list[str]:
        """The columns the SEC layer owns; NaN here, filled by the merge."""
        return sorted(n for n, s in self.outputs.items() if s.kind == "sec")


@dataclass
class TranslationReport:
    """Counts of every value the transform nulled or rescaled, plus the splits applied and rejected."""

    rows_in: int = 0
    zero_nulled: dict[str, int] = dataclass_field(default_factory=dict)
    corrected: dict[str, int] = dataclass_field(default_factory=dict)
    negation_nulled: dict[str, int] = dataclass_field(default_factory=dict)
    sign_nulled: dict[str, int] = dataclass_field(default_factory=dict)
    split_deadjusted: dict[str, int] = dataclass_field(default_factory=dict)
    splits_applied: list[str] = dataclass_field(default_factory=list)
    splits_rejected: list[str] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        """One block a human can read in a log or a test's sanity print."""
        return "\n".join(
            [
                f"rows in                 : {self.rows_in}",
                f"zero-rule NULLs         : {sum(self.zero_nulled.values())} over {len(self.zero_nulled)} field(s) {self.zero_nulled or '{}'}",
                f"correction NULLs        : {sum(self.corrected.values())} {self.corrected or '{}'}",
                f"sign-guard NULLs        : {sum(self.negation_nulled.values())} {self.negation_nulled or '{}'}",
                f"declared-sign NULLs     : {sum(self.sign_nulled.values())} {self.sign_nulled or '{}'}",
                f"split de-adjusted cells : {sum(self.split_deadjusted.values())} {self.split_deadjusted or '{}'}",
                f"splits applied          : {self.splits_applied or 'none'}",
                f"splits rejected         : {self.splits_rejected or 'none'}",
            ]
        )


# --------------------------------------------------------------------------- #
# loading and validation                                                       #
# --------------------------------------------------------------------------- #
def _entries(raw: dict) -> dict:
    """The register's real entries -- documentation keys skipped, as both files declare."""
    return {k: v for k, v in raw.items() if not k.startswith(SHARADAR_REGISTER_DOC_PREFIX)}


def _require_approval(raw: dict, path: Path) -> None:
    """Raise RuntimeError unless the register carries an `_APPROVED` block with `on` and `scope`."""
    block = raw.get(SHARADAR_APPROVAL_KEY)
    if not isinstance(block, dict) or not block.get("on") or not block.get("scope"):
        raise RuntimeError(
            f"{path} carries no usable `{SHARADAR_APPROVAL_KEY}` block (needs `on` and "
            f"`scope`). It is a PROPOSAL until a human approves it, and phase 3 refuses to "
            f"run against a proposal -- see the file's own _README."
        )


def load_zero_rules(config_dir: str = DEFAULT_CONFIG_DIR) -> dict[str, str]:
    """The approved per-field zero rule (`null` or `keep`); raises if any zero-filled field has no rule."""
    path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_ZERO_RULES_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_approval(raw, path)
    rules = {name: block["rule"] for name, block in _entries(raw).items()}
    missing = sorted(SHARADAR_ZERO_FILLED_FIELDS - set(rules))
    if missing:
        raise RuntimeError(f"{path} has no rule for {len(missing)} zero-filled field(s): {missing}")
    unknown = sorted(set(rules.values()) - {"null", "keep"})
    if unknown:
        raise RuntimeError(f"{path} uses unknown rule(s) {unknown}; only `null` and `keep` exist")
    return rules


def load_corrections(config_dir: str = DEFAULT_CONFIG_DIR) -> dict[str, dict[str, dict]]:
    """The approved per-(field, ticker) correction register.

    Raises if the file is missing, unapproved, uses an action outside the closed vocabulary, or an entry lacks
    `reason` or `evidence`.
    """
    path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_CORRECTIONS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the field map refuses to run without the correction register (phase-3 §0)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_approval(raw, path)
    register = _entries(raw)
    for field, by_ticker in register.items():
        for ticker, entry in by_ticker.items():
            _check_correction(f"{path}: {field}/{ticker}", entry)
    return register


def _check_correction(where: str, entry: dict) -> None:
    """Raise unless one register entry uses a known action and states its `reason` and `evidence`."""
    action = entry.get("action")
    if action not in SHARADAR_CORRECTION_ACTIONS:
        raise RuntimeError(f"{where} has action {action!r}; the vocabulary is closed to {sorted(SHARADAR_CORRECTION_ACTIONS)}")
    for key in ("reason", "evidence"):
        if not str(entry.get(key, "")).strip():
            raise RuntimeError(f"{where} has no `{key}`. Every correction states what was measured or which filing was read.")


def _basis_for(name: str, source: str, catalogue: Catalogue) -> str:
    """A direct column's TTM basis: the KPI catalogue's kind where it declares one, else the Sharadar flow set."""
    spec = catalogue.fields.get(name)
    if spec is not None and spec.kind == "instant":
        return INSTANT
    if spec is not None and spec.kind == "duration":
        return DURATION if spec.is_additive else MEAN
    return DURATION if source in SHARADAR_FLOW_FIELDS else INSTANT


def _spec_from(name: str, entry: dict, *, catalogue, path: Path, basis: str | None = None) -> ColumnSpec:
    """One JSON entry -> a validated `ColumnSpec`."""
    kind = entry.get("kind")
    if kind not in SHARADAR_MAP_KINDS:
        raise RuntimeError(f"{path}: {name} has kind {kind!r}; the vocabulary is closed to {sorted(SHARADAR_MAP_KINDS)}")
    source = entry.get("from")
    split_basis = entry.get("split_basis")
    if split_basis is not None and split_basis not in SHARADAR_MAP_SPLIT_BASES:
        raise RuntimeError(f"{path}: {name} has split_basis {split_basis!r}; expected one of {sorted(SHARADAR_MAP_SPLIT_BASES)}")
    negate = entry.get("negate")
    if negate is not None and negate != SHARADAR_NEGATE_IF_NON_POSITIVE:
        raise RuntimeError(
            f"{path}: {name} has negate {negate!r}. The only accepted spelling is "
            f"{SHARADAR_NEGATE_IF_NON_POSITIVE!r} -- `true` flips unconditionally, and 13 of "
            f"1,346 stored rows carry a positive `capex`, so it writes a negative into a "
            f"`non_negative` column."
        )

    if kind == "direct":
        if source not in set(SHARADAR_SF1_COLUMNS):
            raise RuntimeError(f"{path}: {name} maps from {source!r}, which SF1 does not deliver ({len(SHARADAR_SF1_COLUMNS)} columns)")
        return ColumnSpec(
            name=name, kind=kind, source=source, negate=negate, split_basis=split_basis, basis=basis or _basis_for(name, source, catalogue)
        )
    if kind == "derived":
        op = entry.get("op")
        if op not in SHARADAR_MAP_OPS:
            raise RuntimeError(f"{path}: {name} has op {op!r}; the vocabulary is closed to {sorted(SHARADAR_MAP_OPS)}")
        inputs = tuple(entry.get("inputs", ()))
        formula = entry.get("formula")
        _assert_formula_matches(name, op, inputs, formula, path)
        return ColumnSpec(name=name, kind=kind, op=op, inputs=inputs, formula=formula)
    return ColumnSpec(name=name, kind=kind, basis=basis)


def _expected_formula(op: str, inputs: tuple[str, ...]) -> str | None:
    """The prose formula `op` computes over `inputs`, or None when the arity is wrong for `op`."""
    if op == "quarter":
        return f"the DISCRETE quarter's {inputs[0]}" if len(inputs) == 1 else None
    if op == "sum":
        return " + ".join(inputs)
    if op == "sum_optional":
        # the formula must show which legs are optional, or it reads like `sum`
        return " + ".join([inputs[0]] + [f"coalesce({i}, 0)" for i in inputs[1:]]) if len(inputs) >= 2 else None
    if op == "ratio":
        return " / ".join(inputs) if len(inputs) == 2 else None
    return f"{inputs[0]} / {inputs[1]} - 1" if len(inputs) == 2 else None


def _assert_formula_matches(name: str, op: str, inputs: tuple[str, ...], formula: str | None, path: Path) -> None:
    """Raise unless the prose `formula` equals what `op` + `inputs` actually compute."""
    expected = _expected_formula(op, inputs)
    if expected is None:
        raise RuntimeError(f"{path}: {name} op {op!r} has the wrong arity for inputs {inputs}")
    if formula != expected:
        raise RuntimeError(f"{path}: {name} declares formula {formula!r} but `op`/`inputs` compute {expected!r}")


def load_field_map(config_dir: str | None = DEFAULT_CONFIG_DIR) -> FieldMap:
    """The validated map, built once per (process, resolved config directory)."""
    return _field_map_at(resolve_config_dir(config_dir))


@cache
def _field_map_at(config_dir: str) -> FieldMap:
    """Load and validate the map and both registers; raise RuntimeError on any contract violation.

    Rejects an unmapped or stray `HISTORY_STATEMENT_ORDER` name, a `from` SF1 does not deliver, an extra whose
    `to` collides with a contract column or another extra, and a derived input the map does not produce.
    """
    path = Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_FIELD_MAP_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    catalogue = load_catalogue(config_dir)

    columns = {n: _spec_from(n, e, catalogue=catalogue, path=path) for n, e in raw["columns"].items()}
    added = {n: _spec_from(n, e, catalogue=catalogue, path=path) for n, e in raw["added_columns"].items()}
    # An extra is keyed by its vendor name, emitted under its camelCase `to`, and may be a negated outflow.
    extras = {
        e["to"]: ColumnSpec(name=e["to"], kind="direct", source=n, basis=e["basis"], split_basis=e.get("split_basis"), negate=e.get("negate"))
        for n, e in raw["extras"].items()
    }

    unmapped = [n for n in HISTORY_STATEMENT_ORDER if n not in columns]
    if unmapped:
        raise RuntimeError(f"{path} leaves {len(unmapped)} contract column(s) unmapped: {unmapped}")
    stray = sorted(set(columns) - set(HISTORY_STATEMENT_ORDER))
    if stray:
        raise RuntimeError(f"{path} maps {stray}, which are not in HISTORY_STATEMENT_ORDER. A column beyond the 60 belongs in `added_columns`.")
    # SF1 membership is checked on the vendor SOURCE, not the emitted name.
    for name, spec in extras.items():
        if spec.source not in set(SHARADAR_SF1_COLUMNS):
            raise RuntimeError(f"{path}: extra {name!r} reads {spec.source!r}, which is not an SF1 column")
        if name in set(HISTORY_STATEMENT_ORDER) or name in columns or name in added:
            raise RuntimeError(
                f"{path}: extra {spec.source!r} renames to {name!r}, which is "
                f"already a contract column. Two sources would write one "
                f"column and the later one would win silently."
            )
        if spec.basis not in (DURATION, INSTANT):
            raise RuntimeError(f"{path}: extra {name!r} has basis {spec.basis!r}; expected {DURATION!r} or {INSTANT!r}")
        if spec.negate is not None and spec.negate != SHARADAR_NEGATE_IF_NON_POSITIVE:
            raise RuntimeError(
                f"{path}: extra {name!r} has negate {spec.negate!r}; the only "
                f"accepted spelling is {SHARADAR_NEGATE_IF_NON_POSITIVE!r} "
                f"(see the contract-column check for why never `true`)"
            )
    collisions = sorted(n for n in extras if sum(1 for e in raw["extras"].values() if e["to"] == n) > 1)
    if collisions:
        raise RuntimeError(f"{path}: {collisions} is the `to` of more than one extra; the dict would silently keep only the last.")

    field_map = FieldMap(
        columns=columns,
        added=added,
        extras=extras,
        excluded=frozenset(raw["excluded"]),
        zero_rules=load_zero_rules(config_dir),
        corrections=load_corrections(config_dir),
    )
    _assert_derived_inputs_resolve(field_map, path)
    return field_map


def _assert_derived_inputs_resolve(field_map: FieldMap, path: Path) -> None:
    """Raise unless every derived input is a produced, non-derived column (formulas run in one pass)."""
    outputs, derived = field_map.outputs, field_map.derived
    for name, spec in derived.items():
        for source in spec.inputs:
            if source not in outputs:
                raise RuntimeError(f"{path}: {name} reads {source!r}, which the map does not produce")
            if source in derived:
                raise RuntimeError(
                    f"{path}: {name} reads the derived column {source!r}. "
                    f"Formulas run in ONE pass, so a derived input would be "
                    f"read before it is computed."
                )


# --------------------------------------------------------------------------- #
# the vendor-frame cleaning stages                                             #
# --------------------------------------------------------------------------- #
def apply_zero_rules(frame: pd.DataFrame, rules: dict[str, str], *, report: TranslationReport | None = None) -> pd.DataFrame:
    """Replace `0.0` with NaN for every field ruled `"null"`, on the vendor frame before any sum.

    Raises if a zero-filled field present in the frame has no rule.
    """
    ungoverned = sorted((SHARADAR_ZERO_FILLED_FIELDS & set(frame.columns)) - set(rules))
    if ungoverned:
        raise RuntimeError(f"no zero rule for {ungoverned}; every zero-filled field in the frame needs one")
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


def apply_corrections(frame: pd.DataFrame, corrections: dict[str, dict[str, dict]], *, report: TranslationReport | None = None) -> pd.DataFrame:
    """Null the cells the (field, ticker) register targets, on the vendor frame (the stored table stays as sent)."""
    out = frame.copy()
    for field, by_ticker in corrections.items():
        if field not in out.columns:
            continue
        for ticker, entry in by_ticker.items():
            rows = out["ticker"] == ticker
            action = entry["action"]
            hit = _correction_hit(out[field], rows, action)
            count = int(hit.sum())
            if not count:
                continue
            out.loc[hit, field] = np.nan
            if report is not None:
                key = f"{field}/{ticker}:{action}"
                report.corrected[key] = report.corrected.get(key, 0) + count
    return out


def _correction_hit(column: pd.Series, rows: pd.Series, action: str) -> pd.Series:
    """The cells of `column` inside `rows` that a correction `action` nulls."""
    if action == "null":
        return rows & column.notna()
    if action == "null_if_positive":
        return rows & (column > 0)
    return rows & (column < 0)


# --------------------------------------------------------------------------- #
# the split de-adjustment                                                      #
# --------------------------------------------------------------------------- #
#: Calendar days within which a Sharadar and a yfinance split are the same event.
SPLIT_MATCH_DAYS = 7
#: Max distance from a small-integer fraction to read as a split; admits 5-dp rounding, rejects spinoff factors.
SPLIT_INTEGER_TOL = 1e-4
#: Relative ratio gap above which two vendors on one date describe different events (price vs share factor).
SPLIT_RATIO_CONFLICT_TOL = 0.01
#: Largest denominator a genuine split ratio may have (2:1, 3:2, 1:20, 21/20 stock dividend, ...).
SPLIT_MAX_DENOMINATOR = 20


def _is_split_shaped(ratio: float) -> bool:
    """Whether `ratio` is a fraction of small integers (a split or stock dividend), not a spinoff/merger price factor.

    Applied to both vendors: yfinance's `Stock Splits` also carries spinoff factors.
    """
    if not ratio or ratio <= 0 or not np.isfinite(ratio):
        return False
    frac = Fraction(float(ratio)).limit_denominator(SPLIT_MAX_DENOMINATOR)
    return abs(float(frac) - ratio) < SPLIT_INTEGER_TOL


def _is_simple_split(ratio: float) -> bool:
    """Whether `ratio` is split-shaped AND `n:1` or `1:n`; used only to decide what to flag for review."""
    if not _is_split_shaped(ratio):
        return False
    frac = Fraction(float(ratio)).limit_denominator(SPLIT_MAX_DENOMINATOR)
    return frac.denominator == 1 or frac.numerator == 1


def split_events(actions: pd.DataFrame | None, yf_splits: pd.DataFrame | None = None, *, report: TranslationReport | None = None) -> pd.DataFrame:
    """The genuine share splits from both vendors, as `(ticker, date, value)`.

    `sharadar_actions` is incomplete per ticker, so it is cross-validated against yfinance (`prices_splits`) by
    `union_split_sources`. A Sharadar `split` co-dated with a `spinoff` is still a split; the spinoff's price
    factor is separated by the ratio-conflict rule, not by dropping the row. `yf_splits=None` uses Sharadar alone.
    """
    # the empty frame needs a datetime `date`, or the union's date subtraction raises TypeError
    empty = pd.DataFrame(
        {
            "ticker": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="object"),
        }
    )
    codated: set[tuple] = set()
    if actions is None or actions.empty:
        sharadar = empty
    else:
        frame = actions.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        candidates = frame[frame["action"] == SHARADAR_ACTION_SPLIT]
        kept = [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "value": float(row["value"]),
                "label": f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']}",
            }
            for _, row in candidates.iterrows()
        ]
        sharadar = pd.DataFrame(kept, columns=["ticker", "date", "value", "label"]) if kept else empty
        # `spinoff` rows veto nothing; they only name co-dated splits for review
        codated = set(map(tuple, frame.loc[frame["action"] == SHARADAR_ACTION_SPINOFF, ["ticker", "date"]].to_numpy()))

    yf = pd.DataFrame(columns=["ticker", "date", "value"])
    if yf_splits is not None and not yf_splits.empty:
        yf = yf_splits.rename(columns={"ratio": "value"})[["ticker", "date", "value"]].copy()
        yf["date"] = pd.to_datetime(yf["date"], errors="coerce")
        yf = yf.dropna(subset=["date", "value"])
        yf = yf[yf["value"] != 0.0]

    out = union_split_sources(sharadar, yf, report=report)
    _log_codated(out, codated)
    return out


def _log_codated(out: pd.DataFrame, codated: set[tuple]) -> None:
    """Warn on kept splits co-dated with a spinoff, and separately on those that are not `n:1` / `1:n`."""
    if out.empty or not codated:
        return
    hit = [
        f"{r.ticker} {pd.Timestamp(cast(Any, r.date)).date()} x{r.value}"
        for r in out.itertuples()
        if (r.ticker, pd.Timestamp(cast(Any, r.date))) in codated
    ]
    if not hit:
        return
    log.warning(
        "%d split(s) co-dated with a spinoff are now KEPT (the veto that dropped them was wrong -- see `split_events`): %s",
        len(hit),
        ", ".join(sorted(hit)),
    )
    odd = [h for h in hit if not _is_simple_split(float(h.rsplit("x", 1)[1]))]
    if odd:
        log.warning(
            "of those, %d is/are split-shaped but NOT n:1 or 1:n -- the shape test "
            "is the only filter left, so confirm each against the filer's own "
            "disclosure: %s",
            len(odd),
            ", ".join(odd),
        )


def _resolve_ratio_conflict(yf_row: pd.Series, near: pd.DataFrame) -> tuple[float, str | None]:
    """`(ratio to keep, warning line or None)` for one corroborated event.

    yfinance's ratio unless the vendors differ by more than `SPLIT_RATIO_CONFLICT_TOL` and only Sharadar's is
    split-shaped; any material disagreement yields a warning line.
    """
    yf_value = float(yf_row["value"])
    label = f"{yf_row['ticker']} {pd.Timestamp(yf_row['date']).date()}"
    for _, other in near.iterrows():
        sh_value = float(other["value"])
        if not np.isfinite(sh_value) or sh_value <= 0:
            continue
        if abs(yf_value / sh_value - 1.0) <= SPLIT_RATIO_CONFLICT_TOL:
            continue
        yf_ok, sh_ok = _is_split_shaped(yf_value), _is_split_shaped(sh_value)
        if sh_ok and not yf_ok:
            return sh_value, (
                f"{label}: yfinance x{yf_value} vs sharadar x{sh_value} "
                f"-> KEPT x{sh_value} (only the sharadar ratio is "
                f"split-shaped; the yfinance one is the spinoff's PRICE "
                f"factor)"
            )
        return yf_value, (
            f"{label}: yfinance x{yf_value} vs sharadar x{sh_value} "
            f"-> KEPT x{yf_value} (yfinance, "
            f"{'both' if yf_ok and sh_ok else 'neither'} split-shaped -- "
            f"no evidence to prefer the other)"
        )
    return yf_value, None


def _reject_yfinance_only(row: pd.Series, report: TranslationReport | None) -> None:
    """Record an uncorroborated, non-split-shaped yfinance event as rejected on `report`, if any."""
    if report is not None:
        report.splits_rejected.append(f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']} (yfinance-only, not split-shaped)")


def union_split_sources(sharadar: pd.DataFrame, yf: pd.DataFrame, *, report: TranslationReport | None = None) -> pd.DataFrame:
    """Merge two cleaned split lists into one, sorted by `(ticker, date)`.

    In both (within `SPLIT_MATCH_DAYS`): kept on the yfinance DATE (so it steps with `close_split`), ratio per
    `_resolve_ratio_conflict`. One vendor only: kept iff split-shaped (Sharadar-only kept with a warning).
    """
    matched_sharadar: set[int] = set()
    events: list[dict] = []
    conflicts: list[str] = []
    kept_both = kept_yf = kept_sharadar = 0

    for _, row in yf.iterrows():
        near = sharadar.index[(sharadar["ticker"] == row["ticker"]) & ((sharadar["date"] - row["date"]).abs() <= pd.Timedelta(days=SPLIT_MATCH_DAYS))]
        matched_sharadar.update(near)
        value = float(row["value"])
        # corroboration overrides shape; uncorroborated, the shape test decides
        if len(near):
            kept_both += 1
            value, note = _resolve_ratio_conflict(row, sharadar.loc[near])
            if note:
                conflicts.append(note)
        elif _is_split_shaped(value):
            kept_yf += 1
        else:
            _reject_yfinance_only(row, report)
            continue
        events.append({"ticker": row["ticker"], "date": row["date"], "value": value})

    uncorroborated = []
    for idx, row in sharadar.iterrows():
        if idx in matched_sharadar:
            continue
        label = row["label"] if "label" in row else (f"{row['ticker']} {pd.Timestamp(row['date']).date()} x{row['value']}")
        if _is_split_shaped(float(row["value"])):
            events.append({"ticker": row["ticker"], "date": row["date"], "value": float(row["value"])})
            uncorroborated.append(label)
            kept_sharadar += 1
        elif report is not None:
            report.splits_rejected.append(f"{label} (uncorroborated, not split-shaped)")

    if uncorroborated:
        log.warning(
            "%d split event(s) in sharadar_actions but NOT in yfinance, kept because the ratio is split-shaped -- review: %s",
            len(uncorroborated),
            ", ".join(uncorroborated),
        )
    if conflicts:
        log.warning("%d date(s) where the two vendors report DIFFERENT ratios -- resolved: %s", len(conflicts), "; ".join(conflicts))
        if report is not None:
            report.splits_rejected.extend(conflicts)

    out = (
        pd.DataFrame(events, columns=["ticker", "date", "value"])
        .drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    log.info("split events: %d corroborated, %d yfinance-only, %d sharadar-only -> %d total", kept_both, kept_yf, kept_sharadar, len(out))
    if report is not None:
        report.splits_applied.extend(f"{r.ticker} {pd.Timestamp(cast(Any, r.date)).date()} x{r.value}" for r in out.itertuples())
    return out


def forward_split_factor(tickers: pd.Series, dates: pd.Series, splits: pd.DataFrame) -> pd.Series:
    """Product of every split dated STRICTLY AFTER each row's filing date (1.0 if none).

    This is the factor Sharadar applied retroactively: divide a count (multiply a per-share figure) by it to
    recover the as-filed basis.
    """
    factor = pd.Series(1.0, index=dates.index)
    if splits.empty:
        return factor
    stamps = pd.to_datetime(dates, errors="coerce")
    for _, split in splits.iterrows():
        hit = (tickers == split["ticker"]) & (stamps < split["date"])
        factor.loc[hit] = factor.loc[hit] * float(split["value"])
    return factor


def deadjust_splits(
    frame: pd.DataFrame,
    field_map: FieldMap,
    actions: pd.DataFrame | None,
    yf_splits: pd.DataFrame | None = None,
    *,
    report: TranslationReport | None = None,
) -> pd.DataFrame:
    """Undo Sharadar's retroactive split adjustment on every column with a declared `split_basis`.

    SF1 restates the whole share block (`sharesbas`, `shareswa`, `shareswadil`, `eps`, `epsdil`, `dps`) on
    today's basis, so it is not point-in-time. `count` columns are divided and per-share columns multiplied by
    `forward_split_factor` at the row's filing date. Must run on the TTM frame (every quarter in a window shares
    one vendor basis), never on discrete quarters. With no split events, warns and returns the frame unchanged.
    """
    targets = {n: s for n, s in field_map.outputs.items() if s.split_basis}
    if not targets:
        return frame
    splits = split_events(actions, yf_splits, report=report)
    if splits.empty:
        log.warning(
            "no genuine split events available -- the share block stays on Sharadar's retroactively adjusted basis, which is NOT point-in-time"
        )
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
        out.loc[hit, name] = out.loc[hit, name] / factor[hit] if spec.split_basis == "count" else out.loc[hit, name] * factor[hit]
        if report is not None:
            report.split_deadjusted[name] = int(hit.sum())
    return out


# --------------------------------------------------------------------------- #
# the rename                                                                   #
# --------------------------------------------------------------------------- #
def translate(frame: pd.DataFrame, field_map: FieldMap, *, report: TranslationReport | None = None) -> pd.DataFrame:
    """A vendor ARQ frame -> the repo-named ARQ frame, still on the discrete-quarter grain.

    Zero rules, corrections, then direct renames with their sign guards. Derived formulas and split
    de-adjustment run later on the TTM frame (`build_ttm`). `sec` and `null` columns are emitted all-NaN; the
    merge fills the SEC-owned ones. Raises if a mapped vendor column is missing from the frame.
    """
    report = report if report is not None else TranslationReport()
    report.rows_in = len(frame)

    missing = [cast(str, s.source) for s in field_map.direct.values() if s.source not in frame.columns]
    if missing:
        raise RuntimeError(
            f"the vendor frame is missing {len(missing)} mapped column(s): "
            f"{sorted(set(missing))}. `fields=` silently drops an unavailable "
            f"field, so this is a projection or a typo, never an empty column."
        )

    cleaned = apply_zero_rules(frame, field_map.zero_rules, report=report)
    cleaned = apply_corrections(cleaned, field_map.corrections, report=report)

    # concatenated once: per-column inserts fragment the frame and trigger pandas warnings
    columns: dict[str, pd.Series] = {}
    for name, spec in field_map.direct.items():
        values = cleaned[spec.source].astype("float64")
        if spec.negate == SHARADAR_NEGATE_IF_NON_POSITIVE:
            values = _negate_if_non_positive(values, name, report)
        if name in SIGN_ENFORCED:
            values = _null_if_negative(values, name, report)
        columns[name] = values
    for name, spec in field_map.outputs.items():
        if spec.kind in ("sec", "null"):
            columns[name] = pd.Series(np.nan, index=cleaned.index)

    keys = cleaned[[c for c in KEY_COLUMNS if c in cleaned.columns]]
    return pd.concat([keys, pd.DataFrame(columns, index=cleaned.index)], axis=1)


#: Direct columns whose declared `non_negative` sign is enforced at map time by nulling (not flipping) negatives.
SIGN_ENFORCED: frozenset[str] = frozenset({"interestExpense"})


def _null_if_negative(values: pd.Series, name: str, report: TranslationReport) -> pd.Series:
    """NULL (not flip) negative cells in a `non_negative` column -- they are net-basis figures; counted and warned."""
    violations = values < 0
    count = int(violations.sum())
    if count:
        values = values.copy()
        values[violations] = np.nan
        report.sign_nulled[name] = report.sign_nulled.get(name, 0) + count
        log.warning(
            "%s: %d cell(s) are negative in a column declared non_negative -- NULLed (a net-basis or Q4-by-subtraction figure, not a gross one)",
            name,
            count,
        )
    return values


def _negate_if_non_positive(values: pd.Series, name: str, report: TranslationReport) -> pd.Series:
    """Negate an outflow stored negative; NULL (and count) the positive cells instead of flipping them negative."""
    violations = values > 0
    count = int(violations.sum())
    out = -values
    if count:
        out[violations] = np.nan
        report.negation_nulled[name] = report.negation_nulled.get(name, 0) + count
        log.warning("%s: %d row(s) violate Sharadar's sign convention and were NULLed rather than flipped into a negative", name, count)
    return out


# --------------------------------------------------------------------------- #
# the derived formulas -- LAST, on the TTM frame                               #
# --------------------------------------------------------------------------- #
def apply_derived(frame: pd.DataFrame, field_map: FieldMap, only: set[str] | None = None) -> pd.DataFrame:
    """Evaluate every `derived` column on the TTM frame (a ratio of TTM levels, never the TTM of a ratio).

    `op == "quarter"` is skipped (`build_ttm` owns it). A zero denominator yields NaN, not inf. `only` restricts
    the pass to named columns, so the post-merge recompute leaves every other derived column as built.
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
        elif spec.op == "sum_optional":
            # first leg required (its NULLs re-imposed), the rest coalesced to 0
            widened = sum((p.fillna(0.0) for p in parts[1:]), start=parts[0].fillna(0.0))
            computed[name] = widened.where(parts[0].notna())
        else:
            numerator, denominator = parts[0], parts[1].replace(0.0, np.nan)
            values = numerator / denominator
            computed[name] = values - 1.0 if spec.op == "ratio_minus_one" else values
    if not computed:
        return frame.copy()
    # every derived name either replaces an existing column or is new; assign in one pass
    return frame.assign(**computed)
