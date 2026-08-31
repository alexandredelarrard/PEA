"""
merge_history.py  (src/data_extract/utils/fundamentals_sharadar/merge_history.py)
------------------------------------------------------------------------------------
The Sharadar TTM frame + the SEC-owned block -> `fundamentals_history`, the merged table
every consumer reads.

FIELD-BLOCK PRECEDENCE (D14), and it is the whole design: Sharadar owns a declared set of
columns for ALL history, `fundamentals_history_sec` owns the 15 named in `field_map`'s
`sec` kind, and NO COLUMN EVER SWITCHES SOURCE MID-SERIES. There is no per-row fallback and
no `source` column, because a per-row source would be a lie about a per-column decision.

The one exception is the OVERRIDE REGISTER, and it is an exception by construction: it moves
a whole `(ticker, field)` series to SEC, is machine-proposed and human-approved, and the merge
only READS it. Nothing here decides a source at runtime.

## Every column NAME carries its own provenance

  * a SEC-sourced column ends in **`_sec`** -- `goodwill_sec`, `minorityInterest_sec`,
    `regime_sec`, all 15 of them.
  * every Sharadar column is repo camelCase, the 25 EXTRAS included. They are keyed by their
    vendor spelling in `sharadar_field_map.json` and emitted under a repo name
    (`cashneq` -> `cashAndEquivalents`, `ncfx` -> `exchangeRateEffect`). The vendor spelling
    survives only in `fundamentals_sharadar`, which is the table it describes.

This is not decoration, and it is what replaces the `source` column D15 refuses. The two
producers have DIFFERENT COVERAGE -- Sharadar spans every entitled ticker, the SEC block only
the ones both sources have -- so a bare `goodwill` sitting beside a bare `totalRevenue` says
nothing about which producer left a NULL there. With the suffix the question is answered by
reading the column name, and precedence stays per-COLUMN rather than becoming a per-ROW claim
that would be a lie.

⚠ The suffix goes on LAST, after `rederive`. `stockholdersEquityInclNci` reads
`minorityInterest` under the name the field map declares, so renaming any earlier breaks the
one formula that spans both sources.

## The join is BACKWARD, and that is the no-leakage property

`as_of` is Sharadar's `date` -- the SEC filing date on the AR dimensions. Measured on 14
tickers x 5 years it matched `fundamentals_history_sec.as_of` on 279 of 280 (99.64%), the one
miss being a GS 10-K/A that Sharadar has no row for at all.

99.64% is not 100%, so an EXACT join would silently drop the entire SEC block on any date the
two disagree by a day. `merge_asof(direction="backward")` instead gives the latest SEC
snapshot KNOWABLE AT the Sharadar publication date, which is the correct point-in-time
semantics and the same primitive `build_history.carry_latest_known` already uses for
`fundamentals_employees`. **Never forward** -- a forward join would put a value into a row
dated before it was filed, which is the one bug this table exists to not have.

`SHARADAR_SEC_ASOF_TOLERANCE_DAYS` caps the carry. Without it, a ticker the SEC producer
stopped covering keeps its last snapshot forever and a stale number reads as a current one.

## Two traps this module exists to not fall into

  * ⚠ `regime` is TEXT. It is one of the 15 SEC-owned columns and the ONLY non-float value in
    the table. A "cast everything numeric-looking to float64" sweep turns every regime label
    into NaN silently, so the cast excludes it BY NAME.
  * ⚠ The all-float64 cast itself is not cosmetic. `sql/schema.sql` runs only on an EMPTY
    Postgres volume, so on a live one `store.save` creates the table from the FIRST frame via
    `ensure_table`'s dtype inference -- an all-None `object` column becomes **TEXT** and every
    later real number is stored as a string. That has already happened once on
    `fundamentals_history_sec` (APA's values came back as `'1997000000.0'`).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT, SHARADAR_COLLAPSE_KEY,
    SHARADAR_COLLAPSE_ORDER, SHARADAR_CONFIG_SUBDIR, SHARADAR_OVERRIDE_APPROVED_KEY,
    SHARADAR_OVERRIDE_SOURCE_SEC, SHARADAR_REGISTER_DOC_PREFIX,
    SHARADAR_SEC_ASOF_TOLERANCE_DAYS, SHARADAR_SOURCE_OVERRIDES_FILENAME,
)
from src.data_extract.utils.fundamentals.kpi_catalogue import DEFAULT_CONFIG_DIR
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ, build_ttm
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    FieldMap, TranslationReport, apply_derived, load_field_map, translate)
# Top-level, not deferred. `src/data_store/schema.py` imports only `data_store.errors`, so
# there is no package cycle to dodge here -- a local import would only hide the dependency.
from src.context import Context
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Tables

log = logging.getLogger(__name__)

#: The merged table's key columns. `as_of` is Sharadar's `date` (a FILING date) and
#: `fiscal_end` its `reportperiod` (the period end) -- the same two meanings the SEC table
#: gives those names, which is what makes the two tables comparable at all.
MERGE_KEYS: tuple[str, ...] = ("ticker", "as_of", "fiscal_end")

#: Vendor -> merged names for the two keys that are not already repo-named.
_KEY_FROM_VENDOR: dict[str, str] = {"date": "as_of", "reportperiod": "fiscal_end"}

#: The suffix every SEC-SOURCED column wears in the merged table. Not decoration: the table
#: mixes two producers with DIFFERENT COVERAGE -- Sharadar's columns span all 30 entitled
#: tickers, the SEC block only the 14 both sources have -- and a bare `goodwill` beside a bare
#: `totalRevenue` says nothing about which of the two a NULL belongs to. With the suffix the
#: column NAME carries its own provenance, so "why is this empty for 16 tickers?" is answered
#: by reading it rather than by opening this file.
SEC_SUFFIX = "_sec"


def sec_column(name: str) -> str:
    return f"{name}{SEC_SUFFIX}"


#: `regime_sec` apart, every merged column is a float. Listed by NAME (see the module
#: docstring); a "looks numeric" heuristic would catch it and NaN every regime label.
NON_VALUE_COLUMNS: frozenset[str] = frozenset({*MERGE_KEYS, sec_column("regime")})

#: Where the SEC block's `employees` actually lives. It is SEC-owned (D18) but it is NOT a
#: column of `fundamentals_history_sec` -- headcount was moved out to its own table because
#: the source is 10-K PROSE and one failed regex must not fail a 91-column snapshot. So the
#: 15 SEC-owned names come from TWO tables, 14 + 1, not from one.
EMPLOYEES_COLUMN = "employees"

#: The prefix under which an OVERRIDE's SEC value rides along the join, so a SEC
#: `totalRevenue` can sit beside the Sharadar one without a `_x`/`_y` collision. No prefixed
#: name is in the 91-column contract, so the final projection drops every one of them by
#: construction -- there is no separate cleanup step that could be forgotten.
_SEC_PREFIX = "__sec__"

#: The SEC row's own `as_of`, carried through the join so the no-leakage property is
#: MEASURABLE rather than merely argued. Dropped before the write.
SEC_AS_OF = "__sec_as_of__"

# --------------------------------------------------------------------------- #
# the override register (D22)                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Overrides:
    """The approved `(ticker, field) -> sec` decisions, plus what is still awaiting one.

    `pending` is not a diagnostic afterthought: an unapproved proposal must never silently
    change data, and the only way to know it did not is to count it and say so.
    """

    approved: dict[tuple[str, str], dict]
    pending: dict[tuple[str, str], dict]

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted({field for _, field in self.approved}))

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted({ticker for ticker, _ in self.approved}))


def overrides_path(config_dir: str = DEFAULT_CONFIG_DIR) -> Path:
    return Path(config_dir) / SHARADAR_CONFIG_SUBDIR / SHARADAR_SOURCE_OVERRIDES_FILENAME


def load_overrides(config_dir: str = DEFAULT_CONFIG_DIR) -> Overrides:
    """Read the register. A missing file means NO overrides, which is the normal state.

    ⚠ Approval here is PER ENTRY (`approved`), not per file like `sharadar_zero_rules.json`
    and `sharadar_corrections.json`. Deliberate, and the difference is real: those two are
    all-or-nothing inputs to a transform that cannot run without them, so an unsigned file
    must be fatal. This one is a decision LIST, and a freshly proposed entry must be INERT --
    making the whole file fatal would mean `--propose` breaks the pipeline until a human
    happens to be available, which is an incentive to approve without reading.
    """
    path = overrides_path(config_dir)
    if not path.exists():
        return Overrides(approved={}, pending={})
    raw = json.loads(path.read_text(encoding="utf-8"))
    approved: dict[tuple[str, str], dict] = {}
    pending: dict[tuple[str, str], dict] = {}
    for ticker, by_field in raw.items():
        if ticker.startswith(SHARADAR_REGISTER_DOC_PREFIX):
            continue
        for field, entry in by_field.items():
            where = f"{path}: {ticker}/{field}"
            source = entry.get("source")
            if source != SHARADAR_OVERRIDE_SOURCE_SEC:
                raise RuntimeError(
                    f"{where} names source {source!r}. The ONLY legal direction is "
                    f"{SHARADAR_OVERRIDE_SOURCE_SEC!r}: moving a column the other way is a "
                    f"field-BLOCK change (D14) and belongs in sharadar_field_map.json.")
            if not str(entry.get("reason", "")).strip():
                raise RuntimeError(f"{where} has no `reason`. An override that cannot be "
                                   f"re-checked when the roster widens is not a decision.")
            bucket = approved if entry.get(SHARADAR_OVERRIDE_APPROVED_KEY) else pending
            bucket[(ticker, field)] = entry
    return Overrides(approved=approved, pending=pending)


def write_overrides(entries: dict[str, dict[str, dict]], readme: list[str],
                    config_dir: str = DEFAULT_CONFIG_DIR) -> Path:
    """Emit the register with a STABLE hand-readable shape, one line per entry.

    Not `json.dumps(indent=2)` over the whole file: a round-trip through the default emitter
    reformats every line it touches, so a two-entry proposal shows up as a whole-file diff and
    the review that this register exists for becomes impossible. One line per `(ticker,
    field)` also makes a re-propose that changed nothing produce a BYTE-IDENTICAL file, which
    is the property that lets `--propose` be safe to re-run.
    """
    path = overrides_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['{', '  "_README": [']
    lines += [f'    {json.dumps(line, ensure_ascii=False)},' for line in readme[:-1]]
    lines += [f'    {json.dumps(readme[-1], ensure_ascii=False)}', '  ],']
    for i, (ticker, by_field) in enumerate(sorted(entries.items())):
        lines.append('')
        lines.append(f'  {json.dumps(ticker)}: {{')
        items = sorted(by_field.items())
        for j, (field, entry) in enumerate(items):
            comma = "," if j < len(items) - 1 else ""
            lines.append(f'    {json.dumps(field)}: '
                         f'{json.dumps(entry, ensure_ascii=False)}{comma}')
        lines.append('  }' + ("," if i < len(entries) - 1 else ""))
    lines.append('}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the column contract                                                          #
# --------------------------------------------------------------------------- #
def merged_columns(field_map: FieldMap) -> tuple[str, ...]:
    """The 91 columns, asserted against the REGISTRY's own declaration.

    Two independent statements of the contract that must agree: `schema.py`'s `read_columns`
    (what a consumer projecting this table gets) and the field map (what the transform can
    produce). Asserting them against each other is the only thing that makes either one true
    -- and a silent column loss here is invisible downstream, because
    `pit.fundamentals_to_daily` returns an EMPTY FRAME for a column it cannot find rather
    than raising.
    """

    owned = set(field_map.sec_owned)
    columns = (*MERGE_KEYS,
               *(sec_column(n) if n in owned else n for n in field_map.outputs))
    declared = tuple(Tables.fundamentals_history.read_columns)
    if columns != declared:
        extra = [c for c in columns if c not in set(declared)]
        gone = [c for c in declared if c not in set(columns)]
        raise RuntimeError(
            f"the merged column contract disagrees with the registry: {len(columns)} built "
            f"vs {len(declared)} declared; only in the build {extra}; only in "
            f"schema.py {gone}. Fix BOTH -- they are the same decision written twice on "
            f"purpose.")
    return columns


# --------------------------------------------------------------------------- #
# the steps                                                                    #
# --------------------------------------------------------------------------- #
def collapse_same_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per `(ticker, as_of)`, keeping the GREATEST `fiscal_end`.

    Sharadar documents its AR dimensions as possibly carrying several observations in one
    quarter, and ships NO form column -- so `build_history`'s `FORM_PRECEDENCE` has no
    analogue here and a same-day 10-K + 10-Q cannot be resolved by form. The vendor's own rule
    is the most recent period published that day, which is also the one a snapshot grain
    wants: the later period's numbers supersede the earlier one's.

    Returns the collapsed frame AND the rows it dropped, because a collapse that is never
    logged is a row count nobody can explain later.
    """
    ordered = frame.sort_values([*SHARADAR_COLLAPSE_KEY, SHARADAR_COLLAPSE_ORDER])
    duplicated = ordered.duplicated(subset=list(SHARADAR_COLLAPSE_KEY), keep="last")
    return ordered[~duplicated].reset_index(drop=True), ordered[duplicated].copy()


def _asof_join(left: pd.DataFrame, right: pd.DataFrame, *, tolerance_days: int,
               also_to_datetime: tuple[str, ...] = ()) -> pd.DataFrame:
    """Backward as-of join on `as_of`, within a ticker, capped at `tolerance_days`.

    Both sides go to nanoseconds and are globally sorted by `on` first: `merge_asof` REFUSES
    two datetime resolutions rather than coercing them, and a Postgres DATE column round-trips
    as `datetime.date` -- so this normalisation is the difference between working and raising.
    """
    left = left.copy()
    right = right.copy()
    left["as_of"] = pd.to_datetime(left["as_of"]).astype("datetime64[ns]")
    for column in ("as_of", *also_to_datetime):
        right[column] = pd.to_datetime(right[column]).astype("datetime64[ns]")
    right = right.dropna(subset=["as_of"])
    return pd.merge_asof(
        left.sort_values("as_of"), right.sort_values("as_of"),
        on="as_of", by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days)).reset_index(drop=True)


def join_sec_block(sharadar: pd.DataFrame, sec: pd.DataFrame, *,
                   tolerance_days: int = SHARADAR_SEC_ASOF_TOLERANCE_DAYS) -> pd.DataFrame:
    """Attach the SEC-owned columns AS OF each Sharadar publication date, BACKWARD only.

    See the module docstring for why this is not an exact join and never a forward one. The
    SEC row's own `as_of` rides along as `SEC_AS_OF` so the no-leakage property can be
    asserted on the built frame rather than argued about in a comment.
    """
    if sec.empty:
        out = sharadar.copy()
        out[SEC_AS_OF] = pd.NaT
        return out
    right = sec.rename(columns={"as_of": SEC_AS_OF}).copy()
    right["as_of"] = right[SEC_AS_OF]
    return _asof_join(sharadar, right, tolerance_days=tolerance_days,
                      also_to_datetime=(SEC_AS_OF,))


def attach_employees(frame: pd.DataFrame, employees: pd.DataFrame | None, *,
                     tolerance_days: int = SHARADAR_SEC_ASOF_TOLERANCE_DAYS) -> pd.DataFrame:
    """Headcount, forward-filled onto the filing grain.

    Annual 10-K PROSE: it was never on the filing cadence, so a value stated once in the 10-K
    must reach the following three quarters. Same backward as-of alignment as the SEC block,
    for the same no-leakage reason, and THE SAME CAP ON THE CARRY -- 370 days admits the three
    quarters that follow a 10-K and refuses a headcount stale by more than a year, which an
    uncapped join would carry forward forever for a ticker the SEC producer has dropped.
    """
    if employees is None or employees.empty:
        out = frame.copy()
        out[EMPLOYEES_COLUMN] = np.nan
        return out
    right = employees[["ticker", "as_of", EMPLOYEES_COLUMN]]
    return _asof_join(frame, right, tolerance_days=tolerance_days)


def apply_overrides(frame: pd.DataFrame, overrides: Overrides) -> tuple[pd.DataFrame, set[str]]:
    """Replace a registered `(ticker, field)` series with the SEC one, and say what it cost.

    The ONLY place a Sharadar-owned column takes a SEC value, and it happens by explicit
    registered decision -- never by a runtime heuristic on the data in front of it.

    ⚠ An override moves a field to a source with the SEC roster's coverage. For a ticker
    OUTSIDE that roster the override yields NULL, not a fallback to Sharadar -- that is the
    point of field-block precedence, and it is logged per entry rather than discovered as a
    hole in a feature panel months later.
    """
    out = frame.copy()
    changed: set[str] = set()
    for (ticker, field), entry in sorted(overrides.approved.items()):
        column = f"{_SEC_PREFIX}{field}"
        rows = out["ticker"] == ticker
        if not rows.any():
            log.info("override %s/%s: ticker not in this build, nothing to do", ticker, field)
            continue
        if column not in out.columns:
            raise RuntimeError(
                f"override {ticker}/{field}: the SEC block was not loaded for {field!r}. "
                f"The SEC projection is built FROM the register, so this means the two "
                f"disagree -- never silently write a NULL over a real Sharadar value.")
        out.loc[rows, field] = out.loc[rows, column]
        changed.add(field)
        covered = int(out.loc[rows, field].notna().sum())
        log.warning("override %s/%s -> sec (approved %s): %d of %d row(s) carry a value; "
                    "the rest are NULL, NOT a fallback to Sharadar (D14). %s",
                    ticker, field, entry.get(SHARADAR_OVERRIDE_APPROVED_KEY),
                    covered, int(rows.sum()), entry.get("reason", ""))
    if overrides.pending:
        # the COUNT is the actionable part; enumerating 29 proposals on every merge run buries
        # the rest of the log, so the list itself drops to DEBUG
        log.warning("%d override proposal(s) awaiting a decision and IGNORED",
                    len(overrides.pending))
        log.debug("pending overrides: %s", sorted(f"{t}/{f}" for t, f in overrides.pending))
    return out, changed


def rederive(frame: pd.DataFrame, field_map: FieldMap, changed: set[str]) -> pd.DataFrame:
    """Recompute only the derived columns whose inputs the merge actually changed.

    `stockholdersEquityInclNci` is the standing case: its NCI leg is SEC-owned, so it is
    0/598 out of phase 3 by construction and only becomes computable HERE. An override adds
    the rest -- move `totalRevenue` to SEC and every margin that reads it is stale.

    Targeted rather than a blanket re-run of `apply_derived`: a blanket pass would also
    recompute a derived column somebody deliberately overrode, quietly undoing the decision.
    """
    targets = {name for name, spec in field_map.derived.items()
               if spec.op != "quarter" and set(spec.inputs) & changed} - changed
    if not targets:
        return frame
    log.info("re-deriving %d column(s) whose inputs the merge changed: %s",
             len(targets), sorted(targets))
    return apply_derived(frame, field_map, only=targets)


# --------------------------------------------------------------------------- #
# the build                                                                    #
# --------------------------------------------------------------------------- #
def build_frame(sharadar_arq: pd.DataFrame, sec: pd.DataFrame, employees: pd.DataFrame | None,
                actions: pd.DataFrame | None, field_map: FieldMap, overrides: Overrides, *,
                report: TranslationReport | None = None) -> pd.DataFrame:
    """The whole transform, with NO I/O, so every step is testable without a database.

    Step order, and each step's rule, is the phase-4 plan's: translate -> TTM -> collapse ->
    join the SEC block backward -> employees -> overrides -> re-derive -> contract -> cast.
    """
    columns = merged_columns(field_map)
    # An override on a column the SEC ALREADY owns is not a decision, it is a contradiction --
    # and a silently destructive one: the SEC projection would rename that column out from
    # under the join and the contract assertion would report it as "missing" rather than as
    # what it is. Refuse it by name here, where the reason can be stated.
    contradictory = sorted(set(overrides.fields) & set(field_map.sec_owned))
    if contradictory:
        raise RuntimeError(
            f"the override register moves {contradictory} to `sec`, but the field map already "
            f"declares them SEC-owned (D18). An override moves a SHARADAR-owned column; "
            f"changing which block a column belongs to is a field-map edit, not an override.")
    translated = translate(sharadar_arq, field_map, report=report)
    
    # `actions` goes to `build_ttm`, not to `translate`: the split de-adjustment runs AFTER
    # the four-quarter aggregation, or a window straddling a split mixes two share bases.
    ttm = build_ttm(translated, field_map, actions=actions, report=report)
    ttm = ttm.rename(columns=_KEY_FROM_VENDOR)
    for column in ("as_of", "fiscal_end"):
        ttm[column] = pd.to_datetime(ttm[column], errors="coerce").astype("datetime64[ns]")

    collapsed, dropped = collapse_same_date(ttm)
    if not dropped.empty:
        log.warning("same-date collapse: %d row(s) dropped, greatest `fiscal_end` kept "
                    "(Sharadar ships no form column, so FORM_PRECEDENCE has no analogue):\n%s",
                    len(dropped), dropped[["ticker", "as_of", "fiscal_end"]].to_string())

    # The SEC-owned columns are all-NaN out of phase 3 -- drop them so the join can land the
    # real ones, rather than colliding into a `_x`/`_y` pair nothing downstream would read.
    sec_owned = [c for c in field_map.sec_owned if c != EMPLOYEES_COLUMN]
    joined = join_sec_block(collapsed.drop(columns=[*sec_owned, EMPLOYEES_COLUMN],
                                           errors="ignore"), sec)
    joined = attach_employees(joined, employees)
    joined, changed = apply_overrides(joined, overrides)
    joined = rederive(joined, field_map, changed | set(sec_owned))
    # The `_sec` suffix goes on LAST, after `rederive`. `stockholdersEquityInclNci` reads
    # `minorityInterest` under the name the field map declares, so renaming any earlier would
    # break the one formula that spans both sources -- silently, since a `sum` of a missing
    # input is caught by `apply_derived`'s own check but only after the column is gone.
    joined = joined.rename(columns={n: sec_column(n) for n in field_map.sec_owned})

    missing = [c for c in columns if c not in joined.columns]
    if missing:
        raise RuntimeError(f"the merged frame is missing {len(missing)} contract column(s): "
                           f"{missing}")
    carriers = [c for c in joined.columns if c.startswith(_SEC_PREFIX)]
    out = joined[list(columns)].copy()
    # Coverage is measured on `SEC_AS_OF` -- did a SEC row JOIN -- and never on a value
    # column. A ticker that genuinely has no `goodwill` reads as an unjoined ticker under a
    # value-column proxy, which is how a correct join gets reported as a broken one.
    joined_rows = int(joined[SEC_AS_OF].notna().sum())
    log.info("merged frame: %d row(s) x %d column(s); SEC block joined on %d row(s) over %d "
             "ticker(s), lag %s..%s day(s); %d override-only SEC column(s) dropped",
             len(out), len(out.columns), joined_rows,
             joined.loc[joined[SEC_AS_OF].notna(), "ticker"].nunique(),
             *(_lag_range(joined)), len(carriers))
    return _cast(out, columns)


def _lag_range(joined: pd.DataFrame) -> tuple[object, object]:
    """How far back the as-of carry actually reached, in days. Printed every run because the
    tolerance is the only thing standing between "a lag" and "a fabricated present"."""
    lag = (joined["as_of"] - joined[SEC_AS_OF]).dt.days.dropna()
    return (int(lag.min()), int(lag.max())) if not lag.empty else ("-", "-")


def _cast(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Pin every dtype before the write. See the module docstring for why this is load-bearing.

    ⚠ `regime` is excluded BY NAME. A cast that catches it turns every regime label into NaN,
    and nothing downstream would notice: the column would simply be empty.
    """
    out = frame.copy()
    for column in ("as_of", "fiscal_end"):
        out[column] = pd.to_datetime(out[column], errors="coerce").astype("datetime64[ns]")
    out["ticker"] = out["ticker"].astype(str)
    regime = sec_column("regime")
    out[regime] = out[regime].astype(object).where(out[regime].notna(), None)
    for column in columns:
        if column not in NON_VALUE_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    return out


def build_merged_history(context: Context, tickers: list[str], *, full: bool = False,
                         config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    """`fundamentals_sharadar` + `fundamentals_history_sec` -> `fundamentals_history`.

    Both inputs are READ-ONLY here; neither is ever written by this build, which is what makes
    the rollback a drop-and-rebuild costing one CLI run and no network.

    `full=True` DELETES these tickers' rows first. The difference from the default upsert is
    narrow but real: an upsert refreshes every row it rebuilds but cannot remove one that no
    longer exists -- a row the same-date collapse now drops, or a ticker that left the
    entitlement, would otherwise survive forever as a fossil under an unchanged key.

    ⚠ Not on its own schedule. A snapshot is only as fresh as the rows it reads, so this runs
    AFTER both producers, never beside them.
    """

    field_map = load_field_map(config_dir)
    overrides = load_overrides(config_dir)
    names = sorted({t.strip().upper() for t in tickers if t and t.strip()})

    vendor = context.store.load(Tables.sharadar_fundamentals, project=True,
                                where={"ticker": names, "dimension": ARQ}, optional=True)
    if vendor is None or vendor.empty:
        context.log.warning("merged history: no ARQ rows for %d requested ticker(s) -- "
                            "run `fundamentals-sharadar` first", len(names))
        return

    # `sharadar_actions` is MARKET-WIDE -- every ticker Sharadar covers, every action type.
    # Only these tickers' splits and spinoffs move a share count, so both halves of the filter
    # are needed; `project=True` alone would still drag the whole market back.
    actions = context.store.load(
        Tables.sharadar_actions, project=True, optional=True,
        where={"ticker": names, "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]})
    employees = context.store.load(Tables.fundamentals_employees, where={"ticker": names},
                                   optional=True)

    # The SEC projection is built FROM the register, so an approved override can never name a
    # column the join did not bring. `employees` is NOT a column of this table (see
    # EMPLOYEES_COLUMN) and asking for it would raise on a projection, not return NULLs.
    sec_owned = [c for c in field_map.sec_owned if c != EMPLOYEES_COLUMN]
    sec_columns = ["ticker", "as_of", *sec_owned]
    sec = context.store.load(Tables.fundamentals_history_sec,
                             columns=sec_columns + list(overrides.fields),
                             where={"ticker": names}, optional=True)
    if sec is None:
        context.log.warning("merged history: NO SEC rows for these tickers -- all 15 "
                            "SEC-owned columns will be NULL. That is the stated coverage "
                            "asymmetry (D14), not a failure.")
        sec = pd.DataFrame(columns=sec_columns)
    else:
        sec = sec.rename(columns={f: f"{_SEC_PREFIX}{f}" for f in overrides.fields})
    
    report = TranslationReport()
    frame = build_frame(vendor, sec, employees, actions, field_map, overrides, report=report)
    if frame.empty:
        context.log.warning("merged history: the transform produced 0 rows")
        return

    if full:
        # ONE `IN` delete, not one statement per ticker: `names` is the whole resolved
        # universe (~491) whatever the entitlement covers, and a per-ticker loop pays 491
        # round-trips to delete rows for the 30 that have any.
        deleted = context.store.delete(Tables.fundamentals_history, {"ticker": names})
        context.log.warning("merged history: --full deleted %d existing row(s) before the "
                            "rebuild (scope: %d ticker(s))", deleted, len(names))
        
    written = context.store.save(Tables.fundamentals_history, frame)
    covered = int(frame[sec_column("regime")].notna().sum())
    context.log.info(
        "merged history: %d row(s) over %d ticker(s), %s..%s | SEC block on %d row(s) "
        "(%d ticker(s)) -- the stated coverage asymmetry, not a gap | %d approved "
        "override(s), %d awaiting decision | %s",
        written, frame["ticker"].nunique(), frame["as_of"].min().date(),
        frame["as_of"].max().date(), covered,
        frame.loc[frame[sec_column("regime")].notna(), "ticker"].nunique(),
        len(overrides.approved), len(overrides.pending), report.summary())
    record_run(context, Tables.fundamentals_history, len(names), written, is_full_rescan=full)
