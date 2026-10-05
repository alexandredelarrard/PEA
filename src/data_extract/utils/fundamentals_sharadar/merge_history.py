"""Merge the Sharadar TTM frame with the SEC-owned block into `fundamentals_history`, the table consumers read.

Field-block precedence: Sharadar owns its declared columns for all history, `fundamentals_history_sec` (plus
`fundamentals_employees`) owns the `sec`-kind columns, and no column switches source mid-series; the only
exception is a whole `(ticker, field)` series moved to SEC by the approved override register. SEC-sourced
columns carry the `_sec` suffix (applied last, after `rederive`). The SEC block is joined BACKWARD as-of
Sharadar's filing date within `SHARADAR_SEC_ASOF_TOLERANCE_DAYS` -- never forward. Every value column is cast
to float64 before the write except `regime_sec` (text), excluded by name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants.constants import (
    SHARADAR_ACTION_SPINOFF,
    SHARADAR_ACTION_SPLIT,
    SHARADAR_COLLAPSE_KEY,
    SHARADAR_COLLAPSE_ORDER,
    SHARADAR_CONFIG_SUBDIR,
    SHARADAR_OVERRIDE_APPROVED_KEY,
    SHARADAR_OVERRIDE_SOURCE_SEC,
    SHARADAR_REGISTER_DOC_PREFIX,
    SHARADAR_SEC_ASOF_TOLERANCE_DAYS,
    SHARADAR_SOURCE_OVERRIDES_FILENAME,
)
from src.context import Context
from src.data_extract.utils.common.frame_sanitize import pin_dtypes
from src.data_extract.utils.fundamentals.kpi_catalogue import DEFAULT_CONFIG_DIR
from src.data_extract.utils.fundamentals_sharadar.build_ttm import ARQ, build_ttm
from src.data_extract.utils.fundamentals_sharadar.field_map import FieldMap, TranslationReport, apply_derived, load_field_map, translate
from src.data_store.schema import Tables

log = logging.getLogger(__name__)

#: Merged-table key: `as_of` is Sharadar's `date` (FILING date), `fiscal_end` its `reportperiod` (period end).
MERGE_KEYS: tuple[str, ...] = ("ticker", "as_of", "fiscal_end")

#: Vendor -> merged names for the two keys that are not already repo-named.
_KEY_FROM_VENDOR: dict[str, str] = {"date": "as_of", "reportperiod": "fiscal_end"}

#: Suffix on every SEC-sourced column, so the name states which producer (and coverage) a NULL belongs to.
SEC_SUFFIX = "_sec"


def sec_column(name: str) -> str:
    return f"{name}{SEC_SUFFIX}"


#: Keys plus `regime_sec`, the only non-float column; excluded from the float cast by name.
NON_VALUE_COLUMNS: frozenset[str] = frozenset({*MERGE_KEYS, sec_column("regime")})

#: SEC-owned, but read from `fundamentals_employees`, not from `fundamentals_history_sec`.
EMPLOYEES_COLUMN = "employees"

#: Prefix carrying an override's SEC value through the join; not in the contract, so the final projection drops it.
_SEC_PREFIX = "__sec__"

#: The joined SEC row's own `as_of`, kept to measure the backward-join lag; dropped before the write.
SEC_AS_OF = "__sec_as_of__"


# --------------------------------------------------------------------------- #
# the override register (D22)                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Overrides:
    """The approved `(ticker, field) -> sec` decisions, plus the inert `pending` proposals (counted, never applied)."""

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
    """Read the override register; a missing file means no overrides.

    Approval is per entry (unapproved entries land in `pending`). Raises if an entry's source is not `sec` or it
    has no `reason`.
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
                    f"field-BLOCK change (D14) and belongs in sharadar_field_map.json."
                )
            if not str(entry.get("reason", "")).strip():
                raise RuntimeError(f"{where} has no `reason`. An override that cannot be re-checked when the roster widens is not a decision.")
            bucket = approved if entry.get(SHARADAR_OVERRIDE_APPROVED_KEY) else pending
            bucket[(ticker, field)] = entry
    return Overrides(approved=approved, pending=pending)


def write_overrides(entries: dict[str, dict[str, dict]], readme: list[str], config_dir: str = DEFAULT_CONFIG_DIR) -> Path:
    """Write the register in a stable shape, one sorted line per `(ticker, field)`, so an unchanged re-propose is byte-identical."""
    path = overrides_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{", '  "_README": [']
    lines += [f"    {json.dumps(line, ensure_ascii=False)}," for line in readme[:-1]]
    lines += [f"    {json.dumps(readme[-1], ensure_ascii=False)}", "  ],"]
    for i, (ticker, by_field) in enumerate(sorted(entries.items())):
        lines.append("")
        lines.append(f"  {json.dumps(ticker)}: {{")
        items = sorted(by_field.items())
        for j, (field, entry) in enumerate(items):
            comma = "," if j < len(items) - 1 else ""
            lines.append(f"    {json.dumps(field)}: {json.dumps(entry, ensure_ascii=False)}{comma}")
        lines.append("  }" + ("," if i < len(entries) - 1 else ""))
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the column contract                                                          #
# --------------------------------------------------------------------------- #
def merged_columns(field_map: FieldMap) -> tuple[str, ...]:
    """The merged column tuple (keys + field-map outputs, SEC-owned suffixed `_sec`).

    Raises unless it equals `Tables.fundamentals_history.read_columns` exactly, in order.
    """

    owned = set(field_map.sec_owned)
    columns = (*MERGE_KEYS, *(sec_column(n) if n in owned else n for n in field_map.outputs))
    declared = tuple(Tables.fundamentals_history.read_columns)
    if columns != declared:
        extra = [c for c in columns if c not in set(declared)]
        gone = [c for c in declared if c not in set(columns)]
        raise RuntimeError(
            f"the merged column contract disagrees with the registry: {len(columns)} built "
            f"vs {len(declared)} declared; only in the build {extra}; only in "
            f"schema.py {gone}. Fix BOTH -- they are the same decision written twice on "
            f"purpose."
        )
    return columns


# --------------------------------------------------------------------------- #
# the steps                                                                    #
# --------------------------------------------------------------------------- #
def collapse_same_date(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per `(ticker, as_of)`, keeping the greatest `fiscal_end` (Sharadar ships no form column).

    Returns `(collapsed frame, dropped rows)` so the caller can log the drop.
    """
    ordered = frame.sort_values([*SHARADAR_COLLAPSE_KEY, SHARADAR_COLLAPSE_ORDER])
    duplicated = ordered.duplicated(subset=list(SHARADAR_COLLAPSE_KEY), keep="last")
    return ordered[~duplicated].reset_index(drop=True), ordered[duplicated].copy()


def _asof_join(left: pd.DataFrame, right: pd.DataFrame, *, tolerance_days: int, also_to_datetime: tuple[str, ...] = ()) -> pd.DataFrame:
    """Backward as-of join on `as_of` by ticker, capped at `tolerance_days`.

    Both sides are normalised to `datetime64[ns]` first: `merge_asof` refuses mixed resolutions and Postgres DATE
    columns return as `datetime.date`.
    """
    left = left.copy()
    right = right.copy()
    left["as_of"] = pd.to_datetime(left["as_of"]).astype("datetime64[ns]")
    for column in ("as_of", *also_to_datetime):
        right[column] = pd.to_datetime(right[column]).astype("datetime64[ns]")
    right = right.dropna(subset=["as_of"])
    return pd.merge_asof(
        left.sort_values("as_of"),
        right.sort_values("as_of"),
        on="as_of",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    ).reset_index(drop=True)


def join_sec_block(sharadar: pd.DataFrame, sec: pd.DataFrame, *, tolerance_days: int = SHARADAR_SEC_ASOF_TOLERANCE_DAYS) -> pd.DataFrame:
    """Attach the SEC-owned columns as of each Sharadar filing date, backward only; the SEC `as_of` rides as `SEC_AS_OF`."""
    if sec.empty:
        out = sharadar.copy()
        out[SEC_AS_OF] = pd.NaT
        return out
    right = sec.rename(columns={"as_of": SEC_AS_OF}).copy()
    right["as_of"] = right[SEC_AS_OF]
    return _asof_join(sharadar, right, tolerance_days=tolerance_days, also_to_datetime=(SEC_AS_OF,))


def attach_employees(frame: pd.DataFrame, employees: pd.DataFrame | None, *, tolerance_days: int = SHARADAR_SEC_ASOF_TOLERANCE_DAYS) -> pd.DataFrame:
    """Attach annual headcount by backward as-of join, capped at `tolerance_days` so it reaches the following quarters only.

    A NULL row records a 10-K that states no usable count; it is dropped so it neither becomes
    the as-of match nor cuts the prior year's carry short.
    """
    if employees is not None:
        employees = employees.dropna(subset=[EMPLOYEES_COLUMN])
    if employees is None or employees.empty:
        out = frame.copy()
        out[EMPLOYEES_COLUMN] = np.nan
        return out
    right = employees[["ticker", "as_of", EMPLOYEES_COLUMN]]
    return _asof_join(frame, right, tolerance_days=tolerance_days)


def apply_overrides(frame: pd.DataFrame, overrides: Overrides) -> tuple[pd.DataFrame, set[str]]:
    """Replace each approved `(ticker, field)` series with the SEC one; returns `(frame, changed fields)`.

    Where SEC has no value the cell becomes NULL, never a fallback to Sharadar (logged per entry). Raises if an
    override's SEC column was not loaded. Pending proposals are counted and ignored.
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
                f"disagree -- never silently write a NULL over a real Sharadar value."
            )
        out.loc[rows, field] = out.loc[rows, column]
        changed.add(field)
        covered = int(out.loc[rows, field].notna().sum())
        log.warning(
            "override %s/%s -> sec (approved %s): %d of %d row(s) carry a value; the rest are NULL, NOT a fallback to Sharadar (D14). %s",
            ticker,
            field,
            entry.get(SHARADAR_OVERRIDE_APPROVED_KEY),
            covered,
            int(rows.sum()),
            entry.get("reason", ""),
        )
    if overrides.pending:
        # the count is the actionable part; the list goes to DEBUG to keep the log readable
        log.warning("%d override proposal(s) awaiting a decision and IGNORED", len(overrides.pending))
        log.debug("pending overrides: %s", sorted(f"{t}/{f}" for t, f in overrides.pending))
    return out, changed


def rederive(frame: pd.DataFrame, field_map: FieldMap, changed: set[str]) -> pd.DataFrame:
    """Recompute only the derived columns whose inputs the merge changed, never one that is itself in `changed`."""
    targets = {name for name, spec in field_map.derived.items() if spec.op != "quarter" and set(spec.inputs) & changed} - changed
    if not targets:
        return frame
    log.info("re-deriving %d column(s) whose inputs the merge changed: %s", len(targets), sorted(targets))
    return apply_derived(frame, field_map, only=targets)


# --------------------------------------------------------------------------- #
# the build                                                                    #
# --------------------------------------------------------------------------- #
def build_frame(
    sharadar_arq: pd.DataFrame,
    sec: pd.DataFrame,
    employees: pd.DataFrame | None,
    actions: pd.DataFrame | None,
    field_map: FieldMap,
    overrides: Overrides,
    *,
    yf_splits: pd.DataFrame | None = None,
    report: TranslationReport | None = None,
) -> pd.DataFrame:
    """The whole merge transform, no I/O.

    translate -> TTM (+ split de-adjustment) -> same-date collapse -> backward SEC join -> employees -> overrides
    -> re-derive -> `_sec` rename -> contract -> cast. Split events (`actions` + `yf_splits`) de-adjust only the
    `split_basis` columns (`sharesOutstandingPit`); other share columns stay on the vendor's split-adjusted basis.
    """
    columns = merged_columns(field_map)
    # an override on an already SEC-owned column is a contradiction, refused by name
    contradictory = sorted(set(overrides.fields) & set(field_map.sec_owned))
    if contradictory:
        raise RuntimeError(
            f"the override register moves {contradictory} to `sec`, but the field map already "
            f"declares them SEC-owned (D18). An override moves a SHARADAR-owned column; "
            f"changing which block a column belongs to is a field-map edit, not an override."
        )
    translated = translate(sharadar_arq, field_map, report=report)

    # split de-adjustment runs after the four-quarter aggregation, inside `build_ttm`
    ttm = build_ttm(translated, field_map, actions=actions, yf_splits=yf_splits, report=report)
    ttm = pin_dtypes(ttm.rename(columns=_KEY_FROM_VENDOR), dates=("as_of", "fiscal_end"))

    collapsed, dropped = collapse_same_date(ttm)
    if not dropped.empty:
        log.warning(
            "same-date collapse: %d row(s) dropped, greatest `fiscal_end` kept "
            "(Sharadar ships no form column, so FORM_PRECEDENCE has no analogue):\n%s",
            len(dropped),
            dropped[["ticker", "as_of", "fiscal_end"]].to_string(),
        )

    # drop the all-NaN SEC-owned placeholders so the join lands the real ones without `_x`/`_y`
    sec_owned = [c for c in field_map.sec_owned if c != EMPLOYEES_COLUMN]
    joined = join_sec_block(collapsed.drop(columns=[*sec_owned, EMPLOYEES_COLUMN], errors="ignore"), sec)
    joined = attach_employees(joined, employees)
    joined, changed = apply_overrides(joined, overrides)
    joined = rederive(joined, field_map, changed | set(sec_owned))
    # suffix last: `rederive` reads SEC-owned inputs under their field-map names
    joined = joined.rename(columns={n: sec_column(n) for n in field_map.sec_owned})

    missing = [c for c in columns if c not in joined.columns]
    if missing:
        raise RuntimeError(f"the merged frame is missing {len(missing)} contract column(s): {missing}")
    carriers = [c for c in joined.columns if c.startswith(_SEC_PREFIX)]
    out = joined[list(columns)].copy()
    # join coverage is measured on `SEC_AS_OF`, never on a value column that can be legitimately NULL
    joined_rows = int(joined[SEC_AS_OF].notna().sum())
    log.info(
        "merged frame: %d row(s) x %d column(s); SEC block joined on %d row(s) over %d "
        "ticker(s), lag %s..%s day(s); %d override-only SEC column(s) dropped",
        len(out),
        len(out.columns),
        joined_rows,
        joined.loc[joined[SEC_AS_OF].notna(), "ticker"].nunique(),
        *(_lag_range(joined)),
        len(carriers),
    )
    return _cast(out, columns)


def _lag_range(joined: pd.DataFrame) -> tuple[object, object]:
    """`(min, max)` lag in days between each row's `as_of` and its joined SEC `as_of`."""
    lag = (joined["as_of"] - joined[SEC_AS_OF]).dt.days.dropna()
    return (int(lag.min()), int(lag.max())) if not lag.empty else ("-", "-")


def _cast(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Pin every dtype before the write (an all-None object column would create a TEXT column); `regime_sec` stays text."""
    out = pin_dtypes(
        frame,
        dates=("as_of", "fiscal_end"),
        floats=[column for column in columns if column not in NON_VALUE_COLUMNS],
        texts=(sec_column("regime"),),
    )
    out["ticker"] = out["ticker"].astype(str)
    return out


def build_merged_history(context: Context, tickers: list[str], *, full: bool = False, config_dir: str = DEFAULT_CONFIG_DIR) -> None:
    """Build `fundamentals_history` from `fundamentals_sharadar` + `fundamentals_history_sec` for `tickers`.

    Inputs are read-only. Default upserts; `full=True` first deletes these tickers' rows so vanished keys go too.
    Must run after both producers.
    """

    field_map = load_field_map(config_dir)
    overrides = load_overrides(config_dir)
    names = sorted({t.strip().upper() for t in tickers if t and t.strip()})

    vendor = context.store.load(Tables.sharadar_fundamentals, project=True, where={"ticker": names, "dimension": ARQ}, optional=True)
    if vendor is None or vendor.empty:
        context.log.warning("merged history: no ARQ rows for %d requested ticker(s) -- run `fundamentals-sharadar` first", len(names))
        return

    # `sharadar_actions` is market-wide: scope to these tickers' splits and spinoffs (spinoffs only name co-dated splits)
    actions = context.store.load(
        Tables.sharadar_actions, project=True, optional=True, where={"ticker": names, "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]}
    )
    # second split source; `split_events` unions it with `sharadar_actions` under a corroboration rule
    yf_splits = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
    employees = context.store.load(Tables.fundamentals_employees, where={"ticker": names}, optional=True)

    # projection built from the register (so every override column is loaded); `employees` lives elsewhere
    sec_owned = [c for c in field_map.sec_owned if c != EMPLOYEES_COLUMN]
    sec_columns = ["ticker", "as_of", *sec_owned]
    sec = context.store.load(Tables.fundamentals_history_sec, columns=sec_columns + list(overrides.fields), where={"ticker": names}, optional=True)
    if sec is None:
        context.log.warning(
            "merged history: NO SEC rows for these tickers -- all 15 "
            "SEC-owned columns will be NULL. That is the stated coverage "
            "asymmetry (D14), not a failure."
        )
        sec = pd.DataFrame(columns=sec_columns)
    else:
        sec = sec.rename(columns={f: f"{_SEC_PREFIX}{f}" for f in overrides.fields})

    report = TranslationReport()
    frame = build_frame(vendor, sec, employees, actions, field_map, overrides, yf_splits=yf_splits, report=report)
    if frame.empty:
        context.log.warning("merged history: the transform produced 0 rows")
        return

    if full:
        # one `IN` delete rather than a round-trip per ticker
        deleted = context.store.delete(Tables.fundamentals_history, {"ticker": names})
        context.log.warning("merged history: --full deleted %d existing row(s) before the rebuild (scope: %d ticker(s))", deleted, len(names))

    written = context.store.save(Tables.fundamentals_history, frame)
    covered = int(frame[sec_column("regime")].notna().sum())
    context.log.info(
        "merged history: %d row(s) over %d ticker(s), %s..%s | SEC block on %d row(s) "
        "(%d ticker(s)) -- the stated coverage asymmetry, not a gap | %d approved "
        "override(s), %d awaiting decision | %s",
        written,
        frame["ticker"].nunique(),
        frame["as_of"].min().date(),
        frame["as_of"].max().date(),
        covered,
        frame.loc[frame[sec_column("regime")].notna(), "ticker"].nunique(),
        len(overrides.approved),
        len(overrides.pending),
        report.summary(),
    )
