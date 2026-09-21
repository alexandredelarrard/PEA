"""
catalogue.py  (src/validate/checks/catalogue.py)
--------------------------------------------------------------------------------------------
The live columns against a written feature catalogue, asserted in BOTH directions.

ONE DIRECTION IS THE EASY HALF AND THE USELESS ONE. "Every live column has a description"
finds a documentation gap. "Every described column is live" finds the defect that actually
cost this repo a quarter: the 2026-09-04 cube audit traced 46 dead field names and 39 dead
features to an `_sec` rename that was never applied to the consumers. The catalogue still
described them, the builder still referenced them, and sixty-eight tests stayed green because
none of them asked whether the names still resolved. A catalogued name with no live column is
a rename that did not finish.

⚠ ABSTAINS WITH NO `--catalogue`. There is no default catalogue and there must not be one: a
check that invents its own expectation and then meets it is a green light with nothing behind
it. Accepts a JSON object mapping `field -> description`, or a `.py` exposing a dict named
`CATALOGUE`, or one exposing `CATALOGUES` keyed by table name.
"""
from __future__ import annotations

import json
import logging
import runpy
from pathlib import Path
from typing import Any

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import key_columns
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.spec import UndeclaredTableError

log = logging.getLogger(__name__)

CHECK = "catalogue"

_MAX_FINDINGS = 40

#: How long a prefix may be before "the two namings differ by a prefix" stops being credible.
_AFFIX_MAX = 6

#: The share of catalogue entries a single prefix must reconcile before the mismatch is read as
#: a naming convention rather than as a table full of renames.
_AFFIX_SHARE = 0.8

_WHY = ("no --catalogue was given, and there is no default one to fall back on -- a check "
        "that supplies its own expectation and then meets it reports nothing. Pass a JSON "
        "object mapping field -> description, or a .py exposing a dict named CATALOGUE "
        "(or CATALOGUES keyed by table name)")


def _load(path: Path, table: str) -> dict[str, Any]:
    """The catalogue as `{field: description}`, from JSON or from a `.py`.

    A `.py` may expose either `CATALOGUE` (one table's sheet, as the per-report catalogues do)
    or `CATALOGUES` keyed by table name, which is the shape `scripts/cube_feature_catalogue.py`
    already ships for `cube_part_fundamentals` and `cube_part_governance`. Keyed wins when both
    are present: a file that registers several tables and also holds one of them under the bare
    name would otherwise hand every table the same sheet.
    """
    if path.suffix == ".py":
        namespace = runpy.run_path(str(path))
        registry = namespace.get("CATALOGUES")
        if isinstance(registry, dict) and table in registry:
            return dict(registry[table])
        entries = namespace.get("CATALOGUE")
        if not isinstance(entries, dict):
            known = sorted(registry) if isinstance(registry, dict) else []
            raise ValueError(
                f"{path} exposes no dict named CATALOGUE"
                + (f", and its CATALOGUES registers {known} but not `{table}`" if known else ""))
        return dict(entries)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a JSON object mapping field -> description")
    return loaded


def _reconciling_prefix(catalogued: set[str], live: set[str]) -> tuple[str, str, int]:
    """`(prefix, "add"|"strip", hits)` -- one uniform prefix that makes the two namings agree.

    ⚠ WITHOUT THIS THE CHECK REPORTS ITS OWN MISCONFIGURATION AS 216 DEFECTS. Measured
    2026-09-21: `institutionals_catalogue.py` keys on `ic_act_percent_of_class` while the live
    column is `f_ic_act_percent_of_class`, and `scripts/cube_feature_catalogue.py` keys on
    `auditor_tenure` against a live `f_auditor_tenure`. The intersection is EMPTY, so every one
    of the 89 entries reads as a dead catalogued name and every one of the 127 live columns as
    undocumented -- two findings per feature, none of them true, burying the real question,
    which is that the catalogue is written in a different naming convention from the schema.
    """
    best = ("", "", 0)
    prefixes = {name[:n] for name in live for n in range(1, _AFFIX_MAX + 1)}
    prefixes |= {name[:n] for name in catalogued for n in range(1, _AFFIX_MAX + 1)}
    for prefix in prefixes:
        added = sum(1 for name in catalogued if prefix + name in live)
        if added > best[2]:
            best = (prefix, "add", added)
        stripped = sum(1 for name in catalogued
                       if name.startswith(prefix) and name[len(prefix):] in live)
        if stripped > best[2]:
            best = (prefix, "strip", stripped)
    return best


def check_catalogue(context: Context, table: Table | str, *, config: Any = None,
                    cache: Any = None, tickers: list[str] | None = None,
                    catalogue: str | Path | None = None, **kwargs: Any) -> CheckResult:
    """Live columns vs a written catalogue, in both directions."""
    spec = resolve(table)
    if (declined := full_table_only(CHECK, spec.name, tickers)) is not None:
        return declined
    if catalogue is None:
        raise UndeclaredTableError(spec.name, "catalogue", _WHY)

    path = Path(catalogue)
    if not path.exists():
        return CheckResult.abstained(CHECK, spec.name, f"--catalogue {path} does not exist")
    try:
        entries = _load(path, spec.name)
    except (ValueError, json.JSONDecodeError, SyntaxError) as exc:
        return CheckResult.abstained(CHECK, spec.name, f"--catalogue {path}: {exc}")
    if not entries:
        return CheckResult.abstained(CHECK, spec.name, f"--catalogue {path} is empty")

    keys = set(key_columns(spec))
    live = [c for c in context.store.columns(spec) if c not in keys]
    if not live:
        return CheckResult.abstained(CHECK, spec.name,
                                     "the table has no non-key column -- is it built?")

    catalogued = set(entries)
    live_set = set(live)

    # ⚠ Not one defect per feature twice over -- see `_reconciling_prefix`.
    if not (catalogued & live_set):
        prefix, how, hits = _reconciling_prefix(catalogued, live_set)
        if hits >= _AFFIX_SHARE * len(catalogued):
            verb = f"prefixing them with `{prefix}`" if how == "add" else f"dropping `{prefix}`"
            return CheckResult.abstained(
                CHECK, spec.name,
                f"the catalogue and the live schema share NO column name, but {hits} of "
                f"{len(entries)} entries resolve to a live column once {verb} -- the two are "
                f"written in different naming conventions, not describing different tables. "
                f"Asserting either direction here would file {len(catalogued) + len(live_set)} "
                f"findings and none of them would be true. Reconcile the names, then re-run")

    missing = sorted(catalogued - live_set)      # described, not there
    undocumented = sorted(live_set - catalogued)  # there, not described
    blank = sorted(c for c in (catalogued & live_set)
                   if not str(entries[c] or "").strip())

    findings: list[Finding] = []
    for column in missing[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            9, field=column,
            observed=f"`{column}` is catalogued but is not a column of {spec.name}",
            expected="every catalogued field resolves to a live column. A described name "
                     "with nothing behind it is an unfinished rename -- the 2026-09-04 cube "
                     "audit traced 46 dead field names and 39 dead features to exactly this, "
                     "with every test green",
            description=str(entries[column])[:200]))

    for column in undocumented[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            4, field=column,
            observed=f"`{column}` is a live column with no catalogue entry",
            expected="every live column is described; an undocumented feature is one nobody "
                     "can audit the definition of"))

    for column in blank[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            4, field=column,
            observed=f"`{column}` has a catalogue entry with an empty description",
            expected="a description that says what the field means"))

    scope = {"catalogue": str(path), "entries": len(entries), "live_columns": len(live),
             "pk": sorted(keys)}
    metrics = {"entries": len(entries), "live_columns": len(live),
               "both": len(catalogued & live_set),
               "catalogued_not_live": missing, "live_not_catalogued": undocumented,
               "blank_descriptions": blank}
    return CheckResult.measured(CHECK, spec.name, findings, scope=scope, metrics=metrics)
