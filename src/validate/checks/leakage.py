"""
leakage.py  (src/validate/checks/leakage.py)
--------------------------------------------------------------------------------------------
Can a number in this table have been known on the date it is filed under? Two halves, each
answering that for a different kind of column, and each able to run without the other.

**HORIZON RECESSION** -- the labels. A forward label over `n` days cannot be known until `n`
days have passed, so `max(date)` must RECEDE strictly as the horizon grows, and no label may
reach the last price session. Three horizons stopping on the same day means the longest one
was computed from data the shortest one had, and a label reaching the last price means it was
computed from a future that has not happened. Both are score 10, because a leaked label
inflates every backtest built on it and nothing downstream can detect it.

**`as_of` POINT-IN-TIME** -- the features. Per TICKER, a feature's first non-null date must not
precede the first event in its own source table, measured on that source's PUBLICATION clock.
Per ticker rather than per family is what makes it sharp: a family-wide floor hides a leak on
one name behind 490 that are clean.

⚠ THE PUBLICATION CLOCK IS NOT THE PERIOD CLOCK. A 13F describes a quarter that ended 45 days
before it was filed, and a Form 4 describes a trade that happened before it was reported. Four
tables in this registry publish on a different clock than the period they describe --
`pension_facts`, `notes_num` and `notes_text` (`filed`) and `insider_transactions`
(`filing_date`) -- so the comparison reads `resolve(src).freshness_col`, which is the
registry's own answer to that question, never `date_col` and never a name written here.

⚠ AND BOTH SIDES GO THROUGH `frame.as_ts`. `sec_13d.filing_date`, `sec_13g.filing_date` and
`sec_fails_to_deliver.date` are Postgres DATE and come back as `datetime.date`;
`prices.date`, `sec_short_interest.date` and the cube parts' `date` are TIMESTAMP and come
back as `pd.Timestamp`. Comparing one against the other raises rather than answering, and this
is exactly the bug class a parquet-cached harness never sees, because parquet round-trips both
as nanoseconds.

⚠ BOTH HALVES ARE DECLARED, AND EITHER CAN ABSTAIN ALONE. The PIT half needs `pit_sources`:
a feature-to-source map cannot be inferred from column names -- `f_ic_bo_` reads two tables and
`f_ic_sig_insider_` reads the same one as `f_ic_insider_` -- so a guessed map would report a
clean point-in-time result for legs it matched to the wrong clock. The horizon half needs
`label_pattern`, for the same reason one step removed: `_h30` in a column name does not make it
a forward label. `cube_part_momentum` carries `seasonal_h30`, `seasonal_h60` and `seasonal_h90`,
which are seasonality features read BACKWARD off past same-calendar-window returns; they run to
the last session because they are supposed to, and matching the pattern blind filed all three as
leaks at score 10. The result names whichever half did not run.

⚠ ONE KNOWN CONSERVATISM. Some legs read a source COLUMN that is only populated later than the
source table's first row (`percent_of_class` became mandatory long after 13D/G filings began).
Their true clock is therefore later than the one used here, so this check under-reports rather
than over-reports on them. That direction is deliberate: it cannot manufacture a violation.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Table, Tables, resolve
from src.validate.frame import as_ts, column_groups, feature_columns
from src.validate.io import CHUNK_ROWS, cache_used, read_columns
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.spec import load_spec

log = logging.getLogger(__name__)

CHECK = "leakage"

#: A label's horizon in days, read out of its own name. `_h30` -> 30. NOT a default: a table
#: must DECLARE `label_pattern` before the horizon half runs on it, because an `_h<n>` suffix
#: does not mean "forward label". `cube_part_momentum` carries `seasonal_h30/h60/h90`, which are
#: BACKWARD-looking seasonality features computed from past same-calendar-window returns -- they
#: correctly run to the last session, and matched blind they produced 5 findings at score 10.
HORIZON_PATTERN = r"_h(\d+)$"

#: Columns read at once by the PIT half -- one group is one pass over the table.
GROUP = 8

#: Findings filed per half before the rest collapse into the counts in `metrics`.
_MAX_FINDINGS = 25

#: Violating (ticker, leg) pairs carried in `metrics`, worst lead first.
_MAX_EXAMPLES = 50


def _horizons(columns: list[str], pattern: str) -> dict[str, tuple[int, str]]:
    """`leg -> (horizon in days, family)` for every column whose name declares a horizon.

    ⚠ THE FAMILY IS WHAT MAKES THE RECESSION TESTABLE, and leaving it out was a false positive
    measured on the live part. `cube_part_targets` carries THREE label families at each
    horizon -- `target_rank_h30`, `target_zscore_h30`, `target_epsilon_h30` -- and they end on
    the same day BECAUSE they are the same horizon. A ladder that sorts all nine by horizon
    and compares each leg to its predecessor reports six leaks where there are none. The
    family is the leg name with its own `_h<n>` cut out, so recession is only ever asserted
    between two horizons of the SAME label.
    """
    compiled = re.compile(pattern)
    out: dict[str, tuple[int, str]] = {}
    for column in columns:
        match = compiled.search(column)
        if match:
            family = (column[:match.start()] + column[match.end():]) or column
            out[column] = (int(match.group(1)), family)
    return out


def _first_per_ticker(frame: pd.DataFrame, ticker_col: str, date_col: str,
                      leg: str) -> pd.Series:
    """Per ticker, the first date on which `leg` is not null."""
    present = frame.loc[frame[leg].notna(), [ticker_col, date_col]]
    if present.empty:
        return pd.Series(dtype="datetime64[ns]")
    return present.groupby(ticker_col)[date_col].min()


def _source_first(context: Context, source: Any) -> tuple[pd.Series, str]:
    """Per ticker, the first date this source PUBLISHED anything, streamed.

    Returns `(series, clock)` where `clock` names the column used, so the result can show
    which clock each family was judged against rather than asking the reader to trust it."""
    spec = resolve(source)
    clock = spec.freshness_col
    if clock is None or spec.ticker_col is None:
        raise LookupError(f"`{spec.name}` declares no publication clock "
                          f"(freshness_col={clock!r}, ticker_col={spec.ticker_col!r})")
    live = context.store.columns(spec)
    if clock not in live or spec.ticker_col not in live:
        raise LookupError(f"`{spec.name}` does not carry {spec.ticker_col!r}/{clock!r}")

    running: pd.Series | None = None
    for chunk in context.store.iter_load(spec, columns=[spec.ticker_col, clock],
                                         chunksize=CHUNK_ROWS):
        chunk = chunk.assign(**{clock: as_ts(chunk[clock])})
        keys = chunk[spec.ticker_col].astype(str).str.strip().str.upper()
        part = chunk.groupby(keys)[clock].min()
        running = part if running is None else pd.concat([running, part]).groupby(level=0).min()
    return (running if running is not None else pd.Series(dtype="datetime64[ns]")), clock


def check_leakage(context: Context, table: Table | str, *, config: DictConfig,
                  cache: Any = None, tickers: list[str] | None = None,
                  horizon_pattern: str | None = None, reference: Table | str = Tables.prices,
                  group: int = GROUP, **kwargs: Any) -> CheckResult:
    """Horizon recession on the labels, and each feature against its source's own clock."""
    spec_t = resolve(table)
    if (declined := full_table_only(CHECK, spec_t.name, tickers)) is not None:
        return declined
    spec = load_spec(config, spec_t)
    live = context.store.columns(spec_t)
    date_col = spec_t.date_col
    ticker_col = spec_t.ticker_col if spec_t.ticker_col in live else None
    if date_col is None:
        return CheckResult.abstained(CHECK, spec_t.name,
                                     "the table declares no date column, so there is no "
                                     "publication clock to test anything against")

    columns = feature_columns(context, spec_t)
    findings: list[Finding] = []
    metrics: dict[str, Any] = {}
    halves: list[str] = []

    # ------------------------------------------------------------------ horizon recession #
    # An explicit `--pattern` overrides, so the check can still be pointed at a table
    # deliberately; without one it runs only where the table has declared it carries labels.
    pattern = horizon_pattern or spec.label_pattern
    horizon_reason = ""
    horizons = _horizons(columns, pattern) if pattern else {}
    if pattern and not horizons:
        horizon_reason = (f"no column matches the declared label pattern {pattern!r}, so the "
                          f"horizon half found nothing to test")
    elif not pattern:
        horizon_reason = (f"{spec_t.name} declares no `label_pattern`, so the horizon half "
                          f"ABSTAINED -- an `_h<n>` suffix does not make a column a forward "
                          f"label, and cube_part_momentum's backward-looking seasonal_h30/60/90 "
                          f"read as three leaks at score 10 when it was matched blind")
    last_price = as_ts(context.store.max_date(reference))
    if horizons:
        halves.append("horizon")
        frame = read_columns(context, spec_t, [date_col] + sorted(horizons), cache=cache)
        frame[date_col] = as_ts(frame[date_col])
        ladder = []
        for leg, (days, family) in sorted(horizons.items(), key=lambda kv: (kv[1][1], kv[1][0])):
            stamps = frame.loc[frame[leg].notna(), date_col]
            ladder.append({"leg": leg, "family": family, "horizon_days": days,
                           "last_date": stamps.max() if not stamps.empty else pd.NaT,
                           "n_ok": int(stamps.notna().sum())})

        families: dict[str, list[dict[str, Any]]] = {}
        for entry in ladder:
            families.setdefault(entry["family"], []).append(entry)
        for rungs in families.values():
            rungs.sort(key=lambda e: e["horizon_days"])
            for step, entry in enumerate(rungs):
                gap = ((last_price - entry["last_date"]).days
                       if pd.notna(entry["last_date"]) and pd.notna(last_price) else None)
                entry["days_behind_last_price"] = gap
                previous = rungs[step - 1] if step else None
                entry["recedes_by"] = (
                    (previous["last_date"] - entry["last_date"]).days
                    if previous is not None and pd.notna(entry["last_date"])
                    and pd.notna(previous["last_date"]) else None)
                if pd.isna(entry["last_date"]):
                    continue
                if pd.notna(last_price) and entry["last_date"] >= last_price:
                    findings.append(Finding.at(
                        10, field=entry["leg"],
                        observed=f"the {entry['horizon_days']}-day label runs to "
                                 f"{entry['last_date'].date()}, which is not before the last "
                                 f"{resolve(reference).name} session ({last_price.date()})",
                        expected=f"a {entry['horizon_days']}-day forward label cannot be known "
                                 f"until {entry['horizon_days']} days have passed, so its last "
                                 f"date must sit strictly behind the last price session. A "
                                 f"label reaching it was computed from a future that has not "
                                 f"happened",
                        **{k: v for k, v in entry.items() if k != "leg"}))
                if (previous is not None and pd.notna(previous["last_date"])
                        and entry["last_date"] >= previous["last_date"]):
                    findings.append(Finding.at(
                        10, field=entry["leg"],
                        observed=f"the {entry['horizon_days']}-day label ends on "
                                 f"{entry['last_date'].date()}, not before the "
                                 f"{previous['horizon_days']}-day label of the same family "
                                 f"(`{previous['leg']}`, {previous['last_date'].date()})",
                        expected="max(date) must RECEDE strictly as the horizon grows within "
                                 "one label family. Two horizons of the SAME label ending on "
                                 "the same day means the longer one was computed from data "
                                 "only the shorter one could have had",
                        shorter=previous["leg"], **{k: v for k, v in entry.items() if k != "leg"}))
        metrics["horizon_families"] = sorted(families)
        metrics["horizon_ladder"] = ladder
        metrics["last_reference_session"] = last_price

    # -------------------------------------------------------------------------- as_of PIT #
    pit_sheet: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    pit_reason = ""
    if not spec.pit_sources:
        pit_reason = (f"{spec_t.name} declares no `pit_sources`, so the point-in-time half "
                      f"ABSTAINED -- a feature-to-source map cannot be inferred from column "
                      f"names without judging some legs against the wrong clock")
    elif ticker_col is None:
        pit_reason = ("the point-in-time half ABSTAINED -- it is a per-ticker comparison and "
                      "this table carries no ticker column")
    else:
        halves.append("pit")
        clocks: dict[str, tuple[pd.Series, str]] = {}
        for prefix, sources in spec.pit_sources.items():
            firsts: list[pd.Series] = []
            names: list[str] = []
            for source in sources:
                if source not in clocks:
                    try:
                        clocks[source] = _source_first(context, source)
                    except (LookupError, ValueError) as exc:
                        log.warning("leakage %s: %s", spec_t.name, exc)
                        continue
                series, clock = clocks[source]
                firsts.append(series)
                names.append(f"{source}.{clock}")
            legs = [c for c in columns if c.startswith(prefix)]
            if not legs or not firsts:
                pit_sheet.append({
                    "prefix": prefix, "sources": list(sources), "legs": len(legs),
                    "clock": names, "tested": False,
                    "why": ("no live leg carries this prefix" if not legs
                            else "no source table could be read on its publication clock")})
                continue
            # Two sources for one family (`f_ic_bo_` reads 13D and 13G) means the feature can
            # exist as soon as EITHER has filed, so the clock is the earlier of the two.
            source_first = pd.concat(firsts).groupby(level=0).min()

            for block in column_groups(legs, group):
                frame = read_columns(context, spec_t, [ticker_col, date_col] + list(block),
                                     cache=cache)
                frame[date_col] = as_ts(frame[date_col])
                keys = frame[ticker_col].astype(str).str.strip().str.upper()
                frame = frame.assign(**{ticker_col: keys})
                for leg in block:
                    panel_first = _first_per_ticker(frame, ticker_col, date_col, leg)
                    joined = pd.DataFrame({"panel_first": panel_first}).join(
                        source_first.rename("src_first"), how="left")
                    early = joined[joined["panel_first"] < joined["src_first"]]
                    lead = (joined["src_first"] - joined["panel_first"]).dt.days
                    pit_sheet.append({
                        "prefix": prefix, "leg": leg, "clock": names, "tested": True,
                        "tickers_with_values": int(len(joined)),
                        "tickers_without_source": int(joined["src_first"].isna().sum()),
                        "tickers_early": int(len(early)),
                        "worst_lead_days": int(lead.max()) if len(early) else 0})
                    for ticker, row in early.iterrows():
                        violations.append({
                            "prefix": prefix, "leg": leg, "ticker": str(ticker),
                            "panel_first": row["panel_first"], "src_first": row["src_first"],
                            "lead_days": int((row["src_first"] - row["panel_first"]).days)})
                log.info("leakage %s: %s -> %d leg(s) tested", spec_t.name, prefix, len(block))

        by_leg: dict[str, list[dict[str, Any]]] = {}
        for row in violations:
            by_leg.setdefault(row["leg"], []).append(row)
        ranked = sorted(by_leg.items(),
                        key=lambda kv: -max(r["lead_days"] for r in kv[1]))
        for leg, rows in ranked[:_MAX_FINDINGS]:
            worst = max(rows, key=lambda r: r["lead_days"])
            findings.append(Finding.at(
                10, field=leg, ticker=worst["ticker"],
                observed=f"{len(rows)} ticker(s) carry a value before their first event in "
                         f"the source; worst {worst['ticker']}, first value "
                         f"{pd.Timestamp(worst['panel_first']).date()} against a first "
                         f"publication of {pd.Timestamp(worst['src_first']).date()} -- "
                         f"{worst['lead_days']:,} days early",
                expected="a feature cannot be non-null before the first date its own source "
                         "PUBLISHED anything for that ticker. Measured on the source's "
                         "freshness column, so a filing lag is already allowed for",
                n_tickers=len(rows), tickers=[r["ticker"] for r in rows][:20], worst=worst))

        metrics["pit_sheet"] = pit_sheet
        metrics["n_pit_violations"] = len(violations)
        metrics["pit_violations"] = sorted(violations, key=lambda r: -r["lead_days"])[:_MAX_EXAMPLES]
        metrics["pit_clocks"] = {s: c for s, (_, c) in clocks.items()}

    if not halves:
        return CheckResult.abstained(
            CHECK, spec_t.name, f"neither half could run. {horizon_reason}; and {pit_reason}")

    first_date, last_date = context.store.bounds(spec_t)
    scope = {"rows": context.store.row_count(spec_t), "first_date": first_date,
             "last_date": last_date,
             "halves_run": halves, "horizon_legs": len(horizons),
             "horizon_pattern": pattern,
             "reference": resolve(reference).name, "last_reference_session": last_price,
             "pit_prefixes": sorted(spec.pit_sources), "pit_legs_tested":
                 sum(1 for row in pit_sheet if row.get("tested")),
             "source": "cache" if cache_used(cache, spec_t) else "db"}
    reason = "; ".join(part for part in (horizon_reason, pit_reason) if part)
    return CheckResult.measured(CHECK, spec_t.name, findings, scope=scope, metrics=metrics,
                                reason=reason)
