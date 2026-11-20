"""Volume-ledger check: does the saved volume tape stay in one share unit across every Yahoo ledger event?

Yahoo restates earlier volume by each ratio of its adjustment ledger, spin-off price factors included, and the price
build undoes the part that is no share change. Independent evidence is FINRA's total volume: converted to the same
terminal share basis by the genuine events dated after each day, FINRA / saved volume is flat across a correct event
and steps by the ledger ratio across an unrepaired one. Per ledger event, the median of that ratio over `WINDOW`
bars before the event is compared with the median over `WINDOW` bars after it, skipping the first `SKIP` bars (the
distribution days). An event in `REVIEWED` is not judged. An event whose step leaves `[1 / STEP_TOLERANCE, STEP_TOLERANCE]` is a finding, one per ticker;
an event with fewer than `MIN_BARS` matched bars on a side is counted as unverified. The genuine events are the
union of Sharadar and Yahoo splits (`split_events`), without the bugfix register's overrides.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.price.adjustments import split_events, suffix_factor
from src.validate.result import CheckResult, Finding

CHECK = "volume_ledger"
#: Events before this have no FINRA total volume to compare with.
SINCE = pd.Timestamp("2018-08-01")
#: Bars each side of an event whose median FINRA / volume is compared; the first `SKIP` bars after the event are left out.
WINDOW = 60
SKIP = 5
MIN_BARS = 20
#: A step of FINRA / volume beyond this factor (either direction) is a finding. A correct event measured 0.94-1.18 on
#: the five named spin-offs while the unrepaired ones stepped 2.2-6.8 (reports/validate/2026-11-19-yahoo-volume-spinoff).
STEP_TOLERANCE = 1.5
SCORE = 7
#: Reviewed events that stay outside the tolerance without a unit defect: the saved volume is in the right unit and the step
#: is a change of FINRA's share of consolidated volume. GE 2021-08-02 is a 1-for-8 reverse split (ledger 0.125 = genuine 0.125,
#: tape and FINRA carried by the same factor): FINRA / volume is 0.40 before and 0.24 after, not x8 or x0.125.
REVIEWED = frozenset({("GE", "2021-08-02")})
CHUNK = 50
STEP_COLUMNS = ("ticker", "date", "yahoo_ratio", "actual_ratio", "n_pre", "n_post", "pre", "post", "step")


def _median_if_enough(values: np.ndarray) -> tuple[int, float]:
    valid = values[np.isfinite(values) & (values > 0)]
    return len(valid), float(np.median(valid)) if len(valid) >= MIN_BARS else np.nan


def volume_ledger_steps(df_volume: pd.DataFrame, df_finra: pd.DataFrame, df_ledger: pd.DataFrame, df_actual: pd.DataFrame) -> pd.DataFrame:
    """`STEP_COLUMNS` per ledger event since `SINCE`: median terminal FINRA / volume before and after, and their quotient.

    Inputs are long frames `(ticker, date, volume)`, `(ticker, date, total_volume)`, the Yahoo ledger `(ticker, date, ratio)`
    and the genuine events `(ticker, date, value)`; none is modified.
    """
    ledger = df_ledger.assign(date=pd.to_datetime(df_ledger["date"]).dt.normalize(), ratio=pd.to_numeric(df_ledger["ratio"], errors="coerce"))
    ledger = ledger[np.isfinite(ledger["ratio"]) & ledger["ratio"].gt(0) & ledger["date"].ge(SINCE)]
    ledger = ledger.groupby(["ticker", "date"], as_index=False)["ratio"].prod().rename(columns={"ratio": "yahoo_ratio"})
    actual = df_actual.assign(date=pd.to_datetime(df_actual["date"]).dt.normalize()).groupby(["ticker", "date"], as_index=False)["value"].prod()
    ledger = ledger.merge(actual.rename(columns={"value": "actual_ratio"}), on=["ticker", "date"], how="left")
    bars = df_volume.assign(date=pd.to_datetime(df_volume["date"]).dt.normalize()).merge(
        df_finra.assign(date=pd.to_datetime(df_finra["date"]).dt.normalize()), on=["ticker", "date"]
    )
    rows: list[dict[str, Any]] = []
    for ticker, events in ledger.groupby("ticker", sort=True):
        tape = bars[bars["ticker"] == ticker].sort_values("date")
        dates = tape["date"].to_numpy(dtype="datetime64[ns]")
        terminal = tape["total_volume"].to_numpy(dtype="float64") * suffix_factor(actual, [str(ticker)], dates).get(str(ticker), np.ones(len(tape)))
        traded = tape["volume"].to_numpy(dtype="float64")
        ratio = terminal / np.where(traded > 0, traded, np.nan)
        for event in events.itertuples(index=False):
            pos = int(np.searchsorted(dates, np.datetime64(event.date, "ns"), side="left"))
            n_pre, pre = _median_if_enough(ratio[max(0, pos - WINDOW) : pos])
            n_post, post = _median_if_enough(ratio[pos + SKIP : pos + SKIP + WINDOW])
            rows.append(
                {
                    "ticker": ticker,
                    "date": event.date,
                    "yahoo_ratio": event.yahoo_ratio,
                    "actual_ratio": event.actual_ratio,
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "pre": pre,
                    "post": post,
                    "step": post / pre,
                }
            )
    return pd.DataFrame(rows, columns=list(STEP_COLUMNS))


def step_findings(steps: pd.DataFrame) -> list[Finding]:
    """One finding per ticker with an event whose step is outside the tolerance, worst event first in the evidence."""
    reviewed = [(str(t), str(d.date())) in REVIEWED for t, d in zip(steps["ticker"], steps["date"], strict=True)]
    measured = steps[steps["step"].notna() & ~np.array(reviewed, dtype=bool)]
    outside = measured[np.abs(np.log(measured["step"])) > np.log(STEP_TOLERANCE)].assign(_gap=lambda f: np.abs(np.log(f["step"])))
    findings = []
    for ticker, group in outside.groupby("ticker", sort=True):
        group = group.sort_values("_gap", ascending=False, kind="mergesort")
        events = [
            {"date": str(row.date.date()), "step": round(float(row.step), 3), "yahoo_ratio": float(row.yahoo_ratio), "actual_ratio": row.actual_ratio}
            for row in group.itertuples(index=False)
        ]
        worst = group.iloc[0]
        findings.append(
            Finding.at(
                SCORE,
                f"FINRA / volume steps x{worst['step']:.2f} across the ledger event of {worst['date'].date()} (Yahoo ratio {worst['yahoo_ratio']:g}); "
                f"{len(group)} event(s) outside the tolerance",
                f"a step within x{1 / STEP_TOLERANCE:.2f}..x{STEP_TOLERANCE:.2f}: the saved volume is in one share unit across the event",
                field="volume",
                ticker=str(ticker),
                events=events,
            )
        )
    return findings


def _load(context: Context, table: Any, columns: list[str], where: dict[str, Any], **bounds: Any) -> pd.DataFrame:
    rows = context.store.load(table, columns=columns, where=where, optional=True, **bounds)
    return rows if rows is not None else pd.DataFrame(columns=columns)


def check_volume_ledger(context: Context, *, tickers: Sequence[str] | None = None) -> CheckResult:
    """Read-only: the saved `cube_part_prices.volume` against FINRA total volume across every ledger event of the scope."""
    part = Tables.cube_part_prices
    needed = (part, Tables.prices_splits, Tables.sec_short_volume_security)
    scope: dict[str, Any] = {"since": str(SINCE.date()), "window": WINDOW, "skip": SKIP, "step_tolerance": STEP_TOLERANCE}
    if not all(context.store.exists(table) for table in needed):
        return CheckResult.abstained(
            CHECK, part.name, "A required table is missing: " + ", ".join(t.name for t in needed if not context.store.exists(t)), **scope
        )
    names = sorted(tickers if tickers else context.universe.tickers)
    actions = (
        _load(context, Tables.sharadar_actions, ["ticker", "date", "action", "value"], {"ticker": names})
        if context.store.exists(Tables.sharadar_actions)
        else None
    )
    parts, bar_rows, first, last = [], 0, None, None
    for start in range(0, len(names), CHUNK):
        chunk = names[start : start + CHUNK]
        splits = _load(context, Tables.prices_splits, ["ticker", "date", "ratio"], {"ticker": chunk})
        volume = _load(context, part, ["ticker", "date", "volume"], {"ticker": chunk}, since=SINCE - pd.Timedelta(days=200), date_col="date")
        finra = _load(context, Tables.sec_short_volume_security, ["source_symbol", "date", "total_volume"], {"source_symbol": chunk}).rename(
            columns={"source_symbol": "ticker"}
        )
        genuine = split_events(
            None if actions is None else actions[actions["ticker"].isin(chunk)], splits if len(splits) else None, report_events=False
        )
        parts.append(volume_ledger_steps(volume, finra, splits, genuine))
        bar_rows += len(volume)
        if len(volume):
            first = volume["date"].min() if first is None else min(first, volume["date"].min())
            last = volume["date"].max() if last is None else max(last, volume["date"].max())
    steps = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(STEP_COLUMNS))
    findings = step_findings(steps)
    scope.update({"rows": bar_rows, "tickers": len(names), "first_date": first, "last_date": last})
    metrics = {
        "events": len(steps),
        "events_measured": int(steps["step"].notna().sum()),
        "events_unverified": int(steps["step"].isna().sum()),
        "events_reviewed": sum((str(t), str(d.date())) in REVIEWED for t, d in zip(steps["ticker"], steps["date"], strict=True)),
        "events_outside_tolerance": int(sum(len(f.evidence["events"]) for f in findings)),
        "tickers_flagged": len(findings),
    }
    context.log.info(
        "SANITY: %d ledger events since %s, %d measured against FINRA, %d outside x%.2f on %d ticker(s)",
        metrics["events"],
        SINCE.date(),
        metrics["events_measured"],
        metrics["events_outside_tolerance"],
        STEP_TOLERANCE,
        metrics["tickers_flagged"],
    )
    return CheckResult.measured(CHECK, part.name, findings, scope=scope, metrics=metrics)
