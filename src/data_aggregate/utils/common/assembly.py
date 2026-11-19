"""Scoped cube-part presence and validated upstream write footprints for assembly."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
from omegaconf import OmegaConf

from src.context import Context
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_store.schema import Table, resolve
from src.data_store.store import DataStore
from src.utils.data_policy import window_policy
from src.utils.window import Window, WindowPlan


def part_has_rows(store: DataStore, table: Table, where: dict | None = None) -> bool:
    """Probe physical presence before validating requested optional payload columns."""
    if not store.exists(table):
        return False
    probe = store.load(table, columns=store.columns(table)[:1], where=where, limit=1, optional=True)
    return probe is not None and not probe.empty


def write_receipt(context: Context, plans: dict[Table, WindowPlan]) -> str:
    """Serialize committed write bounds and the exact execution cutoff for CLI/DAG handoff."""
    if any(plan.run.as_of != context.run.as_of or plan.run.instant != context.run.instant for plan in plans.values()):
        raise ValueError("Component write plan belongs to another run")
    parts = {
        table.name: {"start": plan.write.start.isoformat(), "end": plan.write.end.isoformat(), "tickers": list(dict(plan.scope)["ticker"])}
        for table, plan in plans.items()
    }
    return json.dumps(
        {
            "as_of": context.run.as_of.isoformat(),
            "instant": context.run.instant.isoformat(),
            "mode": context.run.mode,
            "configuration": _pipeline_configuration(context),
            "parts": parts,
        },
        separators=(",", ":"),
    )


def _pipeline_configuration(context: Context) -> str:
    """Identify source/cube definitions; prediction and portfolio settings stay outside this protocol."""
    definitions = {key: OmegaConf.to_container(context.config[key], resolve=True) for key in ("data", "data_aggregate", "gpt", "universe")}
    return hashlib.sha256(json.dumps(definitions, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_receipts(context: Context, payload: str) -> dict[Table, Window]:
    """Validate component dates, complete key scope and the exact common root cutoff."""
    receipts = json.loads(payload)
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("Component receipts must be a nonempty list of successful upstream results")
    records = [json.loads(receipt) if isinstance(receipt, str) else receipt for receipt in receipts]
    configuration = _pipeline_configuration(context)
    fields = {"as_of", "instant", "mode", "configuration", "parts"}
    if any(
        not isinstance(record, dict)
        or set(record) != fields
        or not isinstance(record.get("parts"), dict)
        or pd.Timestamp(record["as_of"]) != context.run.as_of
        or pd.Timestamp(record["instant"]) != context.run.instant
        or record["mode"] != context.run.mode
        for record in records
    ):
        raise ValueError("A component receipt is missing or belongs to another root run or mode")
    if any(record["configuration"] != configuration for record in records):
        raise ValueError("Component receipt belongs to another data pipeline configuration")

    # Each component must cover the complete requested assembly scope.
    written = {}
    calendar = load_trading_calendar(context)
    for name, item in [(name, item) for record in records for name, item in record["parts"].items()]:
        if name not in PART_BY_NAME or not isinstance(item, dict) or set(item) != {"start", "end", "tickers"}:
            raise ValueError(f"{name}: malformed component write receipt")
        tickers = item["tickers"]
        if (
            not isinstance(tickers, list)
            or not tickers
            or any(not isinstance(ticker, str) or not ticker for ticker in tickers)
            or len(set(tickers)) != len(tickers)
            or set(tickers) != set(context.universe.tickers)
        ):
            raise ValueError(f"{name}: component receipt does not cover the complete requested ticker scope")
        table = resolve(name)
        policy = window_policy(context.config, table, context.run.as_of, calendar)
        interval = Window(pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))
        if policy is None or interval.intersection(policy.requested) != interval:
            raise ValueError(f"{name}: component receipt exceeds configured dates")
        if table in written:
            raise ValueError(f"{name}: duplicate component receipt")
        written[table] = interval
    return written
