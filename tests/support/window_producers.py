"""Actual cube creation/rerun evidence in the permanent pinned-sample lane.

The native steps retain their complete calculation populations. Only real output
rows for the immutable observation keys are persisted, in disposable SQLite.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common import peers_io
from src.data_aggregate.utils.common.incremental import plan_window
from src.data_aggregate.utils.fundamentals.feature_views import HISTORY_FEATURES
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_store.schema import Tables, resolve
from src.data_store.store import DataStore
from src.utils.db import get_engine
from src.utils.window import Window, WindowRun
from src.validate.result import CheckResult, jsonable
from tests import live_guard as _package_guard

# Pytest registers the same guard file as a flat plugin; direct CLI imports its package.
live_guard = sys.modules.get("live_guard", _package_guard)

_CASES = {
    "prices": ("step_cube_prices", "StepCubePrices", (Tables.cube_part_prices,)),
    "targets": ("step_cube_target", "StepCubeTarget", (Tables.cube_part_targets, Tables.cube_part_betas)),
    "momentum": ("step_cube_momentum", "StepCubeMomentum", (Tables.cube_part_momentum,)),
    "fundamentals": ("step_cube_fundamentals", "StepCubeFundamentals", (Tables.cube_part_fundamentals,)),
    "governance": ("step_cube_governance", "StepCubeGovernance", (Tables.cube_part_governance,)),
    "institutionals": ("step_cube_institutionals", "StepCubeInstitutionals", (Tables.cube_part_institutionals,)),
    "sec_thirteen": ("step_cube_sec_thirteen", "StepCubeSecThirteen", (Tables.cube_part_sec_thirteen,)),
    "text": ("step_cube_text", "StepCubeText", (Tables.cube_part_text,)),
    "assembly": ("step_assemble_cube", "StepAssembleCube", (Tables.cube,)),
}
_READS = {
    "exists",
    "columns",
    "load",
    "iter_load",
    "max_date",
    "distinct",
    "row_count",
    "bounds",
    "summarize",
    "key_stats",
    "max_date_by",
    "same_rows",
}


class MissingProducerEvidenceError(ValueError):
    """A required real input is unavailable; no substitute may be invented."""


def producer_cases() -> tuple[str, ...]:
    return tuple(_CASES)


def _checkpoint(record: dict[str, Any], directory: Any) -> None:
    # Completed phase evidence survives the existing process deadline/watchdog.
    (directory / "producer.json").write_text(json.dumps(jsonable(record), indent=2), encoding="utf-8")


def reference_config(config: Any, parts: tuple[Any, ...]) -> Any:
    """Full stored inputs with the native observed calendar; assembly remains interval-only."""
    result = deepcopy(config)
    for table in parts:
        if table != Tables.cube:
            for name in result.data.tables[table.name].warmup:
                result.data.tables[table.name].warmup[name] = None
    return result


def _context(base: Any, config: Any, store: Any, mode: str) -> Any:
    return SimpleNamespace(
        config=config,
        config_dir=base.config_dir,
        paths=base.paths,
        store=store,
        run=WindowRun(mode, "Pinned actual cube creation/rerun", base.run.as_of),
        universe=base.universe,
        log=base.log,
    )


class ProducerStore:
    """The existing native producer seam: readonly inputs and sampled disposable writes."""

    def __init__(
        self,
        source: Any,
        output: DataStore,
        parts: tuple[Any, ...],
        pins: dict[Any, list[str]],
        record: dict[str, Any],
        history: DataStore | None = None,
        directory: Any = None,
    ) -> None:
        self.source, self.output, self.parts, self.pins, self.record, self.history, self.directory = (
            source,
            output,
            parts,
            pins,
            record,
            history,
            directory,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "NOT_NULL":
            return self.source.NOT_NULL
        if name not in _READS:
            raise AttributeError(name)

        def read(table: Any, *args: Any, **kwargs: Any) -> Any:
            store = self.output if table in self.parts else self.source
            fields = set(kwargs.get("columns") or ())
            if table == Tables.cube_part_fundamentals and self.history is not None and name == "load":
                if fields and fields <= {"date", "ticker", *[f"f_{field}" for field in HISTORY_FEATURES]}:
                    store = self.history
            entry = {
                "method": name,
                "table": table.name,
                "route": "source" if store is self.source else "reference_history" if store is self.history else "disposable_output",
                "query": kwargs,
            }
            self.record["reads"].append(entry)
            result = getattr(store, name)(table, *args, **kwargs)
            if name == "iter_load":
                return self._stream(result, entry)
            if isinstance(result, pd.DataFrame):
                entry.update(rows=len(result), columns=list(result))
            return result

        return read

    @staticmethod
    def _stream(stream: Any, entry: dict[str, Any]) -> Any:
        entry.update(rows=0, chunks=0)
        try:
            for frame in stream:
                entry["rows"] += len(frame)
                entry["chunks"] += 1
                yield frame
        finally:
            if hasattr(stream, "close"):
                stream.close()

    @contextmanager
    def transaction(self) -> Any:
        with self.output.transaction() as bound:
            yield ProducerStore(self.source, bound, self.parts, self.pins, self.record, self.history, self.directory)

    def replace_window(self, table: Any, rows: pd.DataFrame, window: Window, scope: dict[str, Any], *, schema_rebuild: bool = False) -> Any:
        if table not in self.parts:
            raise RuntimeError(f"{table.name}: producer attempted a source mutation")
        selected = rows.loc[rows.ticker.isin(self.pins[table])]
        effects = self.output.replace_window(table, selected, window, {**scope, "ticker": self.pins[table]}, schema_rebuild=schema_rebuild)
        self.record["writes"].append(
            {
                "table": table.name,
                "window": {"start": window.start, "end": window.end},
                "scope": scope,
                "observed_keys": self.pins[table],
                "deleted": effects.deleted,
                "inserted": effects.inserted,
            }
        )
        return effects


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result.date)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def _tolerance(table: Any) -> tuple[float, float]:
    # Existing full/tail producer tolerances, retained from the original parity cases.
    return {
        Tables.cube_part_prices: (0.0, 0.0),
        Tables.cube_part_momentum: (1e-9, 0.0),
        Tables.cube_part_fundamentals: (5e-7, 2e-6),
        Tables.cube_part_betas: (1e-10, 1e-5),
    }.get(table, (1e-8, 1e-5))


def _numeric_fields(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame
        if name not in {"date", "ticker"} and pd.api.types.is_numeric_dtype(frame[name].dtype) and not pd.api.types.is_bool_dtype(frame[name].dtype)
    ]


def compare_producer_rows(expected: pd.DataFrame, actual: pd.DataFrame, table: Any) -> dict[str, Any]:
    expected, actual = _ordered(expected), _ordered(actual)
    if expected.empty:
        raise MissingProducerEvidenceError(f"{table.name}: no actual output rows in this observation interval")
    assert list(expected.columns) == list(actual.columns), f"{table.name}: physical feature columns differ"
    pd.testing.assert_frame_equal(expected[["date", "ticker"]], actual[["date", "ticker"]], check_dtype=False)
    fields = [name for name in expected if name not in {"date", "ticker"}]
    if not fields:
        raise MissingProducerEvidenceError(f"{table.name}: no physical economic feature columns")
    assert expected[fields].isna().equals(actual[fields].isna()), f"{table.name}: output null masks differ"
    actual_numeric = set(_numeric_fields(actual))
    numeric = [name for name in _numeric_fields(expected) if name in actual_numeric]
    exact = [name for name in fields if name not in numeric]
    pd.testing.assert_frame_equal(expected[exact], actual[exact], check_dtype=False, check_exact=True)
    left = expected[numeric].to_numpy(dtype=float, na_value=np.nan)
    right = actual[numeric].to_numpy(dtype=float, na_value=np.nan)
    atol, rtol = _tolerance(table)
    np.testing.assert_allclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
    finite = np.isfinite(left) & np.isfinite(right)
    return {
        "rows": len(expected),
        "features": len(fields),
        "numeric_fields": numeric,
        "exact_fields": exact,
        "finite_cells": int(finite.sum()),
        "null_cells": int(expected[fields].isna().sum().sum()),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": float(np.abs(left[finite] - right[finite]).max()) if finite.any() else None,
        "per_key": {
            ticker: {
                "rows": int(expected.ticker.eq(ticker).sum()),
                "finite_cells": int(np.isfinite(expected.loc[expected.ticker.eq(ticker), numeric].to_numpy(dtype=float, na_value=np.nan)).sum()),
            }
            for ticker in expected.ticker.unique()
        },
    }


def _execute(base: Any, config: Any, route: ProducerStore, module: Any, cls: Any, phase: str, record: dict[str, Any]) -> dict[Any, Any]:
    context = _context(base, config, route, "full" if phase == "reference" else "incremental")
    step = cls(context, config)
    original = module.write_part

    def observe(store: Any, table: Any, rows: pd.DataFrame, window: Any, **kwargs: Any) -> Any:
        if rows is not None:
            record.setdefault("candidates", []).append(
                {"phase": phase, "table": table.name, "rows": len(rows), "tickers": int(rows.ticker.nunique())}
            )
            if phase == "reference" and table == Tables.cube_part_fundamentals and route.history is not None:
                fields = [f"f_{field}" for field in sorted(HISTORY_FEATURES) if f"f_{field}" in rows]
                route.history.replace(table, rows[["date", "ticker", *fields]], chunksize=1_000)
                record["reference_history"] = {
                    "rows": len(rows),
                    "fields": fields,
                    "source": "Actual full native candidate before sampled persistence",
                }
        return original(store, table, rows, window, **kwargs)

    started = time.perf_counter()
    with (
        patch.object(module, "write_part", observe),
        patch.object(
            peers_io.StepDeducePeers,
            "run",
            side_effect=MissingProducerEvidenceError("Required peer cache unavailable; peer recomputation is forbidden"),
        ),
    ):
        if record["case"] == "text":
            calendar = load_trading_calendar(context)
            plan = step.plan_window(calendar)
            if plan is None or not plan.eligible:
                raise MissingProducerEvidenceError("No eligible cached earnings-call feature inputs; inference is forbidden")
            record.setdefault("text_cache", {})[phase] = {
                "eligible": len(plan.eligible),
                "missing_sentiment": len(plan.sentiment_work),
                "missing_embeddings": len(plan.embedding_work),
            }
            cached = {}
            for ticker in route.pins[Tables.cube_part_text]:
                cached[ticker] = {
                    "source_calls": int(plan.source.ticker.eq(ticker).sum()),
                    "eligible_cached_sentiment_calls": sum(key[0] == ticker for key in plan.eligible),
                    "missing_sentiment_calls": sum(key[0] == ticker for key in plan.sentiment_work),
                    "missing_embedding_calls": sum(key[0] == ticker for key in plan.embedding_work),
                }
            record["text_cache"][phase]["original_keys"] = cached
            if any(not item["source_calls"] or item["missing_sentiment_calls"] or item["missing_embedding_calls"] for item in cached.values()):
                raise MissingProducerEvidenceError("Original text keys lack complete cached source/feature evidence; inference is forbidden")
            step.run_features(calendar, plan)
            plans = {Tables.cube_part_text: plan.window}
        else:
            plans = step.run(tickers=route.pins[Tables.cube_part_prices]) if record["case"] == "prices" else step.run()
    record["phases"].append(
        {
            "phase": phase,
            "elapsed_seconds": time.perf_counter() - started,
            "plans": {
                table.name: {
                    "requested": {"start": plan.requested.start, "end": plan.requested.end},
                    "read": {"start": plan.read.start, "end": plan.read.end},
                    "write": {"start": plan.write.start, "end": plan.write.end},
                }
                for table, plan in plans.items()
            },
        }
    )
    _checkpoint(record, route.directory)
    return plans


class _AssemblyInputs:
    """Assembly may consume only completed actual producer checkpoints of this policy."""

    def __init__(self, source: Any, stores: dict[Any, DataStore]) -> None:
        self.source, self.stores = source, stores

    def __getattr__(self, name: str) -> Any:
        if name not in _READS:
            return getattr(self.source, name)

        def read(table: Any, *args: Any, **kwargs: Any) -> Any:
            return getattr(self.stores.get(table, self.source), name)(table, *args, **kwargs)

        return read


def _assembly_inputs(run: Any, source: Any, pins: list[str]) -> Any:
    stores = {}
    for name in run.context.config.data.tables.cube.warmup:
        table = resolve(name)
        case = next(key for key, (_, _, parts) in _CASES.items() if table in parts)
        directory = run.out / f"producer-{case}"
        path, database = directory / "producer.json", directory / "outputs.sqlite"
        if not path.is_file() or not database.is_file():
            raise MissingProducerEvidenceError(f"Assembly requires completed actual {case} producer checkpoints")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") != "pass" or metadata.get("identity") != run.identity:
            raise MissingProducerEvidenceError(f"Assembly {case} producer checkpoint has no matching accepted policy/source identity")
        if metadata.get("database_sha256") != hashlib.sha256(database.read_bytes()).hexdigest():
            raise MissingProducerEvidenceError(f"Assembly {case} producer checkpoint hash changed")
        if not set(pins) <= set(metadata["pins"][table.name]):
            raise MissingProducerEvidenceError(
                f"Assembly fixed keys are absent from actual {case} producer checkpoint; original pins are not replaced"
            )
        stores[table] = DataStore(get_engine("sqlite:///" + str(database.resolve()).replace("\\", "/")))
    return _AssemblyInputs(source, stores)


def measure_window_producer(run: Any, case: str) -> CheckResult:
    """One deadline, current policy, immutable pins, native producer and disposable writes."""
    if case not in _CASES:
        raise ValueError(f"Unknown cube producer case: {case}")
    module_name, class_name, parts = _CASES[case]
    record: dict[str, Any] = {
        "case": case,
        "status": "abstain",
        "identity": run.identity,
        "limitations": "Current-policy native full/bounded comparison; immutable sampled source cells do not freeze all vendor revisions. Accepted governance later-filing fills and pooled13F repairs retain their retrospective semantics.",
        "source_identity_scope": "Immutable manifest cells only; full native calculation context is shared in one readonly source transaction",
        "reads": [],
        "writes": [],
        "phases": [],
        "comparisons": {},
    }
    directory = run.out / f"producer-{case}"
    started = time.perf_counter()
    error = None
    owned = False
    run.producers[case] = CheckResult.abstained("window_producer", case, "Not completed before interruption/deadline").to_json()
    run.publish()
    try:
        if run.context is None or run.capturing:
            raise MissingProducerEvidenceError("Actual producers require an existing immutable source baseline and readonly source context")
        if not live_guard._installed:
            raise RuntimeError("Actual producers require live-effect guards before project imports")
        directory.mkdir(exist_ok=False)
        owned = True
        _checkpoint(record, directory)
        pins = {table: run.manifest["tables"].get(table.name, {}).get("sample_scope", {}).get("groups", []) for table in parts}
        if any(len(values) != 5 or len(set(values)) != 5 for values in pins.values()):
            raise MissingProducerEvidenceError("Original immutable manifest must provide exactly five distinct output keys per table")
        if any(not set(values) <= set(run.context.universe.tickers) for values in pins.values()):
            raise MissingProducerEvidenceError("Original fixed output keys are absent from the current real universe")
        record["pins"] = {table.name: values for table, values in pins.items()}
        record["calculation_population"] = len(run.context.universe.tickers)
        peer = run.context.paths.get("SECTOR_PEERS_PATH")
        if case not in {"text", "assembly"} and (peer is None or not peer.is_file()):
            raise MissingProducerEvidenceError("Required existing peer cache absent; it cannot be recomputed")
        peer_hash = hashlib.sha256(peer.read_bytes()).hexdigest() if peer is not None and peer.is_file() else None
        record["peer_artifact_sha256"] = peer_hash
        database = directory / "outputs.sqlite"
        output = DataStore(get_engine("sqlite:///" + str(database.resolve()).replace("\\", "/")))
        history = (
            DataStore(get_engine("sqlite:///" + str((directory / "history.sqlite").resolve()).replace("\\", "/"))) if case == "fundamentals" else None
        )
        module = importlib.import_module(f"src.data_aggregate.transformers.{module_name}")
        cls = getattr(module, class_name)
        config, base = run.context.config, run.context
        dependencies = {resolve(name) for table in parts for name in config.data.tables[table.name].warmup}
        with base.store.transaction() as source:
            try:
                run.context = _context(base, config, source, base.run.mode)
                # Compare raw pinned inputs, not obsolete pipeline-policy metadata.
                for table in sorted(dependencies, key=lambda item: item.name):
                    result = run.measure(table)
                    if result.status != "pass":
                        raise MissingProducerEvidenceError(f"{table.name}: unchanged pinned source evidence unavailable: {result.reason}")
            finally:
                run.context = base
            if case == "assembly":
                source = _assembly_inputs(run, source, pins[Tables.cube])
            route = ProducerStore(source, output, parts, pins, record, history, directory)
            full = reference_config(config, parts)
            record["reference_input_policy"] = {table.name: dict(full.data.tables[table.name].warmup) for table in parts}
            _execute(base, full, route, module, cls, "reference", record)
            expected, corruptions = {}, {}
            planning = _context(base, config, route, "incremental")
            calendar = load_trading_calendar(planning)
            for table in parts:
                frame = output.load(table, where={"ticker": pins[table]}, optional=True)
                if frame is None or frame.empty:
                    raise MissingProducerEvidenceError(f"{table.name}: native full producer yielded no actual sampled rows")
                expected[table] = frame.copy()
                plan = plan_window(planning, table, calendar, pins[table] if case == "prices" else None)
                if plan is None:
                    raise MissingProducerEvidenceError(f"{table.name}: no native incremental write interval")
                tail = frame.loc[plan.write.mask(frame.date)]
                fields = _numeric_fields(tail)
                finite = np.argwhere(np.isfinite(tail[fields].to_numpy(dtype=float, na_value=np.nan)))
                if not len(finite):
                    raise MissingProducerEvidenceError(f"{table.name}: no real finite rewrite cell for overwrite probe")
                row, column = finite[0]
                day, ticker = tail.iloc[row][["date", "ticker"]]
                field, value = fields[column], float(tail.iloc[row][fields[column]])
                wrong = value + max(17.0, abs(value) * 0.01)
                output.update(table, {field: wrong}, where={"date": pd.Timestamp(day), "ticker": ticker})
                changed = output.load(table, columns=[field], where={"date": pd.Timestamp(day), "ticker": ticker})
                assert changed is not None and len(changed) == 1, "Overwrite probe did not select exactly one real saved row"
                atol, rtol = _tolerance(table)
                assert not np.isclose(float(changed.iloc[0][field]), value, atol=atol, rtol=rtol), (
                    "Overwrite probe did not alter the saved numeric value"
                )
                corruptions[table] = {"date": pd.Timestamp(day), "ticker": ticker, "field": field, "expected": value, "wrong": wrong}
            plans = _execute(base, config, route, module, cls, "incremental", record)
            saved, unavailable = {}, {}
            for table in parts:
                frame = output.load(table, where={"ticker": pins[table]})
                saved[table] = frame.copy()
                plan = plans[table]
                comparison = compare_producer_rows(expected[table], frame, table)
                year = int(config.validate.window_samples.year)
                observation = pd.to_datetime(expected[table].date).dt.year.eq(year)
                comparison["fixed_year"] = compare_producer_rows(
                    expected[table].loc[observation], frame.loc[pd.to_datetime(frame.date).dt.year.eq(year)], table
                )
                missing = sorted(set(pins[table]) - set(frame.loc[pd.to_datetime(frame.date).dt.year.eq(year), "ticker"]))
                for ticker in missing:
                    comparison["fixed_year"]["per_key"][ticker] = {
                        "rows": 0,
                        "finite_cells": 0,
                        "status": "abstain",
                        "reason": "Native full producer yielded no saved output for this original key",
                    }
                if missing:
                    unavailable[table.name] = missing
                boundary = expected[table].loc[plan.write.mask(expected[table].date)].date.min()
                comparison["first_rewrite_date"] = boundary
                comparison["first_rewrite"] = compare_producer_rows(
                    expected[table].loc[expected[table].date.eq(boundary)], frame.loc[frame.date.eq(boundary)], table
                )
                outside = expected[table].loc[~plan.write.mask(expected[table].date)]
                pd.testing.assert_frame_equal(
                    _ordered(outside), _ordered(frame.loc[~plan.write.mask(frame.date)]), check_dtype=False, check_exact=True
                )
                comparison["outside_rows_preserved"] = len(outside)
                comparison["overwrite_probe"] = corruptions[table]
                assert plan.write.start <= corruptions[table]["date"] <= plan.write.end, "Native write did not cover the overwritten probe"
                record["comparisons"][table.name] = comparison
            _execute(base, config, route, module, cls, "unchanged_rerun", record)
            for table in parts:
                pd.testing.assert_frame_equal(
                    _ordered(saved[table]), _ordered(output.load(table, where={"ticker": pins[table]})), check_dtype=False, check_exact=True
                )
                record["comparisons"][table.name]["unchanged_rerun_exact"] = True
            if unavailable:
                raise MissingProducerEvidenceError(f"Original fixed-year output keys unavailable from the native producer: {unavailable}")
        if peer_hash is not None and hashlib.sha256(peer.read_bytes()).hexdigest() != peer_hash:
            raise AssertionError("Readonly peer artifact changed during producer validation")
        if hashlib.sha256((run.baseline / "manifest.json").read_bytes()).hexdigest() != run.identity["source_manifest_sha256"]:
            raise AssertionError("Immutable source manifest changed during producer validation")
        if live_guard.TRIPS:
            raise AssertionError(f"Live-effect guard attempts: {len(live_guard.TRIPS)}")
        record.update(status="pass", database_sha256=hashlib.sha256(database.read_bytes()).hexdigest())
    except (MissingProducerEvidenceError, FileExistsError, FileNotFoundError) as caught:
        record.update(reason=str(caught))
    except Exception as caught:
        record.update(status="fail", error_type=type(caught).__name__, reason=str(caught))
        error = caught
    finally:
        record.update(elapsed_seconds=time.perf_counter() - started, tripwire_attempts=len(live_guard.TRIPS))
        if owned:
            _checkpoint(record, directory)
        result = CheckResult(
            "window_producer",
            case,
            record["status"],
            record.get("reason", "Actual native full/bounded creation and unchanged rerun compared"),
            metrics=record,
        )
        run.producers[case] = result.to_json()
        run.publish()
    if error is not None:
        raise error
    return result
