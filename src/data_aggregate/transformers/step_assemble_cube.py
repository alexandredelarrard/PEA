"""Merge the pre-built cube parts onto the price grid and save the wide prediction-safe cube."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.assembly import part_has_rows
from src.data_aggregate.utils.common.frames import downcast_float32, normalize_date_col
from src.data_aggregate.utils.common.gics import apply_categorical_codes
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PartWindow, write_part
from src.data_aggregate.utils.common.panel_merge import PANEL_KEYS, PanelMerger
from src.data_aggregate.utils.common.part_status import cube_part_edge_report
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_store.schema import Table, Tables, resolve
from src.utils.step import Step

_CHUNK_ROWS = 200_000
_MIN_TARGET_COVERAGE_PCT = 50.0

# An existing cube is re-read and rewritten from this many calendar days before today (or its last date, if older).
_OVERLAP_DAYS = 385


class StepAssembleCube(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.build_cube

    def run(self, full: bool = False) -> None:
        self._warn_on_part_edges()
        window = PartWindow(None, None) if full else self._plan_window()
        cube, labels = self._assemble(window.since)
        if self._persist(cube, window) == COLUMNS_CHANGED:
            self._log.warning("%s column set changed -> full reassembly.", Tables.cube)
            window = PartWindow(None, None)
            cube, labels = self._assemble(None)
            self._persist(cube, window)
        self._report(cube, labels)

    def _plan_window(self) -> PartWindow:
        """Full when no cube is stored; else re-read from the earlier of its last date and today - overlap."""
        last = self._store.max_date(Tables.cube)
        if last is None:
            return PartWindow(None, None)
        start = min(last, pd.Timestamp.today().normalize() - pd.Timedelta(days=_OVERLAP_DAYS))
        return PartWindow(last, start, start)

    def _assemble(self, since: pd.Timestamp | None) -> tuple[pd.DataFrame, list[str]]:
        assembly = self._cfg.assembly
        optional = set(assembly.optional_parts)
        selected = assembly.get("columns") or {}
        root = self._load_root(since)
        merger = PanelMerger(anchor=root[PANEL_KEYS])
        merger.add(root, Tables.cube_part_prices.name)
        labels: list[str] = []
        for name in [*assembly.compulsory_parts, *assembly.optional_parts]:
            table = resolve(name)
            if table == Tables.cube_part_prices:
                continue
            panel = self._read_part(table, selected.get(name), since, optional=name in optional)
            if panel is None:
                continue
            if table == Tables.cube_part_targets:
                labels = [col for col in panel.columns if col not in PANEL_KEYS]
            merger.add(panel, name)
        return merger.to_long(), labels

    def _warn_on_part_edges(self) -> None:
        edge = cube_part_edge_report(self._context)
        for name in edge["misaligned"]:
            detail = edge["parts"][name]
            self._log.warning(
                "Cube part edge mismatch: part=%s status=%s part_max=%s reference_price=%s; run %s. cube-status remains the readiness gate.",
                name,
                detail["status"],
                detail["max_date"],
                edge["reference_price_date"],
                detail["refresh_command"],
            )

    def _load_root(self, since: pd.Timestamp | None) -> pd.DataFrame:
        """The price grid (float64: raw prices stay exact) with the peer basket and GICS codes attached."""
        root = self._store.load(Tables.cube_part_prices, since=since)
        assert root is not None
        root = normalize_date_col(root)
        assert root is not None
        root = root.sort_values(PANEL_KEYS, ignore_index=True)
        peers = load_peers_or_raise(self._context, self._config)
        peer_json = {ticker: json.dumps(peers.get(ticker, {}), ensure_ascii=False) for ticker in root["ticker"].unique()}
        root["peers"] = root["ticker"].map(peer_json)
        return apply_categorical_codes(root, self._context)

    def _read_part(self, table: Table, columns: Sequence[str] | None, since: pd.Timestamp | None, *, optional: bool) -> pd.DataFrame | None:
        """One part projected on its configured columns (all when None/empty), float32 payload."""
        store = self._store
        command = part_for(table).command
        if not part_has_rows(store, table):
            if optional:
                self._log.info("Skipping optional %s: absent or empty; refresh with %s", table.name, command)
                return None
            raise RuntimeError(f"{table.name} missing/empty; run {command}")

        actual = store.columns(table)
        wanted = [*PANEL_KEYS, *(col for col in (columns or actual) if col not in PANEL_KEYS)]
        missing = sorted(set(wanted) - set(actual))
        if missing:
            raise ValueError(f"{table.name} missing columns {missing}; run {command}")
        chunks = [downcast_float32(chunk) for chunk in store.iter_load(table, columns=wanted, chunksize=_CHUNK_ROWS, since=since)]
        return normalize_date_col(pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=wanted))

    def _persist(self, cube: pd.DataFrame, window: PartWindow) -> int:
        store = self._store
        if window.is_full and set(store.columns(Tables.cube)) - set(cube.columns):
            store.drop(Tables.cube)
        written = write_part(store, Tables.cube, cube, window)
        if written == COLUMNS_CHANGED:
            return written
        if set(store.columns(Tables.cube)) != set(cube.columns) or (window.is_full and store.row_count(Tables.cube) != len(cube)):
            raise RuntimeError("Stored cube schema/row count differs from the assembled frame; rerun assemble-cube")
        return written

    def _report(self, cube: pd.DataFrame, labels: list[str]) -> None:
        total = len(cube)
        matched = int(cube[labels].notna().any(axis=1).sum()) if labels else 0
        coverage = 100 * matched / total if total else 0.0
        if coverage < _MIN_TARGET_COVERAGE_PCT:
            self._log.warning(
                "Only %.1f%% of cube rows carry any target (%s/%s); inspect target joins/maturity. Unlabelled prediction rows are retained.",
                coverage,
                matched,
                total,
            )
        self._log.info(
            "Assembled %s: %s price rows x %s columns, %s targets, %.1f%% labelled", Tables.cube, total, cube.shape[1], len(labels), coverage
        )
