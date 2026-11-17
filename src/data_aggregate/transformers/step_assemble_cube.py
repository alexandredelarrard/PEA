"""Merge the pre-built cube parts onto the price grid and save the wide prediction-safe cube."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.assembly import part_has_rows
from src.data_aggregate.utils.common.frames import downcast_float32, normalize_date_col
from src.data_aggregate.utils.common.incremental import plan_window, write_part
from src.data_aggregate.utils.common.inputs import load_source
from src.data_aggregate.utils.common.panel_merge import PANEL_KEYS, PanelMerger
from src.data_aggregate.utils.common.part_status import cube_part_edge_report
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_store.schema import Table, Tables, resolve
from src.utils.data_policy import window_policy
from src.utils.step import Step
from src.utils.window import Window, WindowPlan

_CHUNK_ROWS = 400_000
_MIN_TARGET_COVERAGE_PCT = 50.0


class StepAssembleCube(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.data_aggregate

    def run(self, written: dict[Table, Window] | None = None) -> dict[Table, WindowPlan]:
        self._warn_on_part_edges()
        window = self._plan_window(written)
        if window is None:
            return {}
        cube, labels = self._assemble(window.read)
        self._persist(cube, window)
        self._report(cube, labels)
        return {Tables.cube: window}

    def _plan_window(self, written: dict[Table, Window] | None) -> WindowPlan | None:
        calendar = load_trading_calendar(self._context)
        assembly = self._cfg.assembly
        selected = {resolve(name) for name in [*assembly.compulsory_parts, *assembly.optional_parts]}
        starts = []
        for table in sorted(selected, key=lambda value: value.name):
            policy = window_policy(self._config, table, self._context.run.as_of)
            if policy is None:
                continue
            if table.name in assembly.optional_parts and not part_has_rows(self._store, table, {"ticker": self._universe.tickers}):
                continue
            if written is not None:
                if table not in written:
                    raise ValueError(f"{table.name}: successful upstream write receipt required before assembly")
                starts.append(written[table].start)
            else:
                component = plan_window(self._context, table, calendar)
                if component is not None:
                    starts.append(component.write.start)
        if not starts:
            return None
        return plan_window(self._context, Tables.cube, calendar, inherited_start=min(starts))

    def _assemble(self, window: Window) -> tuple[pd.DataFrame, list[str]]:
        assembly = self._cfg.assembly
        optional = set(assembly.optional_parts)
        selected = assembly.get("columns") or {}
        root = self._load_root(window)
        merger = PanelMerger(anchor=root[PANEL_KEYS])
        merger.add(root, Tables.cube_part_prices.name)
        labels: list[str] = []
        for name in [*assembly.compulsory_parts, *assembly.optional_parts]:
            table = resolve(name)
            if table == Tables.cube_part_prices:
                continue
            panel = self._read_part(table, selected.get(name), window, optional=name in optional)
            if panel is None:
                continue
            if panel.empty:
                panel = root[PANEL_KEYS].reindex(columns=panel.columns)
            panel = self._post_processing(panel)
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

    def _load_root(self, window: Window) -> pd.DataFrame:
        """The price grid (float64: raw prices stay exact) with the peer basket and GICS codes attached."""
        # The quantity-conversion inputs stay in the price part; the cube keeps the price fields it always carried.
        columns = [column for column in self._store.columns(Tables.cube_part_prices) if column not in {"security_basis", "volume_pit"}]
        root = load_source(self._context, Tables.cube_part_prices, columns=columns, window=window)
        assert root is not None
        root = normalize_date_col(root)
        assert root is not None
        root = root.sort_values(PANEL_KEYS, ignore_index=True)
        peers = load_peers_or_raise(self._context, self._config)
        peer_json = {ticker: json.dumps(peers.get(ticker, {}), ensure_ascii=False) for ticker in root["ticker"].unique()}
        root["peers"] = root["ticker"].map(peer_json)
        return self._universe.gics_codes(root)

    def _read_part(self, table: Table, columns: Sequence[str] | None, window: Window, *, optional: bool) -> pd.DataFrame | None:
        """One part projected on its configured columns (all when None/empty), float32 payload."""
        store = self._store
        policy = window_policy(self._config, table, self._context.run.as_of)
        if policy is None:
            return None
        retained = window.intersection(policy.requested)
        command = part_for(table).command
        if not part_has_rows(store, table, {"ticker": self._universe.tickers}):
            if optional:
                self._log.info("Skipping optional %s: absent or empty; refresh with %s", table.name, command)
                return None
            raise RuntimeError(f"{table.name} missing/empty; run {command}")

        actual = store.columns(table)
        wanted = [*PANEL_KEYS, *(col for col in (columns or actual) if col not in PANEL_KEYS)]
        missing = sorted(set(wanted) - set(actual))
        if missing:
            raise ValueError(f"{table.name} missing columns {missing}; run {command}")
        if retained is None:
            return pd.DataFrame(columns=wanted)
        chunks = [
            downcast_float32(chunk)
            for chunk in store.iter_load(
                table, columns=wanted, chunksize=_CHUNK_ROWS, where={"ticker": self._universe.tickers}, since=retained.start, until=retained.end
            )
        ]
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=wanted)

    def _post_processing(self, panel: pd.DataFrame) -> pd.DataFrame:
        """One part's panel after reading: dates normalized to midnight so merges across parts align."""
        normalized = downcast_float32(normalize_date_col(panel))
        assert normalized is not None
        return normalized

    def _persist(self, cube: pd.DataFrame, window: WindowPlan) -> int:
        store = self._store
        written = write_part(store, Tables.cube, cube, window)
        if set(store.columns(Tables.cube)) != set(cube.columns):
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
