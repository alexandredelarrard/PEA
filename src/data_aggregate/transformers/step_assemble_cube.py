"""Project registered parts onto price keys and stream a wide prediction-safe cube."""

from __future__ import annotations

import gc
import json

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import PANEL_KEYS
from src.context import Context
from src.data_aggregate.utils.common.assembly import AssemblyPolicy, part_has_rows
from src.data_aggregate.utils.common.frames import downcast_float32
from src.data_aggregate.utils.common.gics import GICS_COLUMNS, apply_categorical_codes
from src.data_aggregate.utils.common.panel_merge import FeatureCollisionError, PanelMerger
from src.data_aggregate.utils.common.part_status import cube_part_edge_report
from src.data_aggregate.utils.common.parts import CUBE_PARTS, PART_BY_NAME, CubePart
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_store.schema import Tables
from src.utils.step import Step

_CHUNK_ROWS = 200_000
_MIN_TARGET_COVERAGE_PCT = 50.0


class StepAssembleCube(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

    def run(self) -> None:
        policy = AssemblyPolicy.from_config(self._cfg)
        self._warn_on_part_edges()
        prices = self._read_part(PART_BY_NAME[Tables.cube_part_prices.name], policy)
        if prices is None:
            raise RuntimeError(f"{Tables.cube_part_prices} is required as the price grid; run build-prices")
        price_index = pd.MultiIndex.from_frame(prices[PANEL_KEYS])
        merger = PanelMerger(self._log, anchor=prices[PANEL_KEYS])
        owners: dict[str, str] = {name: "metadata" for name in ("peers", *GICS_COLUMNS)}
        base_columns: list[str] = []
        targets = pd.DataFrame(index=price_index[:0])
        for part in CUBE_PARTS:
            panel = prices if part.table == Tables.cube_part_prices else self._read_part(part, policy, prices[PANEL_KEYS])
            if panel is None:
                continue
            payload = [col for col in panel.columns if col not in PANEL_KEYS]
            collisions = set(payload) & owners.keys()
            if collisions:
                raise FeatureCollisionError(f"{part.name} collides on {sorted(collisions)} with { {col: owners[col] for col in sorted(collisions)} }")
            owners.update(dict.fromkeys(payload, part.name))
            if part.table == Tables.cube_part_targets:
                targets = panel.set_index(PANEL_KEYS)
            else:
                merger.add(panel, part.name)
                base_columns.extend(payload)
        base = merger.to_long().set_index(PANEL_KEYS).reindex(index=price_index, columns=base_columns).sort_index()
        for column in set(base_columns) - set(merger.feature_columns):
            base[column] = pd.Series(index=base.index, dtype="float32")
        peers = load_peers_or_raise(self._context, self._config)
        peer_json = {ticker: json.dumps(peers.get(ticker, {}), ensure_ascii=False) for ticker in prices["ticker"].unique()}
        base = base.reset_index()
        base["peers"] = base["ticker"].map(peer_json)
        base = apply_categorical_codes(base, self._context, self._log).set_index(PANEL_KEYS)
        del prices, merger
        self._stream_cube(base, targets)

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

    def _validate_keys(self, frame: pd.DataFrame, part: CubePart) -> None:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dt.normalize()
        ticker = frame["ticker"].str.strip()
        if frame["date"].isna().any() or ticker.isna().any() or ticker.eq("").any() or ticker.ne(frame["ticker"]).any():
            raise ValueError(f"{part.name} has null or invalid date/ticker keys; run {part.command}")
        duplicates = int(frame.duplicated(PANEL_KEYS).sum())
        if duplicates:
            raise RuntimeError(f"{part.name} has {duplicates} duplicate (date,ticker) rows; run {part.command}")

    def _read_part(self, part: CubePart, policy: AssemblyPolicy, price_keys: pd.DataFrame | None = None) -> pd.DataFrame | None:
        store = self._context.store
        if not part_has_rows(store, part):
            if part.name in policy.optional:
                self._log.info("Skipping optional %s: absent or empty; refresh with %s", part.name, part.command)
                return None
            raise RuntimeError(f"{part.name} missing/empty; run {part.command}")
        actual = store.columns(part.table)
        if part.table == Tables.cube_part_targets and "target_horizon" in actual:
            raise RuntimeError(f"{part.name} still carries target_horizon; run build-target --full to rewrite it wide")
        columns = policy.projection(part, actual)
        where: dict[str, object] | None = None
        since, until = None, None
        if price_keys is not None:
            # Check all keys before scoping payload, so null/off-grid keys cannot hide corruption.
            key_chunks = list(store.iter_load(part.table, columns=PANEL_KEYS, chunksize=_CHUNK_ROWS))
            keys = pd.concat(key_chunks, ignore_index=True) if key_chunks else pd.DataFrame(columns=PANEL_KEYS)
            self._validate_keys(keys, part)
            where = {"ticker": price_keys["ticker"].unique().tolist()}
            since, until = price_keys["date"].min(), price_keys["date"].max()
            del keys, key_chunks
        chunks = [
            chunk if part.kind == "prices" else downcast_float32(chunk)
            for chunk in store.iter_load(part.table, columns=columns, chunksize=_CHUNK_ROWS, where=where, since=since, until=until)
        ]
        panel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
        self._validate_keys(panel, part)
        if price_keys is not None:
            index = pd.MultiIndex.from_frame(panel[PANEL_KEYS])
            panel = panel.loc[index.isin(pd.MultiIndex.from_frame(price_keys))]
        if panel.empty and part.name not in policy.optional:
            raise RuntimeError(f"{part.name} has no rows on the price grid; run {part.command}")
        return panel

    def _stream_cube(self, base: pd.DataFrame, targets: pd.DataFrame) -> None:
        store = self._context.store
        total, matched = 0, 0
        labels = list(targets.columns)
        expected = set(PANEL_KEYS) | set(base.columns) | set(labels)
        for start in range(0, len(base), _CHUNK_ROWS):
            chunk = base.iloc[start : start + _CHUNK_ROWS].join(targets, how="left", validate="one_to_one").reset_index()
            matched += int(chunk[labels].notna().any(axis=1).sum())
            if start == 0:
                if set(store.columns(Tables.cube)) - expected:
                    store.drop(Tables.cube)
                store.replace(Tables.cube, chunk)
            else:
                store.bulk_seed(Tables.cube, chunk)
            total += len(chunk)
            del chunk
            gc.collect()
        if set(store.columns(Tables.cube)) != expected or store.row_count(Tables.cube) != len(base):
            raise RuntimeError("Stored cube schema/row count differs from the verified price-grid output; rerun assemble-cube")
        coverage = 100 * matched / total if total else 0.0
        if coverage < _MIN_TARGET_COVERAGE_PCT:
            self._log.warning(
                "Only %.1f%% of cube rows carry any target (%s/%s); inspect target joins/maturity. Unlabelled prediction rows are retained.",
                coverage,
                matched,
                total,
            )
        self._log.info("Saved %s: %s price rows x %s columns, %s targets, %.1f%% labelled", Tables.cube, total, len(expected), len(labels), coverage)
