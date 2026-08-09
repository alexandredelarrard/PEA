"""
step_assemble_cube.py  (src/data_aggregate/transformers/step_assemble_cube.py)
-------------------------------------------------------------------------
The final step: read every persisted part, merge features + composites + betas + peers +
targets into the `cube` table, and save it. Loads NO raw source tables and recomputes NO
features.

MEMORY-LIGHT, and this is the step that used to OOM-kill the DAG. The cube is LONG by
`target_horizon`, so a single `targets.merge(base)` broadcasts every feature column across
all horizons at once -- dates x tickers x horizons x ~570 columns held in RAM, then
serialized in one shot. Instead:

  1. float32 every feature part as it is read,
  2. build the wide horizon-INDEPENDENT `base` ONCE (features + composites + betas + peers +
     GICS codes),
  3. STREAM the write one target_horizon at a time, in bounded row chunks.

Peak memory drops by the horizon factor and there is no giant final serialization spike.

TWO THINGS THIS STEP MUST KEEP DOING:
  * store-facade only. It uses `exists` / `load` / `replace` / `bulk_seed` and never touches
    `store.engine`, so it stays testable against a fake store -- `PartStore` is for the
    part-BUILDING steps. `tests/data_aggregate/test_assemble_cube.py` enforces this.
  * `replace` for the first chunk (clears the table + creates the schema), then `bulk_seed`
    (chunked COPY-append). NOT the slow unchunked upsert -- that was the horizon-2 OOM.

The cross-part merge now runs through `PanelMerger`, which closes a real hole: it used to be
a bare `how="outer"` merge, so a feature name owned by two PARTS silently became `_x`/`_y`
with no error. Six coarse parts make that likelier than fourteen fine ones did.
"""
from __future__ import annotations

import gc
import json

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.constants.constants import CUBE_PART_BETAS, CUBE_PART_TARGETS, CUBE_TABLE
from src.context import Context
from src.data_aggregate.utils.assemble.composites import build_composites
from src.data_aggregate.utils.common.gics import apply_categorical_codes
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.part_io import downcast_float32, normalize_date_col
from src.data_aggregate.utils.common.parts import FEATURE_PARTS
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.utils.step import Step

_CHUNK_ROWS = 200_000


class StepAssembleCube(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

    def run(self) -> None:
        peers = load_peers_or_raise(self._context, self._config)
        panel = self._merge_feature_parts()
        panel = self._add_composites(panel)
        base = self._build_base(panel, peers)
        del panel
        self._stream_cube(base)

    # ---- read the parts ---- #
    def _read_part(self, name: str) -> pd.DataFrame | None:
        if not self._context.store.exists(name):
            # NOT cosmetic: `store.replace` does not DROP the cube table, it DELETEs the rows
            # and COPYs into the existing schema (and `ensure_columns` only ever ADDS). So a
            # missing part leaves its columns in place, entirely NULL, and the cube's column
            # set still looks perfect -- observed here as 574 == 574 columns with 0 added / 0
            # removed while every f_inst_*/f_super_*/f_insider_*/f_ceo_* value was NULL.
            self._log.warning("Feature part '%s' is MISSING -> its features will be ALL-NULL in "
                              "the cube (the column set will still look unchanged). Run its "
                              "build step, then re-run assemble-cube.", name)
            return None
        df = downcast_float32(normalize_date_col(self._context.store.load(name)))
        return None if df is None or df.empty else df

    def _merge_feature_parts(self) -> pd.DataFrame:
        """Outer-align every feature part on (date, ticker), through the collision guard."""
        merger = PanelMerger(self._log)
        merged = 0
        for part in FEATURE_PARTS:
            merged += 1 if merger.add(self._read_part(part.name), part.name) else 0
        panel = merger.to_long()
        if panel.empty or len(panel.columns) <= 2:
            raise RuntimeError("No feature parts found -> run the build-* feature steps first.")
        # report what was ACTUALLY merged, not how many parts are registered: a missing part
        # silently shrinks the cube's feature set, so the two numbers must not be conflated
        if merged < len(FEATURE_PARTS):
            self._log.warning("Only %d of %d registered feature parts were merged -> the cube "
                              "is missing the others' features.", merged, len(FEATURE_PARTS))
        self._log.info("Merged %d/%d feature parts -> %s rows x %s feature columns",
                       merged, len(FEATURE_PARTS), len(panel), len(panel.columns) - 2)
        return panel

    # ---- composites ---- #
    def _add_composites(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Append thematic `comp_<theme>` columns: each theme averages its (sign-oriented,
        re-standardized) member features. ADDITIVE -- raw features are kept, so no
        information is lost. Runs HERE, not in a feature step, because a composite averages
        members from SEVERAL parts and can only be formed once they all exist."""
        cfg = self._cfg.get("composites", {}) or {}
        if not cfg.get("enabled", False):
            return panel
        groups = OmegaConf.to_container(cfg.get("groups", {}), resolve=True) or {}
        if not groups:
            self._log.warning("composites.enabled but no groups configured.")
            return panel
        cols_before = set(panel.columns)
        panel = build_composites(panel, groups, method=cfg.get("method", "zscore"),
                                 log=self._log)   # warns about configured members absent
        # report the composites actually BUILT, not the configured themes: a theme whose
        # members all came from a missing part is silently skipped, and printing the config
        # list would claim it exists
        built = sorted(c for c in set(panel.columns) - cols_before if c.startswith("comp_"))
        skipped = sorted(f"comp_{t}" for t in groups if f"comp_{t}" not in built)
        self._log.info("Built %s/%s composite signals: %s", len(built), len(groups), built)
        if skipped:
            self._log.warning("%s composite(s) NOT built (members absent from the panel): %s",
                              len(skipped), skipped)
        return panel

    # ---- the horizon-independent base ---- #
    def _build_base(self, panel: pd.DataFrame, peers: dict) -> pd.DataFrame:
        betas = self._read_part(CUBE_PART_BETAS)
        base = panel if betas is None else panel.merge(betas, on=["date", "ticker"], how="left")
        if betas is None:
            self._log.warning("%s missing -> the cube will carry no beta columns.",
                              CUBE_PART_BETAS)
        del betas
        # peers JSON PRECOMPUTED PER TICKER (a few hundred unique strings shared across
        # every row). The old per-row json.dumps built millions of DISTINCT strings -- a
        # large object-column memory hog, broadcast again into each horizon.
        peer_json = {t: json.dumps(peers.get(t, {}), ensure_ascii=False)
                     for t in base["ticker"].unique()}
        base["peers"] = base["ticker"].map(peer_json)
        base = apply_categorical_codes(base, self._context, self._log)
        # index once -> a fast per-slice join below
        return base.set_index(["date", "ticker"]).sort_index()

    # ---- stream the write ---- #
    def _load_targets(self) -> pd.DataFrame:
        targets = self._read_part(CUBE_PART_TARGETS)
        if targets is None:
            raise RuntimeError(f"{CUBE_PART_TARGETS} missing/empty -> run `build-target` first.")
        return targets

    def _stream_cube(self, base: pd.DataFrame) -> None:
        targets = self._load_targets()
        horizons = sorted(pd.to_numeric(targets["target_horizon"], errors="coerce")
                          .dropna().unique().tolist())
        total, first = 0, True
        for h in horizons:
            tg = targets[targets["target_horizon"] == h].set_index(["date", "ticker"])
            for j in range(0, len(tg), _CHUNK_ROWS):
                chunk = tg.iloc[j:j + _CHUNK_ROWS].join(base, how="inner").reset_index()
                if chunk.empty:
                    continue
                if first:
                    self._context.store.replace(CUBE_TABLE, chunk)   # clears + creates the schema
                    first = False
                else:
                    self._context.store.bulk_seed(CUBE_TABLE, chunk)  # chunked COPY-append
                total += len(chunk)
                chunk = None
                gc.collect()          # hand the arrays + COPY buffer back before the next
            self._log.info("Cube horizon %s streamed (running total %s rows).", h, total)
            tg = None
            gc.collect()
        self._log.info("Saved cube to DB table '%s' (%s rows across %d horizons)",
                       CUBE_TABLE, total, len(horizons))
