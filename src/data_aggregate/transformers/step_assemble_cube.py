"""
step_assemble_cube.py  (src/data_aggregate/transformers/step_assemble_cube.py)
-------------------------------------------------------------------------
The final step: read every persisted part, merge features + betas + peers + targets into
the `cube` table, and save it. Loads NO raw source tables and recomputes NO features.

THE JOIN, AND WHICH SIDE LEADS:

    base = features (PanelMerger) + betas + peers + GICS codes      grain (date, ticker)
    cube = base  LEFT JOIN  targets_wide  on (date, ticker)         grain (date, ticker)

LEFT FROM THE FEATURES, not inner from the targets. A label at date d needs prices through
d + horizon, so the newest ~max_horizon trading days have no matured label -- an inner join
DELETES exactly the dates `predict_latest` exists to score. They now survive with NaN labels;
training rows are unchanged because `panel_from_cube` still drops null-label rows. The cube's
row count therefore equals the feature panel's, which is a checkable invariant against
`cube_part_prices`.

MEMORY-LIGHT, and this is the step that used to OOM-kill the DAG. The cube was LONG by
`target_horizon`, so a single `targets.merge(base)` broadcast every feature column across all
horizons at once -- dates x tickers x horizons x ~570 columns held in RAM, then serialized in
one shot. The horizon factor is gone from the SHAPE (targets are wide, ~9 label columns on the
same grain), but `base` is still **3.83M rows** x ~570 float32 columns, so the streaming stays.
⚠ The row count is the panel's, NOT the targets part's, and it does not shrink with the wide
pivot: measured 3,830,311 in `cube_part_prices` against 3,211,139 in the wide targets part.
Sizing this step off the targets part is how it came to be under-budgeted:

  1. float32 every feature part as it is read,
  2. build `base` ONCE (features + betas + peers + GICS codes), indexed and sorted,
  3. hold the small wide targets whole and JOIN THEM INTO bounded row chunks of `base`.

The bounded COPY is what keeps the write off one giant serialization -- that reason is
independent of the grain change.

TWO THINGS THIS STEP MUST KEEP DOING:
  * store-facade only: `exists` / `load` / `replace` / `bulk_seed`, never raw SQL. Now enforced
    repo-wide by `tests/data_store/test_store_boundary.py`, not just for this step.
  * `replace` for the first chunk (clears the table + creates the schema), then `bulk_seed`
    (chunked COPY-append). NOT the slow unchunked upsert -- that was the horizon-2 OOM.

The cross-part merge now runs through `PanelMerger`, which closes a real hole: it used to be
a bare `how="outer"` merge, so a feature name owned by two PARTS silently became `_x`/`_y`
with no error. Six coarse parts make that likelier than fourteen fine ones did.

EVERY COMBINE HERE IS ONE-TO-ONE, and each states it: `PanelMerger.add` raises on a duplicate
key (`concat(axis=1)` takes no `validate=`), the betas merge passes `validate="one_to_one"`,
and `_load_targets` asserts the targets part's grain before it is used as a join side.
"""
from __future__ import annotations

import gc
import json

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.context import Context
from src.data_aggregate.utils.common.gics import apply_categorical_codes
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.frames import downcast_float32, normalize_date_col
from src.data_aggregate.utils.common.parts import FEATURE_PARTS
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.utils.step import Step

_CHUNK_ROWS = 200_000
# below this, the targets join matched so little that it is a bug, not immature labels: only
# the newest ~max_horizon dates are legitimately label-free, a few % of the panel
_MIN_TARGET_COVERAGE_PCT = 50.0


class StepAssembleCube(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

    def run(self) -> None:
        peers = load_peers_or_raise(self._context, self._config)
        base = self._build_base(self._merge_feature_parts(), peers)
        self._stream_cube(base)

    # ---- read the parts ---- #
    def _read_part(self, name: str) -> pd.DataFrame | None:
        if not self._context.store.exists(name):
            # NOT cosmetic: `store.replace` does not DROP the cube table, it DELETEs the rows
            # and COPYs into the existing schema (and `ensure_columns` only ever ADDS). So a
            # missing part leaves its columns in place, entirely NULL, and the cube's column
            # set still looks perfect -- observed here as 574 == 574 columns with 0 added / 0
            # removed while every f_ic_inst_*/f_ic_super_*/f_ic_insider_*/f_ceo_* value was NULL.
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

    # ---- the (date, ticker) base ---- #
    def _build_base(self, panel: pd.DataFrame, peers: dict) -> pd.DataFrame:
        betas = self._read_part(Tables.cube_part_betas)
        base = panel if betas is None else panel.merge(
            betas, on=["date", "ticker"], how="left", validate="one_to_one")
        if betas is None:
            self._log.warning("%s missing -> the cube will carry no beta columns.",
                              Tables.cube_part_betas)
        del betas
        # peers JSON PRECOMPUTED PER TICKER (a few hundred unique strings shared across
        # every row). The old per-row json.dumps built millions of DISTINCT strings -- a
        # large object-column memory hog.
        peer_json = {t: json.dumps(peers.get(t, {}), ensure_ascii=False)
                     for t in base["ticker"].unique()}
        base["peers"] = base["ticker"].map(peer_json)
        base = apply_categorical_codes(base, self._context, self._log)
        # index once -> a fast per-slice join below
        return base.set_index(["date", "ticker"]).sort_index()

    # ---- stream the write ---- #
    def _load_targets(self) -> pd.DataFrame:
        """The wide targets part, with its grain asserted BEFORE it becomes a join side.

        The `target_horizon` check is the one that matters operationally: a part left over
        from the long format would merge to a 3x-duplicated cube, and the only thing to
        notice would be a `validate=` failure several hundred columns later."""
        targets = self._read_part(Tables.cube_part_targets)
        if targets is None:
            raise RuntimeError(f"{Tables.cube_part_targets} missing/empty -> run `build-target` first.")
        if "target_horizon" in targets.columns:
            raise RuntimeError(f"{Tables.cube_part_targets} still carries `target_horizon` -> it is "
                               f"the OLD LONG part. Re-run `build-target --full` to rewrite it wide.")
        dup = int(targets.duplicated(["date", "ticker"]).sum())
        if dup:
            raise RuntimeError(f"{Tables.cube_part_targets} has {dup} duplicate (date,ticker) rows "
                               f"-- it must be WIDE by horizon. Re-run `build-target --full`.")
        return targets

    def _stream_cube(self, base: pd.DataFrame) -> None:
        """Write the cube as `base LEFT JOIN targets_wide`, in bounded row chunks.

        `targets` is ~9 label columns on the cube's own grain, so it is the side held whole
        and joined INTO each block; `base` is the ~570-column side and the reason the write
        is still chunked. Both sides are duplicate-checked (`PanelMerger.add`, `_load_targets`)
        and `targets` is indexed uniquely, so the block join is one-to-one by construction."""
        targets = self._load_targets().set_index(["date", "ticker"])
        total, matched, first = 0, 0, True
        label_cols = list(targets.columns)
        for j in range(0, len(base), _CHUNK_ROWS):
            chunk = base.iloc[j:j + _CHUNK_ROWS].merge(
                targets, left_index=True, right_index=True, how="left",
                validate="one_to_one").reset_index()
            if chunk.empty:
                continue
            matched += int(chunk[label_cols].notna().any(axis=1).sum())
            if first:
                self._context.store.replace(Tables.cube, chunk)   # clears + creates the schema
                first = False
            else:
                self._context.store.bulk_seed(Tables.cube, chunk)  # chunked COPY-append
            total += len(chunk)
            chunk = None
            gc.collect()              # hand the arrays + COPY buffer back before the next
        # the one number that surfaces a silently-failed join: a date dtype or ticker-case
        # mismatch leaves the labels entirely NaN and every other check still passes
        cov = 100 * matched / total if total else 0.0
        if cov < _MIN_TARGET_COVERAGE_PCT:
            self._log.warning("Only %.1f%% of cube rows carry ANY target label (%s/%s) -> the "
                              "targets join matched almost nothing. Check the (date, ticker) "
                              "dtypes in %s.", cov, matched, total, Tables.cube_part_targets)
        self._log.info("Saved cube to DB table '%s' (%s rows x %s columns, %s target columns, "
                       "%.1f%% label coverage)", Tables.cube, total,
                       len(base.columns) + len(label_cols) + len(base.index.names),
                       len(label_cols), cov)
