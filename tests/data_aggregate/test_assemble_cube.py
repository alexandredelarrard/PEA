"""
Cube assembly (`StepAssembleCube.run`) — the memory-light chunked write of
`base LEFT JOIN targets_wide`. A FakeStore serves synthetic feature/beta/target parts and
captures the writes, proving:
  * the streamed cube (concatenation of the chunk writes) equals a one-shot
    `base.merge(targets, how="left")` — same rows, same feature values (nothing lost by
    streaming),
  * the write ORDER is `replace` for the first chunk then `bulk_seed` for every later one,
  * the grain is one row per (date, ticker) — the per-horizon row duplication is GONE,
  * a base row with no matching target row SURVIVES with NaN labels (the immature-date fix),
  * a targets part that is still long, or duplicated, is REFUSED rather than silently
    multiplying the cube,
  * feature columns are stored float32 (half the footprint).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.utils.common.parts import FEATURE_PARTS
from src.data_store.schema import name_of


class _FakeStore:
    """A write-ORDER spy, which is what this test is about: the first chunk must go through
    `replace` (clears + creates the schema) and every later chunk through `bulk_seed`. A real
    store would not expose the call sequence. Tables are keyed by `name_of` so it accepts both a
    `Table` and a bare name."""

    def __init__(self, tables: dict):
        self.t = {name_of(k): v for k, v in tables.items()}
        self.writes: list[tuple[str, pd.DataFrame]] = []   # (op, df) in call order

    def exists(self, table): return name_of(table) in self.t

    def load(self, table, columns=None, **kw):
        df = self.t.get(name_of(table))
        return df.copy() if df is not None else pd.DataFrame()

    def _append(self, op, table, df):
        self.writes.append((op, df.copy()))
        prev = self.t.get(name_of(table))
        self.t[name_of(table)] = (pd.concat([prev, df], ignore_index=True)
                                  if prev is not None else df.copy())
        return len(df)

    def replace(self, table, df, chunksize=200_000):
        self.writes.append(("replace", df.copy()))
        self.t[name_of(table)] = df.copy()
        return len(df)

    def save(self, table, df, pk=None): return self._append("save", table, df)

    def bulk_seed(self, table, df): return self._append("bulk_seed", table, df)


class _FakeCtx:
    def __init__(self, store):
        self.store = store
        self.log = logging.getLogger("test")
        self.paths = {"SECTOR_PEERS_PATH": Path("/nonexistent/peers.json")}


_DATES = ["2023-01-02", "2023-01-03"]
_TICKERS = ["AAA", "BBB"]
_GRID = [(d, t) for d in _DATES for t in _TICKERS]
_TARGET_COLS = ["target_fwd_ret_h5", "target_fwd_ret_h20"]


def _targets_wide(drop: tuple[str, str] | None = None) -> pd.DataFrame:
    """The WIDE targets part: one row per (date, ticker), one column per (label, horizon).

    `drop` removes one key entirely, which is what an IMMATURE date looks like coming out of
    `labels_to_wide` — nothing is known for it at any horizon yet, so it stores no row."""
    y = [0.01, 0.02, 0.03, 0.04]
    df = pd.DataFrame(_GRID, columns=["date", "ticker"]).assign(
        target_fwd_ret_h5=np.array([v * 5 for v in y], dtype="float64"),
        target_fwd_ret_h20=np.array([v * 20 for v in y], dtype="float64"))
    return df if drop is None else df[~((df["date"] == drop[0]) & (df["ticker"] == drop[1]))]


def _parts(targets: pd.DataFrame | None = None):
    price = pd.DataFrame(_GRID, columns=["date", "ticker"]).assign(
        f_ret=np.array([0.1, 0.2, 0.3, 0.4], dtype="float64"))
    fund = pd.DataFrame(_GRID, columns=["date", "ticker"]).assign(
        f_val=np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"))
    betas = pd.DataFrame(_GRID, columns=["date", "ticker"]).assign(
        beta_mkt=np.array([0.9, 1.1, 1.0, 1.2], dtype="float64"))
    momentum_part, fundamentals_part = FEATURE_PARTS[0].name, FEATURE_PARTS[1].name
    return {momentum_part: price, fundamentals_part: fund,
            "cube_part_betas": betas,
            "cube_part_targets": _targets_wide() if targets is None else targets,
            "sp500_tickers": pd.DataFrame(columns=["ticker", "sector", "industry_group"])}


def _make_step(store, monkeypatch, chunk_rows: int | None = None):
    step = StepAssembleCube.__new__(StepAssembleCube)     # skip heavy __init__
    step._context = _FakeCtx(store)
    step._config = None
    step._log = logging.getLogger("test")
    step._cfg = {}
    # the peer dict is read through utils/common/peers_io, so stub that rather than an attribute
    monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube.load_peers_or_raise",
                        lambda ctx, cfg=None: {"AAA": {"BBB": 1.0}, "BBB": {"AAA": 1.0}})
    if chunk_rows is not None:
        # 4 fixture rows fit in ONE 200k chunk, so the replace -> bulk_seed ordering would
        # never be exercised at the production size. Shrink the chunk rather than weaken the
        # assertion: that ordering is load-bearing (only `replace` creates the schema).
        monkeypatch.setattr("src.data_aggregate.transformers.step_assemble_cube._CHUNK_ROWS",
                            chunk_rows)
    return step


def test_assemble_streams_chunks_and_matches_oneshot(monkeypatch):
    store = _FakeStore(_parts())
    step = _make_step(store, monkeypatch, chunk_rows=2)    # 4 rows -> 2 chunks

    step.run()

    # ---- streaming shape: chunked COPY (replace first, bulk_seed append) ---------------------
    ops = [op for op, _ in store.writes]
    assert ops == ["replace", "bulk_seed"], f"expected chunked COPY streaming, got {ops}"
    assert all(len(df) <= 2 for _, df in store.writes)     # every write is a bounded row-chunk
    cube = store.t["cube"]

    # ---- correctness: equals a one-shot base LEFT JOIN targets --------------------------------
    p = _parts()
    base_ref = (p[FEATURE_PARTS[0].name].merge(p[FEATURE_PARTS[1].name], on=["date", "ticker"])
                .merge(p["cube_part_betas"], on=["date", "ticker"]))
    ref = base_ref.merge(p["cube_part_targets"], on=["date", "ticker"], how="left",
                         validate="one_to_one")
    # 2 dates x 2 tickers. NOT 8: the horizon axis is columns now, so the row duplication the
    # wide part removes IS the number being asserted.
    assert len(cube) == len(ref) == 4, f"rows {len(cube)} vs {len(ref)}"
    assert not cube.duplicated(["date", "ticker"]).any(), "cube grain is not (date, ticker)"
    assert "target_horizon" not in cube.columns

    key = ["ticker", "date"]
    got = cube.set_index(key).sort_index()
    exp = ref.assign(date=pd.to_datetime(ref["date"]).dt.normalize()).set_index(key).sort_index()
    got.index = got.index.set_levels(pd.to_datetime(got.index.levels[1]).normalize(), level=1)
    for col in ("f_ret", "f_val", "beta_mkt", *_TARGET_COLS):
        assert np.allclose(got[col].to_numpy(float), exp[col].to_numpy(float)), f"{col} mismatch"
    assert "peers" in cube.columns and cube["peers"].notna().all()

    # ---- memory: feature AND label columns are float32 ---------------------------------------
    checked = ("f_ret", "f_val", "beta_mkt", *_TARGET_COLS)
    f32 = [c for c in checked if cube[c].dtype == np.float32]
    assert set(f32) == set(checked), {c: str(cube[c].dtype) for c in checked}

    print("\n=== SANITY CHECK: cube assembly (chunked base LEFT JOIN wide targets) ===")
    print(f"  wrote {len(store.writes)} bounded chunks as {ops} (replace creates the schema, "
          f"bulk_seed appends)")
    print(f"  cube {len(cube)} rows = {len(_DATES)} dates x {len(_TICKERS)} tickers, 0 duplicate "
          f"(date,ticker), no target_horizon column -- the 2-horizon row duplication is gone "
          f"(it was 8 rows)")
    print(f"  values equal the one-shot LEFT JOIN; {len(_TARGET_COLS)} label columns float32. "
          f"Validated.")


def test_base_row_with_no_target_survives_with_nan_labels(monkeypatch):
    """THE immature-date fix. A (date, ticker) the targets part has no row for must keep its
    features and carry NaN labels -- the old `targets.join(base, how="inner")` deleted exactly
    the newest ~max_horizon dates, which are the ones `predict_latest` exists to score."""
    missing = (_DATES[-1], "BBB")
    store = _FakeStore(_parts(targets=_targets_wide(drop=missing)))
    step = _make_step(store, monkeypatch, chunk_rows=2)

    step.run()
    cube = store.t["cube"]

    assert len(cube) == 4, f"the unlabelled row was dropped: {len(cube)} rows"
    hit = (cube["date"] == pd.Timestamp(missing[0])) & (cube["ticker"] == missing[1])
    row = cube[hit]
    assert len(row) == 1
    assert row[_TARGET_COLS].isna().all(axis=None), "labels should be NaN, not filled"
    assert row["f_ret"].notna().all(), "its FEATURES must still be there -- that is the point"
    assert cube[~hit][_TARGET_COLS].notna().all(axis=None), "the labelled rows lost their labels"

    print("\n=== SANITY CHECK: immature (date, ticker) survives the join ===")
    print(f"  targets part holds 3 of 4 keys; cube still has {len(cube)} rows. "
          f"{missing} carries f_ret={float(row['f_ret'].iloc[0]):.2f} with both labels NaN. "
          f"An inner join would have returned 3 rows and hidden the newest date. Validated.")


def test_long_targets_part_is_refused(monkeypatch):
    """A part left over from the LONG era would merge to a horizon-duplicated cube. It must
    raise, naming the part and the fix, rather than be silently broadcast."""
    long_rows = [{"date": d, "ticker": t, "target_horizon": h, "target_fwd_ret": 0.01 * h}
                 for h in (5, 20) for (d, t) in _GRID]
    store = _FakeStore(_parts(targets=pd.DataFrame(long_rows)))
    step = _make_step(store, monkeypatch)

    with pytest.raises(RuntimeError, match="cube_part_targets.*target_horizon"):
        step.run()
    assert not store.writes, "nothing must be written when the part is refused"

    print("\n=== SANITY CHECK: the old LONG targets part is refused ===")
    print("  a part still carrying `target_horizon` raises before any write, naming "
          "cube_part_targets and `build-target --full`. Validated.")


def test_duplicate_target_keys_are_refused(monkeypatch):
    """A wide part that nonetheless repeats a (date, ticker) would double the cube on the
    join. `_load_targets` counts the duplicates and names the part."""
    dup = pd.concat([_targets_wide(), _targets_wide().head(1)], ignore_index=True)
    store = _FakeStore(_parts(targets=dup))
    step = _make_step(store, monkeypatch)

    with pytest.raises(RuntimeError, match=r"cube_part_targets has 1 duplicate"):
        step.run()
    assert not store.writes, "nothing must be written when the part is refused"

    print("\n=== SANITY CHECK: duplicate (date,ticker) in the targets part is refused ===")
    print("  1 repeated key -> RuntimeError naming cube_part_targets and the duplicate count, "
          "before any write. The cube cannot silently double. Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
