"""
Cube assembly (`StepBuildCube.assemble_cube_from_parts`) — the memory-light, per-horizon
streaming write. A FakeStore serves synthetic feature/beta/target parts and captures the
writes, proving:
  * the streamed cube (concatenation of the per-horizon writes) equals a one-shot
    `targets.merge(base)` — same rows, same feature values (no data lost by streaming),
  * only ONE horizon slice is written at a time (replace for the first, append for the rest)
    — the whole horizon-expanded cube is never materialised, which was the OOM,
  * feature columns are stored float32 (half the footprint).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.step_build_cube import StepBuildCube


class _FakeStore:
    def __init__(self, tables: dict):
        self.t = dict(tables)
        self.writes: list[tuple[str, pd.DataFrame]] = []   # (op, df) in call order

    def exists(self, name): return name in self.t
    def load(self, name, columns=None):
        df = self.t.get(name)
        return df.copy() if df is not None else pd.DataFrame()

    def replace(self, name, df, chunksize=200_000):
        self.writes.append(("replace", df.copy()))
        self.t[name] = df.copy()
        return len(df)

    def save(self, name, df, pk=None):
        self.writes.append(("save", df.copy()))
        prev = self.t.get(name)
        self.t[name] = pd.concat([prev, df], ignore_index=True) if prev is not None else df.copy()
        return len(df)

    def bulk_seed(self, name, df):                        # chunked COPY-append (no delete)
        self.writes.append(("bulk_seed", df.copy()))
        prev = self.t.get(name)
        self.t[name] = pd.concat([prev, df], ignore_index=True) if prev is not None else df.copy()
        return len(df)


class _FakeCtx:
    def __init__(self, store): self.store = store; self.log = logging.getLogger("test")


def _parts():
    dates = ["2023-01-02", "2023-01-03"]
    tickers = ["AAA", "BBB"]
    grid = [(d, t) for d in dates for t in tickers]
    price = pd.DataFrame(grid, columns=["date", "ticker"]).assign(
        f_ret=np.array([0.1, 0.2, 0.3, 0.4], dtype="float64"))
    fund = pd.DataFrame(grid, columns=["date", "ticker"]).assign(
        f_val=np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"))
    betas = pd.DataFrame(grid, columns=["date", "ticker"]).assign(
        beta_mkt=np.array([0.9, 1.1, 1.0, 1.2], dtype="float64"))
    # targets: LONG by horizon (2 horizons) -> the broadcast that used to blow up memory
    trows = []
    for h in (5, 20):
        for (d, t), y in zip(grid, [0.01, 0.02, 0.03, 0.04]):
            trows.append({"date": d, "ticker": t, "target_horizon": h, "target_fwd_ret": y * h})
    targets = pd.DataFrame(trows)
    return {"cube_part_price": price, "cube_part_fundamental": fund,
            "cube_part_betas": betas, "cube_part_targets": targets,
            "sp500_tickers": pd.DataFrame(columns=["ticker", "sector", "industry_group"])}


def _make_step(store):
    step = StepBuildCube.__new__(StepBuildCube)          # skip heavy __init__
    step._context = _FakeCtx(store)
    step._log = logging.getLogger("test")
    step._cfg = {}                                       # composites disabled -> build_composite early-returns
    step.peers = {"AAA": {"BBB": 1.0}, "BBB": {"AAA": 1.0}}
    step._prereqs = lambda: None                         # peers already set
    return step


def test_assemble_streams_per_horizon_and_matches_oneshot():
    store = _FakeStore(_parts())
    step = _make_step(store)

    step.assemble_cube_from_parts()

    # ---- streaming shape: chunked COPY per horizon (replace first, bulk_seed append) ----------
    ops = [op for op, _ in store.writes]
    assert ops == ["replace", "bulk_seed"], f"expected chunked COPY streaming, got {ops}"
    # every write is a bounded row-chunk (never the whole horizon-expanded cube at once)
    assert all(len(df) <= 200_000 for _, df in store.writes)
    cube = store.t["cube"]

    # ---- correctness: equals a one-shot targets.merge(base) -----------------------------------
    p = _parts()
    base_ref = (p["cube_part_price"].merge(p["cube_part_fundamental"], on=["date", "ticker"])
                .merge(p["cube_part_betas"], on=["date", "ticker"]))
    ref = p["cube_part_targets"].merge(base_ref, on=["date", "ticker"], how="inner")
    assert len(cube) == len(ref) == 8, f"rows {len(cube)} vs {len(ref)}"      # 2 dates x 2 tk x 2 horizons
    key = ["ticker", "date", "target_horizon"]
    got = cube.set_index(key).sort_index()
    exp = ref.assign(date=pd.to_datetime(ref["date"]).dt.normalize()).set_index(key).sort_index()
    got.index = got.index.set_levels(pd.to_datetime(got.index.levels[1]).normalize(), level=1)
    for col in ("f_ret", "f_val", "beta_mkt", "target_fwd_ret"):
        assert np.allclose(got[col].to_numpy(float), exp[col].to_numpy(float)), f"{col} mismatch"
    assert "peers" in cube.columns and cube["peers"].notna().all()

    # ---- memory: feature columns are float32 --------------------------------------------------
    f32 = [c for c in ("f_ret", "f_val", "beta_mkt", "target_fwd_ret") if cube[c].dtype == np.float32]
    assert set(f32) == {"f_ret", "f_val", "beta_mkt", "target_fwd_ret"}, \
        {c: str(cube[c].dtype) for c in ("f_ret", "f_val", "beta_mkt", "target_fwd_ret")}

    print("\n=== SANITY CHECK: cube assembly (per-horizon streaming) ===")
    print(f"  wrote 2 horizons as {ops} (one slice resident at a time, not all horizons at once); "
          f"cube {len(cube)} rows == one-shot merge; features float32. OOM path removed.")


if __name__ == "__main__":
    test_assemble_streams_per_horizon_and_matches_oneshot()
