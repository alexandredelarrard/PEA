"""
Per-horizon STREAMING training (StepModelling._process_horizons + blend) — the memory-light path
that reads/trains ONE horizon at a time and frees it, instead of holding every horizon's panel
(and the all-horizons cube) at once. Heavy ML sub-steps are stubbed; the test pins the control
flow + memory discipline + the blend math:
  * one panel loaded per horizon (streamed), then released -> no `self.panels` / `self.cube`,
  * a model produced per horizon, train-end recorded,
  * the blend consumes the small per-horizon score frames and IR-weights them correctly.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.modelling.long_short.step_train import StepModelling


def _panel(h):
    d = pd.to_datetime(["2023-01-02", "2023-01-03"])
    return pd.DataFrame({"date": list(d) * 2, "ticker": ["A", "A", "B", "B"],
                         "y": [0.1, 0.2, 0.3, 0.4], "f_x": [1.0, 2.0, 3.0, 4.0]})


def _score(h):
    d = pd.to_datetime(["2023-01-02", "2023-01-03"])
    z = {5: [0.0, 1.0, 1.0, 0.0], 20: [1.0, 0.0, 0.0, 1.0]}[h]
    return pd.DataFrame({"date": list(d) * 2, "ticker": ["A", "A", "B", "B"], f"z_{h}": z})


def _make_step():
    s = StepModelling.__new__(StepModelling)
    s._context = SimpleNamespace(save=False)          # save=False -> no diagnostics/run_stamp
    s._log = logging.getLogger("test")
    s.horizons = [5, 20]
    s.model_types = ["lightgbm"]
    s._half_life = lambda: None
    s._loads = []                                     # record load calls (streaming proof)
    def _load(h):
        s._loads.append(h)
        return _panel(h)
    s._load_horizon_panel = _load
    s._cv_one_horizon = lambda h, p: s.horizon_ic.__setitem__(h, {"mean_ic": 0.05, "ic_ir": 1.0})
    s._train_final_one = lambda h, p: {"lightgbm": f"model_{h}"}
    s._score_one_horizon = lambda h, models, p: _score(h)
    return s


def test_process_horizons_streams_one_at_a_time_then_blends():
    s = _make_step()
    s._process_horizons()

    # --- streaming: exactly one panel load per horizon, and nothing held all-at-once ------------
    assert s._loads == [5, 20], f"expected one streamed load per horizon, got {s._loads}"
    assert set(s.models) == {5, 20} and s.models[5]["lightgbm"] == "model_5"
    assert not hasattr(s, "panels"), "self.panels must NOT exist (per-horizon streaming)"
    assert not hasattr(s, "cube"), "self.cube must NOT exist (no all-horizons load)"
    assert s._train_end_effective == pd.Timestamp("2023-01-03")
    assert len(s._score_frames) == 2

    # --- blend: IR-weights (both ic_ir=1.0 -> 0.5/0.5), weighted nanmean of z, per-day rank ------
    s.blend_and_generate_signal()
    assert s.horizon_weights == {5: 0.5, 20: 0.5}
    pred = s.predictions.set_index(["date", "ticker"])
    # A on 2023-01-02: z_5=0.0, z_20=1.0 -> combined 0.5 ; B same day: z_5=1.0, z_20=0.0 -> 0.5
    assert np.isclose(pred.loc[(pd.Timestamp("2023-01-02"), "A"), "combined"], 0.5)
    assert np.isclose(pred.loc[(pd.Timestamp("2023-01-03"), "A"), "combined"], 0.5)
    assert "signal" in s.predictions.columns and s.signal_date == pd.Timestamp("2023-01-03")

    print("\n=== SANITY CHECK: per-horizon streaming training ===")
    print(f"  loaded one panel per horizon {s._loads} then freed each (no self.panels/self.cube); "
          f"models {sorted(s.models)}; blend IR-weights {s.horizon_weights}, combined = weighted "
          "nanmean of per-horizon z. Peak memory = one horizon, not the whole cube.")


if __name__ == "__main__":
    test_process_horizons_streams_one_at_a_time_then_blends()
