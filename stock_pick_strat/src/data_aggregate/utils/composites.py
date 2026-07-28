"""
composites.py  (src/data_aggregate/utils/composites.py)
-------------------------------------------------------
Thematic COMPOSITE signals: group economically-related features into a handful
of orthogonal themes (value / quality / growth / distress / capital-allocation /
expectations / management / workforce / technical) and average the members into
one score per theme.

Why: at low signal-to-noise, averaging correlated members cancels idiosyncratic
noise (signal-to-noise up) and stops a cluster of collinear features from
over-voting, giving the model a few stable, interpretable inputs.

Design choices matching the brief ("do not lose information; prefer correlation
over no info"):
  * ADDITIVE -- the raw features are kept in the panel; composites are EXTRA
    columns (`comp_<theme>`). Nothing is dropped, so no information is lost.
  * NaN-tolerant -- a composite is the mean of whichever members are present on
    a given (date, ticker); a stock missing some members still gets a score.
  * Sign-aware -- prefix a member with '-' in the config to invert it, so every
    member is oriented "higher = better / long side" before averaging (otherwise
    opposing-signed members cancel and the composite loses its information).
  * Each member is cross-sectionally re-standardized per day first, so members on
    different scales (percentile _xs, peer-z _vs_peers, 0/1 flags) combine fairly.
  * Missing members are skipped but REPORTED (`missing_members` / the `log` warning).
    Skipping silently hid real damage: four members -- gross_profitability,
    net_debt_to_ebitda, interest_coverage and sbc_intensity -- were absent from the
    cube for as long as the panel merge was renaming them to `_x`/`_y`, so quality,
    distress and capital_allocation each ran a member short with no signal that
    anything was wrong.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def _parse_member(m: str) -> tuple[float, str]:
    """'-f_x' -> (-1, 'f_x'); 'f_x' -> (+1, 'f_x')."""
    m = str(m)
    return (-1.0, m[1:].strip()) if m.startswith("-") else (1.0, m.strip())


def _xs_standardize(panel: pd.DataFrame, cols: list[str], method: str,
                    clip: float) -> pd.DataFrame:
    """Cross-sectionally standardize each column within each date."""
    g = panel.groupby("date")[cols]
    if method == "rank":
        return (g.rank(pct=True) - 0.5) * 2.0            # -> ~[-1, 1], mean ~0
    z = g.transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
    return z.clip(-clip, clip)


def missing_members(panel: pd.DataFrame, groups: dict) -> dict[str, list[str]]:
    """`{theme: [member, ...]}` for every configured member absent from `panel`.

    A composite quietly built from half its members still looks healthy, so the
    caller is expected to surface this rather than discover it months later."""
    out: dict[str, list[str]] = {}
    for theme, members in (groups or {}).items():
        gone = sorted({col for _, col in map(_parse_member, members)
                       if col not in panel.columns})
        if gone:
            out[theme] = gone
    return out


def build_composites(panel: pd.DataFrame, groups: dict, method: str = "zscore",
                     clip: float = 4.0, log: logging.Logger | None = None) -> pd.DataFrame:
    """Append `comp_<theme>` columns to a long feature panel.

    `groups` maps theme -> list of member feature names (each optionally '-'-
    prefixed to invert). Members absent from the panel are skipped AND warned about
    via `log`. Returns the SAME panel with the composite columns added (raw features
    untouched)."""
    if not groups:
        return panel

    parsed = {theme: [_parse_member(m) for m in members]
              for theme, members in groups.items()}
    gaps = missing_members(panel, groups)
    if gaps and log is not None:
        total = sum(len(v) for v in gaps.values())
        log.warning(
            "Composites: %d configured member(s) absent from the panel and skipped -- "
            "%s. A theme whose members are missing is built from the remainder, so "
            "check whether the feature was renamed, or its source table is empty.",
            total, "; ".join(f"{t}: {', '.join(m)}" for t, m in sorted(gaps.items())),
        )
    present = sorted({col for members in parsed.values()
                      for _, col in members if col in panel.columns})
    if not present:
        if log is not None:
            log.warning("Composites: NO configured member is present -- none built.")
        return panel

    z = _xs_standardize(panel, present, method, clip)

    out = panel.copy()
    for theme, members in parsed.items():
        signed = {col: sign * z[col] for sign, col in members if col in z.columns}
        if not signed:
            continue
        # NaN-tolerant mean across the members present for this (date, ticker)
        out[f"comp_{theme}"] = pd.DataFrame(signed).mean(axis=1, skipna=True)
    return out
