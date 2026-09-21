"""
clip.py  (src/validate/checks/clip.py)
--------------------------------------------------------------------------------------------
What the standardised views are hiding: how much of a peer-z leg is sitting ON its clip, and
how much of a cross-sectional rank is one tied plateau.

WHAT A CLIPPED VALUE ACTUALLY MEANS, AND WHY THE WORDING MATTERS. A `_vs_peers` leg at exactly
8.0 does not say "this company is eight sigma from its peers". It says the standardisation ran
out of room: the peer group was too small, or its dispersion collapsed, or the raw value was
extreme enough that the z-score was capped. The value that reaches the model is the CLIP, not
the company. So a leg with a large on-clip share is a leg whose top decile carries no ordering
at all -- every name in it has the same number. The fix is upstream, in the peer group or the
raw leg; it is NEVER "clip harder", and this check does not suggest it.

The tie test is the same defect on the other view. A percentile leg where half the
cross-section shares one value is not a ranking, it is a plateau with a decimal point --
measured on `f_bank_roa_xs`, 38 distinct values over 93,012 rows, because the leg only exists
for banks and the rest of the cross-section is one tie.

⚠ ABSTAINS WHEN THE TABLE DECLARES NO SUFFIX CONVENTION. `cube_part_momentum` has no `_xs` and
no `_vs_peers` leg at all; declaring one anyway would make this check measure tie mass over an
empty set of legs and report a clean pass. That is the exact failure this package exists to
prevent, so a table with no declared convention exits 3.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Table, resolve
from src.validate.frame import column_groups, feature_columns
from src.validate.io import cache_used, read_columns
from src.validate.result import CheckResult, Finding, full_table_only
from src.validate.spec import UndeclaredTableError, load_spec

log = logging.getLogger(__name__)

CHECK = "clip"

#: Columns read at once, as in `profile` -- one group is one pass over the table.
GROUP = 8

#: A value within this of the clip counts as ON it. The clip is applied in float64 and the
#: cache stores float32, so an exact `== 8.0` would miss the same value by one ULP.
_CLIP_TOL = 1e-6

_MAX_FINDINGS = 40

_WHY_SUFFIX = ("no `xs_suffix` and no `peer_suffix`, so there is no standardised view to "
               "measure -- and a table with no such leg (cube_part_momentum has none) would "
               "otherwise report a clean on-clip share over an empty set of columns")

_WHY_CLIP = ("peer-z legs exist here but no `clip_peer` value is declared, so there is no "
             "number to test them against -- the clip is a builder constant, not something "
             "this check can infer from the data it is testing")


def _tie_mass(values: pd.Series, dates: pd.Series, min_tickers: int) -> dict[str, Any]:
    """Per date, the share of the cross-section sitting on its single most common value.

    Thin dates are dropped, not averaged in: the modal share of a four-name cross-section is
    at least 0.25 by arithmetic, and a leg that only exists for four names would score as a
    plateau on every date."""
    frame = pd.DataFrame({"v": values, "d": dates}).dropna()
    if frame.empty:
        return {"dates": 0, "mean_modal_share": None, "worst": None, "worst_date": None}
    sizes = frame.groupby("d")["v"].size()
    wide = sizes[sizes >= min_tickers].index
    frame = frame[frame["d"].isin(wide)]
    if frame.empty:
        return {"dates": 0, "mean_modal_share": None, "worst": None, "worst_date": None}
    modal = frame.groupby("d")["v"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    return {"dates": int(len(modal)), "mean_modal_share": float(modal.mean()),
            "worst": float(modal.max()), "worst_date": modal.idxmax()}


def check_clip(context: Context, table: Table | str, *, config: DictConfig,
               cache: Any = None, tickers: list[str] | None = None,
               group: int = GROUP, **kwargs: Any) -> CheckResult:
    """On-clip share per peer-z leg, tie mass per cross-sectional leg."""
    spec_t = resolve(table)
    if (declined := full_table_only(CHECK, spec_t.name, tickers)) is not None:
        return declined
    spec = load_spec(config, spec_t)
    suffixes = spec.view_suffixes()
    if not suffixes:
        raise UndeclaredTableError(spec_t.name, "xs_suffix/peer_suffix", _WHY_SUFFIX)

    date_col = spec_t.date_col
    columns = feature_columns(context, spec_t)
    peer_legs = [c for c in columns if spec.peer_suffix and c.endswith(spec.peer_suffix)]
    xs_legs = [c for c in columns if spec.xs_suffix and c.endswith(spec.xs_suffix)
               and not (spec.peer_suffix and c.endswith(spec.peer_suffix))]
    if not peer_legs and not xs_legs:
        return CheckResult.abstained(
            CHECK, spec_t.name,
            f"the table declares the suffixes {suffixes} but carries no column ending in "
            f"any of them -- the declaration and the live schema disagree")
    if peer_legs and spec.clip_peer is None:
        raise UndeclaredTableError(spec_t.name, "clip_peer", _WHY_CLIP)

    findings: list[Finding] = []
    peer_metrics: dict[str, Any] = {}
    tie_metrics: dict[str, Any] = {}
    rows = 0

    for block in column_groups(peer_legs + xs_legs, group):
        wanted = ([date_col] + list(block)) if date_col else list(block)
        frame = read_columns(context, spec_t, wanted, cache=cache)
        rows = max(rows, len(frame))
        dates = frame[date_col] if date_col else None
        for column in block:
            values = pd.to_numeric(frame[column], errors="coerce")
            finite = np.isfinite(values.values)
            n_ok = int(finite.sum())
            if column in peer_legs:
                limit = float(spec.clip_peer)
                on_clip = int((np.abs(values.values[finite]) >= limit - _CLIP_TOL).sum())
                share = (on_clip / n_ok) if n_ok else None
                peer_metrics[column] = {"n_ok": n_ok, "on_clip": on_clip, "share": share,
                                        "clip": limit,
                                        "max_abs": (float(np.abs(values.values[finite]).max())
                                                    if n_ok else None)}
            else:
                tie_metrics[column] = _tie_mass(values, dates, spec.min_tickers_xs)
        log.info("clip %s: %d/%d legs", spec_t.name,
                 len(peer_metrics) + len(tie_metrics), len(peer_legs) + len(xs_legs))

    over = sorted(((c, m) for c, m in peer_metrics.items()
                   if m["share"] is not None and m["share"] > spec.clip_limit_share),
                  key=lambda kv: -kv[1]["share"])
    for column, metric in over[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            6, field=column,
            observed=f"{metric['share']:.1%} of {metric['n_ok']:,} values sit on the "
                     f"+/-{metric['clip']:g} clip ({metric['on_clip']:,} rows)",
            expected=f"< {spec.clip_limit_share:.0%} on the clip. A clipped value reports "
                     f"that the standardisation ran out of room -- too few peers, collapsed "
                     f"peer dispersion, or an extreme raw leg -- not a distance from peers, "
                     f"and every name in that mass carries the SAME number. Fix the peer "
                     f"group or the raw leg; do not clip harder",
            **metric))

    plateaus = sorted(((c, m) for c, m in tie_metrics.items()
                       if m["mean_modal_share"] is not None
                       and m["mean_modal_share"] >= spec.tie_limit_share),
                      key=lambda kv: -kv[1]["mean_modal_share"])
    for column, metric in plateaus[:_MAX_FINDINGS]:
        findings.append(Finding.at(
            6, field=column,
            observed=f"the modal value covers {metric['mean_modal_share']:.1%} of the "
                     f"cross-section on an average date ({metric['dates']:,} dates with at "
                     f"least {spec.min_tickers_xs} names; worst {metric['worst']:.1%} on "
                     f"{pd.Timestamp(metric['worst_date']).date()})",
            expected=f"< {spec.tie_limit_share:.0%} -- a percentile leg where half the "
                     f"cross-section shares one value is a plateau, not a ranking, and every "
                     f"name inside it is unordered",
            **metric))

    first_date, last_date = (context.store.bounds(spec_t) if date_col else (None, None))
    scope = {"rows": rows, "first_date": first_date, "last_date": last_date,
             "peer_legs": len(peer_legs), "xs_legs": len(xs_legs),
             "peer_suffix": spec.peer_suffix, "xs_suffix": spec.xs_suffix,
             "clip_peer": spec.clip_peer, "min_tickers_xs": spec.min_tickers_xs,
             "source": "cache" if cache_used(cache, spec_t) else "db"}
    metrics = {"peer": peer_metrics, "tie": tie_metrics,
               "peer_over_limit": [c for c, _ in over],
               "tie_over_limit": [c for c, _ in plateaus]}
    return CheckResult.measured(CHECK, spec_t.name, findings, scope=scope, metrics=metrics)
