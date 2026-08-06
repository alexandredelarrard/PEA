"""
superinvestor_features.py  (src/data_aggregate/utils/superinvestor_features.py)
-------------------------------------------------------------------------------
"Smart-money" 13F features restricted to a curated subset of ELITE managers
(the Dataroma superinvestors roster; see fetch_superinvestors), layered ON TOP of
the all-filer institutional features. Where `institutional_features` asks "are
institutions in aggregate accumulating?", this asks the sharper question "are the
proven long-term winners accumulating?" — the buy/sell evolution of the managers
that matter, each scaled by its roster `weight` (top-ranked managers count more).

Same manager-grain 13F input as institutional_features (one row per
manager x security x quarter: cik, period, ticker, shares, value_usd [, filing_date]),
same POINT-IN-TIME discipline: each quarter is stamped `as_of = quarter-end + 45d`
(the 13F filing deadline) so a backtest never sees a position before it was public.

Weighted features (per stock, per quarter; peer-relativized downstream):
    super_holders          Σ weights of roster managers holding      (elite breadth/conviction)
    super_breadth_chg      Δ weighted holders vs prior quarter        (elite entering/leaving)
    super_cluster_buying   (Σw increasers − Σw decreasers)/Σw holders (elite net accumulation)
    super_new_buyer_ratio  Σw fresh initiators / Σw holders           (elite fresh conviction)
    super_shares_chg       % QoQ change in weighted aggregate shares  (VOLUME accumulation)
    super_value_chg        % QoQ change in weighted aggregate value   (VALUE accumulation)
    super_value_to_mcap    weighted elite long value / market cap      (elite ownership weight)
    super_flow_to_mcap     weighted net QoQ $ flow / market cap        (size-scaled elite $ flow)

All values are cross-sectionally z-scored / ranked downstream, so the constant
weight scale is irrelevant — only the ordering across names matters.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from src.context import Context
from src.data_aggregate.utils.target.factors import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.utils.string import pad_cik

_FILING_LAG_DAYS = 45   # 13F filing deadline after quarter-end (leak-free floor)
_HOLDINGS_TABLE = "sec13f_hr"     # the ~20M-row all-filer 13F table (literal, as elsewhere)
_HOLDINGS_COLS = ["cik", "period", "ticker", "shares", "value_usd", "filing_date"]
# normalize a stored TEXT cik the SAME way pad_cik does (digits of the pre-decimal part, left-padded
# to 10) so a Postgres WHERE matches the padded roster keys whether the DB stored it padded, unpadded
# or as "1234.0". Postgres ARE supports the \D escape.
_CIK_SQL_NORM = r"lpad(regexp_replace(split_part(cik, '.', 1), '\D', '', 'g'), 10, '0')"


def _weight_map(roster: dict | list | None) -> dict[str, float]:
    """{padded-cik: weight} from a loaded superinvestors roster.

    Primary shape is the `{cik: investor_name}` map under `cik_to_name` (or a bare
    `{cik: name}` dict): EQUAL-weighted — each elite manager counts the same, and
    since the features are peer-relativized downstream the absolute weight scale is
    irrelevant. Falls back to the legacy `{managers: [{cik, weight}]}` / bare-list
    shape (explicit per-manager weights)."""
    if roster is None:
        return {}
    if isinstance(roster, dict):
        mapping = roster.get("cik_to_name")
        if mapping is None and "managers" not in roster:      # a bare {cik: name} dict
            mapping = {k: v for k, v in roster.items() if isinstance(v, str) and pad_cik(k)}
        if mapping:
            return {c: 1.0 for cik in mapping if (c := pad_cik(cik))}
        managers = roster.get("managers", [])
    else:
        managers = roster or []
    out: dict[str, float] = {}
    for m in managers:
        cik = pad_cik(m.get("cik"))
        w = m.get("weight")
        if cik and w is not None:
            out[cik] = out.get(cik, 0.0) + float(w)
    return out


def load_superinvestor_holdings(context: Context, roster: dict | list | None) -> pd.DataFrame | None:
    """Read ONLY the roster managers' rows from the ~20M-row `sec13f_hr` table — the
    elite subset is a handful of CIKs, so pulling the whole table (then discarding 99% in
    `_super_quarter_features`) is what made this OOM-crash. DB-backed stores push the filter down
    with an engine-side `WHERE <normalized cik> IN (roster)` (cik text normalized exactly like
    `pad_cik`, so padded / unpadded / '1234.0' all match); non-DB / test stores fall back to a
    projected full read filtered in pandas. None if the roster resolves to no manager."""
    weights = _weight_map(roster)
    if not weights:
        return None
    store = context.store
    if hasattr(store, "exists") and not store.exists(_HOLDINGS_TABLE):
        return None
    engine = getattr(store, "engine", None)
    if engine is not None:
        cols = ", ".join(f'"{c}"' for c in _HOLDINGS_COLS)
        sql = text(f'SELECT {cols} FROM "{_HOLDINGS_TABLE}" WHERE {_CIK_SQL_NORM} IN :ciks'
                   ).bindparams(bindparam("ciks", expanding=True))
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"ciks": sorted(weights)})
    df = store.load(_HOLDINGS_TABLE, columns=_HOLDINGS_COLS)      # fallback: full read, filter in memory
    if df is None or df.empty:
        return None
    return df[df["cik"].map(pad_cik).isin(weights)].reset_index(drop=True)


def _super_quarter_features(holdings: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Roster-manager-grain 13F -> one row per (ticker, quarter) with the WEIGHTED
    QoQ elite features, stamped `as_of = quarter-end + 45 days`."""
    h = holdings.copy()
    h["cik"] = h["cik"].map(pad_cik)
    h = h[h["cik"].isin(weights)]
    if h.empty:
        return pd.DataFrame()
    h["period"] = pd.to_datetime(h["period"]).dt.normalize()
    h = h.dropna(subset=["ticker", "period"])
    for c in ("shares", "value_usd"):
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0) if c in h.columns \
            else pd.Series(0.0, index=h.index)
    if "filing_date" in h.columns:                        # amendments: keep last-filed
        h = h.sort_values("filing_date")
    h = h.drop_duplicates(["ticker", "cik", "period"], keep="last")
    h["w"] = h["cik"].map(weights).astype(float)

    rows = []
    for ticker, tdf in h.groupby("ticker"):
        prev_sh: dict = {}
        prev_val, prev_whold = np.nan, np.nan
        for p in sorted(tdf["period"].unique()):
            cur = tdf[tdf["period"] == p]
            sh = dict(zip(cur["cik"], cur["shares"]))
            wt = dict(zip(cur["cik"], cur["w"]))
            cur_ciks, prev_ciks = set(sh), set(prev_sh)
            both = cur_ciks & prev_ciks
            has_prev = len(prev_ciks) > 0

            w_hold = float(sum(wt.values()))
            w_inc = float(sum(wt[c] for c in both if sh[c] > prev_sh[c]))
            w_dec = float(sum(wt[c] for c in both if sh[c] < prev_sh[c]))
            w_new = float(sum(wt[c] for c in (cur_ciks - prev_ciks)))
            w_shares = float((cur["shares"] * cur["w"]).sum())
            # weights are constant per manager (roster-level), so the prior quarter's
            # weighted shares reuse the same per-CIK weight
            prev_shares = float(sum(prev_sh[c] * weights[c] for c in prev_ciks)) \
                if has_prev else np.nan
            w_value = float((cur["value_usd"] * cur["w"]).sum())

            rows.append({
                "ticker": ticker,
                "as_of": pd.Timestamp(p) + pd.Timedelta(days=_FILING_LAG_DAYS),
                "super_holders": w_hold,
                "super_value": w_value,
                "super_value_flow": (w_value - prev_val)
                                    if (has_prev and np.isfinite(prev_val)) else np.nan,
                "super_breadth_chg": (w_hold - prev_whold) if has_prev else np.nan,
                "super_shares_chg": (w_shares / prev_shares - 1.0)
                                    if (has_prev and prev_shares and prev_shares > 0) else np.nan,
                "super_value_chg": (w_value / prev_val - 1.0)
                                   if (has_prev and np.isfinite(prev_val) and prev_val > 0) else np.nan,
                "super_cluster_buying": ((w_inc - w_dec) / w_hold) if w_hold > 0 else np.nan,
                "super_new_buyer_ratio": (w_new / w_hold)
                                         if (w_hold > 0 and has_prev) else np.nan,
            })
            prev_sh, prev_val, prev_whold = sh, w_value, w_hold
    return pd.DataFrame(rows)


def build_superinvestor_feature_panel(
    holdings: pd.DataFrame | None,
    roster: dict | list | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    shares_out_history: pd.DataFrame | None = None,
    stock_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format elite-manager 13F feature panel (`f_super_*_vs_peers`, `_xs`).
    Empty if there are no holdings or the roster resolves to no manager. `roster` is
    the loaded superinvestors JSON (dict with `managers`) or a bare manager list.
    `shares_out_history` (+ `stock_close`) enable the market-cap-scaled weight and flow."""
    need = {"cik", "period", "ticker", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return pd.DataFrame(columns=["date", "ticker"])
    weights = _weight_map(roster)
    if not weights:
        return pd.DataFrame(columns=["date", "ticker"])

    qf = _super_quarter_features(holdings, weights)
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    feats = ["super_holders", "super_breadth_chg", "super_shares_chg", "super_value_chg",
             "super_cluster_buying", "super_new_buyer_ratio"]
    fields = {f: fundamentals_to_daily(qf, f, trading_index) for f in feats}

    have_shares = shares_out_history is not None and not shares_out_history.empty
    if have_shares and stock_close is not None and not stock_close.empty:
        # elite ownership WEIGHT by value and size-scaled net $ flow, via a point-in-time
        # daily market cap (ffilled sharesOutstanding x daily close).
        mcap = daily_market_cap(shares_out_history, stock_close)
        if not mcap.empty:
            mpos = mcap.where(mcap > 0)
            v2m = (fundamentals_to_daily(qf, "super_value", trading_index) / mpos
                   ).replace([np.inf, -np.inf], np.nan)
            if v2m.notna().any().any():
                fields["super_value_to_mcap"] = v2m
            f2m = (fundamentals_to_daily(qf, "super_value_flow", trading_index) / mpos
                   ).replace([np.inf, -np.inf], np.nan)
            if f2m.notna().any().any():
                fields["super_flow_to_mcap"] = f2m

    return build_peer_relative_panel(fields, peer_dict)
