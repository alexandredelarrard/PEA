"""
def14a_impute.py  (src/data_aggregate/utils/def14a_impute.py)
-------------------------------------------------------------
Data-CLEANING step for the cube build: deduce / fill MISSING values in the `def14a_llm`
proxy archive before governance/executive-pay features are computed (the LLM extraction
leaves gaps on filings it couldn't fully parse). Applied clean-on-read in
`StepBuildCube._load_governance` — the raw extraction table is never mutated.

STRICTLY non-destructive: a value is only ever written where it is currently NaN — a real
extracted number is never overwritten.

Deductions:
  1. CEO pay identity. `ceo_total_comp` == sum of the six Summary-Comp-Table components
     (salary + bonus + stock + option + non-equity incentive + all-other). Fill the total
     from the components, or the single missing component from `total - others` (clipped
     >=0). Caveat: the schema omits the SCT "change in pension value" column, so a large
     deduced component may absorb it -- acceptable for a gap-fill.
  2. Board consistency. `pct_technology_directors == n_technology_directors / board_size`
     (either direction); `n_directors == board_size`.
  3. Pay ratio. `ceo_pay_ratio == ceo_total_comp / median_employee_pay` -> fill the median
     employee pay (or the ratio) from the other two.
  4. Temporal gap-fill. Per ticker (sorted by filing date), fill a value missing BETWEEN
     two filled years -- linear interpolation for levels/ratios, carry-forward for stable
     flags -- via `limit_area='inside'`, so leading/trailing gaps and special-meeting
     proxies at the edges are left untouched.

`impute_def14a(df) -> (df, stats)` is pure (returns a copy + per-rule fill counts).
"""
from __future__ import annotations

import pandas as pd

CEO_COMP = ["ceo_salary", "ceo_bonus", "ceo_stock_awards", "ceo_option_awards",
            "ceo_non_equity_incentive", "ceo_all_other_comp"]
# levels / ratios that vary smoothly -> linear interpolate an interior gap
INTERP = ["board_size", "n_directors", "avg_director_age", "avg_board_tenure",
          "pct_independent_directors", "pct_female_directors", "pct_technology_directors",
          "n_technology_directors", "avg_other_public_boards", "insider_ownership_pct",
          "ceo_ownership_pct", "n_five_percent_holders", "say_on_pay_support_pct",
          "ceo_age", "median_employee_pay", "ceo_pay_ratio"]
# stable per-company/CEO facts -> carry the last known value forward within an interior gap
FLAGS = ["ceo_is_founder", "ceo_is_board_chair", "independent_chair", "lead_independent_director",
         "classified_board", "dual_class_shares", "poison_pill", "majority_voting",
         "technology_committee", "ceo_since_year", "ceo_name_proxy"]
INT_COLS = ["n_directors", "board_size", "n_technology_directors", "ceo_age",
            "n_five_percent_holders", "n_neos", "ceo_since_year"]


def _fill(df: pd.DataFrame, col: str, cond: pd.Series, values, stats: dict, tag: str) -> None:
    """Set df[col] = values ONLY where col is currently NaN AND `cond` AND value is finite."""
    if col not in df.columns:
        return
    vals = values if isinstance(values, pd.Series) else pd.Series(values, index=df.index)
    m = df[col].isna() & cond.fillna(False) & vals.notna()
    n = int(m.sum())
    if n:
        df.loc[m, col] = vals[m]
        stats[tag] = stats.get(tag, 0) + n


def _reconcile_rows(df: pd.DataFrame, stats: dict) -> None:
    comps = [c for c in CEO_COMP if c in df.columns]
    if "ceo_total_comp" in df.columns and len(comps) == 6:
        _fill(df, "ceo_total_comp", df[comps].notna().all(axis=1),
              df[comps].sum(axis=1), stats, "ceo_total_comp = sum(components)")
        for c in comps:
            others = [x for x in comps if x != c]
            cond = df["ceo_total_comp"].notna() & df[others].notna().all(axis=1)
            _fill(df, c, cond, (df["ceo_total_comp"] - df[others].sum(axis=1)).clip(lower=0),
                  stats, "ceo component = total - others")
    if {"n_technology_directors", "board_size"} <= set(df.columns):
        bs = df["board_size"]
        _fill(df, "pct_technology_directors", (bs > 0) & df["n_technology_directors"].notna(),
              (df["n_technology_directors"] / bs).clip(upper=1.0), stats, "pct_tech = n_tech / board")
        _fill(df, "n_technology_directors", (bs > 0) & df["pct_technology_directors"].notna(),
              (df["pct_technology_directors"] * bs).round(), stats, "n_tech = pct_tech * board")
    if {"n_directors", "board_size"} <= set(df.columns):
        _fill(df, "n_directors", df["board_size"].notna(), df["board_size"], stats, "n_directors = board_size")
        _fill(df, "board_size", df["n_directors"].notna(), df["n_directors"], stats, "board_size = n_directors")
    if {"ceo_pay_ratio", "median_employee_pay", "ceo_total_comp"} <= set(df.columns):
        r, med, tot = df["ceo_pay_ratio"], df["median_employee_pay"], df["ceo_total_comp"]
        _fill(df, "median_employee_pay", (r > 0) & tot.notna(), tot / r, stats, "median_pay = total / ratio")
        _fill(df, "ceo_pay_ratio", (med > 0) & tot.notna(), tot / med, stats, "pay_ratio = total / median")


def _temporal_fill(df: pd.DataFrame, stats: dict) -> None:
    df.sort_values(["ticker", "as_of"], inplace=True)
    g = df.groupby("ticker", sort=False)
    for col in INTERP:
        if col in df.columns:
            filled = g[col].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
            newly = df[col].isna() & filled.notna()
            n = int(newly.sum())
            if n:
                df.loc[newly, col] = filled[newly]
                stats[f"interp: {col}"] = n
    for col in FLAGS:
        if col in df.columns:
            fwd, bwd = g[col].ffill(), g[col].bfill()
            inside = df[col].isna() & fwd.notna() & bwd.notna()      # bounded by a known value each side
            n = int(inside.sum())
            if n:
                df.loc[inside, col] = fwd[inside]
                stats[f"carry: {col}"] = n


def impute_def14a(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (imputed copy, stats) — fills only NaNs via the identities + temporal gap-fill."""
    if df is None or df.empty:
        return df, {}
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    was_na = {c: df[c].isna() for c in INT_COLS if c in df.columns}  # to round ONLY what we fill
    stats: dict[str, int] = {}
    _reconcile_rows(df, stats)          # within-row identities
    _temporal_fill(df, stats)           # cross-year interior gaps
    _reconcile_rows(df, stats)          # reconcile values the temporal fill unlocked
    for c, na in was_na.items():        # keep DEDUCED counts integral (never touch real values)
        filled = na & df[c].notna()
        df.loc[filled, c] = df.loc[filled, c].round()
    return df.sort_values(["ticker", "as_of"]).reset_index(drop=True), stats
