"""Size the dual-class 13F undercount. MEASURE ONLY -- no fetcher change, no re-ingest (D2a).

`fetch_13f._resolve_tickers` ends `return out[out["ticker"].isin(universe)]`. `GOOG` and
`GOOGL` are ONE CIK but TWO CUSIPs (`02079K107` Class C, `02079K305` Class A), so OpenFIGI maps
them to two tickers -- and `configs.yml: redundant_ticks` excludes `GOOG` from the universe, so
every manager's Alphabet CLASS-C position fails that `isin` and is dropped silently. Same shape
for `FOX` (Fox Corp Class B) and `NWS` (News Corp Class B).

⚠ THIS IS NOT THE SAME TRADE AS THE PRICE TABLES. Keeping one investable line per issuer is BY
DESIGN in `prices` / `short_interest` / `fails_to_deliver`, where a second share class would be
a duplicate row for one economic position. Here it is a straight UNDERCOUNT: a manager's total
exposure to Alphabet is the SUM of both classes, and this is the one place in the repo where
summing across share classes is exactly what is wanted. Issuer-grain `entity_id` is why -- both
CUSIPs sit on one entity.

⚠ THE DROPPED ROWS ARE NOT IN `sec13f_hr` TO BE COUNTED -- the filter runs before the write, so
there is no record of what it removed. `sec13f_manager_holdings` is the measurable table: it is
the same 13F source at CUSIP grain with NO universe filter, so BOTH classes survive in it. The
undercount is therefore measured there and read across, which is sound because `sec13f_hr` and
`sec13f_manager_holdings` are built from the same filings -- but it is an inference about
`sec13f_hr`, not a direct count of it, and the report must say so.

    python scripts/measure_dual_class_13f_gap.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.context import get_config_context
from src.data_store.schema import Tables
from src.utils.universe import load_universe_tickers

CONFIG_DIR = "./configs"
OUT = Path("reports") / pd.Timestamp.today().strftime("%Y-%m-%d")

#: The excluded class -> the class that IS in the universe. `EA` has no partner: the config
#: comment says "EA got acquired", so it is excluded for a completely different reason and is
#: counted apart rather than folded into the dual-class number.
PAIRS = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}


def main() -> None:
    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    OUT.mkdir(parents=True, exist_ok=True)
    universe = set(load_universe_tickers(context))
    excluded = {str(t).strip().upper()
                for t in context.config.data_extract.redundant_ticks}

    cmap = context.store.load(Tables.cusip_ticker_map, columns=["cusip", "ticker"])
    cmap["ticker"] = cmap["ticker"].astype("string").str.upper().str.strip()
    print("\n=== 1. does cusip_ticker_map carry the excluded classes separately? ===")
    print(f"  {len(cmap):,} mapped CUSIP(s); {int(cmap['ticker'].isna().sum()):,} persisted "
          "NULL (unmapped, NEVER RETRIED -- fetch_cusip_map.py:121-127)")
    for tick in sorted(excluded | set(PAIRS.values())):
        hits = sorted(cmap.loc[cmap["ticker"] == tick, "cusip"])
        flag = "EXCLUDED" if tick in excluded else "in universe" if tick in universe else "-"
        print(f"  {tick:<6} {flag:<12} {len(hits)} cusip(s): {', '.join(hits) or '(none)'}")

    by_ticker = cmap.dropna(subset=["ticker"]).groupby("ticker")["cusip"].apply(set).to_dict()
    unmapped = set(cmap.loc[cmap["ticker"].isna(), "cusip"])

    holdings = context.store.load(
        Tables.sec13f_manager_holdings, columns=["period", "cusip", "shares", "value_usd"])
    holdings["cusip"] = holdings["cusip"].astype(str).str.strip()
    print(f"\n  sec13f_manager_holdings: {len(holdings):,} rows, "
          f"{holdings['cusip'].nunique():,} CUSIP(s), "
          f"{holdings['period'].min()} -> {holdings['period'].max()}")

    # ------------------------------------------------- 2. the dual-class buckets --- #
    print("\n=== 2. dropped_dual_class: value the universe filter removes ===")
    rows, per_quarter = [], []
    for dropped, kept in PAIRS.items():
        d_cusips, k_cusips = by_ticker.get(dropped, set()), by_ticker.get(kept, set())
        d = holdings[holdings["cusip"].isin(d_cusips)]
        k = holdings[holdings["cusip"].isin(k_cusips)]
        total_v = d["value_usd"].sum() + k["value_usd"].sum()
        total_n = len(d) + len(k)
        rows.append({
            "dropped_class": dropped, "kept_class": kept,
            "dropped_positions": len(d), "kept_positions": len(k),
            "dropped_value_usd": d["value_usd"].sum(), "kept_value_usd": k["value_usd"].sum(),
            "pct_value_understated": (d["value_usd"].sum() / total_v * 100) if total_v else 0.0,
            "pct_positions_understated": (len(d) / total_n * 100) if total_n else 0.0})
        if not d.empty or not k.empty:
            q = pd.concat([d.assign(leg="dropped"), k.assign(leg="kept")])
            per_quarter.append(
                q.pivot_table(index="period", columns="leg", values="value_usd",
                              aggfunc="sum").assign(issuer=dropped))
    gap = pd.DataFrame(rows)
    print("  " + gap.to_string(index=False).replace("\n", "\n  "))
    for r in rows:
        if r["pct_value_understated"]:
            print(f"\n  ⚠ institutional ownership of the {r['kept_class']} issuer in "
                  f"`sec13f_hr` is understated by {r['pct_value_understated']:.1f}% of value "
                  f"and {r['pct_positions_understated']:.1f}% of positions, because every "
                  f"{r['dropped_class']} position is dropped by isin(universe).")
    gap.to_csv(OUT / "dual_class_13f_gap.csv", index=False)
    if per_quarter:
        pq = pd.concat(per_quarter).reset_index()
        pq.to_csv(OUT / "dual_class_13f_gap_by_quarter.csv", index=False)
        print(f"\n  per-quarter detail -> {OUT}/dual_class_13f_gap_by_quarter.csv "
              f"({len(pq)} rows)")

    # ---------------------------------------- 3. the OTHER cause, kept separate --- #
    print("\n=== 3. dropped_unmapped: a different defect, counted apart ===")
    um = holdings[holdings["cusip"].isin(unmapped)]
    never = holdings[~holdings["cusip"].isin(set(cmap["cusip"]))]
    print(f"  mapped-to-NULL  {len(um):>10,} position(s)  ${um['value_usd'].sum() / 1e9:>12,.1f}bn")
    print(f"  absent from map {len(never):>10,} position(s)  "
          f"${never['value_usd'].sum() / 1e9:>12,.1f}bn")
    print("  ⚠ Neither is the dual-class defect. An unmapped CUSIP is persisted NULL and never "
          "retried, so its rows are dropped for want of a ticker rather than by the universe "
          "filter -- folding them into the headline would make it unattributable.")

    # -------------------------------------------------------- 4. EA, counted apart --- #
    print("\n=== 4. EA: excluded as ACQUIRED, not as a share class ===")
    ea = holdings[holdings["cusip"].isin(by_ticker.get("EA", set()))]
    print(f"  {len(ea):,} position(s), ${ea['value_usd'].sum() / 1e9:,.1f}bn, "
          f"{ea['period'].min() if not ea.empty else '-'} -> "
          f"{ea['period'].max() if not ea.empty else '-'}")
    print("  Counted separately and NOT added to the dual-class figure: `EA` has no sibling "
          "class in the universe, so there is nothing for its value to be summed onto.")

    print("\n  ⚠ NO FETCHER WAS CHANGED (D2a). The fix routes the excluded CUSIP through "
          "`entity_ticker()`; note for whoever scopes it that a SUMMED numerator needs the "
          "CONSOLIDATED `shares_outstanding` denominator the repo already chose -- an "
          "economic-basis denominator under a summed numerator would be a new defect.")


if __name__ == "__main__":
    main()
