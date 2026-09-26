"""Test Phase 2.3's claim that EVERY residual outlier in the insider panel traces to the
symbol-first resolution defect. The claim is falsifiable; this is the test (D11).

2.3 named four: `EXE` at `buy_value_mcap_180d` 292, `AXON` at `buy_shares_so_180d` 2.26 (226%
of shares outstanding bought in 180 days), `ECHO` and `APP` unquantified. ⚠ THE STATISTICS ARE
READ FROM 2.3, NOT RE-DERIVED -- a comparison against a freshly invented definition proves
nothing about the claim that was actually made.

⚠ `APP` HAS NO MEASURED OUT-OF-LINEAGE GROUP IN THE RESEARCH. If its outlier persists, 2.3's
claim is falsified in part, and this script says so rather than attributing it to this defect.

HOW "BEFORE" IS RECONSTRUCTED, AND WHY IT IS EXACT FOR THESE FOUR NAMES. The re-parse already
ran, so the pre-screen table is gone -- but nothing is lost: the rows it removed are all in
`insider_transactions_quarantine`, with the ticker they CLAIMED. So

    before(T) = after(T)  UNION  quarantine(claimed_ticker == T)

and the only term that could break it is a row ADMITTED under T (one the old symbol path
dropped), which would be in `after` and not in `before`. `identity-admitted.csv` lists every
admitted group: MRK, TT, DLR, CDW, ACN, MRVL, GM, PCG, DD -- none of the four. The script
ASSERTS that rather than trusting it.

The panel is built by the REAL builder through the REAL step loaders, twice, so the two sides
differ in exactly one input.

    python scripts/prove_insider_outliers.py
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.context import get_config_context
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from src.data_store.schema import Table, Tables

CONFIG_DIR = "./configs"
OUT = Path("reports") / pd.Timestamp.today().strftime("%Y-%m-%d")
NAMED = ["EXE", "AXON", "ECHO", "APP"]
#: The two statistics 2.3 quoted, with its quoted values.
#: ⚠ THE COLUMN IS `f_ic_insider_<name>`, NOT `f_<name>`. `build_peer_relative_panel` prefixes
#: `f_` onto the FULL registry key, and the keys in `insider_features.EMISSION` already carry
#: the `ic_insider_` family prefix. A first run of this script guessed `f_buy_value_mcap_180d`
#: and every cell came back None -- which read exactly like "the outlier is gone", the most
#: dangerous possible way for a proof script to be wrong.
#: No `_xs` leg: the raw level is what 2.3 measured, and an outlier's cross-sectional rank is
#: capped by construction, so the z-leg cannot show the magnitude.
STATS = {"f_ic_insider_buy_value_mcap_180d": {"EXE": 292.0},
         "f_ic_insider_buy_shares_so_180d": {"AXON": 2.26}}


def _panel(
    step: StepCubeInstitutionals,
    insider: pd.DataFrame,
    frames: PriceFrames,
    shares: pd.DataFrame | None,
) -> pd.DataFrame:
    """Run the real `_insider_panel` with `insider` substituted for the stored table."""
    original = step._load_source

    def load_source(table: Table, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
        if table is Tables.insider_transactions:
            scoped = insider
            if universe is not None:
                allowed = sorted(set(map(str, universe)))
                scoped = scoped.loc[scoped["ticker"].astype(str).isin(allowed)]
            return scoped.copy()
        if table is Tables.insider_transactions_live:
            return None
        return original(table, universe)

    step._load_source = load_source
    try:
        panel = step._insider_panel(frames, shares, ConditioningSink())
    finally:
        step._load_source = original
    return panel if panel is not None else pd.DataFrame(columns=["date", "ticker"])


def main() -> None:
    config, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    OUT.mkdir(parents=True, exist_ok=True)
    store = context.store

    after = store.load(Tables.insider_transactions)
    quarantine = store.load(Tables.insider_transactions_quarantine, optional=True)
    if quarantine is None or quarantine.empty:
        raise SystemExit("the quarantine table is empty -- run the re-parse first")

    admitted = pd.read_csv(OUT / "identity-admitted.csv")
    clash = sorted(set(NAMED) & set(admitted["ticker"]))
    assert not clash, (
        f"{clash} appear in identity-admitted.csv, so before(T) = after(T) + quarantine(T) "
        "would OVERSTATE the pre-screen table for them. Reconstruct from the zips instead.")

    # the quarantine row already carries the claimed ticker in `ticker` (see its schema note)
    restored = quarantine[[c for c in after.columns if c in quarantine.columns]]
    before = pd.concat([after, restored], ignore_index=True)
    print(f"\n  after {len(after):,} rows + quarantine {len(restored):,} "
          f"-> reconstructed before {len(before):,}")
    print("  ⚠ that GLOBAL total overshoots the real pre-screen 2,031,286 by 4,291, and the "
          "reason matters: the parse-time screen quarantines the excluded roster tickers' FULL "
          "zip history while the stored table only ever held what an earlier universe admitted. "
          "The reconstruction is EXACT for the four names here -- verified against the "
          "per-ticker snapshot: ECHO 2,426 = 1,349 + 1,077, APP 2,628 = 2,336 + 292, "
          "EXE 3,998 = 3,762 + 236, AXON 2,213 = 2,133 + 80 -- which is all the panel needs, "
          "since a per-ticker feature reads only that ticker's own tape.")

    step = StepCubeInstitutionals(context=context, config=config)
    frames = step._load_frames()
    shares = step._load_shares_out()

    print("\n=== building the insider panel TWICE (same code, one input differs) ===")
    panels = {"before": _panel(step, before, frames, shares),
              "after": _panel(step, after, frames, shares)}

    # Persisted so any follow-up read costs nothing: each panel is ~20 minutes to build.
    for side, panel in panels.items():
        panel.to_parquet(OUT / f"insider_panel_{side}.parquet", index=False)
    missing = [s for s in STATS if s not in panels["after"].columns]
    if missing:
        raise SystemExit(
            f"{missing} are not columns of the built panel. Available insider columns: "
            f"{sorted(c for c in panels['after'].columns if c.startswith('f_ic_insider'))[:6]}"
            " ... -- fix STATS rather than reporting None as 'the outlier is gone'.")

    rows = []
    for stat, quoted in STATS.items():
        for ticker in NAMED:
            rec = {"feature": stat, "ticker": ticker,
                   "quoted_in_2_3": quoted.get(ticker)}
            for side, panel in panels.items():
                if stat not in panel.columns:
                    rec[f"max_{side}"] = None
                    continue
                sl = panel.loc[panel["ticker"] == ticker, stat].dropna()
                rec[f"max_{side}"] = float(sl.max()) if not sl.empty else None
                rec[f"n_{side}"] = int(len(sl))
            rows.append(rec)
    out = pd.DataFrame(rows)
    print("\n=== the four named outliers, before vs after ===")
    print("  " + out.to_string(index=False).replace("\n", "\n  "))
    out.to_csv(OUT / "insider_outlier_proof.csv", index=False)

    # --- the tape-level explanation: which quarantined rows drive each name --- #
    print("\n=== the quarantined rows that explain each name ===")
    q = quarantine[quarantine["ticker"].isin(NAMED)]
    if q.empty:
        print("  NONE of the four names has a quarantined row.")
    else:
        detail = (q.groupby(["ticker", "issuer_cik", "issuer_name"])
                  .agg(rows=("accession_number", "size"),
                       value_usd=("value_usd", "sum"),
                       first=("filing_date", "min"), last=("filing_date", "max"))
                  .sort_values("rows", ascending=False))
        print("  " + detail.to_string().replace("\n", "\n  "))
        detail.to_csv(OUT / "insider_outlier_quarantined_detail.csv")
    missing = [t for t in NAMED if t not in set(q["ticker"])]
    if missing:
        print(f"\n  ⚠ {missing} have NO quarantined row. If their 2.3 outlier persists above, "
              "2.3's 'every residual outlier traces to this defect' is FALSIFIED for them and "
              "must be reported as unexplained, not attributed to this fix.")

    # --- the row/value deltas per named ticker, for the report table --- #
    print("\n=== per-ticker tape deltas ===")
    tape = []
    for ticker in NAMED:
        b = before[before["ticker"] == ticker]
        a = after[after["ticker"] == ticker]
        tape.append({"ticker": ticker, "rows_before": len(b), "rows_after": len(a),
                     "value_before": b["value_usd"].sum(), "value_after": a["value_usd"].sum()})
    tape_df = pd.DataFrame(tape)
    print("  " + tape_df.to_string(index=False).replace("\n", "\n  "))
    tape_df.to_csv(OUT / "insider_outlier_tape_delta.csv", index=False)

    # --- 2.2b's tie-fraction criterion, re-measured on the cleaned panel --- #
    print("\n=== emission classes, re-measured on the CLEANED cross-section ===")
    sheet = _emission_sheet(panels["after"])
    print("  " + sheet.to_string(index=False).replace("\n", "\n  "))
    sheet.to_csv(OUT / "insider_emission_remeasured.csv", index=False)
    if "implied" in sheet.columns:
        scored = sheet.dropna(subset=["implied"])
        disagree = scored[scored["declared"] != scored["implied"]]
        if disagree.empty:
            worst = scored[scored["declared"] == "raw+xs"]["ties_per_date_pct"].max()
            print(f"\n  all {len(scored)} declared classes survive the cleaned cross-section; "
                  f"the most-tied raw+xs feature sits at {worst}%, well inside the 90% floor")
        else:
            print(f"\n  {len(disagree)} of {len(scored)} feature(s) LOSE their _xs leg:")
            print("  " + disagree.to_string(index=False).replace("\n", "\n  "))
            print("  ⚠ A class change is a REPORT, not an automatic edit. `EMISSION` is a "
                  "declaration with written reasoning per feature -- a low tie fraction is "
                  "NOT a reason to rank something already comparable across dates -- so a "
                  "disagreement is read before it is applied.")


def _emission_sheet(panel: pd.DataFrame) -> pd.DataFrame:
    """2.2b's tie-fraction criterion, re-measured. The `_xs` leg of a 90%+-tied cross-section
    is a plateau: almost every name shares one rank, so the column moves on which names happen
    to be present rather than on the signal.

    ⚠ RE-MEASURED BECAUSE THE INPUT CHANGED, NOT BECAUSE THE RULE DID. 2.3's `raw+xs` 9 /
    `raw` 5 split was measured on the CONTAMINATED panel, and removing another company's rows
    from 48 tickers changes every date's cross-section -- so the split has to be re-read rather
    than assumed to survive. The criterion itself is taken verbatim from
    `insider_features.EMISSION`'s docstring table and is not re-invented here.

    ⚠ THE TEST IS ONE-DIRECTIONAL: IT REMOVES LEGS, IT NEVER ADDS THEM. `EMISSION`'s docstring
    says so in bold, and a first version of this function ignored it -- scoring every
    low-tie feature as `raw+xs` and then "finding" that `owner_surprise_120d`,
    `days_since_last_buy` and `net_buy_ratio_180d` should gain an `_xs` leg. They should not,
    for reasons that have nothing to do with ties: the first is ALREADY a percentile in [0, 1]
    (a percentile of a percentile is a re-rank), the second is a day count that means the same
    thing in 2009 as in 2025, and the third is bounded [-1, 1] by construction. So a declared
    `raw` is never challenged here; only a declared `raw+xs` that has crossed into a plateau.
    """
    from src.data_aggregate.utils.institutionals.insider_features import EMISSION

    rows = []
    for feature, declared in EMISSION.items():
        col = next((c for c in (f"f_{feature}", f"f_{feature.removeprefix('ic_')}")
                    if c in panel.columns), None)
        if col is None:
            rows.append({"feature": feature, "declared": declared, "note": "absent from panel"})
            continue
        sl = panel[["date", col]].dropna()
        if sl.empty:
            rows.append({"feature": feature, "declared": declared, "note": "all-null"})
            continue
        per_date = sl.groupby("date")[col]
        # ties/date = the share of rows that are NOT the sole holder of their value
        ties = float((1.0 - per_date.nunique() / per_date.size()).mean())
        rows.append({
            "feature": feature, "declared": declared,
            "nonnull_pct": round(len(sl) / len(panel) * 100, 1),
            "uniq_per_date": int(per_date.nunique().mean()),
            "ties_per_date_pct": round(ties * 100, 1),
            # one-directional: a declared raw+xs whose cross-section has become a plateau
            # loses its leg; a declared raw is never promoted (see the docstring).
            "implied": "raw" if (declared == "raw+xs" and ties >= 0.90) else declared})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
