"""Size the symbol-reuse exposure of the two CIK-LESS tables. MEASURE ONLY -- no fix (D2).

`sec_short_interest` and `sec_fails_to_deliver` are the only tables in the repo that filter a
GENUINE HISTORICAL TAPE on `Symbol`. FINRA and RegSHO publish every symbol that traded on a
settlement date, and the fetchers keep the rows whose symbol is in today's universe -- so a
2011 fails row for the OLD holder of `COR` is admitted under today's `COR`. Unlike the vendor
fetchers (yfinance, Sharadar), which return today's holder's own back-filled history and so
cannot carry another company's rows at all, these files really do contain them.

⚠ NEITHER TABLE CARRIES A CIK, so the entity comparison the insider screen uses is unavailable
and `symbol_tenure` is the only resolver there is. This is the phase that justifies axis B
being built at all.

⚠ THE OUTCOME IS SIX-WAY, NOT TWO, AND EVERY EXTRA BUCKET EXISTS BECAUSE COLLAPSING IT WOULD
MISSTATE THE DEFECT.

    held                today's holder of the ticker also held the symbol on that date
    other_entity        ANOTHER entity held it -- the defect being sized
    not_in_roster       the ticker has LEFT `sp500_tickers`, so there is no entity to compare
                        against. ⚠ Reported separately after a first version of this script
                        labelled all 2,060 `EA` and 1,934 `AVB` fails rows `other_entity`:
                        `universe_entity` has nothing to return for a departed ticker, so a
                        None expectation compared unequal to every real holder and manufactured
                        a defect out of a survivorship artefact.
    tenure_blind        ⚠ THE SECOND FALSE-POSITIVE CLASS, and it is why `other_entity` alone
                        is only an UPPER BOUND. These tickers' own roster CIK files Forms 3/4/5
                        under a DIFFERENT symbol string, so `entity_for` can never return
                        "held" for them at any date and 100% of their rows read as another
                        company's. `BNY` is the clean case: BNY Mellon still files as `BK`, so
                        the symbol `BNY` in the Form 345 sets is BlackRock New York Municipal
                        Income Trust's throughout. The list is not re-derived here -- it is
                        the D19 allow-list in `entity_lineage_manual.json`, which already
                        carries the written adjudication for each of these 10 tickers.
                        ⚠ These rows are UNRESOLVABLE BY TENURE, not proven clean: `BNY`'s
                        pre-2025 fails rows really are the BlackRock trust's, because the
                        RegSHO tape for that date really did mean that company. The bucket
                        says "this method cannot answer", and a CUSIP-keyed fix could.
    unknown_pre_tenure  the date precedes the symbol's FIRST observation. `valid_from` is an
                        observation (a company's first Form 3/4/5), not a listing date, so this
                        is genuinely unknown -- calling it not-held would manufacture a defect
                        and calling it held would hide one.
    unknown_gap         inside the symbol's observed span but between two tenures
    ambiguous           two entities under one symbol on one date; `entity_for` refuses to
                        break the tie rather than silently picking one

    python scripts/measure_symbol_tenure_exposure.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.context import get_config_context
from src.data_extract.utils.common.entity_lineage import load_d19_allowlist
from src.data_extract.utils.common.identity import AmbiguousSymbolTenure, load_identity
from src.data_store.schema import Tables

CONFIG_DIR = "./configs"
OUT = Path("reports") / pd.Timestamp.today().strftime("%Y-%m-%d")
#: `symbol_tenure` is derived from the Form 345 bulk sets, whose own first quarter is 2006q1.
#: Every row below this date is structurally unresolvable, not evidence of anything (D7).
TENURE_FLOOR = pd.Timestamp("2006-01-01")

#: ⚠ The attribute name is `short_interest` while the TABLE is `sec_short_interest` -- one of
#: the five registry entries whose two names differ.
TABLES = [(Tables.short_interest, "sec_short_interest"),
          (Tables.sec_fails_to_deliver, "sec_fails_to_deliver")]


def _classify(identity, symbol: str, day: pd.Timestamp, expected: str | None,
              blind: frozenset[str]) -> str:
    """One (symbol, date) -> one of the six buckets in the module docstring.

    ⚠ `expected is None` IS CHECKED FIRST AND IT IS NOT A DISAGREEMENT. A ticker that has left
    `sp500_tickers` has no entity to compare against, and `None != holder` for every real
    holder -- which is how a first version of this script reported all 3,994 `EA` + `AVB` fails
    rows as another company's. They are a survivorship artefact, not symbol reuse.
    """
    if expected is None:
        return "not_in_roster"
    if symbol in blind:
        return "tenure_blind"
    try:
        holder = identity.entity_for(symbol, day)
    except AmbiguousSymbolTenure:
        return "ambiguous"
    if holder is None:
        rows = identity.tenure_by_symbol.get(symbol)
        if not rows:
            return "no_tenure"
        # before the symbol's FIRST observation -> genuinely unknown, never "not held"
        return "unknown_pre_tenure" if day < min(r[1] for r in rows) else "unknown_gap"
    return "held" if holder == expected else "other_entity"


def measure(store, identity, table, name: str,
            blind: frozenset[str]) -> pd.DataFrame:
    df = store.load(table, columns=["ticker", "date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n=== {name} ===")
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} ticker(s), "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    below = int((df["date"] < TENURE_FLOOR).sum())
    print(f"  {below:,} row(s) ({below / len(df):.2%}) fall below the {TENURE_FLOOR.date()} "
          "tenure floor and are structurally unresolvable (D7)")

    # One verdict per (ticker, date) is still ~1M lookups; the tape is one row per pair
    # already, so this is the grain and there is nothing to dedupe.
    expected = {t: identity.universe_entity(t) for t in df["ticker"].unique()
                if t in identity.roster_cik}
    verdicts = [_classify(identity, t, d, expected.get(t), blind)
                for t, d in zip(df["ticker"], df["date"])]
    df["verdict"] = verdicts

    counts = df["verdict"].value_counts()
    print("\n  verdicts:")
    for verdict, n in counts.items():
        print(f"    {verdict:<20}{n:>12,}  {n / len(df):>7.2%}")

    wrong = df[df["verdict"] == "other_entity"]
    if not wrong.empty:
        total = df.groupby("ticker").size()
        per = (wrong.groupby("ticker")
               .agg(rows=("date", "size"), first=("date", "min"), last=("date", "max"))
               .assign(ticker_rows=lambda d: total.reindex(d.index),
                       pct_of_ticker=lambda d: (d["rows"] / d["ticker_rows"] * 100).round(1))
               .sort_values("rows", ascending=False))
        print(f"\n  ⚠ {len(wrong):,} row(s) ({len(wrong) / len(df):.3%}) over "
              f"{len(per)} ticker(s) are another entity's, "
              f"{wrong['date'].min().date()} -> {wrong['date'].max().date()}:")
        print("    " + per.head(20).to_string().replace("\n", "\n    "))
        per.to_csv(OUT / f"{name}_other_entity_by_ticker.csv")
    else:
        print("\n  no row resolves to another entity")
    return df


def main() -> None:
    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    identity = load_identity(context, CONFIG_DIR)
    # The D19 allow-list IS the tenure-blind list: every entry on it is a ticker whose roster
    # CIK and whose filings name different entities, with a written reading of why.
    blind = frozenset(load_d19_allowlist(CONFIG_DIR))
    print(f"\n  {len(blind)} tenure-blind ticker(s) from the D19 allow-list: "
          f"{', '.join(sorted(blind))}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for table, name in TABLES:
        df = measure(context.store, identity, table, name, blind)
        row = {"table": name, "rows": len(df), "tickers": df["ticker"].nunique(),
               "first": df["date"].min().date(), "last": df["date"].max().date(),
               "below_tenure_floor": int((df["date"] < TENURE_FLOOR).sum())}
        row.update(df["verdict"].value_counts().to_dict())
        summary.append(row)
    out = pd.DataFrame(summary).fillna(0)
    out.to_csv(OUT / "symbol_tenure_exposure.csv", index=False)
    print(f"\n  written to {OUT}/symbol_tenure_exposure.csv")
    print("\n  ⚠ NO FETCHER WAS CHANGED (D2). The natural fix input for the follow-up task is "
          "the FTD source file's UNREAD CUSIP column (fetch_fails_to_deliver.py:80-81), which "
          "would give these rows the issuer key they lack; it is deliberately not parsed here.")


if __name__ == "__main__":
    main()
