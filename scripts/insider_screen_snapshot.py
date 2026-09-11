"""Snapshot `insider_transactions` before the identity screen re-parse, and diff it after.

⚠ THIS IS THE ROLLBACK CHECK AND THE PROOF DENOMINATOR, IN ONE FILE. The re-parse is the only
step in the identity plan that moves data, and `store.save` upserts, so a mistake there is not
visible as an error -- it is visible only as a row count that nobody wrote down beforehand.

    python scripts/insider_screen_snapshot.py before
    python scripts/insider_screen_snapshot.py after

`before` writes four artefacts under `reports/<today>/`; `after` re-reads them and prints the
closing arithmetic per ticker, per quarter and on the two value repairs. Idempotent: `before`
refuses to overwrite an existing snapshot, because the second run of it would record the
POST-screen table as the baseline and the gate would then pass vacuously.

The two value-repair counts come from `insider_quality.clean_transactions`, the aggregate-layer
repair, NOT from the extract parser -- it is a separate defect with a separate fix, and the
point of carrying it here is that the screen must leave it alone. The 44 repaired rows and the
80 underpriced ones are identified by (accession, security_type, transaction_sk), so a row that
leaves the table is distinguishable from a row whose repair verdict changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.context import get_config_context
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from src.data_store.schema import Tables

CONFIG_DIR = "./configs"
KEYS = ["accession_number", "security_type", "transaction_sk"]
#: Everything `clean_transactions` reads, plus the PK so a repair can be traced to a row.
_REPAIR_COLS = KEYS + ["ticker", "filing_date", "transaction_date", "transaction_code",
                       "security_type", "security_title", "shares", "price_per_share",
                       "value_usd", "officer_title", "is_director", "is_officer",
                       "is_ten_pct_owner", "is_10b5_1"]


def _out_dir() -> Path:
    return Path("reports") / pd.Timestamp.today().strftime("%Y-%m-%d")


def _paths(out: Path) -> dict[str, Path]:
    return {"main": out / "insider_prescreen_snapshot.csv",
            "ticker": out / "insider_prescreen_by_ticker.csv",
            "quarter": out / "insider_prescreen_by_quarter.csv",
            "repair": out / "insider_prescreen_repairs.csv"}


def _measure(store) -> dict:
    """Read the table once and derive every snapshot axis from that one read."""
    df = store.load(Tables.insider_transactions,
                    columns=sorted(set(_REPAIR_COLS + ["quarter"])))
    repaired, diag = clean_transactions(df)
    repairs = (repaired.loc[repaired["price_repaired"], KEYS]
               .assign(kind="price_repaired"))
    return {
        "total": len(df),
        "accessions": df["accession_number"].nunique(),
        "tickers": df["ticker"].nunique(),
        "by_ticker": df.groupby("ticker", dropna=False).size().rename("rows"),
        "by_quarter": df.groupby("quarter", dropna=False).size().rename("rows"),
        "repairs": repairs,
        "diag": diag,
    }


def take_before() -> None:
    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    paths = _paths(out)
    if paths["main"].exists():
        raise SystemExit(
            f"{paths['main']} already exists. A second `before` would snapshot the "
            "POST-screen table and the gate would then pass against itself. Delete it "
            "deliberately if you really are re-baselining.")

    m = _measure(context.store)
    m["by_ticker"].to_csv(paths["ticker"])
    m["by_quarter"].to_csv(paths["quarter"])
    m["repairs"].to_csv(paths["repair"], index=False)
    pd.DataFrame([{"metric": k, "value": v} for k, v in [
        ("total_rows", m["total"]), ("distinct_accessions", m["accessions"]),
        ("distinct_tickers", m["tickers"]),
        *((f"diag_{k}", v) for k, v in m["diag"].items()),
        ("taken_at", pd.Timestamp.now().isoformat()),
    ]]).to_csv(paths["main"], index=False)

    print("\n=== PRE-SCREEN SNAPSHOT ===")
    print(f"  total rows          {m['total']:>12,}")
    print(f"  distinct accessions {m['accessions']:>12,}")
    print(f"  distinct tickers    {m['tickers']:>12,}")
    for k, v in m["diag"].items():
        print(f"  {k:<20}{v:>12,.0f}" if isinstance(v, (int, float)) else f"  {k:<20}{v}")
    print(f"\n  written to {out}/")


def show_after() -> None:
    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    paths = _paths(_out_dir())
    if not paths["main"].exists():
        raise SystemExit(f"no pre-screen snapshot at {paths['main']}; nothing to compare")

    before = pd.read_csv(paths["main"]).set_index("metric")["value"]
    b_ticker = pd.read_csv(paths["ticker"]).set_index("ticker")["rows"]
    b_quarter = pd.read_csv(paths["quarter"]).set_index("quarter")["rows"]
    b_repairs = pd.read_csv(paths["repair"], dtype=str)
    m = _measure(context.store)

    total_before = int(before["total_rows"])
    print("\n=== POST-SCREEN DIFF ===")
    print(f"  rows  {total_before:,} -> {m['total']:,}  "
          f"({m['total'] - total_before:+,})")

    quarantine = context.store.load(Tables.insider_transactions_quarantine,
                                    columns=["reject_reason"], optional=True)
    if quarantine is not None and not quarantine.empty:
        print(f"\n  quarantined {len(quarantine):,} row(s):")
        print(quarantine["reject_reason"].value_counts().to_string()
              .replace("\n", "\n    ").rjust(0))

    moved = (pd.concat([b_ticker.rename("before"), m["by_ticker"].rename("after")], axis=1)
             .fillna(0).astype(int))
    moved = moved[moved["before"] != moved["after"]].assign(
        delta=lambda d: d["after"] - d["before"]).sort_values("delta")
    print(f"\n  {len(moved)} ticker(s) moved, {len(b_ticker) - len(moved)} unchanged:")
    print("    " + moved.to_string().replace("\n", "\n    "))

    q = (pd.concat([b_quarter.rename("before"), m["by_quarter"].rename("after")], axis=1)
         .fillna(0).astype(int))
    q = q[q["before"] != q["after"]]
    print(f"\n  {len(q)} of {len(b_quarter)} quarter(s) moved")

    # ⚠ the aggregate-layer repair must be untouched by an extract-layer screen
    a_repairs = m["repairs"].astype(str)
    b_keys = set(map(tuple, b_repairs[KEYS].to_numpy()))
    a_keys = set(map(tuple, a_repairs[KEYS].to_numpy()))
    print(f"\n  price repairs {len(b_keys)} -> {len(a_keys)}; "
          f"{len(b_keys & a_keys)} identical, {len(b_keys - a_keys)} gone, "
          f"{len(a_keys - b_keys)} new")
    if b_keys - a_keys:
        print("    gone: " + ", ".join(f"{k[0]}/{k[2]}" for k in sorted(b_keys - a_keys)[:10]))
    for key in ("underpriced_rows", "value_before", "value_after", "scoped_rows"):
        print(f"  {key:<18} {float(before[f'diag_{key}']):>20,.0f} -> "
              f"{float(m['diag'][key]):>20,.0f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "before"
    {"before": take_before, "after": show_after}[mode]()
