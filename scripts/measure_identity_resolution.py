"""
Replay the OLD symbol-first resolution against the NEW CIK-first one over the 81 cached
Form 345 quarters, and report the numbers part-3's gate is built on.

Offline -- reads only `data/sec_insider_transactions/*.zip` and the two identity tables. The
figures it prints are the ones written into
`reports/2026-09-11/identity-resolver-phase-3.md`:

    n_admitted       filings/rows the CIK-first path resolves that symbol-first DROPS
    n_quarantined    filings/rows symbol-first keeps that CIK-first rejects outright
    n_relabelled     filings/rows both keep, under DIFFERENT universe tickers
    n_no_issuer_cik  measured 0 of 4,402,307 -- which is why no symbol fallback exists

⚠ The "old" path here is TODAY'S old path: today's 500-ticker universe and today's
`cik_to_ticker` (which already carries the register chains). It is therefore a forward-looking
measure of what the NEXT fetch would do, not a reconstruction of the run that produced the
current table -- that run used an older universe, which is why `EA` and `AVB` rows sit in the
table and appear in neither column here.

    PYTHONPATH=. python scripts/measure_identity_resolution.py
"""
from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from src.context import get_config_context
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.sec_utils import cik_to_ticker, load_cik_mapping

CONFIG_DIR = "./configs"
CACHE = Path("data/sec_insider_transactions")
OUT = Path("reports/2026-09-11")


def _read(z: zipfile.ZipFile, names: dict[str, str], member: str,
          wanted: set[str]) -> pd.DataFrame:
    if member not in names:
        return pd.DataFrame()
    frame = pd.read_csv(z.open(names[member]), sep="\t", dtype=str, low_memory=False,
                        usecols=lambda c: c.upper() in wanted)
    frame.columns = [c.upper() for c in frame.columns]
    return frame


def main() -> None:
    _, context = get_config_context(CONFIG_DIR, use_cache=False, save=False)
    identity = load_identity(context, CONFIG_DIR)
    universe = set(identity.roster_cik)
    cik2tkr = cik_to_ticker(load_cik_mapping(context), config_dir=CONFIG_DIR)

    tally: Counter = Counter()
    admitted, rejected = [], []
    zips = sorted(CACHE.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no cached Form 345 quarters under {CACHE}")

    for path in zips:
        with zipfile.ZipFile(path) as z:
            names = {n.upper(): n for n in z.namelist()}
            sub = _read(z, names, "SUBMISSION.TSV",
                        {"ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "ISSUERNAME"})
            if sub.empty:
                continue
            nonderiv = _read(z, names, "NONDERIV_TRANS.TSV",
                             {"ACCESSION_NUMBER", "NONDERIV_TRANS_SK"})
            deriv = _read(z, names, "DERIV_TRANS.TSV", {"ACCESSION_NUMBER", "DERIV_TRANS_SK"})

        # rows per filing, keyed exactly as the table's PK is -- one per (accession, sk) per
        # security type, and a transaction with no SK cannot be keyed, so the parser drops it
        per_filing: Counter = Counter()
        for frame, sk in ((nonderiv, "NONDERIV_TRANS_SK"), (deriv, "DERIV_TRANS_SK")):
            if frame.empty:
                continue
            kept = frame.dropna(subset=[sk]).drop_duplicates(["ACCESSION_NUMBER", sk])
            per_filing.update(kept["ACCESSION_NUMBER"].value_counts().to_dict())

        symbol = sub["ISSUERTRADINGSYMBOL"].astype("string").str.strip().str.upper()
        cik = (sub["ISSUERCIK"].astype(str).str.strip()
               .str.replace(r"\.0$", "", regex=True).str.zfill(10))
        old = symbol.where(symbol.isin(universe)).fillna(
            pd.Series(cik.map(cik2tkr), index=sub.index))
        old = old.where(old.isin(universe))
        new = pd.Series(cik.map(identity.entity_ticker), index=sub.index)
        rows = sub["ACCESSION_NUMBER"].map(per_filing).fillna(0).astype(int)

        is_admitted = old.isna() & new.notna()
        is_rejected = old.notna() & new.isna()
        is_relabelled = old.notna() & new.notna() & (old != new)
        tally["filings_read"] += len(sub)
        tally["no_issuer_cik"] += int(sub["ISSUERCIK"].isna().sum())
        for label, mask in (("admitted", is_admitted), ("quarantined", is_rejected),
                            ("relabelled", is_relabelled)):
            tally[f"{label}_filings"] += int(mask.sum())
            tally[f"{label}_rows"] += int(rows[mask].sum())
        if is_admitted.any():
            admitted.append(pd.DataFrame({"ticker": new[is_admitted], "cik": cik[is_admitted],
                                          "name": sub.loc[is_admitted, "ISSUERNAME"],
                                          "rows": rows[is_admitted]}))
        moved = is_rejected | is_relabelled
        if moved.any():
            rejected.append(pd.DataFrame({"filed_as": old[moved], "resolves_to": new[moved],
                                          "cik": cik[moved],
                                          "name": sub.loc[moved, "ISSUERNAME"],
                                          "rows": rows[moved]}))

    print("\n=== IDENTITY RESOLUTION REPLAY: 81 cached quarters ===")
    for key in ("filings_read", "no_issuer_cik", "admitted_filings", "admitted_rows",
                "quarantined_filings", "quarantined_rows", "relabelled_filings",
                "relabelled_rows"):
        print(f"  {key:22s} {tally[key]:>12,}")

    for frames, name, keys in ((admitted, "identity-admitted.csv", ["ticker", "cik", "name"]),
                               (rejected, "identity-rejected-replay.csv",
                                ["filed_as", "resolves_to", "cik", "name"])):
        out = (pd.concat(frames).groupby(keys)["rows"].agg(["sum", "size"])
               .rename(columns={"sum": "rows", "size": "filings"})
               .sort_values("rows", ascending=False))
        out.to_csv(OUT / name)
        print(f"\n  {name}: {len(out)} group(s), {out['rows'].sum():,} rows")
        print(out.head(12).to_string())
    print("\n  OK: admitted > 0, so the screen is not purely subtractive.")


if __name__ == "__main__":
    main()
