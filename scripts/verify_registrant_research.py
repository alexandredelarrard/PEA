"""Verify a PROPOSED predecessor CIK against EDGAR's own metadata, before it becomes a segment.

The offline half of this check is already done and it passed: for 30 proposed chains the
research's CURRENT CIK matched our roster 30/30, and its cut date matched the truncation we
measured independently to a median 11 days, 29/30 (AME is the exception -- see the report).
Neither of those facts says the PREDECESSOR CIK is the right one. Only EDGAR does.

So this asks the submissions document for each proposed predecessor exactly what was asked of
BLK's 0001060021, AVGO's 0001441634 and STE's 0000815065 -- and nothing more, because that is
the evidence standard those three were written on:

  name + formerNames   does the entity name match the story, and did it change AT the cut?
  archive span         does it START at or before the ticker's price history (the archive-start
                       test) and STOP at the cut? A predecessor that keeps filing for years
                       afterwards is a CONTINUING SUBSIDIARY, and its boundary rule differs.
  form mix             10-K / DEF 14A / 8-K counts -- a co-registrant debt shell files no proxy,
                       so "it filed its own proxy" is the cheap proof that it was the listed
                       company rather than a financing subsidiary.

It writes NOTHING and proposes NOTHING. Adding a segment stays a hand decision with the
evidence string attached, the way the other three were.

⚠ ONE EDGAR WALK AT A TIME. ~35 submissions documents is a walk like any other; run it when
nothing else is talking to SEC.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.context import get_config_context
from scripts.detect_registrant_cutovers import all_filings, submissions

#: ticker -> (proposed predecessor CIKs oldest-first, the cut the research gives, what the
#: research says the oldest one IS). Cut is the LAST hop's date; a multi-hop chain is checked
#: hop by hop against each CIK's own span rather than against one date.
PROPOSED: dict[str, tuple[list[str], str, str]] = {
    "TPL":   (["0000097517"], "2021-01-11", "Texas Pacific Land Trust"),
    "MRVL":  (["0001058057"], "2021-04-20", "Marvell Technology Group Ltd (Bermuda)"),
    "PSA":   (["0000318380"], "2007-06-01", "Public Storage, Inc. (California)"),
    "MOS":   (["0000820626"], "2004-10-22", "IMC Global Inc (IGL)"),
    "ACN":   (["0001134538"], "2009-09-01", "Accenture Ltd (Bermuda)"),
    "CNP":   (["0000048732"], "2002-08-31", "Reliant Energy, Inc. (REI)"),
    "TROW":  (["0000080255"], "2000-12-28", "T. Rowe Price Associates, Inc."),
    "AME":   (["0000006082"], "1997-08-01", "AMETEK, Inc. (old) -- CUT DOES NOT EXPLAIN HOLE"),
    "EG":    (["0000914748"], "2000-02-24", "Everest Reinsurance Holdings, Inc."),
    "PSKY":  (["0000813828"], "2025-08-07", "Paramount Global"),
    "TKO":   (["0001091907"], "2023-09-12", "World Wrestling Entertainment, Inc."),
    "TT":    (["0000050485", "0001160497"], "2009-07-01", "Ingersoll-Rand Co (NJ) then Ltd"),
    "ORCL":  (["0000777676"], "2006-01-31", "Oracle Corporation (old)"),
    "NEM":   (["0000071824"], "2002-02-16", "Newmont Mining Corporation (old)"),
    "CMCSA": (["0000022301"], "2002-11-18", "Comcast Corporation (old)"),
    "NOC":   (["0000072945"], "2001-04-02", "Northrop Grumman Corporation (old)"),
    "WBD":   (["0001320482"], "2008-09-17", "Discovery Holding Company"),
    "HST":   (["0000314733"], "1998-12-29", "Host Marriott Corporation"),
    "SPG":   (["0000912564"], "1998-09-24", "Simon DeBartolo Group, Inc."),
    "DD":    (["0000030554"], "2017-08-31", "E.I. du Pont de Nemours -- ticker lineage"),
    "VMC":   (["0000103973"], "2007-11-16", "Vulcan Materials (old)"),
    "BNY":   (["0000009626"], "2007-07-01", "The Bank of New York Company"),
    "RF":    (["0000036032"], "2004-07-01", "Regions Financial (old)"),
    "EVRG":  (["0000054507"], "2018-06-04", "Westar Energy -- research picks over GXP"),
    "EXC":   (["0000078100"], "2000-10-20", "PECO Energy"),
    "DUK":   (["0000030371"], "2006-04-03", "Duke Energy (old)"),
    "COP":   (["0000078214"], "2002-08-30", "Phillips Petroleum"),
    "COR":   (["0000855042"], "2001-08-29", "AmeriSource Health"),
    "NI":    (["0000823392"], "2000-11-01", "NiSource (old)"),
    "DVN":   (["0000837330"], "1999-08-17", "Devon Energy (old)"),
}

#: The other public parent in a merger of equals. NOT a segment and never loaded: appending a
#: second parent's 10-K history to this ticker would import another company's accounts. Held
#: here so the evidence string can name it and nobody rediscovers it as a missing hop.
CO_PREDECESSOR = {
    "DD": "0000029915 Dow Chemical (ACCOUNTING acquirer -- see report, DD is the one case "
          "where the two lineages diverge)",
    "VMC": "0000037651 Florida Rock", "BNY": "0000064782 Mellon Financial",
    "RF": "0000100893 Union Planters", "EVRG": "0001143068 Great Plains Energy",
    "EXC": "0000022606 Unicom", "DUK": "0000899652 Cinergy", "COP": "0001066806 Conoco",
    "COR": "0000011454 Bergen Brunswig", "NI": "0000022099 Columbia Energy",
    "DVN": "0000077320 PennzEnergy",
    "TKO": "0001766363 Endeavor (contributed UFC, did NOT terminate)",
    "CMCSA": "0000005907 AT&T (contributed Broadband, continued separately)",
    "WBD": "0000732717 AT&T (contributed WarnerMedia, continued separately)",
    "MOS": "Cargill crop nutrition -- PRIVATE, no CIK history to append",
}

FORMS = ("10-K", "DEF 14A", "8-K", "10-Q", "20-F", "6-K")


def span(context, doc: dict) -> pd.DataFrame:
    """Every filing in the archive -- `recent` PLUS the older shards, via `all_filings`.

    ⚠ THE FIRST VERSION OF THIS READ `recent` ALONE AND ITS OWN DOCSTRING SAID NOT TO. `recent`
    holds at most ~1,000 filings, which for an active registrant is five to ten years, so the
    earliest date it reports is the cap and not the archive start. It showed up in the output
    as eight CIKs whose `n` was 1000-1004 -- BNY, DD, DUK, EVRG, MRVL, PSKY, TKO and a TROW
    with 1,232 filings and zero 10-K, which is the recent 13F/13G block of an investment
    adviser hiding a 1990s issuer history behind it. Every `first` in that run was wrong.
    """
    pairs = all_filings(context, doc)
    return pd.DataFrame({"form": [f for f, _ in pairs],
                         "date": [str(d.date()) for _, d in pairs]})


def main() -> None:
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set(PROPOSED)
    _, context = get_config_context("./configs", use_cache=False, save=False)
    context.ensure_edgar_identity()
    rows = []
    for ticker, (ciks, cut, story) in sorted(PROPOSED.items()):
        if ticker not in only:
            continue
        for cik in ciks:
            try:
                doc = submissions(context, cik)
            except Exception as e:                          # noqa: BLE001
                rows.append({"ticker": ticker, "cik": cik, "name": f"FETCH FAILED: {e}"})
                continue
            df = span(context, doc)
            former = "; ".join(f"{f.get('name')} [{str(f.get('from'))[:10]}.."
                               f"{str(f.get('to'))[:10]}]"
                               for f in doc.get("formerNames", []) or [])
            counts = {f: int((df["form"] == f).sum()) for f in FORMS}
            rows.append({
                "ticker": ticker, "cik": cik, "name": doc.get("name"),
                "sic": doc.get("sicDescription", "")[:22],
                "state": doc.get("stateOfIncorporation"),
                "first": df["date"].min() if not df.empty else None,
                "last": df["date"].max() if not df.empty else None,
                "n": len(df), **counts, "cut": cut, "formerNames": former, "story": story,
            })
    out = pd.DataFrame(rows)
    cols = ["ticker", "cik", "name", "state", "first", "last", "n", "10-K", "DEF 14A", "8-K",
            "20-F", "cut"]
    print(out[[c for c in cols if c in out]].to_string(index=False))
    print("\n--- formerNames (the BLK-grade evidence: did the name change AT the cut?) ---")
    for r in rows:
        print(f"  {r.get('ticker'):6} {r.get('cik')}  {str(r.get('formerNames') or '(none)')}")
        if r.get("ticker") in CO_PREDECESSOR:
            print(f"         co-predecessor, NOT a segment: {CO_PREDECESSOR[r['ticker']]}")
    print("\n--- judge each row on three questions ---")
    print("  1. does `last` stop at the cut?  a predecessor still filing years later is a")
    print("     CONTINUING SUBSIDIARY and takes the overlap boundary rule, not predecessor+1")
    print("  2. does `first` reach the ticker's price history?  if not, there is ANOTHER hop")
    print("     behind this one -- the BLK/AVGO/STE shape, invisible to every co-filer test")
    print("  3. is `DEF 14A` > 0?  a debt co-registrant files none, and is not the issuer")
    json.dump(rows, open("reports/validate/registrant/_out/research_verify.json", "w"),
              indent=2, default=str)


if __name__ == "__main__":
    main()
