"""
build_registrant_register.py  (scripts/)
--------------------------------------------------------------------------------------------
Turn `detect_registrant_cutovers --classify` output into PROPOSED register entries, with the
boundary date MEASURED off both registrants' filings rather than guessed.

WHY THE BOUNDARY NEEDS ITS OWN PASS. The detector's truncation anchor is the ticker's first
stored 8-K, which is a symptom, not a seam. Using it as `valid_to` is wrong in the dangerous
direction for any pre-registered shell: Linde plc registered by S-4 on 2017-06-01 for a merger
that completed 2018-10-31, so a 2017-06-01 boundary would assign Praxair's last seventeen
months of 10-Qs to a shell that had not yet acquired anything -- and they would simply
disappear, because the shell filed none.

TWO SHAPES, TWO RULES -- see `boundary_for`. Where the predecessor became a CONTINUING
SUBSIDIARY the seam is the successor's first consolidating filing; where the predecessor was
the public company until it stopped, it is the day after its last one. Using either rule alone
loses filings silently, in opposite directions, and both cases are measured.

⚠ NEITHER RULE IS UNIVERSAL, and there is a third shape they do not cover: both registrants
filing a FULL quarterly series in parallel. APO is the measured case and is hand-adjudicated;
a `WARNING` fires whenever a proposal has that signature. Cadence cannot decide it either
(VTRS's carve-out reports look quarterly) and neither can the comparative period (LIN's seam
falls on the same day as its predecessor's last filing) -- so the honest design is a safe
default plus a loud flag, not a cleverer rule.

⚠ Emits a PROPOSAL. It never writes the live config unless `--merge` is passed: every entry is
a diff a human reads first, because a wrong boundary cannot raise.

    "$PY" scripts/build_registrant_register.py --classified _out/oracle3.json --out proposed.json
    "$PY" scripts/build_registrant_register.py --classified _out/oracle3.json --merge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context                                       # noqa: E402
from src.data_extract.utils.common.registrant import (                           # noqa: E402
    REGISTRANT_CONFIG_FILENAME, REGISTRANT_CONFIG_SUBDIR, load_registrants)

#: The forms that define the seam. A consolidating filing is the one that speaks for the whole
#: business over a period, so the first one the successor makes is the first period it owns.
#: Event forms cannot define a boundary: both registrants file them on both sides of it, which
#: is exactly why they are UNIONed rather than split.
CONSOLIDATING_FORMS = ["10-K", "10-Q"]


def windows(context, cik: str) -> dict:
    """A CIK's consolidating-filing window: every 10-K/10-Q date it ever filed."""
    from edgar import Company

    try:
        filings = list(Company(int(cik)).get_filings(form=CONSOLIDATING_FORMS))
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e), "dates": []}
    dates = sorted(pd.Timestamp(f.filing_date) for f in filings)
    return {"dates": dates, "n": len(dates),
            "first": dates[0] if dates else None, "last": dates[-1] if dates else None}


#: How long a predecessor may keep filing consolidating forms after the successor's first one
#: before it reads as a CONTINUING SUBSIDIARY rather than a pre-merger co-registrant.
#:
#: The two shapes need opposite boundaries and this is what separates them. A shell registered
#: by S-4 files 10-Qs alongside the predecessor for months before the merger completes (Linde
#: plc from 2017-10 for a 2018-10-31 completion, Viatris from 2020-05 for 2020-11-16), and the
#: predecessor is still the public company throughout. A continuing subsidiary files for YEARS
#: after: Apache Corp filed its own 10-K/10-Q until 2024-11-07, 42 months past APA Corp's
#: first, because it retains registered public debt.
SUBSIDIARY_OVERLAP_MONTHS = 18


def boundary_for(context, ticker: str, pred: dict,
                 succ: dict) -> tuple[pd.Timestamp | None, str]:
    """The seam, and the sentence that justifies it.

    ⚠ TWO SHAPES, TWO RULES, AND USING ONE RULE FOR BOTH LOSES FILINGS SILENTLY.

    `succ["first"]` alone is wrong for a pre-merger co-registrant. Linde plc's first 10-Q is
    2017-10-30 but its merger with Praxair completed 2018-10-31, so a 2017-10-30 boundary puts
    Praxair's last twelve months of 10-Qs past the predecessor segment's `valid_to` while the
    successor segment walks only Linde's CIK -- they belong to neither and simply vanish. Same
    for Viatris (first 10-Q 2020-05-07, merger 2020-11-16) against Mylan.

    `pred["last"] + 1` alone is wrong for a continuing subsidiary. Apache Corp filed its own
    10-K/10-Q until 2024-11-07, so that rule would sweep three and a half years of SUBSIDIARY
    statements into the parent's segment -- the blend the dated split exists to prevent.

    So: measure the overlap. A predecessor still filing consolidating forms more than
    `SUBSIDIARY_OVERLAP_MONTHS` past the successor's first is a subsidiary, and the boundary is
    the successor's first filing. Otherwise the predecessor was the public company until it
    stopped, and the boundary is the day after its last filing -- which loses nothing and
    blends nothing, because there is nothing after it to blend.
    """
    if not succ["dates"]:
        return None, ("the successor has filed no 10-K/10-Q, so no seam can be measured -- "
                      "it has not yet reported as the registrant")
    if not pred["dates"]:
        return None, "the predecessor filed no 10-K/10-Q, which contradicts oracle 3"

    overlap_months = (pred["last"] - succ["first"]).days / 30.44
    if overlap_months > SUBSIDIARY_OVERLAP_MONTHS:
        cut = succ["first"]
        kept = [d for d in pred["dates"] if d < cut]
        dropped = [d for d in pred["dates"] if d >= cut]
        if not kept:
            return None, (f"the successor's first 10-K/10-Q ({cut.date()}) precedes every "
                          "predecessor filing -- the two CIKs are the wrong way round")
        return cut, (
            f"boundary = the successor's first consolidating filing, {cut.date()}, because the "
            f"predecessor kept filing 10-K/10-Q for {overlap_months:.0f} months past it "
            f"({dropped[0].date()} .. {dropped[-1].date()}, {len(dropped)} filings) and is "
            f"therefore a CONTINUING SUBSIDIARY. Those {len(dropped)} are correctly excluded "
            f"from the parent's accounts by the dated split and correctly admitted as events "
            f"by the union. The predecessor keeps {len(kept)} filings "
            f"({kept[0].date()} .. {kept[-1].date()})")

    # ⚠ THIS BRANCH IS SAFE BUT NOT ALWAYS RIGHT, AND APO IS THE EXCEPTION. It never loses a
    # predecessor filing, and where the successor ALSO filed for a pre-boundary period it
    # keeps the predecessor's version -- correct for 11 of the 12 measured entries, because
    # the successor's version there is a shell's sparse report (LIN: 3 filings over 12 months)
    # or a carve-out's (VTRS: Upjohn's own 10-K/10-Qs before the Mylan combination).
    #
    # It is WRONG where both registrants filed a FULL quarterly series in parallel. Apollo
    # Asset Management ran 2022-05 .. 2023-08 at a 91-day cadence alongside the new parent's
    # own 92-day series, so this rule kept the SUBSIDIARY's accounts for five quarters and
    # excluded the parent's. No mechanical rule separates the three shapes -- cadence breaks
    # VTRS, the comparative period breaks LIN -- so such an entry is HAND-ADJUDICATED against
    # oracle 4's matched comparative year and the evidence string says so.
    #
    # The check for it: after this proposal lands, compare the successor's pre-boundary filing
    # cadence against a quarterly one. A parent files every quarter; a shell does not.
    cut = pred["last"] + pd.Timedelta(days=1)
    succ_after = [d for d in succ["dates"] if d >= cut]
    succ_before = [d for d in succ["dates"] if d < cut]
    if len(succ_before) >= 4:
        context.log.warning(
            "%s: the successor filed %d consolidating reports BEFORE the proposed boundary "
            "%s (%s .. %s). If that is a full quarterly series it was already the parent and "
            "this boundary is too late -- adjudicate against oracle 4's comparative year, as "
            "APO required.", ticker, len(succ_before), cut.date(),
            succ_before[0].date(), succ_before[-1].date())
    if not succ_after:
        return None, (f"the successor has filed no 10-K/10-Q on or after {cut.date()}, the day "
                      "after the predecessor's last -- so it has not yet reported as the "
                      "registrant and there is no seam to draw")
    return cut, (
        f"boundary = the day after the predecessor's LAST consolidating filing, {cut.date()}. "
        f"It kept filing until {pred['last'].date()}, only {overlap_months:.0f} months past "
        f"the successor's first ({succ['first'].date()}) -- a pre-merger co-registrant overlap, "
        f"not a continuing subsidiary -- so it was the public company throughout and ALL "
        f"{len(pred['dates'])} of its filings are kept ({pred['first'].date()} .. "
        f"{pred['last'].date()}). The successor contributes {len(succ_after)} filings from "
        f"{succ_after[0].date()}; its {len(succ['dates']) - len(succ_after)} earlier filing(s) "
        "are the shell's own pre-merger reports and are deliberately excluded")


def build(context, record: dict) -> dict | None:
    """One proposed two-segment entry, with per-segment measured evidence."""
    pred_cik, succ_cik = record["predecessor_cik"], record["roster_cik"]
    pred, succ = windows(context, pred_cik), windows(context, succ_cik)
    cut, why = boundary_for(context, record["ticker"], pred, succ)
    if cut is None:
        context.log.warning("%s: no boundary -- %s", record["ticker"], why)
        return None

    prof = record.get("predecessor", {})
    o4 = record.get("oracle4")
    o3 = (f"Found by the co-indexed filer scan on {record.get('co_indexed_on')} of the "
          f"successor's registrant-filed documents nearest the boundary. It filed "
          f"{prof.get('n_proxies_before')} proxies of its own before the seam (last "
          f"{prof.get('last_proxy_before')}), so it was the public registrant, and its filing "
          f"rate went {prof.get('n_before_window')} -> {prof.get('n_after_window')} "
          f"({prof.get('rate_ratio')}x) across it.")
    o4s = (f" Chosen over {len(record.get('candidates', [])) - 1} other collapsing "
           f"predecessor(s) by the comparative-column test: {o4['why']}" if o4 else "")

    return {"kind": "reorganisation", "segments": [
        {"cik": pred_cik, "valid_to": str(cut.date()),
         "evidence": f"PROPOSED 2026-09-10. {record.get('predecessor_name')} (CIK {pred_cik}), "
                     f"the pre-boundary registrant. {o3}{o4s} {why}."},
        {"cik": succ_cik, "valid_from": str(cut.date()),
         "evidence": f"PROPOSED 2026-09-10. The successor, CIK {succ_cik}. Its own archive "
                     f"starts {record.get('own_first')} "
                     f"({record.get('own_lead_days')} d before the truncation) with "
                     f"{record.get('own_active_before')} filings older than the "
                     f"pre-registration grace window, and it has filed {succ['n']} 10-K/10-Q "
                     f"from {succ['first'].date()}."}]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="./configs")
    p.add_argument("--classified", required=True, help="detect_registrant_cutovers --out JSON")
    p.add_argument("--out", help="write the proposal here")
    p.add_argument("--merge", action="store_true",
                   help="merge the proposal INTO the live register (a risk-zone write)")
    p.add_argument("-t", "--tickers", default="", help="restrict to these")
    args = p.parse_args(argv)

    _, context = get_config_context(config_path=args.config, use_cache=False, save=False)
    context.ensure_edgar_identity()
    records = json.loads(Path(args.classified).read_text(encoding="utf-8"))
    only = {t for t in args.tickers.split(",") if t}
    existing = load_registrants(str(context.config_dir))

    proposed: dict[str, dict] = {}
    for r in records:
        if r["cls"] != "cutover" or (only and r["ticker"] not in only):
            continue
        if r["ticker"] in existing:
            context.log.info("%s: already registered, left alone", r["ticker"])
            continue
        entry = build(context, r)
        if entry:
            proposed[r["ticker"]] = entry
            print(f"{r['ticker']:6} {entry['segments'][0]['cik']} -> "
                  f"{entry['segments'][1]['cik']}  at {entry['segments'][1]['valid_from']}")

    print(f"\n{len(proposed)} proposed entr(y|ies)")
    if args.out:
        Path(args.out).write_text(json.dumps(proposed, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}")

    if args.merge:
        path = Path(context.config_dir) / REGISTRANT_CONFIG_SUBDIR / REGISTRANT_CONFIG_FILENAME
        blob = json.loads(path.read_text(encoding="utf-8"))
        clash = sorted(set(proposed) & set(blob))
        if clash:
            raise SystemExit(f"refusing to overwrite existing entries: {clash}")
        blob.update(proposed)
        # `_README` first, then tickers alphabetically -- a stable order so a later merge is a
        # clean diff rather than a reshuffle.
        ordered = {"_README": blob["_README"],
                   **{k: blob[k] for k in sorted(k for k in blob if not k.startswith("_"))}}
        path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"merged into {path}")
        # Re-read through the loader: a merge that produces a register the validator refuses
        # must fail HERE, not in tonight's extraction run.
        load_registrants.__wrapped__ if hasattr(load_registrants, "__wrapped__") else None
        from src.data_extract.utils.common.registrant import _registrants_at
        _registrants_at.cache_clear()
        print(f"validated: {len(load_registrants(str(context.config_dir)))} entries load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
