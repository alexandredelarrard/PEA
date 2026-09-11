"""
detect_registrant_cutovers.py  (scripts/)
--------------------------------------------------------------------------------------------
Find every registrant boundary in the universe, classify what is NOT one, and propose the
register entry. Re-runnable, and non-zero on an unclassified candidate so it can stand as a
nightly check -- the class is silent by construction, so a one-off repair of ~39 tickers
decays unless the next reorganisation announces itself.

THE INSIGHT THE WHOLE DETECTOR RESTS ON: prices follow the ECONOMIC entity, filings follow
the LEGAL registrant. `yfinance` splices Mylan's history into VTRS; EDGAR does not. So a long
price history against a short filing archive *implies* a registrant change -- and that same
asymmetry auto-excludes genuine recent IPOs (COIN, HOOD, CRWD, DDOG), whose price history is
short too. A screen on filing length alone would flag every young company in the index.

FOUR ORACLES, CHEAPEST FIRST. The early ones NOMINATE; only the last CONFIRMS.

  1  DB screen        free, no network   Two signatures, because a boundary before the 2009
                                         bulk-dataset floors leaves a different trace than one
                                         after it.
  2  sharadar_actions free, no network   Names the shell outright: "CORVETTEPORSCHE CORP" is
                                         not a company, it is the Conoco/Phillips merger's
                                         internal codename. ⚠ Suggestive, never conclusive --
                                         Sharadar records a name change whether or not the CIK
                                         moved, which is why IVZ (AMVESCAP plc -> Invesco, one
                                         continuous CIK since 1994) appears here and is NOT a
                                         cutover.
  3  co-indexed scan  ~45 requests each  Some filings are indexed under BOTH registrants at a
                                         reorganisation, so the successor's own headers name
                                         its predecessor. Two gates then separate a
                                         predecessor from a subsidiary co-registrant: it filed
                                         its own proxies, and its filing rate did not rise
                                         across the boundary.
  4  comparative col  ~1 request/cand.   WHOSE P&L IS THE SUCCESSOR'S PRIOR-YEAR COMPARATIVE
                                         COLUMN? The successor's first 10-K restates ONE
                                         predecessor's history as its own -- the accounting
                                         acquirer, the successor in substance, and exactly the
                                         entity whose price series the ticker continues.

⚠ ORACLE 4 CONFIRMS EVERY ENTRY, NOT ONLY THE CONTESTED ONES. Oracles 1-3 cannot tell a
predecessor from a company the successor ACQUIRED: an absorbed target also stops filing and
also filed its own proxies, so with no competitor it wins by default. ORCL resolved to `PORTAL
SOFTWARE INC`, a 2006 Oracle acquisition, on exactly that path. A candidate oracle 4 cannot
confirm is recorded `unconfirmed_cutover` -- nominated, evidenced, and NOT written.

⚠ IT ABSTAINS BEFORE 2009, STRUCTURALLY. XBRL does not exist earlier, so a pre-2009 boundary
has no comparative facts to read and never will. Those tickers stay `ambiguous` or
`unconfirmed_cutover` for hand adjudication. That is the correct outcome: picking the wrong
one of two predecessors attaches another company's accounts to the ticker on every
consolidating form, which is the failure the register exists to prevent.

⚠ THE FPI TEST RUNS FIRST, AND THAT ORDERING MATTERS. A foreign private issuer files 20-F/6-K
under ONE continuous CIK and then transitions to domestic forms; that looks exactly like a
truncation and is not one. The shell-name oracle must never outrank the CIK-continuity test.

⚠ ONE EDGAR WALK AT A TIME. The rate limiter is per-PROCESS, so two concurrent walks put ~18
req/s against SEC's limit of 10; the block is silent and it once cost 18 roster managers their
entire book. Oracle 3 is deliberately serial and there is no `--workers`.

Exits non-zero while any candidate is unresolved, ambiguous or unconfirmed, so it can stand as
a nightly check: this defect class is silent by construction, and a one-off repair decays
unless the next reorganisation announces itself.

    "$PY" scripts/detect_registrant_cutovers.py --offline
    "$PY" scripts/detect_registrant_cutovers.py --classify --out o3.json --report classified.md
    "$PY" scripts/detect_registrant_cutovers.py --offline --out o3.json --report classified.md
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context                                       # noqa: E402
from src.data_extract.utils.common.registrant import load_registrants            # noqa: E402
from src.data_extract.utils.common.sec_utils import sec_get                      # noqa: E402

#: A filing archive starting more than this many years after the first price is the symptom
#: every candidate shares. Four years is loose enough to catch a 2019 boundary on a 1995 price
#: history and tight enough to exclude a 2021 IPO.
LAG_YEARS = 4.0

#: Screen 1 keys on the registrant-scoped tables agreeing with each other. 400 days because a
#: cutover truncates the 8-K, the notes and the facts to within a reporting cycle of one date.
SPREAD_DAYS = 400

#: The bulk data sets start in 2009, so a boundary older than that leaves `notes`/`facts`
#: sitting at the floor carrying no signal. Screen 1 only claims tickers above it.
BULK_FLOOR = "2009-06-01"

#: Screen 2's corroboration window: for an older boundary, what still moves together is the
#: 8-K and the PROXY, because both are registrant-scoped. Asymmetric because a successor's
#: first proxy follows its first 8-K by up to an annual-meeting cycle but can also precede it.
PROXY_BEFORE_DAYS, PROXY_AFTER_DAYS = 200, 550

#: `sharadar_actions` rows whose `contraname` contains one of these is a shell, not a company.
#: The evidence is the WORD, not the match: "TWDC HOLDCO 613 CORP" is not a business.
SHELL_TOKENS = ("HOLDCO", "HOLDING", "PARENT", "MERGER SUB", "MERGERSUB", "NEWCO", "REGCO")

#: The forms that make a filer a foreign private issuer. Four or more of them BEFORE the
#: truncation means the same CIK was filing all along under a different form family -- the
#: repair is form coverage, not registrant resolution, and this phase must not add an entry.
FPI_FORMS = ("20-F", "40-F", "6-K")
FPI_MIN_FILINGS = 4

#: How early a shell may legitimately be registered before the boundary it was created for.
#: A merger's S-4 goes in well ahead of completion -- Linde plc 517 days, DowDuPont 560,
#: PSKY 316 -- so anything inside this window is normal for a real cutover and proves nothing.
#: Filings OLDER than this are what a cutover cannot have.
PRE_REGISTRATION_DAYS = 600

#: Filings the current CIK made before that window, above which it was plainly the registrant
#: all along and no boundary exists. A pre-registered shell files a handful of S-4/A and
#: correspondence; a continuous registrant files hundreds.
CONTINUOUS_MIN_FILINGS = 25

#: Oracle 3 reads this many of the successor's filings' headers, the ones NEAREST the boundary.
#:
#: ⚠ NEAREST THE BOUNDARY, NOT FIRST-RETURNED, AND THAT IS THE WHOLE ORACLE. `get_filings()`
#: yields newest-first, so "the first 40" are a large company's most RECENT 40 -- for any S&P
#: name are SC 13G and Form 4 filings made ABOUT it by institutions; measured 2026-09-10 that
#: returned Vanguard, FMR, State Street, Norges Bank or JPMorgan as the "predecessor" of
#: eleven different tickers. Taking the OLDEST 40 instead fixes those but breaks PSKY, whose
#: successor registered by S-4 ten months before the merger it co-indexes on. DISTANCE FROM
#: THE BOUNDARY is the property both cases actually share: co-indexing under both registrants
#: happens at the reorganisation -- the 8-K12B that registers the successor's securities, the
#: S-4, the first jointly-filed 10-Q.
ORACLE3_MAX_HEADERS = 40

#: Forms whose header filer is the REGISTRANT itself. Everything else -- SC 13D/G, Forms 3/4/5,
#: 13F, 144, PX14A6G -- is filed about the company by somebody else, and its filer list names
#: that somebody. An allowlist rather than a blocklist: a new third-party form type must not
#: silently poison the scan.
REGISTRANT_FORM_PREFIXES = ("10-", "8-K", "S-", "DEF", "PRE", "PRR", "424", "425", "POS",
                            "20-F", "40-F", "6-K", "11-K", "ARS", "N-", "18-K",
                            "SC TO", "SC 14D")

#: A candidate co-filer is only a PREDECESSOR if it was itself a public registrant, and the
#: cheap proof is that it filed its own proxy. A subsidiary debt co-registrant --
#: `NBCUniversal Media, LLC`, `Duke Energy Carolinas, LLC`, `Bunge Ltd Finance Corp` -- files
#: 10-K/10-Q jointly with the parent for decades and files NO proxy, so it passes every
#: filing-count test and fails this one. Measured 2026-09-10: this is what separates the two.
PROXY_FORMS = ("DEF 14A", "DEF 14C", "DEFC14A", "DEFM14A")

#: The predecessor test, measured over a SYMMETRIC window either side of the boundary: a
#: predecessor's filing rate COLLAPSES there, because it stopped being the registrant.
#:
#: A share-of-whole-archive test does not work, and PSKY is why. Paramount Skydance ran a
#: hostile tender for Warner Bros. Discovery weeks after its own reorganisation, so WBD is
#: co-indexed on its filings and -- being a large company whose two decades of history mostly
#: predate 2025 -- passes any "most of its filings are before the boundary" test and outranks
#: the real predecessor on archive size. What separates them is the RATE: Paramount Global
#: stopped filing at the boundary, WBD did not miss a quarter.
#:
#: Apache Corp sets the floor on how strict this can be. It kept filing 10-K/10-Q for 3.7 years
#: after APA Corp became the parent because it retains registered public debt, so the test
#: cannot be "stops dead" -- it is "the rate at least halves".
PREDECESSOR_WINDOW_DAYS = 730
PREDECESSOR_MIN_BEFORE = 3

#: ⚠ THIS IS A CANDIDACY FILTER, NOT A TIE-BREAK, and the distinction was learned the hard way.
#: Set at 0.5 it silently DECIDED merger-of-equals cases by accident: COR's real accounting
#: acquirer, AmeriSource Health, scored 0.53 and was rejected, handing the answer to Bergen
#: Brunswig on a margin of 0.03. A threshold that close to two legitimate candidates is not
#: measuring which one is the predecessor, it is measuring itself.
#:
#: So it is loosened to 1.0 -- "it did not ACCELERATE across the boundary", which is all the
#: rate can honestly tell you -- and every candidate that clears it goes to oracle 4, which
#: reads the answer off the comparative column. Still rejects what it must: Warner Bros.
#: Discovery on PSKY's tender (1.42), Commonwealth Edison under EXC (1.44), Comcast MO Group
#: (2.00). The four register predecessors clear it with room: Apache 0.129, Eaton 0.241,
#: Google 0.058, Exxon Mobil 0.045, measured 2026-09-10.
PREDECESSOR_MAX_RATE_RATIO = 1.0

#: Oracle 4's comparison tags: the two line items every filer states and every first 10-K
#: carries a comparative column for. Revenue alone is not enough -- a bank or an insurer may
#: tag revenue half a dozen ways -- so net income, which is unambiguous, is in the set too.
COMPARATIVE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet", "NetIncomeLoss", "ProfitLoss",
                    "NetIncomeLossAvailableToCommonStockholdersBasic")

#: Relative tolerance on a comparative match. Not zero: the successor may restate a
#: predecessor's figure for a reclassification or a discontinued operation while it is still
#: unmistakably the same company's year. Tight enough that the OTHER predecessor's genuinely
#: different revenue cannot slip inside it.
COMPARATIVE_TOLERANCE = 0.02

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


# --------------------------------------------------------------------------------------- #
# Oracle 1 -- the DB screens                                                               #
# --------------------------------------------------------------------------------------- #
_SCREEN_CTES = """
WITH px AS (SELECT ticker, min(date)::date AS first_px FROM prices GROUP BY 1),
     k8 AS (SELECT ticker, min(filing_date)::date AS d FROM sec_8k GROUP BY 1),
     nt AS (SELECT ticker, min(filed)::date AS d FROM notes_text GROUP BY 1),
     ff AS (SELECT ticker, min(filing_date)::date AS d FROM fundamentals_facts GROUP BY 1),
     dl AS (SELECT ticker, min(as_of)::date AS d FROM def14a_llm GROUP BY 1)
"""

SCREEN_TIGHT = _SCREEN_CTES + f"""
SELECT p.ticker, p.first_px::text, k8.d::text AS first_8k, nt.d::text AS first_notes,
       ff.d::text AS first_facts,
       (GREATEST(k8.d, nt.d, ff.d) - LEAST(k8.d, nt.d, ff.d)) AS spread_days,
       round(((LEAST(k8.d, nt.d, ff.d) - p.first_px) / 365.25)::numeric, 2) AS lag_y
FROM px p JOIN k8 USING(ticker) JOIN nt USING(ticker) JOIN ff USING(ticker)
WHERE (GREATEST(k8.d, nt.d, ff.d) - LEAST(k8.d, nt.d, ff.d)) <= {SPREAD_DAYS}
  AND (LEAST(k8.d, nt.d, ff.d) - p.first_px) / 365.25 > {LAG_YEARS}
  AND nt.d > DATE '{BULK_FLOOR}'
ORDER BY lag_y DESC
"""

SCREEN_PROXY = _SCREEN_CTES + f"""
SELECT p.ticker, p.first_px::text, k8.d::text AS first_8k, dl.d::text AS first_proxy,
       NULL::int AS spread_days,
       round(((k8.d - p.first_px) / 365.25)::numeric, 2) AS lag_y
FROM px p JOIN k8 USING(ticker) JOIN dl USING(ticker)
WHERE (k8.d - p.first_px) / 365.25 > {LAG_YEARS}
  AND dl.d > k8.d - {PROXY_BEFORE_DAYS} AND dl.d < k8.d + {PROXY_AFTER_DAYS}
ORDER BY lag_y DESC
"""

#: Every ticker whose 8-K starts >4 y after its first price -- the 62-candidate population the
#: two screens are drawn from, and the denominator every count in the report is quoted against.
SCREEN_ALL_LATE = _SCREEN_CTES + f"""
SELECT p.ticker, p.first_px::text, k8.d::text AS first_8k,
       round(((k8.d - p.first_px) / 365.25)::numeric, 2) AS lag_y
FROM px p JOIN k8 USING(ticker)
WHERE (k8.d - p.first_px) / 365.25 > {LAG_YEARS}
ORDER BY lag_y DESC
"""


def oracle1(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The tight cluster, the proxy corroboration, and the full late-8-K population.

    ⚠ A ticker ALREADY IN THE REGISTER should not appear in either screen, and that is the
    detector working rather than a gap: APA, ETN and GOOGL show `fundamentals_facts` at the
    2009 floor precisely because their entries repaired that leg. A screen that still flagged
    them would be measuring the roster, not the data.
    """
    return (pd.DataFrame(conn.execute(text(SCREEN_TIGHT)).mappings().all()),
            pd.DataFrame(conn.execute(text(SCREEN_PROXY)).mappings().all()),
            pd.DataFrame(conn.execute(text(SCREEN_ALL_LATE)).mappings().all()))


# --------------------------------------------------------------------------------------- #
# Oracle 2 -- sharadar_actions                                                             #
# --------------------------------------------------------------------------------------- #
ACTIONS_SQL = """
SELECT ticker, date::text AS date, action, contraname
FROM sharadar_actions
WHERE action IN ('namechangefrom', 'tickerchangefrom') AND ticker = ANY(:tickers)
ORDER BY ticker, date
"""


def oracle2(conn, tickers: list[str], anchor: dict[str, str]) -> dict[str, list[dict]]:
    """Name/ticker changes within +-400 d of each candidate's truncation date.

    ⚠ SUGGESTIVE, NEVER CONCLUSIVE, and the reason is IVZ: `namechangefrom AMVESCAP PLC` is a
    real Sharadar action, and CIK 914208 nonetheless filed continuously from 1994. Sharadar
    records a name change whether or not the CIK moved. Absence proves nothing either -- BLK's
    2024 reorganisation has no Sharadar action at all -- so oracle 3 runs for every candidate.
    """
    rows = conn.execute(text(ACTIONS_SQL), {"tickers": tickers}).mappings().all()
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        cut = anchor.get(r["ticker"])
        if cut is None:
            continue
        if abs((pd.Timestamp(r["date"]) - pd.Timestamp(cut)).days) > SPREAD_DAYS:
            continue
        name = str(r["contraname"] or "").upper()
        out[r["ticker"]].append({**dict(r),
                                 "is_shell": any(tok in name for tok in SHELL_TOKENS)})
    return dict(out)


# --------------------------------------------------------------------------------------- #
# Oracle 3 -- submissions continuity, then the co-indexed filer scan                       #
# --------------------------------------------------------------------------------------- #
def submissions(context, cik: str) -> dict:
    """One `data.sec.gov/submissions` document -- the cheapest complete view of a CIK's own
    archive (form list + filing dates, one request)."""
    return sec_get(context, SUBMISSIONS_URL.format(cik=str(cik).zfill(10))).json()


def continuity(context, doc: dict, truncation: str) -> dict:
    """Does the CURRENT CIK's own archive predate the truncation, and how?

    ⚠ MEASURED OVER THE WHOLE ARCHIVE, NOT THE `recent` BLOCK. `recent` holds at most ~1,000
    filings, which for an active name is five to ten years -- so counting a 1990s FPI history
    inside it returns zero and an FPI reads as a cutover. IVZ and RCL are the measured cases:
    both have one continuous CIK since the mid-1990s and both were misclassified until this
    counted the older pages too.

    Three numbers, and it takes all three:

      `own_first`     the CIK's earliest filing. Close to the truncation means the registrant
                      did not exist before the boundary.
      `own_active_before`
                      how many filings it made more than `PRE_REGISTRATION_DAYS` before the
                      truncation. This is what separates "the registrant was already here"
                      from "a shell was registered early", and `own_first` alone cannot:
                      Linde plc registered by S-4 517 days before its merger completed and
                      PSKY's shell 316 days, both entirely normal, while IVZ's CIK had been
                      filing continuously for thirteen years.
      `fpi_before`    20-F / 40-F / 6-K filings before the truncation. Four or more means one
                      continuous CIK filing under a different form family -- the repair is
                      form coverage, not registrant resolution.
    """
    pairs = all_filings(context, doc)
    cut = pd.Timestamp(truncation)
    own_first = min((d for _, d in pairs), default=None)
    grace = cut - pd.Timedelta(days=PRE_REGISTRATION_DAYS)
    return {"own_first": None if own_first is None else str(own_first.date()),
            "fpi_before": sum(1 for f, d in pairs if f in FPI_FORMS and d < cut),
            "own_active_before": sum(1 for _, d in pairs if d < grace),
            "n_total": len(pairs),
            "n_archive_pages": len(doc.get("filings", {}).get("files", [])),
            "own_lead_days": None if own_first is None else int((cut - own_first).days)}


def co_indexed(cik: str, truncation: str,
               limit: int = ORACLE3_MAX_HEADERS) -> list[tuple[str, str, int]]:
    """Distinct (name, cik) appearing as a FILER on the successor's registrant-filed documents
    NEAREST THE BOUNDARY, most common first, with the successor itself removed.

    Two filters carry this, and both were learned by measuring (see `ORACLE3_MAX_HEADERS` and
    `REGISTRANT_FORM_PREFIXES`): the scan is centred on the boundary, because that is where a
    document gets indexed under both registrants, and it skips forms filed about the company
    by third parties, because their header names the third party, not the registrant.

    Sorting costs nothing extra -- `get_filings()` reads the cached submissions index, and the
    requests are the per-filing header reads that follow.
    """
    from edgar import Company

    cut = pd.Timestamp(truncation)
    filings = [f for f in Company(int(cik)).get_filings()
               if str(f.form).startswith(REGISTRANT_FORM_PREFIXES)]
    filings.sort(key=lambda f: abs((pd.Timestamp(f.filing_date) - cut).days))

    seen: collections.Counter = collections.Counter()
    for f in filings[:limit]:
        try:
            header = f.header
            parties = list(header.filers) + list(header.subject_companies)
        except Exception:                                   # noqa: BLE001 -- one bad header
            continue
        # `subject_companies`, not `filers` alone. On a tender offer or a proxy-solicitation
        # form the counterparty is the SUBJECT, not a co-filer, and PSKY is the case that
        # proves it: its successor registered as the shell "New Pluto Global, Inc." whose S-4
        # lists only itself, so a filers-only scan returns nothing for a boundary that is real.
        for x in parties:
            info = getattr(x, "company_information", None)
            if info is None or not getattr(info, "cik", None):
                continue
            seen[(str(info.name), str(info.cik).zfill(10))] += 1
    # Deduped BY CIK, not by (name, CIK). A registrant that changed its name appears under
    # both in different filings' headers -- CIK 0001364742 is "BlackRock Inc." and "BlackRock
    # Finance, Inc." -- and counting them separately turned BLK into a two-candidate
    # ambiguity that does not exist. Aliases are kept because they are evidence of the rename.
    target = str(cik).zfill(10)
    by_cik: dict[str, dict] = {}
    for (name, c), n in seen.most_common():
        if c == target:
            continue
        entry = by_cik.setdefault(c, {"name": name, "n": 0, "aliases": []})
        entry["n"] += n
        if name != entry["name"]:
            entry["aliases"].append(name)
    return [(e["name"], c, e["n"]) for c, e in
            sorted(by_cik.items(), key=lambda kv: -kv[1]["n"])]


def all_filings(context, doc: dict) -> list[tuple[str, pd.Timestamp]]:
    """(form, date) for a CIK's WHOLE archive: the inlined `recent` block plus every older
    page the submissions document references.

    The pages matter. `recent` holds at most ~1,000 filings, which for an active S&P name is
    five to ten years -- not enough to see either side of a 2001 boundary, and a window that
    silently starts empty reads as "stopped filing" for every old candidate. One request per
    page, and most CIKs have none.
    """
    out: list[tuple[str, pd.Timestamp]] = []
    blocks = [doc.get("filings", {}).get("recent", {})]
    for page in doc.get("filings", {}).get("files", []):
        name = page.get("name")
        if not name:
            continue
        try:
            blocks.append(sec_get(context, SUBMISSIONS_ARCHIVE_URL.format(name=name)).json())
        except Exception:                                   # noqa: BLE001 -- one page
            continue
    for block in blocks:
        for form, date in zip(block.get("form", []), block.get("filingDate", [])):
            if date:
                out.append((str(form), pd.Timestamp(date)))
    return out


def predecessor_profile(context, cik: str, truncation: str) -> dict:
    """Is this co-filer a PREDECESSOR, a SUBSIDIARY, or an unrelated counterparty?

    Two tests, both necessary:

      it filed its own PROXIES before the boundary -- it was a public registrant. A subsidiary
      debt co-registrant (`NBCUniversal Media, LLC`, `Duke Energy Carolinas, LLC`, `Bunge Ltd
      Finance Corp`) files 10-K/10-Q jointly with its parent for decades and files no proxy,
      so it passes every count test and fails this one.

      its filing RATE collapsed at the boundary -- it stopped being the registrant. An
      unrelated counterparty (Warner Bros. Discovery on PSKY's tender offer, General Electric
      on Baker Hughes's reorganisation) keeps filing at the same rate and fails this one.
    """
    try:
        doc = submissions(context, cik)
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "why": f"submissions failed: {e}"}
    pairs = all_filings(context, doc)
    cut = pd.Timestamp(truncation)
    window = pd.Timedelta(days=PREDECESSOR_WINDOW_DAYS)
    before_w = [d for _, d in pairs if cut - window <= d < cut]
    after_w = [d for _, d in pairs if cut <= d < cut + window]
    proxies = [d for f, d in pairs if f in PROXY_FORMS and d < cut]
    ratio = len(after_w) / max(len(before_w), 1)

    if not proxies:
        why = "no proxy of its own before the boundary -- never the public registrant"
    elif len(before_w) < PREDECESSOR_MIN_BEFORE:
        why = (f"only {len(before_w)} filings in the {PREDECESSOR_WINDOW_DAYS} d before the "
               "boundary -- not an active registrant at the time")
    elif ratio > PREDECESSOR_MAX_RATE_RATIO:
        why = (f"filing rate ROSE across the boundary: {len(before_w)} before / {len(after_w)} "
               f"after ({ratio:.2f}x) -- an entity that gained filings where this ticker's "
               "predecessor lost them, so a counterparty or an acquirer, not a predecessor")
    else:
        why = (f"{len(proxies)} proxies of its own before the boundary (last "
               f"{max(proxies).date()}); filing rate collapsed {len(before_w)} -> "
               f"{len(after_w)} ({ratio:.2f}x) across it")
    return {"ok": bool(proxies) and len(before_w) >= PREDECESSOR_MIN_BEFORE
            and ratio <= PREDECESSOR_MAX_RATE_RATIO,
            "n_total": len(pairs), "n_before_window": len(before_w),
            "n_after_window": len(after_w), "rate_ratio": round(ratio, 3),
            "n_proxies_before": len(proxies),
            "last_proxy_before": str(max(proxies).date()) if proxies else None, "why": why}


def companyfacts(context, cik: str) -> dict:
    """One `data.sec.gov/api/xbrl/companyfacts` document -- every XBRL fact a CIK ever
    reported, with each fact's period and the accession that reported it. One request."""
    return sec_get(context, COMPANYFACTS_URL.format(cik=str(cik).zfill(10))).json()


def _annual_facts(doc: dict, tags: tuple[str, ...]) -> dict[tuple[str, str], float]:
    """`{(tag, period_end): value}` for the FY duration facts of `tags`, one CIK.

    Restricted to `fp == "FY"` annual durations because that is the grain a first 10-K's
    comparative column is stated on, and comparing a quarter to a year would match nothing
    while looking like a disagreement.
    """
    out: dict[tuple[str, str], float] = {}
    for tag in tags:
        for unit_facts in doc.get("facts", {}).get("us-gaap", {}).get(tag, {}) \
                             .get("units", {}).values():
            for f in unit_facts:
                if f.get("fp") != "FY" or not f.get("start") or not f.get("end"):
                    continue
                out.setdefault((tag, f["end"]), float(f["val"]))
    return out


def oracle4_comparative(context, successor_cik: str, candidates: list[dict],
                        truncation: str) -> tuple[dict, dict] | None:
    """WHOSE P&L IS THE SUCCESSOR'S PRIOR-YEAR COMPARATIVE COLUMN?

    The test that settles a merger of equals, and the only one that can. When two public
    companies combine under a new holding company both predecessors stop filing at the same
    date, so every activity- and proxy-based test in oracle 3 passes for each of them. But
    accounting has already answered the question: the successor's first 10-K restates ONE
    predecessor's history as its own comparative periods -- the accounting acquirer, the
    entity that is the successor in substance -- and reports the other only from the
    acquisition date forward.

    That is exactly the entity whose price series the ticker continues, which is what makes it
    the right predecessor for THIS register: the register exists to make a ticker's filings
    reach as far back as its prices, and the comparative column is the filer's own statement
    of how far back that is.

    So: pull the successor's pre-boundary FY revenue and net income, pull each candidate's own
    reported values for the same fiscal year ends, and see which one the successor restated.

    Returns `(winner, evidence)`, or `None` when no candidate matches -- in which case the
    ticker is recorded `ambiguous` and adjudicated by hand rather than guessed.
    """
    cut = pd.Timestamp(truncation)
    try:
        succ = _annual_facts(companyfacts(context, successor_cik), COMPARATIVE_TAGS)
    except Exception as e:                                  # noqa: BLE001
        return None if not _log_o4(context, f"successor companyfacts failed: {e}") else None

    # Only the periods that ENDED before the boundary: those are comparatives the successor
    # inherited rather than results it earned.
    inherited = {k: v for k, v in succ.items() if pd.Timestamp(k[1]) < cut}
    if not inherited:
        return None

    scored: list[tuple[int, int, dict, dict]] = []
    for cand in candidates:
        try:
            own = _annual_facts(companyfacts(context, cand["cik"]), COMPARATIVE_TAGS)
        except Exception:                                   # noqa: BLE001 -- one candidate
            continue
        shared = set(inherited) & set(own)
        hits = [k for k in shared
                if abs(inherited[k] - own[k]) <= COMPARATIVE_TOLERANCE * max(
                    abs(inherited[k]), abs(own[k]), 1.0)]
        if hits:
            example = sorted(hits)[-1]
            scored.append((len(hits), len(shared), cand,
                           {"matched": len(hits), "compared": len(shared),
                            "example_tag": example[0], "example_period": example[1],
                            "example_value": inherited[example],
                            "why": f"the successor's FY{example[1][:4]} {example[0]} of "
                                   f"{inherited[example]:,.0f} is {cand['name']}'s own "
                                   f"reported figure ({len(hits)} of {len(shared)} shared "
                                   "annual facts agree), so that entity is the accounting "
                                   "acquirer and the successor in substance."}))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1]))
    # A tie on match count is not a decision. Two candidates matching equally means the facts
    # do not separate them, and inventing a preference here would be exactly the silent guess
    # the `ambiguous` class exists to prevent.
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][2], scored[0][3]


def _log_o4(context, msg: str) -> bool:
    context.log.warning("oracle 4: %s", msg)
    return False


def oracle3(context, ticker: str, cik: str, truncation: str) -> dict:
    """Classify ONE candidate. Serial by construction -- see the module docstring."""
    try:
        doc = submissions(context, cik)
    except Exception as e:                                  # noqa: BLE001
        return {"ticker": ticker, "cls": "unresolved", "why": f"submissions failed: {e}"}

    cont = continuity(context, doc, truncation)
    if cont["fpi_before"] >= FPI_MIN_FILINGS:
        return {"ticker": ticker, "cls": "foreign_private_issuer", **cont,
                "why": f"CIK {cik} filed {cont['fpi_before']} 20-F/40-F/6-K documents before "
                       f"{truncation} under one continuous registrant. The repair is form "
                       "coverage, not registrant resolution. NO REGISTER ENTRY."}

    # The current CIK was ALREADY FILING long before the truncation, so no registrant changed
    # and the late archive has some other cause -- a sparse pre-2004 8-K record being the
    # common one. Runs before the co-indexed scan because that scan would happily nominate
    # some counterparty for a ticker that has no boundary at all.
    #
    # ⚠ It tests ACTIVITY, not existence. `own_first` alone cannot do this job: a shell is
    # routinely registered by S-4 a year or more before its merger completes (Linde 517 d,
    # DowDuPont 560 d, PSKY 316 d), so an early first filing is normal for a real cutover.
    # What no cutover has is a decade of the successor's own filings before the boundary.
    if cont["own_active_before"] >= CONTINUOUS_MIN_FILINGS:
        return {"ticker": ticker, "cls": "continuous_registrant", **cont,
                "why": f"CIK {cik} filed {cont['own_active_before']} documents of its own more "
                       f"than {PRE_REGISTRATION_DAYS} d before {truncation} (archive starts "
                       f"{cont['own_first']}, {cont['own_lead_days']} d before it). No "
                       "registrant changed; the late archive has another cause -- most often "
                       "a sparse pre-2004 8-K record. NO REGISTER ENTRY."}

    try:
        peers = co_indexed(cik, truncation)
    except Exception as e:                                  # noqa: BLE001 -- one candidate
        return {"ticker": ticker, "cls": "unresolved", **cont,
                "why": f"co-indexed scan failed: {e}"}

    # Every co-filer is a CANDIDATE predecessor; `predecessor_profile` is what decides. Ranked
    # by pre-boundary filing count so the real registrant outranks an incidental co-filer, and
    # the rejected ones are carried in the result -- a candidate rejected for a stated reason
    # is evidence, while one silently dropped is indistinguishable next quarter from one never
    # examined.
    rejected: list[dict] = []
    scored: list[tuple[int, str, str, int, dict]] = []
    for name, pred, n in peers:
        prof = predecessor_profile(context, pred, truncation)
        if prof.get("ok"):
            # Ranked by how hard the rate collapsed, THEN by pre-boundary activity.
            # Archive size is the wrong key: it hands the answer to the largest company
            # in the room, which is how PSKY resolved to Warner Bros. Discovery.
            scored.append((prof["rate_ratio"], -prof["n_before_window"],
                           name, pred, n, prof))
        else:
            rejected.append({"name": name, "cik": pred, "co_indexed_on": n,
                             "why": prof.get("why")})
    scored.sort()
    candidates = [{"name": name, "cik": pred, "co_indexed_on": n, "profile": prof}
                  for _, _, name, pred, n, prof in scored]

    # ⚠ ORACLE 4 CONFIRMS EVERY ENTRY -- IT IS NOT ONLY A TIE-BREAK, and ORCL is why.
    #
    # A merger of equals collapses BOTH predecessors, so oracle 3 passes for each and ranking
    # cannot choose: Cigna and Express Scripts both stopped at CI's 2018 boundary, as did Duke
    # Energy and Cinergy at DUK's. That much was expected. What was not is the SINGLE-candidate
    # failure: a company the successor ACQUIRED also stops filing and also filed its own
    # proxies, so it passes every gate alone and wins by default. ORCL resolved to `PORTAL
    # SOFTWARE INC` -- a 2006 Oracle acquisition -- with no competitor to expose it.
    #
    # An absorbed target and a predecessor differ in exactly one observable place: whose
    # history the successor restated as its own comparatives. So oracle 4 runs whenever there
    # is a candidate at all, and a candidate it cannot confirm is NOMINATED, never landed.
    oracle4 = None
    winner = None
    if candidates:
        chosen = oracle4_comparative(context, cik, candidates, truncation)
        if chosen is not None:
            winner, oracle4 = chosen

    if winner is None and candidates:
        # Oracle 4 abstained. Before 2009 that is structural -- XBRL does not exist, so COP
        # (2002), DUK (2006), ORCL (2006) and EXC (2000) have no comparative facts to read and
        # never will. After 2009 it means the candidate's numbers are NOT the successor's
        # comparatives, which is positive evidence AGAINST it.
        cls = "ambiguous" if len(candidates) > 1 else "unconfirmed_cutover"
        return {"ticker": ticker, "cls": cls, **cont, "candidates": candidates,
                "rejected_co_filers": rejected,
                "why": f"{len(candidates)} candidate predecessor(s) collapsed at {truncation} "
                       f"({', '.join(c['name'] for c in candidates)}), and the "
                       "comparative-column test could not confirm any of them -- structural "
                       "before 2009, when XBRL did not exist. NOMINATED, NOT CONFIRMED: needs "
                       "adjudication before a register entry, because an ACQUIRED company "
                       "also stops filing and also filed its own proxies."}

    if winner is not None:
        name, pred, n, prof = (winner["name"], winner["cik"],
                               winner["co_indexed_on"], winner["profile"])
        return {"ticker": ticker, "cls": "cutover", **cont, "predecessor_cik": pred,
                "predecessor_name": name, "co_indexed_on": n, "predecessor": prof,
                "candidates": candidates, "oracle4": oracle4,
                "rejected_co_filers": rejected,
                "why": f"co-indexed with {name} (CIK {pred}) on {n} of the successor's "
                       f"registrant-filed documents nearest the boundary; it filed "
                       f"{prof['n_proxies_before']} proxies of its own before {truncation} "
                       f"(last {prof['last_proxy_before']}) so it was the public registrant, "
                       f"and its filing rate collapsed {prof['n_before_window']} -> "
                       f"{prof['n_after_window']} ({prof['rate_ratio']:.2f}x) across it. "
                       f"Successor CIK {cik} first filed {cont['own_first']}, "
                       f"{cont['own_lead_days']} d before the truncation."
                       + (f" ORACLE 4: {oracle4['why']}" if oracle4 else "")}

    return {"ticker": ticker, "cls": "unresolved", **cont, "rejected_co_filers": rejected,
            "why": f"{len(peers)} co-filer(s) among the {ORACLE3_MAX_HEADERS} of {cik}'s "
                   f"registrant-filed documents nearest the boundary, none a predecessor "
                   f"({'; '.join(f'{r['name']}: {r['why']}' for r in rejected[:3]) or 'none'}). "
                   f"Own archive starts {cont['own_first']} ({cont['own_lead_days']} d before "
                   "the truncation). Needs the name fallback."}


# --------------------------------------------------------------------------------------- #
# Driver                                                                                   #
# --------------------------------------------------------------------------------------- #
#: `sp500_tickers.cik` comes from WIKIPEDIA (`prices/fetch_tickers.py`), and Wikipedia had
#: already moved XOM to the holdco -- so the roster itself carried the defect. Sharadar's
#: `secfilings` URL is an independent oracle on the same question.
ROSTER_CROSSCHECK_SQL = """
WITH sh AS (SELECT ticker, lpad(regexp_replace(secfilings, '^.*CIK=0*', ''), 10, '0') AS sh_cik
            FROM sharadar_tickers WHERE secfilings LIKE '%CIK=%')
SELECT s.ticker, lpad(s.cik::text, 10, '0') AS roster_cik, sh.sh_cik
FROM sp500_tickers s JOIN sh USING(ticker)
WHERE lpad(s.cik::text, 10, '0') <> sh.sh_cik
ORDER BY 1
"""


def roster_crosscheck(conn, registrants: dict) -> pd.DataFrame:
    """Tickers where the Wikipedia-derived roster CIK disagrees with Sharadar's.

    Measured 2026-09-09: 498 scored, 497 agree, and the single disagreement is XOM -- where
    SHARADAR IS RIGHT. That check, had it existed, would have caught XOM a year early for
    free, which is why it ships here rather than staying an observation in a report.

    ⚠ REPORT, DO NOT AUTO-OVERRIDE. Sharadar points at the CURRENT registrant too, so it
    cannot recover a predecessor -- and silently rewriting a curated roster from a vendor
    snapshot is how hand-established evidence gets lost. The check names the disagreement;
    the register resolves it. A disagreement on a ticker that HAS a register entry is
    therefore expected and is annotated rather than flagged.
    """
    df = pd.DataFrame(conn.execute(text(ROSTER_CROSSCHECK_SQL)).mappings().all())
    if df.empty:
        return df
    df["explained_by_register"] = df.apply(
        lambda r: r["ticker"] in registrants
        and r["sh_cik"] in registrants[r["ticker"]].all_ciks(), axis=1)
    return df


def audit_chains(context, registrants: dict) -> list[dict]:
    """Does any register entry have an EARLIER hop it does not declare?

    ⚠ A TWO-SEGMENT ENTRY FOR A TWO-HOP CHAIN LOOKS LIKE A FIX AND IS HALF A FIX, which is the
    worst shape a register entry can take: the ticker's history moves back, the screens stop
    flagging it, and a decade stays missing with the entry standing as evidence that it was
    handled.

    VTRS is the measured case and it was caught by accident. Its insider re-parse recovered
    only to 2015-02-27 rather than 2006, because Mylan N.V. was ITSELF created in 2015 -- the
    pre-2015 registrant is Mylan Inc (CIK 69499), a third CIK. Nothing in the detector looked
    for that, because the screens key on the TICKER's stored history and a partially repaired
    ticker no longer trips them.

    So this runs oracle 3 against the OLDEST segment's CIK, asking the same question one hop
    further back. It is the closing loop the offline screens cannot provide.
    """
    out: list[dict] = []
    for ticker, reg in sorted(registrants.items()):
        oldest = reg.segments[0]
        try:
            doc = submissions(context, oldest.cik)
        except Exception as e:                              # noqa: BLE001
            out.append({"ticker": ticker, "cls": "unresolved",
                        "why": f"submissions failed for {oldest.cik}: {e}"})
            continue
        own_first = continuity(context, doc, str(reg.boundaries[0].date()))["own_first"]
        if own_first is None:
            continue
        # Ask at the oldest segment's OWN first filing: is there a registrant before it?
        r = oracle3(context, ticker, oldest.cik, own_first)
        r["oldest_cik"] = oldest.cik
        r["oldest_first_filing"] = own_first
        out.append(r)
    return out


def collect(conn, registered: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """The candidate table and each candidate's truncation anchor (its first 8-K)."""
    tight, proxy, late = oracle1(conn)
    anchor = dict(zip(late["ticker"], late["first_8k"])) if not late.empty else {}
    cand = pd.DataFrame({"ticker": sorted(set(tight.get("ticker", [])) |
                                          set(proxy.get("ticker", [])))})
    cand["screen"] = cand["ticker"].map(
        lambda t: "+".join(s for s, df in (("tight", tight), ("proxy", proxy))
                           if not df.empty and t in set(df["ticker"])))
    cand["first_8k"] = cand["ticker"].map(anchor)
    cand["lag_y"] = cand["ticker"].map(
        dict(zip(late["ticker"], late["lag_y"])) if not late.empty else {})
    cand["already_registered"] = cand["ticker"].isin(registered)
    return cand, anchor


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="./configs")
    p.add_argument("--offline", action="store_true", help="oracles 1 + 2 only, no network")
    p.add_argument("--classify", action="store_true", help="add oracle 3, serially")
    p.add_argument("--audit-chains", action="store_true",
                   help="ask oracle 3 one hop FURTHER BACK on every register entry, to catch a "
                        "two-segment entry standing in for a longer chain (the VTRS shape)")
    p.add_argument("-t", "--tickers", default="", help="restrict oracle 3 to these")
    p.add_argument("--emit-register", metavar="PATH",
                   help="write a PROPOSED register; never touches the live config")
    p.add_argument("--out", metavar="PATH", help="write the classification JSON here")
    p.add_argument("--report", metavar="PATH",
                   help="write classified.md; reads --out's JSON when not classifying, so the "
                        "report can be regenerated offline from a previous walk")
    args = p.parse_args(argv)

    _, context = get_config_context(config_path=args.config, use_cache=False, save=False)
    registrants = load_registrants(str(context.config_dir))
    registered = set(registrants)

    with context.store.engine.connect() as conn:
        cand, anchor = collect(conn, registered)
        tight, proxy, late = oracle1(conn)
        actions = oracle2(conn, list(cand["ticker"]), anchor)
        crosscheck = roster_crosscheck(conn, registrants)
        roster = pd.DataFrame(conn.execute(text(
            "select ticker, lpad(cik::text, 10, '0') as cik from sp500_tickers")).mappings().all())

    ciks = dict(zip(roster["ticker"], roster["cik"]))
    # Two columns, not one. `namechange` is every `namechangefrom` in window and is the
    # SUGGESTIVE signal; `shell_name` is the subset whose contraname is a shell by name and is
    # the CONCLUSIVE one -- "CORVETTEPORSCHE CORP" is not a company, it is the Conoco/Phillips
    # merger's internal codename. Reporting only the second would hide how many candidates
    # Sharadar speaks to at all; reporting only the first would overstate the evidence.
    cand["namechange"] = cand["ticker"].map(
        lambda t: next((a["contraname"] for a in actions.get(t, [])
                        if a["action"] == "namechangefrom"), None))
    cand["shell_name"] = cand["ticker"].map(
        lambda t: next((a["contraname"] for a in actions.get(t, []) if a["is_shell"]), None))

    print(f"oracle 1 -- tight cluster : {len(tight):>3} tickers")
    print(f"oracle 1 -- proxy screen  : {len(proxy):>3} tickers")
    print(f"oracle 1 -- union         : {len(cand):>3} candidates "
          f"({int(cand['already_registered'].sum())} already registered)")
    print(f"oracle 1 -- all late 8-K  : {len(late):>3} tickers  <- the population")
    print(f"oracle 2 -- name changed  : {int(cand['namechange'].notna().sum()):>3} candidates"
          "  (suggestive)")
    print(f"oracle 2 -- shell named   : {int(cand['shell_name'].notna().sum()):>3} candidates"
          "  (conclusive)")
    print()
    print(cand.to_string(index=False))

    unexplained_cik = pd.DataFrame()
    if not crosscheck.empty:
        unexplained_cik = crosscheck[~crosscheck["explained_by_register"]]
        print(f"\nroster CIK cross-check -- {len(crosscheck)} disagreement(s) with "
              f"Sharadar, {len(unexplained_cik)} NOT explained by a register entry")
        print(crosscheck.to_string(index=False))
        if not unexplained_cik.empty:
            print("  ⚠ `sp500_tickers.cik` comes from Wikipedia, and Wikipedia had "
                  "already moved XOM to the holdco. Sharadar is an independent oracle and was "
                  "right there, so investigate each row rather than overriding the roster.")

    if args.audit_chains:
        context.ensure_edgar_identity()
        print(f"\nchain audit -- {len(registrants)} register entr(y|ies), "
              "one hop further back\n")
        # `unresolved` from oracle 3 means "no predecessor found", which in a CHAIN audit is
        # the GOOD answer -- the chain is complete. Relabelled, because printing the raw class
        # here reads as 16 failures when 15 of them are passes.
        labels = {"unresolved": "chain complete", "continuous_registrant": "chain complete",
                  "foreign_private_issuer": "chain complete (FPI)",
                  "cutover": "EARLIER HOP, confirmed",
                  "unconfirmed_cutover": "EARLIER HOP, unconfirmed",
                  "ambiguous": "EARLIER HOP, ambiguous"}
        earlier = []
        for r in audit_chains(context, registrants):
            print(f"  {r['ticker']:6} oldest={r.get('oldest_cik')} "
                  f"first={r.get('oldest_first_filing')} "
                  f"{labels.get(r['cls'], r['cls']):26} "
                  f"{str(r.get('predecessor_name') or '')[:34]}")
            if r["cls"] in ("cutover", "unconfirmed_cutover", "ambiguous"):
                earlier.append(r)
        if earlier:
            print(f"\n  ⚠ {len(earlier)} entr(y|ies) may be missing an earlier segment: "
                  f"{', '.join(r['ticker'] for r in earlier)}")
            print("    A two-segment entry for a two-hop chain looks like a fix and is half "
                  "one -- the screens stop flagging the ticker while a decade stays missing.")
        else:
            print("\n  OK: no register entry has an undeclared earlier hop.")

    results: list[dict] = []
    if args.classify:
        # ⚠ edgartools raises `IdentityNotSetError` on the FIRST request without it, and the
        # traceback surfaces from deep inside a retry wrapper -- a shape that reads as a
        # network fault rather than a missing header. One walk died here already.
        context.ensure_edgar_identity()
        only = {t for t in args.tickers.split(",") if t}
        todo = [t for t in cand["ticker"] if not only or t in only]
        print(f"\noracle 3 -- {len(todo)} candidate(s), SERIAL "
              f"(~{ORACLE3_MAX_HEADERS + 1} requests each)\n")
        for i, ticker in enumerate(todo, 1):
            cik = ciks.get(ticker)
            if cik is None:
                results.append({"ticker": ticker, "cls": "unresolved",
                                "why": "not in sp500_tickers"})
                continue
            r = oracle3(context, ticker, cik, anchor.get(ticker) or str(
                cand.loc[cand["ticker"] == ticker, "first_8k"].iloc[0]))
            r["roster_cik"] = cik
            r["shell_name"] = cand.loc[cand["ticker"] == ticker, "shell_name"].iloc[0]
            results.append(r)
            print(f"[{i:>3}/{len(todo)}] {ticker:6} {r['cls']:<24} {r['why'][:120]}")

    if args.out and results:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.emit_register and results:
        proposed = {r["ticker"]: _propose(r, anchor) for r in results if r["cls"] == "cutover"}
        Path(args.emit_register).write_text(json.dumps(proposed, indent=2, default=str),
                                            encoding="utf-8")
        print(f"proposed {len(proposed)} register entr(y|ies) -> {args.emit_register}")

    if args.report:
        # Regenerable offline from a previous walk's JSON: the report is a VIEW of the
        # classification, and re-walking EDGAR to reformat a table would be absurd.
        if not results and args.out and Path(args.out).exists():
            results = json.loads(Path(args.out).read_text(encoding="utf-8"))
        Path(args.report).write_text(
            classified_report(cand, late, results, registered), encoding="utf-8")
        print(f"wrote {args.report}")

    # Non-zero on anything a human still has to look at, so this can stand as a nightly check.
    # `unconfirmed_cutover` and `ambiguous` count: both name a real boundary that has no
    # register entry, so the ticker is still truncated and still silent about it.
    open_classes = ("unresolved", "ambiguous", "unconfirmed_cutover")
    by_class = {c: sorted(r["ticker"] for r in results if r["cls"] == c) for c in open_classes}
    for cls, names in by_class.items():
        if names:
            print(f"\n⚠ {cls.upper()} ({len(names)}): {', '.join(names)}")
    if not unexplained_cik.empty:
        print(f"\n⚠ ROSTER CIK DISAGREEMENT ({len(unexplained_cik)}): "
              f"{', '.join(unexplained_cik['ticker'])}")
    return 1 if (any(by_class.values()) or not unexplained_cik.empty) else 0


#: What each class means and what phase 9 does about it. Kept beside the classifier so the
#: report and the code cannot drift into describing different things.
CLASS_NOTES = {
    "cutover": ("registrant cutover, CONFIRMED", "register entry"),
    "unconfirmed_cutover": ("registrant cutover, nominated but UNCONFIRMED",
                            "needs adjudication -- no entry"),
    "ambiguous": ("two or more predecessors collapsed; cannot choose",
                  "needs adjudication -- no entry"),
    "foreign_private_issuer": ("foreign private issuer, one continuous CIK",
                               "OUT OF SCOPE -- the repair is form coverage"),
    "continuous_registrant": ("the same CIK filed throughout", "NOT A DEFECT"),
    "unresolved": ("no predecessor found", "needs the name fallback -- no entry"),
    "not_screened": ("late 8-K but neither screen fired",
                     "sparse pre-2000 8-K, most likely a non-defect"),
}


def classified_report(cand: pd.DataFrame, late: pd.DataFrame, results: list[dict],
                      registered: set[str]) -> str:
    """`classified.md`: one row per candidate, EVERY candidate, including the non-defects.

    ⚠ The 62 is the population, not the 42. A candidate silently dropped is indistinguishable
    next quarter from one never examined, which is how this whole defect class survived a
    year -- so the tickers that turned out to need no work are listed with the reason they
    need none, not omitted for being uninteresting.
    """
    by_ticker = {r["ticker"]: r for r in results}
    rows: list[tuple[str, str, str, str]] = []
    for _, c in late.sort_values("lag_y", ascending=False).iterrows():
        t = c["ticker"]
        if t in registered:
            cls, why = "cutover", "already in the register before this pass"
        elif t in by_ticker:
            cls, why = by_ticker[t]["cls"], by_ticker[t].get("why", "")
        else:
            cls = "not_screened"
            why = (f"first 8-K {c['first_8k']} is {c['lag_y']} y after the first price, but "
                   "neither the tight-cluster nor the proxy screen fired -- the other "
                   "registrant-keyed tables do not agree on a truncation date, which is the "
                   "shape of a sparse pre-2004 8-K record rather than a boundary")
        rows.append((t, cls, str(c["lag_y"]), why))

    counts: dict[str, int] = {}
    for _, cls, _, _ in rows:
        counts[cls] = counts.get(cls, 0) + 1

    out = [f"# Registrant-cutover classification — all {len(rows)} candidates", "",
           "Population: every ticker whose `sec_8k` archive starts more than "
           f"{LAG_YEARS:g} years after its first price. Produced by "
           "`scripts/detect_registrant_cutovers.py --classify`.", "",
           "| class | n | what phase 9 does |", "|---|---:|---|"]
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        label, action = CLASS_NOTES.get(cls, (cls, ""))
        out.append(f"| {label} | {n} | {action} |")
    out += ["", "## Every candidate", "",
            "| ticker | class | lag (y) | predecessor | evidence |", "|---|---|---:|---|---|"]
    for t, cls, lag, why in rows:
        r = by_ticker.get(t, {})
        pred = (f"`{r['predecessor_cik']}` {r.get('predecessor_name', '')}"
                if r.get("predecessor_cik") else
                " · ".join(f"`{c['cik']}` {c['name']}" for c in r.get("candidates", []))
                or "—")
        out.append(f"| {t} | {cls} | {lag} | {pred} | {why.replace('|', '/')} |")
    return "\n".join(out) + "\n"


def _propose(r: dict, anchor: dict[str, str]) -> dict:
    """A PROPOSED two-segment entry. Deliberately not written to the live config: every entry
    is a diff a human reads first, because a wrong `valid_to` deletes history silently."""
    boundary = anchor.get(r["ticker"]) or r.get("own_first")
    return {"kind": "reorganisation", "segments": [
        {"cik": r["predecessor_cik"], "valid_to": boundary,
         "evidence": f"PROPOSED -- {r.get('predecessor_name')} (CIK {r['predecessor_cik']}), "
                     f"found by the co-indexed filer scan on {r.get('co_indexed_on')} of the "
                     "successor's own filings. VERIFY the boundary against the predecessor's "
                     "last filing before accepting."},
        {"cik": r["roster_cik"], "valid_from": boundary,
         "evidence": f"PROPOSED -- successor, first filing {r.get('own_first')}. {r['why']}"}]}


if __name__ == "__main__":
    raise SystemExit(main())
