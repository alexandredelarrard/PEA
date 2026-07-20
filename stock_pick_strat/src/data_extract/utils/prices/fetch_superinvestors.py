"""
fetch_superinvestors.py  (src/data_extract/utils/prices/fetch_superinvestors.py)
--------------------------------------------------------------------------------
Build the "superinvestors" roster JSON: the curated top long-term investors from
Dataroma, resolved to their SEC 13F manager CIK, ranked by 13F long-equity AUM,
and rank-weighted. This JSON is the reproducible key that lets the feature layer
recompute the 13F buy/sell-evolution features for JUST the elite managers
(`superinvestor_features.build_superinvestor_feature_panel`), on top of the
all-filer institutional features.

WHY the roster (not returns): Dataroma exposes NO per-manager return and NO CIK —
only a curated list of managers with proven long-term records at `holdings.php?m=CODE`.
So the *inclusion* on Dataroma is the "outperformed over the long run" filter, and
we rank the curated set by 13F AUM (a size/conviction proxy from data we already
hold). CIKs are resolved from the filer names in the cached 13F SUBMISSION.tsv
(the same zips `fetch_13f` caches), with a manual override map for the few misses.

Reproducible + tunable: `build_superinvestors_json(context, top_n=..., weighting=...)`
writes `data/superinvestors/superinvestors.json`; change `top_n` to widen/narrow
the set. The pure pieces (roster parse, name->CIK resolution, rank/weight) are
unit-tested; only the HTTP GET, the zip read, and the DB AUM query touch the world.
"""
from __future__ import annotations

import json
import logging
import re
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

from src.constants.constants import (
    DATAROMA_HEADERS, DATAROMA_HOME_URL, SUPERINVESTOR_CIK_OVERRIDES,
    SUPERINVESTORS_DEFAULT_TOP_N, SUPERINVESTORS_JSON, SUPERINVESTORS_WEIGHTING)
from src.context import Context

logger = logging.getLogger(__name__)

# manager-name tokens that carry no matching signal (legal/entity boilerplate)
_STOP_TOKENS = {
    "LP", "LLP", "LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "LTD",
    "LIMITED", "CAPITAL", "MANAGEMENT", "MGMT", "MGT", "PARTNERS", "PARTNER", "GROUP",
    "ADVISORS", "ADVISERS", "ADVISORY", "ASSET", "ASSETS", "FUND", "FUNDS", "HOLDINGS",
    "HOLDING", "INVESTMENT", "INVESTMENTS", "INTERNATIONAL", "GLOBAL", "AND", "THE",
    "COMPANY", "MASTER", "SECURITIES", "TRUST", "FINANCIAL", "RESEARCH", "SERVICES",
}
_MATCH_THRESHOLD = 0.6   # fraction of Dataroma name-tokens that must be in the filer name
_AUM_QUARTERS_BACK = 4   # look this many recent quarters back for a manager's latest AUM


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                    #
# --------------------------------------------------------------------------- #
def _pad_cik(x: object) -> str:
    """Canonical 10-digit zero-padded CIK (matches sp500_tickers / the roster JSON).
    Tolerates ints, '123', '123.0', already-padded strings; '' if no digits."""
    s = str(x).strip().split(".")[0]
    s = re.sub(r"\D", "", s)
    return s.zfill(10) if s else ""


def _name_tokens(name: str) -> frozenset[str]:
    """Significant upper-case tokens of a manager/fund name (boilerplate dropped)."""
    words = re.sub(r"[^A-Za-z0-9 ]", " ", str(name).upper()).split()
    return frozenset(w for w in words if w not in _STOP_TOKENS and len(w) > 1)


def _fund_part(dataroma_name: str) -> str:
    """Dataroma lists 'Person - Fund'; the SEC filer is the FUND, so match on the
    part after the last dash (fall back to the whole string when there is no dash)."""
    parts = re.split(r"\s[-–—]\s", str(dataroma_name))
    return parts[-1].strip() if len(parts) > 1 else str(dataroma_name).strip()


def _parse_dataroma_roster(html: str) -> list[dict]:
    """Dataroma home page -> [{code, name}] for every `holdings.php?m=CODE` link.
    Deduplicated by code, order preserved. Robust to the surrounding markup."""
    soup = BeautifulSoup(html or "", "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"holdings\.php\?m=([A-Za-z0-9_.\-]+)", a["href"])
        if not m:
            continue
        code = m.group(1)
        name = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        # Dataroma appends "Updated <D Mon YYYY>" to each link text; strip it so the
        # date does not leak into the fund name / matching tokens.
        name = re.sub(r"\s+Updated\b.*$", "", name, flags=re.IGNORECASE).strip()
        if code and code not in seen and name:
            seen.add(code)
            out.append({"code": code, "name": name})
    return out


def _match_score(query: frozenset[str], target: frozenset[str]) -> float:
    """Fraction of the query's tokens present in the target (containment), the more
    forgiving direction since filer names carry extra entity words."""
    if not query:
        return 0.0
    return len(query & target) / len(query)


def _resolve_ciks(roster: list[dict], filer_index: list[tuple[frozenset[str], str, str]],
                  overrides: dict[str, str]) -> list[dict]:
    """Attach a CIK to each roster entry: manual override first, else the best
    token-containment match in the 13F filer index above `_MATCH_THRESHOLD`.
    Unmatched entries keep cik=None (dropped later, logged)."""
    resolved = []
    for r in roster:
        cik = _pad_cik(overrides[r["code"]]) if r["code"] in overrides else None
        matched_name = "override" if cik else None
        if cik is None:
            q = _name_tokens(_fund_part(r["name"]))
            best_score, best = 0.0, None
            for tokens, fcik, fname in filer_index:
                s = _match_score(q, tokens)
                # tie-break toward the LEANER filer name (fewer extra tokens = tighter match)
                if s > best_score or (s == best_score and best is not None
                                      and len(tokens) < len(best[0])):
                    best_score, best = s, (tokens, fcik, fname)
            if best is not None and best_score >= _MATCH_THRESHOLD:
                cik, matched_name = best[1], best[2]
        resolved.append({**r, "cik": cik, "matched_filer": matched_name})
    return resolved


def _rank_and_weight(resolved: list[dict], aum: dict[str, float],
                     top_n: int, weighting: str) -> list[dict]:
    """Keep resolved managers that appear in our 13F (have AUM), sort by AUM desc,
    take top_n, and assign positive weights (sum to 1). `weighting`:
      rank  -> linear decay, rank 1 gets the most (default)
      aum   -> proportional to 13F AUM
      equal -> uniform."""
    cand = [{**r, "aum_usd": float(aum.get(r["cik"], 0.0))}
            for r in resolved if r.get("cik") in aum]
    cand.sort(key=lambda d: d["aum_usd"], reverse=True)
    top = cand[: max(0, int(top_n))]
    n = len(top)
    if n == 0:
        return top
    if weighting == "equal":
        raw = [1.0] * n
    elif weighting == "aum":
        raw = [d["aum_usd"] for d in top]
    else:                                            # "rank": linear decay n, n-1, ..., 1
        raw = [float(n - i) for i in range(n)]
    total = sum(raw) or 1.0
    for i, d in enumerate(top):
        d["rank"] = i + 1
        d["weight"] = raw[i] / total
    return top


# --------------------------------------------------------------------------- #
# IO: Dataroma fetch, cached-13F filer index, DB AUM                            #
# --------------------------------------------------------------------------- #
def _http_get(url: str) -> requests.Response:
    """GET with SSL verification, falling back to an UNVERIFIED retry on SSLError.
    Dataroma serves an incomplete certificate chain (missing intermediate) that
    OpenSSL cannot verify; the data is public and read-only, so an unverified fetch
    is acceptable here and is logged so the relaxation is never silent."""
    try:
        r = requests.get(url, headers=DATAROMA_HEADERS, timeout=60)
    except requests.exceptions.SSLError:
        logger.warning("Dataroma SSL chain incomplete -> retrying unverified (%s)", url)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            r = requests.get(url, headers=DATAROMA_HEADERS, timeout=60, verify=False)
    r.raise_for_status()
    return r


def _read_submission(zip_path: Path) -> pd.DataFrame:
    """Read ONLY SUBMISSION.tsv (small: one row per filer) from a cached 13F zip ->
    [cik, name]. INFOTABLE (huge) is intentionally not touched here."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = {n.upper(): n for n in z.namelist()}
            if "SUBMISSION.TSV" not in names:
                return pd.DataFrame(columns=["cik", "name"])
            sub = pd.read_csv(z.open(names["SUBMISSION.TSV"]), sep="\t",
                              dtype=str, low_memory=False)
    except (zipfile.BadZipFile, OSError):
        return pd.DataFrame(columns=["cik", "name"])
    cols = {c.lower(): c for c in sub.columns}
    cik = sub[cols["cik"]] if "cik" in cols else pd.Series(dtype=str)
    name = sub[cols.get("filingmanager_name", cols.get("name", ""))] \
        if ("filingmanager_name" in cols or "name" in cols) else pd.Series(dtype=str)
    return pd.DataFrame({"cik": cik, "name": name}).dropna()


def _build_cik_name_index(cache_dir: Path) -> list[tuple[frozenset[str], str, str]]:
    """(name-tokens, padded-cik, raw-name) for every distinct 13F filer in the most
    recent cached quarters. Recent-only: superinvestors file every quarter, so a few
    quarters give current CIKs cheaply and avoid renamed/stale entities."""
    if not cache_dir.exists():
        return []
    zips = sorted(cache_dir.glob("*_form13f.zip"))[-_AUM_QUARTERS_BACK:]
    frames = [_read_submission(z) for z in zips]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return []
    idx = pd.concat(frames, ignore_index=True).drop_duplicates("cik", keep="last")
    out = []
    for _, row in idx.iterrows():
        cik = _pad_cik(row["cik"])
        if cik:
            out.append((_name_tokens(row["name"]), cik, str(row["name"])))
    return out


def _manager_aum(context: Context, ciks: set[str]) -> dict[str, float]:
    """Latest-quarter aggregate long-equity value (value_usd) per manager CIK, from
    institutional_holdings. Empty if the table/CIKs are absent."""
    if not ciks or not context.store.exists("institutional_holdings"):
        return {}
    df = context.store.load("institutional_holdings",
                            columns=["cik", "period", "value_usd"])
    if df is None or df.empty:
        return {}
    df["cik"] = df["cik"].map(_pad_cik)
    df = df[df["cik"].isin(ciks)].copy()
    if df.empty:
        return {}
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0.0)
    latest = df.groupby("cik")["period"].transform("max")
    cur = df[df["period"] == latest]
    return cur.groupby("cik")["value_usd"].sum().to_dict()


# --------------------------------------------------------------------------- #
# Entry points                                                                  #
# --------------------------------------------------------------------------- #
def _json_path(context: Context, out_path: str | Path | None) -> Path:
    return Path(out_path) if out_path else (context.paths["DATA_STORE"] / SUPERINVESTORS_JSON)


def build_superinvestors_json(
    context: Context,
    top_n: int | None = None,
    weighting: str | None = None,
    out_path: str | Path | None = None,
) -> dict:
    """Scrape Dataroma's curated superinvestors, resolve each to a 13F CIK, rank the
    top_n by 13F AUM, rank-weight them, and persist the roster JSON (returned too).
    `top_n` is the single tunable knob for how many managers to keep."""
    top_n = SUPERINVESTORS_DEFAULT_TOP_N if top_n is None else int(top_n)
    weighting = weighting or SUPERINVESTORS_WEIGHTING

    resp = _http_get(DATAROMA_HOME_URL)
    roster = _parse_dataroma_roster(resp.text)
    logger.info("Dataroma: parsed %d superinvestors", len(roster))

    filer_index = _build_cik_name_index(context.paths["SEC_13F_INSIDERS_DIR"])
    resolved = _resolve_ciks(roster, filer_index, SUPERINVESTOR_CIK_OVERRIDES)
    n_resolved = sum(1 for r in resolved if r.get("cik"))
    unresolved = [r["name"] for r in resolved if not r.get("cik")]
    if unresolved:
        logger.info("Superinvestors: %d/%d resolved to a CIK; unresolved -> %s",
                    n_resolved, len(roster), ", ".join(unresolved[:10]))

    aum = _manager_aum(context, {r["cik"] for r in resolved if r.get("cik")})
    top = _rank_and_weight(resolved, aum, top_n, weighting)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": DATAROMA_HOME_URL,
        "top_n": top_n,
        "weighting": weighting,
        "n_roster": len(roster),
        "n_resolved": n_resolved,
        "managers": [
            {"rank": m["rank"], "name": m["name"], "code": m["code"], "cik": m["cik"],
             "aum_usd": round(m["aum_usd"], 2), "weight": round(m["weight"], 6),
             "matched_filer": m.get("matched_filer")}
            for m in top
        ],
    }
    path = _json_path(context, out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.warning("Saved superinvestors roster: %d managers (top_n=%d, weighting=%s) -> %s",
                   len(top), top_n, weighting, path)
    return out


def load_superinvestors(context: Context, out_path: str | Path | None = None) -> dict | None:
    """Read the persisted superinvestors roster JSON; None if it hasn't been built."""
    path = _json_path(context, out_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Superinvestors roster at %s is unreadable", path)
        return None
