"""
compare_def14a_baseline.py  (scripts/)
--------------------------------------------------------------------------------------------
The before/after gate for the DEF 14A extraction fix. Loads the Phase-0 `baseline/` parquet and
the Phase-6 `new/` parquet and prints ONE metric table plus the G1-G14 gates.

Two properties make the numbers honest:

  * Fill rate is computed **per field over the SAME accession set** in both snapshots, so a
    coverage change (the backfill was still running) cannot masquerade as a quality change.
  * Every gate is reported as a PASS/FAIL ROW rather than raised, so one regression does not
    hide the rest.

Table mapping across the cutover (the edgar HTML block is retired, the LLM path takes over):

    sec_def14a_executive_comp -> def14a_executive_comp
    sec_def14a_director_comp  -> def14a_director_comp
    sec_def14a_ownership      -> def14a_ownership
    sec_def14a_votes          -> (retired, no successor -- board_recommendation was fabricated)
    sec_8k_item507            -> sec_8k_votes
    def14a_llm / sec_def14a   -> themselves, with changed column sets

    "$PY" scripts/compare_def14a_baseline.py [--out DIR] [--baseline-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix"

#: A single SCT / director-fee component above this is arithmetically impossible for an S&P 500
#: executive and is the signature of the `<br>`-concatenation defect (AMAT `total` = 3.527e23).
IMPLAUSIBLE_USD = 1e9
#: Ratio at which a row's implied shares outstanding reads as a footnote digit glued onto the
#: share count (PG's `217,956,036` + superscript `2` -> `2179560362`). A real holder never owns
#: 10x the total the rest of the same table implies.
FOOTNOTE_DIGIT_RATIO = 5.0
#: A director losing >30% of the vote is the withhold-campaign event; below it the election is
#: routine. Only used to report the vote table's shape, not as a gate.
SUPPORT_ALERT = 0.70

#: Exec-comp / director-comp component columns. The new LLM-side tables deliberately REUSE the
#: retired edgar tables' vocabulary so the two snapshots compare column-for-column.
EXEC_COMPONENTS = ["salary", "bonus", "stock_awards", "option_awards",
                   "non_equity_incentive", "pension_change", "other_compensation", "total"]
DIR_COMPONENTS = ["fees_earned", "stock_awards", "option_awards",
                  "non_equity_incentive", "pension_change", "other_compensation", "total"]

TABLE_MAP = {
    "def14a_llm": "def14a_llm",
    "sec_def14a": "sec_def14a",
    "sec_def14a_executive_comp": "def14a_executive_comp",
    "sec_def14a_director_comp": "def14a_director_comp",
    "sec_def14a_ownership": "def14a_ownership",
    "sec_8k_item507": "sec_8k_votes",
}


# --------------------------------------------------------------------------- #
# loading                                                                     #
# --------------------------------------------------------------------------- #
def _read(d: Path, stem: str) -> pd.DataFrame:
    p = d / f"{stem}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_side(d: Path, mapping_key: str) -> dict[str, pd.DataFrame]:
    """Every snapshot table for one side, keyed by the BASELINE name so the gates can address
    a table by one name regardless of which side it came from."""
    out = {}
    for base_name, new_name in TABLE_MAP.items():
        out[base_name] = _read(d, base_name if mapping_key == "baseline" else new_name)
    if mapping_key == "baseline":
        out["sec_def14a_votes"] = _read(d, "sec_def14a_votes")
    for extra in ("def14a_directors",):                 # new-side only (the table is new)
        out[extra] = _read(d, extra)
    return out


# --------------------------------------------------------------------------- #
# metric primitives                                                           #
# --------------------------------------------------------------------------- #
def _fill(df: pd.DataFrame, col: str, keys: set | None = None) -> float | None:
    """Non-null share of `col`, restricted to `keys` accessions when given so the two sides are
    measured over the SAME population."""
    if df.empty or col not in df.columns:
        return None
    if keys is not None and "accession_number" in df.columns:
        df = df[df["accession_number"].isin(keys)]
    return None if df.empty else round(float(df[col].notna().mean()), 4)


def _n_implausible(df: pd.DataFrame, cols: list[str]) -> int | None:
    """Rows where ANY component exceeds `IMPLAUSIBLE_USD` (G1)."""
    if df.empty:
        return None
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    vals = df[present].apply(pd.to_numeric, errors="coerce")
    return int((vals.abs() > IMPLAUSIBLE_USD).any(axis=1).sum())


def _n_footnote_digit(df: pd.DataFrame) -> int | None:
    """Ownership rows whose implied shares-outstanding is a multiple of what the rest of the
    SAME table implies (G2).

    Self-contained on purpose: `shares / percent_of_class` is an estimate of total shares
    outstanding that every row of one filing must agree on. A row off by ~10x from its own
    filing's median has a footnote digit glued to its share count -- no second table needed.
    """
    if df.empty or not {"shares", "percent_of_class", "accession_number"}.issubset(df.columns):
        return None
    d = df.copy()
    d["shares"] = pd.to_numeric(d["shares"], errors="coerce")
    d["pct"] = pd.to_numeric(d["percent_of_class"], errors="coerce")
    d = d[(d["shares"] > 0) & (d["pct"] > 0)]
    if d.empty:
        return 0
    d["implied"] = d["shares"] / d["pct"]
    med = d.groupby("accession_number")["implied"].transform("median")
    ratio = d["implied"] / med
    return int(((ratio > FOOTNOTE_DIGIT_RATIO) | (ratio < 1 / FOOTNOTE_DIGIT_RATIO)).sum())


def _pct_single_neo(df: pd.DataFrame) -> float | None:
    """Share of 2012+ accessions whose SCT yielded exactly ONE NEO (G5) -- the single best
    summary of whether the table-anchored carve reached the Summary Compensation Table."""
    if df.empty or "n_neos" not in df.columns:
        return None
    d = df[pd.to_datetime(df["as_of"], errors="coerce") >= "2012-01-01"]
    d = d[d["n_neos"].notna()]
    return None if d.empty else round(float((d["n_neos"] == 1).mean()), 4)


#: Key columns carry no extraction signal, and the four PROVISION flags are worse than
#: signal-free: today's prompt orders the model to infer FALSE for them, so a row extracted
#: from a 10 KB EDGAR folder index still comes back with `poison_pill=0, classified_board=0,
#: dual_class_shares=0, majority_voting=0` and reads as "populated". Measured on the baseline:
#: 17 of 26 pre-2001 rows carry NOTHING ELSE. G9 is therefore computed over the SUBSTANTIVE
#: payload, which is why it reads ~92% here against the 34.6% a naive all-null test reports.
_NON_SIGNAL_COLS = {"ticker", "as_of", "period", "accession_number", "def14a_json", "cik",
                    "poison_pill", "classified_board", "dual_class_shares", "majority_voting",
                    "technology_committee", "n_technology_directors", "pct_technology_directors"}


def _pct_fully_null(df: pd.DataFrame, until: str) -> tuple[int, int] | None:
    """(empty rows, total rows) for filings before `until` (G9) -- the `_doc_url` bug's
    footprint. 'Empty' = every SUBSTANTIVE field is null (see `_NON_SIGNAL_COLS`), i.e. the LLM
    was handed a folder index instead of the proxy and invented only the inferred-FALSE flags."""
    if df.empty or "as_of" not in df.columns:
        return None
    d = df[pd.to_datetime(df["as_of"], errors="coerce") < until]
    if d.empty:
        return None
    payload = [c for c in d.columns if c not in _NON_SIGNAL_COLS]
    return int(d[payload].isna().all(axis=1).sum()), int(len(d))


def _peo_zero(df: pd.DataFrame) -> int | None:
    """`peo_total_comp == 0.0` rows (G3) -- the SBUX ECD zero-matrix."""
    if df.empty or "peo_total_comp" not in df.columns:
        return None
    return int((pd.to_numeric(df["peo_total_comp"], errors="coerce") == 0.0).sum())


def _co_peo(df: pd.DataFrame, tickers=("BA", "NKE")) -> dict[str, object]:
    """How many PEOs the ECD path recorded for the known co-PEO filings (G4). Baseline has no
    `n_peos` column at all, so the baseline answer is 1-by-construction."""
    if df.empty:
        return {}
    d = df[df["ticker"].isin(tickers)]
    if d.empty:
        return {}
    if "n_peos" not in d.columns:
        return {t: 1 for t in sorted(d["ticker"].unique())}
    out = {}
    for t, g in d.groupby("ticker"):
        out[t] = int(pd.to_numeric(g["n_peos"], errors="coerce").max(skipna=True) or 0)
    return out


def _or_none(v):
    """An EMPTY result means 'this side was not produced yet' and must report PENDING, not
    FAIL -- a gate that fails because its input is missing is noise, not a finding."""
    return None if v is None or (hasattr(v, "__len__") and len(v) == 0) else v


def _low_say_on_pay(df: pd.DataFrame) -> tuple[int, list[str]] | None:
    """Sub-0.50 say-on-pay values that SURVIVE the cube's clean-on-read stage (G8).

    Measured THROUGH `impute_def14a`, not on the raw table, because the cube reads the cleaned
    frame and that is where a nulling rule would bite: back when a 0.50 floor lived in the
    stage, `def14a_llm` kept the revolts and the cube deleted them, so a raw-table count
    reported 3 on both sides and proved nothing. Returns the count and the tickers -- the
    identity of the survivors IS the evidence (JPM 2023 = 0.31, INTC 2023 = 0.34, SPG 2024 = 0.111).
    """
    if df.empty or "say_on_pay_support_pct" not in df.columns:
        return None
    from src.data_aggregate.utils.governance.def14a_impute import impute_def14a
    cleaned, _ = impute_def14a(df.copy())
    v = pd.to_numeric(cleaned["say_on_pay_support_pct"], errors="coerce")
    d = cleaned[(v > 0) & (v < 0.50)]
    return int(len(d)), sorted(d["ticker"].unique().tolist())


def _director_comp_coverage(dc: pd.DataFrame, llm: pd.DataFrame) -> float | None:
    """Share of post-2008 proxies that produced at least one director-comp row (G7). 2008 is
    the first season the Item 402(k) table exists (Reg S-K 2006, FY ending >= 2006-12-15)."""
    if llm.empty or "as_of" not in llm.columns:
        return None
    proxies = llm[pd.to_datetime(llm["as_of"], errors="coerce") >= "2008-01-01"]
    if proxies.empty:
        return None
    have = set() if dc.empty else set(dc["accession_number"].dropna())
    return round(float(proxies["accession_number"].isin(have).mean()), 4)


def _gender_metrics(llm: pd.DataFrame, dirs: pd.DataFrame) -> dict[str, object]:
    """G12-G14. `gender` is KEPT because it carries alpha, so the bar is accuracy up at no cost
    in coverage -- fill must not fall, every non-null gender must carry a basis, and the filing's
    own women-director count must agree with the per-director count on more filings."""
    m: dict[str, object] = {"pct_female_fill": _fill(llm, "pct_female_directors")}
    if not dirs.empty and "gender" in dirs.columns:
        g = dirs[dirs["gender"].notna()]
        m["gender_rows"] = int(len(g))
        if "gender_basis" in dirs.columns:
            m["basis_coverage"] = round(float(g["gender_basis"].notna().mean()), 4) if len(g) else None
            m["basis_dist"] = g["gender_basis"].value_counts(dropna=False).to_dict()
    if not llm.empty and "n_women_directors_vs_inferred" in llm.columns:
        v = pd.to_numeric(llm["n_women_directors_vs_inferred"], errors="coerce").dropna()
        m["pct_women_count_agrees"] = round(float((v == 0).mean()), 4) if len(v) else None
    return m


def _vote_coverage(votes_new: pd.DataFrame, item507: pd.DataFrame) -> float | None:
    """Share of Item 5.07 filings carrying a comma-grouped number that produced vote rows (G11).

    The denominator is deliberately the comma-number subset, not every 5.07 filing: 8.8% of them
    have no vote table at all (a genuine 5.07(d) board-response filing), and the fabrication
    guard's whole job is to emit NOTHING there."""
    if item507.empty or "item_text" not in item507.columns or votes_new.empty:
        return None
    has_num = item507["item_text"].fillna("").str.contains(r"\d{1,3}(?:,\d{3})+", regex=True)
    denom = item507[has_num]
    if denom.empty:
        return None
    got = set() if votes_new.empty else set(votes_new["accession_number"].dropna())
    return round(float(denom["accession_number"].isin(got).mean()), 4)


def _payload(d: Path) -> float | None:
    """Mean carve payload chars (G10), written by the Phase-6 runner as it extracts. Asserted
    from a measurement, never from an estimate."""
    p = d / "payload.json"
    if not p.exists():
        return None
    vals = json.loads(p.read_text(encoding="utf-8")).get("payload_chars", [])
    return round(float(np.mean(vals)), 1) if vals else None


# --------------------------------------------------------------------------- #
# gates                                                                       #
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.1f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={v[k]}" for k in sorted(v))
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v)) or "-"
    return f"{v:,}" if isinstance(v, int) else str(v)


def build_gates(base: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame],
                base_dir: Path, new_dir: Path) -> list[dict]:
    """The G1-G14 table. Each gate carries its own `check`, so a gate with no `new` side yet
    reports `-` and PENDING rather than a spurious FAIL."""
    b_llm, n_llm = base["def14a_llm"], new["def14a_llm"]
    # same-accession population for every fill comparison
    shared = (set(b_llm.get("accession_number", pd.Series(dtype=str)))
              & set(n_llm.get("accession_number", pd.Series(dtype=str)))) or None

    b_null, n_null = _pct_fully_null(b_llm, "2001-01-01"), _pct_fully_null(n_llm, "2001-01-01")
    b_sop, n_sop = _low_say_on_pay(b_llm), _low_say_on_pay(n_llm)
    b_gender = _gender_metrics(b_llm, base["def14a_directors"])
    n_gender = _gender_metrics(n_llm, new["def14a_directors"])

    def pct(x):
        return None if x is None else round(100 * x, 2)

    gates = [
        dict(id="G1", name="exec-comp rows with a component > $1e9",
             base=_n_implausible(base["sec_def14a_executive_comp"], EXEC_COMPONENTS),
             new=_n_implausible(new["sec_def14a_executive_comp"], EXEC_COMPONENTS),
             req="0", check=lambda v: v == 0),
        dict(id="G2", name="ownership rows with a footnote-digit share count",
             base=_n_footnote_digit(base["sec_def14a_ownership"]),
             new=_n_footnote_digit(new["sec_def14a_ownership"]),
             req="0", check=lambda v: v == 0),
        dict(id="G3", name="sec_def14a rows with peo_total_comp == 0.0",
             base=_peo_zero(base["sec_def14a"]), new=_peo_zero(new["sec_def14a"]),
             req="0", check=lambda v: v == 0),
        dict(id="G4", name="PEOs recorded for BA / NKE co-PEO years",
             base=_or_none(_co_peo(base["sec_def14a"])), new=_or_none(_co_peo(new["sec_def14a"])),
             req="2 each", check=lambda v: bool(v) and all(n >= 2 for n in v.values())),
        dict(id="G5", name="% of 2012+ accessions with n_neos == 1",
             base=pct(_pct_single_neo(b_llm)), new=pct(_pct_single_neo(n_llm)),
             req="< 8%", check=lambda v: v < 8.0),
        dict(id="G6", name="auditor_name fill (def14a_llm; baseline = sec_def14a)",
             base=pct(_fill(base["sec_def14a"], "auditor_name")),
             new=pct(_fill(n_llm, "auditor_name")),
             req="> 80%", check=lambda v: v > 80.0),
        dict(id="G7", name="% of post-2008 proxies with >=1 director-comp row",
             base=pct(_director_comp_coverage(base["sec_def14a_director_comp"], b_llm)),
             new=pct(_director_comp_coverage(new["sec_def14a_director_comp"], n_llm)),
             req="> 90%", check=lambda v: v > 90.0),
        dict(id="G8", name="say-on-pay values < 0.50 present",
             base=None if b_sop is None else b_sop[0], new=None if n_sop is None else n_sop[0],
             req=">= 3", check=lambda v: v >= 3),
        dict(id="G9", name="pre-2001 rows fully NULL",
             base=None if b_null is None else pct(b_null[0] / b_null[1]),
             new=None if n_null is None else pct(n_null[0] / n_null[1]),
             req="< 10%", check=lambda v: v < 10.0),
        # 45,000 and not the plan's original 40,000: that number came from a 36,544-char estimate
        # that predates the same plan's +3,000 widenings of PAY RATIO / SAY ON PAY and its new
        # AUDITOR NAME slice, so it was unreachable as specified. 45,000 leaves 6.6% over the
        # measured 42,225 -- enough to absorb a different filing mix at universe scale, not enough
        # to absorb a defect. It is deliberately NOT 50,000: the baseline is 51,198, so a gate
        # there would pass any build that merely is not worse than the code being replaced, and
        # would not reliably trip when the table classifier silently finds nothing and all three
        # fallback slices (7,000 + 10,000 + 2,500) fire on every filing -- a failure this phase
        # actually hit once, via an lxml encoding-declaration error hidden behind a bare except.
        dict(id="G10", name="mean carve payload chars",
             base=_payload(base_dir), new=_payload(new_dir),
             req="<= 45,000", check=lambda v: v <= 45_000),
        # The baseline is 0 BY CONSTRUCTION -- no code parses Item 5.07 today -- so the
        # denominator (the 5.07 corpus) always comes from the baseline snapshot, and only the
        # numerator moves. Passing the corpus as its own numerator would report a hollow 100%.
        dict(id="G11", name="% of 5.07 filings w/ a comma-number that yielded vote rows",
             base=0.0 if not base["sec_8k_item507"].empty else None,
             new=pct(_vote_coverage(new["sec_8k_item507"], base["sec_8k_item507"])),
             req="> 90%", check=lambda v: v > 90.0),
        dict(id="G12", name="pct_female_directors fill",
             base=pct(b_gender.get("pct_female_fill")), new=pct(n_gender.get("pct_female_fill")),
             req="must not fall",
             check=lambda v, b=b_gender.get("pct_female_fill"): b is None or v >= 100 * b - 1e-9),
        dict(id="G13", name="gender_basis populated wherever gender is set",
             base=pct(b_gender.get("basis_coverage")), new=pct(n_gender.get("basis_coverage")),
             req="100%", check=lambda v: v >= 100.0),
        dict(id="G14", name="% of filings where n_women_directors_vs_inferred == 0",
             base=pct(b_gender.get("pct_women_count_agrees")),
             new=pct(n_gender.get("pct_women_count_agrees")),
             req="> baseline",
             check=lambda v, b=b_gender.get("pct_women_count_agrees"): b is None or v > 100 * b),
    ]
    for g in gates:
        g["status"] = "PENDING" if g["new"] is None else ("PASS" if _safe(g["check"], g["new"]) else "FAIL")
    gates.append(dict(id="--", name="say-on-pay survivors (tickers)", status="INFO",
                      base=None if b_sop is None else b_sop[1],
                      new=None if n_sop is None else n_sop[1], req="JPM/INTC/SPG"))
    gates.append(dict(id="--", name="gender_basis distribution", status="INFO",
                      base=b_gender.get("basis_dist"), new=n_gender.get("basis_dist"), req="-"))
    return gates


def _safe(check, value) -> bool:
    try:
        return bool(check(value))
    except Exception:
        return False


def render(gates: list[dict], base: dict, new: dict) -> str:
    lines = ["# DEF 14A extraction fix — baseline vs. new", "",
             "Generated by `scripts/compare_def14a_baseline.py`. Fill rates are measured over the",
             "same accession set on both sides, so a coverage change cannot read as a quality change.",
             "", "## Row counts", "",
             "| table | baseline | new |", "|---|---|---|"]
    for k in list(TABLE_MAP) + ["sec_def14a_votes", "def14a_directors"]:
        b, n = base.get(k, pd.DataFrame()), new.get(k, pd.DataFrame())
        if b.empty and n.empty:
            continue
        lines.append(f"| `{k}` | {len(b):,} | {len(n):,} |")

    lines += ["", "## Gates", "", "| # | check | baseline | new | required | status |",
              "|---|---|---|---|---|---|"]
    for g in gates:
        lines.append(f"| {g['id']} | {g['name']} | {_fmt(g['base'])} | {_fmt(g['new'])} "
                     f"| {g['req']} | **{g['status']}** |")
    fails = [g["id"] for g in gates if g["status"] == "FAIL"]
    lines += ["", f"**{sum(g['status'] == 'PASS' for g in gates)} PASS / "
                  f"{len(fails)} FAIL / {sum(g['status'] == 'PENDING' for g in gates)} PENDING**"]
    if fails:
        lines.append(f"Failing gates: {', '.join(fails)} — fix in the owning phase and re-run.")

    own = base.get("sec_def14a_ownership", pd.DataFrame())
    have = sorted(own["ticker"].unique()) if not own.empty else []
    lines += ["", "## Baseline caveats", "",
              "- **G2 cannot be demonstrated on the baseline side.** The edgar HTML backfill had only "
              f"reached {len(have)} of the 23 tickers ({', '.join(have)}) when the snapshot was taken, "
              "and PG — the ticker carrying the footnote-digit defect — is not among them. The gate "
              "still guards the NEW side, which is what it is for.",
              "- **G8 is measured THROUGH `impute_def14a`**, the cube's whole clean-on-read stage, "
              "because that is where a nulling rule would bite: back when the stage held a 0.50 "
              "floor, `def14a_llm` kept the revolts and the cube deleted them. ⚠ This makes G8 "
              "the one gate whose BASELINE column is not frozen: the parquet tables are, but this "
              "row re-runs live `src/` code over them, so it reads 3 once Phase 1 removes the floor "
              "and read 0 before. The true pre-fix baseline is **0**, recorded in "
              "`PHASE-0-baseline-harness.md`. Every other gate is a pure query over the frozen "
              "tables and is therefore reproducible at any commit.",
              "- **G9 excludes the four inferred-FALSE provision flags.** Today's prompt orders the "
              "model to return FALSE for them, so a row extracted from a 10 KB folder index still "
              "looks populated; 17 of 26 pre-2001 baseline rows carry nothing else."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the DEF 14A baseline with the new run.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--baseline-only", action="store_true",
                    help="print the 'before' picture with the new column empty")
    args = ap.parse_args()

    out = Path(args.out)
    base = load_side(out / "baseline", "baseline")
    new = ({k: pd.DataFrame() for k in base} if args.baseline_only
           else load_side(out / "new", "new"))

    gates = build_gates(base, new, out / "baseline", out / "new")
    report = render(gates, base, new)
    print(report)
    (out / "COMPARISON.md").write_text(report, encoding="utf-8")
    print(f"written -> {out / 'COMPARISON.md'}")


if __name__ == "__main__":
    main()
