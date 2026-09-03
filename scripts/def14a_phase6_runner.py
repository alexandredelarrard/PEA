"""
def14a_phase6_runner.py  (scripts/)
--------------------------------------------------------------------------------------------
Phase-6 Step 1: run the NEW extraction path over the 23 baseline tickers into parquet, writing
NOTHING to the database. That separation is the whole point — `scripts/compare_def14a_baseline.py`
then gates `new/` against Phase 0's frozen `baseline/`, and the cutover only happens if the gate
passes. Nothing here can damage live data.

Three sub-runs, each independently skippable so a failure in one does not cost the others:

  --proxy   the LLM path off `data/cache/def14a_probe/` (Phase 0 put 744 files there, 709 MB):
            html_to_text -> prepare_def14a_sections -> LLMExtractor -> `_flatten` + the four
            child row builders. ZERO SEC requests. ~656 LLM calls.
  --ecd     the deterministic ECD/PVP block. This one DOES hit the network, because
            `filing.xbrl()` needs the XBRL attachments and Phase 0 cached only the primary
            document. Scoped to 2023+ filings (Item 402(v) applies to FY ending >= 2022-12-16,
            so an earlier proxy has no `ecd:` facts at all). No LLM cost.
  --votes   Item 5.07 tallies, reading `item_text` from the DB READ-ONLY and projected, then
            writing parquet. ~320 LLM calls.

Every LLM call's token usage is recorded (`LLMExtractor.totals`) and written to
`new/tokens.json`, because Phase 6 forbids starting the ~15,400-call full backfill on an
estimate. The per-filing carve payload lands in `new/payload.json`, which is what makes gate
G10 a measurement rather than an assertion.

    "$PY" scripts/def14a_phase6_runner.py [-c ./configs] [--proxy] [--ecd] [--votes]
                                          [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.llm_extractor import LLMExtractor
from src.data_extract.utils.schemas.def14a_schema import Def14AExtract
from src.data_extract.utils.structure.fetch_def14a_llm import (
    _DEF14A_PROMPT, _child_frames, _flatten, prepare_def14a_sections,
)
from src.data_extract.utils.structure.fetch_8k_votes_llm import (
    _SOURCE_COLS, _VOTES_PROMPT, _prepare_frame as _prepare_vote_frame, _proposal_rows,
    _role_map, _role_source, rejection_reason,
)
from src.data_extract.utils.schemas.vote_schema import Item507Extract
from src.data_store.schema import Tables

PLAN = ROOT / "reports/planning/active-tasks/2026-09-01-def14a-extraction-fix"
BASELINE, NEW = PLAN / "baseline", PLAN / "new"
CACHE_DIR = ROOT / "data/cache/def14a_probe"

#: gpt-5-mini list price per 1M tokens, for the spend estimate the full rerun is sized on.
#: Cached input is billed at a tenth of fresh input, which is why `prompt_cache_key` is set.
PRICE_IN, PRICE_CACHED_IN, PRICE_OUT = 0.25, 0.025, 2.00


def _usd(t: dict) -> float:
    fresh = max(t["input_tokens"] - t["cached_input_tokens"], 0)
    return (fresh * PRICE_IN + t["cached_input_tokens"] * PRICE_CACHED_IN
            + t["output_tokens"] * PRICE_OUT) / 1e6


def _filings() -> pd.DataFrame:
    """Phase 0's cached filing index, restricted to rows whose HTML is actually on disk."""
    f = pd.read_parquet(BASELINE / "filings.parquet")
    # Pre-2001 filings carry `primaryDocument == ""`, so `_doc_url` builds a bare DIRECTORY
    # url and `cache_htm` holds a ~10 KB EDGAR FOLDER INDEX rather than the proxy. Phase 0
    # cached the real document as `.txt` alongside it (159,203 chars on A's 2000 filing, vs
    # 10,217 for the index), and Phase 1's `_doc_url` fix is what makes production read it.
    # Reading `cache_htm` for those 88 filings fed the model a directory listing: they came
    # back with 0-2 non-null fields, which reads as "pre-2001 does not extract" when the real
    # cause is the runner opening the wrong artifact.
    #
    # `isinstance(str)` on BOTH hops, not a truthiness test: these are pyarrow-backed string
    # columns, so a null comes back as `pd.NA` (or a float nan once it round-trips through
    # `.map`), and `bool(nan)` is TRUE. Same trap that made the Phase-3 gender consensus count
    # filled rows as overturned ones.
    def _pick(row: pd.Series) -> str:
        for col in (("cache_txt", "cache_htm") if bool(row["pre_2001_empty_primary"])
                    else ("cache_htm", "cache_txt")):
            n = row[col]
            if isinstance(n, str) and n and (CACHE_DIR / n).exists():
                return str(CACHE_DIR / n)
        return ""

    f["path"] = f.apply(_pick, axis=1)
    have = f["path"].map(lambda p: isinstance(p, str) and bool(p))
    n_txt = int(f.loc[have, "path"].str.endswith(".txt").sum())
    print(f"  reading {int(have.sum())} filings ({n_txt} from the .txt document, i.e. the "
          f"pre-2001 filings whose primary document is a folder index)")
    missing = int((~have).sum())
    if missing:
        print(f"  NOTE: {missing} of {len(f)} indexed filings have no cached HTML — skipped")
    return f[have].sort_values(["ticker", "filing_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# --proxy : the LLM path                                                       #
# --------------------------------------------------------------------------- #
def run_proxy(context, model: str, limit: int | None, dry: bool, workers: int) -> None:
    filings = _filings()
    if limit:
        filings = filings.groupby("ticker", group_keys=False).head(limit)
    print(f"\n=== PROXY: {len(filings)} cached filings, {filings['ticker'].nunique()} tickers ===")
    if dry:
        print(filings.groupby("ticker").size().to_string())
        return

    extractor = LLMExtractor(model=model, max_chars=130_000, cache=True)
    parent: list[dict] = []
    children: dict[str, list[dict]] = {}
    payload_chars: list[int] = []
    failures: list[str] = []
    t0 = time.time()
    lock = threading.Lock()
    done = [0]

    def one(f: pd.Series) -> None:
        """Extract ONE filing. Runs in a worker thread; the only shared state is under `lock`."""
        ticker = f["ticker"]
        try:
            raw = Path(f["path"]).read_text(encoding="utf-8", errors="replace")
            text = html_to_text(raw)
            focused = prepare_def14a_sections(raw, text)
            extract = extractor.extract(Def14AExtract, focused, instructions=_DEF14A_PROMPT)
            row = _flatten(ticker, f, extract)
            kids = _child_frames(ticker, f, extract)
        except Exception as e:                      # one filing must not cost the run
            with lock:
                failures.append(f"{ticker} {f['filing_date']}: {type(e).__name__}: {e}")
            return
        with lock:
            payload_chars.append(len(focused))
            parent.append(row)
            for name, rows in kids.items():
                children.setdefault(name, []).extend(rows)
            done[0] += 1
            if done[0] % 25 == 0:
                el = time.time() - t0
                print(f"  {done[0]}/{len(filings)} filings, {el:.0f}s "
                      f"({el / done[0]:.1f}s/filing), ${_usd(extractor.totals):.2f} so far",
                      flush=True)

    # Concurrency is not an optimisation here, it is what makes the phase possible: measured
    # SERIALLY, one modern proxy takes ~94s (a 130k-char payload on a reasoning model), so 656
    # filings would be ~17h and the full 8,700-proxy backfill ~9.5 DAYS. The work is entirely
    # network-bound on an API that is fine with parallel requests, and ONE extractor is shared
    # so every worker keeps hitting the same cached prompt prefix.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, [f for _, f in filings.iterrows()]))

    NEW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parent).to_parquet(NEW / "def14a_llm.parquet", index=False)
    for name, rows in children.items():
        pd.DataFrame(rows).to_parquet(NEW / f"{name}.parquet", index=False)
    (NEW / "payload.json").write_text(
        json.dumps({"payload_chars": payload_chars}), encoding="utf-8")
    _write_tokens("proxy", extractor.totals, len(filings))

    print(f"\n  parent rows      : {len(parent)}")
    for name, rows in sorted(children.items()):
        print(f"  {name:<24}: {len(rows)}")
    print(f"  payload chars    : mean {pd.Series(payload_chars).mean():,.0f} / "
          f"median {pd.Series(payload_chars).median():,.0f} / "
          f"max {max(payload_chars):,}")
    print(f"  failures         : {len(failures)}")
    for msg in failures[:10]:
        print(f"    - {msg}")
    print(f"  spend            : ${_usd(extractor.totals):.2f} over "
          f"{extractor.totals['calls']} calls, {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# --ecd : the deterministic PVP block                                          #
# --------------------------------------------------------------------------- #
def run_ecd(context, limit: int | None, dry: bool) -> None:
    """Reuses the production builder verbatim rather than re-implementing it, so what the gate
    measures is what the pipeline will write. 2023+ only: an earlier proxy carries no `ecd:`
    facts (Item 402(v), FY ending >= 2022-12-16) and correctly gets no row."""
    from src.data_extract.utils.common.sec_utils import load_cik_mapping
    from src.data_extract.utils.structure.fetch_def14a_edgar import build_ticker_def14a_edgar

    # edgartools' identity is PROCESS-GLOBAL and the production fetcher sets it via the
    # Context; a script calling `build_ticker_def14a_edgar` directly bypasses that and every
    # request fails with IdentityNotSetError.
    context.ensure_edgar_identity()

    manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    tickers = manifest["tickers"]
    cutoff = pd.Timestamp("2023-01-01")
    print(f"\n=== ECD: {len(tickers)} tickers, filings from {cutoff.date()} "
          f"(needs live filing.xbrl()) ===")
    if dry:
        print("  tickers:", ", ".join(tickers))
        return

    # The Phase-0 filing index only covers the 22 tickers that HAD a cached proxy, so XOM is
    # absent from it; resolve CIKs the way the pipeline does instead of from that artifact.
    cik_df = load_cik_mapping(context, tickers)
    cik_map = dict(zip(cik_df["ticker"], cik_df["cik"].astype(str)))

    rows: list[dict] = []
    for t in tickers[: limit or len(tickers)]:
        cik = cik_map.get(t)
        if cik is None:
            print(f"  {t}: no CIK in the Phase-0 index — skipped")
            continue
        try:
            # returns {Table: DataFrame}, keyed by the registry object -- one entry since
            # Phase 4 slimmed this path to `sec_def14a` alone
            df = build_ticker_def14a_edgar(t, cik, since=cutoff)[Tables.def14a_edgar]
        except Exception as e:
            print(f"  {t}: FAILED {type(e).__name__}: {e}")
            continue
        n = 0 if df is None or df.empty else len(df)
        print(f"  {t}: {n} row(s)", flush=True)
        if n:
            rows.extend(df.to_dict("records"))

    NEW.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_parquet(NEW / "sec_def14a.parquet", index=False)
    print(f"\n  sec_def14a rows: {len(out)}")
    if len(out):
        print(f"  columns        : {len(out.columns)}")
        for c in ("n_peos", "peo_names_all", "peo_actually_paid_comp", "company_name"):
            if c in out.columns:
                print(f"  {c:<24} fill {out[c].notna().mean():.1%}")


# --------------------------------------------------------------------------- #
# --votes : Item 5.07                                                          #
# --------------------------------------------------------------------------- #
def _parquet_role_source(ticker: str) -> dict[str, pd.DataFrame]:
    """`_role_source`'s three frames, read from `new/` instead of the DB.

    Same keys and same columns, so `_role_map` is exercised unchanged -- the production
    function is a pure function of these frames, and swapping where they come from is what
    makes the role join measurable BEFORE the tables it reads exist.
    """
    def _one(stem: str, cols: list[str]) -> pd.DataFrame:
        p = NEW / f"{stem}.parquet"
        if not p.exists():
            return pd.DataFrame(columns=cols)
        df = pd.read_parquet(p)
        df = df[df["ticker"] == ticker] if "ticker" in df.columns else df.iloc[0:0]
        missing = [c for c in cols if c not in df.columns]
        for c in missing:
            df[c] = None
        return df[cols]

    return {
        "ceo": _one("def14a_llm", ["ticker", "as_of", "ceo_name_proxy"]),
        "exec": _one("def14a_executive_comp",
                     ["ticker", "as_of", "name", "title", "fiscal_year"]),
        "director": _one("def14a_director_comp", ["ticker", "as_of", "name"]),
    }



def run_votes(context, model: str, limit: int | None, dry: bool, workers: int) -> None:
    manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    tickers = manifest["tickers"]
    src = context.store.load(Tables.sec_8k, columns=list(_SOURCE_COLS),
                             where={"item": "5.07", "ticker": tickers})
    src = src.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    if limit:
        src = src.groupby("ticker", group_keys=False).head(limit)
    reasons = src["item_text"].map(rejection_reason)
    todo = src[reasons.isna()]
    print(f"\n=== VOTES: {len(src)} stored 5.07 filings, {len(todo)} readable "
          f"({len(src) - len(todo)} refused before any LLM call) ===")
    print(reasons.dropna().value_counts().to_string() or "  (nothing refused)")
    if dry:
        return

    extractor = LLMExtractor(model=model, max_chars=40_000, cache=True)
    rows: list[dict] = []
    rejected = 0
    t0 = time.time()
    lock = threading.Lock()
    done = [0]

    # The role map is built PER TICKER outside the pool: it is a pure function of three frames
    # plus the meeting date, so resolving it here keeps the workers off the DB entirely.
    #
    # And the frames come from `new/`, NOT from the database. The three `def14a_*` tables the
    # map joins to DO NOT EXIST in Postgres yet -- they are created by this cutover -- so a
    # DB-sourced map returns nothing and the measurement was 96.8% `unmatched`, which says
    # everything about the missing tables and nothing about the join. `new/` holds exactly what
    # Step 4 is about to write, so this measures the join that will actually run.
    role_sources = {str(t): _parquet_role_source(str(t)) for t in todo["ticker"].unique()}
    if not any(len(v["director"]) for v in role_sources.values()):
        print("  WARNING: no director-comp rows in new/ — run --proxy first or the role map "
              "will report everything as `unmatched`")

    def one(f: pd.Series) -> None:
        ticker = str(f["ticker"])
        roles, titles = _role_map(role_sources[ticker], f.get("period_of_report"))
        try:
            extract = extractor.extract(Item507Extract, f["item_text"],
                                        instructions=_VOTES_PROMPT)
        except Exception as e:
            with lock:
                print(f"  {ticker} {f['filing_date']}: FAILED {type(e).__name__}: {e}")
            return
        r, bad = _proposal_rows(ticker, f, extract, f["item_text"], roles, titles)
        with lock:
            rows.extend(r)
            nonlocal rejected
            rejected += bad
            done[0] += 1
            if done[0] % 40 == 0:
                el = time.time() - t0
                print(f"  {done[0]}/{len(todo)} filings, {el:.0f}s, {len(rows)} rows, "
                      f"${_usd(extractor.totals):.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, [f for _, f in todo.iterrows()]))

    NEW.mkdir(parents=True, exist_ok=True)
    out = _prepare_vote_frame(rows) if rows else pd.DataFrame()
    out.to_parquet(NEW / "sec_8k_votes.parquet", index=False)
    _write_tokens("votes", extractor.totals, len(todo))

    print(f"\n  sec_8k_votes rows      : {len(out)}")
    print(f"  guard rejections       : {rejected}")
    if len(out):
        elections = out[out["proposal_type"] == "director_election"]
        print(f"  director-election rows : {len(elections)}")
        if len(elections):
            nom = elections["n_nominees"].sum()
            unm = elections["n_nominees_unmatched"].sum()
            print(f"  nominees               : {nom:,.0f}, unmatched {unm:,.0f} "
                  f"({unm / nom:.1%}) <- the role join's honest error rate")
        flag = out["nominee_sum_matches"].dropna()
        if len(flag):
            print(f"  nominee_sum_matches    : {flag.mean():.1%} of {len(flag)} computable rows")
    print(f"  spend                  : ${_usd(extractor.totals):.2f} over "
          f"{extractor.totals['calls']} calls, {time.time() - t0:.0f}s")


def _write_tokens(kind: str, totals: dict, n_filings: int) -> None:
    """Accumulate token counts across sub-runs so the file survives running them separately."""
    NEW.mkdir(parents=True, exist_ok=True)
    p = NEW / "tokens.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data[kind] = {**totals, "filings": int(n_filings), "usd": round(_usd(totals), 4)}
    data["_price_per_1m"] = {"input": PRICE_IN, "cached_input": PRICE_CACHED_IN,
                             "output": PRICE_OUT}
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-6 Step 1: new path -> parquet, no DB writes.")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--ecd", action="store_true")
    ap.add_argument("--votes", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap filings per ticker (a cheap smoke test)")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent LLM calls (serial is ~94s/proxy -> 17h for 656)")
    ap.add_argument("--dry-run", action="store_true", help="count the work, spend nothing")
    args = ap.parse_args()

    config, context = get_config_context(args.config, use_cache=False, save=False)
    model = config.gpt.llm_model[config.gpt.default_api]
    print(f"model: {model} | out: {NEW}")

    if not (args.proxy or args.ecd or args.votes):
        ap.error("pick at least one of --proxy / --ecd / --votes")
    if args.proxy:
        run_proxy(context, model, args.limit, args.dry_run, args.workers)
    if args.ecd:
        run_ecd(context, args.limit, args.dry_run)
    if args.votes:
        run_votes(context, model, args.limit, args.dry_run, args.workers)


if __name__ == "__main__":
    main()
