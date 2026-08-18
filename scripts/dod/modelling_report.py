"""
modelling_report.py  (scripts/dod/modelling_report.py)
------------------------------------------------------
The MODELLING Definition-of-Done report. A **PURE READER**: it never trains, never re-fits and
never touches the model artifacts. Everything it needs was already written by
`src/modelling/long_short/utils/diagnostics.py` under

    <OUTPUT_DIR>/diagnostics/<run_stamp>/
        kpis.json / kpis.csv          run level, one CSV row per (horizon, member)
        h<H>/kpis.json               this horizon, incl. the `members` dict
        h<H>/ic_over_time.png
        h<H>/[<member>/]shap_importance.{png,csv}   flat for ONE member, sub-folder for 2+

    "$PY" scripts/dod/modelling_report.py --slug retrain-h5
    "$PY" scripts/dod/modelling_report.py --slug retrain-h5 --compare-run 20260801-1200

Gates
    M1  the run's kpis.json / kpis.csv parse
    M2  CV IC and OOS IC are FINITE for every horizon
    M3  SHAP present for every BOOSTER member (a linear member has none -- an accepted
        reason, but one that must still be STATED rather than passed over)
    M4  at least one PDP per booster member
    M5  OOS IC not worse than --compare-run beyond --ic-tolerance

Design notes
  * MEMBERS COME FROM `kpis.json["members"]`, NEVER FROM GLOBBING DIRECTORIES. With exactly one
    booster member the writer puts the artifacts FLAT in `h<H>/`; with two or more each gets
    `h<H>/<member>/`. `pdp/` exists in BOTH layouts, so a directory glob would read `pdp` as a
    member name in the flat case. The kpis file is the only unambiguous source.
  * A MEMBER THAT GOT DIAGNOSTICS IS A BOOSTER. `save_member_diagnostics` is called only for
    tree members, and it is what writes `shap_available` / `n_pdp` into the member dict. A
    member carrying only `cv_mean_ic` (elasticnet, added by the CV-KPI merge in `step_train`)
    never had SHAP to begin with, so M3/M4 report it as N/A **with the reason named**.
  * PNGs ARE COPIED, NOT LINKED. `data/` is gitignored, so a report linking into it is dead for
    every reader but the person who ran it. A bounded set (one SHAP plot per member, one IC
    curve per horizon, hard cap) is copied into `reports/<YYYY-MM-DD>/assets/<slug>/` -- inside
    the report's own day folder, so pruning a day takes its plots with it.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dod.report_common import (                               # noqa: E402
    Gate, announce, metrics_table, report_dir, repo_root, write_report,
)
# The writer's own filesystem-safe name function. Imported rather than re-implemented: if the
# two ever disagree the reader silently looks in the wrong folder.
from src.modelling.long_short.utils.diagnostics import _safe as safe_name   # noqa: E402

GENERATOR = "scripts/dod/modelling_report.py@1"
MAX_ASSETS = 12
SHAP_TOP_N = 12
#: Members that legitimately have no SHAP/PDP. Named so M3/M4 can say WHY, not just "N/A".
LINEAR_MEMBERS = frozenset({"elasticnet", "enet", "ridge", "lasso", "linear", "ols"})


# --------------------------------------------------------------------------- #
# Locating a run                                                              #
# --------------------------------------------------------------------------- #
def resolve_run_dir(explicit: str | None, config: str) -> Path:
    """`--run-dir` (a path or a bare run_stamp), else the newest run under OUTPUT_DIR."""
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p
    from src.context import get_config_context                        # local: needs configs
    _, context = get_config_context(config, use_cache=False, save=False)
    diag = Path(context.paths["OUTPUT_DIR"]) / "diagnostics"
    if explicit:
        cand = diag / explicit
        if cand.is_dir():
            return cand
        raise SystemExit(f"no such run: {explicit} (looked in {diag})")
    if not diag.is_dir():
        raise SystemExit(f"no diagnostics directory at {diag} -- train a model first, or pass "
                         f"--run-dir")
    runs = sorted((d for d in diag.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    if not runs:
        raise SystemExit(f"{diag} has no run directories")
    return runs[-1]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def member_dir(horizon_dir: Path, member: str, n_boosters: int) -> Path:
    """Where `save_member_diagnostics` put this member's artifacts.

    ONE booster -> flat in `h<H>/` (the writer keeps that layout so existing readers of
    `h<H>/pdp/` still work); two or more -> `h<H>/<safe(member)>/`.

    It must be the count of BOOSTERS, not of `kpis.json["members"]`. The writer decides with
    `flat = len(boosters) == 1`, but `step_train` then merges its CV KPIs into the same dict and
    that merge can ADD members which never had diagnostics (elasticnet). An ensemble of
    `[elasticnet, lgbm]` therefore writes the FLAT layout while `members` has two entries --
    counting all of them would send this reader looking in `h<H>/lgbm/`, which does not exist."""
    return horizon_dir if n_boosters == 1 else horizon_dir / safe_name(member)


def is_booster(member_kpis: dict) -> bool:
    """True when this member actually went through `save_member_diagnostics`.

    That function is the ONLY writer of `shap_available` / `n_pdp`, and it is called only for
    tree members -- so the presence of either key is the discriminator."""
    return "shap_available" in member_kpis or "n_pdp" in member_kpis


def read_run(run_dir: Path) -> dict:
    """Everything the report needs, per horizon and per member. No model is loaded."""
    out: dict = {"run_dir": str(run_dir), "run_stamp": run_dir.name,
                 "kpis_json_ok": False, "kpis_csv_ok": False, "horizons": {}}

    run_kpis = _read_json(run_dir / "kpis.json")
    out["kpis_json_ok"] = isinstance(run_kpis, dict) and bool(run_kpis.get("horizons"))

    csv_path = run_dir / "kpis.csv"
    csv = None
    if csv_path.is_file():
        try:
            csv = pd.read_csv(csv_path)
            out["kpis_csv_ok"] = True
        except (OSError, ValueError, pd.errors.ParserError):
            csv = None

    horizons = (run_kpis or {}).get("horizons") or {}
    for h_key, hk in sorted(horizons.items(), key=lambda kv: int(kv[0])):
        h = int(h_key)
        hdir = run_dir / f"h{h}"
        members = dict(hk.get("members") or {})
        n_boosters = sum(1 for mk in members.values() if is_booster(mk))
        info: dict = {
            "horizon": h,
            "dir": str(hdir),
            "blend_weight": hk.get("blend_weight"),
            "cv_mean_ic": hk.get("cv_mean_ic"), "cv_ic_ir": hk.get("cv_ic_ir"),
            "oos_ic_mean": hk.get("oos_ic_mean"),
            "oos_ic_hit_rate": hk.get("oos_ic_hit_rate"),
            "oos_ic_days": hk.get("oos_ic_days"),
            "n_rows": hk.get("n_rows"), "n_tickers": hk.get("n_tickers"),
            "n_days": hk.get("n_days"),
            "ic_curve_png": str(hdir / "ic_over_time.png")
                            if (hdir / "ic_over_time.png").is_file() else None,
            "layout": "flat" if n_boosters == 1 else "per-member",
            "n_boosters": n_boosters,
            "members": {},
        }
        for name, mk in sorted(members.items()):
            booster = is_booster(mk)
            mdir = member_dir(hdir, name, n_boosters)
            # A NON-BOOSTER OWNS NO ARTIFACTS. In the flat layout `member_dir` returns the
            # horizon dir itself, so probing it for a CV-only member (elasticnet) would credit
            # that member with the BOOSTER's shap_importance.csv and pdp/ -- the report would
            # then claim SHAP exists for a linear model that never had any.
            shap_csv = (mdir / "shap_importance.csv") if booster else None
            shap_png = (mdir / "shap_importance.png") if booster else None
            pdp_dir = (mdir / "pdp") if booster else None
            top: list[dict] = []
            if shap_csv is not None and shap_csv.is_file():
                try:
                    s = pd.read_csv(shap_csv)
                    val = s.columns[-1]
                    key = s.columns[0]
                    top = [{"feature": r[key], "shap_mean_abs": float(r[val])}
                           for _, r in s.head(SHAP_TOP_N).iterrows()]
                except (OSError, ValueError, IndexError, pd.errors.ParserError):
                    top = []
            info["members"][name] = {
                "dir": str(mdir) if booster else None,
                "booster": booster,
                "cv_mean_ic": mk.get("cv_mean_ic"), "cv_ic_ir": mk.get("cv_ic_ir"),
                "n_features": mk.get("n_features"),
                "n_pdp": mk.get("n_pdp"),
                "shap_available": mk.get("shap_available"),
                "shap_csv": str(shap_csv) if (shap_csv and shap_csv.is_file()) else None,
                "shap_png": str(shap_png) if (shap_png and shap_png.is_file()) else None,
                "pdp_files": (sorted(p.name for p in pdp_dir.glob("pdp_*.png"))
                              if (pdp_dir and pdp_dir.is_dir()) else []),
                "shap_top": top,
            }
        out["horizons"][h] = info

    # the CSV is the artifact to diff between runs, so keep it verbatim for §3
    out["csv_rows"] = (csv.to_dict("records") if csv is not None else [])
    return out


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #
def _finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))                                # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def build_gates(run: dict, compare: dict | None, tolerance: float) -> list[Gate]:
    gates: list[Gate] = []
    horizons = run["horizons"]

    # ---- M1 --------------------------------------------------------------- #
    gates.append(Gate("M1", "run kpis.json / kpis.csv parse",
                      run["kpis_json_ok"] and run["kpis_csv_ok"],
                      f"kpis.json={'ok' if run['kpis_json_ok'] else 'MISSING/EMPTY'}, "
                      f"kpis.csv={'ok' if run['kpis_csv_ok'] else 'MISSING/UNREADABLE'}, "
                      f"{len(horizons)} horizon(s)"))

    # ---- M2 --------------------------------------------------------------- #
    bad = []
    for h, info in horizons.items():
        missing = [k for k in ("cv_mean_ic", "oos_ic_mean") if not _finite(info.get(k))]
        if missing:
            bad.append(f"h{h}: {', '.join(missing)} not finite")
    gates.append(Gate("M2", "CV and OOS IC finite for every horizon",
                      None if not horizons else not bad,
                      "; ".join(bad) if bad
                      else f"finite across {len(horizons)} horizon(s)"))

    # ---- M3 / M4: boosters only ------------------------------------------- #
    boosters = [(h, name, m) for h, info in horizons.items()
                for name, m in info["members"].items() if m["booster"]]
    linear = [(h, name) for h, info in horizons.items()
              for name, m in info["members"].items() if not m["booster"]]
    linear_note = ("; ".join(f"h{h}:{n}" for h, n in linear)
                   + " -- no SHAP by construction (linear member, never passed to "
                     "save_member_diagnostics)") if linear else ""

    no_shap = [f"h{h}:{n}" for h, n, m in boosters
               if not (m["shap_available"] or m["shap_csv"] or m["shap_png"])]
    gates.append(Gate("M3", "SHAP present for every booster member",
                      None if not boosters else not no_shap,
                      (f"MISSING for {', '.join(no_shap)}. " if no_shap else
                       f"present for {len(boosters)} booster member(s). ")
                      + (f"Stated: {linear_note}" if linear_note else "")))

    no_pdp = [f"h{h}:{n}" for h, n, m in boosters
              if not (m["pdp_files"] or (m["n_pdp"] or 0) > 0)]
    gates.append(Gate("M4", ">=1 PDP per booster member",
                      None if not boosters else not no_pdp,
                      f"MISSING for {', '.join(no_pdp)}" if no_pdp
                      else f"{sum(len(m['pdp_files']) or (m['n_pdp'] or 0) for _, _, m in boosters)}"
                           f" PDP(s) across {len(boosters)} booster member(s)"))

    # ---- M5: versus a previous run ---------------------------------------- #
    if compare is None:
        gates.append(Gate("M5", "OOS IC not worse than the compared run", None,
                          "no --compare-run given, so no regression claim is made"))
    else:
        worse = []
        for h, info in horizons.items():
            prev = (compare["horizons"].get(h) or {}).get("oos_ic_mean")
            now = info.get("oos_ic_mean")
            if not (_finite(prev) and _finite(now)):
                continue
            if float(now) < float(prev) - tolerance:
                worse.append(f"h{h}: {float(prev):+.4f} -> {float(now):+.4f}")
        gates.append(Gate("M5", "OOS IC not worse than the compared run", not worse,
                          (f"regressed beyond {tolerance:g}: " + "; ".join(worse)) if worse
                          else f"within {tolerance:g} of run {compare['run_stamp']}"))
    return gates


# --------------------------------------------------------------------------- #
# Assets                                                                      #
# --------------------------------------------------------------------------- #
def copy_assets(run: dict, slug: str, root: Path) -> tuple[list[str], list[str], int]:
    """Copy a BOUNDED set of PNGs into `reports/<YYYY-MM-DD>/assets/<slug>/`.

    Inside the day folder, not a shared `reports/assets/`, so pruning a day removes its plots
    with it instead of leaving orphaned images nothing links to.

    Returns `(links, repo_paths, skipped)`: `links` are relative to the REPORT (a plain
    `assets/<slug>/x.png`, since the folder is now a sibling -- routing them up through the repo
    root and back down would work but reads as if the plots lived somewhere else), `repo_paths`
    are repo-relative for the machine-readable record."""
    day = report_dir(root)
    dest = day / "assets" / safe_name(slug)
    wanted: list[Path] = []
    for _, info in sorted(run["horizons"].items()):
        if info["ic_curve_png"]:
            wanted.append(Path(info["ic_curve_png"]))
        for _, m in sorted(info["members"].items()):
            if m["shap_png"]:
                wanted.append(Path(m["shap_png"]))
    keep, skipped = wanted[:MAX_ASSETS], max(0, len(wanted) - MAX_ASSETS)
    if not keep:
        return [], [], 0
    dest.mkdir(parents=True, exist_ok=True)
    links: list[str] = []
    repo_paths: list[str] = []
    for src in keep:
        # flatten h5/lgbm/shap_importance.png -> h5__lgbm__shap_importance.png
        try:
            rel = src.relative_to(Path(run["run_dir"]))
            flat = "__".join(rel.parts)
        except ValueError:
            flat = src.name
        target = dest / flat
        try:
            shutil.copy2(src, target)
        except OSError:
            continue
        links.append(target.relative_to(day).as_posix())
        repo_paths.append(target.relative_to(root).as_posix())
    return links, repo_paths, skipped


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MODELLING Definition-of-Done report")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--run-dir", default=None, help="run_stamp or path (default: newest run)")
    ap.add_argument("--compare-run", default=None, help="M5: run_stamp or path to compare against")
    ap.add_argument("--ic-tolerance", type=float, default=0.002,
                    help="M5: allowed OOS IC decline (default 0.002)")
    ap.add_argument("--config", default="./configs")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args(argv)

    root = repo_root()
    run_dir = resolve_run_dir(args.run_dir, args.config)
    run = read_run(run_dir)

    compare = None
    if args.compare_run:
        compare = read_run(resolve_run_dir(args.compare_run, args.config))

    gates = build_gates(run, compare, args.ic_tolerance)
    asset_links, assets, skipped = copy_assets(run, args.slug, root)

    horizon_rows = [{
        "horizon": f"h{h}", "layout": i["layout"], "members": len(i["members"]),
        "boosters": i["n_boosters"],
        "blend_weight": i["blend_weight"], "cv_mean_ic": i["cv_mean_ic"],
        "cv_ic_ir": i["cv_ic_ir"], "oos_ic_mean": i["oos_ic_mean"],
        "oos_ic_hit_rate": i["oos_ic_hit_rate"], "oos_ic_days": i["oos_ic_days"],
        "n_rows": i["n_rows"], "n_tickers": i["n_tickers"], "n_days": i["n_days"],
    } for h, i in sorted(run["horizons"].items())]

    member_rows = [{
        "horizon": f"h{h}", "member": name, "booster": m["booster"],
        "cv_mean_ic": m["cv_mean_ic"], "cv_ic_ir": m["cv_ic_ir"],
        "n_features": m["n_features"], "n_pdp": m["n_pdp"] or len(m["pdp_files"]),
        "shap": bool(m["shap_available"] or m["shap_csv"]),
    } for h, i in sorted(run["horizons"].items()) for name, m in sorted(i["members"].items())]

    shap_md = []
    for h, i in sorted(run["horizons"].items()):
        for name, m in sorted(i["members"].items()):
            if not m["shap_top"]:
                continue
            feats = ", ".join(f"`{t['feature']}` ({t['shap_mean_abs']:.4g})"
                              for t in m["shap_top"])
            shap_md.append(f"- **h{h} / {name}** top {len(m['shap_top'])} by mean|SHAP|: {feats}")

    metrics_parts = [
        "_Observed values only — read straight out of the run's own diagnostics; nothing here "
        "was recomputed and no model was reloaded._",
        "**Per horizon**",
        metrics_table(horizon_rows, ["horizon", "layout", "members", "boosters",
                                    "blend_weight", "cv_mean_ic", "cv_ic_ir", "oos_ic_mean",
                                    "oos_ic_hit_rate", "oos_ic_days", "n_rows",
                                    "n_tickers", "n_days"]),
        "**Per member**",
        metrics_table(member_rows, ["horizon", "member", "booster", "cv_mean_ic", "cv_ic_ir",
                                   "n_features", "n_pdp", "shap"]),
    ]
    if shap_md:
        metrics_parts += ["**SHAP importance**", "\n".join(shap_md)]

    evidence_lines = [f"- run: `{run['run_stamp']}` (`{run['run_dir']}`)"]
    if compare:
        evidence_lines.append(f"- compared against: `{compare['run_stamp']}`")
    evidence_lines += [f"- ![{Path(a).name}]({a})" for a in asset_links]
    if skipped:
        evidence_lines.append(f"- _{skipped} further plot(s) not copied (cap {MAX_ASSETS}); "
                              f"see `{run['run_dir']}`._")
    if not assets:
        evidence_lines.append("- **no PNG artifacts found** — check that diagnostics ran")

    scope_md = "\n".join([
        f"**Run read:** `{run['run_stamp']}` — {len(run['horizons'])} horizon(s), "
        f"member layout(s): "
        f"{', '.join(sorted({i['layout'] for i in run['horizons'].values()})) or 'n/a'}.",
        "",
        "**SAMPLE SCOPE** — as recorded by the run itself:",
        "",
        metrics_table([{"horizon": f"h{h}", "n_rows": i["n_rows"], "n_tickers": i["n_tickers"],
                        "n_days": i["n_days"], "oos_ic_days": i["oos_ic_days"]}
                       for h, i in sorted(run["horizons"].items())],
                      ["horizon", "n_rows", "n_tickers", "n_days", "oos_ic_days"]),
    ])

    payload = {
        "scope": {"run_dir": run["run_dir"], "run_stamp": run["run_stamp"],
                  "compare_run": (compare or {}).get("run_stamp"),
                  "ic_tolerance": args.ic_tolerance},
        "metrics": {"horizons": horizon_rows, "members": member_rows,
                    "csv_rows": run["csv_rows"], "assets": assets},
    }

    path = write_report("MODELLING", args.slug, generator=GENERATOR, gates=gates,
                        metrics_md="\n\n".join(metrics_parts),
                        evidence_md="\n".join(evidence_lines), payload=payload,
                        scope_md=scope_md, root=root, session_id=args.session_id)
    announce(path, gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
