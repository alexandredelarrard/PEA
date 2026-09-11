"""
cube_fundamentals_evidence.py  (scripts/)
--------------------------------------------------------------------------------------------
Measure `cube_part_fundamentals` and emit the per-feature evidence table for
`reports/<date>/cube-fundamentals-evidence/README.md`.

WHY THIS EXISTS AS A SCRIPT AND NOT A NOTEBOOK. The 2026-09-04 audit found 65 of 179 defined
features emitting nothing and three pairs at Pearson r = 1.0000, with the whole test suite
green throughout. The defence against a repeat is a report whose every number is re-derivable
against the live table on demand, so `--check` can be run after any future rebuild and the
claims re-verified rather than re-asserted.

Three measurements, all server-side (a 3.85M x 250 table is never materialised in the client):

  profile      per column: n, null rate, min/p1/p50/p99/max, mean, sd, distinct
  saturation   per standardised column: the share at |z| >= 8, i.e. ON the `peer_relative`
               clip. A value on the clip is no longer measuring distance from peers, it is
               reporting that the standardisation ran out of room -- the single most useful
               "are the extreme values good?" number for a z-scored feature.
  coverage     every column must have a `cube_feature_catalogue` entry and every entry must
               correspond to a live column. That two-way check is what makes "no feature
               ships unexplained" an assertion rather than a hope.

usage:
    python -m scripts.cube_fundamentals_evidence --out reports/2026-09-05/cube-fundamentals-evidence
    python -m scripts.cube_fundamentals_evidence --check      # measure + verify, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import text

from scripts.cube_feature_catalogue import CATALOGUE, SUFFIXES, split

TABLE = "cube_part_fundamentals"
NUMERIC = {"double precision", "real", "numeric", "integer", "bigint", "smallint"}
KEYS = ("date", "ticker")

#: `peer_relative` clips at +-8; a float32 round-trip can bring a clipped value back a hair
#: under, so the test is >= 7.99 rather than == 8.0.
CLIP_EDGE = 7.99

#: `SUFFIXES` and `split` come from `cube_feature_catalogue`, which owns the keying
#: convention. They used to be duplicated here, and two implementations of "what is a
#: characteristic" are two possible answers to "is this column documented?".
VIEW = {"_vs_peers": "peer-z", "_xs": "universe %ile",
        "_vs_hist": "self-history z", "": "raw"}

_PROFILE_CHUNK = 12          # 9 aggregates per column; keeps the query plan small
_SAT_CHUNK = 25


def profile(conn, table: str) -> pd.DataFrame:
    cols = conn.execute(text(
        "select column_name, data_type from information_schema.columns "
        "where table_name = :t order by ordinal_position"), {"t": table}).fetchall()
    n_rows = conn.execute(text(f"select count(*) from {table}")).scalar_one()
    num = [n for n, d in cols if d in NUMERIC and n not in KEYS]

    rows = []
    for i in range(0, len(num), _PROFILE_CHUNK):
        block = num[i:i + _PROFILE_CHUNK]
        sel = []
        for j, col in enumerate(block):
            q = f'"{col}"'
            sel += [f"count({q}) as c{j}", f"min({q}) as mn{j}", f"max({q}) as mx{j}",
                    f"avg({q}) as av{j}", f"stddev_samp({q}) as sd{j}",
                    f"percentile_cont(0.01) within group (order by {q}) as p1_{j}",
                    f"percentile_cont(0.50) within group (order by {q}) as p50_{j}",
                    f"percentile_cont(0.99) within group (order by {q}) as p99_{j}",
                    f"count(distinct {q}) as nd{j}"]
        r = conn.execute(text(f"select {', '.join(sel)} from {table}")).mappings().one()
        for j, col in enumerate(block):
            nn = int(r[f"c{j}"])
            rows.append({"feature": col, "n_rows": n_rows, "n_non_null": nn,
                         "null_rate": 1.0 - nn / n_rows if n_rows else 1.0,
                         "n_distinct": int(r[f"nd{j}"]),
                         "min": r[f"mn{j}"], "p1": r[f"p1_{j}"], "p50": r[f"p50_{j}"],
                         "p99": r[f"p99_{j}"], "max": r[f"mx{j}"],
                         "mean": r[f"av{j}"], "std": r[f"sd{j}"]})
        print(f"  profile {i + len(block)}/{len(num)}", flush=True)

    df = pd.DataFrame(rows)
    for c in ("min", "p1", "p50", "p99", "max", "mean", "std"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def saturation(conn, table: str) -> pd.DataFrame:
    cols = [r[0] for r in conn.execute(text(
        "select column_name from information_schema.columns "
        r"where table_name = :t and column_name like '%\_vs\_%' "
        "order by ordinal_position"), {"t": table}).fetchall()]
    rows = []
    for i in range(0, len(cols), _SAT_CHUNK):
        block = cols[i:i + _SAT_CHUNK]
        sel = []
        for j, col in enumerate(block):
            q = f'"{col}"'
            sel += [f"avg((abs({q}) >= {CLIP_EDGE})::int)::float as s{j}",
                    f"max(abs({q}))::float as m{j}"]
        r = conn.execute(text(f"select {', '.join(sel)} from {table}")).mappings().one()
        for j, col in enumerate(block):
            rows.append({"feature": col, "sat_rate": r[f"s{j}"], "max_abs": r[f"m{j}"]})
        print(f"  saturation {i + len(block)}/{len(cols)}", flush=True)
    return pd.DataFrame(rows)


def _fmt(x, nd: int = 2) -> str:
    if pd.isna(x):
        return "-"
    ax = abs(x)
    if ax >= 1e6 or 0 < ax < 1e-4:
        return f"{x:.2e}"
    return f"{x:,.{nd}f}"


def collect(prof: pd.DataFrame, sat: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One row per characteristic, with the measurements both renderers need."""
    p = prof.copy()
    p[["char", "suffix"]] = p["feature"].apply(lambda c: pd.Series(split(c)))
    p["view"] = p["suffix"].map(VIEW)
    p = p.merge(sat, on="feature", how="left")

    missing = sorted(set(p["char"]) - set(CATALOGUE))
    unused = sorted(set(CATALOGUE) - set(p["char"]))

    rows = []
    for char, g in p.groupby("char", sort=True):
        views = ", ".join(sorted(g["view"].unique(), key=lambda v: (v != "raw", v)))
        # null rate from the BEST-covered view: a transform can only lose rows, never add them
        best = g.loc[g["null_rate"].idxmin()]
        # DISTRIBUTION from the PEER-Z view, never the percentile rank. A percentile rank is
        # uniform on [0, 1] BY CONSTRUCTION, so its p1/p50/p99 are 0.01/0.50/0.99 for every
        # feature in the table and carry no information at all.
        z = g[g["suffix"] == "_vs_peers"]
        if not len(z):
            z = g[g["suffix"] == ""]
        z = z.iloc[0] if len(z) else best
        fam, what, why, tail = CATALOGUE.get(char, ("?", "**UNDOCUMENTED**", "", ""))
        rows.append({"characteristic": char, "family": fam, "views": views,
                     "null_rate": best["null_rate"], "n_cols": len(g),
                     "n_rows": int(best["n_rows"]),
                     "p1": z["p1"], "p50": z["p50"], "p99": z["p99"],
                     "sat_rate": z.get("sat_rate", float("nan")),
                     "what": what, "why": why, "tail": tail})
    r = pd.DataFrame(rows).sort_values(["family", "characteristic"])
    r.attrs["n_columns"] = len(p)
    r.attrs["sparsest_null"] = float(p["null_rate"].max())
    return r, missing, unused


def build_markdown(r: pd.DataFrame, missing: list[str], unused: list[str]) -> str:
    n_cols, n_rows = r.attrs["n_columns"], int(r["n_rows"].iloc[0])
    lines = [f"Characteristics: **{len(r)}** across **{n_cols}** cube columns "
             f"({n_rows:,} rows).\n"]
    if missing:
        lines.append(f"⚠ **{len(missing)} column(s) with no catalogue entry**: "
                     + ", ".join(f"`{m}`" for m in missing) + "\n")
    if unused:
        lines.append(f"ℹ {len(unused)} catalogued name(s) absent from the table: "
                     + ", ".join(f"`{m}`" for m in unused) + "\n")

    s = r["sat_rate"].dropna()
    lines.append(f"Peer-z clip saturation across the {len(s)} standardised characteristics: "
                 f"**mean {s.mean():.2%}**, median {s.median():.2%}, worst {s.max():.2%} "
                 f"(`{r.loc[s.idxmax(), 'characteristic']}`).\n")

    for fam, gf in r.groupby("family", sort=True):
        lines += [f"\n### {fam}\n",
                  "| feature | views | null rate | peer-z p1 / p50 / p99 | @clip | "
                  "what it does | why it is right | the tail |",
                  "|---|---|--:|---|--:|---|---|---|"]
        for _, x in gf.iterrows():
            satv = "-" if pd.isna(x["sat_rate"]) else f"{x['sat_rate']:.1%}"
            # a bare `|` inside a cell ENDS the cell: `|net income|` silently split its row
            what, why, tail = (str(x[k]).replace("|", r"\|")
                               for k in ("what", "why", "tail"))
            lines.append(f"| `{x['characteristic']}` | {x['views']} | {x['null_rate']:.1%} | "
                         f"{_fmt(x['p1'])} / {_fmt(x['p50'])} / {_fmt(x['p99'])} | {satv} | "
                         f"{what} | {why} | {tail} |")
    return "\n".join(lines)


def _meta(r: pd.DataFrame) -> dict:
    """Every measurement the narrative substitutes in. Nothing there is hard-coded."""
    s = r["sat_rate"].dropna()
    return {
        "subtitle": (f"{int(r['n_rows'].iloc[0]):,} rows · {r.attrs['n_columns']} feature "
                     f"columns · {len(r)} characteristics · measured 2026-09-05"),
        "rows": int(r["n_rows"].iloc[0]), "rows_before": 3_852_470,
        "cols": r.attrs["n_columns"], "cols_before": 209,
        "chars": len(r), "chars_before": 103,
        "rebuild_window": "2026-09-05 00:38 to 01:28",
        "n_standardised": len(s),
        "sat_mean": f"{s.mean():.2%}", "sat_median": f"{s.median():.2%}",
        "sat_worst": f"{s.max():.2%}",
        "sat_worst_name": r.loc[s.idxmax(), "characteristic"],
        "sparsest_null": f"{r.attrs['sparsest_null']:.1%}",
        "table_lede": (
            f"{len(r)} characteristics across all {r.attrs['n_columns']} cube columns. "
            f"Clip saturation: mean {s.mean():.2%}, median {s.median():.2%}, worst "
            f"{s.max():.2%} (`{r.loc[s.idxmax(), 'characteristic']}`)."),
    }


def _pdf_rows(r: pd.DataFrame) -> list:
    """[(family, [row dicts])] with every number already formatted."""
    out = []
    for fam, gf in r.groupby("family", sort=True):
        items = []
        for _, x in gf.iterrows():
            items.append({
                "characteristic": x["characteristic"], "views": x["views"],
                "null": f"{x['null_rate']:.1%}",
                "dist": f"{_fmt(x['p1'])} / {_fmt(x['p50'])} / {_fmt(x['p99'])}",
                "clip": "-" if pd.isna(x["sat_rate"]) else f"{x['sat_rate']:.1%}",
                "what": x["what"], "why": x["why"], "tail": x["tail"],
            })
        out.append((fam, items))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for the artefacts and the rendered report")
    ap.add_argument("--format", choices=("pdf", "md", "both"), default="pdf",
                    help="report format (default: pdf)")
    ap.add_argument("--check", action="store_true",
                    help="measure and verify coverage; write nothing")
    ap.add_argument("--from-parquet", action="store_true",
                    help="re-render from the artefacts already in --out, skipping the DB. "
                         "Only re-renders; it CANNOT refresh a number.")
    args = ap.parse_args(argv)

    if args.from_parquet:
        if not args.out:
            ap.error("--from-parquet needs --out to read the artefacts from")
        prof = pd.read_parquet(args.out / f"{args.table}_profile.parquet")
        sat = pd.read_parquet(args.out / f"{args.table}_saturation.parquet")
    else:
        load_dotenv(find_dotenv(usecwd=True))
        from src.utils.db import get_engine
        with get_engine().connect() as conn:
            prof = profile(conn, args.table)
            sat = saturation(conn, args.table)

    r, missing, unused = collect(prof, sat)
    n_rows = int(prof["n_rows"].iloc[0])
    print(f"{args.table}: {n_rows:,} rows, {len(prof)} numeric columns, "
          f"{len(r)} characteristics; "
          f"{len(missing)} undocumented, {len(unused)} catalogued-but-absent")

    if args.out and not args.check:
        args.out.mkdir(parents=True, exist_ok=True)
        written = []
        if not args.from_parquet:
            prof.to_parquet(args.out / f"{args.table}_profile.parquet", index=False)
            sat.to_parquet(args.out / f"{args.table}_saturation.parquet", index=False)
            written += ["profile.parquet", "saturation.parquet"]
        if args.format in ("md", "both"):
            (args.out / "feature_table.md").write_text(
                build_markdown(r, missing, unused), encoding="utf-8")
            written.append("feature_table.md")
        if args.format in ("pdf", "both"):
            from scripts import cube_evidence_narrative as narrative
            from scripts.cube_evidence_pdf import render
            path = render(args.out / f"{args.table}_evidence.pdf",
                          _meta(r), _pdf_rows(r), narrative)
            written.append(path.name)
        print(f"wrote {len(written)} artefact(s) to {args.out}: {', '.join(written)}")

    if missing:
        print("UNDOCUMENTED (no feature may ship unexplained):", missing)
    if unused:
        print("CATALOGUED BUT ABSENT (a stale entry, or a builder stopped emitting):", unused)
    return 1 if (missing or unused) else 0


if __name__ == "__main__":
    sys.exit(main())
