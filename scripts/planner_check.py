"""Planner check (paper log D-61): does MRCM's tree as one more recipe let our planner catch the
methods that beat it, without losing elsewhere?

The sharing pass (`share.share_plan`) prices four recipes per delivery group and takes the
cheapest per delivery. `share_mrcm` adds a fifth: MRCM's cost-weighted tree for the exact aim,
pool droplets offered as e-MRCM offers shared ones (`share.mrcm_droplet`). Per problem both are
run here (cost and seconds); the published methods' costs come from the lit-bench records
(`notes/lit_bench/runs/e22/records.jsonl`: DFM, FacDA, Generic, DFMM, MRCM on s1-s2;
`notes/lit_bench/runs/full/records.jsonl`, if present: e-MRCM and NFSP on s3-s5).

Problems: the six val sets (`val_s1_v2` ... `val_s5_v2`, `heldout_combo_v2`, 242) and
`litbench_s2_v1` (100 fresh single-target mixtures, MRCM's home ground).

Reading, fixed in advance (D-61):
- **Check (not a result):** on set B (the 13 `val_s2_v2` problems where MRCM beats our best
  planner) `share_mrcm` is at or below MRCM on all 13. On one target with demand 1 MRCM's whole
  tree is one of the candidates, so this holds by construction; a miss is a bug (raised).
- **(1) gain beyond MRCM's own problems:** on stages 3-5 and the held-out combination (162),
  where MRCM alone does not apply (demand > 1 or many targets), `share_mrcm` is cheaper than
  `share` on >= 1/4 of problems;
- **(2) keeps generality:** on the 242 val problems `share_mrcm` is dearer than `share` on
  <= 5% (12) and its mean cost / `share` is <= 1.
Both: MRCM's tree joins the planner (the user's go-ahead). (2) only: it fixes single-target
mixtures and costs nothing elsewhere: it joins the planner too, as a fix for s2. (1) only: a
separate variant. Neither: drop it. Reported, no bar: per set and method, how many problems
that method beats `share` on, and `share_mrcm` on; time per problem.

    uv run python scripts/planner_check.py workers=30 out=outputs/planner/records.jsonl
    uv run python scripts/planner_check.py summary=outputs/planner/records.jsonl

Recipe choice (D-62, after E-57): `share_mrcm` was dearer than `share` on 17 of 242, all with
many targets: the cheapest recipe per delivery left a pool that served later targets worse.
`share_mrcm_look3` rolls each delivery group's 3 cheapest recipes out to the end and keeps the
best (the pass's existing `lookahead`); `share_look3` is the same lookahead without MRCM's tree,
the fair comparison. Reading on the 242 val problems, fixed in advance (D-62):
**the greedy choice was the problem** if `share_mrcm_look3` is dearer than `share` on <= 5%
(12) with mean cost / `share` <= 1: the MRCM recipe with lookahead joins the planner, and
choosing recipes is worth a faster chooser (search or a network, next). Otherwise it is not
only the greedy choice: stop there. Reported, no bar: against `share_look3` (cheaper / dearer,
mean ratio), how many of E-57's 17 it fixes, time.

    uv run python scripts/planner_check.py sets=val planners=share,share_mrcm,share_look3,\
share_mrcm_look3 workers=30 out=outputs/planner/look3.jsonl

Search against the stronger planner (D-63, after E-58): E-36/E-39's search on set C (99 s3-s5
problems) was 16% below the best planner of three variants. With `share_mrcm_look3` as a fourth,
the planner alone is about 12% cheaper; how much of the search's margin is left? No new search:
the race records (`search=`, E-39's `race.jsonl`) and this script's records (`look=`, E-58's
`look3.jsonl`), per problem the new planner = min(old best planner, `share_mrcm_look3`). The
search's cost is its whole run (900 s cap, its finishes still the plain pass). Reading on all
of C, fixed in advance (D-63), for `uniform` (flat prior + planner finish, the method):
**the search still earns its place** if it is below the new planner on >= 1/3 of C (D-37's
bar); otherwise the planner's greedy choices were most of what the search was fixing. Reported,
no bar: per stage and arm, below / mean ratio against the old and the new planner, and the new
planner / old.

    uv run python scripts/planner_check.py search=outputs/expert/round2/race.jsonl \
look=outputs/planner/look3.jsonl

Stronger finisher (D-64, after E-59): the search's leaves finished by the new planner
(`search_check.py finishers=strong`, flat prior, set C, the same 900 s cap), its records added
to `search=` (comma list). Reading on all of C, fixed in advance (D-64): **a stronger finisher
makes the search add more** if `uniform/strong`'s mean cost / new planner is at least 0.02 below
`uniform`'s (plain finishes) and it is below the new planner on more problems; otherwise the
search adds about as much whichever finisher, and its gain over our best planner is E-59's.

Arguments are key=value: `sets` (comma list; `val` = the six val sets), `planners` (default
share,share_mrcm), `limit` problems per set, `workers`, `out`; or `summary=`; or `search=` with
`look=`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.problem import Instance
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom  # noqa: E402

WEIGHTS = headroom.WEIGHTS
VAL = ("val_s1_v2", "val_s2_v2", "val_s3_v2", "val_s4_v2", "val_s5_v2", "heldout_combo_v2")
SETS = (*VAL, "litbench_s2_v1")
LIT_S12 = ROOT / "notes/lit_bench/runs/e22/records.jsonl"
LIT_FULL = ROOT / "notes/lit_bench/runs/full/records.jsonl"
METHODS = ("dfm", "facda", "generic", "dfmm", "mrcm", "emrcm", "nfsp")
PLANNERS = ("share", "share_look3", "share_orders", "share_mrcm_look3")
FULL = ("val_s3_v2", "val_s4_v2", "val_s5_v2", "heldout_combo_v2")
RUNS = {  # planner -> (lookahead, MRCM's tree)
    "share": (0, False),
    "share_mrcm": (0, True),
    "share_look3": (3, False),
    "share_mrcm_look3": (3, True),
}
# Fixed in advance (D-61).
GAIN_SHARE, DEARER_SHARE = 1 / 4, 0.05
FOUND_SHARE = 1 / 3  # D-63, D-37's bar
STRONG_GAP = 0.02  # D-64, D-43's gap


def literature() -> dict[tuple[str, str], dict[str, Any]]:
    """(source, id) -> {method: cost, "best_planner": cost} from the lit-bench records."""
    out: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for path in (LIT_S12, LIT_FULL):
        if not path.exists():
            continue
        for r in map(json.loads, path.open()):
            row = out[(r["source"], r["id"])]
            for m in METHODS:
                cost = (r.get(m) or {}).get("cost")
                if cost is not None:
                    row[m] = cost
            planners = [r[k]["cost"] for k in PLANNERS if (r.get(k) or {}).get("cost") is not None]
            if planners:
                row["best_planner"] = min(planners)
    return dict(out)


def problems(sets: list[str], limit: int | None) -> list[tuple[str, Instance]]:
    return [(s, i) for s in sets for i in BenchmarkSet.read(s).instances[:limit]]


def cost_of(inst: Instance, plan: Any) -> float:
    _, totals, _ = replay(inst, plan)
    return float(WEIGHTS.scalar(totals, inst.fluids))


def run_task(task: tuple[str, Instance] | tuple[str, Instance, list[str]]) -> dict[str, Any]:
    source, inst = task[0], task[1]
    names = task[2] if len(task) > 2 else ["share", "share_mrcm"]  # type: ignore[misc]
    rec: dict[str, Any] = {"source": source, "id": inst.instance_id}
    for name in names:
        look, mrcm = RUNS[name]
        t0 = time.perf_counter()
        plan = share.share_plan(inst, WEIGHTS, lookahead=look, mrcm=mrcm)
        if plan is None:
            raise RuntimeError(f"{inst.instance_id}: {name} found no plan")
        rec[name] = {"cost": cost_of(inst, plan), "seconds": round(time.perf_counter() - t0, 3)}
    return rec


# --------------------------------------------------------------------------- summary


def set_b(recs: list[dict[str, Any]], lit: dict[tuple[str, str], dict[str, Any]]) -> list[Any]:
    """Set B (search_check): val_s2 problems where MRCM beats our best planner."""
    out = []
    for r in recs:
        row = lit.get((r["source"], r["id"]), {})
        if r["source"] == "val_s2_v2" and "mrcm" in row and row["mrcm"] < row["best_planner"]:
            out.append((r, row["mrcm"]))
    return out


def reading(recs: list[dict[str, Any]], lit: dict[tuple[str, str], dict[str, Any]]) -> str:
    missed = [r["id"] for r, m in set_b(recs, lit) if r["share_mrcm"]["cost"] > m + 1e-9]
    if missed:  # holds by construction on one target with demand 1 (rule 9)
        raise RuntimeError(f"share_mrcm dearer than MRCM on set B: {missed}")
    full = [r for r in recs if r["source"] in FULL]
    gain = sum(1 for r in full if r["share_mrcm"]["cost"] < r["share"]["cost"] - 1e-9)
    val = [r for r in recs if r["source"] in VAL]
    dearer = sum(1 for r in val if r["share_mrcm"]["cost"] > r["share"]["cost"] + 1e-9)
    ratio = float(np.mean([r["share_mrcm"]["cost"] / r["share"]["cost"] for r in val]))
    one = gain >= GAIN_SHARE * len(full)
    two = dearer <= DEARER_SHARE * len(val) and ratio <= 1.0
    head = (
        f"(1) cheaper than share on {gain} of {len(full)} of s3-s5 + held-out; (2) dearer on "
        f"{dearer} of {len(val)}, mean ratio {ratio:.3f}: "
    )
    if one and two:
        return head + "MRCM's tree joins the planner (user's go-ahead)"
    if two:
        return head + "a fix for single-target mixtures at no cost elsewhere: it joins too"
    if one:
        return head + "gains on s3-s5 but costs elsewhere: a separate variant only"
    return head + "neither: drop it"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    lit = literature()
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[r["source"]].append(r)
    lines = [
        "share_mrcm vs share (cost ratio; cheaper / equal / dearer; median seconds), and how many",
        "problems each published method beats share on -> beats share_mrcm on (of those it takes)",
        "",
        "| set | n | ratio | cheaper / equal / dearer | seconds share / mrcm | methods: beat share"
        " -> beat share_mrcm |",
        "|---|---|---|---|---|---|",
    ]
    for s in [x for x in SETS if x in by]:
        rs = by[s]
        a = [r["share_mrcm"]["cost"] for r in rs]
        p = [r["share"]["cost"] for r in rs]
        cheap = sum(1 for x, y in zip(a, p, strict=True) if x < y - 1e-9)
        dear = sum(1 for x, y in zip(a, p, strict=True) if x > y + 1e-9)
        meth = []
        for m in METHODS:
            takes = [(r, lit[(s, r["id"])][m]) for r in rs if m in lit.get((s, r["id"]), {})]
            if takes:
                before = sum(1 for r, c in takes if c < r["share"]["cost"] - 1e-9)
                after = sum(1 for r, c in takes if c < r["share_mrcm"]["cost"] - 1e-9)
                meth.append(f"{m} {before}->{after}/{len(takes)}")
        lines.append(
            f"| {s} | {len(rs)} | {np.mean(np.array(a) / np.array(p)):.3f} | {cheap} / "
            f"{len(rs) - cheap - dear} / {dear} | "
            f"{np.median([r['share']['seconds'] for r in rs]):.1f} / "
            f"{np.median([r['share_mrcm']['seconds'] for r in rs]):.1f} | {', '.join(meth)} |"
        )
    lines += ["", f"Reading (fixed in advance, D-61): {reading(recs, lit)}"]
    lines += look_reading(recs)
    return "\n".join(lines)


def _vs(rs: list[dict[str, Any]], a: str, b: str) -> tuple[int, int, float]:
    """(cheaper, dearer, mean ratio) of planner `a` against `b`."""
    x = [(r[a]["cost"], r[b]["cost"]) for r in rs]
    cheap = sum(1 for u, v in x if u < v - 1e-9)
    dear = sum(1 for u, v in x if u > v + 1e-9)
    return cheap, dear, float(np.mean([u / v for u, v in x]))


def look_reading(recs: list[dict[str, Any]]) -> list[str]:
    """D-62: does lookahead over recipes fix the greedy choice's losses?"""
    val = [r for r in recs if r["source"] in VAL and "share_mrcm_look3" in r]
    if not val:
        return []
    lines = ["", "Recipe choice (D-62), on the val sets: cheaper / dearer, mean ratio"]
    for a, b in (
        ("share_mrcm", "share"),
        ("share_mrcm_look3", "share"),
        ("share_look3", "share"),
        ("share_mrcm_look3", "share_look3"),
    ):
        if all(a in r and b in r for r in val):
            c, d, m = _vs(val, a, b)
            lines.append(f"  {a} vs {b}: {c} / {d}, {m:.3f}")
    lost = [r for r in val if r["share_mrcm"]["cost"] > r["share"]["cost"] + 1e-9]
    fixed = sum(1 for r in lost if r["share_mrcm_look3"]["cost"] <= r["share"]["cost"] + 1e-9)
    secs = {k: float(np.median([r[k]["seconds"] for r in val])) for k in RUNS if k in val[0]}
    lines.append(f"  share_mrcm's losses fixed by lookahead: {fixed} of {len(lost)}")
    lines.append("  median seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in secs.items()))
    _, dear, ratio = _vs(val, "share_mrcm_look3", "share")
    head = f"dearer than share on {dear} of {len(val)}, mean ratio {ratio:.3f}: "
    if dear <= DEARER_SHARE * len(val) and ratio <= 1.0:
        verdict = "the greedy choice was the problem; MRCM's tree with lookahead joins the planner"
    else:
        verdict = "not only the greedy choice: stop there"
    return [*lines, f"Reading on recipe choice (fixed in advance, D-62): {head}{verdict}"]


def search_recheck(search: list[dict[str, Any]], look: list[dict[str, Any]]) -> list[str]:
    """D-63: the search on set C against the planner with `share_mrcm_look3` added."""
    new_cost = {(r["source"], r["id"]): r["share_mrcm_look3"]["cost"] for r in look}
    rows: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for r in search:
        if not r["set"].startswith("C") or r["sims"] != 256:
            continue
        key = (f"val_s{r['set'][1]}_v2", r["id"])
        if key not in new_cost:
            raise ValueError(f"{key}: no share_mrcm_look3 cost in the look records")
        old = r["planner"]
        new = min(old, new_cost[key])
        found = old if r["search_cost"] is None else min(old, r["search_cost"])
        fin = r.get("finisher", "share")
        kind = r["kind"] if fin == "share" else f"{r['kind']}/{fin}"
        for group in (r["set"], "C"):
            rows[(group, kind)].append((found, old, new))
    if not rows:
        raise ValueError("no set C records at 256 simulations in the search records")
    lines = [
        "Search on set C against the old and the new planner (D-63): below / mean ratio",
        "",
        "| set | arm | n | vs old planner | vs new planner | new planner / old |",
        "|---|---|---|---|---|---|",
    ]
    for (group, kind), xs in sorted(rows.items()):
        f, o, n = (np.array(v) for v in zip(*xs, strict=True))
        lines.append(
            f"| {group} | {kind} | {len(xs)} | {int(np.sum(f < o - 1e-9))}, {np.mean(f / o):.3f} | "
            f"{int(np.sum(f < n - 1e-9))}, {np.mean(f / n):.3f} | {np.mean(n / o):.3f} |"
        )
    if ("C", "uniform") not in rows:
        return lines
    lines += strong_reading(rows)
    xs = rows[("C", "uniform")]
    below = sum(1 for f, _, n in xs if f < n - 1e-9)
    ratio = float(np.mean([f / n for f, _, n in xs]))
    head = f"uniform below the new planner on {below} of {len(xs)}, mean ratio {ratio:.3f}: "
    if below >= FOUND_SHARE * len(xs):
        verdict = "the search still earns its place over the stronger planner"
    else:
        verdict = "the planner's greedy choices were most of what the search was fixing"
    return [*lines, "", f"Reading (fixed in advance, D-63): {head}{verdict}"]


def strong_reading(rows: dict[tuple[str, str], list[tuple[float, float, float]]]) -> list[str]:
    """D-64: does the search add more over the new planner when its leaves use that planner?"""
    if ("C", "uniform/strong") not in rows:
        return []

    def against_new(xs: list[tuple[float, float, float]]) -> tuple[int, float]:
        return sum(1 for f, _, n in xs if f < n - 1e-9), float(np.mean([f / n for f, _, n in xs]))

    plain_below, plain = against_new(rows[("C", "uniform")])
    below, strong = against_new(rows[("C", "uniform/strong")])
    head = (
        f"with strong finishes below the new planner on {below}, mean ratio {strong:.3f} "
        f"(plain finishes {plain_below}, {plain:.3f}): "
    )
    if strong <= plain - STRONG_GAP and below > plain_below:
        verdict = "a stronger finisher makes the search add more"
    else:
        verdict = "the search adds about as much whichever finisher; its gain is what E-59 shows"
    return ["", f"Reading on finishers (fixed in advance, D-64): {head}{verdict}"]


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    if "search" in args:

        def read(paths: str) -> list[dict[str, Any]]:
            return [json.loads(x) for p in paths.split(",") for x in Path(p).open()]

        print("\n".join(search_recheck(read(args["search"]), read(args["look"]))))
        return
    sets = args.get("sets", ",".join(SETS)).split(",")
    if sets == ["val"]:
        sets = list(VAL)
    limit = int(args["limit"]) if "limit" in args else None
    names = args.get("planners", "share,share_mrcm").split(",")
    unknown = set(names) - set(RUNS)
    if unknown or not {"share", "share_mrcm"} <= set(names):
        raise ValueError(f"planners must include share and share_mrcm, from {list(RUNS)}")
    todo = [(s, i, names) for s, i in problems(sets, limit)]
    out = Path(args.get("out", "outputs/planner/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with out.open("w") as fh, Pool(workers) as pool:
        for done, rec in enumerate(pool.imap_unordered(run_task, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {len(todo)} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
