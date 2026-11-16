"""Plan 8.4 step 6 / D-88 / Q-145: does sharing cost run time once droplets need space and travel?

Per problem of the paper test sets: each producer's plan (the same producers as
`sharing_time.py`: no-sharing construction, the sharing pass, our strongest planner, the fast
published methods, and with `method=1` our method) is played out on the grid harness (SPEC
14.6-14.10) on 9 variants: chip 30 x 30, 60 x 60, 100 x 100 x `unit_ticks` 1, 5, 20 (a 3 x 3 mix =
3, 15 or 60 cells travelled; L-19). Per plan and variant: the run's metrics after `validate_run`,
or "deadlock" (`ExecutionDeadlock`: the plan cannot run on that chip), or "error" (the harness
cannot lay it out, e.g. too many ports for the edges).

Reading (D-88, fixed before the run), per comparator, paired, many targets (s3-s5 + held-out) and
s1-s2 apart: (a) *runs*: the share of problems each finishes, among those both have a plan for;
sharing *costs feasibility* if ours finishes on at least 5 points fewer; (b) grid makespan ratio
ours / comparator on problems both finish, read with D-86's rule (`sharing_time.verdict`);
(c) beside it: `storage_cell_ticks` ratio and the travel overhead `makespan / schedule_makespan`.
The default variant is 60 x 60, unit_ticks 5. A verdict counts as robust if every variant where
both finish on at least 90% gives the same one. Also reported: per plan the smallest chip it runs
on (at unit_ticks 5), paired: ours needs a bigger / the same / a smaller chip.

    OMP_NUM_THREADS=1 uv run python scripts/grid_time.py workers=4 out=outputs/grid_time/r.jsonl
    uv run python scripts/grid_time.py summary=outputs/grid_time/r.jsonl

Arguments are key=value: `sets` (comma list, default the six test sets), `limit` per set,
`methods`, `method` (0/1), `seconds`, `sims` (as `sharing_time.py`), `sizes` (comma list, default
30,60,100), `units` (comma list, default 1,5,20), `check` (1: validate every run, default),
`free` (1: also play each plan with `FreeRouter` and record `free_makespan`, the routing gap
of D-89; 2: also `free_turn_makespan` and `free_split_makespan`, D-91; default 0),
`min_targets` (only problems with at least that many targets), `move_rule` and
`diagonal_rule` (override the platform's speed model, L-21), `turns` (0/1: whether A* may turn
droplets, 0.50), `workers`, `out`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.pipeline.executor import ExecutionDeadlock, GridExecutor, grid_opgraph
from meda.pipeline.geometry import GridPlatform, layout
from meda.pipeline.router import FreeRouter
from meda.pipeline.run_check import run_metrics, validate_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import goal_check as gc  # noqa: E402
import sharing_time as st  # noqa: E402

SIZES = (30, 60, 100)
UNITS = (1, 5, 20)
DEFAULT = "60x60_u5"
KEEP = (
    "makespan",
    "schedule_makespan",
    "travel_cells",
    "transit_ticks",
    "trip_floor",
    "storage_ticks",
    "storage_peak",
    "storage_cell_ticks",
    "peak_cells",
    "space_wait",
    "route_wait",
    "storage_wait",
    "route_failures",
)
_BASE = GridPlatform.load()


def variant(size: int, units: int) -> str:
    return f"{size}x{size}_u{units}"


def platforms(
    sizes: tuple[int, ...], units: tuple[int, ...], moves: dict[str, Any] | None = None
) -> dict[str, GridPlatform]:
    base = replace(_BASE, **(moves or {}))
    return {
        variant(n, u): replace(base, name=variant(n, u), grid=(n, n), unit_ticks=u)
        for n in sizes
        for u in units
    }


def play(
    inst: Instance, dag: MixingDAG, gp: GridPlatform, check: bool, free: int = 0
) -> dict[str, Any]:
    """One plan on one grid variant: its metrics, or why it did not run. With `free`, also the
    makespan of the same plan with `FreeRouter` (the routing gap, D-89)."""
    try:
        g = grid_opgraph(dag, inst, gp)
        chip = layout(inst, gp)
    except ValueError as e:
        return {"error": str(e)[:200]}
    try:
        run = GridExecutor(gp).run(g, chip)
    except ExecutionDeadlock as e:
        return {"deadlock": str(e)[:200]}
    if check:
        validate_run(run, g, chip, gp)
    m = run_metrics(run, g, chip, gp).as_dict()
    out: dict[str, Any] = {k: m[k] for k in KEEP}
    routers = {"free_makespan": FreeRouter()} if free else {}
    if free >= 2:  # D-91: the headroom of shape actions
        routers["free_turn_makespan"] = FreeRouter(turn=True)
        routers["free_split_makespan"] = FreeRouter(turn=True, split=True)
    for key, router in routers.items():
        try:
            out[key] = GridExecutor(gp, router=router).run(g, chip).makespan
        except ExecutionDeadlock:
            out[key] = None
    return out


def routing_gap(recs: list[dict[str, Any]], v: str) -> dict[str, Any]:
    """D-89 on one variant, every producer's plans pooled, over plans that run with both
    routers: means of makespan / schedule (A*), free makespan / schedule, the share of the A*
    makespan a perfect router could save (A* - free) / A*, and the share of trip ticks spent
    on detours and waits inside paths (transit - floor) / transit."""
    rows = [
        p[v]
        for r in recs
        for p in r["producers"].values()
        if v in p and p[v].get("free_makespan") is not None and "makespan" in p[v]
    ]
    if not rows:
        return {"n": 0}

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4)

    return {
        "n": len(rows),
        "astar_over_schedule": mean([x["makespan"] / x["schedule_makespan"] for x in rows]),
        "free_over_schedule": mean([x["free_makespan"] / x["schedule_makespan"] for x in rows]),
        "routing_share": mean([(x["makespan"] - x["free_makespan"]) / x["makespan"] for x in rows]),
        "routing_share_max": round(
            max((x["makespan"] - x["free_makespan"]) / x["makespan"] for x in rows), 4
        ),
        "detour_share": mean(
            [(x["transit_ticks"] - x["trip_floor"]) / max(x["transit_ticks"], 1) for x in rows]
        ),
        "route_wait_per_plan": mean([x["route_wait"] for x in rows]),
    }


def run(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    inst = Instance.from_json(task["instance"])
    w = gc.use_goal("fluid")
    rec: dict[str, Any] = {
        "set": task["set"],
        "id": inst.instance_id,
        "T": len(inst.targets),
        "F": len(inst.fluids),
        "producers": {},
    }
    gps = platforms(opts["sizes"], opts["units"], opts.get("moves"))
    for name, plan in st.plans(task, opts).items():
        if isinstance(plan, str):
            rec["producers"][name] = {"none": plan}
            continue
        _, totals, deltas = replay(inst, plan)
        dag = MixingDAG.from_deltas(inst, deltas, totals, name)
        out: dict[str, Any] = {"cost": float(w.scalar(totals, inst.fluids))}
        for vname, gp in gps.items():
            out[vname] = play(inst, dag, gp, opts["check"], opts.get("free", 0))
        rec["producers"][name] = out
    return rec


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run(*args)


def finished(p: dict[str, Any], v: str) -> bool:
    return v in p and "makespan" in p[v]


def compare(recs: list[dict[str, Any]], ours: str, other: str, v: str) -> dict[str, Any]:
    """D-88 (a)-(c) for one comparator on one variant."""
    both = [
        r
        for r in recs
        if "cost" in r["producers"].get(ours, {}) and "cost" in r["producers"].get(other, {})
    ]
    if not both:
        return {"n": 0}
    a_ok = sum(finished(r["producers"][ours], v) for r in both)
    b_ok = sum(finished(r["producers"][other], v) for r in both)
    rate_a, rate_b = a_ok / len(both), b_ok / len(both)
    t, st_ratio, over = [], [], []
    for r in both:
        a, b = r["producers"][ours], r["producers"][other]
        if finished(a, v) and finished(b, v):
            t.append(a[v]["makespan"] / b[v]["makespan"])
            st_ratio.append((a[v]["storage_cell_ticks"] + 1) / (b[v]["storage_cell_ticks"] + 1))
            over.append(a[v]["makespan"] / a[v]["schedule_makespan"])
    out: dict[str, Any] = {
        "n": len(both),
        "runs_ours": round(rate_a, 4),
        "runs_other": round(rate_b, 4),
        "feasibility": "costs feasibility" if rate_a <= rate_b - 0.05 else "no cost",
        "n_both_run": len(t),
    }
    if t:
        lo, hi = st.boot_ci(t)
        mean = sum(t) / len(t)
        out.update(
            {
                "time_ratio": round(mean, 4),
                "time_ci": [round(lo, 4), round(hi, 4)],
                "verdict": st.verdict(mean, lo, hi),
                "storage_cell_ratio": round(sum(st_ratio) / len(st_ratio), 4),
                "travel_overhead_ours": round(sum(over) / len(over), 4),
                "slower": sum(x > 1 for x in t),
                "faster": sum(x < 1 for x in t),
            }
        )
    return out


def smallest_chip(p: dict[str, Any], units: int) -> int | None:
    sizes = sorted(
        int(k.split("x")[0]) for k in p if k.endswith(f"_u{units}") and "makespan" in p[k]
    )
    return sizes[0] if sizes else None


def chip_sizes(recs: list[dict[str, Any]], ours: str, other: str) -> dict[str, int]:
    """Paired: ours needs a bigger / the same / a smaller chip (unit_ticks 5); `neither`: no
    variant runs either plan."""
    out = {"bigger": 0, "same": 0, "smaller": 0, "neither": 0}
    for r in recs:
        a, b = r["producers"].get(ours, {}), r["producers"].get(other, {})
        if "cost" not in a or "cost" not in b:
            continue
        sa, sb = smallest_chip(a, 5), smallest_chip(b, 5)
        if sa is None and sb is None:
            out["neither"] += 1
        elif sb is None or (sa is not None and sa < sb):
            out["smaller"] += 1
        elif sa is None or sa > sb:
            out["bigger"] += 1
        else:
            out["same"] += 1
    return out


def summarise(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ours = "method" if any("method" in r["producers"] for r in recs) else "share_mrcm_look3"
    variants = sorted(
        {k for r in recs for p in r["producers"].values() for k in p if "_u" in k},
        key=lambda k: (int(k.split("x")[0]), int(k.split("_u")[1])),
    )
    groups = {
        "many_targets": [r for r in recs if r["set"] in st.MANY],
        "s1_s2": [r for r in recs if r["set"] not in st.MANY],
    }
    out: dict[str, Any] = {"ours": ours, "variants": variants}
    for gname, rs in groups.items():
        block: dict[str, Any] = {}
        for other in st.COMPARATORS:
            if other == ours:
                continue
            block[other] = {v: compare(rs, ours, other, v) for v in variants}
            block[other]["chip_size"] = chip_sizes(rs, ours, other)
        out[gname] = block
    reading: dict[str, Any] = {}
    for other, per in out["many_targets"].items():
        d = per.get(DEFAULT, {})
        if not d.get("n_both_run"):
            continue
        firm = [
            per[v]
            for v in variants
            if per[v].get("n") and per[v]["runs_ours"] >= 0.9 and per[v]["runs_other"] >= 0.9
        ]
        reading[other] = {
            "time": d["verdict"],
            "feasibility": d["feasibility"],
            "robust": all(x.get("verdict") == d["verdict"] for x in firm) and bool(firm),
        }
    out["reading"] = reading
    if any(
        "free_makespan" in p.get(v, {})
        for r in recs
        for p in r["producers"].values()
        for v in variants
    ):
        out["routing_gap"] = {
            g: {v: routing_gap(rs, v) for v in variants} for g, rs in groups.items()
        }
    return out


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        recs = []
        for path in args["summary"].split(","):
            with open(path) as fh:
                recs += [json.loads(line) for line in fh]
        print(json.dumps(summarise(recs), indent=2))
        return
    moves: dict[str, Any] = {k: args[k] for k in ("move_rule", "diagonal_rule") if k in args}
    if "turns" in args:
        moves["turns"] = args["turns"] == "1"
    sets = args["sets"].split(",") if "sets" in args else list(st.SETS)
    limit = int(args["limit"]) if "limit" in args else None
    opts = {
        "methods": args["methods"].split(",") if "methods" in args else list(st.FAST),
        "method": args.get("method", "0") == "1",
        "seconds": float(args.get("seconds", 60)),
        "sims": int(args.get("sims", 256)),
        "sizes": tuple(int(x) for x in args.get("sizes", "30,60,100").split(",")),
        "units": tuple(int(x) for x in args.get("units", "1,5,20").split(",")),
        "check": args.get("check", "1") == "1",
        "free": int(args.get("free", "0")),
        "moves": moves,
    }
    out = Path(args.get("out", "outputs/grid_time/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "set": s,
            "cell": i.metadata.get("cell") or i.metadata.get("stage", ""),
            "instance": i.to_json(),
        }
        for s in sets
        for i in [
            x
            for x in BenchmarkSet.read(s).instances
            if len(x.targets) >= int(args.get("min_targets", 0))
        ][:limit]
    ]
    recs = []
    with open(out, "w") as fh, Pool(int(args.get("workers", 1))) as pool:
        for r in pool.imap_unordered(_star, [(t, opts) for t in tasks]):
            recs.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{len(recs)}/{len(tasks)} {r['id']}", flush=True)
    print(json.dumps(summarise(recs), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
