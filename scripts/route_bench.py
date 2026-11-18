"""Plan 8.6 step 4: save the router calls of executor runs as a routing benchmark.

Each producer's plan of each problem (the producers of `grid_time.py`) is played on the grid
harness with A* wrapped in `RecordingRouter`; every call (requests, the chip state before it,
A*'s answer and time) is saved. One JSON line per run: set, instance, producer, platform,
outcome ("ok", "deadlock" or "error") and the calls (`RouteCall.to_json`). With `check=1`
(default) every saved call is replayed with A* and must give the same answer.

    OMP_NUM_THREADS=1 uv run python scripts/route_bench.py sets=val_s4_v2,val_s5_v2 \
        min_targets=8 sizes=30,60 units=5 turns=1 workers=4 out=outputs/route_bench/val.jsonl
    uv run python scripts/route_bench.py summary=outputs/route_bench/val.jsonl

Arguments are key=value: `sets`, `limit` per set, `min_targets`, `producers` (comma list, default
all), `methods`, `sizes`, `units`, `turns` (0/1), `check`, `workers`, `out`, as `grid_time.py`.
The summary gives per platform: runs, calls, requests per call, A* failures and time per call.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from meda.core.dag import MixingDAG
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.pipeline.executor import ExecutionDeadlock, GridExecutor, grid_opgraph
from meda.pipeline.geometry import GridPlatform, layout
from meda.pipeline.route_bench import RecordingRouter, RouteCall, gp_to_dict
from meda.pipeline.router import AStarRouter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import grid_time as gt  # noqa: E402
import sharing_time as st  # noqa: E402

OUTCOMES = ("ok", "deadlock", "error")


def record(inst: Instance, dag: MixingDAG, gp: GridPlatform, check: bool) -> dict[str, Any]:
    """One plan on one platform: its outcome and saved calls."""
    try:
        g = grid_opgraph(dag, inst, gp)
        chip = layout(inst, gp)
    except ValueError as e:
        return {"outcome": "error", "why": str(e)[:200], "calls": []}
    rec = RecordingRouter(AStarRouter())
    out: dict[str, Any] = {"outcome": "ok"}
    try:
        out["makespan"] = GridExecutor(gp, router=rec).run(g, chip).makespan
    except ExecutionDeadlock as e:
        out = {"outcome": "deadlock", "why": str(e)[:200]}
    if check:
        for i, call in enumerate(rec.calls):
            got = AStarRouter().route(list(call.requests), call.state(chip, gp))
            if got != call.result:
                raise AssertionError(f"{inst.instance_id} {gp.name}: call {i} replays differently")
    out["calls"] = [c.to_json() for c in rec.calls]
    return out


def run(task: dict[str, Any], opts: dict[str, Any]) -> list[dict[str, Any]]:
    inst = Instance.from_json(task["instance"])
    gps = gt.platforms(opts["sizes"], opts["units"], opts["moves"])
    lines = []
    for name, plan in st.plans(task, opts).items():
        if isinstance(plan, str) or (opts["producers"] and name not in opts["producers"]):
            continue
        _, totals, deltas = replay(inst, plan)
        dag = MixingDAG.from_deltas(inst, deltas, totals, name)
        for gp in gps.values():
            line = {
                "set": task["set"],
                "id": inst.instance_id,
                "producer": name,
                "platform": gp_to_dict(gp),
                "instance": task["instance"],
            }
            line.update(record(inst, dag, gp, opts["check"]))
            lines.append(line)
    return lines


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    return run(*args)


def summarise(lines: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in lines:
        by[x["platform"]["name"]].append(x)
    out = {}
    for v, xs in sorted(by.items()):
        calls = [RouteCall.from_json(c) for x in xs for c in x["calls"]]
        n = [len(c.requests) for c in calls]
        ms = sorted(1000 * c.seconds for c in calls)
        out[v] = {
            "runs": len(xs),
            "outcomes": {k: sum(x["outcome"] == k for x in xs) for k in OUTCOMES},
            "calls": len(calls),
            "requests_per_call": {
                "1": sum(k == 1 for k in n),
                "2": sum(k == 2 for k in n),
                "3+": sum(k >= 3 for k in n),
                "max": max(n, default=0),
            },
            "astar_failed": sum(not c.result.ok for c in calls),
            "ms_per_call": {
                "mean": round(statistics.fmean(ms), 2) if ms else None,
                "p95": round(ms[int(0.95 * (len(ms) - 1))], 2) if ms else None,
                "max": round(ms[-1], 2) if ms else None,
            },
        }
    return out


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        lines = []
        for path in args["summary"].split(","):
            with open(path) as fh:
                lines += [json.loads(line) for line in fh]
        print(json.dumps(summarise(lines), indent=2))
        return
    moves: dict[str, Any] = {}
    if "turns" in args:
        moves["turns"] = args["turns"] == "1"
    opts = {
        "methods": args["methods"].split(",") if "methods" in args else list(st.FAST),
        "method": False,
        "producers": set(args["producers"].split(",")) if "producers" in args else set(),
        "sizes": tuple(int(x) for x in args.get("sizes", "30,60").split(",")),
        "units": tuple(int(x) for x in args.get("units", "5").split(",")),
        "check": args.get("check", "1") == "1",
        "moves": moves,
    }
    sets = args["sets"].split(",") if "sets" in args else ["val_s4_v2", "val_s5_v2"]
    limit = int(args["limit"]) if "limit" in args else None
    tasks = [
        {"set": s, "instance": i.to_json()}
        for s in sets
        for i in [
            x
            for x in BenchmarkSet.read(s).instances
            if len(x.targets) >= int(args.get("min_targets", 0))
        ][:limit]
    ]
    out = Path(args.get("out", "outputs/route_bench/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done: list[dict[str, Any]] = []
    with open(out, "w") as fh, Pool(int(args.get("workers", 1))) as pool:
        for k, got in enumerate(pool.imap_unordered(_star, [(t, opts) for t in tasks]), 1):
            for line in got:
                fh.write(json.dumps(line) + "\n")
            fh.flush()
            done += got
            print(f"{k}/{len(tasks)} {got[0]['id'] if got else ''}", flush=True)
    print(json.dumps(summarise(done), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
