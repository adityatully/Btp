"""D-81: why the search alone loses to the 4 planner runs on some s4-s5 problems (E-77, E-79).

E-79: an hour instead of 15 minutes leaves 5 of A3's 10 losers dearer than the best run, and 3
of them do not improve at all after the switch to `tiered`. The runs that win are mostly
`share_mrcm_look3` (E-77): our planner with MRCM's trees and **lookahead 3**. The search
finishes every leaf **greedily** (lookahead 0). Guess: on big problems the search picks only
the first few moves of a long plan and the greedy finisher does the rest, so the plan is only
as good as a greedy finish, however long the search runs.

**Check.** Run method-v3's search (`switch`, first plan, CPU budget `seconds`) and take its
best plan. The part the tree chose (`prefix`, the plan minus the finisher's tail) is then
finished from where it leaves the chip three ways: the plain sharing pass, greedy MRCM (the
search's own finisher) and MRCM with lookahead 3 (the winning run's planner). Each total is
the prefix's cost plus the completion (`causal_check.finish_cost`).

Problems: the val s4-s5 problems where A3's search alone was dearer than the best run in
`records` (pilot A3's bench records, fluid; the same 10 as E-79), with A3's best run as the
reference (the runs are deterministic).

Reading (D-81, fixed before the run): **the greedy finisher is the root cause** if the search's
prefix finished with lookahead 3 is at most the best run on at least 8 of the 10 (the search's
moves are good, the greedy finish loses the rest). Otherwise the search's own moves are the
problem (they lead where even lookahead 3 cannot reach the best run).

    OMP_NUM_THREADS=1 uv run python scripts/finisher_check.py workers=10 \
        out=outputs/bench/finisher.jsonl
    uv run python scripts/finisher_check.py summary=outputs/bench/finisher.jsonl
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from meda.core.problem import Instance
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import causal_check as cc  # noqa: E402
import goal_check as gc  # noqa: E402
import search_check as sc  # noqa: E402

SETS = ("val_s4_v2", "val_s5_v2")


def alone(rec: dict[str, Any]) -> float | None:
    """The search alone's best cost in a bench record (its trace)."""
    t = [c for _, c in rec["ours"].get("trace", [])]
    return min(t) if t else None


def losers(records: Path) -> dict[str, dict[str, Any]]:
    """The val s4-s5 problems where the search alone was dearer than the best run (fluid)."""
    out = {}
    for line in records.read_text().splitlines():
        r = json.loads(line)
        if r.get("goal") != "fluid" or r["set"] not in SETS or r.get("best_planner") is None:
            continue
        a = alone(r)
        if a is not None and a > r["best_planner"] + 1e-9:
            costs = {n: s.get("cost") for n, s in r["planners"].items()}
            won = min((c, n) for n, c in costs.items() if c is not None)[1]
            secs = {n: s.get("seconds") for n, s in r["planners"].items()}
            out[r["id"]] = {
                "best_run": r["best_planner"],
                "a3_alone": a,
                "won_by": won,
                "run_seconds": secs,
            }
    return out


def finishers(inst: Instance, w: Any) -> dict[str, Any]:
    return {
        "plain": lambda state: share.finish_from(state, inst, w),
        "mrcm_greedy": sc.StrongFinish(inst, lookahead=0),
        "mrcm_look3": sc.StrongFinish(inst, lookahead=3),
    }


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    w = gc.use_goal("fluid")
    inst = Instance.from_json(task["instance"])
    stask = {
        "set": task["set"],
        "id": task["id"],
        "cell": "",
        "instance": task["instance"],
        "optimum": None,
        "planner": task["best_run"],
        "mrcm": None,
        "greedy": None,
    }
    sopts = {
        "agent": None,
        "seconds": opts["seconds"],
        "hints": "planner",
        "reward_scale": 0.01,
        "keep_plan": True,
        "first_plan": True,
    }
    rec = sc.run_task(stask, "uniform", 256, sopts, "switch")
    out = {k: v for k, v in task.items() if k != "instance"}
    out.update(
        search_cost=rec["search_cost"],
        leaves=rec["leaves"],
        finish_mode=rec["finish_mode"],
        plan_steps=None if rec["plan"] is None else len(rec["plan"]),
        prefix_steps=None,
        finishes={},
    )
    if rec["plan"] is None:
        return out
    prefix = rec["plan"][: len(rec["plan"]) - rec["planner_tail"]]
    out["prefix_steps"] = len(prefix)
    for name, finish in finishers(inst, w).items():
        t0 = sc.CLOCK()
        cost = cc.finish_cost(inst, prefix, finish)
        out["finishes"][name] = {"cost": cost, "seconds": round(sc.CLOCK() - t0, 3)}
    return out


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


def summarise(recs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| problem | best run (by) | search | tree steps / plan | plain | greedy MRCM | "
        "look3 MRCM (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    reach = 0
    for r in sorted(recs, key=lambda r: r["id"]):
        f = r["finishes"]

        def c(n: str, f: dict[str, Any] = f) -> str:
            return "-" if f.get(n, {}).get("cost") is None else f"{f[n]['cost']:g}"

        look = f.get("mrcm_look3", {}).get("cost")
        reach += look is not None and look <= r["best_run"] + 1e-9
        secs = f.get("mrcm_look3", {}).get("seconds")
        lines.append(
            f"| {r['id']} | {r['best_run']:g} ({r['won_by']}) | {r['search_cost']} | "
            f"{r['prefix_steps']} / {r['plan_steps']} | {c('plain')} | {c('mrcm_greedy')} | "
            f"{c('mrcm_look3')} ({secs}) |"
        )
    need = 8 if len(recs) >= 10 else -(-4 * len(recs) // 5)
    verdict = (
        f"Reading (D-81): the search's prefix finished with lookahead 3 reaches the best run on "
        f"{reach} of {len(recs)} (bar {need}): "
        + (
            "the greedy finisher is the root cause"
            if reach >= need
            else "the search's own moves are the problem, not only the finisher"
        )
    )
    return [*lines, "", verdict]


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        recs = [json.loads(x) for x in Path(args["summary"]).read_text().splitlines()]
        print("\n".join(summarise(recs)))
        return
    ref = losers(Path(args.get("records", "outputs/bench/pilot_a3.jsonl")))
    only = set(args["ids"].split(",")) if "ids" in args else None
    todo = []
    for s in SETS:
        for i in BenchmarkSet.read(s, ROOT / "benchmarks").instances:
            if i.instance_id in ref and (only is None or i.instance_id in only):
                task = {
                    "set": s,
                    "id": i.instance_id,
                    "instance": i.to_json(),
                    **ref[i.instance_id],
                }
                todo.append((task, {"seconds": float(args.get("seconds", 900))}))
    out = Path(args.get("out", "outputs/bench/finisher.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problems", flush=True)
    recs = []
    with out.open("w") as f:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for i, r in enumerate(results, 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            recs.append(r)
            print(f"{i}/{len(todo)} {r['id']} {r['finishes']}", flush=True)
    print("\n".join(summarise(recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
