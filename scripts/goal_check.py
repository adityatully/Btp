"""Goal check (paper log D-57): under the literature's other objectives, does our planner, and the
search over it, fall short of the optimum?

E-51: where the planner's finish aims at the right goal, the search fixes the planner's misses
and the network adds nothing. Our planner was built for priced fluid (`geometric.yaml`); the
literature's other objective families are steps (mix operations first) and waste, each "this
first, then fluid" (`notes/lit_bench/objectives.md`). Goals:

- `fluid`: `costs/geometric.yaml`, today's goal; the optima on record are used, no new proofs;
- `steps`: `costs/steps_first.yaml`, one mix costs 65,536;
- `waste`: `costs/waste_first.yaml`, one wasted unit costs 65,536.

For `steps` and `waste` a plan's remaining cost must stay below that weight, else the weights
are not lexicographic: checked on every plan, raised if not (rule 9).

Per E2 small problem (`headroom_share.tasks({"e2"})`) and goal: the planners `share`,
`share_look3` and `share_orders`, every plan replayed and validated; exact search seeded with
the cheapest (`seconds`, `nodes`) where no optimum is on record (new goals only); and, where the
optimum is proved and the best planner is more than 2% above it, the flat-prior search with
planner finishes (`search_check.Search`, `sims`, `search_seconds`). Elsewhere the search could
gain at most 2%, so it is not run and the planner's gap stands.

Reading, fixed in advance (D-57), per goal on its proved problems: the search leaves room if,
after it, the share of problems more than 2% above the optimum is >= 1/4 or the mean gap is
>= 5%; fewer than 20 proved problems: inconclusive.

    uv run python scripts/goal_check.py workers=36 out=outputs/goals/records.jsonl
    uv run python scripts/goal_check.py summary=outputs/goals/records.jsonl

Arguments are key=value: `goals` (default fluid,steps,waste), `limit=N` problems, `seconds`
(exact search, default 600), `nodes` (default 1,000,000), `orders` (default 32), `sims`
(default 256), `search_seconds` (default 900), `workers`, `out`.
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
import yaml

from meda.core.actions import Action
from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.solvers.base import SolveBudget
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom  # noqa: E402
import headroom_share as hs  # noqa: E402
import search_check as sc  # noqa: E402
import teacher_study as ts  # noqa: E402

GOALS = {
    "fluid": "costs/geometric.yaml",
    "steps": "costs/steps_first.yaml",
    "waste": "costs/waste_first.yaml",
}
FIRST = {"steps": "mixes", "waste": "waste"}  # the term each new goal ranks first
# Fixed in advance (D-57), plan 2.0's thresholds.
SMALL_GAP, GAP_SHARE, MEAN_GAP, MIN_PROVED = 0.02, 0.25, 0.05, 20


def use_goal(goal: str) -> CostWeights:
    """Every script module's weights and the env's costs set to `goal`."""
    w = CostWeights.from_dict(yaml.safe_load((ROOT / "configs" / GOALS[goal]).read_text()))
    for module in (headroom, ts, hs, sc):
        module.WEIGHTS = w
    sc.COSTS = GOALS[goal]
    return w


def check_first(name: str, goal: str, w: CostWeights, stats: dict[str, Any]) -> None:
    """Raise if a plan's cost outside the first-ranked term reaches that term's weight."""
    if goal not in FIRST or stats.get("cost") is None:
        return
    big = w.w_mix if goal == "steps" else w.w_waste
    rest = Rational.parse(stats["exact"]) - big * stats[FIRST[goal]]
    if rest >= big:
        raise ValueError(
            f"{name}: under {goal} the cost outside {FIRST[goal]} is {rest}, at least the "
            f"weight {big}: the weights are not lexicographic here"
        )


def run_task(task: dict[str, Any], goal: str, opts: dict[str, Any]) -> dict[str, Any]:
    use_goal("fluid")  # problems regenerate (and optima on record hold) under today's weights
    inst, head, known = hs.load_task(task)
    out: dict[str, Any] = {**head, "goal": goal}
    out.pop("bound", None)
    if inst is None:
        return out
    w = use_goal(goal)
    if goal != "fluid":
        known = None
    producers = {
        "share": lambda: share.share_plan(inst, w),
        "share_look3": lambda: share.share_plan(inst, w, lookahead=3),
        "share_orders": lambda: hs.share_orders(inst, opts["orders"]),
    }
    plans: dict[str, tuple[Action, ...] | None] = {}
    for name, make in producers.items():
        t0 = time.perf_counter()
        plans[name] = make()
        out[name] = hs.plan_stats(inst, name, plans[name], time.perf_counter() - t0)
        check_first(f"{out['id']} {name}", goal, w, out[name])
    best_cost, best_name = min(
        (Rational.parse(out[p]["exact"]), p) for p in plans if out[p]["cost"] is not None
    )
    if known is not None and best_cost < known:
        raise RuntimeError(f"{out['id']}: {best_name} costs {best_cost} < the optimum {known}")
    out["planner"] = float(best_cost)
    optimum = known
    if known is None and goal != "fluid":
        seed = plans[best_name]
        assert seed is not None
        t0 = time.perf_counter()
        plan, cost, stats = hs.SeededSearch(seed, best_cost).search(
            inst, SolveBudget(max_seconds=opts["seconds"], max_nodes=opts["nodes"])
        )
        assert plan is not None and cost is not None
        out["exact"] = {
            "status": "timeout" if stats.stopped else "optimal",
            "cost": float(cost),
            "nodes": stats.nodes_expanded,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        check_first(f"{out['id']} exact", goal, w, hs.plan_stats(inst, "exact", plan, 0.0))
        if not stats.stopped:
            optimum = cost
    out["optimum"] = None if optimum is None else float(optimum)
    if optimum is not None and float(best_cost) > (1 + SMALL_GAP) * float(optimum):
        task_ = {
            "set": goal,
            "id": out["id"],
            "cell": out["cell"],
            "instance": inst.to_json(),
            "optimum": float(optimum),
            "planner": float(best_cost),
            "mrcm": None,
            "greedy": None,
        }
        sopts = {
            "agent": None,
            "seconds": opts["search_seconds"],
            "hints": "planner",
            "reward_scale": 0.01,
        }
        rec = sc.run_task(task_, "uniform", opts["sims"], sopts)
        stats_ = {k[len("search_") :]: v for k, v in rec.items() if k.startswith("search_")}
        check_first(f"{out['id']} search", goal, w, stats_)
        out["search"] = {**stats_, "capped": rec["capped"], "planner_tail": rec["planner_tail"]}
    return out


def _star(args: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


# --------------------------------------------------------------------------- summary


def after(rec: dict[str, Any]) -> float:
    """The cheapest known plan after the search (the planner's where it was not run)."""
    found = rec.get("search", {}).get("cost")
    return min(rec["planner"], found) if found is not None else rec["planner"]


def goal_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
    proved = [r for r in rs if r.get("optimum") is not None]
    plan_gap = [r["planner"] / r["optimum"] - 1 for r in proved]
    gap = [after(r) / r["optimum"] - 1 for r in proved]
    return {
        "n": len(rs),
        "proved": len(proved),
        "searched": sum(1 for r in proved if "search" in r),
        "planner_mean": float(np.mean(plan_gap)) if plan_gap else None,
        "planner_over": float(np.mean([g > SMALL_GAP for g in plan_gap])) if plan_gap else None,
        "mean": float(np.mean(gap)) if gap else None,
        "over": float(np.mean([g > SMALL_GAP for g in gap])) if gap else None,
    }


def reading(st: dict[str, Any]) -> str:
    if st["proved"] < MIN_PROVED:
        return f"inconclusive ({st['proved']} proved, fewer than {MIN_PROVED})"
    if st["over"] >= GAP_SHARE or st["mean"] >= MEAN_GAP:
        return "room left after the search: step 2 (a network trained on this goal)"
    return "no room: the search over the planner reaches the optimum here"


def _p(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if "planner" in r:
            by[r["goal"]].append(r)
    lines = [
        "Gap = cost / optimum - 1 on proved problems; over 2% = share of them more than 2% above.",
        "",
        "| goal | problems | proved | searched | planner gap | planner over 2% "
        "| after search gap | after search over 2% |",
        "|---" * 8 + "|",
    ]
    stats = {g: goal_stats(by[g]) for g in GOALS if g in by}
    for g, st in stats.items():
        lines.append(
            f"| {g} | {st['n']} | {st['proved']} | {st['searched']} | {_p(st['planner_mean'])} "
            f"| {_p(st['planner_over'])} | {_p(st['mean'])} | {_p(st['over'])} |"
        )
    lines.append("")
    lines += [f"Reading on {g} (fixed in advance, D-57): {reading(st)}" for g, st in stats.items()]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    goals = args.get("goals", ",".join(GOALS)).split(",")
    opts = {
        "seconds": float(args.get("seconds", 600)),
        "nodes": int(args.get("nodes", 1_000_000)),
        "orders": int(args.get("orders", 32)),
        "sims": int(args.get("sims", 256)),
        "search_seconds": float(args.get("search_seconds", 900)),
    }
    limit = int(args["limit"]) if "limit" in args else None
    todo = [(t, g, opts) for t in hs.tasks({"e2"}, limit) for g in goals]
    out = Path(args.get("out", "outputs/goals/records.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} problem x goal pairs, {workers} workers", flush=True)
    t0 = time.perf_counter()
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for done, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {len(todo)} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
