"""Q-127 / D-76: does the search build intermediate droplets the planner never makes, and do
later steps reuse them? (the "unexpected moves").

Per problem (fluid goal): the 4 planner runs and method-v2's search (flat prior, A3's
`switch` finisher, D-75) are run and their plans kept. Each plan is replayed; every droplet a
`Mix` makes and no `Output` delivers is an **intermediate**. An intermediate of the search's
plan is **novel** if its exact composition is made by no `Mix` in any of the 4 planner plans
nor in the search's own finisher run alone from the empty chip (else the finisher, not the
search, made it), and **reused** if a later `Mix` takes from it. Per problem the record keeps
the costs, the counts, and each novel reused intermediate (composition, volume, the step that
made it, how many later mixes took from it).

Reading (D-76, fixed before the run): the "unexpected moves" story holds if, on the problems
where the search's plan is cheaper than both the best planner run and its finisher alone, at
least half hold a novel reused intermediate, and that share is at least 20 points above the
share on the problems where the search ties that reference. Otherwise the search's gains come
from ordinary choices made better.

Arguments are key=value: `sets` (default the val sets s1-s5 v2 and `val_emrcm_v1`), `limit`
per set, `seconds` (default 900), `sims` (256), `orders` (32), `workers`, `out`.
Summary only: `summary=<records.jsonl>`.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.actions import Action, Mix, Output
from meda.core.composition import Composition
from meda.core.problem import Instance
from meda.core.rules import apply
from meda.core.state import PrepState
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import goal_check as gc  # noqa: E402
import headroom_share as hs  # noqa: E402
import search_check as sc  # noqa: E402

SETS = ("val_s1_v2", "val_s2_v2", "val_s3_v2", "val_s4_v2", "val_s5_v2", "val_emrcm_v1")
FINISHER = "switch"  # method-v2 (D-75, E-74)


def intermediates(inst: Instance, plan: tuple[Action, ...]) -> list[dict[str, Any]]:
    """Every droplet a `Mix` makes in `plan` and no `Output` delivers, in order made: its
    composition, volume, the step that made it, and how many later mixes take from it."""
    state = PrepState.initial(inst)
    made: dict[int, dict[str, Any]] = {}
    delivered: set[int] = set()
    for step, act in enumerate(plan):
        if isinstance(act, Mix):
            for src in (act.a, act.b):
                if not src.is_reservoir and src.index in made:
                    made[src.index]["uses"] += 1
        elif isinstance(act, Output) and not act.src.is_reservoir:
            delivered.add(act.src.index)
        before = {d.uid for d in state.droplets}
        state, _, _ = apply(state, act, inst)
        if isinstance(act, Mix):
            new = [d for d in state.droplets if d.uid not in before]
            if len(new) != 1:  # a mix makes exactly one droplet (rule 9: never silently)
                raise RuntimeError(f"{inst.instance_id}: step {step} made {len(new)} droplets")
            d = new[0]
            made[d.uid] = {"comp": d.composition, "volume": d.volume, "step": step, "uses": 0}
    return [m for uid, m in made.items() if uid not in delivered]


def _show(c: Composition) -> str:
    return "(" + ", ".join(str(n) for n in c.num) + f")/{c.den}"


def novel(search: list[dict[str, Any]], planners: list[list[dict[str, Any]]]) -> list[dict]:
    """The search's reused intermediates whose composition no planner plan makes."""
    seen = {m["comp"] for plan in planners for m in plan}
    return [m for m in search if m["uses"] > 0 and m["comp"] not in seen]


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    w = gc.use_goal("fluid")
    inst = Instance.from_json(task["instance"])
    makers = {
        "share": lambda: share.share_plan(inst, w),
        "share_look3": lambda: share.share_plan(inst, w, lookahead=3),
        "share_orders": lambda: hs.share_orders(inst, opts["orders"]),
        "share_mrcm_look3": lambda: share.share_plan(inst, w, lookahead=3, mrcm=True),
    }
    plans, costs, secs = {}, {}, {}
    # The search's own finisher run alone from the empty chip: not a planner run, so its
    # intermediates are not the search's doing either (they would be the finisher's)
    t0 = sc.CLOCK()
    alone = sc.StrongFinish(inst, lookahead=0)(PrepState.initial(inst))
    secs["finisher_alone"] = round(sc.CLOCK() - t0, 3)
    finisher_plan = alone[0] if alone is not None else None
    finisher_cost = float(alone[1]) if alone is not None else None
    for name, make in makers.items():
        t0 = sc.CLOCK()
        plan = make()
        secs[name] = round(sc.CLOCK() - t0, 3)
        if plan is not None:
            plans[name] = plan
            costs[name] = hs.plan_stats(inst, name, plan, 0.0)["cost"]
    best = min(costs.values()) if costs else None
    stask = {
        "set": task["set"],
        "id": task["id"],
        "cell": task["cell"],
        "instance": task["instance"],
        "optimum": None,
        "planner": best,
        "mrcm": None,
        "greedy": None,
    }
    sopts = {
        "agent": None,
        "seconds": opts["seconds"],
        "hints": "planner",
        "reward_scale": 0.01,
        "keep_plan": True,
    }
    rec = sc.run_task(stask, "uniform", opts["sims"], sopts, FINISHER)
    out: dict[str, Any] = {k: v for k, v in task.items() if k != "instance"}
    out.update(
        T=len(inst.targets),
        F=inst.num_fluids,
        planner_costs=costs,
        planner_seconds=secs,
        best_planner=best,
        finisher_alone=finisher_cost,
        # the reference the search must beat: the best of the 4 runs and its finisher alone
        reference=min(c for c in (best, finisher_cost) if c is not None)
        if best is not None or finisher_cost is not None
        else None,
        search_cost=rec["search_cost"],
        leaves=rec["leaves"],
        finish_mode=rec["finish_mode"],
    )
    if rec["plan"] is None:
        return out | {"novel": None}
    mine = intermediates(inst, rec["plan"])
    theirs = [intermediates(inst, p) for p in plans.values()]
    ref = theirs + ([intermediates(inst, finisher_plan)] if finisher_plan is not None else [])
    found = novel(mine, ref)
    out.update(
        search_intermediates=len(mine),
        search_reused=sum(m["uses"] > 0 for m in mine),
        planner_intermediates={n: len(x) for n, x in zip(plans, theirs, strict=True)},
        novel=[
            {"comp": _show(m["comp"]), "volume": m["volume"], "step": m["step"], "uses": m["uses"]}
            for m in found
        ],
    )
    return out


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


def tasks(sets: list[str], limit: int | None) -> list[dict[str, Any]]:
    out = []
    for s in sets:
        for i in BenchmarkSet.read(s, ROOT / "benchmarks").instances[:limit]:
            out.append(
                {
                    "set": s,
                    "id": i.instance_id,
                    "cell": i.metadata.get("cell") or i.metadata.get("stage", ""),
                    "instance": i.to_json(),
                }
            )
    return out


def summarise(recs: list[dict[str, Any]]) -> list[str]:
    """D-76's reading: novel reused intermediates where the search beats the planner runs,
    against where they tie, per set and over all."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r.get("novel") is not None and r["reference"] is not None:
            groups[r["set"]].append(r)
            groups["all"].append(r)
    lines = [
        "| set | n | search cheaper than runs and finisher | with novel reused (one feeding 2+ "
        "mixes) | tie | "
        "with novel reused | "
        "mean gain where novel / where not |",
        "|---|---|---|---|---|---|---|",
    ]
    verdict = ""
    for name in [*sorted(g for g in groups if g != "all"), "all"]:
        rs = groups[name]
        better = [r for r in rs if r["search_cost"] < r["reference"] - 1e-9]
        tie = [r for r in rs if abs(r["search_cost"] - r["reference"]) <= 1e-9]
        nb = sum(bool(r["novel"]) for r in better)
        nt = sum(bool(r["novel"]) for r in tie)
        shared = sum(any(m["uses"] >= 2 for m in r["novel"]) for r in better)

        def gain(xs: list[dict[str, Any]]) -> str:
            return (
                f"{np.mean([1 - r['search_cost'] / r['reference'] for r in xs]):.3f}" if xs else "-"
            )

        g1 = gain([r for r in better if r["novel"]])
        g0 = gain([r for r in better if not r["novel"]])
        lines.append(
            f"| {name} | {len(rs)} | {len(better)} | {nb} ({shared}) | {len(tie)} | {nt} | "
            f"{g1} / {g0} |"
        )
        if name == "all" and better:
            share_b = nb / len(better)
            share_t = nt / len(tie) if tie else 0.0
            holds = share_b >= 0.5 and share_b > share_t + 0.2
            verdict = (
                f"Reading (D-76): {nb} of {len(better)} problems where the search is cheaper "
                f"({share_b:.0%}) hold a novel reused intermediate, against {nt} of {len(tie)} "
                f"ties ({share_t:.0%}): "
                + (
                    "the unexpected-moves story holds"
                    if holds
                    else "the gains come mostly from ordinary choices made better"
                )
            )
    lines.append("")
    lines.append(verdict)
    ex = sorted(
        (r for r in groups.get("all", []) if r["novel"]),
        key=lambda r: r["search_cost"] / r["reference"],
    )[:10]
    if ex:
        lines += ["", "Examples (largest gains with novel reused intermediates):"]
        for r in ex:
            parts = ", ".join(
                f"{m['comp']} x{m['volume']} used {m['uses']}" for m in r["novel"][:3]
            )
            lines.append(
                f"- {r['id']} ({r['cell']}): {r['search_cost']:g} vs planner runs "
                f"{r['best_planner']:g}, finisher alone {r['finisher_alone']:g}; {parts}"
            )
    return lines


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        recs = [json.loads(x) for x in Path(args["summary"]).read_text().splitlines()]
        print("\n".join(summarise(recs)))
        return
    opts = {
        "seconds": float(args.get("seconds", 900)),
        "sims": int(args.get("sims", 256)),
        "orders": int(args.get("orders", 32)),
    }
    limit = int(args["limit"]) if "limit" in args else None
    todo = [(t, opts) for t in tasks(args.get("sets", ",".join(SETS)).split(","), limit)]
    out = Path(args.get("out", "outputs/novel/novel.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    recs = []
    with out.open("w") as f:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for i, r in enumerate(results, 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            recs.append(r)
            print(f"{i}/{len(todo)} {r['id']} novel {len(r['novel'] or [])}", flush=True)
    print("\n".join(summarise(recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
