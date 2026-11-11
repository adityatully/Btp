"""D-80: do the novel intermediates of E-78 cause the search's gains? Two checks per problem.

**(a) Outside one target's recipes?** Each target is planned alone on an empty chip (the
sharing pass, plain and with MRCM's trees). A novel intermediate that none of those
single-target plans makes is not a step of any one target's own recipe: it is made for
several targets at once, or by a route the planner's recipes do not take.

**(b) Ablation.** The search's plan is replayed up to just before the step that makes a novel
intermediate and finished from there by the search's own finisher (`before`), and replayed
through that step and finished the same way (`after`). If `after` is clearly cheaper than
`before`, that one move is worth the difference: given the same finisher, the plan is cheaper
because the droplet exists. The move worth the most per problem is reported, with the share
of the search's gain over the reference (best of the 4 runs and its root finish) it explains.

Problems: E-78's records (`novel=`), those where the search beat the reference with a novel
intermediate, up to `per_set` per set (first by id), and only where the 4 runs took under
`planner_cap` seconds in total (their plans are recomputed to tell what is novel). The search
is method-v3's (`switch` finisher, first plan, CPU budget `seconds`).

Reading (D-80, fixed before the run): the novel intermediates cause the gain if, on at least
half the problems, one novel move alone explains at least half the search's gain over the
reference (ablation), and most novel intermediates lie outside every single-target recipe.

    OMP_NUM_THREADS=1 uv run python scripts/causal_check.py workers=3 out=outputs/novel/causal.jsonl
    uv run python scripts/causal_check.py summary=outputs/novel/causal.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.actions import Action
from meda.core.composition import Composition
from meda.core.problem import Instance
from meda.core.rules import replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import share

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import goal_check as gc  # noqa: E402
import headroom_share as hs  # noqa: E402
import novel_check as nc  # noqa: E402
import search_check as sc  # noqa: E402

FINISHER = "switch"  # method-v3 (D-79)


def single_target_comps(inst: Instance, w: Any) -> set[Composition]:
    """Every intermediate any target's own plan makes, each target alone on an empty chip."""
    out: set[Composition] = set()
    for t in inst.targets:
        one = replace(inst, targets=(replace(t, id=0),))
        for mrcm in (False, True):
            plan = share.share_plan(one, w, mrcm=mrcm)
            if plan is not None:
                out |= {m["comp"] for m in nc.intermediates(one, plan)}
    return out


def finish_cost(inst: Instance, prefix: tuple[Action, ...], finish: Any) -> float | None:
    """The cost of `prefix` plus the finisher's completion from where it leaves the chip, as
    the search values a leaf (the completion on a relaxed step limit, its leftovers free)."""
    state, totals, _ = replay(inst, prefix)
    found = finish(state)
    if found is None:
        return None
    return float(sc.WEIGHTS.scalar(totals, inst.fluids) + found[1])


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    w = gc.use_goal("fluid")
    inst = Instance.from_json(task["instance"])
    makers = {
        "share": lambda: share.share_plan(inst, w),
        "share_look3": lambda: share.share_plan(inst, w, lookahead=3),
        "share_orders": lambda: hs.share_orders(inst, opts["orders"]),
        "share_mrcm_look3": lambda: share.share_plan(inst, w, lookahead=3, mrcm=True),
    }
    plans = {n: p for n, p in ((n, m()) for n, m in makers.items()) if p is not None}
    costs = {n: hs.plan_stats(inst, n, p, 0.0)["cost"] for n, p in plans.items()}
    stask = {
        "set": task["set"],
        "id": task["id"],
        "cell": task["cell"],
        "instance": task["instance"],
        "optimum": None,
        "planner": min(costs.values()),
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
    rec = sc.run_task(stask, "uniform", 256, sopts, FINISHER)
    out: dict[str, Any] = {k: v for k, v in task.items() if k != "instance"}
    plan, root = rec["plan"], rec["root_finish"]
    ref = min([*costs.values(), *([float(root[1])] if root is not None else [])])
    out.update(best_run=min(costs.values()), reference=ref, search_cost=rec["search_cost"])
    if plan is None:
        return out | {"moves": None}
    refs = [nc.intermediates(inst, p) for p in plans.values()]
    if root is not None:
        refs.append(nc.intermediates(inst, root[0]))
    found = nc.novel(nc.intermediates(inst, plan), refs)
    alone = single_target_comps(inst, w)
    # the search's finisher, as it was when the search ended (A3: MRCM unless it switched)
    finish = (
        sc.StrongFinish(inst, lookahead=0)
        if rec["finish_mode"] == "share_mrcm"
        else (lambda state: share.finish_from(state, inst, w))
    )
    moves = []
    for m in found:
        before = finish_cost(inst, plan[: m["step"]], finish)
        after = finish_cost(inst, plan[: m["step"] + 1], finish)
        moves.append(
            {
                "comp": nc.show(m["comp"]),
                "step": m["step"],
                "uses": m["uses"],
                "single_target": m["comp"] in alone,
                "before": before,
                "after": after,
            }
        )
    return out | {"moves": moves, "finish_mode": rec["finish_mode"]}


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


def pick(novel_path: Path, per_set: int, planner_cap: float) -> list[dict[str, Any]]:
    """E-78's problems where the search beat the reference with a novel intermediate."""
    chosen: dict[str, list[str]] = defaultdict(list)
    for line in novel_path.read_text().splitlines():
        r = json.loads(line)
        if not r.get("novel") or r.get("reference") is None:
            continue
        if r["search_cost"] >= r["reference"] - 1e-9:
            continue
        if sum(r["planner_seconds"].values()) > planner_cap:
            continue
        chosen[r["set"]].append(r["id"])
    out = []
    for s, ids in chosen.items():
        keep = set(sorted(ids)[:per_set])
        for i in BenchmarkSet.read(s, ROOT / "benchmarks").instances:
            if i.instance_id in keep:
                cell = i.metadata.get("cell") or i.metadata.get("stage", "")
                out.append({"set": s, "id": i.instance_id, "cell": cell, "instance": i.to_json()})
    return out


def explained(r: dict[str, Any]) -> float | None:
    """The share of the search's gain over the reference that its best single novel move
    explains (ablation): (before - after) / (reference - search), at most 1."""
    gain = r["reference"] - r["search_cost"]
    worth = [
        m["before"] - m["after"] for m in r["moves"] or [] if None not in (m["before"], m["after"])
    ]
    if gain <= 1e-9 or not worth:
        return None
    return min(1.0, max(worth) / gain)


def together(r: dict[str, Any]) -> float | None:
    """The share of the gain from the step before the first novel move onwards: (before of the
    first novel move - search) / (reference - search), at most 1."""
    gain = r["reference"] - r["search_cost"]
    ms = [m for m in r["moves"] or [] if m["before"] is not None]
    if gain <= 1e-9 or not ms:
        return None
    first = min(ms, key=lambda m: m["step"])
    return min(1.0, max(0.0, (first["before"] - r["search_cost"]) / gain))


def summarise(recs: list[dict[str, Any]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r.get("moves"):
            groups[r["set"]].append(r)
            groups["all"].append(r)
    lines = [
        "| set | problems | novel moves | outside single-target recipes | "
        "one move explains >= half the gain | median share: best move / all novel moves |",
        "|---|---|---|---|---|---|",
    ]
    verdict = ""
    for name in [*sorted(g for g in groups if g != "all"), "all"]:
        rs = groups[name]
        moves = [m for r in rs for m in r["moves"]]
        outside = sum(not m["single_target"] for m in moves)
        ex = [e for e in (explained(r) for r in rs) if e is not None]
        half = sum(e >= 0.5 for e in ex)
        tg = [e for e in (together(r) for r in rs) if e is not None]
        med = f"{np.median(ex):.2f}" if ex else "-"
        med += f" / {np.median(tg):.2f}" if tg else " / -"
        lines.append(
            f"| {name} | {len(rs)} | {len(moves)} | {outside} | {half} of {len(ex)} | {med} |"
        )
        if name == "all" and moves:
            holds = len(ex) > 0 and half >= len(ex) / 2 and outside > len(moves) / 2
            verdict = (
                f"Reading (D-80): one novel move explains at least half the gain on {half} of "
                f"{len(ex)} problems; {outside} of {len(moves)} novel intermediates lie outside "
                "every single-target recipe: "
                + (
                    "the novel intermediates cause the gains"
                    if holds
                    else "the novel intermediates do not alone explain the gains"
                )
            )
    return [*lines, "", verdict]


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        recs = [json.loads(x) for x in Path(args["summary"]).read_text().splitlines()]
        print("\n".join(summarise(recs)))
        return
    opts = {"seconds": float(args.get("seconds", 300)), "orders": int(args.get("orders", 32))}
    todo = [
        (t, opts)
        for t in pick(
            Path(args.get("novel", "outputs/novel/novel.jsonl")),
            int(args.get("per_set", 10)),
            float(args.get("planner_cap", 120)),
        )
    ]
    out = Path(args.get("out", "outputs/novel/causal.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    recs = []
    print(f"{len(todo)} problems", flush=True)
    with out.open("w") as f:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for i, r in enumerate(results, 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            recs.append(r)
            print(f"{i}/{len(todo)} {r['id']} moves {len(r['moves'] or [])}", flush=True)
    print("\n".join(summarise(recs)))


if __name__ == "__main__":
    main(sys.argv[1:])
