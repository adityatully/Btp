"""Paper benchmark B1 (`notes/benchmark_plan.md`, D-67 to D-69): every producer on the paper
test sets, under the three goals the literature uses.

Sets: `test_s1_v1` ... `test_s5_v1`, `test_heldout_v1` (D-68; fresh, never tuned on).
Goals (`goal_check.GOALS`): `fluid` (`costs/geometric`, ours so far), `steps` (mixes first),
`waste` (waste first).

Per problem and goal, in this order (the answer key is fixed before our method runs, rule 3):
1. **Answer key**, from producers that are not ours: the root lower bound and the cheapest
   construction (naive or split tree) under the goal; exact search seeded with that
   construction (`exact_seconds`, `exact_nodes`), whose result counts only when it proves the
   optimum.
2. **Published methods** (`fluid` records only; a plan does not depend on the goal, its cost
   under every goal is stored in `by_goal`): DFM, FacDA, Generic, DFMM, MRCM, e-MRCM, NFSP.
   A method that declines a problem records why (`na`).
3. **Our planner alone:** `share`, `share_look3`, `share_orders`, `share_mrcm_look3` under the
   goal; the best planner is the cheapest.
4. **Our method** (`METHOD`, frozen as `method-v1`): the flat-prior search with greedy MRCM
   finishes (E-61), `sims` simulations per move, cap `seconds`. Its cost at any time t comes from
   its trace (the cheapest whole plan found by t; before its first plan, none). The paper
   reports t = 60 s and t = 900 s (Q-139).

Every plan is replayed by the engine and its DAG validated (rule 7); under `steps` and `waste`
every plan is checked to keep the goal's first term first (`goal_check.check_first`).

    OMP_NUM_THREADS=1 uv run python scripts/bench.py workers=30 out=outputs/bench/b1.jsonl
    uv run python scripts/bench.py summary=outputs/bench/b1.jsonl

Arguments are key=value: `sets` (default all six), `goals` (default fluid,steps,waste),
`limit=N` problems per set, `exact_seconds` (default 60), `exact_nodes` (default 1,000,000),
`nfsp_seconds` (per ILP solve, default 600, as Q-118), `orders` (default 32), `methods`
(published methods to run, default all), `workers`, `out`.
`seconds` and `sims` override the method's settings for light tests only.

B2 (D-70), the papers' own experiments: `sets=litbench_paper_v1` under all three goals and
`sets=test_emrcm_v1 goals=fluid methods=dfm,facda,generic,dfmm,mrcm,emrcm` (e-MRCM's
protocol, its own table `emrcm_table`; NFSP's experiments are 1:1, so B3's).

Reading, fixed in advance (D-69), is in `reading()`; it is given per set (D-70).
"""

from __future__ import annotations

import json
import subprocess
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
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, replay
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.base import SolveBudget
from meda.solvers.baselines import nfsp as nfsp_module
from meda.solvers.baselines import share
from meda.solvers.baselines.paper_graph import NotApplicable, emit
from meda.solvers.exact_search import ExactSearch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import goal_check as gc  # noqa: E402
import headroom_share as hs  # noqa: E402
import lit_bench as lb  # noqa: E402
import lit_bench_full as lbf  # noqa: E402
import search_check as sc  # noqa: E402
from emrcm_table3 import PAPER as EMRCM_PAPER  # noqa: E402

SETS = ("test_s1_v1", "test_s2_v1", "test_s3_v1", "test_s4_v1", "test_s5_v1", "test_heldout_v1")
B2_SETS = ("litbench_paper_v1", "test_emrcm_v1")  # D-70: the papers' own experiments
GOALS = tuple(gc.GOALS)
METHOD = {"search": "uniform", "finisher": "share_mrcm", "sims": 256, "seconds": 900.0}  # v1
PUBLISHED = {**lb.LITERATURE, **lbf.METHODS}
PLANNERS = ("share", "share_look3", "share_orders", "share_mrcm_look3")
# Each published method's own goal (notes/lit_bench/objectives.md): it is compared with our
# method run under that goal in the headline; every goal is reported.
OWN_GOAL = {
    "dfm": "steps",
    "generic": "steps",
    "facda": "fluid",
    "mrcm": "fluid",
    "emrcm": "fluid",
    "nfsp": "fluid",
    "dfmm": "waste",
}
TIMES = (60.0, 900.0)  # Q-139: both are reported
# Fixed in advance (D-69).
BOOT, CI = 10_000, 0.95
LOSS_SHARE = 0.10
WEIGHTS_BY_GOAL = {
    g: CostWeights.from_dict(yaml.safe_load((ROOT / "configs" / p).read_text()))
    for g, p in gc.GOALS.items()
}


def stats(inst: Instance, name: str, plan: tuple[Action, ...] | None, secs: float) -> dict:
    """A validated plan's totals: cost under every goal, sample, reagent, buffer, mixes,
    waste and steps."""
    if plan is None:
        return {"cost": None, "seconds": round(secs, 3)}
    _, totals, deltas = replay(inst, plan)
    report = validate(MixingDAG.from_deltas(inst, deltas, totals, name), inst)
    if not report.ok:
        raise RuntimeError(f"{inst.instance_id}: {name} plan fails validation: {report}")
    units = {
        k: sum(totals.reagent_by_fluid[f.id] for f in inst.fluids if f.kind.name == k)
        for k in ("SAMPLE", "REAGENT", "BUFFER")
    }
    by_goal = {g: w.scalar(totals, inst.fluids) for g, w in WEIGHTS_BY_GOAL.items()}
    return {
        "by_goal": {g: float(c) for g, c in by_goal.items()},
        "by_goal_exact": {g: str(c) for g, c in by_goal.items()},
        "sample": units["SAMPLE"],
        "reagent": units["REAGENT"],
        "buffer": units["BUFFER"],
        "mixes": totals.mix_ops,
        "waste": totals.waste_volume,
        "steps": len(plan),
        "seconds": round(secs, 3),
    }


def _cost(st: dict[str, Any], goal: str) -> float | None:
    return st["by_goal"][goal] if st.get("by_goal") else None


def _check(rid: str, name: str, goal: str, st: dict[str, Any]) -> None:
    if st.get("by_goal"):
        gc.check_first(
            f"{rid} {name}",
            goal,
            WEIGHTS_BY_GOAL[goal],
            {"cost": st["by_goal"][goal], "exact": st["by_goal_exact"][goal], **st},
        )


def tasks(sets: list[str], goals: list[str], limit: int | None) -> list[dict[str, Any]]:
    return [
        {
            "set": s,
            "id": i.instance_id,
            "cell": i.metadata.get("cell") or i.metadata.get("stage", ""),
            "instance": i.to_json(),
            "goal": g,
        }
        for s in sets
        for i in BenchmarkSet.read(s).instances[:limit]
        for g in goals
    ]


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    goal = task["goal"]
    w = gc.use_goal(goal)
    nfsp_module.TIME_LIMIT = opts["nfsp_seconds"]
    inst = Instance.from_json(task["instance"])
    rid = inst.instance_id
    out: dict[str, Any] = {k: v for k, v in task.items() if k != "instance"}
    out.update(
        T=len(inst.targets),
        F=inst.num_fluids,
        M=inst.chip.max_mix_volume,
        S=inst.chip.storage_capacity,
        deliveries=sum(t.demand for t in inst.targets),
        commit=opts["commit"],
        method=METHOD,
    )

    # 1. Answer key: nothing of ours.
    root = ExactSearch(w).solve(inst, SolveBudget(max_nodes=0))
    if root.cost is None or root.lower_bound is None:
        raise RuntimeError(f"{rid}: no construction plan under {goal}")
    out["bound"], out["construction"] = float(root.lower_bound), float(root.cost)
    optimum: Rational | None = root.cost if root.cost == root.lower_bound else None
    if optimum is None and opts["exact_seconds"] > 0:  # seeded with the construction only
        t0 = time.perf_counter()
        plan, cost, st = ExactSearch(w).search(
            inst, SolveBudget(max_seconds=opts["exact_seconds"], max_nodes=opts["exact_nodes"])
        )
        out["exact"] = {
            "status": "timeout" if st.stopped else "optimal",
            "nodes": st.nodes_expanded,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        if not st.stopped and cost is not None:
            optimum = cost
            ex = stats(inst, "exact", plan, 0.0)
            _check(rid, "exact", goal, ex)
    out["optimum"] = None if optimum is None else float(optimum)

    # 2. Published methods, once (their plans do not depend on the goal).
    if goal == "fluid":
        out["published"] = {}
        for name, make in PUBLISHED.items():
            if name not in opts.get("methods", PUBLISHED):
                continue
            t0 = time.perf_counter()
            try:
                st_ = stats(inst, name, emit(inst, make(inst)), time.perf_counter() - t0)
            except NotApplicable as e:
                st_ = {
                    "cost": None,
                    "na": str(e)[:200],
                    "seconds": round(time.perf_counter() - t0, 3),
                }
            except (IllegalActionError, ValueError, AssertionError) as e:  # recorded, never hidden
                st_ = {"cost": None, "error": f"{type(e).__name__}: {e}"[:200]}
            out["published"][name] = st_

    # 3. Our planner alone.
    makers = {
        "share": lambda: share.share_plan(inst, w),
        "share_look3": lambda: share.share_plan(inst, w, lookahead=3),
        "share_orders": lambda: hs.share_orders(inst, opts["orders"]),
        "share_mrcm_look3": lambda: share.share_plan(inst, w, lookahead=3, mrcm=True),
    }
    out["planners"] = {}
    for name, make in makers.items():
        t0 = time.perf_counter()
        out["planners"][name] = stats(inst, name, make(), time.perf_counter() - t0)
        _check(rid, name, goal, out["planners"][name])
    found = [(_cost(s, goal), n) for n, s in out["planners"].items() if _cost(s, goal) is not None]
    out["best_planner"] = min(found)[0] if found else None
    out["planner_seconds"] = sum(s["seconds"] for s in out["planners"].values())

    # 4. Our method.
    task_ = {
        "set": "B1",
        "id": rid,
        "cell": out["cell"],
        "instance": task["instance"],
        "optimum": out["optimum"],
        "planner": out["best_planner"] or out["construction"],
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
    rec = sc.run_task(task_, METHOD["search"], opts["sims"], sopts, METHOD["finisher"])
    plan = rec["plan"]
    ours = stats(inst, "method", plan, rec["search_seconds"]) if plan else {"cost": None}
    _check(rid, "method", goal, ours)
    ours.update(trace=rec["trace"], capped=rec["capped"], leaves=rec["leaves"])
    out["ours"] = ours
    for name, st_ in [("method", ours), *out["planners"].items()]:
        c = _cost(st_, goal)
        if c is not None and optimum is not None and c < float(optimum) - 1e-9:
            raise RuntimeError(f"{rid}: {name} costs {c} < the optimum {optimum} under {goal}")
        if c is not None and c < out["bound"] - 1e-9:
            raise RuntimeError(f"{rid}: {name} costs {c} < the lower bound under {goal}")
    return out


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


# --------------------------------------------------------------------------- summary


def ours_at(rec: dict[str, Any], t: float) -> float | None:
    """Our method's cost if stopped at t seconds (None: no whole plan yet)."""
    found = [c for s, c in rec["ours"].get("trace", []) if s <= t]
    return min(found) if found else None


def boot_ci(ratios: list[float], seed: int = 0) -> tuple[float, float]:
    x = np.array(ratios)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - CI) / 2, (1 + CI) / 2])
    return float(lo), float(hi)


def paired(a: list[float], b: list[float]) -> dict[str, Any]:
    """a against b per problem: wins / ties / losses for a, mean and median ratio a/b, CI."""
    r = [x / y for x, y in zip(a, b, strict=True)]
    lo, hi = boot_ci(r)
    return {
        "n": len(r),
        "wins": sum(x < 1 - 1e-12 for x in r),
        "losses": sum(x > 1 + 1e-12 for x in r),
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "ci": (lo, hi),
    }


def _fmt(p: dict[str, Any]) -> str:
    return (
        f"{p['n']} | {p['wins']} / {p['n'] - p['wins'] - p['losses']} / {p['losses']} | "
        f"{p['mean']:.3f} [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] | {p['median']:.3f}"
    )


def verdict(p: dict[str, Any]) -> str:
    """D-69: beats if the CI of the mean ratio is below 1 and losses <= 10%; loses if the CI
    is above 1; otherwise level."""
    if p["ci"][1] < 1 and p["losses"] <= LOSS_SHARE * p["n"]:
        return "beats"
    if p["ci"][0] > 1:
        return "loses"
    return "level"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    fluid = {(r["set"], r["id"]): r for r in recs if r["goal"] == "fluid"}
    lines = [
        f"Records: `{path}`, {len(recs)} (problem, goal) pairs. Ratios are per problem; "
        "[95% bootstrap CI of the mean]; w / t / l = wins / ties / losses for our method.",
    ]
    for t in TIMES:
        lines += ["", f"## Our method at {t:g} s", ""]
        lines += [
            "| set | goal | n | no plan | ours / bound | ours / optimum (n) | ours vs planner: "
            "n | w / t / l | mean [CI] | median |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            groups[(r["set"], r["goal"])].append(r)
        for key in sorted(groups):
            rs = groups[key]
            have = [r for r in rs if ours_at(r, t) is not None]
            bound = np.mean([ours_at(r, t) / r["bound"] for r in have]) if have else float("nan")
            opt = [ours_at(r, t) / r["optimum"] for r in have if r["optimum"]]
            vs = [r for r in have if r["best_planner"] is not None]
            p = paired([ours_at(r, t) for r in vs], [r["best_planner"] for r in vs])
            lines.append(
                f"| {key[0]} | {key[1]} | {len(rs)} | {len(rs) - len(have)} | {bound:.3f} | "
                + (f"{np.mean(opt):.3f} ({len(opt)})" if opt else "-")
                + f" | {_fmt(p)} |"
            )
    lines += ["", "## Against each published method (problems it accepts)", ""]
    lines += [
        "| method | goal | own goal | n | w / t / l | mean ours/method [CI] | median | "
        "reading at 60 s | reading at 900 s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in PUBLISHED:
        for goal in GOALS:
            rs = [
                r
                for r in recs
                if r["goal"] == goal
                and _cost(fluid[(r["set"], r["id"])]["published"].get(m, {}), goal) is not None
            ]
            if not rs:
                continue
            theirs = [_cost(fluid[(r["set"], r["id"])]["published"].get(m, {}), goal) for r in rs]
            cells = []
            for t in TIMES:
                mine = [ours_at(r, t) for r in rs]
                ok = [i for i, c in enumerate(mine) if c is not None]
                p = paired([mine[i] for i in ok], [theirs[i] for i in ok])
                cells.append((p, verdict(p)))
            p900 = cells[1][0]
            own = "yes" if OWN_GOAL[m] == goal else ""
            lines.append(f"| {m} | {goal} | {own} | {_fmt(p900)} | {cells[0][1]} | {cells[1][1]} |")
    lines += ["", "## Coverage: problems each published method accepts", ""]
    lines += ["| set | " + " | ".join(PUBLISHED) + " |", "|---" * (len(PUBLISHED) + 1) + "|"]
    for s in sorted({k[0] for k in fluid}):
        rs = [r for k, r in fluid.items() if k[0] == s]
        if rs:
            lines.append(
                f"| {s} ({len(rs)}) | "
                + " | ".join(
                    str(sum(r["published"].get(m, {}).get("by_goal") is not None for r in rs))
                    for m in PUBLISHED
                )
                + " |"
            )
    if any(r["set"] == "test_emrcm_v1" for r in recs):
        lines += ["", *emrcm_table(recs)]
    lines += ["", "Reading (fixed in advance, D-69; per set, D-70):"]
    for s in sorted({r["set"] for r in recs}):
        lines.append(f"- {s}: {reading([r for r in recs if r['set'] == s])}")
    return "\n".join(lines)


def emrcm_table(recs: list[dict[str, Any]]) -> list[str]:
    """B2 (D-70): e-MRCM's protocol per (N, T) under the fluid goal; e-MRCM's printed mixes and
    waste (its Table III, many targets) beside ours and e-MRCM (our reading), not graded (L-7)."""
    rows = [
        "## e-MRCM's protocol (`test_emrcm_v1`, fluid goal, our method at 900 s)",
        "",
        "| N-T | n | e-MRCM accepts | ours / e-MRCM (mean) | mixes: ours / e-MRCM / paper "
        "| waste: ours / e-MRCM / paper | ours / bound |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r["set"] == "test_emrcm_v1" and r["goal"] == "fluid":
            groups[r["cell"]].append(r)
    for cell in sorted(groups):
        rs = [r for r in groups[cell] if ours_at(r, TIMES[1]) is not None]
        acc = [r for r in rs if r["published"]["emrcm"].get("by_goal")]
        n, t = (int(x[1:]) for x in cell.split("-"))
        paper = EMRCM_PAPER.get((n, t))

        def mean(xs: list[float]) -> str:
            return f"{np.mean(xs):.2f}" if xs else "-"

        ratio = mean([ours_at(r, TIMES[1]) / _cost(r["published"]["emrcm"], "fluid") for r in acc])
        rows.append(
            f"| {cell} | {len(groups[cell])} | {len(acc)} | {ratio} | "
            f"{mean([r['ours']['mixes'] for r in rs])} / "
            f"{mean([r['published']['emrcm']['mixes'] for r in acc])} / "
            f"{paper[4] if paper else '-'} | "
            f"{mean([r['ours']['waste'] for r in rs])} / "
            f"{mean([r['published']['emrcm']['waste'] for r in acc])} / "
            f"{paper[5] if paper else '-'} | "
            f"{mean([ours_at(r, TIMES[1]) / r['bound'] for r in rs])} |"
        )
    return rows


def reading(recs: list[dict[str, Any]]) -> str:
    """D-69, per published method under its own goal at 900 s (Q-139: 60 s is reported too)."""
    fluid = {(r["set"], r["id"]): r for r in recs if r["goal"] == "fluid"}
    out = []
    for m, goal in OWN_GOAL.items():
        rs = [
            r
            for r in recs
            if r["goal"] == goal
            and _cost(fluid[(r["set"], r["id"])]["published"].get(m, {}), goal) is not None
            and ours_at(r, TIMES[1]) is not None
        ]
        if len(rs) < 10:
            out.append(f"{m} {len(rs)} problems (too few)")
            continue
        p = paired(
            [ours_at(r, TIMES[1]) for r in rs],
            [_cost(fluid[(r["set"], r["id"])]["published"].get(m, {}), goal) for r in rs],
        )
        out.append(f"{m} ({goal}) {verdict(p)} on {p['n']}")
    return "; ".join(out)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    opts = {
        "exact_seconds": float(args.get("exact_seconds", 60)),
        "exact_nodes": int(args.get("exact_nodes", 1_000_000)),
        "nfsp_seconds": float(args.get("nfsp_seconds", 600)),
        "orders": int(args.get("orders", 32)),
        "seconds": float(args.get("seconds", METHOD["seconds"])),
        "sims": int(args.get("sims", METHOD["sims"])),
        "commit": commit,
        "methods": args.get("methods", ",".join(PUBLISHED)).split(","),
    }
    limit = int(args["limit"]) if "limit" in args else None
    todo = [
        (t, opts)
        for t in tasks(
            args.get("sets", ",".join(SETS)).split(","),
            args.get("goals", ",".join(GOALS)).split(","),
            limit,
        )
    ]
    out = Path(args.get("out", "outputs/bench/b1.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 1))
    print(f"{len(todo)} (problem, goal) pairs, {workers} workers, commit {commit}", flush=True)
    t0 = time.perf_counter()
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for done, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"{done}/{len(todo)} {rec['id']} {rec['goal']}  {time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    print(f"wrote {len(todo)} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
