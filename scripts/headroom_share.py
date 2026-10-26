"""Headroom above the sharing pass (plan 2.0, paper log D-34): where can it be beaten, by how much.

The sharing pass solves every problem and is at the optimum on 51 of the 65 proved problems
(E-17). Learning or search is only worth building where it can be beaten. Per problem:

- **H2 producers:** `share` (the teacher), `share_look3` (rollout lookahead 3), `share_orders`
  (the pass over its 4 target orders plus `orders` random ones) and, with `agent=<agent.pt>`,
  the greedy agent. Every plan is replayed by the engine and its DAG validated.
- **H1 proofs:** on the sources in `exact`, problems without a proved optimum get exact search
  seeded with the cheapest plan above as its incumbent, so branch and bound prunes against a
  near-optimal cost instead of the constructions'. `optimal` proves the optimum (and the
  sharing pass's true gap); a cheaper plan found on a timeout is headroom outright; the
  timeout's lower bound tightens the interval.
- **H3:** where nothing proves or improves, the sharing pass's waste and the sample units it
  draws beyond what the targets contain, next to its ratio to the bound.

Sources: `e2` (all 400 small problems of `notes/track2/e2_headroom.jsonl`, 65 with an optimum
on record), `e3` (the 45 medium problems), `val` (`val_s1_v2` ... `heldout_combo_v2`).

    uv run python scripts/headroom_share.py which=e2,e3,val exact=e2,e3 seconds=600 \
        agent=outputs/warm/mixed_s0/checkpoints/agent.pt workers=30 \
        out=notes/headroom/records.jsonl
    uv run python scripts/headroom_share.py summary=notes/headroom/records.jsonl

Arguments are key=value. `limit=N` keeps the first N problems of each source (light tests);
`nodes` caps exact search's expansions per problem (memory). On the Mac keep `workers=1`.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.actions import Action
from meda.core.composition import Rational
from meda.core.dag import MixingDAG, validate
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError, replay
from meda.core.state import PrepState
from meda.solvers.base import SolveBudget
from meda.solvers.baselines import share
from meda.solvers.exact_search import ExactSearch, fluid_bound_heuristic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import teacher_study as ts  # noqa: E402

WEIGHTS = ts.WEIGHTS
# Fixed in advance (plan 2.0): a cell has headroom if the sharing pass's gap on its proved
# problems averages >= 5% or exceeds 2% on a quarter of them, or the best plan found lowers
# the median cost by >= 5%.
MEAN_GAP, GAP_SHARE, SMALL_GAP, MEDIAN_GAIN = 0.05, 0.25, 0.02, 0.05


class SeededSearch(ExactSearch):
    """Exact search whose incumbent is a given plan (the cheapest any producer found)."""

    def __init__(self, plan: tuple[Action, ...], cost: Rational) -> None:
        super().__init__(WEIGHTS)
        self.seed = (plan, cost)

    def incumbent(self, instance: Instance) -> tuple[tuple[Action, ...], Rational] | None:
        return self.seed


def share_orders(inst: Instance, extra: int) -> tuple[Action, ...] | None:
    """The sharing pass over its own target orders plus `extra` random ones (seeded by the
    instance id), the cheapest by replayed cost; never dearer than `share_plan`."""
    best: tuple[Rational, tuple[Action, ...]] | None = None
    base = share.share_plan(inst, WEIGHTS)
    if base is not None:
        best = (cost_of(inst, base), base)
    rng = np.random.default_rng([zlib.crc32(inst.instance_id.encode()), 0])
    tried = [tuple(o) for o in share.orders(inst, WEIGHTS)]
    n = len(inst.targets)
    for _ in range(extra if n > 1 else 0):
        order = tuple(int(i) for i in rng.permutation(n))
        if order in tried:
            continue
        tried.append(order)
        try:
            start = share._Builder(inst, PrepState.initial(inst), [], WEIGHTS)
            plan = tuple(share._run(start, list(order), 0).actions)
        except IllegalActionError:
            continue
        cost = cost_of(inst, plan)
        if best is None or cost < best[0]:
            best = (cost, plan)
    return None if best is None else best[1]


def cost_of(inst: Instance, plan: tuple[Action, ...]) -> Rational:
    _, totals, _ = replay(inst, plan)
    return WEIGHTS.scalar(totals, inst.fluids)


_AGENT: dict[str, Any] = {}


def agent_cost(inst: Instance, path: str) -> dict[str, Any]:
    """The greedy agent's episode: its exact cost on success, else how it ended."""
    import torch
    from omegaconf import OmegaConf

    from meda.agents.base import run_episode
    from meda.agents.ppo.policy import PPOAgent
    from meda.analysis.evaluate import build_env

    if path not in _AGENT:
        torch.set_num_threads(1)
        _AGENT[path] = PPOAgent.load(path)
    load = lambda rel: OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / rel))  # noqa: E731
    cfg = {"costs": load("costs/geometric.yaml"), "env": load("env/sample_prep.yaml")}
    t0 = time.perf_counter()
    try:
        res = run_episode(_AGENT[path], build_env(cfg, [inst]), inst, deterministic=True)
    except Exception as e:  # recorded, not hidden: the summary counts these
        return {"cost": None, "error": f"{type(e).__name__}: {e}"[:200]}
    secs = round(time.perf_counter() - t0, 3)
    if res.status != "feasible":
        return {"cost": None, "failed": res.status, "seconds": secs}
    assert res.dag is not None and res.cost is not None
    validate(res.dag, inst).raise_if_invalid()
    return {"cost": float(res.cost), "exact": str(res.cost), "seconds": secs}


def plan_stats(inst: Instance, name: str, plan: tuple[Action, ...] | None, secs: float) -> dict:
    if plan is None:
        return {"cost": None, "seconds": round(secs, 3)}
    _, totals, deltas = replay(inst, plan)
    report = validate(MixingDAG.from_deltas(inst, deltas, totals, name), inst)
    if not report.ok:
        raise RuntimeError(f"{inst.instance_id}: {name} plan fails validation: {report}")
    cost = WEIGHTS.scalar(totals, inst.fluids)
    sample = sum(totals.reagent_by_fluid[f.id] for f in inst.fluids if f.kind.name == "SAMPLE")
    return {
        "cost": float(cost),
        "exact": str(cost),
        "sample": sample,
        "mixes": totals.mix_ops,
        "waste": totals.waste_volume,
        "steps": len(plan),
        "seconds": round(secs, 3),
    }


def target_sample(inst: Instance) -> float:
    """Sample units the delivered droplets contain: what no plan can avoid drawing."""
    ids = [f.id for f in inst.fluids if f.kind.name == "SAMPLE"]
    return float(
        sum(
            (
                Rational.of(t.demand * t.droplet_volume * t.composition.num[i], t.composition.den)
                for t in inst.targets
                for i in ids
            ),
            Rational.of(0),
        )
    )


def tasks(which: set[str], limit: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "e2" in which:
        e2 = []
        for line in (ROOT / "notes/track2/e2_headroom.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if "skipped" not in rec:
                e2.append({"source": "e2", "rec": rec, "stage": ts.e2_stage(rec["cell"])})
        out += e2[:limit]
    for src in ("e3", "val"):
        if src in which:
            found = ts.tasks({src})
            if src == "val":
                by: dict[str, list] = defaultdict(list)
                for t in found:
                    by[t["source"]].append(t)
                found = [t for ts_ in by.values() for t in ts_[:limit]]
            else:
                found = found[:limit]
            out += found
    return out


def load_task(task: dict[str, Any]) -> tuple[Instance | None, dict[str, Any], Rational | None]:
    """(instance, record header, optimum on record or None)."""
    head: dict[str, Any] = {"source": task["source"], "stage": task["stage"]}
    known: Rational | None = None
    if task["source"] in ("e2", "e3"):
        rec = task["rec"]
        head["cell"], head["id"] = rec["cell"], f"{rec['cell']}-{rec['index']}"
        inst = ts._regenerate(rec, "small" if task["source"] == "e2" else "medium")
        if inst is None:
            head["skipped"] = "regenerated problem does not match the record"
            return None, head, None
        root = fluid_bound_heuristic(WEIGHTS)(PrepState.initial(inst), inst)
        found = rec["exact"] if isinstance(rec["exact"], list) else [rec["exact"]]
        if found[-1]["status"] == "optimal":
            known = Rational.parse(str(found[-1]["cost"]))
        lbs = [Rational.parse(str(e["lower_bound"])) for e in found if e.get("lower_bound")]
        head["bound"] = known if known is not None else max([root, *lbs])
    else:
        inst = Instance.from_json(task["instance"])
        head["cell"], head["id"] = task["cell"], inst.instance_id
        ref = Rational.parse(task["ref"])
        if task["ref_kind"] == "optimal":
            known = ref
        head["bound"] = ref
    return inst, head, known


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    inst, out, known = load_task(task)
    if inst is None:
        return out
    bound: Rational = out.pop("bound")
    out.update(
        deliveries=sum(t.demand for t in inst.targets),
        T=len(inst.targets),
        F=len(inst.fluids),
        M=inst.chip.max_mix_volume,
        S=inst.chip.storage_capacity,
        target_sample=target_sample(inst),
    )
    producers = {
        "share": lambda: share.share_plan(inst, WEIGHTS),
        "share_look3": lambda: share.share_plan(inst, WEIGHTS, lookahead=3),
        "share_orders": lambda: share_orders(inst, opts["orders"]),
    }
    plans: dict[str, tuple[Action, ...] | None] = {}
    for name, make in producers.items():
        t0 = time.perf_counter()
        plans[name] = make()
        out[name] = plan_stats(inst, name, plans[name], time.perf_counter() - t0)
    if opts.get("agent"):
        out["agent"] = agent_cost(inst, opts["agent"])

    found = [(Rational.parse(out[p]["exact"]), p) for p in out if _has_cost(out, p)]
    best_cost, best_name = min(found)
    if known is not None and best_cost < known:
        raise RuntimeError(
            f"{out['id']}: {best_name} costs {best_cost} < the recorded optimum {known}"
        )

    optimum, opt_source = known, "record" if known is not None else None
    if known is None and task["source"] in opts["exact"]:
        seed = plans[best_name] if best_name in plans else plans["share"]
        assert seed is not None
        search = SeededSearch(seed, cost_of(inst, seed))
        t0 = time.perf_counter()
        plan, cost, stats = search.search(
            inst, SolveBudget(max_seconds=opts["seconds"], max_nodes=opts["nodes"])
        )
        assert plan is not None and cost is not None
        out["exact"] = {
            "status": "timeout" if stats.stopped else "optimal",
            "cost": float(cost),
            "lower_bound": float(stats.lower_bound) if stats.lower_bound is not None else None,
            "nodes": stats.nodes_expanded,
            "seconds": round(time.perf_counter() - t0, 1),
            "improved": cost < best_cost,
        }
        if cost < best_cost:
            out["search"] = plan_stats(inst, "search", plan, 0.0)
            best_cost, best_name = cost, "search"
        if not stats.stopped:
            optimum, opt_source = cost, "search"
        elif stats.lower_bound is not None and stats.lower_bound > bound:
            bound = stats.lower_bound
    out["bound"] = float(optimum if optimum is not None else bound)
    out["optimum"] = None if optimum is None else float(optimum)
    out["opt_source"] = opt_source
    out["best"] = {"cost": float(best_cost), "by": best_name}
    return out


def _has_cost(rec: dict[str, Any], name: str) -> bool:
    v = rec.get(name)
    return isinstance(v, dict) and v.get("cost") is not None and "exact" in v


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


# --------------------------------------------------------------------------- summary


def cell_summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers of one row of the headroom table, and the verdict fixed in advance."""
    share_cost = [r["share"]["cost"] for r in rs]
    proved = [r for r in rs if r["optimum"] is not None]
    gaps = [r["share"]["cost"] / r["optimum"] - 1 for r in proved]
    gains = [(r["share"]["cost"] - r["best"]["cost"]) / r["share"]["cost"] for r in rs]
    improved = [(g, r) for g, r in zip(gains, rs, strict=True) if g > 0]
    saved = [
        r["share"]["sample"] - r[r["best"]["by"]].get("sample", r["share"]["sample"])
        for _, r in improved
        if "sample" in r.get(r["best"]["by"], {})
    ]
    rest = [r for g, r in zip(gains, rs, strict=True) if g == 0 and r["optimum"] is None]
    s = {
        "n": len(rs),
        "proved": len(proved),
        "share_at_opt": sum(1 for g in gaps if g == 0),
        "gap_mean": float(np.mean(gaps)) if gaps else None,
        "gap_median": float(np.median(gaps)) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "gap_over_small": sum(1 for g in gaps if g > SMALL_GAP),
        "improved": len(improved),
        "gain_median_all": float(np.median(gains)),
        "gain_median_improved": float(np.median([g for g, _ in improved])) if improved else None,
        "gain_max": max(gains),
        "sample_saved_median": float(np.median(saved)) if saved else None,
        "by": dict(Counter(r["best"]["by"] for _, r in improved)),
        "share_vs_bound": float(
            np.median([c / r["bound"] for c, r in zip(share_cost, rs, strict=True)])
        ),
        "uncertified": len(rest),
        "rest_waste": float(np.median([r["share"]["waste"] for r in rest])) if rest else None,
        "rest_extra_sample": (
            round(float(np.median([r["share"]["sample"] - r["target_sample"] for r in rest])), 1)
            if rest
            else None
        ),
        "agent_solved": sum(1 for r in rs if _has_cost(r, "agent")),
        "agent_errors": sum(
            1 for r in rs if isinstance(r.get("agent"), dict) and "error" in r["agent"]
        ),
    }
    s["headroom"] = bool(
        (gaps and (s["gap_mean"] >= MEAN_GAP or s["gap_over_small"] >= GAP_SHARE * len(gaps)))
        or s["gain_median_all"] >= MEDIAN_GAIN
    )
    return s


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    kept = [r for r in recs if "skipped" not in r]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in kept:
        groups[(r["source"], "all")].append(r)
        groups[(r["source"], r["cell"])].append(r)
    lines = [
        "Gaps and gains in %. gap = sharing pass over the optimum (proved problems); "
        "gain = (sharing pass - best plan found) / sharing pass.",
        "",
        "| set | cell | n | proved | share at opt | gap mean / med / max | gap > 2% | improved "
        "| gain med (all) / med (improved) / max | sample saved med | by | share / bound med "
        "| uncertified: waste, extra sample | agent solved | headroom |",
        "|---" * 15 + "|",
    ]
    for (src, cell), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "all", kv[0][1])
    ):
        s = cell_summary(rs)
        errors = f" ({s['agent_errors']} errors)" if s["agent_errors"] else ""
        lines.append(
            f"| {src} | {cell} | {s['n']} | {s['proved']} | {s['share_at_opt']} "
            f"| {_pct(s['gap_mean'])} / {_pct(s['gap_median'])} / {_pct(s['gap_max'])} "
            f"| {s['gap_over_small']} | {s['improved']} "
            f"| {_pct(s['gain_median_all'])} / {_pct(s['gain_median_improved'])} "
            f"/ {_pct(s['gain_max'])} "
            f"| {s['sample_saved_median'] if s['sample_saved_median'] is not None else '-'} "
            f"| {', '.join(f'{k} {v}' for k, v in sorted(s['by'].items())) or '-'} "
            f"| {s['share_vs_bound']:.2f} "
            f"| {s['uncertified']}: {s['rest_waste'] if s['rest_waste'] is not None else '-'}, "
            f"{s['rest_extra_sample'] if s['rest_extra_sample'] is not None else '-'} "
            f"| {s['agent_solved']}{errors} "
            f"| {'**yes**' if s['headroom'] else 'no'} |"
        )
    searched = [r["exact"] for r in kept if "exact" in r and isinstance(r["exact"], dict)]
    st = Counter(e["status"] for e in searched)
    lines += [
        "",
        f"H1: exact search on {len(searched)} problems: {st.get('optimal', 0)} proved, "
        f"{st.get('timeout', 0)} timed out, {sum(e['improved'] for e in searched)} found a "
        f"cheaper plan than every producer.",
        f"skipped (regenerated problem differs from the record): {len(recs) - len(kept)}",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        print(summarise(Path(args["summary"])))
        return
    which = set(args.get("which", "e2,e3,val").split(","))
    opts = {
        "exact": set(filter(None, args.get("exact", "e2,e3").split(","))),
        "seconds": float(args.get("seconds", 600)),
        "nodes": int(args.get("nodes", 1_000_000)),
        "orders": int(args.get("orders", 32)),
        "agent": args.get("agent"),
    }
    limit = int(args["limit"]) if "limit" in args else None
    out = Path(args.get("out", "notes/headroom/records.jsonl"))
    workers = int(args.get("workers", 1))
    todo = [(t, opts) for t in tasks(which, limit)]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.perf_counter(), 0
    print(f"{len(todo)} problems, {workers} workers, exact on {sorted(opts['exact'])}", flush=True)
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}\n")
    print(summarise(out))


if __name__ == "__main__":
    main(sys.argv[1:])
