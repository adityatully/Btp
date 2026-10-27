"""Literature mini benchmark (plan 2.4, paper log D-36): published single-target methods against
the sharing pass and the agent on stages 1 and 2.

Producers per problem, every plan replayed by the engine and its DAG validated (rule 7):
- the published methods `dfm`, `facda`, `generic` (dilution) and `dfmm`, `mrcm` (mixtures),
  each from its paper (`notes/lit_bench/selection.md`); a method that does not take a problem
  records why (`na`), never an approximate plan;
- our planner: `share`, `share_look3`, `share_orders` (as `scripts/headroom_share.py`); the
  best planner is the cheapest of the three;
- with `agent=<agent.pt>`, the greedy agent, labelled by `label=` (default: the run folder
  name), so a later agent gets its own results next to this one.

Sets (`sets=`, each reported on its own): `val_s1_v2`, `val_s2_v2`, `e2` (the E2 problems of
stages s1/s2 with a proved optimum, from Track 2 or the headroom run's search, E-28,
regenerated as in `headroom_share.py`), `litbench_s1_v1`,
`litbench_s2_v1` (fresh draws) and `litbench_paper_v1` (the papers' own problems).

    OMP_NUM_THREADS=1 uv run python scripts/lit_bench.py \
        agent=outputs/warm/mixed_s0/checkpoints/agent.pt label=e22 workers=36
    uv run python scripts/lit_bench.py summary=notes/lit_bench/runs/e22/records.jsonl

`limit=N` keeps the first N problems of each set (light tests). On the Mac keep `workers=1`.
Without `agent=`, `reuse_agent=notes/headroom/records.jsonl` copies the agent results stored
there (E-22, greedy) for the problems it has (the val and E2 sets).
"""

from __future__ import annotations

import functools
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from meda.core.composition import Rational
from meda.core.costs import CostWeights
from meda.core.problem import Instance
from meda.core.rules import IllegalActionError
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import share
from meda.solvers.baselines.dfmm import dfmm
from meda.solvers.baselines.facda import dfm, facda
from meda.solvers.baselines.generic import generic
from meda.solvers.baselines.mrcm import mrcm
from meda.solvers.baselines.paper_graph import NotApplicable, PaperGraph, emit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom_share as hs  # noqa: E402

WEIGHTS: CostWeights = hs.WEIGHTS
DEFAULT_SETS = "val_s1_v2,val_s2_v2,e2,litbench_s1_v1,litbench_s2_v1,litbench_paper_v1"
LITERATURE: dict[str, Callable[[Instance], PaperGraph]] = {
    "dfm": dfm,
    "facda": facda,
    "generic": generic,
    "dfmm": dfmm,
    "mrcm": functools.partial(mrcm, weights=WEIGHTS),
}
PLANNERS = ("share", "share_look3", "share_orders")


def tasks(sets: list[str], limit: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sets:
        if name == "e2":
            found = [t for t in hs.tasks({"e2"}, None) if t["stage"] in ("s1", "s2")]
            found = [t for t in found if _e2_id(t) in E2_OPTIMA]
        else:
            bench = BenchmarkSet.read(name)
            found = []
            for inst in bench.instances:
                ref = bench.reference[inst.instance_id]
                lb = ref.lower_bound if ref.lower_bound is not None else ref.cost
                found.append(
                    {
                        "source": name,
                        "instance": inst.to_json(),
                        "stage": inst.metadata.get("stage") or name.split("_")[1],
                        "cell": inst.metadata.get("cell") or inst.metadata.get("source", ""),
                        "ref_kind": "optimal" if ref.status == "optimal" else "lower_bound",
                        "ref": str(lb),
                    }
                )
        out += found[:limit]
    return out


def _e2_id(task: dict[str, Any]) -> str:
    return f"{task['rec']['cell']}-{task['rec']['index']}"


def _e2_optima() -> dict[str, Rational]:
    """E2 optima proved in Track 2 or by the headroom run's exact search (E-28)."""
    out = {}
    for line in (ROOT / "notes/headroom/records.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["source"] == "e2" and rec.get("optimum") is not None:
            cost = rec["optimum"]
            if cost != int(cost):
                raise ValueError(f"{rec['id']}: optimum {cost} is not an integer cost")
            out[rec["id"]] = Rational.of(int(cost))
    return out


E2_OPTIMA = _e2_optima()


def literature_stats(inst: Instance, name: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        graph = LITERATURE[name](inst)
        plan = emit(inst, graph)
    except NotApplicable as e:
        return {"cost": None, "na": str(e)[:200], "seconds": round(time.perf_counter() - t0, 3)}
    except (IllegalActionError, ValueError, AssertionError) as e:  # recorded, never hidden
        return {"cost": None, "error": f"{type(e).__name__}: {e}"[:200]}
    out = hs.plan_stats(inst, name, plan, time.perf_counter() - t0)
    counts = graph.counts(inst.targets[0].droplet_volume)
    out["paper"] = {"nm": counts["nm"], "drawn": list(counts["drawn"]), "nw": counts["nw"]}  # type: ignore[call-overload]
    return out


def run_task(task: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    inst, out, known = hs.load_task(task)
    if inst is None:
        return out
    if out["source"] == "e2":
        known = E2_OPTIMA[out["id"]]
    out.pop("bound")
    t = inst.targets[0]
    out.update(
        F=inst.num_fluids,
        M=inst.chip.max_mix_volume,
        target=str(t.composition),
        tolerance=str(t.tolerance),
        volume=t.droplet_volume,
        target_sample=hs.target_sample(inst),
        optimum=None if known is None else float(known),
    )
    if inst.metadata.get("source"):
        out["paper_source"] = inst.metadata["source"]
    makers = {
        "share": lambda: share.share_plan(inst, WEIGHTS),
        "share_look3": lambda: share.share_plan(inst, WEIGHTS, lookahead=3),
        "share_orders": lambda: hs.share_orders(inst, opts["orders"]),
    }
    for name, make in makers.items():
        t0 = time.perf_counter()
        out[name] = hs.plan_stats(inst, name, make(), time.perf_counter() - t0)
    for name in LITERATURE:
        out[name] = literature_stats(inst, name)
    if opts.get("agent"):
        out["agent"] = hs.agent_cost(inst, opts["agent"])
    elif (out["source"], out["id"]) in opts["reused"]:
        out["agent"] = opts["reused"][(out["source"], out["id"])]
    best = min(Rational.parse(out[p]["exact"]) for p in PLANNERS if hs._has_cost(out, p))
    out["best_planner"] = float(best)
    for name in (*LITERATURE, *PLANNERS, "agent"):
        if (
            hs._has_cost(out, name)
            and known is not None
            and Rational.parse(out[name]["exact"]) < known
        ):
            raise RuntimeError(f"{out['id']}: {name} is below the proved optimum {known}")
    return out


def _star(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    return run_task(*args)


# --------------------------------------------------------------------------- summary


def _groups(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        key = r["source"] if r["source"] != "litbench_paper_v1" else f"paper_{r['stage']}"
        if r["source"] == "e2":
            key = f"e2_{r['stage']}"
        out[key].append(r)
    return out


def _mean(xs: list[float]) -> str:
    return f"{np.mean(xs):.3f}" if xs else "-"


def _m(xs: list[float], digits: int = 1) -> str:
    return f"{np.mean(xs):.{digits}f}" if xs else "-"


def method_rows(rs: list[dict[str, Any]], has_agent: bool) -> list[str]:
    rows = []
    for name in (*LITERATURE, *PLANNERS, *(("agent",) if has_agent else ())):
        ok = [r for r in rs if hs._has_cost(r, name)]
        if not ok:
            reasons = {r[name].get("na", r[name].get("error", "?"))[:40] for r in rs if name in r}
            if name in LITERATURE:
                rows.append(f"| {name} | 0/{len(rs)} | {'; '.join(sorted(reasons))[:80]} |"
                            + " - |" * 10)  # fmt: skip
            continue
        ratio = [r[name]["cost"] / r["best_planner"] for r in ok]
        agent_ok = [r for r in ok if hs._has_cost(r, "agent")]
        agent_ratio = [r["agent"]["cost"] / r["best_planner"] for r in agent_ok]
        proved = [r for r in ok if r["optimum"] is not None]
        opt_ratio = [r[name]["cost"] / r["optimum"] for r in proved]
        cheaper = sum(1 for x in ratio if x < 1 - 1e-12)
        dearer = sum(1 for x in ratio if x > 1 + 1e-12)
        paper = [r[name]["paper"] for r in ok if "paper" in r[name]]
        if name == "agent":
            sample = [r[name].get("sample") for r in ok]
            own = "-"
        else:
            sample = [r[name]["sample"] for r in ok]
            own = (
                f"nm {_m([p['nm'] for p in paper])}, units {_m([sum(p['drawn']) for p in paper])}, "
                f"nw {_m([p['nw'] for p in paper])}"
                if paper
                else "-"
            )
        rows.append(
            f"| {name} | {len(ok)}/{len(rs)} | {_mean(ratio)} | {np.median(ratio):.3f} "
            f"| {cheaper} / {len(ok) - cheaper - dearer} / {dearer} "
            f"| {_mean(agent_ratio) if name not in ('agent',) else '-'} "
            f"| {_mean(opt_ratio)} ({len(proved)}) "
            f"| {_m([s for s in sample if s is not None])} "
            f"| {_m([r[name].get('waste') for r in ok if r[name].get('waste') is not None])} "
            f"| {_m([r[name].get('mixes') for r in ok if r[name].get('mixes') is not None])} "
            f"| {own} |"
        )
    return rows


def summarise(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    recs = [r for r in recs if "skipped" not in r]
    has_agent = any("agent" in r for r in recs)
    lines = [
        f"Records: `{path}`. Ratio = cost / best planner (cheapest of share, share_look3, "
        "share_orders), costs 64/8/1, mix 1, waste 1. `vs best: < / = / >` counts problems "
        "where the producer is cheaper / equal / dearer than the best planner. `agent on same` is "
        "the agent's mean ratio on the problems that producer accepted (agent failures left out). "
        "Units, waste and mixes are the engine's (2-input mixes); `own metric` is the paper's "
        "(n-input mixes nm, units drawn, waste nw).",
    ]
    for key, rs in sorted(_groups(recs).items()):
        ran = [r for r in rs if isinstance(r.get("agent"), dict)]
        failed = sum(1 for r in ran if not hs._has_cost(r, "agent"))
        lines += [
            "",
            f"### {key} ({len(rs)} problems"
            + (f", agent ran on {len(ran)}, failed {failed}" if ran else ", agent not run")
            + ")",
            "",
            "| producer | accepted | ratio mean | ratio median | vs best: < / = / > "
            "| agent on same | ratio to optimum (n) | sample units | waste | mixes | own metric |",
            "|---" * 11 + "|",
            *method_rows(rs, has_agent),
        ]
    return "\n".join(lines)


def per_problem(path: Path) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines() if "skipped" not in x]
    names = (*PLANNERS, *LITERATURE, "agent")
    head = "| set | id | M | F | vol | target | tol | opt | " + " | ".join(names) + " |"
    lines = [head, "|---" * (8 + len(names)) + "|"]
    for r in recs:
        cells = []
        for n in names:
            v = r.get(n)
            if not isinstance(v, dict):
                cells.append("")
            elif v.get("cost") is not None:
                cells.append(f"{v['cost']:g}")
            else:
                cells.append("na" if "na" in v else ("fail" if "failed" in v else "err"))
        lines.append(
            f"| {r['source']} | {r['id']} | {r['M']} | {r['F']} | {r['volume']} | {r['target']} "
            f"| {r['tolerance']} | {r['optimum'] if r['optimum'] is not None else ''} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    if "summary" in args:
        path = Path(args["summary"])
        print(summarise(path))
        (path.parent / "per_problem.md").write_text(per_problem(path) + "\n")
        (path.parent / "summary.md").write_text(summarise(path) + "\n")
        return
    agent = args.get("agent")
    label = args.get("label") or (Path(agent).parents[1].name if agent else "no_agent")
    reused: dict[tuple[str, str], Any] = {}
    if not agent and "reuse_agent" in args:
        for line in Path(args["reuse_agent"]).read_text().splitlines():
            rec = json.loads(line)
            if isinstance(rec.get("agent"), dict):
                reused[(rec["source"], rec["id"])] = rec["agent"]
        label = args.get("label", "e22_reused")
    opts = {"agent": agent, "orders": int(args.get("orders", 32)), "reused": reused}
    sets = args.get("sets", DEFAULT_SETS).split(",")
    limit = int(args["limit"]) if "limit" in args else None
    out = Path(args.get("out", f"notes/lit_bench/runs/{label}/records.jsonl"))
    workers = int(args.get("workers", 1))
    todo = [(t, opts) for t in tasks(sets, limit)]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.perf_counter(), 0
    print(f"{len(todo)} problems from {sets}, {workers} workers, agent {agent}", flush=True)
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {done} records to {out}\n")
    main([f"summary={out}"])


if __name__ == "__main__":
    main(sys.argv[1:])
