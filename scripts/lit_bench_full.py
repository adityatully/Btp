"""Literature benchmark at full size (plan 6.3, paper log L-8, Q-118): the published
multi-target methods NFSP and e-MRCM against our planner and search on stages 3-5.

Problems: all of `val_s3_v2`, `val_s4_v2`, `val_s5_v2`; set C (the 99 where the best planner is
more than 2% above the lower bound, D-41) is marked and reported on its own. Producers, every
plan replayed by the engine and its DAG validated (rule 7):
- `nfsp` (`meda.solvers.baselines.nfsp`), each ILP solve limited to `seconds` (Q-118: 600);
  a solve over the limit is reported as `na` with that reason, never an approximate plan;
- `emrcm` (`meda.solvers.baselines.emrcm`, "our reading", L-7);
- the best planner (cheapest of share, share_look3, share_orders), from
  `notes/headroom/records.jsonl`;
- with `search=`, E-36's search (`net` at 256 simulations) on set C.

    OMP_NUM_THREADS=1 uv run python scripts/lit_bench_full.py seconds=600 workers=16 \
        out=notes/lit_bench/runs/full/records.jsonl
    uv run python scripts/lit_bench_full.py summary=notes/lit_bench/runs/full/records.jsonl \
        search=outputs/expert/round2/search_c.jsonl

Arguments are key=value: `seconds` (per ILP solve, default 600), `workers` (default 1),
`limit=N` problems per stage (light tests), `out`, `search` (E-36's records, optional).
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

from meda.core.problem import Instance
from meda.core.rules import IllegalActionError
from meda.generators.benchmark_io import BenchmarkSet
from meda.solvers.baselines import nfsp as nfsp_module
from meda.solvers.baselines.emrcm import emrcm
from meda.solvers.baselines.paper_graph import NotApplicable, PaperGraph, emit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import headroom_share as hs  # noqa: E402
import search_check as sc  # noqa: E402

WEIGHTS = hs.WEIGHTS
SETS = ("val_s3_v2", "val_s4_v2", "val_s5_v2")
METHODS: dict[str, Callable[[Instance], PaperGraph]] = {
    "nfsp": functools.partial(nfsp_module.nfsp, weights=WEIGHTS),
    "emrcm": functools.partial(emrcm, weights=WEIGHTS),
}


def tasks(limit: int | None) -> list[dict[str, Any]]:
    recs = {
        (r["source"], r["id"]): r
        for r in map(json.loads, (ROOT / "notes/headroom/records.jsonl").open())
        if r.get("source") in SETS
    }
    out = []
    for name in SETS:
        for inst in BenchmarkSet.read(name).instances[:limit]:
            rec = recs[(name, inst.instance_id)]
            planner = sc.best_planner(rec)
            out.append(
                {
                    "source": name,
                    "id": inst.instance_id,
                    "instance": inst.to_json(),
                    "planner": planner,
                    "bound": rec["bound"],
                    "set_c": planner > (1 + sc.MISS) * rec["bound"],
                }
            )
    return out


def run_task(task: dict[str, Any], seconds: float) -> dict[str, Any]:
    nfsp_module.TIME_LIMIT = seconds
    inst = Instance.from_json(task["instance"])
    out = {k: v for k, v in task.items() if k != "instance"}
    out.update(
        F=inst.num_fluids,
        M=inst.chip.max_mix_volume,
        T=len(inst.targets),
        deliveries=sum(t.demand for t in inst.targets),
    )
    for name, method in METHODS.items():
        t0 = time.perf_counter()
        try:
            plan = emit(inst, method(inst))
        except NotApplicable as e:
            secs = round(time.perf_counter() - t0, 1)
            out[name] = {"cost": None, "na": str(e)[:200], "seconds": secs}
            continue
        except (IllegalActionError, ValueError, AssertionError) as e:  # recorded, never hidden
            out[name] = {"cost": None, "error": f"{type(e).__name__}: {e}"[:200]}
            continue
        out[name] = hs.plan_stats(inst, name, plan, time.perf_counter() - t0)
        if out[name]["cost"] < task["bound"] - 1e-9:
            raise RuntimeError(f"{task['id']}: {name} is below the lower bound {task['bound']}")
    return out


def _star(args: tuple[dict[str, Any], float]) -> dict[str, Any]:
    return run_task(*args)


# --------------------------------------------------------------------------- summary


def _reason(v: dict[str, Any]) -> str:
    """The `na` reason with its numbers left out, so reasons group."""
    text = v.get("na") or v.get("error") or "?"
    if "not solved within" in text:
        return "ILP over the time limit"
    if "1/2^d grid" in text:
        return "target off the 1/2^d grid"
    if "prime factor above M" in text:
        return "denominator not M-smooth"
    return text.split(":")[0][:60]


def rows(rs: list[dict[str, Any]], search: dict[str, float]) -> list[str]:
    out = []
    for name in (*METHODS, "search"):
        if name == "search":
            ok = [r for r in rs if r["id"] in search]
            cost = {r["id"]: search[r["id"]] for r in ok}
        else:
            ok = [r for r in rs if r[name].get("cost") is not None]
            cost = {r["id"]: r[name]["cost"] for r in ok}
        if not ok:
            continue
        ratio = [cost[r["id"]] / r["planner"] for r in ok]
        cheaper = sum(x < 1 - 1e-12 for x in ratio)
        dearer = sum(x > 1 + 1e-12 for x in ratio)
        both = [r for r in ok if r["id"] in search and name != "search"]
        vs_search = (
            f"{np.mean([cost[r['id']] / search[r['id']] for r in both]):.3f} ({len(both)})"
            if both
            else "-"
        )
        out.append(
            f"| {name} | {len(ok)}/{len(rs)} | {np.mean(ratio):.3f} | {np.median(ratio):.3f} "
            f"| {cheaper} / {len(ok) - cheaper - dearer} / {dearer} | {vs_search} |"
        )
    return out


def summarise(path: Path, search_path: Path | None) -> str:
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    search: dict[str, float] = {}
    if search_path is not None:
        for x in search_path.read_text().splitlines():
            r = json.loads(x)
            if r["kind"] == "net" and r["sims"] == 256 and r["search_cost"] is not None:
                search[r["id"]] = r["search_cost"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        stage = r["source"].split("_")[1]
        groups[stage].append(r)
        if r["set_c"]:
            groups[f"C{stage[1]}"].append(r)
            groups["C"].append(r)
    lines = [
        f"Records: `{path}`. Ratio = cost / best planner (cheapest of share, share_look3, "
        "share_orders), costs 64/8/1, mix 1, waste 1. `< / = / >`: cheaper / equal / dearer than "
        "the best planner. `vs search`: mean cost / E-36 search cost on the problems both have."
    ]
    for key in sorted(groups):
        rs = groups[key]
        lines += [
            "",
            f"### {key} ({len(rs)} problems)",
            "",
            "| producer | accepted | ratio mean | ratio median | < / = / > | vs search (n) |",
            "|---" * 6 + "|",
            *(rows(rs, search) or ["| none | - | - | - | - | - |"]),
        ]
        for name in METHODS:
            why: dict[str, int] = defaultdict(int)
            for r in rs:
                if r[name].get("cost") is None:
                    why[_reason(r[name])] += 1
            if why:
                lines.append(
                    f"\n{name} not applicable: "
                    + "; ".join(f"{k} {n}" for k, n in sorted(why.items(), key=lambda x: -x[1]))
                )
    timed = [r["nfsp"]["seconds"] for r in recs if "seconds" in r["nfsp"]]
    if timed:
        lines.append(
            f"\nNFSP seconds per problem: median {np.median(timed):.1f}, max {max(timed):.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    args = dict(a.split("=", 1) for a in argv)
    search = Path(args["search"]) if "search" in args else None
    if "summary" in args:
        path = Path(args["summary"])
        text = summarise(path, search)
        print(text)
        (path.parent / "summary.md").write_text(text + "\n")
        return
    seconds = float(args.get("seconds", 600))
    workers = int(args.get("workers", 1))
    limit = int(args["limit"]) if "limit" in args else None
    out = Path(args.get("out", "notes/lit_bench/runs/full/records.jsonl"))
    todo = [(t, seconds) for t in tasks(limit)]
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"{len(todo)} problems, {workers} workers, {seconds:g} s per ILP solve", flush=True)
    with out.open("w") as fh:
        results = map(_star, todo) if workers == 1 else Pool(workers).imap_unordered(_star, todo)
        for done, rec in enumerate(results, 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{done}/{len(todo)} {rec['id']}  {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"wrote {len(todo)} records to {out}\n")
    main([f"summary={out}", *([f"search={search}"] if search else [])])


if __name__ == "__main__":
    main(sys.argv[1:])
